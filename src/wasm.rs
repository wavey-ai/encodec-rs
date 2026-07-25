use crate::arithmetic::{ArithmeticDecoder, ArithmeticEncoder};
use crate::binary::{
    read_chunk_payload, read_ecdc_header, read_exactly, write_chunk, write_ecdc_header,
};
use crate::ecdc_presets::{
    bandwidth_preset_from_kbps, chunk_preset_from_ms, fixed_bundle_name, fixed_context_samples,
};
use crate::entropy::{
    probability_columns_from_flat_logits, ProbabilityParameters, ProbabilityScratch,
};
use crate::format::{
    ecdc_chunk_layout_for_chunk_count, ecdc_chunk_layout_from_metadata, ecdc_lm_frame_length,
    segment_frame_length, segment_starts, validate_metadata, EcdcChunkLayout, EcdcMetadata,
    ARITHMETIC_TOTAL_RANGE_BITS, DEFAULT_FP_SCALE, DEFAULT_MIN_RANGE,
    QUANTIZED_LM_BITSTREAM_VERSION,
};
use crate::metadata::OnnxFrameBundleMetadata;
use crate::quantized_lm::{QuantizedLm, QuantizedLmState, QuantizedLmWeights};
use crate::seam::{repair_cubic_hermite_seams_planar, FIXED_CONTEXT_SEAM_REPAIR_SAMPLES};
use crate::stable_hash::stable_hash_hex;
use serde::Serialize;
use std::io::Cursor;
use wasm_bindgen::prelude::*;

#[derive(Debug, Clone, serde::Serialize)]
#[serde(rename_all = "camelCase")]
struct LmEcdcChunk {
    offset: usize,
    samples: usize,
    frame_length: usize,
    payload: Vec<u8>,
}

#[derive(Debug, Clone, serde::Serialize)]
#[serde(rename_all = "camelCase")]
struct LmEcdcChunks {
    metadata: EcdcMetadata,
    chunks: Vec<LmEcdcChunk>,
}

#[wasm_bindgen(js_name = fixedEcdcBundleName)]
pub fn fixed_ecdc_bundle_name_js(
    bandwidth_kbps: JsValue,
    chunk_ms: JsValue,
) -> Result<String, JsValue> {
    let bandwidth_kbps = optional_f64(&bandwidth_kbps)?;
    let chunk_ms = optional_f64(&chunk_ms)?;
    let bandwidth = bandwidth_preset_from_kbps(bandwidth_kbps).map_err(to_js_error)?;
    let chunk = chunk_preset_from_ms(chunk_ms).map_err(to_js_error)?;
    Ok(fixed_bundle_name(bandwidth, chunk).to_owned())
}

fn optional_f64(value: &JsValue) -> Result<Option<f64>, JsValue> {
    if value.is_null() || value.is_undefined() {
        return Ok(None);
    }
    value
        .as_f64()
        .map(Some)
        .ok_or_else(|| to_js_error("expected a number, null, or undefined"))
}

#[wasm_bindgen(js_name = initPanicHook)]
pub fn init_panic_hook() {
    console_error_panic_hook::set_once();
}

#[wasm_bindgen(js_name = bundleMetadata)]
pub fn bundle_metadata(bundle_json: &str) -> Result<JsValue, JsValue> {
    let meta = parse_bundle(bundle_json)?;
    to_js_value(&meta)
}

#[wasm_bindgen(js_name = stableHashHex)]
pub fn stable_hash_hex_js(bytes: &[u8]) -> String {
    stable_hash_hex(bytes)
}

#[wasm_bindgen(js_name = ecdcMetadata)]
pub fn ecdc_metadata(payload: &[u8]) -> Result<JsValue, JsValue> {
    let metadata: EcdcMetadata =
        read_ecdc_header(&mut Cursor::new(payload)).map_err(to_js_error)?;
    to_js_value(&metadata)
}

#[wasm_bindgen(js_name = ecdcOverlapAdd)]
pub fn ecdc_overlap_add(
    bundle_json: &str,
    audio_length: usize,
    decoded_frames: &[f32],
) -> Result<Vec<f32>, JsValue> {
    let meta = parse_bundle(bundle_json)?;
    if let Some(context) =
        fixed_context_samples(meta.segment_samples, meta.segment_stride).map_err(to_js_error)?
    {
        return fixed_context_crop_concat(
            &meta,
            audio_length,
            decoded_frames,
            meta.segment_stride,
            context,
        )
        .map_err(to_js_error);
    }
    let layout = EcdcChunkLayout {
        samples: meta.segment_samples,
        stride: meta.segment_stride.max(1),
    };
    overlap_add_decoded_frames(&meta, audio_length, decoded_frames, layout).map_err(to_js_error)
}

#[wasm_bindgen(js_name = ecdcOverlapAddForMetadata)]
pub fn ecdc_overlap_add_for_metadata(
    bundle_json: &str,
    metadata_json: &str,
    decoded_frames: &[f32],
) -> Result<Vec<f32>, JsValue> {
    let meta = parse_bundle(bundle_json)?;
    let metadata: EcdcMetadata = serde_json::from_str(metadata_json).map_err(to_js_error)?;
    // Recognised fixed-context bundles never go through ordinary
    // triangle overlap-add: each decoded model window is
    // segment_samples long and only `decoded[context..context+owned_len]`
    // is logical audio. Legacy (non-fixed-context) bundles keep the
    // existing overlap-add path.
    if let Some(context) =
        fixed_context_samples(meta.segment_samples, meta.segment_stride).map_err(to_js_error)?
    {
        return fixed_context_crop_concat(
            &meta,
            metadata.audio_length,
            decoded_frames,
            meta.segment_stride,
            context,
        )
        .map_err(to_js_error);
    }
    let layout = ecdc_chunk_layout_from_metadata(&meta, &metadata).map_err(to_js_error)?;
    overlap_add_decoded_frames(&meta, metadata.audio_length, decoded_frames, layout)
        .map_err(to_js_error)
}

#[wasm_bindgen(js_name = lmEcdcHeaderForWeights)]
pub fn lm_ecdc_header_for_weights(
    bundle_json: &str,
    audio_length: usize,
    bitstream_version: u8,
    weights: &[u8],
) -> Result<Vec<u8>, JsValue> {
    lm_ecdc_header_for_weights_impl(bundle_json, audio_length, bitstream_version, weights, false)
}

#[wasm_bindgen(js_name = lmEcdcFixedHeaderForWeights)]
pub fn lm_ecdc_fixed_header_for_weights(
    bundle_json: &str,
    audio_length: usize,
    bitstream_version: u8,
    weights: &[u8],
) -> Result<Vec<u8>, JsValue> {
    lm_ecdc_header_for_weights_impl(bundle_json, audio_length, bitstream_version, weights, true)
}

fn lm_ecdc_header_for_weights_impl(
    bundle_json: &str,
    audio_length: usize,
    bitstream_version: u8,
    weights: &[u8],
    fixed_frame_length: bool,
) -> Result<Vec<u8>, JsValue> {
    if bitstream_version != QUANTIZED_LM_BITSTREAM_VERSION {
        return Err(to_js_error(format!(
            "only q8 acv={} is supported, got acv={bitstream_version}",
            QUANTIZED_LM_BITSTREAM_VERSION
        )));
    }
    let meta = parse_bundle(bundle_json)?;
    let mut metadata =
        EcdcMetadata::from_bundle(&meta, audio_length, None, Some(stable_hash_hex(weights)));
    if fixed_frame_length {
        // For recognised fixed-context bundles the model/code window
        // always spans the full bundle frame_length (e.g. 203), never a
        // proportion of the logical audio_length (which would otherwise
        // undercount it, e.g. 200).
        let frame_length = match fixed_context_samples(meta.segment_samples, meta.segment_stride)
            .map_err(to_js_error)?
        {
            Some(_context) => meta.frame_length,
            None => segment_frame_length(audio_length, meta.segment_samples, meta.frame_length),
        };
        metadata.lm_frame_length = Some(frame_length);
    }
    let mut out = Vec::new();
    write_ecdc_header(&mut out, &metadata).map_err(to_js_error)?;
    Ok(out)
}

#[wasm_bindgen(js_name = lmEcdcChunk)]
pub fn lm_ecdc_chunk(payload: &[u8]) -> Result<Vec<u8>, JsValue> {
    let mut out = Vec::new();
    write_chunk(&mut out, payload, true).map_err(to_js_error)?;
    Ok(out)
}

#[wasm_bindgen(js_name = lmEcdcChunkFromFrame)]
pub fn lm_ecdc_chunk_from_frame(
    bundle_json: &str,
    weights: &[u8],
    scale: f32,
    codes: &[u16],
    frame_length: usize,
) -> Result<Vec<u8>, JsValue> {
    let meta = parse_bundle(bundle_json)?;
    validate_lm_metadata(&meta).map_err(to_js_error)?;
    let expected_codes = meta
        .num_codebooks
        .checked_mul(meta.frame_length)
        .ok_or_else(|| to_js_error("ECDC frame code shape overflows usize"))?;
    if codes.len() != expected_codes {
        return Err(to_js_error(format!(
            "ECDC frame code length {} does not match {} codebooks * {} frames",
            codes.len(),
            meta.num_codebooks,
            meta.frame_length
        )));
    }

    validate_encoded_frame_length(&meta, frame_length).map_err(to_js_error)?;

    let mut encoder = QuantizedLmChunkEncoder::new(bundle_json, weights, scale)?;
    for step in 0..frame_length {
        let mut step_codes = Vec::with_capacity(meta.num_codebooks);
        for codebook in 0..meta.num_codebooks {
            step_codes.push(codes[codebook * meta.frame_length + step]);
        }
        encoder.push(&step_codes)?;
    }
    let payload = encoder.finish();
    lm_ecdc_chunk(&payload)
}

#[wasm_bindgen(js_name = lmEcdcDecodeChunks)]
pub fn lm_ecdc_decode_chunks(bundle_json: &str, payload: &[u8]) -> Result<JsValue, JsValue> {
    let meta = parse_bundle(bundle_json)?;
    let mut reader = Cursor::new(payload);
    let metadata: EcdcMetadata = read_ecdc_header(&mut reader).map_err(to_js_error)?;
    validate_metadata(&meta, &metadata).map_err(to_js_error)?;
    if !metadata.use_lm {
        return Err(to_js_error("ECDC payload does not use LM coding"));
    }

    let mut raw_chunks = Vec::new();
    while (reader.position() as usize) < payload.len() {
        raw_chunks.push(read_chunk_payload(&mut reader, true).map_err(to_js_error)?);
    }

    let layout = ecdc_chunk_layout_for_chunk_count(&meta, &metadata, raw_chunks.len())
        .map_err(to_js_error)?;
    let context =
        fixed_context_samples(meta.segment_samples, meta.segment_stride).map_err(to_js_error)?;
    let mut chunks = Vec::new();
    let starts = segment_starts(metadata.audio_length, layout.stride);
    if starts.len() != raw_chunks.len() {
        return Err(to_js_error(format!(
            "LM ECDC payload has {} chunks, but metadata implies {} chunks",
            raw_chunks.len(),
            starts.len()
        )));
    }
    for (offset, payload) in starts.into_iter().zip(raw_chunks) {
        // `samples` is the logical owned length of this chunk, bounded by
        // segment_stride; it must never include the private context
        // samples that make up the rest of a fixed model window.
        let samples = (metadata.audio_length - offset).min(layout.stride);
        let frame_length = if context.is_some() {
            metadata
                .lm_frame_length
                .filter(|value| *value > 0)
                .unwrap_or(meta.frame_length)
        } else {
            ecdc_lm_frame_length(&metadata, samples, meta.segment_samples, meta.frame_length)
        };
        chunks.push(LmEcdcChunk {
            offset,
            samples,
            frame_length,
            payload,
        });
    }
    to_js_value(&LmEcdcChunks { metadata, chunks })
}

#[wasm_bindgen]
pub struct QuantizedLmChunkEncoder {
    meta: OnnxFrameBundleMetadata,
    lm: QuantizedLm,
    state: QuantizedLmState,
    lm_window_frame_length: usize,
    input_symbols: Vec<usize>,
    encoder: ArithmeticEncoder,
    probability_scratch: ProbabilityScratch,
    prefix: Vec<u8>,
    pushed_steps: usize,
}

#[wasm_bindgen]
impl QuantizedLmChunkEncoder {
    #[wasm_bindgen(constructor)]
    pub fn new(
        bundle_json: &str,
        weights: &[u8],
        scale: f32,
    ) -> Result<QuantizedLmChunkEncoder, JsValue> {
        let meta = parse_bundle(bundle_json)?;
        validate_lm_metadata(&meta).map_err(to_js_error)?;
        let weights = QuantizedLmWeights::from_bytes(weights).map_err(to_js_error)?;
        weights.validate_for_metadata(&meta).map_err(to_js_error)?;
        let lm_window_frame_length = weights.frame_length.max(1);
        let lm = QuantizedLm::new(weights);
        let state = lm.initial_state();
        let mut prefix = Vec::new();
        if meta.normalize {
            prefix.extend_from_slice(&scale.to_be_bytes());
        }
        Ok(Self {
            input_symbols: vec![0; meta.num_codebooks],
            meta,
            lm,
            state,
            lm_window_frame_length,
            encoder: ArithmeticEncoder::new(ARITHMETIC_TOTAL_RANGE_BITS).map_err(to_js_error)?,
            probability_scratch: ProbabilityScratch::default(),
            prefix,
            pushed_steps: 0,
        })
    }

    pub fn bitstream_version(&self) -> u8 {
        QUANTIZED_LM_BITSTREAM_VERSION
    }

    #[wasm_bindgen(js_name = lmWindowFrameLength)]
    pub fn lm_window_frame_length(&self) -> usize {
        self.lm_window_frame_length
    }

    pub fn push(&mut self, codes: &[u16]) -> Result<(), JsValue> {
        let symbols = symbols_from_codes(codes, &self.meta).map_err(to_js_error)?;
        self.push_symbols(&symbols).map_err(to_js_error)
    }

    #[wasm_bindgen(js_name = finishPadded)]
    pub fn finish_padded(&mut self, frame_length: usize) -> Result<Vec<u8>, JsValue> {
        let target = if frame_length == 0 {
            self.meta.frame_length
        } else {
            frame_length
        };
        if target > self.meta.frame_length {
            return Err(to_js_error(format!(
                "padded LM frame length {target} exceeds bundle frame length {}",
                self.meta.frame_length
            )));
        }
        if self.pushed_steps != target {
            return Err(to_js_error(format!(
                "zero-code padding is not permitted: pushed {} of {target} required steps",
                self.pushed_steps
            )));
        }
        Ok(self.finish())
    }

    fn push_symbols(&mut self, symbols: &[usize]) -> anyhow::Result<()> {
        if self.pushed_steps >= self.lm_window_frame_length {
            anyhow::bail!(
                "chunk exceeds LM positional capacity {}",
                self.lm_window_frame_length
            );
        }
        let logits = self.lm.forward_step(&mut self.state, &self.input_symbols)?;
        let pdf = probability_columns_from_flat_logits(
            &logits,
            self.meta.lm_cardinality(),
            self.meta.num_codebooks,
            1,
            ProbabilityParameters {
                tau: 1.0,
                logit_step: self.meta.lm_entropy_logit_step(),
                fp_scale: DEFAULT_FP_SCALE,
            },
            &mut self.probability_scratch,
        )?;
        self.encoder.push_pdf_symbols(
            pdf,
            self.meta.lm_cardinality(),
            self.meta.num_codebooks,
            symbols,
            DEFAULT_FP_SCALE,
            DEFAULT_MIN_RANGE,
        )?;
        for (dst, symbol) in self.input_symbols.iter_mut().zip(symbols.iter().copied()) {
            *dst = symbol + 1;
        }
        self.pushed_steps += 1;
        Ok(())
    }

    pub fn finish(&mut self) -> Vec<u8> {
        let mut out = std::mem::take(&mut self.prefix);
        out.extend_from_slice(&self.encoder.finish());
        out
    }
}

#[wasm_bindgen]
pub struct QuantizedLmChunkDecoder {
    meta: OnnxFrameBundleMetadata,
    lm: QuantizedLm,
    state: QuantizedLmState,
    lm_window_frame_length: usize,
    input_symbols: Vec<usize>,
    decoder: ArithmeticDecoder,
    probability_scratch: ProbabilityScratch,
    scale: f32,
    pulled_steps: usize,
}

#[wasm_bindgen]
impl QuantizedLmChunkDecoder {
    #[wasm_bindgen(constructor)]
    pub fn new(
        bundle_json: &str,
        weights: &[u8],
        payload: &[u8],
    ) -> Result<QuantizedLmChunkDecoder, JsValue> {
        let meta = parse_bundle(bundle_json)?;
        validate_lm_metadata(&meta).map_err(to_js_error)?;
        let weights = QuantizedLmWeights::from_bytes(weights).map_err(to_js_error)?;
        weights.validate_for_metadata(&meta).map_err(to_js_error)?;
        let lm_window_frame_length = weights.frame_length.max(1);
        let lm = QuantizedLm::new(weights);
        let state = lm.initial_state();
        let mut cursor = Cursor::new(payload);
        let scale = if meta.normalize {
            let bytes = read_exactly(&mut cursor, 4).map_err(to_js_error)?;
            f32::from_be_bytes(bytes.try_into().expect("slice length"))
        } else {
            1.0
        };
        let remaining = payload.len().saturating_sub(cursor.position() as usize);
        let encoded = read_exactly(&mut cursor, remaining).map_err(to_js_error)?;
        Ok(Self {
            input_symbols: vec![0; meta.num_codebooks],
            meta,
            lm,
            state,
            lm_window_frame_length,
            decoder: ArithmeticDecoder::new(encoded, ARITHMETIC_TOTAL_RANGE_BITS)
                .map_err(to_js_error)?,
            probability_scratch: ProbabilityScratch::default(),
            scale,
            pulled_steps: 0,
        })
    }

    pub fn bitstream_version(&self) -> u8 {
        QUANTIZED_LM_BITSTREAM_VERSION
    }

    #[wasm_bindgen(js_name = lmWindowFrameLength)]
    pub fn lm_window_frame_length(&self) -> usize {
        self.lm_window_frame_length
    }

    pub fn scale(&self) -> f32 {
        self.scale
    }

    pub fn pull(&mut self) -> Result<Vec<u16>, JsValue> {
        if self.pulled_steps >= self.lm_window_frame_length {
            return Err(to_js_error(format!(
                "chunk exceeds LM positional capacity {}",
                self.lm_window_frame_length
            )));
        }
        let logits = self
            .lm
            .forward_step(&mut self.state, &self.input_symbols)
            .map_err(to_js_error)?;
        let pdf = probability_columns_from_flat_logits(
            &logits,
            self.meta.lm_cardinality(),
            self.meta.num_codebooks,
            1,
            ProbabilityParameters {
                tau: 1.0,
                logit_step: self.meta.lm_entropy_logit_step(),
                fp_scale: DEFAULT_FP_SCALE,
            },
            &mut self.probability_scratch,
        )
        .map_err(to_js_error)?;
        let symbols = self
            .decoder
            .pull_symbols(
                pdf,
                self.meta.lm_cardinality(),
                self.meta.num_codebooks,
                DEFAULT_FP_SCALE,
                DEFAULT_MIN_RANGE,
            )
            .map_err(to_js_error)?;
        for (dst, symbol) in self.input_symbols.iter_mut().zip(symbols.iter().copied()) {
            *dst = symbol + 1;
        }
        self.pulled_steps += 1;
        symbols
            .into_iter()
            .map(|symbol| {
                u16::try_from(symbol)
                    .map_err(|_| to_js_error(format!("LM symbol {symbol} does not fit u16")))
            })
            .collect()
    }
}

fn parse_bundle(bundle_json: &str) -> Result<OnnxFrameBundleMetadata, JsValue> {
    let metadata: OnnxFrameBundleMetadata =
        serde_json::from_str(bundle_json).map_err(to_js_error)?;
    metadata.validate().map_err(to_js_error)?;
    Ok(metadata)
}

fn validate_lm_metadata(meta: &OnnxFrameBundleMetadata) -> anyhow::Result<()> {
    meta.validate_lm()
}

fn validate_encoded_frame_length(
    meta: &OnnxFrameBundleMetadata,
    frame_length: usize,
) -> anyhow::Result<()> {
    if frame_length == 0 || frame_length > meta.frame_length {
        anyhow::bail!(
            "ECDC frame length {frame_length} is outside 1..={}",
            meta.frame_length
        );
    }
    if fixed_context_samples(meta.segment_samples, meta.segment_stride)?.is_some()
        && frame_length != meta.frame_length
    {
        anyhow::bail!(
            "fixed-context ECDC requires all {} model-code frames; got {frame_length}",
            meta.frame_length
        );
    }
    Ok(())
}

fn symbols_from_codes(codes: &[u16], meta: &OnnxFrameBundleMetadata) -> anyhow::Result<Vec<usize>> {
    if codes.len() != meta.num_codebooks {
        anyhow::bail!(
            "LM code step length {} does not match num_codebooks {}",
            codes.len(),
            meta.num_codebooks
        );
    }
    let cardinality = meta.lm_cardinality();
    codes
        .iter()
        .copied()
        .map(|code| {
            let symbol = code as usize;
            if symbol >= cardinality {
                anyhow::bail!(
                    "LM symbol {} is outside cardinality {}",
                    symbol,
                    cardinality
                );
            }
            Ok(symbol)
        })
        .collect()
}

/// Crops each fully-decoded fixed model window (`segment_samples` long, the
/// same layout the native decoder produces) down to its owned region and
/// concatenates the results at logical stride positions. The fixed-context
/// seam repair keeps the output length unchanged.
fn fixed_context_crop_concat(
    meta: &OnnxFrameBundleMetadata,
    audio_length: usize,
    decoded_frames: &[f32],
    stride: usize,
    context: usize,
) -> anyhow::Result<Vec<f32>> {
    let starts = segment_starts(audio_length, stride);
    let frame_count = starts.len();
    let channels = meta.channels;
    let segment_samples = meta.segment_samples;

    let expected_total = frame_count
        .checked_mul(channels)
        .and_then(|value| value.checked_mul(segment_samples))
        .ok_or_else(|| {
            anyhow::anyhow!("decoded fixed-context frame buffer size overflows usize")
        })?;
    if decoded_frames.len() != expected_total {
        anyhow::bail!(
            "decoded frame sample count {} does not match expected {} for {} fixed-context frames",
            decoded_frames.len(),
            expected_total,
            frame_count
        );
    }

    let mut output = vec![0.0_f32; channels * audio_length];
    for (frame_index, offset) in starts.into_iter().enumerate() {
        let owned_len = (audio_length - offset).min(stride);
        if context + owned_len > segment_samples {
            anyhow::bail!(
                "decoded model output is too short: context={} owned_len={} decoded={}",
                context,
                owned_len,
                segment_samples
            );
        }
        let frame_base = frame_index * channels * segment_samples;
        for channel in 0..channels {
            let src_base = frame_base + channel * segment_samples + context;
            let dst_base = channel * audio_length + offset;
            output[dst_base..dst_base + owned_len]
                .copy_from_slice(&decoded_frames[src_base..src_base + owned_len]);
        }
    }
    repair_cubic_hermite_seams_planar(
        &mut output,
        channels,
        audio_length,
        stride,
        FIXED_CONTEXT_SEAM_REPAIR_SAMPLES,
    )?;
    Ok(output)
}

fn overlap_add_decoded_frames(
    meta: &OnnxFrameBundleMetadata,
    audio_length: usize,
    decoded_frames: &[f32],
    layout: EcdcChunkLayout,
) -> anyhow::Result<Vec<f32>> {
    let starts = segment_starts(audio_length, layout.stride);
    let frame_count = starts.len();
    if frame_count == 0 {
        return Ok(Vec::new());
    }

    let chunk_samples: Vec<usize> = starts
        .iter()
        .copied()
        .map(|offset| (audio_length - offset).min(layout.samples))
        .collect();
    let native_samples_per_code = meta
        .segment_samples
        .checked_div(meta.frame_length.max(1))
        .filter(|samples| *samples > 0)
        .unwrap_or(320);
    let native_decoded_samples: Vec<usize> = chunk_samples
        .iter()
        .copied()
        .map(|samples| {
            segment_frame_length(samples, meta.segment_samples, meta.frame_length)
                * native_samples_per_code
        })
        .collect();
    let exact_native_total: usize = native_decoded_samples
        .iter()
        .map(|samples| samples * meta.channels)
        .sum();
    let full_native_total = frame_count * meta.channels * meta.segment_samples;
    let exact_chunk_total: usize = chunk_samples
        .iter()
        .map(|samples| samples * meta.channels)
        .sum();
    let source_samples_per_frame = if decoded_frames.len() == exact_native_total {
        native_decoded_samples
    } else if decoded_frames.len() == full_native_total {
        vec![meta.segment_samples; frame_count]
    } else if decoded_frames.len() == exact_chunk_total {
        chunk_samples.clone()
    } else {
        let frame_channel_count = frame_count * meta.channels;
        if !decoded_frames.len().is_multiple_of(frame_channel_count) {
            anyhow::bail!(
                "decoded frame sample count {} is not divisible by {} frame channels for {} frames",
                decoded_frames.len(),
                frame_channel_count,
                frame_count
            );
        }
        vec![decoded_frames.len() / frame_channel_count; frame_count]
    };

    for (index, (decoded_samples, required_samples)) in source_samples_per_frame
        .iter()
        .copied()
        .zip(chunk_samples.iter().copied())
        .enumerate()
    {
        if decoded_samples < required_samples {
            anyhow::bail!(
                "decoded frame {index} sample count {decoded_samples} cannot satisfy ECDC chunk sample count {required_samples}"
            );
        }
    }
    let expected_total: usize = source_samples_per_frame
        .iter()
        .map(|samples| samples * meta.channels)
        .sum();
    if decoded_frames.len() != expected_total {
        anyhow::bail!(
            "decoded frame sample count {} does not match expected {} for {} ECDC frames",
            decoded_frames.len(),
            expected_total,
            frame_count
        );
    }

    let total_size = layout.stride * (frame_count - 1) + layout.samples;
    let mut output = vec![0.0_f32; meta.channels * total_size];
    let mut sum_weight = vec![0.0_f32; total_size];
    let weight = triangle_weight(layout.samples);
    let mut source_frame_base = 0usize;

    for frame in 0..frame_count {
        let offset = frame * layout.stride;
        let decoded_samples = source_samples_per_frame[frame];
        for sample in 0..chunk_samples[frame] {
            let w = weight[sample];
            sum_weight[offset + sample] += w;
            for channel in 0..meta.channels {
                let source_index = source_frame_base + channel * decoded_samples + sample;
                let target_index = channel * total_size + offset + sample;
                output[target_index] += decoded_frames[source_index] * w;
            }
        }
        source_frame_base += meta.channels * decoded_samples;
    }

    let mut trimmed = vec![0.0_f32; meta.channels * audio_length];
    for sample in 0..audio_length {
        let denom = sum_weight[sample];
        if denom <= 0.0 {
            continue;
        }
        for channel in 0..meta.channels {
            trimmed[channel * audio_length + sample] =
                output[channel * total_size + sample] / denom;
        }
    }
    Ok(trimmed)
}

fn triangle_weight(frame_length: usize) -> Vec<f32> {
    (0..frame_length)
        .map(|index| {
            let t = (index + 1) as f32 / (frame_length + 1) as f32;
            0.5 - (t - 0.5).abs()
        })
        .collect()
}

fn to_js_value<T: Serialize + ?Sized>(value: &T) -> Result<JsValue, JsValue> {
    let serializer = serde_wasm_bindgen::Serializer::new().serialize_maps_as_objects(true);
    value.serialize(&serializer).map_err(to_js_error)
}

fn to_js_error(error: impl std::fmt::Display) -> JsValue {
    JsValue::from_str(&error.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ecdc::encode_lm_chunk_payload;
    use crate::portable_lm::PortableLmCodec;
    use ndarray::{Array2, Array3};
    use std::fs;
    use std::path::Path;

    fn test_bundle_json() -> String {
        serde_json::to_string(&OnnxFrameBundleMetadata {
            schema_version: 1,
            model_name: "encodec_48khz_test".to_string(),
            bandwidth_kbps: 12.0,
            sample_rate: 48_000,
            channels: 2,
            segment_samples: 48_000,
            segment_stride: 47_520,
            normalize: true,
            num_codebooks: 8,
            frame_length: 200,
            bits_per_codebook: Some(10),
            codebook_cardinality: Some(1024),
            encode_model: "encode_frame.onnx".to_string(),
            decode_model: "decode_frame.onnx".to_string(),
            lm_quant_weight_model: Some("lm_weights_q8.bin".to_string()),
            lm_dim: Some(128),
            lm_num_layers: Some(1),
            lm_past_context: Some(0),
            lm_logit_step: Some(1.0 / 64.0),
            lm_entropy_logit_step: Some(crate::metadata::Q8_LM_LOGIT_STEP as f32),
            lm_cardinality: Some(1024),
            opset_version: 17,
        })
        .unwrap()
    }

    #[test]
    fn fixed_block_header_validates_one_packet_entry() {
        let bundle_json = test_bundle_json();
        let bundle_meta: OnnxFrameBundleMetadata = serde_json::from_str(&bundle_json).unwrap();
        let weights = [0x42_u8; 32];

        for block_samples in [48_123_usize, 65_537_usize] {
            let mut entry = lm_ecdc_fixed_header_for_weights(
                &bundle_json,
                block_samples,
                QUANTIZED_LM_BITSTREAM_VERSION,
                &weights,
            )
            .unwrap();
            write_chunk(&mut entry, &[1, 2, 3, 4], true).unwrap();

            let mut reader = Cursor::new(entry.as_slice());
            let metadata: EcdcMetadata = read_ecdc_header(&mut reader).unwrap();
            let mut packet_count = 0usize;
            while (reader.position() as usize) < entry.len() {
                read_chunk_payload(&mut reader, true).unwrap();
                packet_count += 1;
            }

            let layout =
                ecdc_chunk_layout_for_chunk_count(&bundle_meta, &metadata, packet_count).unwrap();
            assert_eq!(metadata.audio_length, block_samples);
            assert_eq!(layout.samples, block_samples);
            assert_eq!(layout.stride, block_samples);
            assert_eq!(packet_count, 1);
        }
    }

    /// The real 1.333s fixed-context bundle geometry: a 64,960-sample
    /// private model window holding 480 previous-context samples, 64,000
    /// owned samples, and 480 following-context samples.
    fn fixed_block_bundle_json() -> String {
        serde_json::to_string(&OnnxFrameBundleMetadata {
            schema_version: 1,
            model_name: "encodec_48khz_test".to_string(),
            bandwidth_kbps: 12.0,
            sample_rate: 48_000,
            channels: 2,
            segment_samples: 64_960,
            segment_stride: 64_000,
            normalize: true,
            num_codebooks: 8,
            frame_length: 203,
            bits_per_codebook: Some(10),
            codebook_cardinality: Some(1024),
            encode_model: "encode_frame.onnx".to_string(),
            decode_model: "decode_frame.onnx".to_string(),
            lm_quant_weight_model: Some("lm_weights_q8.bin".to_string()),
            lm_dim: Some(128),
            lm_num_layers: Some(1),
            lm_past_context: Some(0),
            lm_logit_step: Some(1.0 / 64.0),
            lm_entropy_logit_step: Some(crate::metadata::Q8_LM_LOGIT_STEP as f32),
            lm_cardinality: Some(1024),
            opset_version: 17,
        })
        .unwrap()
    }

    #[test]
    fn fixed_block_header_uses_logical_audio_length_and_full_frame_length() {
        // The public ECDC header reports logical owned audio length (al =
        // 64,000), never the private 64,960-sample model window, and the
        // full bundle frame_length (fl = 203), never a proportion of al.
        let bundle_json = fixed_block_bundle_json();
        let weights = [0x42_u8; 32];
        let header = lm_ecdc_fixed_header_for_weights(
            &bundle_json,
            64_000,
            QUANTIZED_LM_BITSTREAM_VERSION,
            &weights,
        )
        .unwrap();
        let metadata: EcdcMetadata = read_ecdc_header(&mut Cursor::new(header.as_slice())).unwrap();
        assert_eq!(metadata.audio_length, 64_000);
        assert_eq!(metadata.lm_frame_length, Some(203));
        assert!(metadata.extra.is_empty());
    }

    #[test]
    fn fixed_block_1800ms_header_uses_logical_audio_length_and_full_frame_length() {
        let bundle_json = serde_json::to_string(&OnnxFrameBundleMetadata {
            schema_version: 1,
            model_name: "encodec_48khz_test".to_string(),
            bandwidth_kbps: 12.0,
            sample_rate: 48_000,
            channels: 2,
            segment_samples: 87_360,
            segment_stride: 86_400,
            normalize: true,
            num_codebooks: 8,
            frame_length: 273,
            bits_per_codebook: Some(10),
            codebook_cardinality: Some(1024),
            encode_model: "encode_frame.onnx".to_string(),
            decode_model: "decode_frame.onnx".to_string(),
            lm_quant_weight_model: Some("lm_weights_q8.bin".to_string()),
            lm_dim: Some(128),
            lm_num_layers: Some(1),
            lm_past_context: Some(0),
            lm_logit_step: Some(1.0 / 64.0),
            lm_entropy_logit_step: Some(crate::metadata::Q8_LM_LOGIT_STEP as f32),
            lm_cardinality: Some(1024),
            opset_version: 17,
        })
        .unwrap();
        let weights = [0x42_u8; 32];
        let header = lm_ecdc_fixed_header_for_weights(
            &bundle_json,
            86_400,
            QUANTIZED_LM_BITSTREAM_VERSION,
            &weights,
        )
        .unwrap();
        let metadata: EcdcMetadata = read_ecdc_header(&mut Cursor::new(header.as_slice())).unwrap();
        assert_eq!(metadata.audio_length, 86_400);
        assert_eq!(metadata.lm_frame_length, Some(273));
        assert!(metadata.extra.is_empty());
    }

    #[test]
    fn fixed_context_geometry_is_recognized_with_480_sample_context() {
        assert_eq!(fixed_context_samples(64_960, 64_000).unwrap(), Some(480));
        assert_eq!(fixed_context_samples(87_360, 86_400).unwrap(), Some(480));
    }

    #[test]
    fn non_fixed_geometry_is_not_misclassified_as_fixed_context() {
        // Legacy EnCodec bundles may also have segment_samples >
        // segment_stride for ordinary overlap-add; only the explicitly
        // recognised fixed-context geometries should be treated specially.
        assert_eq!(fixed_context_samples(48_000, 47_520).unwrap(), None);
        assert_eq!(fixed_context_samples(64_960, 64_960).unwrap(), None);
    }

    #[test]
    fn fixed_context_geometry_rejects_malformed_profiles() {
        // segment_samples smaller than segment_stride for a recognized
        // owned-duration stride.
        assert!(fixed_context_samples(63_999, 64_000).is_err());
        // Odd context excess (cannot be split symmetrically).
        assert!(fixed_context_samples(64_961, 64_000).is_err());
        // Even excess, but not the required 480 samples per side.
        assert!(fixed_context_samples(65_000, 64_000).is_err());
    }

    #[test]
    fn fixed_context_entropy_rejects_partial_model_codes() {
        let meta: OnnxFrameBundleMetadata =
            serde_json::from_str(&fixed_block_bundle_json()).unwrap();
        assert!(validate_encoded_frame_length(&meta, meta.frame_length).is_ok());
        assert!(validate_encoded_frame_length(&meta, meta.frame_length - 1).is_err());
        assert!(validate_encoded_frame_length(&meta, 0).is_err());
    }

    #[test]
    fn fixed_context_crop_concat_repairs_seams_and_returns_owned_audio() {
        let meta: OnnxFrameBundleMetadata =
            serde_json::from_str(&fixed_block_bundle_json()).unwrap();
        let stride = meta.segment_stride;
        let context = 480usize;

        // Two full chunks plus a final partial chunk of 2,000 owned samples.
        let audio_length = stride * 2 + 2_000;
        let frame_count = 3usize;
        let mut decoded = vec![0.0_f32; frame_count * meta.channels * meta.segment_samples];
        for frame in 0..frame_count {
            for channel in 0..meta.channels {
                let base =
                    frame * meta.channels * meta.segment_samples + channel * meta.segment_samples;
                for index in 0..meta.segment_samples {
                    // Encode a value that lets us check exactly which
                    // window position (and therefore which logical sample)
                    // ended up in the output.
                    decoded[base + index] = (frame * 1_000_000 + channel * 100_000 + index) as f32;
                }
            }
        }

        let output =
            fixed_context_crop_concat(&meta, audio_length, &decoded, stride, context).unwrap();
        assert_eq!(output.len(), meta.channels * audio_length);

        // First owned sample of chunk 0 comes from model index `context`.
        assert_eq!(output[0], 480.0);
        // Samples outside the 24-sample repair remain exact crop results.
        assert_eq!(output[stride - 13], (context + stride - 13) as f32);
        assert_eq!(output[stride + 12], (1_000_000 + context + 12) as f32);
        // Twelve samples on each side of the join use the Hermite repair.
        assert_ne!(output[stride - 1], (context + stride - 1) as f32);
        assert_ne!(output[stride], (1_000_000 + context) as f32);
        // The final partial chunk contributes only its true owned length.
        assert_eq!(output[2 * stride + 12], (2_000_000 + context + 12) as f32);
        assert_eq!(output[audio_length - 1], (2_000_000 + 480 + 1_999) as f32);
    }

    #[test]
    fn fixed_context_crop_concat_rejects_too_short_decoded_output() {
        let meta: OnnxFrameBundleMetadata =
            serde_json::from_str(&fixed_block_bundle_json()).unwrap();
        let stride = meta.segment_stride;
        let context = 480usize;
        let audio_length = stride;
        // Too short: missing the full segment_samples length per channel.
        let decoded = vec![0.0_f32; meta.channels * (context + stride - 1)];
        assert!(fixed_context_crop_concat(&meta, audio_length, &decoded, stride, context).is_err());
    }

    #[test]
    fn native_and_wasm_entropy_paths_preserve_codebook_time_order() {
        let bundle_dir = Path::new("onnx-bundles/encodec_48khz_12kbps_1333ms");
        let bundle_path = bundle_dir.join("bundle.json");
        let weights_path = bundle_dir.join("lm_weights_q8.bin");
        if !bundle_path.exists() || !weights_path.exists() {
            eprintln!("skipping entropy parity test; fixed q8 bundle is unavailable");
            return;
        }

        let bundle_json = fs::read_to_string(bundle_path).unwrap();
        let metadata: OnnxFrameBundleMetadata = serde_json::from_str(&bundle_json).unwrap();
        let weights = fs::read(weights_path).unwrap();
        let steps = 4;
        let scale = f32::from_bits(0x3f12_3456);
        let mut codes = Array3::<i64>::zeros((1, metadata.num_codebooks, steps));
        for time in 0..steps {
            for codebook in 0..metadata.num_codebooks {
                codes[[0, codebook, time]] =
                    ((time * 257 + codebook * 73 + 19) % metadata.lm_cardinality()) as i64;
            }
        }

        let mut native_lm =
            PortableLmCodec::from_quantized_weights(metadata.clone(), &weights).unwrap();
        let scales = Array2::from_elem((1, 1), scale);
        let native_payload = encode_lm_chunk_payload(
            &mut native_lm,
            &codes,
            &scales,
            0,
            steps,
            DEFAULT_FP_SCALE,
            DEFAULT_MIN_RANGE,
            1.0,
        )
        .unwrap();

        let mut wasm_encoder = QuantizedLmChunkEncoder::new(&bundle_json, &weights, scale).unwrap();
        for time in 0..steps {
            let step_codes: Vec<u16> = (0..metadata.num_codebooks)
                .map(|codebook| codes[[0, codebook, time]] as u16)
                .collect();
            wasm_encoder.push(&step_codes).unwrap();
        }
        let wasm_payload = wasm_encoder.finish();
        assert_eq!(wasm_payload, native_payload);

        let mut wasm_decoder =
            QuantizedLmChunkDecoder::new(&bundle_json, &weights, &native_payload).unwrap();
        assert_eq!(wasm_decoder.scale().to_bits(), scale.to_bits());
        for time in 0..steps {
            let decoded = wasm_decoder.pull().unwrap();
            let expected: Vec<u16> = (0..metadata.num_codebooks)
                .map(|codebook| codes[[0, codebook, time]] as u16)
                .collect();
            assert_eq!(decoded, expected);
        }
    }
}
