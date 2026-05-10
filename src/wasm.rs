use std::io::Cursor;

use serde::Serialize;
use wasm_bindgen::prelude::*;

use crate::arithmetic::{ArithmeticDecoder, ArithmeticEncoder};
use crate::binary::{
    read_chunk_payload, read_ecdc_header, read_exactly, write_chunk, write_ecdc_header,
};
use crate::format::{
    segment_frame_length, segment_starts, validate_metadata, EcdcMetadata,
    ARITHMETIC_TOTAL_RANGE_BITS, DEFAULT_FP_SCALE, DEFAULT_MIN_RANGE,
};
use crate::metadata::OnnxFrameBundleMetadata;
use crate::raw::{
    decode_raw_frames, encode_raw_ecdc, encode_raw_frame_payload, encode_raw_header,
    overlap_add_decoded_frames, raw_segment_count, RawEcdcFrame,
};

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

#[wasm_bindgen(js_name = initPanicHook)]
pub fn init_panic_hook() {
    console_error_panic_hook::set_once();
}

#[wasm_bindgen(js_name = bundleMetadata)]
pub fn bundle_metadata(bundle_json: &str) -> Result<JsValue, JsValue> {
    let meta = parse_bundle(bundle_json)?;
    to_js_value(&meta)
}

#[wasm_bindgen(js_name = rawEcdcHeader)]
pub fn raw_ecdc_header(bundle_json: &str, audio_length: usize) -> Result<Vec<u8>, JsValue> {
    let meta = parse_bundle(bundle_json)?;
    encode_raw_header(&meta, audio_length).map_err(to_js_error)
}

#[wasm_bindgen(js_name = rawEcdcFramePayload)]
pub fn raw_ecdc_frame_payload(
    bundle_json: &str,
    codes: &[u16],
    scale: f32,
    frame_length: usize,
) -> Result<Vec<u8>, JsValue> {
    let meta = parse_bundle(bundle_json)?;
    encode_raw_frame_payload(&meta, codes, scale, frame_length).map_err(to_js_error)
}

#[wasm_bindgen(js_name = rawEcdcEncode)]
pub fn raw_ecdc_encode(
    bundle_json: &str,
    audio_length: usize,
    frames: JsValue,
) -> Result<Vec<u8>, JsValue> {
    let meta = parse_bundle(bundle_json)?;
    let frames: Vec<RawEcdcFrame> = serde_wasm_bindgen::from_value(frames).map_err(to_js_error)?;
    encode_raw_ecdc(&meta, audio_length, &frames).map_err(to_js_error)
}

#[wasm_bindgen(js_name = rawEcdcMetadata)]
pub fn raw_ecdc_metadata(payload: &[u8]) -> Result<JsValue, JsValue> {
    let metadata: EcdcMetadata =
        read_ecdc_header(&mut Cursor::new(payload)).map_err(to_js_error)?;
    to_js_value(&metadata)
}

#[wasm_bindgen(js_name = rawEcdcDecodeFrames)]
pub fn raw_ecdc_decode_frames(bundle_json: &str, payload: &[u8]) -> Result<JsValue, JsValue> {
    let meta = parse_bundle(bundle_json)?;
    let frames = decode_raw_frames(&meta, payload).map_err(to_js_error)?;
    to_js_value(&frames)
}

#[wasm_bindgen(js_name = rawEcdcSegmentCount)]
pub fn raw_ecdc_segment_count(bundle_json: &str, audio_length: usize) -> Result<usize, JsValue> {
    let meta = parse_bundle(bundle_json)?;
    Ok(raw_segment_count(&meta, audio_length))
}

#[wasm_bindgen(js_name = rawEcdcOverlapAdd)]
pub fn raw_ecdc_overlap_add(
    bundle_json: &str,
    audio_length: usize,
    decoded_frames: &[f32],
) -> Result<Vec<f32>, JsValue> {
    let meta = parse_bundle(bundle_json)?;
    overlap_add_decoded_frames(&meta, audio_length, decoded_frames).map_err(to_js_error)
}

#[wasm_bindgen(js_name = lmEcdcHeader)]
pub fn lm_ecdc_header(bundle_json: &str, audio_length: usize) -> Result<Vec<u8>, JsValue> {
    let meta = parse_bundle(bundle_json)?;
    let metadata = EcdcMetadata::from_bundle(&meta, audio_length, None, true, Some(1.0), false);
    let mut out = Vec::new();
    write_ecdc_header(&mut out, &metadata).map_err(to_js_error)?;
    Ok(out)
}

#[wasm_bindgen(js_name = lmEcdcChunk)]
pub fn lm_ecdc_chunk(payload: &[u8]) -> Result<Vec<u8>, JsValue> {
    let mut out = Vec::new();
    write_chunk(&mut out, payload, false).map_err(to_js_error)?;
    Ok(out)
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

    let mut chunks = Vec::new();
    for offset in segment_starts(metadata.audio_length, meta.segment_stride.max(1)) {
        let samples = (metadata.audio_length - offset).min(meta.segment_samples);
        let frame_length = segment_frame_length(samples, meta.segment_samples, meta.frame_length);
        let payload =
            read_chunk_payload(&mut reader, metadata.chunk_crc_enabled()).map_err(to_js_error)?;
        chunks.push(LmEcdcChunk {
            offset,
            samples,
            frame_length,
            payload,
        });
    }
    if reader.position() as usize != payload.len() {
        return Err(to_js_error(
            "LM ECDC payload has trailing bytes after expected chunks",
        ));
    }

    to_js_value(&LmEcdcChunks { metadata, chunks })
}

#[wasm_bindgen]
pub struct LmChunkEncoder {
    meta: OnnxFrameBundleMetadata,
    encoder: ArithmeticEncoder,
    prefix: Vec<u8>,
}

#[wasm_bindgen]
impl LmChunkEncoder {
    #[wasm_bindgen(constructor)]
    pub fn new(bundle_json: &str, scale: f32) -> Result<LmChunkEncoder, JsValue> {
        let meta = parse_bundle(bundle_json)?;
        validate_lm_metadata(&meta).map_err(to_js_error)?;
        let mut prefix = Vec::new();
        if meta.normalize {
            prefix.extend_from_slice(&scale.to_be_bytes());
        }
        Ok(Self {
            meta,
            encoder: ArithmeticEncoder::new(ARITHMETIC_TOTAL_RANGE_BITS).map_err(to_js_error)?,
            prefix,
        })
    }

    pub fn push(&mut self, logits: &[f32], codes: &[u16]) -> Result<(), JsValue> {
        let symbols = symbols_from_codes(codes, &self.meta).map_err(to_js_error)?;
        let pdf = probability_columns_from_logits(logits, &self.meta, 1.0).map_err(to_js_error)?;
        self.encoder
            .push_pdf_symbols(
                &pdf,
                self.meta.lm_cardinality(),
                self.meta.num_codebooks,
                &symbols,
                DEFAULT_FP_SCALE,
                DEFAULT_MIN_RANGE,
            )
            .map_err(to_js_error)
    }

    pub fn finish(&mut self) -> Vec<u8> {
        let mut out = std::mem::take(&mut self.prefix);
        out.extend_from_slice(&self.encoder.finish());
        out
    }
}

#[wasm_bindgen]
pub struct LmChunkDecoder {
    meta: OnnxFrameBundleMetadata,
    decoder: ArithmeticDecoder,
    scale: f32,
}

#[wasm_bindgen]
impl LmChunkDecoder {
    #[wasm_bindgen(constructor)]
    pub fn new(bundle_json: &str, payload: &[u8]) -> Result<LmChunkDecoder, JsValue> {
        let meta = parse_bundle(bundle_json)?;
        validate_lm_metadata(&meta).map_err(to_js_error)?;
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
            meta,
            decoder: ArithmeticDecoder::new(encoded, ARITHMETIC_TOTAL_RANGE_BITS)
                .map_err(to_js_error)?,
            scale,
        })
    }

    pub fn scale(&self) -> f32 {
        self.scale
    }

    pub fn pull(&mut self, logits: &[f32]) -> Result<Vec<u16>, JsValue> {
        let pdf = probability_columns_from_logits(logits, &self.meta, 1.0).map_err(to_js_error)?;
        let symbols = self
            .decoder
            .pull_symbols(
                &pdf,
                self.meta.lm_cardinality(),
                self.meta.num_codebooks,
                DEFAULT_FP_SCALE,
                DEFAULT_MIN_RANGE,
            )
            .map_err(to_js_error)?;
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
    serde_json::from_str(bundle_json).map_err(to_js_error)
}

fn validate_lm_metadata(meta: &OnnxFrameBundleMetadata) -> anyhow::Result<()> {
    meta.lm_dim()?;
    meta.lm_num_layers()?;
    meta.lm_past_context()?;
    if meta.lm_cardinality() == 0 {
        anyhow::bail!("LM cardinality must be non-zero");
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

fn probability_columns_from_logits(
    logits: &[f32],
    meta: &OnnxFrameBundleMetadata,
    lm_tau: f64,
) -> anyhow::Result<Vec<f64>> {
    let card = meta.lm_cardinality();
    let codebooks = meta.num_codebooks;
    if logits.len() != card * codebooks {
        anyhow::bail!(
            "LM logits length {} does not match cardinality {} * codebooks {}",
            logits.len(),
            card,
            codebooks
        );
    }

    let mut pdf = vec![0.0_f64; card * codebooks];
    let mut quantized = vec![0.0_f64; card];
    let mut probs = vec![0.0_f64; card];
    let uniform = 1.0 / card as f64;
    let near_pdf_threshold = 0.25 / DEFAULT_FP_SCALE as f64;
    let logit_step = meta.lm_logit_step();

    for codebook in 0..codebooks {
        let mut max_value = f64::NEG_INFINITY;
        let mut min_value = f64::INFINITY;
        for bin in 0..card {
            let raw = logits[bin * codebooks + codebook] as f64 / lm_tau;
            let quantized_value = quantize_logit(raw, logit_step);
            quantized[bin] = quantized_value;
            max_value = max_value.max(quantized_value);
            min_value = min_value.min(quantized_value);
        }

        let mut denom = 0.0_f64;
        for bin in 0..card {
            let value = (quantized[bin] - max_value).exp();
            probs[bin] = value;
            denom += value;
        }
        if !denom.is_finite() || denom <= 0.0 {
            for bin in 0..card {
                pdf[bin * codebooks + codebook] = uniform;
            }
            continue;
        }

        let mut max_pdf = 0.0_f64;
        let mut min_pdf = f64::INFINITY;
        for prob in probs.iter_mut() {
            *prob /= denom;
            max_pdf = max_pdf.max(*prob);
            min_pdf = min_pdf.min(*prob);
        }
        let near_uniform = (max_value - min_value) <= (2.0 * logit_step)
            || (max_pdf - min_pdf) <= near_pdf_threshold;
        for bin in 0..card {
            pdf[bin * codebooks + codebook] = if near_uniform { uniform } else { probs[bin] };
        }
    }

    Ok(pdf)
}

fn quantize_logit(value: f64, step: f64) -> f64 {
    let eps = 2_f64.powi(-40);
    let y = value / step;
    (y + 0.5 - eps).floor() * step
}

fn to_js_value<T: Serialize + ?Sized>(value: &T) -> Result<JsValue, JsValue> {
    let serializer = serde_wasm_bindgen::Serializer::new().serialize_maps_as_objects(true);
    value.serialize(&serializer).map_err(to_js_error)
}

fn to_js_error(error: impl std::fmt::Display) -> JsValue {
    JsValue::from_str(&error.to_string())
}
