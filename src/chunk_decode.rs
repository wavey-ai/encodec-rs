//! Reusable packet decoder extracted unchanged from the browser bindings.
//! SoundKit and the existing WASM API call this same entropy implementation.
use crate::arithmetic::{ArithmeticDecoder, CdfScratch};
use crate::binary::{read_chunk_payload, read_ecdc_header, read_exactly};
use crate::ecdc_presets::fixed_context_samples;
use crate::entropy::{
    probability_columns_from_flat_logits, ProbabilityParameters, ProbabilityScratch,
};
use crate::format::{
    ecdc_chunk_layout_for_chunk_count, ecdc_lm_frame_length, segment_starts, validate_metadata,
    EcdcMetadata, ARITHMETIC_TOTAL_RANGE_BITS, DEFAULT_FP_SCALE, DEFAULT_MIN_RANGE,
    QUANTIZED_LM_BITSTREAM_VERSION,
};
use crate::metadata::FrameBundleMetadata;
use crate::quantized_lm::{QuantizedLm, QuantizedLmState, QuantizedLmWeights};
use std::io::Cursor;

#[derive(Debug, Clone, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct LmEcdcChunk {
    pub offset: usize,
    pub samples: usize,
    pub frame_length: usize,
    pub payload: Vec<u8>,
}

#[derive(Debug, Clone, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct LmEcdcChunks {
    pub metadata: EcdcMetadata,
    pub chunks: Vec<LmEcdcChunk>,
}

pub fn lm_ecdc_decode_chunks(
    bundle_json: &str,
    payload: &[u8],
) -> Result<LmEcdcChunks, anyhow::Error> {
    let meta = parse_bundle(bundle_json)?;
    let mut reader = Cursor::new(payload);
    let metadata: EcdcMetadata = read_ecdc_header(&mut reader)?;
    validate_metadata(&meta, &metadata)?;
    if !metadata.use_lm {
        return Err(anyhow::anyhow!("ECDC payload does not use LM coding"));
    }

    let mut raw_chunks = Vec::new();
    while (reader.position() as usize) < payload.len() {
        raw_chunks.push(read_chunk_payload(&mut reader, true)?);
    }

    let layout = ecdc_chunk_layout_for_chunk_count(&meta, &metadata, raw_chunks.len())?;
    let context = fixed_context_samples(meta.segment_samples, meta.segment_stride)?;
    let mut chunks = Vec::new();
    let starts = segment_starts(metadata.audio_length, layout.stride);
    if starts.len() != raw_chunks.len() {
        return Err(anyhow::anyhow!(format!(
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
    Ok(LmEcdcChunks { metadata, chunks })
}

/// The parsed LM of one model bundle: the metadata, the weights and their
/// scratch space. One model decodes any number of chunks, so a host that
/// decodes a stream parses the bundle and the weights once.
pub struct QuantizedLmModel {
    meta: FrameBundleMetadata,
    lm: QuantizedLm,
    lm_window_frame_length: usize,
}

impl QuantizedLmModel {
    pub fn new(bundle_json: &str, weights: &[u8]) -> Result<QuantizedLmModel, anyhow::Error> {
        let meta = parse_bundle(bundle_json)?;
        validate_lm_metadata(&meta)?;
        let weights = QuantizedLmWeights::from_bytes(weights)?;
        weights.validate_for_metadata(&meta)?;
        let lm_window_frame_length = weights.frame_length.max(1);
        Ok(Self {
            meta,
            lm: QuantizedLm::new(weights),
            lm_window_frame_length,
        })
    }

    pub fn lm_window_frame_length(&self) -> usize {
        self.lm_window_frame_length
    }

    /// Decodes `frame_length` steps of one chunk payload: the chunk scale and
    /// the codes, codebook-major. The result is the result of a new
    /// `QuantizedLmChunkDecoder` for the payload.
    pub fn decode_chunk(
        &mut self,
        payload: &[u8],
        frame_length: usize,
    ) -> Result<(f32, Vec<u16>), anyhow::Error> {
        let mut cursor = ChunkCursor::new(self, payload)?;
        let codes = cursor.pull_all(self, frame_length)?;
        Ok((cursor.scale, codes))
    }
}

/// The per-chunk decoding state: the LM state, the arithmetic decoder and
/// their scratch space.
struct ChunkCursor {
    state: QuantizedLmState,
    input_symbols: Vec<usize>,
    decoder: ArithmeticDecoder,
    probability_scratch: ProbabilityScratch,
    cdf_scratch: CdfScratch,
    scale: f32,
    pulled_steps: usize,
}

impl ChunkCursor {
    fn new(model: &QuantizedLmModel, payload: &[u8]) -> Result<ChunkCursor, anyhow::Error> {
        let state = model.lm.initial_state();
        let mut cursor = Cursor::new(payload);
        let scale = if model.meta.normalize {
            let bytes = read_exactly(&mut cursor, 4)?;
            f32::from_be_bytes(bytes.try_into().expect("slice length"))
        } else {
            1.0
        };
        let remaining = payload.len().saturating_sub(cursor.position() as usize);
        let encoded = read_exactly(&mut cursor, remaining)?;
        Ok(Self {
            input_symbols: vec![0; model.meta.num_codebooks],
            state,
            decoder: ArithmeticDecoder::new(encoded, ARITHMETIC_TOTAL_RANGE_BITS)?,
            probability_scratch: ProbabilityScratch::default(),
            cdf_scratch: CdfScratch::default(),
            scale,
            pulled_steps: 0,
        })
    }

    fn pull(&mut self, model: &mut QuantizedLmModel) -> Result<Vec<u16>, anyhow::Error> {
        let mut symbols = vec![0; model.meta.num_codebooks];
        self.pull_symbols_into(model, &mut symbols)?;
        symbols
            .into_iter()
            .map(|symbol| {
                u16::try_from(symbol)
                    .map_err(|_| anyhow::anyhow!(format!("LM symbol {symbol} does not fit u16")))
            })
            .collect()
    }

    fn pull_all(
        &mut self,
        model: &mut QuantizedLmModel,
        frame_length: usize,
    ) -> Result<Vec<u16>, anyhow::Error> {
        let remaining = model
            .lm_window_frame_length
            .saturating_sub(self.pulled_steps);
        if frame_length > remaining {
            return Err(anyhow::anyhow!(format!(
                "requested {frame_length} LM steps with only {remaining} remaining",
            )));
        }

        let codebooks = model.meta.num_codebooks;
        let mut codes = vec![0_u16; codebooks * frame_length];
        let mut symbols = vec![0_usize; codebooks];
        for step in 0..frame_length {
            self.pull_symbols_into(model, &mut symbols)?;
            for (codebook, symbol) in symbols.iter().copied().enumerate() {
                codes[codebook * frame_length + step] = u16::try_from(symbol)
                    .map_err(|_| anyhow::anyhow!(format!("LM symbol {symbol} does not fit u16")))?;
            }
        }
        Ok(codes)
    }

    fn pull_symbols_into(
        &mut self,
        model: &mut QuantizedLmModel,
        symbols: &mut [usize],
    ) -> Result<(), anyhow::Error> {
        if self.pulled_steps >= model.lm_window_frame_length {
            return Err(anyhow::anyhow!(format!(
                "chunk exceeds LM positional capacity {}",
                model.lm_window_frame_length
            )));
        }
        let logits = model
            .lm
            .forward_step_borrowed(&mut self.state, &self.input_symbols)?;
        let pdf = probability_columns_from_flat_logits(
            logits,
            model.meta.lm_cardinality(),
            model.meta.num_codebooks,
            1,
            ProbabilityParameters {
                tau: 1.0,
                logit_step: model.meta.lm_entropy_logit_step(),
                fp_scale: DEFAULT_FP_SCALE,
            },
            &mut self.probability_scratch,
        )?;
        self.decoder.pull_valid_pdf_symbols_into_with_scratch(
            pdf,
            model.meta.lm_cardinality(),
            model.meta.num_codebooks,
            DEFAULT_FP_SCALE,
            DEFAULT_MIN_RANGE,
            &mut self.cdf_scratch,
            symbols,
        )?;
        for (dst, symbol) in self.input_symbols.iter_mut().zip(symbols.iter().copied()) {
            *dst = symbol + 1;
        }
        self.pulled_steps += 1;
        Ok(())
    }
}

/// Decodes one chunk payload step by step with its own copy of the model.
pub struct QuantizedLmChunkDecoder {
    model: QuantizedLmModel,
    cursor: ChunkCursor,
}

impl QuantizedLmChunkDecoder {
    pub fn new(
        bundle_json: &str,
        weights: &[u8],
        payload: &[u8],
    ) -> Result<QuantizedLmChunkDecoder, anyhow::Error> {
        let model = QuantizedLmModel::new(bundle_json, weights)?;
        let cursor = ChunkCursor::new(&model, payload)?;
        Ok(Self { model, cursor })
    }

    pub fn bitstream_version(&self) -> u8 {
        QUANTIZED_LM_BITSTREAM_VERSION
    }

    pub fn lm_window_frame_length(&self) -> usize {
        self.model.lm_window_frame_length
    }

    pub fn scale(&self) -> f32 {
        self.cursor.scale
    }

    pub fn pull(&mut self) -> Result<Vec<u16>, anyhow::Error> {
        self.cursor.pull(&mut self.model)
    }

    pub fn pull_all(&mut self, frame_length: usize) -> Result<Vec<u16>, anyhow::Error> {
        self.cursor.pull_all(&mut self.model, frame_length)
    }
}

fn parse_bundle(bundle_json: &str) -> anyhow::Result<FrameBundleMetadata> {
    let metadata: FrameBundleMetadata = serde_json::from_str(bundle_json)?;
    metadata.validate()?;
    Ok(metadata)
}
fn validate_lm_metadata(meta: &FrameBundleMetadata) -> anyhow::Result<()> {
    meta.validate_lm()
}

#[cfg(test)]
mod model_reuse_tests {
    use super::*;
    use std::path::Path;

    /// One model decoding a run of chunks gives each chunk the scale, codes
    /// and errors of a new chunk decoder, including after a chunk that
    /// fails partway.
    #[test]
    fn reused_model_matches_new_decoder_per_chunk() {
        let bundle_dir = Path::new("onnx-bundles/encodec_48khz_12kbps_1333ms");
        let bundle_path = bundle_dir.join("bundle.json");
        let weights_path = bundle_dir.join("lm_weights_q8.bin");
        if !bundle_path.exists() || !weights_path.exists() {
            eprintln!("skipping model reuse test; fixed q8 bundle is unavailable");
            return;
        }
        let bundle_json = std::fs::read_to_string(bundle_path).unwrap();
        let weights = std::fs::read(weights_path).unwrap();

        let mut state = 0x9e37_79b9_7f4a_7c15u64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        let mut model = QuantizedLmModel::new(&bundle_json, &weights).unwrap();
        let window = model.lm_window_frame_length();
        let mut decoded = 0;
        for round in 0..12 {
            let length = match round % 4 {
                0 => 3,
                1 => 64,
                2 => 600,
                _ => (next() % 2_000) as usize,
            };
            let payload: Vec<u8> = (0..length).map(|_| next() as u8).collect();
            let steps = match round % 3 {
                0 => 4,
                1 => window.min(20),
                _ => window + 1,
            };
            let reused = model
                .decode_chunk(&payload, steps)
                .map_err(|e| e.to_string());
            let fresh = QuantizedLmChunkDecoder::new(&bundle_json, &weights, &payload)
                .and_then(|mut decoder| {
                    let scale = decoder.scale();
                    decoder.pull_all(steps).map(|codes| (scale, codes))
                })
                .map_err(|e| e.to_string());
            let view = |result: &Result<(f32, Vec<u16>), String>| {
                result
                    .as_ref()
                    .map(|(scale, codes)| (scale.to_bits(), codes.clone()))
                    .map_err(Clone::clone)
            };
            assert_eq!(view(&reused), view(&fresh), "round {round}");
            decoded += usize::from(reused.is_ok());
        }
        assert!(decoded >= 4, "{decoded} chunks decoded");
    }
}
