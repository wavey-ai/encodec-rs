use std::collections::BTreeMap;
use std::io::Cursor;

use anyhow::{Result, bail};
use ndarray::{Array2, Array3, Array4};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::arithmetic::{ArithmeticDecoder, ArithmeticEncoder};
use crate::binary::{
    BitPacker, BitUnpacker, read_chunk_payload, read_ecdc_header, read_exactly, write_chunk,
    write_ecdc_header,
};
use crate::onnx::{OnnxFrameBundleMetadata, OnnxFrameCodec, OnnxLmCodec};

pub const DEFAULT_FP_SCALE: i64 = 1 << 13;
pub const DEFAULT_MIN_RANGE: i64 = 2;
pub const RAW_BITSTREAM_VERSION: u8 = 0;
pub const DETERMINISTIC_LM_BITSTREAM_VERSION: u8 = 4;
pub const ARITHMETIC_TOTAL_RANGE_BITS: u32 = 24;

#[derive(Debug, Clone, Default)]
pub struct SourceAudioMetadata {
    pub sample_rate: Option<u32>,
    pub channels: Option<u16>,
    pub total_frames: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EcdcMetadata {
    #[serde(rename = "m")]
    pub model_name: String,
    #[serde(rename = "al")]
    pub audio_length: usize,
    #[serde(rename = "nc")]
    pub num_codebooks: usize,
    #[serde(rename = "lm")]
    pub use_lm: bool,
    #[serde(rename = "fp", default = "default_fp_scale")]
    pub fp_scale: i64,
    #[serde(rename = "mr", default = "default_min_range")]
    pub min_range: i64,
    #[serde(rename = "acv", default)]
    pub bitstream_version: u8,
    #[serde(rename = "tau", skip_serializing_if = "Option::is_none")]
    pub lm_tau: Option<f32>,
    #[serde(rename = "osr", skip_serializing_if = "Option::is_none")]
    pub original_sample_rate: Option<u32>,
    #[serde(rename = "och", skip_serializing_if = "Option::is_none")]
    pub original_channels: Option<u16>,
    #[serde(rename = "ofr", skip_serializing_if = "Option::is_none")]
    pub original_total_frames: Option<usize>,
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

impl EcdcMetadata {
    pub fn from_codec(
        codec: &OnnxFrameCodec,
        audio_length: usize,
        source: Option<&SourceAudioMetadata>,
        use_lm: bool,
        lm_tau: Option<f32>,
    ) -> Self {
        let meta = codec.metadata();
        Self {
            model_name: meta.model_name.clone(),
            audio_length,
            num_codebooks: meta.num_codebooks,
            use_lm,
            fp_scale: DEFAULT_FP_SCALE,
            min_range: DEFAULT_MIN_RANGE,
            bitstream_version: if use_lm {
                DETERMINISTIC_LM_BITSTREAM_VERSION
            } else {
                RAW_BITSTREAM_VERSION
            },
            lm_tau,
            original_sample_rate: source.and_then(|value| value.sample_rate),
            original_channels: source.and_then(|value| value.channels),
            original_total_frames: source.and_then(|value| value.total_frames),
            extra: BTreeMap::new(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct DecodedEcdcAudio {
    pub metadata: EcdcMetadata,
    pub audio: Array3<f32>,
}

pub fn encode_audio_to_ecdc(
    codec: &mut OnnxFrameCodec,
    lm_codec: Option<&mut OnnxLmCodec>,
    audio: &Array3<f32>,
    source: Option<&SourceAudioMetadata>,
) -> Result<Vec<u8>> {
    encode_audio_to_ecdc_impl(codec, lm_codec, audio, source)
}

pub fn encode_audio_to_raw_ecdc(
    codec: &mut OnnxFrameCodec,
    audio: &Array3<f32>,
    source: Option<&SourceAudioMetadata>,
) -> Result<Vec<u8>> {
    encode_audio_to_ecdc_impl(codec, None, audio, source)
}

pub fn decode_ecdc(
    codec: &mut OnnxFrameCodec,
    lm_codec: Option<&mut OnnxLmCodec>,
    payload: &[u8],
) -> Result<DecodedEcdcAudio> {
    decode_ecdc_impl(codec, lm_codec, payload)
}

pub fn decode_raw_ecdc(codec: &mut OnnxFrameCodec, payload: &[u8]) -> Result<DecodedEcdcAudio> {
    decode_ecdc_impl(codec, None, payload)
}

fn encode_audio_to_ecdc_impl(
    codec: &mut OnnxFrameCodec,
    mut lm_codec: Option<&mut OnnxLmCodec>,
    audio: &Array3<f32>,
    source: Option<&SourceAudioMetadata>,
) -> Result<Vec<u8>> {
    let shape = audio.shape();
    if shape.len() != 3 || shape[0] != 1 {
        bail!("audio must have shape [1, channels, samples], got {:?}", shape);
    }

    let model_meta = codec.metadata().clone();
    if shape[1] != model_meta.channels {
        bail!(
            "audio channel mismatch, expected {}, got {}",
            model_meta.channels,
            shape[1]
        );
    }

    let lm_tau = if lm_codec.is_some() { Some(1.0_f32) } else { None };
    let use_lm = lm_codec.is_some();
    let mut out = Vec::new();
    let metadata = EcdcMetadata::from_codec(codec, shape[2], source, use_lm, lm_tau);
    write_ecdc_header(&mut out, &metadata)?;

    for (frame_length, batch) in encode_segments(audio, &model_meta) {
        let (codes_full, scale) = codec.encode_frame(&batch)?;
        let codes = trim_codes(&codes_full, frame_length);
        if let Some(lm_codec) = lm_codec.as_deref_mut() {
            let payload = encode_lm_chunk_payload(
                lm_codec,
                &codes,
                &scale,
                metadata.fp_scale,
                metadata.min_range,
                metadata.lm_tau.unwrap_or(1.0) as f64,
            )?;
            write_chunk(&mut out, &payload)?;
        } else {
            write_raw_frame_payload(&mut out, &codes, &scale, &model_meta)?;
        }
    }

    Ok(out)
}

fn decode_ecdc_impl(
    codec: &mut OnnxFrameCodec,
    mut lm_codec: Option<&mut OnnxLmCodec>,
    payload: &[u8],
) -> Result<DecodedEcdcAudio> {
    let mut reader = Cursor::new(payload);
    let metadata: EcdcMetadata = read_ecdc_header(&mut reader)?;
    let bundle_meta = codec.metadata().clone();
    validate_metadata(&bundle_meta, &metadata)?;

    let mut frames = Vec::new();
    let starts = segment_starts(metadata.audio_length, bundle_meta.segment_stride.max(1));
    for offset in starts {
        let this_len = (metadata.audio_length - offset).min(bundle_meta.segment_samples);
        let frame_length =
            segment_frame_length(this_len, bundle_meta.segment_samples, bundle_meta.frame_length);
        let frame = match metadata.bitstream_version {
            RAW_BITSTREAM_VERSION => {
                decode_raw_frame_payload(codec, &mut reader, &bundle_meta, this_len, frame_length)?
            }
            DETERMINISTIC_LM_BITSTREAM_VERSION => {
                let Some(lm_codec) = lm_codec.as_deref_mut() else {
                    bail!("payload requires LM decoding, but no LM bundle was provided");
                };
                match read_chunk_payload(&mut reader) {
                    Ok(chunk) => decode_lm_chunk_payload(
                        codec,
                        lm_codec,
                        &bundle_meta,
                        &metadata,
                        &chunk,
                        this_len,
                        frame_length,
                    )
                    .unwrap_or_else(|_| silence_frame(bundle_meta.channels, this_len)),
                    Err(_) => silence_frame(bundle_meta.channels, this_len),
                }
            }
            other => bail!("unsupported ECDC bitstream version {other}"),
        };
        frames.push(frame);
    }

    let reconstructed = if frames.len() <= 1 {
        frames
            .into_iter()
            .next()
            .unwrap_or_else(|| Array3::<f32>::zeros((1, bundle_meta.channels, 0)))
    } else {
        linear_overlap_add(&frames, bundle_meta.segment_stride)
    };
    let mut trimmed = Array3::<f32>::zeros((1, bundle_meta.channels, metadata.audio_length));
    for channel in 0..bundle_meta.channels {
        for index in 0..metadata.audio_length {
            trimmed[[0, channel, index]] = reconstructed[[0, channel, index]];
        }
    }

    Ok(DecodedEcdcAudio {
        metadata,
        audio: trimmed,
    })
}

fn write_raw_frame_payload(
    out: &mut Vec<u8>,
    codes: &Array3<i64>,
    scale: &Array2<f32>,
    meta: &OnnxFrameBundleMetadata,
) -> Result<()> {
    if meta.normalize {
        out.extend_from_slice(&scale[[0, 0]].to_be_bytes());
    }
    let frame_length = codes.shape()[2];
    let mut packer = BitPacker::new(meta.bits_per_codebook());
    for t in 0..frame_length {
        for codebook in 0..meta.num_codebooks {
            let value = codes[[0, codebook, t]];
            if value < 0 || value > u16::MAX as i64 {
                bail!("code value {value} is out of range for raw ECDC bitpacking");
            }
            packer.push(value as u16);
        }
    }
    out.extend_from_slice(&packer.finish());
    Ok(())
}

fn decode_raw_frame_payload(
    codec: &mut OnnxFrameCodec,
    reader: &mut Cursor<&[u8]>,
    meta: &OnnxFrameBundleMetadata,
    this_len: usize,
    frame_length: usize,
) -> Result<Array3<f32>> {
    let scale = if meta.normalize {
        let bytes = read_exactly(reader, 4)?;
        Array2::from_shape_vec(
            (1, 1),
            vec![f32::from_be_bytes(bytes.try_into().expect("slice length"))],
        )
        .expect("shape")
    } else {
        Array2::from_shape_vec((1, 1), vec![1.0_f32]).expect("shape")
    };

    let bit_count = frame_length * meta.num_codebooks * meta.bits_per_codebook() as usize;
    let byte_len = bit_count.div_ceil(8);
    let packed = read_exactly(reader, byte_len)?;
    let mut unpacker = BitUnpacker::new(meta.bits_per_codebook(), packed);
    let mut codes = Array3::<i64>::zeros((1, meta.num_codebooks, meta.frame_length));
    for t in 0..frame_length {
        for codebook in 0..meta.num_codebooks {
            let value = unpacker
                .pull()
                .ok_or_else(|| anyhow::anyhow!("raw ECDC stream ended before expected code values"))?;
            codes[[0, codebook, t]] = value as i64;
        }
    }
    decode_codes(codec, &codes, &scale, this_len)
}

fn encode_lm_chunk_payload(
    lm_codec: &mut OnnxLmCodec,
    codes: &Array3<i64>,
    scale: &Array2<f32>,
    fp_scale: i64,
    min_range: i64,
    lm_tau: f64,
) -> Result<Vec<u8>> {
    let meta = lm_codec.metadata().clone();
    let frame_shape = codes.shape();
    let frame_length = frame_shape[2];
    let teacher = teacher_forced_indices(codes);
    let initial_states = lm_codec.initial_states(1)?;
    let (logits, _offset, _states) = lm_codec.forward_logits(&teacher, 0, &initial_states)?;
    let pdf = probability_columns_from_logits(&logits, lm_tau, meta.lm_logit_step(), fp_scale)?;
    let symbols = flatten_symbols_time_major(codes)?;

    let mut payload = Vec::new();
    if meta.normalize {
        payload.extend_from_slice(&scale[[0, 0]].to_be_bytes());
    }
    let mut encoder = ArithmeticEncoder::new(ARITHMETIC_TOTAL_RANGE_BITS)?;
    encoder.push_pdf_symbols(
        &pdf,
        meta.lm_cardinality(),
        meta.num_codebooks * frame_length,
        &symbols,
        fp_scale,
        min_range,
    )?;
    payload.extend_from_slice(&encoder.finish());
    Ok(payload)
}

fn decode_lm_chunk_payload(
    codec: &mut OnnxFrameCodec,
    lm_codec: &mut OnnxLmCodec,
    model_meta: &OnnxFrameBundleMetadata,
    metadata: &EcdcMetadata,
    payload: &[u8],
    this_len: usize,
    frame_length: usize,
) -> Result<Array3<f32>> {
    let mut cursor = Cursor::new(payload);
    let scale = if model_meta.normalize {
        let bytes = read_exactly(&mut cursor, 4)?;
        Array2::from_shape_vec(
            (1, 1),
            vec![f32::from_be_bytes(bytes.try_into().expect("slice length"))],
        )
        .expect("shape")
    } else {
        Array2::from_shape_vec((1, 1), vec![1.0_f32]).expect("shape")
    };
    let remaining = payload.len().saturating_sub(cursor.position() as usize);
    let encoded = read_exactly(&mut cursor, remaining)?;
    let mut decoder = ArithmeticDecoder::new(encoded, ARITHMETIC_TOTAL_RANGE_BITS)?;
    let mut codes = Array3::<i64>::zeros((1, model_meta.num_codebooks, model_meta.frame_length));
    let mut states = lm_codec.initial_states(1)?;
    let mut offset = 0_i64;
    let mut input = Array3::<i64>::zeros((1, model_meta.num_codebooks, 1));
    let lm_tau = metadata.lm_tau.unwrap_or(1.0) as f64;

    for t in 0..frame_length {
        let (logits, next_offset, next_states) = lm_codec.forward_logits(&input, offset, &states)?;
        let pdf = probability_columns_from_logits(
            &logits,
            lm_tau,
            lm_codec.metadata().lm_logit_step(),
            metadata.fp_scale,
        )?;
        let symbols = decoder.pull_symbols(
            &pdf,
            lm_codec.metadata().lm_cardinality(),
            model_meta.num_codebooks,
            metadata.fp_scale,
            metadata.min_range,
        )?;
        for codebook in 0..model_meta.num_codebooks {
            let value = symbols[codebook] as i64;
            codes[[0, codebook, t]] = value;
            input[[0, codebook, 0]] = value + 1;
        }
        states = next_states;
        offset = next_offset;
    }

    decode_codes(codec, &codes, &scale, this_len)
}

fn decode_codes(
    codec: &mut OnnxFrameCodec,
    codes: &Array3<i64>,
    scale: &Array2<f32>,
    this_len: usize,
) -> Result<Array3<f32>> {
    let decoded = codec.decode_frame(codes, scale)?;
    let channels = decoded.shape()[1];
    let mut trimmed = Array3::<f32>::zeros((1, channels, this_len));
    for channel in 0..channels {
        for index in 0..this_len {
            trimmed[[0, channel, index]] = decoded[[0, channel, index]];
        }
    }
    Ok(trimmed)
}

fn encode_segments(
    audio: &Array3<f32>,
    meta: &OnnxFrameBundleMetadata,
) -> Vec<(usize, Array3<f32>)> {
    let total_samples = audio.shape()[2];
    let mut segments = Vec::new();
    for offset in segment_starts(total_samples, meta.segment_stride.max(1)) {
        let copy_len = (total_samples - offset).min(meta.segment_samples);
        let frame_length = segment_frame_length(copy_len, meta.segment_samples, meta.frame_length);
        let mut batch = Array3::<f32>::zeros((1, meta.channels, meta.segment_samples));
        for channel in 0..meta.channels {
            for index in 0..copy_len {
                batch[[0, channel, index]] = audio[[0, channel, offset + index]];
            }
        }
        segments.push((frame_length, batch));
    }
    segments
}

fn trim_codes(codes: &Array3<i64>, frame_length: usize) -> Array3<i64> {
    let codebooks = codes.shape()[1];
    let mut trimmed = Array3::<i64>::zeros((1, codebooks, frame_length));
    for codebook in 0..codebooks {
        for t in 0..frame_length {
            trimmed[[0, codebook, t]] = codes[[0, codebook, t]];
        }
    }
    trimmed
}

fn teacher_forced_indices(codes: &Array3<i64>) -> Array3<i64> {
    let shape = codes.shape();
    let codebooks = shape[1];
    let frame_length = shape[2];
    let mut teacher = Array3::<i64>::zeros((1, codebooks, frame_length));
    for codebook in 0..codebooks {
        for t in 1..frame_length {
            teacher[[0, codebook, t]] = 1 + codes[[0, codebook, t - 1]];
        }
    }
    teacher
}

fn flatten_symbols_time_major(codes: &Array3<i64>) -> Result<Vec<usize>> {
    let shape = codes.shape();
    let codebooks = shape[1];
    let frame_length = shape[2];
    let mut out = Vec::with_capacity(codebooks * frame_length);
    for t in 0..frame_length {
        for codebook in 0..codebooks {
            let value = codes[[0, codebook, t]];
            if value < 0 {
                bail!("code symbol must be non-negative, got {value}");
            }
            out.push(value as usize);
        }
    }
    Ok(out)
}

fn probability_columns_from_logits(
    logits: &Array4<f32>,
    lm_tau: f64,
    logit_step: f64,
    fp_scale: i64,
) -> Result<Vec<f64>> {
    let shape = logits.shape();
    if shape.len() != 4 || shape[0] != 1 {
        bail!("LM logits must have shape [1, card, codebooks, steps], got {:?}", shape);
    }
    let card = shape[1];
    let codebooks = shape[2];
    let steps = shape[3];
    let columns = codebooks * steps;
    let mut pdf = vec![0.0_f64; card * columns];
    let mut quantized = vec![0.0_f64; card];
    let mut probs = vec![0.0_f64; card];
    let uniform = 1.0 / card as f64;
    let near_pdf_threshold = 0.25 / fp_scale as f64;

    for step in 0..steps {
        for codebook in 0..codebooks {
            let mut max_value = f64::NEG_INFINITY;
            let mut min_value = f64::INFINITY;
            for bin in 0..card {
                let raw = logits[[0, bin, codebook, step]] as f64 / lm_tau;
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
                let column = step * codebooks + codebook;
                for bin in 0..card {
                    pdf[bin * columns + column] = uniform;
                }
                continue;
            }
            let mut max_pdf = 0.0_f64;
            let mut min_pdf = f64::INFINITY;
            for prob in &mut probs {
                *prob /= denom;
                max_pdf = max_pdf.max(*prob);
                min_pdf = min_pdf.min(*prob);
            }
            let near_uniform =
                (max_value - min_value) <= (2.0 * logit_step) || (max_pdf - min_pdf) <= near_pdf_threshold;
            let column = step * codebooks + codebook;
            for bin in 0..card {
                pdf[bin * columns + column] = if near_uniform { uniform } else { probs[bin] };
            }
        }
    }

    Ok(pdf)
}

fn quantize_logit(value: f64, step: f64) -> f64 {
    let eps = 2_f64.powi(-40);
    let y = value / step;
    (y + 0.5 - eps).floor() * step
}

fn validate_metadata(bundle_meta: &OnnxFrameBundleMetadata, metadata: &EcdcMetadata) -> Result<()> {
    if metadata.model_name != bundle_meta.model_name {
        bail!(
            "ECDC payload model {} does not match bundle model {}",
            metadata.model_name,
            bundle_meta.model_name
        );
    }
    if metadata.num_codebooks != bundle_meta.num_codebooks {
        bail!(
            "ECDC payload num_codebooks {} does not match bundle {}",
            metadata.num_codebooks,
            bundle_meta.num_codebooks
        );
    }
    match metadata.bitstream_version {
        RAW_BITSTREAM_VERSION => {
            if metadata.use_lm {
                bail!("raw acv=0 payload unexpectedly advertises lm=true");
            }
        }
        DETERMINISTIC_LM_BITSTREAM_VERSION => {
            if !metadata.use_lm {
                bail!("deterministic acv=4 payload unexpectedly advertises lm=false");
            }
        }
        other => bail!("unsupported ECDC bitstream version {other}"),
    }
    Ok(())
}

fn segment_starts(total_samples: usize, stride: usize) -> Vec<usize> {
    let mut starts = Vec::new();
    let mut offset = 0usize;
    while offset < total_samples {
        starts.push(offset);
        offset += stride.max(1);
    }
    starts
}

fn segment_frame_length(samples: usize, segment_samples: usize, frame_length: usize) -> usize {
    (samples * frame_length).div_ceil(segment_samples)
}

fn silence_frame(channels: usize, samples: usize) -> Array3<f32> {
    Array3::<f32>::zeros((1, channels, samples))
}

fn linear_overlap_add(frames: &[Array3<f32>], stride: usize) -> Array3<f32> {
    if frames.is_empty() {
        return Array3::<f32>::zeros((1, 0, 0));
    }

    let channels = frames[0].shape()[1];
    let frame_length = frames[0].shape()[2];
    let total_size = stride * (frames.len() - 1) + frame_length;
    let mut output = Array3::<f32>::zeros((1, channels, total_size));
    let mut sum_weight = vec![0.0_f32; total_size];
    let weight = triangle_weight(frame_length);

    let mut offset = 0usize;
    for frame in frames {
        let frame_len = frame.shape()[2];
        for index in 0..frame_len {
            let w = weight[index];
            sum_weight[offset + index] += w;
            for channel in 0..channels {
                output[[0, channel, offset + index]] += frame[[0, channel, index]] * w;
            }
        }
        offset += stride;
    }

    for index in 0..total_size {
        let denom = sum_weight[index];
        if denom > 0.0 {
            for channel in 0..channels {
                output[[0, channel, index]] /= denom;
            }
        }
    }
    output
}

fn triangle_weight(frame_length: usize) -> Vec<f32> {
    (0..frame_length)
        .map(|index| {
            let t = (index + 1) as f32 / (frame_length + 1) as f32;
            0.5 - (t - 0.5).abs()
        })
        .collect()
}

fn default_fp_scale() -> i64 {
    DEFAULT_FP_SCALE
}

fn default_min_range() -> i64 {
    DEFAULT_MIN_RANGE
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn metadata_serde_roundtrip_keeps_source_fields() {
        let metadata = EcdcMetadata {
            model_name: "encodec_48khz".into(),
            audio_length: 48000,
            num_codebooks: 4,
            use_lm: false,
            fp_scale: DEFAULT_FP_SCALE,
            min_range: DEFAULT_MIN_RANGE,
            bitstream_version: 0,
            lm_tau: None,
            original_sample_rate: Some(44_100),
            original_channels: Some(2),
            original_total_frames: Some(44_100),
            extra: BTreeMap::new(),
        };
        let json = serde_json::to_string(&metadata).unwrap();
        let decoded: EcdcMetadata = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.original_sample_rate, Some(44_100));
        assert_eq!(decoded.original_channels, Some(2));
        assert_eq!(decoded.original_total_frames, Some(44_100));
    }

    #[test]
    fn frame_length_matches_python_formula() {
        assert_eq!(segment_frame_length(48_000, 48_000, 75), 75);
        assert_eq!(segment_frame_length(24_000, 48_000, 75), 38);
        assert_eq!(segment_frame_length(1, 48_000, 75), 1);
    }

    #[test]
    fn teacher_forced_indices_prefix_zero() {
        let mut codes = Array3::<i64>::zeros((1, 2, 3));
        codes[[0, 0, 0]] = 4;
        codes[[0, 1, 0]] = 9;
        codes[[0, 0, 1]] = 7;
        codes[[0, 1, 1]] = 3;
        let teacher = teacher_forced_indices(&codes);
        assert_eq!(teacher[[0, 0, 0]], 0);
        assert_eq!(teacher[[0, 1, 0]], 0);
        assert_eq!(teacher[[0, 0, 1]], 5);
        assert_eq!(teacher[[0, 1, 1]], 10);
        assert_eq!(teacher[[0, 0, 2]], 8);
        assert_eq!(teacher[[0, 1, 2]], 4);
    }
}
