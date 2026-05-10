use std::io::Cursor;

use anyhow::{bail, Result};
use serde::{Deserialize, Serialize};

use crate::binary::{read_ecdc_header, read_exactly, write_ecdc_header, BitPacker, BitUnpacker};
use crate::format::{
    segment_frame_length, segment_starts, validate_metadata, EcdcMetadata, RAW_BITSTREAM_VERSION,
};
use crate::metadata::OnnxFrameBundleMetadata;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RawEcdcFrame {
    pub offset: usize,
    pub samples: usize,
    pub frame_length: usize,
    pub scale: f32,
    pub codes: Vec<u16>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RawEcdcFrames {
    pub metadata: EcdcMetadata,
    pub frames: Vec<RawEcdcFrame>,
}

pub fn encode_raw_header(meta: &OnnxFrameBundleMetadata, audio_length: usize) -> Result<Vec<u8>> {
    let metadata = EcdcMetadata::from_bundle(meta, audio_length, None, false, None, false);
    let mut out = Vec::new();
    write_ecdc_header(&mut out, &metadata)?;
    Ok(out)
}

pub fn encode_raw_frame_payload(
    meta: &OnnxFrameBundleMetadata,
    codes: &[u16],
    scale: f32,
    frame_length: usize,
) -> Result<Vec<u8>> {
    if frame_length == 0 || frame_length > meta.frame_length {
        bail!(
            "frame_length must be in 1..={}, got {}",
            meta.frame_length,
            frame_length
        );
    }
    let compact_len = meta.num_codebooks * frame_length;
    let full_len = meta.num_codebooks * meta.frame_length;
    let bits = validate_raw_codebook_bits(meta)?;
    let max_code = if bits == 16 {
        u16::MAX
    } else {
        ((1_u32 << bits) - 1) as u16
    };
    let codebook_stride = match codes.len() {
        len if len == compact_len => frame_length,
        len if len == full_len => meta.frame_length,
        len => {
            bail!(
                "codes length {} does not match compact {} or full {} layout",
                len,
                compact_len,
                full_len
            );
        }
    };

    let mut out = Vec::new();
    if meta.normalize {
        out.extend_from_slice(&scale.to_be_bytes());
    }
    let mut packer = BitPacker::new(bits);
    for t in 0..frame_length {
        for codebook in 0..meta.num_codebooks {
            let value = codes[codebook * codebook_stride + t];
            if value > max_code {
                bail!("code value {} exceeds {}-bit codebook range", value, bits);
            }
            packer.push(value);
        }
    }
    out.extend_from_slice(&packer.finish());
    Ok(out)
}

pub fn encode_raw_ecdc(
    meta: &OnnxFrameBundleMetadata,
    audio_length: usize,
    frames: &[RawEcdcFrame],
) -> Result<Vec<u8>> {
    let expected = segment_starts(audio_length, meta.segment_stride.max(1));
    if frames.len() != expected.len() {
        bail!(
            "frame count {} does not match audio length {} expectation {}",
            frames.len(),
            audio_length,
            expected.len()
        );
    }

    let mut out = encode_raw_header(meta, audio_length)?;
    for (index, frame) in frames.iter().enumerate() {
        if frame.offset != expected[index] {
            bail!(
                "frame {} offset {} does not match expected {}",
                index,
                frame.offset,
                expected[index]
            );
        }
        let samples = (audio_length - frame.offset).min(meta.segment_samples);
        let frame_length = segment_frame_length(samples, meta.segment_samples, meta.frame_length);
        if frame.samples != samples || frame.frame_length != frame_length {
            bail!(
                "frame {} shape metadata mismatch, expected samples={} frame_length={}, got samples={} frame_length={}",
                index,
                samples,
                frame_length,
                frame.samples,
                frame.frame_length
            );
        }
        let payload =
            encode_raw_frame_payload(meta, &frame.codes, frame.scale, frame.frame_length)?;
        out.extend_from_slice(&payload);
    }
    Ok(out)
}

pub fn decode_raw_frames(meta: &OnnxFrameBundleMetadata, payload: &[u8]) -> Result<RawEcdcFrames> {
    let mut reader = Cursor::new(payload);
    let metadata: EcdcMetadata = read_ecdc_header(&mut reader)?;
    validate_metadata(meta, &metadata)?;
    if metadata.bitstream_version != RAW_BITSTREAM_VERSION {
        bail!(
            "raw frame extraction only supports acv={}, got {}",
            RAW_BITSTREAM_VERSION,
            metadata.bitstream_version
        );
    }

    let mut frames = Vec::new();
    for offset in segment_starts(metadata.audio_length, meta.segment_stride.max(1)) {
        let samples = (metadata.audio_length - offset).min(meta.segment_samples);
        let frame_length = segment_frame_length(samples, meta.segment_samples, meta.frame_length);
        let frame = read_raw_frame(&mut reader, meta, offset, samples, frame_length)?;
        frames.push(frame);
    }
    if reader.position() as usize != payload.len() {
        bail!("raw ECDC payload has trailing bytes after expected frames");
    }
    Ok(RawEcdcFrames { metadata, frames })
}

pub fn raw_segment_count(meta: &OnnxFrameBundleMetadata, audio_length: usize) -> usize {
    segment_starts(audio_length, meta.segment_stride.max(1)).len()
}

pub fn overlap_add_decoded_frames(
    meta: &OnnxFrameBundleMetadata,
    audio_length: usize,
    decoded_frames: &[f32],
) -> Result<Vec<f32>> {
    let frame_count = raw_segment_count(meta, audio_length);
    let expected = frame_count * meta.channels * meta.segment_samples;
    if decoded_frames.len() != expected {
        bail!(
            "decoded frame sample count {} does not match expected {} for {} frames",
            decoded_frames.len(),
            expected,
            frame_count
        );
    }
    if frame_count == 0 {
        return Ok(Vec::new());
    }

    let stride = meta.segment_stride.max(1);
    let total_size = stride * (frame_count - 1) + meta.segment_samples;
    let mut output = vec![0.0_f32; meta.channels * total_size];
    let mut sum_weight = vec![0.0_f32; total_size];
    let weight = triangle_weight(meta.segment_samples);

    for frame in 0..frame_count {
        let offset = frame * stride;
        for sample in 0..meta.segment_samples {
            let w = weight[sample];
            sum_weight[offset + sample] += w;
            for channel in 0..meta.channels {
                let source_index =
                    (frame * meta.channels + channel) * meta.segment_samples + sample;
                let target_index = channel * total_size + offset + sample;
                output[target_index] += decoded_frames[source_index] * w;
            }
        }
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

fn read_raw_frame(
    reader: &mut Cursor<&[u8]>,
    meta: &OnnxFrameBundleMetadata,
    offset: usize,
    samples: usize,
    frame_length: usize,
) -> Result<RawEcdcFrame> {
    let scale = if meta.normalize {
        let bytes = read_exactly(reader, 4)?;
        f32::from_be_bytes(bytes.try_into().expect("slice length"))
    } else {
        1.0
    };

    let bits = validate_raw_codebook_bits(meta)?;
    let bit_count = frame_length * meta.num_codebooks * bits as usize;
    let byte_len = bit_count.div_ceil(8);
    let packed = read_exactly(reader, byte_len)?;
    let mut unpacker = BitUnpacker::new(bits, packed);
    let mut codes = vec![0_u16; meta.num_codebooks * meta.frame_length];
    for t in 0..frame_length {
        for codebook in 0..meta.num_codebooks {
            let value = unpacker
                .pull()
                .ok_or_else(|| anyhow::anyhow!("raw ECDC stream ended before expected codes"))?;
            codes[codebook * meta.frame_length + t] = value;
        }
    }

    Ok(RawEcdcFrame {
        offset,
        samples,
        frame_length,
        scale,
        codes,
    })
}

fn validate_raw_codebook_bits(meta: &OnnxFrameBundleMetadata) -> Result<u8> {
    let bits = meta.bits_per_codebook();
    if bits == 0 || bits > 16 {
        bail!("raw wasm ECDC supports 1..=16 bits per codebook, got {bits}");
    }
    Ok(bits)
}

fn triangle_weight(frame_length: usize) -> Vec<f32> {
    (0..frame_length)
        .map(|index| {
            let t = (index + 1) as f32 / (frame_length + 1) as f32;
            0.5 - (t - 0.5).abs()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_meta() -> OnnxFrameBundleMetadata {
        OnnxFrameBundleMetadata {
            schema_version: 1,
            model_name: "encodec_48khz".into(),
            bandwidth_kbps: 6.0,
            sample_rate: 48_000,
            channels: 2,
            segment_samples: 8,
            segment_stride: 4,
            normalize: true,
            num_codebooks: 2,
            frame_length: 4,
            bits_per_codebook: Some(10),
            codebook_cardinality: Some(1024),
            encode_model: "encode_frame.onnx".into(),
            decode_model: "decode_frame.onnx".into(),
            lm_model: None,
            lm_dim: None,
            lm_num_layers: None,
            lm_past_context: None,
            lm_logit_step: None,
            lm_cardinality: None,
            lm_dtype: None,
            opset_version: 17,
        }
    }

    #[test]
    fn raw_ecdc_roundtrip_keeps_codes_and_scale() {
        let meta = test_meta();
        let frames = vec![
            RawEcdcFrame {
                offset: 0,
                samples: 7,
                frame_length: 4,
                scale: 0.5,
                codes: vec![0, 1, 2, 3, 10, 11, 12, 13],
            },
            RawEcdcFrame {
                offset: 4,
                samples: 3,
                frame_length: 2,
                scale: 0.25,
                codes: vec![20, 21, 0, 0, 30, 31, 0, 0],
            },
        ];
        let payload = encode_raw_ecdc(&meta, 7, &frames).unwrap();
        let decoded = decode_raw_frames(&meta, &payload).unwrap();

        assert_eq!(decoded.metadata.audio_length, 7);
        assert_eq!(decoded.frames.len(), 2);
        assert_eq!(decoded.frames[0].scale, 0.5);
        assert_eq!(decoded.frames[0].codes, frames[0].codes);
        assert_eq!(decoded.frames[1].scale, 0.25);
        assert_eq!(decoded.frames[1].frame_length, 2);
        assert_eq!(decoded.frames[1].codes, frames[1].codes);
    }

    #[test]
    fn incremental_raw_ecdc_matches_full_payload() {
        let meta = test_meta();
        let frames = vec![
            RawEcdcFrame {
                offset: 0,
                samples: 7,
                frame_length: 4,
                scale: 0.5,
                codes: vec![0, 1, 2, 3, 10, 11, 12, 13],
            },
            RawEcdcFrame {
                offset: 4,
                samples: 3,
                frame_length: 2,
                scale: 0.25,
                codes: vec![20, 21, 0, 0, 30, 31, 0, 0],
            },
        ];

        let full = encode_raw_ecdc(&meta, 7, &frames).unwrap();
        let mut incremental = encode_raw_header(&meta, 7).unwrap();
        for frame in &frames {
            incremental.extend_from_slice(
                &encode_raw_frame_payload(&meta, &frame.codes, frame.scale, frame.frame_length)
                    .unwrap(),
            );
        }

        assert_eq!(incremental, full);
    }
}
