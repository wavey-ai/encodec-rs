use std::collections::BTreeMap;

use anyhow::{bail, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::metadata::OnnxFrameBundleMetadata;

pub const DEFAULT_FP_SCALE: i64 = 1 << 13;
pub const DEFAULT_MIN_RANGE: i64 = 2;
pub const QUANTIZED_LM_BITSTREAM_VERSION: u8 = 2;
pub const ARITHMETIC_TOTAL_RANGE_BITS: u32 = 24;

#[derive(Debug, Clone, Default)]
pub struct SourceAudioMetadata {
    pub sample_rate: Option<u32>,
    pub channels: Option<u16>,
    pub total_frames: Option<usize>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EcdcChunkLayout {
    pub samples: usize,
    pub stride: usize,
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
    #[serde(rename = "lmh", skip_serializing_if = "Option::is_none")]
    pub lm_hash: Option<String>,
    #[serde(rename = "cs", skip_serializing_if = "Option::is_none")]
    pub chunk_samples: Option<usize>,
    #[serde(rename = "cst", skip_serializing_if = "Option::is_none")]
    pub chunk_stride: Option<usize>,
    #[serde(rename = "fl", skip_serializing_if = "Option::is_none")]
    pub lm_frame_length: Option<usize>,
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

impl EcdcMetadata {
    pub fn from_bundle(
        bundle: &OnnxFrameBundleMetadata,
        audio_length: usize,
        _source: Option<&SourceAudioMetadata>,
        lm_hash: Option<String>,
    ) -> Self {
        Self {
            model_name: bundle.model_name.clone(),
            audio_length,
            num_codebooks: bundle.num_codebooks,
            use_lm: true,
            fp_scale: DEFAULT_FP_SCALE,
            min_range: DEFAULT_MIN_RANGE,
            bitstream_version: QUANTIZED_LM_BITSTREAM_VERSION,
            lm_tau: Some(1.0),
            lm_hash,
            chunk_samples: None,
            chunk_stride: None,
            lm_frame_length: None,
            extra: BTreeMap::new(),
        }
    }
}

pub fn validate_metadata(
    bundle_meta: &OnnxFrameBundleMetadata,
    metadata: &EcdcMetadata,
) -> Result<()> {
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
    if metadata.bitstream_version != QUANTIZED_LM_BITSTREAM_VERSION {
        bail!(
            "unsupported ECDC bitstream version {}; only q8 acv={} is supported",
            metadata.bitstream_version,
            QUANTIZED_LM_BITSTREAM_VERSION
        );
    }
    if !metadata.use_lm {
        bail!("q8 ECDC payload unexpectedly advertises lm=false");
    }
    if let Some(lm_frame_length) = metadata.lm_frame_length {
        if lm_frame_length == 0 || lm_frame_length > bundle_meta.frame_length {
            bail!(
                "ECDC LM frame length {} is out of range for bundle frame length {}",
                lm_frame_length,
                bundle_meta.frame_length
            );
        }
    }
    Ok(())
}

pub fn ecdc_chunk_layout_from_ms(
    bundle_meta: &OnnxFrameBundleMetadata,
    chunk_ms: Option<f64>,
) -> Result<EcdcChunkLayout> {
    match chunk_ms {
        None => Ok(EcdcChunkLayout {
            samples: bundle_meta.segment_samples,
            stride: bundle_meta.segment_stride.max(1),
        }),
        Some(ms) => {
            if !ms.is_finite() || ms <= 0.0 {
                bail!("chunk_ms must be a positive finite value");
            }

            let samples_f64 = ms * bundle_meta.sample_rate as f64 / 1000.0;
            if !samples_f64.is_finite() || samples_f64 < 1.0 || samples_f64 > usize::MAX as f64 {
                bail!("chunk_ms {ms} cannot be represented as a valid PCM sample count");
            }

            let samples = samples_f64.round() as usize;
            if samples == 0 {
                bail!("chunk_ms {ms} rounds to zero PCM samples");
            }

            Ok(EcdcChunkLayout {
                samples,
                stride: samples,
            })
        }
    }
}

pub fn ecdc_chunk_layout_from_metadata(
    bundle_meta: &OnnxFrameBundleMetadata,
    metadata: &EcdcMetadata,
) -> Result<EcdcChunkLayout> {
    let samples = metadata
        .chunk_samples
        .unwrap_or(bundle_meta.segment_samples);
    let stride = metadata
        .chunk_stride
        .unwrap_or(bundle_meta.segment_stride.max(1));

    if samples == 0 {
        bail!("ECDC metadata has invalid chunk sample count 0");
    }
    if stride == 0 {
        bail!("ECDC metadata has invalid chunk stride 0");
    }

    Ok(EcdcChunkLayout { samples, stride })
}

pub fn ecdc_chunk_layout_for_chunk_count(
    bundle_meta: &OnnxFrameBundleMetadata,
    metadata: &EcdcMetadata,
    chunk_count: usize,
) -> Result<EcdcChunkLayout> {
    let explicit = metadata.chunk_samples.is_some() || metadata.chunk_stride.is_some();
    let default = ecdc_chunk_layout_from_metadata(bundle_meta, metadata)?;
    if explicit {
        return Ok(default);
    }

    let mut candidates = vec![default];
    if bundle_meta.sample_rate == 48_000 {
        candidates.push(EcdcChunkLayout {
            samples: 63_998,
            stride: 63_998,
        });
        candidates.push(EcdcChunkLayout {
            samples: 86_400,
            stride: 86_400,
        });
    }

    for layout in candidates {
        if segment_starts(metadata.audio_length, layout.stride).len() == chunk_count {
            return Ok(layout);
        }
    }

    anyhow::bail!(
        "ECDC payload has {chunk_count} chunks, but metadata implies {} chunks",
        segment_starts(metadata.audio_length, default.stride).len()
    );
}

pub fn segment_starts(total_samples: usize, stride: usize) -> Vec<usize> {
    let mut starts = Vec::new();
    let mut offset = 0usize;
    while offset < total_samples {
        starts.push(offset);
        offset += stride.max(1);
    }
    starts
}

pub fn segment_frame_length(samples: usize, segment_samples: usize, frame_length: usize) -> usize {
    (samples * frame_length).div_ceil(segment_samples)
}

pub fn ecdc_lm_frame_length(
    metadata: &EcdcMetadata,
    samples: usize,
    segment_samples: usize,
    frame_length: usize,
) -> usize {
    metadata
        .lm_frame_length
        .filter(|value| *value > 0)
        .unwrap_or_else(|| segment_frame_length(samples, segment_samples, frame_length))
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
    fn metadata_serde_roundtrip_preserves_unknown_source_fields() {
        let metadata = EcdcMetadata {
            model_name: "encodec_48khz".into(),
            audio_length: 48000,
            num_codebooks: 4,
            use_lm: false,
            fp_scale: DEFAULT_FP_SCALE,
            min_range: DEFAULT_MIN_RANGE,
            bitstream_version: QUANTIZED_LM_BITSTREAM_VERSION,
            lm_tau: None,
            lm_hash: None,
            chunk_samples: None,
            chunk_stride: None,
            lm_frame_length: None,
            extra: BTreeMap::new(),
        };
        let json = serde_json::to_string(&metadata).unwrap();
        let source_json =
            json.trim_end_matches('}').to_owned() + ",\"osr\":44100,\"och\":2,\"ofr\":44100}";
        let decoded: EcdcMetadata = serde_json::from_str(&source_json).unwrap();
        assert_eq!(
            decoded.extra.get("osr").and_then(Value::as_u64),
            Some(44_100)
        );
        assert_eq!(decoded.extra.get("och").and_then(Value::as_u64), Some(2));
        assert_eq!(
            decoded.extra.get("ofr").and_then(Value::as_u64),
            Some(44_100)
        );
    }

    #[test]
    fn metadata_version_is_q8_only() {
        let bundle = OnnxFrameBundleMetadata {
            schema_version: 1,
            model_name: "encodec_48khz".into(),
            bandwidth_kbps: 12.0,
            sample_rate: 48_000,
            channels: 2,
            segment_samples: 48_000,
            segment_stride: 47_040,
            normalize: true,
            num_codebooks: 8,
            frame_length: 150,
            bits_per_codebook: Some(10),
            codebook_cardinality: Some(1024),
            encode_model: "encode_frame.onnx".into(),
            decode_model: "decode_frame.onnx".into(),
            lm_quant_weight_model: Some("lm_weights_q8.bin".into()),
            lm_dim: Some(128),
            lm_num_layers: Some(1),
            lm_past_context: Some(0),
            lm_logit_step: Some(1.0 / 64.0),
            lm_entropy_logit_step: Some(2.1),
            lm_cardinality: Some(1024),
            opset_version: 17,
        };

        let lm = EcdcMetadata::from_bundle(&bundle, 48_000, None, Some("hash".into()));
        assert_eq!(lm.bitstream_version, QUANTIZED_LM_BITSTREAM_VERSION);
        validate_metadata(&bundle, &lm).unwrap();

        let mut unsupported = lm.clone();
        unsupported.bitstream_version = 1;
        assert!(validate_metadata(&bundle, &unsupported).is_err());
    }

    #[test]
    fn frame_length_matches_python_formula() {
        assert_eq!(segment_frame_length(48_000, 48_000, 75), 75);
        assert_eq!(segment_frame_length(24_000, 48_000, 75), 38);
        assert_eq!(segment_frame_length(1, 48_000, 75), 1);
    }

    #[test]
    fn chunk_ms_uses_sample_rate_and_rounds_to_samples() {
        let bundle = OnnxFrameBundleMetadata {
            schema_version: 1,
            model_name: "encodec_48khz".into(),
            bandwidth_kbps: 12.0,
            sample_rate: 48_000,
            channels: 2,
            segment_samples: 48_000,
            segment_stride: 47_040,
            normalize: true,
            num_codebooks: 8,
            frame_length: 150,
            bits_per_codebook: Some(10),
            codebook_cardinality: Some(1024),
            encode_model: "encode_frame.onnx".into(),
            decode_model: "decode_frame.onnx".into(),
            lm_quant_weight_model: Some("lm_weights_q8.bin".into()),
            lm_dim: Some(128),
            lm_num_layers: Some(1),
            lm_past_context: Some(0),
            lm_logit_step: Some(1.0 / 64.0),
            lm_entropy_logit_step: Some(2.1),
            lm_cardinality: Some(1024),
            opset_version: 17,
        };

        let default_layout = ecdc_chunk_layout_from_ms(&bundle, None).unwrap();
        assert_eq!(default_layout.samples, 48_000);
        assert_eq!(default_layout.stride, 47_040);

        let custom_layout = ecdc_chunk_layout_from_ms(&bundle, Some(250.0)).unwrap();
        assert_eq!(custom_layout.samples, 12_000);
        assert_eq!(custom_layout.stride, 12_000);
    }
}
