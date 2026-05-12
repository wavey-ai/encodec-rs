use std::collections::BTreeMap;

use anyhow::{bail, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::metadata::OnnxFrameBundleMetadata;

pub const DEFAULT_FP_SCALE: i64 = 1 << 13;
pub const DEFAULT_MIN_RANGE: i64 = 2;
pub const RAW_BITSTREAM_VERSION: u8 = 0;
pub const PORTABLE_LM_BITSTREAM_VERSION: u8 = 1;
pub const DETERMINISTIC_LM_BITSTREAM_VERSION: u8 = PORTABLE_LM_BITSTREAM_VERSION;
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
    #[serde(rename = "cc", skip_serializing_if = "Option::is_none")]
    pub chunk_crc: Option<bool>,
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

impl EcdcMetadata {
    pub fn from_bundle(
        bundle: &OnnxFrameBundleMetadata,
        audio_length: usize,
        _source: Option<&SourceAudioMetadata>,
        use_lm: bool,
        lm_tau: Option<f32>,
        chunk_crc: bool,
    ) -> Self {
        Self {
            model_name: bundle.model_name.clone(),
            audio_length,
            num_codebooks: bundle.num_codebooks,
            use_lm,
            fp_scale: DEFAULT_FP_SCALE,
            min_range: DEFAULT_MIN_RANGE,
            bitstream_version: if use_lm {
                PORTABLE_LM_BITSTREAM_VERSION
            } else {
                RAW_BITSTREAM_VERSION
            },
            lm_tau,
            chunk_crc: if use_lm && !chunk_crc {
                Some(false)
            } else {
                None
            },
            extra: BTreeMap::new(),
        }
    }

    pub fn chunk_crc_enabled(&self) -> bool {
        self.chunk_crc.unwrap_or(true)
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
    let bitstream_version = metadata.bitstream_version;
    match bitstream_version {
        RAW_BITSTREAM_VERSION => {
            if metadata.use_lm {
                bail!("raw acv=0 payload unexpectedly advertises lm=true");
            }
        }
        PORTABLE_LM_BITSTREAM_VERSION => {
            if !metadata.use_lm {
                bail!(
                    "deterministic acv={} payload unexpectedly advertises lm=false",
                    bitstream_version
                );
            }
        }
        other => bail!("unsupported ECDC bitstream version {other}"),
    }
    Ok(())
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
            bitstream_version: 0,
            lm_tau: None,
            chunk_crc: None,
            extra: BTreeMap::new(),
        };
        let json = serde_json::to_string(&metadata).unwrap();
        let source_json =
            json.trim_end_matches('}').to_owned() + ",\"osr\":44100,\"och\":2,\"ofr\":44100}";
        let decoded: EcdcMetadata = serde_json::from_str(&source_json).unwrap();
        assert!(decoded.chunk_crc_enabled());
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
    fn metadata_versions_are_raw_zero_and_lm_one() {
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
            lm_model: Some("lm_logits.onnx".into()),
            lm_weight_model: Some("lm_weights.bin".into()),
            lm_dim: Some(128),
            lm_num_layers: Some(1),
            lm_past_context: Some(0),
            lm_logit_step: Some(1.0 / 64.0),
            lm_portable_logit_step: Some(2.1),
            lm_cardinality: Some(1024),
            lm_dtype: Some("float32".into()),
            opset_version: 17,
        };

        let raw = EcdcMetadata::from_bundle(&bundle, 48_000, None, false, None, false);
        assert_eq!(raw.bitstream_version, RAW_BITSTREAM_VERSION);
        validate_metadata(&bundle, &raw).unwrap();

        let lm = EcdcMetadata::from_bundle(&bundle, 48_000, None, true, Some(1.0), false);
        assert_eq!(lm.bitstream_version, PORTABLE_LM_BITSTREAM_VERSION);
        validate_metadata(&bundle, &lm).unwrap();

        let mut unsupported = lm.clone();
        unsupported.bitstream_version = 5;
        assert!(validate_metadata(&bundle, &unsupported).is_err());
    }

    #[test]
    fn frame_length_matches_python_formula() {
        assert_eq!(segment_frame_length(48_000, 48_000, 75), 75);
        assert_eq!(segment_frame_length(24_000, 48_000, 75), 38);
        assert_eq!(segment_frame_length(1, 48_000, 75), 1);
    }
}
