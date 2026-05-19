use anyhow::Result;
use serde::{Deserialize, Serialize};

pub const Q8_LM_LOGIT_STEP: f64 = 2.1;

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct OnnxFrameBundleMetadata {
    pub schema_version: u32,
    pub model_name: String,
    pub bandwidth_kbps: f32,
    pub sample_rate: usize,
    pub channels: usize,
    pub segment_samples: usize,
    pub segment_stride: usize,
    pub normalize: bool,
    pub num_codebooks: usize,
    pub frame_length: usize,
    #[serde(default)]
    pub bits_per_codebook: Option<u8>,
    #[serde(default)]
    pub codebook_cardinality: Option<usize>,
    pub encode_model: String,
    pub decode_model: String,
    #[serde(default)]
    pub lm_quant_weight_model: Option<String>,
    #[serde(default)]
    pub lm_dim: Option<usize>,
    #[serde(default)]
    pub lm_num_layers: Option<usize>,
    #[serde(default)]
    pub lm_past_context: Option<usize>,
    #[serde(default)]
    pub lm_logit_step: Option<f32>,
    #[serde(default)]
    pub lm_entropy_logit_step: Option<f32>,
    #[serde(default)]
    pub lm_cardinality: Option<usize>,
    pub opset_version: usize,
}

impl OnnxFrameBundleMetadata {
    pub fn bits_per_codebook(&self) -> u8 {
        if let Some(bits) = self.bits_per_codebook {
            return bits;
        }
        let cardinality = self
            .codebook_cardinality
            .or(self.lm_cardinality)
            .unwrap_or(1024);
        cardinality.ilog2() as u8
    }

    pub fn lm_logit_step(&self) -> f64 {
        self.lm_logit_step.unwrap_or(1.0 / 64.0) as f64
    }

    pub fn lm_entropy_logit_step(&self) -> f64 {
        self.lm_logit_step().max(
            self.lm_entropy_logit_step
                .unwrap_or(Q8_LM_LOGIT_STEP as f32) as f64,
        )
    }

    pub fn lm_num_layers(&self) -> Result<usize> {
        self.lm_num_layers
            .ok_or_else(|| anyhow::anyhow!("bundle metadata is missing lm_num_layers"))
    }

    pub fn lm_dim(&self) -> Result<usize> {
        self.lm_dim
            .ok_or_else(|| anyhow::anyhow!("bundle metadata is missing lm_dim"))
    }

    pub fn lm_past_context(&self) -> Result<usize> {
        self.lm_past_context
            .ok_or_else(|| anyhow::anyhow!("bundle metadata is missing lm_past_context"))
    }

    pub fn lm_cardinality(&self) -> usize {
        self.lm_cardinality
            .or(self.codebook_cardinality)
            .unwrap_or(1024)
    }
}
