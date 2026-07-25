use anyhow::{bail, Result};
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
    pub fn validate(&self) -> Result<()> {
        if self.schema_version != 1 {
            bail!("unsupported bundle schema_version {}", self.schema_version);
        }
        if self.model_name.is_empty() {
            bail!("bundle model_name must not be empty");
        }
        if !self.bandwidth_kbps.is_finite() || self.bandwidth_kbps <= 0.0 {
            bail!("bundle bandwidth_kbps must be a positive finite value");
        }
        if self.sample_rate == 0 || self.channels == 0 {
            bail!("bundle sample rate and channel count must be non-zero");
        }
        if self.segment_samples == 0 || self.segment_stride == 0 || self.frame_length == 0 {
            bail!("bundle segment and frame dimensions must be non-zero");
        }
        if self.segment_stride > self.segment_samples {
            bail!(
                "bundle segment_stride {} exceeds segment_samples {}",
                self.segment_stride,
                self.segment_samples,
            );
        }
        if self.num_codebooks == 0 {
            bail!("bundle num_codebooks must be non-zero");
        }
        if self.encode_model.is_empty() || self.decode_model.is_empty() {
            bail!("bundle encoder and decoder model names must not be empty");
        }
        if self.opset_version == 0 {
            bail!("bundle opset_version must be non-zero");
        }

        let cardinality = self
            .codebook_cardinality
            .or(self.lm_cardinality)
            .unwrap_or(1024);
        if cardinality < 2 || cardinality > u16::MAX as usize + 1 {
            bail!("bundle codebook cardinality {cardinality} is outside 2..=65536");
        }
        if let (Some(codebook), Some(lm)) = (self.codebook_cardinality, self.lm_cardinality) {
            if codebook != lm {
                bail!(
                    "bundle codebook cardinality {} does not match LM cardinality {}",
                    codebook,
                    lm,
                );
            }
        }
        let bits = self.bits_per_codebook();
        if bits == 0 || bits > 16 || (1_usize << bits) < cardinality {
            bail!(
                "bundle bits_per_codebook {} cannot represent cardinality {}",
                bits,
                cardinality,
            );
        }
        Ok(())
    }

    pub fn validate_lm(&self) -> Result<()> {
        self.validate()?;
        if self.lm_dim()? == 0 {
            bail!("bundle lm_dim must be non-zero");
        }
        if self.lm_num_layers()? == 0 {
            bail!("bundle lm_num_layers must be non-zero");
        }
        self.lm_past_context()?;
        let weight_model = self
            .lm_quant_weight_model
            .as_deref()
            .ok_or_else(|| anyhow::anyhow!("bundle is missing lm_quant_weight_model"))?;
        if weight_model.is_empty() {
            bail!("bundle lm_quant_weight_model must not be empty");
        }
        if let Some(logit_step) = self.lm_logit_step {
            if !logit_step.is_finite() || logit_step <= 0.0 {
                bail!("bundle lm_logit_step must be a positive finite value");
            }
        }
        if let Some(logit_step) = self.lm_entropy_logit_step {
            if !logit_step.is_finite() || logit_step <= 0.0 {
                bail!("bundle lm_entropy_logit_step must be a positive finite value");
            }
        }
        let logit_step = self.lm_entropy_logit_step();
        if !logit_step.is_finite() || logit_step <= 0.0 {
            bail!("bundle LM entropy logit step must be a positive finite value");
        }
        Ok(())
    }

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
