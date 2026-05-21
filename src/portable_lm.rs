use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use ndarray::{Array3, Array4};

use crate::ecdc::{LmCodec, QUANTIZED_LM_BITSTREAM_VERSION};
use crate::metadata::OnnxFrameBundleMetadata;
use crate::quantized_lm::{QuantizedLm, QuantizedLmState, QuantizedLmWeights};
use crate::stable_hash::stable_hash_hex;

pub struct PortableLmCodec {
    bundle_dir: Option<PathBuf>,
    metadata: OnnxFrameBundleMetadata,
    backend: PortableLmBackend,
}

enum PortableLmBackend {
    Q8 {
        lm: QuantizedLm,
        state: Option<QuantizedLmState>,
        hash: String,
    },
}

impl PortableLmCodec {
    pub fn from_dir(dir: impl AsRef<Path>) -> Result<Self> {
        let bundle_dir = dir.as_ref().to_path_buf();
        let metadata_path = bundle_dir.join("bundle.json");
        let metadata: OnnxFrameBundleMetadata = serde_json::from_str(
            &fs::read_to_string(&metadata_path)
                .with_context(|| format!("failed to read {}", metadata_path.display()))?,
        )
        .with_context(|| format!("failed to parse {}", metadata_path.display()))?;

        let weight_model = metadata
            .lm_quant_weight_model
            .clone()
            .ok_or_else(|| anyhow::anyhow!("bundle does not include lm_quant_weight_model"))?;
        let weights_path = bundle_dir.join(weight_model);
        let weights = fs::read(&weights_path)
            .with_context(|| format!("failed to read {}", weights_path.display()))?;
        let mut codec = Self::from_quantized_weights(metadata, &weights)?;
        codec.bundle_dir = Some(bundle_dir);
        Ok(codec)
    }

    pub fn from_quantized_weights(
        metadata: OnnxFrameBundleMetadata,
        weights: &[u8],
    ) -> Result<Self> {
        let hash = stable_hash_hex(weights);
        if metadata.schema_version != 1 {
            bail!(
                "unsupported bundle schema_version {}",
                metadata.schema_version
            );
        }
        metadata.lm_dim()?;
        metadata.lm_num_layers()?;
        metadata.lm_past_context()?;

        let weights = QuantizedLmWeights::from_bytes(weights)
            .context("failed to parse quantized LM weights")?;
        weights.validate_for_codebooks(metadata.num_codebooks)?;
        Ok(Self {
            bundle_dir: None,
            metadata,
            backend: PortableLmBackend::Q8 {
                lm: QuantizedLm::new(weights),
                state: None,
                hash,
            },
        })
    }

    pub fn bundle_dir(&self) -> Option<&Path> {
        self.bundle_dir.as_deref()
    }

    pub fn metadata(&self) -> &OnnxFrameBundleMetadata {
        &self.metadata
    }
}

impl LmCodec for PortableLmCodec {
    fn metadata(&self) -> &OnnxFrameBundleMetadata {
        &self.metadata
    }

    fn bitstream_version(&self) -> u8 {
        QUANTIZED_LM_BITSTREAM_VERSION
    }

    fn bitstream_lm_hash(&self) -> Option<&str> {
        match &self.backend {
            PortableLmBackend::Q8 { hash, .. } => Some(hash),
        }
    }

    fn initial_states(&self, batch: usize) -> Result<Vec<Array3<f32>>> {
        let dim = self.metadata.lm_dim()?;
        let layers = self.metadata.lm_num_layers()?;
        Ok((0..layers)
            .map(|_| Array3::<f32>::zeros((batch, 1, dim)))
            .collect())
    }

    fn forward_logits(
        &mut self,
        indices: &Array3<i64>,
        offset: i64,
        _states: &[Array3<f32>],
    ) -> Result<(Array4<f32>, i64, Vec<Array3<f32>>)> {
        let shape = indices.shape();
        if shape.len() != 3 {
            bail!("LM indices must have shape [batch, codebooks, steps]");
        }
        if shape[0] != 1 || shape[2] != 1 {
            bail!("q8 LM only supports shape [1, codebooks, 1]");
        }
        if shape[1] > self.metadata.num_codebooks {
            bail!(
                "LM indices use {} codebooks, but bundle only supports {}",
                shape[1],
                self.metadata.num_codebooks
            );
        }
        let mut input_symbols = Vec::with_capacity(shape[1]);
        for codebook in 0..shape[1] {
            let value = indices[[0, codebook, 0]];
            if value < 0 {
                bail!("LM input symbol must be non-negative, got {value}");
            }
            input_symbols.push(value as usize);
        }
        let logits = match &mut self.backend {
            PortableLmBackend::Q8 { lm, state, .. } => {
                if offset == 0 || state.is_none() {
                    *state = Some(lm.initial_state());
                }
                let state = state.as_mut().expect("state initialized");
                lm.forward_step(state, &input_symbols)?
            }
        };
        let card = self.metadata.lm_cardinality();
        let codebooks = self.metadata.num_codebooks;
        let logits = Array4::from_shape_vec((1, card, codebooks, 1), logits).expect("shape");
        Ok((logits, offset + 1, self.initial_states(shape[0])?))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    #[test]
    fn downloaded_lm_weights_run_without_onnx() -> Result<()> {
        let bundle_dir = Path::new("onnx-bundles/encodec_48khz_6kbps");
        if !bundle_dir.exists() {
            eprintln!("skipping LM fixture test; run scripts/download-onnx-bundles.sh first");
            return Ok(());
        }

        let mut codec = PortableLmCodec::from_dir(bundle_dir)?;
        let meta = codec.metadata().clone();
        let states = codec.initial_states(1)?;
        let indices = Array3::<i64>::zeros((1, meta.num_codebooks, 1));
        let (logits, next_offset, next_states) = codec.forward_logits(&indices, 0, &states)?;

        assert_eq!(
            logits.shape(),
            &[1, meta.lm_cardinality(), meta.num_codebooks, 1]
        );
        assert_eq!(next_offset, 1);
        assert_eq!(next_states.len(), meta.lm_num_layers()?);
        Ok(())
    }
}
