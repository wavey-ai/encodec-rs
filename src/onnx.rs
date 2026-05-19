use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use gpu_worker_ort::{
    build_session_from_target, default_intra_threads, ort_error, GraphOptimizationLevel, LogLevel,
    OrtTensor, Session, SessionConfig,
};
pub use gpu_worker_ort::{CoreMlComputeUnits, ExecutionTarget};
use ndarray::{Array2, Array3, Array4, Ix2, Ix3};

use crate::ecdc::{FrameCodec, LmCodec, QUANTIZED_LM_BITSTREAM_VERSION};
pub use crate::metadata::OnnxFrameBundleMetadata;
use crate::quantized_lm::{QuantizedLm, QuantizedLmState, QuantizedLmWeights};
use crate::stable_hash::stable_hash_hex;

fn ort_session_config() -> SessionConfig {
    let intra_threads = std::env::var("ENCODEC_RS_ORT_THREADS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or_else(|| default_intra_threads(4));
    SessionConfig::new(
        GraphOptimizationLevel::Level3,
        LogLevel::Warning,
        intra_threads,
    )
}

pub struct OnnxFrameCodec {
    bundle_dir: PathBuf,
    metadata: OnnxFrameBundleMetadata,
    encoder: Session,
    decoder: Session,
}

impl OnnxFrameCodec {
    pub fn from_dir(dir: impl AsRef<Path>, target: ExecutionTarget) -> Result<Self> {
        let bundle_dir = dir.as_ref().to_path_buf();
        let metadata_path = bundle_dir.join("bundle.json");
        let metadata: OnnxFrameBundleMetadata = serde_json::from_str(
            &fs::read_to_string(&metadata_path)
                .with_context(|| format!("failed to read {}", metadata_path.display()))?,
        )
        .with_context(|| format!("failed to parse {}", metadata_path.display()))?;

        if metadata.schema_version != 1 {
            bail!(
                "unsupported bundle schema_version {}",
                metadata.schema_version
            );
        }

        let encoder_path = bundle_dir.join(&metadata.encode_model);
        let decoder_path = bundle_dir.join(&metadata.decode_model);
        if !encoder_path.exists() || !decoder_path.exists() {
            bail!(
                "bundle is missing required model file(s): {} {}",
                encoder_path.display(),
                decoder_path.display()
            );
        }

        let session_cfg = ort_session_config();
        let (encoder, decoder) = match target {
            ExecutionTarget::CoreMl {
                compute_units,
                model_cache_dir,
                low_precision_accumulation_on_gpu,
            } => {
                let encoder_target = ExecutionTarget::CoreMl {
                    compute_units,
                    model_cache_dir: model_cache_dir
                        .as_ref()
                        .map(|path| path.join("encode_frame")),
                    low_precision_accumulation_on_gpu,
                };
                let decoder_target = ExecutionTarget::CoreMl {
                    compute_units,
                    model_cache_dir: model_cache_dir
                        .as_ref()
                        .map(|path| path.join("decode_frame")),
                    low_precision_accumulation_on_gpu,
                };
                (
                    build_session_from_target(&encoder_path, &encoder_target, &session_cfg, true)?,
                    build_session_from_target(&decoder_path, &decoder_target, &session_cfg, true)?,
                )
            }
            other => (
                build_session_from_target(&encoder_path, &other, &session_cfg, true)?,
                build_session_from_target(&decoder_path, &other, &session_cfg, true)?,
            ),
        };

        Ok(Self {
            bundle_dir,
            metadata,
            encoder,
            decoder,
        })
    }

    pub fn bundle_dir(&self) -> &Path {
        &self.bundle_dir
    }

    pub fn metadata(&self) -> &OnnxFrameBundleMetadata {
        &self.metadata
    }

    pub fn encode_frame(&mut self, audio: &Array3<f32>) -> Result<(Array3<i64>, Array2<f32>)> {
        let shape = audio.shape();
        if shape.len() != 3 {
            bail!("audio must have shape [batch, channels, samples]");
        }
        if shape[1] != self.metadata.channels || shape[2] != self.metadata.segment_samples {
            bail!(
                "audio shape mismatch, expected [batch, {}, {}], got {:?}",
                self.metadata.channels,
                self.metadata.segment_samples,
                shape
            );
        }

        let tensor = OrtTensor::from_array(audio.to_owned()).map_err(ort_error)?;
        let outputs = self.encoder.run([tensor.into()]).map_err(ort_error)?;
        if outputs.len() < 2 {
            bail!("encoder output count {} too small", outputs.len());
        }
        let codes = outputs[0]
            .try_extract_array::<i64>()
            .map_err(ort_error)?
            .to_owned()
            .into_dimensionality::<Ix3>()
            .map_err(ort_error)?;
        let scale = outputs[1]
            .try_extract_array::<f32>()
            .map_err(ort_error)?
            .to_owned()
            .into_dimensionality::<Ix2>()
            .map_err(ort_error)?;
        Ok((codes, scale))
    }

    pub fn decode_frame(
        &mut self,
        codes: &Array3<i64>,
        scale: &Array2<f32>,
    ) -> Result<Array3<f32>> {
        let code_shape = codes.shape();
        if code_shape.len() != 3 {
            bail!("codes must have shape [batch, num_codebooks, frame_length]");
        }
        if code_shape[1] != self.metadata.num_codebooks
            || code_shape[2] != self.metadata.frame_length
        {
            bail!(
                "codes shape mismatch, expected [batch, {}, {}], got {:?}",
                self.metadata.num_codebooks,
                self.metadata.frame_length,
                code_shape
            );
        }
        let scale_shape = scale.shape();
        if scale_shape.len() != 2 || scale_shape[0] != code_shape[0] || scale_shape[1] != 1 {
            bail!("scale must have shape [batch, 1], got {:?}", scale_shape);
        }

        let code_tensor = OrtTensor::from_array(codes.to_owned()).map_err(ort_error)?;
        let scale_tensor = OrtTensor::from_array(scale.to_owned()).map_err(ort_error)?;
        let outputs = self
            .decoder
            .run([code_tensor.into(), scale_tensor.into()])
            .map_err(ort_error)?;
        if outputs.len() == 0 {
            bail!("decoder returned no outputs");
        }
        outputs[0]
            .try_extract_array::<f32>()
            .map_err(ort_error)?
            .to_owned()
            .into_dimensionality::<Ix3>()
            .map_err(ort_error)
    }
}

impl FrameCodec for OnnxFrameCodec {
    fn metadata(&self) -> &OnnxFrameBundleMetadata {
        OnnxFrameCodec::metadata(self)
    }

    fn encode_frame(&mut self, audio: &Array3<f32>) -> Result<(Array3<i64>, Array2<f32>)> {
        OnnxFrameCodec::encode_frame(self, audio)
    }

    fn decode_frame(&mut self, codes: &Array3<i64>, scale: &Array2<f32>) -> Result<Array3<f32>> {
        OnnxFrameCodec::decode_frame(self, codes, scale)
    }
}

pub struct OnnxLmCodec {
    bundle_dir: PathBuf,
    metadata: OnnxFrameBundleMetadata,
    backend: OnnxLmBackend,
}

enum OnnxLmBackend {
    Quantized {
        lm: QuantizedLm,
        state: Option<QuantizedLmState>,
        hash: String,
    },
}

impl OnnxLmCodec {
    pub fn from_dir(dir: impl AsRef<Path>, target: ExecutionTarget) -> Result<Self> {
        let bundle_dir = dir.as_ref().to_path_buf();
        let metadata_path = bundle_dir.join("bundle.json");
        let metadata: OnnxFrameBundleMetadata = serde_json::from_str(
            &fs::read_to_string(&metadata_path)
                .with_context(|| format!("failed to read {}", metadata_path.display()))?,
        )
        .with_context(|| format!("failed to parse {}", metadata_path.display()))?;

        if metadata.schema_version != 1 {
            bail!(
                "unsupported bundle schema_version {}",
                metadata.schema_version
            );
        }
        metadata.lm_dim()?;
        metadata.lm_num_layers()?;
        metadata.lm_past_context()?;

        let _ = target;
        let weight_model = metadata
            .lm_quant_weight_model
            .clone()
            .ok_or_else(|| anyhow::anyhow!("bundle does not include lm_quant_weight_model"))?;
        let weights_path = bundle_dir.join(weight_model);
        let weight_bytes = fs::read(&weights_path)
            .with_context(|| format!("failed to read {}", weights_path.display()))?;
        let hash = stable_hash_hex(&weight_bytes);
        let weights = QuantizedLmWeights::from_bytes(&weight_bytes)
            .with_context(|| format!("failed to parse {}", weights_path.display()))?;
        weights.validate_for_codebooks(metadata.num_codebooks)?;
        let backend = OnnxLmBackend::Quantized {
            lm: QuantizedLm::new(weights),
            state: None,
            hash,
        };

        Ok(Self {
            bundle_dir,
            metadata,
            backend,
        })
    }

    pub fn bundle_dir(&self) -> &Path {
        &self.bundle_dir
    }

    pub fn metadata(&self) -> &OnnxFrameBundleMetadata {
        &self.metadata
    }

    pub fn initial_states(&self, batch: usize) -> Result<Vec<Array3<f32>>> {
        let dim = self.metadata.lm_dim()?;
        let layers = self.metadata.lm_num_layers()?;
        Ok((0..layers)
            // The reference Python path starts teacher-forced LM evaluation with
            // `states=None`, which becomes a single zero timestep per layer.
            // Feeding a full `past_context` block of zeros changes attention and
            // destroys compression efficiency.
            .map(|_| Array3::<f32>::zeros((batch, 1, dim)))
            .collect())
    }

    pub fn forward_logits(
        &mut self,
        indices: &Array3<i64>,
        offset: i64,
        states: &[Array3<f32>],
    ) -> Result<(Array4<f32>, i64, Vec<Array3<f32>>)> {
        let shape = indices.shape();
        if shape.len() != 3 {
            bail!("LM indices must have shape [batch, codebooks, steps]");
        }
        if shape[1] > self.metadata.num_codebooks {
            bail!(
                "LM indices use {} codebooks, but bundle only supports {}",
                shape[1],
                self.metadata.num_codebooks
            );
        }
        if states.len() != self.metadata.lm_num_layers()? {
            bail!(
                "LM state count {} does not match bundle layer count {}",
                states.len(),
                self.metadata.lm_num_layers()?
            );
        }

        let OnnxLmBackend::Quantized { lm, state, .. } = &mut self.backend;
        if shape[0] != 1 || shape[2] != 1 {
            bail!("q8 LM only supports shape [1, codebooks, 1]");
        }
        if offset == 0 || state.is_none() {
            *state = Some(lm.initial_state());
        }
        let state = state.as_mut().expect("state initialized");
        let mut input_symbols = Vec::with_capacity(shape[1]);
        for codebook in 0..shape[1] {
            let value = indices[[0, codebook, 0]];
            if value < 0 {
                bail!("LM input symbol must be non-negative, got {value}");
            }
            input_symbols.push(value as usize);
        }
        let logits = lm.forward_step(state, &input_symbols)?;
        let card = self.metadata.lm_cardinality();
        let codebooks = self.metadata.num_codebooks;
        let logits = Array4::from_shape_vec((1, card, codebooks, 1), logits).expect("shape");
        Ok((logits, offset + 1, self.initial_states(shape[0])?))
    }
}

impl LmCodec for OnnxLmCodec {
    fn metadata(&self) -> &OnnxFrameBundleMetadata {
        OnnxLmCodec::metadata(self)
    }

    fn bitstream_version(&self) -> u8 {
        QUANTIZED_LM_BITSTREAM_VERSION
    }

    fn bitstream_lm_hash(&self) -> Option<&str> {
        match &self.backend {
            OnnxLmBackend::Quantized { hash, .. } => Some(hash),
        }
    }

    fn initial_states(&self, batch: usize) -> Result<Vec<Array3<f32>>> {
        OnnxLmCodec::initial_states(self, batch)
    }

    fn forward_logits(
        &mut self,
        indices: &Array3<i64>,
        offset: i64,
        states: &[Array3<f32>],
    ) -> Result<(Array4<f32>, i64, Vec<Array3<f32>>)> {
        OnnxLmCodec::forward_logits(self, indices, offset, states)
    }
}
