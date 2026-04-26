use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use ndarray::{Array0, Array2, Array3, Array4, Ix0, Ix2, Ix3, Ix4};
use ort::execution_providers::{
    coreml, CPUExecutionProvider, CUDAExecutionProvider, CoreML, ExecutionProvider,
    ExecutionProviderDispatch, TensorRT,
};
use ort::inputs;
use ort::logging::LogLevel;
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::Tensor as OrtTensor;
use serde::Deserialize;

fn ort_error<E: std::fmt::Display>(error: E) -> anyhow::Error {
    anyhow::anyhow!(error.to_string())
}

fn ort_intra_threads() -> usize {
    std::env::var("ENCODEC_RS_ORT_THREADS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or_else(|| {
            std::thread::available_parallelism()
                .map(|value| value.get().min(4))
                .unwrap_or(1)
        })
}

fn session_from_providers(
    path: &Path,
    providers: impl AsRef<[ExecutionProviderDispatch]>,
) -> Result<Session> {
    Session::builder()
        .map_err(ort_error)?
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .map_err(ort_error)?
        .with_log_level(LogLevel::Warning)
        .map_err(ort_error)?
        .with_execution_providers(providers)
        .map_err(ort_error)?
        .with_intra_threads(ort_intra_threads())
        .map_err(ort_error)?
        .commit_from_file(path)
        .map_err(ort_error)
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ExecutionTarget {
    Cpu,
    Cuda {
        device_id: i32,
    },
    CoreMl {
        compute_units: CoreMlComputeUnits,
        model_cache_dir: Option<PathBuf>,
        low_precision_accumulation_on_gpu: bool,
    },
    TensorRt {
        device_id: i32,
        fp16: bool,
        engine_cache_path: Option<PathBuf>,
        timing_cache_path: Option<PathBuf>,
    },
}

impl Default for ExecutionTarget {
    fn default() -> Self {
        Self::Cpu
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CoreMlComputeUnits {
    All,
    CpuAndNeuralEngine,
    CpuAndGpu,
    CpuOnly,
}

impl From<CoreMlComputeUnits> for coreml::ComputeUnits {
    fn from(value: CoreMlComputeUnits) -> Self {
        match value {
            CoreMlComputeUnits::All => Self::All,
            CoreMlComputeUnits::CpuAndNeuralEngine => Self::CPUAndNeuralEngine,
            CoreMlComputeUnits::CpuAndGpu => Self::CPUAndGPU,
            CoreMlComputeUnits::CpuOnly => Self::CPUOnly,
        }
    }
}

fn coreml_provider(
    compute_units: CoreMlComputeUnits,
    model_cache_dir: Option<&Path>,
    low_precision_accumulation_on_gpu: bool,
) -> Result<ExecutionProviderDispatch> {
    let base = CoreML::default();
    if !base.is_available().unwrap_or(false) {
        bail!("CoreML Execution Provider is not available");
    }

    let mut coreml = base
        .with_compute_units(compute_units.into())
        .with_specialization_strategy(coreml::SpecializationStrategy::FastPrediction);
    if let Some(path) = model_cache_dir {
        fs::create_dir_all(path).with_context(|| {
            format!("failed to create CoreML model cache dir {}", path.display())
        })?;
        coreml = coreml.with_model_cache_dir(path.display().to_string());
    }
    if low_precision_accumulation_on_gpu {
        coreml = coreml.with_low_precision_accumulation_on_gpu(true);
    }

    Ok(coreml.build().error_on_failure())
}

#[derive(Clone, Debug, Deserialize)]
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
    pub lm_model: Option<String>,
    #[serde(default)]
    pub lm_dim: Option<usize>,
    #[serde(default)]
    pub lm_num_layers: Option<usize>,
    #[serde(default)]
    pub lm_past_context: Option<usize>,
    #[serde(default)]
    pub lm_logit_step: Option<f32>,
    #[serde(default)]
    pub lm_cardinality: Option<usize>,
    #[serde(default)]
    pub lm_dtype: Option<String>,
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

        let cpu = CPUExecutionProvider::default().build();
        let (encoder, decoder) = match target {
            ExecutionTarget::Cpu => (
                session_from_providers(&encoder_path, [cpu.clone()])?,
                session_from_providers(&decoder_path, [cpu])?,
            ),
            ExecutionTarget::Cuda { device_id } => {
                if !CUDAExecutionProvider::default()
                    .is_available()
                    .unwrap_or(false)
                {
                    bail!("CUDA Execution Provider is not available");
                }
                let cuda = CUDAExecutionProvider::default()
                    .with_device_id(device_id)
                    .build();
                let encoder =
                    match session_from_providers(&encoder_path, [cuda.clone(), cpu.clone()]) {
                        Ok(session) => session,
                        Err(_) => session_from_providers(&encoder_path, [cpu.clone()])?,
                    };
                let decoder = match session_from_providers(&decoder_path, [cuda, cpu.clone()]) {
                    Ok(session) => session,
                    Err(_) => session_from_providers(&decoder_path, [cpu])?,
                };
                (encoder, decoder)
            }
            ExecutionTarget::CoreMl {
                compute_units,
                model_cache_dir,
                low_precision_accumulation_on_gpu,
            } => {
                let encoder_cache_dir = model_cache_dir
                    .as_ref()
                    .map(|path| path.join("encode_frame"));
                let decoder_cache_dir = model_cache_dir
                    .as_ref()
                    .map(|path| path.join("decode_frame"));
                let encoder = session_from_providers(
                    &encoder_path,
                    [
                        coreml_provider(
                            compute_units,
                            encoder_cache_dir.as_deref(),
                            low_precision_accumulation_on_gpu,
                        )?,
                        cpu.clone(),
                    ],
                )?;
                let decoder = session_from_providers(
                    &decoder_path,
                    [
                        coreml_provider(
                            compute_units,
                            decoder_cache_dir.as_deref(),
                            low_precision_accumulation_on_gpu,
                        )?,
                        cpu,
                    ],
                )?;
                (encoder, decoder)
            }
            ExecutionTarget::TensorRt {
                device_id,
                fp16,
                engine_cache_path,
                timing_cache_path,
            } => {
                let mut tensorrt = TensorRT::default()
                    .with_device_id(device_id)
                    .with_engine_cache(true)
                    .with_force_sequential_engine_build(true)
                    .with_builder_optimization_level(5)
                    .with_timing_cache(true)
                    .with_fp16(fp16);
                if let Some(path) = &engine_cache_path {
                    fs::create_dir_all(path).with_context(|| {
                        format!(
                            "failed to create TensorRT engine cache dir {}",
                            path.display()
                        )
                    })?;
                    tensorrt = tensorrt.with_engine_cache_path(path.display().to_string());
                }
                if let Some(path) = &timing_cache_path {
                    if let Some(parent) = path.parent() {
                        fs::create_dir_all(parent).with_context(|| {
                            format!(
                                "failed to create TensorRT timing cache dir {}",
                                parent.display()
                            )
                        })?;
                    }
                    tensorrt = tensorrt.with_timing_cache_path(path.display().to_string());
                }

                let cuda = CUDAExecutionProvider::default()
                    .with_device_id(device_id)
                    .build();
                let encoder = session_from_providers(
                    &encoder_path,
                    [tensorrt.clone().build(), cuda.clone(), cpu.clone()],
                )?;
                let decoder = session_from_providers(&decoder_path, [tensorrt.build(), cuda, cpu])?;
                (encoder, decoder)
            }
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

pub struct OnnxLmCodec {
    bundle_dir: PathBuf,
    metadata: OnnxFrameBundleMetadata,
    session: Session,
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
        let lm_model = metadata
            .lm_model
            .clone()
            .ok_or_else(|| anyhow::anyhow!("bundle does not include an LM ONNX model"))?;
        let lm_path = bundle_dir.join(lm_model);
        if !lm_path.exists() {
            bail!("bundle is missing LM model file {}", lm_path.display());
        }

        let cpu = CPUExecutionProvider::default().build();
        let session = match target {
            ExecutionTarget::Cpu => session_from_providers(&lm_path, [cpu])?,
            ExecutionTarget::Cuda { device_id } => {
                if !CUDAExecutionProvider::default()
                    .is_available()
                    .unwrap_or(false)
                {
                    bail!("CUDA Execution Provider is not available");
                }
                let cuda = CUDAExecutionProvider::default()
                    .with_device_id(device_id)
                    .build();
                match session_from_providers(&lm_path, [cuda, cpu.clone()]) {
                    Ok(session) => session,
                    Err(_) => session_from_providers(&lm_path, [cpu])?,
                }
            }
            ExecutionTarget::CoreMl {
                compute_units,
                model_cache_dir,
                low_precision_accumulation_on_gpu,
            } => {
                let lm_cache_dir = model_cache_dir.as_ref().map(|path| path.join("lm_logits"));
                session_from_providers(
                    &lm_path,
                    [
                        coreml_provider(
                            compute_units,
                            lm_cache_dir.as_deref(),
                            low_precision_accumulation_on_gpu,
                        )?,
                        cpu,
                    ],
                )?
            }
            ExecutionTarget::TensorRt {
                device_id,
                fp16,
                engine_cache_path,
                timing_cache_path,
            } => {
                let mut tensorrt = TensorRT::default()
                    .with_device_id(device_id)
                    .with_engine_cache(true)
                    .with_force_sequential_engine_build(true)
                    .with_builder_optimization_level(5)
                    .with_timing_cache(true)
                    .with_fp16(fp16);
                if let Some(path) = &engine_cache_path {
                    fs::create_dir_all(path).with_context(|| {
                        format!(
                            "failed to create TensorRT engine cache dir {}",
                            path.display()
                        )
                    })?;
                    tensorrt = tensorrt.with_engine_cache_path(path.display().to_string());
                }
                if let Some(path) = &timing_cache_path {
                    if let Some(parent) = path.parent() {
                        fs::create_dir_all(parent).with_context(|| {
                            format!(
                                "failed to create TensorRT timing cache dir {}",
                                parent.display()
                            )
                        })?;
                    }
                    tensorrt = tensorrt.with_timing_cache_path(path.display().to_string());
                }

                let cuda = CUDAExecutionProvider::default()
                    .with_device_id(device_id)
                    .build();
                session_from_providers(&lm_path, [tensorrt.build(), cuda, cpu])?
            }
        };
        metadata.lm_dim()?;
        metadata.lm_num_layers()?;
        metadata.lm_past_context()?;

        Ok(Self {
            bundle_dir,
            metadata,
            session,
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

        let offset_tensor = Array0::from_elem((), offset);
        let mut inputs = inputs![
            "indices" => OrtTensor::from_array(indices.to_owned()).map_err(ort_error)?,
            "offset" => OrtTensor::from_array(offset_tensor).map_err(ort_error)?,
        ];
        for (index, state) in states.iter().enumerate() {
            inputs.push((
                format!("state_{index}").into(),
                OrtTensor::from_array(state.to_owned())
                    .map_err(ort_error)?
                    .into(),
            ));
        }

        let outputs = self.session.run(inputs).map_err(ort_error)?;
        let logits = outputs["logits"]
            .try_extract_array::<f32>()
            .map_err(ort_error)?
            .to_owned()
            .into_dimensionality::<Ix4>()
            .map_err(ort_error)?;
        let next_offset = outputs["offset_out"]
            .try_extract_array::<i64>()
            .map_err(ort_error)?
            .to_owned()
            .into_dimensionality::<Ix0>()
            .map_err(ort_error)?
            .into_scalar();
        let next_states = (0..states.len())
            .map(|index| {
                outputs[format!("next_state_{index}")]
                    .try_extract_array::<f32>()
                    .map_err(ort_error)?
                    .to_owned()
                    .into_dimensionality::<Ix3>()
                    .map_err(ort_error)
            })
            .collect::<Result<Vec<_>>>()?;
        Ok((logits, next_offset, next_states))
    }
}
