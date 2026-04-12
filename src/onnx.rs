use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use ndarray::{Array2, Array3, Ix2, Ix3};
use ort::execution_providers::{
    CPUExecutionProvider, CUDAExecutionProvider, ExecutionProvider, ExecutionProviderDispatch,
    TensorRT,
};
use ort::logging::LogLevel;
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::Tensor as OrtTensor;
use serde::Deserialize;

fn ort_error<E: std::fmt::Display>(error: E) -> anyhow::Error {
    anyhow::anyhow!(error.to_string())
}

fn session_from_providers(path: &Path, providers: impl AsRef<[ExecutionProviderDispatch]>) -> Result<Session> {
    Session::builder()
        .map_err(ort_error)?
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .map_err(ort_error)?
        .with_log_level(LogLevel::Warning)
        .map_err(ort_error)?
        .with_execution_providers(providers)
        .map_err(ort_error)?
        .with_intra_threads(1)
        .map_err(ort_error)?
        .commit_from_file(path)
        .map_err(ort_error)
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ExecutionTarget {
    Cpu,
    Cuda { device_id: i32 },
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
    pub encode_model: String,
    pub decode_model: String,
    pub opset_version: usize,
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
            bail!("unsupported bundle schema_version {}", metadata.schema_version);
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
                let encoder = match session_from_providers(&encoder_path, [cuda.clone()]) {
                    Ok(session) => session,
                    Err(_) => session_from_providers(&encoder_path, [cpu.clone()])?,
                };
                let decoder = match session_from_providers(&decoder_path, [cuda]) {
                    Ok(session) => session,
                    Err(_) => session_from_providers(&decoder_path, [cpu])?,
                };
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
                    fs::create_dir_all(path)
                        .with_context(|| format!("failed to create TensorRT engine cache dir {}", path.display()))?;
                    tensorrt = tensorrt.with_engine_cache_path(path.display().to_string());
                }
                if let Some(path) = &timing_cache_path {
                    if let Some(parent) = path.parent() {
                        fs::create_dir_all(parent).with_context(|| {
                            format!("failed to create TensorRT timing cache dir {}", parent.display())
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
                let decoder = session_from_providers(
                    &decoder_path,
                    [tensorrt.build(), cuda, cpu],
                )?;
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

    pub fn decode_frame(&mut self, codes: &Array3<i64>, scale: &Array2<f32>) -> Result<Array3<f32>> {
        let code_shape = codes.shape();
        if code_shape.len() != 3 {
            bail!("codes must have shape [batch, num_codebooks, frame_length]");
        }
        if code_shape[1] != self.metadata.num_codebooks || code_shape[2] != self.metadata.frame_length {
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
