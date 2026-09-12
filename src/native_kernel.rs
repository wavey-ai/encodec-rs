//! Native FFI bindings to the shared EnCodec C kernels.
//!
//! The decoder bundle carries a `decoder/metadata.json` model description and
//! a packed float32 weight file. This module packs those weights for the C
//! kernels and drives them exactly like
//! `browser-runtime/custom-decoder-runtime.js` drives the WASM kernels, so the
//! two backends can be compared frame for frame.

use std::collections::HashMap;
use std::ffi::c_int;
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use ndarray::{Array2, Array3};

use crate::ecdc::FrameCodec;
use crate::metadata::FrameBundleMetadata;

extern "C" {
    fn pack_conv_transpose1d_weights(
        weights: *const f32,
        packed: *mut f32,
        input_channels: c_int,
        output_channels: c_int,
        stride: c_int,
    ) -> c_int;
    fn conv_transpose1d_phase_simd_8x8_nhwc(
        input: *const f32,
        packed: *const f32,
        bias: *const f32,
        output: *mut f32,
        time: c_int,
        input_channels: c_int,
        output_channels: c_int,
        stride: c_int,
    ) -> c_int;
    fn crop_nhwc(
        input: *const f32,
        output: *mut f32,
        raw_time: c_int,
        channels: c_int,
        crop_left: c_int,
        crop_right: c_int,
    ) -> c_int;

    fn pack_conv1d_nhwc_weights_8(
        weights: *const f32,
        packed: *mut f32,
        input_channels: c_int,
        output_channels: c_int,
        kernel: c_int,
    ) -> c_int;
    fn pack_linear_weights_8(
        weights: *const f32,
        packed: *mut f32,
        input_size: c_int,
        output_size: c_int,
    ) -> c_int;
    fn reflect_pad_nhwc(
        input: *const f32,
        output: *mut f32,
        time: c_int,
        channels: c_int,
        padding_left: c_int,
        padding_right: c_int,
    ) -> c_int;
    fn reflect_pad_elu_nhwc(
        input: *const f32,
        output: *mut f32,
        time: c_int,
        channels: c_int,
        padding_left: c_int,
        padding_right: c_int,
    ) -> c_int;
    fn group_norm_nhwc_in_place(
        values: *mut f32,
        scale: *const f32,
        bias: *const f32,
        time: c_int,
        channels: c_int,
    ) -> c_int;
    fn elu_nhwc_in_place(values: *mut f32, length: c_int) -> c_int;
    fn add_elu_nhwc_in_place(
        destination: *mut f32,
        source: *const f32,
        length: c_int,
    ) -> c_int;
    fn rvq_decode_codes_nhwc(
        codes: *const u16,
        embeddings: *const f32,
        output: *mut f32,
        time: c_int,
        dimension: c_int,
        entries: c_int,
        codebooks: c_int,
    ) -> c_int;
    fn conv1d_nhwc_simd_8x8(
        input: *const f32,
        packed: *const f32,
        bias: *const f32,
        output: *mut f32,
        padded_time: c_int,
        input_channels: c_int,
        output_channels: c_int,
        kernel: c_int,
        stride: c_int,
    ) -> c_int;
    fn lstm_layer_simd_64(
        input: *const f32,
        packed_input: *const f32,
        packed_recurrent: *const f32,
        bias: *const f32,
        output: *mut f32,
        hidden_state: *mut f32,
        cell_state: *mut f32,
        input_projection: *mut f32,
        sequence_length: c_int,
        hidden_size: c_int,
    ) -> c_int;
    fn compact_nhwc_channels(
        input: *const f32,
        output: *mut f32,
        time: c_int,
        input_channels: c_int,
        output_channels: c_int,
    ) -> c_int;
    fn scale_nhwc_to_nct(
        input: *const f32,
        output: *mut f32,
        scale: f32,
        time: c_int,
        channels: c_int,
    ) -> c_int;
    fn normalize_audio_planar_to_nhwc(
        input: *const f32,
        output: *mut f32,
        time: c_int,
        channels: c_int,
    ) -> f32;
    fn add_nhwc_in_place(destination: *mut f32, source: *const f32, length: c_int) -> c_int;
    fn rvq_encode_simd_8(
        input: *const f32,
        residual: *mut f32,
        embeddings: *const f32,
        embedding_norms: *const f32,
        codes: *mut i32,
        time: c_int,
        dimension: c_int,
        entries: c_int,
        codebooks: c_int,
    ) -> c_int;
}

fn check(status: c_int, what: &str) -> Result<()> {
    if status != 1 {
        bail!("native EnCodec kernel {what} failed with status {status}");
    }
    Ok(())
}

#[derive(Debug, Clone, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DecoderMetadata {
    pub frame_length: usize,
    pub num_codebooks: usize,
    pub channels: usize,
    pub segment_samples: usize,
    pub front: DecoderFront,
    pub post: DecoderPost,
    pub layers: Vec<DecoderTransposeLayer>,
}

#[derive(Debug, Clone, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DecoderFront {
    pub rvq: DecoderRvq,
    pub conv: DecoderConvShape,
    pub lstm_layers: Vec<DecoderLstmLayer>,
}

#[derive(Debug, Clone, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DecoderRvq {
    pub codebooks: usize,
    pub entries: usize,
    pub dimension: usize,
}

#[derive(Debug, Clone, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DecoderConvShape {
    pub input_channels: usize,
    pub output_channels: usize,
    pub kernel: usize,
    pub stride: usize,
    pub input_time: usize,
    pub padding_left: usize,
    pub padding_right: usize,
    pub padded_input_time: usize,
    pub output_time: usize,
}

#[derive(Debug, Clone, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DecoderLstmLayer {
    pub layer: usize,
    pub input_size: usize,
    pub hidden_size: usize,
    pub gate_size: usize,
}

#[derive(Debug, Clone, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DecoderPost {
    pub conv_layers: Vec<DecoderConvLayer>,
    pub blocks: Vec<[usize; 3]>,
    pub final_conv: DecoderFinalConv,
}

#[derive(Debug, Clone, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DecoderConvLayer {
    pub layer: usize,
    pub input_channels: usize,
    pub output_channels: usize,
    pub kernel: usize,
    pub stride: usize,
    pub input_time: usize,
    pub padding_left: usize,
    pub padding_right: usize,
    pub padded_input_time: usize,
    pub output_time: usize,
}

#[derive(Debug, Clone, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DecoderFinalConv {
    pub input_channels: usize,
    pub output_channels: usize,
    pub kernel_output_channels: usize,
    pub kernel: usize,
    pub stride: usize,
    pub input_time: usize,
    pub padding_left: usize,
    pub padding_right: usize,
    pub padded_input_time: usize,
    pub output_time: usize,
}

#[derive(Debug, Clone, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DecoderTransposeLayer {
    pub layer: usize,
    pub input_channels: usize,
    pub output_channels: usize,
    pub kernel: usize,
    pub stride: usize,
    pub input_time: usize,
    pub raw_output_time: usize,
    pub crop_left: usize,
    pub crop_right: usize,
    pub cropped_output_time: usize,
}

#[derive(Debug, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
struct WeightsManifest {
    file: String,
    byte_length: usize,
    tensors: HashMap<String, TensorEntry>,
}

#[derive(Debug, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
struct TensorEntry {
    offset_bytes: usize,
    length: usize,
}

enum WeightSource {
    Files(PathBuf),
    Packed {
        data: Vec<f32>,
        tensors: HashMap<String, (usize, usize)>,
    },
}

impl WeightSource {
    fn open(dir: &Path) -> Result<Self> {
        let manifest_path = dir.join("weights.json");
        if manifest_path.is_file() {
            let manifest: WeightsManifest = serde_json::from_str(
                &std::fs::read_to_string(&manifest_path)
                    .with_context(|| format!("failed to read {}", manifest_path.display()))?,
            )
            .with_context(|| format!("failed to parse {}", manifest_path.display()))?;
            let packed_path = dir.join(&manifest.file);
            let bytes = std::fs::read(&packed_path)
                .with_context(|| format!("failed to read {}", packed_path.display()))?;
            if bytes.len() != manifest.byte_length {
                bail!(
                    "packed decoder weight length mismatch: {} != {}",
                    bytes.len(),
                    manifest.byte_length,
                );
            }
            let data = bytes_to_f32(&bytes)?;
            let mut tensors = HashMap::with_capacity(manifest.tensors.len());
            for (name, entry) in manifest.tensors {
                if entry.offset_bytes % 4 != 0 {
                    bail!("decoder weight {name} is not 4-byte aligned");
                }
                tensors.insert(name, (entry.offset_bytes / 4, entry.length));
            }
            return Ok(Self::Packed { data, tensors });
        }
        Ok(Self::Files(dir.to_path_buf()))
    }

    fn get(&self, name: &str) -> Result<Vec<f32>> {
        match self {
            Self::Files(root) => {
                let path = root.join(name);
                let bytes = std::fs::read(&path)
                    .with_context(|| format!("failed to read {}", path.display()))?;
                bytes_to_f32(&bytes)
            }
            Self::Packed { data, tensors } => {
                let (offset, length) = tensors
                    .get(name)
                    .copied()
                    .with_context(|| format!("packed decoder weight is missing: {name}"))?;
                let end = offset
                    .checked_add(length)
                    .context("decoder weight extent overflows usize")?;
                if end > data.len() {
                    bail!("decoder weight {name} runs past the packed asset");
                }
                Ok(data[offset..end].to_vec())
            }
        }
    }
}

fn bytes_to_f32(bytes: &[u8]) -> Result<Vec<f32>> {
    if bytes.len() % 4 != 0 {
        bail!("decoder weight asset is not float32 aligned");
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect())
}

fn packed_or_error(
    raw: Vec<f32>,
    pack: impl FnOnce(*const f32, *mut f32) -> c_int,
    what: &str,
) -> Result<Vec<f32>> {
    let mut packed = vec![0.0; raw.len()];
    check(pack(raw.as_ptr(), packed.as_mut_ptr()), what)?;
    Ok(packed)
}

struct LstmLayer {
    packed_input: Vec<f32>,
    packed_recurrent: Vec<f32>,
    bias: Vec<f32>,
}

struct TransposeLayer {
    packed: Vec<f32>,
    bias: Vec<f32>,
    norm_scale: Vec<f32>,
    norm_bias: Vec<f32>,
}

struct ConvLayer {
    packed: Vec<f32>,
    bias: Vec<f32>,
    norm_scale: Vec<f32>,
    norm_bias: Vec<f32>,
}

struct FinalConv {
    packed: Vec<f32>,
    bias: Vec<f32>,
    norm_scale: Vec<f32>,
    norm_bias: Vec<f32>,
}

/// A prepared native decoder for one bundle profile.
pub struct NativeEncodecDecoder {
    metadata: DecoderMetadata,
    embeddings: Vec<f32>,
    front_padded: Vec<f32>,
    front_activation0: Vec<f32>,
    front_activation1: Vec<f32>,
    front_activation2: Vec<f32>,
    front_conv_packed: Vec<f32>,
    front_conv_bias: Vec<f32>,
    front_conv_norm_scale: Vec<f32>,
    front_conv_norm_bias: Vec<f32>,
    lstm_hidden: Vec<f32>,
    lstm_cell: Vec<f32>,
    lstm_input_projection: Vec<f32>,
    lstm_layers: Vec<LstmLayer>,
    layers: Vec<TransposeLayer>,
    post_conv_layers: Vec<ConvLayer>,
    post_final: FinalConv,
    post_shortcut: Vec<f32>,
    post_reduced: Vec<f32>,
    post_padded: Vec<f32>,
    post_in: Vec<f32>,
    post_out: Vec<f32>,
    planar_output: Vec<f32>,
    codes: Vec<u16>,
}

impl NativeEncodecDecoder {
    /// Loads `metadata.json` and the decoder weights from a decoder asset dir.
    pub fn from_dir(dir: impl AsRef<Path>) -> Result<Self> {
        let dir = dir.as_ref();
        let metadata_path = dir.join("metadata.json");
        let metadata: DecoderMetadata = serde_json::from_str(
            &std::fs::read_to_string(&metadata_path)
                .with_context(|| format!("failed to read {}", metadata_path.display()))?,
        )
        .with_context(|| format!("failed to parse {}", metadata_path.display()))?;
        let weights = WeightSource::open(dir)?;
        Self::new(metadata, &weights)
    }

    fn new(metadata: DecoderMetadata, weights: &WeightSource) -> Result<Self> {
        let front = &metadata.front;
        let frame_length = front.conv.input_time;
        let hidden_size = front.conv.output_channels;
        let front_activation_length = frame_length
            .checked_mul(hidden_size)
            .context("front activation length overflows usize")?;
        if front.lstm_layers.len() != 2 {
            bail!("native decoder front must have two LSTM layers");
        }
        if metadata.layers.len() != 4 {
            bail!("native decoder must have four transpose layers");
        }
        if metadata.post.conv_layers.len() != 12 {
            bail!("native decoder post must have twelve convolution layers");
        }
        if metadata.post.blocks.len() != metadata.layers.len() {
            bail!("native decoder block count does not match transpose layers");
        }

        let code_count = front
            .rvq
            .codebooks
            .checked_mul(frame_length)
            .context("decoder code count overflows usize")?;

        let embeddings = weights.get("front-rvq-embeddings.f32le")?;
        let front_conv_packed = {
            let raw = weights.get("front-conv-weight.f32le")?;
            let input_channels = front.conv.input_channels as c_int;
            let output_channels = front.conv.output_channels as c_int;
            let kernel = front.conv.kernel as c_int;
            packed_or_error(
                raw,
                move |weights, packed| unsafe {
                    pack_conv1d_nhwc_weights_8(
                        weights,
                        packed,
                        input_channels,
                        output_channels,
                        kernel,
                    )
                },
                "front convolution weight packing",
            )?
        };
        let front_conv_bias = weights.get("front-conv-bias.f32le")?;
        let front_conv_norm_scale = weights.get("front-conv-norm-scale.f32le")?;
        let front_conv_norm_bias = weights.get("front-conv-norm-bias.f32le")?;

        let mut lstm_layers = Vec::with_capacity(2);
        for layer in &front.lstm_layers {
            let input_weights = weights.get(&format!(
                "front-lstm-{}-input-weight.f32le",
                layer.layer
            ))?;
            let recurrent_weights = weights.get(&format!(
                "front-lstm-{}-recurrent-weight.f32le",
                layer.layer
            ))?;
            let bias = weights.get(&format!("front-lstm-{}-bias.f32le", layer.layer))?;
            let hidden = layer.hidden_size as c_int;
            let gate = layer.gate_size as c_int;
            let packed_input = packed_or_error(
                input_weights,
                move |weights, packed| unsafe {
                    pack_linear_weights_8(weights, packed, hidden, gate)
                },
                "front LSTM input weight packing",
            )?;
            let packed_recurrent = packed_or_error(
                recurrent_weights,
                move |weights, packed| unsafe {
                    pack_linear_weights_8(weights, packed, hidden, gate)
                },
                "front LSTM recurrent weight packing",
            )?;
            lstm_layers.push(LstmLayer {
                packed_input,
                packed_recurrent,
                bias,
            });
        }

        let mut layers = Vec::with_capacity(metadata.layers.len());
        for layer in &metadata.layers {
            let raw = weights.get(&format!("layer-{}-weight.f32le", layer.layer))?;
            let input_channels = layer.input_channels as c_int;
            let output_channels = layer.output_channels as c_int;
            let stride = layer.stride as c_int;
            let packed = packed_or_error(
                raw,
                move |weights, packed| unsafe {
                    pack_conv_transpose1d_weights(
                        weights,
                        packed,
                        input_channels,
                        output_channels,
                        stride,
                    )
                },
                "transpose weight packing",
            )?;
            layers.push(TransposeLayer {
                packed,
                bias: weights.get(&format!("layer-{}-bias.f32le", layer.layer))?,
                norm_scale: weights.get(&format!("layer-{}-norm-scale.f32le", layer.layer))?,
                norm_bias: weights.get(&format!("layer-{}-norm-bias.f32le", layer.layer))?,
            });
        }

        let mut post_conv_layers = Vec::with_capacity(metadata.post.conv_layers.len());
        for layer in &metadata.post.conv_layers {
            let raw = weights.get(&format!("post-conv-{}-weight.f32le", layer.layer))?;
            let input_channels = layer.input_channels as c_int;
            let output_channels = layer.output_channels as c_int;
            let kernel = layer.kernel as c_int;
            let packed = packed_or_error(
                raw,
                move |weights, packed| unsafe {
                    pack_conv1d_nhwc_weights_8(
                        weights,
                        packed,
                        input_channels,
                        output_channels,
                        kernel,
                    )
                },
                "post convolution weight packing",
            )?;
            post_conv_layers.push(ConvLayer {
                packed,
                bias: weights.get(&format!("post-conv-{}-bias.f32le", layer.layer))?,
                norm_scale: weights.get(&format!("post-conv-{}-norm-scale.f32le", layer.layer))?,
                norm_bias: weights.get(&format!("post-conv-{}-norm-bias.f32le", layer.layer))?,
            });
        }

        let final_conv = &metadata.post.final_conv;
        let final_weights = weights.get("final-conv-weight.f32le")?;
        let final_bias = weights.get("final-conv-bias.f32le")?;
        let padded_weight_length = final_conv
            .kernel_output_channels
            .checked_mul(final_conv.input_channels)
            .and_then(|value| value.checked_mul(final_conv.kernel))
            .context("final convolution weight length overflows usize")?;
        let mut padded_weights = vec![0.0; padded_weight_length];
        if final_weights.len() > padded_weights.len() {
            bail!("final convolution weights exceed the padded kernel geometry");
        }
        padded_weights[..final_weights.len()].copy_from_slice(&final_weights);
        let mut padded_bias = vec![0.0; final_conv.kernel_output_channels];
        if final_bias.len() > padded_bias.len() {
            bail!("final convolution bias exceeds the padded kernel geometry");
        }
        padded_bias[..final_bias.len()].copy_from_slice(&final_bias);
        let final_input_channels = final_conv.input_channels as c_int;
        let final_output_channels = final_conv.kernel_output_channels as c_int;
        let final_kernel = final_conv.kernel as c_int;
        let final_packed = packed_or_error(
            padded_weights,
            move |weights, packed| unsafe {
                pack_conv1d_nhwc_weights_8(
                    weights,
                    packed,
                    final_input_channels,
                    final_output_channels,
                    final_kernel,
                )
            },
            "final convolution weight packing",
        )?;
        let post_final = FinalConv {
            packed: final_packed,
            bias: padded_bias,
            norm_scale: weights.get("final-conv-norm-scale.f32le")?,
            norm_bias: weights.get("final-conv-norm-bias.f32le")?,
        };

        let max_layer_input = metadata
            .layers
            .iter()
            .map(|layer| layer.input_channels * layer.input_time)
            .max()
            .unwrap_or(1);
        let max_layer_output = metadata
            .layers
            .iter()
            .map(|layer| layer.output_channels * layer.raw_output_time)
            .max()
            .unwrap_or(1);
        let activation_length = max_layer_input.max(max_layer_output);
        let max_shortcut = metadata
            .layers
            .iter()
            .map(|layer| layer.cropped_output_time * layer.output_channels)
            .max()
            .unwrap_or(1);
        let max_reduced = metadata
            .post
            .conv_layers
            .iter()
            .map(|layer| layer.output_time * layer.output_channels)
            .chain(std::iter::once(
                final_conv.output_time * final_conv.output_channels,
            ))
            .max()
            .unwrap_or(1);
        let max_padded = metadata
            .post
            .conv_layers
            .iter()
            .map(|layer| layer.padded_input_time * layer.input_channels)
            .chain(std::iter::once(
                final_conv.padded_input_time * final_conv.input_channels,
            ))
            .max()
            .unwrap_or(1);

        let front_padded_length = front.conv.padded_input_time * front.conv.input_channels;
        let planar_output_length = final_conv.output_time * final_conv.output_channels;

        Ok(Self {
            metadata,
            embeddings,
            front_padded: vec![0.0; front_padded_length],
            front_activation0: vec![0.0; front_activation_length],
            front_activation1: vec![0.0; front_activation_length],
            front_activation2: vec![0.0; front_activation_length],
            front_conv_packed,
            front_conv_bias,
            front_conv_norm_scale,
            front_conv_norm_bias,
            lstm_hidden: vec![0.0; hidden_size],
            lstm_cell: vec![0.0; hidden_size],
            lstm_input_projection: vec![0.0; frame_length * 4 * hidden_size],
            lstm_layers,
            layers,
            post_conv_layers,
            post_final,
            post_shortcut: vec![0.0; max_shortcut],
            post_reduced: vec![0.0; max_reduced],
            post_padded: vec![0.0; max_padded],
            post_in: vec![0.0; activation_length],
            post_out: vec![0.0; activation_length],
            planar_output: vec![0.0; planar_output_length],
            codes: vec![0; code_count],
        })
    }

    pub fn metadata(&self) -> &DecoderMetadata {
        &self.metadata
    }

    fn run_front(&mut self, codes: &[u16]) -> Result<()> {
        let frame_length = self.metadata.front.conv.input_time;
        let hidden_size = self.metadata.front.conv.output_channels;
        if codes.len() != self.codes.len() {
            bail!(
                "decoder frame has {} codes; expected {}",
                codes.len(),
                self.codes.len()
            );
        }
        self.codes.copy_from_slice(codes);

        {
            let conv = &self.metadata.front.conv;
            let rvq = &self.metadata.front.rvq;
            unsafe {
                check(
                    rvq_decode_codes_nhwc(
                        self.codes.as_ptr(),
                        self.embeddings.as_ptr(),
                        self.front_activation0.as_mut_ptr(),
                        frame_length as c_int,
                        rvq.dimension as c_int,
                        rvq.entries as c_int,
                        rvq.codebooks as c_int,
                    ),
                    "codebook reconstruction",
                )?;
                check(
                    reflect_pad_nhwc(
                        self.front_activation0.as_ptr(),
                        self.front_padded.as_mut_ptr(),
                        frame_length as c_int,
                        conv.input_channels as c_int,
                        conv.padding_left as c_int,
                        conv.padding_right as c_int,
                    ),
                    "front padding",
                )?;
                check(
                    conv1d_nhwc_simd_8x8(
                        self.front_padded.as_ptr(),
                        self.front_conv_packed.as_ptr(),
                        self.front_conv_bias.as_ptr(),
                        self.front_activation1.as_mut_ptr(),
                        conv.padded_input_time as c_int,
                        conv.input_channels as c_int,
                        conv.output_channels as c_int,
                        conv.kernel as c_int,
                        conv.stride as c_int,
                    ),
                    "front convolution",
                )?;
                check(
                    group_norm_nhwc_in_place(
                        self.front_activation1.as_mut_ptr(),
                        self.front_conv_norm_scale.as_ptr(),
                        self.front_conv_norm_bias.as_ptr(),
                        frame_length as c_int,
                        hidden_size as c_int,
                    ),
                    "front normalization",
                )?;
            }
        }

        let activation1 = self.front_activation1.as_ptr();
        let activation0 = self.front_activation0.as_mut_ptr();
        {
            let layer = &self.lstm_layers[0];
            let (packed_input, packed_recurrent, bias) = (
                layer.packed_input.as_ptr(),
                layer.packed_recurrent.as_ptr(),
                layer.bias.as_ptr(),
            );
            unsafe {
                check(
                    lstm_layer_simd_64(
                        activation1,
                        packed_input,
                        packed_recurrent,
                        bias,
                        activation0,
                        self.lstm_hidden.as_mut_ptr(),
                        self.lstm_cell.as_mut_ptr(),
                        self.lstm_input_projection.as_mut_ptr(),
                        frame_length as c_int,
                        hidden_size as c_int,
                    ),
                    "front LSTM 0",
                )?;
            }
        }

        let activation0 = self.front_activation0.as_ptr();
        let activation2 = self.front_activation2.as_mut_ptr();
        {
            let layer = &self.lstm_layers[1];
            let (packed_input, packed_recurrent, bias) = (
                layer.packed_input.as_ptr(),
                layer.packed_recurrent.as_ptr(),
                layer.bias.as_ptr(),
            );
            unsafe {
                check(
                    lstm_layer_simd_64(
                        activation0,
                        packed_input,
                        packed_recurrent,
                        bias,
                        activation2,
                        self.lstm_hidden.as_mut_ptr(),
                        self.lstm_cell.as_mut_ptr(),
                        self.lstm_input_projection.as_mut_ptr(),
                        frame_length as c_int,
                        hidden_size as c_int,
                    ),
                    "front LSTM 1",
                )?;
            }
        }

        unsafe {
            check(
                add_elu_nhwc_in_place(
                    self.front_activation2.as_mut_ptr(),
                    self.front_activation1.as_ptr(),
                    (frame_length * hidden_size) as c_int,
                ),
                "front output assembly",
            )?;
        }
        Ok(())
    }

    fn run_post_conv(
        &mut self,
        index: usize,
        input: *const f32,
        output: *mut f32,
        apply_elu: bool,
    ) -> Result<()> {
        let layer = self.metadata.post.conv_layers[index].clone();
        let mut convolution_input = input;
        unsafe {
            if layer.padding_left != 0 || layer.padding_right != 0 {
                let status = if apply_elu {
                    reflect_pad_elu_nhwc(
                        input,
                        self.post_padded.as_mut_ptr(),
                        layer.input_time as c_int,
                        layer.input_channels as c_int,
                        layer.padding_left as c_int,
                        layer.padding_right as c_int,
                    )
                } else {
                    reflect_pad_nhwc(
                        input,
                        self.post_padded.as_mut_ptr(),
                        layer.input_time as c_int,
                        layer.input_channels as c_int,
                        layer.padding_left as c_int,
                        layer.padding_right as c_int,
                    )
                };
                check(status, "post padding")?;
                convolution_input = self.post_padded.as_ptr();
            } else if apply_elu {
                check(
                    elu_nhwc_in_place(
                        input as *mut f32,
                        (layer.input_time * layer.input_channels) as c_int,
                    ),
                    "post ELU",
                )?;
            }

            let layer_state = &self.post_conv_layers[index];
            check(
                conv1d_nhwc_simd_8x8(
                    convolution_input,
                    layer_state.packed.as_ptr(),
                    layer_state.bias.as_ptr(),
                    output,
                    layer.padded_input_time as c_int,
                    layer.input_channels as c_int,
                    layer.output_channels as c_int,
                    layer.kernel as c_int,
                    layer.stride as c_int,
                ),
                "post convolution",
            )?;
            check(
                group_norm_nhwc_in_place(
                    output,
                    layer_state.norm_scale.as_ptr(),
                    layer_state.norm_bias.as_ptr(),
                    layer.output_time as c_int,
                    layer.output_channels as c_int,
                ),
                "post normalization",
            )?;
        }
        Ok(())
    }

    /// Decodes one `[codebooks, frame_length]` frame into planar PCM.
    pub fn decode_frame(&mut self, codes: &[u16], scale: f32) -> Result<Vec<f32>> {
        self.run_front(codes)?;

        let post_out_address = self.post_out.as_ptr() as usize;
        let mut current = self.front_activation2.as_mut_ptr();

        for layer_index in 0..self.metadata.layers.len() {
            let layer = self.metadata.layers[layer_index].clone();
            let current_address = current as usize;
            let (raw, cropped) = if current_address == post_out_address {
                (self.post_in.as_mut_ptr(), self.post_out.as_mut_ptr())
            } else {
                (self.post_out.as_mut_ptr(), self.post_in.as_mut_ptr())
            };

            let layer_state = &self.layers[layer_index];
            let (packed, bias, norm_scale, norm_bias) = (
                layer_state.packed.as_ptr(),
                layer_state.bias.as_ptr(),
                layer_state.norm_scale.as_ptr(),
                layer_state.norm_bias.as_ptr(),
            );
            unsafe {
                check(
                    conv_transpose1d_phase_simd_8x8_nhwc(
                        current,
                        packed,
                        bias,
                        raw,
                        layer.input_time as c_int,
                        layer.input_channels as c_int,
                        layer.output_channels as c_int,
                        layer.stride as c_int,
                    ),
                    "transpose layer",
                )?;
                check(
                    group_norm_nhwc_in_place(
                        raw,
                        norm_scale,
                        norm_bias,
                        layer.raw_output_time as c_int,
                        layer.output_channels as c_int,
                    ),
                    "transpose normalization",
                )?;
                check(
                    crop_nhwc(
                        raw,
                        cropped,
                        layer.raw_output_time as c_int,
                        layer.output_channels as c_int,
                        layer.crop_left as c_int,
                        layer.crop_right as c_int,
                    ),
                    "transpose crop",
                )?;
            }

            let block = self.metadata.post.blocks[layer_index];
            let shortcut = self.post_shortcut.as_mut_ptr();
            let reduced = self.post_reduced.as_mut_ptr();
            self.run_post_conv(block[0], cropped, shortcut, false)?;
            self.run_post_conv(block[1], cropped, reduced, true)?;
            self.run_post_conv(block[2], self.post_reduced.as_ptr(), raw, true)?;
            unsafe {
                check(
                    add_elu_nhwc_in_place(
                        raw,
                        self.post_shortcut.as_ptr(),
                        (layer.cropped_output_time * layer.output_channels) as c_int,
                    ),
                    "residual add",
                )?;
            }
            current = raw;
        }

        let final_conv = self.metadata.post.final_conv.clone();
        let shortcut = self.post_shortcut.as_mut_ptr();
        let reduced = self.post_reduced.as_mut_ptr();
        unsafe {
            check(
                reflect_pad_nhwc(
                    current,
                    self.post_padded.as_mut_ptr(),
                    final_conv.input_time as c_int,
                    final_conv.input_channels as c_int,
                    final_conv.padding_left as c_int,
                    final_conv.padding_right as c_int,
                ),
                "final padding",
            )?;
            check(
                conv1d_nhwc_simd_8x8(
                    self.post_padded.as_ptr(),
                    self.post_final.packed.as_ptr(),
                    self.post_final.bias.as_ptr(),
                    shortcut,
                    final_conv.padded_input_time as c_int,
                    final_conv.input_channels as c_int,
                    final_conv.kernel_output_channels as c_int,
                    final_conv.kernel as c_int,
                    final_conv.stride as c_int,
                ),
                "final convolution",
            )?;
            check(
                compact_nhwc_channels(
                    self.post_shortcut.as_ptr(),
                    reduced,
                    final_conv.output_time as c_int,
                    final_conv.kernel_output_channels as c_int,
                    final_conv.output_channels as c_int,
                ),
                "final channel compaction",
            )?;
            check(
                group_norm_nhwc_in_place(
                    reduced,
                    self.post_final.norm_scale.as_ptr(),
                    self.post_final.norm_bias.as_ptr(),
                    final_conv.output_time as c_int,
                    final_conv.output_channels as c_int,
                ),
                "final normalization",
            )?;
            check(
                scale_nhwc_to_nct(
                    self.post_reduced.as_ptr(),
                    self.planar_output.as_mut_ptr(),
                    scale,
                    final_conv.output_time as c_int,
                    final_conv.output_channels as c_int,
                ),
                "final scaling",
            )?;
        }

        let samples = self.metadata.channels * self.metadata.segment_samples;
        if self.planar_output.len() < samples {
            bail!(
                "native decoder produced {} planar samples; expected {samples}",
                self.planar_output.len()
            );
        }
        Ok(self.planar_output[..samples].to_vec())
    }
}

#[derive(Debug, Clone, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EncoderMetadata {
    pub sample_rate: usize,
    pub channels: usize,
    pub segment_samples: usize,
    pub segment_stride: usize,
    pub frame_length: usize,
    pub num_codebooks: usize,
    pub conv_layers: Vec<EncoderConvLayer>,
    pub lstm_layers: Vec<EncoderLstmLayer>,
    pub rvq: EncoderRvq,
}

#[derive(Debug, Clone, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EncoderConvLayer {
    pub layer: usize,
    pub input_channels: usize,
    pub output_channels: usize,
    pub kernel: usize,
    pub stride: usize,
    pub input_time: usize,
    pub padding_left: usize,
    pub padding_right: usize,
    pub padded_input_time: usize,
    pub output_time: usize,
}

#[derive(Debug, Clone, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EncoderLstmLayer {
    pub layer: usize,
    pub input_size: usize,
    pub hidden_size: usize,
    pub gate_size: usize,
}

#[derive(Debug, Clone, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EncoderRvq {
    pub codebooks: usize,
    pub entries: usize,
    pub dimension: usize,
}

struct EncoderConv {
    packed: Vec<f32>,
    bias: Vec<f32>,
    norm_scale: Vec<f32>,
    norm_bias: Vec<f32>,
}

struct EncoderLstm {
    packed_input: Vec<f32>,
    packed_recurrent: Vec<f32>,
    bias: Vec<f32>,
}

fn unused_activation(used: &[usize]) -> usize {
    (0..3)
        .find(|index| !used.contains(index))
        .expect("three activation buffers cover every residual step")
}

/// A prepared native encoder for one bundle profile.
pub struct NativeEncodecEncoder {
    metadata: EncoderMetadata,
    embeddings: Vec<f32>,
    norms: Vec<f32>,
    audio: Vec<f32>,
    normalized: Vec<f32>,
    padded: Vec<f32>,
    activations: [Vec<f32>; 3],
    lstm_hidden: Vec<f32>,
    lstm_cell: Vec<f32>,
    lstm_input_projection: Vec<f32>,
    rvq_residual: Vec<f32>,
    rvq_codes: Vec<i32>,
    conv_layers: Vec<EncoderConv>,
    lstm_layers: Vec<EncoderLstm>,
}

impl NativeEncodecEncoder {
    /// Loads `metadata.json` and the encoder weights from an encoder asset dir.
    pub fn from_dir(dir: impl AsRef<Path>) -> Result<Self> {
        let dir = dir.as_ref();
        let metadata_path = dir.join("metadata.json");
        let metadata: EncoderMetadata = serde_json::from_str(
            &std::fs::read_to_string(&metadata_path)
                .with_context(|| format!("failed to read {}", metadata_path.display()))?,
        )
        .with_context(|| format!("failed to parse {}", metadata_path.display()))?;
        let weights = WeightSource::open(dir)?;
        Self::new(metadata, &weights)
    }

    fn new(metadata: EncoderMetadata, weights: &WeightSource) -> Result<Self> {
        if metadata.conv_layers.len() != 18 {
            bail!("native encoder must have eighteen convolution layers");
        }
        if metadata.lstm_layers.len() != 2 {
            bail!("native encoder must have two LSTM layers");
        }
        let frame_length = metadata.frame_length;
        let hidden_size = metadata.lstm_layers[0].hidden_size;
        let gate_size = metadata.lstm_layers[0].gate_size;

        let embeddings = weights.get("rvq-embeddings.f32le")?;
        let norms = weights.get("rvq-norms.f32le")?;

        let mut conv_layers = Vec::with_capacity(metadata.conv_layers.len());
        for layer in &metadata.conv_layers {
            let raw = weights.get(&format!("conv-{}-weight.f32le", layer.layer))?;
            let input_channels = layer.input_channels as c_int;
            let output_channels = layer.output_channels as c_int;
            let kernel = layer.kernel as c_int;
            let packed = packed_or_error(
                raw,
                move |weights, packed| unsafe {
                    pack_conv1d_nhwc_weights_8(
                        weights,
                        packed,
                        input_channels,
                        output_channels,
                        kernel,
                    )
                },
                "encoder convolution weight packing",
            )?;
            conv_layers.push(EncoderConv {
                packed,
                bias: weights.get(&format!("conv-{}-bias.f32le", layer.layer))?,
                norm_scale: weights.get(&format!("conv-{}-norm-scale.f32le", layer.layer))?,
                norm_bias: weights.get(&format!("conv-{}-norm-bias.f32le", layer.layer))?,
            });
        }

        let mut lstm_layers = Vec::with_capacity(2);
        for layer in &metadata.lstm_layers {
            let input_weights =
                weights.get(&format!("lstm-{}-input-weight.f32le", layer.layer))?;
            let recurrent_weights =
                weights.get(&format!("lstm-{}-recurrent-weight.f32le", layer.layer))?;
            let bias = weights.get(&format!("lstm-{}-bias.f32le", layer.layer))?;
            let input_size = layer.hidden_size as c_int;
            let gate = layer.gate_size as c_int;
            let packed_input = packed_or_error(
                input_weights,
                move |weights, packed| unsafe {
                    pack_linear_weights_8(weights, packed, input_size, gate)
                },
                "encoder LSTM input weight packing",
            )?;
            let packed_recurrent = packed_or_error(
                recurrent_weights,
                move |weights, packed| unsafe {
                    pack_linear_weights_8(weights, packed, input_size, gate)
                },
                "encoder LSTM recurrent weight packing",
            )?;
            lstm_layers.push(EncoderLstm {
                packed_input,
                packed_recurrent,
                bias,
            });
        }

        let max_activation = metadata
            .conv_layers
            .iter()
            .map(|layer| layer.output_time * layer.output_channels)
            .max()
            .unwrap_or(1);
        let max_padded = metadata
            .conv_layers
            .iter()
            .map(|layer| layer.padded_input_time * layer.input_channels)
            .max()
            .unwrap_or(1);
        let audio_length = metadata.channels * metadata.segment_samples;
        let rvq_code_count = metadata.rvq.codebooks * frame_length;
        let rvq_residual_length = frame_length * metadata.rvq.dimension;

        Ok(Self {
            metadata,
            embeddings,
            norms,
            audio: vec![0.0; audio_length],
            normalized: vec![0.0; audio_length],
            padded: vec![0.0; max_padded],
            activations: [
                vec![0.0; max_activation],
                vec![0.0; max_activation],
                vec![0.0; max_activation],
            ],
            lstm_hidden: vec![0.0; hidden_size],
            lstm_cell: vec![0.0; hidden_size],
            lstm_input_projection: vec![0.0; frame_length * gate_size],
            rvq_residual: vec![0.0; rvq_residual_length],
            rvq_codes: vec![0; rvq_code_count],
            conv_layers,
            lstm_layers,
        })
    }

    pub fn metadata(&self) -> &EncoderMetadata {
        &self.metadata
    }

    fn run_layer(
        &mut self,
        index: usize,
        input: *const f32,
        output: *mut f32,
        apply_elu: bool,
    ) -> Result<()> {
        let (input_time, input_channels, padding_left, padding_right, padded_input_time, output_time, output_channels, kernel, stride) = {
            let layer = &self.metadata.conv_layers[index];
            (
                layer.input_time,
                layer.input_channels,
                layer.padding_left,
                layer.padding_right,
                layer.padded_input_time,
                layer.output_time,
                layer.output_channels,
                layer.kernel,
                layer.stride,
            )
        };
        let mut convolution_input = input;
        unsafe {
            if padding_left != 0 || padding_right != 0 {
                let status = if apply_elu {
                    reflect_pad_elu_nhwc(
                        input,
                        self.padded.as_mut_ptr(),
                        input_time as c_int,
                        input_channels as c_int,
                        padding_left as c_int,
                        padding_right as c_int,
                    )
                } else {
                    reflect_pad_nhwc(
                        input,
                        self.padded.as_mut_ptr(),
                        input_time as c_int,
                        input_channels as c_int,
                        padding_left as c_int,
                        padding_right as c_int,
                    )
                };
                check(status, "encoder padding")?;
                convolution_input = self.padded.as_ptr();
            } else if apply_elu {
                check(
                    elu_nhwc_in_place(input as *mut f32, (input_time * input_channels) as c_int),
                    "encoder ELU",
                )?;
            }

            let layer_state = &self.conv_layers[index];
            let (packed, bias, norm_scale, norm_bias) = (
                layer_state.packed.as_ptr(),
                layer_state.bias.as_ptr(),
                layer_state.norm_scale.as_ptr(),
                layer_state.norm_bias.as_ptr(),
            );
            check(
                conv1d_nhwc_simd_8x8(
                    convolution_input,
                    packed,
                    bias,
                    output,
                    padded_input_time as c_int,
                    input_channels as c_int,
                    output_channels as c_int,
                    kernel as c_int,
                    stride as c_int,
                ),
                "encoder convolution",
            )?;
            check(
                group_norm_nhwc_in_place(
                    output,
                    norm_scale,
                    norm_bias,
                    output_time as c_int,
                    output_channels as c_int,
                ),
                "encoder normalization",
            )?;
        }
        Ok(())
    }

    fn run_lstm(&mut self, index: usize, input: *const f32, output: *mut f32) -> Result<()> {
        let frame_length = self.metadata.frame_length;
        let hidden_size = self.metadata.lstm_layers[index].hidden_size;
        let layer = &self.lstm_layers[index];
        let (packed_input, packed_recurrent, bias) = (
            layer.packed_input.as_ptr(),
            layer.packed_recurrent.as_ptr(),
            layer.bias.as_ptr(),
        );
        unsafe {
            check(
                lstm_layer_simd_64(
                    input,
                    packed_input,
                    packed_recurrent,
                    bias,
                    output,
                    self.lstm_hidden.as_mut_ptr(),
                    self.lstm_cell.as_mut_ptr(),
                    self.lstm_input_projection.as_mut_ptr(),
                    frame_length as c_int,
                    hidden_size as c_int,
                ),
                "encoder LSTM",
            )?;
        }
        Ok(())
    }

    /// Encodes one planar `[channel][segment_samples]` model window.
    pub fn encode(&mut self, audio: &[f32]) -> Result<(Vec<u16>, f32)> {
        let channels = self.metadata.channels;
        let segment_samples = self.metadata.segment_samples;
        if audio.len() != channels * segment_samples {
            bail!(
                "encoder input has {} values; expected {channels} channels by {segment_samples} samples",
                audio.len()
            );
        }
        self.audio.copy_from_slice(audio);
        let scale = unsafe {
            normalize_audio_planar_to_nhwc(
                self.audio.as_ptr(),
                self.normalized.as_mut_ptr(),
                segment_samples as c_int,
                channels as c_int,
            )
        };

        {
            let input = self.normalized.as_ptr();
            let output = self.activations[0].as_mut_ptr();
            self.run_layer(0, input, output, false)?;
        }
        let mut current = 0usize;
        for (shortcut_layer, first_main, second_main, downsample) in [
            (1usize, 2usize, 3usize, 4usize),
            (5, 6, 7, 8),
            (9, 10, 11, 12),
            (13, 14, 15, 16),
        ] {
            let shortcut = unused_activation(&[current]);
            let main = unused_activation(&[current, shortcut]);
            {
                let input = self.activations[current].as_ptr();
                let output = self.activations[shortcut].as_mut_ptr();
                self.run_layer(shortcut_layer, input, output, false)?;
            }
            {
                let input = self.activations[current].as_ptr();
                let output = self.activations[main].as_mut_ptr();
                self.run_layer(first_main, input, output, true)?;
            }
            let first_len = self.metadata.conv_layers[first_main].output_time
                * self.metadata.conv_layers[first_main].output_channels;
            {
                let output = self.activations[main].as_mut_ptr();
                check(
                    unsafe { elu_nhwc_in_place(output, first_len as c_int) },
                    "encoder residual ELU",
                )?;
            }
            {
                let input = self.activations[main].as_ptr();
                let output = self.activations[current].as_mut_ptr();
                self.run_layer(second_main, input, output, false)?;
            }
            let second_len = self.metadata.conv_layers[second_main].output_time
                * self.metadata.conv_layers[second_main].output_channels;
            {
                let destination = self.activations[current].as_mut_ptr();
                let source = self.activations[shortcut].as_ptr();
                check(
                    unsafe { add_nhwc_in_place(destination, source, second_len as c_int) },
                    "encoder residual add",
                )?;
            }
            {
                let input = self.activations[current].as_ptr();
                let output = self.activations[shortcut].as_mut_ptr();
                self.run_layer(downsample, input, output, true)?;
            }
            current = shortcut;
        }

        let first_output = unused_activation(&[current]);
        let second_output = unused_activation(&[current, first_output]);
        {
            let input = self.activations[current].as_ptr();
            let output = self.activations[first_output].as_mut_ptr();
            self.run_lstm(0, input, output)?;
        }
        {
            let input = self.activations[first_output].as_ptr();
            let output = self.activations[second_output].as_mut_ptr();
            self.run_lstm(1, input, output)?;
        }
        let frame_length = self.metadata.frame_length;
        let hidden_size = self.metadata.lstm_layers[0].hidden_size;
        {
            let destination = self.activations[second_output].as_mut_ptr();
            let source = self.activations[current].as_ptr();
            check(
                unsafe {
                    add_nhwc_in_place(destination, source, (frame_length * hidden_size) as c_int)
                },
                "encoder LSTM residual add",
            )?;
        }
        {
            let input = self.activations[second_output].as_ptr();
            let output = self.activations[first_output].as_mut_ptr();
            self.run_layer(17, input, output, true)?;
        }

        let dimension = self.metadata.rvq.dimension;
        let entries = self.metadata.rvq.entries;
        let codebooks = self.metadata.rvq.codebooks;
        unsafe {
            check(
                rvq_encode_simd_8(
                    self.activations[first_output].as_ptr(),
                    self.rvq_residual.as_mut_ptr(),
                    self.embeddings.as_ptr(),
                    self.norms.as_ptr(),
                    self.rvq_codes.as_mut_ptr(),
                    frame_length as c_int,
                    dimension as c_int,
                    entries as c_int,
                    codebooks as c_int,
                ),
                "encoder residual vector quantizer",
            )?;
        }
        let codes = self.rvq_codes.iter().map(|value| *value as u16).collect();
        Ok((codes, scale))
    }
}

/// A [`FrameCodec`] whose neural encoder and decoder are the native C kernel.
pub struct NativeFrameCodec {
    bundle: FrameBundleMetadata,
    decoder: NativeEncodecDecoder,
    encoder: Option<NativeEncodecEncoder>,
}

impl NativeFrameCodec {
    /// Opens a bundle directory that holds `bundle.json`, the q8 LM weights,
    /// and a `decoder/` asset directory.
    pub fn from_bundle_dir(bundle_dir: impl AsRef<Path>) -> Result<Self> {
        let bundle_dir = bundle_dir.as_ref();
        let bundle_path = bundle_dir.join("bundle.json");
        let bundle: FrameBundleMetadata = serde_json::from_str(
            &std::fs::read_to_string(&bundle_path)
                .with_context(|| format!("failed to read {}", bundle_path.display()))?,
        )
        .with_context(|| format!("failed to parse {}", bundle_path.display()))?;
        let decoder = NativeEncodecDecoder::from_dir(bundle_dir.join("decoder"))?;
        if decoder.metadata().num_codebooks != bundle.num_codebooks
            || decoder.metadata().frame_length != bundle.frame_length
            || decoder.metadata().channels != bundle.channels
            || decoder.metadata().segment_samples != bundle.segment_samples
        {
            bail!(
                "decoder metadata at {} does not match bundle.json",
                bundle_dir.join("decoder").display()
            );
        }
        Ok(Self {
            bundle,
            decoder,
            encoder: None,
        })
    }

    /// Opens a bundle directory and loads the native encoder as well.
    pub fn from_bundle_dir_with_encoder(bundle_dir: impl AsRef<Path>) -> Result<Self> {
        let mut codec = Self::from_bundle_dir(&bundle_dir)?;
        let encoder = NativeEncodecEncoder::from_dir(bundle_dir.as_ref().join("encoder"))?;
        if encoder.metadata().num_codebooks != codec.bundle.num_codebooks
            || encoder.metadata().frame_length != codec.bundle.frame_length
            || encoder.metadata().channels != codec.bundle.channels
            || encoder.metadata().segment_samples != codec.bundle.segment_samples
        {
            bail!(
                "encoder metadata at {} does not match bundle.json",
                bundle_dir.as_ref().join("encoder").display()
            );
        }
        codec.encoder = Some(encoder);
        Ok(codec)
    }

    pub fn decoder(&self) -> &NativeEncodecDecoder {
        &self.decoder
    }

    pub fn encoder(&self) -> Option<&NativeEncodecEncoder> {
        self.encoder.as_ref()
    }
}

impl FrameCodec for NativeFrameCodec {
    fn metadata(&self) -> &FrameBundleMetadata {
        &self.bundle
    }

    fn encode_frame(&mut self, audio: &Array3<f32>) -> Result<(Array3<i64>, Array2<f32>)> {
        let (batch, channels, samples) = audio.dim();
        if channels != self.bundle.channels || samples != self.bundle.segment_samples {
            bail!(
                "encoder input shape {batch}x{channels}x{samples} does not match the bundle",
            );
        }
        let encoder = self
            .encoder
            .as_mut()
            .context("the native kernel codec has no encoder loaded")?;
        let slice = audio
            .as_slice_memory_order()
            .context("encoder audio is not contiguous")?;
        let mut codes = Vec::with_capacity(batch * self.bundle.num_codebooks * self.bundle.frame_length);
        let mut scales = Vec::with_capacity(batch);
        for item in 0..batch {
            let start = item * channels * samples;
            let (frame_codes, scale) = encoder.encode(&slice[start..start + channels * samples])?;
            codes.extend(frame_codes.iter().map(|value| *value as i64));
            scales.push(scale);
        }
        let codes = Array3::from_shape_vec(
            (batch, self.bundle.num_codebooks, self.bundle.frame_length),
            codes,
        )
        .context("encoded code shape does not match the bundle")?;
        let scales = Array2::from_shape_vec((batch, 1), scales)
            .context("encoded scale shape does not match the batch")?;
        Ok((codes, scales))
    }


    fn decode_frame(&mut self, codes: &Array3<i64>, scale: &Array2<f32>) -> Result<Array3<f32>> {
        let (batch, codebooks, frame_length) = codes.dim();
        if codebooks != self.bundle.num_codebooks || frame_length != self.bundle.frame_length {
            bail!(
                "decoder code shape {batch}x{codebooks}x{frame_length} does not match the bundle",
            );
        }
        if scale.nrows() != batch {
            bail!(
                "decoder scale batch {} does not match code batch {batch}",
                scale.nrows()
            );
        }
        let code_slice = codes
            .as_slice_memory_order()
            .context("decoder codes are not contiguous")?;
        let scale_slice = scale
            .as_slice_memory_order()
            .context("decoder scales are not contiguous")?;
        let samples = self.bundle.channels * self.bundle.segment_samples;
        let mut audio = Vec::with_capacity(batch * samples);
        for item in 0..batch {
            let start = item * codebooks * frame_length;
            let frame = &code_slice[start..start + codebooks * frame_length];
            let frame_codes: Vec<u16> = frame.iter().map(|value| *value as u16).collect();
            let frame_scale = scale_slice[item * scale.ncols()];
            audio.extend_from_slice(&self.decoder.decode_frame(&frame_codes, frame_scale)?);
        }
        Array3::from_shape_vec(
            (batch, self.bundle.channels, self.bundle.segment_samples),
            audio,
        )
        .context("decoded audio shape does not match the bundle")
    }
}
