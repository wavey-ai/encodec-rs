use anyhow::{bail, Result};

const MAGIC: &[u8; 8] = b"ELMQ0001";
const HEADER_U32S: usize = 7;
const LAYER_NORM_EPS: f64 = 1.0e-5;
const SQRT_2: f64 = 1.414_213_562_373_095_1;

#[derive(Clone, Debug)]
pub struct QuantizedLmWeights {
    pub dim: usize,
    pub layers: usize,
    pub heads: usize,
    pub codebooks: usize,
    pub cardinality: usize,
    pub frame_length: usize,
    pub past_context: usize,
    norm_in_weight: Vec<f32>,
    norm_in_bias: Vec<f32>,
    pos_emb: Vec<f32>,
    layer_weights: Vec<QuantizedTransformerLayerWeights>,
    embeddings: Vec<Vec<f32>>,
    output_weights: Vec<QuantizedLinear>,
    output_biases: Vec<Vec<f32>>,
}

#[derive(Clone, Debug)]
struct QuantizedTransformerLayerWeights {
    in_proj_weight: QuantizedLinear,
    in_proj_bias: Vec<f32>,
    out_proj_weight: QuantizedLinear,
    out_proj_bias: Vec<f32>,
    linear1_weight: QuantizedLinear,
    linear1_bias: Vec<f32>,
    linear2_weight: QuantizedLinear,
    linear2_bias: Vec<f32>,
    norm1_weight: Vec<f32>,
    norm1_bias: Vec<f32>,
    norm2_weight: Vec<f32>,
    norm2_bias: Vec<f32>,
}

#[derive(Clone, Debug)]
struct QuantizedLinear {
    rows: usize,
    cols: usize,
    scales: Vec<f32>,
    weights: Vec<i8>,
}

#[derive(Clone, Debug)]
pub struct QuantizedLmState {
    offset: usize,
    layers: Vec<LayerState>,
}

#[derive(Clone, Debug)]
struct LayerState {
    keys: Vec<f32>,
    values: Vec<f32>,
    len: usize,
}

#[derive(Debug)]
pub struct QuantizedLm {
    weights: QuantizedLmWeights,
    scratch: QuantizedLmScratch,
}

#[derive(Debug, Default)]
struct QuantizedLmScratch {
    x: Vec<f32>,
    y: Vec<f32>,
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    attn: Vec<f32>,
    ff: Vec<f32>,
    scores: Vec<f64>,
    input_q: Vec<i16>,
}

impl QuantizedLmWeights {
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        let mut reader = WeightReader { bytes, pos: 0 };
        let magic = reader.read_bytes(MAGIC.len())?;
        if magic != MAGIC {
            bail!("invalid quantized LM weight file magic");
        }

        let mut header = [0_u32; HEADER_U32S];
        for value in &mut header {
            *value = reader.read_u32()?;
        }
        let dim = header[0] as usize;
        let layers = header[1] as usize;
        let heads = header[2] as usize;
        let codebooks = header[3] as usize;
        let cardinality = header[4] as usize;
        let frame_length = header[5] as usize;
        let past_context = header[6] as usize;
        if dim == 0 || layers == 0 || heads == 0 || codebooks == 0 || cardinality == 0 {
            bail!("quantized LM weight header contains a zero dimension");
        }
        if dim % heads != 0 {
            bail!("quantized LM dim {dim} is not divisible by heads {heads}");
        }
        let hidden_dim = dim * 4;

        let norm_in_weight = reader.read_f32_vec(dim)?;
        let norm_in_bias = reader.read_f32_vec(dim)?;
        let pos_emb = reader.read_f32_vec(frame_length * dim)?;

        let mut layer_weights = Vec::with_capacity(layers);
        for _ in 0..layers {
            layer_weights.push(QuantizedTransformerLayerWeights {
                in_proj_weight: reader.read_quantized_linear(3 * dim, dim)?,
                in_proj_bias: reader.read_f32_vec(3 * dim)?,
                out_proj_weight: reader.read_quantized_linear(dim, dim)?,
                out_proj_bias: reader.read_f32_vec(dim)?,
                linear1_weight: reader.read_quantized_linear(hidden_dim, dim)?,
                linear1_bias: reader.read_f32_vec(hidden_dim)?,
                linear2_weight: reader.read_quantized_linear(dim, hidden_dim)?,
                linear2_bias: reader.read_f32_vec(dim)?,
                norm1_weight: reader.read_f32_vec(dim)?,
                norm1_bias: reader.read_f32_vec(dim)?,
                norm2_weight: reader.read_f32_vec(dim)?,
                norm2_bias: reader.read_f32_vec(dim)?,
            });
        }

        let mut embeddings = Vec::with_capacity(codebooks);
        for _ in 0..codebooks {
            embeddings.push(reader.read_f32_vec((cardinality + 1) * dim)?);
        }

        let mut output_weights = Vec::with_capacity(codebooks);
        let mut output_biases = Vec::with_capacity(codebooks);
        for _ in 0..codebooks {
            output_weights.push(reader.read_quantized_linear(cardinality, dim)?);
            output_biases.push(reader.read_f32_vec(cardinality)?);
        }

        if reader.remaining() != 0 {
            bail!(
                "quantized LM weight file has {} trailing bytes",
                reader.remaining()
            );
        }

        Ok(Self {
            dim,
            layers,
            heads,
            codebooks,
            cardinality,
            frame_length,
            past_context,
            norm_in_weight,
            norm_in_bias,
            pos_emb,
            layer_weights,
            embeddings,
            output_weights,
            output_biases,
        })
    }

    pub fn validate_for_codebooks(&self, codebooks: usize) -> Result<()> {
        if codebooks > self.codebooks {
            bail!(
                "quantized LM weights contain {} codebooks, but {} were requested",
                self.codebooks,
                codebooks
            );
        }
        Ok(())
    }
}

impl QuantizedLm {
    pub fn new(weights: QuantizedLmWeights) -> Self {
        Self {
            weights,
            scratch: QuantizedLmScratch::default(),
        }
    }

    pub fn initial_state(&self) -> QuantizedLmState {
        let dim = self.weights.dim;
        let mut layers = Vec::with_capacity(self.weights.layers);
        for layer in &self.weights.layer_weights {
            let mut keys = vec![0.0_f32; dim];
            let mut values = vec![0.0_f32; dim];
            keys.copy_from_slice(&layer.in_proj_bias[dim..2 * dim]);
            values.copy_from_slice(&layer.in_proj_bias[2 * dim..3 * dim]);
            layers.push(LayerState {
                keys,
                values,
                len: 1,
            });
        }
        QuantizedLmState { offset: 0, layers }
    }

    pub fn forward_step(
        &mut self,
        state: &mut QuantizedLmState,
        input_symbols: &[usize],
    ) -> Result<Vec<f32>> {
        let dim = self.weights.dim;
        let codebooks = input_symbols.len();
        self.weights.validate_for_codebooks(codebooks)?;
        if state.offset >= self.weights.frame_length {
            bail!(
                "LM offset {} exceeds frame_length {}",
                state.offset,
                self.weights.frame_length
            );
        }
        if state.layers.len() != self.weights.layers {
            bail!(
                "LM state layer count {} does not match weights {}",
                state.layers.len(),
                self.weights.layers
            );
        }

        self.scratch.x.resize(dim, 0.0);
        for value in &mut self.scratch.x {
            *value = 0.0;
        }
        for (codebook, symbol) in input_symbols.iter().copied().enumerate() {
            if symbol > self.weights.cardinality {
                bail!(
                    "LM input symbol {} exceeds cardinality {}",
                    symbol,
                    self.weights.cardinality
                );
            }
            let emb = &self.weights.embeddings[codebook];
            let base = symbol * dim;
            for d in 0..dim {
                self.scratch.x[d] += emb[base + d];
            }
        }

        layer_norm_into(
            &self.scratch.x,
            &self.weights.norm_in_weight,
            &self.weights.norm_in_bias,
            &mut self.scratch.y,
        );
        std::mem::swap(&mut self.scratch.x, &mut self.scratch.y);
        let pos_base = state.offset * dim;
        for d in 0..dim {
            self.scratch.x[d] += self.weights.pos_emb[pos_base + d];
        }

        for layer_index in 0..self.weights.layers {
            self.forward_layer(state, layer_index)?;
        }

        let logits = self.output_logits(codebooks);
        state.offset += 1;
        Ok(logits)
    }

    fn forward_layer(&mut self, state: &mut QuantizedLmState, layer_index: usize) -> Result<()> {
        let dim = self.weights.dim;
        let layer = &self.weights.layer_weights[layer_index];
        let layer_state = &mut state.layers[layer_index];
        if layer_state.len > self.weights.past_context + 1 {
            bail!("LM layer state exceeded past_context");
        }

        let input_scale = quantize_input_i16(&self.scratch.x, &mut self.scratch.input_q);
        quantized_linear_part_with_input(
            &self.scratch.input_q,
            input_scale,
            &layer.in_proj_weight,
            &layer.in_proj_bias,
            0,
            dim,
            &mut self.scratch.q,
        );
        quantized_linear_part_with_input(
            &self.scratch.input_q,
            input_scale,
            &layer.in_proj_weight,
            &layer.in_proj_bias,
            dim,
            dim,
            &mut self.scratch.k,
        );
        quantized_linear_part_with_input(
            &self.scratch.input_q,
            input_scale,
            &layer.in_proj_weight,
            &layer.in_proj_bias,
            2 * dim,
            dim,
            &mut self.scratch.v,
        );

        layer_state.keys.extend_from_slice(&self.scratch.k);
        layer_state.values.extend_from_slice(&self.scratch.v);
        layer_state.len += 1;
        if layer_state.len > self.weights.past_context + 1 {
            let remove = layer_state.len - (self.weights.past_context + 1);
            let remove_values = remove * dim;
            layer_state.keys.drain(0..remove_values);
            layer_state.values.drain(0..remove_values);
            layer_state.len -= remove;
        }

        attention_into(
            &self.scratch.q,
            &layer_state.keys,
            &layer_state.values,
            layer_state.len,
            self.weights.heads,
            &mut self.scratch.attn,
            &mut self.scratch.scores,
        )?;

        quantized_linear(
            &self.scratch.attn,
            &layer.out_proj_weight,
            &layer.out_proj_bias,
            &mut self.scratch.input_q,
            &mut self.scratch.y,
        );
        for d in 0..dim {
            self.scratch.y[d] += self.scratch.x[d];
        }
        layer_norm_into(
            &self.scratch.y,
            &layer.norm1_weight,
            &layer.norm1_bias,
            &mut self.scratch.x,
        );

        quantized_linear(
            &self.scratch.x,
            &layer.linear1_weight,
            &layer.linear1_bias,
            &mut self.scratch.input_q,
            &mut self.scratch.ff,
        );
        for value in &mut self.scratch.ff {
            *value = gelu(*value as f64) as f32;
        }
        quantized_linear(
            &self.scratch.ff,
            &layer.linear2_weight,
            &layer.linear2_bias,
            &mut self.scratch.input_q,
            &mut self.scratch.y,
        );
        for d in 0..dim {
            self.scratch.y[d] += self.scratch.x[d];
        }
        layer_norm_into(
            &self.scratch.y,
            &layer.norm2_weight,
            &layer.norm2_bias,
            &mut self.scratch.x,
        );
        Ok(())
    }

    fn output_logits(&mut self, codebooks: usize) -> Vec<f32> {
        let card = self.weights.cardinality;
        let mut logits = vec![0.0_f32; card * codebooks];
        let input_scale = quantize_input_i16(&self.scratch.x, &mut self.scratch.input_q);
        for codebook in 0..codebooks {
            let weight = &self.weights.output_weights[codebook];
            let bias = &self.weights.output_biases[codebook];
            for bin in 0..card {
                let acc = dot_i8_i16(weight.row(bin), &self.scratch.input_q);
                logits[bin * codebooks + codebook] =
                    bias[bin] + (acc as f32) * input_scale * weight.scales[bin];
            }
        }
        logits
    }
}

impl QuantizedLinear {
    fn row(&self, row: usize) -> &[i8] {
        debug_assert!(row < self.rows);
        let start = row * self.cols;
        &self.weights[start..start + self.cols]
    }
}

fn quantized_linear(
    input: &[f32],
    weight: &QuantizedLinear,
    bias: &[f32],
    input_q: &mut Vec<i16>,
    out: &mut Vec<f32>,
) {
    let input_scale = quantize_input_i16(input, input_q);
    quantized_linear_part_with_input(input_q, input_scale, weight, bias, 0, weight.rows, out);
}

fn quantized_linear_part_with_input(
    input_q: &[i16],
    input_scale: f32,
    weight: &QuantizedLinear,
    bias: &[f32],
    row_offset: usize,
    out_dim: usize,
    out: &mut Vec<f32>,
) {
    debug_assert!(row_offset + out_dim <= weight.rows);
    debug_assert_eq!(input_q.len(), weight.cols);
    out.resize(out_dim, 0.0);
    for row in 0..out_dim {
        let source_row = row_offset + row;
        let acc = dot_i8_i16(weight.row(source_row), input_q);
        out[row] = bias[source_row] + (acc as f32) * input_scale * weight.scales[source_row];
    }
}

fn quantize_input_i16(input: &[f32], out: &mut Vec<i16>) -> f32 {
    out.resize(input.len(), 0);
    let mut max_abs = 0.0_f32;
    for value in input {
        max_abs = max_abs.max(value.abs());
    }
    if !max_abs.is_finite() || max_abs <= 0.0 {
        for value in out.iter_mut() {
            *value = 0;
        }
        return 1.0;
    }
    let scale = max_abs / i16::MAX as f32;
    let inv = 1.0 / scale;
    for (dst, src) in out.iter_mut().zip(input.iter().copied()) {
        *dst = (src * inv).round().clamp(i16::MIN as f32, i16::MAX as f32) as i16;
    }
    scale
}

fn dot_i8_i16(weights: &[i8], input: &[i16]) -> i64 {
    debug_assert_eq!(weights.len(), input.len());
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: the helper only performs unaligned vector loads inside the
        // checked slice bounds and handles any tail elements with scalar code.
        unsafe { dot_i8_i16_aarch64(weights, input) }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        dot_i8_i16_scalar(weights, input)
    }
}

#[cfg(not(target_arch = "aarch64"))]
fn dot_i8_i16_scalar(weights: &[i8], input: &[i16]) -> i64 {
    let mut acc = 0_i64;
    for (w, x) in weights.iter().zip(input.iter()) {
        acc += (*w as i64) * (*x as i64);
    }
    acc
}

#[cfg(target_arch = "aarch64")]
unsafe fn dot_i8_i16_aarch64(weights: &[i8], input: &[i16]) -> i64 {
    use core::arch::aarch64::{
        vaddvq_s32, vget_high_s16, vget_low_s16, vld1_s8, vld1q_s16, vmovl_s8, vmull_s16,
    };

    let len = weights.len().min(input.len());
    let mut index = 0usize;
    let mut acc = 0_i64;
    while index + 8 <= len {
        let w8 = vld1_s8(weights.as_ptr().add(index));
        let w16 = vmovl_s8(w8);
        let x16 = vld1q_s16(input.as_ptr().add(index));
        let lo = vmull_s16(vget_low_s16(w16), vget_low_s16(x16));
        let hi = vmull_s16(vget_high_s16(w16), vget_high_s16(x16));
        acc += vaddvq_s32(lo) as i64;
        acc += vaddvq_s32(hi) as i64;
        index += 8;
    }
    while index < len {
        acc += (*weights.get_unchecked(index) as i64) * (*input.get_unchecked(index) as i64);
        index += 1;
    }
    acc
}

fn layer_norm_into(input: &[f32], weight: &[f32], bias: &[f32], out: &mut Vec<f32>) {
    let dim = input.len();
    out.resize(dim, 0.0);
    let mut mean = 0.0_f64;
    for value in input {
        mean += *value as f64;
    }
    mean /= dim as f64;

    let mut var = 0.0_f64;
    for value in input {
        let delta = *value as f64 - mean;
        var += delta * delta;
    }
    var /= dim as f64;
    let inv_std = 1.0 / (var + LAYER_NORM_EPS).sqrt();

    for i in 0..dim {
        let normalized = (input[i] as f64 - mean) * inv_std;
        out[i] = (normalized * weight[i] as f64 + bias[i] as f64) as f32;
    }
}

fn attention_into(
    query: &[f32],
    keys: &[f32],
    values: &[f32],
    len: usize,
    heads: usize,
    out: &mut Vec<f32>,
    scores: &mut Vec<f64>,
) -> Result<()> {
    let dim = query.len();
    if dim % heads != 0 {
        bail!("attention dim {dim} is not divisible by heads {heads}");
    }
    if keys.len() != len * dim || values.len() != len * dim {
        bail!("attention cache shape mismatch");
    }
    let head_dim = dim / heads;
    let scale = 1.0 / (head_dim as f64).sqrt();
    out.clear();
    out.resize(dim, 0.0);
    scores.resize(len, 0.0);

    for head in 0..heads {
        let head_base = head * head_dim;
        let mut max_score = f64::NEG_INFINITY;
        for t in 0..len {
            let base = t * dim + head_base;
            let mut dot = 0.0_f64;
            for d in 0..head_dim {
                dot += (query[head_base + d] as f64) * (keys[base + d] as f64);
            }
            let score = dot * scale;
            scores[t] = score;
            max_score = max_score.max(score);
        }

        let mut denom = 0.0_f64;
        for score in scores.iter_mut().take(len) {
            let value = libm::exp(*score - max_score);
            *score = value;
            denom += value;
        }
        if !denom.is_finite() || denom <= 0.0 {
            let uniform = 1.0 / len as f64;
            for t in 0..len {
                let base = t * dim + head_base;
                for d in 0..head_dim {
                    out[head_base + d] += (uniform * values[base + d] as f64) as f32;
                }
            }
            continue;
        }

        for t in 0..len {
            let prob = scores[t] / denom;
            let base = t * dim + head_base;
            for d in 0..head_dim {
                out[head_base + d] += (prob * values[base + d] as f64) as f32;
            }
        }
    }
    Ok(())
}

fn gelu(value: f64) -> f64 {
    0.5 * value * (1.0 + libm::erf(value / SQRT_2))
}

struct WeightReader<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> WeightReader<'a> {
    fn read_bytes(&mut self, len: usize) -> Result<&'a [u8]> {
        let end = self
            .pos
            .checked_add(len)
            .ok_or_else(|| anyhow::anyhow!("quantized LM weight offset overflow"))?;
        let bytes = self
            .bytes
            .get(self.pos..end)
            .ok_or_else(|| anyhow::anyhow!("quantized LM weight file ended early"))?;
        self.pos = end;
        Ok(bytes)
    }

    fn read_u32(&mut self) -> Result<u32> {
        let bytes = self.read_bytes(4)?;
        Ok(u32::from_le_bytes(bytes.try_into().expect("slice length")))
    }

    fn read_f32_vec(&mut self, len: usize) -> Result<Vec<f32>> {
        let bytes = self.read_bytes(len * 4)?;
        let mut out = Vec::with_capacity(len);
        for chunk in bytes.chunks_exact(4) {
            out.push(f32::from_le_bytes(chunk.try_into().expect("slice length")));
        }
        Ok(out)
    }

    fn read_quantized_linear(&mut self, rows: usize, cols: usize) -> Result<QuantizedLinear> {
        let scales = self.read_f32_vec(rows)?;
        let bytes = self.read_bytes(rows * cols)?;
        let weights = bytes.iter().map(|value| *value as i8).collect();
        Ok(QuantizedLinear {
            rows,
            cols,
            scales,
            weights,
        })
    }

    fn remaining(&self) -> usize {
        self.bytes.len().saturating_sub(self.pos)
    }
}
