use anyhow::{bail, Result};

const MAGIC: &[u8; 8] = b"ELMW0001";
const HEADER_U32S: usize = 7;
const LAYER_NORM_EPS: f64 = 1.0e-5;
const SQRT_2: f64 = 1.414_213_562_373_095_1;

#[derive(Clone, Debug)]
pub struct DeterministicLmWeights {
    pub dim: usize,
    pub layers: usize,
    pub heads: usize,
    pub codebooks: usize,
    pub cardinality: usize,
    pub frame_length: usize,
    pub past_context: usize,
    hidden_dim: usize,
    norm_in_weight: Vec<f32>,
    norm_in_bias: Vec<f32>,
    pos_emb: Vec<f32>,
    layer_weights: Vec<TransformerLayerWeights>,
    embeddings: Vec<Vec<f32>>,
    output_weights: Vec<Vec<f32>>,
    output_biases: Vec<Vec<f32>>,
}

#[derive(Clone, Debug)]
struct TransformerLayerWeights {
    in_proj_weight: Vec<f32>,
    in_proj_bias: Vec<f32>,
    out_proj_weight: Vec<f32>,
    out_proj_bias: Vec<f32>,
    linear1_weight: Vec<f32>,
    linear1_bias: Vec<f32>,
    linear2_weight: Vec<f32>,
    linear2_bias: Vec<f32>,
    norm1_weight: Vec<f32>,
    norm1_bias: Vec<f32>,
    norm2_weight: Vec<f32>,
    norm2_bias: Vec<f32>,
}

#[derive(Clone, Debug)]
pub struct DeterministicLmState {
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
pub struct DeterministicLm {
    weights: DeterministicLmWeights,
    scratch: DeterministicLmScratch,
}

#[derive(Debug, Default)]
struct DeterministicLmScratch {
    x: Vec<f32>,
    y: Vec<f32>,
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    attn: Vec<f32>,
    ff: Vec<f32>,
    scores: Vec<f64>,
}

impl DeterministicLmWeights {
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        let mut reader = WeightReader { bytes, pos: 0 };
        let magic = reader.read_bytes(MAGIC.len())?;
        if magic != MAGIC {
            bail!("invalid deterministic LM weight file magic");
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
            bail!("deterministic LM weight header contains a zero dimension");
        }
        if dim % heads != 0 {
            bail!("deterministic LM dim {dim} is not divisible by heads {heads}");
        }
        let hidden_dim = dim * 4;

        let norm_in_weight = reader.read_f32_vec(dim)?;
        let norm_in_bias = reader.read_f32_vec(dim)?;
        let pos_emb = reader.read_f32_vec(frame_length * dim)?;

        let mut layer_weights = Vec::with_capacity(layers);
        for _ in 0..layers {
            layer_weights.push(TransformerLayerWeights {
                in_proj_weight: reader.read_f32_vec(3 * dim * dim)?,
                in_proj_bias: reader.read_f32_vec(3 * dim)?,
                out_proj_weight: reader.read_f32_vec(dim * dim)?,
                out_proj_bias: reader.read_f32_vec(dim)?,
                linear1_weight: reader.read_f32_vec(hidden_dim * dim)?,
                linear1_bias: reader.read_f32_vec(hidden_dim)?,
                linear2_weight: reader.read_f32_vec(dim * hidden_dim)?,
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
            output_weights.push(reader.read_f32_vec(cardinality * dim)?);
            output_biases.push(reader.read_f32_vec(cardinality)?);
        }

        if reader.remaining() != 0 {
            bail!(
                "deterministic LM weight file has {} trailing bytes",
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
            hidden_dim,
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
                "deterministic LM weights contain {} codebooks, but {} were requested",
                self.codebooks,
                codebooks
            );
        }
        Ok(())
    }
}

impl DeterministicLm {
    pub fn new(weights: DeterministicLmWeights) -> Self {
        Self {
            weights,
            scratch: DeterministicLmScratch::default(),
        }
    }

    pub fn weights(&self) -> &DeterministicLmWeights {
        &self.weights
    }

    pub fn initial_state(&self) -> DeterministicLmState {
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
        DeterministicLmState { offset: 0, layers }
    }

    pub fn forward_step(
        &mut self,
        state: &mut DeterministicLmState,
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

    fn forward_layer(
        &mut self,
        state: &mut DeterministicLmState,
        layer_index: usize,
    ) -> Result<()> {
        let dim = self.weights.dim;
        let hidden_dim = self.weights.hidden_dim;
        let layer = &self.weights.layer_weights[layer_index];
        let layer_state = &mut state.layers[layer_index];
        if layer_state.len > self.weights.past_context + 1 {
            bail!("LM layer state exceeded past_context");
        }

        linear_row_major_part(
            &self.scratch.x,
            &layer.in_proj_weight,
            &layer.in_proj_bias,
            0,
            dim,
            dim,
            &mut self.scratch.q,
        );
        linear_row_major_part(
            &self.scratch.x,
            &layer.in_proj_weight,
            &layer.in_proj_bias,
            dim,
            dim,
            dim,
            &mut self.scratch.k,
        );
        linear_row_major_part(
            &self.scratch.x,
            &layer.in_proj_weight,
            &layer.in_proj_bias,
            2 * dim,
            dim,
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
        linear_row_major(
            &self.scratch.attn,
            &layer.out_proj_weight,
            &layer.out_proj_bias,
            dim,
            dim,
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

        linear_row_major(
            &self.scratch.x,
            &layer.linear1_weight,
            &layer.linear1_bias,
            hidden_dim,
            dim,
            &mut self.scratch.ff,
        );
        for value in &mut self.scratch.ff {
            *value = gelu(*value as f64) as f32;
        }
        linear_row_major(
            &self.scratch.ff,
            &layer.linear2_weight,
            &layer.linear2_bias,
            dim,
            hidden_dim,
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
        let dim = self.weights.dim;
        let card = self.weights.cardinality;
        let mut logits = vec![0.0_f32; card * codebooks];
        for codebook in 0..codebooks {
            let weight = &self.weights.output_weights[codebook];
            let bias = &self.weights.output_biases[codebook];
            for bin in 0..card {
                let mut acc = bias[bin] as f64;
                let base = bin * dim;
                for d in 0..dim {
                    acc += (weight[base + d] as f64) * (self.scratch.x[d] as f64);
                }
                logits[bin * codebooks + codebook] = acc as f32;
            }
        }
        logits
    }
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

fn linear_row_major(
    input: &[f32],
    weight: &[f32],
    bias: &[f32],
    out_dim: usize,
    in_dim: usize,
    out: &mut Vec<f32>,
) {
    linear_row_major_part(input, weight, bias, 0, out_dim, in_dim, out);
}

fn linear_row_major_part(
    input: &[f32],
    weight: &[f32],
    bias: &[f32],
    row_offset: usize,
    out_dim: usize,
    in_dim: usize,
    out: &mut Vec<f32>,
) {
    out.resize(out_dim, 0.0);
    for row in 0..out_dim {
        let source_row = row_offset + row;
        let mut acc = bias[source_row] as f64;
        let base = source_row * in_dim;
        for col in 0..in_dim {
            acc += (weight[base + col] as f64) * (input[col] as f64);
        }
        out[row] = acc as f32;
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
            .ok_or_else(|| anyhow::anyhow!("deterministic LM weight offset overflow"))?;
        let bytes = self
            .bytes
            .get(self.pos..end)
            .ok_or_else(|| anyhow::anyhow!("deterministic LM weight file ended early"))?;
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

    fn remaining(&self) -> usize {
        self.bytes.len().saturating_sub(self.pos)
    }
}
