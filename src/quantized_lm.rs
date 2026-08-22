use anyhow::{bail, Result};

use crate::metadata::OnnxFrameBundleMetadata;

const MAGIC: &[u8; 8] = b"ELMQ0001";
const HEADER_U32S: usize = 7;
const LAYER_NORM_EPS: f64 = 1.0e-5;

#[derive(Clone, Debug)]
pub struct QuantizedLmWeights {
    pub dim: usize,
    pub layers: usize,
    pub heads: usize,
    pub codebooks: usize,
    pub cardinality: usize,
    pub frame_length: usize,
    pub past_context: usize,
    attention_scale: f64,
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
    #[cfg(target_arch = "aarch64")]
    row_sums: Vec<i32>,
    #[cfg(target_arch = "aarch64")]
    use_dotprod: bool,
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
    quantized_input: QuantizedInputScratch,
    logit_column: Vec<f32>,
    logits: Vec<f32>,
}

#[derive(Debug, Default)]
struct QuantizedInputScratch {
    values: Vec<i16>,
    #[cfg(target_arch = "aarch64")]
    dotprod_low: Vec<i8>,
    #[cfg(target_arch = "aarch64")]
    dotprod_high: Vec<i8>,
    #[cfg(target_arch = "aarch64")]
    dotprod_prepared: bool,
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
        if !dim.is_multiple_of(heads) {
            bail!("quantized LM dim {dim} is not divisible by heads {heads}");
        }
        let hidden_dim = dim * 4;
        let attention_scale = 1.0 / ((dim / heads) as f64).sqrt();

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
            attention_scale,
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

    pub fn validate_for_metadata(&self, metadata: &OnnxFrameBundleMetadata) -> Result<()> {
        metadata.validate_lm()?;
        if self.codebooks != metadata.num_codebooks {
            bail!(
                "quantized LM weights contain {} codebooks, but bundle requires {}",
                self.codebooks,
                metadata.num_codebooks,
            );
        }
        if self.cardinality != metadata.lm_cardinality() {
            bail!(
                "quantized LM cardinality {} does not match bundle {}",
                self.cardinality,
                metadata.lm_cardinality(),
            );
        }
        if self.dim != metadata.lm_dim()? {
            bail!(
                "quantized LM dimension {} does not match bundle {}",
                self.dim,
                metadata.lm_dim()?,
            );
        }
        if self.layers != metadata.lm_num_layers()? {
            bail!(
                "quantized LM layer count {} does not match bundle {}",
                self.layers,
                metadata.lm_num_layers()?,
            );
        }
        if self.past_context != metadata.lm_past_context()? {
            bail!(
                "quantized LM past context {} does not match bundle {}",
                self.past_context,
                metadata.lm_past_context()?,
            );
        }
        if metadata.frame_length > self.frame_length {
            bail!(
                "bundle frame length {} exceeds quantized LM capacity {}",
                metadata.frame_length,
                self.frame_length,
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
        let cache_capacity = self.state_cache_capacity();
        let mut layers = Vec::with_capacity(self.weights.layers);
        for layer in &self.weights.layer_weights {
            let mut keys = Vec::with_capacity(cache_capacity);
            let mut values = Vec::with_capacity(cache_capacity);
            keys.extend_from_slice(&layer.in_proj_bias[dim..2 * dim]);
            values.extend_from_slice(&layer.in_proj_bias[2 * dim..3 * dim]);
            layers.push(LayerState {
                keys,
                values,
                len: 1,
            });
        }
        QuantizedLmState { offset: 0, layers }
    }

    pub(crate) fn reset_state(&self, state: &mut QuantizedLmState) {
        if state.layers.len() != self.weights.layers {
            *state = self.initial_state();
            return;
        }

        let dim = self.weights.dim;
        let cache_capacity = self.state_cache_capacity();
        state.offset = 0;
        for (layer_state, layer) in state.layers.iter_mut().zip(&self.weights.layer_weights) {
            layer_state.keys.clear();
            layer_state.values.clear();
            if layer_state.keys.capacity() < cache_capacity {
                layer_state
                    .keys
                    .reserve(cache_capacity - layer_state.keys.capacity());
            }
            if layer_state.values.capacity() < cache_capacity {
                layer_state
                    .values
                    .reserve(cache_capacity - layer_state.values.capacity());
            }
            layer_state
                .keys
                .extend_from_slice(&layer.in_proj_bias[dim..2 * dim]);
            layer_state
                .values
                .extend_from_slice(&layer.in_proj_bias[2 * dim..3 * dim]);
            layer_state.len = 1;
        }
    }

    fn state_cache_capacity(&self) -> usize {
        let cached_steps = self
            .weights
            .frame_length
            .saturating_add(1)
            .min(self.weights.past_context.saturating_add(1));
        cached_steps.saturating_mul(self.weights.dim)
    }

    pub fn forward_step(
        &mut self,
        state: &mut QuantizedLmState,
        input_symbols: &[usize],
    ) -> Result<Vec<f32>> {
        Ok(self.forward_step_borrowed(state, input_symbols)?.to_vec())
    }

    pub(crate) fn forward_step_borrowed<'a>(
        &'a mut self,
        state: &mut QuantizedLmState,
        input_symbols: &[usize],
    ) -> Result<&'a [f32]> {
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

        self.output_logits(codebooks);
        state.offset += 1;
        Ok(&self.scratch.logits)
    }

    fn forward_layer(&mut self, state: &mut QuantizedLmState, layer_index: usize) -> Result<()> {
        let dim = self.weights.dim;
        let layer = &self.weights.layer_weights[layer_index];
        let layer_state = &mut state.layers[layer_index];
        if layer_state.len > self.weights.past_context + 1 {
            bail!("LM layer state exceeded past_context");
        }

        let input_scale = self.scratch.quantized_input.quantize(&self.scratch.x);
        quantized_linear_part_with_input(
            &mut self.scratch.quantized_input,
            input_scale,
            &layer.in_proj_weight,
            &layer.in_proj_bias,
            0,
            dim,
            &mut self.scratch.q,
        );
        quantized_linear_part_with_input(
            &mut self.scratch.quantized_input,
            input_scale,
            &layer.in_proj_weight,
            &layer.in_proj_bias,
            dim,
            dim,
            &mut self.scratch.k,
        );
        quantized_linear_part_with_input(
            &mut self.scratch.quantized_input,
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
            self.weights.attention_scale,
            &mut self.scratch.attn,
            &mut self.scratch.scores,
        )?;

        quantized_linear(
            &self.scratch.attn,
            &layer.out_proj_weight,
            &layer.out_proj_bias,
            &mut self.scratch.quantized_input,
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
            &mut self.scratch.quantized_input,
            &mut self.scratch.ff,
        );
        for value in &mut self.scratch.ff {
            *value = gelu(*value as f64) as f32;
        }
        quantized_linear(
            &self.scratch.ff,
            &layer.linear2_weight,
            &layer.linear2_bias,
            &mut self.scratch.quantized_input,
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

    fn output_logits(&mut self, codebooks: usize) {
        let card = self.weights.cardinality;
        self.scratch.logits.resize(card * codebooks, 0.0);
        let input_scale = self.scratch.quantized_input.quantize(&self.scratch.x);
        for codebook in 0..codebooks {
            let weight = &self.weights.output_weights[codebook];
            let bias = &self.weights.output_biases[codebook];
            quantized_linear_part_with_input(
                &mut self.scratch.quantized_input,
                input_scale,
                weight,
                bias,
                0,
                card,
                &mut self.scratch.logit_column,
            );
            for bin in 0..card {
                self.scratch.logits[bin * codebooks + codebook] = self.scratch.logit_column[bin];
            }
        }
    }
}

impl QuantizedLinear {
    fn row(&self, row: usize) -> &[i8] {
        debug_assert!(row < self.rows);
        let start = row * self.cols;
        &self.weights[start..start + self.cols]
    }
}

impl QuantizedInputScratch {
    fn quantize(&mut self, input: &[f32]) -> f32 {
        let scale = quantize_input_i16(input, &mut self.values);
        #[cfg(target_arch = "aarch64")]
        {
            self.dotprod_prepared = false;
        }
        scale
    }

    #[cfg(target_arch = "aarch64")]
    fn prepare_dotprod(&mut self) {
        if self.dotprod_prepared {
            return;
        }
        self.dotprod_low.resize(self.values.len(), 0);
        self.dotprod_high.resize(self.values.len(), 0);
        for (index, value) in self.values.iter().copied().enumerate() {
            self.dotprod_low[index] = (((value as i32) & 0xff) - 128) as i8;
            self.dotprod_high[index] = (value >> 8) as i8;
        }
        self.dotprod_prepared = true;
    }
}

fn quantized_linear(
    input: &[f32],
    weight: &QuantizedLinear,
    bias: &[f32],
    input_q: &mut QuantizedInputScratch,
    out: &mut Vec<f32>,
) {
    let input_scale = input_q.quantize(input);
    quantized_linear_part_with_input(input_q, input_scale, weight, bias, 0, weight.rows, out);
}

fn quantized_linear_part_with_input(
    input_q: &mut QuantizedInputScratch,
    input_scale: f32,
    weight: &QuantizedLinear,
    bias: &[f32],
    row_offset: usize,
    out_dim: usize,
    out: &mut Vec<f32>,
) {
    debug_assert!(row_offset + out_dim <= weight.rows);
    debug_assert_eq!(input_q.values.len(), weight.cols);
    out.resize(out_dim, 0.0);

    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    {
        // SAFETY: the helper checks complete eight-row tiles and complete
        // eight-column vectors before each unaligned load. It handles all
        // remaining rows and columns with the checked one-row path.
        unsafe {
            quantized_linear_part_wasm_simd128(
                &input_q.values,
                input_scale,
                weight,
                bias,
                row_offset,
                out,
            );
        }
        return;
    }

    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: the helper checks complete eight-row tiles and complete
        // eight-column vectors before each unaligned load. It handles all
        // remaining rows and columns with the checked one-row path.
        unsafe {
            quantized_linear_part_aarch64(input_q, input_scale, weight, bias, row_offset, out);
        }
    }

    #[cfg(all(
        not(target_arch = "aarch64"),
        not(all(target_arch = "wasm32", target_feature = "simd128"))
    ))]
    for (row, output) in out.iter_mut().enumerate() {
        let source_row = row_offset + row;
        let acc = dot_i8_i16(weight.row(source_row), &input_q.values);
        *output = bias[source_row] + (acc as f32) * input_scale * weight.scales[source_row];
    }
}

#[cfg(target_arch = "aarch64")]
unsafe fn quantized_linear_part_aarch64(
    input_q: &mut QuantizedInputScratch,
    input_scale: f32,
    weight: &QuantizedLinear,
    bias: &[f32],
    row_offset: usize,
    out: &mut [f32],
) {
    if weight.use_dotprod {
        input_q.prepare_dotprod();
    }
    let input_values = &input_q.values;
    let mut row = 0_usize;
    while row + 8 <= out.len() {
        let source_row = row_offset + row;
        let accumulators = if weight.use_dotprod {
            dot_i8_i16_8_rows_dotprod_packed_aarch64(
                &weight.weights,
                &weight.row_sums,
                source_row,
                weight.cols,
                &input_q.dotprod_low,
                &input_q.dotprod_high,
            )
        } else {
            dot_i8_i16_8_rows_aarch64(&weight.weights, source_row, weight.cols, input_values)
        };
        for lane in 0..8 {
            let output_row = source_row + lane;
            out[row + lane] = bias[output_row]
                + (accumulators[lane] as f32) * input_scale * weight.scales[output_row];
        }
        row += 8;
    }

    while row < out.len() {
        let source_row = row_offset + row;
        let acc = dot_i8_i16_aarch64(weight.row(source_row), input_values);
        out[row] = bias[source_row] + (acc as f32) * input_scale * weight.scales[source_row];
        row += 1;
    }
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[target_feature(enable = "simd128")]
unsafe fn quantized_linear_part_wasm_simd128(
    input_q: &[i16],
    input_scale: f32,
    weight: &QuantizedLinear,
    bias: &[f32],
    row_offset: usize,
    out: &mut [f32],
) {
    let mut row = 0_usize;
    while row + 8 <= out.len() {
        let source_row = row_offset + row;
        let accumulators =
            dot_i8_i16_8_rows_wasm_simd128(&weight.weights, source_row, weight.cols, input_q);
        for lane in 0..8 {
            let output_row = source_row + lane;
            out[row + lane] = bias[output_row]
                + (accumulators[lane] as f32) * input_scale * weight.scales[output_row];
        }
        row += 8;
    }

    while row < out.len() {
        let source_row = row_offset + row;
        let acc = dot_i8_i16_wasm_simd128(weight.row(source_row), input_q);
        out[row] = bias[source_row] + (acc as f32) * input_scale * weight.scales[source_row];
        row += 1;
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

#[cfg_attr(
    any(
        target_arch = "aarch64",
        all(target_arch = "wasm32", target_feature = "simd128")
    ),
    allow(dead_code)
)]
fn dot_i8_i16(weights: &[i8], input: &[i16]) -> i64 {
    debug_assert_eq!(weights.len(), input.len());
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: the helper only performs unaligned vector loads inside the
        // checked slice bounds and handles any tail elements with scalar code.
        unsafe { dot_i8_i16_aarch64(weights, input) }
    }
    #[cfg(all(
        target_arch = "wasm32",
        target_feature = "simd128",
        not(target_arch = "aarch64")
    ))]
    {
        // SAFETY: the helper only performs unaligned vector loads inside the
        // checked slice bounds and handles any tail elements with scalar code.
        unsafe { dot_i8_i16_wasm_simd128(weights, input) }
    }
    #[cfg(all(
        not(target_arch = "aarch64"),
        not(all(target_arch = "wasm32", target_feature = "simd128"))
    ))]
    {
        dot_i8_i16_scalar(weights, input)
    }
}

#[cfg_attr(not(test), allow(dead_code))]
fn dot_i8_i16_scalar(weights: &[i8], input: &[i16]) -> i64 {
    let mut acc = 0_i64;
    for (w, x) in weights.iter().zip(input.iter()) {
        acc += (*w as i64) * (*x as i64);
    }
    acc
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[inline(always)]
unsafe fn add_wasm_i32x4_lanes(lanes: core::arch::wasm32::v128, acc: &mut i64) {
    use core::arch::wasm32::i32x4_extract_lane;

    *acc += i32x4_extract_lane::<0>(lanes) as i64;
    *acc += i32x4_extract_lane::<1>(lanes) as i64;
    *acc += i32x4_extract_lane::<2>(lanes) as i64;
    *acc += i32x4_extract_lane::<3>(lanes) as i64;
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[target_feature(enable = "simd128")]
unsafe fn dot_i8_i16_8_rows_wasm_simd128(
    weights: &[i8],
    first_row: usize,
    cols: usize,
    input: &[i16],
) -> [i64; 8] {
    use core::arch::wasm32::{
        i16x8_extend_low_i8x16, i32x4_add, i32x4_dot_i16x8, i32x4_splat, v128_load,
        v128_load64_zero,
    };

    const VECTORS_PER_FLUSH: usize = 128;

    debug_assert_eq!(input.len(), cols);
    debug_assert!((first_row + 8) * cols <= weights.len());

    let row0 = weights.as_ptr().add(first_row * cols);
    let row1 = row0.add(cols);
    let row2 = row1.add(cols);
    let row3 = row2.add(cols);
    let row4 = row3.add(cols);
    let row5 = row4.add(cols);
    let row6 = row5.add(cols);
    let row7 = row6.add(cols);
    let mut lane0 = i32x4_splat(0);
    let mut lane1 = i32x4_splat(0);
    let mut lane2 = i32x4_splat(0);
    let mut lane3 = i32x4_splat(0);
    let mut lane4 = i32x4_splat(0);
    let mut lane5 = i32x4_splat(0);
    let mut lane6 = i32x4_splat(0);
    let mut lane7 = i32x4_splat(0);
    let mut accumulators = [0_i64; 8];
    let mut index = 0_usize;
    let mut vectors = 0_usize;

    while index + 8 <= cols {
        let input_i16 = v128_load(input.as_ptr().add(index).cast());
        let weights0 = i16x8_extend_low_i8x16(v128_load64_zero(row0.add(index).cast::<u64>()));
        let weights1 = i16x8_extend_low_i8x16(v128_load64_zero(row1.add(index).cast::<u64>()));
        let weights2 = i16x8_extend_low_i8x16(v128_load64_zero(row2.add(index).cast::<u64>()));
        let weights3 = i16x8_extend_low_i8x16(v128_load64_zero(row3.add(index).cast::<u64>()));
        let weights4 = i16x8_extend_low_i8x16(v128_load64_zero(row4.add(index).cast::<u64>()));
        let weights5 = i16x8_extend_low_i8x16(v128_load64_zero(row5.add(index).cast::<u64>()));
        let weights6 = i16x8_extend_low_i8x16(v128_load64_zero(row6.add(index).cast::<u64>()));
        let weights7 = i16x8_extend_low_i8x16(v128_load64_zero(row7.add(index).cast::<u64>()));
        lane0 = i32x4_add(lane0, i32x4_dot_i16x8(weights0, input_i16));
        lane1 = i32x4_add(lane1, i32x4_dot_i16x8(weights1, input_i16));
        lane2 = i32x4_add(lane2, i32x4_dot_i16x8(weights2, input_i16));
        lane3 = i32x4_add(lane3, i32x4_dot_i16x8(weights3, input_i16));
        lane4 = i32x4_add(lane4, i32x4_dot_i16x8(weights4, input_i16));
        lane5 = i32x4_add(lane5, i32x4_dot_i16x8(weights5, input_i16));
        lane6 = i32x4_add(lane6, i32x4_dot_i16x8(weights6, input_i16));
        lane7 = i32x4_add(lane7, i32x4_dot_i16x8(weights7, input_i16));
        vectors += 1;
        index += 8;

        if vectors == VECTORS_PER_FLUSH {
            add_wasm_i32x4_lanes(lane0, &mut accumulators[0]);
            add_wasm_i32x4_lanes(lane1, &mut accumulators[1]);
            add_wasm_i32x4_lanes(lane2, &mut accumulators[2]);
            add_wasm_i32x4_lanes(lane3, &mut accumulators[3]);
            add_wasm_i32x4_lanes(lane4, &mut accumulators[4]);
            add_wasm_i32x4_lanes(lane5, &mut accumulators[5]);
            add_wasm_i32x4_lanes(lane6, &mut accumulators[6]);
            add_wasm_i32x4_lanes(lane7, &mut accumulators[7]);
            lane0 = i32x4_splat(0);
            lane1 = i32x4_splat(0);
            lane2 = i32x4_splat(0);
            lane3 = i32x4_splat(0);
            lane4 = i32x4_splat(0);
            lane5 = i32x4_splat(0);
            lane6 = i32x4_splat(0);
            lane7 = i32x4_splat(0);
            vectors = 0;
        }
    }

    add_wasm_i32x4_lanes(lane0, &mut accumulators[0]);
    add_wasm_i32x4_lanes(lane1, &mut accumulators[1]);
    add_wasm_i32x4_lanes(lane2, &mut accumulators[2]);
    add_wasm_i32x4_lanes(lane3, &mut accumulators[3]);
    add_wasm_i32x4_lanes(lane4, &mut accumulators[4]);
    add_wasm_i32x4_lanes(lane5, &mut accumulators[5]);
    add_wasm_i32x4_lanes(lane6, &mut accumulators[6]);
    add_wasm_i32x4_lanes(lane7, &mut accumulators[7]);

    while index < cols {
        let value = *input.get_unchecked(index) as i64;
        accumulators[0] += (*row0.add(index) as i64) * value;
        accumulators[1] += (*row1.add(index) as i64) * value;
        accumulators[2] += (*row2.add(index) as i64) * value;
        accumulators[3] += (*row3.add(index) as i64) * value;
        accumulators[4] += (*row4.add(index) as i64) * value;
        accumulators[5] += (*row5.add(index) as i64) * value;
        accumulators[6] += (*row6.add(index) as i64) * value;
        accumulators[7] += (*row7.add(index) as i64) * value;
        index += 1;
    }
    accumulators
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[target_feature(enable = "simd128")]
unsafe fn dot_i8_i16_wasm_simd128(weights: &[i8], input: &[i16]) -> i64 {
    use core::arch::wasm32::{
        i16x8_extend_low_i8x16, i32x4_add, i32x4_dot_i16x8, i32x4_extract_lane, i32x4_splat,
        v128_load, v128_load64_zero,
    };

    const VECTORS_PER_FLUSH: usize = 128;

    let len = weights.len().min(input.len());
    let mut index = 0_usize;
    let mut vectors = 0_usize;
    let mut lanes = i32x4_splat(0);
    let mut acc = 0_i64;
    while index + 8 <= len {
        let packed_weights = v128_load64_zero(weights.as_ptr().add(index).cast::<u64>());
        let weights_i16 = i16x8_extend_low_i8x16(packed_weights);
        let input_i16 = v128_load(input.as_ptr().add(index).cast());
        lanes = i32x4_add(lanes, i32x4_dot_i16x8(weights_i16, input_i16));
        vectors += 1;
        index += 8;

        // Each lane contains two products per vector. Flushing after 128
        // vectors keeps even the i8/i16 extreme bound inside signed i32.
        if vectors == VECTORS_PER_FLUSH {
            acc += i32x4_extract_lane::<0>(lanes) as i64;
            acc += i32x4_extract_lane::<1>(lanes) as i64;
            acc += i32x4_extract_lane::<2>(lanes) as i64;
            acc += i32x4_extract_lane::<3>(lanes) as i64;
            lanes = i32x4_splat(0);
            vectors = 0;
        }
    }
    acc += i32x4_extract_lane::<0>(lanes) as i64;
    acc += i32x4_extract_lane::<1>(lanes) as i64;
    acc += i32x4_extract_lane::<2>(lanes) as i64;
    acc += i32x4_extract_lane::<3>(lanes) as i64;

    while index < len {
        acc += (*weights.get_unchecked(index) as i64) * (*input.get_unchecked(index) as i64);
        index += 1;
    }
    acc
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn sum_aarch64_i32x4_pair(
    low: core::arch::aarch64::int32x4_t,
    high: core::arch::aarch64::int32x4_t,
) -> i64 {
    use core::arch::aarch64::vaddvq_s32;

    vaddvq_s32(low) as i64 + vaddvq_s32(high) as i64
}

#[cfg(target_arch = "aarch64")]
unsafe fn dot_i8_i16_8_rows_aarch64(
    weights: &[i8],
    first_row: usize,
    cols: usize,
    input: &[i16],
) -> [i64; 8] {
    use core::arch::aarch64::{
        vdupq_n_s32, vget_high_s16, vget_high_s8, vget_low_s16, vget_low_s8, vld1_s8, vld1q_s16,
        vld1q_s8, vmlal_s16, vmovl_s8,
    };

    const VECTORS_PER_FLUSH: usize = 128;

    debug_assert_eq!(input.len(), cols);
    debug_assert!((first_row + 8) * cols <= weights.len());

    let row0 = weights.as_ptr().add(first_row * cols);
    let row1 = row0.add(cols);
    let row2 = row1.add(cols);
    let row3 = row2.add(cols);
    let row4 = row3.add(cols);
    let row5 = row4.add(cols);
    let row6 = row5.add(cols);
    let row7 = row6.add(cols);
    let mut low0 = vdupq_n_s32(0);
    let mut low1 = vdupq_n_s32(0);
    let mut low2 = vdupq_n_s32(0);
    let mut low3 = vdupq_n_s32(0);
    let mut low4 = vdupq_n_s32(0);
    let mut low5 = vdupq_n_s32(0);
    let mut low6 = vdupq_n_s32(0);
    let mut low7 = vdupq_n_s32(0);
    let mut high0 = vdupq_n_s32(0);
    let mut high1 = vdupq_n_s32(0);
    let mut high2 = vdupq_n_s32(0);
    let mut high3 = vdupq_n_s32(0);
    let mut high4 = vdupq_n_s32(0);
    let mut high5 = vdupq_n_s32(0);
    let mut high6 = vdupq_n_s32(0);
    let mut high7 = vdupq_n_s32(0);
    let mut accumulators = [0_i64; 8];
    let mut index = 0_usize;
    let mut vectors = 0_usize;

    macro_rules! flush_accumulators {
        () => {{
            accumulators[0] += sum_aarch64_i32x4_pair(low0, high0);
            accumulators[1] += sum_aarch64_i32x4_pair(low1, high1);
            accumulators[2] += sum_aarch64_i32x4_pair(low2, high2);
            accumulators[3] += sum_aarch64_i32x4_pair(low3, high3);
            accumulators[4] += sum_aarch64_i32x4_pair(low4, high4);
            accumulators[5] += sum_aarch64_i32x4_pair(low5, high5);
            accumulators[6] += sum_aarch64_i32x4_pair(low6, high6);
            accumulators[7] += sum_aarch64_i32x4_pair(low7, high7);
            low0 = vdupq_n_s32(0);
            low1 = vdupq_n_s32(0);
            low2 = vdupq_n_s32(0);
            low3 = vdupq_n_s32(0);
            low4 = vdupq_n_s32(0);
            low5 = vdupq_n_s32(0);
            low6 = vdupq_n_s32(0);
            low7 = vdupq_n_s32(0);
            high0 = vdupq_n_s32(0);
            high1 = vdupq_n_s32(0);
            high2 = vdupq_n_s32(0);
            high3 = vdupq_n_s32(0);
            high4 = vdupq_n_s32(0);
            high5 = vdupq_n_s32(0);
            high6 = vdupq_n_s32(0);
            high7 = vdupq_n_s32(0);
            vectors = 0;
        }};
    }

    while index + 16 <= cols {
        let input0 = vld1q_s16(input.as_ptr().add(index));
        let input1 = vld1q_s16(input.as_ptr().add(index + 8));
        let input0_low = vget_low_s16(input0);
        let input0_high = vget_high_s16(input0);
        let input1_low = vget_low_s16(input1);
        let input1_high = vget_high_s16(input1);

        macro_rules! accumulate_row {
            ($row:expr, $low:ident, $high:ident) => {{
                let packed = vld1q_s8($row.add(index));
                let first = vmovl_s8(vget_low_s8(packed));
                let second = vmovl_s8(vget_high_s8(packed));
                $low = vmlal_s16($low, vget_low_s16(first), input0_low);
                $high = vmlal_s16($high, vget_high_s16(first), input0_high);
                $low = vmlal_s16($low, vget_low_s16(second), input1_low);
                $high = vmlal_s16($high, vget_high_s16(second), input1_high);
            }};
        }

        accumulate_row!(row0, low0, high0);
        accumulate_row!(row1, low1, high1);
        accumulate_row!(row2, low2, high2);
        accumulate_row!(row3, low3, high3);
        accumulate_row!(row4, low4, high4);
        accumulate_row!(row5, low5, high5);
        accumulate_row!(row6, low6, high6);
        accumulate_row!(row7, low7, high7);

        vectors += 2;
        index += 16;
        if vectors == VECTORS_PER_FLUSH {
            flush_accumulators!();
        }
    }

    if index + 8 <= cols {
        let input_i16 = vld1q_s16(input.as_ptr().add(index));
        let input_low = vget_low_s16(input_i16);
        let input_high = vget_high_s16(input_i16);

        macro_rules! accumulate_half_row {
            ($row:expr, $low:ident, $high:ident) => {{
                let packed = vmovl_s8(vld1_s8($row.add(index)));
                $low = vmlal_s16($low, vget_low_s16(packed), input_low);
                $high = vmlal_s16($high, vget_high_s16(packed), input_high);
            }};
        }

        accumulate_half_row!(row0, low0, high0);
        accumulate_half_row!(row1, low1, high1);
        accumulate_half_row!(row2, low2, high2);
        accumulate_half_row!(row3, low3, high3);
        accumulate_half_row!(row4, low4, high4);
        accumulate_half_row!(row5, low5, high5);
        accumulate_half_row!(row6, low6, high6);
        accumulate_half_row!(row7, low7, high7);

        index += 8;
    }

    accumulators[0] += sum_aarch64_i32x4_pair(low0, high0);
    accumulators[1] += sum_aarch64_i32x4_pair(low1, high1);
    accumulators[2] += sum_aarch64_i32x4_pair(low2, high2);
    accumulators[3] += sum_aarch64_i32x4_pair(low3, high3);
    accumulators[4] += sum_aarch64_i32x4_pair(low4, high4);
    accumulators[5] += sum_aarch64_i32x4_pair(low5, high5);
    accumulators[6] += sum_aarch64_i32x4_pair(low6, high6);
    accumulators[7] += sum_aarch64_i32x4_pair(low7, high7);

    while index < cols {
        let value = *input.get_unchecked(index) as i64;
        accumulators[0] += (*row0.add(index) as i64) * value;
        accumulators[1] += (*row1.add(index) as i64) * value;
        accumulators[2] += (*row2.add(index) as i64) * value;
        accumulators[3] += (*row3.add(index) as i64) * value;
        accumulators[4] += (*row4.add(index) as i64) * value;
        accumulators[5] += (*row5.add(index) as i64) * value;
        accumulators[6] += (*row6.add(index) as i64) * value;
        accumulators[7] += (*row7.add(index) as i64) * value;
        index += 1;
    }
    accumulators
}

#[cfg(target_arch = "aarch64")]
#[inline]
#[target_feature(enable = "dotprod")]
unsafe fn dotprod_accumulate_aarch64(
    mut accumulator: core::arch::aarch64::int32x4_t,
    weights: core::arch::aarch64::int8x16_t,
    input: core::arch::aarch64::int8x16_t,
) -> core::arch::aarch64::int32x4_t {
    use core::arch::asm;

    asm!(
        "sdot {accumulator:v}.4s, {weights:v}.16b, {input:v}.16b",
        accumulator = inout(vreg) accumulator,
        weights = in(vreg) weights,
        input = in(vreg) input,
        options(pure, nomem, nostack, preserves_flags)
    );
    accumulator
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "dotprod")]
unsafe fn dot_i8_i16_8_rows_dotprod_packed_aarch64(
    weights: &[i8],
    row_sums: &[i32],
    first_row: usize,
    cols: usize,
    input_low: &[i8],
    input_high: &[i8],
) -> [i64; 8] {
    use core::arch::aarch64::{vcombine_s8, vdup_n_s8, vdupq_n_s32, vld1_s8, vld1q_s8};

    debug_assert_eq!(input_low.len(), cols);
    debug_assert_eq!(input_high.len(), cols);
    debug_assert!((first_row + 8) * cols <= weights.len());
    debug_assert!(first_row + 8 <= row_sums.len());

    let row0 = weights.as_ptr().add(first_row * cols);
    let row1 = row0.add(cols);
    let row2 = row1.add(cols);
    let row3 = row2.add(cols);
    let row4 = row3.add(cols);
    let row5 = row4.add(cols);
    let row6 = row5.add(cols);
    let row7 = row6.add(cols);
    let mut low0 = vdupq_n_s32(0);
    let mut low1 = vdupq_n_s32(0);
    let mut low2 = vdupq_n_s32(0);
    let mut low3 = vdupq_n_s32(0);
    let mut low4 = vdupq_n_s32(0);
    let mut low5 = vdupq_n_s32(0);
    let mut low6 = vdupq_n_s32(0);
    let mut low7 = vdupq_n_s32(0);
    let mut high0 = vdupq_n_s32(0);
    let mut high1 = vdupq_n_s32(0);
    let mut high2 = vdupq_n_s32(0);
    let mut high3 = vdupq_n_s32(0);
    let mut high4 = vdupq_n_s32(0);
    let mut high5 = vdupq_n_s32(0);
    let mut high6 = vdupq_n_s32(0);
    let mut high7 = vdupq_n_s32(0);
    let zero_i8x8 = vdup_n_s8(0);
    let mut index = 0_usize;

    macro_rules! accumulate_row_16 {
        ($row:expr, $input_low:expr, $input_high:expr, $low:ident, $high:ident) => {{
            let packed = vld1q_s8($row.add(index));
            $low = dotprod_accumulate_aarch64($low, packed, $input_low);
            $high = dotprod_accumulate_aarch64($high, packed, $input_high);
        }};
    }

    while index + 16 <= cols {
        let input_low = vld1q_s8(input_low.as_ptr().add(index));
        let input_high = vld1q_s8(input_high.as_ptr().add(index));

        accumulate_row_16!(row0, input_low, input_high, low0, high0);
        accumulate_row_16!(row1, input_low, input_high, low1, high1);
        accumulate_row_16!(row2, input_low, input_high, low2, high2);
        accumulate_row_16!(row3, input_low, input_high, low3, high3);
        accumulate_row_16!(row4, input_low, input_high, low4, high4);
        accumulate_row_16!(row5, input_low, input_high, low5, high5);
        accumulate_row_16!(row6, input_low, input_high, low6, high6);
        accumulate_row_16!(row7, input_low, input_high, low7, high7);
        index += 16;
    }

    if index + 8 <= cols {
        let input_low = vcombine_s8(vld1_s8(input_low.as_ptr().add(index)), zero_i8x8);
        let input_high = vcombine_s8(vld1_s8(input_high.as_ptr().add(index)), zero_i8x8);

        macro_rules! accumulate_row_8 {
            ($row:expr, $low:ident, $high:ident) => {{
                let packed = vcombine_s8(vld1_s8($row.add(index)), zero_i8x8);
                $low = dotprod_accumulate_aarch64($low, packed, input_low);
                $high = dotprod_accumulate_aarch64($high, packed, input_high);
            }};
        }

        accumulate_row_8!(row0, low0, high0);
        accumulate_row_8!(row1, low1, high1);
        accumulate_row_8!(row2, low2, high2);
        accumulate_row_8!(row3, low3, high3);
        accumulate_row_8!(row4, low4, high4);
        accumulate_row_8!(row5, low5, high5);
        accumulate_row_8!(row6, low6, high6);
        accumulate_row_8!(row7, low7, high7);
        index += 8;
    }

    let vector_end = index;
    let mut tail_weight_sums = [0_i32; 8];
    let mut accumulators = [0_i64; 8];
    while index < cols {
        let value = *input_low.get_unchecked(index) as i64
            + 128
            + 256 * *input_high.get_unchecked(index) as i64;
        let weights = [
            *row0.add(index),
            *row1.add(index),
            *row2.add(index),
            *row3.add(index),
            *row4.add(index),
            *row5.add(index),
            *row6.add(index),
            *row7.add(index),
        ];
        for row in 0..8 {
            tail_weight_sums[row] += weights[row] as i32;
            accumulators[row] += weights[row] as i64 * value;
        }
        index += 1;
    }

    debug_assert_eq!(vector_end, cols - (cols % 8));
    let low = [low0, low1, low2, low3, low4, low5, low6, low7];
    let high = [high0, high1, high2, high3, high4, high5, high6, high7];
    for row in 0..8 {
        let vector_weight_sum = row_sums[first_row + row] - tail_weight_sums[row];
        accumulators[row] += sum_aarch64_i32x4_pair(low[row], high[row])
            + 255 * core::arch::aarch64::vaddvq_s32(high[row]) as i64
            + 128 * vector_weight_sum as i64;
    }
    accumulators
}

#[cfg(all(target_arch = "aarch64", test))]
#[target_feature(enable = "dotprod")]
unsafe fn dot_i8_i16_8_rows_dotprod_aarch64(
    weights: &[i8],
    row_sums: &[i32],
    first_row: usize,
    cols: usize,
    input: &[i16],
) -> [i64; 8] {
    let mut scratch = QuantizedInputScratch {
        values: input.to_vec(),
        ..QuantizedInputScratch::default()
    };
    scratch.prepare_dotprod();
    dot_i8_i16_8_rows_dotprod_packed_aarch64(
        weights,
        row_sums,
        first_row,
        cols,
        &scratch.dotprod_low,
        &scratch.dotprod_high,
    )
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
    scale: f64,
    out: &mut Vec<f32>,
    scores: &mut Vec<f64>,
) -> Result<()> {
    let dim = query.len();
    if !dim.is_multiple_of(heads) {
        bail!("attention dim {dim} is not divisible by heads {heads}");
    }
    if keys.len() != len * dim || values.len() != len * dim {
        bail!("attention cache shape mismatch");
    }
    let head_dim = dim / heads;
    out.clear();
    out.resize(dim, 0.0);
    scores.resize(len, 0.0);

    for head in 0..heads {
        let head_base = head * head_dim;
        let mut max_score = f64::NEG_INFINITY;
        for (t, output_score) in scores.iter_mut().enumerate().take(len) {
            let base = t * dim + head_base;
            let mut dot = 0.0_f64;
            for d in 0..head_dim {
                dot += (query[head_base + d] as f64) * (keys[base + d] as f64);
            }
            let score = dot * scale;
            *output_score = score;
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

        for (t, score) in scores.iter().copied().enumerate().take(len) {
            let prob = score / denom;
            let base = t * dim + head_base;
            for d in 0..head_dim {
                out[head_base + d] += (prob * values[base + d] as f64) as f32;
            }
        }
    }
    Ok(())
}

fn gelu(value: f64) -> f64 {
    0.5 * value * (1.0 + libm::erf(value / std::f64::consts::SQRT_2))
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
        let weights: Vec<i8> = bytes.iter().map(|value| *value as i8).collect();
        #[cfg(target_arch = "aarch64")]
        let row_sums = weights
            .chunks_exact(cols)
            .map(|row| row.iter().map(|value| *value as i32).sum())
            .collect();
        Ok(QuantizedLinear {
            rows,
            cols,
            scales,
            weights,
            #[cfg(target_arch = "aarch64")]
            row_sums,
            #[cfg(target_arch = "aarch64")]
            use_dotprod: aarch64_dotprod_enabled(),
        })
    }

    fn remaining(&self) -> usize {
        self.bytes.len().saturating_sub(self.pos)
    }
}

#[cfg(target_arch = "aarch64")]
fn aarch64_dotprod_enabled() -> bool {
    std::arch::is_aarch64_feature_detected!("dotprod")
        && std::env::var_os("BITNEEDLE_LM_DISABLE_DOTPROD").is_none()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reset_state_reuses_cache_storage_and_matches_a_fresh_state() -> Result<()> {
        let bundle_dir = std::path::Path::new("onnx-bundles/encodec_48khz_6kbps_1333ms");
        if !bundle_dir.exists() {
            eprintln!("skipping LM fixture test; run scripts/download-onnx-bundles.sh first");
            return Ok(());
        }
        let metadata: OnnxFrameBundleMetadata =
            serde_json::from_str(&std::fs::read_to_string(bundle_dir.join("bundle.json"))?)?;
        let weight_name = metadata
            .lm_quant_weight_model
            .as_deref()
            .ok_or_else(|| anyhow::anyhow!("fixture has no q8 LM weights"))?;
        let weights =
            QuantizedLmWeights::from_bytes(&std::fs::read(bundle_dir.join(weight_name))?)?;
        let mut reused_lm = QuantizedLm::new(weights.clone());
        let mut fresh_lm = QuantizedLm::new(weights);
        let mut reused_state = reused_lm.initial_state();
        let initial_storage = reused_state
            .layers
            .iter()
            .map(|layer| {
                (
                    layer.keys.as_ptr(),
                    layer.keys.capacity(),
                    layer.values.as_ptr(),
                    layer.values.capacity(),
                )
            })
            .collect::<Vec<_>>();

        for step in 0..8 {
            let symbols = (0..metadata.num_codebooks)
                .map(|codebook| (step * 17 + codebook * 31) % metadata.lm_cardinality())
                .collect::<Vec<_>>();
            reused_lm.forward_step_borrowed(&mut reused_state, &symbols)?;
        }
        reused_lm.reset_state(&mut reused_state);

        for (layer, (keys, keys_capacity, values, values_capacity)) in
            reused_state.layers.iter().zip(initial_storage)
        {
            assert_eq!(layer.keys.as_ptr(), keys);
            assert_eq!(layer.keys.capacity(), keys_capacity);
            assert_eq!(layer.values.as_ptr(), values);
            assert_eq!(layer.values.capacity(), values_capacity);
            assert_eq!(layer.len, 1);
        }

        let mut fresh_state = fresh_lm.initial_state();
        for step in 0..8 {
            let symbols = (0..metadata.num_codebooks)
                .map(|codebook| (step * 43 + codebook * 13) % metadata.lm_cardinality())
                .collect::<Vec<_>>();
            let reused = reused_lm
                .forward_step_borrowed(&mut reused_state, &symbols)?
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>();
            let fresh = fresh_lm
                .forward_step_borrowed(&mut fresh_state, &symbols)?
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>();
            assert_eq!(reused, fresh, "logits differ after reset at step {step}");
        }
        Ok(())
    }

    #[test]
    fn optimized_dot_matches_scalar_for_extreme_and_tail_lengths() {
        for len in [0, 1, 7, 8, 9, 25, 200, 800, 1025] {
            let weights: Vec<i8> = (0..len)
                .map(|index| match index % 4 {
                    0 => i8::MIN,
                    1 => i8::MAX,
                    2 => -1,
                    _ => 1,
                })
                .collect();
            let input: Vec<i16> = (0..len)
                .map(|index| match index % 4 {
                    0 => i16::MIN,
                    1 => i16::MAX,
                    2 => -17_123,
                    _ => 19_937,
                })
                .collect();
            assert_eq!(
                dot_i8_i16(&weights, &input),
                dot_i8_i16_scalar(&weights, &input)
            );
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn eight_row_aarch64_dot_matches_scalar_for_extreme_and_tail_lengths() {
        for len in [0, 1, 7, 8, 9, 25, 200, 800, 1025] {
            let weights: Vec<i8> = (0..8 * len)
                .map(|index| match index % 4 {
                    0 => i8::MIN,
                    1 => i8::MAX,
                    2 => -1,
                    _ => 1,
                })
                .collect();
            let input: Vec<i16> = (0..len)
                .map(|index| match index % 4 {
                    0 => i16::MIN,
                    1 => i16::MAX,
                    2 => -17_123,
                    _ => 19_937,
                })
                .collect();

            // SAFETY: the test provides eight complete rows and an input with
            // exactly the declared column count.
            let actual = unsafe { dot_i8_i16_8_rows_aarch64(&weights, 0, len, &input) };
            for row in 0..8 {
                assert_eq!(
                    actual[row],
                    dot_i8_i16_scalar(&weights[row * len..(row + 1) * len], &input)
                );
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn eight_row_aarch64_dotprod_matches_scalar_for_extreme_and_tail_lengths() {
        if !std::arch::is_aarch64_feature_detected!("dotprod") {
            return;
        }

        for len in [0, 1, 7, 8, 9, 25, 200, 800, 1025] {
            let weights: Vec<i8> = (0..8 * len)
                .map(|index| match index % 4 {
                    0 => i8::MIN,
                    1 => i8::MAX,
                    2 => -1,
                    _ => 1,
                })
                .collect();
            let row_sums: Vec<i32> = weights
                .chunks_exact(len.max(1))
                .take(8)
                .map(|row| row.iter().map(|value| *value as i32).sum())
                .collect();
            let row_sums = if len == 0 { vec![0; 8] } else { row_sums };
            let input: Vec<i16> = (0..len)
                .map(|index| match index % 4 {
                    0 => i16::MIN,
                    1 => i16::MAX,
                    2 => -17_123,
                    _ => 19_937,
                })
                .collect();

            // SAFETY: feature detection passed and the test provides eight
            // complete rows, row sums, and the declared input column count.
            let actual =
                unsafe { dot_i8_i16_8_rows_dotprod_aarch64(&weights, &row_sums, 0, len, &input) };
            for row in 0..8 {
                assert_eq!(
                    actual[row],
                    dot_i8_i16_scalar(&weights[row * len..(row + 1) * len], &input)
                );
            }
        }
    }
}
