#include <emscripten/emscripten.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <wasm_simd128.h>

int conv1d_nhwc_simd_8x8(const float *input, const float *packed,
                          const float *bias, float *output, int padded_time,
                          int input_channels, int output_channels, int kernel,
                          int stride);
static v128_t exp_f32x4_polynomial(v128_t value);

static v128_t elu_f32x4(v128_t value) {
  const v128_t zero = wasm_f32x4_splat(0.0f);
  const v128_t one = wasm_f32x4_splat(1.0f);
  return wasm_v128_bitselect(
      wasm_f32x4_sub(exp_f32x4_polynomial(value), one), value,
      wasm_f32x4_lt(value, zero));
}

static int valid_conv_geometry(int padded_time, int input_channels,
                               int output_channels, int kernel, int stride) {
  return padded_time >= kernel && input_channels > 0 && output_channels > 0 &&
         (output_channels % 8) == 0 && kernel > 0 && stride > 0;
}

/*
 * Convert ONNX [output_channel, input_channel, kernel] weights into
 * [output_channel / 8, input_channel, kernel, 8].
 */
EMSCRIPTEN_KEEPALIVE
int pack_conv1d_nhwc_weights_8(const float *weights, float *packed,
                               int input_channels, int output_channels,
                               int kernel) {
  if (weights == NULL || packed == NULL || input_channels <= 0 ||
      output_channels <= 0 || (output_channels % 8) != 0 || kernel <= 0) {
    return 0;
  }

  const size_t reduction = (size_t)input_channels * kernel;
  for (int output_channel = 0; output_channel < output_channels;
       output_channel += 8) {
    const size_t output_block = (size_t)(output_channel / 8) * reduction * 8;
    for (int input_channel = 0; input_channel < input_channels;
         ++input_channel) {
      for (int tap = 0; tap < kernel; ++tap) {
        const size_t reduction_index = (size_t)input_channel * kernel + tap;
        const size_t destination = output_block + reduction_index * 8;
        for (int lane = 0; lane < 8; ++lane) {
          const size_t source =
              ((size_t)(output_channel + lane) * input_channels +
               input_channel) *
                  kernel +
              tap;
          packed[destination + lane] = weights[source];
        }
      }
    }
  }
  return 1;
}

/* Convert [output, input] weights into [output / 8, input, 8]. */
EMSCRIPTEN_KEEPALIVE
int pack_linear_weights_8(const float *weights, float *packed, int input_size,
                          int output_size) {
  if (weights == NULL || packed == NULL || input_size <= 0 ||
      output_size <= 0 || (output_size % 8) != 0) {
    return 0;
  }
  for (int output = 0; output < output_size; output += 8) {
    const size_t block = (size_t)(output / 8) * input_size * 8;
    for (int input = 0; input < input_size; ++input) {
      for (int lane = 0; lane < 8; ++lane) {
        packed[block + (size_t)input * 8 + lane] =
            weights[(size_t)(output + lane) * input_size + input];
      }
    }
  }
  return 1;
}

/* Convert planar [channel, time] input into padded [time, channel] input. */
EMSCRIPTEN_KEEPALIVE
int reflect_pad_planar_to_nhwc(const float *input, float *output, int time,
                               int channels, int padding_left,
                               int padding_right) {
  if (input == NULL || output == NULL || time <= 0 || channels <= 0 ||
      padding_left < 0 || padding_right < 0 || padding_left >= time ||
      padding_right >= time) {
    return 0;
  }

  const int padded_time = padding_left + time + padding_right;
  for (int output_time = 0; output_time < padded_time; ++output_time) {
    int source_time = output_time - padding_left;
    if (source_time < 0) {
      source_time = -source_time;
    } else if (source_time >= time) {
      source_time = 2 * time - 2 - source_time;
    }
    for (int channel = 0; channel < channels; ++channel) {
      output[(size_t)output_time * channels + channel] =
          input[(size_t)channel * time + source_time];
    }
  }
  return 1;
}

/* Pad input that already uses [time, channel] storage. */
EMSCRIPTEN_KEEPALIVE
int reflect_pad_nhwc(const float *input, float *output, int time, int channels,
                     int padding_left, int padding_right) {
  if (input == NULL || output == NULL || time <= 0 || channels <= 0 ||
      padding_left < 0 || padding_right < 0 || padding_left >= time ||
      padding_right >= time) {
    return 0;
  }

  const int padded_time = padding_left + time + padding_right;
  for (int output_time = 0; output_time < padded_time; ++output_time) {
    int source_time = output_time - padding_left;
    if (source_time < 0) {
      source_time = -source_time;
    } else if (source_time >= time) {
      source_time = 2 * time - 2 - source_time;
    }
    const float *source = input + (size_t)source_time * channels;
    float *destination = output + (size_t)output_time * channels;
    for (int channel = 0; channel < channels; ++channel) {
      destination[channel] = source[channel];
    }
  }
  return 1;
}

EMSCRIPTEN_KEEPALIVE
int reflect_pad_elu_nhwc(const float *input, float *output, int time,
                         int channels, int padding_left, int padding_right) {
  if (input == NULL || output == NULL || time <= 0 || channels <= 0 ||
      padding_left < 0 || padding_right < 0 || padding_left >= time ||
      padding_right >= time) {
    return 0;
  }

  const int padded_time = padding_left + time + padding_right;
  for (int output_time = 0; output_time < padded_time; ++output_time) {
    int source_time = output_time - padding_left;
    if (source_time < 0) {
      source_time = -source_time;
    } else if (source_time >= time) {
      source_time = 2 * time - 2 - source_time;
    }
    const float *source = input + (size_t)source_time * channels;
    float *destination = output + (size_t)output_time * channels;
    int channel = 0;
    for (; channel + 3 < channels; channel += 4) {
      wasm_v128_store(destination + channel,
                      elu_f32x4(wasm_v128_load(source + channel)));
    }
    for (; channel < channels; ++channel) {
      const float value = source[channel];
      destination[channel] = value < 0.0f ? expf(value) - 1.0f : value;
    }
  }
  return 1;
}

/*
 * Match the EnCodec frame normalization and convert planar audio to NHWC.
 * The returned value is the scale that the decoder restores.
 */
EMSCRIPTEN_KEEPALIVE
float normalize_audio_planar_to_nhwc(const float *input, float *output,
                                     int time, int channels) {
  if (input == NULL || output == NULL || time <= 0 || channels <= 0) {
    return 0.0f;
  }

  float squared_sum = 0.0f;
  for (int index = 0; index < time; ++index) {
    float mono = 0.0f;
    for (int channel = 0; channel < channels; ++channel) {
      mono += input[(size_t)channel * time + index];
    }
    mono /= (float)channels;
    squared_sum += mono * mono;
  }
  const float scale = sqrtf(squared_sum / (float)time) + 1.0e-8f;
  const float inverse_scale = 1.0f / scale;
  for (int index = 0; index < time; ++index) {
    for (int channel = 0; channel < channels; ++channel) {
      output[(size_t)index * channels + channel] =
          input[(size_t)channel * time + index] * inverse_scale;
    }
  }
  return scale;
}

/* GroupNorm with one group, represented by ONNX as InstanceNormalization. */
EMSCRIPTEN_KEEPALIVE
int group_norm_nhwc_in_place(float *values, const float *scale,
                             const float *bias, int time, int channels) {
  if (values == NULL || scale == NULL || bias == NULL || time <= 0 ||
      channels <= 0) {
    return 0;
  }

  const size_t length = (size_t)time * channels;
  float sum = 0.0f;
  for (int channel = 0; channel < channels; ++channel) {
    for (int time_index = 0; time_index < time; ++time_index) {
      sum += values[(size_t)time_index * channels + channel];
    }
  }
  const float mean = sum / (float)length;

  float squared_norm = 0.0f;
  for (int channel = 0; channel < channels; ++channel) {
    for (int time_index = 0; time_index < time; ++time_index) {
      const float difference =
          values[(size_t)time_index * channels + channel] - mean;
      squared_norm += difference * difference;
    }
  }

  const float inverse_standard_deviation =
      1.0f / sqrtf(squared_norm / (float)length + 1.0e-5f);

  const v128_t inverse_standard_deviation_vector =
      wasm_f32x4_splat(inverse_standard_deviation);
  const v128_t shift_vector =
      wasm_f32x4_splat(-mean * inverse_standard_deviation);
  for (int time_index = 0; time_index < time; ++time_index) {
    float *row = values + (size_t)time_index * channels;
    int channel = 0;
    for (; channel + 3 < channels; channel += 4) {
      const v128_t normalized = wasm_f32x4_add(
          wasm_f32x4_mul(wasm_v128_load(row + channel),
                         inverse_standard_deviation_vector),
          shift_vector);
      const v128_t affine = wasm_f32x4_add(
          wasm_f32x4_mul(normalized, wasm_v128_load(scale + channel)),
          wasm_v128_load(bias + channel));
      wasm_v128_store(row + channel, affine);
    }
    for (; channel < channels; ++channel) {
      const float normalized =
          row[channel] * inverse_standard_deviation -
          mean * inverse_standard_deviation;
      row[channel] = normalized * scale[channel] + bias[channel];
    }
  }
  return 1;
}

static v128_t exp_f32x4_polynomial(v128_t value) {
  const v128_t one = wasm_f32x4_splat(1.0f);
  const v128_t clamped = wasm_f32x4_min(
      wasm_f32x4_max(value, wasm_f32x4_splat(-80.0f)),
      wasm_f32x4_splat(80.0f));
  const v128_t exponent_float = wasm_f32x4_nearest(wasm_f32x4_mul(
      clamped, wasm_f32x4_splat(1.44269504088896341f)));
  const v128_t exponent_integer =
      wasm_i32x4_trunc_sat_f32x4(exponent_float);
  v128_t remainder = wasm_f32x4_sub(
      clamped,
      wasm_f32x4_mul(exponent_float, wasm_f32x4_splat(0.693359375f)));
  remainder = wasm_f32x4_sub(
      remainder,
      wasm_f32x4_mul(exponent_float, wasm_f32x4_splat(-2.12194440e-4f)));

  v128_t polynomial = wasm_f32x4_splat(1.0f / 5040.0f);
  polynomial = wasm_f32x4_add(
      wasm_f32x4_mul(polynomial, remainder), wasm_f32x4_splat(1.0f / 720.0f));
  polynomial = wasm_f32x4_add(
      wasm_f32x4_mul(polynomial, remainder), wasm_f32x4_splat(1.0f / 120.0f));
  polynomial = wasm_f32x4_add(
      wasm_f32x4_mul(polynomial, remainder), wasm_f32x4_splat(1.0f / 24.0f));
  polynomial = wasm_f32x4_add(
      wasm_f32x4_mul(polynomial, remainder), wasm_f32x4_splat(1.0f / 6.0f));
  polynomial = wasm_f32x4_add(
      wasm_f32x4_mul(polynomial, remainder), wasm_f32x4_splat(0.5f));
  polynomial = wasm_f32x4_add(wasm_f32x4_mul(polynomial, remainder), one);
  polynomial = wasm_f32x4_add(wasm_f32x4_mul(polynomial, remainder), one);

  const v128_t power_of_two = wasm_i32x4_shl(
      wasm_i32x4_add(exponent_integer, wasm_i32x4_splat(127)), 23);
  return wasm_f32x4_mul(polynomial, power_of_two);
}

EMSCRIPTEN_KEEPALIVE
int elu_nhwc_in_place(float *values, int length) {
  if (values == NULL || length <= 0) {
    return 0;
  }
  int index = 0;
  for (; index + 3 < length; index += 4) {
    wasm_v128_store(values + index,
                    elu_f32x4(wasm_v128_load(values + index)));
  }
  for (; index < length; ++index) {
    if (values[index] < 0.0f) {
      values[index] = expf(values[index]) - 1.0f;
    }
  }
  return 1;
}

EMSCRIPTEN_KEEPALIVE
int add_nhwc_in_place(float *destination, const float *source, int length) {
  if (destination == NULL || source == NULL || length <= 0) {
    return 0;
  }
  int index = 0;
  for (; index + 3 < length; index += 4) {
    const v128_t left = wasm_v128_load(destination + index);
    const v128_t right = wasm_v128_load(source + index);
    wasm_v128_store(destination + index, wasm_f32x4_add(left, right));
  }
  for (; index < length; ++index) {
    destination[index] += source[index];
  }
  return 1;
}

EMSCRIPTEN_KEEPALIVE
int add_elu_nhwc_in_place(float *destination, const float *source,
                           int length) {
  if (destination == NULL || source == NULL || length <= 0) {
    return 0;
  }
  int index = 0;
  for (; index + 3 < length; index += 4) {
    const v128_t sum = wasm_f32x4_add(wasm_v128_load(destination + index),
                                      wasm_v128_load(source + index));
    wasm_v128_store(destination + index, elu_f32x4(sum));
  }
  for (; index < length; ++index) {
    const float value = destination[index] + source[index];
    destination[index] = value < 0.0f ? expm1f(value) : value;
  }
  return 1;
}

/* Decode codebook-major residual-quantizer codes into time-major vectors. */
EMSCRIPTEN_KEEPALIVE
int rvq_decode_codes_nhwc(const uint16_t *codes, const float *embeddings,
                          float *output, int time, int dimension, int entries,
                          int codebooks) {
  if (codes == NULL || embeddings == NULL || output == NULL || time <= 0 ||
      dimension <= 0 || (dimension % 4) != 0 || entries <= 0 ||
      codebooks <= 0) {
    return 0;
  }

  for (int time_index = 0; time_index < time; ++time_index) {
    float *output_row = output + (size_t)time_index * dimension;
    for (int channel = 0; channel < dimension; channel += 4) {
      v128_t value = wasm_f32x4_splat(0.0f);
      for (int codebook = 0; codebook < codebooks; ++codebook) {
        const int code = codes[(size_t)codebook * time + time_index];
        if (code >= entries) {
          return 0;
        }
        const float *embedding =
            embeddings +
            ((size_t)codebook * entries + code) * dimension + channel;
        value = wasm_f32x4_add(value, wasm_v128_load(embedding));
      }
      wasm_v128_store(output_row + channel, value);
    }
  }
  return 1;
}

EMSCRIPTEN_KEEPALIVE
int nhwc_to_nct(const float *input, float *output, int time, int channels) {
  if (input == NULL || output == NULL || time <= 0 || channels <= 0) {
    return 0;
  }
  for (int channel = 0; channel < channels; ++channel) {
    float *output_channel = output + (size_t)channel * time;
    for (int time_index = 0; time_index < time; ++time_index) {
      output_channel[time_index] =
          input[(size_t)time_index * channels + channel];
    }
  }
  return 1;
}

EMSCRIPTEN_KEEPALIVE
int compact_nhwc_channels(const float *input, float *output, int time,
                          int input_channels, int output_channels) {
  if (input == NULL || output == NULL || time <= 0 || input_channels <= 0 ||
      output_channels <= 0 || output_channels > input_channels) {
    return 0;
  }
  for (int time_index = 0; time_index < time; ++time_index) {
    const float *source = input + (size_t)time_index * input_channels;
    float *destination = output + (size_t)time_index * output_channels;
    for (int channel = 0; channel < output_channels; ++channel) {
      destination[channel] = source[channel];
    }
  }
  return 1;
}

EMSCRIPTEN_KEEPALIVE
int scale_nhwc_to_nct(const float *input, float *output, float scale,
                      int time, int channels) {
  if (input == NULL || output == NULL || time <= 0 || channels <= 0) {
    return 0;
  }
  for (int channel = 0; channel < channels; ++channel) {
    float *output_channel = output + (size_t)channel * time;
    for (int time_index = 0; time_index < time; ++time_index) {
      output_channel[time_index] =
          input[(size_t)time_index * channels + channel] * scale;
    }
  }
  return 1;
}

static float sigmoid_scalar(float value) {
  return 1.0f / (1.0f + expf(-value));
}

static v128_t sigmoid_f32x4(v128_t value) {
  const v128_t one = wasm_f32x4_splat(1.0f);
  return wasm_f32x4_div(
      one, wasm_f32x4_add(one, exp_f32x4_polynomial(wasm_f32x4_neg(value))));
}

static v128_t tanh_f32x4_polynomial(v128_t value) {
  const v128_t one = wasm_f32x4_splat(1.0f);
  return wasm_f32x4_sub(
      wasm_f32x4_mul(wasm_f32x4_splat(2.0f),
                     sigmoid_f32x4(wasm_f32x4_mul(
                         wasm_f32x4_splat(2.0f), value))),
      one);
}

/*
 * Run one forward LSTM layer. ONNX stores the gates in input, output, forget,
 * and cell order. Input projections run for the full sequence before the
 * recurrent scan so the input matrix can reuse weights across time.
 */
EMSCRIPTEN_KEEPALIVE
int lstm_layer_simd_64(const float *input, const float *packed_input,
                       const float *packed_recurrent, const float *bias,
                       float *output, float *hidden_state, float *cell_state,
                       float *input_projection, int sequence_length,
                       int hidden_size) {
  if (input == NULL || packed_input == NULL || packed_recurrent == NULL ||
      bias == NULL || output == NULL ||
      hidden_state == NULL || cell_state == NULL || input_projection == NULL ||
      sequence_length <= 0 || hidden_size <= 0 ||
      ((4 * hidden_size) % 64) != 0) {
    return 0;
  }

  const int gate_size = 4 * hidden_size;
  if (!conv1d_nhwc_simd_8x8(input, packed_input, bias, input_projection,
                             sequence_length, hidden_size, gate_size, 1, 1)) {
    return 0;
  }
  for (int index = 0; index < hidden_size; ++index) {
    hidden_state[index] = 0.0f;
    cell_state[index] = 0.0f;
  }

  for (int time = 0; time < sequence_length; ++time) {
    float *gates = input_projection + (size_t)time * gate_size;
    for (int gate = 0; gate < gate_size; gate += 64) {
      v128_t low0 = wasm_v128_load(gates + gate);
      v128_t high0 = wasm_v128_load(gates + gate + 4);
      v128_t low1 = wasm_v128_load(gates + gate + 8);
      v128_t high1 = wasm_v128_load(gates + gate + 12);
      v128_t low2 = wasm_v128_load(gates + gate + 16);
      v128_t high2 = wasm_v128_load(gates + gate + 20);
      v128_t low3 = wasm_v128_load(gates + gate + 24);
      v128_t high3 = wasm_v128_load(gates + gate + 28);
      v128_t low4 = wasm_v128_load(gates + gate + 32);
      v128_t high4 = wasm_v128_load(gates + gate + 36);
      v128_t low5 = wasm_v128_load(gates + gate + 40);
      v128_t high5 = wasm_v128_load(gates + gate + 44);
      v128_t low6 = wasm_v128_load(gates + gate + 48);
      v128_t high6 = wasm_v128_load(gates + gate + 52);
      v128_t low7 = wasm_v128_load(gates + gate + 56);
      v128_t high7 = wasm_v128_load(gates + gate + 60);
      const size_t block0 = (size_t)(gate / 8) * hidden_size * 8;
      const size_t block1 = block0 + (size_t)hidden_size * 8;
      const size_t block2 = block1 + (size_t)hidden_size * 8;
      const size_t block3 = block2 + (size_t)hidden_size * 8;
      const size_t block4 = block3 + (size_t)hidden_size * 8;
      const size_t block5 = block4 + (size_t)hidden_size * 8;
      const size_t block6 = block5 + (size_t)hidden_size * 8;
      const size_t block7 = block6 + (size_t)hidden_size * 8;

      for (int reduction = 0; reduction < hidden_size; ++reduction) {
        const v128_t sample = wasm_f32x4_splat(hidden_state[reduction]);
#define ACCUMULATE_LSTM_BLOCK(number, block)                                    \
  do {                                                                          \
    const float *weights =                                                      \
        packed_recurrent + (block) + (size_t)reduction * 8;                    \
    low##number = wasm_f32x4_add(                                               \
        low##number, wasm_f32x4_mul(sample, wasm_v128_load(weights)));          \
    high##number = wasm_f32x4_add(                                              \
        high##number, wasm_f32x4_mul(sample, wasm_v128_load(weights + 4)));     \
  } while (0)
        ACCUMULATE_LSTM_BLOCK(0, block0);
        ACCUMULATE_LSTM_BLOCK(1, block1);
        ACCUMULATE_LSTM_BLOCK(2, block2);
        ACCUMULATE_LSTM_BLOCK(3, block3);
        ACCUMULATE_LSTM_BLOCK(4, block4);
        ACCUMULATE_LSTM_BLOCK(5, block5);
        ACCUMULATE_LSTM_BLOCK(6, block6);
        ACCUMULATE_LSTM_BLOCK(7, block7);
#undef ACCUMULATE_LSTM_BLOCK
      }
      wasm_v128_store(gates + gate, low0);
      wasm_v128_store(gates + gate + 4, high0);
      wasm_v128_store(gates + gate + 8, low1);
      wasm_v128_store(gates + gate + 12, high1);
      wasm_v128_store(gates + gate + 16, low2);
      wasm_v128_store(gates + gate + 20, high2);
      wasm_v128_store(gates + gate + 24, low3);
      wasm_v128_store(gates + gate + 28, high3);
      wasm_v128_store(gates + gate + 32, low4);
      wasm_v128_store(gates + gate + 36, high4);
      wasm_v128_store(gates + gate + 40, low5);
      wasm_v128_store(gates + gate + 44, high5);
      wasm_v128_store(gates + gate + 48, low6);
      wasm_v128_store(gates + gate + 52, high6);
      wasm_v128_store(gates + gate + 56, low7);
      wasm_v128_store(gates + gate + 60, high7);
    }

    float *output_row = output + (size_t)time * hidden_size;
    int index = 0;
    for (; index + 3 < hidden_size; index += 4) {
      const v128_t input_gate = sigmoid_f32x4(wasm_v128_load(gates + index));
      const v128_t output_gate =
          sigmoid_f32x4(wasm_v128_load(gates + hidden_size + index));
      const v128_t forget_gate =
          sigmoid_f32x4(wasm_v128_load(gates + 2 * hidden_size + index));
      const v128_t cell_gate = tanh_f32x4_polynomial(
          wasm_v128_load(gates + 3 * hidden_size + index));
      const v128_t cell = wasm_f32x4_add(
          wasm_f32x4_mul(forget_gate, wasm_v128_load(cell_state + index)),
          wasm_f32x4_mul(input_gate, cell_gate));
      const v128_t hidden =
          wasm_f32x4_mul(output_gate, tanh_f32x4_polynomial(cell));
      wasm_v128_store(cell_state + index, cell);
      wasm_v128_store(hidden_state + index, hidden);
      wasm_v128_store(output_row + index, hidden);
    }
    for (; index < hidden_size; ++index) {
      const float input_gate = sigmoid_scalar(gates[index]);
      const float output_gate = sigmoid_scalar(gates[hidden_size + index]);
      const float forget_gate = sigmoid_scalar(gates[2 * hidden_size + index]);
      const float cell_gate = tanhf(gates[3 * hidden_size + index]);
      const float cell = forget_gate * cell_state[index] +
                         input_gate * cell_gate;
      const float hidden = output_gate * tanhf(cell);
      cell_state[index] = cell;
      hidden_state[index] = hidden;
      output_row[index] = hidden;
    }
  }
  return 1;
}

static float horizontal_sum_f32x4(v128_t value) {
  return wasm_f32x4_extract_lane(value, 0) +
         wasm_f32x4_extract_lane(value, 1) +
         wasm_f32x4_extract_lane(value, 2) +
         wasm_f32x4_extract_lane(value, 3);
}

EMSCRIPTEN_KEEPALIVE
int rvq_encode_simd_8(const float *input, float *residual,
                      const float *embeddings, const float *embedding_norms,
                      int32_t *codes, int time, int dimension, int entries,
                      int codebooks) {
  if (input == NULL || residual == NULL || embeddings == NULL ||
      embedding_norms == NULL || codes == NULL || time <= 0 ||
      dimension <= 0 || (dimension % 4) != 0 || entries <= 0 ||
      (entries % 8) != 0 || codebooks <= 0) {
    return 0;
  }

  const size_t activation_length = (size_t)time * dimension;
  for (size_t index = 0; index < activation_length; index += 4) {
    wasm_v128_store(residual + index, wasm_v128_load(input + index));
  }

  for (int codebook = 0; codebook < codebooks; ++codebook) {
    const float *book =
        embeddings + (size_t)codebook * entries * dimension;
    const float *norms = embedding_norms + (size_t)codebook * entries;
    for (int time_index = 0; time_index < time; ++time_index) {
      float *vector = residual + (size_t)time_index * dimension;
      float best_score = -INFINITY;
      int best_code = 0;
      for (int code = 0; code < entries; code += 8) {
        const float *embedding0 = book + (size_t)code * dimension;
        const float *embedding1 = embedding0 + dimension;
        const float *embedding2 = embedding1 + dimension;
        const float *embedding3 = embedding2 + dimension;
        const float *embedding4 = embedding3 + dimension;
        const float *embedding5 = embedding4 + dimension;
        const float *embedding6 = embedding5 + dimension;
        const float *embedding7 = embedding6 + dimension;
        v128_t dot0 = wasm_f32x4_splat(0.0f);
        v128_t dot1 = wasm_f32x4_splat(0.0f);
        v128_t dot2 = wasm_f32x4_splat(0.0f);
        v128_t dot3 = wasm_f32x4_splat(0.0f);
        v128_t dot4 = wasm_f32x4_splat(0.0f);
        v128_t dot5 = wasm_f32x4_splat(0.0f);
        v128_t dot6 = wasm_f32x4_splat(0.0f);
        v128_t dot7 = wasm_f32x4_splat(0.0f);
        for (int channel = 0; channel < dimension; channel += 4) {
          const v128_t sample = wasm_v128_load(vector + channel);
          dot0 = wasm_f32x4_add(
              dot0, wasm_f32x4_mul(sample, wasm_v128_load(embedding0 + channel)));
          dot1 = wasm_f32x4_add(
              dot1, wasm_f32x4_mul(sample, wasm_v128_load(embedding1 + channel)));
          dot2 = wasm_f32x4_add(
              dot2, wasm_f32x4_mul(sample, wasm_v128_load(embedding2 + channel)));
          dot3 = wasm_f32x4_add(
              dot3, wasm_f32x4_mul(sample, wasm_v128_load(embedding3 + channel)));
          dot4 = wasm_f32x4_add(
              dot4, wasm_f32x4_mul(sample, wasm_v128_load(embedding4 + channel)));
          dot5 = wasm_f32x4_add(
              dot5, wasm_f32x4_mul(sample, wasm_v128_load(embedding5 + channel)));
          dot6 = wasm_f32x4_add(
              dot6, wasm_f32x4_mul(sample, wasm_v128_load(embedding6 + channel)));
          dot7 = wasm_f32x4_add(
              dot7, wasm_f32x4_mul(sample, wasm_v128_load(embedding7 + channel)));
        }
        const float score0 = 2.0f * horizontal_sum_f32x4(dot0) - norms[code];
        const float score1 =
            2.0f * horizontal_sum_f32x4(dot1) - norms[code + 1];
        const float score2 =
            2.0f * horizontal_sum_f32x4(dot2) - norms[code + 2];
        const float score3 =
            2.0f * horizontal_sum_f32x4(dot3) - norms[code + 3];
        const float score4 =
            2.0f * horizontal_sum_f32x4(dot4) - norms[code + 4];
        const float score5 =
            2.0f * horizontal_sum_f32x4(dot5) - norms[code + 5];
        const float score6 =
            2.0f * horizontal_sum_f32x4(dot6) - norms[code + 6];
        const float score7 =
            2.0f * horizontal_sum_f32x4(dot7) - norms[code + 7];
#define UPDATE_BEST(offset)                                                     \
  do {                                                                          \
    if (score##offset > best_score) {                                           \
      best_score = score##offset;                                               \
      best_code = code + (offset);                                              \
    }                                                                           \
  } while (0)
        UPDATE_BEST(0);
        UPDATE_BEST(1);
        UPDATE_BEST(2);
        UPDATE_BEST(3);
        UPDATE_BEST(4);
        UPDATE_BEST(5);
        UPDATE_BEST(6);
        UPDATE_BEST(7);
#undef UPDATE_BEST
      }
      codes[(size_t)codebook * time + time_index] = best_code;
      const float *selected = book + (size_t)best_code * dimension;
      for (int channel = 0; channel < dimension; channel += 4) {
        wasm_v128_store(
            vector + channel,
            wasm_f32x4_sub(wasm_v128_load(vector + channel),
                           wasm_v128_load(selected + channel)));
      }
    }
  }
  return 1;
}

EMSCRIPTEN_KEEPALIVE
int conv1d_nhwc_scalar(const float *input, const float *packed,
                       const float *bias, float *output, int padded_time,
                       int input_channels, int output_channels, int kernel,
                       int stride) {
  if (input == NULL || packed == NULL || bias == NULL || output == NULL ||
      !valid_conv_geometry(padded_time, input_channels, output_channels,
                           kernel, stride)) {
    return 0;
  }

  const int output_time = (padded_time - kernel) / stride + 1;
  const size_t reduction = (size_t)input_channels * kernel;
  for (int time = 0; time < output_time; ++time) {
    for (int output_channel = 0; output_channel < output_channels;
         ++output_channel) {
      const int output_block = output_channel / 8;
      const int output_lane = output_channel % 8;
      const size_t weight_block = (size_t)output_block * reduction * 8;
      float value = bias[output_channel];
      for (int input_channel = 0; input_channel < input_channels;
           ++input_channel) {
        for (int tap = 0; tap < kernel; ++tap) {
          const size_t input_index =
              (size_t)(time * stride + tap) * input_channels + input_channel;
          const size_t reduction_index =
              (size_t)input_channel * kernel + tap;
          value += input[input_index] *
                   packed[weight_block + reduction_index * 8 + output_lane];
        }
      }
      output[(size_t)time * output_channels + output_channel] = value;
    }
  }
  return 1;
}

EMSCRIPTEN_KEEPALIVE
int conv1d_nhwc_simd_8x8(const float *input, const float *packed,
                          const float *bias, float *output, int padded_time,
                          int input_channels, int output_channels, int kernel,
                          int stride) {
  if (input == NULL || packed == NULL || bias == NULL || output == NULL ||
      !valid_conv_geometry(padded_time, input_channels, output_channels,
                           kernel, stride)) {
    return 0;
  }

  const int output_time = (padded_time - kernel) / stride + 1;
  const size_t reduction = (size_t)input_channels * kernel;
  for (int output_channel = 0; output_channel < output_channels;
       output_channel += 8) {
    const size_t weight_block =
        (size_t)(output_channel / 8) * reduction * 8;
    const v128_t bias_low = wasm_v128_load(bias + output_channel);
    const v128_t bias_high = wasm_v128_load(bias + output_channel + 4);
    int time = 0;
    for (; time + 7 < output_time; time += 8) {
      v128_t low0 = bias_low;
      v128_t high0 = bias_high;
      v128_t low1 = bias_low;
      v128_t high1 = bias_high;
      v128_t low2 = bias_low;
      v128_t high2 = bias_high;
      v128_t low3 = bias_low;
      v128_t high3 = bias_high;
      v128_t low4 = bias_low;
      v128_t high4 = bias_high;
      v128_t low5 = bias_low;
      v128_t high5 = bias_high;
      v128_t low6 = bias_low;
      v128_t high6 = bias_high;
      v128_t low7 = bias_low;
      v128_t high7 = bias_high;

      for (int input_channel = 0; input_channel < input_channels;
           ++input_channel) {
        for (int tap = 0; tap < kernel; ++tap) {
          const size_t reduction_index =
              (size_t)input_channel * kernel + tap;
          const float *weights = packed + weight_block + reduction_index * 8;
          const v128_t weight_low = wasm_v128_load(weights);
          const v128_t weight_high = wasm_v128_load(weights + 4);
#define ACCUMULATE_TIME(offset)                                                 \
  do {                                                                          \
    const size_t input_index =                                                  \
        (size_t)((time + (offset)) * stride + tap) * input_channels +          \
        input_channel;                                                          \
    const v128_t sample = wasm_f32x4_splat(input[input_index]);                 \
    low##offset = wasm_f32x4_add(                                               \
        low##offset, wasm_f32x4_mul(sample, weight_low));                       \
    high##offset = wasm_f32x4_add(                                              \
        high##offset, wasm_f32x4_mul(sample, weight_high));                     \
  } while (0)
          ACCUMULATE_TIME(0);
          ACCUMULATE_TIME(1);
          ACCUMULATE_TIME(2);
          ACCUMULATE_TIME(3);
          ACCUMULATE_TIME(4);
          ACCUMULATE_TIME(5);
          ACCUMULATE_TIME(6);
          ACCUMULATE_TIME(7);
#undef ACCUMULATE_TIME
        }
      }

#define STORE_TIME(offset)                                                      \
  do {                                                                          \
    float *destination =                                                        \
        output + (size_t)(time + (offset)) * output_channels + output_channel; \
    wasm_v128_store(destination, low##offset);                                  \
    wasm_v128_store(destination + 4, high##offset);                             \
  } while (0)
      STORE_TIME(0);
      STORE_TIME(1);
      STORE_TIME(2);
      STORE_TIME(3);
      STORE_TIME(4);
      STORE_TIME(5);
      STORE_TIME(6);
      STORE_TIME(7);
#undef STORE_TIME
    }

    for (; time < output_time; ++time) {
      v128_t low = bias_low;
      v128_t high = bias_high;
      for (int input_channel = 0; input_channel < input_channels;
           ++input_channel) {
        for (int tap = 0; tap < kernel; ++tap) {
          const size_t reduction_index =
              (size_t)input_channel * kernel + tap;
          const float *weights = packed + weight_block + reduction_index * 8;
          const v128_t weight_low = wasm_v128_load(weights);
          const v128_t weight_high = wasm_v128_load(weights + 4);
          const size_t input_index =
              (size_t)(time * stride + tap) * input_channels + input_channel;
          const v128_t sample = wasm_f32x4_splat(input[input_index]);
          low = wasm_f32x4_add(low, wasm_f32x4_mul(sample, weight_low));
          high = wasm_f32x4_add(high, wasm_f32x4_mul(sample, weight_high));
        }
      }
      float *destination =
          output + (size_t)time * output_channels + output_channel;
      wasm_v128_store(destination, low);
      wasm_v128_store(destination + 4, high);
    }
  }
  return 1;
}
