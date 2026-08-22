#include <emscripten/emscripten.h>
#include <stddef.h>
#include <stdint.h>
#include <wasm_simd128.h>

static int valid_geometry(int time, int input_channels, int output_channels,
                          int stride) {
  return time > 0 && input_channels > 0 && output_channels > 0 &&
         (output_channels % 4) == 0 && stride > 0;
}

/*
 * Convert ONNX [input_channel, output_channel, 2 * stride] weights into
 * [phase, tap, input_channel, output_channel]. Four adjacent output-channel
 * weights can then be loaded with one v128 load.
 */
EMSCRIPTEN_KEEPALIVE
int pack_conv_transpose1d_weights(const float *weights, float *packed,
                                  int input_channels, int output_channels,
                                  int stride) {
  if (weights == NULL || packed == NULL ||
      !valid_geometry(1, input_channels, output_channels, stride)) {
    return 0;
  }

  const int kernel = stride * 2;
  for (int phase = 0; phase < stride; ++phase) {
    for (int tap = 0; tap < 2; ++tap) {
      const int kernel_index = phase + tap * stride;
      for (int input_channel = 0; input_channel < input_channels;
           ++input_channel) {
        for (int output_channel = 0; output_channel < output_channels;
             ++output_channel) {
          const size_t source =
              ((size_t)input_channel * output_channels + output_channel) *
                  kernel +
              kernel_index;
          const size_t destination =
              (((size_t)phase * 2 + tap) * input_channels + input_channel) *
                  output_channels +
              output_channel;
          packed[destination] = weights[source];
        }
      }
    }
  }
  return 1;
}

EMSCRIPTEN_KEEPALIVE
int conv_transpose1d_phase_scalar(const float *input, const float *packed,
                                  const float *bias, float *output, int time,
                                  int input_channels, int output_channels,
                                  int stride) {
  if (input == NULL || packed == NULL || bias == NULL || output == NULL ||
      !valid_geometry(time, input_channels, output_channels, stride)) {
    return 0;
  }

  const int output_time = (time + 1) * stride;
  for (int output_channel = 0; output_channel < output_channels;
       ++output_channel) {
    for (int frame = 0; frame <= time; ++frame) {
      for (int phase = 0; phase < stride; ++phase) {
        float value = bias[output_channel];
        if (frame < time) {
          const size_t weight_base =
              ((size_t)phase * 2 * input_channels) * output_channels;
          for (int input_channel = 0; input_channel < input_channels;
               ++input_channel) {
            value += input[(size_t)input_channel * time + frame] *
                     packed[weight_base +
                            (size_t)input_channel * output_channels +
                            output_channel];
          }
        }
        if (frame > 0) {
          const size_t weight_base =
              (((size_t)phase * 2 + 1) * input_channels) * output_channels;
          for (int input_channel = 0; input_channel < input_channels;
               ++input_channel) {
            value += input[(size_t)input_channel * time + frame - 1] *
                     packed[weight_base +
                            (size_t)input_channel * output_channels +
                            output_channel];
          }
        }
        output[(size_t)output_channel * output_time + frame * stride + phase] =
            value;
      }
    }
  }
  return 1;
}

EMSCRIPTEN_KEEPALIVE
int conv_transpose1d_phase_simd(const float *input, const float *packed,
                                const float *bias, float *output, int time,
                                int input_channels, int output_channels,
                                int stride) {
  if (input == NULL || packed == NULL || bias == NULL || output == NULL ||
      !valid_geometry(time, input_channels, output_channels, stride)) {
    return 0;
  }

  const int output_time = (time + 1) * stride;
  for (int frame = 0; frame <= time; ++frame) {
    for (int phase = 0; phase < stride; ++phase) {
      for (int output_channel = 0; output_channel < output_channels;
           output_channel += 4) {
        v128_t value = wasm_v128_load(bias + output_channel);
        if (frame < time) {
          const size_t weight_base =
              ((size_t)phase * 2 * input_channels) * output_channels;
          for (int input_channel = 0; input_channel < input_channels;
               ++input_channel) {
            const v128_t sample = wasm_f32x4_splat(
                input[(size_t)input_channel * time + frame]);
            const v128_t weight = wasm_v128_load(
                packed + weight_base +
                (size_t)input_channel * output_channels + output_channel);
            value = wasm_f32x4_add(value, wasm_f32x4_mul(sample, weight));
          }
        }
        if (frame > 0) {
          const size_t weight_base =
              (((size_t)phase * 2 + 1) * input_channels) * output_channels;
          for (int input_channel = 0; input_channel < input_channels;
               ++input_channel) {
            const v128_t sample = wasm_f32x4_splat(
                input[(size_t)input_channel * time + frame - 1]);
            const v128_t weight = wasm_v128_load(
                packed + weight_base +
                (size_t)input_channel * output_channels + output_channel);
            value = wasm_f32x4_add(value, wasm_f32x4_mul(sample, weight));
          }
        }

        const size_t output_index = (size_t)frame * stride + phase;
        output[(size_t)(output_channel + 0) * output_time + output_index] =
            wasm_f32x4_extract_lane(value, 0);
        output[(size_t)(output_channel + 1) * output_time + output_index] =
            wasm_f32x4_extract_lane(value, 1);
        output[(size_t)(output_channel + 2) * output_time + output_index] =
            wasm_f32x4_extract_lane(value, 2);
        output[(size_t)(output_channel + 3) * output_time + output_index] =
            wasm_f32x4_extract_lane(value, 3);
      }
    }
  }
  return 1;
}

EMSCRIPTEN_KEEPALIVE
int conv_transpose1d_phase_simd_8x4(const float *input, const float *packed,
                                    const float *bias, float *output, int time,
                                    int input_channels, int output_channels,
                                    int stride) {
  if (input == NULL || packed == NULL || bias == NULL || output == NULL ||
      !valid_geometry(time, input_channels, output_channels, stride)) {
    return 0;
  }

  const int output_time = (time + 1) * stride;
  for (int phase = 0; phase < stride; ++phase) {
    const size_t current_weight_base =
        ((size_t)phase * 2 * input_channels) * output_channels;
    const size_t previous_weight_base =
        (((size_t)phase * 2 + 1) * input_channels) * output_channels;
    for (int output_channel = 0; output_channel < output_channels;
         output_channel += 4) {
      const v128_t initial = wasm_v128_load(bias + output_channel);

      /* The first phase group has no previous input frame. */
      v128_t first = initial;
      for (int input_channel = 0; input_channel < input_channels;
           ++input_channel) {
        const v128_t sample =
            wasm_f32x4_splat(input[(size_t)input_channel * time]);
        const v128_t weight = wasm_v128_load(
            packed + current_weight_base +
            (size_t)input_channel * output_channels + output_channel);
        first = wasm_f32x4_add(first, wasm_f32x4_mul(sample, weight));
      }
      const size_t first_output = phase;
      output[(size_t)(output_channel + 0) * output_time + first_output] =
          wasm_f32x4_extract_lane(first, 0);
      output[(size_t)(output_channel + 1) * output_time + first_output] =
          wasm_f32x4_extract_lane(first, 1);
      output[(size_t)(output_channel + 2) * output_time + first_output] =
          wasm_f32x4_extract_lane(first, 2);
      output[(size_t)(output_channel + 3) * output_time + first_output] =
          wasm_f32x4_extract_lane(first, 3);

      int frame = 1;
      for (; frame + 7 < time; frame += 8) {
        v128_t value0 = initial;
        v128_t value1 = initial;
        v128_t value2 = initial;
        v128_t value3 = initial;
        v128_t value4 = initial;
        v128_t value5 = initial;
        v128_t value6 = initial;
        v128_t value7 = initial;
        for (int input_channel = 0; input_channel < input_channels;
             ++input_channel) {
          const size_t input_base = (size_t)input_channel * time + frame;
          const v128_t current_weight = wasm_v128_load(
              packed + current_weight_base +
              (size_t)input_channel * output_channels + output_channel);
          const v128_t previous_weight = wasm_v128_load(
              packed + previous_weight_base +
              (size_t)input_channel * output_channels + output_channel);
          value0 = wasm_f32x4_add(
              value0,
              wasm_f32x4_mul(wasm_f32x4_splat(input[input_base + 0]),
                             current_weight));
          value0 = wasm_f32x4_add(
              value0,
              wasm_f32x4_mul(wasm_f32x4_splat(input[input_base - 1]),
                             previous_weight));
          value1 = wasm_f32x4_add(
              value1,
              wasm_f32x4_mul(wasm_f32x4_splat(input[input_base + 1]),
                             current_weight));
          value1 = wasm_f32x4_add(
              value1,
              wasm_f32x4_mul(wasm_f32x4_splat(input[input_base + 0]),
                             previous_weight));
          value2 = wasm_f32x4_add(
              value2,
              wasm_f32x4_mul(wasm_f32x4_splat(input[input_base + 2]),
                             current_weight));
          value2 = wasm_f32x4_add(
              value2,
              wasm_f32x4_mul(wasm_f32x4_splat(input[input_base + 1]),
                             previous_weight));
          value3 = wasm_f32x4_add(
              value3,
              wasm_f32x4_mul(wasm_f32x4_splat(input[input_base + 3]),
                             current_weight));
          value3 = wasm_f32x4_add(
              value3,
              wasm_f32x4_mul(wasm_f32x4_splat(input[input_base + 2]),
                             previous_weight));
          value4 = wasm_f32x4_add(
              value4,
              wasm_f32x4_mul(wasm_f32x4_splat(input[input_base + 4]),
                             current_weight));
          value4 = wasm_f32x4_add(
              value4,
              wasm_f32x4_mul(wasm_f32x4_splat(input[input_base + 3]),
                             previous_weight));
          value5 = wasm_f32x4_add(
              value5,
              wasm_f32x4_mul(wasm_f32x4_splat(input[input_base + 5]),
                             current_weight));
          value5 = wasm_f32x4_add(
              value5,
              wasm_f32x4_mul(wasm_f32x4_splat(input[input_base + 4]),
                             previous_weight));
          value6 = wasm_f32x4_add(
              value6,
              wasm_f32x4_mul(wasm_f32x4_splat(input[input_base + 6]),
                             current_weight));
          value6 = wasm_f32x4_add(
              value6,
              wasm_f32x4_mul(wasm_f32x4_splat(input[input_base + 5]),
                             previous_weight));
          value7 = wasm_f32x4_add(
              value7,
              wasm_f32x4_mul(wasm_f32x4_splat(input[input_base + 7]),
                             current_weight));
          value7 = wasm_f32x4_add(
              value7,
              wasm_f32x4_mul(wasm_f32x4_splat(input[input_base + 6]),
                             previous_weight));
        }

#define STORE_FRAME_RESULT(value, frame_offset)                                \
  do {                                                                         \
    const size_t output_index =                                                \
        (size_t)(frame + (frame_offset)) * stride + phase;                     \
    output[(size_t)(output_channel + 0) * output_time + output_index] =         \
        wasm_f32x4_extract_lane((value), 0);                                   \
    output[(size_t)(output_channel + 1) * output_time + output_index] =         \
        wasm_f32x4_extract_lane((value), 1);                                   \
    output[(size_t)(output_channel + 2) * output_time + output_index] =         \
        wasm_f32x4_extract_lane((value), 2);                                   \
    output[(size_t)(output_channel + 3) * output_time + output_index] =         \
        wasm_f32x4_extract_lane((value), 3);                                   \
  } while (0)
        STORE_FRAME_RESULT(value0, 0);
        STORE_FRAME_RESULT(value1, 1);
        STORE_FRAME_RESULT(value2, 2);
        STORE_FRAME_RESULT(value3, 3);
        STORE_FRAME_RESULT(value4, 4);
        STORE_FRAME_RESULT(value5, 5);
        STORE_FRAME_RESULT(value6, 6);
        STORE_FRAME_RESULT(value7, 7);
#undef STORE_FRAME_RESULT
      }

      /* Finish any complete interior frames not covered by an eight-frame tile. */
      for (; frame < time; ++frame) {
        v128_t value = initial;
        for (int input_channel = 0; input_channel < input_channels;
             ++input_channel) {
          const size_t input_index = (size_t)input_channel * time + frame;
          const v128_t current_weight = wasm_v128_load(
              packed + current_weight_base +
              (size_t)input_channel * output_channels + output_channel);
          const v128_t previous_weight = wasm_v128_load(
              packed + previous_weight_base +
              (size_t)input_channel * output_channels + output_channel);
          value = wasm_f32x4_add(
              value,
              wasm_f32x4_mul(wasm_f32x4_splat(input[input_index]),
                             current_weight));
          value = wasm_f32x4_add(
              value,
              wasm_f32x4_mul(wasm_f32x4_splat(input[input_index - 1]),
                             previous_weight));
        }
        const size_t output_index = (size_t)frame * stride + phase;
        output[(size_t)(output_channel + 0) * output_time + output_index] =
            wasm_f32x4_extract_lane(value, 0);
        output[(size_t)(output_channel + 1) * output_time + output_index] =
            wasm_f32x4_extract_lane(value, 1);
        output[(size_t)(output_channel + 2) * output_time + output_index] =
            wasm_f32x4_extract_lane(value, 2);
        output[(size_t)(output_channel + 3) * output_time + output_index] =
            wasm_f32x4_extract_lane(value, 3);
      }

      /* The last phase group has no current input frame. */
      v128_t last = initial;
      for (int input_channel = 0; input_channel < input_channels;
           ++input_channel) {
        const v128_t sample = wasm_f32x4_splat(
            input[(size_t)input_channel * time + time - 1]);
        const v128_t weight = wasm_v128_load(
            packed + previous_weight_base +
            (size_t)input_channel * output_channels + output_channel);
        last = wasm_f32x4_add(last, wasm_f32x4_mul(sample, weight));
      }
      const size_t last_output = (size_t)time * stride + phase;
      output[(size_t)(output_channel + 0) * output_time + last_output] =
          wasm_f32x4_extract_lane(last, 0);
      output[(size_t)(output_channel + 1) * output_time + last_output] =
          wasm_f32x4_extract_lane(last, 1);
      output[(size_t)(output_channel + 2) * output_time + last_output] =
          wasm_f32x4_extract_lane(last, 2);
      output[(size_t)(output_channel + 3) * output_time + last_output] =
          wasm_f32x4_extract_lane(last, 3);
    }
  }
  return 1;
}

/*
 * Process eight time positions and eight output channels in each interior
 * tile. The second channel vector reuses all input loads and broadcasts.
 */
EMSCRIPTEN_KEEPALIVE
int conv_transpose1d_phase_simd_8x8(const float *input, const float *packed,
                                    const float *bias, float *output, int time,
                                    int input_channels, int output_channels,
                                    int stride) {
  if (input == NULL || packed == NULL || bias == NULL || output == NULL ||
      !valid_geometry(time, input_channels, output_channels, stride) ||
      (output_channels % 8) != 0) {
    return 0;
  }

  const int output_time = (time + 1) * stride;
  for (int phase = 0; phase < stride; ++phase) {
    const size_t current_weight_base =
        ((size_t)phase * 2 * input_channels) * output_channels;
    const size_t previous_weight_base =
        (((size_t)phase * 2 + 1) * input_channels) * output_channels;
    for (int output_channel = 0; output_channel < output_channels;
         output_channel += 8) {
      const v128_t initial_low = wasm_v128_load(bias + output_channel);
      const v128_t initial_high = wasm_v128_load(bias + output_channel + 4);

      v128_t first_low = initial_low;
      v128_t first_high = initial_high;
      for (int input_channel = 0; input_channel < input_channels;
           ++input_channel) {
        const v128_t sample =
            wasm_f32x4_splat(input[(size_t)input_channel * time]);
        const size_t weight_index =
            current_weight_base +
            (size_t)input_channel * output_channels + output_channel;
        first_low = wasm_f32x4_add(
            first_low,
            wasm_f32x4_mul(sample, wasm_v128_load(packed + weight_index)));
        first_high = wasm_f32x4_add(
            first_high,
            wasm_f32x4_mul(sample, wasm_v128_load(packed + weight_index + 4)));
      }

#define STORE_OUTPUT_8(low, high, output_index)                                \
  do {                                                                         \
    output[(size_t)(output_channel + 0) * output_time + (output_index)] =       \
        wasm_f32x4_extract_lane((low), 0);                                     \
    output[(size_t)(output_channel + 1) * output_time + (output_index)] =       \
        wasm_f32x4_extract_lane((low), 1);                                     \
    output[(size_t)(output_channel + 2) * output_time + (output_index)] =       \
        wasm_f32x4_extract_lane((low), 2);                                     \
    output[(size_t)(output_channel + 3) * output_time + (output_index)] =       \
        wasm_f32x4_extract_lane((low), 3);                                     \
    output[(size_t)(output_channel + 4) * output_time + (output_index)] =       \
        wasm_f32x4_extract_lane((high), 0);                                    \
    output[(size_t)(output_channel + 5) * output_time + (output_index)] =       \
        wasm_f32x4_extract_lane((high), 1);                                    \
    output[(size_t)(output_channel + 6) * output_time + (output_index)] =       \
        wasm_f32x4_extract_lane((high), 2);                                    \
    output[(size_t)(output_channel + 7) * output_time + (output_index)] =       \
        wasm_f32x4_extract_lane((high), 3);                                    \
  } while (0)

      STORE_OUTPUT_8(first_low, first_high, (size_t)phase);

      int frame = 1;
      for (; frame + 7 < time; frame += 8) {
        v128_t value_low[8];
        v128_t value_high[8];
#pragma clang loop unroll(full)
        for (int tile = 0; tile < 8; ++tile) {
          value_low[tile] = initial_low;
          value_high[tile] = initial_high;
        }

        for (int input_channel = 0; input_channel < input_channels;
             ++input_channel) {
          const size_t input_base = (size_t)input_channel * time + frame;
          const size_t current_index =
              current_weight_base +
              (size_t)input_channel * output_channels + output_channel;
          const size_t previous_index =
              previous_weight_base +
              (size_t)input_channel * output_channels + output_channel;
          const v128_t current_low = wasm_v128_load(packed + current_index);
          const v128_t current_high =
              wasm_v128_load(packed + current_index + 4);
          const v128_t previous_low = wasm_v128_load(packed + previous_index);
          const v128_t previous_high =
              wasm_v128_load(packed + previous_index + 4);

#pragma clang loop unroll(full)
          for (int tile = 0; tile < 8; ++tile) {
            const v128_t current_sample =
                wasm_f32x4_splat(input[input_base + tile]);
            const v128_t previous_sample =
                wasm_f32x4_splat(input[input_base + tile - 1]);
            value_low[tile] = wasm_f32x4_add(
                value_low[tile], wasm_f32x4_mul(current_sample, current_low));
            value_low[tile] = wasm_f32x4_add(
                value_low[tile], wasm_f32x4_mul(previous_sample, previous_low));
            value_high[tile] = wasm_f32x4_add(
                value_high[tile],
                wasm_f32x4_mul(current_sample, current_high));
            value_high[tile] = wasm_f32x4_add(
                value_high[tile],
                wasm_f32x4_mul(previous_sample, previous_high));
          }
        }

#pragma clang loop unroll(full)
        for (int tile = 0; tile < 8; ++tile) {
          const size_t output_index = (size_t)(frame + tile) * stride + phase;
          STORE_OUTPUT_8(value_low[tile], value_high[tile], output_index);
        }
      }

      for (; frame < time; ++frame) {
        v128_t value_low = initial_low;
        v128_t value_high = initial_high;
        for (int input_channel = 0; input_channel < input_channels;
             ++input_channel) {
          const size_t input_index = (size_t)input_channel * time + frame;
          const size_t current_index =
              current_weight_base +
              (size_t)input_channel * output_channels + output_channel;
          const size_t previous_index =
              previous_weight_base +
              (size_t)input_channel * output_channels + output_channel;
          const v128_t current_sample = wasm_f32x4_splat(input[input_index]);
          const v128_t previous_sample =
              wasm_f32x4_splat(input[input_index - 1]);
          value_low = wasm_f32x4_add(
              value_low,
              wasm_f32x4_mul(current_sample,
                             wasm_v128_load(packed + current_index)));
          value_low = wasm_f32x4_add(
              value_low,
              wasm_f32x4_mul(previous_sample,
                             wasm_v128_load(packed + previous_index)));
          value_high = wasm_f32x4_add(
              value_high,
              wasm_f32x4_mul(current_sample,
                             wasm_v128_load(packed + current_index + 4)));
          value_high = wasm_f32x4_add(
              value_high,
              wasm_f32x4_mul(previous_sample,
                             wasm_v128_load(packed + previous_index + 4)));
        }
        const size_t output_index = (size_t)frame * stride + phase;
        STORE_OUTPUT_8(value_low, value_high, output_index);
      }

      v128_t last_low = initial_low;
      v128_t last_high = initial_high;
      for (int input_channel = 0; input_channel < input_channels;
           ++input_channel) {
        const v128_t sample = wasm_f32x4_splat(
            input[(size_t)input_channel * time + time - 1]);
        const size_t weight_index =
            previous_weight_base +
            (size_t)input_channel * output_channels + output_channel;
        last_low = wasm_f32x4_add(
            last_low,
            wasm_f32x4_mul(sample, wasm_v128_load(packed + weight_index)));
        last_high = wasm_f32x4_add(
            last_high,
            wasm_f32x4_mul(sample, wasm_v128_load(packed + weight_index + 4)));
      }
      const size_t last_output = (size_t)time * stride + phase;
      STORE_OUTPUT_8(last_low, last_high, last_output);
#undef STORE_OUTPUT_8
    }
  }
  return 1;
}

/* NHWC variant for an all-custom decoder pipeline. */
EMSCRIPTEN_KEEPALIVE
int conv_transpose1d_phase_simd_8x8_nhwc(
    const float *input, const float *packed, const float *bias, float *output,
    int time, int input_channels, int output_channels, int stride) {
  if (input == NULL || packed == NULL || bias == NULL || output == NULL ||
      !valid_geometry(time, input_channels, output_channels, stride) ||
      (output_channels % 8) != 0) {
    return 0;
  }

  const int output_time = (time + 1) * stride;
  for (int phase = 0; phase < stride; ++phase) {
    const size_t current_weight_base =
        ((size_t)phase * 2 * input_channels) * output_channels;
    const size_t previous_weight_base =
        (((size_t)phase * 2 + 1) * input_channels) * output_channels;
    for (int output_channel = 0; output_channel < output_channels;
         output_channel += 8) {
      const v128_t initial_low = wasm_v128_load(bias + output_channel);
      const v128_t initial_high = wasm_v128_load(bias + output_channel + 4);

      v128_t first_low = initial_low;
      v128_t first_high = initial_high;
      for (int input_channel = 0; input_channel < input_channels;
           ++input_channel) {
        const v128_t sample = wasm_f32x4_splat(input[input_channel]);
        const size_t weight_index =
            current_weight_base +
            (size_t)input_channel * output_channels + output_channel;
        first_low = wasm_f32x4_add(
            first_low,
            wasm_f32x4_mul(sample, wasm_v128_load(packed + weight_index)));
        first_high = wasm_f32x4_add(
            first_high,
            wasm_f32x4_mul(sample, wasm_v128_load(packed + weight_index + 4)));
      }
      float *first_output = output +
                            ((size_t)phase * output_channels + output_channel);
      wasm_v128_store(first_output, first_low);
      wasm_v128_store(first_output + 4, first_high);

      int frame = 1;
      for (; frame + 7 < time; frame += 8) {
        v128_t value_low[8];
        v128_t value_high[8];
#pragma clang loop unroll(full)
        for (int tile = 0; tile < 8; ++tile) {
          value_low[tile] = initial_low;
          value_high[tile] = initial_high;
        }
        for (int input_channel = 0; input_channel < input_channels;
             ++input_channel) {
          const size_t current_index =
              current_weight_base +
              (size_t)input_channel * output_channels + output_channel;
          const size_t previous_index =
              previous_weight_base +
              (size_t)input_channel * output_channels + output_channel;
          const v128_t current_low = wasm_v128_load(packed + current_index);
          const v128_t current_high =
              wasm_v128_load(packed + current_index + 4);
          const v128_t previous_low = wasm_v128_load(packed + previous_index);
          const v128_t previous_high =
              wasm_v128_load(packed + previous_index + 4);
#pragma clang loop unroll(full)
          for (int tile = 0; tile < 8; ++tile) {
            const size_t current_input =
                (size_t)(frame + tile) * input_channels + input_channel;
            const size_t previous_input = current_input - input_channels;
            const v128_t current_sample =
                wasm_f32x4_splat(input[current_input]);
            const v128_t previous_sample =
                wasm_f32x4_splat(input[previous_input]);
            value_low[tile] = wasm_f32x4_add(
                value_low[tile], wasm_f32x4_mul(current_sample, current_low));
            value_low[tile] = wasm_f32x4_add(
                value_low[tile],
                wasm_f32x4_mul(previous_sample, previous_low));
            value_high[tile] = wasm_f32x4_add(
                value_high[tile],
                wasm_f32x4_mul(current_sample, current_high));
            value_high[tile] = wasm_f32x4_add(
                value_high[tile],
                wasm_f32x4_mul(previous_sample, previous_high));
          }
        }
#pragma clang loop unroll(full)
        for (int tile = 0; tile < 8; ++tile) {
          const size_t output_index =
              (size_t)((frame + tile) * stride + phase) * output_channels +
              output_channel;
          wasm_v128_store(output + output_index, value_low[tile]);
          wasm_v128_store(output + output_index + 4, value_high[tile]);
        }
      }

      for (; frame < time; ++frame) {
        v128_t value_low = initial_low;
        v128_t value_high = initial_high;
        for (int input_channel = 0; input_channel < input_channels;
             ++input_channel) {
          const size_t current_input =
              (size_t)frame * input_channels + input_channel;
          const size_t previous_input = current_input - input_channels;
          const size_t current_index =
              current_weight_base +
              (size_t)input_channel * output_channels + output_channel;
          const size_t previous_index =
              previous_weight_base +
              (size_t)input_channel * output_channels + output_channel;
          const v128_t current_sample =
              wasm_f32x4_splat(input[current_input]);
          const v128_t previous_sample =
              wasm_f32x4_splat(input[previous_input]);
          value_low = wasm_f32x4_add(
              value_low,
              wasm_f32x4_mul(current_sample,
                             wasm_v128_load(packed + current_index)));
          value_low = wasm_f32x4_add(
              value_low,
              wasm_f32x4_mul(previous_sample,
                             wasm_v128_load(packed + previous_index)));
          value_high = wasm_f32x4_add(
              value_high,
              wasm_f32x4_mul(current_sample,
                             wasm_v128_load(packed + current_index + 4)));
          value_high = wasm_f32x4_add(
              value_high,
              wasm_f32x4_mul(previous_sample,
                             wasm_v128_load(packed + previous_index + 4)));
        }
        const size_t output_index =
            (size_t)(frame * stride + phase) * output_channels +
            output_channel;
        wasm_v128_store(output + output_index, value_low);
        wasm_v128_store(output + output_index + 4, value_high);
      }

      v128_t last_low = initial_low;
      v128_t last_high = initial_high;
      const float *last_input = input + (size_t)(time - 1) * input_channels;
      for (int input_channel = 0; input_channel < input_channels;
           ++input_channel) {
        const v128_t sample = wasm_f32x4_splat(last_input[input_channel]);
        const size_t weight_index =
            previous_weight_base +
            (size_t)input_channel * output_channels + output_channel;
        last_low = wasm_f32x4_add(
            last_low,
            wasm_f32x4_mul(sample, wasm_v128_load(packed + weight_index)));
        last_high = wasm_f32x4_add(
            last_high,
            wasm_f32x4_mul(sample, wasm_v128_load(packed + weight_index + 4)));
      }
      const size_t last_output =
          (size_t)(time * stride + phase) * output_channels + output_channel;
      wasm_v128_store(output + last_output, last_low);
      wasm_v128_store(output + last_output + 4, last_high);
    }
  }
  return 1;
}

EMSCRIPTEN_KEEPALIVE
int crop_nhwc(const float *input, float *output, int raw_time, int channels,
              int crop_left, int crop_right) {
  if (input == NULL || output == NULL || raw_time <= 0 || channels <= 0 ||
      (channels % 4) != 0 || crop_left < 0 || crop_right < 0 ||
      crop_left + crop_right >= raw_time) {
    return 0;
  }
  const int output_time = raw_time - crop_left - crop_right;
  for (int time = 0; time < output_time; ++time) {
    const float *source = input + (size_t)(time + crop_left) * channels;
    float *destination = output + (size_t)time * channels;
    for (int channel = 0; channel < channels; channel += 4) {
      wasm_v128_store(destination + channel,
                      wasm_v128_load(source + channel));
    }
  }
  return 1;
}
