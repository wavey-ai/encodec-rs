#ifndef ENCODEC_NATIVE_WASM_SIMD128_H
#define ENCODEC_NATIVE_WASM_SIMD128_H

/*
 * Native stand-in for the WebAssembly SIMD128 intrinsics the custom EnCodec
 * kernels use. Only the operations those kernels reference are provided, and
 * `v128_t` is a 32-bit float vector: the integer intrinsics reinterpret lanes
 * exactly as the WASM intrinsics do, so one shared C source compiles for both
 * the Emscripten (WASM SIMD) build and the native Rust FFI build.
 */

#if defined(__wasm__)
#error "the native wasm_simd128 shim must not be used for a wasm target"
#endif

#include <stdint.h>

#if defined(__aarch64__) || defined(_M_ARM64)
#include <arm_neon.h>

typedef float32x4_t v128_t;

static inline v128_t wasm_v128_load(const void *pointer) {
  return vld1q_f32((const float *)pointer);
}

static inline void wasm_v128_store(void *pointer, v128_t value) {
  vst1q_f32((float *)pointer, value);
}

static inline v128_t wasm_f32x4_splat(float value) {
  return vdupq_n_f32(value);
}

static inline v128_t wasm_f32x4_add(v128_t left, v128_t right) {
  return vaddq_f32(left, right);
}

static inline v128_t wasm_f32x4_sub(v128_t left, v128_t right) {
  return vsubq_f32(left, right);
}

static inline v128_t wasm_f32x4_mul(v128_t left, v128_t right) {
  return vmulq_f32(left, right);
}

static inline v128_t wasm_f32x4_div(v128_t left, v128_t right) {
  return vdivq_f32(left, right);
}

static inline v128_t wasm_f32x4_neg(v128_t value) {
  return vnegq_f32(value);
}

static inline v128_t wasm_f32x4_min(v128_t left, v128_t right) {
  return vminq_f32(left, right);
}

static inline v128_t wasm_f32x4_max(v128_t left, v128_t right) {
  return vmaxq_f32(left, right);
}

static inline v128_t wasm_f32x4_nearest(v128_t value) {
  return vrndnq_f32(value);
}

static inline v128_t wasm_f32x4_lt(v128_t left, v128_t right) {
  return vreinterpretq_f32_u32(vcltq_f32(left, right));
}

static inline v128_t wasm_v128_bitselect(v128_t true_value,
                                         v128_t false_value, v128_t mask) {
  return vbslq_f32(vreinterpretq_u32_f32(mask), true_value, false_value);
}

static inline v128_t wasm_i32x4_splat(int32_t value) {
  return vreinterpretq_f32_s32(vdupq_n_s32(value));
}

static inline v128_t wasm_i32x4_add(v128_t left, v128_t right) {
  return vreinterpretq_f32_s32(vaddq_s32(vreinterpretq_s32_f32(left),
                                         vreinterpretq_s32_f32(right)));
}

static inline v128_t wasm_i32x4_trunc_sat_f32x4(v128_t value) {
  return vreinterpretq_f32_s32(vcvtq_s32_f32(value));
}

/* The lane index and shift amount must reach the NEON builtin as constants,
 * and a macro is the only way to guarantee that through a call. */
#define wasm_f32x4_extract_lane(value, lane) vgetq_lane_f32((value), (lane))
#define wasm_i32x4_shl(value, shift)                                       \
  vreinterpretq_f32_s32(                                                   \
      vshlq_n_s32(vreinterpretq_s32_f32(value), (shift)))

#else
#error "the native wasm_simd128 shim supports aarch64 only"
#endif

#endif
