# Encodec MLX Runtime

This Swift package owns the Apple MLX frame runtime for `encodec-rs`.

`encodec-rs` owns:

- exported MLX frame archives under `target/mlx-bundles/`
- the Rust `.ecdc` container and portable q8 LM
- the C ABI bridge in `src/mlx_bridge.rs`
- this Swift/MLX package and tests

The Swift runtime executes only `encode_frame` and `decode_frame` through MLX on Metal.

Rust calls the Swift frame callbacks.

Rust keeps ECDC and language-model coding deterministic across native and browser paths.

The in-memory API encodes and decodes complete ECDC files.

`encodeEcdcOutputs(..., derived6KBundleURL:)` optionally returns canonical
12 kbps and 6 kbps files from one neural encoder pass. The entropy stage keeps
two deterministic nonlinear states but retains only the 12 kbps weights,
shares the first-four-codebook embedding sum, and uses paired AArch64 matrix
kernels. Passing no 6 kbps bundle preserves the ordinary single-output path.

A streaming encoder writes progressive output to disk.

Fixed-context in-memory decoding writes borrowed MLX windows directly to final interleaved PCM.

Fixed-context file decoding writes the same windows directly to planar `f32le` output.

These paths do not allocate an intermediate complete planar track.

iOS uses batch size one by default. Larger batches caused device memory pressure.

macOS uses batch size eight by default.

Callers can override either default.

Call `prewarm(frameBatchSize:)` before the first timed operation.

Use the same batch size for prewarm and normal operation.

## Build

```sh
cd ..
scripts/build-apple-mlx.sh
```

The script builds the release Rust library and Swift package.

It also compiles the generated MLX Metal sources into `mlx.metallib`.

Swift Package Manager does not compile this Metal library.

The `EncodecMLXRuntime` library is the product interface.

The `EncodecMLXEncode` executable is a local benchmark tool.

The benchmark tool accepts RIFF PCM16, packed PCM24, PCM32, and float32 WAV input.

It also accepts PCM and float `WAVE_FORMAT_EXTENSIBLE` input.

It does not resample audio. Current bundles require 48 kHz stereo input.

The library product leaves Rust bridge symbols for the parent app to provide.

The macOS tests and executable link `libencodec_rs` from `../target`.

Set the library path when you run package tests directly:

```sh
export DYLD_LIBRARY_PATH="$(pwd)/../target/debug:$(pwd)/../target/release:${DYLD_LIBRARY_PATH:-}"
```

## Generate MLX bundles

```sh
cd ..
PYTHON=/opt/anaconda3/envs/encodec-export/bin/python \
  scripts/create_mlx_fixed_bundles.sh
```

The helper creates 1333 ms and 1800 ms bundles for both bitrates.

The 1333 ms bundle owns 64,000 samples and uses a 64,960-sample model window.

The 1800 ms bundle owns 86,400 samples and uses an 87,360-sample model window.

Each window includes two 480-sample guards.

The model outputs contain 203 and 273 code frames, respectively.

Both bundles use the portable 300-step q8 language model.

## Run tests

```sh
cd apple
swift test
```

## Run the Westside benchmark

```sh
cd apple
BITNEEDLE_MLX_BENCH=1 \
BITNEEDLE_MLX_BENCH_WAV="../target/lori-asha-wasm-native/wav/02 - Lori Asha - Westside.48k-stereo.wav" \
BITNEEDLE_MLX_BENCH_BUNDLES=encodec_48khz_6kbps_1333ms,encodec_48khz_12kbps_1333ms \
BITNEEDLE_MLX_BENCH_LM=1 \
BITNEEDLE_MLX_BENCH_BATCH_SIZE=1 \
BITNEEDLE_MLX_BENCH_OUT="../target/mlx-bench-current" \
swift test --filter EncodecMLXRuntimeTests/testBenchmarkNativeMLXEcdcRoundtrip
```

## Benchmark optional 6 kbps output

```sh
cd apple
BITNEEDLE_MLX_DUAL_BENCH=1 \
BITNEEDLE_MLX_BENCH_WAV="/path/to/48khz-stereo.wav" \
BITNEEDLE_MLX_DUAL_PROFILE_MS=1333 \
BITNEEDLE_MLX_BENCH_BATCH_SIZE=8 \
swift test -c release \
  --filter EncodecMLXRuntimeTests/testBenchmarkOptionalDerivedSixKbpsEncode
```

The original, unfused implementation produced byte-identical corresponding
outputs for the 227.863-second Lori master in all conditions:

| Profile | Mode | Wall time | RTFx | 12k bytes | 6k bytes |
|---|---|---:|---:|---:|---:|
| 1333 ms | 12k only | 33.512 s | 6.799× | 296,596 | — |
| 1333 ms | 12k + derived 6k | 45.417 s | 5.017× | 296,596 | 145,906 |
| 1333 ms | Separate 12k then 6k | 45.902 s | 4.964× | 296,596 | 145,906 |
| 1800 ms | 12k only | 37.923 s | 6.009× | 294,319 | — |
| 1800 ms | 12k + derived 6k | 58.532 s | 3.893× | 294,319 | 144,521 |
| 1800 ms | Separate 12k then 6k | 53.452 s | 4.263× | 294,319 | 144,521 |

These are single ordered full-track trials, not acceptance medians. The 1333 ms
dual path was 1.1% faster than separate encodes; the 1800 ms result was 9.5%
slower and demonstrates that host load and ordering can outweigh the avoided
neural pass. Use randomized repeated trials for performance decisions.

The retained experiment now uses one weight set and fused two-input matrix
kernels. Exact-output tests cover both profiles. A four-second end-to-end probe
on Apple Silicon measured 3.116 s for 12k only, 4.534 s for the paired 12k+6k
path, and 6.999 s for two separate encodes: 1.54× faster than separate output
generation in that ordered run. This is a directional probe, not a quiet-host
acceptance result.

Randomized LM-only trials on the currently contended development host found:

| Runtime | 1333 ms paired vs separate LM | 1800 ms paired vs separate LM |
|---|---:|---:|
| AArch64 | 1.04–1.18× | approximately 1.00–1.07× |
| WASM SIMD in Node | approximately 0.98–1.06× | approximately 0.98–1.07× |

WASM timings moved by more than the paired/separate difference as host load
changed, so they establish parity rather than a stable kernel speedup. The
larger end-to-end saving comes from avoiding the second neural encode. The
paired runtime also avoids retaining the 6 kbps LM weight file, saving
6,861,604 bytes (6.54 MiB) of steady-state weight memory.

The scalable-container feasibility analysis is documented in
[`../docs/scalable-ecdc-design.md`](../docs/scalable-ecdc-design.md).
