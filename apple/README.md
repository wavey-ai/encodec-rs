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
