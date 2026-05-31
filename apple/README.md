# Encodec MLX Runtime

This Swift package owns the Apple MLX frame runtime for `encodec-rs`.

`encodec-rs` owns:

- exported MLX frame archives under `target/mlx-bundles/`
- the Rust `.ecdc` container and portable q8 LM
- the C ABI bridge in `src/mlx_bridge.rs`
- this Swift/MLX package and tests

The Swift runtime owns only `encode_frame` and `decode_frame` execution through
MLX on Metal. Rust calls Swift frame callbacks while keeping `.ecdc` and LM
coding deterministic and shared with the native/browser paths.

## Build prerequisites

```sh
cd ..
cargo build --features ecdc
```

The Swift package links `libencodec_rs` from `../target/debug` or
`../target/release`. At runtime, make sure dyld can find the library:

```sh
export DYLD_LIBRARY_PATH="$(pwd)/../target/debug:$(pwd)/../target/release:${DYLD_LIBRARY_PATH:-}"
```

## Generate MLX bundles

```sh
cd ..
target/quant-venv/bin/python scripts/export-mlx-frame-archive.py \
  onnx-bundles/encodec_48khz_6kbps \
  target/mlx-bundles/encodec_48khz_6kbps

target/quant-venv/bin/python scripts/export-mlx-frame-archive.py \
  onnx-bundles/encodec_48khz_12kbps \
  target/mlx-bundles/encodec_48khz_12kbps

scripts/create_mlx_fixed_bundles.sh
```

The fixed bundle helper creates 1333ms and 1800ms MLX bundles for both 6kbps
and 12kbps models by exporting the fixed ONNX bundles. The standard fixed MLX
bundles therefore use the ONNX 300-step q8 LM weights, with decoder frame sizes
of 64,000 samples / 200 code frames for 1333ms and 86,400 samples / 270 code
frames for 1800ms. If present, `*_mobygratisv0` compatibility bundles are also
exported with the old 150-step LM weights.

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
BITNEEDLE_MLX_BENCH_BUNDLES=encodec_48khz_6kbps,encodec_48khz_12kbps \
BITNEEDLE_MLX_BENCH_LM=1 \
BITNEEDLE_MLX_BENCH_BATCH_SIZE=1 \
BITNEEDLE_MLX_BENCH_OUT="../target/mlx-bench-current" \
swift test --filter EncodecMLXRuntimeTests/testBenchmarkNativeMLXEcdcRoundtrip
```
