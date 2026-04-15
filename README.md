# encodec-rs

`encodec-rs` is a Rust runtime for EnCodec-compatible ONNX bundles.

It performs:
- frame encode in Rust through ONNX Runtime
- frame decode in Rust through ONNX Runtime
- LM-assisted `.ecdc` packing in Rust
- `.ecdc` decode back to PCM in Rust

There is no Python bridge and no external `encodec` process in the runtime path.

## Scope

The current implementation targets the `48 kHz` stereo EnCodec model family exported as ONNX bundles.

The runtime supports:
- CPU
- CUDA
- TensorRT

## Bundle Layout

Each bundle directory must contain:
- `bundle.json`
- `encode_frame.onnx`
- `decode_frame.onnx`
- the LM ONNX file referenced by `bundle.json`

Example bundle directories in this repo:
- `onnx-bundles/encodec_48khz_6kbps`
- `onnx-bundles/encodec_48khz_12kbps`

Both checked-in example bundles include `lm_logits.onnx`, so LM-assisted `.ecdc`
compression works without any extra export step.

## CLI

Inspect a bundle:

```bash
encodec-rs onnx-inspect onnx-bundles/encodec_48khz_6kbps
```

Encode WAV to `.ecdc`:

```bash
encodec-rs onnx-encode \
  onnx-bundles/encodec_48khz_6kbps \
  input.wav \
  output.ecdc
```

Decode `.ecdc` to WAV:

```bash
encodec-rs onnx-decode \
  onnx-bundles/encodec_48khz_6kbps \
  input.ecdc \
  output.wav
```

Roundtrip a WAV directly through the frame model:

```bash
encodec-rs onnx-roundtrip-wav \
  onnx-bundles/encodec_48khz_6kbps \
  input.wav \
  output.wav
```

Run on CUDA:

```bash
encodec-rs onnx-encode \
  onnx-bundles/encodec_48khz_6kbps \
  input.wav \
  output.ecdc \
  --cuda
```

Run on TensorRT:

```bash
encodec-rs onnx-encode \
  onnx-bundles/encodec_48khz_6kbps \
  input.wav \
  output.ecdc \
  --tensorrt \
  --fp16
```

Disable LM compression:

```bash
encodec-rs onnx-encode \
  onnx-bundles/encodec_48khz_6kbps \
  input.wav \
  output.ecdc \
  --no-lm
```

## Library

Add the crate with the ONNX feature:

```toml
encodec-rs = { git = "https://github.com/wavey-ai/encodec-rs.git", features = ["onnx"] }
```

Load a frame codec:

```rust
use encodec_rs::onnx::{ExecutionTarget, OnnxFrameCodec};

let mut codec = OnnxFrameCodec::from_dir(
    "onnx-bundles/encodec_48khz_6kbps",
    ExecutionTarget::Cpu,
)?;
println!("{:?}", codec.metadata());
```

## Notes

- The CLI currently expects WAV input for encode.
- Input resampling is not done inside the CLI. Use `48 kHz` stereo WAV for the `48 kHz` model.
- `.ecdc` metadata preserves original source sample rate, channel count, and frame count when provided through the CLI path.
