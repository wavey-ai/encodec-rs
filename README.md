# encodec-rs

`encodec-rs` is a pure Rust EnCodec runtime with native `.ecdc` encode and
decode.

It does not shell out to Python. It does not call an external `encodec`
binary. The runtime path is Rust plus ONNX Runtime only.

## What It Does

- loads EnCodec-compatible ONNX bundles
- encodes `48 kHz` stereo WAV to real `.ecdc`
- decodes `.ecdc` back to WAV
- runs LM-assisted entropy coding in Rust
- runs on CPU, CUDA, CoreML, or TensorRT

## Current Scope

The current checked-in bundles target the `48 kHz` stereo model family:

- `onnx-bundles/encodec_48khz_6kbps`
- `onnx-bundles/encodec_48khz_12kbps`

Both checked-in bundles include:

- `encode_frame.onnx`
- `decode_frame.onnx`
- `lm_logits.onnx`
- `bundle.json`

So LM-assisted `.ecdc` compression works out of the box.

## Runtime Guarantees

- Pure Rust `.ecdc` container logic
- Pure Rust arithmetic coding
- Pure Rust LM-driven entropy path
- No Python bridge
- No external codec subprocess

The only non-Rust runtime dependency is ONNX Runtime for model execution.

## Build

```bash
cargo build --release --features onnx
```

Run tests:

```bash
cargo test --features onnx
```

## CLI

Inspect a bundle:

```bash
encodec-rs onnx-inspect onnx-bundles/encodec_48khz_6kbps
```

Smoke-test model execution:

```bash
encodec-rs onnx-smoke onnx-bundles/encodec_48khz_6kbps
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

Direct frame roundtrip without `.ecdc`:

```bash
encodec-rs onnx-roundtrip-wav \
  onnx-bundles/encodec_48khz_6kbps \
  input.wav \
  output.wav
```

## Execution Targets

CPU is the default.

Use CUDA:

```bash
encodec-rs onnx-encode \
  onnx-bundles/encodec_48khz_6kbps \
  input.wav \
  output.ecdc \
  --cuda
```

Select a GPU explicitly:

```bash
encodec-rs onnx-encode \
  onnx-bundles/encodec_48khz_6kbps \
  input.wav \
  output.ecdc \
  --cuda \
  --device-id 0
```

Use TensorRT:

```bash
encodec-rs onnx-encode \
  onnx-bundles/encodec_48khz_6kbps \
  input.wav \
  output.ecdc \
  --tensorrt \
  --fp16
```

Use CoreML on Apple Silicon:

```bash
encodec-rs onnx-encode \
  onnx-bundles/encodec_48khz_6kbps \
  input.wav \
  output.ecdc \
  --coreml \
  --coreml-compute-units cpu-and-gpu
```

CoreML caches compiled model artifacts under `bundle_dir/.coreml-cache/` by
default. Override that with `--coreml-cache-dir` if needed.

Disable LM compression:

```bash
encodec-rs onnx-encode \
  onnx-bundles/encodec_48khz_6kbps \
  input.wav \
  output.ecdc \
  --no-lm
```

Enable per-chunk CRC wrapping for LM chunk payloads:

```bash
encodec-rs onnx-encode \
  onnx-bundles/encodec_48khz_6kbps \
  input.wav \
  output.ecdc \
  --chunk-crc
```

CRC is off by default for new Rust-written `.ecdc` files.

Adjust frame batching:

```bash
encodec-rs onnx-encode \
  onnx-bundles/encodec_48khz_6kbps \
  input.wav \
  output.ecdc \
  --batch-size 16
```

## Input Rules

- `onnx-encode` currently expects WAV input
- input sample rate must match the bundle sample rate
- the checked-in bundles are for `48 kHz` stereo audio
- CLI resampling is not implemented yet

If your source is not already `48 kHz` stereo WAV, normalize it first.

## Output Metadata

`encodec-rs` writes only the minimal metadata needed to decode the payload:

- model name
- audio length
- codebook count
- LM / arithmetic settings
- bitstream version
- optional chunk CRC flag

## Library Use

Add the crate:

```toml
encodec-rs = { git = "https://github.com/wavey-ai/encodec-rs.git", features = ["onnx"] }
```

Load the frame codec:

```rust
use encodec_rs::onnx::{ExecutionTarget, OnnxFrameCodec};

let mut codec = OnnxFrameCodec::from_dir(
    "onnx-bundles/encodec_48khz_6kbps",
    ExecutionTarget::Cpu,
)?;

println!("{:#?}", codec.metadata());
```

## Benchmark Snapshot

On the `Lori Asha - Westside` premix test track, using LM-assisted `.ecdc`
encoding on both runtimes, the latest local comparison was:

| Codec | Bitrate | Encode | Decode | `.ecdc` size |
|---|---:|---:|---:|---:|
| upstream | 6 kbps | 39.97s | 42.77s | 112,942 bytes |
| upstream | 12 kbps | 44.73s | 49.30s | 239,325 bytes |
| `encodec-rs` | 6 kbps | 27.74s | 26.41s | 116,454 bytes |
| `encodec-rs` | 12 kbps | 31.46s | 30.13s | 243,944 bytes |

So the current Rust runtime is materially faster than upstream on both encode
and decode, while payload size is still slightly larger than upstream.

### Apple M4 CoreML Check

On April 26, 2026, the same `Lori Asha - Westside` track was also tested on an
Apple M4 host using the new CoreML execution target and LM-assisted `6 kbps`
`.ecdc` encode/decode:

| Runtime | Bitrate | Encode | Decode | `.ecdc` size |
|---|---:|---:|---:|---:|
| `encodec-rs` CoreML (`--coreml --coreml-compute-units cpu-and-gpu`) | 6 kbps | 163.84s | 157.26s | 115,572 bytes |

That is roughly `5.9x` slower than the current `encodec-rs` benchmark snapshot
above (`27.74s` encode / `26.41s` decode at `6 kbps`), so CoreML support is
functional on Apple Silicon but not yet competitive with the current Linux /
NVIDIA path.

## Status

What is done:

- pure Rust runtime path
- pure Rust `.ecdc`
- checked-in LM-capable `6 kbps` and `12 kbps` bundles
- CPU / CUDA / CoreML / TensorRT execution targets

What is still missing:

- CLI resampling
- broader model coverage beyond the current `48 kHz` stereo family
- further compression-ratio tuning versus upstream
