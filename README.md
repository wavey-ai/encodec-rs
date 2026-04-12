# encodec-rs

ONNX-first Rust runtime for the frame encoder and decoder exported from `wavey-ai/encodec`.

The fast path is the `onnx` feature. It loads `encode_frame.onnx` and `decode_frame.onnx` directly from Rust and runs them on CPU or CUDA.

The crate also keeps a small compatibility wrapper around the existing `encodec` CLI for the full `.ecdc` path, because ONNX does not cover the bitstream, LM, or container logic yet.

## ONNX

```toml
encodec-rs = { git = "https://github.com/wavey-ai/encodec-rs.git", features = ["onnx"] }
```

```rust
use encodec_rs::onnx::{ExecutionTarget, OnnxFrameCodec};

let mut codec = OnnxFrameCodec::from_dir(
    "model/encodec_48khz_6kbps_onnx",
    ExecutionTarget::Cuda { device_id: 0 },
)?;
println!("{:?}", codec.metadata());
```

Bundle layout:
- `encode_frame.onnx`
- `decode_frame.onnx`
- `bundle.json`

## CLI wrapper

For the legacy full-file path, `encodec-rs` can still launch the existing `encodec` binary:

```bash
encodec-rs encode input.wav output.ecdc --hq --lm --bandwidth 6 --force
encodec-rs decode input.ecdc output.wav --force
```

Environment:
- `ENCODEC_BIN=/path/to/encodec`
- `ENCODEC_PYTHON=/path/to/python`
