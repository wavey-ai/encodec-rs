# encodec-rs

Rust bindings for the current `wavey-ai/encodec` binary boundary.

This crate is deliberately small and honest: it does not reimplement EnCodec in Rust yet. It wraps the existing `encodec` CLI for the full bitstream path, and it can also run the exported frame-level ONNX bundle directly from Rust.

## What it is

- Rust library for launching `encodec` against files
- Rust CLI wrapper with `encode`, `decode`, and `roundtrip`
- Optional ONNX Runtime loader for the exported frame encoder / decoder bundle
- Environment-based launcher selection for either:
  - `encodec` already on `PATH`
  - `python -m encodec`

## What it is not

- not a native EnCodec implementation
- not a PyO3 bridge
- not a rewrite of the model/runtime
- not a replacement for the `.ecdc` bitstream logic yet

## Environment

- `ENCODEC_BIN=/path/to/encodec`
- `ENCODEC_PYTHON=/path/to/python`

If neither is set, the crate defaults to `encodec` on `PATH`.

## ONNX frame runtime

Enable the `onnx` feature to load the ONNX frame bundle exported by `wavey-ai/encodec`:

```toml
encodec-rs = { git = "https://github.com/wavey-ai/encodec-rs.git", features = ["onnx"] }
```

Expected bundle layout:

- `encode_frame.onnx`
- `decode_frame.onnx`
- `bundle.json`

This only covers the neural frame codec boundary. Rust is still expected to own segmentation, overlap-add, and bitstream logic around it.

```rust
use encodec_rs::onnx::{ExecutionTarget, OnnxFrameCodec};

let mut codec = OnnxFrameCodec::from_dir(
    "model/encodec_48khz_12kbps_onnx",
    ExecutionTarget::Cuda { device_id: 0 },
)?;
println!("{:?}", codec.metadata());
```

## Library example

```rust
use encodec_rs::{Encodec, EncodecOptions};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let encodec = Encodec::from_env();
    let opts = EncodecOptions {
        bandwidth: Some(6.0),
        high_quality: true,
        language_model: true,
        force: true,
        ..Default::default()
    };
    encodec.encode_file("input.wav", "output.ecdc", &opts)?;
    encodec.decode_file("output.ecdc", "decoded.wav", &EncodecOptions::default())?;
    Ok(())
}
```

## CLI example

```bash
encodec-rs encode input.wav output.ecdc --hq --lm --bandwidth 6 --force
encodec-rs decode input.ecdc output.wav --force
```
