# encodec-rs

Rust bindings for the current `wavey-ai/encodec` binary boundary.

This crate is deliberately small and honest: it does not reimplement EnCodec in Rust yet. It wraps the existing `encodec` CLI so Rust services can standardize how they invoke the codec today, while keeping the API surface stable enough to replace the process boundary with native bindings later.

## What it is

- Rust library for launching `encodec` against files
- Rust CLI wrapper with `encode`, `decode`, and `roundtrip`
- Environment-based launcher selection for either:
  - `encodec` already on `PATH`
  - `python -m encodec`

## What it is not

- not a native EnCodec implementation
- not a PyO3 bridge
- not a rewrite of the model/runtime

## Environment

- `ENCODEC_BIN=/path/to/encodec`
- `ENCODEC_PYTHON=/path/to/python`

If neither is set, the crate defaults to `encodec` on `PATH`.

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

