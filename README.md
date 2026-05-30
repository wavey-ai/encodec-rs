# encodec-rs

Live browser demo:
[`https://wavey.ai/code/encodec-rs/browser-smoke/`](https://wavey.ai/code/encodec-rs/browser-smoke/)

`encodec-rs` is a Rust EnCodec runtime with native and browser `.ecdc`
encode/decode paths.

Native execution is implemented in Rust on top of ONNX Runtime and has no
Python runtime dependency. It does not require a Python bridge or external codec
subprocess. The browser path runs the EnCodec ONNX frame models with
`onnxruntime-web` and uses Rust wasm for `.ecdc` packaging, parsing,
overlap-add, and deterministic LM arithmetic coding. It also has no Python
runtime dependency.

The native path loads EnCodec-compatible ONNX bundles, encodes `48 kHz` stereo
WAV to `.ecdc`, decodes `.ecdc` back to WAV, and supports CPU, CUDA, CoreML,
and TensorRT execution targets. LM-assisted entropy coding is implemented in
Rust.

## Browser Support

The browser path supports the current q8 LM `.ecdc` bitstream (`acv=2`):

- encode a full audio file in the browser with `encode_frame.onnx`
- package q8 LM arithmetic-coded chunks with Rust wasm
- decode q8 `.ecdc` payloads with `decode_frame.onnx`
- overlap-add decoded frames in Rust wasm
- run ONNX frame models through WebGPU, with WASM available for unsupported
  nodes

Build the wasm package:

```bash
rustup target add wasm32-unknown-unknown
cargo check --lib --no-default-features --features wasm --target wasm32-unknown-unknown
cargo install wasm-pack
wasm-pack build --target web --no-default-features --features wasm
```

Run the local browser encode/decode/playback page:

```bash
npm install --prefix browser-smoke
python3 browser-smoke/serve.py
```

Then open:

```text
http://127.0.0.1:8787/browser-smoke/
```

The scripted WebGPU matrix runner is:

```bash
node scripts/webgpu-matrix.mjs
```

It writes browser WebGPU artifacts under `target/webgpu-matrix/`. See
`MATRIX.md` for the current full-track matrix output folders.

Safari requires Safari 26 or newer for WebGPU, or Safari Technology Preview
with the WebGPU feature enabled. Apple Silicon hardware is not enough by itself;
the browser must expose `navigator.gpu` to the page. In Safari, enable
`Show features for web developers`, then open `Develop > Feature Flags`, search
for `WebGPU`, and enable it. If present, also enable `GPU Process: DOM Rendering`
and `GPU Process: Canvas Rendering`, then quit and reopen Safari.

The exported wasm helpers used by the q8 matrix path are:

- `ecdcMetadata(payload)`
- `ecdcOverlapAdd(bundleJson, audioLength, decodedFrames)`
- `lmEcdcHeaderForWeights(bundleJson, audioLength, 2, weights)`
- `lmEcdcFixedHeaderForWeights(bundleJson, audioLength, 2, weights)`
- `lmEcdcChunk(payload)`
- `lmEcdcDecodeChunks(bundleJson, payload)`
- `QuantizedLmChunkEncoder`
- `QuantizedLmChunkDecoder`
- `stableHashHex(bytes)`

Use `lmEcdcHeaderForWeights` for dynamic bundles. Use
`lmEcdcFixedHeaderForWeights` when writing ECDC against a fixed-length ONNX
graph; it records the fixed chunk samples, stride, and LM frame length (`fl`) so
decoders pull the full graph width for every chunk, including the final chunk.
For fixed graph chunks, finish LM packet encoding with
`QuantizedLmChunkEncoder.finishPadded(frameLength)` so encodec-rs writes zero-code
padding for any short final segment before the ECDC packet is wrapped.

## Native Scope

Model bundles are hosted on Hugging Face:

- [`wavey-ai/encodec-rs-onnx-bundles`](https://huggingface.co/wavey-ai/encodec-rs-onnx-bundles)

Download them into the checkout before running ONNX/browser model paths:

```bash
scripts/download-onnx-bundles.sh
```

The hosted bundles target the `48 kHz` stereo model family:

- `onnx-bundles/encodec_48khz_6kbps`
- `onnx-bundles/encodec_48khz_12kbps`

Both bundles include:

- `encode_frame.onnx`
- `decode_frame.onnx`
- `lm_weights_q8.bin`
- `bundle.json`

So LM-assisted `.ecdc` compression works after the bundle download step.

Native and browser LM entropy coding use the q8 Rust/wasm LM backend. Older raw
and f32/ONNX-LM bitstreams are intentionally not supported.

### Bundle Sizes

The dynamic bundles are the default native bundles. Their frame models accept a
variable final frame, so ECDC can derive each chunk's LM frame length from the
actual sample count:

| Bundle | Bandwidth | Nominal chunk | Samples | Stride | LM frames | Codebooks |
|---|---:|---:|---:|---:|---:|---:|
| `encodec_48khz_6kbps` | 6 kbps | 1000ms | 48,000 | 47,520 | 150 | 4 |
| `encodec_48khz_12kbps` | 12 kbps | 1000ms | 48,000 | 47,520 | 150 | 8 |

Fixed bundles trace the ONNX graph at one chunk size. ECDC written for these
bundles should include `cs`, `cst`, and `fl`, and should entropy-code the full
`fl` steps. The PCM input segment is already zero-padded before EnCodec encode;
the ECDC writer must not shorten the LM stream for the final partial chunk.

| Fixed chunk | Samples | Stride | LM frames | Bundle suffix |
|---|---:|---:|---:|---|
| 1000ms | 48,000 | 47,520 | 150 | `_1000ms` |
| 1333ms | 64,000 | 63,520 | 200 | `_1333ms` |
| 1800ms | 86,400 | 85,920 | 270 | `_1800ms` |

The default wasm fixed-bundle package currently ships the `1333ms` and
`1800ms` variants for both `6 kbps` and `12 kbps`. The export tooling also knows
the `1000ms` shape; include it in `BUNDLES` when a fixed 1s graph is needed.

## Runtime Notes

- Pure Rust `.ecdc` container logic
- Pure Rust arithmetic coding
- Pure Rust deterministic LM-driven entropy path
- No Python bridge
- No external codec subprocess

The only non-Rust runtime dependency is ONNX Runtime for the neural frame
encoder/decoder.

## Apple Native Backend Boundary

The `.ecdc` layer is now model-runtime agnostic. Build it without ONNX Runtime:

```bash
cargo check --features ecdc
```

Native callers can keep the Rust bitstream path and provide only the neural
frame runtime:

- `ecdc::FrameCodec`: metadata plus `encode_frame` / `decode_frame`
- `ecdc::LmCodec`: LM logits for portable arithmetic-coded chunks
- `portable_lm::PortableLmCodec`: loads `bundle.json` + `lm_weights_q8.bin`
  without ONNX Runtime

The ONNX runtime implements those traits through `OnnxFrameCodec`
and `OnnxLmCodec`, so existing CLI/browser parity remains the validation
harness. For iOS/macOS product code, the intended final shape is a Swift/MLX
frame backend, with Core ML or ONNX Runtime used only as transitional parity
checks.

### Apple MLX Runtime

Apple MLX support now lives in this repository under `apple/`. The Swift package
loads MLX Swift `.safetensors` archives for frame `encode_frame` /
`decode_frame`, while the Rust crate owns `.ecdc`, portable q8 LM coding, and
the C ABI bridge in `src/mlx_bridge.rs`. See `apple/README.md` for Swift package
build, test, and Westside benchmark commands.

After downloading the bundles, convert them with:

```bash
target/quant-venv/bin/python scripts/export-mlx-frame-archive.py \
  onnx-bundles/encodec_48khz_6kbps \
  target/mlx-bundles/encodec_48khz_6kbps

target/quant-venv/bin/python scripts/export-mlx-frame-archive.py \
  onnx-bundles/encodec_48khz_12kbps \
  target/mlx-bundles/encodec_48khz_12kbps
```

Each MLX bundle contains `bundle.json`, `lm_weights_q8.bin`,
`encode_frame.safetensors`, `decode_frame.safetensors`, and
`mlx-manifest.json`. The Python step is offline conversion tooling only; the
native app path is Swift/MLX plus the Rust `.ecdc`/portable-LM boundary.

## Native Build

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

LM chunk payloads are CRC-wrapped by default. The CRC is stored next to each
length-prefixed chunk and lets decoders identify corrupted recovered chunks
before arithmetic decoding.

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
- the hosted bundles are for `48 kHz` stereo audio
- CLI resampling is not implemented yet

If your source is not already `48 kHz` stereo WAV, normalize it first.

## Output Metadata

`encodec-rs` writes only the minimal metadata needed to decode the payload:

- model name
- audio length
- codebook count
- LM / arithmetic settings
- q8 bitstream version (`acv=2`)
- q8 LM weight hash
- fixed chunk sample count (`cs`), stride (`cst`), and LM frame length (`fl`)
  when the payload targets a fixed-length graph

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

### Apple M1 ONNX CPU Check

On May 19, 2026, after splitting `.ecdc` from the concrete ONNX runtime, the
same `Lori Asha - Westside` 48 kHz stereo fixture was measured on an Apple M1
host using ONNX Runtime `1.25.1` CPU, release build, batch size `8`, and
LM-assisted `.ecdc` with chunk CRC enabled:

| Runtime | Bitrate | Encode | Decode | `.ecdc` size | vs native snapshot |
|---|---:|---:|---:|---:|---:|
| `encodec-rs` ONNX CPU on Apple M1 | 6 kbps | 101.44s | 105.67s | 121,816 bytes | 3.66x / 4.00x slower |
| `encodec-rs` ONNX CPU on Apple M1 | 12 kbps | 126.48s | 143.18s | 255,061 bytes | 4.02x / 4.75x slower |

This confirms the trait/backend split did not change the neural runtime: Apple
native performance still needs a real MLX/Metal frame backend rather than the
current ONNX CPU path.

### MLX Archive Comparison

On the same frame models, the MLX archive export keeps only the
initializers needed by the Swift/MLX runtime and the manifest needed to rebuild
the graph:

| Bundle | Model | Initializers | Parameters | ONNX file | MLX safetensors |
|---|---|---:|---:|---:|---:|
| `6 kbps` | encode frame | 81 | 8,345,360 | 32M | 32M |
| `6 kbps` | decode frame | 78 | 7,951,766 | 31M | 30M |
| `12 kbps` | encode frame | 89 | 9,393,936 | 36M | 36M |
| `12 kbps` | decode frame | 82 | 8,476,054 | 33M | 32M |

The exported graphs still contain the same neural work as the ONNX benchmark:
convolutions, transposed convolutions, instance normalization, LSTMs, and RVQ
math. The Apple MLX runtime now loads these archives, evaluates native
`encode_frame` and `decode_frame`, and bridges q8 LM-assisted `.ecdc`
encode/decode through Rust with Swift/MLX frame callbacks.

On the same Apple M1 host as the ONNX CPU check above, the full `Lori Asha -
Westside` fixture (`208.509s`, 48 kHz stereo) was measured through the Release
Apple test bundle with q8 LM entropy coding:

| Runtime | Mode | Bitrate | Encode | Decode | `.ecdc` size |
|---|---|---:|---:|---:|---:|
| Swift/MLX + Rust bridge | q8 LM | 6 kbps | 36.55s | 42.02s | 107,327 bytes |
| Swift/MLX + Rust bridge | q8 LM | 12 kbps | 43.89s | 46.76s | 232,944 bytes |

The q8 LM path is the only supported `.ecdc` payload path in this checkout.

## Status

What is done:

- pure Rust runtime path
- pure Rust `.ecdc`
- hosted LM-capable `6 kbps` and `12 kbps` bundles
- CPU / CUDA / CoreML / TensorRT execution targets

What is still missing:

- CLI resampling
- broader model coverage beyond the current `48 kHz` stereo family
- further compression-ratio tuning versus upstream
