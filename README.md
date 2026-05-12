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

The browser path supports raw `acv=0` and LM-assisted `acv=1` `.ecdc`
encode/decode:

- encode a full audio file in the browser with `encode_frame.onnx`
- encode incrementally by emitting the `.ecdc` header first and one chunk per
  ONNX segment
- package raw frame payloads or LM arithmetic-coded chunks with Rust wasm
- run deterministic LM entropy encode/decode in Rust wasm from `lm_weights.bin`
- decode raw or LM `.ecdc` payloads with `decode_frame.onnx`
- overlap-add decoded frames in Rust wasm
- play reconstructed audio through Web Audio
- run ONNX models through either WASM CPU or WebGPU in the browser

The checked-in browser smoke page uses the short JFK sample from a sibling
`mel-spec` checkout:

```text
../mel-spec/testdata/jfk_f32le.wav
```

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

Click `Encode file` to encode the full JFK clip in the browser. The `Coding`
selector switches between raw `acv=0` and LM `acv=1`. The `Mode` selector
switches between:

- `Incremental`: writes the `.ecdc` header, runs `encode_frame.onnx` one
  segment at a time, and appends a raw frame payload or LM chunk immediately
- `Batch`: runs all segments in one ONNX batch, then packages the complete raw
  frame list or LM chunk list

With `Decode + play` checked, the page decodes the generated raw or LM `.ecdc`
payload and plays it back.

The `Runtime` selector controls ONNX Runtime Web session creation:

- `WASM CPU`: runs the ONNX models through the browser WASM backend
- `WebGPU`: asks ONNX Runtime Web to use the browser WebGPU execution provider,
  with WASM available for unsupported nodes. On macOS this browser GPU path is
  backed by Metal in the browser implementation; JavaScript does not get a
  direct Metal execution provider.

Safari requires Safari 26 or newer for WebGPU, or Safari Technology Preview
with the WebGPU feature enabled. Apple Silicon hardware is not enough by itself;
the browser must expose `navigator.gpu` to the page. In Safari, enable
`Show features for web developers`, then open `Develop > Feature Flags`, search
for `WebGPU`, and enable it. If present, also enable `GPU Process: DOM Rendering`
and `GPU Process: Canvas Rendering`, then quit and reopen Safari.

The page reports total encode and decode time after each run. Those totals
include ONNX session creation when the selected bundle/runtime has not already
been cached in the page.

`acv=0` is the raw EnCodec-code payload. `acv=1` is the portable LM bitstream.
New LM encodes use the same deterministic Rust LM implementation in native and
wasm, loaded from `lm_weights.bin`, before the integer arithmetic coder. ONNX
Runtime is still used for `encode_frame.onnx` and `decode_frame.onnx`, but it is
no longer on the entropy-decoding critical path. Native and browser decode also
accept transitional `acv=4` and `acv=5` LM files written before the public
version reset.

Local cross-runtime matrix on this M1 Air, using the checked-in
`testdata/westside_4s_48khz_stereo.wav` fixture and the `48 kHz 12 kbps`
bundle:

| Encoder | Coding | ECDC version | ECDC bytes | Rust decode | WASM decode |
|---|---|---:|---:|---|---|
| Rust ONNX | raw | `acv=0` | 6,165 | pass, 192,000 samples | pass, 192,000 samples |
| Rust ONNX | LM | `acv=1` | 4,503 | pass, 192,000 samples | pass, 192,000 samples |
| Browser WASM | raw | `acv=0` | 6,165 | pass, 192,000 samples | pass, 192,000 samples |
| Browser WASM | LM | `acv=1` | 4,507 | pass, 192,000 samples | pass, 192,000 samples |

On this fixture, portable LM saves about `27%` versus raw mode while remaining
decodable across native Rust and browser/WASM runtimes.

Broader local matrix on the same M1 Air, using the `48 kHz 12 kbps` bundle and
four-second fixtures:

| Fixture | WASM LM encode bytes | Native decode | Native LM encode bytes | WASM decode |
|---|---:|---|---:|---|
| Westside music | 4,507 | pass | 4,503 | pass |
| JFK speech | 4,089 | pass | 4,091 | pass |
| 440 Hz sine | 2,821 | pass | 2,826 | pass |
| silence | 219 | pass | 219 | pass |
| noise | 1,373 | pass | 1,377 | pass |

Full-song cross-architecture matrix, run on May 12, 2026, using the
`Lori Asha - Westside` premix track normalized to `48 kHz` stereo WAV, the
`48 kHz 12 kbps` bundle, and the portable LM `.ecdc` bitstream now written as
`acv=1`:

| Encoder | `.ecdc` bytes | Mac WASM decode | Mac Rust decode | Ada Linux WASM decode | Ada Linux Rust decode |
|---|---:|---|---|---|---|
| Mac WASM | 254,263 | pass, 208.509s | pass, 208.509s | pass, 208.509s | pass, 208.509s |
| Mac Rust | 254,228 | pass, 208.509s | pass, 208.509s | pass, 208.509s | pass, 208.509s |
| Ada Linux WASM | 254,263 | pass, 208.509s | pass, 208.509s | pass, 208.509s | pass, 208.509s |
| Ada Linux Rust | 254,223 | pass, 208.509s | pass, 208.509s | pass, 208.509s | pass, 208.509s |

This validates the portable LM bitstream across browser/WASM and native Rust on
Apple Silicon macOS and x86_64 Linux. WASM decode output was bit-identical
between Mac and Ada. Native ONNX Runtime decode can differ by small PCM-level
amounts after the neural `decode_frame.onnx` stage, but all 16 full-song
cross-runtime decodes completed and produced full-length audio.

The Cloudflare deployment is staged under `/code/encodec-rs/browser-smoke/`.
Large ONNX files are split into static parts during deployment so they stay
under Cloudflare Pages' per-file asset limit; the browser reassembles those
parts before creating ONNX Runtime Web sessions.

Deploy the Cloudflare demo:

```bash
wasm-pack build --target web --no-default-features --features wasm
npm install --prefix browser-smoke
node scripts/build-cloudflare-browser-smoke.mjs

CLOUDFLARE_EMAIL=jamie@wavey.ai \
CLOUDFLARE_API_KEY="$(tr -d '\n\r' < /Users/jamie/wavey.ai/.cloudflare-token)" \
npx wrangler pages deploy build/cloudflare-pages \
  --project-name encodec-rs-browser-smoke \
  --branch main \
  --commit-dirty=true

CLOUDFLARE_EMAIL=jamie@wavey.ai \
CLOUDFLARE_API_KEY="$(tr -d '\n\r' < /Users/jamie/wavey.ai/.cloudflare-token)" \
npx wrangler deploy
```

The exported wasm helpers used by the page are:

- `rawEcdcHeader(bundleJson, audioLength)`
- `rawEcdcFramePayload(bundleJson, codes, scale, frameLength)`
- `rawEcdcEncode(bundleJson, audioLength, frames)`
- `rawEcdcDecodeFrames(bundleJson, payload)`
- `rawEcdcOverlapAdd(bundleJson, audioLength, decodedFrames)`
- `lmEcdcHeader(bundleJson, audioLength)`
- `lmEcdcChunk(payload)`
- `lmEcdcDecodeChunks(bundleJson, payload)`
- `LmChunkEncoder`
- `LmChunkDecoder`
- `DeterministicLmChunkEncoder`
- `DeterministicLmChunkDecoder`

## Native Scope

The current checked-in bundles target the `48 kHz` stereo model family:

- `onnx-bundles/encodec_48khz_6kbps`
- `onnx-bundles/encodec_48khz_12kbps`

Both checked-in bundles include:

- `encode_frame.onnx`
- `decode_frame.onnx`
- `lm_logits.onnx`
- `lm_weights.bin`
- `bundle.json`

So LM-assisted `.ecdc` compression works out of the box.

When `lm_weights.bin` is present, native and browser LM entropy coding use the
deterministic Rust/wasm LM backend by default. `lm_logits.onnx` is retained for
inspection and fallback. Set `ENCODEC_RS_LM_BACKEND=ort` to force the native
runtime back to the ONNX LM path for comparison.

## Runtime Notes

- Pure Rust `.ecdc` container logic
- Pure Rust arithmetic coding
- Pure Rust deterministic LM-driven entropy path
- No Python bridge
- No external codec subprocess

The only non-Rust runtime dependency is ONNX Runtime for the neural frame
encoder/decoder.

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
