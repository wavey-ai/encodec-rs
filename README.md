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
WAV to `.ecdc`, and decodes `.ecdc` back to WAV. It supports CPU, CUDA, CoreML,
and TensorRT execution targets. Rust implements LM-assisted entropy coding.

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

### Chunked WASM Round-Trip Test

`scripts/westside-chunk-wasm-roundtrip.mjs` exercises the full wasm
encode/decode path on the `Lori Asha - Westside` track in independent
fixed-chunk mode. It uses only the exported wasm helpers (no native runtime):

1. reads the source WAV from
   `target/lori-asha-wasm-native/wav/02 - Lori Asha - Westside.48k-stereo.wav`
2. splits it, soundkit-style, into non-overlapping `1.333s` PCM chunks (one
   chunk per `encodec_48khz_12kbps_1333ms` owned hop, `64,000`
   samples)
3. wasm-encodes each chunk to its own standalone `.ecdc` in `testdata/out/ecdc/`
4. wasm-decodes each `.ecdc` (read back from disk) to PCM in `testdata/out/pcm/`
5. concatenates the per-chunk PCM into one contiguous
   `testdata/out/westside.contiguous.wav`

```bash
# full track (~4.5 min, 158 chunks)
node scripts/westside-chunk-wasm-roundtrip.mjs

# quick smoke test over the first N chunks
WESTSIDE_MAX_CHUNKS=3 node scripts/westside-chunk-wasm-roundtrip.mjs
```

Chatty progress is written to stderr. A JSON summary is written to stdout
(`node scripts/westside-chunk-wasm-roundtrip.mjs 2>/dev/null` to keep only the
summary). Each chunk re-uses `lmEcdcFixedHeaderForWeights`, so the chunk size is
the bundle's `63,520`-frame non-overlapping stride (`~1.323s`). This tiles the
track gaplessly and reconstructs every source frame.

Because `wasm-bindgen --target web` does not emit a `package.json`, Node treats
the generated `pkg/encodec_rs.js` as CommonJS. If the import fails, add the ESM
marker to the gitignored build output:

```bash
echo '{ "type": "module" }' > pkg/package.json
```

Safari requires Safari 26 or newer for WebGPU, or Safari Technology Preview
with the WebGPU feature enabled. Apple Silicon hardware is not enough by itself.
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
graph. It records the fixed chunk samples, stride, and LM frame length (`fl`) so
decoders have the full graph width for each chunk. This also applies to the final
chunk. Encode all model codes for each fixed graph chunk. This rule also applies
to a short final owned region. Finish the LM packet with
`QuantizedLmChunkEncoder.finish()`. Do not replace unowned model codes with code
zero. `finishPadded` now rejects an incomplete code sequence.

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
`fl` steps. The PCM input segment is already zero-padded before EnCodec encode.
the ECDC writer must not shorten the LM stream for the final partial chunk.

Fixed bundles are guarded: each logical chunk is encoded with ±10 ms (480
samples) of real neighbouring source context on each side. The model window is
`owned + 2 × 480`. The guard samples are codec context only and are cropped
after decode, leaving the exact owned-sample timeline. Adjacent decoded chunks
are then joined with a deterministic `cubic-hermite-v1` 0.5 ms (24-sample) seam
repair (see `chunk-continuity.md`).

| Fixed chunk | Owned | Model window | LM frames | Bundle suffix |
|---|---:|---:|---:|---|
| 1333ms | 64,000 | 64,960 | 203 | `_1333ms` |
| 1800ms | 86,400 | 87,360 | 273 | `_1800ms` |

The default wasm fixed-bundle package ships the `1333ms` and
`1800ms` variants for both `6 kbps` and `12 kbps`.

## Runtime Notes

- Pure Rust `.ecdc` container logic
- Pure Rust arithmetic coding
- Pure Rust deterministic LM-driven entropy path
- No Python bridge
- No external codec subprocess

The only non-Rust runtime dependency is ONNX Runtime for the neural frame
encoder/decoder.

### ONNX Session State

The ONNX frame encoder and decoder sessions do not keep codec state between
`run()` calls. Each call depends only on its input tensors. Applications can
reuse one session for independent chunks or interleave chunks from different
tracks.

ONNX Runtime can retain graph optimizations, memory allocations, thread pools,
and internal caches. This operational state can change performance, but it does
not change encoded output.

The q8 LM encoder and decoder start with new entropy state for each independent
chunk. Callers must pass state explicitly when they use a model that exposes
recurrent state tensors. The fixed frame models in these bundles do not use
cross-call state.

A single-thread WASM test compared 316 chunks in isolated sessions and one
randomly interleaved shared session. All 316 interleaved chunk SHA-256 values
matched their isolated values. This result confirms byte-for-byte deterministic
output for that test.

### WASM Entropy Performance

The single-thread test uses ONNX Runtime Web WASM and the fixed 12 kbps,
1333 ms bundle. Track A has 227.863 seconds in 171 chunks. Track B has
192.936 seconds in 145 chunks.

Higher realtime factors are faster. Lower LM times are faster. The table shows
one complete run for each stage.

| Stage | A warm realtime | B warm realtime | A median LM | B median LM |
|---|---:|---:|---:|---:|
| Scalar baseline | 1.583× | 1.569× | 515.547 ms | 514.621 ms |
| Exact WASM SIMD | 2.471× | 2.449× | 207.956 ms | 211.098 ms |
| Exact CDF selection | 2.651× | 2.532× | 186.162 ms | 191.117 ms |
| Reused CDF storage | 2.548× | 2.516× | 189.152 ms | 190.439 ms |
| Exact probability cache | 2.652× | 2.628× | 183.708 ms | 184.515 ms |

Each stage matched all 316 scalar baseline chunk hashes. Each randomized
interleave also matched all 316 isolated chunk hashes. The final stage passes
the 2× realtime gate for both tracks.

Use these commands to build and measure a stage:

```bash
scripts/build-entropy-wasm-stage.sh <stage-name>
node scripts/profile-encodec-wasm-session.mjs \
  --wasm-root target/performance/entropy-optimization/<stage-name>/wasm \
  --output target/performance/entropy-optimization/<stage-name>/profile.json \
  <track-a.wav> <track-b.wav>
```

Use `scripts/compare-encodec-wasm-profiles.mjs` to compare timings and chunk
hashes between two reports.

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
harness. The intended Apple product code uses a Swift/MLX frame backend. Use
Core ML or ONNX Runtime only for transitional parity checks.

### Apple MLX Runtime

Apple MLX support now lives in this repository under `apple/`. The Swift package
loads MLX Swift `.safetensors` archives for `encode_frame` and `decode_frame`.
The Rust crate owns `.ecdc`, portable q8 LM coding, and the C ABI bridge in
`src/mlx_bridge.rs`. See `apple/README.md` for Swift build, test, and Westside
benchmark commands.

After downloading the bundles, convert them with:

```bash
target/quant-venv/bin/python scripts/export-mlx-frame-archive.py \
  onnx-bundles/encodec_48khz_6kbps \
  target/mlx-bundles/encodec_48khz_6kbps

target/quant-venv/bin/python scripts/export-mlx-frame-archive.py \
  onnx-bundles/encodec_48khz_12kbps \
  target/mlx-bundles/encodec_48khz_12kbps

scripts/create_mlx_fixed_bundles.sh
```

Each MLX bundle contains `bundle.json`, `lm_weights_q8.bin`,
`encode_frame.safetensors`, `decode_frame.safetensors`, and
`mlx-manifest.json`. The Python step is offline conversion tooling only. The
native app path is Swift/MLX plus the Rust `.ecdc`/portable-LM boundary.
The fixed-bundle helper exports from the fixed ONNX bundles. Thus, the standard
1333ms and 1800ms MLX bundles use the same 300-step q8 LM weights as ONNX. It
does not create application-specific compatibility bundles.

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

Export qualification-only frame evidence:

```bash
encodec-rs onnx-encode-evidence \
  onnx-bundles/encodec_48khz_6kbps_1333ms \
  input.wav \
  evidence/fixed-1333-6kbps
```

This command writes the exact model input, codes, scale, raw entropy, recovered
codes, and codebook order. The manifest contains file shapes, SHA-256 digests,
and exact code recovery status.

Use `--true-variable-tail` with a dynamic model bundle to preserve the actual
final input length.

Export one canonical fixed-code LM vector:

```bash
encodec-rs onnx-lm-evidence \
  onnx-bundles/encodec_48khz_6kbps_1333ms \
  evidence/lm-6kbps \
  --steps 203
```

These commands do not create a Profile 1 container. See
[`docs/frame-evidence.md`](docs/frame-evidence.md).

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

## ECDC Container Layout

An `.ecdc` file is one self-contained container. The file header is written
once, followed by one or more framed chunk payloads:

```text
4 bytes   magic: "ECDC"
1 byte    version: 0
4 bytes   metadata JSON byte length, big-endian u32
N bytes   metadata JSON

repeated chunks:
4 bytes   chunk payload length, big-endian u32
4 bytes   CRC32 of the chunk payload, big-endian u32
M bytes   chunk payload
```

The normal q8 LM `.ecdc` path always writes CRC-wrapped chunks. Chunk count is
not stored as a separate top-level field. Decoders read chunk frames after the
metadata header. They validate the count against the audio length and the chunk
layout from metadata (`al`, `cs`, `cst`, `fl`).

Do not concatenate multiple `.ecdc` files to make one record payload. A record
spiral carries one complete `.ecdc` byte stream. That stream may contain many
framed chunks internally, but each independently playable record needs its own
container header and metadata.

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

On April 26, 2026, the same `Lori Asha - Westside` track was tested on an Apple
M4 host. The test used the new CoreML target and LM-assisted `6 kbps` `.ecdc`
encoding and decoding:

| Runtime | Bitrate | Encode | Decode | `.ecdc` size |
|---|---:|---:|---:|---:|
| `encodec-rs` CoreML (`--coreml --coreml-compute-units cpu-and-gpu`) | 6 kbps | 163.84s | 157.26s | 115,572 bytes |

This is approximately `5.9x` slower than the current `encodec-rs` benchmark.
That benchmark took `27.74s` to encode and `26.41s` to decode at `6 kbps`.
CoreML support works on Apple Silicon. It is not yet competitive with the
current Linux and NVIDIA path.

### Apple M1 ONNX CPU Check

On May 19, 2026, the test measured the same `Lori Asha - Westside` fixture on an
Apple M1 host. The fixture was 48 kHz stereo. The test used ONNX Runtime `1.25.1`
on the CPU. It used a release build, batch size `8`, and LM-assisted `.ecdc` with
chunk CRC. This test occurred after the `.ecdc` and ONNX runtime split:

| Runtime | Bitrate | Encode | Decode | `.ecdc` size | vs native snapshot |
|---|---:|---:|---:|---:|---:|
| `encodec-rs` ONNX CPU on Apple M1 | 6 kbps | 101.44s | 105.67s | 121,816 bytes | 3.66x / 4.00x slower |
| `encodec-rs` ONNX CPU on Apple M1 | 12 kbps | 126.48s | 143.18s | 255,061 bytes | 4.02x / 4.75x slower |

This confirms that the trait and backend split did not change the neural
runtime. Apple-native performance still needs an MLX/Metal frame backend. The
current ONNX CPU path is not sufficient.

### MLX Archive Comparison

For the same frame models, the MLX archive export keeps only required files. It
keeps the initializers for the Swift/MLX runtime. It also keeps the manifest
that rebuilds the graph:

| Bundle | Model | Initializers | Parameters | ONNX file | MLX safetensors |
|---|---|---:|---:|---:|---:|
| `6 kbps` | encode frame | 81 | 8,345,360 | 32M | 32M |
| `6 kbps` | decode frame | 78 | 7,951,766 | 31M | 30M |
| `12 kbps` | encode frame | 89 | 9,393,936 | 36M | 36M |
| `12 kbps` | decode frame | 82 | 8,476,054 | 33M | 32M |

The exported graphs still contain the same neural work as the ONNX benchmark:
convolutions, transposed convolutions, instance normalization, LSTMs, and RVQ
math. The Apple MLX runtime loads these archives. It evaluates native
`encode_frame` and `decode_frame`. Swift/MLX frame callbacks bridge q8
LM-assisted `.ecdc` encoding and decoding through Rust.

The release Apple test bundle measured the full `Lori Asha - Westside` fixture.
The test used the same Apple M1 host as the ONNX CPU check. The fixture was
`208.509s`, 48 kHz stereo. The test used q8 LM entropy coding:

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

## Local Node Memory Benchmark

On June 30, 2026, the production `encodec_48khz_12kbps_1333ms` bundle was
measured locally from Node in this repo with:

```bash
node --expose-gc tools/benchmark-encodec-memory.mjs
/usr/bin/time -l node --expose-gc tools/benchmark-encodec-memory.mjs
```

This benchmark uses the production model bundle at
`../encodec-worker/wasm/encodec_48khz_12kbps_1333ms/`, records
`process.memoryUsage()` checkpoints plus short-interval peak sampling, and
writes the machine-readable report to `tmp/encodec-memory-benchmark.json`.

The observed result was that this current local implementation does not fit
within a `128 MiB` process budget. The peak sampled RSS was `396.453 MiB`
(`415711232` raw bytes from `/usr/bin/time -l`), or roughly `268.453 MiB` over
that limit. Repeated encodes did not show ongoing memory growth after warm-up:
RSS fell after the first encode. It then stayed effectively flat from 5 to 20
segments (`318.688 MiB` to `319.047 MiB`). After encoder release and GC, the
final RSS was still `319.141 MiB`. A Cloudflare Worker also needs memory for the
Workers runtime and request handling.

Full timed-run output:

```text
EnCodec local memory benchmark
Model:
  encode_frame.onnx: 36.042 MiB (37792462 bytes)
  lm_weights_q8.bin: 10.484 MiB (10993572 bytes)
  total model bytes: 46.526 MiB (48786034 bytes)
Baseline RSS: 42.453 MiB
RSS after model load: 144.172 MiB
RSS after encoder initialisation: 359.953 MiB
RSS after first encode: 337.500 MiB
RSS after 20 encodes: 319.047 MiB
Peak sampled RSS: 396.453 MiB
Final RSS after release and GC: 319.141 MiB
Peak increase over baseline: 354.000 MiB
Estimated model/init retained memory: 317.500 MiB
Observed growth across repeated encodes: -18.453 MiB
Initialisation time: 18214.804 ms
First encode time: 905.591 ms
Steady-state average encode time: 863.046 ms

Checkpoints:
1. Node process started
  rss: 42.453 MiB (44515328 bytes, +0.000 MiB from baseline)
  heapTotal: 6.344 MiB (6651904 bytes, +0.000 MiB from baseline)
  heapUsed: 3.739 MiB (3920456 bytes, +0.000 MiB from baseline)
  external: 1.652 MiB (1732062 bytes, +0.000 MiB from baseline)
  arrayBuffers: 0.010 MiB (10475 bytes, +0.000 MiB from baseline)
2. Encoder module imported
  rss: 48.906 MiB (51281920 bytes, +6.453 MiB from baseline)
  heapTotal: 6.594 MiB (6914048 bytes, +0.250 MiB from baseline)
  heapUsed: 4.117 MiB (4316528 bytes, +0.378 MiB from baseline)
  external: 1.909 MiB (2001512 bytes, +0.257 MiB from baseline)
  arrayBuffers: 0.010 MiB (10475 bytes, +0.000 MiB from baseline)
3. WASM runtime initialized
  rss: 50.813 MiB (53280768 bytes, +8.359 MiB from baseline)
  heapTotal: 6.844 MiB (7176192 bytes, +0.500 MiB from baseline)
  heapUsed: 4.196 MiB (4400192 bytes, +0.458 MiB from baseline)
  external: 3.108 MiB (3259032 bytes, +1.456 MiB from baseline)
  arrayBuffers: 0.421 MiB (441409 bytes, +0.411 MiB from baseline)
4. ONNX model bytes loaded
  rss: 123.156 MiB (129138688 bytes, +80.703 MiB from baseline)
  heapTotal: 6.844 MiB (7176192 bytes, +0.500 MiB from baseline)
  heapUsed: 4.208 MiB (4412672 bytes, +0.469 MiB from baseline)
  external: 39.150 MiB (41051494 bytes, +37.498 MiB from baseline)
  arrayBuffers: 36.463 MiB (38233871 bytes, +36.453 MiB from baseline)
5. LM weights loaded
  rss: 144.172 MiB (151175168 bytes, +101.719 MiB from baseline)
  heapTotal: 6.844 MiB (7176192 bytes, +0.500 MiB from baseline)
  heapUsed: 4.216 MiB (4421184 bytes, +0.478 MiB from baseline)
  external: 60.118 MiB (63038638 bytes, +58.467 MiB from baseline)
  arrayBuffers: 46.947 MiB (49227443 bytes, +46.937 MiB from baseline)
6. Encoder instance created
  rss: 359.953 MiB (377438208 bytes, +317.500 MiB from baseline)
  heapTotal: 8.406 MiB (8814592 bytes, +2.063 MiB from baseline)
  heapUsed: 5.980 MiB (6270832 bytes, +2.241 MiB from baseline)
  external: 62.112 MiB (65128690 bytes, +60.460 MiB from baseline)
  arrayBuffers: 46.947 MiB (49227492 bytes, +46.937 MiB from baseline)
7. Production PCM segment allocated
  rss: 360.750 MiB (378273792 bytes, +318.297 MiB from baseline)
  heapTotal: 8.406 MiB (8814592 bytes, +2.063 MiB from baseline)
  heapUsed: 5.988 MiB (6278600 bytes, +2.249 MiB from baseline)
  external: 50.188 MiB (52625965 bytes, +48.536 MiB from baseline)
  arrayBuffers: 47.443 MiB (49747172 bytes, +47.433 MiB from baseline)
8. First segment encoded
  rss: 337.500 MiB (353894400 bytes, +295.047 MiB from baseline)
  heapTotal: 8.656 MiB (9076736 bytes, +2.313 MiB from baseline)
  heapUsed: 6.130 MiB (6428176 bytes, +2.392 MiB from baseline)
  external: 73.580 MiB (77153809 bytes, +71.928 MiB from baseline)
  arrayBuffers: 47.459 MiB (49764552 bytes, +47.449 MiB from baseline)
9. Five segments encoded
  rss: 318.688 MiB (334168064 bytes, +276.234 MiB from baseline)
  heapTotal: 8.656 MiB (9076736 bytes, +2.313 MiB from baseline)
  heapUsed: 6.475 MiB (6789256 bytes, +2.736 MiB from baseline)
  external: 73.646 MiB (77223329 bytes, +71.994 MiB from baseline)
  arrayBuffers: 47.525 MiB (49834072 bytes, +47.515 MiB from baseline)
10. Twenty segments encoded
  rss: 319.047 MiB (334544896 bytes, +276.594 MiB from baseline)
  heapTotal: 7.906 MiB (8290304 bytes, +1.563 MiB from baseline)
  heapUsed: 6.371 MiB (6680296 bytes, +2.632 MiB from baseline)
  external: 73.616 MiB (77191977 bytes, +71.964 MiB from baseline)
  arrayBuffers: 47.496 MiB (49802720 bytes, +47.486 MiB from baseline)
11. Encoder references released
  rss: 319.141 MiB (334643200 bytes, +276.688 MiB from baseline)
  heapTotal: 7.906 MiB (8290304 bytes, +1.563 MiB from baseline)
  heapUsed: 6.192 MiB (6492312 bytes, +2.453 MiB from baseline)
  external: 73.616 MiB (77191977 bytes, +71.964 MiB from baseline)
  arrayBuffers: 0.421 MiB (441458 bytes, +0.411 MiB from baseline)
12. Garbage collection requested
  rss: 319.141 MiB (334643200 bytes, +276.688 MiB from baseline)
  heapTotal: 7.906 MiB (8290304 bytes, +1.563 MiB from baseline)
  heapUsed: 6.194 MiB (6494488 bytes, +2.455 MiB from baseline)
  external: 26.541 MiB (27830715 bytes, +24.890 MiB from baseline)
  arrayBuffers: 0.421 MiB (441458 bytes, +0.411 MiB from baseline)
13. Final settled memory after a short delay
  rss: 319.141 MiB (334643200 bytes, +276.688 MiB from baseline)
  heapTotal: 7.906 MiB (8290304 bytes, +1.563 MiB from baseline)
  heapUsed: 6.204 MiB (6505336 bytes, +2.465 MiB from baseline)
  external: 26.541 MiB (27830715 bytes, +24.890 MiB from baseline)
  arrayBuffers: 0.421 MiB (441458 bytes, +0.411 MiB from baseline)

Peak sampled values:
  rss: 396.453 MiB (415711232 bytes, +354.000 MiB from baseline)
  heapTotal: 8.656 MiB (9076736 bytes, +2.313 MiB from baseline)
  heapUsed: 6.474 MiB (6788576 bytes, +2.735 MiB from baseline)
  external: 74.554 MiB (78175918 bytes, +72.903 MiB from baseline)
  arrayBuffers: 71.809 MiB (75297125 bytes, +71.799 MiB from baseline)

Timings:
  Initialisation: 18214.804 ms
  First encode: 905.591 ms
  Average encode time for segments 2-5: 871.123 ms
  Average encode time for segments 6-20: 863.046 ms

JSON report: tmp/encodec-memory-benchmark.json
       18.93 real        18.11 user         0.16 sys
           415711232  maximum resident set size
                   0  average shared memory size
                   0  average unshared data size
                   0  average unshared stack size
               25857  page reclaims
                   8  page faults
                   0  swaps
                   0  block input operations
                   0  block output operations
                   0  messages sent
                   0  messages received
                   0  signals received
                  13  voluntary context switches
               19112  involuntary context switches
        311848026927  instructions retired
         55398022140  cycles elapsed
           379002176  peak memory footprint
```
