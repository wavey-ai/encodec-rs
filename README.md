# encodec-rs

`encodec-rs` encodes and decodes EnCodec audio as deterministic, independently
framed ECDC chunks. The fixed profiles carry 48 kHz stereo at 3, 6, 12 or
24 kbps.

## Runtime architecture

| Work | Implementation |
|---|---|
| Neural encoder and decoder | WebGPU compute kernels, with a custom C/WASM SIMD kernel as fallback |
| Neural weights | Packed `float32` blobs |
| Quantized language model | Rust/WASM |
| Arithmetic coding | Rust/WASM |
| ECDC framing and CRC32 | Rust/WASM |
| Guard cropping and optional triangle overlap | Rust/WASM |
| Apple neural backend | MLX on Metal |

Both neural backends implement the EnCodec convolutions, recurrent layers,
normalisation and residual vector quantisation. The kernels hold no
track-specific values; one bundle works for all valid audio of its profile.

## ONNX Runtime

Removed to use our far smaller foot-print and faster custom backend.

## Browser backends

The browser package contains a WebGPU backend and a single-thread WASM SIMD
backend. The playback adapter selects WebGPU when it can create an adapter and
device, and falls back to WASM SIMD when WebGPU is absent or initialisation
fails. Encoding uses WASM SIMD by default; tests can request WebGPU encoding
explicitly. Backend detection belongs to the browser runtime — the Rust codec
core does not inspect GPU state.

Call `createBrowserEncoder()` or `createBrowserDecoder()` with
`backend: "auto"` for runtime selection, or `"webgpu"` / `"wasm-simd"` to
require one backend during tests.

The decoder processes each ECDC chunk in this order: decode the LM and
arithmetic payload, run the neural decoder, crop the private model context, and
deliver the owned planar PCM. The callback also receives the complete guarded
model window, which the caller can use for optional seam repair. Complete PCM is
retained only when the caller asks for it.

### Mobile Safari result

A full-track physical iPhone test (Safari 26.5.2) delivered all 171 chunks of a
227.863-second song.

| Operation | Neural time | LM entropy time | Total time | RTFx |
|---|---:|---:|---:|---:|
| Encode | 58.916 s | 34.404 s | 93.320 s | 2.442× |
| Incremental decode | 55.266 s | 34.297 s | 90.389 s | 2.521× |

The first playable chunk was ready 498 ms after prewarm; cold decoder setup
took 32.812 s. All 277,704 encoder codes matched the frozen reference and the
ECDC file size was unchanged. The four-second decoder gate measured 80.136 dB
SNR against frozen decoded PCM. Applications should prewarm and retain this
backend. See [the WebKit WebGPU benchmark](docs/benchmarks/webkit-webgpu-20260822/README.md).

## Apple MLX backend

The Apple package provides complete ECDC encoding and decoding through MLX on
Metal. Rust owns deterministic q8 LM inference, arithmetic coding, ECDC framing
and CRC32. The MLX bundles contain `encode_frame.safetensors`,
`decode_frame.safetensors` and `lm_weights_q8.bin`, and load no ONNX models.

One initialised backend can process many files. iOS uses one neural frame per
call by default, which avoids the memory pressure measured with larger batches;
macOS groups up to eight compatible frames by default. Callers may override
either. Call `prewarm(frameBatchSize:)` before the first timed operation, with
the same batch size used in normal operation.

Apple callers can request a canonical 6 kbps ECDC alongside 12 kbps with
`encodeEcdcOutputs(..., derived6KBundleURL:)`. The neural encoder runs once;
Rust then retains one 12 kbps LM weight set, advances two independent LM states
and fuses their shared matrix reads. Both files are byte-identical to separate
canonical encodes. Omitting the bundle takes the single-output path.
Measurements are in [`apple/README.md`](apple/README.md).

Fixed-context decoding writes MLX windows directly to the final PCM (borrowed
windows in memory, planar PCM for file output) and allocates no intermediate
complete planar track. The Apple CLI accepts RIFF PCM16, packed PCM24, PCM32 and
float32 input, including PCM and float `WAVE_FORMAT_EXTENSIBLE`, and does not
resample; the current profiles require 48 kHz stereo input.

```bash
scripts/build-apple-mlx.sh
```

### Apple performance

The Apple tests used the same 227.863-second master and the 12 kbps, 1333 ms
profile.

| Runtime | Device path | Batch | Encode RTFx | Decode RTFx |
|---|---|---:|---:|---:|
| Custom WASM | Apple M1 CPU | 1 | 2.922× | 3.121× |
| MLX | Apple M1 Metal | 8 | 8.136× | 7.252× |
| MLX | Physical iPhone Metal | 1 | approximately 7× | approximately 5× |

Batch-one MLX decoding matched the frozen PCM bit-for-bit on macOS. Batch-eight
decoding measured 111.27 dB SNR, 0.000000596 RMSE and 0.000202447 maximum error
against that output. MLX encoding produced the same ECDC bytes at batch one and
batch eight. Decode remains slower on iPhone because it runs serial q8 entropy
decoding before neural synthesis.

## Performance

The August 2026 audit used an Apple M1 host and one WASM thread, processing a
227.863-second, 48 kHz stereo PCM24 master. Higher RTFx is faster, where
`RTFx = audio duration / wall time`.

Against the ONNX-WASM controls: encoding model time is 4.6% faster and the
bytes are identical; decoding executes the model 3.19× faster; in the browser,
model execution is 1.263× faster. Against official Meta on this host: complete
encode and decode are faster, the payload is 6.63% larger, and quality is a
tie.

### Encoding

| Path | Model time | Total time | RTFx | Result |
|---|---:|---:|---:|---|
| Paired ONNX-WASM control | 53.149 s | 79.090 s | 2.881× | Exact reference |
| Custom WASM encoder | 50.699 s | 76.623 s | 2.974× | Byte-identical ECDC |

The custom encoder reduced paired model time by 4.6% and paired total time by
3.1%. The first direct scalar baseline reached approximately 1.58× realtime; the
current path reaches 2.97×. Exact Rust entropy optimisations reduced full-track
encode time from 31.304 s to 25.831 s, a 17.5% reduction in that stage with no
payload change.

### Decoding

| Path | Model time | Total time | RTFx |
|---|---:|---:|---:|
| Custom WASM A | 49.817 s | 76.612 s | 2.974× |
| ONNX-WASM control | 149.130 s | 175.174 s | 1.301× |
| Custom WASM B | 46.752 s | 73.452 s | 3.102× |

The second custom run executed the model 3.19× faster and completed warm
decoding 2.39× faster than the control. Exact Rust entropy optimisations reduced
full-track decode time from 31.286 s to 25.433 s, an 18.7% reduction in that
stage.

### Browser control

Headless Chrome 151 processed three warm frames with a single-thread WASM
backend.

| Browser decoder | Setup | Median model time | Model RTFx |
|---|---:|---:|---:|
| ONNX Runtime Web | 514.1 ms | 1,075.2 ms | 3.72× |
| Custom WASM | 280.2 ms | 851.1 ms | 4.70× |

The custom browser decoder was 1.263× faster than ONNX Runtime Web for model
execution in this test. The production package completed an ONNX-free browser
encode and decode round trip: the encoder produced the exact 4,589-byte
reference ECDC file and the decoder produced bit-identical PCM for that
four-second test.

## Numerical parity

The custom encoder produces the same codes, scale and ECDC bytes as the
ONNX-WASM control. The full-track ECDC SHA-256 is:

```text
35cd76f783228d79268cbc2ced6901baf37874c571598cbc32098485de1721c4
```

The full custom decoder measured 85.363 dB SNR against the ONNX-WASM decoder,
with the same frame count, RMS, peak and clipping count; one unusual frame
caused most of the full-track numerical difference. Standard SIMD improved
parity but reduced speed, so the release selects relaxed SIMD when the browser
supports it and falls back to standard WASM SIMD otherwise.

## Supported fixed profiles

| Bundle | Rate | Owned samples | Model samples | LM steps | Codebooks |
|---|---:|---:|---:|---:|---:|
| `encodec_48khz_3kbps_1333ms` | 3 kbps | 64,000 | 64,960 | 203 | 2 |
| `encodec_48khz_6kbps_1333ms` | 6 kbps | 64,000 | 64,960 | 203 | 4 |
| `encodec_48khz_12kbps_1333ms` | 12 kbps | 64,000 | 64,960 | 203 | 8 |
| `encodec_48khz_12kbps_7cb_1333ms` | 12 kbps profile / 10.5 kbps raw | 64,000 | 64,960 | 203 | 7 |
| `encodec_48khz_24kbps_1333ms` | 24 kbps | 64,000 | 64,960 | 203 | 16 |

Each model window has 480 guard samples before and after the owned region.
`fixedEcdcBundleName(bandwidthKbps, chunkMs)` keeps the standard upstream
mapping, including 12 kbps to 8 codebooks; use
`fixedEcdcBundleNameWithCodebooks(12, 7, chunkMs)` to select the seven-codebook
prefix. Its raw neural rate is 10.5 kbps and the final LM-compressed ECDC rate
is content-dependent.

The encoder supplies real source guards where those samples exist, and zeros at
the start and end of a file. The model always receives its fixed input shape, so
a short final owned region uses zero padding. The encoder processes all fixed
latent steps, and the ECDC chunk records the owned sample count.

## Guards and reconstruction

Guard samples give the neural model real context near each owned boundary. The
low-level decoder returns the complete decoded model window, including both
guards. The standard ECDC assembly crops the guards and concatenates untouched
owned PCM, with no implicit seam repair. A caller can instead apply
triangle-weighted overlap across adjacent decoded guard windows; the `seam` API
provides `triangle_overlap_add_planar_frames` for that, and the caller must
select it explicitly.

## State and session reuse

The neural frame encoder and decoder are not stateful between calls, so one
initialised runtime can process independent chunks or different tracks in any
order. Session reuse keeps allocations, packed weights and prepared kernels; it
changes speed, not results. The q8 language model and the arithmetic coder each
start fresh for every ECDC chunk, and the model's state cache allocation is
cleared and reused between chunks.

## Runtime-change compatibility

The custom WASM and MLX backends write and read the same fixed-profile ECDC
format, so existing ECDC files written by an earlier encoder remain decodable
when their recorded profile and q8 language model are available. A compatibility
gate decoded a complete 192.936-second, 145-chunk ECDC file produced by the
former ONNX Runtime Web encoder with the MLX decoder and recovered all
9,260,919 stereo frames. A randomized audit interleaved two tracks through one
session: all 316 interleaved chunk hashes matched their isolated-session hashes.
This applies to the current fixed-profile envelope only.

## Difference from official Meta EnCodec

The official Meta CLI accepts a complete file, but its neural model still
processes segments: 48 kHz audio uses 48,000-sample segments with a
47,520-sample stride, a 480-sample difference that creates a 10 ms overlap, and
Meta combines decoded segments with triangle-weighted overlap-add. Meta starts
one language-model state and one arithmetic coder per neural segment.

| Property | `encodec-rs` fixed profile | Official Meta 48 kHz profile |
|---|---|---|
| Long-file unit | 64,000 or 86,400 owned samples | 48,000 model samples |
| Source context | 480 guard samples on each side | 480 adjacent overlap samples |
| Reconstruction | Caller-selected crop or triangle overlap | Triangle overlap-add |
| Entropy reset | Each owned ECDC chunk | Each neural segment |
| Entropy probabilities | Deterministic q8 integer path | Floating-point PyTorch path |
| Segment framing | Explicit length and CRC32 | Expected symbol count |

Meta ECDC version 0 does not store an encoded length or CRC per segment; its
decoder infers each boundary from the expected symbol count, so
floating-point probability differences across architectures can change
arithmetic bit consumption and make later segments unreadable. `encodec-rs`
contains an arithmetic failure within one length-framed, CRC-protected chunk.
The comparison uses Meta commit `0e2d0aed29362c8e8f52494baf3e6f99056b214f`.

- [Model segmentation](https://github.com/facebookresearch/encodec/blob/0e2d0aed29362c8e8f52494baf3e6f99056b214f/encodec/model.py)
- [Entropy compression](https://github.com/facebookresearch/encodec/blob/0e2d0aed29362c8e8f52494baf3e6f99056b214f/encodec/compress.py)
- [Binary container](https://github.com/facebookresearch/encodec/blob/0e2d0aed29362c8e8f52494baf3e6f99056b214f/encodec/binary.py)
- [Triangle overlap-add](https://github.com/facebookresearch/encodec/blob/0e2d0aed29362c8e8f52494baf3e6f99056b214f/encodec/utils.py)

## Full-file comparison

The same Apple M1 host processed the 227.863-second master with one CPU thread.
The Meta core row excludes setup; the Meta CLI row includes process and model
setup.

| Path | Encode time | Encode RTFx | Decode time | Decode RTFx | ECDC bytes |
|---|---:|---:|---:|---:|---:|
| Custom `encodec-rs` | 76.623 s | 2.974× | 73.452 s | 3.102× | 296,562 |
| Meta loaded core API | 104.004 s | 2.191× | 104.171 s | 2.187× | 278,134 |
| Meta standard fresh CLI | 108.605 s | 2.098× | 106.209 s | 2.145× | 278,134 |

On this host the custom path completed encoding and decoding faster than both
Meta rows; the `encodec-rs` payload is 6.63% larger than the Meta payload. The
comparison measures complete implementations and does not isolate Python or FFI
overhead.

## Full-file quality

The seamless PCM24 master is the reference for each result, and the
`encodec-rs` row uses untouched owned PCM from the reference decoder.

| Candidate | SNR | SI-SDR | Log-spectral distance | Spectral convergence | Loudness delta |
|---|---:|---:|---:|---:|---:|
| `encodec-rs`, untouched PCM | 6.594 dB | 5.723 dB | 12.524 dB | 0.27869 | -0.410 LU |
| Official Meta | 6.600 dB | 5.719 dB | 12.596 dB | 0.27893 | -0.434 LU |

Official ViSQOL scored ten matched, active eight-second excerpts.

| Candidate | Mean MOS-LQO | Median MOS-LQO | Standard deviation |
|---|---:|---:|---:|
| `encodec-rs` | 4.2874 | 4.2809 | 0.0663 |
| Official Meta | 4.2769 | 4.2847 | 0.0676 |

The paired mean difference was `+0.0105` for `encodec-rs`, with a 95% confidence
interval of `-0.0038` to `+0.0248`. The aggregate results do not show a material
quality difference, and the interval spans zero, so neither candidate is a
reliable winner on quality.

## Seam analysis

The master has no codec join; each candidate join uses the same master samples
as its reference. The analysis uses a 20 ms window around each join; lower seam
excess is better and higher seam SNR is better.

| Reconstruction | Joins | Median excess | P90 excess | Median seam SNR | Median step error |
|---|---:|---:|---:|---:|---:|
| Cubic Hermite experiment | 170 | 0.587 dB | 3.929 dB | 5.169 dB | 0.02957 |
| Untouched owned PCM | 170 | 0.476 dB | 3.011 dB | 5.245 dB | 0.08958 |
| Triangle guard overlap | 230 | 0.019 dB | 2.099 dB | 6.118 dB | 0.02699 |

Triangle guard overlap produced the best measured join distribution. Hermite
reduced the sample step but degraded master-reference fidelity at 124 of 170
joins and raised the output peak from `+2.552 dBFS` to `+5.190 dBFS`, so the
results do not support it for this compressed musical audio.

![Largest Hermite repair effect](docs/benchmarks/encodec-full-file-20260821/encodec-rs-largest-repair-effect.png)

![Worst Hermite residual spectrogram](docs/benchmarks/encodec-full-file-20260821/encodec-rs-worst-join-residual.png)

![Worst triangle overlap join](docs/benchmarks/encodec-full-file-20260821/meta-worst-overlap-join.png)

Python produced the analysis and figures; runtime reconstruction remains Rust
and WASM.

## ECDC layout

One `.ecdc` file holds one header and one or more independent chunks.

```text
4 bytes   magic: "ECDC"
1 byte    version: 0
4 bytes   metadata JSON length, big-endian u32
N bytes   metadata JSON

repeated chunks:
4 bytes   payload length, big-endian u32
4 bytes   payload CRC32, big-endian u32
M bytes   payload
```

The metadata records the model, audio length, codebook count, LM mode and q8
weight hash; fixed containers record the LM frame length through `fl`. The q8
entropy path uses bitstream version `acv=2`. Older raw payloads and
floating-point LM payloads are not supported. Do not concatenate complete ECDC
files — one file can already hold many independent chunks. The current `acv=2`
envelope is repository-specific and is not the proposed Profile 1 container.

## Release bundle layout

Run the release builder to replace `dist/wasm-fixed-bundles`:

```bash
scripts/build_wasm_fixed_bundles.sh
```

The builder uses `/opt/anaconda3/envs/encodec-export/bin/python` by default; set
`PYTHON_BIN` to point at the environment holding the ONNX package the weights
are read from. It also requires Emscripten, Rust nightly and `wasm-bindgen`.

```text
dist/wasm-fixed-bundles/
  browser-neural-runtime.js
  custom-encoder-runtime.js
  custom-decoder-runtime.js
  webgpu-kernel-runtime.js
  webgpu-encoder-runtime.js
  webgpu-decoder-runtime.js
  webgpu-ecdc-decoder-runtime.js
  manifest.json
  pkg/encodec_rs.js + encodec_rs_bg.wasm
  bundles/<profile>/
    bundle.json
    lm_weights_q8.bin
    manifest.json
    encoder/  metadata.json, weights.json, weights.f32le, encodec-encoder*.wasm/mjs
    decoder/  metadata.json, weights.json, weights.f32le, encodec-convtranspose*.wasm/mjs
```

The manifest records each asset's size and SHA-256.

## Browser verification

```bash
npm ci --prefix browser-smoke
PORT=8798 python3 browser-smoke/serve.py
```

Run the packaged encoder lifecycle test:

```bash
BROWSER_BENCH_MODE=runtime-chunk \
node scripts/benchmark-browser-decoders.mjs
```

Run the complete custom encode and decode test:

```bash
BROWSER_BENCH_MODE=roundtrip \
BROWSER_CUSTOM_ENCODER_ROOT=dist/wasm-fixed-bundles/bundles/encodec_48khz_12kbps_1333ms/encoder/ \
BROWSER_CUSTOM_ENCODER_KERNEL=dist/wasm-fixed-bundles/bundles/encodec_48khz_12kbps_1333ms/encoder/encodec-encoder-relaxed.mjs \
BROWSER_CUSTOM_DECODER_ROOT=dist/wasm-fixed-bundles/bundles/encodec_48khz_12kbps_1333ms/decoder/ \
BROWSER_CUSTOM_DECODER_KERNEL=dist/wasm-fixed-bundles/bundles/encodec_48khz_12kbps_1333ms/decoder/encodec-convtranspose-relaxed.mjs \
node scripts/benchmark-browser-decoders.mjs
```

These tests use local headless Chrome and do not enable WebGPU or browser
threading.

## Library features

Container and entropy functions without a neural runtime:

```toml
encodec-rs = { git = "https://github.com/wavey-ai/encodec-rs.git", features = ["ecdc"] }
```

Explicit PCM seam operations:

```toml
encodec-rs = { git = "https://github.com/wavey-ai/encodec-rs.git", features = ["seam"] }
```

The `ecdc::FrameCodec` trait separates neural frame execution from ECDC framing.
The project is licensed under the MIT License.
