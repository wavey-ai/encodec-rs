# encodec-rs

`encodec-rs` encodes and decodes EnCodec audio as deterministic, independently
framed ECDC chunks. The fixed profiles carry 48 kHz stereo at 3, 6, 12 or
24 kbps.

## Runtime

Neural inference runs in a small, optimised WebAssembly kernel. There is a
WebGPU backend where the browser can give one, and a single-threaded WASM SIMD
backend everywhere else; the playback adapter selects WebGPU when it can and
falls back to WASM SIMD when it cannot. The kernel implements the EnCodec
convolutions, recurrent layers, normalisation and residual vector quantisation,
and the weights ship beside it as packed `float32` blobs.

Rust/WASM owns the rest: the quantized language model, arithmetic coding, ECDC
framing and CRC32, and the optional triangle-overlap seam.

**ONNX Runtime is deliberately not supported.** The release packages contain no
ONNX files and never load an ONNX runtime. The build reads fixed ONNX models
only as a source of structure and weights, which it packs into the kernel's own
form; nothing ONNX-shaped ships or runs.

The Apple package provides the same complete encode and decode through MLX on
Metal, again with no ONNX runtime at run time.

## Fixed profiles

| Bundle | Rate | Owned samples | Model samples | Codebooks |
|---|---|---:|---:|---:|
| `encodec_48khz_3kbps_1333ms` | 3 kbps | 64,000 | 64,960 | 2 |
| `encodec_48khz_6kbps_1333ms` | 6 kbps | 64,000 | 64,960 | 4 |
| `encodec_48khz_12kbps_1333ms` | 12 kbps | 64,000 | 64,960 | 8 |
| `encodec_48khz_12kbps_7cb_1333ms` | 12 kbps profile / 10.5 kbps raw | 64,000 | 64,960 | 7 |
| `encodec_48khz_24kbps_1333ms` | 24 kbps | 64,000 | 64,960 | 16 |

Each model window has 480 guard samples before and after the owned region. The
encoder supplies real source guards where they exist and zeros at the start and
end of a file, and processes every fixed latent step; the ECDC chunk records the
owned sample count. `fixedEcdcBundleName(bandwidthKbps, chunkMs)` keeps the
standard mapping, including 12 kbps to 8 codebooks; use
`fixedEcdcBundleNameWithCodebooks(12, 7, chunkMs)` for the seven-codebook
prefix.

## Guards and reconstruction

Guard samples give the model real context near each owned boundary. The
low-level decoder returns the complete model window, including both guards. The
standard ECDC assembly crops the guards and concatenates the owned PCM
untouched, with no implicit seam repair; a caller that wants one asks for
`triangle_overlap_add_planar_frames` explicitly.

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
files — one file can already hold many independent chunks.

## State and session reuse

The neural frame encoder and decoder are stateless between calls, so one
initialised runtime can process chunks or tracks in any order. Session reuse
keeps allocations, packed weights and prepared kernels; it changes speed, not
results. The q8 language model and the arithmetic coder each start fresh for
every chunk.

## Compatibility

The custom WASM and MLX backends write and read the same fixed-profile ECDC
format. Existing ECDC files written by an earlier encoder remain decodable when
their recorded profile and q8 language model are available. This applies to the
current fixed-profile envelope only.

## Release bundle layout

```bash
scripts/build_wasm_fixed_bundles.sh
```

The builder needs Emscripten, Rust nightly and `wasm-bindgen`. `PYTHON_BIN`
points at the environment holding the ONNX package the weights are read from.

```text
dist/wasm-fixed-bundles/
  browser-neural-runtime.js
  custom-encoder-runtime.js
  custom-decoder-runtime.js
  webgpu-kernel-runtime.js
  pkg/encodec_rs.js + encodec_rs_bg.wasm
  bundles/<profile>/
    bundle.json
    lm_weights_q8.bin
    manifest.json
    encoder/  metadata.json, weights.json, weights.f32le, encodec-encoder*.wasm/mjs
    decoder/  metadata.json, weights.json, weights.f32le, encodec-convtranspose*.wasm/mjs
```

The manifest records each asset's size and SHA-256.

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
