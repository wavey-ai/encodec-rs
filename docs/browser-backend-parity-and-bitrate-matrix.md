# Browser backend parity and the PRAY 4 ME bitrate matrix

Status: measured investigation. One open defect and one open question.

Test date: 2026-08-26.

## Objective

Encode one master at every 1333 ms profile through the browser runtime.

Decode each result back to WAV in the same runtime.

Record payload size, effective bitrate, and decoded parity for each profile.

Then repeat the encode without WebGPU and compare the bytes.

## Source

`PRAY 4 ME_MIX 4 CONFIRMATION_130323.wav`.

222.532 s, 48 kHz, stereo, 24-bit PCM.

Each profile produced 167 chunks.

## Bundle roots are not interchangeable

The browser runtime has two bundle roots and they are not equivalent.

`onnx-bundles/` holds ONNX graphs for the onnxruntime-web path.

`dist/wasm-fixed-bundles/bundles/` holds the custom WASM kernels and their own weights.

The seven-codebook profile only exists as a real encoder in the second root.

`onnx-bundles/encodec_48khz_12kbps_7cb_1333ms/` hardlinks the eight-codebook
`encode_frame.onnx` from the 12 kbps bundle and declares `num_codebooks: 7`.

Its `bundle.json` declares `codebook_prefix_source`, but no code reads that field.

The encoder therefore emits eight codebooks where seven are expected.

Both the native CLI and the onnxruntime-web page fail on it:

```text
encoded code shape mismatch, expected [batch, 7, frames], got [7, 8, 203]   # src/ecdc.rs:1004
segment 0 codes length 1624 does not match 1421                             # browser-smoke/webgpu-matrix.js
```

`dist/wasm-fixed-bundles/bundles/encodec_48khz_12kbps_7cb_1333ms/encoder/` is correct.

Its `metadata.json` declares `numCodebooks: 7`.

Its `weights.f32le` is 33,395,648 B against 33,924,032 B for the eight-codebook bundle.

That difference is exactly one RVQ codebook.

Entry points in `browser-smoke/webgpu-matrix.js` split along the same line:

| Entry point | Bundle root | Runtime |
|---|---|---|
| `encode` / `decode` | `onnx-bundles/` | onnxruntime-web |
| `customRoundTrip` | `onnx-bundles/` for metadata, explicit kernel roots | custom WASM kernels |
| `webGpuRoundTrip` / `webGpuDecodeEcdc` | `dist/wasm-fixed-bundles/bundles/` | WebGPU kernels |

`customRoundTrip` hardcodes the ONNX root for `bundle.json` and the LM weights.

That is safe today because `lm_weights_q8.bin` is byte-identical in both roots.

It is the same latent trap that produced the seven-codebook failure.

## Bitrate matrix

WebGPU kernels, `dist` bundles, encode and decode in one browser session.

Effective kbps is whole-file bytes over 222.532 s, including header and framing.

SNR is against the 24-bit source.

| Profile | Codebooks | ECDC bytes | Effective kbps | SNR | Wall time |
|---|---:|---:|---:|---:|---:|
| 3 kbps | 2 | 68,005 | 2.445 | 7.77 dB | 130 s |
| 6 kbps | 4 | 136,978 | 4.924 | 9.50 dB | 137 s |
| 12 kbps 7cb | 7 | 246,737 | 8.870 | 10.98 dB | 149 s |
| 12 kbps | 8 | 283,879 | 10.205 | 11.33 dB | 144 s |
| 24 kbps | 16 | 594,176 | 21.361 | 13.04 dB | 194 s |

The seven-codebook profile is 13.1% smaller than eight for 0.35 dB.

The step down from seven to four codebooks costs 1.48 dB.

The step up from eight to sixteen costs 109% more bytes for 1.71 dB.

The 12 kbps row agrees with the recorded qualification figure for this track.

That record is `qualification-results/20260814-lori-confirmation-bitrate/`.

It lists 282,347 payload bytes and 10.150 kbps, excluding header and framing.

## Backend parity: the encoders do not agree

The same audio and profile produce different bytes on different backends.

WebGPU and custom WASM were byte-compared directly. Every profile differs.

| Profile | WebGPU bytes | Custom WASM bytes | First mismatch | Differing bytes | CPU wall time |
|---|---:|---:|---:|---:|---:|
| 3 kbps 2cb | 68,005 | 67,994 | 182 | 64,079 | 200 s |
| 6 kbps 4cb | 136,978 | 137,012 | 182 | 134,205 | 221 s |
| 12 kbps 7cb | 246,737 | 246,751 | 181 | 244,667 | 210 s |
| 12 kbps 8cb | 283,879 | 283,869 | 181 | 274,992 | 203 s |
| 24 kbps 16cb | 594,176 | 594,220 | 182 | 591,663 | 234 s |

The first mismatch is always in the first chunk payload, at 181 or 182.

The header is 96 B and each chunk adds 8 B of framing.

The divergence therefore starts in the opening frame's codes, and does not
accumulate over the track. One early symbol disagreement explains every later
byte, because the arithmetic coder state diverges from that point on.

Decoded quality is unaffected at every profile:

| Profile | WebGPU SNR | Custom WASM SNR |
|---|---:|---:|
| 3 kbps 2cb | 7.767 dB | 7.763 dB |
| 6 kbps 4cb | 9.498 dB | 9.495 dB |
| 12 kbps 7cb | 10.980 dB | 10.975 dB |
| 12 kbps 8cb | 11.332 dB | 11.328 dB |
| 24 kbps 16cb | 13.038 dB | 13.033 dB |

The encoders are choosing near-identical codes, not identical codes.

This is float arithmetic in the conv, LSTM, and RVQ nearest-neighbour stages.

It is not a logic defect in the entropy path.

The ONNX path was measured at three profiles, by size only, on a separate run:
67,994 B at 3 kbps, 137,012 B at 6 kbps, and 283,870 B at 12 kbps 8cb.

The first two match the custom WASM sizes and the third does not.

Equal size is not equal bytes, so no two of the three implementations are
proven identical. Only the WebGPU and custom WASM streams were byte-compared.

The seven-codebook profile runs on both backends.

Custom WASM takes 200 to 234 s for a 222.532 s track.

A browser without WebGPU can encode this material, at roughly realtime.

## Why this matters

The q8 LM and the arithmetic coder are deterministic by construction.

`scripts/verify-lm-wasm-parity.mjs` verifies that stage.

Nothing verifies that the neural encoder produces identical codes across backends.

Deterministic payload bytes are load-bearing downstream.

A record's ECDC payload is the content its pressing is built from.

If a GPU browser and a CPU browser encode one master to different bytes, they
mint different records, which affects content-addressed identity, dedup, and any
signature computed over the payload.

The paired encoder is unaffected by this.

It derives both streams from one code set, so primary and derived cannot disagree.

## Open items

Decide whether encoding is pinned to one backend for published records.

Add a neural-encoder parity check across WebGPU, custom WASM, and ONNX.

Make the ONNX and native paths honour `codebook_prefix_source`, or remove the
stub seven-codebook bundle from `onnx-bundles/`.

Decide whether the CPU fallback needs a non-SIMD tier. Both kernel builds
require simd128 (`-msimd128` and `-mrelaxed-simd`), and the Rust wasm is built
with `-C target-feature=+simd128`, so a browser without SIMD cannot run either.
That targets pre-16.4 Safari and pre-91 Chrome.

## Appendix: the payload has no compression headroom

Measured on `lori-asha-westside-lp-hq.ecdc`, 255,061 B.

| Component | Bytes | Share |
|---|---:|---:|
| Magic, version, length | 9 | 0.004% |
| Header JSON | 87 | 0.03% |
| Chunk framing, 211 x 8 B | 1,688 | 0.66% |
| Arithmetic-coded payload | 253,277 | 99.30% |

Payload byte entropy is 7.9991 bits of 8, which is 99.99% of maximum.

The order-0 ideal is 253,248 B against an actual 253,277 B.

That is 29 B of slack in a quarter megabyte.

Brotli, Zopfli, Snappy, gzip, bzip2, xz, LZMA, PPMd, zstd, and lz4 were each run
over the payload and over four text renderings of its RGB carrier pixels.

No codec and no representation went below 100% of the original payload size.

The best result was Brotli q11 over a hex rendering, at 255,083 B.

That is 22 B worse than storing the payload directly.

Remaining slack is structural and bounded by roughly 1.3 KB:

- per-chunk CRC32, 211 x 4 B, redundant inside a record that is separately signed;
- chunk length fields, 211 x 4 B, where no chunk exceeds 1,344 B;
- the header, already stripped on-record by `record-core/src/ecdc.rs`.

The lever on payload size is codebook count, not compression.
