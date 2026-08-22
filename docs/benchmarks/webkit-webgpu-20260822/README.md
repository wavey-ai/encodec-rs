# WebKit WebGPU kernel gate

This gate ran on 22 August 2026.

It compares WebGPU with the current WASM SIMD fallback in Safari.

Both paths process identical deterministic inputs and weights in a dedicated worker.

Each case represents one production-sized EnCodec convolution with 425,721,856 multiply-accumulates.

The encoder case uses the largest strided convolution.

The decoder case uses its mirrored transposed convolution.

## Results

Higher speedup is better.

| Device | Direction | WASM median | WebGPU batched | Speedup | SNR |
|---|---|---:|---:|---:|---:|
| Apple M1, Safari 26.5 | Encode | 26.000 ms | 5.667 ms | 4.588× | 129.849 dB |
| Apple M1, Safari 26.5 | Decode | 24.000 ms | 4.667 ms | 5.143× | 132.749 dB |
| iPhone 13 mini, Safari 26.5.2, run 1 | Encode | 25.000 ms | 13.111 ms | 1.907× | 129.849 dB |
| iPhone 13 mini, Safari 26.5.2, run 2 | Encode | 25.000 ms | 13.889 ms | 1.800× | 129.849 dB |
| iPhone 13 mini, Safari 26.5.2, run 1 | Decode | 23.000 ms | 12.778 ms | 1.800× | 132.749 dB |
| iPhone 13 mini, Safari 26.5.2, run 2 | Decode | 24.000 ms | 15.222 ms | 1.577× | 132.749 dB |

The WebGPU value divides one nine-dispatch command buffer by nine.

This method approximates chained graph execution and excludes setup and transfers.

An isolated result includes one queue submission and one completion wait.

Safari 26.5 rejected the WASM relaxed-SIMD module on macOS.

The Safari control therefore uses the production standard-SIMD fallback.

## Capability result

WebGPU compute worked in the window and a dedicated worker on both devices.

Both devices expose `shader-f16`, packed 4×8 integer dot products, and timestamp queries.

Neither device exposes subgroups.

The iPhone exposes 32 KiB workgroup memory and a 683 MiB storage-buffer limit.

These limits can hold either complete float32 encoder or decoder weights.

## Decision

Continue with an ORT-free float32 WebGPU encoder and decoder.

Keep WASM SIMD as the automatic fallback.

Do not select WebGPU by default until the complete codec passes quality, memory, and sustained thermal gates.

The full test must include LSTM, normalization, quantization, entropy coding, transfers, and complete-file assembly.

Raw measurements are in [results.json](results.json).

## Complete codec result

A physical iPhone ran the complete 12 kbps, 1333 ms codec in Safari 26.5.2.

The source was a 227.863-second, 48 kHz stereo master.

The ECDC file contained 171 independently coded chunks and used 296,596 bytes.

Higher RTFx is faster.

| Operation | Neural work | LM entropy work | Pipeline time | RTFx |
|---|---:|---:|---:|---:|
| Encode | 58.916 s | 34.404 s | 93.320 s | 2.442× |
| Incremental decode | 55.266 s | 34.297 s | 90.389 s | 2.521× |

The incremental decoder completed LM decode and neural decode for one chunk before it started the next chunk.

It delivered all 171 chunks and 10,937,424 owned PCM frames in order.

The test did not retain a complete PCM track.

The first 1.333-second playable chunk was ready in 498 ms after prewarm.

Cold decoder setup took 32.812 seconds.

Applications must prewarm and retain the decoder.

The encoder produced all 277,704 neural codes identically.

It changed 78 scale values by at most 2.98e-8.

The ECDC size did not change.

The four-second decoder gate measured 80.136 dB SNR against frozen decoded PCM.

It measured 0.000009251 RMSE and 0.000037253 maximum error.

No decoded sample was non-finite.

The incremental callback returns cropped playable PCM and the complete guarded model window.

The caller can apply seam repair to the guarded windows.

The decoder does not apply seam repair automatically.

Complete measurements are in [full-codec-results.json](full-codec-results.json).

References:

- [WebGPU in Safari 26](https://webkit.org/blog/16993/news-from-wwdc25-web-technology-coming-this-fall-in-safari-26-beta/)
- [Enable WebDriver on iOS](https://developer.apple.com/documentation/safari-developer-tools/ios-enabling-webdriver)
