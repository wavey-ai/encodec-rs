# Matrix Run Notes

Latest run date: 2026-05-19.

Input audio: `target/lori-asha-wasm-native/wav/02 - Lori Asha - Westside.48k-stereo.wav`

Scope: current q8 LM entropy path only. This run is not intended to preserve AVC/backward-compatible payload behavior.

All generated WAVs from the latest local Linux runs have been pulled back to this machine.

## Latest Run Folders

These are the folders to use for manual listening and matrix inspection.

| Folder | Contents | Use |
|---|---:|---|
| `target/gpu-matrix/` | 32 WAV, 8 ECDC, ~1.2G | Main native CPU/GPU matrix. Encoders and decoders cover macOS ARM64 CPU, macOS CoreML GPU, Linux x86_64 CPU, and Linux CUDA GPU across 6 kbps and 12 kbps. |
| `target/webgpu-matrix/` | 4 WAV, 2 ECDC, ~154M | Real browser WebGPU rows. macOS Chrome WebGPU encode to Linux CPU decode, plus Linux CPU encode to macOS Chrome WebGPU decode, across 6 kbps and 12 kbps. |
| `target/q8-matrix/` | 8 WAV, 4 ECDC, ~306M | Native q8 CPU cross-arch baseline. macOS ARM64 CPU and Linux x86_64 CPU encode/decode combinations across 6 kbps and 12 kbps. |
| `target/matrix-wav-manifest.txt` | SHA-256 manifest | Hash list for the selected latest-run WAV outputs. |

## Latest Environments

The main native GPU matrix uses these environment names in file names:

- `native-rust-macos-arm64-cpu`
- `native-rust-macos-arm64-coreml-gpu`
- `native-rust-linux-x86_64-cpu`
- `native-rust-linux-x86_64-cuda-gpu`

The browser WebGPU matrix uses:

- `browser-webgpu-macos-arm64`
- `native-rust-linux-x86_64-cpu`

File names follow this shape:

```text
westside.encode-<encoder-env>.decode-<decoder-env>.encodec_48khz_<bitrate>.q8.wav
westside.encode-<encoder-env>.encodec_48khz_<bitrate>.q8.ecdc
```

## Supporting Current Artifacts

These were produced while building or validating the latest run, but are not the main manual-listening folders.

| Folder | Contents | Notes |
|---|---:|---|
| `target/env-matrix/` | 6 ECDC, ~1.1M | Staging payloads, including Node/V8 WASM q8 full-track ECDC. Not the browser WebGPU result set. |
| `target/wasm-matrix/` | 2 WAV, 1 ECDC, ~1.5M | Short Node/V8 WASM smoke output. Not the full WebGPU matrix. |

## Removed Older Folders

The previous raw/acv compatibility, smoke, and benchmark target folders were
removed during cleanup. The remaining matrix-like target folders are only:

- `target/gpu-matrix/`
- `target/webgpu-matrix/`
- `target/q8-matrix/`
- `target/env-matrix/`
- `target/wasm-matrix/`

## Cleanup

Do not delete the latest-run folders until manual checking is complete:

- `target/gpu-matrix/`
- `target/webgpu-matrix/`
- `target/q8-matrix/`
- `target/matrix-wav-manifest.txt`

No older target matrix folders are expected to remain after cleanup.
