# Architecture Matrix Plan

Scope: audio encode and decode, plus the code and entropy payload path. The approved six-file corpus stays unchanged.

| Environment | Architecture | Work | Status |
| --- | --- | --- | --- |
| macOS local | ARM64 Apple M1 CPU | Canonical CPU baseline for both candidates and both rates | Completed; current acv2 baseline |
| macOS Chromium WASM | ARM64 | Encode, decode, code order, and entropy parity | Blocked by current `acv=2` browser path |
| macOS Chromium WebGPU | ARM64 GPU | Same parity checks with WebGPU execution | Blocked by current `acv=2` browser path |
| GCP Linux | x86_64 CPU | Repeat the locked CPU cases | Blocked; package transfer failed before execution |
| GCP Linux CUDA | x86_64 NVIDIA GPU | FP32 encode and decode repeat | Blocked; project GPU quota is zero |

The codec input is audio. This run treats code and entropy as the text-like payload. A literal text benchmark needs a separate corpus and contract.

Each environment must use the same corpus manifest, model bundle hashes, vector hashes, codebook order, scale bits, entropy bytes, and salvage cases. Compare exact bytes first. Compare PCM and runtime second.

Do not compare GPU and CPU outputs when the model, precision, or container differs.
