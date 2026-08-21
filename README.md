# encodec-rs

`encodec-rs` provides EnCodec frame execution, deterministic entropy coding, and a chunk-framed `.ecdc` container.

The current scope is 48 kHz stereo audio at 6 kbps or 12 kbps.

Native execution uses ONNX Runtime. Browser execution uses ONNX Runtime Web and Rust WASM.

Neither production path requires Python. Python supports only upstream comparison and quality analysis.

Live browser demo: [wavey.ai/code/encodec-rs/browser-smoke](https://wavey.ai/code/encodec-rs/browser-smoke/)

## Runtime boundary

| Function | Native path | Browser path |
|---|---|---|
| Neural frame encode and decode | ONNX Runtime | ONNX Runtime Web |
| Quantized language model | Rust q8 | Rust q8 WASM |
| Arithmetic coding | Rust | Rust WASM |
| `.ecdc` framing and CRC | Rust | Rust WASM |
| Fixed-chunk reconstruction | Rust | Rust WASM |

The q8 entropy path uses integer operations and bitstream version `acv=2`.

Older raw payloads and floating-point LM payloads are not supported.

## Supported bundles

Download the model bundles before you run ONNX or browser commands:

```bash
scripts/download-onnx-bundles.sh
```

The files come from [wavey-ai/encodec-rs-onnx-bundles](https://huggingface.co/wavey-ai/encodec-rs-onnx-bundles).

Each bundle contains these files:

- `bundle.json`
- `encode_frame.onnx`
- `decode_frame.onnx`
- `lm_weights_q8.bin`

The repository supports dynamic 1,000 ms bundles and these fixed browser bundles:

| Fixed bundle | Bitrate | Owned samples | Guarded model samples | LM steps | Codebooks |
|---|---:|---:|---:|---:|---:|
| `encodec_48khz_6kbps_1333ms` | 6 kbps | 64,000 | 64,960 | 203 | 4 |
| `encodec_48khz_12kbps_1333ms` | 12 kbps | 64,000 | 64,960 | 203 | 8 |
| `encodec_48khz_6kbps_1800ms` | 6 kbps | 86,400 | 87,360 | 273 | 4 |
| `encodec_48khz_12kbps_1800ms` | 12 kbps | 86,400 | 87,360 | 273 | 8 |

Each fixed model window contains 480 source samples of guard context on each side.

The owned region defines the output timeline. The decoder removes both guards after frame decoding.

## Fixed browser pipeline

The fixed browser encoder processes one owned chunk at a time.

1. Split the 48 kHz stereo source into nonoverlapping owned regions.
2. Add 480 real source samples on each available side.
3. Add zeros where file boundaries do not provide guard samples.
4. Run one guarded window through `encode_frame.onnx`.
5. Encode all fixed LM steps with the q8 Rust WASM path.
6. Store the independent entropy payload with its length and CRC32.

The final owned region can be short. Its model input and LM sequence keep the fixed graph length.

The decoder reverses these steps. It crops each guarded result to its owned region before concatenation.

The current decoder changes 24 samples at each join with `cubic-hermite-v1` repair.

This repair covers 12 samples before the join and 12 samples after the join.

## State, sessions, and batches

Codec state is data that changes the next encoded or decoded result.

The provided frame graphs do not keep codec state between ONNX `run()` calls.

Each call depends only on its input tensors. One session can process independent chunks or different tracks in any order.

ONNX Runtime keeps allocations, thread pools, graph optimizations, and caches.

This runtime state affects speed. It does not affect the encoded result in the tested configuration.

The q8 LM and arithmetic coder start with new state for every independent `encodec-rs` chunk.

A randomized test interleaved two complete tracks through one session. All 316 chunk hashes matched their isolated-session hashes.

The WASM encoder API accepts one fixed model window per call. It does not combine encoder windows into a tensor batch.

The fixed decoder benchmark can batch windows. Batch size two was 0.04% slower than batch size one.

Batch sizes eight and 32 exceeded available memory. They have no reported performance result.

Session reuse removes setup cost. It does not make chunks dependent on earlier calls.

## Difference from official Meta EnCodec

Official Meta EnCodec also divides long 48 kHz audio into neural segments.

Its CLI accepts one file, but its model processes 48,000-sample segments with a 47,520-sample stride.

The 480-sample difference creates a 10 ms overlap. Meta combines decoded segments with triangle-weighted overlap-add.

Official Meta creates one arithmetic coder and one LM state for each neural segment.

It does not reset the coder at each code timestep or codebook.

The word `frame` can mean several units. This README uses `neural segment` for Meta's one-second coding unit.

| Property | `encodec-rs` fixed profile | Official Meta 48 kHz profile |
|---|---|---|
| Long-file unit | 64,000 owned samples | 48,000-sample neural segment |
| Stride | 64,000 samples | 47,520 samples |
| Source context | 480 samples on each side | 480-sample adjacent overlap |
| Reconstruction | Crop, concatenate, then repair | Triangle overlap-add |
| Entropy reset | Each owned chunk | Each neural segment |
| Entropy arithmetic | Deterministic q8 integer path | Floating-point PyTorch LM path |
| Segment framing | Length and CRC32 | Inferred from expected symbols |

Meta ECDC version 0 does not store an encoded length or CRC for each neural segment.

Its decoder infers each boundary from the expected symbol count.

Floating-point CDF differences can change arithmetic bit consumption across architectures.

One changed boundary can make later segments unreadable because the container has no explicit resynchronization point.

`encodec-rs` contains each failure within one length-framed and CRC-protected chunk.

See the pinned official implementation at commit `0e2d0aed29362c8e8f52494baf3e6f99056b214f`:

- [model segmentation](https://github.com/facebookresearch/encodec/blob/0e2d0aed29362c8e8f52494baf3e6f99056b214f/encodec/model.py)
- [entropy compression](https://github.com/facebookresearch/encodec/blob/0e2d0aed29362c8e8f52494baf3e6f99056b214f/encodec/compress.py)
- [binary container](https://github.com/facebookresearch/encodec/blob/0e2d0aed29362c8e8f52494baf3e6f99056b214f/encodec/binary.py)
- [triangle overlap-add](https://github.com/facebookresearch/encodec/blob/0e2d0aed29362c8e8f52494baf3e6f99056b214f/encodec/utils.py)

## ECDC layout

One `.ecdc` file contains one header and one or more independent chunk payloads:

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

The metadata records the model, audio length, codebook count, LM mode, and q8 weight hash.

Fixed containers record the LM frame length through `fl`.

The selected fixed bundle supplies the guarded sample count and owned stride.

Do not concatenate complete `.ecdc` files. One complete file can already contain many framed chunks.

The current `acv=2` envelope is repository-specific. It is not the proposed Profile 1 container.

## Build and run

### Native

Build the native ONNX CLI:

```bash
cargo build --release --features onnx
```

Inspect a bundle:

```bash
target/release/encodec-rs onnx-inspect \
  onnx-bundles/encodec_48khz_12kbps
```

Encode a WAV file:

```bash
target/release/encodec-rs onnx-encode \
  onnx-bundles/encodec_48khz_12kbps \
  input.wav output.ecdc
```

Decode an ECDC file:

```bash
target/release/encodec-rs onnx-decode \
  onnx-bundles/encodec_48khz_12kbps \
  input.ecdc output.wav
```

The CLI accepts WAV input. The input must be 48 kHz stereo for the hosted bundles.

The CLI does not resample input.

Native execution supports CPU, CUDA, TensorRT, and CoreML targets.

Run `target/release/encodec-rs --help` for target and batch options.

### Browser WASM

Build the WASM package:

```bash
rustup target add wasm32-unknown-unknown
cargo check --lib --no-default-features --features wasm \
  --target wasm32-unknown-unknown
wasm-pack build --target web --no-default-features --features wasm
```

Run the browser test page:

```bash
npm install --prefix browser-smoke
python3 browser-smoke/serve.py
```

Open `http://127.0.0.1:8787/browser-smoke/`.

Run the scripted WebGPU matrix:

```bash
node scripts/webgpu-matrix.mjs
```

The matrix writes its results under `target/webgpu-matrix/`. See [MATRIX.md](MATRIX.md) for the tested browser paths.

An entropy-only implementation change requires a new `encodec_rs_bg.wasm` downstream.

Update the generated JavaScript and TypeScript bindings only when the exported WASM interface changes.

## Entropy optimization benchmark

The August 21, 2026 audit used an Apple M1 host and macOS 26.5.

Node was version 26.3.0. ONNX Runtime Web was version 1.26.0 with one WASM thread.

The test used the fixed 12 kbps, 1,333 ms bundle.

Higher RTFx is faster. `RTFx = audio duration / wall time`.

| Stage | Track A warm RTFx | Track B warm RTFx | A median LM | B median LM | Baseline hashes |
|---|---:|---:|---:|---:|---:|
| Scalar baseline | 1.583× | 1.569× | 515.547 ms | 514.621 ms | 316/316 |
| Exact handwritten WASM SIMD | 2.471× | 2.449× | 207.956 ms | 211.098 ms | 316/316 |
| Exact CDF selection | 2.651× | 2.532× | 186.162 ms | 191.117 ms | 316/316 |
| Reused CDF storage | 2.548× | 2.516× | 189.152 ms | 190.439 ms | 316/316 |
| Exact probability cache | 2.645× | 2.514× | 184.749 ms | 189.520 ms | 316/316 |
| Final clean audit | 2.652× | 2.628× | 183.708 ms | 184.515 ms | 316/316 |

The final pooled LM time was 64.334% lower than the scalar baseline.

Pooled total encoding time was 40.287% lower. Pooled ONNX time changed by 0.028%.

Handwritten SIMD produced the largest reduction. The smaller CDF changes added the remaining improvement.

Every stage encoded both complete tracks in isolated sessions and in one randomized interleave.

All 316 isolated chunk payloads matched the scalar baseline byte for byte at every stage.

All 316 interleaved payloads matched their isolated results. Both final tracks passed the 2× warm RTFx gate.

### Audit inputs

| Track | Source basename | Duration | Chunks | SHA-256 |
|---|---|---:|---:|---|
| A | `eMastered_1979_MIX-4-CONFIRMATION_130323_HD.wav` | 227.863 s | 171 | `771460cec7c8be31bb162f855f4eb3858e11e9b5d2015fe6e50f0ec4b5a6f865` |
| B | `eMastered_WESTSIDE_V2_MIX-4-CONFIRMATION_140323_HD.wav` | 192.936 s | 145 | `dd30787dc269100eaf0b5faf713401d84dd2f36b4da96fbe0de6380107ab885f` |

Both inputs are 48 kHz stereo PCM24 masters.

### Reproduce the stage audit

Build one stage:

```bash
scripts/build-entropy-wasm-stage.sh 05-final-audit
```

Profile both complete files:

```bash
node scripts/profile-encodec-wasm-session.mjs \
  --wasm-root target/performance/entropy-optimization/05-final-audit/wasm \
  --output target/performance/entropy-optimization/05-final-audit/profile.json \
  "$TRACK_A_WAV" "$TRACK_B_WAV"
```

Compare the final profile with the baseline:

```bash
node scripts/compare-encodec-wasm-profiles.mjs \
  target/performance/entropy-optimization/00-baseline/profile.json \
  target/performance/entropy-optimization/05-final-audit/profile.json
```

The profile report checkpoints after each chunk. It records source hashes and per-chunk output hashes.

## Full-file comparison with official Meta EnCodec

The full-file comparison used Track A at 12 kbps with LM entropy coding.

All rows used one CPU thread on the same Apple M1 host.

The `encodec-rs` row uses ONNX Runtime Web WASM and Rust WASM.

The Meta rows use native ARM64 PyTorch through the pinned official Python package.

| Path | Encode time | Encode RTFx | Decode time | Decode RTFx | ECDC bytes | Effective rate |
|---|---:|---:|---:|---:|---:|---:|
| `encodec-rs`, warm direct path | 86.932 s | 2.621× | 183.373 s | 1.243× | 296,562 | 10.412 kbps |
| Meta, loaded core API | 104.004 s | 2.191× | 104.171 s | 2.187× | 278,134 | 9.765 kbps |
| Meta, standard fresh CLI | 108.605 s | 2.098× | 106.209 s | 2.145× | 278,134 | 9.765 kbps |

The warm `encodec-rs` encoder had 19.6% higher throughput than the loaded Meta path.

Its encode wall time was 16.4% lower. Its payload was 6.63% larger.

The loaded Meta decoder had 1.76 times the throughput of the `encodec-rs` decoder.

The Meta core row excludes model setup. The standard CLI row includes process startup, model setup, and file input and output.

Both Meta rows reused cached model checkpoints. Their ECDC outputs were byte-identical.

The `encodec-rs` row excludes session setup from warm RTFx. It includes every JS-to-WASM call and memory copy in the measured path.

No FFI estimate was subtracted from any result.

This comparison measures complete implementations, not Python against Rust or one entropy algorithm in isolation.

## Full-file quality

The seamless PCM24 master is the reference for every result.

All decoded candidates matched the master length and had zero measured sample lag.

Higher SNR and SI-SDR are better. Lower log-spectral distance and spectral convergence are better.

| Candidate | SNR | SI-SDR | Mean segment SNR | Log-spectral distance | Spectral convergence | Loudness delta |
|---|---:|---:|---:|---:|---:|---:|
| `encodec-rs`, repaired | 6.579 dB | 5.706 dB | 7.011 dB | 12.460 dB | 0.27915 | -0.408 LU |
| `encodec-rs`, before repair | 6.594 dB | 5.723 dB | 7.015 dB | 12.524 dB | 0.27869 | -0.410 LU |
| Official Meta | 6.600 dB | 5.719 dB | 7.017 dB | 12.596 dB | 0.27893 | -0.434 LU |

Meta SNR exceeded repaired `encodec-rs` by 0.021 dB.

Repaired `encodec-rs` log-spectral distance was 0.136 dB lower than Meta.

These full-file metrics show no material aggregate quality difference between the two decoded files.

Entropy framing does not change decoded PCM when code recovery is exact.

Model geometry and reconstruction determine the measured waveform differences.

### ViSQOL

Official [ViSQOL](https://github.com/google/visqol/tree/38d0b0163e441047d4429bf07ad09e5b9031d02c) scored ten matched active excerpts.

Each excerpt was eight seconds. One shared gain protected the master and both candidates from PCM16 clipping.

| Candidate | Mean MOS-LQO | Median MOS-LQO | Standard deviation |
|---|---:|---:|---:|
| `encodec-rs` | 4.2874 | 4.2809 | 0.0663 |
| Official Meta | 4.2769 | 4.2847 | 0.0676 |

The paired mean difference was `+0.0105` for `encodec-rs`.

Its 95% confidence interval was `-0.0038` to `+0.0248`, so the result does not show a reliable winner.

[ViSQOL audio guidance](https://github.com/google/visqol/blob/38d0b0163e441047d4429bf07ad09e5b9031d02c/README.md) states that audio mode downmixes stereo to mono.

Its audio model was trained at rates of 24 kbps and higher.

These payloads are below 12 kbps. Treat ViSQOL as secondary evidence at this rate.

## Seam analysis

The master has no codec join. Each candidate join is compared with the same master samples.

The analysis uses a 20 ms window centered on each join.

Two nearby nonjoin windows provide an equal-duration control error.

`Seam excess error` compares join MSE with control MSE. Lower values are better.

`Seam SNR` compares master signal power with join error. Higher values are better.

`Step error` subtracts the master's own sample step from the decoded sample step.

| Reconstruction | Joins | Median excess | P90 excess | Worst excess | Median seam SNR | P10 seam SNR | Median step error |
|---|---:|---:|---:|---:|---:|---:|---:|
| `encodec-rs`, repaired | 170 | 0.587 dB | 3.929 dB | 5.996 dB | 5.169 dB | 0.817 dB | 0.02957 |
| `encodec-rs`, before repair | 170 | 0.476 dB | 3.011 dB | 5.082 dB | 5.245 dB | 1.581 dB | 0.08958 |
| Official Meta overlap-add | 230 | 0.019 dB | 2.099 dB | 5.792 dB | 6.118 dB | 2.791 dB | 0.02699 |

Meta's 10 ms overlap-add produced the best join distribution on this file.

The current cubic repair reduced median step error, but it did not improve master-reference fidelity.

The repair changed 8,160 sample values, or 0.0373% of the decoded stereo values.

It improved join-window fidelity at 46 joins and degraded it at 124 joins.

Median join fidelity changed by `-0.043 dB`. Median step error improved by `7.378 dB`.

The repair also increased output peak from `+2.552 dBFS` to `+5.190 dBFS`.

The largest repair effect occurred at 86.667 seconds.

It reduced local fidelity by 2.186 dB while improving step error by 2.628 dB.

![Waveforms and spectrograms for the largest encodec-rs repair effect](docs/benchmarks/encodec-full-file-20260821/encodec-rs-largest-repair-effect.png)

The next figure shows decoded-minus-master residual energy at the worst repaired join.

![Residual spectrograms at the worst repaired encodec-rs join](docs/benchmarks/encodec-full-file-20260821/encodec-rs-worst-join-residual.png)

The final figure uses the same scales for Meta's worst measured overlap-add join.

![Waveforms and spectrograms for the worst Meta overlap-add join](docs/benchmarks/encodec-full-file-20260821/meta-worst-overlap-join.png)

These results do not support the current cubic repair as the final reconstruction method.

A slope-limited and amplitude-clamped Hermite repair is a suitable next experiment.

A short crossfade against real overlapping context is another suitable experiment.

Python is useful for these sweeps and plots. The selected arithmetic can then move unchanged into Rust and WASM.

## Reproduce the upstream and quality comparison

Create the pinned Conda environment:

```bash
conda env create -f benchmark-environment.yml
conda activate encodec-upstream-benchmark
```

Benchmark the loaded official implementation:

```bash
python scripts/benchmark-meta-encodec.py \
  "$MASTER_WAV" \
  --output-root target/performance/upstream-comparison/full/meta
```

Benchmark the standard official CLI:

```bash
python scripts/benchmark-meta-encodec-cli.py \
  "$MASTER_WAV" \
  --encodec-cli "$CONDA_PREFIX/bin/encodec" \
  --output-root target/performance/upstream-comparison/full/meta-cli
```

Generate full-file metrics, excerpts, and spectrograms:

```bash
python scripts/compare-encodec-quality.py \
  "$MASTER_WAV" \
  --encodec-rs target/performance/upstream-comparison/full/encodec-rs-decoded.wav \
  --encodec-rs-pre-repair target/performance/upstream-comparison/full/encodec-rs-before-seam-repair.wav \
  --meta target/performance/upstream-comparison/full/meta/meta-decoded.wav \
  --output-root target/performance/upstream-comparison/full/quality \
  --figure-root docs/benchmarks/encodec-full-file-20260821
```

Build ViSQOL from its pinned source before you run the perceptual metric.

Then score the generated excerpts:

```bash
python scripts/benchmark-visqol.py \
  --visqol target/third-party/visqol/bazel-bin/visqol \
  --excerpt-root target/performance/upstream-comparison/full/quality/perceptual-excerpts \
  --output-root target/performance/upstream-comparison/full/quality/visqol
```

The scripts write JSON and CSV evidence under `target/performance/`. Git does not track that directory.

## Library use

Enable only container and entropy functions:

```toml
encodec-rs = { git = "https://github.com/wavey-ai/encodec-rs.git", features = ["ecdc"] }
```

Enable the native ONNX runtime:

```toml
encodec-rs = { git = "https://github.com/wavey-ai/encodec-rs.git", features = ["onnx"] }
```

The `ecdc::FrameCodec` trait separates neural frame execution from the container.

The `ecdc::LmCodec` and `portable_lm::PortableLmCodec` paths provide deterministic q8 entropy coding.

## Current limitations

- Hosted bundles support only 48 kHz stereo audio.
- The CLI does not resample input.
- The browser encoder processes one fixed window per call.
- The current cubic seam repair can overshoot and can reduce local fidelity.
- The Apple M1 WASM decoder is slower than the pinned Meta CPU decoder.
- Full-file timing results come from one host and one complete run per condition.
- ViSQOL evidence is outside its documented training bitrate range.
- The current `acv=2` container is not Profile 1.

## Tests

Run the Rust container and WASM tests:

```bash
cargo test --lib --features wasm,ecdc
```

Run native ONNX tests after you download the bundles:

```bash
cargo test --features onnx
```

See [docs/README.md](docs/README.md) for protocol, qualification, and implementation documents.
