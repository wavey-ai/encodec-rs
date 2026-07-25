# EnCodec-RS Qualification Matrix

## Purpose

This document defines the release qualification matrix for the Wavey archival profile.

The matrix tests deterministic code recovery, durable decoding, compression, quality, and runtime performance.

The matrix applies only to a release candidate that uses the pinned Wavey model artifacts.

The matrix does not authorize a cloud run.

## Terms

**Release candidate** means one Git commit and one locked set of build artifacts.

**Canonical input** means a source file converted to stereo, 48 kHz, planar `f32le` PCM.

**Canonical decoder** means the macOS ARM64 ONNX CPU decoder for the release candidate.

**Code tensor** means EnCodec indices in `[segment, codebook, time]` order.

**Fixed-code vector** means a code tensor and scale that do not use neural encoding.

**Strict decode** means a decode that stops at the first invalid field or segment.

**Salvage decode** means a decode that reports damage and conceals only declared ranges.

**Mandatory environment** means an environment that contributes to the release gate.

**Conditional environment** means an environment that needs separate owner authorization.

**Qualification driver** means the versioned test program shown as `<QUALIFY_DRIVER>`.

## Two-stage process

Geometry selection and release qualification are separate test stages.

Stage A compares three geometry candidates at 6 kbps and 12 kbps.

Stage B qualifies only the geometry that the owner selects for Profile 1.

Stage A results MUST not be presented as release qualification results.

This matrix does not test backward-compatible decoding.

## Stage A: geometry-selection matrix

### Geometry candidates

Compare these candidates on one locked reference implementation.

| Geometry ID | Model samples | Stride | Full code frames | Final tail | Reconstruction |
|---|---:|---:|---:|---|---|
| `meta-1000` | 48,000 | 47,520 | 150 | True variable tail | Triangle overlap |
| `fixed-1333` | 64,960 | 64,000 | 203 | Full fixed window | Crop, concatenate, and repair |
| `fixed-1800` | 87,360 | 86,400 | 273 | Full fixed window | Crop, concatenate, and repair |

Test each geometry at 6 kbps and 12 kbps.

This requirement creates these six Stage A rows:

| Selection row | Geometry | Rate | Codebooks |
|---|---|---:|---:|
| `select-meta-1000-6` | `meta-1000` | 6 kbps | 4 |
| `select-meta-1000-12` | `meta-1000` | 12 kbps | 8 |
| `select-fixed-1333-6` | `fixed-1333` | 6 kbps | 4 |
| `select-fixed-1333-12` | `fixed-1333` | 12 kbps | 8 |
| `select-fixed-1800-6` | `fixed-1800` | 6 kbps | 4 |
| `select-fixed-1800-12` | `fixed-1800` | 12 kbps | 8 |

### Official Meta geometry

`meta-1000` uses one 48,000-sample model segment.

Its segment stride is 47,520 samples.

Its adjacent decoded segments use triangle overlap.

Its final segment uses its true source length.

The neural encoder MUST receive that true final length.

The final code-frame count MUST come from the encoder output.

The entropy writer MUST encode only those returned code frames.

The writer MUST not add code-zero frames.

The decoder MUST reconstruct the true final segment before output trimming.

### Fixed geometry candidates

Both fixed candidates use 480 source-context samples on each side.

The first window uses zero samples only before the source start.

Each middle window uses real source context on both sides.

The final window uses zeros only after the source end.

Each fixed final window uses the full declared code-frame count.

The decoder crops 480 samples from each complete decoded model window.

The decoder then places each owned range at its declared source start.

The current fixed postprocessor applies the locked seam-repair profile.

The Stage A lock MUST identify that seam algorithm and all its parameters.

### Stage A reference environment

Run Stage A on `mac-arm64-onnx-cpu`.

Use one Git commit, model family, LM, preprocessing path, and metric tool set.

Use the complete CONFIRMATION corpus.

Use all regression supplements and generated vectors.

Run three measured repetitions after one warm-up run.

Stage A MUST record every artifact that Stage B records.

Stage A MUST use a separate output root:

```text
target/geometry-selection/<selection-run-id>/
```

Use this command contract:

```sh
<QUALIFY_DRIVER> geometry-select \
  --selection-lock "<selection-lock.json>" \
  --corpus-manifest "<corpus-manifest.json>" \
  --geometries "meta-1000,fixed-1333,fixed-1800" \
  --rates "6,12" \
  --environment "mac-arm64-onnx-cpu" \
  --output "target/geometry-selection/<selection-run-id>"
```

### Stage A comparisons

Compare these values for each corpus item and rate:

- Complete decodable object bytes
- Entropy payload bytes
- Container and recovery bytes
- Actual bitrate
- Quality metrics
- Seam metrics
- Final-tail correctness
- Encode and decode time
- Peak memory
- Manual listening results

The geometry comparison MUST use identical source bytes.

The geometry comparison MUST use the same metric reference PCM.

The comparison MUST not reuse decoded PCM from another geometry.

The comparison MUST report every failure without substituting another result.

### Geometry selection record

Write the decision to `geometry-selection.json`.

The record MUST contain these values:

- Selection run identifier
- Selected geometry ID
- Owner approval time
- Owner approval identifier
- 6 kbps result digest
- 12 kbps result digest
- Complete size ledger digest
- Quality summary digest
- Listening sheet digest
- Selection reason
- Rejected candidate reasons

The owner MUST select one geometry for both Profile 1 rates.

Do not combine geometry rules from different candidates.

Stop Stage B if `geometry-selection.json` is absent.

Stop Stage B if its digest is not in the release candidate lock.

## Stage B: final release qualification

Stage B tests only the selected Profile 1 geometry.

The release matrix contains these two profile rows:

| Profile ID | Geometry | Rate | Codebooks |
|---|---|---:|---:|
| `profile1-6` | Value from `geometry-selection.json` | 6 kbps | 4 |
| `profile1-12` | Value from `geometry-selection.json` | 12 kbps | 8 |

The release lock MUST contain the selected geometry fields.

The release lock MUST contain the selected tail rule.

The release lock MUST contain the selected reconstruction rule.

Every Stage B test MUST reject a different geometry identifier.

## Release candidate lock

Create `qualification-lock.json` before any matrix row starts.

The lock MUST contain these values:

- Git commit
- Cargo package version
- `Cargo.lock` SHA-256
- Rust toolchain and target
- ONNX Runtime version
- `gpu-worker-ort` version
- Browser runtime version
- Browser user agent
- WASM toolchain version
- Each bundle file SHA-256
- Each executable or WASM file SHA-256
- Preprocessing profile identifier
- Postprocessing profile identifier
- Bitstream profile identifier
- Entropy profile identifier
- `geometry-selection.json` SHA-256
- Selected geometry identifier
- Selected model samples and stride
- Selected final-tail rule
- Selected reconstruction rule
- Current fixed-size baseline lock SHA-256

The lock MUST name these files for each bundle:

- `bundle.json`
- `encode_frame.onnx`
- `decode_frame.onnx`
- `lm_weights_q8.bin`

The lock MUST name these browser files:

- `dist/wasm-fixed-bundles/manifest.json`
- `dist/wasm-fixed-bundles/encodec-ecdc-runtime.js`
- `dist/wasm-fixed-bundles/pkg/encodec_rs.js`
- `dist/wasm-fixed-bundles/pkg/encodec_rs_bg.wasm`

Create one sorted SHA-256 manifest for each bundle directory.

Use UTF-8 paths, lowercase hexadecimal digests, two spaces, and LF line endings.

Define `bundle_manifest_sha256` as the SHA-256 digest of those manifest bytes.

Stop if the worktree has uncommitted code or model changes.

Stop if an artifact changes after the lock is written.

Use this command shape to create the file manifest:

```sh
export QUAL_RUN_ID="<UTC-date>-<candidate-commit>"
export QUAL_OUT="$PWD/target/qualification/$QUAL_RUN_ID"
export PROFILE1_BUNDLE_6="<selected-6-kbps-bundle-directory>"
export PROFILE1_BUNDLE_12="<selected-12-kbps-bundle-directory>"
mkdir -p "$QUAL_OUT/manifests"

find \
  Cargo.lock \
  Cargo.toml \
  "$PROFILE1_BUNDLE_6" \
  "$PROFILE1_BUNDLE_12" \
  dist/wasm-fixed-bundles \
  -type f -print0 \
  | sort -z \
  | xargs -0 shasum -a 256 \
  > "$QUAL_OUT/manifests/release-files.sha256"
```

The operator MUST replace each angle-bracket placeholder before a release run.

## CONFIRMATION corpus

Set `CONFIRMATION_CORPUS_ROOT` to the directory that contains the approved files.

The test driver MUST use only the following explicit mappings.

| Corpus ID | Required relative path | Class | Source SHA-256 |
|---|---|---|---|
| `confirmation-i-want-her-v1` | `I WANT HER_MIX 4 CONFIRMATION_130323.wav` | Original | `4e984cdae6e57fc85b3fd316bd14edd5700de07ee8c5e94abba783e31bde53d4` |
| `confirmation-pray-4-me-v1` | `PRAY 4 ME_MIX 4 CONFIRMATION_130323.wav` | Original | `742d36fd78b1263f0c91f76127f3ea77e4cf7ce781c37715c2a7c890549ecf59` |
| `confirmation-westside-v1` | `WESTSIDE_MIX 4 CONFIRMATION_130323.wav` | Original | `dec1b383d58aea9848728126efab169f42e54375b64ca55363d2f234696474c9` |
| `confirmation-westside-v2` | `WESTSIDE_V2_MIX 4 CONFIRMATION_140323.wav` | Original | `15518ce0885a9b0ed0ce2b32410e79da74dfcf1914bda002d1bbe88b0d947f6b` |
| `confirmation-after-dark-v1` | `AFTER DARK_MIX 4 CONFIRMATION_130323.wav` | Original | `1812df5e6062549a4947a78b5bb08474b6d11026292c019c2da9b5b1950bc655` |
| `confirmation-1979-v1` | `Lori Asha - Lori Asha Album Premix/1979_MIX 4 CONFIRMATION_130323.wav` | Premix | `ed1db8e4a35679be511b12be32fcd0945582f9623a088247431862a884311679` |

Each approved source is stereo, 48 kHz, and 24-bit PCM.

The driver MUST reject every file whose name starts with `eMastered`.

The driver MUST reject every path that contains an `eMastered` component.

The driver MUST not select a source by a partial title match.

Stop if a required file is absent.

Stop if a source digest differs from this table.

Stop if two corpus IDs resolve to the same source file.

## Regression supplements

Regression supplements do not replace a CONFIRMATION corpus row.

Use these files when their digests match.

| Corpus ID | Path below `STRESS_CORPUS_ROOT` | Source SHA-256 | Purpose |
|---|---|---|---|
| `stress-baroness` | `regression/Baroness.wav` | `6b4c1af6806b9655730dcc8eb49f6e44e54cf10ad660c4e6b1d687adc5e00d23` | Long mono music |
| `stress-mono` | `regression/anshul_bogdan_mono.wav` | `ac68cf421dbda5f6b19150c55036ea0ec7ee2c49d68d798559b375d625065469` | Short 16 kHz mono |
| `stress-real-silence` | `real-silent-vinyl-record-groove.wav` | `3c100b8cfddc7864cf9c2d2abb30fec585ac13f5ae6b030b681c22c5f87f3dc5` | Long low-level stereo audio |
| `stress-speech` | `utterance_01K0SV3HH7BJKEVPBT429FGQHD.wav` | `93ac90162158fad2e4bf8fba555432e28fda711a1ccf47aa9c3c3eef820bb252` | Short mono speech |

The preparation driver MUST duplicate mono inputs into two equal channels.

Generate a ten-second digital-silence vector with exact zero samples.

Generate a ten-second impulse vector with signed impulses at declared sample positions.

Use these impulse positions:

```text
0, 1, 318, 319, 320, 479, 480, 481,
47518, 47519, 47520, 47521,
47998, 47999, 48000, 48001,
63998, 63999, 64000, 64001,
86398, 86399, 86400, 86401,
last-1, last
```

Alternate impulse amplitudes between `0.99` and `-0.99`.

Generate separate tail vectors with these owned sample counts:

```text
1, 319, 320, 479, 480, 481,
47519, 47520, 47521, 47999, 48000, 48001,
63999, 64000, 64001, 64959, 64960, 64961,
86399, 86400, 86401, 87359, 87360, 87361
```

The generator MUST use a fixed implementation and a versioned generator hash.

## Canonical input preparation

Keep the source file unchanged.

Convert each source to stereo, 48 kHz, `pcm_f32le` WAV.

Use one pinned FFmpeg build for all preparation.

Use this command shape:

```sh
ffmpeg \
  -nostdin \
  -hide_banner \
  -loglevel error \
  -bitexact \
  -i "<source.wav>" \
  -map 0:a:0 \
  -map_metadata -1 \
  -ar 48000 \
  -ac 2 \
  -c:a pcm_f32le \
  -flags:a +bitexact \
  -n \
  "<canonical.wav>"
```

Record the FFmpeg binary SHA-256 and version output.

Record the source and canonical WAV SHA-256 values.

Extract planar `f32le` PCM with the qualification driver.

Record the planar PCM SHA-256 value.

Record sample rate, channel count, sample count, duration, and sample format.

Stop if preparation changes the duration by more than one output sample.

Stop if the canonical input contains `NaN` or infinity.

## Required environments

### macOS ARM64

These rows are mandatory on the local Apple Silicon host:

| Environment ID | Frame inference | Entropy runtime |
|---|---|---|
| `mac-arm64-onnx-cpu` | ONNX Runtime CPU | Native Rust |
| `mac-arm64-coreml-cpu` | Core ML CPU only | Native Rust |
| `mac-arm64-coreml-gpu` | Core ML CPU and GPU | Native Rust |
| `mac-arm64-chromium-ort-wasm` | ONNX Runtime WebAssembly | `wasm32` Rust |
| `mac-arm64-chromium-webgpu` | ONNX Runtime WebGPU | `wasm32` Rust |

Record the macOS version and hardware model.

Record the CPU, GPU, and memory configuration.

Record Core ML compute-unit options for each row.

Record the full Chromium user agent and executable SHA-256.

### GCP Linux x86-64

These GCP rows are mandatory:

| Environment ID | Required capability | Frame inference | Entropy runtime |
|---|---|---|---|
| `gcp-linux-x86_64-cpu` | x86-64 CPU | ONNX Runtime CPU | Native Rust |
| `gcp-linux-x86_64-cuda-fp32` | NVIDIA GPU | ONNX Runtime CUDA FP32 | Native Rust |

Add `gcp-linux-x86_64-tensorrt-fp16` only if the release exposes that mode.

Record the GCP project, zone, instance type, image, and boot-disk image digest.

Record the CPU model, GPU model, driver, CUDA, and TensorRT versions.

Record whether the instance is preemptible.

Use these placeholders for remote execution:

```sh
export GCP_PROJECT="<approved-project>"
export GCP_ZONE="<approved-zone>"
export GCP_CPU_INSTANCE="<existing-or-approved-cpu-instance>"
export GCP_GPU_INSTANCE="<existing-or-approved-gpu-instance>"

gcloud compute ssh "$GCP_CPU_INSTANCE" \
  --project "$GCP_PROJECT" \
  --zone "$GCP_ZONE" \
  --command '<PINNED_CPU_QUALIFICATION_COMMAND>'

gcloud compute ssh "$GCP_GPU_INSTANCE" \
  --project "$GCP_PROJECT" \
  --zone "$GCP_ZONE" \
  --command '<PINNED_GPU_QUALIFICATION_COMMAND>'
```

Do not put credentials in result files.

Stop if the remote commit or release manifest differs from the candidate lock.

Stop if a preemption removes an unrecorded partial result.

### WASM and browser

Run entropy-only vectors in a `wasm32` runtime.

Run full browser rows in Chromium with ONNX Runtime WebAssembly and WebGPU.

The browser driver MUST use the selected Profile 1 segment builder.

The driver MUST hash each complete model input before inference.

The driver MUST apply the selected final-tail rule.

The driver MUST not replace final code positions with code zero.

Do not use the current browser matrix as a release oracle until these checks pass.

Stop if a browser model-input hash differs from the native model-input hash.

Stop if a browser code tensor has a different shape or element order.

### Conditional Azure rows

Azure rows require explicit owner authorization for Azure CLI use.

Do not run `az account show` before that authorization.

After authorization, add these rows:

| Environment ID | Required capability | Frame inference | Entropy runtime |
|---|---|---|---|
| `azure-linux-x86_64-cpu` | x86-64 CPU | ONNX Runtime CPU | Native Rust |
| `azure-linux-x86_64-cuda-fp32` | NVIDIA GPU | ONNX Runtime CUDA FP32 | Native Rust |

Use these placeholders only after authorization:

```sh
az account show
<AZURE_CPU_QUALIFICATION_COMMAND>
<AZURE_GPU_QUALIFICATION_COMMAND>
```

An unauthorized Azure row has status `conditional_not_authorized`.

That status does not qualify an Azure claim.

## Preflight commands

Run these local checks before any corpus encode:

```sh
git status --short
git rev-parse HEAD
rustc --version --verbose
cargo --version
cargo metadata --locked --no-deps --format-version 1

<PROFILE1_BUNDLE_CHECK_COMMAND>
<PROFILE1_WASM_CHECK_COMMAND>

cargo test --locked --features ecdc
cargo test --locked --features wasm
cargo check --locked --features onnx
cargo check \
  --locked \
  --target wasm32-unknown-unknown \
  --features wasm
```

The run MUST record stdout, stderr, exit status, and elapsed time.

Stop if any preflight command fails.

## Native command templates

Use the locked release binary when the driver is available.

These Cargo commands are temporary command templates.

### CPU encode

```sh
cargo run --locked --release --features onnx -- \
  onnx-encode \
  "<bundle-directory>" \
  "<canonical-input.wav>" \
  "<output.ecdc>" \
  --batch-size 8
```

### CPU decode

```sh
cargo run --locked --release --features onnx -- \
  onnx-decode \
  "<bundle-directory>" \
  "<input.ecdc>" \
  "<output.wav>"
```

### Core ML encode

```sh
cargo run --locked --release --features onnx -- \
  onnx-encode \
  "<bundle-directory>" \
  "<canonical-input.wav>" \
  "<output.ecdc>" \
  --batch-size 8 \
  --coreml \
  --coreml-compute-units "<cpu-only-or-cpu-and-gpu>" \
  --coreml-cache-dir "<run-cache-directory>"
```

### Core ML decode

```sh
cargo run --locked --release --features onnx -- \
  onnx-decode \
  "<bundle-directory>" \
  "<input.ecdc>" \
  "<output.wav>" \
  --coreml \
  --coreml-compute-units "<cpu-only-or-cpu-and-gpu>" \
  --coreml-cache-dir "<run-cache-directory>"
```

### CUDA encode and decode

Add `--cuda --device-id <device-id>` to the native command.

### LM probe

```sh
cargo run --locked --release --features onnx -- \
  onnx-lm-probe \
  "<bundle-directory>" \
  --steps "<step-count>"
```

The LM probe is diagnostic evidence.

Fixed-code vectors remain the entropy conformance oracle.

## Test artifact layout

Store every result below one immutable run directory.

```text
target/qualification/<run-id>/
  qualification-lock.json
  geometry-selection.json
  current-fixed-baseline-lock.json
  corpus-manifest.json
  manifests/
    release-files.sha256
    inputs.sha256
    outputs.sha256
  environments/
    <environment-id>.json
  inputs/
    source/
    canonical-wav/
    canonical-planar-f32le/
  vectors/
    entropy/
    tails/
    impulses/
    silence/
  model-inputs/
    <corpus-id>/<profile-id>/<segment>.f32le
  codes/
    <corpus-id>/<profile-id>/<encoder-id>.u16le
  scales/
    <corpus-id>/<profile-id>/<encoder-id>.f32le
  encodes/
    <corpus-id>/<profile-id>/<encoder-id>.ecdc
  decodes/
    <corpus-id>/<profile-id>/<encoder-id>/<decoder-id>.f32le
    <corpus-id>/<profile-id>/<encoder-id>/<decoder-id>.wav
  mutations/
    <mutation-id>/
  metrics/
    results.jsonl
    summary.json
    gates.json
    size-ledger.json
  logs/
    <environment-id>/<case-id>.log
```

Use little-endian `u16` for standalone code artifacts.

Store code tensors in `[segment, codebook, time]` order.

Use little-endian `f32` for standalone scale and PCM artifacts.

Store decoded PCM in planar `[channel, sample]` order.

Record a SHA-256 digest and byte count for every artifact.

## Fixed-code entropy matrix

Entropy tests MUST not call a neural encoder or decoder.

Test both four-codebook and eight-codebook vectors.

Test these time lengths:

```text
1, 2, 149, 150, 151,
selected_full_code_frames - 1,
selected_full_code_frames,
selected_full_code_frames + 1
```

Remove duplicate lengths before execution.

Do not add an unselected geometry boundary to Stage B.

Test these symbol patterns:

- All zero
- All maximum
- Alternating zero and maximum
- Codebook index ramp
- Time index ramp
- Seeded pseudorandom
- Long repeated run
- Single-symbol change at each LM reset boundary

Each vector MUST define these artifacts:

- Initial LM inputs
- Input code tensor
- Scale bytes
- Quantized logits
- Each integer CDF
- Arithmetic symbol order
- Encoded chunk bytes
- Recovered code tensor
- Final arithmetic state

Run each vector on every mandatory native and WASM environment.

All CDF bytes MUST match exactly.

All encoded chunk bytes MUST match exactly.

All recovered code tensors MUST match exactly.

Any mismatch fails the release candidate.

## Neural input and encode matrix

Run all six CONFIRMATION tracks through `profile1-6` and `profile1-12`.

Run all regression supplements through both Profile 1 rates.

Run all tail, impulse, and digital-silence vectors through both Profile 1 rates.

Run each case on every mandatory encoder environment.

Record the SHA-256 digest of every model window.

Model-window digests MUST match across environments.

Record each code tensor and scale before entropy coding.

Code tensor shape and order MUST match the profile.

Every final segment MUST follow the selected Profile 1 tail rule.

A fixed geometry MUST keep its complete fixed code-frame count.

A variable geometry MUST keep its exact returned code-frame count.

Neither geometry may synthesize code-zero tail frames.

The driver MUST report each cross-environment code difference.

Different neural codes do not fail durable decoding by themselves.

Different neural codes fail a deterministic-encoding claim.

The driver MUST classify every code difference before the run continues.

Scale comparisons MUST report exact bytes and ULP distance.

An unexplained scale difference above one `f32` ULP stops the run.

## Cross-decoder matrix

Decode every generated ECDC object on every mandatory decoder environment.

This requirement creates a complete encoder-by-decoder cross product.

The decoder MUST recover the encoder code tensor exactly.

The decoder MUST verify all asset and profile digests before neural decoding.

The decoded sample count MUST equal the canonical input sample count.

The decoded channel count MUST equal two.

The decoded PCM MUST not contain `NaN` or infinity.

Use the canonical decoder PCM as the cross-runtime reference.

Apply one shared PCM conversion after the float comparison.

Do not use runtime-specific WAV rounding as the comparison oracle.

## Cross-architecture invariants

The following values MUST match exactly:

- Release candidate lock
- Geometry-selection record
- Selected geometry identifier
- Source file SHA-256
- Canonical planar PCM SHA-256
- Model-window SHA-256
- Fixed-code CDF sequence SHA-256
- Fixed-code entropy bytes SHA-256
- Recovered code tensor SHA-256
- Segment count
- Segment ordinals
- Source start and sample count
- Code-frame count
- Final-tail and reconstruction identifiers
- Model, LM, preprocessing, and postprocessing identifiers

Neural encoder codes SHOULD match exactly.

Neural normalization scales SHOULD match exactly.

If codes differ, the result MUST identify every changed index.

If scales differ, the result MUST identify every changed segment and ULP distance.

Decoded float PCM uses the tolerances in the gate table.

## Quality metrics

Measure quality against the canonical input.

Align decoded output only by the delay declared in the profile.

Do not search for a better alignment during qualification.

Record these objective metrics:

- Scale-invariant signal-to-distortion ratio in dB
- ViSQOL Audio score
- Multi-resolution STFT distance
- Log-spectral distance in dB
- Integrated loudness difference in LU
- True-peak difference in dBTP
- Left and right channel correlation
- Inter-channel balance change in dB
- Maximum seam residual
- RMS seam residual
- Clipped sample count
- `NaN` and infinity count

Measure seam residuals across the complete selected reconstruction support.

Include 24 unchanged samples on each side of that support.

Also measure a matched set of non-seam windows.

Listen to both Westside versions and all other CONFIRMATION tracks.

The owner MUST approve the manual listening sheet.

The listening sheet MUST include clicks, image stability, bass, transients, vocals, and tails.

## Compression and strict size gate

### Complete byte ledger

Create `size-ledger.json` for every encoded object.

Use this command contract:

```sh
<QUALIFY_DRIVER> size-ledger \
  --object "<encoded-object>" \
  --required-sidecars "<comma-separated-paths-or-empty>" \
  --output "<size-ledger.json>"
```

The ledger MUST account for these disjoint byte classes:

- File magic and version
- Metadata header
- Profile descriptor
- Index
- Segment headers
- Normalization scales
- Arithmetic payload
- CRC fields
- Recovery or parity data
- Trailer
- Required geometry-specific sidecars

The object classes MUST sum to the exact file byte count.

Record a zero value when a class is absent.

Record geometry-specific sidecars at their complete byte count.

Do not move required bytes outside the object to reduce the reported size.

Exclude only release assets that all compared geometries share.

Shared release assets include pinned weights, graphs, and decoder source.

Report the shared release asset bytes separately.

Use these formulas:

```text
object_accounted_bytes =
  magic_version_bytes
  + metadata_header_bytes
  + profile_descriptor_bytes
  + index_bytes
  + segment_header_bytes
  + normalization_scale_bytes
  + arithmetic_payload_bytes
  + crc_bytes
  + recovery_bytes
  + trailer_bytes

comparable_bytes =
  object_accounted_bytes
  + required_geometry_sidecar_bytes

bitrate_bps =
  8 * comparable_bytes / source_duration_seconds

overhead_ratio =
  (
    metadata_header_bytes
    + profile_descriptor_bytes
    + index_bytes
    + segment_header_bytes
    + crc_bytes
    + recovery_bytes
    + trailer_bytes
    + required_geometry_sidecar_bytes
  )
  / comparable_bytes
```

The ledger MUST satisfy `object_accounted_bytes == file_size_bytes`.

The ledger MUST identify the SHA-256 digest for each counted object.

### Current fixed baseline

Freeze the current fixed baseline before Profile 1 release testing.

Store its identity in `current-fixed-baseline-lock.json`.

The baseline lock MUST contain these values:

- Baseline Git commit
- Baseline format identifier
- Corpus manifest digest
- Model and LM digests
- Preprocessing and postprocessing identifiers
- Fixed 1,333 ms object digests
- Fixed 1,800 ms object digests
- Complete size-ledger digest

Generate both current fixed objects for every CONFIRMATION source and rate.

Each baseline object MUST pass strict decode before size comparison.

Use this command contract:

```sh
<QUALIFY_DRIVER> fixed-size-baseline \
  --baseline-lock "<current-fixed-baseline-lock.json>" \
  --corpus-manifest "<corpus-manifest.json>" \
  --geometries "fixed-1333,fixed-1800" \
  --rates "6,12" \
  --output "<fixed-baseline-output-directory>"
```

Use the smaller valid fixed object as the per-case baseline.

Use this definition:

```text
best_fixed_baseline_bytes(corpus_id, rate) =
  min(
    fixed_1333_comparable_bytes,
    fixed_1800_comparable_bytes
  )
```

Block the size gate if either candidate has no complete byte ledger.

Block the size gate if both fixed candidates fail strict decode.

### No-size-regression gate

Compare one selected Profile 1 object with its per-case fixed baseline.

Use identical canonical input bytes for both objects.

Compare the complete `comparable_bytes` value without rounding.

For each CONFIRMATION source and rate, require this result:

```text
profile1_comparable_bytes <= best_fixed_baseline_bytes
```

Also require the corpus sum at each rate:

```text
sum(profile1_comparable_bytes)
  <= sum(best_fixed_baseline_bytes)
```

The allowed size regression is zero bytes.

Report every byte-class increase and decrease.

Do not average a failing track with a passing track.

Regression supplements and generated vectors MUST report the same ledger.

Their size results do not replace a required CONFIRMATION result.

### Additional compression metrics

Record these values for every encoded object:

- Source duration
- Complete comparable bytes
- Bits per second
- Bits per sample
- Compression ratio against canonical PCM
- Compression ratio against fixed-width code packing
- Bytes per segment
- Final-segment bytes

Report `profile1-6` and `profile1-12` separately.

## Performance metrics

Warm each runtime once before a measured run.

Run three measured repetitions for each full-track case.

Record median and p95 values.

Record these measurements:

- Model load time
- First-segment latency
- Encode wall time
- Decode wall time
- Entropy encode time
- Entropy decode time
- Real-time factor
- Peak resident memory
- Peak GPU memory
- Output bytes per second
- Segments per second
- Browser main-thread blocking time

Use this real-time factor:

```text
real_time_factor = wall_time_seconds / source_duration_seconds
```

Keep thermal and power conditions in the environment record.

Do not compare a cold run with a warm baseline.

## Mutation and recovery matrix

Create each mutation from one locked, valid ECDC object.

Store the parent object digest and mutation recipe.

Test these header mutations:

- Flip one magic byte.
- Change the file version.
- Increase the header length by one.
- Decrease the header length by one.
- Remove one required profile identifier.
- Replace the model digest.
- Replace the LM digest.
- Replace the preprocessing identifier.
- Replace the postprocessing identifier.
- Replace the geometry identifier.
- Replace the final-tail rule.
- Replace the reconstruction rule.
- Replace the total segment count.

Test these segment mutations:

- Flip one payload bit.
- Flip one segment-header bit.
- Replace one segment length.
- Remove one segment.
- Duplicate one segment.
- Reorder two segments.
- Replace one segment ordinal.
- Replace one source start.
- Replace one source sample count.
- Replace one code-frame count.
- Corrupt one segment CRC.
- Truncate one segment payload.

Test these file mutations:

- Truncate after each header byte.
- Truncate before the first segment.
- Truncate inside a middle segment.
- Truncate inside the final segment.
- Append random bytes.
- Remove the index.
- Corrupt one index entry.
- Point one index entry outside the object.

Run every mutation in strict and salvage modes.

Strict decode MUST reject every damaged object.

Strict decode MUST not label partial output as canonical.

Salvage decode MUST report every damaged sample range.

Salvage decode MUST preserve every valid segment exactly.

Salvage output MUST keep the declared total sample count.

Salvage output MUST identify its concealment method.

No mutation may cause an out-of-bounds access, panic, or unbounded allocation.

## Pass and fail gates

Use these gates for the first qualification run.

| Gate | Requirement |
|---|---|
| `G00-selection` | Stage B uses only the approved Profile 1 geometry. |
| `G01-input` | Every required source path and SHA-256 value matches. |
| `G02-lock` | Every runtime and artifact matches `qualification-lock.json`. |
| `G03-preflight` | All build, test, and check commands pass. |
| `G04-model-input` | All mandatory environments produce exact model-window bytes. |
| `G05-entropy` | All fixed-code CDFs, bytes, and recovered codes match exactly. |
| `G06-tail` | Every tail case follows the selected tail rule without synthetic code frames. |
| `G07-cross-decode` | Every decoder recovers the encoder codes exactly. |
| `G08-length` | Every decoded output has the exact declared sample count. |
| `G09-cpu-pcm` | Native CPU and Core ML CPU PCM have RMS error at most `0.05` s16 units and maximum error `1`. |
| `G10-accelerator-pcm` | GPU and WebGPU PCM have RMS error at most `4` s16 units and maximum error `32`. |
| `G11-browser-wasm-pcm` | Browser ORT WebAssembly PCM has RMS error at most `4` s16 units and maximum error `32`. |
| `G12-quality` | No metric exceeds its approved reference regression limit. |
| `G13-size` | Every CONFIRMATION object and rate is no larger than its best current fixed baseline. |
| `G14-rate` | Long-program bitrate is at most `105%` of the profile rate. |
| `G15-overhead` | Long-program container overhead is at most `0.5%`. |
| `G16-performance` | Median performance regresses by no more than `15%`. |
| `G17-memory` | Peak memory regresses by no more than `15%`. |
| `G18-mutation` | Strict and salvage results match every expected mutation result. |
| `G19-listening` | The owner approves the complete listening sheet. |
| `G20-artifacts` | Every required artifact and digest exists in the run directory. |

The approved quality reference MUST use the same corpus, profile, preprocessing, and metric versions.

The first accepted run establishes the performance and memory reference.

The owner MUST approve an initial quality reference before `G12-quality` can pass.

The owner MUST approve that first reference before release.

The release fails when one mandatory gate fails.

The release remains blocked when one mandatory gate has no result.

Conditional Azure rows do not affect non-Azure claims.

## Result record schema

Write one JSON object per case to `metrics/results.jsonl`.

Use this schema shape:

```json
{
  "schema_version": 1,
  "run_id": "<run-id>",
  "case_id": "<stable-case-id>",
  "status": "pass|fail|blocked|conditional_not_authorized",
  "phase": "geometry_selection|preflight|entropy|encode|decode|quality|mutation",
  "candidate": {
    "git_commit": "<40-hex>",
    "package_version": "<semver>",
    "lock_sha256": "<64-hex>",
    "geometry_selection_sha256": "<64-hex>",
    "fixed_baseline_lock_sha256": "<64-hex>"
  },
  "environment": {
    "id": "<environment-id>",
    "os": "<name-and-version>",
    "arch": "aarch64|x86_64|wasm32",
    "cpu": "<model>",
    "gpu": "<model-or-null>",
    "provider": "<cpu|coreml|cuda|webgpu|wasm>",
    "provider_version": "<version>",
    "browser_user_agent": "<value-or-null>"
  },
  "input": {
    "corpus_id": "<corpus-id>",
    "source_sha256": "<64-hex>",
    "canonical_pcm_sha256": "<64-hex>",
    "sample_rate": 48000,
    "channels": 2,
    "samples": 0
  },
  "profile": {
    "id": "<profile-id>",
    "geometry_id": "<selected-geometry-id>",
    "model_samples": 0,
    "stride_samples": 0,
    "full_code_frames": 0,
    "final_tail_rule": "<identifier>",
    "reconstruction_id": "<identifier>",
    "bundle_manifest_sha256": "<64-hex>",
    "bundle_json_sha256": "<64-hex>",
    "encoder_sha256": "<64-hex>",
    "decoder_sha256": "<64-hex>",
    "lm_sha256": "<64-hex>",
    "preprocess_id": "<identifier>",
    "postprocess_id": "<identifier>"
  },
  "artifacts": {
    "model_inputs_sha256": "<64-hex>",
    "codes_sha256": "<64-hex>",
    "scales_sha256": "<64-hex>",
    "ecdc_sha256": "<64-hex>",
    "decoded_f32_sha256": "<64-hex>",
    "decoded_s16_sha256": "<64-hex>",
    "size_ledger_sha256": "<64-hex>",
    "bytes": 0
  },
  "size": {
    "file_size_bytes": 0,
    "object_accounted_bytes": 0,
    "required_geometry_sidecar_bytes": 0,
    "comparable_bytes": 0,
    "best_fixed_baseline_bytes": 0,
    "size_delta_bytes": 0,
    "magic_version_bytes": 0,
    "metadata_header_bytes": 0,
    "profile_descriptor_bytes": 0,
    "index_bytes": 0,
    "segment_header_bytes": 0,
    "normalization_scale_bytes": 0,
    "arithmetic_payload_bytes": 0,
    "crc_bytes": 0,
    "recovery_bytes": 0,
    "trailer_bytes": 0
  },
  "metrics": {
    "bitrate_bps": 0.0,
    "compression_ratio": 0.0,
    "si_sdr_db": 0.0,
    "visqol": 0.0,
    "mrstft": 0.0,
    "lsd_db": 0.0,
    "loudness_delta_lu": 0.0,
    "true_peak_delta_dbtp": 0.0,
    "pcm_rms_s16": 0.0,
    "pcm_max_s16": 0,
    "encode_rtf": 0.0,
    "decode_rtf": 0.0,
    "peak_rss_bytes": 0,
    "peak_gpu_bytes": 0
  },
  "mutation": {
    "id": null,
    "strict_result": null,
    "salvage_result": null,
    "concealed_ranges": []
  },
  "gates": [
    {
      "id": "<gate-id>",
      "status": "pass|fail|blocked",
      "observed": "<value>",
      "limit": "<value>"
    }
  ],
  "timing": {
    "started_at": "<RFC-3339>",
    "elapsed_ms": 0
  },
  "log_path": "<relative-path>",
  "stop_reason": null
}
```

Use `null` only when the schema permits it.

Do not omit a failed or blocked measurement.

The summary MUST count pass, fail, blocked, and conditional results.

## Stop conditions

Stop the complete run for any of these conditions:

- A required CONFIRMATION source is absent.
- An eMastered file enters the source mapping.
- Stage B uses an unselected geometry.
- The geometry-selection record changes.
- A source or release artifact digest changes.
- The worktree changes during the run.
- A model window differs across runtimes.
- A fixed-code entropy byte differs.
- A decoder recovers a different code index.
- A final segment loses or replaces code positions.
- A CONFIRMATION object exceeds its best current fixed baseline.
- A size ledger does not equal the complete file size.
- A runtime uses an unrecorded model or LM file.
- A strict decoder accepts known damage.
- A decoder panics or allocates without a declared limit.
- Decoded PCM contains `NaN` or infinity.
- A mandatory result or log cannot be stored.

Stop one environment for any of these conditions:

- The environment does not match its recorded specification.
- A GPU driver resets.
- A cloud instance is preempted during an uncheckpointed case.
- The run exceeds its approved cost limit.
- The run exceeds its approved wall-time limit.

Stop Azure preparation until the owner authorizes Azure CLI use.

Do not convert a stopped case to a pass.

Record the stop condition in the result object.

## Release decision

Publish a qualification summary only after all mandatory gates pass.

Publish the release lock and output digest manifest with the summary.

Keep the complete run directory with the durable decoder artifact set.

State the exact environments that passed.

Do not make a claim for an environment that has no passing row.

The durable-decode claim requires exact code recovery on every mandatory decoder.

The deterministic-encoding claim requires exact neural codes and canonical scale bytes.

The salvage claim requires all mutation and recovery rows to pass.
