# Rust Refactor Review

Status: Source review complete

Review date: 2026-07-25

## Result

The current Rust source has no detected RVQ-to-LM permutation error.

Native Rust and WebAssembly now use one entropy probability implementation.

The runtime now rejects incomplete LM weight and metadata combinations.

The native WAV path now reads the approved 24-bit CONFIRMATION sources.

The current `acv=2` container is not a Profile 1 release candidate.

Do not publish the current container as a durable archive format.

## Scope

This review examined these decode-critical areas:

- Neural frame input construction
- Neural frame output reconstruction
- RVQ codebook order
- LM embedding and output-head order
- BOS and next-input rules
- Entropy probability construction
- Arithmetic symbol order
- Chunk state reset behavior
- Fixed-context tail behavior
- Native and WebAssembly differences
- Bundle and weight validation
- PCM input and output conversion

This review did not qualify a release.

This review did not run the CONFIRMATION corpus matrix.

This review did not run GCP or Azure jobs.

## Confirmed RVQ and LM mapping

The neural encoder produces codes with shape `[batch, codebook, time]`.

RVQ layer `k` maps to LM embedding table `k`.

LM output head `k` predicts RVQ layer `k`.

The entropy loop traverses time before codebook.

The arithmetic symbol is the unchanged RVQ code value.

Only the next LM input adds one to the prior code value.

The first LM input contains zero for each codebook.

All codebooks at one time use the same prior-time input.

The implementation does not use delayed codebooks.

The implementation starts new LM and arithmetic state for each chunk.

The [lineage contract](encodec-lineage-and-lm-contract.md) gives the full mapping.

## High-value refactor changes

### Shared entropy implementation

The new [`entropy.rs`](../src/entropy.rs) module owns probability construction.

Native Rust and WebAssembly both call this module.

The module defines one flat logit layout and one probability column layout.

Probability columns use time-major, codebook-minor order.

The module validates temperature, logit step, and probability scale.

The module reuses scratch buffers across LM steps.

The module uses `libm::exp` in both targets.

This change removes two separate softmax implementations.

This change reduces a cross-runtime drift risk.

It does not prove cross-architecture entropy identity.

### Complete LM weight validation

[`QuantizedLmWeights::validate_for_metadata`](../src/quantized_lm.rs) now checks decode-critical fields.

It checks codebook count, cardinality, dimension, layer count, and past context.

It also checks that the bundle frame length fits the LM positional capacity.

Native ONNX, portable Rust, and WebAssembly constructors call this validation.

The runtime now requires the exact codebook count.

It no longer accepts an arbitrary subset of the loaded codebooks.

LM logit reshape failures now return errors.

They no longer use an unchecked `expect`.

### Canonical current entropy values

The current metadata validator now requires the canonical probability scale.

It also requires the canonical arithmetic minimum range.

It requires the binary32 LM temperature value `1.0`.

This rule removes a native and WebAssembly interpretation difference.

Profile 1 must bind these values through its profile root.

### Fixed model-code tails

Fixed graphs now require every model-code position.

The WebAssembly path rejects an incomplete fixed frame.

The path no longer inserts code value zero for missing tail positions.

The native path rejects a chunk that exceeds LM positional capacity.

It no longer inserts a hidden BOS reset inside a chunk.

### Input window and reconstruction corrections

Native and browser test paths now include the declared left and right context.

Fixed final windows use the complete fixed model shape.

The direct ONNX round-trip path now uses the fixed-context crop and seam rules.

These corrections make the local test oracle match the current product geometry.

They do not select the Profile 1 geometry.

### PCM conversion

The new [`pcm.rs`](../src/pcm.rs) module defines Rust PCM conversion.

Signed 16-bit input uses division by `32768`.

Signed 24-bit input uses division by `8388608`.

Signed 32-bit input uses division by `2147483648`.

Signed 16-bit output uses the full asymmetric integer range.

The updated browser fixtures use the same output rounding rule.

Non-finite Rust output samples become zero.

The Profile 1 manifest must bind this conversion or another selected conversion.

## Local evidence

The following source checks pass:

```text
cargo check --all-features
cargo test --features wasm --no-fail-fast
node --check browser-smoke/webgpu-matrix.js
node --check scripts/wasm-encode-fixture.mjs
node --check scripts/westside-chunk-wasm-roundtrip.mjs
```

The Rust test run passed 45 tests.

The tests include exact native-to-WebAssembly entropy parity.

The tests include BOS and prior-code-plus-one tracing.

The tests include fixed-tail rejection and LM capacity rejection.

The tests include context construction, reconstruction, seam, and PCM boundaries.

The provider-neutral frame interface now exports the exact model input.

It also exports codes, scale bits, raw entropy, recovered codes, and codebook order.

The LM interface records one canonical fixed-code vector for each rate.

Both local 203-step vectors recover the original codes exactly.

A two-step ARM64 LM probe produced this CDF sequence digest:

```text
69eba2389663d486bd1ed016bb17519c40632f8a1b29d2b56fd254b6120b12be
```

The 6 kbps 203-step ARM64 probe produced this digest:

```text
9a243c837a6e4b99b8da788be10262a1e626b0f695244094babf5569ea9340e2
```

The 12 kbps 203-step ARM64 probe produced this digest:

```text
10261b7f0ec351dec0fd62c67967d3bc8c125a9aae710e2c0f4dc37d93ea36d2
```

These 203-step digests are the local vector targets for their rates.

Prior CPU and Core ML tests produced identical EnCodec codes.

They also produced identical arithmetic payload bytes.

Two normalization scales differed by one binary32 unit in the last place.

The CPU decoder changed 82 of 384,000 signed 16-bit samples between those streams.

The maximum signed 16-bit difference was one.

Identical codes produced 698 differing samples across CPU and Core ML decoders.

The maximum signed 16-bit difference was one.

These results support exact code recovery as the primary cross-backend invariant.

They do not support exact neural PCM across all accelerators.

## Superseded WAV digests

The earlier seam and allocation changes retained two decoded WAV digests.

The short fixture had this SHA-256 digest:

```text
bd8806cd42a5bd7ac75dbe0ab59fc6e98a36b97b5056f53c83657402ccefb8a7
```

The long fixture had this SHA-256 digest:

```text
6649ead796cb4dd7413c7eecf28fb5902912297fc149b70686a9a7ede13db40a
```

The later PCM conversion refactor intentionally replaced the old output rounding.

Therefore, these WAV digests are historical evidence only.

The qualification run must create new Profile 1 reference digests.

## Original Meta compatibility

Profile 1 does not target Meta entropy-bitstream compatibility.

The pinned Meta neural model remains the semantic model source.

The original float LM path can produce different CDFs across runtimes.

PyTorch and ONNX also produced different later codebooks from one identical input.

The first codebook matched in that comparison.

Codebooks one through seven did not retain identical digests.

The compared normalization scales also differed.

Therefore, shared model weights do not create a complete decode contract.

Profile 1 must bind every decode-critical transformation and artifact.

## Remaining release blockers

### Profile 1 container

The Rust source still writes and reads the current `ECDC` version-zero envelope.

The current format uses `acv=2` as its entropy identifier.

Profile 1 needs a new magic value and a new version boundary.

Profile 1 must not use content guessing or legacy fallback.

### Geometry selection

The Profile 1 neural geometry is not frozen.

The qualification run must compare Meta geometry with both fixed-context candidates.

The selected geometry must not increase encoded size.

It must also pass quality, recovery, and runtime gates.

### Complete artifact binding

The current stream binds the LM weight digest.

It does not bind every neural graph and decoder artifact.

It also does not bind the complete preprocessing and postprocessing contract.

Profile 1 must bind these values through a canonical profile root.

### Resynchronization

The current length and CRC framing cannot recover safely after a damaged length.

It has no checked end directory.

Profile 1 needs redundant segment locations and bounded resynchronization.

### Resource limits

The current parser does not define all archive resource limits.

Profile 1 must limit header, segment, directory, sample, and allocation sizes.

### Cross-architecture entropy evidence

The shared entropy source reduces implementation drift.

It does not replace ARM64, x86-64, and WebAssembly conformance runs.

Each required environment must reproduce the 203-step digest.

Each environment must produce identical bytes from every fixed-code vector.

### Independent decoder

Only one decoder implementation currently defines the q8 entropy path.

Profile 1 needs a separately implemented decoder before release.

The second decoder must not share container or entropy code.

### Distributed WebAssembly artifact

The checked-in WebAssembly binary predates the latest source refactor.

The qualification owner must rebuild it from the locked release candidate.

Do not treat the present binary as release evidence.

### Release package

The decoder capsule does not yet exist.

The capsule needs sources, models, tools, vectors, manifests, and licenses.

The [durability policy](durability-and-bitcoin.md) defines the complete set.

## Handoff

Use the [qualification matrix](qualification-matrix.md) for the next test run.

Use the [implementation handoff](implementation-handoff.md) for the remaining Profile 1 work.

Do not spend more work on compatibility with `acv=2`, `acv=3`, or `acv=4`.
