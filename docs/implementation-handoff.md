# Profile 1 Implementation and Test Handoff

Status: Ready for the next model

Review date: 2026-07-25

## Objective

Select one Profile 1 neural geometry.

Implement one clean Profile 1 format.

Prove exact bits-to-codes behavior across required architectures.

Package all artifacts required for independent future decoding.

Do not implement backward compatibility.

## Read first

Read these documents in this order:

1. [Rust refactor review](refactor-review-2026-07-25.md)
2. [Lineage and LM contract](encodec-lineage-and-lm-contract.md)
3. [Profile 1 container draft](profile-1-container.md)
4. [Qualification matrix](qualification-matrix.md)
5. [Durability and Bitcoin policy](durability-and-bitcoin.md)

Read the workspace `AGENTS.md` before any change.

Apply the workspace documentation rules to every changed document.

## Non-goals

Do not preserve Meta ECDC byte compatibility.

Do not preserve Wavey `acv=2`, `acv=3`, or `acv=4`.

Do not add a legacy fallback to the Profile 1 decoder.

Do not infer a format from payload contents.

Do not optimize small allocations before the release gates pass.

Do not use an eMastered source file.

Do not publish a crate, container, model, or Bitcoin transaction during qualification.

Do not claim perpetual or guaranteed decoding.

## Current source boundary

The current crate version is `0.1.4`.

That version already exists outside this unreleased Profile 1 work.

Do not overwrite or republish version `0.1.4`.

Use version `0.2.0` when the Profile 1 API and format implementation start.

The current Rust source remains a baseline and migration scaffold.

The current `ECDC` version-zero envelope is not Profile 1.

The checked-in WebAssembly binary does not represent the latest source.

Rebuild WebAssembly only from a locked qualification commit.

## Current local progress

The provider-neutral frame evidence interface is implemented.

It exports the exact model input, codes, scale bits, raw entropy, recovered codes, and codebook order.

The interface supports a true variable final input for a dynamic model.

No local dynamic Meta bundle is available to run that path.

The LM evidence command records one canonical 203-step vector for each fixed-1333 rate.

Both local vectors recover their codes exactly.

The complete fixed-code pattern and length matrix has not run.

The full corpus and provider matrix has not run.

The next qualification step is to run these interfaces with the missing bundles and providers.

Do not start the Profile 1 wire implementation before geometry approval.

## Frozen logical invariants

Do not change these invariants during geometry testing:

- Neural codes use `[batch, codebook, time]`.
- RVQ layer `k` maps to LM channel `k`.
- LM embedding table `k` consumes channel `k`.
- LM output head `k` predicts channel `k`.
- Arithmetic symbols equal unchanged RVQ code values.
- The next LM input equals the prior code plus one.
- The first LM input contains zero on every codebook.
- Time is the outer entropy loop.
- Codebook is the inner entropy loop.
- All codebooks at one time are predicted in parallel.
- Each chunk starts new LM and arithmetic states.
- No chunk contains a hidden LM reset.
- Scale values use big-endian binary32.

Any proposed change to these rules needs a new design review.

## Work sequence

Complete each phase before the next phase starts.

Record each phase result in the qualification output directory.

Do not hide a failed result by rerunning without a new run identifier.

### Phase 0: Inspect and lock the handoff state

1. Run `git status --short`.
2. Save the output in the run record.
3. Review every existing change before editing an overlapping file.
4. Run `git diff --check`.
5. Run the local source checks.
6. Record the Git commit and dirty diff digest.
7. Create a qualification branch or isolated worktree.
8. Commit only after the owner approves the change set.

Use these local source checks:

```sh
cargo fmt --all -- --check
cargo check --all-features
cargo test --features wasm --no-fail-fast
cargo check \
  --target wasm32-unknown-unknown \
  --no-default-features \
  --features wasm

node --check browser-smoke/webgpu-matrix.js
node --check scripts/wasm-encode-fixture.mjs
node --check scripts/westside-chunk-wasm-roundtrip.mjs
```

Stop after any failed check.

Do not change a test expectation until the failure cause is understood.

### Phase 1: Build the geometry selection harness

Test these candidates at 6 kbps and 12 kbps:

| Candidate | Model window | Owned stride | Tail | Reconstruction |
| --- | ---: | ---: | --- | --- |
| `meta-1000` | 48,000 | 47,520 | True `T` | Meta triangle overlap-add |
| `fixed-1333` | 64,960 | 64,000 | Full 203 codes | Context crop and bound seam |
| `fixed-1800` | 87,360 | 86,400 | Full 273 codes | Context crop and bound seam |

The harness must use one canonical PCM input for each candidate.

The harness must hash every complete model input before inference.

The harness must export each `[1,K,T]` code tensor.

The harness must export each binary32 scale by its raw bits.

The harness must export each arithmetic payload before container framing.

The harness must export each complete candidate container.

The harness must record each structural byte category.

Use these byte categories:

- Fixed container header
- Embedded manifest
- Header CRC
- Chunk headers
- Scale values
- Arithmetic payloads
- Directory
- Trailer

Use the proposed Profile 1 overhead for estimates.

Label an estimate as `estimated`.

Do not label an estimate as an encoded Profile 1 object.

Implement a temporary exact framing writer if estimates can change the winner.

Keep that writer outside the public decoder API.

### Phase 2: Run the local geometry matrix

Use the six approved CONFIRMATION sources.

Use both Westside versions.

Use no eMastered file.

Verify every source digest before conversion.

Run all three geometry candidates on macOS ARM64.

Run the CPU neural backend first.

Run Core ML CPU and GPU rows after the CPU row passes.

Run the fixed-code entropy vectors before full audio.

Run the tail vectors before full mixes.

Record these size values for every object:

- Source sample frames
- Entropy bytes
- Scale bytes
- Structural bytes
- Complete container bytes
- Bytes per source second
- Percentage against each fixed baseline

Record these quality values for every object:

- Peak error
- Root mean square error
- Segmental signal-to-noise ratio
- Selected perceptual metric
- Seam-local error
- Tail-local error

Use one pinned implementation for each metric.

Record every metric tool digest.

Run a blinded listening check for final geometry approval.

Do not let an objective score replace the listening check.

### Phase 3: Apply the geometry gate

Select the smallest current fixed result for each source.

Use that result as the source baseline.

Count the complete candidate container.

Count an embedded manifest when release policy requires one.

Apply these gates separately at 6 kbps and 12 kbps:

- Aggregate candidate bytes do not exceed aggregate baseline bytes.
- No CONFIRMATION mix exceeds its matching baseline.
- Every strict bits-to-codes round trip passes.
- Every required quality threshold passes.
- Every tail length passes.
- Every damaged chunk remains independently bounded.

Reject `meta-1000` after any size-gate failure.

Do not weaken the gate without owner approval.

If `meta-1000` passes, propose it as the Profile 1 geometry.

If `meta-1000` fails, compare `fixed-1333` and `fixed-1800`.

Select no geometry until recovery and listening evidence also pass.

Write `geometry-decision.md` with all raw result links.

Request owner approval for the proposed geometry.

Stop before the public format freeze if approval is absent.

### Phase 4: Freeze Profile 1 wire choices

Resolve every choice in the owner-approval table.

The table is in [Profile 1 container](profile-1-container.md).

Record each approved value in one decision record.

Freeze these values:

- Public magic values
- File extension
- Media type
- Manifest canonicalization
- Manifest embedding policy
- Offset width
- Container size limit
- Local checksum
- Directory entry format
- Selected geometry
- Salvage conflict behavior
- Salvage concealment behavior
- Canonical PCM output
- Reference runtime
- Resource limits
- Entropy parameters

Change any provisional specification text after approval.

Calculate a draft profile root only after all manifest fields exist.

Do not use a draft root in published media.

### Phase 5: Create the Profile 1 module boundary

Set the crate version to `0.2.0`.

Create these modules or equivalent modules:

```text
src/profile1/mod.rs
src/profile1/container.rs
src/profile1/directory.rs
src/profile1/limits.rs
src/profile1/manifest.rs
src/profile1/segment.rs
src/profile1/strict.rs
src/profile1/salvage.rs
```

Keep shared neural traits outside the container module.

Keep shared q8 entropy code in [`entropy.rs`](../src/entropy.rs).

Keep arithmetic coding in [`arithmetic.rs`](../src/arithmetic.rs).

Keep canonical PCM conversion in [`pcm.rs`](../src/pcm.rs).

Remove the current format from the default public API.

Use Git history for obsolete format recovery.

Do not ship an automatic compatibility reader.

Do not copy old parser heuristics into Profile 1.

### Phase 6: Implement the strict container

Implement fixed-header writing.

Implement fixed-header parsing.

Implement optional manifest embedding.

Implement header CRC verification.

Implement chunk writing.

Implement chunk parsing.

Implement footer directory writing.

Implement footer directory parsing.

Implement trailer writing.

Implement trailer parsing.

Use checked arithmetic for every offset and length.

Apply resource limits before allocation.

Reject unknown flags and reserved values.

Reject trailing bytes.

Reject missing, duplicate, and reordered chunks.

Verify a chunk CRC before entropy decoding.

Verify the profile root before model loading.

Verify every artifact digest before first use.

Return structured errors with byte offsets and field names.

Do not include local paths in canonical errors or manifests.

### Phase 7: Implement salvage decoding

Keep salvage entry points separate from strict entry points.

Use the directory when its CRC passes.

Scan for `WVCK` only after directory failure.

Advance one byte after a failed magic candidate.

Validate every candidate length before reading its payload.

Validate the candidate CRC before accepting its ordinal.

Reject ordinals outside the declared count.

Treat byte-identical duplicates as one recovered chunk.

Treat different valid duplicates as a conflict.

Conceal only missing, invalid, or conflicted ordinals.

Report every concealed source range.

Report header, directory, and trailer damage separately.

Mark salvage output as noncanonical.

Never let salvage success satisfy a strict release gate.

### Phase 8: Bind the complete decoder

Create the canonical Profile 1 manifest schema.

Implement RFC 8785 canonicalization with conformance tests.

Reject duplicate JSON keys before canonicalization.

Validate required root keys.

Reject unknown root keys.

Validate integer ranges before serialization.

Validate every artifact role, length, and SHA-256 digest.

Bind these decode-critical artifacts:

- Neural decoder graph
- Neural decoder weights
- Neural operator contract
- RVQ layout
- LM evaluator
- LM weights
- Entropy specification
- Container specification
- Geometry specification
- Normalization specification
- Reconstruction specification
- PCM conversion specification
- Conformance vectors
- Licenses

Do not use an encoder digest as a decoder substitute.

Record encoder artifacts separately for provenance.

### Phase 9: Create conformance vectors

Create valid vectors for each supported codebook profile.

Create fixed-code vectors without neural inference.

Cover code values `0`, `1`, `1022`, and `1023`.

Cover every codebook position.

Cover BOS and prior-code-plus-one transitions.

Cover every valid final `T`.

Cover maximum chunk payloads.

Cover container boundaries.

Create one mutation for every protected field.

Create truncation vectors at every structure boundary.

Create missing, duplicate, reordered, and conflicting chunk vectors.

Store expected strict errors as structured data.

Store expected salvage ranges as structured data.

Store recovered code tensors and scale bits.

Store canonical PCM only for the selected reference runtime.

Do not require exact accelerator PCM unless qualification proves it.

### Phase 10: Build the test mux

Create one provider-neutral qualification command.

Give each provider adapter the same input lock.

Give each provider adapter the same result schema.

Suggested tooling layout:

```text
scripts/qualification/prepare.mjs
scripts/qualification/run-local.mjs
scripts/qualification/run-gcp.mjs
scripts/qualification/run-azure.mjs
scripts/qualification/merge.mjs
scripts/qualification/verify.mjs
```

The mux must never choose a corpus file by partial title.

The mux must reject every eMastered path.

The mux must verify source digests before upload.

The mux must verify release artifacts after upload.

The mux must record commands, exit status, and elapsed time.

The mux must write one JSONL record for each test case.

The mux must support resume by exact case identifier.

The mux must not overwrite a completed record.

The mux must label retries with a new attempt number.

The mux must record cloud resource identifiers.

The mux must stop resources after result retrieval.

The mux must not record credentials.

### Phase 11: Run required architectures

Run macOS ARM64 CPU.

Run macOS ARM64 Core ML CPU.

Run macOS ARM64 Core ML GPU.

Run Chromium WebAssembly entropy.

Run Chromium ONNX WebAssembly inference.

Run Chromium WebGPU inference.

Run GCP Linux x86-64 CPU.

Run GCP Linux x86-64 CUDA FP32.

Run TensorRT only when the release exposes TensorRT.

Run Azure only after explicit CLI authorization.

Do not run `az account show` before that authorization.

Use Azure for second-vendor evidence or a GCP coverage gap.

Do not let Azure delay a release that makes no Azure claim.

### Phase 12: Apply cross-architecture gates

Require identical fixed-code arithmetic bytes.

Require identical recovered code tensors.

Require identical scale bits for fixed-scale vectors.

Require the 203-step CDF sequence digest.

Require identical strict integrity decisions.

Require identical salvage accepted-ordinal sets.

Compare neural encoder codes separately from entropy behavior.

Compare neural decoder PCM separately from bits-to-codes behavior.

Use the selected CPU reference for canonical PCM.

Apply published tolerances to accelerator PCM.

Investigate every code mismatch before performance work.

Investigate every CDF mismatch before full-corpus work.

### Phase 13: Build the independent decoder

Create a separate source tree.

Use a different primary language or runtime.

Do not share container parsing code.

Do not share entropy probability code.

Do not share arithmetic coding code.

Do not generate one decoder from the other.

Use the same specification and immutable model artifacts.

Run all fixed-code and mutation vectors.

Require identical bits-to-codes output.

Require the same strict and salvage classifications.

Record all implementation differences.

### Phase 14: Assemble the decoder capsule

Follow the [durability policy](durability-and-bitcoin.md).

Include the complete reference decoder source.

Include the complete independent decoder source.

Include locked transitive source dependencies.

Include exact model and LM artifacts.

Include toolchain definitions.

Include build procedures.

Include ready-to-run portable artifacts.

Include all conformance vectors.

Include all expected results.

Include licenses and notices.

Include a software bill of materials.

Build and test the capsule without network access.

Calculate the capsule root after final assembly.

### Phase 15: Produce release evidence

Write one signed qualification report.

Include the qualification lock digest.

Include every environment result digest.

Include the selected geometry decision.

Include the profile root.

Include the complete container digest.

Include the capsule root.

Include strict and salvage summaries.

Include size and quality summaries.

Include all known limitations.

Do not publish until every required gate passes.

## Required result record

Each JSONL result needs these fields:

```json
{
  "schema": "wavey.encodec.qualification",
  "schema_version": 1,
  "run_id": "<stable-run-id>",
  "case_id": "<stable-case-id>",
  "attempt": 1,
  "status": "pass",
  "candidate": "meta-1000",
  "bandwidth_kbps": 6,
  "corpus_id": "confirmation-westside-v1",
  "source_sha256": "<sha256>",
  "canonical_pcm_sha256": "<sha256>",
  "model_input_sha256": "<sha256>",
  "codes_sha256": "<sha256>",
  "scale_bits_sha256": "<sha256>",
  "entropy_sha256": "<sha256>",
  "container_sha256": "<sha256>",
  "container_bytes": 0,
  "environment_id": "mac-arm64-onnx-cpu",
  "artifact_lock_sha256": "<sha256>",
  "elapsed_ms": 0,
  "metrics": {},
  "damage": null,
  "error": null
}
```

Use `status` values `pass`, `fail`, `skipped`, or `conditional_not_authorized`.

Do not omit a required field.

Use `null` only where the schema permits it.

## Stop conditions

Stop the run after a source digest mismatch.

Stop the run after an artifact lock mismatch.

Stop the run after a model-input hash mismatch.

Stop the run after a fixed-code entropy mismatch.

Stop the run after an unexpected codebook permutation.

Stop the run after an unclassified strict-parser difference.

Stop cloud work after an unbounded cost or quota error.

Stop Azure work before owner authorization.

Stop release work after any required gate failure.

Do not mark a failure as an expected tolerance without evidence.

## Expected handoff outputs

The next model must return these artifacts:

- Qualification lock
- Local preflight report
- Geometry result records
- Geometry decision report
- Size comparison
- Quality comparison
- Fixed-code conformance report
- Tail-vector report
- GCP CPU and GPU reports
- Browser and WebAssembly reports
- Conditional Azure status
- Mutation and salvage report
- Exact unresolved blocker list
- Recommended next commit boundary

Do not return only a prose summary.

## Copyable task for the next model

Use this task text:

```text
Work in /Users/jamie/wavey.ai/encodec-rs.

Read the workspace AGENTS.md and every document in encodec-rs/docs.
Preserve all existing worktree changes.
Do not implement backward compatibility.
Do not use any eMastered file.

First, run the local preflight from docs/implementation-handoff.md.
Then, build the provider-neutral qualification mux.
Run the geometry-selection matrix from docs/qualification-matrix.md.
Use only the approved Lori Asha CONFIRMATION sources.

Compare meta-1000, fixed-1333, and fixed-1800 at 6 and 12 kbps.
Count complete proposed Profile 1 container bytes.
Apply the strict no-size-regression gate.
Keep neural inference, bits-to-codes, and codes-to-PCM results separate.

Run Mac ARM64, browser/WASM, GCP x86-64 CPU, and GCP CUDA FP32 rows.
Do not use Azure until the owner authorizes the CLI.
Stop and report any exact entropy or recovered-code mismatch.

Do not publish crates, models, containers, or Bitcoin transactions.
Return the locked artifacts, JSONL results, decision report, and blocker list.
```
