# EnCodec-RS Durability Work

Status: Profile 1 design and test handoff

Review date: 2026-07-25

## Assessment

Wavey has the strongest archive-focused EnCodec design found in this review.

The current design adds deterministic q8 entropy, independent chunks, and local CRC checks.

However, the current `acv=2` format is not ready for a durable-publication claim.

It does not bind the complete decoder and reconstruction contract.

It also lacks bounded resynchronization after a damaged chunk length.

Profile 1 must replace the current unpublished format.

Profile 1 does not need Meta or earlier Wavey bitstream compatibility.

The pinned Meta model semantics and weight lineage remain important.

## Current decision

Do not freeze the neural geometry before the size matrix.

Test Meta geometry against both fixed-context candidates.

Select Meta geometry only when it has no complete-container size regression.

Keep independent chunks in every candidate.

Use a new Profile 1 identity after the geometry decision.

Do not promise that an object is always decodable.

Use the evidence-based claims in the durability policy.

Keep a separate lossless master.

## Documents

- [Rust refactor review](refactor-review-2026-07-25.md)
- [EnCodec lineage and LM contract](encodec-lineage-and-lm-contract.md)
- [Profile 1 container draft](profile-1-container.md)
- [Qualification matrix](qualification-matrix.md)
- [Implementation and test handoff](implementation-handoff.md)
- [Durability and Bitcoin policy](durability-and-bitcoin.md)
- [Browser backend parity and the PRAY 4 ME bitrate matrix](browser-backend-parity-and-bitrate-matrix.md)

## Recorded results

- [Qualification run 20260725T050000Z](../qualification-results/20260725T050000Z-redo-e3fd8a8e64f64988/README.md)

This record contains 24 direct Meta-versus-fork quality comparisons.
The recorded environment is local macOS ARM64.

The record also contains the GCP provisioning result.
No GCP codec row completed.

## Next action

Give the implementation handoff to the next testing model.

Run the local and GCP geometry matrix before any wire-format freeze.

Authorize Azure separately if the release needs second-vendor evidence.

Do not publish Profile 1 bytes before all owner decisions and release gates pass.
