# Geometry Decision

Status: blocked pending Profile 1 qualification evidence.

Run: `20260725T050000Z-redo-e3fd8a8e64f64988`

## Measurement state

The run records current unpublished `acv=2` measurements only. These bytes are not Profile 1 objects. The final Profile 1 container remains intentionally unimplemented.

The selection record is [geometry-selection.json](geometry-selection.json). Its current status is `blocked`.

## Candidates

- `meta-1000`: 48,000-sample window, 47,520-sample stride, true variable tail, triangle overlap-add.
- `fixed-1333`: 64,960-sample window, 64,000 owned samples, 203 code frames, bound seam repair.
- `fixed-1800`: 87,360-sample window, 86,400 owned samples, 273 code frames, bound seam repair.

## Unresolved blockers

- The worktree contains uncommitted code and model changes.
- The current CLI does not export fixed-code vectors, recovered code tensors, or complete Profile 1 objects.
- The browser path is hard-wired to the current `acv=2` envelope.
- GCP CPU execution did not complete because the temporary worker package transfer failed.
- GCP CUDA execution is blocked by a project-wide GPU quota of zero.
- Owner approval is required before a geometry selection can be locked.

## Next implementation boundary

Implement the provider-neutral fixed-code and candidate evidence interfaces. Then implement the approved Profile 1 container and its strict and salvage decoders. Freeze one geometry only after complete-container size, entropy, recovery, quality, and runtime gates pass.

Raw evidence is under [metrics/results.jsonl](metrics/results.jsonl), [metrics/summary.json](metrics/summary.json), [metrics/gates.json](metrics/gates.json), [corpus-manifest.json](corpus-manifest.json), and [logs/commands.jsonl](logs/commands.jsonl).
