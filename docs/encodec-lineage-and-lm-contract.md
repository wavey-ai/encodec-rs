# EnCodec Lineage and Language-Model Contract

Status: Candidate Profile 1 design contract

Review date: 2026-07-25

## Purpose

This document records the EnCodec lineage that affects durable Wavey decoding.

It also defines the neural, RVQ, language-model, and entropy rules for Profile 1.

Profile 1 defines Wavey bytes and Meta model semantics. It does not define Meta ECDC byte identity.

## Terms

A **code** is one integer index from one residual vector quantizer layer.

A **codebook** is one residual vector quantizer layer and its ordered set of vectors.

**RVQ** means residual vector quantization. Each RVQ layer quantizes the residual from all earlier layers.

An **LM** is the EnCodec language model that estimates code probabilities.

A **segment** is one independent neural and entropy coding unit.

An **entropy profile** defines the complete mapping between an ordered code sequence and bytes.

A **profile root** is the SHA-256 digest of a canonical profile manifest.

In this document, **MUST** identifies a Profile 1 requirement. **SHOULD** identifies a recommended Profile 1 property.

## Pinned Meta lineage

Profile 1 uses Meta commit [`0e2d0aed29362c8e8f52494baf3e6f99056b214f`](https://github.com/facebookresearch/encodec/tree/0e2d0aed29362c8e8f52494baf3e6f99056b214f) as its semantic review point.

The commit date is 2023-06-20. The source reports version `0.1.2a3`.

| Date | Release or source point | Durability effect |
| --- | --- | --- |
| 2022-10-24 | [EnCodec paper, arXiv v1](https://arxiv.org/abs/2210.13438) | Defines the neural codec and LM entropy method. |
| 2022-10-25 | [PyPI `0.1.0`](https://pypi.org/project/encodec/0.1.0/) | Publishes the first Python package. |
| 2022-10-25 | [PyPI `0.1.1`](https://pypi.org/project/encodec/0.1.1/) | Removes a command warning. |
| 2023-04-26 | [Meta MIT release commit](https://github.com/facebookresearch/encodec/commit/349b72939f57cb3bc7b60906c0ee8228c849485d) | Changes the source license to MIT. |
| 2023-06-20 | [Pinned Meta commit](https://github.com/facebookresearch/encodec/commit/0e2d0aed29362c8e8f52494baf3e6f99056b214f) | Adds Transformers information without a format change. |

The upstream [changelog](https://github.com/facebookresearch/encodec/blob/main/CHANGELOG.md) lists no durability or recovery release after `0.1.1`.

The source changelog contains an unfinished `0.1.2a1` section. Meta did not publish this name on PyPI.

Meta ECDC keeps outer header version `0` at the pinned commit.

Meta ECDC has no independent segment checksum, profile root, artifact manifest, or salvage contract.

## Wavey format lineage

Wavey added deterministic entropy rules and independent chunks after the Meta source point.

The `acv` field identifies Wavey entropy behavior. It does not replace the outer ECDC header version.

The Python and Rust `acv` values use separate namespaces. Their numbers do not define one ordered sequence.

| Implementation | Identifier | Relevant property |
| --- | --- | --- |
| Meta Python | ECDC header `0` | Uses the original float LM and arithmetic coder. |
| Wavey Python | `acv=3` | Uses deterministic entropy rules without independent CRC chunks. |
| Wavey Python | `acv=4` | Adds a length and CRC to each independent segment. |
| `encodec-rs` | `acv=2` | Uses the Wavey q8 LM and CRC-wrapped chunks. |
| Profile 1 | New profile root | Replaces all unpublished Wavey byte formats. |

Profile 1 decoders MUST reject these earlier Wavey payloads. Production decoders MUST not guess an earlier entropy profile.

The Wavey Python history is in [`compress.py`](../../encodec/encodec/compress.py).

The Rust format constants are in [`format.rs`](../src/format.rs).

## Pinned Meta neural model

Profile 1 uses the Meta `encodec_48khz` neural model.

The model is stereo, noncausal, normalized, and sampled at 48,000 Hz.

The model supports target bandwidths of 3, 6, 12, and 24 kbps.

The frame rate is 150 code frames per second. Each codebook contains 1,024 entries.

| Target bandwidth | RVQ codebooks |
| --- | ---: |
| 3 kbps | 2 |
| 6 kbps | 4 |
| 12 kbps | 8 |
| 24 kbps | 16 |

The pinned model construction is in Meta [`model.py`](https://github.com/facebookresearch/encodec/blob/0e2d0aed29362c8e8f52494baf3e6f99056b214f/encodec/model.py).

## Confirmed RVQ and LM mapping

Meta codes have shape `[B, K, T]`.

`B` is the batch count. `K` is the codebook count. `T` is the code-frame count.

The following mapping is normative for Profile 1.

| Stage | Mapping |
| --- | --- |
| RVQ encode | Residual layer `k` produces `codes[B, k, t]`. |
| LM input | Input channel `k` uses embedding table `emb[k]`. |
| LM output | Output channel `k` uses prediction head `linears[k]`. |
| Probability tensor | The layout is `[B, 1024, K, T]`. |
| CDF selection | CDF column `k` predicts codebook `k`. |
| Arithmetic symbol | Code value `c` is symbol `c`. |
| RVQ decode | Symbol `c` selects entry `c` from residual layer `k`. |

No codebook permutation occurs in this path.

No code-value permutation occurs in this path.

Meta predicts all codebooks for time `t` in parallel. Meta does not use delayed codebooks.

The RVQ layer order is visible in Meta [`core_vq.py`](https://github.com/facebookresearch/encodec/blob/0e2d0aed29362c8e8f52494baf3e6f99056b214f/encodec/quantization/core_vq.py).

The LM embedding and output-head order is visible in Meta [`model.py`](https://github.com/facebookresearch/encodec/blob/0e2d0aed29362c8e8f52494baf3e6f99056b214f/encodec/model.py).

### BOS and next input

The first LM input is an all-zero tensor with shape `[1, K, 1]`.

Embedding row `0` represents BOS or a missing previous code.

Embedding rows `1` through `1024` represent code values `0` through `1023`.

After coding value `c`, the next LM input on that codebook is `c + 1`.

The arithmetic symbol remains `c`. The `+1` operation applies only to the next LM input.

### Traversal order

The encoder traverses time first. It traverses codebooks from `0` through `K - 1` within each time.

The exact symbol order is:

```text
(t=0,k=0), (t=0,k=1), ... (t=0,k=K-1),
(t=1,k=0), (t=1,k=1), ... (t=1,k=K-1)
```

### State and termination

Each Meta segment starts with `states=None`, `offset=0`, and the BOS input.

The transformer keeps at most `past_context` prior frames. This limit does not reset the offset or BOS input.

The 48 kHz LM uses `past_context=525`. The value equals 3.5 seconds at 150 frames per second.

The arithmetic coder and LM state end after the declared `T` frames.

Meta defines no end symbol. The segment length determines termination.

The pinned loop is in Meta [`compress.py`](https://github.com/facebookresearch/encodec/blob/0e2d0aed29362c8e8f52494baf3e6f99056b214f/encodec/compress.py).

## Pinned Meta segment processing

Each full segment contains 48,000 samples. The segment stride is 47,520 samples.

The 480-sample overlap equals one percent of a full segment.

For a segment with `N` source samples, the code-frame count is:

```text
T = ceil(N * 150 / 48000)
```

The last segment uses its true `T`. Meta does not append synthetic code frames.

For normalized audio, Meta calculates one scale from the mono root-mean-square value.

Meta adds `1e-8` to the scale before normalization.

ECDC stores each scale as one big-endian IEEE 754 binary32 value before its segment payload.

Meta reconstructs overlapping segments with triangular linear overlap-add.

The pinned processing is in Meta [`model.py`](https://github.com/facebookresearch/encodec/blob/0e2d0aed29362c8e8f52494baf3e6f99056b214f/encodec/model.py).

## Wavey q8 entropy differences

Wavey q8 preserves the confirmed codebook and code-value mapping.

The q8 output buffer uses index `bin * K + k`. Rust reshapes this buffer to `[1, 1024, K, 1]`.

The q8 weight file starts with magic `ELMQ0001`.

The file stores embeddings, biases, normalization values, and positional values as binary32.

Linear weights use signed 8-bit values with one binary32 scale per output row.

Linear inputs use dynamic signed 16-bit quantization. Dot products use signed 64-bit accumulation.

Layer normalization and attention scores use binary64 calculations.

Attention exponential and GELU calculations use `libm`.

The q8 implementation is in [`quantized_lm.rs`](../src/quantized_lm.rs).

### Probability and CDF rules

The current q8 bundles declare an LM logit step of `1/64` and an entropy logit step of `2.1`.

The entropy path uses the larger declared step. It therefore uses `2.1` for these bundles.

The entropy path rounds logits to this grid before softmax.

It uses binary64 exponential, ordered summation, and conditional near-uniform replacement.

It multiplies probabilities by `8192` and floors the results to integer counts.

It applies a deterministic perturbation near integer boundaries.

It then uses largest-remainder allocation with a minimum range of `2`.

The final CDF value is exactly `2^24`.

The current rules are in [`metadata.rs`](../src/metadata.rs), [`ecdc.rs`](../src/ecdc.rs), and [`arithmetic.rs`](../src/arithmetic.rs).

The fixed-bundle exporter is [`export_encodec_fixed_chunk_bundles.sh`](../scripts/export_encodec_fixed_chunk_bundles.sh).

Meta uses different probability rules.

Meta quantizes each probability downward to a `1e-8` grid.

Meta adds the minimum range after direct floor allocation.

Meta does not distribute the unused remainder. Its final CDF value can be less than `2^24`.

The pinned Meta rules are in [`quantization/ac.py`](https://github.com/facebookresearch/encodec/blob/0e2d0aed29362c8e8f52494baf3e6f99056b214f/encodec/quantization/ac.py).

### Arithmetic interval rules

Meta and Wavey use different interval equations.

Meta uses:

```text
low  = base + ceil(cum_low * range / 2^24)
high = base + floor((cum_high - 1) * range / 2^24)
```

Wavey q8 uses:

```text
low  = base + floor(cum_low * range / 2^24)
high = base + floor(cum_high * range / 2^24) - 1
```

Both implementations pack arithmetic bits least-significant bit first within each byte.

These differences make Wavey q8 a separate lossless representation of the same RVQ codes.

## Pre-Profile hazards

These hazards apply to the audited unpublished formats. Profile 1 must not preserve their behavior.

### Extra LM reset

The audited Rust path reset the LM when `t` reached the exported LM frame length.

Meta resets the LM only at a model segment boundary.

Meta uses `past_context` to discard old attention state. It does not restart positional offset or BOS state.

The fixed Wavey bundles kept chunk length equal to exported LM length.

Those bundles did not exercise the extra reset branch. A longer chunk could produce a different code-to-byte mapping.

The correction point is the Rust [`ecdc.rs`](../src/ecdc.rs) segment loop.

### Hidden LM state

The audited native q8 adapter accepted state tensors but kept its effective state inside the adapter.

This design was sequentially usable but not reentrant.

It also made reset behavior dependent on hidden object history.

The adapter is in [`onnx.rs`](../src/onnx.rs).

### Fixed tail geometry

The fixed `1333ms` bundle uses a 64,960-sample model window and a 64,000-sample owned stride.

The fixed `1800ms` bundle uses an 87,360-sample model window and an 86,400-sample owned stride.

Both windows add 480 context samples on each side.

Their fixed code-frame counts are 203 and 273.

These values do not match Meta's 48,000-sample segment and 47,520-sample stride.

### Synthetic final codes

The pre-Profile WASM path could append code value `0` until the final chunk reached fixed `T`.

The native path could encode only the declared final `T`.

Therefore, the same short tail could produce different bytes across those paths.

Zero PCM padding and zero code symbols are different operations.

Synthetic code frames can change a noncausal decoder's output near the tail.

The relevant paths are [`wasm.rs`](../src/wasm.rs) and [`ecdc.rs`](../src/ecdc.rs).

### Non-Meta seam processing

The fixed-bundle path cropped private context from every decoded window.

It also used custom seam processing instead of Meta overlap-add.

These operations changed the postprocessing contract.

The seam implementation is in [`seam.rs`](../src/seam.rs).

## Profile 1 requirements

### Profile identity

Profile 1 MUST use a new format identifier.

The canonical manifest MUST include the pinned Meta commit.

The manifest MUST include SHA-256 digests for every decode artifact.

Artifacts MUST include the decoder graph, decoder weights, LM weights, and entropy implementation.

The manifest MUST bind preprocessing, normalization, segmentation, tail handling, and overlap-add rules.

The manifest MUST bind channel order, sample rate, PCM format, and output rounding.

Any semantic change MUST create a new profile root.

Profile 1 MUST not use `acv=2`, `acv=3`, or `acv=4` as its identity.

### Neural and RVQ contract

Profile 1 MUST use the pinned Meta `encodec_48khz` model semantics.

The artifact manifest MUST identify exact neural graph and weight bytes.

The encoder MUST preserve RVQ residual layer order.

The decoder MUST apply codebook `k` to RVQ layer `k`.

Code value `c` MUST select entry `c` without permutation.

Profile 1 MUST use 1,024 entries per codebook.

### Segment contract

The Profile 1 neural geometry is not frozen.

The geometry qualification MUST compare the Meta and fixed-context candidates.

The selected candidate MUST not increase encoded size against the best current baseline.

The selected candidate MUST pass all recovery and quality gates.

The profile manifest MUST bind the selected window, stride, context, tail, and reconstruction rules.

The Meta geometry candidate uses 48,000-sample windows and a 47,520-sample stride.

Its final segment uses the true source length.

Its final `T` uses `ceil(N * 150 / 48000)`.

Its implementations do not append synthetic code frames.

Its decoder graphs accept every valid `T` from `1` through `150`.

Its decoders use Meta triangular linear overlap-add.

The fixed-context candidates use complete fixed model windows.

Their final segments include all model code positions.

Their manifests bind context cropping and seam processing as decode rules.

Every candidate MUST store one big-endian binary32 scale for each normalized segment.

The release profile MUST contain only the selected geometry.

The release decoder MUST not infer geometry from an earlier Wavey payload.

### LM contract

Each segment MUST start with BOS input, zero state, and offset `0`.

LM input channel `k` MUST use embedding table `k`.

LM output channel `k` MUST use output head `k`.

The LM MUST predict all codebooks for one time in parallel.

The LM MUST not use delayed codebooks.

The encoder MUST traverse `t` before `k`.

The arithmetic symbol MUST equal the RVQ code value.

The next LM input MUST equal the prior code value plus one.

The LM MUST reset once at each segment start.

The LM MUST not reset inside a segment.

`past_context` trimming MUST preserve positional offset.

The API MUST pass effective state explicitly.

The API MUST not keep decode-critical hidden state.

### Entropy contract

Profile 1 MAY use q8 LM weights after cross-architecture conformance succeeds.

The q8 weight format MUST have an explicit version and complete schema.

The manifest MUST bind the complete q8 weight file.

The entropy profile MUST define logit quantization without implementation-defined rounding.

The entropy profile MUST define exponential or replace it with an integer transform.

The entropy profile MUST define summation order and near-uniform behavior.

The entropy profile MUST define `fp_scale`, minimum range, and CDF allocation.

The entropy profile MUST define arithmetic interval equations and total range.

The entropy profile MUST define bit order, byte order, termination, and flush behavior.

Two conforming implementations MUST produce identical bytes from identical codes.

GPU kernels MUST not control entropy bytes unless they pass the same exact test.

### Resilient framing

Each segment MUST have an independent LM state and arithmetic state.

Each segment MUST start with its encoded byte length.

Each segment MUST include its ordinal and source start.

Each segment MUST include its source sample count and code-frame count.

Each segment MUST include CRC-32/ISO-HDLC over its header and complete payload.

The container header MUST include the total segment count.

The container SHOULD include an index for bounded random access.

The container MUST detect truncation, reordering, duplication, and missing segments.

Strict decoding MUST stop and report the first invalid segment.

Salvage decoding MAY conceal only the invalid segment.

Salvage decoding MUST report every concealed range.

Salvage output MUST not be presented as canonical decode output.

### Durable artifact set

An archive MUST contain every artifact named by the profile manifest.

Alternatively, the same durable store MUST contain content-addressed copies of those artifacts.

A hash anchor alone proves identity. It does not preserve unavailable weights or source.

The durable set MUST include a buildable reference decoder.

The durable set MUST include a second, independent decoder.

The durable set MUST include source, build instructions, toolchain definitions, and licenses.

The durable set MUST include all conformance vectors.

The publisher MUST record the complete object SHA-256 digest with its Bitcoin reference.

CRC detects accidental damage. The object digest establishes content identity.

## Conformance requirements

Entropy conformance MUST begin with fixed codes. It MUST not depend on neural inference.

Vectors MUST include `T=1`, `T=149`, and `T=150`.

Vectors MUST include four-codebook and eight-codebook profiles.

Vectors MUST include uniform, peaked, alternating, and boundary code patterns.

Each vector MUST define exact LM inputs, CDF values, symbols, bytes, and decoded codes.

The exact bytes MUST match on `aarch64`, `x86_64`, and `wasm32`.

The exact decoded codes MUST match on macOS, Linux, and supported browser engines.

Neural conformance MUST test the same code tensors through every decoder backend.

One CPU reference backend SHOULD define canonical PCM output.

Other backends MUST satisfy published sample and perceptual tolerances.

Corruption tests MUST flip, remove, duplicate, reorder, and truncate segment bytes.

Tests MUST verify strict and salvage results for every corruption case.

The music matrix SHOULD include the Lori Asha CONFIRMATION mixes.

The matrix SHOULD include silence, impulses, speech, dense transients, and long program material.

## Claim boundary

Wavey can claim deterministic code recovery only after the entropy conformance suite passes.

Wavey can claim durable decode only when the durable artifact set remains available.

Wavey should not claim that a Profile 1 payload is a Meta ECDC bitstream.

Wavey should state that Profile 1 preserves the pinned EnCodec model semantics and adds independent resilient framing.

No design can guarantee decode after loss of the payload, manifest, weights, and decoder sources.

## Source index

- [Pinned Meta source tree](https://github.com/facebookresearch/encodec/tree/0e2d0aed29362c8e8f52494baf3e6f99056b214f)
- [EnCodec paper](https://arxiv.org/abs/2210.13438)
- [Meta model and LM](https://github.com/facebookresearch/encodec/blob/0e2d0aed29362c8e8f52494baf3e6f99056b214f/encodec/model.py)
- [Meta ECDC coding loop](https://github.com/facebookresearch/encodec/blob/0e2d0aed29362c8e8f52494baf3e6f99056b214f/encodec/compress.py)
- [Meta arithmetic coder](https://github.com/facebookresearch/encodec/blob/0e2d0aed29362c8e8f52494baf3e6f99056b214f/encodec/quantization/ac.py)
- [Meta RVQ implementation](https://github.com/facebookresearch/encodec/blob/0e2d0aed29362c8e8f52494baf3e6f99056b214f/encodec/quantization/core_vq.py)
- [Meta ECDC header](https://github.com/facebookresearch/encodec/blob/0e2d0aed29362c8e8f52494baf3e6f99056b214f/encodec/binary.py)
- [Wavey Python entropy implementation](../../encodec/encodec/compress.py)
- [Wavey Python model fork](../../encodec/encodec/model.py)
- [Rust ECDC implementation](../src/ecdc.rs)
- [Rust q8 LM implementation](../src/quantized_lm.rs)
- [Rust arithmetic coder](../src/arithmetic.rs)
- [Rust native adapter](../src/onnx.rs)
- [Rust WASM adapter](../src/wasm.rs)
