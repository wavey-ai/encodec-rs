# Scalable 12/6 kbps ECDC design

Status: design and measured feasibility study; not a production bitstream.

## Objective

A newly encoded 12 kbps file should contain an independently extractable 6 kbps base layer while preserving all of these properties:

- identical RVQ codes and decoded PCM at both rates;
- no increase over the current 12 kbps file size;
- no encode, decode, memory, or bundle-size regression;
- extraction by byte copying and CRC calculation only;
- continued decoding of existing `acv=2` files;
- identical behavior for the 1333 ms and 1800 ms profiles.

## Frozen-model result

The current eight-codebook LM cannot meet every gate merely by splitting its arithmetic stream. Its transformer state at time `t` is conditioned on all eight codebooks from earlier timesteps. A four-codebook decoder does not have codebooks 5–8 and therefore cannot reproduce the same probability state.

The best frozen-model construction uses the canonical four-codebook LM for the base and the full eight-codebook state for the enhancement. It preserves exact audio and permits exact 6 kbps extraction, but it requires two nonlinear LM trajectories and is intrinsically larger:

| Profile | Current 12k entropy | Canonical base | Full-state enhancement | Layered entropy increase | Estimated file increase |
|---|---:|---:|---:|---:|---:|
| 1333 ms | 294,366 B | 143,676 B | 157,948 B | 7,258 B / 2.47% | approximately 2.71% |
| 1800 ms | 292,617 B | 142,819 B | 157,038 B | 7,240 B / 2.47% | approximately 2.66% |

Independent arithmetic termination accounts for only 418 B and 312 B respectively. The material penalty comes from making the base probabilities independent of enhancement history.

A one-transformer prototype conditioned only on the base produced a byte-identical canonical 6 kbps layer, but its full Westside smoke file was 4,998 B versus 4,594 B for `acv=2`, an 8.8% increase. Raw, gzip, Brotli, Deflate, and arithmetic-stream patch representations also failed the size gate.

These candidates are rejected and must not become production formats.

## Production design

Meeting every gate requires an entropy model trained for a base-first factorization. This changes compression only; the neural encoder, RVQ codes, decoder, audio quality, and model profiles remain unchanged.

### Base path

Keep the existing four-codebook q8 transformer, embeddings, heads, logit snapping, CDF construction, and arithmetic coder frozen. It produces an ordinary canonical `acv=2` 6 kbps payload:

```text
base_state[t] = base_transformer(base_state[t-1], base_codes[t-1])
base_codes[t] ~ base_heads(base_state[t])
```

An extracted base must match a standalone 6 kbps encode byte-for-byte.

### Enhancement path

Replace the current four enhancement embeddings and 1,024-way heads with a small q8 predictor trained only for codebooks 5–8:

- a 64- or 96-dimensional GRU state;
- a projection of the frozen base hidden state;
- embeddings of the current four base symbols;
- embeddings of prior enhancement symbols;
- sequential enhancement heads so a later enhancement codebook can condition on earlier enhancement codebooks from the same timestep.

The full-rate path therefore performs one existing five-layer transformer step plus one small enhancement step. The enhancement predictor should be cheaper than the four current `1024 × 200` output heads it replaces and should keep the combined LM artifact below the current 12 kbps LM size.

Training freezes every base-path value and optimizes enhancement cross-entropy only. Quantization-aware training must reproduce the q8 runtime, 2.1 logit snapping, deterministic CDF, profile resets, padding, and final-chunk behavior. Model selection uses actual arithmetic-coded bytes rather than floating-point loss.

On the measured master, the enhancement coder must save at least approximately 7.3 KB, or 4.6% relative to the frozen full-state enhancement, to offset the independent canonical base and framing.

### Container layout

Use a new arithmetic-code version while keeping the existing outer ECDC version and the unchanged `acv=2` decoder:

```text
header:
  nc = 8
  acv = 3
  base_codebooks = 4
  base_lm_hash
  enhancement_model_hash

chunk:
  scale_f32_be
  base_entropy_length_u32
  base_entropy
  enhancement_entropy
```

The existing outer chunk length and CRC protect the complete chunk. Base extraction writes a canonical four-codebook `acv=2` header and copies `scale + base_entropy` into ordinary CRC-wrapped chunks. It performs no neural inference, LM inference, or arithmetic decoding.

New decoders dispatch `acv=2` to the current path and `acv=3` to the layered path. Existing encoded files remain supported. Older binaries will reject `acv=3`, so rollout must be decode-first and the `acv=2` writer must remain available.

## Qualification gates

An `acv=3` writer remains experimental until a representative music, speech, silence, noise, and transient corpus passes all gates:

1. Extracted 6 kbps files are byte-identical to canonical `acv=2` encodes.
2. Full decode recovers all eight codebooks and scale bits exactly.
3. Full decoded PCM is identical to the corresponding current-code path.
4. Aggregate and per-file container sizes do not exceed `acv=2`.
5. Warm encode and decode do not regress on single-thread WASM, macOS MLX, or physical iPhone MLX.
6. Peak memory and shipped model bytes do not increase.
7. Both fixed geometries and short padded tails pass deterministic tests.

Benchmarks must use randomized AB/BA ordering, repeated control runs, stage timing, and a quiet host.

## Near-term path

Keep `acv=2` as the production format. When both outputs are explicitly
requested during encoding, reuse the eight-codebook neural result and produce
canonical 12 kbps and 6 kbps entropy streams with the paired LM
runtime. It:

- validates that the four-codebook LM is an exact prefix of the eight-codebook
  transformer, embeddings, and output heads;
- retains one eight-codebook weight set and two independent nonlinear states;
- computes the shared first-four embedding sum once;
- uses fused two-input AArch64 and WASM SIMD matrix kernels that load common
  weights once;
- runs two independent arithmetic coders, preserving both canonical files
  byte-for-byte.

The normal single-output path is unchanged. In randomized LM microbenchmarks,
the paired AArch64 path ranged from roughly neutral to 1.18× faster depending
on profile and host load. WASM medians ranged from 0.98× to 1.07× across both
profiles, establishing parity but not a stable kernel speedup. A single
ordered four-second Apple probe was 1.54× faster than two complete encodes
because it also avoids the second neural pass. These are provisional results
from a busy host, not acceptance numbers.

The browser-facing paired API adds 30,559 uncompressed bytes across generated
WASM and JavaScript, 10,279 bytes with gzip, or 7,032 bytes with Brotli. It
does not add model assets and avoids retaining the separate 6 kbps weights,
saving 6.54 MiB of live weight memory during paired operation.

For deferred generation, retain a temporary handle containing the first four codebooks and scale values. It can later produce canonical 6 kbps ECDC with one four-codebook entropy pass and no neural re-encode or eight-codebook entropy decode. If that handle has been discarded, an existing 12 kbps ECDC must be entropy-decoded before its four-codebook prefix can be re-encoded.
