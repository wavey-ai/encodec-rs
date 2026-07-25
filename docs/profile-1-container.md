# Profile 1 Container

Status: Draft for owner approval

Review date: 2026-07-25

## Purpose

This document proposes the Profile 1 container and recovery contract.

Profile 1 is a clean format break.

Profile 1 does not read or write Meta ECDC bytes.

Profile 1 does not read or write Wavey `acv=2`, `acv=3`, or `acv=4` bytes.

The [lineage contract](encodec-lineage-and-lm-contract.md) defines the related EnCodec and LM semantics.

## Requirement terms

In this document, **MUST** identifies a required property.

**SHOULD** identifies a recommended property.

**MAY** identifies an optional property.

The wire constants in this draft remain provisional until owner approval.

## Design goals

Profile 1 has the following goals:

- Bind every decode rule to one profile root.
- Keep each chunk independently decodable.
- Recover valid chunks when the directory is damaged.
- Detect damaged headers, chunks, directories, and trailers.
- Permit strict and salvage decoding.
- Bound all input-controlled resource use.
- Keep complete container size at or below the current baseline.
- Preserve exact LM and RVQ symbol order.
- Preserve every artifact required for future decoding.

Profile 1 does not promise recovery from lost bytes.

## Terms

A **profile manifest** defines all decode semantics and artifacts.

A **profile root** is the SHA-256 digest of the canonical profile manifest.

A **container header** identifies the profile and the source length.

A **chunk** contains one independently coded neural segment.

A **directory** lists every chunk offset in ordinal order.

A **trailer** locates and verifies the directory.

A **strict decode** accepts only one complete and valid container.

A **salvage decode** recovers independently valid chunks and reports all damage.

A **decoder pack** contains every artifact required for offline decoding.

## Byte conventions

All container integers use unsigned big-endian encoding.

All offsets start at the first container byte.

All lengths count bytes unless a field name states another unit.

All binary32 values use IEEE 754 binary32 encoding.

Profile 1 stores segment scale values in big-endian byte order.

No structure has implicit padding or alignment.

Reserved fields MUST contain zero.

Readers MUST reject nonzero reserved fields.

## Checksum

Profile 1 uses CRC-32/ISO-HDLC for local damage detection.

The polynomial is `0x04C11DB7`.

The reflected polynomial is `0xEDB88320`.

The initial value is `0xffffffff`.

The final exclusive-or value is `0xffffffff`.

Input and output reflection are enabled.

The check value for ASCII `123456789` is `0xcbf43926`.

CRC does not provide authenticity.

An external SHA-256 digest identifies the complete container object.

## Proposed file layout

The proposed file layout is:

```text
container header
optional canonical profile manifest
header CRC
chunk 0
chunk 1
...
chunk N-1
footer directory
trailer
```

The file MUST end after the trailer.

Strict readers MUST reject trailing bytes.

## Container header

The fixed container header has 64 bytes.

| Offset | Size | Field | Required value |
| ---: | ---: | --- | --- |
| 0 | 8 | `magic` | ASCII `WVYENC01` |
| 8 | 2 | `container_version` | `1` |
| 10 | 2 | `flags` | Defined below |
| 12 | 32 | `profile_root` | Raw SHA-256 digest |
| 44 | 8 | `total_source_samples` | Positive sample-frame count |
| 52 | 4 | `chunk_count` | Positive chunk count |
| 56 | 4 | `embedded_manifest_bytes` | Canonical manifest length |
| 60 | 4 | `reserved` | `0` |

Flag bit `0` indicates an embedded profile manifest.

All other flag bits MUST contain zero.

`embedded_manifest_bytes` MUST be zero when flag bit `0` is zero.

`embedded_manifest_bytes` MUST be positive when flag bit `0` is one.

The optional manifest starts immediately after the fixed header.

The four-byte header CRC follows the optional manifest.

The CRC input contains the fixed header and optional manifest.

The CRC input does not contain the stored CRC value.

The first chunk starts after the header CRC.

The first chunk offset is:

```text
68 + embedded_manifest_bytes
```

## Profile binding

The header stores the raw 32-byte profile root.

An embedded manifest MUST use the canonical form defined below.

Its SHA-256 digest MUST equal `profile_root`.

An external manifest MUST pass the same check before decoding starts.

The decoder MUST not select artifacts from names, URLs, or local defaults.

The decoder MUST select every artifact by its bound SHA-256 digest.

The decoder MUST verify each artifact before its first use.

A decoder MUST not guess a profile from audio properties.

Bundle metadata MUST not supply a missing profile rule.

The decoder MUST compare bundle metadata with the profile manifest.

The decoder MUST reject every mismatch.

## Canonical profile manifest

Profile 1 proposes JSON Canonicalization Scheme, RFC 8785.

The canonical manifest uses UTF-8 without a byte-order mark.

The manifest MUST not contain duplicate object keys.

The manifest MUST use Unicode Normalization Form C for strings.

The manifest MUST use ASCII schema keys.

The manifest MUST use integers for all numeric values.

The manifest MUST not use binary floating-point JSON values.

Exact fractions MUST use integer numerator and denominator fields.

Every integer MUST be within the exact I-JSON integer range.

Every SHA-256 value MUST use 64 lowercase hexadecimal characters.

Artifact arrays MUST sort entries by `role`, then `sha256`.

Other arrays MUST define their order in the profile schema.

The manifest MUST not contain its own profile root.

The profile root calculation is:

```text
profile_root = SHA-256(RFC8785(profile_manifest))
```

Any decode-semantic change MUST produce a new profile root.

Descriptive release data MUST remain outside the profile manifest.

## Required manifest subjects

The manifest schema MUST define each subject in this table.

| Subject | Required content |
| --- | --- |
| `schema` | Schema name and schema version |
| `container` | Container version, checksum, and byte conventions |
| `source` | Pinned Meta commit and Wavey source archive |
| `neural_decoder` | Exact graph, weights, operators, shapes, and numerical rules |
| `rvq` | Codebook count, cardinality, layer order, and code-value mapping |
| `lm` | Architecture, weights, state, BOS, offset, and tensor layout |
| `entropy` | Logit, CDF, arithmetic, bit, and termination rules |
| `geometry` | Sample rate, channels, segment length, stride, and tail rule |
| `normalization` | Scale calculation and scale serialization |
| `reconstruction` | Neural output length, overlap-add, trim, and PCM conversion |
| `artifacts` | Role, SHA-256, byte length, media type, and format version |
| `conformance` | Vector-set digest and required results |
| `licenses` | License artifact digests for all required components |

Every required artifact role MUST have one record.

One byte-identical artifact MAY have records for multiple roles.

A URL MAY appear in a separate location index.

A URL MUST not define artifact identity.

One profile manifest MUST define exactly one codebook count.

Different codebook counts MUST have different profile roots.

### Root object

The root object MUST contain exactly these keys.

| Key | JSON type | Required rule |
| --- | --- | --- |
| `schema` | String | Exact value `wavey.encodec.profile` |
| `schema_version` | Integer | Exact value `1` |
| `container` | Object | Container rules |
| `source` | Object | Source identity |
| `neural_decoder` | Object | Neural decode contract |
| `rvq` | Object | RVQ contract |
| `lm` | Object | LM contract |
| `entropy` | Object | Entropy contract |
| `geometry` | Object | Segment geometry |
| `normalization` | Object | Scale rules |
| `reconstruction` | Object | Output rules |
| `artifacts` | Array | Required artifact records |
| `conformance` | Object | Conformance identity |
| `licenses` | Array | License artifact roles |

A schema version change MUST produce a new profile root.

Readers MUST reject unknown root keys.

### Semantic objects

The `container` object MUST identify version `1` and CRC-32/ISO-HDLC.

The `source` object MUST contain the pinned Meta commit.

The `source` object MUST contain a SHA-256 digest for the reviewed Wavey source archive.

The `neural_decoder` object MUST identify graph, weight, and operator-contract artifact roles.

It MUST define input layout, output layout, tensor element types, and dynamic axes.

The `rvq` object MUST define codebook count, cardinality, layer order, and code-value mapping.

The `lm` object MUST define BOS, next-input offset, state layout, tensor layout, and reset scope.

It MUST identify the LM weight and LM evaluator artifact roles.

The `entropy` object MUST identify its normative specification artifact.

It MUST bind every numeric parameter used to make CDF values and arithmetic bytes.

The `geometry` object MUST define model window, owned stride, private context, and tail formulas.

It MUST define left and right private context separately.

The `normalization` object MUST define scale calculation and binary32 serialization.

The `reconstruction` object MUST define crop, overlap-add, seam, trim, canonical PCM, and rounding.

The `conformance` object MUST identify the normative vector artifact and its SHA-256 digest.

### Artifact records

Each artifact record MUST contain exactly these keys.

| Key | JSON type | Required rule |
| --- | --- | --- |
| `role` | String | Unique stable role |
| `sha256` | String | Lowercase SHA-256 digest |
| `bytes` | Integer | Exact artifact length |
| `media_type` | String | Registered or documented type |
| `format` | String | Stable artifact format |
| `format_version` | Integer | Exact format version |

Artifact role names MUST use lowercase ASCII with underscores.

Readers MUST reject unknown artifact keys.

The digest and length MUST match before an artifact becomes available to decoding.

### Exact fractions

An exact fraction object MUST contain `numerator` and `denominator`.

Both values MUST be integers.

The denominator MUST be positive.

The fraction MUST use lowest terms.

The manifest MUST use fraction objects for noninteger entropy parameters.

## Chunk layout

Each chunk starts with a 16-byte header.

| Offset | Size | Field | Required value |
| ---: | ---: | --- | --- |
| 0 | 4 | `chunk_magic` | ASCII `WVCK` |
| 4 | 4 | `ordinal` | Zero-based chunk ordinal |
| 8 | 4 | `payload_bytes` | Positive payload length |
| 12 | 4 | `chunk_crc` | Header and payload CRC |

The chunk payload starts immediately after this header.

The chunk CRC covers the first 12 header bytes and the complete payload.

The chunk CRC does not contain the stored CRC field.

The chunk format has no optional padding.

The next structure starts after `16 + payload_bytes`.

### Chunk-derived values

The profile manifest defines the segment geometry.

The decoder derives the segment source start from `ordinal`.

The decoder derives the segment source length from the source start.

The decoder derives code-frame count `T` from that source length.

The decoder obtains codebook count `K` from the profile manifest.

Strict readers MUST reject an ordinal outside `0..chunk_count-1`.

Strict readers MUST reject duplicate or missing ordinals.

This design keeps the chunk header small.

It requires an intact container header or an external copy of its values.

### Candidate chunk payload

The Meta geometry candidate uses normalized 48 kHz EnCodec.

Its payload starts with one four-byte segment scale.

The arithmetic stream follows the scale.

The scale uses big-endian binary32 encoding.

The arithmetic stream contains exactly `K * T` symbols.

The stream contains no end symbol.

The chunk length defines the arithmetic byte boundary.

The profile manifest defines arithmetic termination and permitted trailing bits.

## Independent recovery

Each chunk MUST start with fresh LM state.

Each chunk MUST start with fresh arithmetic state.

No chunk MAY depend on an earlier chunk payload.

The decoder MUST derive BOS, offset, and geometry without earlier chunk data.

A recovered chunk requires the matching profile manifest.

A recovered chunk also requires `total_source_samples` and `chunk_count`.

Chunks are independently recoverable within this context.

They are not standalone Profile 1 containers.

## Footer directory

The directory starts after the last chunk.

The directory header has 12 bytes.

| Offset | Size | Field | Required value |
| ---: | ---: | --- | --- |
| 0 | 4 | `directory_magic` | ASCII `WVDI` |
| 4 | 2 | `directory_version` | `1` |
| 6 | 2 | `entry_bytes` | `4` |
| 8 | 4 | `entry_count` | Equal to `chunk_count` |

The directory contains one four-byte offset for each chunk.

Entry `i` contains the unsigned offset of chunk `i`.

Entries MUST use ordinal order.

Offsets MUST increase strictly.

The four-byte directory CRC follows the final entry.

The directory CRC covers the directory header and all entries.

The directory CRC does not contain the stored CRC value.

The complete directory length is:

```text
16 + 4 * chunk_count
```

## Trailer

The trailer has 16 bytes.

| Offset | Size | Field | Required value |
| ---: | ---: | --- | --- |
| 0 | 4 | `trailer_magic` | ASCII `WVND` |
| 4 | 4 | `directory_offset` | Directory start |
| 8 | 4 | `directory_bytes` | Complete directory length |
| 12 | 4 | `trailer_crc` | Trailer CRC |

The trailer CRC covers the first 12 trailer bytes.

The trailer MUST occupy the final 16 file bytes.

The directory MUST end at the trailer start.

The directory offset limit makes the complete container smaller than 4 GiB.

This limit keeps every directory entry at four bytes.

## Strict directory checks

Strict decoding MUST validate the trailer before reading directory entries.

Strict decoding MUST validate the directory CRC.

The header, directory, and derived chunk counts MUST match.

The first directory offset MUST equal the first chunk offset.

Each directory offset MUST identify the expected `WVCK` header.

Each later offset MUST equal the prior chunk end.

The final chunk end MUST equal `directory_offset`.

Each physical chunk ordinal MUST equal its directory index.

Each chunk CRC MUST pass before that chunk enters neural decoding.

All offset and length calculations MUST use checked arithmetic.

## Exact LM and RVQ order

Decoded EnCodec codes have shape `[1, K, T]`.

Codebook `k` maps directly to RVQ residual layer `k`.

Code value `c` maps directly to entry `c` in that layer.

No Profile 1 implementation may permute codebooks.

No Profile 1 implementation may permute code values.

No Profile 1 implementation may delay a codebook.

The encoder visits time `t` before codebook `k`.

The exact arithmetic symbol order is:

```text
(t=0,k=0), (t=0,k=1), ... (t=0,k=K-1),
(t=1,k=0), (t=1,k=1), ... (t=1,k=K-1)
```

The LM input has shape `[1, K, 1]` at each step.

The first LM input contains zero on every codebook.

LM embedding row `0` represents BOS.

After symbol `c`, the next LM input is `c + 1`.

The arithmetic symbol remains `c`.

LM output column `k` predicts RVQ codebook `k`.

The LM output layout is `[1, cardinality, K, 1]`.

Each chunk starts with LM offset `0`.

The LM MUST not reset within a chunk.

The arithmetic coder MUST not reset within a chunk.

## Meta geometry candidate

The official Meta geometry remains a candidate until the size gate passes.

The proposed geometry has these values:

| Property | Candidate value |
| --- | ---: |
| Sample rate | 48,000 Hz |
| Channels | 2 |
| Maximum model window | 48,000 samples |
| Owned stride | 47,520 samples |
| Left private context | 0 samples |
| Right private context | 0 samples |
| Frame rate | 150 frames per second |
| Overlap | 480 samples |
| Overlap method | Meta triangular linear overlap-add |
| Crop | None |
| Seam processing | None |
| Final tail | True source length |
| Normalization | Mono RMS plus `1e-8` |
| Scale representation | Big-endian binary32 |

For chunk ordinal `i`, the source start is:

```text
start = i * 47520
```

The source length is:

```text
N = min(48000, total_source_samples - start)
```

The code-frame count is:

```text
T = ceil(N * 150 / 48000)
```

The chunk count is:

```text
ceil(total_source_samples / 47520)
```

The final decoded timeline MUST trim to `total_source_samples`.

The candidate MUST not add synthetic code frames.

The candidate MUST not use private context cropping.

The candidate MUST not use custom seam repair.

Bundle selection MUST not change any candidate geometry value.

Failure of the size gate rejects this geometry.

The failure does not change the container framing.

A replacement geometry requires owner approval and a different profile root.

## No-size-regression gate

The gate compares complete containers, not isolated entropy payloads.

The candidate and baseline MUST use identical source samples.

The candidate and baseline MUST use the same codebook count.

The baseline MUST use the smallest current complete Wavey q8 result.

The baseline MUST use the real encode and decode path.

A benchmark-only framing path MUST not define the baseline.

The candidate byte count includes its header, chunks, directory, and trailer.

An embedded manifest counts when the release policy requires it.

The report MUST show entropy bytes and structural bytes separately.

The report MUST also show the one-time decoder-pack size.

The initial gate has these proposed size rules:

- Candidate aggregate bytes MUST not exceed baseline aggregate bytes.
- No full CONFIRMATION mix MAY exceed its matching baseline.
- Short vectors MAY report fixed overhead separately.
- Six-kbps and twelve-kbps profiles MUST pass separately.

The gate also requires equivalent or better decoded quality.

Owner-approved objective thresholds MUST exist before the final run.

Owner-approved listening results MUST accompany the objective report.

The gate MUST use the Lori Asha CONFIRMATION mixes.

The gate SHOULD include silence, impulses, speech, and dense transients.

The gate MUST include final segments for every `T` from `1` through `150`.

The gate MUST report macOS, Linux, and browser results.

Any failed strict round trip MUST fail the gate.

Any bits-to-codes mismatch MUST fail the gate.

## Structural overhead

Without an embedded manifest, proposed structural overhead is:

```text
100 + 20 * chunk_count bytes
```

This value includes the header CRC, chunk headers, directory, and trailer.

It does not include four-byte scale values inside normalized payloads.

The baseline comparison MUST count equivalent scale bytes.

## Strict decoding

Strict mode requires a valid profile manifest and complete decoder pack.

Strict mode MUST validate the container header and header CRC.

Strict mode MUST validate the profile root before artifact loading.

Strict mode MUST validate every artifact digest before use.

Strict mode MUST validate the trailer and complete directory.

Strict mode MUST validate each chunk CRC before decoding that chunk.

Strict mode MUST reject any missing, duplicate, or reordered chunk.

Strict mode MUST reject unknown flags and unsupported versions.

Strict mode MUST reject any geometry mismatch.

Strict mode MUST reject any arithmetic decode error.

Strict mode MUST reject unexpected code counts or code values.

Strict mode MUST produce exactly `total_source_samples` sample frames.

Strict mode MUST report failure as structured data.

A streaming strict decoder MAY emit provisional PCM before final validation.

It MUST not mark provisional PCM as complete after any failure.

## Salvage decoding

Salvage mode requires a valid profile manifest.

Salvage mode MUST not guess model, LM, entropy, or geometry rules.

Salvage mode SHOULD use a valid directory when one is available.

If the directory fails, salvage mode MAY scan for `WVCK`.

The scanner MUST validate the ordinal, length, resource limits, and chunk CRC.

An invalid candidate MUST not consume later candidate bytes.

The scanner SHOULD continue after the candidate magic position.

Salvage mode MUST place accepted chunks by ordinal.

Byte-identical duplicate chunks MAY count as one recovered chunk.

Different valid chunks with one ordinal create a conflict.

Salvage mode MUST conceal a conflicted ordinal.

Salvage mode MUST conceal each missing or invalid ordinal.

The proposed concealment uses one zero-valued decoded segment.

The zero segment MUST have the expected neural output length.

Salvage mode then applies the profile overlap-add and final trim.

Salvage mode MUST report each damaged source range.

Salvage mode MUST report footer and directory damage separately.

Salvage output is not canonical Profile 1 output.

Applications MUST label salvage output as recovered audio.

## Resource limits

Profile 1 uses these proposed hard limits:

| Resource | Proposed limit |
| --- | ---: |
| Complete container | `4,294,967,295` bytes |
| Embedded manifest | `1,048,576` bytes |
| Chunk count | `1,000,000` |
| One chunk payload | `65,536` bytes |
| Codebooks | `16` |
| Codebook cardinality | `1,024` |
| Code frames per chunk | `150` |
| Channels | `2` |
| Sample rate | `48,000` Hz |

A decoder MUST reject a value above these limits.

A decoder MUST support caller limits below these values.

The reference decoder SHOULD default to at most `2^32 - 1` output sample frames.

A caller MAY raise its output limit without changing the profile root.

The decoder MUST validate all lengths before allocation.

The decoder MUST use checked addition and multiplication.

The decoder MUST not allocate full output PCM when streaming is requested.

The decoder MUST bound arithmetic work to `K * T` symbols per chunk.

The decoder MUST bound neural tensor shapes from profile values.

The decoder MUST not follow decode-critical network references.

## Decoder completeness

The decoder pack MUST support decoding without a network connection.

The pack MUST contain the canonical profile manifest.

The pack MUST contain the neural decoder graph and all neural weights.

The pack MUST contain the LM implementation and exact LM weights.

The pack MUST contain the entropy and container specifications.

The pack MUST contain a buildable reference decoder.

The pack MUST contain all transitive source dependencies.

The pack MUST contain pinned toolchain definitions.

The pack MUST contain conformance vectors and expected results.

The pack MUST contain all required licenses and notices.

The pack MUST contain a human-readable build and decode procedure.

The procedure MUST not require a package registry.

The pack SHOULD contain one ready-to-run portable decoder.

The pack SHOULD contain a second independent decoder.

At least one decoder MUST run on a documented CPU reference environment.

The manifest MUST bind the canonical PCM conversion rule.

Non-reference decoders MUST meet published PCM and perceptual tolerances.

Bits-to-codes results MUST match exactly on every conforming decoder.

## Decoder conformance

Container vectors MUST cover every header, chunk, directory, and trailer field.

Vectors MUST cover zero, maximum, and overflowing length values.

Vectors MUST cover bad CRC values for every protected structure.

Vectors MUST cover truncated data at every structure boundary.

Vectors MUST cover missing, duplicate, reordered, and conflicting chunks.

LM vectors MUST start from fixed code tensors.

LM vectors MUST cover BOS, `c + 1`, and every codebook position.

LM vectors MUST prove time-major and codebook-minor traversal.

LM vectors MUST prove direct RVQ layer and code-value mapping.

Entropy bytes MUST match on `aarch64`, `x86_64`, and `wasm32`.

Neural vectors MUST use exact code tensors across every decoder backend.

The conformance report MUST separate bits-to-codes from codes-to-PCM results.

PyTorch inference MUST not serve as an unstated substitute for the bound decoder graph.

## Publishing and Bitcoin binding

The publisher MUST calculate SHA-256 over the complete container.

The publisher MUST record this digest with the Bitcoin content reference.

The publisher MUST place the matching decoder pack in durable storage.

The publisher MUST record the decoder-pack reference with its SHA-256 digest.

The publisher MUST record the profile root with both references.

A Bitcoin hash reference proves identity.

It does not preserve an artifact stored only at a temporary URL.

The durable claim requires the container and complete decoder pack.

## Owner approval required

The owner must approve these choices before the format freezes.

| Choice | Proposed position |
| --- | --- |
| Public magic | `WVYENC01`, `WVCK`, `WVDI`, and `WVND` |
| File extension | `.wenc` |
| Media type | `audio/vnd.wavey.encodec` |
| Manifest form | RFC 8785 canonical JSON |
| Embedded manifest | Optional when a durable external copy exists |
| Offset width | 32 bits with a container limit below 4 GiB |
| Local checksum | CRC-32/ISO-HDLC |
| Cryptographic chunk digest | Omit it to protect size |
| Directory entry | One four-byte chunk offset |
| Initial codebook profiles | Four and eight codebooks |
| Neural geometry | Official Meta geometry after the size gate |
| Size gate | No aggregate or full-mix byte increase |
| Salvage conflict | Conceal conflicting ordinals |
| Salvage audio | Zero segment before canonical overlap-add |
| Reference output | Owner must select PCM format and rounding |
| Reference runtime | Owner must select CPU, WASI, or another archival target |
| Resource defaults | Use the limits in this draft |
| Entropy profile | Freeze only after compression and architecture tests |

No implementation should publish Profile 1 bytes before these decisions freeze.

## Local implementation references

- [Current container code](../src/binary.rs)
- [Current entropy loop](../src/ecdc.rs)
- [Current arithmetic coder](../src/arithmetic.rs)
- [Current q8 LM](../src/quantized_lm.rs)
- [Current native adapter](../src/onnx.rs)
- [Current WASM adapter](../src/wasm.rs)
- [Lineage and LM contract](encodec-lineage-and-lm-contract.md)
