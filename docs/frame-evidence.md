# Frame and LM Evidence Interfaces

Status: qualification-only.

These interfaces record values before Profile 1 container framing.

They support exact comparisons between codec providers.

They do not define Profile 1 bytes.

## Frame command

Run this command from the repository root:

```sh
target/release/encodec-rs onnx-encode-evidence \
  onnx-bundles/encodec_48khz_6kbps_1333ms \
  input.wav \
  evidence/fixed-1333-6kbps \
  --batch-size 8
```

Use `--coreml`, `--cuda`, or `--tensorrt` when needed.

The input WAV sample rate and channel count MUST match the bundle.

Use `--true-variable-tail` only with a model that accepts a dynamic final input.

The fixed-shape bundles reject this option.

## Frame manifest

The command writes a schema version 2 `manifest.json` file.

The schema name is `wavey.encodec.frame-evidence`.

The manifest sets `qualification_only` to `true`.

It sets `profile_container` to `false` and `wire_format` to `none`.

It records the provider, bundle metadata, input digest, geometry, and batch size.

It also records the LM hash, LM version, temperature bits, probability scale, and arithmetic minimum range.

The `segments` array uses zero-based indices in encode order.

Each segment records:

- Source offset and owned sample count
- Actual model input length
- Code frame count
- Tensor shapes and file names
- Scale bits
- Entropy byte count
- SHA-256 values for each file
- Exact code recovery status

The top-level digests hash concatenated segment bytes in segment order.

## Frame files

Each segment writes these files:

- `segment-NNNNNN.model-input.f32le`
- `segment-NNNNNN.codes.i64le`
- `segment-NNNNNN.recovered-codes.i64le`
- `segment-NNNNNN.scale.f32le`
- `segment-NNNNNN.entropy.bin`

Model input uses planar IEEE 754 binary32 values in little-endian order.

Codes use signed 64-bit values in little-endian order.

Recovered codes use the same layout.

Scale evidence uses IEEE 754 binary32 values in little-endian order.

Entropy files contain the raw arithmetic payload.

They do not contain the scale or container framing.

`codebook-order.u32le` stores the zero-based codebook order.

It uses unsigned 32-bit values in little-endian order.

## LM command

Use the LM command for a canonical fixed-code vector:

```sh
target/release/encodec-rs onnx-lm-evidence \
  onnx-bundles/encodec_48khz_6kbps_1333ms \
  evidence/lm-6kbps \
  --steps 203
```

Set `--steps 0` to use the bundle frame length.

The command uses this deterministic symbol rule:

```text
symbol[codebook,time] = (time * 17 + codebook * 31) % cardinality
```

It uses scale bits `0x3f123456`.

It writes the source codes, recovered codes, scale, framed payload, raw entropy, and codebook order.

Its `wavey.encodec.lm-evidence` manifest records every file digest and the exact recovery result.

It also records the BOS rule, next-input rule, and iteration order.

## Limits

These interfaces do not prove parity with Meta EnCodec.

They do not run the full fixed-code pattern and length matrix.

They do not prove the true variable-tail path without a dynamic model bundle.

They do not create or validate a Profile 1 container.

They do not replace cross-provider, cross-architecture, quality, or listening tests.
