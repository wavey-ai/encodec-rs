#!/usr/bin/env bash
set -euo pipefail

repo_id="${ENCODEC_BUNDLE_REPO:-wavey-ai/encodec-rs-onnx-bundles}"
revision="${ENCODEC_BUNDLE_REVISION:-main}"
base_url="https://huggingface.co/${repo_id}/resolve/${revision}"

base_bundles=(
  "encodec_48khz_6kbps"
  "encodec_48khz_12kbps"
)

fixed_bundles=(
  "encodec_48khz_6kbps_1000ms"
  "encodec_48khz_6kbps_1333ms"
  "encodec_48khz_6kbps_1800ms"
  "encodec_48khz_12kbps_1000ms"
  "encodec_48khz_12kbps_1333ms"
  "encodec_48khz_12kbps_1800ms"
  "encodec_48khz_12kbps_1333ms_mobygratisv0"
  "encodec_48khz_12kbps_1800ms_mobygratisv0"
)

files=("SHA256SUMS")

for bundle in "${base_bundles[@]}"; do
  files+=(
    "onnx-bundles/$bundle/bundle.json"
    "onnx-bundles/$bundle/decode_frame.onnx"
    "onnx-bundles/$bundle/decode_frame.onnx.parts.json"
    "onnx-bundles/$bundle/encode_frame.onnx"
    "onnx-bundles/$bundle/encode_frame.onnx.parts.json"
    "onnx-bundles/$bundle/lm_weights_q8.bin"
  )
done

for bundle in "${fixed_bundles[@]}"; do
  files+=(
    "onnx-bundles/$bundle/bundle.json"
    "onnx-bundles/$bundle/decode_frame.onnx"
    "onnx-bundles/$bundle/encode_frame.onnx"
    "onnx-bundles/$bundle/lm_weights_q8.bin"
  )
done

for file in "${files[@]}"; do
  mkdir -p "$(dirname "$file")"
  curl -fL --retry 3 --retry-delay 2 -o "$file" "${base_url}/${file}"
done

if command -v shasum >/dev/null 2>&1; then
  shasum -a 256 -c SHA256SUMS
else
  echo "shasum not found; skipped checksum verification" >&2
fi
