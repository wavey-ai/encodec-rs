#!/usr/bin/env bash
set -euo pipefail

repo_id="${ENCODEC_BUNDLE_REPO:-wavey-ai/encodec-rs-onnx-bundles}"
revision="${ENCODEC_BUNDLE_REVISION:-main}"
base_url="https://huggingface.co/${repo_id}/resolve/${revision}"

files=(
  "SHA256SUMS"
  "onnx-bundles/encodec_48khz_6kbps/bundle.json"
  "onnx-bundles/encodec_48khz_6kbps/decode_frame.onnx"
  "onnx-bundles/encodec_48khz_6kbps/decode_frame.onnx.parts.json"
  "onnx-bundles/encodec_48khz_6kbps/encode_frame.onnx"
  "onnx-bundles/encodec_48khz_6kbps/encode_frame.onnx.parts.json"
  "onnx-bundles/encodec_48khz_6kbps/lm_weights_q8.bin"
  "onnx-bundles/encodec_48khz_12kbps/bundle.json"
  "onnx-bundles/encodec_48khz_12kbps/decode_frame.onnx"
  "onnx-bundles/encodec_48khz_12kbps/decode_frame.onnx.parts.json"
  "onnx-bundles/encodec_48khz_12kbps/encode_frame.onnx"
  "onnx-bundles/encodec_48khz_12kbps/encode_frame.onnx.parts.json"
  "onnx-bundles/encodec_48khz_12kbps/lm_weights_q8.bin"
)

for file in "${files[@]}"; do
  mkdir -p "$(dirname "$file")"
  curl -fL --retry 3 --retry-delay 2 -o "$file" "${base_url}/${file}"
done

if command -v shasum >/dev/null 2>&1; then
  shasum -a 256 -c SHA256SUMS
else
  echo "shasum not found; skipped checksum verification" >&2
fi
