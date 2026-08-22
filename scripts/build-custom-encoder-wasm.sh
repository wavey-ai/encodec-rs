#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: scripts/build-custom-encoder-wasm.sh <bundle-dir> <output-dir>" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
BUNDLE_DIR="$(cd "$1" && pwd)"
OUTPUT_DIR="$2"
PYTHON_BIN="${PYTHON_BIN:-python3}"
EMCC_BIN="${EMCC_BIN:-emcc}"
EM_NODE_BIN="${EM_NODE_BIN:-$(command -v node)}"

mkdir -p "${OUTPUT_DIR}"
OUTPUT_DIR="$(cd "${OUTPUT_DIR}" && pwd)"

"${PYTHON_BIN}" \
  "${SCRIPT_DIR}/extract-custom-encoder-stages.py" \
  "${BUNDLE_DIR}" \
  "${OUTPUT_DIR}"

KERNEL_FLAGS=(
  "${SCRIPT_DIR}/kernels/encodec_encoder.c"
  -O3
  -s MODULARIZE=1
  -s EXPORT_ES6=1
  -s ENVIRONMENT=web,node
  -s ALLOW_MEMORY_GROWTH=1
  -s INITIAL_MEMORY=268435456
  -s 'EXPORTED_FUNCTIONS=["_malloc","_free","_pack_conv1d_nhwc_weights_8","_pack_linear_weights_8","_reflect_pad_planar_to_nhwc","_reflect_pad_nhwc","_reflect_pad_elu_nhwc","_normalize_audio_planar_to_nhwc","_group_norm_nhwc_in_place","_elu_nhwc_in_place","_add_nhwc_in_place","_conv1d_nhwc_scalar","_conv1d_nhwc_simd_8x8","_lstm_layer_simd_64","_rvq_encode_simd_8"]'
  -s 'EXPORTED_RUNTIME_METHODS=["HEAPF32"]'
)

EM_NODE_JS="${EM_NODE_BIN}" "${EMCC_BIN}" \
  "${KERNEL_FLAGS[@]}" \
  -msimd128 \
  -ffp-contract=off \
  -o "${OUTPUT_DIR}/encodec-encoder.mjs"

EM_NODE_JS="${EM_NODE_BIN}" "${EMCC_BIN}" \
  "${KERNEL_FLAGS[@]}" \
  -mrelaxed-simd \
  -ffp-contract=fast \
  -o "${OUTPUT_DIR}/encodec-encoder-relaxed.mjs"

shasum -a 256 \
  "${OUTPUT_DIR}/metadata.json" \
  "${OUTPUT_DIR}/encodec-encoder.mjs" \
  "${OUTPUT_DIR}/encodec-encoder.wasm" \
  "${OUTPUT_DIR}/encodec-encoder-relaxed.mjs" \
  "${OUTPUT_DIR}/encodec-encoder-relaxed.wasm"
