#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: scripts/build-custom-decoder-wasm.sh <bundle-dir> <output-dir>" >&2
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
  "${SCRIPT_DIR}/extract-custom-decoder-stages.py" \
  "${BUNDLE_DIR}" \
  "${OUTPUT_DIR}"

KERNEL_FLAGS=(
  "${SCRIPT_DIR}/kernels/encodec_convtranspose.c"
  "${SCRIPT_DIR}/kernels/encodec_encoder.c"
  -O3
  -s MODULARIZE=1
  -s EXPORT_ES6=1
  -s ENVIRONMENT=web,node
  -s ALLOW_MEMORY_GROWTH=1
  -s INITIAL_MEMORY=134217728
  -s 'EXPORTED_FUNCTIONS=["_malloc","_free","_pack_conv_transpose1d_weights","_conv_transpose1d_phase_scalar","_conv_transpose1d_phase_simd","_conv_transpose1d_phase_simd_8x4","_conv_transpose1d_phase_simd_8x8","_conv_transpose1d_phase_simd_8x8_nhwc","_crop_nhwc","_pack_conv1d_nhwc_weights_8","_pack_linear_weights_8","_reflect_pad_nhwc","_reflect_pad_elu_nhwc","_group_norm_nhwc_in_place","_elu_nhwc_in_place","_add_nhwc_in_place","_add_elu_nhwc_in_place","_conv1d_nhwc_simd_8x8","_lstm_layer_simd_64","_rvq_decode_codes_nhwc","_nhwc_to_nct","_compact_nhwc_channels","_scale_nhwc_to_nct"]'
  -s 'EXPORTED_RUNTIME_METHODS=["HEAPF32","HEAPU16"]'
)

EM_NODE_JS="${EM_NODE_BIN}" "${EMCC_BIN}" \
  "${KERNEL_FLAGS[@]}" \
  -msimd128 \
  -ffp-contract=off \
  -o "${OUTPUT_DIR}/encodec-convtranspose.mjs"

EM_NODE_JS="${EM_NODE_BIN}" "${EMCC_BIN}" \
  "${KERNEL_FLAGS[@]}" \
  -mrelaxed-simd \
  -ffp-contract=fast \
  -o "${OUTPUT_DIR}/encodec-convtranspose-relaxed.mjs"

shasum -a 256 \
  "${OUTPUT_DIR}/metadata.json" \
  "${OUTPUT_DIR}/encodec-convtranspose.mjs" \
  "${OUTPUT_DIR}/encodec-convtranspose.wasm" \
  "${OUTPUT_DIR}/encodec-convtranspose-relaxed.mjs" \
  "${OUTPUT_DIR}/encodec-convtranspose-relaxed.wasm"
