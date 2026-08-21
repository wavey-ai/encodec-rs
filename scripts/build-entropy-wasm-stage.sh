#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: scripts/build-entropy-wasm-stage.sh <stage-name>" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
VENDOR_WASM_ROOT="${VENDOR_WASM_ROOT:-${ROOT}/../vin.yl.vendor/wasm}"
STAGE_NAME="$1"
BUILD_ROOT="${ROOT}/target/wasm-entropy-optimization-build"
STAGE_ROOT="${ROOT}/target/performance/entropy-optimization/${STAGE_NAME}/wasm"
PKG_ROOT="${STAGE_ROOT}/encodec-rs/pkg"
RAW_WASM="${BUILD_ROOT}/wasm32-unknown-unknown/release/encodec_rs.wasm"

mkdir -p "${PKG_ROOT}" "${STAGE_ROOT}/encodec-rs"

STAGE_RUSTFLAGS="${RUSTFLAGS:-} -C target-feature=+simd128"
CARGO_TARGET_DIR="${BUILD_ROOT}" \
RUSTFLAGS="${STAGE_RUSTFLAGS}" \
  cargo +nightly build \
    --locked \
    --lib \
    --features wasm,ecdc \
    --target wasm32-unknown-unknown \
    --release

wasm-bindgen "${RAW_WASM}" \
  --target web \
  --out-dir "${PKG_ROOT}"
printf '%s\n' '{ "type": "module" }' > "${PKG_ROOT}/package.json"

ln -sfn "${VENDOR_WASM_ROOT}/encodec-rs/bundles" \
  "${STAGE_ROOT}/encodec-rs/bundles"
ln -sfn "${VENDOR_WASM_ROOT}/onnxruntime-web" \
  "${STAGE_ROOT}/onnxruntime-web"

shasum -a 256 \
  "${PKG_ROOT}/encodec_rs.js" \
  "${PKG_ROOT}/encodec_rs_bg.wasm"
