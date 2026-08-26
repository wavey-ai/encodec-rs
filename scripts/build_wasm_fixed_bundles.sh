#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/Users/jamie/wavey.ai/encodec-rs}"
OUT="${OUT:-${ROOT}/dist/wasm-fixed-bundles}"
BINDGEN_TARGET="${BINDGEN_TARGET:-web}"
RUST_TOOLCHAIN="${RUST_TOOLCHAIN:-nightly}"
RUST_WASM_TARGET="${RUST_WASM_TARGET:-wasm32-unknown-unknown}"
RUST_WASM_TARGET_FEATURES="${RUST_WASM_TARGET_FEATURES:-+simd128}"
PYTHON_BIN="${PYTHON_BIN:-/opt/anaconda3/envs/encodec-export/bin/python}"
BUNDLES="${BUNDLES:-encodec_48khz_3kbps_1333ms encodec_48khz_6kbps_1333ms encodec_48khz_12kbps_1333ms encodec_48khz_12kbps_7cb_1333ms encodec_48khz_24kbps_1333ms encodec_48khz_3kbps_1800ms encodec_48khz_6kbps_1800ms encodec_48khz_12kbps_1800ms encodec_48khz_12kbps_7cb_1800ms encodec_48khz_24kbps_1800ms}"

cd "$ROOT"

case "$OUT" in
  "$ROOT/dist/wasm-fixed-bundles"|"$ROOT/target/"*) ;;
  *)
    echo "refusing to replace unexpected output directory: $OUT" >&2
    exit 1
    ;;
esac

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "missing Python environment: $PYTHON_BIN" >&2
  exit 1
fi

mkdir -p "$ROOT/target"
build_root="$(mktemp -d "$ROOT/target/custom-wasm-bundles.XXXXXX")"
cleanup() {
  rm -rf "$build_root"
}
trap cleanup EXIT

rustup target add "$RUST_WASM_TARGET" --toolchain "$RUST_TOOLCHAIN"

version="$(
  awk '
    /^name = "wasm-bindgen"$/ { found = 1; next }
    found && /^version = / { gsub(/[",]/, "", $3); print $3; exit }
  ' Cargo.lock
)"
if [[ -z "$version" ]]; then
  echo "wasm-bindgen version not found in Cargo.lock" >&2
  exit 1
fi

installed_version=""
if command -v wasm-bindgen >/dev/null 2>&1; then
  installed_version="$(wasm-bindgen --version | awk '{print $2}')"
fi
if [[ "$installed_version" != "$version" ]]; then
  # wasm-bindgen's generated schema must exactly match the Rust crate version.
  cargo +"$RUST_TOOLCHAIN" install wasm-bindgen-cli --version "$version" --force --locked
fi

rm -rf "$OUT" "$ROOT/pkg"
mkdir -p "$OUT/pkg" "$OUT/bundles" "$ROOT/pkg"

WASM_RUSTFLAGS="${RUSTFLAGS:-} -C target-feature=${RUST_WASM_TARGET_FEATURES}"
RUSTFLAGS="$WASM_RUSTFLAGS" cargo +"$RUST_TOOLCHAIN" build \
  --lib \
  --features wasm,ecdc \
  --target "$RUST_WASM_TARGET" \
  --release

# Crate name from [package].name, normalized to the wasm artifact name.
crate_name="$(
  awk '
    /^\[package\]/ { in_pkg = 1; next }
    /^\[/ { in_pkg = 0 }
    in_pkg && /^name[[:space:]]*=/ { gsub(/[",]/, "", $3); gsub(/-/, "_", $3); print $3; exit }
  ' Cargo.toml
)"
if [[ -z "$crate_name" ]]; then
  echo "package name not found in Cargo.toml" >&2
  exit 1
fi

cargo_target_dir="$(
  cargo +"$RUST_TOOLCHAIN" metadata --format-version 1 --no-deps \
    | "$PYTHON_BIN" -c 'import json, sys; print(json.load(sys.stdin)["target_directory"])'
)"
wasm_path="$cargo_target_dir/${RUST_WASM_TARGET}/release/${crate_name}.wasm"

if [[ ! -f "$wasm_path" ]]; then
  echo "missing wasm output: $wasm_path" >&2
  find "$cargo_target_dir/${RUST_WASM_TARGET}/release" -maxdepth 1 -name '*.wasm' -print >&2 || true
  exit 1
fi

wasm-bindgen "$wasm_path" \
  --target "$BINDGEN_TARGET" \
  --out-dir "$ROOT/pkg"

cp "$ROOT/scripts/wasm-pkg-package.json" "$ROOT/pkg/package.json"
cp -R "$ROOT/pkg/." "$OUT/pkg/"
cp "$ROOT/browser-runtime/encodec-ecdc-runtime.js" "$OUT/encodec-ecdc-runtime.js"
cp "$ROOT/browser-runtime/browser-neural-runtime.js" "$OUT/browser-neural-runtime.js"
cp "$ROOT/browser-runtime/custom-encoder-runtime.js" "$OUT/custom-encoder-runtime.js"
cp "$ROOT/browser-runtime/custom-decoder-runtime.js" "$OUT/custom-decoder-runtime.js"
cp "$ROOT/browser-runtime/webgpu-kernel-runtime.js" "$OUT/webgpu-kernel-runtime.js"
cp "$ROOT/browser-runtime/webgpu-encoder-runtime.js" "$OUT/webgpu-encoder-runtime.js"
cp "$ROOT/browser-runtime/webgpu-decoder-runtime.js" "$OUT/webgpu-decoder-runtime.js"
cp "$ROOT/browser-runtime/webgpu-ecdc-decoder-runtime.js" "$OUT/webgpu-ecdc-decoder-runtime.js"

read -r -a bundle_names <<< "$BUNDLES"
if [[ "${#bundle_names[@]}" -eq 0 ]]; then
  echo "BUNDLES must contain at least one bundle" >&2
  exit 1
fi

kernel_bundle="${bundle_names[0]}"
kernel_source="$ROOT/onnx-bundles/$kernel_bundle"
kernel_encoder="$build_root/$kernel_bundle/encoder"
kernel_decoder="$build_root/$kernel_bundle/decoder"

PYTHON_BIN="$PYTHON_BIN" \
  "$ROOT/scripts/build-custom-encoder-wasm.sh" \
  "$kernel_source" \
  "$kernel_encoder"
PYTHON_BIN="$PYTHON_BIN" \
  "$ROOT/scripts/build-custom-decoder-wasm.sh" \
  "$kernel_source" \
  "$kernel_decoder"

for bundle_name in "${bundle_names[@]}"; do
  source_root="$ROOT/onnx-bundles/$bundle_name"
  stage_root="$build_root/$bundle_name"
  encoder_stage="$stage_root/encoder"
  decoder_stage="$stage_root/decoder"
  bundle_out="$OUT/bundles/$bundle_name"

  if [[ ! -f "$source_root/bundle.json" || ! -f "$source_root/lm_weights_q8.bin" ]]; then
    echo "missing source bundle: $source_root" >&2
    exit 1
  fi

  if [[ "$bundle_name" != "$kernel_bundle" ]]; then
    mkdir -p "$encoder_stage" "$decoder_stage"
    "$PYTHON_BIN" \
      "$ROOT/scripts/extract-custom-encoder-stages.py" \
      "$source_root" \
      "$encoder_stage"
    "$PYTHON_BIN" \
      "$ROOT/scripts/extract-custom-decoder-stages.py" \
      "$source_root" \
      "$decoder_stage"
  fi

  mkdir -p "$bundle_out/encoder" "$bundle_out/decoder"
  node "$ROOT/scripts/pack-custom-weights.mjs" \
    "$encoder_stage" \
    "$bundle_out/encoder"
  node "$ROOT/scripts/pack-custom-weights.mjs" \
    "$decoder_stage" \
    "$bundle_out/decoder"

  cp "$kernel_encoder/encodec-encoder.mjs" "$bundle_out/encoder/"
  cp "$kernel_encoder/encodec-encoder.wasm" "$bundle_out/encoder/"
  cp "$kernel_encoder/encodec-encoder-relaxed.mjs" "$bundle_out/encoder/"
  cp "$kernel_encoder/encodec-encoder-relaxed.wasm" "$bundle_out/encoder/"
  cp "$kernel_decoder/encodec-convtranspose.mjs" "$bundle_out/decoder/"
  cp "$kernel_decoder/encodec-convtranspose.wasm" "$bundle_out/decoder/"
  cp "$kernel_decoder/encodec-convtranspose-relaxed.mjs" "$bundle_out/decoder/"
  cp "$kernel_decoder/encodec-convtranspose-relaxed.wasm" "$bundle_out/decoder/"
  cp "$source_root/lm_weights_q8.bin" "$bundle_out/lm_weights_q8.bin"
done

node "$ROOT/scripts/finalize-custom-wasm-bundles.mjs" \
  "$OUT" \
  "${bundle_names[@]}"

find "$OUT" -maxdepth 3 -type f | sort
