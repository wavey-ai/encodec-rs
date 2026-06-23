#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/Users/jamie/wavey.ai/encodec-rs}"
OUT="${OUT:-${ROOT}/dist/wasm-fixed-bundles}"
BINDGEN_TARGET="${BINDGEN_TARGET:-web}"
RUST_TOOLCHAIN="${RUST_TOOLCHAIN:-nightly}"
RUST_WASM_TARGET="${RUST_WASM_TARGET:-wasm32-unknown-unknown}"
BUNDLES="${BUNDLES:-encodec_48khz_6kbps_1333ms_guard10 encodec_48khz_6kbps_1800ms_guard10 encodec_48khz_12kbps_1333ms_guard10 encodec_48khz_12kbps_1800ms_guard10}"

cd "$ROOT"

rustup target add "$RUST_WASM_TARGET" --toolchain "$RUST_TOOLCHAIN"

if ! command -v wasm-bindgen >/dev/null 2>&1; then
  # Pin wasm-bindgen-cli to the locked wasm-bindgen version from Cargo.lock.
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
  cargo +"$RUST_TOOLCHAIN" install wasm-bindgen-cli --version "$version"
fi

rm -rf "$OUT" "$ROOT/pkg"
mkdir -p "$OUT/pkg" "$OUT/bundles" "$ROOT/pkg"

cargo +"$RUST_TOOLCHAIN" build \
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

wasm_path="$ROOT/target/${RUST_WASM_TARGET}/release/${crate_name}.wasm"

if [[ ! -f "$wasm_path" ]]; then
  echo "missing wasm output: $wasm_path" >&2
  find "$ROOT/target/${RUST_WASM_TARGET}/release" -maxdepth 1 -name '*.wasm' -print >&2 || true
  exit 1
fi

wasm-bindgen "$wasm_path" \
  --target "$BINDGEN_TARGET" \
  --out-dir "$ROOT/pkg"

cp -R "$ROOT/pkg/." "$OUT/pkg/"
cp "$ROOT/browser-runtime/encodec-ecdc-runtime.js" "$OUT/encodec-ecdc-runtime.js"

# Assemble each bundle (copy models + weights, write per-bundle and top-level
# manifests) with the native `fix-bundles` subcommand. The only runtime
# dependency is the encodec-rs crate itself — no python.
cargo +"$RUST_TOOLCHAIN" run \
  --no-default-features \
  --features ecdc \
  --bin encodec-rs \
  -- fix-bundles \
  --out-dir "$OUT" \
  --onnx-bundles-dir "$ROOT/onnx-bundles" \
  $BUNDLES

find "$OUT" -maxdepth 3 -type f | sort
