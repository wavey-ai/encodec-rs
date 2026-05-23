#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/Users/jamie/wavey.ai/encodec-rs}"
OUT="${OUT:-${ROOT}/dist/wasm-fixed-bundles}"
BINDGEN_TARGET="${BINDGEN_TARGET:-web}"
RUST_TOOLCHAIN="${RUST_TOOLCHAIN:-nightly}"
RUST_WASM_TARGET="${RUST_WASM_TARGET:-wasm32-unknown-unknown}"
BUNDLES="${BUNDLES:-encodec_48khz_6kbps_1333ms encodec_48khz_6kbps_1800ms encodec_48khz_12kbps_1333ms encodec_48khz_12kbps_1800ms}"

cd "$ROOT"

rustup target add "$RUST_WASM_TARGET" --toolchain "$RUST_TOOLCHAIN"

if ! command -v wasm-bindgen >/dev/null 2>&1; then
  version="$(
    python - <<'PY'
from pathlib import Path

text = Path("Cargo.lock").read_text()
lines = text.splitlines()
for i, line in enumerate(lines):
    if line.strip() == 'name = "wasm-bindgen"':
        for j in range(i, min(i + 10, len(lines))):
            item = lines[j].strip()
            if item.startswith("version = "):
                print(item.split('"')[1])
                raise SystemExit
raise SystemExit("wasm-bindgen version not found in Cargo.lock")
PY
  )"
  cargo +"$RUST_TOOLCHAIN" install wasm-bindgen-cli --version "$version"
fi

rm -rf "$OUT" "$ROOT/pkg"
mkdir -p "$OUT/pkg" "$OUT/bundles" "$ROOT/pkg"

cargo +"$RUST_TOOLCHAIN" build \
  --lib \
  --features wasm,ecdc \
  --target "$RUST_WASM_TARGET" \
  --release

crate_name="$(
  python - <<'PY'
from pathlib import Path

text = Path("Cargo.toml").read_text()
in_package = False
for raw in text.splitlines():
    line = raw.strip()
    if line == "[package]":
        in_package = True
        continue
    if line.startswith("[") and line != "[package]":
        in_package = False
    if in_package and line.startswith("name"):
        name = line.split("=", 1)[1].strip().strip('"')
        print(name.replace("-", "_"))
        raise SystemExit
raise SystemExit("package name not found in Cargo.toml")
PY
)"

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

copy_model_asset() {
  local src_dir="$1"
  local dst_dir="$2"
  local model_name="$3"

  if [[ -f "${src_dir}/${model_name}" ]]; then
    cp "${src_dir}/${model_name}" "${dst_dir}/${model_name}"
  elif [[ -f "${src_dir}/${model_name}.parts.json" ]]; then
    cp "${src_dir}/${model_name}.parts.json" "${dst_dir}/${model_name}.parts.json"
    while IFS= read -r part; do
      [[ -n "$part" ]] || continue
      mkdir -p "${dst_dir}/$(dirname "$part")"
      cp "${src_dir}/${part}" "${dst_dir}/${part}"
    done < <(
      python - "${src_dir}/${model_name}.parts.json" <<'PY'
import json
import sys
from pathlib import Path

manifest = json.loads(Path(sys.argv[1]).read_text())
for part in manifest.get("parts", []):
    print(part)
PY
    )
  else
    echo "missing ${src_dir}/${model_name} or ${src_dir}/${model_name}.parts.json" >&2
    exit 1
  fi
}

for bundle in $BUNDLES; do
  src="${ROOT}/onnx-bundles/${bundle}"
  dst="${OUT}/bundles/${bundle}"

  if [[ ! -f "${src}/bundle.json" ]]; then
    echo "missing ${src}/bundle.json" >&2
    exit 1
  fi

  if [[ ! -f "${src}/lm_weights_q8.bin" ]]; then
    echo "missing ${src}/lm_weights_q8.bin" >&2
    exit 1
  fi

  mkdir -p "$dst"
  cp "${src}/bundle.json" "$dst/bundle.json"
  cp "${src}/lm_weights_q8.bin" "$dst/lm_weights_q8.bin"

  encode_model="$(
    python - "${src}/bundle.json" <<'PY'
import json
import sys
from pathlib import Path

bundle = json.loads(Path(sys.argv[1]).read_text())
print(bundle.get("encode_model", "encode_frame.onnx"))
PY
  )"

  copy_model_asset "$src" "$dst" "$encode_model"

  python - "$dst" <<'PY'
import json
import struct
import sys
from pathlib import Path

bundle_dir = Path(sys.argv[1])
bundle = json.loads((bundle_dir / "bundle.json").read_text())
header = struct.unpack("<7I", (bundle_dir / "lm_weights_q8.bin").read_bytes()[8:36])
lm = dict(zip(
    ["dim", "layers", "heads", "codebooks", "cardinality", "frame_length", "past_context"],
    header,
))
manifest = {
    "name": bundle_dir.name,
    "bundleJson": "bundle.json",
    "lmWeights": "lm_weights_q8.bin",
    "encodeModel": bundle.get("encode_model", "encode_frame.onnx"),
    "modelName": bundle.get("model_name"),
    "bandwidthKbps": bundle.get("bandwidth_kbps"),
    "sampleRate": bundle.get("sample_rate"),
    "channels": bundle.get("channels"),
    "segmentSamples": bundle.get("segment_samples"),
    "segmentStride": bundle.get("segment_stride"),
    "frameLength": bundle.get("frame_length"),
    "numCodebooks": bundle.get("num_codebooks"),
    "lm": lm,
}
(bundle_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(json.dumps(manifest, sort_keys=True))
PY
done

python - "$OUT" $BUNDLES <<'PY'
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
names = sys.argv[2:]
manifest = {
    "pkg": "pkg",
    "bundles": [
        json.loads((out / "bundles" / name / "manifest.json").read_text())
        for name in names
    ],
}
(out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(json.dumps(manifest, indent=2))
PY

find "$OUT" -maxdepth 3 -type f | sort
