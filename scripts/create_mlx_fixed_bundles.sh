#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ONNX_ROOT="${ONNX_ROOT:-$ROOT/onnx-bundles}"
OUT_ROOT="${OUT_ROOT:-$ROOT/target/mlx-bundles}"
BANDWIDTHS="${BANDWIDTHS:-6 12}"
CHUNKS="${CHUNKS:-1333 1800}"
INCLUDE_MOBYGRATIS_V0="${INCLUDE_MOBYGRATIS_V0:-1}"
PYTHON="${PYTHON:-python3}"

q8_lm_window_frame_length() {
  "$PYTHON" - "$1" <<'PY'
import struct
import sys
from pathlib import Path

weights = Path(sys.argv[1]).read_bytes()
if len(weights) < 36 or weights[:8] != b"ELMQ0001":
    raise SystemExit(f"invalid q8 LM weights: {sys.argv[1]}")
print(struct.unpack_from("<7I", weights, 8)[5])
PY
}

mark_mlx_fixed_bundle() {
  local bundle_json="$1"
  local q8_weights="$2"
  local lm_window
  lm_window="$(q8_lm_window_frame_length "$q8_weights")"
  "$PYTHON" - "$bundle_json" "$lm_window" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
lm_window = int(sys.argv[2])
bundle = json.loads(path.read_text())
bundle["bitneedle_mlx_fixed_chunk"] = True
bundle["bitneedle_lm_window_frame_length"] = lm_window
path.write_text(json.dumps(bundle, indent=2) + "\n")
PY
}

export_mlx_fixed_bundle() {
  local src="$1"
  local dst="$2"
  if [[ ! -f "$src/bundle.json" ]]; then
    echo "missing ONNX bundle: $src" >&2
    exit 1
  fi
  "$PYTHON" "$ROOT/scripts/export-mlx-frame-archive.py" "$src" "$dst"
  mark_mlx_fixed_bundle "$dst/bundle.json" "$dst/lm_weights_q8.bin"
}

for bandwidth in $BANDWIDTHS; do
  for chunk in $CHUNKS; do
    case "$chunk" in
      1333|1333ms)
        suffix="1333ms"
        ;;
      1800|1800ms)
        suffix="1800ms"
        ;;
      *)
        echo "unsupported chunk preset: $chunk" >&2
        exit 1
        ;;
    esac

    name="encodec_48khz_${bandwidth}kbps_${suffix}"
    export_mlx_fixed_bundle "$ONNX_ROOT/$name" "$OUT_ROOT/$name"
    echo "created $OUT_ROOT/$name"
  done
done

if [[ "$INCLUDE_MOBYGRATIS_V0" != "0" ]]; then
  for chunk in $CHUNKS; do
    case "$chunk" in
      1333|1333ms) suffix="1333ms" ;;
      1800|1800ms) suffix="1800ms" ;;
      *) continue ;;
    esac
    name="encodec_48khz_12kbps_${suffix}_mobygratisv0"
    if [[ -f "$ONNX_ROOT/$name/bundle.json" ]]; then
      export_mlx_fixed_bundle "$ONNX_ROOT/$name" "$OUT_ROOT/$name"
      echo "created $OUT_ROOT/$name"
    fi
  done
fi
