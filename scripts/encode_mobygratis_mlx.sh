#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SOURCE_DIR="${SOURCE_DIR:-$ROOT/../bitneedle/mobygratis}"
OUT_DIR="${OUT_DIR:-$ROOT/target/mobygratis-ecdc}"
BUNDLE_DIR="${BUNDLE_DIR:-$ROOT/target/mlx-bundles/encodec_48khz_12kbps}"
BATCH_SIZE="${BATCH_SIZE:-8}"
CHUNK_MS="${CHUNK_MS:-1333.333333}"
LIMIT="${LIMIT:-0}"

WAV_CACHE="$OUT_DIR/.cache/wav"
LOG_DIR="$OUT_DIR/logs"
ECDC_DIR="$OUT_DIR/ecdc"
STATE_DIR="$OUT_DIR/state"

mkdir -p "$WAV_CACHE" "$LOG_DIR" "$ECDC_DIR" "$STATE_DIR"

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "ffmpeg is required to decode source audio" >&2
  exit 1
fi

echo "building Rust release dylib"
(cd "$ROOT" && cargo build --release --features ecdc >/dev/null)

echo "building EncodecMLXEncode"
BIN_PATH="$(
  cd "$ROOT/apple"
  swift build -c release --product EncodecMLXEncode --show-bin-path
)"
(cd "$ROOT/apple" && swift build -c release --product EncodecMLXEncode >/dev/null)
ENCODER="$BIN_PATH/EncodecMLXEncode"

METALLIB_SOURCE="${MLX_METALLIB:-}"
if [[ -z "$METALLIB_SOURCE" ]]; then
  METALLIB_SOURCE="$(
    find "$HOME/Library/Developer/Xcode/DerivedData" \
      -path '*/mlx-swift_Cmlx.bundle/*/default.metallib' \
      -o -path '*/mlx-swift_Cmlx.bundle/default.metallib' 2>/dev/null \
      | head -n 1 || true
  )"
fi
if [[ -n "$METALLIB_SOURCE" && -f "$METALLIB_SOURCE" ]]; then
  mkdir -p "$BIN_PATH/Resources"
  cp "$METALLIB_SOURCE" "$BIN_PATH/mlx.metallib"
  cp "$METALLIB_SOURCE" "$BIN_PATH/Resources/default.metallib"
else
  echo "warning: could not find MLX default.metallib; set MLX_METALLIB=/path/default.metallib if MLX fails" >&2
fi

export DYLD_LIBRARY_PATH="$ROOT/target/release/deps:$ROOT/target/release:$ROOT/target/debug/deps:$ROOT/target/debug:${DYLD_LIBRARY_PATH:-}"

slugify() {
  local value="$1"
  value="${value%.*}"
  value="$(printf '%s' "$value" | tr '[:upper:]' '[:lower:]' | sed -E 's/[^a-z0-9]+/-/g; s/^-+//; s/-+$//')"
  if [[ -z "$value" ]]; then
    value="track"
  fi
  printf '%s' "$value"
}

json_escape() {
  python3 -c 'import json,sys; print(json.dumps(sys.argv[1]))' "$1"
}

count=0
encoded=0
skipped=0
failed=0

while IFS= read -r -d '' src; do
  count=$((count + 1))
  if [[ "$LIMIT" != "0" && "$count" -gt "$LIMIT" ]]; then
    break
  fi

  base="$(basename "$src")"
  slug="$(slugify "$base")"
  hash="$(shasum -a 256 "$src" | awk '{print $1}')"
  short_hash="${hash:0:12}"
  stem="$slug-$short_hash"
  wav="$WAV_CACHE/$stem.wav"
  ecdc="$ECDC_DIR/$stem.ecdc"
  meta="$ECDC_DIR/$stem.json"
  done_marker="$STATE_DIR/$stem.done"
  lock_dir="$STATE_DIR/$stem.lock"
  log="$LOG_DIR/$stem.log"

  if [[ -s "$ecdc" && -f "$done_marker" ]]; then
    skipped=$((skipped + 1))
    echo "skip $base"
    continue
  fi

  if ! mkdir "$lock_dir" 2>/dev/null; then
    echo "locked $base"
    continue
  fi
  trap 'rm -rf "$lock_dir"' EXIT

  {
    echo "source=$src"
    echo "wav=$wav"
    echo "ecdc=$ecdc"
    echo "bundle=$BUNDLE_DIR"
  } >"$log"

  if [[ ! -s "$wav" ]]; then
    echo "decode $base"
    if ! ffmpeg -hide_banner -loglevel error -y -i "$src" -ac 2 -ar 48000 -sample_fmt s16 "$wav" >>"$log" 2>&1; then
      failed=$((failed + 1))
      echo "failed decode $base"
      rm -rf "$lock_dir"
      trap - EXIT
      continue
    fi
  fi

  tmp_ecdc="$ecdc.tmp.$$"
  tmp_json="$meta.tmp.$$"
  echo "encode $base"
  if "$ENCODER" \
      --bundle "$BUNDLE_DIR" \
      --input "$wav" \
      --output "$tmp_ecdc" \
      --batch-size "$BATCH_SIZE" \
      --chunk-ms "$CHUNK_MS" >"$tmp_json" 2>>"$log"; then
    mv "$tmp_ecdc" "$ecdc"
    python3 - "$tmp_json" "$meta" "$src" "$wav" "$hash" "$BUNDLE_DIR" "$BATCH_SIZE" "$CHUNK_MS" <<'PY'
import json, sys
tmp_json, meta, src, wav, sha, bundle, batch, chunk_ms = sys.argv[1:]
with open(tmp_json, "r", encoding="utf-8") as handle:
    data = json.load(handle)
data.update({
    "source": src,
    "wav_cache": wav,
    "source_sha256": sha,
    "bundle": bundle,
    "batch_size": int(batch),
    "chunk_ms": float(chunk_ms),
})
with open(meta, "w", encoding="utf-8") as handle:
    json.dump(data, handle, indent=2, sort_keys=True)
    handle.write("\n")
PY
    rm -f "$tmp_json"
    date -u +"%Y-%m-%dT%H:%M:%SZ" >"$done_marker"
    encoded=$((encoded + 1))
  else
    failed=$((failed + 1))
    echo "failed encode $base"
    rm -f "$tmp_ecdc" "$tmp_json"
  fi

  rm -rf "$lock_dir"
  trap - EXIT
done < <(find "$SOURCE_DIR" -maxdepth 1 -type f \( -iname '*.mp3' -o -iname '*.wav' -o -iname '*.flac' -o -iname '*.m4a' \) -print0 | sort -z)

echo "done count=$count encoded=$encoded skipped=$skipped failed=$failed out=$OUT_DIR"
