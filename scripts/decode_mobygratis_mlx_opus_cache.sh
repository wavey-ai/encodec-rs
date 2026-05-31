#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${OUT_DIR:-$ROOT/target/mobygratis-ecdc}"
ECDC_DIR="${ECDC_DIR:-$OUT_DIR/ecdc}"
BUNDLE_DIR="${BUNDLE_DIR:-$ROOT/target/mlx-bundles/encodec_48khz_12kbps_1333ms_mobygratisv0}"
BUNDLE_NAME="${BUNDLE_NAME:-$(basename "$BUNDLE_DIR")}"
RECORD_PROFILE="${RECORD_PROFILE:-single45}"
BUILD_ID="${BUILD_ID:-dev}"
OPUS_BITRATE="${OPUS_BITRATE:-64000}"
OPUS_FRAME_MS="${OPUS_FRAME_MS:-20}"
LIMIT="${LIMIT:-0}"

OPUS_CACHE="$OUT_DIR/.cache/soundkit-opus"
TMP_DIR="$OUT_DIR/.cache/tmp"
LOG_DIR="$OUT_DIR/logs"
STATE_DIR="$OUT_DIR/state"

mkdir -p "$OPUS_CACHE" "$TMP_DIR" "$LOG_DIR" "$STATE_DIR"

if [[ ! -d "$ECDC_DIR" ]]; then
  echo "ECDC_DIR does not exist: $ECDC_DIR" >&2
  exit 1
fi
if [[ ! -d "$BUNDLE_DIR" ]]; then
  echo "MLX bundle does not exist: $BUNDLE_DIR" >&2
  echo "Run scripts/create_mlx_fixed_bundles.sh or set BUNDLE_DIR explicitly." >&2
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
DECODER="$BIN_PATH/EncodecMLXEncode"

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

count=0
cached=0
skipped=0
failed=0

while IFS= read -r -d '' ecdc; do
  count=$((count + 1))
  if [[ "$LIMIT" != "0" && "$count" -gt "$LIMIT" ]]; then
    break
  fi

  base="$(basename "$ecdc")"
  stem="${base%.ecdc}"
  hash="$(shasum -a 256 "$ecdc" | awk '{print $1}')"
  cache_body="$OPUS_CACHE/$stem.bnp"
  meta="$OPUS_CACHE/$stem.json"
  encode_meta="$ECDC_DIR/$stem.json"
  done_marker="$STATE_DIR/$stem.soundkit-opus.done"
  lock_dir="$STATE_DIR/$stem.soundkit-opus.lock"
  log="$LOG_DIR/$stem.soundkit-opus.log"
  tmp_pcm="$TMP_DIR/$stem.f32le.tmp.$$"
  tmp_decode_json="$TMP_DIR/$stem.decode.json.tmp.$$"

  if [[ -s "$cache_body" && -f "$done_marker" && -f "$meta" ]] \
      && python3 - "$meta" "$hash" "$OPUS_BITRATE" "$OPUS_FRAME_MS" <<'PY'
import json, sys
meta, expected_hash, expected_bitrate, expected_frame_ms = sys.argv[1:]
with open(meta, "r", encoding="utf-8") as handle:
    data = json.load(handle)
payload = data.get("payload") or {}
bitrate = int(data.get("bitrate") or payload.get("bitrate") or 0)
frame_ms = int(data.get("frameDurationMs") or payload.get("frameDurationMs") or 0)
sys.exit(0 if (
    data.get("ecdc_sha256") == expected_hash
    and data.get("format") == "soundkit_opus_packets"
    and bitrate == int(expected_bitrate)
    and frame_ms == int(expected_frame_ms)
) else 1)
PY
  then
    skipped=$((skipped + 1))
    echo "skip $base"
    continue
  fi

  if ! mkdir "$lock_dir" 2>/dev/null; then
    echo "locked $base"
    continue
  fi
  trap 'rm -f "$tmp_pcm" "$tmp_decode_json"; rm -rf "$lock_dir"' EXIT

  {
    echo "ecdc=$ecdc"
    echo "cache_body=$cache_body"
    echo "bundle=$BUNDLE_DIR"
    echo "opus_bitrate=$OPUS_BITRATE"
  } >"$log"

  if ! read -r chunk_index chunk_byte_length < <(python3 - "$ecdc" <<'PY'
import struct, sys
path = sys.argv[1]
data = open(path, "rb").read()
if len(data) < 9 or data[:4] != b"ECDC" or data[4] != 0:
    raise SystemExit("invalid ECDC header")
meta_len = struct.unpack(">I", data[5:9])[0]
header_end = 9 + meta_len
if header_end + 8 > len(data):
    raise SystemExit("ECDC has no CRC chunk")
payload_len = struct.unpack(">I", data[header_end:header_end + 4])[0]
chunk_byte_length = 8 + payload_len
if payload_len <= 0 or header_end + chunk_byte_length > len(data):
    raise SystemExit("invalid first ECDC chunk")
print(0, chunk_byte_length)
PY
  ); then
    failed=$((failed + 1))
    echo "failed inspect $base"
    rm -rf "$lock_dir"
    trap - EXIT
    continue
  fi

  echo "decode $base"
  if ! "$DECODER" \
      --decode \
      --bundle "$BUNDLE_DIR" \
      --input "$ecdc" \
      --output "$tmp_pcm" >"$tmp_decode_json" 2>>"$log"; then
    failed=$((failed + 1))
    echo "failed decode $base"
    rm -f "$tmp_pcm" "$tmp_decode_json"
    rm -rf "$lock_dir"
    trap - EXIT
    continue
  fi

  if ! read -r sample_rate channels frames < <(python3 - "$tmp_decode_json" <<'PY'
import json, sys
with open(sys.argv[1], "r", encoding="utf-8") as handle:
    data = json.load(handle)
print(int(data["sample_rate"]), int(data["channels"]), int(data["frames"]))
PY
  ); then
    failed=$((failed + 1))
    echo "failed read decode metadata $base"
    rm -f "$tmp_pcm" "$tmp_decode_json"
    rm -rf "$lock_dir"
    trap - EXIT
    continue
  fi

  tmp_cache="$cache_body.tmp.$$"
  tmp_meta="$meta.tmp.$$"
  echo "soundkit opus $base"
  if cargo run --quiet --features ecdc --example soundkit_opus_packet_cache -- \
      --input-f32le "$tmp_pcm" \
      --output-cache "$tmp_cache" \
      --output-json "$tmp_meta" \
      --source-hash "$hash" \
      --bundle-name "$BUNDLE_NAME" \
      --record-profile "$RECORD_PROFILE" \
      --chunk-index "$chunk_index" \
      --chunk-byte-length "$chunk_byte_length" \
      --sample-rate "$sample_rate" \
      --channels "$channels" \
      --frames "$frames" \
      --bitrate "$OPUS_BITRATE" \
      --frame-duration-ms "$OPUS_FRAME_MS" \
      --build-id "$BUILD_ID" >>"$log" 2>&1; then
    mv "$tmp_cache" "$cache_body"
    python3 - "$tmp_meta" "$meta" "$ecdc" "$hash" "$encode_meta" "$tmp_decode_json" <<'PY'
import json, os, sys
tmp_meta, meta, ecdc, sha, encode_meta, decode_json = sys.argv[1:]
with open(tmp_meta, "r", encoding="utf-8") as handle:
    data = json.load(handle)
encoded = {}
if os.path.exists(encode_meta):
    with open(encode_meta, "r", encoding="utf-8") as handle:
        encoded = json.load(handle)
decoded = {}
if os.path.exists(decode_json):
    with open(decode_json, "r", encoding="utf-8") as handle:
        decoded = json.load(handle)
data.update({
    "ecdc": ecdc,
    "ecdc_sha256": sha,
    "encoded_source": encoded.get("source"),
    "encoded_source_sha256": encoded.get("source_sha256"),
    "encoded_wav_cache": encoded.get("wav_cache"),
    "encoded_batch_size": encoded.get("batch_size"),
    "encoded_chunk_ms": encoded.get("chunk_ms"),
    "decode_s": decoded.get("decode_s"),
    "decode_rtfx": decoded.get("decode_rtfx"),
})
with open(meta, "w", encoding="utf-8") as handle:
    json.dump(data, handle, indent=2, sort_keys=True)
    handle.write("\n")
PY
    rm -f "$tmp_meta" "$tmp_pcm" "$tmp_decode_json"
    date -u +"%Y-%m-%dT%H:%M:%SZ" >"$done_marker"
    cached=$((cached + 1))
  else
    failed=$((failed + 1))
    echo "failed soundkit opus $base"
    rm -f "$tmp_cache" "$tmp_meta" "$tmp_pcm" "$tmp_decode_json"
  fi

  rm -rf "$lock_dir"
  trap - EXIT
done < <(find "$ECDC_DIR" -maxdepth 1 -type f -name '*.ecdc' -print0 | sort -z)

echo "done count=$count cached=$cached skipped=$skipped failed=$failed opus_cache=$OPUS_CACHE"
