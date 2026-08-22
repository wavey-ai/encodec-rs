#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${TARGET_BUILD_DIR:-}" || -z "${UNLOCALIZED_RESOURCES_FOLDER_PATH:-}" ]]; then
  echo "error: the Xcode resource environment is incomplete" >&2
  exit 1
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
device_root="$(cd "${script_dir}/.." && pwd)"
repo_root="$(cd "${device_root}/../.." && pwd)"
resource_root="${TARGET_BUILD_DIR}/${UNLOCALIZED_RESOURCES_FOLDER_PATH}"
model_name="encodec_48khz_12kbps_1333ms"
model_source="${repo_root}/target/mlx-bundles/${model_name}"
track_source="${ENCODEC_DEVICE_BENCH_WAV:-}"

if [[ ! -d "${model_source}" ]]; then
  echo "error: missing MLX model bundle at ${model_source}" >&2
  exit 1
fi
if [[ -z "${track_source}" || ! -f "${track_source}" ]]; then
  echo "error: set ENCODEC_DEVICE_BENCH_WAV to the Confirmation WAV path" >&2
  exit 1
fi

mkdir -p "${resource_root}/mlx-bundles/${model_name}"
rsync -a --delete \
  "${model_source}/" \
  "${resource_root}/mlx-bundles/${model_name}/"
cp "${track_source}" "${resource_root}/confirmation.wav"
