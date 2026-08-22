#!/usr/bin/env bash
set -euo pipefail

if [[ "${ACTION:-build}" == "clean" ]]; then
  exit 0
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
device_root="$(cd "${script_dir}/.." && pwd)"
repo_root="$(cd "${device_root}/../.." && pwd)"
library_dir="${device_root}/RustLib/${PLATFORM_NAME}"
target_root="${device_root}/build/rust-target"

case "${PLATFORM_NAME}:${ARCHS}" in
  iphoneos:arm64)
    rust_target="aarch64-apple-ios"
    ;;
  iphonesimulator:arm64)
    rust_target="aarch64-apple-ios-sim"
    ;;
  *)
    echo "error: unsupported device benchmark target ${PLATFORM_NAME}/${ARCHS}" >&2
    exit 1
    ;;
esac

mkdir -p "${library_dir}" "${target_root}"
if command -v rustup >/dev/null 2>&1; then
  rustup target add "${rust_target}" >/dev/null
fi

CARGO_TARGET_DIR="${target_root}" cargo rustc \
  --manifest-path "${repo_root}/Cargo.toml" \
  --release \
  --target "${rust_target}" \
  --features ecdc \
  --lib \
  -- \
  --crate-type staticlib

cp \
  "${target_root}/${rust_target}/release/deps/libencodec_rs.a" \
  "${library_dir}/libencodec_rs.a"
