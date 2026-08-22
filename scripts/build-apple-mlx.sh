#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
apple_root="${repo_root}/apple"
mlx_checkout="${apple_root}/.build/checkouts/mlx-swift"
metal_source_root="${mlx_checkout}/Source/Cmlx/mlx-generated/metal"

if [[ ! -d "${metal_source_root}" ]]; then
  echo "MLX sources are missing. Swift Package Manager must resolve the package first." >&2
  exit 1
fi

cargo build \
  --manifest-path "${repo_root}/Cargo.toml" \
  --release \
  --features ecdc

swift build \
  --package-path "${apple_root}" \
  --configuration release

bin_path="$(
  swift build \
    --package-path "${apple_root}" \
    --configuration release \
    --show-bin-path
)"
air_dir="$(mktemp -d "${TMPDIR:-/tmp}/encodec-mlx-metal.XXXXXX")"
trap 'rm -rf "${air_dir}"' EXIT

metal_flags=(
  -x metal
  -Wall
  -Wextra
  -fno-fast-math
  -Wno-c++17-extensions
  -Wno-c++20-extensions
  -mmacosx-version-min=14.0
  -I "${metal_source_root}"
)

air_files=()
index=0
while IFS= read -r source; do
  air_file="${air_dir}/kernel-${index}.air"
  xcrun -sdk macosx metal \
    "${metal_flags[@]}" \
    -c "${source}" \
    -o "${air_file}"
  air_files+=("${air_file}")
  index=$((index + 1))
done < <(find "${metal_source_root}" -type f -name '*.metal' -print | sort)

if [[ ${#air_files[@]} -eq 0 ]]; then
  echo "MLX did not provide generated Metal sources." >&2
  exit 1
fi

metallib="${bin_path}/mlx.metallib"
xcrun -sdk macosx metallib "${air_files[@]}" -o "${metallib}"

while IFS= read -r test_bin_dir; do
  cp "${metallib}" "${test_bin_dir}/mlx.metallib"
done < <(find "${bin_path}" -type d -path '*.xctest/Contents/MacOS' -print)

echo "Built ${bin_path}/EncodecMLXEncode"
echo "Built ${metallib} from ${#air_files[@]} MLX Metal sources"
