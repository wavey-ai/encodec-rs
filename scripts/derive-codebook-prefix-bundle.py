#!/usr/bin/env python3
"""Derive a lower-codebook fixed bundle from a canonical RVQ prefix."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import struct
from pathlib import Path


LM_MAGIC = b"ELMQ0001"
LM_HEADER_U32S = 7


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--codebooks", type=int, required=True)
    parser.add_argument("--profile-bandwidth-kbps", type=float, default=None)
    return parser.parse_args()


def quantized_linear_bytes(rows: int, columns: int) -> int:
    return rows * 4 + rows * columns


def transformer_prefix_end(header: tuple[int, ...]) -> int:
    dim, layers, _heads, _codebooks, _cardinality, frame_length, _past_context = header
    hidden_dim = dim * 4
    position = len(LM_MAGIC) + LM_HEADER_U32S * 4
    position += dim * 4
    position += dim * 4
    position += frame_length * dim * 4
    layer_bytes = (
        quantized_linear_bytes(3 * dim, dim)
        + 3 * dim * 4
        + quantized_linear_bytes(dim, dim)
        + dim * 4
        + quantized_linear_bytes(hidden_dim, dim)
        + hidden_dim * 4
        + quantized_linear_bytes(dim, hidden_dim)
        + dim * 4
        + 4 * dim * 4
    )
    return position + layers * layer_bytes


def truncate_lm(source: Path, output: Path, target_codebooks: int) -> dict[str, int]:
    data = source.read_bytes()
    if data[: len(LM_MAGIC)] != LM_MAGIC:
        raise ValueError(f"{source} is not an ELMQ0001 q8 LM weight file")
    header = struct.unpack_from(f"<{LM_HEADER_U32S}I", data, len(LM_MAGIC))
    dim, layers, heads, source_codebooks, cardinality, frame_length, past_context = header
    if not 0 < target_codebooks < source_codebooks:
        raise ValueError(
            f"target codebooks must be in 1..{source_codebooks - 1}, got {target_codebooks}"
        )

    common_end = transformer_prefix_end(header)
    embedding_bytes = (cardinality + 1) * dim * 4
    output_bytes = quantized_linear_bytes(cardinality, dim) + cardinality * 4
    source_output_start = common_end + source_codebooks * embedding_bytes
    expected_bytes = source_output_start + source_codebooks * output_bytes
    if len(data) != expected_bytes:
        raise ValueError(
            f"q8 LM size does not match its header: {len(data)} != {expected_bytes}"
        )

    target_header = (
        dim,
        layers,
        heads,
        target_codebooks,
        cardinality,
        frame_length,
        past_context,
    )
    derived = bytearray(LM_MAGIC)
    derived.extend(struct.pack(f"<{LM_HEADER_U32S}I", *target_header))
    header_end = len(LM_MAGIC) + LM_HEADER_U32S * 4
    derived.extend(data[header_end:common_end])
    derived.extend(data[common_end : common_end + target_codebooks * embedding_bytes])
    derived.extend(
        data[source_output_start : source_output_start + target_codebooks * output_bytes]
    )
    output.write_bytes(derived)
    return {
        "sourceCodebooks": source_codebooks,
        "targetCodebooks": target_codebooks,
        "sourceBytes": len(data),
        "targetBytes": len(derived),
    }


def link_or_copy(source: Path, output: Path) -> None:
    try:
        os.link(source, output)
    except OSError:
        shutil.copy2(source, output)


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    bundle_path = source / "bundle.json"
    bundle = json.loads(bundle_path.read_text())
    source_codebooks = int(bundle["num_codebooks"])
    if not 0 < args.codebooks < source_codebooks:
        raise ValueError(
            f"--codebooks must be in 1..{source_codebooks - 1}, got {args.codebooks}"
        )
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"output bundle is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    for key in ("encode_model", "decode_model"):
        name = bundle[key]
        link_or_copy(source / name, output / name)

    lm_name = bundle["lm_quant_weight_model"]
    lm_report = truncate_lm(source / lm_name, output / lm_name, args.codebooks)
    # The exact 48 kHz model rate is 150 latent frames per second.
    raw_kbps = args.codebooks * 1.5
    bundle["num_codebooks"] = args.codebooks
    bundle["codebook_prefix_source"] = source.name
    bundle["nominal_codebook_kbps"] = raw_kbps
    if args.profile_bandwidth_kbps is not None:
        bundle["profile_bandwidth_kbps"] = args.profile_bandwidth_kbps
    (output / "bundle.json").write_text(json.dumps(bundle, indent=2) + "\n")

    print(
        json.dumps(
            {
                "source": str(source),
                "output": str(output),
                "profileBandwidthKbps": args.profile_bandwidth_kbps,
                "nominalCodebookKbps": raw_kbps,
                "lm": lm_report,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
