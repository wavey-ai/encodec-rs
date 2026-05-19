#!/usr/bin/env python3
"""Quantize deterministic LM weights to the native q8 LM format."""

from __future__ import annotations

import argparse
import json
import math
import struct
from pathlib import Path


SOURCE_MAGIC = b"ELMW0001"
TARGET_MAGIC = b"ELMQ0001"
HEADER_U32S = 7


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path, help="source ELMW0001 f32 LM weight file")
    parser.add_argument("--output", type=Path, default=None, help="output lm_weights_q8.bin")
    parser.add_argument(
        "--bundle",
        type=Path,
        default=None,
        help="bundle directory or bundle.json to update when --update-bundle is set",
    )
    parser.add_argument("--update-bundle", action="store_true")
    return parser.parse_args()


class Reader:
    def __init__(self, data: bytes) -> None:
        self.data = data
        self.pos = 0

    def read(self, length: int) -> bytes:
        end = self.pos + length
        if end > len(self.data):
            raise EOFError("LM weight file ended early")
        chunk = self.data[self.pos : end]
        self.pos = end
        return chunk

    def read_u32s(self, count: int) -> tuple[int, ...]:
        return struct.unpack(f"<{count}I", self.read(count * 4))

    def read_f32_bytes(self, count: int) -> bytes:
        return self.read(count * 4)

    def read_quantized_linear(self, rows: int, cols: int) -> tuple[bytes, bytes]:
        source = self.read_f32_bytes(rows * cols)
        scales = bytearray(rows * 4)
        weights = bytearray(rows * cols)
        row_bytes = cols * 4

        for row in range(rows):
            row_start = row * row_bytes
            row_values = struct.unpack_from(f"<{cols}f", source, row_start)
            max_abs = 0.0
            for value in row_values:
                value_abs = abs(value)
                if math.isfinite(value_abs) and value_abs > max_abs:
                    max_abs = value_abs

            scale = max_abs / 127.0 if max_abs > 0.0 else 1.0
            inv_scale = 1.0 / scale
            struct.pack_into("<f", scales, row * 4, scale)

            out_start = row * cols
            for col, value in enumerate(row_values):
                quantized = int(round(value * inv_scale))
                if quantized < -127:
                    quantized = -127
                elif quantized > 127:
                    quantized = 127
                weights[out_start + col] = quantized & 0xFF

        return bytes(scales), bytes(weights)

    def remaining(self) -> int:
        return len(self.data) - self.pos


def copy_f32(reader: Reader, out, count: int) -> None:
    out.write(reader.read_f32_bytes(count))


def copy_qlinear(reader: Reader, out, rows: int, cols: int) -> None:
    scales, weights = reader.read_quantized_linear(rows, cols)
    out.write(scales)
    out.write(weights)


def bundle_json_path(bundle: Path | None, source: Path) -> Path:
    if bundle is None:
        return source.parent / "bundle.json"
    if bundle.name == "bundle.json":
        return bundle
    return bundle / "bundle.json"


def main() -> None:
    args = parse_args()
    output = args.output or (args.input.parent / "lm_weights_q8.bin")

    reader = Reader(args.input.read_bytes())
    magic = reader.read(len(SOURCE_MAGIC))
    if magic != SOURCE_MAGIC:
        raise SystemExit("input is not an ELMW0001 deterministic LM weight file")

    header = reader.read_u32s(HEADER_U32S)
    dim, layers, _heads, codebooks, cardinality, frame_length, _past_context = header
    if dim == 0 or layers == 0 or codebooks == 0 or cardinality == 0:
        raise SystemExit(f"invalid LM header: {header!r}")
    hidden_dim = dim * 4

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as out:
        out.write(TARGET_MAGIC)
        out.write(struct.pack("<7I", *header))

        copy_f32(reader, out, dim)
        copy_f32(reader, out, dim)
        copy_f32(reader, out, frame_length * dim)

        for _layer in range(layers):
            copy_qlinear(reader, out, 3 * dim, dim)
            copy_f32(reader, out, 3 * dim)
            copy_qlinear(reader, out, dim, dim)
            copy_f32(reader, out, dim)
            copy_qlinear(reader, out, hidden_dim, dim)
            copy_f32(reader, out, hidden_dim)
            copy_qlinear(reader, out, dim, hidden_dim)
            copy_f32(reader, out, dim)
            copy_f32(reader, out, dim)
            copy_f32(reader, out, dim)
            copy_f32(reader, out, dim)
            copy_f32(reader, out, dim)

        for _codebook in range(codebooks):
            copy_f32(reader, out, (cardinality + 1) * dim)

        for _codebook in range(codebooks):
            copy_qlinear(reader, out, cardinality, dim)
            copy_f32(reader, out, cardinality)

    if reader.remaining() != 0:
        raise SystemExit(f"input has {reader.remaining()} trailing bytes")

    if args.update_bundle:
        metadata_path = bundle_json_path(args.bundle, args.input)
        metadata = json.loads(metadata_path.read_text())
        metadata["lm_quant_weight_model"] = output.name
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")

    print(
        json.dumps(
            {
                "input": str(args.input),
                "output": str(output),
                "bytes": output.stat().st_size,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
