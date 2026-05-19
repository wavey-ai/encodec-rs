#!/usr/bin/env python3
"""Export Encodec frame ONNX initializers as MLX Swift-loadable safetensors.

This is an offline conversion step for Apple native builds. The generated
bundle contains no ONNX Runtime dependency: `bundle.json`, optional
`lm_weights_q8.bin`, frame-model `.safetensors`, and an `mlx-manifest.json`.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import shutil
import struct
from pathlib import Path
from typing import Any

import numpy as np
import onnx
from onnx import TensorProto, numpy_helper


ONNX_DTYPE_NAMES = {
    TensorProto.FLOAT: "float32",
    TensorProto.UINT8: "uint8",
    TensorProto.INT8: "int8",
    TensorProto.UINT16: "uint16",
    TensorProto.INT16: "int16",
    TensorProto.INT32: "int32",
    TensorProto.INT64: "int64",
    TensorProto.DOUBLE: "float64",
    TensorProto.BOOL: "bool",
    TensorProto.FLOAT16: "float16",
}

SAFE_DTYPE_BY_NUMPY = {
    np.dtype("float16"): "F16",
    np.dtype("float32"): "F32",
    np.dtype("float64"): "F64",
    np.dtype("int8"): "I8",
    np.dtype("int16"): "I16",
    np.dtype("int32"): "I32",
    np.dtype("int64"): "I64",
    np.dtype("uint8"): "U8",
    np.dtype("uint16"): "U16",
    np.dtype("uint32"): "U32",
    np.dtype("uint64"): "U64",
    np.dtype("bool"): "BOOL",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("bundle", type=Path, help="Source ONNX bundle directory")
    parser.add_argument(
        "output",
        type=Path,
        nargs="?",
        help="Output MLX bundle directory; defaults to target/mlx-bundles/<bundle-name>",
    )
    return parser.parse_args()


def value_info_shape(value_info: onnx.ValueInfoProto) -> list[str]:
    dims: list[str] = []
    for dim in value_info.type.tensor_type.shape.dim:
        if dim.dim_param:
            dims.append(dim.dim_param)
        elif dim.dim_value:
            dims.append(str(dim.dim_value))
        else:
            dims.append("?")
    return dims


def value_info_entry(value_info: onnx.ValueInfoProto) -> dict[str, Any]:
    elem_type = value_info.type.tensor_type.elem_type
    return {
        "name": value_info.name,
        "dtype": ONNX_DTYPE_NAMES.get(elem_type, f"onnx:{elem_type}"),
        "shape": value_info_shape(value_info),
    }


def normalize_array(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array)
    base_dtype = array.dtype.newbyteorder("=")
    if base_dtype not in SAFE_DTYPE_BY_NUMPY:
        raise ValueError(f"unsupported initializer dtype for safetensors: {array.dtype}")
    if array.dtype.byteorder == ">":
        array = array.byteswap().view(base_dtype)
    else:
        array = array.astype(base_dtype, copy=False)
    return np.ascontiguousarray(array)


def write_safetensors(path: Path, arrays: list[tuple[str, np.ndarray]], metadata: dict[str, str]) -> str:
    header: dict[str, Any] = {"__metadata__": metadata}
    chunks: list[bytes] = []
    offset = 0

    for name, array in arrays:
        array = normalize_array(array)
        dtype = SAFE_DTYPE_BY_NUMPY[array.dtype]
        chunk = array.tobytes(order="C")
        header[name] = {
            "dtype": dtype,
            "shape": list(array.shape),
            "data_offsets": [offset, offset + len(chunk)],
        }
        chunks.append(chunk)
        offset += len(chunk)

    header_bytes = json.dumps(header, separators=(",", ":"), sort_keys=False).encode("utf-8")
    padding = (8 - (len(header_bytes) % 8)) % 8
    header_bytes += b" " * padding

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.write(struct.pack("<Q", len(header_bytes)))
        handle.write(header_bytes)
        for chunk in chunks:
            handle.write(chunk)

    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_safetensors(path: Path) -> None:
    with path.open("rb") as handle:
        header_len = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(header_len))
        data_len = path.stat().st_size - 8 - header_len

    max_offset = 0
    for name, entry in header.items():
        if name == "__metadata__":
            continue
        start, end = entry["data_offsets"]
        if start < 0 or end < start:
            raise ValueError(f"invalid safetensors offsets for {name}: {start}..{end}")
        max_offset = max(max_offset, end)
    if max_offset != data_len:
        raise ValueError(f"{path} data size mismatch: header={max_offset} actual={data_len}")


def export_model(bundle: Path, output: Path, model_name: str) -> dict[str, Any]:
    model_path = bundle / f"{model_name}.onnx"
    model = onnx.load(model_path)
    op_histogram = collections.Counter(node.op_type for node in model.graph.node)

    arrays: list[tuple[str, np.ndarray]] = []
    tensors: list[dict[str, Any]] = []
    parameter_count = 0
    seen: set[str] = set()

    for initializer in model.graph.initializer:
        if initializer.name in seen:
            raise ValueError(f"duplicate initializer name in {model_path}: {initializer.name}")
        seen.add(initializer.name)
        array = normalize_array(numpy_helper.to_array(initializer))
        parameter_count += int(array.size)
        arrays.append((initializer.name, array))
        tensors.append(
            {
                "name": initializer.name,
                "dtype": SAFE_DTYPE_BY_NUMPY[array.dtype],
                "shape": list(array.shape),
            }
        )

    safetensors_name = f"{model_name}.safetensors"
    safetensors_path = output / safetensors_name
    digest = write_safetensors(
        safetensors_path,
        arrays,
        {
            "source_model": str(model_path),
            "source_format": "onnx-initializers",
            "target_runtime": "mlx-swift",
        },
    )
    verify_safetensors(safetensors_path)

    return {
        "sourceModel": model_path.name,
        "safetensors": safetensors_name,
        "initializerCount": len(arrays),
        "parameterCount": parameter_count,
        "opHistogram": dict(sorted(op_histogram.items())),
        "inputs": [value_info_entry(value) for value in model.graph.input],
        "outputs": [value_info_entry(value) for value in model.graph.output],
        "tensors": tensors,
        "sha256": digest,
    }


def main() -> None:
    args = parse_args()
    bundle = args.bundle.resolve()
    output = args.output or Path("target") / "mlx-bundles" / bundle.name
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    bundle_json = bundle / "bundle.json"
    if not bundle_json.exists():
        raise FileNotFoundError(bundle_json)
    shutil.copy2(bundle_json, output / "bundle.json")

    lm_weights = bundle / "lm_weights_q8.bin"
    if lm_weights.exists():
        shutil.copy2(lm_weights, output / "lm_weights_q8.bin")

    models = {
        "encode_frame": export_model(bundle, output, "encode_frame"),
        "decode_frame": export_model(bundle, output, "decode_frame"),
    }
    manifest = {
        "schemaVersion": 1,
        "format": "encodec-rs-mlx-frame-archive",
        "sourceBundle": str(bundle),
        "sourceBundleName": bundle.name,
        "models": models,
    }
    (output / "mlx-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    print(json.dumps({
        "output": str(output),
        "models": {
            key: {
                "initializers": value["initializerCount"],
                "parameters": value["parameterCount"],
                "safetensors": value["safetensors"],
                "sha256": value["sha256"],
            }
            for key, value in models.items()
        },
    }, indent=2))


if __name__ == "__main__":
    main()
