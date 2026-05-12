#!/usr/bin/env python3
"""Export EnCodec LM weights to the deterministic Rust/wasm LM format."""

from __future__ import annotations

import argparse
import json
import math
import struct
import sys
from pathlib import Path

import numpy as np
import torch


MAGIC = b"ELMW0001"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--encodec-path", type=Path, default=Path("../encodec"))
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--update-bundle", action="store_true")
    return parser.parse_args()


def write_array(fo, tensor: torch.Tensor) -> None:
    array = tensor.detach().cpu().to(torch.float32).contiguous().numpy()
    fo.write(array.astype("<f4", copy=False).tobytes())


def sin_embedding(frame_length: int, dim: int, max_period: float = 10000.0) -> np.ndarray:
    positions = np.arange(frame_length, dtype=np.float64).reshape(frame_length, 1)
    adim = np.arange(dim // 2, dtype=np.float64).reshape(1, dim // 2)
    phase = positions / (max_period ** (adim / ((dim // 2) - 1)))
    emb = np.concatenate([np.cos(phase), np.sin(phase)], axis=1)
    return emb.astype("<f4", copy=False)


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(args.encodec_path.resolve()))

    from encodec import EncodecModel

    bundle_json = args.bundle / "bundle.json"
    meta = json.loads(bundle_json.read_text())
    output = args.output or (args.bundle / "lm_weights.bin")

    model_name = meta["model_name"]
    if model_name != "encodec_48khz":
        raise SystemExit(f"unsupported model_name {model_name!r}")

    model = EncodecModel.encodec_model_48khz()
    model.set_target_bandwidth(float(meta["bandwidth_kbps"]))
    lm = model.get_lm_model(dtype=torch.float32).eval()
    state = lm.state_dict()

    dim = int(meta["lm_dim"])
    layers = int(meta["lm_num_layers"])
    heads = 8
    codebooks = int(meta["num_codebooks"])
    cardinality = int(meta.get("lm_cardinality") or meta.get("codebook_cardinality") or 1024)
    frame_length = int(meta["frame_length"])
    past_context = int(meta["lm_past_context"])

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as fo:
        fo.write(MAGIC)
        fo.write(
            struct.pack(
                "<7I",
                dim,
                layers,
                heads,
                codebooks,
                cardinality,
                frame_length,
                past_context,
            )
        )

        write_array(fo, state["transformer.norm_in.weight"])
        write_array(fo, state["transformer.norm_in.bias"])
        fo.write(sin_embedding(frame_length, dim).tobytes())

        for layer in range(layers):
            prefix = f"transformer.layers.{layer}"
            write_array(fo, state[f"{prefix}.self_attn.in_proj_weight"])
            write_array(fo, state[f"{prefix}.self_attn.in_proj_bias"])
            write_array(fo, state[f"{prefix}.self_attn.out_proj.weight"])
            write_array(fo, state[f"{prefix}.self_attn.out_proj.bias"])
            write_array(fo, state[f"{prefix}.linear1.weight"])
            write_array(fo, state[f"{prefix}.linear1.bias"])
            write_array(fo, state[f"{prefix}.linear2.weight"])
            write_array(fo, state[f"{prefix}.linear2.bias"])
            write_array(fo, state[f"{prefix}.norm1.weight"])
            write_array(fo, state[f"{prefix}.norm1.bias"])
            write_array(fo, state[f"{prefix}.norm2.weight"])
            write_array(fo, state[f"{prefix}.norm2.bias"])

        for codebook in range(codebooks):
            write_array(fo, state[f"emb.{codebook}.weight"])

        for codebook in range(codebooks):
            write_array(fo, state[f"linears.{codebook}.weight"])
            write_array(fo, state[f"linears.{codebook}.bias"])

    if args.update_bundle:
        meta["lm_weight_model"] = output.name
        bundle_json.write_text(json.dumps(meta, indent=2) + "\n")

    print(json.dumps({"output": str(output), "bytes": output.stat().st_size}, indent=2))


if __name__ == "__main__":
    main()
