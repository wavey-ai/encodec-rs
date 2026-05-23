#!/usr/bin/env bash
set -euo pipefail

ENCODEC_REPO="${ENCODEC_REPO:-/Users/jamie/wavey.ai/encodec}"
ENCODEC_RS_REPO="${ENCODEC_RS_REPO:-/Users/jamie/wavey.ai/encodec-rs}"
MODEL="${MODEL:-encodec_48khz}"
DEVICE="${DEVICE:-cpu}"
OPSET="${OPSET:-17}"
LM_FRAME_LENGTH="${LM_FRAME_LENGTH:-300}"
LM_ENTROPY_LOGIT_STEP="${LM_ENTROPY_LOGIT_STEP:-2.1}"
BANDWIDTHS="${BANDWIDTHS:-6.0 12.0}"

CHUNKS="${CHUNKS:-1000ms:48000:47520 1333ms:63998:63998 1800ms:86400:86400}"

cd "$ENCODEC_REPO"

export PYTHONPATH="$ENCODEC_REPO${PYTHONPATH:+:$PYTHONPATH}"
export KMP_DUPLICATE_LIB_OK="${KMP_DUPLICATE_LIB_OK:-TRUE}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

for BANDWIDTH in $BANDWIDTHS; do
  bw_tag="$(python - "$BANDWIDTH" <<'PY'
import sys
v = float(sys.argv[1])
print(str(int(v)) if v.is_integer() else str(v).replace(".", "p"))
PY
)"
  for CHUNK_SPEC in $CHUNKS; do
    IFS=: read -r chunk_tag trace_samples trace_stride <<< "$CHUNK_SPEC"
    OUTPUT_DIR="${ENCODEC_RS_REPO}/onnx-bundles/${MODEL}_${bw_tag}kbps_${chunk_tag}"
    mkdir -p "$OUTPUT_DIR"

    echo
    echo "exporting model=${MODEL} bandwidth=${BANDWIDTH} chunk=${chunk_tag} samples=${trace_samples} stride=${trace_stride} output=${OUTPUT_DIR} lm_frame_length=${LM_FRAME_LENGTH}"

    ENCODEC_ONNX_TRACE_SAMPLES="$trace_samples" \
    ENCODEC_ONNX_TRACE_STRIDE="$trace_stride" \
    python scripts/export_frame_onnx.py \
      --model "$MODEL" \
      --bandwidth "$BANDWIDTH" \
      --output-dir "$OUTPUT_DIR" \
      --device "$DEVICE" \
      --opset-version "$OPSET"

    python - "$OUTPUT_DIR" "$MODEL" "$BANDWIDTH" "$DEVICE" "$LM_FRAME_LENGTH" "$LM_ENTROPY_LOGIT_STEP" <<'PY'
from __future__ import annotations

import inspect
import json
import struct
import sys
from pathlib import Path

import numpy as np
import torch

from encodec.compress import MODELS

output_dir = Path(sys.argv[1])
model_name = sys.argv[2]
bandwidth = float(sys.argv[3])
device = sys.argv[4]
frame_length = int(sys.argv[5])
entropy_logit_step = float(sys.argv[6])

model = MODELS[model_name]().to(device).eval()
model.set_target_bandwidth(bandwidth)

lm = model.get_lm_model(device=torch.device("cpu"), dtype=torch.float64).eval()
lm.tau = 1.0

num_codebooks = int(model.quantizer.get_num_quantizers_for_bandwidth(model.frame_rate, model.bandwidth))
cardinality = int(model.quantizer.bins)
dim = int(lm.dim)
layers = int(len(lm.transformer.layers))
heads = int(lm.transformer.layers[0].self_attn.num_heads)
past_context = int(3.5 * model.frame_rate)

if num_codebooks > int(lm.n_q):
    raise SystemExit(f"LM only has {lm.n_q} codebooks, requested {num_codebooks}")

def tensor_to_f32_bytes(t: torch.Tensor) -> bytes:
    a = t.detach().cpu().to(torch.float32).contiguous().numpy()
    return a.astype("<f4", copy=False).tobytes()

def quantized_linear_bytes(weight: torch.Tensor) -> tuple[bytes, bytes]:
    w = weight.detach().cpu().to(torch.float32).contiguous().numpy()
    if w.ndim != 2:
        raise SystemExit(f"expected 2D weight, got {w.shape}")
    max_abs = np.max(np.abs(w), axis=1)
    scales = max_abs / 127.0
    scales = np.where(scales > 0.0, scales, 1.0).astype("<f4")
    q = np.round(w / scales[:, None]).clip(-127, 127).astype(np.int8)
    return scales.tobytes(), q.tobytes()

def write_f32_vec(out, t: torch.Tensor) -> None:
    out.write(tensor_to_f32_bytes(t.reshape(-1)))

def write_quantized_linear(out, linear_or_weight) -> None:
    weight = linear_or_weight.weight if hasattr(linear_or_weight, "weight") else linear_or_weight
    scales, q = quantized_linear_bytes(weight)
    out.write(scales)
    out.write(q)

def norm_pair(module, fallback_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    if module is not None and hasattr(module, "weight") and hasattr(module, "bias"):
        return module.weight.detach(), module.bias.detach()
    return torch.ones(fallback_dim, dtype=torch.float32), torch.zeros(fallback_dim, dtype=torch.float32)

def transformer_norm_in(transformer, fallback_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    for name in ("norm_in", "norm", "input_norm"):
        module = getattr(transformer, name, None)
        if module is not None and hasattr(module, "weight") and module.weight.numel() == fallback_dim:
            return norm_pair(module, fallback_dim)
    return norm_pair(None, fallback_dim)

def call_create_sin_embedding(positions: torch.Tensor, dim: int, max_period: float) -> torch.Tensor:
    from encodec.modules import transformer as transformer_module
    create = getattr(transformer_module, "create_sin_embedding")
    sig = inspect.signature(create)
    kwargs = {}
    if "max_period" in sig.parameters:
        kwargs["max_period"] = max_period
    if "dtype" in sig.parameters:
        kwargs["dtype"] = torch.float64
    try:
        return create(positions, dim, **kwargs)
    except TypeError:
        try:
            return create(positions, dim, max_period)
        except TypeError:
            return create(positions, dim)

def fallback_sin_embedding(frame_length: int, dim: int, max_period: float) -> torch.Tensor:
    if dim % 2 != 0:
        raise SystemExit("fallback sinusoidal embedding requires even dim")
    half = dim // 2
    positions = torch.arange(frame_length, dtype=torch.float64).view(frame_length, 1)
    adim = torch.arange(half, dtype=torch.float64).view(1, half)
    denom = max_period ** (adim / max(1, half - 1))
    phase = positions / denom
    return torch.cat([torch.cos(phase), torch.sin(phase)], dim=1).to(torch.float32)

def positional_embedding(transformer, frame_length: int, dim: int) -> torch.Tensor:
    for name in ("positional_embedding", "pos_emb"):
        value = getattr(transformer, name, None)
        if isinstance(value, torch.Tensor):
            pos = value.detach().cpu()
            if pos.dim() == 3:
                pos = pos[0]
            return pos[:frame_length, :dim].to(torch.float32)
    max_period = float(getattr(transformer, "max_period", 10000.0))
    scale = getattr(transformer, "positional_scale", 1.0)
    if isinstance(scale, torch.Tensor):
        scale = float(scale.detach().cpu().item())
    try:
        positions = torch.arange(frame_length, dtype=torch.float64).view(1, frame_length, 1)
        pos = call_create_sin_embedding(positions, dim, max_period)
        if pos.dim() == 3:
            pos = pos[0]
        pos = pos[:frame_length, :dim].detach().cpu().to(torch.float32)
    except Exception:
        pos = fallback_sin_embedding(frame_length, dim, max_period)
    return pos * float(scale)

def layer_attr(layer, *names):
    cur = layer
    for name in names:
        cur = getattr(cur, name)
    return cur

def write_layer(out, layer) -> None:
    attn = layer_attr(layer, "self_attn")
    write_quantized_linear(out, attn.in_proj_weight)
    write_f32_vec(out, attn.in_proj_bias)
    write_quantized_linear(out, attn.out_proj)
    write_f32_vec(out, attn.out_proj.bias)
    write_quantized_linear(out, layer.linear1)
    write_f32_vec(out, layer.linear1.bias)
    write_quantized_linear(out, layer.linear2)
    write_f32_vec(out, layer.linear2.bias)
    write_f32_vec(out, layer.norm1.weight)
    write_f32_vec(out, layer.norm1.bias)
    write_f32_vec(out, layer.norm2.weight)
    write_f32_vec(out, layer.norm2.bias)

norm_weight, norm_bias = transformer_norm_in(lm.transformer, dim)
pos = positional_embedding(lm.transformer, frame_length, dim)

if pos.shape != (frame_length, dim):
    raise SystemExit(f"positional embedding shape {tuple(pos.shape)} does not match {(frame_length, dim)}")

lm_path = output_dir / "lm_weights_q8.bin"

with lm_path.open("wb") as out:
    out.write(b"ELMQ0001")
    out.write(struct.pack("<7I", dim, layers, heads, num_codebooks, cardinality, frame_length, past_context))
    write_f32_vec(out, norm_weight)
    write_f32_vec(out, norm_bias)
    write_f32_vec(out, pos)
    for layer in lm.transformer.layers:
        write_layer(out, layer)
    for codebook in range(num_codebooks):
        write_f32_vec(out, lm.emb[codebook].weight)
    for codebook in range(num_codebooks):
        write_quantized_linear(out, lm.linears[codebook])
        write_f32_vec(out, lm.linears[codebook].bias)

bundle_path = output_dir / "bundle.json"
bundle = json.loads(bundle_path.read_text())
bundle.update({
    "lm_quant_weight_model": lm_path.name,
    "lm_dim": dim,
    "lm_num_layers": layers,
    "lm_past_context": past_context,
    "lm_logit_step": float(getattr(lm, "logit_step", 1.0 / 64.0)),
    "lm_entropy_logit_step": entropy_logit_step,
    "lm_cardinality": cardinality,
})
bundle_path.write_text(json.dumps(bundle, indent=2) + "\n")

print(json.dumps({
    "lm_weights": str(lm_path),
    "dim": dim,
    "layers": layers,
    "heads": heads,
    "codebooks": num_codebooks,
    "cardinality": cardinality,
    "frame_length": frame_length,
    "past_context": past_context,
    "bytes": lm_path.stat().st_size,
}, indent=2, sort_keys=True))
PY

    python - "$OUTPUT_DIR" "$trace_samples" <<'PY'
import json
import struct
import sys
from pathlib import Path

import onnx

output_dir = Path(sys.argv[1])
trace_samples = int(sys.argv[2])

for name in ["encode_frame.onnx", "decode_frame.onnx"]:
    path = output_dir / name
    model = onnx.load(path)
    print(path)
    print("opset", model.opset_import[0].version)
    for item in model.graph.input:
        dims = [d.dim_param or d.dim_value for d in item.type.tensor_type.shape.dim]
        print(" input", item.name, dims)
    for item in model.graph.output:
        dims = [d.dim_param or d.dim_value for d in item.type.tensor_type.shape.dim]
        print(" output", item.name, dims)
    print()

lm_path = output_dir / "lm_weights_q8.bin"
header = struct.unpack("<7I", lm_path.read_bytes()[8:36])
print(dict(zip(
    ["dim", "layers", "heads", "codebooks", "cardinality", "frame_length", "past_context"],
    header
)))

bundle = json.loads((output_dir / "bundle.json").read_text())
print(json.dumps({
    "bundle": str(output_dir / "bundle.json"),
    "model_name": bundle.get("model_name"),
    "bandwidth_kbps": bundle.get("bandwidth_kbps"),
    "segment_samples": bundle.get("segment_samples"),
    "segment_stride": bundle.get("segment_stride"),
    "num_codebooks": bundle.get("num_codebooks"),
    "frame_length": bundle.get("frame_length"),
    "lm_quant_weight_model": bundle.get("lm_quant_weight_model"),
    "lm_dim": bundle.get("lm_dim"),
    "lm_num_layers": bundle.get("lm_num_layers"),
    "lm_past_context": bundle.get("lm_past_context"),
    "lm_cardinality": bundle.get("lm_cardinality"),
}, indent=2, sort_keys=True))

if bundle.get("segment_samples") != trace_samples:
    raise SystemExit(f"segment_samples {bundle.get('segment_samples')} != trace_samples {trace_samples}")
PY

  done
done
