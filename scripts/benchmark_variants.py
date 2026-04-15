#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark EnCodec fork and encodec-rs ONNX variants on one machine."
    )
    parser.add_argument("--input", type=Path, required=True, help="Input WAV/audio file.")
    parser.add_argument(
        "--fork-repo",
        type=Path,
        action="append",
        default=[],
        help="Path to an encodec Python repo checkout to benchmark. May be passed multiple times.",
    )
    parser.add_argument(
        "--bundle-dir",
        type=Path,
        help="Path to an encodec-rs ONNX bundle directory to benchmark.",
    )
    parser.add_argument(
        "--encodec-rs-repo",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Path to the encodec-rs checkout. Defaults to this script's repo.",
    )
    parser.add_argument(
        "--model",
        choices=["encodec_24khz", "encodec_48khz"],
        default="encodec_48khz",
    )
    parser.add_argument("--bandwidth", type=float, default=6.0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--onnx-target",
        choices=["cpu", "cuda", "tensorrt"],
        default="cuda",
    )
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument(
        "--runs",
        type=int,
        default=3,
        help="Number of runs per variant. The script reports per-run results and best/median rollups.",
    )
    parser.add_argument(
        "--lm",
        action="store_true",
        help="Enable LM for Python fork benchmarks.",
    )
    parser.add_argument(
        "--python-device",
        default="cuda",
        help="Device for Python fork encode path, e.g. cpu or cuda.",
    )
    parser.add_argument(
        "--python-decode-device",
        default="cuda",
        help="Decode device for Python fork path, e.g. cpu or cuda.",
    )
    parser.add_argument(
        "--keep-artifacts",
        action="store_true",
        help="Keep temporary output WAVs created for ONNX runs.",
    )
    return parser.parse_args()


def run_command(
    argv: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> str:
    completed = subprocess.run(
        argv,
        cwd=str(cwd) if cwd else None,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def parse_json_output(stdout: str) -> dict[str, Any]:
    lines = [line.strip() for line in stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError("command produced no stdout")
    for index in range(len(lines) - 1, -1, -1):
        if lines[index].startswith("{"):
            candidate = "\n".join(lines[index:])
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                pass
    raise RuntimeError(f"could not find JSON object in stdout:\n{stdout}")


def summarize_numeric_runs(runs: list[dict[str, Any]], key: str) -> dict[str, float] | None:
    values = [float(run[key]) for run in runs if key in run]
    if not values:
        return None
    values.sort()
    mid = len(values) // 2
    median = values[mid] if len(values) % 2 else (values[mid - 1] + values[mid]) / 2.0
    return {
        "best": min(values),
        "median": median,
        "worst": max(values),
    }


def benchmark_python_repo(
    repo_path: Path,
    *,
    input_path: Path,
    model: str,
    bandwidth: float,
    device: str,
    decode_device: str,
    lm: bool,
    runs: int,
) -> dict[str, Any]:
    full_eval = textwrap.dedent(
        """
        import json
        import math
        import sys
        import time
        from pathlib import Path

        import soundfile as sf
        import torch

        repo_path = Path(sys.argv[1])
        input_path = Path(sys.argv[2])
        model_name = sys.argv[3]
        bandwidth = float(sys.argv[4])
        device_name = sys.argv[5]
        decode_device_name = sys.argv[6]
        lm = sys.argv[7] == "1"

        sys.path.insert(0, str(repo_path))

        from encodec.compress import MODELS, compress, decompress
        from encodec.utils import convert_audio

        def sync_device(name: str) -> None:
            if name.startswith("cuda") and torch.cuda.is_available():
                torch.cuda.synchronize()

        wav, sr = sf.read(input_path, always_2d=True, dtype="float32")
        wav = torch.from_numpy(wav.T.copy())
        source_duration = wav.shape[-1] / sr

        model = MODELS[model_name]().to(device_name)
        model.set_target_bandwidth(bandwidth)

        wav_in = convert_audio(wav, sr, model.sample_rate, model.channels).to(device_name)
        wav_ref = wav_in.detach().cpu()

        sync_device(device_name)
        start_encode = time.perf_counter()
        payload = compress(model, wav_in, use_lm=lm)
        sync_device(device_name)
        encode_s = time.perf_counter() - start_encode

        sync_device(decode_device_name)
        start_decode = time.perf_counter()
        wav_out, out_sr = decompress(payload, device=decode_device_name)
        sync_device(decode_device_name)
        decode_s = time.perf_counter() - start_decode

        wav_out = wav_out.detach().cpu()
        if wav_out.dim() == 1:
            wav_out = wav_out.unsqueeze(0)
        wav_out = wav_out[:, : wav_ref.shape[-1]]
        if wav_out.shape[-1] < wav_ref.shape[-1]:
            pad = wav_ref.shape[-1] - wav_out.shape[-1]
            wav_out = torch.nn.functional.pad(wav_out, (0, pad))

        diff = wav_out - wav_ref
        mse = float(diff.pow(2).mean().item())
        mae = float(diff.abs().mean().item())
        signal_power = float(wav_ref.pow(2).mean().item())
        snr_db = float("inf") if mse == 0 else 10.0 * math.log10(max(signal_power, 1e-12) / mse)

        print(json.dumps({
            "repo_path": str(repo_path),
            "input": str(input_path),
            "model": model_name,
            "bandwidth": bandwidth,
            "device": device_name,
            "decode_device": decode_device_name,
            "lm": lm,
            "success": True,
            "source_duration_s": source_duration,
            "input_sr": sr,
            "model_sr": model.sample_rate,
            "input_channels": int(wav.shape[0]),
            "model_channels": int(model.channels),
            "encoded_samples": int(wav_in.shape[-1]),
            "encoded_bytes": len(payload),
            "payload_bytes": len(payload),
            "decode_sr": out_sr,
            "decoded_samples": int(wav_out.shape[-1]),
            "encode_s": encode_s,
            "decode_s": decode_s,
            "rtf_encode": encode_s / max(source_duration, 1e-9),
            "rtf_decode": decode_s / max(source_duration, 1e-9),
            "mse": mse,
            "mae": mae,
            "max_abs_err": float(diff.abs().max().item()),
            "snr_db": snr_db,
            "bps": (len(payload) * 8.0) / max(source_duration, 1e-9),
        }, sort_keys=True))
        """
    ).strip()

    per_run: list[dict[str, Any]] = []
    for index in range(runs):
        stdout = run_command(
            [
                sys.executable,
                "-c",
                full_eval,
                str(repo_path),
                str(input_path),
                model,
                str(bandwidth),
                device,
                decode_device,
                "1" if lm else "0",
            ]
        )
        payload = parse_json_output(stdout)
        payload["run_index"] = index
        per_run.append(payload)

    return {
        "variant_kind": "python-fork",
        "repo_path": str(repo_path),
        "runs": per_run,
        "summary": {
            "encode_s": summarize_numeric_runs(per_run, "encode_s"),
            "decode_s": summarize_numeric_runs(per_run, "decode_s"),
            "rtf_encode": summarize_numeric_runs(per_run, "rtf_encode"),
            "rtf_decode": summarize_numeric_runs(per_run, "rtf_decode"),
            "encoded_bytes": summarize_numeric_runs(per_run, "encoded_bytes"),
            "bps": summarize_numeric_runs(per_run, "bps"),
        },
    }


def benchmark_python_repo_frame(
    repo_path: Path,
    *,
    input_path: Path,
    model: str,
    bandwidth: float,
    device: str,
    runs: int,
) -> dict[str, Any]:
    frame_eval = textwrap.dedent(
        """
        import json
        import math
        import sys
        import time
        from pathlib import Path

        import soundfile as sf
        import torch

        repo_path = Path(sys.argv[1])
        input_path = Path(sys.argv[2])
        model_name = sys.argv[3]
        bandwidth = float(sys.argv[4])
        device_name = sys.argv[5]

        sys.path.insert(0, str(repo_path))

        from encodec.compress import MODELS
        from encodec.utils import convert_audio

        def sync_device() -> None:
            if device_name.startswith("cuda") and torch.cuda.is_available():
                torch.cuda.synchronize()

        wav, sr = sf.read(input_path, always_2d=True, dtype="float32")
        wav = torch.from_numpy(wav.T.copy())

        model = MODELS[model_name]().to(device_name)
        model.set_target_bandwidth(bandwidth)

        wav_in = convert_audio(wav, sr, model.sample_rate, model.channels).to(device_name)
        source_duration = wav_in.shape[-1] / model.sample_rate

        sync_device()
        start_encode = time.perf_counter()
        encoded_frames = model.encode(wav_in[None])
        sync_device()
        encode_s = time.perf_counter() - start_encode

        sync_device()
        start_decode = time.perf_counter()
        decoded = model.decode(encoded_frames)[..., : wav_in.shape[-1]]
        sync_device()
        decode_s = time.perf_counter() - start_decode

        decoded = decoded.detach().cpu()
        wav_ref = wav_in.detach().cpu().unsqueeze(0)
        diff = decoded - wav_ref
        mse = float(diff.pow(2).mean().item())
        mae = float(diff.abs().mean().item())
        signal_power = float(wav_ref.pow(2).mean().item())
        snr_db = float("inf") if mse == 0 else 10.0 * math.log10(max(signal_power, 1e-12) / mse)

        print(json.dumps({
            "repo_path": str(repo_path),
            "input": str(input_path),
            "model": model_name,
            "bandwidth": bandwidth,
            "device": device_name,
            "success": True,
            "source_duration_s": source_duration,
            "model_sr": model.sample_rate,
            "model_channels": int(model.channels),
            "segments": len(encoded_frames),
            "encode_s": encode_s,
            "decode_s": decode_s,
            "rtf_encode": encode_s / max(source_duration, 1e-9),
            "rtf_decode": decode_s / max(source_duration, 1e-9),
            "mse": mse,
            "mae": mae,
            "max_abs_err": float(diff.abs().max().item()),
            "snr_db": snr_db,
        }, sort_keys=True))
        """
    ).strip()

    per_run: list[dict[str, Any]] = []
    for index in range(runs):
        stdout = run_command(
            [
                sys.executable,
                "-c",
                frame_eval,
                str(repo_path),
                str(input_path),
                model,
                str(bandwidth),
                device,
            ]
        )
        payload = parse_json_output(stdout)
        payload["run_index"] = index
        per_run.append(payload)

    return {
        "variant_kind": "python-fork-frame",
        "repo_path": str(repo_path),
        "runs": per_run,
        "summary": {
            "encode_s": summarize_numeric_runs(per_run, "encode_s"),
            "decode_s": summarize_numeric_runs(per_run, "decode_s"),
            "rtf_encode": summarize_numeric_runs(per_run, "rtf_encode"),
            "rtf_decode": summarize_numeric_runs(per_run, "rtf_decode"),
            "segments": summarize_numeric_runs(per_run, "segments"),
            "snr_db": summarize_numeric_runs(per_run, "snr_db"),
        },
    }


def benchmark_onnx(
    *,
    encodec_rs_repo: Path,
    bundle_dir: Path,
    input_path: Path,
    onnx_target: str,
    device_id: int,
    batch_size: int,
    runs: int,
    keep_artifacts: bool,
) -> dict[str, Any]:
    if not bundle_dir.exists():
        raise FileNotFoundError(f"missing ONNX bundle dir {bundle_dir}")

    cargo = shutil.which("cargo")
    if cargo is None:
        raise RuntimeError("cargo is required to benchmark encodec-rs ONNX")

    per_run: list[dict[str, Any]] = []
    for index in range(runs):
        with tempfile.TemporaryDirectory(prefix="encodec-rs-bench-") as tempdir:
            output_wav = Path(tempdir) / f"onnx-roundtrip-{index}.wav"
            argv = [
                cargo,
                "run",
                "--release",
                "--features",
                "onnx",
                "--",
                "onnx-roundtrip-wav",
                str(bundle_dir),
                str(input_path),
                str(output_wav),
                "--batch-size",
                str(batch_size),
                "--device-id",
                str(device_id),
            ]
            if onnx_target == "cuda":
                argv.append("--cuda")
            elif onnx_target == "tensorrt":
                argv.extend(["--tensorrt", "--fp16"])

            stdout = run_command(argv, cwd=encodec_rs_repo)
            payload = parse_json_output(stdout)
            payload["run_index"] = index
            payload["output_wav_exists"] = output_wav.exists()
            if keep_artifacts and output_wav.exists():
                saved = encodec_rs_repo / f"bench-output-{index}.wav"
                saved.write_bytes(output_wav.read_bytes())
                payload["saved_output_wav"] = str(saved)
            per_run.append(payload)

    return {
        "variant_kind": "onnx",
        "bundle_dir": str(bundle_dir),
        "runs": per_run,
        "summary": {
            "encode_seconds": summarize_numeric_runs(per_run, "encode_seconds"),
            "decode_seconds": summarize_numeric_runs(per_run, "decode_seconds"),
            "encode_rtf": summarize_numeric_runs(per_run, "encode_rtf"),
            "decode_rtf": summarize_numeric_runs(per_run, "decode_rtf"),
        },
    }


def main() -> None:
    args = parse_args()
    results: list[dict[str, Any]] = []

    for repo_path in args.fork_repo:
        results.append(
            benchmark_python_repo(
                repo_path=repo_path.resolve(),
                input_path=args.input.resolve(),
                model=args.model,
                bandwidth=args.bandwidth,
                device=args.python_device,
                decode_device=args.python_decode_device,
                lm=args.lm,
                runs=args.runs,
            )
        )
        results.append(
            benchmark_python_repo_frame(
                repo_path=repo_path.resolve(),
                input_path=args.input.resolve(),
                model=args.model,
                bandwidth=args.bandwidth,
                device=args.python_device,
                runs=args.runs,
            )
        )

    if args.bundle_dir is not None:
        results.append(
            benchmark_onnx(
                encodec_rs_repo=args.encodec_rs_repo.resolve(),
                bundle_dir=args.bundle_dir.resolve(),
                input_path=args.input.resolve(),
                onnx_target=args.onnx_target,
                device_id=args.device_id,
                batch_size=max(1, args.batch_size),
                runs=max(1, args.runs),
                keep_artifacts=args.keep_artifacts,
            )
        )

    if not results:
        raise SystemExit("no variants selected; pass --fork-repo and/or --bundle-dir")

    payload = {
        "input": str(args.input.resolve()),
        "model": args.model,
        "bandwidth": args.bandwidth,
        "runs": args.runs,
        "results": results,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
