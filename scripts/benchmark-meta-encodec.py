#!/usr/bin/env python3
"""Benchmark the pinned official Meta EnCodec implementation on one WAV file."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import platform
import subprocess
import sys
import time
import traceback
from pathlib import Path
from unittest import mock

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import soundfile as sf
import torch
import torchaudio
from encodec import EncodecModel


PINNED_COMMIT = "0e2d0aed29362c8e8f52494baf3e6f99056b214f"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_wav", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--bandwidth", type=float, default=12.0)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--upstream-commit", default=PINNED_COMMIT)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def cpu_name() -> str:
    try:
        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return platform.processor() or "unknown"


def seconds() -> float:
    return time.perf_counter()


def timed(function):
    started = seconds()
    value = function()
    return value, seconds() - started


def round_number(value: float, digits: int = 6) -> float:
    return round(float(value), digits)


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def log(message: str) -> None:
    print(f"[meta-encodec-benchmark] {message}", file=sys.stderr, flush=True)


def main() -> None:
    args = parse_args()
    if args.threads < 1:
        raise ValueError("--threads must be positive")

    input_path = args.input_wav.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    report_path = output_root / "report.json"
    ecdc_path = output_root / "meta.ecdc"
    decoded_path = output_root / "meta-decoded.wav"
    neural_decoded_path = output_root / "meta-neural-decoded.wav"

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(args.threads)

    audio, sample_rate = sf.read(input_path, dtype="float32", always_2d=True)
    if sample_rate != 48_000:
        raise ValueError(f"input sample rate is {sample_rate}; expected 48000")
    if audio.shape[1] != 2:
        raise ValueError(f"input has {audio.shape[1]} channels; expected 2")
    wav = torch.from_numpy(np.ascontiguousarray(audio.T))
    audio_samples = int(wav.shape[-1])
    audio_seconds = audio_samples / sample_rate

    compress_module = importlib.import_module("encodec.compress")
    encodec_package = importlib.import_module("encodec")
    report: dict = {
        "schema": "wavey.encodec.meta-full-file-benchmark",
        "schema_version": 1,
        "status": "starting",
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "configuration": {
            "implementation": "official facebookresearch/encodec",
            "upstream_commit": args.upstream_commit,
            "model": "encodec_48khz",
            "bandwidth_kbps": args.bandwidth,
            "device": "cpu",
            "threads": args.threads,
            "use_lm": True,
            "session_setup_excluded_from_rtf": True,
        },
        "environment": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "cpu": cpu_name(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "torchaudio": torchaudio.__version__,
            "encodec": getattr(encodec_package, "__version__", "unknown"),
        },
        "source": {
            "path": str(input_path),
            "sha256": sha256_file(input_path),
            "sample_rate": sample_rate,
            "channels": int(audio.shape[1]),
            "samples_per_channel": audio_samples,
            "seconds": round_number(audio_seconds),
            "soundfile_format": sf.info(input_path).format,
            "soundfile_subtype": sf.info(input_path).subtype,
        },
        "artifacts": {},
        "timings": {},
    }
    write_json(report_path, report)

    try:
        log("loading the official 48 kHz model")
        model, model_setup_seconds = timed(
            lambda: EncodecModel.encodec_model_48khz().to("cpu").eval()
        )
        model.set_target_bandwidth(args.bandwidth)
        report["timings"]["model_setup_seconds"] = round_number(model_setup_seconds)
        report["model_geometry"] = {
            "segment_samples": model.segment_length,
            "segment_stride": model.segment_stride,
            "overlap_samples": model.segment_length - model.segment_stride,
            "code_timesteps_per_full_segment": int(
                np.ceil(model.segment_length * model.frame_rate / model.sample_rate)
            ),
            "frame_rate_hz": model.frame_rate,
        }
        report["status"] = "model_loaded"
        write_json(report_path, report)

        log("loading the official floating-point language model")
        language_model, lm_setup_seconds = timed(model.get_lm_model)
        report["timings"]["lm_setup_seconds"] = round_number(lm_setup_seconds)

        with mock.patch.object(EncodecModel, "get_lm_model", lambda _self: language_model):
            with mock.patch.dict(
                compress_module.MODELS,
                {"encodec_48khz": lambda: model},
                clear=False,
            ):
                warm_samples = min(audio_samples, model.segment_length)
                warm_wav = wav[:, :warm_samples]
                log("warming neural and entropy paths with one model segment")
                with torch.inference_mode():
                    warm_bytes = compress_module.compress(model, warm_wav, use_lm=True)
                    compress_module.decompress(warm_bytes, device="cpu")

                log("timing loaded-model neural encode")
                with torch.inference_mode():
                    frames, neural_encode_seconds = timed(
                        lambda: model.encode(wav.unsqueeze(0))
                    )
                report["timings"]["neural_encode_seconds"] = round_number(
                    neural_encode_seconds
                )
                report["timings"]["neural_encode_rtfx"] = round_number(
                    audio_seconds / neural_encode_seconds
                )
                report["model_geometry"]["full_file_segments"] = len(frames)
                report["status"] = "neural_encoded"
                write_json(report_path, report)

                log("timing loaded-model neural decode")
                with torch.inference_mode():
                    neural_decoded, neural_decode_seconds = timed(
                        lambda: model.decode(frames)[..., :audio_samples]
                    )
                neural_pcm = neural_decoded[0].detach().cpu().numpy().T
                sf.write(
                    neural_decoded_path,
                    neural_pcm,
                    sample_rate,
                    subtype="FLOAT",
                )
                report["timings"]["neural_decode_seconds"] = round_number(
                    neural_decode_seconds
                )
                report["timings"]["neural_decode_rtfx"] = round_number(
                    audio_seconds / neural_decode_seconds
                )
                report["status"] = "neural_decoded"
                write_json(report_path, report)
                del neural_decoded, neural_pcm, frames

                log("timing official LM-assisted ECDC compression")
                with torch.inference_mode():
                    ecdc_bytes, full_encode_seconds = timed(
                        lambda: compress_module.compress(model, wav, use_lm=True)
                    )
                ecdc_path.write_bytes(ecdc_bytes)
                report["timings"]["full_encode_seconds"] = round_number(
                    full_encode_seconds
                )
                report["timings"]["full_encode_rtfx"] = round_number(
                    audio_seconds / full_encode_seconds
                )
                report["artifacts"]["ecdc"] = {
                    "path": str(ecdc_path),
                    "bytes": len(ecdc_bytes),
                    "sha256": hashlib.sha256(ecdc_bytes).hexdigest(),
                    "effective_kbps": round_number(
                        len(ecdc_bytes) * 8 / 1000 / audio_seconds
                    ),
                }
                report["status"] = "full_encoded"
                write_json(report_path, report)

                log("timing official LM-assisted ECDC decompression")
                with torch.inference_mode():
                    decoded_result, full_decode_seconds = timed(
                        lambda: compress_module.decompress(ecdc_bytes, device="cpu")
                    )
                decoded, decoded_sample_rate = decoded_result
                decoded = decoded[:, :audio_samples].detach().cpu()
                sf.write(
                    decoded_path,
                    decoded.numpy().T,
                    decoded_sample_rate,
                    subtype="FLOAT",
                )
                report["timings"]["full_decode_seconds"] = round_number(
                    full_decode_seconds
                )
                report["timings"]["full_decode_rtfx"] = round_number(
                    audio_seconds / full_decode_seconds
                )
                report["artifacts"]["decoded_wav"] = {
                    "path": str(decoded_path),
                    "sha256": sha256_file(decoded_path),
                    "sample_rate": decoded_sample_rate,
                    "channels": int(decoded.shape[0]),
                    "samples_per_channel": int(decoded.shape[-1]),
                    "subtype": "FLOAT",
                }
                report["artifacts"]["neural_decoded_wav"] = {
                    "path": str(neural_decoded_path),
                    "sha256": sha256_file(neural_decoded_path),
                }

        report["status"] = "passed"
        report["finished_at"] = time.strftime(
            "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
        )
        write_json(report_path, report)
        print(json.dumps(report, indent=2, sort_keys=True))
    except Exception as error:
        report["status"] = "error"
        report["error"] = f"{type(error).__name__}: {error}"
        report["traceback"] = traceback.format_exc()
        report["finished_at"] = time.strftime(
            "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
        )
        write_json(report_path, report)
        raise


if __name__ == "__main__":
    main()
