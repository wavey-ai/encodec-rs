#!/usr/bin/env python3
"""Measure the standard official EnCodec CLI in fresh processes."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import soundfile as sf


PINNED_COMMIT = "0e2d0aed29362c8e8f52494baf3e6f99056b214f"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_wav", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--encodec-cli", type=Path, required=True)
    parser.add_argument("--bandwidth", type=float, default=12.0)
    parser.add_argument("--threads", type=int, default=1)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_timed(command: list[str], environment: dict[str, str]) -> dict:
    started = time.perf_counter()
    result = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    elapsed = time.perf_counter() - started
    if result.returncode != 0:
        raise RuntimeError(
            f"command failed with exit code {result.returncode}: {' '.join(command)}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    return {
        "command": command,
        "wall_seconds": round(elapsed, 6),
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def main() -> None:
    args = parse_args()
    input_path = args.input_wav.resolve()
    output_root = args.output_root.resolve()
    cli_path = args.encodec_cli.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    ecdc_path = output_root / "meta-cli.ecdc"
    decoded_path = output_root / "meta-cli-decoded.wav"

    info = sf.info(input_path)
    audio_seconds = info.frames / info.samplerate
    environment = os.environ.copy()
    environment.update(
        {
            "OMP_NUM_THREADS": str(args.threads),
            "MKL_NUM_THREADS": str(args.threads),
            "VECLIB_MAXIMUM_THREADS": str(args.threads),
            "NUMEXPR_NUM_THREADS": str(args.threads),
        }
    )

    encode_command = [
        str(cli_path),
        "--hq",
        "--bandwidth",
        str(args.bandwidth),
        "--lm",
        "--force",
        str(input_path),
        str(ecdc_path),
    ]
    print("[meta-encodec-cli] timing standard CLI compression", file=sys.stderr, flush=True)
    encode = run_timed(encode_command, environment)

    decode_command = [
        str(cli_path),
        "--force",
        str(ecdc_path),
        str(decoded_path),
    ]
    print("[meta-encodec-cli] timing standard CLI decompression", file=sys.stderr, flush=True)
    decode = run_timed(decode_command, environment)

    report = {
        "schema": "wavey.encodec.meta-standard-cli-benchmark",
        "schema_version": 1,
        "status": "passed",
        "upstream_commit": PINNED_COMMIT,
        "environment": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python_executable": str(cli_path),
            "threads": args.threads,
            "checkpoint_files_cached_on_disk": True,
        },
        "source": {
            "path": str(input_path),
            "sha256": sha256_file(input_path),
            "sample_rate": info.samplerate,
            "channels": info.channels,
            "samples_per_channel": info.frames,
            "seconds": round(audio_seconds, 6),
        },
        "configuration": {
            "entrypoint": "official encodec CLI",
            "model": "encodec_48khz",
            "bandwidth_kbps": args.bandwidth,
            "use_lm": True,
            "fresh_process_per_command": True,
            "includes_model_and_lm_setup": True,
            "includes_wav_and_ecdc_io": True,
        },
        "encode": {
            **encode,
            "rtfx": round(audio_seconds / encode["wall_seconds"], 6),
            "ecdc_path": str(ecdc_path),
            "ecdc_bytes": ecdc_path.stat().st_size,
            "ecdc_sha256": sha256_file(ecdc_path),
        },
        "decode": {
            **decode,
            "rtfx": round(audio_seconds / decode["wall_seconds"], 6),
            "wav_path": str(decoded_path),
            "wav_sha256": sha256_file(decoded_path),
            "wav_subtype": sf.info(decoded_path).subtype,
        },
    }
    report_path = output_root / "report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
