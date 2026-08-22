#!/usr/bin/env python3
"""Run the high-frequency sidecar spike across a complete audio file."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path


VARIATIONS = [
    "00-source.wav",
    "01-encodec-baseline.wav",
    "02-oracle-hf-replacement.wav",
    "03-envelope-only.wav",
    "04-sidecar-envelope-excitation.wav",
    "05-mid-grid-sidecar.wav",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--ecdc", type=Path, required=True)
    parser.add_argument("--hard-crop", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--chunk-seconds", type=float, default=24.0)
    parser.add_argument(
        "--sidecar-script",
        type=Path,
        default=Path(__file__).with_name("spike-hf-sidecar.py"),
    )
    return parser.parse_args()


def run(command: list[str]) -> None:
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode:
        if result.stdout:
            print(result.stdout, file=sys.stderr)
        if result.stderr:
            print(result.stderr, file=sys.stderr)
        raise subprocess.CalledProcessError(result.returncode, command)


def probe_audio(path: Path) -> dict[str, int | float]:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "a:0",
        "-show_entries",
        "stream=sample_rate,channels,duration",
        "-of",
        "json",
        str(path),
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    stream = json.loads(result.stdout)["streams"][0]
    sample_rate = int(stream["sample_rate"])
    duration = float(stream["duration"])
    return {
        "sample_rate": sample_rate,
        "channels": int(stream["channels"]),
        "duration": duration,
        "samples": int(round(duration * sample_rate)),
    }


def extract_samples(
    source: Path,
    output: Path,
    start_sample: int,
    end_sample: int,
    sample_rate: int,
    channels: int,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-i",
            str(source),
            "-map",
            "0:a:0",
            "-af",
            f"atrim=start_sample={start_sample}:end_sample={end_sample},asetpts=PTS-STARTPTS",
            "-ar",
            str(sample_rate),
            "-ac",
            str(channels),
            "-c:a",
            "pcm_f32le",
            str(output),
        ]
    )


def concat_wavs(inputs: list[Path], output: Path, manifest: Path) -> None:
    for input_path in inputs:
        if "'" in str(input_path.resolve()):
            raise ValueError("concat input path contains an apostrophe")
    manifest.write_text(
        "".join(f"file '{path.resolve()}'\n" for path in inputs)
    )
    run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(manifest),
            "-c:a",
            "copy",
            str(output),
        ]
    )


def weighted_band_metrics(reports: list[dict[str, object]]) -> dict[str, object]:
    output: dict[str, object] = {}
    names = reports[0]["metrics"].keys()
    total_duration = sum(float(report["sizes"]["duration_seconds"]) for report in reports)
    for name in names:
        rows = []
        first_rows = reports[0]["metrics"][name]["bands"]
        for index, first in enumerate(first_rows):
            magnitude_sum = 0.0
            energy_bias_sum = 0.0
            for report in reports:
                weight = float(report["sizes"]["duration_seconds"])
                row = report["metrics"][name]["bands"][index]
                magnitude_sum += float(row["log_magnitude_mae_db"]) * weight
                energy_bias_sum += float(row["energy_bias_db"]) * weight
            rows.append(
                {
                    "band_hz": first["band_hz"],
                    "chunk_weighted_log_magnitude_mae_db": magnitude_sum / total_duration,
                    "chunk_weighted_energy_bias_db": energy_bias_sum / total_duration,
                }
            )
        output[name] = {"bands": rows}
    return output


def main() -> None:
    args = parse_args()
    source_info = probe_audio(args.source)
    baseline_info = probe_audio(args.baseline)
    for field in ["sample_rate", "channels", "samples"]:
        if source_info[field] != baseline_info[field]:
            raise ValueError(
                f"source and baseline {field} differ: "
                f"{source_info[field]} != {baseline_info[field]}"
            )

    sample_rate = int(source_info["sample_rate"])
    channels = int(source_info["channels"])
    total_samples = int(source_info["samples"])
    chunk_samples = int(round(args.chunk_seconds * sample_rate))
    if chunk_samples <= 0:
        raise ValueError("chunk duration must be positive")
    chunk_count = math.ceil(total_samples / chunk_samples)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.work_dir.mkdir(parents=True, exist_ok=True)
    reports: list[dict[str, object]] = []
    manifest_chunks: list[dict[str, object]] = []

    for chunk_index in range(chunk_count):
        start_sample = chunk_index * chunk_samples
        end_sample = min(total_samples, start_sample + chunk_samples)
        chunk_dir = args.work_dir / f"chunk-{chunk_index:03d}"
        chunk_dir.mkdir(parents=True, exist_ok=True)
        source_chunk = chunk_dir / "source-input.wav"
        baseline_chunk = chunk_dir / "baseline-input.wav"
        extract_samples(
            args.source,
            source_chunk,
            start_sample,
            end_sample,
            sample_rate,
            channels,
        )
        extract_samples(
            args.baseline,
            baseline_chunk,
            start_sample,
            end_sample,
            sample_rate,
            channels,
        )
        run(
            [
                sys.executable,
                str(args.sidecar_script),
                "--source",
                str(source_chunk),
                "--baseline",
                str(baseline_chunk),
                "--ecdc",
                str(chunk_dir / "not-present.ecdc"),
                "--output-dir",
                str(chunk_dir),
            ]
        )
        report = json.loads((chunk_dir / "report.json").read_text())
        reports.append(report)
        coarse_path = chunk_dir / "westside-hf-sidecar.ehf"
        grid_path = chunk_dir / "westside-mid-grid.ehg"
        manifest_chunks.append(
            {
                "index": chunk_index,
                "start_sample": start_sample,
                "samples": end_sample - start_sample,
                "coarse_sidecar": str(coarse_path.resolve()),
                "coarse_bytes": coarse_path.stat().st_size,
                "mid_grid_sidecar": str(grid_path.resolve()),
                "mid_grid_bytes": grid_path.stat().st_size,
            }
        )
        source_chunk.unlink()
        baseline_chunk.unlink()
        print(
            f"[hf-full] chunk {chunk_index + 1}/{chunk_count} "
            f"({start_sample / sample_rate:.3f}-{end_sample / sample_rate:.3f}s)",
            flush=True,
        )

    concat_root = args.work_dir / "concat"
    concat_root.mkdir(parents=True, exist_ok=True)
    for variation in VARIATIONS:
        inputs = [args.work_dir / f"chunk-{index:03d}" / variation for index in range(chunk_count)]
        concat_wavs(inputs, args.output_dir / variation, concat_root / f"{variation}.txt")

    if args.hard_crop and args.hard_crop.is_file():
        shutil.copy2(args.hard_crop, args.output_dir / "01b-encodec-hard-crop.wav")

    output_probes = {
        variation: probe_audio(args.output_dir / variation) for variation in VARIATIONS
    }
    for variation, info in output_probes.items():
        if info["samples"] != total_samples:
            raise ValueError(
                f"{variation} has {info['samples']} samples; expected {total_samples}"
            )

    duration = total_samples / sample_rate
    ecdc_bytes = args.ecdc.stat().st_size
    coarse_bytes = sum(int(chunk["coarse_bytes"]) for chunk in manifest_chunks)
    grid_bytes = sum(int(chunk["mid_grid_bytes"]) for chunk in manifest_chunks)
    ecdc_kbps = ecdc_bytes * 8.0 / duration / 1_000.0
    coarse_kbps = coarse_bytes * 8.0 / duration / 1_000.0
    grid_kbps = grid_bytes * 8.0 / duration / 1_000.0
    manifest = {
        "schema": "wavey.encodec.hf-sidecar-full-spike.v1",
        "source": str(args.source.resolve()),
        "baseline": str(args.baseline.resolve()),
        "ecdc": str(args.ecdc.resolve()),
        "sample_rate": sample_rate,
        "channels": channels,
        "samples": total_samples,
        "duration_seconds": duration,
        "chunk_samples": chunk_samples,
        "chunks": manifest_chunks,
        "sizes": {
            "ecdc_bytes": ecdc_bytes,
            "ecdc_effective_kbps": ecdc_kbps,
            "coarse_sidecar_bytes": coarse_bytes,
            "coarse_sidecar_kbps": coarse_kbps,
            "coarse_combined_kbps": ecdc_kbps + coarse_kbps,
            "mid_grid_sidecar_bytes": grid_bytes,
            "mid_grid_sidecar_kbps": grid_kbps,
            "mid_grid_combined_kbps": ecdc_kbps + grid_kbps,
        },
        "outputs": {
            variation: str((args.output_dir / variation).resolve())
            for variation in VARIATIONS
        },
        "output_probes": output_probes,
        "metrics": weighted_band_metrics(reports),
    }
    report_path = args.output_dir / "full-report.json"
    report_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest["sizes"], indent=2))
    print(f"[hf-full] outputs: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
