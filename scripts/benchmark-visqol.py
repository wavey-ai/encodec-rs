#!/usr/bin/env python3
"""Score matched codec excerpts with the official ViSQOL audio model."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import stats


SAMPLE_RATE = 48_000
TREATMENTS = ("encodec-rs", "meta")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--visqol", type=Path, required=True)
    parser.add_argument("--excerpt-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--headroom-db", type=float, default=0.1)
    parser.add_argument("--reuse-results", action="store_true")
    return parser.parse_args()


def load_audio(path: Path) -> np.ndarray:
    audio, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    if sample_rate != SAMPLE_RATE:
        raise ValueError(f"{path} is {sample_rate} Hz; expected {SAMPLE_RATE} Hz")
    if not np.isfinite(audio).all():
        raise ValueError(f"{path} contains non-finite samples")
    return audio


def prepare_pcm16_excerpts(
    excerpt_root: Path,
    output_root: Path,
    headroom_db: float,
) -> tuple[list[dict], list[dict]]:
    pcm_root = output_root / "pcm16"
    pcm_root.mkdir(parents=True, exist_ok=True)
    pairs = []
    normalization = []
    master_paths = sorted(excerpt_root.glob("*-master.wav"))
    if not master_paths:
        raise ValueError(f"no *-master.wav excerpts found in {excerpt_root}")

    target_peak = 10.0 ** (-headroom_db / 20.0)
    for master_path in master_paths:
        index = master_path.name.split("-", 1)[0]
        source_paths = {"master": master_path}
        source_paths.update(
            {treatment: excerpt_root / f"{index}-{treatment}.wav" for treatment in TREATMENTS}
        )
        audio = {label: load_audio(path) for label, path in source_paths.items()}
        shapes = {values.shape for values in audio.values()}
        if len(shapes) != 1:
            raise ValueError(f"excerpt {index} has mismatched shapes: {sorted(shapes)}")
        peak = max(float(np.max(np.abs(values))) for values in audio.values())
        gain = min(1.0, target_peak / max(peak, np.finfo(np.float32).tiny))
        output_paths = {}
        for label, values in audio.items():
            path = (pcm_root / f"{index}-{label}.wav").resolve()
            sf.write(path, values * gain, SAMPLE_RATE, subtype="PCM_16")
            output_paths[label] = path
        normalization.append(
            {
                "index": int(index),
                "shared_input_peak": peak,
                "shared_gain": gain,
                "shared_gain_db": 20.0 * math.log10(gain),
            }
        )
        for treatment in TREATMENTS:
            pairs.append(
                {
                    "index": int(index),
                    "treatment": treatment,
                    "reference": output_paths["master"],
                    "degraded": output_paths[treatment],
                }
            )
    return pairs, normalization


def run_visqol(visqol: Path, model: Path, output_root: Path, pairs: list[dict]) -> Path:
    input_csv = output_root / "pairs.csv"
    result_csv = output_root / "results.csv"
    result_csv.unlink(missing_ok=True)
    with input_csv.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["reference", "degraded"])
        writer.writerows((row["reference"], row["degraded"]) for row in pairs)
    subprocess.run(
        [
            str(visqol),
            "--batch_input_csv",
            str(input_csv.resolve()),
            "--results_csv",
            str(result_csv.resolve()),
            "--similarity_to_quality_model",
            str(model),
            "--use_lattice_model=false",
        ],
        cwd=visqol.parent.parent,
        check=True,
    )
    return result_csv


def source_commit(visqol_root: Path) -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=visqol_root,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def main() -> None:
    args = parse_args()
    visqol = args.visqol.absolute()
    excerpt_root = args.excerpt_root.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    visqol_root = visqol.parent.parent
    model = (visqol_root / "model/libsvm_nu_svr_model.txt").resolve()
    if not visqol.is_file():
        raise ValueError(f"ViSQOL executable not found: {visqol}")
    if not model.is_file():
        raise ValueError(f"ViSQOL audio model not found: {model}")

    pairs, normalization = prepare_pcm16_excerpts(
        excerpt_root,
        output_root,
        args.headroom_db,
    )
    result_csv = output_root / "results.csv"
    if not args.reuse_results:
        result_csv = run_visqol(visqol, model, output_root, pairs)
    elif not result_csv.is_file():
        raise ValueError(f"existing ViSQOL results not found: {result_csv}")
    with result_csv.open(newline="") as handle:
        result_rows = list(csv.DictReader(handle))
    if len(result_rows) != len(pairs):
        raise ValueError(f"ViSQOL returned {len(result_rows)} rows; expected {len(pairs)}")

    rows = []
    scores = {treatment: [] for treatment in TREATMENTS}
    for pair, result in zip(pairs, result_rows, strict=True):
        score = float(result["moslqo"])
        rows.append(
            {
                "index": pair["index"],
                "treatment": pair["treatment"],
                "mos_lqo": score,
            }
        )
        scores[pair["treatment"]].append(score)

    summary = {
        treatment: {
            "count": len(values),
            "mean": statistics.fmean(values),
            "median": statistics.median(values),
            "standard_deviation": statistics.stdev(values),
            "minimum": min(values),
            "maximum": max(values),
        }
        for treatment, values in scores.items()
    }
    paired_differences = [
        encodec_rs - meta
        for encodec_rs, meta in zip(scores["encodec-rs"], scores["meta"], strict=True)
    ]
    paired_mean = statistics.fmean(paired_differences)
    paired_standard_error = stats.sem(paired_differences)
    paired_interval = stats.t.interval(
        0.95,
        len(paired_differences) - 1,
        loc=paired_mean,
        scale=paired_standard_error,
    )
    paired_summary = {
        "count": len(paired_differences),
        "mean": paired_mean,
        "median": statistics.median(paired_differences),
        "standard_deviation": statistics.stdev(paired_differences),
        "confidence_interval_95": [float(paired_interval[0]), float(paired_interval[1])],
        "encodec_rs_wins": sum(value > 0.0 for value in paired_differences),
        "meta_wins": sum(value < 0.0 for value in paired_differences),
        "ties": sum(value == 0.0 for value in paired_differences),
    }
    report = {
        "schema": "wavey.encodec.visqol-comparison",
        "schema_version": 1,
        "configuration": {
            "mode": "audio",
            "sample_rate": SAMPLE_RATE,
            "input_subtype": "PCM_16",
            "headroom_db": args.headroom_db,
            "shared_normalization_per_excerpt": True,
            "visqol_commit": source_commit(visqol_root),
            "model": str(model),
        },
        "summary": summary,
        "paired_encodec_rs_minus_meta": paired_summary,
        "scores": rows,
        "normalization": normalization,
    }
    report_path = output_root / "visqol.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
