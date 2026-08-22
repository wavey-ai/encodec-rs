#!/usr/bin/env python3
"""Measure full-file objective and sampled ViSQOL HF sidecar quality."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import statistics
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal, stats


SAMPLE_RATE = 48_000
CHANNELS = 2
EXCERPT_SECONDS = 8.0
EXCERPT_COUNT = 4
RANDOM_SEED = 20_260_822
BANDS_HZ = [
    (0, 4_000),
    (4_000, 8_000),
    (8_000, 12_000),
    (12_000, 16_000),
    (16_000, 20_000),
    (20_000, 24_000),
]
FILES = {
    "baseline": "01-encodec-baseline.wav",
    "hard_crop": "01b-encodec-hard-crop.wav",
    "oracle": "02-oracle-hf-replacement.wav",
    "envelope": "03-envelope-only.wav",
    "excitation": "04-sidecar-envelope-excitation.wav",
    "mid_grid": "05-mid-grid-sidecar.wav",
}
VISQOL_TREATMENTS = ("source_control", *FILES.keys())


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--variation-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--visqol",
        type=Path,
        default=repo_root / "target/third-party/visqol/bazel-bin/visqol",
    )
    parser.add_argument("--headroom-db", type=float, default=0.1)
    parser.add_argument("--sample-count", type=int, default=EXCERPT_COUNT)
    parser.add_argument("--sample-seconds", type=float, default=EXCERPT_SECONDS)
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED)
    return parser.parse_args()


def load_quality_module() -> object:
    path = Path(__file__).resolve().parents[2] / "scripts/compare-encodec-quality.py"
    module_spec = importlib.util.spec_from_file_location("encodec_quality", path)
    if module_spec is None or module_spec.loader is None:
        raise RuntimeError(f"cannot load quality module from {path}")
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def band_metrics(reference: np.ndarray, candidate: np.ndarray) -> list[dict[str, object]]:
    n_fft = 2_048
    hop = 512
    accumulators = [
        {
            "signal": 0.0,
            "error": 0.0,
            "candidate": 0.0,
            "log_error": 0.0,
            "count": 0,
        }
        for _ in BANDS_HZ
    ]
    frequencies = np.fft.rfftfreq(n_fft, 1.0 / SAMPLE_RATE)
    for start in range(0, reference.shape[0], SAMPLE_RATE * 24):
        end = min(reference.shape[0], start + SAMPLE_RATE * 24)
        if end - start < n_fft:
            continue
        _, _, reference_stft = signal.stft(
            reference[start:end],
            fs=SAMPLE_RATE,
            window="hann",
            nperseg=n_fft,
            noverlap=n_fft - hop,
            nfft=n_fft,
            boundary=None,
            padded=False,
            axis=0,
        )
        _, _, candidate_stft = signal.stft(
            candidate[start:end],
            fs=SAMPLE_RATE,
            window="hann",
            nperseg=n_fft,
            noverlap=n_fft - hop,
            nfft=n_fft,
            boundary=None,
            padded=False,
            axis=0,
        )
        for index, (low, high) in enumerate(BANDS_HZ):
            mask = (frequencies >= low) & (frequencies < high)
            if high == SAMPLE_RATE // 2:
                mask = (frequencies >= low) & (frequencies <= high)
            ref = reference_stft[mask]
            output = candidate_stft[mask]
            ref_magnitude = np.abs(ref).astype(np.float64)
            output_magnitude = np.abs(output).astype(np.float64)
            delta = output - ref
            accumulators[index]["signal"] += float(np.sum(ref_magnitude**2))
            accumulators[index]["error"] += float(np.sum(np.abs(delta) ** 2))
            accumulators[index]["candidate"] += float(np.sum(output_magnitude**2))
            ref_db = 20.0 * np.log10(np.maximum(ref_magnitude, 1e-7))
            output_db = 20.0 * np.log10(np.maximum(output_magnitude, 1e-7))
            accumulators[index]["log_error"] += float(np.sum(np.abs(output_db - ref_db)))
            accumulators[index]["count"] += ref_db.size

    rows = []
    for (low, high), values in zip(BANDS_HZ, accumulators, strict=True):
        rows.append(
            {
                "band_hz": [low, high],
                "complex_snr_db": 10.0
                * math.log10(max(values["signal"], 1e-30) / max(values["error"], 1e-30)),
                "log_magnitude_mae_db": values["log_error"] / values["count"],
                "energy_bias_db": 10.0
                * math.log10(
                    max(values["candidate"], 1e-30) / max(values["signal"], 1e-30)
                ),
            }
        )
    return rows


def select_random_excerpts(
    sample_count: int,
    excerpt_seconds: float,
    count: int,
    seed: int,
) -> list[dict[str, object]]:
    length = int(round(excerpt_seconds * SAMPLE_RATE))
    latest = sample_count - length
    if length <= 0 or latest < 0:
        raise ValueError("sample duration must fit inside the source")
    if count <= 0:
        raise ValueError("sample count must be positive")

    # Use a one-second grid and reject overlaps. This keeps locations
    # reproducible and makes the listening timestamps easy to use.
    candidates = np.arange(0, latest + 1, SAMPLE_RATE, dtype=np.int64)
    rng = np.random.default_rng(seed)
    selected: list[int] = []
    for candidate in rng.permutation(candidates):
        start = int(candidate)
        if all(abs(start - other) >= length for other in selected):
            selected.append(start)
            if len(selected) == count:
                break
    if len(selected) != count:
        raise ValueError("not enough non-overlapping random sample positions")

    return [
        {
            "index": index,
            "start_sample": start,
            "end_sample": start + length,
            "start_seconds": start / SAMPLE_RATE,
            "seconds": excerpt_seconds,
            "random_seed": seed,
        }
        for index, start in enumerate(sorted(selected))
    ]


def source_commit(root: Path) -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def read_slice(path: Path, start: int, frames: int) -> np.ndarray:
    with sf.SoundFile(path) as handle:
        handle.seek(start)
        audio = handle.read(frames, dtype="float32", always_2d=True)
    if audio.shape != (frames, CHANNELS):
        raise ValueError(f"{path} returned shape {audio.shape}; expected {(frames, CHANNELS)}")
    return audio


def prepare_visqol_inputs(
    source: Path,
    paths: dict[str, Path],
    excerpt_sets: dict[str, list[dict[str, object]]],
    output_root: Path,
    headroom_db: float,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    pcm_root = output_root / "pcm16"
    pcm_root.mkdir(parents=True, exist_ok=True)
    target_peak = 10.0 ** (-headroom_db / 20.0)
    pairs: list[dict[str, object]] = []
    normalization = []
    for set_name, excerpts in excerpt_sets.items():
        for excerpt in excerpts:
            index = int(excerpt["index"])
            start = int(excerpt["start_sample"])
            frames = int(excerpt["end_sample"]) - start
            audio = {"source_control": read_slice(source, start, frames)}
            audio.update(
                {name: read_slice(path, start, frames) for name, path in paths.items()}
            )
            peak = max(float(np.max(np.abs(values))) for values in audio.values())
            gain = min(1.0, target_peak / max(peak, np.finfo(np.float32).tiny))
            reference_path = pcm_root / f"{set_name}-{index:02d}-reference.wav"
            sf.write(reference_path, audio["source_control"] * gain, SAMPLE_RATE, subtype="PCM_16")
            for treatment in VISQOL_TREATMENTS:
                degraded_path = pcm_root / f"{set_name}-{index:02d}-{treatment}.wav"
                sf.write(degraded_path, audio[treatment] * gain, SAMPLE_RATE, subtype="PCM_16")
                pairs.append(
                    {
                        "set": set_name,
                        "index": index,
                        "start_seconds": excerpt["start_seconds"],
                        "treatment": treatment,
                        "reference": reference_path.resolve(),
                        "degraded": degraded_path.resolve(),
                    }
                )
            normalization.append(
                {
                    "set": set_name,
                    "index": index,
                    "start_seconds": excerpt["start_seconds"],
                    "shared_peak": peak,
                    "shared_gain": gain,
                    "shared_gain_db": 20.0 * math.log10(gain),
                }
            )
    return pairs, normalization


def run_visqol(
    executable: Path,
    output_root: Path,
    pairs: list[dict[str, object]],
) -> list[dict[str, object]]:
    # Keep the Bazel symlink path. Resolving it moves the apparent repository
    # root into Bazel's temporary execroot, where the model file is not stored.
    executable = executable.absolute()
    visqol_root = executable.parent.parent
    model = visqol_root / "model/libsvm_nu_svr_model.txt"
    input_csv = output_root / "pairs.csv"
    result_csv = output_root / "results.csv"
    with input_csv.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["reference", "degraded"])
        writer.writerows((pair["reference"], pair["degraded"]) for pair in pairs)
    result_csv.unlink(missing_ok=True)
    result = subprocess.run(
        [
            str(executable),
            "--batch_input_csv",
            str(input_csv.resolve()),
            "--results_csv",
            str(result_csv.resolve()),
            "--similarity_to_quality_model",
            str(model.resolve()),
            "--use_lattice_model=false",
            "--disable_global_alignment=true",
        ],
        cwd=visqol_root,
        capture_output=True,
        text=True,
    )
    if result.returncode:
        print(result.stdout)
        print(result.stderr)
        result.check_returncode()
    with result_csv.open(newline="") as handle:
        results = list(csv.DictReader(handle))
    if len(results) != len(pairs):
        raise ValueError(f"ViSQOL returned {len(results)} rows; expected {len(pairs)}")
    rows = []
    for pair, result_row in zip(pairs, results, strict=True):
        rows.append(
            {
                "set": pair["set"],
                "index": pair["index"],
                "start_seconds": pair["start_seconds"],
                "treatment": pair["treatment"],
                "mos_lqo": float(result_row["moslqo"]),
            }
        )
    return rows


def confidence_interval(values: list[float]) -> list[float] | None:
    if len(values) < 2:
        return None
    mean = statistics.fmean(values)
    if all(value == mean for value in values):
        return [mean, mean]
    standard_error = stats.sem(values)
    interval = stats.t.interval(0.95, len(values) - 1, loc=mean, scale=standard_error)
    return [float(interval[0]), float(interval[1])]


def summarize_visqol(
    rows: list[dict[str, object]],
    excerpt_sets: dict[str, list[dict[str, object]]],
) -> dict[str, object]:
    output: dict[str, object] = {}
    for set_name in excerpt_sets:
        set_rows = [row for row in rows if row["set"] == set_name]
        by_treatment = {
            treatment: [
                float(row["mos_lqo"])
                for row in set_rows
                if row["treatment"] == treatment
            ]
            for treatment in VISQOL_TREATMENTS
        }
        baseline = by_treatment["baseline"]
        summary = {}
        for treatment, values in by_treatment.items():
            differences = [
                value - baseline[index] for index, value in enumerate(values)
            ]
            summary[treatment] = {
                "count": len(values),
                "mean": statistics.fmean(values),
                "median": statistics.median(values),
                "standard_deviation": statistics.stdev(values) if len(values) > 1 else None,
                "minimum": min(values),
                "maximum": max(values),
                "paired_delta_vs_baseline_mean": statistics.fmean(differences),
                "paired_delta_vs_baseline_ci95": confidence_interval(differences),
                "wins_vs_baseline": sum(value > 0 for value in differences),
                "losses_vs_baseline": sum(value < 0 for value in differences),
                "ties_vs_baseline": sum(value == 0 for value in differences),
            }
        output[set_name] = summary
    return output


def write_summary_markdown(report: dict[str, object], path: Path) -> None:
    lines = [
        "# Westside V2 HF sidecar quality audit",
        "",
        "## ViSQOL audio MOS-LQO",
        "",
    ]
    for set_name in report["visqol"]["summary"]:
        lines.extend(
            [
                f"### {set_name.replace('_', ' ').title()}",
                "",
                "| Candidate | Mean | Delta vs baseline | 95% CI |",
                "|---|---:|---:|---:|",
            ]
        )
        for treatment, values in report["visqol"]["summary"][set_name].items():
            interval = values["paired_delta_vs_baseline_ci95"]
            interval_text = (
                f"{interval[0]:+.4f} to {interval[1]:+.4f}" if interval else "n/a"
            )
            lines.append(
                f"| {treatment} | {values['mean']:.4f} | "
                f"{values['paired_delta_vs_baseline_mean']:+.4f} | {interval_text} |"
            )
        lines.append("")

    lines.extend(
        [
            "## Full-file objective metrics",
            "",
            "| Candidate | SNR dB | SI-SDR dB | Segmental SNR dB | LSD dB | Spectral convergence | Loudness delta LU |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for treatment, values in report["objective"].items():
        lines.append(
            f"| {treatment} | {values['snr_db']:.3f} | "
            f"{values['scale_invariant_sdr_db']:.3f} | "
            f"{values['segmental_snr']['mean_db']:.3f} | "
            f"{values['spectral']['log_spectral_distance_db']:.3f} | "
            f"{values['spectral']['spectral_convergence']:.5f} | "
            f"{values['loudness_delta_lu']:+.3f} |"
        )
    lines.extend(
        [
            "",
            "ViSQOL audio mode downmixes stereo to mono. Its audio model was trained at bitrates of 24 kbps and above.",
            "Treat small score changes at this lower bitrate as secondary evidence.",
            "",
        ]
    )
    path.write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    source_path = args.source.resolve()
    variation_root = args.variation_root.resolve()
    output_root = args.output_root.resolve()
    visqol_path = args.visqol.absolute()
    output_root.mkdir(parents=True, exist_ok=True)
    paths = {name: (variation_root / filename).resolve() for name, filename in FILES.items()}
    missing = [str(path) for path in [source_path, visqol_path, *paths.values()] if not path.is_file()]
    if missing:
        raise ValueError(f"required files are missing: {missing}")

    quality = load_quality_module()
    reference = quality.load_audio(source_path)
    reference_loudness = quality.integrated_loudness(reference)
    objective = {}
    artifacts = {
        "source": {"path": str(source_path), "sha256": sha256_file(source_path)}
    }
    for name, path in paths.items():
        print(f"[quality] full-file metrics: {name}", flush=True)
        candidate = quality.load_audio(path)
        metrics = quality.full_metrics(reference, candidate, reference_loudness)
        metrics["bands"] = band_metrics(reference, candidate)
        objective[name] = metrics
        artifacts[name] = {"path": str(path), "sha256": sha256_file(path)}
        del candidate

    excerpt_sets = {
        "random_samples": select_random_excerpts(
            reference.shape[0],
            args.sample_seconds,
            args.sample_count,
            args.random_seed,
        )
    }
    visqol_root = output_root / "visqol"
    visqol_root.mkdir(parents=True, exist_ok=True)
    pairs, normalization = prepare_visqol_inputs(
        source_path,
        paths,
        excerpt_sets,
        visqol_root,
        args.headroom_db,
    )
    print(f"[quality] ViSQOL pairs: {len(pairs)}", flush=True)
    visqol_rows = run_visqol(visqol_path, visqol_root, pairs)
    visqol_summary = summarize_visqol(visqol_rows, excerpt_sets)

    report = {
        "schema": "wavey.encodec.hf-sidecar-quality-audit.v1",
        "configuration": {
            "sample_rate": SAMPLE_RATE,
            "channels": CHANNELS,
            "visqol_scope": "reproducible random samples",
            "duration_seconds": reference.shape[0] / SAMPLE_RATE,
            "visqol_sample_count": args.sample_count,
            "visqol_sample_seconds": args.sample_seconds,
            "visqol_random_seed": args.random_seed,
            "visqol_global_alignment": False,
            "headroom_db": args.headroom_db,
            "visqol_mode": "audio",
            "visqol_input_subtype": "PCM_16",
            "visqol_commit": source_commit(visqol_path.parent.parent),
            "visqol_model": str(
                (visqol_path.parent.parent / "model/libsvm_nu_svr_model.txt")
            ),
        },
        "artifacts": artifacts,
        "objective": objective,
        "excerpt_sets": excerpt_sets,
        "visqol": {
            "summary": visqol_summary,
            "scores": visqol_rows,
            "normalization": normalization,
        },
    }
    report_path = output_root / "quality-report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    write_summary_markdown(report, output_root / "SUMMARY.md")
    print(f"[quality] report: {report_path}")


if __name__ == "__main__":
    main()
