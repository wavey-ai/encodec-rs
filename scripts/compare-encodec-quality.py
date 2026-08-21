#!/usr/bin/env python3
"""Compare full-file codec quality and inspect model-segment joins."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyloudnorm as pyln
import soundfile as sf
from scipy import signal


SAMPLE_RATE = 48_000
CHANNELS = 2
BLOCK_SAMPLES = SAMPLE_RATE * 20
SEAM_HALF_WINDOW = 480


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("master", type=Path)
    parser.add_argument("--encodec-rs", type=Path, required=True)
    parser.add_argument("--encodec-rs-pre-repair", type=Path, required=True)
    parser.add_argument("--meta", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--figure-root", type=Path, required=True)
    parser.add_argument("--excerpt-count", type=int, default=10)
    parser.add_argument("--excerpt-seconds", type=float, default=8.0)
    return parser.parse_args()


def load_audio(path: Path) -> np.ndarray:
    audio, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    if sample_rate != SAMPLE_RATE:
        raise ValueError(f"{path} is {sample_rate} Hz; expected {SAMPLE_RATE} Hz")
    if audio.shape[1] != CHANNELS:
        raise ValueError(f"{path} has {audio.shape[1]} channels; expected {CHANNELS}")
    if not np.isfinite(audio).all():
        raise ValueError(f"{path} contains non-finite samples")
    return audio


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ratio_db(numerator: float, denominator: float) -> float:
    return 10.0 * math.log10(max(numerator, 1.0e-30) / max(denominator, 1.0e-30))


def block_sums(reference: np.ndarray, candidate: np.ndarray) -> dict:
    ref_energy = np.zeros(CHANNELS, dtype=np.float64)
    candidate_energy = np.zeros(CHANNELS, dtype=np.float64)
    cross = np.zeros(CHANNELS, dtype=np.float64)
    error_energy = np.zeros(CHANNELS, dtype=np.float64)
    max_abs_error = 0.0
    for start in range(0, reference.shape[0], BLOCK_SAMPLES):
        end = min(reference.shape[0], start + BLOCK_SAMPLES)
        ref = reference[start:end].astype(np.float64)
        output = candidate[start:end].astype(np.float64)
        error = output - ref
        ref_energy += np.sum(ref * ref, axis=0)
        candidate_energy += np.sum(output * output, axis=0)
        cross += np.sum(ref * output, axis=0)
        error_energy += np.sum(error * error, axis=0)
        max_abs_error = max(max_abs_error, float(np.max(np.abs(error))))
    return {
        "reference_energy": ref_energy,
        "candidate_energy": candidate_energy,
        "cross": cross,
        "error_energy": error_energy,
        "max_abs_error": max_abs_error,
    }


def estimate_alignment(reference: np.ndarray, candidate: np.ndarray) -> dict:
    window_samples = min(reference.shape[0], SAMPLE_RATE * 60)
    start = max(0, (reference.shape[0] - window_samples) // 2)
    stop = start + window_samples
    decimation = 8
    ref = reference[start:stop:decimation].mean(axis=1).astype(np.float64)
    output = candidate[start:stop:decimation].mean(axis=1).astype(np.float64)
    ref -= ref.mean()
    output -= output.mean()
    correlation = signal.correlate(output, ref, mode="full", method="fft")
    lags = signal.correlation_lags(output.size, ref.size, mode="full")
    max_lag = math.ceil(2048 / decimation)
    allowed = np.abs(lags) <= max_lag
    selected = int(np.argmax(correlation[allowed]))
    best_lag_decimated = int(lags[allowed][selected])
    return {
        "search_samples": 2048,
        "analysis_seconds": window_samples / SAMPLE_RATE,
        "decimation": decimation,
        "best_lag_samples": best_lag_decimated * decimation,
    }


def segmental_snr(reference: np.ndarray, candidate: np.ndarray) -> dict:
    window = int(0.020 * SAMPLE_RATE)
    hop = window // 2
    values = []
    for start in range(0, reference.shape[0] - window + 1, hop):
        ref = reference[start : start + window].astype(np.float64)
        error = candidate[start : start + window].astype(np.float64) - ref
        signal_power = float(np.mean(ref * ref))
        if signal_power < 10.0 ** (-70.0 / 10.0):
            continue
        error_power = float(np.mean(error * error))
        values.append(np.clip(ratio_db(signal_power, error_power), -10.0, 35.0))
    array = np.asarray(values, dtype=np.float64)
    return {
        "window_ms": 20.0,
        "hop_ms": 10.0,
        "active_floor_dbfs": -70.0,
        "active_windows": int(array.size),
        "mean_db": float(np.mean(array)),
        "median_db": float(np.median(array)),
        "p10_db": float(np.quantile(array, 0.10)),
    }


def spectral_metrics(reference: np.ndarray, candidate: np.ndarray) -> dict:
    n_fft = 2048
    hop = 512
    squared_log_difference = 0.0
    spectral_error = 0.0
    spectral_reference = 0.0
    bins = 0
    for start in range(0, reference.shape[0], SAMPLE_RATE * 30):
        end = min(reference.shape[0], start + SAMPLE_RATE * 30)
        if end - start < n_fft:
            continue
        ref = reference[start:end]
        output = candidate[start:end]
        _, _, ref_stft = signal.stft(
            ref,
            fs=SAMPLE_RATE,
            window="hann",
            nperseg=n_fft,
            noverlap=n_fft - hop,
            nfft=n_fft,
            boundary=None,
            padded=False,
            axis=0,
        )
        _, _, output_stft = signal.stft(
            output,
            fs=SAMPLE_RATE,
            window="hann",
            nperseg=n_fft,
            noverlap=n_fft - hop,
            nfft=n_fft,
            boundary=None,
            padded=False,
            axis=0,
        )
        ref_magnitude = np.abs(ref_stft).astype(np.float64)
        output_magnitude = np.abs(output_stft).astype(np.float64)
        ref_db = 20.0 * np.log10(np.maximum(ref_magnitude, 1.0e-7))
        output_db = 20.0 * np.log10(np.maximum(output_magnitude, 1.0e-7))
        difference = output_db - ref_db
        squared_log_difference += float(np.sum(difference * difference))
        delta = output_magnitude - ref_magnitude
        spectral_error += float(np.sum(delta * delta))
        spectral_reference += float(np.sum(ref_magnitude * ref_magnitude))
        bins += difference.size
    return {
        "n_fft": n_fft,
        "hop_samples": hop,
        "log_spectral_distance_db": math.sqrt(squared_log_difference / bins),
        "spectral_convergence": math.sqrt(spectral_error / spectral_reference),
    }


def integrated_loudness(audio: np.ndarray) -> float:
    meter = pyln.Meter(SAMPLE_RATE, block_size=0.400)
    return float(meter.integrated_loudness(audio))


def full_metrics(
    reference: np.ndarray,
    candidate: np.ndarray,
    reference_loudness: float,
) -> dict:
    compared = min(reference.shape[0], candidate.shape[0])
    ref = reference[:compared]
    output = candidate[:compared]
    sums = block_sums(ref, output)
    ref_energy = sums["reference_energy"]
    output_energy = sums["candidate_energy"]
    error_energy = sums["error_energy"]
    cross = sums["cross"]
    total_ref = float(np.sum(ref_energy))
    total_error = float(np.sum(error_energy))
    sample_values = compared * CHANNELS
    channel_snr = [ratio_db(ref_energy[index], error_energy[index]) for index in range(CHANNELS)]
    sisdr = []
    for channel in range(CHANNELS):
        scale = cross[channel] / max(ref_energy[channel], 1.0e-30)
        residual_energy = (
            output_energy[channel]
            - 2.0 * scale * cross[channel]
            + scale * scale * ref_energy[channel]
        )
        target_energy = scale * scale * ref_energy[channel]
        sisdr.append(ratio_db(target_energy, residual_energy))
    candidate_loudness = integrated_loudness(output)
    return {
        "reference_samples_per_channel": int(reference.shape[0]),
        "candidate_samples_per_channel": int(candidate.shape[0]),
        "length_match": reference.shape[0] == candidate.shape[0],
        "compared_samples_per_channel": compared,
        "alignment": estimate_alignment(ref, output),
        "snr_db": ratio_db(total_ref, total_error),
        "channel_snr_db": channel_snr,
        "scale_invariant_sdr_db": float(np.mean(sisdr)),
        "channel_scale_invariant_sdr_db": sisdr,
        "reference_rms_dbfs": 10.0 * math.log10(total_ref / sample_values),
        "candidate_rms_dbfs": 10.0 * math.log10(float(np.sum(output_energy)) / sample_values),
        "error_rms_dbfs": 10.0 * math.log10(total_error / sample_values),
        "max_abs_error": sums["max_abs_error"],
        "reference_integrated_lufs": reference_loudness,
        "candidate_integrated_lufs": candidate_loudness,
        "loudness_delta_lu": candidate_loudness - reference_loudness,
        "candidate_peak_dbfs": 20.0 * math.log10(max(float(np.max(np.abs(output))), 1.0e-30)),
        "candidate_clipped_values": int(np.count_nonzero(np.abs(output) >= 1.0)),
        "segmental_snr": segmental_snr(ref, output),
        "spectral": spectral_metrics(ref, output),
    }


def seam_rows(
    reference: np.ndarray,
    candidate: np.ndarray,
    stride: int,
    label: str,
) -> list[dict]:
    rows = []
    half = SEAM_HALF_WINDOW
    for boundary in range(stride, min(reference.shape[0], candidate.shape[0]), stride):
        if boundary - 3 * half < 0 or boundary + 3 * half > reference.shape[0]:
            continue
        seam_error = (
            candidate[boundary - half : boundary + half].astype(np.float64)
            - reference[boundary - half : boundary + half].astype(np.float64)
        )
        left_error = (
            candidate[boundary - 3 * half : boundary - 2 * half].astype(np.float64)
            - reference[boundary - 3 * half : boundary - 2 * half].astype(np.float64)
        )
        right_error = (
            candidate[boundary + 2 * half : boundary + 3 * half].astype(np.float64)
            - reference[boundary + 2 * half : boundary + 3 * half].astype(np.float64)
        )
        controls = np.concatenate([left_error, right_error], axis=0)
        seam_ref = reference[boundary - half : boundary + half].astype(np.float64)
        seam_mse = float(np.mean(seam_error * seam_error))
        control_mse = float(np.mean(controls * controls))
        signal_power = float(np.mean(seam_ref * seam_ref))
        output_step = candidate[boundary].astype(np.float64) - candidate[
            boundary - 1
        ].astype(np.float64)
        reference_step = reference[boundary].astype(np.float64) - reference[
            boundary - 1
        ].astype(np.float64)
        step_error = output_step - reference_step
        output_curve = candidate[boundary].astype(np.float64) - 0.5 * (
            candidate[boundary - 1].astype(np.float64)
            + candidate[boundary + 1].astype(np.float64)
        )
        reference_curve = reference[boundary].astype(np.float64) - 0.5 * (
            reference[boundary - 1].astype(np.float64)
            + reference[boundary + 1].astype(np.float64)
        )
        curve_error = output_curve - reference_curve
        rows.append(
            {
                "codec": label,
                "boundary_sample": boundary,
                "boundary_seconds": boundary / SAMPLE_RATE,
                "seam_window_ms": 2 * half * 1000 / SAMPLE_RATE,
                "seam_mse": seam_mse,
                "control_mse": control_mse,
                "seam_excess_error_db": ratio_db(seam_mse, control_mse),
                "seam_snr_db": ratio_db(signal_power, seam_mse),
                "step_error_rms": float(math.sqrt(np.mean(step_error * step_error))),
                "curvature_error_rms": float(math.sqrt(np.mean(curve_error * curve_error))),
                "master_rms_dbfs": 10.0 * math.log10(max(signal_power, 1.0e-30)),
            }
        )
    return rows


def summarize_seams(rows: list[dict]) -> dict:
    def values(key: str) -> np.ndarray:
        return np.asarray([row[key] for row in rows], dtype=np.float64)

    excess = values("seam_excess_error_db")
    seam_snr = values("seam_snr_db")
    step = values("step_error_rms")
    curve = values("curvature_error_rms")
    return {
        "seams": len(rows),
        "analysis_window_ms": rows[0]["seam_window_ms"] if rows else None,
        "median_seam_excess_error_db": float(np.median(excess)),
        "p90_seam_excess_error_db": float(np.quantile(excess, 0.90)),
        "worst_seam_excess_error_db": float(np.max(excess)),
        "median_seam_snr_db": float(np.median(seam_snr)),
        "p10_seam_snr_db": float(np.quantile(seam_snr, 0.10)),
        "median_step_error_rms": float(np.median(step)),
        "p90_step_error_rms": float(np.quantile(step, 0.90)),
        "median_curvature_error_rms": float(np.median(curve)),
        "p90_curvature_error_rms": float(np.quantile(curve, 0.90)),
        "worst_boundary_sample": int(rows[int(np.argmax(excess))]["boundary_sample"]),
        "worst_boundary_seconds": float(rows[int(np.argmax(excess))]["boundary_seconds"]),
    }


def repair_ablation(
    before: np.ndarray,
    after: np.ndarray,
    before_rows: list[dict],
    after_rows: list[dict],
) -> tuple[dict, list[dict]]:
    by_boundary_after = {row["boundary_sample"]: row for row in after_rows}
    comparisons = []
    for before_row in before_rows:
        after_row = by_boundary_after[before_row["boundary_sample"]]
        comparisons.append(
            {
                "boundary_sample": before_row["boundary_sample"],
                "boundary_seconds": before_row["boundary_seconds"],
                "error_improvement_db": ratio_db(
                    before_row["seam_mse"], after_row["seam_mse"]
                ),
                "step_error_improvement_db": ratio_db(
                    before_row["step_error_rms"] ** 2,
                    after_row["step_error_rms"] ** 2,
                ),
                "before_seam_snr_db": before_row["seam_snr_db"],
                "after_seam_snr_db": after_row["seam_snr_db"],
                "before_seam_excess_error_db": before_row["seam_excess_error_db"],
                "after_seam_excess_error_db": after_row["seam_excess_error_db"],
            }
        )

    changed_values = 0
    changed_energy = 0.0
    changed_peak = 0.0
    for start in range(0, before.shape[0], BLOCK_SAMPLES):
        difference = (
            after[start : start + BLOCK_SAMPLES].astype(np.float64)
            - before[start : start + BLOCK_SAMPLES].astype(np.float64)
        )
        changed_values += int(np.count_nonzero(difference))
        changed_energy += float(np.sum(difference * difference))
        changed_peak = max(changed_peak, float(np.max(np.abs(difference))))
    improvements = np.asarray(
        [row["error_improvement_db"] for row in comparisons], dtype=np.float64
    )
    step_improvements = np.asarray(
        [row["step_error_improvement_db"] for row in comparisons], dtype=np.float64
    )
    largest = comparisons[int(np.argmax(np.abs(improvements)))]
    return (
        {
            "seams": len(comparisons),
            "repair_samples_per_seam": 24,
            "changed_sample_values": changed_values,
            "changed_fraction": changed_values / before.size,
            "repair_delta_rms": math.sqrt(changed_energy / before.size),
            "repair_delta_peak": changed_peak,
            "improved_seams": int(np.count_nonzero(improvements > 0.0)),
            "degraded_seams": int(np.count_nonzero(improvements < 0.0)),
            "unchanged_seams": int(np.count_nonzero(improvements == 0.0)),
            "median_error_improvement_db": float(np.median(improvements)),
            "p10_error_improvement_db": float(np.quantile(improvements, 0.10)),
            "p90_error_improvement_db": float(np.quantile(improvements, 0.90)),
            "median_step_error_improvement_db": float(np.median(step_improvements)),
            "largest_absolute_effect": largest,
        },
        comparisons,
    )


def write_seam_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def mono(audio: np.ndarray) -> np.ndarray:
    return audio.mean(axis=1, dtype=np.float64)


def spectrogram_db(
    audio: np.ndarray,
    start: int,
    end: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = mono(audio[start:end])
    frequencies, times, magnitude = signal.spectrogram(
        values,
        fs=SAMPLE_RATE,
        window="hann",
        nperseg=256,
        noverlap=248,
        nfft=256,
        detrend=False,
        scaling="spectrum",
        mode="magnitude",
    )
    return frequencies, times, 20.0 * np.log10(np.maximum(magnitude, 1.0e-7))


def plot_join(
    reference: np.ndarray,
    series: list[tuple[str, np.ndarray]],
    boundary: int,
    title: str,
    output: Path,
) -> None:
    spectrogram_half = int(0.050 * SAMPLE_RATE)
    waveform_half = int(0.005 * SAMPLE_RATE)
    start = boundary - spectrogram_half
    end = boundary + spectrogram_half
    all_audio = [("Master", reference), *series]
    common_peak = max(
        float(np.max(np.abs(mono(audio[boundary - waveform_half : boundary + waveform_half]))))
        for _, audio in all_audio
    )
    figure, axes = plt.subplots(
        len(all_audio),
        2,
        figsize=(13.0, 2.25 * len(all_audio)),
        constrained_layout=True,
    )
    image = None
    for row, (label, audio) in enumerate(all_audio):
        waveform = mono(audio[boundary - waveform_half : boundary + waveform_half])
        waveform_time = (
            np.arange(-waveform_half, waveform_half, dtype=np.float64)
            / SAMPLE_RATE
            * 1000.0
        )
        axes[row, 0].plot(waveform_time, waveform, linewidth=0.8)
        axes[row, 0].axvline(0.0, color="tab:red", linewidth=0.8)
        axes[row, 0].set_ylim(-common_peak * 1.05, common_peak * 1.05)
        axes[row, 0].set_ylabel(label)
        axes[row, 0].grid(alpha=0.2)
        frequencies, times, db_values = spectrogram_db(audio, start, end)
        relative_time = (times - spectrogram_half / SAMPLE_RATE) * 1000.0
        image = axes[row, 1].pcolormesh(
            relative_time,
            frequencies / 1000.0,
            db_values,
            shading="auto",
            cmap="magma",
            vmin=-100.0,
            vmax=0.0,
        )
        axes[row, 1].axvline(0.0, color="cyan", linewidth=0.8)
        axes[row, 1].set_ylim(0.0, 24.0)
    axes[0, 0].set_title("Mid-channel waveform around join")
    axes[0, 1].set_title("Mid-channel spectrogram around join")
    axes[-1, 0].set_xlabel("Time from join (ms)")
    axes[-1, 1].set_xlabel("Time from join (ms)")
    for axis in axes[:, 1]:
        axis.set_ylabel("Frequency (kHz)")
    if image is not None:
        figure.colorbar(image, ax=axes[:, 1], label="Magnitude (dBFS)", shrink=0.85)
    figure.suptitle(title)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def plot_residual_join(
    reference: np.ndarray,
    series: list[tuple[str, np.ndarray]],
    boundary: int,
    title: str,
    output: Path,
) -> None:
    half = int(0.050 * SAMPLE_RATE)
    start = boundary - half
    end = boundary + half
    figure, axes = plt.subplots(
        len(series),
        1,
        figsize=(10.0, 2.35 * len(series)),
        constrained_layout=True,
        squeeze=False,
    )
    image = None
    for row, (label, audio) in enumerate(series):
        residual = audio[start:end] - reference[start:end]
        frequencies, times, db_values = spectrogram_db(residual, 0, residual.shape[0])
        relative_time = (times - half / SAMPLE_RATE) * 1000.0
        axis = axes[row, 0]
        image = axis.pcolormesh(
            relative_time,
            frequencies / 1000.0,
            db_values,
            shading="auto",
            cmap="magma",
            vmin=-110.0,
            vmax=-10.0,
        )
        axis.axvline(0.0, color="cyan", linewidth=0.8)
        axis.set_ylim(0.0, 24.0)
        axis.set_ylabel(f"{label}\nFrequency (kHz)")
    axes[-1, 0].set_xlabel("Time from join (ms)")
    if image is not None:
        figure.colorbar(image, ax=axes[:, 0], label="Residual magnitude (dBFS)", shrink=0.85)
    figure.suptitle(title)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def select_active_excerpts(reference: np.ndarray, count: int, seconds: float) -> list[dict]:
    length = int(round(seconds * SAMPLE_RATE))
    mono_power = np.mean(reference.astype(np.float64) ** 2, axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(mono_power)])
    duration = reference.shape[0]
    rows = []
    for index in range(count):
        region_start = int(index * duration / count)
        region_end = int((index + 1) * duration / count)
        latest = min(duration - length, region_end - length)
        if latest < region_start:
            continue
        starts = np.arange(region_start, latest + 1, SAMPLE_RATE, dtype=np.int64)
        energies = (cumulative[starts + length] - cumulative[starts]) / length
        selected = int(starts[int(np.argmax(energies))])
        rows.append(
            {
                "index": index,
                "start_sample": selected,
                "end_sample": selected + length,
                "start_seconds": selected / SAMPLE_RATE,
                "seconds": seconds,
                "master_rms_dbfs": 10.0 * math.log10(max(float(np.max(energies)), 1.0e-30)),
            }
        )
    return rows


def write_excerpts(
    output_root: Path,
    excerpts: list[dict],
    tracks: dict[str, np.ndarray],
) -> list[dict]:
    clip_root = output_root / "perceptual-excerpts"
    manifest = []
    for excerpt in excerpts:
        row = dict(excerpt)
        row["files"] = {}
        for label, audio in tracks.items():
            path = clip_root / f"{excerpt['index']:02d}-{label}.wav"
            path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(
                path,
                audio[excerpt["start_sample"] : excerpt["end_sample"]],
                SAMPLE_RATE,
                subtype="FLOAT",
            )
            row["files"][label] = str(path)
        manifest.append(row)
    return manifest


def main() -> None:
    args = parse_args()
    paths = {
        "master": args.master.resolve(),
        "encodec_rs": args.encodec_rs.resolve(),
        "encodec_rs_before_repair": args.encodec_rs_pre_repair.resolve(),
        "meta": args.meta.resolve(),
    }
    output_root = args.output_root.resolve()
    figure_root = args.figure_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    figure_root.mkdir(parents=True, exist_ok=True)

    audio = {label: load_audio(path) for label, path in paths.items()}
    reference = audio["master"]
    for label, values in audio.items():
        if values.shape != reference.shape:
            raise ValueError(
                f"{label} shape {values.shape} does not match master {reference.shape}"
            )

    reference_loudness = integrated_loudness(reference)
    metrics = {
        label: full_metrics(reference, values, reference_loudness)
        for label, values in audio.items()
        if label != "master"
    }

    seam_sets = {
        "encodec_rs_before_repair": seam_rows(
            reference, audio["encodec_rs_before_repair"], 64_000, "encodec_rs_before_repair"
        ),
        "encodec_rs": seam_rows(reference, audio["encodec_rs"], 64_000, "encodec_rs"),
        "meta": seam_rows(reference, audio["meta"], 47_520, "meta"),
    }
    seam_summary = {label: summarize_seams(rows) for label, rows in seam_sets.items()}
    ablation, repair_rows = repair_ablation(
        audio["encodec_rs_before_repair"],
        audio["encodec_rs"],
        seam_sets["encodec_rs_before_repair"],
        seam_sets["encodec_rs"],
    )

    all_seam_rows = [row for rows in seam_sets.values() for row in rows]
    write_seam_csv(output_root / "seams.csv", all_seam_rows)
    write_seam_csv(output_root / "repair-ablation.csv", repair_rows)

    effect_boundary = int(ablation["largest_absolute_effect"]["boundary_sample"])
    worst_repaired = int(seam_summary["encodec_rs"]["worst_boundary_sample"])
    worst_meta = int(seam_summary["meta"]["worst_boundary_sample"])
    plot_join(
        reference,
        [
            ("encodec-rs before repair", audio["encodec_rs_before_repair"]),
            ("encodec-rs after repair", audio["encodec_rs"]),
            ("Meta at the same time", audio["meta"]),
        ],
        effect_boundary,
        f"Largest encodec-rs seam-repair effect at {effect_boundary / SAMPLE_RATE:.3f} s",
        figure_root / "encodec-rs-largest-repair-effect.png",
    )
    plot_residual_join(
        reference,
        [
            ("Before repair", audio["encodec_rs_before_repair"]),
            ("After repair", audio["encodec_rs"]),
            ("Meta at same time", audio["meta"]),
        ],
        worst_repaired,
        (
            "Residual spectrogram at worst repaired encodec-rs join, "
            f"{worst_repaired / SAMPLE_RATE:.3f} s"
        ),
        figure_root / "encodec-rs-worst-join-residual.png",
    )
    plot_join(
        reference,
        [
            ("Meta overlap-add", audio["meta"]),
            ("encodec-rs at the same time", audio["encodec_rs"]),
        ],
        worst_meta,
        f"Worst Meta overlap-add join at {worst_meta / SAMPLE_RATE:.3f} s",
        figure_root / "meta-worst-overlap-join.png",
    )

    excerpts = select_active_excerpts(reference, args.excerpt_count, args.excerpt_seconds)
    excerpt_manifest = write_excerpts(
        output_root,
        excerpts,
        {
            "master": reference,
            "encodec-rs": audio["encodec_rs"],
            "meta": audio["meta"],
        },
    )

    report = {
        "schema": "wavey.encodec.full-file-quality-comparison",
        "schema_version": 1,
        "configuration": {
            "sample_rate": SAMPLE_RATE,
            "channels": CHANNELS,
            "seam_analysis_half_window_samples": SEAM_HALF_WINDOW,
            "seam_analysis_window_ms": 2 * SEAM_HALF_WINDOW * 1000 / SAMPLE_RATE,
            "encodec_rs_stride_samples": 64_000,
            "meta_stride_samples": 47_520,
        },
        "artifacts": {
            label: {
                "path": str(path),
                "sha256": sha256_file(path),
                "subtype": sf.info(path).subtype,
            }
            for label, path in paths.items()
        },
        "full_file_metrics": metrics,
        "seam_metrics": seam_summary,
        "repair_ablation": ablation,
        "perceptual_excerpt_manifest": excerpt_manifest,
        "figures": {
            "largest_repair_effect": str(figure_root / "encodec-rs-largest-repair-effect.png"),
            "worst_encodec_rs_residual": str(figure_root / "encodec-rs-worst-join-residual.png"),
            "worst_meta_overlap": str(figure_root / "meta-worst-overlap-join.png"),
        },
    }
    report_path = output_root / "quality.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
