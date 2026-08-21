#!/usr/bin/env python3

"""Create matched listening excerpts and spectrograms for seam strategies."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from scipy import signal


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("master", type=Path)
    parser.add_argument("pack_root", type=Path)
    parser.add_argument("--excerpt-seconds", type=float, default=8.0)
    return parser.parse_args()


def load_audio(path: Path, sample_rate: int, channels: int) -> np.ndarray:
    audio, actual_rate = sf.read(path, dtype="float32", always_2d=True)
    if actual_rate != sample_rate:
        raise ValueError(f"{path} uses {actual_rate} Hz; expected {sample_rate}")
    if audio.shape[1] != channels:
        raise ValueError(f"{path} has {audio.shape[1]} channels; expected {channels}")
    if not np.isfinite(audio).all():
        raise ValueError(f"{path} contains a non-finite sample")
    return audio


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def mono(audio: np.ndarray) -> np.ndarray:
    return audio.mean(axis=1, dtype=np.float64)


def find_worst_hard_join(
    master: np.ndarray,
    hard: np.ndarray,
    stride: int,
) -> tuple[int, list[dict]]:
    rows = []
    boundary = stride
    while boundary < hard.shape[0]:
        hard_step = hard[boundary].astype(np.float64) - hard[boundary - 1].astype(
            np.float64
        )
        master_step = master[boundary].astype(np.float64) - master[
            boundary - 1
        ].astype(np.float64)
        residual = hard_step - master_step
        rows.append(
            {
                "boundarySample": boundary,
                "boundarySeconds": boundary / 48_000,
                "hardStepErrorRms": float(math.sqrt(np.mean(residual**2))),
                "hardStepErrorPeak": float(np.max(np.abs(residual))),
            }
        )
        boundary += stride
    if not rows:
        raise ValueError("audio has no complete chunk join")
    worst = max(rows, key=lambda row: row["hardStepErrorRms"])
    return int(worst["boundarySample"]), rows


def spectrogram_db(
    audio: np.ndarray,
    sample_rate: int,
    nperseg: int,
    noverlap: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    frequencies, times, magnitude = signal.spectrogram(
        mono(audio),
        fs=sample_rate,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=nperseg,
        detrend=False,
        scaling="spectrum",
        mode="magnitude",
    )
    db = 20.0 * np.log10(np.maximum(magnitude, 1.0e-7))
    return frequencies, times, db


def plot_seam_detail(
    master: np.ndarray,
    candidate: np.ndarray,
    sample_rate: int,
    boundary: int,
    title: str,
    output: Path,
) -> None:
    spectrogram_half = int(round(0.050 * sample_rate))
    waveform_half = int(round(0.005 * sample_rate))
    spec_start = boundary - spectrogram_half
    spec_end = boundary + spectrogram_half
    wave_start = boundary - waveform_half
    wave_end = boundary + waveform_half
    master_wave = mono(master[wave_start:wave_end])
    candidate_wave = mono(candidate[wave_start:wave_end])
    common_peak = max(float(np.max(np.abs(master_wave))), float(np.max(np.abs(candidate_wave))))
    time_ms = (
        np.arange(-waveform_half, waveform_half, dtype=np.float64)
        / sample_rate
        * 1000.0
    )

    figure, axes = plt.subplots(1, 3, figsize=(17.0, 4.6), constrained_layout=True)
    axes[0].plot(time_ms, master_wave, color="0.55", linewidth=0.9, label="Master")
    axes[0].plot(time_ms, candidate_wave, color="tab:blue", linewidth=0.8, label="Candidate")
    axes[0].axvline(0.0, color="tab:red", linewidth=0.9)
    axes[0].set_ylim(-common_peak * 1.05, common_peak * 1.05)
    axes[0].set_xlabel("Time from join (ms)")
    axes[0].set_ylabel("Mid amplitude")
    axes[0].set_title("Waveform")
    axes[0].legend(loc="upper right")
    axes[0].grid(alpha=0.2)

    candidate_region = candidate[spec_start:spec_end]
    frequencies, times, candidate_db = spectrogram_db(
        candidate_region, sample_rate, 256, 248
    )
    relative_ms = (times - spectrogram_half / sample_rate) * 1000.0
    image = axes[1].pcolormesh(
        relative_ms,
        frequencies / 1000.0,
        candidate_db,
        shading="auto",
        cmap="magma",
        vmin=-100.0,
        vmax=0.0,
    )
    axes[1].axvline(0.0, color="cyan", linewidth=0.9)
    axes[1].set_ylim(0.0, 24.0)
    axes[1].set_xlabel("Time from join (ms)")
    axes[1].set_ylabel("Frequency (kHz)")
    axes[1].set_title("Candidate spectrogram")
    figure.colorbar(image, ax=axes[1], label="Magnitude (dBFS)")

    residual = candidate_region.astype(np.float64) - master[spec_start:spec_end].astype(
        np.float64
    )
    frequencies, times, residual_db = spectrogram_db(residual, sample_rate, 256, 248)
    residual_image = axes[2].pcolormesh(
        relative_ms,
        frequencies / 1000.0,
        residual_db,
        shading="auto",
        cmap="magma",
        vmin=-110.0,
        vmax=-10.0,
    )
    axes[2].axvline(0.0, color="cyan", linewidth=0.9)
    axes[2].set_ylim(0.0, 24.0)
    axes[2].set_xlabel("Time from join (ms)")
    axes[2].set_ylabel("Frequency (kHz)")
    axes[2].set_title("Decoded minus master")
    figure.colorbar(residual_image, ax=axes[2], label="Residual (dBFS)")
    figure.suptitle(title)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def plot_full_spectrogram(
    candidate: np.ndarray,
    sample_rate: int,
    title: str,
    output: Path,
) -> None:
    frequencies, times, db = spectrogram_db(candidate, sample_rate, 2048, 0)
    figure, axis = plt.subplots(figsize=(16.0, 4.5), constrained_layout=True)
    image = axis.pcolormesh(
        times,
        frequencies / 1000.0,
        db,
        shading="auto",
        cmap="magma",
        vmin=-100.0,
        vmax=0.0,
    )
    axis.set_ylim(0.0, 24.0)
    axis.set_xlabel("Time (seconds)")
    axis.set_ylabel("Frequency (kHz)")
    axis.set_title(title)
    figure.colorbar(image, ax=axis, label="Magnitude (dBFS)")
    figure.savefig(output, dpi=160)
    plt.close(figure)


def plot_combined_seam_spectrograms(
    master_region: np.ndarray,
    strategies: list[tuple[str, np.ndarray]],
    sample_rate: int,
    boundary_seconds: float,
    output: Path,
) -> None:
    rows = [("Master", master_region), *strategies]
    half_seconds = master_region.shape[0] / sample_rate / 2.0
    figure, axes = plt.subplots(
        len(rows),
        1,
        figsize=(12.0, 2.35 * len(rows)),
        constrained_layout=True,
        squeeze=False,
    )
    image = None
    for row, (label, audio) in enumerate(rows):
        frequencies, times, db = spectrogram_db(audio, sample_rate, 256, 248)
        relative_ms = (times - half_seconds) * 1000.0
        axis = axes[row, 0]
        image = axis.pcolormesh(
            relative_ms,
            frequencies / 1000.0,
            db,
            shading="auto",
            cmap="magma",
            vmin=-100.0,
            vmax=0.0,
        )
        axis.axvline(0.0, color="cyan", linewidth=0.9)
        axis.set_ylim(0.0, 24.0)
        axis.set_ylabel(f"{label}\nFrequency (kHz)")
    axes[-1, 0].set_xlabel("Time from join (ms)")
    if image is not None:
        figure.colorbar(
            image,
            ax=axes[:, 0],
            label="Magnitude (dBFS)",
            shrink=0.85,
        )
    figure.suptitle(
        f"Same decoded windows, seam strategies at {boundary_seconds:.3f} s"
    )
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    manifest_path = args.pack_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    sample_rate = int(manifest["sampleRate"])
    channels = int(manifest["channels"])
    audio_samples = int(manifest["audioSamples"])
    stride = int(manifest["ownedStrideSamples"])

    master = load_audio(args.master, sample_rate, channels)
    if master.shape[0] != audio_samples:
        raise ValueError(
            f"master has {master.shape[0]} samples; expected {audio_samples}"
        )

    hard_entry = next(
        strategy
        for strategy in manifest["strategies"]
        if strategy["id"] == "hard-owned-crop"
    )
    hard = load_audio(args.pack_root / hard_entry["wav"], sample_rate, channels)
    boundary, join_rows = find_worst_hard_join(master, hard, stride)
    del hard

    excerpt_frames = int(round(args.excerpt_seconds * sample_rate))
    excerpt_start = max(0, boundary - excerpt_frames // 2)
    excerpt_end = min(audio_samples, excerpt_start + excerpt_frames)
    excerpt_start = max(0, excerpt_end - excerpt_frames)
    figures_root = args.pack_root / "spectrograms"
    excerpts_root = args.pack_root / "listening-excerpts"
    figures_root.mkdir(parents=True, exist_ok=True)
    excerpts_root.mkdir(parents=True, exist_ok=True)

    strategy_reports = []
    detail_half = int(round(0.050 * sample_rate))
    comparison_regions = []
    for index, strategy in enumerate(manifest["strategies"], start=1):
        wav_path = args.pack_root / strategy["wav"]
        candidate = load_audio(wav_path, sample_rate, channels)
        if candidate.shape != master.shape:
            raise ValueError(
                f"{wav_path} shape {candidate.shape} does not match master {master.shape}"
            )
        candidate_step = candidate[boundary].astype(np.float64) - candidate[
            boundary - 1
        ].astype(np.float64)
        master_step = master[boundary].astype(np.float64) - master[
            boundary - 1
        ].astype(np.float64)
        step_error = candidate_step - master_step
        residual = candidate.astype(np.float64) - master.astype(np.float64)
        mse = float(np.mean(residual**2))
        signal_power = float(np.mean(master.astype(np.float64) ** 2))
        snr_db = 10.0 * math.log10(signal_power / mse)
        peak = float(np.max(np.abs(candidate)))
        rms = float(math.sqrt(np.mean(candidate.astype(np.float64) ** 2)))

        stem = f"{index:02d}-{strategy['id']}"
        excerpt_path = excerpts_root / f"{stem}-around-{boundary}.wav"
        sf.write(
            excerpt_path,
            candidate[excerpt_start:excerpt_end],
            sample_rate,
            subtype="FLOAT",
        )
        detail_path = figures_root / f"{stem}-seam-detail.png"
        full_path = figures_root / f"{stem}-full-spectrogram.png"
        plot_seam_detail(
            master,
            candidate,
            sample_rate,
            boundary,
            f"{strategy['id']} at {boundary / sample_rate:.3f} s",
            detail_path,
        )
        plot_full_spectrogram(
            candidate,
            sample_rate,
            f"{strategy['id']} — full song",
            full_path,
        )
        comparison_regions.append(
            (
                strategy["id"],
                candidate[boundary - detail_half : boundary + detail_half].copy(),
            )
        )
        strategy_reports.append(
            {
                "id": strategy["id"],
                "wav": strategy["wav"],
                "wavSha256": sha256_file(wav_path),
                "samples": int(candidate.shape[0]),
                "peak": peak,
                "rms": rms,
                "snrDbVsMaster": snr_db,
                "selectedJoinStepErrorRms": float(
                    math.sqrt(np.mean(step_error**2))
                ),
                "excerpt": str(excerpt_path.relative_to(args.pack_root)),
                "seamDetailFigure": str(detail_path.relative_to(args.pack_root)),
                "fullSpectrogram": str(full_path.relative_to(args.pack_root)),
            }
        )
        del candidate, residual

    combined_path = figures_root / "00-all-strategies-seam-comparison.png"
    plot_combined_seam_spectrograms(
        master[boundary - detail_half : boundary + detail_half],
        comparison_regions,
        sample_rate,
        boundary / sample_rate,
        combined_path,
    )

    report = {
        "schema": "wavey.encodec.seam-listening-analysis",
        "schemaVersion": 1,
        "master": str(args.master.resolve()),
        "masterSha256": sha256_file(args.master),
        "selectedJoin": {
            "selection": "largest hard-cut sample-step error versus the master",
            "boundarySample": boundary,
            "boundarySeconds": boundary / sample_rate,
            "excerptStartSample": excerpt_start,
            "excerptEndSample": excerpt_end,
        },
        "joinSelectionRows": join_rows,
        "combinedSeamSpectrogram": str(combined_path.relative_to(args.pack_root)),
        "strategies": strategy_reports,
    }
    report_path = args.pack_root / "analysis.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
