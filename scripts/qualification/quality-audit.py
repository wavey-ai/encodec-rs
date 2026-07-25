#!/usr/bin/env python3
"""Run qualification-only PCM quality and Meta-versus-fork checks.

This script reads the prepared canonical PCM and the existing fork outputs.
It does not create or parse a Profile 1 object.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import torchaudio


SAMPLE_RATE = 48_000
CHANNELS = 2
GEOMETRIES = {
    "fixed-1333": {"stride": 64_000, "model_samples": 64_960},
    "fixed-1800": {"stride": 86_400, "model_samples": 87_360},
}


def acquire_qualification_run_lock():
    lock_path = Path(__file__).resolve().parents[2] / "target/qualification/.run.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    qualification_lock = lock_path.open("a+")
    try:
        fcntl.flock(
            qualification_lock.fileno(),
            fcntl.LOCK_EX | fcntl.LOCK_NB,
        )
    except BlockingIOError as error:
        qualification_lock.close()
        raise RuntimeError(
            "another Encodec qualification process is active; quality-audit did not start"
        ) from error
    return qualification_lock


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    metrics = subparsers.add_parser("fork-metrics")
    metrics.add_argument("--run-root", type=Path, required=True)
    metrics.add_argument("--corpus-ids", default="")
    metrics.add_argument("--candidates", default="")
    metrics.add_argument("--rates", default="")

    reference = subparsers.add_parser("original-fork")
    reference.add_argument("--run-root", type=Path, required=True)
    reference.add_argument("--upstream-repo", type=Path, required=True)
    reference.add_argument("--checkpoint-dir", type=Path, required=True)
    reference.add_argument("--corpus-ids", default="confirmation-westside-v1")
    reference.add_argument("--candidates", default="fixed-1333")
    reference.add_argument("--rates", default="6,12")
    return parser.parse_args()


def read_json(path: Path):
    return json.loads(path.read_text())


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def append_jsonl(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(value, sort_keys=True) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def csv_values(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def read_planar(path: Path) -> torch.Tensor:
    values = np.fromfile(path, dtype="<f4")
    if values.size % CHANNELS:
        raise ValueError(f"{path} does not contain two-channel planar f32 data")
    return torch.from_numpy(values.reshape(CHANNELS, values.size // CHANNELS).copy())


def write_planar(path: Path, value: torch.Tensor) -> None:
    array = value.detach().cpu().to(torch.float32).contiguous().numpy()
    path.parent.mkdir(parents=True, exist_ok=True)
    array.astype("<f4", copy=False).tofile(path)


def db(value: float, floor: float = 1.0e-12) -> float:
    return 20.0 * math.log10(max(abs(value), floor))


def correlation(left: torch.Tensor, right: torch.Tensor) -> float | None:
    left = left - left.mean()
    right = right - right.mean()
    denominator = torch.sqrt(torch.sum(left.square()) * torch.sum(right.square()))
    if float(denominator) == 0.0:
        return None
    return float(torch.sum(left * right) / denominator)


def loudness(value: torch.Tensor) -> float | None:
    try:
        return float(torchaudio.functional.loudness(value, SAMPLE_RATE))
    except Exception:
        return None


def true_peak_dbtp(value: torch.Tensor) -> float | None:
    try:
        oversampled = torchaudio.functional.resample(
            value,
            SAMPLE_RATE,
            SAMPLE_RATE * 4,
            lowpass_filter_width=64,
            rolloff=0.99,
            resampling_method="sinc_interp_hann",
        )
        return db(float(oversampled.abs().amax()))
    except Exception:
        return None


def stft_metrics(reference: torch.Tensor, candidate: torch.Tensor) -> dict:
    """Return bounded, repeatable multi-resolution spectral metrics.

    The first 60 seconds are used for this diagnostic report. The full PCM
    metrics still cover the complete object.
    """

    count = min(reference.shape[-1], candidate.shape[-1], SAMPLE_RATE * 60)
    reference = reference[:, :count]
    candidate = candidate[:, :count]
    values = []
    spectral_convergence = []
    for n_fft in (256, 512, 1024, 2048):
        hop = n_fft // 4
        window = torch.hann_window(n_fft, dtype=reference.dtype)
        reference_spec = torch.stft(
            reference,
            n_fft=n_fft,
            hop_length=hop,
            win_length=n_fft,
            window=window,
            return_complex=True,
            center=False,
        ).abs().clamp_min(1.0e-8)
        candidate_spec = torch.stft(
            candidate,
            n_fft=n_fft,
            hop_length=hop,
            win_length=n_fft,
            window=window,
            return_complex=True,
            center=False,
        ).abs().clamp_min(1.0e-8)
        log_difference = 20.0 * torch.log10(candidate_spec) - 20.0 * torch.log10(reference_spec)
        values.append(float(log_difference.abs().mean()))
        spectral_convergence.append(
            float(torch.linalg.vector_norm(candidate_spec - reference_spec) / torch.linalg.vector_norm(reference_spec))
        )
    return {
        "window_seconds": count / SAMPLE_RATE,
        "log_spectral_distance_db": float(sum(values) / len(values)),
        "multi_resolution_stft_spectral_convergence": float(sum(spectral_convergence) / len(spectral_convergence)),
    }


def seam_metrics(candidate: torch.Tensor, stride: int) -> dict:
    boundaries = list(range(stride, candidate.shape[-1], stride))
    if not boundaries:
        return {
            "seam_count": 0,
            "max_seam_residual": None,
            "rms_seam_residual": None,
        }
    residuals = []
    for boundary in boundaries:
        if boundary + 1 >= candidate.shape[-1]:
            continue
        residuals.append(candidate[:, boundary] - 0.5 * (candidate[:, boundary - 1] + candidate[:, boundary + 1]))
    if not residuals:
        return {"seam_count": 0, "max_seam_residual": None, "rms_seam_residual": None}
    residual = torch.stack(residuals)
    return {
        "seam_count": len(residuals),
        "max_seam_residual": float(residual.abs().max()),
        "rms_seam_residual": float(torch.sqrt(residual.square().mean())),
    }


def compare(
    reference: torch.Tensor,
    candidate: torch.Tensor,
    stride: int | None = None,
    expensive_metrics: bool = True,
) -> dict:
    reference_count = int(reference.shape[-1])
    candidate_count = int(candidate.shape[-1])
    count = min(reference_count, candidate_count)
    reference = reference[:, :count]
    candidate = candidate[:, :count]
    error = candidate - reference
    signal_power = float(reference.square().mean())
    error_power = float(error.square().mean())
    sisdr_values = []
    for channel in range(CHANNELS):
        ref_channel = reference[channel]
        candidate_channel = candidate[channel]
        projection_denominator = float(ref_channel.square().sum())
        if projection_denominator == 0.0:
            continue
        projection = float((candidate_channel * ref_channel).sum()) / projection_denominator
        target = projection * ref_channel
        residual = candidate_channel - target
        residual_power = float(residual.square().mean())
        target_power = float(target.square().mean())
        sisdr_values.append(10.0 * math.log10(max(target_power, 1.0e-12) / max(residual_power, 1.0e-12)))

    reference_peak = float(reference.abs().amax()) if count else 0.0
    candidate_peak = float(candidate.abs().amax()) if count else 0.0
    reference_rms = math.sqrt(max(signal_power, 0.0))
    candidate_rms = float(torch.sqrt(candidate.square().mean())) if count else 0.0
    result = {
        "reference_samples_per_channel": reference_count,
        "candidate_samples_per_channel": candidate_count,
        "compared_samples_per_channel": count,
        "length_match": reference_count == candidate_count,
        "reference_rms_f32": reference_rms,
        "candidate_rms_f32": candidate_rms,
        "error_rms_f32": math.sqrt(max(error_power, 0.0)),
        "error_rms_dbfs": db(math.sqrt(max(error_power, 0.0))),
        "reference_rms_dbfs": db(reference_rms),
        "candidate_rms_dbfs": db(candidate_rms),
        "reference_peak_dbfs": db(reference_peak),
        "candidate_peak_dbfs": db(candidate_peak),
        "snr_db": float("inf") if error_power == 0.0 else 10.0 * math.log10(max(signal_power, 1.0e-12) / error_power),
        "scale_invariant_sdr_db": float(sum(sisdr_values) / len(sisdr_values)) if sisdr_values else None,
        "max_abs_error_f32": float(error.abs().amax()) if count else None,
        "reference_integrated_loudness_lu": loudness(reference) if expensive_metrics else None,
        "candidate_integrated_loudness_lu": loudness(candidate) if expensive_metrics else None,
        "integrated_loudness_difference_lu": None,
        "reference_true_peak_dbtp": true_peak_dbtp(reference) if expensive_metrics else None,
        "candidate_true_peak_dbtp": true_peak_dbtp(candidate) if expensive_metrics else None,
        "true_peak_difference_db": None,
        "channel_correlation_reference": correlation(reference[0], reference[1]),
        "channel_correlation_candidate": correlation(candidate[0], candidate[1]),
        "reference_inter_channel_balance_db": db(float(torch.sqrt(reference[0].square().mean()) / torch.sqrt(reference[1].square().mean()))),
        "candidate_inter_channel_balance_db": db(float(torch.sqrt(candidate[0].square().mean()) / torch.sqrt(candidate[1].square().mean()))),
        "reference_clipped_sample_count": int((reference.abs() >= 1.0).sum()),
        "candidate_clipped_sample_count": int((candidate.abs() >= 1.0).sum()),
        "reference_non_finite_count": int((~torch.isfinite(reference)).sum()),
        "candidate_non_finite_count": int((~torch.isfinite(candidate)).sum()),
        "stft": stft_metrics(reference, candidate) if count and expensive_metrics else None,
    }
    if result["reference_integrated_loudness_lu"] is not None and result["candidate_integrated_loudness_lu"] is not None:
        result["integrated_loudness_difference_lu"] = result["candidate_integrated_loudness_lu"] - result["reference_integrated_loudness_lu"]
    if result["reference_true_peak_dbtp"] is not None and result["candidate_true_peak_dbtp"] is not None:
        result["true_peak_difference_db"] = result["candidate_true_peak_dbtp"] - result["reference_true_peak_dbtp"]
    if stride is not None:
        result.update(seam_metrics(candidate, stride))
    return result


def git_revision(repo: Path) -> str:
    result = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], check=True, capture_output=True, text=True)
    return result.stdout.strip()


def corpus_entries(run_root: Path) -> dict[str, dict]:
    manifest = read_json(run_root / "corpus-manifest.json")
    return {entry["id"]: entry for entry in manifest["entries"]}


def fork_output_path(run_root: Path, corpus_id: str, candidate: str, rate: int) -> Path:
    return run_root / "decodes" / corpus_id / f"{candidate}-{rate}kbps.f32le"


def fork_metrics(run_root: Path, corpus_ids: list[str] | None = None, candidates: list[str] | None = None, rates: list[int] | None = None) -> None:
    entries = corpus_entries(run_root)
    result_path = run_root / "metrics" / "results.jsonl"
    rows = read_jsonl(result_path)
    existing = {row.get("case_id") for row in rows}
    quality_rows = []
    for row in rows:
        if row.get("phase") != "geometry_selection" or row.get("status") != "pass":
            continue
        candidate = row["candidate"]
        rate = int(row["bandwidth_kbps"])
        corpus_id = row["corpus_id"]
        if corpus_ids and corpus_id not in corpus_ids:
            continue
        if candidates and candidate not in candidates:
            continue
        if rates and rate not in rates:
            continue
        case_id = f"quality-fork-vs-canonical-{candidate}-{rate}-{corpus_id}"
        if case_id in existing:
            continue
        reference = read_planar(run_root / entries[corpus_id]["canonical_pcm_path"])
        output = fork_output_path(run_root, corpus_id, candidate, rate)
        if not output.exists():
            quality = {"error": f"missing fork output {output}"}
            status = "blocked"
        else:
            decoded = read_planar(output)
            quality = compare(reference, decoded, GEOMETRIES[candidate]["stride"])
            status = "pass" if quality["candidate_non_finite_count"] == 0 else "fail"
        record = {
            "schema": "wavey.encodec.qualification",
            "schema_version": 1,
            "run_id": run_root.name,
            "case_id": case_id,
            "attempt": 1,
            "status": status,
            "phase": "quality",
            "candidate": candidate,
            "bandwidth_kbps": rate,
            "corpus_id": corpus_id,
            "source_sha256": row.get("source_sha256"),
            "canonical_pcm_sha256": row.get("canonical_pcm_sha256"),
            "model_input_sha256": None,
            "codes_sha256": None,
            "scale_bits_sha256": None,
            "entropy_sha256": None,
            "container_sha256": row.get("container_sha256"),
            "container_bytes": row.get("container_bytes"),
            "environment_id": row.get("environment_id"),
            "artifact_lock_sha256": None,
            "elapsed_ms": None,
            "metrics": quality,
            "damage": None,
            "error": quality.get("error"),
            "stop_reason": "objective PCM diagnostic; it does not pass the owner listening gate",
        }
        append_jsonl(result_path, record)
        write_json(run_root / "metrics" / "quality" / f"{case_id}.json", record)
        quality_rows.append(record)

    all_quality_rows = [row for row in read_jsonl(result_path) if row.get("phase") == "quality"]
    summary = {
        "schema": "wavey.encodec.qualification.quality-summary",
        "run_id": run_root.name,
        "rows": len(all_quality_rows),
        "pass": sum(row["status"] == "pass" for row in all_quality_rows),
        "blocked": sum(row["status"] == "blocked" for row in all_quality_rows),
        "fail": sum(row["status"] == "fail" for row in all_quality_rows),
        "metrics": [
            "snr_db",
            "scale_invariant_sdr_db",
            "log_spectral_distance_db",
            "integrated_loudness_difference_lu",
            "true_peak_difference_db",
            "channel_correlation_candidate",
            "inter_channel_balance_db",
            "max_seam_residual",
            "rms_seam_residual",
            "clipped_sample_count",
            "non_finite_count",
        ],
    }
    write_json(run_root / "reports" / "quality-metrics.json", summary)
    report = [
        "# Objective Quality Metrics",
        "",
        "Status: provisional diagnostic evidence.",
        "",
        f"This run contains {len(all_quality_rows)} fork-versus-canonical rows. It measures the complete decoded PCM without alignment search.",
        "",
        "The rows include linear error, SNR, scale-invariant SDR, dBFS, loudness, true-peak, channel, spectral, seam, clipping, and finite-value checks.",
        "",
        "ViSQOL and owner-approved listening results remain unavailable. These rows do not pass G12 or G19.",
    ]
    (run_root / "reports" / "quality-metrics.md").write_text("\n".join(report) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


def original_fork(run_root: Path, upstream_repo: Path, checkpoint_dir: Path, corpus_ids: list[str], candidates: list[str], rates: list[int]) -> None:
    entries = corpus_entries(run_root)
    result_path = run_root / "metrics" / "results.jsonl"
    existing_rows = read_jsonl(result_path)
    existing = {row.get("case_id") for row in existing_rows}
    fork_quality_by_case = {
        row.get("case_id"): row.get("metrics")
        for row in existing_rows
        if row.get("phase") == "quality"
        and str(row.get("case_id", "")).startswith("quality-fork-vs-canonical-")
    }
    sys.path.insert(0, str(upstream_repo))
    from encodec import EncodecModel, compress  # type: ignore

    original_commit = git_revision(upstream_repo)
    fork_commit = git_revision(Path(__file__).resolve().parents[2])
    model = EncodecModel.encodec_model_48khz(repository=checkpoint_dir).to("cpu").eval()
    records = []
    for corpus_id in corpus_ids:
        if corpus_id not in entries:
            raise ValueError(f"unknown corpus id {corpus_id}")
        source = read_planar(run_root / entries[corpus_id]["canonical_pcm_path"])
        for candidate in candidates:
            if candidate not in GEOMETRIES:
                raise ValueError(f"unsupported direct comparison candidate {candidate}")
            geometry = GEOMETRIES[candidate]
            model.segment = geometry["model_samples"] / SAMPLE_RATE
            model.overlap = 1.0 - geometry["stride"] / geometry["model_samples"]
            for rate in rates:
                case_id = f"quality-original-vs-fork-{candidate}-{rate}-{corpus_id}"
                if case_id in existing:
                    continue
                model.set_target_bandwidth(float(rate))
                with torch.inference_mode():
                    original_bytes = compress(model, source, use_lm=True)
                    frames = model.encode(source.unsqueeze(0))
                    original = model.decode(frames).squeeze(0).cpu().to(torch.float32)
                original_path = run_root / "metrics" / "original-fork" / corpus_id / f"{candidate}-{rate}kbps-original.f32le"
                write_planar(original_path, original)
                original_stream_path = run_root / "metrics" / "original-fork" / corpus_id / f"{candidate}-{rate}kbps-original.ecdc"
                original_stream_path.parent.mkdir(parents=True, exist_ok=True)
                original_stream_path.write_bytes(original_bytes)
                fork_path = fork_output_path(run_root, corpus_id, candidate, rate)
                if not fork_path.exists():
                    raise FileNotFoundError(fork_path)
                fork = read_planar(fork_path)
                fork_container_path = run_root / "encodes" / corpus_id / f"{candidate}-{rate}kbps.ecdc"
                if not fork_container_path.exists():
                    raise FileNotFoundError(fork_container_path)
                fork_container_bytes = fork_container_path.read_bytes()
                source_vs_original = compare(
                    source,
                    original,
                    geometry["stride"],
                    expensive_metrics=False,
                )
                source_vs_fork = fork_quality_by_case.get(
                    f"quality-fork-vs-canonical-{candidate}-{rate}-{corpus_id}"
                )
                if source_vs_fork is None:
                    source_vs_fork = compare(source, fork, geometry["stride"])
                original_vs_fork = compare(
                    original,
                    fork,
                    geometry["stride"],
                    expensive_metrics=False,
                )
                record = {
                    "schema": "wavey.encodec.qualification",
                    "schema_version": 1,
                    "run_id": run_root.name,
                    "case_id": case_id,
                    "attempt": 1,
                    "status": "pass" if original_vs_fork["candidate_non_finite_count"] == 0 else "fail",
                    "phase": "quality",
                    "candidate": candidate,
                    "bandwidth_kbps": rate,
                    "corpus_id": corpus_id,
                    "source_sha256": entries[corpus_id]["source_sha256"],
                    "canonical_pcm_sha256": entries[corpus_id]["canonical_pcm_sha256"],
                    "model_input_sha256": None,
                    "codes_sha256": None,
                    "scale_bits_sha256": None,
                    "entropy_sha256": None,
                    "container_sha256": None,
                    "container_bytes": len(fork_container_bytes),
                    "environment_id": "mac-arm64-original-meta-vs-fork-cpu",
                    "artifact_lock_sha256": None,
                    "elapsed_ms": None,
                    "metrics": {
                        "source_vs_original": source_vs_original,
                        "source_vs_fork": source_vs_fork,
                        "original_vs_fork": original_vs_fork,
                        "size": {
                            "original_meta_ecdc_bytes": len(original_bytes),
                            "fork_current_acv2_ecdc_bytes": len(fork_container_bytes),
                            "fork_minus_original_bytes": len(fork_container_bytes) - len(original_bytes),
                            "original_meta_ecdc_sha256": hashlib.sha256(original_bytes).hexdigest(),
                            "fork_current_acv2_ecdc_sha256": hashlib.sha256(fork_container_bytes).hexdigest(),
                        },
                        "original_runtime": "Meta pinned commit",
                        "original_commit": original_commit,
                        "fork_commit": fork_commit,
                        "fork_output_includes_current_acv2_entropy": True,
                        "raw_code_tensor_parity_measured": False,
                    },
                    "damage": None,
                    "error": None,
                    "stop_reason": "direct neural PCM comparison; raw code parity still requires matching Meta and fork evidence files",
                }
                append_jsonl(result_path, record)
                write_json(run_root / "metrics" / "original-fork" / f"{case_id}.json", record)
                records.append(record)
                print(json.dumps({"case_id": case_id, "status": record["status"], "snr_db_original_vs_fork": original_vs_fork["snr_db"]}, sort_keys=True), flush=True)
    direct_rows = [
        row
        for row in read_jsonl(result_path)
        if row.get("phase") == "quality"
        and str(row.get("case_id", "")).startswith("quality-original-vs-fork-")
        and row.get("metrics", {}).get("size")
    ]
    report = [
        "# Original Meta EnCodec versus encodec-rs",
        "",
        "Status: direct neural PCM and encoded-size comparison.",
        "",
        f"Meta reference commit: `{original_commit}`.",
        f"encodec-rs commit: `{fork_commit}`.",
        "",
        f"This run contains {len(direct_rows)} direct comparisons. It uses the same pinned neural checkpoint and candidate window geometry.",
        "The fork output uses its current ONNX and acv=2 path, so this is not Profile 1 evidence.",
        "",
        "The size comparison uses the unmodified Meta ECDC stream with LM entropy and the fork current acv2 ECDC stream. The fork now exports qualification-only raw code and scale evidence, but direct code and entropy parity remains a separate blocked gate.",
        "The report retains decoded-length mismatches. The Meta fixed-window decoder can return a shorter final output, while the fork path targets the source length.",
        "Each JSON row also retains SI-SDR, error RMS dBFS, peak, channel, clipping, non-finite, and seam metrics.",
        "",
        "| Geometry | Rate | Rows | Meta length mismatches | Fork length mismatches | Source-to-original SNR (dB) | Source-to-fork SNR (dB) | Original-to-fork SNR (dB) | Meta ECDC (bytes) | Fork acv2 ECDC (bytes) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for candidate in candidates:
        for rate in rates:
            matching = [
                row
                for row in direct_rows
                if row.get("candidate") == candidate and int(row.get("bandwidth_kbps", 0)) == rate
            ]
            if not matching:
                continue

            def mean_metric(name: str) -> float:
                values = [
                    row["metrics"][name]["snr_db"]
                    for row in matching
                    if math.isfinite(row["metrics"][name]["snr_db"])
                ]
                return sum(values) / len(values) if values else float("nan")

            original_sizes = [row["metrics"]["size"]["original_meta_ecdc_bytes"] for row in matching]
            fork_sizes = [row["metrics"]["size"]["fork_current_acv2_ecdc_bytes"] for row in matching]
            original_length_mismatches = sum(
                not row["metrics"]["source_vs_original"]["length_match"] for row in matching
            )
            fork_length_mismatches = sum(
                not row["metrics"]["source_vs_fork"]["length_match"] for row in matching
            )
            report.append(
                f"| {candidate} | {rate} | {len(matching)} | "
                f"{original_length_mismatches} | {fork_length_mismatches} | "
                f"{mean_metric('source_vs_original'):.3f} | "
                f"{mean_metric('source_vs_fork'):.3f} | "
                f"{mean_metric('original_vs_fork'):.3f} | "
                f"{sum(original_sizes) / len(original_sizes):.1f} | "
                f"{sum(fork_sizes) / len(fork_sizes):.1f} |"
            )
    (run_root / "reports" / "original-fork.md").write_text("\n".join(report) + "\n")
    print(json.dumps({"rows": len(records), "report_rows": len(direct_rows), "original_commit": original_commit, "fork_commit": fork_commit}, indent=2, sort_keys=True))


def main() -> None:
    args = parse_args()
    qualification_lock = acquire_qualification_run_lock()
    try:
        torch.set_num_threads(1)
        if args.command == "fork-metrics":
            fork_metrics(
                args.run_root.resolve(),
                csv_values(args.corpus_ids),
                csv_values(args.candidates),
                [int(value) for value in csv_values(args.rates)],
            )
        else:
            original_fork(
                args.run_root.resolve(),
                args.upstream_repo.resolve(),
                args.checkpoint_dir.resolve(),
                csv_values(args.corpus_ids),
                csv_values(args.candidates),
                [int(value) for value in csv_values(args.rates)],
            )
    finally:
        qualification_lock.close()


if __name__ == "__main__":
    main()
