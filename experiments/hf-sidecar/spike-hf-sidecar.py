#!/usr/bin/env python3
"""Test a compact high-frequency sidecar on an EnCodec round trip.

The sidecar stores a coarse source-to-decode energy correction and a spectral
flatness class. The decoder uses only the EnCodec PCM and the sidecar. It makes
deterministic harmonic or noise excitation to fill missing high-band energy.

This is an experiment. It is not a stable bitstream format.
"""

from __future__ import annotations

import argparse
import json
import math
import struct
import subprocess
import zlib
from pathlib import Path

import numpy as np


MAGIC = b"ECHFSPK1"
VERSION = 1
BITS_PER_CELL = 6
GAIN_MIN_DB = -12
GAIN_MAX_DB = 15
HEADER = struct.Struct("<8sBIHHBBBBHIbbI")
GRID_MAGIC = b"ECHFGRD1"
GRID_BITS = 3
GRID_GAIN_LIMIT_DB = 18
GRID_HEADER = struct.Struct("<8sBIHHBBBBHIHHbbII")


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=repo_root / "testdata/westside_4s_48khz_stereo.wav",
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=(
            repo_root
            / "target/performance/entropy-optimization/06-roundtrip-audit"
            / "westside_4s.decoded.f32.wav"
        ),
    )
    parser.add_argument(
        "--ecdc",
        type=Path,
        default=(
            repo_root
            / "target/performance/entropy-optimization/06-roundtrip-audit"
            / "westside_4s.lm.ecdc"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root / "target/experiments/hf-sidecar-westside",
    )
    parser.add_argument("--sample-rate", type=int, default=48_000)
    parser.add_argument("--channels", type=int, default=2)
    parser.add_argument("--fft-size", type=int, default=2_048)
    parser.add_argument("--hop-size", type=int, default=512)
    parser.add_argument("--group-frames", type=int, default=6)
    parser.add_argument("--grid-low-hz", type=int, default=4_000)
    parser.add_argument("--grid-high-hz", type=int, default=20_000)
    parser.add_argument("--grid-bands", type=int, default=20)
    parser.add_argument("--grid-group-frames", type=int, default=3)
    return parser.parse_args()


def read_audio(path: Path, sample_rate: int, channels: int) -> np.ndarray:
    command = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        str(path),
        "-f",
        "f32le",
        "-acodec",
        "pcm_f32le",
        "-ar",
        str(sample_rate),
        "-ac",
        str(channels),
        "pipe:1",
    ]
    result = subprocess.run(command, check=True, capture_output=True)
    audio = np.frombuffer(result.stdout, dtype="<f4").astype(np.float64)
    if audio.size % channels:
        raise ValueError(f"{path} has an incomplete audio frame")
    return audio.reshape(-1, channels)


def write_audio(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-y",
        "-v",
        "error",
        "-f",
        "f32le",
        "-ar",
        str(sample_rate),
        "-ac",
        str(audio.shape[1]),
        "-i",
        "pipe:0",
        "-c:a",
        "pcm_f32le",
        str(path),
    ]
    subprocess.run(
        command,
        input=np.asarray(audio, dtype="<f4").tobytes(),
        check=True,
    )


def lr_to_ms(audio: np.ndarray) -> np.ndarray:
    scale = 1.0 / math.sqrt(2.0)
    return np.column_stack(
        ((audio[:, 0] + audio[:, 1]) * scale, (audio[:, 0] - audio[:, 1]) * scale)
    )


def ms_to_lr(audio: np.ndarray) -> np.ndarray:
    scale = 1.0 / math.sqrt(2.0)
    return np.column_stack(
        ((audio[:, 0] + audio[:, 1]) * scale, (audio[:, 0] - audio[:, 1]) * scale)
    )


def stft(audio: np.ndarray, fft_size: int, hop_size: int) -> tuple[np.ndarray, int]:
    half = fft_size // 2
    padded = np.pad(audio, ((half, half), (0, 0)), mode="reflect")
    extra = (-(padded.shape[0] - fft_size)) % hop_size
    if extra:
        padded = np.pad(padded, ((0, extra), (0, 0)))
    frame_count = 1 + (padded.shape[0] - fft_size) // hop_size
    window = np.hanning(fft_size)
    result = np.empty(
        (frame_count, fft_size // 2 + 1, audio.shape[1]), dtype=np.complex128
    )
    for frame in range(frame_count):
        start = frame * hop_size
        block = padded[start : start + fft_size] * window[:, None]
        result[frame] = np.fft.rfft(block, axis=0)
    return result, half


def istft(
    spectrum: np.ndarray,
    fft_size: int,
    hop_size: int,
    crop: int,
    output_samples: int,
) -> np.ndarray:
    window = np.hanning(fft_size)
    total_samples = (spectrum.shape[0] - 1) * hop_size + fft_size
    audio = np.zeros((total_samples, spectrum.shape[2]), dtype=np.float64)
    weight = np.zeros(total_samples, dtype=np.float64)
    for frame in range(spectrum.shape[0]):
        start = frame * hop_size
        block = np.fft.irfft(spectrum[frame], n=fft_size, axis=0)
        audio[start : start + fft_size] += block * window[:, None]
        weight[start : start + fft_size] += window * window
    valid = weight > 1e-12
    audio[valid] /= weight[valid, None]
    return audio[crop : crop + output_samples]


def band_edges(sample_rate: int) -> np.ndarray:
    nyquist = sample_rate // 2
    edges = np.array([8_000, 10_500, 13_000, 16_000, 19_500, nyquist])
    if nyquist != 24_000:
        raise ValueError("this spike requires a 48 kHz sample rate")
    return edges


def quantize_gain(gain_db: float) -> int:
    clipped = float(np.clip(gain_db, GAIN_MIN_DB, GAIN_MAX_DB))
    unit = (clipped - GAIN_MIN_DB) / (GAIN_MAX_DB - GAIN_MIN_DB)
    return int(np.clip(np.rint(unit * 15.0), 0, 15))


def dequantize_gain(code: int) -> float:
    return GAIN_MIN_DB + (GAIN_MAX_DB - GAIN_MIN_DB) * code / 15.0


def encode_cells(
    source: np.ndarray,
    baseline: np.ndarray,
    frequencies: np.ndarray,
    edges: np.ndarray,
    group_frames: int,
) -> tuple[np.ndarray, np.ndarray]:
    group_count = math.ceil(source.shape[0] / group_frames)
    gains = np.empty((group_count, len(edges) - 1, source.shape[2]), dtype=np.uint8)
    tones = np.empty_like(gains)
    epsilon = 1e-12

    for group in range(group_count):
        frame_slice = slice(group * group_frames, min((group + 1) * group_frames, source.shape[0]))
        for band in range(len(edges) - 1):
            mask = (frequencies >= edges[band]) & (frequencies < edges[band + 1])
            if band == len(edges) - 2:
                mask = (frequencies >= edges[band]) & (frequencies <= edges[band + 1])
            for channel in range(source.shape[2]):
                source_magnitude = np.abs(source[frame_slice, mask, channel])
                baseline_magnitude = np.abs(baseline[frame_slice, mask, channel])
                source_rms = math.sqrt(float(np.mean(source_magnitude**2)) + epsilon)
                baseline_rms = math.sqrt(float(np.mean(baseline_magnitude**2)) + epsilon)
                gain_db = 20.0 * math.log10(source_rms / baseline_rms)
                gains[group, band, channel] = quantize_gain(gain_db)

                arithmetic = float(np.mean(source_magnitude)) + epsilon
                geometric = math.exp(float(np.mean(np.log(source_magnitude + epsilon))))
                flatness = float(np.clip(geometric / arithmetic, 0.0, 1.0))
                tones[group, band, channel] = int(np.clip(np.rint(flatness * 3.0), 0, 3))
    return gains, tones


def pack_values(values: list[int], bits: int) -> bytes:
    output = bytearray()
    accumulator = 0
    used = 0
    for value in values:
        accumulator |= value << used
        used += bits
        while used >= 8:
            output.append(accumulator & 0xFF)
            accumulator >>= 8
            used -= 8
    if used:
        output.append(accumulator & 0xFF)
    return bytes(output)


def unpack_values(payload: bytes, count: int, bits: int) -> list[int]:
    values: list[int] = []
    accumulator = 0
    used = 0
    offset = 0
    mask = (1 << bits) - 1
    while len(values) < count:
        while used < bits:
            if offset >= len(payload):
                raise ValueError("sidecar payload is truncated")
            accumulator |= payload[offset] << used
            offset += 1
            used += 8
        values.append(accumulator & mask)
        accumulator >>= bits
        used -= bits
    return values


def serialize_sidecar(
    gains: np.ndarray,
    tones: np.ndarray,
    edges: np.ndarray,
    sample_rate: int,
    fft_size: int,
    hop_size: int,
    group_frames: int,
    stft_frames: int,
    audio_samples: int,
) -> bytes:
    cells: list[int] = []
    for gain, tone in zip(gains.flat, tones.flat, strict=True):
        cells.append(int(gain) | (int(tone) << 4))
    payload = pack_values(cells, BITS_PER_CELL)
    edge_bytes = struct.pack(f"<{len(edges)}H", *[int(value) for value in edges])
    checksum = zlib.crc32(edge_bytes + payload)
    header = HEADER.pack(
        MAGIC,
        VERSION,
        sample_rate,
        fft_size,
        hop_size,
        group_frames,
        gains.shape[2],
        gains.shape[1],
        BITS_PER_CELL,
        stft_frames,
        audio_samples,
        GAIN_MIN_DB,
        GAIN_MAX_DB,
        checksum,
    )
    return header + edge_bytes + payload


def deserialize_sidecar(data: bytes) -> dict[str, object]:
    if len(data) < HEADER.size:
        raise ValueError("sidecar header is truncated")
    fields = HEADER.unpack_from(data)
    (
        magic,
        version,
        sample_rate,
        fft_size,
        hop_size,
        group_frames,
        channels,
        band_count,
        bits_per_cell,
        stft_frames,
        audio_samples,
        gain_min_db,
        gain_max_db,
        checksum,
    ) = fields
    if magic != MAGIC or version != VERSION:
        raise ValueError("sidecar format is not supported")
    if gain_min_db != GAIN_MIN_DB or gain_max_db != GAIN_MAX_DB:
        raise ValueError("sidecar gain range is not supported")
    edge_size = (band_count + 1) * 2
    edge_start = HEADER.size
    edge_end = edge_start + edge_size
    edge_bytes = data[edge_start:edge_end]
    payload = data[edge_end:]
    if zlib.crc32(edge_bytes + payload) != checksum:
        raise ValueError("sidecar checksum does not match")
    edges = np.array(struct.unpack(f"<{band_count + 1}H", edge_bytes))
    group_count = math.ceil(stft_frames / group_frames)
    cell_count = group_count * band_count * channels
    cells = unpack_values(payload, cell_count, bits_per_cell)
    packed = np.array(cells, dtype=np.uint8).reshape(group_count, band_count, channels)
    return {
        "sample_rate": sample_rate,
        "fft_size": fft_size,
        "hop_size": hop_size,
        "group_frames": group_frames,
        "channels": channels,
        "stft_frames": stft_frames,
        "audio_samples": audio_samples,
        "edges": edges,
        "gains": packed & 0x0F,
        "tones": packed >> 4,
    }


def encode_mid_grid(
    source: np.ndarray,
    baseline: np.ndarray,
    frequencies: np.ndarray,
    low_hz: int,
    high_hz: int,
    band_count: int,
    group_frames: int,
) -> np.ndarray:
    """Encode log-magnitude corrections for the stereo mid channel."""
    edges = np.linspace(low_hz, high_hz, band_count + 1)
    group_count = math.ceil(source.shape[0] / group_frames)
    codes = np.empty((group_count, band_count), dtype=np.uint8)
    levels = (1 << GRID_BITS) - 1
    floor = 1e-5
    for group in range(group_count):
        frame_slice = slice(
            group * group_frames,
            min((group + 1) * group_frames, source.shape[0]),
        )
        for band in range(band_count):
            upper = edges[band + 1] + (1.0 if band == band_count - 1 else 0.0)
            mask = (frequencies >= edges[band]) & (frequencies < upper)
            source_db = 20.0 * np.log10(
                np.maximum(np.abs(source[frame_slice, mask, 0]), floor)
            )
            baseline_db = 20.0 * np.log10(
                np.maximum(np.abs(baseline[frame_slice, mask, 0]), floor)
            )
            correction_db = float(np.mean(source_db - baseline_db))
            unit = np.clip(
                (correction_db + GRID_GAIN_LIMIT_DB) / (2 * GRID_GAIN_LIMIT_DB),
                0.0,
                1.0,
            )
            codes[group, band] = int(np.rint(unit * levels))
    return codes


def serialize_mid_grid(
    codes: np.ndarray,
    sample_rate: int,
    fft_size: int,
    hop_size: int,
    group_frames: int,
    stft_frames: int,
    audio_samples: int,
    low_hz: int,
    high_hz: int,
) -> bytes:
    packed = pack_values([int(value) for value in codes.flat], GRID_BITS)
    compressed = zlib.compress(packed, level=9)
    header = GRID_HEADER.pack(
        GRID_MAGIC,
        VERSION,
        sample_rate,
        fft_size,
        hop_size,
        group_frames,
        0,
        codes.shape[1],
        GRID_BITS,
        stft_frames,
        audio_samples,
        low_hz,
        high_hz,
        -GRID_GAIN_LIMIT_DB,
        GRID_GAIN_LIMIT_DB,
        len(packed),
        zlib.crc32(packed),
    )
    return header + compressed


def deserialize_mid_grid(data: bytes) -> dict[str, object]:
    if len(data) < GRID_HEADER.size:
        raise ValueError("mid-grid sidecar header is truncated")
    (
        magic,
        version,
        sample_rate,
        fft_size,
        hop_size,
        group_frames,
        channel,
        band_count,
        bits,
        stft_frames,
        audio_samples,
        low_hz,
        high_hz,
        gain_min_db,
        gain_max_db,
        raw_size,
        checksum,
    ) = GRID_HEADER.unpack_from(data)
    if magic != GRID_MAGIC or version != VERSION:
        raise ValueError("mid-grid sidecar format is not supported")
    if channel != 0 or bits != GRID_BITS:
        raise ValueError("mid-grid sidecar profile is not supported")
    if gain_min_db != -GRID_GAIN_LIMIT_DB or gain_max_db != GRID_GAIN_LIMIT_DB:
        raise ValueError("mid-grid gain range is not supported")
    packed = zlib.decompress(data[GRID_HEADER.size :])
    if len(packed) != raw_size or zlib.crc32(packed) != checksum:
        raise ValueError("mid-grid sidecar checksum does not match")
    group_count = math.ceil(stft_frames / group_frames)
    values = unpack_values(packed, group_count * band_count, bits)
    codes = np.array(values, dtype=np.uint8).reshape(group_count, band_count)
    return {
        "sample_rate": sample_rate,
        "fft_size": fft_size,
        "hop_size": hop_size,
        "group_frames": group_frames,
        "stft_frames": stft_frames,
        "audio_samples": audio_samples,
        "low_hz": low_hz,
        "high_hz": high_hz,
        "codes": codes,
    }


def decode_mid_grid(
    baseline: np.ndarray,
    frequencies: np.ndarray,
    sidecar: dict[str, object],
) -> np.ndarray:
    codes = np.asarray(sidecar["codes"])
    group_frames = int(sidecar["group_frames"])
    levels = (1 << GRID_BITS) - 1
    gains = -GRID_GAIN_LIMIT_DB + 2 * GRID_GAIN_LIMIT_DB * codes / levels
    frame_gains = np.repeat(gains, group_frames, axis=0)[: baseline.shape[0]]
    frame_gains = smooth_time(frame_gains[:, :, None])[:, :, 0]
    edges = np.linspace(
        int(sidecar["low_hz"]),
        int(sidecar["high_hz"]),
        codes.shape[1] + 1,
    )
    output = baseline.copy()
    for band in range(codes.shape[1]):
        upper = edges[band + 1] + (1.0 if band == codes.shape[1] - 1 else 0.0)
        mask = (frequencies >= edges[band]) & (frequencies < upper)
        output[:, mask, 0] *= 10.0 ** (frame_gains[:, band, None] / 20.0)
    return output


def smooth_time(values: np.ndarray) -> np.ndarray:
    kernel = np.array([1.0, 2.0, 3.0, 2.0, 1.0]) / 9.0
    padded = np.pad(values, ((2, 2), (0, 0), (0, 0)), mode="edge")
    output = np.zeros_like(values, dtype=np.float64)
    for index, weight in enumerate(kernel):
        output += weight * padded[index : index + values.shape[0]]
    return output


def expand_controls(sidecar: dict[str, object]) -> tuple[np.ndarray, np.ndarray]:
    group_frames = int(sidecar["group_frames"])
    frame_count = int(sidecar["stft_frames"])
    gain_codes = np.asarray(sidecar["gains"])
    tone_codes = np.asarray(sidecar["tones"])
    gains = GAIN_MIN_DB + (GAIN_MAX_DB - GAIN_MIN_DB) * gain_codes / 15.0
    tones = tone_codes / 3.0
    frame_gains = np.repeat(gains, group_frames, axis=0)[:frame_count]
    frame_tones = np.repeat(tones, group_frames, axis=0)[:frame_count]
    return smooth_time(frame_gains), smooth_time(frame_tones)


def make_excitation(baseline: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    frames, bins, channels = baseline.shape
    harmonic = np.empty_like(baseline)
    noise = np.empty_like(baseline)
    for bin_index in range(bins):
        source_bin = max(1, bin_index // 2)
        magnitude = np.abs(baseline[:, source_bin, :])
        phase = 2.0 * np.angle(baseline[:, source_bin, :])
        harmonic[:, bin_index, :] = magnitude * np.exp(1j * phase)

    frame_index = np.arange(frames, dtype=np.uint64)[:, None, None]
    bin_index = np.arange(bins, dtype=np.uint64)[None, :, None]
    channel_index = np.arange(channels, dtype=np.uint64)[None, None, :]
    state = (
        frame_index * np.uint64(0x9E3779B185EBCA87)
        + bin_index * np.uint64(0xC2B2AE3D27D4EB4F)
        + channel_index * np.uint64(0x165667B19E3779F9)
        + np.uint64(0xD6E8FEB86659FD93)
    )
    state ^= state >> np.uint64(30)
    state *= np.uint64(0xBF58476D1CE4E5B9)
    state ^= state >> np.uint64(27)
    phase = (state.astype(np.float64) / float(2**64)) * (2.0 * np.pi)
    mapped_magnitude = np.abs(baseline[:, np.maximum(1, np.arange(bins) // 2), :])
    noise[:] = mapped_magnitude * np.exp(1j * phase)
    return harmonic, noise


def decode_sidecar(
    baseline: np.ndarray,
    frequencies: np.ndarray,
    sidecar: dict[str, object],
    add_excitation: bool,
) -> np.ndarray:
    edges = np.asarray(sidecar["edges"])
    gains_db, noise_weights = expand_controls(sidecar)
    output = baseline.copy()
    harmonic, noise = make_excitation(baseline)
    epsilon = 1e-18

    for frame in range(baseline.shape[0]):
        for band in range(len(edges) - 1):
            mask = (frequencies >= edges[band]) & (frequencies < edges[band + 1])
            if band == len(edges) - 2:
                mask = (frequencies >= edges[band]) & (frequencies <= edges[band + 1])
            for channel in range(baseline.shape[2]):
                current = baseline[frame, mask, channel]
                ratio = 10.0 ** (gains_db[frame, band, channel] / 20.0)
                if not add_excitation or ratio <= 1.0:
                    output[frame, mask, channel] = current * ratio
                    continue

                weight = float(np.clip(noise_weights[frame, band, channel], 0.0, 1.0))
                candidate = (
                    math.sqrt(1.0 - weight) * harmonic[frame, mask, channel]
                    + math.sqrt(weight) * noise[frame, mask, channel]
                )
                current_power = float(np.mean(np.abs(current) ** 2))
                target_power = current_power * ratio * ratio
                candidate_power = float(np.mean(np.abs(candidate) ** 2)) + epsilon
                cross = float(np.mean(np.real(current * np.conj(candidate))))
                discriminant = max(0.0, cross * cross + candidate_power * (target_power - current_power))
                amplitude = (-cross + math.sqrt(discriminant)) / candidate_power
                output[frame, mask, channel] = current + max(0.0, amplitude) * candidate
    return output


def oracle_high_band(
    source: np.ndarray,
    baseline: np.ndarray,
    frequencies: np.ndarray,
    cutoff: float = 8_000.0,
    transition: float = 1_000.0,
) -> np.ndarray:
    low = cutoff - transition / 2.0
    high = cutoff + transition / 2.0
    position = np.clip((frequencies - low) / (high - low), 0.0, 1.0)
    mask = 0.5 - 0.5 * np.cos(np.pi * position)
    return baseline * (1.0 - mask[None, :, None]) + source * mask[None, :, None]


def waveform_snr(reference: np.ndarray, candidate: np.ndarray) -> float:
    signal = float(np.sum(reference**2))
    error = float(np.sum((reference - candidate) ** 2))
    return 10.0 * math.log10(signal / max(error, 1e-30))


def spectrum_metrics(
    reference: np.ndarray,
    candidate: np.ndarray,
    frequencies: np.ndarray,
) -> dict[str, object]:
    rows = []
    for low, high in [(0, 4_000), (4_000, 8_000), (8_000, 12_000), (12_000, 16_000), (16_000, 20_000), (20_000, 24_000)]:
        mask = (frequencies >= low) & (frequencies < high)
        if high == 24_000:
            mask = (frequencies >= low) & (frequencies <= high)
        reference_band = reference[:, mask, :]
        candidate_band = candidate[:, mask, :]
        signal = float(np.sum(np.abs(reference_band) ** 2))
        error = float(np.sum(np.abs(reference_band - candidate_band) ** 2))
        reference_db = 20.0 * np.log10(np.maximum(np.abs(reference_band), 1e-5))
        candidate_db = 20.0 * np.log10(np.maximum(np.abs(candidate_band), 1e-5))
        rows.append(
            {
                "band_hz": [low, high],
                "complex_snr_db": 10.0 * math.log10(signal / max(error, 1e-30)),
                "log_magnitude_mae_db": float(np.mean(np.abs(reference_db - candidate_db))),
                "energy_bias_db": 10.0
                * math.log10(
                    max(float(np.sum(np.abs(candidate_band) ** 2)), 1e-30)
                    / max(signal, 1e-30)
                ),
            }
        )
    return {"bands": rows}


def main() -> None:
    args = parse_args()
    source = read_audio(args.source, args.sample_rate, args.channels)
    baseline = read_audio(args.baseline, args.sample_rate, args.channels)
    samples = min(source.shape[0], baseline.shape[0])
    source = source[:samples]
    baseline = baseline[:samples]

    source_ms = lr_to_ms(source)
    baseline_ms = lr_to_ms(baseline)
    source_spectrum, crop = stft(source_ms, args.fft_size, args.hop_size)
    baseline_spectrum, baseline_crop = stft(baseline_ms, args.fft_size, args.hop_size)
    if crop != baseline_crop or source_spectrum.shape != baseline_spectrum.shape:
        raise ValueError("source and baseline STFT layouts do not match")

    frequencies = np.fft.rfftfreq(args.fft_size, 1.0 / args.sample_rate)
    edges = band_edges(args.sample_rate)
    gains, tones = encode_cells(
        source_spectrum,
        baseline_spectrum,
        frequencies,
        edges,
        args.group_frames,
    )
    sidecar_bytes = serialize_sidecar(
        gains,
        tones,
        edges,
        args.sample_rate,
        args.fft_size,
        args.hop_size,
        args.group_frames,
        source_spectrum.shape[0],
        samples,
    )
    decoded_sidecar = deserialize_sidecar(sidecar_bytes)
    grid_codes = encode_mid_grid(
        source_spectrum,
        baseline_spectrum,
        frequencies,
        args.grid_low_hz,
        args.grid_high_hz,
        args.grid_bands,
        args.grid_group_frames,
    )
    grid_bytes = serialize_mid_grid(
        grid_codes,
        args.sample_rate,
        args.fft_size,
        args.hop_size,
        args.grid_group_frames,
        source_spectrum.shape[0],
        samples,
        args.grid_low_hz,
        args.grid_high_hz,
    )
    decoded_grid = deserialize_mid_grid(grid_bytes)

    envelope_spectrum = decode_sidecar(
        baseline_spectrum, frequencies, decoded_sidecar, add_excitation=False
    )
    enhanced_spectrum = decode_sidecar(
        baseline_spectrum, frequencies, decoded_sidecar, add_excitation=True
    )
    grid_spectrum = decode_mid_grid(baseline_spectrum, frequencies, decoded_grid)
    oracle_spectrum = oracle_high_band(source_spectrum, baseline_spectrum, frequencies)

    candidates_ms = {
        "baseline": baseline_ms,
        "envelope": istft(
            envelope_spectrum, args.fft_size, args.hop_size, crop, samples
        ),
        "sidecar": istft(
            enhanced_spectrum, args.fft_size, args.hop_size, crop, samples
        ),
        "mid_grid": istft(
            grid_spectrum, args.fft_size, args.hop_size, crop, samples
        ),
        "oracle": istft(oracle_spectrum, args.fft_size, args.hop_size, crop, samples),
    }
    candidates = {name: ms_to_lr(audio) for name, audio in candidates_ms.items()}
    spectra = {
        "baseline": baseline_spectrum,
        "envelope": envelope_spectrum,
        "sidecar": enhanced_spectrum,
        "mid_grid": grid_spectrum,
        "oracle": oracle_spectrum,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "source": args.output_dir / "00-source.wav",
        "baseline": args.output_dir / "01-encodec-baseline.wav",
        "oracle": args.output_dir / "02-oracle-hf-replacement.wav",
        "envelope": args.output_dir / "03-envelope-only.wav",
        "sidecar": args.output_dir / "04-sidecar-envelope-excitation.wav",
        "mid_grid": args.output_dir / "05-mid-grid-sidecar.wav",
    }
    write_audio(paths["source"], source, args.sample_rate)
    for name in ["baseline", "oracle", "envelope", "sidecar", "mid_grid"]:
        write_audio(paths[name], candidates[name], args.sample_rate)

    sidecar_path = args.output_dir / "westside-hf-sidecar.ehf"
    sidecar_path.write_bytes(sidecar_bytes)
    grid_path = args.output_dir / "westside-mid-grid.ehg"
    grid_path.write_bytes(grid_bytes)
    duration = samples / args.sample_rate
    sidecar_kbps = len(sidecar_bytes) * 8.0 / duration / 1_000.0
    grid_kbps = len(grid_bytes) * 8.0 / duration / 1_000.0
    ecdc_bytes = args.ecdc.stat().st_size if args.ecdc.is_file() else None
    ecdc_kbps = ecdc_bytes * 8.0 / duration / 1_000.0 if ecdc_bytes else None

    report: dict[str, object] = {
        "schema": "wavey.encodec.hf-sidecar-spike.v1",
        "source": str(args.source.resolve()),
        "baseline": str(args.baseline.resolve()),
        "settings": {
            "sample_rate": args.sample_rate,
            "channels": args.channels,
            "fft_size": args.fft_size,
            "hop_size": args.hop_size,
            "group_frames": args.group_frames,
            "control_interval_ms": args.group_frames * args.hop_size * 1_000.0 / args.sample_rate,
            "band_edges_hz": edges.tolist(),
            "gain_bits": 4,
            "flatness_bits": 2,
            "mid_grid": {
                "low_hz": args.grid_low_hz,
                "high_hz": args.grid_high_hz,
                "bands": args.grid_bands,
                "group_frames": args.grid_group_frames,
                "control_interval_ms": (
                    args.grid_group_frames
                    * args.hop_size
                    * 1_000.0
                    / args.sample_rate
                ),
                "gain_bits": GRID_BITS,
                "channel": "mid",
            },
        },
        "sizes": {
            "duration_seconds": duration,
            "ecdc_bytes": ecdc_bytes,
            "ecdc_effective_kbps": ecdc_kbps,
            "sidecar_bytes": len(sidecar_bytes),
            "sidecar_kbps": sidecar_kbps,
            "combined_effective_kbps": ecdc_kbps + sidecar_kbps if ecdc_kbps else None,
            "mid_grid_sidecar_bytes": len(grid_bytes),
            "mid_grid_sidecar_kbps": grid_kbps,
            "mid_grid_combined_effective_kbps": (
                ecdc_kbps + grid_kbps if ecdc_kbps else None
            ),
        },
        "outputs": {name: str(path.resolve()) for name, path in paths.items()},
        "metrics": {},
    }
    for name, candidate in candidates.items():
        report["metrics"][name] = {
            "waveform_snr_db": waveform_snr(source, candidate),
            "peak": float(np.max(np.abs(candidate))),
            **spectrum_metrics(source_spectrum, spectra[name], frequencies),
        }
    report_path = args.output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")

    print(json.dumps(report["sizes"], indent=2))
    print("\nHigh-band log-magnitude MAE (8-24 kHz):")
    for name in ["baseline", "envelope", "sidecar", "mid_grid", "oracle"]:
        high_rows = report["metrics"][name]["bands"][2:]
        mean_mae = float(np.mean([row["log_magnitude_mae_db"] for row in high_rows]))
        print(f"  {name:9s} {mean_mae:7.3f} dB")
    print(f"\nArtifacts: {args.output_dir}")


if __name__ == "__main__":
    main()
