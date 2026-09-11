#!/usr/bin/env python3
"""What predicts the seven-codebook bitrate, beyond level.

`corpus-7cb-bitrate.mjs` established that level explains 42% of the
clip-to-clip variance and leaves the rest unaccounted for, with the residue
separating by record in ways level does not explain. This measures the two
obvious candidates on the same forty clips, without re-encoding anything:

  flux      normalised spectral flux — how fast the *shape* of the spectrum
            changes, frame to frame. Each magnitude frame is normalised to
            unit sum before differencing, so a loud passage and a quiet one
            with the same spectral motion score the same. Without that
            normalisation this would simply re-measure level and confirm
            itself.

  flatness  spectral flatness, the geometric mean of the power spectrum over
            its arithmetic mean. 0 is a pure tone, 1 is white noise. This is
            the direct test of the claim that noise-like material is
            expensive: if brushes and hiss cost bytes, flatness should carry
            a positive coefficient.

Usage: python3 scripts/corpus-7cb-predictors.py --results DIR
"""

import argparse
import json
import math
import os
import struct
import sys
import wave

import numpy as np

WINDOW = 1024
HOP = 512

TITLES = {
    "bill-evans-secret-sessions": "Bill Evans",
    "blue-nile-hats": "Blue Nile",
    "lori-asha": "Lori Asha",
    "nocturnal-animals": "Nocturnal",
}


def read_float_wav(path):
    """The clips as mono float, at whatever the writer put down."""
    with open(path, "rb") as handle:
        raw = handle.read()
    assert raw[:4] == b"RIFF" and raw[8:12] == b"WAVE", path
    channels = struct.unpack_from("<H", raw, 22)[0]
    bits = struct.unpack_from("<H", raw, 34)[0]
    assert bits == 32, f"{path}: expected float32, got {bits}-bit"
    data = np.frombuffer(raw, dtype="<f4", offset=44)
    return data.reshape(-1, channels).mean(axis=1)


def spectra(signal):
    """Magnitude spectra, one row per frame."""
    window = np.hanning(WINDOW)
    frames = 1 + (len(signal) - WINDOW) // HOP
    out = np.empty((frames, WINDOW // 2 + 1))
    for i in range(frames):
        block = signal[i * HOP: i * HOP + WINDOW] * window
        out[i] = np.abs(np.fft.rfft(block))
    return out


def normalised_flux(mag):
    """Mean L1 distance between consecutive unit-sum spectra.

    Unit-sum first: this is a measure of spectral *motion*, and it has to be
    blind to how loud the passage is or it is a second level meter.
    """
    total = mag.sum(axis=1, keepdims=True)
    total[total == 0] = 1.0
    shape = mag / total
    return float(np.abs(np.diff(shape, axis=0)).sum(axis=1).mean())


def flatness(mag):
    """Mean spectral flatness. 0 tonal, 1 white noise."""
    power = np.maximum(mag ** 2, 1e-20)
    geometric = np.exp(np.log(power).mean(axis=1))
    arithmetic = power.mean(axis=1)
    return float((geometric / arithmetic).mean())


def corr(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.corrcoef(a, b)[0, 1])


def partial(x, y, control):
    """r(x, y) with `control` held out of both."""
    rxy, rxc, ryc = corr(x, y), corr(x, control), corr(y, control)
    return (rxy - rxc * ryc) / math.sqrt((1 - rxc ** 2) * (1 - ryc ** 2))


def regress(predictors, target):
    """Standardised betas and R^2."""
    X = np.column_stack([
        (np.asarray(p, float) - np.mean(p)) / np.std(p) for p in predictors
    ])
    y = (np.asarray(target, float) - np.mean(target)) / np.std(target)
    betas, *_ = np.linalg.lstsq(X, y, rcond=None)
    fitted = X @ betas
    ss_res = float(((y - fitted) ** 2).sum())
    ss_tot = float((y ** 2).sum())
    return betas, 1 - ss_res / ss_tot


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="target/corpus-7cb-bitrate")
    args = parser.parse_args()

    summary = json.load(open(os.path.join(args.results, "corpus-7cb-bitrate.json")))
    rows = []
    for row in summary["rows"]:
        name = f"{row['group']}-{row['slot']:02d}.wav"
        signal = read_float_wav(os.path.join(args.results, "clips", name))
        mag = spectra(signal)
        rows.append({
            "group": row["group"],
            "slot": row["slot"],
            "kbps": row["payloadBytes"] * 8 / row["seconds"] / 1000,
            "rms": row["rmsDbfs"],
            "flux": normalised_flux(mag),
            "flatness": flatness(mag),
        })

    kbps = [r["kbps"] for r in rows]
    rms = [r["rms"] for r in rows]
    flux = [r["flux"] for r in rows]
    flat = [r["flatness"] for r in rows]

    print("Per record, mean of each measure\n")
    print(f"{'record':12} {'kbps':>7} {'RMS':>8} {'flux':>8} {'flatness':>10}")
    for gid, title in TITLES.items():
        sel = [r for r in rows if r["group"] == gid]
        print(f"{title:12} {np.mean([r['kbps'] for r in sel]):7.3f} "
              f"{np.mean([r['rms'] for r in sel]):8.1f} "
              f"{np.mean([r['flux'] for r in sel]):8.4f} "
              f"{np.mean([r['flatness'] for r in sel]):10.5f}")

    print("\nSimple correlation with cost, all 40 clips\n")
    for label, series in (("level (RMS dBFS)", rms), ("spectral flux", flux),
                          ("spectral flatness", flat)):
        r = corr(series, kbps)
        print(f"  {label:20} r = {r:+.3f}   explains {r*r*100:4.1f}%")

    print("\nHolding level constant\n")
    print(f"  flux     vs cost | level   r = {partial(flux, kbps, rms):+.3f}")
    print(f"  flatness vs cost | level   r = {partial(flat, kbps, rms):+.3f}")
    print(f"  flux     vs level          r = {corr(flux, rms):+.3f}")
    print(f"  flatness vs level          r = {corr(flat, rms):+.3f}")

    print("\nTogether\n")
    for labels, preds in (
        (["level"], [rms]),
        (["level", "flux"], [rms, flux]),
        (["level", "flatness"], [rms, flat]),
        (["level", "flux", "flatness"], [rms, flux, flat]),
    ):
        betas, r2 = regress(preds, kbps)
        shown = "  ".join(f"{n} {b:+.3f}" for n, b in zip(labels, betas))
        print(f"  {' + '.join(labels):26} R2 = {r2:.3f}   beta: {shown}")

    out = os.path.join(args.results, "corpus-7cb-predictors.json")
    with open(out, "w") as handle:
        json.dump({"rows": rows}, handle, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    sys.exit(main())
