# High-frequency sidecar experiment

Status: closed as a no-go on 2026-08-22.

## Question

Can a small, source-informed sidecar improve the thin or cold high-frequency
character of the 48 kHz EnCodec decode without retraining the codec?

## Fixture

- Source: `WESTSIDE_V2_MIX 4 CONFIRMATION_140323.wav`
- Source format: 48 kHz stereo
- Duration: 192.9358125 seconds
- Samples per channel: 9,260,919
- EnCodec payload: 257,395 bytes, or 10.6728 kbps

The source audio, decoded WAV files, ViSQOL PCM files, model weights, and work
files are generated or local assets. They remain under the ignored
`target/experiments/hf-sidecar-westside-v2/` tree and are not committed.

## Candidates

- `baseline`: the normal overlap-repaired EnCodec decode.
- `hard_crop`: a control without overlap repair.
- `oracle`: EnCodec below 8 kHz and the source above 8 kHz. This is an upper
  bound, not a usable codec.
- `envelope`: a coarse source-to-decode high-band energy correction.
- `excitation`: the coarse envelope plus deterministic harmonic and noise
  excitation.
- `mid_grid`: a denser 4-20 kHz correction for the stereo mid channel.

The coarse sidecar is 23,078 bytes, or 0.9569 kbps. The combined payload is
11.6297 kbps. The mid-grid sidecar is 30,947 bytes, or 1.2832 kbps. Its combined
payload is 11.9560 kbps.

## Results

Full-file objective metrics:

| Candidate | SNR dB | SI-SDR dB | LSD dB | Spectral convergence |
|---|---:|---:|---:|---:|
| baseline | 8.002 | 7.360 | 13.666 | 0.24414 |
| hard crop | 8.004 | 7.362 | 13.744 | 0.24411 |
| oracle | 8.306 | 7.706 | 4.621 | 0.23832 |
| envelope | 7.975 | 7.330 | 14.242 | 0.24447 |
| excitation | 8.002 | 7.359 | 13.764 | 0.24407 |
| mid-grid | 7.907 | 7.258 | 11.452 | 0.24598 |

ViSQOL audio MOS-LQO used four reproducible random eight-second samples. The
starts were 38, 71, 100, and 159 seconds. The random seed was `20260822`.

| Candidate | Mean | Delta vs baseline | Nominal 95% CI |
|---|---:|---:|---:|
| baseline | 4.2094 | 0.0000 | 0.0000 to 0.0000 |
| hard crop | 4.2032 | -0.0062 | -0.0145 to +0.0020 |
| oracle | 4.5644 | +0.3550 | +0.2150 to +0.4950 |
| envelope | 4.2663 | +0.0569 | +0.0313 to +0.0826 |
| excitation | 4.2436 | +0.0342 | +0.0201 to +0.0483 |
| mid-grid | 4.2314 | +0.0221 | -0.0485 to +0.0926 |

ViSQOL downmixes stereo to mono. Its audio model was trained at bitrates of
24 kbps and above. The tested payload is about 10.7-12.0 kbps. Four samples are
also too few to treat small deltas as proof of an audible improvement. The
ViSQOL changes below 0.06 are secondary evidence only.

## Decision

Do not integrate a sidecar into ECDC.

The oracle result confirms that correct high-frequency reconstruction can
improve the decode. The practical sidecars do not recover that information.
Envelope and excitation get small ViSQOL increases, but they do not improve
full-file fidelity. Mid-grid improves log-magnitude error but reduces waveform
accuracy, has an inconclusive ViSQOL result, and over-boosts the 16-20 kHz band.
None of these tradeoffs justify another 0.96-1.28 kbps.

DAC is also not a viable WASM replacement. Its tested decoder graph is about
216.85 MB, compared with roughly 30 MB for EnCodec. Its full-track compressed
artifact was 25.88 kbps, or 2.64 times the existing 9.79 kbps golden ECDC
artifact. See [Alternative Codec Benchmarks](../../ALTERNATIVE_CODEC_BENCHMARKS.md).

Keep the current EnCodec path. Revisit high-frequency quality only with a
materially better compact codec, codec retraining, or a blind listening result
that demonstrates a useful preference before adding payload bits.

## Files

- `spike-hf-sidecar.py` implements the experimental sidecar formats and
  reconstruction methods.
- `spike-hf-sidecar-full.py` applies the spike to a complete file in exact
  sample chunks and joins the outputs.
- `benchmark-hf-sidecar-quality.py` computes the objective metrics and runs the
  reproducible sampled ViSQOL audit.
- `results.json` contains the compact machine-readable result record.
