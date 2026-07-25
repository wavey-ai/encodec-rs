# Original Meta EnCodec versus encodec-rs

Status: direct neural PCM and encoded-size comparison.

Meta reference commit: `0e2d0aed29362c8e8f52494baf3e6f99056b214f`.
encodec-rs commit: `e3fd8a8e64f6498817dbe9a8a138f25a5d4a18d6`.

This run contains 24 direct comparisons. It uses the same pinned neural checkpoint and candidate window geometry.
The fork output uses its current ONNX and acv=2 path, so this is not Profile 1 evidence.

The size comparison uses the unmodified Meta ECDC stream with LM entropy and the fork current acv2 ECDC stream. The fork now exports qualification-only raw code and scale evidence, but direct code and entropy parity remains a separate blocked gate.
The report retains decoded-length mismatches. The Meta fixed-window decoder can return a shorter final output, while the fork path targets the source length.
Each JSON row also retains SI-SDR, error RMS dBFS, peak, channel, clipping, non-finite, and seam metrics.

| Geometry | Rate | Rows | Meta length mismatches | Fork length mismatches | Source-to-original SNR (dB) | Source-to-fork SNR (dB) | Original-to-fork SNR (dB) | Meta ECDC (bytes) | Fork acv2 ECDC (bytes) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fixed-1333 | 6 | 6 | 5 | 0 | 7.327 | 7.312 | 6.690 | 115981.5 | 124839.2 |
| fixed-1333 | 12 | 6 | 5 | 0 | 9.006 | 8.994 | 8.532 | 241831.7 | 257531.7 |
| fixed-1800 | 6 | 6 | 5 | 0 | 7.325 | 7.322 | 6.701 | 114894.8 | 123349.8 |
| fixed-1800 | 12 | 6 | 5 | 0 | 9.008 | 9.003 | 8.545 | 240218.5 | 255443.5 |
