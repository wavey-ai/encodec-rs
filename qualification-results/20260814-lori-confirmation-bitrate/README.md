# Lori Confirmation Mix Bitrate Variation

Test date: 2026-08-14.

This test measures effective payload bitrate variation across eight Lori Asha
confirmation mixes. The encoder used the `encodec_48khz_12kbps_1333ms`
bundle and the q8 language model.

The calculation excludes the ECDC header and per-chunk framing. It uses this
formula:

```text
effective payload kbps = payload bytes * 8 / duration seconds / 1000
```

The duration comes from the ECDC `al` source-frame field at 48 kHz.

## Results

| Track | Duration | Payload | Effective payload bitrate | Difference from mean |
|---|---:|---:|---:|---:|
| I Want Her | 148.000 s | 188,244 bytes | 10.175 kbps | -0.077 kbps |
| Sugar Free | 166.443 s | 214,918 bytes | 10.330 kbps | +0.077 kbps |
| As It Seems | 174.336 s | 214,994 bytes | 9.866 kbps | -0.387 kbps |
| Westside V2 | 192.936 s | 256,045 bytes | 10.617 kbps | +0.364 kbps |
| After Dark | 198.571 s | 257,742 bytes | 10.384 kbps | +0.131 kbps |
| Pray 4 Me | 222.532 s | 282,347 bytes | 10.150 kbps | -0.102 kbps |
| 1979 | 227.863 s | 295,912 bytes | 10.389 kbps | +0.136 kbps |
| There Is a Light | 232.913 s | 294,367 bytes | 10.111 kbps | -0.142 kbps |

## Variation

| Measurement | Result |
|---|---:|
| Arithmetic mean | 10.253 kbps |
| Population variance | 0.0451 kbps² |
| Population standard deviation | 0.212 kbps |
| Minimum | 9.866 kbps |
| Maximum | 10.617 kbps |
| Range | 0.751 kbps |
| Coefficient of variation | 2.071% |

The duration-weighted mean is 10.256 kbps. The arithmetic and weighted means
differ by 0.003 kbps.
