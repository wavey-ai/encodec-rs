# The seven-codebook profile, measured across a genre corpus

Test date: 2026-09-04.

Bundle: `dist/wasm-fixed-bundles/bundles/encodec_48khz_12kbps_7cb_1333ms`.

Backend: custom SIMD WASM kernels, q8 language model, in Node.

## Why this exists

Every recorded bitrate for this profile was one track long.
`docs/browser-backend-parity-and-bitrate-matrix.md` measures PRAY 4 ME and
nothing else: 246,737 B over 222.532 s, 8.870 kbps whole-file. The
eight-track study in `qualification-results/20260814-lori-confirmation-bitrate/`
looks like a corpus but is the **eight**-codebook bundle, and is one artist.

So the number that sizes a record — how much audio fits in a groove, which
format a programme is offered — rested on a single song. This measures forty
clips from four records that do not sound alike.

## Corpus

`soundkit/testdata/flac-packet-bench/diverse-v1`, seed `soundkit-flac-diverse-v1`.
Four groups of ten 10 s slots, 48 kHz stereo, 24-bit.

## Method

Each clip is the head of its slot, cut to a whole number of encoder chunks:
the profile strides 64,000 samples, so seven chunks is 448,000 samples,
9.33333 s. Ragged clips would have measured the encoder's own zero padding
as if it were programme.

The corpus keeps 24-bit audio right-justified in 32-bit words — peak
5,310,116 against 2^23. Read as s32 it is 48 dB down, and quiet audio codes
small; that mistake produces a plausible and completely wrong answer. Clips
are converted to float over 2^23.

Payload bitrate excludes the 96-byte header and the 8 bytes of framing each
chunk carries, because a groove carries chunks. Whole-file figures are kept
beside it since that is what a file on disk measures.

Every clip is asserted to have encoded `nc == 7`, so this cannot silently be
measuring the eight-codebook bundle — which is exactly how the ONNX root's
seven-codebook bundle went wrong.

Reproduce with `node scripts/corpus-7cb-bitrate.mjs`.

## Results, by record

Effective payload kbps.

| Record | Material | Clips | Min | Mean | Max | Range | SD |
|---|---|---:|---:|---:|---:|---:|---:|
| Bill Evans, *The Secret Sessions* | jazz trio, live to two-track | 10 | 5.797 | 7.900 | 9.130 | 3.333 | 1.120 |
| The Blue Nile, *Hats* | studio pop, 1989 | 10 | 7.755 | 8.177 | 8.544 | 0.789 | 0.267 |
| Lori Asha | contemporary R&B | 10 | 8.417 | 8.883 | 9.507 | 1.090 | 0.412 |
| *Nocturnal Animals* | orchestral score | 10 | 6.377 | 7.448 | 8.220 | 1.843 | 0.502 |
| **All four** | | **40** | **5.797** | **8.102** | **9.507** | **3.710** | **0.841** |

## What this says

The spread is the finding, not the mean.

Across the corpus the payload runs **5.797 to 9.507 kbps** — the densest clip
costs 1.64x the sparsest. A capacity number taken from the mean is wrong for a
quarter of the material.

### The symbol count never changes

This is what makes the rest legible. Seven codebooks at 203 frames per
1.3333 s chunk is **1,066 symbols per second, always** — one index per
codebook per frame, each into a 1024-entry table. Ten bits apiece is
**10.658 kbps raw**, and every clip in this corpus carries the identical
symbol count.

| | Payload kbps | Saving on the raw budget |
|---|---:|---:|
| Cheapest clip | 5.797 | 45.6% |
| Mean | 8.102 | 24.0% |
| Dearest clip | 9.507 | 10.8% |

So none of the 3.7 kbps of spread is *more information*. All of it is the
language model predicting the next code well or badly. The bitrate of this
profile is a direct read-out of how predictable the material is, and nothing
else.

### Level is the driver

Cost tracks how loud the material is, and it does so strongly.

| Record | Mean RMS | Mean kbps | Min | Max | kbps SD | RMS spread |
|---|---:|---:|---:|---:|---:|---:|
| Bill Evans | −24.3 dBFS | **7.900** | 5.797 | 9.130 | 1.120 | 16.9 dB |
| *Nocturnal Animals* | −23.2 dBFS | **7.448** | 6.377 | 8.220 | 0.502 | 21.2 dB |
| The Blue Nile, *Hats* | −20.4 dBFS | **8.177** | 7.755 | 8.544 | 0.267 | 9.6 dB |
| Lori Asha | −16.3 dBFS | **8.883** | 8.417 | 9.507 | 0.412 | 5.0 dB |

Record mean level against record mean cost is **r = +0.906**. Clip by clip
across all forty it is **r = +0.645**, so level alone explains 42% of the
variance. Crest factor explains nothing at all (r = +0.046).

The reading is the plain one: a quiet passage produces codes drawn from a
narrow, stationary part of the distribution, the model predicts them well, and
the arithmetic coder spends little. Loud, dense material moves through the
codebook constantly and the model is wrong more often. **Entropy of the coded
signal, and level is most of it.**

So the cheapest record here is the quietest. Bill Evans averages −24.3 dBFS
and costs 7.900 kbps; Lori Asha averages −16.3 and costs 8.883. Eight decibels
of level is roughly a kilobit per second, and on a disc that is centimetres of
groove.

Within a record the same rule holds wherever there is level to vary. Bill
Evans is the clearest case, and its clips sort almost perfectly by level:

| Slot | kbps | RMS |
|---:|---:|---:|
| 7 | 5.797 | −30.2 dBFS |
| 6 | 6.943 | −33.3 dBFS |
| 4 | 8.450 | −24.9 dBFS |
| 5 | 8.588 | −16.6 dBFS |
| 3 | 9.027 | −16.4 dBFS |

That is r = +0.720 within the one record, and it is why Bill Evans holds both
the corpus minimum and a maximum within 0.4 kbps of the corpus maximum. It is
not an expensive record. It is a quiet record that gets loud.

### The other half is spectral motion, and it is not noise

Level leaves 58% of the clip-to-clip variance unexplained. Two candidates were
measured on the same forty clips, without re-encoding: **normalised spectral
flux** — how fast the *shape* of the spectrum moves, with every magnitude
frame normalised to unit sum first so it cannot double as a level meter — and
**spectral flatness**, 0 for a pure tone and 1 for white noise.
`scripts/corpus-7cb-predictors.py` computes both.

| Predictor | r with cost | Explains | Holding level constant |
|---|---:|---:|---:|
| Level (RMS dBFS) | +0.645 | 41.6% | — |
| Spectral flux | **+0.564** | 31.9% | **+0.403** |
| Spectral flatness | **−0.264** | 6.9% | +0.173 |

| Model | R² |
|---|---:|
| level | 0.416 |
| **level + flux** | **0.511** |
| level + flatness | 0.433 |
| level + flux + flatness | 0.515 |

Flux is a real second driver: it survives holding level constant at +0.403 and
adds nine and a half points of R² on its own. Flatness adds essentially
nothing over flux, and its coefficient turns negative in the full model.

**Noise-like material is cheaper, not dearer.** Flatness correlates
*negatively* with cost. The intuition that a brushed snare or tape hiss must
be expensive because noise is high-entropy audio does not survive contact with
the measurement, and this corpus is well placed to test it: Bill Evans is the
noisiest record in it by a factor of four, and the second cheapest.

| Record | kbps | RMS | Flux | Flatness |
|---|---:|---:|---:|---:|
| *Nocturnal Animals* | 7.448 | −23.2 dBFS | **0.3815** | 0.00236 |
| Bill Evans | 7.900 | −24.3 dBFS | 0.4656 | **0.00970** |
| The Blue Nile, *Hats* | 8.177 | −20.4 dBFS | 0.4332 | 0.00253 |
| Lori Asha | 8.883 | −16.3 dBFS | **0.4681** | 0.00147 |

The reason is that entropy at the *code* level is not entropy at the sample
level. Hiss is unpredictable sample to sample and entirely predictable frame
to frame: its code distribution is stationary, so the language model settles
on it and stays right. What the model cannot follow is *change* — an onset, a
chord moving, a spectrum sliding. Stillness is cheap however noisy it is.

That resolves the records. *Nocturnal Animals* is quiet and spectrally still,
so it is cheapest despite the widest dynamic range in the corpus. Lori Asha is
loud and the busiest, so it is dearest. Bill Evans is quiet but busy, and
lands low. *Hats* sits mid on both, which is where its cost sits.

Level and flux together account for 51.5%. The remaining half is not measured
here. Code-transition rate — taken from the encoder's own output rather than
from the signal — is the obvious next term, and it needs the codes kept rather
than counted.

### What it looks like on a record

For a groove that carries its own payload, this is not an abstraction: the
length of the cut *is* the entropy of the music. The same three minutes, cut
on the 10 in profile, at each end of the corpus:

| Material | kB/s | Payload | Ends at | Travel | Band used | Deadwax |
|---|---:|---:|---:|---:|---:|---:|
| Bill Evans, sparsest clip | 0.725 | 130,500 B | 220.6 px | 25.5 mm | 41% | 36.1 mm |
| Corpus mean | 1.013 | 182,340 B | 192.5 px | 37.8 mm | 61% | 23.8 mm |
| Lori Asha, densest clip | 1.188 | 213,840 B | 173.2 px | 46.2 mm | 75% | 15.4 mm |

Two records of identical duration finish 47 px apart, and the run-out more
than doubles. On a real disc that gradient is duration — a long side runs
closer to the label. Here it is predictability, so a sparse acoustic recording
leaves a wide deadwax and a dense produced one runs in toward the paper. The
record is a plot of its own information content, at life size.

| Measurement | Payload kbps | kB/s |
|---|---:|---:|
| Minimum | 5.797 | 0.725 |
| Mean | 8.102 | 1.013 |
| Median | 8.332 | 1.041 |
| Maximum | 9.507 | 1.188 |
| Standard deviation | 0.841 | 0.105 |
| Whole-file mean | 8.232 | 1.029 |

The single-track figure this replaces, PRAY 4 ME at 8.870 kbps whole-file,
sits above the corpus whole-file mean of 8.232 but inside its range.
It was not a bad estimate. It was an estimate with no error bar, and the
error bar is 46% of the mean.

## Per clip

| Group | Slot | ECDC B | Payload B | Payload kbps | kB/s | RMS dBFS |
|---|---:|---:|---:|---:|---:|---:|
| Bill Evans | 1 | 9,303 | 9,151 | 7.844 | 0.980 | -31.9 |
| Bill Evans | 2 | 10,030 | 9,878 | 8.467 | 1.058 | -21.9 |
| Bill Evans | 3 | 10,683 | 10,531 | 9.027 | 1.128 | -16.4 |
| Bill Evans | 4 | 10,010 | 9,858 | 8.450 | 1.056 | -24.9 |
| Bill Evans | 5 | 10,171 | 10,019 | 8.588 | 1.074 | -16.6 |
| Bill Evans | 6 | 8,252 | 8,100 | 6.943 | 0.868 | -33.3 |
| Bill Evans | 7 | 6,915 | 6,763 | 5.797 | 0.725 | -30.2 |
| Bill Evans | 8 | 7,383 | 7,231 | 6.198 | 0.775 | -26.4 |
| Bill Evans | 9 | 10,803 | 10,651 | 9.130 | 1.141 | -21.6 |
| Bill Evans | 10 | 10,130 | 9,978 | 8.553 | 1.069 | -19.6 |
| Blue Nile | 1 | 9,855 | 9,703 | 8.317 | 1.040 | -27.1 |
| Blue Nile | 2 | 9,821 | 9,669 | 8.288 | 1.036 | -17.6 |
| Blue Nile | 3 | 9,199 | 9,047 | 7.755 | 0.969 | -18.1 |
| Blue Nile | 4 | 9,254 | 9,102 | 7.802 | 0.975 | -20.9 |
| Blue Nile | 5 | 9,872 | 9,720 | 8.332 | 1.041 | -18.9 |
| Blue Nile | 6 | 10,120 | 9,968 | 8.544 | 1.068 | -18.0 |
| Blue Nile | 7 | 10,013 | 9,861 | 8.453 | 1.057 | -19.5 |
| Blue Nile | 8 | 9,900 | 9,748 | 8.356 | 1.044 | -27.1 |
| Blue Nile | 9 | 9,470 | 9,318 | 7.987 | 0.998 | -18.0 |
| Blue Nile | 10 | 9,406 | 9,254 | 7.932 | 0.992 | -18.9 |
| Lori Asha | 1 | 10,144 | 9,992 | 8.565 | 1.071 | -14.4 |
| Lori Asha | 2 | 11,047 | 10,895 | 9.339 | 1.167 | -15.5 |
| Lori Asha | 3 | 10,089 | 9,937 | 8.518 | 1.065 | -16.0 |
| Lori Asha | 4 | 10,601 | 10,449 | 8.957 | 1.120 | -19.4 |
| Lori Asha | 5 | 9,976 | 9,824 | 8.421 | 1.053 | -17.4 |
| Lori Asha | 6 | 10,161 | 10,009 | 8.579 | 1.072 | -16.1 |
| Lori Asha | 7 | 11,137 | 10,985 | 9.416 | 1.177 | -14.8 |
| Lori Asha | 8 | 11,243 | 11,091 | 9.507 | 1.188 | -15.4 |
| Lori Asha | 9 | 10,777 | 10,625 | 9.107 | 1.138 | -15.8 |
| Lori Asha | 10 | 9,971 | 9,819 | 8.417 | 1.052 | -18.3 |
| Nocturnal Animals | 1 | 8,722 | 8,570 | 7.346 | 0.918 | -22.9 |
| Nocturnal Animals | 2 | 9,742 | 9,590 | 8.220 | 1.028 | -11.4 |
| Nocturnal Animals | 3 | 8,855 | 8,703 | 7.460 | 0.932 | -32.6 |
| Nocturnal Animals | 4 | 9,113 | 8,961 | 7.681 | 0.960 | -15.2 |
| Nocturnal Animals | 5 | 7,592 | 7,440 | 6.377 | 0.797 | -25.2 |
| Nocturnal Animals | 6 | 8,404 | 8,252 | 7.073 | 0.884 | -22.4 |
| Nocturnal Animals | 7 | 8,962 | 8,810 | 7.552 | 0.944 | -25.7 |
| Nocturnal Animals | 8 | 8,645 | 8,493 | 7.280 | 0.910 | -30.9 |
| Nocturnal Animals | 9 | 8,696 | 8,544 | 7.324 | 0.915 | -30.3 |
| Nocturnal Animals | 10 | 9,678 | 9,526 | 8.165 | 1.021 | -15.4 |

