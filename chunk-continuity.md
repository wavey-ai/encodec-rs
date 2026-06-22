# Best Way to Adjoin Chunked EnCodec Audio

## Executive answer

The best current approach is:

```text
encode each logical chunk with ±10 ms of real neighbouring source context
→ decode the full guarded window
→ crop back to the exact owned chunk length
→ adjoin chunks at the exact timeline boundary
→ apply a very short 0.5 ms cubic Hermite seam repair
```

For the Bitneedle 48 kHz revolution profile:

```text
left guard:          480 samples
owned chunk:      64,000 samples
right guard:         480 samples
model window:     64,960 samples
frame length:          203
average ECDC size overhead: ~1.47%
```

The logical chunk remains exactly 64,000 samples. The guard samples are codec context only and are discarded after decoding.

This approach gave the best balance found so far between:

- seam quality;
- exact timeline preservation;
- independently decodable chunks;
- low encoded-size overhead;
- low implementation complexity;
- minimal alteration of the decoded waveform.

A more conservative alternative is:

```text
±30 ms guard
+ 0.25 ms cubic Hermite repair
```

This slightly improved the worst measured seam, but increased average ECDC size by about 3.81%, compared with about 1.47% for the ±10 ms design.

---

## Why simple hard concatenation fails

Independently encoded and decoded EnCodec chunks do not necessarily meet continuously at their boundaries.

The source may be locally smooth:

```text
source sample N     → source sample N+1
```

but the decoded chunks may produce:

```text
last sample of chunk A
    ≠
first sample of chunk B
```

Real-audio testing showed that the codec could reverse the direction of the waveform at a boundary and introduce a much larger discontinuity than was present in the source.

Example:

```text
source jump, channel 1:   +0.007690430
decoded jump:             -0.088623047
codec-added discontinuity: -0.096313477
```

That produces a broadband impulse perceived as a tick.

The defect has two parts:

1. **A broader decoder-edge degradation region** near the start and end of independently processed windows.
2. **A final instantaneous sample and slope mismatch** at the exact join.

The best solution treats these separately:

```text
small source-context guard
    handles broader edge degradation

short Hermite interpolation
    handles the final local mismatch
```

---

## Approaches tested

## 1. Hard concatenation of exact chunks

Initial design:

```text
64,000-sample source chunk
→ encode independently
→ decode independently
→ hard concatenate
```

Result:

- exact chunk ownership was preserved;
- output length was correct;
- audible ticks occurred at joins;
- sample-domain inspection confirmed codec-added discontinuities.

Conclusion:

> Hard concatenation of independently encoded exact EnCodec windows is not reliable.

---

## 2. Treating the end click as a playback-stop artefact

An isolated WAV can click when playback moves directly from its final non-zero sample to silence.

That possibility was checked, but it did not explain the joins between adjacent decoded chunks.

The relevant test is:

```text
last decoded sample of chunk A
versus
first decoded sample of chunk B
```

not:

```text
last sample
versus
silence
```

Conclusion:

> The seam tick was real and occurred between independently decoded chunks.

---

## 3. Waveform and spectrogram inspection

Waveform plots showed a clear boundary discontinuity and, in some tests, a visible amplitude collapse and recovery around the model edge.

Spectrograms showed a broadband vertical event at the seam.

However, the spectrogram used a 1,024-sample FFT at 48 kHz:

```text
1024 / 48000 ≈ 21.33 ms
```

So its apparent seam width was dominated by time-frequency smearing and could not determine the actual damaged-edge width.

Conclusion:

> Spectrograms are useful for confirming a broadband click, but source-versus-decoded sample-domain error is the correct tool for sizing the guard.

---

## 4. Using the model’s nominal 10 ms segment overlap

The original bundle reported:

```text
segment_samples = 64,000
segment_stride  = 63,520
difference      =    480 samples = 10 ms
```

This suggested a 480-sample overlap-add might solve the problem.

Tests used:

- 64,000-sample windows;
- 63,520-sample stride;
- 480-sample linear overlap-add;
- 480-sample equal-power overlap-add.

These sounded worse than the longer overlap treatments.

Conclusion:

> The bundle’s nominal segment stride does not, by itself, solve independently encoded chunk seams.

---

## 5. Long overlap crossfades

Decoded exact chunks were recombined using overlaps from 5 ms through 120 ms.

The 80–120 ms versions sounded best.

This demonstrated that blending well inside the stable interior of adjacent decoded chunks could hide the seam.

But long overlap crossfades have serious drawbacks:

- they shorten the timeline unless duplicate source context is explicitly encoded;
- they blur exact chunk ownership;
- they modify a significant region of audio;
- they hide the edge defect rather than preventing it;
- they complicate random access and exact addressing.

Conclusion:

> Long crossfades are useful diagnostically but are not the preferred format-level solution.

---

## 6. Large guard-context graphs

The next design encoded each logical chunk with real neighbouring source context on both sides.

For example, ±80 ms:

```text
left guard:       3,840 samples
owned chunk:     64,000 samples
right guard:      3,840 samples
model window:    71,680 samples
```

After decoding, only the central 64,000 samples were retained.

This preserved:

- exact logical chunk length;
- exact timeline;
- independent ECDC objects;
- direct hard concatenation of retained chunks.

Initial short tests showed monotonic improvement with guard size.

A longer 30-second test showed:

| Variant | Average ECDC overhead | Median discontinuity | P95 | Maximum |
|---|---:|---:|---:|---:|
| Baseline | 0% | 0.0331 | 0.1509 | 0.1658 |
| Guard 120 ms | 15.5% | 0.0160 | 0.0655 | 0.0790 |
| Guard 160 ms | 21.0% | 0.0122 | 0.0513 | 0.0683 |
| Guard 200 ms | 26.0% | 0.0217 | 0.0493 | 0.0782 |
| Guard 240 ms | 31.1% | 0.0155 | 0.0520 | 0.0700 |

Large guards helped, but:

- encoded-size overhead became unacceptable;
- improvement stopped being monotonic;
- results plateaued around 160 ms;
- larger windows did not guarantee better seams.

Conclusion:

> Guard context is architecturally correct, but large guards are too expensive and eventually reach diminishing returns.

---

## 7. Post-decode seam repairs with a ±40 ms guard

Several local repairs were tested after decoding and cropping:

- short equal-power crossfades;
- endpoint correction ramps;
- cubic Hermite interpolation;
- local linear prediction.

### Crossfade

Longer crossfades reduced the mathematical seam jump, but changed too much audio.

For a 10 ms crossfade:

```text
maximum discontinuity:     0.02006
average modification RMS:  0.14785
maximum sample change:     0.73967
0–5 ms source error:       0.15085
```

Conclusion:

> Long crossfades smooth the join by substantially rewriting the waveform.

### Endpoint ramp

A 0.5 ms endpoint ramp was extremely conservative:

```text
maximum discontinuity:     0.03156
average modification RMS:  0.00638
maximum sample change:     0.02869
```

It was the least invasive method, but did not improve slope continuity as strongly as Hermite.

### Cubic Hermite

A 0.5 ms Hermite repair gave:

```text
maximum discontinuity: 0.03133
P95 slope jump:        0.00189
```

It preserved both value and slope continuity much better than a simple ramp.

Longer Hermite windows were rejected because they changed too much audio and could overshoot.

### Linear prediction

LPC-style repair did not materially outperform Hermite and was more complex and more invasive.

Conclusion:

> Very short Hermite repair is the best intelligent local seam treatment tested.

---

## 8. Small-guard plus Hermite sweep

The final focused experiment tested:

```text
guard: 0, 5, 10, 20, 30, 40 ms each side
repair: hard, Hermite 0.25 ms, Hermite 0.5 ms
```

Key results:

| Variant | ECDC overhead | Median discontinuity | P95 / max | 0–5 ms source error | Avg repair RMS | Max sample modification |
|---|---:|---:|---:|---:|---:|---:|
| Guard 0 + H 0.5 ms | 0% | 0.00490 | 0.04080 | 0.07173 | 0.01895 | 0.11149 |
| Guard 5 + H 0.5 ms | 0.84% | 0.00479 | 0.02840 | 0.05038 | 0.03885 | 0.21004 |
| Guard 10 + H 0.25 ms | 1.47% | 0.00625 | 0.02518 | 0.04936 | 0.01885 | 0.09721 |
| Guard 10 + H 0.5 ms | 1.47% | 0.00332 | 0.02204 | 0.04976 | 0.02828 | 0.19867 |
| Guard 20 + H 0.5 ms | 2.59% | 0.00631 | 0.04229 | 0.04605 | 0.03189 | 0.30533 |
| Guard 30 + H 0.25 ms | 3.81% | 0.00410 | 0.02041 | 0.04334 | 0.02175 | 0.14155 |
| Guard 40 + H 0.25 ms | 5.04% | 0.00314 | 0.03863 | 0.04267 | 0.01515 | 0.06719 |

### Interpretation

#### No guard

Hermite fixed much of the exact discontinuity, but the wider 0–5 ms codec-edge error remained much higher than interior error.

So repair alone is insufficient.

#### Guard 5 ms

This nearly normalised the seam-region error but sometimes required larger Hermite corrections.

#### Guard 10 ms

This was the efficiency sweet spot:

```text
average ECDC overhead: 1.47%
0–5 ms seam error:     close to interior error
maximum discontinuity: 0.02204 with H 0.5 ms
```

#### Guard 30 ms

This produced the best measured worst seam among the sub-5% options:

```text
maximum discontinuity: 0.02041
ECDC overhead:         3.81%
```

But the gain over guard10 + H 0.5 ms was only about 7.4% in the worst-discontinuity metric, at substantially higher storage cost.

#### Guard 20 and guard 40

These were not consistently better than guard10 or guard30.

That suggests frame alignment, quantisation and convolution geometry matter alongside guard duration.

Conclusion:

> More context is not automatically better. The best current cost-quality balance is ±10 ms plus 0.5 ms Hermite.

---

## Recommended production design

## Codec stage

For logical chunk `r`:

```text
owned_start = r × 64,000
model_start = owned_start - 480
model_end   = owned_start + 64,000 + 480
```

Build a 64,960-sample model input:

```text
480 samples real left source context
64,000 owned samples
480 samples real right source context
```

At the beginning or end of a track, use deterministic padding for unavailable context.

Encode and decode the complete 64,960-sample model window.

Crop:

```text
discard decoded samples 0..479
retain decoded samples 480..64,479
discard decoded samples 64,480..64,959
```

The retained output is exactly 64,000 samples.

## Assembly stage

When adjoining two retained chunks:

1. Preserve the exact timeline boundary.
2. Apply a centred 0.5 ms cubic Hermite repair.
3. At 48 kHz, 0.5 ms is 24 samples total if interpreted as the complete repair span.
4. Use local values and slopes outside the repair region as Hermite endpoints.
5. Replace only the declared short repair region.
6. Do not change output length.
7. Do not overlap or shift later chunks.
8. Constrain or clamp overshoot if necessary.
9. Apply the same deterministic algorithm on every implementation.

The repair depends on both adjacent chunks, so it belongs most naturally in the deterministic chunk-assembly layer rather than inside an isolated single-chunk decoder.

---

## Why this is the best current answer

It separates the two seam problems:

```text
±10 ms source guard
    prevents broad decoder-edge degradation

0.5 ms Hermite repair
    removes the final value and slope mismatch
```

It also preserves all core format requirements:

- one independently encoded ECDC per logical chunk;
- exact 64,000-sample logical ownership;
- exact output timeline;
- no long playback overlap;
- no timeline shortening;
- low average encoded-size increase;
- deterministic assembly;
- random access remains practical.

At approximately 1.47% average ECDC overhead, it is safely below the preferred 5% ceiling.

---

## Conservative alternative

Use:

```text
±30 ms guard
+ 0.25 ms Hermite repair
```

when the lowest measured worst-seam discontinuity is more important than minimising size.

Geometry:

```text
left guard:       1,440 samples
owned chunk:     64,000 samples
right guard:      1,440 samples
model window:    66,880 samples
frame length:       209
average overhead: 3.81%
```

This should remain a validation and fallback profile rather than the default unless blind listening clearly favours it.

---

## Approaches not recommended

### Exact hard join with no context

Rejected because it creates audible codec-induced discontinuities.

### Nominal 10 ms segment-stride overlap-add

Rejected because it sounded worse and did not solve independent chunk boundaries.

### 80–120 ms playback crossfade

Rejected because it changes a large audio region and complicates exact chunk ownership and timing.

### Large 80–240 ms model guards

Rejected because storage overhead is too high and quality gains eventually plateau.

### Long Hermite windows

Rejected because they cause large sample modifications and overshoot.

### LPC repair

Rejected because it is more complex and did not meaningfully outperform Hermite.

### Long equal-power crossfades

Rejected because they reduce seam metrics by heavily rewriting the waveform.

---

## Remaining validation before freezing the format

The current recommendation should still be verified with:

- sustained tones;
- vocals;
- bass-heavy music;
- sharp percussion;
- quiet sections;
- dense mixes;
- both 6 kbps and 12 kbps;
- mono and stereo if both are supported;
- first and final track chunks;
- random-access single-chunk playback;
- forward, reverse and scratch playback;
- clipping and Hermite overshoot checks;
- blind listening to worst-seam clips.

The key blind comparison should be:

```text
guard0 hard
guard10 + Hermite 0.25 ms
guard10 + Hermite 0.5 ms
guard30 + Hermite 0.25 ms
```

---

## Final conclusion

The best current way to adjoin chunked EnCodec audio is:

```text
use a small amount of real neighbouring source context during encoding,
crop back to the exact logical chunk after decoding,
then apply a very short deterministic cubic Hermite interpolation at the join.
```

For the tested 48 kHz, 64,000-sample Bitneedle chunks, the leading profile is:

```text
±10 ms guard context
+ 0.5 ms Hermite seam repair
```

This is currently the best balance of continuity, fidelity, exact timing, independent chunking and encoded size.
