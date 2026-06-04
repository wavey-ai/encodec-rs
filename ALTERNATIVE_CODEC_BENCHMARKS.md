# Alternative Codec Benchmarks

Status as of 2026-06-03.

This note tracks local tests of non-EnCodec alternatives for compressing the
Lori Asha - Westside stereo fixture. The comparison target is the existing
golden `.ecdc` output, not an upstream EnCodec rerun.

## Baseline

Full-track source:

- `/Users/jamie/wavey.ai/bitneedle/.tmp-local/fixtures/lori-asha-westside-source.local.wav`
- 48 kHz stereo PCM WAV
- duration: 208.51 seconds

Current golden `.ecdc` reference:

- `/Users/jamie/wavey.ai/bitneedle/goldenfiles/records/lori-asha-westside-single45-hq/lori-asha-westside-single45-hq.ecdc`
- size: 255,061 bytes
- effective file bitrate: about 9.79 kbps

## DAC

Package tested:

- `descript-audio-codec==1.0.0`
- local venv: `test/codec-benchmark/dac-venv`
- model: DAC `44khz`, `8kbps`
- device: `mps`

Short 4-second smoke test:

| Codec | Artifact | Size | Effective bitrate |
|---|---:|---:|---:|
| DAC 44 kHz 8 kbps | `.dac` | 12,977 bytes | 25.95 kbps |
| DAC 44 kHz 16 kbps | `.dac` | 25,397 bytes | 50.79 kbps |

The decoded DAC WAVs were 48 kHz stereo, so the round trip is format-compatible
with the fixture.

Full-track result:

| Codec | Artifact | Size | Effective bitrate | Encode wall time |
|---|---:|---:|---:|---:|
| golden `.ecdc` | `.ecdc` | 255,061 bytes | 9.79 kbps | n/a |
| DAC 44 kHz 8 kbps | `.dac` | 674,486 bytes | 25.88 kbps | 125.8s |

Decision: DAC is ruled out for this target. The packaged `.dac` artifact is
about 2.64x larger than the current golden `.ecdc` file on the full Westside
track, even before considering quality. Full-track DAC decode was started, then
stopped once the size result made it non-competitive.

## FlowDec / NDAC

Source:

- `https://github.com/facebookresearch/FlowDec`
- checkpoint archive: `test/codec-benchmark/FlowDec/checkpoints.zip`

FlowDec uses an underlying NDAC codec plus a FlowDec postfilter. The compressed
artifact size is determined by the NDAC `.dac` codes; the postfilter can improve
decoded quality but does not reduce file size.

Local changes made only inside the benchmark checkout:

- Patched FlowDec's CUDA-only NCSN++ fused op imports to fall back to the existing
  pure PyTorch path when CUDA is unavailable.
- Added `test/codec-benchmark/run_flowdec_postfilter.py` for inference-only
  loading with eval/training-only imports stubbed.

Short 4-second smoke test:

| Codec | Artifact | Size | Effective bitrate | Notes |
|---|---:|---:|---:|---|
| NDAC-75 `nq=10` | `.dac` | 12,557 bytes | 25.11 kbps | stereo, 48 kHz |
| NDAC-75 `nq=4` | `.dac` | 5,357 bytes | 10.71 kbps | stereo, 48 kHz |

Full-track result:

| Codec | Artifact | Size | Effective bitrate | Encode wall time |
|---|---:|---:|---:|---:|
| golden `.ecdc` | `.ecdc` | 255,061 bytes | 9.79 kbps | n/a |
| NDAC-75 `nq=4` | `.dac` | 276,278 bytes | 10.60 kbps | 95.3s |

Decoded NDAC-75 `nq=4` output is valid 48 kHz stereo. Objective metrics against
the source were weak before the FlowDec postfilter: SNR 2.88 dB, SI-SDR 5.52 dB,
log spectral distance 10.59 dB, spectral convergence 0.559.

The FlowDec postfilter was attempted on M1/MPS after patching the CUDA extension
imports and casting the model to float32. It was still running after more than
one minute for a 4-second stereo clip and was interrupted. That makes FlowDec a
poor local M1 candidate unless we invest in a cleaner CPU/MPS inference path or
run it on an NVIDIA CUDA machine.

Decision: keep FlowDec/NDAC as "near miss, not next". The stock NDAC artifact is
only 8.3% larger than `.ecdc`, so custom bit-packing/entropy coding could make
it size-competitive, but the postfilter path is too heavy locally and the raw
NDAC decode quality is not enough by itself.

## MOSS-Audio-Tokenizer-Nano

Source:

- model: `https://huggingface.co/OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano`
- docs: `https://github.com/OpenMOSS/MOSS-TTS-Nano`
- benchmark runner: `test/codec-benchmark/run_moss_nano_benchmark.py`
- helper deps: `test/codec-benchmark/requirements-moss.txt`
  (`transformers==5.9.0`, `safetensors==0.7.0`, `huggingface_hub==1.17.0`)

MOSS-Audio-Tokenizer-Nano is a 48 kHz stereo tokenizer/codec with a decoder. The
official MOSS-TTS-Nano README says Nano supports 48 kHz input/output, stereo
audio, a 12.5 Hz token stream, RVQ with 16 codebooks, and variable bitrates from
0.125 kbps to 2 kbps. The local benchmark uses the Hugging Face PyTorch model
with `trust_remote_code=True`, streaming chunk duration `0.08`, and CPU.

MOSS does not currently expose a standard compressed file container, so the
benchmark writes a local `.mossnano` packed-token artifact for size accounting:
8-byte magic, 24 bytes of metadata, then 10-bit packed RVQ codes. This is not an
interchange format; it is a minimal artifact to estimate payload size from the
actual model tokens.

Short 4-second Westside smoke test:

| Codec | Artifact | Size | Effective bitrate | SNR | SI-SDR | LSD | Spectral convergence |
|---|---:|---:|---:|---:|---:|---:|---:|
| MOSS Nano RVQ6 | `.mossnano` | 407 bytes | 0.814 kbps | 1.62 dB | -2.36 dB | 10.28 dB | 0.514 |
| MOSS Nano RVQ8 | `.mossnano` | 532 bytes | 1.064 kbps | 2.22 dB | -1.14 dB | 10.13 dB | 0.463 |
| MOSS Nano RVQ12 | `.mossnano` | 782 bytes | 1.564 kbps | 3.06 dB | 0.45 dB | 10.21 dB | 0.425 |
| MOSS Nano RVQ16 | `.mossnano` | 1,032 bytes | 2.064 kbps | 3.63 dB | 1.43 dB | 10.16 dB | 0.388 |

All decoded short-test WAVs were valid 48 kHz stereo. Outputs live under
`test/codec-benchmark/moss-nano-westside-4s/`.

Full-track result:

| Codec | Artifact | Size | Effective bitrate | Encode wall time | Decode wall time | SNR | SI-SDR |
|---|---:|---:|---:|---:|---:|---:|---:|
| golden `.ecdc` | `.ecdc` | 255,061 bytes | 9.79 kbps | n/a | n/a | n/a | n/a |
| MOSS Nano RVQ16 | `.mossnano` | 52,152 bytes | 2.00 kbps | 114.5s | 171.9s | 4.18 dB | 2.37 dB |

Full-track RVQ16 additional metrics: log spectral distance 10.81 dB, spectral
convergence 0.380. Output lives under
`test/codec-benchmark/full-moss-nano-rvq16-westside/`.

Decision: MOSS Nano is playable and much smaller than `.ecdc`, but it is not a
good next replacement candidate for the high-fidelity music target. The 2 kbps
ceiling is extremely aggressive for stereo music-with-vocals, and the objective
metrics on Westside are weak. Keep the artifacts for listening/reference, but
prioritize higher-bitrate music codecs next.

## SpectroStream

Source:

- paper: `https://arxiv.org/abs/2508.05207`

Availability check:

- The paper describes a full-band multi-channel neural codec for 48 kHz stereo
  music at 4-16 kbps.
- No public encoder/decoder repository, pretrained checkpoint, Hugging Face
  model, PyPI package, or runnable demo was found.
- The arXiv page exposes the paper and TeX source links, but no implementation
  artifact suitable for local benchmarking.

Decision: SpectroStream is worth tracking because its published target range is
closer to the `.ecdc` fidelity/bitrate region than MOSS Nano, but it cannot be
tested locally right now. Move it out of the runnable benchmark queue until code
and weights are released.

## TQCodec

Source:

- paper: `https://arxiv.org/abs/2603.01592`
- arXiv source archive: `test/codec-benchmark/tqcodec/tqcodec-src.tar.gz`

Availability check:

- GitHub repository search for `TQCodec audio codec`: no results.
- Hugging Face model search for `TQCodec`: no results.
- PyPI package search for `tqcodec`: no matching distribution.
- arXiv source contains only LaTeX/images; no code, checkpoint, demo, or release
  URL.

Useful paper details:

- Target: high-fidelity music streaming, not ultra-low bitrate.
- Sample rate: 44.1 kHz.
- Published target bitrates: 32, 64, and 128 kbps.
- Stereo handling: the paper says the model is applied twice for two-channel
  audio because a two-channel neural network did not work well empirically.
- Training recipe: 400k iterations, batch size 32, `8 x H20`, about 30 hours.
- Reported subjective comparison: AICodec-64kbps vs Ogg-96kbps, average MOS
  4.18, with preference split roughly evenly between Ogg and AICodec.

Decision: TQCodec cannot be tested locally right now because there is no public
encoder/decoder implementation or pretrained weights. It remains interesting on
paper for the quality-first frontier, but it should be moved out of the active
benchmark queue until code or weights are released.

## Shortlist

Next candidates should be filtered by these requirements before running:

- Must support stereo round-trip, or be adaptable to stereo without doubling
  the file size beyond the `.ecdc` baseline.
- Must produce an actual compressed artifact, not only model tokens or a latent
  tensor for a downstream generator.
- Must be quality-gated first. The goal is not the lowest possible bitrate; it is
  the smallest file that still gives excellent stereo music-with-vocals quality.
- Must be tested on the full Westside fixture, because short clips overstate
  container/header overhead.

Ranked candidate notes:

| Candidate | Status |
|---|---|
| MOSS-Audio-Tokenizer-Nano | Tested locally. Playable 48 kHz stereo and very small, but RVQ16 tops out near 2 kbps and objective Westside metrics are too weak for the high-fidelity music target. |
| SpectroStream | Strong paper target for 48 kHz stereo at 4-16 kbps, but public testing is blocked until an encoder/decoder implementation and weights are released. |
| APCodec | Strong paper claim: 48 kHz at 6 kbps and better than EnCodec/DAC at the same bitrate. Needs code/weights availability check before local testing; if available, test at the highest quality mode first, not only the smallest mode. |
| DisCodec | Music-specific 44.1 kHz codec with code/checkpoints claimed. Need check whether the public repo exposes encoder/decoder bitstreams rather than a vocoder-only workflow. |
| Opus 1.6 | Non-neural quality/size sanity baseline. It is open, fullband, stereo, and fast; test a sweep such as 24, 32, 48, and 64 kbps to establish the conventional-codec quality bar. |
| TSAC | Lower priority after the quality-first clarification. It is interesting for very small stereo files, but its default target is likely too aggressive unless the subjective quality is surprisingly good. |
| WavTokenizer | Easy open-source tokenizer with music models, but released configs are 24 kHz and likely mono. Useful only if stereo can be packed efficiently or quality is much better than expected. |
| TQCodec | Interesting on paper for high-fidelity music streaming, but blocked: no public code or weights found. |
| SNAC / Mimi / speech tokenizers | Lower priority for this requirement because public models tend to be mono/24 kHz and optimized for speech or generative tokenization, not stereo music compression. |

## Local Artifacts

Generated local outputs live under:

- `test/codec-benchmark/`

That directory includes the temporary DAC venv, `.dac` files, decoded smoke WAVs,
and a small `codec_metrics.py` helper. These artifacts are not source-of-truth
goldens.
