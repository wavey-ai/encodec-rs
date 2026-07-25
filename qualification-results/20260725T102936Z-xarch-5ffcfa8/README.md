# Cross-Architecture Qualification

This directory contains the local and GCP results for source commit
`5ffcfa864cd3476b90de16d2aa3360b0775e03c9`.

The run status is `pass`.

## Test Environments

The local environment used macOS on Apple ARM64.

The GCP environments used Linux on x86-64. Three `n2-highmem-4` workers ran
in separate European regions. Each worker used ONNX Runtime 1.25.1 on CPU.

The GCP project had no GPU quota. Thus, this run contains no GPU result.

## Entropy Result

The local environment and each GCP worker tested all four model bundles.

All CDF sequence hashes match exactly. All entropy and payload hashes match
exactly. Each decoder recovered the input codes exactly.

The comparison completed 384 exact field checks. It found no mismatch.

## Full-Song Result

The GCP workers processed six complete WAV files. Each file used these four
configurations:

- 6 kbps with 1333 ms chunks
- 6 kbps with 1800 ms chunks
- 12 kbps with 1333 ms chunks
- 12 kbps with 1800 ms chunks

This matrix contains 24 full encode and decode cases. All 24 cases passed.

Each case tested real chunk construction, context windows, batched chunk
processing, the final partial chunk, and reconstructed seams.

The GCP results match the local ARM64 quality baseline. The maximum scalar
metric difference is 0.00008021. The comparison limit is 0.001.

The maximum container size difference is seven bytes. The maximum relative
difference is 0.002466%. The comparison limit is 0.1%.

The SNR range is 5.2740 dB to 11.3331 dB. The SI-SDR range is 4.0231 dB
to 11.0446 dB.

ONNX neural inference can select different codes after small CPU arithmetic
differences. Thus, complete song containers do not require identical hashes.
The decoded quality metrics must stay within the defined limits.

## Corpus

The GCP workers downloaded the supplied Dropbox ZIP. The corpus staging step
found all six locked tracks.

The `I WANT HER` WAV has different non-audio header data from the locked local
file. Its PCM `data` chunk is identical. The PCM chunk contains 42,624,000
bytes and has this SHA-256 value:

`47b57614ea47b090469fee7705eb5892201ccb2c5727d9018a4b247e4dbc4532`

## Contents

- `analysis/comparison.json` contains the cross-architecture comparison.
- `corpus-manifest.json` identifies the six input files.
- `local/` contains the fresh ARM64 evidence and confirmation result.
- `gcp/` contains all results and logs from the three x86-64 workers.
- `files.sha256` contains a checksum for each result file.
