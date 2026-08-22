#pragma once

#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef int (*EncodecRsMlxEncodeFrameFn)(
    void *user_data,
    const float *audio,
    size_t batch,
    size_t channels,
    size_t samples,
    long long *codes_out,
    size_t codes_len,
    float *scales_out,
    size_t scales_len
);

typedef int (*EncodecRsMlxDecodeFrameFn)(
    void *user_data,
    const long long *codes,
    size_t batch,
    size_t codebooks,
    size_t frames,
    const float *scales,
    size_t scales_len,
    float *audio_out,
    size_t audio_len
);

typedef struct EncodecRsMlxFrameCallbacks {
    void *user_data;
    EncodecRsMlxEncodeFrameFn encode_frame;
    EncodecRsMlxDecodeFrameFn decode_frame;
} EncodecRsMlxFrameCallbacks;

typedef struct EncodecRsMlxByteResult {
    bool ok;
    unsigned char *ptr;
    size_t len;
    char *error;
} EncodecRsMlxByteResult;

typedef struct EncodecRsMlxAudioResult {
    bool ok;
    float *ptr;
    size_t len;
    size_t channels;
    size_t samples;
    char *error;
} EncodecRsMlxAudioResult;

EncodecRsMlxByteResult encodec_rs_mlx_encode_ecdc(
    const char *bundle_dir,
    const float *audio,
    size_t channels,
    size_t samples,
    bool use_lm,
    size_t frame_batch_size,
    bool chunk_crc,
    double chunk_ms,
    bool has_chunk_ms,
    EncodecRsMlxFrameCallbacks callbacks
);

EncodecRsMlxByteResult encodec_rs_mlx_encode_ecdc_stream_to_path(
    const char *bundle_dir,
    const float *audio,
    size_t channels,
    size_t samples,
    bool use_lm,
    size_t frame_batch_size,
    bool chunk_crc,
    double chunk_ms,
    bool has_chunk_ms,
    const char *output_path,
    const char *progress_path,
    EncodecRsMlxFrameCallbacks callbacks
);

EncodecRsMlxAudioResult encodec_rs_mlx_decode_ecdc(
    const char *bundle_dir,
    const unsigned char *payload,
    size_t payload_len,
    EncodecRsMlxFrameCallbacks callbacks
);

void encodec_rs_mlx_free_string(char *value);
void encodec_rs_mlx_free_bytes(unsigned char *ptr, size_t len);
void encodec_rs_mlx_free_audio(float *ptr, size_t len);

#ifdef __cplusplus
}
#endif
