use std::ffi::{c_char, c_void, CStr, CString};
use std::fs::File;
use std::io::{Seek, SeekFrom, Write};
use std::os::raw::c_double;
use std::path::{Path, PathBuf};
use std::ptr;
use std::slice;

use anyhow::{bail, Context, Result};
use ndarray::{s, Array2, Array3, ArrayView3, ShapeBuilder};

use crate::ecdc::{
    decode_ecdc_model_window_views_with_batch_size_and_progress, decode_ecdc_with_batch_size,
    decode_ecdc_with_batch_size_and_progress, encode_audio_view_to_ecdc_stream_with_options,
    encode_audio_view_to_ecdc_with_options, EcdcDecodeProgressStage, FrameCodec,
};
use crate::ecdc::{encode_audio_view_to_dual_ecdc_with_options, DualEcdcEncodeResult};
use crate::ecdc_presets::fixed_context_samples;
use crate::format::{ecdc_chunk_layout_from_ms, segment_frame_length, segment_starts};
use crate::metadata::OnnxFrameBundleMetadata;
use crate::portable_lm::PortableLmCodec;
use crate::portable_lm::PortablePairedLmCodec;

fn planar_f32le_byte_len(channels: usize, samples: usize) -> Result<usize> {
    channels
        .checked_mul(samples)
        .and_then(|value| value.checked_mul(std::mem::size_of::<f32>()))
        .ok_or_else(|| anyhow::anyhow!("decoded planar output size overflow"))
}

fn write_owned_planar_f32le<W: Write + Seek>(
    output: &mut W,
    window: ArrayView3<'_, f32>,
    channels: usize,
    audio_length: usize,
    context: usize,
    output_offset: usize,
    owned_samples: usize,
) -> Result<()> {
    if window.shape()[0] != 1 || window.shape()[1] != channels {
        bail!(
            "decoded window shape mismatch, expected [1, {}, samples], got {:?}",
            channels,
            window.shape(),
        );
    }
    let source_end = context
        .checked_add(owned_samples)
        .ok_or_else(|| anyhow::anyhow!("decoded sample range overflow"))?;
    if source_end > window.shape()[2] {
        bail!(
            "decoded window is too short: context={} owned={} decoded={}",
            context,
            owned_samples,
            window.shape()[2],
        );
    }
    let output_end = output_offset
        .checked_add(owned_samples)
        .ok_or_else(|| anyhow::anyhow!("decoded output range overflow"))?;
    if output_end > audio_length {
        bail!(
            "decoded output range exceeds audio length: offset={} owned={} length={}",
            output_offset,
            owned_samples,
            audio_length,
        );
    }

    for channel in 0..channels {
        let sample_offset = channel
            .checked_mul(audio_length)
            .and_then(|value| value.checked_add(output_offset))
            .ok_or_else(|| anyhow::anyhow!("decoded planar offset overflow"))?;
        let byte_offset = sample_offset
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| anyhow::anyhow!("decoded byte offset overflow"))?;
        output.seek(SeekFrom::Start(u64::try_from(byte_offset)?))?;

        let owned = window.slice(s![0, channel, context..source_end]);
        let samples = owned
            .as_slice()
            .context("decoded channel samples are not contiguous")?;
        #[cfg(target_endian = "little")]
        {
            let bytes = unsafe {
                slice::from_raw_parts(
                    samples.as_ptr().cast::<u8>(),
                    samples.len() * std::mem::size_of::<f32>(),
                )
            };
            output.write_all(bytes)?;
        }
        #[cfg(target_endian = "big")]
        {
            for sample in samples {
                output.write_all(&sample.to_le_bytes())?;
            }
        }
    }
    Ok(())
}

fn copy_owned_interleaved(
    output: &mut [f32],
    window: ArrayView3<'_, f32>,
    channels: usize,
    audio_length: usize,
    context: usize,
    output_offset: usize,
    owned_samples: usize,
) -> Result<()> {
    if window.shape()[0] != 1 || window.shape()[1] != channels {
        bail!(
            "decoded window shape mismatch, expected [1, {}, samples], got {:?}",
            channels,
            window.shape(),
        );
    }
    let expected_len = channels
        .checked_mul(audio_length)
        .ok_or_else(|| anyhow::anyhow!("decoded interleaved output size overflow"))?;
    if output.len() != expected_len {
        bail!(
            "decoded interleaved output length mismatch, expected {}, got {}",
            expected_len,
            output.len(),
        );
    }
    let source_end = context
        .checked_add(owned_samples)
        .ok_or_else(|| anyhow::anyhow!("decoded sample range overflow"))?;
    if source_end > window.shape()[2] {
        bail!(
            "decoded window is too short: context={} owned={} decoded={}",
            context,
            owned_samples,
            window.shape()[2],
        );
    }
    let output_end = output_offset
        .checked_add(owned_samples)
        .ok_or_else(|| anyhow::anyhow!("decoded output range overflow"))?;
    if output_end > audio_length {
        bail!(
            "decoded output range exceeds audio length: offset={} owned={} length={}",
            output_offset,
            owned_samples,
            audio_length,
        );
    }

    if channels == 2 {
        let left_view = window.slice(s![0, 0, context..source_end]);
        let left = left_view
            .as_slice()
            .context("decoded left-channel samples are not contiguous")?;
        let right_view = window.slice(s![0, 1, context..source_end]);
        let right = right_view
            .as_slice()
            .context("decoded right-channel samples are not contiguous")?;
        let destination = &mut output[output_offset * 2..output_end * 2];
        for ((frame, left), right) in destination
            .chunks_exact_mut(2)
            .zip(left.iter().copied())
            .zip(right.iter().copied())
        {
            frame[0] = left;
            frame[1] = right;
        }
        return Ok(());
    }

    for sample in 0..owned_samples {
        let destination_base = (output_offset + sample) * channels;
        for channel in 0..channels {
            output[destination_base + channel] = window[[0, channel, context + sample]];
        }
    }
    Ok(())
}

pub type EncodecRsMlxEncodeFrameFn = unsafe extern "C" fn(
    user_data: *mut c_void,
    audio: *const f32,
    batch: usize,
    channels: usize,
    samples: usize,
    codes_out: *mut i64,
    codes_len: usize,
    scales_out: *mut f32,
    scales_len: usize,
) -> i32;

pub type EncodecRsMlxDecodeFrameFn = unsafe extern "C" fn(
    user_data: *mut c_void,
    codes: *const i64,
    batch: usize,
    codebooks: usize,
    frames: usize,
    scales: *const f32,
    scales_len: usize,
    audio_out: *mut f32,
    audio_len: usize,
) -> i32;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct EncodecRsMlxFrameCallbacks {
    pub user_data: *mut c_void,
    pub encode_frame: Option<EncodecRsMlxEncodeFrameFn>,
    pub decode_frame: Option<EncodecRsMlxDecodeFrameFn>,
}

#[repr(C)]
pub struct EncodecRsMlxByteResult {
    pub ok: bool,
    pub ptr: *mut u8,
    pub len: usize,
    pub error: *mut c_char,
}

#[repr(C)]
pub struct EncodecRsMlxDualByteResult {
    pub ok: bool,
    pub primary_ptr: *mut u8,
    pub primary_len: usize,
    pub derived_ptr: *mut u8,
    pub derived_len: usize,
    pub error: *mut c_char,
}

#[repr(C)]
pub struct EncodecRsMlxAudioResult {
    pub ok: bool,
    pub ptr: *mut f32,
    pub len: usize,
    pub channels: usize,
    pub samples: usize,
    pub error: *mut c_char,
}

struct CallbackFrameCodec {
    metadata: OnnxFrameBundleMetadata,
    callbacks: EncodecRsMlxFrameCallbacks,
}

impl CallbackFrameCodec {
    fn from_bundle_dir(
        bundle_dir: impl AsRef<Path>,
        callbacks: EncodecRsMlxFrameCallbacks,
    ) -> Result<Self> {
        let metadata_path = bundle_dir.as_ref().join("bundle.json");
        let metadata: OnnxFrameBundleMetadata = serde_json::from_str(
            &std::fs::read_to_string(&metadata_path)
                .with_context(|| format!("failed to read {}", metadata_path.display()))?,
        )
        .with_context(|| format!("failed to parse {}", metadata_path.display()))?;
        Ok(Self {
            metadata,
            callbacks,
        })
    }
}

impl FrameCodec for CallbackFrameCodec {
    fn metadata(&self) -> &OnnxFrameBundleMetadata {
        &self.metadata
    }

    fn encode_frame(&mut self, audio: &Array3<f32>) -> Result<(Array3<i64>, Array2<f32>)> {
        let Some(callback) = self.callbacks.encode_frame else {
            bail!("MLX frame encode callback is not installed");
        };
        let shape = audio.shape();
        if shape.len() != 3 {
            bail!("MLX encode callback expected [batch, channels, samples] audio");
        }
        let batch = shape[0];
        let channels = shape[1];
        let samples = shape[2];
        let audio = audio
            .as_slice_memory_order()
            .context("MLX encode audio batch is not contiguous")?;
        let frame_length = segment_frame_length(
            samples,
            self.metadata.segment_samples,
            self.metadata.frame_length,
        );
        let mut codes = vec![0_i64; batch * self.metadata.num_codebooks * frame_length];
        let mut scales = vec![1.0_f32; batch];
        let status = unsafe {
            callback(
                self.callbacks.user_data,
                audio.as_ptr(),
                batch,
                channels,
                samples,
                codes.as_mut_ptr(),
                codes.len(),
                scales.as_mut_ptr(),
                scales.len(),
            )
        };
        if status != 0 {
            bail!("MLX frame encode callback failed with status {status}");
        }
        let codes =
            Array3::from_shape_vec((batch, self.metadata.num_codebooks, frame_length), codes)?;
        let scales = Array2::from_shape_vec((batch, 1), scales)?;
        Ok((codes, scales))
    }

    fn decode_frame(&mut self, codes: &Array3<i64>, scale: &Array2<f32>) -> Result<Array3<f32>> {
        let Some(callback) = self.callbacks.decode_frame else {
            bail!("MLX frame decode callback is not installed");
        };
        let shape = codes.shape();
        if shape.len() != 3 {
            bail!("MLX decode callback expected [batch, codebooks, frames] codes");
        }
        let batch = shape[0];
        let codebooks = shape[1];
        let frames = shape[2];
        let codes = codes
            .as_slice_memory_order()
            .context("MLX decode code batch is not contiguous")?;
        let scales = scale
            .as_slice_memory_order()
            .context("MLX decode scale batch is not contiguous")?;
        let decoded_samples = frames
            .saturating_mul(self.metadata.segment_samples)
            .div_ceil(self.metadata.frame_length);
        let mut audio = vec![0.0_f32; batch * self.metadata.channels * decoded_samples];
        let status = unsafe {
            callback(
                self.callbacks.user_data,
                codes.as_ptr(),
                batch,
                codebooks,
                frames,
                scales.as_ptr(),
                scales.len(),
                audio.as_mut_ptr(),
                audio.len(),
            )
        };
        if status != 0 {
            bail!("MLX frame decode callback failed with status {status}");
        }
        Ok(Array3::from_shape_vec(
            (batch, self.metadata.channels, decoded_samples),
            audio,
        )?)
    }
}

unsafe fn bundle_dir_from_c(bundle_dir: *const c_char) -> Result<PathBuf> {
    if bundle_dir.is_null() {
        bail!("bundle_dir pointer is null");
    }
    let value = CStr::from_ptr(bundle_dir)
        .to_str()
        .context("bundle_dir is not valid UTF-8")?;
    if value.is_empty() {
        bail!("bundle_dir is empty");
    }
    Ok(PathBuf::from(value))
}

unsafe fn planar_audio_view<'a>(
    audio: *const f32,
    channels: usize,
    samples: usize,
) -> Result<ArrayView3<'a, f32>> {
    let len = channels.saturating_mul(samples);
    if audio.is_null() && len > 0 {
        bail!("audio pointer is null");
    }
    if len == 0 {
        return Ok(ArrayView3::from_shape((1, channels, samples), &[])?);
    }
    Ok(ArrayView3::from_shape_ptr((1, channels, samples), audio))
}

unsafe fn interleaved_audio_view<'a>(
    audio: *const f32,
    channels: usize,
    samples: usize,
) -> Result<ArrayView3<'a, f32>> {
    let len = channels.saturating_mul(samples);
    if audio.is_null() && len > 0 {
        bail!("audio pointer is null");
    }
    if len == 0 {
        return Ok(ArrayView3::from_shape((1, channels, samples), &[])?);
    }
    Ok(ArrayView3::from_shape_ptr(
        (1, channels, samples).strides((len, 1, channels)),
        audio,
    ))
}

#[allow(clippy::too_many_arguments)]
fn encode_audio_view(
    bundle_dir: &Path,
    audio: ArrayView3<'_, f32>,
    use_lm: bool,
    frame_batch_size: usize,
    chunk_crc: bool,
    chunk_ms: Option<f64>,
    callbacks: EncodecRsMlxFrameCallbacks,
) -> Result<Vec<u8>> {
    let mut codec = CallbackFrameCodec::from_bundle_dir(bundle_dir, callbacks)?;
    if audio.shape()[1] != codec.metadata.channels {
        bail!(
            "audio channel count {} does not match bundle {}",
            audio.shape()[1],
            codec.metadata.channels
        );
    }
    if !use_lm {
        bail!("use_lm=false is unsupported for q8 ECDC payloads in this build");
    }
    let mut lm_codec = PortableLmCodec::from_dir(bundle_dir)?;
    encode_audio_view_to_ecdc_with_options(
        &mut codec,
        &mut lm_codec,
        audio,
        None,
        frame_batch_size.max(1),
        chunk_crc,
        chunk_ms,
    )
}

#[allow(clippy::too_many_arguments)]
fn encode_audio_view_dual(
    primary_bundle_dir: &Path,
    derived_bundle_dir: &Path,
    audio: ArrayView3<'_, f32>,
    use_lm: bool,
    frame_batch_size: usize,
    chunk_crc: bool,
    chunk_ms: Option<f64>,
    callbacks: EncodecRsMlxFrameCallbacks,
) -> Result<DualEcdcEncodeResult> {
    let mut codec = CallbackFrameCodec::from_bundle_dir(primary_bundle_dir, callbacks)?;
    if audio.shape()[1] != codec.metadata.channels {
        bail!(
            "audio channel count {} does not match bundle {}",
            audio.shape()[1],
            codec.metadata.channels
        );
    }
    if !use_lm {
        bail!("use_lm=false is unsupported for q8 ECDC payloads in this build");
    }
    let mut lm = PortablePairedLmCodec::from_dirs(primary_bundle_dir, derived_bundle_dir)?;
    encode_audio_view_to_dual_ecdc_with_options(
        &mut codec,
        &mut lm,
        audio,
        None,
        frame_batch_size.max(1),
        chunk_crc,
        chunk_ms,
    )
}

#[allow(clippy::too_many_arguments)]
fn encode_audio_view_to_path(
    bundle_dir: &Path,
    audio: ArrayView3<'_, f32>,
    use_lm: bool,
    frame_batch_size: usize,
    chunk_crc: bool,
    chunk_ms: Option<f64>,
    output_path: &Path,
    progress_path: Option<&Path>,
    callbacks: EncodecRsMlxFrameCallbacks,
) -> Result<usize> {
    let mut codec = CallbackFrameCodec::from_bundle_dir(bundle_dir, callbacks)?;
    if audio.shape()[1] != codec.metadata.channels {
        bail!(
            "audio channel count {} does not match bundle {}",
            audio.shape()[1],
            codec.metadata.channels
        );
    }
    if !use_lm {
        bail!("use_lm=false is unsupported for q8 ECDC payloads in this build");
    }
    let chunk_layout = ecdc_chunk_layout_from_ms(&codec.metadata, chunk_ms)?;
    let total_chunks = segment_starts(audio.shape()[2], chunk_layout.stride).len();
    let mut lm_codec = PortableLmCodec::from_dir(bundle_dir)?;
    let mut output = File::create(output_path)
        .with_context(|| format!("failed to create {}", output_path.display()))?;
    let mut bytes_written = 0_usize;
    let mut emissions = 0_usize;
    encode_audio_view_to_ecdc_stream_with_options(
        &mut codec,
        &mut lm_codec,
        audio,
        None,
        frame_batch_size.max(1),
        chunk_crc,
        chunk_ms,
        |bytes| {
            output.write_all(bytes)?;
            bytes_written += bytes.len();
            emissions += 1;
            if let Some(progress_path) = progress_path {
                let completed = emissions.saturating_sub(1).min(total_chunks);
                let progress = format!(
                    "{{\"stage\":\"encode\",\"completed\":{completed},\"total\":{total_chunks},\"bytes_written\":{bytes_written},\"emissions\":{emissions}}}\n"
                );
                std::fs::write(progress_path, progress)?;
            }
            Ok(())
        },
    )?;
    output.flush()?;
    Ok(bytes_written)
}

fn c_error(error: impl std::fmt::Display) -> *mut c_char {
    CString::new(error.to_string())
        .map(CString::into_raw)
        .unwrap_or(ptr::null_mut())
}

fn byte_success(bytes: Vec<u8>) -> EncodecRsMlxByteResult {
    let mut bytes = bytes.into_boxed_slice();
    let len = bytes.len();
    let ptr = bytes.as_mut_ptr();
    std::mem::forget(bytes);
    EncodecRsMlxByteResult {
        ok: true,
        ptr,
        len,
        error: ptr::null_mut(),
    }
}

fn byte_count_success(len: usize) -> EncodecRsMlxByteResult {
    EncodecRsMlxByteResult {
        ok: true,
        ptr: ptr::null_mut(),
        len,
        error: ptr::null_mut(),
    }
}

fn byte_error(error: impl std::fmt::Display) -> EncodecRsMlxByteResult {
    EncodecRsMlxByteResult {
        ok: false,
        ptr: ptr::null_mut(),
        len: 0,
        error: c_error(error),
    }
}

fn leak_bytes(bytes: Vec<u8>) -> (*mut u8, usize) {
    let mut bytes = bytes.into_boxed_slice();
    let len = bytes.len();
    let ptr = bytes.as_mut_ptr();
    std::mem::forget(bytes);
    (ptr, len)
}

fn dual_byte_success(result: DualEcdcEncodeResult) -> EncodecRsMlxDualByteResult {
    let (primary_ptr, primary_len) = leak_bytes(result.primary);
    let (derived_ptr, derived_len) = leak_bytes(result.derived);
    EncodecRsMlxDualByteResult {
        ok: true,
        primary_ptr,
        primary_len,
        derived_ptr,
        derived_len,
        error: ptr::null_mut(),
    }
}

fn dual_byte_error(error: impl std::fmt::Display) -> EncodecRsMlxDualByteResult {
    EncodecRsMlxDualByteResult {
        ok: false,
        primary_ptr: ptr::null_mut(),
        primary_len: 0,
        derived_ptr: ptr::null_mut(),
        derived_len: 0,
        error: c_error(error),
    }
}

fn audio_success(audio: Array3<f32>) -> EncodecRsMlxAudioResult {
    let shape = audio.shape().to_vec();
    let (data, offset) = audio.into_raw_vec_and_offset();
    debug_assert_eq!(offset, Some(0));
    let mut data = data.into_boxed_slice();
    let len = data.len();
    let ptr = data.as_mut_ptr();
    std::mem::forget(data);
    EncodecRsMlxAudioResult {
        ok: true,
        ptr,
        len,
        channels: shape[1],
        samples: shape[2],
        error: ptr::null_mut(),
    }
}

fn audio_samples_success(
    samples: Vec<f32>,
    channels: usize,
    frame_count: usize,
) -> EncodecRsMlxAudioResult {
    debug_assert_eq!(samples.len(), channels.saturating_mul(frame_count));
    let mut samples = samples.into_boxed_slice();
    let len = samples.len();
    let ptr = samples.as_mut_ptr();
    std::mem::forget(samples);
    EncodecRsMlxAudioResult {
        ok: true,
        ptr,
        len,
        channels,
        samples: frame_count,
        error: ptr::null_mut(),
    }
}

fn audio_error(error: impl std::fmt::Display) -> EncodecRsMlxAudioResult {
    EncodecRsMlxAudioResult {
        ok: false,
        ptr: ptr::null_mut(),
        len: 0,
        channels: 0,
        samples: 0,
        error: c_error(error),
    }
}

/// Releases a string that an MLX bridge function returned.
///
/// # Safety
///
/// `value` must be null or a live string pointer from this library.
#[no_mangle]
pub unsafe extern "C" fn encodec_rs_mlx_free_string(value: *mut c_char) {
    if value.is_null() {
        return;
    }
    drop(CString::from_raw(value));
}

/// Releases a byte buffer that an MLX bridge function returned.
///
/// # Safety
///
/// `ptr` and `len` must identify one live byte buffer from this library.
#[no_mangle]
pub unsafe extern "C" fn encodec_rs_mlx_free_bytes(ptr: *mut u8, len: usize) {
    if ptr.is_null() {
        return;
    }
    drop(Box::from_raw(ptr::slice_from_raw_parts_mut(ptr, len)));
}

/// Releases an audio buffer that an MLX bridge function returned.
///
/// # Safety
///
/// `ptr` and `len` must identify one live audio buffer from this library.
#[no_mangle]
pub unsafe extern "C" fn encodec_rs_mlx_free_audio(ptr: *mut f32, len: usize) {
    if ptr.is_null() {
        return;
    }
    drop(Box::from_raw(ptr::slice_from_raw_parts_mut(ptr, len)));
}

/// Encodes planar audio through caller-supplied MLX frame callbacks.
///
/// # Safety
///
/// C string pointers must be valid and terminated.
/// `audio` must contain `channels * samples` readable values.
/// The callback pointers must remain valid during this call.
#[no_mangle]
pub unsafe extern "C" fn encodec_rs_mlx_encode_ecdc(
    bundle_dir: *const c_char,
    audio: *const f32,
    channels: usize,
    samples: usize,
    use_lm: bool,
    frame_batch_size: usize,
    chunk_crc: bool,
    chunk_ms: c_double,
    has_chunk_ms: bool,
    callbacks: EncodecRsMlxFrameCallbacks,
) -> EncodecRsMlxByteResult {
    let result = (|| -> Result<Vec<u8>> {
        let bundle_dir = bundle_dir_from_c(bundle_dir)?;
        let audio = planar_audio_view(audio, channels, samples)?;
        encode_audio_view(
            &bundle_dir,
            audio,
            use_lm,
            frame_batch_size,
            chunk_crc,
            has_chunk_ms.then_some(chunk_ms),
            callbacks,
        )
    })();

    match result {
        Ok(bytes) => byte_success(bytes),
        Err(error) => byte_error(error),
    }
}

/// Encodes interleaved audio through caller-supplied MLX frame callbacks.
///
/// # Safety
///
/// C string pointers must be valid and terminated.
/// `audio` must contain `channels * samples` readable values.
/// The callback pointers must remain valid during this call.
#[no_mangle]
pub unsafe extern "C" fn encodec_rs_mlx_encode_ecdc_interleaved(
    bundle_dir: *const c_char,
    audio: *const f32,
    channels: usize,
    samples: usize,
    use_lm: bool,
    frame_batch_size: usize,
    chunk_crc: bool,
    chunk_ms: c_double,
    has_chunk_ms: bool,
    callbacks: EncodecRsMlxFrameCallbacks,
) -> EncodecRsMlxByteResult {
    let result = (|| -> Result<Vec<u8>> {
        let bundle_dir = bundle_dir_from_c(bundle_dir)?;
        let audio = interleaved_audio_view(audio, channels, samples)?;
        encode_audio_view(
            &bundle_dir,
            audio,
            use_lm,
            frame_batch_size,
            chunk_crc,
            has_chunk_ms.then_some(chunk_ms),
            callbacks,
        )
    })();

    match result {
        Ok(bytes) => byte_success(bytes),
        Err(error) => byte_error(error),
    }
}

/// Encodes interleaved audio once through the primary MLX frame encoder and
/// returns canonical primary and codebook-prefix streams using shared LM
/// weights with independent state and arithmetic coders.
///
/// # Safety
///
/// C string pointers must be valid and terminated. `audio` must contain
/// `channels * samples` readable values. Callback pointers must remain valid.
#[no_mangle]
pub unsafe extern "C" fn encodec_rs_mlx_encode_ecdc_interleaved_dual(
    primary_bundle_dir: *const c_char,
    derived_bundle_dir: *const c_char,
    audio: *const f32,
    channels: usize,
    samples: usize,
    use_lm: bool,
    frame_batch_size: usize,
    chunk_crc: bool,
    chunk_ms: c_double,
    has_chunk_ms: bool,
    callbacks: EncodecRsMlxFrameCallbacks,
) -> EncodecRsMlxDualByteResult {
    let result = (|| -> Result<DualEcdcEncodeResult> {
        let primary_bundle_dir = bundle_dir_from_c(primary_bundle_dir)?;
        let derived_bundle_dir = bundle_dir_from_c(derived_bundle_dir)?;
        let audio = interleaved_audio_view(audio, channels, samples)?;
        encode_audio_view_dual(
            &primary_bundle_dir,
            &derived_bundle_dir,
            audio,
            use_lm,
            frame_batch_size,
            chunk_crc,
            has_chunk_ms.then_some(chunk_ms),
            callbacks,
        )
    })();

    match result {
        Ok(result) => dual_byte_success(result),
        Err(error) => dual_byte_error(error),
    }
}

/// Encodes planar audio and streams the output to a file.
///
/// # Safety
///
/// C string pointers must be valid and terminated.
/// `audio` must contain `channels * samples` readable values.
/// The callback pointers must remain valid during this call.
#[no_mangle]
pub unsafe extern "C" fn encodec_rs_mlx_encode_ecdc_stream_to_path(
    bundle_dir: *const c_char,
    audio: *const f32,
    channels: usize,
    samples: usize,
    use_lm: bool,
    frame_batch_size: usize,
    chunk_crc: bool,
    chunk_ms: c_double,
    has_chunk_ms: bool,
    output_path: *const c_char,
    progress_path: *const c_char,
    callbacks: EncodecRsMlxFrameCallbacks,
) -> EncodecRsMlxByteResult {
    let result = (|| -> Result<usize> {
        if output_path.is_null() {
            bail!("output_path pointer is null");
        }
        let output_path = PathBuf::from(
            CStr::from_ptr(output_path)
                .to_str()
                .context("output_path is not valid UTF-8")?,
        );
        let progress_path = if progress_path.is_null() {
            None
        } else {
            let value = CStr::from_ptr(progress_path)
                .to_str()
                .context("progress_path is not valid UTF-8")?;
            (!value.is_empty()).then(|| PathBuf::from(value))
        };
        let bundle_dir = bundle_dir_from_c(bundle_dir)?;
        let audio = planar_audio_view(audio, channels, samples)?;
        encode_audio_view_to_path(
            &bundle_dir,
            audio,
            use_lm,
            frame_batch_size,
            chunk_crc,
            has_chunk_ms.then_some(chunk_ms),
            &output_path,
            progress_path.as_deref(),
            callbacks,
        )
    })();

    match result {
        Ok(len) => byte_count_success(len),
        Err(error) => byte_error(error),
    }
}

/// Encodes interleaved audio and streams the output to a file.
///
/// # Safety
///
/// C string pointers must be valid and terminated.
/// `audio` must contain `channels * samples` readable values.
/// The callback pointers must remain valid during this call.
#[no_mangle]
pub unsafe extern "C" fn encodec_rs_mlx_encode_ecdc_interleaved_stream_to_path(
    bundle_dir: *const c_char,
    audio: *const f32,
    channels: usize,
    samples: usize,
    use_lm: bool,
    frame_batch_size: usize,
    chunk_crc: bool,
    chunk_ms: c_double,
    has_chunk_ms: bool,
    output_path: *const c_char,
    progress_path: *const c_char,
    callbacks: EncodecRsMlxFrameCallbacks,
) -> EncodecRsMlxByteResult {
    let result = (|| -> Result<usize> {
        if output_path.is_null() {
            bail!("output_path pointer is null");
        }
        let output_path = PathBuf::from(
            CStr::from_ptr(output_path)
                .to_str()
                .context("output_path is not valid UTF-8")?,
        );
        let progress_path = if progress_path.is_null() {
            None
        } else {
            let value = CStr::from_ptr(progress_path)
                .to_str()
                .context("progress_path is not valid UTF-8")?;
            (!value.is_empty()).then(|| PathBuf::from(value))
        };
        let bundle_dir = bundle_dir_from_c(bundle_dir)?;
        let audio = interleaved_audio_view(audio, channels, samples)?;
        encode_audio_view_to_path(
            &bundle_dir,
            audio,
            use_lm,
            frame_batch_size,
            chunk_crc,
            has_chunk_ms.then_some(chunk_ms),
            &output_path,
            progress_path.as_deref(),
            callbacks,
        )
    })();

    match result {
        Ok(len) => byte_count_success(len),
        Err(error) => byte_error(error),
    }
}

/// Decodes an ECDC payload through caller-supplied MLX frame callbacks.
///
/// # Safety
///
/// `bundle_dir` must be a valid terminated C string.
/// `payload` must contain `payload_len` readable bytes.
/// The callback pointers must remain valid during this call.
#[no_mangle]
pub unsafe extern "C" fn encodec_rs_mlx_decode_ecdc(
    bundle_dir: *const c_char,
    payload: *const u8,
    payload_len: usize,
    frame_batch_size: usize,
    callbacks: EncodecRsMlxFrameCallbacks,
) -> EncodecRsMlxAudioResult {
    let result = (|| -> Result<Array3<f32>> {
        if payload.is_null() && payload_len > 0 {
            bail!("payload pointer is null");
        }
        let bundle_dir = bundle_dir_from_c(bundle_dir)?;
        let mut codec = CallbackFrameCodec::from_bundle_dir(&bundle_dir, callbacks)?;
        let payload = if payload_len == 0 {
            &[]
        } else {
            slice::from_raw_parts(payload, payload_len)
        };
        let mut lm_codec = PortableLmCodec::from_dir(&bundle_dir)?;
        let decoded = decode_ecdc_with_batch_size(
            &mut codec,
            &mut lm_codec,
            payload,
            frame_batch_size.max(1),
        )?;
        Ok(decoded.audio)
    })();

    match result {
        Ok(audio) => audio_success(audio),
        Err(error) => audio_error(error),
    }
}

/// Decodes an ECDC payload directly into frame-major interleaved PCM.
///
/// Fixed-context payloads write borrowed decoder windows into the final
/// output. They do not assemble a complete planar track first.
///
/// # Safety
///
/// `bundle_dir` must be a valid terminated C string.
/// `payload` must contain `payload_len` readable bytes.
/// The callback pointers must remain valid during this call.
#[no_mangle]
pub unsafe extern "C" fn encodec_rs_mlx_decode_ecdc_interleaved(
    bundle_dir: *const c_char,
    payload: *const u8,
    payload_len: usize,
    frame_batch_size: usize,
    callbacks: EncodecRsMlxFrameCallbacks,
) -> EncodecRsMlxAudioResult {
    let result = (|| -> Result<(Vec<f32>, usize, usize)> {
        if payload.is_null() && payload_len > 0 {
            bail!("payload pointer is null");
        }
        let bundle_dir = bundle_dir_from_c(bundle_dir)?;
        let mut codec = CallbackFrameCodec::from_bundle_dir(&bundle_dir, callbacks)?;
        let payload = if payload_len == 0 {
            &[]
        } else {
            slice::from_raw_parts(payload, payload_len)
        };
        let mut lm_codec = PortableLmCodec::from_dir(&bundle_dir)?;
        let bundle_meta = codec.metadata().clone();
        let channels = bundle_meta.channels;
        let context =
            fixed_context_samples(bundle_meta.segment_samples, bundle_meta.segment_stride)?;

        if let Some(expected_context) = context {
            let mut output = None;
            let info = decode_ecdc_model_window_views_with_batch_size_and_progress(
                &mut codec,
                &mut lm_codec,
                payload,
                frame_batch_size.max(1),
                |info, _window_index, offset, owned_samples, window| {
                    let actual_context = info.context_samples.ok_or_else(|| {
                        anyhow::anyhow!("fixed-context bundle produced overlap-add geometry")
                    })?;
                    if actual_context != expected_context {
                        bail!(
                            "decoded context mismatch, expected {}, got {}",
                            expected_context,
                            actual_context,
                        );
                    }
                    let sample_count = channels
                        .checked_mul(info.metadata.audio_length)
                        .ok_or_else(|| anyhow::anyhow!("decoded sample count overflow"))?;
                    let output = output.get_or_insert_with(|| vec![0.0; sample_count]);
                    if output.len() != sample_count {
                        bail!("decoded sample count changed during decode");
                    }
                    copy_owned_interleaved(
                        output,
                        window,
                        channels,
                        info.metadata.audio_length,
                        actual_context,
                        offset,
                        owned_samples,
                    )
                },
                |_, _, _| Ok(()),
            )?;
            let sample_count = channels
                .checked_mul(info.metadata.audio_length)
                .ok_or_else(|| anyhow::anyhow!("decoded sample count overflow"))?;
            return Ok((
                output.unwrap_or_else(|| vec![0.0; sample_count]),
                channels,
                info.metadata.audio_length,
            ));
        }

        // Preserve legacy overlap-add reconstruction before interleaving.
        let decoded = decode_ecdc_with_batch_size(
            &mut codec,
            &mut lm_codec,
            payload,
            frame_batch_size.max(1),
        )?;
        let shape = decoded.audio.shape();
        if shape.len() != 3 || shape[0] != 1 || shape[1] != channels {
            bail!("decoded audio shape is invalid: {:?}", shape);
        }
        let frame_count = shape[2];
        let mut output = vec![0.0_f32; channels.saturating_mul(frame_count)];
        for frame in 0..frame_count {
            for channel in 0..channels {
                output[frame * channels + channel] = decoded.audio[[0, channel, frame]];
            }
        }
        Ok((output, channels, frame_count))
    })();

    match result {
        Ok((samples, channels, frame_count)) => {
            audio_samples_success(samples, channels, frame_count)
        }
        Err(error) => audio_error(error),
    }
}

/// Decodes ECDC audio and writes contiguous planar f32le samples to a file.
///
/// # Safety
///
/// C string pointers must be valid and terminated.
/// `payload` must contain `payload_len` readable bytes.
/// The callback pointers must remain valid during this call.
#[no_mangle]
pub unsafe extern "C" fn encodec_rs_mlx_decode_ecdc_planar_f32le_to_path(
    bundle_dir: *const c_char,
    payload: *const u8,
    payload_len: usize,
    frame_batch_size: usize,
    output_path: *const c_char,
    progress_path: *const c_char,
    callbacks: EncodecRsMlxFrameCallbacks,
) -> EncodecRsMlxByteResult {
    let result = (|| -> Result<usize> {
        if payload.is_null() && payload_len > 0 {
            bail!("payload pointer is null");
        }
        if output_path.is_null() {
            bail!("output_path pointer is null");
        }
        let output_path = PathBuf::from(
            CStr::from_ptr(output_path)
                .to_str()
                .context("output_path is not valid UTF-8")?,
        );
        let progress_path = if progress_path.is_null() {
            None
        } else {
            let value = CStr::from_ptr(progress_path)
                .to_str()
                .context("progress_path is not valid UTF-8")?;
            (!value.is_empty()).then(|| PathBuf::from(value))
        };
        let bundle_dir = bundle_dir_from_c(bundle_dir)?;
        let mut codec = CallbackFrameCodec::from_bundle_dir(&bundle_dir, callbacks)?;
        let payload = if payload_len == 0 {
            &[]
        } else {
            slice::from_raw_parts(payload, payload_len)
        };
        let mut lm_codec = PortableLmCodec::from_dir(&bundle_dir)?;
        let bundle_meta = codec.metadata().clone();
        let context =
            fixed_context_samples(bundle_meta.segment_samples, bundle_meta.segment_stride)?;

        if let Some(expected_context) = context {
            let channels = bundle_meta.channels;
            let mut output = File::create(&output_path)
                .with_context(|| format!("failed to create {}", output_path.display()))?;
            let mut output_len = None;
            let info = decode_ecdc_model_window_views_with_batch_size_and_progress(
                &mut codec,
                &mut lm_codec,
                payload,
                frame_batch_size.max(1),
                |info, _window_index, offset, owned_samples, window| {
                    let actual_context = info.context_samples.ok_or_else(|| {
                        anyhow::anyhow!("fixed-context bundle produced overlap-add geometry")
                    })?;
                    if actual_context != expected_context {
                        bail!(
                            "decoded context mismatch, expected {}, got {}",
                            expected_context,
                            actual_context,
                        );
                    }
                    let bytes_len = planar_f32le_byte_len(channels, info.metadata.audio_length)?;
                    match output_len {
                        Some(previous) if previous != bytes_len => {
                            bail!("decoded planar output length changed during decode")
                        }
                        None => {
                            output.set_len(u64::try_from(bytes_len)?)?;
                            output_len = Some(bytes_len);
                        }
                        _ => {}
                    }
                    write_owned_planar_f32le(
                        &mut output,
                        window,
                        channels,
                        info.metadata.audio_length,
                        actual_context,
                        offset,
                        owned_samples,
                    )
                },
                |stage, completed, total| {
                    if let Some(progress_path) = progress_path.as_deref() {
                        let stage = match stage {
                            EcdcDecodeProgressStage::Entropy => "entropy",
                            EcdcDecodeProgressStage::Model => "model",
                        };
                        let progress = format!(
                            "{{\"stage\":\"{stage}\",\"completed\":{completed},\"total\":{total}}}\n"
                        );
                        std::fs::write(progress_path, progress)?;
                    }
                    Ok(())
                },
            )?;
            let bytes_len = planar_f32le_byte_len(channels, info.metadata.audio_length)?;
            if output_len.is_none() {
                output.set_len(u64::try_from(bytes_len)?)?;
            }
            output.flush()?;
            return Ok(bytes_len);
        }

        // Legacy overlap-add bundles still require complete reconstruction.
        let decoded = decode_ecdc_with_batch_size_and_progress(
            &mut codec,
            &mut lm_codec,
            payload,
            frame_batch_size.max(1),
            |stage, completed, total| {
                if let Some(progress_path) = progress_path.as_deref() {
                    let stage = match stage {
                        EcdcDecodeProgressStage::Entropy => "entropy",
                        EcdcDecodeProgressStage::Model => "model",
                    };
                    let progress = format!(
                        "{{\"stage\":\"{stage}\",\"completed\":{completed},\"total\":{total}}}\n"
                    );
                    std::fs::write(progress_path, progress)?;
                }
                Ok(())
            },
        )?;
        let samples = decoded
            .audio
            .as_slice_memory_order()
            .context("decoded audio is not contiguous")?;
        let bytes_len = samples.len().saturating_mul(std::mem::size_of::<f32>());
        let mut output = File::create(&output_path)
            .with_context(|| format!("failed to create {}", output_path.display()))?;
        #[cfg(target_endian = "little")]
        {
            let bytes = slice::from_raw_parts(samples.as_ptr().cast::<u8>(), bytes_len);
            output.write_all(bytes)?;
        }
        #[cfg(target_endian = "big")]
        {
            for sample in samples {
                output.write_all(&sample.to_le_bytes())?;
            }
        }
        output.flush()?;
        Ok(bytes_len)
    })();

    match result {
        Ok(len) => byte_count_success(len),
        Err(error) => byte_error(error),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn indexed_window(base: f32) -> Array3<f32> {
        let mut window = Array3::<f32>::zeros((1, 2, 5));
        for channel in 0..2 {
            for sample in 0..5 {
                window[[0, channel, sample]] = base + (channel * 100 + sample) as f32;
            }
        }
        window
    }

    #[test]
    fn planar_writer_places_short_final_window_without_an_output_buffer() {
        let mut output = std::io::Cursor::new(vec![0_u8; 2 * 6 * 4]);
        let first = indexed_window(0.0);
        let final_window = indexed_window(10.0);

        write_owned_planar_f32le(&mut output, first.view(), 2, 6, 1, 0, 2).unwrap();
        write_owned_planar_f32le(&mut output, final_window.view(), 2, 6, 1, 2, 4).unwrap();

        let samples: Vec<f32> = output
            .into_inner()
            .chunks_exact(4)
            .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
            .collect();
        assert_eq!(
            samples,
            vec![1.0, 2.0, 11.0, 12.0, 13.0, 14.0, 101.0, 102.0, 111.0, 112.0, 113.0, 114.0,],
        );
    }

    #[test]
    fn interleaved_copy_places_short_final_window_in_soundkit_layout() {
        let mut output = vec![0.0_f32; 2 * 6];
        let first = indexed_window(0.0);
        let final_window = indexed_window(10.0);

        copy_owned_interleaved(&mut output, first.view(), 2, 6, 1, 0, 2).unwrap();
        copy_owned_interleaved(&mut output, final_window.view(), 2, 6, 1, 2, 4).unwrap();

        assert_eq!(
            output,
            vec![1.0, 101.0, 2.0, 102.0, 11.0, 111.0, 12.0, 112.0, 13.0, 113.0, 14.0, 114.0,],
        );
    }
}
