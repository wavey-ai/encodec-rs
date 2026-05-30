use std::ffi::{c_char, c_void, CStr, CString};
use std::fs::File;
use std::io::Write;
use std::os::raw::c_double;
use std::path::{Path, PathBuf};
use std::ptr;
use std::slice;

use anyhow::{bail, Context, Result};
use ndarray::{Array2, Array3};

use crate::ecdc::{decode_ecdc, encode_audio_to_ecdc_with_options, FrameCodec};
use crate::ecdc::encode_audio_to_ecdc_stream_with_options;
use crate::format::segment_frame_length;
use crate::metadata::OnnxFrameBundleMetadata;
use crate::portable_lm::PortableLmCodec;

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
    fn from_bundle_dir(bundle_dir: impl AsRef<Path>, callbacks: EncodecRsMlxFrameCallbacks) -> Result<Self> {
        let metadata_path = bundle_dir.as_ref().join("bundle.json");
        let metadata: OnnxFrameBundleMetadata = serde_json::from_str(
            &std::fs::read_to_string(&metadata_path)
                .with_context(|| format!("failed to read {}", metadata_path.display()))?,
        )
        .with_context(|| format!("failed to parse {}", metadata_path.display()))?;
        Ok(Self { metadata, callbacks })
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
        let audio = audio.as_slice_memory_order().context("MLX encode audio batch is not contiguous")?;
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
        let codes = Array3::from_shape_vec((batch, self.metadata.num_codebooks, frame_length), codes)?;
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
        let codes = codes.as_slice_memory_order().context("MLX decode code batch is not contiguous")?;
        let scales = scale.as_slice_memory_order().context("MLX decode scale batch is not contiguous")?;
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
        Ok(Array3::from_shape_vec((batch, self.metadata.channels, decoded_samples), audio)?)
    }
}

unsafe fn bundle_dir_from_c(bundle_dir: *const c_char) -> Result<PathBuf> {
    if bundle_dir.is_null() {
        bail!("bundle_dir pointer is null");
    }
    let value = CStr::from_ptr(bundle_dir).to_str().context("bundle_dir is not valid UTF-8")?;
    if value.is_empty() {
        bail!("bundle_dir is empty");
    }
    Ok(PathBuf::from(value))
}

fn c_error(error: impl std::fmt::Display) -> *mut c_char {
    CString::new(error.to_string()).map(CString::into_raw).unwrap_or(ptr::null_mut())
}

fn byte_success(bytes: Vec<u8>) -> EncodecRsMlxByteResult {
    let mut bytes = bytes.into_boxed_slice();
    let len = bytes.len();
    let ptr = bytes.as_mut_ptr();
    std::mem::forget(bytes);
    EncodecRsMlxByteResult { ok: true, ptr, len, error: ptr::null_mut() }
}

fn byte_count_success(len: usize) -> EncodecRsMlxByteResult {
    EncodecRsMlxByteResult { ok: true, ptr: ptr::null_mut(), len, error: ptr::null_mut() }
}

fn byte_error(error: impl std::fmt::Display) -> EncodecRsMlxByteResult {
    EncodecRsMlxByteResult { ok: false, ptr: ptr::null_mut(), len: 0, error: c_error(error) }
}

fn audio_success(audio: Array3<f32>) -> EncodecRsMlxAudioResult {
    let shape = audio.shape().to_vec();
    let (data, offset) = audio.into_raw_vec_and_offset();
    debug_assert_eq!(offset, Some(0));
    let mut data = data.into_boxed_slice();
    let len = data.len();
    let ptr = data.as_mut_ptr();
    std::mem::forget(data);
    EncodecRsMlxAudioResult { ok: true, ptr, len, channels: shape[1], samples: shape[2], error: ptr::null_mut() }
}

fn audio_error(error: impl std::fmt::Display) -> EncodecRsMlxAudioResult {
    EncodecRsMlxAudioResult { ok: false, ptr: ptr::null_mut(), len: 0, channels: 0, samples: 0, error: c_error(error) }
}

#[no_mangle]
pub unsafe extern "C" fn encodec_rs_mlx_free_string(value: *mut c_char) {
    if value.is_null() {
        return;
    }
    drop(CString::from_raw(value));
}

#[no_mangle]
pub unsafe extern "C" fn encodec_rs_mlx_free_bytes(ptr: *mut u8, len: usize) {
    if ptr.is_null() {
        return;
    }
    drop(Box::from_raw(ptr::slice_from_raw_parts_mut(ptr, len)));
}

#[no_mangle]
pub unsafe extern "C" fn encodec_rs_mlx_free_audio(ptr: *mut f32, len: usize) {
    if ptr.is_null() {
        return;
    }
    drop(Box::from_raw(ptr::slice_from_raw_parts_mut(ptr, len)));
}

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
        if audio.is_null() && channels.saturating_mul(samples) > 0 {
            bail!("audio pointer is null");
        }
        let bundle_dir = bundle_dir_from_c(bundle_dir)?;
        let mut codec = CallbackFrameCodec::from_bundle_dir(&bundle_dir, callbacks)?;
        if channels != codec.metadata.channels {
            bail!("audio channel count {channels} does not match bundle {}", codec.metadata.channels);
        }
        let audio = if channels.saturating_mul(samples) == 0 {
            &[]
        } else {
            slice::from_raw_parts(audio, channels.saturating_mul(samples))
        };
        let audio = Array3::from_shape_vec((1, channels, samples), audio.to_vec())?;
        if !use_lm {
            bail!("use_lm=false is unsupported for q8 ECDC payloads in this build");
        }
        let mut lm_codec = PortableLmCodec::from_dir(&bundle_dir)?;
        encode_audio_to_ecdc_with_options(
            &mut codec,
            &mut lm_codec,
            &audio,
            None,
            frame_batch_size.max(1),
            chunk_crc,
            has_chunk_ms.then_some(chunk_ms as f64),
        )
    })();

    match result {
        Ok(bytes) => byte_success(bytes),
        Err(error) => byte_error(error),
    }
}

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
        if audio.is_null() && channels.saturating_mul(samples) > 0 {
            bail!("audio pointer is null");
        }
        if output_path.is_null() {
            bail!("output_path pointer is null");
        }
        let output_path = CStr::from_ptr(output_path)
            .to_str()
            .context("output_path is not valid UTF-8")?;
        let progress_path = if progress_path.is_null() {
            None
        } else {
            let value = CStr::from_ptr(progress_path)
                .to_str()
                .context("progress_path is not valid UTF-8")?;
            (!value.is_empty()).then_some(value.to_owned())
        };
        let bundle_dir = bundle_dir_from_c(bundle_dir)?;
        let mut codec = CallbackFrameCodec::from_bundle_dir(&bundle_dir, callbacks)?;
        if channels != codec.metadata.channels {
            bail!("audio channel count {channels} does not match bundle {}", codec.metadata.channels);
        }
        let audio = if channels.saturating_mul(samples) == 0 {
            &[]
        } else {
            slice::from_raw_parts(audio, channels.saturating_mul(samples))
        };
        let audio = Array3::from_shape_vec((1, channels, samples), audio.to_vec())?;
        if !use_lm {
            bail!("use_lm=false is unsupported for q8 ECDC payloads in this build");
        }
        let mut lm_codec = PortableLmCodec::from_dir(&bundle_dir)?;
        let mut output = File::create(output_path)
            .with_context(|| format!("failed to create {output_path}"))?;
        let mut bytes_written = 0_usize;
        let mut emissions = 0_usize;
        encode_audio_to_ecdc_stream_with_options(
            &mut codec,
            &mut lm_codec,
            &audio,
            None,
            frame_batch_size.max(1),
            chunk_crc,
            has_chunk_ms.then_some(chunk_ms as f64),
            |bytes| {
                output.write_all(bytes)?;
                bytes_written += bytes.len();
                emissions += 1;
                if let Some(progress_path) = progress_path.as_deref() {
                    let progress = format!(
                        "{{\"bytes_written\":{bytes_written},\"emissions\":{emissions}}}\n"
                    );
                    std::fs::write(progress_path, progress)?;
                }
                Ok(())
            },
        )?;
        output.flush()?;
        Ok(bytes_written)
    })();

    match result {
        Ok(len) => byte_count_success(len),
        Err(error) => byte_error(error),
    }
}

#[no_mangle]
pub unsafe extern "C" fn encodec_rs_mlx_decode_ecdc(
    bundle_dir: *const c_char,
    payload: *const u8,
    payload_len: usize,
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
        let decoded = decode_ecdc(&mut codec, &mut lm_codec, payload)?;
        Ok(decoded.audio)
    })();

    match result {
        Ok(audio) => audio_success(audio),
        Err(error) => audio_error(error),
    }
}
