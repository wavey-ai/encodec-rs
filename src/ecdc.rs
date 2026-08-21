use std::io::Cursor;
use std::time::Instant;

use anyhow::{bail, Result};
use ndarray::{Array2, Array3, Array4};

use crate::arithmetic::{ArithmeticDecoder, ArithmeticEncoder, CdfScratch};
use crate::binary::{
    read_chunk_payload, read_ecdc_header, read_exactly, write_chunk, write_ecdc_header,
};
use crate::ecdc_presets::fixed_context_samples;
use crate::entropy::{
    probability_columns_from_flat_logits, ProbabilityParameters, ProbabilityScratch,
};
use crate::format::{
    ecdc_chunk_layout_for_chunk_count, ecdc_chunk_layout_from_ms, ecdc_lm_frame_length,
    segment_frame_length, segment_starts, validate_metadata, EcdcChunkLayout,
};
pub use crate::format::{
    EcdcMetadata, SourceAudioMetadata, ARITHMETIC_TOTAL_RANGE_BITS, DEFAULT_FP_SCALE,
    DEFAULT_MIN_RANGE, QUANTIZED_LM_BITSTREAM_VERSION,
};
use crate::metadata::OnnxFrameBundleMetadata;

pub trait FrameCodec {
    fn metadata(&self) -> &OnnxFrameBundleMetadata;

    fn encode_frame(&mut self, audio: &Array3<f32>) -> Result<(Array3<i64>, Array2<f32>)>;

    fn decode_frame(&mut self, codes: &Array3<i64>, scale: &Array2<f32>) -> Result<Array3<f32>>;
}

pub trait LmCodec {
    fn metadata(&self) -> &OnnxFrameBundleMetadata;

    fn bitstream_version(&self) -> u8 {
        QUANTIZED_LM_BITSTREAM_VERSION
    }

    fn bitstream_lm_hash(&self) -> Option<&str> {
        None
    }

    fn lm_window_frame_length(&self) -> usize {
        self.metadata().frame_length
    }

    fn initial_states(&self, batch: usize) -> Result<Vec<Array3<f32>>>;

    fn forward_logits(
        &mut self,
        indices: &Array3<i64>,
        offset: i64,
        states: &[Array3<f32>],
    ) -> Result<(Array4<f32>, i64, Vec<Array3<f32>>)>;
}

impl EcdcMetadata {
    pub fn from_codec(
        codec: &dyn FrameCodec,
        audio_length: usize,
        source: Option<&SourceAudioMetadata>,
        lm_hash: Option<String>,
    ) -> Self {
        Self::from_bundle(codec.metadata(), audio_length, source, lm_hash)
    }
}

#[derive(Debug, Clone)]
pub struct DecodedEcdcAudio {
    pub metadata: EcdcMetadata,
    pub audio: Array3<f32>,
}

/// Describes the complete decoded model windows in one ECDC payload.
#[derive(Debug, Clone)]
pub struct DecodedEcdcWindowInfo {
    pub metadata: EcdcMetadata,
    pub chunk_layout: EcdcChunkLayout,
    pub context_samples: Option<usize>,
    pub window_count: usize,
}

#[derive(Debug, Clone)]
pub struct EncodedFrameEvidence {
    pub offset_samples: usize,
    pub owned_samples: usize,
    pub model_input: Array3<f32>,
    pub codes: Array3<i64>,
    pub scale: Array2<f32>,
}

#[derive(Debug, Clone)]
pub struct LmChunkEvidence {
    pub payload: Vec<u8>,
    pub entropy: Vec<u8>,
    pub recovered_codes: Array3<i64>,
    pub recovered_scale: Array2<f32>,
}

/// Encodes each neural model input and retains the tensors needed for qualification.
///
/// Fixed-context bundles always use their complete model window. A non-fixed
/// bundle can use its actual final input length when `true_variable_tail` is
/// enabled. Variable-length batches contain only inputs with the same length.
pub fn encode_audio_frame_evidence(
    codec: &mut dyn FrameCodec,
    audio: &Array3<f32>,
    frame_batch_size: usize,
    true_variable_tail: bool,
) -> Result<Vec<EncodedFrameEvidence>> {
    let meta = codec.metadata().clone();
    let shape = audio.shape();
    if shape.len() != 3 || shape[0] != 1 || shape[1] != meta.channels {
        bail!(
            "audio must have shape [1, {}, samples], got {:?}",
            meta.channels,
            shape,
        );
    }
    let total_samples = shape[2];
    if total_samples == 0 {
        bail!("audio must contain at least one sample");
    }

    let context = fixed_context_samples(meta.segment_samples, meta.segment_stride)?;
    if true_variable_tail && context.is_some() {
        bail!("true variable tails are not valid for a fixed-context bundle");
    }

    #[derive(Clone, Copy)]
    struct SegmentPlan {
        offset: usize,
        owned_samples: usize,
        model_samples: usize,
    }

    let plans: Vec<_> = segment_starts(total_samples, meta.segment_stride)
        .into_iter()
        .map(|offset| SegmentPlan {
            offset,
            owned_samples: (total_samples - offset).min(meta.segment_stride),
            model_samples: if true_variable_tail {
                (total_samples - offset).min(meta.segment_samples)
            } else {
                meta.segment_samples
            },
        })
        .collect();

    let batch_limit = frame_batch_size.max(1);
    let mut evidence = Vec::with_capacity(plans.len());
    let mut plan_index = 0;
    while plan_index < plans.len() {
        let model_samples = plans[plan_index].model_samples;
        let mut batch_end = plan_index + 1;
        while batch_end < plans.len()
            && batch_end - plan_index < batch_limit
            && plans[batch_end].model_samples == model_samples
        {
            batch_end += 1;
        }
        let batch_plans = &plans[plan_index..batch_end];
        let mut batch = Array3::<f32>::zeros((batch_plans.len(), meta.channels, model_samples));
        let context = context.unwrap_or(0);
        for (batch_index, plan) in batch_plans.iter().enumerate() {
            for channel in 0..meta.channels {
                for model_index in 0..model_samples {
                    let source_index =
                        plan.offset as isize - context as isize + model_index as isize;
                    if source_index >= 0 && (source_index as usize) < total_samples {
                        batch[[batch_index, channel, model_index]] =
                            audio[[0, channel, source_index as usize]];
                    }
                }
            }
        }

        let (batch_codes, batch_scales) = codec.encode_frame(&batch)?;
        let code_shape = batch_codes.shape();
        if code_shape.len() != 3
            || code_shape[0] != batch_plans.len()
            || code_shape[1] != meta.num_codebooks
        {
            bail!(
                "encoded code shape mismatch, expected [{}, {}, frames], got {:?}",
                batch_plans.len(),
                meta.num_codebooks,
                code_shape,
            );
        }
        let expected_frame_length = if true_variable_tail {
            segment_frame_length(model_samples, meta.segment_samples, meta.frame_length)
        } else {
            meta.frame_length
        };
        if code_shape[2] != expected_frame_length {
            bail!(
                "encoded frame length {} does not match expected length {} for {} model samples",
                code_shape[2],
                expected_frame_length,
                model_samples,
            );
        }
        if batch_scales.shape() != [batch_plans.len(), 1] {
            bail!(
                "encoded scale shape mismatch, expected [{}, 1], got {:?}",
                batch_plans.len(),
                batch_scales.shape(),
            );
        }

        for (batch_index, plan) in batch_plans.iter().enumerate() {
            let mut model_input = Array3::<f32>::zeros((1, meta.channels, model_samples));
            let mut codes = Array3::<i64>::zeros((1, meta.num_codebooks, expected_frame_length));
            let mut scale = Array2::<f32>::zeros((1, 1));
            for channel in 0..meta.channels {
                for sample in 0..model_samples {
                    model_input[[0, channel, sample]] = batch[[batch_index, channel, sample]];
                }
            }
            for codebook in 0..meta.num_codebooks {
                for frame in 0..expected_frame_length {
                    codes[[0, codebook, frame]] = batch_codes[[batch_index, codebook, frame]];
                }
            }
            scale[[0, 0]] = batch_scales[[batch_index, 0]];
            evidence.push(EncodedFrameEvidence {
                offset_samples: plan.offset,
                owned_samples: plan.owned_samples,
                model_input,
                codes,
                scale,
            });
        }
        plan_index = batch_end;
    }

    Ok(evidence)
}

pub fn encode_audio_to_ecdc(
    codec: &mut dyn FrameCodec,
    lm_codec: &mut dyn LmCodec,
    audio: &Array3<f32>,
    source: Option<&SourceAudioMetadata>,
) -> Result<Vec<u8>> {
    collect_ecdc_bytes(|emit| {
        encode_audio_to_ecdc_impl(
            codec,
            lm_codec,
            audio,
            source,
            frame_encode_batch_size(),
            true,
            None,
            emit,
        )
    })
}

pub fn encode_audio_to_ecdc_with_batch_size(
    codec: &mut dyn FrameCodec,
    lm_codec: &mut dyn LmCodec,
    audio: &Array3<f32>,
    source: Option<&SourceAudioMetadata>,
    frame_batch_size: usize,
) -> Result<Vec<u8>> {
    collect_ecdc_bytes(|emit| {
        encode_audio_to_ecdc_impl(
            codec,
            lm_codec,
            audio,
            source,
            frame_batch_size.max(1),
            true,
            None,
            emit,
        )
    })
}

pub fn encode_audio_to_ecdc_with_options(
    codec: &mut dyn FrameCodec,
    lm_codec: &mut dyn LmCodec,
    audio: &Array3<f32>,
    source: Option<&SourceAudioMetadata>,
    frame_batch_size: usize,
    chunk_crc: bool,
    chunk_ms: Option<f64>,
) -> Result<Vec<u8>> {
    if !chunk_crc {
        bail!("q8 ECDC always writes CRC-wrapped chunks");
    }
    collect_ecdc_bytes(|emit| {
        encode_audio_to_ecdc_impl(
            codec,
            lm_codec,
            audio,
            source,
            frame_batch_size.max(1),
            chunk_crc,
            chunk_ms,
            emit,
        )
    })
}

#[allow(clippy::too_many_arguments)]
pub fn encode_audio_to_ecdc_stream_with_options<F>(
    codec: &mut dyn FrameCodec,
    lm_codec: &mut dyn LmCodec,
    audio: &Array3<f32>,
    source: Option<&SourceAudioMetadata>,
    frame_batch_size: usize,
    chunk_crc: bool,
    chunk_ms: Option<f64>,
    mut on_bytes: F,
) -> Result<()>
where
    F: FnMut(&[u8]) -> Result<()>,
{
    if !chunk_crc {
        bail!("q8 ECDC always writes CRC-wrapped chunks");
    }
    encode_audio_to_ecdc_impl(
        codec,
        lm_codec,
        audio,
        source,
        frame_batch_size.max(1),
        chunk_crc,
        chunk_ms,
        &mut on_bytes,
    )
}

pub fn encode_ecdc_header_with_options(
    codec: &dyn FrameCodec,
    audio_length: usize,
    source: Option<&SourceAudioMetadata>,
    lm_hash: Option<String>,
    chunk_layout: Option<EcdcChunkLayout>,
) -> Result<Vec<u8>> {
    let bundle_meta = codec.metadata();
    let mut metadata = EcdcMetadata::from_bundle(bundle_meta, audio_length, source, lm_hash);
    if let Some(chunk_layout) = chunk_layout {
        metadata.lm_frame_length = Some(segment_frame_length(
            chunk_layout.samples,
            bundle_meta.segment_samples,
            bundle_meta.frame_length,
        ));
    } else if fixed_context_samples(bundle_meta.segment_samples, bundle_meta.segment_stride)?
        .is_some()
    {
        // The fixed model/code window always covers the full bundle
        // frame_length, regardless of how much logical owned audio a
        // chunk carries; segment_frame_length()'s al/segment_samples
        // proportion would otherwise undercount it (e.g. 200 vs 203).
        metadata.lm_frame_length = Some(bundle_meta.frame_length);
    }
    let mut header = Vec::new();
    write_ecdc_header(&mut header, &metadata)?;
    Ok(header)
}

pub fn encode_ecdc_segment_batch_with_options<F>(
    codec: &mut dyn FrameCodec,
    lm_codec: &mut dyn LmCodec,
    batch: &Array3<f32>,
    frame_lengths: &[usize],
    mut on_bytes: F,
) -> Result<()>
where
    F: FnMut(&[u8]) -> Result<()>,
{
    encode_ecdc_segment_batch_impl(codec, lm_codec, batch, frame_lengths, true, &mut on_bytes)
}

pub fn decode_ecdc(
    codec: &mut dyn FrameCodec,
    lm_codec: &mut dyn LmCodec,
    payload: &[u8],
) -> Result<DecodedEcdcAudio> {
    decode_ecdc_impl(codec, lm_codec, payload)
}

/// Decodes complete model windows without cropping their context samples.
///
/// The callback receives the window index, logical owned offset, owned sample
/// count, and decoded `[1, channel, sample]` window.
pub fn decode_ecdc_model_windows<F>(
    codec: &mut dyn FrameCodec,
    lm_codec: &mut dyn LmCodec,
    payload: &[u8],
    on_window: F,
) -> Result<DecodedEcdcWindowInfo>
where
    F: FnMut(&DecodedEcdcWindowInfo, usize, usize, usize, Array3<f32>) -> Result<()>,
{
    decode_ecdc_model_windows_impl(codec, lm_codec, payload, on_window)
}

fn collect_ecdc_bytes<F>(encode: F) -> Result<Vec<u8>>
where
    F: FnOnce(&mut dyn FnMut(&[u8]) -> Result<()>) -> Result<()>,
{
    let mut out = Vec::new();
    let mut emit = |bytes: &[u8]| -> Result<()> {
        out.extend_from_slice(bytes);
        Ok(())
    };
    encode(&mut emit)?;
    Ok(out)
}

#[allow(clippy::too_many_arguments)]
fn encode_audio_to_ecdc_impl(
    codec: &mut dyn FrameCodec,
    lm_codec: &mut dyn LmCodec,
    audio: &Array3<f32>,
    source: Option<&SourceAudioMetadata>,
    frame_batch_size: usize,
    chunk_crc: bool,
    chunk_ms: Option<f64>,
    emit: &mut dyn FnMut(&[u8]) -> Result<()>,
) -> Result<()> {
    let profile_enabled = std::env::var_os("ENCODEC_RS_PROFILE").is_some();
    let shape = audio.shape();
    if shape.len() != 3 || shape[0] != 1 {
        bail!(
            "audio must have shape [1, channels, samples], got {:?}",
            shape
        );
    }

    let model_meta = codec.metadata().clone();
    if shape[1] != model_meta.channels {
        bail!(
            "audio channel mismatch, expected {}, got {}",
            model_meta.channels,
            shape[1]
        );
    }

    if lm_codec.bitstream_version() != QUANTIZED_LM_BITSTREAM_VERSION {
        bail!(
            "only q8 LM acv={} is supported, runtime provides acv={}",
            QUANTIZED_LM_BITSTREAM_VERSION,
            lm_codec.bitstream_version()
        );
    }

    let chunk_layout = ecdc_chunk_layout_from_ms(&model_meta, chunk_ms)?;
    let lm_hash = lm_codec
        .bitstream_lm_hash()
        .map(str::to_owned)
        .ok_or_else(|| anyhow::anyhow!("q8 LM runtime does not expose an LM hash"))?;
    let header = encode_ecdc_header_with_options(
        codec,
        shape[2],
        source,
        Some(lm_hash),
        chunk_ms.map(|_| chunk_layout),
    )?;
    emit(&header)?;

    let fixed_lm_frame_length = chunk_ms.map(|_| {
        segment_frame_length(
            chunk_layout.samples,
            model_meta.segment_samples,
            model_meta.frame_length,
        )
    });
    for (batch_index, (frame_lengths, batch)) in encode_segment_batches_with_size(
        audio,
        &model_meta,
        frame_batch_size,
        chunk_layout,
        fixed_lm_frame_length,
    )?
    .into_iter()
    .enumerate()
    {
        encode_ecdc_segment_batch_impl(codec, lm_codec, &batch, &frame_lengths, chunk_crc, emit)?;
        if profile_enabled {
            eprintln!(
                "encode_segment_batch batch={} segments={}",
                batch_index,
                frame_lengths.len(),
            );
        }
    }

    Ok(())
}

fn encode_ecdc_segment_batch_impl(
    codec: &mut dyn FrameCodec,
    lm_codec: &mut dyn LmCodec,
    batch: &Array3<f32>,
    frame_lengths: &[usize],
    chunk_crc: bool,
    emit: &mut dyn FnMut(&[u8]) -> Result<()>,
) -> Result<()> {
    let profile_enabled = std::env::var_os("ENCODEC_RS_PROFILE").is_some();
    let model_meta = codec.metadata().clone();
    let batch_shape = batch.shape();
    if batch_shape.len() != 3 {
        bail!("segment batch must have shape [batch, channels, samples]");
    }
    if batch_shape[1] != model_meta.channels {
        bail!(
            "segment batch channel mismatch, expected {}, got {:?}",
            model_meta.channels,
            batch_shape
        );
    }
    if batch_shape[2] == 0 {
        bail!("segment batch sample length must be non-zero");
    }
    if batch_shape[0] != frame_lengths.len() {
        bail!(
            "segment batch size {} does not match frame_lengths {}",
            batch_shape[0],
            frame_lengths.len()
        );
    }

    let frame_started = profile_enabled.then(Instant::now);
    let (codes_full, scales) = codec.encode_frame(batch)?;
    if let Some(frame_started) = frame_started {
        let frame_done = Instant::now();
        eprintln!(
            "encode_segment_batch segments={} frame_encode_ms={:.3}",
            frame_lengths.len(),
            (frame_done - frame_started).as_secs_f64() * 1000.0,
        );
    }

    let code_shape = codes_full.shape();
    if code_shape.len() != 3
        || code_shape[0] != batch_shape[0]
        || code_shape[1] != model_meta.num_codebooks
    {
        bail!(
            "encoded code shape mismatch, expected [batch, {}, frames], got {:?}",
            model_meta.num_codebooks,
            code_shape
        );
    }
    let encoded_frame_length = code_shape[2];

    for (segment_index, frame_length) in frame_lengths.iter().copied().enumerate() {
        if frame_length == 0 || frame_length > encoded_frame_length {
            bail!(
                "segment frame length {} is out of range for encoded frame length {}",
                frame_length,
                encoded_frame_length
            );
        }
        let mut encoded_chunk = Vec::new();
        let payload = encode_lm_chunk_payload(
            lm_codec,
            &codes_full,
            &scales,
            segment_index,
            frame_length,
            DEFAULT_FP_SCALE,
            DEFAULT_MIN_RANGE,
            1.0,
        )?;
        write_chunk(&mut encoded_chunk, &payload, chunk_crc)?;
        emit(&encoded_chunk)?;
    }

    Ok(())
}

fn decode_ecdc_impl(
    codec: &mut dyn FrameCodec,
    lm_codec: &mut dyn LmCodec,
    payload: &[u8],
) -> Result<DecodedEcdcAudio> {
    let channels = codec.metadata().channels;
    let mut fixed_audio = None;
    let mut windows = Vec::new();
    let info = decode_ecdc_model_windows_impl(
        codec,
        lm_codec,
        payload,
        |info, _window_index, offset, owned_samples, window| {
            if let Some(context) = info.context_samples {
                let output = fixed_audio.get_or_insert_with(|| {
                    Array3::<f32>::zeros((1, channels, info.metadata.audio_length))
                });
                copy_owned_segment_into(&window, context, owned_samples, output, offset)?;
            } else {
                windows.push(window);
            }
            Ok(())
        },
    )?;

    let audio_length = info.metadata.audio_length;
    let stride = info.chunk_layout.stride;
    let audio = if info.context_samples.is_some() {
        fixed_audio.ok_or_else(|| anyhow::anyhow!("fixed-context payload has no model windows"))?
    } else {
        let reconstructed = if windows.len() <= 1 {
            windows
                .into_iter()
                .next()
                .unwrap_or_else(|| Array3::<f32>::zeros((1, channels, 0)))
        } else {
            linear_overlap_add(&windows, stride)
        };
        trim_audio_to_length(reconstructed, channels, audio_length)?
    };

    Ok(DecodedEcdcAudio {
        metadata: info.metadata,
        audio,
    })
}

fn decode_ecdc_model_windows_impl<F>(
    codec: &mut dyn FrameCodec,
    lm_codec: &mut dyn LmCodec,
    payload: &[u8],
    mut on_window: F,
) -> Result<DecodedEcdcWindowInfo>
where
    F: FnMut(&DecodedEcdcWindowInfo, usize, usize, usize, Array3<f32>) -> Result<()>,
{
    let mut reader = Cursor::new(payload);
    let metadata: EcdcMetadata = read_ecdc_header(&mut reader)?;
    let bundle_meta = codec.metadata().clone();
    validate_metadata(&bundle_meta, &metadata)?;
    let mut raw_chunks = Vec::new();
    while (reader.position() as usize) < payload.len() {
        raw_chunks.push(read_chunk_payload(&mut reader, true)?);
    }

    let chunk_layout =
        ecdc_chunk_layout_for_chunk_count(&bundle_meta, &metadata, raw_chunks.len())?;
    let context = fixed_context_samples(chunk_layout.samples, chunk_layout.stride)?;

    let lm_version = lm_codec.bitstream_version();
    if lm_version != metadata.bitstream_version {
        bail!(
            "payload requires LM bitstream acv={}, but bundle/runtime provides acv={}",
            metadata.bitstream_version,
            lm_version,
        );
    }
    let Some(expected_hash) = metadata.lm_hash.as_deref() else {
        bail!("q8 LM payload is missing required LM hash");
    };
    let Some(actual_hash) = lm_codec.bitstream_lm_hash() else {
        bail!("q8 LM runtime does not expose an LM hash");
    };
    if actual_hash != expected_hash {
        bail!(
            "payload requires q8 LM hash {}, but bundle/runtime provides {}",
            expected_hash,
            actual_hash,
        );
    }

    let starts = segment_starts(metadata.audio_length, chunk_layout.stride);
    if starts.len() != raw_chunks.len() {
        bail!(
            "LM ECDC payload has {} chunks, but metadata implies {} chunks",
            raw_chunks.len(),
            starts.len()
        );
    }
    let info = DecodedEcdcWindowInfo {
        metadata,
        chunk_layout,
        context_samples: context,
        window_count: raw_chunks.len(),
    };
    for (window_index, (offset, chunk)) in starts.into_iter().zip(raw_chunks).enumerate() {
        let owned_len = (info.metadata.audio_length - offset).min(info.chunk_layout.stride);
        let decode_len = if context.is_some() {
            info.chunk_layout.samples
        } else {
            owned_len
        };
        let frame_length = if context.is_some() {
            info.metadata
                .lm_frame_length
                .filter(|value| *value > 0)
                .unwrap_or(bundle_meta.frame_length)
        } else {
            ecdc_lm_frame_length(
                &info.metadata,
                owned_len,
                bundle_meta.segment_samples,
                bundle_meta.frame_length,
            )
        };
        let frame = decode_lm_chunk_payload(
            codec,
            lm_codec,
            &bundle_meta,
            &info.metadata,
            &chunk,
            decode_len,
            frame_length,
        )?;
        on_window(&info, window_index, offset, owned_len, frame)?;
    }
    Ok(info)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_lm_chunk_payload(
    lm_codec: &mut dyn LmCodec,
    codes: &Array3<i64>,
    scales: &Array2<f32>,
    batch_index: usize,
    frame_length: usize,
    fp_scale: i64,
    min_range: i64,
    lm_tau: f64,
) -> Result<Vec<u8>> {
    let profile_enabled = std::env::var_os("ENCODEC_RS_PROFILE").is_some();
    let started = profile_enabled.then(Instant::now);
    let meta = lm_codec.metadata().clone();
    let mut payload = Vec::new();
    if meta.normalize {
        payload.extend_from_slice(&scales[[batch_index, 0]].to_be_bytes());
    }
    let mut encoder = ArithmeticEncoder::new(ARITHMETIC_TOTAL_RANGE_BITS)?;
    let mut states = lm_codec.initial_states(1)?;
    let mut offset = 0_i64;
    let mut input = Array3::<i64>::zeros((1, meta.num_codebooks, 1));
    let mut symbols = vec![0_usize; meta.num_codebooks];
    let mut scratch = ProbabilityScratch::default();
    let mut cdf_scratch = CdfScratch::default();
    let lm_window_frame_length = lm_codec.lm_window_frame_length().max(1);
    if frame_length > lm_window_frame_length {
        bail!(
            "chunk frame length {} exceeds LM positional capacity {}",
            frame_length,
            lm_window_frame_length,
        );
    }
    let mut lm_elapsed = 0.0_f64;
    let mut pdf_elapsed = 0.0_f64;
    let mut arithmetic_elapsed = 0.0_f64;

    for t in 0..frame_length {
        let lm_started = profile_enabled.then(Instant::now);
        let (logits, next_offset, next_states) =
            lm_codec.forward_logits(&input, offset, &states)?;
        if let Some(lm_started) = lm_started {
            lm_elapsed += lm_started.elapsed().as_secs_f64() * 1000.0;
        }

        let pdf_started = profile_enabled.then(Instant::now);
        let pdf = probability_columns_from_logits(
            &logits,
            lm_tau,
            meta.lm_entropy_logit_step(),
            fp_scale,
            &mut scratch,
        )?;
        if let Some(pdf_started) = pdf_started {
            pdf_elapsed += pdf_started.elapsed().as_secs_f64() * 1000.0;
        }

        for codebook in 0..meta.num_codebooks {
            let value = codes[[batch_index, codebook, t]];
            if value < 0 {
                bail!("code symbol must be non-negative, got {value}");
            }
            symbols[codebook] = value as usize;
            input[[0, codebook, 0]] = value + 1;
        }

        let arithmetic_started = profile_enabled.then(Instant::now);
        encoder.push_pdf_symbols_with_scratch(
            pdf,
            meta.lm_cardinality(),
            meta.num_codebooks,
            &symbols,
            fp_scale,
            min_range,
            &mut cdf_scratch,
        )?;
        if let Some(arithmetic_started) = arithmetic_started {
            arithmetic_elapsed += arithmetic_started.elapsed().as_secs_f64() * 1000.0;
        }

        states = next_states;
        offset = next_offset;
    }

    payload.extend_from_slice(&encoder.finish());
    if let Some(started) = started {
        let done = Instant::now();
        eprintln!(
            "encode_lm_chunk_payload frame_length={} lm_ms={:.3} pdf_ms={:.3} arithmetic_ms={:.3} total_ms={:.3}",
            frame_length,
            lm_elapsed,
            pdf_elapsed,
            arithmetic_elapsed,
            (done - started).as_secs_f64() * 1000.0,
        );
    }
    Ok(payload)
}

/// Encodes and decodes one LM chunk without container framing.
///
/// This interface exists for qualification. The returned entropy bytes exclude
/// the optional four-byte normalization scale at the start of the payload.
pub fn encode_lm_chunk_evidence(
    lm_codec: &mut dyn LmCodec,
    codes: &Array3<i64>,
    scale: &Array2<f32>,
) -> Result<LmChunkEvidence> {
    let shape = codes.shape();
    if shape.len() != 3 || shape[0] != 1 {
        bail!(
            "qualification codes must have shape [1, codebooks, frames], got {:?}",
            shape,
        );
    }
    let frame_length = shape[2];
    if frame_length == 0 {
        bail!("qualification codes must contain at least one frame");
    }
    let meta = lm_codec.metadata().clone();
    if shape[1] != meta.num_codebooks {
        bail!(
            "qualification codebook count {} does not match LM count {}",
            shape[1],
            meta.num_codebooks,
        );
    }
    if scale.shape() != [1, 1] {
        bail!(
            "qualification scale must have shape [1, 1], got {:?}",
            scale.shape(),
        );
    }

    let payload = encode_lm_chunk_payload(
        lm_codec,
        codes,
        scale,
        0,
        frame_length,
        DEFAULT_FP_SCALE,
        DEFAULT_MIN_RANGE,
        1.0,
    )?;
    let entropy_offset = if meta.normalize { 4 } else { 0 };
    if payload.len() < entropy_offset {
        bail!("LM payload is shorter than its normalization scale");
    }
    let entropy = payload[entropy_offset..].to_vec();
    let (recovered_codes, recovered_scale) = decode_lm_chunk_codes(
        lm_codec,
        &meta,
        &payload,
        frame_length,
        DEFAULT_FP_SCALE,
        DEFAULT_MIN_RANGE,
        1.0,
    )?;
    if recovered_codes != *codes {
        bail!("LM qualification round trip changed one or more code symbols");
    }
    if meta.normalize && recovered_scale[[0, 0]].to_bits() != scale[[0, 0]].to_bits() {
        bail!("LM qualification round trip changed the normalization scale bits");
    }

    Ok(LmChunkEvidence {
        payload,
        entropy,
        recovered_codes,
        recovered_scale,
    })
}

fn decode_lm_chunk_payload(
    codec: &mut dyn FrameCodec,
    lm_codec: &mut dyn LmCodec,
    model_meta: &OnnxFrameBundleMetadata,
    metadata: &EcdcMetadata,
    payload: &[u8],
    this_len: usize,
    frame_length: usize,
) -> Result<Array3<f32>> {
    let profile_enabled = std::env::var_os("ENCODEC_RS_PROFILE").is_some();
    let started = profile_enabled.then(Instant::now);
    let (codes, scale) = decode_lm_chunk_codes(
        lm_codec,
        model_meta,
        payload,
        frame_length,
        metadata.fp_scale,
        metadata.min_range,
        metadata.lm_tau.unwrap_or(1.0) as f64,
    )?;
    let decoded = decode_codes(codec, &codes, &scale, this_len)?;
    if let Some(started) = started {
        eprintln!(
            "decode_lm_chunk_payload frame_length={} total_ms={:.3}",
            frame_length,
            started.elapsed().as_secs_f64() * 1000.0,
        );
    }
    Ok(decoded)
}

fn decode_lm_chunk_codes(
    lm_codec: &mut dyn LmCodec,
    model_meta: &OnnxFrameBundleMetadata,
    payload: &[u8],
    frame_length: usize,
    fp_scale: i64,
    min_range: i64,
    lm_tau: f64,
) -> Result<(Array3<i64>, Array2<f32>)> {
    let profile_enabled = std::env::var_os("ENCODEC_RS_PROFILE").is_some();
    let mut cursor = Cursor::new(payload);
    let scale = if model_meta.normalize {
        let bytes = read_exactly(&mut cursor, 4)?;
        Array2::from_shape_vec(
            (1, 1),
            vec![f32::from_be_bytes(bytes.try_into().expect("slice length"))],
        )
        .expect("shape")
    } else {
        Array2::from_shape_vec((1, 1), vec![1.0_f32]).expect("shape")
    };
    let remaining = payload.len().saturating_sub(cursor.position() as usize);
    let encoded = read_exactly(&mut cursor, remaining)?;
    let mut decoder = ArithmeticDecoder::new(encoded, ARITHMETIC_TOTAL_RANGE_BITS)?;
    let mut codes = Array3::<i64>::zeros((1, model_meta.num_codebooks, frame_length));
    let mut states = lm_codec.initial_states(1)?;
    let mut offset = 0_i64;
    let mut input = Array3::<i64>::zeros((1, model_meta.num_codebooks, 1));
    let mut scratch = ProbabilityScratch::default();
    let mut cdf_scratch = CdfScratch::default();
    let lm_logit_step = lm_codec.metadata().lm_entropy_logit_step();
    let lm_window_frame_length = lm_codec.lm_window_frame_length().max(1);
    if frame_length > lm_window_frame_length {
        bail!(
            "chunk frame length {} exceeds LM positional capacity {}",
            frame_length,
            lm_window_frame_length,
        );
    }
    let mut lm_elapsed = 0.0_f64;
    let mut pdf_elapsed = 0.0_f64;
    let mut arithmetic_elapsed = 0.0_f64;

    for t in 0..frame_length {
        let lm_started = profile_enabled.then(Instant::now);
        let (logits, next_offset, next_states) =
            lm_codec.forward_logits(&input, offset, &states)?;
        if let Some(lm_started) = lm_started {
            lm_elapsed += lm_started.elapsed().as_secs_f64() * 1000.0;
        }

        let pdf_started = profile_enabled.then(Instant::now);
        let pdf = probability_columns_from_logits(
            &logits,
            lm_tau,
            lm_logit_step,
            fp_scale,
            &mut scratch,
        )?;
        if let Some(pdf_started) = pdf_started {
            pdf_elapsed += pdf_started.elapsed().as_secs_f64() * 1000.0;
        }

        let arithmetic_started = profile_enabled.then(Instant::now);
        let symbols = decoder.pull_symbols_with_scratch(
            pdf,
            lm_codec.metadata().lm_cardinality(),
            model_meta.num_codebooks,
            fp_scale,
            min_range,
            &mut cdf_scratch,
        )?;
        if let Some(arithmetic_started) = arithmetic_started {
            arithmetic_elapsed += arithmetic_started.elapsed().as_secs_f64() * 1000.0;
        }

        for codebook in 0..model_meta.num_codebooks {
            let value = symbols[codebook] as i64;
            codes[[0, codebook, t]] = value;
            input[[0, codebook, 0]] = value + 1;
        }
        states = next_states;
        offset = next_offset;
    }

    if std::env::var_os("ENCODEC_RS_PROFILE").is_some() {
        eprintln!(
            "decode_lm_chunk_codes frame_length={} lm_ms={:.3} pdf_ms={:.3} arithmetic_ms={:.3}",
            frame_length, lm_elapsed, pdf_elapsed, arithmetic_elapsed,
        );
    }
    Ok((codes, scale))
}

fn decode_codes(
    codec: &mut dyn FrameCodec,
    codes: &Array3<i64>,
    scale: &Array2<f32>,
    this_len: usize,
) -> Result<Array3<f32>> {
    let profile_enabled = std::env::var_os("ENCODEC_RS_PROFILE").is_some();
    let frame_started = profile_enabled.then(Instant::now);
    let decoded = codec.decode_frame(codes, scale)?;
    if let Some(frame_started) = frame_started {
        eprintln!(
            "decode_codes batch={} frame_decode_ms={:.3}",
            codes.shape()[0],
            frame_started.elapsed().as_secs_f64() * 1000.0,
        );
    }

    let decoded_shape = decoded.shape();
    if decoded_shape.len() != 3 || decoded_shape[0] != 1 || decoded_shape[2] < this_len {
        bail!(
            "decoded audio shape {:?} cannot satisfy requested length {}",
            decoded_shape,
            this_len
        );
    }

    let trim_started = profile_enabled.then(Instant::now);
    let channels = decoded.shape()[1];
    let mut trimmed = Array3::<f32>::zeros((1, channels, this_len));
    for channel in 0..channels {
        for index in 0..this_len {
            trimmed[[0, channel, index]] = decoded[[0, channel, index]];
        }
    }
    if let Some(trim_started) = trim_started {
        eprintln!(
            "decode_codes batch={} trim_ms={:.3}",
            codes.shape()[0],
            trim_started.elapsed().as_secs_f64() * 1000.0,
        );
    }
    Ok(trimmed)
}

fn encode_segment_batches_with_size(
    audio: &Array3<f32>,
    meta: &OnnxFrameBundleMetadata,
    batch_size: usize,
    chunk_layout: EcdcChunkLayout,
    fixed_lm_frame_length: Option<usize>,
) -> Result<Vec<(Vec<usize>, Array3<f32>)>> {
    let total_samples = audio.shape()[2];
    let starts = segment_starts(total_samples, chunk_layout.stride);
    let context = fixed_context_samples(chunk_layout.samples, chunk_layout.stride)?;
    let batch_size = batch_size.max(1);
    let mut batches = Vec::new();
    for offsets in starts.chunks(batch_size) {
        let mut frame_lengths = Vec::with_capacity(offsets.len());
        let mut batch = Array3::<f32>::zeros((offsets.len(), meta.channels, chunk_layout.samples));
        for (batch_index, offset) in offsets.iter().copied().enumerate() {
            let owned_len = (total_samples - offset).min(chunk_layout.stride);
            let frame_length = fixed_lm_frame_length.unwrap_or_else(|| {
                if context.is_some() {
                    meta.frame_length
                } else {
                    segment_frame_length(owned_len, meta.segment_samples, meta.frame_length)
                }
            });
            frame_lengths.push(frame_length);
            let context = context.unwrap_or(0);
            for channel in 0..meta.channels {
                for index in 0..chunk_layout.samples {
                    let src_index = offset as isize - context as isize + index as isize;
                    let value = if src_index >= 0 && (src_index as usize) < total_samples {
                        audio[[0, channel, src_index as usize]]
                    } else {
                        0.0
                    };
                    batch[[batch_index, channel, index]] = value;
                }
            }
        }
        batches.push((frame_lengths, batch));
    }
    Ok(batches)
}

fn frame_encode_batch_size() -> usize {
    std::env::var("ENCODEC_RS_FRAME_BATCH")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(8)
}

fn probability_columns_from_logits<'a>(
    logits: &Array4<f32>,
    lm_tau: f64,
    logit_step: f64,
    fp_scale: i64,
    scratch: &'a mut ProbabilityScratch,
) -> Result<&'a [f64]> {
    let shape = logits.shape();
    if shape.len() != 4 || shape[0] != 1 {
        bail!(
            "LM logits must have shape [1, card, codebooks, steps], got {:?}",
            shape
        );
    }
    let cardinality = shape[1];
    let codebooks = shape[2];
    let steps = shape[3];
    let logits = logits
        .as_slice()
        .ok_or_else(|| anyhow::anyhow!("LM logits must use contiguous storage"))?;
    probability_columns_from_flat_logits(
        logits,
        cardinality,
        codebooks,
        steps,
        ProbabilityParameters {
            tau: lm_tau,
            logit_step,
            fp_scale,
        },
        scratch,
    )
}

pub fn deterministic_pdf_from_logits(
    logits: &Array4<f32>,
    lm_tau: f64,
    logit_step: f64,
    fp_scale: i64,
) -> Result<Vec<f64>> {
    let mut scratch = ProbabilityScratch::default();
    Ok(
        probability_columns_from_logits(logits, lm_tau, logit_step, fp_scale, &mut scratch)?
            .to_vec(),
    )
}

/// Copies one decoded model window into its final owned sample range.
fn copy_owned_segment_into(
    frame: &Array3<f32>,
    context: usize,
    owned_len: usize,
    output: &mut Array3<f32>,
    output_offset: usize,
) -> Result<()> {
    if frame.shape()[0] != 1 || output.shape()[0] != 1 {
        bail!("decoded audio must contain one batch");
    }
    if frame.shape()[1] != output.shape()[1] {
        bail!(
            "decoded channel count {} does not match output channel count {}",
            frame.shape()[1],
            output.shape()[1],
        );
    }
    let decoded_len = frame.shape()[2];
    let source_end = context
        .checked_add(owned_len)
        .ok_or_else(|| anyhow::anyhow!("decoded sample range overflow"))?;
    if source_end > decoded_len {
        bail!(
            "decoded model output is too short: context={} owned_len={} decoded={}",
            context,
            owned_len,
            decoded_len,
        );
    }
    let output_len = output.shape()[2];
    let output_end = output_offset
        .checked_add(owned_len)
        .ok_or_else(|| anyhow::anyhow!("output sample range overflow"))?;
    if output_end > output_len {
        bail!(
            "decoded owned range exceeds output: offset={} owned_len={} output={}",
            output_offset,
            owned_len,
            output_len,
        );
    }

    let channels = frame.shape()[1];
    let source = frame
        .as_slice()
        .ok_or_else(|| anyhow::anyhow!("decoded model output is not contiguous"))?;
    let destination = output
        .as_slice_mut()
        .ok_or_else(|| anyhow::anyhow!("decoded audio output is not contiguous"))?;
    for channel in 0..channels {
        let source_start = channel * decoded_len + context;
        let destination_start = channel * output_len + output_offset;
        destination[destination_start..destination_start + owned_len]
            .copy_from_slice(&source[source_start..source_start + owned_len]);
    }
    Ok(())
}

fn trim_audio_to_length(
    reconstructed: Array3<f32>,
    channels: usize,
    audio_length: usize,
) -> Result<Array3<f32>> {
    if reconstructed.shape()[0] != 1 || reconstructed.shape()[1] != channels {
        bail!(
            "decoded audio shape {:?} does not match one batch and {} channels",
            reconstructed.shape(),
            channels,
        );
    }
    let reconstructed_len = reconstructed.shape()[2];
    if reconstructed_len < audio_length {
        bail!(
            "decoded audio is too short: expected {} samples, got {}",
            audio_length,
            reconstructed_len,
        );
    }
    if reconstructed_len == audio_length {
        return Ok(reconstructed);
    }

    let source = reconstructed
        .as_slice()
        .ok_or_else(|| anyhow::anyhow!("decoded audio is not contiguous"))?;
    let mut trimmed = Array3::<f32>::zeros((1, channels, audio_length));
    let destination = trimmed
        .as_slice_mut()
        .ok_or_else(|| anyhow::anyhow!("trimmed audio is not contiguous"))?;
    for channel in 0..channels {
        let source_start = channel * reconstructed_len;
        let destination_start = channel * audio_length;
        destination[destination_start..destination_start + audio_length]
            .copy_from_slice(&source[source_start..source_start + audio_length]);
    }
    Ok(trimmed)
}

fn linear_overlap_add(frames: &[Array3<f32>], stride: usize) -> Array3<f32> {
    if frames.is_empty() {
        return Array3::<f32>::zeros((1, 0, 0));
    }

    let channels = frames[0].shape()[1];
    let frame_length = frames[0].shape()[2];
    let total_size = stride * (frames.len() - 1) + frame_length;
    let mut output = Array3::<f32>::zeros((1, channels, total_size));
    let mut sum_weight = vec![0.0_f32; total_size];
    let weight = triangle_weight(frame_length);

    let mut offset = 0usize;
    for frame in frames {
        let frame_len = frame.shape()[2];
        for index in 0..frame_len {
            let w = weight[index];
            sum_weight[offset + index] += w;
            for channel in 0..channels {
                output[[0, channel, offset + index]] += frame[[0, channel, index]] * w;
            }
        }
        offset += stride;
    }

    for index in 0..total_size {
        let denom = sum_weight[index];
        if denom > 0.0 {
            for channel in 0..channels {
                output[[0, channel, index]] /= denom;
            }
        }
    }
    output
}

fn triangle_weight(frame_length: usize) -> Vec<f32> {
    (0..frame_length)
        .map(|index| {
            let t = (index + 1) as f32 / (frame_length + 1) as f32;
            0.5 - (t - 0.5).abs()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    struct TracingLm {
        meta: OnnxFrameBundleMetadata,
        capacity: usize,
        calls: Vec<(i64, Vec<i64>)>,
    }

    struct EvidenceFrameCodec {
        meta: OnnxFrameBundleMetadata,
    }

    impl FrameCodec for EvidenceFrameCodec {
        fn metadata(&self) -> &OnnxFrameBundleMetadata {
            &self.meta
        }

        fn encode_frame(&mut self, audio: &Array3<f32>) -> Result<(Array3<i64>, Array2<f32>)> {
            let frame_length = segment_frame_length(
                audio.shape()[2],
                self.meta.segment_samples,
                self.meta.frame_length,
            );
            let mut codes =
                Array3::<i64>::zeros((audio.shape()[0], self.meta.num_codebooks, frame_length));
            for batch in 0..audio.shape()[0] {
                for codebook in 0..self.meta.num_codebooks {
                    for frame in 0..frame_length {
                        codes[[batch, codebook, frame]] =
                            (batch * 100 + codebook * 10 + frame) as i64;
                    }
                }
            }
            Ok((codes, Array2::from_elem((audio.shape()[0], 1), 1.0)))
        }

        fn decode_frame(
            &mut self,
            _codes: &Array3<i64>,
            _scale: &Array2<f32>,
        ) -> Result<Array3<f32>> {
            Ok(Array3::zeros((
                1,
                self.meta.channels,
                self.meta.segment_samples,
            )))
        }
    }

    impl TracingLm {
        fn new(capacity: usize) -> Self {
            Self {
                meta: OnnxFrameBundleMetadata {
                    schema_version: 1,
                    model_name: "trace_lm".into(),
                    bandwidth_kbps: 1.0,
                    sample_rate: 48_000,
                    channels: 1,
                    segment_samples: 4,
                    segment_stride: 4,
                    normalize: true,
                    num_codebooks: 2,
                    frame_length: 4,
                    bits_per_codebook: Some(2),
                    codebook_cardinality: Some(4),
                    encode_model: "unused".into(),
                    decode_model: "unused".into(),
                    lm_quant_weight_model: None,
                    lm_dim: Some(4),
                    lm_num_layers: Some(1),
                    lm_past_context: Some(0),
                    lm_logit_step: Some(1.0),
                    lm_entropy_logit_step: Some(1.0),
                    lm_cardinality: Some(4),
                    opset_version: 17,
                },
                capacity,
                calls: Vec::new(),
            }
        }
    }

    impl LmCodec for TracingLm {
        fn metadata(&self) -> &OnnxFrameBundleMetadata {
            &self.meta
        }

        fn lm_window_frame_length(&self) -> usize {
            self.capacity
        }

        fn initial_states(&self, _batch: usize) -> Result<Vec<Array3<f32>>> {
            Ok(Vec::new())
        }

        fn forward_logits(
            &mut self,
            indices: &Array3<i64>,
            offset: i64,
            _states: &[Array3<f32>],
        ) -> Result<(Array4<f32>, i64, Vec<Array3<f32>>)> {
            self.calls.push((offset, indices.iter().copied().collect()));
            Ok((
                Array4::zeros((1, self.meta.lm_cardinality(), self.meta.num_codebooks, 1)),
                offset + 1,
                Vec::new(),
            ))
        }
    }

    fn fixed_1333ms_meta() -> OnnxFrameBundleMetadata {
        OnnxFrameBundleMetadata {
            schema_version: 1,
            model_name: "encodec_48khz_test".into(),
            bandwidth_kbps: 12.0,
            sample_rate: 48_000,
            channels: 1,
            segment_samples: 64_960,
            segment_stride: 64_000,
            normalize: true,
            num_codebooks: 8,
            frame_length: 203,
            bits_per_codebook: Some(10),
            codebook_cardinality: Some(1024),
            encode_model: "encode_frame.onnx".into(),
            decode_model: "decode_frame.onnx".into(),
            lm_quant_weight_model: Some("lm_weights_q8.bin".into()),
            lm_dim: Some(128),
            lm_num_layers: Some(1),
            lm_past_context: Some(0),
            lm_logit_step: Some(1.0 / 64.0),
            lm_entropy_logit_step: Some(2.1),
            lm_cardinality: Some(1024),
            opset_version: 17,
        }
    }

    fn variable_tail_meta() -> OnnxFrameBundleMetadata {
        let mut meta = fixed_1333ms_meta();
        meta.segment_samples = 10;
        meta.segment_stride = 8;
        meta.frame_length = 5;
        meta.num_codebooks = 2;
        meta.lm_cardinality = Some(4);
        meta.codebook_cardinality = Some(4);
        meta.bits_per_codebook = Some(2);
        meta
    }

    /// 2 full owned chunks (64,000 each) plus a final partial chunk of
    /// 2,000 owned samples: 130,000 logical samples total.
    fn indexed_audio(total_samples: usize) -> Array3<f32> {
        let mut audio = Array3::<f32>::zeros((1, 1, total_samples));
        for index in 0..total_samples {
            audio[[0, 0, index]] = index as f32;
        }
        audio
    }

    #[test]
    fn encode_segment_batches_builds_context_window_for_every_chunk() {
        let meta = fixed_1333ms_meta();
        let stride = meta.segment_stride;
        let window = meta.segment_samples;
        let context = 480usize;
        let total_samples = stride * 2 + 2_000;
        let audio = indexed_audio(total_samples);
        let layout = EcdcChunkLayout {
            samples: window,
            stride,
        };

        let batches = encode_segment_batches_with_size(&audio, &meta, 8, layout, None).unwrap();
        assert_eq!(batches.len(), 1);
        let (frame_lengths, batch) = &batches[0];
        assert_eq!(batch.shape(), &[3, 1, window]);
        // Every chunk, including the final partial one, encodes the full
        // bundle frame length -- never a reduced owned-region count.
        assert_eq!(frame_lengths, &vec![meta.frame_length; 3]);

        // Chunk 0: start of track, left context zero-filled.
        for index in 0..context {
            assert_eq!(batch[[0, 0, index]], 0.0);
        }
        for index in 0..stride {
            assert_eq!(batch[[0, 0, context + index]], index as f32);
        }
        for index in 0..context {
            assert_eq!(
                batch[[0, 0, context + stride + index]],
                (stride + index) as f32
            );
        }

        // Chunk 1: real source samples on both sides.
        for index in 0..context {
            assert_eq!(batch[[1, 0, index]], (stride - context + index) as f32);
        }
        for index in 0..stride {
            assert_eq!(batch[[1, 0, context + index]], (stride + index) as f32);
        }
        for index in 0..context {
            assert_eq!(
                batch[[1, 0, context + stride + index]],
                (2 * stride + index) as f32
            );
        }

        // Chunk 2: final partial owned region (2,000 real samples), the
        // rest of the owned region and the right context are zero-filled
        // past the end of the track.
        let owned_len = total_samples - 2 * stride;
        for index in 0..context {
            assert_eq!(batch[[2, 0, index]], (2 * stride - context + index) as f32);
        }
        for index in 0..owned_len {
            assert_eq!(batch[[2, 0, context + index]], (2 * stride + index) as f32);
        }
        for index in owned_len..stride {
            assert_eq!(batch[[2, 0, context + index]], 0.0);
        }
        for index in 0..context {
            assert_eq!(batch[[2, 0, context + stride + index]], 0.0);
        }
    }

    #[test]
    fn frame_evidence_preserves_true_variable_tail_input() {
        let mut codec = EvidenceFrameCodec {
            meta: variable_tail_meta(),
        };
        let audio = indexed_audio(18);
        let evidence = encode_audio_frame_evidence(&mut codec, &audio, 8, true).unwrap();

        assert_eq!(evidence.len(), 3);
        assert_eq!(evidence[0].offset_samples, 0);
        assert_eq!(evidence[0].owned_samples, 8);
        assert_eq!(evidence[0].model_input.shape(), &[1, 1, 10]);
        assert_eq!(evidence[0].codes.shape(), &[1, 2, 5]);
        assert_eq!(evidence[1].offset_samples, 8);
        assert_eq!(evidence[1].model_input[[0, 0, 0]], 8.0);
        assert_eq!(evidence[1].model_input[[0, 0, 9]], 17.0);
        assert_eq!(evidence[2].offset_samples, 16);
        assert_eq!(evidence[2].owned_samples, 2);
        assert_eq!(evidence[2].model_input.shape(), &[1, 1, 2]);
        assert_eq!(evidence[2].model_input[[0, 0, 0]], 16.0);
        assert_eq!(evidence[2].model_input[[0, 0, 1]], 17.0);
        assert_eq!(evidence[2].codes.shape(), &[1, 2, 1]);
    }

    #[test]
    fn frame_evidence_pads_tail_when_variable_tail_is_disabled() {
        let mut codec = EvidenceFrameCodec {
            meta: variable_tail_meta(),
        };
        let audio = indexed_audio(18);
        let evidence = encode_audio_frame_evidence(&mut codec, &audio, 8, false).unwrap();

        assert_eq!(evidence[2].model_input.shape(), &[1, 1, 10]);
        assert_eq!(evidence[2].model_input[[0, 0, 0]], 16.0);
        assert_eq!(evidence[2].model_input[[0, 0, 1]], 17.0);
        assert_eq!(evidence[2].model_input[[0, 0, 2]], 0.0);
        assert_eq!(evidence[2].codes.shape(), &[1, 2, 5]);
    }

    #[test]
    fn owned_segments_copy_directly_without_overlap() {
        let context = 480usize;
        let stride = 64_000usize;
        let window = 64_960usize;
        let owned_lens = [stride, stride, 2_000usize];

        let frames: Vec<Array3<f32>> = owned_lens
            .iter()
            .enumerate()
            .map(|(frame_index, _)| {
                let mut frame = Array3::<f32>::zeros((1, 1, window));
                for index in 0..window {
                    frame[[0, 0, index]] = (frame_index * 1_000_000 + index) as f32;
                }
                frame
            })
            .collect();

        let audio_length: usize = owned_lens.iter().sum();
        let mut concatenated = Array3::<f32>::zeros((1, 1, audio_length));
        let mut output_offset = 0;
        for (frame, owned_len) in frames.iter().zip(owned_lens) {
            copy_owned_segment_into(frame, context, owned_len, &mut concatenated, output_offset)
                .unwrap();
            output_offset += owned_len;
        }
        assert_eq!(concatenated.shape(), &[1, 1, audio_length]);

        // No overlap-add, no duplicated/missing samples: each owned region
        // lands exactly at its logical stride offset with no blending.
        assert_eq!(concatenated[[0, 0, 0]], context as f32);
        assert_eq!(
            concatenated[[0, 0, stride - 1]],
            (context + stride - 1) as f32
        );
        assert_eq!(concatenated[[0, 0, stride]], (1_000_000 + context) as f32);
        assert_eq!(
            concatenated[[0, 0, 2 * stride]],
            (2_000_000 + context) as f32
        );
        assert_eq!(
            concatenated[[0, 0, audio_length - 1]],
            (2_000_000 + context + 1_999) as f32
        );

        // Decode reconstruction returns the exact cropped chunks. A caller can
        // opt into `encodec_rs::seam` after it receives adjacent chunks.
        assert_eq!(
            concatenated[[0, 0, stride - 1]],
            (context + stride - 1) as f32
        );
        assert_eq!(concatenated[[0, 0, stride]], (1_000_000 + context) as f32);
    }

    #[test]
    fn owned_segment_copy_rejects_short_decoded_output() {
        let frame = Array3::<f32>::zeros((1, 1, 500));
        let mut output = Array3::<f32>::zeros((1, 1, 64_000));
        assert!(copy_owned_segment_into(&frame, 480, 64_000, &mut output, 0).is_err());
    }

    #[test]
    fn lm_mapping_uses_bos_then_previous_codes_plus_one() {
        let mut lm = TracingLm::new(4);
        let mut codes = Array3::<i64>::zeros((1, 2, 4));
        codes[[0, 0, 0]] = 0;
        codes[[0, 1, 0]] = 3;
        codes[[0, 0, 1]] = 2;
        codes[[0, 1, 1]] = 1;
        codes[[0, 0, 2]] = 1;
        codes[[0, 1, 2]] = 2;
        codes[[0, 0, 3]] = 3;
        codes[[0, 1, 3]] = 0;
        let scale = f32::from_bits(0x3f12_3456);
        let scales = Array2::from_elem((1, 1), scale);

        let payload = encode_lm_chunk_payload(
            &mut lm,
            &codes,
            &scales,
            0,
            4,
            DEFAULT_FP_SCALE,
            DEFAULT_MIN_RANGE,
            1.0,
        )
        .unwrap();

        assert_eq!(&payload[..4], &scale.to_be_bytes());
        assert_eq!(
            lm.calls,
            vec![
                (0, vec![0, 0]),
                (1, vec![1, 4]),
                (2, vec![3, 2]),
                (3, vec![2, 3]),
            ]
        );

        encode_lm_chunk_payload(
            &mut lm,
            &codes,
            &scales,
            0,
            1,
            DEFAULT_FP_SCALE,
            DEFAULT_MIN_RANGE,
            1.0,
        )
        .unwrap();
        assert_eq!(lm.calls.last(), Some(&(0, vec![0, 0])));
    }

    #[test]
    fn lm_chunk_evidence_recovers_codes_scale_and_entropy() {
        let mut lm = TracingLm::new(4);
        let mut codes = Array3::<i64>::zeros((1, 2, 4));
        codes[[0, 0, 0]] = 0;
        codes[[0, 1, 0]] = 3;
        codes[[0, 0, 1]] = 2;
        codes[[0, 1, 1]] = 1;
        codes[[0, 0, 2]] = 1;
        codes[[0, 1, 2]] = 2;
        codes[[0, 0, 3]] = 3;
        codes[[0, 1, 3]] = 0;
        let scale = Array2::from_elem((1, 1), f32::from_bits(0x3f12_3456));

        let evidence = encode_lm_chunk_evidence(&mut lm, &codes, &scale).unwrap();

        assert_eq!(&evidence.payload[..4], &scale[[0, 0]].to_be_bytes());
        assert_eq!(evidence.entropy, evidence.payload[4..]);
        assert_eq!(evidence.recovered_codes, codes);
        assert_eq!(
            evidence.recovered_scale[[0, 0]].to_bits(),
            scale[[0, 0]].to_bits(),
        );
    }

    #[test]
    fn lm_chunk_rejects_internal_reset_instead_of_inserting_bos() {
        let mut lm = TracingLm::new(2);
        let codes = Array3::<i64>::zeros((1, 2, 4));
        let scales = Array2::from_elem((1, 1), 1.0);
        let error = encode_lm_chunk_payload(
            &mut lm,
            &codes,
            &scales,
            0,
            4,
            DEFAULT_FP_SCALE,
            DEFAULT_MIN_RANGE,
            1.0,
        )
        .unwrap_err();

        assert!(error.to_string().contains("exceeds LM positional capacity"));
        assert!(lm.calls.is_empty());
    }
}
