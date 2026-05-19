use std::io::Cursor;
use std::time::Instant;

use anyhow::{bail, Result};
use ndarray::{Array2, Array3, Array4};

use crate::arithmetic::{ArithmeticDecoder, ArithmeticEncoder};
use crate::binary::{
    read_chunk_payload, read_ecdc_header, read_exactly, write_chunk, write_ecdc_header,
};
use crate::format::{segment_frame_length, segment_starts, validate_metadata};
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

#[derive(Default)]
struct ProbabilityScratch {
    pdf: Vec<f64>,
    quantized: Vec<f64>,
    probs: Vec<f64>,
}

impl ProbabilityScratch {
    fn prepare(&mut self, card: usize, columns: usize) {
        self.pdf.resize(card * columns, 0.0);
        self.quantized.resize(card, 0.0);
        self.probs.resize(card, 0.0);
    }
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
            emit,
        )
    })
}

pub fn encode_audio_to_ecdc_stream_with_options<F>(
    codec: &mut dyn FrameCodec,
    lm_codec: &mut dyn LmCodec,
    audio: &Array3<f32>,
    source: Option<&SourceAudioMetadata>,
    frame_batch_size: usize,
    chunk_crc: bool,
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
        &mut on_bytes,
    )
}

pub fn encode_ecdc_header_with_options(
    codec: &dyn FrameCodec,
    audio_length: usize,
    source: Option<&SourceAudioMetadata>,
    lm_hash: Option<String>,
) -> Result<Vec<u8>> {
    let metadata = EcdcMetadata::from_bundle(codec.metadata(), audio_length, source, lm_hash);
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

fn encode_audio_to_ecdc_impl(
    codec: &mut dyn FrameCodec,
    lm_codec: &mut dyn LmCodec,
    audio: &Array3<f32>,
    source: Option<&SourceAudioMetadata>,
    frame_batch_size: usize,
    chunk_crc: bool,
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
    let lm_hash = lm_codec
        .bitstream_lm_hash()
        .map(str::to_owned)
        .ok_or_else(|| anyhow::anyhow!("q8 LM runtime does not expose an LM hash"))?;
    let header = encode_ecdc_header_with_options(codec, shape[2], source, Some(lm_hash))?;
    emit(&header)?;

    for (batch_index, (frame_lengths, batch)) in
        encode_segment_batches_with_size(audio, &model_meta, frame_batch_size)
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
    if batch_shape[1] != model_meta.channels || batch_shape[2] != model_meta.segment_samples {
        bail!(
            "segment batch shape mismatch, expected [batch, {}, {}], got {:?}",
            model_meta.channels,
            model_meta.segment_samples,
            batch_shape
        );
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

    for (segment_index, frame_length) in frame_lengths.iter().copied().enumerate() {
        if frame_length == 0 || frame_length > model_meta.frame_length {
            bail!(
                "segment frame length {} is out of range for frame_length {}",
                frame_length,
                model_meta.frame_length
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
    let mut reader = Cursor::new(payload);
    let metadata: EcdcMetadata = read_ecdc_header(&mut reader)?;
    let bundle_meta = codec.metadata().clone();
    validate_metadata(&bundle_meta, &metadata)?;

    let mut frames = Vec::new();
    let starts = segment_starts(metadata.audio_length, bundle_meta.segment_stride.max(1));
    for offset in starts {
        let this_len = (metadata.audio_length - offset).min(bundle_meta.segment_samples);
        let frame_length = segment_frame_length(
            this_len,
            bundle_meta.segment_samples,
            bundle_meta.frame_length,
        );
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
        let chunk = read_chunk_payload(&mut reader, true)?;
        let frame = decode_lm_chunk_payload(
            codec,
            lm_codec,
            &bundle_meta,
            &metadata,
            &chunk,
            this_len,
            frame_length,
        )?;
        frames.push(frame);
    }

    let reconstructed = if frames.len() <= 1 {
        frames
            .into_iter()
            .next()
            .unwrap_or_else(|| Array3::<f32>::zeros((1, bundle_meta.channels, 0)))
    } else {
        linear_overlap_add(&frames, bundle_meta.segment_stride)
    };
    let mut trimmed = Array3::<f32>::zeros((1, bundle_meta.channels, metadata.audio_length));
    for channel in 0..bundle_meta.channels {
        for index in 0..metadata.audio_length {
            trimmed[[0, channel, index]] = reconstructed[[0, channel, index]];
        }
    }

    Ok(DecodedEcdcAudio {
        metadata,
        audio: trimmed,
    })
}

fn encode_lm_chunk_payload(
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
        encoder.push_pdf_symbols(
            pdf,
            meta.lm_cardinality(),
            meta.num_codebooks,
            &symbols,
            fp_scale,
            min_range,
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
    let mut codes = Array3::<i64>::zeros((1, model_meta.num_codebooks, model_meta.frame_length));
    let mut states = lm_codec.initial_states(1)?;
    let mut offset = 0_i64;
    let mut input = Array3::<i64>::zeros((1, model_meta.num_codebooks, 1));
    let mut scratch = ProbabilityScratch::default();
    let lm_tau = metadata.lm_tau.unwrap_or(1.0) as f64;
    let lm_logit_step = lm_codec.metadata().lm_entropy_logit_step();
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
            metadata.fp_scale,
            &mut scratch,
        )?;
        if let Some(pdf_started) = pdf_started {
            pdf_elapsed += pdf_started.elapsed().as_secs_f64() * 1000.0;
        }

        let arithmetic_started = profile_enabled.then(Instant::now);
        let symbols = decoder.pull_symbols(
            pdf,
            lm_codec.metadata().lm_cardinality(),
            model_meta.num_codebooks,
            metadata.fp_scale,
            metadata.min_range,
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

    let decoded = decode_codes(codec, &codes, &scale, this_len)?;
    if let Some(started) = started {
        eprintln!(
            "decode_lm_chunk_payload frame_length={} lm_ms={:.3} pdf_ms={:.3} arithmetic_ms={:.3} total_ms={:.3}",
            frame_length,
            lm_elapsed,
            pdf_elapsed,
            arithmetic_elapsed,
            started.elapsed().as_secs_f64() * 1000.0,
        );
    }
    Ok(decoded)
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
) -> Vec<(Vec<usize>, Array3<f32>)> {
    let total_samples = audio.shape()[2];
    let starts = segment_starts(total_samples, meta.segment_stride.max(1));
    let batch_size = batch_size.max(1);
    let mut batches = Vec::new();
    for offsets in starts.chunks(batch_size) {
        let mut frame_lengths = Vec::with_capacity(offsets.len());
        let mut batch = Array3::<f32>::zeros((offsets.len(), meta.channels, meta.segment_samples));
        for (batch_index, offset) in offsets.iter().copied().enumerate() {
            let copy_len = (total_samples - offset).min(meta.segment_samples);
            let frame_length =
                segment_frame_length(copy_len, meta.segment_samples, meta.frame_length);
            frame_lengths.push(frame_length);
            for channel in 0..meta.channels {
                for index in 0..copy_len {
                    batch[[batch_index, channel, index]] = audio[[0, channel, offset + index]];
                }
            }
        }
        batches.push((frame_lengths, batch));
    }
    batches
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
    let card = shape[1];
    let codebooks = shape[2];
    let steps = shape[3];
    let columns = codebooks * steps;
    scratch.prepare(card, columns);
    let pdf = &mut scratch.pdf;
    let quantized = &mut scratch.quantized;
    let probs = &mut scratch.probs;
    let uniform = 1.0 / card as f64;
    let near_pdf_threshold = 0.25 / fp_scale as f64;

    for step in 0..steps {
        for codebook in 0..codebooks {
            let mut max_value = f64::NEG_INFINITY;
            let mut min_value = f64::INFINITY;
            for bin in 0..card {
                let raw = logits[[0, bin, codebook, step]] as f64 / lm_tau;
                let quantized_value = quantize_logit(raw, logit_step);
                quantized[bin] = quantized_value;
                max_value = max_value.max(quantized_value);
                min_value = min_value.min(quantized_value);
            }

            let mut denom = 0.0_f64;
            for bin in 0..card {
                let value = (quantized[bin] - max_value).exp();
                probs[bin] = value;
                denom += value;
            }
            if !denom.is_finite() || denom <= 0.0 {
                let column = step * codebooks + codebook;
                for bin in 0..card {
                    pdf[bin * columns + column] = uniform;
                }
                continue;
            }
            let mut max_pdf = 0.0_f64;
            let mut min_pdf = f64::INFINITY;
            for prob in probs.iter_mut() {
                *prob /= denom;
                max_pdf = max_pdf.max(*prob);
                min_pdf = min_pdf.min(*prob);
            }
            let near_uniform = (max_value - min_value) <= (2.0 * logit_step)
                || (max_pdf - min_pdf) <= near_pdf_threshold;
            let column = step * codebooks + codebook;
            for bin in 0..card {
                pdf[bin * columns + column] = if near_uniform { uniform } else { probs[bin] };
            }
        }
    }

    Ok(&pdf[..card * columns])
}

pub fn deterministic_pdf_from_logits(
    logits: &Array4<f32>,
    lm_tau: f64,
    logit_step: f64,
    fp_scale: i64,
) -> Result<Vec<f64>> {
    let mut scratch = ProbabilityScratch::default();
    probability_columns_from_logits(logits, lm_tau, logit_step, fp_scale, &mut scratch)?;
    Ok(scratch.pdf)
}

fn quantize_logit(value: f64, step: f64) -> f64 {
    let eps = 2_f64.powi(-40);
    let y = value / step;
    (y + 0.5 - eps).floor() * step
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
