#![cfg(feature = "onnx")]

use std::env;
use std::io::Cursor;
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{bail, Context, Result};
use encodec_rs::binary::{read_chunk_payload, read_ecdc_header};
use encodec_rs::ecdc::{encode_audio_to_ecdc_stream_with_options, EcdcMetadata, LmCodec};
use encodec_rs::format::{segment_frame_length, segment_starts};
use encodec_rs::metadata::OnnxFrameBundleMetadata;
use encodec_rs::onnx::{CoreMlComputeUnits, ExecutionTarget, OnnxFrameCodec, OnnxLmCodec};
use hound::{SampleFormat, WavReader};
use ndarray::Array3;

#[derive(Clone, Copy)]
struct Case {
    label: &'static str,
    bandwidth: &'static str,
    bundle_suffix: &'static str,
    display_chunk_ms: Option<f64>,
}

#[derive(Debug)]
struct ChunkMetric {
    index: usize,
    offset: usize,
    samples: usize,
    duration_ms: f64,
    frame_length: usize,
    chunk_bytes: usize,
    payload_bytes: usize,
    cumulative_chunk_bytes: usize,
}

#[derive(Debug)]
struct SummaryRow {
    label: &'static str,
    bandwidth: &'static str,
    bundle_suffix: &'static str,
    display_chunk_ms: Option<f64>,
    chunks: usize,
    encode_seconds: f64,
    audio_seconds: f64,
    average_chunk_bytes: f64,
    chunk_total_bytes: usize,
    payload_total_bytes: usize,
    file_total_bytes: usize,
    header_bytes: usize,
    segment_samples: usize,
    segment_stride: usize,
    bundle_frame_length: usize,
    num_codebooks: usize,
}

#[test]
#[ignore]
fn ecdc_fixed_bundle_chunk_matrix_verbose() -> Result<()> {
    let bundle_root = env::var("ENCODEC_RS_MATRIX_BUNDLE_ROOT")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("onnx-bundles"));
    let wav_path = env::var("ENCODEC_RS_MATRIX_WAV")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("testdata/westside_4s_48khz_stereo.wav"));

    if !wav_path.exists() {
        eprintln!(
            "skipping ecdc_fixed_bundle_chunk_matrix_verbose because {} does not exist",
            wav_path.display()
        );
        return Ok(());
    }

    let cases = [
        Case {
            label: "1000ms",
            bandwidth: "6kbps",
            bundle_suffix: "encodec_48khz_6kbps_1000ms",
            display_chunk_ms: Some(1000.0),
        },
        Case {
            label: "1333.3ms",
            bandwidth: "6kbps",
            bundle_suffix: "encodec_48khz_6kbps_1333ms",
            display_chunk_ms: Some(1333.3),
        },
        Case {
            label: "1800.0ms",
            bandwidth: "6kbps",
            bundle_suffix: "encodec_48khz_6kbps_1800ms",
            display_chunk_ms: Some(1800.0),
        },
        Case {
            label: "1000ms",
            bandwidth: "12kbps",
            bundle_suffix: "encodec_48khz_12kbps_1000ms",
            display_chunk_ms: Some(1000.0),
        },
        Case {
            label: "1333.3ms",
            bandwidth: "12kbps",
            bundle_suffix: "encodec_48khz_12kbps_1333ms",
            display_chunk_ms: Some(1333.3),
        },
        Case {
            label: "1800.0ms",
            bandwidth: "12kbps",
            bundle_suffix: "encodec_48khz_12kbps_1800ms",
            display_chunk_ms: Some(1800.0),
        },
    ];

    let mut summaries = Vec::new();
    let mut ran = 0usize;

    for case in cases {
        let bundle_dir = bundle_root.join(case.bundle_suffix);
        if !bundle_dir.join("bundle.json").exists() {
            eprintln!(
                "skipping case {} {} because {} does not exist",
                case.bandwidth,
                case.label,
                bundle_dir.join("bundle.json").display()
            );
            continue;
        }

        ran += 1;

        println!();
        println!(
            "case {} {} bundle={}",
            case.bandwidth,
            case.label,
            bundle_dir.display()
        );
        println!("{}", "=".repeat(140));

        let target = execution_target(&bundle_dir)?;
        let mut codec = OnnxFrameCodec::from_dir(&bundle_dir, target)?;
        let meta = codec.metadata().clone();
        let (audio, input_frames, input_sample_rate) = read_wav_f32(&wav_path, meta.channels)?;

        if input_sample_rate as usize != meta.sample_rate {
            bail!(
                "input WAV sample rate {} does not match bundle sample rate {}",
                input_sample_rate,
                meta.sample_rate
            );
        }

        let target = execution_target(&bundle_dir)?;
        let mut lm_codec = OnnxLmCodec::from_dir(&bundle_dir, target)?;
        let mut ecdc_bytes = Vec::new();
        let mut emitted_parts = 0usize;
        let started = Instant::now();

        encode_audio_to_ecdc_stream_with_options(
            &mut codec,
            &mut lm_codec as &mut dyn LmCodec,
            &audio,
            None,
            matrix_batch_size(),
            true,
            None,
            |bytes| {
                ecdc_bytes.extend_from_slice(bytes);
                emitted_parts += 1;
                Ok(())
            },
        )?;

        let encode_seconds = started.elapsed().as_secs_f64();
        let audio_seconds = input_frames as f64 / meta.sample_rate as f64;

        println!(
            "encoded bandwidth={} case={} parts={} bytes={} encode_s={:.6} audio_s={:.6} rtf={:.6} segment_samples={} segment_stride={} frame_length={} codebooks={}",
            case.bandwidth,
            case.label,
            emitted_parts,
            ecdc_bytes.len(),
            encode_seconds,
            audio_seconds,
            encode_seconds / audio_seconds,
            meta.segment_samples,
            meta.segment_stride,
            meta.frame_length,
            meta.num_codebooks
        );

        let inspected = inspect_ecdc_chunks(
            case.label,
            case.bandwidth,
            case.display_chunk_ms,
            &ecdc_bytes,
            &meta,
        )?;

        summaries.push(SummaryRow {
            label: case.label,
            bandwidth: case.bandwidth,
            bundle_suffix: case.bundle_suffix,
            display_chunk_ms: case.display_chunk_ms,
            chunks: inspected.metrics.len(),
            encode_seconds,
            audio_seconds,
            average_chunk_bytes: if inspected.metrics.is_empty() {
                0.0
            } else {
                inspected.chunk_total_bytes as f64 / inspected.metrics.len() as f64
            },
            chunk_total_bytes: inspected.chunk_total_bytes,
            payload_total_bytes: inspected.payload_total_bytes,
            file_total_bytes: ecdc_bytes.len(),
            header_bytes: inspected.header_bytes,
            segment_samples: meta.segment_samples,
            segment_stride: meta.segment_stride,
            bundle_frame_length: meta.frame_length,
            num_codebooks: meta.num_codebooks,
        });
    }

    if ran == 0 {
        eprintln!(
            "skipping ecdc_fixed_bundle_chunk_matrix_verbose because no fixed chunk bundles were found under {}",
            bundle_root.display()
        );
        return Ok(());
    }

    println!();
    println!("summary");
    println!("{}", "=".repeat(180));
    println!(
        "{:<8} {:<10} {:>10} {:>8} {:>12} {:>12} {:>12} {:>16} {:>16} {:>16} {:>12} {:>14} {:>14} {:>8} {:>10}  {}",
        "bw",
        "case",
        "chunk_ms",
        "chunks",
        "encode_s",
        "rtf",
        "avg_chunk",
        "chunk_total",
        "payload_total",
        "file_total",
        "header",
        "seg_samples",
        "seg_stride",
        "frames",
        "codebooks",
        "bundle"
    );

    for row in summaries {
        println!(
            "{:<8} {:<10} {:>10} {:>8} {:>12.6} {:>12.6} {:>12.1} {:>16} {:>16} {:>16} {:>12} {:>14} {:>14} {:>8} {:>10}  {}",
            row.bandwidth,
            row.label,
            format_chunk_ms(row.display_chunk_ms),
            row.chunks,
            row.encode_seconds,
            row.encode_seconds / row.audio_seconds,
            row.average_chunk_bytes,
            row.chunk_total_bytes,
            row.payload_total_bytes,
            row.file_total_bytes,
            row.header_bytes,
            row.segment_samples,
            row.segment_stride,
            row.bundle_frame_length,
            row.num_codebooks,
            row.bundle_suffix
        );
    }

    Ok(())
}

struct InspectedChunks {
    metrics: Vec<ChunkMetric>,
    chunk_total_bytes: usize,
    payload_total_bytes: usize,
    header_bytes: usize,
}

fn inspect_ecdc_chunks(
    label: &'static str,
    bandwidth: &'static str,
    display_chunk_ms: Option<f64>,
    ecdc_bytes: &[u8],
    meta: &OnnxFrameBundleMetadata,
) -> Result<InspectedChunks> {
    let mut reader = Cursor::new(ecdc_bytes);
    let metadata: EcdcMetadata = read_ecdc_header(&mut reader)?;
    let header_bytes = reader.position() as usize;

    let chunk_samples = metadata.chunk_samples.unwrap_or(meta.segment_samples);
    let chunk_stride = metadata.chunk_stride.unwrap_or(meta.segment_stride.max(1));

    if chunk_samples == 0 {
        bail!("case {} {} has invalid chunk_samples=0", bandwidth, label);
    }
    if chunk_stride == 0 {
        bail!("case {} {} has invalid chunk_stride=0", bandwidth, label);
    }

    println!(
        "metadata bandwidth={} case={} display_chunk_ms={} audio_length={} sample_rate={} chunk_samples={} chunk_stride={} chunk_duration_ms={:.6} frame_length={} num_codebooks={} header_bytes={}",
        bandwidth,
        label,
        format_chunk_ms(display_chunk_ms),
        metadata.audio_length,
        meta.sample_rate,
        chunk_samples,
        chunk_stride,
        chunk_samples as f64 * 1000.0 / meta.sample_rate as f64,
        meta.frame_length,
        meta.num_codebooks,
        header_bytes
    );

    println!(
        "{:>6} {:>12} {:>12} {:>12} {:>8} {:>14} {:>14} {:>14}",
        "chunk", "offset", "samples", "ms", "frames", "chunk_bytes", "payload_bytes", "cum_chunk"
    );

    let starts = segment_starts(metadata.audio_length, chunk_stride);
    let mut metrics = Vec::with_capacity(starts.len());
    let mut chunk_total_bytes = 0usize;
    let mut payload_total_bytes = 0usize;

    for (index, offset) in starts.iter().copied().enumerate() {
        let before = reader.position() as usize;
        let payload = read_chunk_payload(&mut reader, true).with_context(|| {
            format!(
                "failed to read chunk {} for case {} {}",
                index, bandwidth, label
            )
        })?;
        let after = reader.position() as usize;

        let samples = (metadata.audio_length - offset).min(chunk_samples);
        let duration_ms = samples as f64 * 1000.0 / meta.sample_rate as f64;
        let frame_length = segment_frame_length(samples, meta.segment_samples, meta.frame_length);
        let chunk_bytes = after - before;
        let payload_bytes = payload.len();

        chunk_total_bytes += chunk_bytes;
        payload_total_bytes += payload_bytes;

        let metric = ChunkMetric {
            index,
            offset,
            samples,
            duration_ms,
            frame_length,
            chunk_bytes,
            payload_bytes,
            cumulative_chunk_bytes: chunk_total_bytes,
        };

        println!(
            "{:>6} {:>12} {:>12} {:>12.3} {:>8} {:>14} {:>14} {:>14}",
            metric.index,
            metric.offset,
            metric.samples,
            metric.duration_ms,
            metric.frame_length,
            metric.chunk_bytes,
            metric.payload_bytes,
            metric.cumulative_chunk_bytes
        );

        metrics.push(metric);
    }

    if reader.position() as usize != ecdc_bytes.len() {
        bail!(
            "case {} {} has trailing bytes: position={} total={}",
            bandwidth,
            label,
            reader.position(),
            ecdc_bytes.len()
        );
    }

    Ok(InspectedChunks {
        metrics,
        chunk_total_bytes,
        payload_total_bytes,
        header_bytes,
    })
}

fn execution_target(bundle_dir: &Path) -> Result<ExecutionTarget> {
    if env_enabled("ENCODEC_RS_MATRIX_COREML") {
        return Ok(ExecutionTarget::CoreMl {
            compute_units: CoreMlComputeUnits::CpuAndGpu,
            model_cache_dir: Some(bundle_dir.join(".coreml-cache")),
            low_precision_accumulation_on_gpu: false,
        });
    }

    if env_enabled("ENCODEC_RS_MATRIX_CUDA") {
        return Ok(ExecutionTarget::Cuda { device_id: 0 });
    }

    Ok(ExecutionTarget::Cpu)
}

fn matrix_batch_size() -> usize {
    env::var("ENCODEC_RS_MATRIX_BATCH")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(8)
}

fn env_enabled(name: &str) -> bool {
    env::var(name)
        .ok()
        .map(|value| {
            let value = value.to_ascii_lowercase();
            value == "1" || value == "true" || value == "yes" || value == "on"
        })
        .unwrap_or(false)
}

fn format_chunk_ms(value: Option<f64>) -> String {
    match value {
        Some(value) => format!("{value:.1}"),
        None => "bundle".to_string(),
    }
}

fn read_wav_f32(path: &Path, expected_channels: usize) -> Result<(Array3<f32>, usize, u32)> {
    let mut reader =
        WavReader::open(path).with_context(|| format!("failed to open {}", path.display()))?;
    let spec = reader.spec();

    if spec.channels as usize != expected_channels {
        bail!(
            "expected {} channels in {}, got {}",
            expected_channels,
            path.display(),
            spec.channels
        );
    }

    let interleaved: Vec<f32> = match (spec.sample_format, spec.bits_per_sample) {
        (SampleFormat::Int, 16) => reader
            .samples::<i16>()
            .map(|sample| sample.map(|value| value as f32 / i16::MAX as f32))
            .collect::<std::result::Result<Vec<_>, _>>()?,
        (SampleFormat::Float, 32) => reader
            .samples::<f32>()
            .collect::<std::result::Result<Vec<_>, _>>()?,
        _ => {
            bail!(
                "unsupported WAV format in {}: {:?} {} bits",
                path.display(),
                spec.sample_format,
                spec.bits_per_sample
            );
        }
    };

    let samples_per_channel = interleaved.len() / expected_channels;
    let mut audio = Array3::<f32>::zeros((1, expected_channels, samples_per_channel));

    for (index, sample) in interleaved.into_iter().enumerate() {
        let channel = index % expected_channels;
        let frame = index / expected_channels;
        audio[[0, channel, frame]] = sample;
    }

    Ok((audio, samples_per_channel, spec.sample_rate))
}
