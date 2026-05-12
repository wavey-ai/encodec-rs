#![cfg(feature = "onnx")]

use std::env;
use std::fs::{self, File};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{bail, Context, Result};
use encodec_rs::ecdc::{decode_ecdc, encode_audio_to_ecdc_stream_with_options};
use encodec_rs::format::segment_starts;
use encodec_rs::onnx::{ExecutionTarget, OnnxFrameCodec, OnnxLmCodec};
use hound::{SampleFormat, WavReader, WavSpec, WavWriter};
use ndarray::Array3;

#[test]
#[ignore = "slow fixture generator; run with BITNEEDLE_FIXTURE_* env vars and --nocapture"]
fn bitneedle_fixture_roundtrips_through_onnx_code() -> Result<()> {
    let bundle_dir = required_path("BITNEEDLE_FIXTURE_BUNDLE")?;
    let input_wav = required_path("BITNEEDLE_FIXTURE_WAV")?;
    let output_ecdc = required_path("BITNEEDLE_FIXTURE_ECDC_OUT")?;
    let output_wav = required_path("BITNEEDLE_FIXTURE_WAV_OUT")?;
    let batch_size = env::var("BITNEEDLE_FIXTURE_BATCH_SIZE")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(8);

    println!("fixture: bundle={}", bundle_dir.display());
    println!("fixture: source_wav={}", input_wav.display());
    println!("fixture: output_ecdc={}", output_ecdc.display());
    println!("fixture: output_wav={}", output_wav.display());
    println!("fixture: batch_size={batch_size}");

    let target = execution_target_from_env(&bundle_dir)?;
    let mut codec = OnnxFrameCodec::from_dir(&bundle_dir, target.clone())?;
    let meta = codec.metadata().clone();
    let (audio, frames, sample_rate) = read_wav_f32(&input_wav, meta.channels)?;
    if sample_rate as usize != meta.sample_rate {
        bail!(
            "input WAV sample rate {sample_rate} does not match bundle sample rate {}",
            meta.sample_rate
        );
    }

    let total_segments = segment_starts(frames, meta.segment_stride.max(1)).len();
    println!(
        "fixture: source frames={} duration={:.6}s segments={}",
        frames,
        frames as f64 / meta.sample_rate as f64,
        total_segments
    );

    if let Some(parent) = output_ecdc.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut out = File::create(&output_ecdc)
        .with_context(|| format!("failed to create {}", output_ecdc.display()))?;
    let mut lm_codec = OnnxLmCodec::from_dir(&bundle_dir, target)?;
    let started = Instant::now();
    let mut emitted_chunks = 0usize;
    let mut emitted_bytes = 0usize;

    encode_audio_to_ecdc_stream_with_options(
        &mut codec,
        Some(&mut lm_codec),
        &audio,
        None,
        batch_size,
        false,
        |bytes| {
            out.write_all(bytes)?;
            emitted_bytes += bytes.len();
            if emitted_chunks == 0 {
                println!("fixture: wrote header bytes={}", bytes.len());
            } else {
                let segment = emitted_chunks;
                if segment == 1 || segment == total_segments || segment % 5 == 0 {
                    println!(
                        "fixture: encoded segment {}/{} ({:.1}%) bytes={} elapsed={:.1}s",
                        segment,
                        total_segments,
                        segment as f64 * 100.0 / total_segments as f64,
                        emitted_bytes,
                        started.elapsed().as_secs_f64()
                    );
                }
            }
            emitted_chunks += 1;
            Ok(())
        },
    )?;
    out.flush()?;

    let payload = fs::read(&output_ecdc)
        .with_context(|| format!("failed to read {}", output_ecdc.display()))?;
    println!(
        "fixture: encode complete ecdc_bytes={} elapsed={:.1}s",
        payload.len(),
        started.elapsed().as_secs_f64()
    );

    let mut decode_codec =
        OnnxFrameCodec::from_dir(&bundle_dir, execution_target_from_env(&bundle_dir)?)?;
    let mut decode_lm =
        OnnxLmCodec::from_dir(&bundle_dir, execution_target_from_env(&bundle_dir)?)?;
    let decode_started = Instant::now();
    let decoded = decode_ecdc(&mut decode_codec, Some(&mut decode_lm), &payload)?;
    write_wav_f32(&output_wav, &decoded.audio, meta.sample_rate)?;
    println!(
        "fixture: decode complete samples={} wav={} elapsed={:.1}s",
        decoded.audio.shape()[2],
        output_wav.display(),
        decode_started.elapsed().as_secs_f64()
    );

    assert_eq!(decoded.audio.shape()[1], meta.channels);
    assert_eq!(decoded.audio.shape()[2], frames);
    Ok(())
}

fn required_path(name: &str) -> Result<PathBuf> {
    env::var_os(name)
        .map(PathBuf::from)
        .filter(|path| !path.as_os_str().is_empty())
        .with_context(|| format!("{name} must be set"))
}

fn execution_target_from_env(bundle_dir: &Path) -> Result<ExecutionTarget> {
    match env::var("BITNEEDLE_FIXTURE_RUNTIME")
        .unwrap_or_else(|_| "cpu".to_string())
        .as_str()
    {
        "cpu" => Ok(ExecutionTarget::Cpu),
        "coreml" => Ok(ExecutionTarget::CoreMl {
            compute_units: encodec_rs::onnx::CoreMlComputeUnits::All,
            model_cache_dir: Some(bundle_dir.join(".coreml-cache")),
            low_precision_accumulation_on_gpu: false,
        }),
        other => bail!("unsupported BITNEEDLE_FIXTURE_RUNTIME={other}; expected cpu or coreml"),
    }
}

fn read_wav_f32(path: &Path, expected_channels: usize) -> Result<(Array3<f32>, usize, u32)> {
    let mut reader = WavReader::open(path)?;
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
            .collect::<Result<Vec<_>, _>>()?,
        (SampleFormat::Float, 32) => reader.samples::<f32>().collect::<Result<Vec<_>, _>>()?,
        _ => bail!(
            "unsupported WAV format in {}: {:?} {} bits",
            path.display(),
            spec.sample_format,
            spec.bits_per_sample
        ),
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

fn write_wav_f32(path: &Path, audio: &Array3<f32>, sample_rate: usize) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let shape = audio.shape();
    let channels = shape[1];
    let samples = shape[2];
    let spec = WavSpec {
        channels: channels as u16,
        sample_rate: sample_rate as u32,
        bits_per_sample: 16,
        sample_format: SampleFormat::Int,
    };
    let mut writer = WavWriter::create(path, spec)?;
    for frame in 0..samples {
        for channel in 0..channels {
            let sample = audio[[0, channel, frame]].clamp(-0.99, 0.99);
            writer.write_sample((sample * i16::MAX as f32) as i16)?;
        }
    }
    writer.finalize()?;
    Ok(())
}
