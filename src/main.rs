use std::path::PathBuf;
#[cfg(feature = "onnx")]
use std::time::Instant;

use clap::{Parser, Subcommand};
use encodec_rs::{Encodec, EncodecOptions};
#[cfg(feature = "onnx")]
use encodec_rs::onnx::{ExecutionTarget, OnnxFrameCodec};
#[cfg(feature = "onnx")]
use hound::{SampleFormat, WavReader, WavSpec, WavWriter};
#[cfg(feature = "onnx")]
use ndarray::{Array2, Array3};
#[cfg(feature = "onnx")]
use serde_json::json;

#[derive(Debug, Parser)]
#[command(name = "encodec-rs")]
#[command(about = "Rust CLI wrapper around the wavey-ai EnCodec binary boundary and ONNX frame bundle")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Debug, Subcommand)]
enum Commands {
    Encode {
        input: PathBuf,
        output: PathBuf,
        #[arg(long)]
        bandwidth: Option<f32>,
        #[arg(long = "hq")]
        high_quality: bool,
        #[arg(long = "lm")]
        language_model: bool,
        #[arg(long)]
        force: bool,
        #[arg(long)]
        rescale: bool,
    },
    Decode {
        input: PathBuf,
        output: PathBuf,
        #[arg(long)]
        force: bool,
        #[arg(long)]
        rescale: bool,
    },
    Roundtrip {
        input: PathBuf,
        output: PathBuf,
        #[arg(long)]
        bandwidth: Option<f32>,
        #[arg(long = "hq")]
        high_quality: bool,
        #[arg(long = "lm")]
        language_model: bool,
        #[arg(long)]
        force: bool,
        #[arg(long)]
        rescale: bool,
    },
    #[cfg(feature = "onnx")]
    OnnxInspect {
        bundle_dir: PathBuf,
        #[arg(long)]
        cuda: bool,
        #[arg(long, default_value_t = 0)]
        device_id: i32,
    },
    #[cfg(feature = "onnx")]
    OnnxSmoke {
        bundle_dir: PathBuf,
        #[arg(long)]
        cuda: bool,
        #[arg(long, default_value_t = 0)]
        device_id: i32,
    },
    #[cfg(feature = "onnx")]
    OnnxRoundtripWav {
        bundle_dir: PathBuf,
        input_wav: PathBuf,
        output_wav: PathBuf,
        #[arg(long)]
        cuda: bool,
        #[arg(long, default_value_t = 0)]
        device_id: i32,
    },
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();
    let encodec = Encodec::from_env();

    match cli.command {
        Commands::Encode {
            input,
            output,
            bandwidth,
            high_quality,
            language_model,
            force,
            rescale,
        } => encodec.encode_file(
            input,
            output,
            &EncodecOptions {
                bandwidth,
                high_quality,
                language_model,
                force,
                rescale,
            },
        )?,
        Commands::Decode {
            input,
            output,
            force,
            rescale,
        } => encodec.decode_file(
            input,
            output,
            &EncodecOptions {
                force,
                rescale,
                ..Default::default()
            },
        )?,
        Commands::Roundtrip {
            input,
            output,
            bandwidth,
            high_quality,
            language_model,
            force,
            rescale,
        } => encodec.roundtrip_to_wav(
            input,
            output,
            &EncodecOptions {
                bandwidth,
                high_quality,
                language_model,
                force,
                rescale,
            },
        )?,
        #[cfg(feature = "onnx")]
        Commands::OnnxInspect {
            bundle_dir,
            cuda,
            device_id,
        } => {
            let target = if cuda {
                ExecutionTarget::Cuda { device_id }
            } else {
                ExecutionTarget::Cpu
            };
            let codec = OnnxFrameCodec::from_dir(bundle_dir, target)?;
            println!("{:#?}", codec.metadata());
        }
        #[cfg(feature = "onnx")]
        Commands::OnnxSmoke {
            bundle_dir,
            cuda,
            device_id,
        } => {
            let target = if cuda {
                ExecutionTarget::Cuda { device_id }
            } else {
                ExecutionTarget::Cpu
            };
            let mut codec = OnnxFrameCodec::from_dir(bundle_dir, target)?;
            let meta = codec.metadata().clone();
            let mut audio = Array3::<f32>::zeros((1, meta.channels, meta.segment_samples));
            for t in 0..meta.segment_samples {
                let phase = (t as f32 / meta.sample_rate as f32) * 440.0 * std::f32::consts::TAU;
                let sample = phase.sin() * 0.05;
                for c in 0..meta.channels {
                    audio[[0, c, t]] = sample;
                }
            }
            let (codes, scale) = codec.encode_frame(&audio)?;
            let decoded = codec.decode_frame(&codes, &scale)?;
            let mut max_abs = 0.0_f32;
            let mut mean_abs = 0.0_f64;
            let mut count = 0_u64;
            for (left, right) in audio.iter().zip(decoded.iter()) {
                let diff = (left - right).abs();
                if diff > max_abs {
                    max_abs = diff;
                }
                mean_abs += diff as f64;
                count += 1;
            }
            let mean_abs = if count == 0 { 0.0 } else { mean_abs / count as f64 };
            println!(
                "{}",
                serde_json::to_string_pretty(&json!({
                    "model_name": meta.model_name,
                    "bandwidth_kbps": meta.bandwidth_kbps,
                    "codes_shape": codes.shape(),
                    "scale_shape": scale.shape(),
                    "decoded_shape": decoded.shape(),
                    "max_abs_diff_vs_input": max_abs,
                    "mean_abs_diff_vs_input": mean_abs,
                }))?
            );
        }
        #[cfg(feature = "onnx")]
        Commands::OnnxRoundtripWav {
            bundle_dir,
            input_wav,
            output_wav,
            cuda,
            device_id,
        } => {
            let target = if cuda {
                ExecutionTarget::Cuda { device_id }
            } else {
                ExecutionTarget::Cpu
            };
            let mut codec = OnnxFrameCodec::from_dir(bundle_dir, target)?;
            let meta = codec.metadata().clone();
            let (audio, input_frames) = read_wav_f32(&input_wav, meta.channels)?;
            let start_encode = Instant::now();
            let (codes, scales) = encode_audio_segments(&mut codec, &audio)?;
            let encode_seconds = start_encode.elapsed().as_secs_f64();
            let start_decode = Instant::now();
            let decoded = decode_audio_segments(&mut codec, &codes, &scales, input_frames)?;
            let decode_seconds = start_decode.elapsed().as_secs_f64();
            write_wav_f32(&output_wav, &decoded, meta.sample_rate)?;
            let audio_seconds = input_frames as f64 / meta.sample_rate as f64;
            println!(
                "{}",
                serde_json::to_string_pretty(&json!({
                    "model_name": meta.model_name,
                    "bandwidth_kbps": meta.bandwidth_kbps,
                    "input_wav": input_wav,
                    "output_wav": output_wav,
                    "audio_seconds": audio_seconds,
                    "segments": codes.len(),
                    "encode_seconds": encode_seconds,
                    "encode_rtf": encode_seconds / audio_seconds,
                    "decode_seconds": decode_seconds,
                    "decode_rtf": decode_seconds / audio_seconds,
                }))?
            );
        }
    }

    Ok(())
}

#[cfg(feature = "onnx")]
fn read_wav_f32(path: &PathBuf, expected_channels: usize) -> Result<(Array3<f32>, usize), Box<dyn std::error::Error>> {
    let mut reader = WavReader::open(path)?;
    let spec = reader.spec();
    if spec.channels as usize != expected_channels {
        return Err(format!(
            "expected {} channels in {}, got {}",
            expected_channels,
            path.display(),
            spec.channels
        )
        .into());
    }
    let interleaved: Vec<f32> = match (spec.sample_format, spec.bits_per_sample) {
        (SampleFormat::Int, 16) => reader
            .samples::<i16>()
            .map(|sample| sample.map(|value| value as f32 / i16::MAX as f32))
            .collect::<Result<Vec<_>, _>>()?,
        (SampleFormat::Float, 32) => reader.samples::<f32>().collect::<Result<Vec<_>, _>>()?,
        _ => {
            return Err(format!(
                "unsupported WAV format in {}: {:?} {} bits",
                path.display(),
                spec.sample_format,
                spec.bits_per_sample
            )
            .into());
        }
    };
    let samples_per_channel = interleaved.len() / expected_channels;
    let mut audio = Array3::<f32>::zeros((1, expected_channels, samples_per_channel));
    for (index, sample) in interleaved.into_iter().enumerate() {
        let channel = index % expected_channels;
        let frame = index / expected_channels;
        audio[[0, channel, frame]] = sample;
    }
    Ok((audio, samples_per_channel))
}

#[cfg(feature = "onnx")]
fn write_wav_f32(path: &PathBuf, audio: &Array3<f32>, sample_rate: usize) -> Result<(), Box<dyn std::error::Error>> {
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

#[cfg(feature = "onnx")]
fn encode_audio_segments(
    codec: &mut OnnxFrameCodec,
    audio: &Array3<f32>,
) -> Result<(Vec<Array3<i64>>, Vec<Array2<f32>>), Box<dyn std::error::Error>> {
    let meta = codec.metadata().clone();
    let total_samples = audio.shape()[2];
    let mut codes = Vec::new();
    let mut scales = Vec::new();
    let mut offset = 0usize;
    while offset < total_samples {
        let mut segment = Array3::<f32>::zeros((1, meta.channels, meta.segment_samples));
        let copy_len = (total_samples - offset).min(meta.segment_samples);
        for channel in 0..meta.channels {
            for t in 0..copy_len {
                segment[[0, channel, t]] = audio[[0, channel, offset + t]];
            }
        }
        let (segment_codes, segment_scale) = codec.encode_frame(&segment)?;
        codes.push(segment_codes);
        scales.push(segment_scale);
        offset += meta.segment_stride;
    }
    Ok((codes, scales))
}

#[cfg(feature = "onnx")]
fn decode_audio_segments(
    codec: &mut OnnxFrameCodec,
    codes: &[Array3<i64>],
    scales: &[Array2<f32>],
    output_length: usize,
) -> Result<Array3<f32>, Box<dyn std::error::Error>> {
    let meta = codec.metadata().clone();
    let mut frames = Vec::with_capacity(codes.len());
    for (segment_codes, segment_scale) in codes.iter().zip(scales.iter()) {
        frames.push(codec.decode_frame(segment_codes, segment_scale)?);
    }
    let reconstructed = linear_overlap_add(&frames, meta.segment_stride);
    let mut trimmed = Array3::<f32>::zeros((1, meta.channels, output_length));
    for channel in 0..meta.channels {
        for t in 0..output_length {
            trimmed[[0, channel, t]] = reconstructed[[0, channel, t]];
        }
    }
    Ok(trimmed)
}

#[cfg(feature = "onnx")]
fn linear_overlap_add(frames: &[Array3<f32>], stride: usize) -> Array3<f32> {
    let channels = frames[0].shape()[1];
    let frame_length = frames[0].shape()[2];
    let total_size = stride * (frames.len() - 1) + frame_length;
    let mut output = Array3::<f32>::zeros((1, channels, total_size));
    let mut sum_weight = vec![0.0_f32; total_size];
    let weight = triangle_weight(frame_length);

    let mut offset = 0usize;
    for frame in frames {
        let frame_len = frame.shape()[2];
        for t in 0..frame_len {
            let w = weight[t];
            sum_weight[offset + t] += w;
            for channel in 0..channels {
                output[[0, channel, offset + t]] += frame[[0, channel, t]] * w;
            }
        }
        offset += stride;
    }
    for t in 0..total_size {
        let denom = sum_weight[t];
        if denom > 0.0 {
            for channel in 0..channels {
                output[[0, channel, t]] /= denom;
            }
        }
    }
    output
}

#[cfg(feature = "onnx")]
fn triangle_weight(frame_length: usize) -> Vec<f32> {
    (0..frame_length)
        .map(|index| {
            let t = (index + 1) as f32 / (frame_length + 1) as f32;
            0.5 - (t - 0.5).abs()
        })
        .collect()
}
