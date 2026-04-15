use std::path::PathBuf;
#[cfg(feature = "onnx")]
use std::time::Instant;
#[cfg(feature = "onnx")]
use std::fs;

use clap::{Parser, Subcommand};
#[cfg(feature = "onnx")]
use encodec_rs::ecdc::{
    decode_ecdc, encode_audio_to_ecdc, DecodedEcdcAudio, SourceAudioMetadata as EcdcSourceAudioMetadata,
};
#[cfg(feature = "onnx")]
use encodec_rs::onnx::{ExecutionTarget, OnnxFrameCodec, OnnxLmCodec};
#[cfg(feature = "onnx")]
use hound::{SampleFormat, WavReader, WavSpec, WavWriter};
#[cfg(feature = "onnx")]
use ndarray::{Array2, Array3};
#[cfg(feature = "onnx")]
use serde_json::json;

#[derive(Debug, Parser)]
#[command(name = "encodec-rs")]
#[command(about = "Rust ONNX EnCodec runtime with native ECDC encode/decode")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Debug, Subcommand)]
enum Commands {
    #[cfg(not(feature = "onnx"))]
    Unavailable,
    #[cfg(feature = "onnx")]
    OnnxInspect {
        bundle_dir: PathBuf,
        #[arg(long)]
        cuda: bool,
        #[arg(long)]
        tensorrt: bool,
        #[arg(long)]
        fp16: bool,
        #[arg(long, default_value_t = 0)]
        device_id: i32,
    },
    #[cfg(feature = "onnx")]
    OnnxSmoke {
        bundle_dir: PathBuf,
        #[arg(long)]
        cuda: bool,
        #[arg(long)]
        tensorrt: bool,
        #[arg(long)]
        fp16: bool,
        #[arg(long, default_value_t = 0)]
        device_id: i32,
    },
    #[cfg(feature = "onnx")]
    OnnxRoundtripWav {
        bundle_dir: PathBuf,
        input_wav: PathBuf,
        output_wav: PathBuf,
        #[arg(long, default_value_t = 16)]
        batch_size: usize,
        #[arg(long)]
        cuda: bool,
        #[arg(long)]
        tensorrt: bool,
        #[arg(long)]
        fp16: bool,
        #[arg(long, default_value_t = 0)]
        device_id: i32,
    },
    #[cfg(feature = "onnx")]
    OnnxEncode {
        bundle_dir: PathBuf,
        input_wav: PathBuf,
        output_ecdc: PathBuf,
        #[arg(long, default_value_t = 8)]
        batch_size: usize,
        #[arg(long)]
        no_lm: bool,
        #[arg(long)]
        cuda: bool,
        #[arg(long)]
        tensorrt: bool,
        #[arg(long)]
        fp16: bool,
        #[arg(long, default_value_t = 0)]
        device_id: i32,
    },
    #[cfg(feature = "onnx")]
    OnnxDecode {
        bundle_dir: PathBuf,
        input_ecdc: PathBuf,
        output_wav: PathBuf,
        #[arg(long)]
        cuda: bool,
        #[arg(long)]
        tensorrt: bool,
        #[arg(long)]
        fp16: bool,
        #[arg(long, default_value_t = 0)]
        device_id: i32,
    },
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();

    match cli.command {
        #[cfg(not(feature = "onnx"))]
        Commands::Unavailable => {
            return Err("encodec-rs CLI requires the `onnx` feature".into());
        }
        #[cfg(feature = "onnx")]
        Commands::OnnxEncode {
            bundle_dir,
            input_wav,
            output_ecdc,
            batch_size,
            no_lm,
            cuda,
            tensorrt,
            fp16,
            device_id,
        } => {
            let target = execution_target(&bundle_dir, cuda, tensorrt, fp16, device_id)?;
            let mut codec = OnnxFrameCodec::from_dir(&bundle_dir, target)?;
            let meta = codec.metadata().clone();
            let (audio, input_frames, input_sample_rate) = read_wav_f32(&input_wav, meta.channels)?;
            if input_sample_rate as usize != meta.sample_rate {
                return Err(format!(
                    "input WAV sample rate {} does not match bundle sample rate {}; resampling is not implemented in encodec-rs yet",
                    input_sample_rate,
                    meta.sample_rate
                )
                .into());
            }
            let mut lm_codec = if no_lm {
                None
            } else {
                Some(OnnxLmCodec::from_dir(
                    bundle_dir.clone(),
                    execution_target(&bundle_dir, cuda, tensorrt, fp16, device_id)?,
                )?)
            };
            let payload = encode_audio_to_ecdc(
                &mut codec,
                lm_codec.as_mut(),
                &audio,
                Some(&EcdcSourceAudioMetadata {
                    sample_rate: Some(input_sample_rate),
                    channels: Some(meta.channels as u16),
                    total_frames: Some(input_frames),
                }),
            )?;
            fs::write(&output_ecdc, &payload)?;
            let payload_bytes = fs::metadata(&output_ecdc)?.len();
            println!(
                "{}",
                serde_json::to_string_pretty(&json!({
                    "bundle_dir": codec.bundle_dir(),
                    "input_wav": input_wav,
                    "output_ecdc": output_ecdc,
                    "payload_bytes": payload_bytes,
                    "model_name": meta.model_name,
                    "bandwidth_kbps": meta.bandwidth_kbps,
                    "sample_rate": meta.sample_rate,
                    "original_sample_rate": input_sample_rate,
                    "original_frames": input_frames,
                    "batch_size": batch_size.max(1),
                    "language_model": !no_lm,
                }))?
            );
        }
        #[cfg(feature = "onnx")]
        Commands::OnnxDecode {
            bundle_dir,
            input_ecdc,
            output_wav,
            cuda,
            tensorrt,
            fp16,
            device_id,
        } => {
            let target = execution_target(&bundle_dir, cuda, tensorrt, fp16, device_id)?;
            let mut codec = OnnxFrameCodec::from_dir(&bundle_dir, target)?;
            let mut lm_codec = OnnxLmCodec::from_dir(
                bundle_dir.clone(),
                execution_target(&bundle_dir, cuda, tensorrt, fp16, device_id)?,
            )
            .ok();
            let payload = fs::read(&input_ecdc)?;
            let decoded = DecodedOnnxAudioCompat::from_ecdc(decode_ecdc(
                &mut codec,
                lm_codec.as_mut(),
                &payload,
            )?);
            write_wav_f32(&output_wav, &decoded.audio, codec.metadata().sample_rate)?;
            println!(
                "{}",
                serde_json::to_string_pretty(&json!({
                    "bundle_dir": codec.bundle_dir(),
                    "input_ecdc": input_ecdc,
                    "output_wav": output_wav,
                    "decoded_samples": decoded.audio.shape()[2],
                    "sample_rate": codec.metadata().sample_rate,
                    "original_sample_rate": decoded.metadata.original_sample_rate,
                    "original_channels": decoded.metadata.original_channels,
                    "original_total_frames": decoded.metadata.original_total_frames,
                }))?
            );
        }
        #[cfg(feature = "onnx")]
        Commands::OnnxInspect {
            bundle_dir,
            cuda,
            tensorrt,
            fp16,
            device_id,
        } => {
            let target = execution_target(&bundle_dir, cuda, tensorrt, fp16, device_id)?;
            let codec = OnnxFrameCodec::from_dir(bundle_dir, target)?;
            println!("{:#?}", codec.metadata());
        }
        #[cfg(feature = "onnx")]
        Commands::OnnxSmoke {
            bundle_dir,
            cuda,
            tensorrt,
            fp16,
            device_id,
        } => {
            let target = execution_target(&bundle_dir, cuda, tensorrt, fp16, device_id)?;
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
            let mean_abs = if count == 0 {
                0.0
            } else {
                mean_abs / count as f64
            };
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
            batch_size,
            cuda,
            tensorrt,
            fp16,
            device_id,
        } => {
            let target = execution_target(&bundle_dir, cuda, tensorrt, fp16, device_id)?;
            let mut codec = OnnxFrameCodec::from_dir(bundle_dir, target)?;
            let meta = codec.metadata().clone();
            let (audio, input_frames, input_sample_rate) = read_wav_f32(&input_wav, meta.channels)?;
            if input_sample_rate as usize != meta.sample_rate {
                return Err(format!(
                    "input WAV sample rate {} does not match bundle sample rate {}; resampling is not implemented in encodec-rs yet",
                    input_sample_rate,
                    meta.sample_rate
                )
                .into());
            }
            let start_encode = Instant::now();
            let (codes, scales) = encode_audio_segments(&mut codec, &audio, batch_size.max(1))?;
            let encode_seconds = start_encode.elapsed().as_secs_f64();
            let start_decode = Instant::now();
            let decoded = decode_audio_segments(
                &mut codec,
                &codes,
                &scales,
                input_frames,
                batch_size.max(1),
            )?;
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
                    "batch_size": batch_size.max(1),
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
fn execution_target(
    bundle_dir: &PathBuf,
    cuda: bool,
    tensorrt: bool,
    fp16: bool,
    device_id: i32,
) -> Result<ExecutionTarget, Box<dyn std::error::Error>> {
    if cuda && tensorrt {
        return Err("choose only one of --cuda or --tensorrt".into());
    }
    if tensorrt {
        let cache_root = bundle_dir.join(".trt-cache");
        return Ok(ExecutionTarget::TensorRt {
            device_id,
            fp16,
            engine_cache_path: Some(cache_root.join("engines")),
            timing_cache_path: Some(cache_root.join("timing.cache")),
        });
    }
    if cuda {
        return Ok(ExecutionTarget::Cuda { device_id });
    }
    if fp16 {
        return Err("--fp16 requires --tensorrt".into());
    }
    Ok(ExecutionTarget::Cpu)
}

#[cfg(feature = "onnx")]
struct DecodedOnnxAudioCompat {
    metadata: CompatMetadata,
    audio: Array3<f32>,
}

#[cfg(feature = "onnx")]
struct CompatMetadata {
    original_sample_rate: Option<u32>,
    original_channels: Option<u16>,
    original_total_frames: Option<usize>,
}

#[cfg(feature = "onnx")]
impl DecodedOnnxAudioCompat {
    fn from_ecdc(value: DecodedEcdcAudio) -> Self {
        Self {
            metadata: CompatMetadata {
                original_sample_rate: value.metadata.original_sample_rate,
                original_channels: value.metadata.original_channels,
                original_total_frames: value.metadata.original_total_frames,
            },
            audio: value.audio,
        }
    }
}

#[cfg(feature = "onnx")]
fn read_wav_f32(
    path: &PathBuf,
    expected_channels: usize,
) -> Result<(Array3<f32>, usize, u32), Box<dyn std::error::Error>> {
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
    Ok((audio, samples_per_channel, spec.sample_rate))
}

#[cfg(feature = "onnx")]
fn write_wav_f32(
    path: &PathBuf,
    audio: &Array3<f32>,
    sample_rate: usize,
) -> Result<(), Box<dyn std::error::Error>> {
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
    batch_size: usize,
) -> Result<(Vec<Array3<i64>>, Vec<Array2<f32>>), Box<dyn std::error::Error>> {
    let meta = codec.metadata().clone();
    let total_samples = audio.shape()[2];
    let mut codes = Vec::new();
    let mut scales = Vec::new();
    let segment_starts = segment_starts(total_samples, meta.segment_stride);
    for chunk in segment_starts.chunks(batch_size) {
        let mut batch = Array3::<f32>::zeros((chunk.len(), meta.channels, meta.segment_samples));
        for (batch_index, offset) in chunk.iter().copied().enumerate() {
            let copy_len = (total_samples - offset).min(meta.segment_samples);
            for channel in 0..meta.channels {
                for t in 0..copy_len {
                    batch[[batch_index, channel, t]] = audio[[0, channel, offset + t]];
                }
            }
        }
        let (batch_codes, batch_scales) = codec.encode_frame(&batch)?;
        for batch_index in 0..chunk.len() {
            let mut segment_codes =
                Array3::<i64>::zeros((1, meta.num_codebooks, meta.frame_length));
            let mut segment_scale = Array2::<f32>::zeros((1, 1));
            for codebook in 0..meta.num_codebooks {
                for t in 0..meta.frame_length {
                    segment_codes[[0, codebook, t]] = batch_codes[[batch_index, codebook, t]];
                }
            }
            segment_scale[[0, 0]] = batch_scales[[batch_index, 0]];
            codes.push(segment_codes);
            scales.push(segment_scale);
        }
    }
    Ok((codes, scales))
}

#[cfg(feature = "onnx")]
fn decode_audio_segments(
    codec: &mut OnnxFrameCodec,
    codes: &[Array3<i64>],
    scales: &[Array2<f32>],
    output_length: usize,
    batch_size: usize,
) -> Result<Array3<f32>, Box<dyn std::error::Error>> {
    let meta = codec.metadata().clone();
    let mut frames = Vec::with_capacity(codes.len());
    for (code_chunk, scale_chunk) in codes.chunks(batch_size).zip(scales.chunks(batch_size)) {
        let mut batch_codes =
            Array3::<i64>::zeros((code_chunk.len(), meta.num_codebooks, meta.frame_length));
        let mut batch_scales = Array2::<f32>::zeros((code_chunk.len(), 1));
        for (batch_index, (segment_codes, segment_scale)) in
            code_chunk.iter().zip(scale_chunk.iter()).enumerate()
        {
            for codebook in 0..meta.num_codebooks {
                for t in 0..meta.frame_length {
                    batch_codes[[batch_index, codebook, t]] = segment_codes[[0, codebook, t]];
                }
            }
            batch_scales[[batch_index, 0]] = segment_scale[[0, 0]];
        }
        let batch_frames = codec.decode_frame(&batch_codes, &batch_scales)?;
        for batch_index in 0..code_chunk.len() {
            let mut frame = Array3::<f32>::zeros((1, meta.channels, meta.segment_samples));
            for channel in 0..meta.channels {
                for t in 0..meta.segment_samples {
                    frame[[0, channel, t]] = batch_frames[[batch_index, channel, t]];
                }
            }
            frames.push(frame);
        }
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
fn segment_starts(total_samples: usize, stride: usize) -> Vec<usize> {
    let mut starts = Vec::new();
    let mut offset = 0usize;
    while offset < total_samples {
        starts.push(offset);
        offset += stride;
    }
    starts
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
