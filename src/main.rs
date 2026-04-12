use std::path::PathBuf;

use clap::{Parser, Subcommand};
use encodec_rs::{Encodec, EncodecOptions};
#[cfg(feature = "onnx")]
use encodec_rs::onnx::{ExecutionTarget, OnnxFrameCodec};
#[cfg(feature = "onnx")]
use ndarray::Array3;
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
    }

    Ok(())
}
