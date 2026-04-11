use std::path::PathBuf;

use clap::{Parser, Subcommand};
use encodec_rs::{Encodec, EncodecOptions};

#[derive(Debug, Parser)]
#[command(name = "encodec-rs")]
#[command(about = "Rust CLI wrapper around the wavey-ai EnCodec binary boundary")]
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
    }

    Ok(())
}
