#[cfg(feature = "onnx")]
use std::fs;
#[cfg(feature = "onnx")]
use std::path::{Path, PathBuf};
#[cfg(feature = "onnx")]
use std::time::Instant;

#[cfg(feature = "onnx")]
use clap::{Args, ValueEnum};
use clap::{Parser, Subcommand};
#[cfg(feature = "onnx")]
use encodec_rs::arithmetic::deterministic_cdf_multi;
#[cfg(feature = "onnx")]
use encodec_rs::ecdc::{
    decode_ecdc, deterministic_pdf_from_logits, encode_audio_frame_evidence,
    encode_audio_to_ecdc_with_options, encode_lm_chunk_evidence, LmCodec,
    ARITHMETIC_TOTAL_RANGE_BITS, DEFAULT_FP_SCALE, DEFAULT_MIN_RANGE,
};
#[cfg(feature = "onnx")]
use encodec_rs::ecdc_presets::fixed_context_samples;
#[cfg(feature = "onnx")]
use encodec_rs::onnx::{CoreMlComputeUnits, ExecutionTarget, OnnxFrameCodec, OnnxLmCodec};
#[cfg(feature = "onnx")]
use encodec_rs::pcm::{f32_to_s16, s16_to_f32, s24_to_f32, s32_to_f32};
#[cfg(feature = "onnx")]
#[cfg(feature = "onnx")]
use encodec_rs::stable_hash::stable_hash_hex;
#[cfg(feature = "onnx")]
use hound::{SampleFormat, WavReader, WavSpec, WavWriter};
#[cfg(feature = "onnx")]
use ndarray::{Array2, Array3};
#[cfg(feature = "onnx")]
use serde_json::json;

#[cfg(feature = "onnx")]
type EncodedSegments = (Vec<Array3<i64>>, Vec<Array2<f32>>);

#[derive(Debug, Parser)]
#[command(name = "encodec-rs")]
#[command(about = "Rust ONNX EnCodec runtime with native ECDC encode/decode")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[cfg(feature = "onnx")]
#[derive(Clone, Debug, Args)]
struct OnnxRuntimeArgs {
    #[arg(long)]
    cuda: bool,
    #[arg(long)]
    tensorrt: bool,
    #[arg(long)]
    coreml: bool,
    #[arg(long)]
    fp16: bool,
    #[arg(long, default_value_t = 0)]
    device_id: i32,
    #[arg(long, value_enum, default_value_t = CoreMlComputeUnitsArg::CpuAndGpu)]
    coreml_compute_units: CoreMlComputeUnitsArg,
    #[arg(long)]
    coreml_low_precision_accumulation_on_gpu: bool,
    #[arg(long)]
    coreml_cache_dir: Option<PathBuf>,
}

#[cfg(feature = "onnx")]
#[derive(Clone, Copy, Debug, ValueEnum)]
enum CoreMlComputeUnitsArg {
    All,
    CpuAndNeuralEngine,
    CpuAndGpu,
    CpuOnly,
}

#[cfg(feature = "onnx")]
impl From<CoreMlComputeUnitsArg> for CoreMlComputeUnits {
    fn from(value: CoreMlComputeUnitsArg) -> Self {
        match value {
            CoreMlComputeUnitsArg::All => Self::All,
            CoreMlComputeUnitsArg::CpuAndNeuralEngine => Self::CpuAndNeuralEngine,
            CoreMlComputeUnitsArg::CpuAndGpu => Self::CpuAndGpu,
            CoreMlComputeUnitsArg::CpuOnly => Self::CpuOnly,
        }
    }
}

#[derive(Debug, Subcommand)]
enum Commands {
    #[cfg(all(not(feature = "onnx"), not(feature = "ecdc")))]
    Unavailable,
    /// Assemble fixed wasm bundles: copy models + write manifests for each
    /// bundle, then write the top-level manifest. Replaces the python/bash
    /// glue in scripts/build_wasm_fixed_bundles.sh.
    #[cfg(feature = "ecdc")]
    FixBundles {
        /// Output root that will contain `bundles/<name>` and `manifest.json`.
        #[arg(long)]
        out_dir: std::path::PathBuf,
        /// Source `onnx-bundles` directory holding each `<name>` bundle.
        #[arg(long)]
        onnx_bundles_dir: std::path::PathBuf,
        /// Bundle names to assemble, in manifest order.
        bundles: Vec<String>,
    },
    #[cfg(feature = "onnx")]
    OnnxInspect {
        bundle_dir: PathBuf,
        #[command(flatten)]
        runtime: OnnxRuntimeArgs,
    },
    #[cfg(feature = "onnx")]
    OnnxSmoke {
        bundle_dir: PathBuf,
        #[command(flatten)]
        runtime: OnnxRuntimeArgs,
    },
    #[cfg(feature = "onnx")]
    OnnxLmProbe {
        bundle_dir: PathBuf,
        #[arg(long, default_value_t = 150)]
        steps: usize,
        #[command(flatten)]
        runtime: OnnxRuntimeArgs,
    },
    #[cfg(feature = "onnx")]
    OnnxLmEvidence {
        bundle_dir: PathBuf,
        output_dir: PathBuf,
        /// Code frames to test. Zero selects the bundle frame length.
        #[arg(long, default_value_t = 0)]
        steps: usize,
        #[command(flatten)]
        runtime: OnnxRuntimeArgs,
    },
    #[cfg(feature = "onnx")]
    OnnxRoundtripWav {
        bundle_dir: PathBuf,
        input_wav: PathBuf,
        output_wav: PathBuf,
        #[arg(long, default_value_t = 16)]
        batch_size: usize,
        #[command(flatten)]
        runtime: OnnxRuntimeArgs,
    },
    #[cfg(feature = "onnx")]
    OnnxEncode {
        bundle_dir: PathBuf,
        input_wav: PathBuf,
        output_ecdc: PathBuf,
        #[arg(long, default_value_t = 8)]
        batch_size: usize,
        #[arg(long)]
        chunk_ms: Option<f64>,
        #[command(flatten)]
        runtime: OnnxRuntimeArgs,
    },
    #[cfg(feature = "onnx")]
    OnnxEncodeEvidence {
        bundle_dir: PathBuf,
        input_wav: PathBuf,
        output_dir: PathBuf,
        #[arg(long, default_value_t = 8)]
        batch_size: usize,
        /// Pass the actual final input length to a dynamic non-fixed model.
        #[arg(long)]
        true_variable_tail: bool,
        #[command(flatten)]
        runtime: OnnxRuntimeArgs,
    },
    #[cfg(feature = "onnx")]
    OnnxDecode {
        bundle_dir: PathBuf,
        input_ecdc: PathBuf,
        output_wav: PathBuf,
        #[command(flatten)]
        runtime: OnnxRuntimeArgs,
    },
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();

    #[cfg(feature = "ecdc")]
    #[allow(irrefutable_let_patterns)]
    if let Commands::FixBundles {
        out_dir,
        onnx_bundles_dir,
        bundles,
    } = &cli.command
    {
        return fix_bundles(out_dir, onnx_bundles_dir, bundles);
    }

    #[cfg(not(feature = "onnx"))]
    {
        let _ = cli;
        return Err("encodec-rs CLI requires the `onnx` or `ecdc` feature".into());
    }

    #[cfg(feature = "onnx")]
    match cli.command {
        Commands::FixBundles { .. } => unreachable!("FixBundles dispatched before onnx match"),
        Commands::OnnxEncode {
            bundle_dir,
            input_wav,
            output_ecdc,
            batch_size,
            chunk_ms,
            runtime,
        } => {
            let target = execution_target(&bundle_dir, &runtime)?;
            let mut codec = OnnxFrameCodec::from_dir(&bundle_dir, target)?;
            let meta = codec.metadata().clone();
            let (audio, _input_frames, input_sample_rate) =
                read_wav_f32(&input_wav, meta.channels)?;
            if input_sample_rate as usize != meta.sample_rate {
                return Err(format!(
                    "input WAV sample rate {} does not match bundle sample rate {}; resampling is not implemented in encodec-rs yet",
                    input_sample_rate,
                    meta.sample_rate
                )
                .into());
            }
            let mut lm_codec = OnnxLmCodec::from_dir(
                bundle_dir.clone(),
                execution_target(&bundle_dir, &runtime)?,
            )?;
            let payload = encode_audio_to_ecdc_with_options(
                &mut codec,
                &mut lm_codec as &mut dyn LmCodec,
                &audio,
                None,
                batch_size.max(1),
                true,
                chunk_ms,
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
                    "batch_size": batch_size.max(1),
                    "chunk_crc": true,
                    "chunk_ms": chunk_ms,
                    "language_model": "q8",
                }))?
            );
        }
        #[cfg(feature = "onnx")]
        Commands::OnnxEncodeEvidence {
            bundle_dir,
            input_wav,
            output_dir,
            batch_size,
            true_variable_tail,
            runtime,
        } => {
            let target = execution_target(&bundle_dir, &runtime)?;
            let mut codec = OnnxFrameCodec::from_dir(&bundle_dir, target)?;
            let mut lm_codec = OnnxLmCodec::from_dir(
                bundle_dir.clone(),
                execution_target(&bundle_dir, &runtime)?,
            )?;
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
            let frame_evidence = encode_audio_frame_evidence(
                &mut codec,
                &audio,
                batch_size.max(1),
                true_variable_tail,
            )?;
            let input_wav_sha256 = stable_hash_hex(&fs::read(&input_wav)?);
            let mut model_input_digest = Vec::new();
            let mut code_digest = Vec::new();
            let mut recovered_code_digest = Vec::new();
            let mut scale_digest = Vec::new();
            let mut entropy_digest = Vec::new();
            let mut segments = Vec::with_capacity(frame_evidence.len());
            fs::create_dir_all(&output_dir)?;
            let codebook_order: Vec<u32> = (0..meta.num_codebooks as u32).collect();
            let codebook_order_path = output_dir.join("codebook-order.u32le");
            let codebook_order_bytes = write_u32le(&codebook_order_path, &codebook_order)?;
            for (index, segment) in frame_evidence.iter().enumerate() {
                let model_input_path =
                    output_dir.join(format!("segment-{index:06}.model-input.f32le"));
                let codes_path = output_dir.join(format!("segment-{index:06}.codes.i64le"));
                let recovered_codes_path =
                    output_dir.join(format!("segment-{index:06}.recovered-codes.i64le"));
                let scale_path = output_dir.join(format!("segment-{index:06}.scale.f32le"));
                let entropy_path = output_dir.join(format!("segment-{index:06}.entropy.bin"));
                let model_input_bytes =
                    write_array3_f32le(&model_input_path, &segment.model_input)?;
                let codes_bytes = write_i64le(&codes_path, &segment.codes)?;
                let scale_bytes = write_f32le(&scale_path, &segment.scale)?;
                let lm_evidence =
                    encode_lm_chunk_evidence(&mut lm_codec, &segment.codes, &segment.scale)?;
                let recovered_codes_bytes =
                    write_i64le(&recovered_codes_path, &lm_evidence.recovered_codes)?;
                fs::write(&entropy_path, &lm_evidence.entropy)?;
                model_input_digest.extend_from_slice(&model_input_bytes);
                code_digest.extend_from_slice(&codes_bytes);
                recovered_code_digest.extend_from_slice(&recovered_codes_bytes);
                scale_digest.extend_from_slice(&scale_bytes);
                entropy_digest.extend_from_slice(&lm_evidence.entropy);
                let model_input_file = evidence_file_name(&model_input_path)?;
                let codes_file = codes_path
                    .file_name()
                    .and_then(|name| name.to_str())
                    .ok_or("codes evidence filename is not valid UTF-8")?;
                let recovered_codes_file = evidence_file_name(&recovered_codes_path)?;
                let scale_file = evidence_file_name(&scale_path)?;
                let entropy_file = evidence_file_name(&entropy_path)?;
                segments.push(json!({
                    "index": index,
                    "offset_samples": segment.offset_samples,
                    "owned_samples": segment.owned_samples,
                    "model_samples": segment.model_input.shape()[2],
                    "frame_length": segment.codes.shape()[2],
                    "model_input_file": model_input_file,
                    "codes_file": codes_file,
                    "recovered_codes_file": recovered_codes_file,
                    "scale_file": scale_file,
                    "entropy_file": entropy_file,
                    "model_input_shape": segment.model_input.shape(),
                    "codes_shape": segment.codes.shape(),
                    "recovered_codes_shape": lm_evidence.recovered_codes.shape(),
                    "scale_shape": segment.scale.shape(),
                    "model_input_sha256": stable_hash_hex(&model_input_bytes),
                    "codes_sha256": stable_hash_hex(&codes_bytes),
                    "recovered_codes_sha256": stable_hash_hex(&recovered_codes_bytes),
                    "scale_sha256": stable_hash_hex(&scale_bytes),
                    "scale_bits_hex": format!("{:08x}", segment.scale[[0, 0]].to_bits()),
                    "entropy_sha256": stable_hash_hex(&lm_evidence.entropy),
                    "entropy_bytes": lm_evidence.entropy.len(),
                    "codes_exactly_recovered": codes_bytes == recovered_codes_bytes,
                }));
            }
            let manifest = json!({
                "schema": "wavey.encodec.frame-evidence",
                "schema_version": 2,
                "evidence_type": "model-input-codes-scales-entropy-recovered-codes",
                "qualification_only": true,
                "profile_container": false,
                "wire_format": "none",
                "provider": runtime_label(&runtime),
                "bundle_dir": codec.bundle_dir(),
                "input_wav": input_wav,
                "input_wav_sha256": input_wav_sha256,
                "input_samples": input_frames,
                "sample_rate": meta.sample_rate,
                "channels": meta.channels,
                "model_name": meta.model_name,
                "bandwidth_kbps": meta.bandwidth_kbps,
                "num_codebooks": meta.num_codebooks,
                "segment_samples": meta.segment_samples,
                "segment_stride": meta.segment_stride,
                "batch_size": batch_size.max(1),
                "true_variable_tail": true_variable_tail,
                "model_input_encoding": "planar little-endian IEEE-754 binary32",
                "codes_encoding": "little-endian signed 64-bit",
                "scale_encoding": "little-endian IEEE-754 binary32",
                "entropy_encoding": "raw arithmetic payload without scale or container framing",
                "recovered_codes_encoding": "little-endian signed 64-bit",
                "codebook_order_file": evidence_file_name(&codebook_order_path)?,
                "codebook_order": codebook_order,
                "codebook_order_sha256": stable_hash_hex(&codebook_order_bytes),
                "model_inputs_sha256": stable_hash_hex(&model_input_digest),
                "codes_sha256": stable_hash_hex(&code_digest),
                "recovered_codes_sha256": stable_hash_hex(&recovered_code_digest),
                "scales_sha256": stable_hash_hex(&scale_digest),
                "entropy_sha256": stable_hash_hex(&entropy_digest),
                "codes_exactly_recovered": code_digest == recovered_code_digest,
                "lm_hash": lm_codec.bitstream_lm_hash(),
                "lm_bitstream_version": lm_codec.bitstream_version(),
                "lm_temperature_bits_hex": format!("{:08x}", 1.0_f32.to_bits()),
                "probability_scale": DEFAULT_FP_SCALE,
                "arithmetic_min_range": DEFAULT_MIN_RANGE,
                "segments": segments,
            });
            let manifest_path = output_dir.join("manifest.json");
            fs::write(&manifest_path, serde_json::to_vec_pretty(&manifest)?)?;
            println!("{}", serde_json::to_string_pretty(&manifest)?);
        }
        #[cfg(feature = "onnx")]
        Commands::OnnxDecode {
            bundle_dir,
            input_ecdc,
            output_wav,
            runtime,
        } => {
            let target = execution_target(&bundle_dir, &runtime)?;
            let mut codec = OnnxFrameCodec::from_dir(&bundle_dir, target)?;
            let mut lm_codec = OnnxLmCodec::from_dir(
                bundle_dir.clone(),
                execution_target(&bundle_dir, &runtime)?,
            )?;
            let payload = fs::read(&input_ecdc)?;
            let decoded = decode_ecdc(&mut codec, &mut lm_codec as &mut dyn LmCodec, &payload)?;
            write_wav_f32(&output_wav, &decoded.audio, codec.metadata().sample_rate)?;
            println!(
                "{}",
                serde_json::to_string_pretty(&json!({
                    "bundle_dir": codec.bundle_dir(),
                    "input_ecdc": input_ecdc,
                    "output_wav": output_wav,
                    "decoded_samples": decoded.audio.shape()[2],
                    "sample_rate": codec.metadata().sample_rate,
                }))?
            );
        }
        #[cfg(feature = "onnx")]
        Commands::OnnxLmProbe {
            bundle_dir,
            steps,
            runtime,
        } => {
            let target = execution_target(&bundle_dir, &runtime)?;
            let mut lm_codec = OnnxLmCodec::from_dir(bundle_dir.clone(), target)?;
            let meta = lm_codec.metadata().clone();
            let steps = steps.min(meta.frame_length);
            let mut states = lm_codec.initial_states(1)?;
            let mut offset = 0_i64;
            let mut input = Array3::<i64>::zeros((1, meta.num_codebooks, 1));
            let mut digest_bytes = Vec::new();
            let card = meta.lm_cardinality();

            for step in 0..steps {
                let (logits, next_offset, next_states) =
                    lm_codec.forward_logits(&input, offset, &states)?;
                let pdf = deterministic_pdf_from_logits(
                    &logits,
                    1.0,
                    meta.lm_entropy_logit_step(),
                    DEFAULT_FP_SCALE,
                )?;
                let cdf = deterministic_cdf_multi(
                    &pdf,
                    card,
                    meta.num_codebooks,
                    ARITHMETIC_TOTAL_RANGE_BITS,
                    DEFAULT_FP_SCALE,
                    DEFAULT_MIN_RANGE,
                )?;
                for value in cdf {
                    digest_bytes.extend_from_slice(&value.to_be_bytes());
                }

                for codebook in 0..meta.num_codebooks {
                    let symbol = ((step * 17) + (codebook * 31)) % card;
                    input[[0, codebook, 0]] = symbol as i64 + 1;
                    digest_bytes.extend_from_slice(&(symbol as u32).to_be_bytes());
                }
                states = next_states;
                offset = next_offset;
            }

            println!(
                "{}",
                serde_json::to_string_pretty(&json!({
                    "bundle_dir": bundle_dir,
                    "arch": std::env::consts::ARCH,
                    "os": std::env::consts::OS,
                    "model_name": meta.model_name,
                    "bandwidth_kbps": meta.bandwidth_kbps,
                    "num_codebooks": meta.num_codebooks,
                    "cardinality": card,
                    "steps": steps,
                    "bitstream_version": lm_codec.bitstream_version(),
                    "lm_hash": lm_codec.bitstream_lm_hash(),
                    "cdf_sequence_hash": stable_hash_hex(&digest_bytes),
                }))?
            );
        }
        #[cfg(feature = "onnx")]
        Commands::OnnxLmEvidence {
            bundle_dir,
            output_dir,
            steps,
            runtime,
        } => {
            let target = execution_target(&bundle_dir, &runtime)?;
            let mut lm_codec = OnnxLmCodec::from_dir(bundle_dir.clone(), target)?;
            let meta = lm_codec.metadata().clone();
            let steps = if steps == 0 { meta.frame_length } else { steps };
            if steps > lm_codec.lm_window_frame_length() {
                return Err(format!(
                    "requested {} LM steps exceed positional capacity {}",
                    steps,
                    lm_codec.lm_window_frame_length(),
                )
                .into());
            }
            let cardinality = meta.lm_cardinality();
            let mut codes = Array3::<i64>::zeros((1, meta.num_codebooks, steps));
            for step in 0..steps {
                for codebook in 0..meta.num_codebooks {
                    codes[[0, codebook, step]] =
                        (((step * 17) + (codebook * 31)) % cardinality) as i64;
                }
            }
            let scale = Array2::from_elem((1, 1), f32::from_bits(0x3f12_3456));
            let evidence = encode_lm_chunk_evidence(&mut lm_codec, &codes, &scale)?;
            fs::create_dir_all(&output_dir)?;
            let codes_path = output_dir.join("codes.i64le");
            let recovered_codes_path = output_dir.join("recovered-codes.i64le");
            let scale_path = output_dir.join("scale.f32le");
            let payload_path = output_dir.join("payload.bin");
            let entropy_path = output_dir.join("entropy.bin");
            let codebook_order_path = output_dir.join("codebook-order.u32le");
            let codes_bytes = write_i64le(&codes_path, &codes)?;
            let recovered_codes_bytes =
                write_i64le(&recovered_codes_path, &evidence.recovered_codes)?;
            let scale_bytes = write_f32le(&scale_path, &scale)?;
            fs::write(&payload_path, &evidence.payload)?;
            fs::write(&entropy_path, &evidence.entropy)?;
            let codebook_order: Vec<u32> = (0..meta.num_codebooks as u32).collect();
            let codebook_order_bytes = write_u32le(&codebook_order_path, &codebook_order)?;
            let manifest = json!({
                "schema": "wavey.encodec.lm-evidence",
                "schema_version": 1,
                "qualification_only": true,
                "provider": runtime_label(&runtime),
                "bundle_dir": bundle_dir,
                "model_name": meta.model_name,
                "bandwidth_kbps": meta.bandwidth_kbps,
                "num_codebooks": meta.num_codebooks,
                "cardinality": cardinality,
                "steps": steps,
                "codes_shape": codes.shape(),
                "codes_file": evidence_file_name(&codes_path)?,
                "codes_sha256": stable_hash_hex(&codes_bytes),
                "recovered_codes_file": evidence_file_name(&recovered_codes_path)?,
                "recovered_codes_sha256": stable_hash_hex(&recovered_codes_bytes),
                "codes_exactly_recovered": codes_bytes == recovered_codes_bytes,
                "scale_file": evidence_file_name(&scale_path)?,
                "scale_sha256": stable_hash_hex(&scale_bytes),
                "scale_bits_hex": format!("{:08x}", scale[[0, 0]].to_bits()),
                "payload_file": evidence_file_name(&payload_path)?,
                "payload_sha256": stable_hash_hex(&evidence.payload),
                "entropy_file": evidence_file_name(&entropy_path)?,
                "entropy_sha256": stable_hash_hex(&evidence.entropy),
                "entropy_bytes": evidence.entropy.len(),
                "codebook_order_file": evidence_file_name(&codebook_order_path)?,
                "codebook_order": codebook_order,
                "codebook_order_sha256": stable_hash_hex(&codebook_order_bytes),
                "symbol_rule": "symbol[codebook,time]=(time*17+codebook*31)%cardinality",
                "bos_rule": "zero on every codebook",
                "next_input_rule": "prior code plus one",
                "iteration_order": "time-major, codebook-minor",
                "lm_hash": lm_codec.bitstream_lm_hash(),
                "lm_bitstream_version": lm_codec.bitstream_version(),
                "lm_temperature_bits_hex": format!("{:08x}", 1.0_f32.to_bits()),
                "probability_scale": DEFAULT_FP_SCALE,
                "arithmetic_min_range": DEFAULT_MIN_RANGE,
            });
            fs::write(
                output_dir.join("manifest.json"),
                serde_json::to_vec_pretty(&manifest)?,
            )?;
            println!("{}", serde_json::to_string_pretty(&manifest)?);
        }
        #[cfg(feature = "onnx")]
        Commands::OnnxInspect {
            bundle_dir,
            runtime,
        } => {
            let target = execution_target(&bundle_dir, &runtime)?;
            let codec = OnnxFrameCodec::from_dir(bundle_dir, target)?;
            println!("{:#?}", codec.metadata());
        }
        #[cfg(feature = "onnx")]
        Commands::OnnxSmoke {
            bundle_dir,
            runtime,
        } => {
            let target = execution_target(&bundle_dir, &runtime)?;
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
            runtime,
        } => {
            let target = execution_target(&bundle_dir, &runtime)?;
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

    #[cfg(feature = "onnx")]
    Ok(())
}

#[cfg(feature = "onnx")]
fn execution_target(
    bundle_dir: &Path,
    runtime: &OnnxRuntimeArgs,
) -> Result<ExecutionTarget, Box<dyn std::error::Error>> {
    let selected_targets = runtime.cuda as u8 + runtime.tensorrt as u8 + runtime.coreml as u8;
    if selected_targets > 1 {
        return Err("choose only one of --cuda, --tensorrt, or --coreml".into());
    }
    if runtime.tensorrt {
        let cache_root = bundle_dir.join(".trt-cache");
        return Ok(ExecutionTarget::TensorRt {
            device_id: runtime.device_id,
            fp16: runtime.fp16,
            engine_cache_path: Some(cache_root.join("engines")),
            timing_cache_path: Some(cache_root.join("timing.cache")),
        });
    }
    if runtime.cuda {
        return Ok(ExecutionTarget::Cuda {
            device_id: runtime.device_id,
        });
    }
    if runtime.coreml {
        return Ok(ExecutionTarget::CoreMl {
            compute_units: runtime.coreml_compute_units.into(),
            model_cache_dir: Some(
                runtime
                    .coreml_cache_dir
                    .clone()
                    .unwrap_or_else(|| bundle_dir.join(".coreml-cache")),
            ),
            low_precision_accumulation_on_gpu: runtime.coreml_low_precision_accumulation_on_gpu,
        });
    }
    if runtime.fp16 {
        return Err("--fp16 requires --tensorrt".into());
    }
    if runtime.coreml_low_precision_accumulation_on_gpu {
        return Err("--coreml-low-precision-accumulation-on-gpu requires --coreml".into());
    }
    if runtime.coreml_cache_dir.is_some() {
        return Err("--coreml-cache-dir requires --coreml".into());
    }
    Ok(ExecutionTarget::Cpu)
}

#[cfg(feature = "onnx")]
fn runtime_label(runtime: &OnnxRuntimeArgs) -> &'static str {
    if runtime.tensorrt {
        "tensorrt"
    } else if runtime.cuda {
        "cuda"
    } else if runtime.coreml {
        "coreml"
    } else {
        "cpu"
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
            .map(|sample| sample.map(s16_to_f32))
            .collect::<Result<Vec<_>, _>>()?,
        (SampleFormat::Int, 24) => reader
            .samples::<i32>()
            .map(|sample| sample.map(s24_to_f32))
            .collect::<Result<Vec<_>, _>>()?,
        (SampleFormat::Int, 32) => reader
            .samples::<i32>()
            .map(|sample| sample.map(s32_to_f32))
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
    if !interleaved.len().is_multiple_of(expected_channels) {
        return Err(format!(
            "WAV sample count {} is not divisible by channel count {}",
            interleaved.len(),
            expected_channels,
        )
        .into());
    }
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
            writer.write_sample(f32_to_s16(audio[[0, channel, frame]]))?;
        }
    }
    writer.finalize()?;
    Ok(())
}

#[cfg(feature = "onnx")]
fn write_i64le(
    path: &PathBuf,
    values: &ndarray::Array3<i64>,
) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let mut bytes = Vec::with_capacity(values.len() * std::mem::size_of::<i64>());
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    fs::write(path, &bytes)?;
    Ok(bytes)
}

#[cfg(feature = "onnx")]
fn write_array3_f32le(
    path: &PathBuf,
    values: &ndarray::Array3<f32>,
) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let mut bytes = Vec::with_capacity(values.len() * std::mem::size_of::<f32>());
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    fs::write(path, &bytes)?;
    Ok(bytes)
}

#[cfg(feature = "onnx")]
fn write_f32le(
    path: &PathBuf,
    values: &ndarray::Array2<f32>,
) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let mut bytes = Vec::with_capacity(values.len() * std::mem::size_of::<f32>());
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    fs::write(path, &bytes)?;
    Ok(bytes)
}

#[cfg(feature = "onnx")]
fn write_u32le(path: &PathBuf, values: &[u32]) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let mut bytes = Vec::with_capacity(std::mem::size_of_val(values));
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    fs::write(path, &bytes)?;
    Ok(bytes)
}

#[cfg(feature = "onnx")]
fn evidence_file_name(path: &std::path::Path) -> Result<&str, Box<dyn std::error::Error>> {
    path.file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| "evidence filename is not valid UTF-8".into())
}

#[cfg(feature = "onnx")]
fn encode_audio_segments(
    codec: &mut OnnxFrameCodec,
    audio: &Array3<f32>,
    batch_size: usize,
) -> Result<EncodedSegments, Box<dyn std::error::Error>> {
    let meta = codec.metadata().clone();
    let total_samples = audio.shape()[2];
    let mut codes = Vec::new();
    let mut scales = Vec::new();
    let segment_starts = segment_starts(total_samples, meta.segment_stride);
    let context = fixed_context_samples(meta.segment_samples, meta.segment_stride)?.unwrap_or(0);
    for chunk in segment_starts.chunks(batch_size) {
        let mut batch = Array3::<f32>::zeros((chunk.len(), meta.channels, meta.segment_samples));
        for (batch_index, offset) in chunk.iter().copied().enumerate() {
            for channel in 0..meta.channels {
                for model_index in 0..meta.segment_samples {
                    let source_index = offset as isize - context as isize + model_index as isize;
                    if source_index >= 0 && (source_index as usize) < total_samples {
                        batch[[batch_index, channel, model_index]] =
                            audio[[0, channel, source_index as usize]];
                    }
                }
            }
        }
        let (batch_codes, batch_scales) = codec.encode_frame(&batch)?;
        for batch_index in 0..chunk.len() {
            let mut segment_codes =
                Array3::<i64>::zeros((1, meta.num_codebooks, batch_codes.shape()[2]));
            let mut segment_scale = Array2::<f32>::zeros((1, 1));
            for codebook in 0..meta.num_codebooks {
                for t in 0..batch_codes.shape()[2] {
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
    let context = fixed_context_samples(meta.segment_samples, meta.segment_stride)?;
    let mut frames = Vec::with_capacity(codes.len());
    for (code_chunk, scale_chunk) in codes.chunks(batch_size).zip(scales.chunks(batch_size)) {
        let frame_length = code_chunk
            .first()
            .map(|codes| codes.shape()[2])
            .unwrap_or(0);
        let mut batch_codes =
            Array3::<i64>::zeros((code_chunk.len(), meta.num_codebooks, frame_length));
        let mut batch_scales = Array2::<f32>::zeros((code_chunk.len(), 1));
        for (batch_index, (segment_codes, segment_scale)) in
            code_chunk.iter().zip(scale_chunk.iter()).enumerate()
        {
            if segment_codes.shape()[2] != frame_length {
                return Err(
                    "all code frames in a decode batch must have the same frame length".into(),
                );
            }
            for codebook in 0..meta.num_codebooks {
                for t in 0..frame_length {
                    batch_codes[[batch_index, codebook, t]] = segment_codes[[0, codebook, t]];
                }
            }
            batch_scales[[batch_index, 0]] = segment_scale[[0, 0]];
        }
        let batch_frames = codec.decode_frame(&batch_codes, &batch_scales)?;
        for batch_index in 0..code_chunk.len() {
            let mut frame = Array3::<f32>::zeros((1, meta.channels, batch_frames.shape()[2]));
            for channel in 0..meta.channels {
                for t in 0..batch_frames.shape()[2] {
                    frame[[0, channel, t]] = batch_frames[[batch_index, channel, t]];
                }
            }
            frames.push(frame);
        }
    }

    if let Some(context) = context {
        let starts = segment_starts(output_length, meta.segment_stride);
        if frames.len() != starts.len() {
            return Err(format!(
                "decoded frame count {} does not match fixed segment count {}",
                frames.len(),
                starts.len()
            )
            .into());
        }
        let mut output = Array3::<f32>::zeros((1, meta.channels, output_length));
        for (frame, offset) in frames.iter().zip(starts) {
            let owned_length = (output_length - offset).min(meta.segment_stride);
            let source_end = context
                .checked_add(owned_length)
                .ok_or("fixed-context source range overflow")?;
            if frame.shape()[2] < source_end {
                return Err(format!(
                    "decoded fixed frame has {} samples; {} are required",
                    frame.shape()[2],
                    source_end
                )
                .into());
            }
            for channel in 0..meta.channels {
                for index in 0..owned_length {
                    output[[0, channel, offset + index]] = frame[[0, channel, context + index]];
                }
            }
        }
        return Ok(output);
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

// ---------------------------------------------------------------------------
// `fix-bundles`: assemble the deployable fixed wasm bundles. This is the Rust
// port of the python/bash glue that previously lived in
// scripts/build_wasm_fixed_bundles.sh: copy each bundle's models + weights,
// read the quantized LM header, and emit per-bundle and top-level manifests.
// ---------------------------------------------------------------------------
#[cfg(feature = "ecdc")]
type FixResult<T> = Result<T, Box<dyn std::error::Error>>;

#[cfg(feature = "ecdc")]
#[derive(serde::Serialize)]
struct LmHeader {
    dim: u32,
    layers: u32,
    heads: u32,
    codebooks: u32,
    cardinality: u32,
    frame_length: u32,
    past_context: u32,
}

#[cfg(feature = "ecdc")]
#[derive(serde::Serialize)]
#[serde(rename_all = "camelCase")]
struct BundleManifest {
    name: String,
    bundle_json: String,
    lm_weights: String,
    encode_model: String,
    decode_model: String,
    model_name: serde_json::Value,
    bandwidth_kbps: serde_json::Value,
    sample_rate: serde_json::Value,
    channels: serde_json::Value,
    segment_samples: serde_json::Value,
    segment_stride: serde_json::Value,
    frame_length: serde_json::Value,
    num_codebooks: serde_json::Value,
    lm: LmHeader,
}

#[cfg(feature = "ecdc")]
#[derive(serde::Serialize)]
struct TopManifest<'a> {
    pkg: &'a str,
    bundles: Vec<BundleManifest>,
}

#[cfg(feature = "ecdc")]
fn fix_bundles(
    out_dir: &std::path::Path,
    onnx_bundles_dir: &std::path::Path,
    bundles: &[String],
) -> FixResult<()> {
    use std::fs;

    if bundles.is_empty() {
        return Err("fix-bundles requires at least one bundle name".into());
    }

    let mut manifests = Vec::with_capacity(bundles.len());
    for name in bundles {
        let src = onnx_bundles_dir.join(name);
        let dst = out_dir.join("bundles").join(name);

        let bundle_json_path = src.join("bundle.json");
        let lm_weights_path = src.join("lm_weights_q8.bin");
        if !bundle_json_path.is_file() {
            return Err(format!("missing {}", bundle_json_path.display()).into());
        }
        if !lm_weights_path.is_file() {
            return Err(format!("missing {}", lm_weights_path.display()).into());
        }

        fs::create_dir_all(&dst)?;
        fs::copy(&bundle_json_path, dst.join("bundle.json"))?;
        fs::copy(&lm_weights_path, dst.join("lm_weights_q8.bin"))?;

        let bundle: serde_json::Value = serde_json::from_slice(&fs::read(&bundle_json_path)?)?;
        let encode_model = bundle
            .get("encode_model")
            .and_then(|v| v.as_str())
            .unwrap_or("encode_frame.onnx")
            .to_string();
        let decode_model = bundle
            .get("decode_model")
            .and_then(|v| v.as_str())
            .unwrap_or("decode_frame.onnx")
            .to_string();

        copy_model_asset(&src, &dst, &encode_model)?;
        copy_model_asset(&src, &dst, &decode_model)?;

        let lm = read_lm_header(&lm_weights_path)?;
        let field = |key: &str| bundle.get(key).cloned().unwrap_or(serde_json::Value::Null);
        let manifest = BundleManifest {
            name: name.clone(),
            bundle_json: "bundle.json".to_string(),
            lm_weights: "lm_weights_q8.bin".to_string(),
            encode_model,
            decode_model,
            model_name: field("model_name"),
            bandwidth_kbps: field("bandwidth_kbps"),
            sample_rate: field("sample_rate"),
            channels: field("channels"),
            segment_samples: field("segment_samples"),
            segment_stride: field("segment_stride"),
            frame_length: field("frame_length"),
            num_codebooks: field("num_codebooks"),
            lm,
        };
        write_json(&dst.join("manifest.json"), &manifest)?;
        manifests.push(manifest);
    }

    let top = TopManifest {
        pkg: "pkg",
        bundles: manifests,
    };
    write_json(&out_dir.join("manifest.json"), &top)?;
    println!("{}", serde_json::to_string_pretty(&top)?);
    Ok(())
}

// Copy a model asset that is either a single file `<model>` or a split
// `<model>.parts.json` describing chunk files to copy alongside it.
#[cfg(feature = "ecdc")]
fn copy_model_asset(
    src_dir: &std::path::Path,
    dst_dir: &std::path::Path,
    model_name: &str,
) -> FixResult<()> {
    use std::fs;

    let direct = src_dir.join(model_name);
    if direct.is_file() {
        fs::copy(&direct, dst_dir.join(model_name))?;
        return Ok(());
    }

    let parts_name = format!("{model_name}.parts.json");
    let parts_path = src_dir.join(&parts_name);
    if parts_path.is_file() {
        fs::copy(&parts_path, dst_dir.join(&parts_name))?;
        let parts: serde_json::Value = serde_json::from_slice(&fs::read(&parts_path)?)?;
        if let Some(entries) = parts.get("parts").and_then(|v| v.as_array()) {
            for entry in entries {
                let Some(part) = entry.as_str() else { continue };
                let dst = dst_dir.join(part);
                if let Some(parent) = dst.parent() {
                    fs::create_dir_all(parent)?;
                }
                fs::copy(src_dir.join(part), dst)?;
            }
        }
        return Ok(());
    }

    Err(format!(
        "missing {} or {} in {}",
        model_name,
        parts_name,
        src_dir.display()
    )
    .into())
}

// Read the 7 little-endian u32 header values that follow the 8-byte magic in a
// quantized LM weight file (see quantized_lm.rs).
#[cfg(feature = "ecdc")]
fn read_lm_header(path: &std::path::Path) -> FixResult<LmHeader> {
    let bytes = std::fs::read(path)?;
    if bytes.len() < 36 {
        return Err(format!("{} is too small to hold an LM header", path.display()).into());
    }
    let read = |index: usize| -> u32 {
        let start = 8 + index * 4;
        u32::from_le_bytes([
            bytes[start],
            bytes[start + 1],
            bytes[start + 2],
            bytes[start + 3],
        ])
    };
    Ok(LmHeader {
        dim: read(0),
        layers: read(1),
        heads: read(2),
        codebooks: read(3),
        cardinality: read(4),
        frame_length: read(5),
        past_context: read(6),
    })
}

#[cfg(feature = "ecdc")]
fn write_json<T: serde::Serialize>(path: &std::path::Path, value: &T) -> FixResult<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut text = serde_json::to_string_pretty(value)?;
    text.push('\n');
    std::fs::write(path, text)?;
    Ok(())
}
