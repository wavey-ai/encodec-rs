use clap::{Parser, Subcommand};

#[derive(Debug, Parser)]
#[command(name = "encodec-rs")]
#[command(about = "Deterministic EnCodec runtime with custom neural kernels")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Debug, Subcommand)]
enum Commands {
    #[cfg(not(feature = "ecdc"))]
    Unavailable,
    /// Assemble fixed wasm bundles: copy models + write manifests for each
    /// bundle, then write the top-level manifest. Replaces the python/bash
    /// glue in scripts/build_wasm_fixed_bundles.sh.
    #[cfg(feature = "ecdc")]
    FixBundles {
        /// Output root that will contain `bundles/<name>` and `manifest.json`.
        #[arg(long)]
        out_dir: std::path::PathBuf,
        /// Source model directory holding each `<name>` bundle.
        #[arg(long)]
        onnx_bundles_dir: std::path::PathBuf,
        /// Bundle names to assemble, in manifest order.
        bundles: Vec<String>,
    },
    /// Decode an ECDC payload with the native C neural decoder.
    #[cfg(all(feature = "ecdc", feature = "native-kernel"))]
    Decode {
        /// Bundle directory holding bundle.json, the q8 LM weights and decoder/.
        #[arg(long)]
        bundle: std::path::PathBuf,
        /// Input ECDC file.
        #[arg(long)]
        input: std::path::PathBuf,
        /// Output float32 WAV file.
        #[arg(long)]
        output: std::path::PathBuf,
    },
    /// Encode a WAV into ECDC with the native C neural encoder.
    #[cfg(all(feature = "ecdc", feature = "native-kernel"))]
    Encode {
        /// Bundle directory holding bundle.json, the q8 LM weights, encoder/ and decoder/.
        #[arg(long)]
        bundle: std::path::PathBuf,
        /// Input WAV file (48 kHz, 16/24/32-bit PCM or 32-bit float).
        #[arg(long)]
        input: std::path::PathBuf,
        /// Output ECDC file.
        #[arg(long)]
        output: std::path::PathBuf,
    },
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();

    match cli.command {
        #[cfg(not(feature = "ecdc"))]
        Commands::Unavailable => Err("encodec-rs CLI requires the `ecdc` feature".into()),
        #[cfg(feature = "ecdc")]
        Commands::FixBundles {
            out_dir,
            onnx_bundles_dir,
            bundles,
        } => fix_bundles(&out_dir, &onnx_bundles_dir, &bundles),
        #[cfg(all(feature = "ecdc", feature = "native-kernel"))]
        Commands::Decode {
            bundle,
            input,
            output,
        } => decode_native(&bundle, &input, &output).map_err(|error| error.to_string().into()),
        #[cfg(all(feature = "ecdc", feature = "native-kernel"))]
        Commands::Encode {
            bundle,
            input,
            output,
        } => encode_native(&bundle, &input, &output).map_err(|error| error.to_string().into()),
    }
}

// ---------------------------------------------------------------------------
// Decode: native C neural decoder + deterministic q8 LM.
// ---------------------------------------------------------------------------

#[cfg(all(feature = "ecdc", feature = "native-kernel"))]
fn decode_native(
    bundle_dir: &std::path::Path,
    input: &std::path::Path,
    output: &std::path::Path,
) -> anyhow::Result<()> {
    use anyhow::{bail, Context};
    use encodec_rs::ecdc::decode_ecdc_model_windows;
    use encodec_rs::ecdc_presets::fixed_context_samples;
    use encodec_rs::format::segment_starts;
    use encodec_rs::metadata::FrameBundleMetadata;
    use encodec_rs::native_kernel::NativeFrameCodec;
    use encodec_rs::portable_lm::PortableLmCodec;

    let bundle_path = bundle_dir.join("bundle.json");
    let bundle: FrameBundleMetadata =
        serde_json::from_slice(&std::fs::read(&bundle_path).with_context(|| {
            format!("failed to read {}", bundle_path.display())
        })?)
        .with_context(|| format!("failed to parse {}", bundle_path.display()))?;
    bundle.validate_lm()?;

    let mut codec = NativeFrameCodec::from_bundle_dir(bundle_dir)?;
    let mut lm_codec = PortableLmCodec::from_dir(bundle_dir)?;
    let payload = std::fs::read(input)
        .with_context(|| format!("failed to read {}", input.display()))?;

    let mut windows: Vec<f32> = Vec::new();
    let info = decode_ecdc_model_windows(
        &mut codec,
        &mut lm_codec,
        &payload,
        |_info, _window_index, _offset, _owned_samples, window| {
            let values = window
                .as_slice()
                .ok_or_else(|| anyhow::anyhow!("decoded model window is not contiguous"))?;
            windows.extend_from_slice(values);
            Ok(())
        },
    )?;

    let audio_length = info.metadata.audio_length;
    let Some(context) =
        fixed_context_samples(bundle.segment_samples, bundle.segment_stride)?
    else {
        bail!("the native decode CLI currently supports fixed-context bundles only");
    };

    let expected = segment_starts(audio_length, bundle.segment_stride);
    let frame_count = expected.len();
    let expected_values = frame_count
        .checked_mul(bundle.channels)
        .and_then(|value| value.checked_mul(bundle.segment_samples))
        .context("decoded window buffer overflows usize")?;
    if windows.len() != expected_values {
        bail!(
            "decoded window buffer has {} values; expected {expected_values} for {frame_count} frames",
            windows.len(),
        );
    }

    let mut audio = vec![0.0_f32; bundle.channels * audio_length];
    for (frame_index, offset) in expected.into_iter().enumerate() {
        let owned_len = (audio_length - offset).min(bundle.segment_stride);
        if context + owned_len > bundle.segment_samples {
            bail!(
                "decoded model output is too short: context={context} owned={owned_len} window={}",
                bundle.segment_samples,
            );
        }
        let frame_base = frame_index * bundle.channels * bundle.segment_samples;
        for channel in 0..bundle.channels {
            let source = frame_base + channel * bundle.segment_samples + context;
            let destination = channel * audio_length + offset;
            audio[destination..destination + owned_len]
                .copy_from_slice(&windows[source..source + owned_len]);
        }
    }

    write_float_wav(output, &audio, bundle.channels, bundle.sample_rate)?;
    println!(
        "decoded {} frames, {} samples, {} Hz, {} channels -> {}",
        frame_count,
        audio_length,
        bundle.sample_rate,
        bundle.channels,
        output.display(),
    );
    Ok(())
}

#[cfg(all(feature = "ecdc", feature = "native-kernel"))]
fn write_float_wav(
    path: &std::path::Path,
    planar: &[f32],
    channels: usize,
    sample_rate: usize,
) -> anyhow::Result<()> {
    use anyhow::{bail, Context};

    if channels == 0 {
        bail!("WAV output requires at least one channel");
    }
    let frames = planar.len() / channels;
    if frames * channels != planar.len() {
        bail!("planar audio length is not a multiple of the channel count");
    }
    let data_bytes = planar.len() * 4;
    let mut out = Vec::with_capacity(44 + data_bytes);
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&(36 + data_bytes as u32).to_le_bytes());
    out.extend_from_slice(b"WAVE");
    out.extend_from_slice(b"fmt ");
    out.extend_from_slice(&16_u32.to_le_bytes());
    out.extend_from_slice(&3_u16.to_le_bytes());
    out.extend_from_slice(&(channels as u16).to_le_bytes());
    out.extend_from_slice(&(sample_rate as u32).to_le_bytes());
    out.extend_from_slice(&((sample_rate * channels * 4) as u32).to_le_bytes());
    out.extend_from_slice(&((channels * 4) as u16).to_le_bytes());
    out.extend_from_slice(&32_u16.to_le_bytes());
    out.extend_from_slice(b"data");
    out.extend_from_slice(&(data_bytes as u32).to_le_bytes());
    for frame in 0..frames {
        for channel in 0..channels {
            out.extend_from_slice(&planar[channel * frames + frame].to_le_bytes());
        }
    }
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("failed to create {}", parent.display()))?;
    }
    std::fs::write(path, out).with_context(|| format!("failed to write {}", path.display()))?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Encode: deterministic q8 LM + native C neural encoder.
// ---------------------------------------------------------------------------

#[cfg(all(feature = "ecdc", feature = "native-kernel"))]
fn encode_native(
    bundle_dir: &std::path::Path,
    input: &std::path::Path,
    output: &std::path::Path,
) -> anyhow::Result<()> {
    use anyhow::{bail, Context};
    use encodec_rs::ecdc::{
        encode_ecdc_header_with_options, encode_ecdc_segment_batch_with_options, LmCodec,
    };
    use encodec_rs::ecdc_presets::fixed_context_samples;
    use encodec_rs::format::segment_starts;
    use encodec_rs::metadata::FrameBundleMetadata;
    use encodec_rs::native_kernel::NativeFrameCodec;
    use encodec_rs::portable_lm::PortableLmCodec;
    use ndarray::Array3;

    let bundle_path = bundle_dir.join("bundle.json");
    let bundle: FrameBundleMetadata = serde_json::from_slice(
        &std::fs::read(&bundle_path)
            .with_context(|| format!("failed to read {}", bundle_path.display()))?,
    )
    .with_context(|| format!("failed to parse {}", bundle_path.display()))?;
    bundle.validate_lm()?;

    let (planar, channels, frames, sample_rate) = read_wav(input)?;
    if sample_rate != bundle.sample_rate {
        bail!(
            "WAV sample rate {sample_rate} does not match the bundle's {}",
            bundle.sample_rate
        );
    }
    if channels != bundle.channels {
        bail!(
            "WAV channel count {channels} does not match the bundle's {}",
            bundle.channels
        );
    }

    let mut codec = NativeFrameCodec::from_bundle_dir_with_encoder(bundle_dir)?;
    let mut lm_codec = PortableLmCodec::from_dir(bundle_dir)?;
    let lm_hash = lm_codec
        .bitstream_lm_hash()
        .map(str::to_owned)
        .context("bundle LM does not expose its weight hash")?;
    let header = encode_ecdc_header_with_options(&codec, frames, None, Some(lm_hash), None)?;
    let context = fixed_context_samples(bundle.segment_samples, bundle.segment_stride)?
        .context("the native encode CLI supports fixed-context bundles only")?;

    let mut packets: Vec<u8> = Vec::new();
    let mut window = vec![0.0f32; channels * bundle.segment_samples];
    let mut chunk_count = 0usize;
    for start in segment_starts(frames, bundle.segment_stride) {
        for value in window.iter_mut() {
            *value = 0.0;
        }
        for channel in 0..channels {
            for sample in 0..bundle.segment_samples {
                let absolute = start as isize - context as isize + sample as isize;
                if absolute >= 0 && (absolute as usize) < frames {
                    window[channel * bundle.segment_samples + sample] =
                        planar[channel * frames + absolute as usize];
                }
            }
        }
        let batch = Array3::from_shape_vec(
            (1, channels, bundle.segment_samples),
            window.clone(),
        )
        .context("failed to shape the encoder window")?;
        encode_ecdc_segment_batch_with_options(
            &mut codec,
            &mut lm_codec,
            &batch,
            &[bundle.frame_length],
            |bytes| {
                packets.extend_from_slice(bytes);
                Ok(())
            },
        )?;
        chunk_count += 1;
    }

    let mut out = header;
    out.extend_from_slice(&packets);
    if let Some(parent) = output.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("failed to create {}", parent.display()))?;
    }
    std::fs::write(output, &out)
        .with_context(|| format!("failed to write {}", output.display()))?;
    println!(
        "encoded {frames} frames in {chunk_count} chunks -> {} ({} bytes)",
        output.display(),
        out.len(),
    );
    Ok(())
}

#[cfg(all(feature = "ecdc", feature = "native-kernel"))]
fn read_wav(path: &std::path::Path) -> anyhow::Result<(Vec<f32>, usize, usize, usize)> {
    use anyhow::{bail, Context};

    let bytes =
        std::fs::read(path).with_context(|| format!("failed to read {}", path.display()))?;
    if bytes.len() < 12 || &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        bail!("{} is not a RIFF/WAVE file", path.display());
    }
    let mut format: Option<(u32, usize, usize, usize)> = None;
    let mut data: Option<&[u8]> = None;
    let mut offset = 12usize;
    while offset + 8 <= bytes.len() {
        let id = &bytes[offset..offset + 4];
        let size = u32::from_le_bytes([
            bytes[offset + 4],
            bytes[offset + 5],
            bytes[offset + 6],
            bytes[offset + 7],
        ]) as usize;
        let body = offset + 8;
        if body + size > bytes.len() {
            bail!("a WAV chunk runs past the end of the file");
        }
        if id == b"fmt " {
            if size < 16 {
                bail!("the WAV fmt chunk is too small");
            }
            let tag = u16::from_le_bytes([bytes[body], bytes[body + 1]]) as u32;
            let channels = u16::from_le_bytes([bytes[body + 2], bytes[body + 3]]) as usize;
            let rate = u32::from_le_bytes([
                bytes[body + 4],
                bytes[body + 5],
                bytes[body + 6],
                bytes[body + 7],
            ]) as usize;
            let bits = u16::from_le_bytes([bytes[body + 14], bytes[body + 15]]) as usize;
            let sub = if tag == 0xfffe && size >= 40 {
                u32::from_le_bytes([
                    bytes[body + 24],
                    bytes[body + 25],
                    bytes[body + 26],
                    bytes[body + 27],
                ])
            } else {
                tag
            };
            format = Some((sub, channels, rate, bits));
        } else if id == b"data" {
            data = Some(&bytes[body..body + size]);
        }
        offset = body + size + (size & 1);
    }
    let (sub, channels, rate, bits) = format.context("the WAV has no fmt chunk")?;
    let data = data.context("the WAV has no data chunk")?;
    if channels == 0 {
        bail!("the WAV has no channels");
    }
    let bytes_per_sample = bits / 8;
    if bytes_per_sample == 0 {
        bail!("the WAV bit depth is zero");
    }
    let frames = data.len() / (channels * bytes_per_sample);
    let mut planar = vec![0.0f32; channels * frames];
    for frame in 0..frames {
        for channel in 0..channels {
            let at = (frame * channels + channel) * bytes_per_sample;
            planar[channel * frames + frame] = match (sub, bits) {
                (1, 16) => i16::from_le_bytes([data[at], data[at + 1]]) as f32 / 32_768.0,
                (1, 24) => {
                    let packed =
                        data[at] as i32 | (data[at + 1] as i32) << 8 | (data[at + 2] as i32) << 16;
                    let signed = (packed << 8) >> 8;
                    signed as f32 / 8_388_608.0
                }
                (1, 32) => {
                    i32::from_le_bytes([data[at], data[at + 1], data[at + 2], data[at + 3]]) as f32
                        / 2_147_483_648.0
                }
                (3, 32) => {
                    f32::from_le_bytes([data[at], data[at + 1], data[at + 2], data[at + 3]])
                }
                _ => bail!("unsupported WAV format: subformat {sub} at {bits} bits"),
            };
        }
    }
    Ok((planar, channels, frames, rate))
}

// ---------------------------------------------------------------------------
// FixBundles: copy the models and the q8 LM, and write the manifests.
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
