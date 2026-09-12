use clap::{Parser, Subcommand};

#[derive(Debug, Parser)]
#[command(name = "encodec-rs")]
#[command(about = "Deterministic EnCodec runtime with custom WASM neural kernels")]
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

    let _ = cli;
    Err("encodec-rs CLI requires the `ecdc` feature".into())
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
