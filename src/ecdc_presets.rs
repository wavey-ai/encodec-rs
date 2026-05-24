use std::path::{Path, PathBuf};

use anyhow::{bail, Result};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EcdcBandwidthPreset {
    Kbps6,
    Kbps12,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EcdcChunkPreset {
    Ms1333,
    Ms1800,
}

impl Default for EcdcBandwidthPreset {
    fn default() -> Self {
        Self::Kbps6
    }
}

impl Default for EcdcChunkPreset {
    fn default() -> Self {
        Self::Ms1333
    }
}

pub fn bandwidth_preset_from_kbps(value: Option<f64>) -> Result<EcdcBandwidthPreset> {
    match value {
        None => Ok(EcdcBandwidthPreset::Kbps6),
        Some(v) if (v - 6.0).abs() <= 0.001 => Ok(EcdcBandwidthPreset::Kbps6),
        Some(v) if (v - 12.0).abs() <= 0.001 => Ok(EcdcBandwidthPreset::Kbps12),
        Some(v) => bail!("unsupported ECDC bandwidth {v}; supported values are 6.0 and 12.0"),
    }
}

pub fn chunk_preset_from_ms(value: Option<f64>) -> Result<EcdcChunkPreset> {
    match value {
        None => Ok(EcdcChunkPreset::Ms1333),
        Some(v) if (v - 1333.0).abs() <= 5.0 || (v - 1333.3).abs() <= 5.0 => {
            Ok(EcdcChunkPreset::Ms1333)
        }
        Some(v) if (v - 1800.0).abs() <= 5.0 => Ok(EcdcChunkPreset::Ms1800),
        Some(v) => bail!(
            "unsupported ECDC chunk duration {v}ms; supported fixed durations are 1333ms and 1800ms"
        ),
    }
}

pub fn fixed_bundle_name(bandwidth: EcdcBandwidthPreset, chunk: EcdcChunkPreset) -> &'static str {
    match (bandwidth, chunk) {
        (EcdcBandwidthPreset::Kbps6, EcdcChunkPreset::Ms1333) => "encodec_48khz_6kbps_1333ms",
        (EcdcBandwidthPreset::Kbps6, EcdcChunkPreset::Ms1800) => "encodec_48khz_6kbps_1800ms",
        (EcdcBandwidthPreset::Kbps12, EcdcChunkPreset::Ms1333) => "encodec_48khz_12kbps_1333ms",
        (EcdcBandwidthPreset::Kbps12, EcdcChunkPreset::Ms1800) => "encodec_48khz_12kbps_1800ms",
    }
}

pub fn fixed_bundle_dir(
    bundle_root: impl AsRef<Path>,
    bandwidth_kbps: Option<f64>,
    chunk_ms: Option<f64>,
) -> Result<PathBuf> {
    let bandwidth = bandwidth_preset_from_kbps(bandwidth_kbps)?;
    let chunk = chunk_preset_from_ms(chunk_ms)?;
    Ok(bundle_root
        .as_ref()
        .join(fixed_bundle_name(bandwidth, chunk)))
}
