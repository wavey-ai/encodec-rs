use std::path::{Path, PathBuf};

use anyhow::{anyhow, bail, Result};

/// (segment_samples, segment_stride) pairs for the fixed contextual bundles.
/// A bundle is only treated as a fixed-context profile (full-window decode +
/// sample-domain crop, no overlap-add) when its geometry matches one of
/// these exactly. Legacy EnCodec bundles may also have segment_samples >
/// segment_stride for ordinary overlap-add and must not be misclassified.
const FIXED_CONTEXT_GEOMETRIES: &[(usize, usize)] = &[(64_960, 64_000), (87_360, 86_400)];

/// Returns the symmetric per-side context sample count for a recognized
/// fixed contextual bundle profile, or `None` if the geometry does not
/// match a known fixed-context profile (e.g. a legacy overlap-add bundle).
pub fn fixed_context_samples(segment_samples: usize, segment_stride: usize) -> Result<Option<usize>> {
    if !FIXED_CONTEXT_GEOMETRIES.contains(&(segment_samples, segment_stride)) {
        return Ok(None);
    }
    let excess = segment_samples.checked_sub(segment_stride).ok_or_else(|| {
        anyhow!(
            "segment_samples {} is smaller than segment_stride {}",
            segment_samples,
            segment_stride,
        )
    })?;
    if excess % 2 != 0 {
        bail!(
            "fixed model geometry must be symmetric: samples={} stride={}",
            segment_samples,
            segment_stride,
        );
    }
    Ok(Some(excess / 2))
}

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
