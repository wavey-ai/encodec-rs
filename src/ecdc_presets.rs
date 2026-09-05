use std::path::{Path, PathBuf};

use anyhow::{anyhow, bail, Result};

/// Per-side context sample count required by every recognised fixed
/// contextual bundle profile.
const FIXED_CONTEXT_SAMPLES_PER_SIDE: usize = 480;

/// Owned-audio stride (segment_stride) of the recognised fixed contextual
/// bundle profile: 1.3333s, 64,000 samples — one revolution at 45 rpm, and
/// two chunks to a revolution at 33 1/3.
/// A bundle is only ever treated as a fixed-context profile (full-window
/// decode + sample-domain crop, no overlap-add) when its segment_stride
/// matches this. Legacy EnCodec bundles may also have
/// segment_samples > segment_stride for ordinary overlap-add (e.g. with a
/// different stride) and must not be misclassified.
const FIXED_CONTEXT_STRIDES: &[usize] = &[64_000];

/// Returns the symmetric per-side context sample count for a recognized
/// fixed contextual bundle profile, or `None` if `segment_stride` does not
/// match a known fixed-context owned duration (e.g. a legacy overlap-add
/// bundle). Once a bundle's stride is recognized, the geometry is validated
/// strictly: `segment_samples` must be at least `segment_stride`, the
/// excess must be evenly split, and it must equal exactly
/// [`FIXED_CONTEXT_SAMPLES_PER_SIDE`] per side; any other relationship is a
/// malformed fixed-profile bundle and is rejected rather than silently
/// falling back to non-context behaviour.
pub fn fixed_context_samples(
    segment_samples: usize,
    segment_stride: usize,
) -> Result<Option<usize>> {
    if !FIXED_CONTEXT_STRIDES.contains(&segment_stride) {
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
    let context = excess / 2;
    if context != FIXED_CONTEXT_SAMPLES_PER_SIDE {
        bail!(
            "recognised fixed-context bundles must use {} samples of context per side, \
             got {} for samples={} stride={}",
            FIXED_CONTEXT_SAMPLES_PER_SIDE,
            context,
            segment_samples,
            segment_stride,
        );
    }
    Ok(Some(context))
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum EcdcBandwidthPreset {
    #[default]
    Kbps6,
    Kbps3,
    Kbps12,
    Kbps24,
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum EcdcChunkPreset {
    #[default]
    Ms1333,
}

pub fn bandwidth_preset_from_kbps(value: Option<f64>) -> Result<EcdcBandwidthPreset> {
    match value {
        None => Ok(EcdcBandwidthPreset::Kbps6),
        Some(v) if (v - 3.0).abs() <= 0.001 => Ok(EcdcBandwidthPreset::Kbps3),
        Some(v) if (v - 6.0).abs() <= 0.001 => Ok(EcdcBandwidthPreset::Kbps6),
        Some(v) if (v - 12.0).abs() <= 0.001 => Ok(EcdcBandwidthPreset::Kbps12),
        Some(v) if (v - 24.0).abs() <= 0.001 => Ok(EcdcBandwidthPreset::Kbps24),
        Some(v) => {
            bail!("unsupported ECDC bandwidth {v}; supported values are 3.0, 6.0, 12.0, and 24.0")
        }
    }
}

pub fn chunk_preset_from_ms(value: Option<f64>) -> Result<EcdcChunkPreset> {
    match value {
        None => Ok(EcdcChunkPreset::Ms1333),
        Some(v) if (v - 1333.0).abs() <= 5.0 || (v - 1333.3).abs() <= 5.0 => {
            Ok(EcdcChunkPreset::Ms1333)
        }
        Some(v) => bail!(
            "unsupported ECDC chunk duration {v}ms; the supported fixed duration is 1333ms"
        ),
    }
}

pub fn fixed_bundle_name(bandwidth: EcdcBandwidthPreset, chunk: EcdcChunkPreset) -> &'static str {
    match (bandwidth, chunk) {
        (EcdcBandwidthPreset::Kbps3, EcdcChunkPreset::Ms1333) => "encodec_48khz_3kbps_1333ms",
        (EcdcBandwidthPreset::Kbps6, EcdcChunkPreset::Ms1333) => "encodec_48khz_6kbps_1333ms",
        (EcdcBandwidthPreset::Kbps12, EcdcChunkPreset::Ms1333) => "encodec_48khz_12kbps_1333ms",
        (EcdcBandwidthPreset::Kbps24, EcdcChunkPreset::Ms1333) => "encodec_48khz_24kbps_1333ms",
    }
}

/// Resolves a fixed bundle while allowing an explicitly supported codebook
/// count. The ordinary bandwidth selector continues to choose the upstream
/// profile default; currently the only alternate is the exact seven-codebook
/// prefix of the 12 kbps model (10.5 kbps before entropy coding).
pub fn fixed_bundle_name_with_codebooks(
    bandwidth: EcdcBandwidthPreset,
    num_codebooks: usize,
    chunk: EcdcChunkPreset,
) -> Result<&'static str> {
    let standard_codebooks = match bandwidth {
        EcdcBandwidthPreset::Kbps3 => 2,
        EcdcBandwidthPreset::Kbps6 => 4,
        EcdcBandwidthPreset::Kbps12 => 8,
        EcdcBandwidthPreset::Kbps24 => 16,
    };
    if num_codebooks == standard_codebooks {
        return Ok(fixed_bundle_name(bandwidth, chunk));
    }
    match (bandwidth, num_codebooks, chunk) {
        (EcdcBandwidthPreset::Kbps12, 7, EcdcChunkPreset::Ms1333) => {
            Ok("encodec_48khz_12kbps_7cb_1333ms")
        }
        _ => bail!(
            "unsupported {num_codebooks}-codebook ECDC variant for this bandwidth; \
             supported alternate is the 12 kbps profile with 7 codebooks"
        ),
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recognizes_known_fixed_context_geometries() {
        assert_eq!(fixed_context_samples(64_960, 64_000).unwrap(), Some(480));
    }

    #[test]
    fn does_not_recognize_unrelated_strides() {
        assert_eq!(fixed_context_samples(48_000, 47_520).unwrap(), None);
        // The 1.8s geometry an LP was once cut at: no longer a fixed profile.
        assert_eq!(fixed_context_samples(87_360, 86_400).unwrap(), None);
        assert_eq!(fixed_context_samples(64_960, 64_960).unwrap(), None);
    }

    #[test]
    fn rejects_samples_smaller_than_stride() {
        assert!(fixed_context_samples(63_999, 64_000).is_err());
    }

    #[test]
    fn rejects_odd_context_excess() {
        assert!(fixed_context_samples(64_961, 64_000).is_err());
    }

    #[test]
    fn rejects_context_not_equal_to_480_samples_per_side() {
        assert!(fixed_context_samples(65_000, 64_000).is_err());
    }

    #[test]
    fn maps_all_supported_bandwidths_to_fixed_bundles() {
        let cases = [
            (
                3.0,
                EcdcBandwidthPreset::Kbps3,
                "encodec_48khz_3kbps_1333ms",
            ),
            (
                6.0,
                EcdcBandwidthPreset::Kbps6,
                "encodec_48khz_6kbps_1333ms",
            ),
            (
                12.0,
                EcdcBandwidthPreset::Kbps12,
                "encodec_48khz_12kbps_1333ms",
            ),
            (
                24.0,
                EcdcBandwidthPreset::Kbps24,
                "encodec_48khz_24kbps_1333ms",
            ),
        ];
        for (rate, preset, bundle) in cases {
            assert_eq!(bandwidth_preset_from_kbps(Some(rate)).unwrap(), preset);
            assert_eq!(fixed_bundle_name(preset, EcdcChunkPreset::Ms1333), bundle);
        }
    }

    #[test]
    fn rejects_unsupported_chunk_duration() {
        assert!(chunk_preset_from_ms(Some(1800.0)).is_err());
        assert_eq!(chunk_preset_from_ms(None).unwrap(), EcdcChunkPreset::Ms1333);
    }

    #[test]
    fn maps_seven_codebook_12kbps_prefix() {
        assert_eq!(
            fixed_bundle_name_with_codebooks(
                EcdcBandwidthPreset::Kbps12,
                7,
                EcdcChunkPreset::Ms1333,
            )
            .unwrap(),
            "encodec_48khz_12kbps_7cb_1333ms"
        );
    }

    #[test]
    fn codebook_aware_mapping_preserves_standard_profiles() {
        assert_eq!(
            fixed_bundle_name_with_codebooks(
                EcdcBandwidthPreset::Kbps24,
                16,
                EcdcChunkPreset::Ms1333,
            )
            .unwrap(),
            "encodec_48khz_24kbps_1333ms"
        );
        assert!(fixed_bundle_name_with_codebooks(
            EcdcBandwidthPreset::Kbps6,
            7,
            EcdcChunkPreset::Ms1333,
        )
        .is_err());
    }

    #[test]
    fn rejects_unsupported_bandwidth() {
        assert!(bandwidth_preset_from_kbps(Some(9.0)).is_err());
    }
}
