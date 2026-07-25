use anyhow::{bail, Result};

/// Total samples replaced around each fixed-context chunk boundary.
pub const FIXED_CONTEXT_SEAM_REPAIR_SAMPLES: usize = 24;

/// Repairs fixed-context chunk boundaries in planar PCM without changing its length.
pub fn repair_cubic_hermite_seams_planar(
    audio: &mut [f32],
    channels: usize,
    frames: usize,
    seam_stride: usize,
    repair_samples: usize,
) -> Result<()> {
    if channels == 0 {
        bail!("seam repair requires at least one channel");
    }
    if audio.len() != channels.saturating_mul(frames) {
        bail!(
            "planar audio length {} does not match {} channels and {} frames",
            audio.len(),
            channels,
            frames,
        );
    }
    if seam_stride == 0 {
        bail!("seam repair stride must be positive");
    }
    if repair_samples == 0 || !repair_samples.is_multiple_of(2) {
        bail!("seam repair sample count must be a positive even number");
    }
    if frames < 3 {
        return Ok(());
    }

    let each_side = repair_samples / 2;
    let mut seam = seam_stride;
    while seam < frames {
        let start = seam.saturating_sub(each_side).max(1);
        let end = seam.saturating_add(each_side).min(frames - 1);
        let span = end.saturating_sub(start);
        if span > 0 {
            for channel in 0..channels {
                let base = channel * frames;
                let y0 = audio[base + start - 1] as f64;
                let y1 = audio[base + end] as f64;
                let m0 = audio[base + start] as f64 - y0;
                let m1 = y1 - audio[base + end - 1] as f64;
                for index in 0..span {
                    let t = (index + 1) as f64 / (span + 1) as f64;
                    let t2 = t * t;
                    let t3 = t2 * t;
                    let h00 = 2.0 * t3 - 3.0 * t2 + 1.0;
                    let h10 = t3 - 2.0 * t2 + t;
                    let h01 = -2.0 * t3 + 3.0 * t2;
                    let h11 = t3 - t2;
                    audio[base + start + index] =
                        (h00 * y0 + h10 * span as f64 * m0 + h01 * y1 + h11 * span as f64 * m1)
                            as f32;
                }
            }
        }
        seam = seam.saturating_add(seam_stride);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn repair_is_deterministic_and_preserves_length() {
        let mut first = vec![0.0_f32; 80];
        first[40..].fill(1.0);
        let mut second = first.clone();

        repair_cubic_hermite_seams_planar(&mut first, 1, 80, 40, 24).unwrap();
        repair_cubic_hermite_seams_planar(&mut second, 1, 80, 40, 24).unwrap();

        assert_eq!(first, second);
        assert_eq!(first.len(), 80);
        assert_eq!(first[..28], vec![0.0; 28]);
        assert_eq!(first[52..], vec![1.0; 28]);
        assert!(first[39] < first[40]);
    }

    #[test]
    fn repair_uses_the_same_curve_for_each_channel() {
        let frames = 80;
        let mut audio = vec![0.0_f32; frames * 2];
        audio[40..frames].fill(1.0);
        audio[frames + 40..].fill(2.0);

        repair_cubic_hermite_seams_planar(&mut audio, 2, frames, 40, 24).unwrap();

        for frame in 28..52 {
            assert_eq!(audio[frames + frame], audio[frame] * 2.0);
        }
    }

    #[test]
    fn repair_rejects_invalid_geometry() {
        assert!(repair_cubic_hermite_seams_planar(&mut [0.0; 4], 0, 4, 2, 24).is_err());
        assert!(repair_cubic_hermite_seams_planar(&mut [0.0; 4], 1, 5, 2, 24).is_err());
        assert!(repair_cubic_hermite_seams_planar(&mut [0.0; 4], 1, 4, 0, 24).is_err());
        assert!(repair_cubic_hermite_seams_planar(&mut [0.0; 4], 1, 4, 2, 23).is_err());
    }
}
