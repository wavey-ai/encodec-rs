use anyhow::{bail, Result};

/// Applies triangle-weighted overlap-add to decoded PCM windows.
///
/// `decoded_windows` uses `[window][channel][sample]` storage. The returned
/// buffer uses `[channel][sample]` planar storage. Each window receives a
/// triangle weight. The function accumulates overlapping samples and divides
/// them by their total weight.
pub fn triangle_overlap_add_planar_frames(
    decoded_windows: &[f32],
    window_count: usize,
    channels: usize,
    window_samples: usize,
    stride: usize,
) -> Result<Vec<f32>> {
    if window_count == 0 {
        bail!("overlap-add requires at least one window");
    }
    if channels == 0 {
        bail!("overlap-add requires at least one channel");
    }
    if window_samples == 0 {
        bail!("overlap-add window length must be positive");
    }
    if stride == 0 || stride > window_samples {
        bail!(
            "overlap-add stride {} must be in 1..={}",
            stride,
            window_samples,
        );
    }

    let window_values = channels
        .checked_mul(window_samples)
        .ok_or_else(|| anyhow::anyhow!("overlap-add window size overflows usize"))?;
    let expected_values = window_count
        .checked_mul(window_values)
        .ok_or_else(|| anyhow::anyhow!("overlap-add input size overflows usize"))?;
    if decoded_windows.len() != expected_values {
        bail!(
            "overlap-add input length {} does not match {} windows, {} channels, and {} samples",
            decoded_windows.len(),
            window_count,
            channels,
            window_samples,
        );
    }

    let total_samples = stride
        .checked_mul(window_count - 1)
        .and_then(|value| value.checked_add(window_samples))
        .ok_or_else(|| anyhow::anyhow!("overlap-add output size overflows usize"))?;
    let output_values = channels
        .checked_mul(total_samples)
        .ok_or_else(|| anyhow::anyhow!("overlap-add planar output size overflows usize"))?;

    let denominator = (window_samples + 1) as f32;
    let window_weights: Vec<f32> = (0..window_samples)
        .map(|sample| {
            let position = (sample + 1) as f32 / denominator;
            0.5_f32 - (position - 0.5_f32).abs()
        })
        .collect();

    let mut total_weight = vec![0.0_f32; total_samples];
    let mut output = vec![0.0_f32; output_values];
    for window in 0..window_count {
        let offset = window * stride;
        let input_window_base = window * window_values;
        for sample in 0..window_samples {
            total_weight[offset + sample] += window_weights[sample];
        }
        for channel in 0..channels {
            let input_base = input_window_base + channel * window_samples;
            let output_base = channel * total_samples + offset;
            for sample in 0..window_samples {
                output[output_base + sample] +=
                    window_weights[sample] * decoded_windows[input_base + sample];
            }
        }
    }

    if total_weight.iter().any(|weight| *weight <= 0.0) {
        bail!("overlap-add produced an uncovered output sample");
    }
    for channel in 0..channels {
        let output_base = channel * total_samples;
        for sample in 0..total_samples {
            output[output_base + sample] /= total_weight[sample];
        }
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn triangle_overlap_add_matches_expected_weights() {
        let windows = [1.0_f32, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0];
        let output = triangle_overlap_add_planar_frames(&windows, 2, 1, 4, 2).unwrap();
        let expected = [1.0_f32, 2.0, 16.0 / 3.0, 44.0 / 3.0, 30.0, 40.0];
        assert_eq!(output.len(), expected.len());
        for (actual, expected) in output.iter().zip(expected) {
            assert!((actual - expected).abs() < 1.0e-6, "{actual} != {expected}");
        }
    }

    #[test]
    fn triangle_overlap_add_preserves_planar_channel_layout() {
        let windows = [
            1.0_f32, 2.0, 3.0, 4.0, // window 0, channel 0
            10.0, 20.0, 30.0, 40.0, // window 0, channel 1
            5.0, 6.0, 7.0, 8.0, // window 1, channel 0
            50.0, 60.0, 70.0, 80.0, // window 1, channel 1
        ];
        let output = triangle_overlap_add_planar_frames(&windows, 2, 2, 4, 2).unwrap();
        assert_eq!(output.len(), 12);
        for index in 0..6 {
            assert!((output[6 + index] - output[index] * 10.0).abs() < 1.0e-5);
        }
    }

    #[test]
    fn triangle_overlap_add_rejects_invalid_geometry() {
        assert!(triangle_overlap_add_planar_frames(&[], 0, 1, 4, 2).is_err());
        assert!(triangle_overlap_add_planar_frames(&[0.0; 4], 1, 0, 4, 2).is_err());
        assert!(triangle_overlap_add_planar_frames(&[0.0; 4], 1, 1, 4, 5).is_err());
        assert!(triangle_overlap_add_planar_frames(&[0.0; 3], 1, 1, 4, 2).is_err());
    }
}
