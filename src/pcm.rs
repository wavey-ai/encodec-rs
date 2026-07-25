/// Converts one signed 16-bit PCM sample to normalized floating-point PCM.
pub fn s16_to_f32(sample: i16) -> f32 {
    sample as f32 / 32_768.0
}

/// Converts one signed 24-bit PCM sample in an `i32` to normalized PCM.
pub fn s24_to_f32(sample: i32) -> f32 {
    sample as f32 / 8_388_608.0
}

/// Converts one signed 32-bit PCM sample to normalized floating-point PCM.
pub fn s32_to_f32(sample: i32) -> f32 {
    sample as f32 / 2_147_483_648.0
}

/// Converts normalized floating-point PCM to signed 16-bit PCM.
///
/// The conversion clips finite inputs to `[-1, 1]`. It maps non-finite
/// inputs to zero. Negative and nonnegative values use their full ranges.
pub fn f32_to_s16(sample: f32) -> i16 {
    if !sample.is_finite() {
        return 0;
    }
    let sample = sample.clamp(-1.0, 1.0);
    if sample < 0.0 {
        (sample * 32_768.0).round() as i16
    } else {
        (sample * 32_767.0).round() as i16
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn s16_conversion_uses_the_full_asymmetric_range() {
        assert_eq!(s16_to_f32(i16::MIN), -1.0);
        assert_eq!(f32_to_s16(-1.0), i16::MIN);
        assert_eq!(f32_to_s16(1.0), i16::MAX);
        assert_eq!(f32_to_s16(0.0), 0);
    }

    #[test]
    fn non_finite_output_samples_become_zero() {
        assert_eq!(f32_to_s16(f32::NAN), 0);
        assert_eq!(f32_to_s16(f32::INFINITY), 0);
        assert_eq!(f32_to_s16(f32::NEG_INFINITY), 0);
    }

    #[test]
    fn integer_input_scales_include_negative_full_scale() {
        assert_eq!(s24_to_f32(-8_388_608), -1.0);
        assert_eq!(s32_to_f32(i32::MIN), -1.0);
    }
}
