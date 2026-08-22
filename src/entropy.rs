use anyhow::{bail, Result};

const EXACT_EXP_CACHE_SLOTS: usize = 4096;
const LOGIT_ROUND_EPSILON: f64 = f64::from_bits(0x3d70_0000_0000_0000);

#[derive(Debug, Clone, Copy)]
pub struct ProbabilityParameters {
    pub tau: f64,
    pub logit_step: f64,
    pub fp_scale: i64,
}

impl ProbabilityParameters {
    pub fn validate(self) -> Result<Self> {
        if !self.tau.is_finite() || self.tau <= 0.0 {
            bail!("LM temperature must be a positive finite value");
        }
        if !self.logit_step.is_finite() || self.logit_step <= 0.0 {
            bail!("LM logit step must be a positive finite value");
        }
        if self.fp_scale <= 0 {
            bail!("LM probability scale must be positive");
        }
        Ok(self)
    }
}

#[derive(Debug, Default)]
pub struct ProbabilityScratch {
    pdf: Vec<f64>,
    quantized: Vec<f64>,
    probs: Vec<f64>,
    exp_cache_keys: Vec<u64>,
    exp_cache_values: Vec<f64>,
    exp_cache_valid: Vec<u8>,
}

impl ProbabilityScratch {
    fn prepare(&mut self, cardinality: usize, columns: usize) {
        self.pdf.resize(cardinality * columns, 0.0);
        self.quantized.resize(cardinality, 0.0);
        self.probs.resize(cardinality, 0.0);
        if self.exp_cache_keys.is_empty() {
            self.exp_cache_keys.resize(EXACT_EXP_CACHE_SLOTS, 0);
            self.exp_cache_values.resize(EXACT_EXP_CACHE_SLOTS, 0.0);
            self.exp_cache_valid.resize(EXACT_EXP_CACHE_SLOTS, 0);
        }
    }
}

fn exact_cached_exp(value: f64, keys: &mut [u64], values: &mut [f64], valid: &mut [u8]) -> f64 {
    let key = value.to_bits();
    let folded = (key as u32) ^ ((key >> 32) as u32);
    let slot = folded.wrapping_mul(0x9e37_79b1) as usize & (EXACT_EXP_CACHE_SLOTS - 1);
    if valid[slot] != 0 && keys[slot] == key {
        return values[slot];
    }

    let result = libm::exp(value);
    keys[slot] = key;
    values[slot] = result;
    valid[slot] = 1;
    result
}

/// Converts `[cardinality, codebooks, steps]` logits into probability columns.
///
/// The result uses `[cardinality, steps * codebooks]` row-major storage.
/// Columns use time-major, codebook-minor order.
pub fn probability_columns_from_flat_logits<'a>(
    logits: &[f32],
    cardinality: usize,
    codebooks: usize,
    steps: usize,
    parameters: ProbabilityParameters,
    scratch: &'a mut ProbabilityScratch,
) -> Result<&'a [f64]> {
    probability_columns_from_flat_logits_impl::<true>(
        logits,
        cardinality,
        codebooks,
        steps,
        parameters,
        scratch,
    )
}

fn probability_columns_from_flat_logits_impl<'a, const OPTIMIZED: bool>(
    logits: &[f32],
    cardinality: usize,
    codebooks: usize,
    steps: usize,
    parameters: ProbabilityParameters,
    scratch: &'a mut ProbabilityScratch,
) -> Result<&'a [f64]> {
    let parameters = parameters.validate()?;
    if cardinality == 0 || codebooks == 0 || steps == 0 {
        bail!("LM logit dimensions must be non-zero");
    }
    let columns = codebooks
        .checked_mul(steps)
        .ok_or_else(|| anyhow::anyhow!("LM probability column count overflow"))?;
    let expected = cardinality
        .checked_mul(columns)
        .ok_or_else(|| anyhow::anyhow!("LM logit element count overflow"))?;
    if logits.len() != expected {
        bail!(
            "LM logits length {} does not match cardinality {} * codebooks {} * steps {}",
            logits.len(),
            cardinality,
            codebooks,
            steps,
        );
    }

    scratch.prepare(cardinality, columns);
    let pdf = &mut scratch.pdf;
    let quantized = &mut scratch.quantized;
    let probs = &mut scratch.probs;
    let exp_cache_keys = &mut scratch.exp_cache_keys;
    let exp_cache_values = &mut scratch.exp_cache_values;
    let exp_cache_valid = &mut scratch.exp_cache_valid;
    let uniform = 1.0 / cardinality as f64;
    let near_pdf_threshold = 0.25 / parameters.fp_scale as f64;
    let unit_tau = parameters.tau == 1.0;

    for step in 0..steps {
        for codebook in 0..codebooks {
            let mut max_value = f64::NEG_INFINITY;
            let mut min_value = f64::INFINITY;
            for (bin, quantized_value) in quantized.iter_mut().enumerate().take(cardinality) {
                let input_index = (bin * codebooks + codebook) * steps + step;
                let raw = logits[input_index] as f64;
                let raw = if unit_tau { raw } else { raw / parameters.tau };
                let value = quantize_logit(raw, parameters.logit_step);
                *quantized_value = value;
                max_value = max_value.max(value);
                min_value = min_value.min(value);
            }

            if OPTIMIZED && (max_value - min_value) <= (2.0 * parameters.logit_step) {
                let column = step * codebooks + codebook;
                for bin in 0..cardinality {
                    pdf[bin * columns + column] = uniform;
                }
                continue;
            }

            let mut denominator = 0.0_f64;
            for (probability, quantized_value) in probs
                .iter_mut()
                .zip(quantized.iter().copied())
                .take(cardinality)
            {
                let delta = quantized_value - max_value;
                let value = if OPTIMIZED {
                    exact_cached_exp(delta, exp_cache_keys, exp_cache_values, exp_cache_valid)
                } else {
                    libm::exp(delta)
                };
                *probability = value;
                denominator += value;
            }

            let column = step * codebooks + codebook;
            if !denominator.is_finite() || denominator <= 0.0 {
                for bin in 0..cardinality {
                    pdf[bin * columns + column] = uniform;
                }
                continue;
            }

            let mut max_pdf = 0.0_f64;
            let mut min_pdf = f64::INFINITY;
            for probability in probs.iter_mut() {
                *probability /= denominator;
                max_pdf = max_pdf.max(*probability);
                min_pdf = min_pdf.min(*probability);
            }
            let near_uniform = (max_value - min_value) <= (2.0 * parameters.logit_step)
                || (max_pdf - min_pdf) <= near_pdf_threshold;
            for bin in 0..cardinality {
                pdf[bin * columns + column] = if near_uniform { uniform } else { probs[bin] };
            }
        }
    }

    Ok(&pdf[..expected])
}

fn quantize_logit(value: f64, step: f64) -> f64 {
    let scaled = value / step;
    (scaled + 0.5 - LOGIT_ROUND_EPSILON).floor() * step
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_exp_cache_matches_uncached_libm_bits() {
        let mut keys = vec![0_u64; EXACT_EXP_CACHE_SLOTS];
        let mut values = vec![0.0_f64; EXACT_EXP_CACHE_SLOTS];
        let mut valid = vec![0_u8; EXACT_EXP_CACHE_SLOTS];
        let inputs = [
            0.0,
            -0.0,
            -2.1,
            -4.2,
            -42.0,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::NAN,
        ];

        for _ in 0..3 {
            for input in inputs {
                assert_eq!(
                    exact_cached_exp(input, &mut keys, &mut values, &mut valid).to_bits(),
                    libm::exp(input).to_bits(),
                );
            }
        }
    }

    #[test]
    fn cached_probability_path_matches_uncached_path_bit_for_bit() {
        let configurations = [(1_usize, 1_usize, 1_usize), (7, 3, 2), (31, 2, 3)];
        let parameters = ProbabilityParameters {
            tau: 1.0,
            logit_step: 2.1,
            fp_scale: 1 << 13,
        };
        let mut seed = 0xd1b5_4a32_d192_ed03_u64;
        let mut optimized_scratch = ProbabilityScratch::default();
        let mut reference_scratch = ProbabilityScratch::default();

        for _ in 0..4 {
            for (cardinality, codebooks, steps) in configurations {
                let len = cardinality * codebooks * steps;
                let mut logits = Vec::with_capacity(len);
                for index in 0..len {
                    seed = seed
                        .wrapping_mul(6_364_136_223_846_793_005)
                        .wrapping_add(1_442_695_040_888_963_407);
                    let value = if index % 17 == 0 {
                        0.0
                    } else {
                        ((seed >> 40) as i32 - 8_388_608) as f32 / 262_144.0
                    };
                    logits.push(value);
                }

                let optimized = probability_columns_from_flat_logits_impl::<true>(
                    &logits,
                    cardinality,
                    codebooks,
                    steps,
                    parameters,
                    &mut optimized_scratch,
                )
                .unwrap()
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>();
                let reference = probability_columns_from_flat_logits_impl::<false>(
                    &logits,
                    cardinality,
                    codebooks,
                    steps,
                    parameters,
                    &mut reference_scratch,
                )
                .unwrap()
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>();
                assert_eq!(optimized, reference);
            }
        }
    }

    #[test]
    fn columns_use_time_major_codebook_minor_order() {
        let cardinality = 2;
        let codebooks = 2;
        let steps = 2;
        // Input layout is [bin, codebook, step].
        let logits = [
            8.0, 0.0, // bin 0, codebook 0
            0.0, 8.0, // bin 0, codebook 1
            0.0, 8.0, // bin 1, codebook 0
            8.0, 0.0, // bin 1, codebook 1
        ];
        let mut scratch = ProbabilityScratch::default();
        let probabilities = probability_columns_from_flat_logits(
            &logits,
            cardinality,
            codebooks,
            steps,
            ProbabilityParameters {
                tau: 1.0,
                logit_step: 1.0,
                fp_scale: 1 << 13,
            },
            &mut scratch,
        )
        .unwrap();

        // Columns are (time 0, codebook 0), (time 0, codebook 1),
        // (time 1, codebook 0), and (time 1, codebook 1).
        assert!(probabilities[0] > probabilities[4]);
        assert!(probabilities[1] < probabilities[5]);
        assert!(probabilities[2] < probabilities[6]);
        assert!(probabilities[3] > probabilities[7]);
    }

    #[test]
    fn rejects_invalid_probability_parameters() {
        let mut scratch = ProbabilityScratch::default();
        let error = probability_columns_from_flat_logits(
            &[0.0, 0.0],
            2,
            1,
            1,
            ProbabilityParameters {
                tau: 0.0,
                logit_step: 1.0,
                fp_scale: 1,
            },
            &mut scratch,
        )
        .unwrap_err();
        assert!(error.to_string().contains("temperature"));
    }
}
