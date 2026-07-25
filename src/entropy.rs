use anyhow::{bail, Result};

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
}

impl ProbabilityScratch {
    fn prepare(&mut self, cardinality: usize, columns: usize) {
        self.pdf.resize(cardinality * columns, 0.0);
        self.quantized.resize(cardinality, 0.0);
        self.probs.resize(cardinality, 0.0);
    }
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
    let uniform = 1.0 / cardinality as f64;
    let near_pdf_threshold = 0.25 / parameters.fp_scale as f64;

    for step in 0..steps {
        for codebook in 0..codebooks {
            let mut max_value = f64::NEG_INFINITY;
            let mut min_value = f64::INFINITY;
            for (bin, quantized_value) in quantized.iter_mut().enumerate().take(cardinality) {
                let input_index = (bin * codebooks + codebook) * steps + step;
                let raw = logits[input_index] as f64 / parameters.tau;
                let value = quantize_logit(raw, parameters.logit_step);
                *quantized_value = value;
                max_value = max_value.max(value);
                min_value = min_value.min(value);
            }

            let mut denominator = 0.0_f64;
            for (probability, quantized_value) in probs
                .iter_mut()
                .zip(quantized.iter().copied())
                .take(cardinality)
            {
                let value = libm::exp(quantized_value - max_value);
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
    let epsilon = 2_f64.powi(-40);
    let scaled = value / step;
    (scaled + 0.5 - epsilon).floor() * step
}

#[cfg(test)]
mod tests {
    use super::*;

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
