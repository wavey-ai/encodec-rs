use anyhow::{bail, Result};

const EPS_EDGE: f64 = 9.094_947_017_729_282e-13;
const EPS_PERTURB: f64 = 8.673_617_379_884_035e-19;

fn counts_from_pdf_flat(pdf: &[f64], fp_scale: i64, out: &mut Vec<i64>) {
    out.resize(pdf.len(), 0);
    let scale = fp_scale as f64;
    for (idx, (value, count)) in pdf.iter().zip(out.iter_mut()).enumerate() {
        let mut x = value.max(0.0) * scale;
        let frac = x - x.floor();
        if frac <= EPS_EDGE || frac >= 1.0 - EPS_EDGE {
            let sign = if idx % 2 == 0 { -1.0 } else { 1.0 };
            x = (x + sign * EPS_PERTURB).max(0.0);
        }
        *count = x.floor() as i64;
    }
}

#[derive(Debug, Default)]
pub struct CdfScratch {
    normalized: Vec<f64>,
    counts: Vec<i64>,
    cdf: Vec<i64>,
    base: Vec<i64>,
    order: Vec<(i64, usize)>,
}

impl CdfScratch {
    fn prepare(&mut self, pdf_len: usize, n_bins: usize, needs_normalized: bool) {
        if needs_normalized {
            self.normalized.resize(pdf_len, 0.0);
        }
        self.counts.resize(pdf_len, 0);
        self.cdf.resize(pdf_len, 0);
        self.base.resize(n_bins, 0);
        if self.order.capacity() < n_bins {
            self.order.reserve(n_bins - self.order.len());
        }
    }
}

fn increment_largest_remainders(base: &mut [i64], order: &mut [(i64, usize)], remainder: usize) {
    if remainder == 0 {
        return;
    }

    debug_assert!(remainder <= order.len());
    if remainder < order.len() {
        // The old implementation sorted every bin in descending tuple order.
        // Arithmetic output depends only on which rows occupy the first
        // `remainder` positions. Partitioning selects that exact set in linear
        // average time without ordering either partition.
        order.select_nth_unstable_by(remainder, |left, right| right.cmp(left));
    }
    for &(_, row) in &order[..remainder] {
        base[row] += 1;
    }
}

pub fn deterministic_cdf_multi(
    pdf: &[f64],
    n_bins: usize,
    n_cols: usize,
    total_range_bits: u32,
    fp_scale: i64,
    min_range: i64,
) -> Result<Vec<i64>> {
    let mut scratch = CdfScratch::default();
    deterministic_cdf_multi_with_scratch(
        pdf,
        n_bins,
        n_cols,
        total_range_bits,
        fp_scale,
        min_range,
        &mut scratch,
    )?;
    Ok(scratch.cdf)
}

pub fn deterministic_cdf_multi_with_scratch<'a>(
    pdf: &[f64],
    n_bins: usize,
    n_cols: usize,
    total_range_bits: u32,
    fp_scale: i64,
    min_range: i64,
    scratch: &'a mut CdfScratch,
) -> Result<&'a [i64]> {
    deterministic_cdf_multi_impl::<false>(
        pdf,
        n_bins,
        n_cols,
        total_range_bits,
        fp_scale,
        min_range,
        scratch,
    )
}

pub(crate) fn deterministic_cdf_multi_from_valid_pdf_with_scratch<'a>(
    pdf: &[f64],
    n_bins: usize,
    n_cols: usize,
    total_range_bits: u32,
    fp_scale: i64,
    min_range: i64,
    scratch: &'a mut CdfScratch,
) -> Result<&'a [i64]> {
    deterministic_cdf_multi_impl::<true>(
        pdf,
        n_bins,
        n_cols,
        total_range_bits,
        fp_scale,
        min_range,
        scratch,
    )
}

fn deterministic_cdf_multi_impl<'a, const VALID_PDF: bool>(
    pdf: &[f64],
    n_bins: usize,
    n_cols: usize,
    total_range_bits: u32,
    fp_scale: i64,
    min_range: i64,
    scratch: &'a mut CdfScratch,
) -> Result<&'a [i64]> {
    if n_bins == 0 || n_cols == 0 {
        bail!("pdf matrix must be non-empty");
    }
    if pdf.len() != n_bins * n_cols {
        bail!(
            "pdf matrix buffer length {} does not match shape {}x{}",
            pdf.len(),
            n_bins,
            n_cols
        );
    }

    let total = 1_i64
        .checked_shl(total_range_bits)
        .ok_or_else(|| anyhow::anyhow!("total_range_bits {total_range_bits} is too large"))?;
    let alloc = total - min_range * n_bins as i64;
    if alloc <= 0 {
        bail!("invalid total_range_bits/min_range combination");
    }

    scratch.prepare(pdf.len(), n_bins, !VALID_PDF);
    let CdfScratch {
        normalized,
        counts,
        cdf,
        base,
        order,
    } = scratch;
    if VALID_PDF {
        counts_from_pdf_flat(pdf, fp_scale, counts);
    } else {
        for col in 0..n_cols {
            let mut sum = 0.0_f64;
            for row in 0..n_bins {
                let value = pdf[row * n_cols + col].max(0.0);
                normalized[row * n_cols + col] = value;
                sum += value;
            }
            if !sum.is_finite() || sum <= 0.0 {
                for row in 0..n_bins {
                    normalized[row * n_cols + col] = 1.0;
                }
            }
        }

        counts_from_pdf_flat(normalized, fp_scale, counts);
        for col in 0..n_cols {
            let mut sum = 0_i64;
            for row in 0..n_bins {
                sum += counts[row * n_cols + col];
            }
            if sum <= 0 {
                for row in 0..n_bins {
                    counts[row * n_cols + col] = 1;
                }
            }
        }
    }

    for col in 0..n_cols {
        let mut num_sum = 0_i64;
        for row in 0..n_bins {
            num_sum += counts[row * n_cols + col];
        }
        if num_sum <= 0 {
            bail!("invalid zero-count column in pdf matrix");
        }

        let mut base_sum = 0_i64;
        for row in 0..n_bins {
            let num = counts[row * n_cols + col];
            let value = (alloc * num) / num_sum;
            base[row] = value;
            base_sum += value;
        }

        let remainder = alloc - base_sum;
        if remainder > 0 {
            order.clear();
            for row in 0..n_bins {
                let num = counts[row * n_cols + col];
                let prio = (alloc * num) - (num_sum * base[row]);
                let key = prio * (n_bins as i64 + 1) - row as i64;
                order.push((key, row));
            }
            increment_largest_remainders(base, order, remainder as usize);
        }

        let mut running = 0_i64;
        for row in 0..n_bins {
            running += base[row] + min_range;
            cdf[row * n_cols + col] = running;
        }
        if running != total {
            bail!("cdf sum mismatch: expected {total}, got {running}");
        }
    }

    Ok(cdf)
}

struct BitWriter {
    current_value: u64,
    current_bits: u8,
    bytes: Vec<u8>,
}

impl BitWriter {
    fn new() -> Self {
        Self {
            current_value: 0,
            current_bits: 0,
            bytes: Vec::new(),
        }
    }

    fn push_bit(&mut self, bit: u8) {
        self.current_value += (bit as u64) << self.current_bits;
        self.current_bits += 1;
        while self.current_bits >= 8 {
            let lower = (self.current_value & 0xff) as u8;
            self.current_bits -= 8;
            self.current_value >>= 8;
            self.bytes.push(lower);
        }
    }

    fn finish(mut self) -> Vec<u8> {
        if self.current_bits > 0 {
            self.bytes.push(self.current_value as u8);
            self.current_value = 0;
            self.current_bits = 0;
        }
        self.bytes
    }
}

struct BitReader {
    data: Vec<u8>,
    offset: usize,
    current_value: u64,
    current_bits: u8,
}

impl BitReader {
    fn new(data: Vec<u8>) -> Self {
        Self {
            data,
            offset: 0,
            current_value: 0,
            current_bits: 0,
        }
    }

    fn pull_bit(&mut self) -> Option<u8> {
        while self.current_bits < 1 {
            let byte = *self.data.get(self.offset)?;
            self.offset += 1;
            self.current_value += (byte as u64) << self.current_bits;
            self.current_bits += 8;
        }
        let out = (self.current_value & 1) as u8;
        self.current_value >>= 1;
        self.current_bits -= 1;
        Some(out)
    }
}

pub struct ArithmeticEncoder {
    total_range_bits: u32,
    low: u64,
    high: u64,
    max_bit: i32,
    writer: BitWriter,
}

impl ArithmeticEncoder {
    pub fn new(total_range_bits: u32) -> Result<Self> {
        if total_range_bits > 30 {
            bail!("total_range_bits must be <= 30");
        }
        Ok(Self {
            total_range_bits,
            low: 0,
            high: 0,
            max_bit: -1,
            writer: BitWriter::new(),
        })
    }

    pub fn push_pdf_symbols(
        &mut self,
        pdf: &[f64],
        n_bins: usize,
        n_cols: usize,
        symbols: &[usize],
        fp_scale: i64,
        min_range: i64,
    ) -> Result<()> {
        let mut scratch = CdfScratch::default();
        self.push_pdf_symbols_with_scratch(
            pdf,
            n_bins,
            n_cols,
            symbols,
            fp_scale,
            min_range,
            &mut scratch,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn push_pdf_symbols_with_scratch(
        &mut self,
        pdf: &[f64],
        n_bins: usize,
        n_cols: usize,
        symbols: &[usize],
        fp_scale: i64,
        min_range: i64,
        scratch: &mut CdfScratch,
    ) -> Result<()> {
        if symbols.len() != n_cols {
            bail!(
                "symbol length {} does not match pdf column count {}",
                symbols.len(),
                n_cols
            );
        }
        let cdf = deterministic_cdf_multi_with_scratch(
            pdf,
            n_bins,
            n_cols,
            self.total_range_bits,
            fp_scale,
            min_range,
            scratch,
        )?;
        for (col, symbol) in symbols.iter().copied().enumerate() {
            self.push_symbol(symbol, cdf, n_bins, n_cols, col)?;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn push_valid_pdf_symbols_with_scratch(
        &mut self,
        pdf: &[f64],
        n_bins: usize,
        n_cols: usize,
        symbols: &[usize],
        fp_scale: i64,
        min_range: i64,
        scratch: &mut CdfScratch,
    ) -> Result<()> {
        if symbols.len() != n_cols {
            bail!(
                "symbol length {} does not match pdf column count {}",
                symbols.len(),
                n_cols
            );
        }
        let cdf = deterministic_cdf_multi_from_valid_pdf_with_scratch(
            pdf,
            n_bins,
            n_cols,
            self.total_range_bits,
            fp_scale,
            min_range,
            scratch,
        )?;
        for (col, symbol) in symbols.iter().copied().enumerate() {
            self.push_symbol(symbol, cdf, n_bins, n_cols, col)?;
        }
        Ok(())
    }

    pub fn finish(&mut self) -> Vec<u8> {
        while self.max_bit >= 0 {
            let bit = ((self.low >> self.max_bit as u32) & 1) as u8;
            self.writer.push_bit(bit);
            self.max_bit -= 1;
        }
        std::mem::replace(&mut self.writer, BitWriter::new()).finish()
    }

    fn delta(&self) -> u64 {
        self.high - self.low + 1
    }

    fn flush_common_prefix(&mut self) {
        while self.max_bit >= 0 {
            let b1 = self.low >> self.max_bit as u32;
            let b2 = self.high >> self.max_bit as u32;
            if b1 == b2 {
                self.low -= b1 << self.max_bit as u32;
                self.high -= b1 << self.max_bit as u32;
                self.max_bit -= 1;
                self.writer.push_bit(b1 as u8);
            } else {
                break;
            }
        }
    }

    fn push_symbol(
        &mut self,
        symbol: usize,
        cdf: &[i64],
        n_bins: usize,
        n_cols: usize,
        col: usize,
    ) -> Result<()> {
        while self.delta() < (1_u64 << self.total_range_bits) {
            self.low <<= 1;
            self.high = (self.high << 1) | 1;
            self.max_bit += 1;
        }
        if symbol >= n_bins {
            bail!("symbol {symbol} is out of range for {n_bins} bins");
        }
        let total = 1_u64 << self.total_range_bits;
        let range = self.delta();
        let cum_high = cdf[symbol * n_cols + col] as u64;
        let cum_low = if symbol == 0 {
            0
        } else {
            cdf[(symbol - 1) * n_cols + col] as u64
        };
        let base = self.low;
        self.low = base + (range * cum_low) / total;
        self.high = base + (range * cum_high) / total - 1;
        self.flush_common_prefix();
        Ok(())
    }
}

pub struct ArithmeticDecoder {
    total_range_bits: u32,
    low: u64,
    high: u64,
    current: u64,
    max_bit: i32,
    reader: BitReader,
}

impl ArithmeticDecoder {
    pub fn new(data: Vec<u8>, total_range_bits: u32) -> Result<Self> {
        if total_range_bits > 30 {
            bail!("total_range_bits must be <= 30");
        }
        Ok(Self {
            total_range_bits,
            low: 0,
            high: 0,
            current: 0,
            max_bit: -1,
            reader: BitReader::new(data),
        })
    }

    pub fn pull_symbols(
        &mut self,
        pdf: &[f64],
        n_bins: usize,
        n_cols: usize,
        fp_scale: i64,
        min_range: i64,
    ) -> Result<Vec<usize>> {
        let mut scratch = CdfScratch::default();
        self.pull_symbols_with_scratch(pdf, n_bins, n_cols, fp_scale, min_range, &mut scratch)
    }

    pub fn pull_symbols_with_scratch(
        &mut self,
        pdf: &[f64],
        n_bins: usize,
        n_cols: usize,
        fp_scale: i64,
        min_range: i64,
        scratch: &mut CdfScratch,
    ) -> Result<Vec<usize>> {
        let mut out = vec![0; n_cols];
        self.pull_symbols_into_with_scratch(
            pdf, n_bins, n_cols, fp_scale, min_range, scratch, &mut out,
        )?;
        Ok(out)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn pull_symbols_into_with_scratch(
        &mut self,
        pdf: &[f64],
        n_bins: usize,
        n_cols: usize,
        fp_scale: i64,
        min_range: i64,
        scratch: &mut CdfScratch,
        out: &mut [usize],
    ) -> Result<()> {
        if out.len() != n_cols {
            bail!(
                "output symbol count {} does not match pdf column count {}",
                out.len(),
                n_cols,
            );
        }
        let cdf = deterministic_cdf_multi_with_scratch(
            pdf,
            n_bins,
            n_cols,
            self.total_range_bits,
            fp_scale,
            min_range,
            scratch,
        )?;
        for (col, symbol) in out.iter_mut().enumerate() {
            *symbol = self.pull_symbol(cdf, n_bins, n_cols, col)?;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn pull_valid_pdf_symbols_into_with_scratch(
        &mut self,
        pdf: &[f64],
        n_bins: usize,
        n_cols: usize,
        fp_scale: i64,
        min_range: i64,
        scratch: &mut CdfScratch,
        out: &mut [usize],
    ) -> Result<()> {
        if out.len() != n_cols {
            bail!(
                "output symbol count {} does not match pdf column count {}",
                out.len(),
                n_cols,
            );
        }
        let cdf = deterministic_cdf_multi_from_valid_pdf_with_scratch(
            pdf,
            n_bins,
            n_cols,
            self.total_range_bits,
            fp_scale,
            min_range,
            scratch,
        )?;
        for (col, symbol) in out.iter_mut().enumerate() {
            *symbol = self.pull_symbol(cdf, n_bins, n_cols, col)?;
        }
        Ok(())
    }

    fn delta(&self) -> u64 {
        self.high - self.low + 1
    }

    fn flush_common_prefix(&mut self) {
        while self.max_bit >= 0 {
            let b1 = self.low >> self.max_bit as u32;
            let b2 = self.high >> self.max_bit as u32;
            if b1 == b2 {
                self.low -= b1 << self.max_bit as u32;
                self.high -= b1 << self.max_bit as u32;
                self.current -= b1 << self.max_bit as u32;
                self.max_bit -= 1;
            } else {
                break;
            }
        }
    }

    fn pull_symbol(
        &mut self,
        cdf: &[i64],
        n_bins: usize,
        n_cols: usize,
        col: usize,
    ) -> Result<usize> {
        while self.delta() < (1_u64 << self.total_range_bits) {
            let bit = self
                .reader
                .pull_bit()
                .ok_or_else(|| anyhow::anyhow!("arithmetic stream exhausted"))?
                as u64;
            self.low <<= 1;
            self.high = (self.high << 1) | 1;
            self.current = (self.current << 1) | bit;
            self.max_bit += 1;
        }

        let total = 1_u64 << self.total_range_bits;
        let range = self.delta();
        let target = (((self.current - self.low + 1) * total) - 1) / range;
        let mut lo = 0usize;
        let mut hi = n_bins;
        while lo < hi {
            let mid = (lo + hi) / 2;
            let value = cdf[mid * n_cols + col] as u64;
            if target < value {
                hi = mid;
            } else {
                lo = mid + 1;
            }
        }
        if lo >= n_bins {
            bail!("arithmetic decoder binary search failed");
        }
        let symbol = lo;
        let cum_high = cdf[symbol * n_cols + col] as u64;
        let cum_low = if symbol == 0 {
            0
        } else {
            cdf[(symbol - 1) * n_cols + col] as u64
        };
        let base = self.low;
        self.low = base + (range * cum_low) / total;
        self.high = base + (range * cum_high) / total - 1;
        self.flush_common_prefix();
        Ok(symbol)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn valid_pdf_fast_path_matches_the_defensive_cdf_path() {
        let mut seed = 0x517c_c1b7_2722_0a95_u64;
        for (n_bins, n_cols) in [(1_usize, 1_usize), (7, 3), (31, 8), (1024, 8)] {
            let mut pdf = vec![0.0_f64; n_bins * n_cols];
            for col in 0..n_cols {
                let mut sum = 0.0_f64;
                for row in 0..n_bins {
                    seed = seed
                        .wrapping_mul(6_364_136_223_846_793_005)
                        .wrapping_add(1_442_695_040_888_963_407);
                    let value = ((seed >> 32) as f64 + 1.0) / (u32::MAX as f64 + 1.0);
                    pdf[row * n_cols + col] = value;
                    sum += value;
                }
                for row in 0..n_bins {
                    pdf[row * n_cols + col] /= sum;
                }
            }

            let mut defensive_scratch = CdfScratch::default();
            let defensive = deterministic_cdf_multi_with_scratch(
                &pdf,
                n_bins,
                n_cols,
                24,
                1 << 13,
                2,
                &mut defensive_scratch,
            )
            .unwrap()
            .to_vec();
            let mut valid_scratch = CdfScratch::default();
            let valid = deterministic_cdf_multi_from_valid_pdf_with_scratch(
                &pdf,
                n_bins,
                n_cols,
                24,
                1 << 13,
                2,
                &mut valid_scratch,
            )
            .unwrap();
            assert_eq!(defensive, valid);
        }
    }

    fn increment_largest_remainders_by_full_sort(
        base: &mut [i64],
        order: &mut [(i64, usize)],
        remainder: usize,
    ) {
        order.sort_by(|left, right| right.cmp(left));
        for &(_, row) in &order[..remainder] {
            base[row] += 1;
        }
    }

    #[test]
    fn remainder_selection_matches_full_sort_exactly() {
        let mut seed = 0x9e37_79b9_7f4a_7c15_u64;
        for n_bins in [1_usize, 2, 3, 7, 31, 256, 1024] {
            for remainder in 0..n_bins {
                let mut order = Vec::with_capacity(n_bins);
                for row in 0..n_bins {
                    seed = seed
                        .wrapping_mul(6_364_136_223_846_793_005)
                        .wrapping_add(1_442_695_040_888_963_407);
                    // Deliberately use a small priority range to exercise ties.
                    let priority = ((seed >> 32) % 17) as i64;
                    order.push((priority, row));
                }

                let mut selected_base = vec![0_i64; n_bins];
                let mut sorted_base = vec![0_i64; n_bins];
                let mut selected_order = order.clone();
                let mut sorted_order = order;
                increment_largest_remainders(&mut selected_base, &mut selected_order, remainder);
                increment_largest_remainders_by_full_sort(
                    &mut sorted_base,
                    &mut sorted_order,
                    remainder,
                );
                assert_eq!(
                    selected_base, sorted_base,
                    "n_bins={n_bins} remainder={remainder}"
                );
            }
        }
    }

    #[test]
    fn cdf_scratch_reuse_matches_fresh_storage_across_shapes() {
        let cases = [
            (3_usize, 2_usize, vec![0.7, 0.0, 0.2, 0.0, 0.1, 0.0]),
            (5, 1, vec![0.125, 0.25, 0.375, 0.125, 0.125]),
            (2, 4, vec![0.1, 0.2, 0.3, 0.4, 0.9, 0.8, 0.7, 0.6]),
            (1, 1, vec![1.0]),
        ];
        let mut scratch = CdfScratch::default();

        for _ in 0..2 {
            for (n_bins, n_cols, pdf) in &cases {
                let expected =
                    deterministic_cdf_multi(pdf, *n_bins, *n_cols, 24, 1 << 13, 2).unwrap();
                let actual = deterministic_cdf_multi_with_scratch(
                    pdf,
                    *n_bins,
                    *n_cols,
                    24,
                    1 << 13,
                    2,
                    &mut scratch,
                )
                .unwrap();
                assert_eq!(actual, expected);
            }
        }
    }

    #[test]
    fn arithmetic_roundtrip_matches_symbols() {
        let pdf = vec![
            0.7, 0.2, 0.1, //
            0.2, 0.5, 0.2, //
            0.1, 0.3, 0.7, //
        ];
        let symbols = vec![0, 1, 2];
        let mut encoder = ArithmeticEncoder::new(24).unwrap();
        encoder
            .push_pdf_symbols(&pdf, 3, 3, &symbols, 1 << 13, 2)
            .unwrap();
        let bytes = encoder.finish();

        let mut decoder = ArithmeticDecoder::new(bytes, 24).unwrap();
        let decoded = decoder.pull_symbols(&pdf, 3, 3, 1 << 13, 2).unwrap();
        assert_eq!(decoded, symbols);
    }
}
