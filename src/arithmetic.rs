use anyhow::{Result, bail};

const EPS_EDGE: f64 = 9.094_947_017_729_282e-13;
const EPS_PERTURB: f64 = 8.673_617_379_884_035e-19;

fn counts_from_pdf_flat(pdf: &[f64], fp_scale: i64) -> Vec<i64> {
    let mut out = Vec::with_capacity(pdf.len());
    let scale = fp_scale as f64;
    for (idx, value) in pdf.iter().enumerate() {
        let mut x = value.max(0.0) * scale;
        let frac = x - x.floor();
        if frac <= EPS_EDGE || frac >= 1.0 - EPS_EDGE {
            let sign = if idx % 2 == 0 { -1.0 } else { 1.0 };
            x = (x + sign * EPS_PERTURB).max(0.0);
        }
        out.push(x.floor() as i64);
    }
    out
}

fn deterministic_cdf_multi_impl(
    pdf: &[f64],
    n_bins: usize,
    n_cols: usize,
    total_range_bits: u32,
    fp_scale: i64,
    min_range: i64,
) -> Result<Vec<i64>> {
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

    let mut normalized = vec![0.0_f64; pdf.len()];
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

    let mut counts = counts_from_pdf_flat(&normalized, fp_scale);
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

    let mut cdf = vec![0_i64; pdf.len()];
    for col in 0..n_cols {
        let mut num_sum = 0_i64;
        for row in 0..n_bins {
            num_sum += counts[row * n_cols + col];
        }
        if num_sum <= 0 {
            bail!("invalid zero-count column in pdf matrix");
        }

        let mut base = vec![0_i64; n_bins];
        let mut base_sum = 0_i64;
        for row in 0..n_bins {
            let num = counts[row * n_cols + col];
            let value = (alloc * num) / num_sum;
            base[row] = value;
            base_sum += value;
        }

        let remainder = alloc - base_sum;
        if remainder > 0 {
            let mut order: Vec<(i64, usize)> = (0..n_bins)
                .map(|row| {
                    let num = counts[row * n_cols + col];
                    let prio = (alloc * num) - (num_sum * base[row]);
                    let key = prio * (n_bins as i64 + 1) - row as i64;
                    (key, row)
                })
                .collect();
            order.sort_by(|left, right| right.cmp(left));
            for (_, row) in order.into_iter().take(remainder as usize) {
                base[row] += 1;
            }
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
        if symbols.len() != n_cols {
            bail!(
                "symbol length {} does not match pdf column count {}",
                symbols.len(),
                n_cols
            );
        }
        let cdf = deterministic_cdf_multi_impl(
            pdf,
            n_bins,
            n_cols,
            self.total_range_bits,
            fp_scale,
            min_range,
        )?;
        for (col, symbol) in symbols.iter().copied().enumerate() {
            self.push_symbol(symbol, &cdf, n_bins, n_cols, col)?;
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
        let cdf = deterministic_cdf_multi_impl(
            pdf,
            n_bins,
            n_cols,
            self.total_range_bits,
            fp_scale,
            min_range,
        )?;
        let mut out = Vec::with_capacity(n_cols);
        for col in 0..n_cols {
            out.push(self.pull_symbol(&cdf, n_bins, n_cols, col)?);
        }
        Ok(out)
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
