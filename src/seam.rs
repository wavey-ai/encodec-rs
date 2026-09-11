use crate::ecdc_presets::fixed_context_samples;
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

/// One owned span the cursor has made final, and is ready to hand to a caller.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SeamSegment {
    pub chunk_index: usize,
    pub start_frame: usize,
    pub end_frame: usize,
}

/// The streaming decode-side mirror of [`triangle_overlap_add_planar_frames`].
///
/// A record is decoded one model window at a time, so it is never all in
/// memory. This cursor holds the overlap-add state — the triangle weights, the
/// guard context, and the running sums for the samples that are not yet final
/// — accepts windows (and the stream's own silent and cached spans), and hands
/// back only the owned PCM that can no longer change. It is the decode-side
/// counterpart of the encode window cursor, and it lives here so the two
/// cannot disagree about the geometry.
///
/// Samples are accumulated as the JavaScript player does: multiply and add in
/// f64 and round to f32 on store, so a port of the same algorithm agrees to the
/// bit. The output is signed 16-bit planar PCM.
pub struct SeamCursor {
    channels: usize,
    model_samples: usize,
    owned_samples: usize,
    context: usize,
    audio_length: usize,
    frame_count: usize,
    retain_full: bool,
    weight: Vec<f32>,
    capacity: usize,
    mask: usize,
    sums: Vec<f32>,
    total_weight: Vec<f32>,
    final_ring: Vec<i16>,
    channel_data: Vec<i16>,
    emitted: usize,
    max_written: usize,
    emitted_segment_index: usize,
    collected: usize,
    cached_ranges: Vec<(usize, usize)>,
    cached_pointer: usize,
}

impl SeamCursor {
    pub fn new(
        channels: usize,
        model_samples: usize,
        owned_samples: usize,
        audio_length: usize,
        frame_count: usize,
        retain_full: bool,
    ) -> Result<Self> {
        if channels == 0 {
            bail!("seam cursor requires at least one channel");
        }
        if owned_samples == 0 || model_samples == 0 || owned_samples > model_samples {
            bail!("seam cursor requires 1 <= owned <= model window samples");
        }
        let context = fixed_context_samples(model_samples, owned_samples)?
            .ok_or_else(|| anyhow::anyhow!("seam cursor requires a fixed-context geometry"))?;
        let weight = (0..model_samples)
            .map(|sample| {
                let position = (sample + 1) as f64 / (model_samples + 1) as f64;
                (0.5_f64 - (position - 0.5_f64).abs()) as f32
            })
            .collect();
        let capacity = next_pow2((model_samples.saturating_mul(2)).max(1));
        let channel_data = if retain_full {
            vec![0_i16; channels.saturating_mul(audio_length)]
        } else {
            Vec::new()
        };
        Ok(Self {
            channels,
            model_samples,
            owned_samples,
            context,
            audio_length,
            frame_count,
            retain_full,
            weight,
            capacity,
            mask: capacity - 1,
            sums: vec![0.0_f32; channels.saturating_mul(capacity)],
            total_weight: vec![0.0_f32; capacity],
            final_ring: vec![0_i16; channels.saturating_mul(capacity)],
            channel_data,
            emitted: 0,
            max_written: 0,
            emitted_segment_index: 0,
            collected: 0,
            cached_ranges: Vec::new(),
            cached_pointer: 0,
        })
    }

    fn window_start(&self, frame_index: usize) -> isize {
        (frame_index.saturating_mul(self.owned_samples)) as isize - self.context as isize
    }

    /// `(source_start, source_end, written_end)` for one placed window, in the
    /// same clipping the player uses: nothing before sample zero, nothing past
    /// the programme length.
    fn bounds(&self, window_start: isize) -> (usize, usize, usize) {
        let source_start = 0_isize.max(-window_start) as usize;
        let remaining = self.audio_length as isize - window_start;
        let source_end = source_start
            .max((self.model_samples as isize).min(remaining).max(0) as usize);
        let written = window_start + source_end as isize;
        let written_end = 0_isize
            .max((self.audio_length as isize).min(written)) as usize;
        (source_start, source_end, written_end)
    }

    fn ensure_span(&mut self, end_sample: usize) {
        if end_sample.saturating_sub(self.emitted) > self.capacity {
            self.grow(end_sample.saturating_sub(self.emitted));
        }
    }

    fn grow(&mut self, required: usize) {
        let capacity = next_pow2(required);
        let mask = capacity - 1;
        let mut sums = vec![0.0_f32; self.channels * capacity];
        let mut total_weight = vec![0.0_f32; capacity];
        let mut final_ring = vec![0_i16; self.channels * capacity];
        for global in self.emitted..self.max_written {
            let old = global & self.mask;
            let new = global & mask;
            total_weight[new] = self.total_weight[old];
            for channel in 0..self.channels {
                sums[channel * capacity + new] = self.sums[channel * self.capacity + old];
            }
        }
        for global in self.collected..self.max_written {
            let old = global & self.mask;
            let new = global & mask;
            for channel in 0..self.channels {
                final_ring[channel * capacity + new] = self.final_ring[channel * self.capacity + old];
            }
        }
        self.sums = sums;
        self.total_weight = total_weight;
        self.final_ring = final_ring;
        self.capacity = capacity;
        self.mask = mask;
    }

    /// Adds one decoded planar model window (`channels * model_samples`) at
    /// `frame_index`, weighted across the window.
    pub fn add_decoded_frame(&mut self, frame_index: usize, window: &[f32]) -> Result<()> {
        let decoded_samples = window.len() / self.channels;
        if decoded_samples < self.model_samples {
            bail!(
                "decoded frame {} has {} samples per channel; expected {} including guards",
                frame_index + 1,
                decoded_samples,
                self.model_samples,
            );
        }
        let window_start = self.window_start(frame_index);
        let (source_start, source_end, written_end) = self.bounds(window_start);
        self.ensure_span(written_end);
        for sample in source_start..source_end {
            let global = window_start + sample as isize;
            if global < 0 || global as usize >= self.audio_length {
                continue;
            }
            let global = global as usize;
            if global < self.emitted {
                continue;
            }
            let ring = global & self.mask;
            let w = self.weight[sample] as f64;
            self.total_weight[ring] = ((self.total_weight[ring] as f64) + w) as f32;
            for channel in 0..self.channels {
                let value = window[channel * decoded_samples + sample] as f64;
                let index = channel * self.capacity + ring;
                self.sums[index] = ((self.sums[index] as f64) + value * w) as f32;
            }
        }
        self.max_written = self.max_written.max(written_end);
        Ok(())
    }

    /// Adds a span whose PCM is already final (a cache hit), bypassing the
    /// overlap-add. Interleaved signed 16-bit.
    pub fn add_cached_range(&mut self, start: usize, end: usize, interleaved: &[i16]) -> bool {
        let start = start.min(self.audio_length);
        let end = end.min(self.audio_length).max(start);
        if end <= start || interleaved.len() < (end - start) * self.channels {
            return false;
        }
        self.ensure_span(end);
        for global in start..end {
            let local = (global - start) * self.channels;
            let ring = global & self.mask;
            for channel in 0..self.channels {
                let sample = interleaved[local + channel];
                self.final_ring[channel * self.capacity + ring] = sample;
                if self.retain_full {
                    self.channel_data[channel * self.audio_length + global] = sample;
                }
            }
        }
        self.cached_ranges.push((start, end));
        self.max_written = self.max_written.max(end);
        true
    }

    /// Adds a silent model window (an inter-track gap), weighted exactly as a
    /// decoded window but contributing no signal.
    pub fn add_silent_frame(&mut self, frame_index: usize) -> Result<()> {
        let window_start = self.window_start(frame_index);
        let (source_start, source_end, written_end) = self.bounds(window_start);
        self.ensure_span(written_end);
        for sample in source_start..source_end {
            let global = window_start + sample as isize;
            if global < 0 || global as usize >= self.audio_length {
                continue;
            }
            let global = global as usize;
            if global < self.emitted {
                continue;
            }
            let ring = global & self.mask;
            let w = self.weight[sample] as f64;
            self.total_weight[ring] = ((self.total_weight[ring] as f64) + w) as f32;
        }
        self.max_written = self.max_written.max(written_end);
        Ok(())
    }

    fn emit_until(&mut self, sample_end: usize) {
        let safe_end = sample_end
            .max(self.emitted)
            .min(self.audio_length);
        for global in self.emitted..safe_end {
            let ring = global & self.mask;
            while self.cached_pointer < self.cached_ranges.len()
                && global >= self.cached_ranges[self.cached_pointer].1
            {
                self.cached_pointer += 1;
            }
            let is_cached = self.cached_pointer < self.cached_ranges.len()
                && global >= self.cached_ranges[self.cached_pointer].0;
            let denominator = self.total_weight[ring] as f64;
            for channel in 0..self.channels {
                let index = channel * self.capacity + ring;
                if !is_cached {
                    let value = if denominator > 0.0 {
                        self.sums[index] as f64 / denominator
                    } else {
                        0.0
                    };
                    let sample = float_to_s16(value);
                    self.final_ring[index] = sample;
                    if self.retain_full {
                        self.channel_data[channel * self.audio_length + global] = sample;
                    }
                }
                self.sums[index] = 0.0;
            }
            self.total_weight[ring] = 0.0;
        }
        self.emitted = safe_end;
        self.max_written = self.max_written.max(self.emitted);
    }

    fn safe_emit_end(&self, next_frame_index: usize) -> usize {
        if next_frame_index >= self.frame_count {
            return self.audio_length;
        }
        let window_start = self.window_start(next_frame_index);
        window_start.max(0).min(self.audio_length as isize) as usize
    }

    fn collect(&mut self) -> Vec<SeamSegment> {
        let mut segments = Vec::new();
        while self.emitted_segment_index < self.frame_count {
            let index = self.emitted_segment_index;
            let start = self
                .audio_length
                .min(index.saturating_mul(self.owned_samples));
            let end = if index + 1 < self.frame_count {
                self.audio_length
                    .min((index + 1).saturating_mul(self.owned_samples))
            } else {
                self.audio_length
            }
            .max(start);
            if end > self.emitted {
                break;
            }
            if end > start {
                segments.push(SeamSegment {
                    chunk_index: index,
                    start_frame: start,
                    end_frame: end,
                });
            }
            self.collected = self.collected.max(end);
            self.emitted_segment_index += 1;
        }
        segments
    }

    /// Emits everything that is final before the next undecoded window, and
    /// returns the owned spans that just became available.
    pub fn emit_after_batch(&mut self, next_frame_index: usize) -> Vec<SeamSegment> {
        let end = self.safe_emit_end(next_frame_index);
        self.emit_until(end);
        self.collect()
    }

    /// Emits the whole tail and returns the remaining owned spans.
    pub fn flush(&mut self) -> Vec<SeamSegment> {
        self.emit_until(self.audio_length);
        self.collect()
    }

    /// The PCM for a span, planar. Reads the retained full buffer when it is
    /// held, otherwise the ring, which still covers everything since the last
    /// collected span.
    pub fn segment_pcm(&self, segment: &SeamSegment) -> Vec<i16> {
        let length = segment.end_frame - segment.start_frame;
        let mut pcm = Vec::with_capacity(self.channels * length);
        for channel in 0..self.channels {
            for global in segment.start_frame..segment.end_frame {
                let sample = if self.retain_full {
                    self.channel_data[channel * self.audio_length + global]
                } else {
                    self.final_ring[channel * self.capacity + (global & self.mask)]
                };
                pcm.push(sample);
            }
        }
        pcm
    }

    /// The complete planar channel data, for a caller that retained it.
    pub fn full_channel_data(&self) -> &[i16] {
        &self.channel_data
    }

    pub fn retains_full_pcm(&self) -> bool {
        self.retain_full
    }
}

fn next_pow2(value: usize) -> usize {
    let mut capacity = 1_usize;
    while capacity < value {
        capacity *= 2;
    }
    capacity
}

/// `Math.round`: halves go toward positive infinity, unlike Rust's `round`.
fn js_round(value: f64) -> f64 {
    (value + 0.5).floor()
}

fn float_to_s16(value: f64) -> i16 {
    let value = if value.is_finite() { value } else { 0.0 };
    let value = value.clamp(-1.0, 1.0);
    let rounded = if value < 0.0 {
        js_round(value * 32_768.0)
    } else {
        js_round(value * 32_767.0)
    };
    rounded.clamp(-32_768.0, 32_767.0) as i16
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

    fn ramp(length: usize, base: f32) -> Vec<f32> {
        (0..length)
            .map(|index| (index as f32 * 0.0001 + base).sin() * 0.4)
            .collect()
    }

    #[test]
    fn seam_cursor_matches_batch_overlap_add_cropped_by_its_context() {
        let (channels, model, owned) = (1_usize, 64_960_usize, 64_000_usize);
        let frame_count = 2_usize;
        let audio_length = owned * frame_count;
        let window0 = ramp(model, 0.1);
        let window1 = ramp(model, 0.5);

        let mut cursor =
            SeamCursor::new(channels, model, owned, audio_length, frame_count, true).unwrap();
        cursor.add_decoded_frame(0, &window0).unwrap();
        cursor.emit_after_batch(1);
        cursor.add_decoded_frame(1, &window1).unwrap();
        cursor.flush();

        let mut windows = Vec::new();
        windows.extend_from_slice(&window0);
        windows.extend_from_slice(&window1);
        let batch = triangle_overlap_add_planar_frames(&windows, 2, channels, model, owned).unwrap();
        let context = (model - owned) / 2;
        let output = cursor.full_channel_data();
        for sample in 0..audio_length {
            let expected = float_to_s16(batch[context + sample] as f64);
            assert!(
                (output[sample] as i32 - expected as i32).abs() <= 1,
                "sample {sample}: {} != {expected}",
                output[sample],
            );
        }
    }

    #[test]
    fn seam_cursor_segments_cover_the_programme_in_owned_spans() {
        let (channels, model, owned) = (1_usize, 64_960_usize, 64_000_usize);
        let frame_count = 3_usize;
        let audio_length = owned * frame_count;
        let window = ramp(model, 0.2);
        let mut cursor =
            SeamCursor::new(channels, model, owned, audio_length, frame_count, true).unwrap();
        let mut segments = Vec::new();
        for frame in 0..frame_count {
            cursor.add_decoded_frame(frame, &window).unwrap();
            segments.extend(cursor.emit_after_batch(frame + 1));
        }
        segments.extend(cursor.flush());
        let mut expected_start = 0;
        for segment in &segments {
            assert_eq!(segment.start_frame, expected_start);
            assert_eq!(segment.end_frame, expected_start + owned);
            expected_start = segment.end_frame;
        }
        assert_eq!(expected_start, audio_length);
    }
}
