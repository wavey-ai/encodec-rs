use std::fs;
use std::io::Cursor;
use std::path::PathBuf;

use frame_header::{EncodingFlag, FrameHeader};
use libopus_rs::Decoder as LibopusDecoder;
use serde_json::Value;
use soundkit::audio_packet::Decoder as SoundkitPacketDecoder;
use soundkit::audio_types::PcmData;
use soundkit::wav::generate_wav_buffer;

const CACHE_MAGIC: &[u8; 4] = b"BNP1";
const CACHE_ENVELOPE_FORMAT: &str = "bitneedle-player-cache-v1";
const OPUS_PACKET_AUDIO_FORMAT: &str = "soundkit_opus_packets";

#[derive(Debug, Default)]
struct Options {
    input_cache: PathBuf,
    output_wav: PathBuf,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let options = parse_options()?;
    let body = fs::read(&options.input_cache)?;
    let decoded = decode_bitneedle_soundkit_opus_cache(&body)?;
    let wav = generate_wav_buffer(
        &PcmData::I16(split_interleaved_i16(
            &decoded.samples,
            decoded.channels,
            decoded.frames,
        )?),
        decoded.sample_rate,
    )?;

    if let Some(parent) = options.output_wav.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(&options.output_wav, wav)?;

    println!(
        "{}",
        serde_json::to_string(&serde_json::json!({
            "input": options.input_cache.display().to_string(),
            "output": options.output_wav.display().to_string(),
            "format": OPUS_PACKET_AUDIO_FORMAT,
            "sampleRate": decoded.sample_rate,
            "channels": decoded.channels,
            "frames": decoded.frames,
            "duration": decoded.frames as f64 / decoded.sample_rate as f64,
            "packetCount": decoded.packet_count,
        }))?
    );
    Ok(())
}

fn parse_options() -> Result<Options, Box<dyn std::error::Error>> {
    let mut options = Options::default();
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut index = 0;
    while index < args.len() {
        let key = &args[index];
        let value = |index: &mut usize| -> Result<String, Box<dyn std::error::Error>> {
            *index += 1;
            args.get(*index)
                .cloned()
                .ok_or_else(|| format!("missing value for {key}").into())
        };
        match key.as_str() {
            "--input-cache" => options.input_cache = PathBuf::from(value(&mut index)?),
            "--output-wav" => options.output_wav = PathBuf::from(value(&mut index)?),
            _ => return Err(format!("unknown argument: {key}").into()),
        }
        index += 1;
    }

    if options.input_cache.as_os_str().is_empty() {
        return Err("--input-cache is required".into());
    }
    if options.output_wav.as_os_str().is_empty() {
        return Err("--output-wav is required".into());
    }
    Ok(options)
}

struct DecodedAudio {
    samples: Vec<i16>,
    sample_rate: u32,
    channels: usize,
    frames: usize,
    packet_count: usize,
}

fn decode_bitneedle_soundkit_opus_cache(
    body: &[u8],
) -> Result<DecodedAudio, Box<dyn std::error::Error>> {
    if body.len() < 8 || &body[0..4] != CACHE_MAGIC {
        return Err("input is not a BNP1 Bitneedle player cache body".into());
    }
    let json_len = u32::from_be_bytes(body[4..8].try_into().expect("slice length")) as usize;
    let json_end = 8 + json_len;
    if json_end > body.len() {
        return Err("truncated Bitneedle player cache envelope".into());
    }
    let envelope: Value = serde_json::from_slice(&body[8..json_end])?;
    if envelope.get("format").and_then(Value::as_str) != Some(CACHE_ENVELOPE_FORMAT) {
        return Err("unsupported Bitneedle player cache envelope format".into());
    }
    let payload = envelope
        .get("payload")
        .ok_or("Bitneedle player cache is missing payload")?;
    if decoded_segment_cache_format(payload) != OPUS_PACKET_AUDIO_FORMAT {
        return Err("Bitneedle player cache payload is not SoundKit Opus packets".into());
    }
    let buffers = read_envelope_buffers(&envelope, body, json_end)?;
    let packet_indexes = soundkit_packet_indexes(payload)?;
    if packet_indexes.is_empty() {
        return Err("Bitneedle player cache has no SoundKit packets".into());
    }

    let mut sample_rate = 0_u32;
    let mut channels = 0_usize;
    let mut decoder: Option<PlayerSoundkitOpusDecoder> = None;
    let mut samples = Vec::new();

    for (packet_number, buffer_index) in packet_indexes.iter().copied().enumerate() {
        let packet = buffers.get(buffer_index).ok_or_else(|| {
            format!("packet {packet_number} references missing buffer {buffer_index}")
        })?;
        let header = FrameHeader::decode(&mut Cursor::new(packet.as_slice()))?;
        if header.encoding() != &EncodingFlag::Opus {
            return Err(format!("packet {packet_number} is not Opus").into());
        }
        if sample_rate == 0 {
            sample_rate = header.sample_rate();
            channels = header.channels() as usize;
            decoder = Some(PlayerSoundkitOpusDecoder::new(sample_rate, channels)?);
        } else if header.sample_rate() != sample_rate || header.channels() as usize != channels {
            return Err(
                format!("packet {packet_number} changes sample rate or channel count").into(),
            );
        }
        let payload_bytes = &packet[header.size()..];
        let mut packet_pcm = vec![0_i16; header.sample_size() as usize * channels];
        let decoded_samples = decoder.as_mut().expect("decoder initialized").decode_i16(
            payload_bytes,
            &mut packet_pcm,
            false,
        )?;
        samples.extend_from_slice(&packet_pcm[..decoded_samples]);
    }

    let declared_frames = payload
        .get("audioLength")
        .or_else(|| payload.get("endFrame"))
        .and_then(Value::as_u64)
        .map(|value| value as usize)
        .unwrap_or_else(|| samples.len() / channels.max(1));
    let target_samples = declared_frames.saturating_mul(channels);
    if samples.len() > target_samples {
        samples.truncate(target_samples);
    }

    Ok(DecodedAudio {
        samples,
        sample_rate,
        channels,
        frames: declared_frames,
        packet_count: packet_indexes.len(),
    })
}

fn read_envelope_buffers(
    envelope: &Value,
    body: &[u8],
    mut offset: usize,
) -> Result<Vec<Vec<u8>>, Box<dyn std::error::Error>> {
    let entries = envelope
        .get("buffers")
        .and_then(Value::as_array)
        .ok_or("Bitneedle player cache is missing buffers")?;
    let mut buffers = Vec::with_capacity(entries.len());
    for (index, entry) in entries.iter().enumerate() {
        let byte_length = entry
            .get("byteLength")
            .and_then(Value::as_u64)
            .ok_or_else(|| format!("buffer {index} is missing byteLength"))?
            as usize;
        let end = offset + byte_length;
        if end > body.len() {
            return Err(format!("buffer {index} extends beyond cache body").into());
        }
        buffers.push(body[offset..end].to_vec());
        offset = end;
    }
    Ok(buffers)
}

fn decoded_segment_cache_format(payload: &Value) -> String {
    payload
        .get("audioFormat")
        .or_else(|| payload.get("audio_format"))
        .or_else(|| payload.get("format"))
        .or_else(|| payload.get("codec"))
        .and_then(Value::as_str)
        .unwrap_or("")
        .trim()
        .to_ascii_lowercase()
        .chars()
        .map(|ch| if ch.is_ascii_alphanumeric() { ch } else { '_' })
        .collect::<String>()
        .trim_matches('_')
        .to_string()
}

fn soundkit_packet_indexes(payload: &Value) -> Result<Vec<usize>, Box<dyn std::error::Error>> {
    let packets = payload
        .get("soundkitPackets")
        .or_else(|| payload.get("packets"))
        .or_else(|| payload.get("opusPackets"))
        .and_then(Value::as_array)
        .ok_or("Bitneedle player cache payload has no packet placeholders")?;
    packets
        .iter()
        .enumerate()
        .map(|(index, packet)| {
            packet
                .get("__bitneedleBuffer")
                .and_then(Value::as_u64)
                .map(|value| value as usize)
                .ok_or_else(|| format!("packet {index} has no __bitneedleBuffer").into())
        })
        .collect()
}

fn split_interleaved_i16(
    samples: &[i16],
    channels: usize,
    frames: usize,
) -> Result<Vec<Vec<i16>>, Box<dyn std::error::Error>> {
    if channels == 0 {
        return Err("decoded audio has no channels".into());
    }
    let frame_count = frames.min(samples.len() / channels);
    let mut output = vec![vec![0_i16; frame_count]; channels];
    for frame in 0..frame_count {
        for channel in 0..channels {
            output[channel][frame] = samples[frame * channels + channel];
        }
    }
    Ok(output)
}

struct PlayerSoundkitOpusDecoder {
    inner: LibopusDecoder,
}

impl PlayerSoundkitOpusDecoder {
    fn new(sample_rate: u32, channels: usize) -> Result<Self, Box<dyn std::error::Error>> {
        Ok(Self {
            inner: LibopusDecoder::new(sample_rate as i32, channels)
                .map_err(|error| format!("failed to create libopus-rs decoder: {error:?}"))?,
        })
    }
}

impl SoundkitPacketDecoder for PlayerSoundkitOpusDecoder {
    fn decode_i16(&mut self, input: &[u8], output: &mut [i16], fec: bool) -> Result<usize, String> {
        let decoded = self
            .inner
            .decode_i16(input, fec)
            .map_err(|error| format!("{error:?}"))?;
        if decoded.len() > output.len() {
            return Err(format!(
                "decoded Opus packet has {} samples, output buffer has {}",
                decoded.len(),
                output.len()
            ));
        }
        output[..decoded.len()].copy_from_slice(&decoded);
        Ok(decoded.len())
    }

    fn decode_i32(
        &mut self,
        _input: &[u8],
        _output: &mut [i32],
        _fec: bool,
    ) -> Result<usize, String> {
        Err("SoundKit Opus cache decoder only outputs i16 PCM".to_string())
    }

    fn decode_f32(&mut self, input: &[u8], output: &mut [f32], fec: bool) -> Result<usize, String> {
        let mut decoded_i16 = vec![0_i16; output.len()];
        let decoded = self.decode_i16(input, &mut decoded_i16, fec)?;
        for index in 0..decoded {
            output[index] = decoded_i16[index] as f32 / 32768.0;
        }
        Ok(decoded)
    }
}
