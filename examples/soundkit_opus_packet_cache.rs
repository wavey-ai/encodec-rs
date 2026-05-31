use std::fs;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use frame_header::{EncodingFlag, Endianness, FrameHeader};
use libopus_rs::{Application, Encoder as LibopusEncoder, CELT_MAX_BITRATE, CELT_MIN_BITRATE};
use serde::Serialize;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use soundkit::audio_packet::Encoder as SoundkitPacketEncoder;

const CACHE_MAGIC: &[u8; 4] = b"BNP1";
const CACHE_ENVELOPE_FORMAT: &str = "bitneedle-player-cache-v1";
const OPUS_PACKET_AUDIO_FORMAT: &str = "soundkit_opus_packets";
const DECODED_CACHE_KEY_VERSION: &str = "bitneedle-player-decoded-ecdc-opus-v1";

#[derive(Debug, Default)]
struct Options {
    input_f32le: PathBuf,
    output_cache: PathBuf,
    output_json: Option<PathBuf>,
    key: Option<String>,
    source_hash: String,
    bundle_name: String,
    record_profile: String,
    chunk_index: usize,
    chunk_byte_length: usize,
    sample_rate: u32,
    channels: usize,
    frames: usize,
    bitrate: u32,
    frame_duration_ms: u32,
    build_id: String,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let options = parse_options()?;
    if options.sample_rate != 48_000 {
        return Err(format!(
            "SoundKit Opus cache expects 48000 Hz PCM, got {}",
            options.sample_rate
        )
        .into());
    }
    if options.channels == 0 || options.channels > 16 {
        return Err(format!("invalid channel count {}", options.channels).into());
    }
    let bitrate = i32::try_from(options.bitrate)
        .map_err(|_| format!("invalid Opus bitrate {}", options.bitrate))?;
    if !(CELT_MIN_BITRATE..=CELT_MAX_BITRATE).contains(&bitrate) {
        return Err(format!("invalid Opus bitrate {}", options.bitrate).into());
    }
    if options.frames == 0 {
        return Err("decoded audio has no frames".into());
    }

    let frame_size =
        ((options.sample_rate as u64 * options.frame_duration_ms as u64) / 1000).max(1) as usize;
    if frame_size > 4095 {
        return Err(format!("SoundKit frame size {frame_size} exceeds header limit").into());
    }

    let pcm_bytes = fs::read(&options.input_f32le)?;
    let required_bytes = options
        .frames
        .checked_mul(options.channels)
        .and_then(|samples| samples.checked_mul(std::mem::size_of::<f32>()))
        .ok_or("decoded PCM size overflow")?;
    if pcm_bytes.len() < required_bytes {
        return Err(format!(
            "decoded PCM is too short: got {} bytes, need {required_bytes}",
            pcm_bytes.len()
        )
        .into());
    }

    let packets = encode_soundkit_opus_packets(&pcm_bytes, &options, frame_size)?;
    let packet_byte_length: usize = packets.iter().map(Vec::len).sum();
    let spec_json = decoded_ecdc_opus_cache_spec_json(&options, frame_size)?;
    let key = options
        .key
        .clone()
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| sha256_hex(spec_json.as_bytes()));
    let payload = decoded_ecdc_opus_cache_payload(
        &options,
        &key,
        frame_size,
        packet_byte_length,
        packets.len(),
    );
    let body = serialize_remote_cache_payload(payload.clone(), &packets)?;

    if let Some(parent) = options.output_cache.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(&options.output_cache, body)?;

    if let Some(output_json) = &options.output_json {
        if let Some(parent) = output_json.parent() {
            fs::create_dir_all(parent)?;
        }
        let summary = json!({
            "key": key,
            "format": OPUS_PACKET_AUDIO_FORMAT,
            "cacheEnvelopeFormat": CACHE_ENVELOPE_FORMAT,
            "cacheBody": options.output_cache.display().to_string(),
            "sampleRate": options.sample_rate,
            "channels": options.channels,
            "bitsPerSample": 16,
            "frameSize": frame_size,
            "frameDurationMs": options.frame_duration_ms,
            "bitrate": options.bitrate,
            "cbr": true,
            "audioLength": options.frames,
            "duration": options.frames as f64 / options.sample_rate as f64,
            "packetCount": packets.len(),
            "packetByteLength": packet_byte_length,
            "payload": payload,
        });
        fs::write(output_json, serde_json::to_vec_pretty(&summary)?)?;
    }

    println!(
        "{}",
        serde_json::to_string(&json!({
            "key": key,
            "output": options.output_cache.display().to_string(),
            "format": OPUS_PACKET_AUDIO_FORMAT,
            "packetCount": packets.len(),
            "packetByteLength": packet_byte_length,
            "audioLength": options.frames,
        }))?
    );
    Ok(())
}

fn parse_options() -> Result<Options, Box<dyn std::error::Error>> {
    let mut options = Options {
        sample_rate: 48_000,
        channels: 2,
        bitrate: 64_000,
        frame_duration_ms: 20,
        build_id: "dev".to_string(),
        record_profile: "single45".to_string(),
        ..Options::default()
    };
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
            "--input-f32le" => options.input_f32le = PathBuf::from(value(&mut index)?),
            "--output-cache" => options.output_cache = PathBuf::from(value(&mut index)?),
            "--output-json" => options.output_json = Some(PathBuf::from(value(&mut index)?)),
            "--key" => options.key = Some(value(&mut index)?),
            "--source-hash" => options.source_hash = value(&mut index)?,
            "--bundle-name" => options.bundle_name = value(&mut index)?,
            "--record-profile" => options.record_profile = value(&mut index)?,
            "--chunk-index" => options.chunk_index = value(&mut index)?.parse()?,
            "--chunk-byte-length" => options.chunk_byte_length = value(&mut index)?.parse()?,
            "--sample-rate" => options.sample_rate = value(&mut index)?.parse()?,
            "--channels" => options.channels = value(&mut index)?.parse()?,
            "--frames" => options.frames = value(&mut index)?.parse()?,
            "--bitrate" => options.bitrate = value(&mut index)?.parse()?,
            "--frame-duration-ms" => options.frame_duration_ms = value(&mut index)?.parse()?,
            "--build-id" => options.build_id = value(&mut index)?,
            _ => return Err(format!("unknown argument: {key}").into()),
        }
        index += 1;
    }

    if options.input_f32le.as_os_str().is_empty() {
        return Err("--input-f32le is required".into());
    }
    if options.output_cache.as_os_str().is_empty() {
        return Err("--output-cache is required".into());
    }
    if options.source_hash.is_empty() {
        return Err("--source-hash is required".into());
    }
    if options.bundle_name.is_empty() {
        return Err("--bundle-name is required".into());
    }
    Ok(options)
}

fn encode_soundkit_opus_packets(
    pcm_bytes: &[u8],
    options: &Options,
    frame_size: usize,
) -> Result<Vec<Vec<u8>>, Box<dyn std::error::Error>> {
    let mut encoder = <PlayerSoundkitOpusEncoder as SoundkitPacketEncoder>::new(
        options.sample_rate,
        16,
        options.channels as u32,
        frame_size as u32,
        options.bitrate,
    );
    encoder.init()?;

    let packet_count = options.frames.div_ceil(frame_size);
    let mut frame = vec![0_i16; frame_size * options.channels];
    let mut encoded = vec![0_u8; 1500];
    let mut packets = Vec::with_capacity(packet_count);
    for packet_index in 0..packet_count {
        let source_frame = packet_index * frame_size;
        let copy_frames = frame_size.min(options.frames - source_frame);
        frame.fill(0);
        for sample_index in 0..copy_frames {
            for channel in 0..options.channels {
                let sample_offset =
                    ((source_frame + sample_index) * options.channels + channel) * 4;
                let sample = f32::from_le_bytes(
                    pcm_bytes[sample_offset..sample_offset + 4]
                        .try_into()
                        .expect("slice length"),
                );
                frame[sample_index * options.channels + channel] = float_to_i16(sample);
            }
        }
        let encoded_len = encoder
            .encode_i16(&frame[..], &mut encoded)
            .map_err(|error| {
                format!(
                    "libopus-rs encode failed at packet {}/{}: {error}",
                    packet_index + 1,
                    packet_count
                )
            })?;
        if encoded_len == 0 {
            return Err(format!(
                "Opus encode produced an empty packet at {}",
                packet_index + 1
            )
            .into());
        }

        let header = FrameHeader::new(
            EncodingFlag::Opus,
            frame_size as u16,
            options.sample_rate,
            options.channels as u8,
            16,
            Endianness::LittleEndian,
            None,
            Some(source_frame as u64),
        )
        .map_err(|error| format!("failed to build SoundKit Opus frame header: {error}"))?;
        let mut packet = Vec::with_capacity(header.size() + encoded_len);
        header.encode(&mut packet)?;
        packet.extend_from_slice(&encoded[..encoded_len]);
        packets.push(packet);
    }
    Ok(packets)
}

struct PlayerSoundkitOpusEncoder {
    inner: LibopusEncoder,
    frame_size: usize,
    bitrate: i32,
}

impl SoundkitPacketEncoder for PlayerSoundkitOpusEncoder {
    fn new(
        sample_rate: u32,
        bits_per_sample: u32,
        channels: u32,
        frame_size: u32,
        bitrate: u32,
    ) -> Self {
        assert_eq!(sample_rate, 48_000);
        assert_eq!(bits_per_sample, 16);
        let mut inner = LibopusEncoder::new(
            sample_rate as i32,
            channels as usize,
            Application::RestrictedLowDelay,
        )
        .expect("libopus-rs encoder");
        let bitrate = bitrate as i32;
        inner.set_bitrate(bitrate).expect("libopus-rs bitrate");
        inner.set_vbr(false).expect("libopus-rs CBR");
        Self {
            inner,
            frame_size: frame_size as usize,
            bitrate,
        }
    }

    fn init(&mut self) -> Result<(), String> {
        self.reset()
    }

    fn encode_i16(&mut self, input: &[i16], output: &mut [u8]) -> Result<usize, String> {
        let encoded = self
            .inner
            .encode_i16(input, self.frame_size)
            .map_err(|error| format!("{error:?}"))?;
        if encoded.len() > output.len() {
            return Err(format!(
                "encoded Opus packet has {} bytes, output buffer has {}",
                encoded.len(),
                output.len()
            ));
        }
        output[..encoded.len()].copy_from_slice(&encoded);
        Ok(encoded.len())
    }

    fn encode_i32(&mut self, _input: &[i32], _output: &mut [u8]) -> Result<usize, String> {
        Err("SoundKit Opus cache encoder only accepts i16 PCM".to_string())
    }

    fn reset(&mut self) -> Result<(), String> {
        self.inner
            .set_bitrate(self.bitrate)
            .map_err(|error| format!("{error:?}"))?;
        self.inner
            .set_vbr(false)
            .map_err(|error| format!("{error:?}"))?;
        Ok(())
    }
}

#[derive(Serialize)]
struct DecodedEcdcOpusCacheSpec<'a> {
    #[serde(rename = "keyVersion")]
    key_version: &'static str,
    #[serde(rename = "sourceHash")]
    source_hash: &'a str,
    #[serde(rename = "bundleName")]
    bundle_name: &'a str,
    #[serde(rename = "recordProfile")]
    record_profile: &'a str,
    #[serde(rename = "audioFormat")]
    audio_format: &'static str,
    codec: &'static str,
    #[serde(rename = "codecMode")]
    codec_mode: &'static str,
    #[serde(rename = "sampleRate")]
    sample_rate: u32,
    channels: usize,
    bitrate: u32,
    #[serde(rename = "frameSize")]
    frame_size: usize,
    #[serde(rename = "frameDurationMs")]
    frame_duration_ms: u32,
}

fn decoded_ecdc_opus_cache_spec_json(
    options: &Options,
    frame_size: usize,
) -> Result<String, serde_json::Error> {
    serde_json::to_string(&DecodedEcdcOpusCacheSpec {
        key_version: DECODED_CACHE_KEY_VERSION,
        source_hash: &options.source_hash,
        bundle_name: &options.bundle_name,
        record_profile: &options.record_profile,
        audio_format: OPUS_PACKET_AUDIO_FORMAT,
        codec: "libopus-rs",
        codec_mode: "cbr",
        sample_rate: options.sample_rate,
        channels: options.channels,
        bitrate: options.bitrate,
        frame_size,
        frame_duration_ms: options.frame_duration_ms,
    })
}

fn decoded_ecdc_opus_cache_payload(
    options: &Options,
    key: &str,
    frame_size: usize,
    packet_byte_length: usize,
    packet_count: usize,
) -> Value {
    let accessed_at = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_millis() as u64)
        .unwrap_or(0);
    json!({
        "key": key,
        "buildId": options.build_id,
        "accessedAt": accessed_at,
        "sourceHash": options.source_hash,
        "bundleName": options.bundle_name,
        "recordProfile": options.record_profile,
        "chunkIndex": options.chunk_index,
        "chunkByteLength": options.chunk_byte_length,
        "audioFormat": OPUS_PACKET_AUDIO_FORMAT,
        "format": OPUS_PACKET_AUDIO_FORMAT,
        "codec": OPUS_PACKET_AUDIO_FORMAT,
        "codecSource": "libopus-rs",
        "codecMode": "cbr",
        "cbr": true,
        "bitrate": options.bitrate,
        "frameSize": frame_size,
        "frameDurationMs": options.frame_duration_ms,
        "sampleRate": options.sample_rate,
        "channels": options.channels,
        "bitsPerSample": 16,
        "bytesPerFrame": options.channels * 2,
        "startFrame": 0,
        "endFrame": options.frames,
        "audioLength": options.frames,
        "duration": options.frames as f64 / options.sample_rate as f64,
        "packetCount": packet_count,
        "packetByteLength": packet_byte_length,
    })
}

fn serialize_remote_cache_payload(
    mut payload: Value,
    packets: &[Vec<u8>],
) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let placeholders: Vec<Value> = (0..packets.len())
        .map(|index| json!({ "__bitneedleBuffer": index }))
        .collect();
    payload["soundkitPackets"] = Value::Array(placeholders);
    let envelope = json!({
        "format": CACHE_ENVELOPE_FORMAT,
        "payload": payload,
        "buffers": packets.iter().map(|packet| json!({ "byteLength": packet.len() })).collect::<Vec<_>>(),
    });
    let json_bytes = serde_json::to_vec(&envelope)?;
    let byte_length = 8 + json_bytes.len() + packets.iter().map(Vec::len).sum::<usize>();
    let mut output = Vec::with_capacity(byte_length);
    output.extend_from_slice(CACHE_MAGIC);
    output.extend_from_slice(&(json_bytes.len() as u32).to_be_bytes());
    output.extend_from_slice(&json_bytes);
    for packet in packets {
        output.extend_from_slice(packet);
    }
    Ok(output)
}

fn float_to_i16(value: f32) -> i16 {
    let sample = value.clamp(-1.0, 1.0);
    if sample < 0.0 {
        (sample * 32768.0).round() as i16
    } else {
        (sample * 32767.0).round() as i16
    }
}

fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    digest.iter().map(|byte| format!("{byte:02x}")).collect()
}
