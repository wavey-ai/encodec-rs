use std::io::{Read, Write};

use anyhow::{bail, Context, Result};
use crc32fast::Hasher;
use serde::de::DeserializeOwned;
use serde::Serialize;

const ECDC_MAGIC: &[u8; 4] = b"ECDC";
const ECDC_VERSION: u8 = 0;
const CHUNK_CRC_HEADER_SIZE: usize = 8;
const CHUNK_PLAIN_HEADER_SIZE: usize = 4;

pub fn write_tagged_header(
    writer: &mut impl Write,
    magic: &[u8; 4],
    version: u8,
    metadata: &impl Serialize,
) -> Result<()> {
    let metadata_json =
        serde_json::to_vec(metadata).context("failed to serialize header metadata")?;
    writer
        .write_all(magic)
        .context("failed to write file magic")?;
    writer
        .write_all(&[version])
        .context("failed to write file version")?;
    writer
        .write_all(&(metadata_json.len() as u32).to_be_bytes())
        .context("failed to write metadata size")?;
    writer
        .write_all(&metadata_json)
        .context("failed to write metadata body")?;
    writer.flush().context("failed to flush file header")?;
    Ok(())
}

pub fn read_tagged_header<T: DeserializeOwned>(
    reader: &mut impl Read,
    magic: &[u8; 4],
    version: u8,
) -> Result<T> {
    let header = read_exactly(reader, 9)?;
    let actual_magic: [u8; 4] = header[0..4].try_into().expect("slice length");
    if &actual_magic != magic {
        bail!("file has unexpected magic");
    }
    let actual_version = header[4];
    if actual_version != version {
        bail!("unsupported file version {actual_version}");
    }
    let meta_len = u32::from_be_bytes(header[5..9].try_into().expect("slice length")) as usize;
    let meta_bytes = read_exactly(reader, meta_len)?;
    serde_json::from_slice(&meta_bytes).context("failed to parse metadata JSON")
}

pub fn read_exactly(reader: &mut impl Read, size: usize) -> Result<Vec<u8>> {
    let mut remaining = size;
    let mut out = Vec::new();
    const MAX_READ_CHUNK: usize = 64 * 1024;
    while remaining > 0 {
        let mut buf = vec![0_u8; remaining.min(MAX_READ_CHUNK)];
        let count = reader
            .read(&mut buf)
            .with_context(|| format!("failed to read {remaining} bytes from stream"))?;
        if count == 0 {
            bail!("stream ended early with {remaining} bytes remaining");
        }
        out.extend_from_slice(&buf[..count]);
        remaining -= count;
    }
    Ok(out)
}

pub fn write_ecdc_header(writer: &mut impl Write, metadata: &impl Serialize) -> Result<()> {
    write_tagged_header(writer, ECDC_MAGIC, ECDC_VERSION, metadata)
}

pub fn read_ecdc_header<T: DeserializeOwned>(reader: &mut impl Read) -> Result<T> {
    read_tagged_header(reader, ECDC_MAGIC, ECDC_VERSION)
}

pub fn write_chunk(writer: &mut impl Write, payload: &[u8], with_crc: bool) -> Result<()> {
    writer
        .write_all(&(payload.len() as u32).to_be_bytes())
        .context("failed to write chunk length")?;
    if with_crc {
        let mut hasher = Hasher::new();
        hasher.update(payload);
        let checksum = hasher.finalize();
        writer
            .write_all(&checksum.to_be_bytes())
            .context("failed to write chunk checksum")?;
    }
    writer
        .write_all(payload)
        .context("failed to write chunk payload")?;
    Ok(())
}

pub fn read_chunk_payload(reader: &mut impl Read, with_crc: bool) -> Result<Vec<u8>> {
    let header = read_exactly(
        reader,
        if with_crc {
            CHUNK_CRC_HEADER_SIZE
        } else {
            CHUNK_PLAIN_HEADER_SIZE
        },
    )?;
    let payload_len = u32::from_be_bytes(header[0..4].try_into().expect("slice length")) as usize;
    let payload = read_exactly(reader, payload_len)?;
    if with_crc {
        let expected_crc = u32::from_be_bytes(header[4..8].try_into().expect("slice length"));
        let mut hasher = Hasher::new();
        hasher.update(&payload);
        let actual_crc = hasher.finalize();
        if actual_crc != expected_crc {
            bail!("chunk CRC mismatch: expected {expected_crc:#010x}, got {actual_crc:#010x}");
        }
    }
    Ok(payload)
}

#[derive(Debug, Clone)]
pub struct BitPacker {
    current_value: u64,
    current_bits: u8,
    bits: u8,
    bytes: Vec<u8>,
}

impl BitPacker {
    pub fn new(bits: u8) -> Self {
        Self {
            current_value: 0,
            current_bits: 0,
            bits,
            bytes: Vec::new(),
        }
    }

    pub fn push(&mut self, value: u16) {
        self.current_value |= (value as u64) << self.current_bits;
        self.current_bits += self.bits;
        while self.current_bits >= 8 {
            self.bytes.push((self.current_value & 0xff) as u8);
            self.current_value >>= 8;
            self.current_bits -= 8;
        }
    }

    pub fn finish(mut self) -> Vec<u8> {
        if self.current_bits > 0 {
            self.bytes.push(self.current_value as u8);
            self.current_value = 0;
            self.current_bits = 0;
        }
        self.bytes
    }
}

#[derive(Debug, Clone)]
pub struct BitUnpacker {
    bits: u8,
    mask: u64,
    current_value: u64,
    current_bits: u8,
    bytes: Vec<u8>,
    offset: usize,
}

impl BitUnpacker {
    pub fn new(bits: u8, bytes: Vec<u8>) -> Self {
        Self {
            bits,
            mask: (1_u64 << bits) - 1,
            current_value: 0,
            current_bits: 0,
            bytes,
            offset: 0,
        }
    }

    pub fn pull(&mut self) -> Option<u16> {
        while self.current_bits < self.bits {
            let next = *self.bytes.get(self.offset)?;
            self.offset += 1;
            self.current_value |= (next as u64) << self.current_bits;
            self.current_bits += 8;
        }
        let out = (self.current_value & self.mask) as u16;
        self.current_value >>= self.bits;
        self.current_bits -= self.bits;
        Some(out)
    }
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use super::*;

    #[test]
    fn header_roundtrip() {
        #[derive(Debug, serde::Serialize, serde::Deserialize, PartialEq)]
        struct Metadata {
            m: String,
            al: usize,
            lm: bool,
        }

        let metadata = Metadata {
            m: "encodec_48khz".into(),
            al: 123,
            lm: false,
        };
        let mut buf = Vec::new();
        write_ecdc_header(&mut buf, &metadata).unwrap();
        let decoded: Metadata = read_ecdc_header(&mut Cursor::new(buf)).unwrap();
        assert_eq!(decoded, metadata);
    }

    #[test]
    fn chunk_roundtrip() {
        let payload = b"hello chunk";
        let mut buf = Vec::new();
        write_chunk(&mut buf, payload, true).unwrap();
        let decoded = read_chunk_payload(&mut Cursor::new(buf), true).unwrap();
        assert_eq!(decoded, payload);
    }

    #[test]
    fn chunk_roundtrip_without_crc() {
        let payload = b"hello chunk";
        let mut buf = Vec::new();
        write_chunk(&mut buf, payload, false).unwrap();
        let decoded = read_chunk_payload(&mut Cursor::new(buf), false).unwrap();
        assert_eq!(decoded, payload);
    }

    #[test]
    fn bitpack_roundtrip_matches_python_layout() {
        let values = [0_u16, 1, 5, 1023, 17, 999, 2, 511, 255];
        let mut packer = BitPacker::new(10);
        for value in values {
            packer.push(value);
        }
        let packed = packer.finish();
        let mut unpacker = BitUnpacker::new(10, packed);
        let rebuilt: Vec<u16> = std::iter::from_fn(|| unpacker.pull())
            .take(values.len())
            .collect();
        assert_eq!(rebuilt, values);
    }
}
