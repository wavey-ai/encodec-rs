use std::io::{Read, Write};

use anyhow::{Context, Result, bail};
use crc32fast::Hasher;
use serde::Serialize;
use serde::de::DeserializeOwned;

const ECDC_MAGIC: &[u8; 4] = b"ECDC";
const ECDC_VERSION: u8 = 0;
const CHUNK_HEADER_SIZE: usize = 8;

pub fn read_exactly(reader: &mut impl Read, size: usize) -> Result<Vec<u8>> {
    let mut remaining = size;
    let mut out = Vec::with_capacity(size);
    while remaining > 0 {
        let mut buf = vec![0_u8; remaining];
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
    let metadata_json = serde_json::to_vec(metadata).context("failed to serialize ECDC metadata")?;
    writer
        .write_all(ECDC_MAGIC)
        .context("failed to write ECDC magic")?;
    writer
        .write_all(&[ECDC_VERSION])
        .context("failed to write ECDC version")?;
    writer
        .write_all(&(metadata_json.len() as u32).to_be_bytes())
        .context("failed to write ECDC metadata size")?;
    writer
        .write_all(&metadata_json)
        .context("failed to write ECDC metadata body")?;
    writer.flush().context("failed to flush ECDC header")?;
    Ok(())
}

pub fn read_ecdc_header<T: DeserializeOwned>(reader: &mut impl Read) -> Result<T> {
    let header = read_exactly(reader, 9)?;
    let magic: [u8; 4] = header[0..4].try_into().expect("slice length");
    if &magic != ECDC_MAGIC {
        bail!("file is not in ECDC format");
    }
    let version = header[4];
    if version != ECDC_VERSION {
        bail!("unsupported ECDC version {version}");
    }
    let meta_len = u32::from_be_bytes(header[5..9].try_into().expect("slice length")) as usize;
    let meta_bytes = read_exactly(reader, meta_len)?;
    serde_json::from_slice(&meta_bytes).context("failed to parse ECDC metadata JSON")
}

pub fn write_chunk(writer: &mut impl Write, payload: &[u8]) -> Result<()> {
    let mut hasher = Hasher::new();
    hasher.update(payload);
    let checksum = hasher.finalize();
    writer
        .write_all(&(payload.len() as u32).to_be_bytes())
        .context("failed to write chunk length")?;
    writer
        .write_all(&checksum.to_be_bytes())
        .context("failed to write chunk checksum")?;
    writer
        .write_all(payload)
        .context("failed to write chunk payload")?;
    Ok(())
}

pub fn read_chunk_payload(reader: &mut impl Read) -> Result<Vec<u8>> {
    let header = read_exactly(reader, CHUNK_HEADER_SIZE)?;
    let payload_len = u32::from_be_bytes(header[0..4].try_into().expect("slice length")) as usize;
    let expected_crc = u32::from_be_bytes(header[4..8].try_into().expect("slice length"));
    let payload = read_exactly(reader, payload_len)?;
    let mut hasher = Hasher::new();
    hasher.update(&payload);
    let actual_crc = hasher.finalize();
    if actual_crc != expected_crc {
        bail!(
            "chunk CRC mismatch: expected {expected_crc:#010x}, got {actual_crc:#010x}"
        );
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
        write_chunk(&mut buf, payload).unwrap();
        let decoded = read_chunk_payload(&mut Cursor::new(buf)).unwrap();
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
        let rebuilt: Vec<u16> = std::iter::from_fn(|| unpacker.pull()).take(values.len()).collect();
        assert_eq!(rebuilt, values);
    }
}
