//! Chunk types and the chunk table for NEO files.

use serde::{Deserialize, Serialize};

/// The type of a chunk within a NEO file.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u8)]
pub enum ChunkType {
    /// Audio stem data (compressed or lossless)
    Stem = 0x01,
    /// JSON-LD metadata
    Metadata = 0x02,
    /// C2PA content credentials
    Credentials = 0x03,
    /// Spatial audio positioning data
    Spatial = 0x04,
    /// Non-destructive edit history
    EditHistory = 0x05,
    /// Low-quality preview for progressive loading
    Preview = 0x06,
    /// Lossless residual data
    Residual = 0x07,
    /// Temporal metadata (lyrics, chords, BPM map)
    Temporal = 0x08,
    /// Web3 rights and royalty information
    Rights = 0x09,
}

impl ChunkType {
    pub fn from_u8(value: u8) -> Option<Self> {
        match value {
            0x01 => Some(Self::Stem),
            0x02 => Some(Self::Metadata),
            0x03 => Some(Self::Credentials),
            0x04 => Some(Self::Spatial),
            0x05 => Some(Self::EditHistory),
            0x06 => Some(Self::Preview),
            0x07 => Some(Self::Residual),
            0x08 => Some(Self::Temporal),
            0x09 => Some(Self::Rights),
            _ => None,
        }
    }
}

/// An entry in the chunk table describing a single chunk.
///
/// Layout (57 bytes, little-endian):
/// - `[0]`      chunk_type: u8
/// - `[1..9]`   offset: u64 (byte offset from file start)
/// - `[9..17]`  size: u64 (chunk data size in bytes)
/// - `[17..49]` blake3_hash: [u8; 32]
/// - `[49..57]` reserved: [u8; 8]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChunkEntry {
    /// Type of this chunk
    pub chunk_type: ChunkType,
    /// Byte offset from file start to chunk data
    pub offset: u64,
    /// Size of chunk data in bytes
    pub size: u64,
    /// BLAKE3 hash of the chunk data for integrity verification
    pub blake3_hash: [u8; 32],
}

/// Size of a single chunk table entry in bytes
pub const CHUNK_ENTRY_SIZE: usize = 57;

impl ChunkEntry {
    pub fn new(chunk_type: ChunkType, offset: u64, size: u64, data: &[u8]) -> Self {
        let hash = blake3::hash(data);
        Self {
            chunk_type,
            offset,
            size,
            blake3_hash: *hash.as_bytes(),
        }
    }
}
