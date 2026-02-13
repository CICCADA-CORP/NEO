//! Error types for the NEO format crate.

use thiserror::Error;

/// Errors that can occur when reading or writing NEO files.
#[derive(Error, Debug)]
pub enum NeoFormatError {
    #[error("Invalid magic bytes: expected NEO! (0x4E454F21)")]
    InvalidMagic,

    #[error("Unsupported format version: {0}")]
    UnsupportedVersion(u16),

    #[error("Chunk checksum mismatch: expected {expected}, got {actual}")]
    ChecksumMismatch { expected: String, actual: String },

    #[error("Invalid chunk type: {0}")]
    InvalidChunkType(u8),

    #[error("Stem not found: {0}")]
    StemNotFound(u8),

    #[error("Maximum stem count exceeded (max {max}, got {got})")]
    TooManyStems { max: u8, got: u8 },

    #[error("Invalid offset: chunk at {offset} exceeds file size {file_size}")]
    InvalidOffset { offset: u64, file_size: u64 },

    #[error("Invalid codec identifier: {0}")]
    InvalidCodecId(u8),

    #[error("Invalid stem identifier: {id} (must be 0-7)")]
    InvalidStemId { id: u8 },

    #[error("Chunk count exceeded maximum (max {max}, got {got})")]
    ChunkCountExceeded { max: u64, got: u64 },

    #[error("Allocation too large: requested {requested} bytes, limit is {limit} bytes")]
    AllocationTooLarge { requested: u64, limit: u64 },

    #[error("Invalid UTF-8 in stem label: {0}")]
    InvalidUtf8(#[from] std::string::FromUtf8Error),

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
}

pub type Result<T> = std::result::Result<T, NeoFormatError>;
