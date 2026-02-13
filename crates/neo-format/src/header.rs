//! NEO file header — the first 64 bytes of every `.neo` file.

use serde::{Deserialize, Serialize};

/// Magic bytes identifying a NEO file: `NEO!` (0x4E454F21)
pub const NEO_MAGIC: [u8; 4] = [0x4E, 0x45, 0x4F, 0x21];

/// Current format version
pub const NEO_VERSION: u16 = 1;

/// Maximum number of stems per file
pub const MAX_STEMS: u8 = 8;

/// Size of the fixed header in bytes
pub const HEADER_SIZE: usize = 64;

/// The fixed-size header at the beginning of every `.neo` file.
///
/// Layout (64 bytes, little-endian):
/// - `[0..4]`   magic: `NEO!`
/// - `[4..6]`   version: u16
/// - `[6..14]`  feature_flags: u64
/// - `[14]`     stem_count: u8
/// - `[15..19]` sample_rate: u32
/// - `[19..27]` duration_us: u64 (total duration in microseconds)
/// - `[27..35]` chunk_table_offset: u64
/// - `[35..43]` chunk_count: u64
/// - `[43..64]` reserved: [u8; 21] (zero-filled, future use)
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NeoHeader {
    /// Format version (currently 1)
    pub version: u16,
    /// Feature flags bitfield
    pub feature_flags: FeatureFlags,
    /// Number of audio stems in this file
    pub stem_count: u8,
    /// Sample rate in Hz (e.g., 44100, 48000, 96000)
    pub sample_rate: u32,
    /// Total duration in microseconds
    pub duration_us: u64,
    /// Byte offset of the chunk table from file start
    pub chunk_table_offset: u64,
    /// Number of chunks in the chunk table
    pub chunk_count: u64,
}

/// Feature flags stored as a u64 bitfield.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct FeatureFlags(pub u64);

impl FeatureFlags {
    /// File contains a lossless residual layer
    pub const LOSSLESS: u64 = 1 << 0;
    /// File contains spatial/3D audio metadata
    pub const SPATIAL: u64 = 1 << 1;
    /// File contains C2PA content credentials
    pub const C2PA: u64 = 1 << 2;
    /// File contains non-destructive edit history
    pub const EDIT_HISTORY: u64 = 1 << 3;
    /// File contains a low-quality preview layer (LOD)
    pub const PREVIEW_LOD: u64 = 1 << 4;
    /// File uses neural codec (DAC) for stems
    pub const NEURAL_CODEC: u64 = 1 << 5;
    /// File contains Web3/royalty metadata
    pub const WEB3_RIGHTS: u64 = 1 << 6;
    /// File contains temporal metadata (lyrics, chords, BPM)
    pub const TEMPORAL_META: u64 = 1 << 7;

    pub fn new() -> Self {
        Self(0)
    }

    pub fn set(&mut self, flag: u64) {
        self.0 |= flag;
    }

    pub fn has(&self, flag: u64) -> bool {
        self.0 & flag != 0
    }
}

impl Default for FeatureFlags {
    fn default() -> Self {
        Self::new()
    }
}

impl NeoHeader {
    /// Create a new header with the given sample rate and stem count.
    pub fn new(sample_rate: u32, stem_count: u8) -> Self {
        Self {
            version: NEO_VERSION,
            feature_flags: FeatureFlags::new(),
            stem_count,
            sample_rate,
            duration_us: 0,
            chunk_table_offset: 0,
            chunk_count: 0,
        }
    }
}
