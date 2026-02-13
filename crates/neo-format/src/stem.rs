//! Stem definitions — individual audio tracks within a NEO file.

use std::str::FromStr;

use serde::{Deserialize, Serialize};

/// Identifies the codec used to compress a stem.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u8)]
pub enum CodecId {
    /// Descript Audio Codec (neural, lossy)
    Dac = 0x01,
    /// FLAC (lossless)
    Flac = 0x02,
    /// Opus (traditional lossy, lightweight)
    Opus = 0x03,
    /// Raw PCM (uncompressed)
    Pcm = 0x04,
}

impl CodecId {
    pub fn from_u8(value: u8) -> Option<Self> {
        match value {
            0x01 => Some(Self::Dac),
            0x02 => Some(Self::Flac),
            0x03 => Some(Self::Opus),
            0x04 => Some(Self::Pcm),
            _ => None,
        }
    }
}

/// Well-known stem labels.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum StemLabel {
    Vocals,
    Drums,
    Bass,
    Melody,
    Instrumental,
    /// Full stereo mix (for backward compatibility)
    Mix,
    /// User-defined label
    Custom(String),
}

impl StemLabel {
    pub fn as_str(&self) -> &str {
        match self {
            StemLabel::Vocals => "vocals",
            StemLabel::Drums => "drums",
            StemLabel::Bass => "bass",
            StemLabel::Melody => "melody",
            StemLabel::Instrumental => "instrumental",
            StemLabel::Mix => "mix",
            StemLabel::Custom(s) => s.as_str(),
        }
    }
}

impl FromStr for StemLabel {
    type Err = std::convert::Infallible;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        Ok(match s.to_lowercase().as_str() {
            "vocals" | "voice" | "vox" => StemLabel::Vocals,
            "drums" | "percussion" => StemLabel::Drums,
            "bass" => StemLabel::Bass,
            "melody" | "melodies" | "other" => StemLabel::Melody,
            "instrumental" | "inst" => StemLabel::Instrumental,
            "mix" | "master" | "stereo" => StemLabel::Mix,
            _ => StemLabel::Custom(s.to_string()),
        })
    }
}

/// Configuration for a single stem within the NEO file.
///
/// Stored as part of the STEM chunk header.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StemConfig {
    /// Unique stem identifier (0-7)
    pub stem_id: u8,
    /// Human-readable label for this stem
    pub label: StemLabel,
    /// Codec used for this stem's audio data
    pub codec: CodecId,
    /// Number of audio channels (1=mono, 2=stereo)
    pub channels: u8,
    /// Sample rate in Hz (may differ from container if resampled)
    pub sample_rate: u32,
    /// Bit depth for PCM/FLAC (16, 24, 32)
    pub bit_depth: u8,
    /// Bitrate hint in kbps for lossy codecs (0 = VBR/unknown)
    pub bitrate_kbps: u32,
    /// Total number of audio samples in this stem
    pub sample_count: u64,
}

impl StemConfig {
    /// Create a new stem configuration with default bit depth and bitrate.
    pub fn new(
        stem_id: u8,
        label: StemLabel,
        codec: CodecId,
        channels: u8,
        sample_rate: u32,
    ) -> Self {
        Self {
            stem_id,
            label,
            codec,
            channels,
            sample_rate,
            bit_depth: 16,
            bitrate_kbps: 0,
            sample_count: 0,
        }
    }
}
