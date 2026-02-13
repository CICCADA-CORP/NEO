//! # neo-codec
//!
//! Audio codec abstraction for the NEO format.
//! Provides a unified trait for encoding/decoding audio via:
//! - **DAC** (Descript Audio Codec) — neural lossy compression via ONNX
//! - **FLAC** — lossless compression for residual layer
//! - **Opus** — lightweight lossy fallback
//! - **PCM** — raw uncompressed lossless baseline

pub mod dac;
pub mod enhance;
pub mod error;
pub mod flac;
pub mod opus;
pub mod pcm;
pub mod residual;

pub use error::CodecError;
pub use pcm::PcmCodec;

pub use enhance::{
    default_enhancement_pipeline, AudioEnhancer, BandwidthExtender, EnhanceConfig, EnhancementType,
    SourceSeparator, StereoWidener,
};

/// Configuration for audio encoding.
#[derive(Debug, Clone)]
pub struct EncodeConfig {
    /// Sample rate in Hz (e.g., 44100, 48000).
    pub sample_rate: u32,
    /// Number of audio channels (1 = mono, 2 = stereo).
    pub channels: u8,
    /// Target bitrate in kbps for lossy codecs. `None` for lossless or VBR.
    pub bitrate_kbps: Option<u32>,
}

/// Unified codec trait for all supported audio codecs.
pub trait AudioCodec: Send + Sync {
    /// Encode PCM f32 samples to compressed bytes.
    fn encode(&self, pcm: &[f32], config: &EncodeConfig) -> Result<Vec<u8>, CodecError>;

    /// Decode compressed bytes back to PCM f32 samples.
    fn decode(&self, data: &[u8], config: &EncodeConfig) -> Result<Vec<f32>, CodecError>;

    /// Return the codec identifier.
    fn codec_id(&self) -> neo_format::CodecId;

    /// Human-readable codec name.
    fn name(&self) -> &str;
}

/// Create a codec instance from a [`neo_format::CodecId`].
///
/// Returns a boxed [`AudioCodec`] trait object for the requested codec.
/// Currently only PCM is fully implemented; other codecs return descriptive errors.
pub fn create_codec(codec_id: neo_format::CodecId) -> Result<Box<dyn AudioCodec>, CodecError> {
    match codec_id {
        neo_format::CodecId::Pcm => Ok(Box::new(pcm::PcmCodec)),
        neo_format::CodecId::Dac => Err(CodecError::EncodingError(
            "DAC codec requires ONNX models — use `neo model download` first".into(),
        )),
        neo_format::CodecId::Flac => Err(CodecError::EncodingError(
            "FLAC codec awaiting integration of a Rust FLAC encoder library. \
             Use PCM codec as a lossless alternative."
                .into(),
        )),
        neo_format::CodecId::Opus => Err(CodecError::EncodingError(
            "Opus codec awaiting integration of libopus bindings (e.g., `audiopus` crate). \
             Use PCM or DAC codec instead."
                .into(),
        )),
    }
}
