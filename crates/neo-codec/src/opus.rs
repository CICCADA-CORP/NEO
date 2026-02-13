//! # Opus — Lightweight Lossy Compression
//!
//! This module provides the Opus codec for bandwidth-efficient lossy audio.
//! In the NEO format, Opus serves as the **lightweight fallback codec** for
//! low-bandwidth streaming, preview layers, and environments where neural
//! codec inference (DAC) is not feasible.
//!
//! ## Role in the NEO format
//!
//! Opus is the **lightweight lossy codec** (see SPECIFICATION.md §8.3). It is
//! used for:
//! - Preview / low-bandwidth streaming layers
//! - Fallback when ONNX Runtime is unavailable for DAC decoding
//! - Progressive Level-of-Detail (LOD) streaming where a quick decode is needed
//!
//! ## Current status
//!
//! This implementation is a **stub awaiting external dependencies**:
//!
//! - **`audiopus`** or **`opus`** crate: Rust bindings to libopus are needed
//!   for both encoding and decoding. These require the C libopus library to
//!   be installed on the system or vendored via `opus-sys`.
//!
//! Once libopus bindings are integrated, the encode/decode paths will perform
//! real Opus compression and decompression.

use crate::{AudioCodec, CodecError, EncodeConfig};

/// Opus lossy audio codec for lightweight streaming and fallback layers.
///
/// Provides efficient lossy compression using the Opus interactive audio
/// codec (RFC 6716). Typical bitrates range from 32–128 kbps depending on
/// quality requirements and channel count.
///
/// # External dependencies
///
/// - **`audiopus`** or **`opus`** crate: Rust bindings to the C libopus library.
/// - **System dependency**: `libopus-dev` (Debian/Ubuntu) or `opus` (Homebrew/macOS).
///
/// # Specification reference
///
/// See SPECIFICATION.md §8.3 — Opus Lightweight Codec for the chunk layout
/// and bitrate negotiation during progressive streaming.
pub struct OpusCodec;

impl AudioCodec for OpusCodec {
    /// Encode PCM f32 samples to Opus-compressed bytes.
    ///
    /// When fully implemented, this will:
    /// 1. Resample to 48 kHz if needed (Opus native rate)
    /// 2. Frame the input into 20 ms Opus frames
    /// 3. Encode each frame at the configured bitrate
    /// 4. Return the concatenated Opus packet stream
    ///
    /// # Arguments
    ///
    /// * `_pcm` — Interleaved PCM f32 samples in the range `[-1.0, 1.0]`.
    /// * `_config` — Encoding parameters; `bitrate_kbps` controls Opus target bitrate.
    ///
    /// # Errors
    ///
    /// Currently always returns [`CodecError::EncodingError`] because no Opus
    /// encoding library is integrated yet.
    fn encode(&self, _pcm: &[f32], _config: &EncodeConfig) -> Result<Vec<u8>, CodecError> {
        tracing::warn!(
            "OpusCodec::encode() is a stub — Opus encoding requires integration with \
             libopus via the `audiopus` or `opus` crate."
        );
        Err(CodecError::EncodingError(
            "Opus encoding is not yet available. Awaiting integration of libopus \
             bindings (e.g., `audiopus` crate). Use PCM or DAC codec instead."
                .into(),
        ))
    }

    /// Decode Opus-compressed bytes back to PCM f32 samples.
    ///
    /// When fully implemented, this will:
    /// 1. Parse Opus packet boundaries from the byte stream
    /// 2. Decode each 20 ms frame back to PCM samples
    /// 3. Resample from 48 kHz to the target sample rate if needed
    ///
    /// # Arguments
    ///
    /// * `_data` — Opus-encoded byte stream.
    /// * `_config` — Decoding parameters (sample rate, channels).
    ///
    /// # Errors
    ///
    /// Currently always returns [`CodecError::DecodingError`] because no Opus
    /// decoding library is integrated yet.
    fn decode(&self, _data: &[u8], _config: &EncodeConfig) -> Result<Vec<f32>, CodecError> {
        tracing::warn!(
            "OpusCodec::decode() is a stub — Opus decoding requires integration with \
             libopus via the `audiopus` or `opus` crate."
        );
        Err(CodecError::DecodingError(
            "Opus decoding is not yet available. Awaiting integration of libopus \
             bindings (e.g., `audiopus` crate)."
                .into(),
        ))
    }

    fn codec_id(&self) -> neo_format::CodecId {
        neo_format::CodecId::Opus
    }

    fn name(&self) -> &str {
        "Opus"
    }
}
