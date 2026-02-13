//! # FLAC — Lossless Audio Compression
//!
//! This module provides the FLAC codec for bit-perfect lossless audio storage.
//! In the NEO format, FLAC serves as the **residual layer codec**: after DAC
//! lossy encoding, the difference between the original PCM and the DAC
//! reconstruction is FLAC-compressed, enabling bit-perfect restoration when
//! both layers are combined.
//!
//! ## Role in the NEO format
//!
//! FLAC is the **lossless codec** (see SPECIFICATION.md §8.2). It is used for:
//! - Residual layer data (difference between original and DAC-decoded audio)
//! - Standalone lossless stems when maximum quality is required
//! - Archival-grade storage where no lossy compression is acceptable
//!
//! ## Current status
//!
//! This implementation is a **stub awaiting external dependencies**:
//!
//! - **Encoding**: Requires a Rust FLAC encoder such as `flac-bound` (C
//!   libFLAC wrapper) or a pure-Rust implementation like `claxon` (read-only)
//!   paired with a write-capable library.
//! - **Decoding**: Will use `symphonia` or `claxon` for FLAC frame decoding.
//!
//! Once a suitable encoding library is integrated, the encode/decode paths
//! will perform real FLAC compression and decompression.

use crate::{AudioCodec, CodecError, EncodeConfig};

/// FLAC lossless audio codec for residual-layer and archival storage.
///
/// Provides bit-perfect round-trip encoding of PCM audio using the FLAC
/// (Free Lossless Audio Codec) format. Compression ratios typically range
/// from 50–70 % of the original PCM size.
///
/// # External dependencies
///
/// - **Encoding**: `flac-bound` (C libFLAC bindings) or a pure-Rust encoder.
/// - **Decoding**: `symphonia` or `claxon` for FLAC stream parsing.
///
/// # Specification reference
///
/// See SPECIFICATION.md §8.2 — FLAC Lossless Codec for chunk layout and
/// interaction with the residual reconstruction pipeline.
pub struct FlacCodec;

impl AudioCodec for FlacCodec {
    /// Encode PCM f32 samples to FLAC-compressed bytes.
    ///
    /// When fully implemented, this will:
    /// 1. Convert f32 samples to the appropriate integer bit depth (16/24-bit)
    /// 2. Configure the FLAC encoder (compression level, block size)
    /// 3. Compress the audio frames and return the FLAC byte stream
    ///
    /// # Arguments
    ///
    /// * `_pcm` — Interleaved PCM f32 samples in the range `[-1.0, 1.0]`.
    /// * `_config` — Encoding parameters (sample rate, channels, bitrate unused for lossless).
    ///
    /// # Errors
    ///
    /// Currently always returns [`CodecError::EncodingError`] because no FLAC
    /// encoding library is integrated yet.
    fn encode(&self, _pcm: &[f32], _config: &EncodeConfig) -> Result<Vec<u8>, CodecError> {
        tracing::warn!(
            "FlacCodec::encode() is a stub — FLAC encoding requires integration with \
             a FLAC encoder library (e.g., `flac-bound` or a pure-Rust implementation)."
        );
        Err(CodecError::EncodingError(
            "FLAC encoding is not yet available. Awaiting integration of a Rust FLAC \
             encoder library (e.g., `flac-bound`). Use PCM codec as a lossless alternative."
                .into(),
        ))
    }

    /// Decode FLAC-compressed bytes back to PCM f32 samples.
    ///
    /// When fully implemented, this will:
    /// 1. Parse the FLAC stream header and frame boundaries
    /// 2. Decompress each audio frame to integer samples
    /// 3. Convert integer samples back to f32 in the range `[-1.0, 1.0]`
    ///
    /// # Arguments
    ///
    /// * `_data` — FLAC-encoded byte stream.
    /// * `_config` — Decoding parameters (sample rate, channels).
    ///
    /// # Errors
    ///
    /// Currently always returns [`CodecError::DecodingError`] because no FLAC
    /// decoding library is integrated yet.
    fn decode(&self, _data: &[u8], _config: &EncodeConfig) -> Result<Vec<f32>, CodecError> {
        tracing::warn!(
            "FlacCodec::decode() is a stub — FLAC decoding requires integration with \
             a FLAC decoder library (e.g., `symphonia` or `claxon`)."
        );
        Err(CodecError::DecodingError(
            "FLAC decoding is not yet available. Awaiting integration of a Rust FLAC \
             decoder library (e.g., `symphonia` or `claxon`)."
                .into(),
        ))
    }

    fn codec_id(&self) -> neo_format::CodecId {
        neo_format::CodecId::Flac
    }

    fn name(&self) -> &str {
        "FLAC (Free Lossless Audio Codec)"
    }
}
