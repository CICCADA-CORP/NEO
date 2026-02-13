//! # DAC (Descript Audio Codec) — Neural Audio Compression
//!
//! This module provides the DAC neural codec, which uses ONNX Runtime to run
//! a learned encoder/decoder pair for high-quality lossy audio compression.
//! DAC achieves transparent quality at ~8 kbps per stem by leveraging residual
//! vector quantization (RVQ), making it the primary lossy codec in the NEO format.
//!
//! ## Role in the NEO format
//!
//! DAC is the **primary lossy codec** (see SPECIFICATION.md §8.1). Each audio
//! stem is independently encoded through DAC's encoder, producing a compact
//! sequence of quantized codes. A paired FLAC residual layer can optionally
//! restore bit-perfect quality (see [`crate::residual`]).
//!
//! ## Current status
//!
//! This implementation is a **stub awaiting external dependencies**:
//!
//! - **`ort` (ONNX Runtime)**: The Rust ONNX Runtime bindings must be configured
//!   with a valid ONNX Runtime shared library at build time.
//! - **Pre-trained ONNX models**: The DAC encoder (`encoder.onnx`) and decoder
//!   (`decoder.onnx`) must be exported from the Python DAC fork located at
//!   `neural/neo_neural/dac_fork/` using `neural/neo_neural/export_onnx.py`.
//!
//! Once models are available, the encode/decode paths will perform real
//! neural inference via `ort::Session`.

use crate::{AudioCodec, CodecError, EncodeConfig};

/// DAC neural codec using ONNX Runtime for inference.
///
/// Wraps a pair of ONNX sessions (encoder + decoder) loaded from a model
/// directory. The directory must contain `encoder.onnx` and `decoder.onnx`
/// files exported from the Descript Audio Codec Python package.
///
/// # External dependencies
///
/// - **ONNX Runtime** (`ort` crate): provides the inference backend.
/// - **Model weights**: obtained via `neo model download 44khz` or by running
///   `python neural/neo_neural/export_onnx.py`.
///
/// # Specification reference
///
/// See SPECIFICATION.md §8.1 — DAC Neural Codec for the binary layout of
/// DAC-encoded audio chunks and RVQ code serialization format.
pub struct DacCodec {
    // Awaiting external dependencies — will hold `ort::Session` instances
    // for the encoder and decoder ONNX models.
    _model_dir: std::path::PathBuf,
}

impl DacCodec {
    /// Create a new DAC codec, loading ONNX models from the given directory.
    ///
    /// # Arguments
    ///
    /// * `model_dir` — Path to a directory containing `encoder.onnx` and
    ///   `decoder.onnx`. Use `neo model download 44khz` to obtain pre-trained
    ///   weights, or export them manually with `export_onnx.py`.
    ///
    /// # Errors
    ///
    /// Returns [`CodecError::ModelNotFound`] if `model_dir` does not exist.
    /// Once fully implemented, will also return errors if the ONNX files are
    /// missing or corrupt.
    pub fn new(model_dir: std::path::PathBuf) -> Result<Self, CodecError> {
        if !model_dir.exists() {
            return Err(CodecError::ModelNotFound(model_dir.display().to_string()));
        }
        Ok(Self {
            _model_dir: model_dir,
        })
    }
}

impl AudioCodec for DacCodec {
    /// Encode PCM f32 samples to DAC-compressed bytes.
    ///
    /// When fully implemented, this will:
    /// 1. Reshape PCM to a `[1, channels, samples]` tensor
    /// 2. Run the encoder ONNX session to produce RVQ quantized codes
    /// 3. Serialize the codes to the NEO DAC binary format (see SPECIFICATION.md §8.1)
    ///
    /// # Errors
    ///
    /// Currently always returns [`CodecError::EncodingError`] because the ONNX
    /// inference pipeline is not yet wired up.
    fn encode(&self, _pcm: &[f32], _config: &EncodeConfig) -> Result<Vec<u8>, CodecError> {
        tracing::warn!(
            "DacCodec::encode() is a stub — DAC encoding requires ONNX Runtime \
             integration and pre-trained model files. Use `neo model download 44khz` \
             to obtain models, then pass the model directory to `DacCodec::new()`."
        );
        Err(CodecError::EncodingError(
            "DAC encoding requires ONNX model files. Use `neo model download 44khz` \
             to obtain them, then pass the model directory to `DacCodec::new()`."
                .into(),
        ))
    }

    /// Decode DAC-compressed bytes back to PCM f32 samples.
    ///
    /// When fully implemented, this will:
    /// 1. Deserialize RVQ codes from the NEO DAC binary format
    /// 2. Run the decoder ONNX session to reconstruct the PCM waveform
    /// 3. Reshape the output tensor to flat f32 samples
    ///
    /// # Errors
    ///
    /// Currently always returns [`CodecError::DecodingError`] because the ONNX
    /// inference pipeline is not yet wired up.
    fn decode(&self, _data: &[u8], _config: &EncodeConfig) -> Result<Vec<f32>, CodecError> {
        tracing::warn!(
            "DacCodec::decode() is a stub — DAC decoding requires ONNX Runtime \
             integration and pre-trained model files. Use `neo model download 44khz` \
             to obtain models, then pass the model directory to `DacCodec::new()`."
        );
        Err(CodecError::DecodingError(
            "DAC decoding requires ONNX model files. Use `neo model download 44khz` \
             to obtain them, then pass the model directory to `DacCodec::new()`."
                .into(),
        ))
    }

    fn codec_id(&self) -> neo_format::CodecId {
        neo_format::CodecId::Dac
    }

    fn name(&self) -> &str {
        "DAC (Descript Audio Codec)"
    }
}
