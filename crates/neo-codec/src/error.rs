//! Error types for the NEO codec crate.

use thiserror::Error;

#[derive(Error, Debug)]
pub enum CodecError {
    #[error("ONNX model not found at path: {0}")]
    ModelNotFound(String),

    #[error("ONNX inference error: {0}")]
    InferenceError(String),

    #[error("Unsupported sample rate: {0}")]
    UnsupportedSampleRate(u32),

    #[error("Encoding error: {0}")]
    EncodingError(String),

    #[error("Decoding error: {0}")]
    DecodingError(String),

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
}
