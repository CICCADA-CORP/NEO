//! Error types for the spatial audio crate.

use thiserror::Error;

/// Errors that can occur during spatial audio operations.
#[derive(Error, Debug)]
pub enum SpatialError {
    /// Azimuth value is outside the valid range.
    #[error("invalid azimuth {0}: must be -180.0..=180.0")]
    InvalidAzimuth(f64),

    /// Elevation value is outside the valid range.
    #[error("invalid elevation {0}: must be -90.0..=90.0")]
    InvalidElevation(f64),

    /// Distance value is outside the valid range.
    #[error("invalid distance {0}: must be 0.0..=1.0")]
    InvalidDistance(f64),

    /// An audio object has no keyframes to interpolate.
    #[error("no keyframes for audio object (stem {0})")]
    NoKeyframes(u8),

    /// Keyframes are not in chronological order.
    #[error("keyframes not in chronological order at index {0}")]
    KeyframeOrder(usize),

    /// Gain value is outside the valid range.
    #[error("invalid gain {0}: must be 0.0..=1.0")]
    InvalidGain(f64),

    /// Spread value is outside the valid range.
    #[error("invalid spread {0}: must be 0.0..=1.0")]
    InvalidSpread(f64),

    /// The requested Ambisonics order is not supported.
    #[error("unsupported ambisonics order: {0}")]
    UnsupportedOrder(u8),

    /// The number of channels does not match the expected count.
    #[error("channel count mismatch: expected {expected}, got {got}")]
    ChannelMismatch {
        /// The expected number of channels.
        expected: usize,
        /// The actual number of channels.
        got: usize,
    },

    /// Serialization/deserialization error.
    #[error(transparent)]
    SerdeJson(#[from] serde_json::Error),
}

/// Convenience Result type for spatial audio operations.
pub type Result<T> = std::result::Result<T, SpatialError>;
