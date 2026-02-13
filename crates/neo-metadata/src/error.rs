//! Error types for the NEO metadata crate.

use thiserror::Error;

/// Errors that can occur during metadata operations.
#[derive(Error, Debug)]
pub enum MetadataError {
    /// Metadata JSON failed to parse.
    #[error("invalid metadata JSON: {0}")]
    InvalidJson(String),

    /// A required field is missing from the metadata.
    #[error("missing required field: {0}")]
    MissingField(String),

    /// A field value is out of the valid range.
    #[error("field out of range: {field} = {value} (expected {expected})")]
    OutOfRange {
        /// The name of the field that is out of range.
        field: String,
        /// The actual value that was provided.
        value: String,
        /// A description of the expected range.
        expected: String,
    },

    /// Temporal events are not in chronological order.
    #[error("temporal events not in order at index {index}: {details}")]
    TemporalOrderError {
        /// The index at which the ordering violation was detected.
        index: usize,
        /// A description of the ordering violation.
        details: String,
    },

    /// Royalty splits do not sum to 1.0.
    #[error("royalty splits sum to {sum:.4} (expected 1.0, tolerance {tolerance})")]
    InvalidSplitSum {
        /// The actual sum of the royalty splits.
        sum: f64,
        /// The tolerance used for the comparison.
        tolerance: f64,
    },

    /// A royalty share is outside the valid range [0.0, 1.0].
    #[error("royalty share out of range: {0} (expected 0.0..=1.0)")]
    InvalidShare(f64),

    /// An invalid blockchain chain identifier.
    #[error("unsupported chain: {0}")]
    UnsupportedChain(String),

    /// C2PA credential error.
    #[error("credential error: {0}")]
    CredentialError(String),

    /// Serialization/deserialization error.
    #[error(transparent)]
    SerdeJson(#[from] serde_json::Error),
}

/// Convenience Result type for metadata operations.
pub type Result<T> = std::result::Result<T, MetadataError>;
