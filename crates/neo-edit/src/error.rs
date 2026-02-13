//! Error types for the editing crate.

use thiserror::Error;

/// All errors that can occur in `neo-edit` operations.
#[derive(Error, Debug)]
pub enum EditError {
    /// The edit history has no commits.
    #[error("empty edit history — no commits")]
    EmptyHistory,

    /// A commit with the given hash was not found.
    #[error("commit not found: {0}")]
    CommitNotFound(String),

    /// An invalid operation was attempted.
    #[error("invalid operation: {0}")]
    InvalidOperation(String),

    /// The requested stem does not exist.
    #[error("stem {0} not found")]
    StemNotFound(u8),

    /// Trim range is invalid (start >= end or negative).
    #[error("trim range invalid: start={start} end={end}")]
    InvalidTrimRange {
        /// Start time in seconds.
        start: f64,
        /// End time in seconds.
        end: f64,
    },

    /// Gain value is outside the allowed ±60 dB range.
    #[error("gain out of range: {0} dB (max ±60 dB)")]
    GainOutOfRange(f64),

    /// Pan value is outside the allowed -1.0..=1.0 range.
    #[error("pan out of range: {0} (must be -1.0..=1.0)")]
    PanOutOfRange(f64),

    /// EQ frequency is outside the audible range.
    #[error("EQ frequency out of range: {0} Hz")]
    EqFreqOutOfRange(f64),

    /// There are no commits to revert.
    #[error("history is empty, nothing to revert")]
    NothingToRevert,

    /// JSON serialization/deserialization error.
    #[error(transparent)]
    SerdeJson(#[from] serde_json::Error),
}

/// A convenience result type for `neo-edit` operations.
pub type Result<T> = std::result::Result<T, EditError>;
