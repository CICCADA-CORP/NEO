//! Error types for the streaming crate.

use thiserror::Error;

/// Errors that can occur during streaming operations.
#[derive(Error, Debug)]
pub enum StreamError {
    /// Attempted to build a Merkle tree from empty data.
    #[error("data is empty, cannot build Merkle tree")]
    EmptyData,

    /// Invalid chunk size (must be greater than zero).
    #[error("invalid chunk size: {0} (must be > 0)")]
    InvalidChunkSize(usize),

    /// Merkle proof verification failed for a specific block.
    #[error("Merkle proof verification failed for block {0}")]
    ProofVerificationFailed(usize),

    /// Block index is out of range for the given tree.
    #[error("block index {index} out of range (total: {total})")]
    BlockOutOfRange {
        /// The requested block index.
        index: usize,
        /// The total number of blocks in the tree.
        total: usize,
    },

    /// Error during CID encoding or decoding.
    #[error("CID encoding error: {0}")]
    CidError(String),

    /// Requested quality layer is not available.
    #[error("layer {0:?} not available")]
    LayerNotAvailable(crate::progressive::QualityLayer),

    /// Underlying I/O error.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
}

/// A convenience type alias for `std::result::Result<T, StreamError>`.
pub type Result<T> = std::result::Result<T, StreamError>;
