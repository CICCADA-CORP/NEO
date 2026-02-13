//! # neo-stream — Streaming and distribution for the NEO format
//!
//! Provides content-addressed chunking (Merkle tree with BLAKE3),
//! progressive Level-of-Detail loading, CID generation, and P2P distribution.
//!
//! See SPECIFICATION.md Sections 12 and 14.
//!
//! ## Modules
//!
//! - [`merkle`] — Binary Merkle tree with BLAKE3 hashing for content integrity
//! - [`progressive`] — Level-of-Detail layer management (Preview → Standard → Lossless)
//! - [`cid`] — Content Identifier generation (CIDv1 with BLAKE3)
//! - [`p2p`] — P2P distribution architecture (stub for future iroh integration)
//! - [`error`] — Error types for the streaming crate

pub mod cid;
pub mod error;
pub mod merkle;
pub mod p2p;
pub mod progressive;

pub use cid::ContentId;
pub use error::{Result, StreamError};
pub use merkle::{MerkleNode, MerkleProof, MerkleTree, DEFAULT_CHUNK_SIZE};
pub use p2p::{PeerConfig, PeerNode};
pub use progressive::{LayerInfo, ProgressiveManager, QualityLayer};
