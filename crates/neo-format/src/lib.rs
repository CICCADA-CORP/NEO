//! # neo-format
//!
//! The NEO container format library. Handles reading and writing `.neo` files
//! including headers, chunk tables, stem entries, and metadata blocks.
//!
//! ## Format Overview
//!
//! A `.neo` file consists of:
//! - **Header** (64 bytes): Magic bytes, version, feature flags, stem count
//! - **Chunk Table**: Index of all chunks with types, offsets, sizes, and BLAKE3 hashes
//! - **Chunks**: STEM, META, CRED, SPAT, EDIT, PREV data blocks
//!
//! ## Example
//! ```rust,no_run
//! use std::path::Path;
//! use neo_format::{NeoWriter, NeoReader, StemConfig, CodecId};
//!
//! // Writing
//! let writer = NeoWriter::new(44100, 2);
//! // ... add stems, metadata, finalize
//!
//! // Reading
//! let reader = NeoReader::open(Path::new("track.neo")).unwrap();
//! println!("{:?}", reader.header());
//! ```

pub mod chunk;
pub mod error;
pub mod header;
pub mod reader;
pub mod stem;
pub mod writer;

pub use chunk::*;
pub use error::NeoFormatError;
pub use header::*;
pub use reader::NeoReader;
pub use stem::*;
pub use writer::NeoWriter;
