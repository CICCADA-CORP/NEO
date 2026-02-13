//! # neo-metadata — Smart metadata for the NEO audio format
//!
//! This crate provides structured metadata types for NEO files, including:
//!
//! - **JSON-LD schema** ([`NeoMetadata`]): Track info following `schema.org/MusicRecording`
//!   with NEO extensions for stems and encoding parameters
//! - **Temporal metadata** ([`TemporalMap`]): Time-stamped chords, lyrics, and BPM changes
//! - **Web3 rights** ([`RightsInfo`]): On-chain royalty splits and smart contract references
//! - **C2PA credentials** ([`NeoCredentials`]): Content provenance and origin tracking
//!
//! All types support builder patterns, JSON serialization, and validation.
//!
//! ## Phase 3 — Metadata (Complete)
//!
//! This crate implements Phase 3 of the NEO development roadmap. It covers:
//!
//! - **Sections 9.1–9.2** of the specification: JSON-LD metadata with `schema.org`
//!   vocabulary and NEO namespace extensions (`neo:stems`, `neo:encoding`).
//! - **Section 9.3**: Temporal metadata maps with chronological validation for
//!   chord progressions, synchronized lyrics, and tempo changes.
//! - **Section 9.4**: Web3 rights management with royalty split definitions,
//!   smart contract references, and supported chain validation.
//! - **Section 10**: C2PA content credentials with content origin classification
//!   (human / AI / hybrid), raw JUMBF manifest storage, and provenance tracking.
//!
//! ## Quick Start
//!
//! ```rust
//! use neo_metadata::{NeoMetadata, TemporalMap, RightsInfo, RoyaltySplit, NeoCredentials, ContentOrigin};
//!
//! // JSON-LD metadata
//! let meta = NeoMetadata::new("My Song")
//!     .with_artist("Artist Name")
//!     .with_genre("Electronic")
//!     .with_isrc("USRC12345678");
//! meta.validate().unwrap();
//!
//! // Temporal metadata
//! let temporal = TemporalMap::new(120.0, "C major")
//!     .with_chord(0.0, "C")
//!     .with_lyric(0.0, 2.0, "Hello world");
//!
//! // Web3 rights
//! let rights = RightsInfo::new()
//!     .with_split(RoyaltySplit::new("0xAAAA", "ethereum", 0.6))
//!     .with_split(RoyaltySplit::new("0xBBBB", "ethereum", 0.4));
//! rights.validate().unwrap();
//!
//! // C2PA credentials
//! let creds = NeoCredentials::new(ContentOrigin::Human)
//!     .with_creator("Producer Name");
//! ```

pub mod c2pa;
pub mod error;
pub mod schema;
pub mod temporal;
pub mod web3;

pub use c2pa::{ContentOrigin, CredentialSummary, NeoCredentials};
pub use error::{MetadataError, Result};
pub use schema::{
    Album, Artist, ContextEntry, MetadataContext, NeoEncoding, NeoMetadata, NeoStemInfo,
};
pub use temporal::{BpmEvent, ChordEvent, LyricEvent, TemporalMap};
pub use web3::{RightsInfo, RoyaltySplit};
