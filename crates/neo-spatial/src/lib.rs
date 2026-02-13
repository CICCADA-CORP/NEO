//! # neo-spatial — Spatial audio engine for the NEO format
//!
//! Provides 3D audio object positioning, Ambisonics B-format encoding/decoding,
//! and binaural rendering via HRTF for headphone output.
//!
//! See SPECIFICATION.md Section 11 for the spatial audio format.
//!
//! ## Architecture
//!
//! - **[`object`]**: Spatial scene model with 3D audio objects, keyframe interpolation,
//!   and room configuration. Produces the JSON data for the SPAT chunk.
//! - **[`ambisonics`]**: Encodes mono sources to B-format and decodes to speaker arrays
//!   (stereo, surround) using spherical harmonics.
//! - **[`binaural`]**: Renders spatial audio to headphone-optimized stereo using
//!   HRTF-based ITD/ILD model.
//! - **[`error`]**: Error types for all spatial audio operations.
//!
//! ## Quick Start
//!
//! ```rust
//! use neo_spatial::{SpatialScene, AudioObject, Position3D, RoomConfig};
//! use neo_spatial::object::RoomType;
//!
//! // Build a spatial scene
//! let scene = SpatialScene::new()
//!     .with_object(
//!         AudioObject::new(0)
//!             .with_gain(1.0)
//!             .with_spread(0.2)
//!             .with_keyframe(0.0, Position3D::new(-30.0, 0.0, 0.8).unwrap())
//!             .with_keyframe(5.0, Position3D::new(30.0, 10.0, 0.6).unwrap()),
//!     )
//!     .with_room(RoomConfig::new(RoomType::Studio, 0.3));
//!
//! // Serialize to JSON for the SPAT chunk
//! let json = scene.to_json().unwrap();
//!
//! // Interpolate position at a specific time
//! let pos = scene.position_at(0, 2.5).unwrap();
//! ```

pub mod ambisonics;
pub mod binaural;
pub mod error;
pub mod object;

pub use ambisonics::AmbisonicsOrder;
pub use binaural::{BinauralRenderer, HrtfDataset};
pub use error::{Result, SpatialError};
pub use object::{AudioObject, Position3D, PositionKeyframe, RoomConfig, SpatialScene};
