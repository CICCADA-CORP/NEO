//! JSON-LD metadata schema for NEO files.
//!
//! Implements the metadata format defined in SPECIFICATION.md Section 9.
//! The schema uses JSON-LD with `schema.org/MusicRecording` as the base type,
//! extended with NEO-specific properties under the `neo:` namespace.

use serde::{Deserialize, Serialize};

use crate::error::{MetadataError, Result};

// ── Core Metadata ──────────────────────────────────────────────────────────

/// Top-level JSON-LD metadata for a NEO file.
///
/// Based on `schema.org/MusicRecording` with NEO extensions.
/// The `@context` and `@type` fields are automatically set during serialization.
///
/// # Example
/// ```
/// use neo_metadata::NeoMetadata;
///
/// let meta = NeoMetadata::new("My Song")
///     .with_artist("Artist Name")
///     .with_genre("Electronic")
///     .with_isrc("USRC12345678");
///
/// let json = meta.to_json_pretty().unwrap();
/// let parsed = NeoMetadata::from_json(&json).unwrap();
/// assert_eq!(parsed.name, "My Song");
/// ```
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct NeoMetadata {
    /// JSON-LD context — always includes schema.org and neo vocabulary.
    #[serde(rename = "@context")]
    pub context: MetadataContext,

    /// JSON-LD type — always `"MusicRecording"`.
    #[serde(rename = "@type")]
    pub schema_type: String,

    /// Track title (required).
    pub name: String,

    /// Artist information.
    #[serde(rename = "byArtist", skip_serializing_if = "Option::is_none")]
    pub by_artist: Option<Artist>,

    /// Duration in ISO 8601 format (e.g., "PT3M45S").
    #[serde(skip_serializing_if = "Option::is_none")]
    pub duration: Option<String>,

    /// International Standard Recording Code.
    #[serde(rename = "isrcCode", skip_serializing_if = "Option::is_none")]
    pub isrc_code: Option<String>,

    /// Album information.
    #[serde(rename = "inAlbum", skip_serializing_if = "Option::is_none")]
    pub in_album: Option<Album>,

    /// Genre tag.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub genre: Option<String>,

    /// Publication date in ISO 8601 format (e.g., "2025-01-15").
    #[serde(rename = "datePublished", skip_serializing_if = "Option::is_none")]
    pub date_published: Option<String>,

    /// NEO extension: stem information embedded in this file.
    #[serde(rename = "neo:stems", skip_serializing_if = "Option::is_none")]
    pub neo_stems: Option<Vec<NeoStemInfo>>,

    /// NEO extension: encoding parameters used.
    #[serde(rename = "neo:encoding", skip_serializing_if = "Option::is_none")]
    pub neo_encoding: Option<NeoEncoding>,
}

/// JSON-LD context supporting both schema.org and NEO namespace.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(untagged)]
pub enum MetadataContext {
    /// Simple string context (schema.org only).
    Simple(String),
    /// Extended context with NEO namespace.
    Extended(Vec<ContextEntry>),
}

/// A single entry in the JSON-LD @context array.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(untagged)]
pub enum ContextEntry {
    /// A URI string (e.g., `https://schema.org`).
    Uri(String),
    /// A namespace mapping (e.g., `{"neo": "https://neo-audio.org/ns/"}`).
    Mapping(std::collections::HashMap<String, String>),
}

/// Artist information following schema.org/Person.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Artist {
    /// JSON-LD type — always `"Person"`.
    #[serde(rename = "@type")]
    pub schema_type: String,

    /// Artist name.
    pub name: String,
}

/// Album information following schema.org/MusicAlbum.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Album {
    /// JSON-LD type — always `"MusicAlbum"`.
    #[serde(rename = "@type")]
    pub schema_type: String,

    /// Album title.
    pub name: String,

    /// Number of tracks in the album.
    #[serde(rename = "numTracks", skip_serializing_if = "Option::is_none")]
    pub num_tracks: Option<u32>,
}

// ── NEO Extensions (Section 9.2) ──────────────────────────────────────────

/// NEO extension: information about a stem in the file.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct NeoStemInfo {
    /// Stem identifier (0-based index).
    pub id: u8,

    /// Human-readable stem label (e.g., "vocals", "drums").
    pub label: String,

    /// Codec used for this stem (e.g., "dac", "pcm", "flac").
    pub codec: String,

    /// Number of audio channels.
    pub channels: u8,

    /// Sample rate in Hz.
    pub sample_rate: u32,
}

/// NEO extension: encoding parameters.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct NeoEncoding {
    /// Format version string (e.g., "0.1.0").
    pub version: String,

    /// Target bitrate in kbps (0 for lossless).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bitrate_kbps: Option<u32>,

    /// Whether lossless residual data is included.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lossless: Option<bool>,

    /// Whether spatial audio data is included.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub spatial: Option<bool>,
}

// ── Default Context ────────────────────────────────────────────────────────

/// Creates the default JSON-LD context with schema.org + NEO namespace.
fn default_context() -> MetadataContext {
    MetadataContext::Extended(vec![
        ContextEntry::Uri("https://schema.org".to_string()),
        ContextEntry::Mapping({
            let mut m = std::collections::HashMap::new();
            m.insert("neo".to_string(), "https://neo-audio.org/ns/".to_string());
            m
        }),
    ])
}

// ── Builders & Convenience Methods ─────────────────────────────────────────

impl NeoMetadata {
    /// Creates a new `NeoMetadata` with the given track name.
    ///
    /// All optional fields are set to `None`. The JSON-LD context
    /// includes both schema.org and the NEO namespace.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            context: default_context(),
            schema_type: "MusicRecording".to_string(),
            name: name.into(),
            by_artist: None,
            duration: None,
            isrc_code: None,
            in_album: None,
            genre: None,
            date_published: None,
            neo_stems: None,
            neo_encoding: None,
        }
    }

    /// Sets the artist name.
    pub fn with_artist(mut self, name: impl Into<String>) -> Self {
        self.by_artist = Some(Artist {
            schema_type: "Person".to_string(),
            name: name.into(),
        });
        self
    }

    /// Sets the album name.
    pub fn with_album(mut self, name: impl Into<String>) -> Self {
        self.in_album = Some(Album {
            schema_type: "MusicAlbum".to_string(),
            name: name.into(),
            num_tracks: None,
        });
        self
    }

    /// Sets the album name and track count.
    pub fn with_album_tracks(mut self, name: impl Into<String>, num_tracks: u32) -> Self {
        self.in_album = Some(Album {
            schema_type: "MusicAlbum".to_string(),
            name: name.into(),
            num_tracks: Some(num_tracks),
        });
        self
    }

    /// Sets the genre.
    pub fn with_genre(mut self, genre: impl Into<String>) -> Self {
        self.genre = Some(genre.into());
        self
    }

    /// Sets the ISRC code.
    pub fn with_isrc(mut self, isrc: impl Into<String>) -> Self {
        self.isrc_code = Some(isrc.into());
        self
    }

    /// Sets the duration in ISO 8601 format.
    pub fn with_duration(mut self, duration: impl Into<String>) -> Self {
        self.duration = Some(duration.into());
        self
    }

    /// Sets the publication date in ISO 8601 format.
    pub fn with_date_published(mut self, date: impl Into<String>) -> Self {
        self.date_published = Some(date.into());
        self
    }

    /// Sets the NEO stem information extension.
    pub fn with_stems(mut self, stems: Vec<NeoStemInfo>) -> Self {
        self.neo_stems = Some(stems);
        self
    }

    /// Sets the NEO encoding extension.
    pub fn with_encoding(mut self, encoding: NeoEncoding) -> Self {
        self.neo_encoding = Some(encoding);
        self
    }

    /// Serializes this metadata to a JSON string.
    pub fn to_json(&self) -> Result<String> {
        serde_json::to_string(self).map_err(MetadataError::from)
    }

    /// Serializes this metadata to a pretty-printed JSON string.
    pub fn to_json_pretty(&self) -> Result<String> {
        serde_json::to_string_pretty(self).map_err(MetadataError::from)
    }

    /// Deserializes metadata from a JSON string.
    pub fn from_json(json: &str) -> Result<Self> {
        serde_json::from_str(json).map_err(MetadataError::from)
    }

    /// Validates this metadata, returning errors for any issues.
    ///
    /// Checks:
    /// - Name is not empty
    /// - ISRC format (if present): 12 alphanumeric characters
    /// - Duration format (if present): starts with "PT"
    pub fn validate(&self) -> Result<()> {
        if self.name.trim().is_empty() {
            return Err(MetadataError::MissingField("name".to_string()));
        }

        if let Some(ref isrc) = self.isrc_code {
            if isrc.len() != 12 || !isrc.chars().all(|c| c.is_ascii_alphanumeric()) {
                return Err(MetadataError::OutOfRange {
                    field: "isrc_code".to_string(),
                    value: isrc.clone(),
                    expected: "12 alphanumeric characters (e.g., USRC12345678)".to_string(),
                });
            }
        }

        if let Some(ref dur) = self.duration {
            if !dur.starts_with("PT") {
                return Err(MetadataError::OutOfRange {
                    field: "duration".to_string(),
                    value: dur.clone(),
                    expected: "ISO 8601 duration starting with 'PT' (e.g., PT3M45S)".to_string(),
                });
            }
        }

        Ok(())
    }
}

impl Default for NeoMetadata {
    fn default() -> Self {
        Self::new("Untitled")
    }
}

impl NeoEncoding {
    /// Creates a new encoding info with the given version.
    pub fn new(version: impl Into<String>) -> Self {
        Self {
            version: version.into(),
            bitrate_kbps: None,
            lossless: None,
            spatial: None,
        }
    }
}

impl NeoStemInfo {
    /// Creates a new stem info entry.
    pub fn new(
        id: u8,
        label: impl Into<String>,
        codec: impl Into<String>,
        channels: u8,
        sample_rate: u32,
    ) -> Self {
        Self {
            id,
            label: label.into(),
            codec: codec.into(),
            channels,
            sample_rate,
        }
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metadata_new() {
        let meta = NeoMetadata::new("Test Song");
        assert_eq!(meta.name, "Test Song");
        assert_eq!(meta.schema_type, "MusicRecording");
        assert!(meta.by_artist.is_none());
    }

    #[test]
    fn test_metadata_builder_chain() {
        let meta = NeoMetadata::new("My Track")
            .with_artist("DJ Test")
            .with_genre("House")
            .with_album_tracks("Best Of", 12)
            .with_isrc("USRC12345678")
            .with_duration("PT4M30S")
            .with_date_published("2025-06-15");

        assert_eq!(meta.by_artist.as_ref().unwrap().name, "DJ Test");
        assert_eq!(meta.genre.as_ref().unwrap(), "House");
        assert_eq!(meta.in_album.as_ref().unwrap().num_tracks, Some(12));
        assert_eq!(meta.isrc_code.as_ref().unwrap(), "USRC12345678");
    }

    #[test]
    fn test_metadata_json_round_trip() {
        let meta = NeoMetadata::new("Round Trip Test")
            .with_artist("Test Artist")
            .with_genre("Jazz");

        let json = meta.to_json_pretty().unwrap();
        let parsed = NeoMetadata::from_json(&json).unwrap();

        assert_eq!(meta, parsed);
    }

    #[test]
    fn test_metadata_with_neo_extensions() {
        let meta = NeoMetadata::new("Extended")
            .with_stems(vec![
                NeoStemInfo::new(0, "vocals", "dac", 1, 44100),
                NeoStemInfo::new(1, "instrumental", "pcm", 2, 44100),
            ])
            .with_encoding(NeoEncoding {
                version: "0.1.0".to_string(),
                bitrate_kbps: Some(128),
                lossless: Some(false),
                spatial: None,
            });

        let json = meta.to_json_pretty().unwrap();
        assert!(json.contains("neo:stems"));
        assert!(json.contains("neo:encoding"));

        let parsed = NeoMetadata::from_json(&json).unwrap();
        assert_eq!(parsed.neo_stems.as_ref().unwrap().len(), 2);
        assert_eq!(parsed.neo_encoding.as_ref().unwrap().version, "0.1.0");
    }

    #[test]
    fn test_metadata_validate_ok() {
        let meta = NeoMetadata::new("Valid")
            .with_isrc("USRC12345678")
            .with_duration("PT3M");
        assert!(meta.validate().is_ok());
    }

    #[test]
    fn test_metadata_validate_empty_name() {
        let meta = NeoMetadata::new("  ");
        assert!(meta.validate().is_err());
    }

    #[test]
    fn test_metadata_validate_bad_isrc() {
        let meta = NeoMetadata::new("Test").with_isrc("short");
        let err = meta.validate().unwrap_err();
        assert!(err.to_string().contains("isrc_code"));
    }

    #[test]
    fn test_metadata_validate_bad_duration() {
        let meta = NeoMetadata::new("Test").with_duration("3:45");
        let err = meta.validate().unwrap_err();
        assert!(err.to_string().contains("duration"));
    }

    #[test]
    fn test_metadata_default() {
        let meta = NeoMetadata::default();
        assert_eq!(meta.name, "Untitled");
    }

    #[test]
    fn test_metadata_json_ld_context() {
        let meta = NeoMetadata::new("Context Test");
        let json = meta.to_json_pretty().unwrap();
        assert!(json.contains("schema.org"));
        assert!(json.contains("neo-audio.org/ns/"));
    }
}
