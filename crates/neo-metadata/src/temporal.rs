//! Temporal metadata for NEO files.
//!
//! Provides time-stamped annotations including chord progressions, lyrics,
//! and BPM changes. See SPECIFICATION.md Section 9.3.

use serde::{Deserialize, Serialize};

use crate::error::{MetadataError, Result};

/// Time-stamped metadata map for a NEO file.
///
/// Contains global tempo/key information and arrays of time-stamped events
/// for chords, lyrics, and BPM changes.
///
/// # Example
/// ```
/// use neo_metadata::TemporalMap;
///
/// let temporal = TemporalMap::new(120.0, "C major")
///     .with_chord(0.0, "C")
///     .with_chord(2.0, "Am")
///     .with_lyric(0.0, 1.5, "Hello world")
///     .with_bpm_change(30.0, 140.0);
///
/// let json = temporal.to_json_pretty().unwrap();
/// let parsed = TemporalMap::from_json(&json).unwrap();
/// assert_eq!(parsed.bpm, 120.0);
/// ```
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TemporalMap {
    /// Global BPM (beats per minute). Must be > 0.
    pub bpm: f64,

    /// Musical key (e.g., "C major", "A minor", "F# mixolydian").
    pub key: String,

    /// Time signature as a string (e.g., "4/4", "3/4", "6/8").
    pub time_signature: String,

    /// Chord events, in chronological order by `time`.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub chords: Vec<ChordEvent>,

    /// Lyric events, in chronological order by `time`.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub lyrics: Vec<LyricEvent>,

    /// BPM change events, in chronological order by `time`.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub bpm_changes: Vec<BpmEvent>,
}

/// A chord change at a specific timestamp.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ChordEvent {
    /// Time offset in seconds from the start of the audio.
    pub time: f64,

    /// Chord symbol (e.g., "Cmaj7", "Am", "F#dim").
    pub chord: String,
}

/// A lyric line spanning a time range.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LyricEvent {
    /// Start time in seconds.
    pub time: f64,

    /// End time in seconds. Must be >= `time`.
    pub end: f64,

    /// The lyric text for this time range.
    pub line: String,
}

/// A BPM change at a specific timestamp.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct BpmEvent {
    /// Time offset in seconds from the start of the audio.
    pub time: f64,

    /// New BPM value. Must be > 0.
    pub bpm: f64,
}

// ── Builders ───────────────────────────────────────────────────────────────

impl TemporalMap {
    /// Creates a new temporal map with the given BPM and key.
    ///
    /// Defaults to 4/4 time signature with empty event lists.
    pub fn new(bpm: f64, key: impl Into<String>) -> Self {
        Self {
            bpm,
            key: key.into(),
            time_signature: "4/4".to_string(),
            chords: Vec::new(),
            lyrics: Vec::new(),
            bpm_changes: Vec::new(),
        }
    }

    /// Sets the time signature.
    pub fn with_time_signature(mut self, time_sig: impl Into<String>) -> Self {
        self.time_signature = time_sig.into();
        self
    }

    /// Adds a chord event at the given time.
    pub fn with_chord(mut self, time: f64, chord: impl Into<String>) -> Self {
        self.chords.push(ChordEvent {
            time,
            chord: chord.into(),
        });
        self
    }

    /// Adds a lyric event spanning the given time range.
    pub fn with_lyric(mut self, time: f64, end: f64, line: impl Into<String>) -> Self {
        self.lyrics.push(LyricEvent {
            time,
            end,
            line: line.into(),
        });
        self
    }

    /// Adds a BPM change event at the given time.
    pub fn with_bpm_change(mut self, time: f64, bpm: f64) -> Self {
        self.bpm_changes.push(BpmEvent { time, bpm });
        self
    }

    /// Adds multiple chord events at once.
    pub fn with_chords(mut self, chords: Vec<ChordEvent>) -> Self {
        self.chords.extend(chords);
        self
    }

    /// Adds multiple lyric events at once.
    pub fn with_lyrics(mut self, lyrics: Vec<LyricEvent>) -> Self {
        self.lyrics.extend(lyrics);
        self
    }

    /// Serializes this temporal map to a JSON string.
    pub fn to_json(&self) -> Result<String> {
        serde_json::to_string(self).map_err(MetadataError::from)
    }

    /// Serializes this temporal map to a pretty-printed JSON string.
    pub fn to_json_pretty(&self) -> Result<String> {
        serde_json::to_string_pretty(self).map_err(MetadataError::from)
    }

    /// Deserializes a temporal map from a JSON string.
    pub fn from_json(json: &str) -> Result<Self> {
        serde_json::from_str(json).map_err(MetadataError::from)
    }

    /// Validates this temporal map.
    ///
    /// Checks:
    /// - BPM is positive
    /// - Key is not empty
    /// - Time signature matches pattern (e.g., "4/4")
    /// - Chord events are in chronological order with non-negative times
    /// - Lyric events are in chronological order with valid ranges (end >= time)
    /// - BPM change events are in chronological order with positive BPM values
    pub fn validate(&self) -> Result<()> {
        // Validate BPM
        if self.bpm <= 0.0 {
            return Err(MetadataError::OutOfRange {
                field: "bpm".to_string(),
                value: self.bpm.to_string(),
                expected: "positive value (> 0)".to_string(),
            });
        }

        // Validate key
        if self.key.trim().is_empty() {
            return Err(MetadataError::MissingField("key".to_string()));
        }

        // Validate time signature format
        if !self.time_signature.contains('/') {
            return Err(MetadataError::OutOfRange {
                field: "time_signature".to_string(),
                value: self.time_signature.clone(),
                expected: "format like '4/4', '3/4', '6/8'".to_string(),
            });
        }

        // Validate chord events: chronological order, non-negative times
        for (i, chord) in self.chords.iter().enumerate() {
            if chord.time < 0.0 {
                return Err(MetadataError::OutOfRange {
                    field: format!("chords[{i}].time"),
                    value: chord.time.to_string(),
                    expected: "non-negative value".to_string(),
                });
            }
            if i > 0 && chord.time < self.chords[i - 1].time {
                return Err(MetadataError::TemporalOrderError {
                    index: i,
                    details: format!(
                        "chord at {:.3}s comes before previous at {:.3}s",
                        chord.time,
                        self.chords[i - 1].time
                    ),
                });
            }
        }

        // Validate lyric events: chronological order, end >= time
        for (i, lyric) in self.lyrics.iter().enumerate() {
            if lyric.time < 0.0 {
                return Err(MetadataError::OutOfRange {
                    field: format!("lyrics[{i}].time"),
                    value: lyric.time.to_string(),
                    expected: "non-negative value".to_string(),
                });
            }
            if lyric.end < lyric.time {
                return Err(MetadataError::OutOfRange {
                    field: format!("lyrics[{i}].end"),
                    value: lyric.end.to_string(),
                    expected: format!(">= time ({:.3})", lyric.time),
                });
            }
            if i > 0 && lyric.time < self.lyrics[i - 1].time {
                return Err(MetadataError::TemporalOrderError {
                    index: i,
                    details: format!(
                        "lyric at {:.3}s comes before previous at {:.3}s",
                        lyric.time,
                        self.lyrics[i - 1].time
                    ),
                });
            }
        }

        // Validate BPM changes: chronological order, positive BPM
        for (i, bpm_ev) in self.bpm_changes.iter().enumerate() {
            if bpm_ev.time < 0.0 {
                return Err(MetadataError::OutOfRange {
                    field: format!("bpm_changes[{i}].time"),
                    value: bpm_ev.time.to_string(),
                    expected: "non-negative value".to_string(),
                });
            }
            if bpm_ev.bpm <= 0.0 {
                return Err(MetadataError::OutOfRange {
                    field: format!("bpm_changes[{i}].bpm"),
                    value: bpm_ev.bpm.to_string(),
                    expected: "positive value (> 0)".to_string(),
                });
            }
            if i > 0 && bpm_ev.time < self.bpm_changes[i - 1].time {
                return Err(MetadataError::TemporalOrderError {
                    index: i,
                    details: format!(
                        "bpm_change at {:.3}s comes before previous at {:.3}s",
                        bpm_ev.time,
                        self.bpm_changes[i - 1].time
                    ),
                });
            }
        }

        Ok(())
    }

    /// Returns the BPM at a given time, accounting for BPM changes.
    ///
    /// If no BPM change events exist before the given time, returns the global BPM.
    pub fn bpm_at(&self, time: f64) -> f64 {
        let mut current_bpm = self.bpm;
        for ev in &self.bpm_changes {
            if ev.time <= time {
                current_bpm = ev.bpm;
            } else {
                break;
            }
        }
        current_bpm
    }

    /// Returns the chord active at a given time, if any.
    ///
    /// Returns the most recent chord event at or before the given time.
    pub fn chord_at(&self, time: f64) -> Option<&str> {
        let mut current: Option<&str> = None;
        for ev in &self.chords {
            if ev.time <= time {
                current = Some(&ev.chord);
            } else {
                break;
            }
        }
        current
    }

    /// Returns all lyrics active at a given time.
    ///
    /// A lyric is active if `time >= lyric.time && time <= lyric.end`.
    pub fn lyrics_at(&self, time: f64) -> Vec<&LyricEvent> {
        self.lyrics
            .iter()
            .filter(|l| time >= l.time && time <= l.end)
            .collect()
    }
}

impl Default for TemporalMap {
    fn default() -> Self {
        Self::new(120.0, "C major")
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_temporal_map_new() {
        let map = TemporalMap::new(128.0, "A minor");
        assert_eq!(map.bpm, 128.0);
        assert_eq!(map.key, "A minor");
        assert_eq!(map.time_signature, "4/4");
        assert!(map.chords.is_empty());
    }

    #[test]
    fn test_temporal_builder_chain() {
        let map = TemporalMap::new(120.0, "C major")
            .with_time_signature("3/4")
            .with_chord(0.0, "C")
            .with_chord(2.0, "Am")
            .with_chord(4.0, "F")
            .with_lyric(0.0, 1.5, "Hello")
            .with_lyric(2.0, 3.5, "World")
            .with_bpm_change(16.0, 140.0);

        assert_eq!(map.time_signature, "3/4");
        assert_eq!(map.chords.len(), 3);
        assert_eq!(map.lyrics.len(), 2);
        assert_eq!(map.bpm_changes.len(), 1);
    }

    #[test]
    fn test_temporal_json_round_trip() {
        let map = TemporalMap::new(120.0, "D minor")
            .with_chord(0.0, "Dm")
            .with_chord(4.0, "Gm")
            .with_lyric(0.0, 3.0, "Test lyric");

        let json = map.to_json_pretty().unwrap();
        let parsed = TemporalMap::from_json(&json).unwrap();

        assert_eq!(map, parsed);
    }

    #[test]
    fn test_temporal_validate_ok() {
        let map = TemporalMap::new(120.0, "C major")
            .with_chord(0.0, "C")
            .with_chord(2.0, "Am")
            .with_lyric(0.0, 1.0, "Hello")
            .with_bpm_change(8.0, 140.0);

        assert!(map.validate().is_ok());
    }

    #[test]
    fn test_temporal_validate_bad_bpm() {
        let map = TemporalMap::new(0.0, "C major");
        assert!(map.validate().is_err());

        let map = TemporalMap::new(-10.0, "C major");
        assert!(map.validate().is_err());
    }

    #[test]
    fn test_temporal_validate_empty_key() {
        let map = TemporalMap::new(120.0, "  ");
        assert!(map.validate().is_err());
    }

    #[test]
    fn test_temporal_validate_chord_order() {
        let map = TemporalMap::new(120.0, "C major")
            .with_chord(4.0, "Am")
            .with_chord(2.0, "C"); // Out of order!

        let err = map.validate().unwrap_err();
        assert!(err.to_string().contains("comes before previous"));
    }

    #[test]
    fn test_temporal_validate_lyric_range() {
        let map = TemporalMap::new(120.0, "C major").with_lyric(3.0, 1.0, "Bad range"); // end < time!

        let err = map.validate().unwrap_err();
        assert!(err.to_string().contains("end"));
    }

    #[test]
    fn test_temporal_validate_lyric_order() {
        let map = TemporalMap::new(120.0, "C major")
            .with_lyric(5.0, 6.0, "Second")
            .with_lyric(2.0, 3.0, "First"); // Out of order!

        let err = map.validate().unwrap_err();
        assert!(err.to_string().contains("comes before previous"));
    }

    #[test]
    fn test_temporal_validate_bpm_change_positive() {
        let map = TemporalMap::new(120.0, "C major").with_bpm_change(8.0, -5.0);

        assert!(map.validate().is_err());
    }

    #[test]
    fn test_bpm_at() {
        let map = TemporalMap::new(120.0, "C major")
            .with_bpm_change(10.0, 140.0)
            .with_bpm_change(20.0, 160.0);

        assert_eq!(map.bpm_at(5.0), 120.0);
        assert_eq!(map.bpm_at(10.0), 140.0);
        assert_eq!(map.bpm_at(15.0), 140.0);
        assert_eq!(map.bpm_at(25.0), 160.0);
    }

    #[test]
    fn test_chord_at() {
        let map = TemporalMap::new(120.0, "C major")
            .with_chord(0.0, "C")
            .with_chord(4.0, "Am");

        assert_eq!(map.chord_at(0.0), Some("C"));
        assert_eq!(map.chord_at(3.0), Some("C"));
        assert_eq!(map.chord_at(4.0), Some("Am"));
        assert_eq!(map.chord_at(10.0), Some("Am"));
    }

    #[test]
    fn test_lyrics_at() {
        let map = TemporalMap::new(120.0, "C major")
            .with_lyric(0.0, 2.0, "Line one")
            .with_lyric(1.5, 3.0, "Line two"); // overlapping

        let at_1 = map.lyrics_at(1.0);
        assert_eq!(at_1.len(), 1);
        assert_eq!(at_1[0].line, "Line one");

        let at_1_7 = map.lyrics_at(1.7);
        assert_eq!(at_1_7.len(), 2); // both active

        let at_2_5 = map.lyrics_at(2.5);
        assert_eq!(at_2_5.len(), 1);
        assert_eq!(at_2_5[0].line, "Line two");
    }

    #[test]
    fn test_temporal_default() {
        let map = TemporalMap::default();
        assert_eq!(map.bpm, 120.0);
        assert_eq!(map.key, "C major");
    }
}
