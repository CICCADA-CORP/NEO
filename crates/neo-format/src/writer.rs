//! NEO file writer — serializes audio stems and metadata into the `.neo` container format.
//!
//! The writer uses a builder pattern: create a [`NeoWriter`], add stems and metadata,
//! then call [`NeoWriter::finalize`] to write the complete `.neo` file to disk.
//!
//! # Binary Layout
//!
//! The writer produces files conforming to the NEO specification:
//! - **Header** (64 bytes): magic, version, flags, stem count, sample rate, etc.
//! - **Chunk data**: PREV, STEM, META, TEMP, CRED, SPAT, EDIT, RGHT chunks written sequentially
//! - **Chunk table**: index of all chunks with BLAKE3 hashes (57 bytes per entry)
//!
//! # Example
//!
//! ```rust,no_run
//! use neo_format::writer::NeoWriter;
//! use neo_format::stem::{StemConfig, StemLabel, CodecId};
//! use std::path::Path;
//!
//! let config = StemConfig::new(0, StemLabel::Vocals, CodecId::Dac, 2, 44100);
//! let audio_data: Vec<u8> = vec![0u8; 1024]; // compressed audio bytes
//!
//! let mut writer = NeoWriter::new(44100, 1);
//! writer
//!     .add_stem(config, audio_data).unwrap()
//!     .set_metadata(r#"{"@context":"https://schema.org"}"#.to_string())
//!     .set_duration_us(3_000_000);
//! writer.finalize(Path::new("output.neo")).unwrap();
//! ```

use std::io::{Seek, SeekFrom, Write};
use std::path::Path;

use byteorder::{LittleEndian, WriteBytesExt};

use crate::chunk::{ChunkEntry, ChunkType, CHUNK_ENTRY_SIZE};
use crate::error::{NeoFormatError, Result};
use crate::header::{FeatureFlags, NeoHeader, HEADER_SIZE, MAX_STEMS, NEO_MAGIC, NEO_VERSION};
use crate::stem::StemConfig;

/// A pending chunk to be written, holding its type and serialized data bytes.
#[derive(Debug, Clone)]
struct PendingChunk {
    /// The chunk type identifier.
    chunk_type: ChunkType,
    /// The fully serialized chunk data (stem header + audio, or raw JSON bytes).
    data: Vec<u8>,
}

/// Builder for creating `.neo` files.
///
/// Collects stems, metadata, and other chunk data, then writes the complete
/// binary container in a single [`finalize`](NeoWriter::finalize) call.
///
/// # Chunk Ordering
///
/// Per the specification (Section 12.3), chunks are ordered as:
/// PREV, STEM (by stem_id), META, TEMP, RESD, CRED, SPAT, EDIT, RGHT.
///
/// # Example
///
/// ```rust,no_run
/// use neo_format::{NeoWriter, StemConfig, StemLabel, CodecId};
/// use std::path::Path;
///
/// let mut writer = NeoWriter::new(44100, 2);
/// let vocals = StemConfig::new(0, StemLabel::Vocals, CodecId::Dac, 2, 44100);
/// let drums = StemConfig::new(1, StemLabel::Drums, CodecId::Dac, 2, 44100);
/// writer.add_stem(vocals, vec![0u8; 512]).unwrap();
/// writer.add_stem(drums, vec![0u8; 512]).unwrap();
/// writer.set_metadata(r#"{"@context":"https://schema.org"}"#.to_string());
/// writer.finalize(Path::new("track.neo")).unwrap();
/// ```
pub struct NeoWriter {
    /// The file header (updated during finalize with chunk_table_offset and chunk_count).
    header: NeoHeader,
    /// Stem configs paired with their compressed audio data.
    stems: Vec<(StemConfig, Vec<u8>)>,
    /// Optional JSON-LD metadata string (META chunk, type 0x02).
    metadata_json: Option<String>,
    /// Optional temporal metadata JSON string (TEMP chunk, type 0x08).
    temporal_json: Option<String>,
    /// Optional Web3 rights JSON string (RGHT chunk, type 0x09).
    rights_json: Option<String>,
    /// Optional C2PA content credentials (raw JUMBF manifest bytes, CRED chunk, type 0x03).
    credentials_bytes: Option<Vec<u8>>,
    /// Optional spatial audio scene JSON (SPAT chunk, type 0x04).
    spatial_json: Option<String>,
    /// Optional non-destructive edit history JSON (EDIT chunk, type 0x05).
    edit_history_json: Option<String>,
    /// Optional low-quality preview audio bytes (PREV chunk, type 0x06).
    preview_data: Option<Vec<u8>>,
    /// Feature flags accumulated from the added content.
    feature_flags: FeatureFlags,
}

impl NeoWriter {
    /// Create a new NEO file writer with the given sample rate and expected stem count.
    ///
    /// # Arguments
    ///
    /// * `sample_rate` — The master sample rate in Hz (e.g., 44100, 48000).
    /// * `stem_count` — The number of audio stems this file will contain (1–8).
    pub fn new(sample_rate: u32, stem_count: u8) -> Self {
        Self {
            header: NeoHeader::new(sample_rate, stem_count),
            stems: Vec::new(),
            metadata_json: None,
            temporal_json: None,
            rights_json: None,
            credentials_bytes: None,
            spatial_json: None,
            edit_history_json: None,
            preview_data: None,
            feature_flags: FeatureFlags::new(),
        }
    }

    /// Add an audio stem with its compressed data.
    ///
    /// The stem's `stem_id` must be unique and within `0..stem_count`.
    /// The `data` parameter contains the already-compressed audio bytes
    /// (DAC codes, FLAC stream, Opus packets, or raw PCM).
    ///
    /// # Errors
    ///
    /// Returns [`NeoFormatError::TooManyStems`] if adding this stem would exceed
    /// the [`MAX_STEMS`] limit (8).
    pub fn add_stem(&mut self, config: StemConfig, data: Vec<u8>) -> Result<&mut Self> {
        let new_count = self.stems.len() as u8 + 1;
        if new_count > MAX_STEMS {
            return Err(NeoFormatError::TooManyStems {
                max: MAX_STEMS,
                got: new_count,
            });
        }
        tracing::debug!(
            stem_id = config.stem_id,
            label = config.label.as_str(),
            codec = ?config.codec,
            data_len = data.len(),
            "Adding stem to writer"
        );
        self.stems.push((config, data));
        Ok(self)
    }

    /// Set the JSON-LD metadata for this file (META chunk, type 0x02).
    ///
    /// The string should be valid JSON conforming to the NEO metadata schema
    /// (see SPECIFICATION.md Section 9).
    pub fn set_metadata(&mut self, json: String) -> &mut Self {
        self.metadata_json = Some(json);
        self
    }

    /// Set the temporal metadata for this file (TEMP chunk, type 0x08).
    ///
    /// The string should be valid JSON containing BPM, key, chords, lyrics,
    /// and other temporal annotations (see SPECIFICATION.md Section 9.3).
    pub fn set_temporal(&mut self, json: String) -> &mut Self {
        self.temporal_json = Some(json);
        self.feature_flags.set(FeatureFlags::TEMPORAL_META);
        self
    }

    /// Set the Web3 rights/royalty information for this file (RGHT chunk, type 0x09).
    ///
    /// The string should be valid JSON containing split information, contract
    /// addresses, and license URIs (see SPECIFICATION.md Section 9.4).
    pub fn set_rights(&mut self, json: String) -> &mut Self {
        self.rights_json = Some(json);
        self.feature_flags.set(FeatureFlags::WEB3_RIGHTS);
        self
    }

    /// Sets the C2PA content credentials (raw JUMBF manifest bytes).
    ///
    /// These bytes are stored as-is in the CRED chunk. The writer
    /// sets the `C2PA` feature flag automatically.
    pub fn set_credentials(&mut self, bytes: Vec<u8>) -> &mut Self {
        self.header.feature_flags.set(FeatureFlags::C2PA);
        self.credentials_bytes = Some(bytes);
        self
    }

    /// Set the spatial audio scene data for this file (SPAT chunk, type 0x04).
    ///
    /// The string should be valid JSON containing objects, keyframes, and room
    /// configuration (see SPECIFICATION.md Section 11).
    pub fn set_spatial(&mut self, json: String) -> &mut Self {
        self.spatial_json = Some(json);
        self.feature_flags.set(FeatureFlags::SPATIAL);
        self
    }

    /// Set the non-destructive edit history for this file (EDIT chunk, type 0x05).
    ///
    /// The string should be valid JSON containing the edit DAG with commits
    /// and operations (see SPECIFICATION.md Section 13).
    pub fn set_edit_history(&mut self, json: String) -> &mut Self {
        self.edit_history_json = Some(json);
        self.feature_flags.set(FeatureFlags::EDIT_HISTORY);
        self
    }

    /// Set the low-quality preview audio for this file (PREV chunk, type 0x06).
    ///
    /// This should be Opus-encoded mono/stereo audio at ≤32 kbps for progressive
    /// loading. The PREV chunk is always written first per spec Section 12.3.
    pub fn set_preview(&mut self, data: Vec<u8>) -> &mut Self {
        self.preview_data = Some(data);
        self.feature_flags.set(FeatureFlags::PREVIEW_LOD);
        self
    }

    /// Set the total duration in microseconds.
    ///
    /// This value is written into the header's `duration_us` field and represents
    /// the total playback duration of the file.
    pub fn set_duration_us(&mut self, duration_us: u64) -> &mut Self {
        self.header.duration_us = duration_us;
        self
    }

    /// Set additional feature flags on the header.
    ///
    /// Flags are OR'd with any flags already set by the builder (e.g., from
    /// [`set_temporal`](NeoWriter::set_temporal) or [`set_rights`](NeoWriter::set_rights)).
    pub fn set_feature_flags(&mut self, flags: FeatureFlags) -> &mut Self {
        self.feature_flags.0 |= flags.0;
        self
    }

    /// Serialize a stem header and its compressed audio data into a single byte vector.
    ///
    /// The stem chunk data layout (per SPECIFICATION.md Section 7.1.1):
    /// - stem_id (u8)
    /// - codec_id (u8)
    /// - channels (u8)
    /// - sample_rate (u32 LE)
    /// - bit_depth (u8)
    /// - bitrate_kbps (u32 LE)
    /// - sample_count (u64 LE)
    /// - label_len (u16 LE)
    /// - label (UTF-8 bytes)
    /// - [compressed audio data bytes]
    fn serialize_stem_chunk(config: &StemConfig, sample_count: u64, audio_data: &[u8]) -> Vec<u8> {
        let label_bytes = config.label.as_str().as_bytes();
        // Stem header size: 1 + 1 + 1 + 4 + 1 + 4 + 8 + 2 + label_len = 22 + label_len
        let capacity = 22 + label_bytes.len() + audio_data.len();
        let mut buf = Vec::with_capacity(capacity);

        // Write stem header fields
        buf.write_u8(config.stem_id)
            .expect("write to Vec cannot fail");
        buf.write_u8(config.codec as u8)
            .expect("write to Vec cannot fail");
        buf.write_u8(config.channels)
            .expect("write to Vec cannot fail");
        buf.write_u32::<LittleEndian>(config.sample_rate)
            .expect("write to Vec cannot fail");
        buf.write_u8(config.bit_depth)
            .expect("write to Vec cannot fail");
        buf.write_u32::<LittleEndian>(config.bitrate_kbps)
            .expect("write to Vec cannot fail");
        buf.write_u64::<LittleEndian>(sample_count)
            .expect("write to Vec cannot fail");
        buf.write_u16::<LittleEndian>(label_bytes.len() as u16)
            .expect("write to Vec cannot fail");
        buf.write_all(label_bytes)
            .expect("write to Vec cannot fail");

        // Write compressed audio data
        buf.write_all(audio_data).expect("write to Vec cannot fail");

        buf
    }

    /// Write the 64-byte NEO header to the given writer.
    ///
    /// The header layout is defined in SPECIFICATION.md Section 5.1.
    fn write_header<W: Write>(writer: &mut W, header: &NeoHeader) -> Result<()> {
        // [0..4]: Magic bytes
        writer.write_all(&NEO_MAGIC)?;
        // [4..6]: Version (u16 LE)
        writer.write_u16::<LittleEndian>(header.version)?;
        // [6..14]: Feature flags (u64 LE)
        writer.write_u64::<LittleEndian>(header.feature_flags.0)?;
        // [14]: Stem count (u8)
        writer.write_u8(header.stem_count)?;
        // [15..19]: Sample rate (u32 LE)
        writer.write_u32::<LittleEndian>(header.sample_rate)?;
        // [19..27]: Duration in microseconds (u64 LE)
        writer.write_u64::<LittleEndian>(header.duration_us)?;
        // [27..35]: Chunk table offset (u64 LE)
        writer.write_u64::<LittleEndian>(header.chunk_table_offset)?;
        // [35..43]: Chunk count (u64 LE)
        writer.write_u64::<LittleEndian>(header.chunk_count)?;
        // [43..64]: Reserved (21 bytes of zeros)
        writer.write_all(&[0u8; 21])?;
        Ok(())
    }

    /// Write a single chunk table entry (57 bytes) to the given writer.
    ///
    /// Layout per SPECIFICATION.md Section 6.1:
    /// - chunk_type (u8)
    /// - offset (u64 LE)
    /// - size (u64 LE)
    /// - blake3_hash ([u8; 32])
    /// - reserved ([u8; 8])
    fn write_chunk_entry<W: Write>(writer: &mut W, entry: &ChunkEntry) -> Result<()> {
        writer.write_u8(entry.chunk_type as u8)?;
        writer.write_u64::<LittleEndian>(entry.offset)?;
        writer.write_u64::<LittleEndian>(entry.size)?;
        writer.write_all(&entry.blake3_hash)?;
        // Reserved 8 bytes
        writer.write_all(&[0u8; 8])?;
        Ok(())
    }

    /// Collect all pending chunks in the correct order per spec Section 12.3:
    /// PREV, STEM (by stem_id), META, TEMP, RESD, CRED, SPAT, EDIT, RGHT.
    fn build_pending_chunks(&self) -> Vec<PendingChunk> {
        let mut pending = Vec::new();

        // PREV chunk (must be first after header, per spec Section 12.3)
        if let Some(ref data) = self.preview_data {
            pending.push(PendingChunk {
                chunk_type: ChunkType::Preview,
                data: data.clone(),
            });
        }

        // STEM chunks sorted by stem_id
        let mut sorted_stems: Vec<&(StemConfig, Vec<u8>)> = self.stems.iter().collect();
        sorted_stems.sort_by_key(|(config, _)| config.stem_id);

        for (config, audio_data) in &sorted_stems {
            // Derive sample_count from audio data size and config.
            // For the writer, sample_count is stored as 0 unless the caller sets it.
            // We use 0 as a placeholder — the actual sample count depends on the codec.
            // TODO: Accept sample_count as part of stem data or StemConfig in a future iteration.
            let sample_count = 0u64;
            let data = Self::serialize_stem_chunk(config, sample_count, audio_data);
            pending.push(PendingChunk {
                chunk_type: ChunkType::Stem,
                data,
            });
        }

        // META chunk
        if let Some(ref json) = self.metadata_json {
            pending.push(PendingChunk {
                chunk_type: ChunkType::Metadata,
                data: json.as_bytes().to_vec(),
            });
        }

        // TEMP chunk
        if let Some(ref json) = self.temporal_json {
            pending.push(PendingChunk {
                chunk_type: ChunkType::Temporal,
                data: json.as_bytes().to_vec(),
            });
        }

        // CRED chunk (raw JUMBF bytes)
        if let Some(ref bytes) = self.credentials_bytes {
            pending.push(PendingChunk {
                chunk_type: ChunkType::Credentials,
                data: bytes.clone(),
            });
        }

        // SPAT chunk
        if let Some(ref json) = self.spatial_json {
            pending.push(PendingChunk {
                chunk_type: ChunkType::Spatial,
                data: json.as_bytes().to_vec(),
            });
        }

        // EDIT chunk
        if let Some(ref json) = self.edit_history_json {
            pending.push(PendingChunk {
                chunk_type: ChunkType::EditHistory,
                data: json.as_bytes().to_vec(),
            });
        }

        // RGHT chunk
        if let Some(ref json) = self.rights_json {
            pending.push(PendingChunk {
                chunk_type: ChunkType::Rights,
                data: json.as_bytes().to_vec(),
            });
        }

        pending
    }

    /// Finalize and write the complete `.neo` file to disk.
    ///
    /// This method:
    /// 1. Writes a placeholder 64-byte header
    /// 2. Writes all chunk data sequentially (PREV, STEM, META, TEMP, CRED, SPAT, EDIT, RGHT)
    /// 3. Computes BLAKE3 hashes and records offsets for each chunk
    /// 4. Writes the chunk table at the current file position
    /// 5. Seeks back to byte 0 and rewrites the header with final
    ///    `chunk_table_offset` and `chunk_count` values
    ///
    /// # Errors
    ///
    /// Returns [`NeoFormatError::Io`] if any file I/O operation fails.
    /// Returns [`NeoFormatError::TooManyStems`] if stem count validation fails.
    pub fn finalize(&mut self, path: &Path) -> Result<()> {
        tracing::info!(path = %path.display(), "Finalizing NEO file");

        // Validate stem count matches
        let actual_stems = self.stems.len() as u8;
        if actual_stems > MAX_STEMS {
            return Err(NeoFormatError::TooManyStems {
                max: MAX_STEMS,
                got: actual_stems,
            });
        }
        // Update stem_count to match the actual number of stems added
        self.header.stem_count = actual_stems;
        self.header.version = NEO_VERSION;

        // Merge accumulated feature flags into the header
        self.header.feature_flags.0 |= self.feature_flags.0;

        // Collect all chunks in spec-defined order
        let pending_chunks = self.build_pending_chunks();

        tracing::debug!(
            chunk_count = pending_chunks.len(),
            stem_count = actual_stems,
            "Writing chunks"
        );

        // Open the output file
        let file = std::fs::File::create(path)?;
        let mut writer = std::io::BufWriter::new(file);

        // Step 1: Write placeholder header (will be overwritten at the end)
        Self::write_header(&mut writer, &self.header)?;

        // Step 2: Write each chunk's data and record chunk table entries
        let mut chunk_entries: Vec<ChunkEntry> = Vec::with_capacity(pending_chunks.len());
        let mut current_offset = HEADER_SIZE as u64;

        for pending in &pending_chunks {
            let data = &pending.data;
            let size = data.len() as u64;

            // Compute BLAKE3 hash of the chunk data
            let hash = blake3::hash(data);

            // Record the chunk table entry
            let entry = ChunkEntry {
                chunk_type: pending.chunk_type,
                offset: current_offset,
                size,
                blake3_hash: *hash.as_bytes(),
            };
            chunk_entries.push(entry);

            tracing::debug!(
                chunk_type = ?pending.chunk_type,
                offset = current_offset,
                size = size,
                hash = %hash.to_hex(),
                "Writing chunk"
            );

            // Write the chunk data
            writer.write_all(data)?;
            current_offset += size;
        }

        // Step 3: Record chunk table offset (current position)
        let chunk_table_offset = current_offset;

        tracing::debug!(
            chunk_table_offset = chunk_table_offset,
            entry_count = chunk_entries.len(),
            table_size = chunk_entries.len() * CHUNK_ENTRY_SIZE,
            "Writing chunk table"
        );

        // Step 4: Write the chunk table
        for entry in &chunk_entries {
            Self::write_chunk_entry(&mut writer, entry)?;
        }

        // Step 5: Update the header with final values
        self.header.chunk_table_offset = chunk_table_offset;
        self.header.chunk_count = chunk_entries.len() as u64;

        // Step 6: Seek back to the beginning and rewrite the header
        writer.seek(SeekFrom::Start(0))?;
        Self::write_header(&mut writer, &self.header)?;

        // Flush all buffered data
        writer.flush()?;

        tracing::info!(
            path = %path.display(),
            chunks = chunk_entries.len(),
            chunk_table_offset = chunk_table_offset,
            file_size = chunk_table_offset + (chunk_entries.len() as u64 * CHUNK_ENTRY_SIZE as u64),
            "NEO file written successfully"
        );

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stem::{CodecId, StemLabel};
    use std::io::{Read, Seek, SeekFrom};

    use byteorder::{LittleEndian, ReadBytesExt};

    #[test]
    fn test_header_round_trip() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.neo");

        let mut writer = NeoWriter::new(44100, 1);
        let config = StemConfig::new(0, StemLabel::Vocals, CodecId::Dac, 2, 44100);
        writer.add_stem(config, vec![0xAB; 64]).unwrap();
        writer.set_duration_us(5_000_000);
        writer.finalize(&path).unwrap();

        // Read back and verify header
        let mut file = std::fs::File::open(&path).unwrap();
        let mut magic = [0u8; 4];
        file.read_exact(&mut magic).unwrap();
        assert_eq!(magic, NEO_MAGIC);

        let version = file.read_u16::<LittleEndian>().unwrap();
        assert_eq!(version, NEO_VERSION);

        let _flags = file.read_u64::<LittleEndian>().unwrap();
        let stem_count = file.read_u8().unwrap();
        assert_eq!(stem_count, 1);

        let sample_rate = file.read_u32::<LittleEndian>().unwrap();
        assert_eq!(sample_rate, 44100);

        let duration_us = file.read_u64::<LittleEndian>().unwrap();
        assert_eq!(duration_us, 5_000_000);

        let chunk_table_offset = file.read_u64::<LittleEndian>().unwrap();
        let chunk_count = file.read_u64::<LittleEndian>().unwrap();
        assert_eq!(chunk_count, 1); // one STEM chunk

        // Verify reserved bytes are zero
        let mut reserved = [0u8; 21];
        file.read_exact(&mut reserved).unwrap();
        assert_eq!(reserved, [0u8; 21]);

        // Verify chunk table offset makes sense
        assert!(chunk_table_offset > HEADER_SIZE as u64);
    }

    #[test]
    fn test_stem_chunk_data_layout() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("stem_test.neo");

        let audio = vec![0xDE, 0xAD, 0xBE, 0xEF];
        let config = StemConfig::new(0, StemLabel::Vocals, CodecId::Dac, 2, 44100);

        let mut writer = NeoWriter::new(44100, 1);
        writer.add_stem(config, audio.clone()).unwrap();
        writer.finalize(&path).unwrap();

        // Read past header, then parse the stem chunk data
        let mut file = std::fs::File::open(&path).unwrap();
        file.seek(SeekFrom::Start(HEADER_SIZE as u64)).unwrap();

        let stem_id = file.read_u8().unwrap();
        assert_eq!(stem_id, 0);

        let codec_id = file.read_u8().unwrap();
        assert_eq!(codec_id, CodecId::Dac as u8);

        let channels = file.read_u8().unwrap();
        assert_eq!(channels, 2);

        let sr = file.read_u32::<LittleEndian>().unwrap();
        assert_eq!(sr, 44100);

        let bit_depth = file.read_u8().unwrap();
        assert_eq!(bit_depth, 16);

        let bitrate = file.read_u32::<LittleEndian>().unwrap();
        assert_eq!(bitrate, 0);

        let sample_count = file.read_u64::<LittleEndian>().unwrap();
        assert_eq!(sample_count, 0);

        let label_len = file.read_u16::<LittleEndian>().unwrap();
        assert_eq!(label_len, 6); // "vocals" = 6 bytes

        let mut label_buf = vec![0u8; label_len as usize];
        file.read_exact(&mut label_buf).unwrap();
        assert_eq!(std::str::from_utf8(&label_buf).unwrap(), "vocals");

        // Verify compressed audio data follows
        let mut audio_read = vec![0u8; 4];
        file.read_exact(&mut audio_read).unwrap();
        assert_eq!(audio_read, audio);
    }

    #[test]
    fn test_chunk_table_entries() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("table_test.neo");

        let mut writer = NeoWriter::new(48000, 2);
        let vocals = StemConfig::new(0, StemLabel::Vocals, CodecId::Flac, 2, 48000);
        let drums = StemConfig::new(1, StemLabel::Drums, CodecId::Flac, 2, 48000);
        writer.add_stem(vocals, vec![1; 128]).unwrap();
        writer.add_stem(drums, vec![2; 256]).unwrap();
        writer.set_metadata(r#"{"@context":"https://schema.org"}"#.to_string());
        writer.finalize(&path).unwrap();

        // Read back header to get chunk table location
        let mut file = std::fs::File::open(&path).unwrap();
        file.seek(SeekFrom::Start(27)).unwrap(); // chunk_table_offset field
        let cto = file.read_u64::<LittleEndian>().unwrap();
        let cc = file.read_u64::<LittleEndian>().unwrap();
        assert_eq!(cc, 3); // 2 stems + 1 meta

        // Seek to chunk table
        file.seek(SeekFrom::Start(cto)).unwrap();

        // Read first entry (STEM 0)
        let ct = file.read_u8().unwrap();
        assert_eq!(ct, ChunkType::Stem as u8);
        let offset = file.read_u64::<LittleEndian>().unwrap();
        assert_eq!(offset, HEADER_SIZE as u64);
        let _size = file.read_u64::<LittleEndian>().unwrap();
        let mut hash = [0u8; 32];
        file.read_exact(&mut hash).unwrap();
        let mut reserved = [0u8; 8];
        file.read_exact(&mut reserved).unwrap();
        assert_eq!(reserved, [0u8; 8]);

        // Read second entry (STEM 1)
        let ct2 = file.read_u8().unwrap();
        assert_eq!(ct2, ChunkType::Stem as u8);

        // Skip rest of entry 2: offset(8) + size(8) + hash(32) + reserved(8) = 56 bytes
        file.seek(SeekFrom::Current(56)).unwrap();
        let ct3 = file.read_u8().unwrap();
        assert_eq!(ct3, ChunkType::Metadata as u8);
    }

    #[test]
    fn test_blake3_hash_integrity() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("hash_test.neo");

        let audio_data = vec![42u8; 100];
        let config = StemConfig::new(0, StemLabel::Mix, CodecId::Pcm, 1, 16000);

        let mut writer = NeoWriter::new(16000, 1);
        writer.add_stem(config, audio_data).unwrap();
        writer.finalize(&path).unwrap();

        // Read the whole file
        let file_bytes = std::fs::read(&path).unwrap();

        // Parse header to get chunk table
        let cto = u64::from_le_bytes(file_bytes[27..35].try_into().unwrap()) as usize;
        let cc = u64::from_le_bytes(file_bytes[35..43].try_into().unwrap()) as usize;
        assert_eq!(cc, 1);

        // Parse chunk table entry
        let entry_start = cto;
        let chunk_offset = u64::from_le_bytes(
            file_bytes[entry_start + 1..entry_start + 9]
                .try_into()
                .unwrap(),
        ) as usize;
        let chunk_size = u64::from_le_bytes(
            file_bytes[entry_start + 9..entry_start + 17]
                .try_into()
                .unwrap(),
        ) as usize;
        let stored_hash: [u8; 32] = file_bytes[entry_start + 17..entry_start + 49]
            .try_into()
            .unwrap();

        // Recompute BLAKE3 hash from chunk data
        let chunk_data = &file_bytes[chunk_offset..chunk_offset + chunk_size];
        let computed_hash = blake3::hash(chunk_data);
        assert_eq!(stored_hash, *computed_hash.as_bytes());
    }

    #[test]
    fn test_metadata_chunks() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("meta_test.neo");

        let meta = r#"{"@context":"https://schema.org","name":"Test"}"#.to_string();
        let temporal = r#"{"bpm":120.0,"key":"Cm"}"#.to_string();
        let rights = r#"{"splits":[]}"#.to_string();

        let config = StemConfig::new(0, StemLabel::Mix, CodecId::Opus, 2, 48000);
        let mut writer = NeoWriter::new(48000, 1);
        writer.add_stem(config, vec![0; 32]).unwrap();
        writer.set_metadata(meta.clone());
        writer.set_temporal(temporal.clone());
        writer.set_rights(rights.clone());
        writer.finalize(&path).unwrap();

        // Read file and check chunk count
        let file_bytes = std::fs::read(&path).unwrap();
        let cc = u64::from_le_bytes(file_bytes[35..43].try_into().unwrap());
        assert_eq!(cc, 4); // 1 STEM + META + TEMP + RGHT

        // Verify feature flags include TEMPORAL_META and WEB3_RIGHTS
        let flags = u64::from_le_bytes(file_bytes[6..14].try_into().unwrap());
        assert!(flags & FeatureFlags::TEMPORAL_META != 0);
        assert!(flags & FeatureFlags::WEB3_RIGHTS != 0);
    }

    #[test]
    fn test_too_many_stems_rejected() {
        let mut writer = NeoWriter::new(44100, 8);
        for i in 0..8 {
            let config = StemConfig::new(
                i,
                StemLabel::Custom(format!("stem_{i}")),
                CodecId::Pcm,
                1,
                44100,
            );
            writer.add_stem(config, vec![0; 16]).unwrap();
        }
        // The 9th stem should fail
        let config = StemConfig::new(
            8,
            StemLabel::Custom("overflow".into()),
            CodecId::Pcm,
            1,
            44100,
        );
        let result = writer.add_stem(config, vec![0; 16]);
        assert!(result.is_err());
    }

    #[test]
    fn test_file_size_matches_expected() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("size_test.neo");

        let audio = vec![0xFF; 50];
        let meta = r#"{"test":true}"#.to_string();
        let config = StemConfig::new(0, StemLabel::Vocals, CodecId::Dac, 2, 44100);

        let mut writer = NeoWriter::new(44100, 1);
        writer.add_stem(config, audio.clone()).unwrap();
        writer.set_metadata(meta.clone());
        writer.finalize(&path).unwrap();

        let file_bytes = std::fs::read(&path).unwrap();

        // Expected: 64 (header) + stem_chunk_size + meta_chunk_size + 2 * 57 (chunk table)
        let label_bytes = "vocals".len();
        let stem_chunk_size = 22 + label_bytes + audio.len(); // stem header + audio
        let meta_chunk_size = meta.len();
        let chunk_table_size = 2 * CHUNK_ENTRY_SIZE;
        let expected_size = HEADER_SIZE + stem_chunk_size + meta_chunk_size + chunk_table_size;

        assert_eq!(file_bytes.len(), expected_size);
    }

    #[test]
    fn test_spatial_chunk_written() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("spatial_test.neo");

        let spatial = r#"{"objects":[{"id":"singer","position":[0,0,1]}]}"#.to_string();
        let config = StemConfig::new(0, StemLabel::Vocals, CodecId::Dac, 2, 44100);

        let mut writer = NeoWriter::new(44100, 1);
        writer.add_stem(config, vec![0; 32]).unwrap();
        writer.set_spatial(spatial.clone());
        writer.finalize(&path).unwrap();

        // Read file and check chunk count
        let file_bytes = std::fs::read(&path).unwrap();
        let cc = u64::from_le_bytes(file_bytes[35..43].try_into().unwrap());
        assert_eq!(cc, 2); // 1 STEM + SPAT

        // Verify SPATIAL feature flag is set
        let flags = u64::from_le_bytes(file_bytes[6..14].try_into().unwrap());
        assert!(flags & FeatureFlags::SPATIAL != 0);

        // Find the SPAT chunk in the chunk table
        let cto = u64::from_le_bytes(file_bytes[27..35].try_into().unwrap()) as usize;
        // Second entry (index 1) should be SPAT
        let spat_entry_start = cto + CHUNK_ENTRY_SIZE;
        let chunk_type_byte = file_bytes[spat_entry_start];
        assert_eq!(chunk_type_byte, ChunkType::Spatial as u8);
    }

    #[test]
    fn test_edit_history_chunk_written() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("edit_test.neo");

        let edit = r#"{"commits":[{"id":"abc123","ops":[]}]}"#.to_string();
        let config = StemConfig::new(0, StemLabel::Mix, CodecId::Flac, 2, 48000);

        let mut writer = NeoWriter::new(48000, 1);
        writer.add_stem(config, vec![0; 64]).unwrap();
        writer.set_edit_history(edit.clone());
        writer.finalize(&path).unwrap();

        // Read file and check chunk count
        let file_bytes = std::fs::read(&path).unwrap();
        let cc = u64::from_le_bytes(file_bytes[35..43].try_into().unwrap());
        assert_eq!(cc, 2); // 1 STEM + EDIT

        // Verify EDIT_HISTORY feature flag is set
        let flags = u64::from_le_bytes(file_bytes[6..14].try_into().unwrap());
        assert!(flags & FeatureFlags::EDIT_HISTORY != 0);

        // Find the EDIT chunk in the chunk table
        let cto = u64::from_le_bytes(file_bytes[27..35].try_into().unwrap()) as usize;
        // Second entry (index 1) should be EDIT
        let edit_entry_start = cto + CHUNK_ENTRY_SIZE;
        let chunk_type_byte = file_bytes[edit_entry_start];
        assert_eq!(chunk_type_byte, ChunkType::EditHistory as u8);
    }

    #[test]
    fn test_preview_chunk_first() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("preview_test.neo");

        let preview = vec![0xCC; 128]; // fake Opus preview data
        let config = StemConfig::new(0, StemLabel::Vocals, CodecId::Dac, 2, 44100);

        let mut writer = NeoWriter::new(44100, 1);
        writer.add_stem(config, vec![0; 32]).unwrap();
        writer.set_metadata(r#"{"test":true}"#.to_string());
        writer.set_preview(preview);
        writer.finalize(&path).unwrap();

        // Read file and check chunk count
        let file_bytes = std::fs::read(&path).unwrap();
        let cc = u64::from_le_bytes(file_bytes[35..43].try_into().unwrap());
        assert_eq!(cc, 3); // PREV + STEM + META

        // Verify PREVIEW_LOD feature flag is set
        let flags = u64::from_le_bytes(file_bytes[6..14].try_into().unwrap());
        assert!(flags & FeatureFlags::PREVIEW_LOD != 0);

        // The first chunk in the chunk table should be PREV
        let cto = u64::from_le_bytes(file_bytes[27..35].try_into().unwrap()) as usize;
        let first_chunk_type = file_bytes[cto];
        assert_eq!(first_chunk_type, ChunkType::Preview as u8);

        // The PREV chunk data should start immediately after the header
        let prev_offset = u64::from_le_bytes(file_bytes[cto + 1..cto + 9].try_into().unwrap());
        assert_eq!(prev_offset, HEADER_SIZE as u64);
    }

    #[test]
    fn test_full_chunk_ordering() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("ordering_test.neo");

        let mut writer = NeoWriter::new(44100, 1);
        let config = StemConfig::new(0, StemLabel::Vocals, CodecId::Dac, 2, 44100);
        writer.add_stem(config, vec![0; 32]).unwrap();

        // Set all optional chunks
        writer.set_preview(vec![0xAA; 16]);
        writer.set_metadata(r#"{"meta":true}"#.to_string());
        writer.set_temporal(r#"{"bpm":120}"#.to_string());
        writer.set_credentials(vec![0xBB; 24]);
        writer.set_spatial(r#"{"objects":[]}"#.to_string());
        writer.set_edit_history(r#"{"commits":[]}"#.to_string());
        writer.set_rights(r#"{"splits":[]}"#.to_string());
        writer.finalize(&path).unwrap();

        // Read file
        let file_bytes = std::fs::read(&path).unwrap();
        let cc = u64::from_le_bytes(file_bytes[35..43].try_into().unwrap()) as usize;
        // PREV + STEM + META + TEMP + CRED + SPAT + EDIT + RGHT = 8
        assert_eq!(cc, 8);

        // Read chunk types from the chunk table in order
        let cto = u64::from_le_bytes(file_bytes[27..35].try_into().unwrap()) as usize;
        let mut chunk_types = Vec::new();
        for i in 0..cc {
            let entry_start = cto + i * CHUNK_ENTRY_SIZE;
            chunk_types.push(file_bytes[entry_start]);
        }

        let expected_order = vec![
            ChunkType::Preview as u8,
            ChunkType::Stem as u8,
            ChunkType::Metadata as u8,
            ChunkType::Temporal as u8,
            ChunkType::Credentials as u8,
            ChunkType::Spatial as u8,
            ChunkType::EditHistory as u8,
            ChunkType::Rights as u8,
        ];
        assert_eq!(chunk_types, expected_order);
    }
}
