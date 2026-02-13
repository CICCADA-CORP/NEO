//! NEO file reader — deserializes `.neo` files for playback and inspection.
//!
//! The reader parses the 64-byte header, chunk table, and individual chunks
//! from a NEO container file. It validates magic bytes, format version,
//! BLAKE3 checksums, and structural integrity before exposing the data
//! through a safe API.
//!
//! # Example
//!
//! ```rust,no_run
//! use std::path::Path;
//! use neo_format::NeoReader;
//!
//! let reader = NeoReader::open(Path::new("track.neo")).unwrap();
//! println!("Stems: {}", reader.header().stem_count);
//! for stem in reader.stem_configs() {
//!     println!("  {} ({})", stem.label.as_str(), stem.stem_id);
//! }
//! ```

use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::Path;

use byteorder::{LittleEndian, ReadBytesExt};

use crate::chunk::{ChunkEntry, ChunkType, CHUNK_ENTRY_SIZE};
use crate::error::{NeoFormatError, Result};
use crate::header::{FeatureFlags, NeoHeader, MAX_STEMS, NEO_MAGIC};
use crate::stem::{CodecId, StemConfig, StemLabel};

/// Maximum number of chunks allowed per file (security limit).
const MAX_CHUNK_COUNT: u64 = 256;

/// Maximum size of a single chunk in bytes (2 GiB security limit).
const MAX_CHUNK_SIZE: u64 = 2 * 1024 * 1024 * 1024;

/// Default per-allocation memory limit (256 MiB).
/// Can be overridden via `NeoReader::set_allocation_limit`.
const DEFAULT_ALLOCATION_LIMIT: u64 = 256 * 1024 * 1024;

/// Size of the stem header before the variable-length label, in bytes.
///
/// Layout: stem_id(1) + codec_id(1) + channels(1) + sample_rate(4) +
///         bit_depth(1) + bitrate_kbps(4) + sample_count(8) + label_len(2) = 22
const STEM_HEADER_FIXED_SIZE: usize = 22;

/// Internal bookkeeping for a parsed stem: its config plus where the
/// audio data lives in the file.
#[derive(Debug, Clone)]
struct StemEntry {
    /// Parsed stem configuration.
    config: StemConfig,
    /// Byte offset of the compressed audio data within the file
    /// (immediately after the stem header inside the STEM chunk).
    data_offset: u64,
    /// Size of the compressed audio data in bytes.
    data_size: u64,
}

/// Reader for `.neo` files.
///
/// `NeoReader` opens a NEO container, validates the header and chunk table,
/// verifies BLAKE3 integrity checksums, and exposes accessor methods for
/// the header, chunks, stem configurations, metadata, and raw chunk data.
///
/// The reader keeps a [`BufReader`] handle to the underlying file so that
/// chunk data can be read lazily on demand.
pub struct NeoReader {
    /// The parsed 64-byte file header.
    header: NeoHeader,
    /// All recognised chunk entries from the chunk table.
    chunks: Vec<ChunkEntry>,
    /// Parsed stem entries with data offsets for lazy loading.
    stem_entries: Vec<StemEntry>,
    /// Cached stem configurations for the `stem_configs()` accessor.
    stem_configs_cache: Vec<StemConfig>,
    /// Buffered file handle for on-demand reads.
    inner: BufReader<File>,
    /// Total file size in bytes (used for bounds checking).
    file_size: u64,
    /// Per-allocation memory limit in bytes.
    allocation_limit: u64,
}

impl NeoReader {
    /// Open and fully parse a `.neo` file at the given path.
    ///
    /// This performs the following steps:
    /// 1. Reads and validates the 64-byte header (magic, version, stem count).
    /// 2. Seeks to the chunk table and parses all chunk entries.
    /// 3. Validates that every chunk offset + size falls within the file.
    /// 4. Verifies the BLAKE3 checksum of each chunk.
    /// 5. Parses stem headers from all `STEM` chunks.
    ///
    /// # Errors
    ///
    /// Returns [`NeoFormatError`] if the file is missing, corrupt, or does
    /// not conform to the NEO specification.
    pub fn open(path: &Path) -> Result<Self> {
        tracing::info!("Opening NEO file: {}", path.display());

        let file = File::open(path)?;
        let file_size = file.metadata()?.len();
        let mut reader = BufReader::new(file);

        // --- 1. Parse the 64-byte header ---
        let header = Self::read_header(&mut reader)?;
        tracing::info!(
            version = header.version,
            stems = header.stem_count,
            sample_rate = header.sample_rate,
            duration_us = header.duration_us,
            "Parsed NEO header"
        );

        // --- 2. Parse the chunk table ---
        let chunks = Self::read_chunk_table(&mut reader, &header, file_size)?;
        tracing::info!(count = chunks.len(), "Parsed chunk table");

        // --- 3. Verify BLAKE3 checksums ---
        Self::verify_checksums(&mut reader, &chunks)?;
        tracing::debug!("All chunk checksums verified");

        // --- 4. Parse stem chunks ---
        let stem_entries = Self::parse_stem_chunks(&mut reader, &chunks)?;
        let stem_configs_cache: Vec<StemConfig> =
            stem_entries.iter().map(|e| e.config.clone()).collect();
        tracing::info!(stems = stem_entries.len(), "Parsed stem configurations");

        Ok(Self {
            header,
            chunks,
            stem_entries,
            stem_configs_cache,
            inner: reader,
            file_size,
            allocation_limit: DEFAULT_ALLOCATION_LIMIT,
        })
    }

    /// Get the total file size in bytes.
    pub fn file_size(&self) -> u64 {
        self.file_size
    }

    /// Set the maximum size in bytes for a single memory allocation
    /// when reading chunk data. This protects against out-of-memory
    /// conditions caused by maliciously crafted files.
    ///
    /// The default limit is 256 MiB. Set to `u64::MAX` to disable.
    pub fn set_allocation_limit(&mut self, limit: u64) {
        self.allocation_limit = limit;
    }

    /// Get a reference to the parsed file header.
    pub fn header(&self) -> &NeoHeader {
        &self.header
    }

    /// Get a slice of all recognised chunk entries in the file.
    pub fn chunks(&self) -> &[ChunkEntry] {
        &self.chunks
    }

    /// Get a slice of all stem configurations found in the file.
    pub fn stem_configs(&self) -> &[StemConfig] {
        &self.stem_configs_cache
    }

    /// Read the raw compressed audio data for a specific stem.
    ///
    /// The returned bytes are the codec-compressed payload (e.g. DAC codes,
    /// FLAC stream, Opus packets) without the stem header.
    ///
    /// # Errors
    ///
    /// Returns [`NeoFormatError::StemNotFound`] if no stem with the given
    /// `stem_id` exists in the file.
    pub fn read_stem_data(&mut self, stem_id: u8) -> Result<Vec<u8>> {
        let entry = self
            .stem_entries
            .iter()
            .find(|e| e.config.stem_id == stem_id)
            .ok_or(NeoFormatError::StemNotFound(stem_id))?;

        let data_offset = entry.data_offset;
        let data_size = entry.data_size;

        tracing::debug!(
            stem_id,
            offset = data_offset,
            size = data_size,
            "Reading stem data"
        );

        self.inner.seek(SeekFrom::Start(data_offset))?;
        self.check_allocation(data_size)?;
        let mut buf = vec![0u8; data_size as usize];
        self.inner.read_exact(&mut buf)?;
        Ok(buf)
    }

    /// Read the JSON-LD metadata string from the META chunk, if present.
    ///
    /// Returns `Ok(None)` if the file does not contain a META chunk.
    pub fn read_metadata(&mut self) -> Result<Option<String>> {
        self.read_chunk_as_string(ChunkType::Metadata)
    }

    /// Read the temporal metadata (lyrics, chords, BPM) from the TEMP chunk,
    /// if present.
    ///
    /// Returns `Ok(None)` if the file does not contain a TEMP chunk.
    pub fn read_temporal(&mut self) -> Result<Option<String>> {
        self.read_chunk_as_string(ChunkType::Temporal)
    }

    /// Read the raw C2PA content credentials (JUMBF manifest bytes) from the
    /// CRED chunk, if present.
    ///
    /// Unlike the text-based metadata methods, this returns the raw binary
    /// data because CRED chunks contain JUMBF (JPEG Universal Metadata Box
    /// Format) bytes rather than UTF-8 text.
    ///
    /// Returns `Ok(None)` if the file does not contain a CRED chunk.
    pub fn read_credentials(&mut self) -> std::result::Result<Option<Vec<u8>>, NeoFormatError> {
        self.read_chunk_data(ChunkType::Credentials)
    }

    /// Read the Web3 rights/royalty metadata from the RGHT chunk, if present.
    ///
    /// The returned string contains JSON with split information, contract
    /// addresses, and license URIs (see SPECIFICATION.md Section 9.4).
    ///
    /// Returns `Ok(None)` if the file does not contain a RGHT chunk.
    pub fn read_rights(&mut self) -> std::result::Result<Option<String>, NeoFormatError> {
        self.read_chunk_as_string(ChunkType::Rights)
    }

    /// Read the spatial audio scene JSON from the SPAT chunk, if present.
    ///
    /// Returns `Ok(None)` if the file does not contain a SPAT chunk.
    pub fn read_spatial(&mut self) -> Result<Option<String>> {
        self.read_chunk_as_string(ChunkType::Spatial)
    }

    /// Read the non-destructive edit history JSON from the EDIT chunk, if present.
    ///
    /// Returns `Ok(None)` if the file does not contain an EDIT chunk.
    pub fn read_edit_history(&mut self) -> Result<Option<String>> {
        self.read_chunk_as_string(ChunkType::EditHistory)
    }

    /// Read the low-quality preview audio from the PREV chunk, if present.
    ///
    /// Returns the raw binary audio data (Opus-encoded), or `Ok(None)` if
    /// the file does not contain a PREV chunk.
    pub fn read_preview(&mut self) -> Result<Option<Vec<u8>>> {
        self.read_chunk_data(ChunkType::Preview)
    }

    /// Read the raw data for the first chunk matching `chunk_type`.
    ///
    /// Returns `Ok(None)` if no chunk of the given type exists.
    pub fn read_chunk_data(&mut self, chunk_type: ChunkType) -> Result<Option<Vec<u8>>> {
        let entry = match self.chunks.iter().find(|c| c.chunk_type == chunk_type) {
            Some(e) => e.clone(),
            None => return Ok(None),
        };

        tracing::debug!(
            chunk_type = ?entry.chunk_type,
            offset = entry.offset,
            size = entry.size,
            "Reading chunk data"
        );

        self.inner.seek(SeekFrom::Start(entry.offset))?;
        self.check_allocation(entry.size)?;
        let mut buf = vec![0u8; entry.size as usize];
        self.inner.read_exact(&mut buf)?;
        Ok(Some(buf))
    }

    // ---------------------------------------------------------------
    // Private helpers
    // ---------------------------------------------------------------

    /// Check that a requested allocation does not exceed the configured limit.
    fn check_allocation(&self, size: u64) -> Result<()> {
        if size > self.allocation_limit {
            return Err(NeoFormatError::AllocationTooLarge {
                requested: size,
                limit: self.allocation_limit,
            });
        }
        Ok(())
    }

    /// Read and validate the 64-byte header from the current reader position.
    fn read_header(reader: &mut BufReader<File>) -> Result<NeoHeader> {
        // Magic bytes (4 bytes)
        let mut magic = [0u8; 4];
        reader.read_exact(&mut magic)?;
        if magic != NEO_MAGIC {
            return Err(NeoFormatError::InvalidMagic);
        }

        // Version (u16 LE)
        let version = reader.read_u16::<LittleEndian>()?;
        if version > 1 {
            return Err(NeoFormatError::UnsupportedVersion(version));
        }

        // Feature flags (u64 LE)
        let flags_raw = reader.read_u64::<LittleEndian>()?;
        let feature_flags = FeatureFlags(flags_raw);

        // Stem count (u8)
        let stem_count = reader.read_u8()?;
        if stem_count == 0 || stem_count > MAX_STEMS {
            return Err(NeoFormatError::TooManyStems {
                max: MAX_STEMS,
                got: stem_count,
            });
        }

        // Sample rate (u32 LE)
        let sample_rate = reader.read_u32::<LittleEndian>()?;

        // Duration in microseconds (u64 LE)
        let duration_us = reader.read_u64::<LittleEndian>()?;

        // Chunk table offset (u64 LE)
        let chunk_table_offset = reader.read_u64::<LittleEndian>()?;

        // Chunk count (u64 LE)
        let chunk_count = reader.read_u64::<LittleEndian>()?;

        // Skip 21 reserved bytes
        let mut reserved = [0u8; 21];
        reader.read_exact(&mut reserved)?;

        Ok(NeoHeader {
            version,
            feature_flags,
            stem_count,
            sample_rate,
            duration_us,
            chunk_table_offset,
            chunk_count,
        })
    }

    /// Seek to the chunk table and parse all entries.
    ///
    /// Unknown chunk types are logged and skipped gracefully per the spec.
    fn read_chunk_table(
        reader: &mut BufReader<File>,
        header: &NeoHeader,
        file_size: u64,
    ) -> Result<Vec<ChunkEntry>> {
        // Validate chunk_table_offset is within the file
        let table_start = header.chunk_table_offset;
        let table_size = header.chunk_count * CHUNK_ENTRY_SIZE as u64;
        if table_start
            .checked_add(table_size)
            .is_none_or(|end| end > file_size)
        {
            return Err(NeoFormatError::InvalidOffset {
                offset: table_start,
                file_size,
            });
        }

        // Enforce maximum chunk count (security)
        if header.chunk_count > MAX_CHUNK_COUNT {
            return Err(NeoFormatError::ChunkCountExceeded {
                max: MAX_CHUNK_COUNT,
                got: header.chunk_count,
            });
        }

        reader.seek(SeekFrom::Start(table_start))?;

        let mut chunks = Vec::with_capacity(header.chunk_count as usize);

        for i in 0..header.chunk_count {
            // chunk_type (u8)
            let type_byte = reader.read_u8()?;

            // offset (u64 LE)
            let offset = reader.read_u64::<LittleEndian>()?;

            // size (u64 LE)
            let size = reader.read_u64::<LittleEndian>()?;

            // blake3_hash ([u8; 32])
            let mut blake3_hash = [0u8; 32];
            reader.read_exact(&mut blake3_hash)?;

            // reserved ([u8; 8])
            let mut _reserved = [0u8; 8];
            reader.read_exact(&mut _reserved)?;

            // Resolve chunk type — skip unknowns gracefully
            let chunk_type = match ChunkType::from_u8(type_byte) {
                Some(ct) => ct,
                None => {
                    tracing::warn!(
                        chunk_index = i,
                        type_byte = format!("0x{:02X}", type_byte),
                        "Skipping unknown chunk type"
                    );
                    continue;
                }
            };

            // Validate offset + size within file bounds
            if offset.checked_add(size).is_none_or(|end| end > file_size) {
                return Err(NeoFormatError::InvalidOffset { offset, file_size });
            }

            // Enforce per-chunk size limit (2 GiB)
            if size > MAX_CHUNK_SIZE {
                return Err(NeoFormatError::InvalidOffset { offset, file_size });
            }

            tracing::debug!(
                index = i,
                chunk_type = ?chunk_type,
                offset,
                size,
                "Parsed chunk entry"
            );

            chunks.push(ChunkEntry {
                chunk_type,
                offset,
                size,
                blake3_hash,
            });
        }

        Ok(chunks)
    }

    /// Verify the BLAKE3 checksum for every chunk in the table.
    fn verify_checksums(reader: &mut BufReader<File>, chunks: &[ChunkEntry]) -> Result<()> {
        for entry in chunks {
            reader.seek(SeekFrom::Start(entry.offset))?;

            // Read chunk data in 64 KiB blocks to avoid huge allocations
            // for the hash computation.
            let mut hasher = blake3::Hasher::new();
            let mut remaining = entry.size;
            let mut buf = vec![0u8; 64 * 1024];

            while remaining > 0 {
                let to_read = std::cmp::min(remaining, buf.len() as u64) as usize;
                reader.read_exact(&mut buf[..to_read])?;
                hasher.update(&buf[..to_read]);
                remaining -= to_read as u64;
            }

            let computed = hasher.finalize();
            if computed.as_bytes() != &entry.blake3_hash {
                return Err(NeoFormatError::ChecksumMismatch {
                    expected: hex_encode(&entry.blake3_hash),
                    actual: hex_encode(computed.as_bytes()),
                });
            }

            tracing::debug!(
                chunk_type = ?entry.chunk_type,
                "Checksum verified"
            );
        }
        Ok(())
    }

    /// Parse the stem header from every STEM chunk and collect results.
    fn parse_stem_chunks(
        reader: &mut BufReader<File>,
        chunks: &[ChunkEntry],
    ) -> Result<Vec<StemEntry>> {
        let mut stems = Vec::new();

        for entry in chunks.iter().filter(|c| c.chunk_type == ChunkType::Stem) {
            reader.seek(SeekFrom::Start(entry.offset))?;

            // Ensure the chunk is large enough for the fixed stem header
            if entry.size < STEM_HEADER_FIXED_SIZE as u64 {
                return Err(NeoFormatError::Io(std::io::Error::new(
                    std::io::ErrorKind::UnexpectedEof,
                    format!(
                        "STEM chunk too small: {} bytes (need at least {})",
                        entry.size, STEM_HEADER_FIXED_SIZE
                    ),
                )));
            }

            // stem_id (u8)
            let stem_id = reader.read_u8()?;
            if stem_id > 7 {
                return Err(NeoFormatError::InvalidStemId { id: stem_id });
            }

            // codec_id (u8)
            let codec_byte = reader.read_u8()?;
            let codec =
                CodecId::from_u8(codec_byte).ok_or(NeoFormatError::InvalidCodecId(codec_byte))?;

            // channels (u8)
            let channels = reader.read_u8()?;

            // sample_rate (u32 LE)
            let sample_rate = reader.read_u32::<LittleEndian>()?;

            // bit_depth (u8)
            let bit_depth = reader.read_u8()?;

            // bitrate_kbps (u32 LE)
            let bitrate_kbps = reader.read_u32::<LittleEndian>()?;

            // sample_count (u64 LE)
            let sample_count = reader.read_u64::<LittleEndian>()?;

            // label_len (u16 LE)
            let label_len = reader.read_u16::<LittleEndian>()?;

            // Ensure the chunk can hold the label
            let header_total = STEM_HEADER_FIXED_SIZE as u64 + label_len as u64;
            if entry.size < header_total {
                return Err(NeoFormatError::Io(std::io::Error::new(
                    std::io::ErrorKind::UnexpectedEof,
                    format!(
                        "STEM chunk too small for label: need {} bytes, have {}",
                        header_total, entry.size
                    ),
                )));
            }

            // label (UTF-8 string)
            let mut label_buf = vec![0u8; label_len as usize];
            reader.read_exact(&mut label_buf)?;
            let label_string = String::from_utf8(label_buf)?;
            let label = label_string.parse::<StemLabel>().map_err(|e| {
                NeoFormatError::Io(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("Invalid stem label '{}': {}", label_string, e),
                ))
            })?;

            // Compute the offset and size of the audio data that follows
            // the stem header inside this chunk.
            let data_offset = entry.offset + header_total;
            let data_size = entry.size - header_total;

            tracing::debug!(
                stem_id,
                label = label.as_str(),
                codec = ?codec,
                channels,
                sample_rate,
                data_size,
                "Parsed stem"
            );

            stems.push(StemEntry {
                config: StemConfig {
                    stem_id,
                    label,
                    codec,
                    channels,
                    sample_rate,
                    bit_depth,
                    bitrate_kbps,
                    sample_count,
                },
                data_offset,
                data_size,
            });
        }

        Ok(stems)
    }

    /// Read a chunk's raw data and interpret it as a UTF-8 string.
    ///
    /// Returns `Ok(None)` if no chunk of the given type exists.
    fn read_chunk_as_string(&mut self, chunk_type: ChunkType) -> Result<Option<String>> {
        match self.read_chunk_data(chunk_type)? {
            Some(data) => {
                let s = String::from_utf8(data).map_err(NeoFormatError::InvalidUtf8)?;
                Ok(Some(s))
            }
            None => Ok(None),
        }
    }
}

/// Encode a byte slice as a lowercase hex string.
fn hex_encode(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{:02x}", b)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chunk::ChunkType;
    use crate::header::{FeatureFlags, HEADER_SIZE, NEO_MAGIC};
    use crate::stem::CodecId;
    use byteorder::{LittleEndian, WriteBytesExt};
    use std::io::Write;

    /// Helper: build a minimal valid NEO file in memory and return the bytes.
    fn build_test_neo_file() -> Vec<u8> {
        let mut buf: Vec<u8> = Vec::new();

        // --- Header (64 bytes) ---
        buf.write_all(&NEO_MAGIC).unwrap(); // magic
        buf.write_u16::<LittleEndian>(1).unwrap(); // version
        buf.write_u64::<LittleEndian>(FeatureFlags::NEURAL_CODEC)
            .unwrap(); // feature_flags
        buf.write_u8(1).unwrap(); // stem_count
        buf.write_u32::<LittleEndian>(44100).unwrap(); // sample_rate
        buf.write_u64::<LittleEndian>(5_000_000).unwrap(); // duration_us (5 sec)
                                                           // chunk_table_offset — we'll patch this after writing chunk data
        let cto_pos = buf.len();
        buf.write_u64::<LittleEndian>(0).unwrap(); // placeholder
        buf.write_u64::<LittleEndian>(1).unwrap(); // chunk_count = 1
        buf.write_all(&[0u8; 21]).unwrap(); // reserved
        assert_eq!(buf.len(), HEADER_SIZE);

        // --- STEM chunk data ---
        let stem_data_offset = buf.len() as u64;
        let mut stem_chunk: Vec<u8> = Vec::new();
        // Stem header
        stem_chunk.write_u8(0).unwrap(); // stem_id
        stem_chunk.write_u8(CodecId::Dac as u8).unwrap(); // codec_id
        stem_chunk.write_u8(2).unwrap(); // channels
        stem_chunk.write_u32::<LittleEndian>(44100).unwrap(); // sample_rate
        stem_chunk.write_u8(16).unwrap(); // bit_depth
        stem_chunk.write_u32::<LittleEndian>(128).unwrap(); // bitrate_kbps
        stem_chunk.write_u64::<LittleEndian>(220500).unwrap(); // sample_count
        let label = b"vocals";
        stem_chunk
            .write_u16::<LittleEndian>(label.len() as u16)
            .unwrap(); // label_len
        stem_chunk.write_all(label).unwrap(); // label
                                              // Some fake audio data
        let audio_payload = vec![0xAB; 64];
        stem_chunk.write_all(&audio_payload).unwrap();

        let stem_chunk_size = stem_chunk.len() as u64;
        let stem_hash = blake3::hash(&stem_chunk);

        buf.write_all(&stem_chunk).unwrap();

        // --- Chunk table ---
        let chunk_table_offset = buf.len() as u64;
        // Patch the header's chunk_table_offset
        let cto_bytes = chunk_table_offset.to_le_bytes();
        buf[cto_pos..cto_pos + 8].copy_from_slice(&cto_bytes);

        // Write one chunk entry (57 bytes)
        buf.write_u8(ChunkType::Stem as u8).unwrap(); // chunk_type
        buf.write_u64::<LittleEndian>(stem_data_offset).unwrap(); // offset
        buf.write_u64::<LittleEndian>(stem_chunk_size).unwrap(); // size
        buf.write_all(stem_hash.as_bytes()).unwrap(); // blake3_hash
        buf.write_all(&[0u8; 8]).unwrap(); // reserved

        buf
    }

    /// Helper: write bytes to a temp file and return the path.
    fn write_temp_file(data: &[u8]) -> tempfile::NamedTempFile {
        let mut tmp = tempfile::NamedTempFile::new().unwrap();
        tmp.write_all(data).unwrap();
        tmp.flush().unwrap();
        tmp
    }

    #[test]
    fn test_open_valid_file() {
        let data = build_test_neo_file();
        let tmp = write_temp_file(&data);

        let reader = NeoReader::open(tmp.path()).expect("should open successfully");

        assert_eq!(reader.header().version, 1);
        assert_eq!(reader.header().stem_count, 1);
        assert_eq!(reader.header().sample_rate, 44100);
        assert_eq!(reader.header().duration_us, 5_000_000);
        assert!(reader
            .header()
            .feature_flags
            .has(FeatureFlags::NEURAL_CODEC));
        assert_eq!(reader.chunks().len(), 1);
        assert_eq!(reader.chunks()[0].chunk_type, ChunkType::Stem);

        let stems = reader.stem_configs();
        assert_eq!(stems.len(), 1);
        assert_eq!(stems[0].stem_id, 0);
        assert_eq!(stems[0].codec, CodecId::Dac);
        assert_eq!(stems[0].channels, 2);
        assert_eq!(stems[0].label.as_str(), "vocals");
        assert_eq!(stems[0].sample_count, 220500);
    }

    #[test]
    fn test_read_stem_data() {
        let data = build_test_neo_file();
        let tmp = write_temp_file(&data);

        let mut reader = NeoReader::open(tmp.path()).unwrap();
        let stem_data = reader.read_stem_data(0).unwrap();
        // Our test payload was 64 bytes of 0xAB
        assert_eq!(stem_data.len(), 64);
        assert!(stem_data.iter().all(|&b| b == 0xAB));
    }

    #[test]
    fn test_stem_not_found() {
        let data = build_test_neo_file();
        let tmp = write_temp_file(&data);

        let mut reader = NeoReader::open(tmp.path()).unwrap();
        let result = reader.read_stem_data(7);
        assert!(matches!(result, Err(NeoFormatError::StemNotFound(7))));
    }

    #[test]
    fn test_invalid_magic() {
        let mut data = build_test_neo_file();
        data[0] = 0x00; // corrupt magic
        let tmp = write_temp_file(&data);

        let result = NeoReader::open(tmp.path());
        assert!(matches!(result, Err(NeoFormatError::InvalidMagic)));
    }

    #[test]
    fn test_unsupported_version() {
        let mut data = build_test_neo_file();
        // version is at offset 4-5 (u16 LE); set to 99
        data[4] = 99;
        data[5] = 0;
        let tmp = write_temp_file(&data);

        let result = NeoReader::open(tmp.path());
        assert!(matches!(
            result,
            Err(NeoFormatError::UnsupportedVersion(99))
        ));
    }

    #[test]
    fn test_too_many_stems() {
        let mut data = build_test_neo_file();
        // stem_count is at offset 14; set to 10 (>8)
        data[14] = 10;
        let tmp = write_temp_file(&data);

        let result = NeoReader::open(tmp.path());
        assert!(matches!(
            result,
            Err(NeoFormatError::TooManyStems { max: 8, got: 10 })
        ));
    }

    #[test]
    fn test_zero_stems_rejected() {
        let mut data = build_test_neo_file();
        data[14] = 0;
        let tmp = write_temp_file(&data);

        let result = NeoReader::open(tmp.path());
        assert!(matches!(
            result,
            Err(NeoFormatError::TooManyStems { max: 8, got: 0 })
        ));
    }

    #[test]
    fn test_checksum_mismatch() {
        let mut data = build_test_neo_file();
        // Corrupt a byte in the stem chunk data (right after header)
        data[HEADER_SIZE] ^= 0xFF;
        let tmp = write_temp_file(&data);

        let result = NeoReader::open(tmp.path());
        assert!(matches!(
            result,
            Err(NeoFormatError::ChecksumMismatch { .. })
        ));
    }

    #[test]
    fn test_metadata_returns_none_when_absent() {
        let data = build_test_neo_file();
        let tmp = write_temp_file(&data);

        let mut reader = NeoReader::open(tmp.path()).unwrap();
        assert_eq!(reader.read_metadata().unwrap(), None);
    }

    #[test]
    fn test_temporal_returns_none_when_absent() {
        let data = build_test_neo_file();
        let tmp = write_temp_file(&data);

        let mut reader = NeoReader::open(tmp.path()).unwrap();
        assert_eq!(reader.read_temporal().unwrap(), None);
    }

    #[test]
    fn test_spatial_returns_none_when_absent() {
        let data = build_test_neo_file();
        let tmp = write_temp_file(&data);

        let mut reader = NeoReader::open(tmp.path()).unwrap();
        assert_eq!(reader.read_spatial().unwrap(), None);
    }

    #[test]
    fn test_edit_history_returns_none_when_absent() {
        let data = build_test_neo_file();
        let tmp = write_temp_file(&data);

        let mut reader = NeoReader::open(tmp.path()).unwrap();
        assert_eq!(reader.read_edit_history().unwrap(), None);
    }
}
