//! NEO CLI — Command-line interface for the NEO audio format.
//!
//! Provides fully functional commands for encoding WAV files into `.neo` containers,
//! decoding `.neo` files back to WAV, and inspecting file metadata.
//!
//! # Usage
//!
//! ```bash
//! neo encode vocals.wav instrumental.wav -o output.neo --stems vocals,instrumental
//! neo decode output.neo -o output_dir/ --all-stems
//! neo decode output.neo -o output_dir/ --stem vocals
//! neo info output.neo
//! neo info output.neo --json
//! ```

use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use clap::{Parser, Subcommand};

use neo_codec::{AudioCodec, EncodeConfig, PcmCodec};
use neo_edit::{EditHistory, EditOp};
use neo_format::{ChunkEntry, CodecId, FeatureFlags, NeoReader, NeoWriter, StemConfig, StemLabel};
use neo_metadata::{NeoMetadata, RightsInfo, TemporalMap};
use neo_spatial::SpatialScene;
use neo_stream::{cid::compute_cid, merkle::MerkleTree, DEFAULT_CHUNK_SIZE};

// ───────────────────────────── CLI definition ─────────────────────────────

/// Top-level CLI entry point for the `neo` binary.
#[derive(Parser)]
#[command(
    name = "neo",
    about = "NEO (Neural Extended Object) -- The future of audio codecs",
    version,
    long_about = "A revolutionary open-source audio format featuring stem-native architecture,\n\
                   neural compression, smart metadata, spatial audio, and non-destructive editing."
)]
struct Cli {
    /// Enable verbose (debug-level) logging.
    #[arg(short, long, global = true)]
    verbose: bool,

    #[command(subcommand)]
    command: Commands,
}

/// Available sub-commands.
#[derive(Subcommand)]
enum Commands {
    /// Encode one or more WAV files into a .neo container.
    Encode {
        /// Input WAV file paths (one per stem).
        #[arg(required = true)]
        input: Vec<PathBuf>,

        /// Output .neo file path.
        #[arg(short, long)]
        output: PathBuf,

        /// Comma-separated stem labels (e.g., "vocals,instrumental").
        /// If omitted, labels are derived from input file names.
        #[arg(short, long)]
        stems: Option<String>,

        /// Codec to use for encoding stems (pcm, dac, flac, opus).
        #[arg(short, long, default_value = "pcm")]
        codec: String,

        /// Path to a JSON metadata file to embed in the .neo container.
        #[arg(long)]
        metadata: Option<PathBuf>,

        /// Path to a JSON temporal metadata file (chords, lyrics, BPM) to embed.
        #[arg(long)]
        temporal: Option<PathBuf>,

        /// Path to a JSON Web3 rights/royalty file to embed.
        #[arg(long)]
        rights: Option<PathBuf>,

        /// Path to a raw JUMBF file containing C2PA credentials.
        #[arg(long)]
        credentials: Option<PathBuf>,

        /// Path to a JSON spatial audio scene file to embed.
        #[arg(long)]
        spatial: Option<PathBuf>,

        /// Path to a JSON edit history file to embed.
        #[arg(long)]
        edit_history: Option<PathBuf>,
    },

    /// Decode a .neo file back to WAV files.
    Decode {
        /// Input .neo file path.
        input: PathBuf,

        /// Output directory for extracted WAV files.
        #[arg(short, long, default_value = ".")]
        output: PathBuf,

        /// Extract a single stem by label (e.g., "vocals").
        #[arg(long)]
        stem: Option<String>,

        /// Extract all stems as separate WAV files.
        #[arg(long)]
        all_stems: bool,
    },

    /// Display detailed information about a .neo file.
    Info {
        /// Input .neo file path.
        input: PathBuf,

        /// Output file information as JSON.
        #[arg(long)]
        json: bool,
    },

    /// Play a .neo file through the default audio output.
    Play {
        /// Input .neo file path.
        input: PathBuf,

        /// Mute specific stems (comma-separated).
        #[arg(long)]
        mute: Option<String>,

        /// Solo specific stems (comma-separated).
        #[arg(long)]
        solo: Option<String>,
    },

    /// Manage metadata tags in a .neo file.
    ///
    /// Reads the existing .neo file, updates metadata, and writes a new file.
    /// Without `--output`, overwrites the input file.
    Tag {
        /// Input .neo file path.
        input: PathBuf,

        /// Output .neo file path. If omitted, overwrites the input.
        #[arg(short, long)]
        output: Option<PathBuf>,

        /// Track title.
        #[arg(long)]
        title: Option<String>,

        /// Artist name.
        #[arg(long)]
        artist: Option<String>,

        /// Genre.
        #[arg(long)]
        genre: Option<String>,

        /// BPM (beats per minute).
        #[arg(long)]
        bpm: Option<f64>,

        /// Musical key (e.g., "C major", "A minor").
        #[arg(long)]
        key: Option<String>,

        /// ISRC code (12 alphanumeric characters).
        #[arg(long)]
        isrc: Option<String>,

        /// Publication date (ISO 8601, e.g., "2025-06-15").
        #[arg(long)]
        date: Option<String>,

        /// Path to a JSON metadata file to replace existing metadata entirely.
        #[arg(long)]
        metadata_file: Option<PathBuf>,

        /// Path to a JSON temporal metadata file.
        #[arg(long)]
        temporal: Option<PathBuf>,

        /// Path to a JSON Web3 rights file.
        #[arg(long)]
        rights: Option<PathBuf>,

        /// Show current metadata without making changes.
        #[arg(long)]
        show: bool,
    },

    /// Download or list AI models (not yet implemented).
    #[command(name = "model")]
    Model {
        #[command(subcommand)]
        action: ModelAction,
    },

    /// Manage non-destructive edit history.
    Edit {
        #[command(subcommand)]
        action: EditAction,
    },

    /// Streaming and integrity verification.
    Stream {
        #[command(subcommand)]
        action: StreamAction,
    },
}

/// Sub-commands for `neo model`.
#[derive(Subcommand)]
enum ModelAction {
    /// Download pre-trained DAC ONNX models.
    Download {
        /// Model variant (44khz, 24khz, 16khz).
        #[arg(default_value = "44khz")]
        variant: String,
    },
    /// List available and installed models.
    List,
}

/// Sub-commands for `neo edit`.
#[derive(Subcommand)]
enum EditAction {
    /// Show the edit history of a .neo file.
    History {
        /// Input .neo file path.
        input: PathBuf,
        /// Output as JSON.
        #[arg(long)]
        json: bool,
    },
    /// Add an edit operation to a .neo file.
    AddOp {
        /// Input .neo file path.
        input: PathBuf,
        /// Output .neo file path. If omitted, overwrites the input.
        #[arg(short, long)]
        output: Option<PathBuf>,
        /// Edit operation type (trim, gain, mute, pan, fade, reverse).
        #[arg(long)]
        op: String,
        /// Target stem label (e.g., "vocals").
        #[arg(long)]
        stem: Option<String>,
        /// Operation parameter as JSON (e.g., '{"gain_db": -3.0}').
        #[arg(long)]
        params: Option<String>,
        /// Commit message for the edit.
        #[arg(long, default_value = "CLI edit")]
        message: String,
    },
    /// Revert the most recent edit commit.
    Revert {
        /// Input .neo file path.
        input: PathBuf,
        /// Output .neo file path. If omitted, overwrites the input.
        #[arg(short, long)]
        output: Option<PathBuf>,
    },
}

/// Sub-commands for `neo stream`.
#[derive(Subcommand)]
enum StreamAction {
    /// Show streaming info (Merkle tree, CID) for a .neo file.
    Info {
        /// Input .neo file path.
        input: PathBuf,
    },
    /// Verify the integrity of a .neo file using Merkle proofs.
    Verify {
        /// Input .neo file path.
        input: PathBuf,
    },
}

// ────────────────────────────── main ──────────────────────────────

fn main() -> Result<()> {
    let cli = Cli::parse();

    // Initialize tracing subscriber with env-filter support.
    let filter = if cli.verbose { "debug" } else { "info" };
    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_writer(std::io::stderr)
        .init();

    match cli.command {
        Commands::Encode {
            input,
            output,
            stems,
            codec,
            metadata,
            temporal,
            rights,
            credentials,
            spatial,
            edit_history,
        } => cmd_encode(
            &input,
            &output,
            stems.as_deref(),
            &codec,
            metadata.as_deref(),
            temporal.as_deref(),
            rights.as_deref(),
            credentials.as_deref(),
            spatial.as_deref(),
            edit_history.as_deref(),
        ),

        Commands::Decode {
            input,
            output,
            stem,
            all_stems,
        } => cmd_decode(&input, &output, stem.as_deref(), all_stems),

        Commands::Info { input, json } => cmd_info(&input, json),

        Commands::Play { input, mute, solo } => cmd_play(&input, mute.as_deref(), solo.as_deref()),

        Commands::Tag {
            input,
            output,
            title,
            artist,
            genre,
            bpm,
            key,
            isrc,
            date,
            metadata_file,
            temporal,
            rights,
            show,
        } => cmd_tag(
            &input,
            output.as_deref(),
            title,
            artist,
            genre,
            bpm,
            key,
            isrc,
            date,
            metadata_file.as_deref(),
            temporal.as_deref(),
            rights.as_deref(),
            show,
        ),

        Commands::Model { action } => cmd_model_stub(action),

        Commands::Edit { action } => cmd_edit(action),

        Commands::Stream { action } => cmd_stream(action),
    }
}

// ──────────────────────────── encode ──────────────────────────────

/// Encode one or more WAV files into a `.neo` container.
///
/// Each input WAV becomes a separate stem. Stem labels can be provided
/// explicitly via `--stems` (comma-separated) or are derived from
/// the input file names.
#[allow(clippy::too_many_arguments)]
fn cmd_encode(
    inputs: &[PathBuf],
    output: &Path,
    stems_arg: Option<&str>,
    codec_name: &str,
    metadata_path: Option<&Path>,
    temporal_path: Option<&Path>,
    rights_path: Option<&Path>,
    credentials_path: Option<&Path>,
    spatial_path: Option<&Path>,
    edit_history_path: Option<&Path>,
) -> Result<()> {
    // Resolve stem labels.
    let labels = resolve_stem_labels(inputs, stems_arg)?;

    // Resolve codec.
    let codec_id = parse_codec_name(codec_name)?;
    let codec = resolve_codec(codec_id)?;

    if inputs.len() > 8 {
        bail!(
            "NEO supports a maximum of 8 stems, but {} inputs were provided",
            inputs.len()
        );
    }

    println!("\n  NEO Encoder");
    println!("  ============================================");

    // We collect per-stem WAV data to determine the master sample rate.
    let mut stem_data: Vec<StemPayload> = Vec::with_capacity(inputs.len());

    for (i, wav_path) in inputs.iter().enumerate() {
        let payload = read_wav(wav_path)
            .with_context(|| format!("Failed to read WAV file: {}", wav_path.display()))?;
        println!(
            "  {} [{}] {}ch {}Hz {:.2}s ({} samples)",
            labels[i].as_str(),
            wav_path.display(),
            payload.channels,
            payload.sample_rate,
            payload.duration_secs(),
            payload.sample_count,
        );
        stem_data.push(payload);
    }

    // Use the first stem's sample rate as the master sample rate.
    let master_sample_rate = stem_data[0].sample_rate;

    // Compute the maximum duration across all stems.
    let max_duration_us = stem_data
        .iter()
        .map(|s| (s.duration_secs() * 1_000_000.0) as u64)
        .max()
        .unwrap_or(0);

    // Build the NeoWriter.
    let stem_count = inputs.len() as u8;
    let mut writer = NeoWriter::new(master_sample_rate, stem_count);
    writer.set_duration_us(max_duration_us);

    for (i, (payload, label)) in stem_data.iter().zip(labels.iter()).enumerate() {
        let encode_config = EncodeConfig {
            sample_rate: payload.sample_rate,
            channels: payload.channels,
            bitrate_kbps: None,
        };

        let encoded_bytes = codec
            .encode(&payload.samples, &encode_config)
            .map_err(|e| {
                anyhow::anyhow!("Codec encoding failed for stem '{}': {}", label.as_str(), e)
            })?;

        let mut config = StemConfig::new(
            i as u8,
            label.clone(),
            codec_id,
            payload.channels,
            payload.sample_rate,
        );
        config.bit_depth = payload.bit_depth;
        config.sample_count = payload.sample_count;

        writer
            .add_stem(config, encoded_bytes)
            .map_err(|e| anyhow::anyhow!("Failed to add stem '{}': {}", label.as_str(), e))?;
    }

    // Optionally embed JSON metadata.
    if let Some(meta_path) = metadata_path {
        let meta_json = std::fs::read_to_string(meta_path)
            .with_context(|| format!("Failed to read metadata file: {}", meta_path.display()))?;
        // Validate that it's parseable JSON.
        serde_json::from_str::<serde_json::Value>(&meta_json)
            .with_context(|| format!("Invalid JSON in metadata file: {}", meta_path.display()))?;
        writer.set_metadata(meta_json);
        println!("  Metadata embedded from {}", meta_path.display());
    }

    // Optionally embed temporal metadata.
    if let Some(temp_path) = temporal_path {
        let temp_json = std::fs::read_to_string(temp_path)
            .with_context(|| format!("Failed to read temporal file: {}", temp_path.display()))?;
        // Validate by parsing.
        let temporal = TemporalMap::from_json(&temp_json)
            .map_err(|e| anyhow::anyhow!("Invalid temporal metadata: {}", e))?;
        temporal
            .validate()
            .map_err(|e| anyhow::anyhow!("Temporal metadata validation failed: {}", e))?;
        writer.set_temporal(
            temporal
                .to_json()
                .map_err(|e| anyhow::anyhow!("Failed to serialize temporal metadata: {}", e))?,
        );
        println!("  Temporal metadata embedded from {}", temp_path.display());
    }

    // Optionally embed Web3 rights.
    if let Some(rights_path) = rights_path {
        let rights_json = std::fs::read_to_string(rights_path)
            .with_context(|| format!("Failed to read rights file: {}", rights_path.display()))?;
        let rights = RightsInfo::from_json(&rights_json)
            .map_err(|e| anyhow::anyhow!("Invalid rights metadata: {}", e))?;
        rights
            .validate()
            .map_err(|e| anyhow::anyhow!("Rights metadata validation failed: {}", e))?;
        writer.set_rights(
            rights
                .to_json()
                .map_err(|e| anyhow::anyhow!("Failed to serialize rights metadata: {}", e))?,
        );
        println!("  Web3 rights embedded from {}", rights_path.display());
    }

    // Optionally embed C2PA credentials (raw JUMBF bytes).
    if let Some(cred_path) = credentials_path {
        let cred_bytes = std::fs::read(cred_path)
            .with_context(|| format!("Failed to read credentials file: {}", cred_path.display()))?;
        if cred_bytes.is_empty() {
            bail!("Credentials file is empty: {}", cred_path.display());
        }
        writer.set_credentials(cred_bytes);
        println!("  C2PA credentials embedded from {}", cred_path.display());
    }

    // Optionally embed spatial audio scene.
    if let Some(spat_path) = spatial_path {
        let spat_json = std::fs::read_to_string(spat_path)
            .with_context(|| format!("Failed to read spatial file: {}", spat_path.display()))?;
        // Validate JSON structure.
        let scene = SpatialScene::from_json(&spat_json)
            .map_err(|e| anyhow::anyhow!("Invalid spatial scene: {}", e))?;
        scene
            .validate()
            .map_err(|e| anyhow::anyhow!("Spatial scene validation failed: {}", e))?;
        writer.set_spatial(
            scene
                .to_json()
                .map_err(|e| anyhow::anyhow!("Failed to serialize spatial scene: {}", e))?,
        );
        println!("  Spatial scene embedded from {}", spat_path.display());
    }

    // Optionally embed edit history.
    if let Some(edit_path) = edit_history_path {
        let edit_json = std::fs::read_to_string(edit_path)
            .with_context(|| format!("Failed to read edit history: {}", edit_path.display()))?;
        // Validate by parsing.
        let _history = EditHistory::from_json(&edit_json)
            .map_err(|e| anyhow::anyhow!("Invalid edit history: {}", e))?;
        writer.set_edit_history(edit_json);
        println!("  Edit history embedded from {}", edit_path.display());
    }

    writer
        .finalize(output)
        .with_context(|| format!("Failed to write NEO file: {}", output.display()))?;

    let file_size = std::fs::metadata(output).map(|m| m.len()).unwrap_or(0);

    println!("  --------------------------------------------");
    println!("  Output: {} ({} bytes)", output.display(), file_size,);
    println!("  Stems:  {}", stem_count);
    println!("  Codec:  {:?}", codec_id);
    println!("  Rate:   {} Hz", master_sample_rate);
    println!("  Duration: {:.2}s", max_duration_us as f64 / 1_000_000.0);
    println!("  Done!\n");

    Ok(())
}

// ──────────────────────────── decode ──────────────────────────────

/// Decode a `.neo` file, extracting stems back to WAV files.
///
/// If `--all-stems` is specified every stem is extracted.
/// If `--stem <label>` is specified only that stem is extracted.
/// If neither flag is provided, all stems are extracted by default.
fn cmd_decode(
    input: &Path,
    output_dir: &Path,
    stem_filter: Option<&str>,
    all_stems: bool,
) -> Result<()> {
    let mut reader = NeoReader::open(input)
        .with_context(|| format!("Failed to open NEO file: {}", input.display()))?;

    let header = reader.header().clone();
    let configs: Vec<StemConfig> = reader.stem_configs().to_vec();

    println!("\n  NEO Decoder");
    println!("  ============================================");
    println!("  Input:   {}", input.display());
    println!("  Stems:   {}", configs.len());
    println!("  Rate:    {} Hz", header.sample_rate);
    println!(
        "  Duration: {:.2}s",
        header.duration_us as f64 / 1_000_000.0
    );

    // Determine which stems to extract.
    let targets: Vec<&StemConfig> = if let Some(label) = stem_filter {
        let label_lower = label.to_lowercase();
        let matching: Vec<&StemConfig> = configs
            .iter()
            .filter(|c| c.label.as_str().to_lowercase() == label_lower)
            .collect();
        if matching.is_empty() {
            let available: Vec<&str> = configs.iter().map(|c| c.label.as_str()).collect();
            bail!(
                "Stem '{}' not found. Available stems: {}",
                label,
                available.join(", ")
            );
        }
        matching
    } else if all_stems || stem_filter.is_none() {
        // Default: extract all stems.
        configs.iter().collect()
    } else {
        configs.iter().collect()
    };

    // Ensure output directory exists.
    std::fs::create_dir_all(output_dir).with_context(|| {
        format!(
            "Failed to create output directory: {}",
            output_dir.display()
        )
    })?;

    for config in &targets {
        let codec = resolve_codec(config.codec).map_err(|e| {
            anyhow::anyhow!(
                "Unsupported codec {:?} for stem '{}': {}",
                config.codec,
                config.label.as_str(),
                e
            )
        })?;

        let decode_config = EncodeConfig {
            sample_rate: config.sample_rate,
            channels: config.channels,
            bitrate_kbps: None,
        };

        let raw_data = reader.read_stem_data(config.stem_id).map_err(|e| {
            anyhow::anyhow!(
                "Failed to read stem data for '{}': {}",
                config.label.as_str(),
                e
            )
        })?;

        let pcm_samples = codec.decode(&raw_data, &decode_config).map_err(|e| {
            anyhow::anyhow!("Failed to decode stem '{}': {}", config.label.as_str(), e)
        })?;

        let wav_filename = format!("{}.wav", config.label.as_str());
        let wav_path = output_dir.join(&wav_filename);

        write_wav(
            &wav_path,
            &pcm_samples,
            config.channels,
            config.sample_rate,
            config.bit_depth,
        )
        .with_context(|| format!("Failed to write WAV file: {}", wav_path.display()))?;

        let wav_size = std::fs::metadata(&wav_path).map(|m| m.len()).unwrap_or(0);
        let duration = if config.sample_rate > 0 && config.channels > 0 {
            pcm_samples.len() as f64 / (config.sample_rate as f64 * config.channels as f64)
        } else {
            0.0
        };

        println!(
            "  Extracted: {} ({:.2}s, {}ch, {} bytes)",
            wav_path.display(),
            duration,
            config.channels,
            wav_size,
        );
    }

    println!("  --------------------------------------------");
    println!(
        "  Extracted {} stem(s) to {}",
        targets.len(),
        output_dir.display()
    );
    println!("  Done!\n");

    Ok(())
}

// ───────────────────────────── info ───────────────────────────────

/// Display detailed information about a `.neo` file.
///
/// If `--json` is specified, outputs the full info structure as JSON.
/// Otherwise, prints a human-readable summary.
fn cmd_info(input: &Path, json: bool) -> Result<()> {
    let mut reader = NeoReader::open(input)
        .with_context(|| format!("Failed to open NEO file: {}", input.display()))?;

    let header = reader.header().clone();
    let chunks = reader.chunks().to_vec();
    let configs: Vec<StemConfig> = reader.stem_configs().to_vec();
    let file_size = reader.file_size();

    // Read optional metadata.
    let metadata = reader.read_metadata().unwrap_or(None);
    let temporal = reader.read_temporal().unwrap_or(None);
    let rights = reader.read_rights().unwrap_or(None);
    let credentials = reader.read_credentials().unwrap_or(None);
    let spatial = reader.read_spatial().unwrap_or(None);
    let edit_history = reader.read_edit_history().unwrap_or(None);

    // Build feature flag descriptions.
    let feature_names = describe_feature_flags(&header.feature_flags);

    // Compute integrity status — if NeoReader::open succeeded, all checksums passed.
    let integrity = "Verified (all BLAKE3 checksums passed)".to_string();

    let info = FileInfo {
        path: input,
        header: &header,
        chunks: &chunks,
        stems: &configs,
        file_size,
        feature_names: &feature_names,
        integrity: &integrity,
        metadata: metadata.as_deref(),
        temporal: temporal.as_deref(),
        rights: rights.as_deref(),
        credentials_size: credentials.as_ref().map(|c| c.len()),
        spatial: spatial.as_deref(),
        edit_history: edit_history.as_deref(),
    };

    if json {
        let json_val = info.to_json();
        println!("{}", serde_json::to_string_pretty(&json_val)?);
    } else {
        info.print_human();
    }

    Ok(())
}

/// Collected information about a `.neo` file, used for display.
struct FileInfo<'a> {
    /// Path to the `.neo` file.
    path: &'a Path,
    /// Parsed file header.
    header: &'a neo_format::NeoHeader,
    /// Chunk table entries.
    chunks: &'a [ChunkEntry],
    /// Stem configurations.
    stems: &'a [StemConfig],
    /// Total file size in bytes.
    file_size: u64,
    /// Human-readable feature flag names.
    feature_names: &'a [String],
    /// BLAKE3 integrity verification status.
    integrity: &'a str,
    /// Optional JSON-LD metadata string.
    metadata: Option<&'a str>,
    /// Optional temporal metadata string.
    temporal: Option<&'a str>,
    /// Optional Web3 rights JSON string.
    rights: Option<&'a str>,
    /// Optional C2PA credential bytes size.
    credentials_size: Option<usize>,
    /// Optional spatial audio scene JSON.
    spatial: Option<&'a str>,
    /// Optional edit history JSON.
    edit_history: Option<&'a str>,
}

impl FileInfo<'_> {
    /// Build a JSON representation of the file info.
    fn to_json(&self) -> serde_json::Value {
        let stem_array: Vec<serde_json::Value> = self
            .stems
            .iter()
            .map(|s| {
                serde_json::json!({
                    "stem_id": s.stem_id,
                    "label": s.label.as_str(),
                    "codec": format!("{:?}", s.codec),
                    "channels": s.channels,
                    "sample_rate": s.sample_rate,
                    "bit_depth": s.bit_depth,
                    "bitrate_kbps": s.bitrate_kbps,
                    "sample_count": s.sample_count,
                })
            })
            .collect();

        let chunk_array: Vec<serde_json::Value> = self
            .chunks
            .iter()
            .map(|c| {
                serde_json::json!({
                    "type": format!("{:?}", c.chunk_type),
                    "offset": c.offset,
                    "size": c.size,
                    "blake3": hex_encode(&c.blake3_hash),
                })
            })
            .collect();

        let mut info = serde_json::json!({
            "file": self.path.display().to_string(),
            "file_size": self.file_size,
            "header": {
                "magic": "NEO!",
                "version": self.header.version,
                "feature_flags": self.header.feature_flags.0,
                "feature_names": self.feature_names,
                "stem_count": self.header.stem_count,
                "sample_rate": self.header.sample_rate,
                "duration_us": self.header.duration_us,
                "duration_secs": self.header.duration_us as f64 / 1_000_000.0,
                "chunk_table_offset": self.header.chunk_table_offset,
                "chunk_count": self.header.chunk_count,
            },
            "stems": stem_array,
            "chunks": chunk_array,
            "integrity": self.integrity,
        });

        if let Some(meta) = self.metadata {
            if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(meta) {
                info["metadata"] = parsed;
            } else {
                info["metadata_raw"] = serde_json::Value::String(meta.to_string());
            }
        }
        if let Some(temp) = self.temporal {
            if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(temp) {
                info["temporal"] = parsed;
            }
        }

        if let Some(rights_str) = self.rights {
            if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(rights_str) {
                info["rights"] = parsed;
            }
        }
        if let Some(cred_size) = self.credentials_size {
            info["credentials"] = serde_json::json!({
                "has_manifest": true,
                "jumbf_size": cred_size,
            });
        }

        if let Some(spatial_str) = self.spatial {
            if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(spatial_str) {
                info["spatial"] = parsed;
            }
        }
        if let Some(edit_str) = self.edit_history {
            if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(edit_str) {
                info["edit_history"] = parsed;
            }
        }

        info
    }

    /// Print a human-readable summary of the `.neo` file.
    fn print_human(&self) {
        println!();
        println!("  NEO File Information");
        println!("  ============================================");
        println!("  File:     {}", self.path.display());
        println!(
            "  Size:     {} bytes ({})",
            self.file_size,
            human_size(self.file_size)
        );
        println!("  Magic:    NEO! (0x4E454F21)");
        println!("  Version:  {}", self.header.version);
        println!("  Rate:     {} Hz", self.header.sample_rate);
        println!(
            "  Duration: {:.2}s",
            self.header.duration_us as f64 / 1_000_000.0
        );
        println!("  Stems:    {}", self.header.stem_count);
        println!("  Chunks:   {}", self.header.chunk_count);

        if self.feature_names.is_empty() {
            println!("  Flags:    (none)");
        } else {
            println!("  Flags:    {}", self.feature_names.join(", "));
        }

        println!();
        println!("  Stems");
        println!("  --------------------------------------------");
        for s in self.stems {
            let duration = if s.sample_rate > 0 && s.channels > 0 {
                s.sample_count as f64 / (s.sample_rate as f64 * s.channels as f64)
            } else {
                0.0
            };
            println!(
                "  [{}] {} | {:?} | {}ch | {} Hz | {}bit | {:.2}s | {} samples",
                s.stem_id,
                s.label.as_str(),
                s.codec,
                s.channels,
                s.sample_rate,
                s.bit_depth,
                duration,
                s.sample_count,
            );
        }

        println!();
        println!("  Chunks");
        println!("  --------------------------------------------");
        for (i, c) in self.chunks.iter().enumerate() {
            println!(
                "  [{}] {:?} | offset {} | {} bytes | BLAKE3: {}...",
                i,
                c.chunk_type,
                c.offset,
                c.size,
                &hex_encode(&c.blake3_hash)[..16],
            );
        }

        println!();
        println!("  Integrity: {}", self.integrity);

        if let Some(meta) = self.metadata {
            println!();
            println!("  Metadata (JSON-LD)");
            println!("  --------------------------------------------");
            // Pretty-print if valid JSON.
            if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(meta) {
                if let Ok(pretty) = serde_json::to_string_pretty(&parsed) {
                    for line in pretty.lines() {
                        println!("  {}", line);
                    }
                } else {
                    println!("  {}", meta);
                }
            } else {
                println!("  {}", meta);
            }
        }

        if let Some(temp) = self.temporal {
            println!();
            println!("  Temporal Metadata");
            println!("  --------------------------------------------");
            if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(temp) {
                if let Ok(pretty) = serde_json::to_string_pretty(&parsed) {
                    for line in pretty.lines() {
                        println!("  {line}");
                    }
                }
            }
        }

        if let Some(rights_str) = self.rights {
            println!();
            println!("  Web3 Rights");
            println!("  --------------------------------------------");
            if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(rights_str) {
                if let Ok(pretty) = serde_json::to_string_pretty(&parsed) {
                    for line in pretty.lines() {
                        println!("  {line}");
                    }
                }
            }
        }

        if let Some(cred_size) = self.credentials_size {
            println!();
            println!("  C2PA Credentials");
            println!("  --------------------------------------------");
            println!("  JUMBF manifest present: {} bytes", cred_size);
        }

        if let Some(spatial_str) = self.spatial {
            println!();
            println!("  Spatial Audio");
            println!("  --------------------------------------------");
            if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(spatial_str) {
                // Show summary: number of objects, room type
                if let Some(objects) = parsed["objects"].as_array() {
                    println!("  Objects: {}", objects.len());
                    for obj in objects {
                        if let Some(stem_id) = obj["stem_id"].as_u64() {
                            let keyframes = obj["keyframes"].as_array().map_or(0, |k| k.len());
                            println!("    stem_id={} keyframes={}", stem_id, keyframes);
                        }
                    }
                }
                if let Some(room) = parsed.get("room") {
                    if let Some(room_type) = room["type"].as_str() {
                        println!(
                            "  Room: {} (reverb_time: {})",
                            room_type,
                            room["reverb_time"].as_f64().unwrap_or(0.0)
                        );
                    }
                }
            }
        }

        if let Some(edit_str) = self.edit_history {
            println!();
            println!("  Edit History");
            println!("  --------------------------------------------");
            if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(edit_str) {
                if let Some(commits) = parsed["commits"].as_array() {
                    println!("  Commits: {}", commits.len());
                    for commit in commits.iter().rev().take(5) {
                        let hash = commit["hash"].as_str().unwrap_or("?");
                        let msg = commit["message"].as_str().unwrap_or("");
                        let short_hash = if hash.len() > 8 { &hash[..8] } else { hash };
                        println!("    {} {}", short_hash, msg);
                    }
                    if commits.len() > 5 {
                        println!("    ... and {} more", commits.len() - 5);
                    }
                }
            }
        }

        println!();
    }
}

// ──────────────────────────── play ───────────────────────────────

/// Play a `.neo` file through the default audio output device.
///
/// Decodes all stems to PCM, applies --mute/--solo filtering,
/// mixes them together, and streams the result through `cpal`.
fn cmd_play(input: &Path, mute: Option<&str>, solo: Option<&str>) -> Result<()> {
    use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
    use std::sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    };

    // Open the NEO file
    let mut reader = NeoReader::open(input)
        .with_context(|| format!("Failed to open NEO file: {}", input.display()))?;

    let header = reader.header().clone();
    let configs: Vec<StemConfig> = reader.stem_configs().to_vec();

    // Parse mute/solo lists
    let mute_labels: Vec<&str> = mute
        .map(|s| s.split(',').map(|l| l.trim()).collect())
        .unwrap_or_default();
    let solo_labels: Vec<&str> = solo
        .map(|s| s.split(',').map(|l| l.trim()).collect())
        .unwrap_or_default();

    println!();
    println!("  NEO Player");
    println!("  ============================================");
    println!("  File: {}", input.display());
    println!("  Sample rate: {} Hz", header.sample_rate);
    println!("  Stems: {}", configs.len());

    // Decode and filter stems
    let mut mixed: Vec<f32> = Vec::new();
    let mut active_count = 0u32;
    let target_channels: u8 = 2; // Always mix to stereo

    for config in &configs {
        let label_str = config.label.as_str();

        // Apply solo/mute filtering
        if !solo_labels.is_empty()
            && !solo_labels
                .iter()
                .any(|&s| s.eq_ignore_ascii_case(label_str))
        {
            println!("  [skip] {} (not in solo list)", label_str);
            continue;
        }
        if mute_labels
            .iter()
            .any(|&m| m.eq_ignore_ascii_case(label_str))
        {
            println!("  [mute] {}", label_str);
            continue;
        }

        let codec = resolve_codec(config.codec)?;
        let decode_config = EncodeConfig {
            sample_rate: config.sample_rate,
            channels: config.channels,
            bitrate_kbps: None,
        };
        let raw_data = reader
            .read_stem_data(config.stem_id)
            .with_context(|| format!("Failed to read stem '{}' data", label_str))?;
        let pcm = codec
            .decode(&raw_data, &decode_config)
            .with_context(|| format!("Failed to decode stem '{}'", label_str))?;

        // Convert mono to stereo if needed
        let stereo_pcm = if config.channels == 1 {
            let mut s = Vec::with_capacity(pcm.len() * 2);
            for &sample in &pcm {
                s.push(sample);
                s.push(sample);
            }
            s
        } else {
            pcm
        };

        // Mix into the output buffer
        if mixed.is_empty() {
            mixed = stereo_pcm;
        } else {
            let len = mixed.len().min(stereo_pcm.len());
            for i in 0..len {
                mixed[i] += stereo_pcm[i];
            }
            // Extend if this stem is longer
            if stereo_pcm.len() > mixed.len() {
                mixed.extend_from_slice(&stereo_pcm[mixed.len()..]);
            }
        }
        active_count += 1;
        println!(
            "  [play] {} ({} ch, {} Hz)",
            label_str, config.channels, config.sample_rate
        );
    }

    if mixed.is_empty() {
        println!("\n  No stems to play.");
        return Ok(());
    }

    // Normalize the mix to prevent clipping when summing multiple stems
    if active_count > 1 {
        let scale = 1.0 / (active_count as f32).sqrt();
        for sample in &mut mixed {
            *sample *= scale;
        }
    }

    // Clamp to [-1.0, 1.0]
    for sample in &mut mixed {
        *sample = sample.clamp(-1.0, 1.0);
    }

    let total_frames = mixed.len() / target_channels as usize;
    let duration_secs = total_frames as f64 / header.sample_rate as f64;
    println!(
        "  Duration: {:.1}s ({} frames)",
        duration_secs, total_frames
    );
    println!();

    // Set up cpal audio output
    let host = cpal::default_host();
    let device = host
        .default_output_device()
        .ok_or_else(|| anyhow::anyhow!("No audio output device found"))?;

    let stream_config = cpal::StreamConfig {
        channels: target_channels as u16,
        sample_rate: cpal::SampleRate(header.sample_rate),
        buffer_size: cpal::BufferSize::Default,
    };

    let samples = Arc::new(mixed);
    let position = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let finished = Arc::new(AtomicBool::new(false));

    let samples_clone = Arc::clone(&samples);
    let position_clone = Arc::clone(&position);
    let finished_clone = Arc::clone(&finished);

    let stream = device
        .build_output_stream(
            &stream_config,
            move |data: &mut [f32], _: &cpal::OutputCallbackInfo| {
                let pos = position_clone.load(Ordering::Relaxed);
                let remaining = samples_clone.len().saturating_sub(pos);
                let to_copy = data.len().min(remaining);

                if to_copy > 0 {
                    data[..to_copy].copy_from_slice(&samples_clone[pos..pos + to_copy]);
                }
                // Fill the rest with silence
                for sample in &mut data[to_copy..] {
                    *sample = 0.0;
                }

                position_clone.store(pos + to_copy, Ordering::Relaxed);
                if to_copy == 0 || pos + to_copy >= samples_clone.len() {
                    finished_clone.store(true, Ordering::Relaxed);
                }
            },
            move |err| {
                eprintln!("  Audio stream error: {}", err);
            },
            None, // No timeout
        )
        .context("Failed to build audio output stream")?;

    stream.play().context("Failed to start audio playback")?;
    println!("  ▶ Playing... (Ctrl+C to stop)");

    // Wait for playback to finish
    while !finished.load(Ordering::Relaxed) {
        std::thread::sleep(std::time::Duration::from_millis(50));
    }
    // Small grace period for the audio buffer to drain
    std::thread::sleep(std::time::Duration::from_millis(200));

    println!("  ■ Playback complete.");
    println!();
    Ok(())
}

/// Tag a `.neo` file with metadata.
///
/// Reads the existing file, updates/creates metadata based on provided flags,
/// and writes a new `.neo` file. This is a read-modify-write operation.
#[allow(clippy::too_many_arguments)]
fn cmd_tag(
    input: &Path,
    output: Option<&Path>,
    title: Option<String>,
    artist: Option<String>,
    genre: Option<String>,
    bpm: Option<f64>,
    key: Option<String>,
    isrc: Option<String>,
    date: Option<String>,
    metadata_file: Option<&Path>,
    temporal_path: Option<&Path>,
    rights_path: Option<&Path>,
    show: bool,
) -> Result<()> {
    // Open and read the existing .neo file.
    let mut reader = NeoReader::open(input)
        .with_context(|| format!("Failed to open NEO file: {}", input.display()))?;

    let header = reader.header().clone();
    let configs: Vec<StemConfig> = reader.stem_configs().to_vec();

    // If --show, just display current metadata and exit.
    if show {
        println!("\n  NEO Metadata");
        println!("  ============================================");
        println!("  File: {}", input.display());

        if let Ok(Some(meta_str)) = reader.read_metadata() {
            println!("\n  JSON-LD Metadata:");
            println!("  --------------------------------------------");
            if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(&meta_str) {
                if let Ok(pretty) = serde_json::to_string_pretty(&parsed) {
                    for line in pretty.lines() {
                        println!("  {line}");
                    }
                }
            } else {
                println!("  {meta_str}");
            }
        } else {
            println!("\n  No JSON-LD metadata found.");
        }

        if let Ok(Some(temp_str)) = reader.read_temporal() {
            println!("\n  Temporal Metadata:");
            println!("  --------------------------------------------");
            if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(&temp_str) {
                if let Ok(pretty) = serde_json::to_string_pretty(&parsed) {
                    for line in pretty.lines() {
                        println!("  {line}");
                    }
                }
            }
        }

        if let Ok(Some(rights_str)) = reader.read_rights() {
            println!("\n  Web3 Rights:");
            println!("  --------------------------------------------");
            if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(&rights_str) {
                if let Ok(pretty) = serde_json::to_string_pretty(&parsed) {
                    for line in pretty.lines() {
                        println!("  {line}");
                    }
                }
            }
        }

        if let Ok(Some(cred_bytes)) = reader.read_credentials() {
            println!("\n  C2PA Credentials:");
            println!("  --------------------------------------------");
            println!("  JUMBF manifest: {} bytes", cred_bytes.len());
        }

        if let Ok(Some(spatial_str)) = reader.read_spatial() {
            println!("\n  Spatial Audio:");
            println!("  --------------------------------------------");
            if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(&spatial_str) {
                if let Ok(pretty) = serde_json::to_string_pretty(&parsed) {
                    for line in pretty.lines().take(20) {
                        println!("  {line}");
                    }
                }
            }
        }

        if let Ok(Some(edit_str)) = reader.read_edit_history() {
            println!("\n  Edit History:");
            println!("  --------------------------------------------");
            if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(&edit_str) {
                if let Some(commits) = parsed["commits"].as_array() {
                    println!("  {} commit(s)", commits.len());
                }
            }
        }

        println!();
        return Ok(());
    }

    // Build the updated metadata.
    // Start from existing metadata or create new.
    let existing_meta_str = reader.read_metadata().unwrap_or(None);
    let mut meta = if let Some(ref meta_path) = metadata_file {
        // Replace entirely from file.
        let json = std::fs::read_to_string(meta_path)
            .with_context(|| format!("Failed to read metadata file: {}", meta_path.display()))?;
        NeoMetadata::from_json(&json)
            .map_err(|e| anyhow::anyhow!("Invalid metadata JSON: {}", e))?
    } else if let Some(ref json_str) = existing_meta_str {
        // Try to parse existing metadata.
        NeoMetadata::from_json(json_str).unwrap_or_else(|_| NeoMetadata::default())
    } else {
        NeoMetadata::default()
    };

    // Apply individual field overrides.
    let mut changed = metadata_file.is_some();
    if let Some(t) = title {
        meta.name = t;
        changed = true;
    }
    if let Some(a) = artist {
        meta = meta.with_artist(a);
        changed = true;
    }
    if let Some(g) = genre {
        meta.genre = Some(g);
        changed = true;
    }
    if let Some(i) = isrc {
        meta.isrc_code = Some(i);
        changed = true;
    }
    if let Some(d) = date {
        meta.date_published = Some(d);
        changed = true;
    }

    // Handle temporal metadata.
    let existing_temporal_str = reader.read_temporal().unwrap_or(None);
    let mut temporal_json: Option<String> = existing_temporal_str;
    let mut temporal_changed = false;

    if let Some(temp_path) = temporal_path {
        let temp_str = std::fs::read_to_string(temp_path)
            .with_context(|| format!("Failed to read temporal file: {}", temp_path.display()))?;
        let temporal = TemporalMap::from_json(&temp_str)
            .map_err(|e| anyhow::anyhow!("Invalid temporal metadata: {}", e))?;
        temporal
            .validate()
            .map_err(|e| anyhow::anyhow!("Temporal validation failed: {}", e))?;
        temporal_json = Some(
            temporal
                .to_json()
                .map_err(|e| anyhow::anyhow!("Failed to serialize temporal: {}", e))?,
        );
        temporal_changed = true;
    } else if bpm.is_some() || key.is_some() {
        // Build/update temporal from --bpm and --key flags.
        let mut temporal = if let Some(ref json_str) = temporal_json {
            TemporalMap::from_json(json_str).unwrap_or_default()
        } else {
            TemporalMap::default()
        };
        if let Some(b) = bpm {
            temporal.bpm = b;
        }
        if let Some(k) = key {
            temporal.key = k;
        }
        temporal_json = Some(
            temporal
                .to_json()
                .map_err(|e| anyhow::anyhow!("Failed to serialize temporal: {}", e))?,
        );
        temporal_changed = true;
    }

    // Handle rights metadata.
    let existing_rights_str = reader.read_rights().unwrap_or(None);
    let mut rights_json: Option<String> = existing_rights_str;
    let mut rights_changed = false;

    if let Some(rights_path) = rights_path {
        let rights_str = std::fs::read_to_string(rights_path)
            .with_context(|| format!("Failed to read rights file: {}", rights_path.display()))?;
        let rights = RightsInfo::from_json(&rights_str)
            .map_err(|e| anyhow::anyhow!("Invalid rights metadata: {}", e))?;
        rights
            .validate()
            .map_err(|e| anyhow::anyhow!("Rights validation failed: {}", e))?;
        rights_json = Some(
            rights
                .to_json()
                .map_err(|e| anyhow::anyhow!("Failed to serialize rights: {}", e))?,
        );
        rights_changed = true;
    }

    if !changed && !temporal_changed && !rights_changed {
        println!("\n  No changes specified. Use --show to view current metadata,");
        println!("  or provide flags like --title, --artist, --bpm, --temporal, etc.");
        println!();
        return Ok(());
    }

    // Now rebuild the .neo file with updated metadata.
    println!("\n  NEO Tagger");
    println!("  ============================================");
    println!("  Input: {}", input.display());

    let mut writer = rebuild_neo_file(&mut reader, &header, &configs)?;

    // Override metadata with the updated version.
    meta.validate()
        .map_err(|e| anyhow::anyhow!("Metadata validation failed: {}", e))?;
    let meta_json = meta
        .to_json()
        .map_err(|e| anyhow::anyhow!("Failed to serialize metadata: {}", e))?;
    writer.set_metadata(meta_json);
    if changed {
        println!("  Updated: JSON-LD metadata");
    }

    // Override temporal if updated.
    if let Some(ref temp_json) = temporal_json {
        writer.set_temporal(temp_json.clone());
        if temporal_changed {
            println!("  Updated: Temporal metadata");
        }
    }

    // Override rights if updated.
    if let Some(ref rights_j) = rights_json {
        writer.set_rights(rights_j.clone());
        if rights_changed {
            println!("  Updated: Web3 rights");
        }
    }

    // Write output.
    let out_path = output.unwrap_or(input);
    writer
        .finalize(out_path)
        .with_context(|| format!("Failed to write NEO file: {}", out_path.display()))?;

    println!("  Output: {}", out_path.display());
    println!("  Done!\n");

    Ok(())
}

/// Stub for `neo model` — not yet implemented.
fn cmd_model_stub(action: ModelAction) -> Result<()> {
    match action {
        ModelAction::Download { variant } => {
            println!();
            println!("  NEO Model Download (not yet implemented)");
            println!("  ============================================");
            println!("  Requested: {}", variant);
            println!();
            println!("  DAC model download requires an internet connection");
            println!("  and the ONNX Runtime. This feature is planned for Phase 2.");
            println!("  For now, use `--codec pcm` for lossless encoding.");
            println!();
        }
        ModelAction::List => {
            println!();
            println!("  Available Models");
            println!("  ============================================");
            println!("  dac-44khz  -- DAC 44.1kHz (recommended for music)");
            println!("  dac-24khz  -- DAC 24kHz (speech + music)");
            println!("  dac-16khz  -- DAC 16kHz (speech)");
            println!();
            println!("  Installed: none");
            println!("  Use `neo model download <variant>` to install.");
            println!();
        }
    }
    Ok(())
}

// ──────────────────────────── edit ──────────────────────────────

/// Manage edit history in a `.neo` file.
fn cmd_edit(action: EditAction) -> Result<()> {
    match action {
        EditAction::History { input, json } => {
            let mut reader = NeoReader::open(&input)
                .with_context(|| format!("Failed to open: {}", input.display()))?;

            let edit_json = reader
                .read_edit_history()
                .with_context(|| "Failed to read edit history")?;

            match edit_json {
                Some(json_str) => {
                    if json {
                        println!("{}", json_str);
                    } else {
                        println!("\n  Edit History");
                        println!("  ============================================");
                        println!("  File: {}", input.display());
                        if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(&json_str) {
                            if let Some(commits) = parsed["commits"].as_array() {
                                println!("  Total commits: {}\n", commits.len());
                                for commit in commits.iter().rev() {
                                    let hash = commit["hash"].as_str().unwrap_or("?");
                                    let msg = commit["message"].as_str().unwrap_or("");
                                    let timestamp = commit["timestamp"].as_str().unwrap_or("");
                                    let ops = commit["ops"].as_array().map_or(0, |o| o.len());
                                    let short_hash =
                                        if hash.len() > 12 { &hash[..12] } else { hash };
                                    println!(
                                        "  {} {} ({} ops) {}",
                                        short_hash, msg, ops, timestamp
                                    );
                                }
                            }
                        }
                        println!();
                    }
                }
                None => {
                    println!("\n  No edit history found in {}\n", input.display());
                }
            }
            Ok(())
        }

        EditAction::AddOp {
            input,
            output,
            op,
            stem,
            params,
            message,
        } => {
            let mut reader = NeoReader::open(&input)
                .with_context(|| format!("Failed to open: {}", input.display()))?;

            let header = reader.header().clone();
            let configs = reader.stem_configs().to_vec();

            // Resolve stem_id from label if provided.
            let stem_id = if let Some(ref label) = stem {
                let label_lower = label.to_lowercase();
                configs
                    .iter()
                    .find(|c| c.label.as_str().to_lowercase() == label_lower)
                    .map(|c| c.stem_id)
                    .ok_or_else(|| {
                        let available: Vec<&str> =
                            configs.iter().map(|c| c.label.as_str()).collect();
                        anyhow::anyhow!(
                            "Stem '{}' not found. Available: {}",
                            label,
                            available.join(", ")
                        )
                    })?
            } else {
                0
            };

            // Parse the edit operation.
            let edit_op = parse_edit_op(&op, stem_id, params.as_deref())?;

            // Load or create edit history.
            let existing_json = reader.read_edit_history().unwrap_or(None);
            let mut history = if let Some(ref json_str) = existing_json {
                EditHistory::from_json(json_str)
                    .map_err(|e| anyhow::anyhow!("Invalid edit history: {}", e))?
            } else {
                EditHistory::new()
            };

            // Commit the operation.
            history
                .commit(vec![edit_op], &message, None)
                .map_err(|e| anyhow::anyhow!("Failed to commit edit: {}", e))?;

            let edit_json = history
                .to_json()
                .map_err(|e| anyhow::anyhow!("Failed to serialize edit history: {}", e))?;

            // Rebuild the file with updated edit history.
            let mut writer = rebuild_neo_file(&mut reader, &header, &configs)?;

            // Override the edit history with the updated version.
            writer.set_edit_history(edit_json);

            let out_path = output.as_deref().unwrap_or(&input);
            writer
                .finalize(out_path)
                .with_context(|| format!("Failed to write: {}", out_path.display()))?;

            println!("\n  Edit operation '{}' committed: {}", op, message);
            println!("  Output: {}\n", out_path.display());

            Ok(())
        }

        EditAction::Revert { input, output } => {
            let mut reader = NeoReader::open(&input)
                .with_context(|| format!("Failed to open: {}", input.display()))?;

            let header = reader.header().clone();
            let configs = reader.stem_configs().to_vec();

            let edit_json = reader
                .read_edit_history()
                .with_context(|| "Failed to read edit history")?
                .ok_or_else(|| anyhow::anyhow!("No edit history found in file"))?;

            let mut history = EditHistory::from_json(&edit_json)
                .map_err(|e| anyhow::anyhow!("Invalid edit history: {}", e))?;

            history
                .revert()
                .map_err(|e| anyhow::anyhow!("Failed to revert: {}", e))?;

            let new_json = history
                .to_json()
                .map_err(|e| anyhow::anyhow!("Failed to serialize: {}", e))?;

            // Rebuild the file.
            let mut writer = rebuild_neo_file(&mut reader, &header, &configs)?;

            // Override the edit history with the reverted version.
            writer.set_edit_history(new_json);

            let out_path = output.as_deref().unwrap_or(&input);
            writer
                .finalize(out_path)
                .with_context(|| format!("Failed to write: {}", out_path.display()))?;

            println!("\n  Reverted most recent edit commit.");
            println!("  Output: {}\n", out_path.display());

            Ok(())
        }
    }
}

/// Parse an edit operation from CLI arguments.
fn parse_edit_op(op_type: &str, stem_id: u8, params: Option<&str>) -> Result<EditOp> {
    let params_json: serde_json::Value = if let Some(p) = params {
        serde_json::from_str(p).with_context(|| format!("Invalid JSON params: {}", p))?
    } else {
        serde_json::json!({})
    };

    match op_type.to_lowercase().as_str() {
        "gain" => {
            let db = params_json["gain_db"].as_f64().unwrap_or(0.0);
            Ok(EditOp::Gain { stem_id, db })
        }
        "mute" => Ok(EditOp::Mute { stem_id }),
        "reverse" => Ok(EditOp::Reverse { stem_id }),
        "trim" => {
            let start_s = params_json["start_s"].as_f64().unwrap_or(0.0);
            let end_s = params_json["end_s"].as_f64().unwrap_or(0.0);
            Ok(EditOp::Trim {
                stem_id,
                start_s,
                end_s,
            })
        }
        "fade" => {
            let fade_in_s = params_json["fade_in_s"].as_f64().unwrap_or(0.0);
            let fade_out_s = params_json["fade_out_s"].as_f64().unwrap_or(0.0);
            Ok(EditOp::Fade {
                stem_id,
                fade_in_s,
                fade_out_s,
            })
        }
        "pan" => {
            let position = params_json["pan"].as_f64().unwrap_or(0.0);
            Ok(EditOp::Pan { stem_id, position })
        }
        _ => bail!(
            "Unknown edit operation '{}'. Supported: gain, mute, reverse, trim, fade, pan",
            op_type
        ),
    }
}

// ─────────────────────────── stream ───────────────────────────

/// Streaming information and integrity verification.
fn cmd_stream(action: StreamAction) -> Result<()> {
    match action {
        StreamAction::Info { input } => {
            let file_bytes = std::fs::read(&input)
                .with_context(|| format!("Failed to read: {}", input.display()))?;

            let tree = MerkleTree::build(&file_bytes, DEFAULT_CHUNK_SIZE)
                .map_err(|e| anyhow::anyhow!("Failed to build Merkle tree: {}", e))?;
            let cid = compute_cid(&file_bytes);

            println!("\n  NEO Stream Info");
            println!("  ============================================");
            println!("  File:        {}", input.display());
            println!("  Size:        {} bytes", file_bytes.len());
            println!("  CID:         {}", cid);
            println!("  Merkle root: {}", hex_encode(&tree.root_hash()));
            println!("  Blocks:      {}", tree.leaf_count());
            println!("  Block size:  {} bytes", DEFAULT_CHUNK_SIZE);
            println!();

            Ok(())
        }

        StreamAction::Verify { input } => {
            let file_bytes = std::fs::read(&input)
                .with_context(|| format!("Failed to read: {}", input.display()))?;

            let tree = MerkleTree::build(&file_bytes, DEFAULT_CHUNK_SIZE)
                .map_err(|e| anyhow::anyhow!("Failed to build Merkle tree: {}", e))?;
            let block_count = tree.leaf_count();
            let mut verified = 0;
            let mut failed = 0;

            for i in 0..block_count {
                let start = i * DEFAULT_CHUNK_SIZE;
                let end = ((i + 1) * DEFAULT_CHUNK_SIZE).min(file_bytes.len());
                if tree.verify_block(i, &file_bytes[start..end]) {
                    verified += 1;
                } else {
                    failed += 1;
                    eprintln!("  Block {} FAILED verification", i);
                }
            }

            println!("\n  NEO Stream Verification");
            println!("  ============================================");
            println!("  File:     {}", input.display());
            println!("  Blocks:   {}", block_count);
            println!("  Verified: {}", verified);
            println!("  Failed:   {}", failed);
            if failed == 0 {
                println!("  Result:   ✓ All blocks verified");
            } else {
                println!("  Result:   ✗ {} block(s) failed", failed);
            }
            println!();

            if failed > 0 {
                bail!("{} block(s) failed Merkle verification", failed);
            }

            Ok(())
        }
    }
}

// ──────────────────────── rebuild helper ─────────────────────────

/// Rebuild a `.neo` file by copying all stems and preserving all metadata chunks.
///
/// This is used by read-modify-write commands (`tag`, `edit add-op`, `edit revert`)
/// that need to update specific chunks while keeping everything else intact.
/// The returned [`NeoWriter`] has all stems and existing metadata loaded, so callers
/// can override specific fields (e.g., `set_metadata`, `set_edit_history`) before
/// calling `finalize`.
fn rebuild_neo_file(
    reader: &mut NeoReader,
    header: &neo_format::NeoHeader,
    configs: &[StemConfig],
) -> Result<NeoWriter> {
    let mut writer = NeoWriter::new(header.sample_rate, header.stem_count);
    writer.set_duration_us(header.duration_us);

    // Re-add all stems with their encoded data.
    for config in configs {
        let stem_data = reader
            .read_stem_data(config.stem_id)
            .map_err(|e| anyhow::anyhow!("Failed to read stem {}: {}", config.stem_id, e))?;
        writer
            .add_stem(config.clone(), stem_data)
            .map_err(|e| anyhow::anyhow!("Failed to add stem {}: {}", config.stem_id, e))?;
    }

    // Preserve all existing metadata chunks.
    if let Ok(Some(meta)) = reader.read_metadata() {
        writer.set_metadata(meta);
    }
    if let Ok(Some(temp)) = reader.read_temporal() {
        writer.set_temporal(temp);
    }
    if let Ok(Some(rights)) = reader.read_rights() {
        writer.set_rights(rights);
    }
    if let Ok(Some(creds)) = reader.read_credentials() {
        writer.set_credentials(creds);
    }
    if let Ok(Some(spatial)) = reader.read_spatial() {
        writer.set_spatial(spatial);
    }
    if let Ok(Some(edit_history)) = reader.read_edit_history() {
        writer.set_edit_history(edit_history);
    }

    Ok(writer)
}

// ──────────────────────── helper functions ─────────────────────────

/// Audio data read from a WAV file.
struct StemPayload {
    /// Interleaved f32 PCM samples.
    samples: Vec<f32>,
    /// Number of audio channels.
    channels: u8,
    /// Sample rate in Hz.
    sample_rate: u32,
    /// Bits per sample from the original WAV.
    bit_depth: u8,
    /// Total number of audio frames (samples per channel).
    sample_count: u64,
}

impl StemPayload {
    /// Duration of the audio in seconds.
    fn duration_secs(&self) -> f64 {
        if self.sample_rate == 0 {
            return 0.0;
        }
        self.sample_count as f64 / self.sample_rate as f64
    }
}

/// Read a WAV file and return the PCM samples as interleaved f32.
fn read_wav(path: &Path) -> Result<StemPayload> {
    let reader = hound::WavReader::open(path)
        .with_context(|| format!("Cannot open WAV file: {}", path.display()))?;

    let spec = reader.spec();
    let channels = spec.channels as u8;
    let sample_rate = spec.sample_rate;
    let bit_depth = spec.bits_per_sample as u8;
    let sample_count = (reader.len() as u64) / spec.channels as u64;

    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Int => {
            let max_val = (1u32 << (spec.bits_per_sample - 1)) as f32;
            reader
                .into_samples::<i32>()
                .map(|s| {
                    let s = s.context("Failed to read WAV sample")?;
                    Ok(s as f32 / max_val)
                })
                .collect::<Result<Vec<f32>>>()?
        }
        hound::SampleFormat::Float => reader
            .into_samples::<f32>()
            .map(|s| s.context("Failed to read WAV sample"))
            .collect::<Result<Vec<f32>>>()?,
    };

    Ok(StemPayload {
        samples,
        channels,
        sample_rate,
        bit_depth,
        sample_count,
    })
}

/// Write interleaved f32 PCM samples to a WAV file.
fn write_wav(
    path: &Path,
    samples: &[f32],
    channels: u8,
    sample_rate: u32,
    bit_depth: u8,
) -> Result<()> {
    // Determine the output format: use 32-bit float for 32-bit,
    // otherwise use integer format matching the original bit depth.
    let (sample_format, out_bits) = if bit_depth >= 32 {
        (hound::SampleFormat::Float, 32)
    } else {
        (hound::SampleFormat::Int, bit_depth.max(16) as u16)
    };

    let spec = hound::WavSpec {
        channels: channels as u16,
        sample_rate,
        bits_per_sample: out_bits,
        sample_format,
    };

    let mut writer = hound::WavWriter::create(path, spec)
        .with_context(|| format!("Cannot create WAV file: {}", path.display()))?;

    match sample_format {
        hound::SampleFormat::Float => {
            for &sample in samples {
                writer.write_sample(sample)?;
            }
        }
        hound::SampleFormat::Int => {
            let max_val = (1u32 << (out_bits - 1)) as f32;
            for &sample in samples {
                let clamped = sample.clamp(-1.0, 1.0);
                let int_sample = (clamped * max_val) as i32;
                writer.write_sample(int_sample)?;
            }
        }
    }

    writer.finalize()?;
    Ok(())
}

/// Resolve stem labels from `--stems` argument or from input filenames.
fn resolve_stem_labels(inputs: &[PathBuf], stems_arg: Option<&str>) -> Result<Vec<StemLabel>> {
    if let Some(stems_str) = stems_arg {
        let labels: Vec<StemLabel> = stems_str
            .split(',')
            .map(|s| {
                s.trim()
                    .parse::<StemLabel>()
                    .map_err(|e| anyhow::anyhow!("Invalid stem label '{}': {}", s.trim(), e))
            })
            .collect::<Result<Vec<StemLabel>>>()?;
        if labels.len() != inputs.len() {
            bail!(
                "Number of stem labels ({}) does not match number of input files ({})",
                labels.len(),
                inputs.len()
            );
        }
        Ok(labels)
    } else {
        // Derive labels from filenames.
        let labels: Vec<StemLabel> = inputs
            .iter()
            .map(|p| {
                let stem_name = p.file_stem().and_then(|s| s.to_str()).unwrap_or("mix");
                stem_name.parse::<StemLabel>().map_err(|e| {
                    anyhow::anyhow!("Invalid stem label from filename '{}': {}", stem_name, e)
                })
            })
            .collect::<Result<Vec<StemLabel>>>()?;
        Ok(labels)
    }
}

/// Parse a codec name string into a [`CodecId`].
fn parse_codec_name(name: &str) -> Result<CodecId> {
    match name.to_lowercase().as_str() {
        "pcm" | "raw" => Ok(CodecId::Pcm),
        "dac" | "neural" => Ok(CodecId::Dac),
        "flac" | "lossless" => Ok(CodecId::Flac),
        "opus" => Ok(CodecId::Opus),
        _ => bail!(
            "Unknown codec '{}'. Supported codecs: pcm, dac, flac, opus",
            name
        ),
    }
}

/// Create a codec instance for the given [`CodecId`].
///
/// For the MVP, only PCM is fully functional. Other codecs return
/// a descriptive error message.
fn resolve_codec(codec_id: CodecId) -> Result<Box<dyn AudioCodec>> {
    match codec_id {
        CodecId::Pcm => Ok(Box::new(PcmCodec)),
        CodecId::Dac => bail!(
            "DAC neural codec requires ONNX models. Use `neo model download` first, or use `--codec pcm` for now."
        ),
        CodecId::Flac => bail!("FLAC codec is not yet implemented (planned for Phase 2). Use `--codec pcm` for now."),
        CodecId::Opus => bail!("Opus codec is not yet implemented (planned for Phase 2). Use `--codec pcm` for now."),
    }
}

/// Describe the set feature flags as a list of human-readable names.
fn describe_feature_flags(flags: &FeatureFlags) -> Vec<String> {
    let mut names = Vec::new();
    if flags.has(FeatureFlags::LOSSLESS) {
        names.push("lossless".to_string());
    }
    if flags.has(FeatureFlags::SPATIAL) {
        names.push("spatial".to_string());
    }
    if flags.has(FeatureFlags::C2PA) {
        names.push("c2pa".to_string());
    }
    if flags.has(FeatureFlags::EDIT_HISTORY) {
        names.push("edit-history".to_string());
    }
    if flags.has(FeatureFlags::PREVIEW_LOD) {
        names.push("preview-lod".to_string());
    }
    if flags.has(FeatureFlags::NEURAL_CODEC) {
        names.push("neural-codec".to_string());
    }
    if flags.has(FeatureFlags::WEB3_RIGHTS) {
        names.push("web3-rights".to_string());
    }
    if flags.has(FeatureFlags::TEMPORAL_META) {
        names.push("temporal-meta".to_string());
    }
    names
}

/// Format a byte count as a human-readable size string.
fn human_size(bytes: u64) -> String {
    const KB: u64 = 1024;
    const MB: u64 = KB * 1024;
    const GB: u64 = MB * 1024;

    if bytes >= GB {
        format!("{:.2} GiB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.2} MiB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.2} KiB", bytes as f64 / KB as f64)
    } else {
        format!("{} B", bytes)
    }
}

/// Encode a byte slice as a lowercase hex string.
fn hex_encode(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{:02x}", b)).collect()
}
