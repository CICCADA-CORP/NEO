# NEO API Guide

This guide covers the Rust API for working with NEO files programmatically.

## Reading NEO Files

```rust
use std::path::Path;
use neo_format::{NeoReader, StemConfig};

// Open a NEO file
let reader = NeoReader::open(Path::new("song.neo"))?;

// Access the header
let header = reader.header();
println!("Version: {}", header.version);
println!("Stems: {}", header.stem_count);
println!("Sample rate: {} Hz", header.sample_rate);

// List stem configurations
for config in reader.stem_configs() {
    println!(
        "Stem {}: {} ({} ch, {} Hz, {:?})",
        config.stem_id,
        config.label,
        config.channels,
        config.sample_rate,
        config.codec
    );
}

// Read raw stem data
let stem_data = reader.read_stem_data(0)?;

// Read metadata
if let Some(metadata) = reader.read_metadata()? {
    println!("Metadata: {}", metadata);
}
```

## Writing NEO Files

```rust
use neo_format::{
    CodecId, FeatureFlags, NeoWriter, StemConfig, StemLabel,
};

// Create a writer with sample rate and stem count
let mut writer = NeoWriter::new(48000, 2);

// Define stems
let configs = vec![
    StemConfig {
        stem_id: 0,
        label: StemLabel::Vocals,
        codec: CodecId::Pcm,
        channels: 2,
        sample_rate: 48000,
        bit_depth: 32,
        bitrate_kbps: 0,
        sample_count: 240_000,
    },
    StemConfig {
        stem_id: 1,
        label: StemLabel::Instrumental,
        codec: CodecId::Pcm,
        channels: 2,
        sample_rate: 48000,
        bit_depth: 32,
        bitrate_kbps: 0,
        sample_count: 240_000,
    },
];

// Write the file
// Add stems, metadata, and finalize to disk
```

## Audio Codecs

```rust
use neo_codec::{AudioCodec, PcmCodec, EncodeConfig};

let codec = PcmCodec;
let config = EncodeConfig {
    sample_rate: 48000,
    channels: 2,
    bitrate_kbps: None,
};

// Encode f32 samples to bytes
let samples: Vec<f32> = vec![0.0, 0.5, -0.5, 1.0];
let encoded = codec.encode(&samples, &config)?;

// Decode bytes back to f32 samples
let decoded = codec.decode(&encoded, &config)?;
assert_eq!(samples, decoded);
```

### Available Codecs

| Codec | `CodecId` | Description |
|-------|-----------|-------------|
| PCM | `CodecId::Pcm` | Uncompressed lossless baseline (32-bit float) |
| DAC | `CodecId::Dac` | Neural compression via Descript Audio Codec (~90x ratio) |
| FLAC | `CodecId::Flac` | Lossless compression, also used for residual layers |
| Opus | `CodecId::Opus` | Lightweight lossy codec for preview/streaming |

## Metadata

```rust
use neo_metadata::{NeoMetadata, TemporalMap, RightsInfo};

// Create JSON-LD metadata
let metadata = NeoMetadata::builder()
    .title("My Song")
    .artist("Artist")
    .genre("Electronic")
    .bpm(128.0)
    .build();

let json = metadata.to_json_ld()?;

// Temporal metadata (lyrics, chords, BPM maps)
let temporal = TemporalMap::new();
// ... add time-stamped annotations

// Web3 rights information
let rights = RightsInfo::new();
// ... add royalty splits, wallet addresses
```

### JSON-LD Metadata Schema

NEO metadata follows the [schema.org](https://schema.org/) vocabulary:

```json
{
  "@context": "https://schema.org/",
  "@type": "MusicRecording",
  "name": "My Song",
  "byArtist": { "@type": "Person", "name": "Artist Name" },
  "genre": "Electronic",
  "duration": "PT3M24S",
  "isrcCode": "US1234567890",
  "musicalKey": "C major",
  "tempo": 128
}
```

## Spatial Audio

```rust
use neo_spatial::SpatialScene;

// Create a spatial scene for 3D audio positioning
let scene = SpatialScene::new();

// Position stems in 3D space (azimuth, elevation, distance)
// Render binaurally for headphone playback
// Or render to surround speaker layouts (5.1, 7.1.4)
```

## Streaming

```rust
use neo_stream::{merkle::MerkleTree, cid::compute_cid};

// Build a Merkle tree for integrity verification
let tree = MerkleTree::from_chunks(&chunk_data)?;
let root_hash = tree.root();

// Compute a CID (Content Identifier) for IPFS compatibility
let cid = compute_cid(&data);
```

## Non-Destructive Editing

```rust
use neo_edit::{EditHistory, EditOp};

// Create an edit history
let mut history = EditHistory::new();

// Add operations — these modify playback without touching original audio
// Supported operations: trim, gain, mute, pan, fade, reverse

// View the version history
let commits = history.commits();
for commit in commits {
    println!("  {} — {}", commit.timestamp, commit.message);
}
```

## C FFI

The `neo-ffi` crate exposes a C-compatible API for use from other languages:

```c
#include "neo.h"

// Open a NEO file
NeoHandle handle = neo_open("song.neo");
if (handle == NEO_INVALID_HANDLE) {
    fprintf(stderr, "Failed to open file\n");
    return 1;
}

// Get stem count
uint8_t stems = neo_stem_count(handle);

// Read stem data
size_t size = 0;
const uint8_t* data = neo_read_stem(handle, 0, &size);

// Clean up
neo_close(handle);
```

## Error Handling

All library functions return `Result` types with descriptive errors:

```rust
use neo_format::{NeoReader, NeoFormatError};
use std::path::Path;

match NeoReader::open(Path::new("bad_file.neo")) {
    Ok(reader) => { /* use reader */ }
    Err(NeoFormatError::InvalidMagic) => {
        eprintln!("Not a valid NEO file");
    }
    Err(NeoFormatError::ChecksumMismatch { expected, actual }) => {
        eprintln!("File corrupted: hash mismatch");
    }
    Err(e) => eprintln!("Error: {}", e),
}
```

### Error Types by Crate

| Crate | Error Type | Common Variants |
|-------|-----------|-----------------|
| `neo-format` | `NeoFormatError` | `InvalidMagic`, `ChecksumMismatch`, `UnsupportedVersion`, `Io` |
| `neo-codec` | `CodecError` | `EncodeFailed`, `DecodeFailed`, `UnsupportedCodec`, `ModelNotFound` |
| `neo-metadata` | `MetadataError` | `InvalidJsonLd`, `SchemaViolation`, `SerializationError` |
| `neo-cli` | `anyhow::Error` | Wraps all library errors with context messages |

## Adding NEO as a Dependency

Add the crates you need to your `Cargo.toml`:

```toml
[dependencies]
neo-format = { path = "../neo-codec/crates/neo-format" }
neo-codec = { path = "../neo-codec/crates/neo-codec" }
neo-metadata = { path = "../neo-codec/crates/neo-metadata" }
```

Or, once published to crates.io:

```toml
[dependencies]
neo-format = "0.1"
neo-codec = "0.1"
```

## Further Reading

- [Format Specification](../SPECIFICATION.md) — Complete binary format definition
- [Architecture](./architecture.md) — Crate structure and design decisions
- [Getting Started](./getting-started.md) — CLI usage guide
- [Contributing](../CONTRIBUTING.md) — How to contribute to the project
