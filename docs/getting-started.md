# Getting Started with NEO

NEO (Neural Extended Object) is an open-source, public domain audio codec format that supports stem-native multi-track architecture, neural compression, smart metadata, spatial audio, and more.

## Installation

### From Source

```bash
# Clone the repository
git clone https://github.com/anthropics/neo-codec.git
cd neo-codec

# Build the CLI tool
cd crates && cargo build --release -p neo-cli

# The binary is at crates/target/release/neo
```

### Requirements

- **Rust** 1.75 or later
- **Python** 3.10+ (optional, for neural/ML components)

## Quick Start

### Encode a WAV file to NEO

```bash
# Basic encode (single stem, PCM codec)
neo encode input.wav -o output.neo

# Encode with stem labels
neo encode vocals.wav -o song.neo --stems vocals

# Multi-stem encode
neo encode vocals.wav instrumental.wav -o song.neo --stems vocals,instrumental

# Encode with a specific codec (pcm, dac, flac, opus)
neo encode input.wav -o output.neo --codec dac

# Encode with embedded metadata
neo encode input.wav -o output.neo --metadata metadata.json
```

### Inspect a NEO file

```bash
neo info output.neo

# Output as JSON for programmatic use
neo info output.neo --json
```

This displays the file header, stem configurations, metadata, and chunk table.

### Decode back to WAV

```bash
# Decode all stems
neo decode output.neo -o output_dir/ --all-stems

# Decode a specific stem
neo decode output.neo -o output_dir/ --stem vocals
```

### Play a NEO file

```bash
# Play all stems mixed together
neo play song.neo

# Mute specific stems
neo play song.neo --mute vocals

# Solo specific stems
neo play song.neo --solo drums,bass
```

## Working with Metadata

### Add metadata to a NEO file

```bash
# Tag with basic metadata
neo tag song.neo --title "My Song" --artist "Artist Name" --genre "Electronic" --bpm 128

# Add musical key and ISRC code
neo tag song.neo --key "C major" --isrc US1234567890

# Show current metadata
neo tag song.neo --show

# Import JSON-LD metadata from file
neo tag song.neo --metadata-file metadata.json

# Import temporal metadata (lyrics, chords, BPM map)
neo tag song.neo --temporal temporal.json

# Import Web3 rights/royalty information
neo tag song.neo --rights rights.json

# Write changes to a new file instead of overwriting
neo tag song.neo -o song_tagged.neo --title "My Song"
```

### Metadata format (JSON-LD)

```json
{
  "@context": "https://schema.org/",
  "@type": "MusicRecording",
  "name": "My Song",
  "byArtist": { "@type": "Person", "name": "Artist Name" },
  "genre": "Electronic",
  "duration": "PT3M24S"
}
```

## Non-Destructive Editing

NEO supports a non-destructive edit graph — the original audio is never modified.

```bash
# Add a gain edit operation to a stem
neo edit add-op song.neo --op gain --stem vocals --params '{"gain_db": -3.0}' --message "Lower vocals"

# Trim, mute, pan, fade, reverse operations are also available
neo edit add-op song.neo --op mute --stem drums --message "Mute drums"

# View edit history
neo edit history song.neo

# View history as JSON
neo edit history song.neo --json

# Revert the most recent edit
neo edit revert song.neo

# Write edit changes to a new file
neo edit add-op song.neo -o song_edited.neo --op gain --stem vocals --params '{"gain_db": -3.0}'
```

## Streaming & Integrity

```bash
# Show streaming info (Merkle tree, content identifiers)
neo stream info song.neo

# Verify file integrity using Merkle proofs
neo stream verify song.neo
```

## AI Models

```bash
# Download a pre-trained DAC ONNX model
neo model download 44khz

# List available and installed models
neo model list
```

## File Format Overview

A `.neo` file consists of:
- **64-byte header** with magic bytes `NEO!`, version, feature flags, and stem count
- **Chunk table** listing all data chunks with BLAKE3 hashes for integrity
- **Stem chunks** (`STEM`) containing the audio data (one per stem)
- **Optional chunks** for metadata (`META`), credentials (`CRED`), spatial audio (`SPAT`), edit history (`EDIT`), and preview (`PREV`)

Key properties:
- All multi-byte integers are **little-endian**
- All strings are **UTF-8** with u16 length prefix
- Every chunk is **BLAKE3-hashed** for tamper detection
- Unknown chunk types are skipped gracefully (forward compatibility)
- Maximum 8 stems, 256 chunks per file

See the [Format Specification](../SPECIFICATION.md) for the complete binary format definition.

## Verbose Output

Add the `--verbose` flag to any command for debug-level logging:

```bash
neo --verbose encode input.wav -o output.neo
```

## License

NEO is released under the [Unlicense](../LICENSE) — fully public domain. No attribution required.
