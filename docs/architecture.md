# NEO Architecture

## Crate Structure

NEO is organized as a Rust workspace with 8 crates:

```
crates/
├── neo-format    Core container format (.neo file reading/writing)
├── neo-codec     Audio codecs (PCM, DAC neural, FLAC, Opus)
├── neo-metadata  Metadata handling (JSON-LD, C2PA, temporal, Web3 rights)
├── neo-spatial   Spatial audio (3D positioning, Ambisonics, binaural HRTF)
├── neo-stream    Streaming support (Merkle trees, progressive LOD, P2P/IPFS)
├── neo-edit      Non-destructive editing (edit DAG, version history)
├── neo-cli       Command-line interface binary
└── neo-ffi       Foreign function interface (C bindings, WASM, plugin stubs)
```

## Dependency Graph

```
neo-cli ──→ neo-format
        ├─→ neo-codec ──→ neo-format
        ├─→ neo-metadata
        ├─→ neo-spatial
        ├─→ neo-stream
        └─→ neo-edit ──→ neo-format

neo-ffi ──→ neo-format
        ├─→ neo-codec
        ├─→ neo-metadata
        └─→ neo-spatial
```

### Crate Responsibilities

| Crate | Purpose |
|-------|---------|
| `neo-format` | Defines the binary container format: header, chunk table, stem entries. Provides `NeoReader` and `NeoWriter` for file I/O. All other crates depend on its types (`CodecId`, `StemConfig`, `ChunkEntry`, `FeatureFlags`). |
| `neo-codec` | Implements the `AudioCodec` trait for encoding/decoding audio. Includes PCM (lossless baseline), DAC (neural compression via ONNX), FLAC, and Opus codecs. |
| `neo-metadata` | Handles JSON-LD structured metadata (schema.org), C2PA content credentials, temporal annotations (lyrics, chords, BPM maps), and Web3 rights/royalty information. |
| `neo-spatial` | 3D object-based audio positioning. Supports Ambisonics encoding/decoding and binaural HRTF rendering for headphone playback. |
| `neo-stream` | Content-addressed streaming infrastructure. Builds Merkle trees (BLAKE3) for integrity verification, computes CIDs for IPFS compatibility, and supports progressive level-of-detail loading. |
| `neo-edit` | Non-destructive editing engine with a git-like edit history. Operations (gain, trim, mute, pan, fade, reverse) are stored as metadata without modifying original audio. |
| `neo-cli` | The `neo` command-line tool. Provides encode, decode, info, play, tag, edit, stream, and model subcommands. Uses `anyhow` for error handling and `clap` for argument parsing. |
| `neo-ffi` | C FFI bindings for cross-language interop. Enables integration with C/C++, Python, and other languages. Includes stubs for WASM and VST3/CLAP plugin targets. |

## Key Design Decisions

### Binary Format
- Magic bytes: `0x4E 0x45 0x4F 0x21` (ASCII: `NEO!`)
- Header is exactly **64 bytes**, fixed layout
- All multi-byte integers are **little-endian** (via `byteorder` crate)
- Every chunk is **BLAKE3-hashed** for integrity verification
- Unknown chunk types are skipped gracefully (forward compatibility)
- Maximum 8 stems per file, 256 chunks per file, 2 GiB per chunk

### Error Handling
- Library crates use `thiserror` with custom error enums (e.g., `NeoFormatError`)
- The CLI binary uses `anyhow` for ergonomic error propagation with context
- All errors are descriptive and actionable

### Codec Architecture
- The `AudioCodec` trait defines a common encode/decode interface
- **PCM**: Built-in lossless baseline, always available
- **DAC** (Descript Audio Codec): Neural compression (~90x ratio) via ONNX inference
- **FLAC**: Standard lossless compression; also used for residual layers
- **Opus**: Lightweight lossy codec for preview/streaming layers

### Logging
- All crates use the `tracing` crate for structured logging
- The CLI configures log level via the `--verbose` flag or `RUST_LOG` environment variable

### Serialization
- Public data structures derive `serde::Serialize` and `serde::Deserialize`
- Binary format uses `byteorder` for precise byte-level control
- Metadata is serialized as JSON (JSON-LD compatible)

## Key External Dependencies

| Crate | Purpose |
|-------|---------|
| `blake3` | BLAKE3 hashing for chunk integrity |
| `serde` + `serde_json` | Serialization (JSON-LD metadata) |
| `byteorder` | Little-endian binary I/O |
| `bytes` | Zero-copy buffers for container parsing |
| `thiserror` | Error type definitions in library crates |
| `anyhow` | Error propagation in the CLI binary |
| `clap` | CLI argument parsing (with derive) |
| `ort` | ONNX Runtime for DAC neural codec inference |
| `hound` | WAV file I/O |
| `cpal` | Cross-platform audio output |
| `tracing` | Structured logging |

## Security Model

- **Input validation**: All size fields are validated before memory allocation to prevent OOM attacks
- **Hash verification**: BLAKE3 hashes are checked on all chunk data during reads
- **Configurable limits**: The reader enforces configurable per-allocation memory limits
- **Hard maximums**: `stem_count ≤ 8`, `chunk_count ≤ 256`, `chunk_size ≤ 2 GiB`
- **ONNX safety**: Neural codec models are only loaded from trusted, user-specified paths
- **Web3 read-only**: Wallet addresses in rights metadata are informational only — no transactions are auto-executed

## Python Components

The `neural/` directory contains ML components that complement the Rust core:

- **`dac_fork/`**: Modified Descript Audio Codec with stem-aware encoding support
- **`export_onnx.py`**: Pipeline to convert PyTorch DAC models to ONNX format for Rust inference via `ort`
- **`train_stems.py`**: Fine-tuning pipeline for training stem-specific codec models
- **`separator.py`**: Source separation using Demucs for automatic stem extraction

Models are exported to ONNX and stored in `neural/models/` for use by `neo-codec`.

## Testing Strategy

- **Unit tests**: Co-located in `#[cfg(test)]` modules within each source file
- **Integration tests**: In per-crate `tests/` directories, testing cross-module interactions
- **Round-trip tests**: Write → read → compare for all data structures
- **Fuzz testing**: `cargo-fuzz` targets for the container parser
- **Benchmarks**: Criterion-based performance benchmarks in `benches/`
- **Quality metrics**: PESQ/ViSQOL for codec quality evaluation

### Running Tests

```bash
# All tests
cd crates && cargo test --workspace

# Specific crate
cd crates && cargo test -p neo-format

# With output
cd crates && cargo test -p neo-format -- --nocapture

# Clippy linting
cd crates && cargo clippy --workspace --all-targets -- -D warnings
```
