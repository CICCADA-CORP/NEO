# Changelog

All notable changes to the NEO project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- Implemented TimeStretch edit operation (linear interpolation DSP)
- Implemented parametric Eq edit operation (biquad peaking filter DSP)
- Expanded FFI test coverage to all 13 exported C functions
- Added `cargo audit` job to CI pipeline
- Added code coverage reporting (`cargo-tarpaulin`) to CI pipeline
- Added MSRV (Minimum Supported Rust Version) check to CI
- Created this CHANGELOG.md

### Fixed
- Fixed `unwrap()` calls in CLI `resolve_stem_labels()` — now uses proper error propagation
- Fixed `unwrap()` on `StemLabel::parse()` in format reader — now returns error
- Updated README.md roadmap checkboxes to reflect completed Phases 1–8

### Changed
- Extracted read-modify-write helper in CLI to deduplicate file rebuild logic
- Improved DAC, FLAC, and Opus codec stub documentation and error messages
- Implemented Python `_build_dataset()` and `_compute_loss()` in `train_stems.py`

## [0.1.0] — 2026-02-12

### Added
- **Phase 0 — Foundation**: Project structure, Rust workspace (8 crates), Python ML scaffold, format specification v0.1, CI/CD pipelines
- **Phase 1 — Container Format**: Binary container reader/writer with 64-byte header, 9 chunk types, BLAKE3 integrity checksums, chunk table at EOF
- **Phase 2 — Audio Codec (MVP)**: `AudioCodec` trait, PCM codec (functional), DAC/FLAC/Opus stubs, residual reconstruction math, CLI encode/decode/info commands
- **Phase 3 — Metadata**: JSON-LD metadata schema (`schema.org/MusicRecording`), C2PA content credentials (JUMBF storage), temporal metadata (lyrics, chords, BPM), Web3 rights/royalty splits
- **Phase 4 — Spatial Audio**: 3D audio object positioning with keyframe interpolation, Ambisonics encoding/decoding (FOA/SOA/TOA), binaural HRTF rendering, room acoustics model
- **Phase 5 — Streaming**: BLAKE3 Merkle tree construction and verification, IPFS-compatible CID computation (multihash), progressive LOD (Level of Detail) streaming, P2P peer node stub (awaiting iroh)
- **Phase 6 — Editing**: Non-destructive edit DAG with 8 operation types, git-like edit history with commit/revert, edit diff and patch operations, BLAKE3-hashed node chain
- **Phase 7 — AI Features**: Bandwidth extension (spectral mirroring + noise shaping), stereo widening (Haas effect + decorrelation), Demucs source separation stub, Python ONNX export pipeline
- **Phase 8 — Ecosystem**: C FFI bindings with 13 exported functions and C header, WASM decoder architecture stub, VST3/CLAP plugin architecture stubs
- **Testing**: 291 tests across all crates (263 unit + 15 doc + 10 integration + 3 FFI)
- **CI/CD**: 6-job CI pipeline (check, fmt, clippy, test ×3 OS, doc, python-lint) + 5-target release pipeline
- **Specification**: Complete 686-line format specification (SPECIFICATION.md)

[Unreleased]: https://github.com/anthropics/neo-codec/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/anthropics/neo-codec/releases/tag/v0.1.0
