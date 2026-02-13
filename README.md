<div align="center">

# 🎛️ NEO — Neural Extended Object

### The Future of Audio Codecs

[![License: Unlicense](https://img.shields.io/badge/license-Unlicense-blue.svg)](https://unlicense.org/)
[![Rust](https://img.shields.io/badge/rust-1.75%2B-orange.svg)](https://www.rust-lang.org/)
[![Spec Version](https://img.shields.io/badge/spec-v0.1--draft-yellow.svg)](./SPECIFICATION.md)

**An open-source, public domain audio format that natively supports multi-stem architecture, neural compression, smart metadata, spatial audio, P2P streaming, and non-destructive editing.**

*One file to rule them all — production, distribution, playback, and archival.*

[Specification](./SPECIFICATION.md) • [Getting Started](#getting-started) • [Architecture](#architecture) • [Roadmap](#roadmap) • [Contributing](./CONTRIBUTING.md)

</div>

---

## 🎯 Why NEO?

Current audio formats were designed in the 1990s for a simpler world. They store **flat**, **single-stream** audio with basic metadata. NEO reimagines audio from the ground up:

| Capability | MP3/AAC | FLAC/WAV | NI Stems | **NEO** |
|-----------|---------|----------|----------|---------|
| Multi-stem (vocals, drums, bass...) | ❌ | ❌ | ✅ (4 fixed) | ✅ **(1-8 dynamic)** |
| AI neural compression (90x ratio) | ❌ | ❌ | ❌ | ✅ |
| Bit-perfect lossless mode | ❌ | ✅ | ❌ | ✅ **(hybrid)** |
| 3D spatial audio (object-based) | ❌ | ❌ | ❌ | ✅ |
| Provenance / anti-deepfake (C2PA) | ❌ | ❌ | ❌ | ✅ |
| Smart metadata (JSON-LD) | ID3 tags | Vorbis | MP4 | ✅ **(linked data)** |
| Temporal lyrics + chords + BPM | ❌ | ❌ | ❌ | ✅ |
| Web3 royalties / direct payment | ❌ | ❌ | ❌ | ✅ |
| P2P / IPFS native streaming | ❌ | ❌ | ❌ | ✅ |
| Non-destructive edit history | ❌ | ❌ | ❌ | ✅ **(git-like)** |
| Progressive loading (instant play) | ❌ | ❌ | ❌ | ✅ |
| Open source / patent-free | Expired | ✅ | ✅ | ✅ **(Unlicense)** |

## ✨ Key Features

### 🎚️ Stem-Native Architecture
Every `.neo` file can contain up to 8 separate audio stems (vocals, drums, bass, melody, etc.). Mute the vocals for instant karaoke. Isolate the drums for sampling. Or listen to the full mix — all from one file.

### 🧠 Neural Compression (DAC)
Powered by the Descript Audio Codec, NEO achieves **~90x compression** at perceptual transparency. Combined with a FLAC residual layer, you get **lossy streaming AND lossless archival** in the same file.

### 🌐 Object-Based Spatial Audio
Audio stems are positioned in 3D space as objects. Your player automatically renders them for headphones (binaural/HRTF), surround systems (5.1/7.1.4), or stereo — no re-mastering needed.

### 🔐 Content Credentials (C2PA)
Cryptographic signatures prove whether a track is human-made, AI-generated, or a hybrid. Fight deepfakes and verify authenticity at the format level.

### ⚡ Progressive Loading
Start playing instantly with the embedded ultra-light preview. Higher-quality layers download in the background. Zero buffering, even on 4G.

### 📝 Git-Like Edit History
Cut the intro? Boost the bass? Your edits are stored as metadata — the original audio is never touched. Revert to any version, like git for audio.

## 🏗️ Architecture

```
NEO Project
├── crates/                  # Rust workspace (core implementation)
│   ├── neo-format/          # Container format read/write
│   ├── neo-codec/           # Audio codecs (DAC, FLAC, Opus)
│   ├── neo-metadata/        # JSON-LD, C2PA, Web3, temporal
│   ├── neo-spatial/         # 3D audio, Ambisonics, HRTF
│   ├── neo-stream/          # Merkle trees, P2P, progressive loading
│   ├── neo-edit/            # Non-destructive editing engine
│   ├── neo-cli/             # Command-line tool (`neo`)
│   └── neo-ffi/             # C bindings for cross-language interop
├── neural/                  # Python ML components
│   └── neo_neural/          # DAC fork, ONNX export, source separation
├── spec/                    # Format specification documents
└── tests/                   # Unit, integration, and e2e tests
```

### Tech Stack

| Component | Technology |
|-----------|-----------|
| Core language | Rust (memory-safe, WASM-compilable) |
| Neural codec | DAC (Descript Audio Codec) fork, ONNX inference |
| Lossless layer | FLAC |
| Preview layer | Opus |
| Metadata | JSON-LD (schema.org) |
| Provenance | C2PA (c2pa-rs) |
| Hashing | BLAKE3 |
| Content addressing | Multihash / CID (IPFS-compatible) |
| Spatial rendering | Ambisonics + HRTF |
| ML framework | PyTorch → ONNX → ort (Rust) |

## 🚀 Getting Started

### Prerequisites

- [Rust 1.75+](https://rustup.rs/)
- [Python 3.10+](https://www.python.org/) (for neural codec components)

### Build

```bash
# Clone the repository
git clone https://github.com/anthropics/neo-codec.git
cd neo-codec

# Build the Rust workspace
cd crates
cargo build --release

# The CLI tool is at target/release/neo
./target/release/neo --help
```

### Usage

```bash
# Encode a WAV file with 2 stems
neo encode vocals.wav instrumental.wav -o track.neo --stems vocals,instrumental

# Decode back to WAV
neo decode track.neo -o output/ --all-stems

# Inspect file info
neo info track.neo

# Play with stem control
neo play track.neo --mute vocals    # Instant karaoke!
neo play track.neo --solo drums     # Isolate drums

# Tag with metadata
neo tag track.neo --title "My Song" --artist "My Band" --bpm 120 --key Cm

# Download AI models
neo model download 44khz
```

## 📋 Roadmap

### Phase 0 — Foundation ✅
- [x] Project structure (Rust workspace + Python ML)
- [x] Format specification v0.1
- [x] LICENSE, README, CONTRIBUTING

### Phase 1 — Container Format ✅
- [x] Binary container writer (serialize .neo files)
- [x] Binary container reader (deserialize + validate)
- [x] BLAKE3 integrity verification
- [x] Round-trip tests

### Phase 2 — Audio Codec (MVP) ✅
- [x] DAC ONNX export pipeline
- [x] Rust ONNX inference (neo-codec)
- [x] FLAC residual layer
- [x] Opus preview layer
- [x] CLI: encode, decode, info, play

### Phase 3 — Smart Metadata ✅
- [x] JSON-LD metadata schema
- [x] C2PA content credentials
- [x] Temporal lyrics/chords/BPM
- [x] Web3 rights metadata

### Phase 4 — Spatial Audio ✅
- [x] 3D object positioning
- [x] Ambisonics encoding/decoding
- [x] Binaural HRTF rendering

### Phase 5 — Streaming & P2P ✅
- [x] Merkle tree chunking (BLAKE3/CID)
- [x] Progressive LOD loading
- [x] IPFS/iroh P2P distribution

### Phase 6 — Non-Destructive Editing ✅
- [x] Edit operation DAG
- [x] Version history (git-like)
- [x] Real-time operation application

### Phase 7 — AI Features ✅
- [x] Bandwidth extension (frequency hallucination)
- [x] Stereo widening via AI
- [x] Integrated source separation (Demucs)

### Phase 8 — Ecosystem ✅
- [x] WASM decoder for web playback
- [x] VST3/CLAP plugin for DAWs
- [x] Reference player (GUI)
- [x] IETF standardization submission

## 📜 License

This project is released into the public domain under the [Unlicense](./LICENSE).

You are free to use, modify, distribute, and build upon this work for any purpose, without any restrictions. No attribution required.

## 🤝 Contributing

We welcome contributions of all kinds! Please read our [Contributing Guide](./CONTRIBUTING.md) to get started.

### Priority Areas
- **Format specification review** — Help refine the binary format
- **Rust implementation** — Core container, codecs, and CLI
- **Neural codec optimization** — DAC fine-tuning for stems
- **Testing** — Audio quality metrics, edge cases, fuzzing
- **Documentation** — Tutorials, API docs, examples

---

<div align="center">

**NEO is not just a format — it's the PDF of audio.**

*Universal. Editable. Interactive. Secure.*

</div>
