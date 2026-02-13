# NEO Format Specification v0.1

> **Neural Extended Object** — An open audio format for the future.
>
> **Status**: Draft
> **License**: Unlicense (Public Domain)
> **Date**: 2026-02-12
> **Authors**: NEO Project Contributors

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Design Goals](#2-design-goals)
3. [Terminology](#3-terminology)
4. [File Structure Overview](#4-file-structure-overview)
5. [Header](#5-header)
6. [Chunk Table](#6-chunk-table)
7. [Chunk Types](#7-chunk-types)
8. [Stem Encoding](#8-stem-encoding)
9. [Metadata Schema](#9-metadata-schema)
10. [Content Credentials](#10-content-credentials)
11. [Spatial Audio](#11-spatial-audio)
12. [Progressive Loading (LOD)](#12-progressive-loading-lod)
13. [Non-Destructive Editing](#13-non-destructive-editing)
14. [Streaming & Content Addressing](#14-streaming--content-addressing)
15. [Conformance Levels](#15-conformance-levels)
16. [Security Considerations](#16-security-considerations)
17. [IANA Considerations](#17-iana-considerations)
18. [References](#18-references)

---

## 1. Introduction

NEO (Neural Extended Object) is an open, royalty-free audio container format designed to replace existing formats (MP3, AAC, FLAC, WAV) with a unified solution that natively supports:

- **Multi-stem architecture**: Individual audio tracks (vocals, drums, bass, melody) stored in a single file
- **Neural compression**: AI-based audio codecs (DAC) with optional lossless residual for bit-perfect reconstruction
- **Smart metadata**: JSON-LD structured metadata with temporal annotations, C2PA provenance, and Web3 royalty information
- **Spatial audio**: Object-based 3D positioning with adaptive rendering (binaural, surround, stereo)
- **Progressive streaming**: Content-addressed chunks for P2P distribution and level-of-detail loading
- **Non-destructive editing**: Git-like version history of edit operations stored within the file

The format is placed in the public domain under the Unlicense to ensure maximum adoption and prevent patent encumbrance.

### 1.1 Relationship to Existing Formats

| Feature | WAV | MP3 | FLAC | AAC | Opus | NI Stems | **NEO** |
|---------|-----|-----|------|-----|------|----------|---------|
| Lossless | ✅ | ❌ | ✅ | ❌ | ❌ | ❌ | ✅ |
| Multi-stem | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ (4 fixed) | ✅ (1-8) |
| Neural codec | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |
| Spatial audio | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |
| Smart metadata | ❌ | ID3 | Vorbis | MP4 | Vorbis | MP4 | ✅ (JSON-LD) |
| Provenance (C2PA) | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |
| P2P streaming | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |
| Edit history | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |
| Open/Royalty-free | ✅ | ❌* | ✅ | ❌ | ✅ | ✅ | ✅ |

*MP3 patents have expired, but the format was not designed as open.

## 2. Design Goals

1. **Universality**: One file format for production, distribution, playback, and archival.
2. **Backward compatibility**: Decoders MUST be able to produce a standard stereo PCM output from any NEO file.
3. **Extensibility**: Unknown chunk types MUST be gracefully skipped without breaking the reader.
4. **Content integrity**: Every chunk is BLAKE3-hashed for tamper detection.
5. **Size efficiency**: Neural compression achieves 10-90x reduction vs PCM while maintaining perceptual quality.
6. **Streaming-first**: Files are structured for progressive loading and content-addressed distribution.
7. **No patents**: All required technologies must be royalty-free and patent-unencumbered.

## 3. Terminology

| Term | Definition |
|------|-----------|
| **Stem** | An individual audio track within the NEO file (e.g., vocals, drums) |
| **Chunk** | A typed, hashable data block within the container |
| **LOD** | Level of Detail — progressive quality layers |
| **DAC** | Descript Audio Codec — neural audio compression |
| **RVQ** | Residual Vector Quantization — the quantization method used by DAC |
| **Residual** | The difference between the original PCM and lossy-decoded PCM |
| **CID** | Content Identifier — a self-describing hash for content addressing |

### 3.1 Conventions

- All multi-byte integers are stored in **little-endian** byte order.
- All string fields are **UTF-8** encoded with a u16 length prefix.
- Byte offsets are absolute from the beginning of the file.
- The key words "MUST", "MUST NOT", "SHOULD", "MAY" are as defined in RFC 2119.

## 4. File Structure Overview

A NEO file has the following top-level structure:

```
┌─────────────────────────────────────────┐
│ HEADER (64 bytes, fixed)                │
├─────────────────────────────────────────┤
│ CHUNK DATA                              │
│   ┌─────────────────────────────────┐   │
│   │ Chunk 0: PREV (preview layer)   │   │
│   ├─────────────────────────────────┤   │
│   │ Chunk 1: STEM (vocals, DAC)     │   │
│   ├─────────────────────────────────┤   │
│   │ Chunk 2: STEM (instrumental)    │   │
│   ├─────────────────────────────────┤   │
│   │ Chunk 3: RESD (FLAC residual)   │   │
│   ├─────────────────────────────────┤   │
│   │ Chunk 4: META (JSON-LD)         │   │
│   ├─────────────────────────────────┤   │
│   │ Chunk 5: TEMP (temporal map)    │   │
│   ├─────────────────────────────────┤   │
│   │ Chunk 6: CRED (C2PA manifest)   │   │
│   ├─────────────────────────────────┤   │
│   │ Chunk 7: SPAT (spatial data)    │   │
│   ├─────────────────────────────────┤   │
│   │ Chunk 8: EDIT (edit history)    │   │
│   ├─────────────────────────────────┤   │
│   │ Chunk 9: RGHT (Web3 rights)     │   │
│   └─────────────────────────────────┘   │
├─────────────────────────────────────────┤
│ CHUNK TABLE (variable size)             │
│   Array of ChunkEntry (57 bytes each)   │
└─────────────────────────────────────────┘
```

### 4.1 File Extension

- Primary: `.neo`
- MIME type: `audio/neo` (to be registered with IANA)
- Magic bytes: `0x4E 0x45 0x4F 0x21` (ASCII: `NEO!`)

## 5. Header

The header is exactly **64 bytes** and appears at byte offset 0 of every NEO file.

### 5.1 Header Layout

| Offset | Size | Type | Field | Description |
|--------|------|------|-------|-------------|
| 0 | 4 | `[u8; 4]` | `magic` | Magic bytes: `0x4E 0x45 0x4F 0x21` (`NEO!`) |
| 4 | 2 | `u16` | `version` | Format version (currently `1`) |
| 6 | 8 | `u64` | `feature_flags` | Bitfield of enabled features |
| 14 | 1 | `u8` | `stem_count` | Number of audio stems (1-8) |
| 15 | 4 | `u32` | `sample_rate` | Sample rate in Hz |
| 19 | 8 | `u64` | `duration_us` | Total duration in microseconds |
| 27 | 8 | `u64` | `chunk_table_offset` | Byte offset of the chunk table |
| 35 | 8 | `u64` | `chunk_count` | Number of entries in the chunk table |
| 43 | 21 | `[u8; 21]` | `reserved` | Reserved for future use (MUST be zero) |

**Total: 64 bytes**

### 5.2 Feature Flags

| Bit | Flag Name | Description |
|-----|-----------|-------------|
| 0 | `LOSSLESS` | File contains lossless residual data |
| 1 | `SPATIAL` | File contains spatial audio positioning |
| 2 | `C2PA` | File contains C2PA content credentials |
| 3 | `EDIT_HISTORY` | File contains non-destructive edit history |
| 4 | `PREVIEW_LOD` | File contains a preview layer for progressive loading |
| 5 | `NEURAL_CODEC` | File uses neural codec (DAC) for at least one stem |
| 6 | `WEB3_RIGHTS` | File contains Web3 royalty/rights metadata |
| 7 | `TEMPORAL_META` | File contains temporal metadata (lyrics, chords, BPM) |
| 8-63 | Reserved | MUST be zero; readers MUST ignore unknown flags |

### 5.3 Validation Rules

- Readers MUST verify the magic bytes are `0x4E454F21`.
- Readers MUST reject files with `version > 1` (until future versions are specified).
- `stem_count` MUST be between 1 and 8 inclusive.
- `sample_rate` MUST be one of: 8000, 11025, 16000, 22050, 32000, 44100, 48000, 88200, 96000, 176400, 192000.
- `chunk_table_offset` MUST point to a valid location within the file.
- `reserved` bytes MUST be zero on write; readers MUST ignore their contents.

## 6. Chunk Table

The chunk table is an array of `ChunkEntry` structures located at `chunk_table_offset`. It contains `chunk_count` entries.

### 6.1 ChunkEntry Layout

| Offset | Size | Type | Field | Description |
|--------|------|------|-------|-------------|
| 0 | 1 | `u8` | `chunk_type` | Chunk type identifier |
| 1 | 8 | `u64` | `offset` | Byte offset of chunk data from file start |
| 9 | 8 | `u64` | `size` | Size of chunk data in bytes |
| 17 | 32 | `[u8; 32]` | `blake3_hash` | BLAKE3 hash of the chunk data |
| 49 | 8 | `[u8; 8]` | `reserved` | Reserved (MUST be zero) |

**Total per entry: 57 bytes**

### 6.2 Chunk Type Identifiers

| Value | Name | Description | Required |
|-------|------|-------------|----------|
| `0x01` | `STEM` | Audio stem data | YES (at least 1) |
| `0x02` | `META` | JSON-LD metadata | NO |
| `0x03` | `CRED` | C2PA content credentials | NO |
| `0x04` | `SPAT` | Spatial audio positioning | NO |
| `0x05` | `EDIT` | Non-destructive edit history | NO |
| `0x06` | `PREV` | Preview layer (low-quality) | NO |
| `0x07` | `RESD` | Lossless residual data | NO |
| `0x08` | `TEMP` | Temporal metadata (lyrics/chords/BPM) | NO |
| `0x09` | `RGHT` | Web3 rights/royalty information | NO |
| `0x0A-0xFF` | Reserved | Future use; readers MUST skip unknown types | — |

### 6.3 Integrity Verification

Readers SHOULD verify the BLAKE3 hash of each chunk against its `blake3_hash` field. If a mismatch is detected, the reader MUST report an error and MAY attempt to continue reading other chunks.

## 7. Chunk Types

### 7.1 STEM Chunk (0x01)

Each STEM chunk contains one audio stem. The chunk data begins with a stem header followed by the compressed audio data.

#### 7.1.1 Stem Header

| Offset | Size | Type | Field | Description |
|--------|------|------|-------|-------------|
| 0 | 1 | `u8` | `stem_id` | Stem identifier (0-7) |
| 1 | 1 | `u8` | `codec_id` | Codec used for this stem |
| 2 | 1 | `u8` | `channels` | Number of audio channels |
| 3 | 4 | `u32` | `sample_rate` | Stem sample rate in Hz |
| 7 | 1 | `u8` | `bit_depth` | Bit depth (16, 24, 32) |
| 8 | 4 | `u32` | `bitrate_kbps` | Bitrate hint (0 = VBR) |
| 12 | 8 | `u64` | `sample_count` | Total number of samples |
| 20 | 2 | `u16` | `label_len` | Length of the label string |
| 22 | var | `UTF-8` | `label` | Human-readable stem label |

After the stem header, the remaining bytes are the compressed audio data.

#### 7.1.2 Codec Identifiers

| Value | Codec | Description |
|-------|-------|-------------|
| `0x01` | DAC | Descript Audio Codec (neural, lossy) |
| `0x02` | FLAC | Free Lossless Audio Codec |
| `0x03` | Opus | Opus (lossy, lightweight) |
| `0x04` | PCM | Raw PCM (uncompressed) |

#### 7.1.3 Well-Known Stem Labels

These labels are RECOMMENDED for interoperability:

| Label | Description |
|-------|-------------|
| `vocals` | Lead and backing vocals |
| `drums` | Drums and percussion |
| `bass` | Bass guitar, synth bass |
| `melody` | Melodic instruments, synths, guitars |
| `instrumental` | Full instrumental (no vocals) |
| `mix` | Complete stereo mix |

### 7.2 META Chunk (0x02)

Contains a UTF-8 JSON-LD string following the schema defined in Section 9.

### 7.3 CRED Chunk (0x03)

Contains a C2PA manifest in JUMBF format as defined in Section 10.

### 7.4 SPAT Chunk (0x04)

Contains spatial audio positioning data in JSON format as defined in Section 11.

### 7.5 EDIT Chunk (0x05)

Contains the non-destructive edit history as defined in Section 13.

### 7.6 PREV Chunk (0x06)

Contains a low-quality Opus-encoded preview of the full mix. This MUST be the first chunk after the header to enable instant playback.

#### 7.6.1 Preview Specifications

- Codec: Opus
- Channels: 1 (mono) or 2 (stereo)
- Bitrate: 32 kbps or lower
- Sample rate: 48000 Hz (Opus native)
- Duration: Full track duration

### 7.7 RESD Chunk (0x07)

Contains the FLAC-encoded residual data for lossless reconstruction. The residual is computed as: `residual[i] = original[i] - dac_decoded[i]`

To reconstruct lossless audio: `original[i] = dac_decoded[i] + residual[i]`

### 7.8 TEMP Chunk (0x08)

Contains temporal metadata (lyrics, chords, BPM map) in JSON format as defined in Section 9.3.

### 7.9 RGHT Chunk (0x09)

Contains Web3 rights and royalty information in JSON format as defined in Section 9.4.

## 8. Stem Encoding

### 8.1 DAC Encoding (Neural)

The primary codec for NEO is the Descript Audio Codec (DAC), a neural audio codec using Improved RVQGAN architecture.

#### 8.1.1 DAC Parameters

| Parameter | Value |
|-----------|-------|
| Architecture | Improved RVQGAN (convolutional encoder → RVQ → convolutional decoder) |
| Supported sample rates | 16000, 24000, 44100 Hz |
| Codebook size | 1024 entries per codebook |
| Number of codebooks (n_q) | 1-32 (variable bitrate) |
| Frame size | Approximately 1 second |
| Compression ratio | ~90x at 8 kbps |

#### 8.1.2 DAC Data Format

DAC-encoded data consists of serialized RVQ codes:

```
┌──────────────────────────────┐
│ n_codebooks: u8              │
│ n_frames: u32                │
│ codes: [u16; n_codebooks *   │
│         n_frames]            │
│ (row-major: codebook × frame)│
└──────────────────────────────┘
```

### 8.2 FLAC Encoding (Lossless)

Standard FLAC encoding as per the FLAC specification. Used for:
- Direct lossless stem storage (when NEURAL_CODEC flag is not set)
- Residual layer encoding (RESD chunk)

### 8.3 Opus Encoding (Lightweight)

Standard Opus encoding as per RFC 6716. Used for:
- Preview layer (PREV chunk)
- Lightweight streaming scenarios
- Fallback when neural codec models are unavailable

### 8.4 Lossless Reconstruction

When both a DAC-encoded STEM chunk and a RESD chunk are present for the same stem, lossless reconstruction is performed:

1. Decode the DAC stem to PCM: `dac_pcm = DAC.decode(stem_data)`
2. Decode the FLAC residual: `residual = FLAC.decode(resd_data)`
3. Reconstruct: `original[i] = dac_pcm[i] + residual[i]` for all samples

The reconstruction MUST be bit-identical to the original PCM input (within floating-point precision for f32 samples).

## 9. Metadata Schema

### 9.1 JSON-LD Core Metadata

The META chunk contains a JSON-LD document using the schema.org vocabulary.

```json
{
  "@context": "https://schema.org",
  "@type": "MusicRecording",
  "name": "Track Title",
  "byArtist": {
    "@type": "MusicGroup",
    "name": "Artist Name"
  },
  "duration": "PT3M45S",
  "isrcCode": "USRC17607839",
  "inAlbum": {
    "@type": "MusicAlbum",
    "name": "Album Title"
  },
  "genre": "Electronic",
  "datePublished": "2026-01-15"
}
```

### 9.2 NEO Extensions

Custom NEO properties use the `neo:` prefix:

```json
{
  "@context": [
    "https://schema.org",
    {"neo": "https://neo-codec.org/schema/v1#"}
  ],
  "neo:stems": [
    {"id": 0, "label": "vocals", "codec": "dac"},
    {"id": 1, "label": "instrumental", "codec": "dac"}
  ],
  "neo:encoding": {
    "codec": "dac",
    "model": "44khz",
    "bitrate_kbps": 8,
    "has_residual": true
  }
}
```

### 9.3 Temporal Metadata

Stored in the TEMP chunk as JSON:

```json
{
  "bpm": 120.0,
  "key": "Cm",
  "time_signature": "4/4",
  "chords": [
    {"time": 0.0, "chord": "Cm"},
    {"time": 2.5, "chord": "Fm"},
    {"time": 5.0, "chord": "Bb"},
    {"time": 7.5, "chord": "Eb"}
  ],
  "lyrics": [
    {"time": 5.2, "end": 7.8, "line": "Hello world"},
    {"time": 8.0, "end": 10.5, "line": "This is NEO"}
  ],
  "bpm_changes": [
    {"time": 60.0, "bpm": 125.0}
  ]
}
```

### 9.4 Web3 Rights

Stored in the RGHT chunk as JSON:

```json
{
  "splits": [
    {
      "address": "0x1234...abcd",
      "chain": "eth",
      "share": 0.6,
      "name": "Producer"
    },
    {
      "address": "0x5678...efgh",
      "chain": "eth",
      "share": 0.4,
      "name": "Vocalist"
    }
  ],
  "contract_address": "0xabcd...1234",
  "contract_chain": "eth",
  "license_uri": "https://creativecommons.org/licenses/by/4.0/"
}
```

## 10. Content Credentials

NEO files MAY contain a C2PA manifest in the CRED chunk for cryptographic provenance.

### 10.1 C2PA Integration

- The manifest follows the C2PA specification v2.x
- The manifest is stored as a JUMBF (JPEG Universal Metadata Box Format) blob
- The manifest's hard binding hash covers: all STEM chunks, the META chunk, and the TEMP chunk
- The `origin` assertion MUST declare one of: `human`, `ai`, `hybrid`

### 10.2 Verification

Readers SHOULD verify the C2PA manifest signature chain. Verification failure MUST NOT prevent playback but SHOULD be surfaced to the user.

## 11. Spatial Audio

### 11.1 Object-Based Positioning

Each stem can be positioned in 3D space using the SPAT chunk. Positions are stored as JSON:

```json
{
  "objects": [
    {
      "stem_id": 0,
      "gain": 1.0,
      "spread": 0.2,
      "keyframes": [
        {"time": 0.0, "position": {"azimuth": -30, "elevation": 0, "distance": 0.8}},
        {"time": 5.0, "position": {"azimuth": 30, "elevation": 10, "distance": 0.6}}
      ]
    }
  ],
  "room": {
    "type": "studio",
    "reverb_time": 0.3
  }
}
```

### 11.2 Coordinate System

- **Azimuth**: -180° to +180° (0° = front, +90° = right, -90° = left, ±180° = rear)
- **Elevation**: -90° to +90° (0° = ear level, +90° = above, -90° = below)
- **Distance**: 0.0 to 1.0 (normalized; 0.0 = inside head, 1.0 = maximum distance)

### 11.3 Rendering Targets

Decoders MUST support at least stereo output. Decoders SHOULD support:
- Binaural (headphone, HRTF-based)
- 5.1 surround
- 7.1.4 (Dolby Atmos-compatible)

Position interpolation between keyframes MUST use linear interpolation.

## 12. Progressive Loading (LOD)

### 12.1 Layer Structure

| Layer | Content | Typical Size | Quality |
|-------|---------|-------------|---------|
| 0 (Preview) | PREV chunk — Opus mono 32kbps | 50-200 KB | Low |
| 1 (Standard) | STEM chunks — DAC compressed | 2-10 MB | High |
| 2 (Lossless) | RESD chunk — FLAC residual | 10-50 MB | Bit-perfect |

### 12.2 Loading Strategy

1. Reader downloads/reads the 64-byte header
2. Reader downloads/reads the chunk table (typically < 1KB)
3. Reader immediately plays the PREV chunk (Layer 0)
4. In background, reader fetches STEM chunks (Layer 1) and seamlessly crossfades
5. Optionally, reader fetches RESD chunk (Layer 2) for lossless

### 12.3 Chunk Ordering

Encoders SHOULD order chunks as: PREV, STEM (by stem_id), META, TEMP, RESD, CRED, SPAT, EDIT, RGHT. The PREV chunk SHOULD immediately follow the header for fastest streaming.

## 13. Non-Destructive Editing

### 13.1 Edit Operations

The EDIT chunk contains a JSON-serialized DAG (directed acyclic graph) of edit operations:

```json
{
  "commits": [
    {
      "hash": "base64(blake3)",
      "parent": null,
      "timestamp": "2026-02-12T10:30:00Z",
      "message": "Initial import",
      "ops": []
    },
    {
      "hash": "base64(blake3)",
      "parent": "base64(blake3)",
      "timestamp": "2026-02-12T11:00:00Z",
      "message": "Remove intro, boost bass",
      "ops": [
        {"type": "trim", "stem_id": 0, "start_s": 2.5, "end_s": 180.0},
        {"type": "gain", "stem_id": 2, "db": 3.0}
      ]
    }
  ],
  "head": "base64(blake3)"
}
```

### 13.2 Supported Operations

| Operation | Fields | Description |
|-----------|--------|-------------|
| `trim` | `stem_id`, `start_s`, `end_s` | Keep only the specified time range |
| `gain` | `stem_id`, `db` | Apply gain in decibels |
| `eq` | `stem_id`, `freq_hz`, `gain_db`, `q` | Parametric EQ band |
| `fade` | `stem_id`, `fade_in_s`, `fade_out_s` | Fade in/out |
| `mute` | `stem_id` | Mute a stem |
| `pan` | `stem_id`, `position` | Stereo pan (-1.0 to 1.0) |
| `reverse` | `stem_id` | Reverse the audio |
| `timestretch` | `stem_id`, `factor` | Time stretch without pitch change |

### 13.3 Application Order

Operations within a single commit are applied in array order. Commits are applied in topological order from root to `head`.

Decoders MUST support the `trim`, `gain`, `mute`, and `pan` operations. Support for `eq`, `fade`, `reverse`, and `timestretch` is OPTIONAL.

## 14. Streaming & Content Addressing

### 14.1 Merkle Tree Structure

Each chunk's data is divided into blocks of `DEFAULT_BLOCK_SIZE` (256 KiB). Each block is BLAKE3-hashed. Hashes are combined into a binary Merkle tree. The root hash serves as the Content Identifier (CID) for the chunk.

### 14.2 Content Identifiers

CIDs follow the multihash/CID specification:
- Hash function: BLAKE3 (multicodec 0x1e)
- CID version: 1
- Codec: raw (0x55)

### 14.3 HTTP Streaming

NEO files support HTTP range requests for progressive loading:
- Byte range `[0, 64)`: Header
- Byte range from chunk table: Individual chunks
- Servers SHOULD support HTTP/2 or HTTP/3 for multiplexed chunk fetching

### 14.4 IPFS Distribution

The Merkle tree structure makes NEO files natively compatible with IPFS:
- Each 256 KiB block can be stored as an IPFS block
- The root CID identifies the complete file
- Partial retrieval (e.g., just the PREV chunk) is possible via the chunk table offsets

## 15. Conformance Levels

### 15.1 NEO Basic

Minimum viable implementation:
- Read/write header and chunk table
- Read/write at least one STEM chunk with FLAC or Opus codec
- Read META chunk
- Stereo output

### 15.2 NEO Standard

Full consumer-grade implementation:
- All of NEO Basic
- DAC neural codec support
- Lossless residual reconstruction
- Preview LOD layer
- Temporal metadata
- C2PA verification

### 15.3 NEO Professional

Full professional implementation:
- All of NEO Standard
- Spatial audio rendering (binaural + surround)
- Non-destructive editing
- Merkle tree / content addressing
- Web3 rights
- IPFS P2P streaming

## 16. Security Considerations

1. **Buffer overflows**: Implementations MUST validate all size fields before allocating memory.
2. **Hash verification**: Implementations SHOULD verify BLAKE3 hashes before processing chunk data.
3. **C2PA trust**: C2PA verification depends on certificate trust chains; implementations MUST NOT treat unverified credentials as trusted.
4. **Web3 addresses**: Implementations MUST NOT automatically execute transactions. Wallet addresses are informational only.
5. **Neural codec models**: ONNX models SHOULD be loaded from trusted sources only. Model integrity SHOULD be verified via hash.
6. **Maximum file size**: Implementations SHOULD support files up to 4 GiB. Support for larger files is OPTIONAL in v0.1.
7. **Denial of service**: Implementations MUST set reasonable limits on stem count (≤8), chunk count (≤256), and individual chunk size (≤2 GiB).

## 17. IANA Considerations

### 17.1 Media Type Registration

- Type: `audio`
- Subtype: `neo`
- File extension: `.neo`
- Magic number: `0x4E454F21` at offset 0

### 17.2 Future Registration

A formal IANA media type registration will be submitted when the specification reaches v1.0 stability.

## 18. References

### 18.1 Normative References

- [RFC 2119] Key words for use in RFCs
- [BLAKE3] BLAKE3 cryptographic hash function — https://github.com/BLAKE3-team/BLAKE3
- [FLAC] Free Lossless Audio Codec — https://xiph.org/flac/
- [RFC 6716] Opus Audio Codec — https://www.rfc-editor.org/rfc/rfc6716
- [C2PA] Coalition for Content Provenance and Authenticity — https://c2pa.org/specifications/
- [CID] Content Identifiers — https://github.com/multiformats/cid

### 18.2 Informative References

- [DAC] Descript Audio Codec — https://github.com/descriptinc/descript-audio-codec
- [EnCodec] Meta EnCodec — https://github.com/facebookresearch/encodec
- [MKV] Matroska container — https://www.matroska.org/technical/specs/index.html
- [JSON-LD] JSON for Linked Data — https://json-ld.org/
- [IPFS] InterPlanetary File System — https://ipfs.io/
- [Ambisonics] Ambisonics — https://en.wikipedia.org/wiki/Ambisonics
- [HRTF] Head-Related Transfer Functions — MIT KEMAR dataset
- [NI Stems] Native Instruments Stems format — https://www.stems-music.com/

---

*This specification is placed in the public domain under the Unlicense. Anyone is free to implement, modify, and distribute implementations without restriction.*
