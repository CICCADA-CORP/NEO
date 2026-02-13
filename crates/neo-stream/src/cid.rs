//! Content Identifier (CID) generation for content-addressed storage.
//!
//! Implements CID generation as specified in SPECIFICATION.md Section 14.2.
//! Uses CIDv1 with BLAKE3 hashing and the raw multicodec.
//!
//! # Examples
//!
//! ```
//! use neo_stream::cid::{compute_cid, CidVersion};
//!
//! let data = b"hello neo";
//! let cid = compute_cid(data);
//! assert_eq!(cid.version, CidVersion::V1);
//! assert_eq!(cid.hash, *blake3::hash(data).as_bytes());
//! ```

use serde::{Deserialize, Serialize};

use crate::error::{Result, StreamError};

/// CID version.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CidVersion {
    /// CIDv1 — self-describing content identifier.
    V1 = 1,
}

/// Multicodec identifier for raw binary data.
const RAW_CODEC: u64 = 0x55;

/// Multicodec identifier for the BLAKE3 hash function.
const BLAKE3_HASH_FUNCTION: u64 = 0x1e;

/// BLAKE3 digest length in bytes.
const BLAKE3_DIGEST_LEN: usize = 32;

/// A content identifier (CID) for content-addressed data.
///
/// Encodes the CID version, multicodec, hash function, and hash digest,
/// following the CIDv1 structure used by IPFS and compatible systems.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ContentId {
    /// CID version (always V1).
    pub version: CidVersion,
    /// Multicodec for the data encoding (raw = 0x55).
    pub codec: u64,
    /// Multicodec for the hash function (BLAKE3 = 0x1e).
    pub hash_function: u64,
    /// The BLAKE3 hash digest.
    pub hash: [u8; 32],
}

/// Encode a `u64` as an unsigned varint (LEB128), appending bytes to `buf`.
fn encode_varint(mut value: u64, buf: &mut Vec<u8>) {
    loop {
        let mut byte = (value & 0x7F) as u8;
        value >>= 7;
        if value != 0 {
            byte |= 0x80;
        }
        buf.push(byte);
        if value == 0 {
            break;
        }
    }
}

/// Decode an unsigned varint (LEB128) from the front of `data`.
/// Returns `(value, bytes_consumed)`.
fn decode_varint(data: &[u8]) -> std::result::Result<(u64, usize), String> {
    let mut value: u64 = 0;
    let mut shift = 0u32;
    for (i, &byte) in data.iter().enumerate() {
        let low_bits = (byte & 0x7F) as u64;
        value |= low_bits << shift;
        if byte & 0x80 == 0 {
            return Ok((value, i + 1));
        }
        shift += 7;
        if shift >= 64 {
            return Err("varint too long".to_string());
        }
    }
    Err("unexpected end of varint data".to_string())
}

impl ContentId {
    /// Serialize this CID to its binary representation.
    ///
    /// Layout: `<version-varint> <codec-varint> <hash-fn-varint> <digest-len-varint> <digest>`
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        // CID version
        encode_varint(self.version as u64, &mut buf);
        // Content multicodec
        encode_varint(self.codec, &mut buf);
        // Multihash: hash function code
        encode_varint(self.hash_function, &mut buf);
        // Multihash: digest length
        encode_varint(BLAKE3_DIGEST_LEN as u64, &mut buf);
        // Multihash: digest
        buf.extend_from_slice(&self.hash);
        buf
    }

    /// Deserialize a CID from its binary representation.
    ///
    /// # Errors
    ///
    /// Returns [`StreamError::CidError`] if the bytes cannot be parsed.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        let mut offset = 0;

        // Version
        let (version_raw, consumed) =
            decode_varint(&bytes[offset..]).map_err(|e| StreamError::CidError(e.clone()))?;
        offset += consumed;
        if version_raw != 1 {
            return Err(StreamError::CidError(format!(
                "unsupported CID version: {version_raw}"
            )));
        }

        // Codec
        let (codec, consumed) =
            decode_varint(&bytes[offset..]).map_err(|e| StreamError::CidError(e.clone()))?;
        offset += consumed;

        // Hash function
        let (hash_function, consumed) =
            decode_varint(&bytes[offset..]).map_err(|e| StreamError::CidError(e.clone()))?;
        offset += consumed;

        // Digest length
        let (digest_len, consumed) =
            decode_varint(&bytes[offset..]).map_err(|e| StreamError::CidError(e.clone()))?;
        offset += consumed;

        if digest_len != BLAKE3_DIGEST_LEN as u64 {
            return Err(StreamError::CidError(format!(
                "unexpected digest length: {digest_len} (expected {BLAKE3_DIGEST_LEN})"
            )));
        }

        if offset + BLAKE3_DIGEST_LEN > bytes.len() {
            return Err(StreamError::CidError(
                "not enough bytes for digest".to_string(),
            ));
        }

        let mut hash = [0u8; 32];
        hash.copy_from_slice(&bytes[offset..offset + BLAKE3_DIGEST_LEN]);

        Ok(Self {
            version: CidVersion::V1,
            codec,
            hash_function,
            hash,
        })
    }
}

impl std::fmt::Display for ContentId {
    /// Display the CID as a lowercase hex string of its binary encoding.
    ///
    /// A production implementation would use base32-lower or base58btc,
    /// but hex is used here for simplicity and debuggability.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let bytes = self.to_bytes();
        for b in &bytes {
            write!(f, "{b:02x}")?;
        }
        Ok(())
    }
}

/// Compute a [`ContentId`] for the given data using CIDv1 with BLAKE3.
///
/// The CID uses the raw multicodec (`0x55`) and BLAKE3 hash function (`0x1e`).
pub fn compute_cid(data: &[u8]) -> ContentId {
    let hash = *blake3::hash(data).as_bytes();
    ContentId {
        version: CidVersion::V1,
        codec: RAW_CODEC,
        hash_function: BLAKE3_HASH_FUNCTION,
        hash,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_cid() {
        let data = b"hello neo stream";
        let cid = compute_cid(data);

        assert_eq!(cid.version, CidVersion::V1);
        assert_eq!(cid.codec, RAW_CODEC);
        assert_eq!(cid.hash_function, BLAKE3_HASH_FUNCTION);
        assert_eq!(cid.hash, *blake3::hash(data).as_bytes());
    }

    #[test]
    fn test_round_trip() {
        let data = b"round trip test data";
        let cid = compute_cid(data);

        let bytes = cid.to_bytes();
        let decoded = ContentId::from_bytes(&bytes).unwrap();

        assert_eq!(cid, decoded);
    }

    #[test]
    fn test_known_hash_verification() {
        let data = b"known";
        let expected_hash = *blake3::hash(data).as_bytes();
        let cid = compute_cid(data);
        assert_eq!(cid.hash, expected_hash);
    }

    #[test]
    fn test_different_data_different_cid() {
        let cid1 = compute_cid(b"alpha");
        let cid2 = compute_cid(b"beta");
        assert_ne!(cid1.hash, cid2.hash);
        assert_ne!(cid1.to_string(), cid2.to_string());
    }

    #[test]
    fn test_to_string_is_hex() {
        let cid = compute_cid(b"hex test");
        let s = cid.to_string();
        // Should be valid hex characters only
        assert!(s.chars().all(|c| c.is_ascii_hexdigit()));
        // Should be non-empty
        assert!(!s.is_empty());
    }

    #[test]
    fn test_from_bytes_invalid_version() {
        // Version 2 is not supported
        let mut bytes = Vec::new();
        encode_varint(2, &mut bytes); // bad version
        encode_varint(RAW_CODEC, &mut bytes);
        encode_varint(BLAKE3_HASH_FUNCTION, &mut bytes);
        encode_varint(32, &mut bytes);
        bytes.extend_from_slice(&[0u8; 32]);

        let err = ContentId::from_bytes(&bytes).unwrap_err();
        assert!(err.to_string().contains("unsupported CID version"));
    }

    #[test]
    fn test_from_bytes_truncated() {
        let cid = compute_cid(b"truncated");
        let bytes = cid.to_bytes();
        // Truncate the digest
        let truncated = &bytes[..bytes.len() - 10];
        let err = ContentId::from_bytes(truncated).unwrap_err();
        assert!(err.to_string().contains("not enough bytes"));
    }

    #[test]
    fn test_from_bytes_bad_digest_len() {
        let mut bytes = Vec::new();
        encode_varint(1, &mut bytes);
        encode_varint(RAW_CODEC, &mut bytes);
        encode_varint(BLAKE3_HASH_FUNCTION, &mut bytes);
        encode_varint(64, &mut bytes); // wrong digest length
        bytes.extend_from_slice(&[0u8; 64]);

        let err = ContentId::from_bytes(&bytes).unwrap_err();
        assert!(err.to_string().contains("unexpected digest length"));
    }

    #[test]
    fn test_empty_data_cid() {
        // Empty data should still produce a valid CID (BLAKE3 of empty input)
        let cid = compute_cid(b"");
        assert_eq!(cid.hash, *blake3::hash(b"").as_bytes());
        let bytes = cid.to_bytes();
        let decoded = ContentId::from_bytes(&bytes).unwrap();
        assert_eq!(cid, decoded);
    }

    #[test]
    fn test_varint_encoding() {
        // Test small value
        let mut buf = Vec::new();
        encode_varint(1, &mut buf);
        assert_eq!(buf, vec![1]);

        // Test 0x55 (raw codec)
        let mut buf = Vec::new();
        encode_varint(0x55, &mut buf);
        let (val, _) = decode_varint(&buf).unwrap();
        assert_eq!(val, 0x55);

        // Test 0x1e (BLAKE3)
        let mut buf = Vec::new();
        encode_varint(0x1e, &mut buf);
        let (val, _) = decode_varint(&buf).unwrap();
        assert_eq!(val, 0x1e);
    }

    #[test]
    fn test_cid_serialization_serde() {
        let cid = compute_cid(b"serde test");
        let json = serde_json::to_string(&cid).unwrap();
        let deserialized: ContentId = serde_json::from_str(&json).unwrap();
        assert_eq!(cid, deserialized);
    }
}
