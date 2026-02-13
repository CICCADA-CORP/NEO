//! PCM codec — raw uncompressed audio. Stores f32 samples as little-endian bytes.
//! Used as lossless baseline and for testing the full encode/decode pipeline.

use crate::{AudioCodec, CodecError, EncodeConfig};

/// Raw PCM codec — stores f32 samples as little-endian bytes with zero compression.
///
/// This is the simplest codec and serves as the lossless baseline for testing.
/// Each f32 sample occupies exactly 4 bytes in little-endian order.
pub struct PcmCodec;

impl AudioCodec for PcmCodec {
    /// Encode PCM f32 samples to raw little-endian bytes.
    ///
    /// Each sample is converted to its 4-byte LE representation and concatenated.
    /// The output length is always `pcm.len() * 4` bytes.
    fn encode(&self, pcm: &[f32], _config: &EncodeConfig) -> Result<Vec<u8>, CodecError> {
        let mut buf = Vec::with_capacity(pcm.len() * 4);
        for &sample in pcm {
            buf.extend_from_slice(&sample.to_le_bytes());
        }
        Ok(buf)
    }

    /// Decode raw little-endian bytes back to PCM f32 samples.
    ///
    /// Reads 4 bytes at a time and converts each group to an f32 value.
    /// Returns an error if the input length is not a multiple of 4.
    fn decode(&self, data: &[u8], _config: &EncodeConfig) -> Result<Vec<f32>, CodecError> {
        #[allow(clippy::manual_is_multiple_of)]
        if data.len() % 4 != 0 {
            return Err(CodecError::DecodingError(format!(
                "PCM data length {} is not a multiple of 4 bytes",
                data.len()
            )));
        }
        let samples: Vec<f32> = data
            .chunks_exact(4)
            .map(|chunk| {
                let bytes: [u8; 4] = chunk.try_into().expect("chunk is exactly 4 bytes");
                f32::from_le_bytes(bytes)
            })
            .collect();
        Ok(samples)
    }

    /// Return the codec identifier for raw PCM.
    fn codec_id(&self) -> neo_format::CodecId {
        neo_format::CodecId::Pcm
    }

    /// Human-readable codec name.
    fn name(&self) -> &str {
        "PCM (Raw Uncompressed)"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper to create a default config for testing.
    fn test_config() -> EncodeConfig {
        EncodeConfig {
            sample_rate: 44100,
            channels: 1,
            bitrate_kbps: None,
        }
    }

    #[test]
    fn test_pcm_round_trip() {
        let codec = PcmCodec;
        let config = test_config();
        let original = vec![0.0_f32, 0.5, -0.5, 1.0, -1.0, 0.123_456_79];

        let encoded = codec.encode(&original, &config).unwrap();
        assert_eq!(encoded.len(), original.len() * 4);

        let decoded = codec.decode(&encoded, &config).unwrap();
        assert_eq!(original, decoded, "PCM round-trip must be bit-perfect");
    }

    #[test]
    fn test_pcm_empty_input() {
        let codec = PcmCodec;
        let config = test_config();
        let original: Vec<f32> = vec![];

        let encoded = codec.encode(&original, &config).unwrap();
        assert!(encoded.is_empty());

        let decoded = codec.decode(&encoded, &config).unwrap();
        assert!(decoded.is_empty());
    }

    #[test]
    fn test_pcm_single_sample() {
        let codec = PcmCodec;
        let config = test_config();
        let original = vec![std::f32::consts::PI];

        let encoded = codec.encode(&original, &config).unwrap();
        assert_eq!(encoded.len(), 4);

        let decoded = codec.decode(&encoded, &config).unwrap();
        assert_eq!(original, decoded);
    }

    #[test]
    fn test_pcm_large_buffer() {
        let codec = PcmCodec;
        let config = test_config();
        // Generate 1 second of 44.1kHz mono audio (a sine wave)
        let sample_count = 44100;
        let original: Vec<f32> = (0..sample_count)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 44100.0).sin())
            .collect();

        let encoded = codec.encode(&original, &config).unwrap();
        assert_eq!(encoded.len(), sample_count * 4);

        let decoded = codec.decode(&encoded, &config).unwrap();
        assert_eq!(original.len(), decoded.len());
        assert_eq!(
            original, decoded,
            "Large buffer round-trip must be bit-perfect"
        );
    }

    #[test]
    fn test_pcm_decode_invalid_length() {
        let codec = PcmCodec;
        let config = test_config();
        // 5 bytes is not a multiple of 4
        let bad_data = vec![0u8; 5];

        let result = codec.decode(&bad_data, &config);
        assert!(result.is_err());
        match result {
            Err(CodecError::DecodingError(msg)) => {
                assert!(msg.contains("not a multiple of 4"));
            }
            _ => panic!("Expected DecodingError"),
        }
    }

    #[test]
    fn test_pcm_special_float_values() {
        let codec = PcmCodec;
        let config = test_config();
        let original = vec![f32::INFINITY, f32::NEG_INFINITY, 0.0, -0.0];

        let encoded = codec.encode(&original, &config).unwrap();
        let decoded = codec.decode(&encoded, &config).unwrap();

        assert_eq!(decoded[0], f32::INFINITY);
        assert_eq!(decoded[1], f32::NEG_INFINITY);
        assert_eq!(decoded[2], 0.0);
        // -0.0 and 0.0 compare equal via ==, so check bits
        assert_eq!(decoded[3].to_bits(), (-0.0_f32).to_bits());
    }

    #[test]
    fn test_pcm_codec_id() {
        let codec = PcmCodec;
        assert_eq!(codec.codec_id(), neo_format::CodecId::Pcm);
    }

    #[test]
    fn test_pcm_name() {
        let codec = PcmCodec;
        assert_eq!(codec.name(), "PCM (Raw Uncompressed)");
    }
}
