//! Fuzz target for write → read round-trip.
//!
//! Uses structured fuzzer input to generate valid NEO parameters,
//! writes a file, reads it back, and verifies consistency.

#![no_main]

use libfuzzer_sys::fuzz_target;
use neo_format::{CodecId, NeoReader, NeoWriter, StemConfig, StemLabel};

fuzz_target!(|data: &[u8]| {
    // Need at least 16 bytes to derive meaningful parameters
    if data.len() < 16 {
        return;
    }

    // Derive parameters from fuzz input
    let stem_count = (data[0] % 8).max(1); // 1-8 stems
    let sample_rate = match data[1] % 3 {
        0 => 44100,
        1 => 48000,
        _ => 96000,
    };
    let channels = (data[2] % 2) + 1; // 1 or 2

    let duration_us = u64::from_le_bytes(data[4..12].try_into().unwrap_or([0; 8]));

    // Create writer with derived parameters
    let mut writer = NeoWriter::new(sample_rate, stem_count);
    writer.set_duration_us(duration_us);

    // Generate stem configs with small PCM data
    let mut configs = Vec::new();

    for i in 0..stem_count {
        let config = StemConfig {
            stem_id: i,
            label: StemLabel::Custom(format!("fuzz_{}", i)),
            codec: CodecId::Pcm,
            channels,
            sample_rate,
            bit_depth: 32,
            bitrate_kbps: 0,
            sample_count: 4,
        };

        // 4 samples × channels × 4 bytes each = small valid PCM
        let pcm_size = 4 * channels as usize * 4;
        let pcm: Vec<u8> = (0..pcm_size).map(|j| data[(12 + j) % data.len()]).collect();

        if writer.add_stem(config.clone(), pcm).is_err() {
            return;
        }

        configs.push(config);
    }

    // Write to temp file
    let tmpdir = match tempfile::tempdir() {
        Ok(d) => d,
        Err(_) => return,
    };
    let path = tmpdir.path().join("fuzz_round_trip.neo");

    if writer.finalize(&path).is_err() {
        return;
    }

    // Read back and verify basics
    if let Ok(reader) = NeoReader::open(&path) {
        assert_eq!(reader.header().version, 1);
        assert_eq!(reader.header().stem_count, stem_count);
        assert_eq!(reader.header().sample_rate, sample_rate);
        assert_eq!(reader.stem_configs().len(), configs.len());
    }
});
