//! Cross-crate integration tests: neo-format + neo-codec.
//!
//! Tests the full pipeline: create NEO file with PCM codec via NeoWriter →
//! read back via NeoReader → decode → verify data integrity.

use neo_format::{CodecId, FeatureFlags, NeoReader, NeoWriter, StemConfig, StemLabel};

/// Helper: generate a sine wave as f32 samples.
fn generate_sine(freq: f32, sample_rate: u32, duration_secs: f32) -> Vec<f32> {
    let count = (sample_rate as f32 * duration_secs) as usize;
    (0..count)
        .map(|i| (2.0 * std::f32::consts::PI * freq * i as f32 / sample_rate as f32).sin())
        .collect()
}

/// Helper: encode f32 samples to PCM bytes (little-endian f32).
fn pcm_encode(samples: &[f32]) -> Vec<u8> {
    samples.iter().flat_map(|s| s.to_le_bytes()).collect()
}

/// Helper: decode PCM bytes back to f32 samples.
fn pcm_decode(data: &[u8]) -> Vec<f32> {
    data.chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

#[test]
fn test_full_pipeline_single_stem() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("single.neo");

    let samples = generate_sine(440.0, 48000, 0.5);
    let pcm_bytes = pcm_encode(&samples);

    let config = StemConfig {
        stem_id: 0,
        label: StemLabel::Mix,
        codec: CodecId::Pcm,
        channels: 1,
        sample_rate: 48000,
        bit_depth: 32,
        bitrate_kbps: 0,
        sample_count: samples.len() as u64,
    };

    // Write
    let mut writer = NeoWriter::new(48000, 1);
    writer.add_stem(config, pcm_bytes.clone()).unwrap();
    writer.set_duration_us(500_000);
    writer.finalize(&path).unwrap();

    // Read back
    let mut reader = NeoReader::open(&path).unwrap();
    assert_eq!(reader.header().stem_count, 1);
    assert_eq!(reader.header().sample_rate, 48000);
    assert_eq!(reader.stem_configs().len(), 1);
    assert_eq!(reader.stem_configs()[0].label.as_str(), "mix");

    let read_data = reader.read_stem_data(0).unwrap();
    assert_eq!(read_data.len(), pcm_bytes.len());

    // Decode and verify bit-perfect round-trip
    let decoded = pcm_decode(&read_data);
    assert_eq!(decoded.len(), samples.len());
    for (i, (orig, dec)) in samples.iter().zip(decoded.iter()).enumerate() {
        assert_eq!(orig.to_bits(), dec.to_bits(), "Sample {} mismatch", i);
    }
}

#[test]
fn test_full_pipeline_multi_stem() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("multi.neo");

    let vocals = generate_sine(440.0, 44100, 1.0);
    let drums = generate_sine(220.0, 44100, 1.0);
    let bass = generate_sine(110.0, 44100, 1.0);

    let stems_data: Vec<(&[f32], StemLabel)> = vec![
        (&vocals, StemLabel::Vocals),
        (&drums, StemLabel::Drums),
        (&bass, StemLabel::Bass),
    ];

    let configs: Vec<StemConfig> = stems_data
        .iter()
        .enumerate()
        .map(|(i, (samples, label))| StemConfig {
            stem_id: i as u8,
            label: label.clone(),
            codec: CodecId::Pcm,
            channels: 1,
            sample_rate: 44100,
            bit_depth: 32,
            bitrate_kbps: 0,
            sample_count: samples.len() as u64,
        })
        .collect();

    let mut writer = NeoWriter::new(44100, 3);
    for (i, (samples, _)) in stems_data.iter().enumerate() {
        writer
            .add_stem(configs[i].clone(), pcm_encode(samples))
            .unwrap();
    }
    writer.set_duration_us(1_000_000);
    writer.finalize(&path).unwrap();

    // Read and verify each stem independently
    let mut reader = NeoReader::open(&path).unwrap();
    assert_eq!(reader.header().stem_count, 3);
    assert_eq!(reader.stem_configs().len(), 3);

    for (i, (original_samples, label)) in stems_data.iter().enumerate() {
        let config = &reader.stem_configs()[i];
        assert_eq!(config.label.as_str(), label.as_str());

        let data = reader.read_stem_data(i as u8).unwrap();
        let decoded = pcm_decode(&data);
        assert_eq!(decoded.len(), original_samples.len());
        assert_eq!(
            blake3::hash(&pcm_encode(original_samples)),
            blake3::hash(&data),
            "Stem {} hash mismatch",
            i
        );
    }
}

#[test]
fn test_metadata_round_trip() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("meta.neo");

    let metadata_json = r#"{"@context":"https://schema.org/","@type":"MusicRecording","name":"Test Song","byArtist":{"@type":"Person","name":"Test Artist"}}"#;
    let pcm_data = vec![0u8; 48000 * 4];

    let config = StemConfig {
        stem_id: 0,
        label: StemLabel::Mix,
        codec: CodecId::Pcm,
        channels: 1,
        sample_rate: 48000,
        bit_depth: 32,
        bitrate_kbps: 0,
        sample_count: 48000,
    };

    let mut writer = NeoWriter::new(48000, 1);
    writer.add_stem(config, pcm_data).unwrap();
    writer.set_metadata(metadata_json.to_string());
    writer.set_duration_us(1_000_000);
    writer.finalize(&path).unwrap();

    let mut reader = NeoReader::open(&path).unwrap();
    let meta = reader.read_metadata().unwrap();
    assert!(meta.is_some());
    let meta_str = meta.unwrap();
    assert_eq!(meta_str, metadata_json);
}

#[test]
fn test_max_stems() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("max_stems.neo");

    let stem_count = 8u8;

    let labels = [
        StemLabel::Vocals,
        StemLabel::Drums,
        StemLabel::Bass,
        StemLabel::Melody,
        StemLabel::Instrumental,
        StemLabel::Mix,
        StemLabel::Custom("fx1".to_string()),
        StemLabel::Custom("fx2".to_string()),
    ];

    let pcm_data = vec![0u8; 4800 * 4];

    let mut writer = NeoWriter::new(48000, stem_count);
    for i in 0..stem_count {
        let config = StemConfig {
            stem_id: i,
            label: labels[i as usize].clone(),
            codec: CodecId::Pcm,
            channels: 1,
            sample_rate: 48000,
            bit_depth: 32,
            bitrate_kbps: 0,
            sample_count: 4800,
        };
        writer.add_stem(config, pcm_data.clone()).unwrap();
    }
    writer.set_duration_us(100_000);
    writer.finalize(&path).unwrap();

    let reader = NeoReader::open(&path).unwrap();
    assert_eq!(reader.header().stem_count, 8);
    assert_eq!(reader.stem_configs().len(), 8);
}

#[test]
fn test_empty_stem_data() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("empty_stem.neo");

    let config = StemConfig {
        stem_id: 0,
        label: StemLabel::Mix,
        codec: CodecId::Pcm,
        channels: 1,
        sample_rate: 48000,
        bit_depth: 32,
        bitrate_kbps: 0,
        sample_count: 0,
    };

    let mut writer = NeoWriter::new(48000, 1);
    writer.add_stem(config, vec![]).unwrap();
    writer.set_duration_us(0);
    writer.finalize(&path).unwrap();

    let mut reader = NeoReader::open(&path).unwrap();
    let data = reader.read_stem_data(0).unwrap();
    assert!(data.is_empty());
}

#[test]
fn test_allocation_limit() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("alloc_test.neo");

    // Create a file with a reasonably sized stem (192KB of PCM data)
    let pcm_data = vec![0u8; 48000 * 4]; // 192KB

    let config = StemConfig {
        stem_id: 0,
        label: StemLabel::Mix,
        codec: CodecId::Pcm,
        channels: 1,
        sample_rate: 48000,
        bit_depth: 32,
        bitrate_kbps: 0,
        sample_count: 48000,
    };

    let mut writer = NeoWriter::new(48000, 1);
    writer.add_stem(config, pcm_data).unwrap();
    writer.set_duration_us(1_000_000);
    writer.finalize(&path).unwrap();

    // Set a very low allocation limit and verify it blocks reading
    let mut reader = NeoReader::open(&path).unwrap();
    reader.set_allocation_limit(1024); // 1KB limit
    let result = reader.read_stem_data(0);
    assert!(
        result.is_err(),
        "Should fail with allocation limit of 1KB for 192KB stem"
    );
}

#[test]
fn test_temporal_metadata_round_trip() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("temporal.neo");

    let temporal_json = r#"{"bpm":120.0,"key":"C major","chords":[{"time":0.0,"label":"C"}]}"#;

    let config = StemConfig {
        stem_id: 0,
        label: StemLabel::Mix,
        codec: CodecId::Pcm,
        channels: 1,
        sample_rate: 48000,
        bit_depth: 32,
        bitrate_kbps: 0,
        sample_count: 4800,
    };

    let mut writer = NeoWriter::new(48000, 1);
    writer.add_stem(config, vec![0u8; 4800 * 4]).unwrap();
    writer.set_temporal(temporal_json.to_string());
    writer.set_duration_us(100_000);
    writer.finalize(&path).unwrap();

    let mut reader = NeoReader::open(&path).unwrap();
    assert!(reader
        .header()
        .feature_flags
        .has(FeatureFlags::TEMPORAL_META));
    let temporal = reader.read_temporal().unwrap();
    assert!(temporal.is_some());
    assert_eq!(temporal.unwrap(), temporal_json);
}

#[test]
fn test_spatial_metadata_round_trip() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("spatial.neo");

    let spatial_json =
        r#"{"objects":[{"id":0,"position":{"azimuth":30.0,"elevation":0.0,"distance":1.0}}]}"#;

    let config = StemConfig {
        stem_id: 0,
        label: StemLabel::Vocals,
        codec: CodecId::Pcm,
        channels: 1,
        sample_rate: 48000,
        bit_depth: 32,
        bitrate_kbps: 0,
        sample_count: 4800,
    };

    let mut writer = NeoWriter::new(48000, 1);
    writer.add_stem(config, vec![0u8; 4800 * 4]).unwrap();
    writer.set_spatial(spatial_json.to_string());
    writer.set_duration_us(100_000);
    writer.finalize(&path).unwrap();

    let mut reader = NeoReader::open(&path).unwrap();
    assert!(reader.header().feature_flags.has(FeatureFlags::SPATIAL));
    let spatial = reader.read_spatial().unwrap();
    assert!(spatial.is_some());
    assert_eq!(spatial.unwrap(), spatial_json);
}

#[test]
fn test_edit_history_round_trip() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("edit.neo");

    let edit_json = r#"{"commits":[{"id":"abc123","message":"initial","ops":[]}]}"#;

    let config = StemConfig {
        stem_id: 0,
        label: StemLabel::Mix,
        codec: CodecId::Pcm,
        channels: 1,
        sample_rate: 44100,
        bit_depth: 32,
        bitrate_kbps: 0,
        sample_count: 4410,
    };

    let mut writer = NeoWriter::new(44100, 1);
    writer.add_stem(config, vec![0u8; 4410 * 4]).unwrap();
    writer.set_edit_history(edit_json.to_string());
    writer.set_duration_us(100_000);
    writer.finalize(&path).unwrap();

    let mut reader = NeoReader::open(&path).unwrap();
    assert!(reader
        .header()
        .feature_flags
        .has(FeatureFlags::EDIT_HISTORY));
    let edit = reader.read_edit_history().unwrap();
    assert!(edit.is_some());
    assert_eq!(edit.unwrap(), edit_json);
}

#[test]
fn test_rights_metadata_round_trip() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("rights.neo");

    let rights_json =
        r#"{"splits":[{"address":"0xAAAA","chain":"ethereum","share":0.6}],"license":"CC-BY-4.0"}"#;

    let config = StemConfig {
        stem_id: 0,
        label: StemLabel::Mix,
        codec: CodecId::Pcm,
        channels: 1,
        sample_rate: 48000,
        bit_depth: 32,
        bitrate_kbps: 0,
        sample_count: 4800,
    };

    let mut writer = NeoWriter::new(48000, 1);
    writer.add_stem(config, vec![0u8; 4800 * 4]).unwrap();
    writer.set_rights(rights_json.to_string());
    writer.set_duration_us(100_000);
    writer.finalize(&path).unwrap();

    let mut reader = NeoReader::open(&path).unwrap();
    assert!(reader.header().feature_flags.has(FeatureFlags::WEB3_RIGHTS));
    let rights = reader.read_rights().unwrap();
    assert!(rights.is_some());
    assert_eq!(rights.unwrap(), rights_json);
}

#[test]
fn test_credentials_round_trip() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("creds.neo");

    // Simulate raw JUMBF bytes for C2PA credentials
    let cred_bytes: Vec<u8> = (0..128).collect();

    let config = StemConfig {
        stem_id: 0,
        label: StemLabel::Mix,
        codec: CodecId::Pcm,
        channels: 1,
        sample_rate: 48000,
        bit_depth: 32,
        bitrate_kbps: 0,
        sample_count: 4800,
    };

    let mut writer = NeoWriter::new(48000, 1);
    writer.add_stem(config, vec![0u8; 4800 * 4]).unwrap();
    writer.set_credentials(cred_bytes.clone());
    writer.set_duration_us(100_000);
    writer.finalize(&path).unwrap();

    let mut reader = NeoReader::open(&path).unwrap();
    assert!(reader.header().feature_flags.has(FeatureFlags::C2PA));
    let creds = reader.read_credentials().unwrap();
    assert!(creds.is_some());
    assert_eq!(creds.unwrap(), cred_bytes);
}

#[test]
fn test_preview_round_trip() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("preview.neo");

    let preview_data = vec![0xCC; 256]; // Fake preview audio

    let config = StemConfig {
        stem_id: 0,
        label: StemLabel::Mix,
        codec: CodecId::Pcm,
        channels: 1,
        sample_rate: 48000,
        bit_depth: 32,
        bitrate_kbps: 0,
        sample_count: 4800,
    };

    let mut writer = NeoWriter::new(48000, 1);
    writer.add_stem(config, vec![0u8; 4800 * 4]).unwrap();
    writer.set_preview(preview_data.clone());
    writer.set_duration_us(100_000);
    writer.finalize(&path).unwrap();

    let mut reader = NeoReader::open(&path).unwrap();
    assert!(reader.header().feature_flags.has(FeatureFlags::PREVIEW_LOD));
    let preview = reader.read_preview().unwrap();
    assert!(preview.is_some());
    assert_eq!(preview.unwrap(), preview_data);
}

#[test]
fn test_full_file_with_all_chunk_types() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("full.neo");

    let pcm_data = vec![0u8; 4800 * 4];
    let metadata_json = r#"{"@context":"https://schema.org","name":"Full Test"}"#;
    let temporal_json = r#"{"bpm":128.0,"key":"Am"}"#;
    let spatial_json = r#"{"objects":[]}"#;
    let edit_json = r#"{"commits":[]}"#;
    let rights_json = r#"{"splits":[]}"#;
    let cred_bytes = vec![0xDD; 64];
    let preview_data = vec![0xEE; 32];

    let config = StemConfig {
        stem_id: 0,
        label: StemLabel::Vocals,
        codec: CodecId::Pcm,
        channels: 2,
        sample_rate: 48000,
        bit_depth: 32,
        bitrate_kbps: 0,
        sample_count: 4800,
    };

    let mut writer = NeoWriter::new(48000, 1);
    writer.add_stem(config, pcm_data.clone()).unwrap();
    writer.set_metadata(metadata_json.to_string());
    writer.set_temporal(temporal_json.to_string());
    writer.set_spatial(spatial_json.to_string());
    writer.set_edit_history(edit_json.to_string());
    writer.set_rights(rights_json.to_string());
    writer.set_credentials(cred_bytes.clone());
    writer.set_preview(preview_data.clone());
    writer.set_duration_us(100_000);
    writer.finalize(&path).unwrap();

    // Verify all chunk types are readable
    let mut reader = NeoReader::open(&path).unwrap();

    // All feature flags should be set
    let flags = reader.header().feature_flags;
    assert!(flags.has(FeatureFlags::TEMPORAL_META));
    assert!(flags.has(FeatureFlags::SPATIAL));
    assert!(flags.has(FeatureFlags::EDIT_HISTORY));
    assert!(flags.has(FeatureFlags::WEB3_RIGHTS));
    assert!(flags.has(FeatureFlags::C2PA));
    assert!(flags.has(FeatureFlags::PREVIEW_LOD));

    // Read all chunk types
    assert_eq!(reader.read_metadata().unwrap().unwrap(), metadata_json);
    assert_eq!(reader.read_temporal().unwrap().unwrap(), temporal_json);
    assert_eq!(reader.read_spatial().unwrap().unwrap(), spatial_json);
    assert_eq!(reader.read_edit_history().unwrap().unwrap(), edit_json);
    assert_eq!(reader.read_rights().unwrap().unwrap(), rights_json);
    assert_eq!(reader.read_credentials().unwrap().unwrap(), cred_bytes);
    assert_eq!(reader.read_preview().unwrap().unwrap(), preview_data);

    // Verify stem data integrity
    let stem_data = reader.read_stem_data(0).unwrap();
    assert_eq!(stem_data, pcm_data);
}

#[test]
fn test_stem_label_preservation() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("labels.neo");

    let labels = [
        StemLabel::Vocals,
        StemLabel::Drums,
        StemLabel::Bass,
        StemLabel::Melody,
    ];

    let mut writer = NeoWriter::new(44100, 4);
    for (i, label) in labels.iter().enumerate() {
        let config = StemConfig {
            stem_id: i as u8,
            label: label.clone(),
            codec: CodecId::Pcm,
            channels: 1,
            sample_rate: 44100,
            bit_depth: 32,
            bitrate_kbps: 0,
            sample_count: 0,
        };
        writer.add_stem(config, vec![]).unwrap();
    }
    writer.finalize(&path).unwrap();

    let reader = NeoReader::open(&path).unwrap();
    let configs = reader.stem_configs();
    assert_eq!(configs.len(), 4);
    for (i, label) in labels.iter().enumerate() {
        assert_eq!(
            configs[i].label.as_str(),
            label.as_str(),
            "Label mismatch at stem {}",
            i
        );
    }
}

#[test]
fn test_blake3_integrity_across_stems() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("integrity.neo");

    // Create unique data patterns per stem for strong integrity checking
    let stem0_data: Vec<u8> = (0..1000).map(|i| (i % 256) as u8).collect();
    let stem1_data: Vec<u8> = (0..2000).map(|i| ((i * 7 + 13) % 256) as u8).collect();

    let config0 = StemConfig {
        stem_id: 0,
        label: StemLabel::Vocals,
        codec: CodecId::Pcm,
        channels: 1,
        sample_rate: 44100,
        bit_depth: 16,
        bitrate_kbps: 0,
        sample_count: 500,
    };

    let config1 = StemConfig {
        stem_id: 1,
        label: StemLabel::Drums,
        codec: CodecId::Pcm,
        channels: 1,
        sample_rate: 44100,
        bit_depth: 16,
        bitrate_kbps: 0,
        sample_count: 1000,
    };

    let mut writer = NeoWriter::new(44100, 2);
    writer.add_stem(config0, stem0_data.clone()).unwrap();
    writer.add_stem(config1, stem1_data.clone()).unwrap();
    writer.finalize(&path).unwrap();

    // The NeoReader internally verifies BLAKE3 checksums on open,
    // so if we get here without error, integrity is valid.
    let mut reader = NeoReader::open(&path).unwrap();

    // Additionally verify byte-for-byte equality of the read-back data
    let read0 = reader.read_stem_data(0).unwrap();
    let read1 = reader.read_stem_data(1).unwrap();

    assert_eq!(read0, stem0_data);
    assert_eq!(read1, stem1_data);
}

#[test]
fn test_codec_id_preservation() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("codecs.neo");

    // Write stems with different codec IDs and verify they round-trip
    let codecs = [CodecId::Pcm, CodecId::Dac, CodecId::Flac, CodecId::Opus];

    let mut writer = NeoWriter::new(48000, 4);
    for (i, codec) in codecs.iter().enumerate() {
        let config = StemConfig {
            stem_id: i as u8,
            label: StemLabel::Custom(format!("stem_{}", i)),
            codec: *codec,
            channels: 2,
            sample_rate: 48000,
            bit_depth: 24,
            bitrate_kbps: 128,
            sample_count: 0,
        };
        writer.add_stem(config, vec![0u8; 64]).unwrap();
    }
    writer.finalize(&path).unwrap();

    let reader = NeoReader::open(&path).unwrap();
    let configs = reader.stem_configs();
    assert_eq!(configs.len(), 4);
    for (i, codec) in codecs.iter().enumerate() {
        assert_eq!(configs[i].codec, *codec, "Codec mismatch at stem {}", i);
    }
}

#[test]
fn test_stereo_stem_data() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("stereo.neo");

    // Generate interleaved stereo samples (L/R alternating)
    let sample_count: usize = 4800;
    let mut stereo_samples = Vec::with_capacity(sample_count * 2);
    for i in 0..sample_count {
        let t = i as f32 / 48000.0;
        let left = (2.0 * std::f32::consts::PI * 440.0 * t).sin();
        let right = (2.0 * std::f32::consts::PI * 880.0 * t).sin();
        stereo_samples.push(left);
        stereo_samples.push(right);
    }
    let pcm_bytes = pcm_encode(&stereo_samples);

    let config = StemConfig {
        stem_id: 0,
        label: StemLabel::Mix,
        codec: CodecId::Pcm,
        channels: 2,
        sample_rate: 48000,
        bit_depth: 32,
        bitrate_kbps: 0,
        sample_count: sample_count as u64,
    };

    let mut writer = NeoWriter::new(48000, 1);
    writer.add_stem(config, pcm_bytes.clone()).unwrap();
    writer.set_duration_us(100_000);
    writer.finalize(&path).unwrap();

    let mut reader = NeoReader::open(&path).unwrap();
    let cfg = &reader.stem_configs()[0];
    assert_eq!(cfg.channels, 2);

    let read_data = reader.read_stem_data(0).unwrap();
    let decoded = pcm_decode(&read_data);
    assert_eq!(decoded.len(), stereo_samples.len());

    // Verify left and right channels are preserved
    for i in 0..sample_count {
        let orig_l = stereo_samples[i * 2];
        let orig_r = stereo_samples[i * 2 + 1];
        let dec_l = decoded[i * 2];
        let dec_r = decoded[i * 2 + 1];
        assert_eq!(orig_l.to_bits(), dec_l.to_bits(), "L sample {} mismatch", i);
        assert_eq!(orig_r.to_bits(), dec_r.to_bits(), "R sample {} mismatch", i);
    }
}
