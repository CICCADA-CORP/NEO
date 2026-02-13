//! Integration tests for the NEO CLI binary.
//!
//! Tests the full encode → info → decode round-trip using the `neo` binary,
//! verifying that a programmatically generated WAV file survives the pipeline
//! bit-perfectly when using the PCM codec.

use std::f32::consts::PI;
use std::path::Path;

use assert_cmd::Command;
use predicates::prelude::*;
use tempfile::TempDir;

// ──────────────────────── helpers ────────────────────────

/// Generate a mono 440 Hz sine wave at 44100 Hz for the given duration in seconds.
/// Returns the interleaved f32 samples.
fn generate_sine_wave(sample_rate: u32, frequency: f32, duration_secs: f32) -> Vec<f32> {
    let num_samples = (sample_rate as f32 * duration_secs) as usize;
    (0..num_samples)
        .map(|i| {
            let t = i as f32 / sample_rate as f32;
            (2.0 * PI * frequency * t).sin()
        })
        .collect()
}

/// Write a mono 32-bit float WAV file using `hound`.
fn write_wav_f32(path: &Path, samples: &[f32], sample_rate: u32) {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };
    let mut writer = hound::WavWriter::create(path, spec).expect("Failed to create WAV writer");
    for &s in samples {
        writer.write_sample(s).expect("Failed to write sample");
    }
    writer.finalize().expect("Failed to finalize WAV");
}

/// Read a mono 32-bit float WAV file and return the samples.
fn read_wav_f32(path: &Path) -> Vec<f32> {
    let reader = hound::WavReader::open(path).expect("Failed to open WAV for reading");
    let spec = reader.spec();
    assert_eq!(spec.channels, 1, "Expected mono WAV");
    match spec.sample_format {
        hound::SampleFormat::Float => reader
            .into_samples::<f32>()
            .map(|s| s.expect("Failed to read sample"))
            .collect(),
        hound::SampleFormat::Int => {
            let max_val = (1u32 << (spec.bits_per_sample - 1)) as f32;
            reader
                .into_samples::<i32>()
                .map(|s| s.expect("Failed to read sample") as f32 / max_val)
                .collect()
        }
    }
}

/// Get a `Command` for the `neo` CLI binary.
#[allow(deprecated)]
fn neo_cmd() -> Command {
    Command::cargo_bin("neo").expect("Failed to find `neo` binary")
}

// ──────────────────────── tests ─────────────────────────

#[test]
fn test_encode_decode_round_trip_pcm() {
    // 1. Create a temporary directory.
    let tmp = TempDir::new().expect("Failed to create temp dir");
    let tmp_path = tmp.path();

    // 2. Generate a 1-second 440 Hz sine wave at 44100 Hz (mono).
    let sample_rate = 44100u32;
    let original_samples = generate_sine_wave(sample_rate, 440.0, 1.0);
    let wav_path = tmp_path.join("sine.wav");
    write_wav_f32(&wav_path, &original_samples, sample_rate);

    // Sanity check: the WAV file was created and has the right sample count.
    assert!(wav_path.exists(), "WAV file should exist");
    let original_readback = read_wav_f32(&wav_path);
    assert_eq!(original_readback.len(), original_samples.len());

    // 3. Encode the WAV to a .neo file using the CLI.
    let neo_path = tmp_path.join("test.neo");
    neo_cmd()
        .args([
            "encode",
            wav_path.to_str().unwrap(),
            "-o",
            neo_path.to_str().unwrap(),
            "--stems",
            "test_tone",
            "--codec",
            "pcm",
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains("NEO Encoder"))
        .stdout(predicate::str::contains("Done!"));

    assert!(neo_path.exists(), ".neo file should exist");
    assert!(
        std::fs::metadata(&neo_path).unwrap().len() > 64,
        ".neo file should be larger than the 64-byte header"
    );

    // 4. Run `neo info` and verify expected output fields.
    neo_cmd()
        .args(["info", neo_path.to_str().unwrap()])
        .assert()
        .success()
        .stdout(predicate::str::contains("NEO File Information"))
        .stdout(predicate::str::contains("Magic:    NEO! (0x4E454F21)"))
        .stdout(predicate::str::contains("Version:  1"))
        .stdout(predicate::str::contains("Rate:     44100 Hz"))
        .stdout(predicate::str::contains("Stems:    1"))
        .stdout(predicate::str::contains("test_tone"))
        .stdout(predicate::str::contains("Pcm"))
        .stdout(predicate::str::contains("Integrity: Verified"));

    // 5. Run `neo info --json` and verify JSON structure.
    let info_json_output = neo_cmd()
        .args(["info", neo_path.to_str().unwrap(), "--json"])
        .assert()
        .success();
    let stdout_bytes = info_json_output.get_output().stdout.clone();
    let stdout_str = String::from_utf8(stdout_bytes).expect("Invalid UTF-8 in JSON output");
    let json_val: serde_json::Value =
        serde_json::from_str(&stdout_str).expect("Info --json output should be valid JSON");
    assert_eq!(json_val["header"]["sample_rate"], 44100);
    assert_eq!(json_val["header"]["stem_count"], 1);
    assert_eq!(json_val["stems"][0]["label"], "test_tone");

    // 6. Decode the .neo file back to WAV.
    let output_dir = tmp_path.join("decoded");
    neo_cmd()
        .args([
            "decode",
            neo_path.to_str().unwrap(),
            "-o",
            output_dir.to_str().unwrap(),
            "--all-stems",
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains("NEO Decoder"))
        .stdout(predicate::str::contains("Extracted"))
        .stdout(predicate::str::contains("Done!"));

    // 7. Read the decoded WAV and verify it matches the original (bit-perfect for PCM).
    let decoded_wav = output_dir.join("test_tone.wav");
    assert!(
        decoded_wav.exists(),
        "Decoded WAV should exist at {:?}",
        decoded_wav
    );

    let decoded_samples = read_wav_f32(&decoded_wav);
    assert_eq!(
        decoded_samples.len(),
        original_samples.len(),
        "Decoded sample count ({}) should match original ({})",
        decoded_samples.len(),
        original_samples.len()
    );

    // Bit-perfect comparison for PCM codec.
    for (i, (orig, dec)) in original_samples
        .iter()
        .zip(decoded_samples.iter())
        .enumerate()
    {
        assert!(
            (orig - dec).abs() < 1e-6,
            "Sample {} differs: original={}, decoded={} (delta={})",
            i,
            orig,
            dec,
            (orig - dec).abs()
        );
    }
}

#[test]
fn test_encode_rejects_unknown_codec() {
    let tmp = TempDir::new().expect("Failed to create temp dir");
    let wav_path = tmp.path().join("dummy.wav");
    let neo_path = tmp.path().join("dummy.neo");

    // Create a minimal WAV so the file exists.
    write_wav_f32(&wav_path, &[0.0; 100], 44100);

    neo_cmd()
        .args([
            "encode",
            wav_path.to_str().unwrap(),
            "-o",
            neo_path.to_str().unwrap(),
            "--codec",
            "mp3",
        ])
        .assert()
        .failure()
        .stderr(predicate::str::contains("Unknown codec"));
}

#[test]
fn test_info_rejects_nonexistent_file() {
    neo_cmd()
        .args(["info", "/tmp/nonexistent_file_abcdef.neo"])
        .assert()
        .failure();
}

#[test]
fn test_encode_multiple_stems() {
    let tmp = TempDir::new().expect("Failed to create temp dir");
    let tmp_path = tmp.path();

    // Generate two different sine waves as two stems.
    let samples_a = generate_sine_wave(44100, 440.0, 0.5);
    let samples_b = generate_sine_wave(44100, 880.0, 0.5);

    let wav_a = tmp_path.join("tone_a.wav");
    let wav_b = tmp_path.join("tone_b.wav");
    write_wav_f32(&wav_a, &samples_a, 44100);
    write_wav_f32(&wav_b, &samples_b, 44100);

    let neo_path = tmp_path.join("multi_stem.neo");

    // Encode with two stems.
    neo_cmd()
        .args([
            "encode",
            wav_a.to_str().unwrap(),
            wav_b.to_str().unwrap(),
            "-o",
            neo_path.to_str().unwrap(),
            "--stems",
            "vocals,bass",
            "--codec",
            "pcm",
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains("Stems:  2"));

    // Info should show both stems.
    neo_cmd()
        .args(["info", neo_path.to_str().unwrap()])
        .assert()
        .success()
        .stdout(predicate::str::contains("vocals"))
        .stdout(predicate::str::contains("bass"))
        .stdout(predicate::str::contains("Stems:    2"));

    // Decode all stems.
    let output_dir = tmp_path.join("decoded_multi");
    neo_cmd()
        .args([
            "decode",
            neo_path.to_str().unwrap(),
            "-o",
            output_dir.to_str().unwrap(),
            "--all-stems",
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains("Extracted 2 stem(s)"));

    // Verify both decoded WAVs exist and are correct length.
    let decoded_a = read_wav_f32(&output_dir.join("vocals.wav"));
    let decoded_b = read_wav_f32(&output_dir.join("bass.wav"));
    assert_eq!(decoded_a.len(), samples_a.len());
    assert_eq!(decoded_b.len(), samples_b.len());
}

#[test]
fn test_decode_single_stem() {
    let tmp = TempDir::new().expect("Failed to create temp dir");
    let tmp_path = tmp.path();

    let samples_a = generate_sine_wave(44100, 440.0, 0.25);
    let samples_b = generate_sine_wave(44100, 880.0, 0.25);

    let wav_a = tmp_path.join("vocal.wav");
    let wav_b = tmp_path.join("drum.wav");
    write_wav_f32(&wav_a, &samples_a, 44100);
    write_wav_f32(&wav_b, &samples_b, 44100);

    let neo_path = tmp_path.join("two_stem.neo");
    neo_cmd()
        .args([
            "encode",
            wav_a.to_str().unwrap(),
            wav_b.to_str().unwrap(),
            "-o",
            neo_path.to_str().unwrap(),
            "--stems",
            "vocals,drums",
        ])
        .assert()
        .success();

    // Extract only the "vocals" stem.
    let output_dir = tmp_path.join("single_out");
    neo_cmd()
        .args([
            "decode",
            neo_path.to_str().unwrap(),
            "-o",
            output_dir.to_str().unwrap(),
            "--stem",
            "vocals",
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains("Extracted 1 stem(s)"));

    assert!(output_dir.join("vocals.wav").exists());
}

#[test]
fn test_cli_help_works() {
    neo_cmd()
        .arg("--help")
        .assert()
        .success()
        .stdout(predicate::str::contains("audio format"))
        .stdout(predicate::str::contains("encode"))
        .stdout(predicate::str::contains("decode"))
        .stdout(predicate::str::contains("info"));
}

// ──────────────── Phase 3: Metadata integration tests ───────────────

#[test]
fn test_encode_with_metadata_file() {
    let tmp = TempDir::new().expect("Failed to create temp dir");
    let tmp_path = tmp.path();

    // Create a WAV file.
    let samples = generate_sine_wave(44100, 440.0, 0.5);
    let wav_path = tmp_path.join("tone.wav");
    write_wav_f32(&wav_path, &samples, 44100);

    // Create a JSON metadata file.
    let meta_json = serde_json::json!({
        "@context": ["https://schema.org", {"neo": "https://neo-audio.org/ns/"}],
        "@type": "MusicRecording",
        "name": "Test Track",
        "byArtist": {"@type": "Person", "name": "Test Artist"},
        "genre": "Electronic"
    });
    let meta_path = tmp_path.join("meta.json");
    std::fs::write(
        &meta_path,
        serde_json::to_string_pretty(&meta_json).unwrap(),
    )
    .unwrap();

    let neo_path = tmp_path.join("with_meta.neo");

    // Encode with metadata.
    neo_cmd()
        .args([
            "encode",
            wav_path.to_str().unwrap(),
            "-o",
            neo_path.to_str().unwrap(),
            "--stems",
            "vocals",
            "--codec",
            "pcm",
            "--metadata",
            meta_path.to_str().unwrap(),
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains("Metadata embedded"));

    // Verify info shows the metadata.
    neo_cmd()
        .args(["info", neo_path.to_str().unwrap()])
        .assert()
        .success()
        .stdout(predicate::str::contains("Test Track"))
        .stdout(predicate::str::contains("Test Artist"));

    // Verify JSON info includes metadata.
    let output = neo_cmd()
        .args(["info", neo_path.to_str().unwrap(), "--json"])
        .assert()
        .success();
    let stdout_bytes = output.get_output().stdout.clone();
    let json_str = String::from_utf8(stdout_bytes).unwrap();
    let json_val: serde_json::Value = serde_json::from_str(&json_str).unwrap();
    assert_eq!(json_val["metadata"]["name"], "Test Track");
    assert_eq!(json_val["metadata"]["genre"], "Electronic");
}

#[test]
fn test_encode_with_temporal_and_rights() {
    let tmp = TempDir::new().expect("Failed to create temp dir");
    let tmp_path = tmp.path();

    let samples = generate_sine_wave(44100, 440.0, 0.5);
    let wav_path = tmp_path.join("tone.wav");
    write_wav_f32(&wav_path, &samples, 44100);

    // Create temporal metadata.
    let temporal_json = serde_json::json!({
        "bpm": 128.0,
        "key": "A minor",
        "time_signature": "4/4",
        "chords": [
            {"time": 0.0, "chord": "Am"},
            {"time": 2.0, "chord": "F"}
        ],
        "lyrics": [
            {"time": 0.0, "end": 1.5, "line": "Hello world"}
        ]
    });
    let temporal_path = tmp_path.join("temporal.json");
    std::fs::write(
        &temporal_path,
        serde_json::to_string_pretty(&temporal_json).unwrap(),
    )
    .unwrap();

    // Create rights metadata.
    let rights_json = serde_json::json!({
        "splits": [
            {"address": "0xAAAA", "chain": "ethereum", "share": 0.6, "name": "Producer"},
            {"address": "0xBBBB", "chain": "ethereum", "share": 0.4, "name": "Vocalist"}
        ],
        "license_uri": "https://creativecommons.org/licenses/by/4.0/"
    });
    let rights_path = tmp_path.join("rights.json");
    std::fs::write(
        &rights_path,
        serde_json::to_string_pretty(&rights_json).unwrap(),
    )
    .unwrap();

    let neo_path = tmp_path.join("full_meta.neo");

    // Encode with temporal and rights.
    neo_cmd()
        .args([
            "encode",
            wav_path.to_str().unwrap(),
            "-o",
            neo_path.to_str().unwrap(),
            "--stems",
            "vocals",
            "--codec",
            "pcm",
            "--temporal",
            temporal_path.to_str().unwrap(),
            "--rights",
            rights_path.to_str().unwrap(),
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains("Temporal metadata embedded"))
        .stdout(predicate::str::contains("Web3 rights embedded"));

    // Verify info shows temporal and web3 flags.
    neo_cmd()
        .args(["info", neo_path.to_str().unwrap()])
        .assert()
        .success()
        .stdout(predicate::str::contains("temporal-meta"))
        .stdout(predicate::str::contains("web3-rights"));

    // Verify JSON output includes temporal and rights.
    let output = neo_cmd()
        .args(["info", neo_path.to_str().unwrap(), "--json"])
        .assert()
        .success();
    let stdout_bytes = output.get_output().stdout.clone();
    let json_str = String::from_utf8(stdout_bytes).unwrap();
    let json_val: serde_json::Value = serde_json::from_str(&json_str).unwrap();
    assert_eq!(json_val["temporal"]["bpm"], 128.0);
    assert_eq!(json_val["temporal"]["key"], "A minor");
    assert_eq!(json_val["rights"]["splits"][0]["name"], "Producer");
}

#[test]
fn test_tag_command_add_metadata() {
    let tmp = TempDir::new().expect("Failed to create temp dir");
    let tmp_path = tmp.path();

    // Create a basic .neo file with no metadata.
    let samples = generate_sine_wave(44100, 440.0, 0.5);
    let wav_path = tmp_path.join("tone.wav");
    write_wav_f32(&wav_path, &samples, 44100);

    let neo_path = tmp_path.join("taggable.neo");
    neo_cmd()
        .args([
            "encode",
            wav_path.to_str().unwrap(),
            "-o",
            neo_path.to_str().unwrap(),
            "--stems",
            "vocals",
            "--codec",
            "pcm",
        ])
        .assert()
        .success();

    // Tag the file with metadata.
    let tagged_path = tmp_path.join("tagged.neo");
    neo_cmd()
        .args([
            "tag",
            neo_path.to_str().unwrap(),
            "-o",
            tagged_path.to_str().unwrap(),
            "--title",
            "My Song",
            "--artist",
            "DJ Test",
            "--genre",
            "House",
            "--bpm",
            "128",
            "--key",
            "A minor",
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains("NEO Tagger"))
        .stdout(predicate::str::contains("Updated: JSON-LD metadata"))
        .stdout(predicate::str::contains("Updated: Temporal metadata"))
        .stdout(predicate::str::contains("Done!"));

    // Verify the tagged file has the metadata.
    let output = neo_cmd()
        .args(["info", tagged_path.to_str().unwrap(), "--json"])
        .assert()
        .success();
    let stdout_bytes = output.get_output().stdout.clone();
    let json_str = String::from_utf8(stdout_bytes).unwrap();
    let json_val: serde_json::Value = serde_json::from_str(&json_str).unwrap();
    assert_eq!(json_val["metadata"]["name"], "My Song");
    assert_eq!(json_val["temporal"]["bpm"], 128.0);
    assert_eq!(json_val["temporal"]["key"], "A minor");

    // Verify the audio is still intact.
    let output_dir = tmp_path.join("decoded_tagged");
    neo_cmd()
        .args([
            "decode",
            tagged_path.to_str().unwrap(),
            "-o",
            output_dir.to_str().unwrap(),
            "--all-stems",
        ])
        .assert()
        .success();

    let decoded_samples = read_wav_f32(&output_dir.join("vocals.wav"));
    assert_eq!(decoded_samples.len(), samples.len());
    // Verify audio integrity (PCM should be bit-perfect).
    for (i, (orig, dec)) in samples.iter().zip(decoded_samples.iter()).enumerate() {
        assert!(
            (orig - dec).abs() < 1e-6,
            "Sample {i} differs after tagging: original={orig}, decoded={dec}"
        );
    }
}

#[test]
fn test_tag_show_metadata() {
    let tmp = TempDir::new().expect("Failed to create temp dir");
    let tmp_path = tmp.path();

    let samples = generate_sine_wave(44100, 440.0, 0.25);
    let wav_path = tmp_path.join("tone.wav");
    write_wav_f32(&wav_path, &samples, 44100);

    // Create a metadata file.
    let meta_json = serde_json::json!({
        "@context": ["https://schema.org", {"neo": "https://neo-audio.org/ns/"}],
        "@type": "MusicRecording",
        "name": "Showable Track",
        "genre": "Jazz"
    });
    let meta_path = tmp_path.join("meta.json");
    std::fs::write(
        &meta_path,
        serde_json::to_string_pretty(&meta_json).unwrap(),
    )
    .unwrap();

    let neo_path = tmp_path.join("showable.neo");
    neo_cmd()
        .args([
            "encode",
            wav_path.to_str().unwrap(),
            "-o",
            neo_path.to_str().unwrap(),
            "--stems",
            "mix",
            "--codec",
            "pcm",
            "--metadata",
            meta_path.to_str().unwrap(),
        ])
        .assert()
        .success();

    // Use tag --show to display metadata.
    neo_cmd()
        .args(["tag", neo_path.to_str().unwrap(), "--show"])
        .assert()
        .success()
        .stdout(predicate::str::contains("NEO Metadata"))
        .stdout(predicate::str::contains("Showable Track"))
        .stdout(predicate::str::contains("Jazz"));
}
