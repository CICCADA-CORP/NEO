//! Example: Write → Read round-trip proving bit-perfect PCM integrity.
//!
//! Demonstrates that NEO preserves audio data exactly through
//! a write → read cycle with BLAKE3 hash verification.

use std::path::Path;

use neo_format::{CodecId, NeoReader, NeoWriter, StemConfig, StemLabel};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dir = tempfile::tempdir()?;
    let path = dir.path().join("round_trip.neo");

    println!("=== NEO Round-Trip Test ===\n");

    // Generate test audio: 1 second of 440 Hz sine at 48 kHz mono
    let sample_rate = 48000u32;
    let samples: Vec<f32> = (0..sample_rate)
        .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / sample_rate as f32).sin())
        .collect();

    // Convert to PCM bytes (f32 little-endian)
    let original_bytes: Vec<u8> = samples.iter().flat_map(|s| s.to_le_bytes()).collect();

    let original_hash = blake3::hash(&original_bytes);
    println!(
        "Original: {} samples, {} bytes",
        samples.len(),
        original_bytes.len()
    );
    println!("BLAKE3:   {}\n", original_hash);

    // Write to NEO file
    let config = StemConfig::new(0, StemLabel::Mix, CodecId::Pcm, 1, sample_rate);

    let mut writer = NeoWriter::new(sample_rate, 1);
    writer.add_stem(config, original_bytes.clone())?;
    writer.set_duration_us(1_000_000);
    writer.finalize(&path)?;

    let file_size = std::fs::metadata(&path)?.len();
    println!("Written:  {} ({} bytes on disk)", path.display(), file_size);

    // Read back
    let mut reader = NeoReader::open(Path::new(&path))?;
    let recovered_bytes = reader.read_stem_data(0)?;
    let recovered_hash = blake3::hash(&recovered_bytes);

    println!("Read:     {} bytes", recovered_bytes.len());
    println!("BLAKE3:   {}\n", recovered_hash);

    // Verify
    assert_eq!(
        original_bytes.len(),
        recovered_bytes.len(),
        "Size mismatch!"
    );
    assert_eq!(original_hash, recovered_hash, "Hash mismatch!");
    assert_eq!(original_bytes, recovered_bytes, "Data mismatch!");

    println!("✓ Round-trip is bit-perfect!");
    println!("✓ BLAKE3 hashes match!");
    println!("✓ All {} bytes identical!", original_bytes.len());

    Ok(())
}
