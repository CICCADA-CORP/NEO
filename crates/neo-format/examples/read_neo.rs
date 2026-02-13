//! Example: Read and inspect a NEO file.
//!
//! Creates a temporary NEO file, then reads it back and prints
//! all header fields, stem configs, and metadata.

use std::path::Path;

use neo_format::{CodecId, NeoReader, NeoWriter, StemConfig, StemLabel};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // First, create a test file in a temp directory
    let dir = tempfile::tempdir()?;
    let path = dir.path().join("demo.neo");

    let vocals_config = StemConfig::new(0, StemLabel::Vocals, CodecId::Pcm, 1, 44100);
    let drums_config = StemConfig::new(1, StemLabel::Drums, CodecId::Pcm, 2, 44100);

    // Prepare sample data: 1 second of silence
    let mono_pcm: Vec<u8> = vec![0u8; 44100 * 4]; // 1s mono (f32 = 4 bytes/sample)
    let stereo_pcm: Vec<u8> = vec![0u8; 44100 * 2 * 4]; // 1s stereo

    let mut writer = NeoWriter::new(44100, 2);
    writer.add_stem(vocals_config, mono_pcm)?;
    writer.add_stem(drums_config, stereo_pcm)?;
    writer.set_metadata(r#"{"@context":"https://schema.org/","name":"Demo"}"#.to_string());
    writer.set_duration_us(1_000_000);
    writer.finalize(&path)?;

    // Now read it back
    println!("=== NEO File Inspector ===\n");

    let mut reader = NeoReader::open(Path::new(&path))?;
    let h = reader.header();

    println!("Header:");
    println!("  Version:     {}", h.version);
    println!("  Flags:       {:?}", h.feature_flags);
    println!("  Stem count:  {}", h.stem_count);
    println!("  Sample rate: {} Hz", h.sample_rate);
    println!("  Duration:    {:.2}s", h.duration_us as f64 / 1_000_000.0);

    println!("\nStems:");
    // Collect configs first so we can iterate without borrowing reader
    let configs: Vec<_> = reader.stem_configs().to_vec();
    for config in &configs {
        println!(
            "  [{}] {} — {} ch, {} Hz, {:?}",
            config.stem_id,
            config.label.as_str(),
            config.channels,
            config.sample_rate,
            config.codec,
        );

        // Read the actual data
        let data = reader.read_stem_data(config.stem_id)?;
        println!("      Data size: {} bytes", data.len());
    }

    // Read metadata
    if let Some(meta) = reader.read_metadata()? {
        println!("\nMetadata (JSON-LD):");
        println!("  {}", meta);
    }

    println!("\nChunks:");
    for chunk in reader.chunks() {
        println!(
            "  {:?} — {} bytes at offset {}",
            chunk.chunk_type, chunk.size, chunk.offset
        );
    }

    println!("\n✓ File read successfully");
    Ok(())
}
