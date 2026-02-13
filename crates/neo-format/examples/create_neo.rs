//! Example: Create a NEO file with synthetic audio data.
//!
//! Generates a sine wave, encodes it as PCM, and writes a .neo file
//! with two stems (vocals and instrumental).

use std::path::Path;

use neo_format::{CodecId, NeoWriter, StemConfig, StemLabel};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let sample_rate = 48000u32;
    let duration_secs = 2u32;
    let channels = 2u8;

    // Generate a stereo sine wave (440 Hz left, 880 Hz right)
    let pcm_vocals = generate_sine_stereo(440.0, 880.0, sample_rate, duration_secs);
    let pcm_instrumental = generate_sine_stereo(220.0, 330.0, sample_rate, duration_secs);

    // Encode to PCM bytes (f32 little-endian)
    let vocals_bytes = samples_to_bytes(&pcm_vocals);
    let instrumental_bytes = samples_to_bytes(&pcm_instrumental);

    // Define stem configurations
    let vocals_config = StemConfig::new(0, StemLabel::Vocals, CodecId::Pcm, channels, sample_rate);
    let inst_config = StemConfig::new(
        1,
        StemLabel::Instrumental,
        CodecId::Pcm,
        channels,
        sample_rate,
    );

    // Build the NEO file using the writer builder
    let mut writer = NeoWriter::new(sample_rate, 2);
    writer.add_stem(vocals_config, vocals_bytes)?;
    writer.add_stem(inst_config, instrumental_bytes)?;

    // Add JSON-LD metadata
    writer.set_metadata(
        r#"{
        "@context": "https://schema.org/",
        "@type": "MusicRecording",
        "name": "NEO Example",
        "byArtist": {"@type": "Person", "name": "NEO SDK"},
        "genre": "Synthetic",
        "duration": "PT2S"
    }"#
        .to_string(),
    );

    // Set total duration
    writer.set_duration_us((duration_secs as u64) * 1_000_000);

    // Write the file to disk
    let output_path = Path::new("example_output.neo");
    writer.finalize(output_path)?;

    println!("Created: {}", output_path.display());
    println!("  Stems: vocals (440/880 Hz), instrumental (220/330 Hz)");
    println!(
        "  Duration: {}s, Sample rate: {} Hz",
        duration_secs, sample_rate
    );

    // Clean up
    std::fs::remove_file(output_path)?;
    println!("  (Cleaned up temp file)");

    Ok(())
}

/// Generate interleaved stereo sine wave samples.
fn generate_sine_stereo(freq_l: f32, freq_r: f32, rate: u32, secs: u32) -> Vec<f32> {
    let total_frames = (rate * secs) as usize;
    let mut samples = Vec::with_capacity(total_frames * 2);
    for i in 0..total_frames {
        let t = i as f32 / rate as f32;
        samples.push((2.0 * std::f32::consts::PI * freq_l * t).sin() * 0.5);
        samples.push((2.0 * std::f32::consts::PI * freq_r * t).sin() * 0.5);
    }
    samples
}

/// Convert f32 samples to little-endian byte buffer.
fn samples_to_bytes(samples: &[f32]) -> Vec<u8> {
    let mut buf = Vec::with_capacity(samples.len() * 4);
    for &s in samples {
        buf.extend_from_slice(&s.to_le_bytes());
    }
    buf
}
