//! Benchmarks for the NEO container format: write, read, and BLAKE3 hashing.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use neo_format::{CodecId, NeoReader, NeoWriter, StemConfig, StemLabel};

/// Well-known labels to cycle through when building multi-stem configs.
const LABELS: [&str; 8] = [
    "vocals",
    "drums",
    "bass",
    "melody",
    "instrumental",
    "mix",
    "fx1",
    "fx2",
];

/// Create stem configs for the given count.
fn test_configs(count: u8) -> Vec<StemConfig> {
    (0..count)
        .map(|i| {
            let label: StemLabel = LABELS[i as usize].parse().unwrap();
            StemConfig {
                stem_id: i,
                label,
                codec: CodecId::Pcm,
                channels: 2,
                sample_rate: 48000,
                bit_depth: 32,
                bitrate_kbps: 0,
                sample_count: 48000, // 1 second
            }
        })
        .collect()
}

/// Generate fake PCM audio data (1 second stereo at 48 kHz, f32 samples).
fn generate_pcm_data(sample_count: usize, channels: usize) -> Vec<u8> {
    let total = sample_count * channels;
    let mut buf = Vec::with_capacity(total * 4);
    for i in 0..total {
        let sample = (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 48000.0).sin();
        buf.extend_from_slice(&sample.to_le_bytes());
    }
    buf
}

/// Write a complete NEO file via NeoWriter and return the temp dir + path.
fn write_test_file(stem_count: u8) -> (tempfile::TempDir, std::path::PathBuf) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("bench.neo");

    let configs = test_configs(stem_count);
    let pcm = generate_pcm_data(48000, 2);

    let mut writer = NeoWriter::new(48000, stem_count);
    writer.set_duration_us(1_000_000);
    for config in &configs {
        writer.add_stem(config.clone(), pcm.clone()).unwrap();
    }
    writer.finalize(&path).unwrap();

    (dir, path)
}

fn bench_write(c: &mut Criterion) {
    let pcm = generate_pcm_data(48000, 2);

    let mut group = c.benchmark_group("neo_write");
    for stem_count in [1u8, 2, 4, 8] {
        let configs = test_configs(stem_count);
        group.bench_with_input(
            BenchmarkId::new("stems", stem_count),
            &stem_count,
            |b, &count| {
                b.iter(|| {
                    let dir = tempfile::tempdir().unwrap();
                    let path = dir.path().join("bench.neo");
                    let mut writer = NeoWriter::new(48000, count);
                    writer.set_duration_us(1_000_000);
                    for config in &configs {
                        writer
                            .add_stem(config.clone(), black_box(pcm.clone()))
                            .unwrap();
                    }
                    writer.finalize(black_box(&path)).unwrap();
                });
            },
        );
    }
    group.finish();
}

fn bench_read(c: &mut Criterion) {
    let mut group = c.benchmark_group("neo_read");
    for stem_count in [1u8, 2, 4, 8] {
        let (_dir, path) = write_test_file(stem_count);
        group.bench_with_input(
            BenchmarkId::new("stems", stem_count),
            &stem_count,
            |b, &count| {
                b.iter(|| {
                    let mut reader = NeoReader::open(black_box(&path)).unwrap();
                    for i in 0..count {
                        let _ = black_box(reader.read_stem_data(i).unwrap());
                    }
                });
            },
        );
    }
    group.finish();
}

fn bench_blake3_hash(c: &mut Criterion) {
    let mut group = c.benchmark_group("blake3_hash");
    for size in [1024usize, 65536, 1_048_576] {
        let data: Vec<u8> = (0..size).map(|i| (i % 256) as u8).collect();
        group.bench_with_input(BenchmarkId::new("bytes", size), &data, |b, data| {
            b.iter(|| {
                let hash = blake3::hash(black_box(data));
                black_box(hash);
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_write, bench_read, bench_blake3_hash);
criterion_main!(benches);
