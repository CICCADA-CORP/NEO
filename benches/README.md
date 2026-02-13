# NEO Benchmarks

Performance benchmarks for the NEO audio codec.

## Running Benchmarks

```bash
# Run all benchmarks
cd crates && cargo bench

# Run specific benchmark group
cd crates && cargo bench -p neo-format

# Run with specific filter
cd crates && cargo bench -p neo-format -- neo_write
```

## Benchmark Groups

### neo-format (`crates/neo-format/benches/format_bench.rs`)
- **neo_write**: Container write performance with 1/2/4/8 stems
- **neo_read**: Container read + stem extraction with 1/2/4/8 stems
- **blake3_hash**: BLAKE3 hashing at various data sizes (1KB/64KB/1MB)

## Results

Results are stored in `crates/target/criterion/` with HTML reports.
Open `crates/target/criterion/report/index.html` after a run.
