# NEO Examples

Example programs demonstrating the NEO audio codec library API.

## Examples

| Example | Description |
|---------|-------------|
| `create_neo.rs` | Create a simple .neo file with synthetic audio |
| `read_neo.rs` | Read and inspect a .neo file |
| `round_trip.rs` | Encode → decode round-trip demonstrating bit-perfect PCM |

## Running

```bash
cd crates

# Create a .neo file
cargo run --example create_neo

# Read and inspect it
cargo run --example read_neo

# Full round-trip test
cargo run --example round_trip
```

Note: Examples require being run from the `crates/` directory as they create temporary files.
