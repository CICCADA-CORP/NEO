//! Fuzz target for the NEO file reader.
//!
//! Feeds arbitrary bytes to `NeoReader::open` to find crashes, panics,
//! and hangs in the container parser.

#![no_main]

use libfuzzer_sys::fuzz_target;
use std::io::Write;

fuzz_target!(|data: &[u8]| {
    // Write the fuzz input to a temporary file
    let mut tmpfile = match tempfile::NamedTempFile::new() {
        Ok(f) => f,
        Err(_) => return,
    };
    if tmpfile.write_all(data).is_err() {
        return;
    }
    if tmpfile.flush().is_err() {
        return;
    }

    // Attempt to open it as a NEO file — should never panic
    let path = tmpfile.path();
    if let Ok(mut reader) = neo_format::NeoReader::open(path) {
        // If the file parsed successfully, try reading all data
        let _ = reader.header();
        let configs: Vec<_> = reader.stem_configs().to_vec();
        for config in &configs {
            let _ = reader.read_stem_data(config.stem_id);
        }
        let _ = reader.read_metadata();
        let _ = reader.read_temporal();
        let _ = reader.read_spatial();
        let _ = reader.read_rights();
    }
});
