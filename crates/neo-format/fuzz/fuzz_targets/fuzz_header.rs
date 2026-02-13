//! Fuzz target for NEO header parsing.
//!
//! Generates inputs that start with the NEO magic bytes to increase
//! coverage of header validation logic.

#![no_main]

use libfuzzer_sys::fuzz_target;
use std::io::Write;

fuzz_target!(|data: &[u8]| {
    // Prepend NEO magic to increase chance of reaching header parsing
    let mut input = vec![0x4E, 0x45, 0x4F, 0x21]; // "NEO!"
    input.extend_from_slice(data);

    let mut tmpfile = match tempfile::NamedTempFile::new() {
        Ok(f) => f,
        Err(_) => return,
    };
    if tmpfile.write_all(&input).is_err() {
        return;
    }
    if tmpfile.flush().is_err() {
        return;
    }

    // Try to parse — should never panic, only return errors
    let _ = neo_format::NeoReader::open(tmpfile.path());
});
