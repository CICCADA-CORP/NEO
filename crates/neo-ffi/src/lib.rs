//! C FFI bindings for the NEO audio format.
//!
//! Provides a C-compatible API for reading `.neo` files from any language
//! that can call C functions (Python, Swift, Go, C++, etc.).
//!
//! # Safety
//!
//! All functions in this module use raw pointers and are `unsafe` by nature
//! of the C FFI. Callers must ensure that:
//! - Pointers are valid and non-null
//! - Strings are valid UTF-8 and null-terminated
//! - Handles are properly opened and closed
//! - Thread safety is managed by the caller

pub mod plugin;
pub mod wasm;

use std::ffi::{CStr, CString};
use std::os::raw::c_char;
use std::path::Path;
use std::ptr;

/// Opaque handle to a NEO file reader.
///
/// Created by [`neo_open`] and must be freed with [`neo_close`].
pub struct NeoHandle {
    reader: neo_format::NeoReader,
}

/// Result codes returned by FFI functions.
#[repr(i32)]
pub enum NeoResult {
    /// Operation succeeded.
    Ok = 0,
    /// Null pointer argument.
    NullPointer = -1,
    /// File not found or I/O error.
    IoError = -2,
    /// Invalid NEO file format.
    FormatError = -3,
    /// Stem not found.
    StemNotFound = -4,
    /// Invalid UTF-8 string.
    Utf8Error = -5,
    /// Internal error.
    InternalError = -99,
}

/// Information about a stem in a NEO file.
#[repr(C)]
pub struct NeoStemInfo {
    /// Stem identifier (0-7).
    pub stem_id: u8,
    /// Codec identifier (1=PCM, 2=DAC, 3=FLAC, 4=Opus).
    pub codec_id: u8,
    /// Number of audio channels.
    pub channels: u8,
    /// Sample rate in Hz.
    pub sample_rate: u32,
    /// Bits per sample.
    pub bit_depth: u8,
    /// Bitrate in kbps (0 for lossless).
    pub bitrate_kbps: u32,
    /// Total number of audio frames.
    pub sample_count: u64,
}

/// Information about a NEO file header.
#[repr(C)]
pub struct NeoFileInfo {
    /// Format version.
    pub version: u16,
    /// Feature flags bitfield.
    pub feature_flags: u64,
    /// Number of stems.
    pub stem_count: u8,
    /// Master sample rate in Hz.
    pub sample_rate: u32,
    /// Total duration in microseconds.
    pub duration_us: u64,
    /// Number of chunks in the file.
    pub chunk_count: u64,
}

// ─────────────────────────── Core API ───────────────────────────

/// Return the library version string.
///
/// The returned pointer is valid for the lifetime of the library.
/// Do NOT free the returned string.
#[no_mangle]
pub extern "C" fn neo_version() -> *const c_char {
    c"0.1.0".as_ptr()
}

/// Open a NEO file for reading.
///
/// Returns a handle pointer on success, or null on failure.
/// The caller must call [`neo_close`] when done.
///
/// # Safety
///
/// `path` must be a valid null-terminated UTF-8 string.
#[no_mangle]
pub unsafe extern "C" fn neo_open(path: *const c_char) -> *mut NeoHandle {
    if path.is_null() {
        return ptr::null_mut();
    }

    let c_str = match unsafe { CStr::from_ptr(path) }.to_str() {
        Ok(s) => s,
        Err(_) => return ptr::null_mut(),
    };

    match neo_format::NeoReader::open(Path::new(c_str)) {
        Ok(reader) => Box::into_raw(Box::new(NeoHandle { reader })),
        Err(_) => ptr::null_mut(),
    }
}

/// Close a NEO file handle and free associated resources.
///
/// # Safety
///
/// `handle` must be a valid pointer returned by [`neo_open`], or null.
/// After calling this function, the handle is invalid and must not be used.
#[no_mangle]
pub unsafe extern "C" fn neo_close(handle: *mut NeoHandle) {
    if !handle.is_null() {
        drop(unsafe { Box::from_raw(handle) });
    }
}

/// Get file information from an open NEO handle.
///
/// Writes the header info into the provided `info` struct.
/// Returns [`NeoResult::Ok`] on success.
///
/// # Safety
///
/// Both `handle` and `info` must be valid non-null pointers.
#[no_mangle]
pub unsafe extern "C" fn neo_get_info(
    handle: *const NeoHandle,
    info: *mut NeoFileInfo,
) -> NeoResult {
    if handle.is_null() || info.is_null() {
        return NeoResult::NullPointer;
    }

    let handle = unsafe { &*handle };
    let header = handle.reader.header();

    unsafe {
        (*info).version = header.version;
        (*info).feature_flags = header.feature_flags.0;
        (*info).stem_count = header.stem_count;
        (*info).sample_rate = header.sample_rate;
        (*info).duration_us = header.duration_us;
        (*info).chunk_count = header.chunk_count;
    }

    NeoResult::Ok
}

/// Get the number of stems in the file.
///
/// # Safety
///
/// `handle` must be a valid non-null pointer returned by [`neo_open`].
#[no_mangle]
pub unsafe extern "C" fn neo_stem_count(handle: *const NeoHandle) -> i32 {
    if handle.is_null() {
        return -1;
    }
    let handle = unsafe { &*handle };
    handle.reader.header().stem_count as i32
}

/// Get information about a specific stem by index.
///
/// `index` is the positional index (0-based) into the stems array,
/// NOT the stem_id. Returns [`NeoResult::StemNotFound`] if the index
/// is out of range.
///
/// # Safety
///
/// Both `handle` and `info` must be valid non-null pointers.
#[no_mangle]
pub unsafe extern "C" fn neo_get_stem_info(
    handle: *const NeoHandle,
    index: u32,
    info: *mut NeoStemInfo,
) -> NeoResult {
    if handle.is_null() || info.is_null() {
        return NeoResult::NullPointer;
    }

    let handle = unsafe { &*handle };
    let stems = handle.reader.stem_configs();

    if index as usize >= stems.len() {
        return NeoResult::StemNotFound;
    }

    let stem = &stems[index as usize];
    unsafe {
        (*info).stem_id = stem.stem_id;
        (*info).codec_id = stem.codec as u8;
        (*info).channels = stem.channels;
        (*info).sample_rate = stem.sample_rate;
        (*info).bit_depth = stem.bit_depth;
        (*info).bitrate_kbps = stem.bitrate_kbps;
        (*info).sample_count = stem.sample_count;
    }

    NeoResult::Ok
}

/// Get the label of a stem as a null-terminated string.
///
/// The returned string is allocated with `malloc` and must be freed
/// by the caller using [`neo_free_string`].
///
/// Returns null if the handle is invalid or the index is out of range.
///
/// # Safety
///
/// `handle` must be a valid non-null pointer returned by [`neo_open`].
#[no_mangle]
pub unsafe extern "C" fn neo_get_stem_label(handle: *const NeoHandle, index: u32) -> *mut c_char {
    if handle.is_null() {
        return ptr::null_mut();
    }

    let handle = unsafe { &*handle };
    let stems = handle.reader.stem_configs();

    if index as usize >= stems.len() {
        return ptr::null_mut();
    }

    let label = stems[index as usize].label.as_str();
    match CString::new(label) {
        Ok(c_string) => c_string.into_raw(),
        Err(_) => ptr::null_mut(),
    }
}

/// Read the JSON-LD metadata string from the file.
///
/// The returned string is allocated and must be freed using [`neo_free_string`].
/// Returns null if no metadata is present or on error.
///
/// # Safety
///
/// `handle` must be a valid non-null pointer returned by [`neo_open`].
#[no_mangle]
pub unsafe extern "C" fn neo_read_metadata(handle: *mut NeoHandle) -> *mut c_char {
    if handle.is_null() {
        return ptr::null_mut();
    }

    let handle = unsafe { &mut *handle };
    match handle.reader.read_metadata() {
        Ok(Some(json)) => match CString::new(json) {
            Ok(c_string) => c_string.into_raw(),
            Err(_) => ptr::null_mut(),
        },
        _ => ptr::null_mut(),
    }
}

/// Read the spatial audio JSON from the SPAT chunk.
///
/// The returned string is allocated and must be freed using [`neo_free_string`].
/// Returns null if no spatial data is present or on error.
///
/// # Safety
///
/// `handle` must be a valid non-null pointer returned by [`neo_open`].
#[no_mangle]
pub unsafe extern "C" fn neo_read_spatial(handle: *mut NeoHandle) -> *mut c_char {
    if handle.is_null() {
        return ptr::null_mut();
    }

    let handle = unsafe { &mut *handle };
    match handle.reader.read_spatial() {
        Ok(Some(json)) => match CString::new(json) {
            Ok(c_string) => c_string.into_raw(),
            Err(_) => ptr::null_mut(),
        },
        _ => ptr::null_mut(),
    }
}

/// Read the edit history JSON from the EDIT chunk.
///
/// The returned string is allocated and must be freed using [`neo_free_string`].
/// Returns null if no edit history is present or on error.
///
/// # Safety
///
/// `handle` must be a valid non-null pointer returned by [`neo_open`].
#[no_mangle]
pub unsafe extern "C" fn neo_read_edit_history(handle: *mut NeoHandle) -> *mut c_char {
    if handle.is_null() {
        return ptr::null_mut();
    }

    let handle = unsafe { &mut *handle };
    match handle.reader.read_edit_history() {
        Ok(Some(json)) => match CString::new(json) {
            Ok(c_string) => c_string.into_raw(),
            Err(_) => ptr::null_mut(),
        },
        _ => ptr::null_mut(),
    }
}

/// Read raw stem audio data.
///
/// Writes the data length to `out_len` and returns a pointer to the
/// allocated buffer. The caller must free the buffer using [`neo_free_buffer`].
///
/// Returns null if the stem is not found or on error.
///
/// # Safety
///
/// `handle` and `out_len` must be valid non-null pointers.
#[no_mangle]
pub unsafe extern "C" fn neo_read_stem_data(
    handle: *mut NeoHandle,
    stem_id: u8,
    out_len: *mut u64,
) -> *mut u8 {
    if handle.is_null() || out_len.is_null() {
        return ptr::null_mut();
    }

    let handle = unsafe { &mut *handle };
    match handle.reader.read_stem_data(stem_id) {
        Ok(data) => {
            let len = data.len() as u64;
            let mut boxed = data.into_boxed_slice();
            let ptr = boxed.as_mut_ptr();
            unsafe { *out_len = len };
            std::mem::forget(boxed);
            ptr
        }
        Err(_) => ptr::null_mut(),
    }
}

/// Free a string previously returned by a `neo_read_*` or `neo_get_*` function.
///
/// # Safety
///
/// `ptr` must be a valid pointer returned by one of the string-returning
/// FFI functions, or null (in which case this is a no-op).
#[no_mangle]
pub unsafe extern "C" fn neo_free_string(ptr: *mut c_char) {
    if !ptr.is_null() {
        drop(unsafe { CString::from_raw(ptr) });
    }
}

/// Free a byte buffer previously returned by [`neo_read_stem_data`].
///
/// # Safety
///
/// `ptr` must be a valid pointer returned by `neo_read_stem_data`,
/// and `len` must be the length that was written to `out_len`.
#[no_mangle]
pub unsafe extern "C" fn neo_free_buffer(ptr: *mut u8, len: u64) {
    if !ptr.is_null() {
        let slice = unsafe { std::slice::from_raw_parts_mut(ptr, len as usize) };
        drop(unsafe { Box::from_raw(slice as *mut [u8]) });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::{CStr, CString};
    use std::ptr;

    use neo_format::stem::{CodecId, StemConfig, StemLabel};
    use neo_format::writer::NeoWriter;

    /// Helper: create a valid `.neo` file with one stem, metadata, spatial,
    /// and edit-history chunks, and return its path as a [`CString`].
    ///
    /// The caller receives the [`tempfile::NamedTempFile`] handle (to keep
    /// the file alive) *and* a [`CString`] of its path for FFI calls.
    fn create_test_neo_file() -> (tempfile::NamedTempFile, CString) {
        let tmp = tempfile::NamedTempFile::new().expect("failed to create temp file");
        let path = tmp.path().to_path_buf();

        let config = StemConfig::new(0, StemLabel::Vocals, CodecId::Pcm, 1, 44100);
        let audio_data = vec![0u8; 100];

        let mut writer = NeoWriter::new(44100, 1);
        writer.add_stem(config, audio_data).unwrap();
        writer.set_metadata(r#"{"@context":"https://schema.org","title":"test"}"#.to_string());
        writer.set_spatial(r#"{"objects":[]}"#.to_string());
        writer.set_edit_history(r#"{"commits":[]}"#.to_string());
        writer.set_duration_us(1_000_000);
        writer.finalize(&path).unwrap();

        let c_path = CString::new(path.to_str().unwrap()).unwrap();
        (tmp, c_path)
    }

    // ───────────────── 1. neo_version ─────────────────

    #[test]
    fn test_neo_version() {
        let ptr = neo_version();
        assert!(!ptr.is_null());
        let version = unsafe { CStr::from_ptr(ptr) }.to_str().unwrap();
        assert_eq!(version, "0.1.0");
    }

    // ───────────────── 2–4. neo_open ─────────────────

    #[test]
    fn test_neo_open_valid_file() {
        let (_tmp, c_path) = create_test_neo_file();
        let handle = unsafe { neo_open(c_path.as_ptr()) };
        assert!(!handle.is_null(), "neo_open should return a valid handle");
        unsafe { neo_close(handle) };
    }

    #[test]
    fn test_neo_open_null_path() {
        let handle = unsafe { neo_open(ptr::null()) };
        assert!(handle.is_null(), "neo_open(null) should return null");
    }

    #[test]
    fn test_neo_open_invalid_path() {
        let bad = CString::new("/tmp/nonexistent_neo_file_12345.neo").unwrap();
        let handle = unsafe { neo_open(bad.as_ptr()) };
        assert!(
            handle.is_null(),
            "neo_open with bad path should return null"
        );
    }

    // ───────────────── 5. neo_close ─────────────────

    #[test]
    fn test_neo_close_null() {
        // Must not crash.
        unsafe { neo_close(ptr::null_mut()) };
    }

    // ───────────────── 6–8. neo_get_info ─────────────────

    #[test]
    fn test_neo_get_info() {
        let (_tmp, c_path) = create_test_neo_file();
        let handle = unsafe { neo_open(c_path.as_ptr()) };
        assert!(!handle.is_null());

        let mut info = std::mem::MaybeUninit::<NeoFileInfo>::zeroed();
        let result = unsafe { neo_get_info(handle, info.as_mut_ptr()) };
        assert_eq!(result as i32, NeoResult::Ok as i32);

        let info = unsafe { info.assume_init() };
        assert_eq!(info.version, 1);
        assert_eq!(info.stem_count, 1);
        assert_eq!(info.sample_rate, 44100);

        unsafe { neo_close(handle) };
    }

    #[test]
    fn test_neo_get_info_null_handle() {
        let mut info = std::mem::MaybeUninit::<NeoFileInfo>::zeroed();
        let result = unsafe { neo_get_info(ptr::null(), info.as_mut_ptr()) };
        assert_eq!(result as i32, NeoResult::NullPointer as i32);
    }

    #[test]
    fn test_neo_get_info_null_info() {
        let (_tmp, c_path) = create_test_neo_file();
        let handle = unsafe { neo_open(c_path.as_ptr()) };
        assert!(!handle.is_null());

        let result = unsafe { neo_get_info(handle, ptr::null_mut()) };
        assert_eq!(result as i32, NeoResult::NullPointer as i32);

        unsafe { neo_close(handle) };
    }

    // ───────────────── 9–10. neo_stem_count ─────────────────

    #[test]
    fn test_neo_stem_count() {
        let (_tmp, c_path) = create_test_neo_file();
        let handle = unsafe { neo_open(c_path.as_ptr()) };
        assert!(!handle.is_null());

        let count = unsafe { neo_stem_count(handle) };
        assert_eq!(count, 1);

        unsafe { neo_close(handle) };
    }

    #[test]
    fn test_neo_stem_count_null() {
        let count = unsafe { neo_stem_count(ptr::null()) };
        assert_eq!(count, -1);
    }

    // ───────────────── 11–12. neo_get_stem_info ─────────────────

    #[test]
    fn test_neo_get_stem_info() {
        let (_tmp, c_path) = create_test_neo_file();
        let handle = unsafe { neo_open(c_path.as_ptr()) };
        assert!(!handle.is_null());

        let mut info = std::mem::MaybeUninit::<NeoStemInfo>::zeroed();
        let result = unsafe { neo_get_stem_info(handle, 0, info.as_mut_ptr()) };
        assert_eq!(result as i32, NeoResult::Ok as i32);

        let info = unsafe { info.assume_init() };
        assert_eq!(info.stem_id, 0);
        assert_eq!(info.codec_id, CodecId::Pcm as u8);
        assert_eq!(info.channels, 1);
        assert_eq!(info.sample_rate, 44100);

        unsafe { neo_close(handle) };
    }

    #[test]
    fn test_neo_get_stem_info_out_of_range() {
        let (_tmp, c_path) = create_test_neo_file();
        let handle = unsafe { neo_open(c_path.as_ptr()) };
        assert!(!handle.is_null());

        let mut info = std::mem::MaybeUninit::<NeoStemInfo>::zeroed();
        let result = unsafe { neo_get_stem_info(handle, 99, info.as_mut_ptr()) };
        assert_eq!(result as i32, NeoResult::StemNotFound as i32);

        unsafe { neo_close(handle) };
    }

    // ───────────────── 13–15. neo_get_stem_label ─────────────────

    #[test]
    fn test_neo_get_stem_label() {
        let (_tmp, c_path) = create_test_neo_file();
        let handle = unsafe { neo_open(c_path.as_ptr()) };
        assert!(!handle.is_null());

        let label_ptr = unsafe { neo_get_stem_label(handle, 0) };
        assert!(!label_ptr.is_null());

        let label = unsafe { CStr::from_ptr(label_ptr) }.to_str().unwrap();
        assert_eq!(label, "vocals");

        unsafe { neo_free_string(label_ptr) };
        unsafe { neo_close(handle) };
    }

    #[test]
    fn test_neo_get_stem_label_null() {
        let label_ptr = unsafe { neo_get_stem_label(ptr::null(), 0) };
        assert!(label_ptr.is_null());
    }

    #[test]
    fn test_neo_get_stem_label_out_of_range() {
        let (_tmp, c_path) = create_test_neo_file();
        let handle = unsafe { neo_open(c_path.as_ptr()) };
        assert!(!handle.is_null());

        let label_ptr = unsafe { neo_get_stem_label(handle, 99) };
        assert!(label_ptr.is_null());

        unsafe { neo_close(handle) };
    }

    // ───────────────── 16–17. neo_read_metadata ─────────────────

    #[test]
    fn test_neo_read_metadata() {
        let (_tmp, c_path) = create_test_neo_file();
        let handle = unsafe { neo_open(c_path.as_ptr()) };
        assert!(!handle.is_null());

        let meta_ptr = unsafe { neo_read_metadata(handle) };
        assert!(!meta_ptr.is_null());

        let meta = unsafe { CStr::from_ptr(meta_ptr) }.to_str().unwrap();
        assert!(
            meta.contains("schema.org"),
            "metadata should contain schema.org"
        );
        assert!(meta.contains("test"), "metadata should contain 'test'");

        unsafe { neo_free_string(meta_ptr) };
        unsafe { neo_close(handle) };
    }

    #[test]
    fn test_neo_read_metadata_null() {
        let meta_ptr = unsafe { neo_read_metadata(ptr::null_mut()) };
        assert!(meta_ptr.is_null());
    }

    // ───────────────── 18. neo_read_spatial ─────────────────

    #[test]
    fn test_neo_read_spatial() {
        let (_tmp, c_path) = create_test_neo_file();
        let handle = unsafe { neo_open(c_path.as_ptr()) };
        assert!(!handle.is_null());

        let spat_ptr = unsafe { neo_read_spatial(handle) };
        assert!(!spat_ptr.is_null());

        let spat = unsafe { CStr::from_ptr(spat_ptr) }.to_str().unwrap();
        assert!(spat.contains("objects"), "spatial should contain 'objects'");

        unsafe { neo_free_string(spat_ptr) };
        unsafe { neo_close(handle) };
    }

    // ───────────────── 19. neo_read_edit_history ─────────────────

    #[test]
    fn test_neo_read_edit_history() {
        let (_tmp, c_path) = create_test_neo_file();
        let handle = unsafe { neo_open(c_path.as_ptr()) };
        assert!(!handle.is_null());

        let edit_ptr = unsafe { neo_read_edit_history(handle) };
        assert!(!edit_ptr.is_null());

        let edit = unsafe { CStr::from_ptr(edit_ptr) }.to_str().unwrap();
        assert!(
            edit.contains("commits"),
            "edit history should contain 'commits'"
        );

        unsafe { neo_free_string(edit_ptr) };
        unsafe { neo_close(handle) };
    }

    // ───────────────── 20–22. neo_read_stem_data ─────────────────

    #[test]
    fn test_neo_read_stem_data() {
        let (_tmp, c_path) = create_test_neo_file();
        let handle = unsafe { neo_open(c_path.as_ptr()) };
        assert!(!handle.is_null());

        let mut out_len: u64 = 0;
        let data_ptr = unsafe { neo_read_stem_data(handle, 0, &mut out_len) };
        assert!(!data_ptr.is_null());
        assert_eq!(out_len, 100, "stem data should be 100 bytes");

        // Verify the data content (we wrote 100 zero bytes)
        let data = unsafe { std::slice::from_raw_parts(data_ptr, out_len as usize) };
        assert!(
            data.iter().all(|&b| b == 0),
            "stem data should be all zeros"
        );

        unsafe { neo_free_buffer(data_ptr, out_len) };
        unsafe { neo_close(handle) };
    }

    #[test]
    fn test_neo_read_stem_data_null() {
        let mut out_len: u64 = 0;
        let data_ptr = unsafe { neo_read_stem_data(ptr::null_mut(), 0, &mut out_len) };
        assert!(data_ptr.is_null());
    }

    #[test]
    fn test_neo_read_stem_data_invalid_stem() {
        let (_tmp, c_path) = create_test_neo_file();
        let handle = unsafe { neo_open(c_path.as_ptr()) };
        assert!(!handle.is_null());

        let mut out_len: u64 = 0;
        let data_ptr = unsafe { neo_read_stem_data(handle, 7, &mut out_len) };
        assert!(
            data_ptr.is_null(),
            "reading non-existent stem should return null"
        );

        unsafe { neo_close(handle) };
    }

    // ───────────────── 23–24. neo_free_string / neo_free_buffer null safety ─────────────────

    #[test]
    fn test_neo_free_string_null() {
        // Must not crash.
        unsafe { neo_free_string(ptr::null_mut()) };
    }

    #[test]
    fn test_neo_free_buffer_null() {
        // Must not crash.
        unsafe { neo_free_buffer(ptr::null_mut(), 0) };
    }
}
