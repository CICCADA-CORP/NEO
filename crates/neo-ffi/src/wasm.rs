//! WebAssembly decoder stub for the NEO format.
//!
//! This module provides the architecture for a WASM-compatible decoder
//! that can run in web browsers. The actual implementation requires
//! `wasm-bindgen` and is planned for a future iteration.
//!
//! # Architecture
//!
//! The WASM decoder will:
//! 1. Accept a `.neo` file as a `Uint8Array` from JavaScript
//! 2. Parse the header and chunk table
//! 3. Decode stems to PCM f32 arrays
//! 4. Return decoded audio buffers to the Web Audio API
//!
//! # Future Dependencies
//!
//! - `wasm-bindgen` for JS interop
//! - `web-sys` for Web Audio API access
//! - `js-sys` for JavaScript type conversions

/// WASM decoder state (placeholder).
///
/// In a full implementation, this would hold the parsed file data
/// and provide methods callable from JavaScript.
#[cfg(target_arch = "wasm32")]
pub struct WasmDecoder {
    // Will hold parsed NeoReader state for in-memory operation
}

/// Target platforms for the WASM build.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WasmTarget {
    /// Web browser via wasm-bindgen.
    Browser,
    /// Node.js via wasm-bindgen with node target.
    Node,
    /// WASI for server-side or CLI usage.
    Wasi,
}

/// WASM build configuration.
#[derive(Debug, Clone)]
pub struct WasmConfig {
    /// Target platform.
    pub target: WasmTarget,
    /// Enable SIMD optimizations (requires browser support).
    pub enable_simd: bool,
    /// Enable threading via SharedArrayBuffer.
    pub enable_threads: bool,
    /// Maximum memory in pages (64 KiB each).
    pub max_memory_pages: u32,
}

impl Default for WasmConfig {
    fn default() -> Self {
        Self {
            target: WasmTarget::Browser,
            enable_simd: false,
            enable_threads: false,
            max_memory_pages: 256, // 16 MiB
        }
    }
}
