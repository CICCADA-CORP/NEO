//! # neo-edit — Non-destructive editing for the NEO format
//!
//! Stores edit operations as a DAG (directed acyclic graph) without modifying
//! the original audio data. Supports version history, git-like commits, and diffs.
//!
//! ## Modules
//!
//! - [`graph`] — Edit operations, the edit DAG, and PCM sample application.
//! - [`history`] — Git-like commit history for grouping edits.
//! - [`diff`] — Diff/patch between two history versions.
//! - [`error`] — Error types used throughout the crate.
//!
//! See SPECIFICATION.md Section 13.

pub mod diff;
pub mod error;
pub mod graph;
pub mod history;

pub use diff::{apply_patch, compute_diff, EditDiff};
pub use error::{EditError, Result};
pub use graph::{apply_edit_ops, EditGraph, EditNode, EditOp};
pub use history::{EditCommit, EditHistory};
