//! Edit operation graph — a DAG of non-destructive audio operations.
//!
//! The edit graph stores a sequence of [`EditNode`]s, each containing an
//! [`EditOp`] and linking back to its parent via a BLAKE3 hash. Operations
//! can be validated, serialized, and applied to PCM sample buffers without
//! modifying the original audio data.
//!
//! See SPECIFICATION.md Section 13.

use serde::{Deserialize, Serialize};

use crate::error::{EditError, Result};

/// A non-destructive edit operation applied to a stem.
///
/// Each variant targets a specific `stem_id` and carries the parameters
/// needed for its audio-processing effect. Call [`EditOp::validate`] to
/// check that the parameters are within legal bounds before adding an
/// operation to a graph or commit.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum EditOp {
    /// Trim audio: keep only the range `[start_s, end_s]`.
    Trim {
        /// Target stem index.
        stem_id: u8,
        /// Start time in seconds (inclusive).
        start_s: f64,
        /// End time in seconds (exclusive).
        end_s: f64,
    },
    /// Apply gain in decibels (±60 dB max).
    Gain {
        /// Target stem index.
        stem_id: u8,
        /// Gain in decibels.
        db: f64,
    },
    /// Parametric EQ band.
    Eq {
        /// Target stem index.
        stem_id: u8,
        /// Center frequency in Hz.
        freq_hz: f64,
        /// Gain in decibels.
        gain_db: f64,
        /// Q factor (bandwidth).
        q: f64,
    },
    /// Fade in/out envelope.
    Fade {
        /// Target stem index.
        stem_id: u8,
        /// Fade-in duration in seconds.
        fade_in_s: f64,
        /// Fade-out duration in seconds.
        fade_out_s: f64,
    },
    /// Mute a stem entirely.
    Mute {
        /// Target stem index.
        stem_id: u8,
    },
    /// Stereo pan position (-1.0 = left, 0.0 = center, 1.0 = right).
    Pan {
        /// Target stem index.
        stem_id: u8,
        /// Pan position in range `[-1.0, 1.0]`.
        position: f64,
    },
    /// Reverse the audio for a stem.
    Reverse {
        /// Target stem index.
        stem_id: u8,
    },
    /// Time-stretch without pitch change.
    TimeStretch {
        /// Target stem index.
        stem_id: u8,
        /// Stretch factor (>1.0 = slower, <1.0 = faster).
        factor: f64,
    },
}

impl EditOp {
    /// Validate that the operation parameters are within legal bounds.
    ///
    /// Returns `Ok(())` if valid, or an appropriate [`EditError`] variant.
    pub fn validate(&self) -> Result<()> {
        match self {
            EditOp::Trim { start_s, end_s, .. } => {
                if *start_s < 0.0 || *end_s < 0.0 || *start_s >= *end_s {
                    return Err(EditError::InvalidTrimRange {
                        start: *start_s,
                        end: *end_s,
                    });
                }
            }
            EditOp::Gain { db, .. } => {
                if db.abs() > 60.0 {
                    return Err(EditError::GainOutOfRange(*db));
                }
            }
            EditOp::Eq { freq_hz, .. } => {
                if *freq_hz < 20.0 || *freq_hz > 20_000.0 {
                    return Err(EditError::EqFreqOutOfRange(*freq_hz));
                }
            }
            EditOp::Fade {
                fade_in_s,
                fade_out_s,
                ..
            } => {
                if *fade_in_s < 0.0 || *fade_out_s < 0.0 {
                    return Err(EditError::InvalidOperation(format!(
                        "fade durations must be non-negative: fade_in={fade_in_s}, fade_out={fade_out_s}"
                    )));
                }
            }
            EditOp::Pan { position, .. } => {
                if *position < -1.0 || *position > 1.0 {
                    return Err(EditError::PanOutOfRange(*position));
                }
            }
            EditOp::TimeStretch { factor, .. } => {
                if *factor <= 0.0 {
                    return Err(EditError::InvalidOperation(format!(
                        "time stretch factor must be positive: {factor}"
                    )));
                }
            }
            EditOp::Mute { .. } | EditOp::Reverse { .. } => {}
        }
        Ok(())
    }

    /// Return the stem ID targeted by this operation.
    pub fn stem_id(&self) -> u8 {
        match self {
            EditOp::Trim { stem_id, .. }
            | EditOp::Gain { stem_id, .. }
            | EditOp::Eq { stem_id, .. }
            | EditOp::Fade { stem_id, .. }
            | EditOp::Mute { stem_id }
            | EditOp::Pan { stem_id, .. }
            | EditOp::Reverse { stem_id }
            | EditOp::TimeStretch { stem_id, .. } => *stem_id,
        }
    }
}

/// A node in the edit DAG.
///
/// Each node stores its BLAKE3 hash (derived from the parent hash and the
/// serialized operation), a link to the parent, the operation itself, a
/// timestamp, and an optional description.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EditNode {
    /// BLAKE3 hash of this node (computed from parent_hash + serialized op).
    pub hash: [u8; 32],
    /// Hash of the parent node (`None` for the root node).
    pub parent_hash: Option<[u8; 32]>,
    /// The edit operation stored at this node.
    pub operation: EditOp,
    /// ISO 8601 timestamp of when this edit was created.
    pub timestamp: String,
    /// Optional human-readable description of the edit.
    pub description: Option<String>,
}

impl EditNode {
    /// Create a new edit node, computing its BLAKE3 hash.
    ///
    /// The hash is derived from the concatenation of the parent hash bytes
    /// (or 32 zero-bytes if root) and the JSON-serialized operation.
    pub fn new(op: EditOp, parent_hash: Option<[u8; 32]>, timestamp: String) -> Self {
        let hash = Self::compute_hash(&parent_hash, &op);
        Self {
            hash,
            parent_hash,
            operation: op,
            timestamp,
            description: None,
        }
    }

    /// Compute the BLAKE3 hash for a node given its parent hash and operation.
    fn compute_hash(parent_hash: &Option<[u8; 32]>, op: &EditOp) -> [u8; 32] {
        let mut hasher = blake3::Hasher::new();
        match parent_hash {
            Some(h) => hasher.update(h),
            None => hasher.update(&[0u8; 32]),
        };
        let op_json = serde_json::to_vec(op).expect("EditOp serialization should not fail");
        hasher.update(&op_json);
        *hasher.finalize().as_bytes()
    }
}

/// The edit DAG — a directed acyclic graph of non-destructive edit operations.
///
/// Operations are appended linearly (each new node points to the previous
/// head as its parent). The graph supports validation, JSON serialization,
/// and querying operations for a specific stem.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EditGraph {
    /// All nodes in the graph, in insertion order (oldest first).
    nodes: Vec<EditNode>,
}

impl EditGraph {
    /// Create a new, empty edit graph.
    pub fn new() -> Self {
        Self { nodes: Vec::new() }
    }

    /// Add an operation to the graph.
    ///
    /// The operation is validated, wrapped in an [`EditNode`] with the current
    /// head as its parent, and appended. Returns the BLAKE3 hash of the new node.
    pub fn add_op(&mut self, op: EditOp, description: Option<String>) -> Result<[u8; 32]> {
        op.validate()?;

        let parent_hash = self.nodes.last().map(|n| n.hash);
        let timestamp = current_timestamp();
        let mut node = EditNode::new(op, parent_hash, timestamp);
        node.description = description;
        let hash = node.hash;
        self.nodes.push(node);
        Ok(hash)
    }

    /// Return the current head (most recently added node), if any.
    pub fn head(&self) -> Option<&EditNode> {
        self.nodes.last()
    }

    /// Return all nodes in insertion order (oldest first).
    pub fn nodes(&self) -> &[EditNode] {
        &self.nodes
    }

    /// Return all operations that affect a given stem, in insertion order.
    pub fn ops_for_stem(&self, stem_id: u8) -> Vec<&EditOp> {
        self.nodes
            .iter()
            .map(|n| &n.operation)
            .filter(|op| op.stem_id() == stem_id)
            .collect()
    }

    /// Serialize the graph to a JSON string.
    pub fn to_json(&self) -> Result<String> {
        Ok(serde_json::to_string_pretty(self)?)
    }

    /// Deserialize a graph from a JSON string.
    pub fn from_json(json: &str) -> Result<Self> {
        Ok(serde_json::from_str(json)?)
    }

    /// Validate the entire graph.
    ///
    /// Checks that:
    /// - Every node's hash is correct (recomputed from parent + op).
    /// - Every non-root node points to the preceding node.
    /// - All operations pass validation.
    pub fn validate(&self) -> Result<()> {
        for (i, node) in self.nodes.iter().enumerate() {
            // Validate the operation itself.
            node.operation.validate()?;

            // Verify the hash.
            let expected = EditNode::compute_hash(&node.parent_hash, &node.operation);
            if node.hash != expected {
                return Err(EditError::InvalidOperation(format!(
                    "node {i} hash mismatch: expected {}, got {}",
                    hex::encode(expected),
                    hex::encode(node.hash),
                )));
            }

            // Verify parent linkage.
            if i == 0 {
                if node.parent_hash.is_some() {
                    return Err(EditError::InvalidOperation(
                        "root node must not have a parent hash".into(),
                    ));
                }
            } else {
                let expected_parent = self.nodes[i - 1].hash;
                match node.parent_hash {
                    Some(ph) if ph == expected_parent => {}
                    _ => {
                        return Err(EditError::InvalidOperation(format!(
                            "node {i} parent hash does not match node {}",
                            i - 1
                        )));
                    }
                }
            }
        }
        Ok(())
    }
}

/// Apply a sequence of edit operations to PCM sample data.
///
/// `samples` is interleaved PCM (`[ch0_s0, ch1_s0, ch0_s1, ch1_s1, …]`).
/// `sample_rate` is in Hz. `channels` is typically 1 or 2.
///
/// Operations are applied in the order given:
/// - **Trim**: slices the buffer to the given time range.
/// - **Gain**: multiplies by `10^(db/20)`.
/// - **Mute**: zeros all samples.
/// - **Pan**: adjusts L/R balance for stereo (2-channel) data.
/// - **Fade**: applies linear fade-in and fade-out envelopes.
/// - **Reverse**: reverses the buffer (frame-wise for multi-channel).
/// - **TimeStretch**: linear-interpolation time stretch (factor > 1 = slower).
/// - **Eq**: biquad peaking EQ filter (Audio EQ Cookbook).
pub fn apply_edit_ops(
    samples: &[f32],
    ops: &[&EditOp],
    sample_rate: u32,
    channels: u16,
) -> Vec<f32> {
    let mut buf = samples.to_vec();
    let ch = channels as usize;

    for op in ops {
        match op {
            EditOp::Trim { start_s, end_s, .. } => {
                let start_frame = (*start_s * sample_rate as f64) as usize;
                let end_frame = (*end_s * sample_rate as f64) as usize;
                let total_frames = buf.len() / ch;
                let start = start_frame.min(total_frames) * ch;
                let end = end_frame.min(total_frames) * ch;
                buf = buf[start..end].to_vec();
            }
            EditOp::Gain { db, .. } => {
                let linear = 10.0_f32.powf(*db as f32 / 20.0);
                for s in &mut buf {
                    *s *= linear;
                }
            }
            EditOp::Mute { .. } => {
                buf.fill(0.0);
            }
            EditOp::Pan { position, .. } => {
                if ch == 2 {
                    // Constant-power-ish pan: left_gain = cos(θ), right_gain = sin(θ)
                    // where θ = (position + 1) / 2 * π/2
                    let theta = ((*position as f32 + 1.0) / 2.0) * std::f32::consts::FRAC_PI_2;
                    let left_gain = theta.cos();
                    let right_gain = theta.sin();
                    for frame in buf.chunks_exact_mut(2) {
                        let mono = (frame[0] + frame[1]) * 0.5;
                        frame[0] = mono * left_gain;
                        frame[1] = mono * right_gain;
                    }
                }
                // For mono, pan is a no-op.
            }
            EditOp::Fade {
                fade_in_s,
                fade_out_s,
                ..
            } => {
                let total_frames = buf.len() / ch;
                let fade_in_frames = (*fade_in_s * sample_rate as f64) as usize;
                let fade_out_frames = (*fade_out_s * sample_rate as f64) as usize;

                // Fade in: linear ramp 0.0 → 1.0
                for f in 0..fade_in_frames.min(total_frames) {
                    let gain = f as f32 / fade_in_frames as f32;
                    for c in 0..ch {
                        buf[f * ch + c] *= gain;
                    }
                }

                // Fade out: linear ramp 1.0 → 0.0
                let fade_out_start = total_frames.saturating_sub(fade_out_frames);
                for f in fade_out_start..total_frames {
                    let remaining = total_frames - f;
                    let gain = remaining as f32 / fade_out_frames as f32;
                    for c in 0..ch {
                        buf[f * ch + c] *= gain;
                    }
                }
            }
            EditOp::Reverse { .. } => {
                // Reverse frame-by-frame (preserving channel order within each frame).
                let total_frames = buf.len() / ch;
                for f in 0..total_frames / 2 {
                    let mirror = total_frames - 1 - f;
                    for c in 0..ch {
                        buf.swap(f * ch + c, mirror * ch + c);
                    }
                }
            }
            EditOp::TimeStretch { factor, .. } => {
                // Simple linear-interpolation time stretch.
                // factor > 1.0 = slower (longer), factor < 1.0 = faster (shorter).
                let total_frames = buf.len() / ch;
                if total_frames == 0 || *factor <= 0.0 {
                    continue;
                }
                let new_frames = (total_frames as f64 * factor) as usize;
                if new_frames == 0 {
                    buf.clear();
                    continue;
                }
                let mut stretched = vec![0.0f32; new_frames * ch];
                for f in 0..new_frames {
                    // Map output frame to input frame position.
                    let src = f as f64 / *factor;
                    let src_floor = src.floor() as usize;
                    let frac = src - src_floor as f64;
                    let src0 = src_floor.min(total_frames - 1);
                    let src1 = (src_floor + 1).min(total_frames - 1);
                    for c in 0..ch {
                        let s0 = buf[src0 * ch + c];
                        let s1 = buf[src1 * ch + c];
                        stretched[f * ch + c] = s0 + (s1 - s0) * frac as f32;
                    }
                }
                buf = stretched;
            }
            EditOp::Eq {
                freq_hz,
                gain_db,
                q,
                ..
            } => {
                // Biquad peaking EQ filter (Audio EQ Cookbook).
                if ch == 0 || buf.is_empty() {
                    continue;
                }
                let sr = sample_rate as f64;
                let omega = 2.0 * std::f64::consts::PI * freq_hz / sr;
                let sin_w = omega.sin();
                let cos_w = omega.cos();
                let alpha = sin_w / (2.0 * q);
                let a_lin = 10.0_f64.powf(gain_db / 40.0); // sqrt of linear gain

                // Peaking EQ coefficients
                let b0 = 1.0 + alpha * a_lin;
                let b1 = -2.0 * cos_w;
                let b2 = 1.0 - alpha * a_lin;
                let a0 = 1.0 + alpha / a_lin;
                let a1 = -2.0 * cos_w;
                let a2 = 1.0 - alpha / a_lin;

                // Normalise by a0
                let b0 = b0 / a0;
                let b1 = b1 / a0;
                let b2 = b2 / a0;
                let a1 = a1 / a0;
                let a2 = a2 / a0;

                // Apply per-channel using Direct Form II Transposed
                for c in 0..ch {
                    let mut z1: f64 = 0.0;
                    let mut z2: f64 = 0.0;
                    let total_frames = buf.len() / ch;
                    for f in 0..total_frames {
                        let idx = f * ch + c;
                        let x = buf[idx] as f64;
                        let y = b0 * x + z1;
                        z1 = b1 * x - a1 * y + z2;
                        z2 = b2 * x - a2 * y;
                        buf[idx] = y as f32;
                    }
                }
            }
        }
    }

    buf
}

/// Return an ISO 8601 timestamp string for "now" (or a fixed string in tests).
fn current_timestamp() -> String {
    // In production this would use chrono or similar.  We keep it dependency-free
    // by returning a fixed placeholder.  Callers who need real timestamps can
    // override the node's `timestamp` field after creation.
    "2026-01-01T00:00:00Z".to_string()
}

/// Tiny hex-encoding helper (avoids pulling in the `hex` crate).
mod hex {
    pub fn encode(bytes: impl AsRef<[u8]>) -> String {
        bytes.as_ref().iter().map(|b| format!("{b:02x}")).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Validation tests ────────────────────────────────────────────

    #[test]
    fn test_validate_trim_valid() {
        let op = EditOp::Trim {
            stem_id: 0,
            start_s: 0.0,
            end_s: 10.0,
        };
        assert!(op.validate().is_ok());
    }

    #[test]
    fn test_validate_trim_invalid_range() {
        let op = EditOp::Trim {
            stem_id: 0,
            start_s: 10.0,
            end_s: 5.0,
        };
        assert!(op.validate().is_err());
    }

    #[test]
    fn test_validate_trim_negative() {
        let op = EditOp::Trim {
            stem_id: 0,
            start_s: -1.0,
            end_s: 5.0,
        };
        assert!(op.validate().is_err());
    }

    #[test]
    fn test_validate_gain_valid() {
        let op = EditOp::Gain {
            stem_id: 0,
            db: 30.0,
        };
        assert!(op.validate().is_ok());
    }

    #[test]
    fn test_validate_gain_out_of_range() {
        let op = EditOp::Gain {
            stem_id: 0,
            db: 61.0,
        };
        assert!(op.validate().is_err());
    }

    #[test]
    fn test_validate_pan_valid() {
        let op = EditOp::Pan {
            stem_id: 0,
            position: -1.0,
        };
        assert!(op.validate().is_ok());
    }

    #[test]
    fn test_validate_pan_out_of_range() {
        let op = EditOp::Pan {
            stem_id: 0,
            position: 1.5,
        };
        assert!(op.validate().is_err());
    }

    #[test]
    fn test_validate_eq_valid() {
        let op = EditOp::Eq {
            stem_id: 0,
            freq_hz: 1000.0,
            gain_db: 3.0,
            q: 1.0,
        };
        assert!(op.validate().is_ok());
    }

    #[test]
    fn test_validate_eq_freq_too_low() {
        let op = EditOp::Eq {
            stem_id: 0,
            freq_hz: 10.0,
            gain_db: 3.0,
            q: 1.0,
        };
        assert!(op.validate().is_err());
    }

    #[test]
    fn test_validate_fade_negative() {
        let op = EditOp::Fade {
            stem_id: 0,
            fade_in_s: -1.0,
            fade_out_s: 2.0,
        };
        assert!(op.validate().is_err());
    }

    #[test]
    fn test_validate_timestretch_zero() {
        let op = EditOp::TimeStretch {
            stem_id: 0,
            factor: 0.0,
        };
        assert!(op.validate().is_err());
    }

    #[test]
    fn test_validate_mute_always_valid() {
        let op = EditOp::Mute { stem_id: 7 };
        assert!(op.validate().is_ok());
    }

    #[test]
    fn test_validate_reverse_always_valid() {
        let op = EditOp::Reverse { stem_id: 3 };
        assert!(op.validate().is_ok());
    }

    // ── stem_id tests ───────────────────────────────────────────────

    #[test]
    fn test_stem_id() {
        assert_eq!(EditOp::Mute { stem_id: 5 }.stem_id(), 5);
        assert_eq!(
            EditOp::Gain {
                stem_id: 2,
                db: 3.0
            }
            .stem_id(),
            2
        );
        assert_eq!(EditOp::Reverse { stem_id: 7 }.stem_id(), 7);
        assert_eq!(
            EditOp::TimeStretch {
                stem_id: 1,
                factor: 1.5
            }
            .stem_id(),
            1
        );
    }

    // ── Node hash tests ─────────────────────────────────────────────

    #[test]
    fn test_node_hash_is_deterministic() {
        let op = EditOp::Gain {
            stem_id: 0,
            db: 6.0,
        };
        let a = EditNode::new(op.clone(), None, "t".into());
        let b = EditNode::new(op, None, "t".into());
        assert_eq!(a.hash, b.hash);
    }

    #[test]
    fn test_node_hash_differs_with_parent() {
        let op = EditOp::Gain {
            stem_id: 0,
            db: 6.0,
        };
        let a = EditNode::new(op.clone(), None, "t".into());
        let b = EditNode::new(op, Some([1u8; 32]), "t".into());
        assert_ne!(a.hash, b.hash);
    }

    #[test]
    fn test_node_hash_differs_with_op() {
        let a = EditNode::new(
            EditOp::Gain {
                stem_id: 0,
                db: 6.0,
            },
            None,
            "t".into(),
        );
        let b = EditNode::new(
            EditOp::Gain {
                stem_id: 0,
                db: 3.0,
            },
            None,
            "t".into(),
        );
        assert_ne!(a.hash, b.hash);
    }

    // ── Graph tests ─────────────────────────────────────────────────

    #[test]
    fn test_graph_empty() {
        let g = EditGraph::new();
        assert!(g.head().is_none());
        assert!(g.nodes().is_empty());
    }

    #[test]
    fn test_graph_add_and_head() {
        let mut g = EditGraph::new();
        let hash = g
            .add_op(EditOp::Mute { stem_id: 0 }, Some("mute vocals".into()))
            .unwrap();
        assert_eq!(g.head().unwrap().hash, hash);
        assert_eq!(g.nodes().len(), 1);
        assert!(g.head().unwrap().parent_hash.is_none());
    }

    #[test]
    fn test_graph_parent_linkage() {
        let mut g = EditGraph::new();
        let h1 = g.add_op(EditOp::Mute { stem_id: 0 }, None).unwrap();
        let _h2 = g
            .add_op(
                EditOp::Gain {
                    stem_id: 1,
                    db: 3.0,
                },
                None,
            )
            .unwrap();
        assert_eq!(g.nodes()[1].parent_hash, Some(h1));
    }

    #[test]
    fn test_graph_ops_for_stem() {
        let mut g = EditGraph::new();
        g.add_op(EditOp::Mute { stem_id: 0 }, None).unwrap();
        g.add_op(
            EditOp::Gain {
                stem_id: 1,
                db: 3.0,
            },
            None,
        )
        .unwrap();
        g.add_op(
            EditOp::Gain {
                stem_id: 0,
                db: -6.0,
            },
            None,
        )
        .unwrap();

        let ops_0 = g.ops_for_stem(0);
        assert_eq!(ops_0.len(), 2);
        let ops_1 = g.ops_for_stem(1);
        assert_eq!(ops_1.len(), 1);
        let ops_2 = g.ops_for_stem(2);
        assert!(ops_2.is_empty());
    }

    #[test]
    fn test_graph_rejects_invalid_op() {
        let mut g = EditGraph::new();
        let result = g.add_op(
            EditOp::Gain {
                stem_id: 0,
                db: 100.0,
            },
            None,
        );
        assert!(result.is_err());
        assert!(g.nodes().is_empty());
    }

    #[test]
    fn test_graph_validate_ok() {
        let mut g = EditGraph::new();
        g.add_op(EditOp::Mute { stem_id: 0 }, None).unwrap();
        g.add_op(
            EditOp::Gain {
                stem_id: 0,
                db: 3.0,
            },
            None,
        )
        .unwrap();
        assert!(g.validate().is_ok());
    }

    #[test]
    fn test_graph_json_round_trip() {
        let mut g = EditGraph::new();
        g.add_op(EditOp::Mute { stem_id: 0 }, Some("mute".into()))
            .unwrap();
        g.add_op(
            EditOp::Gain {
                stem_id: 1,
                db: -3.0,
            },
            None,
        )
        .unwrap();

        let json = g.to_json().unwrap();
        let g2 = EditGraph::from_json(&json).unwrap();
        assert_eq!(g2.nodes().len(), 2);
        assert_eq!(g2.nodes()[0].hash, g.nodes()[0].hash);
        assert_eq!(g2.nodes()[1].hash, g.nodes()[1].hash);
        assert!(g2.validate().is_ok());
    }

    // ── apply_edit_ops tests ────────────────────────────────────────

    #[test]
    fn test_apply_gain() {
        let samples = vec![1.0_f32; 10];
        let op = EditOp::Gain {
            stem_id: 0,
            db: 0.0,
        };
        let result = apply_edit_ops(&samples, &[&op], 44100, 1);
        // 0 dB gain → unchanged
        for s in &result {
            assert!((s - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_apply_gain_6db() {
        let samples = vec![0.5_f32; 4];
        let op = EditOp::Gain {
            stem_id: 0,
            db: 6.0,
        };
        let result = apply_edit_ops(&samples, &[&op], 44100, 1);
        let expected = 0.5 * 10.0_f32.powf(6.0 / 20.0);
        for s in &result {
            assert!((s - expected).abs() < 1e-4, "got {s}, expected {expected}");
        }
    }

    #[test]
    fn test_apply_mute() {
        let samples = vec![0.5, -0.3, 0.7, 1.0];
        let op = EditOp::Mute { stem_id: 0 };
        let result = apply_edit_ops(&samples, &[&op], 44100, 1);
        assert!(result.iter().all(|&s| s == 0.0));
    }

    #[test]
    fn test_apply_trim() {
        // 10 frames at 10 Hz → 1 second.
        let samples: Vec<f32> = (0..10).map(|i| i as f32).collect();
        let op = EditOp::Trim {
            stem_id: 0,
            start_s: 0.2, // frame 2
            end_s: 0.5,   // frame 5
        };
        let result = apply_edit_ops(&samples, &[&op], 10, 1);
        assert_eq!(result, vec![2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_apply_reverse_mono() {
        let samples = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let op = EditOp::Reverse { stem_id: 0 };
        let result = apply_edit_ops(&samples, &[&op], 44100, 1);
        assert_eq!(result, vec![5.0, 4.0, 3.0, 2.0, 1.0]);
    }

    #[test]
    fn test_apply_reverse_stereo() {
        // 3 frames, stereo: [L0,R0, L1,R1, L2,R2]
        let samples = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let op = EditOp::Reverse { stem_id: 0 };
        let result = apply_edit_ops(&samples, &[&op], 44100, 2);
        // Reversed frames: [L2,R2, L1,R1, L0,R0]
        assert_eq!(result, vec![5.0, 6.0, 3.0, 4.0, 1.0, 2.0]);
    }

    #[test]
    fn test_apply_fade_in() {
        // 4 frames mono at 4 Hz = 1 second.  Fade in 0.5s = 2 frames.
        let samples = vec![1.0, 1.0, 1.0, 1.0];
        let op = EditOp::Fade {
            stem_id: 0,
            fade_in_s: 0.5,
            fade_out_s: 0.0,
        };
        let result = apply_edit_ops(&samples, &[&op], 4, 1);
        // Frame 0: gain = 0/2 = 0.0  → 0.0
        // Frame 1: gain = 1/2 = 0.5  → 0.5
        // Frame 2: gain = 1.0 (no fade) → 1.0
        // Frame 3: gain = 1.0 (no fade) → 1.0
        assert!((result[0] - 0.0).abs() < 1e-6);
        assert!((result[1] - 0.5).abs() < 1e-6);
        assert!((result[2] - 1.0).abs() < 1e-6);
        assert!((result[3] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_apply_fade_out() {
        let samples = vec![1.0, 1.0, 1.0, 1.0];
        let op = EditOp::Fade {
            stem_id: 0,
            fade_in_s: 0.0,
            fade_out_s: 0.5, // 2 frames at 4 Hz
        };
        let result = apply_edit_ops(&samples, &[&op], 4, 1);
        // Frame 0: 1.0
        // Frame 1: 1.0
        // Frame 2: gain = 2/2 = 1.0 → 1.0
        // Frame 3: gain = 1/2 = 0.5 → 0.5
        assert!((result[0] - 1.0).abs() < 1e-6);
        assert!((result[1] - 1.0).abs() < 1e-6);
        assert!((result[2] - 1.0).abs() < 1e-6);
        assert!((result[3] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_apply_pan_center() {
        let samples = vec![1.0, 1.0, 1.0, 1.0]; // 2 frames stereo
        let op = EditOp::Pan {
            stem_id: 0,
            position: 0.0, // center
        };
        let result = apply_edit_ops(&samples, &[&op], 44100, 2);
        // Center pan: left_gain ≈ cos(π/4), right_gain ≈ sin(π/4) — both ≈ 0.707
        let expected = std::f32::consts::FRAC_1_SQRT_2;
        for chunk in result.chunks(2) {
            assert!(
                (chunk[0] - expected).abs() < 1e-4,
                "L={}, expected ~{expected}",
                chunk[0]
            );
            assert!(
                (chunk[1] - expected).abs() < 1e-4,
                "R={}, expected ~{expected}",
                chunk[1]
            );
        }
    }

    #[test]
    fn test_apply_pan_hard_left() {
        let samples = vec![1.0, 1.0]; // 1 frame stereo
        let op = EditOp::Pan {
            stem_id: 0,
            position: -1.0, // hard left
        };
        let result = apply_edit_ops(&samples, &[&op], 44100, 2);
        // Hard left: θ = 0 → cos(0)=1, sin(0)=0
        assert!((result[0] - 1.0).abs() < 1e-4, "L should be ~1.0");
        assert!(result[1].abs() < 1e-4, "R should be ~0.0");
    }

    #[test]
    fn test_apply_pan_hard_right() {
        let samples = vec![1.0, 1.0]; // 1 frame stereo
        let op = EditOp::Pan {
            stem_id: 0,
            position: 1.0, // hard right
        };
        let result = apply_edit_ops(&samples, &[&op], 44100, 2);
        // Hard right: θ = π/2 → cos(π/2)≈0, sin(π/2)=1
        assert!(result[0].abs() < 1e-4, "L should be ~0.0");
        assert!((result[1] - 1.0).abs() < 1e-4, "R should be ~1.0");
    }

    #[test]
    fn test_apply_multiple_ops() {
        // Apply gain then mute → result should be all zeros.
        let samples = vec![1.0; 8];
        let gain = EditOp::Gain {
            stem_id: 0,
            db: 6.0,
        };
        let mute = EditOp::Mute { stem_id: 0 };
        let result = apply_edit_ops(&samples, &[&gain, &mute], 44100, 1);
        assert!(result.iter().all(|&s| s == 0.0));
    }

    #[test]
    fn test_apply_empty_ops() {
        let samples = vec![1.0, 2.0, 3.0];
        let result = apply_edit_ops(&samples, &[], 44100, 1);
        assert_eq!(result, samples);
    }

    #[test]
    fn test_apply_timestretch_slower() {
        // 4 frames mono, stretch by 2.0 → 8 frames
        let samples = vec![0.0, 1.0, 2.0, 3.0];
        let op = EditOp::TimeStretch {
            stem_id: 0,
            factor: 2.0,
        };
        let result = apply_edit_ops(&samples, &[&op], 44100, 1);
        assert_eq!(result.len(), 8);
        // First sample should be 0.0, last should approach 3.0
        assert!((result[0] - 0.0).abs() < 1e-4);
        assert!((result[result.len() - 1] - 3.0).abs() < 0.5);
    }

    #[test]
    fn test_apply_timestretch_faster() {
        // 8 frames mono, stretch by 0.5 → 4 frames
        let samples: Vec<f32> = (0..8).map(|i| i as f32).collect();
        let op = EditOp::TimeStretch {
            stem_id: 0,
            factor: 0.5,
        };
        let result = apply_edit_ops(&samples, &[&op], 44100, 1);
        assert_eq!(result.len(), 4);
    }

    #[test]
    fn test_apply_eq_flat() {
        // 0 dB gain EQ should leave signal essentially unchanged
        let samples = vec![1.0_f32; 100];
        let op = EditOp::Eq {
            stem_id: 0,
            freq_hz: 1000.0,
            gain_db: 0.0,
            q: 1.0,
        };
        let result = apply_edit_ops(&samples, &[&op], 44100, 1);
        assert_eq!(result.len(), 100);
        // After transient settles, output should be close to input
        for s in &result[10..] {
            assert!(
                (s - 1.0).abs() < 0.01,
                "0 dB EQ should not change signal significantly, got {s}"
            );
        }
    }

    #[test]
    fn test_apply_eq_boost() {
        // Boosting a frequency should increase energy
        // Generate a 1kHz sine wave at 44100 Hz
        let sr = 44100.0_f32;
        let freq = 1000.0_f32;
        let samples: Vec<f32> = (0..4410)
            .map(|i| (2.0 * std::f32::consts::PI * freq * i as f32 / sr).sin() * 0.5)
            .collect();
        let op = EditOp::Eq {
            stem_id: 0,
            freq_hz: 1000.0,
            gain_db: 12.0,
            q: 1.0,
        };
        let result = apply_edit_ops(&samples, &[&op], 44100, 1);
        // RMS of output should be greater than RMS of input
        let rms_in: f32 =
            (samples.iter().map(|s| s * s).sum::<f32>() / samples.len() as f32).sqrt();
        let rms_out: f32 = (result.iter().map(|s| s * s).sum::<f32>() / result.len() as f32).sqrt();
        assert!(
            rms_out > rms_in * 1.5,
            "12 dB boost should significantly increase RMS: in={rms_in}, out={rms_out}"
        );
    }
}
