//! Diff and patch operations between edit history versions.
//!
//! [`compute_diff`] compares two [`EditHistory`] instances and produces an
//! [`EditDiff`] describing the operations added, removed, or shared.
//! [`apply_patch`] takes a diff and applies its added operations as a new
//! commit on an existing history.

use serde::{Deserialize, Serialize};

use crate::error::Result;
use crate::graph::EditOp;
use crate::history::EditHistory;

/// A diff between two edit histories.
///
/// Computed by [`compute_diff`], this struct captures the operations present
/// in one history but not the other, as well as the count of shared ops.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EditDiff {
    /// Operations present in history B but not in A.
    pub added_ops: Vec<EditOp>,
    /// Operations present in history A but not in B.
    pub removed_ops: Vec<EditOp>,
    /// Number of operations shared between both histories.
    pub common_ops: usize,
}

/// Compute the diff between two edit histories.
///
/// Flattens all operations from each history and compares them sequentially.
/// Operations that are identical (via [`PartialEq`]) and appear in the same
/// position are considered common; divergences are tracked as added or removed.
pub fn compute_diff(history_a: &EditHistory, history_b: &EditHistory) -> EditDiff {
    let ops_a: Vec<&EditOp> = history_a.all_ops();
    let ops_b: Vec<&EditOp> = history_b.all_ops();

    let mut common_ops = 0usize;
    let mut added_ops = Vec::new();
    let mut removed_ops = Vec::new();

    // Find the common prefix.
    let min_len = ops_a.len().min(ops_b.len());
    let mut prefix_len = 0;
    for i in 0..min_len {
        if ops_a[i] == ops_b[i] {
            prefix_len += 1;
        } else {
            break;
        }
    }
    common_ops += prefix_len;

    // Everything after the common prefix in A is "removed".
    for op in &ops_a[prefix_len..] {
        removed_ops.push((*op).clone());
    }

    // Everything after the common prefix in B is "added".
    for op in &ops_b[prefix_len..] {
        added_ops.push((*op).clone());
    }

    EditDiff {
        added_ops,
        removed_ops,
        common_ops,
    }
}

/// Apply a diff as a new commit on an existing history.
///
/// Creates a new commit containing the `added_ops` from the diff. This is
/// a forward-only operation — removed ops are **not** undone automatically.
/// Returns the BLAKE3 hash of the new commit.
pub fn apply_patch(
    history: &mut EditHistory,
    diff: &EditDiff,
    message: impl Into<String>,
) -> Result<[u8; 32]> {
    history.commit(diff.added_ops.clone(), message, None)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::EditOp;

    #[test]
    fn test_diff_identical_histories() {
        let mut a = EditHistory::new();
        a.commit(vec![EditOp::Mute { stem_id: 0 }], "init", None)
            .unwrap();

        let mut b = EditHistory::new();
        b.commit(vec![EditOp::Mute { stem_id: 0 }], "init", None)
            .unwrap();

        let diff = compute_diff(&a, &b);
        assert_eq!(diff.common_ops, 1);
        assert!(diff.added_ops.is_empty());
        assert!(diff.removed_ops.is_empty());
    }

    #[test]
    fn test_diff_empty_histories() {
        let a = EditHistory::new();
        let b = EditHistory::new();
        let diff = compute_diff(&a, &b);
        assert_eq!(diff.common_ops, 0);
        assert!(diff.added_ops.is_empty());
        assert!(diff.removed_ops.is_empty());
    }

    #[test]
    fn test_diff_b_has_more_ops() {
        let mut a = EditHistory::new();
        a.commit(vec![EditOp::Mute { stem_id: 0 }], "init", None)
            .unwrap();

        let mut b = EditHistory::new();
        b.commit(vec![EditOp::Mute { stem_id: 0 }], "init", None)
            .unwrap();
        b.commit(
            vec![EditOp::Gain {
                stem_id: 1,
                db: 3.0,
            }],
            "boost",
            None,
        )
        .unwrap();

        let diff = compute_diff(&a, &b);
        assert_eq!(diff.common_ops, 1);
        assert_eq!(diff.added_ops.len(), 1);
        assert!(diff.removed_ops.is_empty());
    }

    #[test]
    fn test_diff_a_has_more_ops() {
        let mut a = EditHistory::new();
        a.commit(vec![EditOp::Mute { stem_id: 0 }], "init", None)
            .unwrap();
        a.commit(
            vec![EditOp::Gain {
                stem_id: 1,
                db: 3.0,
            }],
            "boost",
            None,
        )
        .unwrap();

        let mut b = EditHistory::new();
        b.commit(vec![EditOp::Mute { stem_id: 0 }], "init", None)
            .unwrap();

        let diff = compute_diff(&a, &b);
        assert_eq!(diff.common_ops, 1);
        assert!(diff.added_ops.is_empty());
        assert_eq!(diff.removed_ops.len(), 1);
    }

    #[test]
    fn test_diff_divergent_histories() {
        let mut a = EditHistory::new();
        a.commit(vec![EditOp::Mute { stem_id: 0 }], "init", None)
            .unwrap();
        a.commit(
            vec![EditOp::Gain {
                stem_id: 1,
                db: 3.0,
            }],
            "boost",
            None,
        )
        .unwrap();

        let mut b = EditHistory::new();
        b.commit(vec![EditOp::Mute { stem_id: 0 }], "init", None)
            .unwrap();
        b.commit(
            vec![EditOp::Pan {
                stem_id: 1,
                position: -0.5,
            }],
            "pan left",
            None,
        )
        .unwrap();

        let diff = compute_diff(&a, &b);
        assert_eq!(diff.common_ops, 1);
        assert_eq!(diff.added_ops.len(), 1);
        assert_eq!(diff.removed_ops.len(), 1);
    }

    #[test]
    fn test_diff_a_empty() {
        let a = EditHistory::new();
        let mut b = EditHistory::new();
        b.commit(vec![EditOp::Mute { stem_id: 0 }], "init", None)
            .unwrap();

        let diff = compute_diff(&a, &b);
        assert_eq!(diff.common_ops, 0);
        assert_eq!(diff.added_ops.len(), 1);
        assert!(diff.removed_ops.is_empty());
    }

    #[test]
    fn test_apply_patch() {
        let mut a = EditHistory::new();
        a.commit(vec![EditOp::Mute { stem_id: 0 }], "init", None)
            .unwrap();

        let mut b = EditHistory::new();
        b.commit(vec![EditOp::Mute { stem_id: 0 }], "init", None)
            .unwrap();
        b.commit(
            vec![EditOp::Gain {
                stem_id: 1,
                db: 3.0,
            }],
            "boost",
            None,
        )
        .unwrap();

        let diff = compute_diff(&a, &b);
        let hash = apply_patch(&mut a, &diff, "apply boost from B").unwrap();

        assert_eq!(a.log().len(), 2);
        assert_eq!(a.head().unwrap().hash, hash);
        assert_eq!(a.all_ops().len(), 2);
    }

    #[test]
    fn test_apply_empty_patch() {
        let mut a = EditHistory::new();
        a.commit(vec![EditOp::Mute { stem_id: 0 }], "init", None)
            .unwrap();

        let diff = EditDiff {
            added_ops: vec![],
            removed_ops: vec![],
            common_ops: 1,
        };
        // Empty ops commit is allowed.
        let hash = apply_patch(&mut a, &diff, "no-op patch").unwrap();
        assert_eq!(a.log().len(), 2);
        assert_eq!(a.head().unwrap().hash, hash);
    }

    #[test]
    fn test_diff_serialization_round_trip() {
        let diff = EditDiff {
            added_ops: vec![EditOp::Mute { stem_id: 0 }],
            removed_ops: vec![EditOp::Gain {
                stem_id: 1,
                db: 3.0,
            }],
            common_ops: 5,
        };
        let json = serde_json::to_string_pretty(&diff).unwrap();
        let diff2: EditDiff = serde_json::from_str(&json).unwrap();
        assert_eq!(diff2.added_ops.len(), 1);
        assert_eq!(diff2.removed_ops.len(), 1);
        assert_eq!(diff2.common_ops, 5);
    }
}
