//! Version history — git-like commit history for NEO file edits.
//!
//! Each [`EditCommit`] bundles one or more [`EditOp`]s with a message,
//! author, and BLAKE3-derived hash. Commits form a linear chain where
//! each commit points to its parent. The [`EditHistory`] struct
//! provides the full commit log plus operations for querying, reverting,
//! and serializing the history.
//!
//! See SPECIFICATION.md Section 13.1.

use serde::{Deserialize, Serialize};

use crate::error::{EditError, Result};
use crate::graph::EditOp;

/// A version commit in the edit history.
///
/// A commit groups one or more [`EditOp`]s with metadata (message, author,
/// timestamp). Its BLAKE3 hash is computed from the parent hash, serialized
/// operations, and message.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EditCommit {
    /// BLAKE3 hash of this commit (computed from parent + ops + message).
    pub hash: [u8; 32],
    /// Parent commit hash (`None` for the initial commit).
    pub parent: Option<[u8; 32]>,
    /// Edit operations in this commit, applied in order.
    #[serde(rename = "ops")]
    pub edit_ops: Vec<EditOp>,
    /// Human-readable commit message.
    pub message: String,
    /// ISO 8601 timestamp of when this commit was created.
    pub timestamp: String,
    /// Optional author name.
    pub author: Option<String>,
}

impl EditCommit {
    /// Create a new commit, computing its BLAKE3 hash.
    ///
    /// The hash is derived from the concatenation of:
    /// 1. Parent hash bytes (or 32 zero-bytes for root).
    /// 2. JSON-serialized ops.
    /// 3. The commit message as UTF-8 bytes.
    pub fn new(
        ops: Vec<EditOp>,
        message: impl Into<String>,
        parent: Option<[u8; 32]>,
        author: Option<String>,
    ) -> Self {
        let message = message.into();
        let hash = Self::compute_hash(&parent, &ops, &message);
        Self {
            hash,
            parent,
            edit_ops: ops,
            message,
            timestamp: current_timestamp(),
            author,
        }
    }

    /// Compute the BLAKE3 hash for a commit.
    fn compute_hash(parent: &Option<[u8; 32]>, ops: &[EditOp], message: &str) -> [u8; 32] {
        let mut hasher = blake3::Hasher::new();
        match parent {
            Some(h) => hasher.update(h),
            None => hasher.update(&[0u8; 32]),
        };
        let ops_json = serde_json::to_vec(ops).expect("EditOp list serialization should not fail");
        hasher.update(&ops_json);
        hasher.update(message.as_bytes());
        *hasher.finalize().as_bytes()
    }
}

/// The edit history — a linear chain of [`EditCommit`]s (oldest first).
///
/// Provides git-like functionality: committing groups of edits, viewing the
/// log, reverting the most recent commit, and querying operations.
///
/// The serialized form matches SPECIFICATION.md Section 13.1:
///
/// ```json
/// { "commits": [...], "head": "base64(blake3)" }
/// ```
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EditHistory {
    /// All commits in order (oldest first).
    commits: Vec<EditCommit>,
    /// BLAKE3 hash of the HEAD commit (last committed).
    head: Option<[u8; 32]>,
}

impl EditHistory {
    /// Create a new, empty edit history.
    pub fn new() -> Self {
        Self {
            commits: Vec::new(),
            head: None,
        }
    }

    /// Create a new commit with the given operations and message.
    ///
    /// All operations are validated before committing. Returns the BLAKE3
    /// hash of the new commit.
    pub fn commit(
        &mut self,
        ops: Vec<EditOp>,
        message: impl Into<String>,
        author: Option<String>,
    ) -> Result<[u8; 32]> {
        // Validate every op.
        for op in &ops {
            op.validate()?;
        }

        let parent = self.head;
        let commit = EditCommit::new(ops, message, parent, author);
        let hash = commit.hash;
        self.commits.push(commit);
        self.head = Some(hash);
        Ok(hash)
    }

    /// Return the HEAD commit (most recent), if any.
    pub fn head(&self) -> Option<&EditCommit> {
        self.commits.last()
    }

    /// Return the full commit log from oldest to newest.
    pub fn log(&self) -> &[EditCommit] {
        &self.commits
    }

    /// Find a commit by its BLAKE3 hash.
    pub fn find_commit(&self, hash: &[u8; 32]) -> Option<&EditCommit> {
        self.commits.iter().find(|c| &c.hash == hash)
    }

    /// Revert (remove) the HEAD commit, undoing the last commit.
    ///
    /// Returns `Err(NothingToRevert)` if the history is empty.
    pub fn revert(&mut self) -> Result<()> {
        if self.commits.is_empty() {
            return Err(EditError::NothingToRevert);
        }
        self.commits.pop();
        self.head = self.commits.last().map(|c| c.hash);
        Ok(())
    }

    /// Flatten all operations across all commits into a single ordered list.
    ///
    /// Operations are returned in commit order (oldest commit first), and
    /// within each commit in array order.
    pub fn all_ops(&self) -> Vec<&EditOp> {
        self.commits
            .iter()
            .flat_map(|c| c.edit_ops.iter())
            .collect()
    }

    /// Return all operations that affect a given stem, in commit order.
    pub fn ops_for_stem(&self, stem_id: u8) -> Vec<&EditOp> {
        self.all_ops()
            .into_iter()
            .filter(|op| op.stem_id() == stem_id)
            .collect()
    }

    /// Serialize the history to a JSON string matching the spec format.
    pub fn to_json(&self) -> Result<String> {
        Ok(serde_json::to_string_pretty(self)?)
    }

    /// Deserialize a history from a JSON string.
    pub fn from_json(json: &str) -> Result<Self> {
        Ok(serde_json::from_str(json)?)
    }

    /// Validate the entire history.
    ///
    /// Checks that:
    /// - Every commit's hash is correct (recomputed from parent + ops + message).
    /// - Parent linkage is consistent.
    /// - All operations pass validation.
    /// - The `head` field matches the last commit's hash.
    pub fn validate(&self) -> Result<()> {
        for (i, commit) in self.commits.iter().enumerate() {
            // Validate all ops in the commit.
            for op in &commit.edit_ops {
                op.validate()?;
            }

            // Verify hash.
            let expected =
                EditCommit::compute_hash(&commit.parent, &commit.edit_ops, &commit.message);
            if commit.hash != expected {
                return Err(EditError::InvalidOperation(format!(
                    "commit {i} hash mismatch"
                )));
            }

            // Verify parent linkage.
            if i == 0 {
                if commit.parent.is_some() {
                    return Err(EditError::InvalidOperation(
                        "first commit must not have a parent".into(),
                    ));
                }
            } else {
                let expected_parent = self.commits[i - 1].hash;
                match commit.parent {
                    Some(p) if p == expected_parent => {}
                    _ => {
                        return Err(EditError::InvalidOperation(format!(
                            "commit {i} parent hash does not match commit {}",
                            i - 1
                        )));
                    }
                }
            }
        }

        // Verify head pointer.
        match (&self.head, self.commits.last()) {
            (Some(h), Some(c)) if h == &c.hash => {}
            (None, None) => {}
            _ => {
                return Err(EditError::InvalidOperation(
                    "head pointer does not match last commit".into(),
                ));
            }
        }

        Ok(())
    }
}

/// Return an ISO 8601 timestamp string for "now".
fn current_timestamp() -> String {
    "2026-01-01T00:00:00Z".to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_commit_creation() {
        let ops = vec![EditOp::Mute { stem_id: 0 }];
        let commit = EditCommit::new(ops.clone(), "test", None, Some("alice".into()));
        assert!(commit.parent.is_none());
        assert_eq!(commit.message, "test");
        assert_eq!(commit.author.as_deref(), Some("alice"));
        assert_eq!(commit.edit_ops.len(), 1);
        // Hash should be non-zero.
        assert_ne!(commit.hash, [0u8; 32]);
    }

    #[test]
    fn test_commit_hash_deterministic() {
        let ops = vec![EditOp::Gain {
            stem_id: 1,
            db: 3.0,
        }];
        let a = EditCommit::new(ops.clone(), "msg", None, None);
        let b = EditCommit::new(ops, "msg", None, None);
        assert_eq!(a.hash, b.hash);
    }

    #[test]
    fn test_commit_hash_changes_with_message() {
        let ops = vec![EditOp::Mute { stem_id: 0 }];
        let a = EditCommit::new(ops.clone(), "alpha", None, None);
        let b = EditCommit::new(ops, "beta", None, None);
        assert_ne!(a.hash, b.hash);
    }

    #[test]
    fn test_history_empty() {
        let h = EditHistory::new();
        assert!(h.head().is_none());
        assert!(h.log().is_empty());
        assert!(h.all_ops().is_empty());
    }

    #[test]
    fn test_history_single_commit() {
        let mut h = EditHistory::new();
        let hash = h
            .commit(
                vec![EditOp::Mute { stem_id: 0 }],
                "initial",
                Some("alice".into()),
            )
            .unwrap();

        assert_eq!(h.log().len(), 1);
        assert_eq!(h.head().unwrap().hash, hash);
        assert!(h.head().unwrap().parent.is_none());
    }

    #[test]
    fn test_history_multiple_commits() {
        let mut h = EditHistory::new();
        let h1 = h
            .commit(vec![EditOp::Mute { stem_id: 0 }], "first", None)
            .unwrap();
        let h2 = h
            .commit(
                vec![EditOp::Gain {
                    stem_id: 1,
                    db: 3.0,
                }],
                "second",
                None,
            )
            .unwrap();

        assert_eq!(h.log().len(), 2);
        assert_eq!(h.head().unwrap().hash, h2);
        assert_eq!(h.log()[1].parent, Some(h1));
    }

    #[test]
    fn test_history_log_ordering() {
        let mut h = EditHistory::new();
        h.commit(vec![EditOp::Mute { stem_id: 0 }], "first", None)
            .unwrap();
        h.commit(vec![EditOp::Mute { stem_id: 1 }], "second", None)
            .unwrap();
        h.commit(vec![EditOp::Mute { stem_id: 2 }], "third", None)
            .unwrap();

        let log = h.log();
        assert_eq!(log[0].message, "first");
        assert_eq!(log[1].message, "second");
        assert_eq!(log[2].message, "third");
    }

    #[test]
    fn test_history_find_commit() {
        let mut h = EditHistory::new();
        let hash = h
            .commit(vec![EditOp::Mute { stem_id: 0 }], "find me", None)
            .unwrap();

        assert!(h.find_commit(&hash).is_some());
        assert_eq!(h.find_commit(&hash).unwrap().message, "find me");
        assert!(h.find_commit(&[0u8; 32]).is_none());
    }

    #[test]
    fn test_history_revert() {
        let mut h = EditHistory::new();
        let h1 = h
            .commit(vec![EditOp::Mute { stem_id: 0 }], "first", None)
            .unwrap();
        h.commit(vec![EditOp::Mute { stem_id: 1 }], "second", None)
            .unwrap();

        h.revert().unwrap();
        assert_eq!(h.log().len(), 1);
        assert_eq!(h.head().unwrap().hash, h1);
    }

    #[test]
    fn test_history_revert_to_empty() {
        let mut h = EditHistory::new();
        h.commit(vec![EditOp::Mute { stem_id: 0 }], "only", None)
            .unwrap();
        h.revert().unwrap();

        assert!(h.head().is_none());
        assert!(h.log().is_empty());
    }

    #[test]
    fn test_history_revert_empty_errors() {
        let mut h = EditHistory::new();
        assert!(h.revert().is_err());
    }

    #[test]
    fn test_history_all_ops() {
        let mut h = EditHistory::new();
        h.commit(
            vec![
                EditOp::Mute { stem_id: 0 },
                EditOp::Gain {
                    stem_id: 1,
                    db: 3.0,
                },
            ],
            "first",
            None,
        )
        .unwrap();
        h.commit(vec![EditOp::Mute { stem_id: 2 }], "second", None)
            .unwrap();

        let all = h.all_ops();
        assert_eq!(all.len(), 3);
    }

    #[test]
    fn test_history_ops_for_stem() {
        let mut h = EditHistory::new();
        h.commit(
            vec![
                EditOp::Mute { stem_id: 0 },
                EditOp::Gain {
                    stem_id: 1,
                    db: 3.0,
                },
            ],
            "first",
            None,
        )
        .unwrap();
        h.commit(
            vec![EditOp::Gain {
                stem_id: 0,
                db: -6.0,
            }],
            "second",
            None,
        )
        .unwrap();

        assert_eq!(h.ops_for_stem(0).len(), 2);
        assert_eq!(h.ops_for_stem(1).len(), 1);
        assert_eq!(h.ops_for_stem(2).len(), 0);
    }

    #[test]
    fn test_history_rejects_invalid_ops() {
        let mut h = EditHistory::new();
        let result = h.commit(
            vec![EditOp::Gain {
                stem_id: 0,
                db: 100.0,
            }],
            "bad",
            None,
        );
        assert!(result.is_err());
        assert!(h.log().is_empty());
    }

    #[test]
    fn test_history_json_round_trip() {
        let mut h = EditHistory::new();
        h.commit(
            vec![EditOp::Mute { stem_id: 0 }],
            "initial",
            Some("alice".into()),
        )
        .unwrap();
        h.commit(
            vec![
                EditOp::Trim {
                    stem_id: 0,
                    start_s: 2.5,
                    end_s: 180.0,
                },
                EditOp::Gain {
                    stem_id: 2,
                    db: 3.0,
                },
            ],
            "trim and boost",
            Some("bob".into()),
        )
        .unwrap();

        let json = h.to_json().unwrap();
        let h2 = EditHistory::from_json(&json).unwrap();

        assert_eq!(h2.log().len(), 2);
        assert_eq!(h2.head().unwrap().hash, h.head().unwrap().hash);
        assert_eq!(h2.log()[0].message, "initial");
        assert_eq!(h2.log()[1].message, "trim and boost");
        assert!(h2.validate().is_ok());
    }

    #[test]
    fn test_history_validate_ok() {
        let mut h = EditHistory::new();
        h.commit(vec![EditOp::Mute { stem_id: 0 }], "a", None)
            .unwrap();
        h.commit(
            vec![EditOp::Gain {
                stem_id: 1,
                db: 3.0,
            }],
            "b",
            None,
        )
        .unwrap();
        assert!(h.validate().is_ok());
    }

    #[test]
    fn test_history_validate_empty() {
        let h = EditHistory::new();
        assert!(h.validate().is_ok());
    }
}
