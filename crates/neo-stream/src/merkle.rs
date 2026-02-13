//! Merkle tree chunking for content-addressed storage and verification.
//!
//! Implements a binary Merkle tree using BLAKE3 hashes, as specified in
//! SPECIFICATION.md Section 14.1. Data is split into fixed-size chunks
//! (default 256 KiB), each chunk is hashed to form a leaf, and parent
//! nodes are computed by hashing the concatenation of their children's hashes.
//!
//! # Examples
//!
//! ```
//! use neo_stream::merkle::{MerkleTree, DEFAULT_CHUNK_SIZE};
//!
//! let data = vec![42u8; 1024];
//! let tree = MerkleTree::build(&data, DEFAULT_CHUNK_SIZE).unwrap();
//! assert!(tree.verify_block(0, &data[..1024]));
//! ```

use serde::{Deserialize, Serialize};

use crate::error::{Result, StreamError};

/// Default chunk size for Merkle tree leaves (256 KiB).
pub const DEFAULT_CHUNK_SIZE: usize = 256 * 1024;

/// A node in the Merkle tree.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MerkleNode {
    /// BLAKE3 hash of this node.
    pub hash: [u8; 32],
    /// Child node indices (empty for leaf nodes).
    pub children: Vec<usize>,
    /// Byte range `[start, end)` this node covers in the original data.
    pub byte_range: (u64, u64),
    /// Whether this node is a leaf (contains actual data).
    pub is_leaf: bool,
}

/// An inclusion proof for a single block in the Merkle tree.
///
/// Contains the sibling hashes along the path from the leaf to the root,
/// allowing verification of a block against the known root hash without
/// needing the full tree.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MerkleProof {
    /// Index of the block this proof is for.
    pub block_index: usize,
    /// BLAKE3 hash of the block data.
    pub block_hash: [u8; 32],
    /// Sibling hashes along the path from leaf to root.
    /// Each entry is `(hash, is_left)` where `is_left` indicates whether
    /// the sibling is on the left side of the parent computation.
    pub siblings: Vec<([u8; 32], bool)>,
}

/// A binary Merkle tree built from chunked data using BLAKE3.
///
/// The tree is stored as a flat vector of [`MerkleNode`]s. Leaf nodes
/// correspond to fixed-size chunks of the original data, and internal
/// nodes are computed by hashing `left_hash || right_hash`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MerkleTree {
    /// All nodes in the tree, stored level by level (root is last).
    nodes: Vec<MerkleNode>,
    /// Index of the root node in `nodes`.
    root_index: usize,
    /// Number of leaf nodes.
    leaf_count: usize,
    /// Indices of leaf nodes in the `nodes` vec, in order.
    leaf_indices: Vec<usize>,
    /// Chunk size used to build the tree.
    chunk_size: usize,
}

/// Compute the parent hash from two child hashes using BLAKE3.
///
/// The parent hash is `BLAKE3(left_hash || right_hash)`.
fn compute_parent_hash(left: &[u8; 32], right: &[u8; 32]) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();
    hasher.update(left);
    hasher.update(right);
    *hasher.finalize().as_bytes()
}

impl MerkleTree {
    /// Build a Merkle tree from the given data, splitting it into chunks
    /// of the specified size.
    ///
    /// # Errors
    ///
    /// Returns [`StreamError::EmptyData`] if `data` is empty.
    /// Returns [`StreamError::InvalidChunkSize`] if `chunk_size` is zero.
    pub fn build(data: &[u8], chunk_size: usize) -> Result<Self> {
        if data.is_empty() {
            return Err(StreamError::EmptyData);
        }
        if chunk_size == 0 {
            return Err(StreamError::InvalidChunkSize(0));
        }

        let mut nodes = Vec::new();
        let mut leaf_indices = Vec::new();

        // Create leaf nodes from data chunks
        let mut offset = 0usize;
        while offset < data.len() {
            let end = (offset + chunk_size).min(data.len());
            let chunk = &data[offset..end];
            let hash = *blake3::hash(chunk).as_bytes();

            let idx = nodes.len();
            leaf_indices.push(idx);
            nodes.push(MerkleNode {
                hash,
                children: Vec::new(),
                byte_range: (offset as u64, end as u64),
                is_leaf: true,
            });
            offset = end;
        }

        let leaf_count = leaf_indices.len();

        // Build tree bottom-up
        // current_level holds the indices of nodes at the current level
        let mut current_level: Vec<usize> = leaf_indices.clone();

        while current_level.len() > 1 {
            let mut next_level = Vec::new();

            let mut i = 0;
            while i < current_level.len() {
                if i + 1 < current_level.len() {
                    // Pair two nodes
                    let left_idx = current_level[i];
                    let right_idx = current_level[i + 1];
                    let left_hash = nodes[left_idx].hash;
                    let right_hash = nodes[right_idx].hash;
                    let parent_hash = compute_parent_hash(&left_hash, &right_hash);

                    let left_start = nodes[left_idx].byte_range.0;
                    let right_end = nodes[right_idx].byte_range.1;

                    let parent_idx = nodes.len();
                    nodes.push(MerkleNode {
                        hash: parent_hash,
                        children: vec![left_idx, right_idx],
                        byte_range: (left_start, right_end),
                        is_leaf: false,
                    });
                    next_level.push(parent_idx);
                    i += 2;
                } else {
                    // Odd node out — promote to next level
                    next_level.push(current_level[i]);
                    i += 1;
                }
            }

            current_level = next_level;
        }

        let root_index = current_level[0];

        Ok(Self {
            nodes,
            root_index,
            leaf_count,
            leaf_indices,
            chunk_size,
        })
    }

    /// Returns the BLAKE3 root hash of the tree.
    pub fn root_hash(&self) -> [u8; 32] {
        self.nodes[self.root_index].hash
    }

    /// Returns the number of leaf nodes (data chunks) in the tree.
    pub fn leaf_count(&self) -> usize {
        self.leaf_count
    }

    /// Returns the chunk size used to build the tree.
    pub fn chunk_size(&self) -> usize {
        self.chunk_size
    }

    /// Returns a reference to all nodes in the tree.
    pub fn nodes(&self) -> &[MerkleNode] {
        &self.nodes
    }

    /// Verify that the given `block_data` matches the hash stored for
    /// the leaf at `index`.
    ///
    /// # Panics
    ///
    /// Returns `false` if `index` is out of range.
    pub fn verify_block(&self, index: usize, block_data: &[u8]) -> bool {
        if index >= self.leaf_count {
            return false;
        }
        let leaf_idx = self.leaf_indices[index];
        let expected = self.nodes[leaf_idx].hash;
        let actual = *blake3::hash(block_data).as_bytes();
        expected == actual
    }

    /// Generate a Merkle inclusion proof for the block at `index`.
    ///
    /// The proof contains the sibling hashes along the path from the leaf
    /// to the root, enabling verification against the root hash alone.
    ///
    /// # Errors
    ///
    /// Returns [`StreamError::BlockOutOfRange`] if `index >= leaf_count`.
    pub fn proof(&self, index: usize) -> Result<MerkleProof> {
        if index >= self.leaf_count {
            return Err(StreamError::BlockOutOfRange {
                index,
                total: self.leaf_count,
            });
        }

        let leaf_node_idx = self.leaf_indices[index];
        let block_hash = self.nodes[leaf_node_idx].hash;

        // Walk from the leaf up to the root, collecting sibling hashes.
        // We need to find the path from this leaf to the root.
        let siblings = self.collect_siblings(index);

        Ok(MerkleProof {
            block_index: index,
            block_hash,
            siblings,
        })
    }

    /// Collect sibling hashes for a proof by replaying the tree-build algorithm.
    ///
    /// This reconstructs the path from the leaf at `leaf_position` to the root
    /// by simulating the bottom-up pairing process.
    fn collect_siblings(&self, leaf_position: usize) -> Vec<([u8; 32], bool)> {
        let mut siblings = Vec::new();

        // Reconstruct levels by replaying the build
        let mut current_level: Vec<usize> = self.leaf_indices.clone();
        let mut pos = leaf_position;

        while current_level.len() > 1 {
            let mut next_level = Vec::new();
            let mut next_pos = 0;
            let mut found_pos = false;

            let mut i = 0;
            while i < current_level.len() {
                if i + 1 < current_level.len() {
                    // Paired nodes
                    if i == pos {
                        // Our node is the left child; sibling is the right child
                        let sibling_idx = current_level[i + 1];
                        siblings.push((self.nodes[sibling_idx].hash, false));
                        if !found_pos {
                            next_pos = next_level.len();
                            found_pos = true;
                        }
                    } else if i + 1 == pos {
                        // Our node is the right child; sibling is the left child
                        let sibling_idx = current_level[i];
                        siblings.push((self.nodes[sibling_idx].hash, true));
                        if !found_pos {
                            next_pos = next_level.len();
                            found_pos = true;
                        }
                    }

                    // Find the parent node index
                    let left_idx = current_level[i];
                    let right_idx = current_level[i + 1];
                    // Find the parent in the nodes vec
                    let parent_idx = self
                        .nodes
                        .iter()
                        .position(|n| {
                            n.children.len() == 2
                                && n.children[0] == left_idx
                                && n.children[1] == right_idx
                        })
                        .unwrap();
                    next_level.push(parent_idx);
                    i += 2;
                } else {
                    // Odd node — promoted
                    if i == pos && !found_pos {
                        next_pos = next_level.len();
                        found_pos = true;
                    }
                    next_level.push(current_level[i]);
                    i += 1;
                }
            }

            current_level = next_level;
            pos = next_pos;
        }

        siblings
    }

    /// Verify a [`MerkleProof`] against a known root hash.
    ///
    /// Returns `true` if the proof is valid and the reconstructed root hash
    /// matches `root_hash`.
    pub fn verify_proof(proof: &MerkleProof, root_hash: &[u8; 32]) -> bool {
        let mut current_hash = proof.block_hash;

        for (sibling_hash, is_left) in &proof.siblings {
            if *is_left {
                current_hash = compute_parent_hash(sibling_hash, &current_hash);
            } else {
                current_hash = compute_parent_hash(&current_hash, sibling_hash);
            }
        }

        current_hash == *root_hash
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_from_known_data() {
        let data = vec![0xABu8; 1024];
        let tree = MerkleTree::build(&data, 512).unwrap();
        assert_eq!(tree.leaf_count(), 2);
        assert_eq!(tree.nodes().len(), 3); // 2 leaves + 1 root
    }

    #[test]
    fn test_root_hash_deterministic() {
        let data = b"hello world merkle tree";
        let tree1 = MerkleTree::build(data, 8).unwrap();
        let tree2 = MerkleTree::build(data, 8).unwrap();
        assert_eq!(tree1.root_hash(), tree2.root_hash());
    }

    #[test]
    fn test_different_data_different_root() {
        let data1 = b"data one";
        let data2 = b"data two";
        let tree1 = MerkleTree::build(data1, 4).unwrap();
        let tree2 = MerkleTree::build(data2, 4).unwrap();
        assert_ne!(tree1.root_hash(), tree2.root_hash());
    }

    #[test]
    fn test_verify_all_blocks() {
        let data = vec![0u8; 2048];
        let chunk_size = 512;
        let tree = MerkleTree::build(&data, chunk_size).unwrap();

        assert_eq!(tree.leaf_count(), 4);

        for i in 0..tree.leaf_count() {
            let start = i * chunk_size;
            let end = ((i + 1) * chunk_size).min(data.len());
            assert!(
                tree.verify_block(i, &data[start..end]),
                "block {i} failed verification"
            );
        }
    }

    #[test]
    fn test_verify_block_bad_data() {
        let data = vec![0u8; 1024];
        let tree = MerkleTree::build(&data, 512).unwrap();
        let bad_data = vec![1u8; 512];
        assert!(!tree.verify_block(0, &bad_data));
    }

    #[test]
    fn test_verify_block_out_of_range() {
        let data = vec![0u8; 1024];
        let tree = MerkleTree::build(&data, 512).unwrap();
        assert!(!tree.verify_block(99, &[]));
    }

    #[test]
    fn test_proof_generation_and_verification() {
        let data = vec![42u8; 4096];
        let chunk_size = 512;
        let tree = MerkleTree::build(&data, chunk_size).unwrap();
        let root = tree.root_hash();

        for i in 0..tree.leaf_count() {
            let proof = tree.proof(i).unwrap();
            assert_eq!(proof.block_index, i);
            assert!(
                MerkleTree::verify_proof(&proof, &root),
                "proof for block {i} failed verification"
            );
        }
    }

    #[test]
    fn test_proof_fails_with_wrong_root() {
        let data = vec![42u8; 2048];
        let tree = MerkleTree::build(&data, 512).unwrap();
        let proof = tree.proof(0).unwrap();
        let wrong_root = [0xFFu8; 32];
        assert!(!MerkleTree::verify_proof(&proof, &wrong_root));
    }

    #[test]
    fn test_proof_out_of_range() {
        let data = vec![0u8; 1024];
        let tree = MerkleTree::build(&data, 512).unwrap();
        let err = tree.proof(10).unwrap_err();
        assert!(matches!(
            err,
            StreamError::BlockOutOfRange {
                index: 10,
                total: 2
            }
        ));
    }

    #[test]
    fn test_empty_data_rejection() {
        let result = MerkleTree::build(&[], 512);
        assert!(matches!(result, Err(StreamError::EmptyData)));
    }

    #[test]
    fn test_zero_chunk_size_rejection() {
        let result = MerkleTree::build(&[1, 2, 3], 0);
        assert!(matches!(result, Err(StreamError::InvalidChunkSize(0))));
    }

    #[test]
    fn test_single_block_tree() {
        let data = b"small";
        let tree = MerkleTree::build(data, DEFAULT_CHUNK_SIZE).unwrap();
        assert_eq!(tree.leaf_count(), 1);
        // Root hash should equal the leaf hash
        let expected = *blake3::hash(data).as_bytes();
        assert_eq!(tree.root_hash(), expected);
    }

    #[test]
    fn test_single_block_proof() {
        let data = b"tiny block";
        let tree = MerkleTree::build(data, DEFAULT_CHUNK_SIZE).unwrap();
        let proof = tree.proof(0).unwrap();
        assert!(proof.siblings.is_empty());
        assert!(MerkleTree::verify_proof(&proof, &tree.root_hash()));
    }

    #[test]
    fn test_odd_number_of_chunks() {
        // 3 chunks: two paired + one promoted
        let data = vec![7u8; 300];
        let tree = MerkleTree::build(&data, 100).unwrap();
        assert_eq!(tree.leaf_count(), 3);

        // All proofs should still verify
        let root = tree.root_hash();
        for i in 0..3 {
            let proof = tree.proof(i).unwrap();
            assert!(
                MerkleTree::verify_proof(&proof, &root),
                "proof for block {i} failed with odd chunk count"
            );
        }
    }

    #[test]
    fn test_large_data() {
        let data = vec![0xCC; 1024 * 1024]; // 1 MiB
        let tree = MerkleTree::build(&data, DEFAULT_CHUNK_SIZE).unwrap();
        assert_eq!(tree.leaf_count(), 4); // 1 MiB / 256 KiB = 4
        let root = tree.root_hash();
        for i in 0..tree.leaf_count() {
            let proof = tree.proof(i).unwrap();
            assert!(MerkleTree::verify_proof(&proof, &root));
        }
    }

    #[test]
    fn test_compute_parent_hash() {
        let left = *blake3::hash(b"left").as_bytes();
        let right = *blake3::hash(b"right").as_bytes();
        let parent = compute_parent_hash(&left, &right);

        // Verify manually
        let mut hasher = blake3::Hasher::new();
        hasher.update(&left);
        hasher.update(&right);
        let expected = *hasher.finalize().as_bytes();
        assert_eq!(parent, expected);
    }
}
