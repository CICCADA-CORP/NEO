//! Peer-to-peer distribution architecture for the NEO format.
//!
//! This module provides the type definitions and stub implementations for
//! P2P content distribution. The architecture is designed for future
//! integration with [iroh](https://iroh.computer/) for BLAKE3-based
//! content-addressed blob transfer over QUIC.
//!
//! # Current Status
//!
//! This is an **architecture stub**. The types and API surface are defined
//! to allow upstream crates to build against the interface, but actual
//! network operations are not yet implemented. Methods that would perform
//! I/O log their intent via `tracing` and return appropriate stub values.
//!
//! # Future Integration
//!
//! When iroh is integrated, [`PeerNode`] will manage:
//! - Local content announcement to the DHT
//! - Content discovery and retrieval from peers
//! - Bandwidth tracking and peer management

use serde::{Deserialize, Serialize};
use std::time::Duration;

use crate::cid::ContentId;
use crate::error::{Result, StreamError};

/// Configuration for a P2P peer node.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PeerConfig {
    /// Address to listen on for incoming connections (e.g. `"0.0.0.0:4433"`).
    pub listen_addr: String,
    /// Bootstrap peer addresses to connect to on startup.
    pub bootstrap_peers: Vec<String>,
}

impl Default for PeerConfig {
    fn default() -> Self {
        Self {
            listen_addr: "0.0.0.0:4433".to_string(),
            bootstrap_peers: Vec::new(),
        }
    }
}

/// A P2P peer node for content distribution.
///
/// **Stub implementation** — see module-level docs for details.
///
/// When fully implemented, this will use iroh-blobs for BLAKE3
/// content-addressed blob transfer over QUIC.
#[derive(Debug)]
pub struct PeerNode {
    /// The configuration for this peer.
    config: PeerConfig,
    /// Accumulated transfer statistics.
    stats: TransferStats,
}

/// Statistics about P2P data transfers.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TransferStats {
    /// Total bytes downloaded from peers.
    pub bytes_downloaded: u64,
    /// Total bytes uploaded to peers.
    pub bytes_uploaded: u64,
    /// Number of currently connected peers.
    pub peers_connected: u32,
    /// Cumulative transfer duration.
    pub transfer_duration: Duration,
}

impl PeerNode {
    /// Create a new peer node with the given configuration.
    ///
    /// **Stub**: Does not actually bind to the network. Logs the
    /// configuration for debugging.
    pub fn new(config: PeerConfig) -> Self {
        tracing::info!(
            listen_addr = %config.listen_addr,
            bootstrap_count = config.bootstrap_peers.len(),
            "creating P2P peer node (stub)"
        );
        Self {
            config,
            stats: TransferStats::default(),
        }
    }

    /// Announce content availability to the network.
    ///
    /// **Stub**: Logs the CID being announced but does not perform
    /// any network operations.
    pub fn announce(&self, cid: &ContentId) {
        tracing::info!(
            cid = %cid,
            listen_addr = %self.config.listen_addr,
            "announcing content to P2P network (stub — no actual network operation)"
        );
    }

    /// Fetch content from the P2P network by its content identifier.
    ///
    /// **Stub**: Always returns an error indicating P2P is not yet connected.
    ///
    /// # Errors
    ///
    /// Currently always returns [`StreamError::Io`] with a message
    /// indicating that P2P networking is not yet implemented.
    pub fn fetch(&self, cid: &ContentId) -> Result<Vec<u8>> {
        tracing::warn!(
            cid = %cid,
            "P2P fetch requested but networking is not yet implemented"
        );
        Err(StreamError::Io(std::io::Error::new(
            std::io::ErrorKind::NotConnected,
            "P2P not yet connected — iroh integration pending",
        )))
    }

    /// Returns a reference to the current peer configuration.
    pub fn config(&self) -> &PeerConfig {
        &self.config
    }

    /// Returns a snapshot of the current transfer statistics.
    pub fn stats(&self) -> &TransferStats {
        &self.stats
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_peer_config_default() {
        let config = PeerConfig::default();
        assert_eq!(config.listen_addr, "0.0.0.0:4433");
        assert!(config.bootstrap_peers.is_empty());
    }

    #[test]
    fn test_peer_node_creation() {
        let config = PeerConfig {
            listen_addr: "127.0.0.1:5555".to_string(),
            bootstrap_peers: vec!["10.0.0.1:4433".to_string()],
        };
        let node = PeerNode::new(config);
        assert_eq!(node.config().listen_addr, "127.0.0.1:5555");
        assert_eq!(node.config().bootstrap_peers.len(), 1);
    }

    #[test]
    fn test_announce_does_not_panic() {
        let node = PeerNode::new(PeerConfig::default());
        let cid = crate::cid::compute_cid(b"test content");
        // Should not panic; just logs
        node.announce(&cid);
    }

    #[test]
    fn test_fetch_returns_error() {
        let node = PeerNode::new(PeerConfig::default());
        let cid = crate::cid::compute_cid(b"unreachable content");
        let result = node.fetch(&cid);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("P2P not yet connected"));
    }

    #[test]
    fn test_initial_stats_are_zero() {
        let node = PeerNode::new(PeerConfig::default());
        let stats = node.stats();
        assert_eq!(stats.bytes_downloaded, 0);
        assert_eq!(stats.bytes_uploaded, 0);
        assert_eq!(stats.peers_connected, 0);
        assert_eq!(stats.transfer_duration, Duration::ZERO);
    }

    #[test]
    fn test_transfer_stats_serde() {
        let stats = TransferStats {
            bytes_downloaded: 1024,
            bytes_uploaded: 512,
            peers_connected: 3,
            transfer_duration: Duration::from_secs(10),
        };
        let json = serde_json::to_string(&stats).unwrap();
        let deserialized: TransferStats = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.bytes_downloaded, 1024);
        assert_eq!(deserialized.bytes_uploaded, 512);
        assert_eq!(deserialized.peers_connected, 3);
        assert_eq!(deserialized.transfer_duration, Duration::from_secs(10));
    }
}
