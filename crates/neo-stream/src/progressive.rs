//! Progressive Level-of-Detail (LOD) loading.
//!
//! A NEO file supports three quality layers for progressive streaming,
//! as specified in SPECIFICATION.md Section 12:
//!
//! - **Layer 0 (Preview)**: Opus mono 32 kbps — instant playback, a few KB
//! - **Layer 1 (Standard)**: DAC-compressed stems — normal listening, a few MB
//! - **Layer 2 (Lossless)**: FLAC residual — bit-perfect reconstruction
//!
//! The [`ProgressiveManager`] holds data for each layer and provides
//! retrieval by quality level.
//!
//! # Examples
//!
//! ```
//! use neo_stream::progressive::{ProgressiveManager, QualityLayer};
//!
//! let mut manager = ProgressiveManager::new();
//! manager.add_layer(QualityLayer::Preview, vec![1, 2, 3]);
//! manager.add_layer(QualityLayer::Standard, vec![4, 5, 6, 7, 8]);
//!
//! assert_eq!(manager.best_available(), Some(QualityLayer::Standard));
//! assert_eq!(manager.total_size(), 8);
//! ```

use std::collections::BTreeMap;
use std::fmt;

use serde::{Deserialize, Serialize};

/// Quality layer for progressive loading.
///
/// Layers are ordered from lowest to highest quality. Each successive
/// layer provides better audio fidelity at the cost of more data.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum QualityLayer {
    /// Ultra-low quality preview (Opus 32 kbps mono).
    Preview = 0,
    /// Standard quality (DAC neural compressed stems).
    Standard = 1,
    /// Lossless quality (original + FLAC residual).
    Lossless = 2,
}

impl fmt::Display for QualityLayer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            QualityLayer::Preview => write!(f, "Preview (Layer 0)"),
            QualityLayer::Standard => write!(f, "Standard (Layer 1)"),
            QualityLayer::Lossless => write!(f, "Lossless (Layer 2)"),
        }
    }
}

/// Metadata about a single quality layer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerInfo {
    /// The quality layer this info describes.
    pub layer: QualityLayer,
    /// Size of the layer data in bytes.
    pub size_bytes: usize,
    /// Human-readable description of the codec used.
    pub codec_description: String,
    /// Whether this layer's data is currently available.
    pub available: bool,
}

/// Manages progressive Level-of-Detail layer data.
///
/// Stores the raw data for each quality layer and provides methods
/// to query availability, retrieve data, and inspect layer metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProgressiveManager {
    /// Layer data, keyed by quality layer. BTreeMap keeps them sorted.
    layers: BTreeMap<QualityLayer, Vec<u8>>,
}

impl ProgressiveManager {
    /// Create a new, empty progressive manager with no layers loaded.
    pub fn new() -> Self {
        Self {
            layers: BTreeMap::new(),
        }
    }

    /// Add (or replace) data for the given quality layer.
    pub fn add_layer(&mut self, layer: QualityLayer, data: Vec<u8>) {
        self.layers.insert(layer, data);
    }

    /// Retrieve the data for a given quality layer, if available.
    pub fn get_layer(&self, layer: QualityLayer) -> Option<&[u8]> {
        self.layers.get(&layer).map(|v| v.as_slice())
    }

    /// Returns a sorted list of quality layers that have data loaded.
    pub fn available_layers(&self) -> Vec<QualityLayer> {
        self.layers.keys().copied().collect()
    }

    /// Returns the highest quality layer currently available, or `None`
    /// if no layers are loaded.
    pub fn best_available(&self) -> Option<QualityLayer> {
        self.layers.keys().next_back().copied()
    }

    /// Returns the total size in bytes across all loaded layers.
    pub fn total_size(&self) -> usize {
        self.layers.values().map(|v| v.len()).sum()
    }

    /// Returns metadata for all three quality layers, indicating which
    /// are available and their sizes.
    pub fn layer_info(&self) -> Vec<LayerInfo> {
        let all_layers = [
            (QualityLayer::Preview, "Opus mono 32 kbps"),
            (QualityLayer::Standard, "DAC neural compressed"),
            (QualityLayer::Lossless, "FLAC residual lossless"),
        ];

        all_layers
            .iter()
            .map(|(layer, codec_desc)| {
                let data = self.layers.get(layer);
                LayerInfo {
                    layer: *layer,
                    size_bytes: data.map_or(0, |v| v.len()),
                    codec_description: codec_desc.to_string(),
                    available: data.is_some(),
                }
            })
            .collect()
    }
}

impl Default for ProgressiveManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_manager_is_empty() {
        let manager = ProgressiveManager::new();
        assert!(manager.available_layers().is_empty());
        assert_eq!(manager.best_available(), None);
        assert_eq!(manager.total_size(), 0);
    }

    #[test]
    fn test_add_and_get_layer() {
        let mut manager = ProgressiveManager::new();
        let preview_data = vec![1, 2, 3, 4, 5];
        manager.add_layer(QualityLayer::Preview, preview_data.clone());

        assert_eq!(
            manager.get_layer(QualityLayer::Preview),
            Some(preview_data.as_slice())
        );
        assert_eq!(manager.get_layer(QualityLayer::Standard), None);
        assert_eq!(manager.get_layer(QualityLayer::Lossless), None);
    }

    #[test]
    fn test_replace_layer() {
        let mut manager = ProgressiveManager::new();
        manager.add_layer(QualityLayer::Preview, vec![1, 2, 3]);
        manager.add_layer(QualityLayer::Preview, vec![10, 20]);

        assert_eq!(
            manager.get_layer(QualityLayer::Preview),
            Some([10u8, 20].as_slice())
        );
        assert_eq!(manager.total_size(), 2);
    }

    #[test]
    fn test_available_layers_ordering() {
        let mut manager = ProgressiveManager::new();
        // Add in non-sequential order
        manager.add_layer(QualityLayer::Lossless, vec![0; 100]);
        manager.add_layer(QualityLayer::Preview, vec![0; 10]);

        let available = manager.available_layers();
        assert_eq!(
            available,
            vec![QualityLayer::Preview, QualityLayer::Lossless]
        );
    }

    #[test]
    fn test_best_available() {
        let mut manager = ProgressiveManager::new();

        manager.add_layer(QualityLayer::Preview, vec![1]);
        assert_eq!(manager.best_available(), Some(QualityLayer::Preview));

        manager.add_layer(QualityLayer::Lossless, vec![2]);
        assert_eq!(manager.best_available(), Some(QualityLayer::Lossless));

        manager.add_layer(QualityLayer::Standard, vec![3]);
        assert_eq!(manager.best_available(), Some(QualityLayer::Lossless));
    }

    #[test]
    fn test_total_size() {
        let mut manager = ProgressiveManager::new();
        manager.add_layer(QualityLayer::Preview, vec![0; 100]);
        manager.add_layer(QualityLayer::Standard, vec![0; 5000]);
        manager.add_layer(QualityLayer::Lossless, vec![0; 50000]);

        assert_eq!(manager.total_size(), 55100);
    }

    #[test]
    fn test_layer_info() {
        let mut manager = ProgressiveManager::new();
        manager.add_layer(QualityLayer::Preview, vec![0; 256]);
        manager.add_layer(QualityLayer::Standard, vec![0; 8192]);

        let info = manager.layer_info();
        assert_eq!(info.len(), 3);

        assert_eq!(info[0].layer, QualityLayer::Preview);
        assert!(info[0].available);
        assert_eq!(info[0].size_bytes, 256);

        assert_eq!(info[1].layer, QualityLayer::Standard);
        assert!(info[1].available);
        assert_eq!(info[1].size_bytes, 8192);

        assert_eq!(info[2].layer, QualityLayer::Lossless);
        assert!(!info[2].available);
        assert_eq!(info[2].size_bytes, 0);
    }

    #[test]
    fn test_quality_layer_ordering() {
        assert!(QualityLayer::Preview < QualityLayer::Standard);
        assert!(QualityLayer::Standard < QualityLayer::Lossless);
    }

    #[test]
    fn test_quality_layer_display() {
        assert_eq!(format!("{}", QualityLayer::Preview), "Preview (Layer 0)");
        assert_eq!(format!("{}", QualityLayer::Standard), "Standard (Layer 1)");
        assert_eq!(format!("{}", QualityLayer::Lossless), "Lossless (Layer 2)");
    }

    #[test]
    fn test_default_trait() {
        let manager = ProgressiveManager::default();
        assert!(manager.available_layers().is_empty());
    }
}
