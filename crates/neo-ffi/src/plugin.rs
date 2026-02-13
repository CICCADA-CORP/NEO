//! VST3/CLAP audio plugin architecture for the NEO format.
//!
//! Defines the plugin interfaces for integrating NEO file playback
//! and stem manipulation into DAWs (Digital Audio Workstations).
//!
//! # Architecture
//!
//! The NEO plugin will provide:
//! - **NEO Player**: Load and play `.neo` files with per-stem controls
//! - **NEO Encoder**: Export DAW stems to `.neo` format
//! - **NEO Spatial**: 3D spatial audio panner using NEO's object model
//!
//! # Plugin Formats
//!
//! - **VST3**: Via `vst3-sys` crate (Windows, macOS, Linux)
//! - **CLAP**: Via `clap-sys` crate (emerging open standard)
//! - **AU**: Audio Units for macOS (via Core Audio)
//!
//! # Future Dependencies
//!
//! - `vst3-sys` for VST3 plugin interface
//! - `clap-sys` or `clack` for CLAP plugin interface
//! - `baseview` for cross-platform plugin GUI

/// Supported audio plugin formats.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PluginFormat {
    /// Steinberg VST3 format.
    Vst3,
    /// CLAP (CLever Audio Plugin) format.
    Clap,
    /// Apple Audio Units (macOS only).
    AudioUnit,
}

/// NEO plugin type variants.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NeoPluginType {
    /// Player — loads .neo files and provides per-stem playback controls.
    Player,
    /// Encoder — captures DAW audio and exports to .neo format.
    Encoder,
    /// Spatial — 3D audio panner using NEO's spatial object model.
    SpatialPanner,
}

/// Plugin descriptor with metadata for DAW registration.
#[derive(Debug, Clone)]
pub struct PluginDescriptor {
    /// Unique plugin identifier (reverse domain notation).
    pub id: String,
    /// Human-readable plugin name.
    pub name: String,
    /// Plugin vendor.
    pub vendor: String,
    /// Plugin version string.
    pub version: String,
    /// Plugin format.
    pub format: PluginFormat,
    /// Plugin type.
    pub plugin_type: NeoPluginType,
    /// Number of audio input channels.
    pub input_channels: u32,
    /// Number of audio output channels.
    pub output_channels: u32,
}

/// Default plugin descriptors for the NEO plugin suite.
pub fn neo_plugin_descriptors() -> Vec<PluginDescriptor> {
    vec![
        PluginDescriptor {
            id: "com.neo-audio.player".into(),
            name: "NEO Player".into(),
            vendor: "NEO Audio Project".into(),
            version: "0.1.0".into(),
            format: PluginFormat::Vst3,
            plugin_type: NeoPluginType::Player,
            input_channels: 0,
            output_channels: 2,
        },
        PluginDescriptor {
            id: "com.neo-audio.encoder".into(),
            name: "NEO Encoder".into(),
            vendor: "NEO Audio Project".into(),
            version: "0.1.0".into(),
            format: PluginFormat::Vst3,
            plugin_type: NeoPluginType::Encoder,
            input_channels: 16,
            output_channels: 0,
        },
        PluginDescriptor {
            id: "com.neo-audio.spatial".into(),
            name: "NEO Spatial Panner".into(),
            vendor: "NEO Audio Project".into(),
            version: "0.1.0".into(),
            format: PluginFormat::Vst3,
            plugin_type: NeoPluginType::SpatialPanner,
            input_channels: 2,
            output_channels: 2,
        },
    ]
}

/// Plugin parameter definition for DAW automation.
#[derive(Debug, Clone)]
pub struct PluginParameter {
    /// Parameter ID (unique within the plugin).
    pub id: u32,
    /// Human-readable parameter name.
    pub name: String,
    /// Minimum value.
    pub min: f64,
    /// Maximum value.
    pub max: f64,
    /// Default value.
    pub default: f64,
    /// Parameter unit label (e.g., "dB", "%", "Hz").
    pub unit: String,
}

/// Standard parameters for the NEO Player plugin.
pub fn player_parameters() -> Vec<PluginParameter> {
    let mut params = Vec::new();
    // Master volume
    params.push(PluginParameter {
        id: 0,
        name: "Master Volume".into(),
        min: -96.0,
        max: 12.0,
        default: 0.0,
        unit: "dB".into(),
    });
    // Per-stem volume (up to 8 stems)
    for i in 0..8u32 {
        params.push(PluginParameter {
            id: 100 + i,
            name: format!("Stem {} Volume", i),
            min: -96.0,
            max: 12.0,
            default: 0.0,
            unit: "dB".into(),
        });
        params.push(PluginParameter {
            id: 200 + i,
            name: format!("Stem {} Mute", i),
            min: 0.0,
            max: 1.0,
            default: 0.0,
            unit: "".into(),
        });
        params.push(PluginParameter {
            id: 300 + i,
            name: format!("Stem {} Solo", i),
            min: 0.0,
            max: 1.0,
            default: 0.0,
            unit: "".into(),
        });
        params.push(PluginParameter {
            id: 400 + i,
            name: format!("Stem {} Pan", i),
            min: -1.0,
            max: 1.0,
            default: 0.0,
            unit: "".into(),
        });
    }
    params
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wasm::WasmConfig;
    use crate::wasm::WasmTarget;

    #[test]
    fn test_plugin_descriptors() {
        let descs = neo_plugin_descriptors();
        assert_eq!(descs.len(), 3);
        assert_eq!(descs[0].plugin_type, NeoPluginType::Player);
        assert_eq!(descs[1].plugin_type, NeoPluginType::Encoder);
        assert_eq!(descs[2].plugin_type, NeoPluginType::SpatialPanner);
    }

    #[test]
    fn test_player_parameters() {
        let params = player_parameters();
        // 1 master + 8 stems * 4 params = 33
        assert_eq!(params.len(), 33);
        assert_eq!(params[0].name, "Master Volume");
        assert_eq!(params[0].unit, "dB");
    }

    #[test]
    fn test_wasm_config_default() {
        let config = WasmConfig::default();
        assert_eq!(config.target, WasmTarget::Browser);
        assert!(!config.enable_simd);
        assert!(!config.enable_threads);
        assert_eq!(config.max_memory_pages, 256);
    }
}
