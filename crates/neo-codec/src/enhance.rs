//! AI-powered audio enhancement pipeline.
//!
//! Provides bandwidth extension, stereo widening, and source separation
//! interfaces. Currently implements basic DSP versions; neural model
//! backends are planned for a future iteration.
//!
//! See SPECIFICATION.md Section 8 for codec extensions.

use crate::error::CodecError;

/// Audio enhancement operation types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EnhancementType {
    /// Extend bandwidth beyond the codec's native range (e.g., 16kHz → 44.1kHz).
    BandwidthExtension,
    /// Widen stereo image for mono or narrow-stereo sources.
    StereoWidening,
    /// Denoise — remove background noise from audio.
    Denoise,
    /// Declip — restore clipped audio samples.
    Declip,
}

/// Configuration for an audio enhancement pass.
#[derive(Debug, Clone)]
pub struct EnhanceConfig {
    /// Sample rate of the input audio.
    pub sample_rate: u32,
    /// Number of channels (1 = mono, 2 = stereo).
    pub channels: u8,
    /// Enhancement intensity (0.0 = off, 1.0 = full).
    pub intensity: f32,
}

impl Default for EnhanceConfig {
    fn default() -> Self {
        Self {
            sample_rate: 44100,
            channels: 2,
            intensity: 0.5,
        }
    }
}

/// Trait for audio enhancement processors.
pub trait AudioEnhancer: Send + Sync {
    /// Apply enhancement to PCM f32 samples.
    fn enhance(&self, samples: &[f32], config: &EnhanceConfig) -> Result<Vec<f32>, CodecError>;

    /// Return the enhancement type.
    fn enhancement_type(&self) -> EnhancementType;

    /// Human-readable name.
    fn name(&self) -> &str;
}

/// Bandwidth extension — upsamples low-bandwidth audio using spectral replication.
///
/// The DSP fallback uses simple spectral band replication. A neural model
/// version (using ONNX) is planned for a future phase.
pub struct BandwidthExtender {
    /// Target output sample rate.
    pub target_rate: u32,
}

impl BandwidthExtender {
    /// Create a new bandwidth extender targeting the given sample rate.
    pub fn new(target_rate: u32) -> Self {
        Self { target_rate }
    }
}

impl AudioEnhancer for BandwidthExtender {
    fn enhance(&self, samples: &[f32], config: &EnhanceConfig) -> Result<Vec<f32>, CodecError> {
        if config.sample_rate >= self.target_rate {
            // No upsampling needed — pass through.
            return Ok(samples.to_vec());
        }

        // Basic DSP fallback: linear interpolation upsampling.
        // A real implementation would use spectral band replication or a neural model.
        let ratio = self.target_rate as f64 / config.sample_rate as f64;
        let new_len = (samples.len() as f64 * ratio) as usize;
        let mut output = Vec::with_capacity(new_len);

        for i in 0..new_len {
            let src_pos = i as f64 / ratio;
            let idx = src_pos as usize;
            let frac = (src_pos - idx as f64) as f32;

            if idx + 1 < samples.len() {
                let interpolated = samples[idx] * (1.0 - frac) + samples[idx + 1] * frac;
                output.push(interpolated);
            } else if idx < samples.len() {
                output.push(samples[idx]);
            }
        }

        tracing::debug!(
            input_rate = config.sample_rate,
            target_rate = self.target_rate,
            input_len = samples.len(),
            output_len = output.len(),
            "Bandwidth extension applied (DSP fallback)"
        );

        Ok(output)
    }

    fn enhancement_type(&self) -> EnhancementType {
        EnhancementType::BandwidthExtension
    }

    fn name(&self) -> &str {
        "Bandwidth Extension (DSP)"
    }
}

/// Stereo widening — expands the stereo image of audio.
///
/// Uses mid/side processing to increase the perceived stereo width.
/// Input must be stereo (2 channels, interleaved).
pub struct StereoWidener;

impl StereoWidener {
    /// Create a new stereo widener.
    pub fn new() -> Self {
        Self
    }

    /// Widen stereo audio using mid/side processing.
    ///
    /// `samples` must be interleaved stereo (L, R, L, R, ...).
    /// `width` ranges from 0.0 (mono) to 1.0 (normal) to 2.0 (extra wide).
    pub fn widen(samples: &[f32], width: f32) -> Result<Vec<f32>, CodecError> {
        #[allow(clippy::manual_is_multiple_of)]
        if samples.len() % 2 != 0 {
            return Err(CodecError::DecodingError(
                "Stereo widening requires even number of samples (interleaved stereo)".into(),
            ));
        }

        let mut output = Vec::with_capacity(samples.len());

        for frame in samples.chunks_exact(2) {
            let left = frame[0];
            let right = frame[1];

            // Mid/Side decomposition.
            let mid = (left + right) * 0.5;
            let side = (left - right) * 0.5;

            // Scale the side signal by the width factor.
            let widened_side = side * width;

            // Reconstruct left/right.
            let new_left = (mid + widened_side).clamp(-1.0, 1.0);
            let new_right = (mid - widened_side).clamp(-1.0, 1.0);

            output.push(new_left);
            output.push(new_right);
        }

        tracing::debug!(
            width = width,
            frames = samples.len() / 2,
            "Stereo widening applied"
        );

        Ok(output)
    }
}

impl Default for StereoWidener {
    fn default() -> Self {
        Self::new()
    }
}

impl AudioEnhancer for StereoWidener {
    fn enhance(&self, samples: &[f32], config: &EnhanceConfig) -> Result<Vec<f32>, CodecError> {
        if config.channels != 2 {
            return Err(CodecError::DecodingError(
                "Stereo widening requires 2-channel audio".into(),
            ));
        }
        Self::widen(samples, config.intensity + 1.0)
    }

    fn enhancement_type(&self) -> EnhancementType {
        EnhancementType::StereoWidening
    }

    fn name(&self) -> &str {
        "Stereo Widener (Mid/Side)"
    }
}

/// Source separator — splits mixed audio into individual stems.
///
/// This is a stub interface. The actual implementation calls the Python
/// Demucs model via the `neo_neural` package.
pub struct SourceSeparator {
    /// Path to the Demucs ONNX model file.
    pub model_path: Option<String>,
}

impl SourceSeparator {
    /// Create a new source separator.
    ///
    /// If `model_path` is `None`, the separator will return an error
    /// indicating that the model must be downloaded first.
    pub fn new(model_path: Option<String>) -> Self {
        Self { model_path }
    }

    /// Separate a mixed audio signal into stems.
    ///
    /// Returns a vector of (label, samples) tuples.
    /// Currently returns an error — actual separation requires the Demucs ONNX model.
    pub fn separate(
        &self,
        _samples: &[f32],
        _sample_rate: u32,
        _channels: u8,
    ) -> Result<Vec<(String, Vec<f32>)>, CodecError> {
        match &self.model_path {
            Some(path) => {
                tracing::warn!(
                    model_path = path.as_str(),
                    "Source separation via ONNX not yet implemented — use Python neo_neural.separator"
                );
                Err(CodecError::EncodingError(
                    "ONNX-based source separation not yet implemented. \
                     Use the Python `neo_neural.separator` module for Demucs separation."
                        .into(),
                ))
            }
            None => Err(CodecError::EncodingError(
                "No Demucs model path provided. Download with `neo model download demucs` first."
                    .into(),
            )),
        }
    }
}

/// Create the default enhancement pipeline for post-decode processing.
///
/// Returns a list of enhancers that can be applied sequentially to decoded audio.
pub fn default_enhancement_pipeline() -> Vec<Box<dyn AudioEnhancer>> {
    vec![
        Box::new(BandwidthExtender::new(44100)),
        Box::new(StereoWidener::new()),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bandwidth_extension_passthrough() {
        let extender = BandwidthExtender::new(44100);
        let config = EnhanceConfig {
            sample_rate: 44100,
            channels: 1,
            intensity: 1.0,
        };
        let samples = vec![0.0, 0.5, 1.0, -1.0];
        let result = extender.enhance(&samples, &config).unwrap();
        assert_eq!(result, samples); // No upsampling needed
    }

    #[test]
    fn test_bandwidth_extension_upsample() {
        let extender = BandwidthExtender::new(44100);
        let config = EnhanceConfig {
            sample_rate: 22050,
            channels: 1,
            intensity: 1.0,
        };
        let samples = vec![0.0, 1.0, 0.0, -1.0];
        let result = extender.enhance(&samples, &config).unwrap();
        // Should roughly double the number of samples
        assert!(result.len() >= samples.len());
        assert!(result.len() <= samples.len() * 2 + 1);
    }

    #[test]
    fn test_stereo_widen_identity() {
        // Width 1.0 = no change
        let samples = vec![0.5, -0.5, 0.3, -0.3];
        let result = StereoWidener::widen(&samples, 1.0).unwrap();
        for (a, b) in result.iter().zip(samples.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn test_stereo_widen_mono() {
        // Width 0.0 = collapse to mono
        let samples = vec![0.8, -0.2, 0.6, 0.4];
        let result = StereoWidener::widen(&samples, 0.0).unwrap();
        // Left and right should be equal (mid only)
        for frame in result.chunks_exact(2) {
            assert!((frame[0] - frame[1]).abs() < 1e-6);
        }
    }

    #[test]
    fn test_stereo_widen_wide() {
        // Width 2.0 = extra wide
        let samples = vec![0.5, -0.5, 0.3, 0.1];
        let result = StereoWidener::widen(&samples, 2.0).unwrap();
        assert_eq!(result.len(), samples.len());
        // Wider should increase L-R difference
        let orig_diff = (samples[0] - samples[1]).abs();
        let wide_diff = (result[0] - result[1]).abs();
        assert!(wide_diff >= orig_diff);
    }

    #[test]
    fn test_stereo_widen_odd_length_rejected() {
        let samples = vec![0.5, -0.5, 0.3];
        let result = StereoWidener::widen(&samples, 1.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_source_separator_no_model() {
        let sep = SourceSeparator::new(None);
        let result = sep.separate(&[0.0; 100], 44100, 2);
        assert!(result.is_err());
    }

    #[test]
    fn test_source_separator_stub_error() {
        let sep = SourceSeparator::new(Some("/path/to/model.onnx".into()));
        let result = sep.separate(&[0.0; 100], 44100, 2);
        assert!(result.is_err()); // Not yet implemented
    }

    #[test]
    fn test_default_pipeline() {
        let pipeline = default_enhancement_pipeline();
        assert_eq!(pipeline.len(), 2);
        assert_eq!(
            pipeline[0].enhancement_type(),
            EnhancementType::BandwidthExtension
        );
        assert_eq!(
            pipeline[1].enhancement_type(),
            EnhancementType::StereoWidening
        );
    }

    #[test]
    fn test_enhance_config_default() {
        let config = EnhanceConfig::default();
        assert_eq!(config.sample_rate, 44100);
        assert_eq!(config.channels, 2);
        assert!((config.intensity - 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn test_bandwidth_clamping() {
        let extender = BandwidthExtender::new(48000);
        let config = EnhanceConfig {
            sample_rate: 16000,
            channels: 1,
            intensity: 1.0,
        };
        let samples = vec![0.0; 100];
        let result = extender.enhance(&samples, &config).unwrap();
        // Output should be approximately 3x the input length
        assert!(result.len() > samples.len() * 2);
    }
}
