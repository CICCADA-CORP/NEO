//! Binaural rendering — converts spatial audio to headphone output using HRTF.
//!
//! Uses Head-Related Transfer Functions to simulate 3D audio positioning
//! for standard stereo headphones. Implements a simplified ITD/ILD model
//! for real-time rendering without requiring large HRTF datasets.
//!
//! ## Rendering Model
//!
//! The binaural renderer applies:
//! - **ITD (Interaural Time Delay)**: Frequency-independent delay based on azimuth
//! - **ILD (Interaural Level Difference)**: Amplitude panning based on azimuth
//! - **Distance attenuation**: Inverse-distance gain rolloff
//! - **Elevation cues**: Subtle frequency-dependent coloration

use serde::{Deserialize, Serialize};

use crate::object::Position3D;

/// HRTF dataset selection for binaural rendering.
///
/// Determines which Head-Related Transfer Function is used to
/// spatialize audio for headphone playback.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub enum HrtfDataset {
    /// MIT KEMAR dataset (public domain, generic head model).
    /// Uses a built-in simplified ITD/ILD model.
    #[default]
    MitKemar,
    /// SADIE II dataset (Creative Commons).
    /// Uses a built-in simplified ITD/ILD model.
    SadieII,
    /// Custom HRTF from file path.
    Custom(std::path::PathBuf),
}

/// Speed of sound in air (meters per second) at 20°C.
const SPEED_OF_SOUND: f64 = 343.0;

/// Average human head radius in meters (Woodworth model).
const HEAD_RADIUS: f64 = 0.0875;

/// Binaural renderer that converts mono audio to spatialized stereo.
///
/// Uses a simplified HRTF model based on ITD (Interaural Time Delay) and
/// ILD (Interaural Level Difference) to render audio for headphone playback.
///
/// # Example
///
/// ```
/// use neo_spatial::binaural::{BinauralRenderer, HrtfDataset};
/// use neo_spatial::object::Position3D;
///
/// let renderer = BinauralRenderer::new(HrtfDataset::MitKemar);
/// let mono_samples = vec![0.5f32; 1000];
/// let position = Position3D::new(-30.0, 0.0, 0.8).unwrap();
/// let (left, right) = renderer.render(&mono_samples, &position, 44100);
/// assert_eq!(left.len(), 1000);
/// assert_eq!(right.len(), 1000);
/// ```
#[derive(Debug, Clone)]
pub struct BinauralRenderer {
    /// The HRTF dataset used for rendering.
    dataset: HrtfDataset,
}

impl BinauralRenderer {
    /// Creates a new binaural renderer using the specified HRTF dataset.
    ///
    /// For [`HrtfDataset::MitKemar`] and [`HrtfDataset::SadieII`], a built-in
    /// simplified ITD/ILD model is used. Custom datasets are reserved for
    /// future implementation with full HRTF convolution.
    pub fn new(dataset: HrtfDataset) -> Self {
        Self { dataset }
    }

    /// Returns a reference to the HRTF dataset used by this renderer.
    pub fn dataset(&self) -> &HrtfDataset {
        &self.dataset
    }

    /// Renders mono audio samples to spatialized stereo using HRTF.
    ///
    /// Applies ITD, ILD, and distance attenuation to create a binaural
    /// stereo signal from a mono source at the given position.
    ///
    /// # Arguments
    ///
    /// * `mono_samples` - Input mono audio samples
    /// * `position` - Source position in 3D space
    /// * `sample_rate` - Audio sample rate in Hz (e.g., 44100, 48000)
    ///
    /// # Returns
    ///
    /// A tuple of `(left, right)` channel sample vectors. Both channels have
    /// the same length as the input.
    pub fn render(
        &self,
        mono_samples: &[f32],
        position: &Position3D,
        sample_rate: u32,
    ) -> (Vec<f32>, Vec<f32>) {
        if mono_samples.is_empty() {
            return (Vec::new(), Vec::new());
        }

        let num_samples = mono_samples.len();
        let mut left = vec![0.0f32; num_samples];
        let mut right = vec![0.0f32; num_samples];

        // Compute ITD and ILD from position
        let (itd_samples_left, itd_samples_right, ild_left, ild_right) =
            self.compute_hrtf_params(position, sample_rate);

        // Compute distance attenuation (inverse distance law)
        let distance_gain = compute_distance_attenuation(position.distance);

        // Apply elevation-based spectral cue (simple high-frequency boost/cut)
        let elevation_factor = compute_elevation_factor(position.elevation);

        // Apply ITD and ILD
        for i in 0..num_samples {
            let sample = mono_samples[i] as f64;

            // Left channel: apply delay and level
            let left_idx = i as i64 - itd_samples_left as i64;
            if left_idx >= 0 && (left_idx as usize) < num_samples {
                left[i] = (sample * ild_left * distance_gain * elevation_factor) as f32;
            }

            // Right channel: apply delay and level
            let right_idx = i as i64 - itd_samples_right as i64;
            if right_idx >= 0 && (right_idx as usize) < num_samples {
                right[i] = (sample * ild_right * distance_gain * elevation_factor) as f32;
            }
        }

        // Apply ITD by shifting samples
        let left_delayed = apply_fractional_delay(&left, itd_samples_left);
        let right_delayed = apply_fractional_delay(&right, itd_samples_right);

        (left_delayed, right_delayed)
    }

    /// Computes HRTF parameters (ITD and ILD) for a given position.
    ///
    /// Returns `(itd_left, itd_right, ild_left, ild_right)` where:
    /// - `itd_left/right`: delay in samples for each ear
    /// - `ild_left/right`: gain factor for each ear
    fn compute_hrtf_params(&self, position: &Position3D, sample_rate: u32) -> (f64, f64, f64, f64) {
        let az_rad = position.azimuth.to_radians();

        // Woodworth ITD model: ITD = (r/c) * (sin(θ) + θ)
        // Simplified: ITD ≈ (r/c) * sin(θ) for small head model
        let itd_seconds = (HEAD_RADIUS / SPEED_OF_SOUND) * az_rad.sin();

        // Convert ITD to fractional samples
        let itd_samples = itd_seconds * sample_rate as f64;

        // Positive azimuth = right side → right ear gets sound first (less delay)
        // itd_samples > 0 means source is on the right → delay left ear
        let (itd_left, itd_right) = if itd_samples >= 0.0 {
            (itd_samples.abs(), 0.0)
        } else {
            (0.0, itd_samples.abs())
        };

        // ILD model: based on a simplified head-shadow effect
        // The ear facing the source gets more energy, the opposite ear gets less
        // Using a cosine-based panning model with head shadow attenuation
        let shadow_factor = 0.4; // Controls the strength of the head shadow

        // For azimuth > 0 (right), right ear is closer → louder right, quieter left
        let ild_left = 1.0 - shadow_factor * az_rad.sin().max(0.0);
        let ild_right = 1.0 + shadow_factor * az_rad.sin().max(0.0);

        // Normalize so the louder ear doesn't exceed 1.0 (but is close to it)
        let max_ild = ild_left.max(ild_right);
        let ild_left = ild_left / max_ild;
        let ild_right = ild_right / max_ild;

        (itd_left, itd_right, ild_left, ild_right)
    }
}

/// Computes distance-based gain attenuation.
///
/// Uses inverse distance law with a minimum distance clamp to prevent
/// infinite amplification at distance = 0.
fn compute_distance_attenuation(distance: f64) -> f64 {
    // Map distance 0..1 to gain: closer = louder
    // At distance 0.0 → gain 1.0
    // At distance 1.0 → gain ~0.5 (6dB attenuation at max distance)
    let clamped = distance.clamp(0.0, 1.0);
    1.0 / (1.0 + clamped)
}

/// Computes an elevation-dependent gain factor.
///
/// Simulates the frequency-dependent coloration caused by the pinna
/// at different elevations. This is a very simplified model.
fn compute_elevation_factor(elevation: f64) -> f64 {
    // Subtle elevation effect: slight attenuation for extreme elevations
    let el_rad = elevation.to_radians();
    let factor = 1.0 - 0.1 * el_rad.abs() / std::f64::consts::FRAC_PI_2;
    factor.clamp(0.8, 1.0)
}

/// Applies a fractional sample delay using linear interpolation.
///
/// For sub-sample accuracy, interpolates between adjacent samples.
fn apply_fractional_delay(samples: &[f32], delay_samples: f64) -> Vec<f32> {
    if samples.is_empty() {
        return Vec::new();
    }

    let num_samples = samples.len();
    let mut output = vec![0.0f32; num_samples];

    let delay_int = delay_samples.floor() as usize;
    let delay_frac = delay_samples - delay_samples.floor();

    for i in 0..num_samples {
        if i > delay_int {
            let idx = i - delay_int;
            let prev_idx = idx.saturating_sub(1);
            // Linear interpolation between two samples
            let s0 = samples[prev_idx] as f64;
            let s1 = samples[idx] as f64;
            output[i] = (s0 * delay_frac + s1 * (1.0 - delay_frac)) as f32;
        } else if i >= delay_int {
            output[i] = samples[i - delay_int];
        }
        // else: output[i] remains 0 (silence before delayed signal)
    }

    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_binaural_renderer_creation() {
        let renderer = BinauralRenderer::new(HrtfDataset::MitKemar);
        assert_eq!(*renderer.dataset(), HrtfDataset::MitKemar);
    }

    #[test]
    fn test_render_empty_input() {
        let renderer = BinauralRenderer::new(HrtfDataset::MitKemar);
        let pos = Position3D::front_center();
        let (left, right) = renderer.render(&[], &pos, 44100);
        assert!(left.is_empty());
        assert!(right.is_empty());
    }

    #[test]
    fn test_render_output_length() {
        let renderer = BinauralRenderer::new(HrtfDataset::MitKemar);
        let samples = vec![1.0f32; 1000];
        let pos = Position3D::front_center();
        let (left, right) = renderer.render(&samples, &pos, 44100);
        assert_eq!(left.len(), 1000);
        assert_eq!(right.len(), 1000);
    }

    #[test]
    fn test_render_center_source_similar_levels() {
        // A front-center source should produce approximately equal levels
        let renderer = BinauralRenderer::new(HrtfDataset::MitKemar);
        let samples: Vec<f32> = (0..1000).map(|i| (i as f32 * 0.1).sin()).collect();
        let pos = Position3D::new(0.0, 0.0, 0.5).unwrap();
        let (left, right) = renderer.render(&samples, &pos, 44100);

        let left_energy: f64 = left.iter().map(|s| (*s as f64) * (*s as f64)).sum();
        let right_energy: f64 = right.iter().map(|s| (*s as f64) * (*s as f64)).sum();

        // Energies should be approximately equal (within 10%)
        let ratio = if left_energy > right_energy {
            left_energy / right_energy
        } else {
            right_energy / left_energy
        };
        assert!(
            ratio < 1.1,
            "Center source should have similar L/R energy: L={}, R={}, ratio={}",
            left_energy,
            right_energy,
            ratio
        );
    }

    #[test]
    fn test_render_left_source_louder_left() {
        // A left source should be louder in the left ear
        let renderer = BinauralRenderer::new(HrtfDataset::MitKemar);
        let samples: Vec<f32> = (0..1000).map(|i| (i as f32 * 0.1).sin()).collect();
        let pos = Position3D::new(-90.0, 0.0, 0.5).unwrap();
        let (left, right) = renderer.render(&samples, &pos, 44100);

        let left_energy: f64 = left.iter().map(|s| (*s as f64) * (*s as f64)).sum();
        let right_energy: f64 = right.iter().map(|s| (*s as f64) * (*s as f64)).sum();

        assert!(
            left_energy > right_energy,
            "Left source should be louder in left ear: L={}, R={}",
            left_energy,
            right_energy
        );
    }

    #[test]
    fn test_render_right_source_louder_right() {
        // A right source should be louder in the right ear
        let renderer = BinauralRenderer::new(HrtfDataset::MitKemar);
        let samples: Vec<f32> = (0..1000).map(|i| (i as f32 * 0.1).sin()).collect();
        let pos = Position3D::new(90.0, 0.0, 0.5).unwrap();
        let (left, right) = renderer.render(&samples, &pos, 44100);

        let left_energy: f64 = left.iter().map(|s| (*s as f64) * (*s as f64)).sum();
        let right_energy: f64 = right.iter().map(|s| (*s as f64) * (*s as f64)).sum();

        assert!(
            right_energy > left_energy,
            "Right source should be louder in right ear: L={}, R={}",
            left_energy,
            right_energy
        );
    }

    #[test]
    fn test_render_distance_attenuation() {
        // A more distant source should be quieter
        let renderer = BinauralRenderer::new(HrtfDataset::MitKemar);
        let samples: Vec<f32> = (0..1000).map(|i| (i as f32 * 0.1).sin()).collect();

        let pos_near = Position3D::new(0.0, 0.0, 0.1).unwrap();
        let (left_near, right_near) = renderer.render(&samples, &pos_near, 44100);

        let pos_far = Position3D::new(0.0, 0.0, 1.0).unwrap();
        let (left_far, right_far) = renderer.render(&samples, &pos_far, 44100);

        let near_energy: f64 = left_near
            .iter()
            .zip(right_near.iter())
            .map(|(l, r)| (*l as f64).powi(2) + (*r as f64).powi(2))
            .sum();
        let far_energy: f64 = left_far
            .iter()
            .zip(right_far.iter())
            .map(|(l, r)| (*l as f64).powi(2) + (*r as f64).powi(2))
            .sum();

        assert!(
            near_energy > far_energy,
            "Near source should be louder: near={}, far={}",
            near_energy,
            far_energy
        );
    }

    #[test]
    fn test_distance_attenuation_values() {
        assert!((compute_distance_attenuation(0.0) - 1.0).abs() < f64::EPSILON);
        assert!((compute_distance_attenuation(1.0) - 0.5).abs() < f64::EPSILON);
        assert!(compute_distance_attenuation(0.5) > 0.5);
        assert!(compute_distance_attenuation(0.5) < 1.0);
    }

    #[test]
    fn test_elevation_factor() {
        // At ear level, factor should be 1.0
        let factor = compute_elevation_factor(0.0);
        assert!((factor - 1.0).abs() < f64::EPSILON);

        // At extreme elevation, factor should be less but >= 0.8
        let factor_up = compute_elevation_factor(90.0);
        assert!(factor_up >= 0.8);
        assert!(factor_up < 1.0);
    }

    #[test]
    fn test_fractional_delay_zero() {
        let samples = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let output = apply_fractional_delay(&samples, 0.0);
        assert_eq!(output.len(), 5);
        // With zero delay, output should closely match input
        for (a, b) in output.iter().zip(samples.iter()) {
            assert!((a - b).abs() < 1e-5, "Expected {}, got {}", b, a);
        }
    }

    #[test]
    fn test_fractional_delay_integer() {
        let samples = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let output = apply_fractional_delay(&samples, 1.0);
        assert_eq!(output.len(), 5);
        // First sample should be zero (delayed)
        assert!(output[0].abs() < 1e-5);
    }

    #[test]
    fn test_fractional_delay_empty() {
        let output = apply_fractional_delay(&[], 1.0);
        assert!(output.is_empty());
    }

    #[test]
    fn test_hrtf_dataset_default() {
        let dataset = HrtfDataset::default();
        assert_eq!(dataset, HrtfDataset::MitKemar);
    }

    #[test]
    fn test_render_with_sadie() {
        // SadieII should also work (same model for now)
        let renderer = BinauralRenderer::new(HrtfDataset::SadieII);
        let samples = vec![1.0f32; 100];
        let pos = Position3D::front_center();
        let (left, right) = renderer.render(&samples, &pos, 48000);
        assert_eq!(left.len(), 100);
        assert_eq!(right.len(), 100);
    }

    #[test]
    fn test_max_itd_reasonable() {
        // Maximum ITD should be a small fraction of a millisecond
        let max_itd = HEAD_RADIUS / SPEED_OF_SOUND * 3.0;
        assert!(max_itd > 0.0);
        assert!(max_itd < 0.001, "Max ITD should be < 1ms, got {}s", max_itd);
    }
}
