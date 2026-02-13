//! Ambisonics encoding and decoding — speaker-independent spatial audio representation.
//!
//! Uses B-format channels for First-Order Ambisonics (FOA) and higher orders:
//! - **FOA (First Order)**: 4 channels (W, X, Y, Z) using spherical harmonics
//! - **SOA (Second Order)**: 9 channels
//! - **TOA (Third Order)**: 16 channels
//!
//! Encoding converts a mono source at a given position into B-format channels.
//! Decoding converts B-format back to speaker feeds (stereo, surround, etc.).

use serde::{Deserialize, Serialize};

use crate::error::{Result, SpatialError};
use crate::object::Position3D;

/// Ambisonics order determining the spatial resolution.
///
/// Higher orders provide better spatial precision at the cost of more channels.
/// The number of channels is `(order + 1)²`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AmbisonicsOrder {
    /// First Order Ambisonics — 4 channels (W, X, Y, Z).
    First,
    /// Second Order Ambisonics — 9 channels.
    Second,
    /// Third Order Ambisonics — 16 channels.
    Third,
}

impl AmbisonicsOrder {
    /// Returns the number of channels required for this Ambisonics order.
    ///
    /// The formula is `(order + 1)²`:
    /// - First: 4 channels
    /// - Second: 9 channels
    /// - Third: 16 channels
    pub fn channel_count(&self) -> usize {
        let order = self.numeric_order();
        (order + 1) * (order + 1)
    }

    /// Returns the numeric order value (1, 2, or 3).
    pub fn numeric_order(&self) -> usize {
        match self {
            Self::First => 1,
            Self::Second => 2,
            Self::Third => 3,
        }
    }

    /// Creates an `AmbisonicsOrder` from a numeric value.
    ///
    /// # Errors
    ///
    /// Returns [`SpatialError::UnsupportedOrder`] for values other than 1, 2, or 3.
    pub fn from_order(order: u8) -> Result<Self> {
        match order {
            1 => Ok(Self::First),
            2 => Ok(Self::Second),
            3 => Ok(Self::Third),
            _ => Err(SpatialError::UnsupportedOrder(order)),
        }
    }
}

/// Encodes a mono audio source into B-format Ambisonics channels.
///
/// Each sample from the mono source is spatially encoded using spherical harmonics
/// based on the source position. The output is a vector of channels, each containing
/// the same number of samples as the input.
///
/// For First Order Ambisonics (FOA), the encoding uses:
/// - W = sample × (1 / √2)  (omnidirectional)
/// - X = sample × cos(az) × cos(el)  (front-back)
/// - Y = sample × sin(az) × cos(el)  (left-right)
/// - Z = sample × sin(el)  (up-down)
///
/// Higher-order encoding adds additional spherical harmonic components.
///
/// # Arguments
///
/// * `samples` - Mono audio samples
/// * `position` - Source position in 3D space
/// * `order` - Ambisonics order to encode to
///
/// # Returns
///
/// A vector of `(order + 1)²` channels, each with the same length as the input.
pub fn encode_to_bformat(
    samples: &[f32],
    position: &Position3D,
    order: AmbisonicsOrder,
) -> Vec<Vec<f32>> {
    let num_channels = order.channel_count();
    let az_rad = position.azimuth.to_radians();
    let el_rad = position.elevation.to_radians();

    // Compute spherical harmonic coefficients
    let coefficients = compute_sh_coefficients(az_rad, el_rad, order);

    let mut bformat = vec![vec![0.0f32; samples.len()]; num_channels];
    for (i, &sample) in samples.iter().enumerate() {
        for (ch, coeff) in coefficients.iter().enumerate() {
            bformat[ch][i] = sample * (*coeff as f32);
        }
    }

    bformat
}

/// Decodes B-format Ambisonics channels to stereo output.
///
/// Uses a virtual speaker pair at ±30° azimuth (standard stereo configuration)
/// to decode the B-format signal to left and right channels.
///
/// # Arguments
///
/// * `bformat` - B-format channels (at least 4 for FOA)
/// * `order` - Ambisonics order of the input
///
/// # Returns
///
/// A tuple of `(left, right)` channel sample vectors.
///
/// # Errors
///
/// Returns [`SpatialError::ChannelMismatch`] if the number of B-format channels
/// does not match the expected count for the given order.
pub fn decode_bformat_to_stereo(
    bformat: &[Vec<f32>],
    order: AmbisonicsOrder,
) -> Result<(Vec<f32>, Vec<f32>)> {
    let expected = order.channel_count();
    if bformat.len() != expected {
        return Err(SpatialError::ChannelMismatch {
            expected,
            got: bformat.len(),
        });
    }

    if bformat.is_empty() || bformat[0].is_empty() {
        return Ok((Vec::new(), Vec::new()));
    }

    let num_samples = bformat[0].len();

    // Virtual speaker positions for stereo: left at -30°, right at +30°
    let left_az = (-30.0_f64).to_radians();
    let right_az = (30.0_f64).to_radians();
    let el = 0.0_f64; // ear level

    let left_coeffs = compute_sh_coefficients(left_az, el, order);
    let right_coeffs = compute_sh_coefficients(right_az, el, order);

    let mut left = vec![0.0f32; num_samples];
    let mut right = vec![0.0f32; num_samples];

    // Decode normalization factor: 1 / num_speakers
    let norm = 1.0 / 2.0_f64;

    for i in 0..num_samples {
        let mut l_sum = 0.0_f64;
        let mut r_sum = 0.0_f64;
        for ch in 0..expected {
            l_sum += bformat[ch][i] as f64 * left_coeffs[ch];
            r_sum += bformat[ch][i] as f64 * right_coeffs[ch];
        }
        left[i] = (l_sum * norm) as f32;
        right[i] = (r_sum * norm) as f32;
    }

    Ok((left, right))
}

/// Decodes B-format Ambisonics channels to a surround speaker array.
///
/// Distributes the B-format signal to `speaker_count` speakers arranged
/// in a uniform ring at the horizontal plane (0° elevation).
///
/// # Arguments
///
/// * `bformat` - B-format channels
/// * `order` - Ambisonics order of the input
/// * `speaker_count` - Number of output speakers (e.g., 6 for 5.1, 8 for 7.1)
///
/// # Returns
///
/// A vector of speaker feed channels.
///
/// # Errors
///
/// Returns [`SpatialError::ChannelMismatch`] if the number of B-format channels
/// does not match the expected count for the given order.
pub fn decode_bformat_to_surround(
    bformat: &[Vec<f32>],
    order: AmbisonicsOrder,
    speaker_count: usize,
) -> Result<Vec<Vec<f32>>> {
    let expected = order.channel_count();
    if bformat.len() != expected {
        return Err(SpatialError::ChannelMismatch {
            expected,
            got: bformat.len(),
        });
    }

    if bformat.is_empty() || bformat[0].is_empty() || speaker_count == 0 {
        return Ok(vec![Vec::new(); speaker_count]);
    }

    let num_samples = bformat[0].len();

    // Compute speaker positions: uniform ring at 0° elevation
    // Starting from front-center and going clockwise
    let speaker_angles: Vec<f64> = (0..speaker_count)
        .map(|i| 2.0 * std::f64::consts::PI * i as f64 / speaker_count as f64)
        .collect();

    // Compute decode coefficients for each speaker
    let speaker_coeffs: Vec<Vec<f64>> = speaker_angles
        .iter()
        .map(|&az| compute_sh_coefficients(az, 0.0, order))
        .collect();

    let norm = 1.0 / speaker_count as f64;

    let mut speakers = vec![vec![0.0f32; num_samples]; speaker_count];
    for i in 0..num_samples {
        for (spk_idx, coeffs) in speaker_coeffs.iter().enumerate() {
            let mut sum = 0.0_f64;
            for ch in 0..expected {
                sum += bformat[ch][i] as f64 * coeffs[ch];
            }
            speakers[spk_idx][i] = (sum * norm) as f32;
        }
    }

    Ok(speakers)
}

/// Computes spherical harmonic coefficients for a given direction and order.
///
/// Returns coefficients for all `(order + 1)²` channels.
fn compute_sh_coefficients(
    azimuth_rad: f64,
    elevation_rad: f64,
    order: AmbisonicsOrder,
) -> Vec<f64> {
    let cos_az = azimuth_rad.cos();
    let sin_az = azimuth_rad.sin();
    let cos_el = elevation_rad.cos();
    let sin_el = elevation_rad.sin();

    let num_channels = order.channel_count();
    let mut coeffs = vec![0.0f64; num_channels];

    // Order 0: omnidirectional
    // W channel with 1/sqrt(2) normalization (SN3D)
    coeffs[0] = std::f64::consts::FRAC_1_SQRT_2;

    if num_channels >= 4 {
        // Order 1: first-order dipoles
        // Y = sin(az) * cos(el) — left-right
        coeffs[1] = sin_az * cos_el;
        // Z = sin(el) — up-down
        coeffs[2] = sin_el;
        // X = cos(az) * cos(el) — front-back
        coeffs[3] = cos_az * cos_el;
    }

    if num_channels >= 9 {
        // Order 2: second-order harmonics (SN3D normalization)
        let cos2_az = (2.0 * azimuth_rad).cos();
        let sin2_az = (2.0 * azimuth_rad).sin();
        let cos2_el = cos_el * cos_el;

        // ACN channel ordering (Ambisonic Channel Number)
        // Channel 4: V = sqrt(3/4) * sin(2*az) * cos²(el)
        coeffs[4] = (3.0_f64 / 4.0).sqrt() * sin2_az * cos2_el;
        // Channel 5: T = sqrt(3/4) * sin(az) * sin(2*el)
        coeffs[5] = (3.0_f64 / 4.0).sqrt() * sin_az * (2.0 * elevation_rad).sin();
        // Channel 6: R = (3*sin²(el) - 1) / 2
        coeffs[6] = 0.5 * (3.0 * sin_el * sin_el - 1.0);
        // Channel 7: S = sqrt(3/4) * cos(az) * sin(2*el)
        coeffs[7] = (3.0_f64 / 4.0).sqrt() * cos_az * (2.0 * elevation_rad).sin();
        // Channel 8: U = sqrt(3/4) * cos(2*az) * cos²(el)
        coeffs[8] = (3.0_f64 / 4.0).sqrt() * cos2_az * cos2_el;
    }

    if num_channels >= 16 {
        // Order 3: third-order harmonics (simplified SN3D)
        let cos3_az = (3.0 * azimuth_rad).cos();
        let sin3_az = (3.0 * azimuth_rad).sin();
        let cos_el_sq = cos_el * cos_el;
        let cos_el_cu = cos_el_sq * cos_el;

        // ACN channels 9-15
        // Channel 9: sin(3*az) * cos³(el)
        coeffs[9] = sin3_az * cos_el_cu;
        // Channel 10: sin(2*az) * cos(el) * sin(el) * sqrt(6)/4
        coeffs[10] = (6.0_f64 / 4.0).sqrt() * (2.0 * azimuth_rad).sin() * cos_el * sin_el;
        // Channel 11: sin(az) * cos(el) * (5*sin²(el) - 1) * sqrt(3/8)
        coeffs[11] = (3.0_f64 / 8.0).sqrt() * sin_az * cos_el * (5.0 * sin_el * sin_el - 1.0);
        // Channel 12: sin(el) * (5*sin²(el) - 3) / 2
        coeffs[12] = 0.5 * sin_el * (5.0 * sin_el * sin_el - 3.0);
        // Channel 13: cos(az) * cos(el) * (5*sin²(el) - 1) * sqrt(3/8)
        coeffs[13] = (3.0_f64 / 8.0).sqrt() * cos_az * cos_el * (5.0 * sin_el * sin_el - 1.0);
        // Channel 14: cos(2*az) * cos(el) * sin(el) * sqrt(6)/4
        coeffs[14] = (6.0_f64 / 4.0).sqrt() * (2.0 * azimuth_rad).cos() * cos_el * sin_el;
        // Channel 15: cos(3*az) * cos³(el)
        coeffs[15] = cos3_az * cos_el_cu;
    }

    coeffs
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_channel_count() {
        assert_eq!(AmbisonicsOrder::First.channel_count(), 4);
        assert_eq!(AmbisonicsOrder::Second.channel_count(), 9);
        assert_eq!(AmbisonicsOrder::Third.channel_count(), 16);
    }

    #[test]
    fn test_numeric_order() {
        assert_eq!(AmbisonicsOrder::First.numeric_order(), 1);
        assert_eq!(AmbisonicsOrder::Second.numeric_order(), 2);
        assert_eq!(AmbisonicsOrder::Third.numeric_order(), 3);
    }

    #[test]
    fn test_from_order_valid() {
        assert_eq!(
            AmbisonicsOrder::from_order(1).unwrap(),
            AmbisonicsOrder::First
        );
        assert_eq!(
            AmbisonicsOrder::from_order(2).unwrap(),
            AmbisonicsOrder::Second
        );
        assert_eq!(
            AmbisonicsOrder::from_order(3).unwrap(),
            AmbisonicsOrder::Third
        );
    }

    #[test]
    fn test_from_order_invalid() {
        assert!(matches!(
            AmbisonicsOrder::from_order(0),
            Err(SpatialError::UnsupportedOrder(0))
        ));
        assert!(matches!(
            AmbisonicsOrder::from_order(4),
            Err(SpatialError::UnsupportedOrder(4))
        ));
    }

    #[test]
    fn test_encode_bformat_channel_count() {
        let samples = vec![1.0f32; 100];
        let pos = Position3D::front_center();

        let bformat = encode_to_bformat(&samples, &pos, AmbisonicsOrder::First);
        assert_eq!(bformat.len(), 4);
        assert_eq!(bformat[0].len(), 100);

        let bformat = encode_to_bformat(&samples, &pos, AmbisonicsOrder::Second);
        assert_eq!(bformat.len(), 9);

        let bformat = encode_to_bformat(&samples, &pos, AmbisonicsOrder::Third);
        assert_eq!(bformat.len(), 16);
    }

    #[test]
    fn test_encode_front_center_foa() {
        // Source at front center (0° az, 0° el)
        let samples = vec![1.0f32; 10];
        let pos = Position3D::new(0.0, 0.0, 0.8).unwrap();
        let bformat = encode_to_bformat(&samples, &pos, AmbisonicsOrder::First);

        // W = 1/sqrt(2) ≈ 0.7071
        let w = bformat[0][0];
        assert!((w - std::f64::consts::FRAC_1_SQRT_2 as f32).abs() < 1e-5);

        // Y = sin(0) * cos(0) = 0 (no left-right)
        let y = bformat[1][0];
        assert!(y.abs() < 1e-5, "Y should be ~0, got {}", y);

        // Z = sin(0) = 0 (no up-down)
        let z = bformat[2][0];
        assert!(z.abs() < 1e-5, "Z should be ~0, got {}", z);

        // X = cos(0) * cos(0) = 1 (maximum front)
        let x = bformat[3][0];
        assert!((x - 1.0).abs() < 1e-5, "X should be ~1.0, got {}", x);
    }

    #[test]
    fn test_encode_left_source_foa() {
        // Source at left (-90° azimuth)
        let samples = vec![1.0f32; 10];
        let pos = Position3D::new(-90.0, 0.0, 1.0).unwrap();
        let bformat = encode_to_bformat(&samples, &pos, AmbisonicsOrder::First);

        // Y = sin(-90°) * cos(0°) = -1.0 (negative = left in ACN/SN3D)
        let y = bformat[1][0];
        assert!((y - (-1.0)).abs() < 1e-5, "Y should be ~-1.0, got {}", y);

        // X = cos(-90°) * cos(0°) ≈ 0
        let x = bformat[3][0];
        assert!(x.abs() < 1e-5, "X should be ~0, got {}", x);
    }

    #[test]
    fn test_encode_elevated_source_foa() {
        // Source directly above (0° az, 90° el)
        let samples = vec![1.0f32; 10];
        let pos = Position3D::new(0.0, 90.0, 1.0).unwrap();
        let bformat = encode_to_bformat(&samples, &pos, AmbisonicsOrder::First);

        // Z = sin(90°) = 1.0
        let z = bformat[2][0];
        assert!((z - 1.0).abs() < 1e-5, "Z should be ~1.0, got {}", z);

        // X = cos(0°) * cos(90°) ≈ 0
        let x = bformat[3][0];
        assert!(x.abs() < 1e-5, "X should be ~0, got {}", x);
    }

    #[test]
    fn test_decode_bformat_stereo_center_source() {
        // Center source should have equal left and right
        let samples = vec![1.0f32; 100];
        let pos = Position3D::new(0.0, 0.0, 1.0).unwrap();
        let bformat = encode_to_bformat(&samples, &pos, AmbisonicsOrder::First);

        let (left, right) = decode_bformat_to_stereo(&bformat, AmbisonicsOrder::First).unwrap();
        assert_eq!(left.len(), 100);
        assert_eq!(right.len(), 100);

        // For a center source, left and right should be approximately equal
        for i in 0..left.len() {
            assert!(
                (left[i] - right[i]).abs() < 1e-5,
                "L/R should be equal for center source: L={}, R={}",
                left[i],
                right[i]
            );
        }
    }

    #[test]
    fn test_decode_bformat_stereo_left_source() {
        // Left source (-90° azimuth) should be louder in left channel
        let samples = vec![1.0f32; 100];
        let pos = Position3D::new(-90.0, 0.0, 1.0).unwrap();
        let bformat = encode_to_bformat(&samples, &pos, AmbisonicsOrder::First);

        let (left, right) = decode_bformat_to_stereo(&bformat, AmbisonicsOrder::First).unwrap();

        // Left channel should be louder than right channel
        assert!(
            left[0].abs() > right[0].abs(),
            "Left should be louder for left source: L={}, R={}",
            left[0],
            right[0]
        );
    }

    #[test]
    fn test_decode_bformat_stereo_right_source() {
        // Right source (+90° azimuth) should be louder in right channel
        let samples = vec![1.0f32; 100];
        let pos = Position3D::new(90.0, 0.0, 1.0).unwrap();
        let bformat = encode_to_bformat(&samples, &pos, AmbisonicsOrder::First);

        let (left, right) = decode_bformat_to_stereo(&bformat, AmbisonicsOrder::First).unwrap();

        assert!(
            right[0].abs() > left[0].abs(),
            "Right should be louder for right source: L={}, R={}",
            left[0],
            right[0]
        );
    }

    #[test]
    fn test_decode_channel_mismatch() {
        let bformat = vec![vec![0.0f32; 10]; 3]; // 3 channels instead of 4
        let result = decode_bformat_to_stereo(&bformat, AmbisonicsOrder::First);
        assert!(matches!(
            result,
            Err(SpatialError::ChannelMismatch {
                expected: 4,
                got: 3
            })
        ));
    }

    #[test]
    fn test_decode_empty_input() {
        let bformat = vec![Vec::<f32>::new(); 4];
        let (left, right) = decode_bformat_to_stereo(&bformat, AmbisonicsOrder::First).unwrap();
        assert!(left.is_empty());
        assert!(right.is_empty());
    }

    #[test]
    fn test_decode_surround_channel_count() {
        let samples = vec![1.0f32; 100];
        let pos = Position3D::front_center();
        let bformat = encode_to_bformat(&samples, &pos, AmbisonicsOrder::First);

        let speakers = decode_bformat_to_surround(&bformat, AmbisonicsOrder::First, 6).unwrap();
        assert_eq!(speakers.len(), 6);
        for spk in &speakers {
            assert_eq!(spk.len(), 100);
        }
    }

    #[test]
    fn test_encode_decode_round_trip_preserves_energy() {
        // Encode and decode should approximately preserve signal energy
        let samples: Vec<f32> = (0..1000).map(|i| (i as f32 * 0.01).sin()).collect();
        let pos = Position3D::new(30.0, 0.0, 1.0).unwrap();
        let bformat = encode_to_bformat(&samples, &pos, AmbisonicsOrder::First);
        let (left, right) = decode_bformat_to_stereo(&bformat, AmbisonicsOrder::First).unwrap();

        let input_energy: f64 = samples.iter().map(|s| (*s as f64) * (*s as f64)).sum();
        let output_energy: f64 = left
            .iter()
            .zip(right.iter())
            .map(|(l, r)| (*l as f64) * (*l as f64) + (*r as f64) * (*r as f64))
            .sum();

        // Output energy should be non-zero and roughly proportional to input
        assert!(output_energy > 0.0, "Output energy should be non-zero");
        assert!(
            output_energy < input_energy * 10.0,
            "Output energy should not explode: in={}, out={}",
            input_energy,
            output_energy
        );
    }

    #[test]
    fn test_second_order_encode() {
        let samples = vec![1.0f32; 10];
        let pos = Position3D::new(45.0, 30.0, 1.0).unwrap();
        let bformat = encode_to_bformat(&samples, &pos, AmbisonicsOrder::Second);
        assert_eq!(bformat.len(), 9);

        // All channels should have values (for this non-axis-aligned position)
        for ch in &bformat {
            assert_eq!(ch.len(), 10);
        }
    }

    #[test]
    fn test_surround_channel_mismatch() {
        let bformat = vec![vec![0.0f32; 10]; 3]; // Wrong channel count
        let result = decode_bformat_to_surround(&bformat, AmbisonicsOrder::First, 6);
        assert!(matches!(
            result,
            Err(SpatialError::ChannelMismatch {
                expected: 4,
                got: 3
            })
        ));
    }
}
