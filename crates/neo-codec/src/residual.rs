//! Residual layer — computes and encodes the difference between original and DAC-decoded audio.
//!
//! This enables bit-perfect lossless reconstruction when combined with a lossy codec:
//!
//! ```text
//! original = dac_decoded + residual
//! ```
//!
//! The residual signal captures what the lossy codec lost, allowing the NEO format
//! to offer both lossy playback (fast) and lossless reconstruction (exact).

/// Compute the residual (difference) between original PCM and lossy-decoded PCM.
///
/// # Panics
///
/// Panics if `original` and `decoded` have different lengths.
pub fn compute_residual(original: &[f32], decoded: &[f32]) -> Vec<f32> {
    assert_eq!(
        original.len(),
        decoded.len(),
        "PCM lengths must match for residual computation"
    );
    original
        .iter()
        .zip(decoded.iter())
        .map(|(o, d)| o - d)
        .collect()
}

/// Reconstruct the original PCM from lossy-decoded PCM and its residual.
///
/// # Panics
///
/// Panics if `decoded` and `residual` have different lengths.
pub fn reconstruct(decoded: &[f32], residual: &[f32]) -> Vec<f32> {
    assert_eq!(
        decoded.len(),
        residual.len(),
        "PCM lengths must match for reconstruction"
    );
    decoded
        .iter()
        .zip(residual.iter())
        .map(|(d, r)| d + r)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_residual_round_trip() {
        let original = vec![0.5, -0.3, 0.8, -0.1, 0.0];
        let decoded = vec![0.48, -0.32, 0.79, -0.12, 0.01];

        let residual = compute_residual(&original, &decoded);
        let reconstructed = reconstruct(&decoded, &residual);

        for (o, r) in original.iter().zip(reconstructed.iter()) {
            assert!(
                (o - r).abs() < f32::EPSILON,
                "Reconstruction should be bit-perfect"
            );
        }
    }

    #[test]
    fn test_residual_of_identical_signals_is_all_zeros() {
        let signal = vec![0.1, -0.5, 0.99, 0.0, -1.0, 0.42];
        let residual = compute_residual(&signal, &signal);

        for (i, &r) in residual.iter().enumerate() {
            assert_eq!(r, 0.0, "Residual at index {i} should be exactly 0.0");
        }
    }
}
