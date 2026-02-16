//! Floating-point energy computation.
//!
//! Upstream C: `silk/float/energy_FLP.c`

/// Upstream C: silk/float/SigProc_FLP.h:silk_energy_FLP
///
/// Sum of squares of a float array, with result as a double
pub fn silk_energy_FLP(data: &[f32]) -> f64 {
    // opus sources unfold it manually, but LLVM seems to be able to 4x unfold it by itself
    // SIMD might still be nice idk
    data.iter()
        .fold(0.0f64, |acc, &x| acc + x as f64 * x as f64)
}
