//! SILK math operations.
//!
//! Upstream c: (no direct c equivalent, SILK-specific math)

/// Upstream c: (Rust-specific helper, no direct c equivalent)
pub fn silk_exp2(x: f32) -> f32 {
    2f64.powf(x as f64) as f32
}
