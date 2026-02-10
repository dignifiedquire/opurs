/// Upstream C: (Rust-specific helper, no direct C equivalent)
pub fn silk_exp2(x: f32) -> f32 {
    2f64.powf(x as f64) as f32
}
