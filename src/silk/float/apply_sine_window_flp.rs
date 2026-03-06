//! Floating-point sine window application.
//!
//! Upstream c: `silk/float/apply_sine_window_FLP.c`

/// Upstream c: silk/float/apply_sine_window_FLP.c:silk_apply_sine_window_FLP
pub fn silk_apply_sine_window_flp(px_win: &mut [f32], px: &[f32], win_type: i32, length: i32) {
    let mut k: i32;

    let mut s0: f32;
    let mut s1: f32;
    debug_assert!(win_type == 1 || win_type == 2);
    debug_assert!(length & 3 == 0);
    let freq: f32 = std::f32::consts::PI / (length + 1) as f32;
    let c: f32 = 2.0f32 - freq * freq;
    if win_type < 2 {
        s0 = 0.0f32;
        s1 = freq;
    } else {
        s0 = 1.0f32;
        s1 = 0.5f32 * c;
    }
    k = 0;
    while k < length {
        px_win[k as usize] = px[k as usize] * 0.5f32 * (s0 + s1);
        px_win[(k + 1) as usize] = px[(k + 1) as usize] * s1;
        s0 = c * s1 - s0;
        px_win[(k + 2) as usize] = px[(k + 2) as usize] * 0.5f32 * (s1 + s0);
        px_win[(k + 3) as usize] = px[(k + 3) as usize] * s0;
        s1 = c * s0 - s1;
        k += 4;
    }
}
