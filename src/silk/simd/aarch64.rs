//! aarch64 NEON SIMD implementations for SILK functions.
//!
//! NEON is always available on aarch64, so these are selected at compile time.

use core::arch::aarch64::*;

/// NEON implementation of `silk_noise_shape_quantizer_short_prediction`.
/// Port of `silk/arm/NSQ_neon.h`.
///
/// # Safety
/// Requires aarch64 NEON (always available on aarch64).
#[target_feature(enable = "neon")]
pub unsafe fn silk_noise_shape_quantizer_short_prediction_neon(
    buf32: &[i32],
    coef16: &[i16],
    order: i32,
) -> i32 {
    let b = buf32.len();
    debug_assert!(b >= order as usize);
    debug_assert!(coef16.len() >= order as usize);
    debug_assert!(order == 10 || order == 16);

    let mut out: i32 = order >> 1;

    // Process elements in groups of 4 using NEON
    // buf32 is indexed backwards from end, coef16 forwards
    let mut acc = vdupq_n_s64(0);

    // Process first 8 elements (order 10 has 10, order 16 has 16)
    let iterations = if order == 16 { 4 } else { 2 };

    for k in 0..iterations {
        let base = b - (k * 4 + 4);
        let coef_base = k * 4;

        let buf = vld1q_s32(buf32.as_ptr().add(base));

        // Sign-extend 4 x i16 to 4 x i32, then reverse order so that
        // buf[base+i] pairs with coef16[coef_base + 3 - i] (matching the
        // scalar code where buf32[b-1-j] pairs with coef16[j]).
        let c16 = vld1_s16(coef16.as_ptr().add(coef_base));
        let coef_fwd = vmovl_s16(c16);
        // Reverse 4 x i32: vrev64q reverses within 64-bit halves, then swap halves
        let coef = vextq_s32(vrev64q_s32(coef_fwd), vrev64q_s32(coef_fwd), 2);

        // We need (buf[i] * coef[i]) >> 16 for each element
        // Use widening multiply: i32 * i32 -> i64, then >> 16
        let prod_lo = vmull_s32(vget_low_s32(buf), vget_low_s32(coef));
        let prod_hi = vmull_s32(vget_high_s32(buf), vget_high_s32(coef));

        // Shift right by 16 and accumulate
        acc = vaddq_s64(acc, vshrq_n_s64(prod_lo, 16));
        acc = vaddq_s64(acc, vshrq_n_s64(prod_hi, 16));
    }

    // Horizontal sum of i64 accumulator
    out += (vgetq_lane_s64(acc, 0) + vgetq_lane_s64(acc, 1)) as i32;

    // For order 10, handle remaining 2 elements scalar
    if order == 10 {
        out = (out as i64 + ((buf32[b - 9] as i64 * coef16[8] as i64) >> 16)) as i32;
        out = (out as i64 + ((buf32[b - 10] as i64 * coef16[9] as i64) >> 16)) as i32;
    }

    out
}

/// NEON implementation of `silk_inner_product_FLP`.
/// f32→f64 inner product using NEON widening conversion and FMA.
///
/// # Safety
/// Requires aarch64 NEON (always available on aarch64).
#[target_feature(enable = "neon")]
pub unsafe fn silk_inner_product_flp_neon(data1: &[f32], data2: &[f32]) -> f64 {
    let n = data1.len().min(data2.len());
    let mut acc1 = vdupq_n_f64(0.0);
    let mut acc2 = vdupq_n_f64(0.0);
    let mut i = 0usize;

    // Main loop: 4 f32s per iteration → 2 pairs of f64s
    while i + 3 < n {
        let x = vld1q_f32(data1.as_ptr().add(i));
        let y = vld1q_f32(data2.as_ptr().add(i));

        // Low 2 elements: f32 → f64
        let x_lo = vcvt_f64_f32(vget_low_f32(x));
        let y_lo = vcvt_f64_f32(vget_low_f32(y));
        acc1 = vfmaq_f64(acc1, x_lo, y_lo);

        // High 2 elements: f32 → f64
        let x_hi = vcvt_f64_f32(vget_high_f32(x));
        let y_hi = vcvt_f64_f32(vget_high_f32(y));
        acc2 = vfmaq_f64(acc2, x_hi, y_hi);

        i += 4;
    }

    // Combine accumulators and horizontal sum
    acc1 = vaddq_f64(acc1, acc2);
    let mut result = vgetq_lane_f64(acc1, 0) + vgetq_lane_f64(acc1, 1);

    // Scalar tail for remaining 0-3 elements
    while i < n {
        result += *data1.get_unchecked(i) as f64 * *data2.get_unchecked(i) as f64;
        i += 1;
    }

    result
}
