//! aarch64 NEON SIMD implementations for CELT functions.
//!
//! NEON is always available on aarch64, so these are selected at compile time
//! (no runtime detection needed).

use core::arch::aarch64::*;

/// NEON implementation of `xcorr_kernel`.
/// Port of `celt/arm/pitch_neon_intr.c:xcorr_kernel_neon_float`.
///
/// # Safety
/// Requires aarch64 NEON (always available on aarch64).
#[target_feature(enable = "neon")]
pub unsafe fn xcorr_kernel_neon(x: &[f32], y: &[f32], sum: &mut [f32; 4], len: usize) {
    debug_assert!(len >= 3);
    debug_assert!(x.len() >= len);
    debug_assert!(y.len() >= len + 3);

    let mut xsum1 = vld1q_f32(sum.as_ptr());
    let mut xsum2 = vdupq_n_f32(0.0);

    let mut j = 0usize;
    while j + 3 < len {
        let x0 = vld1q_f32(x.as_ptr().add(j));
        let yj = vld1q_f32(y.as_ptr().add(j));
        let y3 = vld1q_f32(y.as_ptr().add(j + 3));

        // x[j+0] broadcast * y[j..j+3]
        xsum1 = vmlaq_laneq_f32(xsum1, yj, x0, 0);
        // x[j+1] broadcast * y[j+1..j+4]
        xsum2 = vmlaq_laneq_f32(xsum2, vextq_f32(yj, y3, 1), x0, 1);
        // x[j+2] broadcast * y[j+2..j+5]
        xsum1 = vmlaq_laneq_f32(xsum1, vextq_f32(yj, y3, 2), x0, 2);
        // x[j+3] broadcast * y[j+3..j+6]
        xsum2 = vmlaq_laneq_f32(xsum2, y3, x0, 3);

        j += 4;
    }

    xsum1 = vaddq_f32(xsum1, xsum2);

    // Handle remaining 1-3 elements
    while j < len {
        let xj = vdupq_n_f32(*x.get_unchecked(j));
        let yj = vld1q_f32(y.as_ptr().add(j));
        xsum1 = vmlaq_f32(xsum1, xj, yj);
        j += 1;
    }

    vst1q_f32(sum.as_mut_ptr(), xsum1);
}

/// NEON implementation of `celt_inner_prod`.
/// Port of `celt/arm/pitch_neon_intr.c:celt_inner_prod_neon`.
///
/// # Safety
/// Requires aarch64 NEON (always available on aarch64).
#[target_feature(enable = "neon")]
pub unsafe fn celt_inner_prod_neon(x: &[f32], y: &[f32], n: usize) -> f32 {
    debug_assert!(x.len() >= n);
    debug_assert!(y.len() >= n);

    let mut sum1 = vdupq_n_f32(0.0);
    let mut sum2 = vdupq_n_f32(0.0);
    let mut i = 0usize;

    // Process 8 floats at a time with two accumulators
    while i + 7 < n {
        let x0 = vld1q_f32(x.as_ptr().add(i));
        let y0 = vld1q_f32(y.as_ptr().add(i));
        let x1 = vld1q_f32(x.as_ptr().add(i + 4));
        let y1 = vld1q_f32(y.as_ptr().add(i + 4));
        sum1 = vmlaq_f32(sum1, x0, y0);
        sum2 = vmlaq_f32(sum2, x1, y1);
        i += 8;
    }

    // Process remaining 4
    if i + 3 < n {
        let x0 = vld1q_f32(x.as_ptr().add(i));
        let y0 = vld1q_f32(y.as_ptr().add(i));
        sum1 = vmlaq_f32(sum1, x0, y0);
        i += 4;
    }

    sum1 = vaddq_f32(sum1, sum2);

    // Horizontal sum
    let sum_f32x2 = vadd_f32(vget_low_f32(sum1), vget_high_f32(sum1));
    let mut result = vget_lane_f32(vpadd_f32(sum_f32x2, sum_f32x2), 0);

    // Handle remaining elements
    while i < n {
        result += *x.get_unchecked(i) * *y.get_unchecked(i);
        i += 1;
    }

    result
}

/// NEON implementation of `dual_inner_prod`.
///
/// # Safety
/// Requires aarch64 NEON (always available on aarch64).
#[target_feature(enable = "neon")]
pub unsafe fn dual_inner_prod_neon(x: &[f32], y01: &[f32], y02: &[f32], n: usize) -> (f32, f32) {
    debug_assert!(x.len() >= n);
    debug_assert!(y01.len() >= n);
    debug_assert!(y02.len() >= n);

    let mut sum1 = vdupq_n_f32(0.0);
    let mut sum2 = vdupq_n_f32(0.0);
    let mut i = 0usize;

    while i + 3 < n {
        let xv = vld1q_f32(x.as_ptr().add(i));
        let y1v = vld1q_f32(y01.as_ptr().add(i));
        let y2v = vld1q_f32(y02.as_ptr().add(i));
        sum1 = vmlaq_f32(sum1, xv, y1v);
        sum2 = vmlaq_f32(sum2, xv, y2v);
        i += 4;
    }

    // Horizontal sum for sum1
    let s1 = vadd_f32(vget_low_f32(sum1), vget_high_f32(sum1));
    let mut xy01 = vget_lane_f32(vpadd_f32(s1, s1), 0);

    // Horizontal sum for sum2
    let s2 = vadd_f32(vget_low_f32(sum2), vget_high_f32(sum2));
    let mut xy02 = vget_lane_f32(vpadd_f32(s2, s2), 0);

    // Handle remaining elements
    while i < n {
        let xi = *x.get_unchecked(i);
        xy01 += xi * *y01.get_unchecked(i);
        xy02 += xi * *y02.get_unchecked(i);
        i += 1;
    }

    (xy01, xy02)
}

/// NEON implementation of `celt_pitch_xcorr`.
/// Processes 4 correlations at a time using `xcorr_kernel_neon`.
///
/// # Safety
/// Requires aarch64 NEON (always available on aarch64).
#[target_feature(enable = "neon")]
pub unsafe fn celt_pitch_xcorr_neon(x: &[f32], y: &[f32], xcorr: &mut [f32], len: usize) {
    let max_pitch = xcorr.len();
    debug_assert!(max_pitch > 0);
    debug_assert!(x.len() >= len);

    let mut i = 0i32;
    while i < max_pitch as i32 - 3 {
        let mut sum = [0.0f32; 4];
        xcorr_kernel_neon(&x[..len], &y[i as usize..], &mut sum, len);
        xcorr[i as usize] = sum[0];
        xcorr[i as usize + 1] = sum[1];
        xcorr[i as usize + 2] = sum[2];
        xcorr[i as usize + 3] = sum[3];
        i += 4;
    }
    while (i as usize) < max_pitch {
        xcorr[i as usize] = celt_inner_prod_neon(x, &y[i as usize..], len);
        i += 1;
    }
}
