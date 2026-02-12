//! aarch64 NEON SIMD implementations for CELT functions.
//!
//! NEON is always available on aarch64, so these are selected at compile time
//! (no runtime detection needed).

use core::arch::aarch64::*;

/// NEON implementation of `xcorr_kernel`.
/// Port of `celt/arm/celt_neon_intr.c:xcorr_kernel_neon_float`.
///
/// # Safety
/// Requires aarch64 NEON (always available on aarch64).
#[target_feature(enable = "neon")]
pub unsafe fn xcorr_kernel_neon(x: &[f32], y: &[f32], sum: &mut [f32; 4], len: usize) {
    debug_assert!(len > 0);
    debug_assert!(x.len() >= len);
    debug_assert!(y.len() >= len + 3);

    let xi = x.as_ptr();
    let mut yi = y.as_ptr();
    let mut xi_off = 0usize;

    let mut yy = [vdupq_n_f32(0.0); 3];
    let mut yext = [vdupq_n_f32(0.0); 3];
    let mut xx = [vdupq_n_f32(0.0); 2];

    yy[0] = vld1q_f32(yi);
    let mut summ = vdupq_n_f32(0.0);

    let mut remaining = len;

    // Process 8 elements at a time (C: while (len > 8))
    while remaining > 8 {
        yi = yi.add(4);
        yy[1] = vld1q_f32(yi);
        yi = yi.add(4);
        yy[2] = vld1q_f32(yi);

        xx[0] = vld1q_f32(xi.add(xi_off));
        xi_off += 4;
        xx[1] = vld1q_f32(xi.add(xi_off));
        xi_off += 4;

        // C: vmlaq_lane_f32 → vfmaq_lane_f32 (FMA remapping)
        // Uses float32x2_t from vget_low/vget_high, lane 0 or 1
        summ = vfmaq_lane_f32(summ, yy[0], vget_low_f32(xx[0]), 0);
        yext[0] = vextq_f32(yy[0], yy[1], 1);
        summ = vfmaq_lane_f32(summ, yext[0], vget_low_f32(xx[0]), 1);
        yext[1] = vextq_f32(yy[0], yy[1], 2);
        summ = vfmaq_lane_f32(summ, yext[1], vget_high_f32(xx[0]), 0);
        yext[2] = vextq_f32(yy[0], yy[1], 3);
        summ = vfmaq_lane_f32(summ, yext[2], vget_high_f32(xx[0]), 1);

        summ = vfmaq_lane_f32(summ, yy[1], vget_low_f32(xx[1]), 0);
        yext[0] = vextq_f32(yy[1], yy[2], 1);
        summ = vfmaq_lane_f32(summ, yext[0], vget_low_f32(xx[1]), 1);
        yext[1] = vextq_f32(yy[1], yy[2], 2);
        summ = vfmaq_lane_f32(summ, yext[1], vget_high_f32(xx[1]), 0);
        yext[2] = vextq_f32(yy[1], yy[2], 3);
        summ = vfmaq_lane_f32(summ, yext[2], vget_high_f32(xx[1]), 1);

        yy[0] = yy[2];
        remaining -= 8;
    }

    // Process 4 more elements if available (C: if (len > 4))
    if remaining > 4 {
        yi = yi.add(4);
        yy[1] = vld1q_f32(yi);

        xx[0] = vld1q_f32(xi.add(xi_off));
        xi_off += 4;

        summ = vfmaq_lane_f32(summ, yy[0], vget_low_f32(xx[0]), 0);
        yext[0] = vextq_f32(yy[0], yy[1], 1);
        summ = vfmaq_lane_f32(summ, yext[0], vget_low_f32(xx[0]), 1);
        yext[1] = vextq_f32(yy[0], yy[1], 2);
        summ = vfmaq_lane_f32(summ, yext[1], vget_high_f32(xx[0]), 0);
        yext[2] = vextq_f32(yy[0], yy[1], 3);
        summ = vfmaq_lane_f32(summ, yext[2], vget_high_f32(xx[0]), 1);

        yy[0] = yy[1];
        remaining -= 4;
    }

    // Scalar tail (C: while (--len > 0) { ... } then final element)
    while remaining > 1 {
        remaining -= 1;
        let xx_2 = vld1_dup_f32(xi.add(xi_off));
        xi_off += 1;
        summ = vfmaq_lane_f32(summ, yy[0], xx_2, 0);
        yi = yi.add(1);
        yy[0] = vld1q_f32(yi);
    }

    // Final element
    let xx_2 = vld1_dup_f32(xi.add(xi_off));
    summ = vfmaq_lane_f32(summ, yy[0], xx_2, 0);

    vst1q_f32(sum.as_mut_ptr(), summ);
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

    let mut xy_f32x4 = vdupq_n_f32(0.0);
    let mut i = 0usize;

    // Process 8 floats at a time with ONE accumulator (matching C exactly)
    while i + 7 < n {
        let x_f32x4 = vld1q_f32(x.as_ptr().add(i));
        let y_f32x4 = vld1q_f32(y.as_ptr().add(i));
        xy_f32x4 = vfmaq_f32(xy_f32x4, x_f32x4, y_f32x4);
        let x_f32x4 = vld1q_f32(x.as_ptr().add(i + 4));
        let y_f32x4 = vld1q_f32(y.as_ptr().add(i + 4));
        xy_f32x4 = vfmaq_f32(xy_f32x4, x_f32x4, y_f32x4);
        i += 8;
    }

    // Process remaining 4
    if n - i >= 4 {
        let x_f32x4 = vld1q_f32(x.as_ptr().add(i));
        let y_f32x4 = vld1q_f32(y.as_ptr().add(i));
        xy_f32x4 = vfmaq_f32(xy_f32x4, x_f32x4, y_f32x4);
        i += 4;
    }

    // Horizontal sum
    let xy_f32x2 = vadd_f32(vget_low_f32(xy_f32x4), vget_high_f32(xy_f32x4));
    let xy_f32x2 = vpadd_f32(xy_f32x2, xy_f32x2);
    let mut xy = vget_lane_f32(xy_f32x2, 0);

    // Handle remaining elements
    while i < n {
        xy += *x.get_unchecked(i) * *y.get_unchecked(i);
        i += 1;
    }

    xy
}

/// NEON implementation of `dual_inner_prod`.
/// Port of `celt/arm/pitch_neon_intr.c:dual_inner_prod_neon` (float path).
///
/// # Safety
/// Requires aarch64 NEON (always available on aarch64).
#[target_feature(enable = "neon")]
pub unsafe fn dual_inner_prod_neon(x: &[f32], y01: &[f32], y02: &[f32], n: usize) -> (f32, f32) {
    debug_assert!(x.len() >= n);
    debug_assert!(y01.len() >= n);
    debug_assert!(y02.len() >= n);

    let mut xy01_f32x4 = vdupq_n_f32(0.0);
    let mut xy02_f32x4 = vdupq_n_f32(0.0);
    let mut i = 0usize;

    // Process 8 at a time with interleaved accumulation (matching C exactly)
    while i + 7 < n {
        let x_f32x4 = vld1q_f32(x.as_ptr().add(i));
        let y01_f32x4 = vld1q_f32(y01.as_ptr().add(i));
        let y02_f32x4 = vld1q_f32(y02.as_ptr().add(i));
        xy01_f32x4 = vfmaq_f32(xy01_f32x4, x_f32x4, y01_f32x4);
        xy02_f32x4 = vfmaq_f32(xy02_f32x4, x_f32x4, y02_f32x4);
        let x_f32x4 = vld1q_f32(x.as_ptr().add(i + 4));
        let y01_f32x4 = vld1q_f32(y01.as_ptr().add(i + 4));
        let y02_f32x4 = vld1q_f32(y02.as_ptr().add(i + 4));
        xy01_f32x4 = vfmaq_f32(xy01_f32x4, x_f32x4, y01_f32x4);
        xy02_f32x4 = vfmaq_f32(xy02_f32x4, x_f32x4, y02_f32x4);
        i += 8;
    }

    // Process remaining 4
    if n - i >= 4 {
        let x_f32x4 = vld1q_f32(x.as_ptr().add(i));
        let y01_f32x4 = vld1q_f32(y01.as_ptr().add(i));
        let y02_f32x4 = vld1q_f32(y02.as_ptr().add(i));
        xy01_f32x4 = vfmaq_f32(xy01_f32x4, x_f32x4, y01_f32x4);
        xy02_f32x4 = vfmaq_f32(xy02_f32x4, x_f32x4, y02_f32x4);
        i += 4;
    }

    // Horizontal sum for xy01
    let xy01_f32x2 = vadd_f32(vget_low_f32(xy01_f32x4), vget_high_f32(xy01_f32x4));
    let xy01_f32x2 = vpadd_f32(xy01_f32x2, xy01_f32x2);
    let mut xy01 = vget_lane_f32(xy01_f32x2, 0);

    // Horizontal sum for xy02
    let xy02_f32x2 = vadd_f32(vget_low_f32(xy02_f32x4), vget_high_f32(xy02_f32x4));
    let xy02_f32x2 = vpadd_f32(xy02_f32x2, xy02_f32x2);
    let mut xy02 = vget_lane_f32(xy02_f32x2, 0);

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
