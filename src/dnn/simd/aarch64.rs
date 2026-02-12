//! aarch64 NEON SIMD implementations for DNN vector primitives.
//!
//! NEON is always available on aarch64, so these are selected at compile time
//! (no runtime detection needed).
//!
//! Port of `dnn/vec_neon.h` from libopus 1.5.2.

use core::arch::aarch64::*;

// =========================================================================
// Activation helpers
// =========================================================================

/// NEON fast exp approximation (4-wide).
/// Port of `vec_neon.h:exp4_approx`.
#[target_feature(enable = "neon")]
unsafe fn exp4_approx(x: float32x4_t) -> float32x4_t {
    let x = vmaxq_f32(vminq_f32(x, vdupq_n_f32(88.0)), vdupq_n_f32(-88.0));

    // exp(x) = exp2(x/log(2)); add 127 for the exponent later
    // C remaps vmlaq_f32 → vfmaq_f32 when __ARM_FEATURE_FMA is defined.
    // We use vfmaq_f32 explicitly to match.
    let x = vfmaq_f32(vdupq_n_f32(127.0), x, vdupq_n_f32(1.44269504));

    // Split into integer and fractional parts
    let i = vcvtq_s32_f32(x);
    let xf = vcvtq_f32_s32(i);
    let x = vsubq_f32(x, xf);

    let k0 = vdupq_n_f32(0.99992522);
    let k1 = vdupq_n_f32(0.69583354);
    let k2 = vdupq_n_f32(0.22606716);
    let k3 = vdupq_n_f32(0.078024523);
    let y = vfmaq_f32(k0, x, vfmaq_f32(k1, x, vfmaq_f32(k2, k3, x)));

    // Compute 2^i by shifting integer into exponent bits
    let exponent = vreinterpretq_f32_s32(vshlq_n_s32::<23>(i));
    vmulq_f32(y, exponent)
}

/// NEON fast tanh approximation (4-wide).
/// Port of `vec_neon.h:tanh4_approx`.
#[target_feature(enable = "neon")]
unsafe fn tanh4_approx(x: float32x4_t) -> float32x4_t {
    let n0 = vdupq_n_f32(952.52801514);
    let n1 = vdupq_n_f32(96.39235687);
    let n2 = vdupq_n_f32(0.60863042);
    let d0 = vdupq_n_f32(952.72399902);
    let d1 = vdupq_n_f32(413.36801147);
    let d2 = vdupq_n_f32(11.88600922);
    let max_out = vdupq_n_f32(1.0);
    let min_out = vdupq_n_f32(-1.0);

    let x2 = vmulq_f32(x, x);
    let num = vfmaq_f32(n0, x2, vfmaq_f32(n1, n2, x2));
    let den = vfmaq_f32(d0, x2, vfmaq_f32(d1, d2, x2));
    let num = vmulq_f32(num, x);
    let den = vrecpeq_f32(den);
    let num = vmulq_f32(num, den);
    vmaxq_f32(min_out, vminq_f32(max_out, num))
}

/// NEON fast sigmoid approximation (4-wide).
/// Port of `vec_neon.h:sigmoid4_approx`.
#[target_feature(enable = "neon")]
unsafe fn sigmoid4_approx(x: float32x4_t) -> float32x4_t {
    let n0 = vdupq_n_f32(238.13200378);
    let n1 = vdupq_n_f32(6.02452230);
    let n2 = vdupq_n_f32(0.00950985);
    let d0 = vdupq_n_f32(952.72399902);
    let d1 = vdupq_n_f32(103.34200287);
    let d2 = vdupq_n_f32(0.74287558);
    let half = vdupq_n_f32(0.5);
    let max_out = vdupq_n_f32(1.0);
    let min_out = vdupq_n_f32(0.0);

    let x2 = vmulq_f32(x, x);
    let num = vfmaq_f32(n0, x2, vfmaq_f32(n1, n2, x2));
    let den = vfmaq_f32(d0, x2, vfmaq_f32(d1, d2, x2));
    let num = vmulq_f32(num, x);
    let den = vrecpeq_f32(den);
    let num = vfmaq_f32(half, num, den);
    vmaxq_f32(min_out, vminq_f32(max_out, num))
}

// =========================================================================
// Scalar wrappers via NEON (matching C vec_neon.h behavior)
// =========================================================================
// On aarch64, C's vec_neon.h redefines `lpcnet_exp`, `tanh_approx`, and
// `sigmoid_approx` to broadcast the scalar into a NEON register, call the
// 4-wide approximation (with FMA + vrecpe), and extract lane 0. This gives
// different results than the scalar code due to FMA and approximate reciprocal.

/// Scalar lpcnet_exp via NEON exp4_approx.
/// Port of `vec_neon.h:lpcnet_exp` (NEON override).
///
/// # Safety
/// Requires aarch64 NEON (always available on aarch64).
#[target_feature(enable = "neon")]
pub unsafe fn lpcnet_exp_neon(x: f32) -> f32 {
    let xv = vdupq_n_f32(x);
    let yv = exp4_approx(xv);
    vgetq_lane_f32(yv, 0)
}

/// Scalar tanh_approx via NEON tanh4_approx.
/// Port of `vec_neon.h:tanh_approx` (NEON override).
///
/// # Safety
/// Requires aarch64 NEON (always available on aarch64).
#[target_feature(enable = "neon")]
pub unsafe fn tanh_approx_neon(x: f32) -> f32 {
    let xv = vdupq_n_f32(x);
    let yv = tanh4_approx(xv);
    vgetq_lane_f32(yv, 0)
}

/// Scalar sigmoid_approx via NEON sigmoid4_approx.
/// Port of `vec_neon.h:sigmoid_approx` (NEON override).
///
/// # Safety
/// Requires aarch64 NEON (always available on aarch64).
#[target_feature(enable = "neon")]
pub unsafe fn sigmoid_approx_neon(x: f32) -> f32 {
    let xv = vdupq_n_f32(x);
    let yv = sigmoid4_approx(xv);
    vgetq_lane_f32(yv, 0)
}

// =========================================================================
// Batch activation functions
// =========================================================================

/// NEON batch tanh approximation.
/// Port of `vec_neon.h:vec_tanh`.
///
/// # Safety
/// Requires aarch64 NEON (always available on aarch64).
#[target_feature(enable = "neon")]
pub unsafe fn vec_tanh_neon(y: &mut [f32], x: &[f32]) {
    let n = x.len().min(y.len());
    let mut i = 0;
    while i + 4 <= n {
        let xv = vld1q_f32(x.as_ptr().add(i));
        let yv = tanh4_approx(xv);
        vst1q_f32(y.as_mut_ptr().add(i), yv);
        i += 4;
    }
    // Scalar tail: C uses lpcnet_exp (NEON version) based formula
    while i < n {
        let ex2 = lpcnet_exp_neon(2.0 * x[i]);
        y[i] = (ex2 - 1.0) / (ex2 + 1.0);
        i += 1;
    }
}

/// NEON batch sigmoid approximation.
/// Port of `vec_neon.h:vec_sigmoid`.
///
/// # Safety
/// Requires aarch64 NEON (always available on aarch64).
#[target_feature(enable = "neon")]
pub unsafe fn vec_sigmoid_neon(y: &mut [f32], x: &[f32]) {
    let n = x.len().min(y.len());
    let mut i = 0;
    while i + 4 <= n {
        let xv = vld1q_f32(x.as_ptr().add(i));
        let yv = sigmoid4_approx(xv);
        vst1q_f32(y.as_mut_ptr().add(i), yv);
        i += 4;
    }
    // Scalar tail: C uses lpcnet_exp (NEON version) based formula
    while i < n {
        let ex = lpcnet_exp_neon(x[i]);
        y[i] = ex / (ex + 1.0);
        i += 1;
    }
}

/// NEON batch softmax (unnormalized exp).
/// Port of `vec_neon.h:softmax`.
///
/// # Safety
/// Requires aarch64 NEON (always available on aarch64).
#[target_feature(enable = "neon")]
pub unsafe fn softmax_neon(y: &mut [f32], x: &[f32]) {
    let n = x.len().min(y.len());
    let mut i = 0;
    while i + 4 <= n {
        let xv = vld1q_f32(x.as_ptr().add(i));
        let yv = exp4_approx(xv);
        vst1q_f32(y.as_mut_ptr().add(i), yv);
        i += 4;
    }
    // Scalar tail: use NEON lpcnet_exp for consistency
    while i < n {
        y[i] = lpcnet_exp_neon(x[i]);
        i += 1;
    }
}

// =========================================================================
// Dense float GEMV
// =========================================================================

/// NEON dense float matrix-vector multiply: out = weights^T * x.
/// Port of `vec_neon.h:sgemv` (dispatches to sgemv16x1/sgemv8x1).
///
/// # Safety
/// Requires aarch64 NEON (always available on aarch64).
#[target_feature(enable = "neon")]
pub unsafe fn sgemv_neon(
    out: &mut [f32],
    weights: &[f32],
    rows: usize,
    cols: usize,
    col_stride: usize,
    x: &[f32],
) {
    if rows & 0xf == 0 {
        sgemv16x1_neon(out, weights, rows, cols, col_stride, x);
    } else if rows & 0x7 == 0 {
        sgemv8x1_neon(out, weights, rows, cols, col_stride, x);
    } else {
        // Generic scalar fallback
        for i in 0..rows {
            out[i] = 0.0;
            for j in 0..cols {
                out[i] += weights[j * col_stride + i] * x[j];
            }
        }
    }
}

/// NEON sgemv for 16-aligned rows.
/// Port of `vec_neon.h:sgemv16x1`.
#[target_feature(enable = "neon")]
unsafe fn sgemv16x1_neon(
    out: &mut [f32],
    weights: &[f32],
    rows: usize,
    cols: usize,
    col_stride: usize,
    x: &[f32],
) {
    let mut i = 0;
    while i < rows {
        let mut y0_3 = vdupq_n_f32(0.0);
        let mut y4_7 = vdupq_n_f32(0.0);
        let mut y8_11 = vdupq_n_f32(0.0);
        let mut y12_15 = vdupq_n_f32(0.0);

        for j in 0..cols {
            let w = weights.as_ptr().add(j * col_stride + i);
            let xj = vld1q_dup_f32(&x[j]);
            y0_3 = vfmaq_f32(y0_3, vld1q_f32(w), xj);
            y4_7 = vfmaq_f32(y4_7, vld1q_f32(w.add(4)), xj);
            y8_11 = vfmaq_f32(y8_11, vld1q_f32(w.add(8)), xj);
            y12_15 = vfmaq_f32(y12_15, vld1q_f32(w.add(12)), xj);
        }

        let y = out.as_mut_ptr().add(i);
        vst1q_f32(y, y0_3);
        vst1q_f32(y.add(4), y4_7);
        vst1q_f32(y.add(8), y8_11);
        vst1q_f32(y.add(12), y12_15);

        i += 16;
    }
}

/// NEON sgemv for 8-aligned rows.
/// Port of `vec_neon.h:sgemv8x1`.
#[target_feature(enable = "neon")]
unsafe fn sgemv8x1_neon(
    out: &mut [f32],
    weights: &[f32],
    rows: usize,
    cols: usize,
    col_stride: usize,
    x: &[f32],
) {
    let mut i = 0;
    while i < rows {
        let mut y0_3 = vdupq_n_f32(0.0);
        let mut y4_7 = vdupq_n_f32(0.0);

        for j in 0..cols {
            let w = weights.as_ptr().add(j * col_stride + i);
            let xj = vld1q_dup_f32(&x[j]);
            y0_3 = vfmaq_f32(y0_3, vld1q_f32(w), xj);
            y4_7 = vfmaq_f32(y4_7, vld1q_f32(w.add(4)), xj);
        }

        let y = out.as_mut_ptr().add(i);
        vst1q_f32(y, y0_3);
        vst1q_f32(y.add(4), y4_7);

        i += 8;
    }
}

// =========================================================================
// Sparse float GEMV
// =========================================================================

/// NEON sparse float matrix-vector multiply (8x4 block sparse).
/// Port of `vec_neon.h:sparse_sgemv8x4` (scalar version — NEON optimized).
///
/// # Safety
/// Requires aarch64 NEON (always available on aarch64).
#[target_feature(enable = "neon")]
pub unsafe fn sparse_sgemv8x4_neon(
    out: &mut [f32],
    w: &[f32],
    idx: &[i32],
    rows: usize,
    x: &[f32],
) {
    let mut w_pos = 0;
    let mut idx_pos = 0;

    for i in (0..rows).step_by(8) {
        let cols = idx[idx_pos] as usize;
        idx_pos += 1;
        let mut y0_3 = vdupq_n_f32(0.0);
        let mut y4_7 = vdupq_n_f32(0.0);

        for _j in 0..cols {
            let pos = idx[idx_pos] as usize;
            idx_pos += 1;

            let wp = w.as_ptr().add(w_pos);

            let xj0 = vld1q_dup_f32(&x[pos]);
            y0_3 = vfmaq_f32(y0_3, vld1q_f32(wp), xj0);
            y4_7 = vfmaq_f32(y4_7, vld1q_f32(wp.add(4)), xj0);

            let xj1 = vld1q_dup_f32(&x[pos + 1]);
            y0_3 = vfmaq_f32(y0_3, vld1q_f32(wp.add(8)), xj1);
            y4_7 = vfmaq_f32(y4_7, vld1q_f32(wp.add(12)), xj1);

            let xj2 = vld1q_dup_f32(&x[pos + 2]);
            y0_3 = vfmaq_f32(y0_3, vld1q_f32(wp.add(16)), xj2);
            y4_7 = vfmaq_f32(y4_7, vld1q_f32(wp.add(20)), xj2);

            let xj3 = vld1q_dup_f32(&x[pos + 3]);
            y0_3 = vfmaq_f32(y0_3, vld1q_f32(wp.add(24)), xj3);
            y4_7 = vfmaq_f32(y4_7, vld1q_f32(wp.add(28)), xj3);

            w_pos += 32;
        }

        let y = out.as_mut_ptr().add(i);
        vst1q_f32(y, y0_3);
        vst1q_f32(y.add(4), y4_7);
    }
}

// =========================================================================
// Dense int8 GEMV
// =========================================================================

/// NEON dotprod emulation (no DOTPROD extension required).
/// Computes dot product of int8x16 vectors and accumulates into int32x4.
/// Port of `vec_neon.h:vdotprod` (non-DOTPROD path).
#[inline]
#[target_feature(enable = "neon")]
unsafe fn vdotprod(acc: int32x4_t, a: int8x16_t, b: int8x16_t) -> int32x4_t {
    vpadalq_s16(
        acc,
        vpaddq_s16(
            vmull_s8(vget_low_s8(a), vget_low_s8(b)),
            vmull_high_s8(a, b),
        ),
    )
}

/// NEON dense int8 matrix-vector multiply (8x4 blocking).
/// Port of `vec_neon.h:cgemv8x4`.
///
/// Uses signed i8 quantization: round(127 * x).
///
/// # Safety
/// Requires aarch64 NEON (always available on aarch64).
#[target_feature(enable = "neon")]
pub unsafe fn cgemv8x4_neon(
    out: &mut [f32],
    w: &[i8],
    scale: &[f32],
    rows: usize,
    cols: usize,
    _x: &[f32],
) {
    const MAX_INPUTS: usize = 2048;
    let mut x = [0i8; MAX_INPUTS];
    let const127 = vdupq_n_f32(127.0);
    let mut qi = 0;
    while qi + 8 <= cols {
        let xi0 = vcvtnq_s32_f32(vmulq_f32(const127, vld1q_f32(_x.as_ptr().add(qi))));
        let xi4 = vcvtnq_s32_f32(vmulq_f32(const127, vld1q_f32(_x.as_ptr().add(qi + 4))));
        let x_short = vcombine_s16(vmovn_s32(xi0), vmovn_s32(xi4));
        vst1_s8(x.as_mut_ptr().add(qi), vmovn_s16(x_short));
        qi += 8;
    }
    while qi < cols {
        x[qi] = (0.5 + 127.0 * _x[qi]).floor() as i8;
        qi += 1;
    }

    let mut w_pos = 0;
    for i in (0..rows).step_by(8) {
        let mut acc0 = vdupq_n_s32(0);
        let mut acc1 = vdupq_n_s32(0);
        let mut acc2 = vdupq_n_s32(0);
        let mut acc3 = vdupq_n_s32(0);
        let mut j = 0;

        // Unrolled by 2
        while j + 8 <= cols {
            let vx0: int8x16_t =
                vreinterpretq_s8_s32(vld1q_dup_s32(x.as_ptr().add(j) as *const i32));
            let vw0 = vld1q_s8(w.as_ptr().add(w_pos));
            let vw1 = vld1q_s8(w.as_ptr().add(w_pos + 16));
            acc0 = vdotprod(acc0, vw0, vx0);
            acc1 = vdotprod(acc1, vw1, vx0);

            let vx1: int8x16_t =
                vreinterpretq_s8_s32(vld1q_dup_s32(x.as_ptr().add(j + 4) as *const i32));
            let vw2 = vld1q_s8(w.as_ptr().add(w_pos + 32));
            let vw3 = vld1q_s8(w.as_ptr().add(w_pos + 48));
            acc2 = vdotprod(acc2, vw2, vx1);
            acc3 = vdotprod(acc3, vw3, vx1);

            w_pos += 64;
            j += 8;
        }

        acc0 = vaddq_s32(acc0, acc2);
        acc1 = vaddq_s32(acc1, acc3);

        while j < cols {
            let vx: int8x16_t =
                vreinterpretq_s8_s32(vld1q_dup_s32(x.as_ptr().add(j) as *const i32));
            let vw0 = vld1q_s8(w.as_ptr().add(w_pos));
            let vw1 = vld1q_s8(w.as_ptr().add(w_pos + 16));
            acc0 = vdotprod(acc0, vw0, vx);
            acc1 = vdotprod(acc1, vw1, vx);
            w_pos += 32;
            j += 4;
        }

        vst1q_f32(
            out.as_mut_ptr().add(i),
            vmulq_f32(vld1q_f32(scale.as_ptr().add(i)), vcvtq_f32_s32(acc0)),
        );
        vst1q_f32(
            out.as_mut_ptr().add(i + 4),
            vmulq_f32(vld1q_f32(scale.as_ptr().add(i + 4)), vcvtq_f32_s32(acc1)),
        );
    }
}

// =========================================================================
// Sparse int8 GEMV
// =========================================================================

/// NEON sparse int8 matrix-vector multiply (8x4 block sparse).
/// Port of `vec_neon.h:sparse_cgemv8x4`.
///
/// # Safety
/// Requires aarch64 NEON (always available on aarch64).
#[target_feature(enable = "neon")]
pub unsafe fn sparse_cgemv8x4_neon(
    out: &mut [f32],
    w: &[i8],
    idx: &[i32],
    scale: &[f32],
    rows: usize,
    cols: usize,
    _x: &[f32],
) {
    const MAX_INPUTS: usize = 2048;
    let mut x = [0i8; MAX_INPUTS];
    let const127 = vdupq_n_f32(127.0);
    let mut qi = 0;
    while qi + 8 <= cols {
        let xi0 = vcvtnq_s32_f32(vmulq_f32(const127, vld1q_f32(_x.as_ptr().add(qi))));
        let xi4 = vcvtnq_s32_f32(vmulq_f32(const127, vld1q_f32(_x.as_ptr().add(qi + 4))));
        let x_short = vcombine_s16(vmovn_s32(xi0), vmovn_s32(xi4));
        vst1_s8(x.as_mut_ptr().add(qi), vmovn_s16(x_short));
        qi += 8;
    }
    while qi < cols {
        x[qi] = (0.5 + 127.0 * _x[qi]).floor() as i8;
        qi += 1;
    }

    let mut w_pos = 0;
    let mut idx_pos = 0;

    for i in (0..rows).step_by(8) {
        let colblocks = idx[idx_pos] as usize;
        idx_pos += 1;
        let mut acc0 = vdupq_n_s32(0);
        let mut acc1 = vdupq_n_s32(0);

        for _j in 0..colblocks {
            let pos = idx[idx_pos] as usize;
            idx_pos += 1;

            let vx: int8x16_t =
                vreinterpretq_s8_s32(vld1q_dup_s32(x.as_ptr().add(pos) as *const i32));
            let vw0 = vld1q_s8(w.as_ptr().add(w_pos));
            let vw1 = vld1q_s8(w.as_ptr().add(w_pos + 16));
            acc0 = vdotprod(acc0, vw0, vx);
            acc1 = vdotprod(acc1, vw1, vx);
            w_pos += 32;
        }

        vst1q_f32(
            out.as_mut_ptr().add(i),
            vmulq_f32(vld1q_f32(scale.as_ptr().add(i)), vcvtq_f32_s32(acc0)),
        );
        vst1q_f32(
            out.as_mut_ptr().add(i + 4),
            vmulq_f32(vld1q_f32(scale.as_ptr().add(i + 4)), vcvtq_f32_s32(acc1)),
        );
    }
}
