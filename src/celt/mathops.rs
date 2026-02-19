//! Fixed-point and floating-point math operations.
//!
//! Upstream C: `celt/mathops.c`
#![allow(clippy::approx_constant, clippy::excessive_precision)]

use crate::silk::macros::EC_CLZ0;

use std::f32::consts::PI;

pub const cA: f32 = 0.43157974f32;
pub const cB: f32 = 0.678_484_f32;
pub const cC: f32 = 0.08595542f32;
pub const cE: f32 = PI / 2_f32;

/// Upstream C: celt/mathops.h:isqrt32
pub fn isqrt32(mut _val: u32) -> u32 {
    let mut g: u32 = 0;
    let mut bshift: i32 = (EC_CLZ0 - _val.leading_zeros() as i32 - 1) >> 1;
    let mut b: u32 = 1u32 << bshift;
    loop {
        let t: u32 = (g << 1).wrapping_add(b) << bshift;
        if t <= _val {
            g = g.wrapping_add(b);
            _val = _val.wrapping_sub(t);
        }
        b >>= 1;
        bshift -= 1;
        if bshift < 0 {
            break;
        }
    }
    g
}

/// Upstream C: celt/mathops.h:fast_atan2f
#[inline]
pub fn fast_atan2f(y: f32, x: f32) -> f32 {
    let x2 = x * x;
    let y2 = y * y;
    if x2 + y2 < 1e-18f32 {
        return 0.0f32;
    }
    if x2 < y2 {
        let den = (y2 + cB * x2) * (y2 + cC * x2);
        -x * y * (y2 + cA * x2) / den + (if y < 0.0f32 { -cE } else { cE })
    } else {
        let den = (x2 + cB * y2) * (x2 + cC * y2);
        x * y * (x2 + cA * y2) / den + (if y < 0.0f32 { -cE } else { cE })
            - (if x * y < 0.0f32 { -cE } else { cE })
    }
}

/// Arctangent approximation normalized to return `(2/pi)*atan(x)` for x in [0, 1].
/// Uses a 15th-order Remez polynomial (odd powers only).
///
/// Upstream C: celt/mathops.h:celt_atan_norm (new in 1.6.1)
#[inline]
#[allow(clippy::excessive_precision)]
pub fn celt_atan_norm(x: f32) -> f32 {
    const ATAN2_2_OVER_PI: f32 = 0.636619772367581f32;
    let x_sq = x * x;

    const ATAN2_COEFF_A03: f32 = -3.3331659436225891113281250000e-01;
    const ATAN2_COEFF_A05: f32 = 1.99627041816711425781250000000e-01;
    const ATAN2_COEFF_A07: f32 = -1.3976582884788513183593750000e-01;
    const ATAN2_COEFF_A09: f32 = 9.79423448443412780761718750000e-02;
    const ATAN2_COEFF_A11: f32 = -5.7773590087890625000000000000e-02;
    const ATAN2_COEFF_A13: f32 = 2.30401363223791122436523437500e-02;
    const ATAN2_COEFF_A15: f32 = -4.3554059229791164398193359375e-03;

    ATAN2_2_OVER_PI
        * (x + x
            * x_sq
            * (ATAN2_COEFF_A03
                + x_sq
                    * (ATAN2_COEFF_A05
                        + x_sq
                            * (ATAN2_COEFF_A07
                                + x_sq
                                    * (ATAN2_COEFF_A09
                                        + x_sq
                                            * (ATAN2_COEFF_A11
                                                + x_sq
                                                    * (ATAN2_COEFF_A13
                                                        + x_sq * ATAN2_COEFF_A15)))))))
}

/// Arctangent of y/x for positive y, x, normalized to return `(2/pi)*atan2(y,x)`.
/// Result is in [0, 1].
///
/// Upstream C: celt/mathops.h:celt_atan2p_norm (new in 1.6.1)
#[inline]
pub fn celt_atan2p_norm(y: f32, x: f32) -> f32 {
    debug_assert!(x >= 0.0 && y >= 0.0);
    if (x * x + y * y) < 1e-18f32 {
        return 0.0;
    }
    if y < x {
        celt_atan_norm(y / x)
    } else {
        1.0f32 - celt_atan_norm(x / y)
    }
}

/// Upstream C: celt/mathops.h:celt_maxabs16
#[inline]
pub fn celt_maxabs16(x: &[f32]) -> f32 {
    let mut maxval: f32 = 0.0;
    let mut minval: f32 = 0.0;
    for &v in x {
        if v > maxval {
            maxval = v;
        }
        if v < minval {
            minval = v;
        }
    }
    if maxval > -minval {
        maxval
    } else {
        -minval
    }
}

// the functions below are analogous to the macros defined in mathops.h header.
// importantly, some of them do conversion to f64 before doing the operation, to make sure the results will match the original implementation.
// it uses the f64 math functions because they are more portable, so we are stuck with them too if we want reproducible bitcode.

/// Upstream C: celt/mathops.h:celt_sqrt
#[inline]
pub fn celt_sqrt(x: f32) -> f32 {
    (x as f64).sqrt() as f32
}

/// Upstream C: celt/mathops.h:celt_rsqrt
#[inline]
pub fn celt_rsqrt(x: f32) -> f32 {
    1.0f32 / celt_sqrt(x)
}

/// Upstream C: celt/mathops.h:celt_rsqrt_norm
#[inline]
pub fn celt_rsqrt_norm(x: f32) -> f32 {
    celt_rsqrt(x)
}

/// Upstream C: celt/mathops.h:celt_cos_norm
///
/// C definition: `((float)cos((.5f*PI)*(x)))` where PI is a double literal.
/// In C, `.5f * PI` promotes to double (since PI is 3.1415926535897931, a double),
/// then `* x` also promotes x to double. So the entire argument is computed in f64.
#[inline]
pub fn celt_cos_norm(x: f32) -> f32 {
    (0.5 * std::f64::consts::PI * x as f64).cos() as f32
}

/// Polynomial approximation of cos(PI/2 * x) using only even-powered terms.
///
/// This is NOT the same as `celt_cos_norm` — it uses a Lolremez polynomial
/// approximation that must match the C reference exactly for QEXT bit-exactness.
///
/// Upstream C: celt/mathops.h:celt_cos_norm2
#[cfg(any(feature = "qext", feature = "osce"))]
#[inline]
#[allow(clippy::excessive_precision)]
pub fn celt_cos_norm2(x: f32) -> f32 {
    const COS_COEFF_A0: f32 = 9.999999403953552246093750000000e-01;
    const COS_COEFF_A2: f32 = -1.233698248863220214843750000000000;
    const COS_COEFF_A4: f32 = 2.536507546901702880859375000000e-01;
    const COS_COEFF_A6: f32 = -2.08106283098459243774414062500e-02;
    const COS_COEFF_A8: f32 = 8.581906440667808055877685546875e-04;

    // Restrict x to [-1, 3]
    let mut x = x - 4.0 * (0.25 * (x + 1.0)).floor();
    // Negative sign for [1, 3]
    let output_sign: f32 = if x > 1.0 { -1.0 } else { 1.0 };
    // Restrict to [-1, 1]
    if x > 1.0 {
        x -= 2.0;
    }
    let x_norm_sq = x * x;
    output_sign
        * (COS_COEFF_A0
            + x_norm_sq
                * (COS_COEFF_A2
                    + x_norm_sq
                        * (COS_COEFF_A4 + x_norm_sq * (COS_COEFF_A6 + x_norm_sq * COS_COEFF_A8))))
}

/// Upstream C: celt/mathops.h:celt_log
///
/// C 1.6.1: `celt_log2(x) * 0.6931471805599453f`
#[inline]
pub fn celt_log(x: f32) -> f32 {
    celt_log2(x) * 0.6931471805599453f32
}

/// Upstream C: celt/mathops.h:celt_log10
#[inline]
pub fn celt_log10(x: f32) -> f32 {
    (x as f64).log10() as f32
}

/// Upstream C: celt/mathops.h:celt_log2
///
/// We match the default non-FLOAT_APPROX float path:
/// `((float)(1.442695040888963387*log(x)))`.
#[inline]
pub fn celt_log2(x: f32) -> f32 {
    (1.442695040888963387_f64 * (x as f64).ln()) as f32
}

/// Upstream C: celt/mathops.h:celt_exp2
///
/// We match the default non-FLOAT_APPROX float path:
/// `((float)exp(0.6931471805599453094*(x)))`.
#[inline]
pub fn celt_exp2(x: f32) -> f32 {
    (0.6931471805599453094_f64 * x as f64).exp() as f32
}
