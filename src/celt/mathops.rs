//! Fixed-point and floating-point math operations.
//!
//! Upstream C: `celt/mathops.c`

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

/// Base-2 log approximation using a 4th-degree polynomial with mantissa
/// normalization and correction lookup tables.
///
/// Upstream C: celt/mathops.h:celt_log2 (1.6.1 Remez approximation)
///
/// Note: special cases (denormals, inf, NaN, zero) are not handled — same as C.
#[inline]
#[allow(clippy::excessive_precision)]
pub fn celt_log2(x: f32) -> f32 {
    // Log2 x normalization coefficients: 1 / (1 + 0.125 * index)
    const LOG2_X_NORM_COEFF: [f32; 8] = [
        1.000000000000000000000000000f32,
        8.88888895511627197265625e-01f32,
        8.00000000000000000000000e-01f32,
        7.27272748947143554687500e-01f32,
        6.66666686534881591796875e-01f32,
        6.15384638309478759765625e-01f32,
        5.71428596973419189453125e-01f32,
        5.33333361148834228515625e-01f32,
    ];
    // Log2 y normalization coefficients: log2(1 + 0.125 * index)
    const LOG2_Y_NORM_COEFF: [f32; 8] = [
        0.0000000000000000000000000000f32,
        1.699250042438507080078125e-01f32,
        3.219280838966369628906250e-01f32,
        4.594316184520721435546875e-01f32,
        5.849624872207641601562500e-01f32,
        7.004396915435791015625000e-01f32,
        8.073549270629882812500000e-01f32,
        9.068905711174011230468750e-01f32,
    ];

    let bits = x.to_bits();
    let integer = (bits >> 23) as i32 - 127;
    let bits = bits.wrapping_sub((integer as u32) << 23);

    // Normalize mantissa range from [1, 2] to [1, 1.125], then shift by -1.0625
    let range_idx = ((bits >> 20) & 0x7) as usize;
    let frac = f32::from_bits(bits) * LOG2_X_NORM_COEFF[range_idx] - 1.0625f32;

    // 4th-degree polynomial (Lolremez on [-0.0625, 0.0625])
    const LOG2_COEFF_A0: f32 = 8.74628424644470214843750000e-02;
    const LOG2_COEFF_A1: f32 = 1.357829570770263671875000000000;
    const LOG2_COEFF_A2: f32 = -6.3897705078125000000000000e-01;
    const LOG2_COEFF_A3: f32 = 4.01971250772476196289062500e-01;
    const LOG2_COEFF_A4: f32 = -2.8415444493293762207031250e-01;

    let poly = LOG2_COEFF_A0
        + frac
            * (LOG2_COEFF_A1
                + frac * (LOG2_COEFF_A2 + frac * (LOG2_COEFF_A3 + frac * LOG2_COEFF_A4)));

    integer as f32 + poly + LOG2_Y_NORM_COEFF[range_idx]
}

/// Base-2 exponential approximation using a 5th-degree Remez polynomial
/// with IEEE754 bit manipulation to apply the integer exponent.
///
/// Upstream C: celt/mathops.h:celt_exp2 (1.6.1 Remez approximation)
#[inline]
#[allow(clippy::excessive_precision)]
pub fn celt_exp2(x: f32) -> f32 {
    let integer = x.floor() as i32;
    if integer < -50 {
        return 0.0;
    }
    let frac = x - integer as f32;

    // 5th-degree Remez polynomial on [0, 1]
    const EXP2_COEFF_A0: f32 = 9.999999403953552246093750000000e-01;
    const EXP2_COEFF_A1: f32 = 6.931530833244323730468750000000e-01;
    const EXP2_COEFF_A2: f32 = 2.401536107063293457031250000000e-01;
    const EXP2_COEFF_A3: f32 = 5.582631751894950866699218750000e-02;
    const EXP2_COEFF_A4: f32 = 8.989339694380760192871093750000e-03;
    const EXP2_COEFF_A5: f32 = 1.877576694823801517486572265625e-03;

    let poly = EXP2_COEFF_A0
        + frac
            * (EXP2_COEFF_A1
                + frac
                    * (EXP2_COEFF_A2
                        + frac * (EXP2_COEFF_A3 + frac * (EXP2_COEFF_A4 + frac * EXP2_COEFF_A5))));

    // Combine polynomial result with integer exponent via IEEE754 bit manipulation
    let bits = poly.to_bits();
    let bits = ((bits as i32).wrapping_add((integer as u32 as i32) << 23) as u32) & 0x7fff_ffff;
    f32::from_bits(bits)
}
