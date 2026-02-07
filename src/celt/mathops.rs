use crate::silk::macros::EC_CLZ0;

use std::f32::consts::PI;

pub const cA: f32 = 0.43157974f32;
pub const cB: f32 = 0.67848403f32;
pub const cC: f32 = 0.08595542f32;
pub const cE: f32 = PI / 2 as f32;

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
#[inline]
pub fn celt_cos_norm(x: f32) -> f32 {
    ((0.5f32 * PI * x) as f64).cos() as f32
}

/// Upstream C: celt/mathops.h:celt_log
#[inline]
pub fn celt_log(x: f32) -> f32 {
    (x as f64).ln() as f32
}

/// Upstream C: celt/mathops.h:celt_log10
#[inline]
pub fn celt_log10(x: f32) -> f32 {
    (x as f64).log10() as f32
}

/// Upstream C: celt/mathops.h:celt_log2
#[inline]
pub fn celt_log2(f: f32) -> f32 {
    (std::f64::consts::LOG2_E * (f as f64).ln()) as f32
}

/// Upstream C: celt/mathops.h:celt_exp2
#[inline]
pub fn celt_exp2(f: f32) -> f32 {
    (std::f64::consts::LN_2 * f as f64).exp() as f32
}
