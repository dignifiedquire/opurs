//! Inline function equivalents.
//!
//! Upstream C: `silk/Inlines.h`

use crate::silk::macros::silk_clz32;
use crate::silk::SigProc_FIX::silk_ror32;

///
/// get number of leading zeros and fractional part (the bits right after the leading one
/// Upstream C: silk/Inlines.h:silk_CLZ_FRAC
#[inline]
pub fn silk_clz_frac(in_0: i32, lz: &mut i32, frac_Q7: &mut i32) {
    let lzeros: i32 = silk_clz32(in_0);
    *lz = lzeros;
    *frac_Q7 = silk_ror32(in_0, 24 - lzeros) & 0x7f;
}

///
///  Approximation of square root
///  Accuracy: < +/- 10%  for output values > 15
///            < +/- 2.5% for output values > 120
/// Upstream C: silk/Inlines.h:silk_SQRT_APPROX
#[inline]
pub fn silk_sqrt_approx(x: i32) -> i32 {
    let mut y: i32;
    let mut lz: i32 = 0;
    let mut frac_Q7: i32 = 0;
    if x <= 0 {
        return 0;
    }
    silk_clz_frac(x, &mut lz, &mut frac_Q7);
    if lz & 1 != 0 {
        y = 32768;
    } else {
        y = 46214;
    }
    y >>= lz >> 1;
    y = (y as i64 + ((y as i64 * (213 * frac_Q7 as i16 as i32) as i16 as i64) >> 16)) as i32;
    y
}

///
/// Divide two int32 values and return result as int32 in a given Q-domain
/// Upstream C: silk/Inlines.h:silk_DIV32_varQ
#[inline]
pub fn silk_div32_varq(a32: i32, b32: i32, Qres: i32) -> i32 {
    let mut a32_nrm: i32;

    let mut result: i32;
    let a_headrm: i32 = silk_clz32(if a32 > 0 { a32 } else { -a32 }) - 1;
    a32_nrm = ((a32 as u32) << a_headrm) as i32;
    let b_headrm: i32 = silk_clz32(if b32 > 0 { b32 } else { -b32 }) - 1;
    let b32_nrm: i32 = ((b32 as u32) << b_headrm) as i32;
    let b32_inv: i32 = (0x7fffffff >> 2) / (b32_nrm >> 16);
    result = ((a32_nrm as i64 * b32_inv as i16 as i64) >> 16) as i32;
    a32_nrm = (a32_nrm as u32)
        .wrapping_sub(((((b32_nrm as i64 * result as i64) >> 32) as i32 as u32) << 3) as i32 as u32)
        as i32;
    result = (result as i64 + ((a32_nrm as i64 * b32_inv as i16 as i64) >> 16)) as i32;
    let lshift: i32 = 29 + a_headrm - b_headrm - Qres;
    if lshift < 0 {
        (((if 0x80000000_u32 as i32 >> -lshift > 0x7fffffff >> -lshift {
            if result > 0x80000000_u32 as i32 >> -lshift {
                0x80000000_u32 as i32 >> -lshift
            } else if result < 0x7fffffff >> -lshift {
                0x7fffffff >> -lshift
            } else {
                result
            }
        } else if result > 0x7fffffff >> -lshift {
            0x7fffffff >> -lshift
        } else if result < 0x80000000_u32 as i32 >> -lshift {
            0x80000000_u32 as i32 >> -lshift
        } else {
            result
        }) as u32)
            << -lshift) as i32
    } else if lshift < 32 {
        result >> lshift
    } else {
        0
    }
}

///
/// Invert int32 value and return result as int32 in a given Q-domain
///
/// returns a good approximation of "(1 << Qres) / b32"
/// Upstream C: silk/Inlines.h:silk_INVERSE32_varQ
#[inline]
pub fn silk_inverse32_varq(b32: i32, Qres: i32) -> i32 {
    let mut result: i32;
    let b_headrm: i32 = silk_clz32(if b32 > 0 { b32 } else { -b32 }) - 1;
    let b32_nrm: i32 = ((b32 as u32) << b_headrm) as i32;
    let b32_inv: i32 = (0x7fffffff >> 2) / (b32_nrm >> 16);
    result = ((b32_inv as u32) << 16) as i32;
    let err_Q32: i32 = (((((1) << 29) - ((b32_nrm as i64 * b32_inv as i16 as i64) >> 16) as i32)
        as u32)
        << 3) as i32;
    result = (result as i64 + ((err_Q32 as i64 * b32_inv as i64) >> 16)) as i32;
    let lshift: i32 = 61 - b_headrm - Qres;
    if lshift <= 0 {
        (((if 0x80000000_u32 as i32 >> -lshift > 0x7fffffff >> -lshift {
            if result > 0x80000000_u32 as i32 >> -lshift {
                0x80000000_u32 as i32 >> -lshift
            } else if result < 0x7fffffff >> -lshift {
                0x7fffffff >> -lshift
            } else {
                result
            }
        } else if result > 0x7fffffff >> -lshift {
            0x7fffffff >> -lshift
        } else if result < 0x80000000_u32 as i32 >> -lshift {
            0x80000000_u32 as i32 >> -lshift
        } else {
            result
        }) as u32)
            << -lshift) as i32
    } else if lshift < 32 {
        result >> lshift
    } else {
        0
    }
}

#[allow(unused_imports)]
pub use silk_clz_frac as silk_CLZ_FRAC;
#[allow(unused_imports)]
pub use silk_div32_varq as silk_DIV32_varQ;
#[allow(unused_imports)]
pub use silk_inverse32_varq as silk_INVERSE32_varQ;
#[allow(unused_imports)]
pub use silk_sqrt_approx as silk_SQRT_APPROX;
