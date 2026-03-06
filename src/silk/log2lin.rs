//! Log to linear scale conversion.
//!
//! Upstream c: `silk/log2lin.c`

use crate::silk::typedefs::SILK_INT32_MAX;

///
/// Approximation of 2^() (very close inverse of silk_lin2log()) */
/// Convert input to a linear scale
/// Upstream c: silk/log2lin.c:silk_log2lin
pub fn silk_log2lin(in_log_q7: i32) -> i32 {
    let mut out: i32;

    if in_log_q7 < 0 {
        return 0;
    } else if in_log_q7 >= 3967 {
        return SILK_INT32_MAX;
    }
    out = (1) << (in_log_q7 >> 7);
    let frac_q7: i32 = in_log_q7 & 0x7f;
    if in_log_q7 < 2048 {
        out = out
            + ((out
                * (frac_q7 as i64
                    + (((frac_q7 as i16 as i32 * (128 - frac_q7) as i16 as i32) as i64
                        * -174_i16 as i64)
                        >> 16)) as i32)
                >> 7);
    } else {
        out = out
            + (out >> 7)
                * (frac_q7 as i64
                    + (((frac_q7 as i16 as i32 * (128 - frac_q7) as i16 as i32) as i64
                        * -174_i16 as i64)
                        >> 16)) as i32;
    }
    out
}
