//! Linear to log scale conversion.
//!
//! Upstream c: `silk/lin2log.c`

use crate::silk::inlines::silk_clz_frac;

///
// Approximation of 128 * log2() (very close inverse of silk_log2lin())
// Convert input to a log scale
/// Upstream c: silk/lin2log.c:silk_lin2log
pub fn silk_lin2log(in_lin: i32) -> i32 {
    let mut lz: i32 = 0;
    let mut frac_q7: i32 = 0;
    silk_clz_frac(in_lin, &mut lz, &mut frac_q7);
    /* Piece-wise parabolic approximation */
    (frac_q7 as i64 + (((frac_q7 * (128 - frac_q7)) as i64 * 179_i64) >> 16)) as i32
        + (((31 - lz) as u32) << 7) as i32
}
