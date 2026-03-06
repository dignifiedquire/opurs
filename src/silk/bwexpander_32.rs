//! Bandwidth expansion (32-bit precision).
//!
//! Upstream c: `silk/bwexpander_32.c`

use crate::silk::macros::silk_smulww;
use crate::silk::sigproc_fix::silk_rshift_round;

///
/// Chirp (bandwidth expand) LP AR filter
///
/// ```text
/// ar          I/O   AR filter to be expanded (without leading 1)
/// d           I     Length of ar
/// chirp_q16   I     Chirp factor in Q16
/// ```
/// Upstream c: silk/bwexpander_32.c:silk_bwexpander_32
pub fn silk_bwexpander_32(ar: &mut [i32], mut chirp_q16: i32) {
    let d = ar.len();

    let chirp_minus_one_q16: i32 = chirp_q16 - 65536;

    for ar in ar.iter_mut().take(d - 1) {
        *ar = silk_smulww(chirp_q16, *ar);
        chirp_q16 += silk_rshift_round(chirp_q16 * chirp_minus_one_q16, 16)
    }

    ar[d - 1] = silk_smulww(chirp_q16, ar[d - 1]);
}
