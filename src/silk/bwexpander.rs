//! Bandwidth expansion (chirp filtering).
//!
//! Upstream c: `silk/bwexpander.c`

use crate::silk::sigproc_fix::silk_rshift_round;

///
/// Chirp (bandwidth expand) LP AR filter
///
/// `ar`:        I/O  AR filter to be expanded (without leading 1)
/// `d`:         I    number of parameters in the AR filter
/// `chirp_q16`: I    chirp factor (typically in the range 0 to 1)
/// Upstream c: silk/bwexpander.c:silk_bwexpander
pub fn silk_bwexpander(ar: &mut [i16], mut chirp_q16: i32) {
    let d = ar.len();

    let chirp_minus_one_q16: i32 = chirp_q16 - 65536;

    /* NB: Dont use silk_smulwb, instead of silk_rshift_round( silk_MUL(), 16 ), below.  */
    /* Bias in silk_smulwb can lead to unstable filters                                */
    for a in ar[..d - 1].iter_mut() {
        *a = silk_rshift_round(chirp_q16 * *a as i32, 16) as i16;
        chirp_q16 += silk_rshift_round(chirp_q16 * chirp_minus_one_q16, 16);
    }

    ar[d - 1] = silk_rshift_round(chirp_q16 * ar[d - 1] as i32, 16) as i16;
}
