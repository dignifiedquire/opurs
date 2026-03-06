//! LPC coefficient stabilization.
//!
//! Upstream c: `silk/LPC_fit.c`

use crate::silk::bwexpander_32::silk_bwexpander_32;
use crate::silk::sigproc_fix::{silk_rshift_round, silk_sat16, SILK_FIX_CONST};

///
/// Convert int32 coefficients to int16 coefs and make sure there's no wrap-around
///
/// ```text
/// a_qout   O     Output signal
/// a_qin    I/O   Input signal
/// qout     I     Input Q domain
/// qin      I     Input Q domain
/// d        I     Filter Order
/// ```
/// Upstream c: silk/LPC_fit.c:silk_LPC_fit
#[inline]
pub fn silk_lpc_fit(a_qout: &mut [i16], a_qin: &mut [i32], qout: i32, qin: i32) {
    let d = a_qout.len();
    assert_eq!(a_qin.len(), d);

    /* Limit the maximum absolute value of the prediction coefficients, so that they'll fit in int16 */
    let mut _i = 0;
    while _i < 10 {
        /* Find maximum absolute value and its index */
        let mut maxabs = 0;
        let mut idx = 0;
        let mut k = 0;
        while k < d {
            let absval = a_qin[k].abs();
            if absval > maxabs {
                maxabs = absval;
                idx = k;
            }
            k += 1;
        }
        maxabs = silk_rshift_round(maxabs, qin - qout);

        if maxabs > i16::MAX as i32 {
            /* Reduce magnitude of prediction coefficients */
            maxabs = std::cmp::min(maxabs, 163838); /* ( SILK_INT32_MAX >> 14 ) + SILK_INT16_MAX = 163838 */
            let chirp_q16 = SILK_FIX_CONST!(0.999f64, 16)
                - ((maxabs - i16::MAX as i32) << 14) / ((maxabs * (idx as i32 + 1)) >> 2);
            silk_bwexpander_32(a_qin, chirp_q16);
        } else {
            break;
        }

        _i += 1;
    }

    if _i == 10 {
        /* Reached the last iteration, clip the coefficients */
        for (out, input) in a_qout.iter_mut().zip(a_qin.iter_mut()) {
            *out = silk_sat16(silk_rshift_round(*input, qin - qout)) as i16;
            *input = (*out as i32) << (qin - qout);
        }
    } else {
        for (out, input) in a_qout.iter_mut().zip(a_qin.iter()) {
            *out = silk_rshift_round(*input, qin - qout) as i16;
        }
    };
}
