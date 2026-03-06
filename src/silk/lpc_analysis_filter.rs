//! LPC analysis filter.
//!
//! Upstream c: `silk/LPC_analysis_filter.c`

use crate::silk::sigproc_fix::{silk_rshift_round, silk_sat16};

///
/// LPC analysis filter
///
/// NB! State is kept internally and the
/// filter always starts with zero state
/// first d output samples are set to zero
///
/// ```text
/// out   O   Output signal
/// in    I   Input signal
/// b     I   MA prediction coefficients, Q12 [Order]
/// len   I   Signal length
/// d     I   Filter Order
/// ```
/// Upstream c: silk/LPC_analysis_filter.c:silk_LPC_analysis_filter
#[inline]
pub fn silk_lpc_analysis_filter(out: &mut [i16], input: &[i16], b: &[i16]) {
    let len = input.len();
    let d = b.len();

    assert!(d >= 6);
    assert_eq!(d % 2, 0);
    assert!(d <= len);
    assert_eq!(out.len(), len);

    for _i in 0..(len - d) {
        let mut out32_q12 = 0i32;
        /* Allowing wrap around so that two wraps can cancel each other. The rare
        cases where the result wraps around can only be triggered by invalid streams*/
        for j in 0..d {
            out32_q12 = out32_q12.wrapping_add(input[_i + d - 1 - j] as i32 * b[j] as i32);
        }
        /* Subtract prediction */
        out32_q12 = ((input[_i + d] as i32) << 12).wrapping_sub(out32_q12);

        /* Scale to Q0 */
        let out32 = silk_rshift_round(out32_q12, 12);

        /* Saturate output */
        out[_i + d] = silk_sat16(out32) as i16;
    }

    /* Set first d output samples to zero */
    out[..d].fill(0);
}
