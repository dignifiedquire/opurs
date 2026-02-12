//! LPC analysis filter.
//!
//! Upstream C: `silk/LPC_analysis_filter.c`

use crate::silk::SigProc_FIX::{silk_RSHIFT_ROUND, silk_SAT16};

/// Upstream C: silk/LPC_analysis_filter.c:silk_LPC_analysis_filter
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
/// B     I   MA prediction coefficients, Q12 [order]
/// len   I   Signal length
/// d     I   Filter order
/// ```
#[inline]
pub fn silk_LPC_analysis_filter(out: &mut [i16], input: &[i16], B: &[i16]) {
    let len = input.len();
    let d = B.len();

    assert!(d >= 6);
    assert_eq!(d % 2, 0);
    assert!(d <= len);
    assert_eq!(out.len(), len);

    for i in 0..(len - d) {
        let mut out32_Q12 = 0i32;
        /* Allowing wrap around so that two wraps can cancel each other. The rare
        cases where the result wraps around can only be triggered by invalid streams*/
        for j in 0..d {
            out32_Q12 = out32_Q12.wrapping_add(input[i + d - 1 - j] as i32 * B[j] as i32);
        }
        /* Subtract prediction */
        out32_Q12 = ((input[i + d] as i32) << 12).wrapping_sub(out32_Q12);

        /* Scale to Q0 */
        let out32 = silk_RSHIFT_ROUND(out32_Q12, 12);

        /* Saturate output */
        out[i + d] = silk_SAT16(out32) as i16;
    }

    /* Set first d output samples to zero */
    out[..d].fill(0);
}
