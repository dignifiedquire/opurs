//! Second-Order biquad filter.
//!
//! Upstream c: `silk/biquad_alt.c`

use crate::silk::macros::{silk_smlawb, silk_smulwb};
use crate::silk::sigproc_fix::{silk_rshift_round, silk_sat16};

///
/// Second Order ARMA filter, alternative implementation
///
/// Slower than biquad() but uses more precise coefficients.
/// Can handle (slowly) varying coefficients.
///
/// ```text
/// b_q28  _i     MA coefficients [3]
/// a_q28  _i     AR coefficients [2]
/// s      _i/O   State vector [2]
/// in     _i/O   input/output signal, length must be even
/// ```
/// Upstream c: silk/biquad_alt.c:silk_biquad_alt_stride1
pub fn silk_biquad_alt_stride1(
    b_q28: &[i32; 3],
    a_q28: &[i32; 2],
    s: &mut [i32; 2],
    signal: &mut [i16],
) {
    /* DIRECT FORM II TRANSPOSED (uses 2 element state vector) */

    /* Negate a_q28 values and split in two parts */
    let a0_l_q28 = -a_q28[0] & 0x3fff;
    let a0_u_q28 = -a_q28[0] >> 14;
    let a1_l_q28 = -a_q28[1] & 0x3fff;
    let a1_u_q28 = -a_q28[1] >> 14;

    assert_eq!(signal.len() % 2, 0);

    for signal in signal.iter_mut() {
        let inval = *signal as i32;

        /* s[ 0 ], s[ 1 ]: Q12 */
        let out32_q14 = silk_smlawb(s[0], b_q28[0], inval) << 2;

        s[0] = s[1] + silk_rshift_round(silk_smulwb(out32_q14, a0_l_q28), 14);
        s[0] = silk_smlawb(s[0], out32_q14, a0_u_q28);
        s[0] = silk_smlawb(s[0], b_q28[1], inval);

        s[1] = silk_rshift_round(silk_smulwb(out32_q14, a1_l_q28), 14);
        s[1] = silk_smlawb(s[1], out32_q14, a1_u_q28);
        s[1] = silk_smlawb(s[1], b_q28[2], inval);

        /* Scale back to Q0 and saturate */
        *signal = silk_sat16((out32_q14 + (1 << 14) - 1) >> 14) as i16;
    }
}
