//! Conversion from LSF to LPC coefficients.
//!
//! Upstream c: `silk/NLSF2A.c`

use crate::arch::Arch;
use crate::silk::bwexpander_32::silk_bwexpander_32;
use crate::silk::define::{LSF_COS_TAB_SZ_FIX, MAX_LPC_STABILIZE_ITERATIONS};
use crate::silk::lpc_fit::silk_lpc_fit;
#[cfg(not(feature = "simd"))]
use crate::silk::lpc_inv_pred_gain::silk_lpc_inverse_pred_gain_c;
use crate::silk::sigproc_fix::{silk_rshift_round, silk_rshift_round64, SILK_MAX_ORDER_LPC};
#[cfg(feature = "simd")]
use crate::silk::simd::silk_lpc_inverse_pred_gain;
use crate::silk::table_lsf_cos::SILK_LSFCOSTAB_FIX_Q12;

pub const QA: i32 = 16;

///
/// helper function for NLSF2A(..)
///
/// ```text
/// out     O   intermediate polynomial, QA [dd+1]
/// c_lsf    I   vector of interleaved 2*cos(LSFs), QA [d]
/// dd      I   polynomial Order (= 1/2 * filter Order)
/// ```
/// Upstream c: silk/NLSF2A.c:silk_NLSF2A_find_poly
#[inline]
fn silk_nlsf2a_find_poly(out: &mut [i32], c_lsf: &[i32]) {
    let d = c_lsf.len();
    let dd = d / 2;
    assert_eq!(out.len(), dd + 1);

    out[0] = 1 << QA;
    out[1] = -c_lsf[0];

    for k in 1..dd {
        let ftmp = c_lsf[2 * k]; /* QA */
        out[k + 1] = out[k - 1] * 2 - silk_rshift_round64(ftmp as i64 * out[k] as i64, QA) as i32;

        for n in (2..=k).rev() {
            out[n] += out[n - 2] - silk_rshift_round64(ftmp as i64 * out[n - 1] as i64, QA) as i32;
        }

        out[1] -= ftmp;
    }
}

///
/// compute whitening filter coefficients from normalized line spectral frequencies
///
/// ```text
/// a_q12   O   monic whitening filter coefficients in Q12,  [ d ]
/// nlsf    I   normalized line spectral frequencies in Q15, [ d ]
/// d       I   filter Order (should be even)
/// arch    I   Run-time architecture
/// ```
/// Upstream c: silk/NLSF2A.c:silk_NLSF2A
#[inline]
pub fn silk_nlsf2a(a_q12: &mut [i16], nlsf: &[i16], arch: Arch) {
    let d = a_q12.len();

    /* This ordering was found to maximize quality. It improves the numerical accuracy of
    silk_nlsf2a_find_poly() compared to "standard" ordering. */
    const ORDERING16: [u8; 16] = [0, 15, 8, 7, 4, 11, 12, 3, 2, 13, 10, 5, 6, 9, 14, 1];
    const ORDERING10: [u8; 10] = [0, 9, 6, 3, 4, 5, 8, 1, 2, 7];

    debug_assert!(d == 10 || d == 16);

    /* convert LSFs to 2*cos(LSF), using piecewise linear curve from table */
    let ordering = if d == 16 {
        ORDERING16.as_slice()
    } else {
        ORDERING10.as_slice()
    };

    let mut cos_lsf_qa: [i32; SILK_MAX_ORDER_LPC] = [0; 24];
    let cos_lsf_qa = &mut cos_lsf_qa[..d + 1];
    for (&ordering, &nlsf) in ordering.iter().zip(nlsf.iter()) {
        debug_assert!(nlsf >= 0);

        /* f_int on a scale 0-127 (rounded down) */
        let f_int = nlsf as i32 >> (15 - 7);

        /* f_frac, range: 0..255 */
        let f_frac = nlsf as i32 - (f_int << (15 - 7));

        debug_assert!(f_int >= 0);
        debug_assert!(f_int < LSF_COS_TAB_SZ_FIX);

        /* Read start and end value from table */
        let cos_val = SILK_LSFCOSTAB_FIX_Q12[f_int as usize] as i32; /* Q12 */
        let delta = SILK_LSFCOSTAB_FIX_Q12[(f_int + 1) as usize] as i32 - cos_val; /* Q12, with a range of 0..200 */

        cos_lsf_qa[ordering as usize] = silk_rshift_round((cos_val << 8) + delta * f_frac, 20 - QA);
        /* QA */
    }

    let dd = d / 2;

    /* generate even and odd polynomials using convolution */
    let mut p: [i32; SILK_MAX_ORDER_LPC / 2 + 1] = [0; 13];
    let mut q: [i32; SILK_MAX_ORDER_LPC / 2 + 1] = [0; 13];
    let p = &mut p[..dd + 1];
    let q = &mut q[..dd + 1];
    silk_nlsf2a_find_poly(p, &cos_lsf_qa[..d]);
    silk_nlsf2a_find_poly(q, &cos_lsf_qa[1..][..d]);

    /* convert even and odd polynomials to opus_int32 Q12 filter coefs */
    let mut a32_qa1: [i32; SILK_MAX_ORDER_LPC] = [0; 24];
    let a32_qa1 = &mut a32_qa1[..d];
    for k in 0..dd {
        let ptmp = p[k + 1] + p[k];
        let qtmp = q[k + 1] - q[k];
        /* the ptmp and qtmp values at this stage need to fit in int32 */
        a32_qa1[k] = -qtmp - ptmp; /* QA+1 */
        a32_qa1[d - k - 1] = qtmp - ptmp; /* QA+1 */
    }

    /* Convert int32 coefficients to Q12 int16 coefs */
    silk_lpc_fit(a_q12, &mut a32_qa1[..d], 12, QA + 1);

    let mut _i = 0;
    #[cfg(feature = "simd")]
    let pred_gain_fn = |a: &[i16]| silk_lpc_inverse_pred_gain(a, arch);
    #[cfg(not(feature = "simd"))]
    let pred_gain_fn = {
        let _ = arch;
        silk_lpc_inverse_pred_gain_c
    };
    while pred_gain_fn(a_q12) == 0 && _i < MAX_LPC_STABILIZE_ITERATIONS {
        /* Prediction coefficients are (too close to) unstable; apply bandwidth expansion   */
        /* on the unscaled coefficients, convert to Q12 and measure again                   */
        silk_bwexpander_32(a32_qa1, 65536 - (2 << _i));

        for (a_q12, &a32_qa1) in a_q12.iter_mut().zip(a32_qa1.iter()) {
            *a_q12 = silk_rshift_round(a32_qa1, QA + 1 - 12) as i16; /* QA+1 -> Q12 */
        }

        _i += 1;
    }
}
