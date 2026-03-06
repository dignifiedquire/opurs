//! LPC inverse prediction gain computation.
//!
//! Upstream c: `silk/LPC_inv_pred_gain.c`

use crate::silk::define::MAX_PREDICTION_POWER_GAIN;
use crate::silk::inlines::silk_inverse32_varq;
use crate::silk::macros::silk_clz32;
use crate::silk::sigproc_fix::{
    silk_rshift_round64, silk_smmul, SILK_FIX_CONST, SILK_MAX_ORDER_LPC,
};

const QA: i32 = 24;
const A_LIMIT: i32 = SILK_FIX_CONST!(0.99975, QA);

fn mul32_frac_q(a32: i32, b32: i32, q: i32) -> i32 {
    silk_rshift_round64(a32 as i64 * b32 as i64, q) as i32
}

///
/// Compute inverse of LPC prediction gain, and test if LPC coefficients are stable (all poles within unit circle)
///
/// ```text
///                              O   Returns inverse prediction gain in energy domain, Q30
/// a_qa[ SILK_MAX_ORDER_LPC ]   I   Prediction coefficients
/// order                        I   Prediction order
/// ```
/// Upstream c: silk/LPC_inv_pred_gain.c:LPC_inverse_pred_gain_QA_c
fn lpc_inverse_pred_gain_qa_c(a_qa: &mut [i32]) -> i32 {
    let order = a_qa.len();

    let mut inv_gain_q30 = SILK_FIX_CONST!(1.0, 30);
    let mut k = order - 1;
    while k > 0 {
        /* Check for stability */
        if a_qa[k] > A_LIMIT || a_qa[k] < -A_LIMIT {
            return 0;
        }

        /* Set RC equal to negated AR coef */
        let rc_q31 = -(a_qa[k] << (31 - QA));

        /* rc_mult1_q30 range: [ 1 : 2^30 ] */
        let rc_mult1_q30 = SILK_FIX_CONST!(1, 30) - silk_smmul(rc_q31, rc_q31);

        /* Update inverse gain */
        /* inv_gain_q30 range: [ 0 : 2^30 ] */
        inv_gain_q30 = silk_smmul(inv_gain_q30, rc_mult1_q30) << 2;
        if inv_gain_q30 < SILK_FIX_CONST!(1.0 / MAX_PREDICTION_POWER_GAIN, 30) {
            return 0;
        }

        /* rc_mult2 range: [ 2^30 : SILK_INT32_MAX ] */
        let mult2_q = 32 - silk_clz32(rc_mult1_q30.abs());
        let rc_mult2 = silk_inverse32_varq(rc_mult1_q30, mult2_q + 30);

        /* Update AR coefficient */
        let mut n = 0;
        while n < k.div_ceil(2) {
            let tmp1 = a_qa[n];
            let tmp2 = a_qa[k - n - 1];
            let tmp64 = silk_rshift_round64(
                tmp1.saturating_sub(mul32_frac_q(tmp2, rc_q31, 31)) as i64 * rc_mult2 as i64,
                mult2_q,
            );

            if tmp64 > i32::MAX as i64 || tmp64 < i32::MIN as i64 {
                return 0;
            }
            a_qa[n] = tmp64 as i32;
            let tmp64 = silk_rshift_round64(
                tmp2.saturating_sub(mul32_frac_q(tmp1, rc_q31, 31)) as i64 * rc_mult2 as i64,
                mult2_q,
            );

            if tmp64 > i32::MAX as i64 || tmp64 < i32::MIN as i64 {
                return 0;
            }
            a_qa[k - n - 1] = tmp64 as i32;
            n += 1;
        }
        k -= 1;
    }

    /* Check for stability */
    if a_qa[k] > A_LIMIT || a_qa[k] < -A_LIMIT {
        return 0;
    }

    /* Set RC equal to negated AR coef */
    let rc_q31 = -(a_qa[0] << (31 - QA));

    /* Range: [ 1 : 2^30 ] */
    let rc_mult1_q30 = SILK_FIX_CONST!(1, 30) - silk_smmul(rc_q31, rc_q31);

    /* Update inverse gain */
    /* Range: [ 0 : 2^30 ] */
    let inv_gain_q30 = silk_smmul(inv_gain_q30, rc_mult1_q30) << 2;
    if inv_gain_q30 < SILK_FIX_CONST!(1.0 / MAX_PREDICTION_POWER_GAIN, 30) {
        0
    } else {
        inv_gain_q30
    }
}

///
/// Compute inverse of LPC prediction gain, and test if LPC coefficients are stable (all poles within unit circle).
///
/// ```text
///         O   Returns inverse prediction gain in energy domain, Q30
/// a_q12   I   Prediction coefficients, Q12 [order]
/// order   I   Prediction order
/// ```
/// Upstream c: silk/LPC_inv_pred_gain.c:silk_LPC_inverse_pred_gain_c
#[inline]
pub fn silk_lpc_inverse_pred_gain_c(a_q12: &[i16]) -> i32 {
    let mut atmp_qa: [i32; SILK_MAX_ORDER_LPC] = [0; 24];
    let mut dc_resp: i32 = 0;

    let atmp_qa = &mut atmp_qa[..a_q12.len()];

    /* Increase q domain of the AR coefficients */
    let mut k = 0;
    while k < a_q12.len() {
        dc_resp += a_q12[k] as i32;
        atmp_qa[k] = (a_q12[k] as i32) << (QA - 12);
        k += 1;
    }
    /* If the DC is unstable, we don't even need to do the full calculations */
    if dc_resp >= 4096 {
        return 0;
    }
    lpc_inverse_pred_gain_qa_c(atmp_qa)
}
