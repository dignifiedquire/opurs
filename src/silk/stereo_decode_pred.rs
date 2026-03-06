//! Stereo predictor decoding.
//!
//! Upstream c: `silk/stereo_decode_pred.c`

use crate::celt::entdec::{ec_dec_icdf, EcDec};
use crate::silk::define::STEREO_QUANT_SUB_STEPS;
use crate::silk::macros::silk_smulwb;
use crate::silk::sigproc_fix::SILK_FIX_CONST;
use crate::silk::tables_other::{
    SILK_STEREO_ONLY_CODE_MID_ICDF, SILK_STEREO_PRED_JOINT_ICDF, SILK_STEREO_PRED_QUANT_Q13,
    SILK_UNIFORM3_ICDF, SILK_UNIFORM5_ICDF,
};

/// Decode mid/side predictors
///
/// ```text
/// ps_range_dec    I/O   Compressor data structure
/// pred_q13[]    O     Predictors
/// ```
pub fn silk_stereo_decode_pred(ps_range_dec: &mut EcDec, pred_q13: &mut [i32; 2]) {
    let n = ec_dec_icdf(ps_range_dec, &SILK_STEREO_PRED_JOINT_ICDF, 8) as usize;

    let mut ix: [[usize; 3]; 2] = [[0; 3]; 2];
    ix[0][2] = n / 5;
    ix[1][2] = n - 5 * ix[0][2];

    /* Entropy decoding */
    let mut n = 0;
    while n < 2 {
        ix[n][0] = ec_dec_icdf(ps_range_dec, &SILK_UNIFORM3_ICDF, 8) as usize;
        ix[n][1] = ec_dec_icdf(ps_range_dec, &SILK_UNIFORM5_ICDF, 8) as usize;
        n += 1;
    }

    /* Dequantize */
    let mut n = 0;
    while n < 2 {
        ix[n][0] += 3 * ix[n][2];
        let low_q13 = SILK_STEREO_PRED_QUANT_Q13[ix[n][0]] as i32;
        let step_q13 = silk_smulwb(
            SILK_STEREO_PRED_QUANT_Q13[ix[n][0] + 1] as i32 - low_q13,
            SILK_FIX_CONST!(0.5 / STEREO_QUANT_SUB_STEPS as f64, 16),
        );

        pred_q13[n] = low_q13 + step_q13 as i16 as i32 * (2 * ix[n][1] + 1) as i16 as i32;
        n += 1;
    }

    /* Subtract second from first predictor (helps when actually applying these) */
    pred_q13[0] -= pred_q13[1];
}

/// Decode mid-only flag
///
/// ```text
/// ps_range_dec        I/O   Compressor data structure
/// decode_only_mid   O     Flag that only mid channel has been coded
/// ```
pub fn silk_stereo_decode_mid_only(ps_range_dec: &mut EcDec, decode_only_mid: &mut bool) {
    /* Decode flag that only mid channel is coded */
    *decode_only_mid = ec_dec_icdf(ps_range_dec, &SILK_STEREO_ONLY_CODE_MID_ICDF, 8) != 0;
}
