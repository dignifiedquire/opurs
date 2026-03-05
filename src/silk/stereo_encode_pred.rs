//! Stereo predictor encoding.
//!
//! Upstream C: `silk/stereo_encode_pred.c`

use crate::celt::entenc::{ec_enc, ec_enc_icdf};
use crate::silk::tables_other::{
    SILK_STEREO_ONLY_CODE_MID_ICDF, SILK_STEREO_PRED_JOINT_ICDF, SILK_UNIFORM3_ICDF,
    SILK_UNIFORM5_ICDF,
};

/// Upstream C: silk/stereo_encode_pred.c:silk_stereo_encode_pred
pub fn silk_stereo_encode_pred(psRangeEnc: &mut ec_enc, ix: &[[i8; 3]]) {
    let mut n: i32;
    n = 5 * ix[0][2_usize] as i32 + ix[1][2_usize] as i32;
    debug_assert!(n < 25);
    ec_enc_icdf(psRangeEnc, n, &SILK_STEREO_PRED_JOINT_ICDF, 8);
    n = 0;
    while n < 2 {
        debug_assert!((ix[n as usize][0_usize] as i32) < 3);
        debug_assert!((ix[n as usize][1_usize] as i32) < 5);
        ec_enc_icdf(
            psRangeEnc,
            ix[n as usize][0_usize] as i32,
            &SILK_UNIFORM3_ICDF,
            8,
        );
        ec_enc_icdf(
            psRangeEnc,
            ix[n as usize][1_usize] as i32,
            &SILK_UNIFORM5_ICDF,
            8,
        );
        n += 1;
    }
}
/// Upstream C: silk/stereo_encode_pred.c:silk_stereo_encode_mid_only
pub fn silk_stereo_encode_mid_only(psRangeEnc: &mut ec_enc, mid_only_flag: i8) {
    ec_enc_icdf(
        psRangeEnc,
        mid_only_flag as i32,
        &SILK_STEREO_ONLY_CODE_MID_ICDF,
        8,
    );
}
