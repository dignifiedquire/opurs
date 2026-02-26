//! Stereo predictor encoding.
//!
//! Upstream C: `silk/stereo_encode_pred.c`

use crate::celt::entenc::{ec_enc, ec_enc_icdf};
use crate::silk::tables_other::{
    silk_stereo_only_code_mid_iCDF, silk_stereo_pred_joint_iCDF, silk_uniform3_iCDF,
    silk_uniform5_iCDF,
};

/// Upstream C: silk/stereo_encode_pred.c:silk_stereo_encode_pred
pub fn silk_stereo_encode_pred(psRangeEnc: &mut ec_enc, ix: &[[i8; 3]]) {
    let mut n: i32 = 0;
    n = 5 * ix[0][2_usize] as i32 + ix[1][2_usize] as i32;
    debug_assert!(n < 25);
    ec_enc_icdf(psRangeEnc, n, &silk_stereo_pred_joint_iCDF, 8);
    n = 0;
    while n < 2 {
        debug_assert!((ix[n as usize][0_usize] as i32) < 3);
        debug_assert!((ix[n as usize][1_usize] as i32) < 5);
        ec_enc_icdf(
            psRangeEnc,
            ix[n as usize][0_usize] as i32,
            &silk_uniform3_iCDF,
            8,
        );
        ec_enc_icdf(
            psRangeEnc,
            ix[n as usize][1_usize] as i32,
            &silk_uniform5_iCDF,
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
        &silk_stereo_only_code_mid_iCDF,
        8,
    );
}
