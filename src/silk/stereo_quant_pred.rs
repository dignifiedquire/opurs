//! Stereo predictor quantization.
//!
//! Upstream c: `silk/stereo_quant_pred.c`

use crate::silk::define::{STEREO_QUANT_SUB_STEPS, STEREO_QUANT_TAB_SIZE};
use crate::silk::tables_other::SILK_STEREO_PRED_QUANT_Q13;
use crate::silk::typedefs::SILK_INT32_MAX;

/// Upstream c: silk/stereo_quant_pred.c:silk_stereo_quant_pred
pub fn silk_stereo_quant_pred(pred_q13: &mut [i32], ix: &mut [[i8; 3]]) {
    let mut _i: i32;
    let mut j: i32;
    let mut n: i32;
    let mut low_q13: i32;
    let mut step_q13: i32;
    let mut lvl_q13: i32;
    let mut err_min_q13: i32;
    let mut err_q13: i32;
    let mut quant_pred_q13: i32 = 0;
    n = 0;
    while n < 2 {
        err_min_q13 = SILK_INT32_MAX;
        _i = 0;
        's_18: while _i < STEREO_QUANT_TAB_SIZE - 1 {
            low_q13 = SILK_STEREO_PRED_QUANT_Q13[_i as usize] as i32;
            step_q13 = (((SILK_STEREO_PRED_QUANT_Q13[(_i + 1) as usize] as i32 - low_q13) as i64
                * (0.5f64 / 5_f64 * ((1) << 16) as f64 + 0.5f64) as i32 as i16 as i64)
                >> 16) as i32;
            j = 0;
            while j < STEREO_QUANT_SUB_STEPS {
                lvl_q13 = low_q13 + step_q13 as i16 as i32 * (2 * j + 1) as i16 as i32;
                err_q13 = if pred_q13[n as usize] - lvl_q13 > 0 {
                    pred_q13[n as usize] - lvl_q13
                } else {
                    -(pred_q13[n as usize] - lvl_q13)
                };
                if err_q13 >= err_min_q13 {
                    break 's_18;
                }
                err_min_q13 = err_q13;
                quant_pred_q13 = lvl_q13;
                ix[n as usize][0_usize] = _i as i8;
                ix[n as usize][1_usize] = j as i8;
                j += 1;
            }
            _i += 1;
        }
        ix[n as usize][2_usize] = (ix[n as usize][0_usize] as i32 / 3) as i8;
        ix[n as usize][0_usize] =
            (ix[n as usize][0_usize] as i32 - ix[n as usize][2_usize] as i32 * 3) as i8;
        pred_q13[n as usize] = quant_pred_q13;
        n += 1;
    }
    pred_q13[0] -= pred_q13[1];
}
