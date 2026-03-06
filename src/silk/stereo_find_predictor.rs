//! Stereo predictor search.
//!
//! Upstream c: `silk/stereo_find_predictor.c`

use crate::silk::inner_prod_aligned::silk_inner_prod_aligned_scale;
use crate::silk::sigproc_fix::silk_max_int;
use crate::silk::sum_sqr_shift::silk_sum_sqr_shift;

use crate::silk::inlines::{silk_div32_varq, silk_sqrt_approx};

/// Upstream c: silk/stereo_find_predictor.c:silk_stereo_find_predictor
pub fn silk_stereo_find_predictor(
    ratio_q14: &mut i32,
    x: &[i16],
    y: &[i16],
    mid_res_amp_q0: &mut [i32],
    length: i32,
    mut smooth_coef_q16: i32,
) -> i32 {
    let mut scale: i32;
    let mut scale1: i32 = 0;
    let mut scale2: i32 = 0;
    let mut nrgx: i32 = 0;
    let mut nrgy: i32 = 0;

    let mut pred_q13: i32;

    silk_sum_sqr_shift(&mut nrgx, &mut scale1, &x[..length as usize]);
    silk_sum_sqr_shift(&mut nrgy, &mut scale2, &y[..length as usize]);
    scale = silk_max_int(scale1, scale2);
    scale = scale + (scale & 1);
    nrgy >>= scale - scale2;
    nrgx >>= scale - scale1;
    nrgx = silk_max_int(nrgx, 1);
    let corr: i32 =
        silk_inner_prod_aligned_scale(&x[..length as usize], &y[..length as usize], scale, length);
    pred_q13 = silk_div32_varq(corr, nrgx, 13);
    pred_q13 = pred_q13.clamp(-((1) << 14), (1) << 14);
    let pred2_q10: i32 = ((pred_q13 as i64 * pred_q13 as i16 as i64) >> 16) as i32;
    smooth_coef_q16 = silk_max_int(
        smooth_coef_q16,
        if pred2_q10 > 0 { pred2_q10 } else { -pred2_q10 },
    );
    scale >>= 1;
    mid_res_amp_q0[0] = (mid_res_amp_q0[0] as i64
        + (((((silk_sqrt_approx(nrgx) as u32) << scale) as i32 - mid_res_amp_q0[0]) as i64
            * smooth_coef_q16 as i16 as i64)
            >> 16)) as i32;
    nrgy -= ((((corr as i64 * pred_q13 as i16 as i64) >> 16) as i32 as u32) << (3 + 1)) as i32;
    nrgy += ((((nrgx as i64 * pred2_q10 as i16 as i64) >> 16) as i32 as u32) << 6) as i32;
    mid_res_amp_q0[1] = (mid_res_amp_q0[1] as i64
        + (((((silk_sqrt_approx(nrgy) as u32) << scale) as i32 - mid_res_amp_q0[1]) as i64
            * smooth_coef_q16 as i16 as i64)
            >> 16)) as i32;
    *ratio_q14 = silk_div32_varq(
        mid_res_amp_q0[1],
        if mid_res_amp_q0[0] > 1 {
            mid_res_amp_q0[0]
        } else {
            1
        },
        14,
    );
    *ratio_q14 = (*ratio_q14).clamp(0, 32767);
    pred_q13
}
