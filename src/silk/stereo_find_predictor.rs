use crate::silk::inner_prod_aligned::silk_inner_prod_aligned_scale;
use crate::silk::sum_sqr_shift::silk_sum_sqr_shift;
use crate::silk::SigProc_FIX::silk_max_int;

use crate::silk::Inlines::{silk_DIV32_varQ, silk_SQRT_APPROX};

/// Upstream C: silk/stereo_find_predictor.c:silk_stereo_find_predictor
pub fn silk_stereo_find_predictor(
    ratio_Q14: &mut i32,
    x: &[i16],
    y: &[i16],
    mid_res_amp_Q0: &mut [i32],
    length: i32,
    mut smooth_coef_Q16: i32,
) -> i32 {
    let mut scale: i32 = 0;
    let mut scale1: i32 = 0;
    let mut scale2: i32 = 0;
    let mut nrgx: i32 = 0;
    let mut nrgy: i32 = 0;
    let mut corr: i32 = 0;
    let mut pred_Q13: i32 = 0;
    let mut pred2_Q10: i32 = 0;
    silk_sum_sqr_shift(&mut nrgx, &mut scale1, &x[..length as usize]);
    silk_sum_sqr_shift(&mut nrgy, &mut scale2, &y[..length as usize]);
    scale = silk_max_int(scale1, scale2);
    scale = scale + (scale & 1);
    nrgy >>= scale - scale2;
    nrgx >>= scale - scale1;
    nrgx = silk_max_int(nrgx, 1);
    corr =
        silk_inner_prod_aligned_scale(&x[..length as usize], &y[..length as usize], scale, length);
    pred_Q13 = silk_DIV32_varQ(corr, nrgx, 13);
    pred_Q13 = if -((1) << 14) > (1) << 14 {
        if pred_Q13 > -((1) << 14) {
            -((1) << 14)
        } else if pred_Q13 < (1) << 14 {
            (1) << 14
        } else {
            pred_Q13
        }
    } else if pred_Q13 > (1) << 14 {
        (1) << 14
    } else if pred_Q13 < -((1) << 14) {
        -((1) << 14)
    } else {
        pred_Q13
    };
    pred2_Q10 = ((pred_Q13 as i64 * pred_Q13 as i16 as i64) >> 16) as i32;
    smooth_coef_Q16 = silk_max_int(
        smooth_coef_Q16,
        if pred2_Q10 > 0 { pred2_Q10 } else { -pred2_Q10 },
    );
    scale >>= 1;
    mid_res_amp_Q0[0] = (mid_res_amp_Q0[0] as i64
        + (((((silk_SQRT_APPROX(nrgx) as u32) << scale) as i32 - mid_res_amp_Q0[0]) as i64
            * smooth_coef_Q16 as i16 as i64)
            >> 16)) as i32;
    nrgy -= ((((corr as i64 * pred_Q13 as i16 as i64) >> 16) as i32 as u32) << (3 + 1)) as i32;
    nrgy += ((((nrgx as i64 * pred2_Q10 as i16 as i64) >> 16) as i32 as u32) << 6) as i32;
    mid_res_amp_Q0[1] = (mid_res_amp_Q0[1] as i64
        + (((((silk_SQRT_APPROX(nrgy) as u32) << scale) as i32 - mid_res_amp_Q0[1]) as i64
            * smooth_coef_Q16 as i16 as i64)
            >> 16)) as i32;
    *ratio_Q14 = silk_DIV32_varQ(
        mid_res_amp_Q0[1],
        if mid_res_amp_Q0[0] > 1 {
            mid_res_amp_Q0[0]
        } else {
            1
        },
        14,
    );
    *ratio_Q14 = (*ratio_Q14).clamp(0, 32767);
    pred_Q13
}
