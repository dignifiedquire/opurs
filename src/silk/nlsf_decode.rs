//! nlsf codebook decoding.
//!
//! Upstream c: `silk/NLSF_decode.c`

use crate::silk::nlsf_stabilize::silk_nlsf_stabilize;
use crate::silk::nlsf_unpack::silk_nlsf_unpack;

use crate::silk::structs::silk_NLSF_CB_struct;

///
/// Predictive dequantizer for nlsf residuals
///
/// Returns RD value in Q30
/// Upstream c: silk/NLSF_decode.c:silk_NLSF_residual_dequant
#[inline]
fn silk_nlsf_residual_dequant(
    x_q10: &mut [i16],
    indices: &[i8],
    pred_coef_q8: &[u8],
    quant_step_size_q16: i32,
) {
    assert_eq!(x_q10.len(), indices.len());
    assert_eq!(x_q10.len(), pred_coef_q8.len());

    let mut out_q10 = 0;

    for (x_q10, (&index, &pref_coef_q8)) in x_q10
        .iter_mut()
        .zip(indices.iter().zip(pred_coef_q8.iter()))
        .rev()
    {
        let pred_q10 = (out_q10 as i16 as i32 * pref_coef_q8 as i16 as i32) >> 8;
        out_q10 = ((index as u32) << 10) as i32;
        if out_q10 > 0 {
            out_q10 -= (0.1f64 * ((1) << 10) as f64 + 0.5f64) as i32;
        } else if out_q10 < 0 {
            out_q10 += (0.1f64 * ((1) << 10) as f64 + 0.5f64) as i32;
        }
        out_q10 =
            (pred_q10 as i64 + ((out_q10 as i64 * quant_step_size_q16 as i16 as i64) >> 16)) as i32;
        *x_q10 = out_q10 as i16;
    }
}

///
/// nlsf vector decoder
/// Upstream c: silk/NLSF_decode.c:silk_NLSF_decode
pub fn silk_nlsf_decode(
    p_nlsf_q15: &mut [i16],
    nlsfindices: &[i8],
    ps_nlsf_cb: &silk_NLSF_CB_struct,
) {
    assert_eq!(p_nlsf_q15.len(), ps_nlsf_cb.order as usize);
    assert_eq!(nlsfindices.len(), 1 + ps_nlsf_cb.order as usize);

    let mut pred_q8: [u8; 16] = [0; 16];
    let mut ec_ix: [i16; 16] = [0; 16];
    let mut res_q10: [i16; 16] = [0; 16];
    let mut nlsf_q15_tmp: i32;

    // Unpack entropy table indices and predictor for current CB1 index
    silk_nlsf_unpack(&mut ec_ix, &mut pred_q8, ps_nlsf_cb, nlsfindices[0] as i32);

    // Predictive residual dequantizer
    silk_nlsf_residual_dequant(
        &mut res_q10[..ps_nlsf_cb.order as usize],
        &nlsfindices[1..],
        &pred_q8[..ps_nlsf_cb.order as usize],
        ps_nlsf_cb.quant_step_size_q16 as i32,
    );

    // Apply inverse square-rooted weights to first stage and add to output
    let p_cb_element =
        &ps_nlsf_cb.cb1_nlsf_q8[(nlsfindices[0] as i32 * ps_nlsf_cb.order as i32) as usize..];
    let p_cb_wght_q9 =
        &ps_nlsf_cb.cb1_wght_q9[(nlsfindices[0] as i32 * ps_nlsf_cb.order as i32) as usize..];
    for (out, ((&res_q10, &p_cb_wght_q9), &p_cb_element)) in p_nlsf_q15
        .iter_mut()
        .zip(res_q10.iter().zip(p_cb_wght_q9).zip(p_cb_element))
    {
        nlsf_q15_tmp = ((res_q10 as i32 as u32) << 14) as i32 / p_cb_wght_q9 as i32
            + ((p_cb_element as i16 as u32) << 7) as i32;
        *out = nlsf_q15_tmp.clamp(0, 32767) as i16;
    }
    silk_nlsf_stabilize(p_nlsf_q15, ps_nlsf_cb.delta_min_q15);
}
