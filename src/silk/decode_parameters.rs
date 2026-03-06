//! Decoding of quantized parameters.
//!
//! Upstream c: `silk/decode_parameters.c`

use crate::silk::bwexpander::silk_bwexpander;
use crate::silk::decode_pitch::silk_decode_pitch;
use crate::silk::define::{BWE_AFTER_LOSS_Q16, CODE_CONDITIONALLY, LTP_ORDER, TYPE_VOICED};
use crate::silk::gain_quant::silk_gains_dequant;
use crate::silk::nlsf2a::silk_nlsf2a;
use crate::silk::nlsf_decode::silk_nlsf_decode;
use crate::silk::structs::{silk_decoder_control, silk_decoder_state};
use crate::silk::tables_ltp::SILK_LTP_VQ_PTRS_Q7;
use crate::silk::tables_other::SILK_LTPSCALES_TABLE_Q14;

///
/// Decode parameters from payload
///
/// ```text
/// ps_dec        I/O   State
/// ps_dec_ctrl    I/O   Decoder control
/// cond_coding   I     The type of conditional coding to use
/// ```
/// Upstream c: silk/decode_parameters.c:silk_decode_parameters
#[inline]
pub fn silk_decode_parameters(
    ps_dec: &mut silk_decoder_state,
    ps_dec_ctrl: &mut silk_decoder_control,
    cond_coding: i32,
) {
    let [pred_coef_q12_0, pred_coef_q12_1] = &mut ps_dec_ctrl.pred_coef_q12;
    let pred_coef_q12_0 = &mut pred_coef_q12_0[..ps_dec.lpc_order];
    let pred_coef_q12_1 = &mut pred_coef_q12_1[..ps_dec.lpc_order];

    let gains_q16 = &mut ps_dec_ctrl.gains_q16[..ps_dec.nb_subfr];
    let gains_indices = &ps_dec.indices.gains_indices[..ps_dec.nb_subfr];

    let nlsfindices = &ps_dec.indices.nlsfindices[..ps_dec.ps_nlsf_cb.order as usize + 1];

    let prev_nlsf_q15 = &mut ps_dec.prev_nlsf_q15[..ps_dec.lpc_order];

    let pitch_l = &mut ps_dec_ctrl.pitch_l[..ps_dec.nb_subfr];

    let ltpcoef_q14 = &mut ps_dec_ctrl.ltpcoef_q14[..ps_dec.nb_subfr * LTP_ORDER];

    /* Dequant Gains */
    silk_gains_dequant(
        gains_q16,
        gains_indices,
        &mut ps_dec.last_gain_index,
        cond_coding == CODE_CONDITIONALLY,
    );

    /****************/
    /* Decode NLSFs */
    /****************/
    let mut p_nlsf_q15: [i16; 16] = [0; 16];
    let p_nlsf_q15 = &mut p_nlsf_q15[..ps_dec.lpc_order];
    silk_nlsf_decode(p_nlsf_q15, nlsfindices, ps_dec.ps_nlsf_cb);

    /* Convert NLSF parameters to AR prediction filter coefficients */
    silk_nlsf2a(pred_coef_q12_1, p_nlsf_q15, ps_dec.arch);

    /* If just reset, e.g., because internal fs changed, do not allow interpolation */
    /* improves the case of packet loss in the first frame after a switch           */
    if ps_dec.first_frame_after_reset == 1 {
        ps_dec.indices.nlsfinterp_coef_q2 = 4;
    }
    if (ps_dec.indices.nlsfinterp_coef_q2 as i32) < 4 {
        /* Calculation of the interpolated NLSF0 vector from the interpolation factor, */
        /* the previous NLSF1, and the current NLSF1                                   */
        let mut p_nlsf0_q15: [i16; 16] = [0; 16];
        let p_nlsf0_q15 = &mut p_nlsf0_q15[..ps_dec.lpc_order];

        for _i in 0..ps_dec.lpc_order {
            p_nlsf0_q15[_i] = (prev_nlsf_q15[_i] as i32
                + ((ps_dec.indices.nlsfinterp_coef_q2 as i32
                    * (p_nlsf_q15[_i] as i32 - prev_nlsf_q15[_i] as i32))
                    >> 2)) as i16;
        }

        /* Convert NLSF parameters to AR prediction filter coefficients */
        silk_nlsf2a(pred_coef_q12_0, p_nlsf0_q15, ps_dec.arch);
    } else {
        /* Copy LPC coefficients for first half from second half */
        pred_coef_q12_0.copy_from_slice(pred_coef_q12_1);
    }

    prev_nlsf_q15[..ps_dec.lpc_order].copy_from_slice(&p_nlsf_q15[..ps_dec.lpc_order]);

    /* After a packet loss do BWE of LPC coefs */
    if ps_dec.loss_cnt != 0 {
        silk_bwexpander(pred_coef_q12_0, BWE_AFTER_LOSS_Q16);
        silk_bwexpander(pred_coef_q12_1, BWE_AFTER_LOSS_Q16);
    }

    if ps_dec.indices.signal_type as i32 == TYPE_VOICED {
        /*********************/
        /* Decode pitch lags */
        /*********************/

        /* Decode pitch values */
        silk_decode_pitch(
            ps_dec.indices.lag_index,
            ps_dec.indices.contour_index,
            pitch_l,
            ps_dec.fs_k_hz,
        );

        /* Decode Codebook Index */
        let cbk_ptr_q7 = SILK_LTP_VQ_PTRS_Q7[ps_dec.indices.perindex as usize];

        for k in 0..ps_dec.nb_subfr {
            let ix = ps_dec.indices.ltpindex[k] as usize;
            for _i in 0..LTP_ORDER {
                // ugh, I tried making it into a 2D array, but stuff broke
                // no idea why
                // ltpcoef_q14[k * LTP_ORDER as usize + _i] = (cbk_ptr_q7[ix][_i] as i16) << 7;
                ltpcoef_q14[k * LTP_ORDER + _i] = (cbk_ptr_q7[ix][_i] as i16) << 7;
            }
        }

        /**********************/
        /* Decode LTP scaling */
        /**********************/
        let ix = ps_dec.indices.ltp_scale_index as usize;
        ps_dec_ctrl.ltp_scale_q14 = SILK_LTPSCALES_TABLE_Q14[ix] as i32;
    } else {
        pitch_l.fill(0);
        ltpcoef_q14.fill(0);

        ps_dec.indices.perindex = 0;
        ps_dec_ctrl.ltp_scale_q14 = 0;
    };
}
