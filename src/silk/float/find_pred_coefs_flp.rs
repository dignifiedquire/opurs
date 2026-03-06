//! Floating-point prediction coefficient search.
//!
//! Upstream c: `silk/float/find_pred_coefs_FLP.c`

use crate::silk::define::{
    LTP_ORDER, MAX_LPC_ORDER, MAX_NB_SUBFR, MAX_PREDICTION_POWER_GAIN,
    MAX_PREDICTION_POWER_GAIN_AFTER_RESET, TYPE_VOICED,
};
use crate::silk::float::find_lpc_flp::silk_find_lpc_flp;
use crate::silk::float::find_ltp_flp::silk_find_ltp_flp;
use crate::silk::float::ltp_analysis_filter_flp::silk_ltp_analysis_filter_flp;
use crate::silk::float::ltp_scale_ctrl_flp::silk_ltp_scale_ctrl_flp;
use crate::silk::float::residual_energy_flp::silk_residual_energy_flp;
use crate::silk::float::scale_copy_vector_flp::silk_scale_copy_vector_flp;
use crate::silk::float::structs_flp::{silk_encoder_control_FLP, silk_encoder_state_FLP};
use crate::silk::float::wrappers_flp::{silk_process_nlsfs_flp, silk_quant_ltp_gains_flp};
use crate::silk::mathops::silk_exp2;
use crate::util::nalgebra::make_viewr_mut_generic;
use nalgebra::{Const, Dyn, VectorView};

/// Upstream c: silk/float/find_pred_coefs_FLP.c:silk_find_pred_coefs_FLP
pub fn silk_find_pred_coefs_flp(
    ps_enc: &mut silk_encoder_state_FLP,
    ps_enc_ctrl: &mut silk_encoder_control_FLP,
    res_pitch: &[f32],
    x: &[f32],
    cond_coding: i32,
) {
    let mut _i: i32;
    let mut xxltp: [f32; MAX_NB_SUBFR * LTP_ORDER * LTP_ORDER] = [0.; 100];
    let mut x_xltp: [f32; MAX_NB_SUBFR * LTP_ORDER] = [0.; 20];
    let mut inv_gains: [f32; MAX_NB_SUBFR] = [0.; 4];
    let mut nlsf_q15: [i16; MAX_LPC_ORDER] = [0; 16];
    let mut lpc_in_pre: [f32; MAX_NB_SUBFR * MAX_LPC_ORDER + 320] = [0.; 384];
    let mut min_inv_gain: f32;
    _i = 0;
    while _i < ps_enc.s_cmn.nb_subfr as i32 {
        inv_gains[_i as usize] = 1.0f32 / ps_enc_ctrl.gains[_i as usize];
        _i += 1;
    }
    if ps_enc.s_cmn.indices.signal_type as i32 == TYPE_VOICED {
        debug_assert!(
            ps_enc.s_cmn.ltp_mem_length as i32 - ps_enc.s_cmn.predict_lpcorder
                >= ps_enc_ctrl.pitch_l[0_usize] + 5 / 2
        );
        let nb_subfr = ps_enc.s_cmn.nb_subfr;
        let subfr_length = ps_enc.s_cmn.subfr_length;

        const LTP_ORDER: usize = crate::silk::define::LTP_ORDER;

        let mut xxltp_mat = make_viewr_mut_generic(
            &mut xxltp,
            Dyn(nb_subfr * LTP_ORDER),
            Const::<{ LTP_ORDER }>,
        );

        let mut x_xltp_mat =
            make_viewr_mut_generic(&mut x_xltp, Dyn(nb_subfr), Const::<{ LTP_ORDER }>);

        let r_ptr = ps_enc.s_cmn.ltp_mem_length;
        // res_pitch is passed already offset to start at -ltp_mem_length
        let lag = VectorView::<i32, Dyn>::from_slice(&(&ps_enc_ctrl.pitch_l)[..nb_subfr], nb_subfr);

        silk_find_ltp_flp(
            &mut xxltp_mat,
            &mut x_xltp_mat,
            res_pitch,
            r_ptr,
            &lag,
            subfr_length,
        );
        silk_quant_ltp_gains_flp(
            &mut ps_enc_ctrl.ltp_coef,
            &mut ps_enc.s_cmn.indices.ltpindex,
            &mut ps_enc.s_cmn.indices.perindex,
            &mut ps_enc.s_cmn.sum_log_gain_q7,
            &mut ps_enc_ctrl.lt_pred_cod_gain,
            &xxltp,
            &x_xltp,
            ps_enc.s_cmn.subfr_length as i32,
            ps_enc.s_cmn.nb_subfr as i32,
            ps_enc.s_cmn.arch,
        );
        silk_ltp_scale_ctrl_flp(ps_enc, ps_enc_ctrl, cond_coding);
        {
            let ltp_mem = ps_enc.s_cmn.ltp_mem_length;
            let pred_order = ps_enc.s_cmn.predict_lpcorder as usize;
            let nb = ps_enc.s_cmn.nb_subfr;
            let subfr_len = ps_enc.s_cmn.subfr_length;
            // x starts at -ltp_mem_length, total_len = ltp_mem + nb * subfr_len
            let x_offset = ltp_mem - pred_order;
            silk_ltp_analysis_filter_flp(
                &mut lpc_in_pre,
                x,
                x_offset,
                &ps_enc_ctrl.ltp_coef,
                &ps_enc_ctrl.pitch_l,
                &inv_gains,
                subfr_len as i32,
                nb as i32,
                pred_order as i32,
            );
        }
    } else {
        let ltp_mem = ps_enc.s_cmn.ltp_mem_length;
        let pred_order = ps_enc.s_cmn.predict_lpcorder as usize;
        let subfr_len = ps_enc.s_cmn.subfr_length;
        let copy_len = subfr_len + pred_order;
        // x starts at offset 0 of x_buf; frame data starts at ltp_mem.
        // Each subframe needs pred_order samples before it, so base = ltp_mem - pred_order.
        let x_base = ltp_mem - pred_order;
        _i = 0;
        while _i < ps_enc.s_cmn.nb_subfr as i32 {
            let x_off = x_base + _i as usize * subfr_len;
            let pre_off = _i as usize * copy_len;
            silk_scale_copy_vector_flp(
                &mut lpc_in_pre[pre_off..pre_off + copy_len],
                &x[x_off..x_off + copy_len],
                inv_gains[_i as usize],
                copy_len as i32,
            );
            _i += 1;
        }
        (&mut ps_enc_ctrl.ltp_coef)[..(ps_enc.s_cmn.nb_subfr * 5)].fill(0.0);
        ps_enc_ctrl.lt_pred_cod_gain = 0.0f32;
        ps_enc.s_cmn.sum_log_gain_q7 = 0;
    }
    if ps_enc.s_cmn.first_frame_after_reset != 0 {
        min_inv_gain = 1.0f32 / MAX_PREDICTION_POWER_GAIN_AFTER_RESET;
    } else {
        min_inv_gain = silk_exp2(ps_enc_ctrl.lt_pred_cod_gain / 3.0) / MAX_PREDICTION_POWER_GAIN;
        min_inv_gain /= 0.25f32 + 0.75f32 * ps_enc_ctrl.coding_quality;
    }
    silk_find_lpc_flp(&mut ps_enc.s_cmn, &mut nlsf_q15, &lpc_in_pre, min_inv_gain);
    let prev_nlsfq_q15 = ps_enc.s_cmn.prev_nlsfq_q15;
    silk_process_nlsfs_flp(
        &mut ps_enc.s_cmn,
        &mut ps_enc_ctrl.pred_coef,
        &mut nlsf_q15,
        &prev_nlsfq_q15,
    );
    silk_residual_energy_flp(
        &mut ps_enc_ctrl.res_nrg,
        &lpc_in_pre,
        &ps_enc_ctrl.pred_coef,
        &ps_enc_ctrl.gains,
        ps_enc.s_cmn.subfr_length as i32,
        ps_enc.s_cmn.nb_subfr as i32,
        ps_enc.s_cmn.predict_lpcorder,
    );
    ps_enc.s_cmn.prev_nlsfq_q15.copy_from_slice(&nlsf_q15);
}
