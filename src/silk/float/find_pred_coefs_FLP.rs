//! Floating-point prediction coefficient search.
//!
//! Upstream C: `silk/float/find_pred_coefs_FLP.c`

use crate::silk::define::{
    LTP_ORDER, MAX_LPC_ORDER, MAX_NB_SUBFR, MAX_PREDICTION_POWER_GAIN,
    MAX_PREDICTION_POWER_GAIN_AFTER_RESET, TYPE_VOICED,
};
use crate::silk::float::find_LPC_FLP::silk_find_LPC_FLP;
use crate::silk::float::find_LTP_FLP::silk_find_LTP_FLP;
use crate::silk::float::residual_energy_FLP::silk_residual_energy_FLP;
use crate::silk::float::scale_copy_vector_FLP::silk_scale_copy_vector_FLP;
use crate::silk::float::structs_FLP::{silk_encoder_control_FLP, silk_encoder_state_FLP};
use crate::silk::float::wrappers_FLP::{silk_process_NLSFs_FLP, silk_quant_LTP_gains_FLP};
use crate::silk::float::LTP_analysis_filter_FLP::silk_LTP_analysis_filter_FLP;
use crate::silk::float::LTP_scale_ctrl_FLP::silk_LTP_scale_ctrl_FLP;
use crate::silk::mathops::silk_exp2;
use crate::util::nalgebra::make_viewr_mut_generic;
use nalgebra::{Const, Dyn, VectorView};

/// Upstream C: silk/float/find_pred_coefs_FLP.c:silk_find_pred_coefs_FLP
pub fn silk_find_pred_coefs_FLP(
    psEnc: &mut silk_encoder_state_FLP,
    psEncCtrl: &mut silk_encoder_control_FLP,
    res_pitch: &[f32],
    x: &[f32],
    condCoding: i32,
) {
    let mut i: i32 = 0;
    let mut XXLTP: [f32; MAX_NB_SUBFR * LTP_ORDER * LTP_ORDER] = [0.; 100];
    let mut xXLTP: [f32; MAX_NB_SUBFR * LTP_ORDER] = [0.; 20];
    let mut invGains: [f32; MAX_NB_SUBFR] = [0.; 4];
    let mut NLSF_Q15: [i16; MAX_LPC_ORDER] = [0; 16];
    let mut LPC_in_pre: [f32; MAX_NB_SUBFR * MAX_LPC_ORDER + 320] = [0.; 384];
    let mut minInvGain: f32;
    i = 0;
    while i < psEnc.sCmn.nb_subfr as i32 {
        invGains[i as usize] = 1.0f32 / psEncCtrl.Gains[i as usize];
        i += 1;
    }
    if psEnc.sCmn.indices.signalType as i32 == TYPE_VOICED {
        assert!(
            psEnc.sCmn.ltp_mem_length as i32 - psEnc.sCmn.predictLPCOrder
                >= psEncCtrl.pitchL[0_usize] + 5 / 2
        );
        let nb_subfr = psEnc.sCmn.nb_subfr;
        let subfr_length = psEnc.sCmn.subfr_length;

        const LTP_ORDER: usize = crate::silk::define::LTP_ORDER;

        let mut XXLTP_mat = make_viewr_mut_generic(
            &mut XXLTP,
            Dyn(nb_subfr * LTP_ORDER),
            Const::<{ LTP_ORDER }>,
        );

        let mut xXLTP_mat =
            make_viewr_mut_generic(&mut xXLTP, Dyn(nb_subfr), Const::<{ LTP_ORDER }>);

        let r_ptr = psEnc.sCmn.ltp_mem_length;
        // res_pitch is passed already offset to start at -ltp_mem_length
        let lag = VectorView::<i32, Dyn>::from_slice(&(&psEncCtrl.pitchL)[..nb_subfr], nb_subfr);

        silk_find_LTP_FLP(
            &mut XXLTP_mat,
            &mut xXLTP_mat,
            res_pitch,
            r_ptr,
            &lag,
            subfr_length,
        );
        silk_quant_LTP_gains_FLP(
            &mut psEncCtrl.LTPCoef,
            &mut psEnc.sCmn.indices.LTPIndex,
            &mut psEnc.sCmn.indices.PERIndex,
            &mut psEnc.sCmn.sum_log_gain_Q7,
            &mut psEncCtrl.LTPredCodGain,
            &XXLTP,
            &xXLTP,
            psEnc.sCmn.subfr_length as i32,
            psEnc.sCmn.nb_subfr as i32,
            psEnc.sCmn.arch,
        );
        silk_LTP_scale_ctrl_FLP(psEnc, psEncCtrl, condCoding);
        {
            let ltp_mem = psEnc.sCmn.ltp_mem_length;
            let pred_order = psEnc.sCmn.predictLPCOrder as usize;
            let nb = psEnc.sCmn.nb_subfr;
            let subfr_len = psEnc.sCmn.subfr_length;
            // x starts at -ltp_mem_length, total_len = ltp_mem + nb * subfr_len
            let x_offset = ltp_mem - pred_order;
            silk_LTP_analysis_filter_FLP(
                &mut LPC_in_pre,
                x,
                x_offset,
                &psEncCtrl.LTPCoef,
                &psEncCtrl.pitchL,
                &invGains,
                subfr_len as i32,
                nb as i32,
                pred_order as i32,
            );
        }
    } else {
        let ltp_mem = psEnc.sCmn.ltp_mem_length;
        let pred_order = psEnc.sCmn.predictLPCOrder as usize;
        let subfr_len = psEnc.sCmn.subfr_length;
        let copy_len = subfr_len + pred_order;
        // x starts at offset 0 of x_buf; frame data starts at ltp_mem.
        // Each subframe needs pred_order samples before it, so base = ltp_mem - pred_order.
        let x_base = ltp_mem - pred_order;
        i = 0;
        while i < psEnc.sCmn.nb_subfr as i32 {
            let x_off = x_base + i as usize * subfr_len;
            let pre_off = i as usize * copy_len;
            silk_scale_copy_vector_FLP(
                &mut LPC_in_pre[pre_off..pre_off + copy_len],
                &x[x_off..x_off + copy_len],
                invGains[i as usize],
                copy_len as i32,
            );
            i += 1;
        }
        (&mut psEncCtrl.LTPCoef)[..(psEnc.sCmn.nb_subfr * 5)].fill(0.0);
        psEncCtrl.LTPredCodGain = 0.0f32;
        psEnc.sCmn.sum_log_gain_Q7 = 0;
    }
    if psEnc.sCmn.first_frame_after_reset != 0 {
        minInvGain = 1.0f32 / MAX_PREDICTION_POWER_GAIN_AFTER_RESET;
    } else {
        minInvGain = silk_exp2(psEncCtrl.LTPredCodGain / 3.0) / MAX_PREDICTION_POWER_GAIN;
        minInvGain /= 0.25f32 + 0.75f32 * psEncCtrl.coding_quality;
    }
    silk_find_LPC_FLP(&mut psEnc.sCmn, &mut NLSF_Q15, &LPC_in_pre, minInvGain);
    let prev_NLSFq_Q15 = psEnc.sCmn.prev_NLSFq_Q15;
    silk_process_NLSFs_FLP(
        &mut psEnc.sCmn,
        &mut psEncCtrl.PredCoef,
        &mut NLSF_Q15,
        &prev_NLSFq_Q15,
    );
    silk_residual_energy_FLP(
        &mut psEncCtrl.ResNrg,
        &LPC_in_pre,
        &psEncCtrl.PredCoef,
        &psEncCtrl.Gains,
        psEnc.sCmn.subfr_length as i32,
        psEnc.sCmn.nb_subfr as i32,
        psEnc.sCmn.predictLPCOrder,
    );
    psEnc.sCmn.prev_NLSFq_Q15.copy_from_slice(&NLSF_Q15);
}
