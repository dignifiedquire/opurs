//! Floating-point wrappers for fixed-point SILK functions.
//!
//! Upstream c: `silk/float/wrappers_FLP.c`

use crate::arch::Arch;
use crate::silk::a2nlsf::silk_a2nlsf;
use crate::silk::nlsf2a::silk_nlsf2a;

use crate::silk::define::{LTP_ORDER, MAX_SHAPE_LPC_ORDER, TYPE_VOICED};
use crate::silk::float::sigproc_flp::silk_float2int;
use crate::silk::float::structs_flp::silk_encoder_control_FLP;
use crate::silk::nsq::silk_nsq;
use crate::silk::nsq_del_dec::silk_nsq_del_dec;
use crate::silk::process_nlsfs::silk_process_nlsfs;
use crate::silk::quant_ltp_gains::silk_quant_ltp_gains;
use crate::silk::structs::{silk_encoder_state, silk_nsq_state, NsqConfig, SideInfoIndices};
use crate::silk::tables_other::SILK_LTPSCALES_TABLE_Q14;

/// Upstream c: silk/float/wrappers_FLP.c:silk_A2NLSF_FLP
pub fn silk_a2nlsf_flp(nlsf_q15: &mut [i16], p_ar: &[f32], lpc_order: i32) {
    let mut _i: i32;
    let mut a_fix_q16: [i32; 16] = [0; 16];
    _i = 0;
    while _i < lpc_order {
        a_fix_q16[_i as usize] = silk_float2int(p_ar[_i as usize] * 65536.0f32);
        _i += 1;
    }
    silk_a2nlsf(
        &mut nlsf_q15[..lpc_order as usize],
        &mut a_fix_q16,
        lpc_order,
    );
}
/// Upstream c: silk/float/wrappers_FLP.c:silk_NLSF2A_FLP
pub fn silk_nlsf2a_flp(p_ar: &mut [f32], nlsf_q15: &[i16], lpc_order: i32, arch: Arch) {
    let mut _i: i32;
    let mut a_fix_q12: [i16; 16] = [0; 16];
    silk_nlsf2a(
        &mut a_fix_q12[..lpc_order as usize],
        &nlsf_q15[..lpc_order as usize],
        arch,
    );
    _i = 0;
    while _i < lpc_order {
        p_ar[_i as usize] = a_fix_q12[_i as usize] as f32 * (1.0f32 / 4096.0f32);
        _i += 1;
    }
}
/// Upstream c: silk/float/wrappers_FLP.c:silk_process_NLSFs_FLP
pub fn silk_process_nlsfs_flp(
    ps_enc_c: &mut silk_encoder_state,
    pred_coef: &mut [[f32; 16]; 2],
    nlsf_q15: &mut [i16],
    prev_nlsf_q15: &[i16],
) {
    let mut _i: i32;
    let mut j: i32;
    let mut pred_coef_q12: [[i16; 16]; 2] = [[0; 16]; 2];
    silk_process_nlsfs(ps_enc_c, &mut pred_coef_q12, nlsf_q15, prev_nlsf_q15);
    j = 0;
    while j < 2 {
        _i = 0;
        while _i < ps_enc_c.predict_lpcorder {
            pred_coef[j as usize][_i as usize] =
                pred_coef_q12[j as usize][_i as usize] as f32 * (1.0f32 / 4096.0f32);
            _i += 1;
        }
        j += 1;
    }
}
/// Upstream c: silk/float/wrappers_FLP.c:silk_NSQ_wrapper_FLP
pub fn silk_nsq_wrapper_flp(
    ps_enc_c: &NsqConfig,
    ps_enc_ctrl: &silk_encoder_control_FLP,
    ps_indices: &mut SideInfoIndices,
    ps_nsq: &mut silk_nsq_state,
    pulses: &mut [i8],
    x: &[f32],
) {
    let mut _i: i32;
    let mut j: i32;
    let mut x16: [i16; 320] = [0; 320];
    let mut gains_q16: [i32; 4] = [0; 4];
    let mut pred_coef_q12: [[i16; 16]; 2] = [[0; 16]; 2];
    let mut ltpcoef_q14: [i16; 20] = [0; 20];
    let mut ar_q13: [i16; 96] = [0; 96];
    let mut lf_shp_q14: [i32; 4] = [0; 4];

    let mut tilt_q14: [i32; 4] = [0; 4];
    let mut harm_shape_gain_q14: [i32; 4] = [0; 4];
    _i = 0;
    while _i < ps_enc_c.nb_subfr as i32 {
        j = 0;
        while j < ps_enc_c.shaping_lpcorder {
            ar_q13[(_i * MAX_SHAPE_LPC_ORDER + j) as usize] =
                silk_float2int(ps_enc_ctrl.ar[(_i * MAX_SHAPE_LPC_ORDER + j) as usize] * 8192.0f32)
                    as i16;
            j += 1;
        }
        _i += 1;
    }
    _i = 0;
    while _i < ps_enc_c.nb_subfr as i32 {
        lf_shp_q14[_i as usize] =
            ((silk_float2int(ps_enc_ctrl.lf_ar_shp[_i as usize] * 16384.0f32) as u32) << 16) as i32
                | silk_float2int(ps_enc_ctrl.lf_ma_shp[_i as usize] * 16384.0f32) as u16 as i32;
        tilt_q14[_i as usize] = silk_float2int(ps_enc_ctrl.tilt[_i as usize] * 16384.0f32);
        harm_shape_gain_q14[_i as usize] =
            silk_float2int(ps_enc_ctrl.harm_shape_gain[_i as usize] * 16384.0f32);
        _i += 1;
    }
    let lambda_q10: i32 = silk_float2int(ps_enc_ctrl.lambda * 1024.0f32);
    _i = 0;
    while _i < ps_enc_c.nb_subfr as i32 * LTP_ORDER as i32 {
        ltpcoef_q14[_i as usize] =
            silk_float2int(ps_enc_ctrl.ltp_coef[_i as usize] * 16384.0f32) as i16;
        _i += 1;
    }
    j = 0;
    while j < 2 {
        _i = 0;
        while _i < ps_enc_c.predict_lpcorder {
            pred_coef_q12[j as usize][_i as usize] =
                silk_float2int(ps_enc_ctrl.pred_coef[j as usize][_i as usize] * 4096.0f32) as i16;
            _i += 1;
        }
        j += 1;
    }
    _i = 0;
    while _i < ps_enc_c.nb_subfr as i32 {
        gains_q16[_i as usize] = silk_float2int(ps_enc_ctrl.gains[_i as usize] * 65536.0f32);
        _i += 1;
    }
    let ltp_scale_q14: i32 = if ps_indices.signal_type as i32 == TYPE_VOICED {
        SILK_LTPSCALES_TABLE_Q14[ps_indices.ltp_scale_index as usize] as i32
    } else {
        0
    };
    let frame_length = ps_enc_c.frame_length;
    _i = 0;
    while _i < frame_length as i32 {
        x16[_i as usize] = silk_float2int(x[_i as usize]) as i16;
        _i += 1;
    }
    if ps_enc_c.n_states_delayed_decision > 1 || ps_enc_c.warping_q16 > 0 {
        silk_nsq_del_dec(
            ps_enc_c,
            ps_nsq,
            ps_indices,
            &x16[..frame_length],
            pulses,
            pred_coef_q12.as_flattened(),
            &ltpcoef_q14,
            &ar_q13,
            &harm_shape_gain_q14,
            &tilt_q14,
            &lf_shp_q14,
            &gains_q16,
            &ps_enc_ctrl.pitch_l,
            lambda_q10,
            ltp_scale_q14,
        );
    } else {
        silk_nsq(
            ps_enc_c,
            ps_nsq,
            ps_indices,
            &x16[..frame_length],
            pulses,
            pred_coef_q12.as_flattened(),
            &ltpcoef_q14,
            &ar_q13,
            &harm_shape_gain_q14,
            &tilt_q14,
            &lf_shp_q14,
            &gains_q16,
            &ps_enc_ctrl.pitch_l,
            lambda_q10,
            ltp_scale_q14,
        );
    };
}
/// Upstream c: silk/float/wrappers_FLP.c:silk_quant_LTP_gains_FLP
#[allow(clippy::too_many_arguments)]
pub fn silk_quant_ltp_gains_flp(
    b: &mut [f32],
    cbk_index: &mut [i8],
    periodicity_index: &mut i8,
    sum_log_gain_q7: &mut i32,
    pred_gain_d_b: &mut f32,
    xx: &[f32],
    x_x: &[f32],
    subfr_len: i32,
    nb_subfr: i32,
    arch: Arch,
) {
    let mut _i: i32;
    let mut pred_gain_d_b_q7: i32 = 0;
    let mut b_q14: [i16; 20] = [0; 20];
    let mut xx_q17: [i32; 100] = [0; 100];
    let mut x_x_q17: [i32; 20] = [0; 20];
    _i = 0;
    while _i < nb_subfr * LTP_ORDER as i32 * LTP_ORDER as i32 {
        xx_q17[_i as usize] = silk_float2int(xx[_i as usize] * 131072.0f32);
        _i += 1;
    }
    _i = 0;
    while _i < nb_subfr * LTP_ORDER as i32 {
        x_x_q17[_i as usize] = silk_float2int(x_x[_i as usize] * 131072.0f32);
        _i += 1;
    }
    silk_quant_ltp_gains(
        &mut b_q14,
        cbk_index,
        periodicity_index,
        sum_log_gain_q7,
        &mut pred_gain_d_b_q7,
        &xx_q17,
        &x_x_q17,
        subfr_len,
        nb_subfr,
        arch,
    );
    _i = 0;
    while _i < nb_subfr * LTP_ORDER as i32 {
        b[_i as usize] = b_q14[_i as usize] as f32 * (1.0f32 / 16384.0f32);
        _i += 1;
    }
    *pred_gain_d_b = pred_gain_d_b_q7 as f32 * (1.0f32 / 128.0f32);
}
