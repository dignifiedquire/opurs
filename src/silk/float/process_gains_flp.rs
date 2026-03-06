//! Floating-point gain processing.
//!
//! Upstream c: `silk/float/process_gains_FLP.c`

use crate::celt::mathops::celt_sqrt;
use crate::silk::define::{CODE_CONDITIONALLY, TYPE_VOICED};
use crate::silk::float::sigproc_flp::silk_sigmoid;
use crate::silk::float::structs_flp::{silk_encoder_control_FLP, silk_encoder_state_FLP};
use crate::silk::gain_quant::silk_gains_quant;
use crate::silk::mathops::silk_exp2;
use crate::silk::tables_other::SILK_QUANTIZATION_OFFSETS_Q10;
use crate::silk::tuning_parameters::{
    LAMBDA_CODING_QUALITY, LAMBDA_DELAYED_DECISIONS, LAMBDA_INPUT_QUALITY, LAMBDA_OFFSET,
    LAMBDA_QUANT_OFFSET, LAMBDA_SPEECH_ACT,
};

/// Upstream c: silk/float/process_gains_FLP.c:silk_process_gains_FLP
pub fn silk_process_gains_flp(
    ps_enc: &mut silk_encoder_state_FLP,
    ps_enc_ctrl: &mut silk_encoder_control_FLP,
    cond_coding: i32,
) {
    let mut k: i32;
    let mut p_gains_q16: [i32; 4] = [0; 4];
    let s: f32;

    let mut gain: f32;

    if ps_enc.s_cmn.indices.signal_type as i32 == TYPE_VOICED {
        s = 1.0f32 - 0.5f32 * silk_sigmoid(0.25f32 * (ps_enc_ctrl.lt_pred_cod_gain - 12.0f32));
        k = 0;
        while k < ps_enc.s_cmn.nb_subfr as i32 {
            ps_enc_ctrl.gains[k as usize] *= s;
            k += 1;
        }
    }
    let inv_max_sqr_val: f32 =
        silk_exp2(0.33f32 * (21.0f32 - ps_enc.s_cmn.snr_d_b_q7 as f32 * (1.0 / 128.0)))
            / ps_enc.s_cmn.subfr_length as f32;
    k = 0;
    while k < ps_enc.s_cmn.nb_subfr as i32 {
        gain = ps_enc_ctrl.gains[k as usize];
        gain = celt_sqrt(gain * gain + ps_enc_ctrl.res_nrg[k as usize] * inv_max_sqr_val);
        ps_enc_ctrl.gains[k as usize] = if gain < 32767.0f32 { gain } else { 32767.0f32 };
        k += 1;
    }
    k = 0;
    while k < ps_enc.s_cmn.nb_subfr as i32 {
        p_gains_q16[k as usize] = (ps_enc_ctrl.gains[k as usize] * 65536.0f32) as i32;
        k += 1;
    }
    let nb = ps_enc.s_cmn.nb_subfr;
    ps_enc_ctrl.gains_unq_q16[..nb].copy_from_slice(&p_gains_q16[..nb]);
    ps_enc_ctrl.last_gain_index_prev = ps_enc.s_shape.last_gain_index;
    silk_gains_quant(
        &mut ps_enc.s_cmn.indices.gains_indices[..nb],
        &mut p_gains_q16[..nb],
        &mut ps_enc.s_shape.last_gain_index,
        cond_coding == CODE_CONDITIONALLY,
    );
    k = 0;
    while k < ps_enc.s_cmn.nb_subfr as i32 {
        ps_enc_ctrl.gains[k as usize] = p_gains_q16[k as usize] as f32 / 65536.0f32;
        k += 1;
    }
    if ps_enc.s_cmn.indices.signal_type as i32 == TYPE_VOICED {
        if ps_enc_ctrl.lt_pred_cod_gain + ps_enc.s_cmn.input_tilt_q15 as f32 * (1.0f32 / 32768.0f32)
            > 1.0f32
        {
            ps_enc.s_cmn.indices.quant_offset_type = 0;
        } else {
            ps_enc.s_cmn.indices.quant_offset_type = 1;
        }
    }
    let quant_offset: f32 = SILK_QUANTIZATION_OFFSETS_Q10
        [(ps_enc.s_cmn.indices.signal_type as i32 >> 1) as usize]
        [ps_enc.s_cmn.indices.quant_offset_type as usize] as i32 as f32
        / 1024.0f32;
    ps_enc_ctrl.lambda = LAMBDA_OFFSET
        + LAMBDA_DELAYED_DECISIONS * ps_enc.s_cmn.n_states_delayed_decision as f32
        + LAMBDA_SPEECH_ACT * ps_enc.s_cmn.speech_activity_q8 as f32 * (1.0f32 / 256.0f32)
        + LAMBDA_INPUT_QUALITY * ps_enc_ctrl.input_quality
        + LAMBDA_CODING_QUALITY * ps_enc_ctrl.coding_quality
        + LAMBDA_QUANT_OFFSET * quant_offset;
}
