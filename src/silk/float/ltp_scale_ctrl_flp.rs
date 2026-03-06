//! Floating-point LTP scaling control.
//!
//! Upstream c: `silk/float/LTP_scale_ctrl_FLP.c`

use crate::silk::define::CODE_INDEPENDENTLY;
use crate::silk::float::structs_flp::{silk_encoder_control_FLP, silk_encoder_state_FLP};
use crate::silk::log2lin::silk_log2lin;
use crate::silk::macros::silk_smulbb;
use crate::silk::tables_other::SILK_LTPSCALES_TABLE_Q14;

/// Upstream c: silk/float/LTP_scale_ctrl_FLP.c:silk_LTP_scale_ctrl_FLP
pub fn silk_ltp_scale_ctrl_flp(
    ps_enc: &mut silk_encoder_state_FLP,
    ps_enc_ctrl: &mut silk_encoder_control_FLP,
    cond_coding: i32,
) {
    if cond_coding == CODE_INDEPENDENTLY {
        /* Only scale if first frame in packet */
        let mut round_loss = ps_enc.s_cmn.packet_loss_perc * ps_enc.s_cmn.n_frames_per_packet;
        if ps_enc.s_cmn.lbrr_flag != 0 {
            /* LBRR reduces the effective loss. In practice, it does not square the loss because
            losses aren't independent, but that still seems to work best. We also never go below 2%. */
            round_loss = 2 + silk_smulbb(round_loss, round_loss) / 100;
        }
        let ltp_pred_cod_gain_i32 = ps_enc_ctrl.lt_pred_cod_gain as i32;
        ps_enc.s_cmn.indices.ltp_scale_index = (silk_smulbb(ltp_pred_cod_gain_i32, round_loss)
            > silk_log2lin(2900 - ps_enc.s_cmn.snr_d_b_q7))
            as i8;
        ps_enc.s_cmn.indices.ltp_scale_index += (silk_smulbb(ltp_pred_cod_gain_i32, round_loss)
            > silk_log2lin(3900 - ps_enc.s_cmn.snr_d_b_q7))
            as i8;
    } else {
        /* Default is minimum scaling */
        ps_enc.s_cmn.indices.ltp_scale_index = 0;
    }
    ps_enc_ctrl.ltp_scale =
        SILK_LTPSCALES_TABLE_Q14[ps_enc.s_cmn.indices.ltp_scale_index as usize] as f32 / 16384.0f32;
}
