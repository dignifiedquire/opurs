//! Floating-point LTP scaling control.
//!
//! Upstream C: `silk/float/LTP_scale_ctrl_FLP.c`

use crate::silk::define::CODE_INDEPENDENTLY;
use crate::silk::float::structs_FLP::{silk_encoder_control_FLP, silk_encoder_state_FLP};
use crate::silk::log2lin::silk_log2lin;
use crate::silk::macros::silk_SMULBB;
use crate::silk::tables_other::silk_LTPScales_table_Q14;

/// Upstream C: silk/float/LTP_scale_ctrl_FLP.c:silk_LTP_scale_ctrl_FLP
pub fn silk_LTP_scale_ctrl_FLP(
    psEnc: &mut silk_encoder_state_FLP,
    psEncCtrl: &mut silk_encoder_control_FLP,
    condCoding: i32,
) {
    if condCoding == CODE_INDEPENDENTLY {
        /* Only scale if first frame in packet */
        let mut round_loss = psEnc.sCmn.PacketLoss_perc * psEnc.sCmn.nFramesPerPacket;
        if psEnc.sCmn.LBRR_flag != 0 {
            /* LBRR reduces the effective loss. In practice, it does not square the loss because
            losses aren't independent, but that still seems to work best. We also never go below 2%. */
            round_loss = 2 + silk_SMULBB(round_loss, round_loss) / 100;
        }
        let ltp_pred_cod_gain_i32 = psEncCtrl.LTPredCodGain as i32;
        psEnc.sCmn.indices.LTP_scaleIndex = (silk_SMULBB(ltp_pred_cod_gain_i32, round_loss)
            > silk_log2lin(2900 - psEnc.sCmn.SNR_dB_Q7))
            as i8;
        psEnc.sCmn.indices.LTP_scaleIndex += (silk_SMULBB(ltp_pred_cod_gain_i32, round_loss)
            > silk_log2lin(3900 - psEnc.sCmn.SNR_dB_Q7))
            as i8;
    } else {
        /* Default is minimum scaling */
        psEnc.sCmn.indices.LTP_scaleIndex = 0;
    }
    psEncCtrl.LTP_scale =
        silk_LTPScales_table_Q14[psEnc.sCmn.indices.LTP_scaleIndex as usize] as f32 / 16384.0f32;
}
