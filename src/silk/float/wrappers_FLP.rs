//! Floating-point wrappers for fixed-point SILK functions.
//!
//! Upstream C: `silk/float/wrappers_FLP.c`

use crate::arch::Arch;
use crate::silk::A2NLSF::silk_A2NLSF;
use crate::silk::NLSF2A::silk_NLSF2A;

use crate::silk::define::{LTP_ORDER, MAX_SHAPE_LPC_ORDER, TYPE_VOICED};
use crate::silk::float::structs_FLP::silk_encoder_control_FLP;
use crate::silk::float::SigProc_FLP::silk_float2int;
use crate::silk::process_NLSFs::silk_process_NLSFs;
use crate::silk::quant_LTP_gains::silk_quant_LTP_gains;
use crate::silk::structs::{silk_encoder_state, silk_nsq_state, NsqConfig, SideInfoIndices};
use crate::silk::tables_other::silk_LTPScales_table_Q14;
use crate::silk::NSQ_del_dec::silk_NSQ_del_dec_c;
use crate::silk::NSQ::silk_NSQ_c;

/// Upstream C: silk/float/wrappers_FLP.c:silk_A2NLSF_FLP
pub fn silk_A2NLSF_FLP(NLSF_Q15: &mut [i16], pAR: &[f32], LPC_order: i32) {
    let mut i: i32 = 0;
    let mut a_fix_Q16: [i32; 16] = [0; 16];
    i = 0;
    while i < LPC_order {
        a_fix_Q16[i as usize] = silk_float2int(pAR[i as usize] * 65536.0f32);
        i += 1;
    }
    silk_A2NLSF(
        &mut NLSF_Q15[..LPC_order as usize],
        &mut a_fix_Q16,
        LPC_order,
    );
}
/// Upstream C: silk/float/wrappers_FLP.c:silk_NLSF2A_FLP
pub fn silk_NLSF2A_FLP(pAR: &mut [f32], NLSF_Q15: &[i16], LPC_order: i32, arch: Arch) {
    let mut i: i32 = 0;
    let mut a_fix_Q12: [i16; 16] = [0; 16];
    silk_NLSF2A(
        &mut a_fix_Q12[..LPC_order as usize],
        &NLSF_Q15[..LPC_order as usize],
        arch,
    );
    i = 0;
    while i < LPC_order {
        pAR[i as usize] = a_fix_Q12[i as usize] as f32 * (1.0f32 / 4096.0f32);
        i += 1;
    }
}
/// Upstream C: silk/float/wrappers_FLP.c:silk_process_NLSFs_FLP
pub fn silk_process_NLSFs_FLP(
    psEncC: &mut silk_encoder_state,
    PredCoef: &mut [[f32; 16]; 2],
    NLSF_Q15: &mut [i16],
    prev_NLSF_Q15: &[i16],
) {
    let mut i: i32 = 0;
    let mut j: i32 = 0;
    let mut PredCoef_Q12: [[i16; 16]; 2] = [[0; 16]; 2];
    silk_process_NLSFs(psEncC, &mut PredCoef_Q12, NLSF_Q15, prev_NLSF_Q15);
    j = 0;
    while j < 2 {
        i = 0;
        while i < psEncC.predictLPCOrder {
            PredCoef[j as usize][i as usize] =
                PredCoef_Q12[j as usize][i as usize] as f32 * (1.0f32 / 4096.0f32);
            i += 1;
        }
        j += 1;
    }
}
/// Upstream C: silk/float/wrappers_FLP.c:silk_NSQ_wrapper_FLP
pub fn silk_NSQ_wrapper_FLP(
    psEncC: &NsqConfig,
    psEncCtrl: &silk_encoder_control_FLP,
    psIndices: &mut SideInfoIndices,
    psNSQ: &mut silk_nsq_state,
    pulses: &mut [i8],
    x: &[f32],
) {
    let mut i: i32 = 0;
    let mut j: i32 = 0;
    let mut x16: [i16; 320] = [0; 320];
    let mut Gains_Q16: [i32; 4] = [0; 4];
    let mut PredCoef_Q12: [[i16; 16]; 2] = [[0; 16]; 2];
    let mut LTPCoef_Q14: [i16; 20] = [0; 20];
    let mut AR_Q13: [i16; 96] = [0; 96];
    let mut LF_shp_Q14: [i32; 4] = [0; 4];

    let mut Tilt_Q14: [i32; 4] = [0; 4];
    let mut HarmShapeGain_Q14: [i32; 4] = [0; 4];
    i = 0;
    while i < psEncC.nb_subfr as i32 {
        j = 0;
        while j < psEncC.shapingLPCOrder {
            AR_Q13[(i * MAX_SHAPE_LPC_ORDER + j) as usize] =
                silk_float2int(psEncCtrl.AR[(i * MAX_SHAPE_LPC_ORDER + j) as usize] * 8192.0f32)
                    as i16;
            j += 1;
        }
        i += 1;
    }
    i = 0;
    while i < psEncC.nb_subfr as i32 {
        LF_shp_Q14[i as usize] =
            ((silk_float2int(psEncCtrl.LF_AR_shp[i as usize] * 16384.0f32) as u32) << 16) as i32
                | silk_float2int(psEncCtrl.LF_MA_shp[i as usize] * 16384.0f32) as u16 as i32;
        Tilt_Q14[i as usize] = silk_float2int(psEncCtrl.Tilt[i as usize] * 16384.0f32);
        HarmShapeGain_Q14[i as usize] =
            silk_float2int(psEncCtrl.HarmShapeGain[i as usize] * 16384.0f32);
        i += 1;
    }
    let Lambda_Q10: i32 = silk_float2int(psEncCtrl.Lambda * 1024.0f32);
    i = 0;
    while i < psEncC.nb_subfr as i32 * LTP_ORDER as i32 {
        LTPCoef_Q14[i as usize] = silk_float2int(psEncCtrl.LTPCoef[i as usize] * 16384.0f32) as i16;
        i += 1;
    }
    j = 0;
    while j < 2 {
        i = 0;
        while i < psEncC.predictLPCOrder {
            PredCoef_Q12[j as usize][i as usize] =
                silk_float2int(psEncCtrl.PredCoef[j as usize][i as usize] * 4096.0f32) as i16;
            i += 1;
        }
        j += 1;
    }
    i = 0;
    while i < psEncC.nb_subfr as i32 {
        Gains_Q16[i as usize] = silk_float2int(psEncCtrl.Gains[i as usize] * 65536.0f32);
        i += 1;
    }
    let LTP_scale_Q14: i32 = if psIndices.signalType as i32 == TYPE_VOICED {
        silk_LTPScales_table_Q14[psIndices.LTP_scaleIndex as usize] as i32
    } else {
        0
    };
    let frame_length = psEncC.frame_length;
    i = 0;
    while i < frame_length as i32 {
        x16[i as usize] = silk_float2int(x[i as usize]) as i16;
        i += 1;
    }
    if psEncC.nStatesDelayedDecision > 1 || psEncC.warping_Q16 > 0 {
        silk_NSQ_del_dec_c(
            psEncC,
            psNSQ,
            psIndices,
            &x16[..frame_length],
            pulses,
            PredCoef_Q12.as_flattened(),
            &LTPCoef_Q14,
            &AR_Q13,
            &HarmShapeGain_Q14,
            &Tilt_Q14,
            &LF_shp_Q14,
            &Gains_Q16,
            &psEncCtrl.pitchL,
            Lambda_Q10,
            LTP_scale_Q14,
        );
    } else {
        silk_NSQ_c(
            psEncC,
            psNSQ,
            psIndices,
            &x16[..frame_length],
            pulses,
            PredCoef_Q12.as_flattened(),
            &LTPCoef_Q14,
            &AR_Q13,
            &HarmShapeGain_Q14,
            &Tilt_Q14,
            &LF_shp_Q14,
            &Gains_Q16,
            &psEncCtrl.pitchL,
            Lambda_Q10,
            LTP_scale_Q14,
        );
    };
}
/// Upstream C: silk/float/wrappers_FLP.c:silk_quant_LTP_gains_FLP
pub fn silk_quant_LTP_gains_FLP(
    B: &mut [f32],
    cbk_index: &mut [i8],
    periodicity_index: &mut i8,
    sum_log_gain_Q7: &mut i32,
    pred_gain_dB: &mut f32,
    XX: &[f32],
    xX: &[f32],
    subfr_len: i32,
    nb_subfr: i32,
    arch: Arch,
) {
    let mut i: i32 = 0;
    let mut pred_gain_dB_Q7: i32 = 0;
    let mut B_Q14: [i16; 20] = [0; 20];
    let mut XX_Q17: [i32; 100] = [0; 100];
    let mut xX_Q17: [i32; 20] = [0; 20];
    i = 0;
    while i < nb_subfr * LTP_ORDER as i32 * LTP_ORDER as i32 {
        XX_Q17[i as usize] = silk_float2int(XX[i as usize] * 131072.0f32);
        i += 1;
    }
    i = 0;
    while i < nb_subfr * LTP_ORDER as i32 {
        xX_Q17[i as usize] = silk_float2int(xX[i as usize] * 131072.0f32);
        i += 1;
    }
    silk_quant_LTP_gains(
        &mut B_Q14,
        cbk_index,
        periodicity_index,
        sum_log_gain_Q7,
        &mut pred_gain_dB_Q7,
        &XX_Q17,
        &xX_Q17,
        subfr_len,
        nb_subfr,
        arch,
    );
    i = 0;
    while i < nb_subfr * LTP_ORDER as i32 {
        B[i as usize] = B_Q14[i as usize] as f32 * (1.0f32 / 16384.0f32);
        i += 1;
    }
    *pred_gain_dB = pred_gain_dB_Q7 as f32 * (1.0f32 / 128.0f32);
}
