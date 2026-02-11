//! Floating-point pitch lag search.
//!
//! Upstream C: `silk/float/find_pitch_lags_FLP.c`

use crate::silk::define::{TYPE_NO_VOICE_ACTIVITY, TYPE_UNVOICED, TYPE_VOICED};
use crate::silk::float::apply_sine_window_FLP::silk_apply_sine_window_FLP;
use crate::silk::float::autocorrelation_FLP::silk_autocorrelation_FLP;
use crate::silk::float::bwexpander_FLP::silk_bwexpander_FLP;
use crate::silk::float::k2a_FLP::silk_k2a_FLP;
use crate::silk::float::pitch_analysis_core_FLP::silk_pitch_analysis_core_FLP;
use crate::silk::float::schur_FLP::silk_schur_FLP;
use crate::silk::float::structs_FLP::{silk_encoder_control_FLP, silk_encoder_state_FLP};
use crate::silk::float::LPC_analysis_filter_FLP::silk_LPC_analysis_filter_FLP;
use crate::silk::tuning_parameters::{
    FIND_PITCH_BANDWIDTH_EXPANSION, FIND_PITCH_WHITE_NOISE_FRACTION,
};

/// Upstream C: silk/float/find_pitch_lags_FLP.c:silk_find_pitch_lags_FLP
pub fn silk_find_pitch_lags_FLP(
    psEnc: &mut silk_encoder_state_FLP,
    psEncCtrl: &mut silk_encoder_control_FLP,
    res: &mut [f32],
    x: &[f32],
    arch: i32,
) {
    let mut thrhld: f32;

    let mut auto_corr: [f32; 17] = [0.; 17];
    let mut A: [f32; 16] = [0.; 16];
    let mut refl_coef: [f32; 16] = [0.; 16];
    let mut Wsig: [f32; 384] = [0.; 384];
    let buf_len: i32 =
        psEnc.sCmn.la_pitch + psEnc.sCmn.frame_length as i32 + psEnc.sCmn.ltp_mem_length as i32;
    assert!(buf_len >= psEnc.sCmn.pitch_LPC_win_length);
    // x starts at offset 0, covers ltp_mem_length + frame_length + la_pitch = buf_len
    let x_buf = x;
    let la = psEnc.sCmn.la_pitch as usize;
    let win_len = psEnc.sCmn.pitch_LPC_win_length as usize;
    let x_buf_off = buf_len as usize - win_len;
    // Apply first half sine window
    silk_apply_sine_window_FLP(
        &mut Wsig[..la],
        &x_buf[x_buf_off..x_buf_off + la],
        1,
        la as i32,
    );
    // Copy flat middle section
    let flat_len = win_len - (la << 1);
    Wsig[la..la + flat_len].copy_from_slice(&x_buf[x_buf_off + la..x_buf_off + la + flat_len]);
    // Apply second half sine window
    let shift = la + flat_len;
    silk_apply_sine_window_FLP(
        &mut Wsig[shift..shift + la],
        &x_buf[x_buf_off + shift..x_buf_off + shift + la],
        2,
        la as i32,
    );
    silk_autocorrelation_FLP(
        &mut auto_corr[..(psEnc.sCmn.pitchEstimationLPCOrder + 1) as usize],
        &Wsig[..psEnc.sCmn.pitch_LPC_win_length as usize],
    );
    auto_corr[0_usize] += auto_corr[0_usize] * FIND_PITCH_WHITE_NOISE_FRACTION + 1_f32;
    let res_nrg: f32 = silk_schur_FLP(
        &mut refl_coef,
        &auto_corr,
        psEnc.sCmn.pitchEstimationLPCOrder,
    );
    psEncCtrl.predGain = auto_corr[0_usize] / (if res_nrg > 1.0f32 { res_nrg } else { 1.0f32 });
    silk_k2a_FLP(&mut A, &refl_coef, psEnc.sCmn.pitchEstimationLPCOrder);
    silk_bwexpander_FLP(
        &mut A,
        psEnc.sCmn.pitchEstimationLPCOrder,
        FIND_PITCH_BANDWIDTH_EXPANSION,
    );
    silk_LPC_analysis_filter_FLP(
        &mut res[..buf_len as usize],
        &A,
        x_buf,
        buf_len,
        psEnc.sCmn.pitchEstimationLPCOrder,
    );
    if psEnc.sCmn.indices.signalType as i32 != TYPE_NO_VOICE_ACTIVITY
        && psEnc.sCmn.first_frame_after_reset == 0
    {
        thrhld = 0.6f32;
        thrhld -= 0.004f32 * psEnc.sCmn.pitchEstimationLPCOrder as f32;
        thrhld -= 0.1f32 * psEnc.sCmn.speech_activity_Q8 as f32 * (1.0f32 / 256.0f32);
        thrhld -= 0.15f32 * (psEnc.sCmn.prevSignalType as i32 >> 1) as f32;
        thrhld -= 0.1f32 * psEnc.sCmn.input_tilt_Q15 as f32 * (1.0f32 / 32768.0f32);
        if silk_pitch_analysis_core_FLP(
            res,
            &mut psEncCtrl.pitchL,
            &mut psEnc.sCmn.indices.lagIndex,
            &mut psEnc.sCmn.indices.contourIndex,
            &mut psEnc.LTPCorr,
            psEnc.sCmn.prevLag,
            psEnc.sCmn.pitchEstimationThreshold_Q16 as f32 / 65536.0f32,
            thrhld,
            psEnc.sCmn.fs_kHz,
            psEnc.sCmn.pitchEstimationComplexity,
            psEnc.sCmn.nb_subfr as i32,
            arch,
        ) == 0
        {
            psEnc.sCmn.indices.signalType = TYPE_VOICED as i8;
        } else {
            psEnc.sCmn.indices.signalType = TYPE_UNVOICED as i8;
        }
    } else {
        psEncCtrl.pitchL.fill(0);
        psEnc.sCmn.indices.lagIndex = 0;
        psEnc.sCmn.indices.contourIndex = 0;
        psEnc.LTPCorr = 0 as f32;
    };
}
