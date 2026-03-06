//! Floating-point pitch lag search.
//!
//! Upstream c: `silk/float/find_pitch_lags_FLP.c`

use crate::arch::Arch;
use crate::silk::define::{TYPE_NO_VOICE_ACTIVITY, TYPE_UNVOICED, TYPE_VOICED};
use crate::silk::float::apply_sine_window_flp::silk_apply_sine_window_flp;
use crate::silk::float::autocorrelation_flp::silk_autocorrelation_flp;
use crate::silk::float::bwexpander_flp::silk_bwexpander_flp;
use crate::silk::float::k2a_flp::silk_k2a_flp;
use crate::silk::float::lpc_analysis_filter_flp::silk_lpc_analysis_filter_flp;
use crate::silk::float::pitch_analysis_core_flp::silk_pitch_analysis_core_flp;
use crate::silk::float::schur_flp::silk_schur_flp;
use crate::silk::float::structs_flp::{silk_encoder_control_FLP, silk_encoder_state_FLP};
use crate::silk::tuning_parameters::{
    FIND_PITCH_BANDWIDTH_EXPANSION, FIND_PITCH_WHITE_NOISE_FRACTION,
};

/// Upstream c: silk/float/find_pitch_lags_FLP.c:silk_find_pitch_lags_FLP
pub fn silk_find_pitch_lags_flp(
    ps_enc: &mut silk_encoder_state_FLP,
    ps_enc_ctrl: &mut silk_encoder_control_FLP,
    res: &mut [f32],
    x: &[f32],
    arch: Arch,
) {
    let mut thrhld: f32;

    let mut auto_corr: [f32; 17] = [0.; 17];
    let mut a: [f32; 16] = [0.; 16];
    let mut refl_coef: [f32; 16] = [0.; 16];
    let mut wsig: [f32; 384] = [0.; 384];
    let buf_len: i32 = ps_enc.s_cmn.la_pitch
        + ps_enc.s_cmn.frame_length as i32
        + ps_enc.s_cmn.ltp_mem_length as i32;
    debug_assert!(buf_len >= ps_enc.s_cmn.pitch_lpc_win_length);
    // x starts at offset 0, covers ltp_mem_length + frame_length + la_pitch = buf_len
    let x_buf = x;
    let la = ps_enc.s_cmn.la_pitch as usize;
    let win_len = ps_enc.s_cmn.pitch_lpc_win_length as usize;
    let x_buf_off = buf_len as usize - win_len;
    // Apply first half sine window
    silk_apply_sine_window_flp(
        &mut wsig[..la],
        &x_buf[x_buf_off..x_buf_off + la],
        1,
        la as i32,
    );
    // Copy flat middle section
    let flat_len = win_len - (la << 1);
    wsig[la..la + flat_len].copy_from_slice(&x_buf[x_buf_off + la..x_buf_off + la + flat_len]);
    // Apply second half sine window
    let shift = la + flat_len;
    silk_apply_sine_window_flp(
        &mut wsig[shift..shift + la],
        &x_buf[x_buf_off + shift..x_buf_off + shift + la],
        2,
        la as i32,
    );
    silk_autocorrelation_flp(
        &mut auto_corr[..(ps_enc.s_cmn.pitch_estimation_lpcorder + 1) as usize],
        &wsig[..ps_enc.s_cmn.pitch_lpc_win_length as usize],
        arch,
    );
    auto_corr[0_usize] += auto_corr[0_usize] * FIND_PITCH_WHITE_NOISE_FRACTION + 1_f32;
    let res_nrg: f32 = silk_schur_flp(
        &mut refl_coef,
        &auto_corr,
        ps_enc.s_cmn.pitch_estimation_lpcorder,
    );
    ps_enc_ctrl.pred_gain = auto_corr[0_usize] / (if res_nrg > 1.0f32 { res_nrg } else { 1.0f32 });
    silk_k2a_flp(&mut a, &refl_coef, ps_enc.s_cmn.pitch_estimation_lpcorder);
    silk_bwexpander_flp(
        &mut a,
        ps_enc.s_cmn.pitch_estimation_lpcorder,
        FIND_PITCH_BANDWIDTH_EXPANSION,
    );
    silk_lpc_analysis_filter_flp(
        &mut res[..buf_len as usize],
        &a,
        x_buf,
        buf_len,
        ps_enc.s_cmn.pitch_estimation_lpcorder,
    );
    if ps_enc.s_cmn.indices.signal_type as i32 != TYPE_NO_VOICE_ACTIVITY
        && ps_enc.s_cmn.first_frame_after_reset == 0
    {
        thrhld = 0.6f32;
        thrhld -= 0.004f32 * ps_enc.s_cmn.pitch_estimation_lpcorder as f32;
        thrhld -= 0.1f32 * ps_enc.s_cmn.speech_activity_q8 as f32 * (1.0f32 / 256.0f32);
        thrhld -= 0.15f32 * (ps_enc.s_cmn.prev_signal_type as i32 >> 1) as f32;
        thrhld -= 0.1f32 * ps_enc.s_cmn.input_tilt_q15 as f32 * (1.0f32 / 32768.0f32);
        if silk_pitch_analysis_core_flp(
            res,
            &mut ps_enc_ctrl.pitch_l,
            &mut ps_enc.s_cmn.indices.lag_index,
            &mut ps_enc.s_cmn.indices.contour_index,
            &mut ps_enc.ltpcorr,
            ps_enc.s_cmn.prev_lag,
            ps_enc.s_cmn.pitch_estimation_threshold_q16 as f32 / 65536.0f32,
            thrhld,
            ps_enc.s_cmn.fs_k_hz,
            ps_enc.s_cmn.pitch_estimation_complexity,
            ps_enc.s_cmn.nb_subfr as i32,
            arch,
        ) == 0
        {
            ps_enc.s_cmn.indices.signal_type = TYPE_VOICED as i8;
        } else {
            ps_enc.s_cmn.indices.signal_type = TYPE_UNVOICED as i8;
        }
    } else {
        ps_enc_ctrl.pitch_l.fill(0);
        ps_enc.s_cmn.indices.lag_index = 0;
        ps_enc.s_cmn.indices.contour_index = 0;
        ps_enc.ltpcorr = 0 as f32;
    };
}
