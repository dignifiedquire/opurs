//! High-pass filter with variable cutoff.
//!
//! Upstream c: `silk/HP_variable_cutoff.c`

use crate::silk::lin2log::silk_lin2log;

use crate::silk::define::TYPE_VOICED;
use crate::silk::float::structs_flp::silk_encoder_state_FLP;
use crate::silk::structs::silk_encoder_state;
use crate::silk::tuning_parameters::{
    VARIABLE_HP_MAX_CUTOFF_HZ, VARIABLE_HP_MAX_DELTA_FREQ, VARIABLE_HP_MIN_CUTOFF_HZ,
    VARIABLE_HP_SMTH_COEF1,
};

/// Upstream c: silk/HP_variable_cutoff.c:silk_HP_variable_cutoff
pub fn silk_hp_variable_cutoff(state_fxx: &mut [silk_encoder_state_FLP]) {
    let quality_q15: i32;
    let pitch_freq_hz_q16: i32;
    let mut pitch_freq_log_q7: i32;
    let mut delta_freq_q7: i32;
    let ps_enc_c1: &mut silk_encoder_state = &mut state_fxx[0].s_cmn;
    if ps_enc_c1.prev_signal_type as i32 == TYPE_VOICED {
        pitch_freq_hz_q16 = (((ps_enc_c1.fs_k_hz * 1000) as u32) << 16) as i32 / ps_enc_c1.prev_lag;
        pitch_freq_log_q7 = silk_lin2log(pitch_freq_hz_q16) - ((16) << 7);
        quality_q15 = ps_enc_c1.input_quality_bands_q15[0_usize];
        pitch_freq_log_q7 = (pitch_freq_log_q7 as i64
            + ((((((-quality_q15 as u32) << 2) as i32 as i64 * quality_q15 as i16 as i64) >> 16)
                as i32 as i64
                * (pitch_freq_log_q7
                    - (silk_lin2log(
                        ((VARIABLE_HP_MIN_CUTOFF_HZ * ((1) << 16)) as f64 + 0.5f64) as i32,
                    ) - ((16) << 7))) as i16 as i64)
                >> 16)) as i32;
        delta_freq_q7 = pitch_freq_log_q7 - (ps_enc_c1.variable_hp_smth1_q15 >> 8);
        if delta_freq_q7 < 0 {
            delta_freq_q7 *= 3;
        }
        delta_freq_q7 = if -(((VARIABLE_HP_MAX_DELTA_FREQ * ((1) << 7) as f32) as f64 + 0.5f64)
            as i32)
            > ((VARIABLE_HP_MAX_DELTA_FREQ * ((1) << 7) as f32) as f64 + 0.5f64) as i32
        {
            if delta_freq_q7
                > -(((VARIABLE_HP_MAX_DELTA_FREQ * ((1) << 7) as f32) as f64 + 0.5f64) as i32)
            {
                -(((VARIABLE_HP_MAX_DELTA_FREQ * ((1) << 7) as f32) as f64 + 0.5f64) as i32)
            } else if delta_freq_q7
                < ((VARIABLE_HP_MAX_DELTA_FREQ * ((1) << 7) as f32) as f64 + 0.5f64) as i32
            {
                ((VARIABLE_HP_MAX_DELTA_FREQ * ((1) << 7) as f32) as f64 + 0.5f64) as i32
            } else {
                delta_freq_q7
            }
        } else if delta_freq_q7
            > ((VARIABLE_HP_MAX_DELTA_FREQ * ((1) << 7) as f32) as f64 + 0.5f64) as i32
        {
            ((VARIABLE_HP_MAX_DELTA_FREQ * ((1) << 7) as f32) as f64 + 0.5f64) as i32
        } else if delta_freq_q7
            < -(((VARIABLE_HP_MAX_DELTA_FREQ * ((1) << 7) as f32) as f64 + 0.5f64) as i32)
        {
            -(((VARIABLE_HP_MAX_DELTA_FREQ * ((1) << 7) as f32) as f64 + 0.5f64) as i32)
        } else {
            delta_freq_q7
        };
        ps_enc_c1.variable_hp_smth1_q15 = (ps_enc_c1.variable_hp_smth1_q15 as i64
            + (((ps_enc_c1.speech_activity_q8 as i16 as i32 * delta_freq_q7 as i16 as i32) as i64
                * ((VARIABLE_HP_SMTH_COEF1 * ((1) << 16) as f32) as f64 + 0.5f64) as i32 as i16
                    as i64)
                >> 16)) as i32;
        ps_enc_c1.variable_hp_smth1_q15 = if ((silk_lin2log(VARIABLE_HP_MIN_CUTOFF_HZ) as u32) << 8)
            as i32
            > ((silk_lin2log(VARIABLE_HP_MAX_CUTOFF_HZ) as u32) << 8) as i32
        {
            if ps_enc_c1.variable_hp_smth1_q15
                > ((silk_lin2log(VARIABLE_HP_MIN_CUTOFF_HZ) as u32) << 8) as i32
            {
                ((silk_lin2log(VARIABLE_HP_MIN_CUTOFF_HZ) as u32) << 8) as i32
            } else if ps_enc_c1.variable_hp_smth1_q15
                < ((silk_lin2log(VARIABLE_HP_MAX_CUTOFF_HZ) as u32) << 8) as i32
            {
                ((silk_lin2log(VARIABLE_HP_MAX_CUTOFF_HZ) as u32) << 8) as i32
            } else {
                ps_enc_c1.variable_hp_smth1_q15
            }
        } else if ps_enc_c1.variable_hp_smth1_q15
            > ((silk_lin2log(VARIABLE_HP_MAX_CUTOFF_HZ) as u32) << 8) as i32
        {
            ((silk_lin2log(VARIABLE_HP_MAX_CUTOFF_HZ) as u32) << 8) as i32
        } else if ps_enc_c1.variable_hp_smth1_q15
            < ((silk_lin2log(VARIABLE_HP_MIN_CUTOFF_HZ) as u32) << 8) as i32
        {
            ((silk_lin2log(VARIABLE_HP_MIN_CUTOFF_HZ) as u32) << 8) as i32
        } else {
            ps_enc_c1.variable_hp_smth1_q15
        };
    }
}
