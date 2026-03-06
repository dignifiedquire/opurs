//! Decoder sample rate configuration.
//!
//! Upstream c: `silk/decoder_set_fs.c`

use crate::silk::define::{MAX_LPC_ORDER, MAX_NB_SUBFR, MIN_LPC_ORDER, TYPE_NO_VOICE_ACTIVITY};
use crate::silk::resampler::silk_resampler_init;
use crate::silk::structs::silk_decoder_state;
use crate::silk::tables_nlsf_cb_nb_mb::SILK_NLSF_CB_NB_MB;
use crate::silk::tables_nlsf_cb_wb::SILK_NLSF_CB_WB;
use crate::silk::tables_other::{SILK_UNIFORM4_ICDF, SILK_UNIFORM6_ICDF, SILK_UNIFORM8_ICDF};
use crate::silk::tables_pitch_lag::{
    SILK_PITCH_CONTOUR_10_MS_ICDF, SILK_PITCH_CONTOUR_10_MS_NB_ICDF, SILK_PITCH_CONTOUR_ICDF,
    SILK_PITCH_CONTOUR_NB_ICDF,
};

/// Upstream c: silk/decoder_set_fs.c:silk_decoder_set_fs
pub fn silk_decoder_set_fs(ps_dec: &mut silk_decoder_state, fs_k_hz: i32, fs_api_hz: i32) -> i32 {
    let mut ret: i32 = 0;

    debug_assert!(fs_k_hz == 8 || fs_k_hz == 12 || fs_k_hz == 16);
    debug_assert!(ps_dec.nb_subfr == 4 || ps_dec.nb_subfr == 4 / 2);
    ps_dec.subfr_length = 5 * fs_k_hz as usize;
    let frame_length: i32 = ps_dec.nb_subfr as i16 as i32 * ps_dec.subfr_length as i16 as i32;
    if ps_dec.fs_k_hz != fs_k_hz || ps_dec.fs_api_hz != fs_api_hz {
        ret += silk_resampler_init(
            &mut ps_dec.resampler_state,
            fs_k_hz as i16 as i32 * 1000,
            fs_api_hz,
            0,
        );
        ps_dec.fs_api_hz = fs_api_hz;
    }
    if ps_dec.fs_k_hz != fs_k_hz || frame_length != ps_dec.frame_length as i32 {
        if fs_k_hz == 8 {
            if ps_dec.nb_subfr == MAX_NB_SUBFR {
                ps_dec.pitch_contour_i_cdf = &SILK_PITCH_CONTOUR_NB_ICDF;
            } else {
                ps_dec.pitch_contour_i_cdf = &SILK_PITCH_CONTOUR_10_MS_NB_ICDF;
            }
        } else if ps_dec.nb_subfr == MAX_NB_SUBFR {
            ps_dec.pitch_contour_i_cdf = &SILK_PITCH_CONTOUR_ICDF;
        } else {
            ps_dec.pitch_contour_i_cdf = &SILK_PITCH_CONTOUR_10_MS_ICDF;
        }
        if ps_dec.fs_k_hz != fs_k_hz {
            ps_dec.ltp_mem_length = 20 * fs_k_hz as i16 as usize;
            if fs_k_hz == 8 || fs_k_hz == 12 {
                ps_dec.lpc_order = MIN_LPC_ORDER;
                ps_dec.ps_nlsf_cb = &SILK_NLSF_CB_NB_MB;
            } else {
                ps_dec.lpc_order = MAX_LPC_ORDER;
                ps_dec.ps_nlsf_cb = &SILK_NLSF_CB_WB;
            }
            if fs_k_hz == 16 {
                ps_dec.pitch_lag_low_bits_i_cdf = &SILK_UNIFORM8_ICDF;
            } else if fs_k_hz == 12 {
                ps_dec.pitch_lag_low_bits_i_cdf = &SILK_UNIFORM6_ICDF;
            } else if fs_k_hz == 8 {
                ps_dec.pitch_lag_low_bits_i_cdf = &SILK_UNIFORM4_ICDF;
            } else {
                debug_assert!(false, "libopus: assert(0) called");
            }
            ps_dec.first_frame_after_reset = 1;
            ps_dec.lag_prev = 100;
            ps_dec.last_gain_index = 10;
            ps_dec.prev_signal_type = TYPE_NO_VOICE_ACTIVITY;
            ps_dec.out_buf.fill(0);
            ps_dec.s_lpc_q14_buf.fill(0);
        }
        ps_dec.fs_k_hz = fs_k_hz;
        ps_dec.frame_length = frame_length as usize;
    }
    debug_assert!(ps_dec.frame_length > 0 && ps_dec.frame_length <= 5 * 4 * 16);

    ret
}
