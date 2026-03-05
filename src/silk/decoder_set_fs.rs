//! Decoder sample rate configuration.
//!
//! Upstream C: `silk/decoder_set_fs.c`

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

/// Upstream C: silk/decoder_set_fs.c:silk_decoder_set_fs
pub fn silk_decoder_set_fs(psDec: &mut silk_decoder_state, fs_kHz: i32, fs_API_Hz: i32) -> i32 {
    let mut ret: i32 = 0;

    debug_assert!(fs_kHz == 8 || fs_kHz == 12 || fs_kHz == 16);
    debug_assert!(psDec.nb_subfr == 4 || psDec.nb_subfr == 4 / 2);
    psDec.subfr_length = 5 * fs_kHz as usize;
    let frame_length: i32 = psDec.nb_subfr as i16 as i32 * psDec.subfr_length as i16 as i32;
    if psDec.fs_kHz != fs_kHz || psDec.fs_API_hz != fs_API_Hz {
        ret += silk_resampler_init(
            &mut psDec.resampler_state,
            fs_kHz as i16 as i32 * 1000,
            fs_API_Hz,
            0,
        );
        psDec.fs_API_hz = fs_API_Hz;
    }
    if psDec.fs_kHz != fs_kHz || frame_length != psDec.frame_length as i32 {
        if fs_kHz == 8 {
            if psDec.nb_subfr == MAX_NB_SUBFR {
                psDec.pitch_contour_iCDF = &SILK_PITCH_CONTOUR_NB_ICDF;
            } else {
                psDec.pitch_contour_iCDF = &SILK_PITCH_CONTOUR_10_MS_NB_ICDF;
            }
        } else if psDec.nb_subfr == MAX_NB_SUBFR {
            psDec.pitch_contour_iCDF = &SILK_PITCH_CONTOUR_ICDF;
        } else {
            psDec.pitch_contour_iCDF = &SILK_PITCH_CONTOUR_10_MS_ICDF;
        }
        if psDec.fs_kHz != fs_kHz {
            psDec.ltp_mem_length = 20 * fs_kHz as i16 as usize;
            if fs_kHz == 8 || fs_kHz == 12 {
                psDec.LPC_order = MIN_LPC_ORDER;
                psDec.psNLSF_CB = &SILK_NLSF_CB_NB_MB;
            } else {
                psDec.LPC_order = MAX_LPC_ORDER;
                psDec.psNLSF_CB = &SILK_NLSF_CB_WB;
            }
            if fs_kHz == 16 {
                psDec.pitch_lag_low_bits_iCDF = &SILK_UNIFORM8_ICDF;
            } else if fs_kHz == 12 {
                psDec.pitch_lag_low_bits_iCDF = &SILK_UNIFORM6_ICDF;
            } else if fs_kHz == 8 {
                psDec.pitch_lag_low_bits_iCDF = &SILK_UNIFORM4_ICDF;
            } else {
                debug_assert!(false, "libopus: assert(0) called");
            }
            psDec.first_frame_after_reset = 1;
            psDec.lagPrev = 100;
            psDec.LastGainIndex = 10;
            psDec.prevSignalType = TYPE_NO_VOICE_ACTIVITY;
            psDec.outBuf.fill(0);
            psDec.sLPC_Q14_buf.fill(0);
        }
        psDec.fs_kHz = fs_kHz;
        psDec.frame_length = frame_length as usize;
    }
    debug_assert!(psDec.frame_length > 0 && psDec.frame_length <= 5 * 4 * 16);

    ret
}
