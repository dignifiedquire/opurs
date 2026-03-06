//! Decoder initialization.
//!
//! Upstream c: `silk/init_decoder.c`

use crate::arch::{opus_select_arch, Arch};
use crate::silk::cng::silk_cng_reset;
use crate::silk::plc::silk_plc_reset;
use crate::silk::resampler::ResamplerState;
use crate::silk::structs::{silk_CNG_struct, silk_PLC_struct, silk_decoder_state, SideInfoIndices};
use crate::silk::tables_nlsf_cb_wb::SILK_NLSF_CB_WB;

/// Reset decoder state, preserving model data (OSCE, etc.).
///
/// Upstream c: silk/init_decoder.c:silk_reset_decoder
pub fn silk_reset_decoder(dec: &mut silk_decoder_state) -> i32 {
    // Clear everything from prev_gain_q16 onward (SILK_DECODER_STATE_RESET_START)
    dec.prev_gain_q16 = 65536;
    dec.exc_q14 = [0; 320];
    dec.s_lpc_q14_buf = [0; 16];
    dec.out_buf = [0; 480];
    dec.lag_prev = 0;
    dec.last_gain_index = 0;
    dec.fs_k_hz = 0;
    dec.fs_api_hz = 0;
    dec.nb_subfr = 0;
    dec.frame_length = 0;
    dec.subfr_length = 0;
    dec.ltp_mem_length = 0;
    dec.lpc_order = 0;
    dec.prev_nlsf_q15 = [0; 16];
    dec.first_frame_after_reset = 1;
    dec.pitch_lag_low_bits_i_cdf = &[];
    dec.pitch_contour_i_cdf = &[];
    dec.n_frames_decoded = 0;
    dec.n_frames_per_packet = 0;
    dec.ec_prev_signal_type = 0;
    dec.ec_prev_lag_index = 0;
    dec.vad_flags = [0; 3];
    dec.lbrr_flag = 0;
    dec.lbrr_flags = [0; 3];
    dec.resampler_state = ResamplerState::default();
    dec.ps_nlsf_cb = &SILK_NLSF_CB_WB;
    dec.indices = SideInfoIndices::default();
    dec.s_cng = silk_CNG_struct::default();
    dec.loss_cnt = 0;
    dec.prev_signal_type = 0;
    dec.arch = opus_select_arch();
    dec.s_plc = silk_PLC_struct::default();
    #[cfg(feature = "osce")]
    {
        crate::dnn::osce::osce_reset(&mut dec.osce, crate::dnn::osce::OSCE_DEFAULT_METHOD);
    }

    silk_cng_reset(dec);
    silk_plc_reset(dec);

    0
}

fn zeroed_decoder_state() -> silk_decoder_state {
    silk_decoder_state {
        prev_gain_q16: 0,
        exc_q14: [0; 320],
        s_lpc_q14_buf: [0; 16],
        out_buf: [0; 480],
        lag_prev: 0,
        last_gain_index: 0,
        fs_k_hz: 0,
        fs_api_hz: 0,
        nb_subfr: 0,
        frame_length: 0,
        subfr_length: 0,
        ltp_mem_length: 0,
        lpc_order: 0,
        prev_nlsf_q15: [0; 16],
        first_frame_after_reset: 0,
        pitch_lag_low_bits_i_cdf: &[],
        pitch_contour_i_cdf: &[],
        n_frames_decoded: 0,
        n_frames_per_packet: 0,
        ec_prev_signal_type: 0,
        ec_prev_lag_index: 0,
        vad_flags: [0; 3],
        lbrr_flag: 0,
        lbrr_flags: [0; 3],
        resampler_state: ResamplerState::default(),
        ps_nlsf_cb: &SILK_NLSF_CB_WB,
        indices: SideInfoIndices::default(),
        s_cng: silk_CNG_struct::default(),
        loss_cnt: 0,
        prev_signal_type: 0,
        arch: Arch::Scalar,
        s_plc: silk_PLC_struct::default(),
        #[cfg(feature = "osce")]
        osce: crate::dnn::osce::OSCEState::default(),
        #[cfg(feature = "osce")]
        osce_bwe: crate::dnn::osce::OSCEBWE::default(),
    }
}

/// Initialize a decoder state in place.
///
/// Upstream c: silk/init_decoder.c:silk_init_decoder
pub fn silk_init_decoder(dec: &mut silk_decoder_state) -> i32 {
    *dec = zeroed_decoder_state();
    silk_reset_decoder(dec)
}

/// Rust convenience constructor mirroring upstream init sequence.
pub fn silk_decoder_state_new() -> silk_decoder_state {
    let mut dec = zeroed_decoder_state();
    let _ = silk_init_decoder(&mut dec);
    dec
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn silk_reset_decoder_returns_success_and_sets_defaults() {
        let mut dec = zeroed_decoder_state();
        dec.prev_gain_q16 = 123;
        dec.first_frame_after_reset = 0;
        let ret = silk_reset_decoder(&mut dec);
        assert_eq!(ret, 0);
        assert_eq!(dec.prev_gain_q16, 65536);
        assert_eq!(dec.first_frame_after_reset, 1);
    }

    #[test]
    fn silk_init_decoder_in_place_returns_success() {
        let mut dec = zeroed_decoder_state();
        dec.n_frames_decoded = 7;
        dec.prev_signal_type = 2;
        let ret = silk_init_decoder(&mut dec);
        assert_eq!(ret, 0);
        assert_eq!(dec.n_frames_decoded, 0);
        assert_eq!(dec.prev_signal_type, 0);
        assert_eq!(dec.prev_gain_q16, 65536);
        assert_eq!(dec.first_frame_after_reset, 1);
    }
}
