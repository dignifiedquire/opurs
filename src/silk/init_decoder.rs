//! Decoder initialization.
//!
//! Upstream C: `silk/init_decoder.c`

use crate::arch::{opus_select_arch, Arch};
use crate::silk::resampler::ResamplerState;
use crate::silk::structs::{silk_CNG_struct, silk_PLC_struct, silk_decoder_state, SideInfoIndices};
use crate::silk::tables_NLSF_CB_WB::silk_NLSF_CB_WB;
use crate::silk::CNG::silk_CNG_Reset;
use crate::silk::PLC::silk_PLC_Reset;

/// Reset decoder state, preserving model data (OSCE, etc.).
///
/// Upstream C: silk/init_decoder.c:silk_reset_decoder
pub fn silk_reset_decoder(dec: &mut silk_decoder_state) -> i32 {
    // Clear everything from prev_gain_Q16 onward (SILK_DECODER_STATE_RESET_START)
    dec.prev_gain_Q16 = 65536;
    dec.exc_Q14 = [0; 320];
    dec.sLPC_Q14_buf = [0; 16];
    dec.outBuf = [0; 480];
    dec.lagPrev = 0;
    dec.LastGainIndex = 0;
    dec.fs_kHz = 0;
    dec.fs_API_hz = 0;
    dec.nb_subfr = 0;
    dec.frame_length = 0;
    dec.subfr_length = 0;
    dec.ltp_mem_length = 0;
    dec.LPC_order = 0;
    dec.prevNLSF_Q15 = [0; 16];
    dec.first_frame_after_reset = 1;
    dec.pitch_lag_low_bits_iCDF = &[];
    dec.pitch_contour_iCDF = &[];
    dec.nFramesDecoded = 0;
    dec.nFramesPerPacket = 0;
    dec.ec_prevSignalType = 0;
    dec.ec_prevLagIndex = 0;
    dec.VAD_flags = [0; 3];
    dec.LBRR_flag = 0;
    dec.LBRR_flags = [0; 3];
    dec.resampler_state = ResamplerState::default();
    dec.psNLSF_CB = &silk_NLSF_CB_WB;
    dec.indices = SideInfoIndices::default();
    dec.sCNG = silk_CNG_struct::default();
    dec.lossCnt = 0;
    dec.prevSignalType = 0;
    dec.arch = opus_select_arch();
    dec.sPLC = silk_PLC_struct::default();
    #[cfg(feature = "osce")]
    {
        crate::dnn::osce::osce_reset(&mut dec.osce, crate::dnn::osce::OSCE_DEFAULT_METHOD);
    }

    silk_CNG_Reset(dec);
    silk_PLC_Reset(dec);

    0
}

fn zeroed_decoder_state() -> silk_decoder_state {
    silk_decoder_state {
        prev_gain_Q16: 0,
        exc_Q14: [0; 320],
        sLPC_Q14_buf: [0; 16],
        outBuf: [0; 480],
        lagPrev: 0,
        LastGainIndex: 0,
        fs_kHz: 0,
        fs_API_hz: 0,
        nb_subfr: 0,
        frame_length: 0,
        subfr_length: 0,
        ltp_mem_length: 0,
        LPC_order: 0,
        prevNLSF_Q15: [0; 16],
        first_frame_after_reset: 0,
        pitch_lag_low_bits_iCDF: &[],
        pitch_contour_iCDF: &[],
        nFramesDecoded: 0,
        nFramesPerPacket: 0,
        ec_prevSignalType: 0,
        ec_prevLagIndex: 0,
        VAD_flags: [0; 3],
        LBRR_flag: 0,
        LBRR_flags: [0; 3],
        resampler_state: ResamplerState::default(),
        psNLSF_CB: &silk_NLSF_CB_WB,
        indices: SideInfoIndices::default(),
        sCNG: silk_CNG_struct::default(),
        lossCnt: 0,
        prevSignalType: 0,
        arch: Arch::Scalar,
        sPLC: silk_PLC_struct::default(),
        #[cfg(feature = "osce")]
        osce: crate::dnn::osce::OSCEState::default(),
        #[cfg(feature = "osce")]
        osce_bwe: crate::dnn::osce::OSCEBWE::default(),
    }
}

/// Initialize a decoder state in place.
///
/// Upstream C: silk/init_decoder.c:silk_init_decoder
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
        dec.prev_gain_Q16 = 123;
        dec.first_frame_after_reset = 0;
        let ret = silk_reset_decoder(&mut dec);
        assert_eq!(ret, 0);
        assert_eq!(dec.prev_gain_Q16, 65536);
        assert_eq!(dec.first_frame_after_reset, 1);
    }

    #[test]
    fn silk_init_decoder_in_place_returns_success() {
        let mut dec = zeroed_decoder_state();
        dec.nFramesDecoded = 7;
        dec.prevSignalType = 2;
        let ret = silk_init_decoder(&mut dec);
        assert_eq!(ret, 0);
        assert_eq!(dec.nFramesDecoded, 0);
        assert_eq!(dec.prevSignalType, 0);
        assert_eq!(dec.prev_gain_Q16, 65536);
        assert_eq!(dec.first_frame_after_reset, 1);
    }
}
