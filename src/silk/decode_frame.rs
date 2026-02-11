//! SILK frame decoding.
//!
//! Upstream C: `silk/decode_frame.c`

use crate::celt::entdec::ec_dec;

use crate::silk::dec_API::{FLAG_DECODE_LBRR, FLAG_DECODE_NORMAL};
use crate::silk::decode_core::silk_decode_core;
use crate::silk::decode_indices::silk_decode_indices;
use crate::silk::decode_parameters::silk_decode_parameters;
use crate::silk::decode_pulses::silk_decode_pulses;
use crate::silk::define::SHELL_CODEC_FRAME_LENGTH;
use crate::silk::structs::{silk_decoder_control, silk_decoder_state};
use crate::silk::CNG::silk_CNG;
use crate::silk::PLC::{silk_PLC, silk_PLC_glue_frames};

/// Upstream C: silk/decode_frame.c:silk_decode_frame
///
/// Decodes a SILK frame, writing `psDec.frame_length` samples to `pOut`.
/// Returns `(error_code, num_samples_written)`.
pub fn silk_decode_frame(
    psDec: &mut silk_decoder_state,
    psRangeDec: &mut ec_dec,
    pOut: &mut [i16],
    lostFlag: i32,
    condCoding: i32,
    arch: i32,
) -> (i32, i32) {
    let L = psDec.frame_length as i32;
    let ret: i32 = 0;
    let mut psDecCtrl = silk_decoder_control {
        pitchL: [0; 4],
        Gains_Q16: [0; 4],
        PredCoef_Q12: [[0; 16]; 2],
        LTPCoef_Q14: [0; 20],
        LTP_scale_Q14: 0,
    };
    assert!(L > 0 && L <= 5 * 4 * 16);
    assert!(pOut.len() >= L as usize);
    let pOut_slice = &mut pOut[..L as usize];
    if lostFlag == FLAG_DECODE_NORMAL
        || lostFlag == FLAG_DECODE_LBRR && psDec.LBRR_flags[psDec.nFramesDecoded as usize] == 1
    {
        // add room for padding samples so that the samples are a multiple of 16
        // these samples are not _really_ part of the frame
        let padded_frame_length = (L as usize).next_multiple_of(SHELL_CODEC_FRAME_LENGTH);
        let mut pulses: Vec<i16> = vec![0; padded_frame_length];
        silk_decode_indices(
            psDec,
            psRangeDec,
            psDec.nFramesDecoded,
            lostFlag,
            condCoding,
        );
        silk_decode_pulses(
            psRangeDec,
            &mut pulses,
            psDec.indices.signalType as i32,
            psDec.indices.quantOffsetType as i32,
        );
        silk_decode_parameters(psDec, &mut psDecCtrl, condCoding);
        silk_decode_core(
            psDec,
            &mut psDecCtrl,
            &mut pOut_slice[..psDec.frame_length],
            &pulses[..psDec.frame_length],
        );
        silk_PLC(psDec, &mut psDecCtrl, pOut_slice, 0, arch);
        psDec.lossCnt = 0;
        psDec.prevSignalType = psDec.indices.signalType as i32;
        assert!(psDec.prevSignalType >= 0 && psDec.prevSignalType <= 2);
        psDec.first_frame_after_reset = 0;
    } else {
        psDec.indices.signalType = psDec.prevSignalType as i8;
        silk_PLC(psDec, &mut psDecCtrl, pOut_slice, 1, arch);
    }
    assert!(psDec.ltp_mem_length >= psDec.frame_length);
    let mv_len = psDec.ltp_mem_length - psDec.frame_length;
    psDec
        .outBuf
        .copy_within(psDec.frame_length..psDec.ltp_mem_length, 0);
    psDec.outBuf[mv_len..mv_len + psDec.frame_length]
        .copy_from_slice(&pOut_slice[..psDec.frame_length]);
    silk_CNG(psDec, &mut psDecCtrl, pOut_slice);
    silk_PLC_glue_frames(psDec, pOut_slice, L);
    psDec.lagPrev = psDecCtrl.pitchL[psDec.nb_subfr - 1];
    (ret, L)
}
