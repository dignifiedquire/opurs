//! SILK frame decoding.
//!
//! Upstream c: `silk/decode_frame.c`

use crate::arch::Arch;
use crate::celt::entdec::EcDec;

#[cfg(feature = "osce")]
use crate::celt::entcode::ec_tell;
#[cfg(feature = "osce")]
use crate::dnn::osce::{osce_enhance_frame, osce_reset, OSCEModel};

use crate::silk::cng::silk_cng;
use crate::silk::dec_api::{FLAG_DECODE_LBRR, FLAG_DECODE_NORMAL};
use crate::silk::decode_core::silk_decode_core;
use crate::silk::decode_indices::silk_decode_indices;
use crate::silk::decode_parameters::silk_decode_parameters;
use crate::silk::decode_pulses::silk_decode_pulses;
use crate::silk::define::{MAX_FRAME_LENGTH, SHELL_CODEC_FRAME_LENGTH};
use crate::silk::plc::{silk_plc, silk_plc_glue_frames};
use crate::silk::structs::{silk_decoder_control, silk_decoder_state};

#[cfg(feature = "deep-plc")]
use crate::dnn::lpcnet::LPCNetPLCState;

///
/// Decodes a SILK frame, writing `ps_dec.frame_length` samples to `p_out`.
/// Returns `(error_code, num_samples_written)`.
/// Upstream c: silk/decode_frame.c:silk_decode_frame
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn silk_decode_frame(
    ps_dec: &mut silk_decoder_state,
    ps_range_dec: &mut EcDec,
    p_out: &mut [i16],
    lost_flag: i32,
    cond_coding: i32,
    #[cfg(feature = "deep-plc")] lpcnet: Option<&mut LPCNetPLCState>,
    #[cfg(feature = "osce")] osce_model: &OSCEModel,
    arch: Arch,
) -> (i32, i32) {
    let l = ps_dec.frame_length as i32;
    let ret: i32 = 0;
    let mut ps_dec_ctrl = silk_decoder_control {
        pitch_l: [0; 4],
        gains_q16: [0; 4],
        pred_coef_q12: [[0; 16]; 2],
        ltpcoef_q14: [0; 20],
        ltp_scale_q14: 0,
    };
    debug_assert!(l > 0 && l <= 5 * 4 * 16);
    debug_assert!(p_out.len() >= l as usize);
    let p_out_slice = &mut p_out[..l as usize];
    if lost_flag == FLAG_DECODE_NORMAL
        || lost_flag == FLAG_DECODE_LBRR && ps_dec.lbrr_flags[ps_dec.n_frames_decoded as usize] == 1
    {
        #[cfg(feature = "osce")]
        let ec_start = ec_tell(ps_range_dec);

        // add room for padding samples so that the samples are a multiple of 16
        // these samples are not _really_ part of the frame
        let padded_frame_length = (l as usize).next_multiple_of(SHELL_CODEC_FRAME_LENGTH);
        let mut pulses = [0i16; MAX_FRAME_LENGTH];
        silk_decode_indices(
            ps_dec,
            ps_range_dec,
            ps_dec.n_frames_decoded,
            lost_flag,
            cond_coding,
        );
        silk_decode_pulses(
            ps_range_dec,
            &mut pulses[..padded_frame_length],
            ps_dec.indices.signal_type as i32,
            ps_dec.indices.quant_offset_type as i32,
        );
        silk_decode_parameters(ps_dec, &mut ps_dec_ctrl, cond_coding);
        silk_decode_core(
            ps_dec,
            &mut ps_dec_ctrl,
            &mut p_out_slice[..ps_dec.frame_length],
            &pulses[..ps_dec.frame_length],
        );

        // Update output buffer
        debug_assert!(ps_dec.ltp_mem_length >= ps_dec.frame_length);
        let mv_len = ps_dec.ltp_mem_length - ps_dec.frame_length;
        ps_dec
            .out_buf
            .copy_within(ps_dec.frame_length..ps_dec.ltp_mem_length, 0);
        ps_dec.out_buf[mv_len..mv_len + ps_dec.frame_length]
            .copy_from_slice(&p_out_slice[..ps_dec.frame_length]);

        // Run OSCE enhancement
        #[cfg(feature = "osce")]
        {
            let num_bits = ec_tell(ps_range_dec) - ec_start;
            osce_enhance_frame(
                osce_model,
                ps_dec,
                &ps_dec_ctrl,
                p_out_slice,
                num_bits,
                arch,
            );
        }

        silk_plc(
            ps_dec,
            &mut ps_dec_ctrl,
            p_out_slice,
            0,
            #[cfg(feature = "deep-plc")]
            lpcnet,
            arch,
        );
        ps_dec.loss_cnt = 0;
        ps_dec.prev_signal_type = ps_dec.indices.signal_type as i32;
        debug_assert!(ps_dec.prev_signal_type >= 0 && ps_dec.prev_signal_type <= 2);
        ps_dec.first_frame_after_reset = 0;
    } else {
        ps_dec.indices.signal_type = ps_dec.prev_signal_type as i8;
        silk_plc(
            ps_dec,
            &mut ps_dec_ctrl,
            p_out_slice,
            1,
            #[cfg(feature = "deep-plc")]
            lpcnet,
            arch,
        );

        // Reset OSCE on loss
        #[cfg(feature = "osce")]
        {
            let method = ps_dec.osce.method;
            osce_reset(&mut ps_dec.osce, method);
        }

        // Update output buffer
        debug_assert!(ps_dec.ltp_mem_length >= ps_dec.frame_length);
        let mv_len = ps_dec.ltp_mem_length - ps_dec.frame_length;
        ps_dec
            .out_buf
            .copy_within(ps_dec.frame_length..ps_dec.ltp_mem_length, 0);
        ps_dec.out_buf[mv_len..mv_len + ps_dec.frame_length]
            .copy_from_slice(&p_out_slice[..ps_dec.frame_length]);
    }
    silk_cng(ps_dec, &mut ps_dec_ctrl, p_out_slice);
    silk_plc_glue_frames(ps_dec, p_out_slice, l);
    ps_dec.lag_prev = ps_dec_ctrl.pitch_l[ps_dec.nb_subfr - 1];
    (ret, l)
}
