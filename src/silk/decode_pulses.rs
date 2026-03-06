//! Excitation pulse decoding.
//!
//! Upstream c: `silk/decode_pulses.c`

use crate::celt::entdec::{ec_dec_icdf, EcDec};
use crate::silk::code_signs::silk_decode_signs;
use crate::silk::define::{N_RATE_LEVELS, SHELL_CODEC_FRAME_LENGTH, SILK_MAX_PULSES};
use crate::silk::shell_coder::silk_shell_decoder;
use crate::silk::tables_other::SILK_LSB_ICDF;
use crate::silk::tables_pulses_per_block::{SILK_PULSES_PER_BLOCK_ICDF, SILK_RATE_LEVELS_ICDF};
use itertools::izip;

///
/// Decode quantization indices of excitation
///
/// NB: when operating on 10ms frame size @ 12 kHz, the `pulses` should be larger than the frame size (to make it contain a whole amount of shell frames)
///
/// ```text
/// ps_range_dec        _i/O   Compressor data structure
/// pulses[]          O     Excitation signal
/// signal_type        _i     Sigtype
/// quant_offset_type   _i     quant_offset_type
/// frame_length      _i     Frame length
/// ```
/// Upstream c: silk/decode_pulses.c:silk_decode_pulses
#[inline]
pub fn silk_decode_pulses(
    ps_range_dec: &mut EcDec,
    pulses: &mut [i16],
    signal_type: i32,
    quant_offset_type: i32,
) {
    /*********************/
    /* Decode rate level */
    /*********************/
    let rate_level_index = ec_dec_icdf(
        ps_range_dec,
        &(SILK_RATE_LEVELS_ICDF[(signal_type >> 1) as usize]),
        8,
    );
    let frame_length = pulses.len();
    let mut iter = frame_length / SHELL_CODEC_FRAME_LENGTH;
    if iter * SHELL_CODEC_FRAME_LENGTH < frame_length {
        debug_assert_eq!(frame_length, 12 * 10);
        iter += 1;
    }

    let mut sum_pulses: [i32; 20] = [0; 20];
    let mut n_lshifts: [i32; 20] = [0; 20];

    let sum_pulses = &mut sum_pulses[..iter];
    let n_lshifts = &mut n_lshifts[..iter];

    /***************************************************/
    /* Sum-Weighted-Pulses Decoding                    */
    /***************************************************/
    let cdf_ptr = &SILK_PULSES_PER_BLOCK_ICDF[rate_level_index as usize];
    for (out_n_lshifts, out_sum_pulse) in izip!(n_lshifts.iter_mut(), sum_pulses.iter_mut()) {
        let mut n_lshifts = 0;
        let mut sum_pulses = ec_dec_icdf(ps_range_dec, cdf_ptr, 8);
        /* LSB indication */
        while sum_pulses == SILK_MAX_PULSES as i32 + 1 {
            n_lshifts += 1;
            /* When we've already got 10 LSBs, we shift the table to not allow (SILK_MAX_PULSES + 1) */
            sum_pulses = ec_dec_icdf(
                ps_range_dec,
                &SILK_PULSES_PER_BLOCK_ICDF[N_RATE_LEVELS - 1][(n_lshifts == 10) as i32 as usize..],
                8,
            );
        }

        *out_n_lshifts = n_lshifts;
        *out_sum_pulse = sum_pulses;
    }

    let mut decode_pulse_blocks = |pulses_buf: &mut [i16]| {
        /***************************************************/
        /* Shell decoding                                  */
        /***************************************************/
        for (&sum_pulses, pulses_frame) in izip!(
            sum_pulses.iter(),
            pulses_buf.chunks_exact_mut(SHELL_CODEC_FRAME_LENGTH)
        ) {
            if sum_pulses > 0 {
                silk_shell_decoder(pulses_frame, ps_range_dec, sum_pulses);
            } else {
                pulses_frame.fill(0);
            }
        }

        /***************************************************/
        /* LSB Decoding                                    */
        /***************************************************/
        for (&n_lshifts, sum_pulses, pulses_frame) in izip!(
            n_lshifts.iter(),
            sum_pulses.iter_mut(),
            pulses_buf.chunks_exact_mut(SHELL_CODEC_FRAME_LENGTH)
        ) {
            if n_lshifts > 0 {
                for pulse in pulses_frame {
                    let mut abs_q = *pulse as i32;

                    for _ in 0..n_lshifts {
                        abs_q = ((abs_q as u32) << 1) as i32;
                        abs_q += ec_dec_icdf(ps_range_dec, &SILK_LSB_ICDF, 8);
                    }

                    *pulse = abs_q as i16;
                }

                /* Mark the number of pulses non-zero for sign decoding. */
                *sum_pulses |= n_lshifts << 5;
            }
        }

        /****************************************/
        /* Decode and add signs to pulse signal */
        /****************************************/
        silk_decode_signs(
            ps_range_dec,
            pulses_buf,
            signal_type,
            quant_offset_type,
            &sum_pulses[..iter],
        );
    };

    let padded_frame_length = iter * SHELL_CODEC_FRAME_LENGTH;
    if padded_frame_length == frame_length {
        decode_pulse_blocks(pulses);
    } else {
        let mut pulses_padded = vec![0i16; padded_frame_length];
        decode_pulse_blocks(&mut pulses_padded);
        pulses.copy_from_slice(&pulses_padded[..frame_length]);
    }
}
