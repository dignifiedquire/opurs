//! Excitation pulse encoding.
//!
//! Upstream c: `silk/encode_pulses.c`

use crate::celt::entenc::{ec_enc_icdf, EcEnc};
use crate::silk::code_signs::silk_encode_signs;
use crate::silk::define::{N_RATE_LEVELS, SHELL_CODEC_FRAME_LENGTH, SILK_MAX_PULSES};
use crate::silk::shell_coder::silk_shell_encoder;
use crate::silk::tables_other::SILK_LSB_ICDF;
use crate::silk::tables_pulses_per_block::{
    SILK_MAX_PULSES_TABLE, SILK_PULSES_PER_BLOCK_BITS_Q5, SILK_PULSES_PER_BLOCK_ICDF,
    SILK_RATE_LEVELS_BITS_Q5, SILK_RATE_LEVELS_ICDF,
};
use itertools::izip;

/// Upstream c: silk/encode_pulses.c:combine_and_check
#[inline]
fn combine_and_check(pulses_comb: &mut [i32], max_pulses: u8) -> Option<&mut [i32]> {
    let len = pulses_comb.len() / 2;

    for k in 0..len {
        let sum = pulses_comb[2 * k] + pulses_comb[2 * k + 1];
        if sum > max_pulses as i32 {
            return None;
        }
        pulses_comb[k] = sum;
    }

    Some(&mut pulses_comb[..len])
}

///
/// Encode quantization indices of excitation
/// Upstream c: silk/encode_pulses.c:silk_encode_pulses
pub fn silk_encode_pulses(
    ps_range_enc: &mut EcEnc,
    signal_type: i32,
    quant_offset_type: i32,
    pulses_buffer: &mut [i8],
    frame_length: usize,
) {
    /****************************/
    /* Prepare for shell coding */
    /****************************/
    /* Calculate number of shell blocks */
    let mut iter = frame_length / SHELL_CODEC_FRAME_LENGTH;
    // special case for 10 ms @ 12 kHz: the frame length is not a multiple of SHELL_CODEC_FRAME_LENGTH
    // we expand the frame length to the next multiple of SHELL_CODEC_FRAME_LENGTH, filling the extra space with zeros
    if iter * SHELL_CODEC_FRAME_LENGTH < frame_length {
        debug_assert_eq!(frame_length, 12 * 10); /* Make sure only happens for 10 ms @ 12 kHz */
        iter += 1;
    }
    let iter = iter;
    let padded_frame_length = iter * SHELL_CODEC_FRAME_LENGTH;
    debug_assert!(frame_length <= pulses_buffer.len());
    // padded_frame_length max: 320
    const MAX_PADDED: usize = 320;
    debug_assert!(padded_frame_length <= MAX_PADDED);
    let mut pulses_padded_storage = [0i8; MAX_PADDED];
    let pulses_frame = if padded_frame_length <= pulses_buffer.len() {
        let pulses_frame = &mut pulses_buffer[..padded_frame_length];
        if frame_length < padded_frame_length {
            // 10 ms @ 12 kHz uses a partial shell frame; upstream pads this region with zeros.
            pulses_frame[frame_length..].fill(0);
        }
        pulses_frame
    } else {
        debug_assert_eq!(pulses_buffer.len(), frame_length);
        pulses_padded_storage[..frame_length].copy_from_slice(&pulses_buffer[..frame_length]);
        &mut pulses_padded_storage[..padded_frame_length]
    };

    /* Take the absolute value of the pulses */
    let mut abs_pulses = [0i32; MAX_PADDED];
    for (dst, src) in abs_pulses[..padded_frame_length]
        .iter_mut()
        .zip(pulses_frame.iter())
    {
        *dst = (*src as i32).abs();
    }

    /* Calc sum pulses per shell code frame */
    // iter max: MAX_PADDED / SHELL_CODEC_FRAME_LENGTH = 320 / 16 = 20
    const MAX_ITER: usize = 20;
    debug_assert!(iter <= MAX_ITER);
    let mut sum_pulses = [0i32; MAX_ITER];
    let mut n_rshifts = [0i32; MAX_ITER];

    for (abs_pulses_ptr, n_rshifts, sum_pulses) in izip!(
        abs_pulses[..padded_frame_length].chunks_exact_mut(SHELL_CODEC_FRAME_LENGTH),
        n_rshifts[..iter].iter_mut(),
        sum_pulses[..iter].iter_mut()
    ) {
        *n_rshifts = 0;
        loop {
            let mut pulses_comb: [i32; SHELL_CODEC_FRAME_LENGTH] = [0; 16];

            pulses_comb.copy_from_slice(abs_pulses_ptr);

            let Some(pulses_comb) = Some(pulses_comb.as_mut_slice())
                /* 1+1 -> 2 */
                .and_then(|pulses_comb| combine_and_check(pulses_comb, SILK_MAX_PULSES_TABLE[0]))
                /* 2+2 -> 4 */
                .and_then(|pulses_comb| combine_and_check(pulses_comb, SILK_MAX_PULSES_TABLE[1]))
                /* 4+4 -> 8 */
                .and_then(|pulses_comb| combine_and_check(pulses_comb, SILK_MAX_PULSES_TABLE[2]))
                /* 8+8 -> 16 */
                .and_then(|pulses_comb| combine_and_check(pulses_comb, SILK_MAX_PULSES_TABLE[3]))
            else {
                /* We need to downscale the quantization signal */
                *n_rshifts += 1;

                for v in abs_pulses_ptr.iter_mut() {
                    *v >>= 1;
                }

                continue;
            };

            debug_assert_eq!(pulses_comb.len(), 1);

            // it all went fine
            *sum_pulses = pulses_comb[0];

            break;
        }
    }

    /**************/
    /* Rate level */
    /**************/
    /* find rate level that leads to fewest bits for coding of pulses per block info */

    let rate_level_index = {
        let mut rate_level_index = 0;
        let mut min_sum_bits_q5 = i32::MAX;

        for (k, (n_bits_ptr, sum_bits_q5)) in izip!(
            SILK_PULSES_PER_BLOCK_BITS_Q5.iter(),
            SILK_RATE_LEVELS_BITS_Q5[(signal_type >> 1) as usize]
        )
        .enumerate()
        {
            let mut sum_bits_q5 = sum_bits_q5 as i32;

            for (&n_rshifts, &sum_pulses) in
                izip!(n_rshifts[..iter].iter(), sum_pulses[..iter].iter())
            {
                sum_bits_q5 += n_bits_ptr[if n_rshifts > 0 {
                    SILK_MAX_PULSES + 1
                } else {
                    sum_pulses as usize
                }] as i32;
            }

            if sum_bits_q5 < min_sum_bits_q5 {
                min_sum_bits_q5 = sum_bits_q5;
                rate_level_index = k;
            }
        }

        rate_level_index
    };

    ec_enc_icdf(
        ps_range_enc,
        rate_level_index as i32,
        &SILK_RATE_LEVELS_ICDF[(signal_type >> 1) as usize],
        8,
    );

    /***************************************************/
    /* Sum-Weighted-Pulses Encoding                    */
    /***************************************************/
    let cdf_ptr = &SILK_PULSES_PER_BLOCK_ICDF[rate_level_index];
    for (&sum_pulse, &n_rshifts) in izip!(&sum_pulses[..iter], &n_rshifts[..iter]) {
        if n_rshifts == 0 {
            ec_enc_icdf(ps_range_enc, sum_pulse, cdf_ptr, 8);
        } else {
            ec_enc_icdf(ps_range_enc, SILK_MAX_PULSES as i32 + 1, cdf_ptr, 8);

            for _ in 0..n_rshifts - 1 {
                ec_enc_icdf(
                    ps_range_enc,
                    SILK_MAX_PULSES as i32 + 1,
                    &SILK_PULSES_PER_BLOCK_ICDF[N_RATE_LEVELS - 1],
                    8,
                );
            }

            ec_enc_icdf(
                ps_range_enc,
                sum_pulse,
                &SILK_PULSES_PER_BLOCK_ICDF[N_RATE_LEVELS - 1],
                8,
            );
        }
    }

    /******************/
    /* Shell Encoding */
    /******************/
    for (&sum_pulses, abs_pulses_frame) in izip!(
        &sum_pulses[..iter],
        abs_pulses[..padded_frame_length].chunks_exact(SHELL_CODEC_FRAME_LENGTH)
    ) {
        if sum_pulses > 0 {
            silk_shell_encoder(ps_range_enc, abs_pulses_frame);
        }
    }

    /****************/
    /* LSB Encoding */
    /****************/
    for (pulse_frame, &n_rshifts) in izip!(
        pulses_frame.chunks_exact(SHELL_CODEC_FRAME_LENGTH),
        &n_rshifts[..iter]
    ) {
        if n_rshifts > 0 {
            let n_ls = n_rshifts - 1;

            for &q in pulse_frame {
                let abs_q = (q as i32).abs();

                for j in (1..=n_ls).rev() {
                    let bit = abs_q >> j & 1;
                    ec_enc_icdf(ps_range_enc, bit, &SILK_LSB_ICDF, 8);
                }
                let bit = abs_q & 1;
                ec_enc_icdf(ps_range_enc, bit, &SILK_LSB_ICDF, 8);
            }
        }
    }

    /****************/
    /* Encode signs */
    /****************/
    silk_encode_signs(
        ps_range_enc,
        pulses_frame,
        signal_type,
        quant_offset_type,
        &sum_pulses[..iter],
    );
}
