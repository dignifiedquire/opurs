//! Core SILK decoder.
//!
//! Upstream c: `silk/decode_core.c`

use crate::silk::define::{
    LTP_ORDER, MAX_FRAME_LENGTH, MAX_LPC_ORDER, MAX_NB_SUBFR, MAX_SUB_FRAME_LENGTH,
    QUANT_LEVEL_ADJUST_Q10, TYPE_VOICED,
};
use crate::silk::inlines::{silk_div32_varq, silk_inverse32_varq};
use crate::silk::lpc_analysis_filter::silk_lpc_analysis_filter;
use crate::silk::macros::{silk_smlawb, silk_smulwb, silk_smulww};
use crate::silk::sigproc_fix::{
    silk_lshift_sat32, silk_rand, silk_rshift_round, silk_sat16, SILK_FIX_CONST,
};
use crate::silk::structs::{silk_decoder_control, silk_decoder_state};
use crate::silk::tables_other::SILK_QUANTIZATION_OFFSETS_Q10;

///
/// Core decoder. Performs inverse NSQ operation LTP + LPC
///
/// ```text
/// ps_dec                        I/O   Decoder state
/// ps_dec_ctrl                    I     Decoder control
/// xq[]                         O     Decoded speech
/// pulses[ MAX_FRAME_LENGTH ]   I     Pulse signal
/// arch                         I     Run-time architecture
/// ```
/// Upstream c: silk/decode_core.c:silk_decode_core
#[inline]
pub fn silk_decode_core(
    ps_dec: &mut silk_decoder_state,
    ps_dec_ctrl: &mut silk_decoder_control,
    xq: &mut [i16],
    pulses: &[i16],
) {
    // Max sizes: ltp_mem_length=320, frame_length=320, subfr_length=80
    let mut s_ltp = [0i16; MAX_FRAME_LENGTH];
    let mut s_ltp_q15 = [0i32; 2 * MAX_FRAME_LENGTH];
    let mut res_q14 = [0i32; MAX_SUB_FRAME_LENGTH];
    let mut s_lpc_q14 = [0i32; MAX_SUB_FRAME_LENGTH + MAX_LPC_ORDER];

    let offset_q10 = SILK_QUANTIZATION_OFFSETS_Q10
        [(ps_dec.indices.signal_type as i32 >> 1) as usize]
        [ps_dec.indices.quant_offset_type as usize] as i32;

    let nlsf_interpolation_flag = if (ps_dec.indices.nlsfinterp_coef_q2 as i32) < (1) << 2 {
        1
    } else {
        0
    };

    /* Decode excitation */
    let mut rand_seed = ps_dec.indices.seed as i32;
    let mut _i = 0;
    while _i < ps_dec.frame_length {
        rand_seed = silk_rand(rand_seed);
        ps_dec.exc_q14[_i] = (pulses[_i] as i32) << 14;
        if ps_dec.exc_q14[_i] > 0 {
            ps_dec.exc_q14[_i] -= QUANT_LEVEL_ADJUST_Q10 << 4;
        } else if ps_dec.exc_q14[_i] < 0 {
            ps_dec.exc_q14[_i] += QUANT_LEVEL_ADJUST_Q10 << 4;
        }
        ps_dec.exc_q14[_i] += offset_q10 << 4;
        if rand_seed < 0 {
            ps_dec.exc_q14[_i] = -ps_dec.exc_q14[_i];
        }
        rand_seed = rand_seed.wrapping_add(pulses[_i] as i32);
        _i += 1;
    }

    /* Copy LPC state */
    s_lpc_q14[..MAX_LPC_ORDER].copy_from_slice(&ps_dec.s_lpc_q14_buf);

    let mut pexc_q14 = ps_dec.exc_q14.as_mut_slice();
    // let mut pxq = xq;
    let mut s_ltp_buf_idx = ps_dec.ltp_mem_length;
    /* Loop over subframes */
    let mut k = 0;
    while k < ps_dec.nb_subfr {
        let mut pres_q14 = res_q14.as_mut_slice();
        let a_q12 = &ps_dec_ctrl.pred_coef_q12[k >> 1][..ps_dec.lpc_order];

        let mut a_q12_tmp: [i16; MAX_LPC_ORDER] = [0; 16];
        let a_q12_tmp = &mut a_q12_tmp[..ps_dec.lpc_order];

        /* Preload LPC coeficients to array on stack. Gives small performance gain */
        a_q12_tmp.copy_from_slice(a_q12);
        let b_q14 = &mut ps_dec_ctrl.ltpcoef_q14[k * LTP_ORDER..];
        let mut signal_type = ps_dec.indices.signal_type as i32;

        let gain_q10 = ps_dec_ctrl.gains_q16[k] >> 6;
        let mut inv_gain_q31 = silk_inverse32_varq(ps_dec_ctrl.gains_q16[k], 47);

        /* Calculate gain adjustment factor */
        let gain_adj_q16 = if ps_dec_ctrl.gains_q16[k] != ps_dec.prev_gain_q16 {
            let gain_adj_q16 = silk_div32_varq(ps_dec.prev_gain_q16, ps_dec_ctrl.gains_q16[k], 16);

            /* Scale short term state */
            for val in s_lpc_q14[..MAX_LPC_ORDER].iter_mut() {
                *val = silk_smulww(gain_adj_q16, *val);
            }

            gain_adj_q16
        } else {
            1 << 16
        };

        /* Save inv_gain */
        debug_assert!(inv_gain_q31 != 0);
        ps_dec.prev_gain_q16 = ps_dec_ctrl.gains_q16[k];

        /* Avoid abrupt transition from voiced PLC to unvoiced normal decoding */
        if ps_dec.loss_cnt != 0
            && ps_dec.prev_signal_type == TYPE_VOICED
            && ps_dec.indices.signal_type as i32 != TYPE_VOICED
            && k < MAX_NB_SUBFR / 2
        {
            b_q14[..LTP_ORDER].fill(0);
            b_q14[LTP_ORDER / 2] = SILK_FIX_CONST!(0.25, 14) as i16;

            signal_type = TYPE_VOICED;
            ps_dec_ctrl.pitch_l[k] = ps_dec.lag_prev;
        }

        let mut lag = 0;
        if signal_type == TYPE_VOICED {
            /* Voiced */
            lag = ps_dec_ctrl.pitch_l[k] as usize;

            /* Re-whitening */
            if k == 0 || k == 2 && nlsf_interpolation_flag != 0 {
                /* Rewhiten with new a coefs */
                let start_idx = ps_dec.ltp_mem_length - lag - ps_dec.lpc_order - LTP_ORDER / 2;
                debug_assert!(start_idx > 0);

                if k == 2 {
                    ps_dec.out_buf[ps_dec.ltp_mem_length..][..2 * ps_dec.subfr_length]
                        .copy_from_slice(&xq[..2 * ps_dec.subfr_length]);
                }

                silk_lpc_analysis_filter(
                    &mut s_ltp[start_idx..ps_dec.ltp_mem_length],
                    &ps_dec.out_buf[start_idx + k * ps_dec.subfr_length..]
                        [..(ps_dec.ltp_mem_length - start_idx)],
                    a_q12,
                );

                /* After rewhitening the LTP state is unscaled */
                if k == 0 {
                    /* Do LTP downscaling to reduce inter-packet dependency */
                    inv_gain_q31 = silk_smulwb(inv_gain_q31, ps_dec_ctrl.ltp_scale_q14) << 2;
                }
                let mut _i = 0;
                while _i < lag + LTP_ORDER / 2 {
                    s_ltp_q15[s_ltp_buf_idx - _i - 1] =
                        silk_smulwb(inv_gain_q31, s_ltp[ps_dec.ltp_mem_length - _i - 1] as i32);
                    _i += 1;
                }
                /* Update LTP state when Gain changes */
            } else if gain_adj_q16 != (1) << 16 {
                let mut _i = 0;
                while _i < lag + LTP_ORDER / 2 {
                    s_ltp_q15[s_ltp_buf_idx - _i - 1] =
                        silk_smulww(gain_adj_q16, s_ltp_q15[s_ltp_buf_idx - _i - 1]);
                    _i += 1;
                }
            }
        }

        /* Long-term prediction */
        if signal_type == TYPE_VOICED {
            /* Set up pointer */
            let mut pred_lag_ptr = s_ltp_buf_idx - lag + LTP_ORDER / 2;
            let mut _i = 0;
            while _i < ps_dec.subfr_length {
                /* Unrolled loop */
                /* Avoids introducing a bias because silk_smlawb() always rounds to -inf */
                let mut ltp_pred_q13 = 2;
                ltp_pred_q13 = silk_smlawb(ltp_pred_q13, s_ltp_q15[pred_lag_ptr], b_q14[0] as i32);
                ltp_pred_q13 =
                    silk_smlawb(ltp_pred_q13, s_ltp_q15[pred_lag_ptr - 1], b_q14[1] as i32);
                ltp_pred_q13 =
                    silk_smlawb(ltp_pred_q13, s_ltp_q15[pred_lag_ptr - 2], b_q14[2] as i32);
                ltp_pred_q13 =
                    silk_smlawb(ltp_pred_q13, s_ltp_q15[pred_lag_ptr - 3], b_q14[3] as i32);
                ltp_pred_q13 =
                    silk_smlawb(ltp_pred_q13, s_ltp_q15[pred_lag_ptr - 4], b_q14[4] as i32);
                pred_lag_ptr += 1;

                /* Generate LPC excitation */
                pres_q14[_i] = pexc_q14[_i] + (ltp_pred_q13 << 1);

                /* Update states */
                s_ltp_q15[s_ltp_buf_idx] = pres_q14[_i] << 1;
                s_ltp_buf_idx += 1;
                _i += 1;
            }
        } else {
            pres_q14 = pexc_q14;
        }

        let mut _i = 0;
        while _i < ps_dec.subfr_length {
            /* Short-term prediction */
            debug_assert!(ps_dec.lpc_order == 10 || ps_dec.lpc_order == 16);
            /* Avoids introducing a bias because silk_smlawb() always rounds to -inf */
            let mut lpc_pred_q10 = ps_dec.lpc_order as i32 / 2;
            lpc_pred_q10 = silk_smlawb(
                lpc_pred_q10,
                s_lpc_q14[MAX_LPC_ORDER + _i - 1],
                a_q12_tmp[0] as i32,
            );
            lpc_pred_q10 = silk_smlawb(
                lpc_pred_q10,
                s_lpc_q14[MAX_LPC_ORDER + _i - 2],
                a_q12_tmp[1] as i32,
            );
            lpc_pred_q10 = silk_smlawb(
                lpc_pred_q10,
                s_lpc_q14[MAX_LPC_ORDER + _i - 3],
                a_q12_tmp[2] as i32,
            );
            lpc_pred_q10 = silk_smlawb(
                lpc_pred_q10,
                s_lpc_q14[MAX_LPC_ORDER + _i - 4],
                a_q12_tmp[3] as i32,
            );
            lpc_pred_q10 = silk_smlawb(
                lpc_pred_q10,
                s_lpc_q14[MAX_LPC_ORDER + _i - 5],
                a_q12_tmp[4] as i32,
            );
            lpc_pred_q10 = silk_smlawb(
                lpc_pred_q10,
                s_lpc_q14[MAX_LPC_ORDER + _i - 6],
                a_q12_tmp[5] as i32,
            );
            lpc_pred_q10 = silk_smlawb(
                lpc_pred_q10,
                s_lpc_q14[MAX_LPC_ORDER + _i - 7],
                a_q12_tmp[6] as i32,
            );
            lpc_pred_q10 = silk_smlawb(
                lpc_pred_q10,
                s_lpc_q14[MAX_LPC_ORDER + _i - 8],
                a_q12_tmp[7] as i32,
            );
            lpc_pred_q10 = silk_smlawb(
                lpc_pred_q10,
                s_lpc_q14[MAX_LPC_ORDER + _i - 9],
                a_q12_tmp[8] as i32,
            );
            lpc_pred_q10 = silk_smlawb(
                lpc_pred_q10,
                s_lpc_q14[MAX_LPC_ORDER + _i - 10],
                a_q12_tmp[9] as i32,
            );
            if ps_dec.lpc_order == 16 {
                lpc_pred_q10 = silk_smlawb(
                    lpc_pred_q10,
                    s_lpc_q14[MAX_LPC_ORDER + _i - 11],
                    a_q12_tmp[10] as i32,
                );
                lpc_pred_q10 = silk_smlawb(
                    lpc_pred_q10,
                    s_lpc_q14[MAX_LPC_ORDER + _i - 12],
                    a_q12_tmp[11] as i32,
                );
                lpc_pred_q10 = silk_smlawb(
                    lpc_pred_q10,
                    s_lpc_q14[MAX_LPC_ORDER + _i - 13],
                    a_q12_tmp[12] as i32,
                );
                lpc_pred_q10 = silk_smlawb(
                    lpc_pred_q10,
                    s_lpc_q14[MAX_LPC_ORDER + _i - 14],
                    a_q12_tmp[13] as i32,
                );
                lpc_pred_q10 = silk_smlawb(
                    lpc_pred_q10,
                    s_lpc_q14[MAX_LPC_ORDER + _i - 15],
                    a_q12_tmp[14] as i32,
                );
                lpc_pred_q10 = silk_smlawb(
                    lpc_pred_q10,
                    s_lpc_q14[MAX_LPC_ORDER + _i - 16],
                    a_q12_tmp[15] as i32,
                );
            }

            /* Add prediction to LPC excitation */
            s_lpc_q14[MAX_LPC_ORDER + _i] =
                pres_q14[_i].saturating_add(silk_lshift_sat32(lpc_pred_q10, 4));

            /* Scale with gain */

            xq[k * ps_dec.subfr_length + _i] = silk_sat16(silk_rshift_round(
                silk_smulww(s_lpc_q14[MAX_LPC_ORDER + _i], gain_q10),
                8,
            )) as i16;

            _i += 1;
        }

        /* Update LPC filter state */
        s_lpc_q14.copy_within(ps_dec.subfr_length..ps_dec.subfr_length + MAX_LPC_ORDER, 0);
        pexc_q14 = &mut pexc_q14[ps_dec.subfr_length..];
        // pxq = &mut pxq[ps_dec.subfr_length..];
        k += 1;
    }

    /* Save LPC state */
    ps_dec
        .s_lpc_q14_buf
        .copy_from_slice(&s_lpc_q14[..MAX_LPC_ORDER]);
}
