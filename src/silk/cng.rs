//! Comfort noise generation.
//!
//! Upstream c: `silk/CNG.c`

use crate::silk::define::{
    CNG_BUF_MASK_MAX, CNG_GAIN_SMTH_Q16, CNG_GAIN_SMTH_THRESHOLD_Q16, CNG_NLSF_SMTH_Q16,
    MAX_FRAME_LENGTH, MAX_LPC_ORDER, TYPE_NO_VOICE_ACTIVITY,
};
use crate::silk::inlines::silk_sqrt_approx;
use crate::silk::macros::{silk_smlawb, silk_smulwb, silk_smulww};
use crate::silk::nlsf2a::silk_nlsf2a;
use crate::silk::sigproc_fix::{
    silk_lshift_sat32, silk_rand, silk_rshift_round, silk_sat16, silk_smultt,
};
use crate::silk::structs::{silk_CNG_struct, silk_decoder_control, silk_decoder_state};

/// Generates excitation for CNG LPC synthesis
///
/// ```text
/// exc_q14[]       O     CNG excitation signal Q10
/// exc_buf_q14[]   I     Random samples buffer Q10
/// length          I     Length
/// rand_seed       I/O   seed to random index generator
/// ```
#[inline]
/// Upstream c: silk/CNG.c:silk_CNG_exc
fn silk_cng_exc(exc_q14: &mut [i32], exc_buf_q14: &[i32], rand_seed: &mut i32) {
    let mut exc_mask = CNG_BUF_MASK_MAX;
    while exc_mask > exc_q14.len() as i32 {
        exc_mask >>= 1;
    }

    let mut seed = *rand_seed;
    let mut _i = 0;
    while _i < exc_q14.len() {
        seed = silk_rand(seed);
        let idx = (seed >> 24) & exc_mask;
        debug_assert!(idx >= 0);
        debug_assert!(idx <= CNG_BUF_MASK_MAX);
        exc_q14[_i] = exc_buf_q14[idx as usize];
        _i += 1;
    }
    *rand_seed = seed;
}

/// Upstream c: silk/CNG.c:silk_CNG_Reset
pub fn silk_cng_reset(ps_dec: &mut silk_decoder_state) {
    let nlsf_step_q15 = i16::MAX as i32 / (ps_dec.lpc_order as i32 + 1);
    let mut nlsf_acc_q15 = 0;
    for _i in 0..ps_dec.lpc_order {
        nlsf_acc_q15 += nlsf_step_q15;
        ps_dec.s_cng.cng_smth_nlsf_q15[_i] = nlsf_acc_q15 as i16;
    }
    ps_dec.s_cng.cng_smth_gain_q16 = 0;
    ps_dec.s_cng.rand_seed = 3176576;
}

/// Updates CNG estimate, and applies the CNG when packet was lost
///
/// ```text
/// ps_dec         I/O   Decoder state
/// ps_dec_ctrl     I/O   Decoder control
/// frame[]       I/O   Signal
/// length        I     Length of residual
/// ```
#[inline]
/// Upstream c: silk/CNG.c:silk_CNG
pub fn silk_cng(
    ps_dec: &mut silk_decoder_state,
    ps_dec_ctrl: &mut silk_decoder_control,
    frame: &mut [i16],
) {
    if ps_dec.fs_k_hz != ps_dec.s_cng.fs_k_hz {
        /* Reset state */
        silk_cng_reset(ps_dec);

        ps_dec.s_cng.fs_k_hz = ps_dec.fs_k_hz;
    }
    let ps_cng: &mut silk_CNG_struct = &mut ps_dec.s_cng;
    if ps_dec.loss_cnt == 0 && ps_dec.prev_signal_type == TYPE_NO_VOICE_ACTIVITY {
        /* Update CNG parameters */

        /* Smoothing of LSF's  */
        for _i in 0..ps_dec.lpc_order {
            ps_cng.cng_smth_nlsf_q15[_i] += silk_smulwb(
                ps_dec.prev_nlsf_q15[_i] as i32 - ps_cng.cng_smth_nlsf_q15[_i] as i32,
                CNG_NLSF_SMTH_Q16,
            ) as i16;
        }
        /* Find the subframe with the highest gain */
        let mut max_gain_q16 = 0;
        let mut subfr = 0;

        for _i in 0..ps_dec.nb_subfr {
            if ps_dec_ctrl.gains_q16[_i] > max_gain_q16 {
                max_gain_q16 = ps_dec_ctrl.gains_q16[_i];
                subfr = _i;
            }
        }
        /* Update CNG excitation buffer with excitation from this subframe */
        ps_cng.cng_exc_buf_q14.copy_within(
            0..(ps_dec.nb_subfr - 1) * ps_dec.subfr_length,
            ps_dec.subfr_length,
        );
        ps_cng.cng_exc_buf_q14[..ps_dec.subfr_length]
            .copy_from_slice(&ps_dec.exc_q14[subfr * ps_dec.subfr_length..][..ps_dec.subfr_length]);

        /* Smooth gains */
        for _i in 0..ps_dec.nb_subfr {
            ps_cng.cng_smth_gain_q16 += silk_smulwb(
                ps_dec_ctrl.gains_q16[_i] - ps_cng.cng_smth_gain_q16,
                CNG_GAIN_SMTH_Q16,
            );
            if silk_smulww(ps_cng.cng_smth_gain_q16, CNG_GAIN_SMTH_THRESHOLD_Q16)
                > ps_dec_ctrl.gains_q16[_i]
            {
                ps_cng.cng_smth_gain_q16 = ps_dec_ctrl.gains_q16[_i];
            }
        }
    }

    /* Add CNG when packet is lost or during DTX */
    if ps_dec.loss_cnt != 0 {
        // Max: frame_length(320) + MAX_LPC_ORDER(16) = 336
        let mut cng_sig_q14 = [0i32; MAX_FRAME_LENGTH + MAX_LPC_ORDER];

        /* Generate CNG excitation */
        let mut gain_q16 = silk_smulww(
            ps_dec.s_plc.rand_scale_q14 as i32,
            ps_dec.s_plc.prev_gain_q16[1],
        );
        if gain_q16 >= (1 << 21) || ps_cng.cng_smth_gain_q16 > (1 << 23) {
            gain_q16 = silk_smultt(gain_q16, gain_q16);
            gain_q16 =
                silk_smultt(ps_cng.cng_smth_gain_q16, ps_cng.cng_smth_gain_q16) - (gain_q16 << 5);
            gain_q16 = silk_sqrt_approx(gain_q16) << 16;
        } else {
            gain_q16 = silk_smulww(gain_q16, gain_q16);
            gain_q16 =
                silk_smulww(ps_cng.cng_smth_gain_q16, ps_cng.cng_smth_gain_q16) - (gain_q16 << 5);
            gain_q16 = silk_sqrt_approx(gain_q16) << 8;
        }
        let gain_q10 = gain_q16 >> 6;
        silk_cng_exc(
            &mut cng_sig_q14[MAX_LPC_ORDER..MAX_LPC_ORDER + frame.len()],
            &ps_cng.cng_exc_buf_q14[..frame.len()],
            &mut ps_cng.rand_seed,
        );

        let mut a_q12: [i16; MAX_LPC_ORDER] = [0; 16];

        /* Convert CNG nlsf to filter representation */
        silk_nlsf2a(
            &mut a_q12[..ps_dec.lpc_order],
            &ps_cng.cng_smth_nlsf_q15[..ps_dec.lpc_order],
            ps_dec.arch,
        );

        /* Generate CNG signal, by synthesis filtering */
        cng_sig_q14[..MAX_LPC_ORDER].copy_from_slice(&ps_cng.cng_synth_state);
        debug_assert!(ps_dec.lpc_order == 10 || ps_dec.lpc_order == 16);
        for _i in 0..frame.len() {
            /* Avoids introducing a bias because silk_smlawb() always rounds to -inf */
            let mut lpc_pred_q10 = ps_dec.lpc_order as i32 >> 1;
            lpc_pred_q10 = silk_smlawb(
                lpc_pred_q10,
                cng_sig_q14[MAX_LPC_ORDER + _i - 1],
                a_q12[0] as i32,
            );
            lpc_pred_q10 = silk_smlawb(
                lpc_pred_q10,
                cng_sig_q14[MAX_LPC_ORDER + _i - 2],
                a_q12[1] as i32,
            );
            lpc_pred_q10 = silk_smlawb(
                lpc_pred_q10,
                cng_sig_q14[MAX_LPC_ORDER + _i - 3],
                a_q12[2] as i32,
            );
            lpc_pred_q10 = silk_smlawb(
                lpc_pred_q10,
                cng_sig_q14[MAX_LPC_ORDER + _i - 4],
                a_q12[3] as i32,
            );
            lpc_pred_q10 = silk_smlawb(
                lpc_pred_q10,
                cng_sig_q14[MAX_LPC_ORDER + _i - 5],
                a_q12[4] as i32,
            );
            lpc_pred_q10 = silk_smlawb(
                lpc_pred_q10,
                cng_sig_q14[MAX_LPC_ORDER + _i - 6],
                a_q12[5] as i32,
            );
            lpc_pred_q10 = silk_smlawb(
                lpc_pred_q10,
                cng_sig_q14[MAX_LPC_ORDER + _i - 7],
                a_q12[6] as i32,
            );
            lpc_pred_q10 = silk_smlawb(
                lpc_pred_q10,
                cng_sig_q14[MAX_LPC_ORDER + _i - 8],
                a_q12[7] as i32,
            );
            lpc_pred_q10 = silk_smlawb(
                lpc_pred_q10,
                cng_sig_q14[MAX_LPC_ORDER + _i - 9],
                a_q12[8] as i32,
            );
            lpc_pred_q10 = silk_smlawb(
                lpc_pred_q10,
                cng_sig_q14[MAX_LPC_ORDER + _i - 10],
                a_q12[9] as i32,
            );
            if ps_dec.lpc_order == 16 {
                lpc_pred_q10 = silk_smlawb(
                    lpc_pred_q10,
                    cng_sig_q14[MAX_LPC_ORDER + _i - 11],
                    a_q12[10] as i32,
                );
                lpc_pred_q10 = silk_smlawb(
                    lpc_pred_q10,
                    cng_sig_q14[MAX_LPC_ORDER + _i - 12],
                    a_q12[11] as i32,
                );
                lpc_pred_q10 = silk_smlawb(
                    lpc_pred_q10,
                    cng_sig_q14[MAX_LPC_ORDER + _i - 13],
                    a_q12[12] as i32,
                );
                lpc_pred_q10 = silk_smlawb(
                    lpc_pred_q10,
                    cng_sig_q14[MAX_LPC_ORDER + _i - 14],
                    a_q12[13] as i32,
                );
                lpc_pred_q10 = silk_smlawb(
                    lpc_pred_q10,
                    cng_sig_q14[MAX_LPC_ORDER + _i - 15],
                    a_q12[14] as i32,
                );
                lpc_pred_q10 = silk_smlawb(
                    lpc_pred_q10,
                    cng_sig_q14[MAX_LPC_ORDER + _i - 16],
                    a_q12[15] as i32,
                );
            }

            /* Update states */
            cng_sig_q14[MAX_LPC_ORDER + _i] =
                cng_sig_q14[MAX_LPC_ORDER + _i].saturating_add(silk_lshift_sat32(lpc_pred_q10, 4));

            /* Scale with Gain and add to input signal */
            frame[_i] = frame[_i].saturating_add(silk_sat16(silk_rshift_round(
                silk_smulww(cng_sig_q14[MAX_LPC_ORDER + _i], gain_q10),
                8,
            )) as i16);
        }
        ps_cng
            .cng_synth_state
            .copy_from_slice(&cng_sig_q14[frame.len()..][..MAX_LPC_ORDER]);
    } else {
        ps_cng.cng_synth_state[..ps_dec.lpc_order].fill(0);
    };
}
