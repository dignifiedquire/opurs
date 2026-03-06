//! Packet loss concealment.
//!
//! Upstream c: `silk/PLC.c`

use crate::arch::Arch;

use crate::silk::typedefs::{SILK_INT16_MAX, SILK_INT16_MIN};
// const BWE_COEF: f64 = 0.99;

/// 0.7 in Q14
const V_PITCH_GAIN_START_MIN_Q14: i32 = 11469;
/// 0.95 in Q14
const V_PITCH_GAIN_START_MAX_Q14: i32 = 15565;

pub const RAND_BUF_MASK: i32 = RAND_BUF_SIZE - 1;
pub const RAND_BUF_SIZE: i32 = 128;
use crate::silk::bwexpander::silk_bwexpander;
use crate::silk::define::{
    LTP_ORDER, MAX_FRAME_LENGTH, MAX_LPC_ORDER, MAX_SUB_FRAME_LENGTH, TYPE_VOICED,
};
use crate::silk::inlines::{silk_inverse32_varq, silk_sqrt_approx};
use crate::silk::lpc_analysis_filter::silk_lpc_analysis_filter;
#[cfg(not(feature = "simd"))]
use crate::silk::lpc_inv_pred_gain::silk_lpc_inverse_pred_gain_c;
use crate::silk::macros::{silk_clz32, silk_smlawb, silk_smulbb, silk_smulww};
use crate::silk::sigproc_fix::{
    silk_lshift_sat32, silk_max_16, silk_max_32, silk_max_int, silk_min_32, silk_min_int,
    silk_rand, silk_rshift_round, silk_sat16, SILK_FIX_CONST,
};
#[cfg(feature = "simd")]
use crate::silk::simd::silk_lpc_inverse_pred_gain;
use crate::silk::structs::{silk_decoder_control, silk_decoder_state};
use crate::silk::sum_sqr_shift::silk_sum_sqr_shift;

pub const NB_ATT: i32 = 2;
const HARM_ATT_Q15: [i16; 2] = [32440, 31130];
const PLC_RAND_ATTENUATE_V_Q15: [i16; 2] = [31130, 26214];
const PLC_RAND_ATTENUATE_UV_Q15: [i16; 2] = [32440, 29491];

/// Upstream c: silk/PLC.c:silk_PLC_Reset
pub fn silk_plc_reset(ps_dec: &mut silk_decoder_state) {
    ps_dec.s_plc.pitch_l_q8 = (ps_dec.frame_length as i32) << (8 - 1);
    ps_dec.s_plc.prev_gain_q16[0] = SILK_FIX_CONST!(1, 16);
    ps_dec.s_plc.prev_gain_q16[1] = SILK_FIX_CONST!(1, 16);
    ps_dec.s_plc.subfr_length = 20;
    ps_dec.s_plc.nb_subfr = 2;
}

/// Upstream c: silk/PLC.c:silk_PLC
#[inline]
pub fn silk_plc(
    ps_dec: &mut silk_decoder_state,
    ps_dec_ctrl: &mut silk_decoder_control,
    frame: &mut [i16],
    lost: i32,
    #[cfg(feature = "deep-plc")] lpcnet: Option<&mut crate::dnn::lpcnet::LPCNetPLCState>,
    arch: Arch,
) {
    if ps_dec.fs_k_hz != ps_dec.s_plc.fs_k_hz {
        silk_plc_reset(ps_dec);
        ps_dec.s_plc.fs_k_hz = ps_dec.fs_k_hz;
    }
    if lost != 0 {
        silk_plc_conceal(
            ps_dec,
            ps_dec_ctrl,
            frame,
            #[cfg(feature = "deep-plc")]
            lpcnet,
            arch,
        );
        ps_dec.loss_cnt += 1;
    } else {
        silk_plc_update(ps_dec, ps_dec_ctrl);
        #[cfg(feature = "deep-plc")]
        {
            if let Some(lpcnet) = lpcnet {
                if lpcnet.loaded && ps_dec.s_plc.fs_k_hz == 16 {
                    let subfr_length = ps_dec.subfr_length;
                    for k in (0..ps_dec.nb_subfr).step_by(2) {
                        crate::dnn::lpcnet::lpcnet_plc_update(
                            lpcnet,
                            &frame[k * subfr_length..(k + 2) * subfr_length],
                        );
                    }
                }
            }
        }
    };
}

/// Update state of PLC
///
/// ```text
/// ps_dec       I/O   Decoder state
/// ps_dec_ctrl   I/O   Decoder control
/// ```
#[inline]
fn silk_plc_update(ps_dec: &mut silk_decoder_state, ps_dec_ctrl: &mut silk_decoder_control) {
    let ps_plc = &mut ps_dec.s_plc;

    /* Update parameters used in case of packet loss */
    ps_dec.prev_signal_type = ps_dec.indices.signal_type as i32;
    let mut ltp_gain_q14 = 0;
    if ps_dec.indices.signal_type as i32 == TYPE_VOICED {
        /* Find the parameters for the last subframe which contains a pitch pulse */

        // I hope this translation is correct...
        for j in 0..std::cmp::min(
            (ps_dec_ctrl.pitch_l[ps_dec.nb_subfr - 1] as usize).div_ceil(ps_dec.subfr_length),
            ps_dec.nb_subfr,
        ) {
            let mut temp_ltp_gain_q14 = 0;
            for _i in 0..LTP_ORDER {
                temp_ltp_gain_q14 +=
                    ps_dec_ctrl.ltpcoef_q14[(ps_dec.nb_subfr - 1 - j) * LTP_ORDER + _i] as i32;
            }
            if temp_ltp_gain_q14 > ltp_gain_q14 {
                ltp_gain_q14 = temp_ltp_gain_q14;
                ps_plc.ltpcoef_q14.copy_from_slice(
                    &ps_dec_ctrl.ltpcoef_q14[(ps_dec.nb_subfr - 1 - j) * LTP_ORDER..][..LTP_ORDER],
                );
                ps_plc.pitch_l_q8 =
                    ((ps_dec_ctrl.pitch_l[ps_dec.nb_subfr - 1 - j] as u32) << 8) as i32;
            }
        }

        ps_plc.ltpcoef_q14.fill(0);
        ps_plc.ltpcoef_q14[LTP_ORDER / 2] = ltp_gain_q14 as i16;

        /* Limit LT coefs */
        if ltp_gain_q14 < V_PITCH_GAIN_START_MIN_Q14 {
            let tmp = V_PITCH_GAIN_START_MIN_Q14 << 10;
            let scale_q10 = tmp / std::cmp::max(ltp_gain_q14, 1);
            for _i in 0..LTP_ORDER {
                ps_plc.ltpcoef_q14[_i] =
                    (silk_smulbb(ps_plc.ltpcoef_q14[_i] as i32, scale_q10) >> 10) as i16;
            }
        } else if ltp_gain_q14 > V_PITCH_GAIN_START_MAX_Q14 {
            let tmp_0 = V_PITCH_GAIN_START_MAX_Q14 << 14;
            let scale_q14 = tmp_0 / std::cmp::max(ltp_gain_q14, 1);
            for _i in 0..LTP_ORDER {
                ps_plc.ltpcoef_q14[_i] =
                    (silk_smulbb(ps_plc.ltpcoef_q14[_i] as i32, scale_q14) >> 14) as i16;
            }
        }
    } else {
        ps_plc.pitch_l_q8 = silk_smulbb(ps_dec.fs_k_hz, 18) << 8;
        ps_plc.ltpcoef_q14.fill(0);
    }

    /* Save LPC coeficients */
    ps_plc.prev_lpc_q12[..ps_dec.lpc_order]
        .copy_from_slice(&ps_dec_ctrl.pred_coef_q12[1][..ps_dec.lpc_order]);
    ps_plc.prev_ltp_scale_q14 = ps_dec_ctrl.ltp_scale_q14 as i16;

    /* Save last two gains */
    ps_plc
        .prev_gain_q16
        .copy_from_slice(&ps_dec_ctrl.gains_q16[ps_dec.nb_subfr - 2..][..2]);

    ps_plc.subfr_length = ps_dec.subfr_length as i32;
    ps_plc.nb_subfr = ps_dec.nb_subfr as i32;
}

/// Upstream c: silk/PLC.c:silk_PLC_energy
#[inline]
fn silk_plc_energy(
    energy1: &mut i32,
    shift1: &mut i32,
    energy2: &mut i32,
    shift2: &mut i32,
    exc_q14: &[i32],
    prev_gain_q10: &[i32; 2],
    subfr_length: usize,
    nb_subfr: usize,
) {
    // Max: 2 * subfr_length(80) = 160
    let mut exc_buf = [0i16; 2 * MAX_SUB_FRAME_LENGTH];
    for k in 0..2 {
        let exc_off = (k + nb_subfr - 2) * subfr_length;
        for _i in 0..subfr_length {
            let val = ((exc_q14[_i + exc_off] as i64 * prev_gain_q10[k] as i64) >> 16) as i32 >> 8;
            exc_buf[k * subfr_length + _i] = val.clamp(SILK_INT16_MIN, SILK_INT16_MAX) as i16;
        }
    }
    silk_sum_sqr_shift(energy1, shift1, &exc_buf[..subfr_length]);
    silk_sum_sqr_shift(energy2, shift2, &exc_buf[subfr_length..2 * subfr_length]);
}

/// Upstream c: silk/PLC.c:silk_PLC_conceal
#[inline]
fn silk_plc_conceal(
    ps_dec: &mut silk_decoder_state,
    ps_dec_ctrl: &mut silk_decoder_control,
    frame: &mut [i16],
    #[cfg(feature = "deep-plc")] lpcnet: Option<&mut crate::dnn::lpcnet::LPCNetPLCState>,
    _arch: Arch,
) {
    // Max: ltp_mem_length(320) + frame_length(320) = 640
    let mut s_ltp_q14 = [0i32; 2 * MAX_FRAME_LENGTH];
    let mut s_ltp = [0i16; MAX_FRAME_LENGTH];

    let prev_gain_q10: [i32; 2] = [
        ps_dec.s_plc.prev_gain_q16[0] >> 6,
        ps_dec.s_plc.prev_gain_q16[1] >> 6,
    ];

    if ps_dec.first_frame_after_reset != 0 {
        ps_dec.s_plc.prev_lpc_q12.fill(0);
    }

    let mut energy1: i32 = 0;
    let mut shift1: i32 = 0;
    let mut energy2: i32 = 0;
    let mut shift2: i32 = 0;
    silk_plc_energy(
        &mut energy1,
        &mut shift1,
        &mut energy2,
        &mut shift2,
        &ps_dec.exc_q14,
        &prev_gain_q10,
        ps_dec.subfr_length,
        ps_dec.nb_subfr,
    );

    let ps_plc = &ps_dec.s_plc;
    let rand_off = if energy1 >> shift2 < energy2 >> shift1 {
        /* First sub-frame has lowest energy */
        silk_max_int(
            0,
            (ps_plc.nb_subfr - 1) * ps_plc.subfr_length - RAND_BUF_SIZE,
        ) as usize
    } else {
        /* Second sub-frame has lowest energy */
        silk_max_int(0, ps_plc.nb_subfr * ps_plc.subfr_length - RAND_BUF_SIZE) as usize
    };

    /* Set up Gain to random noise component */
    let mut b_q14: [i16; LTP_ORDER] = ps_dec.s_plc.ltpcoef_q14;
    let mut rand_scale_q14: i16 = ps_dec.s_plc.rand_scale_q14;

    /* Set up attenuation gains */
    let harm_gain_q15 = HARM_ATT_Q15[silk_min_int(NB_ATT - 1, ps_dec.loss_cnt) as usize] as i32;
    let mut rand_gain_q15 = if ps_dec.prev_signal_type == TYPE_VOICED {
        PLC_RAND_ATTENUATE_V_Q15[silk_min_int(NB_ATT - 1, ps_dec.loss_cnt) as usize] as i32
    } else {
        PLC_RAND_ATTENUATE_UV_Q15[silk_min_int(NB_ATT - 1, ps_dec.loss_cnt) as usize] as i32
    };

    /* LPC concealment. Apply BWE to previous LPC */
    silk_bwexpander(
        &mut ps_dec.s_plc.prev_lpc_q12[..ps_dec.lpc_order],
        SILK_FIX_CONST!(0.99, 16),
    );

    /* Preload LPC coefficients to array on stack */
    let mut a_q12 = [0i16; MAX_LPC_ORDER];
    a_q12[..ps_dec.lpc_order].copy_from_slice(&ps_dec.s_plc.prev_lpc_q12[..ps_dec.lpc_order]);

    /* First lost frame */
    if ps_dec.loss_cnt == 0 {
        rand_scale_q14 = (1 << 14) as i16;

        if ps_dec.prev_signal_type == TYPE_VOICED {
            /* Reduce random noise Gain for voiced frames */
            for b in &b_q14[..LTP_ORDER] {
                rand_scale_q14 = (rand_scale_q14 as i32 - *b as i32) as i16;
            }
            rand_scale_q14 = silk_max_16(3277, rand_scale_q14); /* 0.2 */
            rand_scale_q14 = (silk_smulbb(
                rand_scale_q14 as i32,
                ps_dec.s_plc.prev_ltp_scale_q14 as i32,
            ) >> 14) as i16;
        } else {
            /* Reduce random noise for unvoiced frames with high LPC gain */
            let inv_gain_q30 = {
                #[cfg(feature = "simd")]
                {
                    silk_lpc_inverse_pred_gain(
                        &ps_dec.s_plc.prev_lpc_q12[..ps_dec.lpc_order],
                        _arch,
                    )
                }
                #[cfg(not(feature = "simd"))]
                {
                    silk_lpc_inverse_pred_gain_c(&ps_dec.s_plc.prev_lpc_q12[..ps_dec.lpc_order])
                }
            };
            let mut down_scale_q30 = silk_min_32((1i32) << 30 >> 3, inv_gain_q30);
            down_scale_q30 = silk_max_32((1i32) << 30 >> 8, down_scale_q30);
            down_scale_q30 = ((down_scale_q30 as u32) << 3) as i32;
            rand_gain_q15 =
                ((down_scale_q30 as i64 * rand_gain_q15 as i16 as i64) >> 16) as i32 >> 14;
        }
    }

    let mut rand_seed = ps_dec.s_plc.rand_seed;
    let mut lag = silk_rshift_round(ps_dec.s_plc.pitch_l_q8, 8);
    let mut s_ltp_buf_idx = ps_dec.ltp_mem_length;

    /* Rewhiten LTP state */
    let idx = ps_dec.ltp_mem_length as i32 - lag - ps_dec.lpc_order as i32 - LTP_ORDER as i32 / 2;
    debug_assert!(idx > 0);
    let idx = idx as usize;
    silk_lpc_analysis_filter(
        &mut s_ltp[idx..ps_dec.ltp_mem_length],
        &ps_dec.out_buf[idx..ps_dec.ltp_mem_length],
        &a_q12[..ps_dec.lpc_order],
    );

    /* Scale LTP state */
    let mut inv_gain_q30 = silk_inverse32_varq(ps_dec.s_plc.prev_gain_q16[1], 46);
    inv_gain_q30 = inv_gain_q30.min(0x7fffffff >> 1);
    for _i in (idx + ps_dec.lpc_order)..ps_dec.ltp_mem_length {
        s_ltp_q14[_i] = ((inv_gain_q30 as i64 * s_ltp[_i] as i64) >> 16) as i32;
    }

    /***************************/
    /* LTP synthesis filtering */
    /***************************/
    for _k in 0..ps_dec.nb_subfr {
        let pred_lag_base = s_ltp_buf_idx as i32 - lag + LTP_ORDER as i32 / 2;
        for _i in 0..ps_dec.subfr_length {
            /* Unrolled LTP prediction */
            let plp = pred_lag_base as usize + _i;
            let mut ltp_pred_q12 = 2i32;
            ltp_pred_q12 = silk_smlawb(ltp_pred_q12, s_ltp_q14[plp], b_q14[0] as i32);
            ltp_pred_q12 = silk_smlawb(ltp_pred_q12, s_ltp_q14[plp - 1], b_q14[1] as i32);
            ltp_pred_q12 = silk_smlawb(ltp_pred_q12, s_ltp_q14[plp - 2], b_q14[2] as i32);
            ltp_pred_q12 = silk_smlawb(ltp_pred_q12, s_ltp_q14[plp - 3], b_q14[3] as i32);
            ltp_pred_q12 = silk_smlawb(ltp_pred_q12, s_ltp_q14[plp - 4], b_q14[4] as i32);

            /* Generate LPC excitation */
            rand_seed = silk_rand(rand_seed);
            let ridx = (rand_seed >> 25 & RAND_BUF_MASK) as usize;
            let rand_val = ps_dec.exc_q14[rand_off + ridx];
            s_ltp_q14[s_ltp_buf_idx + _i] = (((ltp_pred_q12 as i64
                + ((rand_val as i64 * rand_scale_q14 as i64) >> 16))
                as i32 as u32)
                << 2) as i32;
        }
        s_ltp_buf_idx += ps_dec.subfr_length;

        /* Gradually reduce LTP gain */
        for b in b_q14[..LTP_ORDER].iter_mut() {
            *b = ((harm_gain_q15 as i16 as i32 * *b as i32) >> 15) as i16;
        }
        /* Gradually reduce excitation gain */
        rand_scale_q14 = ((rand_scale_q14 as i32 * rand_gain_q15 as i16 as i32) >> 15) as i16;

        /* Slowly increase pitch lag */
        ps_dec.s_plc.pitch_l_q8 =
            silk_smlawb(ps_dec.s_plc.pitch_l_q8, ps_dec.s_plc.pitch_l_q8, 655);
        ps_dec.s_plc.pitch_l_q8 = silk_min_32(
            ps_dec.s_plc.pitch_l_q8,
            (((18 * ps_dec.fs_k_hz as i16 as i32) as u32) << 8) as i32,
        );
        lag = silk_rshift_round(ps_dec.s_plc.pitch_l_q8, 8);
    }

    /***************************/
    /* LPC synthesis filtering */
    /***************************/
    let s_lpc_off = ps_dec.ltp_mem_length - MAX_LPC_ORDER;

    /* Copy LPC state */
    s_ltp_q14[s_lpc_off..s_lpc_off + MAX_LPC_ORDER]
        .copy_from_slice(&ps_dec.s_lpc_q14_buf[..MAX_LPC_ORDER]);

    debug_assert!(ps_dec.lpc_order >= 10); /* check that unrolling works */
    for (_i, frame_i) in frame.iter_mut().take(ps_dec.frame_length).enumerate() {
        /* Partly unrolled LPC prediction */
        let s = s_lpc_off + MAX_LPC_ORDER + _i;
        let mut lpc_pred_q10 = (ps_dec.lpc_order as i32) >> 1;
        lpc_pred_q10 = silk_smlawb(lpc_pred_q10, s_ltp_q14[s - 1], a_q12[0] as i32);
        lpc_pred_q10 = silk_smlawb(lpc_pred_q10, s_ltp_q14[s - 2], a_q12[1] as i32);
        lpc_pred_q10 = silk_smlawb(lpc_pred_q10, s_ltp_q14[s - 3], a_q12[2] as i32);
        lpc_pred_q10 = silk_smlawb(lpc_pred_q10, s_ltp_q14[s - 4], a_q12[3] as i32);
        lpc_pred_q10 = silk_smlawb(lpc_pred_q10, s_ltp_q14[s - 5], a_q12[4] as i32);
        lpc_pred_q10 = silk_smlawb(lpc_pred_q10, s_ltp_q14[s - 6], a_q12[5] as i32);
        lpc_pred_q10 = silk_smlawb(lpc_pred_q10, s_ltp_q14[s - 7], a_q12[6] as i32);
        lpc_pred_q10 = silk_smlawb(lpc_pred_q10, s_ltp_q14[s - 8], a_q12[7] as i32);
        lpc_pred_q10 = silk_smlawb(lpc_pred_q10, s_ltp_q14[s - 9], a_q12[8] as i32);
        lpc_pred_q10 = silk_smlawb(lpc_pred_q10, s_ltp_q14[s - 10], a_q12[9] as i32);
        for j in 10..ps_dec.lpc_order {
            lpc_pred_q10 = silk_smlawb(lpc_pred_q10, s_ltp_q14[s - j - 1], a_q12[j] as i32);
        }

        /* Add prediction to LPC excitation: silk_ADD_SAT32(x, silk_lshift_sat32(lpc_pred_q10, 4)) */
        s_ltp_q14[s] = s_ltp_q14[s].saturating_add(silk_lshift_sat32(lpc_pred_q10, 4));

        /* Scale with Gain */
        *frame_i = silk_sat16(silk_sat16(silk_rshift_round(
            silk_smulww(s_ltp_q14[s], prev_gain_q10[1]),
            8,
        ))) as i16;
    }

    /* Deep PLC: override traditional PLC output with neural concealment */
    #[cfg(feature = "deep-plc")]
    {
        if let Some(lpcnet) = lpcnet {
            if lpcnet.loaded && ps_dec.s_plc.fs_k_hz == 16 {
                let run_deep_plc = ps_dec.s_plc.enable_deep_plc || lpcnet.fec_fill_pos != 0;
                if run_deep_plc {
                    let subfr_length = ps_dec.subfr_length;
                    for k in (0..ps_dec.nb_subfr).step_by(2) {
                        crate::dnn::lpcnet::lpcnet_plc_conceal(
                            lpcnet,
                            &mut frame[k * subfr_length..(k + 2) * subfr_length],
                            _arch,
                        );
                    }
                    // Reconstruct LPC state from neural PLC output so that
                    // the traditional PLC state stays consistent.
                    let s_lpc_ptr = &mut s_ltp_q14[s_lpc_off..];
                    for _i in 0..ps_dec.frame_length {
                        s_lpc_ptr[MAX_LPC_ORDER + _i] = (0.5
                            + frame[_i] as f32 * ((1 << 24) as f32) / prev_gain_q10[1] as f32)
                            as i32;
                    }
                } else {
                    let subfr_length = ps_dec.subfr_length;
                    for k in (0..ps_dec.nb_subfr).step_by(2) {
                        crate::dnn::lpcnet::lpcnet_plc_update(
                            lpcnet,
                            &frame[k * subfr_length..(k + 2) * subfr_length],
                        );
                    }
                }
            }
        }
    }

    /* Save LPC state */
    ps_dec.s_lpc_q14_buf[..MAX_LPC_ORDER].copy_from_slice(
        &s_ltp_q14
            [s_lpc_off + ps_dec.frame_length..s_lpc_off + ps_dec.frame_length + MAX_LPC_ORDER],
    );

    /**************************************/
    /* Update states                      */
    /**************************************/
    ps_dec.s_plc.rand_seed = rand_seed;
    ps_dec.s_plc.rand_scale_q14 = rand_scale_q14;
    ps_dec_ctrl.pitch_l.fill(lag);
}

/// Upstream c: silk/PLC.c:silk_PLC_glue_frames
pub fn silk_plc_glue_frames(ps_dec: &mut silk_decoder_state, frame: &mut [i16], length: i32) {
    let mut _i: i32;
    let mut energy_shift: i32 = 0;
    let mut energy: i32 = 0;
    let ps_plc = &mut ps_dec.s_plc;
    if ps_dec.loss_cnt != 0 {
        silk_sum_sqr_shift(
            &mut ps_plc.conc_energy,
            &mut ps_plc.conc_energy_shift,
            &frame[..length as usize],
        );
        ps_plc.last_frame_lost = 1;
    } else {
        if ps_plc.last_frame_lost != 0 {
            silk_sum_sqr_shift(&mut energy, &mut energy_shift, &frame[..length as usize]);
            if energy_shift > ps_plc.conc_energy_shift {
                ps_plc.conc_energy >>= energy_shift - ps_plc.conc_energy_shift;
            } else if energy_shift < ps_plc.conc_energy_shift {
                energy >>= ps_plc.conc_energy_shift - energy_shift;
            }
            if energy > ps_plc.conc_energy {
                let mut gain_q16: i32;
                let mut slope_q16: i32;
                let lz = silk_clz32(ps_plc.conc_energy) - 1;
                ps_plc.conc_energy = ((ps_plc.conc_energy as u32) << lz) as i32;
                energy >>= silk_max_32(24 - lz, 0);
                let frac_q24 = ps_plc.conc_energy / (if energy > 1 { energy } else { 1 });
                gain_q16 = ((silk_sqrt_approx(frac_q24) as u32) << 4) as i32;
                slope_q16 = (((1) << 16) - gain_q16) / length;
                slope_q16 = ((slope_q16 as u32) << 2) as i32;
                // When deep-plc is compiled in, skip the energy fade-in for 16 kHz
                // SILK frames — the DNN PLC handles it. Matches upstream c behavior:
                //   #ifdef ENABLE_DEEP_PLC
                //   if ( ps_dec->s_plc.fs_k_hz != 16 )
                //   #endif
                #[cfg(feature = "deep-plc")]
                let do_fade = ps_plc.fs_k_hz != 16;
                #[cfg(not(feature = "deep-plc"))]
                let do_fade = true;
                if do_fade {
                    _i = 0;
                    while _i < length {
                        frame[_i as usize] =
                            ((gain_q16 as i64 * frame[_i as usize] as i64) >> 16) as i32 as i16;
                        gain_q16 += slope_q16;
                        if gain_q16 > (1) << 16 {
                            break;
                        }
                        _i += 1;
                    }
                }
            }
        }
        ps_plc.last_frame_lost = 0;
    };
}
