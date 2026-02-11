//! Packet loss concealment.
//!
//! Upstream C: `silk/PLC.c`

// const BWE_COEF: f64 = 0.99;

/// 0.7 in Q14
const V_PITCH_GAIN_START_MIN_Q14: i32 = 11469;
/// 0.95 in Q14
const V_PITCH_GAIN_START_MAX_Q14: i32 = 15565;

pub const RAND_BUF_MASK: i32 = RAND_BUF_SIZE - 1;
pub const RAND_BUF_SIZE: i32 = 128;
pub mod typedef_h {
    pub const silk_int16_MAX: i32 = i16::MAX as i32;
    pub const silk_int16_MIN: i32 = i16::MIN as i32;
}
pub use self::typedef_h::{silk_int16_MAX, silk_int16_MIN};
use crate::silk::bwexpander::silk_bwexpander;
use crate::silk::define::{LTP_ORDER, MAX_LPC_ORDER, TYPE_VOICED};
use crate::silk::macros::{silk_CLZ32, silk_SMLAWB, silk_SMULBB, silk_SMULWW};
use crate::silk::structs::{silk_decoder_control, silk_decoder_state};
use crate::silk::sum_sqr_shift::silk_sum_sqr_shift;
use crate::silk::Inlines::{silk_INVERSE32_varQ, silk_SQRT_APPROX};
use crate::silk::LPC_analysis_filter::silk_LPC_analysis_filter;
use crate::silk::LPC_inv_pred_gain::silk_LPC_inverse_pred_gain_c;
use crate::silk::SigProc_FIX::{
    silk_LSHIFT_SAT32, silk_RAND, silk_RSHIFT_ROUND, silk_SAT16, silk_max_16, silk_max_32,
    silk_max_int, silk_min_32, silk_min_int, SILK_FIX_CONST,
};

pub const NB_ATT: i32 = 2;
static HARM_ATT_Q15: [i16; 2] = [32440, 31130];
static PLC_RAND_ATTENUATE_V_Q15: [i16; 2] = [31130, 26214];
static PLC_RAND_ATTENUATE_UV_Q15: [i16; 2] = [32440, 29491];

pub fn silk_PLC_Reset(psDec: &mut silk_decoder_state) {
    psDec.sPLC.pitchL_Q8 = (psDec.frame_length as i32) << (8 - 1);
    psDec.sPLC.prevGain_Q16[0] = SILK_FIX_CONST!(1, 16);
    psDec.sPLC.prevGain_Q16[1] = SILK_FIX_CONST!(1, 16);
    psDec.sPLC.subfr_length = 20;
    psDec.sPLC.nb_subfr = 2;
}

/// Upstream C: silk/PLC.c:silk_PLC
pub fn silk_PLC(
    psDec: &mut silk_decoder_state,
    psDecCtrl: &mut silk_decoder_control,
    frame: &mut [i16],
    lost: i32,
    arch: i32,
) {
    if psDec.fs_kHz != psDec.sPLC.fs_kHz {
        silk_PLC_Reset(psDec);
        psDec.sPLC.fs_kHz = psDec.fs_kHz;
    }
    if lost != 0 {
        silk_PLC_conceal(psDec, psDecCtrl, frame, arch);
        psDec.lossCnt += 1;
    } else {
        silk_PLC_update(psDec, psDecCtrl);
    };
}

/// Update state of PLC
///
/// ```text
/// psDec       I/O   Decoder state
/// psDecCtrl   I/O   Decoder control
/// ```
#[inline]
fn silk_PLC_update(psDec: &mut silk_decoder_state, psDecCtrl: &mut silk_decoder_control) {
    let psPLC = &mut psDec.sPLC;

    /* Update parameters used in case of packet loss */
    psDec.prevSignalType = psDec.indices.signalType as i32;
    let mut LTP_Gain_Q14 = 0;
    if psDec.indices.signalType as i32 == TYPE_VOICED {
        /* Find the parameters for the last subframe which contains a pitch pulse */

        // I hope this translation is correct...
        for j in 0..std::cmp::min(
            (psDecCtrl.pitchL[psDec.nb_subfr - 1] as usize).div_ceil(psDec.subfr_length),
            psDec.nb_subfr,
        ) {
            let mut temp_LTP_Gain_Q14 = 0;
            for i in 0..LTP_ORDER {
                temp_LTP_Gain_Q14 +=
                    psDecCtrl.LTPCoef_Q14[(psDec.nb_subfr - 1 - j) * LTP_ORDER + i] as i32;
            }
            if temp_LTP_Gain_Q14 > LTP_Gain_Q14 {
                LTP_Gain_Q14 = temp_LTP_Gain_Q14;
                psPLC.LTPCoef_Q14.copy_from_slice(
                    &psDecCtrl.LTPCoef_Q14[(psDec.nb_subfr - 1 - j) * LTP_ORDER..][..LTP_ORDER],
                );
                psPLC.pitchL_Q8 = ((psDecCtrl.pitchL[psDec.nb_subfr - 1 - j] as u32) << 8) as i32;
            }
        }

        psPLC.LTPCoef_Q14.fill(0);
        psPLC.LTPCoef_Q14[LTP_ORDER / 2] = LTP_Gain_Q14 as i16;

        /* Limit LT coefs */
        if LTP_Gain_Q14 < V_PITCH_GAIN_START_MIN_Q14 {
            let tmp = V_PITCH_GAIN_START_MIN_Q14 << 10;
            let scale_Q10 = tmp / std::cmp::max(LTP_Gain_Q14, 1);
            for i in 0..LTP_ORDER {
                psPLC.LTPCoef_Q14[i] =
                    (silk_SMULBB(psPLC.LTPCoef_Q14[i] as i32, scale_Q10) >> 10) as i16;
            }
        } else if LTP_Gain_Q14 > V_PITCH_GAIN_START_MAX_Q14 {
            let tmp_0 = V_PITCH_GAIN_START_MAX_Q14 << 14;
            let scale_Q14 = tmp_0 / std::cmp::max(LTP_Gain_Q14, 1);
            for i in 0..LTP_ORDER {
                psPLC.LTPCoef_Q14[i] =
                    (silk_SMULBB(psPLC.LTPCoef_Q14[i] as i32, scale_Q14) >> 14) as i16;
            }
        }
    } else {
        psPLC.pitchL_Q8 = silk_SMULBB(psDec.fs_kHz, 18) << 8;
        psPLC.LTPCoef_Q14.fill(0);
    }

    /* Save LPC coeficients */
    psPLC.prevLPC_Q12[..psDec.LPC_order]
        .copy_from_slice(&psDecCtrl.PredCoef_Q12[1][..psDec.LPC_order]);
    psPLC.prevLTP_scale_Q14 = psDecCtrl.LTP_scale_Q14 as i16;

    /* Save last two gains */
    psPLC
        .prevGain_Q16
        .copy_from_slice(&psDecCtrl.Gains_Q16[psDec.nb_subfr - 2..][..2]);

    psPLC.subfr_length = psDec.subfr_length as i32;
    psPLC.nb_subfr = psDec.nb_subfr as i32;
}

/// Upstream C: silk/PLC.c:silk_PLC_energy
#[inline]
fn silk_PLC_energy(
    energy1: &mut i32,
    shift1: &mut i32,
    energy2: &mut i32,
    shift2: &mut i32,
    exc_Q14: &[i32],
    prevGain_Q10: &[i32; 2],
    subfr_length: usize,
    nb_subfr: usize,
) {
    let mut exc_buf: Vec<i16> = vec![0; 2 * subfr_length];
    for k in 0..2 {
        let exc_off = (k + nb_subfr - 2) * subfr_length;
        for i in 0..subfr_length {
            let val = ((exc_Q14[i + exc_off] as i64 * prevGain_Q10[k] as i64) >> 16) as i32 >> 8;
            exc_buf[k * subfr_length + i] = val.clamp(silk_int16_MIN, silk_int16_MAX) as i16;
        }
    }
    silk_sum_sqr_shift(energy1, shift1, &exc_buf[..subfr_length]);
    silk_sum_sqr_shift(energy2, shift2, &exc_buf[subfr_length..]);
}

/// Upstream C: silk/PLC.c:silk_PLC_conceal
#[inline]
fn silk_PLC_conceal(
    psDec: &mut silk_decoder_state,
    psDecCtrl: &mut silk_decoder_control,
    frame: &mut [i16],
    _arch: i32,
) {
    let mut sLTP_Q14: Vec<i32> = vec![0; psDec.ltp_mem_length + psDec.frame_length];
    let mut sLTP: Vec<i16> = vec![0; psDec.ltp_mem_length];

    let prevGain_Q10: [i32; 2] = [
        psDec.sPLC.prevGain_Q16[0] >> 6,
        psDec.sPLC.prevGain_Q16[1] >> 6,
    ];

    if psDec.first_frame_after_reset != 0 {
        psDec.sPLC.prevLPC_Q12.fill(0);
    }

    let mut energy1: i32 = 0;
    let mut shift1: i32 = 0;
    let mut energy2: i32 = 0;
    let mut shift2: i32 = 0;
    silk_PLC_energy(
        &mut energy1,
        &mut shift1,
        &mut energy2,
        &mut shift2,
        &psDec.exc_Q14,
        &prevGain_Q10,
        psDec.subfr_length,
        psDec.nb_subfr,
    );

    let psPLC = &psDec.sPLC;
    let rand_off = if energy1 >> shift2 < energy2 >> shift1 {
        /* First sub-frame has lowest energy */
        silk_max_int(0, (psPLC.nb_subfr - 1) * psPLC.subfr_length - RAND_BUF_SIZE) as usize
    } else {
        /* Second sub-frame has lowest energy */
        silk_max_int(0, psPLC.nb_subfr * psPLC.subfr_length - RAND_BUF_SIZE) as usize
    };

    /* Set up Gain to random noise component */
    let mut B_Q14: [i16; LTP_ORDER] = psDec.sPLC.LTPCoef_Q14;
    let mut rand_scale_Q14: i16 = psDec.sPLC.randScale_Q14;

    /* Set up attenuation gains */
    let harm_Gain_Q15 = HARM_ATT_Q15[silk_min_int(NB_ATT - 1, psDec.lossCnt) as usize] as i32;
    let mut rand_Gain_Q15 = if psDec.prevSignalType == TYPE_VOICED {
        PLC_RAND_ATTENUATE_V_Q15[silk_min_int(NB_ATT - 1, psDec.lossCnt) as usize] as i32
    } else {
        PLC_RAND_ATTENUATE_UV_Q15[silk_min_int(NB_ATT - 1, psDec.lossCnt) as usize] as i32
    };

    /* LPC concealment. Apply BWE to previous LPC */
    silk_bwexpander(
        &mut psDec.sPLC.prevLPC_Q12[..psDec.LPC_order],
        SILK_FIX_CONST!(0.99, 16),
    );

    /* Preload LPC coefficients to array on stack */
    let mut A_Q12 = [0i16; MAX_LPC_ORDER];
    A_Q12[..psDec.LPC_order].copy_from_slice(&psDec.sPLC.prevLPC_Q12[..psDec.LPC_order]);

    /* First lost frame */
    if psDec.lossCnt == 0 {
        rand_scale_Q14 = (1 << 14) as i16;

        if psDec.prevSignalType == TYPE_VOICED {
            /* Reduce random noise Gain for voiced frames */
            for b in &B_Q14[..LTP_ORDER] {
                rand_scale_Q14 = (rand_scale_Q14 as i32 - *b as i32) as i16;
            }
            rand_scale_Q14 = silk_max_16(3277, rand_scale_Q14); /* 0.2 */
            rand_scale_Q14 =
                (silk_SMULBB(rand_scale_Q14 as i32, psDec.sPLC.prevLTP_scale_Q14 as i32) >> 14)
                    as i16;
        } else {
            /* Reduce random noise for unvoiced frames with high LPC gain */
            let invGain_Q30 =
                silk_LPC_inverse_pred_gain_c(&psDec.sPLC.prevLPC_Q12[..psDec.LPC_order]);
            let mut down_scale_Q30 = silk_min_32((1i32) << 30 >> 3, invGain_Q30);
            down_scale_Q30 = silk_max_32((1i32) << 30 >> 8, down_scale_Q30);
            down_scale_Q30 = ((down_scale_Q30 as u32) << 3) as i32;
            rand_Gain_Q15 =
                ((down_scale_Q30 as i64 * rand_Gain_Q15 as i16 as i64) >> 16) as i32 >> 14;
        }
    }

    let mut rand_seed = psDec.sPLC.rand_seed;
    let mut lag = silk_RSHIFT_ROUND(psDec.sPLC.pitchL_Q8, 8);
    let mut sLTP_buf_idx = psDec.ltp_mem_length;

    /* Rewhiten LTP state */
    let idx = psDec.ltp_mem_length as i32 - lag - psDec.LPC_order as i32 - LTP_ORDER as i32 / 2;
    assert!(idx > 0);
    let idx = idx as usize;
    silk_LPC_analysis_filter(
        &mut sLTP[idx..psDec.ltp_mem_length],
        &psDec.outBuf[idx..psDec.ltp_mem_length],
        &A_Q12[..psDec.LPC_order],
    );

    /* Scale LTP state */
    let mut inv_gain_Q30 = silk_INVERSE32_varQ(psDec.sPLC.prevGain_Q16[1], 46);
    inv_gain_Q30 = inv_gain_Q30.min(0x7fffffff >> 1);
    for i in (idx + psDec.LPC_order)..psDec.ltp_mem_length {
        sLTP_Q14[i] = ((inv_gain_Q30 as i64 * sLTP[i] as i64) >> 16) as i32;
    }

    /***************************/
    /* LTP synthesis filtering */
    /***************************/
    for _k in 0..psDec.nb_subfr {
        let pred_lag_base = sLTP_buf_idx as i32 - lag + LTP_ORDER as i32 / 2;
        for i in 0..psDec.subfr_length {
            /* Unrolled LTP prediction */
            let plp = pred_lag_base as usize + i;
            let mut LTP_pred_Q12 = 2i32;
            LTP_pred_Q12 = silk_SMLAWB(LTP_pred_Q12, sLTP_Q14[plp], B_Q14[0] as i32);
            LTP_pred_Q12 = silk_SMLAWB(LTP_pred_Q12, sLTP_Q14[plp - 1], B_Q14[1] as i32);
            LTP_pred_Q12 = silk_SMLAWB(LTP_pred_Q12, sLTP_Q14[plp - 2], B_Q14[2] as i32);
            LTP_pred_Q12 = silk_SMLAWB(LTP_pred_Q12, sLTP_Q14[plp - 3], B_Q14[3] as i32);
            LTP_pred_Q12 = silk_SMLAWB(LTP_pred_Q12, sLTP_Q14[plp - 4], B_Q14[4] as i32);

            /* Generate LPC excitation */
            rand_seed = silk_RAND(rand_seed);
            let ridx = (rand_seed >> 25 & RAND_BUF_MASK) as usize;
            let rand_val = psDec.exc_Q14[rand_off + ridx];
            sLTP_Q14[sLTP_buf_idx + i] = (((LTP_pred_Q12 as i64
                + ((rand_val as i64 * rand_scale_Q14 as i64) >> 16))
                as i32 as u32)
                << 2) as i32;
        }
        sLTP_buf_idx += psDec.subfr_length;

        /* Gradually reduce LTP gain */
        for b in B_Q14[..LTP_ORDER].iter_mut() {
            *b = ((harm_Gain_Q15 as i16 as i32 * *b as i32) >> 15) as i16;
        }
        /* Gradually reduce excitation gain */
        rand_scale_Q14 = ((rand_scale_Q14 as i32 * rand_Gain_Q15 as i16 as i32) >> 15) as i16;

        /* Slowly increase pitch lag */
        psDec.sPLC.pitchL_Q8 = silk_SMLAWB(psDec.sPLC.pitchL_Q8, psDec.sPLC.pitchL_Q8, 655);
        psDec.sPLC.pitchL_Q8 = silk_min_32(
            psDec.sPLC.pitchL_Q8,
            (((18 * psDec.fs_kHz as i16 as i32) as u32) << 8) as i32,
        );
        lag = silk_RSHIFT_ROUND(psDec.sPLC.pitchL_Q8, 8);
    }

    /***************************/
    /* LPC synthesis filtering */
    /***************************/
    let sLPC_off = psDec.ltp_mem_length - MAX_LPC_ORDER;

    /* Copy LPC state */
    sLTP_Q14[sLPC_off..sLPC_off + MAX_LPC_ORDER]
        .copy_from_slice(&psDec.sLPC_Q14_buf[..MAX_LPC_ORDER]);

    assert!(psDec.LPC_order >= 10); /* check that unrolling works */
    #[allow(clippy::needless_range_loop)]
    for i in 0..psDec.frame_length {
        /* Partly unrolled LPC prediction */
        let s = sLPC_off + MAX_LPC_ORDER + i;
        let mut LPC_pred_Q10 = (psDec.LPC_order as i32) >> 1;
        LPC_pred_Q10 = silk_SMLAWB(LPC_pred_Q10, sLTP_Q14[s - 1], A_Q12[0] as i32);
        LPC_pred_Q10 = silk_SMLAWB(LPC_pred_Q10, sLTP_Q14[s - 2], A_Q12[1] as i32);
        LPC_pred_Q10 = silk_SMLAWB(LPC_pred_Q10, sLTP_Q14[s - 3], A_Q12[2] as i32);
        LPC_pred_Q10 = silk_SMLAWB(LPC_pred_Q10, sLTP_Q14[s - 4], A_Q12[3] as i32);
        LPC_pred_Q10 = silk_SMLAWB(LPC_pred_Q10, sLTP_Q14[s - 5], A_Q12[4] as i32);
        LPC_pred_Q10 = silk_SMLAWB(LPC_pred_Q10, sLTP_Q14[s - 6], A_Q12[5] as i32);
        LPC_pred_Q10 = silk_SMLAWB(LPC_pred_Q10, sLTP_Q14[s - 7], A_Q12[6] as i32);
        LPC_pred_Q10 = silk_SMLAWB(LPC_pred_Q10, sLTP_Q14[s - 8], A_Q12[7] as i32);
        LPC_pred_Q10 = silk_SMLAWB(LPC_pred_Q10, sLTP_Q14[s - 9], A_Q12[8] as i32);
        LPC_pred_Q10 = silk_SMLAWB(LPC_pred_Q10, sLTP_Q14[s - 10], A_Q12[9] as i32);
        for j in 10..psDec.LPC_order {
            LPC_pred_Q10 = silk_SMLAWB(LPC_pred_Q10, sLTP_Q14[s - j - 1], A_Q12[j] as i32);
        }

        /* Add prediction to LPC excitation: silk_ADD_SAT32(x, silk_LSHIFT_SAT32(LPC_pred_Q10, 4)) */
        sLTP_Q14[s] = sLTP_Q14[s].saturating_add(silk_LSHIFT_SAT32(LPC_pred_Q10, 4));

        /* Scale with Gain */
        frame[i] = silk_SAT16(silk_SAT16(silk_RSHIFT_ROUND(
            silk_SMULWW(sLTP_Q14[s], prevGain_Q10[1]),
            8,
        ))) as i16;
    }

    /* Save LPC state */
    psDec.sLPC_Q14_buf[..MAX_LPC_ORDER].copy_from_slice(
        &sLTP_Q14[sLPC_off + psDec.frame_length..sLPC_off + psDec.frame_length + MAX_LPC_ORDER],
    );

    /**************************************/
    /* Update states                      */
    /**************************************/
    psDec.sPLC.rand_seed = rand_seed;
    psDec.sPLC.randScale_Q14 = rand_scale_Q14;
    psDecCtrl.pitchL.fill(lag);
}

/// Upstream C: silk/PLC.c:silk_PLC_glue_frames
pub fn silk_PLC_glue_frames(psDec: &mut silk_decoder_state, frame: &mut [i16], length: i32) {
    let mut i: i32;
    let mut energy_shift: i32 = 0;
    let mut energy: i32 = 0;
    let psPLC = &mut psDec.sPLC;
    if psDec.lossCnt != 0 {
        silk_sum_sqr_shift(
            &mut psPLC.conc_energy,
            &mut psPLC.conc_energy_shift,
            &frame[..length as usize],
        );
        psPLC.last_frame_lost = 1;
    } else {
        if psPLC.last_frame_lost != 0 {
            silk_sum_sqr_shift(&mut energy, &mut energy_shift, &frame[..length as usize]);
            if energy_shift > psPLC.conc_energy_shift {
                psPLC.conc_energy >>= energy_shift - psPLC.conc_energy_shift;
            } else if energy_shift < psPLC.conc_energy_shift {
                energy >>= psPLC.conc_energy_shift - energy_shift;
            }
            if energy > psPLC.conc_energy {
                let mut gain_Q16: i32;
                let mut slope_Q16: i32;
                let LZ = silk_CLZ32(psPLC.conc_energy) - 1;
                psPLC.conc_energy = ((psPLC.conc_energy as u32) << LZ) as i32;
                energy >>= silk_max_32(24 - LZ, 0);
                let frac_Q24 = psPLC.conc_energy / (if energy > 1 { energy } else { 1 });
                gain_Q16 = ((silk_SQRT_APPROX(frac_Q24) as u32) << 4) as i32;
                slope_Q16 = (((1) << 16) - gain_Q16) / length;
                slope_Q16 = ((slope_Q16 as u32) << 2) as i32;
                i = 0;
                while i < length {
                    frame[i as usize] =
                        ((gain_Q16 as i64 * frame[i as usize] as i64) >> 16) as i32 as i16;
                    gain_Q16 += slope_Q16;
                    if gain_Q16 > (1) << 16 {
                        break;
                    }
                    i += 1;
                }
            }
        }
        psPLC.last_frame_lost = 0;
    };
}
