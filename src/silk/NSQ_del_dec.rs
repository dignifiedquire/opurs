//! Noise shaping quantizer with delayed decision.
//!
//! Upstream C: `silk/NSQ_del_dec.c`

pub mod typedef_h {
    pub const silk_int32_MAX: i32 = i32::MAX;
    pub const silk_int16_MIN: i32 = i16::MIN as i32;
    pub const silk_int16_MAX: i32 = i16::MAX as i32;
}
pub mod NSQ_h {
    pub use crate::silk::NSQ::NSQ_h::silk_noise_shape_quantizer_short_prediction_c;
}
pub use self::typedef_h::{silk_int16_MAX, silk_int16_MIN, silk_int32_MAX};
pub use self::NSQ_h::silk_noise_shape_quantizer_short_prediction_c;
use crate::silk::define::{
    DECISION_DELAY, HARM_SHAPE_FIR_TAPS, LTP_ORDER, MAX_LPC_ORDER, MAX_SHAPE_LPC_ORDER,
    NSQ_LPC_BUF_LENGTH, TYPE_VOICED,
};
use crate::silk::structs::{silk_nsq_state, NsqConfig, SideInfoIndices};
use crate::silk::tables_other::silk_Quantization_Offsets_Q10;
use crate::silk::Inlines::{silk_DIV32_varQ, silk_INVERSE32_varQ};
use crate::silk::LPC_analysis_filter::silk_LPC_analysis_filter;
use crate::silk::SigProc_FIX::{silk_RAND, silk_min_int};

#[derive(Copy, Clone)]
#[repr(C)]
pub struct NSQ_del_dec_struct {
    pub sLPC_Q14: [i32; 96],
    pub RandState: [i32; 40],
    pub Q_Q10: [i32; 40],
    pub Xq_Q14: [i32; 40],
    pub Pred_Q15: [i32; 40],
    pub Shape_Q14: [i32; 40],
    pub sAR2_Q14: [i32; 24],
    pub LF_AR_Q14: i32,
    pub Diff_Q14: i32,
    pub Seed: i32,
    pub SeedInit: i32,
    pub RD_Q10: i32,
}

impl Default for NSQ_del_dec_struct {
    fn default() -> Self {
        Self {
            sLPC_Q14: [0; 96],
            RandState: [0; 40],
            Q_Q10: [0; 40],
            Xq_Q14: [0; 40],
            Pred_Q15: [0; 40],
            Shape_Q14: [0; 40],
            sAR2_Q14: [0; 24],
            LF_AR_Q14: 0,
            Diff_Q14: 0,
            Seed: 0,
            SeedInit: 0,
            RD_Q10: 0,
        }
    }
}

/// Copy all fields of src into dst except sLPC_Q14[0..keep].
/// This matches the C pattern: memcpy(dst+i, src+i, sizeof(struct)-i*sizeof(i32))
/// which copies sLPC_Q14[i..] and all fields after sLPC_Q14.
#[inline]
fn copy_del_dec_state_partial(dst: &mut NSQ_del_dec_struct, src: &NSQ_del_dec_struct, keep: usize) {
    dst.sLPC_Q14[keep..].copy_from_slice(&src.sLPC_Q14[keep..]);
    dst.RandState = src.RandState;
    dst.Q_Q10 = src.Q_Q10;
    dst.Xq_Q14 = src.Xq_Q14;
    dst.Pred_Q15 = src.Pred_Q15;
    dst.Shape_Q14 = src.Shape_Q14;
    dst.sAR2_Q14 = src.sAR2_Q14;
    dst.LF_AR_Q14 = src.LF_AR_Q14;
    dst.Diff_Q14 = src.Diff_Q14;
    dst.Seed = src.Seed;
    dst.SeedInit = src.SeedInit;
    dst.RD_Q10 = src.RD_Q10;
}

#[derive(Copy, Clone)]
#[repr(C)]
#[derive(Default)]
pub struct NSQ_sample_struct {
    pub Q_Q10: i32,
    pub RD_Q10: i32,
    pub xq_Q14: i32,
    pub LF_AR_Q14: i32,
    pub Diff_Q14: i32,
    pub sLTP_shp_Q14: i32,
    pub LPC_exc_Q14: i32,
}

pub type NSQ_sample_pair = [NSQ_sample_struct; 2];

/// Helper: saturating round-shift for xq output: silk_RSHIFT_ROUND + silk_SAT16
#[inline]
fn rshift_round_sat16(val: i32, shift: i32) -> i16 {
    let rounded = if shift == 1 {
        (val >> 1) + (val & 1)
    } else {
        ((val >> (shift - 1)) + 1) >> 1
    };
    if rounded > silk_int16_MAX {
        silk_int16_MAX as i16
    } else if rounded < silk_int16_MIN {
        silk_int16_MIN as i16
    } else {
        rounded as i16
    }
}

/// Upstream C: silk/NSQ_del_dec.c:silk_NSQ_del_dec_c
pub fn silk_NSQ_del_dec_c(
    psEncC: &NsqConfig,
    NSQ: &mut silk_nsq_state,
    psIndices: &mut SideInfoIndices,
    x16: &[i16],
    pulses: &mut [i8],
    PredCoef_Q12: &[i16],
    LTPCoef_Q14: &[i16],
    AR_Q13: &[i16],
    HarmShapeGain_Q14: &[i32],
    Tilt_Q14: &[i32],
    LF_shp_Q14: &[i32],
    Gains_Q16: &[i32],
    pitchL: &[i32],
    Lambda_Q10: i32,
    LTP_scale_Q14: i32,
) {
    let mut lag: i32;
    let mut start_idx: i32;
    let mut Winner_ind: i32;
    let mut subfr: i32;
    let mut last_smple_idx: i32;
    let mut smpl_buf_idx: i32;
    let mut decisionDelay: i32;
    let mut HarmShapeFIRPacked_Q14: i32;
    let mut RDmin_Q10: i32;

    let ltp_mem_len = psEncC.ltp_mem_length;
    let frame_len = psEncC.frame_length;
    let subfr_len = psEncC.subfr_length;
    let nStates = psEncC.nStatesDelayedDecision;

    lag = NSQ.lagPrev;

    let mut psDelDec: Vec<NSQ_del_dec_struct> =
        vec![NSQ_del_dec_struct::default(); nStates as usize];

    #[allow(clippy::needless_range_loop)]
    for k in 0..nStates as usize {
        psDelDec[k].Seed = (k as i32 + psIndices.Seed as i32) & 3;
        psDelDec[k].SeedInit = psDelDec[k].Seed;
        psDelDec[k].RD_Q10 = 0;
        psDelDec[k].LF_AR_Q14 = NSQ.sLF_AR_shp_Q14;
        psDelDec[k].Diff_Q14 = NSQ.sDiff_shp_Q14;
        psDelDec[k].Shape_Q14[0] = NSQ.sLTP_shp_Q14[ltp_mem_len - 1];
        psDelDec[k].sLPC_Q14[..NSQ_LPC_BUF_LENGTH]
            .copy_from_slice(&NSQ.sLPC_Q14[..NSQ_LPC_BUF_LENGTH]);
        psDelDec[k].sAR2_Q14 = NSQ.sAR2_Q14;
    }

    let offset_Q10 = silk_Quantization_Offsets_Q10[(psIndices.signalType as i32 >> 1) as usize]
        [psIndices.quantOffsetType as usize] as i32;
    smpl_buf_idx = 0;
    decisionDelay = silk_min_int(DECISION_DELAY, subfr_len as i32);
    if psIndices.signalType as i32 == TYPE_VOICED {
        for k in 0..psEncC.nb_subfr as i32 {
            decisionDelay =
                silk_min_int(decisionDelay, pitchL[k as usize] - LTP_ORDER as i32 / 2 - 1);
        }
    } else if lag > 0 {
        decisionDelay = silk_min_int(decisionDelay, lag - LTP_ORDER as i32 / 2 - 1);
    }
    let LSF_interpolation_flag: i32 = if psIndices.NLSFInterpCoef_Q2 as i32 == 4 {
        0
    } else {
        1
    };

    let mut sLTP_Q15: Vec<i32> = vec![0; ltp_mem_len + frame_len];
    let mut sLTP: Vec<i16> = vec![0; ltp_mem_len + frame_len];
    let mut x_sc_Q10: Vec<i32> = vec![0; subfr_len];
    let mut delayedGain_Q10: [i32; 40] = [0; 40];

    let mut pxq_off: usize = ltp_mem_len;
    NSQ.sLTP_shp_buf_idx = ltp_mem_len as i32;
    NSQ.sLTP_buf_idx = ltp_mem_len as i32;
    subfr = 0;
    let mut x16_off: usize = 0;
    let mut pulses_off: usize = 0;

    for k in 0..psEncC.nb_subfr as i32 {
        let a_Q12_off = (((k >> 1) | (1 - LSF_interpolation_flag)) * MAX_LPC_ORDER as i32) as usize;
        let a_Q12 = &PredCoef_Q12[a_Q12_off..a_Q12_off + psEncC.predictLPCOrder as usize];
        let b_Q14_off = (k * LTP_ORDER as i32) as usize;
        let b_Q14 = &LTPCoef_Q14[b_Q14_off..b_Q14_off + LTP_ORDER];
        let ar_shp_off = (k * MAX_SHAPE_LPC_ORDER) as usize;
        let ar_shp_Q13 = &AR_Q13[ar_shp_off..ar_shp_off + psEncC.shapingLPCOrder as usize];

        HarmShapeFIRPacked_Q14 = HarmShapeGain_Q14[k as usize] >> 2;
        HarmShapeFIRPacked_Q14 |= (((HarmShapeGain_Q14[k as usize] >> 1) as u32) << 16) as i32;

        NSQ.rewhite_flag = 0;
        if psIndices.signalType as i32 == TYPE_VOICED {
            lag = pitchL[k as usize];
            if k & (3 - ((LSF_interpolation_flag as u32) << 1) as i32) == 0 {
                if k == 2 {
                    // Find winner among delayed decision states
                    RDmin_Q10 = psDelDec[0].RD_Q10;
                    Winner_ind = 0;
                    #[allow(clippy::needless_range_loop)]
                    for i in 1..nStates as usize {
                        if psDelDec[i].RD_Q10 < RDmin_Q10 {
                            RDmin_Q10 = psDelDec[i].RD_Q10;
                            Winner_ind = i as i32;
                        }
                    }
                    // Penalize non-winners
                    #[allow(clippy::needless_range_loop)]
                    for i in 0..nStates as usize {
                        if i as i32 != Winner_ind {
                            psDelDec[i].RD_Q10 += silk_int32_MAX >> 4;
                        }
                    }
                    // Output delayed samples from winner
                    let psDD = &psDelDec[Winner_ind as usize];
                    last_smple_idx = smpl_buf_idx + decisionDelay;
                    for i in 0..decisionDelay {
                        last_smple_idx = (last_smple_idx - 1) % DECISION_DELAY;
                        if last_smple_idx < 0 {
                            last_smple_idx += DECISION_DELAY;
                        }
                        let p_idx = (pulses_off as isize + (i - decisionDelay) as isize) as usize;
                        pulses[p_idx] = (if 10 == 1 {
                            (psDD.Q_Q10[last_smple_idx as usize] >> 1)
                                + (psDD.Q_Q10[last_smple_idx as usize] & 1)
                        } else {
                            ((psDD.Q_Q10[last_smple_idx as usize] >> (10 - 1)) + 1) >> 1
                        }) as i8;
                        let xq_val = (psDD.Xq_Q14[last_smple_idx as usize] as i64
                            * Gains_Q16[1] as i64)
                            >> 16;
                        let xq_idx = (pxq_off as isize + (i - decisionDelay) as isize) as usize;
                        NSQ.xq[xq_idx] = rshift_round_sat16(xq_val as i32, 14);
                        NSQ.sLTP_shp_Q14[(NSQ.sLTP_shp_buf_idx - decisionDelay + i) as usize] =
                            psDD.Shape_Q14[last_smple_idx as usize];
                    }
                    subfr = 0;
                }
                start_idx =
                    ltp_mem_len as i32 - lag - psEncC.predictLPCOrder - LTP_ORDER as i32 / 2;
                assert!(start_idx > 0);
                silk_LPC_analysis_filter(
                    &mut sLTP[start_idx as usize..ltp_mem_len],
                    &NSQ.xq[(start_idx + k * subfr_len as i32) as usize..]
                        [..ltp_mem_len - start_idx as usize],
                    a_Q12,
                );
                NSQ.sLTP_buf_idx = ltp_mem_len as i32;
                NSQ.rewhite_flag = 1;
            }
        }
        silk_nsq_del_dec_scale_states(
            psEncC,
            NSQ,
            &mut psDelDec,
            &x16[x16_off..x16_off + subfr_len],
            &mut x_sc_Q10,
            &sLTP,
            &mut sLTP_Q15,
            k,
            nStates,
            LTP_scale_Q14,
            Gains_Q16,
            pitchL,
            psIndices.signalType as i32,
            decisionDelay,
        );
        let fresh_subfr = subfr;
        subfr += 1;
        silk_noise_shape_quantizer_del_dec(
            NSQ,
            &mut psDelDec,
            psIndices.signalType as i32,
            &x_sc_Q10,
            pulses,
            pulses_off,
            pxq_off,
            &mut sLTP_Q15,
            &mut delayedGain_Q10,
            a_Q12,
            b_Q14,
            ar_shp_Q13,
            lag,
            HarmShapeFIRPacked_Q14,
            Tilt_Q14[k as usize],
            LF_shp_Q14[k as usize],
            Gains_Q16[k as usize],
            Lambda_Q10,
            offset_Q10,
            subfr_len as i32,
            fresh_subfr,
            psEncC.shapingLPCOrder,
            psEncC.predictLPCOrder,
            psEncC.warping_Q16,
            nStates,
            &mut smpl_buf_idx,
            decisionDelay,
            psEncC.arch,
        );
        x16_off += subfr_len;
        pulses_off += subfr_len;
        pxq_off += subfr_len;
    }

    // Find final winner
    RDmin_Q10 = psDelDec[0].RD_Q10;
    Winner_ind = 0;
    #[allow(clippy::needless_range_loop)]
    for k in 1..nStates as usize {
        if psDelDec[k].RD_Q10 < RDmin_Q10 {
            RDmin_Q10 = psDelDec[k].RD_Q10;
            Winner_ind = k as i32;
        }
    }
    let psDD = &psDelDec[Winner_ind as usize];
    psIndices.Seed = psDD.SeedInit as i8;
    last_smple_idx = smpl_buf_idx + decisionDelay;
    let Gain_Q10 = Gains_Q16[psEncC.nb_subfr - 1] >> 6;
    for i in 0..decisionDelay {
        last_smple_idx = (last_smple_idx - 1) % DECISION_DELAY;
        if last_smple_idx < 0 {
            last_smple_idx += DECISION_DELAY;
        }
        let p_idx = (pulses_off as isize + (i - decisionDelay) as isize) as usize;
        pulses[p_idx] = (if 10 == 1 {
            (psDD.Q_Q10[last_smple_idx as usize] >> 1) + (psDD.Q_Q10[last_smple_idx as usize] & 1)
        } else {
            ((psDD.Q_Q10[last_smple_idx as usize] >> (10 - 1)) + 1) >> 1
        }) as i8;
        let xq_val = (psDD.Xq_Q14[last_smple_idx as usize] as i64 * Gain_Q10 as i64) >> 16;
        let xq_idx = (pxq_off as isize + (i - decisionDelay) as isize) as usize;
        NSQ.xq[xq_idx] = rshift_round_sat16(xq_val as i32, 8);
        NSQ.sLTP_shp_Q14[(NSQ.sLTP_shp_buf_idx - decisionDelay + i) as usize] =
            psDD.Shape_Q14[last_smple_idx as usize];
    }

    // Copy winner's state back to NSQ
    NSQ.sLPC_Q14[..NSQ_LPC_BUF_LENGTH]
        .copy_from_slice(&psDD.sLPC_Q14[subfr_len..subfr_len + NSQ_LPC_BUF_LENGTH]);
    NSQ.sAR2_Q14 = psDD.sAR2_Q14;
    NSQ.sLF_AR_shp_Q14 = psDD.LF_AR_Q14;
    NSQ.sDiff_shp_Q14 = psDD.Diff_Q14;
    NSQ.lagPrev = pitchL[psEncC.nb_subfr - 1];

    // Shift buffers
    NSQ.xq.copy_within(frame_len..frame_len + ltp_mem_len, 0);
    NSQ.sLTP_shp_Q14
        .copy_within(frame_len..frame_len + ltp_mem_len, 0);
}

/// Upstream C: silk/NSQ_del_dec.c:silk_noise_shape_quantizer_del_dec
#[inline]
fn silk_noise_shape_quantizer_del_dec(
    NSQ: &mut silk_nsq_state,
    psDelDec: &mut [NSQ_del_dec_struct],
    signalType: i32,
    x_Q10: &[i32],
    pulses: &mut [i8],
    pulses_off: usize,
    xq_off: usize,
    sLTP_Q15: &mut [i32],
    delayedGain_Q10: &mut [i32; 40],
    a_Q12: &[i16],
    b_Q14: &[i16],
    AR_shp_Q13: &[i16],
    lag: i32,
    HarmShapeFIRPacked_Q14: i32,
    Tilt_Q14: i32,
    LF_shp_Q14: i32,
    Gain_Q16: i32,
    Lambda_Q10: i32,
    offset_Q10: i32,
    length: i32,
    subfr: i32,
    shapingLPCOrder: i32,
    predictLPCOrder: i32,
    warping_Q16: i32,
    nStatesDelayedDecision: i32,
    smpl_buf_idx: &mut i32,
    decisionDelay: i32,
    _arch: i32,
) {
    let mut Winner_ind: i32;
    let mut RDmin_ind: i32;
    let mut RDmax_ind: i32;
    let mut last_smple_idx: i32;
    let mut Winner_rand_state: i32;
    let mut LTP_pred_Q14: i32;
    let mut LPC_pred_Q14: i32;
    let mut n_AR_Q14: i32;
    let mut n_LTP_Q14: i32;
    let mut n_LF_Q14: i32;
    let mut r_Q10: i32;
    let mut rr_Q10: i32;
    let mut rd1_Q10: i32;
    let mut rd2_Q10: i32;
    let mut RDmin_Q10: i32;
    let mut RDmax_Q10: i32;
    let mut q1_Q0: i32;
    let mut q1_Q10: i32;
    let mut q2_Q10: i32;
    let mut exc_Q14: i32;
    let mut LPC_exc_Q14: i32;
    let mut xq_Q14: i32;
    let mut tmp1: i32;
    let mut tmp2: i32;
    let mut sLF_AR_shp_Q14: i32;

    assert!(nStatesDelayedDecision > 0);
    let nStates = nStatesDelayedDecision as usize;

    let mut psSampleState: Vec<NSQ_sample_pair> = vec![[NSQ_sample_struct::default(); 2]; nStates];

    let mut shp_lag_idx = (NSQ.sLTP_shp_buf_idx - lag + HARM_SHAPE_FIR_TAPS / 2) as usize;
    let mut pred_lag_idx = (NSQ.sLTP_buf_idx - lag + LTP_ORDER as i32 / 2) as usize;
    let Gain_Q10: i32 = Gain_Q16 >> 6;

    #[allow(clippy::needless_range_loop)]
    for i in 0..length as usize {
        // LTP prediction (shared across all states)
        if signalType == TYPE_VOICED {
            LTP_pred_Q14 = 2;
            LTP_pred_Q14 = (LTP_pred_Q14 as i64
                + ((sLTP_Q15[pred_lag_idx] as i64 * b_Q14[0] as i64) >> 16))
                as i32;
            LTP_pred_Q14 = (LTP_pred_Q14 as i64
                + ((sLTP_Q15[pred_lag_idx - 1] as i64 * b_Q14[1] as i64) >> 16))
                as i32;
            LTP_pred_Q14 = (LTP_pred_Q14 as i64
                + ((sLTP_Q15[pred_lag_idx - 2] as i64 * b_Q14[2] as i64) >> 16))
                as i32;
            LTP_pred_Q14 = (LTP_pred_Q14 as i64
                + ((sLTP_Q15[pred_lag_idx - 3] as i64 * b_Q14[3] as i64) >> 16))
                as i32;
            LTP_pred_Q14 = (LTP_pred_Q14 as i64
                + ((sLTP_Q15[pred_lag_idx - 4] as i64 * b_Q14[4] as i64) >> 16))
                as i32;
            LTP_pred_Q14 = ((LTP_pred_Q14 as u32) << 1) as i32;
            pred_lag_idx += 1;
        } else {
            LTP_pred_Q14 = 0;
        }

        // Harmonic noise shaping (shared)
        if lag > 0 {
            n_LTP_Q14 = (((NSQ.sLTP_shp_Q14[shp_lag_idx] + NSQ.sLTP_shp_Q14[shp_lag_idx - 2])
                as i64
                * HarmShapeFIRPacked_Q14 as i16 as i64)
                >> 16) as i32;
            n_LTP_Q14 = (n_LTP_Q14 as i64
                + ((NSQ.sLTP_shp_Q14[shp_lag_idx - 1] as i64
                    * (HarmShapeFIRPacked_Q14 as i64 >> 16))
                    >> 16)) as i32;
            n_LTP_Q14 = LTP_pred_Q14 - ((n_LTP_Q14 as u32) << 2) as i32;
            shp_lag_idx += 1;
        } else {
            n_LTP_Q14 = 0;
        }

        // Per-state processing
        for k in 0..nStates {
            let psDD = &mut psDelDec[k];
            psDD.Seed = silk_RAND(psDD.Seed);

            // LPC prediction
            let lpc_idx = NSQ_LPC_BUF_LENGTH - 1 + i;
            LPC_pred_Q14 = silk_noise_shape_quantizer_short_prediction_c(
                &psDD.sLPC_Q14[..lpc_idx + 1],
                a_Q12,
                predictLPCOrder,
            );
            LPC_pred_Q14 = ((LPC_pred_Q14 as u32) << 4) as i32;

            // Noise shaping with warping
            assert!(shapingLPCOrder & 1 == 0);
            tmp2 = (psDD.Diff_Q14 as i64
                + ((psDD.sAR2_Q14[0] as i64 * warping_Q16 as i16 as i64) >> 16))
                as i32;
            tmp1 = (psDD.sAR2_Q14[0] as i64
                + (((psDD.sAR2_Q14[1] - tmp2) as i64 * warping_Q16 as i16 as i64) >> 16))
                as i32;
            psDD.sAR2_Q14[0] = tmp2;
            n_AR_Q14 = shapingLPCOrder >> 1;
            n_AR_Q14 = (n_AR_Q14 as i64 + ((tmp2 as i64 * AR_shp_Q13[0] as i64) >> 16)) as i32;

            let mut j = 2;
            while j < shapingLPCOrder {
                tmp2 = (psDD.sAR2_Q14[(j - 1) as usize] as i64
                    + (((psDD.sAR2_Q14[j as usize] - tmp1) as i64 * warping_Q16 as i16 as i64)
                        >> 16)) as i32;
                psDD.sAR2_Q14[(j - 1) as usize] = tmp1;
                n_AR_Q14 = (n_AR_Q14 as i64
                    + ((tmp1 as i64 * AR_shp_Q13[(j - 1) as usize] as i64) >> 16))
                    as i32;
                tmp1 = (psDD.sAR2_Q14[j as usize] as i64
                    + (((psDD.sAR2_Q14[(j + 1) as usize] - tmp2) as i64
                        * warping_Q16 as i16 as i64)
                        >> 16)) as i32;
                psDD.sAR2_Q14[j as usize] = tmp2;
                n_AR_Q14 = (n_AR_Q14 as i64 + ((tmp2 as i64 * AR_shp_Q13[j as usize] as i64) >> 16))
                    as i32;
                j += 2;
            }
            psDD.sAR2_Q14[(shapingLPCOrder - 1) as usize] = tmp1;
            n_AR_Q14 = (n_AR_Q14 as i64
                + ((tmp1 as i64 * AR_shp_Q13[(shapingLPCOrder - 1) as usize] as i64) >> 16))
                as i32;
            n_AR_Q14 = ((n_AR_Q14 as u32) << 1) as i32;
            n_AR_Q14 =
                (n_AR_Q14 as i64 + ((psDD.LF_AR_Q14 as i64 * Tilt_Q14 as i16 as i64) >> 16)) as i32;
            n_AR_Q14 = ((n_AR_Q14 as u32) << 2) as i32;

            n_LF_Q14 = ((psDD.Shape_Q14[*smpl_buf_idx as usize] as i64 * LF_shp_Q14 as i16 as i64)
                >> 16) as i32;
            n_LF_Q14 = (n_LF_Q14 as i64
                + ((psDD.LF_AR_Q14 as i64 * (LF_shp_Q14 as i64 >> 16)) >> 16))
                as i32;
            n_LF_Q14 = ((n_LF_Q14 as u32) << 2) as i32;

            tmp1 = n_AR_Q14 + n_LF_Q14;
            tmp2 = n_LTP_Q14 + LPC_pred_Q14;
            tmp1 = tmp2 - tmp1;
            tmp1 = if 4 == 1 {
                (tmp1 >> 1) + (tmp1 & 1)
            } else {
                ((tmp1 >> (4 - 1)) + 1) >> 1
            };

            r_Q10 = x_Q10[i] - tmp1;
            if psDD.Seed < 0 {
                r_Q10 = -r_Q10;
            }
            r_Q10 = if -((31) << 10) > (30) << 10 {
                if r_Q10 > -((31) << 10) {
                    -((31) << 10)
                } else if r_Q10 < (30) << 10 {
                    (30) << 10
                } else {
                    r_Q10
                }
            } else if r_Q10 > (30) << 10 {
                (30) << 10
            } else if r_Q10 < -((31) << 10) {
                -((31) << 10)
            } else {
                r_Q10
            };

            // Quantize
            q1_Q10 = r_Q10 - offset_Q10;
            q1_Q0 = q1_Q10 >> 10;
            if Lambda_Q10 > 2048 {
                let rdo_offset: i32 = Lambda_Q10 / 2 - 512;
                if q1_Q10 > rdo_offset {
                    q1_Q0 = (q1_Q10 - rdo_offset) >> 10;
                } else if q1_Q10 < -rdo_offset {
                    q1_Q0 = (q1_Q10 + rdo_offset) >> 10;
                } else if q1_Q10 < 0 {
                    q1_Q0 = -1;
                } else {
                    q1_Q0 = 0;
                }
            }
            if q1_Q0 > 0 {
                q1_Q10 = ((q1_Q0 as u32) << 10) as i32 - 80;
                q1_Q10 += offset_Q10;
                q2_Q10 = q1_Q10 + 1024;
                rd1_Q10 = q1_Q10 as i16 as i32 * Lambda_Q10 as i16 as i32;
                rd2_Q10 = q2_Q10 as i16 as i32 * Lambda_Q10 as i16 as i32;
            } else if q1_Q0 == 0 {
                q1_Q10 = offset_Q10;
                q2_Q10 = q1_Q10 + (1024 - 80);
                rd1_Q10 = q1_Q10 as i16 as i32 * Lambda_Q10 as i16 as i32;
                rd2_Q10 = q2_Q10 as i16 as i32 * Lambda_Q10 as i16 as i32;
            } else if q1_Q0 == -1 {
                q2_Q10 = offset_Q10;
                q1_Q10 = q2_Q10 - (1024 - 80);
                rd1_Q10 = -q1_Q10 as i16 as i32 * Lambda_Q10 as i16 as i32;
                rd2_Q10 = q2_Q10 as i16 as i32 * Lambda_Q10 as i16 as i32;
            } else {
                q1_Q10 = ((q1_Q0 as u32) << 10) as i32 + 80;
                q1_Q10 += offset_Q10;
                q2_Q10 = q1_Q10 + 1024;
                rd1_Q10 = -q1_Q10 as i16 as i32 * Lambda_Q10 as i16 as i32;
                rd2_Q10 = -q2_Q10 as i16 as i32 * Lambda_Q10 as i16 as i32;
            }
            rr_Q10 = r_Q10 - q1_Q10;
            rd1_Q10 = (rd1_Q10 + rr_Q10 as i16 as i32 * rr_Q10 as i16 as i32) >> 10;
            rr_Q10 = r_Q10 - q2_Q10;
            rd2_Q10 = (rd2_Q10 + rr_Q10 as i16 as i32 * rr_Q10 as i16 as i32) >> 10;

            if rd1_Q10 < rd2_Q10 {
                psSampleState[k][0].RD_Q10 = psDD.RD_Q10 + rd1_Q10;
                psSampleState[k][1].RD_Q10 = psDD.RD_Q10 + rd2_Q10;
                psSampleState[k][0].Q_Q10 = q1_Q10;
                psSampleState[k][1].Q_Q10 = q2_Q10;
            } else {
                psSampleState[k][0].RD_Q10 = psDD.RD_Q10 + rd2_Q10;
                psSampleState[k][1].RD_Q10 = psDD.RD_Q10 + rd1_Q10;
                psSampleState[k][0].Q_Q10 = q2_Q10;
                psSampleState[k][1].Q_Q10 = q1_Q10;
            }

            // Compute output for best and second-best candidate
            exc_Q14 = ((psSampleState[k][0].Q_Q10 as u32) << 4) as i32;
            if psDD.Seed < 0 {
                exc_Q14 = -exc_Q14;
            }
            LPC_exc_Q14 = exc_Q14 + LTP_pred_Q14;
            xq_Q14 = LPC_exc_Q14 + LPC_pred_Q14;
            psSampleState[k][0].Diff_Q14 = xq_Q14 - ((x_Q10[i] as u32) << 4) as i32;
            sLF_AR_shp_Q14 = psSampleState[k][0].Diff_Q14 - n_AR_Q14;
            psSampleState[k][0].sLTP_shp_Q14 = sLF_AR_shp_Q14 - n_LF_Q14;
            psSampleState[k][0].LF_AR_Q14 = sLF_AR_shp_Q14;
            psSampleState[k][0].LPC_exc_Q14 = LPC_exc_Q14;
            psSampleState[k][0].xq_Q14 = xq_Q14;

            exc_Q14 = ((psSampleState[k][1].Q_Q10 as u32) << 4) as i32;
            if psDD.Seed < 0 {
                exc_Q14 = -exc_Q14;
            }
            LPC_exc_Q14 = exc_Q14 + LTP_pred_Q14;
            xq_Q14 = LPC_exc_Q14 + LPC_pred_Q14;
            psSampleState[k][1].Diff_Q14 = xq_Q14 - ((x_Q10[i] as u32) << 4) as i32;
            sLF_AR_shp_Q14 = psSampleState[k][1].Diff_Q14 - n_AR_Q14;
            psSampleState[k][1].sLTP_shp_Q14 = sLF_AR_shp_Q14 - n_LF_Q14;
            psSampleState[k][1].LF_AR_Q14 = sLF_AR_shp_Q14;
            psSampleState[k][1].LPC_exc_Q14 = LPC_exc_Q14;
            psSampleState[k][1].xq_Q14 = xq_Q14;
        }

        // Update sample buffer index
        *smpl_buf_idx = (*smpl_buf_idx - 1) % DECISION_DELAY;
        if *smpl_buf_idx < 0 {
            *smpl_buf_idx += DECISION_DELAY;
        }
        last_smple_idx = (*smpl_buf_idx + decisionDelay) % DECISION_DELAY;

        // Find winner among best candidates
        RDmin_Q10 = psSampleState[0][0].RD_Q10;
        Winner_ind = 0;
        #[allow(clippy::needless_range_loop)]
        for k in 1..nStates {
            if psSampleState[k][0].RD_Q10 < RDmin_Q10 {
                RDmin_Q10 = psSampleState[k][0].RD_Q10;
                Winner_ind = k as i32;
            }
        }

        // Prune states with different rand state than winner
        Winner_rand_state = psDelDec[Winner_ind as usize].RandState[last_smple_idx as usize];
        for k in 0..nStates {
            if psDelDec[k].RandState[last_smple_idx as usize] != Winner_rand_state {
                psSampleState[k][0].RD_Q10 += 0x7fffffff >> 4;
                psSampleState[k][1].RD_Q10 += 0x7fffffff >> 4;
            }
        }

        // Find worst-best and best-second for state replacement
        RDmax_Q10 = psSampleState[0][0].RD_Q10;
        RDmin_Q10 = psSampleState[0][1].RD_Q10;
        RDmax_ind = 0;
        RDmin_ind = 0;
        #[allow(clippy::needless_range_loop)]
        for k in 1..nStates {
            if psSampleState[k][0].RD_Q10 > RDmax_Q10 {
                RDmax_Q10 = psSampleState[k][0].RD_Q10;
                RDmax_ind = k as i32;
            }
            if psSampleState[k][1].RD_Q10 < RDmin_Q10 {
                RDmin_Q10 = psSampleState[k][1].RD_Q10;
                RDmin_ind = k as i32;
            }
        }

        // Replace worst-best with best-second if beneficial
        if RDmin_Q10 < RDmax_Q10 {
            // Copy state: equivalent to C memcpy from offset i
            // which copies sLPC_Q14[i..] and all subsequent fields
            if RDmax_ind != RDmin_ind {
                let (left, right) = if RDmax_ind < RDmin_ind {
                    let (l, r) = psDelDec.split_at_mut(RDmin_ind as usize);
                    (&mut l[RDmax_ind as usize], &r[0])
                } else {
                    let (l, r) = psDelDec.split_at_mut(RDmax_ind as usize);
                    (&mut r[0], &l[RDmin_ind as usize])
                };
                copy_del_dec_state_partial(left, right, i);
            }
            psSampleState[RDmax_ind as usize][0] = psSampleState[RDmin_ind as usize][1];
        }

        // Output delayed samples
        if subfr > 0 || i as i32 >= decisionDelay {
            let psDD_w = &psDelDec[Winner_ind as usize];
            let out_idx = pulses_off + i - decisionDelay as usize;
            pulses[out_idx] = (if 10 == 1 {
                (psDD_w.Q_Q10[last_smple_idx as usize] >> 1)
                    + (psDD_w.Q_Q10[last_smple_idx as usize] & 1)
            } else {
                ((psDD_w.Q_Q10[last_smple_idx as usize] >> (10 - 1)) + 1) >> 1
            }) as i8;
            let xq_val = (psDD_w.Xq_Q14[last_smple_idx as usize] as i64
                * delayedGain_Q10[last_smple_idx as usize] as i64)
                >> 16;
            NSQ.xq[xq_off + i - decisionDelay as usize] = rshift_round_sat16(xq_val as i32, 8);
            NSQ.sLTP_shp_Q14[(NSQ.sLTP_shp_buf_idx - decisionDelay) as usize] =
                psDD_w.Shape_Q14[last_smple_idx as usize];
            sLTP_Q15[(NSQ.sLTP_buf_idx - decisionDelay) as usize] =
                psDD_w.Pred_Q15[last_smple_idx as usize];
        }
        NSQ.sLTP_shp_buf_idx += 1;
        NSQ.sLTP_buf_idx += 1;

        // Update all states with their best candidate
        for k in 0..nStates {
            let psSS = &psSampleState[k][0];
            let psDD = &mut psDelDec[k];
            psDD.LF_AR_Q14 = psSS.LF_AR_Q14;
            psDD.Diff_Q14 = psSS.Diff_Q14;
            psDD.sLPC_Q14[NSQ_LPC_BUF_LENGTH + i] = psSS.xq_Q14;
            psDD.Xq_Q14[*smpl_buf_idx as usize] = psSS.xq_Q14;
            psDD.Q_Q10[*smpl_buf_idx as usize] = psSS.Q_Q10;
            psDD.Pred_Q15[*smpl_buf_idx as usize] = ((psSS.LPC_exc_Q14 as u32) << 1) as i32;
            psDD.Shape_Q14[*smpl_buf_idx as usize] = psSS.sLTP_shp_Q14;
            psDD.Seed = (psDD.Seed as u32).wrapping_add(
                (if 10 == 1 {
                    (psSS.Q_Q10 >> 1) + (psSS.Q_Q10 & 1)
                } else {
                    ((psSS.Q_Q10 >> (10 - 1)) + 1) >> 1
                }) as u32,
            ) as i32;
            psDD.RandState[*smpl_buf_idx as usize] = psDD.Seed;
            psDD.RD_Q10 = psSS.RD_Q10;
        }
        delayedGain_Q10[*smpl_buf_idx as usize] = Gain_Q10;
    }

    // Copy LPC state for next subframe
    for dd in psDelDec[..nStates].iter_mut() {
        dd.sLPC_Q14
            .copy_within(length as usize..length as usize + NSQ_LPC_BUF_LENGTH, 0);
    }
}

/// Upstream C: silk/NSQ_del_dec.c:silk_nsq_del_dec_scale_states
#[inline]
fn silk_nsq_del_dec_scale_states(
    psEncC: &NsqConfig,
    NSQ: &mut silk_nsq_state,
    psDelDec: &mut [NSQ_del_dec_struct],
    x16: &[i16],
    x_sc_Q10: &mut [i32],
    sLTP: &[i16],
    sLTP_Q15: &mut [i32],
    subfr: i32,
    nStatesDelayedDecision: i32,
    LTP_scale_Q14: i32,
    Gains_Q16: &[i32],
    pitchL: &[i32],
    signal_type: i32,
    decisionDelay: i32,
) {
    let lag = pitchL[subfr as usize];
    let mut inv_gain_Q31 = silk_INVERSE32_varQ(
        if Gains_Q16[subfr as usize] > 1 {
            Gains_Q16[subfr as usize]
        } else {
            1
        },
        47,
    );
    let inv_gain_Q26 = if 5 == 1 {
        (inv_gain_Q31 >> 1) + (inv_gain_Q31 & 1)
    } else {
        ((inv_gain_Q31 >> (5 - 1)) + 1) >> 1
    };

    for i in 0..psEncC.subfr_length {
        x_sc_Q10[i] = ((x16[i] as i64 * inv_gain_Q26 as i64) >> 16) as i32;
    }

    if NSQ.rewhite_flag != 0 {
        if subfr == 0 {
            inv_gain_Q31 = ((((inv_gain_Q31 as i64 * LTP_scale_Q14 as i16 as i64) >> 16) as i32
                as u32)
                << 2) as i32;
        }
        let start = (NSQ.sLTP_buf_idx - lag - LTP_ORDER as i32 / 2) as usize;
        let end = NSQ.sLTP_buf_idx as usize;
        for i in start..end {
            sLTP_Q15[i] = ((inv_gain_Q31 as i64 * sLTP[i] as i64) >> 16) as i32;
        }
    }

    if Gains_Q16[subfr as usize] != NSQ.prev_gain_Q16 {
        let gain_adj_Q16 = silk_DIV32_varQ(NSQ.prev_gain_Q16, Gains_Q16[subfr as usize], 16);

        let shp_start = (NSQ.sLTP_shp_buf_idx - psEncC.ltp_mem_length as i32) as usize;
        let shp_end = NSQ.sLTP_shp_buf_idx as usize;
        for i in shp_start..shp_end {
            NSQ.sLTP_shp_Q14[i] = ((gain_adj_Q16 as i64 * NSQ.sLTP_shp_Q14[i] as i64) >> 16) as i32;
        }

        if signal_type == TYPE_VOICED && NSQ.rewhite_flag == 0 {
            let start = (NSQ.sLTP_buf_idx - lag - LTP_ORDER as i32 / 2) as usize;
            let end = (NSQ.sLTP_buf_idx - decisionDelay) as usize;
            for val in sLTP_Q15[start..end].iter_mut() {
                *val = ((gain_adj_Q16 as i64 * *val as i64) >> 16) as i32;
            }
        }

        for psDD in psDelDec[..nStatesDelayedDecision as usize].iter_mut() {
            psDD.LF_AR_Q14 = ((gain_adj_Q16 as i64 * psDD.LF_AR_Q14 as i64) >> 16) as i32;
            psDD.Diff_Q14 = ((gain_adj_Q16 as i64 * psDD.Diff_Q14 as i64) >> 16) as i32;
            for j in 0..NSQ_LPC_BUF_LENGTH {
                psDD.sLPC_Q14[j] = ((gain_adj_Q16 as i64 * psDD.sLPC_Q14[j] as i64) >> 16) as i32;
            }
            for j in 0..MAX_SHAPE_LPC_ORDER as usize {
                psDD.sAR2_Q14[j] = ((gain_adj_Q16 as i64 * psDD.sAR2_Q14[j] as i64) >> 16) as i32;
            }
            for j in 0..DECISION_DELAY as usize {
                psDD.Pred_Q15[j] = ((gain_adj_Q16 as i64 * psDD.Pred_Q15[j] as i64) >> 16) as i32;
                psDD.Shape_Q14[j] = ((gain_adj_Q16 as i64 * psDD.Shape_Q14[j] as i64) >> 16) as i32;
            }
        }

        NSQ.prev_gain_Q16 = Gains_Q16[subfr as usize];
    }
}
