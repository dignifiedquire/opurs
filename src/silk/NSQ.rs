//! Noise shaping quantizer.
//!
//! Upstream C: `silk/NSQ.c`

pub mod typedef_h {
    pub const silk_int16_MAX: i32 = i16::MAX as i32;
    pub const silk_int16_MIN: i32 = i16::MIN as i32;
}
pub mod NSQ_h {
    ///
    /// Short-term prediction using LPC coefficients. `buf32` is indexed as
    /// `buf32[pos], buf32[pos-1], ..., buf32[pos-order+1]` and `coef16` has
    /// `order` entries. Here we take `buf32` as a slice ending at `pos+1`
    /// (i.e. the element at `buf32[buf32.len()-1]` is `buf32[pos]`).
    /// Upstream C: silk/NSQ.h:silk_noise_shape_quantizer_short_prediction_c
    #[inline(always)]
    pub fn silk_noise_shape_quantizer_short_prediction_c(
        buf32: &[i32],
        coef16: &[i16],
        order: i32,
    ) -> i32 {
        // buf32 is indexed backwards from the end: buf32[len-1] = pos, buf32[len-2] = pos-1, etc.
        // Pre-slice to the last 10 elements to hoist bounds checks.
        let b = buf32.len();
        let buf = &buf32[b - 10..];
        let coef = &coef16[..10];
        let mut out: i32 = order >> 1;
        out = (out as i64 + ((buf[9] as i64 * coef[0] as i64) >> 16)) as i32;
        out = (out as i64 + ((buf[8] as i64 * coef[1] as i64) >> 16)) as i32;
        out = (out as i64 + ((buf[7] as i64 * coef[2] as i64) >> 16)) as i32;
        out = (out as i64 + ((buf[6] as i64 * coef[3] as i64) >> 16)) as i32;
        out = (out as i64 + ((buf[5] as i64 * coef[4] as i64) >> 16)) as i32;
        out = (out as i64 + ((buf[4] as i64 * coef[5] as i64) >> 16)) as i32;
        out = (out as i64 + ((buf[3] as i64 * coef[6] as i64) >> 16)) as i32;
        out = (out as i64 + ((buf[2] as i64 * coef[7] as i64) >> 16)) as i32;
        out = (out as i64 + ((buf[1] as i64 * coef[8] as i64) >> 16)) as i32;
        out = (out as i64 + ((buf[0] as i64 * coef[9] as i64) >> 16)) as i32;
        if order == 16 {
            let buf16 = &buf32[b - 16..];
            let coef16 = &coef16[10..16];
            out = (out as i64 + ((buf16[5] as i64 * coef16[0] as i64) >> 16)) as i32;
            out = (out as i64 + ((buf16[4] as i64 * coef16[1] as i64) >> 16)) as i32;
            out = (out as i64 + ((buf16[3] as i64 * coef16[2] as i64) >> 16)) as i32;
            out = (out as i64 + ((buf16[2] as i64 * coef16[3] as i64) >> 16)) as i32;
            out = (out as i64 + ((buf16[1] as i64 * coef16[4] as i64) >> 16)) as i32;
            out = (out as i64 + ((buf16[0] as i64 * coef16[5] as i64) >> 16)) as i32;
        }
        out
    }

    ///
    /// Noise shape feedback loop. `data0` is the new input value,
    /// `data1` is the shift register (length `order`), `coef` has `order` entries.
    /// Shifts new value into data1 while computing the weighted sum.
    /// Upstream C: silk/NSQ.h:silk_NSQ_noise_shape_feedback_loop_c
    #[inline]
    pub fn silk_NSQ_noise_shape_feedback_loop_c(
        data0: i32,
        data1: &mut [i32],
        coef: &[i16],
        order: i32,
    ) -> i32 {
        let n = order as usize;
        let data1 = &mut data1[..n];
        let coef = &coef[..n];
        let mut tmp2 = data0;
        let mut tmp1 = data1[0];
        data1[0] = tmp2;
        let mut out: i32 = order >> 1;
        out = (out as i64 + ((tmp2 as i64 * coef[0] as i64) >> 16)) as i32;
        let mut j = 2usize;
        while j < n {
            tmp2 = data1[j - 1];
            data1[j - 1] = tmp1;
            out = (out as i64 + ((tmp1 as i64 * coef[j - 1] as i64) >> 16)) as i32;
            tmp1 = data1[j];
            data1[j] = tmp2;
            out = (out as i64 + ((tmp2 as i64 * coef[j] as i64) >> 16)) as i32;
            j += 2;
        }
        data1[n - 1] = tmp1;
        out = (out as i64 + ((tmp1 as i64 * coef[n - 1] as i64) >> 16)) as i32;
        out = ((out as u32) << 1) as i32;
        out
    }
}

pub use self::typedef_h::{silk_int16_MAX, silk_int16_MIN};
pub use self::NSQ_h::{
    silk_NSQ_noise_shape_feedback_loop_c, silk_noise_shape_quantizer_short_prediction_c,
};

/// Dispatch wrapper for short prediction — routes to SIMD when available.
#[cfg(feature = "simd")]
#[inline(always)]
pub fn silk_noise_shape_quantizer_short_prediction(
    buf32: &[i32],
    coef16: &[i16],
    order: i32,
    arch: Arch,
) -> i32 {
    super::simd::silk_noise_shape_quantizer_short_prediction(buf32, coef16, order, arch)
}

/// Dispatch wrapper for short prediction (scalar-only build).
#[cfg(not(feature = "simd"))]
#[inline]
pub fn silk_noise_shape_quantizer_short_prediction(
    buf32: &[i32],
    coef16: &[i16],
    order: i32,
    _arch: Arch,
) -> i32 {
    silk_noise_shape_quantizer_short_prediction_c(buf32, coef16, order)
}

/// Dispatch wrapper for noise shape feedback loop — routes to SIMD when available.
#[cfg(feature = "simd")]
#[inline(always)]
pub fn silk_NSQ_noise_shape_feedback_loop(
    data0: i32,
    data1: &mut [i32],
    coef: &[i16],
    order: i32,
    arch: Arch,
) -> i32 {
    super::simd::silk_NSQ_noise_shape_feedback_loop(data0, data1, coef, order, arch)
}

/// Dispatch wrapper for noise shape feedback loop (scalar-only build).
#[cfg(not(feature = "simd"))]
#[inline]
pub fn silk_NSQ_noise_shape_feedback_loop(
    data0: i32,
    data1: &mut [i32],
    coef: &[i16],
    order: i32,
    _arch: Arch,
) -> i32 {
    silk_NSQ_noise_shape_feedback_loop_c(data0, data1, coef, order)
}

use crate::arch::Arch;
#[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
use crate::silk::define::QUANT_LEVEL_ADJUST_Q10;
use crate::silk::define::{
    HARM_SHAPE_FIR_TAPS, LTP_ORDER, MAX_LPC_ORDER, MAX_SHAPE_LPC_ORDER, NSQ_LPC_BUF_LENGTH,
    TYPE_VOICED,
};
use crate::silk::structs::{silk_nsq_state, NsqConfig, SideInfoIndices};
use crate::silk::tables_other::silk_Quantization_Offsets_Q10;
use crate::silk::Inlines::{silk_DIV32_varQ, silk_INVERSE32_varQ};
use crate::silk::LPC_analysis_filter::silk_LPC_analysis_filter;
use crate::silk::SigProc_FIX::silk_RAND;

/// Dispatch wrapper for NSQ, matching upstream `silk_NSQ` RTCD surface.
#[cfg(feature = "simd")]
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn silk_NSQ(
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
    super::simd::silk_NSQ(
        psEncC,
        NSQ,
        psIndices,
        x16,
        pulses,
        PredCoef_Q12,
        LTPCoef_Q14,
        AR_Q13,
        HarmShapeGain_Q14,
        Tilt_Q14,
        LF_shp_Q14,
        Gains_Q16,
        pitchL,
        Lambda_Q10,
        LTP_scale_Q14,
    );
}

/// Scalar-only build wrapper for NSQ.
#[cfg(not(feature = "simd"))]
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn silk_NSQ(
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
    silk_NSQ_c(
        psEncC,
        NSQ,
        psIndices,
        x16,
        pulses,
        PredCoef_Q12,
        LTPCoef_Q14,
        AR_Q13,
        HarmShapeGain_Q14,
        Tilt_Q14,
        LF_shp_Q14,
        Gains_Q16,
        pitchL,
        Lambda_Q10,
        LTP_scale_Q14,
    );
}

/// Upstream C: silk/NSQ.c:silk_NSQ_c
pub fn silk_NSQ_c(
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
    let mut HarmShapeFIRPacked_Q14: i32;

    NSQ.rand_seed = psIndices.Seed as i32;
    lag = NSQ.lagPrev;
    let offset_Q10 = silk_Quantization_Offsets_Q10[(psIndices.signalType as i32 >> 1) as usize]
        [psIndices.quantOffsetType as usize] as i32;

    // Precompute quantization lookup table for SSE4.1 path (x86 only)
    #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
    let (use_simd_quantizer, table) = {
        let use_it = super::simd::use_nsq_sse4_1(psEncC.arch)
            && psEncC.shapingLPCOrder == 10
            && psEncC.predictLPCOrder == 16;
        let table = if use_it {
            build_quantization_table(offset_Q10, Lambda_Q10)
        } else {
            [[0i32; 4]; 64]
        };
        (use_it, table)
    };

    let LSF_interpolation_flag: i32 = if psIndices.NLSFInterpCoef_Q2 as i32 == 4 {
        0
    } else {
        1
    };
    let ltp_mem_len = psEncC.ltp_mem_length;
    let frame_len = psEncC.frame_length;
    let subfr_len = psEncC.subfr_length;

    // ltp_mem_len + frame_len max: 320 + 320 = 640
    const MAX_LTP_FRAME: usize = 640;
    debug_assert!(ltp_mem_len + frame_len <= MAX_LTP_FRAME);
    let mut sLTP_Q15 = [0i32; MAX_LTP_FRAME];
    let mut sLTP = [0i16; MAX_LTP_FRAME];
    // subfr_len max: MAX_SUB_FRAME_LENGTH = 80
    const MAX_SUBFR: usize = 80;
    debug_assert!(subfr_len <= MAX_SUBFR);
    let mut x_sc_Q10 = [0i32; MAX_SUBFR];

    NSQ.sLTP_shp_buf_idx = ltp_mem_len as i32;
    NSQ.sLTP_buf_idx = ltp_mem_len as i32;
    let mut pxq_off: usize = ltp_mem_len;
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
                start_idx =
                    ltp_mem_len as i32 - lag - psEncC.predictLPCOrder - LTP_ORDER as i32 / 2;
                debug_assert!(start_idx > 0);
                silk_LPC_analysis_filter(
                    &mut sLTP[start_idx as usize..ltp_mem_len],
                    &NSQ.xq[(start_idx + k * subfr_len as i32) as usize..]
                        [..(ltp_mem_len - start_idx as usize)],
                    a_Q12,
                );
                NSQ.rewhite_flag = 1;
                NSQ.sLTP_buf_idx = ltp_mem_len as i32;
            }
        }
        silk_nsq_scale_states(
            psEncC,
            NSQ,
            &x16[x16_off..x16_off + subfr_len],
            &mut x_sc_Q10,
            &sLTP,
            &mut sLTP_Q15,
            k,
            LTP_scale_Q14,
            Gains_Q16,
            pitchL,
            psIndices.signalType as i32,
        );
        #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
        {
            if use_simd_quantizer {
                unsafe {
                    super::simd::silk_noise_shape_quantizer_10_16_sse4_1(
                        NSQ,
                        psIndices.signalType as i32,
                        &x_sc_Q10,
                        &mut pulses[pulses_off..pulses_off + subfr_len],
                        pxq_off,
                        &mut sLTP_Q15,
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
                        &table,
                    );
                }
            } else {
                silk_noise_shape_quantizer(
                    NSQ,
                    psIndices.signalType as i32,
                    &x_sc_Q10,
                    &mut pulses[pulses_off..pulses_off + subfr_len],
                    pxq_off,
                    &mut sLTP_Q15,
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
                    psEncC.shapingLPCOrder,
                    psEncC.predictLPCOrder,
                    psEncC.arch,
                );
            }
        }
        #[cfg(not(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64"))))]
        silk_noise_shape_quantizer(
            NSQ,
            psIndices.signalType as i32,
            &x_sc_Q10,
            &mut pulses[pulses_off..pulses_off + subfr_len],
            pxq_off,
            &mut sLTP_Q15,
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
            psEncC.shapingLPCOrder,
            psEncC.predictLPCOrder,
            psEncC.arch,
        );
        x16_off += subfr_len;
        pulses_off += subfr_len;
        pxq_off += subfr_len;
    }
    NSQ.lagPrev = pitchL[psEncC.nb_subfr - 1];
    NSQ.xq.copy_within(frame_len..frame_len + ltp_mem_len, 0);
    NSQ.sLTP_shp_Q14
        .copy_within(frame_len..frame_len + ltp_mem_len, 0);
}

///
/// Core noise-shape quantizer inner loop. Processes one subframe of samples.
/// `xq_off` is the offset into `NSQ.xq` where output samples are written.
/// Upstream C: silk/NSQ.c:silk_noise_shape_quantizer
#[inline]
fn silk_noise_shape_quantizer(
    NSQ: &mut silk_nsq_state,
    signalType: i32,
    x_sc_Q10: &[i32],
    pulses: &mut [i8],
    xq_off: usize,
    sLTP_Q15: &mut [i32],
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
    shapingLPCOrder: i32,
    predictLPCOrder: i32,
    _arch: Arch,
) {
    let mut LTP_pred_Q13: i32;
    let mut LPC_pred_Q10: i32;
    let mut n_AR_Q12: i32;
    let mut n_LTP_Q13: i32;
    let mut n_LF_Q12: i32;
    let mut r_Q10: i32;
    let mut rr_Q10: i32;
    let mut q1_Q0: i32;
    let mut q1_Q10: i32;
    let mut q2_Q10: i32;
    let mut rd1_Q20: i32;
    let mut rd2_Q20: i32;
    let mut exc_Q14: i32;
    let mut LPC_exc_Q14: i32;
    let mut xq_Q14: i32;
    let mut tmp1: i32;
    let mut tmp2: i32;
    let mut sLF_AR_shp_Q14: i32;

    let Gain_Q10: i32 = Gain_Q16 >> 6;
    let length = length as usize;

    // shp_lag_ptr starts at sLTP_shp_buf_idx - lag + HARM_SHAPE_FIR_TAPS/2
    // and advances by 1 each iteration
    let mut shp_lag_idx = (NSQ.sLTP_shp_buf_idx - lag + HARM_SHAPE_FIR_TAPS / 2) as usize;

    // pred_lag_ptr starts at sLTP_buf_idx - lag + LTP_ORDER/2
    // and advances by 1 each iteration
    let mut pred_lag_idx = (NSQ.sLTP_buf_idx - lag + LTP_ORDER as i32 / 2) as usize;

    // psLPC_Q14 starts at sLPC_Q14[NSQ_LPC_BUF_LENGTH - 1] and advances.
    // In the original C code, psLPC_Q14 points into the middle of sLPC_Q14
    // and is indexed backwards for prediction and forward for writing.
    // We use an index `lpc_idx` that tracks the "current" position.
    let mut lpc_idx: usize = NSQ_LPC_BUF_LENGTH - 1;

    // Pre-slice to hoist bounds checks out of the hot loop.
    let x_sc_Q10 = &x_sc_Q10[..length];
    let pulses = &mut pulses[..length];

    for i in 0..length {
        NSQ.rand_seed = silk_RAND(NSQ.rand_seed);

        // LPC prediction: pass slice ending at current position
        LPC_pred_Q10 = silk_noise_shape_quantizer_short_prediction(
            &NSQ.sLPC_Q14[..lpc_idx + 1],
            a_Q12,
            predictLPCOrder,
            _arch,
        );

        // LTP prediction
        if signalType == TYPE_VOICED {
            LTP_pred_Q13 = 2;
            LTP_pred_Q13 = (LTP_pred_Q13 as i64
                + ((sLTP_Q15[pred_lag_idx] as i64 * b_Q14[0] as i64) >> 16))
                as i32;
            LTP_pred_Q13 = (LTP_pred_Q13 as i64
                + ((sLTP_Q15[pred_lag_idx - 1] as i64 * b_Q14[1] as i64) >> 16))
                as i32;
            LTP_pred_Q13 = (LTP_pred_Q13 as i64
                + ((sLTP_Q15[pred_lag_idx - 2] as i64 * b_Q14[2] as i64) >> 16))
                as i32;
            LTP_pred_Q13 = (LTP_pred_Q13 as i64
                + ((sLTP_Q15[pred_lag_idx - 3] as i64 * b_Q14[3] as i64) >> 16))
                as i32;
            LTP_pred_Q13 = (LTP_pred_Q13 as i64
                + ((sLTP_Q15[pred_lag_idx - 4] as i64 * b_Q14[4] as i64) >> 16))
                as i32;
            pred_lag_idx += 1;
        } else {
            LTP_pred_Q13 = 0;
        }

        // Noise shape feedback
        debug_assert!(shapingLPCOrder & 1 == 0);
        n_AR_Q12 = silk_NSQ_noise_shape_feedback_loop(
            NSQ.sDiff_shp_Q14,
            &mut NSQ.sAR2_Q14,
            AR_shp_Q13,
            shapingLPCOrder,
            _arch,
        );

        n_AR_Q12 =
            (n_AR_Q12 as i64 + ((NSQ.sLF_AR_shp_Q14 as i64 * Tilt_Q14 as i16 as i64) >> 16)) as i32;

        n_LF_Q12 = ((NSQ.sLTP_shp_Q14[(NSQ.sLTP_shp_buf_idx - 1) as usize] as i64
            * LF_shp_Q14 as i16 as i64)
            >> 16) as i32;
        n_LF_Q12 = (n_LF_Q12 as i64
            + ((NSQ.sLF_AR_shp_Q14 as i64 * (LF_shp_Q14 as i64 >> 16)) >> 16))
            as i32;

        debug_assert!(lag > 0 || signalType != 2);

        tmp1 = (((LPC_pred_Q10 as u32) << 2) as i32).wrapping_sub(n_AR_Q12);
        tmp1 = tmp1.wrapping_sub(n_LF_Q12);
        if lag > 0 {
            n_LTP_Q13 = (((NSQ.sLTP_shp_Q14[shp_lag_idx]
                .saturating_add(NSQ.sLTP_shp_Q14[shp_lag_idx - 2]))
                as i64
                * HarmShapeFIRPacked_Q14 as i16 as i64)
                >> 16) as i32;
            n_LTP_Q13 = (n_LTP_Q13 as i64
                + ((NSQ.sLTP_shp_Q14[shp_lag_idx - 1] as i64
                    * (HarmShapeFIRPacked_Q14 as i64 >> 16))
                    >> 16)) as i32;
            n_LTP_Q13 = ((n_LTP_Q13 as u32) << 1) as i32;
            shp_lag_idx += 1;
            tmp2 = LTP_pred_Q13 - n_LTP_Q13;
            tmp1 = tmp2.wrapping_add(((tmp1 as u32) << 1) as i32);
            tmp1 = if 3 == 1 {
                (tmp1 >> 1) + (tmp1 & 1)
            } else {
                ((tmp1 >> (3 - 1)) + 1) >> 1
            };
        } else {
            tmp1 = if 2 == 1 {
                (tmp1 >> 1) + (tmp1 & 1)
            } else {
                ((tmp1 >> (2 - 1)) + 1) >> 1
            };
        }

        r_Q10 = x_sc_Q10[i] - tmp1;
        if NSQ.rand_seed < 0 {
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

        // RD selection
        if q1_Q0 > 0 {
            q1_Q10 = ((q1_Q0 as u32) << 10) as i32 - 80;
            q1_Q10 += offset_Q10;
            q2_Q10 = q1_Q10 + 1024;
            rd1_Q20 = q1_Q10 as i16 as i32 * Lambda_Q10 as i16 as i32;
            rd2_Q20 = q2_Q10 as i16 as i32 * Lambda_Q10 as i16 as i32;
        } else if q1_Q0 == 0 {
            q1_Q10 = offset_Q10;
            q2_Q10 = q1_Q10 + (1024 - 80);
            rd1_Q20 = q1_Q10 as i16 as i32 * Lambda_Q10 as i16 as i32;
            rd2_Q20 = q2_Q10 as i16 as i32 * Lambda_Q10 as i16 as i32;
        } else if q1_Q0 == -1 {
            q2_Q10 = offset_Q10;
            q1_Q10 = q2_Q10 - (1024 - 80);
            rd1_Q20 = -q1_Q10 as i16 as i32 * Lambda_Q10 as i16 as i32;
            rd2_Q20 = q2_Q10 as i16 as i32 * Lambda_Q10 as i16 as i32;
        } else {
            q1_Q10 = ((q1_Q0 as u32) << 10) as i32 + 80;
            q1_Q10 += offset_Q10;
            q2_Q10 = q1_Q10 + 1024;
            rd1_Q20 = -q1_Q10 as i16 as i32 * Lambda_Q10 as i16 as i32;
            rd2_Q20 = -q2_Q10 as i16 as i32 * Lambda_Q10 as i16 as i32;
        }
        rr_Q10 = r_Q10 - q1_Q10;
        rd1_Q20 += rr_Q10 as i16 as i32 * rr_Q10 as i16 as i32;
        rr_Q10 = r_Q10 - q2_Q10;
        rd2_Q20 += rr_Q10 as i16 as i32 * rr_Q10 as i16 as i32;
        if rd2_Q20 < rd1_Q20 {
            q1_Q10 = q2_Q10;
        }

        pulses[i] = (if 10 == 1 {
            (q1_Q10 >> 1) + (q1_Q10 & 1)
        } else {
            ((q1_Q10 >> (10 - 1)) + 1) >> 1
        }) as i8;

        // Excitation
        exc_Q14 = ((q1_Q10 as u32) << 4) as i32;
        if NSQ.rand_seed < 0 {
            exc_Q14 = -exc_Q14;
        }
        LPC_exc_Q14 = exc_Q14 + ((LTP_pred_Q13 as u32) << 1) as i32;
        xq_Q14 = LPC_exc_Q14.wrapping_add(((LPC_pred_Q10 as u32) << 4) as i32);

        NSQ.xq[xq_off + i] = (if (if 8 == 1 {
            (((xq_Q14 as i64 * Gain_Q10 as i64) >> 16) as i32 >> 1)
                + (((xq_Q14 as i64 * Gain_Q10 as i64) >> 16) as i32 & 1)
        } else {
            ((((xq_Q14 as i64 * Gain_Q10 as i64) >> 16) as i32 >> (8 - 1)) + 1) >> 1
        }) > silk_int16_MAX
        {
            silk_int16_MAX
        } else if (if 8 == 1 {
            (((xq_Q14 as i64 * Gain_Q10 as i64) >> 16) as i32 >> 1)
                + (((xq_Q14 as i64 * Gain_Q10 as i64) >> 16) as i32 & 1)
        } else {
            ((((xq_Q14 as i64 * Gain_Q10 as i64) >> 16) as i32 >> (8 - 1)) + 1) >> 1
        }) < silk_int16_MIN
        {
            silk_int16_MIN
        } else if 8 == 1 {
            (((xq_Q14 as i64 * Gain_Q10 as i64) >> 16) as i32 >> 1)
                + (((xq_Q14 as i64 * Gain_Q10 as i64) >> 16) as i32 & 1)
        } else {
            ((((xq_Q14 as i64 * Gain_Q10 as i64) >> 16) as i32 >> (8 - 1)) + 1) >> 1
        }) as i16;

        // Update state
        lpc_idx += 1;
        NSQ.sLPC_Q14[lpc_idx] = xq_Q14;

        NSQ.sDiff_shp_Q14 = xq_Q14 - ((x_sc_Q10[i] as u32) << 4) as i32;
        sLF_AR_shp_Q14 = NSQ
            .sDiff_shp_Q14
            .wrapping_sub(((n_AR_Q12 as u32) << 2) as i32);
        NSQ.sLF_AR_shp_Q14 = sLF_AR_shp_Q14;

        NSQ.sLTP_shp_Q14[NSQ.sLTP_shp_buf_idx as usize] =
            sLF_AR_shp_Q14.wrapping_sub(((n_LF_Q12 as u32) << 2) as i32);
        sLTP_Q15[NSQ.sLTP_buf_idx as usize] = ((LPC_exc_Q14 as u32) << 1) as i32;
        NSQ.sLTP_shp_buf_idx += 1;
        NSQ.sLTP_buf_idx += 1;

        NSQ.rand_seed = (NSQ.rand_seed as u32).wrapping_add(pulses[i] as u32) as i32;
    }

    // Copy last NSQ_LPC_BUF_LENGTH values to the beginning
    NSQ.sLPC_Q14
        .copy_within(length..length + NSQ_LPC_BUF_LENGTH, 0);
}

/// Upstream C: silk/NSQ.c:silk_nsq_scale_states
#[inline]
fn silk_nsq_scale_states(
    psEncC: &NsqConfig,
    NSQ: &mut silk_nsq_state,
    x16: &[i16],
    x_sc_Q10: &mut [i32],
    sLTP: &[i16],
    sLTP_Q15: &mut [i32],
    subfr: i32,
    LTP_scale_Q14: i32,
    Gains_Q16: &[i32],
    pitchL: &[i32],
    signal_type: i32,
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
            let end = NSQ.sLTP_buf_idx as usize;
            for val in sLTP_Q15[start..end].iter_mut() {
                *val = ((gain_adj_Q16 as i64 * *val as i64) >> 16) as i32;
            }
        }

        NSQ.sLF_AR_shp_Q14 = ((gain_adj_Q16 as i64 * NSQ.sLF_AR_shp_Q14 as i64) >> 16) as i32;
        NSQ.sDiff_shp_Q14 = ((gain_adj_Q16 as i64 * NSQ.sDiff_shp_Q14 as i64) >> 16) as i32;

        for i in 0..NSQ_LPC_BUF_LENGTH {
            NSQ.sLPC_Q14[i] = ((gain_adj_Q16 as i64 * NSQ.sLPC_Q14[i] as i64) >> 16) as i32;
        }
        for i in 0..MAX_SHAPE_LPC_ORDER as usize {
            NSQ.sAR2_Q14[i] = ((gain_adj_Q16 as i64 * NSQ.sAR2_Q14[i] as i64) >> 16) as i32;
        }

        NSQ.prev_gain_Q16 = Gains_Q16[subfr as usize];
    }
}

/// Build the precomputed quantization lookup table used by the SSE4.1 quantizer.
/// Port of the table initialization from `silk/x86/NSQ_sse4_1.c:silk_NSQ_sse4_1`.
///
/// table[32 + q1_Q0] = [q1_Q10, q2_Q10, 2*(q1_Q10 - q2_Q10), rd1_Q20 - rd2_Q20 + q1² - q2²]
#[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
fn build_quantization_table(offset_Q10: i32, Lambda_Q10: i32) -> [[i32; 4]; 64] {
    let mut table = [[0i32; 4]; 64];

    // q1_Q0 == 0
    {
        let q1_Q10 = offset_Q10;
        let q2_Q10 = offset_Q10 + (1024 - QUANT_LEVEL_ADJUST_Q10);
        let rd1_Q20 = q1_Q10 * Lambda_Q10;
        let rd2_Q20 = q2_Q10 * Lambda_Q10;
        table[32] = [
            q1_Q10,
            q2_Q10,
            2 * (q1_Q10 - q2_Q10),
            (rd1_Q20 - rd2_Q20) + (q1_Q10 * q1_Q10 - q2_Q10 * q2_Q10),
        ];
    }

    // q1_Q0 == -1
    {
        let q1_Q10 = offset_Q10 - (1024 - QUANT_LEVEL_ADJUST_Q10);
        let q2_Q10 = offset_Q10;
        let rd1_Q20 = -q1_Q10 * Lambda_Q10;
        let rd2_Q20 = q2_Q10 * Lambda_Q10;
        table[31] = [
            q1_Q10,
            q2_Q10,
            2 * (q1_Q10 - q2_Q10),
            (rd1_Q20 - rd2_Q20) + (q1_Q10 * q1_Q10 - q2_Q10 * q2_Q10),
        ];
    }

    // q1_Q0 > 0 (k = 1..31)
    for k in 1..=31 {
        let tmp1 = offset_Q10 + (k << 10);
        let q1_Q10 = tmp1 - QUANT_LEVEL_ADJUST_Q10;
        let q2_Q10 = tmp1 - QUANT_LEVEL_ADJUST_Q10 + 1024;
        let rd1_Q20 = q1_Q10 * Lambda_Q10;
        let rd2_Q20 = q2_Q10 * Lambda_Q10;
        table[(32 + k) as usize] = [
            q1_Q10,
            q2_Q10,
            2 * (q1_Q10 - q2_Q10),
            (rd1_Q20 - rd2_Q20) + (q1_Q10 * q1_Q10 - q2_Q10 * q2_Q10),
        ];
    }

    // q1_Q0 < -1 (k = -32..-2)
    for k in -32..=-2 {
        let tmp1 = offset_Q10 + (k << 10);
        let q1_Q10 = tmp1 + QUANT_LEVEL_ADJUST_Q10;
        let q2_Q10 = tmp1 + QUANT_LEVEL_ADJUST_Q10 + 1024;
        let rd1_Q20 = -q1_Q10 * Lambda_Q10;
        let rd2_Q20 = -q2_Q10 * Lambda_Q10;
        table[(32 + k) as usize] = [
            q1_Q10,
            q2_Q10,
            2 * (q1_Q10 - q2_Q10),
            (rd1_Q20 - rd2_Q20) + (q1_Q10 * q1_Q10 - q2_Q10 * q2_Q10),
        ];
    }

    table
}
