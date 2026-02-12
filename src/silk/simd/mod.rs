//! SIMD-accelerated SILK functions.
//!
//! This module provides SIMD dispatch for performance-critical SILK functions.
//! On x86/x86_64, runtime CPU feature detection selects SSE4.1/AVX2 paths.
//! On aarch64, NEON is always available and selected at compile time.
//! On other architectures (or with the `simd` feature disabled), falls through to scalar.

// Dispatch functions are wired up to callers incrementally across phases.
#![allow(dead_code)]

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub mod x86;

#[cfg(target_arch = "aarch64")]
pub mod aarch64;

// -- CPU feature detection (x86/x86_64) --

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
cpufeatures::new!(cpuid_sse4_1, "sse4.1");

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
cpufeatures::new!(cpuid_sse2, "sse2");

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
cpufeatures::new!(cpuid_avx2_fma, "avx2", "fma");

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
cpufeatures::new!(cpuid_avx2, "avx2");

// -- Dispatch functions --
// Placeholder dispatchers — implementations are added in later phases.
// For now, all dispatch to scalar.

/// SIMD-accelerated short-term prediction for noise shaping quantizer.
#[inline]
pub fn silk_noise_shape_quantizer_short_prediction(
    buf32: &[i32],
    coef16: &[i16],
    order: i32,
) -> i32 {
    #[cfg(target_arch = "aarch64")]
    {
        return unsafe {
            aarch64::silk_noise_shape_quantizer_short_prediction_neon(buf32, coef16, order)
        };
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if cpuid_sse4_1::get() {
            return unsafe {
                x86::silk_noise_shape_quantizer_short_prediction_sse4_1(buf32, coef16, order)
            };
        }
    }

    #[allow(unreachable_code)]
    {
        super::NSQ::silk_noise_shape_quantizer_short_prediction_c(buf32, coef16, order)
    }
}

/// SIMD-accelerated inner product with scaling for SILK.
#[inline]
pub fn silk_inner_prod_aligned_scale(
    in_vec1: &[i16],
    in_vec2: &[i16],
    scale: i32,
    len: i32,
) -> i32 {
    // Scalar fallback for now — SIMD added in Phase 2
    super::inner_prod_aligned::silk_inner_prod_aligned_scale(in_vec1, in_vec2, scale, len)
}

/// SIMD-accelerated f32→f64 inner product.
#[inline]
pub fn silk_inner_product_flp(data1: &[f32], data2: &[f32]) -> f64 {
    #[cfg(target_arch = "aarch64")]
    {
        return unsafe { aarch64::silk_inner_product_flp_neon(data1, data2) };
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        // C reference only dispatches to AVX2 for this function (not SSE2).
        // SSE2 would change float accumulation order vs scalar, causing bit-inexactness.
        if cpuid_avx2_fma::get() {
            return unsafe { x86::silk_inner_product_flp_avx2(data1, data2) };
        }
    }

    #[allow(unreachable_code)]
    {
        super::float::inner_product_FLP::silk_inner_product_FLP_scalar(data1, data2)
    }
}

/// SIMD-accelerated VAD energy accumulation: sum of (X[i] >> 3)^2.
#[inline]
pub fn silk_vad_energy(x: &[i16]) -> i32 {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if cpuid_sse2::get() {
            return unsafe { x86::silk_vad_energy_sse2(x) };
        }
    }

    // Scalar fallback
    #[allow(unreachable_code)]
    {
        silk_vad_energy_scalar(x)
    }
}

/// Scalar implementation of VAD energy accumulation.
fn silk_vad_energy_scalar(x: &[i16]) -> i32 {
    let mut sum: i32 = 0;
    for &sample in x {
        let x_tmp = (sample as i32) >> 3;
        sum += (x_tmp as i16 as i32) * (x_tmp as i16 as i32);
    }
    sum
}

/// SIMD-accelerated noise shape feedback loop.
/// Dispatches to NEON on aarch64, with scalar fallback.
#[inline]
pub fn silk_NSQ_noise_shape_feedback_loop(
    data0: i32,
    data1: &mut [i32],
    coef: &[i16],
    order: i32,
) -> i32 {
    #[cfg(target_arch = "aarch64")]
    {
        return unsafe {
            aarch64::silk_NSQ_noise_shape_feedback_loop_neon(data0, data1, coef, order)
        };
    }

    #[allow(unreachable_code)]
    {
        super::NSQ::silk_NSQ_noise_shape_feedback_loop_c(data0, data1, coef, order)
    }
}

/// SIMD-accelerated VQ_WMat_EC.
/// Dispatches to SSE4.1 on x86, with scalar fallback.
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn silk_VQ_WMat_EC(
    ind: &mut i8,
    res_nrg_Q15: &mut i32,
    rate_dist_Q8: &mut i32,
    gain_Q7: &mut i32,
    XX_Q17: &[i32],
    xX_Q17: &[i32],
    cb_Q7: &[i8],
    cb_gain_Q7: &[u8],
    cl_Q5: &[u8],
    subfr_len: i32,
    max_gain_Q7: i32,
    L: i32,
) {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if cpuid_sse4_1::get() {
            unsafe {
                x86::silk_VQ_WMat_EC_sse4_1(
                    ind,
                    res_nrg_Q15,
                    rate_dist_Q8,
                    gain_Q7,
                    XX_Q17,
                    xX_Q17,
                    cb_Q7,
                    cb_gain_Q7,
                    cl_Q5,
                    subfr_len,
                    max_gain_Q7,
                    L,
                );
            }
            return;
        }
    }

    #[allow(unreachable_code)]
    {
        super::VQ_WMat_EC::silk_VQ_WMat_EC_c(
            ind,
            res_nrg_Q15,
            rate_dist_Q8,
            gain_Q7,
            XX_Q17,
            xX_Q17,
            cb_Q7,
            cb_gain_Q7,
            cl_Q5,
            subfr_len,
            max_gain_Q7,
            L,
        );
    }
}

/// SIMD-accelerated LPC inverse prediction gain.
/// Dispatches to NEON on aarch64, with scalar fallback.
#[inline]
pub fn silk_LPC_inverse_pred_gain(A_Q12: &[i16]) -> i32 {
    #[cfg(target_arch = "aarch64")]
    {
        return unsafe { aarch64::silk_LPC_inverse_pred_gain_neon(A_Q12) };
    }

    #[allow(unreachable_code)]
    {
        super::LPC_inv_pred_gain::silk_LPC_inverse_pred_gain_c(A_Q12)
    }
}

/// Returns true if the SSE4.1 NSQ quantizer should be used.
/// Requires SSE4.1 and the common LPC order combination (shaping=10, predict=16).
#[inline]
pub fn use_nsq_sse4_1() -> bool {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        return cpuid_sse4_1::get();
    }
    #[allow(unreachable_code)]
    false
}

/// Run the SSE4.1 NSQ inner quantizer (specialized for order 10/16).
///
/// # Safety
/// Caller must verify SSE4.1 is available via `use_nsq_sse4_1()`.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[inline]
pub unsafe fn silk_noise_shape_quantizer_10_16_sse4_1(
    NSQ: &mut super::structs::silk_nsq_state,
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
    table: &[[i32; 4]; 64],
) {
    x86::silk_noise_shape_quantizer_10_16_sse4_1(
        NSQ,
        signalType,
        x_sc_Q10,
        pulses,
        xq_off,
        sLTP_Q15,
        a_Q12,
        b_Q14,
        AR_shp_Q13,
        lag,
        HarmShapeFIRPacked_Q14,
        Tilt_Q14,
        LF_shp_Q14,
        Gain_Q16,
        Lambda_Q10,
        offset_Q10,
        length,
        table,
    );
}

/// Run the SSE4.1 NSQ del_dec scale_states.
///
/// # Safety
/// Caller must verify SSE4.1 is available via `use_nsq_sse4_1()`.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[inline]
#[allow(clippy::too_many_arguments)]
pub unsafe fn silk_nsq_del_dec_scale_states_sse4_1(
    psEncC: &super::structs::NsqConfig,
    NSQ: &mut super::structs::silk_nsq_state,
    psDelDec: &mut [super::NSQ_del_dec::NSQ_del_dec_struct],
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
    x86::silk_nsq_del_dec_scale_states_sse4_1(
        psEncC,
        NSQ,
        psDelDec,
        x16,
        x_sc_Q10,
        sLTP,
        sLTP_Q15,
        subfr,
        nStatesDelayedDecision,
        LTP_scale_Q14,
        Gains_Q16,
        pitchL,
        signal_type,
        decisionDelay,
    );
}

/// Returns true if the AVX2 NSQ del_dec path should be used.
/// Requires AVX2 and nStatesDelayedDecision == 3 or 4.
#[inline]
pub fn use_nsq_del_dec_avx2(n_states: i32) -> bool {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        return cpuid_avx2::get() && (n_states == 3 || n_states == 4);
    }
    #[allow(unreachable_code)]
    {
        let _ = n_states;
        false
    }
}

/// Run the AVX2 NSQ del_dec complete outer function.
///
/// # Safety
/// Caller must verify AVX2 is available and nStatesDelayedDecision is 3 or 4.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[inline]
#[allow(clippy::too_many_arguments)]
pub unsafe fn silk_NSQ_del_dec_avx2(
    psEncC: &super::structs::NsqConfig,
    NSQ: &mut super::structs::silk_nsq_state,
    psIndices: &mut super::structs::SideInfoIndices,
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
    x86::silk_NSQ_del_dec_avx2(
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

/// Run the SSE4.1 NSQ del_dec inner quantizer.
///
/// # Safety
/// Caller must verify SSE4.1 is available via `use_nsq_sse4_1()`.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[inline]
#[allow(clippy::too_many_arguments)]
pub unsafe fn silk_noise_shape_quantizer_del_dec_sse4_1(
    NSQ: &mut super::structs::silk_nsq_state,
    psDelDec: &mut [super::NSQ_del_dec::NSQ_del_dec_struct],
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
) {
    x86::silk_noise_shape_quantizer_del_dec_sse4_1(
        NSQ,
        psDelDec,
        signalType,
        x_Q10,
        pulses,
        pulses_off,
        xq_off,
        sLTP_Q15,
        delayedGain_Q10,
        a_Q12,
        b_Q14,
        AR_shp_Q13,
        lag,
        HarmShapeFIRPacked_Q14,
        Tilt_Q14,
        LF_shp_Q14,
        Gain_Q16,
        Lambda_Q10,
        offset_Q10,
        length,
        subfr,
        shapingLPCOrder,
        predictLPCOrder,
        warping_Q16,
        nStatesDelayedDecision,
        smpl_buf_idx,
        decisionDelay,
    );
}

/// Returns true if the aarch64 NEON NSQ del_dec path should be used.
/// Requires nStatesDelayedDecision to be 3 or 4.
#[inline]
pub fn use_neon_nsq_del_dec(n_states: i32) -> bool {
    #[cfg(target_arch = "aarch64")]
    {
        return n_states > 2 && n_states <= 4;
    }
    #[allow(unreachable_code)]
    {
        let _ = n_states;
        false
    }
}

/// Run the aarch64 NEON NSQ del_dec complete outer function.
///
/// # Safety
/// Caller must verify nStatesDelayedDecision is 3 or 4 (via `use_neon_nsq_del_dec`).
#[cfg(target_arch = "aarch64")]
#[inline]
#[allow(clippy::too_many_arguments)]
pub unsafe fn silk_NSQ_del_dec_neon(
    psEncC: &super::structs::NsqConfig,
    NSQ: &mut super::structs::silk_nsq_state,
    psIndices: &mut super::structs::SideInfoIndices,
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
    aarch64::silk_NSQ_del_dec_neon(
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
