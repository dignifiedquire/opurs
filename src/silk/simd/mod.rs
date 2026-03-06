//! SIMD-accelerated SILK functions.
//!
//! This module provides SIMD dispatch for performance-critical SILK functions.
//! On x86/x86_64, runtime CPU feature detection selects SSE4.1/AVX2 paths.
//! On aarch64, NEON is always available and selected at compile time.
//! On other architectures (or with the `simd` feature disabled), falls through to scalar.

use crate::arch::Arch;

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub mod x86;

#[cfg(target_arch = "aarch64")]
pub mod aarch64;

// -- Dispatch functions --
// Placeholder dispatchers — implementations are added in later phases.
// For now, all dispatch to scalar.

/// SIMD-accelerated short-term prediction for noise shaping quantizer.
/// Dispatches to NEON on aarch64, SSE4.1 on x86, with scalar fallback.
#[inline(always)]
pub fn silk_noise_shape_quantizer_short_prediction(
    buf32: &[i32],
    coef16: &[i16],
    order: i32,
    arch: Arch,
) -> i32 {
    #[cfg(target_arch = "aarch64")]
    if arch.has_neon() {
        return unsafe {
            aarch64::silk_noise_shape_quantizer_short_prediction_neon(buf32, coef16, order)
        };
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if arch.has_sse4_1() {
        return unsafe {
            x86::silk_noise_shape_quantizer_short_prediction_sse4_1(buf32, coef16, order)
        };
    }

    let _ = arch;
    super::nsq::silk_noise_shape_quantizer_short_prediction_c(buf32, coef16, order)
}

/// SIMD-accelerated f32→f64 inner product.
/// Dispatches to AVX2 on x86, with scalar fallback on other targets.
///
/// Upstream only overrides this path on x86 AVX2.
#[inline]
pub fn silk_inner_product_flp(data1: &[f32], data2: &[f32], arch: Arch) -> f64 {
    #[cfg(target_arch = "aarch64")]
    if arch.has_neon() {
        return unsafe { aarch64::silk_inner_product_flp_neon(data1, data2) };
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if arch.has_avx2() {
        return unsafe { x86::silk_inner_product_flp_avx2(data1, data2) };
    }

    let _ = arch;
    super::float::inner_product_flp::silk_inner_product_flp_scalar(data1, data2)
}

/// SIMD-accelerated VAD energy accumulation: sum of (X\[_i\] >> 3)^2.
/// Dispatches to SSE2 on x86, with scalar fallback.
#[inline]
pub fn silk_vad_energy(x: &[i16], arch: Arch) -> i32 {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if arch.has_sse2() {
        return unsafe { x86::silk_vad_energy_sse2(x) };
    }

    let _ = arch;
    silk_vad_energy_scalar(x)
}

/// Full-function VAD dispatch, matching upstream RTCD `silk_vad_get_sa_q8`.
#[inline]
pub fn silk_vad_get_sa_q8(ps_enc_c: &mut super::structs::silk_encoder_state, p_in: &[i16]) -> i32 {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if ps_enc_c.arch.has_sse4_1() {
        return unsafe { x86::silk_vad_get_sa_q8_sse4_1(ps_enc_c, p_in) };
    }

    super::vad::silk_vad_get_sa_q8_c(ps_enc_c, p_in)
}

/// Scalar implementation of VAD energy accumulation.
pub fn silk_vad_energy_scalar(x: &[i16]) -> i32 {
    let mut sum: i32 = 0;
    for &sample in x {
        let x_tmp = (sample as i32) >> 3;
        sum += (x_tmp as i16 as i32) * (x_tmp as i16 as i32);
    }
    sum
}

/// SIMD-accelerated noise shape feedback loop.
/// Dispatches to NEON on aarch64, with scalar fallback.
#[inline(always)]
pub fn silk_nsq_noise_shape_feedback_loop(
    data0: i32,
    data1: &mut [i32],
    coef: &[i16],
    order: i32,
    arch: Arch,
) -> i32 {
    #[cfg(target_arch = "aarch64")]
    if arch.has_neon() {
        return unsafe {
            aarch64::silk_nsq_noise_shape_feedback_loop_neon(data0, data1, coef, order)
        };
    }

    let _ = arch;
    super::nsq::silk_nsq_noise_shape_feedback_loop_c(data0, data1, coef, order)
}

/// SIMD-accelerated VQ_WMat_EC.
/// Dispatches to SSE4.1 on x86, with scalar fallback.
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn silk_vq_wmat_ec(
    ind: &mut i8,
    res_nrg_q15: &mut i32,
    rate_dist_q8: &mut i32,
    gain_q7: &mut i32,
    xx_q17: &[i32],
    x_x_q17: &[i32],
    cb_q7: &[i8],
    cb_gain_q7: &[u8],
    cl_q5: &[u8],
    subfr_len: i32,
    max_gain_q7: i32,
    l: i32,
    arch: Arch,
) {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if arch.has_sse4_1() {
        unsafe {
            x86::silk_vq_wmat_ec_sse4_1(
                ind,
                res_nrg_q15,
                rate_dist_q8,
                gain_q7,
                xx_q17,
                x_x_q17,
                cb_q7,
                cb_gain_q7,
                cl_q5,
                subfr_len,
                max_gain_q7,
                l,
            );
        }
        return;
    }

    let _ = arch;
    let vq = super::vq_wmat_ec::silk_vq_wmat_ec_c(&super::vq_wmat_ec::SilkVqWmatEcParams {
        xx_q17,
        x_x_q17,
        cb_q7,
        cb_gain_q7,
        cl_q5,
        subfr_len,
        max_gain_q7,
        l,
    });
    *ind = vq.ind;
    *res_nrg_q15 = vq.res_nrg_q15;
    *rate_dist_q8 = vq.rate_dist_q8;
    *gain_q7 = vq.gain_q7;
}

/// SIMD-accelerated LPC inverse prediction gain.
/// Dispatches to NEON on aarch64, with scalar fallback.
#[inline]
pub fn silk_lpc_inverse_pred_gain(a_q12: &[i16], arch: Arch) -> i32 {
    #[cfg(target_arch = "aarch64")]
    if arch.has_neon() {
        return unsafe { aarch64::silk_lpc_inverse_pred_gain_neon(a_q12) };
    }

    let _ = arch;
    super::lpc_inv_pred_gain::silk_lpc_inverse_pred_gain_c(a_q12)
}

/// Returns true if the SSE4.1 nsq quantizer should be used.
#[inline]
pub fn use_nsq_sse4_1(arch: Arch) -> bool {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        arch.has_sse4_1()
    }
    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    {
        let _ = arch;
        false
    }
}

/// Full-function nsq dispatch, matching upstream RTCD `silk_nsq`.
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn silk_nsq(
    ps_enc_c: &super::structs::NsqConfig,
    nsq: &mut super::structs::silk_nsq_state,
    ps_indices: &mut super::structs::SideInfoIndices,
    x16: &[i16],
    pulses: &mut [i8],
    pred_coef_q12: &[i16],
    ltpcoef_q14: &[i16],
    ar_q13: &[i16],
    harm_shape_gain_q14: &[i32],
    tilt_q14: &[i32],
    lf_shp_q14: &[i32],
    gains_q16: &[i32],
    pitch_l: &[i32],
    lambda_q10: i32,
    ltp_scale_q14: i32,
) {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if ps_enc_c.arch.has_sse4_1() {
        unsafe {
            x86::silk_nsq_sse4_1(
                ps_enc_c,
                nsq,
                ps_indices,
                x16,
                pulses,
                pred_coef_q12,
                ltpcoef_q14,
                ar_q13,
                harm_shape_gain_q14,
                tilt_q14,
                lf_shp_q14,
                gains_q16,
                pitch_l,
                lambda_q10,
                ltp_scale_q14,
            );
        }
        return;
    }

    super::nsq::silk_nsq_c(
        ps_enc_c,
        nsq,
        ps_indices,
        x16,
        pulses,
        pred_coef_q12,
        ltpcoef_q14,
        ar_q13,
        harm_shape_gain_q14,
        tilt_q14,
        lf_shp_q14,
        gains_q16,
        pitch_l,
        lambda_q10,
        ltp_scale_q14,
    );
}

/// Full-function nsq-del-dec dispatch, matching upstream RTCD `silk_nsq_del_dec`.
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn silk_nsq_del_dec(
    ps_enc_c: &super::structs::NsqConfig,
    nsq: &mut super::structs::silk_nsq_state,
    ps_indices: &mut super::structs::SideInfoIndices,
    x16: &[i16],
    pulses: &mut [i8],
    pred_coef_q12: &[i16],
    ltpcoef_q14: &[i16],
    ar_q13: &[i16],
    harm_shape_gain_q14: &[i32],
    tilt_q14: &[i32],
    lf_shp_q14: &[i32],
    gains_q16: &[i32],
    pitch_l: &[i32],
    lambda_q10: i32,
    ltp_scale_q14: i32,
) {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if ps_enc_c.arch.has_avx2()
            && use_nsq_del_dec_avx2(ps_enc_c.arch, ps_enc_c.n_states_delayed_decision)
        {
            unsafe {
                x86::silk_nsq_del_dec_avx2(
                    ps_enc_c,
                    nsq,
                    ps_indices,
                    x16,
                    pulses,
                    pred_coef_q12,
                    ltpcoef_q14,
                    ar_q13,
                    harm_shape_gain_q14,
                    tilt_q14,
                    lf_shp_q14,
                    gains_q16,
                    pitch_l,
                    lambda_q10,
                    ltp_scale_q14,
                );
            }
            return;
        }
        if ps_enc_c.arch.has_sse4_1() {
            unsafe {
                x86::silk_nsq_del_dec_sse4_1(
                    ps_enc_c,
                    nsq,
                    ps_indices,
                    x16,
                    pulses,
                    pred_coef_q12,
                    ltpcoef_q14,
                    ar_q13,
                    harm_shape_gain_q14,
                    tilt_q14,
                    lf_shp_q14,
                    gains_q16,
                    pitch_l,
                    lambda_q10,
                    ltp_scale_q14,
                );
            }
            return;
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if use_neon_nsq_del_dec(ps_enc_c.arch, ps_enc_c.n_states_delayed_decision) {
            unsafe {
                aarch64::silk_nsq_del_dec_neon(
                    ps_enc_c,
                    nsq,
                    ps_indices,
                    x16,
                    pulses,
                    pred_coef_q12,
                    ltpcoef_q14,
                    ar_q13,
                    harm_shape_gain_q14,
                    tilt_q14,
                    lf_shp_q14,
                    gains_q16,
                    pitch_l,
                    lambda_q10,
                    ltp_scale_q14,
                );
            }
            return;
        }
    }

    super::nsq_del_dec::silk_nsq_del_dec_c(
        ps_enc_c,
        nsq,
        ps_indices,
        x16,
        pulses,
        pred_coef_q12,
        ltpcoef_q14,
        ar_q13,
        harm_shape_gain_q14,
        tilt_q14,
        lf_shp_q14,
        gains_q16,
        pitch_l,
        lambda_q10,
        ltp_scale_q14,
    );
}

/// Run the SSE4.1 nsq inner quantizer (specialized for order 10/16).
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn silk_noise_shape_quantizer_10_16_sse4_1(
    nsq: &mut super::structs::silk_nsq_state,
    signal_type: i32,
    x_sc_q10: &[i32],
    pulses: &mut [i8],
    xq_off: usize,
    s_ltp_q15: &mut [i32],
    a_q12: &[i16],
    b_q14: &[i16],
    ar_shp_q13: &[i16],
    lag: i32,
    HarmShapeFIRPacked_Q14: i32,
    tilt_q14: i32,
    lf_shp_q14: i32,
    Gain_Q16: i32,
    lambda_q10: i32,
    offset_q10: i32,
    length: i32,
    table: &[[i32; 4]; 64],
) {
    // SAFETY: call sites gate this wrapper with `use_nsq_sse4_1`.
    unsafe {
        x86::silk_noise_shape_quantizer_10_16_sse4_1(
            nsq,
            signal_type,
            x_sc_q10,
            pulses,
            xq_off,
            s_ltp_q15,
            a_q12,
            b_q14,
            ar_shp_q13,
            lag,
            HarmShapeFIRPacked_Q14,
            tilt_q14,
            lf_shp_q14,
            Gain_Q16,
            lambda_q10,
            offset_q10,
            length,
            table,
        );
    }
}

/// Run the SSE4.1 nsq del_dec scale_states.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn silk_nsq_del_dec_scale_states_sse4_1(
    ps_enc_c: &super::structs::NsqConfig,
    nsq: &mut super::structs::silk_nsq_state,
    psDelDec: &mut [super::nsq_del_dec::NSQ_del_dec_struct],
    x16: &[i16],
    x_sc_q10: &mut [i32],
    s_ltp: &[i16],
    s_ltp_q15: &mut [i32],
    subfr: i32,
    n_states_delayed_decision: i32,
    ltp_scale_q14: i32,
    gains_q16: &[i32],
    pitch_l: &[i32],
    signal_type: i32,
    decisionDelay: i32,
) {
    // SAFETY: call sites gate this wrapper with `use_nsq_sse4_1`.
    unsafe {
        x86::silk_nsq_del_dec_scale_states_sse4_1(
            ps_enc_c,
            nsq,
            psDelDec,
            x16,
            x_sc_q10,
            s_ltp,
            s_ltp_q15,
            subfr,
            n_states_delayed_decision,
            ltp_scale_q14,
            gains_q16,
            pitch_l,
            signal_type,
            decisionDelay,
        );
    }
}

/// Returns true if the AVX2 nsq del_dec path should be used.
/// Requires AVX2 and n_states_delayed_decision == 3 or 4.
#[inline]
pub fn use_nsq_del_dec_avx2(arch: Arch, n_states: i32) -> bool {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        arch.has_avx2() && n_states > 2 && n_states <= 4
    }
    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    {
        let _ = (arch, n_states);
        false
    }
}

/// Run the AVX2 nsq del_dec complete outer function.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn silk_nsq_del_dec_avx2(
    ps_enc_c: &super::structs::NsqConfig,
    nsq: &mut super::structs::silk_nsq_state,
    ps_indices: &mut super::structs::SideInfoIndices,
    x16: &[i16],
    pulses: &mut [i8],
    pred_coef_q12: &[i16],
    ltpcoef_q14: &[i16],
    ar_q13: &[i16],
    harm_shape_gain_q14: &[i32],
    tilt_q14: &[i32],
    lf_shp_q14: &[i32],
    gains_q16: &[i32],
    pitch_l: &[i32],
    lambda_q10: i32,
    ltp_scale_q14: i32,
) {
    // SAFETY: call sites gate this wrapper with `use_nsq_del_dec_avx2`.
    unsafe {
        x86::silk_nsq_del_dec_avx2(
            ps_enc_c,
            nsq,
            ps_indices,
            x16,
            pulses,
            pred_coef_q12,
            ltpcoef_q14,
            ar_q13,
            harm_shape_gain_q14,
            tilt_q14,
            lf_shp_q14,
            gains_q16,
            pitch_l,
            lambda_q10,
            ltp_scale_q14,
        );
    }
}

/// Run the SSE4.1 nsq del_dec inner quantizer.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn silk_noise_shape_quantizer_del_dec_sse4_1(
    nsq: &mut super::structs::silk_nsq_state,
    psDelDec: &mut [super::nsq_del_dec::NSQ_del_dec_struct],
    signal_type: i32,
    x_q10: &[i32],
    pulses: &mut [i8],
    pulses_off: usize,
    xq_off: usize,
    s_ltp_q15: &mut [i32],
    delayedGain_Q10: &mut [i32; 40],
    a_q12: &[i16],
    b_q14: &[i16],
    ar_shp_q13: &[i16],
    lag: i32,
    HarmShapeFIRPacked_Q14: i32,
    tilt_q14: i32,
    lf_shp_q14: i32,
    Gain_Q16: i32,
    lambda_q10: i32,
    offset_q10: i32,
    length: i32,
    subfr: i32,
    shaping_lpcorder: i32,
    predict_lpcorder: i32,
    warping_q16: i32,
    n_states_delayed_decision: i32,
    smpl_buf_idx: &mut i32,
    decisionDelay: i32,
) {
    // SAFETY: call sites gate this wrapper with `use_nsq_sse4_1`.
    unsafe {
        x86::silk_noise_shape_quantizer_del_dec_sse4_1(
            nsq,
            psDelDec,
            signal_type,
            x_q10,
            pulses,
            pulses_off,
            xq_off,
            s_ltp_q15,
            delayedGain_Q10,
            a_q12,
            b_q14,
            ar_shp_q13,
            lag,
            HarmShapeFIRPacked_Q14,
            tilt_q14,
            lf_shp_q14,
            Gain_Q16,
            lambda_q10,
            offset_q10,
            length,
            subfr,
            shaping_lpcorder,
            predict_lpcorder,
            warping_q16,
            n_states_delayed_decision,
            smpl_buf_idx,
            decisionDelay,
        );
    }
}

/// Returns true if the aarch64 NEON nsq del_dec path should be used.
/// Requires NEON and n_states_delayed_decision == 3 or 4.
#[inline]
pub fn use_neon_nsq_del_dec(arch: Arch, n_states: i32) -> bool {
    #[cfg(target_arch = "aarch64")]
    {
        arch.has_neon() && n_states > 2 && n_states <= 4
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        let _ = (arch, n_states);
        false
    }
}

/// Run the aarch64 NEON nsq del_dec complete outer function.
#[cfg(target_arch = "aarch64")]
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn silk_nsq_del_dec_neon(
    ps_enc_c: &super::structs::NsqConfig,
    nsq: &mut super::structs::silk_nsq_state,
    ps_indices: &mut super::structs::SideInfoIndices,
    x16: &[i16],
    pulses: &mut [i8],
    pred_coef_q12: &[i16],
    ltpcoef_q14: &[i16],
    ar_q13: &[i16],
    harm_shape_gain_q14: &[i32],
    tilt_q14: &[i32],
    lf_shp_q14: &[i32],
    gains_q16: &[i32],
    pitch_l: &[i32],
    lambda_q10: i32,
    ltp_scale_q14: i32,
) {
    // SAFETY: call sites gate this wrapper with `use_neon_nsq_del_dec`.
    unsafe {
        aarch64::silk_nsq_del_dec_neon(
            ps_enc_c,
            nsq,
            ps_indices,
            x16,
            pulses,
            pred_coef_q12,
            ltpcoef_q14,
            ar_q13,
            harm_shape_gain_q14,
            tilt_q14,
            lf_shp_q14,
            gains_q16,
            pitch_l,
            lambda_q10,
            ltp_scale_q14,
        );
    }
}

#[cfg(test)]
mod tests {
    #[cfg(target_arch = "aarch64")]
    use super::*;

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn neon_nsq_del_dec_gate_accepts_dotprod_for_3_and_4_states() {
        for arch in [Arch::Neon, Arch::DotProd] {
            assert!(!use_neon_nsq_del_dec(arch, 1));
            assert!(!use_neon_nsq_del_dec(arch, 2));
            assert!(use_neon_nsq_del_dec(arch, 3));
            assert!(use_neon_nsq_del_dec(arch, 4));
            assert!(!use_neon_nsq_del_dec(arch, 5));
        }
    }
}
