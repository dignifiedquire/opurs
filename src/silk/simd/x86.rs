//! x86/x86_64 SIMD implementations for SILK functions.
//!
//! SSE4.1 and AVX2 intrinsics for noise shaping, inner products, etc.
//! All functions require `#[target_feature]` and are called only after cpufeatures detection.

#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

use crate::silk::define::{
    DECISION_DELAY, HARM_SHAPE_FIR_TAPS, LTP_ORDER, MAX_SHAPE_LPC_ORDER, NSQ_LPC_BUF_LENGTH,
    TYPE_VOICED,
};
use crate::silk::inlines::{silk_div32_varq, silk_inverse32_varq};
use crate::silk::nsq_del_dec::{
    copy_del_dec_state_partial, NSQ_del_dec_struct, NSQ_sample_struct, NsqSamplePair,
};
use crate::silk::sigproc_fix::silk_rand;
use crate::silk::structs::{silk_encoder_state, silk_nsq_state, NsqConfig, SideInfoIndices};

/// SSE4.1 implementation of `silk_noise_shape_quantizer_short_prediction`.
/// Port of `silk/x86/NSQ_sse4_1.c`.
///
/// # Safety
/// Requires SSE4.1 support (checked by caller via cpufeatures).
#[target_feature(enable = "sse4.1")]
pub unsafe fn silk_noise_shape_quantizer_short_prediction_sse4_1(
    buf32: &[i32],
    coef16: &[i16],
    order: i32,
) -> i32 {
    let b = buf32.len();
    debug_assert!(b >= order as usize);
    debug_assert!(coef16.len() >= order as usize);
    debug_assert!(order == 10 || order == 16);

    let mut out: i32 = order >> 1;

    // Process first 8 elements (always present for order 10 or 16)
    // buf32 is indexed backwards from end: buf32[b-1] pairs with coef16[0],
    // buf32[b-2] with coef16[1], etc. When loading buf32 in memory order
    // [b-8..b-5] and [b-4..b-1], we must reverse the coefficient order within
    // each group so the pairings are correct.
    let buf_ptr = buf32.as_ptr().add(b - 8);
    let b0 = _mm_loadu_si128(buf_ptr as *const __m128i); // [b-8, b-7, b-6, b-5]
    let b1 = _mm_loadu_si128(buf_ptr.add(4) as *const __m128i); // [b-4, b-3, b-2, b-1]

    // Load coef16[4..8] and reverse to [7,6,5,4] so b0*c0 gives correct pairings
    let c0 = _mm_shuffle_epi32(
        _mm_cvtepi16_epi32(_mm_loadl_epi64(coef16.as_ptr().add(4) as *const __m128i)),
        0x1B, // reverse: 3,2,1,0
    );
    // Load coef16[0..4] and reverse to [3,2,1,0] so b1*c1 gives correct pairings
    let c1 = _mm_shuffle_epi32(
        _mm_cvtepi16_epi32(_mm_loadl_epi64(coef16.as_ptr() as *const __m128i)),
        0x1B, // reverse: 3,2,1,0
    );

    // Widening multiply: (buf * coef) >> 16
    // For each pair: (buf32[_i] as i64 * coef16[_i] as i64) >> 16
    let p0_lo = _mm_mul_epi32(b0, c0);
    let p0_hi = _mm_mul_epi32(_mm_srli_si128(b0, 4), _mm_srli_si128(c0, 4));
    let p1_lo = _mm_mul_epi32(b1, c1);
    let p1_hi = _mm_mul_epi32(_mm_srli_si128(b1, 4), _mm_srli_si128(c1, 4));

    // Shift right by 16 and truncate to i32
    let s0_lo = _mm_shuffle_epi32(_mm_srli_epi64(p0_lo, 16), 0x08);
    let s0_hi = _mm_shuffle_epi32(_mm_srli_epi64(p0_hi, 16), 0x08);
    let s1_lo = _mm_shuffle_epi32(_mm_srli_epi64(p1_lo, 16), 0x08);
    let s1_hi = _mm_shuffle_epi32(_mm_srli_epi64(p1_hi, 16), 0x08);

    let sum0 = _mm_add_epi32(
        _mm_unpacklo_epi32(s0_lo, s0_hi),
        _mm_unpacklo_epi32(s1_lo, s1_hi),
    );

    if order == 16 {
        let buf_ptr2 = buf32.as_ptr().add(b - 16);
        let b2 = _mm_loadu_si128(buf_ptr2 as *const __m128i);
        let b3 = _mm_loadu_si128(buf_ptr2.add(4) as *const __m128i);

        // Reverse coefficient order within each group (same reason as above)
        let c2 = _mm_shuffle_epi32(
            _mm_cvtepi16_epi32(_mm_loadl_epi64(coef16.as_ptr().add(12) as *const __m128i)),
            0x1B,
        );
        let c3 = _mm_shuffle_epi32(
            _mm_cvtepi16_epi32(_mm_loadl_epi64(coef16.as_ptr().add(8) as *const __m128i)),
            0x1B,
        );

        let p2_lo = _mm_mul_epi32(b2, c2);
        let p2_hi = _mm_mul_epi32(_mm_srli_si128(b2, 4), _mm_srli_si128(c2, 4));
        let p3_lo = _mm_mul_epi32(b3, c3);
        let p3_hi = _mm_mul_epi32(_mm_srli_si128(b3, 4), _mm_srli_si128(c3, 4));

        let s2_lo = _mm_shuffle_epi32(_mm_srli_epi64(p2_lo, 16), 0x08);
        let s2_hi = _mm_shuffle_epi32(_mm_srli_epi64(p2_hi, 16), 0x08);
        let s3_lo = _mm_shuffle_epi32(_mm_srli_epi64(p3_lo, 16), 0x08);
        let s3_hi = _mm_shuffle_epi32(_mm_srli_epi64(p3_hi, 16), 0x08);

        let sum1 = _mm_add_epi32(
            _mm_unpacklo_epi32(s2_lo, s2_hi),
            _mm_unpacklo_epi32(s3_lo, s3_hi),
        );
        let total = _mm_add_epi32(sum0, sum1);

        // Horizontal sum
        let hi = _mm_srli_si128(total, 8);
        let sum = _mm_add_epi32(total, hi);
        let hi2 = _mm_srli_si128(sum, 4);
        let sum = _mm_add_epi32(sum, hi2);
        out += _mm_cvtsi128_si32(sum);
    } else {
        // order == 10: process 2 more elements scalar
        let sum_vec = sum0;
        let hi = _mm_srli_si128(sum_vec, 8);
        let sum = _mm_add_epi32(sum_vec, hi);
        let hi2 = _mm_srli_si128(sum, 4);
        let sum = _mm_add_epi32(sum, hi2);
        out += _mm_cvtsi128_si32(sum);

        // Remaining 2 elements (indices 8, 9)
        out = (out as i64 + ((buf32[b - 9] as i64 * coef16[8] as i64) >> 16)) as i32;
        out = (out as i64 + ((buf32[b - 10] as i64 * coef16[9] as i64) >> 16)) as i32;
    }

    out
}

/// SSE2 implementation of VAD energy accumulation.
/// Computes sum of (X[_i] >> 3)^2 for _i in 0..len.
/// Port of `silk/x86/VAD_sse4_1.c` inner loop (uses only SSE2 instructions).
///
/// # Safety
/// Requires SSE2 support (checked by caller via cpufeatures).
#[target_feature(enable = "sse2")]
pub unsafe fn silk_vad_energy_sse2(x: &[i16]) -> i32 {
    let n = x.len();
    let mut acc = _mm_setzero_si128();
    let mut _i = 0usize;

    // Process 8 samples at a time
    while _i + 7 < n {
        let xmm = _mm_loadu_si128(x.as_ptr().add(_i) as *const __m128i);
        // Arithmetic right shift by 3 (stays in i16)
        let shifted = _mm_srai_epi16(xmm, 3);
        // Multiply pairs of i16 and sum adjacent pairs → 4 x i32
        let squared = _mm_madd_epi16(shifted, shifted);
        acc = _mm_add_epi32(acc, squared);
        _i += 8;
    }

    // Horizontal sum of 4 x i32
    let hi64 = _mm_unpackhi_epi64(acc, acc);
    acc = _mm_add_epi32(acc, hi64);
    let hi32 = _mm_shufflelo_epi16(acc, 0x0E);
    acc = _mm_add_epi32(acc, hi32);
    let mut result = _mm_cvtsi128_si32(acc);

    // Handle remaining elements
    while _i < n {
        let x_tmp = (*x.get_unchecked(_i) as i32) >> 3;
        result += (x_tmp as i16 as i32) * (x_tmp as i16 as i32);
        _i += 1;
    }

    result
}

/// SSE4.1 full-function VAD entry.
/// Mirrors upstream RTCD surface `silk/x86/main_sse.h:silk_vad_get_sa_q8`.
///
/// # Safety
/// Requires SSE4.1 support (checked by caller via cpufeatures).
#[target_feature(enable = "sse4.1")]
pub unsafe fn silk_vad_get_sa_q8_sse4_1(ps_enc_c: &mut silk_encoder_state, p_in: &[i16]) -> i32 {
    crate::silk::vad::silk_vad_get_sa_q8_c(ps_enc_c, p_in)
}

/// SSE4.1 full-function nsq entry.
/// Mirrors upstream RTCD surface `silk/x86/main_sse.h:silk_nsq`.
///
/// # Safety
/// Requires SSE4.1 support (checked by caller via cpufeatures).
#[target_feature(enable = "sse4.1")]
#[allow(clippy::too_many_arguments)]
pub unsafe fn silk_nsq_sse4_1(
    ps_enc_c: &NsqConfig,
    nsq: &mut silk_nsq_state,
    ps_indices: &mut SideInfoIndices,
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
    crate::silk::nsq::silk_nsq_c(
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

/// SSE4.1 full-function nsq-del-dec entry.
/// Mirrors upstream RTCD surface `silk/x86/main_sse.h:silk_nsq_del_dec` (SSE tier).
///
/// # Safety
/// Requires SSE4.1 support (checked by caller via cpufeatures).
#[target_feature(enable = "sse4.1")]
#[allow(clippy::too_many_arguments)]
pub unsafe fn silk_nsq_del_dec_sse4_1(
    ps_enc_c: &NsqConfig,
    nsq: &mut silk_nsq_state,
    ps_indices: &mut SideInfoIndices,
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
    crate::silk::nsq_del_dec::silk_nsq_del_dec_c(
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

/// AVX2+FMA implementation of `silk_inner_product_flp`.
/// f32→f64 inner product using dual 256-bit accumulators with fused multiply-add.
/// Port of `silk/float/x86/inner_product_FLP_avx2.c`.
///
/// # Safety
/// Requires AVX2 and FMA support (checked by caller via cpufeatures).
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn silk_inner_product_flp_avx2(data1: &[f32], data2: &[f32]) -> f64 {
    let n = data1.len().min(data2.len());
    let mut accum1 = _mm256_setzero_pd();
    let mut accum2 = _mm256_setzero_pd();
    let mut _i = 0usize;

    // Main loop: 8 f32s per iteration (two groups of 4 → 4 f64s each)
    while _i + 7 < n {
        let x1f = _mm_loadu_ps(data1.as_ptr().add(_i));
        let x2f = _mm_loadu_ps(data2.as_ptr().add(_i));
        let x1d = _mm256_cvtps_pd(x1f);
        let x2d = _mm256_cvtps_pd(x2f);
        accum1 = _mm256_fmadd_pd(x1d, x2d, accum1);

        let x1f = _mm_loadu_ps(data1.as_ptr().add(_i + 4));
        let x2f = _mm_loadu_ps(data2.as_ptr().add(_i + 4));
        let x1d = _mm256_cvtps_pd(x1f);
        let x2d = _mm256_cvtps_pd(x2f);
        accum2 = _mm256_fmadd_pd(x1d, x2d, accum2);

        _i += 8;
    }

    // Secondary loop: 4 f32s for remainder 4-7
    while _i + 3 < n {
        let x1f = _mm_loadu_ps(data1.as_ptr().add(_i));
        let x2f = _mm_loadu_ps(data2.as_ptr().add(_i));
        let x1d = _mm256_cvtps_pd(x1f);
        let x2d = _mm256_cvtps_pd(x2f);
        accum1 = _mm256_fmadd_pd(x1d, x2d, accum1);
        _i += 4;
    }

    // Horizontal reduction: combine two accumulators, then reduce 4 f64s → 1
    accum1 = _mm256_add_pd(accum1, accum2);
    accum1 = _mm256_add_pd(accum1, _mm256_permute2f128_pd(accum1, accum1, 1));
    accum1 = _mm256_hadd_pd(accum1, accum1);
    let mut result = _mm256_cvtsd_f64(accum1);

    // Scalar tail for remaining 0-3 elements
    while _i < n {
        result += *data1.get_unchecked(_i) as f64 * *data2.get_unchecked(_i) as f64;
        _i += 1;
    }

    result
}

/// SSE4.1 implementation of the nsq inner quantizer loop, specialized for
/// shaping_lpcorder=10 and predict_lpcorder=16.
/// Port of `silk/x86/NSQ_sse4_1.c:silk_noise_shape_quantizer_10_16_sse4_1`.
///
/// Maintains LPC and AR filter state in packed i16 SIMD registers for
/// register-resident operation. Uses table-based quantization decisions.
///
/// # Safety
/// Requires SSE4.1 support (checked by caller via cpufeatures).
#[target_feature(enable = "sse4.1")]
#[allow(clippy::too_many_arguments)]
pub unsafe fn silk_noise_shape_quantizer_10_16_sse4_1(
    nsq: &mut silk_nsq_state,
    signal_type: i32,
    x_sc_q10: &[i32],
    pulses: &mut [i8],
    xq_off: usize,
    s_ltp_q15: &mut [i32],
    a_q12: &[i16],
    b_q14: &[i16],
    ar_shp_q13: &[i16],
    lag: i32,
    harm_shape_firpacked_q14: i32,
    tilt_q14: i32,
    lf_shp_q14: i32,
    gain_q16: i32,
    lambda_q10: i32,
    offset_q10: i32,
    length: i32,
    table: &[[i32; 4]; 64],
) {
    let rdo_offset = (lambda_q10 >> 1) - 512;

    let mut shp_lag_idx = (nsq.s_ltp_shp_buf_idx - lag + HARM_SHAPE_FIR_TAPS / 2) as usize;
    let mut pred_lag_idx = (nsq.s_ltp_buf_idx - lag + LTP_ORDER as i32 / 2) as usize;
    let gain_q10 = gain_q16 >> 6;

    let mut lpc_idx: usize = NSQ_LPC_BUF_LENGTH - 1;

    let mut s_lf_ar_shp_q14: i32 = nsq.s_lf_ar_shp_q14;
    let mut xq_q14: i32 = nsq.s_lpc_q14[lpc_idx];
    let s_diff_shp_q14: i32 = nsq.s_diff_shp_q14;
    let mut ltp_pred_q13: i32 = 0;

    // --- Load a_q12 coefficients, byte-reversed for paired computation ---
    let byte_rev = _mm_set_epi8(1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14);
    let a_q12_01234567 =
        _mm_shuffle_epi8(_mm_loadu_si128(a_q12.as_ptr() as *const __m128i), byte_rev);
    let a_q12_89_abcdef = _mm_shuffle_epi8(
        _mm_loadu_si128(a_q12.as_ptr().add(8) as *const __m128i),
        byte_rev,
    );

    // --- Load ar_shp_q13 coefficients (first 8 of 10) ---
    let ar_shp_q13_76543210 = _mm_loadu_si128(ar_shp_q13.as_ptr() as *const __m128i);

    // --- Load psLPC_Q14 state into interleaved hi/lo format ---
    let split_pattern = _mm_set_epi8(15, 14, 11, 10, 7, 6, 3, 2, 13, 12, 9, 8, 5, 4, 1, 0);

    let ps_lpc_ptr = nsq.s_lpc_q14.as_ptr().add(lpc_idx);

    let mut tempa = _mm_shuffle_epi8(
        _mm_loadu_si128(ps_lpc_ptr.sub(16) as *const __m128i),
        split_pattern,
    );
    let mut tempb = _mm_shuffle_epi8(
        _mm_loadu_si128(ps_lpc_ptr.sub(12) as *const __m128i),
        split_pattern,
    );
    let mut ps_lpc_q14_hi_89_abcdef = _mm_unpackhi_epi64(tempa, tempb);
    let mut ps_lpc_q14_lo_89_abcdef = _mm_unpacklo_epi64(tempa, tempb);

    tempa = _mm_shuffle_epi8(
        _mm_loadu_si128(ps_lpc_ptr.sub(8) as *const __m128i),
        split_pattern,
    );
    tempb = _mm_shuffle_epi8(
        _mm_loadu_si128(ps_lpc_ptr.sub(4) as *const __m128i),
        split_pattern,
    );
    let mut ps_lpc_q14_hi_01234567 = _mm_unpackhi_epi64(tempa, tempb);
    let mut ps_lpc_q14_lo_01234567 = _mm_unpacklo_epi64(tempa, tempb);

    // --- Load s_ar2_q14 state into interleaved hi/lo format ---
    tempa = _mm_shuffle_epi8(
        _mm_loadu_si128(nsq.s_ar2_q14.as_ptr() as *const __m128i),
        split_pattern,
    );
    tempb = _mm_shuffle_epi8(
        _mm_loadu_si128(nsq.s_ar2_q14.as_ptr().add(4) as *const __m128i),
        split_pattern,
    );
    let mut s_ar2_q14_hi_76543210 = _mm_unpackhi_epi64(tempa, tempb);
    let mut s_ar2_q14_lo_76543210 = _mm_unpacklo_epi64(tempa, tempb);

    let xmm_one = _mm_set1_epi16(1);

    // =========== Main per-sample loop ===========
    for _i in 0..length as usize {
        // ----- Short-term LPC prediction (order 16) -----
        let mut lpc_pred_q10: i32 = 8;

        // Shift LPC sliding window
        ps_lpc_q14_hi_89_abcdef =
            _mm_alignr_epi8(ps_lpc_q14_hi_01234567, ps_lpc_q14_hi_89_abcdef, 2);
        ps_lpc_q14_lo_89_abcdef =
            _mm_alignr_epi8(ps_lpc_q14_lo_01234567, ps_lpc_q14_lo_89_abcdef, 2);
        ps_lpc_q14_hi_01234567 = _mm_srli_si128(ps_lpc_q14_hi_01234567, 2);
        ps_lpc_q14_lo_01234567 = _mm_srli_si128(ps_lpc_q14_lo_01234567, 2);
        ps_lpc_q14_hi_01234567 = _mm_insert_epi16(ps_lpc_q14_hi_01234567, xq_q14 >> 16, 7);
        ps_lpc_q14_lo_01234567 = _mm_insert_epi16(ps_lpc_q14_lo_01234567, xq_q14, 7);

        // High part: pmaddwd
        let xmm_hi_07 = _mm_madd_epi16(ps_lpc_q14_hi_01234567, a_q12_01234567);
        let xmm_hi_8_f = _mm_madd_epi16(ps_lpc_q14_hi_89_abcdef, a_q12_89_abcdef);

        // Low part: pmulhw + sign correction
        let sign_07 = _mm_cmpgt_epi16(_mm_setzero_si128(), ps_lpc_q14_lo_01234567);
        let sign_8_f = _mm_cmpgt_epi16(_mm_setzero_si128(), ps_lpc_q14_lo_89_abcdef);
        let corr_07 = _mm_and_si128(sign_07, a_q12_01234567);
        let corr_8_f = _mm_and_si128(sign_8_f, a_q12_89_abcdef);
        let mut xmm_lo_07 = _mm_mulhi_epi16(ps_lpc_q14_lo_01234567, a_q12_01234567);
        let mut xmm_lo_8_f = _mm_mulhi_epi16(ps_lpc_q14_lo_89_abcdef, a_q12_89_abcdef);
        xmm_lo_07 = _mm_add_epi16(xmm_lo_07, corr_07);
        xmm_lo_8_f = _mm_add_epi16(xmm_lo_8_f, corr_8_f);
        xmm_lo_07 = _mm_madd_epi16(xmm_lo_07, xmm_one);
        xmm_lo_8_f = _mm_madd_epi16(xmm_lo_8_f, xmm_one);

        // Accumulate
        let mut acc = _mm_add_epi32(
            _mm_add_epi32(xmm_hi_07, xmm_hi_8_f),
            _mm_add_epi32(xmm_lo_07, xmm_lo_8_f),
        );
        acc = _mm_add_epi32(acc, _mm_unpackhi_epi64(acc, acc));
        acc = _mm_add_epi32(acc, _mm_shufflelo_epi16(acc, 0x0E));
        lpc_pred_q10 += _mm_cvtsi128_si32(acc);

        // ----- Long-term prediction -----
        if signal_type == TYPE_VOICED {
            ltp_pred_q13 = 2;
            let b_q14_3210 = _mm_cvtepi16_epi32(_mm_loadl_epi64(b_q14.as_ptr() as *const __m128i));
            let b_q14_0123 = _mm_shuffle_epi32(b_q14_3210, 0x1B);

            let pred_0123 =
                _mm_loadu_si128(s_ltp_q15.as_ptr().add(pred_lag_idx - 3) as *const __m128i);
            let pred_rev = _mm_shuffle_epi32(pred_0123, 0x1B);
            tempa = _mm_srli_si128(_mm_mul_epi32(pred_rev, b_q14_3210), 2);
            tempb = _mm_srli_si128(_mm_mul_epi32(pred_0123, b_q14_0123), 2);
            let sum4 = _mm_add_epi32(tempa, tempb);
            let sum2 = _mm_add_epi32(sum4, _mm_shuffle_epi32(sum4, 0x0E));
            ltp_pred_q13 += _mm_cvtsi128_si32(sum2);

            // 5th tap scalar
            ltp_pred_q13 = (ltp_pred_q13 as i64
                + ((s_ltp_q15[pred_lag_idx - 4] as i64 * b_q14[4] as i64) >> 16))
                as i32;
            pred_lag_idx += 1;
        }

        // ----- Noise shape feedback (SIMD for 8, scalar for 2) -----
        nsq.s_ar2_q14[9] = nsq.s_ar2_q14[8];
        nsq.s_ar2_q14[8] = _mm_cvtsi128_si32(_mm_srli_si128(
            _mm_unpackhi_epi16(s_ar2_q14_lo_76543210, s_ar2_q14_hi_76543210),
            12,
        ));

        s_ar2_q14_hi_76543210 = _mm_slli_si128(s_ar2_q14_hi_76543210, 2);
        s_ar2_q14_lo_76543210 = _mm_slli_si128(s_ar2_q14_lo_76543210, 2);
        s_ar2_q14_hi_76543210 = _mm_insert_epi16(s_ar2_q14_hi_76543210, s_diff_shp_q14 >> 16, 0);
        s_ar2_q14_lo_76543210 = _mm_insert_epi16(s_ar2_q14_lo_76543210, s_diff_shp_q14, 0);

        let ar_hi = _mm_madd_epi16(s_ar2_q14_hi_76543210, ar_shp_q13_76543210);
        let ar_sign = _mm_cmpgt_epi16(_mm_setzero_si128(), s_ar2_q14_lo_76543210);
        let ar_corr = _mm_and_si128(ar_sign, ar_shp_q13_76543210);
        let mut ar_lo = _mm_mulhi_epi16(s_ar2_q14_lo_76543210, ar_shp_q13_76543210);
        ar_lo = _mm_add_epi16(ar_lo, ar_corr);
        ar_lo = _mm_madd_epi16(ar_lo, xmm_one);

        let mut ar_acc = _mm_add_epi32(ar_hi, ar_lo);
        ar_acc = _mm_add_epi32(ar_acc, _mm_unpackhi_epi64(ar_acc, ar_acc));
        ar_acc = _mm_add_epi32(ar_acc, _mm_shufflelo_epi16(ar_acc, 0x0E));
        let mut n_ar_q12: i32 = 5 + _mm_cvtsi128_si32(ar_acc);

        n_ar_q12 =
            (n_ar_q12 as i64 + ((nsq.s_ar2_q14[8] as i64 * ar_shp_q13[8] as i64) >> 16)) as i32;
        n_ar_q12 =
            (n_ar_q12 as i64 + ((nsq.s_ar2_q14[9] as i64 * ar_shp_q13[9] as i64) >> 16)) as i32;

        n_ar_q12 = ((n_ar_q12 as u32) << 1) as i32;
        n_ar_q12 =
            (n_ar_q12 as i64 + ((s_lf_ar_shp_q14 as i64 * tilt_q14 as i16 as i64) >> 16)) as i32;

        let n_lf_q12: i32 = {
            let t1 = ((nsq.s_ltp_shp_q14[(nsq.s_ltp_shp_buf_idx - 1) as usize] as i64
                * lf_shp_q14 as i16 as i64)
                >> 16) as i32;
            (t1 as i64 + ((s_lf_ar_shp_q14 as i64 * (lf_shp_q14 as i64 >> 16)) >> 16)) as i32
        };

        // ----- Combine prediction and noise shaping -----
        let mut tmp1 = (((lpc_pred_q10 as u32) << 2) as i32).wrapping_sub(n_ar_q12);
        tmp1 = tmp1.wrapping_sub(n_lf_q12);
        if lag > 0 {
            let n_ltp_q13 = {
                let t1 = ((nsq.s_ltp_shp_q14[shp_lag_idx]
                    .saturating_add(nsq.s_ltp_shp_q14[shp_lag_idx - 2]))
                    as i64
                    * harm_shape_firpacked_q14 as i16 as i64)
                    >> 16;
                let t2 = (t1
                    + ((nsq.s_ltp_shp_q14[shp_lag_idx - 1] as i64
                        * (harm_shape_firpacked_q14 as i64 >> 16))
                        >> 16)) as i32;
                ((t2 as u32) << 1) as i32
            };
            shp_lag_idx += 1;
            let tmp2 = ltp_pred_q13 - n_ltp_q13;
            tmp1 = tmp2.wrapping_add(((tmp1 as u32) << 1) as i32);
            tmp1 = ((tmp1 >> 2) + 1) >> 1;
        } else {
            tmp1 = ((tmp1 >> 1) + 1) >> 1;
        }

        let mut r_q10 = x_sc_q10[_i] - tmp1;

        nsq.rand_seed = silk_rand(nsq.rand_seed);
        if nsq.rand_seed < 0 {
            r_q10 = -r_q10;
        }
        r_q10 = r_q10.clamp(-(31 << 10), 30 << 10);

        // ----- Table-based quantization -----
        let mut q1_q0 = (r_q10 - offset_q10) >> 10;
        if lambda_q10 > 2048 {
            let q1_q10_tmp = r_q10 - offset_q10;
            if q1_q10_tmp > rdo_offset {
                q1_q0 = (q1_q10_tmp - rdo_offset) >> 10;
            } else if q1_q10_tmp < -rdo_offset {
                q1_q0 = (q1_q10_tmp + rdo_offset) >> 10;
            } else if q1_q10_tmp < 0 {
                q1_q0 = -1;
            } else {
                q1_q0 = 0;
            }
        }

        let tidx = (q1_q0 + 32).clamp(0, 63) as usize;
        let mut q1_q10 = table[tidx][0];
        let q2_q10 = table[tidx][1];
        if (r_q10
            .wrapping_mul(table[tidx][2])
            .wrapping_sub(table[tidx][3]))
            < 0
        {
            q1_q10 = q2_q10;
        }

        pulses[_i] = (((q1_q10 >> 9) + 1) >> 1) as i8;

        // ----- Excitation and state update -----
        let mut exc_q14 = ((q1_q10 as u32) << 4) as i32;
        if nsq.rand_seed < 0 {
            exc_q14 = -exc_q14;
        }
        let lpc_exc_q14 = exc_q14 + ((ltp_pred_q13 as u32) << 1) as i32;
        xq_q14 = lpc_exc_q14.wrapping_add(((lpc_pred_q10 as u32) << 4) as i32);

        lpc_idx += 1;
        nsq.s_lpc_q14[lpc_idx] = xq_q14;

        nsq.s_diff_shp_q14 = xq_q14 - ((x_sc_q10[_i] as u32) << 4) as i32;
        s_lf_ar_shp_q14 = nsq
            .s_diff_shp_q14
            .wrapping_sub(((n_ar_q12 as u32) << 2) as i32);
        nsq.s_lf_ar_shp_q14 = s_lf_ar_shp_q14;

        nsq.s_ltp_shp_q14[nsq.s_ltp_shp_buf_idx as usize] =
            s_lf_ar_shp_q14.wrapping_sub(((n_lf_q12 as u32) << 2) as i32);
        s_ltp_q15[nsq.s_ltp_buf_idx as usize] = ((lpc_exc_q14 as u32) << 1) as i32;
        nsq.s_ltp_shp_buf_idx += 1;
        nsq.s_ltp_buf_idx += 1;

        nsq.rand_seed = (nsq.rand_seed as u32).wrapping_add(pulses[_i] as u32) as i32;
    }

    // =========== Post-loop: write back s_ar2_q14 ===========
    tempa = _mm_unpackhi_epi16(s_ar2_q14_lo_76543210, s_ar2_q14_hi_76543210);
    tempb = _mm_unpacklo_epi16(s_ar2_q14_lo_76543210, s_ar2_q14_hi_76543210);
    _mm_storeu_si128(nsq.s_ar2_q14.as_mut_ptr().add(4) as *mut __m128i, tempa);
    _mm_storeu_si128(nsq.s_ar2_q14.as_mut_ptr() as *mut __m128i, tempb);

    // =========== Post-loop: SIMD XQ output scaling ===========
    let ps_lpc_q14_out = &nsq.s_lpc_q14[NSQ_LPC_BUF_LENGTH..];
    let xmm_round = _mm_set1_epi32(1 << 7);
    let xmm_gain_q10 = _mm_set1_epi32(gain_q10);

    let mut ii = 0i32;
    while ii < length - 7 {
        let ui = ii as usize;
        let xq_3210 = _mm_loadu_si128(ps_lpc_q14_out.as_ptr().add(ui) as *const __m128i);
        let xq_7654 = _mm_loadu_si128(ps_lpc_q14_out.as_ptr().add(ui + 4) as *const __m128i);

        let x3x1 = _mm_shuffle_epi32(xq_3210, 0x39); // (0,3,2,1)
        let x7x5 = _mm_shuffle_epi32(xq_7654, 0x39);

        let mut r_3210 = _mm_srli_epi64(_mm_mul_epi32(xq_3210, xmm_gain_q10), 16);
        let r_x3x1 = _mm_slli_epi64(_mm_mul_epi32(x3x1, xmm_gain_q10), 16);
        let mut r_7654 = _mm_srli_epi64(_mm_mul_epi32(xq_7654, xmm_gain_q10), 16);
        let r_x7x5 = _mm_slli_epi64(_mm_mul_epi32(x7x5, xmm_gain_q10), 16);

        r_3210 = _mm_blend_epi16(r_3210, r_x3x1, 0xCC);
        r_7654 = _mm_blend_epi16(r_7654, r_x7x5, 0xCC);

        r_3210 = _mm_srai_epi32(_mm_add_epi32(r_3210, xmm_round), 8);
        r_7654 = _mm_srai_epi32(_mm_add_epi32(r_7654, xmm_round), 8);

        let packed = _mm_packs_epi32(r_3210, r_7654);
        _mm_storeu_si128(nsq.xq.as_mut_ptr().add(xq_off + ui) as *mut __m128i, packed);
        ii += 8;
    }
    while ii < length {
        let ui = ii as usize;
        let smulww = ((ps_lpc_q14_out[ui] as i64 * gain_q10 as i64) >> 16) as i32;
        let rounded = ((smulww >> 7) + 1) >> 1;
        nsq.xq[xq_off + ui] = rounded.clamp(-32768, 32767) as i16;
        ii += 1;
    }

    // =========== Post-loop: copy LPC buffer ===========
    nsq.s_lpc_q14
        .copy_within(length as usize..length as usize + NSQ_LPC_BUF_LENGTH, 0);
}

/// SSE4.1 implementation of `silk_nsq_del_dec_scale_states`.
/// SIMD-accelerated input scaling and gain adjustment loops.
///
/// # Safety
/// Requires SSE4.1 support (checked by caller via cpufeatures).
#[target_feature(enable = "sse4.1")]
#[allow(clippy::too_many_arguments)]
pub unsafe fn silk_nsq_del_dec_scale_states_sse4_1(
    ps_enc_c: &NsqConfig,
    nsq: &mut silk_nsq_state,
    ps_del_dec: &mut [NSQ_del_dec_struct],
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
    decision_delay: i32,
) {
    let lag = pitch_l[subfr as usize];
    let mut inv_gain_q31 = silk_inverse32_varq(gains_q16[subfr as usize].max(1), 47);

    let inv_gain_q26 = ((inv_gain_q31 >> 4) + 1) >> 1;

    // SIMD input scaling: x_sc_q10[_i] = silk_smulww(x16[_i], inv_gain_q26)
    let xmm_inv_gain = _mm_set1_epi32(inv_gain_q26);
    let subfr_len = ps_enc_c.subfr_length;
    let mut _i = 0usize;
    while _i + 3 < subfr_len {
        let xmm_x16 = _mm_cvtepi16_epi32(_mm_loadl_epi64(x16.as_ptr().add(_i) as *const __m128i));
        let xmm_odd = _mm_shuffle_epi32(xmm_x16, 0x39);
        let mut r_even = _mm_mul_epi32(xmm_x16, xmm_inv_gain);
        let r_odd = _mm_mul_epi32(xmm_odd, xmm_inv_gain);
        r_even = _mm_srli_epi64(r_even, 16);
        let r_odd_s = _mm_slli_epi64(r_odd, 16);
        let result = _mm_blend_epi16(r_even, r_odd_s, 0xCC);
        _mm_storeu_si128(x_sc_q10.as_mut_ptr().add(_i) as *mut __m128i, result);
        _i += 4;
    }
    while _i < subfr_len {
        x_sc_q10[_i] = ((x16[_i] as i64 * inv_gain_q26 as i64) >> 16) as i32;
        _i += 1;
    }

    // LTP state rewhitening (scalar)
    if nsq.rewhite_flag != 0 {
        if subfr == 0 {
            inv_gain_q31 = ((((inv_gain_q31 as i64 * ltp_scale_q14 as i16 as i64) >> 16) as i32
                as u32)
                << 2) as i32;
        }
        let start = (nsq.s_ltp_buf_idx - lag - LTP_ORDER as i32 / 2) as usize;
        let end = nsq.s_ltp_buf_idx as usize;
        for j in start..end {
            s_ltp_q15[j] = ((inv_gain_q31 as i64 * s_ltp[j] as i64) >> 16) as i32;
        }
    }

    // Gain adjustment
    if gains_q16[subfr as usize] != nsq.prev_gain_q16 {
        let gain_adj_q16 = silk_div32_varq(nsq.prev_gain_q16, gains_q16[subfr as usize], 16);

        // SIMD scaling of s_ltp_shp_q14
        let xmm_gain_adj = _mm_set1_epi32(gain_adj_q16);
        let shp_start = (nsq.s_ltp_shp_buf_idx - ps_enc_c.ltp_mem_length as i32) as usize;
        let shp_end = nsq.s_ltp_shp_buf_idx as usize;
        let mut j = shp_start;
        while j + 3 < shp_end {
            let vals = _mm_loadu_si128(nsq.s_ltp_shp_q14.as_ptr().add(j) as *const __m128i);
            let vals_odd = _mm_shuffle_epi32(vals, 0x39);
            let mut r_even = _mm_mul_epi32(vals, xmm_gain_adj);
            let r_odd = _mm_mul_epi32(vals_odd, xmm_gain_adj);
            r_even = _mm_srli_epi64(r_even, 16);
            let r_odd_s = _mm_slli_epi64(r_odd, 16);
            let result = _mm_blend_epi16(r_even, r_odd_s, 0xCC);
            _mm_storeu_si128(
                nsq.s_ltp_shp_q14.as_mut_ptr().add(j) as *mut __m128i,
                result,
            );
            j += 4;
        }
        while j < shp_end {
            nsq.s_ltp_shp_q14[j] =
                ((gain_adj_q16 as i64 * nsq.s_ltp_shp_q14[j] as i64) >> 16) as i32;
            j += 1;
        }

        // Scale LTP prediction state
        if signal_type == TYPE_VOICED && nsq.rewhite_flag == 0 {
            let start = (nsq.s_ltp_buf_idx - lag - LTP_ORDER as i32 / 2) as usize;
            let end = (nsq.s_ltp_buf_idx - decision_delay) as usize;
            for val in s_ltp_q15[start..end].iter_mut() {
                *val = ((gain_adj_q16 as i64 * *val as i64) >> 16) as i32;
            }
        }

        // Per-state scaling
        for ps_dd in ps_del_dec[..n_states_delayed_decision as usize].iter_mut() {
            ps_dd.lf_ar_q14 = ((gain_adj_q16 as i64 * ps_dd.lf_ar_q14 as i64) >> 16) as i32;
            ps_dd.diff_q14 = ((gain_adj_q16 as i64 * ps_dd.diff_q14 as i64) >> 16) as i32;
            for j in 0..NSQ_LPC_BUF_LENGTH {
                ps_dd.s_lpc_q14[j] =
                    ((gain_adj_q16 as i64 * ps_dd.s_lpc_q14[j] as i64) >> 16) as i32;
            }
            for j in 0..MAX_SHAPE_LPC_ORDER as usize {
                ps_dd.s_ar2_q14[j] =
                    ((gain_adj_q16 as i64 * ps_dd.s_ar2_q14[j] as i64) >> 16) as i32;
            }
            for j in 0..DECISION_DELAY as usize {
                ps_dd.pred_q15[j] = ((gain_adj_q16 as i64 * ps_dd.pred_q15[j] as i64) >> 16) as i32;
                ps_dd.shape_q14[j] =
                    ((gain_adj_q16 as i64 * ps_dd.shape_q14[j] as i64) >> 16) as i32;
            }
        }

        nsq.prev_gain_q16 = gains_q16[subfr as usize];
    }
}

/// SSE4.1 implementation of `silk_noise_shape_quantizer_del_dec`.
/// SIMD-accelerated LPC and LTP prediction with scalar noise shaping and quantization.
///
/// # Safety
/// Requires SSE4.1 support (checked by caller via cpufeatures).
#[target_feature(enable = "sse4.1")]
#[allow(clippy::too_many_arguments)]
pub unsafe fn silk_noise_shape_quantizer_del_dec_sse4_1(
    nsq: &mut silk_nsq_state,
    ps_del_dec: &mut [NSQ_del_dec_struct],
    signal_type: i32,
    x_q10: &[i32],
    pulses: &mut [i8],
    pulses_off: usize,
    xq_off: usize,
    s_ltp_q15: &mut [i32],
    delayed_gain_q10: &mut [i32; 40],
    a_q12: &[i16],
    b_q14: &[i16],
    ar_shp_q13: &[i16],
    lag: i32,
    harm_shape_firpacked_q14: i32,
    tilt_q14: i32,
    lf_shp_q14: i32,
    gain_q16: i32,
    lambda_q10: i32,
    offset_q10: i32,
    length: i32,
    subfr: i32,
    shaping_lpcorder: i32,
    predict_lpcorder: i32,
    warping_q16: i32,
    n_states_delayed_decision: i32,
    smpl_buf_idx: &mut i32,
    decision_delay: i32,
) {
    let n_states = n_states_delayed_decision as usize;
    let mut ps_sample_state: Vec<NsqSamplePair> = vec![[NSQ_sample_struct::default(); 2]; n_states];

    let mut shp_lag_idx = (nsq.s_ltp_shp_buf_idx - lag + HARM_SHAPE_FIR_TAPS / 2) as usize;
    let mut pred_lag_idx = (nsq.s_ltp_buf_idx - lag + LTP_ORDER as i32 / 2) as usize;
    let gain_q10: i32 = gain_q16 >> 6;

    let rdo_offset = (lambda_q10 >> 1) - 512;

    // Pre-load a_q12 coefficients into SIMD registers
    let a_q12_0123 = _mm_cvtepi16_epi32(_mm_loadl_epi64(a_q12.as_ptr() as *const __m128i));
    let a_q12_4567 = _mm_cvtepi16_epi32(_mm_loadl_epi64(a_q12.as_ptr().add(4) as *const __m128i));
    let (a_q12_89_ab, a_q12_cdef) = if predict_lpcorder == 16 {
        (
            _mm_cvtepi16_epi32(_mm_loadl_epi64(a_q12.as_ptr().add(8) as *const __m128i)),
            _mm_cvtepi16_epi32(_mm_loadl_epi64(a_q12.as_ptr().add(12) as *const __m128i)),
        )
    } else {
        (_mm_setzero_si128(), _mm_setzero_si128())
    };

    // Pre-load b_q14 for LTP
    let b_q14_0123 = if signal_type == TYPE_VOICED {
        _mm_cvtepi16_epi32(_mm_loadl_epi64(b_q14.as_ptr() as *const __m128i))
    } else {
        _mm_setzero_si128()
    };

    for (_i, &x_q10_i) in x_q10.iter().take(length as usize).enumerate() {
        // ---- LTP prediction (SIMD for 4 taps + 1 scalar) ----
        let mut ltp_pred_q14: i32;
        if signal_type == TYPE_VOICED {
            ltp_pred_q14 = 2;
            let pred_vals =
                _mm_loadu_si128(s_ltp_q15.as_ptr().add(pred_lag_idx - 3) as *const __m128i);
            let pred_rev = _mm_shuffle_epi32(pred_vals, 0x1B);
            let tmpa = _mm_srli_epi64(_mm_mul_epi32(pred_rev, b_q14_0123), 16);
            let pred_rot = _mm_shuffle_epi32(pred_rev, 0x39);
            let b_rot = _mm_shuffle_epi32(b_q14_0123, 0x39);
            let tmpb = _mm_srli_epi64(_mm_mul_epi32(pred_rot, b_rot), 16);
            let sum4 = _mm_add_epi32(tmpa, tmpb);
            let sum2 = _mm_add_epi32(sum4, _mm_shuffle_epi32(sum4, 0x0E));
            ltp_pred_q14 += _mm_cvtsi128_si32(sum2);
            ltp_pred_q14 = (ltp_pred_q14 as i64
                + ((s_ltp_q15[pred_lag_idx - 4] as i64 * b_q14[4] as i64) >> 16))
                as i32;
            ltp_pred_q14 = ((ltp_pred_q14 as u32) << 1) as i32;
            pred_lag_idx += 1;
        } else {
            ltp_pred_q14 = 0;
        }

        // ---- Harmonic noise shaping (scalar, shared across states) ----
        let n_ltp_q14: i32;
        if lag > 0 {
            n_ltp_q14 = {
                let t = ((nsq.s_ltp_shp_q14[shp_lag_idx]
                    .saturating_add(nsq.s_ltp_shp_q14[shp_lag_idx - 2]))
                    as i64
                    * harm_shape_firpacked_q14 as i16 as i64)
                    >> 16;
                let t2 = (t
                    + ((nsq.s_ltp_shp_q14[shp_lag_idx - 1] as i64
                        * (harm_shape_firpacked_q14 as i64 >> 16))
                        >> 16)) as i32;
                ltp_pred_q14 - ((t2 as u32) << 2) as i32
            };
            shp_lag_idx += 1;
        } else {
            n_ltp_q14 = 0;
        }

        // ---- Per-state processing ----
        for k in 0..n_states {
            let ps_dd = &mut ps_del_dec[k];
            ps_dd.seed = silk_rand(ps_dd.seed);

            // ---- SIMD LPC prediction ----
            let lpc_idx = NSQ_LPC_BUF_LENGTH - 1 + _i;
            let ps_lpc_ptr = ps_dd.s_lpc_q14.as_ptr().add(lpc_idx);
            let mut lpc_pred_q14: i32 = predict_lpcorder >> 1;

            let mut acc = _mm_setzero_si128();

            // Step 1: coefficients 0-3
            let lpc_vals = _mm_loadu_si128(ps_lpc_ptr.sub(3) as *const __m128i);
            let lpc_rev = _mm_shuffle_epi32(lpc_vals, 0x1B);
            acc = _mm_add_epi32(acc, _mm_srli_epi64(_mm_mul_epi32(lpc_rev, a_q12_0123), 16));
            let lpc_rot = _mm_shuffle_epi32(lpc_rev, 0x39);
            let a_rot = _mm_shuffle_epi32(a_q12_0123, 0x39);
            acc = _mm_add_epi32(acc, _mm_srli_epi64(_mm_mul_epi32(lpc_rot, a_rot), 16));

            // Step 2: coefficients 4-7
            let lpc_vals = _mm_loadu_si128(ps_lpc_ptr.sub(7) as *const __m128i);
            let lpc_rev = _mm_shuffle_epi32(lpc_vals, 0x1B);
            acc = _mm_add_epi32(acc, _mm_srli_epi64(_mm_mul_epi32(lpc_rev, a_q12_4567), 16));
            let lpc_rot = _mm_shuffle_epi32(lpc_rev, 0x39);
            let a_rot = _mm_shuffle_epi32(a_q12_4567, 0x39);
            acc = _mm_add_epi32(acc, _mm_srli_epi64(_mm_mul_epi32(lpc_rot, a_rot), 16));

            if predict_lpcorder == 16 {
                // Step 3: coefficients 8-11
                let lpc_vals = _mm_loadu_si128(ps_lpc_ptr.sub(11) as *const __m128i);
                let lpc_rev = _mm_shuffle_epi32(lpc_vals, 0x1B);
                acc = _mm_add_epi32(acc, _mm_srli_epi64(_mm_mul_epi32(lpc_rev, a_q12_89_ab), 16));
                let lpc_rot = _mm_shuffle_epi32(lpc_rev, 0x39);
                let a_rot = _mm_shuffle_epi32(a_q12_89_ab, 0x39);
                acc = _mm_add_epi32(acc, _mm_srli_epi64(_mm_mul_epi32(lpc_rot, a_rot), 16));

                // Step 4: coefficients 12-15
                let lpc_vals = _mm_loadu_si128(ps_lpc_ptr.sub(15) as *const __m128i);
                let lpc_rev = _mm_shuffle_epi32(lpc_vals, 0x1B);
                acc = _mm_add_epi32(acc, _mm_srli_epi64(_mm_mul_epi32(lpc_rev, a_q12_cdef), 16));
                let lpc_rot = _mm_shuffle_epi32(lpc_rev, 0x39);
                let a_rot = _mm_shuffle_epi32(a_q12_cdef, 0x39);
                acc = _mm_add_epi32(acc, _mm_srli_epi64(_mm_mul_epi32(lpc_rot, a_rot), 16));

                let hi = _mm_shuffle_epi32(acc, 0x0E);
                acc = _mm_add_epi32(acc, hi);
                lpc_pred_q14 += _mm_cvtsi128_si32(acc);
            } else {
                let hi = _mm_shuffle_epi32(acc, 0x0E);
                acc = _mm_add_epi32(acc, hi);
                lpc_pred_q14 += _mm_cvtsi128_si32(acc);
                lpc_pred_q14 = (lpc_pred_q14 as i64
                    + ((*ps_lpc_ptr.sub(8) as i64 * a_q12[8] as i64) >> 16))
                    as i32;
                lpc_pred_q14 = (lpc_pred_q14 as i64
                    + ((*ps_lpc_ptr.sub(9) as i64 * a_q12[9] as i64) >> 16))
                    as i32;
            }

            lpc_pred_q14 = ((lpc_pred_q14 as u32) << 4) as i32;

            // ---- Noise shaping with warping (scalar) ----
            let mut tmp2 = (ps_dd.diff_q14 as i64
                + ((ps_dd.s_ar2_q14[0] as i64 * warping_q16 as i16 as i64) >> 16))
                as i32;
            let mut tmp1 = (ps_dd.s_ar2_q14[0] as i64
                + (((ps_dd.s_ar2_q14[1].wrapping_sub(tmp2)) as i64 * warping_q16 as i16 as i64)
                    >> 16)) as i32;
            ps_dd.s_ar2_q14[0] = tmp2;
            let mut n_ar_q14: i32 = shaping_lpcorder >> 1;
            n_ar_q14 = (n_ar_q14 as i64 + ((tmp2 as i64 * ar_shp_q13[0] as i64) >> 16)) as i32;
            let mut j = 2;
            while j < shaping_lpcorder {
                tmp2 = (ps_dd.s_ar2_q14[(j - 1) as usize] as i64
                    + (((ps_dd.s_ar2_q14[j as usize].wrapping_sub(tmp1)) as i64
                        * warping_q16 as i16 as i64)
                        >> 16)) as i32;
                ps_dd.s_ar2_q14[(j - 1) as usize] = tmp1;
                n_ar_q14 = (n_ar_q14 as i64
                    + ((tmp1 as i64 * ar_shp_q13[(j - 1) as usize] as i64) >> 16))
                    as i32;
                tmp1 = (ps_dd.s_ar2_q14[j as usize] as i64
                    + (((ps_dd.s_ar2_q14[(j + 1) as usize].wrapping_sub(tmp2)) as i64
                        * warping_q16 as i16 as i64)
                        >> 16)) as i32;
                ps_dd.s_ar2_q14[j as usize] = tmp2;
                n_ar_q14 = (n_ar_q14 as i64 + ((tmp2 as i64 * ar_shp_q13[j as usize] as i64) >> 16))
                    as i32;
                j += 2;
            }
            ps_dd.s_ar2_q14[(shaping_lpcorder - 1) as usize] = tmp1;
            n_ar_q14 = (n_ar_q14 as i64
                + ((tmp1 as i64 * ar_shp_q13[(shaping_lpcorder - 1) as usize] as i64) >> 16))
                as i32;
            n_ar_q14 = ((n_ar_q14 as u32) << 1) as i32;
            n_ar_q14 = (n_ar_q14 as i64 + ((ps_dd.lf_ar_q14 as i64 * tilt_q14 as i16 as i64) >> 16))
                as i32;
            n_ar_q14 = ((n_ar_q14 as u32) << 2) as i32;

            let n_lf_q14: i32 = {
                let t1 = ((ps_dd.shape_q14[*smpl_buf_idx as usize] as i64
                    * lf_shp_q14 as i16 as i64)
                    >> 16) as i32;
                let t2 = (t1 as i64 + ((ps_dd.lf_ar_q14 as i64 * (lf_shp_q14 as i64 >> 16)) >> 16))
                    as i32;
                ((t2 as u32) << 2) as i32
            };

            // ---- Combine prediction and noise feedback ----
            tmp1 = n_ar_q14.saturating_add(n_lf_q14);
            tmp2 = n_ltp_q14 + lpc_pred_q14;
            tmp1 = tmp2.saturating_sub(tmp1);
            tmp1 = ((tmp1 >> 3) + 1) >> 1;

            let mut r_q10 = x_q10_i - tmp1;
            if ps_dd.seed < 0 {
                r_q10 = -r_q10;
            }
            r_q10 = r_q10.clamp(-(31 << 10), 30 << 10);

            // ---- Quantization decision ----
            let mut q1_q10 = r_q10 - offset_q10;
            let mut q1_q0 = q1_q10 >> 10;
            if lambda_q10 > 2048 {
                if q1_q10 > rdo_offset {
                    q1_q0 = (q1_q10 - rdo_offset) >> 10;
                } else if q1_q10 < -rdo_offset {
                    q1_q0 = (q1_q10 + rdo_offset) >> 10;
                } else if q1_q10 < 0 {
                    q1_q0 = -1;
                } else {
                    q1_q0 = 0;
                }
            }
            let q2_q10: i32;
            let rd1_q10: i32;
            let rd2_q10: i32;
            if q1_q0 > 0 {
                q1_q10 = ((q1_q0 as u32) << 10) as i32 - 80 + offset_q10;
                q2_q10 = q1_q10 + 1024;
                rd1_q10 = q1_q10 as i16 as i32 * lambda_q10 as i16 as i32;
                rd2_q10 = q2_q10 as i16 as i32 * lambda_q10 as i16 as i32;
            } else if q1_q0 == 0 {
                q1_q10 = offset_q10;
                q2_q10 = q1_q10 + (1024 - 80);
                rd1_q10 = q1_q10 as i16 as i32 * lambda_q10 as i16 as i32;
                rd2_q10 = q2_q10 as i16 as i32 * lambda_q10 as i16 as i32;
            } else if q1_q0 == -1 {
                q2_q10 = offset_q10;
                q1_q10 = q2_q10 - (1024 - 80);
                rd1_q10 = -q1_q10 as i16 as i32 * lambda_q10 as i16 as i32;
                rd2_q10 = q2_q10 as i16 as i32 * lambda_q10 as i16 as i32;
            } else {
                q1_q10 = ((q1_q0 as u32) << 10) as i32 + 80 + offset_q10;
                q2_q10 = q1_q10 + 1024;
                rd1_q10 = -q1_q10 as i16 as i32 * lambda_q10 as i16 as i32;
                rd2_q10 = -q2_q10 as i16 as i32 * lambda_q10 as i16 as i32;
            }
            let mut rr_q10 = r_q10 - q1_q10;
            let rd1_q10 = (rd1_q10 + rr_q10 as i16 as i32 * rr_q10 as i16 as i32) >> 10;
            rr_q10 = r_q10 - q2_q10;
            let rd2_q10 = (rd2_q10 + rr_q10 as i16 as i32 * rr_q10 as i16 as i32) >> 10;

            if rd1_q10 < rd2_q10 {
                ps_sample_state[k][0].rd_q10 = ps_dd.rd_q10 + rd1_q10;
                ps_sample_state[k][1].rd_q10 = ps_dd.rd_q10 + rd2_q10;
                ps_sample_state[k][0].q_q10 = q1_q10;
                ps_sample_state[k][1].q_q10 = q2_q10;
            } else {
                ps_sample_state[k][0].rd_q10 = ps_dd.rd_q10 + rd2_q10;
                ps_sample_state[k][1].rd_q10 = ps_dd.rd_q10 + rd1_q10;
                ps_sample_state[k][0].q_q10 = q2_q10;
                ps_sample_state[k][1].q_q10 = q1_q10;
            }

            // Compute outputs for best and second-best
            let mut exc_q14 = ((ps_sample_state[k][0].q_q10 as u32) << 4) as i32;
            if ps_dd.seed < 0 {
                exc_q14 = -exc_q14;
            }
            let mut lpc_exc_q14 = exc_q14 + ltp_pred_q14;
            let mut xq_q14 = lpc_exc_q14 + lpc_pred_q14;
            ps_sample_state[k][0].diff_q14 = xq_q14 - ((x_q10_i as u32) << 4) as i32;
            let mut s_lf_ar_shp_q14 = ps_sample_state[k][0].diff_q14 - n_ar_q14;
            ps_sample_state[k][0].s_ltp_shp_q14 = s_lf_ar_shp_q14.saturating_sub(n_lf_q14);
            ps_sample_state[k][0].lf_ar_q14 = s_lf_ar_shp_q14;
            ps_sample_state[k][0].lpc_exc_q14 = lpc_exc_q14;
            ps_sample_state[k][0].xq_q14 = xq_q14;

            exc_q14 = ((ps_sample_state[k][1].q_q10 as u32) << 4) as i32;
            if ps_dd.seed < 0 {
                exc_q14 = -exc_q14;
            }
            lpc_exc_q14 = exc_q14 + ltp_pred_q14;
            xq_q14 = lpc_exc_q14 + lpc_pred_q14;
            ps_sample_state[k][1].diff_q14 = xq_q14 - ((x_q10_i as u32) << 4) as i32;
            s_lf_ar_shp_q14 = ps_sample_state[k][1].diff_q14 - n_ar_q14;
            ps_sample_state[k][1].s_ltp_shp_q14 = s_lf_ar_shp_q14.saturating_sub(n_lf_q14);
            ps_sample_state[k][1].lf_ar_q14 = s_lf_ar_shp_q14;
            ps_sample_state[k][1].lpc_exc_q14 = lpc_exc_q14;
            ps_sample_state[k][1].xq_q14 = xq_q14;
        }

        // ---- Winner selection, pruning, output ----
        *smpl_buf_idx = (*smpl_buf_idx - 1) % DECISION_DELAY;
        if *smpl_buf_idx < 0 {
            *smpl_buf_idx += DECISION_DELAY;
        }
        let last_smple_idx = (*smpl_buf_idx + decision_delay) % DECISION_DELAY;

        let mut rdmin_q10 = ps_sample_state[0][0].rd_q10;
        let mut winner_ind: i32 = 0;
        for (k, sample_state) in ps_sample_state.iter().take(n_states).enumerate().skip(1) {
            if sample_state[0].rd_q10 < rdmin_q10 {
                rdmin_q10 = sample_state[0].rd_q10;
                winner_ind = k as i32;
            }
        }

        let winner_rand_state = ps_del_dec[winner_ind as usize].rand_state[last_smple_idx as usize];
        for k in 0..n_states {
            if ps_del_dec[k].rand_state[last_smple_idx as usize] != winner_rand_state {
                ps_sample_state[k][0].rd_q10 += 0x7fffffff >> 4;
                ps_sample_state[k][1].rd_q10 += 0x7fffffff >> 4;
            }
        }

        let mut rdmax_q10 = ps_sample_state[0][0].rd_q10;
        rdmin_q10 = ps_sample_state[0][1].rd_q10;
        let mut rdmax_ind: i32 = 0;
        let mut rdmin_ind: i32 = 0;
        for (k, sample_state) in ps_sample_state.iter().take(n_states).enumerate().skip(1) {
            if sample_state[0].rd_q10 > rdmax_q10 {
                rdmax_q10 = sample_state[0].rd_q10;
                rdmax_ind = k as i32;
            }
            if sample_state[1].rd_q10 < rdmin_q10 {
                rdmin_q10 = sample_state[1].rd_q10;
                rdmin_ind = k as i32;
            }
        }

        if rdmin_q10 < rdmax_q10 {
            if rdmax_ind != rdmin_ind {
                let (left, right) = if rdmax_ind < rdmin_ind {
                    let (left_states, right_states) = ps_del_dec.split_at_mut(rdmin_ind as usize);
                    (&mut left_states[rdmax_ind as usize], &right_states[0])
                } else {
                    let (left_states, right_states) = ps_del_dec.split_at_mut(rdmax_ind as usize);
                    (&mut right_states[0], &left_states[rdmin_ind as usize])
                };
                copy_del_dec_state_partial(left, right, _i);
            }
            ps_sample_state[rdmax_ind as usize][0] = ps_sample_state[rdmin_ind as usize][1];
        }

        if subfr > 0 || _i as i32 >= decision_delay {
            let ps_dd_w = &ps_del_dec[winner_ind as usize];
            let out_idx = pulses_off + _i - decision_delay as usize;
            pulses[out_idx] = (((ps_dd_w.q_q10[last_smple_idx as usize] >> 9) + 1) >> 1) as i8;
            let xq_val = (ps_dd_w.xq_q14[last_smple_idx as usize] as i64
                * delayed_gain_q10[last_smple_idx as usize] as i64)
                >> 16;
            let rounded = ((xq_val as i32 >> 7) + 1) >> 1;
            nsq.xq[xq_off + _i - decision_delay as usize] = rounded.clamp(-32768, 32767) as i16;
            nsq.s_ltp_shp_q14[(nsq.s_ltp_shp_buf_idx - decision_delay) as usize] =
                ps_dd_w.shape_q14[last_smple_idx as usize];
            s_ltp_q15[(nsq.s_ltp_buf_idx - decision_delay) as usize] =
                ps_dd_w.pred_q15[last_smple_idx as usize];
        }
        nsq.s_ltp_shp_buf_idx += 1;
        nsq.s_ltp_buf_idx += 1;

        for k in 0..n_states {
            let ps_ss = &ps_sample_state[k][0];
            let ps_dd = &mut ps_del_dec[k];
            ps_dd.lf_ar_q14 = ps_ss.lf_ar_q14;
            ps_dd.diff_q14 = ps_ss.diff_q14;
            ps_dd.s_lpc_q14[NSQ_LPC_BUF_LENGTH + _i] = ps_ss.xq_q14;
            ps_dd.xq_q14[*smpl_buf_idx as usize] = ps_ss.xq_q14;
            ps_dd.q_q10[*smpl_buf_idx as usize] = ps_ss.q_q10;
            ps_dd.pred_q15[*smpl_buf_idx as usize] = ((ps_ss.lpc_exc_q14 as u32) << 1) as i32;
            ps_dd.shape_q14[*smpl_buf_idx as usize] = ps_ss.s_ltp_shp_q14;
            ps_dd.seed =
                (ps_dd.seed as u32).wrapping_add((((ps_ss.q_q10 >> 9) + 1) >> 1) as u32) as i32;
            ps_dd.rand_state[*smpl_buf_idx as usize] = ps_dd.seed;
            ps_dd.rd_q10 = ps_ss.rd_q10;
        }
        delayed_gain_q10[*smpl_buf_idx as usize] = gain_q10;
    }

    for dd in ps_del_dec[..n_states].iter_mut() {
        dd.s_lpc_q14
            .copy_within(length as usize..length as usize + NSQ_LPC_BUF_LENGTH, 0);
    }
}

/// SSE4.1 implementation of `silk_VQ_WMat_EC`.
/// Port of `silk/x86/VQ_WMat_EC_sse4_1.c`.
///
/// Entropy-constrained matrix-weighted VQ for 5-element LTP coefficient vectors.
/// The SSE4.1 optimization accelerates the first row's off-diagonal dot product
/// using `_mm_mul_epi32` (32x32→64 widening multiply).
///
/// # Safety
/// Requires SSE4.1 support (checked by caller via cpufeatures).
#[target_feature(enable = "sse4.1")]
#[allow(clippy::too_many_arguments)]
pub unsafe fn silk_vq_wmat_ec_sse4_1(
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
) {
    let mut neg_x_x_q24: [i32; 5] = [0; 5];
    neg_x_x_q24[0] = -(((x_x_q17[0] as u32) << 7) as i32);
    neg_x_x_q24[1] = -(((x_x_q17[1] as u32) << 7) as i32);
    neg_x_x_q24[2] = -(((x_x_q17[2] as u32) << 7) as i32);
    neg_x_x_q24[3] = -(((x_x_q17[3] as u32) << 7) as i32);
    neg_x_x_q24[4] = -(((x_x_q17[4] as u32) << 7) as i32);

    // Load XX_Q17[1..5] and create two shuffled views for the first row SIMD computation
    // v_XX_31_Q17 = [XX_Q17[1], XX_Q17[2], XX_Q17[3], XX_Q17[4]]
    let v_xx_31_q17 = _mm_loadu_si128(xx_q17.as_ptr().add(1) as *const __m128i);
    // v_XX_42_Q17 = [XX_Q17[2], XX_Q17[3], XX_Q17[4], XX_Q17[1]]
    let v_xx_42_q17 = _mm_shuffle_epi32(v_xx_31_q17, 0x39); // _MM_SHUFFLE(0,3,2,1)

    *rate_dist_q8 = i32::MAX;
    *res_nrg_q15 = i32::MAX;
    *ind = 0;
    let mut cb_row_off: usize = 0;

    for k in 0..l as usize {
        let gain_tmp_q7 = cb_gain_q7[k] as i32;
        let mut sum1_q15: i32 = (1.001f64 * ((1) << 15) as f64 + 0.5f64) as i32;

        let penalty: i32 = (((if gain_tmp_q7 - max_gain_q7 > 0 {
            gain_tmp_q7 - max_gain_q7
        } else {
            0
        }) as u32)
            << 11) as i32;

        // First row of XX_Q17 — SIMD accelerated
        // Sign-extend cb_row_Q7[1..5] from i8 to i32
        let cb_ptr = cb_q7.as_ptr().add(cb_row_off + 1);
        let v_cb_row_31_q7 =
            _mm_cvtepi8_epi32(_mm_cvtsi32_si128((cb_ptr as *const i32).read_unaligned()));
        let v_cb_row_42_q7 = _mm_shuffle_epi32(v_cb_row_31_q7, 0x39);

        // Widening multiply: XX_Q17[_i] * cb_Q7[j] -> i64, then horizontal sum
        let v_prod_31 = _mm_mul_epi32(v_xx_31_q17, v_cb_row_31_q7);
        let v_prod_42 = _mm_mul_epi32(v_xx_42_q17, v_cb_row_42_q7);
        let v_acc1 = _mm_add_epi64(v_prod_31, v_prod_42);
        let v_acc2 = _mm_shuffle_epi32(v_acc1, 0x4E); // swap hi/lo 64-bit
        let v_acc1 = _mm_add_epi64(v_acc1, v_acc2);
        let mut sum2_q24: i32 = _mm_cvtsi128_si32(v_acc1);

        sum2_q24 = neg_x_x_q24[0].wrapping_add(sum2_q24);
        sum2_q24 = ((sum2_q24 as u32) << 1) as i32;
        sum2_q24 = sum2_q24.wrapping_add(xx_q17[0].wrapping_mul(cb_q7[cb_row_off] as i32));
        sum1_q15 =
            (sum1_q15 as i64 + ((sum2_q24 as i64 * cb_q7[cb_row_off] as i16 as i64) >> 16)) as i32;

        // Rows 2-5: scalar (same as scalar version)
        sum2_q24 = neg_x_x_q24[1] + xx_q17[7] * cb_q7[cb_row_off + 2] as i32;
        sum2_q24 += xx_q17[8] * cb_q7[cb_row_off + 3] as i32;
        sum2_q24 += xx_q17[9] * cb_q7[cb_row_off + 4] as i32;
        sum2_q24 = ((sum2_q24 as u32) << 1) as i32;
        sum2_q24 += xx_q17[6] * cb_q7[cb_row_off + 1] as i32;
        sum1_q15 = (sum1_q15 as i64
            + ((sum2_q24 as i64 * cb_q7[cb_row_off + 1] as i16 as i64) >> 16))
            as i32;

        sum2_q24 = neg_x_x_q24[2] + xx_q17[13] * cb_q7[cb_row_off + 3] as i32;
        sum2_q24 += xx_q17[14] * cb_q7[cb_row_off + 4] as i32;
        sum2_q24 = ((sum2_q24 as u32) << 1) as i32;
        sum2_q24 += xx_q17[12] * cb_q7[cb_row_off + 2] as i32;
        sum1_q15 = (sum1_q15 as i64
            + ((sum2_q24 as i64 * cb_q7[cb_row_off + 2] as i16 as i64) >> 16))
            as i32;

        sum2_q24 = neg_x_x_q24[3] + xx_q17[19] * cb_q7[cb_row_off + 4] as i32;
        sum2_q24 = ((sum2_q24 as u32) << 1) as i32;
        sum2_q24 += xx_q17[18] * cb_q7[cb_row_off + 3] as i32;
        sum1_q15 = (sum1_q15 as i64
            + ((sum2_q24 as i64 * cb_q7[cb_row_off + 3] as i16 as i64) >> 16))
            as i32;

        sum2_q24 = ((neg_x_x_q24[4] as u32) << 1) as i32;
        sum2_q24 += xx_q17[24] * cb_q7[cb_row_off + 4] as i32;
        sum1_q15 = (sum1_q15 as i64
            + ((sum2_q24 as i64 * cb_q7[cb_row_off + 4] as i16 as i64) >> 16))
            as i32;

        if sum1_q15 >= 0 {
            let bits_res_q8 = subfr_len as i16 as i32
                * (crate::silk::lin2log::silk_lin2log(sum1_q15 + penalty) - ((15) << 7)) as i16
                    as i32;
            let bits_tot_q8 = bits_res_q8 + ((cl_q5[k] as u32) << (3 - 1)) as i32;
            if bits_tot_q8 <= *rate_dist_q8 {
                *rate_dist_q8 = bits_tot_q8;
                *res_nrg_q15 = sum1_q15 + penalty;
                *ind = k as i8;
                *gain_q7 = gain_tmp_q7;
            }
        }
        cb_row_off += LTP_ORDER;
    }
}

// ============================================================================
// AVX2 nsq del_dec implementation
// Port of silk/x86/NSQ_del_dec_avx2.c
// ============================================================================

use crate::silk::define::MAX_SUB_FRAME_LENGTH;
const RAND_MULTIPLIER_I32: i32 = 196314165;
const RAND_INCREMENT_I32: i32 = 907633515;

/// Extract high 32 bits of each 64-bit lane from a 256-bit vector.
/// Equivalent to c: `silk_cvtepi64_epi32_high`
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn silk_cvtepi64_epi32_high(num: __m256i) -> __m128i {
    _mm256_castsi256_si128(_mm256_permutevar8x32_epi32(
        num,
        _mm256_set_epi32(0, 0, 0, 0, 7, 5, 3, 1),
    ))
}

/// Saturate i32 to i16 range.
#[inline]
fn silk_sat16(num: i32) -> i16 {
    let num = if num > i16::MAX as i32 {
        i16::MAX as i32
    } else {
        num
    };
    let num = if num < i16::MIN as i32 {
        i16::MIN as i32
    } else {
        num
    };
    num as i16
}

/// Shift right with rounding: (a + (1 << (bits-1))) >> bits
#[inline]
fn silk_sar_round_32(a: i32, bits: i32) -> i32 {
    debug_assert!(bits > 0 && bits < 31);
    let a = a.wrapping_add(1 << (bits - 1));
    a >> bits
}

/// Multiply and shift with rounding: ((a as i64) * (b as i64) + (1 << (bits+15))) >> (bits + 16)
#[inline]
fn silk_sar_round_smulww(a: i32, b: i32, bits: i32) -> i64 {
    debug_assert!(bits > 0 && bits < 63);
    let t: i64 = (a as i64) * (b as i64);
    let total_bits = bits + 16;
    let round = (1u64 as i64) << (total_bits - 1);
    (t.wrapping_add(round)) >> total_bits
}

/// Saturating add for i32.
#[inline]
fn silk_add_sat32(a: i32, b: i32) -> i32 {
    match a.checked_add(b) {
        Some(sum) => sum,
        None => {
            if a >= 0 {
                i32::MAX
            } else {
                i32::MIN
            }
        }
    }
}

/// SIMD shift right with rounding for 4 x i32.
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn silk_mm_srai_round_epi32(a: __m128i, bits: i32) -> __m128i {
    debug_assert!(bits > 0 && bits < 31);
    _mm_sra_epi32(
        _mm_add_epi32(a, _mm_set1_epi32(1 << (bits - 1))),
        _mm_cvtsi32_si128(bits),
    )
}

/// SIMD saturating add for 4 x i32.
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn silk_mm_add_sat_epi32(a: __m128i, b: __m128i) -> __m128i {
    let r = _mm_add_epi32(a, b);
    let of = _mm_and_si128(_mm_xor_si128(a, r), _mm_xor_si128(b, r));
    let sat = _mm_add_epi32(_mm_srli_epi32(a, 31), _mm_set1_epi32(0x7FFFFFFF));
    _mm_blendv_epi8(r, sat, _mm_srai_epi32(of, 31))
}

/// SIMD saturating sub for 4 x i32.
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn silk_mm_sub_sat_epi32(a: __m128i, b: __m128i) -> __m128i {
    let r = _mm_sub_epi32(a, b);
    let of = _mm_andnot_si128(_mm_xor_si128(b, r), _mm_xor_si128(a, r));
    let sat = _mm_add_epi32(_mm_srli_epi32(a, 31), _mm_set1_epi32(0x7FFFFFFF));
    _mm_blendv_epi8(r, sat, _mm_srai_epi32(of, 31))
}

/// SIMD saturating sub for 8 x i32 (256-bit).
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn silk_mm256_sub_sat_epi32(a: __m256i, b: __m256i) -> __m256i {
    let r = _mm256_sub_epi32(a, b);
    let of = _mm256_andnot_si256(_mm256_xor_si256(b, r), _mm256_xor_si256(a, r));
    let sat = _mm256_add_epi32(_mm256_srli_epi32(a, 31), _mm256_set1_epi32(0x7FFFFFFF));
    _mm256_blendv_epi8(r, sat, _mm256_srai_epi32(of, 31))
}

/// Clamp each 32-bit lane to [min(limit1,limit2), max(limit1,limit2)].
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn silk_mm_limit_epi32(num: __m128i, limit1: i32, limit2: i32) -> __m128i {
    let lo = limit1.min(limit2);
    let hi = limit1.max(limit2);
    let num = _mm_min_epi32(num, _mm_set1_epi32(hi));
    _mm_max_epi32(num, _mm_set1_epi32(lo))
}

/// Conditional negate: if cond < 0 then -num, else num.
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn silk_mm_sign_epi32(num: __m128i, cond: __m128i) -> __m128i {
    _mm_sign_epi32(num, _mm_or_si128(cond, _mm_set1_epi32(1)))
}

/// 256-bit conditional negate: if cond < 0 then -num, else num.
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn silk_mm256_sign_epi32(num: __m256i, cond: __m256i) -> __m256i {
    _mm256_sign_epi32(num, _mm256_or_si256(cond, _mm256_set1_epi32(1)))
}

/// (a32 * b32) >> 16  for 4 x i32 (sign-extended to 64-bit, multiply, shift, pack back)
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn silk_mm_smulww_epi32(a: __m128i, b: i32) -> __m128i {
    silk_cvtepi64_epi32_high(_mm256_slli_epi64(
        _mm256_mul_epi32(_mm256_cvtepi32_epi64(a), _mm256_set1_epi32(b)),
        16,
    ))
}

/// (a32 * (i16)(b32)) >> 16  for 4 x i32
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn silk_mm_smulwb_epi32(a: __m128i, b: i32) -> __m128i {
    silk_cvtepi64_epi32_high(_mm256_mul_epi32(
        _mm256_cvtepi32_epi64(a),
        _mm256_set1_epi32((b as u32).wrapping_shl(16) as i32),
    ))
}

/// i16 x i16 -> i32 multiply (low 16 bits of each 32-bit lane)
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn silk_mm256_smulbb_epi32(a: __m256i, b: __m256i) -> __m256i {
    let ff: i8 = -1; // 0xFF
    let msk = _mm256_set_epi8(
        ff, ff, ff, ff, ff, ff, ff, ff, 13, 12, 9, 8, 5, 4, 1, 0, ff, ff, ff, ff, ff, ff, ff, ff,
        13, 12, 9, 8, 5, 4, 1, 0,
    );
    let lo = _mm256_mullo_epi16(a, b);
    let hi = _mm256_mulhi_epi16(a, b);
    let lo = _mm256_shuffle_epi8(lo, msk);
    let hi = _mm256_shuffle_epi8(hi, msk);
    _mm256_unpacklo_epi16(lo, hi)
}

/// Reverse 8 x i32 elements in a 256-bit vector.
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn silk_mm256_reverse_epi32(v: __m256i) -> __m256i {
    let v = _mm256_shuffle_epi32(v, 0x1B);
    _mm256_permute4x64_epi64(v, 0x4E)
}

/// Horizontal sum of 8 x i32 in a 256-bit vector.
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn silk_mm256_hsum_epi32(v: __m256i) -> i32 {
    let sum = _mm_add_epi32(
        _mm256_extracti128_si256(v, 1),
        _mm256_extracti128_si256(v, 0),
    );
    let sum = _mm_add_epi32(sum, _mm_shuffle_epi32(sum, 0x4E));
    let sum = _mm_add_epi32(sum, _mm_shuffle_epi32(sum, 0xB1));
    _mm_cvtsi128_si32(sum)
}

/// Horizontal min of 4 x i32.
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn silk_mm_hmin_epi32(num: __m128i) -> __m128i {
    let num = _mm_min_epi32(num, _mm_shuffle_epi32(num, 0x4E));
    _mm_min_epi32(num, _mm_shuffle_epi32(num, 0xB1))
}

/// Horizontal max of 4 x i32.
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn silk_mm_hmax_epi32(num: __m128i) -> __m128i {
    let num = _mm_max_epi32(num, _mm_shuffle_epi32(num, 0x4E));
    _mm_max_epi32(num, _mm_shuffle_epi32(num, 0xB1))
}

/// Horizontal min of 4 x i32, with masked lanes set to MAX.
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn silk_mm_mask_hmin_epi32(num: __m128i, mask: __m128i) -> __m128i {
    let num = _mm_blendv_epi8(num, _mm_set1_epi32(i32::MAX), mask);
    silk_mm_hmin_epi32(num)
}

/// Horizontal max of 4 x i32, with masked lanes set to MIN.
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn silk_mm_mask_hmax_epi32(num: __m128i, mask: __m128i) -> __m128i {
    let num = _mm_blendv_epi8(num, _mm_set1_epi32(i32::MIN), mask);
    silk_mm_hmax_epi32(num)
}

/// SIMD RNG: seed = seed * RAND_MULTIPLIER + RAND_INCREMENT
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn silk_mm256_rand_epi32(seed: __m128i) -> __m128i {
    let seed = _mm_mullo_epi32(seed, _mm_set1_epi32(RAND_MULTIPLIER_I32));
    _mm_add_epi32(seed, _mm_set1_epi32(RAND_INCREMENT_I32))
}

/// Find the index (0-3) of the first lane in `a` that equals `b` (broadcast min/max).
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn silk_index_of_first_equal_epi32(a: __m128i, b: __m128i) -> i32 {
    let mask = _mm_movemask_epi8(_mm_cmpeq_epi32(a, b)) as u32 & 0x1111;
    debug_assert!(mask != 0);
    mask.trailing_zeros() as i32 >> 2
}

/// Convert a lane index (0-3) to a byte shuffle selector.
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn silk_index_to_selector(index: i32) -> __m128i {
    debug_assert!(index < 4);
    let index = index << 2;
    _mm_set_epi8(
        (index + 3) as i8,
        (index + 2) as i8,
        (index + 1) as i8,
        index as i8,
        (index + 3) as i8,
        (index + 2) as i8,
        (index + 1) as i8,
        index as i8,
        (index + 3) as i8,
        (index + 2) as i8,
        (index + 1) as i8,
        index as i8,
        (index + 3) as i8,
        (index + 2) as i8,
        (index + 1) as i8,
        index as i8,
    )
}

/// Extract the winner's value from a 4-lane vector using the selector.
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn silk_select_winner(num: __m128i, selector: __m128i) -> i32 {
    _mm_cvtsi128_si32(_mm_shuffle_epi8(num, selector))
}

/// Short-term prediction for 4 states simultaneously (SoA layout).
/// buf32[idx] is the most recent sample; we go backwards from there.
/// coef16[0] pairs with buf32[idx], coef16[1] with buf32[idx-1], etc.
#[target_feature(enable = "avx2")]
unsafe fn silk_noise_shape_quantizer_short_prediction_x4(
    buf32: &[__m128i],
    idx: usize,
    coef16: &[i16],
    order: i32,
) -> __m128i {
    debug_assert!(order == 10 || order == 16);

    // Avoids introducing a bias because silk_smlawb() always rounds to -inf
    let mut out = _mm256_set1_epi32(order >> 1);
    out = _mm256_add_epi32(
        out,
        _mm256_mul_epi32(
            _mm256_cvtepi32_epi64(buf32[idx]),
            _mm256_set1_epi32((coef16[0] as i32) << 16),
        ),
    );
    out = _mm256_add_epi32(
        out,
        _mm256_mul_epi32(
            _mm256_cvtepi32_epi64(buf32[idx - 1]),
            _mm256_set1_epi32((coef16[1] as i32) << 16),
        ),
    );
    out = _mm256_add_epi32(
        out,
        _mm256_mul_epi32(
            _mm256_cvtepi32_epi64(buf32[idx - 2]),
            _mm256_set1_epi32((coef16[2] as i32) << 16),
        ),
    );
    out = _mm256_add_epi32(
        out,
        _mm256_mul_epi32(
            _mm256_cvtepi32_epi64(buf32[idx - 3]),
            _mm256_set1_epi32((coef16[3] as i32) << 16),
        ),
    );
    out = _mm256_add_epi32(
        out,
        _mm256_mul_epi32(
            _mm256_cvtepi32_epi64(buf32[idx - 4]),
            _mm256_set1_epi32((coef16[4] as i32) << 16),
        ),
    );
    out = _mm256_add_epi32(
        out,
        _mm256_mul_epi32(
            _mm256_cvtepi32_epi64(buf32[idx - 5]),
            _mm256_set1_epi32((coef16[5] as i32) << 16),
        ),
    );
    out = _mm256_add_epi32(
        out,
        _mm256_mul_epi32(
            _mm256_cvtepi32_epi64(buf32[idx - 6]),
            _mm256_set1_epi32((coef16[6] as i32) << 16),
        ),
    );
    out = _mm256_add_epi32(
        out,
        _mm256_mul_epi32(
            _mm256_cvtepi32_epi64(buf32[idx - 7]),
            _mm256_set1_epi32((coef16[7] as i32) << 16),
        ),
    );
    out = _mm256_add_epi32(
        out,
        _mm256_mul_epi32(
            _mm256_cvtepi32_epi64(buf32[idx - 8]),
            _mm256_set1_epi32((coef16[8] as i32) << 16),
        ),
    );
    out = _mm256_add_epi32(
        out,
        _mm256_mul_epi32(
            _mm256_cvtepi32_epi64(buf32[idx - 9]),
            _mm256_set1_epi32((coef16[9] as i32) << 16),
        ),
    );

    if order == 16 {
        out = _mm256_add_epi32(
            out,
            _mm256_mul_epi32(
                _mm256_cvtepi32_epi64(buf32[idx - 10]),
                _mm256_set1_epi32((coef16[10] as i32) << 16),
            ),
        );
        out = _mm256_add_epi32(
            out,
            _mm256_mul_epi32(
                _mm256_cvtepi32_epi64(buf32[idx - 11]),
                _mm256_set1_epi32((coef16[11] as i32) << 16),
            ),
        );
        out = _mm256_add_epi32(
            out,
            _mm256_mul_epi32(
                _mm256_cvtepi32_epi64(buf32[idx - 12]),
                _mm256_set1_epi32((coef16[12] as i32) << 16),
            ),
        );
        out = _mm256_add_epi32(
            out,
            _mm256_mul_epi32(
                _mm256_cvtepi32_epi64(buf32[idx - 13]),
                _mm256_set1_epi32((coef16[13] as i32) << 16),
            ),
        );
        out = _mm256_add_epi32(
            out,
            _mm256_mul_epi32(
                _mm256_cvtepi32_epi64(buf32[idx - 14]),
                _mm256_set1_epi32((coef16[14] as i32) << 16),
            ),
        );
        out = _mm256_add_epi32(
            out,
            _mm256_mul_epi32(
                _mm256_cvtepi32_epi64(buf32[idx - 15]),
                _mm256_set1_epi32((coef16[15] as i32) << 16),
            ),
        );
    }
    silk_cvtepi64_epi32_high(out)
}

/// AVX2 LPC analysis filter.
/// Sets first `order` samples of `out` to zero, then computes the FIR filter.
#[target_feature(enable = "avx2")]
unsafe fn silk_lpc_analysis_filter_avx2(
    out: &mut [i16],
    input: &[i16],
    b: &[i16],
    len: i32,
    order: i32,
) {
    debug_assert!(order == 10 || order == 16);

    for _i in order..len {
        let in_ptr = input.as_ptr().add(_i as usize);

        let in_v = _mm256_cvtepi16_epi32(_mm_loadu_si128(in_ptr.sub(8) as *const __m128i));
        let b_v = _mm256_cvtepi16_epi32(_mm_loadu_si128(b.as_ptr() as *const __m128i));
        let mut sum = _mm256_mullo_epi32(in_v, silk_mm256_reverse_epi32(b_v));

        if order > 10 {
            let in_v = _mm256_cvtepi16_epi32(_mm_loadu_si128(in_ptr.sub(16) as *const __m128i));
            let b_v = _mm256_cvtepi16_epi32(_mm_loadu_si128(b.as_ptr().add(8) as *const __m128i));
            let b_v = silk_mm256_reverse_epi32(b_v);
            sum = _mm256_add_epi32(sum, _mm256_mullo_epi32(in_v, b_v));
        } else {
            // order == 10: only need 2 more coefficients
            let in_v = _mm256_cvtepi16_epi32(_mm_cvtsi32_si128(
                (in_ptr.sub(10) as *const i32).read_unaligned(),
            ));
            let b_v = _mm256_cvtepi16_epi32(_mm_cvtsi32_si128(
                (b.as_ptr().add(8) as *const i32).read_unaligned(),
            ));
            let b_v = _mm256_shuffle_epi32(b_v, 0x01);
            sum = _mm256_add_epi32(sum, _mm256_mullo_epi32(in_v, b_v));
        }

        let out32_q12 = silk_mm256_hsum_epi32(sum);

        // Subtract prediction: silk_LSHIFT(in[_i], 12) - out32_q12
        let out32_q12 = ((*in_ptr as i32 as u32) << 12) as i32 - out32_q12;

        // Scale to Q0 with rounding
        let out32 = silk_sar_round_32(out32_q12, 12);

        // Saturate output
        out[_i as usize] = silk_sat16(out32);
    }

    // Set first d output samples to zero
    for val in out.iter_mut().take(order as usize) {
        *val = 0;
    }
}

/// SoA sample struct — each field holds 4 decision states.
#[repr(C)]
#[derive(Copy, Clone)]
struct NsqDelDecSampleAvx2 {
    rand_state: __m128i,
    q_q10: __m128i,
    xq_q14: __m128i,
    pred_q15: __m128i,
    shape_q14: __m128i,
}

impl Default for NsqDelDecSampleAvx2 {
    fn default() -> Self {
        unsafe {
            Self {
                rand_state: _mm_setzero_si128(),
                q_q10: _mm_setzero_si128(),
                xq_q14: _mm_setzero_si128(),
                pred_q15: _mm_setzero_si128(),
                shape_q14: _mm_setzero_si128(),
            }
        }
    }
}

/// SoA delayed decision state — each scalar field is now a __m128i holding 4 states.
#[repr(C)]
struct NsqDelDecAvx2 {
    s_lpc_q14: [__m128i; MAX_SUB_FRAME_LENGTH + NSQ_LPC_BUF_LENGTH],
    lf_ar_q14: __m128i,
    seed: __m128i,
    seed_init: __m128i,
    rd_q10: __m128i,
    diff_q14: __m128i,
    s_ar2_q14: [__m128i; MAX_SHAPE_LPC_ORDER as usize],
    samples: [NsqDelDecSampleAvx2; DECISION_DELAY as usize],
}

impl NsqDelDecAvx2 {
    unsafe fn new_zeroed() -> Self {
        Self {
            s_lpc_q14: [_mm_setzero_si128(); MAX_SUB_FRAME_LENGTH + NSQ_LPC_BUF_LENGTH],
            lf_ar_q14: _mm_setzero_si128(),
            seed: _mm_setzero_si128(),
            seed_init: _mm_setzero_si128(),
            rd_q10: _mm_setzero_si128(),
            diff_q14: _mm_setzero_si128(),
            s_ar2_q14: [_mm_setzero_si128(); MAX_SHAPE_LPC_ORDER as usize],
            samples: [NsqDelDecSampleAvx2::default(); DECISION_DELAY as usize],
        }
    }
}

/// Inner quantizer for one subframe, AVX2 SoA version.
#[target_feature(enable = "avx2")]
#[allow(clippy::too_many_arguments)]
unsafe fn silk_noise_shape_quantizer_del_dec_avx2(
    nsq: &mut silk_nsq_state,
    ps_del_dec: &mut NsqDelDecAvx2,
    signal_type: i32,
    x_q10: &[i32],
    pulses: &mut [i8],
    pulses_off: usize,
    pxq_off: usize,
    s_ltp_q15: &mut [i32],
    delayed_gain_q10: &mut [i32; DECISION_DELAY as usize],
    a_q12: &[i16],
    b_q14: &[i16],
    ar_shp_q13: &[i16],
    lag: i32,
    harm_shape_firpacked_q14: i32,
    tilt_q14: i32,
    lf_shp_q14: i32,
    gain_q16: i32,
    lambda_q10: i32,
    offset_q10: i32,
    length: i32,
    subfr: i32,
    shaping_lpcorder: i32,
    predict_lpcorder: i32,
    warping_q16: i32,
    mask_del_dec: __m128i,
    smpl_buf_idx: &mut i32,
    decision_delay: i32,
) {
    let mut shp_lag_ptr_idx = (nsq.s_ltp_shp_buf_idx - lag + HARM_SHAPE_FIR_TAPS / 2) as usize;
    let mut pred_lag_ptr_idx = (nsq.s_ltp_buf_idx - lag + LTP_ORDER as i32 / 2) as usize;
    let gain_q10 = gain_q16 >> 6;

    for (_i, &x_q10_i) in x_q10.iter().take(length as usize).enumerate() {
        // Long-term prediction
        let ltp_pred_q14: i32;
        if signal_type == TYPE_VOICED {
            let mut ltp = 2i32;
            // silk_smulwb = (a * (i16)b) >> 16
            ltp = (ltp as i64 + ((s_ltp_q15[pred_lag_ptr_idx] as i64 * b_q14[0] as i64) >> 16))
                as i32;
            ltp = (ltp as i64 + ((s_ltp_q15[pred_lag_ptr_idx - 1] as i64 * b_q14[1] as i64) >> 16))
                as i32;
            ltp = (ltp as i64 + ((s_ltp_q15[pred_lag_ptr_idx - 2] as i64 * b_q14[2] as i64) >> 16))
                as i32;
            ltp = (ltp as i64 + ((s_ltp_q15[pred_lag_ptr_idx - 3] as i64 * b_q14[3] as i64) >> 16))
                as i32;
            ltp = (ltp as i64 + ((s_ltp_q15[pred_lag_ptr_idx - 4] as i64 * b_q14[4] as i64) >> 16))
                as i32;
            ltp_pred_q14 = ((ltp as u32) << 1) as i32; // Q13 -> Q14
            pred_lag_ptr_idx += 1;
        } else {
            ltp_pred_q14 = 0;
        }

        // Long-term shaping
        let n_ltp_q14: i32;
        if lag > 0 {
            let mut n = silk_add_sat32(
                nsq.s_ltp_shp_q14[shp_lag_ptr_idx],
                nsq.s_ltp_shp_q14[shp_lag_ptr_idx - 2],
            );
            // silk_smulwb
            n = ((n as i64 * harm_shape_firpacked_q14 as i16 as i64) >> 16) as i32;
            // silk_SMULWT
            n = (n as i64
                + ((nsq.s_ltp_shp_q14[shp_lag_ptr_idx - 1] as i64
                    * (harm_shape_firpacked_q14 as i64 >> 16))
                    >> 16)) as i32;
            n_ltp_q14 = ltp_pred_q14 - ((n as u32) << 2) as i32; // Q12 -> Q14
            shp_lag_ptr_idx += 1;
        } else {
            n_ltp_q14 = 0;
        }

        // Generate dither
        ps_del_dec.seed = silk_mm256_rand_epi32(ps_del_dec.seed);

        // Short-term prediction
        let lpc_pred_q14 = silk_noise_shape_quantizer_short_prediction_x4(
            &ps_del_dec.s_lpc_q14,
            NSQ_LPC_BUF_LENGTH - 1 + _i,
            a_q12,
            predict_lpcorder,
        );
        let lpc_pred_q14 = _mm_slli_epi32(lpc_pred_q14, 4); // Q10 -> Q14

        // Noise shape feedback
        debug_assert!(shaping_lpcorder > 0);
        debug_assert!(shaping_lpcorder & 1 == 0);
        // Output of lowpass section
        let mut tmp0 = _mm_add_epi32(
            ps_del_dec.diff_q14,
            silk_mm_smulwb_epi32(ps_del_dec.s_ar2_q14[0], warping_q16),
        );
        let mut n_ar_q14 = _mm_set1_epi32(shaping_lpcorder >> 1);
        for (j, &ar_shp_q13) in ar_shp_q13
            .iter()
            .take(shaping_lpcorder as usize - 1)
            .enumerate()
        {
            let tmp1 = ps_del_dec.s_ar2_q14[j];
            ps_del_dec.s_ar2_q14[j] = tmp0;
            n_ar_q14 = _mm_add_epi32(n_ar_q14, silk_mm_smulwb_epi32(tmp0, ar_shp_q13 as i32));
            tmp0 = _mm_add_epi32(
                tmp1,
                silk_mm_smulwb_epi32(
                    _mm_sub_epi32(ps_del_dec.s_ar2_q14[j + 1], tmp0),
                    warping_q16,
                ),
            );
        }
        ps_del_dec.s_ar2_q14[shaping_lpcorder as usize - 1] = tmp0;
        n_ar_q14 = _mm_add_epi32(
            n_ar_q14,
            silk_mm_smulwb_epi32(tmp0, ar_shp_q13[shaping_lpcorder as usize - 1] as i32),
        );

        n_ar_q14 = _mm_slli_epi32(n_ar_q14, 1); // Q11 -> Q12
        n_ar_q14 = _mm_add_epi32(
            n_ar_q14,
            silk_mm_smulwb_epi32(ps_del_dec.lf_ar_q14, tilt_q14),
        ); // Q12
        n_ar_q14 = _mm_slli_epi32(n_ar_q14, 2); // Q12 -> Q14

        let tmp0_lf = silk_mm_smulwb_epi32(
            ps_del_dec.samples[*smpl_buf_idx as usize].shape_q14,
            lf_shp_q14,
        );
        let tmp1_lf = silk_mm_smulwb_epi32(ps_del_dec.lf_ar_q14, lf_shp_q14 >> 16);
        let n_lf_q14 = _mm_add_epi32(tmp0_lf, tmp1_lf); // Q12
        let n_lf_q14 = _mm_slli_epi32(n_lf_q14, 2); // Q12 -> Q14

        // r = x[_i] - LTP_pred - LPC_pred + n_AR + n_Tilt + n_LF + n_LTP
        let tmp0 = silk_mm_add_sat_epi32(n_ar_q14, n_lf_q14); // Q14
        let tmp1 = _mm_add_epi32(_mm_set1_epi32(n_ltp_q14), lpc_pred_q14); // Q14
        let tmp0 = silk_mm_sub_sat_epi32(tmp1, tmp0); // Q14
        let tmp0 = silk_mm_srai_round_epi32(tmp0, 4); // Q10

        let r_q10 = _mm_sub_epi32(_mm_set1_epi32(x_q10_i), tmp0);

        // Flip sign depending on dither
        let r_q10 = silk_mm_sign_epi32(r_q10, ps_del_dec.seed);
        let r_q10 = silk_mm_limit_epi32(r_q10, -(31 << 10), 30 << 10);

        // Find two quantization level candidates and measure their rate-distortion
        let mut q1_q10 = _mm_sub_epi32(r_q10, _mm_set1_epi32(offset_q10));
        let mut q1_q0 = _mm_srai_epi32(q1_q10, 10);
        if lambda_q10 > 2048 {
            // For aggressive RDO
            let tmp0 = _mm_sub_epi32(_mm_abs_epi32(q1_q10), _mm_set1_epi32(lambda_q10 / 2 - 512));
            q1_q0 = _mm_srai_epi32(q1_q10, 31);
            let tmp1 = _mm_cmpgt_epi32(tmp0, _mm_setzero_si128());
            let tmp0 = _mm_srai_epi32(silk_mm_sign_epi32(tmp0, q1_q10), 10);
            q1_q0 = _mm_blendv_epi8(q1_q0, tmp0, tmp1);
        }

        let tmp0 = _mm_sign_epi32(
            _mm_set1_epi32(crate::silk::define::QUANT_LEVEL_ADJUST_Q10),
            q1_q0,
        );
        q1_q10 = _mm_sub_epi32(_mm_slli_epi32(q1_q0, 10), tmp0);
        q1_q10 = _mm_add_epi32(q1_q10, _mm_set1_epi32(offset_q10));

        // check if q1_q0 is 0 or -1
        let tmp0 = _mm_add_epi32(_mm_srli_epi32(q1_q0, 31), q1_q0);
        let tmp1 = _mm_cmpeq_epi32(tmp0, _mm_setzero_si128());
        let tmp0 = _mm_blendv_epi8(
            _mm_set1_epi32(1024),
            _mm_set1_epi32(1024 - crate::silk::define::QUANT_LEVEL_ADJUST_Q10),
            tmp1,
        );
        let q2_q10 = _mm_add_epi32(q1_q10, tmp0);
        let q_q10 = _mm256_set_m128i(q2_q10, q1_q10);

        let rr_q10 = _mm256_sub_epi32(_mm256_broadcastsi128_si256(r_q10), q_q10);
        let mut rd_q10 = _mm256_abs_epi32(q_q10);
        let rr_q10 = silk_mm256_smulbb_epi32(rr_q10, rr_q10);
        rd_q10 = silk_mm256_smulbb_epi32(rd_q10, _mm256_set1_epi32(lambda_q10));
        let mut rd_q10 = _mm256_add_epi32(rd_q10, rr_q10);
        rd_q10 = _mm256_srai_epi32(rd_q10, 10);

        let mask = _mm256_broadcastsi128_si256(_mm_cmplt_epi32(
            _mm256_extracti128_si256(rd_q10, 0),
            _mm256_extracti128_si256(rd_q10, 1),
        ));
        let mut ss_rd_q10 = _mm256_add_epi32(
            _mm256_broadcastsi128_si256(ps_del_dec.rd_q10),
            _mm256_blendv_epi8(_mm256_permute2x128_si256(rd_q10, rd_q10, 0x1), rd_q10, mask),
        );
        let mut ss_q_q10 =
            _mm256_blendv_epi8(_mm256_permute2x128_si256(q_q10, q_q10, 0x1), q_q10, mask);

        // Quantized excitation
        let mut exc_q14 = silk_mm256_sign_epi32(
            _mm256_slli_epi32(ss_q_q10, 4),
            _mm256_broadcastsi128_si256(ps_del_dec.seed),
        );

        // Add predictions
        exc_q14 = _mm256_add_epi32(exc_q14, _mm256_set1_epi32(ltp_pred_q14));
        let mut ss_lpc_exc_q14 = _mm256_slli_epi32(exc_q14, 1);
        let mut ss_xq_q14 = _mm256_add_epi32(exc_q14, _mm256_broadcastsi128_si256(lpc_pred_q14));

        // Update states
        let mut ss_diff_q14 =
            _mm256_sub_epi32(ss_xq_q14, _mm256_set1_epi32(((x_q10_i as u32) << 4) as i32));
        let mut ss_lf_ar_q14 = _mm256_sub_epi32(ss_diff_q14, _mm256_broadcastsi128_si256(n_ar_q14));
        let mut ss_s_ltp_shp_q14 =
            silk_mm256_sub_sat_epi32(ss_lf_ar_q14, _mm256_broadcastsi128_si256(n_lf_q14));

        // Update buffer indices
        *smpl_buf_idx = (*smpl_buf_idx + DECISION_DELAY - 1) % DECISION_DELAY;
        let last_smple_idx = (*smpl_buf_idx + decision_delay) % DECISION_DELAY;

        // Copy last sample fields to avoid borrow conflicts when mutating ps_del_dec.samples below
        let last_rand_state = ps_del_dec.samples[last_smple_idx as usize].rand_state;
        let last_q_q10 = ps_del_dec.samples[last_smple_idx as usize].q_q10;
        let last_xq_q14 = ps_del_dec.samples[last_smple_idx as usize].xq_q14;
        let last_shape_q14 = ps_del_dec.samples[last_smple_idx as usize].shape_q14;
        let last_pred_q15 = ps_del_dec.samples[last_smple_idx as usize].pred_q15;

        // Find winner
        let rdmin_q10 = silk_mm_mask_hmin_epi32(_mm256_castsi256_si128(ss_rd_q10), mask_del_dec);
        let winner_selector = silk_index_to_selector(silk_index_of_first_equal_epi32(
            rdmin_q10,
            _mm256_castsi256_si128(ss_rd_q10),
        ));

        // Increase RD values of expired states
        let winner_rand_state = _mm_shuffle_epi8(last_rand_state, winner_selector);

        ss_rd_q10 = _mm256_blendv_epi8(
            _mm256_add_epi32(ss_rd_q10, _mm256_set1_epi32(i32::MAX >> 4)),
            ss_rd_q10,
            _mm256_broadcastsi128_si256(_mm_cmpeq_epi32(last_rand_state, winner_rand_state)),
        );

        // Find worst in first set
        let rdmax_q10 =
            silk_mm_mask_hmax_epi32(_mm256_extracti128_si256(ss_rd_q10, 0), mask_del_dec);
        // Find best in second set
        let rdmin_q10 =
            silk_mm_mask_hmin_epi32(_mm256_extracti128_si256(ss_rd_q10, 1), mask_del_dec);

        // Replace a state if best from second set outperforms worst in first set
        let cmp_tmp = _mm_cmplt_epi32(rdmin_q10, rdmax_q10);
        if _mm_test_all_zeros(cmp_tmp, cmp_tmp) == 0 {
            let rdmax_ind =
                silk_index_of_first_equal_epi32(rdmax_q10, _mm256_extracti128_si256(ss_rd_q10, 0));
            let rdmin_ind =
                silk_index_of_first_equal_epi32(rdmin_q10, _mm256_extracti128_si256(ss_rd_q10, 1));
            let tmp1 = _mm_cvtepi8_epi32(_mm_cvtsi32_si128(
                (0xFFu32 << ((rdmax_ind as u32) << 3)) as i32,
            ));
            let tmp0 = _mm_blendv_epi8(
                _mm_set_epi8(
                    0xF, 0xE, 0xD, 0xC, 0xB, 0xA, 0x9, 0x8, 0x7, 0x6, 0x5, 0x4, 0x3, 0x2, 0x1, 0x0,
                ),
                silk_index_to_selector(rdmin_ind),
                tmp1,
            );
            for t in _i..MAX_SUB_FRAME_LENGTH + NSQ_LPC_BUF_LENGTH {
                ps_del_dec.s_lpc_q14[t] = _mm_shuffle_epi8(ps_del_dec.s_lpc_q14[t], tmp0);
            }
            ps_del_dec.seed = _mm_shuffle_epi8(ps_del_dec.seed, tmp0);
            ps_del_dec.seed_init = _mm_shuffle_epi8(ps_del_dec.seed_init, tmp0);
            for t in 0..MAX_SHAPE_LPC_ORDER as usize {
                ps_del_dec.s_ar2_q14[t] = _mm_shuffle_epi8(ps_del_dec.s_ar2_q14[t], tmp0);
            }
            for t in 0..DECISION_DELAY as usize {
                ps_del_dec.samples[t].rand_state =
                    _mm_shuffle_epi8(ps_del_dec.samples[t].rand_state, tmp0);
                ps_del_dec.samples[t].q_q10 = _mm_shuffle_epi8(ps_del_dec.samples[t].q_q10, tmp0);
                ps_del_dec.samples[t].xq_q14 = _mm_shuffle_epi8(ps_del_dec.samples[t].xq_q14, tmp0);
                ps_del_dec.samples[t].pred_q15 =
                    _mm_shuffle_epi8(ps_del_dec.samples[t].pred_q15, tmp0);
                ps_del_dec.samples[t].shape_q14 =
                    _mm_shuffle_epi8(ps_del_dec.samples[t].shape_q14, tmp0);
            }
            let perm_mask = _mm256_castsi128_si256(_mm_blendv_epi8(
                _mm_set_epi32(0x3, 0x2, 0x1, 0x0),
                _mm_set1_epi32(rdmin_ind + 4),
                tmp1,
            ));
            ss_q_q10 = _mm256_permutevar8x32_epi32(ss_q_q10, perm_mask);
            ss_rd_q10 = _mm256_permutevar8x32_epi32(ss_rd_q10, perm_mask);
            ss_xq_q14 = _mm256_permutevar8x32_epi32(ss_xq_q14, perm_mask);
            ss_lf_ar_q14 = _mm256_permutevar8x32_epi32(ss_lf_ar_q14, perm_mask);
            ss_diff_q14 = _mm256_permutevar8x32_epi32(ss_diff_q14, perm_mask);
            ss_s_ltp_shp_q14 = _mm256_permutevar8x32_epi32(ss_s_ltp_shp_q14, perm_mask);
            ss_lpc_exc_q14 = _mm256_permutevar8x32_epi32(ss_lpc_exc_q14, perm_mask);
        }

        // Write samples from winner to output and long-term filter states
        if subfr > 0 || _i as i32 >= decision_delay {
            pulses[pulses_off + _i - decision_delay as usize] =
                silk_sar_round_32(silk_select_winner(last_q_q10, winner_selector), 10) as i8;
            nsq.xq[pxq_off + _i - decision_delay as usize] = silk_sat16(silk_sar_round_smulww(
                silk_select_winner(last_xq_q14, winner_selector),
                delayed_gain_q10[last_smple_idx as usize],
                8,
            ) as i32);
            nsq.s_ltp_shp_q14[(nsq.s_ltp_shp_buf_idx - decision_delay) as usize] =
                silk_select_winner(last_shape_q14, winner_selector);
            s_ltp_q15[(nsq.s_ltp_buf_idx - decision_delay) as usize] =
                silk_select_winner(last_pred_q15, winner_selector);
        }
        nsq.s_ltp_shp_buf_idx += 1;
        nsq.s_ltp_buf_idx += 1;

        // Update states
        let ps_sample = &mut ps_del_dec.samples[*smpl_buf_idx as usize];
        ps_del_dec.seed = _mm_add_epi32(
            ps_del_dec.seed,
            silk_mm_srai_round_epi32(_mm256_castsi256_si128(ss_q_q10), 10),
        );
        ps_del_dec.lf_ar_q14 = _mm256_castsi256_si128(ss_lf_ar_q14);
        ps_del_dec.diff_q14 = _mm256_castsi256_si128(ss_diff_q14);
        ps_del_dec.s_lpc_q14[_i + NSQ_LPC_BUF_LENGTH] = _mm256_castsi256_si128(ss_xq_q14);
        ps_del_dec.rd_q10 = _mm256_castsi256_si128(ss_rd_q10);
        ps_sample.xq_q14 = _mm256_castsi256_si128(ss_xq_q14);
        ps_sample.q_q10 = _mm256_castsi256_si128(ss_q_q10);
        ps_sample.pred_q15 = _mm256_castsi256_si128(ss_lpc_exc_q14);
        ps_sample.shape_q14 = _mm256_castsi256_si128(ss_s_ltp_shp_q14);
        ps_sample.rand_state = ps_del_dec.seed;
        delayed_gain_q10[*smpl_buf_idx as usize] = gain_q10;
    }

    // Update LPC states
    for ii in 0..NSQ_LPC_BUF_LENGTH {
        ps_del_dec.s_lpc_q14[ii] = ps_del_dec.s_lpc_q14[length as usize + ii];
    }
}

/// Scale states helper, AVX2 SoA version.
#[target_feature(enable = "avx2")]
#[allow(clippy::too_many_arguments)]
unsafe fn silk_nsq_del_dec_scale_states_avx2(
    ps_enc_c: &NsqConfig,
    nsq: &mut silk_nsq_state,
    ps_del_dec: &mut NsqDelDecAvx2,
    x16: &[i16],
    x_sc_q10: &mut [i32],
    s_ltp: &[i16],
    s_ltp_q15: &mut [i32],
    subfr: i32,
    ltp_scale_q14: i32,
    gains_q16: &[i32],
    pitch_l: &[i32],
    signal_type: i32,
    decision_delay: i32,
) {
    let lag = pitch_l[subfr as usize];
    let mut inv_gain_q31 = silk_inverse32_varq(gains_q16[subfr as usize].max(1), 47);

    // Scale input
    let inv_gain_q26 = silk_sar_round_32(inv_gain_q31, 5);
    let mut ii = 0usize;
    while ii + 3 < ps_enc_c.subfr_length {
        let x = _mm256_cvtepi16_epi64(_mm_loadl_epi64(x16.as_ptr().add(ii) as *const __m128i));
        let x = _mm256_slli_epi64(_mm256_mul_epi32(x, _mm256_set1_epi32(inv_gain_q26)), 16);
        _mm_storeu_si128(
            x_sc_q10.as_mut_ptr().add(ii) as *mut __m128i,
            silk_cvtepi64_epi32_high(x),
        );
        ii += 4;
    }
    while ii < ps_enc_c.subfr_length {
        x_sc_q10[ii] = ((x16[ii] as i64 * inv_gain_q26 as i64) >> 16) as i32;
        ii += 1;
    }

    // After rewhitening the LTP state is un-scaled, so scale with inv_gain_Q16
    if nsq.rewhite_flag != 0 {
        if subfr == 0 {
            // Do LTP downscaling
            // silk_LSHIFT(silk_smulwb(inv_gain_q31, ltp_scale_q14), 2)
            inv_gain_q31 = ((((inv_gain_q31 as i64 * ltp_scale_q14 as i16 as i64) >> 16) as i32
                as u32)
                << 2) as i32;
        }
        let start = (nsq.s_ltp_buf_idx - lag - LTP_ORDER as i32 / 2) as usize;
        let end = nsq.s_ltp_buf_idx as usize;
        for jj in start..end {
            s_ltp_q15[jj] = ((inv_gain_q31 as i64 * s_ltp[jj] as i64) >> 16) as i32;
        }
    }

    // Adjust for changing gain
    if gains_q16[subfr as usize] != nsq.prev_gain_q16 {
        let gain_adj_q16 = silk_div32_varq(nsq.prev_gain_q16, gains_q16[subfr as usize], 16);

        // Scale long-term shaping state
        let shp_start = (nsq.s_ltp_shp_buf_idx - ps_enc_c.ltp_mem_length as i32) as usize;
        let shp_end = nsq.s_ltp_shp_buf_idx as usize;
        let mut jj = shp_start;
        while jj + 3 < shp_end {
            let p = nsq.s_ltp_shp_q14.as_mut_ptr().add(jj);
            _mm_storeu_si128(
                p as *mut __m128i,
                silk_mm_smulww_epi32(_mm_loadu_si128(p as *const __m128i), gain_adj_q16),
            );
            jj += 4;
        }
        while jj < shp_end {
            nsq.s_ltp_shp_q14[jj] =
                ((nsq.s_ltp_shp_q14[jj] as i64 * gain_adj_q16 as i64) >> 16) as i32;
            jj += 1;
        }

        // Scale long-term prediction state
        if signal_type == TYPE_VOICED && nsq.rewhite_flag == 0 {
            let start = (nsq.s_ltp_buf_idx - lag - LTP_ORDER as i32 / 2) as usize;
            let end = (nsq.s_ltp_buf_idx - decision_delay) as usize;
            for val in s_ltp_q15[start..end].iter_mut() {
                *val = ((*val as i64 * gain_adj_q16 as i64) >> 16) as i32;
            }
        }

        // Scale scalar states (SoA vectors)
        ps_del_dec.lf_ar_q14 = silk_mm_smulww_epi32(ps_del_dec.lf_ar_q14, gain_adj_q16);
        ps_del_dec.diff_q14 = silk_mm_smulww_epi32(ps_del_dec.diff_q14, gain_adj_q16);

        // Scale short-term prediction and shaping states
        for jj in 0..NSQ_LPC_BUF_LENGTH {
            ps_del_dec.s_lpc_q14[jj] = silk_mm_smulww_epi32(ps_del_dec.s_lpc_q14[jj], gain_adj_q16);
        }
        for jj in 0..DECISION_DELAY as usize {
            ps_del_dec.samples[jj].pred_q15 =
                silk_mm_smulww_epi32(ps_del_dec.samples[jj].pred_q15, gain_adj_q16);
            ps_del_dec.samples[jj].shape_q14 =
                silk_mm_smulww_epi32(ps_del_dec.samples[jj].shape_q14, gain_adj_q16);
        }
        for jj in 0..MAX_SHAPE_LPC_ORDER as usize {
            ps_del_dec.s_ar2_q14[jj] = silk_mm_smulww_epi32(ps_del_dec.s_ar2_q14[jj], gain_adj_q16);
        }

        // Save inverse gain
        nsq.prev_gain_q16 = gains_q16[subfr as usize];
    }
}

/// Complete AVX2 nsq del_dec outer function.
/// Replaces the entire `silk_nsq_del_dec_c` when n_states_delayed_decision is 3 or 4 and AVX2 available.
///
/// # Safety
/// Requires AVX2 support (checked by caller via cpufeatures).
#[target_feature(enable = "avx2")]
#[allow(clippy::too_many_arguments)]
pub unsafe fn silk_nsq_del_dec_avx2(
    ps_enc_c: &NsqConfig,
    nsq: &mut silk_nsq_state,
    ps_indices: &mut SideInfoIndices,
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
    use crate::silk::define::MAX_LPC_ORDER;
    use crate::silk::tables_other::SILK_QUANTIZATION_OFFSETS_Q10;

    let ltp_mem_len = ps_enc_c.ltp_mem_length;
    let frame_len = ps_enc_c.frame_length;
    let subfr_len = ps_enc_c.subfr_length;

    // Build mask_del_dec: lanes beyond n_states_delayed_decision get masked out
    let mask_del_dec = _mm_cvtepi8_epi32(_mm_cvtsi32_si128(
        (0xFFFFFF00u32 << ((ps_enc_c.n_states_delayed_decision as u32 - 1) << 3)) as i32,
    ));

    // Set unvoiced lag to the previous one
    let mut lag = nsq.lag_prev;

    debug_assert!(nsq.prev_gain_q16 != 0);

    let mut ps_del_dec = NsqDelDecAvx2::new_zeroed();
    ps_del_dec.seed = _mm_and_si128(
        _mm_add_epi32(
            _mm_set_epi32(3, 2, 1, 0),
            _mm_set1_epi32(ps_indices.seed as i32),
        ),
        _mm_set1_epi32(3),
    );
    ps_del_dec.seed_init = ps_del_dec.seed;
    ps_del_dec.rd_q10 = _mm_setzero_si128();
    ps_del_dec.lf_ar_q14 = _mm_set1_epi32(nsq.s_lf_ar_shp_q14);
    ps_del_dec.diff_q14 = _mm_set1_epi32(nsq.s_diff_shp_q14);
    ps_del_dec.samples[0].shape_q14 = _mm_set1_epi32(nsq.s_ltp_shp_q14[ltp_mem_len - 1]);
    for ii in 0..NSQ_LPC_BUF_LENGTH {
        ps_del_dec.s_lpc_q14[ii] = _mm_set1_epi32(nsq.s_lpc_q14[ii]);
    }
    for ii in 0..MAX_SHAPE_LPC_ORDER as usize {
        ps_del_dec.s_ar2_q14[ii] = _mm_set1_epi32(nsq.s_ar2_q14[ii]);
    }

    let offset_q10 = SILK_QUANTIZATION_OFFSETS_Q10[(ps_indices.signal_type as i32 >> 1) as usize]
        [ps_indices.quant_offset_type as usize] as i32;
    let mut smpl_buf_idx: i32 = 0;

    let mut decision_delay =
        crate::silk::sigproc_fix::silk_min_int(DECISION_DELAY, subfr_len as i32);

    // For voiced frames limit the decision delay
    if ps_indices.signal_type as i32 == TYPE_VOICED {
        for &pl in pitch_l.iter().take(ps_enc_c.nb_subfr) {
            decision_delay = crate::silk::sigproc_fix::silk_min_int(
                decision_delay,
                pl - LTP_ORDER as i32 / 2 - 1,
            );
        }
    } else if lag > 0 {
        decision_delay =
            crate::silk::sigproc_fix::silk_min_int(decision_delay, lag - LTP_ORDER as i32 / 2 - 1);
    }

    let lsf_interpolation_flag: i32 = if ps_indices.nlsfinterp_coef_q2 as i32 == 4 {
        0
    } else {
        1
    };

    let mut s_ltp_q15: Vec<i32> = vec![0; ltp_mem_len + frame_len];
    let mut s_ltp: Vec<i16> = vec![0; ltp_mem_len + frame_len];
    let mut x_sc_q10 = [0i32; MAX_SUB_FRAME_LENGTH];
    let mut delayed_gain_q10 = [0i32; DECISION_DELAY as usize];

    let mut pxq_off: usize = ltp_mem_len;
    nsq.s_ltp_shp_buf_idx = ltp_mem_len as i32;
    nsq.s_ltp_buf_idx = ltp_mem_len as i32;
    let mut subfr: i32 = 0;
    let mut x16_off: usize = 0;
    let mut pulses_off: usize = 0;

    for k in 0..ps_enc_c.nb_subfr {
        let a_q12_off = ((k >> 1) | ((1 - lsf_interpolation_flag) as usize)) * MAX_LPC_ORDER;
        let a_q12 = &pred_coef_q12[a_q12_off..a_q12_off + ps_enc_c.predict_lpcorder as usize];
        let b_q14_off = k * LTP_ORDER;
        let b_q14 = &ltpcoef_q14[b_q14_off..b_q14_off + LTP_ORDER];
        let ar_shp_off = k * MAX_SHAPE_LPC_ORDER as usize;
        let ar_shp_q13 = &ar_q13[ar_shp_off..ar_shp_off + ps_enc_c.shaping_lpcorder as usize];

        // Noise shape parameters
        debug_assert!(harm_shape_gain_q14[k] >= 0);
        let mut harm_shape_firpacked_q14: i32 = harm_shape_gain_q14[k] >> 2;
        harm_shape_firpacked_q14 |= (((harm_shape_gain_q14[k] >> 1) as u32) << 16) as i32;

        nsq.rewhite_flag = 0;
        if ps_indices.signal_type as i32 == TYPE_VOICED {
            lag = pitch_l[k];

            // Re-whitening
            if (k as i32) & (3 ^ (lsf_interpolation_flag << 1)) == 0 {
                if k == 2 {
                    // RESET DELAYED DECISIONS
                    let rdmin_q10 = silk_mm_mask_hmin_epi32(ps_del_dec.rd_q10, mask_del_dec);
                    let winner_ind = silk_index_of_first_equal_epi32(rdmin_q10, ps_del_dec.rd_q10);
                    let winner_selector = silk_index_to_selector(winner_ind);
                    ps_del_dec.rd_q10 = _mm_add_epi32(
                        ps_del_dec.rd_q10,
                        _mm_blendv_epi8(
                            _mm_set1_epi32(i32::MAX >> 4),
                            _mm_setzero_si128(),
                            _mm_cvtepi8_epi32(_mm_cvtsi32_si128(
                                (0xFFu32 << ((winner_ind as u32) << 3)) as i32,
                            )),
                        ),
                    );

                    // Copy final part of signals from winner state to output
                    let mut last_smple_idx = smpl_buf_idx + decision_delay;
                    for ii in 0..decision_delay {
                        last_smple_idx = (last_smple_idx + DECISION_DELAY - 1) % DECISION_DELAY;
                        let ps_sample = &ps_del_dec.samples[last_smple_idx as usize];
                        pulses[(pulses_off as isize + (ii - decision_delay) as isize) as usize] =
                            silk_sar_round_32(
                                silk_select_winner(ps_sample.q_q10, winner_selector),
                                10,
                            ) as i8;
                        nsq.xq[(pxq_off as isize + (ii - decision_delay) as isize) as usize] =
                            silk_sat16(silk_sar_round_smulww(
                                silk_select_winner(ps_sample.xq_q14, winner_selector),
                                gains_q16[1],
                                14,
                            ) as i32);
                        nsq.s_ltp_shp_q14[(nsq.s_ltp_shp_buf_idx - decision_delay + ii) as usize] =
                            silk_select_winner(ps_sample.shape_q14, winner_selector);
                    }

                    subfr = 0;
                }

                // Rewhiten with new a coefs
                let start_idx =
                    ltp_mem_len as i32 - lag - ps_enc_c.predict_lpcorder - LTP_ORDER as i32 / 2;
                debug_assert!(start_idx > 0);

                silk_lpc_analysis_filter_avx2(
                    &mut s_ltp[start_idx as usize..],
                    &nsq.xq[(start_idx + k as i32 * subfr_len as i32) as usize..],
                    a_q12,
                    ltp_mem_len as i32 - start_idx,
                    ps_enc_c.predict_lpcorder,
                );

                nsq.s_ltp_buf_idx = ltp_mem_len as i32;
                nsq.rewhite_flag = 1;
            }
        }

        silk_nsq_del_dec_scale_states_avx2(
            ps_enc_c,
            nsq,
            &mut ps_del_dec,
            &x16[x16_off..x16_off + subfr_len],
            &mut x_sc_q10[..subfr_len],
            &s_ltp,
            &mut s_ltp_q15,
            k as i32,
            ltp_scale_q14,
            gains_q16,
            pitch_l,
            ps_indices.signal_type as i32,
            decision_delay,
        );

        let fresh_subfr = subfr;
        subfr += 1;

        silk_noise_shape_quantizer_del_dec_avx2(
            nsq,
            &mut ps_del_dec,
            ps_indices.signal_type as i32,
            &x_sc_q10[..subfr_len],
            pulses,
            pulses_off,
            pxq_off,
            &mut s_ltp_q15,
            &mut delayed_gain_q10,
            a_q12,
            b_q14,
            ar_shp_q13,
            lag,
            harm_shape_firpacked_q14,
            tilt_q14[k],
            lf_shp_q14[k],
            gains_q16[k],
            lambda_q10,
            offset_q10,
            subfr_len as i32,
            fresh_subfr,
            ps_enc_c.shaping_lpcorder,
            ps_enc_c.predict_lpcorder,
            ps_enc_c.warping_q16,
            mask_del_dec,
            &mut smpl_buf_idx,
            decision_delay,
        );

        x16_off += subfr_len;
        pulses_off += subfr_len;
        pxq_off += subfr_len;
    }

    // Find winner
    let rdmin_q10 = silk_mm_mask_hmin_epi32(ps_del_dec.rd_q10, mask_del_dec);
    let winner_selector = silk_index_to_selector(silk_index_of_first_equal_epi32(
        rdmin_q10,
        ps_del_dec.rd_q10,
    ));

    // Copy final part of signals from winner state to output
    ps_indices.seed = silk_select_winner(ps_del_dec.seed_init, winner_selector) as i8;
    let mut last_smple_idx = smpl_buf_idx + decision_delay;
    let gain_q10 = gains_q16[ps_enc_c.nb_subfr - 1] >> 6;
    for ii in 0..decision_delay {
        last_smple_idx = (last_smple_idx + DECISION_DELAY - 1) % DECISION_DELAY;
        let ps_sample = &ps_del_dec.samples[last_smple_idx as usize];

        pulses[(pulses_off as isize + (ii - decision_delay) as isize) as usize] =
            silk_sar_round_32(silk_select_winner(ps_sample.q_q10, winner_selector), 10) as i8;
        nsq.xq[(pxq_off as isize + (ii - decision_delay) as isize) as usize] =
            silk_sat16(silk_sar_round_smulww(
                silk_select_winner(ps_sample.xq_q14, winner_selector),
                gain_q10,
                8,
            ) as i32);
        nsq.s_ltp_shp_q14[(nsq.s_ltp_shp_buf_idx - decision_delay + ii) as usize] =
            silk_select_winner(ps_sample.shape_q14, winner_selector);
    }
    for ii in 0..NSQ_LPC_BUF_LENGTH {
        nsq.s_lpc_q14[ii] = silk_select_winner(ps_del_dec.s_lpc_q14[ii], winner_selector);
    }
    for ii in 0..MAX_SHAPE_LPC_ORDER as usize {
        nsq.s_ar2_q14[ii] = silk_select_winner(ps_del_dec.s_ar2_q14[ii], winner_selector);
    }

    // Update states
    nsq.s_lf_ar_shp_q14 = silk_select_winner(ps_del_dec.lf_ar_q14, winner_selector);
    nsq.s_diff_shp_q14 = silk_select_winner(ps_del_dec.diff_q14, winner_selector);
    nsq.lag_prev = pitch_l[ps_enc_c.nb_subfr - 1];

    // Save quantized speech signal
    nsq.xq.copy_within(frame_len..frame_len + ltp_mem_len, 0);
    nsq.s_ltp_shp_q14
        .copy_within(frame_len..frame_len + ltp_mem_len, 0);
}
