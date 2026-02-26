//! x86/x86_64 SIMD implementations for SILK functions.
//!
//! SSE4.1 and AVX2 intrinsics for noise shaping, inner products, etc.
//! All functions require `#[target_feature]` and are called only after cpufeatures detection.

#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

use crate::silk::define::{
    DECISION_DELAY, HARM_SHAPE_FIR_TAPS, LTP_ORDER, MAX_SHAPE_LPC_ORDER, NSQ_LPC_BUF_LENGTH,
    TYPE_VOICED,
};
use crate::silk::structs::{silk_encoder_state, silk_nsq_state, NsqConfig, SideInfoIndices};
use crate::silk::Inlines::{silk_DIV32_varQ, silk_INVERSE32_varQ};
use crate::silk::NSQ_del_dec::{
    copy_del_dec_state_partial, NSQ_del_dec_struct, NSQ_sample_pair, NSQ_sample_struct,
};
use crate::silk::SigProc_FIX::silk_RAND;

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
    // For each pair: (buf32[i] as i64 * coef16[i] as i64) >> 16
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
/// Computes sum of (X[i] >> 3)^2 for i in 0..len.
/// Port of `silk/x86/VAD_sse4_1.c` inner loop (uses only SSE2 instructions).
///
/// # Safety
/// Requires SSE2 support (checked by caller via cpufeatures).
#[target_feature(enable = "sse2")]
pub unsafe fn silk_vad_energy_sse2(x: &[i16]) -> i32 {
    let n = x.len();
    let mut acc = _mm_setzero_si128();
    let mut i = 0usize;

    // Process 8 samples at a time
    while i + 7 < n {
        let xmm = _mm_loadu_si128(x.as_ptr().add(i) as *const __m128i);
        // Arithmetic right shift by 3 (stays in i16)
        let shifted = _mm_srai_epi16(xmm, 3);
        // Multiply pairs of i16 and sum adjacent pairs → 4 x i32
        let squared = _mm_madd_epi16(shifted, shifted);
        acc = _mm_add_epi32(acc, squared);
        i += 8;
    }

    // Horizontal sum of 4 x i32
    let hi64 = _mm_unpackhi_epi64(acc, acc);
    acc = _mm_add_epi32(acc, hi64);
    let hi32 = _mm_shufflelo_epi16(acc, 0x0E);
    acc = _mm_add_epi32(acc, hi32);
    let mut result = _mm_cvtsi128_si32(acc);

    // Handle remaining elements
    while i < n {
        let x_tmp = (*x.get_unchecked(i) as i32) >> 3;
        result += (x_tmp as i16 as i32) * (x_tmp as i16 as i32);
        i += 1;
    }

    result
}

/// SSE4.1 full-function VAD entry.
/// Mirrors upstream RTCD surface `silk/x86/main_sse.h:silk_VAD_GetSA_Q8`.
///
/// # Safety
/// Requires SSE4.1 support (checked by caller via cpufeatures).
#[target_feature(enable = "sse4.1")]
pub unsafe fn silk_VAD_GetSA_Q8_sse4_1(psEncC: &mut silk_encoder_state, pIn: &[i16]) -> i32 {
    crate::silk::VAD::silk_VAD_GetSA_Q8_c(psEncC, pIn)
}

/// SSE4.1 full-function NSQ entry.
/// Mirrors upstream RTCD surface `silk/x86/main_sse.h:silk_NSQ`.
///
/// # Safety
/// Requires SSE4.1 support (checked by caller via cpufeatures).
#[target_feature(enable = "sse4.1")]
#[allow(clippy::too_many_arguments)]
pub unsafe fn silk_NSQ_sse4_1(
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
    crate::silk::NSQ::silk_NSQ_c(
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

/// SSE4.1 full-function NSQ-del-dec entry.
/// Mirrors upstream RTCD surface `silk/x86/main_sse.h:silk_NSQ_del_dec` (SSE tier).
///
/// # Safety
/// Requires SSE4.1 support (checked by caller via cpufeatures).
#[target_feature(enable = "sse4.1")]
#[allow(clippy::too_many_arguments)]
pub unsafe fn silk_NSQ_del_dec_sse4_1(
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
    crate::silk::NSQ_del_dec::silk_NSQ_del_dec_c(
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

/// AVX2+FMA implementation of `silk_inner_product_FLP`.
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
    let mut i = 0usize;

    // Main loop: 8 f32s per iteration (two groups of 4 → 4 f64s each)
    while i + 7 < n {
        let x1f = _mm_loadu_ps(data1.as_ptr().add(i));
        let x2f = _mm_loadu_ps(data2.as_ptr().add(i));
        let x1d = _mm256_cvtps_pd(x1f);
        let x2d = _mm256_cvtps_pd(x2f);
        accum1 = _mm256_fmadd_pd(x1d, x2d, accum1);

        let x1f = _mm_loadu_ps(data1.as_ptr().add(i + 4));
        let x2f = _mm_loadu_ps(data2.as_ptr().add(i + 4));
        let x1d = _mm256_cvtps_pd(x1f);
        let x2d = _mm256_cvtps_pd(x2f);
        accum2 = _mm256_fmadd_pd(x1d, x2d, accum2);

        i += 8;
    }

    // Secondary loop: 4 f32s for remainder 4-7
    while i + 3 < n {
        let x1f = _mm_loadu_ps(data1.as_ptr().add(i));
        let x2f = _mm_loadu_ps(data2.as_ptr().add(i));
        let x1d = _mm256_cvtps_pd(x1f);
        let x2d = _mm256_cvtps_pd(x2f);
        accum1 = _mm256_fmadd_pd(x1d, x2d, accum1);
        i += 4;
    }

    // Horizontal reduction: combine two accumulators, then reduce 4 f64s → 1
    accum1 = _mm256_add_pd(accum1, accum2);
    accum1 = _mm256_add_pd(accum1, _mm256_permute2f128_pd(accum1, accum1, 1));
    accum1 = _mm256_hadd_pd(accum1, accum1);
    let mut result = _mm256_cvtsd_f64(accum1);

    // Scalar tail for remaining 0-3 elements
    while i < n {
        result += *data1.get_unchecked(i) as f64 * *data2.get_unchecked(i) as f64;
        i += 1;
    }

    result
}

/// SSE2 implementation of `silk_inner_product_FLP`.
/// f32→f64 inner product using SSE2 `_mm_cvtps_pd` for widening.
///
/// # Safety
/// Requires SSE2 support (checked by caller via cpufeatures).
#[target_feature(enable = "sse2")]
pub unsafe fn silk_inner_product_flp_sse2(data1: &[f32], data2: &[f32]) -> f64 {
    let n = data1.len().min(data2.len());
    let mut sum = _mm_setzero_pd();
    let mut i = 0usize;

    while i + 3 < n {
        // Load 4 f32s, convert to 2 pairs of f64
        let x = _mm_loadu_ps(data1.as_ptr().add(i));
        let y = _mm_loadu_ps(data2.as_ptr().add(i));

        // Low 2 elements: f32 → f64
        let x_lo = _mm_cvtps_pd(x);
        let y_lo = _mm_cvtps_pd(y);
        sum = _mm_add_pd(sum, _mm_mul_pd(x_lo, y_lo));

        // High 2 elements: f32 → f64
        let x_hi = _mm_cvtps_pd(_mm_movehl_ps(x, x));
        let y_hi = _mm_cvtps_pd(_mm_movehl_ps(y, y));
        sum = _mm_add_pd(sum, _mm_mul_pd(x_hi, y_hi));

        i += 4;
    }

    // Horizontal sum of f64 pair
    let hi = _mm_unpackhi_pd(sum, sum);
    sum = _mm_add_sd(sum, hi);
    let mut result: f64 = 0.0;
    _mm_store_sd(&mut result, sum);

    // Handle remaining elements
    while i < n {
        result += *data1.get_unchecked(i) as f64 * *data2.get_unchecked(i) as f64;
        i += 1;
    }

    result
}

/// SSE4.1 implementation of the NSQ inner quantizer loop, specialized for
/// shapingLPCOrder=10 and predictLPCOrder=16.
/// Port of `silk/x86/NSQ_sse4_1.c:silk_noise_shape_quantizer_10_16_sse4_1`.
///
/// Maintains LPC and AR filter state in packed i16 SIMD registers for
/// register-resident operation. Uses table-based quantization decisions.
///
/// # Safety
/// Requires SSE4.1 support (checked by caller via cpufeatures).
#[target_feature(enable = "sse4.1")]
pub unsafe fn silk_noise_shape_quantizer_10_16_sse4_1(
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
    table: &[[i32; 4]; 64],
) {
    let rdo_offset = (Lambda_Q10 >> 1) - 512;

    let mut shp_lag_idx = (NSQ.sLTP_shp_buf_idx - lag + HARM_SHAPE_FIR_TAPS / 2) as usize;
    let mut pred_lag_idx = (NSQ.sLTP_buf_idx - lag + LTP_ORDER as i32 / 2) as usize;
    let Gain_Q10 = Gain_Q16 >> 6;

    let mut lpc_idx: usize = NSQ_LPC_BUF_LENGTH - 1;

    let mut sLF_AR_shp_Q14: i32 = NSQ.sLF_AR_shp_Q14;
    let mut xq_Q14: i32 = NSQ.sLPC_Q14[lpc_idx];
    let sDiff_shp_Q14: i32 = NSQ.sDiff_shp_Q14;
    let mut LTP_pred_Q13: i32 = 0;

    // --- Load a_Q12 coefficients, byte-reversed for paired computation ---
    let byte_rev = _mm_set_epi8(1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14);
    let a_Q12_01234567 =
        _mm_shuffle_epi8(_mm_loadu_si128(a_Q12.as_ptr() as *const __m128i), byte_rev);
    let a_Q12_89ABCDEF = _mm_shuffle_epi8(
        _mm_loadu_si128(a_Q12.as_ptr().add(8) as *const __m128i),
        byte_rev,
    );

    // --- Load AR_shp_Q13 coefficients (first 8 of 10) ---
    let AR_shp_Q13_76543210 = _mm_loadu_si128(AR_shp_Q13.as_ptr() as *const __m128i);

    // --- Load psLPC_Q14 state into interleaved hi/lo format ---
    let split_pattern = _mm_set_epi8(15, 14, 11, 10, 7, 6, 3, 2, 13, 12, 9, 8, 5, 4, 1, 0);

    let psLPC_ptr = NSQ.sLPC_Q14.as_ptr().add(lpc_idx);

    let mut tempa = _mm_shuffle_epi8(
        _mm_loadu_si128(psLPC_ptr.sub(16) as *const __m128i),
        split_pattern,
    );
    let mut tempb = _mm_shuffle_epi8(
        _mm_loadu_si128(psLPC_ptr.sub(12) as *const __m128i),
        split_pattern,
    );
    let mut psLPC_Q14_hi_89ABCDEF = _mm_unpackhi_epi64(tempa, tempb);
    let mut psLPC_Q14_lo_89ABCDEF = _mm_unpacklo_epi64(tempa, tempb);

    tempa = _mm_shuffle_epi8(
        _mm_loadu_si128(psLPC_ptr.sub(8) as *const __m128i),
        split_pattern,
    );
    tempb = _mm_shuffle_epi8(
        _mm_loadu_si128(psLPC_ptr.sub(4) as *const __m128i),
        split_pattern,
    );
    let mut psLPC_Q14_hi_01234567 = _mm_unpackhi_epi64(tempa, tempb);
    let mut psLPC_Q14_lo_01234567 = _mm_unpacklo_epi64(tempa, tempb);

    // --- Load sAR2_Q14 state into interleaved hi/lo format ---
    tempa = _mm_shuffle_epi8(
        _mm_loadu_si128(NSQ.sAR2_Q14.as_ptr() as *const __m128i),
        split_pattern,
    );
    tempb = _mm_shuffle_epi8(
        _mm_loadu_si128(NSQ.sAR2_Q14.as_ptr().add(4) as *const __m128i),
        split_pattern,
    );
    let mut sAR2_Q14_hi_76543210 = _mm_unpackhi_epi64(tempa, tempb);
    let mut sAR2_Q14_lo_76543210 = _mm_unpacklo_epi64(tempa, tempb);

    let xmm_one = _mm_set1_epi16(1);

    // =========== Main per-sample loop ===========
    for i in 0..length as usize {
        // ----- Short-term LPC prediction (order 16) -----
        let mut LPC_pred_Q10: i32 = 8;

        // Shift LPC sliding window
        psLPC_Q14_hi_89ABCDEF = _mm_alignr_epi8(psLPC_Q14_hi_01234567, psLPC_Q14_hi_89ABCDEF, 2);
        psLPC_Q14_lo_89ABCDEF = _mm_alignr_epi8(psLPC_Q14_lo_01234567, psLPC_Q14_lo_89ABCDEF, 2);
        psLPC_Q14_hi_01234567 = _mm_srli_si128(psLPC_Q14_hi_01234567, 2);
        psLPC_Q14_lo_01234567 = _mm_srli_si128(psLPC_Q14_lo_01234567, 2);
        psLPC_Q14_hi_01234567 = _mm_insert_epi16(psLPC_Q14_hi_01234567, xq_Q14 >> 16, 7);
        psLPC_Q14_lo_01234567 = _mm_insert_epi16(psLPC_Q14_lo_01234567, xq_Q14, 7);

        // High part: pmaddwd
        let xmm_hi_07 = _mm_madd_epi16(psLPC_Q14_hi_01234567, a_Q12_01234567);
        let xmm_hi_8F = _mm_madd_epi16(psLPC_Q14_hi_89ABCDEF, a_Q12_89ABCDEF);

        // Low part: pmulhw + sign correction
        let sign_07 = _mm_cmpgt_epi16(_mm_setzero_si128(), psLPC_Q14_lo_01234567);
        let sign_8F = _mm_cmpgt_epi16(_mm_setzero_si128(), psLPC_Q14_lo_89ABCDEF);
        let corr_07 = _mm_and_si128(sign_07, a_Q12_01234567);
        let corr_8F = _mm_and_si128(sign_8F, a_Q12_89ABCDEF);
        let mut xmm_lo_07 = _mm_mulhi_epi16(psLPC_Q14_lo_01234567, a_Q12_01234567);
        let mut xmm_lo_8F = _mm_mulhi_epi16(psLPC_Q14_lo_89ABCDEF, a_Q12_89ABCDEF);
        xmm_lo_07 = _mm_add_epi16(xmm_lo_07, corr_07);
        xmm_lo_8F = _mm_add_epi16(xmm_lo_8F, corr_8F);
        xmm_lo_07 = _mm_madd_epi16(xmm_lo_07, xmm_one);
        xmm_lo_8F = _mm_madd_epi16(xmm_lo_8F, xmm_one);

        // Accumulate
        let mut acc = _mm_add_epi32(
            _mm_add_epi32(xmm_hi_07, xmm_hi_8F),
            _mm_add_epi32(xmm_lo_07, xmm_lo_8F),
        );
        acc = _mm_add_epi32(acc, _mm_unpackhi_epi64(acc, acc));
        acc = _mm_add_epi32(acc, _mm_shufflelo_epi16(acc, 0x0E));
        LPC_pred_Q10 += _mm_cvtsi128_si32(acc);

        // ----- Long-term prediction -----
        if signalType == TYPE_VOICED {
            LTP_pred_Q13 = 2;
            let b_Q14_3210 = _mm_cvtepi16_epi32(_mm_loadl_epi64(b_Q14.as_ptr() as *const __m128i));
            let b_Q14_0123 = _mm_shuffle_epi32(b_Q14_3210, 0x1B);

            let pred_0123 =
                _mm_loadu_si128(sLTP_Q15.as_ptr().add(pred_lag_idx - 3) as *const __m128i);
            let pred_rev = _mm_shuffle_epi32(pred_0123, 0x1B);
            tempa = _mm_srli_si128(_mm_mul_epi32(pred_rev, b_Q14_3210), 2);
            tempb = _mm_srli_si128(_mm_mul_epi32(pred_0123, b_Q14_0123), 2);
            let sum4 = _mm_add_epi32(tempa, tempb);
            let sum2 = _mm_add_epi32(sum4, _mm_shuffle_epi32(sum4, 0x0E));
            LTP_pred_Q13 += _mm_cvtsi128_si32(sum2);

            // 5th tap scalar
            LTP_pred_Q13 = (LTP_pred_Q13 as i64
                + ((sLTP_Q15[pred_lag_idx - 4] as i64 * b_Q14[4] as i64) >> 16))
                as i32;
            pred_lag_idx += 1;
        }

        // ----- Noise shape feedback (SIMD for 8, scalar for 2) -----
        NSQ.sAR2_Q14[9] = NSQ.sAR2_Q14[8];
        NSQ.sAR2_Q14[8] = _mm_cvtsi128_si32(_mm_srli_si128(
            _mm_unpackhi_epi16(sAR2_Q14_lo_76543210, sAR2_Q14_hi_76543210),
            12,
        ));

        sAR2_Q14_hi_76543210 = _mm_slli_si128(sAR2_Q14_hi_76543210, 2);
        sAR2_Q14_lo_76543210 = _mm_slli_si128(sAR2_Q14_lo_76543210, 2);
        sAR2_Q14_hi_76543210 = _mm_insert_epi16(sAR2_Q14_hi_76543210, sDiff_shp_Q14 >> 16, 0);
        sAR2_Q14_lo_76543210 = _mm_insert_epi16(sAR2_Q14_lo_76543210, sDiff_shp_Q14, 0);

        let ar_hi = _mm_madd_epi16(sAR2_Q14_hi_76543210, AR_shp_Q13_76543210);
        let ar_sign = _mm_cmpgt_epi16(_mm_setzero_si128(), sAR2_Q14_lo_76543210);
        let ar_corr = _mm_and_si128(ar_sign, AR_shp_Q13_76543210);
        let mut ar_lo = _mm_mulhi_epi16(sAR2_Q14_lo_76543210, AR_shp_Q13_76543210);
        ar_lo = _mm_add_epi16(ar_lo, ar_corr);
        ar_lo = _mm_madd_epi16(ar_lo, xmm_one);

        let mut ar_acc = _mm_add_epi32(ar_hi, ar_lo);
        ar_acc = _mm_add_epi32(ar_acc, _mm_unpackhi_epi64(ar_acc, ar_acc));
        ar_acc = _mm_add_epi32(ar_acc, _mm_shufflelo_epi16(ar_acc, 0x0E));
        let mut n_AR_Q12: i32 = 5 + _mm_cvtsi128_si32(ar_acc);

        n_AR_Q12 =
            (n_AR_Q12 as i64 + ((NSQ.sAR2_Q14[8] as i64 * AR_shp_Q13[8] as i64) >> 16)) as i32;
        n_AR_Q12 =
            (n_AR_Q12 as i64 + ((NSQ.sAR2_Q14[9] as i64 * AR_shp_Q13[9] as i64) >> 16)) as i32;

        n_AR_Q12 = ((n_AR_Q12 as u32) << 1) as i32;
        n_AR_Q12 =
            (n_AR_Q12 as i64 + ((sLF_AR_shp_Q14 as i64 * Tilt_Q14 as i16 as i64) >> 16)) as i32;

        let n_LF_Q12: i32 = {
            let t1 = ((NSQ.sLTP_shp_Q14[(NSQ.sLTP_shp_buf_idx - 1) as usize] as i64
                * LF_shp_Q14 as i16 as i64)
                >> 16) as i32;
            (t1 as i64 + ((sLF_AR_shp_Q14 as i64 * (LF_shp_Q14 as i64 >> 16)) >> 16)) as i32
        };

        // ----- Combine prediction and noise shaping -----
        let mut tmp1 = (((LPC_pred_Q10 as u32) << 2) as i32).wrapping_sub(n_AR_Q12);
        tmp1 = tmp1.wrapping_sub(n_LF_Q12);
        if lag > 0 {
            let n_LTP_Q13 = {
                let t1 = ((NSQ.sLTP_shp_Q14[shp_lag_idx]
                    .saturating_add(NSQ.sLTP_shp_Q14[shp_lag_idx - 2]))
                    as i64
                    * HarmShapeFIRPacked_Q14 as i16 as i64)
                    >> 16;
                let t2 = (t1
                    + ((NSQ.sLTP_shp_Q14[shp_lag_idx - 1] as i64
                        * (HarmShapeFIRPacked_Q14 as i64 >> 16))
                        >> 16)) as i32;
                ((t2 as u32) << 1) as i32
            };
            shp_lag_idx += 1;
            let tmp2 = LTP_pred_Q13 - n_LTP_Q13;
            tmp1 = tmp2.wrapping_add(((tmp1 as u32) << 1) as i32);
            tmp1 = ((tmp1 >> 2) + 1) >> 1;
        } else {
            tmp1 = ((tmp1 >> 1) + 1) >> 1;
        }

        let mut r_Q10 = x_sc_Q10[i] - tmp1;

        NSQ.rand_seed = silk_RAND(NSQ.rand_seed);
        if NSQ.rand_seed < 0 {
            r_Q10 = -r_Q10;
        }
        r_Q10 = r_Q10.clamp(-(31 << 10), 30 << 10);

        // ----- Table-based quantization -----
        let mut q1_Q0 = (r_Q10 - offset_Q10) >> 10;
        if Lambda_Q10 > 2048 {
            let q1_Q10_tmp = r_Q10 - offset_Q10;
            if q1_Q10_tmp > rdo_offset {
                q1_Q0 = (q1_Q10_tmp - rdo_offset) >> 10;
            } else if q1_Q10_tmp < -rdo_offset {
                q1_Q0 = (q1_Q10_tmp + rdo_offset) >> 10;
            } else if q1_Q10_tmp < 0 {
                q1_Q0 = -1;
            } else {
                q1_Q0 = 0;
            }
        }

        let tidx = (q1_Q0 + 32).clamp(0, 63) as usize;
        let mut q1_Q10 = table[tidx][0];
        let q2_Q10 = table[tidx][1];
        if (r_Q10
            .wrapping_mul(table[tidx][2])
            .wrapping_sub(table[tidx][3]))
            < 0
        {
            q1_Q10 = q2_Q10;
        }

        pulses[i] = (((q1_Q10 >> 9) + 1) >> 1) as i8;

        // ----- Excitation and state update -----
        let mut exc_Q14 = ((q1_Q10 as u32) << 4) as i32;
        if NSQ.rand_seed < 0 {
            exc_Q14 = -exc_Q14;
        }
        let LPC_exc_Q14 = exc_Q14 + ((LTP_pred_Q13 as u32) << 1) as i32;
        xq_Q14 = LPC_exc_Q14.wrapping_add(((LPC_pred_Q10 as u32) << 4) as i32);

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

    // =========== Post-loop: write back sAR2_Q14 ===========
    tempa = _mm_unpackhi_epi16(sAR2_Q14_lo_76543210, sAR2_Q14_hi_76543210);
    tempb = _mm_unpacklo_epi16(sAR2_Q14_lo_76543210, sAR2_Q14_hi_76543210);
    _mm_storeu_si128(NSQ.sAR2_Q14.as_mut_ptr().add(4) as *mut __m128i, tempa);
    _mm_storeu_si128(NSQ.sAR2_Q14.as_mut_ptr() as *mut __m128i, tempb);

    // =========== Post-loop: SIMD XQ output scaling ===========
    let psLPC_Q14_out = &NSQ.sLPC_Q14[NSQ_LPC_BUF_LENGTH..];
    let xmm_round = _mm_set1_epi32(1 << 7);
    let xmm_Gain_Q10 = _mm_set1_epi32(Gain_Q10);

    let mut ii = 0i32;
    while ii < length - 7 {
        let ui = ii as usize;
        let xq_3210 = _mm_loadu_si128(psLPC_Q14_out.as_ptr().add(ui) as *const __m128i);
        let xq_7654 = _mm_loadu_si128(psLPC_Q14_out.as_ptr().add(ui + 4) as *const __m128i);

        let x3x1 = _mm_shuffle_epi32(xq_3210, 0x39); // (0,3,2,1)
        let x7x5 = _mm_shuffle_epi32(xq_7654, 0x39);

        let mut r_3210 = _mm_srli_epi64(_mm_mul_epi32(xq_3210, xmm_Gain_Q10), 16);
        let r_x3x1 = _mm_slli_epi64(_mm_mul_epi32(x3x1, xmm_Gain_Q10), 16);
        let mut r_7654 = _mm_srli_epi64(_mm_mul_epi32(xq_7654, xmm_Gain_Q10), 16);
        let r_x7x5 = _mm_slli_epi64(_mm_mul_epi32(x7x5, xmm_Gain_Q10), 16);

        r_3210 = _mm_blend_epi16(r_3210, r_x3x1, 0xCC);
        r_7654 = _mm_blend_epi16(r_7654, r_x7x5, 0xCC);

        r_3210 = _mm_srai_epi32(_mm_add_epi32(r_3210, xmm_round), 8);
        r_7654 = _mm_srai_epi32(_mm_add_epi32(r_7654, xmm_round), 8);

        let packed = _mm_packs_epi32(r_3210, r_7654);
        _mm_storeu_si128(NSQ.xq.as_mut_ptr().add(xq_off + ui) as *mut __m128i, packed);
        ii += 8;
    }
    while ii < length {
        let ui = ii as usize;
        let smulww = ((psLPC_Q14_out[ui] as i64 * Gain_Q10 as i64) >> 16) as i32;
        let rounded = ((smulww >> 7) + 1) >> 1;
        NSQ.xq[xq_off + ui] = rounded.clamp(-32768, 32767) as i16;
        ii += 1;
    }

    // =========== Post-loop: copy LPC buffer ===========
    NSQ.sLPC_Q14
        .copy_within(length as usize..length as usize + NSQ_LPC_BUF_LENGTH, 0);
}

/// SSE4.1 implementation of `silk_nsq_del_dec_scale_states`.
/// SIMD-accelerated input scaling and gain adjustment loops.
///
/// # Safety
/// Requires SSE4.1 support (checked by caller via cpufeatures).
#[target_feature(enable = "sse4.1")]
pub unsafe fn silk_nsq_del_dec_scale_states_sse4_1(
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
    let mut inv_gain_Q31 = silk_INVERSE32_varQ(Gains_Q16[subfr as usize].max(1), 47);

    let inv_gain_Q26 = ((inv_gain_Q31 >> 4) + 1) >> 1;

    // SIMD input scaling: x_sc_Q10[i] = silk_SMULWW(x16[i], inv_gain_Q26)
    let xmm_inv_gain = _mm_set1_epi32(inv_gain_Q26);
    let subfr_len = psEncC.subfr_length;
    let mut i = 0usize;
    while i + 3 < subfr_len {
        let xmm_x16 = _mm_cvtepi16_epi32(_mm_loadl_epi64(x16.as_ptr().add(i) as *const __m128i));
        let xmm_odd = _mm_shuffle_epi32(xmm_x16, 0x39);
        let mut r_even = _mm_mul_epi32(xmm_x16, xmm_inv_gain);
        let r_odd = _mm_mul_epi32(xmm_odd, xmm_inv_gain);
        r_even = _mm_srli_epi64(r_even, 16);
        let r_odd_s = _mm_slli_epi64(r_odd, 16);
        let result = _mm_blend_epi16(r_even, r_odd_s, 0xCC);
        _mm_storeu_si128(x_sc_Q10.as_mut_ptr().add(i) as *mut __m128i, result);
        i += 4;
    }
    while i < subfr_len {
        x_sc_Q10[i] = ((x16[i] as i64 * inv_gain_Q26 as i64) >> 16) as i32;
        i += 1;
    }

    // LTP state rewhitening (scalar)
    if NSQ.rewhite_flag != 0 {
        if subfr == 0 {
            inv_gain_Q31 = ((((inv_gain_Q31 as i64 * LTP_scale_Q14 as i16 as i64) >> 16) as i32
                as u32)
                << 2) as i32;
        }
        let start = (NSQ.sLTP_buf_idx - lag - LTP_ORDER as i32 / 2) as usize;
        let end = NSQ.sLTP_buf_idx as usize;
        for j in start..end {
            sLTP_Q15[j] = ((inv_gain_Q31 as i64 * sLTP[j] as i64) >> 16) as i32;
        }
    }

    // Gain adjustment
    if Gains_Q16[subfr as usize] != NSQ.prev_gain_Q16 {
        let gain_adj_Q16 = silk_DIV32_varQ(NSQ.prev_gain_Q16, Gains_Q16[subfr as usize], 16);

        // SIMD scaling of sLTP_shp_Q14
        let xmm_gain_adj = _mm_set1_epi32(gain_adj_Q16);
        let shp_start = (NSQ.sLTP_shp_buf_idx - psEncC.ltp_mem_length as i32) as usize;
        let shp_end = NSQ.sLTP_shp_buf_idx as usize;
        let mut j = shp_start;
        while j + 3 < shp_end {
            let vals = _mm_loadu_si128(NSQ.sLTP_shp_Q14.as_ptr().add(j) as *const __m128i);
            let vals_odd = _mm_shuffle_epi32(vals, 0x39);
            let mut r_even = _mm_mul_epi32(vals, xmm_gain_adj);
            let r_odd = _mm_mul_epi32(vals_odd, xmm_gain_adj);
            r_even = _mm_srli_epi64(r_even, 16);
            let r_odd_s = _mm_slli_epi64(r_odd, 16);
            let result = _mm_blend_epi16(r_even, r_odd_s, 0xCC);
            _mm_storeu_si128(NSQ.sLTP_shp_Q14.as_mut_ptr().add(j) as *mut __m128i, result);
            j += 4;
        }
        while j < shp_end {
            NSQ.sLTP_shp_Q14[j] = ((gain_adj_Q16 as i64 * NSQ.sLTP_shp_Q14[j] as i64) >> 16) as i32;
            j += 1;
        }

        // Scale LTP prediction state
        if signal_type == TYPE_VOICED && NSQ.rewhite_flag == 0 {
            let start = (NSQ.sLTP_buf_idx - lag - LTP_ORDER as i32 / 2) as usize;
            let end = (NSQ.sLTP_buf_idx - decisionDelay) as usize;
            for val in sLTP_Q15[start..end].iter_mut() {
                *val = ((gain_adj_Q16 as i64 * *val as i64) >> 16) as i32;
            }
        }

        // Per-state scaling
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

/// SSE4.1 implementation of `silk_noise_shape_quantizer_del_dec`.
/// SIMD-accelerated LPC and LTP prediction with scalar noise shaping and quantization.
///
/// # Safety
/// Requires SSE4.1 support (checked by caller via cpufeatures).
#[target_feature(enable = "sse4.1")]
pub unsafe fn silk_noise_shape_quantizer_del_dec_sse4_1(
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
) {
    let nStates = nStatesDelayedDecision as usize;
    let mut psSampleState: Vec<NSQ_sample_pair> = vec![[NSQ_sample_struct::default(); 2]; nStates];

    let mut shp_lag_idx = (NSQ.sLTP_shp_buf_idx - lag + HARM_SHAPE_FIR_TAPS / 2) as usize;
    let mut pred_lag_idx = (NSQ.sLTP_buf_idx - lag + LTP_ORDER as i32 / 2) as usize;
    let Gain_Q10: i32 = Gain_Q16 >> 6;

    let rdo_offset = (Lambda_Q10 >> 1) - 512;

    // Pre-load a_Q12 coefficients into SIMD registers
    let a_Q12_0123 = _mm_cvtepi16_epi32(_mm_loadl_epi64(a_Q12.as_ptr() as *const __m128i));
    let a_Q12_4567 = _mm_cvtepi16_epi32(_mm_loadl_epi64(a_Q12.as_ptr().add(4) as *const __m128i));
    let (a_Q12_89AB, a_Q12_CDEF) = if predictLPCOrder == 16 {
        (
            _mm_cvtepi16_epi32(_mm_loadl_epi64(a_Q12.as_ptr().add(8) as *const __m128i)),
            _mm_cvtepi16_epi32(_mm_loadl_epi64(a_Q12.as_ptr().add(12) as *const __m128i)),
        )
    } else {
        (_mm_setzero_si128(), _mm_setzero_si128())
    };

    // Pre-load b_Q14 for LTP
    let b_Q14_0123 = if signalType == TYPE_VOICED {
        _mm_cvtepi16_epi32(_mm_loadl_epi64(b_Q14.as_ptr() as *const __m128i))
    } else {
        _mm_setzero_si128()
    };

    #[allow(clippy::needless_range_loop)]
    for i in 0..length as usize {
        // ---- LTP prediction (SIMD for 4 taps + 1 scalar) ----
        let mut LTP_pred_Q14: i32;
        if signalType == TYPE_VOICED {
            LTP_pred_Q14 = 2;
            let pred_vals =
                _mm_loadu_si128(sLTP_Q15.as_ptr().add(pred_lag_idx - 3) as *const __m128i);
            let pred_rev = _mm_shuffle_epi32(pred_vals, 0x1B);
            let tmpa = _mm_srli_epi64(_mm_mul_epi32(pred_rev, b_Q14_0123), 16);
            let pred_rot = _mm_shuffle_epi32(pred_rev, 0x39);
            let b_rot = _mm_shuffle_epi32(b_Q14_0123, 0x39);
            let tmpb = _mm_srli_epi64(_mm_mul_epi32(pred_rot, b_rot), 16);
            let sum4 = _mm_add_epi32(tmpa, tmpb);
            let sum2 = _mm_add_epi32(sum4, _mm_shuffle_epi32(sum4, 0x0E));
            LTP_pred_Q14 += _mm_cvtsi128_si32(sum2);
            LTP_pred_Q14 = (LTP_pred_Q14 as i64
                + ((sLTP_Q15[pred_lag_idx - 4] as i64 * b_Q14[4] as i64) >> 16))
                as i32;
            LTP_pred_Q14 = ((LTP_pred_Q14 as u32) << 1) as i32;
            pred_lag_idx += 1;
        } else {
            LTP_pred_Q14 = 0;
        }

        // ---- Harmonic noise shaping (scalar, shared across states) ----
        let n_LTP_Q14: i32;
        if lag > 0 {
            n_LTP_Q14 = {
                let t = ((NSQ.sLTP_shp_Q14[shp_lag_idx]
                    .saturating_add(NSQ.sLTP_shp_Q14[shp_lag_idx - 2]))
                    as i64
                    * HarmShapeFIRPacked_Q14 as i16 as i64)
                    >> 16;
                let t2 = (t
                    + ((NSQ.sLTP_shp_Q14[shp_lag_idx - 1] as i64
                        * (HarmShapeFIRPacked_Q14 as i64 >> 16))
                        >> 16)) as i32;
                LTP_pred_Q14 - ((t2 as u32) << 2) as i32
            };
            shp_lag_idx += 1;
        } else {
            n_LTP_Q14 = 0;
        }

        // ---- Per-state processing ----
        for k in 0..nStates {
            let psDD = &mut psDelDec[k];
            psDD.Seed = silk_RAND(psDD.Seed);

            // ---- SIMD LPC prediction ----
            let lpc_idx = NSQ_LPC_BUF_LENGTH - 1 + i;
            let psLPC_ptr = psDD.sLPC_Q14.as_ptr().add(lpc_idx);
            let mut LPC_pred_Q14: i32 = predictLPCOrder >> 1;

            let mut acc = _mm_setzero_si128();

            // Step 1: coefficients 0-3
            let lpc_vals = _mm_loadu_si128(psLPC_ptr.sub(3) as *const __m128i);
            let lpc_rev = _mm_shuffle_epi32(lpc_vals, 0x1B);
            acc = _mm_add_epi32(acc, _mm_srli_epi64(_mm_mul_epi32(lpc_rev, a_Q12_0123), 16));
            let lpc_rot = _mm_shuffle_epi32(lpc_rev, 0x39);
            let a_rot = _mm_shuffle_epi32(a_Q12_0123, 0x39);
            acc = _mm_add_epi32(acc, _mm_srli_epi64(_mm_mul_epi32(lpc_rot, a_rot), 16));

            // Step 2: coefficients 4-7
            let lpc_vals = _mm_loadu_si128(psLPC_ptr.sub(7) as *const __m128i);
            let lpc_rev = _mm_shuffle_epi32(lpc_vals, 0x1B);
            acc = _mm_add_epi32(acc, _mm_srli_epi64(_mm_mul_epi32(lpc_rev, a_Q12_4567), 16));
            let lpc_rot = _mm_shuffle_epi32(lpc_rev, 0x39);
            let a_rot = _mm_shuffle_epi32(a_Q12_4567, 0x39);
            acc = _mm_add_epi32(acc, _mm_srli_epi64(_mm_mul_epi32(lpc_rot, a_rot), 16));

            if predictLPCOrder == 16 {
                // Step 3: coefficients 8-11
                let lpc_vals = _mm_loadu_si128(psLPC_ptr.sub(11) as *const __m128i);
                let lpc_rev = _mm_shuffle_epi32(lpc_vals, 0x1B);
                acc = _mm_add_epi32(acc, _mm_srli_epi64(_mm_mul_epi32(lpc_rev, a_Q12_89AB), 16));
                let lpc_rot = _mm_shuffle_epi32(lpc_rev, 0x39);
                let a_rot = _mm_shuffle_epi32(a_Q12_89AB, 0x39);
                acc = _mm_add_epi32(acc, _mm_srli_epi64(_mm_mul_epi32(lpc_rot, a_rot), 16));

                // Step 4: coefficients 12-15
                let lpc_vals = _mm_loadu_si128(psLPC_ptr.sub(15) as *const __m128i);
                let lpc_rev = _mm_shuffle_epi32(lpc_vals, 0x1B);
                acc = _mm_add_epi32(acc, _mm_srli_epi64(_mm_mul_epi32(lpc_rev, a_Q12_CDEF), 16));
                let lpc_rot = _mm_shuffle_epi32(lpc_rev, 0x39);
                let a_rot = _mm_shuffle_epi32(a_Q12_CDEF, 0x39);
                acc = _mm_add_epi32(acc, _mm_srli_epi64(_mm_mul_epi32(lpc_rot, a_rot), 16));

                let hi = _mm_shuffle_epi32(acc, 0x0E);
                acc = _mm_add_epi32(acc, hi);
                LPC_pred_Q14 += _mm_cvtsi128_si32(acc);
            } else {
                let hi = _mm_shuffle_epi32(acc, 0x0E);
                acc = _mm_add_epi32(acc, hi);
                LPC_pred_Q14 += _mm_cvtsi128_si32(acc);
                LPC_pred_Q14 = (LPC_pred_Q14 as i64
                    + ((*psLPC_ptr.sub(8) as i64 * a_Q12[8] as i64) >> 16))
                    as i32;
                LPC_pred_Q14 = (LPC_pred_Q14 as i64
                    + ((*psLPC_ptr.sub(9) as i64 * a_Q12[9] as i64) >> 16))
                    as i32;
            }

            LPC_pred_Q14 = ((LPC_pred_Q14 as u32) << 4) as i32;

            // ---- Noise shaping with warping (scalar) ----
            let mut tmp2 = (psDD.Diff_Q14 as i64
                + ((psDD.sAR2_Q14[0] as i64 * warping_Q16 as i16 as i64) >> 16))
                as i32;
            let mut tmp1 = (psDD.sAR2_Q14[0] as i64
                + (((psDD.sAR2_Q14[1].wrapping_sub(tmp2)) as i64 * warping_Q16 as i16 as i64)
                    >> 16)) as i32;
            psDD.sAR2_Q14[0] = tmp2;
            let mut n_AR_Q14: i32 = shapingLPCOrder >> 1;
            n_AR_Q14 = (n_AR_Q14 as i64 + ((tmp2 as i64 * AR_shp_Q13[0] as i64) >> 16)) as i32;
            let mut j = 2;
            while j < shapingLPCOrder {
                tmp2 = (psDD.sAR2_Q14[(j - 1) as usize] as i64
                    + (((psDD.sAR2_Q14[j as usize].wrapping_sub(tmp1)) as i64
                        * warping_Q16 as i16 as i64)
                        >> 16)) as i32;
                psDD.sAR2_Q14[(j - 1) as usize] = tmp1;
                n_AR_Q14 = (n_AR_Q14 as i64
                    + ((tmp1 as i64 * AR_shp_Q13[(j - 1) as usize] as i64) >> 16))
                    as i32;
                tmp1 = (psDD.sAR2_Q14[j as usize] as i64
                    + (((psDD.sAR2_Q14[(j + 1) as usize].wrapping_sub(tmp2)) as i64
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

            let n_LF_Q14: i32 = {
                let t1 = ((psDD.Shape_Q14[*smpl_buf_idx as usize] as i64
                    * LF_shp_Q14 as i16 as i64)
                    >> 16) as i32;
                let t2 = (t1 as i64 + ((psDD.LF_AR_Q14 as i64 * (LF_shp_Q14 as i64 >> 16)) >> 16))
                    as i32;
                ((t2 as u32) << 2) as i32
            };

            // ---- Combine prediction and noise feedback ----
            tmp1 = n_AR_Q14.saturating_add(n_LF_Q14);
            tmp2 = n_LTP_Q14 + LPC_pred_Q14;
            tmp1 = tmp2.saturating_sub(tmp1);
            tmp1 = ((tmp1 >> 3) + 1) >> 1;

            let mut r_Q10 = x_Q10[i] - tmp1;
            if psDD.Seed < 0 {
                r_Q10 = -r_Q10;
            }
            r_Q10 = r_Q10.clamp(-(31 << 10), 30 << 10);

            // ---- Quantization decision ----
            let mut q1_Q10 = r_Q10 - offset_Q10;
            let mut q1_Q0 = q1_Q10 >> 10;
            if Lambda_Q10 > 2048 {
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
            let q2_Q10: i32;
            let rd1_Q10: i32;
            let rd2_Q10: i32;
            if q1_Q0 > 0 {
                q1_Q10 = ((q1_Q0 as u32) << 10) as i32 - 80 + offset_Q10;
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
                q1_Q10 = ((q1_Q0 as u32) << 10) as i32 + 80 + offset_Q10;
                q2_Q10 = q1_Q10 + 1024;
                rd1_Q10 = -q1_Q10 as i16 as i32 * Lambda_Q10 as i16 as i32;
                rd2_Q10 = -q2_Q10 as i16 as i32 * Lambda_Q10 as i16 as i32;
            }
            let mut rr_Q10 = r_Q10 - q1_Q10;
            let rd1_Q10 = (rd1_Q10 + rr_Q10 as i16 as i32 * rr_Q10 as i16 as i32) >> 10;
            rr_Q10 = r_Q10 - q2_Q10;
            let rd2_Q10 = (rd2_Q10 + rr_Q10 as i16 as i32 * rr_Q10 as i16 as i32) >> 10;

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

            // Compute outputs for best and second-best
            let mut exc_Q14 = ((psSampleState[k][0].Q_Q10 as u32) << 4) as i32;
            if psDD.Seed < 0 {
                exc_Q14 = -exc_Q14;
            }
            let mut LPC_exc_Q14 = exc_Q14 + LTP_pred_Q14;
            let mut xq_Q14 = LPC_exc_Q14 + LPC_pred_Q14;
            psSampleState[k][0].Diff_Q14 = xq_Q14 - ((x_Q10[i] as u32) << 4) as i32;
            let mut sLF_AR_shp_Q14 = psSampleState[k][0].Diff_Q14 - n_AR_Q14;
            psSampleState[k][0].sLTP_shp_Q14 = sLF_AR_shp_Q14.saturating_sub(n_LF_Q14);
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
            psSampleState[k][1].sLTP_shp_Q14 = sLF_AR_shp_Q14.saturating_sub(n_LF_Q14);
            psSampleState[k][1].LF_AR_Q14 = sLF_AR_shp_Q14;
            psSampleState[k][1].LPC_exc_Q14 = LPC_exc_Q14;
            psSampleState[k][1].xq_Q14 = xq_Q14;
        }

        // ---- Winner selection, pruning, output ----
        *smpl_buf_idx = (*smpl_buf_idx - 1) % DECISION_DELAY;
        if *smpl_buf_idx < 0 {
            *smpl_buf_idx += DECISION_DELAY;
        }
        let last_smple_idx = (*smpl_buf_idx + decisionDelay) % DECISION_DELAY;

        let mut RDmin_Q10 = psSampleState[0][0].RD_Q10;
        let mut Winner_ind: i32 = 0;
        for k in 1..nStates {
            if psSampleState[k][0].RD_Q10 < RDmin_Q10 {
                RDmin_Q10 = psSampleState[k][0].RD_Q10;
                Winner_ind = k as i32;
            }
        }

        let Winner_rand_state = psDelDec[Winner_ind as usize].RandState[last_smple_idx as usize];
        for k in 0..nStates {
            if psDelDec[k].RandState[last_smple_idx as usize] != Winner_rand_state {
                psSampleState[k][0].RD_Q10 += 0x7fffffff >> 4;
                psSampleState[k][1].RD_Q10 += 0x7fffffff >> 4;
            }
        }

        let mut RDmax_Q10 = psSampleState[0][0].RD_Q10;
        RDmin_Q10 = psSampleState[0][1].RD_Q10;
        let mut RDmax_ind: i32 = 0;
        let mut RDmin_ind: i32 = 0;
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

        if RDmin_Q10 < RDmax_Q10 {
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

        if subfr > 0 || i as i32 >= decisionDelay {
            let psDD_w = &psDelDec[Winner_ind as usize];
            let out_idx = pulses_off + i - decisionDelay as usize;
            pulses[out_idx] = (((psDD_w.Q_Q10[last_smple_idx as usize] >> 9) + 1) >> 1) as i8;
            let xq_val = (psDD_w.Xq_Q14[last_smple_idx as usize] as i64
                * delayedGain_Q10[last_smple_idx as usize] as i64)
                >> 16;
            let rounded = ((xq_val as i32 >> 7) + 1) >> 1;
            NSQ.xq[xq_off + i - decisionDelay as usize] = rounded.clamp(-32768, 32767) as i16;
            NSQ.sLTP_shp_Q14[(NSQ.sLTP_shp_buf_idx - decisionDelay) as usize] =
                psDD_w.Shape_Q14[last_smple_idx as usize];
            sLTP_Q15[(NSQ.sLTP_buf_idx - decisionDelay) as usize] =
                psDD_w.Pred_Q15[last_smple_idx as usize];
        }
        NSQ.sLTP_shp_buf_idx += 1;
        NSQ.sLTP_buf_idx += 1;

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
            psDD.Seed =
                (psDD.Seed as u32).wrapping_add((((psSS.Q_Q10 >> 9) + 1) >> 1) as u32) as i32;
            psDD.RandState[*smpl_buf_idx as usize] = psDD.Seed;
            psDD.RD_Q10 = psSS.RD_Q10;
        }
        delayedGain_Q10[*smpl_buf_idx as usize] = Gain_Q10;
    }

    for dd in psDelDec[..nStates].iter_mut() {
        dd.sLPC_Q14
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
pub unsafe fn silk_VQ_WMat_EC_sse4_1(
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
    let mut neg_xX_Q24: [i32; 5] = [0; 5];
    neg_xX_Q24[0] = -(((xX_Q17[0] as u32) << 7) as i32);
    neg_xX_Q24[1] = -(((xX_Q17[1] as u32) << 7) as i32);
    neg_xX_Q24[2] = -(((xX_Q17[2] as u32) << 7) as i32);
    neg_xX_Q24[3] = -(((xX_Q17[3] as u32) << 7) as i32);
    neg_xX_Q24[4] = -(((xX_Q17[4] as u32) << 7) as i32);

    // Load XX_Q17[1..5] and create two shuffled views for the first row SIMD computation
    // v_XX_31_Q17 = [XX_Q17[1], XX_Q17[2], XX_Q17[3], XX_Q17[4]]
    let v_XX_31_Q17 = _mm_loadu_si128(XX_Q17.as_ptr().add(1) as *const __m128i);
    // v_XX_42_Q17 = [XX_Q17[2], XX_Q17[3], XX_Q17[4], XX_Q17[1]]
    let v_XX_42_Q17 = _mm_shuffle_epi32(v_XX_31_Q17, 0x39); // _MM_SHUFFLE(0,3,2,1)

    *rate_dist_Q8 = i32::MAX;
    *res_nrg_Q15 = i32::MAX;
    *ind = 0;
    let mut cb_row_off: usize = 0;

    for k in 0..L as usize {
        let gain_tmp_Q7 = cb_gain_Q7[k] as i32;
        let mut sum1_Q15: i32 = (1.001f64 * ((1) << 15) as f64 + 0.5f64) as i32;

        let penalty: i32 = (((if gain_tmp_Q7 - max_gain_Q7 > 0 {
            gain_tmp_Q7 - max_gain_Q7
        } else {
            0
        }) as u32)
            << 11) as i32;

        // First row of XX_Q17 — SIMD accelerated
        // Sign-extend cb_row_Q7[1..5] from i8 to i32
        let cb_ptr = cb_Q7.as_ptr().add(cb_row_off + 1);
        let v_cb_row_31_Q7 =
            _mm_cvtepi8_epi32(_mm_cvtsi32_si128((cb_ptr as *const i32).read_unaligned()));
        let v_cb_row_42_Q7 = _mm_shuffle_epi32(v_cb_row_31_Q7, 0x39);

        // Widening multiply: XX_Q17[i] * cb_Q7[j] -> i64, then horizontal sum
        let v_prod_31 = _mm_mul_epi32(v_XX_31_Q17, v_cb_row_31_Q7);
        let v_prod_42 = _mm_mul_epi32(v_XX_42_Q17, v_cb_row_42_Q7);
        let v_acc1 = _mm_add_epi64(v_prod_31, v_prod_42);
        let v_acc2 = _mm_shuffle_epi32(v_acc1, 0x4E); // swap hi/lo 64-bit
        let v_acc1 = _mm_add_epi64(v_acc1, v_acc2);
        let mut sum2_Q24: i32 = _mm_cvtsi128_si32(v_acc1);

        sum2_Q24 = neg_xX_Q24[0].wrapping_add(sum2_Q24);
        sum2_Q24 = ((sum2_Q24 as u32) << 1) as i32;
        sum2_Q24 = sum2_Q24.wrapping_add(XX_Q17[0].wrapping_mul(cb_Q7[cb_row_off] as i32));
        sum1_Q15 =
            (sum1_Q15 as i64 + ((sum2_Q24 as i64 * cb_Q7[cb_row_off] as i16 as i64) >> 16)) as i32;

        // Rows 2-5: scalar (same as scalar version)
        sum2_Q24 = neg_xX_Q24[1] + XX_Q17[7] * cb_Q7[cb_row_off + 2] as i32;
        sum2_Q24 += XX_Q17[8] * cb_Q7[cb_row_off + 3] as i32;
        sum2_Q24 += XX_Q17[9] * cb_Q7[cb_row_off + 4] as i32;
        sum2_Q24 = ((sum2_Q24 as u32) << 1) as i32;
        sum2_Q24 += XX_Q17[6] * cb_Q7[cb_row_off + 1] as i32;
        sum1_Q15 = (sum1_Q15 as i64
            + ((sum2_Q24 as i64 * cb_Q7[cb_row_off + 1] as i16 as i64) >> 16))
            as i32;

        sum2_Q24 = neg_xX_Q24[2] + XX_Q17[13] * cb_Q7[cb_row_off + 3] as i32;
        sum2_Q24 += XX_Q17[14] * cb_Q7[cb_row_off + 4] as i32;
        sum2_Q24 = ((sum2_Q24 as u32) << 1) as i32;
        sum2_Q24 += XX_Q17[12] * cb_Q7[cb_row_off + 2] as i32;
        sum1_Q15 = (sum1_Q15 as i64
            + ((sum2_Q24 as i64 * cb_Q7[cb_row_off + 2] as i16 as i64) >> 16))
            as i32;

        sum2_Q24 = neg_xX_Q24[3] + XX_Q17[19] * cb_Q7[cb_row_off + 4] as i32;
        sum2_Q24 = ((sum2_Q24 as u32) << 1) as i32;
        sum2_Q24 += XX_Q17[18] * cb_Q7[cb_row_off + 3] as i32;
        sum1_Q15 = (sum1_Q15 as i64
            + ((sum2_Q24 as i64 * cb_Q7[cb_row_off + 3] as i16 as i64) >> 16))
            as i32;

        sum2_Q24 = ((neg_xX_Q24[4] as u32) << 1) as i32;
        sum2_Q24 += XX_Q17[24] * cb_Q7[cb_row_off + 4] as i32;
        sum1_Q15 = (sum1_Q15 as i64
            + ((sum2_Q24 as i64 * cb_Q7[cb_row_off + 4] as i16 as i64) >> 16))
            as i32;

        if sum1_Q15 >= 0 {
            let bits_res_Q8 = subfr_len as i16 as i32
                * (crate::silk::lin2log::silk_lin2log(sum1_Q15 + penalty) - ((15) << 7)) as i16
                    as i32;
            let bits_tot_Q8 = bits_res_Q8 + ((cl_Q5[k] as u32) << (3 - 1)) as i32;
            if bits_tot_Q8 <= *rate_dist_Q8 {
                *rate_dist_Q8 = bits_tot_Q8;
                *res_nrg_Q15 = sum1_Q15 + penalty;
                *ind = k as i8;
                *gain_Q7 = gain_tmp_Q7;
            }
        }
        cb_row_off += LTP_ORDER;
    }
}

// ============================================================================
// AVX2 NSQ del_dec implementation
// Port of silk/x86/NSQ_del_dec_avx2.c
// ============================================================================

use crate::silk::define::MAX_SUB_FRAME_LENGTH;
const RAND_MULTIPLIER_I32: i32 = 196314165;
const RAND_INCREMENT_I32: i32 = 907633515;

/// Extract high 32 bits of each 64-bit lane from a 256-bit vector.
/// Equivalent to C: `silk_cvtepi64_epi32_high`
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

    // Avoids introducing a bias because silk_SMLAWB() always rounds to -inf
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
unsafe fn silk_LPC_analysis_filter_avx2(
    out: &mut [i16],
    input: &[i16],
    b: &[i16],
    len: i32,
    order: i32,
) {
    debug_assert!(order == 10 || order == 16);

    for i in order..len {
        let in_ptr = input.as_ptr().add(i as usize);

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

        // Subtract prediction: silk_LSHIFT(in[i], 12) - out32_Q12
        let out32_q12 = ((*in_ptr as i32 as u32) << 12) as i32 - out32_q12;

        // Scale to Q0 with rounding
        let out32 = silk_sar_round_32(out32_q12, 12);

        // Saturate output
        out[i as usize] = silk_sat16(out32);
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
    RandState: __m128i,
    Q_Q10: __m128i,
    Xq_Q14: __m128i,
    Pred_Q15: __m128i,
    Shape_Q14: __m128i,
}

impl Default for NsqDelDecSampleAvx2 {
    fn default() -> Self {
        unsafe {
            Self {
                RandState: _mm_setzero_si128(),
                Q_Q10: _mm_setzero_si128(),
                Xq_Q14: _mm_setzero_si128(),
                Pred_Q15: _mm_setzero_si128(),
                Shape_Q14: _mm_setzero_si128(),
            }
        }
    }
}

/// SoA delayed decision state — each scalar field is now a __m128i holding 4 states.
#[repr(C)]
struct NsqDelDecAvx2 {
    sLPC_Q14: [__m128i; MAX_SUB_FRAME_LENGTH + NSQ_LPC_BUF_LENGTH],
    LF_AR_Q14: __m128i,
    Seed: __m128i,
    SeedInit: __m128i,
    RD_Q10: __m128i,
    Diff_Q14: __m128i,
    sAR2_Q14: [__m128i; MAX_SHAPE_LPC_ORDER as usize],
    Samples: [NsqDelDecSampleAvx2; DECISION_DELAY as usize],
}

impl NsqDelDecAvx2 {
    unsafe fn new_zeroed() -> Self {
        Self {
            sLPC_Q14: [_mm_setzero_si128(); MAX_SUB_FRAME_LENGTH + NSQ_LPC_BUF_LENGTH],
            LF_AR_Q14: _mm_setzero_si128(),
            Seed: _mm_setzero_si128(),
            SeedInit: _mm_setzero_si128(),
            RD_Q10: _mm_setzero_si128(),
            Diff_Q14: _mm_setzero_si128(),
            sAR2_Q14: [_mm_setzero_si128(); MAX_SHAPE_LPC_ORDER as usize],
            Samples: [NsqDelDecSampleAvx2::default(); DECISION_DELAY as usize],
        }
    }
}

/// Inner quantizer for one subframe, AVX2 SoA version.
#[target_feature(enable = "avx2")]
#[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
unsafe fn silk_noise_shape_quantizer_del_dec_avx2(
    NSQ: &mut silk_nsq_state,
    psDelDec: &mut NsqDelDecAvx2,
    signalType: i32,
    x_Q10: &[i32],
    pulses: &mut [i8],
    pulses_off: usize,
    pxq_off: usize,
    sLTP_Q15: &mut [i32],
    delayedGain_Q10: &mut [i32; DECISION_DELAY as usize],
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
    MaskDelDec: __m128i,
    smpl_buf_idx: &mut i32,
    decisionDelay: i32,
) {
    let mut shp_lag_ptr_idx = (NSQ.sLTP_shp_buf_idx - lag + HARM_SHAPE_FIR_TAPS / 2) as usize;
    let mut pred_lag_ptr_idx = (NSQ.sLTP_buf_idx - lag + LTP_ORDER as i32 / 2) as usize;
    let Gain_Q10 = Gain_Q16 >> 6;

    for i in 0..length as usize {
        // Long-term prediction
        let LTP_pred_Q14: i32;
        if signalType == TYPE_VOICED {
            let mut ltp = 2i32;
            // silk_SMULWB = (a * (i16)b) >> 16
            ltp =
                (ltp as i64 + ((sLTP_Q15[pred_lag_ptr_idx] as i64 * b_Q14[0] as i64) >> 16)) as i32;
            ltp = (ltp as i64 + ((sLTP_Q15[pred_lag_ptr_idx - 1] as i64 * b_Q14[1] as i64) >> 16))
                as i32;
            ltp = (ltp as i64 + ((sLTP_Q15[pred_lag_ptr_idx - 2] as i64 * b_Q14[2] as i64) >> 16))
                as i32;
            ltp = (ltp as i64 + ((sLTP_Q15[pred_lag_ptr_idx - 3] as i64 * b_Q14[3] as i64) >> 16))
                as i32;
            ltp = (ltp as i64 + ((sLTP_Q15[pred_lag_ptr_idx - 4] as i64 * b_Q14[4] as i64) >> 16))
                as i32;
            LTP_pred_Q14 = ((ltp as u32) << 1) as i32; // Q13 -> Q14
            pred_lag_ptr_idx += 1;
        } else {
            LTP_pred_Q14 = 0;
        }

        // Long-term shaping
        let n_LTP_Q14: i32;
        if lag > 0 {
            let mut n = silk_add_sat32(
                NSQ.sLTP_shp_Q14[shp_lag_ptr_idx],
                NSQ.sLTP_shp_Q14[shp_lag_ptr_idx - 2],
            );
            // silk_SMULWB
            n = ((n as i64 * HarmShapeFIRPacked_Q14 as i16 as i64) >> 16) as i32;
            // silk_SMULWT
            n = (n as i64
                + ((NSQ.sLTP_shp_Q14[shp_lag_ptr_idx - 1] as i64
                    * (HarmShapeFIRPacked_Q14 as i64 >> 16))
                    >> 16)) as i32;
            n_LTP_Q14 = LTP_pred_Q14 - ((n as u32) << 2) as i32; // Q12 -> Q14
            shp_lag_ptr_idx += 1;
        } else {
            n_LTP_Q14 = 0;
        }

        // Generate dither
        psDelDec.Seed = silk_mm256_rand_epi32(psDelDec.Seed);

        // Short-term prediction
        let LPC_pred_Q14 = silk_noise_shape_quantizer_short_prediction_x4(
            &psDelDec.sLPC_Q14,
            NSQ_LPC_BUF_LENGTH - 1 + i,
            a_Q12,
            predictLPCOrder,
        );
        let LPC_pred_Q14 = _mm_slli_epi32(LPC_pred_Q14, 4); // Q10 -> Q14

        // Noise shape feedback
        debug_assert!(shapingLPCOrder > 0);
        debug_assert!(shapingLPCOrder & 1 == 0);
        // Output of lowpass section
        let mut tmp0 = _mm_add_epi32(
            psDelDec.Diff_Q14,
            silk_mm_smulwb_epi32(psDelDec.sAR2_Q14[0], warping_Q16),
        );
        let mut n_AR_Q14 = _mm_set1_epi32(shapingLPCOrder >> 1);
        for j in 0..shapingLPCOrder as usize - 1 {
            let tmp1 = psDelDec.sAR2_Q14[j];
            psDelDec.sAR2_Q14[j] = tmp0;
            n_AR_Q14 = _mm_add_epi32(n_AR_Q14, silk_mm_smulwb_epi32(tmp0, AR_shp_Q13[j] as i32));
            tmp0 = _mm_add_epi32(
                tmp1,
                silk_mm_smulwb_epi32(_mm_sub_epi32(psDelDec.sAR2_Q14[j + 1], tmp0), warping_Q16),
            );
        }
        psDelDec.sAR2_Q14[shapingLPCOrder as usize - 1] = tmp0;
        n_AR_Q14 = _mm_add_epi32(
            n_AR_Q14,
            silk_mm_smulwb_epi32(tmp0, AR_shp_Q13[shapingLPCOrder as usize - 1] as i32),
        );

        n_AR_Q14 = _mm_slli_epi32(n_AR_Q14, 1); // Q11 -> Q12
        n_AR_Q14 = _mm_add_epi32(n_AR_Q14, silk_mm_smulwb_epi32(psDelDec.LF_AR_Q14, Tilt_Q14)); // Q12
        n_AR_Q14 = _mm_slli_epi32(n_AR_Q14, 2); // Q12 -> Q14

        let tmp0_lf = silk_mm_smulwb_epi32(
            psDelDec.Samples[*smpl_buf_idx as usize].Shape_Q14,
            LF_shp_Q14,
        );
        let tmp1_lf = silk_mm_smulwb_epi32(psDelDec.LF_AR_Q14, LF_shp_Q14 >> 16);
        let n_LF_Q14 = _mm_add_epi32(tmp0_lf, tmp1_lf); // Q12
        let n_LF_Q14 = _mm_slli_epi32(n_LF_Q14, 2); // Q12 -> Q14

        // r = x[i] - LTP_pred - LPC_pred + n_AR + n_Tilt + n_LF + n_LTP
        let tmp0 = silk_mm_add_sat_epi32(n_AR_Q14, n_LF_Q14); // Q14
        let tmp1 = _mm_add_epi32(_mm_set1_epi32(n_LTP_Q14), LPC_pred_Q14); // Q14
        let tmp0 = silk_mm_sub_sat_epi32(tmp1, tmp0); // Q14
        let tmp0 = silk_mm_srai_round_epi32(tmp0, 4); // Q10

        let r_Q10 = _mm_sub_epi32(_mm_set1_epi32(x_Q10[i]), tmp0);

        // Flip sign depending on dither
        let r_Q10 = silk_mm_sign_epi32(r_Q10, psDelDec.Seed);
        let r_Q10 = silk_mm_limit_epi32(r_Q10, -(31 << 10), 30 << 10);

        // Find two quantization level candidates and measure their rate-distortion
        let mut q1_Q10 = _mm_sub_epi32(r_Q10, _mm_set1_epi32(offset_Q10));
        let mut q1_Q0 = _mm_srai_epi32(q1_Q10, 10);
        if Lambda_Q10 > 2048 {
            // For aggressive RDO
            let tmp0 = _mm_sub_epi32(_mm_abs_epi32(q1_Q10), _mm_set1_epi32(Lambda_Q10 / 2 - 512));
            q1_Q0 = _mm_srai_epi32(q1_Q10, 31);
            let tmp1 = _mm_cmpgt_epi32(tmp0, _mm_setzero_si128());
            let tmp0 = _mm_srai_epi32(silk_mm_sign_epi32(tmp0, q1_Q10), 10);
            q1_Q0 = _mm_blendv_epi8(q1_Q0, tmp0, tmp1);
        }

        let tmp0 = _mm_sign_epi32(
            _mm_set1_epi32(crate::silk::define::QUANT_LEVEL_ADJUST_Q10),
            q1_Q0,
        );
        q1_Q10 = _mm_sub_epi32(_mm_slli_epi32(q1_Q0, 10), tmp0);
        q1_Q10 = _mm_add_epi32(q1_Q10, _mm_set1_epi32(offset_Q10));

        // check if q1_Q0 is 0 or -1
        let tmp0 = _mm_add_epi32(_mm_srli_epi32(q1_Q0, 31), q1_Q0);
        let tmp1 = _mm_cmpeq_epi32(tmp0, _mm_setzero_si128());
        let tmp0 = _mm_blendv_epi8(
            _mm_set1_epi32(1024),
            _mm_set1_epi32(1024 - crate::silk::define::QUANT_LEVEL_ADJUST_Q10),
            tmp1,
        );
        let q2_Q10 = _mm_add_epi32(q1_Q10, tmp0);
        let q_Q10 = _mm256_set_m128i(q2_Q10, q1_Q10);

        let rr_Q10 = _mm256_sub_epi32(_mm256_broadcastsi128_si256(r_Q10), q_Q10);
        let mut rd_Q10 = _mm256_abs_epi32(q_Q10);
        let rr_Q10 = silk_mm256_smulbb_epi32(rr_Q10, rr_Q10);
        rd_Q10 = silk_mm256_smulbb_epi32(rd_Q10, _mm256_set1_epi32(Lambda_Q10));
        let mut rd_Q10 = _mm256_add_epi32(rd_Q10, rr_Q10);
        rd_Q10 = _mm256_srai_epi32(rd_Q10, 10);

        let mask = _mm256_broadcastsi128_si256(_mm_cmplt_epi32(
            _mm256_extracti128_si256(rd_Q10, 0),
            _mm256_extracti128_si256(rd_Q10, 1),
        ));
        let mut SS_RD_Q10 = _mm256_add_epi32(
            _mm256_broadcastsi128_si256(psDelDec.RD_Q10),
            _mm256_blendv_epi8(_mm256_permute2x128_si256(rd_Q10, rd_Q10, 0x1), rd_Q10, mask),
        );
        let mut SS_Q_Q10 =
            _mm256_blendv_epi8(_mm256_permute2x128_si256(q_Q10, q_Q10, 0x1), q_Q10, mask);

        // Quantized excitation
        let mut exc_Q14 = silk_mm256_sign_epi32(
            _mm256_slli_epi32(SS_Q_Q10, 4),
            _mm256_broadcastsi128_si256(psDelDec.Seed),
        );

        // Add predictions
        exc_Q14 = _mm256_add_epi32(exc_Q14, _mm256_set1_epi32(LTP_pred_Q14));
        let mut SS_LPC_exc_Q14 = _mm256_slli_epi32(exc_Q14, 1);
        let mut SS_xq_Q14 = _mm256_add_epi32(exc_Q14, _mm256_broadcastsi128_si256(LPC_pred_Q14));

        // Update states
        let mut SS_Diff_Q14 = _mm256_sub_epi32(
            SS_xq_Q14,
            _mm256_set1_epi32(((x_Q10[i] as u32) << 4) as i32),
        );
        let mut SS_LF_AR_Q14 = _mm256_sub_epi32(SS_Diff_Q14, _mm256_broadcastsi128_si256(n_AR_Q14));
        let mut SS_sLTP_shp_Q14 =
            silk_mm256_sub_sat_epi32(SS_LF_AR_Q14, _mm256_broadcastsi128_si256(n_LF_Q14));

        // Update buffer indices
        *smpl_buf_idx = (*smpl_buf_idx + DECISION_DELAY - 1) % DECISION_DELAY;
        let last_smple_idx = (*smpl_buf_idx + decisionDelay) % DECISION_DELAY;

        // Copy last sample fields to avoid borrow conflicts when mutating psDelDec.Samples below
        let last_RandState = psDelDec.Samples[last_smple_idx as usize].RandState;
        let last_Q_Q10 = psDelDec.Samples[last_smple_idx as usize].Q_Q10;
        let last_Xq_Q14 = psDelDec.Samples[last_smple_idx as usize].Xq_Q14;
        let last_Shape_Q14 = psDelDec.Samples[last_smple_idx as usize].Shape_Q14;
        let last_Pred_Q15 = psDelDec.Samples[last_smple_idx as usize].Pred_Q15;

        // Find winner
        let RDmin_Q10 = silk_mm_mask_hmin_epi32(_mm256_castsi256_si128(SS_RD_Q10), MaskDelDec);
        let Winner_selector = silk_index_to_selector(silk_index_of_first_equal_epi32(
            RDmin_Q10,
            _mm256_castsi256_si128(SS_RD_Q10),
        ));

        // Increase RD values of expired states
        let Winner_rand_state = _mm_shuffle_epi8(last_RandState, Winner_selector);

        SS_RD_Q10 = _mm256_blendv_epi8(
            _mm256_add_epi32(SS_RD_Q10, _mm256_set1_epi32(i32::MAX >> 4)),
            SS_RD_Q10,
            _mm256_broadcastsi128_si256(_mm_cmpeq_epi32(last_RandState, Winner_rand_state)),
        );

        // Find worst in first set
        let RDmax_Q10 = silk_mm_mask_hmax_epi32(_mm256_extracti128_si256(SS_RD_Q10, 0), MaskDelDec);
        // Find best in second set
        let RDmin_Q10 = silk_mm_mask_hmin_epi32(_mm256_extracti128_si256(SS_RD_Q10, 1), MaskDelDec);

        // Replace a state if best from second set outperforms worst in first set
        let cmp_tmp = _mm_cmplt_epi32(RDmin_Q10, RDmax_Q10);
        if _mm_test_all_zeros(cmp_tmp, cmp_tmp) == 0 {
            let RDmax_ind =
                silk_index_of_first_equal_epi32(RDmax_Q10, _mm256_extracti128_si256(SS_RD_Q10, 0));
            let RDmin_ind =
                silk_index_of_first_equal_epi32(RDmin_Q10, _mm256_extracti128_si256(SS_RD_Q10, 1));
            let tmp1 = _mm_cvtepi8_epi32(_mm_cvtsi32_si128(
                (0xFFu32 << ((RDmax_ind as u32) << 3)) as i32,
            ));
            let tmp0 = _mm_blendv_epi8(
                _mm_set_epi8(
                    0xF, 0xE, 0xD, 0xC, 0xB, 0xA, 0x9, 0x8, 0x7, 0x6, 0x5, 0x4, 0x3, 0x2, 0x1, 0x0,
                ),
                silk_index_to_selector(RDmin_ind),
                tmp1,
            );
            for t in i..MAX_SUB_FRAME_LENGTH + NSQ_LPC_BUF_LENGTH {
                psDelDec.sLPC_Q14[t] = _mm_shuffle_epi8(psDelDec.sLPC_Q14[t], tmp0);
            }
            psDelDec.Seed = _mm_shuffle_epi8(psDelDec.Seed, tmp0);
            psDelDec.SeedInit = _mm_shuffle_epi8(psDelDec.SeedInit, tmp0);
            for t in 0..MAX_SHAPE_LPC_ORDER as usize {
                psDelDec.sAR2_Q14[t] = _mm_shuffle_epi8(psDelDec.sAR2_Q14[t], tmp0);
            }
            for t in 0..DECISION_DELAY as usize {
                psDelDec.Samples[t].RandState =
                    _mm_shuffle_epi8(psDelDec.Samples[t].RandState, tmp0);
                psDelDec.Samples[t].Q_Q10 = _mm_shuffle_epi8(psDelDec.Samples[t].Q_Q10, tmp0);
                psDelDec.Samples[t].Xq_Q14 = _mm_shuffle_epi8(psDelDec.Samples[t].Xq_Q14, tmp0);
                psDelDec.Samples[t].Pred_Q15 = _mm_shuffle_epi8(psDelDec.Samples[t].Pred_Q15, tmp0);
                psDelDec.Samples[t].Shape_Q14 =
                    _mm_shuffle_epi8(psDelDec.Samples[t].Shape_Q14, tmp0);
            }
            let perm_mask = _mm256_castsi128_si256(_mm_blendv_epi8(
                _mm_set_epi32(0x3, 0x2, 0x1, 0x0),
                _mm_set1_epi32(RDmin_ind + 4),
                tmp1,
            ));
            SS_Q_Q10 = _mm256_permutevar8x32_epi32(SS_Q_Q10, perm_mask);
            SS_RD_Q10 = _mm256_permutevar8x32_epi32(SS_RD_Q10, perm_mask);
            SS_xq_Q14 = _mm256_permutevar8x32_epi32(SS_xq_Q14, perm_mask);
            SS_LF_AR_Q14 = _mm256_permutevar8x32_epi32(SS_LF_AR_Q14, perm_mask);
            SS_Diff_Q14 = _mm256_permutevar8x32_epi32(SS_Diff_Q14, perm_mask);
            SS_sLTP_shp_Q14 = _mm256_permutevar8x32_epi32(SS_sLTP_shp_Q14, perm_mask);
            SS_LPC_exc_Q14 = _mm256_permutevar8x32_epi32(SS_LPC_exc_Q14, perm_mask);
        }

        // Write samples from winner to output and long-term filter states
        if subfr > 0 || i as i32 >= decisionDelay {
            pulses[pulses_off + i - decisionDelay as usize] =
                silk_sar_round_32(silk_select_winner(last_Q_Q10, Winner_selector), 10) as i8;
            NSQ.xq[pxq_off + i - decisionDelay as usize] = silk_sat16(silk_sar_round_smulww(
                silk_select_winner(last_Xq_Q14, Winner_selector),
                delayedGain_Q10[last_smple_idx as usize],
                8,
            ) as i32);
            NSQ.sLTP_shp_Q14[(NSQ.sLTP_shp_buf_idx - decisionDelay) as usize] =
                silk_select_winner(last_Shape_Q14, Winner_selector);
            sLTP_Q15[(NSQ.sLTP_buf_idx - decisionDelay) as usize] =
                silk_select_winner(last_Pred_Q15, Winner_selector);
        }
        NSQ.sLTP_shp_buf_idx += 1;
        NSQ.sLTP_buf_idx += 1;

        // Update states
        let psSample = &mut psDelDec.Samples[*smpl_buf_idx as usize];
        psDelDec.Seed = _mm_add_epi32(
            psDelDec.Seed,
            silk_mm_srai_round_epi32(_mm256_castsi256_si128(SS_Q_Q10), 10),
        );
        psDelDec.LF_AR_Q14 = _mm256_castsi256_si128(SS_LF_AR_Q14);
        psDelDec.Diff_Q14 = _mm256_castsi256_si128(SS_Diff_Q14);
        psDelDec.sLPC_Q14[i + NSQ_LPC_BUF_LENGTH] = _mm256_castsi256_si128(SS_xq_Q14);
        psDelDec.RD_Q10 = _mm256_castsi256_si128(SS_RD_Q10);
        psSample.Xq_Q14 = _mm256_castsi256_si128(SS_xq_Q14);
        psSample.Q_Q10 = _mm256_castsi256_si128(SS_Q_Q10);
        psSample.Pred_Q15 = _mm256_castsi256_si128(SS_LPC_exc_Q14);
        psSample.Shape_Q14 = _mm256_castsi256_si128(SS_sLTP_shp_Q14);
        psSample.RandState = psDelDec.Seed;
        delayedGain_Q10[*smpl_buf_idx as usize] = Gain_Q10;
    }

    // Update LPC states
    for ii in 0..NSQ_LPC_BUF_LENGTH {
        psDelDec.sLPC_Q14[ii] = psDelDec.sLPC_Q14[length as usize + ii];
    }
}

/// Scale states helper, AVX2 SoA version.
#[target_feature(enable = "avx2")]
#[allow(clippy::too_many_arguments)]
unsafe fn silk_nsq_del_dec_scale_states_avx2(
    psEncC: &NsqConfig,
    NSQ: &mut silk_nsq_state,
    psDelDec: &mut NsqDelDecAvx2,
    x16: &[i16],
    x_sc_Q10: &mut [i32],
    sLTP: &[i16],
    sLTP_Q15: &mut [i32],
    subfr: i32,
    LTP_scale_Q14: i32,
    Gains_Q16: &[i32],
    pitchL: &[i32],
    signal_type: i32,
    decisionDelay: i32,
) {
    let lag = pitchL[subfr as usize];
    let mut inv_gain_Q31 = silk_INVERSE32_varQ(Gains_Q16[subfr as usize].max(1), 47);

    // Scale input
    let inv_gain_Q26 = silk_sar_round_32(inv_gain_Q31, 5);
    let mut ii = 0usize;
    while ii + 3 < psEncC.subfr_length {
        let x = _mm256_cvtepi16_epi64(_mm_loadl_epi64(x16.as_ptr().add(ii) as *const __m128i));
        let x = _mm256_slli_epi64(_mm256_mul_epi32(x, _mm256_set1_epi32(inv_gain_Q26)), 16);
        _mm_storeu_si128(
            x_sc_Q10.as_mut_ptr().add(ii) as *mut __m128i,
            silk_cvtepi64_epi32_high(x),
        );
        ii += 4;
    }
    while ii < psEncC.subfr_length {
        x_sc_Q10[ii] = ((x16[ii] as i64 * inv_gain_Q26 as i64) >> 16) as i32;
        ii += 1;
    }

    // After rewhitening the LTP state is un-scaled, so scale with inv_gain_Q16
    if NSQ.rewhite_flag != 0 {
        if subfr == 0 {
            // Do LTP downscaling
            // silk_LSHIFT(silk_SMULWB(inv_gain_Q31, LTP_scale_Q14), 2)
            inv_gain_Q31 = ((((inv_gain_Q31 as i64 * LTP_scale_Q14 as i16 as i64) >> 16) as i32
                as u32)
                << 2) as i32;
        }
        let start = (NSQ.sLTP_buf_idx - lag - LTP_ORDER as i32 / 2) as usize;
        let end = NSQ.sLTP_buf_idx as usize;
        for jj in start..end {
            sLTP_Q15[jj] = ((inv_gain_Q31 as i64 * sLTP[jj] as i64) >> 16) as i32;
        }
    }

    // Adjust for changing gain
    if Gains_Q16[subfr as usize] != NSQ.prev_gain_Q16 {
        let gain_adj_Q16 = silk_DIV32_varQ(NSQ.prev_gain_Q16, Gains_Q16[subfr as usize], 16);

        // Scale long-term shaping state
        let shp_start = (NSQ.sLTP_shp_buf_idx - psEncC.ltp_mem_length as i32) as usize;
        let shp_end = NSQ.sLTP_shp_buf_idx as usize;
        let mut jj = shp_start;
        while jj + 3 < shp_end {
            let p = NSQ.sLTP_shp_Q14.as_mut_ptr().add(jj);
            _mm_storeu_si128(
                p as *mut __m128i,
                silk_mm_smulww_epi32(_mm_loadu_si128(p as *const __m128i), gain_adj_Q16),
            );
            jj += 4;
        }
        while jj < shp_end {
            NSQ.sLTP_shp_Q14[jj] =
                ((NSQ.sLTP_shp_Q14[jj] as i64 * gain_adj_Q16 as i64) >> 16) as i32;
            jj += 1;
        }

        // Scale long-term prediction state
        if signal_type == TYPE_VOICED && NSQ.rewhite_flag == 0 {
            let start = (NSQ.sLTP_buf_idx - lag - LTP_ORDER as i32 / 2) as usize;
            let end = (NSQ.sLTP_buf_idx - decisionDelay) as usize;
            for val in sLTP_Q15[start..end].iter_mut() {
                *val = ((*val as i64 * gain_adj_Q16 as i64) >> 16) as i32;
            }
        }

        // Scale scalar states (SoA vectors)
        psDelDec.LF_AR_Q14 = silk_mm_smulww_epi32(psDelDec.LF_AR_Q14, gain_adj_Q16);
        psDelDec.Diff_Q14 = silk_mm_smulww_epi32(psDelDec.Diff_Q14, gain_adj_Q16);

        // Scale short-term prediction and shaping states
        for jj in 0..NSQ_LPC_BUF_LENGTH {
            psDelDec.sLPC_Q14[jj] = silk_mm_smulww_epi32(psDelDec.sLPC_Q14[jj], gain_adj_Q16);
        }
        for jj in 0..DECISION_DELAY as usize {
            psDelDec.Samples[jj].Pred_Q15 =
                silk_mm_smulww_epi32(psDelDec.Samples[jj].Pred_Q15, gain_adj_Q16);
            psDelDec.Samples[jj].Shape_Q14 =
                silk_mm_smulww_epi32(psDelDec.Samples[jj].Shape_Q14, gain_adj_Q16);
        }
        for jj in 0..MAX_SHAPE_LPC_ORDER as usize {
            psDelDec.sAR2_Q14[jj] = silk_mm_smulww_epi32(psDelDec.sAR2_Q14[jj], gain_adj_Q16);
        }

        // Save inverse gain
        NSQ.prev_gain_Q16 = Gains_Q16[subfr as usize];
    }
}

/// Complete AVX2 NSQ del_dec outer function.
/// Replaces the entire `silk_NSQ_del_dec_c` when nStatesDelayedDecision is 3 or 4 and AVX2 available.
///
/// # Safety
/// Requires AVX2 support (checked by caller via cpufeatures).
#[target_feature(enable = "avx2")]
#[allow(clippy::too_many_arguments)]
pub unsafe fn silk_NSQ_del_dec_avx2(
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
    use crate::silk::define::MAX_LPC_ORDER;
    use crate::silk::tables_other::silk_Quantization_Offsets_Q10;

    let ltp_mem_len = psEncC.ltp_mem_length;
    let frame_len = psEncC.frame_length;
    let subfr_len = psEncC.subfr_length;

    // Build MaskDelDec: lanes beyond nStatesDelayedDecision get masked out
    let MaskDelDec = _mm_cvtepi8_epi32(_mm_cvtsi32_si128(
        (0xFFFFFF00u32 << ((psEncC.nStatesDelayedDecision as u32 - 1) << 3)) as i32,
    ));

    // Set unvoiced lag to the previous one
    let mut lag = NSQ.lagPrev;

    debug_assert!(NSQ.prev_gain_Q16 != 0);

    let mut psDelDec = NsqDelDecAvx2::new_zeroed();
    psDelDec.Seed = _mm_and_si128(
        _mm_add_epi32(
            _mm_set_epi32(3, 2, 1, 0),
            _mm_set1_epi32(psIndices.Seed as i32),
        ),
        _mm_set1_epi32(3),
    );
    psDelDec.SeedInit = psDelDec.Seed;
    psDelDec.RD_Q10 = _mm_setzero_si128();
    psDelDec.LF_AR_Q14 = _mm_set1_epi32(NSQ.sLF_AR_shp_Q14);
    psDelDec.Diff_Q14 = _mm_set1_epi32(NSQ.sDiff_shp_Q14);
    psDelDec.Samples[0].Shape_Q14 = _mm_set1_epi32(NSQ.sLTP_shp_Q14[ltp_mem_len - 1]);
    for ii in 0..NSQ_LPC_BUF_LENGTH {
        psDelDec.sLPC_Q14[ii] = _mm_set1_epi32(NSQ.sLPC_Q14[ii]);
    }
    for ii in 0..MAX_SHAPE_LPC_ORDER as usize {
        psDelDec.sAR2_Q14[ii] = _mm_set1_epi32(NSQ.sAR2_Q14[ii]);
    }

    let offset_Q10 = silk_Quantization_Offsets_Q10[(psIndices.signalType as i32 >> 1) as usize]
        [psIndices.quantOffsetType as usize] as i32;
    let mut smpl_buf_idx: i32 = 0;

    let mut decisionDelay =
        crate::silk::SigProc_FIX::silk_min_int(DECISION_DELAY, subfr_len as i32);

    // For voiced frames limit the decision delay
    if psIndices.signalType as i32 == TYPE_VOICED {
        for &pl in pitchL.iter().take(psEncC.nb_subfr) {
            decisionDelay = crate::silk::SigProc_FIX::silk_min_int(
                decisionDelay,
                pl - LTP_ORDER as i32 / 2 - 1,
            );
        }
    } else if lag > 0 {
        decisionDelay =
            crate::silk::SigProc_FIX::silk_min_int(decisionDelay, lag - LTP_ORDER as i32 / 2 - 1);
    }

    let LSF_interpolation_flag: i32 = if psIndices.NLSFInterpCoef_Q2 as i32 == 4 {
        0
    } else {
        1
    };

    let mut sLTP_Q15: Vec<i32> = vec![0; ltp_mem_len + frame_len];
    let mut sLTP: Vec<i16> = vec![0; ltp_mem_len + frame_len];
    let mut x_sc_Q10 = [0i32; MAX_SUB_FRAME_LENGTH];
    let mut delayedGain_Q10 = [0i32; DECISION_DELAY as usize];

    let mut pxq_off: usize = ltp_mem_len;
    NSQ.sLTP_shp_buf_idx = ltp_mem_len as i32;
    NSQ.sLTP_buf_idx = ltp_mem_len as i32;
    let mut subfr: i32 = 0;
    let mut x16_off: usize = 0;
    let mut pulses_off: usize = 0;

    for k in 0..psEncC.nb_subfr {
        let a_Q12_off = ((k >> 1) | ((1 - LSF_interpolation_flag) as usize)) * MAX_LPC_ORDER;
        let a_Q12 = &PredCoef_Q12[a_Q12_off..a_Q12_off + psEncC.predictLPCOrder as usize];
        let b_Q14_off = k * LTP_ORDER;
        let b_Q14 = &LTPCoef_Q14[b_Q14_off..b_Q14_off + LTP_ORDER];
        let ar_shp_off = k * MAX_SHAPE_LPC_ORDER as usize;
        let AR_shp_Q13 = &AR_Q13[ar_shp_off..ar_shp_off + psEncC.shapingLPCOrder as usize];

        // Noise shape parameters
        debug_assert!(HarmShapeGain_Q14[k] >= 0);
        let mut HarmShapeFIRPacked_Q14: i32 = HarmShapeGain_Q14[k] >> 2;
        HarmShapeFIRPacked_Q14 |= (((HarmShapeGain_Q14[k] >> 1) as u32) << 16) as i32;

        NSQ.rewhite_flag = 0;
        if psIndices.signalType as i32 == TYPE_VOICED {
            lag = pitchL[k];

            // Re-whitening
            if (k as i32) & (3 ^ (LSF_interpolation_flag << 1)) == 0 {
                if k == 2 {
                    // RESET DELAYED DECISIONS
                    let RDmin_Q10 = silk_mm_mask_hmin_epi32(psDelDec.RD_Q10, MaskDelDec);
                    let Winner_ind = silk_index_of_first_equal_epi32(RDmin_Q10, psDelDec.RD_Q10);
                    let Winner_selector = silk_index_to_selector(Winner_ind);
                    psDelDec.RD_Q10 = _mm_add_epi32(
                        psDelDec.RD_Q10,
                        _mm_blendv_epi8(
                            _mm_set1_epi32(i32::MAX >> 4),
                            _mm_setzero_si128(),
                            _mm_cvtepi8_epi32(_mm_cvtsi32_si128(
                                (0xFFu32 << ((Winner_ind as u32) << 3)) as i32,
                            )),
                        ),
                    );

                    // Copy final part of signals from winner state to output
                    let mut last_smple_idx = smpl_buf_idx + decisionDelay;
                    for ii in 0..decisionDelay {
                        last_smple_idx = (last_smple_idx + DECISION_DELAY - 1) % DECISION_DELAY;
                        let psSample = &psDelDec.Samples[last_smple_idx as usize];
                        pulses[(pulses_off as isize + (ii - decisionDelay) as isize) as usize] =
                            silk_sar_round_32(
                                silk_select_winner(psSample.Q_Q10, Winner_selector),
                                10,
                            ) as i8;
                        NSQ.xq[(pxq_off as isize + (ii - decisionDelay) as isize) as usize] =
                            silk_sat16(silk_sar_round_smulww(
                                silk_select_winner(psSample.Xq_Q14, Winner_selector),
                                Gains_Q16[1],
                                14,
                            ) as i32);
                        NSQ.sLTP_shp_Q14[(NSQ.sLTP_shp_buf_idx - decisionDelay + ii) as usize] =
                            silk_select_winner(psSample.Shape_Q14, Winner_selector);
                    }

                    subfr = 0;
                }

                // Rewhiten with new A coefs
                let start_idx =
                    ltp_mem_len as i32 - lag - psEncC.predictLPCOrder - LTP_ORDER as i32 / 2;
                debug_assert!(start_idx > 0);

                silk_LPC_analysis_filter_avx2(
                    &mut sLTP[start_idx as usize..],
                    &NSQ.xq[(start_idx + k as i32 * subfr_len as i32) as usize..],
                    a_Q12,
                    ltp_mem_len as i32 - start_idx,
                    psEncC.predictLPCOrder,
                );

                NSQ.sLTP_buf_idx = ltp_mem_len as i32;
                NSQ.rewhite_flag = 1;
            }
        }

        silk_nsq_del_dec_scale_states_avx2(
            psEncC,
            NSQ,
            &mut psDelDec,
            &x16[x16_off..x16_off + subfr_len],
            &mut x_sc_Q10[..subfr_len],
            &sLTP,
            &mut sLTP_Q15,
            k as i32,
            LTP_scale_Q14,
            Gains_Q16,
            pitchL,
            psIndices.signalType as i32,
            decisionDelay,
        );

        let fresh_subfr = subfr;
        subfr += 1;

        silk_noise_shape_quantizer_del_dec_avx2(
            NSQ,
            &mut psDelDec,
            psIndices.signalType as i32,
            &x_sc_Q10[..subfr_len],
            pulses,
            pulses_off,
            pxq_off,
            &mut sLTP_Q15,
            &mut delayedGain_Q10,
            a_Q12,
            b_Q14,
            AR_shp_Q13,
            lag,
            HarmShapeFIRPacked_Q14,
            Tilt_Q14[k],
            LF_shp_Q14[k],
            Gains_Q16[k],
            Lambda_Q10,
            offset_Q10,
            subfr_len as i32,
            fresh_subfr,
            psEncC.shapingLPCOrder,
            psEncC.predictLPCOrder,
            psEncC.warping_Q16,
            MaskDelDec,
            &mut smpl_buf_idx,
            decisionDelay,
        );

        x16_off += subfr_len;
        pulses_off += subfr_len;
        pxq_off += subfr_len;
    }

    // Find winner
    let RDmin_Q10 = silk_mm_mask_hmin_epi32(psDelDec.RD_Q10, MaskDelDec);
    let Winner_selector =
        silk_index_to_selector(silk_index_of_first_equal_epi32(RDmin_Q10, psDelDec.RD_Q10));

    // Copy final part of signals from winner state to output
    psIndices.Seed = silk_select_winner(psDelDec.SeedInit, Winner_selector) as i8;
    let mut last_smple_idx = smpl_buf_idx + decisionDelay;
    let Gain_Q10 = Gains_Q16[psEncC.nb_subfr - 1] >> 6;
    for ii in 0..decisionDelay {
        last_smple_idx = (last_smple_idx + DECISION_DELAY - 1) % DECISION_DELAY;
        let psSample = &psDelDec.Samples[last_smple_idx as usize];

        pulses[(pulses_off as isize + (ii - decisionDelay) as isize) as usize] =
            silk_sar_round_32(silk_select_winner(psSample.Q_Q10, Winner_selector), 10) as i8;
        NSQ.xq[(pxq_off as isize + (ii - decisionDelay) as isize) as usize] =
            silk_sat16(silk_sar_round_smulww(
                silk_select_winner(psSample.Xq_Q14, Winner_selector),
                Gain_Q10,
                8,
            ) as i32);
        NSQ.sLTP_shp_Q14[(NSQ.sLTP_shp_buf_idx - decisionDelay + ii) as usize] =
            silk_select_winner(psSample.Shape_Q14, Winner_selector);
    }
    for ii in 0..NSQ_LPC_BUF_LENGTH {
        NSQ.sLPC_Q14[ii] = silk_select_winner(psDelDec.sLPC_Q14[ii], Winner_selector);
    }
    for ii in 0..MAX_SHAPE_LPC_ORDER as usize {
        NSQ.sAR2_Q14[ii] = silk_select_winner(psDelDec.sAR2_Q14[ii], Winner_selector);
    }

    // Update states
    NSQ.sLF_AR_shp_Q14 = silk_select_winner(psDelDec.LF_AR_Q14, Winner_selector);
    NSQ.sDiff_shp_Q14 = silk_select_winner(psDelDec.Diff_Q14, Winner_selector);
    NSQ.lagPrev = pitchL[psEncC.nb_subfr - 1];

    // Save quantized speech signal
    NSQ.xq.copy_within(frame_len..frame_len + ltp_mem_len, 0);
    NSQ.sLTP_shp_Q14
        .copy_within(frame_len..frame_len + ltp_mem_len, 0);
}
