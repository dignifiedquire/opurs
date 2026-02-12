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
use crate::silk::structs::{silk_nsq_state, NsqConfig};
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
    let mut sDiff_shp_Q14: i32 = NSQ.sDiff_shp_Q14;
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
        if (r_Q10 as i64 * table[tidx][2] as i64 - table[tidx][3] as i64) < 0 {
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

        sDiff_shp_Q14 = xq_Q14 - ((x_sc_Q10[i] as u32) << 4) as i32;
        NSQ.sDiff_shp_Q14 = sDiff_shp_Q14;
        sLF_AR_shp_Q14 = sDiff_shp_Q14.wrapping_sub(((n_AR_Q12 as u32) << 2) as i32);
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
