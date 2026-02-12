//! x86/x86_64 SIMD implementations for SILK functions.
//!
//! SSE4.1 and AVX2 intrinsics for noise shaping, inner products, etc.
//! All functions require `#[target_feature]` and are called only after cpufeatures detection.

#![allow(non_camel_case_types)]

#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

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
