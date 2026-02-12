//! x86/x86_64 SIMD implementations for CELT functions.
//!
//! SSE, SSE2, SSE4.1, and AVX2 intrinsics for pitch analysis and related functions.
//! All functions require `#[target_feature]` and are called only after cpufeatures detection.

#![allow(non_camel_case_types)]

#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

/// SSE implementation of `xcorr_kernel`.
/// Port of `celt/x86/pitch_sse.c:xcorr_kernel_sse`.
///
/// # Safety
/// Requires SSE support (checked by caller via cpufeatures).
#[target_feature(enable = "sse")]
pub unsafe fn xcorr_kernel_sse(x: &[f32], y: &[f32], sum: &mut [f32; 4], len: usize) {
    debug_assert!(len >= 3);
    debug_assert!(x.len() >= len);
    debug_assert!(y.len() >= len + 3);

    let mut xsum1 = _mm_loadu_ps(sum.as_ptr());
    let mut xsum2 = _mm_setzero_ps();

    let mut j = 0usize;
    while j + 3 < len {
        let x0 = _mm_loadu_ps(x.as_ptr().add(j));
        let yj = _mm_loadu_ps(y.as_ptr().add(j));
        let y3 = _mm_loadu_ps(y.as_ptr().add(j + 3));

        // sum[0..4] += x[j+0] * y[j+0..j+3]
        xsum1 = _mm_add_ps(xsum1, _mm_mul_ps(_mm_shuffle_ps(x0, x0, 0x00), yj));
        // sum[0..4] += x[j+1] * y[j+1..j+4]
        xsum2 = _mm_add_ps(
            xsum2,
            _mm_mul_ps(_mm_shuffle_ps(x0, x0, 0x55), _mm_shuffle_ps(yj, y3, 0x49)),
        );
        // sum[0..4] += x[j+2] * y[j+2..j+5]
        xsum1 = _mm_add_ps(
            xsum1,
            _mm_mul_ps(_mm_shuffle_ps(x0, x0, 0xaa), _mm_shuffle_ps(yj, y3, 0x9e)),
        );
        // sum[0..4] += x[j+3] * y[j+3..j+6]
        xsum2 = _mm_add_ps(xsum2, _mm_mul_ps(_mm_shuffle_ps(x0, x0, 0xff), y3));

        j += 4;
    }

    xsum1 = _mm_add_ps(xsum1, xsum2);

    // Handle remaining 1-3 elements
    while j < len {
        let xj = _mm_set1_ps(*x.get_unchecked(j));
        let yj = _mm_loadu_ps(y.as_ptr().add(j));
        xsum1 = _mm_add_ps(xsum1, _mm_mul_ps(xj, yj));
        j += 1;
    }

    _mm_storeu_ps(sum.as_mut_ptr(), xsum1);

    // SIMD correctness is verified via unit tests that compare
    // SIMD output against scalar output for identical inputs.
}

/// AVX2 implementation of 8-wide `xcorr_kernel`.
/// Port of `celt/x86/pitch_avx.c:xcorr_kernel_avx`.
/// Computes 8 cross-correlation results simultaneously using 256-bit vectors.
/// Uses mul+add (no FMA) to maintain bit-exactness with scalar path.
///
/// # Safety
/// Requires AVX2 support (checked by caller via cpufeatures).
#[target_feature(enable = "avx2")]
unsafe fn xcorr_kernel_avx2(x: &[f32], y: &[f32], sum: &mut [f32; 8], len: usize) {
    debug_assert!(x.len() >= len);
    debug_assert!(y.len() >= len + 7);

    let mut xsum0 = _mm256_setzero_ps();
    let mut xsum1 = _mm256_setzero_ps();
    let mut xsum2 = _mm256_setzero_ps();
    let mut xsum3 = _mm256_setzero_ps();
    let mut xsum4 = _mm256_setzero_ps();
    let mut xsum5 = _mm256_setzero_ps();
    let mut xsum6 = _mm256_setzero_ps();
    let mut xsum7 = _mm256_setzero_ps();

    let mut i = 0usize;
    while i + 7 < len {
        let x0 = _mm256_loadu_ps(x.as_ptr().add(i));
        xsum0 = _mm256_add_ps(xsum0, _mm256_mul_ps(x0, _mm256_loadu_ps(y.as_ptr().add(i))));
        xsum1 = _mm256_add_ps(
            xsum1,
            _mm256_mul_ps(x0, _mm256_loadu_ps(y.as_ptr().add(i + 1))),
        );
        xsum2 = _mm256_add_ps(
            xsum2,
            _mm256_mul_ps(x0, _mm256_loadu_ps(y.as_ptr().add(i + 2))),
        );
        xsum3 = _mm256_add_ps(
            xsum3,
            _mm256_mul_ps(x0, _mm256_loadu_ps(y.as_ptr().add(i + 3))),
        );
        xsum4 = _mm256_add_ps(
            xsum4,
            _mm256_mul_ps(x0, _mm256_loadu_ps(y.as_ptr().add(i + 4))),
        );
        xsum5 = _mm256_add_ps(
            xsum5,
            _mm256_mul_ps(x0, _mm256_loadu_ps(y.as_ptr().add(i + 5))),
        );
        xsum6 = _mm256_add_ps(
            xsum6,
            _mm256_mul_ps(x0, _mm256_loadu_ps(y.as_ptr().add(i + 6))),
        );
        xsum7 = _mm256_add_ps(
            xsum7,
            _mm256_mul_ps(x0, _mm256_loadu_ps(y.as_ptr().add(i + 7))),
        );
        i += 8;
    }

    // Handle remaining 1-7 elements with masked loads
    if i < len {
        // Create mask: -1 for valid elements, 0 for padding
        static MASK_TABLE: [i32; 15] = [-1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0];
        let remaining = len - i;
        let m = _mm256_loadu_si256(MASK_TABLE.as_ptr().add(7 - remaining) as *const __m256i);
        let x0 = _mm256_maskload_ps(x.as_ptr().add(i), m);
        xsum0 = _mm256_add_ps(
            xsum0,
            _mm256_mul_ps(x0, _mm256_maskload_ps(y.as_ptr().add(i), m)),
        );
        xsum1 = _mm256_add_ps(
            xsum1,
            _mm256_mul_ps(x0, _mm256_maskload_ps(y.as_ptr().add(i + 1), m)),
        );
        xsum2 = _mm256_add_ps(
            xsum2,
            _mm256_mul_ps(x0, _mm256_maskload_ps(y.as_ptr().add(i + 2), m)),
        );
        xsum3 = _mm256_add_ps(
            xsum3,
            _mm256_mul_ps(x0, _mm256_maskload_ps(y.as_ptr().add(i + 3), m)),
        );
        xsum4 = _mm256_add_ps(
            xsum4,
            _mm256_mul_ps(x0, _mm256_maskload_ps(y.as_ptr().add(i + 4), m)),
        );
        xsum5 = _mm256_add_ps(
            xsum5,
            _mm256_mul_ps(x0, _mm256_maskload_ps(y.as_ptr().add(i + 5), m)),
        );
        xsum6 = _mm256_add_ps(
            xsum6,
            _mm256_mul_ps(x0, _mm256_maskload_ps(y.as_ptr().add(i + 6), m)),
        );
        xsum7 = _mm256_add_ps(
            xsum7,
            _mm256_mul_ps(x0, _mm256_maskload_ps(y.as_ptr().add(i + 7), m)),
        );
    }

    // 8 horizontal sums
    // Compute [0 4] [1 5] [2 6] [3 7] by combining 128-bit halves
    xsum0 = _mm256_add_ps(
        _mm256_permute2f128_ps(xsum0, xsum4, 2 << 4),
        _mm256_permute2f128_ps(xsum0, xsum4, 1 | (3 << 4)),
    );
    xsum1 = _mm256_add_ps(
        _mm256_permute2f128_ps(xsum1, xsum5, 2 << 4),
        _mm256_permute2f128_ps(xsum1, xsum5, 1 | (3 << 4)),
    );
    xsum2 = _mm256_add_ps(
        _mm256_permute2f128_ps(xsum2, xsum6, 2 << 4),
        _mm256_permute2f128_ps(xsum2, xsum6, 1 | (3 << 4)),
    );
    xsum3 = _mm256_add_ps(
        _mm256_permute2f128_ps(xsum3, xsum7, 2 << 4),
        _mm256_permute2f128_ps(xsum3, xsum7, 1 | (3 << 4)),
    );
    // Compute [0 1 4 5] [2 3 6 7]
    xsum0 = _mm256_hadd_ps(xsum0, xsum1);
    xsum1 = _mm256_hadd_ps(xsum2, xsum3);
    // Compute [0 1 2 3 4 5 6 7]
    xsum0 = _mm256_hadd_ps(xsum0, xsum1);
    _mm256_storeu_ps(sum.as_mut_ptr(), xsum0);
}

/// AVX2 implementation of `celt_pitch_xcorr`.
/// Processes 8 correlations at a time using `xcorr_kernel_avx2`.
///
/// # Safety
/// Requires AVX2 support (checked by caller via cpufeatures).
#[target_feature(enable = "avx2")]
pub unsafe fn celt_pitch_xcorr_avx2(x: &[f32], y: &[f32], xcorr: &mut [f32], len: usize) {
    let max_pitch = xcorr.len();
    debug_assert!(max_pitch > 0);
    debug_assert!(x.len() >= len);

    let mut i = 0usize;
    while i + 7 < max_pitch {
        let mut sum = [0.0f32; 8];
        xcorr_kernel_avx2(&x[..len], &y[i..], &mut sum, len);
        xcorr[i] = sum[0];
        xcorr[i + 1] = sum[1];
        xcorr[i + 2] = sum[2];
        xcorr[i + 3] = sum[3];
        xcorr[i + 4] = sum[4];
        xcorr[i + 5] = sum[5];
        xcorr[i + 6] = sum[6];
        xcorr[i + 7] = sum[7];
        i += 8;
    }
    // Handle remaining with SSE (4 at a time) or scalar
    while i + 3 < max_pitch {
        let mut sum = [0.0f32; 4];
        xcorr_kernel_sse(&x[..len], &y[i..], &mut sum, len);
        xcorr[i] = sum[0];
        xcorr[i + 1] = sum[1];
        xcorr[i + 2] = sum[2];
        xcorr[i + 3] = sum[3];
        i += 4;
    }
    while i < max_pitch {
        xcorr[i] = celt_inner_prod_sse(x, &y[i..], len);
        i += 1;
    }
}

/// SSE implementation of `celt_inner_prod`.
/// Port of `celt/x86/pitch_sse.c:celt_inner_prod_sse`.
///
/// # Safety
/// Requires SSE support (checked by caller via cpufeatures).
#[target_feature(enable = "sse")]
pub unsafe fn celt_inner_prod_sse(x: &[f32], y: &[f32], n: usize) -> f32 {
    debug_assert!(x.len() >= n);
    debug_assert!(y.len() >= n);

    let mut sum = _mm_setzero_ps();
    let mut i = 0usize;

    // Process 4 floats at a time
    while i + 3 < n {
        let xv = _mm_loadu_ps(x.as_ptr().add(i));
        let yv = _mm_loadu_ps(y.as_ptr().add(i));
        sum = _mm_add_ps(sum, _mm_mul_ps(xv, yv));
        i += 4;
    }

    // Horizontal sum: sum all 4 lanes
    let hi = _mm_movehl_ps(sum, sum);
    sum = _mm_add_ps(sum, hi);
    let hi2 = _mm_shuffle_ps(sum, sum, 0x55);
    sum = _mm_add_ss(sum, hi2);

    let mut result: f32 = 0.0;
    _mm_store_ss(&mut result, sum);

    // Handle remaining elements
    while i < n {
        result += *x.get_unchecked(i) * *y.get_unchecked(i);
        i += 1;
    }

    result
}

/// SSE implementation of `dual_inner_prod`.
/// Port of `celt/x86/pitch_sse.c:dual_inner_prod_sse`.
///
/// # Safety
/// Requires SSE support (checked by caller via cpufeatures).
#[target_feature(enable = "sse")]
pub unsafe fn dual_inner_prod_sse(x: &[f32], y01: &[f32], y02: &[f32], n: usize) -> (f32, f32) {
    debug_assert!(x.len() >= n);
    debug_assert!(y01.len() >= n);
    debug_assert!(y02.len() >= n);

    let mut sum1 = _mm_setzero_ps();
    let mut sum2 = _mm_setzero_ps();
    let mut i = 0usize;

    while i + 3 < n {
        let xv = _mm_loadu_ps(x.as_ptr().add(i));
        let y1v = _mm_loadu_ps(y01.as_ptr().add(i));
        let y2v = _mm_loadu_ps(y02.as_ptr().add(i));
        sum1 = _mm_add_ps(sum1, _mm_mul_ps(xv, y1v));
        sum2 = _mm_add_ps(sum2, _mm_mul_ps(xv, y2v));
        i += 4;
    }

    // Horizontal sum for sum1
    let hi1 = _mm_movehl_ps(sum1, sum1);
    sum1 = _mm_add_ps(sum1, hi1);
    let hi1b = _mm_shuffle_ps(sum1, sum1, 0x55);
    sum1 = _mm_add_ss(sum1, hi1b);
    let mut xy01: f32 = 0.0;
    _mm_store_ss(&mut xy01, sum1);

    // Horizontal sum for sum2
    let hi2 = _mm_movehl_ps(sum2, sum2);
    sum2 = _mm_add_ps(sum2, hi2);
    let hi2b = _mm_shuffle_ps(sum2, sum2, 0x55);
    sum2 = _mm_add_ss(sum2, hi2b);
    let mut xy02: f32 = 0.0;
    _mm_store_ss(&mut xy02, sum2);

    // Handle remaining elements
    while i < n {
        let xi = *x.get_unchecked(i);
        xy01 += xi * *y01.get_unchecked(i);
        xy02 += xi * *y02.get_unchecked(i);
        i += 1;
    }

    (xy01, xy02)
}

/// SSE implementation of `celt_pitch_xcorr`.
/// Processes 4 correlations at a time using `xcorr_kernel_sse`.
///
/// # Safety
/// Requires SSE support (checked by caller via cpufeatures).
#[target_feature(enable = "sse")]
pub unsafe fn celt_pitch_xcorr_sse(x: &[f32], y: &[f32], xcorr: &mut [f32], len: usize) {
    let max_pitch = xcorr.len();
    debug_assert!(max_pitch > 0);
    debug_assert!(x.len() >= len);

    let mut i = 0i32;
    while i < max_pitch as i32 - 3 {
        let mut sum = [0.0f32; 4];
        xcorr_kernel_sse(&x[..len], &y[i as usize..], &mut sum, len);
        xcorr[i as usize] = sum[0];
        xcorr[i as usize + 1] = sum[1];
        xcorr[i as usize + 2] = sum[2];
        xcorr[i as usize + 3] = sum[3];
        i += 4;
    }
    while (i as usize) < max_pitch {
        xcorr[i as usize] = celt_inner_prod_sse(x, &y[i as usize..], len);
        i += 1;
    }
}
