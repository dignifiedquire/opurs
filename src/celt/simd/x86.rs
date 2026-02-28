//! x86/x86_64 SIMD implementations for CELT functions.
//!
//! SSE, SSE2, SSE4.1, and AVX2 intrinsics for pitch analysis and related functions.
//! All functions require `#[target_feature]` and are called only after cpufeatures detection.

#![allow(non_camel_case_types)]
#![allow(dead_code)]

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

    // Handle remaining 1-3 elements.
    // Must match C accumulation order exactly: alternate xsum1/xsum2
    // for each remaining element, then combine at the end.
    if j < len {
        xsum1 = _mm_add_ps(
            xsum1,
            _mm_mul_ps(
                _mm_load1_ps(x.as_ptr().add(j)),
                _mm_loadu_ps(y.as_ptr().add(j)),
            ),
        );
        j += 1;
        if j < len {
            xsum2 = _mm_add_ps(
                xsum2,
                _mm_mul_ps(
                    _mm_load1_ps(x.as_ptr().add(j)),
                    _mm_loadu_ps(y.as_ptr().add(j)),
                ),
            );
            j += 1;
            if j < len {
                xsum1 = _mm_add_ps(
                    xsum1,
                    _mm_mul_ps(
                        _mm_load1_ps(x.as_ptr().add(j)),
                        _mm_loadu_ps(y.as_ptr().add(j)),
                    ),
                );
            }
        }
    }

    _mm_storeu_ps(sum.as_mut_ptr(), _mm_add_ps(xsum1, xsum2));
}

/// AVX2+FMA implementation of 8-wide `xcorr_kernel`.
/// Port of `celt/x86/pitch_avx.c:xcorr_kernel_avx`.
/// Computes 8 cross-correlation results simultaneously using 256-bit vectors.
/// Uses FMA (`_mm256_fmadd_ps`) to match C reference exactly.
///
/// # Safety
/// Requires AVX2+FMA support (checked by caller via cpufeatures).
#[target_feature(enable = "avx2", enable = "fma")]
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
        xsum0 = _mm256_fmadd_ps(x0, _mm256_loadu_ps(y.as_ptr().add(i)), xsum0);
        xsum1 = _mm256_fmadd_ps(x0, _mm256_loadu_ps(y.as_ptr().add(i + 1)), xsum1);
        xsum2 = _mm256_fmadd_ps(x0, _mm256_loadu_ps(y.as_ptr().add(i + 2)), xsum2);
        xsum3 = _mm256_fmadd_ps(x0, _mm256_loadu_ps(y.as_ptr().add(i + 3)), xsum3);
        xsum4 = _mm256_fmadd_ps(x0, _mm256_loadu_ps(y.as_ptr().add(i + 4)), xsum4);
        xsum5 = _mm256_fmadd_ps(x0, _mm256_loadu_ps(y.as_ptr().add(i + 5)), xsum5);
        xsum6 = _mm256_fmadd_ps(x0, _mm256_loadu_ps(y.as_ptr().add(i + 6)), xsum6);
        xsum7 = _mm256_fmadd_ps(x0, _mm256_loadu_ps(y.as_ptr().add(i + 7)), xsum7);
        i += 8;
    }

    // Handle remaining 1-7 elements with masked loads.
    // This matches upstream `celt/x86/pitch_avx.c` exactly.
    if i < len {
        static MASK_TABLE: [i32; 15] = [-1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0];
        let remaining = len - i;
        let m = _mm256_loadu_si256(MASK_TABLE.as_ptr().add(7 - remaining) as *const __m256i);
        let x0 = _mm256_maskload_ps(x.as_ptr().add(i), m);
        xsum0 = _mm256_fmadd_ps(x0, _mm256_maskload_ps(y.as_ptr().add(i), m), xsum0);
        xsum1 = _mm256_fmadd_ps(x0, _mm256_maskload_ps(y.as_ptr().add(i + 1), m), xsum1);
        xsum2 = _mm256_fmadd_ps(x0, _mm256_maskload_ps(y.as_ptr().add(i + 2), m), xsum2);
        xsum3 = _mm256_fmadd_ps(x0, _mm256_maskload_ps(y.as_ptr().add(i + 3), m), xsum3);
        xsum4 = _mm256_fmadd_ps(x0, _mm256_maskload_ps(y.as_ptr().add(i + 4), m), xsum4);
        xsum5 = _mm256_fmadd_ps(x0, _mm256_maskload_ps(y.as_ptr().add(i + 5), m), xsum5);
        xsum6 = _mm256_fmadd_ps(x0, _mm256_maskload_ps(y.as_ptr().add(i + 6), m), xsum6);
        xsum7 = _mm256_fmadd_ps(x0, _mm256_maskload_ps(y.as_ptr().add(i + 7), m), xsum7);
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
/// Tail uses scalar `celt_inner_prod` (dispatched to SSE via RTCD), matching C.
///
/// # Safety
/// Requires AVX2+FMA support (checked by caller via cpufeatures).
#[target_feature(enable = "avx2", enable = "fma")]
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
    // Handle remaining 1-7 with SSE celt_inner_prod, matching C's
    // `celt_inner_prod(_x, _y+i, len, arch)` which dispatches to SSE.
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

/// SSE2 implementation of `op_pvq_search`.
/// Port of `celt/x86/vq_sse2.c:op_pvq_search_sse2`.
///
/// Uses `_mm_rsqrt_ps` for approximate reciprocal sqrt in the greedy pulse search,
/// which may produce slightly different results from scalar (this matches C behavior).
///
/// # Safety
/// Requires SSE2 support (checked by caller via cpufeatures).
#[target_feature(enable = "sse2")]
pub unsafe fn op_pvq_search_sse2(_X: &mut [f32], iy: &mut [i32], K: i32, N: i32) -> f32 {
    let n = N as usize;
    // Pad to N+3 for safe SIMD overread + sentinel values.
    let mut X = vec![0.0f32; n + 3];
    let mut y = vec![0.0f32; n + 3];
    let mut signy = vec![0.0f32; n + 3];

    X[..n].copy_from_slice(&_X[..n]);
    X[n] = 0.0;
    X[n + 1] = 0.0;
    X[n + 2] = 0.0;

    let signmask = _mm_set_ps1(-0.0f32);
    let fours = _mm_set_epi32(4, 4, 4, 4);

    // Initialize: compute |X|, save signs, clear y and iy
    let mut sums = _mm_setzero_ps();
    let mut j = 0usize;
    while j < n {
        let x4 = _mm_loadu_ps(X.as_ptr().add(j));
        let s4 = _mm_cmplt_ps(x4, _mm_setzero_ps());
        // Get rid of the sign
        let x4 = _mm_andnot_ps(signmask, x4);
        sums = _mm_add_ps(sums, x4);
        _mm_storeu_ps(y.as_mut_ptr().add(j), _mm_setzero_ps());
        _mm_storeu_si128(iy.as_mut_ptr().add(j) as *mut __m128i, _mm_setzero_si128());
        _mm_storeu_ps(X.as_mut_ptr().add(j), x4);
        _mm_storeu_ps(signy.as_mut_ptr().add(j), s4);
        j += 4;
    }

    // Horizontal sum of sums
    sums = _mm_add_ps(sums, _mm_shuffle_ps(sums, sums, 0x4E));
    sums = _mm_add_ps(sums, _mm_shuffle_ps(sums, sums, 0xB1));

    let mut xy: f32 = 0.0;
    let mut yy: f32 = 0.0;
    let mut pulsesLeft = K;

    // Pre-search by projecting on the pyramid
    if K > (N >> 1) {
        let mut sum = _mm_cvtss_f32(sums);
        let epsilon = 1e-15f32;
        if !(sum > epsilon && sum < 64.0) {
            X[0] = 1.0;
            for xj in X[1..n].iter_mut() {
                *xj = 0.0;
            }
            sums = _mm_set_ps1(1.0);
            sum = 1.0;
            let _ = sum;
        }
        let rcp4 = _mm_mul_ps(_mm_set_ps1((K as f32) + 0.8), _mm_rcp_ps(sums));
        let mut xy4 = _mm_setzero_ps();
        let mut yy4 = _mm_setzero_ps();
        let mut pulses_sum = _mm_setzero_si128();

        j = 0;
        while j < n {
            let x4 = _mm_loadu_ps(X.as_ptr().add(j));
            let rx4 = _mm_mul_ps(x4, rcp4);
            let iy4 = _mm_cvttps_epi32(rx4);
            pulses_sum = _mm_add_epi32(pulses_sum, iy4);
            _mm_storeu_si128(iy.as_mut_ptr().add(j) as *mut __m128i, iy4);
            let y4 = _mm_cvtepi32_ps(iy4);
            xy4 = _mm_add_ps(xy4, _mm_mul_ps(x4, y4));
            yy4 = _mm_add_ps(yy4, _mm_mul_ps(y4, y4));
            // Double y[] so we don't have to do it in the search loop
            _mm_storeu_ps(y.as_mut_ptr().add(j), _mm_add_ps(y4, y4));
            j += 4;
        }

        // Horizontal sum of pulses
        pulses_sum = _mm_add_epi32(pulses_sum, _mm_shuffle_epi32(pulses_sum, 0x4E));
        pulses_sum = _mm_add_epi32(pulses_sum, _mm_shuffle_epi32(pulses_sum, 0xB1));
        pulsesLeft -= _mm_cvtsi128_si32(pulses_sum);

        // Horizontal sum of xy
        xy4 = _mm_add_ps(xy4, _mm_shuffle_ps(xy4, xy4, 0x4E));
        xy4 = _mm_add_ps(xy4, _mm_shuffle_ps(xy4, xy4, 0xB1));
        xy = _mm_cvtss_f32(xy4);

        // Horizontal sum of yy
        yy4 = _mm_add_ps(yy4, _mm_shuffle_ps(yy4, yy4, 0x4E));
        yy4 = _mm_add_ps(yy4, _mm_shuffle_ps(yy4, yy4, 0xB1));
        yy = _mm_cvtss_f32(yy4);
    }

    // Sentinel values prevent SIMD overread from affecting results
    X[n] = -100.0;
    X[n + 1] = -100.0;
    X[n + 2] = -100.0;
    y[n] = 100.0;
    y[n + 1] = 100.0;
    y[n + 2] = 100.0;

    // Fill first bin with excess pulses (should never happen, but safety)
    if pulsesLeft > N + 3 {
        let tmp = pulsesLeft as f32;
        yy += tmp * tmp;
        yy += tmp * y[0];
        iy[0] += pulsesLeft;
        pulsesLeft = 0;
    }

    // Greedy per-pulse search
    for _i in 0..pulsesLeft {
        yy += 1.0;
        let xy4 = _mm_load1_ps(&xy);
        let yy4 = _mm_load1_ps(&yy);
        let mut max = _mm_setzero_ps();
        let mut pos = _mm_setzero_si128();
        let mut count = _mm_set_epi32(3, 2, 1, 0);

        j = 0;
        while j < n {
            let x4 = _mm_loadu_ps(X.as_ptr().add(j));
            let y4 = _mm_loadu_ps(y.as_ptr().add(j));
            let x4 = _mm_add_ps(x4, xy4);
            let y4 = _mm_add_ps(y4, yy4);
            let y4 = _mm_rsqrt_ps(y4);
            let r4 = _mm_mul_ps(x4, y4);
            // Update index of max
            pos = _mm_max_epi16(
                pos,
                _mm_and_si128(count, _mm_castps_si128(_mm_cmpgt_ps(r4, max))),
            );
            // Update max
            max = _mm_max_ps(max, r4);
            count = _mm_add_epi32(count, fours);
            j += 4;
        }

        // Horizontal max
        let mut max2 = _mm_max_ps(max, _mm_shuffle_ps(max, max, 0x4E));
        max2 = _mm_max_ps(max2, _mm_shuffle_ps(max2, max2, 0xB1));
        // Find which lane(s) match the global max
        pos = _mm_and_si128(pos, _mm_castps_si128(_mm_cmpeq_ps(max, max2)));
        pos = _mm_max_epi16(pos, _mm_unpackhi_epi64(pos, pos));
        pos = _mm_max_epi16(pos, _mm_shufflelo_epi16(pos, 0x4E));
        let best_id = _mm_cvtsi128_si32(pos) as usize;

        xy += X[best_id];
        yy += y[best_id];
        y[best_id] += 2.0;
        iy[best_id] += 1;
    }

    // Restore original signs
    j = 0;
    while j < n {
        let y4 = _mm_loadu_si128(iy.as_ptr().add(j) as *const __m128i);
        let s4 = _mm_castps_si128(_mm_loadu_ps(signy.as_ptr().add(j)));
        let y4 = _mm_xor_si128(_mm_add_epi32(y4, s4), s4);
        _mm_storeu_si128(iy.as_mut_ptr().add(j) as *mut __m128i, y4);
        j += 4;
    }

    yy
}

/// SSE implementation of `comb_filter_const`.
/// Port of `celt/x86/pitch_sse.c:comb_filter_const_sse`.
///
/// Processes 4 samples at a time using shuffles to slide the pitch-tap window.
/// The C version takes raw pointers with negative indexing; here we use explicit
/// offsets into the buffer slices.
///
/// # Safety
/// Requires SSE support (checked by caller via cpufeatures).
#[target_feature(enable = "sse")]
pub unsafe fn comb_filter_const_sse(
    y: &mut [f32],
    y_start: usize,
    x: &[f32],
    x_start: usize,
    T: i32,
    N: i32,
    g10: f32,
    g11: f32,
    g12: f32,
) {
    let t = T as usize;
    let n = N as usize;
    let g10v = _mm_set1_ps(g10);
    let g11v = _mm_set1_ps(g11);
    let g12v = _mm_set1_ps(g12);

    // x0v = x[x_start - T - 2 .. x_start - T + 2] (the initial 4-sample window)
    let mut x0v = _mm_loadu_ps(x.as_ptr().add(x_start - t - 2));

    let mut i = 0usize;
    while i + 3 < n {
        let xp = x_start + i - t - 2;
        let yi = _mm_loadu_ps(x.as_ptr().add(x_start + i));
        let x4v = _mm_loadu_ps(x.as_ptr().add(xp + 4));

        // Construct shifted windows using shuffles (matches C #else path)
        let x2v = _mm_shuffle_ps(x0v, x4v, 0x4e); // [x0[2],x0[3],x4[0],x4[1]]
        let x1v = _mm_shuffle_ps(x0v, x2v, 0x99); // [x0[1],x0[2],x2[1],x2[2]] => offset+1
        let x3v = _mm_shuffle_ps(x2v, x4v, 0x99); // [x2[1],x2[2],x4[1],x4[2]] => offset+3

        // yi += g10*x2 + (g11*(x3+x1) + g12*(x4+x0))
        let yi = _mm_add_ps(yi, _mm_mul_ps(g10v, x2v));
        let yi2 = _mm_add_ps(
            _mm_mul_ps(g11v, _mm_add_ps(x3v, x1v)),
            _mm_mul_ps(g12v, _mm_add_ps(x4v, x0v)),
        );
        let yi = _mm_add_ps(yi, yi2);

        x0v = x4v;
        _mm_storeu_ps(y.as_mut_ptr().add(y_start + i), yi);
        i += 4;
    }

    // Intentionally no scalar tail: upstream SSE path only processes i < N-3.
    // Tail handling exists in C only under CUSTOM_MODES.
}

/// SSE implementation of in-place `comb_filter_const`.
///
/// # Safety
/// Requires SSE support (checked by caller via cpufeatures).
#[target_feature(enable = "sse")]
pub unsafe fn comb_filter_const_inplace_sse(
    buf: &mut [f32],
    start: usize,
    T: i32,
    N: i32,
    g10: f32,
    g11: f32,
    g12: f32,
) {
    let ptr = buf.as_mut_ptr();
    let len = buf.len();
    let x = core::slice::from_raw_parts(ptr as *const f32, len);
    comb_filter_const_sse(buf, start, x, start, T, N, g10, g11, g12);
}

/// SSE implementation of `celt_pitch_xcorr`.
/// Processes 4 correlations at a time using `xcorr_kernel_sse`.
///
/// # Safety
/// Requires SSE support (checked by caller via cpufeatures).
#[allow(dead_code)]
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

#[cfg(all(test, feature = "tools"))]
mod tests {
    use super::{
        celt_inner_prod_sse as rust_celt_inner_prod_sse,
        celt_pitch_xcorr_avx2 as rust_celt_pitch_xcorr_avx2,
        comb_filter_const_sse as rust_comb_filter_const_sse,
        dual_inner_prod_sse as rust_dual_inner_prod_sse,
        op_pvq_search_sse2 as rust_op_pvq_search_sse2, xcorr_kernel_sse as rust_xcorr_kernel_sse,
    };

    type OpPvqSearchFn =
        unsafe extern "C" fn(x: *mut f32, iy: *mut i32, k: i32, n: i32, arch: i32) -> f32;
    type XcorrKernelFn =
        unsafe extern "C" fn(x: *const f32, y: *const f32, sum: *mut f32, len: i32);
    type CeltInnerProdFn = unsafe extern "C" fn(x: *const f32, y: *const f32, n: i32) -> f32;
    type DualInnerProdFn = unsafe extern "C" fn(
        x: *const f32,
        y01: *const f32,
        y02: *const f32,
        n: i32,
        xy1: *mut f32,
        xy2: *mut f32,
    );
    type CombFilterConstFn = unsafe extern "C" fn(
        y: *mut f32,
        x: *mut f32,
        t: i32,
        n: i32,
        g10: f32,
        g11: f32,
        g12: f32,
    );
    type PitchXcorrFn = unsafe extern "C" fn(
        x: *const f32,
        y: *const f32,
        xcorr: *mut f32,
        len: i32,
        max_pitch: i32,
        arch: i32,
    );
    unsafe extern "C" {
        fn opus_select_arch() -> i32;
        static OP_PVQ_SEARCH_IMPL: [Option<OpPvqSearchFn>; 5];
        static XCORR_KERNEL_IMPL: [Option<XcorrKernelFn>; 5];
        static CELT_INNER_PROD_IMPL: [Option<CeltInnerProdFn>; 5];
        static DUAL_INNER_PROD_IMPL: [Option<DualInnerProdFn>; 5];
        static COMB_FILTER_CONST_IMPL: [Option<CombFilterConstFn>; 5];
        static PITCH_XCORR_IMPL: [Option<PitchXcorrFn>; 5];
    }

    struct Rng(u64);
    impl Rng {
        fn new(seed: u64) -> Self {
            Self(seed)
        }
        fn next_u32(&mut self) -> u32 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (self.0 >> 32) as u32
        }
        fn next_f32(&mut self) -> f32 {
            let v = self.next_u32() as f32 / (u32::MAX as f32);
            2.0 * v - 1.0
        }
    }

    #[test]
    fn op_pvq_search_sse2_matches_upstream_c() {
        let mut rng = Rng::new(0x1234_9876_dead_beef);
        let c_arch = unsafe { opus_select_arch() } as usize;

        for n in 4..=1024 {
            for _ in 0..24 {
                let k = 1 + (rng.next_u32() % 220) as i32;
                let mut x_c = vec![0.0f32; n];
                for v in &mut x_c {
                    *v = rng.next_f32();
                }
                let mut x_r = x_c.clone();
                let mut iy_c = vec![0i32; n + 4];
                let mut iy_r = vec![0i32; n + 4];

                let c_fn = unsafe { OP_PVQ_SEARCH_IMPL[c_arch] }.expect("C pvq fn");
                let yy_c = unsafe {
                    c_fn(
                        x_c.as_mut_ptr(),
                        iy_c.as_mut_ptr(),
                        k,
                        n as i32,
                        c_arch as i32,
                    )
                };
                let yy_r = unsafe { rust_op_pvq_search_sse2(&mut x_r, &mut iy_r, k, n as i32) };

                assert_eq!(yy_r.to_bits(), yy_c.to_bits(), "yy mismatch n={n} k={k}");
                assert_eq!(&iy_r[..n], &iy_c[..n], "iy mismatch n={n} k={k}");
            }
        }
    }

    #[test]
    fn xcorr_inner_dual_comb_match_upstream_c_simd() {
        let mut rng = Rng::new(0x1337_cafe_f00d_beef);
        let c_arch = unsafe { opus_select_arch() } as usize;
        let c_xcorr = unsafe { XCORR_KERNEL_IMPL[c_arch] }.expect("C xcorr impl");
        let c_inner = unsafe { CELT_INNER_PROD_IMPL[c_arch] }.expect("C inner impl");
        let c_dual = unsafe { DUAL_INNER_PROD_IMPL[c_arch] }.expect("C dual impl");
        let c_comb = unsafe { COMB_FILTER_CONST_IMPL[c_arch] }.expect("C comb impl");

        for len in 4..=1536 {
            let x_len = len + 64;
            let mut x = vec![0.0f32; x_len];
            let mut y = vec![0.0f32; x_len + 64];
            let mut y2 = vec![0.0f32; x_len + 64];
            for v in &mut x {
                *v = rng.next_f32();
            }
            for v in &mut y {
                *v = rng.next_f32();
            }
            for v in &mut y2 {
                *v = rng.next_f32();
            }

            let mut sum_c = [
                rng.next_f32(),
                rng.next_f32(),
                rng.next_f32(),
                rng.next_f32(),
            ];
            let mut sum_r = sum_c;
            unsafe {
                c_xcorr(x.as_ptr(), y.as_ptr(), sum_c.as_mut_ptr(), len as i32);
                rust_xcorr_kernel_sse(&x[..len], &y, &mut sum_r, len);
            }
            for i in 0..4 {
                assert_eq!(
                    sum_r[i].to_bits(),
                    sum_c[i].to_bits(),
                    "xcorr len={len} lane={i}"
                );
            }

            let in_c = unsafe { c_inner(x.as_ptr(), y.as_ptr(), len as i32) };
            let in_r = unsafe { rust_celt_inner_prod_sse(&x, &y, len) };
            assert_eq!(in_r.to_bits(), in_c.to_bits(), "inner len={len}");

            let mut d1_c = 0.0f32;
            let mut d2_c = 0.0f32;
            let (d1_r, d2_r) = unsafe { rust_dual_inner_prod_sse(&x, &y, &y2, len) };
            unsafe {
                c_dual(
                    x.as_ptr(),
                    y.as_ptr(),
                    y2.as_ptr(),
                    len as i32,
                    &mut d1_c,
                    &mut d2_c,
                );
            }
            assert_eq!(d1_r.to_bits(), d1_c.to_bits(), "dual1 len={len}");
            assert_eq!(d2_r.to_bits(), d2_c.to_bits(), "dual2 len={len}");

            if len >= 24 {
                let t = 17;
                let n = len as i32 - 8;
                let x_start = t as usize + 8;
                let mut src_c = vec![0.0f32; x_start + n as usize + 8];
                let mut src_r = vec![0.0f32; x_start + n as usize + 8];
                let mut out_c = vec![0.0f32; n as usize + 8];
                let mut out_r = vec![0.0f32; n as usize + 8];
                for i in 0..src_c.len() {
                    let v = rng.next_f32();
                    src_c[i] = v;
                    src_r[i] = v;
                }
                let g10 = 0.75 * rng.next_f32();
                let g11 = 0.75 * rng.next_f32();
                let g12 = 0.75 * rng.next_f32();

                unsafe {
                    c_comb(
                        out_c.as_mut_ptr(),
                        src_c.as_mut_ptr().add(x_start),
                        t,
                        n,
                        g10,
                        g11,
                        g12,
                    );
                    rust_comb_filter_const_sse(&mut out_r, 0, &src_r, x_start, t, n, g10, g11, g12);
                }
                for i in 0..n as usize {
                    assert_eq!(
                        out_r[i].to_bits(),
                        out_c[i].to_bits(),
                        "comb len={len} i={i} g10={g10} g11={g11} g12={g12}"
                    );
                }

                // In-place/alias scenario used by comb_filter_inplace.
                let mut buf_c = vec![0.0f32; x_start + n as usize + 8];
                let mut buf_r = vec![0.0f32; x_start + n as usize + 8];
                for i in 0..buf_c.len() {
                    let v = rng.next_f32();
                    buf_c[i] = v;
                    buf_r[i] = v;
                }
                unsafe {
                    let x_ptr = buf_r.as_ptr();
                    let x_len = buf_r.len();
                    let x_alias = core::slice::from_raw_parts(x_ptr, x_len);
                    c_comb(
                        buf_c.as_mut_ptr().add(x_start),
                        buf_c.as_mut_ptr().add(x_start),
                        t,
                        n,
                        g10,
                        g11,
                        g12,
                    );
                    rust_comb_filter_const_sse(
                        &mut buf_r, x_start, x_alias, x_start, t, n, g10, g11, g12,
                    );
                }
                for i in 0..n as usize {
                    assert_eq!(
                        buf_r[x_start + i].to_bits(),
                        buf_c[x_start + i].to_bits(),
                        "comb-inplace len={len} i={i} g10={g10} g11={g11} g12={g12}"
                    );
                }
            }
        }
    }

    #[test]
    fn pitch_xcorr_avx2_matches_upstream_c_avx2() {
        let c_arch = unsafe { opus_select_arch() } as usize;
        if c_arch != 4 {
            return;
        }
        let c_pitch = unsafe { PITCH_XCORR_IMPL[c_arch] }.expect("C avx2 pitch xcorr impl");
        let mut rng = Rng::new(0xabcdef01_12345678);

        for len in [16usize, 32, 64, 96, 120, 240, 320, 480, 640, 960] {
            for max_pitch in [32usize, 64, 96, 128, 192, 256, 512] {
                let mut x = vec![0.0f32; len];
                let mut y = vec![0.0f32; len + max_pitch + 8];
                let mut out_c = vec![0.0f32; max_pitch];
                let mut out_r = vec![0.0f32; max_pitch];
                for v in &mut x {
                    *v = rng.next_f32();
                }
                for v in &mut y {
                    *v = rng.next_f32();
                }
                unsafe {
                    c_pitch(
                        x.as_ptr(),
                        y.as_ptr(),
                        out_c.as_mut_ptr(),
                        len as i32,
                        max_pitch as i32,
                        c_arch as i32,
                    );
                    rust_celt_pitch_xcorr_avx2(&x, &y, &mut out_r, len);
                }
                for i in 0..max_pitch {
                    assert_eq!(
                        out_r[i].to_bits(),
                        out_c[i].to_bits(),
                        "pitch_xcorr len={len} max_pitch={max_pitch} i={i}"
                    );
                }
            }
        }
    }

    #[test]
    fn report_c_arch_selection() {
        let c_arch = unsafe { opus_select_arch() };
        let avx2 = std::is_x86_feature_detected!("avx2");
        let fma = std::is_x86_feature_detected!("fma");
        let sse = std::is_x86_feature_detected!("sse");
        let sse2 = std::is_x86_feature_detected!("sse2");
        eprintln!("c_arch={c_arch} avx2={avx2} fma={fma} sse={sse} sse2={sse2}");
    }
}
