//! x86/x86_64 AVX2+FMA SIMD implementations for DNN vector primitives.
//!
//! All functions require `#[target_feature(enable = "avx2", enable = "fma")]`
//! and are called only after cpufeatures detection confirms AVX2+FMA support.
//!
//! Port of `dnn/vec_avx.h` from libopus 1.5.2.

#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

// =========================================================================
// Activation helpers
// =========================================================================

/// AVX2 fast 2^x approximation via IEEE 754 bit manipulation.
/// Port of `vec_avx.h:exp8_approx` (AVX2 path).
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn exp8_approx(x: __m256) -> __m256 {
    let k0 = _mm256_set1_ps(0.99992522);
    let k1 = _mm256_set1_ps(0.69583354);
    let k2 = _mm256_set1_ps(0.22606716);
    let k3 = _mm256_set1_ps(0.078024523);
    let log2_e = _mm256_set1_ps(1.44269504);
    let max_in = _mm256_set1_ps(50.0);
    let min_in = _mm256_set1_ps(-50.0);

    let x = _mm256_mul_ps(x, log2_e);
    let x = _mm256_max_ps(min_in, _mm256_min_ps(max_in, x));
    let xf = _mm256_floor_ps(x);
    let i = _mm256_cvtps_epi32(xf);
    let x = _mm256_sub_ps(x, xf);

    // Polynomial: K0 + x*(K1 + x*(K2 + x*K3))
    let y = _mm256_fmadd_ps(_mm256_fmadd_ps(_mm256_fmadd_ps(k3, x, k2), x, k1), x, k0);

    // Multiply by 2^i via exponent bit manipulation
    let i = _mm256_slli_epi32(i, 23);
    _mm256_castsi256_ps(_mm256_add_epi32(i, _mm256_castps_si256(y)))
}

/// AVX2 fast tanh approximation using Padé rational function.
/// Port of `vec_avx.h:tanh8_approx`.
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn tanh8_approx(x: __m256) -> __m256 {
    let n0 = _mm256_set1_ps(952.52801514);
    let n1 = _mm256_set1_ps(96.39235687);
    let n2 = _mm256_set1_ps(0.60863042);
    let d0 = _mm256_set1_ps(952.72399902);
    let d1 = _mm256_set1_ps(413.36801147);
    let d2 = _mm256_set1_ps(11.88600922);
    let max_out = _mm256_set1_ps(1.0);
    let min_out = _mm256_set1_ps(-1.0);

    let x2 = _mm256_mul_ps(x, x);
    let num = _mm256_fmadd_ps(_mm256_fmadd_ps(n2, x2, n1), x2, n0);
    let den = _mm256_fmadd_ps(_mm256_fmadd_ps(d2, x2, d1), x2, d0);
    let num = _mm256_mul_ps(num, x);
    let den = _mm256_rcp_ps(den);
    let num = _mm256_mul_ps(num, den);
    _mm256_max_ps(min_out, _mm256_min_ps(max_out, num))
}

/// AVX2 fast sigmoid approximation using Padé rational function.
/// Port of `vec_avx.h:sigmoid8_approx`.
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn sigmoid8_approx(x: __m256) -> __m256 {
    let n0 = _mm256_set1_ps(238.13200378);
    let n1 = _mm256_set1_ps(6.02452230);
    let n2 = _mm256_set1_ps(0.00950985);
    let d0 = _mm256_set1_ps(952.72399902);
    let d1 = _mm256_set1_ps(103.34200287);
    let d2 = _mm256_set1_ps(0.74287558);
    let half = _mm256_set1_ps(0.5);
    let max_out = _mm256_set1_ps(1.0);
    let min_out = _mm256_set1_ps(0.0);

    let x2 = _mm256_mul_ps(x, x);
    let num = _mm256_fmadd_ps(_mm256_fmadd_ps(n2, x2, n1), x2, n0);
    let den = _mm256_fmadd_ps(_mm256_fmadd_ps(d2, x2, d1), x2, d0);
    let num = _mm256_mul_ps(num, x);
    let den = _mm256_rcp_ps(den);
    let num = _mm256_fmadd_ps(num, den, half);
    _mm256_max_ps(min_out, _mm256_min_ps(max_out, num))
}

// =========================================================================
// Scalar activation via broadcast-and-extract (matching C vec_avx.h)
// =========================================================================
// On x86 with AVX, C's scalar tanh_approx/sigmoid_approx/lpcnet_exp broadcast
// into __m256, call the 8-wide SIMD version, and extract lane 0. This produces
// slightly different results than a pure scalar implementation because the SIMD
// versions use _mm256_rcp_ps (approximate reciprocal, ~12-bit precision) instead
// of true division, and exp8_approx omits the sign-bit mask that scalar lpcnet_exp2
// applies. We must match this behavior for bit-exactness with C.

/// Scalar tanh via AVX2 broadcast. Port of `vec_avx.h:tanh_approx` (AVX path).
///
/// # Safety
/// Requires AVX2+FMA support (checked by caller via cpufeatures).
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn tanh_approx_avx2(x: f32) -> f32 {
    let xv = _mm256_set1_ps(x);
    let yv = tanh8_approx(xv);
    _mm_cvtss_f32(_mm256_castps256_ps128(yv))
}

/// Scalar sigmoid via AVX2 broadcast. Port of `vec_avx.h:sigmoid_approx` (AVX path).
///
/// # Safety
/// Requires AVX2+FMA support (checked by caller via cpufeatures).
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn sigmoid_approx_avx2(x: f32) -> f32 {
    let xv = _mm256_set1_ps(x);
    let yv = sigmoid8_approx(xv);
    _mm_cvtss_f32(_mm256_castps256_ps128(yv))
}

/// Scalar exp via AVX2 broadcast. Port of `vec_avx.h:lpcnet_exp` (AVX path).
///
/// # Safety
/// Requires AVX2+FMA support (checked by caller via cpufeatures).
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn lpcnet_exp_avx2(x: f32) -> f32 {
    let xv = _mm256_set1_ps(x);
    let yv = exp8_approx(xv);
    _mm_cvtss_f32(_mm256_castps256_ps128(yv))
}

// =========================================================================
// Batch activation functions
// =========================================================================

/// AVX2 batch tanh approximation.
/// Port of `vec_avx.h:vec_tanh` (AVX path).
///
/// # Safety
/// Requires AVX2+FMA support (checked by caller via cpufeatures).
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn vec_tanh_avx2(y: &mut [f32], x: &[f32]) {
    let n = x.len().min(y.len());
    let mut i = 0;
    while i + 8 <= n {
        let xv = _mm256_loadu_ps(x.as_ptr().add(i));
        let yv = tanh8_approx(xv);
        _mm256_storeu_ps(y.as_mut_ptr().add(i), yv);
        i += 8;
    }
    // Scalar tail: use broadcast-and-extract to match C's vec_avx.h which
    // redefines tanh_approx to use tanh8_approx (with _mm256_rcp_ps).
    while i < n {
        y[i] = tanh_approx_avx2(x[i]);
        i += 1;
    }
}

/// AVX2 batch sigmoid approximation.
/// Port of `vec_avx.h:vec_sigmoid` (AVX path).
///
/// # Safety
/// Requires AVX2+FMA support (checked by caller via cpufeatures).
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn vec_sigmoid_avx2(y: &mut [f32], x: &[f32]) {
    let n = x.len().min(y.len());
    let mut i = 0;
    while i + 8 <= n {
        let xv = _mm256_loadu_ps(x.as_ptr().add(i));
        let yv = sigmoid8_approx(xv);
        _mm256_storeu_ps(y.as_mut_ptr().add(i), yv);
        i += 8;
    }
    // Scalar tail: use broadcast-and-extract to match C's vec_avx.h.
    while i < n {
        y[i] = sigmoid_approx_avx2(x[i]);
        i += 1;
    }
}

/// AVX2 batch softmax (unnormalized exp).
/// Port of `vec_avx.h:softmax`.
///
/// # Safety
/// Requires AVX2+FMA support (checked by caller via cpufeatures).
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn softmax_avx2(y: &mut [f32], x: &[f32]) {
    let n = x.len().min(y.len());
    let mut i = 0;
    while i + 8 <= n {
        let xv = _mm256_loadu_ps(x.as_ptr().add(i));
        let yv = exp8_approx(xv);
        _mm256_storeu_ps(y.as_mut_ptr().add(i), yv);
        i += 8;
    }
    // Scalar tail: use broadcast-and-extract to match C's vec_avx.h.
    while i < n {
        y[i] = lpcnet_exp_avx2(x[i]);
        i += 1;
    }
}

// =========================================================================
// Dense float GEMV
// =========================================================================

/// AVX2+FMA dense float matrix-vector multiply: out = weights^T * x.
/// Port of `vec_avx.h:sgemv`.
///
/// # Safety
/// Requires AVX2+FMA support (checked by caller via cpufeatures).
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn sgemv_avx2(
    out: &mut [f32],
    weights: &[f32],
    rows: usize,
    cols: usize,
    col_stride: usize,
    x: &[f32],
) {
    let mut i = 0;

    // 16-row blocks
    while i + 16 <= rows {
        let mut vy0 = _mm256_setzero_ps();
        let mut vy8 = _mm256_setzero_ps();
        for j in 0..cols {
            let vxj = _mm256_broadcast_ss(&x[j]);
            let w = weights.as_ptr().add(j * col_stride + i);
            let vw0 = _mm256_loadu_ps(w);
            vy0 = _mm256_fmadd_ps(vw0, vxj, vy0);
            let vw8 = _mm256_loadu_ps(w.add(8));
            vy8 = _mm256_fmadd_ps(vw8, vxj, vy8);
        }
        _mm256_storeu_ps(out.as_mut_ptr().add(i), vy0);
        _mm256_storeu_ps(out.as_mut_ptr().add(i + 8), vy8);
        i += 16;
    }

    // 8-row blocks
    while i + 8 <= rows {
        let mut vy0 = _mm256_setzero_ps();
        for j in 0..cols {
            let vxj = _mm256_broadcast_ss(&x[j]);
            let vw = _mm256_loadu_ps(weights.as_ptr().add(j * col_stride + i));
            vy0 = _mm256_fmadd_ps(vw, vxj, vy0);
        }
        _mm256_storeu_ps(out.as_mut_ptr().add(i), vy0);
        i += 8;
    }

    // 4-row blocks (SSE)
    while i + 4 <= rows {
        let mut vy0 = _mm_setzero_ps();
        for j in 0..cols {
            let vxj = _mm_set1_ps(x[j]);
            let vw = _mm_loadu_ps(weights.as_ptr().add(j * col_stride + i));
            vy0 = _mm_fmadd_ps(vw, vxj, vy0);
        }
        _mm_storeu_ps(out.as_mut_ptr().add(i), vy0);
        i += 4;
    }

    // Scalar tail
    while i < rows {
        out[i] = 0.0;
        for j in 0..cols {
            out[i] += weights[j * col_stride + i] * x[j];
        }
        i += 1;
    }
}

// =========================================================================
// Sparse float GEMV
// =========================================================================

/// AVX2+FMA sparse float matrix-vector multiply (8x4 block sparse).
/// Port of `vec_avx.h:sparse_sgemv8x4`.
///
/// # Safety
/// Requires AVX2+FMA support (checked by caller via cpufeatures).
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn sparse_sgemv8x4_avx2(
    out: &mut [f32],
    w: &[f32],
    idx: &[i32],
    rows: usize,
    x: &[f32],
) {
    let mut w_pos = 0;
    let mut idx_pos = 0;

    for i in (0..rows).step_by(8) {
        let cols = idx[idx_pos] as usize;
        idx_pos += 1;
        let mut vy0 = _mm256_setzero_ps();

        for _j in 0..cols {
            let id = idx[idx_pos] as usize;
            idx_pos += 1;

            let vxj = _mm256_broadcast_ss(&x[id]);
            let vw = _mm256_loadu_ps(w.as_ptr().add(w_pos));
            vy0 = _mm256_fmadd_ps(vw, vxj, vy0);

            let vxj = _mm256_broadcast_ss(&x[id + 1]);
            let vw = _mm256_loadu_ps(w.as_ptr().add(w_pos + 8));
            vy0 = _mm256_fmadd_ps(vw, vxj, vy0);

            let vxj = _mm256_broadcast_ss(&x[id + 2]);
            let vw = _mm256_loadu_ps(w.as_ptr().add(w_pos + 16));
            vy0 = _mm256_fmadd_ps(vw, vxj, vy0);

            let vxj = _mm256_broadcast_ss(&x[id + 3]);
            let vw = _mm256_loadu_ps(w.as_ptr().add(w_pos + 24));
            vy0 = _mm256_fmadd_ps(vw, vxj, vy0);

            w_pos += 32;
        }

        _mm256_storeu_ps(out.as_mut_ptr().add(i), vy0);
    }
}

// =========================================================================
// Dense int8 GEMV
// =========================================================================

/// Quantize f32 input to u8: x[i] = 127 + round(127 * _x[i]).
/// Port of `vec_avx.h:vector_ps_to_epi8` (AVX2 path).
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn vector_ps_to_u8(dst: &mut [u8], src: &[f32], len: usize) {
    let const127 = _mm256_set1_ps(127.0);
    let mut i = 0;
    while i + 8 <= len {
        let xf = _mm256_loadu_ps(src.as_ptr().add(i));
        let xf = _mm256_fmadd_ps(xf, const127, const127);
        let xi = _mm256_cvtps_epi32(xf);
        // Pack i32 -> u16 -> u8 (unsigned saturation)
        let xi = _mm256_packus_epi32(xi, _mm256_setzero_si256());
        let xi = _mm256_permute4x64_epi64(xi, 0xD8);
        let xi = _mm256_packus_epi16(xi, _mm256_setzero_si256());
        // The result has 8 bytes in a scrambled order; fix with permute
        let xi = _mm256_permutevar8x32_epi32(xi, _mm256_setr_epi32(0, 1, 0, 0, 0, 0, 0, 0));
        _mm256_storeu_si256(dst.as_mut_ptr().add(i) as *mut __m256i, xi);
        i += 8;
    }
    // Scalar tail
    while i < len {
        dst[i] = (127.0f64 + (0.5f64 + 127.0f64 * src[i] as f64).floor()) as u8;
        i += 1;
    }
}

/// Emulated dpbusds: unsigned*signed dot product and accumulate.
/// Port of `vec_avx.h:opus_mm256_dpbusds_epi32` (AVX2 path).
///
/// Computes: for each group of 4 bytes in a (unsigned) and b (signed),
/// sum(a[k]*b[k]) and accumulate into src (i32x8).
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn opus_mm256_dpbusds_epi32(src: __m256i, a: __m256i, b: __m256i) -> __m256i {
    let ones = _mm256_set1_epi16(1);
    let tmp = _mm256_maddubs_epi16(a, b);
    let tmp = _mm256_madd_epi16(tmp, ones);
    _mm256_add_epi32(src, tmp)
}

/// AVX2 dense int8 matrix-vector multiply (8x4 blocking).
/// Port of `vec_avx.h:cgemv8x4`.
///
/// Uses unsigned u8 quantization (127 + round(127*x)) with dpbusds emulation.
///
/// # Safety
/// Requires AVX2+FMA support (checked by caller via cpufeatures).
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn cgemv8x4_avx2(
    out: &mut [f32],
    w: &[i8],
    scale: &[f32],
    rows: usize,
    cols: usize,
    _x: &[f32],
) {
    const MAX_INPUTS: usize = 2048;
    let mut x = [0u8; MAX_INPUTS];
    vector_ps_to_u8(&mut x, _x, cols);

    let mut w_pos = 0;
    for i in (0..rows).step_by(8) {
        let mut vy0 = _mm256_setzero_si256();
        let mut j = 0;

        // Unrolled by 4: process 4 groups of 4 columns per iteration.
        // Must match C condition `j < cols - 12` (enters when 13+ columns remain)
        // rather than `j + 16 <= cols` (16+ remain), because _mm256_maddubs_epi16
        // uses saturating i16 addition internally — different accumulation grouping
        // produces different results.
        while j + 12 < cols {
            let vxj = _mm256_broadcastd_epi32(_mm_loadu_si32(x.as_ptr().add(j) as *const _));
            let vw = _mm256_loadu_si256(w.as_ptr().add(w_pos) as *const __m256i);
            vy0 = opus_mm256_dpbusds_epi32(vy0, vxj, vw);
            w_pos += 32;

            let vxj = _mm256_broadcastd_epi32(_mm_loadu_si32(x.as_ptr().add(j + 4) as *const _));
            let vw = _mm256_loadu_si256(w.as_ptr().add(w_pos) as *const __m256i);
            vy0 = opus_mm256_dpbusds_epi32(vy0, vxj, vw);
            w_pos += 32;

            let vxj = _mm256_broadcastd_epi32(_mm_loadu_si32(x.as_ptr().add(j + 8) as *const _));
            let vw = _mm256_loadu_si256(w.as_ptr().add(w_pos) as *const __m256i);
            vy0 = opus_mm256_dpbusds_epi32(vy0, vxj, vw);
            w_pos += 32;

            let vxj = _mm256_broadcastd_epi32(_mm_loadu_si32(x.as_ptr().add(j + 12) as *const _));
            let vw = _mm256_loadu_si256(w.as_ptr().add(w_pos) as *const __m256i);
            vy0 = opus_mm256_dpbusds_epi32(vy0, vxj, vw);
            w_pos += 32;

            j += 16;
        }
        while j < cols {
            let vxj = _mm256_broadcastd_epi32(_mm_loadu_si32(x.as_ptr().add(j) as *const _));
            let vw = _mm256_loadu_si256(w.as_ptr().add(w_pos) as *const __m256i);
            vy0 = opus_mm256_dpbusds_epi32(vy0, vxj, vw);
            w_pos += 32;
            j += 4;
        }

        let vout = _mm256_cvtepi32_ps(vy0);
        let vscale = _mm256_loadu_ps(scale.as_ptr().add(i));
        _mm256_storeu_ps(out.as_mut_ptr().add(i), _mm256_mul_ps(vout, vscale));
    }
}

// =========================================================================
// Sparse int8 GEMV
// =========================================================================

/// AVX2 sparse int8 matrix-vector multiply (8x4 block sparse).
/// Port of `vec_avx.h:sparse_cgemv8x4`.
///
/// # Safety
/// Requires AVX2+FMA support (checked by caller via cpufeatures).
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn sparse_cgemv8x4_avx2(
    out: &mut [f32],
    w: &[i8],
    idx: &[i32],
    scale: &[f32],
    rows: usize,
    cols: usize,
    _x: &[f32],
) {
    const MAX_INPUTS: usize = 2048;
    let mut x = [0u8; MAX_INPUTS];
    vector_ps_to_u8(&mut x, _x, cols);

    let mut w_pos = 0;
    let mut idx_pos = 0;

    for i in (0..rows).step_by(8) {
        let colblocks = idx[idx_pos] as usize;
        idx_pos += 1;
        let mut vy0 = _mm256_setzero_si256();
        let mut j = 0;

        // Unrolled by 4
        while j + 4 <= colblocks {
            let pos = idx[idx_pos] as usize;
            idx_pos += 1;
            let vxj = _mm256_broadcastd_epi32(_mm_loadu_si32(x.as_ptr().add(pos) as *const _));
            let vw = _mm256_loadu_si256(w.as_ptr().add(w_pos) as *const __m256i);
            vy0 = opus_mm256_dpbusds_epi32(vy0, vxj, vw);
            w_pos += 32;

            let pos = idx[idx_pos] as usize;
            idx_pos += 1;
            let vxj = _mm256_broadcastd_epi32(_mm_loadu_si32(x.as_ptr().add(pos) as *const _));
            let vw = _mm256_loadu_si256(w.as_ptr().add(w_pos) as *const __m256i);
            vy0 = opus_mm256_dpbusds_epi32(vy0, vxj, vw);
            w_pos += 32;

            let pos = idx[idx_pos] as usize;
            idx_pos += 1;
            let vxj = _mm256_broadcastd_epi32(_mm_loadu_si32(x.as_ptr().add(pos) as *const _));
            let vw = _mm256_loadu_si256(w.as_ptr().add(w_pos) as *const __m256i);
            vy0 = opus_mm256_dpbusds_epi32(vy0, vxj, vw);
            w_pos += 32;

            let pos = idx[idx_pos] as usize;
            idx_pos += 1;
            let vxj = _mm256_broadcastd_epi32(_mm_loadu_si32(x.as_ptr().add(pos) as *const _));
            let vw = _mm256_loadu_si256(w.as_ptr().add(w_pos) as *const __m256i);
            vy0 = opus_mm256_dpbusds_epi32(vy0, vxj, vw);
            w_pos += 32;

            j += 4;
        }
        while j < colblocks {
            let pos = idx[idx_pos] as usize;
            idx_pos += 1;
            let vxj = _mm256_broadcastd_epi32(_mm_loadu_si32(x.as_ptr().add(pos) as *const _));
            let vw = _mm256_loadu_si256(w.as_ptr().add(w_pos) as *const __m256i);
            vy0 = opus_mm256_dpbusds_epi32(vy0, vxj, vw);
            w_pos += 32;
            j += 1;
        }

        let vout = _mm256_cvtepi32_ps(vy0);
        let vscale = _mm256_loadu_ps(scale.as_ptr().add(i));
        _mm256_storeu_ps(out.as_mut_ptr().add(i), _mm256_mul_ps(vout, vscale));
    }
}
