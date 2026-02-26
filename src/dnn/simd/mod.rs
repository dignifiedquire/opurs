//! SIMD-accelerated DNN vector primitives.
//!
//! This module provides SIMD dispatch for performance-critical DNN functions.
//! On x86/x86_64, runtime CPU feature detection selects the AVX2+FMA path.
//! On aarch64, NEON is always available and selected at compile time.
//! On other architectures (or with the `simd` feature disabled), falls through to scalar.
//!
//! Unlike CELT/SILK (which require bit-exactness), DNN inference is approximate,
//! so FMA instructions are used freely.

use crate::arch::Arch;

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub mod x86;

#[cfg(target_arch = "aarch64")]
pub mod aarch64;

// -- Dispatch functions --
// Each function selects the best available SIMD implementation at runtime,
// falling back to the scalar version.

/// SIMD-accelerated dense float matrix-vector multiply.
#[inline]
pub fn sgemv(
    out: &mut [f32],
    weights: &[f32],
    rows: usize,
    cols: usize,
    col_stride: usize,
    x: &[f32],
    _arch: Arch,
) {
    #[cfg(target_arch = "aarch64")]
    {
        unsafe {
            aarch64::sgemv_neon(out, weights, rows, cols, col_stride, x);
        }
        return;
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if _arch.has_avx2() {
            unsafe {
                x86::sgemv_avx2(out, weights, rows, cols, col_stride, x);
            }
            return;
        }
        if _arch.has_sse2() {
            unsafe {
                x86::sgemv_sse2(out, weights, rows, cols, col_stride, x);
            }
            return;
        }
    }

    #[allow(unreachable_code)]
    {
        super::vec::sgemv_scalar(out, weights, rows, cols, col_stride, x);
    }
}

/// SIMD-accelerated sparse float matrix-vector multiply (8x4 block sparse).
#[inline]
pub fn sparse_sgemv8x4(
    out: &mut [f32],
    w: &[f32],
    idx: &[i32],
    rows: usize,
    x: &[f32],
    _arch: Arch,
) {
    #[cfg(target_arch = "aarch64")]
    {
        unsafe {
            aarch64::sparse_sgemv8x4_neon(out, w, idx, rows, x);
        }
        return;
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if _arch.has_avx2() {
            unsafe {
                x86::sparse_sgemv8x4_avx2(out, w, idx, rows, x);
            }
            return;
        }
        if _arch.has_sse2() {
            unsafe {
                x86::sparse_sgemv8x4_sse2(out, w, idx, rows, x);
            }
            return;
        }
    }

    #[allow(unreachable_code)]
    {
        super::vec::sparse_sgemv8x4_scalar(out, w, idx, rows, x);
    }
}

/// SIMD-accelerated dense int8 matrix-vector multiply (8x4 blocking).
#[inline]
pub fn cgemv8x4(
    out: &mut [f32],
    w: &[i8],
    scale: &[f32],
    rows: usize,
    cols: usize,
    x: &[f32],
    _arch: Arch,
) {
    #[cfg(target_arch = "aarch64")]
    {
        unsafe {
            if _arch.has_dotprod() {
                aarch64::cgemv8x4_dotprod(out, w, scale, rows, cols, x);
            } else {
                aarch64::cgemv8x4_neon(out, w, scale, rows, cols, x);
            }
        }
        return;
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if _arch.has_avx2() {
            unsafe {
                x86::cgemv8x4_avx2(out, w, scale, rows, cols, x);
            }
            return;
        }
        // Upstream x86 SSE4.1 path emulates dpbusds with maddubs i16 saturation.
        if _arch.has_sse4_1() {
            super::vec::cgemv8x4_scalar_su_ssse3(out, w, scale, rows, cols, x);
            return;
        }
        if _arch.has_sse2() {
            // SSE2 fallback path (no maddubs saturation) still uses USE_SU_BIAS quantization.
            super::vec::cgemv8x4_scalar_su(out, w, scale, rows, cols, x);
            return;
        }
        // Upstream x86 compiles vec_avx.h for all runtime arch tiers, including
        // arch=0/1 entries that still dispatch to compute_linear_c.
        // Keep USE_SU_BIAS quantization in this fallback too.
        super::vec::cgemv8x4_scalar_su(out, w, scale, rows, cols, x);
        return;
    }

    #[allow(unreachable_code)]
    {
        super::vec::cgemv8x4_scalar(out, w, scale, rows, cols, x);
    }
}

/// SIMD-accelerated sparse int8 matrix-vector multiply (8x4 block sparse).
#[inline]
pub fn sparse_cgemv8x4(
    out: &mut [f32],
    w: &[i8],
    idx: &[i32],
    scale: &[f32],
    rows: usize,
    cols: usize,
    x: &[f32],
    _arch: Arch,
) {
    #[cfg(target_arch = "aarch64")]
    {
        unsafe {
            if _arch.has_dotprod() {
                aarch64::sparse_cgemv8x4_dotprod(out, w, idx, scale, rows, cols, x);
            } else {
                aarch64::sparse_cgemv8x4_neon(out, w, idx, scale, rows, cols, x);
            }
        }
        return;
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if _arch.has_avx2() {
            unsafe {
                x86::sparse_cgemv8x4_avx2(out, w, idx, scale, rows, cols, x);
            }
            return;
        }
        // Upstream x86 SSE4.1 path emulates dpbusds with maddubs i16 saturation.
        if _arch.has_sse4_1() {
            super::vec::sparse_cgemv8x4_scalar_su_ssse3(out, w, idx, scale, rows, cols, x);
            return;
        }
        if _arch.has_sse2() {
            // SSE2 fallback path (no maddubs saturation) still uses USE_SU_BIAS quantization.
            super::vec::sparse_cgemv8x4_scalar_su(out, w, idx, scale, rows, cols, x);
            return;
        }
        // Upstream x86 compute_linear_c also uses USE_SU_BIAS at low arch tiers.
        super::vec::sparse_cgemv8x4_scalar_su(out, w, idx, scale, rows, cols, x);
        return;
    }

    #[allow(unreachable_code)]
    {
        super::vec::sparse_cgemv8x4_scalar(out, w, idx, scale, rows, cols, x);
    }
}

// =========================================================================
// Scalar activation functions (dispatch to NEON/AVX on supported platforms)
// =========================================================================
// On aarch64, C's vec_neon.h redefines the scalar activation functions to
// broadcast into a NEON register, call the 4-wide version, and extract lane 0.
// This gives slightly different results due to FMA and approximate reciprocal.

/// Scalar tanh approximation, matching platform-specific behavior.
///
/// On aarch64: broadcasts into NEON, calls tanh4_approx, extracts lane 0.
/// On x86 with AVX2: broadcasts into __m256, calls tanh8_approx, extracts lane 0.
/// This matches C's `vec_avx.h:tanh_approx` which uses `_mm256_rcp_ps` (approximate
/// reciprocal) rather than true division, producing slightly different results.
#[inline]
pub fn tanh_approx(x: f32, _arch: Arch) -> f32 {
    #[cfg(target_arch = "aarch64")]
    {
        return unsafe { aarch64::tanh_approx_neon(x) };
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if _arch.has_avx2() {
            return unsafe { x86::tanh_approx_avx2(x) };
        }
        if _arch.has_sse2() {
            return unsafe { x86::tanh_approx_sse2(x) };
        }
    }

    #[allow(unreachable_code)]
    {
        super::vec::tanh_approx(x)
    }
}

/// Scalar sigmoid approximation, matching platform-specific behavior.
///
/// On aarch64: broadcasts into NEON, calls sigmoid4_approx, extracts lane 0.
/// On x86 with AVX2: broadcasts into __m256, calls sigmoid8_approx, extracts lane 0.
#[inline]
pub fn sigmoid_approx(x: f32, _arch: Arch) -> f32 {
    #[cfg(target_arch = "aarch64")]
    {
        return unsafe { aarch64::sigmoid_approx_neon(x) };
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if _arch.has_avx2() {
            return unsafe { x86::sigmoid_approx_avx2(x) };
        }
        if _arch.has_sse2() {
            return unsafe { x86::sigmoid_approx_sse2(x) };
        }
    }

    #[allow(unreachable_code)]
    {
        super::vec::sigmoid_approx(x)
    }
}

/// Scalar lpcnet_exp, matching platform-specific behavior.
///
/// On aarch64: broadcasts into NEON, calls exp4_approx, extracts lane 0.
/// On x86 with AVX2: broadcasts into __m256, calls exp8_approx, extracts lane 0.
#[inline]
pub fn lpcnet_exp(x: f32, _arch: Arch) -> f32 {
    #[cfg(target_arch = "aarch64")]
    {
        return unsafe { aarch64::lpcnet_exp_neon(x) };
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if _arch.has_avx2() {
            return unsafe { x86::lpcnet_exp_avx2(x) };
        }
    }

    #[allow(unreachable_code)]
    {
        super::vec::lpcnet_exp(x)
    }
}

/// SIMD-accelerated batch tanh approximation.
#[inline]
pub fn vec_tanh(y: &mut [f32], x: &[f32], _arch: Arch) {
    #[cfg(target_arch = "aarch64")]
    {
        unsafe {
            aarch64::vec_tanh_neon(y, x);
        }
        return;
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if _arch.has_avx2() {
            unsafe {
                x86::vec_tanh_avx2(y, x);
            }
            return;
        }
        if _arch.has_sse2() {
            unsafe {
                x86::vec_tanh_sse2(y, x);
            }
            return;
        }
    }

    #[allow(unreachable_code)]
    {
        super::vec::vec_tanh_scalar(y, x);
    }
}

/// SIMD-accelerated batch sigmoid approximation.
#[inline]
pub fn vec_sigmoid(y: &mut [f32], x: &[f32], _arch: Arch) {
    #[cfg(target_arch = "aarch64")]
    {
        unsafe {
            aarch64::vec_sigmoid_neon(y, x);
        }
        return;
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if _arch.has_avx2() {
            unsafe {
                x86::vec_sigmoid_avx2(y, x);
            }
            return;
        }
        if _arch.has_sse2() {
            unsafe {
                x86::vec_sigmoid_sse2(y, x);
            }
            return;
        }
    }

    #[allow(unreachable_code)]
    {
        super::vec::vec_sigmoid_scalar(y, x);
    }
}

/// Returns `true` when the active int8 GEMV path uses unsigned u8 quantization.
///
/// On x86, upstream C defines `USE_SU_BIAS` for both AVX2 and SSE2/SSE4.1
/// variants in `vec_avx.h`, so int8 GEMV uses unsigned input quantization and
/// the `subias` correction path.
///
/// On aarch64 NEON (signed i8) and non-x86 scalar fallback, regular `bias` is
/// used.
///
/// Upstream C: `#define USE_SU_BIAS` in `vec_avx.h` (x86 only).
#[inline]
pub fn use_su_bias(arch: Arch) -> bool {
    let _ = arch;

    #[cfg(target_arch = "aarch64")]
    {
        return false; // NEON uses signed i8 quantization
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        let _ = arch;
        return true; // x86 build uses vec_avx.h (USE_SU_BIAS) for all arch tiers
    }

    #[allow(unreachable_code)]
    {
        false // scalar fallback uses signed i8
    }
}

/// SIMD-accelerated batch softmax (unnormalized exp).
#[inline]
pub fn softmax(y: &mut [f32], x: &[f32], _arch: Arch) {
    #[cfg(target_arch = "aarch64")]
    {
        unsafe {
            aarch64::softmax_neon(y, x);
        }
        return;
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if _arch.has_avx2() {
            unsafe {
                x86::softmax_avx2(y, x);
            }
            return;
        }
    }

    #[allow(unreachable_code)]
    {
        super::vec::softmax_scalar(y, x);
    }
}
