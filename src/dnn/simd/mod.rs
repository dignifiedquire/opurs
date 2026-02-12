//! SIMD-accelerated DNN vector primitives.
//!
//! This module provides SIMD dispatch for performance-critical DNN functions.
//! On x86/x86_64, runtime CPU feature detection selects the AVX2+FMA path.
//! On aarch64, NEON is always available and selected at compile time.
//! On other architectures (or with the `simd` feature disabled), falls through to scalar.
//!
//! Unlike CELT/SILK (which require bit-exactness), DNN inference is approximate,
//! so FMA instructions are used freely.

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub mod x86;

#[cfg(target_arch = "aarch64")]
pub mod aarch64;

// -- CPU feature detection (x86/x86_64) --

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
cpufeatures::new!(cpuid_avx2, "avx2", "fma");

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
        if cpuid_avx2::get() {
            unsafe {
                x86::sgemv_avx2(out, weights, rows, cols, col_stride, x);
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
pub fn sparse_sgemv8x4(out: &mut [f32], w: &[f32], idx: &[i32], rows: usize, x: &[f32]) {
    #[cfg(target_arch = "aarch64")]
    {
        unsafe {
            aarch64::sparse_sgemv8x4_neon(out, w, idx, rows, x);
        }
        return;
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if cpuid_avx2::get() {
            unsafe {
                x86::sparse_sgemv8x4_avx2(out, w, idx, rows, x);
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
pub fn cgemv8x4(out: &mut [f32], w: &[i8], scale: &[f32], rows: usize, cols: usize, x: &[f32]) {
    #[cfg(target_arch = "aarch64")]
    {
        unsafe {
            aarch64::cgemv8x4_neon(out, w, scale, rows, cols, x);
        }
        return;
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if cpuid_avx2::get() {
            unsafe {
                x86::cgemv8x4_avx2(out, w, scale, rows, cols, x);
            }
            return;
        }
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
) {
    #[cfg(target_arch = "aarch64")]
    {
        unsafe {
            aarch64::sparse_cgemv8x4_neon(out, w, idx, scale, rows, cols, x);
        }
        return;
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if cpuid_avx2::get() {
            unsafe {
                x86::sparse_cgemv8x4_avx2(out, w, idx, scale, rows, cols, x);
            }
            return;
        }
    }

    #[allow(unreachable_code)]
    {
        super::vec::sparse_cgemv8x4_scalar(out, w, idx, scale, rows, cols, x);
    }
}

/// SIMD-accelerated batch tanh approximation.
#[inline]
pub fn vec_tanh(y: &mut [f32], x: &[f32]) {
    #[cfg(target_arch = "aarch64")]
    {
        unsafe {
            aarch64::vec_tanh_neon(y, x);
        }
        return;
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if cpuid_avx2::get() {
            unsafe {
                x86::vec_tanh_avx2(y, x);
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
pub fn vec_sigmoid(y: &mut [f32], x: &[f32]) {
    #[cfg(target_arch = "aarch64")]
    {
        unsafe {
            aarch64::vec_sigmoid_neon(y, x);
        }
        return;
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if cpuid_avx2::get() {
            unsafe {
                x86::vec_sigmoid_avx2(y, x);
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
/// On x86 with AVX2+FMA, `cgemv8x4` uses `_mm256_maddubs_epi16` which requires
/// unsigned×signed operands, so inputs are quantized as `127 + round(127*x)`.
/// The `subias` in `LinearLayer` compensates for this +127 offset.
///
/// On aarch64 NEON (signed i8) and scalar fallback, regular `bias` is used.
///
/// Upstream C: `#define USE_SU_BIAS` in `vec_avx.h` (x86 only).
#[inline]
pub fn use_su_bias() -> bool {
    #[cfg(target_arch = "aarch64")]
    {
        return false; // NEON uses signed i8 quantization
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        return cpuid_avx2::get(); // AVX2 uses unsigned u8 quantization
    }

    #[allow(unreachable_code)]
    {
        false // scalar fallback uses signed i8
    }
}

/// SIMD-accelerated batch softmax (unnormalized exp).
#[inline]
pub fn softmax(y: &mut [f32], x: &[f32]) {
    #[cfg(target_arch = "aarch64")]
    {
        unsafe {
            aarch64::softmax_neon(y, x);
        }
        return;
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if cpuid_avx2::get() {
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
