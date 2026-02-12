//! SIMD-accelerated CELT functions.
//!
//! This module provides SIMD dispatch for performance-critical CELT functions.
//! On x86/x86_64, runtime CPU feature detection selects SSE/SSE2/SSE4.1/AVX2 paths.
//! On aarch64, NEON is always available and selected at compile time.
//! On other architectures (or with the `simd` feature disabled), falls through to scalar.

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub mod x86;

#[cfg(target_arch = "aarch64")]
pub mod aarch64;

// -- CPU feature detection (x86/x86_64) --

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
cpufeatures::new!(cpuid_sse, "sse");

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
cpufeatures::new!(cpuid_sse2, "sse2");

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
cpufeatures::new!(cpuid_sse4_1, "sse4.1");

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
cpufeatures::new!(cpuid_avx2, "avx2");

// -- Dispatch functions --
// Each function selects the best available SIMD implementation at runtime,
// falling back to the scalar version.

/// SIMD-accelerated 4-way cross-correlation kernel.
/// Dispatches to SSE on x86 or NEON on aarch64, with scalar fallback.
#[inline]
pub fn xcorr_kernel(x: &[f32], y: &[f32], sum: &mut [f32; 4], len: usize) {
    #[cfg(target_arch = "aarch64")]
    {
        // NEON is always available on aarch64
        unsafe {
            aarch64::xcorr_kernel_neon(x, y, sum, len);
        }
        return;
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if cpuid_sse::get() {
            unsafe {
                x86::xcorr_kernel_sse(x, y, sum, len);
            }
            return;
        }
    }

    #[allow(unreachable_code)]
    {
        super::pitch::xcorr_kernel_scalar(x, y, sum, len);
    }
}

/// SIMD-accelerated inner product.
/// Dispatches to SSE on x86 or NEON on aarch64, with scalar fallback.
#[inline]
pub fn celt_inner_prod(x: &[f32], y: &[f32], n: usize) -> f32 {
    #[cfg(target_arch = "aarch64")]
    {
        return unsafe { aarch64::celt_inner_prod_neon(x, y, n) };
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if cpuid_sse::get() {
            return unsafe { x86::celt_inner_prod_sse(x, y, n) };
        }
    }

    #[allow(unreachable_code)]
    {
        super::pitch::celt_inner_prod_scalar(x, y, n)
    }
}

/// SIMD-accelerated dual inner product.
/// Dispatches to SSE on x86 or NEON on aarch64, with scalar fallback.
#[inline]
pub fn dual_inner_prod(x: &[f32], y01: &[f32], y02: &[f32], n: usize) -> (f32, f32) {
    #[cfg(target_arch = "aarch64")]
    {
        return unsafe { aarch64::dual_inner_prod_neon(x, y01, y02, n) };
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if cpuid_sse::get() {
            return unsafe { x86::dual_inner_prod_sse(x, y01, y02, n) };
        }
    }

    #[allow(unreachable_code)]
    {
        super::pitch::dual_inner_prod_scalar(x, y01, y02, n)
    }
}

/// SIMD-accelerated pitch cross-correlation.
/// Dispatches to SSE on x86 or NEON on aarch64, with scalar fallback.
#[inline]
pub fn celt_pitch_xcorr(x: &[f32], y: &[f32], xcorr: &mut [f32], len: usize) {
    #[cfg(target_arch = "aarch64")]
    {
        unsafe {
            aarch64::celt_pitch_xcorr_neon(x, y, xcorr, len);
        }
        return;
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if cpuid_avx2::get() {
            unsafe {
                x86::celt_pitch_xcorr_avx2(x, y, xcorr, len);
            }
            return;
        }
        if cpuid_sse::get() {
            unsafe {
                x86::celt_pitch_xcorr_sse(x, y, xcorr, len);
            }
            return;
        }
    }

    #[allow(unreachable_code)]
    {
        super::pitch::celt_pitch_xcorr_scalar(x, y, xcorr, len);
    }
}
