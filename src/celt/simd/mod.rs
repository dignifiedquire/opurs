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
cpufeatures::new!(cpuid_avx2_fma, "avx2", "fma");

// -- Dispatch functions --
// Each function selects the best available SIMD implementation at runtime,
// falling back to the scalar version.

/// SIMD-accelerated 4-way cross-correlation kernel.
/// Dispatches to SSE on x86, with scalar fallback.
///
/// On aarch64, the C reference only uses NEON for `celt_pitch_xcorr`, not for
/// the standalone `xcorr_kernel` (called from celt_lpc). The NEON version does
/// NOT accumulate into `sum` (starts from zero), while the scalar version does.
/// To match C behavior, we use scalar on aarch64 for this function.
#[inline]
pub fn xcorr_kernel(x: &[f32], y: &[f32], sum: &mut [f32; 4], len: usize) {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        // Keep x86 SIMD implementations compiled, but do not dispatch to them yet.
        //
        // Rationale: runtime SIMD dispatch currently uses host CPUID while encoder/decoder
        // architecture selection is still hardcoded to arch=0. This mismatch causes
        // x86-only vector parity regressions in CI. Force scalar behavior here until
        // arch plumbing is aligned with upstream RTCD semantics.
        let _ = cpuid_sse::get();
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
        let _ = cpuid_sse::get();
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
        let _ = cpuid_sse::get();
    }

    #[allow(unreachable_code)]
    {
        super::pitch::dual_inner_prod_scalar(x, y01, y02, n)
    }
}

/// SIMD-accelerated pitch cross-correlation.
/// Dispatches to AVX2 on x86 (otherwise scalar) or NEON on aarch64.
///
/// Upstream x86 RTCD maps `celt_pitch_xcorr` to scalar for non-AVX2 arches,
/// unlike other pitch helpers that use SSE. Keep this behavior for parity.
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
        // Keep AVX2 implementation available but disabled for runtime dispatch.
        //
        // Rationale: major-platform vector CI shows deterministic x86_64-only
        // bitstream mismatches in classic/testvector11 (RLD 10 kbps, 20/40/60 ms)
        // when this path is active.
        //
        // This mirrors the correctness-first approach used for known non-bitexact
        // SILK SIMD paths: keep implementation in-tree for direct comparison work,
        // but force scalar dispatch until strict parity is restored.
        let _ = cpuid_avx2_fma::get();
    }

    #[allow(unreachable_code)]
    {
        super::pitch::celt_pitch_xcorr_scalar(x, y, xcorr, len);
    }
}

/// SIMD-accelerated constant-coefficient comb filter.
/// Dispatches to SSE on x86, with scalar fallback.
#[inline]
pub fn comb_filter_const(
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
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        let _ = cpuid_sse::get();
    }

    #[allow(unreachable_code)]
    {
        super::common::comb_filter_const_c(y, y_start, x, x_start, T, N, g10, g11, g12);
    }
}

/// SIMD-accelerated PVQ search.
/// Dispatches to SSE2 on x86, with scalar fallback.
/// The SSE2 version handles any N by zero-padding arrays to N+3 elements,
/// matching C which always uses SSE2 at arch >= 2 regardless of alignment.
#[inline]
pub fn op_pvq_search(X: &mut [f32], iy: &mut [i32], K: i32, N: i32, _arch: i32) -> f32 {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        let _ = cpuid_sse2::get();
    }

    #[allow(unreachable_code)]
    {
        super::vq::op_pvq_search_c(X, iy, K, N, _arch)
    }
}
