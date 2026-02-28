//! SIMD-accelerated CELT functions.
//!
//! This module provides SIMD dispatch for performance-critical CELT functions.
//! On x86/x86_64, the `Arch` enum selects SSE/SSE2/SSE4.1/AVX2 paths.
//! On aarch64, NEON is selected when arch indicates it.
//! On other architectures (or with the `simd` feature disabled), falls through to scalar.

use crate::arch::Arch;

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub mod x86;

#[cfg(target_arch = "aarch64")]
pub mod aarch64;

// -- Dispatch functions --
// Each function dispatches based on the `arch` parameter, which was detected
// once at encoder/decoder init via `opus_select_arch()`.

/// SIMD-accelerated 4-way cross-correlation kernel.
/// Dispatches to SSE on x86 (arch >= SSE), with scalar fallback.
///
/// On aarch64, the C reference only uses NEON for `celt_pitch_xcorr`, not for
/// the standalone `xcorr_kernel` (called from celt_lpc). The NEON version does
/// NOT accumulate into `sum` (starts from zero), while the scalar version does.
/// To match C behavior, we use scalar on aarch64 for this function.
#[inline(always)]
pub fn xcorr_kernel(x: &[f32], y: &[f32], sum: &mut [f32; 4], len: usize, arch: Arch) {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if arch.has_sse() {
        return x86::xcorr_kernel_sse_dispatch(x, y, sum, len);
    }

    let _ = arch;
    super::pitch::xcorr_kernel_scalar(x, y, sum, len);
}

/// SIMD-accelerated inner product.
/// Dispatches to SSE on x86 or NEON on aarch64, with scalar fallback.
#[inline(always)]
pub fn celt_inner_prod(x: &[f32], y: &[f32], n: usize, arch: Arch) -> f32 {
    #[cfg(target_arch = "aarch64")]
    if arch.has_neon() {
        return aarch64::celt_inner_prod_neon_dispatch(x, y, n);
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if arch.has_sse() {
        return x86::celt_inner_prod_sse_dispatch(x, y, n);
    }

    let _ = arch;
    super::pitch::celt_inner_prod_scalar(x, y, n)
}

/// SIMD-accelerated dual inner product.
/// Dispatches to SSE on x86 or NEON on aarch64, with scalar fallback.
#[inline(always)]
pub fn dual_inner_prod(x: &[f32], y01: &[f32], y02: &[f32], n: usize, arch: Arch) -> (f32, f32) {
    #[cfg(target_arch = "aarch64")]
    if arch.has_neon() {
        return aarch64::dual_inner_prod_neon_dispatch(x, y01, y02, n);
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if arch.has_sse() {
        return x86::dual_inner_prod_sse_dispatch(x, y01, y02, n);
    }

    let _ = arch;
    super::pitch::dual_inner_prod_scalar(x, y01, y02, n)
}

/// SIMD-accelerated pitch cross-correlation.
/// Dispatches to AVX2 on x86 or NEON on aarch64, with scalar fallback.
///
/// Upstream x86 RTCD maps `celt_pitch_xcorr` to scalar for non-AVX2 arches,
/// unlike other pitch helpers that use SSE. Keep this behavior for parity.
#[inline(always)]
pub fn celt_pitch_xcorr(x: &[f32], y: &[f32], xcorr: &mut [f32], len: usize, arch: Arch) {
    #[cfg(target_arch = "aarch64")]
    if arch.has_neon() {
        aarch64::celt_pitch_xcorr_neon_dispatch(x, y, xcorr, len);
        return;
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if arch.has_avx2() {
        return x86::celt_pitch_xcorr_avx2_dispatch(x, y, xcorr, len);
    }

    let _ = arch;
    super::pitch::celt_pitch_xcorr_scalar(x, y, xcorr, len);
}

/// SIMD-accelerated constant-coefficient comb filter.
/// Dispatches to SSE on x86, with scalar fallback.
#[inline(always)]
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
    arch: Arch,
) {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if arch.has_sse() {
        return x86::comb_filter_const_sse_dispatch(y, y_start, x, x_start, T, N, g10, g11, g12);
    }

    let _ = arch;
    super::common::comb_filter_const_c(y, y_start, x, x_start, T, N, g10, g11, g12);
}

/// SIMD-accelerated constant-coefficient in-place comb filter.
/// Dispatches to SSE on x86, with scalar fallback.
#[inline(always)]
pub fn comb_filter_const_inplace(
    buf: &mut [f32],
    start: usize,
    T: i32,
    N: i32,
    g10: f32,
    g11: f32,
    g12: f32,
    arch: Arch,
) {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if arch.has_sse() {
        return x86::comb_filter_const_inplace_sse_dispatch(buf, start, T, N, g10, g11, g12);
    }

    let _ = arch;
    let t = T as usize;
    let n = N as usize;
    let mut x4 = buf[start - t - 2];
    let mut x3 = buf[start - t - 1];
    let mut x2 = buf[start - t];
    let mut x1 = buf[start - t + 1];
    for i in 0..n {
        let x0 = buf[start + i - t + 2];
        buf[start + i] = buf[start + i] + g10 * x2 + g11 * (x1 + x3) + g12 * (x0 + x4);
        x4 = x3;
        x3 = x2;
        x2 = x1;
        x1 = x0;
    }
}

/// SIMD-accelerated PVQ search.
/// Dispatches to SSE2 on x86, with scalar fallback.
/// The SSE2 version handles any N by zero-padding arrays to N+3 elements,
/// matching C which always uses SSE2 at arch >= 2 regardless of alignment.
#[inline(always)]
pub fn op_pvq_search(X: &mut [f32], iy: &mut [i32], K: i32, N: i32, arch: Arch) -> f32 {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if arch.has_sse2() {
        return x86::op_pvq_search_sse2_dispatch(X, iy, K, N);
    }

    let _ = arch;
    super::vq::op_pvq_search_c(X, iy, K, N, arch)
}
