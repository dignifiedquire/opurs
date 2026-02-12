//! SIMD-accelerated SILK functions.
//!
//! This module provides SIMD dispatch for performance-critical SILK functions.
//! On x86/x86_64, runtime CPU feature detection selects SSE4.1/AVX2 paths.
//! On aarch64, NEON is always available and selected at compile time.
//! On other architectures (or with the `simd` feature disabled), falls through to scalar.

// Dispatch functions are wired up to callers incrementally across phases.
#![allow(dead_code)]

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub mod x86;

#[cfg(target_arch = "aarch64")]
pub mod aarch64;

// -- CPU feature detection (x86/x86_64) --

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
cpufeatures::new!(cpuid_sse4_1, "sse4.1");

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
cpufeatures::new!(cpuid_sse2, "sse2");

// -- Dispatch functions --
// Placeholder dispatchers — implementations are added in later phases.
// For now, all dispatch to scalar.

/// SIMD-accelerated short-term prediction for noise shaping quantizer.
#[inline]
pub fn silk_noise_shape_quantizer_short_prediction(
    buf32: &[i32],
    coef16: &[i16],
    order: i32,
) -> i32 {
    #[cfg(target_arch = "aarch64")]
    {
        return unsafe {
            aarch64::silk_noise_shape_quantizer_short_prediction_neon(buf32, coef16, order)
        };
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if cpuid_sse4_1::get() {
            return unsafe {
                x86::silk_noise_shape_quantizer_short_prediction_sse4_1(buf32, coef16, order)
            };
        }
    }

    #[allow(unreachable_code)]
    {
        super::NSQ::silk_noise_shape_quantizer_short_prediction_c(buf32, coef16, order)
    }
}

/// SIMD-accelerated inner product with scaling for SILK.
#[inline]
pub fn silk_inner_prod_aligned_scale(
    in_vec1: &[i16],
    in_vec2: &[i16],
    scale: i32,
    len: i32,
) -> i32 {
    // Scalar fallback for now — SIMD added in Phase 2
    super::inner_prod_aligned::silk_inner_prod_aligned_scale(in_vec1, in_vec2, scale, len)
}

/// SIMD-accelerated f32→f64 inner product.
#[inline]
pub fn silk_inner_product_flp(data1: &[f32], data2: &[f32]) -> f64 {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if cpuid_sse2::get() {
            return unsafe { x86::silk_inner_product_flp_sse2(data1, data2) };
        }
    }

    #[allow(unreachable_code)]
    {
        super::float::inner_product_FLP::silk_inner_product_FLP_scalar(data1, data2)
    }
}

/// SIMD-accelerated VAD energy accumulation: sum of (X[i] >> 3)^2.
#[inline]
pub fn silk_vad_energy(x: &[i16]) -> i32 {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if cpuid_sse2::get() {
            return unsafe { x86::silk_vad_energy_sse2(x) };
        }
    }

    // Scalar fallback
    #[allow(unreachable_code)]
    {
        silk_vad_energy_scalar(x)
    }
}

/// Scalar implementation of VAD energy accumulation.
fn silk_vad_energy_scalar(x: &[i16]) -> i32 {
    let mut sum: i32 = 0;
    for &sample in x {
        let x_tmp = (sample as i32) >> 3;
        sum += (x_tmp as i16 as i32) * (x_tmp as i16 as i32);
    }
    sum
}
