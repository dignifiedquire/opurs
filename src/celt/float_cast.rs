//! Float/int conversion constants.
//!
//! Upstream C: `celt/float_cast.h`

/// Upstream C: celt/float_cast.h:CELT_SIG_SCALE
pub const CELT_SIG_SCALE: f32 = 32768.0f32;

///
/// Per-sample scalar conversion. On all platforms, C uses `float2int` which is
/// ties-to-even (aarch64: `vcvtns_s32_f32`, x86: `cvtss2si`, or `lrintf`).
/// Upstream C: celt/float_cast.h:FLOAT2INT16
#[inline]
pub fn FLOAT2INT16(x: f32) -> i16 {
    let x = x * CELT_SIG_SCALE;
    let x = x.max(-32768.0);
    let x = x.min(32767.0);
    float2int(x) as i16
}

/// Batch float-to-int16 conversion matching the C `celt_float2int16` function.
///
/// On aarch64 with NEON, C processes blocks of 16 using `vcvtaq_s32_f32`
/// (ties away from zero), with the tail using scalar `FLOAT2INT16` (ties to even).
/// On x86, C uses scalar `FLOAT2INT16` for all samples (ties to even).
#[inline]
pub fn celt_float2int16(input: &[f32], output: &mut [i16], cnt: usize) {
    let mut i = 0;

    #[cfg(target_arch = "aarch64")]
    {
        // Match celt_float2int16_neon: process blocks of 16 with ties-away rounding
        let block_end = cnt / 16 * 16;
        while i < block_end {
            for j in 0..16 {
                let x = input[i + j] * CELT_SIG_SCALE;
                let x = x.max(-32768.0);
                let x = x.min(32767.0);
                // vcvtaq_s32_f32 = round ties away from zero
                output[i + j] = x.round() as i32 as i16;
            }
            i += 16;
        }
    }

    // Scalar tail (or all samples on non-aarch64)
    while i < cnt {
        output[i] = FLOAT2INT16(input[i]);
        i += 1;
    }
}

///
/// Matches upstream conversion semantics by target:
/// - x86/x86_64: use SSE `cvtss2si` (honors current MXCSR rounding mode)
/// - other targets: round-to-nearest-even
///
/// Upstream C: celt/float_cast.h:float2int
#[inline]
pub fn float2int(x: f32) -> i32 {
    float2int_impl(x)
}

#[cfg(target_arch = "x86_64")]
#[inline]
fn float2int_impl(x: f32) -> i32 {
    unsafe {
        use core::arch::x86_64::{_mm_cvtss_si32, _mm_set_ss};
        _mm_cvtss_si32(_mm_set_ss(x))
    }
}

#[cfg(all(target_arch = "x86", target_feature = "sse"))]
#[inline]
fn float2int_impl(x: f32) -> i32 {
    unsafe {
        use core::arch::x86::{_mm_cvtss_si32, _mm_set_ss};
        _mm_cvtss_si32(_mm_set_ss(x))
    }
}

#[cfg(target_arch = "aarch64")]
#[inline]
fn float2int_impl(x: f32) -> i32 {
    unsafe {
        use core::arch::aarch64::vcvtns_s32_f32;
        vcvtns_s32_f32(x)
    }
}

#[cfg(not(any(
    target_arch = "x86_64",
    all(target_arch = "x86", target_feature = "sse"),
    target_arch = "aarch64"
)))]
#[inline]
fn float2int_impl(x: f32) -> i32 {
    x.round_ties_even() as i32
}
