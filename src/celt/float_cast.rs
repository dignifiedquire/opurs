//! Float/int conversion constants.
//!
//! Upstream C: `celt/float_cast.h`

/// Upstream C: celt/float_cast.h:CELT_SIG_SCALE
pub const CELT_SIG_SCALE: f32 = 32768.0f32;

/// Upstream C: celt/float_cast.h:FLOAT2INT16
#[inline]
pub fn FLOAT2INT16(x: f32) -> i16 {
    let x = x * CELT_SIG_SCALE;
    let x = x.max(-32768.0);
    let x = x.min(32767.0);
    float2int(x) as i16
}

/// Upstream C: celt/float_cast.h:float2int
///
/// Uses round-to-nearest-even (IEEE 754 default), matching the C implementation
/// which uses `lrintf()` or SSE `cvtss2si` depending on the platform.
#[inline]
pub fn float2int(x: f32) -> i32 {
    x.round_ties_even() as i32
}
