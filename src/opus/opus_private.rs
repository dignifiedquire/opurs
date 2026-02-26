//! Private structs, codec mode constants, and alignment utilities.
//!
//! Upstream C: `src/opus_private.h`

pub const MODE_SILK_ONLY: i32 = 1000;
pub const MODE_HYBRID: i32 = 1001;
pub const MODE_CELT_ONLY: i32 = 1002;

pub const OPUS_SET_VOICE_RATIO_REQUEST: i32 = 11018;
pub const OPUS_GET_VOICE_RATIO_REQUEST: i32 = 11019;
pub const OPUS_SET_FORCE_MODE_REQUEST: i32 = 11002;

#[inline]
pub fn align(i: i32) -> i32 {
    #[repr(C)]
    union OpusAlign {
        i: i32,
        l: i64,
        f: f32,
        p: *mut core::ffi::c_void,
    }
    let alignment = core::mem::align_of::<OpusAlign>() as u32;
    (i as u32)
        .wrapping_add(alignment)
        .wrapping_sub(1)
        .wrapping_div(alignment)
        .wrapping_mul(alignment) as i32
}
