//! 2x downsampler.
//!
//! Upstream c: `silk/resampler_down2.c`

use super::rom::{SILK_RESAMPLER_DOWN2_0, SILK_RESAMPLER_DOWN2_1};
use crate::silk::typedefs::{SILK_INT16_MAX, SILK_INT16_MIN};

/// Upstream c: silk/resampler_down2.c:silk_resampler_down2
pub fn silk_resampler_down2(s: &mut [i32; 2], out: &mut [i16], in_0: &[i16]) {
    debug_assert_eq!(out.len() * 2, in_0.len());

    debug_assert!(SILK_RESAMPLER_DOWN2_0 as i32 > 0);
    debug_assert!((SILK_RESAMPLER_DOWN2_1 as i32) < 0);
    for k in 0..out.len() {
        let in32 = ((in_0[2 * k] as i32 as u32) << 10) as i32;
        let y = in32 - s[0];
        let x = (y as i64 + ((y as i64 * SILK_RESAMPLER_DOWN2_1 as i64) >> 16)) as i32;
        let mut out32 = s[0] + x;
        s[0] = in32 + x;

        let in32 = ((in_0[2 * k + 1] as i32 as u32) << 10) as i32;
        let y = in32 - s[1];
        let x = ((y as i64 * SILK_RESAMPLER_DOWN2_0 as i64) >> 16) as i32;
        out32 += s[1];
        out32 += x;
        s[1] = in32 + x;

        out[k] = (if (if 11 == 1 {
            (out32 >> 1) + (out32 & 1)
        } else {
            ((out32 >> (11 - 1)) + 1) >> 1
        }) > SILK_INT16_MAX
        {
            SILK_INT16_MAX
        } else if (if 11 == 1 {
            (out32 >> 1) + (out32 & 1)
        } else {
            ((out32 >> (11 - 1)) + 1) >> 1
        }) < SILK_INT16_MIN
        {
            SILK_INT16_MIN
        } else if 11 == 1 {
            (out32 >> 1) + (out32 & 1)
        } else {
            ((out32 >> (11 - 1)) + 1) >> 1
        }) as i16;
    }
}
