//! High-quality 2x upsampler.
//!
//! Upstream c: `silk/resampler_private_up2_HQ.c`

use super::rom::{SILK_RESAMPLER_UP2_HQ_0, SILK_RESAMPLER_UP2_HQ_1};
use crate::silk::typedefs::{SILK_INT16_MAX, SILK_INT16_MIN};

#[derive(Default, Copy, Clone)]
pub struct ResamplerUp2HqState {
    iir_state: [i32; 6],
}

/// Upstream c: silk/resampler_private_up2_HQ.c:silk_resampler_private_up2_HQ
/* Upsample by a factor 2, high quality */
/* Uses 2nd Order allpass filters for the 2x upsampling, followed by a      */
/* notch filter just above Nyquist.                                         */
pub fn silk_resampler_private_up2_hq(
    state: &mut ResamplerUp2HqState,
    out: &mut [i16],
    in_0: &[i16],
) {
    debug_assert_eq!(out.len(), 2 * in_0.len());
    debug_assert!(SILK_RESAMPLER_UP2_HQ_0[0] > 0);
    debug_assert!(SILK_RESAMPLER_UP2_HQ_0[1] > 0);
    debug_assert!(SILK_RESAMPLER_UP2_HQ_0[2] < 0);
    debug_assert!(SILK_RESAMPLER_UP2_HQ_1[0] > 0);
    debug_assert!(SILK_RESAMPLER_UP2_HQ_1[1] > 0);
    debug_assert!(SILK_RESAMPLER_UP2_HQ_1[2] < 0);

    let s = &mut state.iir_state;

    /* Internal variables and state are in Q10 format */
    for k in 0..in_0.len() {
        /* Convert to Q10 */
        let in32 = ((in_0[k] as i32 as u32) << 10) as i32;

        /* First all-pass section for even output sample */
        let y = in32 - s[0];
        let x = ((y as i64 * SILK_RESAMPLER_UP2_HQ_0[0] as i64) >> 16) as i32;
        let out32_1 = s[0] + x;
        s[0] = in32 + x;

        /* Second all-pass section for even output sample */
        let y = out32_1 - s[1];
        let x = ((y as i64 * SILK_RESAMPLER_UP2_HQ_0[1] as i64) >> 16) as i32;
        let out32_2 = s[1] + x;
        s[1] = out32_1 + x;

        /* Third all-pass section for even output sample */
        let y = out32_2 - s[2];
        let x = (y as i64 + ((y as i64 * SILK_RESAMPLER_UP2_HQ_0[2] as i64) >> 16)) as i32;
        let out32_1 = s[2] + x;
        s[2] = out32_2 + x;

        /* Apply gain in Q15, convert back to int16 and store to output */
        out[2 * k] = (if (if 10 == 1 {
            (out32_1 >> 1) + (out32_1 & 1)
        } else {
            ((out32_1 >> (10 - 1)) + 1) >> 1
        }) > SILK_INT16_MAX
        {
            SILK_INT16_MAX
        } else if (if 10 == 1 {
            (out32_1 >> 1) + (out32_1 & 1)
        } else {
            ((out32_1 >> (10 - 1)) + 1) >> 1
        }) < SILK_INT16_MIN
        {
            SILK_INT16_MIN
        } else if 10 == 1 {
            (out32_1 >> 1) + (out32_1 & 1)
        } else {
            ((out32_1 >> (10 - 1)) + 1) >> 1
        }) as i16;

        /* First all-pass section for odd output sample */
        let y = in32 - s[3];
        let x = ((y as i64 * SILK_RESAMPLER_UP2_HQ_1[0] as i64) >> 16) as i32;
        let out32_1 = s[3] + x;
        s[3] = in32 + x;

        /* second all-pass section for odd output sample */
        let y = out32_1 - s[4];
        let x = ((y as i64 * SILK_RESAMPLER_UP2_HQ_1[1] as i64) >> 16) as i32;
        let out32_2 = s[4] + x;
        s[4] = out32_1 + x;

        /* Third all-pass section for odd output sample */
        let y = out32_2 - s[5];
        let x = (y as i64 + ((y as i64 * SILK_RESAMPLER_UP2_HQ_1[2] as i64) >> 16)) as i32;
        let out32_1 = s[5] + x;
        s[5] = out32_2 + x;

        /* Apply gain in Q15, convert back to int16 and store to output */
        out[2 * k + 1] = (if (if 10 == 1 {
            (out32_1 >> 1) + (out32_1 & 1)
        } else {
            ((out32_1 >> (10 - 1)) + 1) >> 1
        }) > SILK_INT16_MAX
        {
            SILK_INT16_MAX
        } else if (if 10 == 1 {
            (out32_1 >> 1) + (out32_1 & 1)
        } else {
            ((out32_1 >> (10 - 1)) + 1) >> 1
        }) < SILK_INT16_MIN
        {
            SILK_INT16_MIN
        } else if 10 == 1 {
            (out32_1 >> 1) + (out32_1 & 1)
        } else {
            ((out32_1 >> (10 - 1)) + 1) >> 1
        }) as i16;
    }
}
