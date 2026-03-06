//! 2/3 rate downsampler.
//!
//! Upstream c: `silk/resampler_down2_3.c`

use crate::silk::typedefs::{SILK_INT16_MAX, SILK_INT16_MIN};
use arrayref::array_mut_ref;

use super::ar2::silk_resampler_private_ar2;
use super::rom::SILK_RESAMPLER_2_3_COEFS_LQ;
use crate::silk::resampler::RESAMPLER_MAX_BATCH_SIZE_IN;

const ORDER_FIR: usize = 4;

///
/// Downsample by a factor 2/3, low quality
/// Upstream c: silk/resampler_down2_3.c:silk_resampler_down2_3
pub fn silk_resampler_down2_3(state: &mut [i32; 6], mut out: &mut [i16], mut in_0: &[i16]) {
    let mut n_samples_in: usize;
    let mut res_q6: i32;
    let mut buf: [i32; RESAMPLER_MAX_BATCH_SIZE_IN + ORDER_FIR] =
        [0; RESAMPLER_MAX_BATCH_SIZE_IN + ORDER_FIR];

    let s = state;

    buf[..ORDER_FIR].copy_from_slice(&s[..ORDER_FIR]);

    loop {
        n_samples_in = in_0.len().min(RESAMPLER_MAX_BATCH_SIZE_IN);
        silk_resampler_private_ar2(
            array_mut_ref![s, ORDER_FIR, 2],
            &mut buf[ORDER_FIR..][..n_samples_in],
            &in_0[..n_samples_in],
            &SILK_RESAMPLER_2_3_COEFS_LQ,
        );
        let mut buf_ptr = buf.as_mut_slice();
        let mut counter = n_samples_in;
        while counter > 2 {
            res_q6 = ((buf_ptr[0] as i64 * SILK_RESAMPLER_2_3_COEFS_LQ[2] as i64) >> 16) as i32;
            res_q6 = (res_q6 as i64
                + ((buf_ptr[1] as i64 * SILK_RESAMPLER_2_3_COEFS_LQ[3] as i64) >> 16))
                as i32;
            res_q6 = (res_q6 as i64
                + ((buf_ptr[2] as i64 * SILK_RESAMPLER_2_3_COEFS_LQ[5] as i64) >> 16))
                as i32;
            res_q6 = (res_q6 as i64
                + ((buf_ptr[3] as i64 * SILK_RESAMPLER_2_3_COEFS_LQ[4] as i64) >> 16))
                as i32;

            out[0] = (if (if 6 == 1 {
                (res_q6 >> 1) + (res_q6 & 1)
            } else {
                ((res_q6 >> (6 - 1)) + 1) >> 1
            }) > SILK_INT16_MAX
            {
                SILK_INT16_MAX
            } else if (if 6 == 1 {
                (res_q6 >> 1) + (res_q6 & 1)
            } else {
                ((res_q6 >> (6 - 1)) + 1) >> 1
            }) < SILK_INT16_MIN
            {
                SILK_INT16_MIN
            } else if 6 == 1 {
                (res_q6 >> 1) + (res_q6 & 1)
            } else {
                ((res_q6 >> (6 - 1)) + 1) >> 1
            }) as i16;
            out = &mut out[1..];

            res_q6 = ((buf_ptr[1] as i64 * SILK_RESAMPLER_2_3_COEFS_LQ[4] as i64) >> 16) as i32;
            res_q6 = (res_q6 as i64
                + ((buf_ptr[2] as i64 * SILK_RESAMPLER_2_3_COEFS_LQ[5] as i64) >> 16))
                as i32;
            res_q6 = (res_q6 as i64
                + ((buf_ptr[3] as i64 * SILK_RESAMPLER_2_3_COEFS_LQ[3] as i64) >> 16))
                as i32;
            res_q6 = (res_q6 as i64
                + ((buf_ptr[4] as i64 * SILK_RESAMPLER_2_3_COEFS_LQ[2] as i64) >> 16))
                as i32;

            out[0] = (if (if 6 == 1 {
                (res_q6 >> 1) + (res_q6 & 1)
            } else {
                ((res_q6 >> (6 - 1)) + 1) >> 1
            }) > SILK_INT16_MAX
            {
                SILK_INT16_MAX
            } else if (if 6 == 1 {
                (res_q6 >> 1) + (res_q6 & 1)
            } else {
                ((res_q6 >> (6 - 1)) + 1) >> 1
            }) < SILK_INT16_MIN
            {
                SILK_INT16_MIN
            } else if 6 == 1 {
                (res_q6 >> 1) + (res_q6 & 1)
            } else {
                ((res_q6 >> (6 - 1)) + 1) >> 1
            }) as i16;
            out = &mut out[1..];

            buf_ptr = &mut buf_ptr[3..];
            counter -= 3;
        }

        in_0 = &in_0[n_samples_in..];
        if in_0.is_empty() {
            break;
        }

        buf.copy_within(n_samples_in..n_samples_in + ORDER_FIR, 0);
    }

    s[..ORDER_FIR].copy_from_slice(&buf[n_samples_in..][..ORDER_FIR]);
}
