//! Combined IIR/FIR resampler.
//!
//! Upstream c: `silk/resampler_private_IIR_FIR.c`

#![forbid(unsafe_code)]

use crate::silk::resampler::{ResamplerParams, RESAMPLER_MAX_BATCH_SIZE_IN};
use crate::silk::typedefs::{SILK_INT16_MAX, SILK_INT16_MIN};

use super::rom::{RESAMPLER_ORDER_FIR_12, SILK_RESAMPLER_FRAC_FIR_12};
use super::up2_hq::{silk_resampler_private_up2_hq, ResamplerUp2HqState};

#[derive(Default, Copy, Clone)]
pub struct ResamplerIirFirState {
    up2_hq: ResamplerUp2HqState,
    fir_state: [i16; RESAMPLER_ORDER_FIR_12],
}

/// Upstream c: silk/resampler_private_IIR_FIR.c:silk_resampler_private_iir_fir_interpol
#[inline]
fn silk_resampler_private_iir_fir_interpol<'a>(
    mut out: &'a mut [i16],
    buf: &[i16],
    max_index_q16: i32,
    index_increment_q16: i32,
) -> &'a mut [i16] {
    let mut res_q15: i32;

    /* Interpolate upsampled signal and store in output array */
    let mut index_q16 = 0;
    while index_q16 < max_index_q16 {
        let table_index = (((index_q16 & 0xffff) as i64 * 12_i64) >> 16) as usize;
        let buf_ptr = &buf[(index_q16 >> 16) as usize..][..8];

        res_q15 = buf_ptr[0] as i32 * SILK_RESAMPLER_FRAC_FIR_12[table_index][0] as i32;
        res_q15 += buf_ptr[1] as i32 * SILK_RESAMPLER_FRAC_FIR_12[table_index][1] as i32;
        res_q15 += buf_ptr[2] as i32 * SILK_RESAMPLER_FRAC_FIR_12[table_index][2] as i32;
        res_q15 += buf_ptr[3] as i32 * SILK_RESAMPLER_FRAC_FIR_12[table_index][3] as i32;
        res_q15 += buf_ptr[4] as i32 * SILK_RESAMPLER_FRAC_FIR_12[11 - table_index][3] as i32;
        res_q15 += buf_ptr[5] as i32 * SILK_RESAMPLER_FRAC_FIR_12[11 - table_index][2] as i32;
        res_q15 += buf_ptr[6] as i32 * SILK_RESAMPLER_FRAC_FIR_12[11 - table_index][1] as i32;
        res_q15 += buf_ptr[7] as i32 * SILK_RESAMPLER_FRAC_FIR_12[11 - table_index][0] as i32;

        out[0] = (if (if 15 == 1 {
            (res_q15 >> 1) + (res_q15 & 1)
        } else {
            ((res_q15 >> (15 - 1)) + 1) >> 1
        }) > SILK_INT16_MAX
        {
            SILK_INT16_MAX
        } else if (if 15 == 1 {
            (res_q15 >> 1) + (res_q15 & 1)
        } else {
            ((res_q15 >> (15 - 1)) + 1) >> 1
        }) < SILK_INT16_MIN
        {
            SILK_INT16_MIN
        } else if 15 == 1 {
            (res_q15 >> 1) + (res_q15 & 1)
        } else {
            ((res_q15 >> (15 - 1)) + 1) >> 1
        }) as i16;

        out = &mut out[1..];

        index_q16 += index_increment_q16;
    }

    out
}

/// Upstream c: silk/resampler_private_IIR_FIR.c:silk_resampler_private_iir_fir
/* Upsample using a combination of allpass-based 2x upsampling and FIR interpolation */
pub(super) fn silk_resampler_private_iir_fir(
    resampler_params: &ResamplerParams,
    state: &mut ResamplerIirFirState,
    mut out: &mut [i16],
    mut in_0: &[i16],
) {
    let mut n_samples_in: usize;
    let mut max_index_q16: i32;
    // Max: 2 * batch_size(480) + 8 = 968
    let mut buf = [0i16; 2 * RESAMPLER_MAX_BATCH_SIZE_IN + RESAMPLER_ORDER_FIR_12];

    /* Copy buffered samples to start of buffer */
    buf[0..RESAMPLER_ORDER_FIR_12].copy_from_slice(&state.fir_state);

    /* Iterate over blocks of frameSizeIn input samples */
    let index_increment_q16 = resampler_params.inv_ratio_q16;
    loop {
        n_samples_in = in_0.len().min(resampler_params.batch_size);
        silk_resampler_private_up2_hq(
            &mut state.up2_hq,
            &mut buf[RESAMPLER_ORDER_FIR_12..][..n_samples_in * 2],
            &in_0[..n_samples_in],
        );
        max_index_q16 = ((n_samples_in as u32) << (16 + 1)) as i32; /* + 1 because 2x upsampling */
        out =
            silk_resampler_private_iir_fir_interpol(out, &buf, max_index_q16, index_increment_q16);
        in_0 = &in_0[n_samples_in..];

        if in_0.is_empty() {
            break;
        }

        /* More iterations to do; copy last part of filtered signal to beginning of buffer */
        buf.copy_within(
            n_samples_in * 2..n_samples_in * 2 + RESAMPLER_ORDER_FIR_12,
            0,
        );
    }

    /* Copy last part of filtered signal to the state for the next call */
    state
        .fir_state
        .copy_from_slice(&buf[n_samples_in * 2..][..RESAMPLER_ORDER_FIR_12]);
}
