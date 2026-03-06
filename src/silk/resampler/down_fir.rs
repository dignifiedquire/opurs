//! FIR downsampling filter.
//!
//! Upstream c: `silk/resampler_private_down_FIR.c`

use super::ar2::silk_resampler_private_ar2;
use super::rom::{RESAMPLER_DOWN_ORDER_FIR0, RESAMPLER_DOWN_ORDER_FIR1, RESAMPLER_DOWN_ORDER_FIR2};
use crate::silk::resampler::{
    ResamplerParams, RESAMPLER_MAX_BATCH_SIZE_IN, SILK_RESAMPLER_MAX_FIR_ORDER,
};
use crate::silk::typedefs::{SILK_INT16_MAX, SILK_INT16_MIN};

#[derive(Copy, Clone)]
pub struct ResamplerDownFirParams {
    pub fir_order: usize,
    pub fir_fracs: i32,
    pub coefs: &'static [i16],
}

#[derive(Copy, Clone)]
pub struct ResamplerDownFirState {
    ar2_state: [i32; 2],
    fir_state: [i32; SILK_RESAMPLER_MAX_FIR_ORDER],
}

// can't derive Default because of the array size
impl Default for ResamplerDownFirState {
    fn default() -> Self {
        ResamplerDownFirState {
            ar2_state: [0; 2],
            fir_state: [0; SILK_RESAMPLER_MAX_FIR_ORDER],
        }
    }
}

/// Upstream c: silk/resampler_private_down_FIR.c:silk_resampler_private_down_fir_interpol
#[inline]
fn silk_resampler_private_down_fir_interpol<'a>(
    mut out: &'a mut [i16],
    buf: &[i32],
    fir_coefs: &[i16],
    fir_order: usize,
    fir_fracs: i32,
    max_index_q16: i32,
    index_increment_q16: i32,
) -> &'a mut [i16] {
    match fir_order {
        RESAMPLER_DOWN_ORDER_FIR0 => {
            let mut index_q16 = 0;
            while index_q16 < max_index_q16 {
                /* Integer part gives pointer to buffered input */
                let buf_ptr = &buf[(index_q16 >> 16) as usize..];

                /* Fractional part gives interpolation coefficients */
                let interpol_ind =
                    (((index_q16 & 0xffff) as i64 * fir_fracs as i16 as i64) >> 16) as usize;

                /* Inner product */
                let interpol_ptr = &fir_coefs[(RESAMPLER_DOWN_ORDER_FIR0 / 2 * interpol_ind)..];
                let mut res_q6 = ((buf_ptr[0] as i64 * interpol_ptr[0] as i64) >> 16) as i32;
                res_q6 += ((buf_ptr[1] as i64 * interpol_ptr[1] as i64) >> 16) as i32;
                res_q6 += ((buf_ptr[2] as i64 * interpol_ptr[2] as i64) >> 16) as i32;
                res_q6 += ((buf_ptr[3] as i64 * interpol_ptr[3] as i64) >> 16) as i32;
                res_q6 += ((buf_ptr[4] as i64 * interpol_ptr[4] as i64) >> 16) as i32;
                res_q6 += ((buf_ptr[5] as i64 * interpol_ptr[5] as i64) >> 16) as i32;
                res_q6 += ((buf_ptr[6] as i64 * interpol_ptr[6] as i64) >> 16) as i32;
                res_q6 += ((buf_ptr[7] as i64 * interpol_ptr[7] as i64) >> 16) as i32;
                res_q6 += ((buf_ptr[8] as i64 * interpol_ptr[8] as i64) >> 16) as i32;

                let interpol_ptr = &fir_coefs
                    [(RESAMPLER_DOWN_ORDER_FIR0 / 2 * (fir_fracs as usize - 1 - interpol_ind))..];
                res_q6 += ((buf_ptr[17] as i64 * interpol_ptr[0] as i64) >> 16) as i32;
                res_q6 += ((buf_ptr[16] as i64 * interpol_ptr[1] as i64) >> 16) as i32;
                res_q6 += ((buf_ptr[15] as i64 * interpol_ptr[2] as i64) >> 16) as i32;
                res_q6 += ((buf_ptr[14] as i64 * interpol_ptr[3] as i64) >> 16) as i32;
                res_q6 += ((buf_ptr[13] as i64 * interpol_ptr[4] as i64) >> 16) as i32;
                res_q6 += ((buf_ptr[12] as i64 * interpol_ptr[5] as i64) >> 16) as i32;
                res_q6 += ((buf_ptr[11] as i64 * interpol_ptr[6] as i64) >> 16) as i32;
                res_q6 += ((buf_ptr[10] as i64 * interpol_ptr[7] as i64) >> 16) as i32;
                res_q6 += ((buf_ptr[9] as i64 * interpol_ptr[8] as i64) >> 16) as i32;

                /* Scale down, saturate and store in output array */
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

                index_q16 += index_increment_q16;
            }
        }
        RESAMPLER_DOWN_ORDER_FIR1 => {
            let mut index_q16 = 0;
            while index_q16 < max_index_q16 {
                /* Integer part gives pointer to buffered input */
                let buf_ptr = &buf[(index_q16 >> 16) as usize..];

                let mut res_q6 =
                    (((buf_ptr[0] + buf_ptr[23]) as i64 * fir_coefs[0] as i64) >> 16) as i32;
                res_q6 += (((buf_ptr[1] + buf_ptr[22]) as i64 * fir_coefs[1] as i64) >> 16) as i32;
                res_q6 += (((buf_ptr[2] + buf_ptr[21]) as i64 * fir_coefs[2] as i64) >> 16) as i32;
                res_q6 += (((buf_ptr[3] + buf_ptr[20]) as i64 * fir_coefs[3] as i64) >> 16) as i32;
                res_q6 += (((buf_ptr[4] + buf_ptr[19]) as i64 * fir_coefs[4] as i64) >> 16) as i32;
                res_q6 += (((buf_ptr[5] + buf_ptr[18]) as i64 * fir_coefs[5] as i64) >> 16) as i32;
                res_q6 += (((buf_ptr[6] + buf_ptr[17]) as i64 * fir_coefs[6] as i64) >> 16) as i32;
                res_q6 += (((buf_ptr[7] + buf_ptr[16]) as i64 * fir_coefs[7] as i64) >> 16) as i32;
                res_q6 += (((buf_ptr[8] + buf_ptr[15]) as i64 * fir_coefs[8] as i64) >> 16) as i32;
                res_q6 += (((buf_ptr[9] + buf_ptr[14]) as i64 * fir_coefs[9] as i64) >> 16) as i32;
                res_q6 +=
                    (((buf_ptr[10] + buf_ptr[13]) as i64 * fir_coefs[10] as i64) >> 16) as i32;
                res_q6 +=
                    (((buf_ptr[11] + buf_ptr[12]) as i64 * fir_coefs[11] as i64) >> 16) as i32;

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

                index_q16 += index_increment_q16;
            }
        }
        RESAMPLER_DOWN_ORDER_FIR2 => {
            let mut index_q16 = 0;
            while index_q16 < max_index_q16 {
                /* Integer part gives pointer to buffered input */
                let buf_ptr = &buf[(index_q16 >> 16) as usize..];

                let mut res_q6 =
                    (((buf_ptr[0] + buf_ptr[35]) as i64 * fir_coefs[0] as i64) >> 16) as i32;
                res_q6 += (((buf_ptr[1] + buf_ptr[34]) as i64 * fir_coefs[1] as i64) >> 16) as i32;
                res_q6 += (((buf_ptr[2] + buf_ptr[33]) as i64 * fir_coefs[2] as i64) >> 16) as i32;
                res_q6 += (((buf_ptr[3] + buf_ptr[32]) as i64 * fir_coefs[3] as i64) >> 16) as i32;
                res_q6 += (((buf_ptr[4] + buf_ptr[31]) as i64 * fir_coefs[4] as i64) >> 16) as i32;
                res_q6 += (((buf_ptr[5] + buf_ptr[30]) as i64 * fir_coefs[5] as i64) >> 16) as i32;
                res_q6 += (((buf_ptr[6] + buf_ptr[29]) as i64 * fir_coefs[6] as i64) >> 16) as i32;
                res_q6 += (((buf_ptr[7] + buf_ptr[28]) as i64 * fir_coefs[7] as i64) >> 16) as i32;
                res_q6 += (((buf_ptr[8] + buf_ptr[27]) as i64 * fir_coefs[8] as i64) >> 16) as i32;
                res_q6 += (((buf_ptr[9] + buf_ptr[26]) as i64 * fir_coefs[9] as i64) >> 16) as i32;
                res_q6 +=
                    (((buf_ptr[10] + buf_ptr[25]) as i64 * fir_coefs[10] as i64) >> 16) as i32;
                res_q6 +=
                    (((buf_ptr[11] + buf_ptr[24]) as i64 * fir_coefs[11] as i64) >> 16) as i32;
                res_q6 +=
                    (((buf_ptr[12] + buf_ptr[23]) as i64 * fir_coefs[12] as i64) >> 16) as i32;
                res_q6 +=
                    (((buf_ptr[13] + buf_ptr[22]) as i64 * fir_coefs[13] as i64) >> 16) as i32;
                res_q6 +=
                    (((buf_ptr[14] + buf_ptr[21]) as i64 * fir_coefs[14] as i64) >> 16) as i32;
                res_q6 +=
                    (((buf_ptr[15] + buf_ptr[20]) as i64 * fir_coefs[15] as i64) >> 16) as i32;
                res_q6 +=
                    (((buf_ptr[16] + buf_ptr[19]) as i64 * fir_coefs[16] as i64) >> 16) as i32;
                res_q6 +=
                    (((buf_ptr[17] + buf_ptr[18]) as i64 * fir_coefs[17] as i64) >> 16) as i32;

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

                index_q16 += index_increment_q16;
            }
        }
        _ => {
            debug_assert!(false, "libopus: assert(0) called");
            return out;
        }
    }

    out
}

/// Upstream c: silk/resampler_private_down_FIR.c:silk_resampler_private_down_fir
pub(super) fn silk_resampler_private_down_fir(
    resampler_params: &ResamplerParams,
    params: &ResamplerDownFirParams,
    state: &mut ResamplerDownFirState,
    mut out: &mut [i16],
    mut in_0: &[i16],
) {
    let mut n_samples_in: usize;

    // Max: batch_size(480) + fir_order(36) = 516
    let mut buf = [0i32; RESAMPLER_MAX_BATCH_SIZE_IN + SILK_RESAMPLER_MAX_FIR_ORDER];

    buf[..params.fir_order].copy_from_slice(&state.fir_state[..params.fir_order]);

    let index_increment_q16 = resampler_params.inv_ratio_q16;
    loop {
        n_samples_in = in_0.len().min(resampler_params.batch_size);
        silk_resampler_private_ar2(
            &mut state.ar2_state,
            &mut buf[params.fir_order..][..n_samples_in],
            &in_0[..n_samples_in],
            &params.coefs[..2],
        );
        let max_index_q16 = ((n_samples_in as u32) << 16) as i32;
        out = silk_resampler_private_down_fir_interpol(
            out,
            &buf,
            &params.coefs[2..],
            params.fir_order,
            params.fir_fracs,
            max_index_q16,
            index_increment_q16,
        );
        in_0 = &in_0[n_samples_in..];
        if in_0.is_empty() {
            break;
        }

        buf.copy_within(n_samples_in..n_samples_in + params.fir_order, 0);
    }
    state.fir_state[..params.fir_order]
        .copy_from_slice(&buf[n_samples_in..n_samples_in + params.fir_order]);
}
