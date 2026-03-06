//! LTP gain quantization.
//!
//! Upstream c: `silk/quant_LTP_gains.c`

use crate::arch::Arch;

use crate::silk::define::LTP_ORDER;
use crate::silk::lin2log::silk_lin2log;
use crate::silk::log2lin::silk_log2lin;
#[cfg(feature = "simd")]
use crate::silk::simd::silk_vq_wmat_ec;
use crate::silk::tables_ltp::{
    SILK_LTP_GAIN_BITS_Q5_PTRS, SILK_LTP_VQ_GAIN_PTRS_Q7, SILK_LTP_VQ_PTRS_Q7, SILK_LTP_VQ_SIZES,
};
use crate::silk::tuning_parameters::MAX_SUM_LOG_GAIN_DB;
use crate::silk::typedefs::SILK_INT32_MAX;
#[cfg(not(feature = "simd"))]
use crate::silk::vq_wmat_ec::{silk_vq_wmat_ec_c, SilkVqWmatEcParams};

/// Upstream c: silk/quant_LTP_gains.c:silk_quant_LTP_gains
#[allow(clippy::too_many_arguments)]
pub fn silk_quant_ltp_gains(
    b_q14: &mut [i16],
    cbk_index: &mut [i8],
    periodicity_index: &mut i8,
    sum_log_gain_q7: &mut i32,
    pred_gain_d_b_q7: &mut i32,
    xx_q17: &[i32],
    x_x_q17: &[i32],
    subfr_len: i32,
    nb_subfr: i32,
    _arch: Arch,
) {
    let mut j: i32;
    let mut k: i32;
    let mut cbk_size: i32;
    let mut temp_idx: [i8; 4] = [0; 4];
    let mut res_nrg_q15: i32 = 0;
    let mut rate_dist_q7: i32;
    let mut min_rate_dist_q7: i32;
    let mut sum_log_gain_tmp_q7: i32;
    let mut best_sum_log_gain_q7: i32;
    let mut max_gain_q7: i32;
    min_rate_dist_q7 = SILK_INT32_MAX;
    best_sum_log_gain_q7 = 0;
    k = 0;
    while k < 3 {
        let gain_safety: i32 = (0.4f64 * ((1) << 7) as f64 + 0.5f64) as i32;
        let cl_ptr_q5 = SILK_LTP_GAIN_BITS_Q5_PTRS[k as usize];
        let cbk_ptr_q7 = SILK_LTP_VQ_PTRS_Q7[k as usize].as_flattened();
        let cbk_gain_ptr_q7 = SILK_LTP_VQ_GAIN_PTRS_Q7[k as usize];
        cbk_size = SILK_LTP_VQ_SIZES[k as usize] as i32;
        let mut xx_off: usize = 0;
        let mut xx_off_small: usize = 0;
        res_nrg_q15 = 0;
        rate_dist_q7 = 0;
        sum_log_gain_tmp_q7 = *sum_log_gain_q7;
        j = 0;
        while j < nb_subfr {
            max_gain_q7 = silk_log2lin(
                (MAX_SUM_LOG_GAIN_DB as f64 / 6.0f64 * ((1) << 7) as f64 + 0.5f64) as i32
                    - sum_log_gain_tmp_q7
                    + ((7 * ((1) << 7)) as f64 + 0.5f64) as i32,
            ) - gain_safety;
            let (res_nrg_q15_subfr, rate_dist_q7_subfr, gain_q7) = {
                #[cfg(feature = "simd")]
                {
                    let mut res_nrg_q15_subfr: i32 = 0;
                    let mut rate_dist_q7_subfr: i32 = 0;
                    let mut gain_q7: i32 = 0;
                    silk_vq_wmat_ec(
                        &mut temp_idx[j as usize],
                        &mut res_nrg_q15_subfr,
                        &mut rate_dist_q7_subfr,
                        &mut gain_q7,
                        &xx_q17[xx_off..xx_off + LTP_ORDER * LTP_ORDER],
                        &x_x_q17[xx_off_small..xx_off_small + LTP_ORDER],
                        &cbk_ptr_q7[..cbk_size as usize * LTP_ORDER],
                        &cbk_gain_ptr_q7[..cbk_size as usize],
                        &cl_ptr_q5[..cbk_size as usize],
                        subfr_len,
                        max_gain_q7,
                        cbk_size,
                        _arch,
                    );
                    (res_nrg_q15_subfr, rate_dist_q7_subfr, gain_q7)
                }
                #[cfg(not(feature = "simd"))]
                {
                    let vq = silk_vq_wmat_ec_c(&SilkVqWmatEcParams {
                        xx_q17: &xx_q17[xx_off..xx_off + LTP_ORDER * LTP_ORDER],
                        x_x_q17: &x_x_q17[xx_off_small..xx_off_small + LTP_ORDER],
                        cb_q7: &cbk_ptr_q7[..cbk_size as usize * LTP_ORDER],
                        cb_gain_q7: &cbk_gain_ptr_q7[..cbk_size as usize],
                        cl_q5: &cl_ptr_q5[..cbk_size as usize],
                        subfr_len,
                        max_gain_q7,
                        l: cbk_size,
                    });
                    temp_idx[j as usize] = vq.ind;
                    (vq.res_nrg_q15, vq.rate_dist_q8, vq.gain_q7)
                }
            };
            res_nrg_q15 = if (res_nrg_q15 as u32).wrapping_add(res_nrg_q15_subfr as u32)
                & 0x80000000_u32
                != 0
            {
                SILK_INT32_MAX
            } else {
                res_nrg_q15 + res_nrg_q15_subfr
            };
            rate_dist_q7 = if (rate_dist_q7 as u32).wrapping_add(rate_dist_q7_subfr as u32)
                & 0x80000000_u32
                != 0
            {
                SILK_INT32_MAX
            } else {
                rate_dist_q7 + rate_dist_q7_subfr
            };
            sum_log_gain_tmp_q7 = if 0 > sum_log_gain_tmp_q7 + silk_lin2log(gain_safety + gain_q7)
                - ((7 * ((1) << 7)) as f64 + 0.5f64) as i32
            {
                0
            } else {
                sum_log_gain_tmp_q7 + silk_lin2log(gain_safety + gain_q7)
                    - ((7 * ((1) << 7)) as f64 + 0.5f64) as i32
            };
            xx_off += LTP_ORDER * LTP_ORDER;
            xx_off_small += LTP_ORDER;
            j += 1;
        }
        if rate_dist_q7 <= min_rate_dist_q7 {
            min_rate_dist_q7 = rate_dist_q7;
            *periodicity_index = k as i8;
            cbk_index[..nb_subfr as usize].copy_from_slice(&temp_idx[..nb_subfr as usize]);
            best_sum_log_gain_q7 = sum_log_gain_tmp_q7;
        }
        k += 1;
    }
    let best_cbk = SILK_LTP_VQ_PTRS_Q7[*periodicity_index as usize];
    j = 0;
    while j < nb_subfr {
        k = 0;
        while k < LTP_ORDER as i32 {
            b_q14[(j * LTP_ORDER as i32 + k) as usize] =
                ((best_cbk[cbk_index[j as usize] as usize][k as usize] as u32) << 7) as i32 as i16;
            k += 1;
        }
        j += 1;
    }
    if nb_subfr == 2 {
        res_nrg_q15 >>= 1;
    } else {
        res_nrg_q15 >>= 2;
    }
    *sum_log_gain_q7 = best_sum_log_gain_q7;
    *pred_gain_d_b_q7 = -3_i16 as i32 * (silk_lin2log(res_nrg_q15) - ((15) << 7)) as i16 as i32;
}
