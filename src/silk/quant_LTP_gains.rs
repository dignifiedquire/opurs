//! LTP gain quantization.
//!
//! Upstream C: `silk/quant_LTP_gains.c`

pub mod typedef_h {
    pub const silk_int32_MAX: i32 = i32::MAX;
}
pub use self::typedef_h::silk_int32_MAX;
use crate::silk::define::LTP_ORDER;
use crate::silk::lin2log::silk_lin2log;
use crate::silk::log2lin::silk_log2lin;
use crate::silk::tables_LTP::{
    silk_LTP_gain_BITS_Q5_ptrs, silk_LTP_vq_gain_ptrs_Q7, silk_LTP_vq_ptrs_Q7, silk_LTP_vq_sizes,
};
use crate::silk::tuning_parameters::MAX_SUM_LOG_GAIN_DB;
use crate::silk::VQ_WMat_EC::silk_VQ_WMat_EC_c;

/// Upstream C: silk/quant_LTP_gains.c:silk_quant_LTP_gains
pub fn silk_quant_LTP_gains(
    B_Q14: &mut [i16],
    cbk_index: &mut [i8],
    periodicity_index: &mut i8,
    sum_log_gain_Q7: &mut i32,
    pred_gain_dB_Q7: &mut i32,
    XX_Q17: &[i32],
    xX_Q17: &[i32],
    subfr_len: i32,
    nb_subfr: i32,
    _arch: i32,
) {
    let mut j: i32 = 0;
    let mut k: i32 = 0;
    let mut cbk_size: i32 = 0;
    let mut temp_idx: [i8; 4] = [0; 4];
    let mut res_nrg_Q15_subfr: i32 = 0;
    let mut res_nrg_Q15: i32 = 0;
    let mut rate_dist_Q7_subfr: i32 = 0;
    let mut rate_dist_Q7: i32 = 0;
    let mut min_rate_dist_Q7: i32 = 0;
    let mut sum_log_gain_tmp_Q7: i32 = 0;
    let mut best_sum_log_gain_Q7: i32 = 0;
    let mut max_gain_Q7: i32 = 0;
    let mut gain_Q7: i32 = 0;
    min_rate_dist_Q7 = silk_int32_MAX;
    best_sum_log_gain_Q7 = 0;
    k = 0;
    while k < 3 {
        let gain_safety: i32 = (0.4f64 * ((1) << 7) as f64 + 0.5f64) as i32;
        let cl_ptr_Q5 = silk_LTP_gain_BITS_Q5_ptrs[k as usize];
        let cbk_ptr_Q7 = silk_LTP_vq_ptrs_Q7[k as usize].as_flattened();
        let cbk_gain_ptr_Q7 = silk_LTP_vq_gain_ptrs_Q7[k as usize];
        cbk_size = silk_LTP_vq_sizes[k as usize] as i32;
        let mut xx_off: usize = 0;
        let mut xx_off_small: usize = 0;
        res_nrg_Q15 = 0;
        rate_dist_Q7 = 0;
        sum_log_gain_tmp_Q7 = *sum_log_gain_Q7;
        j = 0;
        while j < nb_subfr {
            max_gain_Q7 = silk_log2lin(
                (MAX_SUM_LOG_GAIN_DB as f64 / 6.0f64 * ((1) << 7) as f64 + 0.5f64) as i32
                    - sum_log_gain_tmp_Q7
                    + ((7 * ((1) << 7)) as f64 + 0.5f64) as i32,
            ) - gain_safety;
            silk_VQ_WMat_EC_c(
                &mut temp_idx[j as usize],
                &mut res_nrg_Q15_subfr,
                &mut rate_dist_Q7_subfr,
                &mut gain_Q7,
                &XX_Q17[xx_off..xx_off + LTP_ORDER * LTP_ORDER],
                &xX_Q17[xx_off_small..xx_off_small + LTP_ORDER],
                &cbk_ptr_Q7[..cbk_size as usize * LTP_ORDER],
                &cbk_gain_ptr_Q7[..cbk_size as usize],
                &cl_ptr_Q5[..cbk_size as usize],
                subfr_len,
                max_gain_Q7,
                cbk_size,
            );
            res_nrg_Q15 = if (res_nrg_Q15 as u32).wrapping_add(res_nrg_Q15_subfr as u32)
                & 0x80000000_u32
                != 0
            {
                silk_int32_MAX
            } else {
                res_nrg_Q15 + res_nrg_Q15_subfr
            };
            rate_dist_Q7 = if (rate_dist_Q7 as u32).wrapping_add(rate_dist_Q7_subfr as u32)
                & 0x80000000_u32
                != 0
            {
                silk_int32_MAX
            } else {
                rate_dist_Q7 + rate_dist_Q7_subfr
            };
            sum_log_gain_tmp_Q7 = if 0 > sum_log_gain_tmp_Q7 + silk_lin2log(gain_safety + gain_Q7)
                - ((7 * ((1) << 7)) as f64 + 0.5f64) as i32
            {
                0
            } else {
                sum_log_gain_tmp_Q7 + silk_lin2log(gain_safety + gain_Q7)
                    - ((7 * ((1) << 7)) as f64 + 0.5f64) as i32
            };
            xx_off += LTP_ORDER * LTP_ORDER;
            xx_off_small += LTP_ORDER;
            j += 1;
        }
        if rate_dist_Q7 <= min_rate_dist_Q7 {
            min_rate_dist_Q7 = rate_dist_Q7;
            *periodicity_index = k as i8;
            cbk_index[..nb_subfr as usize].copy_from_slice(&temp_idx[..nb_subfr as usize]);
            best_sum_log_gain_Q7 = sum_log_gain_tmp_Q7;
        }
        k += 1;
    }
    let best_cbk = silk_LTP_vq_ptrs_Q7[*periodicity_index as usize];
    j = 0;
    while j < nb_subfr {
        k = 0;
        while k < LTP_ORDER as i32 {
            B_Q14[(j * LTP_ORDER as i32 + k) as usize] =
                ((best_cbk[cbk_index[j as usize] as usize][k as usize] as u32) << 7) as i32 as i16;
            k += 1;
        }
        j += 1;
    }
    if nb_subfr == 2 {
        res_nrg_Q15 >>= 1;
    } else {
        res_nrg_Q15 >>= 2;
    }
    *sum_log_gain_Q7 = best_sum_log_gain_Q7;
    *pred_gain_dB_Q7 = -3_i16 as i32 * (silk_lin2log(res_nrg_Q15) - ((15) << 7)) as i16 as i32;
}
