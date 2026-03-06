//! nlsf delayed-decision quantization.
//!
//! Upstream c: `silk/NLSF_del_dec_quant.c`

use crate::silk::define::{
    NLSF_QUANT_DEL_DEC_STATES, NLSF_QUANT_MAX_AMPLITUDE, NLSF_QUANT_MAX_AMPLITUDE_EXT,
};
use crate::silk::typedefs::SILK_INT32_MAX;

/// Upstream c: silk/NLSF_del_dec_quant.c:silk_NLSF_del_dec_quant
pub fn silk_nlsf_del_dec_quant(
    indices: &mut [i8],
    x_q10: &[i16],
    w_q5: &[i16],
    pred_coef_q8: &[u8],
    ec_ix: &[i16],
    ec_rates_q5: &[u8],
    quant_step_size_q16: i32,
    inv_quant_step_size_q6: i16,
    mu_q20: i32,
    order: i16,
) -> i32 {
    let mut _i: i32;
    let mut j: i32;
    let mut n_states: i32;
    let mut ind_tmp: i32;
    let mut ind_min_max: i32;
    let mut ind_max_min: i32;
    let mut in_q10: i32;
    let mut res_q10: i32;
    let mut pred_q10: i32;
    let mut diff_q10: i32;
    let mut rate0_q5: i32;
    let mut rate1_q5: i32;
    let mut out0_q10: i16;
    let mut out1_q10: i16;
    let mut rd_tmp_q25: i32;
    let mut min_q25: i32;
    let mut min_max_q25: i32;
    let mut max_min_q25: i32;
    let mut ind_sort: [i32; 4] = [0; 4];
    let mut ind: [[i8; 16]; 4] = [[0; 16]; 4];
    let mut prev_out_q10: [i16; 8] = [0; 8];
    let mut rd_q25: [i32; 8] = [0; 8];
    let mut rd_min_q25: [i32; 4] = [0; 4];
    let mut rd_max_q25: [i32; 4] = [0; 4];
    let mut out0_q10_table: [i32; 20] = [0; 20];
    let mut out1_q10_table: [i32; 20] = [0; 20];
    _i = -NLSF_QUANT_MAX_AMPLITUDE_EXT;
    while _i < NLSF_QUANT_MAX_AMPLITUDE_EXT {
        out0_q10 = ((_i as u32) << 10) as i32 as i16;
        out1_q10 = (out0_q10 as i32 + 1024) as i16;
        if _i > 0 {
            out0_q10 = (out0_q10 as i32 - (0.1f64 * ((1) << 10) as f64 + 0.5f64) as i32) as i16;
            out1_q10 = (out1_q10 as i32 - (0.1f64 * ((1) << 10) as f64 + 0.5f64) as i32) as i16;
        } else if _i == 0 {
            out1_q10 = (out1_q10 as i32 - (0.1f64 * ((1) << 10) as f64 + 0.5f64) as i32) as i16;
        } else if _i == -1 {
            out0_q10 = (out0_q10 as i32 + (0.1f64 * ((1) << 10) as f64 + 0.5f64) as i32) as i16;
        } else {
            out0_q10 = (out0_q10 as i32 + (0.1f64 * ((1) << 10) as f64 + 0.5f64) as i32) as i16;
            out1_q10 = (out1_q10 as i32 + (0.1f64 * ((1) << 10) as f64 + 0.5f64) as i32) as i16;
        }
        out0_q10_table[(_i + NLSF_QUANT_MAX_AMPLITUDE_EXT) as usize] =
            (out0_q10 as i32 * quant_step_size_q16 as i16 as i32) >> 16;
        out1_q10_table[(_i + NLSF_QUANT_MAX_AMPLITUDE_EXT) as usize] =
            (out1_q10 as i32 * quant_step_size_q16 as i16 as i32) >> 16;
        _i += 1;
    }
    n_states = 1;
    rd_q25[0_usize] = 0;
    prev_out_q10[0_usize] = 0;
    _i = order as i32 - 1;
    while _i >= 0 {
        let rates_q5 = &ec_rates_q5[ec_ix[_i as usize] as usize..];
        in_q10 = x_q10[_i as usize] as i32;
        j = 0;
        while j < n_states {
            pred_q10 =
                (pred_coef_q8[_i as usize] as i16 as i32 * prev_out_q10[j as usize] as i32) >> 8;
            res_q10 = in_q10 - pred_q10;
            ind_tmp = (inv_quant_step_size_q6 as i32 * res_q10 as i16 as i32) >> 16;
            ind_tmp = ind_tmp.clamp(-(10), 10 - 1);
            ind[j as usize][_i as usize] = ind_tmp as i8;
            out0_q10 = out0_q10_table[(ind_tmp + NLSF_QUANT_MAX_AMPLITUDE_EXT) as usize] as i16;
            out1_q10 = out1_q10_table[(ind_tmp + NLSF_QUANT_MAX_AMPLITUDE_EXT) as usize] as i16;
            out0_q10 = (out0_q10 as i32 + pred_q10) as i16;
            out1_q10 = (out1_q10 as i32 + pred_q10) as i16;
            prev_out_q10[j as usize] = out0_q10;
            prev_out_q10[(j + n_states) as usize] = out1_q10;
            if ind_tmp + 1 >= NLSF_QUANT_MAX_AMPLITUDE {
                if ind_tmp + 1 == NLSF_QUANT_MAX_AMPLITUDE {
                    rate0_q5 = rates_q5[(ind_tmp + NLSF_QUANT_MAX_AMPLITUDE) as usize] as i32;
                    rate1_q5 = 280;
                } else {
                    rate0_q5 = 280 - 43 * 4 + 43 * ind_tmp as i16 as i32;
                    rate1_q5 = rate0_q5 + 43;
                }
            } else if ind_tmp <= -NLSF_QUANT_MAX_AMPLITUDE {
                if ind_tmp == -NLSF_QUANT_MAX_AMPLITUDE {
                    rate0_q5 = 280;
                    rate1_q5 = rates_q5[(ind_tmp + 1 + NLSF_QUANT_MAX_AMPLITUDE) as usize] as i32;
                } else {
                    rate0_q5 = 280 - 43 * 4 + -43_i16 as i32 * ind_tmp as i16 as i32;
                    rate1_q5 = rate0_q5 - 43;
                }
            } else {
                rate0_q5 = rates_q5[(ind_tmp + NLSF_QUANT_MAX_AMPLITUDE) as usize] as i32;
                rate1_q5 = rates_q5[(ind_tmp + 1 + NLSF_QUANT_MAX_AMPLITUDE) as usize] as i32;
            }
            rd_tmp_q25 = rd_q25[j as usize];
            diff_q10 = in_q10 - out0_q10 as i32;
            rd_q25[j as usize] = rd_tmp_q25
                + diff_q10 as i16 as i32 * diff_q10 as i16 as i32 * w_q5[_i as usize] as i32
                + mu_q20 as i16 as i32 * rate0_q5 as i16 as i32;
            diff_q10 = in_q10 - out1_q10 as i32;
            rd_q25[(j + n_states) as usize] = rd_tmp_q25
                + diff_q10 as i16 as i32 * diff_q10 as i16 as i32 * w_q5[_i as usize] as i32
                + mu_q20 as i16 as i32 * rate1_q5 as i16 as i32;
            j += 1;
        }
        if n_states <= NLSF_QUANT_DEL_DEC_STATES / 2 {
            j = 0;
            while j < n_states {
                ind[(j + n_states) as usize][_i as usize] =
                    (ind[j as usize][_i as usize] as i32 + 1) as i8;
                j += 1;
            }
            n_states = ((n_states as u32) << 1) as i32;
            j = n_states;
            while j < NLSF_QUANT_DEL_DEC_STATES {
                ind[j as usize][_i as usize] = ind[(j - n_states) as usize][_i as usize];
                j += 1;
            }
        } else {
            j = 0;
            while j < NLSF_QUANT_DEL_DEC_STATES {
                if rd_q25[j as usize] > rd_q25[(j + NLSF_QUANT_DEL_DEC_STATES) as usize] {
                    rd_max_q25[j as usize] = rd_q25[j as usize];
                    rd_min_q25[j as usize] = rd_q25[(j + NLSF_QUANT_DEL_DEC_STATES) as usize];
                    rd_q25[j as usize] = rd_min_q25[j as usize];
                    rd_q25[(j + NLSF_QUANT_DEL_DEC_STATES) as usize] = rd_max_q25[j as usize];
                    out0_q10 = prev_out_q10[j as usize];
                    prev_out_q10[j as usize] =
                        prev_out_q10[(j + NLSF_QUANT_DEL_DEC_STATES) as usize];
                    prev_out_q10[(j + NLSF_QUANT_DEL_DEC_STATES) as usize] = out0_q10;
                    ind_sort[j as usize] = j + NLSF_QUANT_DEL_DEC_STATES;
                } else {
                    rd_min_q25[j as usize] = rd_q25[j as usize];
                    rd_max_q25[j as usize] = rd_q25[(j + NLSF_QUANT_DEL_DEC_STATES) as usize];
                    ind_sort[j as usize] = j;
                }
                j += 1;
            }
            loop {
                min_max_q25 = SILK_INT32_MAX;
                max_min_q25 = 0;
                ind_min_max = 0;
                ind_max_min = 0;
                j = 0;
                while j < NLSF_QUANT_DEL_DEC_STATES {
                    if min_max_q25 > rd_max_q25[j as usize] {
                        min_max_q25 = rd_max_q25[j as usize];
                        ind_min_max = j;
                    }
                    if max_min_q25 < rd_min_q25[j as usize] {
                        max_min_q25 = rd_min_q25[j as usize];
                        ind_max_min = j;
                    }
                    j += 1;
                }
                if min_max_q25 >= max_min_q25 {
                    break;
                }
                ind_sort[ind_max_min as usize] =
                    ind_sort[ind_min_max as usize] ^ NLSF_QUANT_DEL_DEC_STATES;
                rd_q25[ind_max_min as usize] =
                    rd_q25[(ind_min_max + NLSF_QUANT_DEL_DEC_STATES) as usize];
                prev_out_q10[ind_max_min as usize] =
                    prev_out_q10[(ind_min_max + NLSF_QUANT_DEL_DEC_STATES) as usize];
                rd_min_q25[ind_max_min as usize] = 0;
                rd_max_q25[ind_min_max as usize] = SILK_INT32_MAX;
                let tmp = ind[ind_min_max as usize];
                ind[ind_max_min as usize] = tmp;
            }
            j = 0;
            while j < NLSF_QUANT_DEL_DEC_STATES {
                ind[j as usize][_i as usize] =
                    (ind[j as usize][_i as usize] as i32 + (ind_sort[j as usize] >> 2)) as i8;
                j += 1;
            }
        }
        _i -= 1;
    }
    ind_tmp = 0;
    min_q25 = SILK_INT32_MAX;
    j = 0;
    while j < 2 * NLSF_QUANT_DEL_DEC_STATES {
        if min_q25 > rd_q25[j as usize] {
            min_q25 = rd_q25[j as usize];
            ind_tmp = j;
        }
        j += 1;
    }
    j = 0;
    while j < order as i32 {
        indices[j as usize] = ind[(ind_tmp & (NLSF_QUANT_DEL_DEC_STATES - 1)) as usize][j as usize];
        j += 1;
    }
    indices[0] = (indices[0] as i32 + (ind_tmp >> 2)) as i8;
    min_q25
}
