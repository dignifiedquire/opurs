//! NLSF delayed-decision quantization.
//!
//! Upstream C: `silk/NLSF_del_dec_quant.c`

pub mod typedef_h {
    pub const silk_int32_MAX: i32 = i32::MAX;
}
use crate::silk::define::{
    NLSF_QUANT_DEL_DEC_STATES, NLSF_QUANT_MAX_AMPLITUDE, NLSF_QUANT_MAX_AMPLITUDE_EXT,
};

pub use self::typedef_h::silk_int32_MAX;

/// Upstream C: silk/NLSF_del_dec_quant.c:silk_NLSF_del_dec_quant
pub fn silk_NLSF_del_dec_quant(
    indices: &mut [i8],
    x_Q10: &[i16],
    w_Q5: &[i16],
    pred_coef_Q8: &[u8],
    ec_ix: &[i16],
    ec_rates_Q5: &[u8],
    quant_step_size_Q16: i32,
    inv_quant_step_size_Q6: i16,
    mu_Q20: i32,
    order: i16,
) -> i32 {
    let mut i: i32 = 0;
    let mut j: i32 = 0;
    let mut nStates: i32 = 0;
    let mut ind_tmp: i32 = 0;
    let mut ind_min_max: i32 = 0;
    let mut ind_max_min: i32 = 0;
    let mut in_Q10: i32 = 0;
    let mut res_Q10: i32 = 0;
    let mut pred_Q10: i32 = 0;
    let mut diff_Q10: i32 = 0;
    let mut rate0_Q5: i32 = 0;
    let mut rate1_Q5: i32 = 0;
    let mut out0_Q10: i16 = 0;
    let mut out1_Q10: i16 = 0;
    let mut RD_tmp_Q25: i32 = 0;
    let mut min_Q25: i32 = 0;
    let mut min_max_Q25: i32 = 0;
    let mut max_min_Q25: i32 = 0;
    let mut ind_sort: [i32; 4] = [0; 4];
    let mut ind: [[i8; 16]; 4] = [[0; 16]; 4];
    let mut prev_out_Q10: [i16; 8] = [0; 8];
    let mut RD_Q25: [i32; 8] = [0; 8];
    let mut RD_min_Q25: [i32; 4] = [0; 4];
    let mut RD_max_Q25: [i32; 4] = [0; 4];
    let mut out0_Q10_table: [i32; 20] = [0; 20];
    let mut out1_Q10_table: [i32; 20] = [0; 20];
    i = -NLSF_QUANT_MAX_AMPLITUDE_EXT;
    while i < NLSF_QUANT_MAX_AMPLITUDE_EXT {
        out0_Q10 = ((i as u32) << 10) as i32 as i16;
        out1_Q10 = (out0_Q10 as i32 + 1024) as i16;
        if i > 0 {
            out0_Q10 = (out0_Q10 as i32 - (0.1f64 * ((1) << 10) as f64 + 0.5f64) as i32) as i16;
            out1_Q10 = (out1_Q10 as i32 - (0.1f64 * ((1) << 10) as f64 + 0.5f64) as i32) as i16;
        } else if i == 0 {
            out1_Q10 = (out1_Q10 as i32 - (0.1f64 * ((1) << 10) as f64 + 0.5f64) as i32) as i16;
        } else if i == -1 {
            out0_Q10 = (out0_Q10 as i32 + (0.1f64 * ((1) << 10) as f64 + 0.5f64) as i32) as i16;
        } else {
            out0_Q10 = (out0_Q10 as i32 + (0.1f64 * ((1) << 10) as f64 + 0.5f64) as i32) as i16;
            out1_Q10 = (out1_Q10 as i32 + (0.1f64 * ((1) << 10) as f64 + 0.5f64) as i32) as i16;
        }
        unsafe {
            *out0_Q10_table.get_unchecked_mut((i + NLSF_QUANT_MAX_AMPLITUDE_EXT) as usize) =
                (out0_Q10 as i32 * quant_step_size_Q16 as i16 as i32) >> 16;
            *out1_Q10_table.get_unchecked_mut((i + NLSF_QUANT_MAX_AMPLITUDE_EXT) as usize) =
                (out1_Q10 as i32 * quant_step_size_Q16 as i16 as i32) >> 16;
        }
        i += 1;
    }
    nStates = 1;
    unsafe {
        *RD_Q25.get_unchecked_mut(0_usize) = 0;
    }
    unsafe {
        *prev_out_Q10.get_unchecked_mut(0_usize) = 0;
    }
    i = order as i32 - 1;
    while i >= 0 {
        let rates_Q5 = &ec_rates_Q5[unsafe { *ec_ix.get_unchecked(i as usize) } as usize..];
        in_Q10 = unsafe { *x_Q10.get_unchecked(i as usize) } as i32;
        j = 0;
        while j < nStates {
            pred_Q10 = (unsafe { *pred_coef_Q8.get_unchecked(i as usize) } as i16 as i32
                * unsafe { *prev_out_Q10.get_unchecked(j as usize) } as i32)
                >> 8;
            res_Q10 = in_Q10 - pred_Q10;
            ind_tmp = (inv_quant_step_size_Q6 as i32 * res_Q10 as i16 as i32) >> 16;
            ind_tmp = if -(10) > 10 - 1 {
                if ind_tmp > -(10) {
                    -(10)
                } else if ind_tmp < 10 - 1 {
                    10 - 1
                } else {
                    ind_tmp
                }
            } else if ind_tmp > 10 - 1 {
                10 - 1
            } else if ind_tmp < -(10) {
                -(10)
            } else {
                ind_tmp
            };
            unsafe {
                *ind.get_unchecked_mut(j as usize)
                    .get_unchecked_mut(i as usize) = ind_tmp as i8;
            }
            out0_Q10 = unsafe {
                *out0_Q10_table.get_unchecked((ind_tmp + NLSF_QUANT_MAX_AMPLITUDE_EXT) as usize)
            } as i16;
            out1_Q10 = unsafe {
                *out1_Q10_table.get_unchecked((ind_tmp + NLSF_QUANT_MAX_AMPLITUDE_EXT) as usize)
            } as i16;
            out0_Q10 = (out0_Q10 as i32 + pred_Q10) as i16;
            out1_Q10 = (out1_Q10 as i32 + pred_Q10) as i16;
            unsafe {
                *prev_out_Q10.get_unchecked_mut(j as usize) = out0_Q10;
            }
            unsafe {
                *prev_out_Q10.get_unchecked_mut((j + nStates) as usize) = out1_Q10;
            }
            if ind_tmp + 1 >= NLSF_QUANT_MAX_AMPLITUDE {
                if ind_tmp + 1 == NLSF_QUANT_MAX_AMPLITUDE {
                    rate0_Q5 = unsafe {
                        *rates_Q5.get_unchecked((ind_tmp + NLSF_QUANT_MAX_AMPLITUDE) as usize)
                    } as i32;
                    rate1_Q5 = 280;
                } else {
                    rate0_Q5 = 280 - 43 * 4 + 43 * ind_tmp as i16 as i32;
                    rate1_Q5 = rate0_Q5 + 43;
                }
            } else if ind_tmp <= -NLSF_QUANT_MAX_AMPLITUDE {
                if ind_tmp == -NLSF_QUANT_MAX_AMPLITUDE {
                    rate0_Q5 = 280;
                    rate1_Q5 = unsafe {
                        *rates_Q5.get_unchecked((ind_tmp + 1 + NLSF_QUANT_MAX_AMPLITUDE) as usize)
                    } as i32;
                } else {
                    rate0_Q5 = 280 - 43 * 4 + -43_i16 as i32 * ind_tmp as i16 as i32;
                    rate1_Q5 = rate0_Q5 - 43;
                }
            } else {
                rate0_Q5 = unsafe {
                    *rates_Q5.get_unchecked((ind_tmp + NLSF_QUANT_MAX_AMPLITUDE) as usize)
                } as i32;
                rate1_Q5 = unsafe {
                    *rates_Q5.get_unchecked((ind_tmp + 1 + NLSF_QUANT_MAX_AMPLITUDE) as usize)
                } as i32;
            }
            RD_tmp_Q25 = unsafe { *RD_Q25.get_unchecked(j as usize) };
            diff_Q10 = in_Q10 - out0_Q10 as i32;
            unsafe {
                *RD_Q25.get_unchecked_mut(j as usize) = RD_tmp_Q25
                    + diff_Q10 as i16 as i32
                        * diff_Q10 as i16 as i32
                        * *w_Q5.get_unchecked(i as usize) as i32
                    + mu_Q20 as i16 as i32 * rate0_Q5 as i16 as i32;
            }
            diff_Q10 = in_Q10 - out1_Q10 as i32;
            unsafe {
                *RD_Q25.get_unchecked_mut((j + nStates) as usize) = RD_tmp_Q25
                    + diff_Q10 as i16 as i32
                        * diff_Q10 as i16 as i32
                        * *w_Q5.get_unchecked(i as usize) as i32
                    + mu_Q20 as i16 as i32 * rate1_Q5 as i16 as i32;
            }
            j += 1;
        }
        if nStates <= NLSF_QUANT_DEL_DEC_STATES / 2 {
            j = 0;
            while j < nStates {
                unsafe {
                    *ind.get_unchecked_mut((j + nStates) as usize)
                        .get_unchecked_mut(i as usize) =
                        (*ind.get_unchecked(j as usize).get_unchecked(i as usize) as i32 + 1) as i8;
                }
                j += 1;
            }
            nStates = ((nStates as u32) << 1) as i32;
            j = nStates;
            while j < NLSF_QUANT_DEL_DEC_STATES {
                unsafe {
                    *ind.get_unchecked_mut(j as usize)
                        .get_unchecked_mut(i as usize) = *ind
                        .get_unchecked((j - nStates) as usize)
                        .get_unchecked(i as usize);
                }
                j += 1;
            }
        } else {
            j = 0;
            while j < NLSF_QUANT_DEL_DEC_STATES {
                unsafe {
                    if *RD_Q25.get_unchecked(j as usize)
                        > *RD_Q25.get_unchecked((j + NLSF_QUANT_DEL_DEC_STATES) as usize)
                    {
                        *RD_max_Q25.get_unchecked_mut(j as usize) =
                            *RD_Q25.get_unchecked(j as usize);
                        *RD_min_Q25.get_unchecked_mut(j as usize) =
                            *RD_Q25.get_unchecked((j + NLSF_QUANT_DEL_DEC_STATES) as usize);
                        *RD_Q25.get_unchecked_mut(j as usize) =
                            *RD_min_Q25.get_unchecked(j as usize);
                        *RD_Q25.get_unchecked_mut((j + NLSF_QUANT_DEL_DEC_STATES) as usize) =
                            *RD_max_Q25.get_unchecked(j as usize);
                        out0_Q10 = *prev_out_Q10.get_unchecked(j as usize);
                        *prev_out_Q10.get_unchecked_mut(j as usize) =
                            *prev_out_Q10.get_unchecked((j + NLSF_QUANT_DEL_DEC_STATES) as usize);
                        *prev_out_Q10.get_unchecked_mut((j + NLSF_QUANT_DEL_DEC_STATES) as usize) =
                            out0_Q10;
                        *ind_sort.get_unchecked_mut(j as usize) = j + NLSF_QUANT_DEL_DEC_STATES;
                    } else {
                        *RD_min_Q25.get_unchecked_mut(j as usize) =
                            *RD_Q25.get_unchecked(j as usize);
                        *RD_max_Q25.get_unchecked_mut(j as usize) =
                            *RD_Q25.get_unchecked((j + NLSF_QUANT_DEL_DEC_STATES) as usize);
                        *ind_sort.get_unchecked_mut(j as usize) = j;
                    }
                }
                j += 1;
            }
            loop {
                min_max_Q25 = silk_int32_MAX;
                max_min_Q25 = 0;
                ind_min_max = 0;
                ind_max_min = 0;
                j = 0;
                while j < NLSF_QUANT_DEL_DEC_STATES {
                    unsafe {
                        if min_max_Q25 > *RD_max_Q25.get_unchecked(j as usize) {
                            min_max_Q25 = *RD_max_Q25.get_unchecked(j as usize);
                            ind_min_max = j;
                        }
                        if max_min_Q25 < *RD_min_Q25.get_unchecked(j as usize) {
                            max_min_Q25 = *RD_min_Q25.get_unchecked(j as usize);
                            ind_max_min = j;
                        }
                    }
                    j += 1;
                }
                if min_max_Q25 >= max_min_Q25 {
                    break;
                }
                unsafe {
                    *ind_sort.get_unchecked_mut(ind_max_min as usize) =
                        *ind_sort.get_unchecked(ind_min_max as usize) ^ NLSF_QUANT_DEL_DEC_STATES;
                    *RD_Q25.get_unchecked_mut(ind_max_min as usize) =
                        *RD_Q25.get_unchecked((ind_min_max + NLSF_QUANT_DEL_DEC_STATES) as usize);
                    *prev_out_Q10.get_unchecked_mut(ind_max_min as usize) = *prev_out_Q10
                        .get_unchecked((ind_min_max + NLSF_QUANT_DEL_DEC_STATES) as usize);
                    *RD_min_Q25.get_unchecked_mut(ind_max_min as usize) = 0;
                    *RD_max_Q25.get_unchecked_mut(ind_min_max as usize) = silk_int32_MAX;
                    let tmp = *ind.get_unchecked(ind_min_max as usize);
                    *ind.get_unchecked_mut(ind_max_min as usize) = tmp;
                }
            }
            j = 0;
            while j < NLSF_QUANT_DEL_DEC_STATES {
                unsafe {
                    *ind.get_unchecked_mut(j as usize)
                        .get_unchecked_mut(i as usize) =
                        (*ind.get_unchecked(j as usize).get_unchecked(i as usize) as i32
                            + (*ind_sort.get_unchecked(j as usize) >> 2))
                            as i8;
                }
                j += 1;
            }
        }
        i -= 1;
    }
    ind_tmp = 0;
    min_Q25 = silk_int32_MAX;
    j = 0;
    while j < 2 * NLSF_QUANT_DEL_DEC_STATES {
        unsafe {
            if min_Q25 > *RD_Q25.get_unchecked(j as usize) {
                min_Q25 = *RD_Q25.get_unchecked(j as usize);
                ind_tmp = j;
            }
        }
        j += 1;
    }
    j = 0;
    while j < order as i32 {
        unsafe {
            *indices.get_unchecked_mut(j as usize) = *ind
                .get_unchecked((ind_tmp & (NLSF_QUANT_DEL_DEC_STATES - 1)) as usize)
                .get_unchecked(j as usize);
        }
        j += 1;
    }
    unsafe {
        *indices.get_unchecked_mut(0) = (*indices.get_unchecked(0) as i32 + (ind_tmp >> 2)) as i8;
    }
    min_Q25
}
