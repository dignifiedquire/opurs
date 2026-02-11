use crate::silk::lin2log::silk_lin2log;

pub mod typedef_h {
    pub const silk_int32_MAX: i32 = i32::MAX;
}
pub use self::typedef_h::silk_int32_MAX;
use crate::silk::define::LTP_ORDER;

/// Upstream C: silk/VQ_WMat_EC.c:silk_VQ_WMat_EC_c
pub fn silk_VQ_WMat_EC_c(
    ind: &mut i8,
    res_nrg_Q15: &mut i32,
    rate_dist_Q8: &mut i32,
    gain_Q7: &mut i32,
    XX_Q17: &[i32],
    xX_Q17: &[i32],
    cb_Q7: &[i8],
    cb_gain_Q7: &[u8],
    cl_Q5: &[u8],
    subfr_len: i32,
    max_gain_Q7: i32,
    L: i32,
) {
    let mut k: i32 = 0;
    let mut gain_tmp_Q7: i32 = 0;
    let mut cb_row_off: usize = 0;
    let mut neg_xX_Q24: [i32; 5] = [0; 5];
    let mut sum1_Q15: i32 = 0;
    let mut sum2_Q24: i32 = 0;
    let mut bits_res_Q8: i32 = 0;
    let mut bits_tot_Q8: i32 = 0;
    neg_xX_Q24[0_usize] = -(((xX_Q17[0] as u32) << 7) as i32);
    neg_xX_Q24[1_usize] = -(((xX_Q17[1] as u32) << 7) as i32);
    neg_xX_Q24[2_usize] = -(((xX_Q17[2] as u32) << 7) as i32);
    neg_xX_Q24[3_usize] = -(((xX_Q17[3] as u32) << 7) as i32);
    neg_xX_Q24[4_usize] = -(((xX_Q17[4] as u32) << 7) as i32);
    *rate_dist_Q8 = silk_int32_MAX;
    *res_nrg_Q15 = silk_int32_MAX;
    cb_row_off = 0;
    *ind = 0;
    k = 0;
    while k < L {
        let mut penalty: i32 = 0;
        gain_tmp_Q7 = cb_gain_Q7[k as usize] as i32;
        sum1_Q15 = (1.001f64 * ((1) << 15) as f64 + 0.5f64) as i32;
        penalty = (((if gain_tmp_Q7 - max_gain_Q7 > 0 {
            gain_tmp_Q7 - max_gain_Q7
        } else {
            0
        }) as u32)
            << 11) as i32;
        sum2_Q24 = neg_xX_Q24[0_usize] + XX_Q17[1] * cb_Q7[cb_row_off + 1] as i32;
        sum2_Q24 += XX_Q17[2] * cb_Q7[cb_row_off + 2] as i32;
        sum2_Q24 += XX_Q17[3] * cb_Q7[cb_row_off + 3] as i32;
        sum2_Q24 += XX_Q17[4] * cb_Q7[cb_row_off + 4] as i32;
        sum2_Q24 = ((sum2_Q24 as u32) << 1) as i32;
        sum2_Q24 += XX_Q17[0] * cb_Q7[cb_row_off] as i32;
        sum1_Q15 =
            (sum1_Q15 as i64 + ((sum2_Q24 as i64 * cb_Q7[cb_row_off] as i16 as i64) >> 16)) as i32;
        sum2_Q24 = neg_xX_Q24[1_usize] + XX_Q17[7] * cb_Q7[cb_row_off + 2] as i32;
        sum2_Q24 += XX_Q17[8] * cb_Q7[cb_row_off + 3] as i32;
        sum2_Q24 += XX_Q17[9] * cb_Q7[cb_row_off + 4] as i32;
        sum2_Q24 = ((sum2_Q24 as u32) << 1) as i32;
        sum2_Q24 += XX_Q17[6] * cb_Q7[cb_row_off + 1] as i32;
        sum1_Q15 = (sum1_Q15 as i64
            + ((sum2_Q24 as i64 * cb_Q7[cb_row_off + 1] as i16 as i64) >> 16))
            as i32;
        sum2_Q24 = neg_xX_Q24[2_usize] + XX_Q17[13] * cb_Q7[cb_row_off + 3] as i32;
        sum2_Q24 += XX_Q17[14] * cb_Q7[cb_row_off + 4] as i32;
        sum2_Q24 = ((sum2_Q24 as u32) << 1) as i32;
        sum2_Q24 += XX_Q17[12] * cb_Q7[cb_row_off + 2] as i32;
        sum1_Q15 = (sum1_Q15 as i64
            + ((sum2_Q24 as i64 * cb_Q7[cb_row_off + 2] as i16 as i64) >> 16))
            as i32;
        sum2_Q24 = neg_xX_Q24[3_usize] + XX_Q17[19] * cb_Q7[cb_row_off + 4] as i32;
        sum2_Q24 = ((sum2_Q24 as u32) << 1) as i32;
        sum2_Q24 += XX_Q17[18] * cb_Q7[cb_row_off + 3] as i32;
        sum1_Q15 = (sum1_Q15 as i64
            + ((sum2_Q24 as i64 * cb_Q7[cb_row_off + 3] as i16 as i64) >> 16))
            as i32;
        sum2_Q24 = ((neg_xX_Q24[4_usize] as u32) << 1) as i32;
        sum2_Q24 += XX_Q17[24] * cb_Q7[cb_row_off + 4] as i32;
        sum1_Q15 = (sum1_Q15 as i64
            + ((sum2_Q24 as i64 * cb_Q7[cb_row_off + 4] as i16 as i64) >> 16))
            as i32;
        if sum1_Q15 >= 0 {
            bits_res_Q8 = subfr_len as i16 as i32
                * (silk_lin2log(sum1_Q15 + penalty) - ((15) << 7)) as i16 as i32;
            bits_tot_Q8 = bits_res_Q8 + ((cl_Q5[k as usize] as u32) << (3 - 1)) as i32;
            if bits_tot_Q8 <= *rate_dist_Q8 {
                *rate_dist_Q8 = bits_tot_Q8;
                *res_nrg_Q15 = sum1_Q15 + penalty;
                *ind = k as i8;
                *gain_Q7 = gain_tmp_Q7;
            }
        }
        cb_row_off += LTP_ORDER;
        k += 1;
    }
}
