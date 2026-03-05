//! Weighted matrix vector quantization with entropy coding.
//!
//! Upstream C: `silk/VQ_WMat_EC.c`

use crate::silk::lin2log::silk_lin2log;

use crate::silk::define::LTP_ORDER;
use crate::silk::typedefs::SILK_INT32_MAX;

pub struct SilkVqWmatEcParams<'a> {
    pub xx_q17: &'a [i32],
    pub x_x_q17: &'a [i32],
    pub cb_q7: &'a [i8],
    pub cb_gain_q7: &'a [u8],
    pub cl_q5: &'a [u8],
    pub subfr_len: i32,
    pub max_gain_q7: i32,
    pub l: i32,
}

pub struct SilkVqWmatEcResult {
    pub ind: i8,
    pub res_nrg_q15: i32,
    pub rate_dist_q8: i32,
    pub gain_q7: i32,
}

/// Upstream C: silk/VQ_WMat_EC.c:silk_VQ_WMat_EC_c
pub fn silk_vq_wmat_ec_c(params: &SilkVqWmatEcParams<'_>) -> SilkVqWmatEcResult {
    let mut k: i32;
    let mut gain_tmp_Q7: i32;
    let mut cb_row_off: usize;
    let mut neg_xX_Q24: [i32; 5] = [0; 5];
    let mut sum1_Q15: i32;
    let mut sum2_Q24: i32;
    let mut bits_res_Q8: i32;
    let mut bits_tot_Q8: i32;
    neg_xX_Q24[0_usize] = -(((params.x_x_q17[0] as u32) << 7) as i32);
    neg_xX_Q24[1_usize] = -(((params.x_x_q17[1] as u32) << 7) as i32);
    neg_xX_Q24[2_usize] = -(((params.x_x_q17[2] as u32) << 7) as i32);
    neg_xX_Q24[3_usize] = -(((params.x_x_q17[3] as u32) << 7) as i32);
    neg_xX_Q24[4_usize] = -(((params.x_x_q17[4] as u32) << 7) as i32);
    let mut out = SilkVqWmatEcResult {
        ind: 0,
        res_nrg_q15: SILK_INT32_MAX,
        rate_dist_q8: SILK_INT32_MAX,
        gain_q7: 0,
    };
    cb_row_off = 0;
    k = 0;
    while k < params.l {
        gain_tmp_Q7 = params.cb_gain_q7[k as usize] as i32;
        sum1_Q15 = (1.001f64 * ((1) << 15) as f64 + 0.5f64) as i32;
        let penalty: i32 = (((if gain_tmp_Q7 - params.max_gain_q7 > 0 {
            gain_tmp_Q7 - params.max_gain_q7
        } else {
            0
        }) as u32)
            << 11) as i32;
        sum2_Q24 = neg_xX_Q24[0_usize] + params.xx_q17[1] * params.cb_q7[cb_row_off + 1] as i32;
        sum2_Q24 += params.xx_q17[2] * params.cb_q7[cb_row_off + 2] as i32;
        sum2_Q24 += params.xx_q17[3] * params.cb_q7[cb_row_off + 3] as i32;
        sum2_Q24 += params.xx_q17[4] * params.cb_q7[cb_row_off + 4] as i32;
        sum2_Q24 = ((sum2_Q24 as u32) << 1) as i32;
        sum2_Q24 += params.xx_q17[0] * params.cb_q7[cb_row_off] as i32;
        sum1_Q15 = (sum1_Q15 as i64
            + ((sum2_Q24 as i64 * params.cb_q7[cb_row_off] as i16 as i64) >> 16))
            as i32;
        sum2_Q24 = neg_xX_Q24[1_usize] + params.xx_q17[7] * params.cb_q7[cb_row_off + 2] as i32;
        sum2_Q24 += params.xx_q17[8] * params.cb_q7[cb_row_off + 3] as i32;
        sum2_Q24 += params.xx_q17[9] * params.cb_q7[cb_row_off + 4] as i32;
        sum2_Q24 = ((sum2_Q24 as u32) << 1) as i32;
        sum2_Q24 += params.xx_q17[6] * params.cb_q7[cb_row_off + 1] as i32;
        sum1_Q15 = (sum1_Q15 as i64
            + ((sum2_Q24 as i64 * params.cb_q7[cb_row_off + 1] as i16 as i64) >> 16))
            as i32;
        sum2_Q24 = neg_xX_Q24[2_usize] + params.xx_q17[13] * params.cb_q7[cb_row_off + 3] as i32;
        sum2_Q24 += params.xx_q17[14] * params.cb_q7[cb_row_off + 4] as i32;
        sum2_Q24 = ((sum2_Q24 as u32) << 1) as i32;
        sum2_Q24 += params.xx_q17[12] * params.cb_q7[cb_row_off + 2] as i32;
        sum1_Q15 = (sum1_Q15 as i64
            + ((sum2_Q24 as i64 * params.cb_q7[cb_row_off + 2] as i16 as i64) >> 16))
            as i32;
        sum2_Q24 = neg_xX_Q24[3_usize] + params.xx_q17[19] * params.cb_q7[cb_row_off + 4] as i32;
        sum2_Q24 = ((sum2_Q24 as u32) << 1) as i32;
        sum2_Q24 += params.xx_q17[18] * params.cb_q7[cb_row_off + 3] as i32;
        sum1_Q15 = (sum1_Q15 as i64
            + ((sum2_Q24 as i64 * params.cb_q7[cb_row_off + 3] as i16 as i64) >> 16))
            as i32;
        sum2_Q24 = ((neg_xX_Q24[4_usize] as u32) << 1) as i32;
        sum2_Q24 += params.xx_q17[24] * params.cb_q7[cb_row_off + 4] as i32;
        sum1_Q15 = (sum1_Q15 as i64
            + ((sum2_Q24 as i64 * params.cb_q7[cb_row_off + 4] as i16 as i64) >> 16))
            as i32;
        if sum1_Q15 >= 0 {
            bits_res_Q8 = params.subfr_len as i16 as i32
                * (silk_lin2log(sum1_Q15 + penalty) - ((15) << 7)) as i16 as i32;
            bits_tot_Q8 = bits_res_Q8 + ((params.cl_q5[k as usize] as u32) << (3 - 1)) as i32;
            if bits_tot_Q8 <= out.rate_dist_q8 {
                out.rate_dist_q8 = bits_tot_Q8;
                out.res_nrg_q15 = sum1_Q15 + penalty;
                out.ind = k as i8;
                out.gain_q7 = gain_tmp_Q7;
            }
        }
        cb_row_off += LTP_ORDER;
        k += 1;
    }
    out
}
