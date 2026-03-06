//! One-stage analysis filter bank.
//!
//! Upstream c: `silk/ana_filt_bank_1.c`

use crate::silk::typedefs::{SILK_INT16_MAX, SILK_INT16_MIN};

const A_FB1_20: i16 = ((5394) << 1) as i16;
const A_FB1_21: i16 = -24290_i16;
/// Upstream c: silk/ana_filt_bank_1.c:silk_ana_filt_bank_1
pub fn silk_ana_filt_bank_1(
    in_0: &[i16],
    s: &mut [i32],
    out_l: &mut [i16],
    out_h: &mut [i16],
    n: i32,
) {
    let mut k: i32;
    let n2: i32 = n >> 1;
    let mut in32: i32;
    let mut x: i32;
    let mut y: i32;
    let mut out_1: i32;
    let mut out_2: i32;
    k = 0;
    while k < n2 {
        in32 = ((in_0[(2 * k) as usize] as i32 as u32) << 10) as i32;
        y = in32 - s[0];
        x = (y as i64 + ((y as i64 * A_FB1_21 as i64) >> 16)) as i32;
        out_1 = s[0] + x;
        s[0] = in32 + x;
        in32 = ((in_0[(2 * k + 1) as usize] as i32 as u32) << 10) as i32;
        y = in32 - s[1];
        x = ((y as i64 * A_FB1_20 as i64) >> 16) as i32;
        out_2 = s[1] + x;
        s[1] = in32 + x;
        out_l[k as usize] = (if (if 11 == 1 {
            ((out_2 + out_1) >> 1) + ((out_2 + out_1) & 1)
        } else {
            (((out_2 + out_1) >> (11 - 1)) + 1) >> 1
        }) > SILK_INT16_MAX
        {
            SILK_INT16_MAX
        } else if (if 11 == 1 {
            ((out_2 + out_1) >> 1) + ((out_2 + out_1) & 1)
        } else {
            (((out_2 + out_1) >> (11 - 1)) + 1) >> 1
        }) < SILK_INT16_MIN
        {
            SILK_INT16_MIN
        } else if 11 == 1 {
            ((out_2 + out_1) >> 1) + ((out_2 + out_1) & 1)
        } else {
            (((out_2 + out_1) >> (11 - 1)) + 1) >> 1
        }) as i16;
        out_h[k as usize] = (if (if 11 == 1 {
            ((out_2 - out_1) >> 1) + ((out_2 - out_1) & 1)
        } else {
            (((out_2 - out_1) >> (11 - 1)) + 1) >> 1
        }) > SILK_INT16_MAX
        {
            SILK_INT16_MAX
        } else if (if 11 == 1 {
            ((out_2 - out_1) >> 1) + ((out_2 - out_1) & 1)
        } else {
            (((out_2 - out_1) >> (11 - 1)) + 1) >> 1
        }) < SILK_INT16_MIN
        {
            SILK_INT16_MIN
        } else if 11 == 1 {
            ((out_2 - out_1) >> 1) + ((out_2 - out_1) & 1)
        } else {
            (((out_2 - out_1) >> (11 - 1)) + 1) >> 1
        }) as i16;
        k += 1;
    }
}
