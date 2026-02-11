pub mod typedef_h {
    pub const silk_int16_MIN: i32 = i16::MIN as i32;
    pub const silk_int16_MAX: i32 = i16::MAX as i32;
}

pub use self::typedef_h::{silk_int16_MAX, silk_int16_MIN};

const A_FB1_20: i16 = ((5394) << 1) as i16;
const A_FB1_21: i16 = -24290_i16;
/// Upstream C: silk/ana_filt_bank_1.c:silk_ana_filt_bank_1
pub fn silk_ana_filt_bank_1(
    in_0: &[i16],
    S: &mut [i32],
    outL: &mut [i16],
    outH: &mut [i16],
    N: i32,
) {
    let mut k: i32 = 0;
    let N2: i32 = N >> 1;
    let mut in32: i32 = 0;
    let mut X: i32 = 0;
    let mut Y: i32 = 0;
    let mut out_1: i32 = 0;
    let mut out_2: i32 = 0;
    k = 0;
    while k < N2 {
        in32 = ((in_0[(2 * k) as usize] as i32 as u32) << 10) as i32;
        Y = in32 - S[0];
        X = (Y as i64 + ((Y as i64 * A_FB1_21 as i64) >> 16)) as i32;
        out_1 = S[0] + X;
        S[0] = in32 + X;
        in32 = ((in_0[(2 * k + 1) as usize] as i32 as u32) << 10) as i32;
        Y = in32 - S[1];
        X = ((Y as i64 * A_FB1_20 as i64) >> 16) as i32;
        out_2 = S[1] + X;
        S[1] = in32 + X;
        outL[k as usize] = (if (if 11 == 1 {
            ((out_2 + out_1) >> 1) + ((out_2 + out_1) & 1)
        } else {
            (((out_2 + out_1) >> (11 - 1)) + 1) >> 1
        }) > silk_int16_MAX
        {
            silk_int16_MAX
        } else if (if 11 == 1 {
            ((out_2 + out_1) >> 1) + ((out_2 + out_1) & 1)
        } else {
            (((out_2 + out_1) >> (11 - 1)) + 1) >> 1
        }) < silk_int16_MIN
        {
            silk_int16_MIN
        } else if 11 == 1 {
            ((out_2 + out_1) >> 1) + ((out_2 + out_1) & 1)
        } else {
            (((out_2 + out_1) >> (11 - 1)) + 1) >> 1
        }) as i16;
        outH[k as usize] = (if (if 11 == 1 {
            ((out_2 - out_1) >> 1) + ((out_2 - out_1) & 1)
        } else {
            (((out_2 - out_1) >> (11 - 1)) + 1) >> 1
        }) > silk_int16_MAX
        {
            silk_int16_MAX
        } else if (if 11 == 1 {
            ((out_2 - out_1) >> 1) + ((out_2 - out_1) & 1)
        } else {
            (((out_2 - out_1) >> (11 - 1)) + 1) >> 1
        }) < silk_int16_MIN
        {
            silk_int16_MIN
        } else if 11 == 1 {
            ((out_2 - out_1) >> 1) + ((out_2 - out_1) & 1)
        } else {
            (((out_2 - out_1) >> (11 - 1)) + 1) >> 1
        }) as i16;
        k += 1;
    }
}
