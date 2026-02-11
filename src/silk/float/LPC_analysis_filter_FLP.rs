//! Floating-point LPC analysis filter.
//!
//! Upstream C: `silk/float/LPC_analysis_filter_FLP.c`

#[inline]
fn silk_LPC_analysis_filter16_FLP(r_LPC: &mut [f32], PredCoef: &[f32], s: &[f32], length: i32) {
    let mut ix: i32 = 16;
    while ix < length {
        let si = (ix - 1) as usize;
        let LPC_pred = s[si] * PredCoef[0]
            + s[si - 1] * PredCoef[1]
            + s[si - 2] * PredCoef[2]
            + s[si - 3] * PredCoef[3]
            + s[si - 4] * PredCoef[4]
            + s[si - 5] * PredCoef[5]
            + s[si - 6] * PredCoef[6]
            + s[si - 7] * PredCoef[7]
            + s[si - 8] * PredCoef[8]
            + s[si - 9] * PredCoef[9]
            + s[si - 10] * PredCoef[10]
            + s[si - 11] * PredCoef[11]
            + s[si - 12] * PredCoef[12]
            + s[si - 13] * PredCoef[13]
            + s[si - 14] * PredCoef[14]
            + s[si - 15] * PredCoef[15];
        r_LPC[ix as usize] = s[si + 1] - LPC_pred;
        ix += 1;
    }
}
#[inline]
fn silk_LPC_analysis_filter12_FLP(r_LPC: &mut [f32], PredCoef: &[f32], s: &[f32], length: i32) {
    let mut ix: i32 = 12;
    while ix < length {
        let si = (ix - 1) as usize;
        let LPC_pred = s[si] * PredCoef[0]
            + s[si - 1] * PredCoef[1]
            + s[si - 2] * PredCoef[2]
            + s[si - 3] * PredCoef[3]
            + s[si - 4] * PredCoef[4]
            + s[si - 5] * PredCoef[5]
            + s[si - 6] * PredCoef[6]
            + s[si - 7] * PredCoef[7]
            + s[si - 8] * PredCoef[8]
            + s[si - 9] * PredCoef[9]
            + s[si - 10] * PredCoef[10]
            + s[si - 11] * PredCoef[11];
        r_LPC[ix as usize] = s[si + 1] - LPC_pred;
        ix += 1;
    }
}
#[inline]
fn silk_LPC_analysis_filter10_FLP(r_LPC: &mut [f32], PredCoef: &[f32], s: &[f32], length: i32) {
    let mut ix: i32 = 10;
    while ix < length {
        let si = (ix - 1) as usize;
        let LPC_pred = s[si] * PredCoef[0]
            + s[si - 1] * PredCoef[1]
            + s[si - 2] * PredCoef[2]
            + s[si - 3] * PredCoef[3]
            + s[si - 4] * PredCoef[4]
            + s[si - 5] * PredCoef[5]
            + s[si - 6] * PredCoef[6]
            + s[si - 7] * PredCoef[7]
            + s[si - 8] * PredCoef[8]
            + s[si - 9] * PredCoef[9];
        r_LPC[ix as usize] = s[si + 1] - LPC_pred;
        ix += 1;
    }
}
#[inline]
fn silk_LPC_analysis_filter8_FLP(r_LPC: &mut [f32], PredCoef: &[f32], s: &[f32], length: i32) {
    let mut ix: i32 = 8;
    while ix < length {
        let si = (ix - 1) as usize;
        let LPC_pred = s[si] * PredCoef[0]
            + s[si - 1] * PredCoef[1]
            + s[si - 2] * PredCoef[2]
            + s[si - 3] * PredCoef[3]
            + s[si - 4] * PredCoef[4]
            + s[si - 5] * PredCoef[5]
            + s[si - 6] * PredCoef[6]
            + s[si - 7] * PredCoef[7];
        r_LPC[ix as usize] = s[si + 1] - LPC_pred;
        ix += 1;
    }
}
#[inline]
fn silk_LPC_analysis_filter6_FLP(r_LPC: &mut [f32], PredCoef: &[f32], s: &[f32], length: i32) {
    let mut ix: i32 = 6;
    while ix < length {
        let si = (ix - 1) as usize;
        let LPC_pred = s[si] * PredCoef[0]
            + s[si - 1] * PredCoef[1]
            + s[si - 2] * PredCoef[2]
            + s[si - 3] * PredCoef[3]
            + s[si - 4] * PredCoef[4]
            + s[si - 5] * PredCoef[5];
        r_LPC[ix as usize] = s[si + 1] - LPC_pred;
        ix += 1;
    }
}
/// Upstream C: silk/float/LPC_analysis_filter_FLP.c:silk_LPC_analysis_filter_FLP
pub fn silk_LPC_analysis_filter_FLP(
    r_LPC: &mut [f32],
    PredCoef: &[f32],
    s: &[f32],
    length: i32,
    Order: i32,
) {
    assert!(Order <= length);
    match Order {
        6 => {
            silk_LPC_analysis_filter6_FLP(r_LPC, PredCoef, s, length);
        }
        8 => {
            silk_LPC_analysis_filter8_FLP(r_LPC, PredCoef, s, length);
        }
        10 => {
            silk_LPC_analysis_filter10_FLP(r_LPC, PredCoef, s, length);
        }
        12 => {
            silk_LPC_analysis_filter12_FLP(r_LPC, PredCoef, s, length);
        }
        16 => {
            silk_LPC_analysis_filter16_FLP(r_LPC, PredCoef, s, length);
        }
        _ => {
            panic!("libopus: assert(0) called");
        }
    }
    r_LPC[..Order as usize].fill(0.0);
}
