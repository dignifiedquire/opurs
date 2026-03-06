//! Floating-point LPC analysis filter.
//!
//! Upstream c: `silk/float/LPC_analysis_filter_FLP.c`

#[inline]
fn silk_lpc_analysis_filter16_flp(r_lpc: &mut [f32], pred_coef: &[f32], s: &[f32], length: i32) {
    let mut ix: i32 = 16;
    while ix < length {
        let si = (ix - 1) as usize;
        let lpc_pred = s[si] * pred_coef[0]
            + s[si - 1] * pred_coef[1]
            + s[si - 2] * pred_coef[2]
            + s[si - 3] * pred_coef[3]
            + s[si - 4] * pred_coef[4]
            + s[si - 5] * pred_coef[5]
            + s[si - 6] * pred_coef[6]
            + s[si - 7] * pred_coef[7]
            + s[si - 8] * pred_coef[8]
            + s[si - 9] * pred_coef[9]
            + s[si - 10] * pred_coef[10]
            + s[si - 11] * pred_coef[11]
            + s[si - 12] * pred_coef[12]
            + s[si - 13] * pred_coef[13]
            + s[si - 14] * pred_coef[14]
            + s[si - 15] * pred_coef[15];
        r_lpc[ix as usize] = s[si + 1] - lpc_pred;
        ix += 1;
    }
}
#[inline]
fn silk_lpc_analysis_filter12_flp(r_lpc: &mut [f32], pred_coef: &[f32], s: &[f32], length: i32) {
    let mut ix: i32 = 12;
    while ix < length {
        let si = (ix - 1) as usize;
        let lpc_pred = s[si] * pred_coef[0]
            + s[si - 1] * pred_coef[1]
            + s[si - 2] * pred_coef[2]
            + s[si - 3] * pred_coef[3]
            + s[si - 4] * pred_coef[4]
            + s[si - 5] * pred_coef[5]
            + s[si - 6] * pred_coef[6]
            + s[si - 7] * pred_coef[7]
            + s[si - 8] * pred_coef[8]
            + s[si - 9] * pred_coef[9]
            + s[si - 10] * pred_coef[10]
            + s[si - 11] * pred_coef[11];
        r_lpc[ix as usize] = s[si + 1] - lpc_pred;
        ix += 1;
    }
}
#[inline]
fn silk_lpc_analysis_filter10_flp(r_lpc: &mut [f32], pred_coef: &[f32], s: &[f32], length: i32) {
    let mut ix: i32 = 10;
    while ix < length {
        let si = (ix - 1) as usize;
        let lpc_pred = s[si] * pred_coef[0]
            + s[si - 1] * pred_coef[1]
            + s[si - 2] * pred_coef[2]
            + s[si - 3] * pred_coef[3]
            + s[si - 4] * pred_coef[4]
            + s[si - 5] * pred_coef[5]
            + s[si - 6] * pred_coef[6]
            + s[si - 7] * pred_coef[7]
            + s[si - 8] * pred_coef[8]
            + s[si - 9] * pred_coef[9];
        r_lpc[ix as usize] = s[si + 1] - lpc_pred;
        ix += 1;
    }
}
#[inline]
fn silk_lpc_analysis_filter8_flp(r_lpc: &mut [f32], pred_coef: &[f32], s: &[f32], length: i32) {
    let mut ix: i32 = 8;
    while ix < length {
        let si = (ix - 1) as usize;
        let lpc_pred = s[si] * pred_coef[0]
            + s[si - 1] * pred_coef[1]
            + s[si - 2] * pred_coef[2]
            + s[si - 3] * pred_coef[3]
            + s[si - 4] * pred_coef[4]
            + s[si - 5] * pred_coef[5]
            + s[si - 6] * pred_coef[6]
            + s[si - 7] * pred_coef[7];
        r_lpc[ix as usize] = s[si + 1] - lpc_pred;
        ix += 1;
    }
}
#[inline]
fn silk_lpc_analysis_filter6_flp(r_lpc: &mut [f32], pred_coef: &[f32], s: &[f32], length: i32) {
    let mut ix: i32 = 6;
    while ix < length {
        let si = (ix - 1) as usize;
        let lpc_pred = s[si] * pred_coef[0]
            + s[si - 1] * pred_coef[1]
            + s[si - 2] * pred_coef[2]
            + s[si - 3] * pred_coef[3]
            + s[si - 4] * pred_coef[4]
            + s[si - 5] * pred_coef[5];
        r_lpc[ix as usize] = s[si + 1] - lpc_pred;
        ix += 1;
    }
}
/// Upstream c: silk/float/LPC_analysis_filter_FLP.c:silk_LPC_analysis_filter_FLP
pub fn silk_lpc_analysis_filter_flp(
    r_lpc: &mut [f32],
    pred_coef: &[f32],
    s: &[f32],
    length: i32,
    order: i32,
) {
    debug_assert!(order <= length);
    match order {
        6 => {
            silk_lpc_analysis_filter6_flp(r_lpc, pred_coef, s, length);
        }
        8 => {
            silk_lpc_analysis_filter8_flp(r_lpc, pred_coef, s, length);
        }
        10 => {
            silk_lpc_analysis_filter10_flp(r_lpc, pred_coef, s, length);
        }
        12 => {
            silk_lpc_analysis_filter12_flp(r_lpc, pred_coef, s, length);
        }
        16 => {
            silk_lpc_analysis_filter16_flp(r_lpc, pred_coef, s, length);
        }
        _ => {
            debug_assert!(false, "libopus: assert(0) called");
        }
    }
    r_lpc[..order as usize].fill(0.0);
}
