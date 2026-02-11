//! Floating-point LTP analysis filter.
//!
//! Upstream C: `silk/float/LTP_analysis_filter_FLP.c`

use crate::silk::define::LTP_ORDER;
/// Upstream C: silk/float/LTP_analysis_filter_FLP.c:silk_LTP_analysis_filter_FLP
///
/// `x_offset` is the index within `x` where the first subframe's data starts
/// (corresponding to the `x` pointer in the original C code). The slice `x`
/// must extend backwards from `x_offset` by at least `max(pitchL) + LTP_ORDER/2`
/// samples to cover pitch lag history.
pub fn silk_LTP_analysis_filter_FLP(
    LTP_res: &mut [f32],
    x: &[f32],
    x_offset: usize,
    B: &[f32],
    pitchL: &[i32],
    invGains: &[f32],
    subfr_length: i32,
    nb_subfr: i32,
    pre_length: i32,
) {
    let mut Btmp: [f32; 5] = [0.; 5];
    let mut inv_gain: f32 = 0.;
    let mut k: i32 = 0;
    let mut i: i32 = 0;
    let mut j: i32 = 0;
    let mut x_off: usize = x_offset;
    let mut res_off: usize = 0;
    k = 0;
    while k < nb_subfr {
        let x_lag_base: usize = x_off - pitchL[k as usize] as usize;
        inv_gain = invGains[k as usize];
        i = 0;
        while i < LTP_ORDER as i32 {
            Btmp[i as usize] = B[(k * LTP_ORDER as i32 + i) as usize];
            i += 1;
        }
        i = 0;
        while i < subfr_length + pre_length {
            LTP_res[res_off + i as usize] = x[x_off + i as usize];
            j = 0;
            while j < LTP_ORDER as i32 {
                let lag_idx = (x_lag_base as isize
                    + i as isize
                    + (LTP_ORDER as i32 / 2 - j) as isize) as usize;
                LTP_res[res_off + i as usize] -= Btmp[j as usize] * x[lag_idx];
                j += 1;
            }
            LTP_res[res_off + i as usize] *= inv_gain;
            i += 1;
        }
        res_off += (subfr_length + pre_length) as usize;
        x_off += subfr_length as usize;
        k += 1;
    }
}
