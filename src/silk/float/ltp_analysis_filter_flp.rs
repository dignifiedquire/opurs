//! Floating-point LTP analysis filter.
//!
//! Upstream c: `silk/float/LTP_analysis_filter_FLP.c`

use crate::silk::define::LTP_ORDER;
///
/// `x_offset` is the index within `x` where the first subframe's data starts
/// (corresponding to the `x` pointer in the original c code). The slice `x`
/// must extend backwards from `x_offset` by at least `max(pitch_l) + LTP_ORDER/2`
/// samples to cover pitch lag history.
/// Upstream c: silk/float/LTP_analysis_filter_FLP.c:silk_LTP_analysis_filter_FLP
#[allow(clippy::too_many_arguments)]
pub fn silk_ltp_analysis_filter_flp(
    ltp_res: &mut [f32],
    x: &[f32],
    x_offset: usize,
    b: &[f32],
    pitch_l: &[i32],
    inv_gains: &[f32],
    subfr_length: i32,
    nb_subfr: i32,
    pre_length: i32,
) {
    let mut btmp: [f32; 5] = [0.; 5];
    let mut inv_gain: f32;
    let mut k: i32;
    let mut _i: i32;
    let mut j: i32;
    let mut x_off: usize = x_offset;
    let mut res_off: usize = 0;
    k = 0;
    while k < nb_subfr {
        let x_lag_base: usize = x_off - pitch_l[k as usize] as usize;
        inv_gain = inv_gains[k as usize];
        _i = 0;
        while _i < LTP_ORDER as i32 {
            btmp[_i as usize] = b[(k * LTP_ORDER as i32 + _i) as usize];
            _i += 1;
        }
        _i = 0;
        while _i < subfr_length + pre_length {
            ltp_res[res_off + _i as usize] = x[x_off + _i as usize];
            j = 0;
            while j < LTP_ORDER as i32 {
                let lag_idx = (x_lag_base as isize
                    + _i as isize
                    + (LTP_ORDER as i32 / 2 - j) as isize) as usize;
                ltp_res[res_off + _i as usize] -= btmp[j as usize] * x[lag_idx];
                j += 1;
            }
            ltp_res[res_off + _i as usize] *= inv_gain;
            _i += 1;
        }
        res_off += (subfr_length + pre_length) as usize;
        x_off += subfr_length as usize;
        k += 1;
    }
}
