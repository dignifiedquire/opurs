//! Floating-point warped autocorrelation.
//!
//! Upstream c: `silk/float/warped_autocorrelation_FLP.c`

/// Upstream c: silk/float/warped_autocorrelation_FLP.c:silk_warped_autocorrelation_FLP
pub fn silk_warped_autocorrelation_flp(
    corr: &mut [f32],
    input: &[f32],
    warping: f32,
    length: i32,
    order: i32,
) {
    let mut n: i32;
    let mut _i: i32;
    let mut tmp1: f64;
    let mut tmp2: f64;
    let mut state: [f64; 25] = [
        0 as f64, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0.,
    ];
    let mut c: [f64; 25] = [
        0 as f64, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0.,
    ];
    debug_assert!(order & 1 == 0);
    n = 0;
    while n < length {
        tmp1 = input[n as usize] as f64;
        _i = 0;
        while _i < order {
            // Use two multiplies instead of factoring to reduce dependency chain
            tmp2 = state[_i as usize] + warping as f64 * state[(_i + 1) as usize]
                - warping as f64 * tmp1;
            state[_i as usize] = tmp1;
            c[_i as usize] += state[0_usize] * tmp1;
            tmp1 = state[(_i + 1) as usize] + warping as f64 * state[(_i + 2) as usize]
                - warping as f64 * tmp2;
            state[(_i + 1) as usize] = tmp2;
            c[(_i + 1) as usize] += state[0_usize] * tmp2;
            _i += 2;
        }
        state[order as usize] = tmp1;
        c[order as usize] += state[0_usize] * tmp1;
        n += 1;
    }
    _i = 0;
    while _i < order + 1 {
        corr[_i as usize] = c[_i as usize] as f32;
        _i += 1;
    }
}
