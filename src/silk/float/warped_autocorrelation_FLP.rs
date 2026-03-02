//! Floating-point warped autocorrelation.
//!
//! Upstream C: `silk/float/warped_autocorrelation_FLP.c`

/// Upstream C: silk/float/warped_autocorrelation_FLP.c:silk_warped_autocorrelation_FLP
pub fn silk_warped_autocorrelation_FLP(
    corr: &mut [f32],
    input: &[f32],
    warping: f32,
    length: i32,
    order: i32,
) {
    let mut n: i32 = 0;
    let mut i: i32 = 0;
    let mut tmp1: f64 = 0.;
    let mut tmp2: f64 = 0.;
    let mut state: [f64; 25] = [
        0 as f64, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0.,
    ];
    let mut C: [f64; 25] = [
        0 as f64, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0.,
    ];
    debug_assert!(order & 1 == 0);
    n = 0;
    while n < length {
        tmp1 = unsafe { *input.get_unchecked(n as usize) } as f64;
        i = 0;
        while i < order {
            unsafe {
                // Use two multiplies instead of factoring to reduce dependency chain
                tmp2 = *state.get_unchecked(i as usize)
                    + warping as f64 * *state.get_unchecked((i + 1) as usize)
                    - warping as f64 * tmp1;
                *state.get_unchecked_mut(i as usize) = tmp1;
                *C.get_unchecked_mut(i as usize) += *state.get_unchecked(0) * tmp1;
                tmp1 = *state.get_unchecked((i + 1) as usize)
                    + warping as f64 * *state.get_unchecked((i + 2) as usize)
                    - warping as f64 * tmp2;
                *state.get_unchecked_mut((i + 1) as usize) = tmp2;
                *C.get_unchecked_mut((i + 1) as usize) += *state.get_unchecked(0) * tmp2;
            }
            i += 2;
        }
        unsafe {
            *state.get_unchecked_mut(order as usize) = tmp1;
            *C.get_unchecked_mut(order as usize) += *state.get_unchecked(0) * tmp1;
        }
        n += 1;
    }
    i = 0;
    while i < order + 1 {
        unsafe {
            *corr.get_unchecked_mut(i as usize) = *C.get_unchecked(i as usize) as f32;
        }
        i += 1;
    }
}
