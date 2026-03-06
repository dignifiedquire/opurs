//! Parameter interpolation.
//!
//! Upstream c: `silk/interpolate.c`

// Interpolate two vectors
/// Upstream c: silk/interpolate.c:silk_interpolate
pub fn silk_interpolate(
    xi: &mut [i16],
    x0: &[i16],
    x1: &[i16],
    // interp. factor, weight on 2nd vector
    ifact_q2: i32,
) {
    debug_assert_eq!(xi.len(), x0.len());
    debug_assert_eq!(xi.len(), x1.len());

    debug_assert!((0..=4).contains(&ifact_q2));

    for ((xi, &x0), &x1) in xi.iter_mut().zip(x0.iter()).zip(x1.iter()) {
        let x0 = x0 as i32;
        let x1 = x1 as i32;
        *xi = (x0 + (((x1 - x0) as i16 as i32 * ifact_q2 as i16 as i32) >> 2)) as i16;
    }
}
