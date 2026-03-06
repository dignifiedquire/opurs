//! Aligned inner product computation.
//!
//! Upstream c: `silk/inner_prod_aligned.c`

/// Upstream c: silk/inner_prod_aligned.c:silk_inner_prod_aligned_scale
pub fn silk_inner_prod_aligned_scale(
    in_vec1: &[i16],
    in_vec2: &[i16],
    scale: i32,
    len: i32,
) -> i32 {
    let mut _i: i32;
    let mut sum: i32 = 0;
    _i = 0;
    while _i < len {
        sum += (in_vec1[_i as usize] as i32 * in_vec2[_i as usize] as i32) >> scale;
        _i += 1;
    }
    sum
}
