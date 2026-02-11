/// Upstream C: silk/inner_prod_aligned.c:silk_inner_prod_aligned_scale
pub fn silk_inner_prod_aligned_scale(inVec1: &[i16], inVec2: &[i16], scale: i32, len: i32) -> i32 {
    let mut i: i32 = 0;
    let mut sum: i32 = 0;
    i = 0;
    while i < len {
        sum += (inVec1[i as usize] as i32 * inVec2[i as usize] as i32) >> scale;
        i += 1;
    }
    sum
}
