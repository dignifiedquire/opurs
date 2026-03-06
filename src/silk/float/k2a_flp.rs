//! Floating-point reflection coefficients to LPC conversion.
//!
//! Upstream c: `silk/float/k2a_FLP.c`

/// Upstream c: silk/float/k2a_FLP.c:silk_k2a_FLP
pub fn silk_k2a_flp(a: &mut [f32], rc: &[f32], order: i32) {
    let mut k: i32;
    let mut n: i32;
    let mut rck: f32;
    let mut tmp1: f32;
    let mut tmp2: f32;
    k = 0;
    while k < order {
        rck = rc[k as usize];
        n = 0;
        while n < (k + 1) >> 1 {
            tmp1 = a[n as usize];
            tmp2 = a[(k - n - 1) as usize];
            a[n as usize] = tmp1 + tmp2 * rck;
            a[(k - n - 1) as usize] = tmp2 + tmp1 * rck;
            n += 1;
        }
        a[k as usize] = -rck;
        k += 1;
    }
}
