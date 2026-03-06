//! Floating-point Schur algorithm.
//!
//! Upstream c: `silk/float/schur_FLP.c`

/// Upstream c: silk/float/schur_FLP.c:silk_schur_FLP
pub fn silk_schur_flp(refl_coef: &mut [f32], auto_corr: &[f32], order: i32) -> f32 {
    let mut k: i32;
    let mut n: i32;
    let mut c: [[f64; 2]; 25] = [[0.; 2]; 25];
    let mut ctmp1: f64;
    let mut ctmp2: f64;
    let mut rc_tmp: f64;
    debug_assert!((0..=24).contains(&order));
    k = 0;
    loop {
        c[k as usize][1_usize] = auto_corr[k as usize] as f64;
        c[k as usize][0_usize] = c[k as usize][1_usize];
        k += 1;
        if k > order {
            break;
        }
    }
    k = 0;
    while k < order {
        rc_tmp = -c[(k + 1) as usize][0_usize]
            / (if c[0_usize][1_usize] > 1e-9f32 as f64 {
                c[0_usize][1_usize]
            } else {
                1e-9f32 as f64
            });
        refl_coef[k as usize] = rc_tmp as f32;
        n = 0;
        while n < order - k {
            ctmp1 = c[(n + k + 1) as usize][0_usize];
            ctmp2 = c[n as usize][1_usize];
            c[(n + k + 1) as usize][0_usize] = ctmp1 + ctmp2 * rc_tmp;
            c[n as usize][1_usize] = ctmp2 + ctmp1 * rc_tmp;
            n += 1;
        }
        k += 1;
    }
    c[0_usize][1_usize] as f32
}
