//! Floating-point Schur algorithm.
//!
//! Upstream C: `silk/float/schur_FLP.c`

/// Upstream C: silk/float/schur_FLP.c:silk_schur_FLP
pub fn silk_schur_FLP(refl_coef: &mut [f32], auto_corr: &[f32], order: i32) -> f32 {
    let mut k: i32 = 0;
    let mut n: i32 = 0;
    let mut C: [[f64; 2]; 25] = [[0.; 2]; 25];
    let mut Ctmp1: f64 = 0.;
    let mut Ctmp2: f64 = 0.;
    let mut rc_tmp: f64 = 0.;
    assert!((0..=24).contains(&order));
    k = 0;
    loop {
        C[k as usize][1_usize] = auto_corr[k as usize] as f64;
        C[k as usize][0_usize] = C[k as usize][1_usize];
        k += 1;
        if k > order {
            break;
        }
    }
    k = 0;
    while k < order {
        rc_tmp = -C[(k + 1) as usize][0_usize]
            / (if C[0_usize][1_usize] > 1e-9f32 as f64 {
                C[0_usize][1_usize]
            } else {
                1e-9f32 as f64
            });
        refl_coef[k as usize] = rc_tmp as f32;
        n = 0;
        while n < order - k {
            Ctmp1 = C[(n + k + 1) as usize][0_usize];
            Ctmp2 = C[n as usize][1_usize];
            C[(n + k + 1) as usize][0_usize] = Ctmp1 + Ctmp2 * rc_tmp;
            C[n as usize][1_usize] = Ctmp2 + Ctmp1 * rc_tmp;
            n += 1;
        }
        k += 1;
    }
    C[0_usize][1_usize] as f32
}
