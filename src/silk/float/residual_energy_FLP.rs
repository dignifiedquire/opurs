//! Floating-point residual energy computation.
//!
//! Upstream C: `silk/float/residual_energy_FLP.c`

use crate::silk::define::MAX_NB_SUBFR;
use crate::silk::float::energy_FLP::silk_energy_FLP;
use crate::silk::float::LPC_analysis_filter_FLP::silk_LPC_analysis_filter_FLP;

const MAX_ITERATIONS_RESIDUAL_NRG: usize = 10;
const REGULARIZATION_FACTOR: f32 = 1e-8;

/// Upstream C: silk/float/residual_energy_FLP.c:silk_residual_energy_covar_FLP
pub fn silk_residual_energy_covar_FLP(
    c: &[f32],
    wXX: &mut [f32],
    wXx: &[f32],
    wxx: f32,
    D: i32,
) -> f32 {
    debug_assert!(D >= 0);
    let d = D as usize;
    debug_assert!(c.len() >= d);
    debug_assert!(wXx.len() >= d);
    debug_assert!(wXX.len() >= d * d);

    let mut nrg = 0.0f32;
    let mut regularization = if d > 0 {
        REGULARIZATION_FACTOR * (wXX[0] + wXX[d * d - 1])
    } else {
        0.0
    };
    let mut k = 0usize;
    while k < MAX_ITERATIONS_RESIDUAL_NRG {
        nrg = wxx;

        let mut tmp = 0.0f32;
        for i in 0..d {
            tmp += wXx[i] * c[i];
        }
        nrg -= 2.0 * tmp;

        for i in 0..d {
            tmp = 0.0;
            for j in (i + 1)..d {
                tmp += wXX[i * d + j] * c[j];
            }
            nrg += c[i] * (2.0 * tmp + wXX[i * d + i] * c[i]);
        }

        if nrg > 0.0 {
            break;
        }

        for i in 0..d {
            wXX[i * d + i] += regularization;
        }
        regularization *= 2.0;
        k += 1;
    }

    if k == MAX_ITERATIONS_RESIDUAL_NRG {
        debug_assert!(nrg == 0.0);
        nrg = 1.0;
    }

    nrg
}

/// Upstream C: silk/float/residual_energy_FLP.c:silk_residual_energy_FLP
pub fn silk_residual_energy_FLP(
    nrgs: &mut [f32],
    x: &[f32],
    a: &[[f32; 16]],
    gains: &[f32],
    subfr_length: i32,
    nb_subfr: i32,
    LPC_order: i32,
) {
    let mut shift: i32 = 0;
    let mut LPC_res: [f32; 192] = [0.; 192];
    let res_off = LPC_order as usize;
    shift = LPC_order + subfr_length;
    silk_LPC_analysis_filter_FLP(
        &mut LPC_res,
        &a[0],
        &x[0..(2 * shift) as usize],
        2 * shift,
        LPC_order,
    );
    nrgs[0] = ((gains[0] * gains[0]) as f64
        * silk_energy_FLP(&LPC_res[res_off..res_off + subfr_length as usize])) as f32;
    nrgs[1] = ((gains[1] * gains[1]) as f64
        * silk_energy_FLP(
            &LPC_res[res_off + shift as usize..res_off + shift as usize + subfr_length as usize],
        )) as f32;
    if nb_subfr == MAX_NB_SUBFR as i32 {
        silk_LPC_analysis_filter_FLP(
            &mut LPC_res,
            &a[1],
            &x[(2 * shift) as usize..(4 * shift) as usize],
            2 * shift,
            LPC_order,
        );
        nrgs[2] = ((gains[2] * gains[2]) as f64
            * silk_energy_FLP(&LPC_res[res_off..res_off + subfr_length as usize]))
            as f32;
        nrgs[3] = ((gains[3] * gains[3]) as f64
            * silk_energy_FLP(
                &LPC_res
                    [res_off + shift as usize..res_off + shift as usize + subfr_length as usize],
            )) as f32;
    }
}

#[cfg(all(test, feature = "tools"))]
mod tests {
    use super::silk_residual_energy_covar_FLP as rust_silk_residual_energy_covar_FLP;

    unsafe extern "C" {
        fn silk_residual_energy_covar_FLP(
            c: *const f32,
            wXX: *mut f32,
            wXx: *const f32,
            wxx: f32,
            D: i32,
        ) -> f32;
    }

    struct Rng(u64);
    impl Rng {
        fn new(seed: u64) -> Self {
            Self(seed)
        }
        fn next_u32(&mut self) -> u32 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (self.0 >> 32) as u32
        }
        fn next_f32(&mut self) -> f32 {
            let v = self.next_u32() as f32 / (u32::MAX as f32);
            2.0 * v - 1.0
        }
    }

    #[test]
    fn residual_energy_covar_matches_upstream_c_randomized() {
        let mut rng = Rng::new(0xd3ad_beef_1234_5678);

        for d in 1usize..=16usize {
            for _ in 0..64 {
                let mut c = vec![0.0f32; d];
                let mut wxx_vec = vec![0.0f32; d];
                for i in 0..d {
                    c[i] = rng.next_f32();
                    wxx_vec[i] = rng.next_f32();
                }

                // Build a symmetric matrix exactly as the C path assumes.
                let mut wxx_mat = vec![0.0f32; d * d];
                for i in 0..d {
                    for j in i..d {
                        let v = 0.25 * rng.next_f32();
                        wxx_mat[i * d + j] = v;
                        wxx_mat[j * d + i] = v;
                    }
                }

                let scalar = rng.next_f32();
                let mut rust_mat = wxx_mat.clone();
                let mut c_mat = wxx_mat.clone();

                let rust_nrg = rust_silk_residual_energy_covar_FLP(
                    &c,
                    &mut rust_mat,
                    &wxx_vec,
                    scalar,
                    d as i32,
                );
                let c_nrg = unsafe {
                    silk_residual_energy_covar_FLP(
                        c.as_ptr(),
                        c_mat.as_mut_ptr(),
                        wxx_vec.as_ptr(),
                        scalar,
                        d as i32,
                    )
                };

                assert_eq!(rust_nrg.to_bits(), c_nrg.to_bits(), "nrg mismatch d={d}");
                assert_eq!(rust_mat.len(), c_mat.len());
                for i in 0..rust_mat.len() {
                    assert_eq!(
                        rust_mat[i].to_bits(),
                        c_mat[i].to_bits(),
                        "wXX mismatch d={d} idx={i}"
                    );
                }
            }
        }
    }

    #[test]
    fn residual_energy_covar_regularization_fallback_matches_c() {
        let d = 8usize;
        let c = vec![1.0f32; d];
        let wxx_vec = vec![0.0f32; d];
        let mut mat = vec![0.0f32; d * d];
        for i in 0..d {
            mat[i * d + i] = 1.0;
        }
        let scalar = -1.0f32;

        let mut rust_mat = mat.clone();
        let mut c_mat = mat.clone();

        let rust_nrg =
            rust_silk_residual_energy_covar_FLP(&c, &mut rust_mat, &wxx_vec, scalar, d as i32);
        let c_nrg = unsafe {
            silk_residual_energy_covar_FLP(
                c.as_ptr(),
                c_mat.as_mut_ptr(),
                wxx_vec.as_ptr(),
                scalar,
                d as i32,
            )
        };

        assert_eq!(rust_nrg.to_bits(), c_nrg.to_bits());
        for i in 0..rust_mat.len() {
            assert_eq!(rust_mat[i].to_bits(), c_mat[i].to_bits(), "idx={i}");
        }
    }
}
