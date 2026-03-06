//! Floating-point residual energy computation.
//!
//! Upstream c: `silk/float/residual_energy_FLP.c`

use crate::silk::define::MAX_NB_SUBFR;
use crate::silk::float::energy_flp::silk_energy_flp;
use crate::silk::float::lpc_analysis_filter_flp::silk_lpc_analysis_filter_flp;

const MAX_ITERATIONS_RESIDUAL_NRG: usize = 10;
const REGULARIZATION_FACTOR: f32 = 1e-8;

/// Upstream c: silk/float/residual_energy_FLP.c:silk_residual_energy_covar_FLP
pub fn silk_residual_energy_covar_flp(
    c: &[f32],
    w_xx_mat: &mut [f32],
    w_xx_vec: &[f32],
    wxx: f32,
    d: i32,
) -> f32 {
    debug_assert!(d >= 0);
    let d = d as usize;
    debug_assert!(c.len() >= d);
    debug_assert!(w_xx_vec.len() >= d);
    debug_assert!(w_xx_mat.len() >= d * d);

    let mut nrg = 0.0f32;
    let mut regularization = if d > 0 {
        REGULARIZATION_FACTOR * (w_xx_mat[0] + w_xx_mat[d * d - 1])
    } else {
        0.0
    };
    let mut k = 0usize;
    while k < MAX_ITERATIONS_RESIDUAL_NRG {
        nrg = wxx;

        let mut tmp = 0.0f32;
        for _i in 0..d {
            tmp += w_xx_vec[_i] * c[_i];
        }
        nrg -= 2.0 * tmp;

        for _i in 0..d {
            tmp = 0.0;
            for j in (_i + 1)..d {
                tmp += w_xx_mat[_i * d + j] * c[j];
            }
            nrg += c[_i] * (2.0 * tmp + w_xx_mat[_i * d + _i] * c[_i]);
        }

        if nrg > 0.0 {
            break;
        }

        for _i in 0..d {
            w_xx_mat[_i * d + _i] += regularization;
        }
        regularization *= 2.0;
        k += 1;
    }

    if k == MAX_ITERATIONS_RESIDUAL_NRG {
        #[cfg(feature = "assertions")]
        assert!(nrg == 0.0);
        nrg = 1.0;
    }

    nrg
}

/// Upstream c: silk/float/residual_energy_FLP.c:silk_residual_energy_FLP
pub fn silk_residual_energy_flp(
    nrgs: &mut [f32],
    x: &[f32],
    a: &[[f32; 16]],
    gains: &[f32],
    subfr_length: i32,
    nb_subfr: i32,
    lpc_order: i32,
) {
    let mut lpc_res: [f32; 192] = [0.; 192];
    let res_off = lpc_order as usize;
    let shift: i32 = lpc_order + subfr_length;
    silk_lpc_analysis_filter_flp(
        &mut lpc_res,
        &a[0],
        &x[0..(2 * shift) as usize],
        2 * shift,
        lpc_order,
    );
    nrgs[0] = ((gains[0] * gains[0]) as f64
        * silk_energy_flp(&lpc_res[res_off..res_off + subfr_length as usize])) as f32;
    nrgs[1] = ((gains[1] * gains[1]) as f64
        * silk_energy_flp(
            &lpc_res[res_off + shift as usize..res_off + shift as usize + subfr_length as usize],
        )) as f32;
    if nb_subfr == MAX_NB_SUBFR as i32 {
        silk_lpc_analysis_filter_flp(
            &mut lpc_res,
            &a[1],
            &x[(2 * shift) as usize..(4 * shift) as usize],
            2 * shift,
            lpc_order,
        );
        nrgs[2] = ((gains[2] * gains[2]) as f64
            * silk_energy_flp(&lpc_res[res_off..res_off + subfr_length as usize]))
            as f32;
        nrgs[3] = ((gains[3] * gains[3]) as f64
            * silk_energy_flp(
                &lpc_res
                    [res_off + shift as usize..res_off + shift as usize + subfr_length as usize],
            )) as f32;
    }
}

#[cfg(all(test, feature = "tools"))]
mod tests {
    use super::silk_residual_energy_covar_flp as rust_silk_residual_energy_covar_flp;

    unsafe extern "C" {
        #[link_name = "silk_residual_energy_covar_FLP"]
        fn silk_residual_energy_covar_flp_c(
            c: *const f32,
            w_xx: *mut f32,
            w_xx_vec: *const f32,
            wxx: f32,
            d: i32,
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
                for c_i in c.iter_mut().take(d) {
                    *c_i = rng.next_f32();
                }

                // Build a positive semidefinite matrix M = a^T * a.
                let mut a = vec![0.0f32; d * d];
                for a_i in a.iter_mut().take(d * d) {
                    *a_i = 0.25 * rng.next_f32();
                }

                let mut wxx_mat = vec![0.0f32; d * d];
                for _i in 0..d {
                    for j in 0..d {
                        let mut acc = 0.0f32;
                        for k in 0..d {
                            acc += a[k * d + _i] * a[k * d + j];
                        }
                        wxx_mat[_i * d + j] = acc;
                    }
                }

                // Set w_xx = M*c and wxx = c^T*M*c + margin, keeping nrg positive.
                let mut wxx_vec = vec![0.0f32; d];
                for _i in 0..d {
                    let mut acc = 0.0f32;
                    for j in 0..d {
                        acc += wxx_mat[_i * d + j] * c[j];
                    }
                    wxx_vec[_i] = acc;
                }
                let mut quad = 0.0f32;
                for _i in 0..d {
                    quad += c[_i] * wxx_vec[_i];
                }
                let scalar = quad + 0.1;

                let mut rust_mat = wxx_mat.clone();
                let mut c_mat = wxx_mat.clone();

                let rust_nrg = rust_silk_residual_energy_covar_flp(
                    &c,
                    &mut rust_mat,
                    &wxx_vec,
                    scalar,
                    d as i32,
                );
                let c_nrg = unsafe {
                    silk_residual_energy_covar_flp_c(
                        c.as_ptr(),
                        c_mat.as_mut_ptr(),
                        wxx_vec.as_ptr(),
                        scalar,
                        d as i32,
                    )
                };

                assert_eq!(rust_nrg.to_bits(), c_nrg.to_bits(), "nrg mismatch d={d}");
                assert_eq!(rust_mat.len(), c_mat.len());
                for _i in 0..rust_mat.len() {
                    assert_eq!(
                        rust_mat[_i].to_bits(),
                        c_mat[_i].to_bits(),
                        "w_xx mismatch d={d} idx={_i}"
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
        for _i in 0..d {
            mat[_i * d + _i] = 1.0;
        }
        let scalar = -1.0f32;

        let mut rust_mat = mat.clone();
        let mut c_mat = mat.clone();

        let rust_nrg =
            rust_silk_residual_energy_covar_flp(&c, &mut rust_mat, &wxx_vec, scalar, d as i32);
        let c_nrg = unsafe {
            silk_residual_energy_covar_flp_c(
                c.as_ptr(),
                c_mat.as_mut_ptr(),
                wxx_vec.as_ptr(),
                scalar,
                d as i32,
            )
        };

        assert_eq!(rust_nrg.to_bits(), c_nrg.to_bits());
        for _i in 0..rust_mat.len() {
            assert_eq!(rust_mat[_i].to_bits(), c_mat[_i].to_bits(), "idx={_i}");
        }
    }
}
