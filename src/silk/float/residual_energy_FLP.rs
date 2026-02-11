//! Floating-point residual energy computation.
//!
//! Upstream C: `silk/float/residual_energy_FLP.c`

use crate::silk::define::MAX_NB_SUBFR;
use crate::silk::float::energy_FLP::silk_energy_FLP;
use crate::silk::float::LPC_analysis_filter_FLP::silk_LPC_analysis_filter_FLP;

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
