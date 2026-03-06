//! Floating-point LPC coefficient search.
//!
//! Upstream C: `silk/float/find_LPC_FLP.c`

use crate::silk::define::MAX_NB_SUBFR;
use crate::silk::float::burg_modified_flp::silk_burg_modified_flp;
use crate::silk::float::energy_flp::silk_energy_flp;
use crate::silk::float::lpc_analysis_filter_flp::silk_lpc_analysis_filter_flp;
use crate::silk::float::wrappers_flp::{silk_a2nlsf_flp, silk_nlsf2a_flp};
use crate::silk::interpolate::silk_interpolate;
use crate::silk::structs::silk_encoder_state;
use crate::silk::typedefs::SILK_FLOAT_MAX;

/// Upstream C: silk/float/find_LPC_FLP.c:silk_find_LPC_FLP
pub fn silk_find_lpc_flp(
    psEncC: &mut silk_encoder_state,
    NLSF_Q15: &mut [i16],
    x: &[f32],
    minInvGain: f32,
) {
    let mut k: i32;

    let mut a: [f32; 16] = [0.; 16];
    let mut res_nrg: f32;
    let mut res_nrg_2nd: f32;
    let mut res_nrg_interp: f32;
    let mut NLSF0_Q15: [i16; 16] = [0; 16];
    let mut a_tmp: [f32; 16] = [0.; 16];
    let mut LPC_res: [f32; 384] = [0.; 384];
    let subfr_length: i32 = psEncC.subfr_length as i32 + psEncC.predictLPCOrder;
    psEncC.indices.NLSFInterpCoef_Q2 = 4;
    res_nrg = silk_burg_modified_flp(
        &mut a,
        x,
        minInvGain,
        subfr_length,
        psEncC.nb_subfr as i32,
        psEncC.predictLPCOrder,
        psEncC.arch,
    );
    if psEncC.useInterpolatedNLSFs != 0
        && psEncC.first_frame_after_reset == 0
        && psEncC.nb_subfr == MAX_NB_SUBFR
    {
        let half_off = (MAX_NB_SUBFR as i32 / 2 * subfr_length) as usize;
        res_nrg -= silk_burg_modified_flp(
            &mut a_tmp,
            &x[half_off..],
            minInvGain,
            subfr_length,
            MAX_NB_SUBFR as i32 / 2,
            psEncC.predictLPCOrder,
            psEncC.arch,
        );
        silk_a2nlsf_flp(NLSF_Q15, &a_tmp, psEncC.predictLPCOrder);
        res_nrg_2nd = SILK_FLOAT_MAX;
        k = 3;
        while k >= 0 {
            silk_interpolate(
                &mut NLSF0_Q15[..psEncC.predictLPCOrder as usize],
                &psEncC.prev_NLSFq_Q15[..psEncC.predictLPCOrder as usize],
                &NLSF_Q15[..psEncC.predictLPCOrder as usize],
                k,
            );
            silk_nlsf2a_flp(&mut a_tmp, &NLSF0_Q15, psEncC.predictLPCOrder, psEncC.arch);
            silk_lpc_analysis_filter_flp(
                &mut LPC_res,
                &a_tmp,
                &x[..(2 * subfr_length) as usize],
                2 * subfr_length,
                psEncC.predictLPCOrder,
            );
            res_nrg_interp = (silk_energy_flp(
                &LPC_res[psEncC.predictLPCOrder as usize..]
                    [..(subfr_length - psEncC.predictLPCOrder) as usize],
            ) + silk_energy_flp(
                &LPC_res[(psEncC.predictLPCOrder + subfr_length) as usize..]
                    [..(subfr_length - psEncC.predictLPCOrder) as usize],
            )) as f32;
            if res_nrg_interp < res_nrg {
                res_nrg = res_nrg_interp;
                psEncC.indices.NLSFInterpCoef_Q2 = k as i8;
            } else if res_nrg_interp > res_nrg_2nd {
                break;
            }
            res_nrg_2nd = res_nrg_interp;
            k -= 1;
        }
    }
    if psEncC.indices.NLSFInterpCoef_Q2 as i32 == 4 {
        silk_a2nlsf_flp(NLSF_Q15, &a, psEncC.predictLPCOrder);
    }
    debug_assert!(
        psEncC.indices.NLSFInterpCoef_Q2 as i32 == 4
            || psEncC.useInterpolatedNLSFs != 0
                && psEncC.first_frame_after_reset == 0
                && psEncC.nb_subfr == 4
    );
}
