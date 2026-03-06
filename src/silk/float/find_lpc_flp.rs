//! Floating-point LPC coefficient search.
//!
//! Upstream c: `silk/float/find_LPC_FLP.c`

use crate::silk::define::MAX_NB_SUBFR;
use crate::silk::float::burg_modified_flp::silk_burg_modified_flp;
use crate::silk::float::energy_flp::silk_energy_flp;
use crate::silk::float::lpc_analysis_filter_flp::silk_lpc_analysis_filter_flp;
use crate::silk::float::wrappers_flp::{silk_a2nlsf_flp, silk_nlsf2a_flp};
use crate::silk::interpolate::silk_interpolate;
use crate::silk::structs::silk_encoder_state;
use crate::silk::typedefs::SILK_FLOAT_MAX;

/// Upstream c: silk/float/find_LPC_FLP.c:silk_find_LPC_FLP
pub fn silk_find_lpc_flp(
    ps_enc_c: &mut silk_encoder_state,
    nlsf_q15: &mut [i16],
    x: &[f32],
    min_inv_gain: f32,
) {
    let mut k: i32;

    let mut a: [f32; 16] = [0.; 16];
    let mut res_nrg: f32;
    let mut res_nrg_2nd: f32;
    let mut res_nrg_interp: f32;
    let mut nlsf0_q15: [i16; 16] = [0; 16];
    let mut a_tmp: [f32; 16] = [0.; 16];
    let mut lpc_res: [f32; 384] = [0.; 384];
    let subfr_length: i32 = ps_enc_c.subfr_length as i32 + ps_enc_c.predict_lpcorder;
    ps_enc_c.indices.nlsfinterp_coef_q2 = 4;
    res_nrg = silk_burg_modified_flp(
        &mut a,
        x,
        min_inv_gain,
        subfr_length,
        ps_enc_c.nb_subfr as i32,
        ps_enc_c.predict_lpcorder,
        ps_enc_c.arch,
    );
    if ps_enc_c.use_interpolated_nlsfs != 0
        && ps_enc_c.first_frame_after_reset == 0
        && ps_enc_c.nb_subfr == MAX_NB_SUBFR
    {
        let half_off = (MAX_NB_SUBFR as i32 / 2 * subfr_length) as usize;
        res_nrg -= silk_burg_modified_flp(
            &mut a_tmp,
            &x[half_off..],
            min_inv_gain,
            subfr_length,
            MAX_NB_SUBFR as i32 / 2,
            ps_enc_c.predict_lpcorder,
            ps_enc_c.arch,
        );
        silk_a2nlsf_flp(nlsf_q15, &a_tmp, ps_enc_c.predict_lpcorder);
        res_nrg_2nd = SILK_FLOAT_MAX;
        k = 3;
        while k >= 0 {
            silk_interpolate(
                &mut nlsf0_q15[..ps_enc_c.predict_lpcorder as usize],
                &ps_enc_c.prev_nlsfq_q15[..ps_enc_c.predict_lpcorder as usize],
                &nlsf_q15[..ps_enc_c.predict_lpcorder as usize],
                k,
            );
            silk_nlsf2a_flp(
                &mut a_tmp,
                &nlsf0_q15,
                ps_enc_c.predict_lpcorder,
                ps_enc_c.arch,
            );
            silk_lpc_analysis_filter_flp(
                &mut lpc_res,
                &a_tmp,
                &x[..(2 * subfr_length) as usize],
                2 * subfr_length,
                ps_enc_c.predict_lpcorder,
            );
            res_nrg_interp = (silk_energy_flp(
                &lpc_res[ps_enc_c.predict_lpcorder as usize..]
                    [..(subfr_length - ps_enc_c.predict_lpcorder) as usize],
            ) + silk_energy_flp(
                &lpc_res[(ps_enc_c.predict_lpcorder + subfr_length) as usize..]
                    [..(subfr_length - ps_enc_c.predict_lpcorder) as usize],
            )) as f32;
            if res_nrg_interp < res_nrg {
                res_nrg = res_nrg_interp;
                ps_enc_c.indices.nlsfinterp_coef_q2 = k as i8;
            } else if res_nrg_interp > res_nrg_2nd {
                break;
            }
            res_nrg_2nd = res_nrg_interp;
            k -= 1;
        }
    }
    if ps_enc_c.indices.nlsfinterp_coef_q2 as i32 == 4 {
        silk_a2nlsf_flp(nlsf_q15, &a, ps_enc_c.predict_lpcorder);
    }
    debug_assert!(
        ps_enc_c.indices.nlsfinterp_coef_q2 as i32 == 4
            || ps_enc_c.use_interpolated_nlsfs != 0
                && ps_enc_c.first_frame_after_reset == 0
                && ps_enc_c.nb_subfr == 4
    );
}
