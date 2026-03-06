//! NLSF processing and interpolation.
//!
//! Upstream c: `silk/process_NLSFs.c`

use crate::silk::interpolate::silk_interpolate;
use crate::silk::nlsf2a::silk_nlsf2a;
use crate::silk::nlsf_encode::silk_nlsf_encode;
use crate::silk::nlsf_vq_weights_laroia::silk_nlsf_vq_weights_laroia;
use crate::silk::structs::silk_encoder_state;

/// Upstream c: silk/process_NLSFs.c:silk_process_NLSFs
pub fn silk_process_nlsfs(
    ps_enc_c: &mut silk_encoder_state,
    pred_coef_q12: &mut [[i16; 16]; 2],
    p_nlsf_q15: &mut [i16],
    prev_nlsfq_q15: &[i16],
) {
    let mut _i: i32;

    let mut nlsf_mu_q20: i32;
    let i_sqr_q15: i16;
    let mut p_nlsf0_temp_q15: [i16; 16] = [0; 16];
    let mut p_nlsfw_qw: [i16; 16] = [0; 16];
    let mut p_nlsfw0_temp_qw: [i16; 16] = [0; 16];
    assert!(
        ps_enc_c.use_interpolated_nlsfs == 1
            || ps_enc_c.indices.nlsfinterp_coef_q2 as i32 == (1) << 2
    );
    nlsf_mu_q20 = ((0.003f64 * ((1) << 20) as f64 + 0.5f64) as i32 as i64
        + (((-0.001f64 * ((1) << 28) as f64 + 0.5f64) as i32 as i64
            * ps_enc_c.speech_activity_q8 as i16 as i64)
            >> 16)) as i32;
    if ps_enc_c.nb_subfr == 2 {
        nlsf_mu_q20 = nlsf_mu_q20 + (nlsf_mu_q20 >> 1);
    }
    assert!(nlsf_mu_q20 > 0);
    silk_nlsf_vq_weights_laroia(
        &mut p_nlsfw_qw[..ps_enc_c.predict_lpcorder as usize],
        &p_nlsf_q15[..ps_enc_c.predict_lpcorder as usize],
    );
    let do_interpolate: i32 = (ps_enc_c.use_interpolated_nlsfs == 1
        && (ps_enc_c.indices.nlsfinterp_coef_q2 as i32) < 4) as i32;
    if do_interpolate != 0 {
        silk_interpolate(
            &mut p_nlsf0_temp_q15[..ps_enc_c.predict_lpcorder as usize],
            &prev_nlsfq_q15[..ps_enc_c.predict_lpcorder as usize],
            &p_nlsf_q15[..ps_enc_c.predict_lpcorder as usize],
            ps_enc_c.indices.nlsfinterp_coef_q2 as i32,
        );
        silk_nlsf_vq_weights_laroia(
            &mut p_nlsfw0_temp_qw[..ps_enc_c.predict_lpcorder as usize],
            &p_nlsf0_temp_q15[..ps_enc_c.predict_lpcorder as usize],
        );
        i_sqr_q15 = (((ps_enc_c.indices.nlsfinterp_coef_q2 as i16 as i32
            * ps_enc_c.indices.nlsfinterp_coef_q2 as i16 as i32) as u32)
            << 11) as i32 as i16;
        _i = 0;
        while _i < ps_enc_c.predict_lpcorder {
            p_nlsfw_qw[_i as usize] = ((p_nlsfw_qw[_i as usize] as i32 >> 1)
                + ((p_nlsfw0_temp_qw[_i as usize] as i32 * i_sqr_q15 as i32) >> 16))
                as i16;
            _i += 1;
        }
    }
    silk_nlsf_encode(
        &mut ps_enc_c.indices.nlsfindices,
        p_nlsf_q15,
        ps_enc_c.ps_nlsf_cb,
        &p_nlsfw_qw,
        nlsf_mu_q20,
        ps_enc_c.nlsf_msvq_survivors,
        ps_enc_c.indices.signal_type as i32,
    );
    silk_nlsf2a(
        &mut pred_coef_q12[1][..ps_enc_c.predict_lpcorder as usize],
        &p_nlsf_q15[..ps_enc_c.predict_lpcorder as usize],
        ps_enc_c.arch,
    );
    if do_interpolate != 0 {
        silk_interpolate(
            &mut p_nlsf0_temp_q15[..ps_enc_c.predict_lpcorder as usize],
            &prev_nlsfq_q15[..ps_enc_c.predict_lpcorder as usize],
            &p_nlsf_q15[..ps_enc_c.predict_lpcorder as usize],
            ps_enc_c.indices.nlsfinterp_coef_q2 as i32,
        );
        silk_nlsf2a(
            &mut pred_coef_q12[0][..ps_enc_c.predict_lpcorder as usize],
            &p_nlsf0_temp_q15[..ps_enc_c.predict_lpcorder as usize],
            ps_enc_c.arch,
        );
    } else {
        assert!(ps_enc_c.predict_lpcorder <= 16);
        let order = ps_enc_c.predict_lpcorder as usize;
        let [ref mut dst, ref src] = *pred_coef_q12;
        dst[..order].copy_from_slice(&src[..order]);
    };
}
