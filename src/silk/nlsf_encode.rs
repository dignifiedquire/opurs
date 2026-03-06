//! nlsf codebook encoding.
//!
//! Upstream c: `silk/NLSF_encode.c`

use crate::silk::define::MAX_LPC_ORDER;
use crate::silk::inlines::silk_div32_varq;
use crate::silk::lin2log::silk_lin2log;
use crate::silk::nlsf_decode::silk_nlsf_decode;
use crate::silk::nlsf_del_dec_quant::silk_nlsf_del_dec_quant;
use crate::silk::nlsf_stabilize::silk_nlsf_stabilize;
use crate::silk::nlsf_unpack::silk_nlsf_unpack;
use crate::silk::nlsf_vq::silk_nlsf_vq;
use crate::silk::sort::silk_insertion_sort_increasing;
use crate::silk::structs::silk_NLSF_CB_struct;

/// Upstream c: silk/NLSF_encode.c:silk_NLSF_encode
pub fn silk_nlsf_encode(
    nlsfindices: &mut [i8],
    p_nlsf_q15: &mut [i16],
    ps_nlsf_cb: &silk_NLSF_CB_struct,
    p_w_q2: &[i16],
    nlsf_mu_q20: i32,
    n_survivors: i32,
    signal_type: i32,
) -> i32 {
    let mut _i: i32;
    let mut s: i32;
    let mut ind1: i32;
    let mut best_index: i32 = 0;
    let mut prob_q8: i32;
    let mut bits_q7: i32;
    let mut w_tmp_q9: i32;

    let mut res_q10: [i16; 16] = [0; 16];
    let mut nlsf_tmp_q15: [i16; 16] = [0; 16];
    let mut w_adj_q5: [i16; 16] = [0; 16];
    let mut pred_q8: [u8; 16] = [0; 16];
    let mut ec_ix: [i16; 16] = [0; 16];
    let order = ps_nlsf_cb.order as usize;
    assert!((0..=2).contains(&signal_type));
    silk_nlsf_stabilize(&mut p_nlsf_q15[..order], ps_nlsf_cb.delta_min_q15);
    let vla = ps_nlsf_cb.n_vectors as usize;
    // n_vectors max: 32; n_survivors max: 16
    const MAX_VECTORS: usize = 32;
    const MAX_SURVIVORS: usize = 16;
    debug_assert!(vla <= MAX_VECTORS);
    debug_assert!(n_survivors as usize <= MAX_SURVIVORS);
    let mut err_q24 = [0i32; MAX_VECTORS];
    silk_nlsf_vq(
        &mut err_q24,
        &p_nlsf_q15[..order],
        ps_nlsf_cb.cb1_nlsf_q8,
        ps_nlsf_cb.cb1_wght_q9,
        ps_nlsf_cb.n_vectors as usize,
        order,
    );
    let mut temp_indices1 = [0i32; MAX_SURVIVORS];
    silk_insertion_sort_increasing(
        &mut err_q24,
        &mut temp_indices1,
        ps_nlsf_cb.n_vectors as i32,
        n_survivors,
    );
    let mut rd_q25 = [0i32; MAX_SURVIVORS];
    let mut temp_indices2 = [0i8; MAX_SURVIVORS * MAX_LPC_ORDER];
    s = 0;
    while s < n_survivors {
        ind1 = temp_indices1[s as usize];
        let p_cb_element = &ps_nlsf_cb.cb1_nlsf_q8[(ind1 * ps_nlsf_cb.order as i32) as usize..];
        let p_cb_wght_q9 = &ps_nlsf_cb.cb1_wght_q9[(ind1 * ps_nlsf_cb.order as i32) as usize..];
        _i = 0;
        while _i < ps_nlsf_cb.order as i32 {
            nlsf_tmp_q15[_i as usize] =
                ((p_cb_element[_i as usize] as i16 as u16 as i32) << 7) as i16;
            w_tmp_q9 = p_cb_wght_q9[_i as usize] as i32;
            res_q10[_i as usize] = (((p_nlsf_q15[_i as usize] as i32
                - nlsf_tmp_q15[_i as usize] as i32) as i16
                as i32
                * w_tmp_q9 as i16 as i32)
                >> 14) as i16;
            w_adj_q5[_i as usize] = silk_div32_varq(
                p_w_q2[_i as usize] as i32,
                w_tmp_q9 as i16 as i32 * w_tmp_q9 as i16 as i32,
                21,
            ) as i16;
            _i += 1;
        }
        silk_nlsf_unpack(&mut ec_ix, &mut pred_q8, ps_nlsf_cb, ind1);
        let idx_start = (s * MAX_LPC_ORDER as i32) as usize;
        rd_q25[s as usize] = silk_nlsf_del_dec_quant(
            &mut temp_indices2[idx_start..idx_start + MAX_LPC_ORDER],
            &res_q10,
            &w_adj_q5,
            &pred_q8,
            &ec_ix,
            ps_nlsf_cb.ec_rates_q5,
            ps_nlsf_cb.quant_step_size_q16 as i32,
            ps_nlsf_cb.inv_quant_step_size_q6,
            nlsf_mu_q20,
            ps_nlsf_cb.order,
        );
        let i_cdf_ptr =
            &(ps_nlsf_cb.cb1_i_cdf)[((signal_type >> 1) * ps_nlsf_cb.n_vectors as i32) as usize..];
        if ind1 == 0 {
            prob_q8 = 256 - i_cdf_ptr[ind1 as usize] as i32;
        } else {
            prob_q8 = i_cdf_ptr[(ind1 - 1) as usize] as i32 - i_cdf_ptr[ind1 as usize] as i32;
        }
        bits_q7 = ((8) << 7) - silk_lin2log(prob_q8);
        rd_q25[s as usize] += bits_q7 as i16 as i32 * (nlsf_mu_q20 >> 2) as i16 as i32;
        s += 1;
    }
    silk_insertion_sort_increasing(
        &mut rd_q25,
        std::slice::from_mut(&mut best_index),
        n_survivors,
        1,
    );
    nlsfindices[0] = temp_indices1[best_index as usize] as i8;
    let best_start = (best_index * 16) as usize;
    nlsfindices[1..1 + order].copy_from_slice(&temp_indices2[best_start..best_start + order]);
    silk_nlsf_decode(
        &mut p_nlsf_q15[..order],
        &nlsfindices[..order + 1],
        ps_nlsf_cb,
    );
    let ret: i32 = rd_q25[0];
    ret
}
