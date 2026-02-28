//! NLSF codebook encoding.
//!
//! Upstream C: `silk/NLSF_encode.c`

use crate::silk::define::MAX_LPC_ORDER;
use crate::silk::lin2log::silk_lin2log;
use crate::silk::sort::silk_insertion_sort_increasing;
use crate::silk::structs::silk_NLSF_CB_struct;
use crate::silk::Inlines::silk_DIV32_varQ;
use crate::silk::NLSF_decode::silk_NLSF_decode;
use crate::silk::NLSF_del_dec_quant::silk_NLSF_del_dec_quant;
use crate::silk::NLSF_stabilize::silk_NLSF_stabilize;
use crate::silk::NLSF_unpack::silk_NLSF_unpack;
use crate::silk::NLSF_VQ::silk_NLSF_VQ;

/// Upstream C: silk/NLSF_encode.c:silk_NLSF_encode
pub fn silk_NLSF_encode(
    NLSFIndices: &mut [i8],
    pNLSF_Q15: &mut [i16],
    psNLSF_CB: &silk_NLSF_CB_struct,
    pW_Q2: &[i16],
    NLSF_mu_Q20: i32,
    nSurvivors: i32,
    signalType: i32,
) -> i32 {
    let mut i: i32 = 0;
    let mut s: i32 = 0;
    let mut ind1: i32 = 0;
    let mut bestIndex: i32 = 0;
    let mut prob_Q8: i32 = 0;
    let mut bits_q7: i32 = 0;
    let mut W_tmp_Q9: i32 = 0;
    let mut ret: i32 = 0;
    let mut res_Q10: [i16; 16] = [0; 16];
    let mut NLSF_tmp_Q15: [i16; 16] = [0; 16];
    let mut W_adj_Q5: [i16; 16] = [0; 16];
    let mut pred_Q8: [u8; 16] = [0; 16];
    let mut ec_ix: [i16; 16] = [0; 16];
    let order = psNLSF_CB.order as usize;
    assert!((0..=2).contains(&signalType));
    silk_NLSF_stabilize(&mut pNLSF_Q15[..order], psNLSF_CB.deltaMin_Q15);
    let vla = psNLSF_CB.nVectors as usize;
    // nVectors max: 32; nSurvivors max: 16
    const MAX_VECTORS: usize = 32;
    const MAX_SURVIVORS: usize = 16;
    debug_assert!(vla <= MAX_VECTORS);
    debug_assert!(nSurvivors as usize <= MAX_SURVIVORS);
    let mut err_Q24 = [0i32; MAX_VECTORS];
    silk_NLSF_VQ(
        &mut err_Q24,
        &pNLSF_Q15[..order],
        psNLSF_CB.CB1_NLSF_Q8,
        psNLSF_CB.CB1_Wght_Q9,
        psNLSF_CB.nVectors as usize,
        order,
    );
    let mut tempIndices1 = [0i32; MAX_SURVIVORS];
    silk_insertion_sort_increasing(
        &mut err_Q24,
        &mut tempIndices1,
        psNLSF_CB.nVectors as i32,
        nSurvivors,
    );
    let mut RD_Q25 = [0i32; MAX_SURVIVORS];
    let mut tempIndices2 = [0i8; MAX_SURVIVORS * MAX_LPC_ORDER];
    s = 0;
    while s < nSurvivors {
        ind1 = unsafe { *tempIndices1.get_unchecked(s as usize) };
        let pCB_element = &psNLSF_CB.CB1_NLSF_Q8[(ind1 * psNLSF_CB.order as i32) as usize..];
        let pCB_Wght_Q9 = &psNLSF_CB.CB1_Wght_Q9[(ind1 * psNLSF_CB.order as i32) as usize..];
        i = 0;
        while i < psNLSF_CB.order as i32 {
            unsafe {
                *NLSF_tmp_Q15.get_unchecked_mut(i as usize) =
                    ((*pCB_element.get_unchecked(i as usize) as i16 as u16 as i32) << 7) as i16;
                W_tmp_Q9 = *pCB_Wght_Q9.get_unchecked(i as usize) as i32;
                *res_Q10.get_unchecked_mut(i as usize) = (((*pNLSF_Q15.get_unchecked(i as usize)
                    as i32
                    - *NLSF_tmp_Q15.get_unchecked(i as usize) as i32)
                    as i16 as i32
                    * W_tmp_Q9 as i16 as i32)
                    >> 14) as i16;
                *W_adj_Q5.get_unchecked_mut(i as usize) = silk_DIV32_varQ(
                    *pW_Q2.get_unchecked(i as usize) as i32,
                    W_tmp_Q9 as i16 as i32 * W_tmp_Q9 as i16 as i32,
                    21,
                ) as i16;
            }
            i += 1;
        }
        silk_NLSF_unpack(&mut ec_ix, &mut pred_Q8, psNLSF_CB, ind1);
        let idx_start = (s * MAX_LPC_ORDER as i32) as usize;
        unsafe {
            *RD_Q25.get_unchecked_mut(s as usize) = silk_NLSF_del_dec_quant(
                &mut tempIndices2[idx_start..idx_start + MAX_LPC_ORDER],
                &res_Q10,
                &W_adj_Q5,
                &pred_Q8,
                &ec_ix,
                psNLSF_CB.ec_Rates_Q5,
                psNLSF_CB.quantStepSize_Q16 as i32,
                psNLSF_CB.invQuantStepSize_Q6,
                NLSF_mu_Q20,
                psNLSF_CB.order,
            );
        }
        let iCDF_ptr =
            &(psNLSF_CB.CB1_iCDF)[((signalType >> 1) * psNLSF_CB.nVectors as i32) as usize..];
        if ind1 == 0 {
            prob_Q8 = 256 - iCDF_ptr[ind1 as usize] as i32;
        } else {
            prob_Q8 = iCDF_ptr[(ind1 - 1) as usize] as i32 - iCDF_ptr[ind1 as usize] as i32;
        }
        bits_q7 = ((8) << 7) - silk_lin2log(prob_Q8);
        unsafe {
            *RD_Q25.get_unchecked_mut(s as usize) +=
                bits_q7 as i16 as i32 * (NLSF_mu_Q20 >> 2) as i16 as i32;
        }
        s += 1;
    }
    silk_insertion_sort_increasing(
        &mut RD_Q25,
        std::slice::from_mut(&mut bestIndex),
        nSurvivors,
        1,
    );
    NLSFIndices[0] = unsafe { *tempIndices1.get_unchecked(bestIndex as usize) } as i8;
    let best_start = (bestIndex * 16) as usize;
    NLSFIndices[1..1 + order].copy_from_slice(&tempIndices2[best_start..best_start + order]);
    silk_NLSF_decode(
        &mut pNLSF_Q15[..order],
        &NLSFIndices[..order + 1],
        psNLSF_CB,
    );
    ret = RD_Q25[0];
    ret
}
