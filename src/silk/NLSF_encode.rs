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

pub unsafe fn silk_NLSF_encode(
    NLSFIndices: *mut i8,
    pNLSF_Q15: *mut i16,
    psNLSF_CB: &silk_NLSF_CB_struct,
    pW_Q2: *const i16,
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
    assert!((0..=2).contains(&signalType));
    silk_NLSF_stabilize(
        std::slice::from_raw_parts_mut(pNLSF_Q15, psNLSF_CB.order as usize),
        psNLSF_CB.deltaMin_Q15,
    );
    let vla = psNLSF_CB.nVectors as usize;
    let mut err_Q24: Vec<i32> = ::std::vec::from_elem(0, vla);
    silk_NLSF_VQ(
        &mut err_Q24,
        std::slice::from_raw_parts(pNLSF_Q15, psNLSF_CB.order as usize),
        psNLSF_CB.CB1_NLSF_Q8,
        psNLSF_CB.CB1_Wght_Q9,
        psNLSF_CB.nVectors as usize,
        psNLSF_CB.order as usize,
    );
    let vla_0 = nSurvivors as usize;
    let mut tempIndices1: Vec<i32> = ::std::vec::from_elem(0, vla_0);
    silk_insertion_sort_increasing(
        &mut err_Q24,
        &mut tempIndices1,
        psNLSF_CB.nVectors as i32,
        nSurvivors,
    );
    let vla_1 = nSurvivors as usize;
    let mut RD_Q25: Vec<i32> = ::std::vec::from_elem(0, vla_1);
    let vla_2 = (nSurvivors * 16) as usize;
    let mut tempIndices2: Vec<i8> = ::std::vec::from_elem(0, vla_2);
    s = 0;
    while s < nSurvivors {
        ind1 = tempIndices1[s as usize];
        let pCB_element = &psNLSF_CB.CB1_NLSF_Q8[(ind1 * psNLSF_CB.order as i32) as usize..];
        let pCB_Wght_Q9 = &psNLSF_CB.CB1_Wght_Q9[(ind1 * psNLSF_CB.order as i32) as usize..];
        let pNLSF_Q15_slice = std::slice::from_raw_parts(pNLSF_Q15, psNLSF_CB.order as usize);
        let pW_Q2_slice = std::slice::from_raw_parts(pW_Q2, psNLSF_CB.order as usize);
        i = 0;
        while i < psNLSF_CB.order as i32 {
            NLSF_tmp_Q15[i as usize] = ((pCB_element[i as usize] as i16 as u16 as i32) << 7) as i16;
            W_tmp_Q9 = pCB_Wght_Q9[i as usize] as i32;
            res_Q10[i as usize] = ((pNLSF_Q15_slice[i as usize] as i32
                - NLSF_tmp_Q15[i as usize] as i32) as i16 as i32
                * W_tmp_Q9 as i16 as i32
                >> 14) as i16;
            W_adj_Q5[i as usize] = silk_DIV32_varQ(
                pW_Q2_slice[i as usize] as i32,
                W_tmp_Q9 as i16 as i32 * W_tmp_Q9 as i16 as i32,
                21,
            ) as i16;
            i += 1;
        }
        silk_NLSF_unpack(&mut ec_ix, &mut pred_Q8, psNLSF_CB, ind1);
        let idx_start = (s * MAX_LPC_ORDER as i32) as usize;
        RD_Q25[s as usize] = silk_NLSF_del_dec_quant(
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
        let iCDF_ptr =
            &(psNLSF_CB.CB1_iCDF)[((signalType >> 1) * psNLSF_CB.nVectors as i32) as usize..];
        if ind1 == 0 {
            prob_Q8 = 256 - iCDF_ptr[ind1 as usize] as i32;
        } else {
            prob_Q8 = iCDF_ptr[(ind1 - 1) as usize] as i32 - iCDF_ptr[ind1 as usize] as i32;
        }
        bits_q7 = ((8) << 7) - silk_lin2log(prob_Q8);
        RD_Q25[s as usize] += bits_q7 as i16 as i32 * (NLSF_mu_Q20 >> 2) as i16 as i32;
        s += 1;
    }
    silk_insertion_sort_increasing(
        &mut RD_Q25,
        std::slice::from_mut(&mut bestIndex),
        nSurvivors,
        1,
    );
    let nlsf_indices = std::slice::from_raw_parts_mut(NLSFIndices, psNLSF_CB.order as usize + 1);
    nlsf_indices[0] = tempIndices1[bestIndex as usize] as i8;
    let best_start = (bestIndex * 16) as usize;
    let order = psNLSF_CB.order as usize;
    nlsf_indices[1..1 + order].copy_from_slice(&tempIndices2[best_start..best_start + order]);
    let pNLSF_Q15_slice = std::slice::from_raw_parts_mut(pNLSF_Q15, order);
    silk_NLSF_decode(pNLSF_Q15_slice, nlsf_indices, psNLSF_CB);
    ret = RD_Q25[0];
    return ret;
}
