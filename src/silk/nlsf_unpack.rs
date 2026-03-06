//! NLSF codebook unpacking.
//!
//! Upstream c: `silk/NLSF_unpack.c`

use crate::silk::structs::silk_NLSF_CB_struct;

/// Upstream c: silk/NLSF_unpack.c:silk_NLSF_unpack
pub fn silk_nlsf_unpack(
    ec_ix: &mut [i16],
    pred_q8: &mut [u8],
    ps_nlsf_cb: &silk_NLSF_CB_struct,
    cb1_index: i32,
) {
    let mut entry: u8;
    let mut ec_sel_ptr = &ps_nlsf_cb.ec_sel[(cb1_index * ps_nlsf_cb.order as i32 / 2) as usize..];
    let mut _i = 0usize;
    while _i < ps_nlsf_cb.order as usize {
        entry = ec_sel_ptr[0];
        ec_sel_ptr = &ec_sel_ptr[1..];
        ec_ix[_i] = ((entry as i32 >> 1 & 7) as i16 as i32 * (2 * 4 + 1) as i16 as i32) as i16;
        pred_q8[_i] =
            ps_nlsf_cb.pred_q8[_i + (entry as usize & 1) * (ps_nlsf_cb.order as usize - 1)];
        ec_ix[_i + 1] = ((entry as i32 >> 5 & 7) as i16 as i32 * (2 * 4 + 1) as i16 as i32) as i16;
        pred_q8[_i + 1] = ps_nlsf_cb.pred_q8
            [_i + (entry as usize >> 4 & 1) * (ps_nlsf_cb.order as usize - 1) + 1];
        _i += 2;
    }
}
