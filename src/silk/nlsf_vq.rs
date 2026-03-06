//! nlsf vector quantization.
//!
//! Upstream c: `silk/NLSF_VQ.c`

///
/// Compute quantization errors for an lpc_order element input vector for a VQ codebook
/// Upstream c: silk/NLSF_VQ.c:silk_NLSF_VQ
pub fn silk_nlsf_vq(
    // Quantization errors [k]
    err_q24: &mut [i32],
    // Input vectors to be quantized [lpc_order]
    in_q15: &[i16],
    // Codebook vectors [k*lpc_order]
    p_cb_q8: &[u8],
    // Codebook weights [k*lpc_order]
    p_wght_q9: &[i16],
    // Number of codebook vectors
    k: usize,
    // Number of LPCs
    lpc_order: usize,
) {
    assert_eq!(err_q24.len(), k);
    assert_eq!(in_q15.len(), lpc_order);
    assert_eq!(p_cb_q8.len(), k * lpc_order);
    assert_eq!(p_wght_q9.len(), k * lpc_order);

    assert_eq!(lpc_order & 1, 0);

    let mut diff_q15: i32;
    let mut diffw_q24: i32;
    let mut sum_error_q24: i32;
    let mut pred_q24: i32;
    let mut cb_q8_ptr = p_cb_q8;
    let mut w_q9_ptr = p_wght_q9;

    for err in err_q24.iter_mut() {
        sum_error_q24 = 0;
        pred_q24 = 0;

        for m in (0..=lpc_order - 2).rev().step_by(2) {
            diff_q15 = in_q15[m + 1] as i32 - ((cb_q8_ptr[m + 1] as i32 as u32) << 7) as i32;
            diffw_q24 = diff_q15 as i16 as i32 * w_q9_ptr[m + 1] as i32;
            sum_error_q24 += (diffw_q24 - (pred_q24 >> 1)).abs();
            pred_q24 = diffw_q24;
            diff_q15 = in_q15[m] as i32 - ((cb_q8_ptr[m] as i32 as u32) << 7) as i32;
            diffw_q24 = diff_q15 as i16 as i32 * w_q9_ptr[m] as i32;
            sum_error_q24 += (diffw_q24 - (pred_q24 >> 1)).abs();
            pred_q24 = diffw_q24;
        }
        *err = sum_error_q24;
        cb_q8_ptr = &cb_q8_ptr[lpc_order..];
        w_q9_ptr = &w_q9_ptr[lpc_order..];
    }
}
