//! NLSF VQ weight computation.
//!
//! Upstream c: `silk/NLSF_VQ_weights_laroia.c`

use crate::silk::sigproc_fix::{silk_max_int, silk_min_int};
use crate::silk::typedefs::SILK_INT16_MAX;

const NLSF_W_Q: i32 = 2;

///
/// Laroia low complexity NLSF weights
///
/// R. Laroia, N. Phamdo and N. Farvardin, "Robust and Efficient Quantization of Speech LSP
/// Parameters Using Structured Vector Quantization", Proc. IEEE Int. Conf. Acoust., Speech,
/// Signal Processing, pp. 641-644, 1991.
/// Upstream c: silk/NLSF_VQ_weights_laroia.c:silk_NLSF_VQ_weights_laroia
pub fn silk_nlsf_vq_weights_laroia(p_nlsfw_q_out: &mut [i16], p_nlsf_q15: &[i16]) {
    let mut tmp1_int: i32;
    let mut tmp2_int: i32;

    debug_assert_eq!(p_nlsf_q15.len(), p_nlsfw_q_out.len());
    let d = p_nlsf_q15.len();

    debug_assert!(d > 0);
    debug_assert_eq!(d & 1, 0);

    // First value
    tmp1_int = silk_max_int(p_nlsf_q15[0] as i32, 1);
    tmp1_int = (1 << (15 + NLSF_W_Q)) / tmp1_int;
    tmp2_int = silk_max_int(p_nlsf_q15[1] as i32 - p_nlsf_q15[0] as i32, 1);
    tmp2_int = (1 << (15 + NLSF_W_Q)) / tmp2_int;
    p_nlsfw_q_out[0] = silk_min_int(tmp1_int + tmp2_int, SILK_INT16_MAX) as i16;

    // Main loop
    let mut k = 1;
    while k < d - 1 {
        tmp1_int = silk_max_int(p_nlsf_q15[k + 1] as i32 - p_nlsf_q15[k] as i32, 1);
        tmp1_int = (1 << (15 + NLSF_W_Q)) / tmp1_int;
        p_nlsfw_q_out[k] = silk_min_int(tmp1_int + tmp2_int, SILK_INT16_MAX) as i16;
        tmp2_int = silk_max_int(p_nlsf_q15[k + 2] as i32 - p_nlsf_q15[k + 1] as i32, 1);
        tmp2_int = (1 << (15 + NLSF_W_Q)) / tmp2_int;
        p_nlsfw_q_out[k + 1] = silk_min_int(tmp1_int + tmp2_int, SILK_INT16_MAX) as i16;
        k += 2;
    }

    // Last value
    tmp1_int = silk_max_int((1 << 15) - p_nlsf_q15[d - 1] as i32, 1);
    tmp1_int = (1 << (15 + NLSF_W_Q)) / tmp1_int;
    p_nlsfw_q_out[d - 1] = silk_min_int(tmp1_int + tmp2_int, SILK_INT16_MAX) as i16;
}
