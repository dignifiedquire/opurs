//! Sigmoid function approximation (Q15).
//!
//! Upstream C: `silk/sigm_q15.c`

const SIGM_LUT_SLOPE_Q10: [i32; 6] = [237, 153, 73, 30, 12, 7];
const SIGM_LUT_POS_Q15: [i32; 6] = [16384, 23955, 28861, 31213, 32178, 32548];
const SIGM_LUT_NEG_Q15: [i32; 6] = [16384, 8812, 3906, 1554, 589, 219];
/// Upstream C: silk/sigm_q15.c:silk_sigm_Q15
pub fn silk_sigm_q15(mut in_q5: i32) -> i32 {
    let ind: i32;
    if in_q5 < 0 {
        in_q5 = -in_q5;
        if in_q5 >= 6 * 32 {
            0
        } else {
            ind = in_q5 >> 5;
            SIGM_LUT_NEG_Q15[ind as usize]
                - SIGM_LUT_SLOPE_Q10[ind as usize] as i16 as i32 * (in_q5 & 0x1f) as i16 as i32
        }
    } else if in_q5 >= 6 * 32 {
        32767
    } else {
        ind = in_q5 >> 5;
        SIGM_LUT_POS_Q15[ind as usize]
            + SIGM_LUT_SLOPE_Q10[ind as usize] as i16 as i32 * (in_q5 & 0x1f) as i16 as i32
    }
}
