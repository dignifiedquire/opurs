//! Mid/side to left/right conversion.
//!
//! Upstream c: `silk/stereo_MS_to_LR.c`

use crate::silk::define::STEREO_INTERP_LEN_MS;

use crate::silk::structs::stereo_dec_state;

use crate::silk::macros::{silk_smlawb, silk_smulbb};
use crate::silk::sigproc_fix::{silk_rshift_round, silk_sat16};

/// Convert adaptive Mid/Side representation to Left/Right stereo signal
///
/// ```text
/// state          I/O   State
/// x1[]           I/O   Left input signal, becomes mid signal
/// x2[]           I/O   Right input signal, becomes side signal
/// pred_q13[]     I     Predictors
/// fs_k_hz         I     Samples rate (kHz)
/// frame_length   I     Number of samples
/// ```
/// Upstream c: silk/stereo_MS_to_LR.c:silk_stereo_MS_to_LR
pub fn silk_stereo_ms_to_lr(
    state: &mut stereo_dec_state,
    x1: &mut [i16],
    x2: &mut [i16],
    pred_q13: &[i32; 2],
    fs_k_hz: usize,
    frame_length: i32,
) {
    let frame_length = frame_length as usize;

    assert_eq!(x1.len(), x2.len());
    assert_eq!(x1.len(), frame_length + 2);
    assert!(STEREO_INTERP_LEN_MS * fs_k_hz <= frame_length);

    /* Buffering */
    x1[..2].copy_from_slice(&state.s_mid);
    x2[..2].copy_from_slice(&state.s_side);

    state.s_mid[..2].copy_from_slice(&x1[frame_length..]);
    state.s_side[..2].copy_from_slice(&x2[frame_length..]);

    /* Interpolate predictors and add prediction to side channel */
    let mut pred0_q13 = state.pred_prev_q13[0] as i32;
    let mut pred1_q13 = state.pred_prev_q13[1] as i32;
    let denom_q16 = ((1) << 16) / (8 * fs_k_hz) as i32;
    let delta0_q13 = silk_rshift_round(
        silk_smulbb(pred_q13[0] - state.pred_prev_q13[0] as i32, denom_q16),
        16,
    );
    let delta1_q13 = silk_rshift_round(
        silk_smulbb(pred_q13[1] - state.pred_prev_q13[1] as i32, denom_q16),
        16,
    );

    for n in 0..STEREO_INTERP_LEN_MS * fs_k_hz {
        pred0_q13 += delta0_q13;
        pred1_q13 += delta1_q13;
        let sum = (x1[n] as i32 + x1[n + 2] as i32 + ((x1[n + 1] as i32) << 1)) << 9; /* Q11 */
        let sum = silk_smlawb((x2[n + 1] as i32) << 8, sum, pred0_q13); /* Q8  */
        let sum = silk_smlawb(sum, (x1[n + 1] as i32) << 11, pred1_q13); /* Q8  */
        x2[n + 1] = silk_sat16(silk_rshift_round(sum, 8)) as i16;
    }

    let pred0_q13 = pred_q13[0];
    let pred1_q13 = pred_q13[1];

    for n in STEREO_INTERP_LEN_MS * fs_k_hz..frame_length {
        let sum = (x1[n] as i32 + x1[n + 2] as i32 + ((x1[n + 1] as i32) << 1)) << 9; /* Q11 */
        let sum = silk_smlawb((x2[n + 1] as i32) << 8, sum, pred0_q13); /* Q8  */
        let sum = silk_smlawb(sum, (x1[n + 1] as i32) << 11, pred1_q13); /* Q8  */
        x2[n + 1] = silk_sat16(silk_rshift_round(sum, 8)) as i16;
    }
    state.pred_prev_q13[0] = pred_q13[0] as i16;
    state.pred_prev_q13[1] = pred_q13[1] as i16;

    /* Convert to left/right signals */
    for n in 0..frame_length {
        let sum = x1[n + 1] as i32 + x2[n + 1] as i32;
        let diff = x1[n + 1] as i32 - x2[n + 1] as i32;

        x1[n + 1] = silk_sat16(sum) as i16;
        x2[n + 1] = silk_sat16(diff) as i16;
    }
}
