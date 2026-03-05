//! Mid/side to left/right conversion.
//!
//! Upstream C: `silk/stereo_MS_to_LR.c`

use crate::silk::define::STEREO_INTERP_LEN_MS;

use crate::silk::structs::stereo_dec_state;

use crate::silk::macros::{silk_smlawb, silk_smulbb};
use crate::silk::SigProc_FIX::{silk_rshift_round, silk_sat16};

/// Convert adaptive Mid/Side representation to Left/Right stereo signal
///
/// ```text
/// state          I/O   State
/// x1[]           I/O   Left input signal, becomes mid signal
/// x2[]           I/O   Right input signal, becomes side signal
/// pred_Q13[]     I     Predictors
/// fs_kHz         I     Samples rate (kHz)
/// frame_length   I     Number of samples
/// ```
/// Upstream C: silk/stereo_MS_to_LR.c:silk_stereo_MS_to_LR
pub fn silk_stereo_ms_to_lr(
    state: &mut stereo_dec_state,
    x1: &mut [i16],
    x2: &mut [i16],
    pred_Q13: &[i32; 2],
    fs_kHz: usize,
    frame_length: i32,
) {
    let frame_length = frame_length as usize;

    assert_eq!(x1.len(), x2.len());
    assert_eq!(x1.len(), frame_length + 2);
    assert!(STEREO_INTERP_LEN_MS * fs_kHz <= frame_length);

    /* Buffering */
    x1[..2].copy_from_slice(&state.sMid);
    x2[..2].copy_from_slice(&state.sSide);

    state.sMid[..2].copy_from_slice(&x1[frame_length..]);
    state.sSide[..2].copy_from_slice(&x2[frame_length..]);

    /* Interpolate predictors and add prediction to side channel */
    let mut pred0_Q13 = state.pred_prev_Q13[0] as i32;
    let mut pred1_Q13 = state.pred_prev_Q13[1] as i32;
    let denom_Q16 = ((1) << 16) / (8 * fs_kHz) as i32;
    let delta0_Q13 = silk_rshift_round(
        silk_smulbb(pred_Q13[0] - state.pred_prev_Q13[0] as i32, denom_Q16),
        16,
    );
    let delta1_Q13 = silk_rshift_round(
        silk_smulbb(pred_Q13[1] - state.pred_prev_Q13[1] as i32, denom_Q16),
        16,
    );

    for n in 0..STEREO_INTERP_LEN_MS * fs_kHz {
        pred0_Q13 += delta0_Q13;
        pred1_Q13 += delta1_Q13;
        let sum = (x1[n] as i32 + x1[n + 2] as i32 + ((x1[n + 1] as i32) << 1)) << 9; /* Q11 */
        let sum = silk_smlawb((x2[n + 1] as i32) << 8, sum, pred0_Q13); /* Q8  */
        let sum = silk_smlawb(sum, (x1[n + 1] as i32) << 11, pred1_Q13); /* Q8  */
        x2[n + 1] = silk_sat16(silk_rshift_round(sum, 8)) as i16;
    }

    let pred0_Q13 = pred_Q13[0];
    let pred1_Q13 = pred_Q13[1];

    for n in STEREO_INTERP_LEN_MS * fs_kHz..frame_length {
        let sum = (x1[n] as i32 + x1[n + 2] as i32 + ((x1[n + 1] as i32) << 1)) << 9; /* Q11 */
        let sum = silk_smlawb((x2[n + 1] as i32) << 8, sum, pred0_Q13); /* Q8  */
        let sum = silk_smlawb(sum, (x1[n + 1] as i32) << 11, pred1_Q13); /* Q8  */
        x2[n + 1] = silk_sat16(silk_rshift_round(sum, 8)) as i16;
    }
    state.pred_prev_Q13[0] = pred_Q13[0] as i16;
    state.pred_prev_Q13[1] = pred_Q13[1] as i16;

    /* Convert to left/right signals */
    for n in 0..frame_length {
        let sum = x1[n + 1] as i32 + x2[n + 1] as i32;
        let diff = x1[n + 1] as i32 - x2[n + 1] as i32;

        x1[n + 1] = silk_sat16(sum) as i16;
        x2[n + 1] = silk_sat16(diff) as i16;
    }
}
