//! Left/right to mid/side conversion.
//!
//! Upstream C: `silk/stereo_LR_to_MS.c`

pub mod typedef_h {
    pub const silk_int16_MIN: i32 = i16::MIN as i32;
    pub const silk_int16_MAX: i32 = i16::MAX as i32;
}
pub use self::typedef_h::{silk_int16_MAX, silk_int16_MIN};
use crate::silk::define::{LA_SHAPE_MS, STEREO_INTERP_LEN_MS};
use crate::silk::stereo_find_predictor::silk_stereo_find_predictor;
use crate::silk::stereo_quant_pred::silk_stereo_quant_pred;
use crate::silk::structs::stereo_enc_state;
use crate::silk::Inlines::silk_DIV32_varQ;
use crate::silk::SigProc_FIX::silk_max_int;

///
/// `x1` and `x2` are slices starting 2 samples before the frame data
/// (i.e., they include 2 history samples at index 0..1, frame at 2..frame_length+1).
/// Total length must be `frame_length + 2`.
/// Upstream C: silk/stereo_LR_to_MS.c:silk_stereo_LR_to_MS
pub fn silk_stereo_LR_to_MS(
    state: &mut stereo_enc_state,
    x1: &mut [i16],
    x2: &mut [i16],
    frame_idx: usize,
    mid_side_rates_bps: &mut [i32],
    mut total_rate_bps: i32,
    prev_speech_act_Q8: i32,
    toMono: i32,
    fs_kHz: i32,
    frame_length: i32,
) {
    let mut n: i32;

    let mut sum: i32;
    let mut diff: i32;
    let mut smooth_coef_Q16: i32;
    let mut pred_Q13: [i32; 2] = [0; 2];
    let mut pred0_Q13: i32;
    let mut pred1_Q13: i32;
    let mut LP_ratio_Q14: i32 = 0;
    let mut HP_ratio_Q14: i32 = 0;
    let mut frac_Q16: i32;

    let mut width_Q14: i32;
    let mut w_Q24: i32;

    // x1[0..frame_length+2] — mid is written in-place into x1
    // side is a separate buffer
    let vla = (frame_length + 2) as usize;
    let mut side: Vec<i16> = ::std::vec::from_elem(0, vla);

    // Compute mid and side from L/R
    // mid[n] = round((x1[n] + x2[n]) / 2), stored in x1[n]
    // side[n] = sat16(round((x1[n] - x2[n]) / 2))
    n = 0;
    while n < frame_length + 2 {
        sum = x1[n as usize] as i32 + x2[n as usize] as i32;
        diff = x1[n as usize] as i32 - x2[n as usize] as i32;
        x1[n as usize] = ((sum >> 1) + (sum & 1)) as i16;
        side[n as usize] = ((diff >> 1) + (diff & 1)).clamp(silk_int16_MIN, silk_int16_MAX) as i16;
        n += 1;
    }

    // Copy state history into the beginning of mid/side (first 2 samples),
    // then save the last 2 samples of mid/side into state for next frame.
    // mid = x1, so we work on x1 directly
    x1[..2].copy_from_slice(&state.sMid);
    side[..2].copy_from_slice(&state.sSide);
    state
        .sMid
        .copy_from_slice(&x1[frame_length as usize..frame_length as usize + 2]);
    state
        .sSide
        .copy_from_slice(&side[frame_length as usize..frame_length as usize + 2]);

    // LP/HP decomposition of mid and side
    let vla_0 = frame_length as usize;
    let mut LP_mid: Vec<i16> = ::std::vec::from_elem(0, vla_0);
    let vla_1 = frame_length as usize;
    let mut HP_mid: Vec<i16> = ::std::vec::from_elem(0, vla_1);
    n = 0;
    while n < frame_length {
        // mid = x1
        sum = (((x1[n as usize] as i32
            + x1[(n + 2) as usize] as i32
            + ((x1[(n + 1) as usize] as u32) << 1) as i32)
            >> (2 - 1))
            + 1)
            >> 1;
        LP_mid[n as usize] = sum as i16;
        HP_mid[n as usize] = (x1[(n + 1) as usize] as i32 - sum) as i16;
        n += 1;
    }
    let vla_2 = frame_length as usize;
    let mut LP_side: Vec<i16> = ::std::vec::from_elem(0, vla_2);
    let vla_3 = frame_length as usize;
    let mut HP_side: Vec<i16> = ::std::vec::from_elem(0, vla_3);
    n = 0;
    while n < frame_length {
        sum = (((side[n as usize] as i32
            + side[(n + 2) as usize] as i32
            + ((side[(n + 1) as usize] as u32) << 1) as i32)
            >> (2 - 1))
            + 1)
            >> 1;
        LP_side[n as usize] = sum as i16;
        HP_side[n as usize] = (side[(n + 1) as usize] as i32 - sum) as i16;
        n += 1;
    }
    let is10msFrame: i32 = (frame_length == 10 * fs_kHz) as i32;
    smooth_coef_Q16 = if is10msFrame != 0 {
        (0.01f64 / 2_f64 * ((1) << 16) as f64 + 0.5f64) as i32
    } else {
        (0.01f64 * ((1) << 16) as f64 + 0.5f64) as i32
    };
    smooth_coef_Q16 = (((prev_speech_act_Q8 as i16 as i32 * prev_speech_act_Q8 as i16 as i32)
        as i64
        * smooth_coef_Q16 as i16 as i64)
        >> 16) as i32;
    pred_Q13[0_usize] = silk_stereo_find_predictor(
        &mut LP_ratio_Q14,
        &LP_mid,
        &LP_side,
        &mut (&mut state.mid_side_amp_Q0)[0..2],
        frame_length,
        smooth_coef_Q16,
    );
    pred_Q13[1_usize] = silk_stereo_find_predictor(
        &mut HP_ratio_Q14,
        &HP_mid,
        &HP_side,
        &mut (&mut state.mid_side_amp_Q0)[2..4],
        frame_length,
        smooth_coef_Q16,
    );
    frac_Q16 = HP_ratio_Q14 + LP_ratio_Q14 as i16 as i32 * 3;
    frac_Q16 = if frac_Q16 < (((1) << 16) as f64 + 0.5f64) as i32 {
        frac_Q16
    } else {
        (((1) << 16) as f64 + 0.5f64) as i32
    };
    total_rate_bps -= if is10msFrame != 0 { 1200 } else { 600 };
    if total_rate_bps < 1 {
        total_rate_bps = 1;
    }
    let min_mid_rate_bps: i32 = 2000 + fs_kHz as i16 as i32 * 600;
    let frac_3_Q16: i32 = 3 * frac_Q16;
    mid_side_rates_bps[0] = silk_DIV32_varQ(
        total_rate_bps,
        (((8 + 5) as i64 * ((1) << 16)) as f64 + 0.5f64) as i32 + frac_3_Q16,
        16 + 3,
    );
    if mid_side_rates_bps[0] < min_mid_rate_bps {
        mid_side_rates_bps[0] = min_mid_rate_bps;
        mid_side_rates_bps[1] = total_rate_bps - mid_side_rates_bps[0];
        width_Q14 = silk_DIV32_varQ(
            ((mid_side_rates_bps[1] as u32) << 1) as i32 - min_mid_rate_bps,
            ((((((1) << 16) as f64 + 0.5f64) as i32 + frac_3_Q16) as i64
                * min_mid_rate_bps as i16 as i64)
                >> 16) as i32,
            14 + 2,
        );
        width_Q14 = if 0 > (((1) << 14) as f64 + 0.5f64) as i32 {
            if width_Q14 > 0 {
                0
            } else if width_Q14 < (((1) << 14) as f64 + 0.5f64) as i32 {
                (((1) << 14) as f64 + 0.5f64) as i32
            } else {
                width_Q14
            }
        } else if width_Q14 > (((1) << 14) as f64 + 0.5f64) as i32 {
            (((1) << 14) as f64 + 0.5f64) as i32
        } else if width_Q14 < 0 {
            0
        } else {
            width_Q14
        };
    } else {
        mid_side_rates_bps[1] = total_rate_bps - mid_side_rates_bps[0];
        width_Q14 = (((1) << 14) as f64 + 0.5f64) as i32;
    }
    state.smth_width_Q14 = (state.smth_width_Q14 as i64
        + (((width_Q14 - state.smth_width_Q14 as i32) as i64 * smooth_coef_Q16 as i16 as i64)
            >> 16)) as i32 as i16;
    state.mid_only_flags[frame_idx] = 0;
    if toMono != 0 {
        width_Q14 = 0;
        pred_Q13[0_usize] = 0;
        pred_Q13[1_usize] = 0;
        silk_stereo_quant_pred(&mut pred_Q13, &mut state.predIx[frame_idx]);
    } else if state.width_prev_Q14 as i32 == 0
        && (8 * total_rate_bps < 13 * min_mid_rate_bps
            || (((frac_Q16 as i64 * state.smth_width_Q14 as i64) >> 16) as i32)
                < (0.05f64 * ((1) << 14) as f64 + 0.5f64) as i32)
    {
        pred_Q13[0_usize] = (state.smth_width_Q14 as i32 * pred_Q13[0_usize] as i16 as i32) >> 14;
        pred_Q13[1_usize] = (state.smth_width_Q14 as i32 * pred_Q13[1_usize] as i16 as i32) >> 14;
        silk_stereo_quant_pred(&mut pred_Q13, &mut state.predIx[frame_idx]);
        width_Q14 = 0;
        pred_Q13[0_usize] = 0;
        pred_Q13[1_usize] = 0;
        mid_side_rates_bps[0] = total_rate_bps;
        mid_side_rates_bps[1] = 0;
        state.mid_only_flags[frame_idx] = 1;
    } else if state.width_prev_Q14 as i32 != 0
        && (8 * total_rate_bps < 11 * min_mid_rate_bps
            || (((frac_Q16 as i64 * state.smth_width_Q14 as i64) >> 16) as i32)
                < (0.02f64 * ((1) << 14) as f64 + 0.5f64) as i32)
    {
        pred_Q13[0_usize] = (state.smth_width_Q14 as i32 * pred_Q13[0_usize] as i16 as i32) >> 14;
        pred_Q13[1_usize] = (state.smth_width_Q14 as i32 * pred_Q13[1_usize] as i16 as i32) >> 14;
        silk_stereo_quant_pred(&mut pred_Q13, &mut state.predIx[frame_idx]);
        width_Q14 = 0;
        pred_Q13[0_usize] = 0;
        pred_Q13[1_usize] = 0;
    } else if state.smth_width_Q14 as i32 > (0.95f64 * ((1) << 14) as f64 + 0.5f64) as i32 {
        silk_stereo_quant_pred(&mut pred_Q13, &mut state.predIx[frame_idx]);
        width_Q14 = (((1) << 14) as f64 + 0.5f64) as i32;
    } else {
        pred_Q13[0_usize] = (state.smth_width_Q14 as i32 * pred_Q13[0_usize] as i16 as i32) >> 14;
        pred_Q13[1_usize] = (state.smth_width_Q14 as i32 * pred_Q13[1_usize] as i16 as i32) >> 14;
        silk_stereo_quant_pred(&mut pred_Q13, &mut state.predIx[frame_idx]);
        width_Q14 = state.smth_width_Q14 as i32;
    }
    if state.mid_only_flags[frame_idx] as i32 == 1 {
        state.silent_side_len = (state.silent_side_len as i32
            + (frame_length - STEREO_INTERP_LEN_MS as i32 * fs_kHz))
            as i16;
        if (state.silent_side_len as i32) < LA_SHAPE_MS * fs_kHz {
            state.mid_only_flags[frame_idx] = 0;
        } else {
            state.silent_side_len = 10000;
        }
    } else {
        state.silent_side_len = 0;
    }
    if state.mid_only_flags[frame_idx] as i32 == 0 && mid_side_rates_bps[1] < 1 {
        mid_side_rates_bps[1] = 1;
        mid_side_rates_bps[0] = silk_max_int(1, total_rate_bps - mid_side_rates_bps[1]);
    }
    pred0_Q13 = -(state.pred_prev_Q13[0_usize] as i32);
    pred1_Q13 = -(state.pred_prev_Q13[1_usize] as i32);
    w_Q24 = ((state.width_prev_Q14 as u32) << 10) as i32;
    let denom_Q16: i32 = ((1) << 16) / (8 * fs_kHz);
    let delta0_Q13: i32 = -(((((pred_Q13[0_usize] - state.pred_prev_Q13[0_usize] as i32) as i16
        as i32
        * denom_Q16 as i16 as i32)
        >> (16 - 1))
        + 1)
        >> 1);
    let delta1_Q13: i32 = -(((((pred_Q13[1_usize] - state.pred_prev_Q13[1_usize] as i32) as i16
        as i32
        * denom_Q16 as i16 as i32)
        >> (16 - 1))
        + 1)
        >> 1);
    let deltaw_Q24: i32 = (((((width_Q14 - state.width_prev_Q14 as i32) as i64
        * denom_Q16 as i16 as i64)
        >> 16) as i32 as u32)
        << 10) as i32;

    // Interpolation loop: reconstruct x2 (side channel) from mid and side
    // mid = x1 (in-place), x2 is overwritten with the reconstructed side signal
    n = 0;
    while n < STEREO_INTERP_LEN_MS as i32 * fs_kHz {
        pred0_Q13 += delta0_Q13;
        pred1_Q13 += delta1_Q13;
        w_Q24 += deltaw_Q24;
        sum = (((x1[n as usize] as i32
            + x1[(n + 2) as usize] as i32
            + ((x1[(n + 1) as usize] as u32) << 1) as i32) as u32)
            << 9) as i32;
        sum = (((w_Q24 as i64 * side[(n + 1) as usize] as i64) >> 16) as i32 as i64
            + ((sum as i64 * pred0_Q13 as i16 as i64) >> 16)) as i32;
        sum = (sum as i64
            + ((((x1[(n + 1) as usize] as i32 as u32) << 11) as i32 as i64
                * pred1_Q13 as i16 as i64)
                >> 16)) as i32;
        // x2 output is at offset n-1 relative to mid; but since our slices start 2 before frame,
        // x2[n-1] in the original code with mid=x1-2 corresponds to x2[n-1] here
        // Original: *x2.offset((n-1) as isize) where x2 starts at frame data
        // But our x2 slice also starts 2 before frame data, same as x1.
        // In the C code, x2 was pointing at frame start (offset 2 in inputBuf).
        // mid was x1-2 = pointing 2 before frame start.
        // The output was *x2.offset((n-1) as isize) = x2[n-1] relative to frame start.
        // But in our caller, x2 also starts 2 before frame start (same offset as x1).
        // So we need x2[n-1+2] = x2[n+1] to match.
        // Wait, let me re-examine...
        // In the C caller (enc_API.rs): x1 = &inputBuf[2], x2 = &inputBuf[2]
        // In C original: mid = x1 - 2 = &inputBuf[0]
        // Output: x2[n-1] where n goes from 0..interp_len
        // But our x1 starts at inputBuf[0] (caller passes &inputBuf[0..frame_length+2])
        // so mid = x1 directly (no offset needed)
        // x2 also starts at inputBuf[0], so output x2[n-1]... but n starts at 0, giving x2[-1]?
        //
        // Actually in the original, the caller passes &inputBuf[2] for both x1 and x2.
        // mid = x1 - 2 = &inputBuf[0], and mid[n] for n=0..frame_length+2 covers inputBuf[0..frame_length+2]
        // x2[n-1] for n=0 gives x2[-1] = inputBuf[1], which is the last history sample.
        //
        // For our safe version, we'll change the caller to pass &inputBuf[0..frame_length+2]
        // for both x1 and x2, and adjust the x2 write index accordingly.
        // x2 output index: in C, x2.offset(n-1) = inputBuf[2 + n - 1] = inputBuf[n+1]
        // In our version with x2 starting at inputBuf[0]: x2[n+1]
        x2[(n + 1) as usize] =
            (((sum >> (8 - 1)) + 1) >> 1).clamp(silk_int16_MIN, silk_int16_MAX) as i16;
        n += 1;
    }
    pred0_Q13 = -pred_Q13[0_usize];
    pred1_Q13 = -pred_Q13[1_usize];
    w_Q24 = ((width_Q14 as u32) << 10) as i32;
    n = STEREO_INTERP_LEN_MS as i32 * fs_kHz;
    while n < frame_length {
        sum = (((x1[n as usize] as i32
            + x1[(n + 2) as usize] as i32
            + ((x1[(n + 1) as usize] as u32) << 1) as i32) as u32)
            << 9) as i32;
        sum = (((w_Q24 as i64 * side[(n + 1) as usize] as i64) >> 16) as i32 as i64
            + ((sum as i64 * pred0_Q13 as i16 as i64) >> 16)) as i32;
        sum = (sum as i64
            + ((((x1[(n + 1) as usize] as i32 as u32) << 11) as i32 as i64
                * pred1_Q13 as i16 as i64)
                >> 16)) as i32;
        x2[(n + 1) as usize] =
            (((sum >> (8 - 1)) + 1) >> 1).clamp(silk_int16_MIN, silk_int16_MAX) as i16;
        n += 1;
    }
    state.pred_prev_Q13[0_usize] = pred_Q13[0_usize] as i16;
    state.pred_prev_Q13[1_usize] = pred_Q13[1_usize] as i16;
    state.width_prev_Q14 = width_Q14 as i16;
}
