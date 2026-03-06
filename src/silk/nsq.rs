//! Noise shaping quantizer.
//!
//! Upstream c: `silk/nsq.c`

use crate::silk::typedefs::{SILK_INT16_MAX, SILK_INT16_MIN};

///
/// Short-term prediction using LPC coefficients. `buf32` is indexed as
/// `buf32[pos], buf32[pos-1], ..., buf32[pos-order+1]` and `coef16` has
/// `order` entries. Here we take `buf32` as a slice ending at `pos+1`
/// (_i.e. the element at `buf32[buf32.len()-1]` is `buf32[pos]`).
/// Upstream c: silk/nsq.h:silk_noise_shape_quantizer_short_prediction_c
#[inline(always)]
pub fn silk_noise_shape_quantizer_short_prediction_c(
    buf32: &[i32],
    coef16: &[i16],
    order: i32,
) -> i32 {
    // buf32 is indexed backwards from the end: buf32[len-1] = pos, buf32[len-2] = pos-1, etc.
    // Pre-slice to the last 10 elements to hoist bounds checks.
    let b = buf32.len();
    let buf = &buf32[b - 10..];
    let coef = &coef16[..10];
    let mut out: i32 = order >> 1;
    out = (out as i64 + ((buf[9] as i64 * coef[0] as i64) >> 16)) as i32;
    out = (out as i64 + ((buf[8] as i64 * coef[1] as i64) >> 16)) as i32;
    out = (out as i64 + ((buf[7] as i64 * coef[2] as i64) >> 16)) as i32;
    out = (out as i64 + ((buf[6] as i64 * coef[3] as i64) >> 16)) as i32;
    out = (out as i64 + ((buf[5] as i64 * coef[4] as i64) >> 16)) as i32;
    out = (out as i64 + ((buf[4] as i64 * coef[5] as i64) >> 16)) as i32;
    out = (out as i64 + ((buf[3] as i64 * coef[6] as i64) >> 16)) as i32;
    out = (out as i64 + ((buf[2] as i64 * coef[7] as i64) >> 16)) as i32;
    out = (out as i64 + ((buf[1] as i64 * coef[8] as i64) >> 16)) as i32;
    out = (out as i64 + ((buf[0] as i64 * coef[9] as i64) >> 16)) as i32;
    if order == 16 {
        let buf16 = &buf32[b - 16..];
        let coef16 = &coef16[10..16];
        out = (out as i64 + ((buf16[5] as i64 * coef16[0] as i64) >> 16)) as i32;
        out = (out as i64 + ((buf16[4] as i64 * coef16[1] as i64) >> 16)) as i32;
        out = (out as i64 + ((buf16[3] as i64 * coef16[2] as i64) >> 16)) as i32;
        out = (out as i64 + ((buf16[2] as i64 * coef16[3] as i64) >> 16)) as i32;
        out = (out as i64 + ((buf16[1] as i64 * coef16[4] as i64) >> 16)) as i32;
        out = (out as i64 + ((buf16[0] as i64 * coef16[5] as i64) >> 16)) as i32;
    }
    out
}

///
/// Noise shape feedback loop. `data0` is the new input value,
/// `data1` is the shift register (length `order`), `coef` has `order` entries.
/// Shifts new value into data1 while computing the weighted sum.
/// Upstream c: silk/nsq.h:silk_NSQ_noise_shape_feedback_loop_c
#[inline]
pub fn silk_nsq_noise_shape_feedback_loop_c(
    data0: i32,
    data1: &mut [i32],
    coef: &[i16],
    order: i32,
) -> i32 {
    let n = order as usize;
    let data1 = &mut data1[..n];
    let coef = &coef[..n];
    let mut tmp2 = data0;
    let mut tmp1 = data1[0];
    data1[0] = tmp2;
    let mut out: i32 = order >> 1;
    out = (out as i64 + ((tmp2 as i64 * coef[0] as i64) >> 16)) as i32;
    let mut j = 2usize;
    while j < n {
        tmp2 = data1[j - 1];
        data1[j - 1] = tmp1;
        out = (out as i64 + ((tmp1 as i64 * coef[j - 1] as i64) >> 16)) as i32;
        tmp1 = data1[j];
        data1[j] = tmp2;
        out = (out as i64 + ((tmp2 as i64 * coef[j] as i64) >> 16)) as i32;
        j += 2;
    }
    data1[n - 1] = tmp1;
    out = (out as i64 + ((tmp1 as i64 * coef[n - 1] as i64) >> 16)) as i32;
    out = ((out as u32) << 1) as i32;
    out
}

/// Dispatch wrapper for short prediction — routes to SIMD when available.
#[cfg(feature = "simd")]
#[inline(always)]
pub fn silk_noise_shape_quantizer_short_prediction(
    buf32: &[i32],
    coef16: &[i16],
    order: i32,
    arch: Arch,
) -> i32 {
    super::simd::silk_noise_shape_quantizer_short_prediction(buf32, coef16, order, arch)
}

/// Dispatch wrapper for short prediction (scalar-only build).
#[cfg(not(feature = "simd"))]
#[inline]
pub fn silk_noise_shape_quantizer_short_prediction(
    buf32: &[i32],
    coef16: &[i16],
    order: i32,
    _arch: Arch,
) -> i32 {
    silk_noise_shape_quantizer_short_prediction_c(buf32, coef16, order)
}

/// Dispatch wrapper for noise shape feedback loop — routes to SIMD when available.
#[cfg(feature = "simd")]
#[inline(always)]
pub fn silk_nsq_noise_shape_feedback_loop(
    data0: i32,
    data1: &mut [i32],
    coef: &[i16],
    order: i32,
    arch: Arch,
) -> i32 {
    super::simd::silk_nsq_noise_shape_feedback_loop(data0, data1, coef, order, arch)
}

/// Dispatch wrapper for noise shape feedback loop (scalar-only build).
#[cfg(not(feature = "simd"))]
#[inline]
pub fn silk_nsq_noise_shape_feedback_loop(
    data0: i32,
    data1: &mut [i32],
    coef: &[i16],
    order: i32,
    _arch: Arch,
) -> i32 {
    silk_nsq_noise_shape_feedback_loop_c(data0, data1, coef, order)
}

use crate::arch::Arch;
#[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
use crate::silk::define::QUANT_LEVEL_ADJUST_Q10;
use crate::silk::define::{
    HARM_SHAPE_FIR_TAPS, LTP_ORDER, MAX_LPC_ORDER, MAX_SHAPE_LPC_ORDER, NSQ_LPC_BUF_LENGTH,
    TYPE_VOICED,
};
use crate::silk::inlines::{silk_div32_varq, silk_inverse32_varq};
use crate::silk::lpc_analysis_filter::silk_lpc_analysis_filter;
use crate::silk::sigproc_fix::silk_rand;
use crate::silk::structs::{silk_nsq_state, NsqConfig, SideInfoIndices};
use crate::silk::tables_other::SILK_QUANTIZATION_OFFSETS_Q10;

/// Dispatch wrapper for nsq, matching upstream `silk_nsq` RTCD surface.
#[cfg(feature = "simd")]
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn silk_nsq(
    ps_enc_c: &NsqConfig,
    nsq: &mut silk_nsq_state,
    ps_indices: &mut SideInfoIndices,
    x16: &[i16],
    pulses: &mut [i8],
    pred_coef_q12: &[i16],
    ltpcoef_q14: &[i16],
    ar_q13: &[i16],
    harm_shape_gain_q14: &[i32],
    tilt_q14: &[i32],
    lf_shp_q14: &[i32],
    gains_q16: &[i32],
    pitch_l: &[i32],
    lambda_q10: i32,
    ltp_scale_q14: i32,
) {
    super::simd::silk_nsq(
        ps_enc_c,
        nsq,
        ps_indices,
        x16,
        pulses,
        pred_coef_q12,
        ltpcoef_q14,
        ar_q13,
        harm_shape_gain_q14,
        tilt_q14,
        lf_shp_q14,
        gains_q16,
        pitch_l,
        lambda_q10,
        ltp_scale_q14,
    );
}

/// Scalar-only build wrapper for nsq.
#[cfg(not(feature = "simd"))]
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn silk_nsq(
    ps_enc_c: &NsqConfig,
    nsq: &mut silk_nsq_state,
    ps_indices: &mut SideInfoIndices,
    x16: &[i16],
    pulses: &mut [i8],
    pred_coef_q12: &[i16],
    ltpcoef_q14: &[i16],
    ar_q13: &[i16],
    harm_shape_gain_q14: &[i32],
    tilt_q14: &[i32],
    lf_shp_q14: &[i32],
    gains_q16: &[i32],
    pitch_l: &[i32],
    lambda_q10: i32,
    ltp_scale_q14: i32,
) {
    silk_nsq_c(
        ps_enc_c,
        nsq,
        ps_indices,
        x16,
        pulses,
        pred_coef_q12,
        ltpcoef_q14,
        ar_q13,
        harm_shape_gain_q14,
        tilt_q14,
        lf_shp_q14,
        gains_q16,
        pitch_l,
        lambda_q10,
        ltp_scale_q14,
    );
}

/// Upstream c: silk/nsq.c:silk_NSQ_c
#[allow(clippy::too_many_arguments)]
pub fn silk_nsq_c(
    ps_enc_c: &NsqConfig,
    nsq: &mut silk_nsq_state,
    ps_indices: &mut SideInfoIndices,
    x16: &[i16],
    pulses: &mut [i8],
    pred_coef_q12: &[i16],
    ltpcoef_q14: &[i16],
    ar_q13: &[i16],
    harm_shape_gain_q14: &[i32],
    tilt_q14: &[i32],
    lf_shp_q14: &[i32],
    gains_q16: &[i32],
    pitch_l: &[i32],
    lambda_q10: i32,
    ltp_scale_q14: i32,
) {
    let mut lag: i32;
    let mut start_idx: i32;
    let mut harm_shape_firpacked_q14: i32;

    nsq.rand_seed = ps_indices.seed as i32;
    lag = nsq.lag_prev;
    let offset_q10 = SILK_QUANTIZATION_OFFSETS_Q10[(ps_indices.signal_type as i32 >> 1) as usize]
        [ps_indices.quant_offset_type as usize] as i32;

    // Precompute quantization lookup table for SSE4.1 path (x86 only)
    #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
    let (use_simd_quantizer, table) = {
        let use_it = super::simd::use_nsq_sse4_1(ps_enc_c.arch)
            && ps_enc_c.shaping_lpcorder == 10
            && ps_enc_c.predict_lpcorder == 16;
        let table = if use_it {
            build_quantization_table(offset_q10, lambda_q10)
        } else {
            [[0i32; 4]; 64]
        };
        (use_it, table)
    };

    let lsf_interpolation_flag: i32 = if ps_indices.nlsfinterp_coef_q2 as i32 == 4 {
        0
    } else {
        1
    };
    let ltp_mem_len = ps_enc_c.ltp_mem_length;
    let frame_len = ps_enc_c.frame_length;
    let subfr_len = ps_enc_c.subfr_length;

    // ltp_mem_len + frame_len max: 320 + 320 = 640
    const MAX_LTP_FRAME: usize = 640;
    debug_assert!(ltp_mem_len + frame_len <= MAX_LTP_FRAME);
    let mut s_ltp_q15 = [0i32; MAX_LTP_FRAME];
    let mut s_ltp = [0i16; MAX_LTP_FRAME];
    // subfr_len max: MAX_SUB_FRAME_LENGTH = 80
    const MAX_SUBFR: usize = 80;
    debug_assert!(subfr_len <= MAX_SUBFR);
    let mut x_sc_q10 = [0i32; MAX_SUBFR];

    nsq.s_ltp_shp_buf_idx = ltp_mem_len as i32;
    nsq.s_ltp_buf_idx = ltp_mem_len as i32;
    let mut pxq_off: usize = ltp_mem_len;
    let mut x16_off: usize = 0;
    let mut pulses_off: usize = 0;

    for k in 0..ps_enc_c.nb_subfr as i32 {
        let a_q12_off = (((k >> 1) | (1 - lsf_interpolation_flag)) * MAX_LPC_ORDER as i32) as usize;
        let a_q12 = &pred_coef_q12[a_q12_off..a_q12_off + ps_enc_c.predict_lpcorder as usize];
        let b_q14_off = (k * LTP_ORDER as i32) as usize;
        let b_q14 = &ltpcoef_q14[b_q14_off..b_q14_off + LTP_ORDER];
        let ar_shp_off = (k * MAX_SHAPE_LPC_ORDER) as usize;
        let ar_shp_q13 = &ar_q13[ar_shp_off..ar_shp_off + ps_enc_c.shaping_lpcorder as usize];

        harm_shape_firpacked_q14 = harm_shape_gain_q14[k as usize] >> 2;
        harm_shape_firpacked_q14 |= (((harm_shape_gain_q14[k as usize] >> 1) as u32) << 16) as i32;

        nsq.rewhite_flag = 0;
        if ps_indices.signal_type as i32 == TYPE_VOICED {
            lag = pitch_l[k as usize];
            if k & (3 - ((lsf_interpolation_flag as u32) << 1) as i32) == 0 {
                start_idx =
                    ltp_mem_len as i32 - lag - ps_enc_c.predict_lpcorder - LTP_ORDER as i32 / 2;
                debug_assert!(start_idx > 0);
                silk_lpc_analysis_filter(
                    &mut s_ltp[start_idx as usize..ltp_mem_len],
                    &nsq.xq[(start_idx + k * subfr_len as i32) as usize..]
                        [..(ltp_mem_len - start_idx as usize)],
                    a_q12,
                );
                nsq.rewhite_flag = 1;
                nsq.s_ltp_buf_idx = ltp_mem_len as i32;
            }
        }
        silk_nsq_scale_states(
            ps_enc_c,
            nsq,
            &x16[x16_off..x16_off + subfr_len],
            &mut x_sc_q10,
            &s_ltp,
            &mut s_ltp_q15,
            k,
            ltp_scale_q14,
            gains_q16,
            pitch_l,
            ps_indices.signal_type as i32,
        );
        #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
        {
            if use_simd_quantizer {
                super::simd::silk_noise_shape_quantizer_10_16_sse4_1(
                    nsq,
                    ps_indices.signal_type as i32,
                    &x_sc_q10,
                    &mut pulses[pulses_off..pulses_off + subfr_len],
                    pxq_off,
                    &mut s_ltp_q15,
                    a_q12,
                    b_q14,
                    ar_shp_q13,
                    lag,
                    harm_shape_firpacked_q14,
                    tilt_q14[k as usize],
                    lf_shp_q14[k as usize],
                    gains_q16[k as usize],
                    lambda_q10,
                    offset_q10,
                    subfr_len as i32,
                    &table,
                );
            } else {
                silk_noise_shape_quantizer(
                    nsq,
                    ps_indices.signal_type as i32,
                    &x_sc_q10,
                    &mut pulses[pulses_off..pulses_off + subfr_len],
                    pxq_off,
                    &mut s_ltp_q15,
                    a_q12,
                    b_q14,
                    ar_shp_q13,
                    lag,
                    harm_shape_firpacked_q14,
                    tilt_q14[k as usize],
                    lf_shp_q14[k as usize],
                    gains_q16[k as usize],
                    lambda_q10,
                    offset_q10,
                    subfr_len as i32,
                    ps_enc_c.shaping_lpcorder,
                    ps_enc_c.predict_lpcorder,
                    ps_enc_c.arch,
                );
            }
        }
        #[cfg(not(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64"))))]
        silk_noise_shape_quantizer(
            nsq,
            ps_indices.signal_type as i32,
            &x_sc_q10,
            &mut pulses[pulses_off..pulses_off + subfr_len],
            pxq_off,
            &mut s_ltp_q15,
            a_q12,
            b_q14,
            ar_shp_q13,
            lag,
            harm_shape_firpacked_q14,
            tilt_q14[k as usize],
            lf_shp_q14[k as usize],
            gains_q16[k as usize],
            lambda_q10,
            offset_q10,
            subfr_len as i32,
            ps_enc_c.shaping_lpcorder,
            ps_enc_c.predict_lpcorder,
            ps_enc_c.arch,
        );
        x16_off += subfr_len;
        pulses_off += subfr_len;
        pxq_off += subfr_len;
    }
    nsq.lag_prev = pitch_l[ps_enc_c.nb_subfr - 1];
    nsq.xq.copy_within(frame_len..frame_len + ltp_mem_len, 0);
    nsq.s_ltp_shp_q14
        .copy_within(frame_len..frame_len + ltp_mem_len, 0);
}

///
/// Core noise-shape quantizer inner loop. Processes one subframe of samples.
/// `xq_off` is the offset into `nsq.xq` where output samples are written.
/// Upstream c: silk/nsq.c:silk_noise_shape_quantizer
#[inline]
#[allow(clippy::too_many_arguments)]
fn silk_noise_shape_quantizer(
    nsq: &mut silk_nsq_state,
    signal_type: i32,
    x_sc_q10: &[i32],
    pulses: &mut [i8],
    xq_off: usize,
    s_ltp_q15: &mut [i32],
    a_q12: &[i16],
    b_q14: &[i16],
    ar_shp_q13: &[i16],
    lag: i32,
    harm_shape_firpacked_q14: i32,
    tilt_q14: i32,
    lf_shp_q14: i32,
    gain_q16: i32,
    lambda_q10: i32,
    offset_q10: i32,
    length: i32,
    shaping_lpcorder: i32,
    predict_lpcorder: i32,
    _arch: Arch,
) {
    let mut ltp_pred_q13: i32;
    let mut lpc_pred_q10: i32;
    let mut n_ar_q12: i32;
    let mut n_ltp_q13: i32;
    let mut n_lf_q12: i32;
    let mut r_q10: i32;
    let mut rr_q10: i32;
    let mut q1_q0: i32;
    let mut q1_q10: i32;
    let mut q2_q10: i32;
    let mut rd1_q20: i32;
    let mut rd2_q20: i32;
    let mut exc_q14: i32;
    let mut lpc_exc_q14: i32;
    let mut xq_q14: i32;
    let mut tmp1: i32;
    let mut tmp2: i32;
    let mut s_lf_ar_shp_q14: i32;

    let gain_q10: i32 = gain_q16 >> 6;
    let length = length as usize;

    // shp_lag_ptr starts at s_ltp_shp_buf_idx - lag + HARM_SHAPE_FIR_TAPS/2
    // and advances by 1 each iteration
    let mut shp_lag_idx = (nsq.s_ltp_shp_buf_idx - lag + HARM_SHAPE_FIR_TAPS / 2) as usize;

    // pred_lag_ptr starts at s_ltp_buf_idx - lag + LTP_ORDER/2
    // and advances by 1 each iteration
    let mut pred_lag_idx = (nsq.s_ltp_buf_idx - lag + LTP_ORDER as i32 / 2) as usize;

    // psLPC_Q14 starts at s_lpc_q14[NSQ_LPC_BUF_LENGTH - 1] and advances.
    // In the original c code, psLPC_Q14 points into the middle of s_lpc_q14
    // and is indexed backwards for prediction and forward for writing.
    // We use an index `lpc_idx` that tracks the "current" position.
    let mut lpc_idx: usize = NSQ_LPC_BUF_LENGTH - 1;

    // Pre-slice to hoist bounds checks out of the hot loop.
    let x_sc_q10 = &x_sc_q10[..length];
    let pulses = &mut pulses[..length];

    for _i in 0..length {
        nsq.rand_seed = silk_rand(nsq.rand_seed);

        // LPC prediction: pass slice ending at current position
        lpc_pred_q10 = silk_noise_shape_quantizer_short_prediction(
            &nsq.s_lpc_q14[..lpc_idx + 1],
            a_q12,
            predict_lpcorder,
            _arch,
        );

        // LTP prediction
        if signal_type == TYPE_VOICED {
            ltp_pred_q13 = 2;
            ltp_pred_q13 = (ltp_pred_q13 as i64
                + ((s_ltp_q15[pred_lag_idx] as i64 * b_q14[0] as i64) >> 16))
                as i32;
            ltp_pred_q13 = (ltp_pred_q13 as i64
                + ((s_ltp_q15[pred_lag_idx - 1] as i64 * b_q14[1] as i64) >> 16))
                as i32;
            ltp_pred_q13 = (ltp_pred_q13 as i64
                + ((s_ltp_q15[pred_lag_idx - 2] as i64 * b_q14[2] as i64) >> 16))
                as i32;
            ltp_pred_q13 = (ltp_pred_q13 as i64
                + ((s_ltp_q15[pred_lag_idx - 3] as i64 * b_q14[3] as i64) >> 16))
                as i32;
            ltp_pred_q13 = (ltp_pred_q13 as i64
                + ((s_ltp_q15[pred_lag_idx - 4] as i64 * b_q14[4] as i64) >> 16))
                as i32;
            pred_lag_idx += 1;
        } else {
            ltp_pred_q13 = 0;
        }

        // Noise shape feedback
        debug_assert!(shaping_lpcorder & 1 == 0);
        n_ar_q12 = silk_nsq_noise_shape_feedback_loop(
            nsq.s_diff_shp_q14,
            &mut nsq.s_ar2_q14,
            ar_shp_q13,
            shaping_lpcorder,
            _arch,
        );

        n_ar_q12 = (n_ar_q12 as i64 + ((nsq.s_lf_ar_shp_q14 as i64 * tilt_q14 as i16 as i64) >> 16))
            as i32;

        n_lf_q12 = ((nsq.s_ltp_shp_q14[(nsq.s_ltp_shp_buf_idx - 1) as usize] as i64
            * lf_shp_q14 as i16 as i64)
            >> 16) as i32;
        n_lf_q12 = (n_lf_q12 as i64
            + ((nsq.s_lf_ar_shp_q14 as i64 * (lf_shp_q14 as i64 >> 16)) >> 16))
            as i32;

        debug_assert!(lag > 0 || signal_type != 2);

        tmp1 = (((lpc_pred_q10 as u32) << 2) as i32).wrapping_sub(n_ar_q12);
        tmp1 = tmp1.wrapping_sub(n_lf_q12);
        if lag > 0 {
            n_ltp_q13 = (((nsq.s_ltp_shp_q14[shp_lag_idx]
                .saturating_add(nsq.s_ltp_shp_q14[shp_lag_idx - 2]))
                as i64
                * harm_shape_firpacked_q14 as i16 as i64)
                >> 16) as i32;
            n_ltp_q13 = (n_ltp_q13 as i64
                + ((nsq.s_ltp_shp_q14[shp_lag_idx - 1] as i64
                    * (harm_shape_firpacked_q14 as i64 >> 16))
                    >> 16)) as i32;
            n_ltp_q13 = ((n_ltp_q13 as u32) << 1) as i32;
            shp_lag_idx += 1;
            tmp2 = ltp_pred_q13 - n_ltp_q13;
            tmp1 = tmp2.wrapping_add(((tmp1 as u32) << 1) as i32);
            tmp1 = if 3 == 1 {
                (tmp1 >> 1) + (tmp1 & 1)
            } else {
                ((tmp1 >> (3 - 1)) + 1) >> 1
            };
        } else {
            tmp1 = if 2 == 1 {
                (tmp1 >> 1) + (tmp1 & 1)
            } else {
                ((tmp1 >> (2 - 1)) + 1) >> 1
            };
        }

        r_q10 = x_sc_q10[_i] - tmp1;
        if nsq.rand_seed < 0 {
            r_q10 = -r_q10;
        }
        r_q10 = r_q10.clamp(-((31) << 10), (30) << 10);

        // Quantize
        q1_q10 = r_q10 - offset_q10;
        q1_q0 = q1_q10 >> 10;
        if lambda_q10 > 2048 {
            let rdo_offset: i32 = lambda_q10 / 2 - 512;
            if q1_q10 > rdo_offset {
                q1_q0 = (q1_q10 - rdo_offset) >> 10;
            } else if q1_q10 < -rdo_offset {
                q1_q0 = (q1_q10 + rdo_offset) >> 10;
            } else if q1_q10 < 0 {
                q1_q0 = -1;
            } else {
                q1_q0 = 0;
            }
        }

        // RD selection
        if q1_q0 > 0 {
            q1_q10 = ((q1_q0 as u32) << 10) as i32 - 80;
            q1_q10 += offset_q10;
            q2_q10 = q1_q10 + 1024;
            rd1_q20 = q1_q10 as i16 as i32 * lambda_q10 as i16 as i32;
            rd2_q20 = q2_q10 as i16 as i32 * lambda_q10 as i16 as i32;
        } else if q1_q0 == 0 {
            q1_q10 = offset_q10;
            q2_q10 = q1_q10 + (1024 - 80);
            rd1_q20 = q1_q10 as i16 as i32 * lambda_q10 as i16 as i32;
            rd2_q20 = q2_q10 as i16 as i32 * lambda_q10 as i16 as i32;
        } else if q1_q0 == -1 {
            q2_q10 = offset_q10;
            q1_q10 = q2_q10 - (1024 - 80);
            rd1_q20 = -q1_q10 as i16 as i32 * lambda_q10 as i16 as i32;
            rd2_q20 = q2_q10 as i16 as i32 * lambda_q10 as i16 as i32;
        } else {
            q1_q10 = ((q1_q0 as u32) << 10) as i32 + 80;
            q1_q10 += offset_q10;
            q2_q10 = q1_q10 + 1024;
            rd1_q20 = -q1_q10 as i16 as i32 * lambda_q10 as i16 as i32;
            rd2_q20 = -q2_q10 as i16 as i32 * lambda_q10 as i16 as i32;
        }
        rr_q10 = r_q10 - q1_q10;
        rd1_q20 += rr_q10 as i16 as i32 * rr_q10 as i16 as i32;
        rr_q10 = r_q10 - q2_q10;
        rd2_q20 += rr_q10 as i16 as i32 * rr_q10 as i16 as i32;
        if rd2_q20 < rd1_q20 {
            q1_q10 = q2_q10;
        }

        pulses[_i] = (if 10 == 1 {
            (q1_q10 >> 1) + (q1_q10 & 1)
        } else {
            ((q1_q10 >> (10 - 1)) + 1) >> 1
        }) as i8;

        // Excitation
        exc_q14 = ((q1_q10 as u32) << 4) as i32;
        if nsq.rand_seed < 0 {
            exc_q14 = -exc_q14;
        }
        lpc_exc_q14 = exc_q14 + ((ltp_pred_q13 as u32) << 1) as i32;
        xq_q14 = lpc_exc_q14.wrapping_add(((lpc_pred_q10 as u32) << 4) as i32);

        nsq.xq[xq_off + _i] = (if (if 8 == 1 {
            (((xq_q14 as i64 * gain_q10 as i64) >> 16) as i32 >> 1)
                + (((xq_q14 as i64 * gain_q10 as i64) >> 16) as i32 & 1)
        } else {
            ((((xq_q14 as i64 * gain_q10 as i64) >> 16) as i32 >> (8 - 1)) + 1) >> 1
        }) > SILK_INT16_MAX
        {
            SILK_INT16_MAX
        } else if (if 8 == 1 {
            (((xq_q14 as i64 * gain_q10 as i64) >> 16) as i32 >> 1)
                + (((xq_q14 as i64 * gain_q10 as i64) >> 16) as i32 & 1)
        } else {
            ((((xq_q14 as i64 * gain_q10 as i64) >> 16) as i32 >> (8 - 1)) + 1) >> 1
        }) < SILK_INT16_MIN
        {
            SILK_INT16_MIN
        } else if 8 == 1 {
            (((xq_q14 as i64 * gain_q10 as i64) >> 16) as i32 >> 1)
                + (((xq_q14 as i64 * gain_q10 as i64) >> 16) as i32 & 1)
        } else {
            ((((xq_q14 as i64 * gain_q10 as i64) >> 16) as i32 >> (8 - 1)) + 1) >> 1
        }) as i16;

        // Update state
        lpc_idx += 1;
        nsq.s_lpc_q14[lpc_idx] = xq_q14;

        nsq.s_diff_shp_q14 = xq_q14 - ((x_sc_q10[_i] as u32) << 4) as i32;
        s_lf_ar_shp_q14 = nsq
            .s_diff_shp_q14
            .wrapping_sub(((n_ar_q12 as u32) << 2) as i32);
        nsq.s_lf_ar_shp_q14 = s_lf_ar_shp_q14;

        nsq.s_ltp_shp_q14[nsq.s_ltp_shp_buf_idx as usize] =
            s_lf_ar_shp_q14.wrapping_sub(((n_lf_q12 as u32) << 2) as i32);
        s_ltp_q15[nsq.s_ltp_buf_idx as usize] = ((lpc_exc_q14 as u32) << 1) as i32;
        nsq.s_ltp_shp_buf_idx += 1;
        nsq.s_ltp_buf_idx += 1;

        nsq.rand_seed = (nsq.rand_seed as u32).wrapping_add(pulses[_i] as u32) as i32;
    }

    // Copy last NSQ_LPC_BUF_LENGTH values to the beginning
    nsq.s_lpc_q14
        .copy_within(length..length + NSQ_LPC_BUF_LENGTH, 0);
}

/// Upstream c: silk/nsq.c:silk_nsq_scale_states
#[inline]
#[allow(clippy::too_many_arguments)]
fn silk_nsq_scale_states(
    ps_enc_c: &NsqConfig,
    nsq: &mut silk_nsq_state,
    x16: &[i16],
    x_sc_q10: &mut [i32],
    s_ltp: &[i16],
    s_ltp_q15: &mut [i32],
    subfr: i32,
    ltp_scale_q14: i32,
    gains_q16: &[i32],
    pitch_l: &[i32],
    signal_type: i32,
) {
    let lag = pitch_l[subfr as usize];
    let mut inv_gain_q31 = silk_inverse32_varq(
        if gains_q16[subfr as usize] > 1 {
            gains_q16[subfr as usize]
        } else {
            1
        },
        47,
    );
    let inv_gain_q26 = if 5 == 1 {
        (inv_gain_q31 >> 1) + (inv_gain_q31 & 1)
    } else {
        ((inv_gain_q31 >> (5 - 1)) + 1) >> 1
    };

    for _i in 0..ps_enc_c.subfr_length {
        x_sc_q10[_i] = ((x16[_i] as i64 * inv_gain_q26 as i64) >> 16) as i32;
    }

    if nsq.rewhite_flag != 0 {
        if subfr == 0 {
            inv_gain_q31 = ((((inv_gain_q31 as i64 * ltp_scale_q14 as i16 as i64) >> 16) as i32
                as u32)
                << 2) as i32;
        }
        let start = (nsq.s_ltp_buf_idx - lag - LTP_ORDER as i32 / 2) as usize;
        let end = nsq.s_ltp_buf_idx as usize;
        for _i in start..end {
            s_ltp_q15[_i] = ((inv_gain_q31 as i64 * s_ltp[_i] as i64) >> 16) as i32;
        }
    }

    if gains_q16[subfr as usize] != nsq.prev_gain_q16 {
        let gain_adj_q16 = silk_div32_varq(nsq.prev_gain_q16, gains_q16[subfr as usize], 16);

        let shp_start = (nsq.s_ltp_shp_buf_idx - ps_enc_c.ltp_mem_length as i32) as usize;
        let shp_end = nsq.s_ltp_shp_buf_idx as usize;
        for _i in shp_start..shp_end {
            nsq.s_ltp_shp_q14[_i] =
                ((gain_adj_q16 as i64 * nsq.s_ltp_shp_q14[_i] as i64) >> 16) as i32;
        }

        if signal_type == TYPE_VOICED && nsq.rewhite_flag == 0 {
            let start = (nsq.s_ltp_buf_idx - lag - LTP_ORDER as i32 / 2) as usize;
            let end = nsq.s_ltp_buf_idx as usize;
            for val in s_ltp_q15[start..end].iter_mut() {
                *val = ((gain_adj_q16 as i64 * *val as i64) >> 16) as i32;
            }
        }

        nsq.s_lf_ar_shp_q14 = ((gain_adj_q16 as i64 * nsq.s_lf_ar_shp_q14 as i64) >> 16) as i32;
        nsq.s_diff_shp_q14 = ((gain_adj_q16 as i64 * nsq.s_diff_shp_q14 as i64) >> 16) as i32;

        for _i in 0..NSQ_LPC_BUF_LENGTH {
            nsq.s_lpc_q14[_i] = ((gain_adj_q16 as i64 * nsq.s_lpc_q14[_i] as i64) >> 16) as i32;
        }
        for _i in 0..MAX_SHAPE_LPC_ORDER as usize {
            nsq.s_ar2_q14[_i] = ((gain_adj_q16 as i64 * nsq.s_ar2_q14[_i] as i64) >> 16) as i32;
        }

        nsq.prev_gain_q16 = gains_q16[subfr as usize];
    }
}

/// Build the precomputed quantization lookup table used by the SSE4.1 quantizer.
/// Port of the table initialization from `silk/x86/NSQ_sse4_1.c:silk_nsq_sse4_1`.
///
/// table[32 + q1_q0] = [q1_q10, q2_q10, 2*(q1_q10 - q2_q10), rd1_q20 - rd2_q20 + q1² - q2²]
#[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
fn build_quantization_table(offset_q10: i32, lambda_q10: i32) -> [[i32; 4]; 64] {
    let mut table = [[0i32; 4]; 64];

    // q1_q0 == 0
    {
        let q1_q10 = offset_q10;
        let q2_q10 = offset_q10 + (1024 - QUANT_LEVEL_ADJUST_Q10);
        let rd1_q20 = q1_q10 * lambda_q10;
        let rd2_q20 = q2_q10 * lambda_q10;
        table[32] = [
            q1_q10,
            q2_q10,
            2 * (q1_q10 - q2_q10),
            (rd1_q20 - rd2_q20) + (q1_q10 * q1_q10 - q2_q10 * q2_q10),
        ];
    }

    // q1_q0 == -1
    {
        let q1_q10 = offset_q10 - (1024 - QUANT_LEVEL_ADJUST_Q10);
        let q2_q10 = offset_q10;
        let rd1_q20 = -q1_q10 * lambda_q10;
        let rd2_q20 = q2_q10 * lambda_q10;
        table[31] = [
            q1_q10,
            q2_q10,
            2 * (q1_q10 - q2_q10),
            (rd1_q20 - rd2_q20) + (q1_q10 * q1_q10 - q2_q10 * q2_q10),
        ];
    }

    // q1_q0 > 0 (k = 1..31)
    for k in 1..=31 {
        let tmp1 = offset_q10 + (k << 10);
        let q1_q10 = tmp1 - QUANT_LEVEL_ADJUST_Q10;
        let q2_q10 = tmp1 - QUANT_LEVEL_ADJUST_Q10 + 1024;
        let rd1_q20 = q1_q10 * lambda_q10;
        let rd2_q20 = q2_q10 * lambda_q10;
        table[(32 + k) as usize] = [
            q1_q10,
            q2_q10,
            2 * (q1_q10 - q2_q10),
            (rd1_q20 - rd2_q20) + (q1_q10 * q1_q10 - q2_q10 * q2_q10),
        ];
    }

    // q1_q0 < -1 (k = -32..-2)
    for k in -32..=-2 {
        let tmp1 = offset_q10 + (k << 10);
        let q1_q10 = tmp1 + QUANT_LEVEL_ADJUST_Q10;
        let q2_q10 = tmp1 + QUANT_LEVEL_ADJUST_Q10 + 1024;
        let rd1_q20 = -q1_q10 * lambda_q10;
        let rd2_q20 = -q2_q10 * lambda_q10;
        table[(32 + k) as usize] = [
            q1_q10,
            q2_q10,
            2 * (q1_q10 - q2_q10),
            (rd1_q20 - rd2_q20) + (q1_q10 * q1_q10 - q2_q10 * q2_q10),
        ];
    }

    table
}
