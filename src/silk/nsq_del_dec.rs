//! Noise shaping quantizer with delayed decision.
//!
//! Upstream c: `silk/NSQ_del_dec.c`

use crate::arch::Arch;
use crate::silk::define::{
    DECISION_DELAY, HARM_SHAPE_FIR_TAPS, LTP_ORDER, MAX_LPC_ORDER, MAX_SHAPE_LPC_ORDER,
    NSQ_LPC_BUF_LENGTH, TYPE_VOICED,
};
use crate::silk::inlines::{silk_div32_varq, silk_inverse32_varq};
use crate::silk::lpc_analysis_filter::silk_lpc_analysis_filter;
use crate::silk::sigproc_fix::{silk_min_int, silk_rand};
use crate::silk::structs::{silk_nsq_state, NsqConfig, SideInfoIndices};
use crate::silk::tables_other::SILK_QUANTIZATION_OFFSETS_Q10;
use crate::silk::typedefs::{SILK_INT16_MAX, SILK_INT16_MIN, SILK_INT32_MAX};

#[derive(Copy, Clone)]
#[repr(C)]
pub struct NSQ_del_dec_struct {
    pub s_lpc_q14: [i32; 96],
    pub rand_state: [i32; 40],
    pub q_q10: [i32; 40],
    pub xq_q14: [i32; 40],
    pub pred_q15: [i32; 40],
    pub shape_q14: [i32; 40],
    pub s_ar2_q14: [i32; 24],
    pub lf_ar_q14: i32,
    pub diff_q14: i32,
    pub seed: i32,
    pub seed_init: i32,
    pub rd_q10: i32,
}

impl Default for NSQ_del_dec_struct {
    fn default() -> Self {
        Self {
            s_lpc_q14: [0; 96],
            rand_state: [0; 40],
            q_q10: [0; 40],
            xq_q14: [0; 40],
            pred_q15: [0; 40],
            shape_q14: [0; 40],
            s_ar2_q14: [0; 24],
            lf_ar_q14: 0,
            diff_q14: 0,
            seed: 0,
            seed_init: 0,
            rd_q10: 0,
        }
    }
}

/// Copy all fields of src into dst except s_lpc_q14[0..keep].
/// This matches the c pattern: memcpy(dst+_i, src+_i, sizeof(struct)-_i*sizeof(i32))
/// which copies s_lpc_q14[_i..] and all fields after s_lpc_q14.
#[inline]
pub(crate) fn copy_del_dec_state_partial(
    dst: &mut NSQ_del_dec_struct,
    src: &NSQ_del_dec_struct,
    keep: usize,
) {
    dst.s_lpc_q14[keep..].copy_from_slice(&src.s_lpc_q14[keep..]);
    dst.rand_state = src.rand_state;
    dst.q_q10 = src.q_q10;
    dst.xq_q14 = src.xq_q14;
    dst.pred_q15 = src.pred_q15;
    dst.shape_q14 = src.shape_q14;
    dst.s_ar2_q14 = src.s_ar2_q14;
    dst.lf_ar_q14 = src.lf_ar_q14;
    dst.diff_q14 = src.diff_q14;
    dst.seed = src.seed;
    dst.seed_init = src.seed_init;
    dst.rd_q10 = src.rd_q10;
}

#[derive(Copy, Clone)]
#[repr(C)]
#[derive(Default)]
pub struct NSQ_sample_struct {
    pub q_q10: i32,
    pub rd_q10: i32,
    pub xq_q14: i32,
    pub lf_ar_q14: i32,
    pub diff_q14: i32,
    pub s_ltp_shp_q14: i32,
    pub lpc_exc_q14: i32,
}

pub type NsqSamplePair = [NSQ_sample_struct; 2];

/// Helper: saturating round-shift for xq output: silk_rshift_round + silk_sat16
#[inline]
fn rshift_round_sat16(val: i32, shift: i32) -> i16 {
    let rounded = if shift == 1 {
        (val >> 1) + (val & 1)
    } else {
        ((val >> (shift - 1)) + 1) >> 1
    };
    if rounded > SILK_INT16_MAX {
        SILK_INT16_MAX as i16
    } else if rounded < SILK_INT16_MIN {
        SILK_INT16_MIN as i16
    } else {
        rounded as i16
    }
}

/// Dispatch wrapper for nsq delayed-decision, matching upstream `silk_nsq_del_dec`.
#[cfg(feature = "simd")]
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn silk_nsq_del_dec(
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
    super::simd::silk_nsq_del_dec(
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

/// Scalar-only build wrapper for nsq delayed-decision.
#[cfg(not(feature = "simd"))]
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn silk_nsq_del_dec(
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
    silk_nsq_del_dec_c(
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

/// Upstream c: silk/NSQ_del_dec.c:silk_NSQ_del_dec_c
#[allow(clippy::too_many_arguments)]
pub fn silk_nsq_del_dec_c(
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
    // AVX2 fast path: replaces entire function when n_states_delayed_decision is 3 or 4
    #[cfg(feature = "simd")]
    {
        if super::simd::use_nsq_del_dec_avx2(ps_enc_c.arch, ps_enc_c.n_states_delayed_decision) {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            super::simd::silk_nsq_del_dec_avx2(
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
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            return;
        }
    }

    // NEON fast path: replaces entire function when n_states_delayed_decision is 3 or 4 on aarch64
    #[cfg(feature = "simd")]
    {
        if super::simd::use_neon_nsq_del_dec(ps_enc_c.arch, ps_enc_c.n_states_delayed_decision) {
            #[cfg(target_arch = "aarch64")]
            super::simd::silk_nsq_del_dec_neon(
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
            #[cfg(target_arch = "aarch64")]
            return;
        }
    }

    let mut lag: i32;
    let mut start_idx: i32;
    let mut winner_ind: i32;
    let mut subfr: i32;
    let mut last_smple_idx: i32;
    let mut smpl_buf_idx: i32;
    let mut decision_delay: i32;
    let mut harm_shape_firpacked_q14: i32;
    let mut rdmin_q10: i32;

    let ltp_mem_len = ps_enc_c.ltp_mem_length;
    let frame_len = ps_enc_c.frame_length;
    let subfr_len = ps_enc_c.subfr_length;
    let n_states = ps_enc_c.n_states_delayed_decision;
    let n_states = n_states as usize;

    lag = nsq.lag_prev;

    // MAX_DEL_DEC_STATES = 4; n_states <= 4
    const MAX_STATES: usize = 4;
    debug_assert!(n_states <= MAX_STATES);
    let mut ps_del_dec = [NSQ_del_dec_struct::default(); MAX_STATES];

    for (k, ps_del_dec) in ps_del_dec.iter_mut().take(n_states).enumerate() {
        ps_del_dec.seed = (k as i32 + ps_indices.seed as i32) & 3;
        ps_del_dec.seed_init = ps_del_dec.seed;
        ps_del_dec.rd_q10 = 0;
        ps_del_dec.lf_ar_q14 = nsq.s_lf_ar_shp_q14;
        ps_del_dec.diff_q14 = nsq.s_diff_shp_q14;
        ps_del_dec.shape_q14[0] = nsq.s_ltp_shp_q14[ltp_mem_len - 1];
        ps_del_dec.s_lpc_q14[..NSQ_LPC_BUF_LENGTH]
            .copy_from_slice(&nsq.s_lpc_q14[..NSQ_LPC_BUF_LENGTH]);
        ps_del_dec.s_ar2_q14 = nsq.s_ar2_q14;
    }

    let offset_q10 = SILK_QUANTIZATION_OFFSETS_Q10[(ps_indices.signal_type as i32 >> 1) as usize]
        [ps_indices.quant_offset_type as usize] as i32;
    smpl_buf_idx = 0;
    decision_delay = silk_min_int(DECISION_DELAY, subfr_len as i32);
    if ps_indices.signal_type as i32 == TYPE_VOICED {
        for k in 0..ps_enc_c.nb_subfr as i32 {
            decision_delay = silk_min_int(
                decision_delay,
                pitch_l[k as usize] - LTP_ORDER as i32 / 2 - 1,
            );
        }
    } else if lag > 0 {
        decision_delay = silk_min_int(decision_delay, lag - LTP_ORDER as i32 / 2 - 1);
    }
    let lsf_interpolation_flag: i32 = if ps_indices.nlsfinterp_coef_q2 as i32 == 4 {
        0
    } else {
        1
    };

    // ltp_mem_len + frame_len max: 320 + 320 = 640
    const MAX_LTP_FRAME: usize = 640;
    debug_assert!(ltp_mem_len + frame_len <= MAX_LTP_FRAME);
    let mut s_ltp_q15 = [0i32; MAX_LTP_FRAME];
    let mut s_ltp = [0i16; MAX_LTP_FRAME];
    // subfr_len max: MAX_SUB_FRAME_LENGTH = 80
    const MAX_SUBFR: usize = 80;
    debug_assert!(subfr_len <= MAX_SUBFR);
    let mut x_sc_q10 = [0i32; MAX_SUBFR];
    let mut delayed_gain_q10: [i32; 40] = [0; 40];

    let mut pxq_off: usize = ltp_mem_len;
    nsq.s_ltp_shp_buf_idx = ltp_mem_len as i32;
    nsq.s_ltp_buf_idx = ltp_mem_len as i32;
    subfr = 0;
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
                if k == 2 {
                    // Find winner among delayed decision states
                    rdmin_q10 = ps_del_dec[0].rd_q10;
                    winner_ind = 0;
                    for (_i, ps_del_dec) in ps_del_dec.iter().enumerate().take(n_states).skip(1) {
                        if ps_del_dec.rd_q10 < rdmin_q10 {
                            rdmin_q10 = ps_del_dec.rd_q10;
                            winner_ind = _i as i32;
                        }
                    }
                    // Penalize non-winners
                    for (_i, ps_del_dec) in ps_del_dec.iter_mut().enumerate().take(n_states) {
                        if _i as i32 != winner_ind {
                            ps_del_dec.rd_q10 += SILK_INT32_MAX >> 4;
                        }
                    }
                    // Output delayed samples from winner
                    let ps_dd = &ps_del_dec[winner_ind as usize];
                    last_smple_idx = smpl_buf_idx + decision_delay;
                    for _i in 0..decision_delay {
                        last_smple_idx = (last_smple_idx - 1) % DECISION_DELAY;
                        if last_smple_idx < 0 {
                            last_smple_idx += DECISION_DELAY;
                        }
                        let p_idx = (pulses_off as isize + (_i - decision_delay) as isize) as usize;
                        pulses[p_idx] = (if 10 == 1 {
                            (ps_dd.q_q10[last_smple_idx as usize] >> 1)
                                + (ps_dd.q_q10[last_smple_idx as usize] & 1)
                        } else {
                            ((ps_dd.q_q10[last_smple_idx as usize] >> (10 - 1)) + 1) >> 1
                        }) as i8;
                        let xq_val = (ps_dd.xq_q14[last_smple_idx as usize] as i64
                            * gains_q16[1] as i64)
                            >> 16;
                        let xq_idx = (pxq_off as isize + (_i - decision_delay) as isize) as usize;
                        nsq.xq[xq_idx] = rshift_round_sat16(xq_val as i32, 14);
                        nsq.s_ltp_shp_q14[(nsq.s_ltp_shp_buf_idx - decision_delay + _i) as usize] =
                            ps_dd.shape_q14[last_smple_idx as usize];
                    }
                    subfr = 0;
                }
                start_idx =
                    ltp_mem_len as i32 - lag - ps_enc_c.predict_lpcorder - LTP_ORDER as i32 / 2;
                debug_assert!(start_idx > 0);
                silk_lpc_analysis_filter(
                    &mut s_ltp[start_idx as usize..ltp_mem_len],
                    &nsq.xq[(start_idx + k * subfr_len as i32) as usize..]
                        [..ltp_mem_len - start_idx as usize],
                    a_q12,
                );
                nsq.s_ltp_buf_idx = ltp_mem_len as i32;
                nsq.rewhite_flag = 1;
            }
        }
        #[cfg(feature = "simd")]
        let use_simd = super::simd::use_nsq_sse4_1(ps_enc_c.arch);
        #[cfg(not(feature = "simd"))]
        let use_simd = false;

        if use_simd {
            #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
            super::simd::silk_nsq_del_dec_scale_states_sse4_1(
                ps_enc_c,
                nsq,
                &mut ps_del_dec,
                &x16[x16_off..x16_off + subfr_len],
                &mut x_sc_q10,
                &s_ltp,
                &mut s_ltp_q15,
                k,
                n_states as i32,
                ltp_scale_q14,
                gains_q16,
                pitch_l,
                ps_indices.signal_type as i32,
                decision_delay,
            );
        } else {
            silk_nsq_del_dec_scale_states(
                ps_enc_c,
                nsq,
                &mut ps_del_dec,
                &x16[x16_off..x16_off + subfr_len],
                &mut x_sc_q10,
                &s_ltp,
                &mut s_ltp_q15,
                k,
                n_states as i32,
                ltp_scale_q14,
                gains_q16,
                pitch_l,
                ps_indices.signal_type as i32,
                decision_delay,
            );
        }
        let fresh_subfr = subfr;
        subfr += 1;
        if use_simd {
            #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
            super::simd::silk_noise_shape_quantizer_del_dec_sse4_1(
                nsq,
                &mut ps_del_dec,
                ps_indices.signal_type as i32,
                &x_sc_q10,
                pulses,
                pulses_off,
                pxq_off,
                &mut s_ltp_q15,
                &mut delayed_gain_q10,
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
                fresh_subfr,
                ps_enc_c.shaping_lpcorder,
                ps_enc_c.predict_lpcorder,
                ps_enc_c.warping_q16,
                n_states as i32,
                &mut smpl_buf_idx,
                decision_delay,
            );
        } else {
            silk_noise_shape_quantizer_del_dec(
                nsq,
                &mut ps_del_dec,
                ps_indices.signal_type as i32,
                &x_sc_q10,
                pulses,
                pulses_off,
                pxq_off,
                &mut s_ltp_q15,
                &mut delayed_gain_q10,
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
                fresh_subfr,
                ps_enc_c.shaping_lpcorder,
                ps_enc_c.predict_lpcorder,
                ps_enc_c.warping_q16,
                n_states as i32,
                &mut smpl_buf_idx,
                decision_delay,
                ps_enc_c.arch,
            );
        }
        x16_off += subfr_len;
        pulses_off += subfr_len;
        pxq_off += subfr_len;
    }

    // Find final winner
    rdmin_q10 = ps_del_dec[0].rd_q10;
    winner_ind = 0;
    for (k, ps_del_dec) in ps_del_dec.iter().enumerate().take(n_states).skip(1) {
        if ps_del_dec.rd_q10 < rdmin_q10 {
            rdmin_q10 = ps_del_dec.rd_q10;
            winner_ind = k as i32;
        }
    }
    let ps_dd = &ps_del_dec[winner_ind as usize];
    ps_indices.seed = ps_dd.seed_init as i8;
    last_smple_idx = smpl_buf_idx + decision_delay;
    let gain_q10 = gains_q16[ps_enc_c.nb_subfr - 1] >> 6;
    for _i in 0..decision_delay {
        last_smple_idx = (last_smple_idx - 1) % DECISION_DELAY;
        if last_smple_idx < 0 {
            last_smple_idx += DECISION_DELAY;
        }
        let p_idx = (pulses_off as isize + (_i - decision_delay) as isize) as usize;
        pulses[p_idx] = (if 10 == 1 {
            (ps_dd.q_q10[last_smple_idx as usize] >> 1) + (ps_dd.q_q10[last_smple_idx as usize] & 1)
        } else {
            ((ps_dd.q_q10[last_smple_idx as usize] >> (10 - 1)) + 1) >> 1
        }) as i8;
        let xq_val = (ps_dd.xq_q14[last_smple_idx as usize] as i64 * gain_q10 as i64) >> 16;
        let xq_idx = (pxq_off as isize + (_i - decision_delay) as isize) as usize;
        nsq.xq[xq_idx] = rshift_round_sat16(xq_val as i32, 8);
        nsq.s_ltp_shp_q14[(nsq.s_ltp_shp_buf_idx - decision_delay + _i) as usize] =
            ps_dd.shape_q14[last_smple_idx as usize];
    }

    // Copy winner's state back to nsq
    nsq.s_lpc_q14[..NSQ_LPC_BUF_LENGTH]
        .copy_from_slice(&ps_dd.s_lpc_q14[subfr_len..subfr_len + NSQ_LPC_BUF_LENGTH]);
    nsq.s_ar2_q14 = ps_dd.s_ar2_q14;
    nsq.s_lf_ar_shp_q14 = ps_dd.lf_ar_q14;
    nsq.s_diff_shp_q14 = ps_dd.diff_q14;
    nsq.lag_prev = pitch_l[ps_enc_c.nb_subfr - 1];

    // Shift buffers
    nsq.xq.copy_within(frame_len..frame_len + ltp_mem_len, 0);
    nsq.s_ltp_shp_q14
        .copy_within(frame_len..frame_len + ltp_mem_len, 0);
}

/// Upstream c: silk/NSQ_del_dec.c:silk_noise_shape_quantizer_del_dec
#[inline]
#[allow(clippy::too_many_arguments)]
fn silk_noise_shape_quantizer_del_dec(
    nsq: &mut silk_nsq_state,
    ps_del_dec: &mut [NSQ_del_dec_struct],
    signal_type: i32,
    x_q10: &[i32],
    pulses: &mut [i8],
    pulses_off: usize,
    xq_off: usize,
    s_ltp_q15: &mut [i32],
    delayed_gain_q10: &mut [i32; 40],
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
    subfr: i32,
    shaping_lpcorder: i32,
    predict_lpcorder: i32,
    warping_q16: i32,
    n_states_delayed_decision: i32,
    smpl_buf_idx: &mut i32,
    decision_delay: i32,
    _arch: Arch,
) {
    let mut winner_ind: i32;
    let mut rdmin_ind: i32;
    let mut rdmax_ind: i32;
    let mut last_smple_idx: i32;
    let mut winner_rand_state: i32;
    let mut ltp_pred_q14: i32;
    let mut lpc_pred_q14: i32;
    let mut n_ar_q14: i32;
    let mut n_ltp_q14: i32;
    let mut n_lf_q14: i32;
    let mut r_q10: i32;
    let mut rr_q10: i32;
    let mut rd1_q10: i32;
    let mut rd2_q10: i32;
    let mut rdmin_q10: i32;
    let mut rdmax_q10: i32;
    let mut q1_q0: i32;
    let mut q1_q10: i32;
    let mut q2_q10: i32;
    let mut exc_q14: i32;
    let mut lpc_exc_q14: i32;
    let mut xq_q14: i32;
    let mut tmp1: i32;
    let mut tmp2: i32;
    let mut s_lf_ar_shp_q14: i32;

    debug_assert!(n_states_delayed_decision > 0);
    let n_states = n_states_delayed_decision as usize;
    let length = length as usize;

    const MAX_STATES: usize = 4;
    debug_assert!(n_states <= MAX_STATES);
    let mut ps_sample_state: [NsqSamplePair; MAX_STATES] =
        [[NSQ_sample_struct::default(); 2]; MAX_STATES];

    let mut shp_lag_idx = (nsq.s_ltp_shp_buf_idx - lag + HARM_SHAPE_FIR_TAPS / 2) as usize;
    let mut pred_lag_idx = (nsq.s_ltp_buf_idx - lag + LTP_ORDER as i32 / 2) as usize;
    let gain_q10: i32 = gain_q16 >> 6;

    // Pre-slice to hoist bounds checks out of the hot loop.
    let x_q10 = &x_q10[..length];
    let ar_shp_q13 = &ar_shp_q13[..shaping_lpcorder as usize];

    for (_i, &x_q10_i) in x_q10.iter().take(length).enumerate() {
        // LTP prediction (shared across all states)
        if signal_type == TYPE_VOICED {
            ltp_pred_q14 = 2;
            ltp_pred_q14 = (ltp_pred_q14 as i64
                + ((s_ltp_q15[pred_lag_idx] as i64 * b_q14[0] as i64) >> 16))
                as i32;
            ltp_pred_q14 = (ltp_pred_q14 as i64
                + ((s_ltp_q15[pred_lag_idx - 1] as i64 * b_q14[1] as i64) >> 16))
                as i32;
            ltp_pred_q14 = (ltp_pred_q14 as i64
                + ((s_ltp_q15[pred_lag_idx - 2] as i64 * b_q14[2] as i64) >> 16))
                as i32;
            ltp_pred_q14 = (ltp_pred_q14 as i64
                + ((s_ltp_q15[pred_lag_idx - 3] as i64 * b_q14[3] as i64) >> 16))
                as i32;
            ltp_pred_q14 = (ltp_pred_q14 as i64
                + ((s_ltp_q15[pred_lag_idx - 4] as i64 * b_q14[4] as i64) >> 16))
                as i32;
            ltp_pred_q14 = ((ltp_pred_q14 as u32) << 1) as i32;
            pred_lag_idx += 1;
        } else {
            ltp_pred_q14 = 0;
        }

        // Harmonic noise shaping (shared)
        if lag > 0 {
            n_ltp_q14 = (((nsq.s_ltp_shp_q14[shp_lag_idx]
                .saturating_add(nsq.s_ltp_shp_q14[shp_lag_idx - 2]))
                as i64
                * harm_shape_firpacked_q14 as i16 as i64)
                >> 16) as i32;
            n_ltp_q14 = (n_ltp_q14 as i64
                + ((nsq.s_ltp_shp_q14[shp_lag_idx - 1] as i64
                    * (harm_shape_firpacked_q14 as i64 >> 16))
                    >> 16)) as i32;
            n_ltp_q14 = ltp_pred_q14 - ((n_ltp_q14 as u32) << 2) as i32;
            shp_lag_idx += 1;
        } else {
            n_ltp_q14 = 0;
        }

        // Per-state processing
        for k in 0..n_states {
            let ps_dd = &mut ps_del_dec[k];
            ps_dd.seed = silk_rand(ps_dd.seed);

            // LPC prediction
            let lpc_idx = NSQ_LPC_BUF_LENGTH - 1 + _i;
            lpc_pred_q14 = crate::silk::nsq::silk_noise_shape_quantizer_short_prediction(
                &ps_dd.s_lpc_q14[..lpc_idx + 1],
                a_q12,
                predict_lpcorder,
                _arch,
            );
            lpc_pred_q14 = ((lpc_pred_q14 as u32) << 4) as i32;

            // Noise shaping with warping
            debug_assert!(shaping_lpcorder & 1 == 0);
            tmp2 = (ps_dd.diff_q14 as i64
                + ((ps_dd.s_ar2_q14[0] as i64 * warping_q16 as i16 as i64) >> 16))
                as i32;
            tmp1 = (ps_dd.s_ar2_q14[0] as i64
                + (((ps_dd.s_ar2_q14[1].wrapping_sub(tmp2)) as i64 * warping_q16 as i16 as i64)
                    >> 16)) as i32;
            ps_dd.s_ar2_q14[0] = tmp2;
            n_ar_q14 = shaping_lpcorder >> 1;
            n_ar_q14 = (n_ar_q14 as i64 + ((tmp2 as i64 * ar_shp_q13[0] as i64) >> 16)) as i32;

            let shaping_order = shaping_lpcorder as usize;
            let mut j = 2usize;
            while j < shaping_order {
                tmp2 = (ps_dd.s_ar2_q14[j - 1] as i64
                    + (((ps_dd.s_ar2_q14[j].wrapping_sub(tmp1)) as i64
                        * warping_q16 as i16 as i64)
                        >> 16)) as i32;
                ps_dd.s_ar2_q14[j - 1] = tmp1;
                n_ar_q14 =
                    (n_ar_q14 as i64 + ((tmp1 as i64 * ar_shp_q13[j - 1] as i64) >> 16)) as i32;
                tmp1 = (ps_dd.s_ar2_q14[j] as i64
                    + (((ps_dd.s_ar2_q14[j + 1].wrapping_sub(tmp2)) as i64
                        * warping_q16 as i16 as i64)
                        >> 16)) as i32;
                ps_dd.s_ar2_q14[j] = tmp2;
                n_ar_q14 = (n_ar_q14 as i64 + ((tmp2 as i64 * ar_shp_q13[j] as i64) >> 16)) as i32;
                j += 2;
            }
            ps_dd.s_ar2_q14[shaping_order - 1] = tmp1;
            n_ar_q14 = (n_ar_q14 as i64
                + ((tmp1 as i64 * ar_shp_q13[shaping_order - 1] as i64) >> 16))
                as i32;
            n_ar_q14 = ((n_ar_q14 as u32) << 1) as i32;
            n_ar_q14 = (n_ar_q14 as i64 + ((ps_dd.lf_ar_q14 as i64 * tilt_q14 as i16 as i64) >> 16))
                as i32;
            n_ar_q14 = ((n_ar_q14 as u32) << 2) as i32;

            n_lf_q14 = ((ps_dd.shape_q14[*smpl_buf_idx as usize] as i64 * lf_shp_q14 as i16 as i64)
                >> 16) as i32;
            n_lf_q14 = (n_lf_q14 as i64
                + ((ps_dd.lf_ar_q14 as i64 * (lf_shp_q14 as i64 >> 16)) >> 16))
                as i32;
            n_lf_q14 = ((n_lf_q14 as u32) << 2) as i32;

            tmp1 = n_ar_q14.saturating_add(n_lf_q14);
            tmp2 = n_ltp_q14 + lpc_pred_q14;
            tmp1 = tmp2.saturating_sub(tmp1);
            tmp1 = if 4 == 1 {
                (tmp1 >> 1) + (tmp1 & 1)
            } else {
                ((tmp1 >> (4 - 1)) + 1) >> 1
            };

            r_q10 = x_q10_i - tmp1;
            if ps_dd.seed < 0 {
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
            if q1_q0 > 0 {
                q1_q10 = ((q1_q0 as u32) << 10) as i32 - 80;
                q1_q10 += offset_q10;
                q2_q10 = q1_q10 + 1024;
                rd1_q10 = q1_q10 as i16 as i32 * lambda_q10 as i16 as i32;
                rd2_q10 = q2_q10 as i16 as i32 * lambda_q10 as i16 as i32;
            } else if q1_q0 == 0 {
                q1_q10 = offset_q10;
                q2_q10 = q1_q10 + (1024 - 80);
                rd1_q10 = q1_q10 as i16 as i32 * lambda_q10 as i16 as i32;
                rd2_q10 = q2_q10 as i16 as i32 * lambda_q10 as i16 as i32;
            } else if q1_q0 == -1 {
                q2_q10 = offset_q10;
                q1_q10 = q2_q10 - (1024 - 80);
                rd1_q10 = -q1_q10 as i16 as i32 * lambda_q10 as i16 as i32;
                rd2_q10 = q2_q10 as i16 as i32 * lambda_q10 as i16 as i32;
            } else {
                q1_q10 = ((q1_q0 as u32) << 10) as i32 + 80;
                q1_q10 += offset_q10;
                q2_q10 = q1_q10 + 1024;
                rd1_q10 = -q1_q10 as i16 as i32 * lambda_q10 as i16 as i32;
                rd2_q10 = -q2_q10 as i16 as i32 * lambda_q10 as i16 as i32;
            }
            rr_q10 = r_q10 - q1_q10;
            rd1_q10 = (rd1_q10 + rr_q10 as i16 as i32 * rr_q10 as i16 as i32) >> 10;
            rr_q10 = r_q10 - q2_q10;
            rd2_q10 = (rd2_q10 + rr_q10 as i16 as i32 * rr_q10 as i16 as i32) >> 10;

            if rd1_q10 < rd2_q10 {
                ps_sample_state[k][0].rd_q10 = ps_dd.rd_q10 + rd1_q10;
                ps_sample_state[k][1].rd_q10 = ps_dd.rd_q10 + rd2_q10;
                ps_sample_state[k][0].q_q10 = q1_q10;
                ps_sample_state[k][1].q_q10 = q2_q10;
            } else {
                ps_sample_state[k][0].rd_q10 = ps_dd.rd_q10 + rd2_q10;
                ps_sample_state[k][1].rd_q10 = ps_dd.rd_q10 + rd1_q10;
                ps_sample_state[k][0].q_q10 = q2_q10;
                ps_sample_state[k][1].q_q10 = q1_q10;
            }

            // Compute output for best and second-best candidate
            exc_q14 = ((ps_sample_state[k][0].q_q10 as u32) << 4) as i32;
            if ps_dd.seed < 0 {
                exc_q14 = -exc_q14;
            }
            lpc_exc_q14 = exc_q14 + ltp_pred_q14;
            xq_q14 = lpc_exc_q14 + lpc_pred_q14;
            ps_sample_state[k][0].diff_q14 = xq_q14 - ((x_q10_i as u32) << 4) as i32;
            s_lf_ar_shp_q14 = ps_sample_state[k][0].diff_q14 - n_ar_q14;
            ps_sample_state[k][0].s_ltp_shp_q14 = s_lf_ar_shp_q14.saturating_sub(n_lf_q14);
            ps_sample_state[k][0].lf_ar_q14 = s_lf_ar_shp_q14;
            ps_sample_state[k][0].lpc_exc_q14 = lpc_exc_q14;
            ps_sample_state[k][0].xq_q14 = xq_q14;

            exc_q14 = ((ps_sample_state[k][1].q_q10 as u32) << 4) as i32;
            if ps_dd.seed < 0 {
                exc_q14 = -exc_q14;
            }
            lpc_exc_q14 = exc_q14 + ltp_pred_q14;
            xq_q14 = lpc_exc_q14 + lpc_pred_q14;
            ps_sample_state[k][1].diff_q14 = xq_q14 - ((x_q10_i as u32) << 4) as i32;
            s_lf_ar_shp_q14 = ps_sample_state[k][1].diff_q14 - n_ar_q14;
            ps_sample_state[k][1].s_ltp_shp_q14 = s_lf_ar_shp_q14.saturating_sub(n_lf_q14);
            ps_sample_state[k][1].lf_ar_q14 = s_lf_ar_shp_q14;
            ps_sample_state[k][1].lpc_exc_q14 = lpc_exc_q14;
            ps_sample_state[k][1].xq_q14 = xq_q14;
        }

        // Update sample buffer index
        *smpl_buf_idx = (*smpl_buf_idx - 1) % DECISION_DELAY;
        if *smpl_buf_idx < 0 {
            *smpl_buf_idx += DECISION_DELAY;
        }
        last_smple_idx = (*smpl_buf_idx + decision_delay) % DECISION_DELAY;

        // Find winner among best candidates
        rdmin_q10 = ps_sample_state[0][0].rd_q10;
        winner_ind = 0;
        for (k, ps_sample_state) in ps_sample_state.iter().enumerate().take(n_states).skip(1) {
            if ps_sample_state[0].rd_q10 < rdmin_q10 {
                rdmin_q10 = ps_sample_state[0].rd_q10;
                winner_ind = k as i32;
            }
        }

        // Prune states with different rand state than winner
        winner_rand_state = ps_del_dec[winner_ind as usize].rand_state[last_smple_idx as usize];
        for k in 0..n_states {
            if ps_del_dec[k].rand_state[last_smple_idx as usize] != winner_rand_state {
                ps_sample_state[k][0].rd_q10 += 0x7fffffff >> 4;
                ps_sample_state[k][1].rd_q10 += 0x7fffffff >> 4;
            }
        }

        // Find worst-best and best-second for state replacement
        rdmax_q10 = ps_sample_state[0][0].rd_q10;
        rdmin_q10 = ps_sample_state[0][1].rd_q10;
        rdmax_ind = 0;
        rdmin_ind = 0;
        for (k, ps_sample_state) in ps_sample_state.iter().enumerate().take(n_states).skip(1) {
            if ps_sample_state[0].rd_q10 > rdmax_q10 {
                rdmax_q10 = ps_sample_state[0].rd_q10;
                rdmax_ind = k as i32;
            }
            if ps_sample_state[1].rd_q10 < rdmin_q10 {
                rdmin_q10 = ps_sample_state[1].rd_q10;
                rdmin_ind = k as i32;
            }
        }

        // Replace worst-best with best-second if beneficial
        if rdmin_q10 < rdmax_q10 {
            // Copy state: equivalent to c memcpy from offset _i
            // which copies s_lpc_q14[_i..] and all subsequent fields
            if rdmax_ind != rdmin_ind {
                let (left, right) = if rdmax_ind < rdmin_ind {
                    let (l, r) = ps_del_dec.split_at_mut(rdmin_ind as usize);
                    (&mut l[rdmax_ind as usize], &r[0])
                } else {
                    let (l, r) = ps_del_dec.split_at_mut(rdmax_ind as usize);
                    (&mut r[0], &l[rdmin_ind as usize])
                };
                copy_del_dec_state_partial(left, right, _i);
            }
            ps_sample_state[rdmax_ind as usize][0] = ps_sample_state[rdmin_ind as usize][1];
        }

        // Output delayed samples
        if subfr > 0 || _i as i32 >= decision_delay {
            let ps_dd_w = &ps_del_dec[winner_ind as usize];
            let out_idx = pulses_off + _i - decision_delay as usize;
            pulses[out_idx] = (if 10 == 1 {
                (ps_dd_w.q_q10[last_smple_idx as usize] >> 1)
                    + (ps_dd_w.q_q10[last_smple_idx as usize] & 1)
            } else {
                ((ps_dd_w.q_q10[last_smple_idx as usize] >> (10 - 1)) + 1) >> 1
            }) as i8;
            let xq_val = (ps_dd_w.xq_q14[last_smple_idx as usize] as i64
                * delayed_gain_q10[last_smple_idx as usize] as i64)
                >> 16;
            nsq.xq[xq_off + _i - decision_delay as usize] = rshift_round_sat16(xq_val as i32, 8);
            nsq.s_ltp_shp_q14[(nsq.s_ltp_shp_buf_idx - decision_delay) as usize] =
                ps_dd_w.shape_q14[last_smple_idx as usize];
            s_ltp_q15[(nsq.s_ltp_buf_idx - decision_delay) as usize] =
                ps_dd_w.pred_q15[last_smple_idx as usize];
        }
        nsq.s_ltp_shp_buf_idx += 1;
        nsq.s_ltp_buf_idx += 1;

        // Update all states with their best candidate
        for k in 0..n_states {
            let ps_ss = &ps_sample_state[k][0];
            let ps_dd = &mut ps_del_dec[k];
            ps_dd.lf_ar_q14 = ps_ss.lf_ar_q14;
            ps_dd.diff_q14 = ps_ss.diff_q14;
            ps_dd.s_lpc_q14[NSQ_LPC_BUF_LENGTH + _i] = ps_ss.xq_q14;
            ps_dd.xq_q14[*smpl_buf_idx as usize] = ps_ss.xq_q14;
            ps_dd.q_q10[*smpl_buf_idx as usize] = ps_ss.q_q10;
            ps_dd.pred_q15[*smpl_buf_idx as usize] = ((ps_ss.lpc_exc_q14 as u32) << 1) as i32;
            ps_dd.shape_q14[*smpl_buf_idx as usize] = ps_ss.s_ltp_shp_q14;
            ps_dd.seed = (ps_dd.seed as u32).wrapping_add(
                (if 10 == 1 {
                    (ps_ss.q_q10 >> 1) + (ps_ss.q_q10 & 1)
                } else {
                    ((ps_ss.q_q10 >> (10 - 1)) + 1) >> 1
                }) as u32,
            ) as i32;
            ps_dd.rand_state[*smpl_buf_idx as usize] = ps_dd.seed;
            ps_dd.rd_q10 = ps_ss.rd_q10;
        }
        delayed_gain_q10[*smpl_buf_idx as usize] = gain_q10;
    }

    // Copy LPC state for next subframe
    for dd in ps_del_dec[..n_states].iter_mut() {
        dd.s_lpc_q14
            .copy_within(length..length + NSQ_LPC_BUF_LENGTH, 0);
    }
}

/// Upstream c: silk/NSQ_del_dec.c:silk_nsq_del_dec_scale_states
#[inline]
#[allow(clippy::too_many_arguments)]
fn silk_nsq_del_dec_scale_states(
    ps_enc_c: &NsqConfig,
    nsq: &mut silk_nsq_state,
    ps_del_dec: &mut [NSQ_del_dec_struct],
    x16: &[i16],
    x_sc_q10: &mut [i32],
    s_ltp: &[i16],
    s_ltp_q15: &mut [i32],
    subfr: i32,
    n_states_delayed_decision: i32,
    ltp_scale_q14: i32,
    gains_q16: &[i32],
    pitch_l: &[i32],
    signal_type: i32,
    decision_delay: i32,
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
            let end = (nsq.s_ltp_buf_idx - decision_delay) as usize;
            for val in s_ltp_q15[start..end].iter_mut() {
                *val = ((gain_adj_q16 as i64 * *val as i64) >> 16) as i32;
            }
        }

        for ps_dd in ps_del_dec[..n_states_delayed_decision as usize].iter_mut() {
            ps_dd.lf_ar_q14 = ((gain_adj_q16 as i64 * ps_dd.lf_ar_q14 as i64) >> 16) as i32;
            ps_dd.diff_q14 = ((gain_adj_q16 as i64 * ps_dd.diff_q14 as i64) >> 16) as i32;
            for j in 0..NSQ_LPC_BUF_LENGTH {
                ps_dd.s_lpc_q14[j] =
                    ((gain_adj_q16 as i64 * ps_dd.s_lpc_q14[j] as i64) >> 16) as i32;
            }
            for j in 0..MAX_SHAPE_LPC_ORDER as usize {
                ps_dd.s_ar2_q14[j] =
                    ((gain_adj_q16 as i64 * ps_dd.s_ar2_q14[j] as i64) >> 16) as i32;
            }
            for j in 0..DECISION_DELAY as usize {
                ps_dd.pred_q15[j] = ((gain_adj_q16 as i64 * ps_dd.pred_q15[j] as i64) >> 16) as i32;
                ps_dd.shape_q14[j] =
                    ((gain_adj_q16 as i64 * ps_dd.shape_q14[j] as i64) >> 16) as i32;
            }
        }

        nsq.prev_gain_q16 = gains_q16[subfr as usize];
    }
}
