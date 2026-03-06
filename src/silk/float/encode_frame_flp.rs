//! Floating-point SILK frame encoding.
//!
//! Upstream c: `silk/float/encode_frame_FLP.c`

use crate::celt::entcode::{ec_tell, EcCtxSaved};
use crate::celt::entenc::EcEnc;
use crate::silk::define::{
    CODE_CONDITIONALLY, LA_SHAPE_MS, MAX_CONSECUTIVE_DTX, NB_SPEECH_FRAMES_BEFORE_DTX,
    N_LEVELS_QGAIN, TYPE_NO_VOICE_ACTIVITY, TYPE_UNVOICED, VAD_NO_ACTIVITY,
};
use crate::silk::encode_indices::silk_encode_indices;
use crate::silk::encode_pulses::silk_encode_pulses;
use crate::silk::float::find_pitch_lags_flp::silk_find_pitch_lags_flp;
use crate::silk::float::find_pred_coefs_flp::silk_find_pred_coefs_flp;
use crate::silk::float::noise_shape_analysis_flp::silk_noise_shape_analysis_flp;
use crate::silk::float::process_gains_flp::silk_process_gains_flp;
use crate::silk::float::sigproc_flp::silk_short2float_array;
use crate::silk::float::structs_flp::{silk_encoder_control_FLP, silk_encoder_state_FLP};
use crate::silk::float::wrappers_flp::silk_nsq_wrapper_flp;
use crate::silk::gain_quant::{silk_gains_dequant, silk_gains_id, silk_gains_quant};
use crate::silk::lp_variable_cutoff::silk_lp_variable_cutoff;
use crate::silk::sigproc_fix::silk_min_int;
use crate::silk::structs::silk_nsq_state;
use crate::silk::tuning_parameters::{LBRR_SPEECH_ACTIVITY_THRES, SPEECH_ACTIVITY_DTX_THRES};
use crate::silk::vad::silk_vad_get_sa_q8;

/// Upstream c: silk/float/encode_frame_FLP.c:silk_encode_do_VAD_FLP
pub fn silk_encode_do_vad_flp(ps_enc: &mut silk_encoder_state_FLP, activity: i32) {
    let activity_threshold: i32 =
        ((SPEECH_ACTIVITY_DTX_THRES * ((1) << 8) as f32) as f64 + 0.5f64) as i32;
    let mut vad_input = [0i16; 321];
    vad_input.copy_from_slice(&ps_enc.s_cmn.input_buf[1..]);
    silk_vad_get_sa_q8(&mut ps_enc.s_cmn, &vad_input);
    if activity == VAD_NO_ACTIVITY && ps_enc.s_cmn.speech_activity_q8 >= activity_threshold {
        ps_enc.s_cmn.speech_activity_q8 = activity_threshold - 1;
    }
    if ps_enc.s_cmn.speech_activity_q8 < activity_threshold {
        ps_enc.s_cmn.indices.signal_type = TYPE_NO_VOICE_ACTIVITY as i8;
        ps_enc.s_cmn.no_speech_counter += 1;
        if ps_enc.s_cmn.no_speech_counter <= NB_SPEECH_FRAMES_BEFORE_DTX {
            ps_enc.s_cmn.in_dtx = 0;
        } else if ps_enc.s_cmn.no_speech_counter > MAX_CONSECUTIVE_DTX + NB_SPEECH_FRAMES_BEFORE_DTX
        {
            ps_enc.s_cmn.no_speech_counter = NB_SPEECH_FRAMES_BEFORE_DTX;
            ps_enc.s_cmn.in_dtx = 0;
        }
        ps_enc.s_cmn.vad_flags[ps_enc.s_cmn.n_frames_encoded as usize] = 0;
    } else {
        ps_enc.s_cmn.no_speech_counter = 0;
        ps_enc.s_cmn.in_dtx = 0;
        ps_enc.s_cmn.indices.signal_type = TYPE_UNVOICED as i8;
        ps_enc.s_cmn.vad_flags[ps_enc.s_cmn.n_frames_encoded as usize] = 1;
    };
}
/// Upstream c: silk/float/encode_frame_FLP.c:silk_encode_frame_FLP
pub fn silk_encode_frame_flp(
    ps_enc: &mut silk_encoder_state_FLP,
    pn_bytes_out: &mut i32,
    mut ps_range_enc: Option<&mut EcEnc>,
    cond_coding: i32,
    max_bits: i32,
    use_cbr: i32,
) -> i32 {
    let mut s_enc_ctrl: silk_encoder_control_FLP = silk_encoder_control_FLP {
        gains: [0.; 4],
        pred_coef: [[0.; 16]; 2],
        ltp_coef: [0.; 20],
        ltp_scale: 0.,
        pitch_l: [0; 4],
        ar: [0.; 96],
        lf_ma_shp: [0.; 4],
        lf_ar_shp: [0.; 4],
        tilt: [0.; 4],
        harm_shape_gain: [0.; 4],
        lambda: 0.,
        input_quality: 0.,
        coding_quality: 0.,
        pred_gain: 0.,
        lt_pred_cod_gain: 0.,
        res_nrg: [0.; 4],
        gains_unq_q16: [0; 4],
        last_gain_index_prev: 0,
    };
    let mut _i: i32;
    let mut iter: i32;
    let max_iter: i32;
    let mut found_upper: i32;
    let mut found_lower: i32;
    let ret: i32 = 0;
    let mut res_pitch: [f32; 672] = [0.; 672];
    let s_range_enc_copy: EcCtxSaved;
    let mut s_range_enc_copy2 = EcCtxSaved::default();
    let s_nsq_copy: silk_nsq_state;
    let mut s_nsq_copy2: silk_nsq_state = silk_nsq_state {
        xq: [0; 640],
        s_ltp_shp_q14: [0; 640],
        s_lpc_q14: [0; 96],
        s_ar2_q14: [0; 24],
        s_lf_ar_shp_q14: 0,
        s_diff_shp_q14: 0,
        lag_prev: 0,
        s_ltp_buf_idx: 0,
        s_ltp_shp_buf_idx: 0,
        rand_seed: 0,
        prev_gain_q16: 0,
        rewhite_flag: 0,
    };
    let seed_copy: i32;
    let mut n_bits: i32;
    let mut n_bits_lower: i32;
    let mut n_bits_upper: i32;
    let mut gain_mult_lower: i32;
    let mut gain_mult_upper: i32;
    let mut gains_id: i32;
    let mut gains_id_lower: i32;
    let mut gains_id_upper: i32;
    let mut gain_mult_q8: i16;
    let ec_prev_lag_index_copy: i16;
    let ec_prev_signal_type_copy: i32;
    let mut last_gain_index_copy2: i8;
    let mut p_gains_q16: [i32; 4] = [0; 4];
    let mut ec_buf_copy: [u8; 1275] = [0; 1275];
    let mut gain_lock: [i32; 4] = [0, 0, 0, 0];
    let mut best_gain_mult: [i16; 4] = [0; 4];
    let mut best_sum: [i32; 4] = [0; 4];
    // For CBR, 5 bits below budget is close enough. For VBR, allow up to 25% below the cap.
    let bits_margin: i32 = if use_cbr != 0 { 5 } else { max_bits / 4 };
    gain_mult_upper = 0;
    gain_mult_lower = gain_mult_upper;
    n_bits_upper = gain_mult_lower;
    n_bits_lower = n_bits_upper;
    last_gain_index_copy2 = n_bits_lower as i8;
    let fresh0 = ps_enc.s_cmn.frame_counter;
    ps_enc.s_cmn.frame_counter = fresh0 + 1;
    ps_enc.s_cmn.indices.seed = (fresh0 & 3) as i8;
    let ltp_mem = ps_enc.s_cmn.ltp_mem_length;
    let frame_length = ps_enc.s_cmn.frame_length;
    let x_frame_off = ltp_mem;
    silk_lp_variable_cutoff(
        &mut ps_enc.s_cmn.s_lp,
        &mut ps_enc.s_cmn.input_buf[1..1 + frame_length],
    );
    {
        let la_offset = (LA_SHAPE_MS * ps_enc.s_cmn.fs_k_hz) as usize;
        let dst_start = x_frame_off + la_offset;
        silk_short2float_array(
            &mut ps_enc.x_buf[dst_start..dst_start + frame_length],
            &ps_enc.s_cmn.input_buf[1..1 + frame_length],
        );
    }
    _i = 0;
    while _i < 8 {
        let idx = x_frame_off
            + (LA_SHAPE_MS * ps_enc.s_cmn.fs_k_hz) as usize
            + _i as usize * (frame_length >> 3);
        ps_enc.x_buf[idx] += (1 - (_i & 2)) as f32 * 1e-6f32;
        _i += 1;
    }
    if ps_enc.s_cmn.prefill_flag == 0 {
        let ps_range_enc = &mut **ps_range_enc.as_mut().unwrap();

        // Copy x_buf to local to avoid borrow conflicts (functions take &mut ps_enc
        // while also needing to read from ps_enc.x_buf)
        let x_buf_copy = ps_enc.x_buf;
        {
            let ps_enc = &mut *ps_enc;
            let ltp_mem = ps_enc.s_cmn.ltp_mem_length;
            let frame_len = ps_enc.s_cmn.frame_length;
            let la_pitch = ps_enc.s_cmn.la_pitch as usize;
            let buf_len = la_pitch + frame_len + ltp_mem;
            silk_find_pitch_lags_flp(
                ps_enc,
                &mut s_enc_ctrl,
                &mut res_pitch[..buf_len],
                &x_buf_copy[..buf_len],
                ps_enc.s_cmn.arch,
            );
        }
        {
            let ps_enc = &mut *ps_enc;
            let ltp_mem = ps_enc.s_cmn.ltp_mem_length;
            let la_shape = (LA_SHAPE_MS * ps_enc.s_cmn.fs_k_hz) as usize;
            let x_start = ltp_mem - la_shape;
            let nb_subfr = ps_enc.s_cmn.nb_subfr;
            let subfr_len = ps_enc.s_cmn.subfr_length;
            // x range: from x_start, the function reads shape_win_length per subframe,
            // advancing by subfr_length each iteration
            let x_len = (nb_subfr - 1) * subfr_len + ps_enc.s_cmn.shape_win_length as usize;
            silk_noise_shape_analysis_flp(
                ps_enc,
                &mut s_enc_ctrl,
                &res_pitch[ltp_mem..ltp_mem + nb_subfr * subfr_len],
                &x_buf_copy[x_start..x_start + x_len],
            );
        }
        {
            let ps_enc = &mut *ps_enc;
            let ltp_mem = ps_enc.s_cmn.ltp_mem_length;
            let nb_subfr = ps_enc.s_cmn.nb_subfr;
            let subfr_len = ps_enc.s_cmn.subfr_length;
            let res_total = ltp_mem + nb_subfr * subfr_len + crate::silk::define::LTP_ORDER;
            let x_total = ltp_mem + nb_subfr * subfr_len;
            silk_find_pred_coefs_flp(
                ps_enc,
                &mut s_enc_ctrl,
                &res_pitch[..res_total],
                &x_buf_copy[..x_total],
                cond_coding,
            );
        }
        silk_process_gains_flp(&mut *ps_enc, &mut s_enc_ctrl, cond_coding);
        silk_lbrr_encode_flp(ps_enc, &mut s_enc_ctrl, x_frame_off, cond_coding);
        max_iter = 6;
        gain_mult_q8 = (((1) << 8) as f64 + 0.5f64) as i32 as i16;
        found_lower = 0;
        found_upper = 0;
        gains_id = silk_gains_id(&(&ps_enc.s_cmn.indices.gains_indices)[..ps_enc.s_cmn.nb_subfr]);
        gains_id_lower = -1;
        gains_id_upper = -1;
        s_range_enc_copy = ps_range_enc.save();
        s_nsq_copy = ps_enc.s_cmn.s_nsq;
        seed_copy = ps_enc.s_cmn.indices.seed as i32;
        ec_prev_lag_index_copy = ps_enc.s_cmn.ec_prev_lag_index;
        ec_prev_signal_type_copy = ps_enc.s_cmn.ec_prev_signal_type;
        iter = 0;
        loop {
            if gains_id == gains_id_lower {
                n_bits = n_bits_lower;
            } else if gains_id == gains_id_upper {
                n_bits = n_bits_upper;
            } else {
                if iter > 0 {
                    ps_range_enc.restore(s_range_enc_copy);
                    ps_enc.s_cmn.s_nsq = s_nsq_copy;
                    ps_enc.s_cmn.indices.seed = seed_copy as i8;
                    ps_enc.s_cmn.ec_prev_lag_index = ec_prev_lag_index_copy;
                    ps_enc.s_cmn.ec_prev_signal_type = ec_prev_signal_type_copy;
                }
                {
                    let total_len = ps_enc.s_cmn.nb_subfr * ps_enc.s_cmn.subfr_length;
                    let frame_len = ps_enc.s_cmn.frame_length;
                    let cfg = ps_enc.s_cmn.nsq_config();
                    silk_nsq_wrapper_flp(
                        &cfg,
                        &s_enc_ctrl,
                        &mut ps_enc.s_cmn.indices,
                        &mut ps_enc.s_cmn.s_nsq,
                        &mut ps_enc.s_cmn.pulses[..total_len],
                        &ps_enc.x_buf[x_frame_off..x_frame_off + frame_len],
                    );
                }
                if iter == max_iter && found_lower == 0 {
                    s_range_enc_copy2 = ps_range_enc.save();
                }
                {
                    let n_frames_encoded = ps_enc.s_cmn.n_frames_encoded;
                    silk_encode_indices(
                        &mut ps_enc.s_cmn,
                        ps_range_enc,
                        n_frames_encoded,
                        0,
                        cond_coding,
                    );
                }
                silk_encode_pulses(
                    ps_range_enc,
                    ps_enc.s_cmn.indices.signal_type as i32,
                    ps_enc.s_cmn.indices.quant_offset_type as i32,
                    &mut ps_enc.s_cmn.pulses,
                    ps_enc.s_cmn.frame_length,
                );
                n_bits = ec_tell(ps_range_enc);
                if iter == max_iter && found_lower == 0 && n_bits > max_bits {
                    ps_range_enc.restore(s_range_enc_copy2);
                    ps_enc.s_shape.last_gain_index = s_enc_ctrl.last_gain_index_prev;
                    _i = 0;
                    while _i < ps_enc.s_cmn.nb_subfr as i32 {
                        ps_enc.s_cmn.indices.gains_indices[_i as usize] = 4;
                        _i += 1;
                    }
                    if cond_coding != CODE_CONDITIONALLY {
                        ps_enc.s_cmn.indices.gains_indices[0_usize] =
                            s_enc_ctrl.last_gain_index_prev;
                    }
                    ps_enc.s_cmn.ec_prev_lag_index = ec_prev_lag_index_copy;
                    ps_enc.s_cmn.ec_prev_signal_type = ec_prev_signal_type_copy;
                    _i = 0;
                    while _i < ps_enc.s_cmn.frame_length as i32 {
                        ps_enc.s_cmn.pulses[_i as usize] = 0;
                        _i += 1;
                    }
                    {
                        let n_frames_encoded = ps_enc.s_cmn.n_frames_encoded;
                        silk_encode_indices(
                            &mut ps_enc.s_cmn,
                            ps_range_enc,
                            n_frames_encoded,
                            0,
                            cond_coding,
                        );
                    }
                    silk_encode_pulses(
                        ps_range_enc,
                        ps_enc.s_cmn.indices.signal_type as i32,
                        ps_enc.s_cmn.indices.quant_offset_type as i32,
                        &mut ps_enc.s_cmn.pulses,
                        ps_enc.s_cmn.frame_length,
                    );
                    n_bits = ec_tell(ps_range_enc);
                }
                if use_cbr == 0 && iter == 0 && n_bits <= max_bits {
                    break;
                }
            }
            if iter == max_iter {
                if found_lower != 0 && (gains_id == gains_id_lower || n_bits > max_bits) {
                    ps_range_enc.restore(s_range_enc_copy2);
                    debug_assert!(s_range_enc_copy2.offs <= 1275);
                    let offs = s_range_enc_copy2.offs as usize;
                    ps_range_enc.buf[..offs].copy_from_slice(&ec_buf_copy[..offs]);
                    ps_enc.s_cmn.s_nsq = s_nsq_copy2;
                    ps_enc.s_shape.last_gain_index = last_gain_index_copy2;
                }
                break;
            } else {
                if n_bits > max_bits {
                    if found_lower == 0 && iter >= 2 {
                        s_enc_ctrl.lambda = if s_enc_ctrl.lambda * 1.5f32 > 1.5f32 {
                            s_enc_ctrl.lambda * 1.5f32
                        } else {
                            1.5f32
                        };
                        ps_enc.s_cmn.indices.quant_offset_type = 0;
                        found_upper = 0;
                        gains_id_upper = -1;
                    } else {
                        found_upper = 1;
                        n_bits_upper = n_bits;
                        gain_mult_upper = gain_mult_q8 as i32;
                        gains_id_upper = gains_id;
                    }
                } else {
                    if n_bits >= max_bits - bits_margin {
                        break;
                    }
                    found_lower = 1;
                    n_bits_lower = n_bits;
                    gain_mult_lower = gain_mult_q8 as i32;
                    if gains_id != gains_id_lower {
                        gains_id_lower = gains_id;
                        s_range_enc_copy2 = ps_range_enc.save();
                        debug_assert!(ps_range_enc.offs <= 1275);
                        let offs = ps_range_enc.offs as usize;
                        ec_buf_copy[..offs].copy_from_slice(&ps_range_enc.buf[..offs]);
                        s_nsq_copy2 = ps_enc.s_cmn.s_nsq;
                        last_gain_index_copy2 = ps_enc.s_shape.last_gain_index;
                    }
                }
                if found_lower == 0 && n_bits > max_bits {
                    let mut j: i32;
                    _i = 0;
                    while _i < ps_enc.s_cmn.nb_subfr as i32 {
                        let mut sum: i32 = 0;
                        j = _i * ps_enc.s_cmn.subfr_length as i32;
                        while j < (_i + 1) * ps_enc.s_cmn.subfr_length as i32 {
                            sum += (ps_enc.s_cmn.pulses[j as usize] as i32).abs();
                            j += 1;
                        }
                        if iter == 0 || sum < best_sum[_i as usize] && gain_lock[_i as usize] == 0 {
                            best_sum[_i as usize] = sum;
                            best_gain_mult[_i as usize] = gain_mult_q8;
                        } else {
                            gain_lock[_i as usize] = 1;
                        }
                        _i += 1;
                    }
                }
                if found_lower & found_upper == 0 {
                    if n_bits > max_bits {
                        gain_mult_q8 = 1024i32.min(gain_mult_q8 as i32 * 3 / 2) as i16;
                    } else {
                        gain_mult_q8 = 64i32.max(gain_mult_q8 as i32 * 4 / 5) as i16;
                    }
                } else {
                    gain_mult_q8 = (gain_mult_lower
                        + (gain_mult_upper - gain_mult_lower) * (max_bits - n_bits_lower)
                            / (n_bits_upper - n_bits_lower))
                        as i16;
                    if gain_mult_q8 as i32
                        > gain_mult_lower + ((gain_mult_upper - gain_mult_lower) >> 2)
                    {
                        gain_mult_q8 =
                            (gain_mult_lower + ((gain_mult_upper - gain_mult_lower) >> 2)) as i16;
                    } else if (gain_mult_q8 as i32)
                        < gain_mult_upper - ((gain_mult_upper - gain_mult_lower) >> 2)
                    {
                        gain_mult_q8 =
                            (gain_mult_upper - ((gain_mult_upper - gain_mult_lower) >> 2)) as i16;
                    }
                }
                _i = 0;
                while _i < ps_enc.s_cmn.nb_subfr as i32 {
                    let tmp: i16 = if gain_lock[_i as usize] != 0 {
                        best_gain_mult[_i as usize]
                    } else {
                        gain_mult_q8
                    };
                    p_gains_q16[_i as usize] = (((if 0x80000000_u32 as i32 >> 8 > 0x7fffffff >> 8 {
                        if ((s_enc_ctrl.gains_unq_q16[_i as usize] as i64 * tmp as i64) >> 16)
                            as i32
                            > 0x80000000_u32 as i32 >> 8
                        {
                            0x80000000_u32 as i32 >> 8
                        } else if (((s_enc_ctrl.gains_unq_q16[_i as usize] as i64 * tmp as i64)
                            >> 16) as i32)
                            < 0x7fffffff >> 8
                        {
                            0x7fffffff >> 8
                        } else {
                            ((s_enc_ctrl.gains_unq_q16[_i as usize] as i64 * tmp as i64) >> 16)
                                as i32
                        }
                    } else if ((s_enc_ctrl.gains_unq_q16[_i as usize] as i64 * tmp as i64) >> 16)
                        as i32
                        > 0x7fffffff >> 8
                    {
                        0x7fffffff >> 8
                    } else if (((s_enc_ctrl.gains_unq_q16[_i as usize] as i64 * tmp as i64) >> 16)
                        as i32)
                        < 0x80000000_u32 as i32 >> 8
                    {
                        0x80000000_u32 as i32 >> 8
                    } else {
                        ((s_enc_ctrl.gains_unq_q16[_i as usize] as i64 * tmp as i64) >> 16) as i32
                    }) as u32)
                        << 8) as i32;
                    _i += 1;
                }
                ps_enc.s_shape.last_gain_index = s_enc_ctrl.last_gain_index_prev;
                silk_gains_quant(
                    &mut (&mut ps_enc.s_cmn.indices.gains_indices)[..ps_enc.s_cmn.nb_subfr],
                    &mut p_gains_q16[..ps_enc.s_cmn.nb_subfr],
                    &mut ps_enc.s_shape.last_gain_index,
                    cond_coding == CODE_CONDITIONALLY,
                );
                gains_id =
                    silk_gains_id(&(&ps_enc.s_cmn.indices.gains_indices)[..ps_enc.s_cmn.nb_subfr]);
                _i = 0;
                while _i < ps_enc.s_cmn.nb_subfr as i32 {
                    s_enc_ctrl.gains[_i as usize] = p_gains_q16[_i as usize] as f32 / 65536.0f32;
                    _i += 1;
                }
                iter += 1;
            }
        }
    }
    {
        let shift_len = ltp_mem + 5 * ps_enc.s_cmn.fs_k_hz as usize;
        ps_enc
            .x_buf
            .copy_within(frame_length..frame_length + shift_len, 0);
    }
    if ps_enc.s_cmn.prefill_flag != 0 {
        *pn_bytes_out = 0;
        return ret;
    }
    ps_enc.s_cmn.prev_lag = s_enc_ctrl.pitch_l[ps_enc.s_cmn.nb_subfr - 1];
    ps_enc.s_cmn.prev_signal_type = ps_enc.s_cmn.indices.signal_type;
    ps_enc.s_cmn.first_frame_after_reset = 0;
    *pn_bytes_out = (ec_tell(ps_range_enc.unwrap()) + 7) >> 3;

    ret
}
/// Upstream c: silk/float/encode_frame_FLP.c:silk_LBRR_encode_FLP
#[inline]
pub fn silk_lbrr_encode_flp(
    ps_enc: &mut silk_encoder_state_FLP,
    ps_enc_ctrl: &mut silk_encoder_control_FLP,
    x_frame_off: usize,
    cond_coding: i32,
) {
    let mut k: i32;
    let mut gains_q16: [i32; 4] = [0; 4];
    let nb_subfr = ps_enc.s_cmn.nb_subfr;
    let n_frames_encoded = ps_enc.s_cmn.n_frames_encoded as usize;
    let mut s_nsq_lbrr: silk_nsq_state = ps_enc.s_cmn.s_nsq;
    if ps_enc.s_cmn.lbrr_enabled != 0
        && ps_enc.s_cmn.speech_activity_q8
            > ((LBRR_SPEECH_ACTIVITY_THRES * ((1) << 8) as f32) as f64 + 0.5f64) as i32
    {
        ps_enc.s_cmn.lbrr_flags[n_frames_encoded] = 1;
        ps_enc.s_cmn.indices_lbrr[n_frames_encoded] = ps_enc.s_cmn.indices;
        let temp_gains: [f32; 4] = ps_enc_ctrl.gains;
        let ps_indices_lbrr = &mut ps_enc.s_cmn.indices_lbrr[n_frames_encoded];
        if ps_enc.s_cmn.n_frames_encoded == 0 || ps_enc.s_cmn.lbrr_flags[n_frames_encoded - 1] == 0
        {
            ps_enc.s_cmn.lbrrprev_last_gain_index = ps_enc.s_shape.last_gain_index;
            ps_indices_lbrr.gains_indices[0] =
                (ps_indices_lbrr.gains_indices[0] as i32 + ps_enc.s_cmn.lbrr_gain_increases) as i8;
            ps_indices_lbrr.gains_indices[0] = silk_min_int(
                ps_indices_lbrr.gains_indices[0] as i32,
                N_LEVELS_QGAIN as i32 - 1,
            ) as i8;
        }
        silk_gains_dequant(
            &mut gains_q16[..nb_subfr],
            &ps_indices_lbrr.gains_indices[..nb_subfr],
            &mut ps_enc.s_cmn.lbrrprev_last_gain_index,
            cond_coding == CODE_CONDITIONALLY,
        );
        k = 0;
        while k < nb_subfr as i32 {
            ps_enc_ctrl.gains[k as usize] = gains_q16[k as usize] as f32 * (1.0f32 / 65536.0f32);
            k += 1;
        }
        {
            let total_len = ps_enc.s_cmn.nb_subfr * ps_enc.s_cmn.subfr_length;
            let frame_len = ps_enc.s_cmn.frame_length;
            let cfg = ps_enc.s_cmn.nsq_config();
            silk_nsq_wrapper_flp(
                &cfg,
                ps_enc_ctrl,
                &mut ps_enc.s_cmn.indices_lbrr[n_frames_encoded],
                &mut s_nsq_lbrr,
                &mut ps_enc.s_cmn.pulses_lbrr[n_frames_encoded][..total_len],
                &ps_enc.x_buf[x_frame_off..x_frame_off + frame_len],
            );
        }
        ps_enc_ctrl.gains[..nb_subfr].copy_from_slice(&temp_gains[..nb_subfr]);
    }
}
