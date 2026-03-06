//! Codec control and mode switching.
//!
//! Upstream c: `silk/control_codec.c`

use crate::silk::control_audio_bandwidth::silk_control_audio_bandwidth;
use crate::silk::define::{
    LA_SHAPE_MS, MAX_DEL_DEC_STATES, MAX_LPC_ORDER, MAX_NB_SUBFR, MIN_LPC_ORDER,
    SUB_FRAME_LENGTH_MS, TYPE_NO_VOICE_ACTIVITY,
};
use crate::silk::enc_api::silk_EncControlStruct;
use crate::silk::errors::{SILK_ENC_PACKET_SIZE_NOT_SUPPORTED, SILK_NO_ERROR};
use crate::silk::float::sigproc_flp::{silk_float2short_array, silk_short2float_array};
use crate::silk::float::structs_flp::silk_encoder_state_FLP;
use crate::silk::pitch_est_tables::{
    SILK_PE_MAX_COMPLEX, SILK_PE_MID_COMPLEX, SILK_PE_MIN_COMPLEX,
};
use crate::silk::resampler::{silk_resampler, silk_resampler_init, ResamplerState};
use crate::silk::sigproc_fix::{silk_max_int, silk_min_int};
use crate::silk::structs::silk_encoder_state;
use crate::silk::tables_nlsf_cb_nb_mb::SILK_NLSF_CB_NB_MB;
use crate::silk::tables_nlsf_cb_wb::SILK_NLSF_CB_WB;
use crate::silk::tables_other::{SILK_UNIFORM4_ICDF, SILK_UNIFORM6_ICDF, SILK_UNIFORM8_ICDF};
use crate::silk::tables_pitch_lag::{
    SILK_PITCH_CONTOUR_10_MS_ICDF, SILK_PITCH_CONTOUR_10_MS_NB_ICDF, SILK_PITCH_CONTOUR_ICDF,
    SILK_PITCH_CONTOUR_NB_ICDF,
};
use crate::silk::tuning_parameters::WARPING_MULTIPLIER;

/// Upstream c: silk/control_codec.c:silk_control_encoder
pub fn silk_control_encoder(
    ps_enc: &mut silk_encoder_state_FLP,
    enc_control: &mut silk_EncControlStruct,
    allow_bw_switch: i32,
    channel_nb: i32,
    force_fs_k_hz: i32,
) -> i32 {
    let mut fs_k_hz: i32;
    let mut ret: i32 = 0;
    ps_enc.s_cmn.use_dtx = enc_control.use_dtx;
    ps_enc.s_cmn.use_cbr = enc_control.use_cbr;
    ps_enc.s_cmn.api_fs_hz = enc_control.api_sample_rate;
    ps_enc.s_cmn.max_internal_fs_hz = enc_control.max_internal_sample_rate;
    ps_enc.s_cmn.min_internal_fs_hz = enc_control.min_internal_sample_rate;
    ps_enc.s_cmn.desired_internal_fs_hz = enc_control.desired_internal_sample_rate;
    ps_enc.s_cmn.use_in_band_fec = enc_control.use_in_band_fec;
    ps_enc.s_cmn.n_channels_api = enc_control.n_channels_api;
    ps_enc.s_cmn.n_channels_internal = enc_control.n_channels_internal;
    ps_enc.s_cmn.allow_bandwidth_switch = allow_bw_switch;
    ps_enc.s_cmn.channel_nb = channel_nb;
    if ps_enc.s_cmn.controlled_since_last_payload != 0 && ps_enc.s_cmn.prefill_flag == 0 {
        if ps_enc.s_cmn.api_fs_hz != ps_enc.s_cmn.prev_api_fs_hz && ps_enc.s_cmn.fs_k_hz > 0 {
            ret += silk_setup_resamplers(ps_enc, ps_enc.s_cmn.fs_k_hz);
        }
        return ret;
    }
    fs_k_hz = silk_control_audio_bandwidth(&mut ps_enc.s_cmn, enc_control);
    if force_fs_k_hz != 0 {
        fs_k_hz = force_fs_k_hz;
    }
    ret += silk_setup_resamplers(ps_enc, fs_k_hz);
    ret += silk_setup_fs(ps_enc, fs_k_hz, enc_control.payload_size_ms);
    ret += silk_setup_complexity(&mut ps_enc.s_cmn, enc_control.complexity);
    ps_enc.s_cmn.packet_loss_perc = enc_control.packet_loss_percentage;
    ret += silk_setup_lbrr(&mut ps_enc.s_cmn, enc_control);
    ps_enc.s_cmn.controlled_since_last_payload = 1;
    ret
}
/// Upstream c: silk/control_codec.c:silk_setup_resamplers
fn silk_setup_resamplers(ps_enc: &mut silk_encoder_state_FLP, fs_k_hz: i32) -> i32 {
    let mut ret: i32 = SILK_NO_ERROR;
    if ps_enc.s_cmn.fs_k_hz != fs_k_hz || ps_enc.s_cmn.prev_api_fs_hz != ps_enc.s_cmn.api_fs_hz {
        if ps_enc.s_cmn.fs_k_hz == 0 {
            ret += silk_resampler_init(
                &mut ps_enc.s_cmn.resampler_state,
                ps_enc.s_cmn.api_fs_hz,
                fs_k_hz * 1000,
                1,
            );
        } else {
            let buf_length_ms: i32 =
                (((ps_enc.s_cmn.nb_subfr * 5) as u32) << 1) as i32 + LA_SHAPE_MS;
            let old_buf_samples: i32 = buf_length_ms * ps_enc.s_cmn.fs_k_hz;
            let new_buf_samples: i32 = buf_length_ms * fs_k_hz;
            let vla = (if old_buf_samples > new_buf_samples {
                old_buf_samples
            } else {
                new_buf_samples
            }) as usize;
            let mut x_buf_fix: Vec<i16> = ::std::vec::from_elem(0, vla);
            silk_float2short_array(
                &mut x_buf_fix[..old_buf_samples as usize],
                &ps_enc.x_buf[..old_buf_samples as usize],
            );

            /* Initialize resampler for temporary resampling of x_buf data to api_fs_hz */
            let mut temp_resampler_state = ResamplerState::default();
            ret += silk_resampler_init(
                &mut temp_resampler_state,
                ps_enc.s_cmn.fs_k_hz as i16 as i32 * 1000,
                ps_enc.s_cmn.api_fs_hz,
                0,
            );

            /* Calculate number of samples to temporarily upsample */
            let api_buf_samples: i32 = buf_length_ms * (ps_enc.s_cmn.api_fs_hz / 1000);

            /* Temporary resampling of x_buf data to api_fs_hz */
            let vla_0 = api_buf_samples as usize;
            let mut x_buf_api_fs_hz: Vec<i16> = ::std::vec::from_elem(0, vla_0);
            ret += silk_resampler(
                &mut temp_resampler_state,
                &mut x_buf_api_fs_hz,
                &x_buf_fix[..old_buf_samples as usize],
            );
            ret += silk_resampler_init(
                &mut ps_enc.s_cmn.resampler_state,
                ps_enc.s_cmn.api_fs_hz,
                fs_k_hz as i16 as i32 * 1000,
                1,
            );
            ret += silk_resampler(
                &mut ps_enc.s_cmn.resampler_state,
                &mut x_buf_fix,
                &x_buf_api_fs_hz[..api_buf_samples as usize],
            );
            silk_short2float_array(
                &mut ps_enc.x_buf[..new_buf_samples as usize],
                &x_buf_fix[..new_buf_samples as usize],
            );
        }
    }
    ps_enc.s_cmn.prev_api_fs_hz = ps_enc.s_cmn.api_fs_hz;
    ret
}
/// Upstream c: silk/control_codec.c:silk_setup_fs
fn silk_setup_fs(ps_enc: &mut silk_encoder_state_FLP, fs_k_hz: i32, packet_size_ms: i32) -> i32 {
    let mut ret: i32 = SILK_NO_ERROR;
    if packet_size_ms != ps_enc.s_cmn.packet_size_ms {
        if packet_size_ms != 10
            && packet_size_ms != 20
            && packet_size_ms != 40
            && packet_size_ms != 60
        {
            ret = SILK_ENC_PACKET_SIZE_NOT_SUPPORTED;
        }
        if packet_size_ms <= 10 {
            ps_enc.s_cmn.n_frames_per_packet = 1;
            ps_enc.s_cmn.nb_subfr = if packet_size_ms == 10 { 2 } else { 1 };
            ps_enc.s_cmn.frame_length = packet_size_ms as usize * fs_k_hz as usize;
            ps_enc.s_cmn.pitch_lpc_win_length =
                (10 + ((2) << 1)) as i16 as i32 * fs_k_hz as i16 as i32;
            if ps_enc.s_cmn.fs_k_hz == 8 {
                ps_enc.s_cmn.pitch_contour_i_cdf = &SILK_PITCH_CONTOUR_10_MS_NB_ICDF;
            } else {
                ps_enc.s_cmn.pitch_contour_i_cdf = &SILK_PITCH_CONTOUR_10_MS_ICDF;
            }
        } else {
            ps_enc.s_cmn.n_frames_per_packet = packet_size_ms / (5 * 4);
            ps_enc.s_cmn.nb_subfr = MAX_NB_SUBFR;
            ps_enc.s_cmn.frame_length = 20 * fs_k_hz as usize;
            ps_enc.s_cmn.pitch_lpc_win_length =
                (20 + ((2) << 1)) as i16 as i32 * fs_k_hz as i16 as i32;
            if ps_enc.s_cmn.fs_k_hz == 8 {
                ps_enc.s_cmn.pitch_contour_i_cdf = &SILK_PITCH_CONTOUR_NB_ICDF;
            } else {
                ps_enc.s_cmn.pitch_contour_i_cdf = &SILK_PITCH_CONTOUR_ICDF;
            }
        }
        ps_enc.s_cmn.packet_size_ms = packet_size_ms;
        ps_enc.s_cmn.target_rate_bps = 0;
    }
    debug_assert!(fs_k_hz == 8 || fs_k_hz == 12 || fs_k_hz == 16);
    debug_assert!(ps_enc.s_cmn.nb_subfr == 2 || ps_enc.s_cmn.nb_subfr == 4);
    if ps_enc.s_cmn.fs_k_hz != fs_k_hz {
        ps_enc.s_shape = Default::default();
        ps_enc.s_cmn.s_nsq = Default::default();
        ps_enc.s_cmn.prev_nlsfq_q15.fill(0);
        ps_enc.s_cmn.s_lp.in_lp_state.fill(0);
        ps_enc.s_cmn.input_buf_ix = 0;
        ps_enc.s_cmn.n_frames_encoded = 0;
        ps_enc.s_cmn.target_rate_bps = 0;
        ps_enc.s_cmn.prev_lag = 100;
        ps_enc.s_cmn.first_frame_after_reset = 1;
        ps_enc.s_shape.last_gain_index = 10;
        ps_enc.s_cmn.s_nsq.lag_prev = 100;
        ps_enc.s_cmn.s_nsq.prev_gain_q16 = 65536;
        ps_enc.s_cmn.prev_signal_type = TYPE_NO_VOICE_ACTIVITY as i8;
        ps_enc.s_cmn.fs_k_hz = fs_k_hz;
        if ps_enc.s_cmn.fs_k_hz == 8 {
            if ps_enc.s_cmn.nb_subfr == MAX_NB_SUBFR {
                ps_enc.s_cmn.pitch_contour_i_cdf = &SILK_PITCH_CONTOUR_NB_ICDF;
            } else {
                ps_enc.s_cmn.pitch_contour_i_cdf = &SILK_PITCH_CONTOUR_10_MS_NB_ICDF;
            }
        } else if ps_enc.s_cmn.nb_subfr == MAX_NB_SUBFR {
            ps_enc.s_cmn.pitch_contour_i_cdf = &SILK_PITCH_CONTOUR_ICDF;
        } else {
            ps_enc.s_cmn.pitch_contour_i_cdf = &SILK_PITCH_CONTOUR_10_MS_ICDF;
        }
        if ps_enc.s_cmn.fs_k_hz == 8 || ps_enc.s_cmn.fs_k_hz == 12 {
            ps_enc.s_cmn.predict_lpcorder = MIN_LPC_ORDER as i32;
            ps_enc.s_cmn.ps_nlsf_cb = &SILK_NLSF_CB_NB_MB;
        } else {
            ps_enc.s_cmn.predict_lpcorder = MAX_LPC_ORDER as i32;
            ps_enc.s_cmn.ps_nlsf_cb = &SILK_NLSF_CB_WB;
        }
        ps_enc.s_cmn.subfr_length = SUB_FRAME_LENGTH_MS * fs_k_hz as usize;
        ps_enc.s_cmn.frame_length = ps_enc.s_cmn.subfr_length * ps_enc.s_cmn.nb_subfr;
        ps_enc.s_cmn.ltp_mem_length = 20 * fs_k_hz as usize;
        ps_enc.s_cmn.la_pitch = 2 * fs_k_hz as i16 as i32;
        ps_enc.s_cmn.max_pitch_lag = 18 * fs_k_hz as i16 as i32;
        if ps_enc.s_cmn.nb_subfr == MAX_NB_SUBFR {
            ps_enc.s_cmn.pitch_lpc_win_length =
                (20 + ((2) << 1)) as i16 as i32 * fs_k_hz as i16 as i32;
        } else {
            ps_enc.s_cmn.pitch_lpc_win_length =
                (10 + ((2) << 1)) as i16 as i32 * fs_k_hz as i16 as i32;
        }
        if ps_enc.s_cmn.fs_k_hz == 16 {
            ps_enc.s_cmn.pitch_lag_low_bits_i_cdf = &SILK_UNIFORM8_ICDF;
        } else if ps_enc.s_cmn.fs_k_hz == 12 {
            ps_enc.s_cmn.pitch_lag_low_bits_i_cdf = &SILK_UNIFORM6_ICDF;
        } else {
            ps_enc.s_cmn.pitch_lag_low_bits_i_cdf = &SILK_UNIFORM4_ICDF;
        }
    }
    debug_assert!(ps_enc.s_cmn.subfr_length * ps_enc.s_cmn.nb_subfr == ps_enc.s_cmn.frame_length);
    ret
}
/// Upstream c: silk/control_codec.c:silk_setup_complexity
fn silk_setup_complexity(ps_enc_c: &mut silk_encoder_state, complexity: i32) -> i32 {
    let ret: i32 = 0;
    debug_assert!((0..=10).contains(&complexity));
    if complexity < 1 {
        ps_enc_c.pitch_estimation_complexity = SILK_PE_MIN_COMPLEX;
        ps_enc_c.pitch_estimation_threshold_q16 = (0.8f64 * ((1) << 16) as f64 + 0.5f64) as i32;
        ps_enc_c.pitch_estimation_lpcorder = 6;
        ps_enc_c.shaping_lpcorder = 12;
        ps_enc_c.la_shape = 3 * ps_enc_c.fs_k_hz;
        ps_enc_c.n_states_delayed_decision = 1;
        ps_enc_c.use_interpolated_nlsfs = 0;
        ps_enc_c.nlsf_msvq_survivors = 2;
        ps_enc_c.warping_q16 = 0;
    } else if complexity < 2 {
        ps_enc_c.pitch_estimation_complexity = SILK_PE_MID_COMPLEX;
        ps_enc_c.pitch_estimation_threshold_q16 = (0.76f64 * ((1) << 16) as f64 + 0.5f64) as i32;
        ps_enc_c.pitch_estimation_lpcorder = 8;
        ps_enc_c.shaping_lpcorder = 14;
        ps_enc_c.la_shape = 5 * ps_enc_c.fs_k_hz;
        ps_enc_c.n_states_delayed_decision = 1;
        ps_enc_c.use_interpolated_nlsfs = 0;
        ps_enc_c.nlsf_msvq_survivors = 3;
        ps_enc_c.warping_q16 = 0;
    } else if complexity < 3 {
        ps_enc_c.pitch_estimation_complexity = SILK_PE_MIN_COMPLEX;
        ps_enc_c.pitch_estimation_threshold_q16 = (0.8f64 * ((1) << 16) as f64 + 0.5f64) as i32;
        ps_enc_c.pitch_estimation_lpcorder = 6;
        ps_enc_c.shaping_lpcorder = 12;
        ps_enc_c.la_shape = 3 * ps_enc_c.fs_k_hz;
        ps_enc_c.n_states_delayed_decision = 2;
        ps_enc_c.use_interpolated_nlsfs = 0;
        ps_enc_c.nlsf_msvq_survivors = 2;
        ps_enc_c.warping_q16 = 0;
    } else if complexity < 4 {
        ps_enc_c.pitch_estimation_complexity = SILK_PE_MID_COMPLEX;
        ps_enc_c.pitch_estimation_threshold_q16 = (0.76f64 * ((1) << 16) as f64 + 0.5f64) as i32;
        ps_enc_c.pitch_estimation_lpcorder = 8;
        ps_enc_c.shaping_lpcorder = 14;
        ps_enc_c.la_shape = 5 * ps_enc_c.fs_k_hz;
        ps_enc_c.n_states_delayed_decision = 2;
        ps_enc_c.use_interpolated_nlsfs = 0;
        ps_enc_c.nlsf_msvq_survivors = 4;
        ps_enc_c.warping_q16 = 0;
    } else if complexity < 6 {
        ps_enc_c.pitch_estimation_complexity = SILK_PE_MID_COMPLEX;
        ps_enc_c.pitch_estimation_threshold_q16 = (0.74f64 * ((1) << 16) as f64 + 0.5f64) as i32;
        ps_enc_c.pitch_estimation_lpcorder = 10;
        ps_enc_c.shaping_lpcorder = 16;
        ps_enc_c.la_shape = 5 * ps_enc_c.fs_k_hz;
        ps_enc_c.n_states_delayed_decision = 2;
        ps_enc_c.use_interpolated_nlsfs = 1;
        ps_enc_c.nlsf_msvq_survivors = 6;
        ps_enc_c.warping_q16 =
            ps_enc_c.fs_k_hz * ((WARPING_MULTIPLIER * ((1) << 16) as f32) as f64 + 0.5f64) as i32;
    } else if complexity < 8 {
        ps_enc_c.pitch_estimation_complexity = SILK_PE_MID_COMPLEX;
        ps_enc_c.pitch_estimation_threshold_q16 = (0.72f64 * ((1) << 16) as f64 + 0.5f64) as i32;
        ps_enc_c.pitch_estimation_lpcorder = 12;
        ps_enc_c.shaping_lpcorder = 20;
        ps_enc_c.la_shape = 5 * ps_enc_c.fs_k_hz;
        ps_enc_c.n_states_delayed_decision = 3;
        ps_enc_c.use_interpolated_nlsfs = 1;
        ps_enc_c.nlsf_msvq_survivors = 8;
        ps_enc_c.warping_q16 =
            ps_enc_c.fs_k_hz * ((WARPING_MULTIPLIER * ((1) << 16) as f32) as f64 + 0.5f64) as i32;
    } else {
        ps_enc_c.pitch_estimation_complexity = SILK_PE_MAX_COMPLEX as i32;
        ps_enc_c.pitch_estimation_threshold_q16 = (0.7f64 * ((1) << 16) as f64 + 0.5f64) as i32;
        ps_enc_c.pitch_estimation_lpcorder = 16;
        ps_enc_c.shaping_lpcorder = 24;
        ps_enc_c.la_shape = 5 * ps_enc_c.fs_k_hz;
        ps_enc_c.n_states_delayed_decision = MAX_DEL_DEC_STATES;
        ps_enc_c.use_interpolated_nlsfs = 1;
        ps_enc_c.nlsf_msvq_survivors = 16;
        ps_enc_c.warping_q16 =
            ps_enc_c.fs_k_hz * ((WARPING_MULTIPLIER * ((1) << 16) as f32) as f64 + 0.5f64) as i32;
    }
    ps_enc_c.pitch_estimation_lpcorder = silk_min_int(
        ps_enc_c.pitch_estimation_lpcorder,
        ps_enc_c.predict_lpcorder,
    );
    ps_enc_c.shape_win_length =
        SUB_FRAME_LENGTH_MS as i32 * ps_enc_c.fs_k_hz + 2 * ps_enc_c.la_shape;
    ps_enc_c.complexity = complexity;
    debug_assert!(ps_enc_c.pitch_estimation_lpcorder <= 16);
    debug_assert!(ps_enc_c.shaping_lpcorder <= 24);
    debug_assert!(ps_enc_c.n_states_delayed_decision <= 4);
    debug_assert!(ps_enc_c.warping_q16 <= 32767);
    debug_assert!(ps_enc_c.la_shape <= 5 * 16);
    debug_assert!(ps_enc_c.shape_win_length <= 15 * 16);
    ret
}
/// Upstream c: silk/control_codec.c:silk_setup_LBRR
#[inline]
fn silk_setup_lbrr(ps_enc_c: &mut silk_encoder_state, enc_control: &silk_EncControlStruct) -> i32 {
    let ret: i32 = SILK_NO_ERROR;
    let lbrr_in_previous_packet: i32 = ps_enc_c.lbrr_enabled;
    ps_enc_c.lbrr_enabled = enc_control.lbrr_coded;
    if ps_enc_c.lbrr_enabled != 0 {
        if lbrr_in_previous_packet == 0 {
            ps_enc_c.lbrr_gain_increases = 7;
        } else {
            ps_enc_c.lbrr_gain_increases = silk_max_int(
                7 - ((ps_enc_c.packet_loss_perc as i64
                    * (0.2f64 * ((1) << 16) as f64 + 0.5f64) as i32 as i16 as i64)
                    >> 16) as i32,
                3,
            );
        }
    }
    ret
}
