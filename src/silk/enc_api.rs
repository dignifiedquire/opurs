//! SILK encoder API.
//!
//! Upstream c: `silk/enc_API.c`

#[derive(Copy, Clone, Default)]
#[repr(C)]
pub struct silk_EncControlStruct {
    pub n_channels_api: i32,
    pub n_channels_internal: i32,
    pub api_sample_rate: i32,
    pub max_internal_sample_rate: i32,
    pub min_internal_sample_rate: i32,
    pub desired_internal_sample_rate: i32,
    pub payload_size_ms: i32,
    pub bit_rate: i32,
    pub packet_loss_percentage: i32,
    pub complexity: i32,
    pub use_in_band_fec: i32,
    pub use_dred: i32,
    pub lbrr_coded: i32,
    pub use_dtx: i32,
    pub use_cbr: i32,
    pub max_bits: i32,
    pub to_mono: i32,
    pub opus_can_switch: i32,
    pub reduced_dependency: i32,
    pub internal_sample_rate: i32,
    pub allow_bandwidth_switch: i32,
    pub in_wbmode_without_variable_lp: i32,
    pub stereo_width_q14: i32,
    pub switch_ready: i32,
    pub signal_type: i32,
    pub offset: i32,
}
use crate::arch::Arch;
use crate::celt::entcode::ec_tell;
use crate::celt::entenc::{ec_enc_icdf, ec_enc_patch_initial_bits, EcEnc};
use crate::celt::float_cast::float2int16;
use crate::silk::errors::{SILK_ENC_INPUT_INVALID_NO_OF_SAMPLES, SILK_NO_ERROR};

use crate::silk::check_control_input::check_control_input;
use crate::silk::control_codec::silk_control_encoder;
use crate::silk::control_snr::silk_control_snr;
use crate::silk::define::{
    CODE_CONDITIONALLY, CODE_INDEPENDENTLY, CODE_INDEPENDENTLY_NO_LTP_SCALING,
    ENCODER_NUM_CHANNELS, TYPE_NO_VOICE_ACTIVITY,
};
use crate::silk::encode_indices::silk_encode_indices;
use crate::silk::encode_pulses::silk_encode_pulses;
use crate::silk::float::encode_frame_flp::{silk_encode_do_vad_flp, silk_encode_frame_flp};
use crate::silk::float::structs_flp::{silk_encoder, silk_shape_state_FLP};
use crate::silk::init_encoder::silk_init_encoder;
use crate::silk::resampler::silk_resampler;

use crate::silk::hp_variable_cutoff::silk_hp_variable_cutoff;
use crate::silk::stereo_encode_pred::{silk_stereo_encode_mid_only, silk_stereo_encode_pred};
use crate::silk::stereo_lr_to_ms::silk_stereo_lr_to_ms;
use crate::silk::structs::{silk_LP_state, silk_nsq_state};
use crate::silk::tables_other::{SILK_LBRR_FLAGS_ICDF_PTR, SILK_QUANTIZATION_OFFSETS_Q10};
use crate::silk::tuning_parameters::{
    BITRESERVOIR_DECAY_TIME_MS, MAX_BANDWIDTH_SWITCH_DELAY_MS, SPEECH_ACTIVITY_DTX_THRES,
};

/// Upstream c: silk/enc_API.c:silk_InitEncoder
pub fn silk_init_encoder_api(
    ps_enc: &mut silk_encoder,
    arch: Arch,
    enc_status: &mut silk_EncControlStruct,
) -> i32 {
    // Zero-init the encoder state
    *ps_enc = silk_encoder::default();
    let mut ret: i32 = SILK_NO_ERROR;
    for n in 0..ENCODER_NUM_CHANNELS as usize {
        ret += silk_init_encoder(&mut ps_enc.state_fxx[n], arch);
        if ret != 0 {
            return ret;
        }
    }
    ps_enc.n_channels_api = 1;
    ps_enc.n_channels_internal = 1;
    ret += silk_query_encoder(ps_enc, enc_status);
    if ret != 0 {
        return ret;
    }
    ret
}
/// Upstream c: silk/enc_API.c:silk_QueryEncoder
fn silk_query_encoder(ps_enc: &silk_encoder, enc_status: &mut silk_EncControlStruct) -> i32 {
    let state = &ps_enc.state_fxx[0];
    enc_status.n_channels_api = ps_enc.n_channels_api;
    enc_status.n_channels_internal = ps_enc.n_channels_internal;
    enc_status.api_sample_rate = state.s_cmn.api_fs_hz;
    enc_status.max_internal_sample_rate = state.s_cmn.max_internal_fs_hz;
    enc_status.min_internal_sample_rate = state.s_cmn.min_internal_fs_hz;
    enc_status.desired_internal_sample_rate = state.s_cmn.desired_internal_fs_hz;
    enc_status.payload_size_ms = state.s_cmn.packet_size_ms;
    enc_status.bit_rate = state.s_cmn.target_rate_bps;
    enc_status.packet_loss_percentage = state.s_cmn.packet_loss_perc;
    enc_status.complexity = state.s_cmn.complexity;
    enc_status.use_in_band_fec = state.s_cmn.use_in_band_fec;
    enc_status.use_dtx = state.s_cmn.use_dtx;
    enc_status.use_cbr = state.s_cmn.use_cbr;
    enc_status.internal_sample_rate = state.s_cmn.fs_k_hz as i16 as i32 * 1000;
    enc_status.allow_bandwidth_switch = state.s_cmn.allow_bandwidth_switch;
    enc_status.in_wbmode_without_variable_lp =
        (state.s_cmn.fs_k_hz == 16 && state.s_cmn.s_lp.mode == 0) as i32;
    SILK_NO_ERROR
}
/// Upstream c: silk/enc_API.c:silk_Encode
pub fn silk_encode_api(
    ps_enc: &mut silk_encoder,
    enc_control: &mut silk_EncControlStruct,
    samples_in: &[f32],
    n_samples_in: i32,
    mut ps_range_enc: Option<&mut EcEnc>,
    n_bytes_out: &mut i32,
    prefill_flag: i32,
    activity: i32,
) -> i32 {
    let mut n: i32;
    let mut _i: i32;
    let mut n_bits: i32;
    let mut flags: i32;
    let mut tmp_payload_size_ms: i32 = 0;
    let mut tmp_complexity: i32 = 0;
    let mut ret: i32;
    let mut n_samples_to_buffer: i32;
    let mut n_samples_from_input: i32;
    let mut target_rate_bps: i32;
    let mut mstarget_rates_bps: [i32; 2] = [0; 2];
    let mut channel_rate_bps: i32;
    let mut lbrr_symbol: i32;
    let mut samples_in_off: usize = 0;
    let mut n_samples_in = n_samples_in;

    debug_assert!(
        enc_control.n_channels_api >= enc_control.n_channels_internal
            && enc_control.n_channels_api >= ps_enc.n_channels_internal
    );
    if enc_control.reduced_dependency != 0 {
        n = 0;
        while n < enc_control.n_channels_api {
            ps_enc.state_fxx[n as usize].s_cmn.first_frame_after_reset = 1;
            n += 1;
        }
    }
    n = 0;
    while n < enc_control.n_channels_api {
        ps_enc.state_fxx[n as usize].s_cmn.n_frames_encoded = 0;
        n += 1;
    }
    ret = check_control_input(enc_control);
    if ret != 0 {
        return ret;
    }
    enc_control.switch_ready = 0;
    if enc_control.n_channels_internal > ps_enc.n_channels_internal {
        let arch = ps_enc.state_fxx[0].s_cmn.arch;
        ret += silk_init_encoder(&mut ps_enc.state_fxx[1], arch);
        ps_enc.s_stereo.pred_prev_q13 = [0; 2];
        ps_enc.s_stereo.s_side = [0; 2];
        ps_enc.s_stereo.mid_side_amp_q0[0] = 0;
        ps_enc.s_stereo.mid_side_amp_q0[1] = 1;
        ps_enc.s_stereo.mid_side_amp_q0[2] = 0;
        ps_enc.s_stereo.mid_side_amp_q0[3] = 1;
        ps_enc.s_stereo.width_prev_q14 = 0;
        ps_enc.s_stereo.smth_width_q14 = ((1 << 14) as f64 + 0.5f64) as i32 as i16;
        if ps_enc.n_channels_api == 2 {
            ps_enc.state_fxx[1].s_cmn.resampler_state = ps_enc.state_fxx[0].s_cmn.resampler_state;
            ps_enc.state_fxx[1].s_cmn.in_hp_state = ps_enc.state_fxx[0].s_cmn.in_hp_state;
        }
    }
    let transition = (enc_control.payload_size_ms != ps_enc.state_fxx[0].s_cmn.packet_size_ms
        || ps_enc.n_channels_internal != enc_control.n_channels_internal)
        as i32;
    ps_enc.n_channels_api = enc_control.n_channels_api;
    ps_enc.n_channels_internal = enc_control.n_channels_internal;
    let n_blocks_of10ms = 100 * n_samples_in / enc_control.api_sample_rate;
    let tot_blocks = if n_blocks_of10ms > 1 {
        n_blocks_of10ms >> 1
    } else {
        1
    };
    let mut curr_block: i32 = 0;
    if prefill_flag != 0 {
        let mut save_lp = silk_LP_state {
            in_lp_state: [0; 2],
            transition_frame_no: 0,
            mode: 0,
            saved_fs_k_hz: 0,
        };
        if n_blocks_of10ms != 1 {
            return SILK_ENC_INPUT_INVALID_NO_OF_SAMPLES;
        }
        if prefill_flag == 2 {
            save_lp = ps_enc.state_fxx[0].s_cmn.s_lp;
            save_lp.saved_fs_k_hz = ps_enc.state_fxx[0].s_cmn.fs_k_hz;
        }
        n = 0;
        while n < enc_control.n_channels_internal {
            let arch = ps_enc.state_fxx[n as usize].s_cmn.arch;
            ret = silk_init_encoder(&mut ps_enc.state_fxx[n as usize], arch);
            if prefill_flag == 2 {
                ps_enc.state_fxx[n as usize].s_cmn.s_lp = save_lp;
            }
            debug_assert_eq!(ret, 0);
            n += 1;
        }
        tmp_payload_size_ms = enc_control.payload_size_ms;
        enc_control.payload_size_ms = 10;
        tmp_complexity = enc_control.complexity;
        enc_control.complexity = 0;
        n = 0;
        while n < enc_control.n_channels_internal {
            ps_enc.state_fxx[n as usize]
                .s_cmn
                .controlled_since_last_payload = 0;
            ps_enc.state_fxx[n as usize].s_cmn.prefill_flag = 1;
            n += 1;
        }
    } else {
        if n_blocks_of10ms * enc_control.api_sample_rate != 100 * n_samples_in || n_samples_in < 0 {
            return SILK_ENC_INPUT_INVALID_NO_OF_SAMPLES;
        }
        if 1000 * n_samples_in > enc_control.payload_size_ms * enc_control.api_sample_rate {
            return SILK_ENC_INPUT_INVALID_NO_OF_SAMPLES;
        }
    }
    n = 0;
    while n < enc_control.n_channels_internal {
        let force_fs_k_hz: i32 = if n == 1 {
            ps_enc.state_fxx[0].s_cmn.fs_k_hz
        } else {
            0
        };
        ret = silk_control_encoder(
            &mut ps_enc.state_fxx[n as usize],
            enc_control,
            ps_enc.allow_bandwidth_switch,
            n,
            force_fs_k_hz,
        );
        if ret != 0 {
            return ret;
        }
        if ps_enc.state_fxx[n as usize].s_cmn.first_frame_after_reset != 0 || transition != 0 {
            _i = 0;
            while _i < ps_enc.state_fxx[0].s_cmn.n_frames_per_packet {
                ps_enc.state_fxx[n as usize].s_cmn.lbrr_flags[_i as usize] = 0;
                _i += 1;
            }
        }
        ps_enc.state_fxx[n as usize].s_cmn.in_dtx = ps_enc.state_fxx[n as usize].s_cmn.use_dtx;
        n += 1;
    }
    debug_assert!(
        enc_control.n_channels_internal == 1
            || ps_enc.state_fxx[0].s_cmn.fs_k_hz == ps_enc.state_fxx[1].s_cmn.fs_k_hz
    );
    let n_samples_to_buffer_max = 10 * n_blocks_of10ms * ps_enc.state_fxx[0].s_cmn.fs_k_hz;
    let n_samples_from_input_max = n_samples_to_buffer_max * ps_enc.state_fxx[0].s_cmn.api_fs_hz
        / (ps_enc.state_fxx[0].s_cmn.fs_k_hz * 1000);
    // n_samples_from_input_max max: 10 * 6 * 16 * 48000 / (16 * 1000) = 2880
    const MAX_BUF: usize = 2880;
    debug_assert!((n_samples_from_input_max as usize) <= MAX_BUF);
    let mut buf = [0i16; MAX_BUF];
    loop {
        let mut curr_n_bits_used_lbrr: i32 = 0;
        n_samples_to_buffer =
            ps_enc.state_fxx[0].s_cmn.frame_length as i32 - ps_enc.state_fxx[0].s_cmn.input_buf_ix;
        n_samples_to_buffer = n_samples_to_buffer.min(n_samples_to_buffer_max);
        n_samples_from_input = n_samples_to_buffer * ps_enc.state_fxx[0].s_cmn.api_fs_hz
            / (ps_enc.state_fxx[0].s_cmn.fs_k_hz * 1000);
        if enc_control.n_channels_api == 2 && enc_control.n_channels_internal == 2 {
            let id = ps_enc.state_fxx[0].s_cmn.n_frames_encoded;
            // De-interleave left channel
            for k in 0..n_samples_from_input as usize {
                buf[k] = float2int16(samples_in[samples_in_off + 2 * k]);
            }
            // Making sure to start both resamplers from the same state when switching from mono to stereo
            if ps_enc.n_prev_channels_internal == 1 && id == 0 {
                ps_enc.state_fxx[1].s_cmn.resampler_state =
                    ps_enc.state_fxx[0].s_cmn.resampler_state;
            }
            {
                let ix0 = ps_enc.state_fxx[0].s_cmn.input_buf_ix as usize;
                let [s0, _] = &mut ps_enc.state_fxx;
                ret += silk_resampler(
                    &mut s0.s_cmn.resampler_state,
                    &mut s0.s_cmn.input_buf[ix0 + 2..ix0 + 2 + n_samples_to_buffer as usize],
                    &buf[..n_samples_from_input as usize],
                );
            }
            ps_enc.state_fxx[0].s_cmn.input_buf_ix += n_samples_to_buffer;

            n_samples_to_buffer = ps_enc.state_fxx[1].s_cmn.frame_length as i32
                - ps_enc.state_fxx[1].s_cmn.input_buf_ix;
            n_samples_to_buffer =
                n_samples_to_buffer.min(10 * n_blocks_of10ms * ps_enc.state_fxx[1].s_cmn.fs_k_hz);
            // De-interleave right channel
            for k in 0..n_samples_from_input as usize {
                buf[k] = float2int16(samples_in[samples_in_off + 2 * k + 1]);
            }
            {
                let ix1 = ps_enc.state_fxx[1].s_cmn.input_buf_ix as usize;
                let [_, s1] = &mut ps_enc.state_fxx;
                ret += silk_resampler(
                    &mut s1.s_cmn.resampler_state,
                    &mut s1.s_cmn.input_buf[ix1 + 2..ix1 + 2 + n_samples_to_buffer as usize],
                    &buf[..n_samples_from_input as usize],
                );
            }
            ps_enc.state_fxx[1].s_cmn.input_buf_ix += n_samples_to_buffer;
        } else if enc_control.n_channels_api == 2 && enc_control.n_channels_internal == 1 {
            // Downmix stereo to mono
            for k in 0..n_samples_from_input as usize {
                let sum = float2int16(
                    samples_in[samples_in_off + 2 * k] + samples_in[samples_in_off + 2 * k + 1],
                ) as i32;
                buf[k] = ((sum >> 1) + (sum & 1)) as i16;
            }
            {
                let ix0 = ps_enc.state_fxx[0].s_cmn.input_buf_ix as usize;
                let [s0, _] = &mut ps_enc.state_fxx;
                ret += silk_resampler(
                    &mut s0.s_cmn.resampler_state,
                    &mut s0.s_cmn.input_buf[ix0 + 2..ix0 + 2 + n_samples_to_buffer as usize],
                    &buf[..n_samples_from_input as usize],
                );
            }
            if ps_enc.n_prev_channels_internal == 2
                && ps_enc.state_fxx[0].s_cmn.n_frames_encoded == 0
            {
                {
                    let ix1 = ps_enc.state_fxx[1].s_cmn.input_buf_ix as usize;
                    let [_, s1] = &mut ps_enc.state_fxx;
                    ret += silk_resampler(
                        &mut s1.s_cmn.resampler_state,
                        &mut s1.s_cmn.input_buf[ix1 + 2..],
                        &buf[..n_samples_from_input as usize],
                    );
                }
                let frame_len = ps_enc.state_fxx[0].s_cmn.frame_length as i32;
                let ix0 = ps_enc.state_fxx[0].s_cmn.input_buf_ix;
                let ix1 = ps_enc.state_fxx[1].s_cmn.input_buf_ix;
                for k in 0..frame_len {
                    let idx0 = (ix0 + k + 2) as usize;
                    let idx1 = (ix1 + k + 2) as usize;
                    ps_enc.state_fxx[0].s_cmn.input_buf[idx0] =
                        ((ps_enc.state_fxx[0].s_cmn.input_buf[idx0] as i32
                            + ps_enc.state_fxx[1].s_cmn.input_buf[idx1] as i32)
                            >> 1) as i16;
                }
            }
            ps_enc.state_fxx[0].s_cmn.input_buf_ix += n_samples_to_buffer;
        } else {
            debug_assert!(enc_control.n_channels_api == 1 && enc_control.n_channels_internal == 1);
            for k in 0..n_samples_from_input as usize {
                buf[k] = float2int16(samples_in[samples_in_off + k]);
            }
            {
                let ix0 = ps_enc.state_fxx[0].s_cmn.input_buf_ix as usize;
                let [s0, _] = &mut ps_enc.state_fxx;
                ret += silk_resampler(
                    &mut s0.s_cmn.resampler_state,
                    &mut s0.s_cmn.input_buf[ix0 + 2..ix0 + 2 + n_samples_to_buffer as usize],
                    &buf[..n_samples_from_input as usize],
                );
            }
            ps_enc.state_fxx[0].s_cmn.input_buf_ix += n_samples_to_buffer;
        }
        samples_in_off += (n_samples_from_input * enc_control.n_channels_api) as usize;
        n_samples_in -= n_samples_from_input;
        ps_enc.allow_bandwidth_switch = 0;
        if ps_enc.state_fxx[0].s_cmn.input_buf_ix < ps_enc.state_fxx[0].s_cmn.frame_length as i32 {
            break;
        }
        debug_assert_eq!(
            ps_enc.state_fxx[0].s_cmn.input_buf_ix,
            ps_enc.state_fxx[0].s_cmn.frame_length as i32
        );
        debug_assert!(
            enc_control.n_channels_internal == 1
                || ps_enc.state_fxx[1].s_cmn.input_buf_ix
                    == ps_enc.state_fxx[1].s_cmn.frame_length as i32
        );
        if ps_enc.state_fxx[0].s_cmn.n_frames_encoded == 0 && prefill_flag == 0 {
            let ps_range_enc = &mut **ps_range_enc.as_mut().unwrap();

            let mut i_cdf: [u8; 2] = [0, 0];
            i_cdf[0] = (256
                - (256
                    >> ((ps_enc.state_fxx[0].s_cmn.n_frames_per_packet + 1)
                        * enc_control.n_channels_internal))) as u8;
            ec_enc_icdf(ps_range_enc, 0, &i_cdf, 8);
            curr_n_bits_used_lbrr = ec_tell(ps_range_enc);
            n = 0;
            while n < enc_control.n_channels_internal {
                lbrr_symbol = 0;
                _i = 0;
                while _i < ps_enc.state_fxx[n as usize].s_cmn.n_frames_per_packet {
                    lbrr_symbol |= ((ps_enc.state_fxx[n as usize].s_cmn.lbrr_flags[_i as usize]
                        as u32)
                        << _i) as i32;
                    _i += 1;
                }
                ps_enc.state_fxx[n as usize].s_cmn.lbrr_flag =
                    (if lbrr_symbol > 0 { 1 } else { 0 }) as i8;
                if lbrr_symbol != 0 && ps_enc.state_fxx[n as usize].s_cmn.n_frames_per_packet > 1 {
                    ec_enc_icdf(
                        ps_range_enc,
                        lbrr_symbol - 1,
                        SILK_LBRR_FLAGS_ICDF_PTR
                            [(ps_enc.state_fxx[n as usize].s_cmn.n_frames_per_packet - 2) as usize],
                        8,
                    );
                }
                n += 1;
            }
            _i = 0;
            while _i < ps_enc.state_fxx[0].s_cmn.n_frames_per_packet {
                n = 0;
                while n < enc_control.n_channels_internal {
                    if ps_enc.state_fxx[n as usize].s_cmn.lbrr_flags[_i as usize] != 0 {
                        if enc_control.n_channels_internal == 2 && n == 0 {
                            silk_stereo_encode_pred(
                                ps_range_enc,
                                &ps_enc.s_stereo.pred_ix[_i as usize],
                            );
                            if ps_enc.state_fxx[1].s_cmn.lbrr_flags[_i as usize] == 0 {
                                silk_stereo_encode_mid_only(
                                    ps_range_enc,
                                    ps_enc.s_stereo.mid_only_flags[_i as usize],
                                );
                            }
                        }
                        let cond_coding = if _i > 0
                            && ps_enc.state_fxx[n as usize].s_cmn.lbrr_flags[(_i - 1) as usize] != 0
                        {
                            CODE_CONDITIONALLY
                        } else {
                            CODE_INDEPENDENTLY
                        };
                        silk_encode_indices(
                            &mut ps_enc.state_fxx[n as usize].s_cmn,
                            ps_range_enc,
                            _i,
                            1,
                            cond_coding,
                        );
                        silk_encode_pulses(
                            ps_range_enc,
                            ps_enc.state_fxx[n as usize].s_cmn.indices_lbrr[_i as usize].signal_type
                                as i32,
                            ps_enc.state_fxx[n as usize].s_cmn.indices_lbrr[_i as usize]
                                .quant_offset_type as i32,
                            &mut ps_enc.state_fxx[n as usize].s_cmn.pulses_lbrr[_i as usize],
                            ps_enc.state_fxx[n as usize].s_cmn.frame_length,
                        );
                    }
                    n += 1;
                }
                _i += 1;
            }
            n = 0;
            while n < enc_control.n_channels_internal {
                ps_enc.state_fxx[n as usize].s_cmn.lbrr_flags = [0; 3];
                n += 1;
            }
            curr_n_bits_used_lbrr = ec_tell(ps_range_enc) - curr_n_bits_used_lbrr;
        }
        silk_hp_variable_cutoff(&mut ps_enc.state_fxx);
        n_bits = enc_control.bit_rate * enc_control.payload_size_ms / 1000;
        if prefill_flag == 0 {
            // ps_enc.n_bits_used_lbrr is an exponential moving average of the LBRR usage,
            // except that for the first LBRR frame it does no averaging and for the first
            // frame after LBRR, it goes back to zero immediately.
            if curr_n_bits_used_lbrr < 10 {
                ps_enc.n_bits_used_lbrr = 0;
            } else if ps_enc.n_bits_used_lbrr < 10 {
                ps_enc.n_bits_used_lbrr = curr_n_bits_used_lbrr;
            } else {
                ps_enc.n_bits_used_lbrr = (ps_enc.n_bits_used_lbrr + curr_n_bits_used_lbrr) / 2;
            }
            n_bits -= ps_enc.n_bits_used_lbrr;
        }
        n_bits /= ps_enc.state_fxx[0].s_cmn.n_frames_per_packet;
        if enc_control.payload_size_ms == 10 {
            target_rate_bps = n_bits as i16 as i32 * 100;
        } else {
            target_rate_bps = n_bits as i16 as i32 * 50;
        }
        target_rate_bps -= ps_enc.n_bits_exceeded * 1000 / BITRESERVOIR_DECAY_TIME_MS;
        if prefill_flag == 0 && ps_enc.state_fxx[0].s_cmn.n_frames_encoded > 0 {
            let bits_balance = ec_tell(ps_range_enc.as_mut().unwrap())
                - ps_enc.n_bits_used_lbrr
                - n_bits * ps_enc.state_fxx[0].s_cmn.n_frames_encoded;
            target_rate_bps -= bits_balance * 1000 / BITRESERVOIR_DECAY_TIME_MS;
        }
        target_rate_bps = if enc_control.bit_rate > 5000 {
            target_rate_bps.clamp(5000, enc_control.bit_rate)
        } else {
            target_rate_bps.clamp(enc_control.bit_rate, 5000)
        };
        if enc_control.n_channels_internal == 2 {
            {
                let frame_length = ps_enc.state_fxx[0].s_cmn.frame_length;
                let nfe = ps_enc.state_fxx[0].s_cmn.n_frames_encoded as usize;
                let speech_activity = ps_enc.state_fxx[0].s_cmn.speech_activity_q8;
                let fs_k_hz = ps_enc.state_fxx[0].s_cmn.fs_k_hz;
                let frame_len_i32 = ps_enc.state_fxx[0].s_cmn.frame_length as i32;
                // We need separate mutable borrows for the two channels, so split the array
                let [s0, s1] = &mut ps_enc.state_fxx;
                let x1 = &mut s0.s_cmn.input_buf[..frame_length + 2];
                let x2 = &mut s1.s_cmn.input_buf[..frame_length + 2];
                silk_stereo_lr_to_ms(
                    &mut ps_enc.s_stereo,
                    x1,
                    x2,
                    nfe,
                    &mut mstarget_rates_bps,
                    target_rate_bps,
                    speech_activity,
                    enc_control.to_mono,
                    fs_k_hz,
                    frame_len_i32,
                );
            }
            if ps_enc.s_stereo.mid_only_flags[ps_enc.state_fxx[0].s_cmn.n_frames_encoded as usize]
                as i32
                == 0
            {
                if ps_enc.prev_decode_only_middle == 1 {
                    ps_enc.state_fxx[1].s_shape = silk_shape_state_FLP::default();
                    ps_enc.state_fxx[1].s_cmn.s_nsq = silk_nsq_state::default();
                    ps_enc.state_fxx[1].s_cmn.prev_nlsfq_q15 = [0; 16];
                    ps_enc.state_fxx[1].s_cmn.s_lp.in_lp_state = [0; 2];
                    ps_enc.state_fxx[1].s_cmn.prev_lag = 100;
                    ps_enc.state_fxx[1].s_cmn.s_nsq.lag_prev = 100;
                    ps_enc.state_fxx[1].s_shape.last_gain_index = 10;
                    ps_enc.state_fxx[1].s_cmn.prev_signal_type = TYPE_NO_VOICE_ACTIVITY as i8;
                    ps_enc.state_fxx[1].s_cmn.s_nsq.prev_gain_q16 = 65536;
                    ps_enc.state_fxx[1].s_cmn.first_frame_after_reset = 1;
                }
                silk_encode_do_vad_flp(&mut ps_enc.state_fxx[1], activity);
            } else {
                ps_enc.state_fxx[1].s_cmn.vad_flags
                    [ps_enc.state_fxx[0].s_cmn.n_frames_encoded as usize] = 0;
            }
            if prefill_flag == 0 {
                let ps_range_enc = &mut **ps_range_enc.as_mut().unwrap();
                let nfe = ps_enc.state_fxx[0].s_cmn.n_frames_encoded as usize;
                silk_stereo_encode_pred(ps_range_enc, &ps_enc.s_stereo.pred_ix[nfe]);
                if ps_enc.state_fxx[1].s_cmn.vad_flags[nfe] as i32 == 0 {
                    silk_stereo_encode_mid_only(ps_range_enc, ps_enc.s_stereo.mid_only_flags[nfe]);
                }
            }
        } else {
            let frame_length = ps_enc.state_fxx[0].s_cmn.frame_length;
            ps_enc.state_fxx[0].s_cmn.input_buf[..2].copy_from_slice(&ps_enc.s_stereo.s_mid);
            ps_enc.s_stereo.s_mid.copy_from_slice(
                &ps_enc.state_fxx[0].s_cmn.input_buf[frame_length..frame_length + 2],
            );
        }
        silk_encode_do_vad_flp(&mut ps_enc.state_fxx[0], activity);
        n = 0;
        while n < enc_control.n_channels_internal {
            let mut max_bits: i32;
            let mut use_cbr: i32;
            max_bits = enc_control.max_bits;
            if tot_blocks == 2 && curr_block == 0 {
                max_bits = max_bits * 3 / 5;
            } else if tot_blocks == 3 {
                if curr_block == 0 {
                    max_bits = max_bits * 2 / 5;
                } else if curr_block == 1 {
                    max_bits = max_bits * 3 / 4;
                }
            }
            use_cbr = (enc_control.use_cbr != 0 && curr_block == tot_blocks - 1) as i32;
            if enc_control.n_channels_internal == 1 {
                channel_rate_bps = target_rate_bps;
            } else {
                channel_rate_bps = mstarget_rates_bps[n as usize];
                if n == 0 && mstarget_rates_bps[1] > 0 {
                    use_cbr = 0;
                    max_bits -= enc_control.max_bits / (tot_blocks * 2);
                }
            }
            if channel_rate_bps > 0 {
                let cond_coding_0: i32;
                silk_control_snr(&mut ps_enc.state_fxx[n as usize].s_cmn, channel_rate_bps);
                if ps_enc.state_fxx[0].s_cmn.n_frames_encoded - n <= 0 {
                    cond_coding_0 = CODE_INDEPENDENTLY;
                } else if n > 0 && ps_enc.prev_decode_only_middle != 0 {
                    cond_coding_0 = CODE_INDEPENDENTLY_NO_LTP_SCALING;
                } else {
                    cond_coding_0 = CODE_CONDITIONALLY;
                }
                let ps_range_enc = ps_range_enc.as_deref_mut();
                ret = silk_encode_frame_flp(
                    &mut ps_enc.state_fxx[n as usize],
                    n_bytes_out,
                    ps_range_enc,
                    cond_coding_0,
                    max_bits,
                    use_cbr,
                );
                debug_assert_eq!(ret, 0);
            }
            ps_enc.state_fxx[n as usize]
                .s_cmn
                .controlled_since_last_payload = 0;
            ps_enc.state_fxx[n as usize].s_cmn.input_buf_ix = 0;
            ps_enc.state_fxx[n as usize].s_cmn.n_frames_encoded += 1;
            n += 1;
        }
        ps_enc.prev_decode_only_middle = ps_enc.s_stereo.mid_only_flags
            [(ps_enc.state_fxx[0].s_cmn.n_frames_encoded - 1) as usize]
            as i32;
        if *n_bytes_out > 0
            && ps_enc.state_fxx[0].s_cmn.n_frames_encoded
                == ps_enc.state_fxx[0].s_cmn.n_frames_per_packet
        {
            flags = 0;
            n = 0;
            while n < enc_control.n_channels_internal {
                _i = 0;
                while _i < ps_enc.state_fxx[n as usize].s_cmn.n_frames_per_packet {
                    flags = ((flags as u32) << 1) as i32;
                    flags |= ps_enc.state_fxx[n as usize].s_cmn.vad_flags[_i as usize] as i32;
                    _i += 1;
                }
                flags = ((flags as u32) << 1) as i32;
                flags |= ps_enc.state_fxx[n as usize].s_cmn.lbrr_flag as i32;
                n += 1;
            }
            if prefill_flag == 0 {
                ec_enc_patch_initial_bits(
                    ps_range_enc.as_mut().unwrap(),
                    flags as u32,
                    ((ps_enc.state_fxx[0].s_cmn.n_frames_per_packet + 1)
                        * enc_control.n_channels_internal) as u32,
                );
            }
            if ps_enc.state_fxx[0].s_cmn.in_dtx != 0
                && (enc_control.n_channels_internal == 1 || ps_enc.state_fxx[1].s_cmn.in_dtx != 0)
            {
                *n_bytes_out = 0;
            }
            ps_enc.n_bits_exceeded += *n_bytes_out * 8;
            ps_enc.n_bits_exceeded -= enc_control.bit_rate * enc_control.payload_size_ms / 1000;
            ps_enc.n_bits_exceeded = ps_enc.n_bits_exceeded.clamp(0, 10000);
            let speech_act_thr_for_switch_q8 =
                (((SPEECH_ACTIVITY_DTX_THRES * (1 << 8) as f32) as f64 + 0.5f64) as i32 as i64
                    + (((((1_f32 - SPEECH_ACTIVITY_DTX_THRES) / MAX_BANDWIDTH_SWITCH_DELAY_MS
                        * (1 << (16 + 8)) as f32) as f64
                        + 0.5f64) as i32 as i64
                        * ps_enc.time_since_switch_allowed_ms as i16 as i64)
                        >> 16)) as i32;
            if ps_enc.state_fxx[0].s_cmn.speech_activity_q8 < speech_act_thr_for_switch_q8 {
                ps_enc.allow_bandwidth_switch = 1;
                ps_enc.time_since_switch_allowed_ms = 0;
            } else {
                ps_enc.allow_bandwidth_switch = 0;
                ps_enc.time_since_switch_allowed_ms += enc_control.payload_size_ms;
            }
        }
        if n_samples_in == 0 {
            break;
        }
        curr_block += 1;
    }
    ps_enc.n_prev_channels_internal = enc_control.n_channels_internal;
    enc_control.allow_bandwidth_switch = ps_enc.allow_bandwidth_switch;
    enc_control.in_wbmode_without_variable_lp = (ps_enc.state_fxx[0].s_cmn.fs_k_hz == 16
        && ps_enc.state_fxx[0].s_cmn.s_lp.mode == 0)
        as i32;
    enc_control.internal_sample_rate = ps_enc.state_fxx[0].s_cmn.fs_k_hz as i16 as i32 * 1000;
    enc_control.stereo_width_q14 = if enc_control.to_mono != 0 {
        0
    } else {
        ps_enc.s_stereo.smth_width_q14 as i32
    };
    if prefill_flag != 0 {
        enc_control.payload_size_ms = tmp_payload_size_ms;
        enc_control.complexity = tmp_complexity;
        n = 0;
        while n < enc_control.n_channels_internal {
            ps_enc.state_fxx[n as usize]
                .s_cmn
                .controlled_since_last_payload = 0;
            ps_enc.state_fxx[n as usize].s_cmn.prefill_flag = 0;
            n += 1;
        }
    }
    enc_control.signal_type = ps_enc.state_fxx[0].s_cmn.indices.signal_type as i32;
    enc_control.offset = SILK_QUANTIZATION_OFFSETS_Q10
        [(ps_enc.state_fxx[0].s_cmn.indices.signal_type as i32 >> 1) as usize]
        [ps_enc.state_fxx[0].s_cmn.indices.quant_offset_type as usize]
        as i32;
    ret
}
