//! SILK decoder API.
//!
//! Upstream c: `silk/dec_API.c`

use crate::arch::Arch;

#[derive(Copy, Clone)]
#[repr(C)]
pub struct silk_DecControlStruct {
    pub n_channels_api: usize,
    pub n_channels_internal: i32,
    pub api_sample_rate: i32,
    pub internal_sample_rate: i32,
    pub payload_size_ms: i32,
    pub prev_pitch_lag: i32,
    /// Whether Deep PLC is enabled (complexity >= 5)
    pub enable_deep_plc: bool,
    /// OSCE enhancement method (0=none, 1=LACE, 2=NoLACE)
    #[cfg(feature = "osce")]
    pub osce_method: i32,
    /// Whether OSCE bandwidth extension is enabled
    #[cfg(feature = "osce")]
    pub enable_osce_bwe: bool,
    /// Current extended mode (from packet extensions)
    #[cfg(feature = "osce")]
    pub osce_extended_mode: i32,
    /// Previous extended mode
    #[cfg(feature = "osce")]
    pub prev_osce_extended_mode: i32,
}
pub const FLAG_DECODE_NORMAL: i32 = 0;
pub const FLAG_DECODE_LBRR: i32 = 2;
pub const FLAG_PACKET_LOST: i32 = 1;
use crate::celt::entdec::{ec_dec_bit_logp, ec_dec_icdf, EcDec};
use crate::silk::decode_frame::silk_decode_frame;
use crate::silk::decode_indices::silk_decode_indices;
use crate::silk::decode_pulses::silk_decode_pulses;
use crate::silk::decoder_set_fs::silk_decoder_set_fs;
use crate::silk::define::{
    CODE_CONDITIONALLY, CODE_INDEPENDENTLY, CODE_INDEPENDENTLY_NO_LTP_SCALING, MAX_API_FS_KHZ,
    MAX_FRAME_LENGTH, SHELL_CODEC_FRAME_LENGTH, TYPE_NO_VOICE_ACTIVITY, TYPE_VOICED,
};
use crate::silk::errors::{
    SILK_DEC_INVALID_FRAME_SIZE, SILK_DEC_INVALID_SAMPLING_FREQUENCY, SILK_NO_ERROR,
};
use crate::silk::init_decoder::{
    silk_decoder_state_new, silk_init_decoder as silk_init_decoder_state,
    silk_reset_decoder as silk_reset_decoder_state,
};
use crate::silk::resampler::silk_resampler;
use crate::silk::stereo_decode_pred::{silk_stereo_decode_mid_only, silk_stereo_decode_pred};
use crate::silk::stereo_ms_to_lr::silk_stereo_ms_to_lr;
use crate::silk::structs::{silk_decoder_state, stereo_dec_state};
use crate::silk::tables_other::SILK_LBRR_FLAGS_ICDF_PTR;

#[derive(Clone)]
#[repr(C)]
pub struct silk_decoder {
    pub channel_state: [silk_decoder_state; 2],
    pub s_stereo: stereo_dec_state,
    pub n_channels_api: i32,
    pub n_channels_internal: i32,
    pub prev_decode_only_middle: bool,
    #[cfg(feature = "osce")]
    pub osce_model: crate::dnn::osce::OSCEModel,
}
pub fn silk_init_decoder() -> silk_decoder {
    silk_decoder {
        channel_state: [silk_decoder_state_new(), silk_decoder_state_new()],
        s_stereo: stereo_dec_state::default(),
        n_channels_api: 0,
        n_channels_internal: 0,
        prev_decode_only_middle: false,
        #[cfg(feature = "osce")]
        osce_model: crate::dnn::osce::OSCEModel::default(),
    }
}

/// Reset decoder state without full reinitialization.
///
/// Upstream c: silk/dec_API.c:silk_ResetDecoder
pub fn silk_reset_decoder(dec: &mut silk_decoder) {
    for ch in dec.channel_state.iter_mut() {
        let _ = silk_reset_decoder_state(ch);
    }
    dec.s_stereo = stereo_dec_state::default();
    dec.prev_decode_only_middle = false;
}

/// Upstream c: silk/dec_API.c:silk_Decode
pub fn silk_decode(
    dec_state: &mut silk_decoder,
    dec_control: &mut silk_DecControlStruct,
    lost_flag: i32,
    new_packet_flag: i32,
    ps_range_dec: &mut EcDec,
    samples_out: &mut [f32],
    n_samples_out: &mut i32,
    #[cfg(feature = "deep-plc")] mut lpcnet: Option<&mut crate::dnn::lpcnet::LPCNetPLCState>,
    arch: Arch,
) -> i32 {
    let mut _i: i32;
    let mut n: i32;
    let mut decode_only_middle: bool = false;
    let mut ret: i32 = SILK_NO_ERROR;
    let mut n_samples_out_dec: i32 = 0;
    let mut lbrr_symbol: i32;
    let mut ms_pred_q13: [i32; 2] = [0, 0];
    let ps_dec = dec_state;
    let channel_state = &mut ps_dec.channel_state;

    debug_assert!(dec_control.n_channels_internal == 1 || dec_control.n_channels_internal == 2);
    if new_packet_flag != 0 {
        n = 0;
        while n < dec_control.n_channels_internal {
            channel_state[n as usize].n_frames_decoded = 0;
            n += 1;
        }
    }
    if dec_control.n_channels_internal > ps_dec.n_channels_internal {
        ret += silk_init_decoder_state(&mut channel_state[1]);
    }
    let stereo_to_mono: i32 = (dec_control.n_channels_internal == 1
        && ps_dec.n_channels_internal == 2
        && dec_control.internal_sample_rate == 1000 * (channel_state[0]).fs_k_hz)
        as i32;
    if (channel_state[0]).n_frames_decoded == 0 {
        n = 0;
        while n < dec_control.n_channels_internal {
            if dec_control.payload_size_ms == 0 || dec_control.payload_size_ms == 10 {
                channel_state[n as usize].n_frames_per_packet = 1;
                channel_state[n as usize].nb_subfr = 2;
            } else if dec_control.payload_size_ms == 20 {
                channel_state[n as usize].n_frames_per_packet = 1;
                channel_state[n as usize].nb_subfr = 4;
            } else if dec_control.payload_size_ms == 40 {
                channel_state[n as usize].n_frames_per_packet = 2;
                channel_state[n as usize].nb_subfr = 4;
            } else if dec_control.payload_size_ms == 60 {
                channel_state[n as usize].n_frames_per_packet = 3;
                channel_state[n as usize].nb_subfr = 4;
            } else {
                return SILK_DEC_INVALID_FRAME_SIZE;
            }
            let fs_k_hz_dec: i32 = (dec_control.internal_sample_rate >> 10) + 1;
            if fs_k_hz_dec != 8 && fs_k_hz_dec != 12 && fs_k_hz_dec != 16 {
                return SILK_DEC_INVALID_SAMPLING_FREQUENCY;
            }
            ret += silk_decoder_set_fs(
                &mut channel_state[n as usize],
                fs_k_hz_dec,
                dec_control.api_sample_rate,
            );
            n += 1;
        }
    }
    if dec_control.n_channels_api == 2
        && dec_control.n_channels_internal == 2
        && (ps_dec.n_channels_api == 1 || ps_dec.n_channels_internal == 1)
    {
        ps_dec.s_stereo.pred_prev_q13.fill(0);
        ps_dec.s_stereo.s_side.fill(0);
        channel_state[1].resampler_state = channel_state[0].resampler_state;
    }
    ps_dec.n_channels_api = dec_control.n_channels_api as i32;
    ps_dec.n_channels_internal = dec_control.n_channels_internal;
    if dec_control.api_sample_rate > MAX_API_FS_KHZ * 1000 || dec_control.api_sample_rate < 8000 {
        ret = SILK_DEC_INVALID_SAMPLING_FREQUENCY;
        return ret;
    }
    if lost_flag != FLAG_PACKET_LOST && (channel_state[0]).n_frames_decoded == 0 {
        n = 0;
        while n < dec_control.n_channels_internal {
            _i = 0;
            while _i < channel_state[n as usize].n_frames_per_packet {
                channel_state[n as usize].vad_flags[_i as usize] = ec_dec_bit_logp(ps_range_dec, 1);
                _i += 1;
            }
            channel_state[n as usize].lbrr_flag = ec_dec_bit_logp(ps_range_dec, 1);
            n += 1;
        }
        n = 0;
        while n < dec_control.n_channels_internal {
            channel_state[n as usize].lbrr_flags.fill(0);
            if channel_state[n as usize].lbrr_flag != 0 {
                if channel_state[n as usize].n_frames_per_packet == 1 {
                    channel_state[n as usize].lbrr_flags[0] = 1;
                } else {
                    lbrr_symbol = ec_dec_icdf(
                        ps_range_dec,
                        SILK_LBRR_FLAGS_ICDF_PTR
                            [(channel_state[n as usize].n_frames_per_packet - 2) as usize],
                        8,
                    ) + 1;
                    _i = 0;
                    while _i < channel_state[n as usize].n_frames_per_packet {
                        channel_state[n as usize].lbrr_flags[_i as usize] = lbrr_symbol >> _i & 1;
                        _i += 1;
                    }
                }
            }
            n += 1;
        }
        if lost_flag == FLAG_DECODE_NORMAL {
            _i = 0;
            while _i < channel_state[0].n_frames_per_packet {
                n = 0;
                while n < dec_control.n_channels_internal {
                    if channel_state[n as usize].lbrr_flags[_i as usize] != 0 {
                        let mut pulses: [i16; 320] = [0; 320];
                        if dec_control.n_channels_internal == 2 && n == 0 {
                            silk_stereo_decode_pred(ps_range_dec, &mut ms_pred_q13);
                            if channel_state[1].lbrr_flags[_i as usize] == 0 {
                                silk_stereo_decode_mid_only(ps_range_dec, &mut decode_only_middle);
                            }
                        }
                        let cond_coding: i32 = if _i > 0
                            && channel_state[n as usize].lbrr_flags[(_i - 1) as usize] != 0
                        {
                            CODE_CONDITIONALLY
                        } else {
                            CODE_INDEPENDENTLY
                        };
                        silk_decode_indices(
                            &mut channel_state[n as usize],
                            ps_range_dec,
                            _i,
                            1,
                            cond_coding,
                        );

                        let frame_length = channel_state[n as usize].frame_length;
                        let mut shell_frames = frame_length / SHELL_CODEC_FRAME_LENGTH;
                        if shell_frames * SHELL_CODEC_FRAME_LENGTH < frame_length {
                            debug_assert_eq!(frame_length, 12 * 10);
                            shell_frames += 1;
                        }
                        let frame_buffer_length = shell_frames * SHELL_CODEC_FRAME_LENGTH;

                        silk_decode_pulses(
                            ps_range_dec,
                            &mut pulses[..frame_buffer_length],
                            channel_state[n as usize].indices.signal_type as i32,
                            channel_state[n as usize].indices.quant_offset_type as i32,
                        );
                    }
                    n += 1;
                }
                _i += 1;
            }
        }
    }
    if dec_control.n_channels_internal == 2 {
        if lost_flag == FLAG_DECODE_NORMAL
            || lost_flag == FLAG_DECODE_LBRR
                && channel_state[0].lbrr_flags[channel_state[0].n_frames_decoded as usize] == 1
        {
            silk_stereo_decode_pred(ps_range_dec, &mut ms_pred_q13);
            if lost_flag == FLAG_DECODE_NORMAL
                && channel_state[1].vad_flags[channel_state[0].n_frames_decoded as usize] == 0
                || lost_flag == FLAG_DECODE_LBRR
                    && channel_state[1].lbrr_flags[channel_state[0].n_frames_decoded as usize] == 0
            {
                silk_stereo_decode_mid_only(ps_range_dec, &mut decode_only_middle);
            } else {
                decode_only_middle = false;
            }
        } else {
            n = 0;
            while n < 2 {
                ms_pred_q13[n as usize] = ps_dec.s_stereo.pred_prev_q13[n as usize] as i32;
                n += 1;
            }
        }
    }
    if dec_control.n_channels_internal == 2 && !decode_only_middle && ps_dec.prev_decode_only_middle
    {
        channel_state[1].out_buf.fill(0);
        channel_state[1].s_lpc_q14_buf.fill(0);
        channel_state[1].lag_prev = 100;
        channel_state[1].last_gain_index = 10;
        channel_state[1].prev_signal_type = TYPE_NO_VOICE_ACTIVITY;
        channel_state[1].first_frame_after_reset = 1;
    }

    // Temporary buffers for decoded samples. Each channel needs frame_length + 2 elements.
    // The first 2 elements are stereo prediction state, decoded samples start at offset 2.
    let frame_len = channel_state[0].frame_length;
    let ch_buf_len = frame_len + 2;
    let _n_channels_int = dec_control.n_channels_internal;

    // Always allocate the temp storage (simplifies logic vs. the c "delay_stack_alloc" trick)
    // Max: 2 channels * (320 frame + 2 stereo state) = 644
    let mut samples_out1_tmp_storage = [0i16; 2 * (MAX_FRAME_LENGTH + 2)];

    // Channel offsets into samples_out1_tmp_storage
    let ch0_off: usize = 0;
    let ch1_off: usize = ch_buf_len;

    let has_side: i32 = if lost_flag == FLAG_DECODE_NORMAL {
        (!decode_only_middle) as i32
    } else {
        (!ps_dec.prev_decode_only_middle
            || dec_control.n_channels_internal == 2
                && lost_flag == FLAG_DECODE_LBRR
                && channel_state[1].lbrr_flags[channel_state[1].n_frames_decoded as usize] == 1)
            as i32
    };
    n = 0;
    while n < dec_control.n_channels_internal {
        if n == 0 || has_side != 0 {
            let cond_coding_0: i32;
            let frame_index: i32 = channel_state[0].n_frames_decoded - n;
            if frame_index <= 0 {
                cond_coding_0 = CODE_INDEPENDENTLY;
            } else if lost_flag == FLAG_DECODE_LBRR {
                cond_coding_0 =
                    if channel_state[n as usize].lbrr_flags[(frame_index - 1) as usize] != 0 {
                        CODE_CONDITIONALLY
                    } else {
                        CODE_INDEPENDENTLY
                    };
            } else if n > 0 && ps_dec.prev_decode_only_middle {
                cond_coding_0 = CODE_INDEPENDENTLY_NO_LTP_SCALING;
            } else {
                cond_coding_0 = CODE_CONDITIONALLY;
            }
            let ch_off = if n == 0 { ch0_off } else { ch1_off };
            let out_slice = &mut samples_out1_tmp_storage[ch_off + 2..ch_off + 2 + frame_len];

            // Reset OSCE state if method changed
            #[cfg(feature = "osce")]
            {
                if channel_state[n as usize].osce.method != dec_control.osce_method {
                    crate::dnn::osce::osce_reset(
                        &mut channel_state[n as usize].osce,
                        dec_control.osce_method,
                    );
                }
            }

            // Only pass lpcnet for channel 0 (mid channel)
            #[cfg(feature = "deep-plc")]
            let lpcnet_ch = if n == 0 { lpcnet.as_deref_mut() } else { None };

            let (err, n_out) = silk_decode_frame(
                &mut channel_state[n as usize],
                ps_range_dec,
                out_slice,
                lost_flag,
                cond_coding_0,
                #[cfg(feature = "deep-plc")]
                lpcnet_ch,
                #[cfg(feature = "osce")]
                &ps_dec.osce_model,
                arch,
            );
            ret += err;
            n_samples_out_dec = n_out;
        } else {
            let ch_off = if n == 0 { ch0_off } else { ch1_off };
            samples_out1_tmp_storage[ch_off + 2..ch_off + 2 + n_samples_out_dec as usize].fill(0);
        }
        channel_state[n as usize].n_frames_decoded += 1;
        n += 1;
    }

    if dec_control.n_channels_api == 2 && dec_control.n_channels_internal == 2 {
        let (ch0_slice, ch1_slice) = samples_out1_tmp_storage.split_at_mut(ch1_off);
        silk_stereo_ms_to_lr(
            &mut ps_dec.s_stereo,
            &mut ch0_slice[ch0_off..ch0_off + n_samples_out_dec as usize + 2],
            &mut ch1_slice[..n_samples_out_dec as usize + 2],
            &ms_pred_q13,
            channel_state[0].fs_k_hz as usize,
            n_samples_out_dec,
        );
    } else {
        // Copy s_mid[0..2] to beginning of channel 0 buffer
        samples_out1_tmp_storage[ch0_off] = ps_dec.s_stereo.s_mid[0];
        samples_out1_tmp_storage[ch0_off + 1] = ps_dec.s_stereo.s_mid[1];
        // Save last 2 samples back to s_mid
        ps_dec.s_stereo.s_mid[0] = samples_out1_tmp_storage[ch0_off + n_samples_out_dec as usize];
        ps_dec.s_stereo.s_mid[1] =
            samples_out1_tmp_storage[ch0_off + n_samples_out_dec as usize + 1];
    }

    *n_samples_out = n_samples_out_dec * dec_control.api_sample_rate
        / (channel_state[0].fs_k_hz as i16 as i32 * 1000);

    // Max: API rate 48kHz, 20ms frame = 960 samples
    let mut samples_out2_tmp = [0i16; 960];

    #[cfg(feature = "osce")]
    let mut resamp_buffer = [0i16; 3 * MAX_FRAME_LENGTH];

    n = 0;
    while n
        < (if (dec_control.n_channels_api as i32) < dec_control.n_channels_internal {
            dec_control.n_channels_api as i32
        } else {
            dec_control.n_channels_internal
        })
    {
        let ch_off = if n == 0 { ch0_off } else { ch1_off };
        let resample_input =
            &samples_out1_tmp_storage[ch_off + 1..ch_off + 1 + n_samples_out_dec as usize];

        // Always resample into temp buffer, then convert int16→float into samples_out
        let resample_out: &mut [i16] = &mut samples_out2_tmp;

        #[cfg(feature = "osce")]
        {
            use crate::dnn::osce::{
                osce_bwe, osce_bwe_cross_fade_10ms, osce_bwe_reset, OSCE_MODE_HYBRID,
                OSCE_MODE_SILK_BBWE, OSCE_MODE_SILK_ONLY,
            };

            if dec_control.osce_extended_mode == OSCE_MODE_SILK_BBWE {
                if dec_control.prev_osce_extended_mode != OSCE_MODE_SILK_BBWE {
                    osce_bwe_reset(&mut channel_state[n as usize].osce_bwe);
                }

                osce_bwe(
                    &ps_dec.osce_model,
                    &mut channel_state[n as usize].osce_bwe,
                    resample_out,
                    resample_input,
                    n_samples_out_dec as usize,
                    arch,
                );

                if dec_control.prev_osce_extended_mode == OSCE_MODE_SILK_ONLY
                    || dec_control.prev_osce_extended_mode == OSCE_MODE_HYBRID
                {
                    // Cross-fade with upsampled signal
                    silk_resampler(
                        &mut channel_state[n as usize].resampler_state,
                        &mut resamp_buffer,
                        resample_input,
                    );
                    osce_bwe_cross_fade_10ms(resample_out, &resamp_buffer, 480);
                }
            } else {
                ret += silk_resampler(
                    &mut channel_state[n as usize].resampler_state,
                    resample_out,
                    resample_input,
                );
                if dec_control.prev_osce_extended_mode == OSCE_MODE_SILK_BBWE
                    && dec_control.internal_sample_rate == 16000
                {
                    // Fade out: run BWE into temp buffer and crossfade
                    osce_bwe(
                        &ps_dec.osce_model,
                        &mut channel_state[n as usize].osce_bwe,
                        &mut resamp_buffer,
                        resample_input,
                        n_samples_out_dec as usize,
                        arch,
                    );
                    osce_bwe_cross_fade_10ms(resample_out, &resamp_buffer, 480);
                }
            }
        }

        #[cfg(not(feature = "osce"))]
        {
            ret += silk_resampler(
                &mut channel_state[n as usize].resampler_state,
                resample_out,
                resample_input,
            );
        }

        // Interleave if stereo output, or copy for mono; convert int16→float
        if dec_control.n_channels_api == 2 {
            _i = 0;
            while _i < *n_samples_out {
                samples_out[(n + 2 * _i) as usize] =
                    samples_out2_tmp[_i as usize] as f32 * (1.0 / 32768.0);
                _i += 1;
            }
        } else {
            _i = 0;
            while _i < *n_samples_out {
                samples_out[_i as usize] = samples_out2_tmp[_i as usize] as f32 * (1.0 / 32768.0);
                _i += 1;
            }
        }
        n += 1;
    }

    #[cfg(feature = "osce")]
    {
        dec_control.prev_osce_extended_mode = dec_control.osce_extended_mode;
    }

    // Create two channel output from mono stream
    if dec_control.n_channels_api == 2 && dec_control.n_channels_internal == 1 {
        if stereo_to_mono != 0 {
            let resample_input =
                &samples_out1_tmp_storage[ch0_off + 1..ch0_off + 1 + n_samples_out_dec as usize];
            ret += silk_resampler(
                &mut channel_state[1].resampler_state,
                &mut samples_out2_tmp,
                resample_input,
            );
            _i = 0;
            while _i < *n_samples_out {
                samples_out[(1 + 2 * _i) as usize] =
                    samples_out2_tmp[_i as usize] as f32 * (1.0 / 32768.0);
                _i += 1;
            }
        } else {
            _i = 0;
            while _i < *n_samples_out {
                samples_out[(1 + 2 * _i) as usize] = samples_out[(2 * _i) as usize];
                _i += 1;
            }
        }
    }

    if channel_state[0].prev_signal_type == TYPE_VOICED {
        let mult_tab: [i32; 3] = [6, 4, 3];
        dec_control.prev_pitch_lag =
            channel_state[0].lag_prev * mult_tab[((channel_state[0].fs_k_hz - 8) >> 2) as usize];
    } else {
        dec_control.prev_pitch_lag = 0;
    }

    if lost_flag == FLAG_PACKET_LOST {
        _i = 0;
        while _i < ps_dec.n_channels_internal {
            ps_dec.channel_state[_i as usize].last_gain_index = 10;
            _i += 1;
        }
    } else {
        ps_dec.prev_decode_only_middle = decode_only_middle;
    }
    ret
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arch::Arch;
    use crate::celt::entdec::ec_dec_init;

    fn baseline_control() -> silk_DecControlStruct {
        silk_DecControlStruct {
            n_channels_api: 1,
            n_channels_internal: 1,
            api_sample_rate: 16_000,
            internal_sample_rate: 16_000,
            payload_size_ms: 20,
            prev_pitch_lag: 0,
            enable_deep_plc: false,
            #[cfg(feature = "osce")]
            osce_method: 0,
            #[cfg(feature = "osce")]
            enable_osce_bwe: false,
            #[cfg(feature = "osce")]
            osce_extended_mode: 0,
            #[cfg(feature = "osce")]
            prev_osce_extended_mode: 0,
        }
    }

    fn decode_once(
        dec: &mut silk_decoder,
        ctrl: &mut silk_DecControlStruct,
        out: &mut [f32],
    ) -> i32 {
        let mut bytes = [0u8; 1];
        let mut range_dec = ec_dec_init(&mut bytes);
        let mut n_samples_out = 0;
        #[cfg(feature = "deep-plc")]
        {
            silk_decode(
                dec,
                ctrl,
                FLAG_PACKET_LOST,
                1,
                &mut range_dec,
                out,
                &mut n_samples_out,
                None,
                Arch::default(),
            )
        }
        #[cfg(not(feature = "deep-plc"))]
        {
            silk_decode(
                dec,
                ctrl,
                FLAG_PACKET_LOST,
                1,
                &mut range_dec,
                out,
                &mut n_samples_out,
                Arch::default(),
            )
        }
    }

    #[test]
    fn decode_rejects_invalid_payload_size() {
        let mut dec = silk_init_decoder();
        let mut ctrl = baseline_control();
        let mut out = [0.0f32; 960];
        ctrl.payload_size_ms = 15;
        let ret = decode_once(&mut dec, &mut ctrl, &mut out);
        assert_eq!(ret, SILK_DEC_INVALID_FRAME_SIZE);
    }

    #[test]
    fn decode_rejects_invalid_internal_sampling_frequency() {
        let mut dec = silk_init_decoder();
        let mut ctrl = baseline_control();
        let mut out = [0.0f32; 960];
        ctrl.internal_sample_rate = 44_100;
        let ret = decode_once(&mut dec, &mut ctrl, &mut out);
        assert_eq!(ret, SILK_DEC_INVALID_SAMPLING_FREQUENCY);
    }
}
