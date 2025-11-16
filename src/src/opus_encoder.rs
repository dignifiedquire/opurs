use crate::opus_private::opus_select_arch;
use crate::src::repacketizer::FrameSource;

pub const CELT_SIG_SCALE: f32 = 32768.0f32;
pub const Q15ONE: f32 = 1.0f32;
pub const EPSILON: f32 = 1e-15f32;
pub const VERY_SMALL: f32 = 1e-30f32;

use crate::celt::celt::{
    CELT_GET_MODE_REQUEST, CELT_SET_ANALYSIS_REQUEST, CELT_SET_CHANNELS_REQUEST,
    CELT_SET_END_BAND_REQUEST, CELT_SET_PREDICTION_REQUEST, CELT_SET_SIGNALLING_REQUEST,
    CELT_SET_SILK_INFO_REQUEST, CELT_SET_START_BAND_REQUEST, OPUS_SET_ENERGY_MASK_REQUEST,
    OPUS_SET_LFE_REQUEST,
};
use crate::celt::celt_encoder::{celt_encode_with_ec, OpusCustomEncoder, SILKInfo};
use crate::celt::entcode::ec_tell;
use crate::celt::entenc::{ec_enc_bit_logp, ec_enc_done, ec_enc_init, ec_enc_shrink, ec_enc_uint};
use crate::celt::float_cast::FLOAT2INT16;
use crate::celt::mathops::{celt_exp2, celt_maxabs16, celt_sqrt};
use crate::celt::modes::OpusCustomMode;
use crate::celt::pitch::celt_inner_prod_c;
use crate::silk::define::{
    DTX_ACTIVITY_THRESHOLD, MAX_CONSECUTIVE_DTX, NB_SPEECH_FRAMES_BEFORE_DTX, VAD_NO_DECISION,
};
use crate::silk::enc_API::silk_EncControlStruct;
use crate::silk::enc_API::{silk_Encode, silk_InitEncoder};
use crate::silk::float::structs_FLP::silk_encoder;
use crate::silk::lin2log::silk_lin2log;
use crate::silk::log2lin::silk_log2lin;
use crate::silk::tuning_parameters::{VARIABLE_HP_MIN_CUTOFF_HZ, VARIABLE_HP_SMTH_COEF2};
use crate::src::analysis::{AnalysisInfo, DownmixFn, TonalityAnalysisState};
use crate::src::opus_defines::{
    OPUS_APPLICATION_AUDIO, OPUS_APPLICATION_RESTRICTED_LOWDELAY, OPUS_APPLICATION_VOIP, OPUS_AUTO,
    OPUS_BAD_ARG, OPUS_BANDWIDTH_FULLBAND, OPUS_BANDWIDTH_MEDIUMBAND, OPUS_BANDWIDTH_NARROWBAND,
    OPUS_BANDWIDTH_SUPERWIDEBAND, OPUS_BANDWIDTH_WIDEBAND, OPUS_BITRATE_MAX, OPUS_BUFFER_TOO_SMALL,
    OPUS_FRAMESIZE_100_MS, OPUS_FRAMESIZE_10_MS, OPUS_FRAMESIZE_120_MS, OPUS_FRAMESIZE_20_MS,
    OPUS_FRAMESIZE_2_5_MS, OPUS_FRAMESIZE_40_MS, OPUS_FRAMESIZE_5_MS, OPUS_FRAMESIZE_60_MS,
    OPUS_FRAMESIZE_80_MS, OPUS_FRAMESIZE_ARG, OPUS_GET_APPLICATION_REQUEST,
    OPUS_GET_BANDWIDTH_REQUEST, OPUS_GET_BITRATE_REQUEST, OPUS_GET_COMPLEXITY_REQUEST,
    OPUS_GET_DTX_REQUEST, OPUS_GET_EXPERT_FRAME_DURATION_REQUEST, OPUS_GET_FINAL_RANGE_REQUEST,
    OPUS_GET_FORCE_CHANNELS_REQUEST, OPUS_GET_INBAND_FEC_REQUEST, OPUS_GET_IN_DTX_REQUEST,
    OPUS_GET_LOOKAHEAD_REQUEST, OPUS_GET_LSB_DEPTH_REQUEST, OPUS_GET_MAX_BANDWIDTH_REQUEST,
    OPUS_GET_PACKET_LOSS_PERC_REQUEST, OPUS_GET_PHASE_INVERSION_DISABLED_REQUEST,
    OPUS_GET_PREDICTION_DISABLED_REQUEST, OPUS_GET_SAMPLE_RATE_REQUEST, OPUS_GET_SIGNAL_REQUEST,
    OPUS_GET_VBR_CONSTRAINT_REQUEST, OPUS_GET_VBR_REQUEST, OPUS_INTERNAL_ERROR, OPUS_OK,
    OPUS_RESET_STATE, OPUS_SET_APPLICATION_REQUEST, OPUS_SET_BANDWIDTH_REQUEST,
    OPUS_SET_BITRATE_REQUEST, OPUS_SET_COMPLEXITY_REQUEST, OPUS_SET_DTX_REQUEST,
    OPUS_SET_EXPERT_FRAME_DURATION_REQUEST, OPUS_SET_FORCE_CHANNELS_REQUEST,
    OPUS_SET_INBAND_FEC_REQUEST, OPUS_SET_LSB_DEPTH_REQUEST, OPUS_SET_MAX_BANDWIDTH_REQUEST,
    OPUS_SET_PACKET_LOSS_PERC_REQUEST, OPUS_SET_PHASE_INVERSION_DISABLED_REQUEST,
    OPUS_SET_PREDICTION_DISABLED_REQUEST, OPUS_SET_SIGNAL_REQUEST, OPUS_SET_VBR_CONSTRAINT_REQUEST,
    OPUS_SET_VBR_REQUEST, OPUS_SIGNAL_MUSIC, OPUS_SIGNAL_VOICE, OPUS_UNIMPLEMENTED,
};
use crate::src::opus_private::{
    MODE_CELT_ONLY, MODE_HYBRID, MODE_SILK_ONLY, OPUS_GET_VOICE_RATIO_REQUEST,
    OPUS_SET_FORCE_MODE_REQUEST, OPUS_SET_VOICE_RATIO_REQUEST,
};
use crate::varargs::VarArgs;
use crate::{opus_custom_encoder_ctl, opus_packet_pad, OpusRepacketizer};

#[derive(Clone, Debug)]
#[repr(C)]
pub struct OpusEncoder {
    pub(crate) celt_enc: OpusCustomEncoder,
    pub(crate) silk_enc: silk_encoder,
    pub(crate) silk_mode: silk_EncControlStruct,
    pub(crate) application: i32,
    pub(crate) channels: i32,
    pub(crate) delay_compensation: i32,
    pub(crate) force_channels: i32,
    pub(crate) signal_type: i32,
    pub(crate) user_bandwidth: i32,
    pub(crate) max_bandwidth: i32,
    pub(crate) user_forced_mode: i32,
    pub(crate) voice_ratio: i32,
    pub(crate) Fs: i32,
    pub(crate) use_vbr: i32,
    pub(crate) vbr_constraint: i32,
    pub(crate) variable_duration: i32,
    pub(crate) bitrate_bps: i32,
    pub(crate) user_bitrate_bps: i32,
    pub(crate) lsb_depth: i32,
    pub(crate) encoder_buffer: i32,
    pub(crate) lfe: i32,
    pub(crate) arch: i32,
    pub(crate) use_dtx: i32,
    pub(crate) analysis: TonalityAnalysisState,
    pub(crate) stream_channels: i32,
    pub(crate) hybrid_stereo_width_Q14: i16,
    pub(crate) variable_HP_smth2_Q15: i32,
    pub(crate) prev_HB_gain: f32,
    pub(crate) hp_mem: [f32; 4],
    pub(crate) mode: i32,
    pub(crate) prev_mode: i32,
    pub(crate) prev_channels: i32,
    pub(crate) prev_framesize: i32,
    pub(crate) bandwidth: i32,
    pub(crate) auto_bandwidth: i32,
    pub(crate) silk_bw_switch: i32,
    pub(crate) first: i32,
    pub(crate) energy_masking: *mut f32,
    pub(crate) width_mem: StereoWidthState,
    pub(crate) delay_buffer: [f32; 960],
    pub(crate) detected_bandwidth: i32,
    pub(crate) nb_no_activity_frames: i32,
    pub(crate) peak_signal_energy: f32,
    pub(crate) nonfinal_frame: i32,
    pub(crate) rangeFinal: u32,
}
#[derive(Copy, Clone, Default, Debug)]
#[repr(C)]
pub struct StereoWidthState {
    pub XX: f32,
    pub XY: f32,
    pub YY: f32,
    pub smoothed_width: f32,
    pub max_follower: f32,
}
pub const PSEUDO_SNR_THRESHOLD: f32 = 316.23f32;
const mono_voice_bandwidth_thresholds: [i32; 8] = [9000, 700, 9000, 700, 13500, 1000, 14000, 2000];
const mono_music_bandwidth_thresholds: [i32; 8] = [9000, 700, 9000, 700, 11000, 1000, 12000, 2000];
const stereo_voice_bandwidth_thresholds: [i32; 8] =
    [9000, 700, 9000, 700, 13500, 1000, 14000, 2000];
const stereo_music_bandwidth_thresholds: [i32; 8] =
    [9000, 700, 9000, 700, 11000, 1000, 12000, 2000];
const stereo_voice_threshold: i32 = 19000;
const stereo_music_threshold: i32 = 17000;
const mode_thresholds: [[i32; 2]; 2] = [[64000, 10000], [44000, 10000]];
const fec_thresholds: [i32; 10] = [
    12000, 1000, 14000, 1000, 16000, 1000, 20000, 1000, 22000, 1000,
];

impl OpusEncoder {
    pub fn new(Fs: i32, channels: i32, application: i32) -> Result<Self, i32> {
        if Fs != 48000 && Fs != 24000 && Fs != 16000 && Fs != 12000 && Fs != 8000
            || channels != 1 && channels != 2
            || application != OPUS_APPLICATION_VOIP
                && application != OPUS_APPLICATION_AUDIO
                && application != OPUS_APPLICATION_RESTRICTED_LOWDELAY
        {
            return Err(OPUS_BAD_ARG);
        }
        let arch = opus_select_arch();

        let (silk_enc, mut silk_mode) = silk_InitEncoder(arch);
        silk_mode.nChannelsAPI = channels;
        silk_mode.nChannelsInternal = channels;
        silk_mode.API_sampleRate = Fs;
        silk_mode.maxInternalSampleRate = 16000;
        silk_mode.minInternalSampleRate = 8000;
        silk_mode.desiredInternalSampleRate = 16000;
        silk_mode.payloadSize_ms = 20;
        silk_mode.bitRate = 25000;
        silk_mode.packetLossPercentage = 0;
        silk_mode.complexity = 9;
        silk_mode.useInBandFEC = 0;
        silk_mode.useDTX = 0;
        silk_mode.useCBR = 0;
        silk_mode.reducedDependency = 0;

        let mut celt_enc = OpusCustomEncoder::new(Fs, channels, arch)?;
        opus_custom_encoder_ctl!(&mut celt_enc, CELT_SET_SIGNALLING_REQUEST, 0);
        opus_custom_encoder_ctl!(
            &mut celt_enc,
            OPUS_SET_COMPLEXITY_REQUEST,
            silk_mode.complexity,
        );

        let analysis = TonalityAnalysisState::new(application, Fs);

        Ok(Self {
            celt_enc,
            silk_enc,
            channels,
            stream_channels: channels,
            Fs,
            arch,
            silk_mode,
            use_vbr: 1,
            vbr_constraint: 1,
            user_bitrate_bps: OPUS_AUTO,
            bitrate_bps: 3000 + Fs * channels,
            application: application,
            signal_type: OPUS_AUTO,
            user_bandwidth: OPUS_AUTO,
            max_bandwidth: OPUS_BANDWIDTH_FULLBAND,
            force_channels: OPUS_AUTO,
            user_forced_mode: OPUS_AUTO,
            voice_ratio: -1,
            encoder_buffer: Fs / 100,
            lsb_depth: 24,
            variable_duration: OPUS_FRAMESIZE_ARG,
            delay_compensation: Fs / 250,
            hybrid_stereo_width_Q14: ((1) << 14) as i16,
            prev_HB_gain: Q15ONE,
            variable_HP_smth2_Q15: ((silk_lin2log(VARIABLE_HP_MIN_CUTOFF_HZ) as u32) << 8) as i32,
            first: 1,
            mode: MODE_HYBRID,
            bandwidth: OPUS_BANDWIDTH_FULLBAND,
            analysis,
            lfe: 0,
            use_dtx: 0,
            hp_mem: [0.; 4],
            prev_mode: 0,
            prev_channels: 0,
            prev_framesize: 0,
            auto_bandwidth: 0,
            silk_bw_switch: 0,
            energy_masking: std::ptr::null_mut(),
            width_mem: StereoWidthState::default(),
            delay_buffer: [0.; 960],
            detected_bandwidth: 0,
            nb_no_activity_frames: 0,
            peak_signal_energy: 0.,
            nonfinal_frame: 0,
            rangeFinal: 0,
        })
    }
}

fn gen_toc(mode: i32, mut framerate: i32, bandwidth: i32, channels: i32) -> u8 {
    let mut toc;
    let mut period = 0;
    while framerate < 400 {
        framerate <<= 1;
        period += 1;
    }
    if mode == MODE_SILK_ONLY {
        toc = (bandwidth - OPUS_BANDWIDTH_NARROWBAND << 5) as u8;
        toc = (toc as i32 | (period - 2) << 3) as u8;
    } else if mode == MODE_CELT_ONLY {
        let mut tmp: i32 = bandwidth - OPUS_BANDWIDTH_MEDIUMBAND;
        if tmp < 0 {
            tmp = 0;
        }
        toc = 0x80;
        toc = (toc as i32 | tmp << 5) as u8;
        toc = (toc as i32 | period << 3) as u8;
    } else {
        toc = 0x60;
        toc = (toc as i32 | bandwidth - OPUS_BANDWIDTH_SUPERWIDEBAND << 4) as u8;
        toc = (toc as i32 | (period - 2) << 3) as u8;
    }
    (toc as i32 | ((channels == 2) as i32) << 2) as u8
}

fn silk_biquad_float(
    in_0: &[f32],
    B_Q28: &[i32],
    A_Q28: &[i32],
    S: &mut [f32],
    out: &mut [f32],
    len: i32,
    stride: i32,
) {
    let mut A: [f32; 2] = [0.; 2];
    let mut B: [f32; 3] = [0.; 3];
    A[0] = A_Q28[0] as f32 * (1.0f32 / ((1) << 28) as f32);
    A[1] = A_Q28[1] as f32 * (1.0f32 / ((1) << 28) as f32);
    B[0] = B_Q28[0] as f32 * (1.0f32 / ((1) << 28) as f32);
    B[1] = B_Q28[1] as f32 * (1.0f32 / ((1) << 28) as f32);
    B[2] = B_Q28[2] as f32 * (1.0f32 / ((1) << 28) as f32);

    let mut vout;
    let mut inval;
    for k in 0..len {
        inval = in_0[(k * stride) as usize];
        vout = S[0] + B[0] * inval;
        S[0] = S[1] - vout * A[0] + B[1] * inval;
        S[1] = -vout * A[1] + B[2] * inval + VERY_SMALL;
        out[(k * stride) as usize] = vout;
    }
}

fn hp_cutoff(
    in_0: &[f32],
    cutoff_Hz: i32,
    out: &mut [f32],
    hp_mem: &mut [f32],
    len: i32,
    channels: i32,
    Fs: i32,
    _arch: i32,
) {
    let mut B_Q28: [i32; 3] = [0; 3];
    let mut A_Q28: [i32; 2] = [0; 2];
    let Fc_Q19 = (1.5f64 * 3.14159f64 / 1000 as f64 * ((1) << 19) as f64 + 0.5f64) as i32 as i16
        as i32
        * cutoff_Hz as i16 as i32
        / (Fs / 1000);
    let r_Q28 = (1.0f64 * ((1) << 28) as f64 + 0.5f64) as i32
        - (0.92f64 * ((1) << 9) as f64 + 0.5f64) as i32 * Fc_Q19;
    B_Q28[0 as usize] = r_Q28;
    B_Q28[1 as usize] = ((-r_Q28 as u32) << 1) as i32;
    B_Q28[2 as usize] = r_Q28;
    let r_Q22 = r_Q28 >> 6;
    A_Q28[0 as usize] = (r_Q22 as i64
        * ((Fc_Q19 as i64 * Fc_Q19 as i64 >> 16) as i32
            - (2.0f64 * ((1) << 22) as f64 + 0.5f64) as i32) as i64
        >> 16) as i32;
    A_Q28[1 as usize] = (r_Q22 as i64 * r_Q22 as i64 >> 16) as i32;
    silk_biquad_float(in_0, &mut B_Q28, &mut A_Q28, hp_mem, out, len, channels);
    if channels == 2 {
        silk_biquad_float(
            &in_0[1..],
            &B_Q28,
            &A_Q28,
            &mut hp_mem[2..],
            &mut out[1..],
            len,
            channels,
        );
    }
}
fn dc_reject(
    in_0: &[f32],
    cutoff_Hz: i32,
    out: &mut [f32],
    hp_mem: &mut [f32],
    len: i32,
    channels: i32,
    Fs: i32,
) {
    let coef = 6.3f32 * cutoff_Hz as f32 / Fs as f32;
    let coef2 = 1 as f32 - coef;
    if channels == 2 {
        let mut m0 = hp_mem[0];
        let mut m2 = hp_mem[2];
        for i in 0..len {
            let x0 = in_0[(2 * i + 0) as usize];
            let x1 = in_0[(2 * i + 1) as usize];
            let out0 = x0 - m0;
            let out1 = x1 - m2;
            m0 = coef * x0 + VERY_SMALL + coef2 * m0;
            m2 = coef * x1 + VERY_SMALL + coef2 * m2;
            out[(2 * i + 0) as usize] = out0;
            out[(2 * i + 1) as usize] = out1;
        }
        hp_mem[0] = m0;
        hp_mem[2] = m2;
    } else {
        let mut m0_0 = hp_mem[0];
        for i in 0..len {
            let x = in_0[i as usize];
            let y = x - m0_0;
            m0_0 = coef * x + VERY_SMALL + coef2 * m0_0;
            out[i as usize] = y;
        }
        hp_mem[0] = m0_0;
    };
}

fn stereo_fade(
    in_out: &mut [f32],
    mut g1: f32,
    mut g2: f32,
    overlap48: i32,
    frame_size: i32,
    channels: i32,
    window: &[f32],
    Fs: i32,
) {
    let inc = 48000 / Fs;
    let overlap = overlap48 / inc;
    g1 = Q15ONE - g1;
    g2 = Q15ONE - g2;
    let mut i = 0;
    while i < overlap {
        let w = window[(i * inc) as usize] * window[(i * inc) as usize];
        let g = w * g2 + (1.0f32 - w) * g1;
        let mut diff =
            0.5f32 * (in_out[(i * channels) as usize] - in_out[(i * channels + 1) as usize]);
        diff = g * diff;
        in_out[(i * channels) as usize] = in_out[(i * channels) as usize] - diff;
        in_out[(i * channels + 1) as usize] = in_out[(i * channels + 1) as usize] + diff;
        i += 1;
    }
    while i < frame_size {
        let mut diff_0 =
            0.5f32 * (in_out[(i * channels) as usize] - in_out[(i * channels + 1) as usize]);
        diff_0 = g2 * diff_0;
        in_out[(i * channels) as usize] = in_out[(i * channels) as usize] - diff_0;
        in_out[(i * channels + 1) as usize] = in_out[(i * channels + 1) as usize] + diff_0;
        i += 1;
    }
}

fn gain_fade(
    in_out: &mut [f32],
    g1: f32,
    g2: f32,
    overlap48: i32,
    frame_size: i32,
    channels: i32,
    window: &[f32],
    Fs: i32,
) {
    let inc = 48000 / Fs;
    let overlap = overlap48 / inc;
    if channels == 1 {
        for i in 0..overlap {
            let w = window[(i * inc) as usize] * window[(i * inc) as usize];
            let g = w * g2 + (1.0f32 - w) * g1;
            in_out[i as usize] = g * in_out[i as usize];
        }
    } else {
        for i in 0..overlap {
            let w_0 = window[(i * inc) as usize] * window[(i * inc) as usize];
            let g_0 = w_0 * g2 + (1.0f32 - w_0) * g1;
            in_out[(i * 2) as usize] = g_0 * in_out[(i * 2) as usize];
            in_out[(i * 2 + 1) as usize] = g_0 * in_out[(i * 2 + 1) as usize];
        }
    }
    let mut c = 0;
    loop {
        for i in overlap..frame_size {
            in_out[(i * channels + c) as usize] = g2 * in_out[(i * channels + c) as usize];
        }
        c += 1;
        if !(c < channels) {
            break;
        }
    }
}

fn user_bitrate_to_bitrate(st: &mut OpusEncoder, mut frame_size: i32, max_data_bytes: i32) -> i32 {
    if frame_size == 0 {
        frame_size = (*st).Fs / 400;
    }
    if st.user_bitrate_bps == OPUS_AUTO {
        60 * st.Fs / frame_size + st.Fs * st.channels
    } else if st.user_bitrate_bps == OPUS_BITRATE_MAX {
        max_data_bytes * 8 * (*st).Fs / frame_size
    } else {
        st.user_bitrate_bps
    }
}
pub fn downmix_float(
    x: &[f32],
    y: &mut [f32],
    subframe: i32,
    offset: i32,
    c1: i32,
    c2: i32,
    C: i32,
) {
    for j in 0..subframe {
        y[j as usize] = x[((j + offset) * C + c1) as usize] * CELT_SIG_SCALE;
    }

    if c2 > -1 {
        for j in 0..subframe {
            y[j as usize] += x[((j + offset) * C + c2) as usize] * CELT_SIG_SCALE;
        }
    } else if c2 == -2 {
        for c in 1..C {
            for j in 0..subframe {
                y[j as usize] += x[((j + offset) * C + c) as usize] * CELT_SIG_SCALE;
            }
        }
    }
}

pub fn downmix_int(x: &[i16], y: &mut [f32], subframe: i32, offset: i32, c1: i32, c2: i32, C: i32) {
    for j in 0..subframe {
        y[j as usize] = x[((j + offset) * C + c1) as usize] as f32;
    }
    if c2 > -1 {
        for j in 0..subframe {
            y[j as usize] += x[((j + offset) * C + c2) as usize] as i32 as f32;
        }
    } else if c2 == -(2) {
        for c in 1..C {
            for j in 0..subframe {
                y[j as usize] += x[((j + offset) * C + c) as usize] as i32 as f32;
            }
        }
    }
}

pub fn frame_size_select(frame_size: i32, variable_duration: i32, Fs: i32) -> i32 {
    if frame_size < Fs / 400 {
        return -1;
    }
    let new_size = if variable_duration == OPUS_FRAMESIZE_ARG {
        frame_size
    } else if variable_duration >= OPUS_FRAMESIZE_2_5_MS
        && variable_duration <= OPUS_FRAMESIZE_120_MS
    {
        if variable_duration <= OPUS_FRAMESIZE_40_MS {
            (Fs / 400) << variable_duration - OPUS_FRAMESIZE_2_5_MS
        } else {
            (variable_duration - OPUS_FRAMESIZE_2_5_MS - 2) * Fs / 50
        }
    } else {
        return -1;
    };
    if new_size > frame_size {
        return -1;
    }
    if 400 * new_size != Fs
        && 200 * new_size != Fs
        && 100 * new_size != Fs
        && 50 * new_size != Fs
        && 25 * new_size != Fs
        && 50 * new_size != 3 * Fs
        && 50 * new_size != 4 * Fs
        && 50 * new_size != 5 * Fs
        && 50 * new_size != 6 * Fs
    {
        return -1;
    }
    new_size
}

pub fn compute_stereo_width(
    pcm: &[f32],
    frame_size: i32,
    Fs: i32,
    mem: &mut StereoWidthState,
) -> f32 {
    let frame_rate = Fs / frame_size;
    let short_alpha =
        Q15ONE - 25 as f32 * 1.0f32 / (if 50 > frame_rate { 50 } else { frame_rate }) as f32;
    let mut yy = 0 as f32;
    let mut xy = yy;
    let mut xx = xy;
    for i in (0..frame_size - 3).step_by(4) {
        let mut x = pcm[(2 * i) as usize];
        let mut y = pcm[(2 * i + 1) as usize];
        let mut pxx = x * x;
        let mut pxy = x * y;
        let mut pyy = y * y;
        x = pcm[(2 * i + 2) as usize];
        y = pcm[(2 * i + 3) as usize];
        pxx += x * x;
        pxy += x * y;
        pyy += y * y;
        x = pcm[(2 * i + 4) as usize];
        y = pcm[(2 * i + 5) as usize];
        pxx += x * x;
        pxy += x * y;
        pyy += y * y;
        x = pcm[(2 * i + 6) as usize];
        y = pcm[(2 * i + 7) as usize];
        pxx += x * x;
        pxy += x * y;
        pyy += y * y;
        xx += pxx;
        xy += pxy;
        yy += pyy;
    }
    if !(xx < 1e9f32) || xx != xx || !(yy < 1e9f32) || yy != yy {
        yy = 0 as f32;
        xx = yy;
        xy = xx;
    }
    mem.XX += short_alpha * (xx - mem.XX);
    mem.XY += short_alpha * (xy - mem.XY);
    mem.YY += short_alpha * (yy - mem.YY);
    mem.XX = if 0 as f32 > mem.XX { 0 as f32 } else { mem.XX };
    mem.XY = if 0 as f32 > mem.XY { 0 as f32 } else { mem.XY };
    mem.YY = if 0 as f32 > mem.YY { 0 as f32 } else { mem.YY };
    if (if mem.XX > mem.YY { mem.XX } else { mem.YY }) > 8e-4f32 {
        let sqrt_xx = celt_sqrt(mem.XX);
        let sqrt_yy = celt_sqrt(mem.YY);
        let qrrt_xx = celt_sqrt(sqrt_xx);
        let qrrt_yy = celt_sqrt(sqrt_yy);
        mem.XY = if mem.XY < sqrt_xx * sqrt_yy {
            mem.XY
        } else {
            sqrt_xx * sqrt_yy
        };
        let corr = mem.XY / (1e-15f32 + sqrt_xx * sqrt_yy);
        let ldiff = 1.0f32 * (qrrt_xx - qrrt_yy).abs() / (EPSILON + qrrt_xx + qrrt_yy);
        let width = celt_sqrt(1.0f32 - corr * corr) * ldiff;
        mem.smoothed_width += (width - mem.smoothed_width) / frame_rate as f32;
        mem.max_follower = if mem.max_follower - 0.02f32 / frame_rate as f32 > mem.smoothed_width {
            mem.max_follower - 0.02f32 / frame_rate as f32
        } else {
            mem.smoothed_width
        };
    }
    if 1.0 < 20. * mem.max_follower {
        1.0
    } else {
        20. * mem.max_follower
    }
}

fn decide_fec(
    useInBandFEC: i32,
    PacketLoss_perc: i32,
    last_fec: i32,
    mode: i32,
    bandwidth: &mut i32,
    rate: i32,
) -> i32 {
    if useInBandFEC == 0 || PacketLoss_perc == 0 || mode == MODE_CELT_ONLY {
        return 0;
    }
    let orig_bandwidth = *bandwidth;
    loop {
        let mut LBRR_rate_thres_bps =
            fec_thresholds[(2 * (*bandwidth - OPUS_BANDWIDTH_NARROWBAND)) as usize];
        let hysteresis =
            fec_thresholds[(2 * (*bandwidth - OPUS_BANDWIDTH_NARROWBAND) + 1) as usize];
        if last_fec == 1 {
            LBRR_rate_thres_bps -= hysteresis;
        }
        if last_fec == 0 {
            LBRR_rate_thres_bps += hysteresis;
        }
        LBRR_rate_thres_bps = ((LBRR_rate_thres_bps
            * (125
                - (if PacketLoss_perc < 25 {
                    PacketLoss_perc
                } else {
                    25
                }))) as i64
            * (0.01f64 * ((1) << 16) as f64 + 0.5f64) as i32 as i16 as i64
            >> 16) as i32;
        if rate > LBRR_rate_thres_bps {
            return 1;
        } else if PacketLoss_perc <= 5 {
            return 0;
        } else {
            if !(*bandwidth > OPUS_BANDWIDTH_NARROWBAND) {
                break;
            }
            *bandwidth -= 1;
        }
    }
    *bandwidth = orig_bandwidth;
    0
}

fn compute_silk_rate_for_hybrid(
    mut rate: i32,
    bandwidth: i32,
    frame20ms: i32,
    vbr: i32,
    fec: i32,
    channels: i32,
) -> i32 {
    const NUM_ROWS: usize = 7;
    //  |total| |-------- SILK------------|
    //          |-- No FEC -| |--- FEC ---|
    //           10ms   20ms   10ms   20ms
    const RATE_TABLE: [[i32; 5]; NUM_ROWS] = [
        [0, 0, 0, 0, 0],
        [12000, 10000, 10000, 11000, 11000],
        [16000, 13500, 13500, 15000, 15000],
        [20000, 16000, 16000, 18000, 18000],
        [24000, 18000, 18000, 21000, 21000],
        [32000, 22000, 22000, 28000, 28000],
        [64000, 38000, 38000, 50000, 50000],
    ];
    rate /= channels;
    let entry = 1 + frame20ms + 2 * fec;
    let N = NUM_ROWS;
    let mut i = 1;
    while i < N {
        if RATE_TABLE[i as usize][0 as usize] > rate {
            break;
        }
        i += 1;
    }
    let mut silk_rate;
    if i == N {
        silk_rate = RATE_TABLE[(i - 1) as usize][entry as usize];
        silk_rate += (rate - RATE_TABLE[(i - 1) as usize][0 as usize]) / 2;
    } else {
        let lo = RATE_TABLE[(i - 1) as usize][entry as usize];
        let hi = RATE_TABLE[i as usize][entry as usize];
        let x0 = RATE_TABLE[(i - 1) as usize][0 as usize];
        let x1 = RATE_TABLE[i as usize][0 as usize];
        silk_rate = (lo * (x1 - rate) + hi * (rate - x0)) / (x1 - x0);
    }
    if vbr == 0 {
        silk_rate += 100;
    }
    if bandwidth == OPUS_BANDWIDTH_SUPERWIDEBAND {
        silk_rate += 300;
    }
    silk_rate *= channels;
    if channels == 2 && rate >= 12000 {
        silk_rate -= 1000;
    }
    silk_rate
}

fn compute_equiv_rate(
    bitrate: i32,
    channels: i32,
    frame_rate: i32,
    vbr: i32,
    mode: i32,
    complexity: i32,
    loss: i32,
) -> i32 {
    let mut equiv = bitrate;
    if frame_rate > 50 {
        equiv -= (40 * channels + 20) * (frame_rate - 50);
    }
    if vbr == 0 {
        equiv -= equiv / 12;
    }
    equiv = equiv * (90 + complexity) / 100;
    if mode == MODE_SILK_ONLY || mode == MODE_HYBRID {
        if complexity < 2 {
            equiv = equiv * 4 / 5;
        }
        equiv -= equiv * loss / (6 * loss + 10);
    } else if mode == MODE_CELT_ONLY {
        if complexity < 5 {
            equiv = equiv * 9 / 10;
        }
    } else {
        equiv -= equiv * loss / (12 * loss + 20);
    }
    equiv
}

pub fn is_digital_silence(pcm: &[f32], frame_size: i32, channels: i32, lsb_depth: i32) -> i32 {
    let sample_max = celt_maxabs16(pcm, (frame_size * channels) as usize);
    (sample_max <= 1 as f32 / ((1) << lsb_depth) as f32) as i32
}

fn compute_frame_energy(pcm: &[f32], frame_size: i32, channels: i32, _arch: i32) -> f32 {
    let len: i32 = frame_size * channels;
    celt_inner_prod_c(pcm, pcm, len) / len as f32
}

fn decide_dtx_mode(
    activity_probability: f32,
    nb_no_activity_frames: &mut i32,
    peak_signal_energy: f32,
    pcm: &[f32],
    frame_size: i32,
    channels: i32,
    mut is_silence: i32,
    arch: i32,
) -> i32 {
    if is_silence == 0 {
        if activity_probability < DTX_ACTIVITY_THRESHOLD {
            let noise_energy = compute_frame_energy(pcm, frame_size, channels, arch);
            is_silence = (peak_signal_energy >= PSEUDO_SNR_THRESHOLD * noise_energy) as i32;
        }
    }
    if is_silence != 0 {
        *nb_no_activity_frames += 1;
        if *nb_no_activity_frames > NB_SPEECH_FRAMES_BEFORE_DTX {
            if *nb_no_activity_frames <= NB_SPEECH_FRAMES_BEFORE_DTX + MAX_CONSECUTIVE_DTX {
                return 1;
            } else {
                *nb_no_activity_frames = NB_SPEECH_FRAMES_BEFORE_DTX;
            }
        }
    } else {
        *nb_no_activity_frames = 0;
    }
    0
}

fn encode_multiframe_packet(
    st: &mut OpusEncoder,
    pcm: &[f32],
    nb_frames: i32,
    frame_size: i32,
    data: &mut [u8],
    out_data_bytes: i32,
    to_celt: i32,
    lsb_depth: i32,
    float_api: i32,
) -> i32 {
    let cbr_bytes;
    let repacketize_len;

    // Worst cases:
    // 2 frames: Code 2 with different compressed sizes
    // >2 frames: Code 3 VBR
    let max_header_bytes = if nb_frames == 2 {
        3
    } else {
        2 + (nb_frames - 1) * 2
    };
    if st.use_vbr != 0 || st.user_bitrate_bps == OPUS_BITRATE_MAX {
        repacketize_len = out_data_bytes;
    } else {
        cbr_bytes = 3 * st.bitrate_bps / (3 * 8 * st.Fs / (frame_size * nb_frames));
        repacketize_len = if cbr_bytes < out_data_bytes {
            cbr_bytes
        } else {
            out_data_bytes
        };
    }
    let bytes_per_frame = if (1276) < 1 + (repacketize_len - max_header_bytes) / nb_frames {
        1276
    } else {
        1 + (repacketize_len - max_header_bytes) / nb_frames
    };
    let vla = (nb_frames * bytes_per_frame) as usize;
    let mut tmp_data: Vec<u8> = vec![0u8; vla];
    let mut rp = OpusRepacketizer::default();
    let bak_mode = st.user_forced_mode;
    let bak_bandwidth = st.user_bandwidth;
    let bak_channels = st.force_channels;
    st.user_forced_mode = st.mode;
    st.user_bandwidth = st.bandwidth;
    st.force_channels = st.stream_channels;
    let bak_to_mono = st.silk_mode.toMono;
    if bak_to_mono != 0 {
        st.force_channels = 1;
    } else {
        st.prev_channels = st.stream_channels;
    }
    let mut offsets = Vec::new();
    for i in 0..nb_frames {
        st.silk_mode.toMono = 0;
        st.nonfinal_frame = (i < nb_frames - 1) as i32;

        let start = (i * bytes_per_frame) as usize;

        // When switching from SILK/Hybrid to CELT, only ask for a switch at the last frame
        if to_celt != 0 && i == nb_frames - 1 {
            st.user_forced_mode = MODE_CELT_ONLY;
        }
        let tmp_len = opus_encode_native::<f32>(
            st,
            &pcm[(i * (st.channels * frame_size)) as usize..],
            frame_size,
            &mut tmp_data[start..],
            bytes_per_frame,
            lsb_depth,
            None,
            0,
            0,
            0,
            0,
            None,
            float_api,
        );
        if tmp_len < 0 {
            return OPUS_INTERNAL_ERROR;
        }

        let ret = rp.cat(&tmp_data[start..start + tmp_len as usize]);
        offsets.push((start, start + tmp_len as usize));
        if ret < 0 {
            return OPUS_INTERNAL_ERROR;
        }
    }

    // this relies on `rp.cat` keeping refernces into `tmp_data` and copying the frames from there,
    // instead of from `data` which does not happen anymore, as `rp.frames` no stores offsets, insted of pointers

    let offsets = offsets
        .into_iter()
        .map(|(start, end)| &tmp_data[start..end])
        .collect();
    let ret = rp.out_range_impl(
        0,
        nb_frames,
        &mut data[..repacketize_len as _],
        false,
        st.use_vbr == 0,
        FrameSource::Slice { data: offsets },
    );
    if ret < 0 {
        return OPUS_INTERNAL_ERROR;
    }

    // Discard configs that were forced locally for the purpose of repacketization
    st.user_forced_mode = bak_mode;
    st.user_bandwidth = bak_bandwidth;
    st.force_channels = bak_channels;
    st.silk_mode.toMono = bak_to_mono;

    ret
}

fn compute_redundancy_bytes(
    max_data_bytes: i32,
    bitrate_bps: i32,
    frame_rate: i32,
    channels: i32,
) -> i32 {
    let base_bits = 40 * channels + 20;
    let redundancy_rate = bitrate_bps + base_bits * (200 - frame_rate);
    let redundancy_rate = 3 * redundancy_rate / 2;
    let redundancy_bytes = redundancy_rate / 1600;
    let available_bits = max_data_bytes * 8 - 2 * base_bits;
    let redundancy_bytes_cap = (available_bits * 240 / (240 + 48000 / frame_rate) + base_bits) / 8;
    let mut redundancy_bytes = if redundancy_bytes < redundancy_bytes_cap {
        redundancy_bytes
    } else {
        redundancy_bytes_cap
    };
    if redundancy_bytes > 4 + 8 * channels {
        redundancy_bytes = if (257) < redundancy_bytes {
            257
        } else {
            redundancy_bytes
        };
    } else {
        redundancy_bytes = 0;
    }
    redundancy_bytes
}

pub fn opus_encode_native<T>(
    st: &mut OpusEncoder,
    pcm: &[f32],
    frame_size: i32,
    og_data: &mut [u8],
    out_data_bytes: i32,
    mut lsb_depth: i32,
    analysis_pcm: Option<&[T]>,
    analysis_size: i32,
    c1: i32,
    c2: i32,
    analysis_channels: i32,
    downmix: DownmixFn<T>,
    float_api: i32,
) -> i32 {
    let mut ret: i32 = 0;
    let mut nBytes: i32 = 0;
    let mut prefill: i32 = 0;
    let mut start_band: i32 = 0;
    let mut redundancy: i32 = 0;
    // Number of bytes to use for redundancy frame
    let mut redundancy_bytes: i32 = 0;
    let mut celt_to_silk: i32 = 0;
    let mut to_celt: i32 = 0;
    let mut redundant_rng: u32 = 0;
    let mut celt_mode: *const OpusCustomMode = 0 as *const OpusCustomMode;
    let mut analysis_info = AnalysisInfo::default();
    let mut analysis_read_pos_bak: i32 = -1;
    let mut analysis_read_subframe_bak: i32 = -1;
    let mut is_silence: i32 = 0;
    // Max number of bytes we're allowed to use
    let mut max_data_bytes = out_data_bytes.min(1276);
    st.rangeFinal = 0;
    if frame_size <= 0 || max_data_bytes <= 0 {
        return OPUS_BAD_ARG;
    }
    // Cannot encode 100 ms in 1 byte
    if max_data_bytes == 1 && st.Fs == frame_size * 10 {
        return OPUS_BUFFER_TOO_SMALL;
    }
    let delay_compensation = if st.application == OPUS_APPLICATION_RESTRICTED_LOWDELAY {
        0
    } else {
        st.delay_compensation
    };
    lsb_depth = lsb_depth.min(st.lsb_depth);

    opus_custom_encoder_ctl!(&mut st.celt_enc, CELT_GET_MODE_REQUEST, &mut celt_mode);
    analysis_info.valid = 0;

    if st.silk_mode.complexity >= 7 && st.Fs >= 16000 {
        is_silence = is_digital_silence(pcm, frame_size, st.channels, lsb_depth);
        analysis_read_pos_bak = st.analysis.read_pos;
        analysis_read_subframe_bak = st.analysis.read_subframe;
        analysis_info = st.analysis.run_analysis(
            unsafe { &*celt_mode },
            analysis_pcm,
            analysis_size,
            frame_size,
            c1,
            c2,
            analysis_channels,
            st.Fs,
            lsb_depth,
            downmix,
        );
        //  Track the peak signal energy
        if is_silence == 0 && analysis_info.activity_probability > DTX_ACTIVITY_THRESHOLD {
            st.peak_signal_energy = (0.999 * st.peak_signal_energy).max(compute_frame_energy(
                pcm,
                frame_size,
                st.channels,
                st.arch,
            ));
        }
    } else if st.analysis.initialized != 0 {
        st.analysis = TonalityAnalysisState::new(st.analysis.application, st.analysis.Fs);
    }

    // Reset voice_ratio if this frame is not silent or if analysis is disabled.
    // Otherwise, preserve voice_ratio from the last non-silent frame
    if is_silence == 0 {
        st.voice_ratio = -1;
    }
    st.detected_bandwidth = 0;
    if analysis_info.valid != 0 {
        if st.signal_type == OPUS_AUTO {
            let prob = if st.prev_mode == 0 {
                analysis_info.music_prob
            } else if st.prev_mode == MODE_CELT_ONLY {
                analysis_info.music_prob_max
            } else {
                analysis_info.music_prob_min
            };
            st.voice_ratio = (0.5 + (100.0 * (1.0 - prob))).floor() as i32;
        }
        let analysis_bandwidth = analysis_info.bandwidth;
        if analysis_bandwidth <= 12 {
            st.detected_bandwidth = OPUS_BANDWIDTH_NARROWBAND;
        } else if analysis_bandwidth <= 14 {
            st.detected_bandwidth = OPUS_BANDWIDTH_MEDIUMBAND;
        } else if analysis_bandwidth <= 16 {
            st.detected_bandwidth = OPUS_BANDWIDTH_WIDEBAND;
        } else if analysis_bandwidth <= 18 {
            st.detected_bandwidth = OPUS_BANDWIDTH_SUPERWIDEBAND;
        } else {
            st.detected_bandwidth = OPUS_BANDWIDTH_FULLBAND;
        }
    }
    let stereo_width = if st.channels == 2 && st.force_channels != 1 {
        compute_stereo_width(&pcm, frame_size, st.Fs, &mut st.width_mem)
    } else {
        0.
    };
    let total_buffer = delay_compensation;
    st.bitrate_bps = user_bitrate_to_bitrate(st, frame_size, max_data_bytes);
    let mut frame_rate = st.Fs / frame_size;
    if st.use_vbr == 0 {
        // Multiply by 12 to make sure the division is exact.
        let frame_rate12: i32 = 12 * st.Fs / frame_size;
        // We need to make sure that "int" values always fit in 16 bits.
        let cbrBytes =
            ((12 * st.bitrate_bps / 8 + frame_rate12 / 2) / frame_rate12).min(max_data_bytes);
        st.bitrate_bps = cbrBytes * frame_rate12 * 8 / 12;
        // Make sure we provide at least one byte to avoid failing.
        max_data_bytes = cbrBytes.max(1);
    }
    if max_data_bytes < 3
        || st.bitrate_bps < 3 * frame_rate * 8
        || frame_rate < 50 && (max_data_bytes * frame_rate < 300 || st.bitrate_bps < 2400)
    {
        // If the space is too low to do something useful, emit 'PLC' frames.
        let mut tocmode: i32 = st.mode;
        let mut bw: i32 = if st.bandwidth == 0 {
            OPUS_BANDWIDTH_NARROWBAND
        } else {
            st.bandwidth
        };
        let mut packet_code: i32 = 0;
        let mut num_multiframes: i32 = 0;
        if tocmode == 0 {
            tocmode = MODE_SILK_ONLY;
        }
        if frame_rate > 100 {
            tocmode = MODE_CELT_ONLY;
        }
        // 40 ms -> 2 x 20 ms if in CELT_ONLY or HYBRID mode
        if frame_rate == 25 && tocmode != MODE_SILK_ONLY {
            frame_rate = 50;
            packet_code = 1;
        }
        // >= 60 ms frames
        if frame_rate <= 16 {
            // 1 x 60 ms, 2 x 40 ms, 2 x 60 ms
            if out_data_bytes == 1 || tocmode == MODE_SILK_ONLY && frame_rate != 10 {
                tocmode = MODE_SILK_ONLY;

                packet_code = (frame_rate <= 12) as i32;
                frame_rate = if frame_rate == 12 { 25 } else { 16 };
            } else {
                num_multiframes = 50 / frame_rate;
                frame_rate = 50;
                packet_code = 3;
            }
        }

        if tocmode == MODE_SILK_ONLY && bw > OPUS_BANDWIDTH_WIDEBAND {
            bw = OPUS_BANDWIDTH_WIDEBAND;
        } else if tocmode == MODE_CELT_ONLY && bw == OPUS_BANDWIDTH_MEDIUMBAND {
            bw = OPUS_BANDWIDTH_NARROWBAND;
        } else if tocmode == MODE_HYBRID && bw <= OPUS_BANDWIDTH_SUPERWIDEBAND {
            bw = OPUS_BANDWIDTH_SUPERWIDEBAND;
        }

        og_data[0] = gen_toc(tocmode, frame_rate, bw, st.stream_channels);
        og_data[0] |= packet_code as u8;

        ret = if packet_code <= 1 { 1 } else { 2 };

        max_data_bytes = if max_data_bytes > ret {
            max_data_bytes
        } else {
            ret
        };
        if packet_code == 3 {
            og_data[1] = num_multiframes as u8;
        }
        if st.use_vbr == 0 {
            ret = opus_packet_pad(&mut og_data[..max_data_bytes as _], ret, max_data_bytes);
            if ret == OPUS_OK {
                ret = max_data_bytes;
            } else {
                ret = OPUS_INTERNAL_ERROR;
            }
        }
        return ret;
    }

    // Max bitrate we're allowed to use
    let max_rate = frame_rate * max_data_bytes * 8;

    // Equivalent 20-ms rate for mode/channel/bandwidth decisions
    let mut equiv_rate = compute_equiv_rate(
        st.bitrate_bps,
        st.channels,
        st.Fs / frame_size,
        st.use_vbr,
        0,
        st.silk_mode.complexity,
        st.silk_mode.packetLossPercentage,
    );

    // Probability of voice in Q7
    let voice_est = if st.signal_type == OPUS_SIGNAL_VOICE {
        127
    } else if st.signal_type == OPUS_SIGNAL_MUSIC {
        0
    } else if st.voice_ratio >= 0 {
        let voice_est = st.voice_ratio * 327 >> 8;
        // For AUDIO, never be more than 90% confident of having speech
        if st.application == OPUS_APPLICATION_AUDIO {
            voice_est.min(115)
        } else {
            voice_est
        }
    } else if st.application == OPUS_APPLICATION_VOIP {
        115
    } else {
        48
    };
    if st.force_channels != OPUS_AUTO && st.channels == 2 {
        st.stream_channels = st.force_channels;
    } else {
        // Rate-dependent mono-stereo decision
        if st.channels == 2 {
            let mut stereo_threshold = stereo_music_threshold
                + (voice_est * voice_est * (stereo_voice_threshold - stereo_music_threshold) >> 14);
            if st.stream_channels == 2 {
                stereo_threshold -= 1000;
            } else {
                stereo_threshold += 1000;
            }
            st.stream_channels = if equiv_rate > stereo_threshold { 2 } else { 1 };
        } else {
            st.stream_channels = st.channels;
        }
    }

    // Update equivalent rate for channels decision.
    equiv_rate = compute_equiv_rate(
        st.bitrate_bps,
        st.stream_channels,
        st.Fs / frame_size,
        st.use_vbr,
        0,
        st.silk_mode.complexity,
        st.silk_mode.packetLossPercentage,
    );

    // Allow SILK DTX if DTX is enabled but the generalized DTX cannot be used,
    //  e.g. because of the complexity setting or sample rate.
    st.silk_mode.useDTX =
        (st.use_dtx != 0 && !(analysis_info.valid != 0 || is_silence != 0)) as i32;

    // Mode selection depending on application and signal type
    if st.application == OPUS_APPLICATION_RESTRICTED_LOWDELAY {
        st.mode = MODE_CELT_ONLY;
    } else if st.user_forced_mode == OPUS_AUTO {
        // Interpolate based on stereo width
        let mode_voice = ((1.0 - stereo_width) * mode_thresholds[0][0] as f32
            + stereo_width * mode_thresholds[1][0] as f32) as i32;
        let mode_music = ((1.0 - stereo_width) * mode_thresholds[1][1] as f32
            + stereo_width * mode_thresholds[1][1] as f32) as i32;
        // Interpolate based on speech/music probability
        let mut threshold = mode_music + (voice_est * voice_est * (mode_voice - mode_music) >> 14);
        // Bias towards SILK for VoIP because of some useful features
        if st.application == OPUS_APPLICATION_VOIP {
            threshold += 8000;
        }

        // Hysteresis
        if st.prev_mode == MODE_CELT_ONLY {
            threshold -= 4000;
        } else if st.prev_mode > 0 {
            threshold += 4000;
        }

        st.mode = if equiv_rate >= threshold {
            MODE_CELT_ONLY
        } else {
            MODE_SILK_ONLY
        };
        // When FEC is enabled and there's enough packet loss, use SILK
        if st.silk_mode.useInBandFEC != 0
            && st.silk_mode.packetLossPercentage > 128 - voice_est >> 4
        {
            st.mode = MODE_SILK_ONLY;
        }
        // When encoding voice and DTX is enabled but the generalized DTX cannot be used,
        // use SILK in order to make use of its DTX.
        if st.silk_mode.useDTX != 0 && voice_est > 100 {
            st.mode = MODE_SILK_ONLY;
        }
        // If max_data_bytes represents less than 6 kb/s, switch to CELT-only mode
        if max_data_bytes < (if frame_rate > 50 { 9000 } else { 6000 }) * frame_size / (st.Fs * 8) {
            st.mode = MODE_CELT_ONLY;
        }
    } else {
        st.mode = st.user_forced_mode;
    }

    // Override the chosen mode to make sure we meet the requested frame size
    if st.mode != MODE_CELT_ONLY && frame_size < st.Fs / 100 {
        st.mode = MODE_CELT_ONLY;
    }
    if st.lfe != 0 {
        st.mode = MODE_CELT_ONLY;
    }
    if st.prev_mode > 0
        && (st.mode != MODE_CELT_ONLY && st.prev_mode == MODE_CELT_ONLY
            || st.mode == MODE_CELT_ONLY && st.prev_mode != MODE_CELT_ONLY)
    {
        redundancy = 1;
        celt_to_silk = (st.mode != MODE_CELT_ONLY) as i32;
        if celt_to_silk == 0 {
            if frame_size >= st.Fs / 100 {
                st.mode = st.prev_mode;
                to_celt = 1;
            } else {
                redundancy = 0;
            }
        }
    }
    // When encoding multiframes, we can ask for a switch to CELT only in the last frame. This switch
    // is processed above as the requested mode shouldn't interrupt stereo->mono transition.
    if st.stream_channels == 1
        && st.prev_channels == 2
        && st.silk_mode.toMono == 0
        && st.mode != MODE_CELT_ONLY
        && st.prev_mode != MODE_CELT_ONLY
    {
        st.silk_mode.toMono = 1;
        st.stream_channels = 2;
    } else {
        st.silk_mode.toMono = 0;
    }

    // Update equivalent rate with mode decision.
    equiv_rate = compute_equiv_rate(
        st.bitrate_bps,
        st.stream_channels,
        st.Fs / frame_size,
        st.use_vbr,
        st.mode,
        st.silk_mode.complexity,
        st.silk_mode.packetLossPercentage,
    );

    // Automatic (rate-dependent) bandwidth selection
    if st.mode != MODE_CELT_ONLY && st.prev_mode == MODE_CELT_ONLY {
        let (a, _dummy) = silk_InitEncoder(st.arch);
        st.silk_enc = a;
        prefill = 1;
    }

    // Automatic (rate-dependent) bandwidth selection
    if st.mode == MODE_CELT_ONLY || st.first != 0 || st.silk_mode.allowBandwidthSwitch != 0 {
        let voice_bandwidth_thresholds;
        let music_bandwidth_thresholds;
        let mut bandwidth_thresholds: [i32; 8] = [0; 8];
        let mut bandwidth: i32 = OPUS_BANDWIDTH_FULLBAND;
        if st.channels == 2 && st.force_channels != 1 {
            voice_bandwidth_thresholds = stereo_voice_bandwidth_thresholds;
            music_bandwidth_thresholds = stereo_music_bandwidth_thresholds;
        } else {
            voice_bandwidth_thresholds = mono_voice_bandwidth_thresholds;
            music_bandwidth_thresholds = mono_music_bandwidth_thresholds;
        }
        // Interpolate bandwidth thresholds depending on voice estimation
        for i in 0..8 {
            bandwidth_thresholds[i] = music_bandwidth_thresholds[i]
                + (voice_est
                    * voice_est
                    * (voice_bandwidth_thresholds[i] - music_bandwidth_thresholds[i])
                    >> 14);
        }
        loop {
            let mut threshold_0 =
                bandwidth_thresholds[(2 * (bandwidth - OPUS_BANDWIDTH_MEDIUMBAND)) as usize];
            let hysteresis =
                bandwidth_thresholds[(2 * (bandwidth - OPUS_BANDWIDTH_MEDIUMBAND) + 1) as usize];
            if st.first == 0 {
                if st.auto_bandwidth >= bandwidth {
                    threshold_0 -= hysteresis;
                } else {
                    threshold_0 += hysteresis;
                }
            }
            if equiv_rate >= threshold_0 {
                break;
            }
            bandwidth -= 1;
            if !(bandwidth > OPUS_BANDWIDTH_NARROWBAND) {
                break;
            }
        }
        // We don't use mediumband anymore, except when explicitly requested or during
        //  mode transitions.
        if bandwidth == OPUS_BANDWIDTH_MEDIUMBAND {
            bandwidth = OPUS_BANDWIDTH_WIDEBAND;
        }
        st.auto_bandwidth = bandwidth;
        st.bandwidth = st.auto_bandwidth;

        // Prevents any transition to SWB/FB until the SILK layer has fully
        //  switched to WB mode and turned the variable LP filter off
        if st.first == 0
            && st.mode != MODE_CELT_ONLY
            && st.silk_mode.inWBmodeWithoutVariableLP == 0
            && st.bandwidth > OPUS_BANDWIDTH_WIDEBAND
        {
            st.bandwidth = OPUS_BANDWIDTH_WIDEBAND;
        }
    }
    if st.bandwidth > st.max_bandwidth {
        st.bandwidth = st.max_bandwidth;
    }
    if st.user_bandwidth != OPUS_AUTO {
        st.bandwidth = st.user_bandwidth;
    }
    // This prevents us from using hybrid at unsafe CBR/max rates
    if st.mode != MODE_CELT_ONLY && max_rate < 15000 {
        st.bandwidth = if st.bandwidth < 1103 {
            st.bandwidth
        } else {
            1103
        };
    }
    // Prevents Opus from wasting bits on frequencies that are above
    // the Nyquist rate of the input signal
    if st.Fs <= 24000 && st.bandwidth > OPUS_BANDWIDTH_SUPERWIDEBAND {
        st.bandwidth = OPUS_BANDWIDTH_SUPERWIDEBAND;
    }
    if st.Fs <= 16000 && st.bandwidth > OPUS_BANDWIDTH_WIDEBAND {
        st.bandwidth = OPUS_BANDWIDTH_WIDEBAND;
    }
    if st.Fs <= 12000 && st.bandwidth > OPUS_BANDWIDTH_MEDIUMBAND {
        st.bandwidth = OPUS_BANDWIDTH_MEDIUMBAND;
    }
    if st.Fs <= 8000 && st.bandwidth > OPUS_BANDWIDTH_NARROWBAND {
        st.bandwidth = OPUS_BANDWIDTH_NARROWBAND;
    }

    // Use detected bandwidth to reduce the encoded bandwidth.
    if st.detected_bandwidth != 0 && st.user_bandwidth == OPUS_AUTO {
        // Makes bandwidth detection more conservative just in case the detector
        // gets it wrong when we could have coded a high bandwidth transparently.
        // When operating in SILK/hybrid mode, we don't go below wideband to avoid
        // more complicated switches that require redundancy.
        let min_detected_bandwidth =
            if equiv_rate <= 18000 * st.stream_channels && st.mode == MODE_CELT_ONLY {
                OPUS_BANDWIDTH_NARROWBAND
            } else if equiv_rate <= 24000 * st.stream_channels && st.mode == MODE_CELT_ONLY {
                OPUS_BANDWIDTH_MEDIUMBAND
            } else if equiv_rate <= 30000 * st.stream_channels {
                OPUS_BANDWIDTH_WIDEBAND
            } else if equiv_rate <= 44000 * st.stream_channels {
                OPUS_BANDWIDTH_SUPERWIDEBAND
            } else {
                OPUS_BANDWIDTH_FULLBAND
            };
        st.detected_bandwidth = if st.detected_bandwidth > min_detected_bandwidth {
            st.detected_bandwidth
        } else {
            min_detected_bandwidth
        };
        st.bandwidth = if st.bandwidth < st.detected_bandwidth {
            st.bandwidth
        } else {
            st.detected_bandwidth
        };
    }
    st.silk_mode.LBRR_coded = decide_fec(
        st.silk_mode.useInBandFEC,
        st.silk_mode.packetLossPercentage,
        st.silk_mode.LBRR_coded,
        st.mode,
        &mut st.bandwidth,
        equiv_rate,
    );
    opus_custom_encoder_ctl!(&mut st.celt_enc, OPUS_SET_LSB_DEPTH_REQUEST, lsb_depth);

    // CELT mode doesn't support mediumband, use wideband instead
    if st.mode == MODE_CELT_ONLY && st.bandwidth == OPUS_BANDWIDTH_MEDIUMBAND {
        st.bandwidth = OPUS_BANDWIDTH_WIDEBAND;
    }
    if st.lfe != 0 {
        st.bandwidth = OPUS_BANDWIDTH_NARROWBAND;
    }
    let mut curr_bandwidth = st.bandwidth;
    // Chooses the appropriate mode for speech
    //  *NEVER* switch to/from CELT-only mode here as this will invalidate some assumptions
    if st.mode == MODE_SILK_ONLY && curr_bandwidth > OPUS_BANDWIDTH_WIDEBAND {
        st.mode = MODE_HYBRID;
    }
    if st.mode == MODE_HYBRID && curr_bandwidth <= OPUS_BANDWIDTH_WIDEBAND {
        st.mode = MODE_SILK_ONLY;
    }
    // Can't support higher than >60 ms frames, and >20 ms when in Hybrid or CELT-only modes
    if frame_size > st.Fs / 50 && st.mode != MODE_SILK_ONLY || frame_size > 3 * st.Fs / 50 {
        let enc_frame_size = if st.mode == MODE_SILK_ONLY {
            if frame_size == 2 * st.Fs / 25 {
                st.Fs / 25
            } else if frame_size == 3 * st.Fs / 25 {
                3 * st.Fs / 50
            } else {
                st.Fs / 50
            }
        } else {
            st.Fs / 50
        };
        let nb_frames = frame_size / enc_frame_size;
        if analysis_read_pos_bak != -1 {
            st.analysis.read_pos = analysis_read_pos_bak;
            st.analysis.read_subframe = analysis_read_subframe_bak;
        }
        ret = encode_multiframe_packet(
            st,
            pcm,
            nb_frames,
            enc_frame_size,
            og_data,
            out_data_bytes,
            to_celt,
            lsb_depth,
            float_api,
        );
        return ret;
    }

    // For the first frame at a new SILK bandwidth
    if st.silk_bw_switch != 0 {
        redundancy = 1;
        celt_to_silk = 1;
        st.silk_bw_switch = 0;
        // Do a prefill without reseting the sampling rate control.
        prefill = 2;
    }
    // If we decided to go with CELT, make sure redundancy is off, no matter what
    // we decided earlier.
    if st.mode == MODE_CELT_ONLY {
        redundancy = 0;
    }
    if redundancy != 0 {
        redundancy_bytes = compute_redundancy_bytes(
            max_data_bytes,
            st.bitrate_bps,
            frame_rate,
            st.stream_channels,
        );
        if redundancy_bytes == 0 {
            redundancy = 0;
        }
    }
    let bytes_target =
        (if max_data_bytes - redundancy_bytes < st.bitrate_bps * frame_size / (st.Fs * 8) {
            max_data_bytes - redundancy_bytes
        } else {
            st.bitrate_bps * frame_size / (st.Fs * 8)
        }) - 1;

    let (data0, data) = og_data.split_at_mut(1);
    let mut enc = ec_enc_init(&mut data[..max_data_bytes as usize - 1]);

    let mut pcm_buf = vec![0.; ((total_buffer + frame_size) * st.channels) as usize];
    {
        // OPUS_COPY(
        //   pcm_buf,
        //   &st->delay_buffer[(st->encoder_buffer-total_buffer)*st->channels],
        //   total_buffer*st->channels
        // );
        let start = ((st.encoder_buffer - total_buffer) * st.channels) as usize;
        let len = (total_buffer * st.channels) as usize;
        let end = start + len;
        pcm_buf[..len].copy_from_slice(&st.delay_buffer[start..end]);
    }

    let hp_freq_smth1 = if st.mode == MODE_CELT_ONLY {
        ((silk_lin2log(VARIABLE_HP_MIN_CUTOFF_HZ) as u32) << 8) as i32
    } else {
        st.silk_enc.state_Fxx[0].sCmn.variable_HP_smth1_Q15
    };
    st.variable_HP_smth2_Q15 = (st.variable_HP_smth2_Q15 as i64
        + ((hp_freq_smth1 - st.variable_HP_smth2_Q15) as i64
            * ((VARIABLE_HP_SMTH_COEF2 * ((1) << 16) as f32) as f64 + 0.5f64) as i32 as i16 as i64
            >> 16)) as i32;

    // convert from log scale to Hertz
    let cutoff_Hz = silk_log2lin(st.variable_HP_smth2_Q15 >> 8);

    if st.application == OPUS_APPLICATION_VOIP {
        hp_cutoff(
            &pcm,
            cutoff_Hz,
            &mut pcm_buf[(total_buffer * st.channels) as usize..],
            &mut st.hp_mem,
            frame_size,
            st.channels,
            st.Fs,
            st.arch,
        );
    } else {
        dc_reject(
            &pcm,
            3,
            &mut pcm_buf[(total_buffer * st.channels) as usize..],
            &mut st.hp_mem,
            frame_size,
            st.channels,
            st.Fs,
        );
    }
    if float_api != 0 {
        let sum = celt_inner_prod_c(
            &pcm_buf[(total_buffer * st.channels) as usize..],
            &pcm_buf[(total_buffer * st.channels) as usize..],
            frame_size * st.channels,
        );
        // This should filter out both NaNs and ridiculous signals that could
        // cause NaNs further down.
        if !(sum < 1e9f32) || sum != sum {
            let start = (total_buffer * st.channels) as usize;
            let end = start + (frame_size * st.channels) as usize;
            pcm_buf[start..end].fill(0.);

            st.hp_mem[0] = 0.;
            st.hp_mem[1] = 0.;
            st.hp_mem[2] = 0.;
            st.hp_mem[3] = 0.;
        }
    }

    // SILK processing
    let mut HB_gain = Q15ONE;
    if st.mode != MODE_CELT_ONLY {
        let mut pcm_silk = vec![0; (st.channels * frame_size) as usize];

        let mut activity = VAD_NO_DECISION;
        if analysis_info.valid != 0 {
            // Inform SILK about the Opus VAD decision
            activity = (analysis_info.activity_probability >= DTX_ACTIVITY_THRESHOLD) as i32;
        }
        // Distribute bits between SILK and CELT
        let total_bitRate = 8 * bytes_target * frame_rate;
        if st.mode == MODE_HYBRID {
            // Base rate for SILK
            st.silk_mode.bitRate = compute_silk_rate_for_hybrid(
                total_bitRate,
                curr_bandwidth,
                (st.Fs == 50 * frame_size) as i32,
                st.use_vbr,
                st.silk_mode.LBRR_coded,
                st.stream_channels,
            );
            if (st.energy_masking).is_null() {
                // Increasingly attenuate high band when it gets allocated fewer bits
                let celt_rate = total_bitRate - st.silk_mode.bitRate;
                HB_gain = Q15ONE - celt_exp2(-celt_rate as f32 * (1.0f32 / 1024f32));
            }
        } else {
            // SILK gets all bits
            st.silk_mode.bitRate = total_bitRate;
        }
        // Surround masking for SILK
        if !(st.energy_masking).is_null() && st.use_vbr != 0 && st.lfe == 0 {
            let mut mask_sum: f32 = 0.;
            let mut end: i32 = 17;
            let mut srate: i16 = 16000;
            if st.bandwidth == OPUS_BANDWIDTH_NARROWBAND {
                end = 13;
                srate = 8000;
            } else if st.bandwidth == OPUS_BANDWIDTH_MEDIUMBAND {
                end = 15;
                srate = 12000;
            }
            for c in 0..st.channels {
                for i in 0..end {
                    let a = unsafe { *(st.energy_masking).offset((21 * c + i) as isize) };

                    let mut mask = (a.min(0.5)).max(-2.0);
                    if mask > 0. {
                        mask = 0.5 * mask;
                    }
                    mask_sum += mask;
                }
            }
            // Conservative rate reduction, we cut the masking in half
            let mut masking_depth = mask_sum / end as f32 * st.channels as f32;
            masking_depth += 0.2;
            let mut rate_offset = (srate as f32 * masking_depth) as i32;
            rate_offset = rate_offset.max(-2 * st.silk_mode.bitRate / 3);
            // Split the rate change between the SILK and CELT part for hybrid.
            if st.bandwidth == OPUS_BANDWIDTH_SUPERWIDEBAND
                || st.bandwidth == OPUS_BANDWIDTH_FULLBAND
            {
                st.silk_mode.bitRate += 3 * rate_offset / 5;
            } else {
                st.silk_mode.bitRate += rate_offset;
            }
        }

        st.silk_mode.payloadSize_ms = 1000 * frame_size / st.Fs;
        st.silk_mode.nChannelsAPI = st.channels;
        st.silk_mode.nChannelsInternal = st.stream_channels;
        if curr_bandwidth == OPUS_BANDWIDTH_NARROWBAND {
            st.silk_mode.desiredInternalSampleRate = 8000;
        } else if curr_bandwidth == OPUS_BANDWIDTH_MEDIUMBAND {
            st.silk_mode.desiredInternalSampleRate = 12000;
        } else {
            assert!(st.mode == 1001 || curr_bandwidth == 1103);
            st.silk_mode.desiredInternalSampleRate = 16000;
        }
        if st.mode == MODE_HYBRID {
            // Don't allow bandwidth reduction at lowest bitrates in hybrid mode
            st.silk_mode.minInternalSampleRate = 16000;
        } else {
            st.silk_mode.minInternalSampleRate = 8000;
        }

        st.silk_mode.maxInternalSampleRate = 16000;
        if st.mode == MODE_SILK_ONLY {
            let mut effective_max_rate: i32 = max_rate;
            if frame_rate > 50 {
                effective_max_rate = effective_max_rate * 2 / 3;
            }
            if effective_max_rate < 8000 {
                st.silk_mode.maxInternalSampleRate = 12000;
                st.silk_mode.desiredInternalSampleRate =
                    if (12000) < st.silk_mode.desiredInternalSampleRate {
                        12000
                    } else {
                        st.silk_mode.desiredInternalSampleRate
                    };
            }
            if effective_max_rate < 7000 {
                st.silk_mode.maxInternalSampleRate = 8000;
                st.silk_mode.desiredInternalSampleRate =
                    if (8000) < st.silk_mode.desiredInternalSampleRate {
                        8000
                    } else {
                        st.silk_mode.desiredInternalSampleRate
                    };
            }
        }
        st.silk_mode.useCBR = (st.use_vbr == 0) as i32;
        // Call SILK encoder for the low band

        // Max bits for SILK, counting ToC, redundancy bytes, and optionally redundancy.
        st.silk_mode.maxBits = (max_data_bytes - 1) * 8;
        if redundancy != 0 && redundancy_bytes >= 2 {
            // Counting 1 bit for redundancy position and 20 bits for flag+size (only for hybrid).
            st.silk_mode.maxBits -= redundancy_bytes * 8 + 1;
            if st.mode == MODE_HYBRID {
                st.silk_mode.maxBits -= 20;
            }
        }
        if st.silk_mode.useCBR != 0 {
            if st.mode == MODE_HYBRID {
                st.silk_mode.maxBits = st
                    .silk_mode
                    .maxBits
                    .min(st.silk_mode.bitRate * frame_size / st.Fs);
            }
        } else {
            // Constrained VBR.
            if st.mode == MODE_HYBRID {
                // Compute SILK bitrate corresponding to the max total bits available
                let maxBitRate: i32 = compute_silk_rate_for_hybrid(
                    st.silk_mode.maxBits * st.Fs / frame_size,
                    curr_bandwidth,
                    (st.Fs == 50 * frame_size) as i32,
                    st.use_vbr,
                    st.silk_mode.LBRR_coded,
                    st.stream_channels,
                );
                st.silk_mode.maxBits = maxBitRate * frame_size / st.Fs;
            }
        }
        if prefill != 0 {
            let mut zero: i32 = 0;
            // Use a smooth onset for the SILK prefill to avoid the encoder trying to encode
            // a discontinuity. The exact location is what we need to avoid leaving any "gap"
            // in the audio when mixing with the redundant CELT frame. Here we can afford to
            // overwrite st->delay_buffer because the only thing that uses it before it gets
            // rewritten is tmp_prefill[] and even then only the part after the ramp really
            // gets used (rather than sent to the encoder and discarded)
            let prefill_offset =
                st.channels * (st.encoder_buffer - st.delay_compensation - st.Fs / 400);
            gain_fade(
                &mut st.delay_buffer[prefill_offset as usize..],
                0.,
                Q15ONE,
                unsafe { (*celt_mode).overlap as i32 },
                st.Fs / 400,
                st.channels,
                unsafe { (*celt_mode).window },
                st.Fs,
            );
            st.delay_buffer[..prefill_offset as usize].fill(0.);

            for i in 0..st.encoder_buffer * st.channels {
                pcm_silk[i as usize] = FLOAT2INT16(st.delay_buffer[i as usize]);
            }
            unsafe {
                silk_Encode(
                    &mut st.silk_enc,
                    &mut st.silk_mode,
                    pcm_silk.as_mut_ptr(),
                    st.encoder_buffer,
                    None,
                    &mut zero,
                    prefill,
                    activity,
                );
            }
            // Prevent a second switch in the real encode call.
            st.silk_mode.opusCanSwitch = 0;
        }
        for i in 0..frame_size * st.channels {
            pcm_silk[i as usize] = FLOAT2INT16(pcm_buf[(total_buffer * st.channels + i) as usize]);
        }
        ret = unsafe {
            silk_Encode(
                &mut st.silk_enc,
                &mut st.silk_mode,
                pcm_silk.as_mut_ptr(),
                frame_size,
                Some(&mut enc),
                &mut nBytes,
                0,
                activity,
            )
        };
        if ret != 0 {
            return OPUS_INTERNAL_ERROR;
        }
        // Extract SILK internal bandwidth for signaling in first byte
        if st.mode == MODE_SILK_ONLY {
            if st.silk_mode.internalSampleRate == 8000 {
                curr_bandwidth = OPUS_BANDWIDTH_NARROWBAND;
            } else if st.silk_mode.internalSampleRate == 12000 {
                curr_bandwidth = OPUS_BANDWIDTH_MEDIUMBAND;
            } else if st.silk_mode.internalSampleRate == 16000 {
                curr_bandwidth = OPUS_BANDWIDTH_WIDEBAND;
            }
        } else {
            assert!(st.silk_mode.internalSampleRate == 16000)
        };
        st.silk_mode.opusCanSwitch =
            (st.silk_mode.switchReady != 0 && st.nonfinal_frame == 0) as i32;
        if nBytes == 0 {
            st.rangeFinal = 0;
            data0[0] = gen_toc(
                st.mode,
                st.Fs / frame_size,
                curr_bandwidth,
                st.stream_channels,
            );
            return 1;
        }
        if st.silk_mode.opusCanSwitch != 0 {
            redundancy_bytes = compute_redundancy_bytes(
                max_data_bytes,
                st.bitrate_bps,
                frame_rate,
                st.stream_channels,
            );
            redundancy = (redundancy_bytes != 0) as i32;
            celt_to_silk = 0;
            st.silk_bw_switch = 1;
        }
    }

    // CELT processing
    {
        let mut endband: i32 = 21;
        match curr_bandwidth {
            OPUS_BANDWIDTH_NARROWBAND => {
                endband = 13;
            }
            OPUS_BANDWIDTH_MEDIUMBAND | OPUS_BANDWIDTH_WIDEBAND => {
                endband = 17;
            }
            OPUS_BANDWIDTH_SUPERWIDEBAND => {
                endband = 19;
            }
            OPUS_BANDWIDTH_FULLBAND => {
                endband = 21;
            }
            _ => {}
        }
        opus_custom_encoder_ctl!(&mut st.celt_enc, CELT_SET_END_BAND_REQUEST, endband);
        opus_custom_encoder_ctl!(
            &mut st.celt_enc,
            CELT_SET_CHANNELS_REQUEST,
            st.stream_channels
        );
    }
    opus_custom_encoder_ctl!(&mut st.celt_enc, OPUS_SET_BITRATE_REQUEST, -1);
    if st.mode != MODE_SILK_ONLY {
        let mut celt_pred: f32 = 2.;
        opus_custom_encoder_ctl!(&mut st.celt_enc, OPUS_SET_VBR_REQUEST, 0);
        // We may still decide to disable prediction later
        if st.silk_mode.reducedDependency != 0 {
            celt_pred = 0.;
        }
        opus_custom_encoder_ctl!(
            &mut st.celt_enc,
            CELT_SET_PREDICTION_REQUEST,
            celt_pred as i32
        );

        if st.mode == MODE_HYBRID {
            if st.use_vbr != 0 {
                opus_custom_encoder_ctl!(
                    &mut st.celt_enc,
                    OPUS_SET_BITRATE_REQUEST,
                    st.bitrate_bps - st.silk_mode.bitRate,
                );
                opus_custom_encoder_ctl!(&mut st.celt_enc, OPUS_SET_VBR_CONSTRAINT_REQUEST, 0,);
            }
        } else if st.use_vbr != 0 {
            opus_custom_encoder_ctl!(&mut st.celt_enc, OPUS_SET_VBR_REQUEST, 1);
            opus_custom_encoder_ctl!(
                &mut st.celt_enc,
                OPUS_SET_VBR_CONSTRAINT_REQUEST,
                st.vbr_constraint,
            );
            opus_custom_encoder_ctl!(&mut st.celt_enc, OPUS_SET_BITRATE_REQUEST, st.bitrate_bps);
        }
    }

    let mut tmp_prefill = vec![0.; (st.channels * st.Fs / 400) as usize];
    if st.mode != MODE_SILK_ONLY && st.mode != st.prev_mode && st.prev_mode > 0 {
        // OPUS_COPY(
        //   tmp_prefill,
        //   &st->delay_buffer[(st->encoder_buffer-total_buffer-st->Fs/400)*st->channels],
        //   st->channels*st->Fs/400
        // );
        let start = ((st.encoder_buffer - total_buffer - st.Fs / 400) * st.channels) as usize;
        let len = (st.channels * st.Fs / 400) as usize;
        let end = start + len;
        tmp_prefill[..len].copy_from_slice(&st.delay_buffer[start..end])
    }
    if st.channels * (st.encoder_buffer - (frame_size + total_buffer)) > 0 {
        {
            // OPUS_MOVE(
            //   st->delay_buffer,
            //   &st->delay_buffer[st->channels*frame_size],
            //   st->channels*(st->encoder_buffer-frame_size-total_buffer)
            // );
            let start = (st.channels * frame_size) as usize;
            let len = (st.channels * (st.encoder_buffer - frame_size - total_buffer)) as usize;
            let end = start + len;
            st.delay_buffer.copy_within(start..end, 0);
        }
        {
            // OPUS_COPY(
            //   &st->delay_buffer[st->channels*(st->encoder_buffer-frame_size-total_buffer)],
            //   &pcm_buf[0],
            //   (frame_size+total_buffer)*st->channels
            // );
            let start = (st.channels * (st.encoder_buffer - frame_size - total_buffer)) as usize;
            let len = ((frame_size + total_buffer) * st.channels) as usize;
            let end = start + len;
            st.delay_buffer[start..end].copy_from_slice(&pcm_buf[..len])
        }
    } else {
        // OPUS_COPY(
        //   st->delay_buffer,
        //   &pcm_buf[(frame_size+total_buffer-st->encoder_buffer)*st->channels],
        //   st->encoder_buffer*st->channels
        // );
        let start = ((frame_size + total_buffer - st.encoder_buffer) * st.channels) as usize;
        let len = (st.encoder_buffer * st.channels) as usize;
        let end = start + len;
        st.delay_buffer[..len].copy_from_slice(&pcm_buf[start..end]);
    }

    // gain_fade() and stereo_fade() need to be after the buffer copying
    // because we don't want any of this to affect the SILK part
    if st.prev_HB_gain < Q15ONE || HB_gain < Q15ONE {
        let mode = unsafe { *celt_mode };
        gain_fade(
            &mut pcm_buf,
            st.prev_HB_gain,
            HB_gain,
            mode.overlap as i32,
            frame_size,
            st.channels,
            mode.window,
            st.Fs,
        );
    }
    st.prev_HB_gain = HB_gain;
    if st.mode != MODE_HYBRID || st.stream_channels == 1 {
        if equiv_rate > 32000 {
            st.silk_mode.stereoWidth_Q14 = 16384;
        } else if equiv_rate < 16000 {
            st.silk_mode.stereoWidth_Q14 = 0;
        } else {
            st.silk_mode.stereoWidth_Q14 =
                16384 - 2048 * (32000 - equiv_rate) / (equiv_rate - 14000);
        }
    }
    if (st.energy_masking).is_null() && st.channels == 2 {
        // Apply stereo width reduction (at low bitrates)
        if (st.hybrid_stereo_width_Q14 as i32) < 1 << 14 || st.silk_mode.stereoWidth_Q14 < 1 << 14 {
            let mut g1 = st.hybrid_stereo_width_Q14 as f32;
            let mut g2 = st.silk_mode.stereoWidth_Q14 as f32;
            g1 *= 1.0 / 16384.;
            g2 *= 1.0 / 16384.;
            stereo_fade(
                &mut pcm_buf,
                g1,
                g2,
                unsafe { (*celt_mode).overlap as i32 },
                frame_size,
                st.channels,
                unsafe { (*celt_mode).window },
                st.Fs,
            );
            st.hybrid_stereo_width_Q14 = st.silk_mode.stereoWidth_Q14 as i16;
        }
    }
    if st.mode != MODE_CELT_ONLY
        && ec_tell(&mut enc) + 17 + 20 * (st.mode == MODE_HYBRID) as i32 <= 8 * (max_data_bytes - 1)
    {
        // For SILK mode, the redundancy is inferred from the length
        if st.mode == MODE_HYBRID {
            ec_enc_bit_logp(&mut enc, redundancy, 12);
        }
        if redundancy != 0 {
            ec_enc_bit_logp(&mut enc, celt_to_silk, 1);
            let max_redundancy = if st.mode == MODE_HYBRID {
                max_data_bytes - 1 - (ec_tell(&mut enc) + 8 + 3 + 7 >> 3)
            } else {
                max_data_bytes - 1 - (ec_tell(&mut enc) + 7 >> 3)
            };
            // Reserve the 8 bits needed for the redundancy length,
            // and at least a few bits for CELT if possible
            redundancy_bytes = max_redundancy.min(redundancy_bytes);
            redundancy_bytes = 257.min(2.max(redundancy_bytes));

            if st.mode == MODE_HYBRID {
                ec_enc_uint(&mut enc, (redundancy_bytes - 2) as u32, 256);
            }
        }
    } else {
        redundancy = 0;
    }
    if redundancy == 0 {
        st.silk_bw_switch = 0;
        redundancy_bytes = 0;
    }
    if st.mode != MODE_CELT_ONLY {
        start_band = 17;
    }
    let mut nb_compr_bytes;
    if st.mode == MODE_SILK_ONLY {
        ret = ec_tell(&mut enc) + 7 >> 3;
        ec_enc_done(&mut enc);
        nb_compr_bytes = ret;
    } else {
        nb_compr_bytes = max_data_bytes - 1 - redundancy_bytes;
        ec_enc_shrink(&mut enc, nb_compr_bytes as u32);
    }
    if redundancy != 0 || st.mode != MODE_SILK_ONLY {
        opus_custom_encoder_ctl!(
            &mut st.celt_enc,
            CELT_SET_ANALYSIS_REQUEST,
            &mut analysis_info
        );
    }
    if st.mode == MODE_HYBRID {
        let mut info: SILKInfo = SILKInfo {
            signalType: 0,
            offset: 0,
        };
        info.signalType = st.silk_mode.signalType;
        info.offset = st.silk_mode.offset;
        opus_custom_encoder_ctl!(&mut st.celt_enc, CELT_SET_SILK_INFO_REQUEST, &mut info);
    }

    // 5 ms redundant frame for CELT->SILK
    let mut enc = if redundancy != 0 && celt_to_silk != 0 {
        opus_custom_encoder_ctl!(&mut st.celt_enc, CELT_SET_START_BAND_REQUEST, 0);
        opus_custom_encoder_ctl!(&mut st.celt_enc, OPUS_SET_VBR_REQUEST, 0);
        opus_custom_encoder_ctl!(&mut st.celt_enc, OPUS_SET_BITRATE_REQUEST, -1);

        // save and restore the enc, to allow access to `data`
        let saved_enc = enc.save();
        drop(enc);

        let err = unsafe {
            celt_encode_with_ec(
                &mut st.celt_enc,
                &pcm_buf,
                st.Fs / 200,
                data[nb_compr_bytes as usize..].as_mut_ptr(),
                redundancy_bytes,
                None,
            )
        };
        if err < 0 {
            return OPUS_INTERNAL_ERROR;
        }
        opus_custom_encoder_ctl!(
            &mut st.celt_enc,
            OPUS_GET_FINAL_RANGE_REQUEST,
            &mut redundant_rng
        );
        opus_custom_encoder_ctl!(&mut st.celt_enc, OPUS_RESET_STATE);
        let mut enc = ec_enc_init(&mut data[..max_data_bytes as usize - 1]);
        enc.restore(saved_enc);
        enc
    } else {
        enc
    };

    opus_custom_encoder_ctl!(&mut st.celt_enc, CELT_SET_START_BAND_REQUEST, start_band);

    if st.mode != MODE_SILK_ONLY {
        if st.mode != st.prev_mode && st.prev_mode > 0 {
            let mut dummy_0: [u8; 2] = [0; 2];
            opus_custom_encoder_ctl!(&mut st.celt_enc, OPUS_RESET_STATE);
            // Prefilling
            unsafe {
                celt_encode_with_ec(
                    &mut st.celt_enc,
                    &tmp_prefill,
                    st.Fs / 400,
                    dummy_0.as_mut_ptr(),
                    2,
                    None,
                )
            };
            opus_custom_encoder_ctl!(&mut st.celt_enc, CELT_SET_PREDICTION_REQUEST, 0);
        }
        // If false, we already busted the budget and we'll end up with a "PLC frame"
        if ec_tell(&mut enc) <= 8 * nb_compr_bytes {
            // Set the bitrate again if it was overridden in the redundancy code above
            if redundancy != 0 && celt_to_silk != 0 && st.mode == MODE_HYBRID && st.use_vbr != 0 {
                opus_custom_encoder_ctl!(
                    &mut st.celt_enc,
                    OPUS_SET_BITRATE_REQUEST,
                    st.bitrate_bps - st.silk_mode.bitRate,
                );
            }
            opus_custom_encoder_ctl!(&mut st.celt_enc, OPUS_SET_VBR_REQUEST, st.use_vbr);
            ret = unsafe {
                celt_encode_with_ec(
                    &mut st.celt_enc,
                    &pcm_buf,
                    frame_size,
                    std::ptr::null_mut(),
                    nb_compr_bytes,
                    Some(&mut enc),
                )
            };
            if ret < 0 {
                return OPUS_INTERNAL_ERROR;
            }
            // Put CELT->SILK redundancy data in the right place.
            if redundancy != 0 && celt_to_silk != 0 && st.mode == MODE_HYBRID && st.use_vbr != 0 {
                // OPUS_MOVE(data+ret, data+nb_compr_bytes, redundancy_bytes);
                let start = nb_compr_bytes as usize;
                let end = start + redundancy_bytes as usize;
                enc.buf.copy_within(start..end, ret as usize);
                nb_compr_bytes = nb_compr_bytes + redundancy_bytes;
            }
        }
    }

    // 5 ms redundant frame for SILK->CELT
    if redundancy != 0 && celt_to_silk == 0 {
        let mut dummy_1: [u8; 2] = [0; 2];
        let N2 = st.Fs / 200;
        let N4 = st.Fs / 400;
        opus_custom_encoder_ctl!(&mut st.celt_enc, OPUS_RESET_STATE);
        opus_custom_encoder_ctl!(&mut st.celt_enc, CELT_SET_START_BAND_REQUEST, 0);
        opus_custom_encoder_ctl!(&mut st.celt_enc, CELT_SET_PREDICTION_REQUEST, 0);
        opus_custom_encoder_ctl!(&mut st.celt_enc, OPUS_SET_VBR_REQUEST, 0);
        opus_custom_encoder_ctl!(&mut st.celt_enc, OPUS_SET_BITRATE_REQUEST, -1);

        if st.mode == MODE_HYBRID {
            // Shrink packet to what the encoder actually used.
            nb_compr_bytes = ret;
            ec_enc_shrink(&mut enc, nb_compr_bytes as u32);
        }
        // NOTE: We could speed this up slightly (at the expense of code size) by just adding a function that prefills the buffer
        unsafe {
            celt_encode_with_ec(
                &mut st.celt_enc,
                &pcm_buf[(st.channels * (frame_size - N2 - N4)) as usize..],
                N4,
                dummy_1.as_mut_ptr(),
                2,
                None,
            )
        };

        let err_0 = unsafe {
            celt_encode_with_ec(
                &mut st.celt_enc,
                &pcm_buf[(st.channels * (frame_size - N2)) as usize..],
                N2,
                enc.buf[nb_compr_bytes as usize..].as_mut_ptr(),
                redundancy_bytes,
                None,
            )
        };
        if err_0 < 0 {
            return OPUS_INTERNAL_ERROR;
        }
        opus_custom_encoder_ctl!(
            &mut st.celt_enc,
            OPUS_GET_FINAL_RANGE_REQUEST,
            &mut redundant_rng
        );
    }

    // Signalling the mode in the first byte
    data0[0] = gen_toc(
        st.mode,
        st.Fs / frame_size,
        curr_bandwidth,
        st.stream_channels,
    );
    st.rangeFinal = enc.rng ^ redundant_rng;
    if to_celt != 0 {
        st.prev_mode = MODE_CELT_ONLY;
    } else {
        st.prev_mode = st.mode;
    }
    st.prev_channels = st.stream_channels;
    st.prev_framesize = frame_size;
    st.first = 0;

    // DTX decision
    if st.use_dtx != 0 && (analysis_info.valid != 0 || is_silence != 0) {
        if decide_dtx_mode(
            analysis_info.activity_probability,
            &mut st.nb_no_activity_frames,
            st.peak_signal_energy,
            pcm,
            frame_size,
            st.channels,
            is_silence,
            st.arch,
        ) != 0
        {
            st.rangeFinal = 0;
            data0[0] = gen_toc(
                st.mode,
                st.Fs / frame_size,
                curr_bandwidth,
                st.stream_channels,
            );
            return 1;
        }
    } else {
        st.nb_no_activity_frames = 0;
    }

    // In the unlikely case that the SILK encoder busted its target, tell
    // the decoder to call the PLC
    if ec_tell(&mut enc) > (max_data_bytes - 1) * 8 {
        if max_data_bytes < 2 {
            return OPUS_BUFFER_TOO_SMALL;
        }
        enc.buf[0] = 0;
        ret = 1;
        st.rangeFinal = 0;
    } else if st.mode == MODE_SILK_ONLY && redundancy == 0 {
        // When in LPC only mode it's perfectly
        // reasonable to strip off trailing zero bytes as
        // the required range decoder behavior is to
        // fill these in. This can't be done when the MDCT
        // modes are used because the decoder needs to know
        // the actual length for allocation purposes.
        while ret > 2 && enc.buf[ret as usize - 1] as i32 == 0 {
            ret -= 1;
        }
    }
    // Count ToC and redundancy
    ret += 1 + redundancy_bytes;
    drop(enc);

    if st.use_vbr == 0 {
        if opus_packet_pad(og_data, ret, max_data_bytes) != OPUS_OK {
            return OPUS_INTERNAL_ERROR;
        }
        ret = max_data_bytes;
    }

    ret
}

pub fn opus_encode(
    st: &mut OpusEncoder,
    pcm: &[i16],
    analysis_frame_size: i32,
    data: &mut [u8],
    max_data_bytes: i32,
) -> i32 {
    let frame_size = frame_size_select(analysis_frame_size, st.variable_duration, st.Fs);
    if frame_size <= 0 {
        return OPUS_BAD_ARG;
    }
    let vla = (frame_size * st.channels) as usize;
    let mut in_0: Vec<f32> = ::std::vec::from_elem(0., vla);
    for i in 0..frame_size * st.channels {
        in_0[i as usize] = 1.0f32 / 32768 as f32 * pcm[i as usize] as i32 as f32;
    }
    let ret = opus_encode_native(
        st,
        &in_0,
        frame_size,
        data,
        max_data_bytes,
        16,
        Some(pcm),
        analysis_frame_size,
        0,
        -(2),
        st.channels,
        Some(downmix_int as fn(&[i16], &mut [f32], i32, i32, i32, i32, i32) -> ()),
        0,
    );
    return ret;
}

pub fn opus_encode_float(
    st: &mut OpusEncoder,
    pcm: &[f32],
    analysis_frame_size: i32,
    data: &mut [u8],
    out_data_bytes: i32,
) -> i32 {
    let frame_size = frame_size_select(analysis_frame_size, st.variable_duration, st.Fs);
    opus_encode_native(
        st,
        pcm,
        frame_size,
        data,
        out_data_bytes,
        24,
        Some(pcm),
        analysis_frame_size,
        0,
        -(2),
        st.channels,
        Some(downmix_float as fn(&[f32], &mut [f32], i32, i32, i32, i32, i32) -> ()),
        1,
    )
}

pub fn opus_encoder_ctl_impl(st: &mut OpusEncoder, request: i32, args: VarArgs) -> i32 {
    let mut current_block: u64;
    let mut ret = OPUS_OK;
    let mut ap = args;

    let celt_enc = &mut st.celt_enc;
    match request {
        OPUS_SET_APPLICATION_REQUEST => {
            let value: i32 = ap.arg::<i32>();
            if value != OPUS_APPLICATION_VOIP
                && value != OPUS_APPLICATION_AUDIO
                && value != OPUS_APPLICATION_RESTRICTED_LOWDELAY
                || st.first == 0 && st.application != value
            {
                ret = OPUS_BAD_ARG;
            } else {
                st.application = value;
                st.analysis.application = value;
            }
            current_block = 16167632229894708628;
        }
        OPUS_GET_APPLICATION_REQUEST => {
            let value_0 = ap.arg::<&mut i32>();
            *value_0 = st.application;
            current_block = 16167632229894708628;
        }
        OPUS_SET_BITRATE_REQUEST => {
            let mut value_1: i32 = ap.arg::<i32>();
            if value_1 != OPUS_AUTO && value_1 != OPUS_BITRATE_MAX {
                if value_1 <= 0 {
                    current_block = 12343738388509029619;
                } else {
                    if value_1 <= 500 {
                        value_1 = 500;
                    } else if value_1 > 300000 * st.channels {
                        value_1 = 300000 * st.channels;
                    }
                    current_block = 6057473163062296781;
                }
            } else {
                current_block = 6057473163062296781;
            }
            match current_block {
                12343738388509029619 => {}
                _ => {
                    st.user_bitrate_bps = value_1;
                    current_block = 16167632229894708628;
                }
            }
        }
        OPUS_GET_BITRATE_REQUEST => {
            let value_2 = ap.arg::<&mut i32>();
            *value_2 = user_bitrate_to_bitrate(st, st.prev_framesize, 1276);
            current_block = 16167632229894708628;
        }
        OPUS_SET_FORCE_CHANNELS_REQUEST => {
            let value_3: i32 = ap.arg::<i32>();
            if (value_3 < 1 || value_3 > st.channels) && value_3 != OPUS_AUTO {
                current_block = 12343738388509029619;
            } else {
                st.force_channels = value_3;
                current_block = 16167632229894708628;
            }
        }
        OPUS_GET_FORCE_CHANNELS_REQUEST => {
            let value_4 = ap.arg::<&mut i32>();
            *value_4 = st.force_channels;
            current_block = 16167632229894708628;
        }
        OPUS_SET_MAX_BANDWIDTH_REQUEST => {
            let value_5: i32 = ap.arg::<i32>();
            if value_5 < OPUS_BANDWIDTH_NARROWBAND || value_5 > OPUS_BANDWIDTH_FULLBAND {
                current_block = 12343738388509029619;
            } else {
                st.max_bandwidth = value_5;
                if st.max_bandwidth == OPUS_BANDWIDTH_NARROWBAND {
                    st.silk_mode.maxInternalSampleRate = 8000;
                } else if st.max_bandwidth == OPUS_BANDWIDTH_MEDIUMBAND {
                    st.silk_mode.maxInternalSampleRate = 12000;
                } else {
                    st.silk_mode.maxInternalSampleRate = 16000;
                }
                current_block = 16167632229894708628;
            }
        }
        OPUS_GET_MAX_BANDWIDTH_REQUEST => {
            let value_6 = ap.arg::<&mut i32>();
            *value_6 = st.max_bandwidth;
            current_block = 16167632229894708628;
        }
        OPUS_SET_BANDWIDTH_REQUEST => {
            let value_7: i32 = ap.arg::<i32>();
            if (value_7 < OPUS_BANDWIDTH_NARROWBAND || value_7 > OPUS_BANDWIDTH_FULLBAND)
                && value_7 != OPUS_AUTO
            {
                current_block = 12343738388509029619;
            } else {
                st.user_bandwidth = value_7;
                if st.user_bandwidth == OPUS_BANDWIDTH_NARROWBAND {
                    st.silk_mode.maxInternalSampleRate = 8000;
                } else if st.user_bandwidth == OPUS_BANDWIDTH_MEDIUMBAND {
                    st.silk_mode.maxInternalSampleRate = 12000;
                } else {
                    st.silk_mode.maxInternalSampleRate = 16000;
                }
                current_block = 16167632229894708628;
            }
        }
        OPUS_GET_BANDWIDTH_REQUEST => {
            let value_8 = ap.arg::<&mut i32>();
            *value_8 = st.bandwidth;
            current_block = 16167632229894708628;
        }
        OPUS_SET_DTX_REQUEST => {
            let value_9: i32 = ap.arg::<i32>();
            if value_9 < 0 || value_9 > 1 {
                current_block = 12343738388509029619;
            } else {
                st.use_dtx = value_9;
                current_block = 16167632229894708628;
            }
        }
        OPUS_GET_DTX_REQUEST => {
            let value_10 = ap.arg::<&mut i32>();
            *value_10 = st.use_dtx;
            current_block = 16167632229894708628;
        }
        OPUS_SET_COMPLEXITY_REQUEST => {
            let value_11: i32 = ap.arg::<i32>();
            if value_11 < 0 || value_11 > 10 {
                current_block = 12343738388509029619;
            } else {
                st.silk_mode.complexity = value_11;
                opus_custom_encoder_ctl!(celt_enc, OPUS_SET_COMPLEXITY_REQUEST, value_11);
                current_block = 16167632229894708628;
            }
        }
        OPUS_GET_COMPLEXITY_REQUEST => {
            let value_12 = ap.arg::<&mut i32>();
            *value_12 = st.silk_mode.complexity;
            current_block = 16167632229894708628;
        }
        OPUS_SET_INBAND_FEC_REQUEST => {
            let value_13: i32 = ap.arg::<i32>();
            if value_13 < 0 || value_13 > 1 {
                current_block = 12343738388509029619;
            } else {
                st.silk_mode.useInBandFEC = value_13;
                current_block = 16167632229894708628;
            }
        }
        OPUS_GET_INBAND_FEC_REQUEST => {
            let value_14 = ap.arg::<&mut i32>();
            *value_14 = st.silk_mode.useInBandFEC;
            current_block = 16167632229894708628;
        }
        OPUS_SET_PACKET_LOSS_PERC_REQUEST => {
            let value_15: i32 = ap.arg::<i32>();
            if value_15 < 0 || value_15 > 100 {
                current_block = 12343738388509029619;
            } else {
                st.silk_mode.packetLossPercentage = value_15;
                opus_custom_encoder_ctl!(celt_enc, OPUS_SET_PACKET_LOSS_PERC_REQUEST, value_15);
                current_block = 16167632229894708628;
            }
        }
        OPUS_GET_PACKET_LOSS_PERC_REQUEST => {
            let value_16 = ap.arg::<&mut i32>();
            *value_16 = st.silk_mode.packetLossPercentage;
            current_block = 16167632229894708628;
        }
        OPUS_SET_VBR_REQUEST => {
            let value_17: i32 = ap.arg::<i32>();
            if value_17 < 0 || value_17 > 1 {
                current_block = 12343738388509029619;
            } else {
                st.use_vbr = value_17;
                st.silk_mode.useCBR = 1 - value_17;
                current_block = 16167632229894708628;
            }
        }
        OPUS_GET_VBR_REQUEST => {
            let value_18 = ap.arg::<&mut i32>();
            *value_18 = st.use_vbr;
            current_block = 16167632229894708628;
        }
        OPUS_SET_VOICE_RATIO_REQUEST => {
            let value_19: i32 = ap.arg::<i32>();
            if value_19 < -1 || value_19 > 100 {
                current_block = 12343738388509029619;
            } else {
                st.voice_ratio = value_19;
                current_block = 16167632229894708628;
            }
        }
        OPUS_GET_VOICE_RATIO_REQUEST => {
            let value_20 = ap.arg::<&mut i32>();
            *value_20 = st.voice_ratio;
            current_block = 16167632229894708628;
        }
        OPUS_SET_VBR_CONSTRAINT_REQUEST => {
            let value_21: i32 = ap.arg::<i32>();
            if value_21 < 0 || value_21 > 1 {
                current_block = 12343738388509029619;
            } else {
                st.vbr_constraint = value_21;
                current_block = 16167632229894708628;
            }
        }
        OPUS_GET_VBR_CONSTRAINT_REQUEST => {
            let value_22 = ap.arg::<&mut i32>();
            *value_22 = st.vbr_constraint;
            current_block = 16167632229894708628;
        }
        OPUS_SET_SIGNAL_REQUEST => {
            let value_23: i32 = ap.arg::<i32>();
            if value_23 != OPUS_AUTO
                && value_23 != OPUS_SIGNAL_VOICE
                && value_23 != OPUS_SIGNAL_MUSIC
            {
                current_block = 12343738388509029619;
            } else {
                st.signal_type = value_23;
                current_block = 16167632229894708628;
            }
        }
        OPUS_GET_SIGNAL_REQUEST => {
            let value_24 = ap.arg::<&mut i32>();
            *value_24 = st.signal_type;
            current_block = 16167632229894708628;
        }
        OPUS_GET_LOOKAHEAD_REQUEST => {
            let value_25 = ap.arg::<&mut i32>();
            *value_25 = st.Fs / 400;
            if st.application != OPUS_APPLICATION_RESTRICTED_LOWDELAY {
                *value_25 += st.delay_compensation;
            }
            current_block = 16167632229894708628;
        }
        OPUS_GET_SAMPLE_RATE_REQUEST => {
            let value_26 = ap.arg::<&mut i32>();
            *value_26 = st.Fs;
            current_block = 16167632229894708628;
        }
        OPUS_GET_FINAL_RANGE_REQUEST => {
            let value_27 = ap.arg::<&mut u32>();
            *value_27 = st.rangeFinal;
            current_block = 16167632229894708628;
        }
        OPUS_SET_LSB_DEPTH_REQUEST => {
            let value_28: i32 = ap.arg::<i32>();
            if value_28 < 8 || value_28 > 24 {
                current_block = 12343738388509029619;
            } else {
                st.lsb_depth = value_28;
                current_block = 16167632229894708628;
            }
        }
        OPUS_GET_LSB_DEPTH_REQUEST => {
            let value_29 = ap.arg::<&mut i32>();
            *value_29 = st.lsb_depth;
            current_block = 16167632229894708628;
        }
        OPUS_SET_EXPERT_FRAME_DURATION_REQUEST => {
            let value_30: i32 = ap.arg::<i32>();
            if value_30 != OPUS_FRAMESIZE_ARG
                && value_30 != OPUS_FRAMESIZE_2_5_MS
                && value_30 != OPUS_FRAMESIZE_5_MS
                && value_30 != OPUS_FRAMESIZE_10_MS
                && value_30 != OPUS_FRAMESIZE_20_MS
                && value_30 != OPUS_FRAMESIZE_40_MS
                && value_30 != OPUS_FRAMESIZE_60_MS
                && value_30 != OPUS_FRAMESIZE_80_MS
                && value_30 != OPUS_FRAMESIZE_100_MS
                && value_30 != OPUS_FRAMESIZE_120_MS
            {
                current_block = 12343738388509029619;
            } else {
                st.variable_duration = value_30;
                current_block = 16167632229894708628;
            }
        }
        OPUS_GET_EXPERT_FRAME_DURATION_REQUEST => {
            let value_31 = ap.arg::<&mut i32>();
            *value_31 = st.variable_duration;
            current_block = 16167632229894708628;
        }
        OPUS_SET_PREDICTION_DISABLED_REQUEST => {
            let value_32: i32 = ap.arg::<i32>();
            if value_32 > 1 || value_32 < 0 {
                current_block = 12343738388509029619;
            } else {
                st.silk_mode.reducedDependency = value_32;
                current_block = 16167632229894708628;
            }
        }
        OPUS_GET_PREDICTION_DISABLED_REQUEST => {
            let value_33 = ap.arg::<&mut i32>();
            *value_33 = st.silk_mode.reducedDependency;
            current_block = 16167632229894708628;
        }
        OPUS_SET_PHASE_INVERSION_DISABLED_REQUEST => {
            let value_34: i32 = ap.arg::<i32>();
            if value_34 < 0 || value_34 > 1 {
                current_block = 12343738388509029619;
            } else {
                opus_custom_encoder_ctl!(
                    celt_enc,
                    OPUS_SET_PHASE_INVERSION_DISABLED_REQUEST,
                    value_34,
                );
                current_block = 16167632229894708628;
            }
        }
        OPUS_GET_PHASE_INVERSION_DISABLED_REQUEST => {
            let value_35 = ap.arg::<&mut i32>();
            opus_custom_encoder_ctl!(
                celt_enc,
                OPUS_GET_PHASE_INVERSION_DISABLED_REQUEST,
                value_35
            );
            current_block = 16167632229894708628;
        }
        OPUS_RESET_STATE => {
            st.analysis = TonalityAnalysisState::new(st.application, st.Fs);
            st.stream_channels = 0;
            st.hybrid_stereo_width_Q14 = 0;
            st.variable_HP_smth2_Q15 = 0;
            st.prev_HB_gain = 0.;
            st.hp_mem.fill(0.);
            st.mode = 0;
            st.prev_mode = 0;
            st.prev_channels = 0;
            st.prev_framesize = 0;
            st.bandwidth = 0;
            st.auto_bandwidth = 0;
            st.silk_bw_switch = 0;
            st.first = 0;
            st.energy_masking = std::ptr::null_mut();
            st.width_mem = StereoWidthState::default();
            st.delay_buffer.fill(0.);
            st.detected_bandwidth = 0;
            st.nb_no_activity_frames = 0;
            st.peak_signal_energy = 0.;
            st.nonfinal_frame = 0;
            st.rangeFinal = 0;
            opus_custom_encoder_ctl!(celt_enc, OPUS_RESET_STATE);
            let (silk_enc, _dummy) = silk_InitEncoder(st.arch);
            st.silk_enc = silk_enc;
            st.stream_channels = st.channels;
            st.hybrid_stereo_width_Q14 = ((1) << 14) as i16;
            st.prev_HB_gain = Q15ONE;
            st.first = 1;
            st.mode = MODE_HYBRID;
            st.bandwidth = OPUS_BANDWIDTH_FULLBAND;
            st.variable_HP_smth2_Q15 =
                ((silk_lin2log(VARIABLE_HP_MIN_CUTOFF_HZ) as u32) << 8) as i32;
            current_block = 16167632229894708628;
        }
        OPUS_SET_FORCE_MODE_REQUEST => {
            let value_36: i32 = ap.arg::<i32>();
            if (value_36 < MODE_SILK_ONLY || value_36 > MODE_CELT_ONLY) && value_36 != OPUS_AUTO {
                current_block = 12343738388509029619;
            } else {
                st.user_forced_mode = value_36;
                current_block = 16167632229894708628;
            }
        }
        OPUS_SET_LFE_REQUEST => {
            let value_37: i32 = ap.arg::<i32>();
            st.lfe = value_37;
            ret = opus_custom_encoder_ctl!(celt_enc, OPUS_SET_LFE_REQUEST, value_37);
            current_block = 16167632229894708628;
        }
        OPUS_SET_ENERGY_MASK_REQUEST => {
            let value_38: *mut f32 = ap.arg::<*mut f32>();
            st.energy_masking = value_38;
            ret = opus_custom_encoder_ctl!(celt_enc, OPUS_SET_ENERGY_MASK_REQUEST, unsafe {
                value_38.offset(value_38.offset_from(value_38) as i64 as isize)
            },);
            current_block = 16167632229894708628;
        }
        OPUS_GET_IN_DTX_REQUEST => {
            let value_39: &mut i32 = ap.arg::<&mut i32>();
            if st.silk_mode.useDTX != 0
                && (st.prev_mode == MODE_SILK_ONLY || st.prev_mode == MODE_HYBRID)
            {
                *value_39 = 1;
                for n in 0..st.silk_mode.nChannelsInternal {
                    *value_39 = (*value_39 != 0
                        && st.silk_enc.state_Fxx[n as usize].sCmn.noSpeechCounter
                            >= NB_SPEECH_FRAMES_BEFORE_DTX) as i32;
                }
            } else if st.use_dtx != 0 {
                *value_39 = (st.nb_no_activity_frames >= NB_SPEECH_FRAMES_BEFORE_DTX) as i32;
            } else {
                *value_39 = 0;
            }
            current_block = 16167632229894708628;
        }
        CELT_GET_MODE_REQUEST => {
            let value_40: &mut *const OpusCustomMode = ap.arg::<&mut *const OpusCustomMode>();
            ret = opus_custom_encoder_ctl!(celt_enc, CELT_GET_MODE_REQUEST, value_40);
            current_block = 16167632229894708628;
        }
        _ => {
            ret = OPUS_UNIMPLEMENTED;
            current_block = 16167632229894708628;
        }
    }
    match current_block {
        12343738388509029619 => return OPUS_BAD_ARG,
        _ => return ret,
    };
}

#[macro_export]
macro_rules! opus_encoder_ctl {
    ($st:expr, $request:expr, $($args:expr),*) => {
        $crate::opus_encoder_ctl_impl($st, $request, $crate::varargs!($($args),*))
    };
    ($st:expr, $request:expr, $($args:expr),*,) => {
        opus_encoder_ctl!($st, $request, $($args),*)
    };
    ($st:expr, $request:expr) => {
        opus_encoder_ctl!($st, $request,)
    };
}
