use crate::api::{Application, Bandwidth, Bitrate, Channels, FrameSize, Signal};
use crate::src::repacketizer::FrameSource;

pub mod arch_h {
    pub type opus_val16 = f32;
    pub type opus_val32 = f32;
    pub const Q15ONE: f32 = 1.0f32;
    pub const EPSILON: f32 = 1e-15f32;
    pub const VERY_SMALL: f32 = 1e-30f32;
}
pub mod stddef_h {}
pub mod cpu_support_h {
    #[inline]
    pub fn opus_select_arch() -> i32 {
        return 0;
    }
}
use self::arch_h::{opus_val16, opus_val32, EPSILON, Q15ONE, VERY_SMALL};
pub use self::cpu_support_h::opus_select_arch;
use crate::celt::celt_encoder::{celt_encode_with_ec, OpusCustomEncoder, SILKInfo};
use crate::celt::entcode::ec_tell;
use crate::celt::entenc::ec_enc;
use crate::celt::entenc::{ec_enc_bit_logp, ec_enc_done, ec_enc_init, ec_enc_shrink, ec_enc_uint};
use crate::celt::float_cast::FLOAT2INT16;
use crate::celt::mathops::{celt_exp2, celt_maxabs16, celt_sqrt};
use crate::celt::modes::OpusCustomMode;
use crate::celt::pitch::celt_inner_prod;

use crate::silk::define::{
    DTX_ACTIVITY_THRESHOLD, MAX_CONSECUTIVE_DTX, NB_SPEECH_FRAMES_BEFORE_DTX, VAD_NO_DECISION,
};
use crate::silk::enc_API::silk_EncControlStruct;
use crate::silk::enc_API::{silk_Encode, silk_InitEncoder};
use crate::silk::float::structs_FLP::silk_encoder;
use crate::silk::lin2log::silk_lin2log;
use crate::silk::log2lin::silk_log2lin;
use crate::silk::tuning_parameters::{VARIABLE_HP_MIN_CUTOFF_HZ, VARIABLE_HP_SMTH_COEF2};
use crate::src::analysis::{
    run_analysis, tonality_analysis_init, tonality_analysis_reset, AnalysisInfo, DownmixInput,
    TonalityAnalysisState,
};
use crate::src::opus_defines::{
    OPUS_APPLICATION_AUDIO, OPUS_APPLICATION_RESTRICTED_LOWDELAY, OPUS_APPLICATION_VOIP, OPUS_AUTO,
    OPUS_BAD_ARG, OPUS_BANDWIDTH_FULLBAND, OPUS_BANDWIDTH_MEDIUMBAND, OPUS_BANDWIDTH_NARROWBAND,
    OPUS_BANDWIDTH_SUPERWIDEBAND, OPUS_BANDWIDTH_WIDEBAND, OPUS_BITRATE_MAX, OPUS_BUFFER_TOO_SMALL,
    OPUS_FRAMESIZE_120_MS, OPUS_FRAMESIZE_2_5_MS, OPUS_FRAMESIZE_40_MS, OPUS_FRAMESIZE_ARG,
    OPUS_INTERNAL_ERROR, OPUS_OK, OPUS_SIGNAL_MUSIC, OPUS_SIGNAL_VOICE,
};
use crate::src::opus_private::{MODE_CELT_ONLY, MODE_HYBRID, MODE_SILK_ONLY};
use crate::{opus_packet_pad, OpusRepacketizer};

#[derive(Copy, Clone)]
#[repr(C)]
pub struct OpusEncoder {
    pub(crate) silk_enc: silk_encoder,
    pub(crate) celt_enc: OpusCustomEncoder,
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
    pub(crate) prev_HB_gain: opus_val16,
    pub(crate) hp_mem: [opus_val32; 4],
    pub(crate) mode: i32,
    pub(crate) prev_mode: i32,
    pub(crate) prev_channels: i32,
    pub(crate) prev_framesize: i32,
    pub(crate) bandwidth: i32,
    pub(crate) auto_bandwidth: i32,
    pub(crate) silk_bw_switch: i32,
    pub(crate) first: i32,
    /// Energy mask for surround encoding (set by multistream encoder).
    /// `energy_masking_len == 0` means no mask is active.
    pub(crate) energy_masking: [opus_val16; 2 * 21],
    pub(crate) energy_masking_len: usize,
    pub(crate) width_mem: StereoWidthState,
    pub(crate) delay_buffer: [opus_val16; 960],
    pub(crate) detected_bandwidth: i32,
    pub(crate) nb_no_activity_frames: i32,
    pub(crate) peak_signal_energy: opus_val32,
    pub(crate) nonfinal_frame: i32,
    pub(crate) rangeFinal: u32,
}
#[derive(Copy, Clone)]
#[repr(C)]
pub struct StereoWidthState {
    pub XX: opus_val32,
    pub XY: opus_val32,
    pub YY: opus_val32,
    pub smoothed_width: opus_val16,
    pub max_follower: opus_val16,
}
pub const PSEUDO_SNR_THRESHOLD: f32 = 316.23f32;
static mono_voice_bandwidth_thresholds: [i32; 8] = [9000, 700, 9000, 700, 13500, 1000, 14000, 2000];
static mono_music_bandwidth_thresholds: [i32; 8] = [9000, 700, 9000, 700, 11000, 1000, 12000, 2000];
static stereo_voice_bandwidth_thresholds: [i32; 8] =
    [9000, 700, 9000, 700, 13500, 1000, 14000, 2000];
static stereo_music_bandwidth_thresholds: [i32; 8] =
    [9000, 700, 9000, 700, 11000, 1000, 12000, 2000];
static mut stereo_voice_threshold: i32 = 19000;
static mut stereo_music_threshold: i32 = 17000;
static mut mode_thresholds: [[i32; 2]; 2] = [[64000, 10000], [44000, 10000]];
static fec_thresholds: [i32; 10] = [
    12000, 1000, 14000, 1000, 16000, 1000, 20000, 1000, 22000, 1000,
];

impl OpusEncoder {
    pub fn new(Fs: i32, channels: i32, application: i32) -> Result<OpusEncoder, i32> {
        if Fs != 48000 && Fs != 24000 && Fs != 16000 && Fs != 12000 && Fs != 8000
            || channels != 1 && channels != 2
            || application != OPUS_APPLICATION_VOIP
                && application != OPUS_APPLICATION_AUDIO
                && application != OPUS_APPLICATION_RESTRICTED_LOWDELAY
        {
            return Err(OPUS_BAD_ARG);
        }
        let arch = opus_select_arch();
        // Build silk encoder state
        let mut silk_enc = silk_encoder::default();
        let mut silk_mode = silk_EncControlStruct::default();
        let ret = silk_InitEncoder(&mut silk_enc, arch, &mut silk_mode);
        if ret != 0 {
            return Err(OPUS_INTERNAL_ERROR);
        }
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

        // Build CELT encoder state
        let mut celt_enc = OpusCustomEncoder::new(Fs, channels, arch)?;
        celt_enc.signalling = 0;
        celt_enc.complexity = silk_mode.complexity;

        // Build analysis state
        let mut analysis = TonalityAnalysisState::default();
        tonality_analysis_init(&mut analysis, Fs);
        analysis.application = application;

        Ok(OpusEncoder {
            silk_enc,
            celt_enc,
            silk_mode,
            application,
            channels,
            delay_compensation: Fs / 250,
            force_channels: OPUS_AUTO,
            signal_type: OPUS_AUTO,
            user_bandwidth: OPUS_AUTO,
            max_bandwidth: OPUS_BANDWIDTH_FULLBAND,
            user_forced_mode: OPUS_AUTO,
            voice_ratio: -1,
            Fs,
            use_vbr: 1,
            vbr_constraint: 1,
            variable_duration: OPUS_FRAMESIZE_ARG,
            bitrate_bps: 3000 + Fs * channels,
            user_bitrate_bps: OPUS_AUTO,
            lsb_depth: 24,
            encoder_buffer: Fs / 100,
            lfe: 0,
            arch,
            use_dtx: 0,
            analysis,
            stream_channels: channels,
            hybrid_stereo_width_Q14: ((1) << 14) as i16,
            variable_HP_smth2_Q15: ((silk_lin2log(VARIABLE_HP_MIN_CUTOFF_HZ) as u32) << 8) as i32,
            prev_HB_gain: Q15ONE,
            hp_mem: [0.0; 4],
            mode: MODE_HYBRID,
            prev_mode: 0,
            prev_channels: 0,
            prev_framesize: 0,
            bandwidth: OPUS_BANDWIDTH_FULLBAND,
            auto_bandwidth: 0,
            silk_bw_switch: 0,
            first: 1,
            energy_masking: [0.0; 2 * 21],
            energy_masking_len: 0,
            width_mem: StereoWidthState {
                XX: 0.0,
                XY: 0.0,
                YY: 0.0,
                smoothed_width: 0.0,
                max_follower: 0.0,
            },
            delay_buffer: [0.0; 960],
            detected_bandwidth: 0,
            nb_no_activity_frames: 0,
            peak_signal_energy: 0.0,
            nonfinal_frame: 0,
            rangeFinal: 0,
        })
    }

    /// Encode an audio frame from interleaved `i16` PCM samples.
    ///
    /// `pcm` must contain `frame_size * channels` samples where `frame_size`
    /// is one of the valid Opus frame sizes for the configured sample rate.
    /// `output` is the buffer for the encoded packet.
    ///
    /// Returns the number of bytes written into `output` on success, or a
    /// negative Opus error code on failure.
    pub fn encode(&mut self, pcm: &[i16], output: &mut [u8]) -> i32 {
        let frame_size = pcm.len() as i32 / self.channels;
        opus_encode(self, pcm, frame_size, output)
    }

    /// Encode an audio frame from interleaved `f32` PCM samples.
    ///
    /// `pcm` must contain `frame_size * channels` samples where `frame_size`
    /// is one of the valid Opus frame sizes for the configured sample rate.
    /// Samples should be in the range `[-1.0, 1.0]`; values outside this
    /// range are supported but will be clipped on decoding.
    /// `output` is the buffer for the encoded packet.
    ///
    /// Returns the number of bytes written into `output` on success, or a
    /// negative Opus error code on failure.
    pub fn encode_float(&mut self, pcm: &[f32], output: &mut [u8]) -> i32 {
        let frame_size = pcm.len() as i32 / self.channels;
        opus_encode_float(self, pcm, frame_size, output)
    }

    // -- Type-safe CTL getters and setters --

    pub fn set_application(&mut self, app: Application) -> Result<(), i32> {
        if self.first == 0 && self.application != i32::from(app) {
            return Err(OPUS_BAD_ARG);
        }
        self.application = app.into();
        self.analysis.application = app.into();
        Ok(())
    }

    pub fn application(&self) -> Application {
        Application::try_from(self.application).unwrap()
    }

    pub fn set_bitrate(&mut self, bitrate: Bitrate) {
        let value: i32 = bitrate.into();
        if value != OPUS_AUTO && value != OPUS_BITRATE_MAX {
            let clamped = value.max(500).min(300000 * self.channels);
            self.user_bitrate_bps = clamped;
        } else {
            self.user_bitrate_bps = value;
        }
    }

    pub fn bitrate(&self) -> i32 {
        user_bitrate_to_bitrate(self, self.prev_framesize, 1276)
    }

    pub fn set_complexity(&mut self, complexity: i32) -> Result<(), i32> {
        if complexity < 0 || complexity > 10 {
            return Err(OPUS_BAD_ARG);
        }
        self.silk_mode.complexity = complexity;
        self.celt_enc.complexity = complexity;
        Ok(())
    }

    pub fn complexity(&self) -> i32 {
        self.silk_mode.complexity
    }

    pub fn set_signal(&mut self, signal: Option<Signal>) {
        self.signal_type = match signal {
            Some(s) => s.into(),
            None => OPUS_AUTO,
        };
    }

    pub fn signal(&self) -> Option<Signal> {
        Signal::try_from(self.signal_type).ok()
    }

    pub fn set_bandwidth(&mut self, bw: Option<Bandwidth>) {
        let raw = match bw {
            Some(b) => {
                let v: i32 = b.into();
                match b {
                    Bandwidth::Narrowband => self.silk_mode.maxInternalSampleRate = 8000,
                    Bandwidth::Mediumband => self.silk_mode.maxInternalSampleRate = 12000,
                    _ => self.silk_mode.maxInternalSampleRate = 16000,
                }
                v
            }
            None => OPUS_AUTO,
        };
        self.user_bandwidth = raw;
    }

    pub fn get_bandwidth(&self) -> i32 {
        self.bandwidth
    }

    pub fn set_max_bandwidth(&mut self, bw: Bandwidth) {
        self.max_bandwidth = bw.into();
        match bw {
            Bandwidth::Narrowband => self.silk_mode.maxInternalSampleRate = 8000,
            Bandwidth::Mediumband => self.silk_mode.maxInternalSampleRate = 12000,
            _ => self.silk_mode.maxInternalSampleRate = 16000,
        }
    }

    pub fn max_bandwidth(&self) -> Bandwidth {
        Bandwidth::try_from(self.max_bandwidth).unwrap()
    }

    pub fn set_vbr(&mut self, enabled: bool) {
        self.use_vbr = enabled as i32;
        self.silk_mode.useCBR = (!enabled) as i32;
    }

    pub fn vbr(&self) -> bool {
        self.use_vbr != 0
    }

    pub fn set_vbr_constraint(&mut self, enabled: bool) {
        self.vbr_constraint = enabled as i32;
    }

    pub fn vbr_constraint(&self) -> bool {
        self.vbr_constraint != 0
    }

    pub fn set_force_channels(&mut self, channels: Option<Channels>) -> Result<(), i32> {
        let raw = match channels {
            Some(c) => {
                let v: i32 = c.into();
                if v > self.channels {
                    return Err(OPUS_BAD_ARG);
                }
                v
            }
            None => OPUS_AUTO,
        };
        self.force_channels = raw;
        Ok(())
    }

    pub fn force_channels(&self) -> Option<Channels> {
        if self.force_channels == OPUS_AUTO {
            None
        } else {
            Channels::try_from(self.force_channels).ok()
        }
    }

    pub fn set_inband_fec(&mut self, enabled: bool) {
        self.silk_mode.useInBandFEC = enabled as i32;
    }

    pub fn inband_fec(&self) -> bool {
        self.silk_mode.useInBandFEC != 0
    }

    pub fn set_packet_loss_perc(&mut self, pct: i32) -> Result<(), i32> {
        if pct < 0 || pct > 100 {
            return Err(OPUS_BAD_ARG);
        }
        self.silk_mode.packetLossPercentage = pct;
        self.celt_enc.loss_rate = pct;
        Ok(())
    }

    pub fn packet_loss_perc(&self) -> i32 {
        self.silk_mode.packetLossPercentage
    }

    pub fn set_dtx(&mut self, enabled: bool) {
        self.use_dtx = enabled as i32;
    }

    pub fn dtx(&self) -> bool {
        self.use_dtx != 0
    }

    pub fn set_lsb_depth(&mut self, depth: i32) -> Result<(), i32> {
        if depth < 8 || depth > 24 {
            return Err(OPUS_BAD_ARG);
        }
        self.lsb_depth = depth;
        Ok(())
    }

    pub fn lsb_depth(&self) -> i32 {
        self.lsb_depth
    }

    pub fn set_expert_frame_duration(&mut self, fs: FrameSize) {
        self.variable_duration = fs.into();
    }

    pub fn expert_frame_duration(&self) -> FrameSize {
        FrameSize::try_from(self.variable_duration).unwrap()
    }

    pub fn set_prediction_disabled(&mut self, disabled: bool) {
        self.silk_mode.reducedDependency = disabled as i32;
    }

    pub fn prediction_disabled(&self) -> bool {
        self.silk_mode.reducedDependency != 0
    }

    pub fn set_phase_inversion_disabled(&mut self, disabled: bool) {
        self.celt_enc.disable_inv = disabled as i32;
    }

    pub fn phase_inversion_disabled(&self) -> bool {
        self.celt_enc.disable_inv != 0
    }

    pub fn set_force_mode(&mut self, mode: i32) -> Result<(), i32> {
        if (mode < MODE_SILK_ONLY || mode > MODE_CELT_ONLY) && mode != OPUS_AUTO {
            return Err(OPUS_BAD_ARG);
        }
        self.user_forced_mode = mode;
        Ok(())
    }

    pub fn channels(&self) -> i32 {
        self.channels
    }

    pub fn sample_rate(&self) -> i32 {
        self.Fs
    }

    pub fn lookahead(&self) -> i32 {
        let mut la = self.Fs / 400;
        if self.application != OPUS_APPLICATION_RESTRICTED_LOWDELAY {
            la += self.delay_compensation;
        }
        la
    }

    pub fn final_range(&self) -> u32 {
        self.rangeFinal
    }

    pub fn in_dtx(&self) -> bool {
        if self.silk_mode.useDTX != 0
            && (self.prev_mode == MODE_SILK_ONLY || self.prev_mode == MODE_HYBRID)
        {
            let silk_enc = &self.silk_enc;
            let mut all_dtx = true;
            for n in 0..self.silk_mode.nChannelsInternal {
                if silk_enc.state_Fxx[n as usize].sCmn.noSpeechCounter < NB_SPEECH_FRAMES_BEFORE_DTX
                {
                    all_dtx = false;
                }
            }
            all_dtx
        } else if self.use_dtx != 0 {
            self.nb_no_activity_frames >= NB_SPEECH_FRAMES_BEFORE_DTX
        } else {
            false
        }
    }

    pub fn reset(&mut self) {
        let mut dummy = silk_EncControlStruct::default();
        tonality_analysis_reset(&mut self.analysis);
        // Zero from stream_channels to end of struct (matches C OPUS_RESET_STATE)
        self.stream_channels = 0;
        self.hybrid_stereo_width_Q14 = 0;
        self.variable_HP_smth2_Q15 = 0;
        self.prev_HB_gain = 0.0;
        self.hp_mem = [0.0; 4];
        self.mode = 0;
        self.prev_mode = 0;
        self.prev_channels = 0;
        self.prev_framesize = 0;
        self.bandwidth = 0;
        self.auto_bandwidth = 0;
        self.silk_bw_switch = 0;
        self.first = 0;
        self.energy_masking = [0.0; 2 * 21];
        self.energy_masking_len = 0;
        self.width_mem = StereoWidthState {
            XX: 0.0,
            XY: 0.0,
            YY: 0.0,
            smoothed_width: 0.0,
            max_follower: 0.0,
        };
        self.delay_buffer = [0.0; 960];
        self.detected_bandwidth = 0;
        self.nb_no_activity_frames = 0;
        self.peak_signal_energy = 0.0;
        self.nonfinal_frame = 0;
        self.rangeFinal = 0;

        self.celt_enc.reset();
        silk_InitEncoder(&mut self.silk_enc, self.arch, &mut dummy);
        self.stream_channels = self.channels;
        self.hybrid_stereo_width_Q14 = ((1) << 14) as i16;
        self.prev_HB_gain = Q15ONE;
        self.first = 1;
        self.mode = MODE_HYBRID;
        self.bandwidth = OPUS_BANDWIDTH_FULLBAND;
        self.variable_HP_smth2_Q15 = ((silk_lin2log(VARIABLE_HP_MIN_CUTOFF_HZ) as u32) << 8) as i32;
    }
}
/// Upstream C: src/opus_encoder.c:gen_toc
fn gen_toc(mode: i32, mut framerate: i32, bandwidth: i32, channels: i32) -> u8 {
    let mut period: i32 = 0;
    let mut toc: u8 = 0;
    period = 0;
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
    toc = (toc as i32 | ((channels == 2) as i32) << 2) as u8;
    return toc;
}
/// Upstream C: src/opus_encoder.c:silk_biquad_float
fn silk_biquad_float(
    in_0: &[opus_val16],
    B_Q28: &[i32],
    A_Q28: &[i32],
    S: &mut [opus_val32],
    out: &mut [opus_val16],
    len: i32,
    stride: i32,
) {
    let mut k: i32 = 0;
    let mut vout: opus_val32 = 0.;
    let mut inval: opus_val32 = 0.;
    let mut A: [opus_val32; 2] = [0.; 2];
    let mut B: [opus_val32; 3] = [0.; 3];
    A[0 as usize] = A_Q28[0] as f32 * (1.0f32 / ((1) << 28) as f32);
    A[1 as usize] = A_Q28[1] as f32 * (1.0f32 / ((1) << 28) as f32);
    B[0 as usize] = B_Q28[0] as f32 * (1.0f32 / ((1) << 28) as f32);
    B[1 as usize] = B_Q28[1] as f32 * (1.0f32 / ((1) << 28) as f32);
    B[2 as usize] = B_Q28[2] as f32 * (1.0f32 / ((1) << 28) as f32);
    k = 0;
    while k < len {
        inval = in_0[(k * stride) as usize];
        vout = S[0] + B[0 as usize] * inval;
        S[0] = S[1] - vout * A[0 as usize] + B[1 as usize] * inval;
        S[1] = -vout * A[1 as usize] + B[2 as usize] * inval + VERY_SMALL;
        out[(k * stride) as usize] = vout;
        k += 1;
    }
}
/// Upstream C: src/opus_encoder.c:hp_cutoff
fn hp_cutoff(
    in_0: &[opus_val16],
    cutoff_Hz: i32,
    out: &mut [opus_val16],
    hp_mem: &mut [opus_val32],
    len: i32,
    channels: i32,
    Fs: i32,
    _arch: i32,
) {
    let mut B_Q28: [i32; 3] = [0; 3];
    let mut A_Q28: [i32; 2] = [0; 2];
    let mut Fc_Q19: i32 = 0;
    let mut r_Q28: i32 = 0;
    let mut r_Q22: i32 = 0;
    #[allow(clippy::approx_constant)]
    // Intentional: C reference uses 3.14159, not exact PI
    let pi_approx = 3.14159f64;
    Fc_Q19 = (1.5f64 * pi_approx / 1000 as f64 * ((1) << 19) as f64 + 0.5f64) as i32 as i16 as i32
        * cutoff_Hz as i16 as i32
        / (Fs / 1000);
    r_Q28 = (1.0f64 * ((1) << 28) as f64 + 0.5f64) as i32
        - (0.92f64 * ((1) << 9) as f64 + 0.5f64) as i32 * Fc_Q19;
    B_Q28[0 as usize] = r_Q28;
    B_Q28[1 as usize] = ((-r_Q28 as u32) << 1) as i32;
    B_Q28[2 as usize] = r_Q28;
    r_Q22 = r_Q28 >> 6;
    A_Q28[0 as usize] = (r_Q22 as i64
        * ((Fc_Q19 as i64 * Fc_Q19 as i64 >> 16) as i32
            - (2.0f64 * ((1) << 22) as f64 + 0.5f64) as i32) as i64
        >> 16) as i32;
    A_Q28[1 as usize] = (r_Q22 as i64 * r_Q22 as i64 >> 16) as i32;
    silk_biquad_float(in_0, &B_Q28, &A_Q28, hp_mem, out, len, channels);
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
/// Upstream C: src/opus_encoder.c:dc_reject
fn dc_reject(
    in_0: &[opus_val16],
    cutoff_Hz: i32,
    out: &mut [opus_val16],
    hp_mem: &mut [opus_val32],
    len: i32,
    channels: i32,
    Fs: i32,
) {
    let mut i: i32 = 0;
    let mut coef: f32 = 0.;
    let mut coef2: f32 = 0.;
    coef = 6.3f32 * cutoff_Hz as f32 / Fs as f32;
    coef2 = 1 as f32 - coef;
    if channels == 2 {
        let mut m0: f32 = 0.;
        let mut m2: f32 = 0.;
        m0 = hp_mem[0];
        m2 = hp_mem[2];
        i = 0;
        while i < len {
            let mut x0: opus_val32 = 0.;
            let mut x1: opus_val32 = 0.;
            let mut out0: opus_val32 = 0.;
            let mut out1: opus_val32 = 0.;
            x0 = in_0[(2 * i + 0) as usize];
            x1 = in_0[(2 * i + 1) as usize];
            out0 = x0 - m0;
            out1 = x1 - m2;
            m0 = coef * x0 + VERY_SMALL + coef2 * m0;
            m2 = coef * x1 + VERY_SMALL + coef2 * m2;
            out[(2 * i + 0) as usize] = out0;
            out[(2 * i + 1) as usize] = out1;
            i += 1;
        }
        hp_mem[0] = m0;
        hp_mem[2] = m2;
    } else {
        let mut m0_0: f32 = 0.;
        m0_0 = hp_mem[0];
        i = 0;
        while i < len {
            let mut x: opus_val32 = 0.;
            let mut y: opus_val32 = 0.;
            x = in_0[i as usize];
            y = x - m0_0;
            m0_0 = coef * x + VERY_SMALL + coef2 * m0_0;
            out[i as usize] = y;
            i += 1;
        }
        hp_mem[0] = m0_0;
    };
}
/// Upstream C: src/opus_encoder.c:stereo_fade
fn stereo_fade(
    in_0: &[opus_val16],
    out: &mut [opus_val16],
    mut g1: opus_val16,
    mut g2: opus_val16,
    overlap48: i32,
    frame_size: i32,
    channels: i32,
    window: &[opus_val16],
    Fs: i32,
) {
    let mut i: i32 = 0;
    let mut overlap: i32 = 0;
    let mut inc: i32 = 0;
    inc = 48000 / Fs;
    overlap = overlap48 / inc;
    g1 = Q15ONE - g1;
    g2 = Q15ONE - g2;
    i = 0;
    while i < overlap {
        let mut diff: opus_val32 = 0.;
        let mut g: opus_val16 = 0.;
        let mut w: opus_val16 = 0.;
        w = window[(i * inc) as usize] * window[(i * inc) as usize];
        g = w * g2 + (1.0f32 - w) * g1;
        diff = 0.5f32 * (in_0[(i * channels) as usize] - in_0[(i * channels + 1) as usize]);
        diff = g * diff;
        out[(i * channels) as usize] = out[(i * channels) as usize] - diff;
        out[(i * channels + 1) as usize] = out[(i * channels + 1) as usize] + diff;
        i += 1;
    }
    while i < frame_size {
        let mut diff_0: opus_val32 = 0.;
        diff_0 = 0.5f32 * (in_0[(i * channels) as usize] - in_0[(i * channels + 1) as usize]);
        diff_0 = g2 * diff_0;
        out[(i * channels) as usize] = out[(i * channels) as usize] - diff_0;
        out[(i * channels + 1) as usize] = out[(i * channels + 1) as usize] + diff_0;
        i += 1;
    }
}
/// Upstream C: src/opus_encoder.c:gain_fade
fn gain_fade(
    in_0: &[opus_val16],
    out: &mut [opus_val16],
    g1: opus_val16,
    g2: opus_val16,
    overlap48: i32,
    frame_size: i32,
    channels: i32,
    window: &[opus_val16],
    Fs: i32,
) {
    let mut i: i32 = 0;
    let mut inc: i32 = 0;
    let mut overlap: i32 = 0;
    let mut c: i32 = 0;
    inc = 48000 / Fs;
    overlap = overlap48 / inc;
    if channels == 1 {
        i = 0;
        while i < overlap {
            let mut g: opus_val16 = 0.;
            let mut w: opus_val16 = 0.;
            w = window[(i * inc) as usize] * window[(i * inc) as usize];
            g = w * g2 + (1.0f32 - w) * g1;
            out[i as usize] = g * in_0[i as usize];
            i += 1;
        }
    } else {
        i = 0;
        while i < overlap {
            let mut g_0: opus_val16 = 0.;
            let mut w_0: opus_val16 = 0.;
            w_0 = window[(i * inc) as usize] * window[(i * inc) as usize];
            g_0 = w_0 * g2 + (1.0f32 - w_0) * g1;
            out[(i * 2) as usize] = g_0 * in_0[(i * 2) as usize];
            out[(i * 2 + 1) as usize] = g_0 * in_0[(i * 2 + 1) as usize];
            i += 1;
        }
    }
    c = 0;
    loop {
        i = overlap;
        while i < frame_size {
            out[(i * channels + c) as usize] = g2 * in_0[(i * channels + c) as usize];
            i += 1;
        }
        c += 1;
        if !(c < channels) {
            break;
        }
    }
}

/// Upstream C: src/opus_encoder.c:user_bitrate_to_bitrate
fn user_bitrate_to_bitrate(st: &OpusEncoder, mut frame_size: i32, max_data_bytes: i32) -> i32 {
    if frame_size == 0 {
        frame_size = st.Fs / 400;
    }
    if st.user_bitrate_bps == OPUS_AUTO {
        return 60 * st.Fs / frame_size + st.Fs * st.channels;
    } else if st.user_bitrate_bps == OPUS_BITRATE_MAX {
        return max_data_bytes * 8 * st.Fs / frame_size;
    } else {
        return st.user_bitrate_bps;
    };
}

/// Upstream C: src/opus_encoder.c:frame_size_select
pub fn frame_size_select(frame_size: i32, variable_duration: i32, Fs: i32) -> i32 {
    let mut new_size: i32 = 0;
    if frame_size < Fs / 400 {
        return -1;
    }
    if variable_duration == OPUS_FRAMESIZE_ARG {
        new_size = frame_size;
    } else if variable_duration >= OPUS_FRAMESIZE_2_5_MS
        && variable_duration <= OPUS_FRAMESIZE_120_MS
    {
        if variable_duration <= OPUS_FRAMESIZE_40_MS {
            new_size = (Fs / 400) << variable_duration - OPUS_FRAMESIZE_2_5_MS;
        } else {
            new_size = (variable_duration - OPUS_FRAMESIZE_2_5_MS - 2) * Fs / 50;
        }
    } else {
        return -1;
    }
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
    return new_size;
}
/// Upstream C: src/opus_encoder.c:compute_stereo_width
pub fn compute_stereo_width(
    pcm: &[opus_val16],
    frame_size: i32,
    Fs: i32,
    mem: &mut StereoWidthState,
) -> opus_val16 {
    let mut xx: opus_val32 = 0.;
    let mut xy: opus_val32 = 0.;
    let mut yy: opus_val32 = 0.;
    let mut sqrt_xx: opus_val16 = 0.;
    let mut sqrt_yy: opus_val16 = 0.;
    let mut qrrt_xx: opus_val16 = 0.;
    let mut qrrt_yy: opus_val16 = 0.;
    let mut frame_rate: i32 = 0;
    let mut i: i32 = 0;
    let mut short_alpha: opus_val16 = 0.;
    frame_rate = Fs / frame_size;
    short_alpha =
        Q15ONE - 25 as opus_val32 * 1.0f32 / (if 50 > frame_rate { 50 } else { frame_rate }) as f32;
    yy = 0 as opus_val32;
    xy = yy;
    xx = xy;
    i = 0;
    while i < frame_size - 3 {
        let mut pxx: opus_val32 = 0 as opus_val32;
        let mut pxy: opus_val32 = 0 as opus_val32;
        let mut pyy: opus_val32 = 0 as opus_val32;
        let mut x: opus_val16 = 0.;
        let mut y: opus_val16 = 0.;
        x = pcm[(2 * i) as usize];
        y = pcm[(2 * i + 1) as usize];
        pxx = x * x;
        pxy = x * y;
        pyy = y * y;
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
        i += 4;
    }
    if !(xx < 1e9f32) || xx.is_nan() || !(yy < 1e9f32) || yy.is_nan() {
        yy = 0 as opus_val32;
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
        let mut corr: opus_val16 = 0.;
        let mut ldiff: opus_val16 = 0.;
        let mut width: opus_val16 = 0.;
        sqrt_xx = celt_sqrt(mem.XX);
        sqrt_yy = celt_sqrt(mem.YY);
        qrrt_xx = celt_sqrt(sqrt_xx);
        qrrt_yy = celt_sqrt(sqrt_yy);
        mem.XY = if mem.XY < sqrt_xx * sqrt_yy {
            mem.XY
        } else {
            sqrt_xx * sqrt_yy
        };
        corr = mem.XY / (1e-15f32 + sqrt_xx * sqrt_yy);
        ldiff = 1.0f32 * (qrrt_xx - qrrt_yy).abs() / (EPSILON + qrrt_xx + qrrt_yy);
        width = celt_sqrt(1.0f32 - corr * corr) * ldiff;
        mem.smoothed_width += (width - mem.smoothed_width) / frame_rate as f32;
        mem.max_follower = if mem.max_follower - 0.02f32 / frame_rate as f32 > mem.smoothed_width {
            mem.max_follower - 0.02f32 / frame_rate as f32
        } else {
            mem.smoothed_width
        };
    }
    return if 1.0f32 < 20 as opus_val32 * mem.max_follower {
        1.0f32
    } else {
        20 as opus_val32 * mem.max_follower
    };
}
/// Upstream C: src/opus_encoder.c:decide_fec
fn decide_fec(
    useInBandFEC: i32,
    PacketLoss_perc: i32,
    last_fec: i32,
    mode: i32,
    bandwidth: &mut i32,
    rate: i32,
) -> i32 {
    let mut orig_bandwidth: i32 = 0;
    if useInBandFEC == 0 || PacketLoss_perc == 0 || mode == MODE_CELT_ONLY {
        return 0;
    }
    orig_bandwidth = *bandwidth;
    loop {
        let mut hysteresis: i32 = 0;
        let mut LBRR_rate_thres_bps: i32 = 0;
        LBRR_rate_thres_bps =
            fec_thresholds[(2 * (*bandwidth - OPUS_BANDWIDTH_NARROWBAND)) as usize];
        hysteresis = fec_thresholds[(2 * (*bandwidth - OPUS_BANDWIDTH_NARROWBAND) + 1) as usize];
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
    return 0;
}
/// Upstream C: src/opus_encoder.c:compute_silk_rate_for_hybrid
fn compute_silk_rate_for_hybrid(
    mut rate: i32,
    bandwidth: i32,
    frame20ms: i32,
    vbr: i32,
    fec: i32,
    channels: i32,
) -> i32 {
    let mut entry: i32 = 0;
    let mut i: i32 = 0;
    let mut N: i32 = 0;
    let mut silk_rate: i32 = 0;
    static rate_table: [[i32; 5]; 7] = [
        [0, 0, 0, 0, 0],
        [12000, 10000, 10000, 11000, 11000],
        [16000, 13500, 13500, 15000, 15000],
        [20000, 16000, 16000, 18000, 18000],
        [24000, 18000, 18000, 21000, 21000],
        [32000, 22000, 22000, 28000, 28000],
        [64000, 38000, 38000, 50000, 50000],
    ];
    rate /= channels;
    entry = 1 + frame20ms + 2 * fec;
    N = (::core::mem::size_of::<[[i32; 5]; 7]>() as u64)
        .wrapping_div(::core::mem::size_of::<[i32; 5]>() as u64) as i32;
    i = 1;
    while i < N {
        if rate_table[i as usize][0 as usize] > rate {
            break;
        }
        i += 1;
    }
    if i == N {
        silk_rate = rate_table[(i - 1) as usize][entry as usize];
        silk_rate += (rate - rate_table[(i - 1) as usize][0 as usize]) / 2;
    } else {
        let mut lo: i32 = 0;
        let mut hi: i32 = 0;
        let mut x0: i32 = 0;
        let mut x1: i32 = 0;
        lo = rate_table[(i - 1) as usize][entry as usize];
        hi = rate_table[i as usize][entry as usize];
        x0 = rate_table[(i - 1) as usize][0 as usize];
        x1 = rate_table[i as usize][0 as usize];
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
    return silk_rate;
}
/// Upstream C: src/opus_encoder.c:compute_equiv_rate
fn compute_equiv_rate(
    bitrate: i32,
    channels: i32,
    frame_rate: i32,
    vbr: i32,
    mode: i32,
    complexity: i32,
    loss: i32,
) -> i32 {
    let mut equiv: i32 = 0;
    equiv = bitrate;
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
    return equiv;
}
/// Upstream C: src/analysis.c:is_digital_silence
pub fn is_digital_silence(
    pcm: &[opus_val16],
    frame_size: i32,
    channels: i32,
    lsb_depth: i32,
) -> i32 {
    let sample_max = celt_maxabs16(&pcm[..(frame_size * channels) as usize]);
    (sample_max <= 1 as opus_val16 / ((1) << lsb_depth) as f32) as i32
}
/// Upstream C: src/opus_encoder.c:compute_frame_energy
fn compute_frame_energy(
    pcm: &[opus_val16],
    frame_size: i32,
    channels: i32,
    _arch: i32,
) -> opus_val32 {
    let len = (frame_size * channels) as usize;
    let s = &pcm[..len];
    celt_inner_prod(s, s, len) / len as f32
}
/// Upstream C: src/opus_encoder.c:decide_dtx_mode
fn decide_dtx_mode(
    activity_probability: f32,
    nb_no_activity_frames: &mut i32,
    peak_signal_energy: opus_val32,
    pcm: &[opus_val16],
    frame_size: i32,
    channels: i32,
    mut is_silence: i32,
    arch: i32,
) -> i32 {
    let mut noise_energy: opus_val32 = 0.;
    if is_silence == 0 {
        if activity_probability < DTX_ACTIVITY_THRESHOLD {
            noise_energy = compute_frame_energy(pcm, frame_size, channels, arch);
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
    return 0;
}
/// Upstream C: src/opus_encoder.c:encode_multiframe_packet
fn encode_multiframe_packet(
    st: &mut OpusEncoder,
    pcm: &[opus_val16],
    nb_frames: i32,
    frame_size: i32,
    data: &mut [u8],
    out_data_bytes: i32,
    to_celt: i32,
    lsb_depth: i32,
    float_api: i32,
) -> i32 {
    let mut ret: i32 = 0;

    // Worst cases:
    // 2 frames: Code 2 with different compressed sizes
    // >2 frames: Code 3 VBR
    let max_header_bytes = if nb_frames == 2 {
        3
    } else {
        2 + (nb_frames - 1) * 2
    };
    let repacketize_len;
    if st.use_vbr != 0 || st.user_bitrate_bps == OPUS_BITRATE_MAX {
        repacketize_len = out_data_bytes;
    } else {
        let cbr_bytes = 3 * st.bitrate_bps / (3 * 8 * st.Fs / (frame_size * nb_frames));
        repacketize_len = if cbr_bytes < out_data_bytes {
            cbr_bytes
        } else {
            out_data_bytes
        };
    }
    let bytes_per_frame = if 1276 < 1 + (repacketize_len - max_header_bytes) / nb_frames {
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
        let pcm_offset = (i * (st.channels * frame_size)) as usize;

        // When switching from SILK/Hybrid to CELT, only ask for a switch at the last frame
        if to_celt != 0 && i == nb_frames - 1 {
            st.user_forced_mode = MODE_CELT_ONLY;
        }
        // SAFETY: opus_encode_native still uses raw pointers internally
        let tmp_len = unsafe {
            opus_encode_native(
                st as *mut OpusEncoder,
                pcm[pcm_offset..].as_ptr(),
                frame_size,
                tmp_data[start..].as_mut_ptr(),
                bytes_per_frame,
                lsb_depth,
                std::ptr::null(),
                0,
                0,
                0,
                0,
                float_api,
            )
        };
        if tmp_len < 0 {
            return OPUS_INTERNAL_ERROR;
        }

        ret = rp.cat(&tmp_data[start..start + tmp_len as usize]);
        offsets.push((start, start + tmp_len as usize));
        if ret < 0 {
            return OPUS_INTERNAL_ERROR;
        }
    }

    let offsets = offsets
        .into_iter()
        .map(|(start, end)| &tmp_data[start..end])
        .collect();
    ret = rp.out_range_impl(
        0,
        nb_frames,
        &mut data[..repacketize_len as usize],
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

/// Upstream C: src/opus_encoder.c:compute_redundancy_bytes
fn compute_redundancy_bytes(
    max_data_bytes: i32,
    bitrate_bps: i32,
    frame_rate: i32,
    channels: i32,
) -> i32 {
    let mut redundancy_bytes_cap: i32 = 0;
    let mut redundancy_bytes: i32 = 0;
    let mut redundancy_rate: i32 = 0;
    let mut base_bits: i32 = 0;
    let mut available_bits: i32 = 0;
    base_bits = 40 * channels + 20;
    redundancy_rate = bitrate_bps + base_bits * (200 - frame_rate);
    redundancy_rate = 3 * redundancy_rate / 2;
    redundancy_bytes = redundancy_rate / 1600;
    available_bits = max_data_bytes * 8 - 2 * base_bits;
    redundancy_bytes_cap = (available_bits * 240 / (240 + 48000 / frame_rate) + base_bits) / 8;
    redundancy_bytes = if redundancy_bytes < redundancy_bytes_cap {
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
    return redundancy_bytes;
}
pub unsafe fn opus_encode_native(
    st: *mut OpusEncoder,
    pcm: *const opus_val16,
    frame_size: i32,
    mut data: *mut u8,
    out_data_bytes: i32,
    mut lsb_depth: i32,
    analysis_pcm: *const core::ffi::c_void,
    analysis_size: i32,
    c1: i32,
    c2: i32,
    analysis_channels: i32,
    float_api: i32,
) -> i32 {
    let mut celt_enc: *mut OpusCustomEncoder = 0 as *mut OpusCustomEncoder;
    let mut i: i32 = 0;
    let mut ret: i32 = 0;
    let mut nBytes: i32 = 0;
    let mut enc: ec_enc = ec_enc {
        buf: &mut [],
        storage: 0,
        end_offs: 0,
        end_window: 0,
        nend_bits: 0,
        nbits_total: 0,
        offs: 0,
        rng: 0,
        val: 0,
        ext: 0,
        rem: 0,
        error: 0,
    };
    let mut bytes_target: i32 = 0;
    let mut prefill: i32 = 0;
    let mut start_band: i32 = 0;
    let mut redundancy: i32 = 0;
    let mut redundancy_bytes: i32 = 0;
    let mut celt_to_silk: i32 = 0;
    let mut nb_compr_bytes: i32 = 0;
    let mut to_celt: i32 = 0;
    let mut redundant_rng: u32 = 0;
    let mut cutoff_Hz: i32 = 0;
    let mut hp_freq_smth1: i32 = 0;
    let mut voice_est: i32 = 0;
    let mut equiv_rate: i32 = 0;
    let mut delay_compensation: i32 = 0;
    let mut frame_rate: i32 = 0;
    let mut max_rate: i32 = 0;
    let mut curr_bandwidth: i32 = 0;
    let mut HB_gain: opus_val16 = 0.;
    let mut max_data_bytes: i32 = 0;
    let mut total_buffer: i32 = 0;
    let mut stereo_width: opus_val16 = 0.;
    let mut celt_mode: *const OpusCustomMode = 0 as *const OpusCustomMode;
    let mut analysis_info: AnalysisInfo = AnalysisInfo {
        valid: 0,
        tonality: 0.,
        tonality_slope: 0.,
        noisiness: 0.,
        activity: 0.,
        music_prob: 0.,
        music_prob_min: 0.,
        music_prob_max: 0.,
        bandwidth: 0,
        activity_probability: 0.,
        max_pitch_ratio: 0.,
        leak_boost: [0; 19],
    };
    let mut analysis_read_pos_bak: i32 = -1;
    let mut analysis_read_subframe_bak: i32 = -1;
    let mut is_silence: i32 = 0;
    max_data_bytes = if (1276) < out_data_bytes {
        1276
    } else {
        out_data_bytes
    };
    (*st).rangeFinal = 0;
    if frame_size <= 0 || max_data_bytes <= 0 {
        return OPUS_BAD_ARG;
    }
    if max_data_bytes == 1 && (*st).Fs == frame_size * 10 {
        return OPUS_BUFFER_TOO_SMALL;
    }
    let silk_enc = &mut (*st).silk_enc;
    celt_enc = &mut (*st).celt_enc;
    if (*st).application == OPUS_APPLICATION_RESTRICTED_LOWDELAY {
        delay_compensation = 0;
    } else {
        delay_compensation = (*st).delay_compensation;
    }
    lsb_depth = if lsb_depth < (*st).lsb_depth {
        lsb_depth
    } else {
        (*st).lsb_depth
    };
    celt_mode = (*celt_enc).mode;
    analysis_info.valid = 0;
    if (*st).silk_mode.complexity >= 7 && (*st).Fs >= 16000 {
        is_silence = is_digital_silence(
            std::slice::from_raw_parts(pcm, (frame_size * (*st).channels) as usize),
            frame_size,
            (*st).channels,
            lsb_depth,
        );
        analysis_read_pos_bak = (*st).analysis.read_pos;
        analysis_read_subframe_bak = (*st).analysis.read_subframe;
        {
            let input = if analysis_pcm.is_null() {
                None
            } else if float_api != 0 {
                let pcm_slice = std::slice::from_raw_parts(
                    analysis_pcm as *const f32,
                    (analysis_size * analysis_channels) as usize,
                );
                Some(DownmixInput::Float(pcm_slice))
            } else {
                let pcm_slice = std::slice::from_raw_parts(
                    analysis_pcm as *const i16,
                    (analysis_size * analysis_channels) as usize,
                );
                Some(DownmixInput::Int(pcm_slice))
            };
            run_analysis(
                &mut (*st).analysis,
                &*celt_mode,
                input.as_ref(),
                analysis_size,
                frame_size,
                c1,
                c2,
                analysis_channels,
                (*st).Fs,
                lsb_depth,
                &mut analysis_info,
            );
        }
        if is_silence == 0 && analysis_info.activity_probability > DTX_ACTIVITY_THRESHOLD {
            let pcm_slice = std::slice::from_raw_parts(pcm, (frame_size * (*st).channels) as usize);
            (*st).peak_signal_energy = if 0.999f32 * (*st).peak_signal_energy
                > compute_frame_energy(pcm_slice, frame_size, (*st).channels, (*st).arch)
            {
                0.999f32 * (*st).peak_signal_energy
            } else {
                compute_frame_energy(pcm_slice, frame_size, (*st).channels, (*st).arch)
            };
        }
    } else if (*st).analysis.initialized != 0 {
        tonality_analysis_reset(&mut (*st).analysis);
    }
    if is_silence == 0 {
        (*st).voice_ratio = -1;
    }
    (*st).detected_bandwidth = 0;
    if analysis_info.valid != 0 {
        let mut analysis_bandwidth: i32 = 0;
        if (*st).signal_type == OPUS_AUTO {
            let mut prob: f32 = 0.;
            if (*st).prev_mode == 0 {
                prob = analysis_info.music_prob;
            } else if (*st).prev_mode == MODE_CELT_ONLY {
                prob = analysis_info.music_prob_max;
            } else {
                prob = analysis_info.music_prob_min;
            }
            (*st).voice_ratio = (0.5 + (100.0 * (1.0 - prob))).floor() as i32;
        }
        analysis_bandwidth = analysis_info.bandwidth;
        if analysis_bandwidth <= 12 {
            (*st).detected_bandwidth = OPUS_BANDWIDTH_NARROWBAND;
        } else if analysis_bandwidth <= 14 {
            (*st).detected_bandwidth = OPUS_BANDWIDTH_MEDIUMBAND;
        } else if analysis_bandwidth <= 16 {
            (*st).detected_bandwidth = OPUS_BANDWIDTH_WIDEBAND;
        } else if analysis_bandwidth <= 18 {
            (*st).detected_bandwidth = OPUS_BANDWIDTH_SUPERWIDEBAND;
        } else {
            (*st).detected_bandwidth = OPUS_BANDWIDTH_FULLBAND;
        }
    }
    if (*st).channels == 2 && (*st).force_channels != 1 {
        stereo_width = compute_stereo_width(
            std::slice::from_raw_parts(pcm, (frame_size * 2) as usize),
            frame_size,
            (*st).Fs,
            &mut (*st).width_mem,
        );
    } else {
        stereo_width = 0 as opus_val16;
    }
    total_buffer = delay_compensation;
    (*st).bitrate_bps = user_bitrate_to_bitrate(&*st, frame_size, max_data_bytes);
    frame_rate = (*st).Fs / frame_size;
    if (*st).use_vbr == 0 {
        let mut cbrBytes: i32 = 0;
        let frame_rate12: i32 = 12 * (*st).Fs / frame_size;
        cbrBytes =
            if (12 * (*st).bitrate_bps / 8 + frame_rate12 / 2) / frame_rate12 < max_data_bytes {
                (12 * (*st).bitrate_bps / 8 + frame_rate12 / 2) / frame_rate12
            } else {
                max_data_bytes
            };
        (*st).bitrate_bps = cbrBytes * frame_rate12 * 8 / 12;
        max_data_bytes = if 1 > cbrBytes { 1 } else { cbrBytes };
    }
    if max_data_bytes < 3
        || (*st).bitrate_bps < 3 * frame_rate * 8
        || frame_rate < 50 && (max_data_bytes * frame_rate < 300 || (*st).bitrate_bps < 2400)
    {
        let mut tocmode: i32 = (*st).mode;
        let mut bw: i32 = if (*st).bandwidth == 0 {
            OPUS_BANDWIDTH_NARROWBAND
        } else {
            (*st).bandwidth
        };
        let mut packet_code: i32 = 0;
        let mut num_multiframes: i32 = 0;
        if tocmode == 0 {
            tocmode = MODE_SILK_ONLY;
        }
        if frame_rate > 100 {
            tocmode = MODE_CELT_ONLY;
        }
        if frame_rate == 25 && tocmode != MODE_SILK_ONLY {
            frame_rate = 50;
            packet_code = 1;
        }
        if frame_rate <= 16 {
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
        *data.offset(0 as isize) = gen_toc(tocmode, frame_rate, bw, (*st).stream_channels);
        let ref mut fresh4 = *data.offset(0 as isize);
        *fresh4 = (*fresh4 as i32 | packet_code) as u8;
        ret = if packet_code <= 1 { 1 } else { 2 };
        max_data_bytes = if max_data_bytes > ret {
            max_data_bytes
        } else {
            ret
        };
        if packet_code == 3 {
            *data.offset(1 as isize) = num_multiframes as u8;
        }
        if (*st).use_vbr == 0 {
            let data = std::slice::from_raw_parts_mut(data, max_data_bytes as _);
            ret = opus_packet_pad(data, ret, max_data_bytes);
            if ret == OPUS_OK {
                ret = max_data_bytes;
            } else {
                ret = OPUS_INTERNAL_ERROR;
            }
        }
        return ret;
    }
    max_rate = frame_rate * max_data_bytes * 8;
    equiv_rate = compute_equiv_rate(
        (*st).bitrate_bps,
        (*st).channels,
        (*st).Fs / frame_size,
        (*st).use_vbr,
        0,
        (*st).silk_mode.complexity,
        (*st).silk_mode.packetLossPercentage,
    );
    if (*st).signal_type == OPUS_SIGNAL_VOICE {
        voice_est = 127;
    } else if (*st).signal_type == OPUS_SIGNAL_MUSIC {
        voice_est = 0;
    } else if (*st).voice_ratio >= 0 {
        voice_est = (*st).voice_ratio * 327 >> 8;
        if (*st).application == OPUS_APPLICATION_AUDIO {
            voice_est = if voice_est < 115 { voice_est } else { 115 };
        }
    } else if (*st).application == OPUS_APPLICATION_VOIP {
        voice_est = 115;
    } else {
        voice_est = 48;
    }
    if (*st).force_channels != OPUS_AUTO && (*st).channels == 2 {
        (*st).stream_channels = (*st).force_channels;
    } else if (*st).channels == 2 {
        let mut stereo_threshold: i32 = 0;
        stereo_threshold = stereo_music_threshold
            + (voice_est * voice_est * (stereo_voice_threshold - stereo_music_threshold) >> 14);
        if (*st).stream_channels == 2 {
            stereo_threshold -= 1000;
        } else {
            stereo_threshold += 1000;
        }
        (*st).stream_channels = if equiv_rate > stereo_threshold { 2 } else { 1 };
    } else {
        (*st).stream_channels = (*st).channels;
    }
    equiv_rate = compute_equiv_rate(
        (*st).bitrate_bps,
        (*st).stream_channels,
        (*st).Fs / frame_size,
        (*st).use_vbr,
        0,
        (*st).silk_mode.complexity,
        (*st).silk_mode.packetLossPercentage,
    );
    (*st).silk_mode.useDTX =
        ((*st).use_dtx != 0 && !(analysis_info.valid != 0 || is_silence != 0)) as i32;
    if (*st).application == OPUS_APPLICATION_RESTRICTED_LOWDELAY {
        (*st).mode = MODE_CELT_ONLY;
    } else if (*st).user_forced_mode == OPUS_AUTO {
        let mut mode_voice: i32 = 0;
        let mut mode_music: i32 = 0;
        let mut threshold: i32 = 0;
        mode_voice = ((1.0f32 - stereo_width) * mode_thresholds[0 as usize][0 as usize] as f32
            + stereo_width * mode_thresholds[1 as usize][0 as usize] as f32)
            as i32;
        mode_music = ((1.0f32 - stereo_width) * mode_thresholds[1 as usize][1 as usize] as f32
            + stereo_width * mode_thresholds[1 as usize][1 as usize] as f32)
            as i32;
        threshold = mode_music + (voice_est * voice_est * (mode_voice - mode_music) >> 14);
        if (*st).application == OPUS_APPLICATION_VOIP {
            threshold += 8000;
        }
        if (*st).prev_mode == MODE_CELT_ONLY {
            threshold -= 4000;
        } else if (*st).prev_mode > 0 {
            threshold += 4000;
        }
        (*st).mode = if equiv_rate >= threshold {
            MODE_CELT_ONLY
        } else {
            MODE_SILK_ONLY
        };
        if (*st).silk_mode.useInBandFEC != 0
            && (*st).silk_mode.packetLossPercentage > 128 - voice_est >> 4
        {
            (*st).mode = MODE_SILK_ONLY;
        }
        if (*st).silk_mode.useDTX != 0 && voice_est > 100 {
            (*st).mode = MODE_SILK_ONLY;
        }
        if max_data_bytes
            < (if frame_rate > 50 { 9000 } else { 6000 }) * frame_size / ((*st).Fs * 8)
        {
            (*st).mode = MODE_CELT_ONLY;
        }
    } else {
        (*st).mode = (*st).user_forced_mode;
    }
    if (*st).mode != MODE_CELT_ONLY && frame_size < (*st).Fs / 100 {
        (*st).mode = MODE_CELT_ONLY;
    }
    if (*st).lfe != 0 {
        (*st).mode = MODE_CELT_ONLY;
    }
    if (*st).prev_mode > 0
        && ((*st).mode != MODE_CELT_ONLY && (*st).prev_mode == MODE_CELT_ONLY
            || (*st).mode == MODE_CELT_ONLY && (*st).prev_mode != MODE_CELT_ONLY)
    {
        redundancy = 1;
        celt_to_silk = ((*st).mode != MODE_CELT_ONLY) as i32;
        if celt_to_silk == 0 {
            if frame_size >= (*st).Fs / 100 {
                (*st).mode = (*st).prev_mode;
                to_celt = 1;
            } else {
                redundancy = 0;
            }
        }
    }
    if (*st).stream_channels == 1
        && (*st).prev_channels == 2
        && (*st).silk_mode.toMono == 0
        && (*st).mode != MODE_CELT_ONLY
        && (*st).prev_mode != MODE_CELT_ONLY
    {
        (*st).silk_mode.toMono = 1;
        (*st).stream_channels = 2;
    } else {
        (*st).silk_mode.toMono = 0;
    }
    equiv_rate = compute_equiv_rate(
        (*st).bitrate_bps,
        (*st).stream_channels,
        (*st).Fs / frame_size,
        (*st).use_vbr,
        (*st).mode,
        (*st).silk_mode.complexity,
        (*st).silk_mode.packetLossPercentage,
    );
    if (*st).mode != MODE_CELT_ONLY && (*st).prev_mode == MODE_CELT_ONLY {
        let mut dummy: silk_EncControlStruct = silk_EncControlStruct {
            nChannelsAPI: 0,
            nChannelsInternal: 0,
            API_sampleRate: 0,
            maxInternalSampleRate: 0,
            minInternalSampleRate: 0,
            desiredInternalSampleRate: 0,
            payloadSize_ms: 0,
            bitRate: 0,
            packetLossPercentage: 0,
            complexity: 0,
            useInBandFEC: 0,
            LBRR_coded: 0,
            useDTX: 0,
            useCBR: 0,
            maxBits: 0,
            toMono: 0,
            opusCanSwitch: 0,
            reducedDependency: 0,
            internalSampleRate: 0,
            allowBandwidthSwitch: 0,
            inWBmodeWithoutVariableLP: 0,
            stereoWidth_Q14: 0,
            switchReady: 0,
            signalType: 0,
            offset: 0,
        };
        silk_InitEncoder(silk_enc, (*st).arch, &mut dummy);
        prefill = 1;
    }
    if (*st).mode == MODE_CELT_ONLY || (*st).first != 0 || (*st).silk_mode.allowBandwidthSwitch != 0
    {
        let mut voice_bandwidth_thresholds: *const i32 = 0 as *const i32;
        let mut music_bandwidth_thresholds: *const i32 = 0 as *const i32;
        let mut bandwidth_thresholds: [i32; 8] = [0; 8];
        let mut bandwidth: i32 = OPUS_BANDWIDTH_FULLBAND;
        if (*st).channels == 2 && (*st).force_channels != 1 {
            voice_bandwidth_thresholds = stereo_voice_bandwidth_thresholds.as_ptr();
            music_bandwidth_thresholds = stereo_music_bandwidth_thresholds.as_ptr();
        } else {
            voice_bandwidth_thresholds = mono_voice_bandwidth_thresholds.as_ptr();
            music_bandwidth_thresholds = mono_music_bandwidth_thresholds.as_ptr();
        }
        i = 0;
        while i < 8 {
            bandwidth_thresholds[i as usize] = *music_bandwidth_thresholds.offset(i as isize)
                + (voice_est
                    * voice_est
                    * (*voice_bandwidth_thresholds.offset(i as isize)
                        - *music_bandwidth_thresholds.offset(i as isize))
                    >> 14);
            i += 1;
        }
        loop {
            let mut threshold_0: i32 = 0;
            let mut hysteresis: i32 = 0;
            threshold_0 =
                bandwidth_thresholds[(2 * (bandwidth - OPUS_BANDWIDTH_MEDIUMBAND)) as usize];
            hysteresis =
                bandwidth_thresholds[(2 * (bandwidth - OPUS_BANDWIDTH_MEDIUMBAND) + 1) as usize];
            if (*st).first == 0 {
                if (*st).auto_bandwidth >= bandwidth {
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
        if bandwidth == OPUS_BANDWIDTH_MEDIUMBAND {
            bandwidth = OPUS_BANDWIDTH_WIDEBAND;
        }
        (*st).auto_bandwidth = bandwidth;
        (*st).bandwidth = (*st).auto_bandwidth;
        if (*st).first == 0
            && (*st).mode != MODE_CELT_ONLY
            && (*st).silk_mode.inWBmodeWithoutVariableLP == 0
            && (*st).bandwidth > OPUS_BANDWIDTH_WIDEBAND
        {
            (*st).bandwidth = OPUS_BANDWIDTH_WIDEBAND;
        }
    }
    if (*st).bandwidth > (*st).max_bandwidth {
        (*st).bandwidth = (*st).max_bandwidth;
    }
    if (*st).user_bandwidth != OPUS_AUTO {
        (*st).bandwidth = (*st).user_bandwidth;
    }
    if (*st).mode != MODE_CELT_ONLY && max_rate < 15000 {
        (*st).bandwidth = if (*st).bandwidth < 1103 {
            (*st).bandwidth
        } else {
            1103
        };
    }
    if (*st).Fs <= 24000 && (*st).bandwidth > OPUS_BANDWIDTH_SUPERWIDEBAND {
        (*st).bandwidth = OPUS_BANDWIDTH_SUPERWIDEBAND;
    }
    if (*st).Fs <= 16000 && (*st).bandwidth > OPUS_BANDWIDTH_WIDEBAND {
        (*st).bandwidth = OPUS_BANDWIDTH_WIDEBAND;
    }
    if (*st).Fs <= 12000 && (*st).bandwidth > OPUS_BANDWIDTH_MEDIUMBAND {
        (*st).bandwidth = OPUS_BANDWIDTH_MEDIUMBAND;
    }
    if (*st).Fs <= 8000 && (*st).bandwidth > OPUS_BANDWIDTH_NARROWBAND {
        (*st).bandwidth = OPUS_BANDWIDTH_NARROWBAND;
    }
    if (*st).detected_bandwidth != 0 && (*st).user_bandwidth == OPUS_AUTO {
        let mut min_detected_bandwidth: i32 = 0;
        if equiv_rate <= 18000 * (*st).stream_channels && (*st).mode == MODE_CELT_ONLY {
            min_detected_bandwidth = OPUS_BANDWIDTH_NARROWBAND;
        } else if equiv_rate <= 24000 * (*st).stream_channels && (*st).mode == MODE_CELT_ONLY {
            min_detected_bandwidth = OPUS_BANDWIDTH_MEDIUMBAND;
        } else if equiv_rate <= 30000 * (*st).stream_channels {
            min_detected_bandwidth = OPUS_BANDWIDTH_WIDEBAND;
        } else if equiv_rate <= 44000 * (*st).stream_channels {
            min_detected_bandwidth = OPUS_BANDWIDTH_SUPERWIDEBAND;
        } else {
            min_detected_bandwidth = OPUS_BANDWIDTH_FULLBAND;
        }
        (*st).detected_bandwidth = if (*st).detected_bandwidth > min_detected_bandwidth {
            (*st).detected_bandwidth
        } else {
            min_detected_bandwidth
        };
        (*st).bandwidth = if (*st).bandwidth < (*st).detected_bandwidth {
            (*st).bandwidth
        } else {
            (*st).detected_bandwidth
        };
    }
    (*st).silk_mode.LBRR_coded = decide_fec(
        (*st).silk_mode.useInBandFEC,
        (*st).silk_mode.packetLossPercentage,
        (*st).silk_mode.LBRR_coded,
        (*st).mode,
        &mut (*st).bandwidth,
        equiv_rate,
    );
    (*celt_enc).lsb_depth = lsb_depth;
    if (*st).mode == MODE_CELT_ONLY && (*st).bandwidth == OPUS_BANDWIDTH_MEDIUMBAND {
        (*st).bandwidth = OPUS_BANDWIDTH_WIDEBAND;
    }
    if (*st).lfe != 0 {
        (*st).bandwidth = OPUS_BANDWIDTH_NARROWBAND;
    }
    curr_bandwidth = (*st).bandwidth;
    if (*st).mode == MODE_SILK_ONLY && curr_bandwidth > OPUS_BANDWIDTH_WIDEBAND {
        (*st).mode = MODE_HYBRID;
    }
    if (*st).mode == MODE_HYBRID && curr_bandwidth <= OPUS_BANDWIDTH_WIDEBAND {
        (*st).mode = MODE_SILK_ONLY;
    }
    if frame_size > (*st).Fs / 50 && (*st).mode != MODE_SILK_ONLY || frame_size > 3 * (*st).Fs / 50
    {
        let mut enc_frame_size: i32 = 0;
        let mut nb_frames: i32 = 0;
        if (*st).mode == MODE_SILK_ONLY {
            if frame_size == 2 * (*st).Fs / 25 {
                enc_frame_size = (*st).Fs / 25;
            } else if frame_size == 3 * (*st).Fs / 25 {
                enc_frame_size = 3 * (*st).Fs / 50;
            } else {
                enc_frame_size = (*st).Fs / 50;
            }
        } else {
            enc_frame_size = (*st).Fs / 50;
        }
        nb_frames = frame_size / enc_frame_size;
        if analysis_read_pos_bak != -1 {
            (*st).analysis.read_pos = analysis_read_pos_bak;
            (*st).analysis.read_subframe = analysis_read_subframe_bak;
        }
        ret = encode_multiframe_packet(
            &mut *st,
            std::slice::from_raw_parts(pcm, (frame_size * (*st).channels) as usize),
            nb_frames,
            enc_frame_size,
            std::slice::from_raw_parts_mut(data, out_data_bytes as usize),
            out_data_bytes,
            to_celt,
            lsb_depth,
            float_api,
        );
        return ret;
    }
    if (*st).silk_bw_switch != 0 {
        redundancy = 1;
        celt_to_silk = 1;
        (*st).silk_bw_switch = 0;
        prefill = 2;
    }
    if (*st).mode == MODE_CELT_ONLY {
        redundancy = 0;
    }
    if redundancy != 0 {
        redundancy_bytes = compute_redundancy_bytes(
            max_data_bytes,
            (*st).bitrate_bps,
            frame_rate,
            (*st).stream_channels,
        );
        if redundancy_bytes == 0 {
            redundancy = 0;
        }
    }
    bytes_target =
        (if max_data_bytes - redundancy_bytes < (*st).bitrate_bps * frame_size / ((*st).Fs * 8) {
            max_data_bytes - redundancy_bytes
        } else {
            (*st).bitrate_bps * frame_size / ((*st).Fs * 8)
        }) - 1;
    data = data.offset(1 as isize);
    enc = ec_enc_init(std::slice::from_raw_parts_mut(
        data,
        (max_data_bytes - 1) as usize,
    ));
    let vla = ((total_buffer + frame_size) * (*st).channels) as usize;
    let mut pcm_buf: Vec<opus_val16> = ::std::vec::from_elem(0., vla);
    {
        let src_off = (((*st).encoder_buffer - total_buffer) * (*st).channels) as usize;
        let len = (total_buffer * (*st).channels) as usize;
        pcm_buf[..len].copy_from_slice(&(&(*st).delay_buffer)[src_off..src_off + len]);
    }
    if (*st).mode == MODE_CELT_ONLY {
        hp_freq_smth1 = ((silk_lin2log(VARIABLE_HP_MIN_CUTOFF_HZ) as u32) << 8) as i32;
    } else {
        hp_freq_smth1 = silk_enc.state_Fxx[0].sCmn.variable_HP_smth1_Q15;
    }
    (*st).variable_HP_smth2_Q15 = ((*st).variable_HP_smth2_Q15 as i64
        + ((hp_freq_smth1 - (*st).variable_HP_smth2_Q15) as i64
            * ((VARIABLE_HP_SMTH_COEF2 * ((1) << 16) as f32) as f64 + 0.5f64) as i32 as i16 as i64
            >> 16)) as i32;
    cutoff_Hz = silk_log2lin((*st).variable_HP_smth2_Q15 >> 8);
    {
        let pcm_slice = std::slice::from_raw_parts(pcm, (frame_size * (*st).channels) as usize);
        let out_off = (total_buffer * (*st).channels) as usize;
        if (*st).application == OPUS_APPLICATION_VOIP {
            hp_cutoff(
                pcm_slice,
                cutoff_Hz,
                &mut pcm_buf[out_off..],
                &mut (*st).hp_mem,
                frame_size,
                (*st).channels,
                (*st).Fs,
                (*st).arch,
            );
        } else {
            dc_reject(
                pcm_slice,
                3,
                &mut pcm_buf[out_off..],
                &mut (*st).hp_mem,
                frame_size,
                (*st).channels,
                (*st).Fs,
            );
        }
    }
    if float_api != 0 {
        let mut sum: opus_val32 = 0.;
        {
            let off = (total_buffer * (*st).channels) as usize;
            let n = (frame_size * (*st).channels) as usize;
            sum = celt_inner_prod(&pcm_buf[off..], &pcm_buf[off..], n);
        }
        if !(sum < 1e9f32) || sum.is_nan() {
            {
                let off = (total_buffer * (*st).channels) as usize;
                let len = (frame_size * (*st).channels) as usize;
                pcm_buf[off..off + len].fill(0.0);
            }
            (*st).hp_mem[3 as usize] = 0 as opus_val32;
            (*st).hp_mem[2 as usize] = (*st).hp_mem[3 as usize];
            (*st).hp_mem[1 as usize] = (*st).hp_mem[2 as usize];
            (*st).hp_mem[0 as usize] = (*st).hp_mem[1 as usize];
        }
    }
    HB_gain = Q15ONE;
    if (*st).mode != MODE_CELT_ONLY {
        let mut total_bitRate: i32 = 0;
        let mut celt_rate: i32 = 0;
        let mut activity: i32 = 0;
        let vla_0 = ((*st).channels * frame_size) as usize;
        let mut pcm_silk: Vec<i16> = ::std::vec::from_elem(0, vla_0);
        activity = VAD_NO_DECISION;
        if analysis_info.valid != 0 {
            activity = (analysis_info.activity_probability >= DTX_ACTIVITY_THRESHOLD) as i32;
        }
        total_bitRate = 8 * bytes_target * frame_rate;
        if (*st).mode == MODE_HYBRID {
            (*st).silk_mode.bitRate = compute_silk_rate_for_hybrid(
                total_bitRate,
                curr_bandwidth,
                ((*st).Fs == 50 * frame_size) as i32,
                (*st).use_vbr,
                (*st).silk_mode.LBRR_coded,
                (*st).stream_channels,
            );
            if (*st).energy_masking_len == 0 {
                celt_rate = total_bitRate - (*st).silk_mode.bitRate;
                HB_gain = Q15ONE - celt_exp2(-celt_rate as f32 * (1.0f32 / 1024f32));
            }
        } else {
            (*st).silk_mode.bitRate = total_bitRate;
        }
        if (*st).energy_masking_len > 0 && (*st).use_vbr != 0 && (*st).lfe == 0 {
            let mut mask_sum: opus_val32 = 0 as opus_val32;
            let mut masking_depth: opus_val16 = 0.;
            let mut rate_offset: i32 = 0;
            let mut c: i32 = 0;
            let mut end: i32 = 17;
            let mut srate: i16 = 16000;
            if (*st).bandwidth == OPUS_BANDWIDTH_NARROWBAND {
                end = 13;
                srate = 8000;
            } else if (*st).bandwidth == OPUS_BANDWIDTH_MEDIUMBAND {
                end = 15;
                srate = 12000;
            }
            c = 0;
            while c < (*st).channels {
                i = 0;
                while i < end {
                    let mut mask: opus_val16 = 0.;
                    let em_val = (*st).energy_masking[(21 * c + i) as usize];
                    mask = if (if em_val < 0.5f32 { em_val } else { 0.5f32 }) > -2.0f32 {
                        if em_val < 0.5f32 {
                            em_val
                        } else {
                            0.5f32
                        }
                    } else {
                        -2.0f32
                    };
                    if mask > 0 as f32 {
                        mask = 0.5f32 * mask;
                    }
                    mask_sum += mask;
                    i += 1;
                }
                c += 1;
            }
            masking_depth = mask_sum / end as f32 * (*st).channels as f32;
            masking_depth += 0.2f32;
            rate_offset = (srate as opus_val32 * masking_depth) as i32;
            rate_offset = if rate_offset > -(2) * (*st).silk_mode.bitRate / 3 {
                rate_offset
            } else {
                -(2) * (*st).silk_mode.bitRate / 3
            };
            if (*st).bandwidth == OPUS_BANDWIDTH_SUPERWIDEBAND
                || (*st).bandwidth == OPUS_BANDWIDTH_FULLBAND
            {
                (*st).silk_mode.bitRate += 3 * rate_offset / 5;
            } else {
                (*st).silk_mode.bitRate += rate_offset;
            }
        }
        (*st).silk_mode.payloadSize_ms = 1000 * frame_size / (*st).Fs;
        (*st).silk_mode.nChannelsAPI = (*st).channels;
        (*st).silk_mode.nChannelsInternal = (*st).stream_channels;
        if curr_bandwidth == OPUS_BANDWIDTH_NARROWBAND {
            (*st).silk_mode.desiredInternalSampleRate = 8000;
        } else if curr_bandwidth == OPUS_BANDWIDTH_MEDIUMBAND {
            (*st).silk_mode.desiredInternalSampleRate = 12000;
        } else {
            assert!((*st).mode == 1001 || curr_bandwidth == 1103);
            (*st).silk_mode.desiredInternalSampleRate = 16000;
        }
        if (*st).mode == MODE_HYBRID {
            (*st).silk_mode.minInternalSampleRate = 16000;
        } else {
            (*st).silk_mode.minInternalSampleRate = 8000;
        }
        (*st).silk_mode.maxInternalSampleRate = 16000;
        if (*st).mode == MODE_SILK_ONLY {
            let mut effective_max_rate: i32 = max_rate;
            if frame_rate > 50 {
                effective_max_rate = effective_max_rate * 2 / 3;
            }
            if effective_max_rate < 8000 {
                (*st).silk_mode.maxInternalSampleRate = 12000;
                (*st).silk_mode.desiredInternalSampleRate =
                    if (12000) < (*st).silk_mode.desiredInternalSampleRate {
                        12000
                    } else {
                        (*st).silk_mode.desiredInternalSampleRate
                    };
            }
            if effective_max_rate < 7000 {
                (*st).silk_mode.maxInternalSampleRate = 8000;
                (*st).silk_mode.desiredInternalSampleRate =
                    if (8000) < (*st).silk_mode.desiredInternalSampleRate {
                        8000
                    } else {
                        (*st).silk_mode.desiredInternalSampleRate
                    };
            }
        }
        (*st).silk_mode.useCBR = ((*st).use_vbr == 0) as i32;
        (*st).silk_mode.maxBits = (max_data_bytes - 1) * 8;
        if redundancy != 0 && redundancy_bytes >= 2 {
            (*st).silk_mode.maxBits -= redundancy_bytes * 8 + 1;
            if (*st).mode == MODE_HYBRID {
                (*st).silk_mode.maxBits -= 20;
            }
        }
        if (*st).silk_mode.useCBR != 0 {
            if (*st).mode == MODE_HYBRID {
                (*st).silk_mode.maxBits =
                    if (*st).silk_mode.maxBits < (*st).silk_mode.bitRate * frame_size / (*st).Fs {
                        (*st).silk_mode.maxBits
                    } else {
                        (*st).silk_mode.bitRate * frame_size / (*st).Fs
                    };
            }
        } else if (*st).mode == MODE_HYBRID {
            let maxBitRate: i32 = compute_silk_rate_for_hybrid(
                (*st).silk_mode.maxBits * (*st).Fs / frame_size,
                curr_bandwidth,
                ((*st).Fs == 50 * frame_size) as i32,
                (*st).use_vbr,
                (*st).silk_mode.LBRR_coded,
                (*st).stream_channels,
            );
            (*st).silk_mode.maxBits = maxBitRate * frame_size / (*st).Fs;
        }
        if prefill != 0 {
            let mut zero: i32 = 0;
            let mut prefill_offset: i32 = 0;
            prefill_offset =
                (*st).channels * ((*st).encoder_buffer - (*st).delay_compensation - (*st).Fs / 400);
            {
                let off = prefill_offset as usize;
                let tmp: Vec<opus_val16> = (&(*st).delay_buffer)[off..].to_vec();
                gain_fade(
                    &tmp,
                    &mut (&mut (*st).delay_buffer)[off..],
                    0 as opus_val16,
                    Q15ONE,
                    (*celt_mode).overlap as i32,
                    (*st).Fs / 400,
                    (*st).channels,
                    (*celt_mode).window,
                    (*st).Fs,
                );
            }
            (&mut (*st).delay_buffer)[..prefill_offset as usize].fill(0.0);
            i = 0;
            while i < (*st).encoder_buffer * (*st).channels {
                *pcm_silk.as_mut_ptr().offset(i as isize) =
                    FLOAT2INT16((*st).delay_buffer[i as usize]);
                i += 1;
            }
            silk_Encode(
                silk_enc,
                &mut (*st).silk_mode,
                &pcm_silk,
                (*st).encoder_buffer,
                None,
                &mut zero,
                prefill,
                activity,
            );
            (*st).silk_mode.opusCanSwitch = 0;
        }
        i = 0;
        while i < frame_size * (*st).channels {
            *pcm_silk.as_mut_ptr().offset(i as isize) = FLOAT2INT16(
                *pcm_buf
                    .as_mut_ptr()
                    .offset((total_buffer * (*st).channels + i) as isize),
            );
            i += 1;
        }
        ret = silk_Encode(
            silk_enc,
            &mut (*st).silk_mode,
            &pcm_silk,
            frame_size,
            Some(&mut enc),
            &mut nBytes,
            0,
            activity,
        );
        if ret != 0 {
            return OPUS_INTERNAL_ERROR;
        }
        if (*st).mode == MODE_SILK_ONLY {
            if (*st).silk_mode.internalSampleRate == 8000 {
                curr_bandwidth = OPUS_BANDWIDTH_NARROWBAND;
            } else if (*st).silk_mode.internalSampleRate == 12000 {
                curr_bandwidth = OPUS_BANDWIDTH_MEDIUMBAND;
            } else if (*st).silk_mode.internalSampleRate == 16000 {
                curr_bandwidth = OPUS_BANDWIDTH_WIDEBAND;
            }
        } else {
            assert!((*st).silk_mode.internalSampleRate == 16000)
        };
        (*st).silk_mode.opusCanSwitch =
            ((*st).silk_mode.switchReady != 0 && (*st).nonfinal_frame == 0) as i32;
        if nBytes == 0 {
            (*st).rangeFinal = 0;
            *data.offset(-1 as isize) = gen_toc(
                (*st).mode,
                (*st).Fs / frame_size,
                curr_bandwidth,
                (*st).stream_channels,
            );
            return 1;
        }
        if (*st).silk_mode.opusCanSwitch != 0 {
            redundancy_bytes = compute_redundancy_bytes(
                max_data_bytes,
                (*st).bitrate_bps,
                frame_rate,
                (*st).stream_channels,
            );
            redundancy = (redundancy_bytes != 0) as i32;
            celt_to_silk = 0;
            (*st).silk_bw_switch = 1;
        }
    }
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
    (*celt_enc).end = endband;
    (*celt_enc).stream_channels = (*st).stream_channels;
    (*celt_enc).bitrate = -1;
    if (*st).mode != MODE_SILK_ONLY {
        let mut celt_pred: opus_val32 = 2 as opus_val32;
        (*celt_enc).vbr = 0;
        if (*st).silk_mode.reducedDependency != 0 {
            celt_pred = 0 as opus_val32;
        }
        let celt_pred_i = celt_pred as i32;
        (*celt_enc).disable_pf = (celt_pred_i <= 1) as i32;
        (*celt_enc).force_intra = (celt_pred_i == 0) as i32;
        if (*st).mode == MODE_HYBRID {
            if (*st).use_vbr != 0 {
                (*celt_enc).bitrate = (*st).bitrate_bps - (*st).silk_mode.bitRate;
                (*celt_enc).constrained_vbr = 0;
            }
        } else if (*st).use_vbr != 0 {
            (*celt_enc).vbr = 1;
            (*celt_enc).constrained_vbr = (*st).vbr_constraint;
            (*celt_enc).bitrate = (*st).bitrate_bps;
        }
    }
    let vla_1 = ((*st).channels * (*st).Fs / 400) as usize;
    let mut tmp_prefill: Vec<opus_val16> = ::std::vec::from_elem(0., vla_1);
    if (*st).mode != MODE_SILK_ONLY && (*st).mode != (*st).prev_mode && (*st).prev_mode > 0 {
        {
            let src_off =
                (((*st).encoder_buffer - total_buffer - (*st).Fs / 400) * (*st).channels) as usize;
            let len = ((*st).channels * (*st).Fs / 400) as usize;
            tmp_prefill[..len].copy_from_slice(&(&(*st).delay_buffer)[src_off..src_off + len]);
        }
    }
    if (*st).channels * ((*st).encoder_buffer - (frame_size + total_buffer)) > 0 {
        {
            let src_off = ((*st).channels * frame_size) as usize;
            let len =
                ((*st).channels * ((*st).encoder_buffer - frame_size - total_buffer)) as usize;
            (&mut (*st).delay_buffer).copy_within(src_off..src_off + len, 0);
        }
        {
            let dst_off =
                ((*st).channels * ((*st).encoder_buffer - frame_size - total_buffer)) as usize;
            let len = ((frame_size + total_buffer) * (*st).channels) as usize;
            (&mut (*st).delay_buffer)[dst_off..dst_off + len].copy_from_slice(&pcm_buf[..len]);
        }
    } else {
        {
            let src_off =
                ((frame_size + total_buffer - (*st).encoder_buffer) * (*st).channels) as usize;
            let len = ((*st).encoder_buffer * (*st).channels) as usize;
            (&mut (*st).delay_buffer)[..len].copy_from_slice(&pcm_buf[src_off..src_off + len]);
        }
    }
    if (*st).prev_HB_gain < Q15ONE || HB_gain < Q15ONE {
        let tmp: Vec<opus_val16> = pcm_buf.clone();
        gain_fade(
            &tmp,
            &mut pcm_buf,
            (*st).prev_HB_gain,
            HB_gain,
            (*celt_mode).overlap as i32,
            frame_size,
            (*st).channels,
            (*celt_mode).window,
            (*st).Fs,
        );
    }
    (*st).prev_HB_gain = HB_gain;
    if (*st).mode != MODE_HYBRID || (*st).stream_channels == 1 {
        if equiv_rate > 32000 {
            (*st).silk_mode.stereoWidth_Q14 = 16384;
        } else if equiv_rate < 16000 {
            (*st).silk_mode.stereoWidth_Q14 = 0;
        } else {
            (*st).silk_mode.stereoWidth_Q14 =
                16384 - 2048 * (32000 - equiv_rate) / (equiv_rate - 14000);
        }
    }
    if (*st).energy_masking_len == 0 && (*st).channels == 2 {
        if ((*st).hybrid_stereo_width_Q14 as i32) < (1) << 14
            || (*st).silk_mode.stereoWidth_Q14 < (1) << 14
        {
            let mut g1: opus_val16 = 0.;
            let mut g2: opus_val16 = 0.;
            g1 = (*st).hybrid_stereo_width_Q14 as opus_val16;
            g2 = (*st).silk_mode.stereoWidth_Q14 as opus_val16;
            g1 *= 1.0f32 / 16384 as f32;
            g2 *= 1.0f32 / 16384 as f32;
            {
                let tmp: Vec<opus_val16> = pcm_buf.clone();
                stereo_fade(
                    &tmp,
                    &mut pcm_buf,
                    g1,
                    g2,
                    (*celt_mode).overlap as i32,
                    frame_size,
                    (*st).channels,
                    (*celt_mode).window,
                    (*st).Fs,
                );
            }
            (*st).hybrid_stereo_width_Q14 = (*st).silk_mode.stereoWidth_Q14 as i16;
        }
    }
    if (*st).mode != MODE_CELT_ONLY
        && ec_tell(&mut enc) + 17 + 20 * ((*st).mode == MODE_HYBRID) as i32
            <= 8 * (max_data_bytes - 1)
    {
        if (*st).mode == MODE_HYBRID {
            ec_enc_bit_logp(&mut enc, redundancy, 12);
        }
        if redundancy != 0 {
            let mut max_redundancy: i32 = 0;
            ec_enc_bit_logp(&mut enc, celt_to_silk, 1);
            if (*st).mode == MODE_HYBRID {
                max_redundancy = max_data_bytes - 1 - (ec_tell(&mut enc) + 8 + 3 + 7 >> 3);
            } else {
                max_redundancy = max_data_bytes - 1 - (ec_tell(&mut enc) + 7 >> 3);
            }
            redundancy_bytes = if max_redundancy < redundancy_bytes {
                max_redundancy
            } else {
                redundancy_bytes
            };
            redundancy_bytes = if (257)
                < (if 2 > redundancy_bytes {
                    2
                } else {
                    redundancy_bytes
                }) {
                257
            } else if 2 > redundancy_bytes {
                2
            } else {
                redundancy_bytes
            };
            if (*st).mode == MODE_HYBRID {
                ec_enc_uint(&mut enc, (redundancy_bytes - 2) as u32, 256);
            }
        }
    } else {
        redundancy = 0;
    }
    if redundancy == 0 {
        (*st).silk_bw_switch = 0;
        redundancy_bytes = 0;
    }
    if (*st).mode != MODE_CELT_ONLY {
        start_band = 17;
    }
    if (*st).mode == MODE_SILK_ONLY {
        ret = ec_tell(&mut enc) + 7 >> 3;
        ec_enc_done(&mut enc);
        nb_compr_bytes = ret;
    } else {
        nb_compr_bytes = max_data_bytes - 1 - redundancy_bytes;
        ec_enc_shrink(&mut enc, nb_compr_bytes as u32);
    }
    if redundancy != 0 || (*st).mode != MODE_SILK_ONLY {
        (*celt_enc).analysis = analysis_info;
    }
    if (*st).mode == MODE_HYBRID {
        let mut info: SILKInfo = SILKInfo {
            signalType: 0,
            offset: 0,
        };
        info.signalType = (*st).silk_mode.signalType;
        info.offset = (*st).silk_mode.offset;
        (*celt_enc).silk_info = info;
    }
    if redundancy != 0 && celt_to_silk != 0 {
        let mut err: i32 = 0;
        (*celt_enc).start = 0;
        (*celt_enc).vbr = 0;
        (*celt_enc).bitrate = -1;
        err = celt_encode_with_ec(
            &mut *celt_enc,
            &pcm_buf,
            (*st).Fs / 200,
            unsafe {
                std::slice::from_raw_parts_mut(
                    data.offset(nb_compr_bytes as isize),
                    redundancy_bytes as usize,
                )
            },
            redundancy_bytes,
            None,
        );
        if err < 0 {
            return OPUS_INTERNAL_ERROR;
        }
        redundant_rng = (*celt_enc).rng;
        (*celt_enc).reset();
    }
    (*celt_enc).start = start_band;
    if (*st).mode != MODE_SILK_ONLY {
        if (*st).mode != (*st).prev_mode && (*st).prev_mode > 0 {
            let mut dummy_0: [u8; 2] = [0; 2];
            (*celt_enc).reset();
            celt_encode_with_ec(
                &mut *celt_enc,
                &tmp_prefill,
                (*st).Fs / 400,
                &mut dummy_0,
                2,
                None,
            );
            (*celt_enc).disable_pf = 1;
            (*celt_enc).force_intra = 1;
        }
        if ec_tell(&mut enc) <= 8 * nb_compr_bytes {
            if redundancy != 0
                && celt_to_silk != 0
                && (*st).mode == MODE_HYBRID
                && (*st).use_vbr != 0
            {
                (*celt_enc).bitrate = (*st).bitrate_bps - (*st).silk_mode.bitRate;
            }
            (*celt_enc).vbr = (*st).use_vbr;
            ret = celt_encode_with_ec(
                &mut *celt_enc,
                &pcm_buf,
                frame_size,
                &mut [],
                nb_compr_bytes,
                Some(&mut enc),
            );
            if ret < 0 {
                return OPUS_INTERNAL_ERROR;
            }
            if redundancy != 0
                && celt_to_silk != 0
                && (*st).mode == MODE_HYBRID
                && (*st).use_vbr != 0
            {
                std::ptr::copy(
                    data.offset(nb_compr_bytes as isize),
                    data.offset(ret as isize),
                    redundancy_bytes as usize,
                );
                nb_compr_bytes = nb_compr_bytes + redundancy_bytes;
            }
        }
    }
    if redundancy != 0 && celt_to_silk == 0 {
        let mut err_0: i32 = 0;
        let mut dummy_1: [u8; 2] = [0; 2];
        let mut N2: i32 = 0;
        let mut N4: i32 = 0;
        N2 = (*st).Fs / 200;
        N4 = (*st).Fs / 400;
        (*celt_enc).reset();
        (*celt_enc).start = 0;
        (*celt_enc).disable_pf = 1;
        (*celt_enc).force_intra = 1;
        (*celt_enc).vbr = 0;
        (*celt_enc).bitrate = -1;
        if (*st).mode == MODE_HYBRID {
            nb_compr_bytes = ret;
            ec_enc_shrink(&mut enc, nb_compr_bytes as u32);
        }
        celt_encode_with_ec(
            &mut *celt_enc,
            &pcm_buf[((*st).channels * (frame_size - N2 - N4)) as usize..],
            N4,
            &mut dummy_1,
            2,
            None,
        );
        err_0 = celt_encode_with_ec(
            &mut *celt_enc,
            &pcm_buf[((*st).channels * (frame_size - N2)) as usize..],
            N2,
            unsafe {
                std::slice::from_raw_parts_mut(
                    data.offset(nb_compr_bytes as isize),
                    redundancy_bytes as usize,
                )
            },
            redundancy_bytes,
            None,
        );
        if err_0 < 0 {
            return OPUS_INTERNAL_ERROR;
        }
        redundant_rng = (*celt_enc).rng;
    }
    data = data.offset(-1);
    *data.offset(0 as isize) = gen_toc(
        (*st).mode,
        (*st).Fs / frame_size,
        curr_bandwidth,
        (*st).stream_channels,
    );
    (*st).rangeFinal = enc.rng ^ redundant_rng;
    if to_celt != 0 {
        (*st).prev_mode = MODE_CELT_ONLY;
    } else {
        (*st).prev_mode = (*st).mode;
    }
    (*st).prev_channels = (*st).stream_channels;
    (*st).prev_framesize = frame_size;
    (*st).first = 0;
    if (*st).use_dtx != 0 && (analysis_info.valid != 0 || is_silence != 0) {
        if decide_dtx_mode(
            analysis_info.activity_probability,
            &mut (*st).nb_no_activity_frames,
            (*st).peak_signal_energy,
            std::slice::from_raw_parts(pcm, (frame_size * (*st).channels) as usize),
            frame_size,
            (*st).channels,
            is_silence,
            (*st).arch,
        ) != 0
        {
            (*st).rangeFinal = 0;
            *data.offset(0 as isize) = gen_toc(
                (*st).mode,
                (*st).Fs / frame_size,
                curr_bandwidth,
                (*st).stream_channels,
            );
            return 1;
        }
    } else {
        (*st).nb_no_activity_frames = 0;
    }
    if ec_tell(&mut enc) > (max_data_bytes - 1) * 8 {
        if max_data_bytes < 2 {
            return OPUS_BUFFER_TOO_SMALL;
        }
        *data.offset(1 as isize) = 0;
        ret = 1;
        (*st).rangeFinal = 0;
    } else if (*st).mode == MODE_SILK_ONLY && redundancy == 0 {
        while ret > 2 && *data.offset(ret as isize) as i32 == 0 {
            ret -= 1;
        }
    }
    ret += 1 + redundancy_bytes;
    if (*st).use_vbr == 0 {
        let data = std::slice::from_raw_parts_mut(data, max_data_bytes as _);
        if opus_packet_pad(data, ret, max_data_bytes) != OPUS_OK {
            return OPUS_INTERNAL_ERROR;
        }
        ret = max_data_bytes;
    }
    return ret;
}
/// Upstream C: src/opus_encoder.c:opus_encode
fn opus_encode(
    st: &mut OpusEncoder,
    pcm: &[i16],
    analysis_frame_size: i32,
    data: &mut [u8],
) -> i32 {
    let frame_size = frame_size_select(analysis_frame_size, st.variable_duration, st.Fs);
    if frame_size <= 0 {
        return OPUS_BAD_ARG;
    }
    let vla = (frame_size * st.channels) as usize;
    let mut in_0: Vec<f32> = vec![0.0; vla];
    for i in 0..(frame_size * st.channels) as usize {
        in_0[i] = 1.0f32 / 32768.0f32 * pcm[i] as i32 as f32;
    }
    // SAFETY: opus_encode_native still uses raw pointers internally but operates
    // only within the bounds of the provided slices.
    unsafe {
        opus_encode_native(
            st as *mut OpusEncoder,
            in_0.as_ptr(),
            frame_size,
            data.as_mut_ptr(),
            data.len() as i32,
            16,
            pcm.as_ptr() as *const core::ffi::c_void,
            analysis_frame_size,
            0,
            -2,
            st.channels,
            0,
        )
    }
}
/// Upstream C: src/opus_encoder.c:opus_encode_float
fn opus_encode_float(
    st: &mut OpusEncoder,
    pcm: &[f32],
    analysis_frame_size: i32,
    data: &mut [u8],
) -> i32 {
    let frame_size = frame_size_select(analysis_frame_size, st.variable_duration, st.Fs);
    // SAFETY: opus_encode_native still uses raw pointers internally but operates
    // only within the bounds of the provided slices.
    unsafe {
        opus_encode_native(
            st as *mut OpusEncoder,
            pcm.as_ptr(),
            frame_size,
            data.as_mut_ptr(),
            data.len() as i32,
            24,
            pcm.as_ptr() as *const core::ffi::c_void,
            analysis_frame_size,
            0,
            -2,
            st.channels,
            1,
        )
    }
}
