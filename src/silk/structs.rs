//! SILK data structures.
//!
//! Upstream C: `silk/structs.h`

use crate::silk::define::{LTP_ORDER, MAX_FRAME_LENGTH, MAX_LPC_ORDER, MAX_NB_SUBFR};
use crate::silk::resampler::ResamplerState;
use crate::silk::tables_NLSF_CB_NB_MB::silk_NLSF_CB_NB_MB;

#[derive(Copy, Clone)]
#[repr(C)]
pub struct silk_NLSF_CB_struct {
    pub nVectors: i16,
    pub order: i16,
    pub quantStepSize_Q16: i16,
    pub invQuantStepSize_Q6: i16,
    pub CB1_NLSF_Q8: &'static [u8],
    pub CB1_Wght_Q9: &'static [i16],
    pub CB1_iCDF: &'static [u8; 64],
    pub pred_Q8: &'static [u8],
    pub ec_sel: &'static [u8],
    pub ec_iCDF: &'static [u8; 72],
    pub ec_Rates_Q5: &'static [u8; 72],
    pub deltaMin_Q15: &'static [i16],
}
#[derive(Copy, Clone, Default)]
#[repr(C)]
pub struct SideInfoIndices {
    pub GainsIndices: [i8; 4],
    pub LTPIndex: [i8; 4],
    pub NLSFIndices: [i8; 17],
    pub lagIndex: i16,
    pub contourIndex: i8,
    pub signalType: i8,
    pub quantOffsetType: i8,
    pub NLSFInterpCoef_Q2: i8,
    pub PERIndex: i8,
    pub LTP_scaleIndex: i8,
    pub Seed: i8,
}

/// Struct for Packet Loss Concealment
#[derive(Copy, Clone, Default)]
#[repr(C)]
pub struct silk_PLC_struct {
    /// Pitch lag to use for voiced concealment
    pub pitchL_Q8: i32,
    /// LTP coeficients to use for voiced concealment
    pub LTPCoef_Q14: [i16; LTP_ORDER],
    pub prevLPC_Q12: [i16; MAX_LPC_ORDER],
    /// Was previous frame lost
    pub last_frame_lost: i32,
    /// Seed for unvoiced signal generation
    pub rand_seed: i32,
    /// Scaling of unvoiced random signal
    pub randScale_Q14: i16,
    pub conc_energy: i32,
    pub conc_energy_shift: i32,
    pub prevLTP_scale_Q14: i16,
    pub prevGain_Q16: [i32; 2],
    pub fs_kHz: i32,
    pub nb_subfr: i32,
    pub subfr_length: i32,
    /// Whether Deep PLC is enabled (complexity >= 5)
    pub enable_deep_plc: bool,
}

#[derive(Copy, Clone)]
#[repr(C)]
pub struct silk_CNG_struct {
    pub CNG_exc_buf_Q14: [i32; MAX_FRAME_LENGTH],
    pub CNG_smth_NLSF_Q15: [i16; MAX_LPC_ORDER],
    pub CNG_synth_state: [i32; MAX_LPC_ORDER],
    pub CNG_smth_Gain_Q16: i32,
    pub rand_seed: i32,
    pub fs_kHz: i32,
}

impl Default for silk_CNG_struct {
    fn default() -> Self {
        Self {
            CNG_exc_buf_Q14: [0; 320],
            CNG_smth_NLSF_Q15: [0; 16],
            CNG_synth_state: [0; 16],
            CNG_smth_Gain_Q16: 0,
            rand_seed: 0,
            fs_kHz: 0,
        }
    }
}

#[derive(Clone)]
#[repr(C)]
pub struct silk_decoder_state {
    pub prev_gain_Q16: i32,
    pub exc_Q14: [i32; MAX_FRAME_LENGTH],
    pub sLPC_Q14_buf: [i32; MAX_LPC_ORDER],
    pub outBuf: [i16; 480],
    pub lagPrev: i32,
    pub LastGainIndex: i8,
    pub fs_kHz: i32,
    pub fs_API_hz: i32,
    pub nb_subfr: usize,
    pub frame_length: usize,
    pub subfr_length: usize,
    pub ltp_mem_length: usize,
    pub LPC_order: usize,
    pub prevNLSF_Q15: [i16; MAX_LPC_ORDER],
    pub first_frame_after_reset: i32,
    pub pitch_lag_low_bits_iCDF: &'static [u8],
    pub pitch_contour_iCDF: &'static [u8],
    pub nFramesDecoded: i32,
    pub nFramesPerPacket: i32,
    pub ec_prevSignalType: i32,
    pub ec_prevLagIndex: i16,
    pub VAD_flags: [i32; 3],
    pub LBRR_flag: i32,
    pub LBRR_flags: [i32; 3],
    pub resampler_state: ResamplerState,
    pub psNLSF_CB: &'static silk_NLSF_CB_struct,
    pub indices: SideInfoIndices,
    pub sCNG: silk_CNG_struct,
    pub lossCnt: i32,
    pub prevSignalType: i32,
    pub arch: i32,
    pub sPLC: silk_PLC_struct,
    #[cfg(feature = "osce")]
    pub osce: crate::dnn::osce::OSCEState,
}
#[derive(Copy, Clone)]
#[repr(C)]
pub struct silk_decoder_control {
    pub pitchL: [i32; MAX_NB_SUBFR],
    pub Gains_Q16: [i32; MAX_NB_SUBFR],
    pub PredCoef_Q12: [[i16; MAX_LPC_ORDER]; 2],
    pub LTPCoef_Q14: [i16; LTP_ORDER * MAX_NB_SUBFR],
    pub LTP_scale_Q14: i32,
}

/// Read-only config fields needed by the NSQ quantization pipeline.
/// Extracted from `silk_encoder_state` to avoid borrow conflicts when
/// the caller also needs mutable access to `indices`, `sNSQ`, and `pulses`.
#[derive(Copy, Clone)]
pub struct NsqConfig {
    pub nb_subfr: usize,
    pub frame_length: usize,
    pub subfr_length: usize,
    pub ltp_mem_length: usize,
    pub predictLPCOrder: i32,
    pub shapingLPCOrder: i32,
    pub nStatesDelayedDecision: i32,
    pub warping_Q16: i32,
    pub arch: i32,
}

#[derive(Copy, Clone)]
#[repr(C)]
pub struct silk_nsq_state {
    pub xq: [i16; 640],
    pub sLTP_shp_Q14: [i32; 640],
    pub sLPC_Q14: [i32; 96],
    pub sAR2_Q14: [i32; 24],
    pub sLF_AR_shp_Q14: i32,
    pub sDiff_shp_Q14: i32,
    pub lagPrev: i32,
    pub sLTP_buf_idx: i32,
    pub sLTP_shp_buf_idx: i32,
    pub rand_seed: i32,
    pub prev_gain_Q16: i32,
    pub rewhite_flag: i32,
}

impl Default for silk_nsq_state {
    fn default() -> Self {
        Self {
            xq: [0; 640],
            sLTP_shp_Q14: [0; 640],
            sLPC_Q14: [0; 96],
            sAR2_Q14: [0; 24],
            sLF_AR_shp_Q14: 0,
            sDiff_shp_Q14: 0,
            lagPrev: 0,
            sLTP_buf_idx: 0,
            sLTP_shp_buf_idx: 0,
            rand_seed: 0,
            prev_gain_Q16: 0,
            rewhite_flag: 0,
        }
    }
}
#[derive(Copy, Clone, Default)]
#[repr(C)]
pub struct silk_VAD_state {
    pub AnaState: [i32; 2],
    pub AnaState1: [i32; 2],
    pub AnaState2: [i32; 2],
    pub XnrgSubfr: [i32; 4],
    pub NrgRatioSmth_Q8: [i32; 4],
    pub HPstate: i16,
    pub NL: [i32; 4],
    pub inv_NL: [i32; 4],
    pub NoiseLevelBias: [i32; 4],
    pub counter: i32,
}
#[derive(Copy, Clone, Default)]
#[repr(C)]
pub struct silk_LP_state {
    pub In_LP_State: [i32; 2],
    pub transition_frame_no: i32,
    pub mode: i32,
    pub saved_fs_kHz: i32,
}
#[derive(Copy, Clone)]
#[repr(C)]
pub struct silk_encoder_state {
    pub In_HP_State: [i32; 2],
    pub variable_HP_smth1_Q15: i32,
    pub variable_HP_smth2_Q15: i32,
    pub sLP: silk_LP_state,
    pub sVAD: silk_VAD_state,
    pub sNSQ: silk_nsq_state,
    pub prev_NLSFq_Q15: [i16; 16],
    pub speech_activity_Q8: i32,
    pub allow_bandwidth_switch: i32,
    pub LBRRprevLastGainIndex: i8,
    pub prevSignalType: i8,
    pub prevLag: i32,
    pub pitch_LPC_win_length: i32,
    pub max_pitch_lag: i32,
    pub API_fs_Hz: i32,
    pub prev_API_fs_Hz: i32,
    pub maxInternal_fs_Hz: i32,
    pub minInternal_fs_Hz: i32,
    pub desiredInternal_fs_Hz: i32,
    pub fs_kHz: i32,
    pub nb_subfr: usize,
    pub frame_length: usize,
    pub subfr_length: usize,
    pub ltp_mem_length: usize,
    pub la_pitch: i32,
    pub la_shape: i32,
    pub shapeWinLength: i32,
    pub TargetRate_bps: i32,
    pub PacketSize_ms: i32,
    pub PacketLoss_perc: i32,
    pub frameCounter: i32,
    pub Complexity: i32,
    pub nStatesDelayedDecision: i32,
    pub useInterpolatedNLSFs: i32,
    pub shapingLPCOrder: i32,
    pub predictLPCOrder: i32,
    pub pitchEstimationComplexity: i32,
    pub pitchEstimationLPCOrder: i32,
    pub pitchEstimationThreshold_Q16: i32,
    pub sum_log_gain_Q7: i32,
    pub NLSF_MSVQ_Survivors: i32,
    pub first_frame_after_reset: i32,
    pub controlled_since_last_payload: i32,
    pub warping_Q16: i32,
    pub useCBR: i32,
    pub prefillFlag: i32,
    pub pitch_lag_low_bits_iCDF: &'static [u8],
    pub pitch_contour_iCDF: &'static [u8],
    pub psNLSF_CB: &'static silk_NLSF_CB_struct,
    pub input_quality_bands_Q15: [i32; 4],
    pub input_tilt_Q15: i32,
    pub SNR_dB_Q7: i32,
    pub VAD_flags: [i8; 3],
    pub LBRR_flag: i8,
    pub LBRR_flags: [i32; 3],
    pub indices: SideInfoIndices,
    pub pulses: [i8; 320],
    pub arch: i32,
    pub inputBuf: [i16; 322],
    pub inputBufIx: i32,
    pub nFramesPerPacket: i32,
    pub nFramesEncoded: i32,
    pub nChannelsAPI: i32,
    pub nChannelsInternal: i32,
    pub channelNb: i32,
    pub frames_since_onset: i32,
    pub ec_prevSignalType: i32,
    pub ec_prevLagIndex: i16,
    pub resampler_state: ResamplerState,
    pub useDTX: i32,
    pub inDTX: i32,
    pub noSpeechCounter: i32,
    pub useInBandFEC: i32,
    pub LBRR_enabled: i32,
    pub LBRR_GainIncreases: i32,
    pub indices_LBRR: [SideInfoIndices; 3],
    pub pulses_LBRR: [[i8; 320]; 3],
}

impl silk_encoder_state {
    /// Extract the read-only config fields needed by the NSQ pipeline.
    pub fn nsq_config(&self) -> NsqConfig {
        NsqConfig {
            nb_subfr: self.nb_subfr,
            frame_length: self.frame_length,
            subfr_length: self.subfr_length,
            ltp_mem_length: self.ltp_mem_length,
            predictLPCOrder: self.predictLPCOrder,
            shapingLPCOrder: self.shapingLPCOrder,
            nStatesDelayedDecision: self.nStatesDelayedDecision,
            warping_Q16: self.warping_Q16,
            arch: self.arch,
        }
    }
}

impl Default for silk_encoder_state {
    fn default() -> Self {
        Self {
            In_HP_State: [0; 2],
            variable_HP_smth1_Q15: 0,
            variable_HP_smth2_Q15: 0,
            sLP: Default::default(),
            sVAD: Default::default(),
            sNSQ: Default::default(),
            prev_NLSFq_Q15: [0; 16],
            speech_activity_Q8: 0,
            allow_bandwidth_switch: 0,
            LBRRprevLastGainIndex: 0,
            prevSignalType: 0,
            prevLag: 0,
            pitch_LPC_win_length: 0,
            max_pitch_lag: 0,
            API_fs_Hz: 0,
            prev_API_fs_Hz: 0,
            maxInternal_fs_Hz: 0,
            minInternal_fs_Hz: 0,
            desiredInternal_fs_Hz: 0,
            fs_kHz: 0,
            nb_subfr: 0,
            frame_length: 0,
            subfr_length: 0,
            ltp_mem_length: 0,
            la_pitch: 0,
            la_shape: 0,
            shapeWinLength: 0,
            TargetRate_bps: 0,
            PacketSize_ms: 0,
            PacketLoss_perc: 0,
            frameCounter: 0,
            Complexity: 0,
            nStatesDelayedDecision: 0,
            useInterpolatedNLSFs: 0,
            shapingLPCOrder: 0,
            predictLPCOrder: 0,
            pitchEstimationComplexity: 0,
            pitchEstimationLPCOrder: 0,
            pitchEstimationThreshold_Q16: 0,
            sum_log_gain_Q7: 0,
            NLSF_MSVQ_Survivors: 0,
            first_frame_after_reset: 0,
            controlled_since_last_payload: 0,
            warping_Q16: 0,
            useCBR: 0,
            prefillFlag: 0,
            pitch_lag_low_bits_iCDF: &[],
            pitch_contour_iCDF: &[],
            psNLSF_CB: &silk_NLSF_CB_NB_MB,
            input_quality_bands_Q15: [0; 4],
            input_tilt_Q15: 0,
            SNR_dB_Q7: 0,
            VAD_flags: [0; 3],
            LBRR_flag: 0,
            LBRR_flags: [0; 3],
            indices: Default::default(),
            pulses: [0; 320],
            arch: 0,
            inputBuf: [0; 322],
            inputBufIx: 0,
            nFramesPerPacket: 0,
            nFramesEncoded: 0,
            nChannelsAPI: 0,
            nChannelsInternal: 0,
            channelNb: 0,
            frames_since_onset: 0,
            ec_prevSignalType: 0,
            ec_prevLagIndex: 0,
            resampler_state: Default::default(),
            useDTX: 0,
            inDTX: 0,
            noSpeechCounter: 0,
            useInBandFEC: 0,
            LBRR_enabled: 0,
            LBRR_GainIncreases: 0,
            indices_LBRR: [Default::default(); 3],
            pulses_LBRR: [[0; 320]; 3],
        }
    }
}

#[derive(Copy, Clone, Default)]
#[repr(C)]
pub struct stereo_enc_state {
    pub pred_prev_Q13: [i16; 2],
    pub sMid: [i16; 2],
    pub sSide: [i16; 2],
    pub mid_side_amp_Q0: [i32; 4],
    pub smth_width_Q14: i16,
    pub width_prev_Q14: i16,
    pub silent_side_len: i16,
    pub predIx: [[[i8; 3]; 2]; 3],
    pub mid_only_flags: [i8; 3],
}

#[derive(Copy, Clone, Default)]
#[repr(C)]
pub struct stereo_dec_state {
    pub pred_prev_Q13: [i16; 2],
    pub sMid: [i16; 2],
    pub sSide: [i16; 2],
}
