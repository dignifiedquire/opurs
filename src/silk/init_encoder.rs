use crate::silk::define::ENCODER_NUM_CHANNELS;
use crate::silk::float::structs_FLP::{silk_encoder, silk_encoder_state_FLP, silk_shape_state_FLP};
use crate::silk::lin2log::silk_lin2log;
use crate::silk::resampler::ResamplerState;
use crate::silk::structs::{
    silk_LP_state, silk_VAD_state, silk_encoder_state, silk_nsq_state, stereo_enc_state,
    SideInfoIndices,
};
use crate::silk::tables_NLSF_CB_NB_MB::silk_NLSF_CB_NB_MB;
use crate::silk::tuning_parameters::VARIABLE_HP_MIN_CUTOFF_HZ;

impl silk_encoder_state {
    pub fn new(arch: i32) -> Self {
        Self {
            In_HP_State: [0; 2],
            variable_HP_smth1_Q15: 0,
            variable_HP_smth2_Q15: 0,
            sLP: silk_LP_state::default(),
            sVAD: silk_VAD_state::default(),
            sNSQ: silk_nsq_state::default(),
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
            pitch_lag_low_bits_iCDF: &[][..],
            pitch_contour_iCDF: &[][..],
            psNLSF_CB: &silk_NLSF_CB_NB_MB, // TODO: verify what a good default is
            input_quality_bands_Q15: [0; 4],
            input_tilt_Q15: 0,
            SNR_dB_Q7: 0,
            VAD_flags: [0; 3],
            LBRR_flag: 0,
            LBRR_flags: [0; 3],
            indices: SideInfoIndices::default(),
            pulses: [0; 320],
            arch,
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
            resampler_state: ResamplerState::default(),
            useDTX: 0,
            inDTX: 0,
            noSpeechCounter: 0,
            useInBandFEC: 0,
            LBRR_enabled: 0,
            LBRR_GainIncreases: 0,
            indices_LBRR: [SideInfoIndices::default(); 3],
            pulses_LBRR: [[0; 320]; 3],
        }
    }
}

impl silk_encoder_state_FLP {
    pub fn new(arch: i32) -> Self {
        let mut sCmn = silk_encoder_state::new(arch);
        sCmn.variable_HP_smth1_Q15 =
            (((silk_lin2log(((VARIABLE_HP_MIN_CUTOFF_HZ * ((1) << 16)) as f64 + 0.5f64) as i32)
                - ((16) << 7)) as u32)
                << 8) as i32;
        sCmn.variable_HP_smth2_Q15 = sCmn.variable_HP_smth1_Q15;
        sCmn.first_frame_after_reset = 1;

        sCmn.arch = arch;
        sCmn.variable_HP_smth1_Q15 =
            (((silk_lin2log(((VARIABLE_HP_MIN_CUTOFF_HZ * ((1) << 16)) as f64 + 0.5f64) as i32)
                - ((16) << 7)) as u32)
                << 8) as i32;
        sCmn.variable_HP_smth2_Q15 = sCmn.variable_HP_smth1_Q15;
        sCmn.first_frame_after_reset = 1;

        silk_encoder_state_FLP {
            sCmn,
            sShape: silk_shape_state_FLP::default(),
            x_buf: [0.; 720],
            LTPCorr: 0.,
        }
    }
}

impl silk_encoder {
    pub fn new(arch: i32) -> Self {
        Self {
            state_Fxx: [silk_encoder_state_FLP::new(arch); ENCODER_NUM_CHANNELS as usize],
            sStereo: stereo_enc_state::default(),
            nBitsUsedLBRR: 0,
            nBitsExceeded: 0,
            nChannelsAPI: 0,
            nChannelsInternal: 0,
            nPrevChannelsInternal: 0,
            timeSinceSwitchAllowed_ms: 0,
            allowBandwidthSwitch: 0,
            prev_decode_only_middle: 0,
        }
    }
}
