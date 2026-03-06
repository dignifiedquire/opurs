//! SILK data structures.
//!
//! Upstream c: `silk/structs.h`

use crate::arch::Arch;
use crate::silk::define::{LTP_ORDER, MAX_FRAME_LENGTH, MAX_LPC_ORDER, MAX_NB_SUBFR};
use crate::silk::resampler::ResamplerState;
use crate::silk::tables_nlsf_cb_nb_mb::SILK_NLSF_CB_NB_MB;

#[derive(Copy, Clone)]
#[repr(C)]
pub struct silk_NLSF_CB_struct {
    pub n_vectors: i16,
    pub order: i16,
    pub quant_step_size_q16: i16,
    pub inv_quant_step_size_q6: i16,
    pub cb1_nlsf_q8: &'static [u8],
    pub cb1_wght_q9: &'static [i16],
    pub cb1_i_cdf: &'static [u8; 64],
    pub pred_q8: &'static [u8],
    pub ec_sel: &'static [u8],
    pub ec_i_cdf: &'static [u8; 72],
    pub ec_rates_q5: &'static [u8; 72],
    pub delta_min_q15: &'static [i16],
}
#[derive(Copy, Clone, Default)]
#[repr(C)]
pub struct SideInfoIndices {
    pub gains_indices: [i8; 4],
    pub ltpindex: [i8; 4],
    pub nlsfindices: [i8; 17],
    pub lag_index: i16,
    pub contour_index: i8,
    pub signal_type: i8,
    pub quant_offset_type: i8,
    pub nlsfinterp_coef_q2: i8,
    pub perindex: i8,
    pub ltp_scale_index: i8,
    pub seed: i8,
}

/// Struct for Packet Loss Concealment
#[derive(Copy, Clone, Default)]
#[repr(C)]
pub struct silk_PLC_struct {
    /// Pitch lag to use for voiced concealment
    pub pitch_l_q8: i32,
    /// LTP coeficients to use for voiced concealment
    pub ltpcoef_q14: [i16; LTP_ORDER],
    pub prev_lpc_q12: [i16; MAX_LPC_ORDER],
    /// Was previous frame lost
    pub last_frame_lost: i32,
    /// seed for unvoiced signal generation
    pub rand_seed: i32,
    /// Scaling of unvoiced random signal
    pub rand_scale_q14: i16,
    pub conc_energy: i32,
    pub conc_energy_shift: i32,
    pub prev_ltp_scale_q14: i16,
    pub prev_gain_q16: [i32; 2],
    pub fs_k_hz: i32,
    pub nb_subfr: i32,
    pub subfr_length: i32,
    /// Whether Deep PLC is enabled (complexity >= 5)
    pub enable_deep_plc: bool,
}

#[derive(Copy, Clone)]
#[repr(C)]
pub struct silk_CNG_struct {
    pub cng_exc_buf_q14: [i32; MAX_FRAME_LENGTH],
    pub cng_smth_nlsf_q15: [i16; MAX_LPC_ORDER],
    pub cng_synth_state: [i32; MAX_LPC_ORDER],
    pub cng_smth_gain_q16: i32,
    pub rand_seed: i32,
    pub fs_k_hz: i32,
}

impl Default for silk_CNG_struct {
    fn default() -> Self {
        Self {
            cng_exc_buf_q14: [0; 320],
            cng_smth_nlsf_q15: [0; 16],
            cng_synth_state: [0; 16],
            cng_smth_gain_q16: 0,
            rand_seed: 0,
            fs_k_hz: 0,
        }
    }
}

#[derive(Clone)]
#[repr(C)]
pub struct silk_decoder_state {
    pub prev_gain_q16: i32,
    pub exc_q14: [i32; MAX_FRAME_LENGTH],
    pub s_lpc_q14_buf: [i32; MAX_LPC_ORDER],
    pub out_buf: [i16; 480],
    pub lag_prev: i32,
    pub last_gain_index: i8,
    pub fs_k_hz: i32,
    pub fs_api_hz: i32,
    pub nb_subfr: usize,
    pub frame_length: usize,
    pub subfr_length: usize,
    pub ltp_mem_length: usize,
    pub lpc_order: usize,
    pub prev_nlsf_q15: [i16; MAX_LPC_ORDER],
    pub first_frame_after_reset: i32,
    pub pitch_lag_low_bits_i_cdf: &'static [u8],
    pub pitch_contour_i_cdf: &'static [u8],
    pub n_frames_decoded: i32,
    pub n_frames_per_packet: i32,
    pub ec_prev_signal_type: i32,
    pub ec_prev_lag_index: i16,
    pub vad_flags: [i32; 3],
    pub lbrr_flag: i32,
    pub lbrr_flags: [i32; 3],
    pub resampler_state: ResamplerState,
    pub ps_nlsf_cb: &'static silk_NLSF_CB_struct,
    pub indices: SideInfoIndices,
    pub s_cng: silk_CNG_struct,
    pub loss_cnt: i32,
    pub prev_signal_type: i32,
    pub arch: Arch,
    pub s_plc: silk_PLC_struct,
    #[cfg(feature = "osce")]
    pub osce: crate::dnn::osce::OSCEState,
    #[cfg(feature = "osce")]
    pub osce_bwe: crate::dnn::osce::OSCEBWE,
}
#[derive(Copy, Clone)]
#[repr(C)]
pub struct silk_decoder_control {
    pub pitch_l: [i32; MAX_NB_SUBFR],
    pub gains_q16: [i32; MAX_NB_SUBFR],
    pub pred_coef_q12: [[i16; MAX_LPC_ORDER]; 2],
    pub ltpcoef_q14: [i16; LTP_ORDER * MAX_NB_SUBFR],
    pub ltp_scale_q14: i32,
}

/// Read-only config fields needed by the NSQ quantization pipeline.
/// Extracted from `silk_encoder_state` to avoid borrow conflicts when
/// the caller also needs mutable access to `indices`, `s_nsq`, and `pulses`.
#[derive(Copy, Clone)]
pub struct NsqConfig {
    pub nb_subfr: usize,
    pub frame_length: usize,
    pub subfr_length: usize,
    pub ltp_mem_length: usize,
    pub predict_lpcorder: i32,
    pub shaping_lpcorder: i32,
    pub n_states_delayed_decision: i32,
    pub warping_q16: i32,
    pub arch: Arch,
}

#[derive(Copy, Clone)]
#[repr(C)]
pub struct silk_nsq_state {
    pub xq: [i16; 640],
    pub s_ltp_shp_q14: [i32; 640],
    pub s_lpc_q14: [i32; 96],
    pub s_ar2_q14: [i32; 24],
    pub s_lf_ar_shp_q14: i32,
    pub s_diff_shp_q14: i32,
    pub lag_prev: i32,
    pub s_ltp_buf_idx: i32,
    pub s_ltp_shp_buf_idx: i32,
    pub rand_seed: i32,
    pub prev_gain_q16: i32,
    pub rewhite_flag: i32,
}

impl Default for silk_nsq_state {
    fn default() -> Self {
        Self {
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
        }
    }
}
#[derive(Copy, Clone, Default)]
#[repr(C)]
pub struct silk_VAD_state {
    pub ana_state: [i32; 2],
    pub ana_state1: [i32; 2],
    pub ana_state2: [i32; 2],
    pub xnrg_subfr: [i32; 4],
    pub nrg_ratio_smth_q8: [i32; 4],
    pub hpstate: i16,
    pub nl: [i32; 4],
    pub inv_nl: [i32; 4],
    pub noise_level_bias: [i32; 4],
    pub counter: i32,
}
#[derive(Copy, Clone, Default)]
#[repr(C)]
pub struct silk_LP_state {
    pub in_lp_state: [i32; 2],
    pub transition_frame_no: i32,
    pub mode: i32,
    pub saved_fs_k_hz: i32,
}
#[derive(Copy, Clone)]
#[repr(C)]
pub struct silk_encoder_state {
    pub in_hp_state: [i32; 2],
    pub variable_hp_smth1_q15: i32,
    pub variable_hp_smth2_q15: i32,
    pub s_lp: silk_LP_state,
    pub s_vad: silk_VAD_state,
    pub s_nsq: silk_nsq_state,
    pub prev_nlsfq_q15: [i16; 16],
    pub speech_activity_q8: i32,
    pub allow_bandwidth_switch: i32,
    pub lbrrprev_last_gain_index: i8,
    pub prev_signal_type: i8,
    pub prev_lag: i32,
    pub pitch_lpc_win_length: i32,
    pub max_pitch_lag: i32,
    pub api_fs_hz: i32,
    pub prev_api_fs_hz: i32,
    pub max_internal_fs_hz: i32,
    pub min_internal_fs_hz: i32,
    pub desired_internal_fs_hz: i32,
    pub fs_k_hz: i32,
    pub nb_subfr: usize,
    pub frame_length: usize,
    pub subfr_length: usize,
    pub ltp_mem_length: usize,
    pub la_pitch: i32,
    pub la_shape: i32,
    pub shape_win_length: i32,
    pub target_rate_bps: i32,
    pub packet_size_ms: i32,
    pub packet_loss_perc: i32,
    pub frame_counter: i32,
    pub complexity: i32,
    pub n_states_delayed_decision: i32,
    pub use_interpolated_nlsfs: i32,
    pub shaping_lpcorder: i32,
    pub predict_lpcorder: i32,
    pub pitch_estimation_complexity: i32,
    pub pitch_estimation_lpcorder: i32,
    pub pitch_estimation_threshold_q16: i32,
    pub sum_log_gain_q7: i32,
    pub nlsf_msvq_survivors: i32,
    pub first_frame_after_reset: i32,
    pub controlled_since_last_payload: i32,
    pub warping_q16: i32,
    pub use_cbr: i32,
    pub prefill_flag: i32,
    pub pitch_lag_low_bits_i_cdf: &'static [u8],
    pub pitch_contour_i_cdf: &'static [u8],
    pub ps_nlsf_cb: &'static silk_NLSF_CB_struct,
    pub input_quality_bands_q15: [i32; 4],
    pub input_tilt_q15: i32,
    pub snr_d_b_q7: i32,
    pub vad_flags: [i8; 3],
    pub lbrr_flag: i8,
    pub lbrr_flags: [i32; 3],
    pub indices: SideInfoIndices,
    pub pulses: [i8; 320],
    pub arch: Arch,
    pub input_buf: [i16; 322],
    pub input_buf_ix: i32,
    pub n_frames_per_packet: i32,
    pub n_frames_encoded: i32,
    pub n_channels_api: i32,
    pub n_channels_internal: i32,
    pub channel_nb: i32,
    pub frames_since_onset: i32,
    pub ec_prev_signal_type: i32,
    pub ec_prev_lag_index: i16,
    pub resampler_state: ResamplerState,
    pub use_dtx: i32,
    pub in_dtx: i32,
    pub no_speech_counter: i32,
    pub use_in_band_fec: i32,
    pub lbrr_enabled: i32,
    pub lbrr_gain_increases: i32,
    pub indices_lbrr: [SideInfoIndices; 3],
    pub pulses_lbrr: [[i8; 320]; 3],
}

impl silk_encoder_state {
    /// Extract the read-only config fields needed by the NSQ pipeline.
    pub fn nsq_config(&self) -> NsqConfig {
        NsqConfig {
            nb_subfr: self.nb_subfr,
            frame_length: self.frame_length,
            subfr_length: self.subfr_length,
            ltp_mem_length: self.ltp_mem_length,
            predict_lpcorder: self.predict_lpcorder,
            shaping_lpcorder: self.shaping_lpcorder,
            n_states_delayed_decision: self.n_states_delayed_decision,
            warping_q16: self.warping_q16,
            arch: self.arch,
        }
    }
}

impl Default for silk_encoder_state {
    fn default() -> Self {
        Self {
            in_hp_state: [0; 2],
            variable_hp_smth1_q15: 0,
            variable_hp_smth2_q15: 0,
            s_lp: Default::default(),
            s_vad: Default::default(),
            s_nsq: Default::default(),
            prev_nlsfq_q15: [0; 16],
            speech_activity_q8: 0,
            allow_bandwidth_switch: 0,
            lbrrprev_last_gain_index: 0,
            prev_signal_type: 0,
            prev_lag: 0,
            pitch_lpc_win_length: 0,
            max_pitch_lag: 0,
            api_fs_hz: 0,
            prev_api_fs_hz: 0,
            max_internal_fs_hz: 0,
            min_internal_fs_hz: 0,
            desired_internal_fs_hz: 0,
            fs_k_hz: 0,
            nb_subfr: 0,
            frame_length: 0,
            subfr_length: 0,
            ltp_mem_length: 0,
            la_pitch: 0,
            la_shape: 0,
            shape_win_length: 0,
            target_rate_bps: 0,
            packet_size_ms: 0,
            packet_loss_perc: 0,
            frame_counter: 0,
            complexity: 0,
            n_states_delayed_decision: 0,
            use_interpolated_nlsfs: 0,
            shaping_lpcorder: 0,
            predict_lpcorder: 0,
            pitch_estimation_complexity: 0,
            pitch_estimation_lpcorder: 0,
            pitch_estimation_threshold_q16: 0,
            sum_log_gain_q7: 0,
            nlsf_msvq_survivors: 0,
            first_frame_after_reset: 0,
            controlled_since_last_payload: 0,
            warping_q16: 0,
            use_cbr: 0,
            prefill_flag: 0,
            pitch_lag_low_bits_i_cdf: &[],
            pitch_contour_i_cdf: &[],
            ps_nlsf_cb: &SILK_NLSF_CB_NB_MB,
            input_quality_bands_q15: [0; 4],
            input_tilt_q15: 0,
            snr_d_b_q7: 0,
            vad_flags: [0; 3],
            lbrr_flag: 0,
            lbrr_flags: [0; 3],
            indices: Default::default(),
            pulses: [0; 320],
            arch: Arch::default(),
            input_buf: [0; 322],
            input_buf_ix: 0,
            n_frames_per_packet: 0,
            n_frames_encoded: 0,
            n_channels_api: 0,
            n_channels_internal: 0,
            channel_nb: 0,
            frames_since_onset: 0,
            ec_prev_signal_type: 0,
            ec_prev_lag_index: 0,
            resampler_state: Default::default(),
            use_dtx: 0,
            in_dtx: 0,
            no_speech_counter: 0,
            use_in_band_fec: 0,
            lbrr_enabled: 0,
            lbrr_gain_increases: 0,
            indices_lbrr: [Default::default(); 3],
            pulses_lbrr: [[0; 320]; 3],
        }
    }
}

#[derive(Copy, Clone, Default)]
#[repr(C)]
pub struct stereo_enc_state {
    pub pred_prev_q13: [i16; 2],
    pub s_mid: [i16; 2],
    pub s_side: [i16; 2],
    pub mid_side_amp_q0: [i32; 4],
    pub smth_width_q14: i16,
    pub width_prev_q14: i16,
    pub silent_side_len: i16,
    pub pred_ix: [[[i8; 3]; 2]; 3],
    pub mid_only_flags: [i8; 3],
}

#[derive(Copy, Clone, Default)]
#[repr(C)]
pub struct stereo_dec_state {
    pub pred_prev_q13: [i16; 2],
    pub s_mid: [i16; 2],
    pub s_side: [i16; 2],
}
