//! SILK codec — speech-optimized audio coding.
//!
//! Upstream C: `silk/`

#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(non_upper_case_globals)]
#![allow(unused_assignments)]
#![allow(clippy::too_many_arguments)]

#[cfg(feature = "simd")]
pub mod simd;

pub mod A2NLSF;
pub mod CNG;
pub mod HP_variable_cutoff;
pub mod LPC_analysis_filter;
pub mod LPC_fit;
pub mod LPC_inv_pred_gain;
pub mod LP_variable_cutoff;
pub mod NLSF2A;
pub mod NLSF_VQ;
pub mod NLSF_VQ_weights_laroia;
pub mod NLSF_decode;
pub mod NLSF_del_dec_quant;
pub mod NLSF_encode;
pub mod NLSF_stabilize;
pub mod NLSF_unpack;
pub mod NSQ;
pub mod NSQ_del_dec;
pub mod PLC;
pub mod VAD;
pub mod VQ_WMat_EC;
pub mod ana_filt_bank_1;
pub mod biquad_alt;
pub mod bwexpander;
pub mod bwexpander_32;
pub mod check_control_input;
pub mod code_signs;
pub mod control_SNR;
pub mod control_audio_bandwidth;
pub mod control_codec;
pub mod debug;
pub mod dec_API;
pub mod decode_core;
pub mod decode_frame;
pub mod decode_indices;
pub mod decode_parameters;
pub mod decode_pitch;
pub mod decode_pulses;
pub mod decoder_set_fs;
pub mod enc_API;
pub mod encode_indices;
pub mod encode_pulses;
pub mod float;
pub mod gain_quant;
pub mod init_decoder;
pub mod init_encoder;
pub mod inner_prod_aligned;
pub mod interpolate;
pub mod lin2log;
pub mod log2lin;
pub mod pitch_est_tables;
pub mod process_NLSFs;
pub mod quant_LTP_gains;
pub mod resampler;
pub mod shell_coder;
pub mod sigm_Q15;
pub mod sort;
pub mod stereo_LR_to_MS;
pub mod stereo_MS_to_LR;
pub mod stereo_decode_pred;
pub mod stereo_encode_pred;
pub mod stereo_find_predictor;
pub mod stereo_quant_pred;
pub mod sum_sqr_shift;
pub mod table_LSF_cos;
pub mod tables_LTP;
pub mod tables_NLSF_CB_NB_MB;
pub mod tables_NLSF_CB_WB;
pub mod tables_gain;
pub mod tables_other;
pub mod tables_pitch_lag;
pub mod tables_pulses_per_block;
// stuff for structs that do not have a clear home, named after the header files
pub mod Inlines;
pub mod SigProc_FIX;
pub mod define;
pub mod macros;
pub mod structs;
pub mod tuning_parameters;

pub mod mathops;
