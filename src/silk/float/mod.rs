//! SILK floating-point processing variants.
//!
//! Upstream C: `silk/float/`

pub mod LPC_analysis_filter_FLP;
pub mod LTP_analysis_filter_FLP;
pub mod LTP_scale_ctrl_FLP;
pub mod SigProc_FLP;
pub mod apply_sine_window_FLP;
pub mod autocorrelation_FLP;
pub mod burg_modified_FLP;
pub mod bwexpander_FLP;
pub mod corrMatrix_FLP;
pub mod encode_frame_FLP;
pub mod energy_FLP;
pub mod find_LPC_FLP;
pub mod find_LTP_FLP;
pub mod find_pitch_lags_FLP;
pub mod find_pred_coefs_FLP;
pub mod inner_product_FLP;
pub mod k2a_FLP;
pub mod noise_shape_analysis_FLP;
pub mod pitch_analysis_core_FLP;
pub mod process_gains_FLP;
pub mod residual_energy_FLP;
pub mod scale_copy_vector_FLP;
pub mod schur_FLP;
pub mod sort_FLP;
pub mod structs_FLP;
pub mod warped_autocorrelation_FLP;
pub mod wrappers_FLP;
