//! Floating-point SILK data structures.
//!
//! Upstream c: `silk/float/structs_FLP.h`

use crate::silk::structs::{silk_encoder_state, stereo_enc_state};

#[derive(Copy, Clone, Default)]
#[repr(C)]
pub struct silk_encoder {
    pub state_fxx: [silk_encoder_state_FLP; 2],
    pub s_stereo: stereo_enc_state,
    pub n_bits_used_lbrr: i32,
    pub n_bits_exceeded: i32,
    pub n_channels_api: i32,
    pub n_channels_internal: i32,
    pub n_prev_channels_internal: i32,
    pub time_since_switch_allowed_ms: i32,
    pub allow_bandwidth_switch: i32,
    pub prev_decode_only_middle: i32,
}
#[derive(Copy, Clone)]
#[repr(C)]
pub struct silk_encoder_state_FLP {
    pub s_cmn: silk_encoder_state,
    pub s_shape: silk_shape_state_FLP,
    pub x_buf: [f32; 720],
    pub ltpcorr: f32,
}

impl Default for silk_encoder_state_FLP {
    fn default() -> Self {
        Self {
            s_cmn: Default::default(),
            s_shape: Default::default(),
            x_buf: [0.0; 720],
            ltpcorr: 0.0,
        }
    }
}
#[derive(Copy, Clone, Default)]
#[repr(C)]
pub struct silk_shape_state_FLP {
    pub last_gain_index: i8,
    pub harm_shape_gain_smth: f32,
    pub tilt_smth: f32,
}
#[derive(Copy, Clone)]
#[repr(C)]
pub struct silk_encoder_control_FLP {
    pub gains: [f32; 4],
    pub pred_coef: [[f32; 16]; 2],
    pub ltp_coef: [f32; 20],
    pub ltp_scale: f32,
    pub pitch_l: [i32; 4],
    pub ar: [f32; 96],
    pub lf_ma_shp: [f32; 4],
    pub lf_ar_shp: [f32; 4],
    pub tilt: [f32; 4],
    pub harm_shape_gain: [f32; 4],
    pub lambda: f32,
    pub input_quality: f32,
    pub coding_quality: f32,
    pub pred_gain: f32,
    pub lt_pred_cod_gain: f32,
    pub res_nrg: [f32; 4],
    pub gains_unq_q16: [i32; 4],
    pub last_gain_index_prev: i8,
}
