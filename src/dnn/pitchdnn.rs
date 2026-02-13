//! Neural pitch detector (PitchDNN).
//!
//! Uses a combination of instantaneous frequency features and cross-correlation
//! features to estimate a continuous pitch value via a small DNN.
//!
//! Upstream C: `dnn/pitchdnn.c`, `dnn/pitchdnn.h`, `dnn/pitchdnn_data.h`

use super::nnet::*;

// --- Constants from pitchdnn_data.h ---

const DENSE_IF_UPSAMPLER_1_OUT_SIZE: usize = 64;
const DENSE_IF_UPSAMPLER_2_OUT_SIZE: usize = 64;
const DENSE_DOWNSAMPLER_OUT_SIZE: usize = 64;
const DENSE_FINAL_UPSAMPLER_OUT_SIZE: usize = 192;
const GRU_1_STATE_SIZE: usize = 64;

// --- Constants from pitchdnn.h ---

pub const PITCH_MIN_PERIOD: usize = 32;
pub const PITCH_MAX_PERIOD: usize = 256;
pub const NB_XCORR_FEATURES: usize = PITCH_MAX_PERIOD - PITCH_MIN_PERIOD;

// --- Model ---

/// PitchDNN model: collection of layers for neural pitch estimation.
///
/// Upstream C: dnn/pitchdnn_data.h:PitchDNN
#[derive(Clone, Debug, Default)]
pub struct PitchDNN {
    pub dense_if_upsampler_1: LinearLayer,
    pub dense_if_upsampler_2: LinearLayer,
    pub dense_downsampler: LinearLayer,
    pub dense_final_upsampler: LinearLayer,
    pub conv2d_1: Conv2dLayer,
    pub conv2d_2: Conv2dLayer,
    pub gru_1_input: LinearLayer,
    pub gru_1_recurrent: LinearLayer,
}

/// Initialize PitchDNN model from weight arrays.
///
/// Upstream C: dnn/pitchdnn_data.c:init_pitchdnn
pub fn init_pitchdnn(arrays: &[WeightArray]) -> Option<PitchDNN> {
    Some(PitchDNN {
        dense_if_upsampler_1: linear_init(
            arrays,
            "dense_if_upsampler_1_bias",
            "dense_if_upsampler_1_subias",
            "dense_if_upsampler_1_weights_int8",
            "dense_if_upsampler_1_weights_float",
            "",
            "",
            "dense_if_upsampler_1_scale",
            88,
            64,
        )?,
        dense_if_upsampler_2: linear_init(
            arrays,
            "dense_if_upsampler_2_bias",
            "dense_if_upsampler_2_subias",
            "dense_if_upsampler_2_weights_int8",
            "dense_if_upsampler_2_weights_float",
            "",
            "",
            "dense_if_upsampler_2_scale",
            64,
            64,
        )?,
        dense_downsampler: linear_init(
            arrays,
            "dense_downsampler_bias",
            "dense_downsampler_subias",
            "dense_downsampler_weights_int8",
            "dense_downsampler_weights_float",
            "",
            "",
            "dense_downsampler_scale",
            288,
            64,
        )?,
        dense_final_upsampler: linear_init(
            arrays,
            "dense_final_upsampler_bias",
            "dense_final_upsampler_subias",
            "dense_final_upsampler_weights_int8",
            "dense_final_upsampler_weights_float",
            "",
            "",
            "dense_final_upsampler_scale",
            64,
            192,
        )?,
        conv2d_1: conv2d_init(arrays, "conv2d_1_bias", "conv2d_1_weight_float", 1, 4, 3, 3)?,
        conv2d_2: conv2d_init(arrays, "conv2d_2_bias", "conv2d_2_weight_float", 4, 1, 3, 3)?,
        gru_1_input: linear_init(
            arrays,
            "gru_1_input_bias",
            "gru_1_input_subias",
            "gru_1_input_weights_int8",
            "gru_1_input_weights_float",
            "",
            "",
            "gru_1_input_scale",
            64,
            192,
        )?,
        gru_1_recurrent: linear_init(
            arrays,
            "gru_1_recurrent_bias",
            "gru_1_recurrent_subias",
            "gru_1_recurrent_weights_int8",
            "gru_1_recurrent_weights_float",
            "",
            "",
            "gru_1_recurrent_scale",
            64,
            192,
        )?,
    })
}

// --- State ---

/// PitchDNN state: model + per-frame state.
///
/// Upstream C: dnn/pitchdnn.h:PitchDNNState
#[derive(Clone)]
pub struct PitchDNNState {
    pub model: PitchDNN,
    pub gru_state: Vec<f32>,
    pub xcorr_mem1: Vec<f32>,
    pub xcorr_mem2: Vec<f32>,
}

impl Default for PitchDNNState {
    fn default() -> Self {
        Self::new()
    }
}

impl PitchDNNState {
    pub fn new() -> Self {
        PitchDNNState {
            model: PitchDNN::default(),
            gru_state: vec![0.0; GRU_1_STATE_SIZE],
            xcorr_mem1: vec![0.0; (NB_XCORR_FEATURES + 2) * 2],
            xcorr_mem2: vec![0.0; (NB_XCORR_FEATURES + 2) * 2 * 8],
        }
    }

    /// Initialize with pre-compiled weights (non-USE_WEIGHTS_FILE path).
    ///
    /// Upstream C: dnn/pitchdnn.c:pitchdnn_init
    pub fn init(&mut self, arrays: &[WeightArray]) -> bool {
        match init_pitchdnn(arrays) {
            Some(model) => {
                self.model = model;
                self.gru_state.fill(0.0);
                self.xcorr_mem1.fill(0.0);
                self.xcorr_mem2.fill(0.0);
                true
            }
            None => false,
        }
    }
}

/// Compute neural pitch estimate from IF and cross-correlation features.
///
/// Returns a continuous pitch value in the range used by LPCNet features.
///
/// Upstream C: dnn/pitchdnn.c:compute_pitchdnn
pub fn compute_pitchdnn(
    st: &mut PitchDNNState,
    if_features: &[f32],
    xcorr_features: &[f32],
) -> f32 {
    let model = &st.model;

    // IF path: two dense upsamplers
    let mut if1_out = vec![0.0f32; DENSE_IF_UPSAMPLER_1_OUT_SIZE];
    compute_generic_dense(
        &model.dense_if_upsampler_1,
        &mut if1_out,
        if_features,
        ACTIVATION_TANH,
    );

    let mut downsampler_in = vec![0.0f32; NB_XCORR_FEATURES + DENSE_IF_UPSAMPLER_2_OUT_SIZE];
    compute_generic_dense(
        &model.dense_if_upsampler_2,
        &mut downsampler_in[NB_XCORR_FEATURES..],
        &if1_out,
        ACTIVATION_TANH,
    );

    // Xcorr path: two conv2d layers
    let mut conv1_tmp1 = vec![0.0f32; (NB_XCORR_FEATURES + 2) * 8];
    let mut conv1_tmp2 = vec![0.0f32; (NB_XCORR_FEATURES + 2) * 8];
    conv1_tmp1[1..1 + NB_XCORR_FEATURES].copy_from_slice(&xcorr_features[..NB_XCORR_FEATURES]);

    compute_conv2d(
        &model.conv2d_1,
        &mut conv1_tmp2[1..],
        &mut st.xcorr_mem1,
        &conv1_tmp1,
        NB_XCORR_FEATURES,
        NB_XCORR_FEATURES + 2,
        ACTIVATION_TANH,
    );
    compute_conv2d(
        &model.conv2d_2,
        &mut downsampler_in[..NB_XCORR_FEATURES],
        &mut st.xcorr_mem2,
        &conv1_tmp2,
        NB_XCORR_FEATURES,
        NB_XCORR_FEATURES,
        ACTIVATION_TANH,
    );

    // Merge -> downsampler -> GRU -> upsampler
    let mut downsampler_out = vec![0.0f32; DENSE_DOWNSAMPLER_OUT_SIZE];
    compute_generic_dense(
        &model.dense_downsampler,
        &mut downsampler_out,
        &downsampler_in,
        ACTIVATION_TANH,
    );

    compute_generic_gru(
        &model.gru_1_input,
        &model.gru_1_recurrent,
        &mut st.gru_state,
        &downsampler_out,
    );

    let mut output = vec![0.0f32; DENSE_FINAL_UPSAMPLER_OUT_SIZE];
    compute_generic_dense(
        &model.dense_final_upsampler,
        &mut output,
        &st.gru_state,
        ACTIVATION_LINEAR,
    );

    // Soft argmax: find peak in output, then weighted average around it
    let mut pos = 0;
    let mut maxval: f32 = -1.0;
    for i in 0..180 {
        if output[i] > maxval {
            pos = i;
            maxval = output[i];
        }
    }

    let mut sum: f32 = 0.0;
    let mut count: f32 = 0.0;
    let start = pos.saturating_sub(2);
    let end = 179.min(pos + 2);
    for i in start..=end {
        // C: exp(output[i]) — standard double-precision exp, NOT lpcnet_exp fast approx.
        let p = (output[i] as f64).exp() as f32;
        sum += p * i as f32;
        count += p;
    }

    (1.0 / 60.0) * (sum / count) - 1.5
}
