//! OSCE: Opus Speech Coding Enhancement (LACE and NoLACE models).
//!
//! Post-processes SILK decoded frames using adaptive filtering networks.
//!
//! Upstream C: `dnn/osce.c`, `dnn/osce.h`, `dnn/osce_features.c`,
//! `dnn/osce_config.h`, `dnn/osce_structs.h`

use crate::dnn::freq::{forward_transform, NB_BANDS};
use crate::dnn::nndsp::*;
use crate::dnn::nnet::*;

use crate::celt::kiss_fft::kiss_fft_cpx;

// ========== OSCE Config (osce_config.h) ==========

pub const OSCE_FEATURES_MAX_HISTORY: usize = 350;
pub const OSCE_FEATURE_DIM: usize = 93;
pub const OSCE_MAX_FEATURE_FRAMES: usize = 4;

pub const OSCE_CLEAN_SPEC_NUM_BANDS: usize = 64;
pub const OSCE_NOISY_SPEC_NUM_BANDS: usize = 18;

pub const OSCE_NO_PITCH_VALUE: i32 = 7;
pub const OSCE_PREEMPH: f32 = 0.85;
pub const OSCE_PITCH_HANGOVER: usize = 0;

pub const OSCE_CLEAN_SPEC_START: usize = 0;
pub const OSCE_CLEAN_SPEC_LENGTH: usize = 64;
pub const OSCE_NOISY_CEPSTRUM_START: usize = 64;
pub const OSCE_NOISY_CEPSTRUM_LENGTH: usize = 18;
pub const OSCE_ACORR_START: usize = 82;
pub const OSCE_ACORR_LENGTH: usize = 5;
pub const OSCE_LTP_START: usize = 87;
pub const OSCE_LTP_LENGTH: usize = 5;
pub const OSCE_LOG_GAIN_START: usize = 92;
pub const OSCE_LOG_GAIN_LENGTH: usize = 1;

pub const OSCE_METHOD_NONE: i32 = 0;
pub const OSCE_METHOD_LACE: i32 = 1;
pub const OSCE_METHOD_NOLACE: i32 = 2;

const OSCE_SPEC_WINDOW_SIZE: usize = 320;
const OSCE_SPEC_NUM_FREQS: usize = 161;

// ========== LACE Constants (lace_data.h) ==========

pub const LACE_PREEMPH: f32 = 0.85;
pub const LACE_FRAME_SIZE: usize = 80;
pub const LACE_OVERLAP_SIZE: usize = 40;
pub const LACE_NUM_FEATURES: usize = 93;
pub const LACE_PITCH_MAX: usize = 300;
pub const LACE_PITCH_EMBEDDING_DIM: usize = 64;
pub const LACE_NUMBITS_RANGE_LOW: f32 = 50.0;
pub const LACE_NUMBITS_RANGE_HIGH: f32 = 650.0;
pub const LACE_NUMBITS_EMBEDDING_DIM: usize = 8;
pub const LACE_COND_DIM: usize = 128;
pub const LACE_HIDDEN_FEATURE_DIM: usize = 96;
pub const LACE_NUMBITS_SCALES: [f32; 8] = [
    1.0983514785766602,
    2.0509142875671387,
    3.5729939937591553,
    4.478035926818848,
    5.926519393920898,
    7.152282238006592,
    8.277412414550781,
    8.926830291748047,
];

pub const LACE_CF1_FILTER_GAIN_A: f32 = 0.690776;
pub const LACE_CF1_FILTER_GAIN_B: f32 = 0.0;
pub const LACE_CF1_LOG_GAIN_LIMIT: f32 = 1.151293;
pub const LACE_CF1_KERNEL_SIZE: usize = 16;
pub const LACE_CF1_LEFT_PADDING: usize = 8;

pub const LACE_CF2_FILTER_GAIN_A: f32 = 0.690776;
pub const LACE_CF2_FILTER_GAIN_B: f32 = 0.0;
pub const LACE_CF2_LOG_GAIN_LIMIT: f32 = 1.151293;
pub const LACE_CF2_KERNEL_SIZE: usize = 16;
pub const LACE_CF2_LEFT_PADDING: usize = 8;

pub const LACE_AF1_FILTER_GAIN_A: f32 = 1.381551;
pub const LACE_AF1_FILTER_GAIN_B: f32 = 0.0;
pub const LACE_AF1_SHAPE_GAIN: f32 = 1.0;
pub const LACE_AF1_KERNEL_SIZE: usize = 16;
pub const LACE_AF1_LEFT_PADDING: usize = 15;
pub const LACE_AF1_IN_CHANNELS: usize = 1;
pub const LACE_AF1_OUT_CHANNELS: usize = 1;

// ========== NoLACE Constants (nolace_data.h) ==========

pub const NOLACE_PREEMPH: f32 = 0.85;
pub const NOLACE_FRAME_SIZE: usize = 80;
pub const NOLACE_OVERLAP_SIZE: usize = 40;
pub const NOLACE_NUM_FEATURES: usize = 93;
pub const NOLACE_PITCH_MAX: usize = 300;
pub const NOLACE_PITCH_EMBEDDING_DIM: usize = 64;
pub const NOLACE_NUMBITS_RANGE_LOW: f32 = 50.0;
pub const NOLACE_NUMBITS_RANGE_HIGH: f32 = 650.0;
pub const NOLACE_NUMBITS_EMBEDDING_DIM: usize = 8;
pub const NOLACE_COND_DIM: usize = 160;
pub const NOLACE_HIDDEN_FEATURE_DIM: usize = 96;
pub const NOLACE_NUMBITS_SCALES: [f32; 8] = [
    1.0357311964035034,
    1.735559105873108,
    3.6004557609558105,
    4.552478313446045,
    5.932559490203857,
    7.176970481872559,
    8.114998817443848,
    8.77063274383545,
];

pub const NOLACE_CF1_FILTER_GAIN_A: f32 = 0.690776;
pub const NOLACE_CF1_FILTER_GAIN_B: f32 = 0.0;
pub const NOLACE_CF1_LOG_GAIN_LIMIT: f32 = 1.151293;
pub const NOLACE_CF1_KERNEL_SIZE: usize = 16;
pub const NOLACE_CF1_LEFT_PADDING: usize = 8;

pub const NOLACE_CF2_FILTER_GAIN_A: f32 = 0.690776;
pub const NOLACE_CF2_FILTER_GAIN_B: f32 = 0.0;
pub const NOLACE_CF2_LOG_GAIN_LIMIT: f32 = 1.151293;
pub const NOLACE_CF2_KERNEL_SIZE: usize = 16;
pub const NOLACE_CF2_LEFT_PADDING: usize = 8;

pub const NOLACE_AF1_FILTER_GAIN_A: f32 = 1.381551;
pub const NOLACE_AF1_FILTER_GAIN_B: f32 = 0.0;
pub const NOLACE_AF1_SHAPE_GAIN: f32 = 1.0;
pub const NOLACE_AF1_KERNEL_SIZE: usize = 16;
pub const NOLACE_AF1_LEFT_PADDING: usize = 15;
pub const NOLACE_AF1_IN_CHANNELS: usize = 1;
pub const NOLACE_AF1_OUT_CHANNELS: usize = 2;

pub const NOLACE_AF2_FILTER_GAIN_A: f32 = 1.381551;
pub const NOLACE_AF2_FILTER_GAIN_B: f32 = 0.0;
pub const NOLACE_AF2_SHAPE_GAIN: f32 = 1.0;
pub const NOLACE_AF2_KERNEL_SIZE: usize = 16;
pub const NOLACE_AF2_LEFT_PADDING: usize = 15;
pub const NOLACE_AF2_IN_CHANNELS: usize = 2;
pub const NOLACE_AF2_OUT_CHANNELS: usize = 2;

pub const NOLACE_AF3_FILTER_GAIN_A: f32 = 1.381551;
pub const NOLACE_AF3_FILTER_GAIN_B: f32 = 0.0;
pub const NOLACE_AF3_SHAPE_GAIN: f32 = 1.0;
pub const NOLACE_AF3_KERNEL_SIZE: usize = 16;
pub const NOLACE_AF3_LEFT_PADDING: usize = 15;
pub const NOLACE_AF3_IN_CHANNELS: usize = 2;
pub const NOLACE_AF3_OUT_CHANNELS: usize = 2;

pub const NOLACE_AF4_FILTER_GAIN_A: f32 = 1.381551;
pub const NOLACE_AF4_FILTER_GAIN_B: f32 = 0.0;
pub const NOLACE_AF4_SHAPE_GAIN: f32 = 1.0;
pub const NOLACE_AF4_KERNEL_SIZE: usize = 16;
pub const NOLACE_AF4_LEFT_PADDING: usize = 15;
pub const NOLACE_AF4_IN_CHANNELS: usize = 2;
pub const NOLACE_AF4_OUT_CHANNELS: usize = 1;

pub const NOLACE_TDSHAPE1_FEATURE_DIM: usize = 160;
pub const NOLACE_TDSHAPE1_FRAME_SIZE: usize = 80;
pub const NOLACE_TDSHAPE1_AVG_POOL_K: usize = 4;

pub const NOLACE_TDSHAPE2_FEATURE_DIM: usize = 160;
pub const NOLACE_TDSHAPE2_FRAME_SIZE: usize = 80;
pub const NOLACE_TDSHAPE2_AVG_POOL_K: usize = 4;

pub const NOLACE_TDSHAPE3_FEATURE_DIM: usize = 160;
pub const NOLACE_TDSHAPE3_FRAME_SIZE: usize = 80;
pub const NOLACE_TDSHAPE3_AVG_POOL_K: usize = 4;

// ========== Feature Tables (osce_features.c) ==========

static CENTER_BINS_CLEAN: [usize; 64] = [
    0, 2, 5, 8, 10, 12, 15, 18, 20, 22, 25, 28, 30, 33, 35, 38, 40, 42, 45, 48, 50, 52, 55, 58, 60,
    62, 65, 68, 70, 73, 75, 78, 80, 82, 85, 88, 90, 92, 95, 98, 100, 102, 105, 108, 110, 112, 115,
    118, 120, 122, 125, 128, 130, 132, 135, 138, 140, 142, 145, 148, 150, 152, 155, 160,
];

static CENTER_BINS_NOISY: [usize; 18] = [
    0, 4, 8, 12, 16, 20, 24, 28, 32, 40, 48, 56, 64, 80, 96, 112, 136, 160,
];

static BAND_WEIGHTS_CLEAN: [f32; 64] = [
    0.666666666667,
    0.400000000000,
    0.333333333333,
    0.400000000000,
    0.500000000000,
    0.400000000000,
    0.333333333333,
    0.400000000000,
    0.500000000000,
    0.400000000000,
    0.333333333333,
    0.400000000000,
    0.400000000000,
    0.400000000000,
    0.400000000000,
    0.400000000000,
    0.500000000000,
    0.400000000000,
    0.333333333333,
    0.400000000000,
    0.500000000000,
    0.400000000000,
    0.333333333333,
    0.400000000000,
    0.500000000000,
    0.400000000000,
    0.333333333333,
    0.400000000000,
    0.400000000000,
    0.400000000000,
    0.400000000000,
    0.400000000000,
    0.500000000000,
    0.400000000000,
    0.333333333333,
    0.400000000000,
    0.500000000000,
    0.400000000000,
    0.333333333333,
    0.400000000000,
    0.500000000000,
    0.400000000000,
    0.333333333333,
    0.400000000000,
    0.500000000000,
    0.400000000000,
    0.333333333333,
    0.400000000000,
    0.500000000000,
    0.400000000000,
    0.333333333333,
    0.400000000000,
    0.500000000000,
    0.400000000000,
    0.333333333333,
    0.400000000000,
    0.500000000000,
    0.400000000000,
    0.333333333333,
    0.400000000000,
    0.500000000000,
    0.400000000000,
    0.250000000000,
    0.333333333333,
];

static BAND_WEIGHTS_NOISY: [f32; 18] = [
    0.400000000000,
    0.250000000000,
    0.250000000000,
    0.250000000000,
    0.250000000000,
    0.250000000000,
    0.250000000000,
    0.250000000000,
    0.166666666667,
    0.125000000000,
    0.125000000000,
    0.125000000000,
    0.083333333333,
    0.062500000000,
    0.062500000000,
    0.050000000000,
    0.041666666667,
    0.080000000000,
];

// The 320-sample analysis window is embedded here (from osce_features.c).
// For brevity, include only the generation formula used at init time.
fn generate_osce_window() -> [f32; OSCE_SPEC_WINDOW_SIZE] {
    let mut w = [0.0f32; OSCE_SPEC_WINDOW_SIZE];
    let n = OSCE_SPEC_WINDOW_SIZE;
    for i in 0..n {
        // Hann window: 0.5 * (1 - cos(2*pi*i/(n-1)))... but the C uses a custom sine window.
        // The C window is precomputed. We'll use the formula: sin(pi*(i+0.5)/n) for a sine window.
        // Actually the C array matches: w[i] = sin(pi * (i + 0.5) / n) for i in 0..160, then mirror.
        // Let's just use the exact formula.
        w[i] = (std::f32::consts::PI * (i as f32 + 0.5) / n as f32).sin();
    }
    w
}

// ========== Structs ==========

/// Feature extraction state.
///
/// Upstream C: dnn/osce_structs.h:OSCEFeatureState
#[derive(Clone)]
pub struct OSCEFeatureState {
    pub numbits_smooth: f32,
    pub pitch_hangover_count: usize,
    pub last_lag: i32,
    pub last_type: i32,
    pub signal_history: Vec<f32>,
    pub reset: i32,
}

impl Default for OSCEFeatureState {
    fn default() -> Self {
        OSCEFeatureState {
            numbits_smooth: 0.0,
            pitch_hangover_count: 0,
            last_lag: 0,
            last_type: 0,
            signal_history: vec![0.0; OSCE_FEATURES_MAX_HISTORY],
            reset: 0,
        }
    }
}

/// LACE model layers.
///
/// Upstream C: dnn/lace_data.h:LACELayers
pub struct LACELayers {
    pub pitch_embedding: LinearLayer,
    pub fnet_conv1: LinearLayer,
    pub fnet_conv2: LinearLayer,
    pub fnet_tconv: LinearLayer,
    pub fnet_gru_input: LinearLayer,
    pub fnet_gru_recurrent: LinearLayer,
    pub cf1_kernel: LinearLayer,
    pub cf1_gain: LinearLayer,
    pub cf1_global_gain: LinearLayer,
    pub cf2_kernel: LinearLayer,
    pub cf2_gain: LinearLayer,
    pub cf2_global_gain: LinearLayer,
    pub af1_kernel: LinearLayer,
    pub af1_gain: LinearLayer,
}

/// LACE model (layers + overlap window).
///
/// Upstream C: dnn/osce_structs.h:LACE
pub struct LACE {
    pub layers: LACELayers,
    pub window: Vec<f32>,
}

/// LACE runtime state.
///
/// Upstream C: dnn/osce_structs.h:LACEState
pub struct LACEState {
    pub feature_net_conv2_state: Vec<f32>,
    pub feature_net_gru_state: Vec<f32>,
    pub cf1_state: AdaCombState,
    pub cf2_state: AdaCombState,
    pub af1_state: AdaConvState,
    pub preemph_mem: f32,
    pub deemph_mem: f32,
}

impl Default for LACEState {
    fn default() -> Self {
        LACEState {
            feature_net_conv2_state: vec![0.0; 384], // LACE_FNET_CONV2_STATE_SIZE
            feature_net_gru_state: vec![0.0; LACE_COND_DIM],
            cf1_state: AdaCombState::default(),
            cf2_state: AdaCombState::default(),
            af1_state: AdaConvState::default(),
            preemph_mem: 0.0,
            deemph_mem: 0.0,
        }
    }
}

/// NoLACE model layers.
///
/// Upstream C: dnn/nolace_data.h:NOLACELayers
pub struct NOLACELayers {
    pub pitch_embedding: LinearLayer,
    pub fnet_conv1: LinearLayer,
    pub fnet_conv2: LinearLayer,
    pub fnet_tconv: LinearLayer,
    pub fnet_gru_input: LinearLayer,
    pub fnet_gru_recurrent: LinearLayer,
    pub cf1_kernel: LinearLayer,
    pub cf1_gain: LinearLayer,
    pub cf1_global_gain: LinearLayer,
    pub cf2_kernel: LinearLayer,
    pub cf2_gain: LinearLayer,
    pub cf2_global_gain: LinearLayer,
    pub af1_kernel: LinearLayer,
    pub af1_gain: LinearLayer,
    pub tdshape1_alpha1_f: LinearLayer,
    pub tdshape1_alpha1_t: LinearLayer,
    pub tdshape1_alpha2: LinearLayer,
    pub tdshape2_alpha1_f: LinearLayer,
    pub tdshape2_alpha1_t: LinearLayer,
    pub tdshape2_alpha2: LinearLayer,
    pub tdshape3_alpha1_f: LinearLayer,
    pub tdshape3_alpha1_t: LinearLayer,
    pub tdshape3_alpha2: LinearLayer,
    pub af2_kernel: LinearLayer,
    pub af2_gain: LinearLayer,
    pub af3_kernel: LinearLayer,
    pub af3_gain: LinearLayer,
    pub af4_kernel: LinearLayer,
    pub af4_gain: LinearLayer,
    pub post_cf1: LinearLayer,
    pub post_cf2: LinearLayer,
    pub post_af1: LinearLayer,
    pub post_af2: LinearLayer,
    pub post_af3: LinearLayer,
}

/// NoLACE model (layers + overlap window).
///
/// Upstream C: dnn/osce_structs.h:NoLACE
pub struct NoLACE {
    pub layers: NOLACELayers,
    pub window: Vec<f32>,
}

/// NoLACE runtime state.
///
/// Upstream C: dnn/osce_structs.h:NoLACEState
pub struct NoLACEState {
    pub feature_net_conv2_state: Vec<f32>,
    pub feature_net_gru_state: Vec<f32>,
    pub post_cf1_state: Vec<f32>,
    pub post_cf2_state: Vec<f32>,
    pub post_af1_state: Vec<f32>,
    pub post_af2_state: Vec<f32>,
    pub post_af3_state: Vec<f32>,
    pub cf1_state: AdaCombState,
    pub cf2_state: AdaCombState,
    pub af1_state: AdaConvState,
    pub af2_state: AdaConvState,
    pub af3_state: AdaConvState,
    pub af4_state: AdaConvState,
    pub tdshape1_state: AdaShapeState,
    pub tdshape2_state: AdaShapeState,
    pub tdshape3_state: AdaShapeState,
    pub preemph_mem: f32,
    pub deemph_mem: f32,
}

impl Default for NoLACEState {
    fn default() -> Self {
        NoLACEState {
            feature_net_conv2_state: vec![0.0; 384], // NOLACE_FNET_CONV2_STATE_SIZE
            feature_net_gru_state: vec![0.0; NOLACE_COND_DIM],
            post_cf1_state: vec![0.0; NOLACE_COND_DIM],
            post_cf2_state: vec![0.0; NOLACE_COND_DIM],
            post_af1_state: vec![0.0; NOLACE_COND_DIM],
            post_af2_state: vec![0.0; NOLACE_COND_DIM],
            post_af3_state: vec![0.0; NOLACE_COND_DIM],
            cf1_state: AdaCombState::default(),
            cf2_state: AdaCombState::default(),
            af1_state: AdaConvState::default(),
            af2_state: AdaConvState::default(),
            af3_state: AdaConvState::default(),
            af4_state: AdaConvState::default(),
            tdshape1_state: AdaShapeState::default(),
            tdshape2_state: AdaShapeState::default(),
            tdshape3_state: AdaShapeState::default(),
            preemph_mem: 0.0,
            deemph_mem: 0.0,
        }
    }
}

/// Top-level OSCE model container.
///
/// Upstream C: dnn/osce_structs.h:OSCEModel
#[derive(Default)]
pub struct OSCEModel {
    pub loaded: bool,
    pub lace: Option<LACE>,
    pub nolace: Option<NoLACE>,
}

/// Combined OSCE state (features + method-specific runtime state).
///
/// Upstream C: silk/structs.h:silk_OSCE_struct
pub struct OSCEState {
    pub features: OSCEFeatureState,
    pub lace_state: LACEState,
    pub nolace_state: NoLACEState,
    pub method: i32,
}

impl Default for OSCEState {
    fn default() -> Self {
        OSCEState {
            features: OSCEFeatureState::default(),
            lace_state: LACEState::default(),
            nolace_state: NoLACEState::default(),
            method: OSCE_METHOD_NONE,
        }
    }
}

// ========== Weight Initialization ==========

/// Initialize LACE model from weight arrays.
///
/// Upstream C: dnn/lace_data.c:init_lacelayers
pub fn init_lace(arrays: &[WeightArray]) -> Option<LACE> {
    let layers = LACELayers {
        pitch_embedding: linear_init(
            arrays,
            "lace_pitch_embedding_bias",
            "",
            "",
            "lace_pitch_embedding_weights_float",
            "",
            "",
            "",
            301,
            64,
        )?,
        fnet_conv1: linear_init(
            arrays,
            "lace_fnet_conv1_bias",
            "",
            "",
            "lace_fnet_conv1_weights_float",
            "",
            "",
            "",
            173,
            96,
        )?,
        fnet_conv2: linear_init(
            arrays,
            "lace_fnet_conv2_bias",
            "lace_fnet_conv2_subias",
            "lace_fnet_conv2_weights_int8",
            "lace_fnet_conv2_weights_float",
            "",
            "",
            "lace_fnet_conv2_scale",
            768,
            128,
        )?,
        fnet_tconv: linear_init(
            arrays,
            "lace_fnet_tconv_bias",
            "lace_fnet_tconv_subias",
            "lace_fnet_tconv_weights_int8",
            "lace_fnet_tconv_weights_float",
            "",
            "",
            "lace_fnet_tconv_scale",
            128,
            512,
        )?,
        fnet_gru_input: linear_init(
            arrays,
            "lace_fnet_gru_input_bias",
            "lace_fnet_gru_input_subias",
            "lace_fnet_gru_input_weights_int8",
            "lace_fnet_gru_input_weights_float",
            "",
            "",
            "lace_fnet_gru_input_scale",
            128,
            384,
        )?,
        fnet_gru_recurrent: linear_init(
            arrays,
            "lace_fnet_gru_recurrent_bias",
            "lace_fnet_gru_recurrent_subias",
            "lace_fnet_gru_recurrent_weights_int8",
            "lace_fnet_gru_recurrent_weights_float",
            "",
            "",
            "lace_fnet_gru_recurrent_scale",
            128,
            384,
        )?,
        cf1_kernel: linear_init(
            arrays,
            "lace_cf1_kernel_bias",
            "lace_cf1_kernel_subias",
            "lace_cf1_kernel_weights_int8",
            "lace_cf1_kernel_weights_float",
            "",
            "",
            "lace_cf1_kernel_scale",
            128,
            16,
        )?,
        cf1_gain: linear_init(
            arrays,
            "lace_cf1_gain_bias",
            "",
            "",
            "lace_cf1_gain_weights_float",
            "",
            "",
            "",
            128,
            1,
        )?,
        cf1_global_gain: linear_init(
            arrays,
            "lace_cf1_global_gain_bias",
            "",
            "",
            "lace_cf1_global_gain_weights_float",
            "",
            "",
            "",
            128,
            1,
        )?,
        cf2_kernel: linear_init(
            arrays,
            "lace_cf2_kernel_bias",
            "lace_cf2_kernel_subias",
            "lace_cf2_kernel_weights_int8",
            "lace_cf2_kernel_weights_float",
            "",
            "",
            "lace_cf2_kernel_scale",
            128,
            16,
        )?,
        cf2_gain: linear_init(
            arrays,
            "lace_cf2_gain_bias",
            "",
            "",
            "lace_cf2_gain_weights_float",
            "",
            "",
            "",
            128,
            1,
        )?,
        cf2_global_gain: linear_init(
            arrays,
            "lace_cf2_global_gain_bias",
            "",
            "",
            "lace_cf2_global_gain_weights_float",
            "",
            "",
            "",
            128,
            1,
        )?,
        af1_kernel: linear_init(
            arrays,
            "lace_af1_kernel_bias",
            "lace_af1_kernel_subias",
            "lace_af1_kernel_weights_int8",
            "lace_af1_kernel_weights_float",
            "",
            "",
            "lace_af1_kernel_scale",
            128,
            16,
        )?,
        af1_gain: linear_init(
            arrays,
            "lace_af1_gain_bias",
            "",
            "",
            "lace_af1_gain_weights_float",
            "",
            "",
            "",
            128,
            1,
        )?,
    };
    let mut window = vec![0.0f32; LACE_OVERLAP_SIZE];
    compute_overlap_window(&mut window, LACE_OVERLAP_SIZE);
    Some(LACE { layers, window })
}

/// Initialize NoLACE model from weight arrays.
///
/// Upstream C: dnn/nolace_data.c:init_nolacelayers
pub fn init_nolace(arrays: &[WeightArray]) -> Option<NoLACE> {
    let layers = NOLACELayers {
        pitch_embedding: linear_init(
            arrays,
            "nolace_pitch_embedding_bias",
            "",
            "",
            "nolace_pitch_embedding_weights_float",
            "",
            "",
            "",
            301,
            64,
        )?,
        fnet_conv1: linear_init(
            arrays,
            "nolace_fnet_conv1_bias",
            "",
            "",
            "nolace_fnet_conv1_weights_float",
            "",
            "",
            "",
            173,
            96,
        )?,
        fnet_conv2: linear_init(
            arrays,
            "nolace_fnet_conv2_bias",
            "nolace_fnet_conv2_subias",
            "nolace_fnet_conv2_weights_int8",
            "nolace_fnet_conv2_weights_float",
            "",
            "",
            "nolace_fnet_conv2_scale",
            768,
            160,
        )?,
        fnet_tconv: linear_init(
            arrays,
            "nolace_fnet_tconv_bias",
            "nolace_fnet_tconv_subias",
            "nolace_fnet_tconv_weights_int8",
            "nolace_fnet_tconv_weights_float",
            "",
            "",
            "nolace_fnet_tconv_scale",
            160,
            640,
        )?,
        fnet_gru_input: linear_init(
            arrays,
            "nolace_fnet_gru_input_bias",
            "nolace_fnet_gru_input_subias",
            "nolace_fnet_gru_input_weights_int8",
            "nolace_fnet_gru_input_weights_float",
            "",
            "",
            "nolace_fnet_gru_input_scale",
            160,
            480,
        )?,
        fnet_gru_recurrent: linear_init(
            arrays,
            "nolace_fnet_gru_recurrent_bias",
            "nolace_fnet_gru_recurrent_subias",
            "nolace_fnet_gru_recurrent_weights_int8",
            "nolace_fnet_gru_recurrent_weights_float",
            "",
            "",
            "nolace_fnet_gru_recurrent_scale",
            160,
            480,
        )?,
        cf1_kernel: linear_init(
            arrays,
            "nolace_cf1_kernel_bias",
            "nolace_cf1_kernel_subias",
            "nolace_cf1_kernel_weights_int8",
            "nolace_cf1_kernel_weights_float",
            "",
            "",
            "nolace_cf1_kernel_scale",
            160,
            16,
        )?,
        cf1_gain: linear_init(
            arrays,
            "nolace_cf1_gain_bias",
            "",
            "",
            "nolace_cf1_gain_weights_float",
            "",
            "",
            "",
            160,
            1,
        )?,
        cf1_global_gain: linear_init(
            arrays,
            "nolace_cf1_global_gain_bias",
            "",
            "",
            "nolace_cf1_global_gain_weights_float",
            "",
            "",
            "",
            160,
            1,
        )?,
        cf2_kernel: linear_init(
            arrays,
            "nolace_cf2_kernel_bias",
            "nolace_cf2_kernel_subias",
            "nolace_cf2_kernel_weights_int8",
            "nolace_cf2_kernel_weights_float",
            "",
            "",
            "nolace_cf2_kernel_scale",
            160,
            16,
        )?,
        cf2_gain: linear_init(
            arrays,
            "nolace_cf2_gain_bias",
            "",
            "",
            "nolace_cf2_gain_weights_float",
            "",
            "",
            "",
            160,
            1,
        )?,
        cf2_global_gain: linear_init(
            arrays,
            "nolace_cf2_global_gain_bias",
            "",
            "",
            "nolace_cf2_global_gain_weights_float",
            "",
            "",
            "",
            160,
            1,
        )?,
        af1_kernel: linear_init(
            arrays,
            "nolace_af1_kernel_bias",
            "nolace_af1_kernel_subias",
            "nolace_af1_kernel_weights_int8",
            "nolace_af1_kernel_weights_float",
            "",
            "",
            "nolace_af1_kernel_scale",
            160,
            32,
        )?,
        af1_gain: linear_init(
            arrays,
            "nolace_af1_gain_bias",
            "",
            "",
            "nolace_af1_gain_weights_float",
            "",
            "",
            "",
            160,
            2,
        )?,
        tdshape1_alpha1_f: linear_init(
            arrays,
            "nolace_tdshape1_alpha1_f_bias",
            "nolace_tdshape1_alpha1_f_subias",
            "nolace_tdshape1_alpha1_f_weights_int8",
            "nolace_tdshape1_alpha1_f_weights_float",
            "",
            "",
            "nolace_tdshape1_alpha1_f_scale",
            320,
            80,
        )?,
        tdshape1_alpha1_t: linear_init(
            arrays,
            "nolace_tdshape1_alpha1_t_bias",
            "",
            "",
            "nolace_tdshape1_alpha1_t_weights_float",
            "",
            "",
            "",
            42,
            80,
        )?,
        tdshape1_alpha2: linear_init(
            arrays,
            "nolace_tdshape1_alpha2_bias",
            "",
            "",
            "nolace_tdshape1_alpha2_weights_float",
            "",
            "",
            "",
            160,
            80,
        )?,
        tdshape2_alpha1_f: linear_init(
            arrays,
            "nolace_tdshape2_alpha1_f_bias",
            "nolace_tdshape2_alpha1_f_subias",
            "nolace_tdshape2_alpha1_f_weights_int8",
            "nolace_tdshape2_alpha1_f_weights_float",
            "",
            "",
            "nolace_tdshape2_alpha1_f_scale",
            320,
            80,
        )?,
        tdshape2_alpha1_t: linear_init(
            arrays,
            "nolace_tdshape2_alpha1_t_bias",
            "",
            "",
            "nolace_tdshape2_alpha1_t_weights_float",
            "",
            "",
            "",
            42,
            80,
        )?,
        tdshape2_alpha2: linear_init(
            arrays,
            "nolace_tdshape2_alpha2_bias",
            "",
            "",
            "nolace_tdshape2_alpha2_weights_float",
            "",
            "",
            "",
            160,
            80,
        )?,
        tdshape3_alpha1_f: linear_init(
            arrays,
            "nolace_tdshape3_alpha1_f_bias",
            "nolace_tdshape3_alpha1_f_subias",
            "nolace_tdshape3_alpha1_f_weights_int8",
            "nolace_tdshape3_alpha1_f_weights_float",
            "",
            "",
            "nolace_tdshape3_alpha1_f_scale",
            320,
            80,
        )?,
        tdshape3_alpha1_t: linear_init(
            arrays,
            "nolace_tdshape3_alpha1_t_bias",
            "",
            "",
            "nolace_tdshape3_alpha1_t_weights_float",
            "",
            "",
            "",
            42,
            80,
        )?,
        tdshape3_alpha2: linear_init(
            arrays,
            "nolace_tdshape3_alpha2_bias",
            "",
            "",
            "nolace_tdshape3_alpha2_weights_float",
            "",
            "",
            "",
            160,
            80,
        )?,
        af2_kernel: linear_init(
            arrays,
            "nolace_af2_kernel_bias",
            "nolace_af2_kernel_subias",
            "nolace_af2_kernel_weights_int8",
            "nolace_af2_kernel_weights_float",
            "",
            "",
            "nolace_af2_kernel_scale",
            160,
            64,
        )?,
        af2_gain: linear_init(
            arrays,
            "nolace_af2_gain_bias",
            "",
            "",
            "nolace_af2_gain_weights_float",
            "",
            "",
            "",
            160,
            2,
        )?,
        af3_kernel: linear_init(
            arrays,
            "nolace_af3_kernel_bias",
            "nolace_af3_kernel_subias",
            "nolace_af3_kernel_weights_int8",
            "nolace_af3_kernel_weights_float",
            "",
            "",
            "nolace_af3_kernel_scale",
            160,
            64,
        )?,
        af3_gain: linear_init(
            arrays,
            "nolace_af3_gain_bias",
            "",
            "",
            "nolace_af3_gain_weights_float",
            "",
            "",
            "",
            160,
            2,
        )?,
        af4_kernel: linear_init(
            arrays,
            "nolace_af4_kernel_bias",
            "nolace_af4_kernel_subias",
            "nolace_af4_kernel_weights_int8",
            "nolace_af4_kernel_weights_float",
            "",
            "",
            "nolace_af4_kernel_scale",
            160,
            32,
        )?,
        af4_gain: linear_init(
            arrays,
            "nolace_af4_gain_bias",
            "",
            "",
            "nolace_af4_gain_weights_float",
            "",
            "",
            "",
            160,
            1,
        )?,
        post_cf1: linear_init(
            arrays,
            "nolace_post_cf1_bias",
            "nolace_post_cf1_subias",
            "nolace_post_cf1_weights_int8",
            "nolace_post_cf1_weights_float",
            "",
            "",
            "nolace_post_cf1_scale",
            320,
            160,
        )?,
        post_cf2: linear_init(
            arrays,
            "nolace_post_cf2_bias",
            "nolace_post_cf2_subias",
            "nolace_post_cf2_weights_int8",
            "nolace_post_cf2_weights_float",
            "",
            "",
            "nolace_post_cf2_scale",
            320,
            160,
        )?,
        post_af1: linear_init(
            arrays,
            "nolace_post_af1_bias",
            "nolace_post_af1_subias",
            "nolace_post_af1_weights_int8",
            "nolace_post_af1_weights_float",
            "",
            "",
            "nolace_post_af1_scale",
            320,
            160,
        )?,
        post_af2: linear_init(
            arrays,
            "nolace_post_af2_bias",
            "nolace_post_af2_subias",
            "nolace_post_af2_weights_int8",
            "nolace_post_af2_weights_float",
            "",
            "",
            "nolace_post_af2_scale",
            320,
            160,
        )?,
        post_af3: linear_init(
            arrays,
            "nolace_post_af3_bias",
            "nolace_post_af3_subias",
            "nolace_post_af3_weights_int8",
            "nolace_post_af3_weights_float",
            "",
            "",
            "nolace_post_af3_scale",
            320,
            160,
        )?,
    };
    let mut window = vec![0.0f32; NOLACE_OVERLAP_SIZE];
    compute_overlap_window(&mut window, NOLACE_OVERLAP_SIZE);
    Some(NoLACE { layers, window })
}

/// Load OSCE models from weight data.
///
/// Upstream C: dnn/osce.c:osce_load_models
pub fn osce_load_models(model: &mut OSCEModel, arrays: &[WeightArray]) -> bool {
    model.lace = init_lace(arrays);
    model.nolace = init_nolace(arrays);
    model.loaded = model.lace.is_some() || model.nolace.is_some();
    model.loaded
}

// ========== Feature Extraction ==========

/// Apply filterbank to spectral data.
///
/// Upstream C: dnn/osce_features.c:apply_filterbank
fn apply_filterbank(
    x_out: &mut [f32],
    x_in: &[f32],
    center_bins: &[usize],
    band_weights: &[f32],
    num_bands: usize,
) {
    x_out[0] = 0.0;
    for b in 0..num_bands - 1 {
        x_out[b + 1] = 0.0;
        for i in center_bins[b]..center_bins[b + 1] {
            let frac =
                (center_bins[b + 1] - i) as f32 / (center_bins[b + 1] - center_bins[b]) as f32;
            x_out[b] += band_weights[b] * frac * x_in[i];
            x_out[b + 1] += band_weights[b + 1] * (1.0 - frac) * x_in[i];
        }
    }
    x_out[num_bands - 1] += band_weights[num_bands - 1] * x_in[center_bins[num_bands - 1]];
}

/// Compute magnitude spectrum (one-sided) of 320-sample windowed signal.
///
/// Upstream C: dnn/osce_features.c:mag_spec_320_onesided
fn mag_spec_320_onesided(out: &mut [f32], input: &[f32]) {
    let mut buffer = [kiss_fft_cpx { re: 0.0, im: 0.0 }; OSCE_SPEC_WINDOW_SIZE];
    forward_transform(&mut buffer, input);
    for k in 0..OSCE_SPEC_NUM_FREQS {
        out[k] = OSCE_SPEC_WINDOW_SIZE as f32
            * (buffer[k].re * buffer[k].re + buffer[k].im * buffer[k].im).sqrt();
    }
}

/// Calculate log spectrum from LPC coefficients.
///
/// Upstream C: dnn/osce_features.c:calculate_log_spectrum_from_lpc
fn calculate_log_spectrum_from_lpc(spec: &mut [f32], a_q12: &[i16], lpc_order: usize) {
    let mut buffer = [0.0f32; OSCE_SPEC_WINDOW_SIZE];
    buffer[0] = 1.0;
    for i in 0..lpc_order {
        buffer[i + 1] = -(a_q12[i] as f32) / (1 << 12) as f32;
    }

    mag_spec_320_onesided(&mut buffer.clone(), &buffer);
    // Note: mag_spec writes to out; we use buffer for both
    let mut mag = [0.0f32; OSCE_SPEC_NUM_FREQS];
    mag_spec_320_onesided(&mut mag, &buffer);

    for i in 0..OSCE_SPEC_NUM_FREQS {
        mag[i] = 1.0 / (mag[i] + 1e-9);
    }

    let mut filtered = [0.0f32; OSCE_CLEAN_SPEC_NUM_BANDS];
    apply_filterbank(
        &mut filtered,
        &mag,
        &CENTER_BINS_CLEAN,
        &BAND_WEIGHTS_CLEAN,
        OSCE_CLEAN_SPEC_NUM_BANDS,
    );

    for i in 0..OSCE_CLEAN_SPEC_NUM_BANDS {
        spec[i] = 0.3 * (filtered[i] + 1e-9).ln();
    }
}

/// Calculate cepstrum from signal.
///
/// Upstream C: dnn/osce_features.c:calculate_cepstrum
fn calculate_cepstrum(cepstrum: &mut [f32], signal: &[f32]) {
    let osce_window = generate_osce_window();
    let mut buffer = [0.0f32; OSCE_SPEC_WINDOW_SIZE];
    for n in 0..OSCE_SPEC_WINDOW_SIZE {
        buffer[n] = osce_window[n] * signal[n];
    }

    let mut mag = [0.0f32; OSCE_SPEC_NUM_FREQS + 3 + OSCE_NOISY_SPEC_NUM_BANDS];
    mag_spec_320_onesided(&mut mag, &buffer);

    let mut spec = [0.0f32; OSCE_NOISY_SPEC_NUM_BANDS];
    apply_filterbank(
        &mut spec,
        &mag,
        &CENTER_BINS_NOISY,
        &BAND_WEIGHTS_NOISY,
        OSCE_NOISY_SPEC_NUM_BANDS,
    );

    for n in 0..OSCE_NOISY_SPEC_NUM_BANDS {
        spec[n] = (spec[n] + 1e-9).ln();
    }

    // DCT-II (orthonormal) — uses the same dct function from freq.rs
    assert_eq!(OSCE_NOISY_SPEC_NUM_BANDS, NB_BANDS);
    crate::dnn::freq::dct(cepstrum, &spec);
}

/// Calculate autocorrelation around pitch lag.
///
/// Upstream C: dnn/osce_features.c:calculate_acorr
fn calculate_acorr(acorr: &mut [f32], signal: &[f32], signal_offset: usize, lag: i32) {
    for k in -2i32..=2 {
        let mut xx = 0.0f32;
        let mut xy = 0.0f32;
        let mut yy = 0.0f32;
        for n in 0..80 {
            let x = signal[signal_offset + n];
            let y_idx = signal_offset as i32 + n as i32 - lag + k;
            let y = if y_idx >= 0 && (y_idx as usize) < signal.len() {
                signal[y_idx as usize]
            } else {
                0.0
            };
            xx += x * x;
            yy += y * y;
            xy += x * y;
        }
        acorr[(k + 2) as usize] = xy / (xx * yy + 1e-9).sqrt();
    }
}

/// Pitch postprocessing with hangover.
///
/// Upstream C: dnn/osce_features.c:pitch_postprocessing
fn pitch_postprocessing(features: &mut OSCEFeatureState, lag: i32, signal_type: i32) -> i32 {
    const TYPE_VOICED: i32 = 2;

    let new_lag;
    if signal_type != TYPE_VOICED {
        new_lag = OSCE_NO_PITCH_VALUE;
        features.pitch_hangover_count = 0;
    } else {
        new_lag = lag;
        features.last_lag = lag;
        features.pitch_hangover_count = 0;
    }
    features.last_type = signal_type;
    assert!(new_lag != 0);
    new_lag
}

/// Calculate OSCE features from decoded SILK frame.
///
/// `xq` is decoded speech (i16), `pred_coef_q12` is [num_subframes/2][lpc_order],
/// `pitch_l` is per-subframe pitch lags, `ltp_coef_q14` is [num_subframes * 5],
/// `gains_q16` is per-subframe gains.
///
/// Upstream C: dnn/osce_features.c:osce_calculate_features
#[allow(clippy::too_many_arguments)]
pub fn osce_calculate_features(
    osce_features: &mut OSCEFeatureState,
    num_subframes: usize,
    lpc_order: usize,
    signal_type: i32,
    pred_coef_q12: &[&[i16]],
    pitch_l: &[i32],
    ltp_coef_q14: &[i16],
    gains_q16: &[i32],
    xq: &[i16],
    num_bits: i32,
    features: &mut [f32],
    numbits: &mut [f32],
    periods: &mut [i32],
) {
    let num_samples = num_subframes * 80;

    // Smooth bit count
    osce_features.numbits_smooth = 0.9 * osce_features.numbits_smooth + 0.1 * num_bits as f32;
    numbits[0] = num_bits as f32;
    numbits[1] = osce_features.numbits_smooth;

    let mut buffer = vec![0.0f32; OSCE_FEATURES_MAX_HISTORY + num_samples];
    for n in 0..num_samples {
        buffer[OSCE_FEATURES_MAX_HISTORY + n] = xq[n] as f32 / (1 << 15) as f32;
    }
    buffer[..OSCE_FEATURES_MAX_HISTORY]
        .copy_from_slice(&osce_features.signal_history[..OSCE_FEATURES_MAX_HISTORY]);

    for k in 0..num_subframes {
        let base = k * OSCE_FEATURE_DIM;
        let frame_offset = OSCE_FEATURES_MAX_HISTORY + k * 80;
        features[base..base + OSCE_FEATURE_DIM].fill(0.0);

        // Clean spectrum from LPCs (update every other frame)
        if k % 2 == 0 {
            calculate_log_spectrum_from_lpc(
                &mut features[base + OSCE_CLEAN_SPEC_START..],
                pred_coef_q12[k >> 1],
                lpc_order,
            );
        } else {
            let prev_start = (k - 1) * OSCE_FEATURE_DIM + OSCE_CLEAN_SPEC_START;
            let dst_start = base + OSCE_CLEAN_SPEC_START;
            for i in 0..OSCE_CLEAN_SPEC_LENGTH {
                features[dst_start + i] = features[prev_start + i];
            }
        }

        // Noisy cepstrum from signal (update every other frame)
        if k % 2 == 0 {
            let sig_start = frame_offset - 160;
            calculate_cepstrum(
                &mut features[base + OSCE_NOISY_CEPSTRUM_START..],
                &buffer[sig_start..sig_start + OSCE_SPEC_WINDOW_SIZE],
            );
        } else {
            let prev_start = (k - 1) * OSCE_FEATURE_DIM + OSCE_NOISY_CEPSTRUM_START;
            let dst_start = base + OSCE_NOISY_CEPSTRUM_START;
            for i in 0..OSCE_NOISY_CEPSTRUM_LENGTH {
                features[dst_start + i] = features[prev_start + i];
            }
        }

        // Pitch postprocessing
        periods[k] = pitch_postprocessing(osce_features, pitch_l[k], signal_type);

        // Autocorrelation around pitch lag
        calculate_acorr(
            &mut features[base + OSCE_ACORR_START..],
            &buffer,
            frame_offset,
            periods[k],
        );

        // LTP coefficients
        for i in 0..OSCE_LTP_LENGTH {
            features[base + OSCE_LTP_START + i] =
                ltp_coef_q14[k * OSCE_LTP_LENGTH + i] as f32 / (1 << 14) as f32;
        }

        // Frame gain
        features[base + OSCE_LOG_GAIN_START] =
            (gains_q16[k] as f32 / (1u32 << 16) as f32 + 1e-9).ln();
    }

    // Buffer update
    osce_features.signal_history[..OSCE_FEATURES_MAX_HISTORY]
        .copy_from_slice(&buffer[num_samples..num_samples + OSCE_FEATURES_MAX_HISTORY]);
}

/// Cross-fade enhanced signal with original over 10ms.
///
/// Upstream C: dnn/osce_features.c:osce_cross_fade_10ms
pub fn osce_cross_fade_10ms(x_enhanced: &mut [f32], x_in: &[f32], length: usize) {
    assert!(length >= 160);
    let window = generate_osce_window();
    for i in 0..160 {
        x_enhanced[i] = window[i] * x_enhanced[i] + (1.0 - window[i]) * x_in[i];
    }
}

// ========== LACE Processing ==========

/// Compute numbits embedding (sinusoidal positional encoding).
///
/// Upstream C: dnn/osce.c:compute_lace_numbits_embedding
fn compute_numbits_embedding(
    emb: &mut [f32],
    numbits: f32,
    scales: &[f32; 8],
    min_val: f32,
    max_val: f32,
    logscale: bool,
) {
    let nb = if logscale { numbits.ln() } else { numbits };
    let x = nb.clamp(min_val, max_val) - (max_val + min_val) / 2.0;
    for i in 0..8 {
        emb[i] = (x * scales[i] - 0.5).sin();
    }
}

/// Run LACE feature network.
///
/// Upstream C: dnn/osce.c:lace_feature_net
fn lace_feature_net(
    lace: &LACE,
    state: &mut LACEState,
    output: &mut [f32],
    features: &[f32],
    numbits: &[f32],
    periods: &[i32],
) {
    let max_dim = LACE_COND_DIM.max(LACE_HIDDEN_FEATURE_DIM);
    let mut input_buffer = vec![0.0f32; 4 * max_dim];
    let mut output_buffer = vec![0.0f32; 4 * max_dim];
    let mut numbits_embedded = [0.0f32; 2 * LACE_NUMBITS_EMBEDDING_DIM];

    compute_numbits_embedding(
        &mut numbits_embedded[..LACE_NUMBITS_EMBEDDING_DIM],
        numbits[0],
        &LACE_NUMBITS_SCALES,
        LACE_NUMBITS_RANGE_LOW.ln(),
        LACE_NUMBITS_RANGE_HIGH.ln(),
        true,
    );
    compute_numbits_embedding(
        &mut numbits_embedded[LACE_NUMBITS_EMBEDDING_DIM..],
        numbits[1],
        &LACE_NUMBITS_SCALES,
        LACE_NUMBITS_RANGE_LOW.ln(),
        LACE_NUMBITS_RANGE_HIGH.ln(),
        true,
    );

    // Per-subframe conv1
    let input_size = LACE_NUM_FEATURES + LACE_PITCH_EMBEDDING_DIM + 2 * LACE_NUMBITS_EMBEDDING_DIM;
    for sf in 0..4 {
        input_buffer[..LACE_NUM_FEATURES]
            .copy_from_slice(&features[sf * LACE_NUM_FEATURES..(sf + 1) * LACE_NUM_FEATURES]);
        // Pitch embedding lookup
        let pitch_idx = periods[sf] as usize;
        let embed_start = pitch_idx * LACE_PITCH_EMBEDDING_DIM;
        let embed_end = embed_start + LACE_PITCH_EMBEDDING_DIM;
        if embed_end <= lace.layers.pitch_embedding.float_weights.len() {
            input_buffer[LACE_NUM_FEATURES..LACE_NUM_FEATURES + LACE_PITCH_EMBEDDING_DIM]
                .copy_from_slice(
                    &lace.layers.pitch_embedding.float_weights[embed_start..embed_end],
                );
        }
        input_buffer[LACE_NUM_FEATURES + LACE_PITCH_EMBEDDING_DIM
            ..LACE_NUM_FEATURES + LACE_PITCH_EMBEDDING_DIM + 2 * LACE_NUMBITS_EMBEDDING_DIM]
            .copy_from_slice(&numbits_embedded);

        compute_generic_conv1d(
            &lace.layers.fnet_conv1,
            &mut output_buffer[sf * LACE_HIDDEN_FEATURE_DIM..(sf + 1) * LACE_HIDDEN_FEATURE_DIM],
            &mut [], // NULL mem (no temporal state for conv1)
            &input_buffer[..input_size],
            input_size,
            ACTIVATION_TANH,
        );
    }

    // Subframe accumulation conv2
    input_buffer[..4 * LACE_HIDDEN_FEATURE_DIM]
        .copy_from_slice(&output_buffer[..4 * LACE_HIDDEN_FEATURE_DIM]);
    compute_generic_conv1d(
        &lace.layers.fnet_conv2,
        &mut output_buffer,
        &mut state.feature_net_conv2_state,
        &input_buffer,
        4 * LACE_HIDDEN_FEATURE_DIM,
        ACTIVATION_TANH,
    );

    // Tconv upsampling (dense)
    input_buffer[..4 * LACE_COND_DIM].copy_from_slice(&output_buffer[..4 * LACE_COND_DIM]);
    compute_generic_dense(
        &lace.layers.fnet_tconv,
        &mut output_buffer,
        &input_buffer,
        ACTIVATION_TANH,
    );

    // GRU per subframe
    input_buffer[..4 * LACE_COND_DIM].copy_from_slice(&output_buffer[..4 * LACE_COND_DIM]);
    for sf in 0..4 {
        compute_generic_gru(
            &lace.layers.fnet_gru_input,
            &lace.layers.fnet_gru_recurrent,
            &mut state.feature_net_gru_state,
            &input_buffer[sf * LACE_COND_DIM..(sf + 1) * LACE_COND_DIM],
        );
        output[sf * LACE_COND_DIM..(sf + 1) * LACE_COND_DIM]
            .copy_from_slice(&state.feature_net_gru_state);
    }
}

/// Process one 20ms LACE frame.
///
/// Upstream C: dnn/osce.c:lace_process_20ms_frame
pub fn lace_process_20ms_frame(
    lace: &LACE,
    state: &mut LACEState,
    x_out: &mut [f32],
    x_in: &[f32],
    features: &[f32],
    numbits: &[f32],
    periods: &[i32],
) {
    let mut feature_buffer = vec![0.0f32; 4 * LACE_COND_DIM];
    let mut output_buffer = vec![0.0f32; 4 * LACE_FRAME_SIZE];

    // Pre-emphasis
    for i in 0..4 * LACE_FRAME_SIZE {
        output_buffer[i] = x_in[i] - LACE_PREEMPH * state.preemph_mem;
        state.preemph_mem = x_in[i];
    }

    // Feature network
    lace_feature_net(lace, state, &mut feature_buffer, features, numbits, periods);

    // 1st comb filtering
    for sf in 0..4 {
        let buf_start = sf * LACE_FRAME_SIZE;
        let x_buf: Vec<f32> = output_buffer[buf_start..buf_start + LACE_FRAME_SIZE].to_vec();
        adacomb_process_frame(
            &mut state.cf1_state,
            &mut output_buffer[buf_start..],
            &x_buf,
            &feature_buffer[sf * LACE_COND_DIM..],
            &lace.layers.cf1_kernel,
            &lace.layers.cf1_gain,
            &lace.layers.cf1_global_gain,
            periods[sf],
            LACE_COND_DIM,
            LACE_FRAME_SIZE,
            LACE_OVERLAP_SIZE,
            LACE_CF1_KERNEL_SIZE,
            LACE_CF1_LEFT_PADDING,
            LACE_CF1_FILTER_GAIN_A,
            LACE_CF1_FILTER_GAIN_B,
            LACE_CF1_LOG_GAIN_LIMIT,
            &lace.window,
        );
    }

    // 2nd comb filtering
    for sf in 0..4 {
        let buf_start = sf * LACE_FRAME_SIZE;
        let x_buf: Vec<f32> = output_buffer[buf_start..buf_start + LACE_FRAME_SIZE].to_vec();
        adacomb_process_frame(
            &mut state.cf2_state,
            &mut output_buffer[buf_start..],
            &x_buf,
            &feature_buffer[sf * LACE_COND_DIM..],
            &lace.layers.cf2_kernel,
            &lace.layers.cf2_gain,
            &lace.layers.cf2_global_gain,
            periods[sf],
            LACE_COND_DIM,
            LACE_FRAME_SIZE,
            LACE_OVERLAP_SIZE,
            LACE_CF2_KERNEL_SIZE,
            LACE_CF2_LEFT_PADDING,
            LACE_CF2_FILTER_GAIN_A,
            LACE_CF2_FILTER_GAIN_B,
            LACE_CF2_LOG_GAIN_LIMIT,
            &lace.window,
        );
    }

    // Adaptive filtering
    for sf in 0..4 {
        let buf_start = sf * LACE_FRAME_SIZE;
        let x_buf: Vec<f32> = output_buffer[buf_start..buf_start + LACE_FRAME_SIZE].to_vec();
        adaconv_process_frame(
            &mut state.af1_state,
            &mut output_buffer[buf_start..],
            &x_buf,
            &feature_buffer[sf * LACE_COND_DIM..],
            &lace.layers.af1_kernel,
            &lace.layers.af1_gain,
            LACE_COND_DIM,
            LACE_FRAME_SIZE,
            LACE_OVERLAP_SIZE,
            LACE_AF1_IN_CHANNELS,
            LACE_AF1_OUT_CHANNELS,
            LACE_AF1_KERNEL_SIZE,
            LACE_AF1_LEFT_PADDING,
            LACE_AF1_FILTER_GAIN_A,
            LACE_AF1_FILTER_GAIN_B,
            LACE_AF1_SHAPE_GAIN,
            &lace.window,
        );
    }

    // De-emphasis
    for i in 0..4 * LACE_FRAME_SIZE {
        x_out[i] = output_buffer[i] + LACE_PREEMPH * state.deemph_mem;
        state.deemph_mem = x_out[i];
    }
}

// ========== NoLACE Processing ==========

/// Run NoLACE feature network.
///
/// Upstream C: dnn/osce.c:nolace_feature_net
fn nolace_feature_net(
    nolace: &NoLACE,
    state: &mut NoLACEState,
    output: &mut [f32],
    features: &[f32],
    numbits: &[f32],
    periods: &[i32],
) {
    let max_dim = NOLACE_COND_DIM.max(NOLACE_HIDDEN_FEATURE_DIM);
    let mut input_buffer = vec![0.0f32; 4 * max_dim];
    let mut output_buffer = vec![0.0f32; 4 * max_dim];
    let mut numbits_embedded = [0.0f32; 2 * NOLACE_NUMBITS_EMBEDDING_DIM];

    compute_numbits_embedding(
        &mut numbits_embedded[..NOLACE_NUMBITS_EMBEDDING_DIM],
        numbits[0],
        &NOLACE_NUMBITS_SCALES,
        NOLACE_NUMBITS_RANGE_LOW.ln(),
        NOLACE_NUMBITS_RANGE_HIGH.ln(),
        true,
    );
    compute_numbits_embedding(
        &mut numbits_embedded[NOLACE_NUMBITS_EMBEDDING_DIM..],
        numbits[1],
        &NOLACE_NUMBITS_SCALES,
        NOLACE_NUMBITS_RANGE_LOW.ln(),
        NOLACE_NUMBITS_RANGE_HIGH.ln(),
        true,
    );

    let input_size =
        NOLACE_NUM_FEATURES + NOLACE_PITCH_EMBEDDING_DIM + 2 * NOLACE_NUMBITS_EMBEDDING_DIM;
    for sf in 0..4 {
        input_buffer[..NOLACE_NUM_FEATURES]
            .copy_from_slice(&features[sf * NOLACE_NUM_FEATURES..(sf + 1) * NOLACE_NUM_FEATURES]);
        let pitch_idx = periods[sf] as usize;
        let embed_start = pitch_idx * NOLACE_PITCH_EMBEDDING_DIM;
        let embed_end = embed_start + NOLACE_PITCH_EMBEDDING_DIM;
        if embed_end <= nolace.layers.pitch_embedding.float_weights.len() {
            input_buffer[NOLACE_NUM_FEATURES..NOLACE_NUM_FEATURES + NOLACE_PITCH_EMBEDDING_DIM]
                .copy_from_slice(
                    &nolace.layers.pitch_embedding.float_weights[embed_start..embed_end],
                );
        }
        input_buffer[NOLACE_NUM_FEATURES + NOLACE_PITCH_EMBEDDING_DIM
            ..NOLACE_NUM_FEATURES + NOLACE_PITCH_EMBEDDING_DIM + 2 * NOLACE_NUMBITS_EMBEDDING_DIM]
            .copy_from_slice(&numbits_embedded);

        compute_generic_conv1d(
            &nolace.layers.fnet_conv1,
            &mut output_buffer
                [sf * NOLACE_HIDDEN_FEATURE_DIM..(sf + 1) * NOLACE_HIDDEN_FEATURE_DIM],
            &mut [],
            &input_buffer[..input_size],
            input_size,
            ACTIVATION_TANH,
        );
    }

    input_buffer[..4 * NOLACE_HIDDEN_FEATURE_DIM]
        .copy_from_slice(&output_buffer[..4 * NOLACE_HIDDEN_FEATURE_DIM]);
    compute_generic_conv1d(
        &nolace.layers.fnet_conv2,
        &mut output_buffer,
        &mut state.feature_net_conv2_state,
        &input_buffer,
        4 * NOLACE_HIDDEN_FEATURE_DIM,
        ACTIVATION_TANH,
    );

    input_buffer[..4 * NOLACE_COND_DIM].copy_from_slice(&output_buffer[..4 * NOLACE_COND_DIM]);
    compute_generic_dense(
        &nolace.layers.fnet_tconv,
        &mut output_buffer,
        &input_buffer,
        ACTIVATION_TANH,
    );

    input_buffer[..4 * NOLACE_COND_DIM].copy_from_slice(&output_buffer[..4 * NOLACE_COND_DIM]);
    for sf in 0..4 {
        compute_generic_gru(
            &nolace.layers.fnet_gru_input,
            &nolace.layers.fnet_gru_recurrent,
            &mut state.feature_net_gru_state,
            &input_buffer[sf * NOLACE_COND_DIM..(sf + 1) * NOLACE_COND_DIM],
        );
        output[sf * NOLACE_COND_DIM..(sf + 1) * NOLACE_COND_DIM]
            .copy_from_slice(&state.feature_net_gru_state);
    }
}

/// Process one 20ms NoLACE frame.
///
/// Upstream C: dnn/osce.c:nolace_process_20ms_frame
pub fn nolace_process_20ms_frame(
    nolace: &NoLACE,
    state: &mut NoLACEState,
    x_out: &mut [f32],
    x_in: &[f32],
    features: &[f32],
    numbits: &[f32],
    periods: &[i32],
) {
    let mut feature_buffer = vec![0.0f32; 4 * NOLACE_COND_DIM];
    let mut feature_transform_buffer = vec![0.0f32; 4 * NOLACE_COND_DIM];
    let mut x_buffer1 = vec![0.0f32; 8 * NOLACE_FRAME_SIZE];
    let mut x_buffer2 = vec![0.0f32; 8 * NOLACE_FRAME_SIZE];

    // Pre-emphasis
    for i in 0..4 * NOLACE_FRAME_SIZE {
        x_buffer1[i] = x_in[i] - NOLACE_PREEMPH * state.preemph_mem;
        state.preemph_mem = x_in[i];
    }

    // Feature network
    nolace_feature_net(
        nolace,
        state,
        &mut feature_buffer,
        features,
        numbits,
        periods,
    );

    // 1st comb filtering + post conv
    for sf in 0..4 {
        let buf_start = sf * NOLACE_FRAME_SIZE;
        let x_buf: Vec<f32> = x_buffer1[buf_start..buf_start + NOLACE_FRAME_SIZE].to_vec();
        adacomb_process_frame(
            &mut state.cf1_state,
            &mut x_buffer1[buf_start..],
            &x_buf,
            &feature_buffer[sf * NOLACE_COND_DIM..],
            &nolace.layers.cf1_kernel,
            &nolace.layers.cf1_gain,
            &nolace.layers.cf1_global_gain,
            periods[sf],
            NOLACE_COND_DIM,
            NOLACE_FRAME_SIZE,
            NOLACE_OVERLAP_SIZE,
            NOLACE_CF1_KERNEL_SIZE,
            NOLACE_CF1_LEFT_PADDING,
            NOLACE_CF1_FILTER_GAIN_A,
            NOLACE_CF1_FILTER_GAIN_B,
            NOLACE_CF1_LOG_GAIN_LIMIT,
            &nolace.window,
        );
        compute_generic_conv1d(
            &nolace.layers.post_cf1,
            &mut feature_transform_buffer[sf * NOLACE_COND_DIM..(sf + 1) * NOLACE_COND_DIM],
            &mut state.post_cf1_state,
            &feature_buffer[sf * NOLACE_COND_DIM..],
            NOLACE_COND_DIM,
            ACTIVATION_TANH,
        );
    }
    feature_buffer[..4 * NOLACE_COND_DIM]
        .copy_from_slice(&feature_transform_buffer[..4 * NOLACE_COND_DIM]);

    // 2nd comb filtering + post conv
    for sf in 0..4 {
        let buf_start = sf * NOLACE_FRAME_SIZE;
        let x_buf: Vec<f32> = x_buffer1[buf_start..buf_start + NOLACE_FRAME_SIZE].to_vec();
        adacomb_process_frame(
            &mut state.cf2_state,
            &mut x_buffer1[buf_start..],
            &x_buf,
            &feature_buffer[sf * NOLACE_COND_DIM..],
            &nolace.layers.cf2_kernel,
            &nolace.layers.cf2_gain,
            &nolace.layers.cf2_global_gain,
            periods[sf],
            NOLACE_COND_DIM,
            NOLACE_FRAME_SIZE,
            NOLACE_OVERLAP_SIZE,
            NOLACE_CF2_KERNEL_SIZE,
            NOLACE_CF2_LEFT_PADDING,
            NOLACE_CF2_FILTER_GAIN_A,
            NOLACE_CF2_FILTER_GAIN_B,
            NOLACE_CF2_LOG_GAIN_LIMIT,
            &nolace.window,
        );
        compute_generic_conv1d(
            &nolace.layers.post_cf2,
            &mut feature_transform_buffer[sf * NOLACE_COND_DIM..(sf + 1) * NOLACE_COND_DIM],
            &mut state.post_cf2_state,
            &feature_buffer[sf * NOLACE_COND_DIM..],
            NOLACE_COND_DIM,
            ACTIVATION_TANH,
        );
    }
    feature_buffer[..4 * NOLACE_COND_DIM]
        .copy_from_slice(&feature_transform_buffer[..4 * NOLACE_COND_DIM]);

    // AF1 (1→2 channels) + post conv
    for sf in 0..4 {
        let in_start = sf * NOLACE_FRAME_SIZE;
        let out_start = sf * NOLACE_FRAME_SIZE * NOLACE_AF1_OUT_CHANNELS;
        let x_buf: Vec<f32> = x_buffer1[in_start..in_start + NOLACE_FRAME_SIZE].to_vec();
        adaconv_process_frame(
            &mut state.af1_state,
            &mut x_buffer2[out_start..],
            &x_buf,
            &feature_buffer[sf * NOLACE_COND_DIM..],
            &nolace.layers.af1_kernel,
            &nolace.layers.af1_gain,
            NOLACE_COND_DIM,
            NOLACE_FRAME_SIZE,
            NOLACE_OVERLAP_SIZE,
            NOLACE_AF1_IN_CHANNELS,
            NOLACE_AF1_OUT_CHANNELS,
            NOLACE_AF1_KERNEL_SIZE,
            NOLACE_AF1_LEFT_PADDING,
            NOLACE_AF1_FILTER_GAIN_A,
            NOLACE_AF1_FILTER_GAIN_B,
            NOLACE_AF1_SHAPE_GAIN,
            &nolace.window,
        );
        compute_generic_conv1d(
            &nolace.layers.post_af1,
            &mut feature_transform_buffer[sf * NOLACE_COND_DIM..(sf + 1) * NOLACE_COND_DIM],
            &mut state.post_af1_state,
            &feature_buffer[sf * NOLACE_COND_DIM..],
            NOLACE_COND_DIM,
            ACTIVATION_TANH,
        );
    }
    feature_buffer[..4 * NOLACE_COND_DIM]
        .copy_from_slice(&feature_transform_buffer[..4 * NOLACE_COND_DIM]);

    // 1st shape-mix: TDShape1 on 2nd channel + AF2 (2→2)
    for sf in 0..4 {
        let ch2_start = sf * NOLACE_AF1_OUT_CHANNELS * NOLACE_FRAME_SIZE + NOLACE_FRAME_SIZE;
        let ch2_buf: Vec<f32> = x_buffer2[ch2_start..ch2_start + NOLACE_FRAME_SIZE].to_vec();
        adashape_process_frame(
            &mut state.tdshape1_state,
            &mut x_buffer2[ch2_start..],
            &ch2_buf,
            &feature_buffer[sf * NOLACE_COND_DIM..],
            &nolace.layers.tdshape1_alpha1_f,
            &nolace.layers.tdshape1_alpha1_t,
            &nolace.layers.tdshape1_alpha2,
            NOLACE_TDSHAPE1_FEATURE_DIM,
            NOLACE_TDSHAPE1_FRAME_SIZE,
            NOLACE_TDSHAPE1_AVG_POOL_K,
        );

        let in_start = sf * NOLACE_FRAME_SIZE * NOLACE_AF2_IN_CHANNELS;
        let out_start = sf * NOLACE_FRAME_SIZE * NOLACE_AF2_OUT_CHANNELS;
        let x_buf: Vec<f32> =
            x_buffer2[in_start..in_start + NOLACE_FRAME_SIZE * NOLACE_AF2_IN_CHANNELS].to_vec();
        adaconv_process_frame(
            &mut state.af2_state,
            &mut x_buffer1[out_start..],
            &x_buf,
            &feature_buffer[sf * NOLACE_COND_DIM..],
            &nolace.layers.af2_kernel,
            &nolace.layers.af2_gain,
            NOLACE_COND_DIM,
            NOLACE_FRAME_SIZE,
            NOLACE_OVERLAP_SIZE,
            NOLACE_AF2_IN_CHANNELS,
            NOLACE_AF2_OUT_CHANNELS,
            NOLACE_AF2_KERNEL_SIZE,
            NOLACE_AF2_LEFT_PADDING,
            NOLACE_AF2_FILTER_GAIN_A,
            NOLACE_AF2_FILTER_GAIN_B,
            NOLACE_AF2_SHAPE_GAIN,
            &nolace.window,
        );
        compute_generic_conv1d(
            &nolace.layers.post_af2,
            &mut feature_transform_buffer[sf * NOLACE_COND_DIM..(sf + 1) * NOLACE_COND_DIM],
            &mut state.post_af2_state,
            &feature_buffer[sf * NOLACE_COND_DIM..],
            NOLACE_COND_DIM,
            ACTIVATION_TANH,
        );
    }
    feature_buffer[..4 * NOLACE_COND_DIM]
        .copy_from_slice(&feature_transform_buffer[..4 * NOLACE_COND_DIM]);

    // 2nd shape-mix: TDShape2 on 2nd channel + AF3 (2→2)
    for sf in 0..4 {
        let ch2_start = sf * NOLACE_AF2_OUT_CHANNELS * NOLACE_FRAME_SIZE + NOLACE_FRAME_SIZE;
        let ch2_buf: Vec<f32> = x_buffer1[ch2_start..ch2_start + NOLACE_FRAME_SIZE].to_vec();
        adashape_process_frame(
            &mut state.tdshape2_state,
            &mut x_buffer1[ch2_start..],
            &ch2_buf,
            &feature_buffer[sf * NOLACE_COND_DIM..],
            &nolace.layers.tdshape2_alpha1_f,
            &nolace.layers.tdshape2_alpha1_t,
            &nolace.layers.tdshape2_alpha2,
            NOLACE_TDSHAPE2_FEATURE_DIM,
            NOLACE_TDSHAPE2_FRAME_SIZE,
            NOLACE_TDSHAPE2_AVG_POOL_K,
        );

        let in_start = sf * NOLACE_FRAME_SIZE * NOLACE_AF3_IN_CHANNELS;
        let out_start = sf * NOLACE_FRAME_SIZE * NOLACE_AF3_OUT_CHANNELS;
        let x_buf: Vec<f32> =
            x_buffer1[in_start..in_start + NOLACE_FRAME_SIZE * NOLACE_AF3_IN_CHANNELS].to_vec();
        adaconv_process_frame(
            &mut state.af3_state,
            &mut x_buffer2[out_start..],
            &x_buf,
            &feature_buffer[sf * NOLACE_COND_DIM..],
            &nolace.layers.af3_kernel,
            &nolace.layers.af3_gain,
            NOLACE_COND_DIM,
            NOLACE_FRAME_SIZE,
            NOLACE_OVERLAP_SIZE,
            NOLACE_AF3_IN_CHANNELS,
            NOLACE_AF3_OUT_CHANNELS,
            NOLACE_AF3_KERNEL_SIZE,
            NOLACE_AF3_LEFT_PADDING,
            NOLACE_AF3_FILTER_GAIN_A,
            NOLACE_AF3_FILTER_GAIN_B,
            NOLACE_AF3_SHAPE_GAIN,
            &nolace.window,
        );
        compute_generic_conv1d(
            &nolace.layers.post_af3,
            &mut feature_transform_buffer[sf * NOLACE_COND_DIM..(sf + 1) * NOLACE_COND_DIM],
            &mut state.post_af3_state,
            &feature_buffer[sf * NOLACE_COND_DIM..],
            NOLACE_COND_DIM,
            ACTIVATION_TANH,
        );
    }
    feature_buffer[..4 * NOLACE_COND_DIM]
        .copy_from_slice(&feature_transform_buffer[..4 * NOLACE_COND_DIM]);

    // 3rd shape-mix: TDShape3 on 2nd channel + AF4 (2→1)
    for sf in 0..4 {
        let ch2_start = sf * NOLACE_AF3_OUT_CHANNELS * NOLACE_FRAME_SIZE + NOLACE_FRAME_SIZE;
        let ch2_buf: Vec<f32> = x_buffer2[ch2_start..ch2_start + NOLACE_FRAME_SIZE].to_vec();
        adashape_process_frame(
            &mut state.tdshape3_state,
            &mut x_buffer2[ch2_start..],
            &ch2_buf,
            &feature_buffer[sf * NOLACE_COND_DIM..],
            &nolace.layers.tdshape3_alpha1_f,
            &nolace.layers.tdshape3_alpha1_t,
            &nolace.layers.tdshape3_alpha2,
            NOLACE_TDSHAPE3_FEATURE_DIM,
            NOLACE_TDSHAPE3_FRAME_SIZE,
            NOLACE_TDSHAPE3_AVG_POOL_K,
        );

        let in_start = sf * NOLACE_FRAME_SIZE * NOLACE_AF4_IN_CHANNELS;
        let out_start = sf * NOLACE_FRAME_SIZE * NOLACE_AF4_OUT_CHANNELS;
        let x_buf: Vec<f32> =
            x_buffer2[in_start..in_start + NOLACE_FRAME_SIZE * NOLACE_AF4_IN_CHANNELS].to_vec();
        adaconv_process_frame(
            &mut state.af4_state,
            &mut x_buffer1[out_start..],
            &x_buf,
            &feature_buffer[sf * NOLACE_COND_DIM..],
            &nolace.layers.af4_kernel,
            &nolace.layers.af4_gain,
            NOLACE_COND_DIM,
            NOLACE_FRAME_SIZE,
            NOLACE_OVERLAP_SIZE,
            NOLACE_AF4_IN_CHANNELS,
            NOLACE_AF4_OUT_CHANNELS,
            NOLACE_AF4_KERNEL_SIZE,
            NOLACE_AF4_LEFT_PADDING,
            NOLACE_AF4_FILTER_GAIN_A,
            NOLACE_AF4_FILTER_GAIN_B,
            NOLACE_AF4_SHAPE_GAIN,
            &nolace.window,
        );
    }

    // De-emphasis
    for i in 0..4 * NOLACE_FRAME_SIZE {
        x_out[i] = x_buffer1[i] + NOLACE_PREEMPH * state.deemph_mem;
        state.deemph_mem = x_out[i];
    }
}

// ========== Reset ==========

/// Reset OSCE state for given method.
///
/// Upstream C: dnn/osce.c:osce_reset
pub fn osce_reset(state: &mut OSCEState, method: i32) {
    state.features = OSCEFeatureState::default();
    match method {
        OSCE_METHOD_NONE => {}
        OSCE_METHOD_LACE => {
            state.lace_state = LACEState::default();
        }
        OSCE_METHOD_NOLACE => {
            state.nolace_state = NoLACEState::default();
        }
        _ => {}
    }
    state.method = method;
    state.features.reset = 2;
}
