//! OSCE: Opus Speech Coding Enhancement (LACE and NoLACE models).
//!
//! Post-processes SILK decoded frames using adaptive filtering networks.
//!
//! Upstream C: `dnn/osce.c`, `dnn/osce.h`, `dnn/osce_features.c`,
//! `dnn/osce_config.h`, `dnn/osce_structs.h`

use crate::arch::Arch;
use crate::dnn::freq::{forward_transform, NB_BANDS};
use crate::dnn::nndsp::*;
use crate::dnn::nnet::*;

use crate::celt::kiss_fft::kiss_fft_cpx;
use crate::silk::structs::{silk_decoder_control, silk_decoder_state};

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

// ========== OSCE Extended Modes (osce.h) ==========

pub const OSCE_MODE_SILK_ONLY: i32 = 1000;
pub const OSCE_MODE_HYBRID: i32 = 1001;
pub const OSCE_MODE_CELT_ONLY: i32 = 1002;
pub const OSCE_MODE_SILK_BBWE: i32 = 1003;

// ========== BWE Config (osce_config.h) ==========

pub const OSCE_BWE_MAX_INSTAFREQ_BIN: usize = 40;
pub const OSCE_BWE_HALF_WINDOW_SIZE: usize = 160;
pub const OSCE_BWE_WINDOW_SIZE: usize = 2 * OSCE_BWE_HALF_WINDOW_SIZE;
pub const OSCE_BWE_NUM_BANDS: usize = 32;
pub const OSCE_BWE_FEATURE_DIM: usize = 114;
pub const OSCE_BWE_OUTPUT_DELAY: usize = 21;

// ========== BBWENet Constants (bbwenet_data.h) ==========

pub const BBWENET_FEATURE_DIM: usize = 114;
pub const BBWENET_FRAME_SIZE16: usize = 80;
pub const BBWENET_COND_DIM: usize = 128;

pub const BBWENET_FNET_CONV1_OUT_SIZE: usize = 128;
pub const BBWENET_FNET_CONV1_IN_SIZE: usize = 114;
pub const BBWENET_FNET_CONV1_STATE_SIZE: usize = 114 * 2;
pub const BBWENET_FNET_CONV2_OUT_SIZE: usize = 128;
pub const BBWENET_FNET_CONV2_IN_SIZE: usize = 128;
pub const BBWENET_FNET_CONV2_STATE_SIZE: usize = 128 * 2;
pub const BBWENET_FNET_GRU_OUT_SIZE: usize = 128;
pub const BBWENET_FNET_GRU_STATE_SIZE: usize = 128;
pub const BBWENET_FNET_TCONV_KERNEL_SIZE: usize = 2;
pub const BBWENET_FNET_TCONV_STRIDE: usize = 2;
pub const BBWENET_FNET_TCONV_IN_CHANNELS: usize = 128;
pub const BBWENET_FNET_TCONV_OUT_CHANNELS: usize = 128;

pub const BBWENET_TDSHAPE1_FEATURE_DIM: usize = 128;
pub const BBWENET_TDSHAPE1_FRAME_SIZE: usize = 160;
pub const BBWENET_TDSHAPE1_AVG_POOL_K: usize = 8;
pub const BBWENET_TDSHAPE1_INNOVATE: usize = 0;
pub const BBWENET_TDSHAPE1_POOL_AFTER: usize = 0;
pub const BBWENET_TDSHAPE1_INTERPOLATE_K: usize = 2;

pub const BBWENET_TDSHAPE2_FEATURE_DIM: usize = 128;
pub const BBWENET_TDSHAPE2_FRAME_SIZE: usize = 240;
pub const BBWENET_TDSHAPE2_AVG_POOL_K: usize = 12;
pub const BBWENET_TDSHAPE2_INNOVATE: usize = 0;
pub const BBWENET_TDSHAPE2_POOL_AFTER: usize = 0;
pub const BBWENET_TDSHAPE2_INTERPOLATE_K: usize = 2;

pub const BBWENET_AF1_FILTER_GAIN_A: f32 = 1.381551;
pub const BBWENET_AF1_FILTER_GAIN_B: f32 = 0.0;
pub const BBWENET_AF1_SHAPE_GAIN: f32 = 1.0;
pub const BBWENET_AF1_KERNEL_SIZE: usize = 16;
pub const BBWENET_AF1_FRAME_SIZE: usize = 80;
pub const BBWENET_AF1_LEFT_PADDING: usize = 15;
pub const BBWENET_AF1_OVERLAP_SIZE: usize = 40;
pub const BBWENET_AF1_IN_CHANNELS: usize = 1;
pub const BBWENET_AF1_OUT_CHANNELS: usize = 3;
pub const BBWENET_AF1_NORM_P: usize = 2;
pub const BBWENET_AF1_FEATURE_DIM: usize = 128;

pub const BBWENET_AF2_FILTER_GAIN_A: f32 = 1.381551;
pub const BBWENET_AF2_FILTER_GAIN_B: f32 = 0.0;
pub const BBWENET_AF2_SHAPE_GAIN: f32 = 1.0;
pub const BBWENET_AF2_KERNEL_SIZE: usize = 32;
pub const BBWENET_AF2_FRAME_SIZE: usize = 160;
pub const BBWENET_AF2_LEFT_PADDING: usize = 31;
pub const BBWENET_AF2_OVERLAP_SIZE: usize = 80;
pub const BBWENET_AF2_IN_CHANNELS: usize = 3;
pub const BBWENET_AF2_OUT_CHANNELS: usize = 3;
pub const BBWENET_AF2_NORM_P: usize = 2;
pub const BBWENET_AF2_FEATURE_DIM: usize = 128;

pub const BBWENET_AF3_FILTER_GAIN_A: f32 = 1.381551;
pub const BBWENET_AF3_FILTER_GAIN_B: f32 = 0.0;
pub const BBWENET_AF3_SHAPE_GAIN: f32 = 1.0;
pub const BBWENET_AF3_KERNEL_SIZE: usize = 16;
pub const BBWENET_AF3_FRAME_SIZE: usize = 240;
pub const BBWENET_AF3_LEFT_PADDING: usize = 15;
pub const BBWENET_AF3_OVERLAP_SIZE: usize = 120;
pub const BBWENET_AF3_IN_CHANNELS: usize = 3;
pub const BBWENET_AF3_OUT_CHANNELS: usize = 1;
pub const BBWENET_AF3_NORM_P: usize = 2;
pub const BBWENET_AF3_FEATURE_DIM: usize = 128;

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

/// Precomputed 320-sample sine window, matching the static table in osce_features.c.
///
/// Values are `sin(PI * (i + 0.5) / 320)` for i in 0..160, mirrored for 160..320.
/// Stored as hex literals for bit-exact reproducibility across platforms (different
/// libm implementations of `sin()` can differ by 1 ULP).
#[rustfmt::skip]
const OSCE_WINDOW: [f32; 320] = [
    f32::from_bits(0x3ba0d951), f32::from_bits(0x3c7143fe), f32::from_bits(0x3cc90ab0), f32::from_bits(0x3d0cb735), f32::from_bits(0x3d34e59a),
    f32::from_bits(0x3d5d0f88), f32::from_bits(0x3d829a01), f32::from_bits(0x3d96a905), f32::from_bits(0x3daab451), f32::from_bits(0x3dbebb67),
    f32::from_bits(0x3dd2bdc8), f32::from_bits(0x3de6baf6), f32::from_bits(0x3dfab273), f32::from_bits(0x3e0751e0), f32::from_bits(0x3e114730),
    f32::from_bits(0x3e1b38ea), f32::from_bits(0x3e2526d0), f32::from_bits(0x3e2f10a2), f32::from_bits(0x3e38f623), f32::from_bits(0x3e42d713),
    f32::from_bits(0x3e4cb335), f32::from_bits(0x3e568a4a), f32::from_bits(0x3e605c13), f32::from_bits(0x3e6a2854), f32::from_bits(0x3e73eecd),
    f32::from_bits(0x3e7daf42), f32::from_bits(0x3e83b4ba), f32::from_bits(0x3e888e93), f32::from_bits(0x3e8d650e), f32::from_bits(0x3e92380b),
    f32::from_bits(0x3e97076d), f32::from_bits(0x3e9bd315), f32::from_bits(0x3ea09ae5), f32::from_bits(0x3ea55ebe), f32::from_bits(0x3eaa1e82),
    f32::from_bits(0x3eaeda15), f32::from_bits(0x3eb39156), f32::from_bits(0x3eb8442a), f32::from_bits(0x3ebcf271), f32::from_bits(0x3ec19c0f),
    f32::from_bits(0x3ec640e6), f32::from_bits(0x3ecae0d9), f32::from_bits(0x3ecf7bca), f32::from_bits(0x3ed4119d), f32::from_bits(0x3ed8a234),
    f32::from_bits(0x3edd2d73), f32::from_bits(0x3ee1b33d), f32::from_bits(0x3ee63375), f32::from_bits(0x3eeaadff), f32::from_bits(0x3eef22bf),
    f32::from_bits(0x3ef39198), f32::from_bits(0x3ef7fa6f), f32::from_bits(0x3efc5d27), f32::from_bits(0x3f005cd3), f32::from_bits(0x3f0287e7),
    f32::from_bits(0x3f04afc3), f32::from_bits(0x3f06d459), f32::from_bits(0x3f08f59b), f32::from_bits(0x3f0b137c), f32::from_bits(0x3f0d2dee),
    f32::from_bits(0x3f0f44e5), f32::from_bits(0x3f115853), f32::from_bits(0x3f13682a), f32::from_bits(0x3f15745f), f32::from_bits(0x3f177ce4),
    f32::from_bits(0x3f1981ab), f32::from_bits(0x3f1b82a9), f32::from_bits(0x3f1d7fd1), f32::from_bits(0x3f1f7916), f32::from_bits(0x3f216e6c),
    f32::from_bits(0x3f235fc6), f32::from_bits(0x3f254d18), f32::from_bits(0x3f273656), f32::from_bits(0x3f291b74), f32::from_bits(0x3f2afc65),
    f32::from_bits(0x3f2cd91f), f32::from_bits(0x3f2eb194), f32::from_bits(0x3f3085bb), f32::from_bits(0x3f325586), f32::from_bits(0x3f3420eb),
    f32::from_bits(0x3f35e7de), f32::from_bits(0x3f37aa54), f32::from_bits(0x3f396842), f32::from_bits(0x3f3b219d), f32::from_bits(0x3f3cd659),
    f32::from_bits(0x3f3e866d), f32::from_bits(0x3f4031ce), f32::from_bits(0x3f41d870), f32::from_bits(0x3f437a4a), f32::from_bits(0x3f451752),
    f32::from_bits(0x3f46af7c), f32::from_bits(0x3f4842c0), f32::from_bits(0x3f49d112), f32::from_bits(0x3f4b5a6a), f32::from_bits(0x3f4cdebd),
    f32::from_bits(0x3f4e5e02), f32::from_bits(0x3f4fd830), f32::from_bits(0x3f514d3d), f32::from_bits(0x3f52bd20), f32::from_bits(0x3f5427cf),
    f32::from_bits(0x3f558d43), f32::from_bits(0x3f56ed72), f32::from_bits(0x3f584853), f32::from_bits(0x3f599dde), f32::from_bits(0x3f5aee0a),
    f32::from_bits(0x3f5c38d0), f32::from_bits(0x3f5d7e26), f32::from_bits(0x3f5ebe05), f32::from_bits(0x3f5ff866), f32::from_bits(0x3f612d40),
    f32::from_bits(0x3f625c8b), f32::from_bits(0x3f638641), f32::from_bits(0x3f64aa59), f32::from_bits(0x3f65c8cd), f32::from_bits(0x3f66e196),
    f32::from_bits(0x3f67f4ac), f32::from_bits(0x3f690209), f32::from_bits(0x3f6a09a7), f32::from_bits(0x3f6b0b7e), f32::from_bits(0x3f6c0788),
    f32::from_bits(0x3f6cfdbf), f32::from_bits(0x3f6dee1e), f32::from_bits(0x3f6ed89e), f32::from_bits(0x3f6fbd39), f32::from_bits(0x3f709be9),
    f32::from_bits(0x3f7174aa), f32::from_bits(0x3f724776), f32::from_bits(0x3f731447), f32::from_bits(0x3f73db19), f32::from_bits(0x3f749be7),
    f32::from_bits(0x3f7556ac), f32::from_bits(0x3f760b62), f32::from_bits(0x3f76ba07), f32::from_bits(0x3f776296), f32::from_bits(0x3f780509),
    f32::from_bits(0x3f78a15e), f32::from_bits(0x3f793791), f32::from_bits(0x3f79c79d), f32::from_bits(0x3f7a5180), f32::from_bits(0x3f7ad536),
    f32::from_bits(0x3f7b52bb), f32::from_bits(0x3f7bca0d), f32::from_bits(0x3f7c3b28), f32::from_bits(0x3f7ca60a), f32::from_bits(0x3f7d0ab0),
    f32::from_bits(0x3f7d6918), f32::from_bits(0x3f7dc13f), f32::from_bits(0x3f7e1324), f32::from_bits(0x3f7e5ec3), f32::from_bits(0x3f7ea41c),
    f32::from_bits(0x3f7ee32c), f32::from_bits(0x3f7f1bf2), f32::from_bits(0x3f7f4e6d), f32::from_bits(0x3f7f7a9c), f32::from_bits(0x3f7fa07c),
    f32::from_bits(0x3f7fc00e), f32::from_bits(0x3f7fd951), f32::from_bits(0x3f7fec43), f32::from_bits(0x3f7ff8e5), f32::from_bits(0x3f7fff36),
    f32::from_bits(0x3f7fff36), f32::from_bits(0x3f7ff8e5), f32::from_bits(0x3f7fec43), f32::from_bits(0x3f7fd951), f32::from_bits(0x3f7fc00e),
    f32::from_bits(0x3f7fa07c), f32::from_bits(0x3f7f7a9c), f32::from_bits(0x3f7f4e6d), f32::from_bits(0x3f7f1bf2), f32::from_bits(0x3f7ee32c),
    f32::from_bits(0x3f7ea41c), f32::from_bits(0x3f7e5ec3), f32::from_bits(0x3f7e1324), f32::from_bits(0x3f7dc13f), f32::from_bits(0x3f7d6918),
    f32::from_bits(0x3f7d0ab0), f32::from_bits(0x3f7ca60a), f32::from_bits(0x3f7c3b28), f32::from_bits(0x3f7bca0d), f32::from_bits(0x3f7b52bb),
    f32::from_bits(0x3f7ad536), f32::from_bits(0x3f7a5180), f32::from_bits(0x3f79c79d), f32::from_bits(0x3f793791), f32::from_bits(0x3f78a15e),
    f32::from_bits(0x3f780509), f32::from_bits(0x3f776296), f32::from_bits(0x3f76ba07), f32::from_bits(0x3f760b62), f32::from_bits(0x3f7556ac),
    f32::from_bits(0x3f749be7), f32::from_bits(0x3f73db19), f32::from_bits(0x3f731447), f32::from_bits(0x3f724776), f32::from_bits(0x3f7174aa),
    f32::from_bits(0x3f709be9), f32::from_bits(0x3f6fbd39), f32::from_bits(0x3f6ed89e), f32::from_bits(0x3f6dee1e), f32::from_bits(0x3f6cfdbf),
    f32::from_bits(0x3f6c0788), f32::from_bits(0x3f6b0b7e), f32::from_bits(0x3f6a09a7), f32::from_bits(0x3f690209), f32::from_bits(0x3f67f4ac),
    f32::from_bits(0x3f66e196), f32::from_bits(0x3f65c8cd), f32::from_bits(0x3f64aa59), f32::from_bits(0x3f638641), f32::from_bits(0x3f625c8b),
    f32::from_bits(0x3f612d40), f32::from_bits(0x3f5ff866), f32::from_bits(0x3f5ebe05), f32::from_bits(0x3f5d7e26), f32::from_bits(0x3f5c38d0),
    f32::from_bits(0x3f5aee0a), f32::from_bits(0x3f599dde), f32::from_bits(0x3f584853), f32::from_bits(0x3f56ed72), f32::from_bits(0x3f558d43),
    f32::from_bits(0x3f5427cf), f32::from_bits(0x3f52bd20), f32::from_bits(0x3f514d3d), f32::from_bits(0x3f4fd830), f32::from_bits(0x3f4e5e02),
    f32::from_bits(0x3f4cdebd), f32::from_bits(0x3f4b5a6a), f32::from_bits(0x3f49d112), f32::from_bits(0x3f4842c0), f32::from_bits(0x3f46af7c),
    f32::from_bits(0x3f451752), f32::from_bits(0x3f437a4a), f32::from_bits(0x3f41d870), f32::from_bits(0x3f4031ce), f32::from_bits(0x3f3e866d),
    f32::from_bits(0x3f3cd659), f32::from_bits(0x3f3b219d), f32::from_bits(0x3f396842), f32::from_bits(0x3f37aa54), f32::from_bits(0x3f35e7de),
    f32::from_bits(0x3f3420eb), f32::from_bits(0x3f325586), f32::from_bits(0x3f3085bb), f32::from_bits(0x3f2eb194), f32::from_bits(0x3f2cd91f),
    f32::from_bits(0x3f2afc65), f32::from_bits(0x3f291b74), f32::from_bits(0x3f273656), f32::from_bits(0x3f254d18), f32::from_bits(0x3f235fc6),
    f32::from_bits(0x3f216e6c), f32::from_bits(0x3f1f7916), f32::from_bits(0x3f1d7fd1), f32::from_bits(0x3f1b82a9), f32::from_bits(0x3f1981ab),
    f32::from_bits(0x3f177ce4), f32::from_bits(0x3f15745f), f32::from_bits(0x3f13682a), f32::from_bits(0x3f115853), f32::from_bits(0x3f0f44e5),
    f32::from_bits(0x3f0d2dee), f32::from_bits(0x3f0b137c), f32::from_bits(0x3f08f59b), f32::from_bits(0x3f06d459), f32::from_bits(0x3f04afc3),
    f32::from_bits(0x3f0287e7), f32::from_bits(0x3f005cd3), f32::from_bits(0x3efc5d27), f32::from_bits(0x3ef7fa6f), f32::from_bits(0x3ef39198),
    f32::from_bits(0x3eef22bf), f32::from_bits(0x3eeaadff), f32::from_bits(0x3ee63375), f32::from_bits(0x3ee1b33d), f32::from_bits(0x3edd2d73),
    f32::from_bits(0x3ed8a234), f32::from_bits(0x3ed4119d), f32::from_bits(0x3ecf7bca), f32::from_bits(0x3ecae0d9), f32::from_bits(0x3ec640e6),
    f32::from_bits(0x3ec19c0f), f32::from_bits(0x3ebcf271), f32::from_bits(0x3eb8442a), f32::from_bits(0x3eb39156), f32::from_bits(0x3eaeda15),
    f32::from_bits(0x3eaa1e82), f32::from_bits(0x3ea55ebe), f32::from_bits(0x3ea09ae5), f32::from_bits(0x3e9bd315), f32::from_bits(0x3e97076d),
    f32::from_bits(0x3e92380b), f32::from_bits(0x3e8d650e), f32::from_bits(0x3e888e93), f32::from_bits(0x3e83b4ba), f32::from_bits(0x3e7daf42),
    f32::from_bits(0x3e73eecd), f32::from_bits(0x3e6a2854), f32::from_bits(0x3e605c13), f32::from_bits(0x3e568a4a), f32::from_bits(0x3e4cb335),
    f32::from_bits(0x3e42d713), f32::from_bits(0x3e38f623), f32::from_bits(0x3e2f10a2), f32::from_bits(0x3e2526d0), f32::from_bits(0x3e1b38ea),
    f32::from_bits(0x3e114730), f32::from_bits(0x3e0751e0), f32::from_bits(0x3dfab273), f32::from_bits(0x3de6baf6), f32::from_bits(0x3dd2bdc8),
    f32::from_bits(0x3dbebb67), f32::from_bits(0x3daab451), f32::from_bits(0x3d96a905), f32::from_bits(0x3d829a01), f32::from_bits(0x3d5d0f88),
    f32::from_bits(0x3d34e59a), f32::from_bits(0x3d0cb735), f32::from_bits(0x3cc90ab0), f32::from_bits(0x3c7143fe), f32::from_bits(0x3ba0d951),
];

fn generate_osce_window() -> &'static [f32; OSCE_SPEC_WINDOW_SIZE] {
    &OSCE_WINDOW
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
#[derive(Clone)]
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
#[derive(Clone)]
pub struct LACE {
    pub layers: LACELayers,
    pub window: Vec<f32>,
}

/// LACE runtime state.
///
/// Upstream C: dnn/osce_structs.h:LACEState
#[derive(Clone)]
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
#[derive(Clone)]
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
#[derive(Clone)]
pub struct NoLACE {
    pub layers: NOLACELayers,
    pub window: Vec<f32>,
}

/// NoLACE runtime state.
///
/// Upstream C: dnn/osce_structs.h:NoLACEState
#[derive(Clone)]
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

// ========== BBWENet Structs ==========

/// Resampler state for 2x upsampling and 3:2 interpolation.
///
/// Upstream C: dnn/osce_structs.h:resamp_state
#[derive(Clone)]
pub struct ResampState {
    pub upsamp_buffer: [[f32; 3]; 2],
    pub interpol_buffer: [f32; 8],
}

impl Default for ResampState {
    fn default() -> Self {
        ResampState {
            upsamp_buffer: [[0.0; 3]; 2],
            interpol_buffer: [0.0; 8],
        }
    }
}

/// BBWENet model layers.
///
/// Upstream C: dnn/bbwenet_data.h:BBWENETLayers
#[derive(Clone)]
pub struct BBWENETLayers {
    pub fnet_conv1: LinearLayer,
    pub fnet_conv2: LinearLayer,
    pub fnet_gru_input: LinearLayer,
    pub fnet_gru_recurrent: LinearLayer,
    pub fnet_tconv: LinearLayer,
    pub tdshape1_alpha1_f: LinearLayer,
    pub tdshape1_alpha1_t: LinearLayer,
    pub tdshape1_alpha2: LinearLayer,
    pub tdshape2_alpha1_f: LinearLayer,
    pub tdshape2_alpha1_t: LinearLayer,
    pub tdshape2_alpha2: LinearLayer,
    pub af1_kernel: LinearLayer,
    pub af1_gain: LinearLayer,
    pub af2_kernel: LinearLayer,
    pub af2_gain: LinearLayer,
    pub af3_kernel: LinearLayer,
    pub af3_gain: LinearLayer,
}

/// BBWENet runtime state.
///
/// Upstream C: dnn/osce_structs.h:BBWENetState
#[derive(Clone)]
pub struct BBWENetState {
    pub feature_net_conv1_state: Vec<f32>,
    pub feature_net_conv2_state: Vec<f32>,
    pub feature_net_gru_state: Vec<f32>,
    pub output_buffer: Vec<i16>,
    pub af1_state: AdaConvState,
    pub af2_state: AdaConvState,
    pub af3_state: AdaConvState,
    pub tdshape1_state: AdaShapeState,
    pub tdshape2_state: AdaShapeState,
    pub resampler_state: [ResampState; 3],
}

impl Default for BBWENetState {
    fn default() -> Self {
        BBWENetState {
            feature_net_conv1_state: vec![0.0; BBWENET_FNET_CONV1_STATE_SIZE],
            feature_net_conv2_state: vec![0.0; BBWENET_FNET_CONV2_STATE_SIZE],
            feature_net_gru_state: vec![0.0; BBWENET_FNET_GRU_STATE_SIZE],
            output_buffer: vec![0; OSCE_BWE_OUTPUT_DELAY],
            af1_state: AdaConvState::default(),
            af2_state: AdaConvState::default(),
            af3_state: AdaConvState::default(),
            tdshape1_state: AdaShapeState::default(),
            tdshape2_state: AdaShapeState::default(),
            resampler_state: [
                ResampState::default(),
                ResampState::default(),
                ResampState::default(),
            ],
        }
    }
}

/// BBWENet model (layers + overlap windows at 16/32/48 kHz).
///
/// Upstream C: dnn/osce_structs.h:BBWENet
#[derive(Clone)]
pub struct BBWENet {
    pub layers: BBWENETLayers,
    pub window16: Vec<f32>,
    pub window32: Vec<f32>,
    pub window48: Vec<f32>,
}

/// BWE feature extraction state.
///
/// Upstream C: dnn/osce_structs.h:OSCEBWEFeatureState
#[derive(Clone)]
pub struct OSCEBWEFeatureState {
    pub signal_history: Vec<f32>,
    pub last_spec: Vec<f32>,
}

impl Default for OSCEBWEFeatureState {
    fn default() -> Self {
        OSCEBWEFeatureState {
            signal_history: vec![0.0; OSCE_BWE_HALF_WINDOW_SIZE],
            last_spec: vec![0.0; 2 * OSCE_BWE_MAX_INSTAFREQ_BIN + 2],
        }
    }
}

/// BWE state container (wraps BBWENetState).
///
/// Upstream C: dnn/osce_structs.h:OSCEBWEState
#[derive(Clone, Default)]
pub struct OSCEBWEState {
    pub bbwenet: BBWENetState,
}

/// Top-level OSCE model container.
///
/// Upstream C: dnn/osce_structs.h:OSCEModel
#[derive(Clone, Default)]
pub struct OSCEModel {
    pub loaded: bool,
    pub lace: Option<LACE>,
    pub nolace: Option<NoLACE>,
    pub bbwenet: Option<BBWENet>,
}

/// Combined OSCE state (features + method-specific runtime state).
///
/// Upstream C: silk/structs.h:silk_OSCE_struct
#[derive(Clone)]
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

/// Initialize BBWENet model from weight arrays.
///
/// Upstream C: dnn/bbwenet_data.c:init_bbwenetlayers
pub fn init_bbwenet(arrays: &[WeightArray]) -> Option<BBWENet> {
    let layers = BBWENETLayers {
        fnet_conv1: linear_init(
            arrays,
            "bbwenet_fnet_conv1_bias",
            "",
            "",
            "bbwenet_fnet_conv1_weights_float",
            "",
            "",
            "",
            342,
            128,
        )?,
        fnet_conv2: linear_init(
            arrays,
            "bbwenet_fnet_conv2_bias",
            "bbwenet_fnet_conv2_subias",
            "bbwenet_fnet_conv2_weights_int8",
            "bbwenet_fnet_conv2_weights_float",
            "",
            "",
            "bbwenet_fnet_conv2_scale",
            384,
            128,
        )?,
        fnet_gru_input: linear_init(
            arrays,
            "bbwenet_fnet_gru_input_bias",
            "bbwenet_fnet_gru_input_subias",
            "bbwenet_fnet_gru_input_weights_int8",
            "bbwenet_fnet_gru_input_weights_float",
            "",
            "",
            "bbwenet_fnet_gru_input_scale",
            128,
            384,
        )?,
        fnet_gru_recurrent: linear_init(
            arrays,
            "bbwenet_fnet_gru_recurrent_bias",
            "bbwenet_fnet_gru_recurrent_subias",
            "bbwenet_fnet_gru_recurrent_weights_int8",
            "bbwenet_fnet_gru_recurrent_weights_float",
            "",
            "",
            "bbwenet_fnet_gru_recurrent_scale",
            128,
            384,
        )?,
        fnet_tconv: linear_init(
            arrays,
            "bbwenet_fnet_tconv_bias",
            "bbwenet_fnet_tconv_subias",
            "bbwenet_fnet_tconv_weights_int8",
            "bbwenet_fnet_tconv_weights_float",
            "",
            "",
            "bbwenet_fnet_tconv_scale",
            128,
            256,
        )?,
        tdshape1_alpha1_f: linear_init(
            arrays,
            "bbwenet_tdshape1_alpha1_f_bias",
            "bbwenet_tdshape1_alpha1_f_subias",
            "bbwenet_tdshape1_alpha1_f_weights_int8",
            "bbwenet_tdshape1_alpha1_f_weights_float",
            "",
            "",
            "bbwenet_tdshape1_alpha1_f_scale",
            256,
            80,
        )?,
        tdshape1_alpha1_t: linear_init(
            arrays,
            "bbwenet_tdshape1_alpha1_t_bias",
            "",
            "",
            "bbwenet_tdshape1_alpha1_t_weights_float",
            "",
            "",
            "",
            42,
            80,
        )?,
        tdshape1_alpha2: linear_init(
            arrays,
            "bbwenet_tdshape1_alpha2_bias",
            "",
            "",
            "bbwenet_tdshape1_alpha2_weights_float",
            "",
            "",
            "",
            160,
            80,
        )?,
        tdshape2_alpha1_f: linear_init(
            arrays,
            "bbwenet_tdshape2_alpha1_f_bias",
            "bbwenet_tdshape2_alpha1_f_subias",
            "bbwenet_tdshape2_alpha1_f_weights_int8",
            "bbwenet_tdshape2_alpha1_f_weights_float",
            "",
            "",
            "bbwenet_tdshape2_alpha1_f_scale",
            256,
            120,
        )?,
        tdshape2_alpha1_t: linear_init(
            arrays,
            "bbwenet_tdshape2_alpha1_t_bias",
            "",
            "",
            "bbwenet_tdshape2_alpha1_t_weights_float",
            "",
            "",
            "",
            42,
            120,
        )?,
        tdshape2_alpha2: linear_init(
            arrays,
            "bbwenet_tdshape2_alpha2_bias",
            "",
            "",
            "bbwenet_tdshape2_alpha2_weights_float",
            "",
            "",
            "",
            240,
            120,
        )?,
        af1_kernel: linear_init(
            arrays,
            "bbwenet_af1_kernel_bias",
            "bbwenet_af1_kernel_subias",
            "bbwenet_af1_kernel_weights_int8",
            "bbwenet_af1_kernel_weights_float",
            "",
            "",
            "bbwenet_af1_kernel_scale",
            128,
            48,
        )?,
        af1_gain: linear_init(
            arrays,
            "bbwenet_af1_gain_bias",
            "",
            "",
            "bbwenet_af1_gain_weights_float",
            "",
            "",
            "",
            128,
            3,
        )?,
        af2_kernel: linear_init(
            arrays,
            "bbwenet_af2_kernel_bias",
            "bbwenet_af2_kernel_subias",
            "bbwenet_af2_kernel_weights_int8",
            "bbwenet_af2_kernel_weights_float",
            "",
            "",
            "bbwenet_af2_kernel_scale",
            128,
            288,
        )?,
        af2_gain: linear_init(
            arrays,
            "bbwenet_af2_gain_bias",
            "",
            "",
            "bbwenet_af2_gain_weights_float",
            "",
            "",
            "",
            128,
            3,
        )?,
        af3_kernel: linear_init(
            arrays,
            "bbwenet_af3_kernel_bias",
            "bbwenet_af3_kernel_subias",
            "bbwenet_af3_kernel_weights_int8",
            "bbwenet_af3_kernel_weights_float",
            "",
            "",
            "bbwenet_af3_kernel_scale",
            128,
            48,
        )?,
        af3_gain: linear_init(
            arrays,
            "bbwenet_af3_gain_bias",
            "",
            "",
            "bbwenet_af3_gain_weights_float",
            "",
            "",
            "",
            128,
            1,
        )?,
    };
    let mut window16 = vec![0.0f32; BBWENET_AF1_OVERLAP_SIZE];
    compute_overlap_window(&mut window16, BBWENET_AF1_OVERLAP_SIZE);
    let mut window32 = vec![0.0f32; BBWENET_AF2_OVERLAP_SIZE];
    compute_overlap_window(&mut window32, BBWENET_AF2_OVERLAP_SIZE);
    let mut window48 = vec![0.0f32; BBWENET_AF3_OVERLAP_SIZE];
    compute_overlap_window(&mut window48, BBWENET_AF3_OVERLAP_SIZE);
    Some(BBWENet {
        layers,
        window16,
        window32,
        window48,
    })
}

/// Load OSCE models from weight data.
///
/// Upstream C: dnn/osce.c:osce_load_models
pub fn osce_load_models(model: &mut OSCEModel, arrays: &[WeightArray]) -> bool {
    model.lace = init_lace(arrays);
    model.nolace = init_nolace(arrays);
    model.bbwenet = init_bbwenet(arrays);
    model.loaded = model.lace.is_some() || model.nolace.is_some() || model.bbwenet.is_some();
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
        // C: OSCE_SPEC_WINDOW_SIZE * sqrt(re*re + im*im) — entire expression in double
        let mag_sq = buffer[k].re * buffer[k].re + buffer[k].im * buffer[k].im;
        out[k] = (OSCE_SPEC_WINDOW_SIZE as f64 * (mag_sq as f64).sqrt()) as f32;
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

    // C: mag_spec_320_onesided(buffer, buffer) — in-place
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
        // C: 0.3f * log(spec[i] + 1e-9f) — 0.3f promoted to double, entire expr in double
        // black_box prevents LLVM auto-vectorization of ln() (see nndsp::compute_overlap_window).
        let val = (filtered[i] + 1e-9) as f64;
        spec[i] = (0.3f32 as f64 * std::hint::black_box(val).ln()) as f32;
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
        // C: log(spec[n] + 1e-9f) — log() is double precision
        // black_box prevents LLVM auto-vectorization of ln() (see nndsp::compute_overlap_window).
        let val = (spec[n] + 1e-9) as f64;
        spec[n] = std::hint::black_box(val).ln() as f32;
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
        // C: xy / sqrt(xx * yy + 1e-9f) — xy promoted to double, entire expr in double
        acorr[(k + 2) as usize] = (xy as f64 / ((xx * yy + 1e-9) as f64).sqrt()) as f32;
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
/// `xq` is decoded speech (i16), `pred_coef_q12` is \[num_subframes/2\]\[lpc_order\],
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

        // Frame gain — C: log(gain / 65536 + 1e-9f) — log() is double precision
        let gain_val = (gains_q16[k] as f32 / (1u32 << 16) as f32 + 1e-9) as f64;
        features[base + OSCE_LOG_GAIN_START] = std::hint::black_box(gain_val).ln() as f32;
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
    // C: log() and sin() are double precision
    let nb = if logscale {
        (numbits as f64).ln() as f32
    } else {
        numbits
    };
    // Upstream C uses a buggy CLIP macro:
    //   #define CLIP(a, min, max) (((a) < (min) ? (min) : (a)) > (max) ? (max) : (a))
    // When a < min, this expands to: (min > max ? max : a) = a (not min).
    // So the lower bound is never enforced. We must replicate this behavior.
    let clipped = if (if nb < min_val { min_val } else { nb }) > max_val {
        max_val
    } else {
        nb
    };
    let x = clipped - (max_val + min_val) / 2.0;
    for i in 0..8 {
        // black_box prevents LLVM auto-vectorization of sin() (see nndsp::compute_overlap_window).
        let val = (x * scales[i] - 0.5) as f64;
        emb[i] = std::hint::black_box(val).sin() as f32;
    }
}

/// Run LACE feature network.
///
/// Upstream C: dnn/osce.c:lace_feature_net
pub fn lace_feature_net(
    lace: &LACE,
    state: &mut LACEState,
    output: &mut [f32],
    features: &[f32],
    numbits: &[f32],
    periods: &[i32],
    arch: Arch,
) {
    let max_dim = LACE_COND_DIM.max(LACE_HIDDEN_FEATURE_DIM);
    let mut input_buffer = vec![0.0f32; 4 * max_dim];
    let mut output_buffer = vec![0.0f32; 4 * max_dim];
    let mut numbits_embedded = [0.0f32; 2 * LACE_NUMBITS_EMBEDDING_DIM];

    // C: log(RANGE_LOW), log(RANGE_HIGH) — log() on integer constants, double precision
    let range_low_ln = (LACE_NUMBITS_RANGE_LOW as f64).ln() as f32;
    let range_high_ln = (LACE_NUMBITS_RANGE_HIGH as f64).ln() as f32;
    compute_numbits_embedding(
        &mut numbits_embedded[..LACE_NUMBITS_EMBEDDING_DIM],
        numbits[0],
        &LACE_NUMBITS_SCALES,
        range_low_ln,
        range_high_ln,
        true,
    );
    compute_numbits_embedding(
        &mut numbits_embedded[LACE_NUMBITS_EMBEDDING_DIM..],
        numbits[1],
        &LACE_NUMBITS_SCALES,
        range_low_ln,
        range_high_ln,
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
            arch,
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
        arch,
    );

    // Tconv upsampling (dense)
    input_buffer[..4 * LACE_COND_DIM].copy_from_slice(&output_buffer[..4 * LACE_COND_DIM]);
    compute_generic_dense(
        &lace.layers.fnet_tconv,
        &mut output_buffer,
        &input_buffer,
        ACTIVATION_TANH,
        arch,
    );

    // GRU per subframe
    input_buffer[..4 * LACE_COND_DIM].copy_from_slice(&output_buffer[..4 * LACE_COND_DIM]);
    for sf in 0..4 {
        compute_generic_gru(
            &lace.layers.fnet_gru_input,
            &lace.layers.fnet_gru_recurrent,
            &mut state.feature_net_gru_state,
            &input_buffer[sf * LACE_COND_DIM..(sf + 1) * LACE_COND_DIM],
            arch,
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
    arch: Arch,
) {
    let mut feature_buffer = vec![0.0f32; 4 * LACE_COND_DIM];
    let mut output_buffer = vec![0.0f32; 4 * LACE_FRAME_SIZE];

    // Pre-emphasis
    for i in 0..4 * LACE_FRAME_SIZE {
        output_buffer[i] = x_in[i] - LACE_PREEMPH * state.preemph_mem;
        state.preemph_mem = x_in[i];
    }

    // Feature network
    lace_feature_net(
        lace,
        state,
        &mut feature_buffer,
        features,
        numbits,
        periods,
        arch,
    );

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
            arch,
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
            arch,
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
            arch,
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
    arch: Arch,
) {
    let max_dim = NOLACE_COND_DIM.max(NOLACE_HIDDEN_FEATURE_DIM);
    let mut input_buffer = vec![0.0f32; 4 * max_dim];
    let mut output_buffer = vec![0.0f32; 4 * max_dim];
    let mut numbits_embedded = [0.0f32; 2 * NOLACE_NUMBITS_EMBEDDING_DIM];

    // C: log(RANGE_LOW), log(RANGE_HIGH) — log() on integer constants, double precision
    let range_low_ln = (NOLACE_NUMBITS_RANGE_LOW as f64).ln() as f32;
    let range_high_ln = (NOLACE_NUMBITS_RANGE_HIGH as f64).ln() as f32;
    compute_numbits_embedding(
        &mut numbits_embedded[..NOLACE_NUMBITS_EMBEDDING_DIM],
        numbits[0],
        &NOLACE_NUMBITS_SCALES,
        range_low_ln,
        range_high_ln,
        true,
    );
    compute_numbits_embedding(
        &mut numbits_embedded[NOLACE_NUMBITS_EMBEDDING_DIM..],
        numbits[1],
        &NOLACE_NUMBITS_SCALES,
        range_low_ln,
        range_high_ln,
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
            arch,
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
        arch,
    );

    input_buffer[..4 * NOLACE_COND_DIM].copy_from_slice(&output_buffer[..4 * NOLACE_COND_DIM]);
    compute_generic_dense(
        &nolace.layers.fnet_tconv,
        &mut output_buffer,
        &input_buffer,
        ACTIVATION_TANH,
        arch,
    );

    input_buffer[..4 * NOLACE_COND_DIM].copy_from_slice(&output_buffer[..4 * NOLACE_COND_DIM]);
    for sf in 0..4 {
        compute_generic_gru(
            &nolace.layers.fnet_gru_input,
            &nolace.layers.fnet_gru_recurrent,
            &mut state.feature_net_gru_state,
            &input_buffer[sf * NOLACE_COND_DIM..(sf + 1) * NOLACE_COND_DIM],
            arch,
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
    arch: Arch,
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
        arch,
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
            arch,
        );
        compute_generic_conv1d(
            &nolace.layers.post_cf1,
            &mut feature_transform_buffer[sf * NOLACE_COND_DIM..(sf + 1) * NOLACE_COND_DIM],
            &mut state.post_cf1_state,
            &feature_buffer[sf * NOLACE_COND_DIM..],
            NOLACE_COND_DIM,
            ACTIVATION_TANH,
            arch,
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
            arch,
        );
        compute_generic_conv1d(
            &nolace.layers.post_cf2,
            &mut feature_transform_buffer[sf * NOLACE_COND_DIM..(sf + 1) * NOLACE_COND_DIM],
            &mut state.post_cf2_state,
            &feature_buffer[sf * NOLACE_COND_DIM..],
            NOLACE_COND_DIM,
            ACTIVATION_TANH,
            arch,
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
            arch,
        );
        compute_generic_conv1d(
            &nolace.layers.post_af1,
            &mut feature_transform_buffer[sf * NOLACE_COND_DIM..(sf + 1) * NOLACE_COND_DIM],
            &mut state.post_af1_state,
            &feature_buffer[sf * NOLACE_COND_DIM..],
            NOLACE_COND_DIM,
            ACTIVATION_TANH,
            arch,
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
            1,
            arch,
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
            arch,
        );
        compute_generic_conv1d(
            &nolace.layers.post_af2,
            &mut feature_transform_buffer[sf * NOLACE_COND_DIM..(sf + 1) * NOLACE_COND_DIM],
            &mut state.post_af2_state,
            &feature_buffer[sf * NOLACE_COND_DIM..],
            NOLACE_COND_DIM,
            ACTIVATION_TANH,
            arch,
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
            1,
            arch,
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
            arch,
        );
        compute_generic_conv1d(
            &nolace.layers.post_af3,
            &mut feature_transform_buffer[sf * NOLACE_COND_DIM..(sf + 1) * NOLACE_COND_DIM],
            &mut state.post_af3_state,
            &feature_buffer[sf * NOLACE_COND_DIM..],
            NOLACE_COND_DIM,
            ACTIVATION_TANH,
            arch,
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
            1,
            arch,
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
            arch,
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

/// Enhance one decoded SILK frame using OSCE (LACE or NoLACE).
///
/// Upstream C: dnn/osce.c:osce_enhance_frame
pub fn osce_enhance_frame(
    model: &OSCEModel,
    psDec: &mut silk_decoder_state,
    psDecCtrl: &silk_decoder_control,
    xq: &mut [i16],
    num_bits: i32,
    arch: Arch,
) {
    // Enhancement only implemented for 20 ms frame at 16kHz
    if psDec.fs_kHz != 16 || psDec.nb_subfr != 4 {
        let method = psDec.osce.method;
        osce_reset(&mut psDec.osce, method);
        return;
    }

    let mut features = [0.0f32; 4 * OSCE_FEATURE_DIM];
    let mut numbits = [0.0f32; 2];
    let mut periods = [0i32; 4];

    // Build PredCoef_Q12 slices for feature extraction
    let pred_coef_refs: [&[i16]; 2] = [&psDecCtrl.PredCoef_Q12[0], &psDecCtrl.PredCoef_Q12[1]];

    osce_calculate_features(
        &mut psDec.osce.features,
        psDec.nb_subfr,
        psDec.LPC_order,
        psDec.indices.signalType as i32,
        &pred_coef_refs,
        &psDecCtrl.pitchL,
        &psDecCtrl.LTPCoef_Q14,
        &psDecCtrl.Gains_Q16,
        xq,
        num_bits,
        &mut features,
        &mut numbits,
        &mut periods,
    );

    // Scale input to float [-1, 1]
    let mut in_buffer = [0.0f32; 320];
    for i in 0..320 {
        in_buffer[i] = xq[i] as f32 * (1.0 / 32768.0);
    }

    #[cfg(feature = "osce-dump-debug")]
    {
        use std::io::Write;
        use std::sync::Mutex;
        static FEAT_FILE: std::sync::LazyLock<Mutex<std::fs::File>> =
            std::sync::LazyLock::new(|| {
                Mutex::new(std::fs::File::create("/tmp/osce_rs_features.bin").unwrap())
            });
        static IN_FILE: std::sync::LazyLock<Mutex<std::fs::File>> =
            std::sync::LazyLock::new(|| {
                Mutex::new(std::fs::File::create("/tmp/osce_rs_in_buffer.bin").unwrap())
            });
        static OUT_FILE: std::sync::LazyLock<Mutex<std::fs::File>> =
            std::sync::LazyLock::new(|| {
                Mutex::new(std::fs::File::create("/tmp/osce_rs_out_buffer.bin").unwrap())
            });
        static NB_FILE: std::sync::LazyLock<Mutex<std::fs::File>> =
            std::sync::LazyLock::new(|| {
                Mutex::new(std::fs::File::create("/tmp/osce_rs_numbits.bin").unwrap())
            });
        static PER_FILE: std::sync::LazyLock<Mutex<std::fs::File>> =
            std::sync::LazyLock::new(|| {
                Mutex::new(std::fs::File::create("/tmp/osce_rs_periods.bin").unwrap())
            });
        FEAT_FILE
            .lock()
            .unwrap()
            .write_all(bytemuck::cast_slice(&features))
            .unwrap();
        NB_FILE
            .lock()
            .unwrap()
            .write_all(bytemuck::cast_slice(&numbits))
            .unwrap();
        PER_FILE
            .lock()
            .unwrap()
            .write_all(bytemuck::cast_slice(&periods))
            .unwrap();
    }

    let method = if model.loaded {
        psDec.osce.method
    } else {
        OSCE_METHOD_NONE
    };

    let mut out_buffer = [0.0f32; 320];
    match method {
        OSCE_METHOD_LACE => {
            if let Some(ref lace) = model.lace {
                lace_process_20ms_frame(
                    lace,
                    &mut psDec.osce.lace_state,
                    &mut out_buffer,
                    &in_buffer,
                    &features,
                    &numbits,
                    &periods,
                    arch,
                );
            } else {
                out_buffer.copy_from_slice(&in_buffer);
            }
        }
        OSCE_METHOD_NOLACE => {
            if let Some(ref nolace) = model.nolace {
                nolace_process_20ms_frame(
                    nolace,
                    &mut psDec.osce.nolace_state,
                    &mut out_buffer,
                    &in_buffer,
                    &features,
                    &numbits,
                    &periods,
                    arch,
                );
            } else {
                out_buffer.copy_from_slice(&in_buffer);
            }
        }
        _ => {
            out_buffer.copy_from_slice(&in_buffer);
        }
    }

    #[cfg(feature = "osce-dump-debug")]
    {
        use std::io::Write;
        use std::sync::Mutex;
        static IN_F: std::sync::LazyLock<Mutex<std::fs::File>> = std::sync::LazyLock::new(|| {
            Mutex::new(std::fs::File::create("/tmp/osce_rs_in_buffer.bin").unwrap())
        });
        static OUT_F: std::sync::LazyLock<Mutex<std::fs::File>> = std::sync::LazyLock::new(|| {
            Mutex::new(std::fs::File::create("/tmp/osce_rs_out_buffer.bin").unwrap())
        });
        IN_F.lock()
            .unwrap()
            .write_all(bytemuck::cast_slice(&in_buffer))
            .unwrap();
        OUT_F
            .lock()
            .unwrap()
            .write_all(bytemuck::cast_slice(&out_buffer))
            .unwrap();
    }

    // Cross-fade / bypass on reset (upstream C: osce.c lines 1031-1041)
    if psDec.osce.features.reset > 1 {
        out_buffer.copy_from_slice(&in_buffer);
        psDec.osce.features.reset -= 1;
    } else if psDec.osce.features.reset == 1 {
        osce_cross_fade_10ms(&mut out_buffer, &in_buffer, 320);
        psDec.osce.features.reset = 0;
    }

    // Scale output back to i16
    // C uses float2int(tmp) = lrintf(tmp) — round-to-nearest-even
    for i in 0..320 {
        let tmp = 32768.0f32 * out_buffer[i];
        xq[i] = tmp.clamp(-32767.0, 32767.0).round_ties_even() as i16;
    }
}

// ========== BBWENet: Bandwidth Extension ==========

// BWE filterbank tables (from osce_features.c)

static CENTER_BINS_BWE: [usize; 32] = [
    0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 105, 110,
    115, 120, 125, 130, 135, 140, 145, 150, 160,
];

static BAND_WEIGHTS_BWE: [f32; 32] = [
    0.333333333,
    0.200000000,
    0.200000000,
    0.200000000,
    0.200000000,
    0.200000000,
    0.200000000,
    0.200000000,
    0.200000000,
    0.200000000,
    0.200000000,
    0.200000000,
    0.200000000,
    0.200000000,
    0.200000000,
    0.200000000,
    0.200000000,
    0.200000000,
    0.200000000,
    0.200000000,
    0.200000000,
    0.200000000,
    0.200000000,
    0.200000000,
    0.200000000,
    0.200000000,
    0.200000000,
    0.200000000,
    0.200000000,
    0.200000000,
    0.133333333,
    0.181818182,
];

// Resampling filter coefficients (from osce.c)

static HQ_2X_EVEN: [f32; 3] = [0.026641845703125, 0.228668212890625, -0.4036407470703125];
static HQ_2X_ODD: [f32; 3] = [0.104583740234375, 0.3932037353515625, -0.152496337890625];

static FRAC_01_24: [f32; 8] = [
    0.00576782,
    -0.01831055,
    0.01882935,
    0.9328308,
    0.09143066,
    -0.04196167,
    0.01296997,
    -0.00140381,
];
static FRAC_17_24: [f32; 8] = [
    -3.14331055e-03,
    2.73437500e-02,
    -1.06414795e-01,
    3.64685059e-01,
    8.03863525e-01,
    -1.02233887e-01,
    1.61437988e-02,
    -1.22070312e-04,
];
static FRAC_09_24: [f32; 8] = [
    -0.00146484,
    0.02313232,
    -0.12072754,
    0.7315979,
    0.4621277,
    -0.12075806,
    0.0295105,
    -0.00326538,
];

const DELAY_SAMPLES: usize = 8;

/// 2x upsampling via IIR allpass filter.
///
/// Upstream C: dnn/osce.c:upsamp_2x
fn upsamp_2x(state: &mut ResampState, x_out: &mut [f32], x_in: &[f32], num_samples: usize) {
    debug_assert!(num_samples > 1);
    debug_assert!(num_samples < 4 * BBWENET_FRAME_SIZE16);

    let mut buffer = vec![0.0f32; num_samples];
    buffer[..num_samples].copy_from_slice(&x_in[..num_samples]);

    let (s_even, s_rest) = state.upsamp_buffer.split_at_mut(1);
    let s_even = &mut s_even[0];
    let s_odd = &mut s_rest[0];

    for k in 0..num_samples {
        let x = buffer[k];

        // even sample
        let mut y = x - s_even[0];
        let mut big_x = y * HQ_2X_EVEN[0];
        let mut tmp1 = s_even[0] + big_x;
        s_even[0] = x + big_x;

        y = tmp1 - s_even[1];
        big_x = y * HQ_2X_EVEN[1];
        let mut tmp2 = s_even[1] + big_x;
        s_even[1] = tmp1 + big_x;

        y = tmp2 - s_even[2];
        big_x = y * (1.0 + HQ_2X_EVEN[2]);
        let tmp3 = s_even[2] + big_x;
        s_even[2] = tmp2 + big_x;

        x_out[2 * k] = tmp3;

        // odd sample
        y = x - s_odd[0];
        big_x = y * HQ_2X_ODD[0];
        tmp1 = s_odd[0] + big_x;
        s_odd[0] = x + big_x;

        y = tmp1 - s_odd[1];
        big_x = y * HQ_2X_ODD[1];
        tmp2 = s_odd[1] + big_x;
        s_odd[1] = tmp1 + big_x;

        y = tmp2 - s_odd[2];
        big_x = y * (1.0 + HQ_2X_ODD[2]);
        let tmp3 = s_odd[2] + big_x;
        s_odd[2] = tmp2 + big_x;

        x_out[2 * k + 1] = tmp3;
    }
}

/// 3:2 polyphase interpolation (upsamples by factor 1.5).
///
/// Upstream C: dnn/osce.c:interpol_3_2
fn interpol_3_2(state: &mut ResampState, x_out: &mut [f32], x_in: &[f32], num_samples: usize) {
    debug_assert!(num_samples > 1);
    debug_assert!(num_samples < 8 * BBWENET_FRAME_SIZE16);
    debug_assert!(num_samples.is_multiple_of(2));

    let mut buffer = vec![0.0f32; num_samples + DELAY_SAMPLES];
    buffer[..DELAY_SAMPLES].copy_from_slice(&state.interpol_buffer);
    buffer[DELAY_SAMPLES..DELAY_SAMPLES + num_samples].copy_from_slice(&x_in[..num_samples]);

    let mut i_out = 0;
    let mut i_sample = 0;
    while i_sample < num_samples {
        x_out[i_out] = buffer[i_sample] * FRAC_01_24[0]
            + buffer[i_sample + 1] * FRAC_01_24[1]
            + buffer[i_sample + 2] * FRAC_01_24[2]
            + buffer[i_sample + 3] * FRAC_01_24[3]
            + buffer[i_sample + 4] * FRAC_01_24[4]
            + buffer[i_sample + 5] * FRAC_01_24[5]
            + buffer[i_sample + 6] * FRAC_01_24[6]
            + buffer[i_sample + 7] * FRAC_01_24[7];
        i_out += 1;

        x_out[i_out] = buffer[i_sample] * FRAC_17_24[0]
            + buffer[i_sample + 1] * FRAC_17_24[1]
            + buffer[i_sample + 2] * FRAC_17_24[2]
            + buffer[i_sample + 3] * FRAC_17_24[3]
            + buffer[i_sample + 4] * FRAC_17_24[4]
            + buffer[i_sample + 5] * FRAC_17_24[5]
            + buffer[i_sample + 6] * FRAC_17_24[6]
            + buffer[i_sample + 7] * FRAC_17_24[7];
        i_out += 1;

        x_out[i_out] = buffer[i_sample + 1] * FRAC_09_24[0]
            + buffer[i_sample + 2] * FRAC_09_24[1]
            + buffer[i_sample + 3] * FRAC_09_24[2]
            + buffer[i_sample + 4] * FRAC_09_24[3]
            + buffer[i_sample + 5] * FRAC_09_24[4]
            + buffer[i_sample + 6] * FRAC_09_24[5]
            + buffer[i_sample + 7] * FRAC_09_24[6]
            + buffer[i_sample + 8] * FRAC_09_24[7];
        i_out += 1;

        i_sample += 2;
    }

    // Save last DELAY_SAMPLES for next call
    state
        .interpol_buffer
        .copy_from_slice(&buffer[num_samples..num_samples + DELAY_SAMPLES]);
}

/// Valin activation: x *= sin(log(|x| + eps)).
///
/// Upstream C: dnn/osce.c:apply_valin_activation
fn apply_valin_activation(x: &mut [f32]) {
    use crate::celt::mathops::celt_cos_norm2;

    let len = x.len();
    debug_assert!(len <= 2 * BBWENET_TDSHAPE2_FRAME_SIZE);

    let mut y = vec![0.0f32; len];
    for i in 0..len {
        y[i] = x[i].abs() + 1e-6;
    }
    for i in 0..len {
        // C: celt_log(y[i]) = log(y[i]) via double precision
        y[i] = (y[i] as f64).ln() as f32;
    }
    for i in 0..len {
        // C: celt_sin(y[i]) = celt_cos_norm2(0.5f*PI*y[i] - 1.0f)
        let arg = 0.5 * std::f32::consts::PI * y[i] - 1.0;
        x[i] *= celt_cos_norm2(arg);
    }
}

/// BBWENet feature network: conv1 → conv2 → tconv → GRU.
///
/// Upstream C: dnn/osce.c:bbwe_feature_net
fn bbwe_feature_net(
    model: &BBWENet,
    state: &mut BBWENetState,
    output: &mut [f32],
    features: &[f32],
    num_frames: usize,
    arch: Arch,
) {
    use crate::dnn::nnet::{
        compute_generic_conv1d, compute_generic_dense, compute_generic_gru, ACTIVATION_TANH,
    };

    debug_assert_eq!(BBWENET_FNET_GRU_STATE_SIZE, BBWENET_FNET_TCONV_OUT_CHANNELS);
    debug_assert_eq!(BBWENET_FNET_TCONV_OUT_CHANNELS, BBWENET_FNET_CONV2_OUT_SIZE);
    debug_assert_eq!(BBWENET_FNET_CONV2_OUT_SIZE, BBWENET_FNET_CONV1_OUT_SIZE);

    let mut input_buffer = vec![0.0f32; 4 * BBWENET_FNET_GRU_STATE_SIZE];
    let mut output_buffer = vec![0.0f32; 4 * BBWENET_FNET_GRU_STATE_SIZE];

    // First conv layer
    for i_frame in 0..num_frames {
        compute_generic_conv1d(
            &model.layers.fnet_conv1,
            &mut output_buffer[i_frame * BBWENET_FNET_CONV1_OUT_SIZE
                ..(i_frame + 1) * BBWENET_FNET_CONV1_OUT_SIZE],
            &mut state.feature_net_conv1_state,
            &features[i_frame * BBWENET_FEATURE_DIM..(i_frame + 1) * BBWENET_FEATURE_DIM],
            BBWENET_FEATURE_DIM,
            ACTIVATION_TANH,
            arch,
        );
    }
    input_buffer[..num_frames * BBWENET_FNET_CONV1_OUT_SIZE]
        .copy_from_slice(&output_buffer[..num_frames * BBWENET_FNET_CONV1_OUT_SIZE]);

    // Second conv layer
    for i_frame in 0..num_frames {
        compute_generic_conv1d(
            &model.layers.fnet_conv2,
            &mut output_buffer[i_frame * BBWENET_FNET_CONV2_OUT_SIZE
                ..(i_frame + 1) * BBWENET_FNET_CONV2_OUT_SIZE],
            &mut state.feature_net_conv2_state,
            &input_buffer[i_frame * BBWENET_FNET_CONV1_OUT_SIZE
                ..(i_frame + 1) * BBWENET_FNET_CONV1_OUT_SIZE],
            BBWENET_FNET_CONV1_OUT_SIZE,
            ACTIVATION_TANH,
            arch,
        );
    }
    let tconv_out_len = num_frames * BBWENET_FNET_TCONV_OUT_CHANNELS * BBWENET_FNET_TCONV_STRIDE;
    input_buffer[..num_frames * BBWENET_FNET_CONV2_OUT_SIZE]
        .copy_from_slice(&output_buffer[..num_frames * BBWENET_FNET_CONV2_OUT_SIZE]);

    // Transposed convolution upsampling
    for i_frame in 0..num_frames {
        let out_offset = i_frame * BBWENET_FNET_TCONV_OUT_CHANNELS * BBWENET_FNET_TCONV_STRIDE;
        let in_offset = i_frame * BBWENET_FNET_CONV2_OUT_SIZE;
        compute_generic_dense(
            &model.layers.fnet_tconv,
            &mut output_buffer[out_offset
                ..out_offset + BBWENET_FNET_TCONV_OUT_CHANNELS * BBWENET_FNET_TCONV_STRIDE],
            &input_buffer[in_offset..in_offset + BBWENET_FNET_CONV2_OUT_SIZE],
            ACTIVATION_TANH,
            arch,
        );
    }
    input_buffer[..tconv_out_len].copy_from_slice(&output_buffer[..tconv_out_len]);

    // GRU
    debug_assert_eq!(BBWENET_FNET_TCONV_STRIDE, 2);
    let num_subframes = BBWENET_FNET_TCONV_STRIDE * num_frames;
    for i_subframe in 0..num_subframes {
        let in_offset = i_subframe * BBWENET_FNET_TCONV_OUT_CHANNELS;
        compute_generic_gru(
            &model.layers.fnet_gru_input,
            &model.layers.fnet_gru_recurrent,
            &mut state.feature_net_gru_state,
            &input_buffer[in_offset..in_offset + BBWENET_FNET_TCONV_OUT_CHANNELS],
            arch,
        );
        let out_offset = i_subframe * BBWENET_FNET_GRU_STATE_SIZE;
        output[out_offset..out_offset + BBWENET_FNET_GRU_STATE_SIZE]
            .copy_from_slice(&state.feature_net_gru_state);
    }
}

/// Process frames through BBWENet signal path.
///
/// Pipeline: AF1 → upsamp_2x → TDShape1 → AF2 → interpol_3_2 → TDShape2 → AF3
///
/// Upstream C: dnn/osce.c:bbwenet_process_frames
fn bbwenet_process_frames(
    model: &BBWENet,
    state: &mut BBWENetState,
    x_out: &mut [f32],
    x_in: &[f32],
    features: &[f32],
    num_frames: usize,
    arch: Arch,
) {
    let num_subframes = 2 * num_frames;
    let mut latent_features = vec![0.0f32; 4 * BBWENET_COND_DIM];
    // 3 channels × 4 subframes × 3×FRAME_SIZE16 = enough for 48kHz
    let buf_size = 3 * 3 * 4 * 3 * BBWENET_FRAME_SIZE16;
    let mut x_buffer1 = vec![0.0f32; buf_size];
    let mut x_buffer2 = vec![0.0f32; buf_size];

    // Feature net
    bbwe_feature_net(
        model,
        state,
        &mut latent_features,
        features,
        num_frames,
        arch,
    );

    // Stage 1: Adaptive filtering (1ch → 3ch at 16kHz)
    for i_sub in 0..num_subframes {
        let out_offset = i_sub * BBWENET_AF1_FRAME_SIZE * BBWENET_AF1_OUT_CHANNELS;
        let in_offset = i_sub * BBWENET_AF1_FRAME_SIZE;
        let feat_offset = i_sub * BBWENET_COND_DIM;
        adaconv_process_frame(
            &mut state.af1_state,
            &mut x_buffer1
                [out_offset..out_offset + BBWENET_AF1_FRAME_SIZE * BBWENET_AF1_OUT_CHANNELS],
            &x_in[in_offset..in_offset + BBWENET_AF1_FRAME_SIZE],
            &latent_features[feat_offset..feat_offset + BBWENET_COND_DIM],
            &model.layers.af1_kernel,
            &model.layers.af1_gain,
            BBWENET_COND_DIM,
            BBWENET_AF1_FRAME_SIZE,
            BBWENET_AF1_OVERLAP_SIZE,
            BBWENET_AF1_IN_CHANNELS,
            BBWENET_AF1_OUT_CHANNELS,
            BBWENET_AF1_KERNEL_SIZE,
            BBWENET_AF1_LEFT_PADDING,
            BBWENET_AF1_FILTER_GAIN_A,
            BBWENET_AF1_FILTER_GAIN_B,
            BBWENET_AF1_SHAPE_GAIN,
            &model.window16,
            arch,
        );
    }

    // Stage 2: Upsample 2x (16kHz → 32kHz) + TDShape1 + Valin activation
    debug_assert_eq!(BBWENET_AF1_OUT_CHANNELS, 3);
    debug_assert_eq!(2 * BBWENET_AF1_FRAME_SIZE, BBWENET_TDSHAPE1_FRAME_SIZE);
    for i_sub in 0..num_subframes {
        // 2x upsample each of the 3 channels
        for i_ch in 0..3 {
            let src_offset = i_sub * BBWENET_AF1_FRAME_SIZE * BBWENET_AF1_OUT_CHANNELS
                + i_ch * BBWENET_AF1_FRAME_SIZE;
            let dst_offset = i_sub * BBWENET_TDSHAPE1_FRAME_SIZE * BBWENET_AF1_OUT_CHANNELS
                + i_ch * BBWENET_TDSHAPE1_FRAME_SIZE;
            // Need temporary buffer since x_buffer1 is the source
            let src: Vec<f32> = x_buffer1[src_offset..src_offset + BBWENET_AF1_FRAME_SIZE].to_vec();
            upsamp_2x(
                &mut state.resampler_state[i_ch],
                &mut x_buffer2[dst_offset..dst_offset + BBWENET_TDSHAPE1_FRAME_SIZE],
                &src,
                BBWENET_AF1_FRAME_SIZE,
            );
        }

        // TDShape on second channel (in place)
        let shape_offset = i_sub * BBWENET_AF1_OUT_CHANNELS * BBWENET_TDSHAPE1_FRAME_SIZE
            + BBWENET_TDSHAPE1_FRAME_SIZE;
        let feat_offset = i_sub * BBWENET_COND_DIM;
        // Need to copy x_in for in-place processing
        let shape_in: Vec<f32> =
            x_buffer2[shape_offset..shape_offset + BBWENET_TDSHAPE1_FRAME_SIZE].to_vec();
        adashape_process_frame(
            &mut state.tdshape1_state,
            &mut x_buffer2[shape_offset..shape_offset + BBWENET_TDSHAPE1_FRAME_SIZE],
            &shape_in,
            &latent_features[feat_offset..feat_offset + BBWENET_COND_DIM],
            &model.layers.tdshape1_alpha1_f,
            &model.layers.tdshape1_alpha1_t,
            &model.layers.tdshape1_alpha2,
            BBWENET_TDSHAPE1_FEATURE_DIM,
            BBWENET_TDSHAPE1_FRAME_SIZE,
            BBWENET_TDSHAPE1_AVG_POOL_K,
            BBWENET_TDSHAPE1_INTERPOLATE_K,
            arch,
        );

        // Valin activation on third channel (in place)
        let act_offset = i_sub * BBWENET_AF1_OUT_CHANNELS * BBWENET_TDSHAPE1_FRAME_SIZE
            + 2 * BBWENET_TDSHAPE1_FRAME_SIZE;
        apply_valin_activation(
            &mut x_buffer2[act_offset..act_offset + BBWENET_TDSHAPE1_FRAME_SIZE],
        );
    }

    // Stage 3: Mixing via AF2 (3ch → 3ch at 32kHz)
    for i_sub in 0..num_subframes {
        let out_offset = i_sub * BBWENET_AF2_FRAME_SIZE * BBWENET_AF2_OUT_CHANNELS;
        let in_offset = i_sub * BBWENET_AF2_FRAME_SIZE * BBWENET_AF1_OUT_CHANNELS;
        let feat_offset = i_sub * BBWENET_COND_DIM;
        adaconv_process_frame(
            &mut state.af2_state,
            &mut x_buffer1
                [out_offset..out_offset + BBWENET_AF2_FRAME_SIZE * BBWENET_AF2_OUT_CHANNELS],
            &x_buffer2[in_offset..in_offset + BBWENET_AF2_FRAME_SIZE * BBWENET_AF1_OUT_CHANNELS],
            &latent_features[feat_offset..feat_offset + BBWENET_COND_DIM],
            &model.layers.af2_kernel,
            &model.layers.af2_gain,
            BBWENET_COND_DIM,
            BBWENET_AF2_FRAME_SIZE,
            BBWENET_AF2_OVERLAP_SIZE,
            BBWENET_AF2_IN_CHANNELS,
            BBWENET_AF2_OUT_CHANNELS,
            BBWENET_AF2_KERNEL_SIZE,
            BBWENET_AF2_LEFT_PADDING,
            BBWENET_AF2_FILTER_GAIN_A,
            BBWENET_AF2_FILTER_GAIN_B,
            BBWENET_AF2_SHAPE_GAIN,
            &model.window32,
            arch,
        );
    }

    // Stage 4: Interpolate 3:2 (32kHz → 48kHz) + TDShape2 + Valin activation
    debug_assert_eq!(BBWENET_AF2_OUT_CHANNELS, 3);
    debug_assert_eq!(3 * BBWENET_AF2_FRAME_SIZE, 2 * BBWENET_TDSHAPE2_FRAME_SIZE);
    for i_sub in 0..num_subframes {
        // 3:2 interpolation on each of the 3 channels
        for i_ch in 0..3 {
            let src_offset = i_sub * BBWENET_TDSHAPE1_FRAME_SIZE * BBWENET_AF2_OUT_CHANNELS
                + i_ch * BBWENET_TDSHAPE1_FRAME_SIZE;
            let dst_offset = i_sub * BBWENET_AF3_FRAME_SIZE * BBWENET_AF2_OUT_CHANNELS
                + i_ch * BBWENET_TDSHAPE2_FRAME_SIZE;
            let src: Vec<f32> =
                x_buffer1[src_offset..src_offset + BBWENET_TDSHAPE1_FRAME_SIZE].to_vec();
            interpol_3_2(
                &mut state.resampler_state[i_ch],
                &mut x_buffer2[dst_offset..dst_offset + BBWENET_TDSHAPE2_FRAME_SIZE],
                &src,
                BBWENET_TDSHAPE1_FRAME_SIZE,
            );
        }

        // TDShape on second channel (in place)
        let shape_offset = i_sub * BBWENET_AF2_OUT_CHANNELS * BBWENET_TDSHAPE2_FRAME_SIZE
            + BBWENET_TDSHAPE2_FRAME_SIZE;
        let feat_offset = i_sub * BBWENET_COND_DIM;
        let shape_in: Vec<f32> =
            x_buffer2[shape_offset..shape_offset + BBWENET_TDSHAPE2_FRAME_SIZE].to_vec();
        adashape_process_frame(
            &mut state.tdshape2_state,
            &mut x_buffer2[shape_offset..shape_offset + BBWENET_TDSHAPE2_FRAME_SIZE],
            &shape_in,
            &latent_features[feat_offset..feat_offset + BBWENET_COND_DIM],
            &model.layers.tdshape2_alpha1_f,
            &model.layers.tdshape2_alpha1_t,
            &model.layers.tdshape2_alpha2,
            BBWENET_TDSHAPE2_FEATURE_DIM,
            BBWENET_TDSHAPE2_FRAME_SIZE,
            BBWENET_TDSHAPE2_AVG_POOL_K,
            BBWENET_TDSHAPE2_INTERPOLATE_K,
            arch,
        );

        // Valin activation on third channel (in place)
        let act_offset = i_sub * BBWENET_AF2_OUT_CHANNELS * BBWENET_TDSHAPE2_FRAME_SIZE
            + 2 * BBWENET_TDSHAPE2_FRAME_SIZE;
        apply_valin_activation(
            &mut x_buffer2[act_offset..act_offset + BBWENET_TDSHAPE2_FRAME_SIZE],
        );
    }

    // Stage 5: Final mixing via AF3 (3ch → 1ch at 48kHz)
    debug_assert_eq!(BBWENET_AF3_OUT_CHANNELS, 1);
    for i_sub in 0..num_subframes {
        let out_offset = i_sub * BBWENET_AF3_FRAME_SIZE;
        let in_offset = i_sub * BBWENET_TDSHAPE2_FRAME_SIZE * BBWENET_AF2_OUT_CHANNELS;
        let feat_offset = i_sub * BBWENET_COND_DIM;
        adaconv_process_frame(
            &mut state.af3_state,
            &mut x_out[out_offset..out_offset + BBWENET_AF3_FRAME_SIZE],
            &x_buffer2
                [in_offset..in_offset + BBWENET_TDSHAPE2_FRAME_SIZE * BBWENET_AF2_OUT_CHANNELS],
            &latent_features[feat_offset..feat_offset + BBWENET_COND_DIM],
            &model.layers.af3_kernel,
            &model.layers.af3_gain,
            BBWENET_COND_DIM,
            BBWENET_AF3_FRAME_SIZE,
            BBWENET_AF3_OVERLAP_SIZE,
            BBWENET_AF3_IN_CHANNELS,
            BBWENET_AF3_OUT_CHANNELS,
            BBWENET_AF3_KERNEL_SIZE,
            BBWENET_AF3_LEFT_PADDING,
            BBWENET_AF3_FILTER_GAIN_A,
            BBWENET_AF3_FILTER_GAIN_B,
            BBWENET_AF3_SHAPE_GAIN,
            &model.window48,
            arch,
        );
    }
}

/// Calculate BWE features (log magnitude spectrum + instantaneous frequency).
///
/// Upstream C: dnn/osce_features.c:osce_bwe_calculate_features
pub fn osce_bwe_calculate_features(
    ps_features: &mut OSCEBWEFeatureState,
    features: &mut [f32],
    xq: &[i16],
    num_samples: usize,
) {
    debug_assert!(num_samples.is_multiple_of(OSCE_BWE_HALF_WINDOW_SIZE));
    debug_assert_eq!(OSCE_BWE_WINDOW_SIZE, 320);

    let osce_window = generate_osce_window();
    let num_frames = num_samples / OSCE_BWE_HALF_WINDOW_SIZE;

    for frame in 0..num_frames {
        let feat_offset = frame * OSCE_BWE_FEATURE_DIM;
        // Clear feature vector for this frame
        for v in features[feat_offset..feat_offset + OSCE_BWE_FEATURE_DIM].iter_mut() {
            *v = 0.0;
        }

        let lmspec_offset = feat_offset;
        let instafreq_offset = feat_offset + OSCE_BWE_NUM_BANDS;
        let x = &xq[frame * OSCE_BWE_HALF_WINDOW_SIZE..];

        // Build analysis window: [history | new samples]
        let mut buffer = [0.0f32; OSCE_BWE_WINDOW_SIZE];
        buffer[..OSCE_BWE_HALF_WINDOW_SIZE].copy_from_slice(&ps_features.signal_history);
        for n in 0..OSCE_BWE_HALF_WINDOW_SIZE {
            buffer[n + OSCE_BWE_HALF_WINDOW_SIZE] = x[n] as f32 / (1u32 << 15) as f32;
        }

        // Update signal history
        ps_features
            .signal_history
            .copy_from_slice(&buffer[OSCE_BWE_HALF_WINDOW_SIZE..]);

        // Apply window
        for n in 0..OSCE_BWE_WINDOW_SIZE {
            buffer[n] *= osce_window[n];
        }

        // DFT
        let mut fft_buffer = [kiss_fft_cpx { re: 0.0, im: 0.0 }; OSCE_BWE_WINDOW_SIZE];
        forward_transform(&mut fft_buffer, &buffer);

        // Instantaneous frequency
        let mut spec = vec![0.0f32; 2 * OSCE_BWE_MAX_INSTAFREQ_BIN + 2];
        for k in 0..=OSCE_BWE_MAX_INSTAFREQ_BIN {
            spec[2 * k] = OSCE_BWE_WINDOW_SIZE as f32 * fft_buffer[k].re + 1e-9;
            spec[2 * k + 1] = OSCE_BWE_WINDOW_SIZE as f32 * fft_buffer[k].im;

            let re1 = spec[2 * k];
            let im1 = spec[2 * k + 1];
            let re2 = ps_features.last_spec[2 * k];
            let im2 = ps_features.last_spec[2 * k + 1];
            let aux_r = re1 * re2 + im1 * im2;
            let aux_i = im1 * re2 - re1 * im2;
            // C uses double-precision sqrt
            let aux_abs = ((aux_r * aux_r + aux_i * aux_i) as f64).sqrt() as f32;
            features[instafreq_offset + k] = aux_r / (aux_abs + 1e-9);
            features[instafreq_offset + k + OSCE_BWE_MAX_INSTAFREQ_BIN + 1] =
                aux_i / (aux_abs + 1e-9);
        }

        // ERB-scale magnitude spectrogram
        let mut mag_spec = [0.0f32; OSCE_SPEC_NUM_FREQS];
        for k in 0..OSCE_SPEC_NUM_FREQS {
            // C uses double-precision sqrt
            mag_spec[k] = (OSCE_BWE_WINDOW_SIZE as f64
                * ((fft_buffer[k].re * fft_buffer[k].re + fft_buffer[k].im * fft_buffer[k].im)
                    as f64)
                    .sqrt()) as f32;
        }

        apply_filterbank(
            &mut features[lmspec_offset..lmspec_offset + OSCE_BWE_NUM_BANDS],
            &mag_spec,
            &CENTER_BINS_BWE,
            &BAND_WEIGHTS_BWE,
            OSCE_BWE_NUM_BANDS,
        );

        for k in 0..OSCE_BWE_NUM_BANDS {
            // C: log(lmspec[k] + 1e-9) — log() is double precision
            let val = (features[lmspec_offset + k] + 1e-9) as f64;
            features[lmspec_offset + k] = std::hint::black_box(val).ln() as f32;
        }

        // Update instafreq buffer
        ps_features.last_spec.copy_from_slice(&spec);
    }
}

/// Reset BBWENet state.
///
/// Upstream C: dnn/osce.c:reset_bbwenet_state
fn reset_bbwenet_state(state: &mut BBWENetState) {
    *state = BBWENetState::default();
}

/// Reset OSCE BWE state.
///
/// Upstream C: dnn/osce.c:osce_bwe_reset
pub fn osce_bwe_reset(bwe: &mut OSCEBWEState, bwe_features: &mut OSCEBWEFeatureState) {
    *bwe_features = OSCEBWEFeatureState::default();
    // "weird python initialization: Fix eventually!" — matches C
    for k in 0..=OSCE_BWE_MAX_INSTAFREQ_BIN {
        bwe_features.last_spec[2 * k] = 1e-9;
    }
    reset_bbwenet_state(&mut bwe.bbwenet);
}

/// Cross-fade BWE output with resampled fallback over 10ms.
///
/// Upstream C: dnn/osce_features.c:osce_bwe_cross_fade_10ms
pub fn osce_bwe_cross_fade_10ms(x_fadein: &mut [i16], x_fadeout: &[i16], length: usize) {
    debug_assert!(length >= 480);
    let osce_window = generate_osce_window();
    let f = 1.0f32 / 3.0;
    for i in 0..160 {
        let diff = if i == 159 {
            0.0
        } else {
            osce_window[i + 1] - osce_window[i]
        };
        let mut w_curr = osce_window[i];
        x_fadein[3 * i] = (w_curr * x_fadein[3 * i] as f32
            + (1.0 - w_curr) * x_fadeout[3 * i] as f32
            + 0.5) as i16;
        w_curr += diff * f;
        x_fadein[3 * i + 1] = (w_curr * x_fadein[3 * i + 1] as f32
            + (1.0 - w_curr) * x_fadeout[3 * i + 1] as f32
            + 0.5) as i16;
        w_curr += diff * f;
        x_fadein[3 * i + 2] = (w_curr * x_fadein[3 * i + 2] as f32
            + (1.0 - w_curr) * x_fadeout[3 * i + 2] as f32
            + 0.5) as i16;
    }
}

/// Top-level BWE function: extract features, run BBWENet, scale/delay output.
///
/// Upstream C: dnn/osce.c:osce_bwe
pub fn osce_bwe(
    model: &OSCEModel,
    bwe: &mut OSCEBWEState,
    bwe_features: &mut OSCEBWEFeatureState,
    xq48: &mut [i16],
    xq16: &[i16],
    xq16_len: usize,
    arch: Arch,
) {
    debug_assert!(xq16_len == 160 || xq16_len == 320);

    let bbwenet = match &model.bbwenet {
        Some(b) => b,
        None => return,
    };

    let num_frames = xq16_len / 160;

    // Scale input to float [-1, 1]
    let mut in_buffer = vec![0.0f32; xq16_len];
    for i in 0..xq16_len {
        in_buffer[i] = xq16[i] as f32 * (1.0 / 32768.0);
    }

    // Calculate features
    let mut features = vec![0.0f32; 2 * OSCE_BWE_FEATURE_DIM];
    osce_bwe_calculate_features(bwe_features, &mut features, xq16, xq16_len);

    // Process frames through BBWENet
    let out_len = 3 * xq16_len;
    let mut out_buffer = vec![0.0f32; out_len];
    bbwenet_process_frames(
        bbwenet,
        &mut bwe.bbwenet,
        &mut out_buffer,
        &in_buffer,
        &features,
        num_frames,
        arch,
    );

    // Scale and delay output
    // Copy delayed samples from previous call
    xq48[..OSCE_BWE_OUTPUT_DELAY].copy_from_slice(&bwe.bbwenet.output_buffer);

    // Convert float output to i16 with clipping
    for i in 0..out_len - OSCE_BWE_OUTPUT_DELAY {
        let tmp = 32768.0f32 * out_buffer[i];
        let tmp = tmp.clamp(-32767.0, 32767.0);
        // C uses float2int(tmp) = lrintf(tmp) — round-to-nearest-even
        xq48[i + OSCE_BWE_OUTPUT_DELAY] = tmp.round_ties_even() as i16;
    }

    // Save tail samples for next call's delay
    for i in 0..OSCE_BWE_OUTPUT_DELAY {
        let tmp = 32768.0f32 * out_buffer[out_len - OSCE_BWE_OUTPUT_DELAY + i];
        let tmp = tmp.clamp(-32767.0, 32767.0);
        bwe.bbwenet.output_buffer[i] = tmp.round_ties_even() as i16;
    }
}
