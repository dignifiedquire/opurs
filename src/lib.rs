//! Pure Rust implementation of the Opus audio codec, bit-exact with libopus 1.5.2.

extern crate core;

mod enums;
mod error;
pub mod util;

#[cfg(feature = "tools")]
pub mod tools;

#[cfg(not(feature = "tools"))]
mod celt;
#[cfg(feature = "tools")]
pub mod celt;

mod silk;

mod opus;

#[cfg(feature = "deep-plc")]
pub mod dnn;

// TODO: copy over the docs
// =====
// opus.h
// =====

// opus_encoder
pub use crate::opus::opus_encoder::OpusEncoder;
// opus_multistream (layout/config scaffolding + packet helpers)
pub use crate::opus::opus_multistream::{OpusMultistreamConfig, OpusMultistreamLayout};
pub use crate::opus::opus_multistream_decoder::{
    opus_multistream_decode, opus_multistream_decode24, opus_multistream_decode_float,
    opus_multistream_decoder_create, opus_multistream_decoder_destroy,
    opus_multistream_decoder_get_decoder_state, opus_multistream_decoder_get_size,
    opus_multistream_decoder_init, OpusMSDecoder,
};
pub use crate::opus::opus_multistream_encoder::{
    opus_multistream_encode, opus_multistream_encode24, opus_multistream_encode_float,
    opus_multistream_encoder_create, opus_multistream_encoder_destroy,
    opus_multistream_encoder_get_encoder_state, opus_multistream_encoder_get_size,
    opus_multistream_encoder_init, opus_multistream_surround_encoder_create,
    opus_multistream_surround_encoder_get_size, opus_multistream_surround_encoder_init,
    OpusMSEncoder,
};
pub use crate::opus::opus_projection_decoder::{
    opus_projection_decode, opus_projection_decode24, opus_projection_decode_float,
    opus_projection_decoder_create, opus_projection_decoder_destroy,
    opus_projection_decoder_get_decoder_state, opus_projection_decoder_get_size,
    opus_projection_decoder_init, OpusProjectionDecoder,
};
pub use crate::opus::opus_projection_encoder::{
    opus_projection_ambisonics_encoder_create, opus_projection_ambisonics_encoder_get_size,
    opus_projection_ambisonics_encoder_init, opus_projection_encode, opus_projection_encode24,
    opus_projection_encode_float, opus_projection_encoder_destroy, OpusProjectionEncoder,
};
// opus_decoder
pub use crate::opus::opus_decoder::{
    opus_decode, opus_decode_float, opus_decoder_get_nb_samples, opus_packet_get_bandwidth,
    opus_packet_get_nb_channels, opus_packet_get_nb_frames, opus_packet_get_nb_samples,
    OpusDecoder,
};

pub use crate::opus::packet::{
    opus_packet_get_samples_per_frame, opus_packet_parse, opus_pcm_soft_clip,
};
// opus_repacketizer
pub use crate::opus::repacketizer::{
    opus_multistream_packet_pad, opus_multistream_packet_unpad, opus_packet_pad, opus_packet_unpad,
    OpusRepacketizer,
};

// =====
// opus_defines.h
// =====
// opus_errorcodes
pub use crate::opus::opus_defines::{
    OPUS_ALLOC_FAIL, OPUS_BAD_ARG, OPUS_BUFFER_TOO_SMALL, OPUS_INTERNAL_ERROR, OPUS_INVALID_PACKET,
    OPUS_INVALID_STATE, OPUS_OK, OPUS_UNIMPLEMENTED,
};
pub use crate::opus::opus_defines::{
    OPUS_GET_APPLICATION_REQUEST, OPUS_GET_BANDWIDTH_REQUEST, OPUS_GET_BITRATE_REQUEST,
    OPUS_GET_COMPLEXITY_REQUEST, OPUS_GET_DRED_DURATION_REQUEST, OPUS_GET_DTX_REQUEST,
    OPUS_GET_EXPERT_FRAME_DURATION_REQUEST, OPUS_GET_FINAL_RANGE_REQUEST,
    OPUS_GET_FORCE_CHANNELS_REQUEST, OPUS_GET_GAIN_REQUEST, OPUS_GET_INBAND_FEC_REQUEST,
    OPUS_GET_IN_DTX_REQUEST, OPUS_GET_LAST_PACKET_DURATION_REQUEST, OPUS_GET_LOOKAHEAD_REQUEST,
    OPUS_GET_LSB_DEPTH_REQUEST, OPUS_GET_MAX_BANDWIDTH_REQUEST, OPUS_GET_PACKET_LOSS_PERC_REQUEST,
    OPUS_GET_PHASE_INVERSION_DISABLED_REQUEST, OPUS_GET_PITCH_REQUEST,
    OPUS_GET_PREDICTION_DISABLED_REQUEST, OPUS_GET_SAMPLE_RATE_REQUEST, OPUS_GET_SIGNAL_REQUEST,
    OPUS_GET_VBR_CONSTRAINT_REQUEST, OPUS_GET_VBR_REQUEST,
    OPUS_MULTISTREAM_GET_DECODER_STATE_REQUEST, OPUS_MULTISTREAM_GET_ENCODER_STATE_REQUEST,
    OPUS_PROJECTION_GET_DEMIXING_MATRIX_GAIN_REQUEST, OPUS_PROJECTION_GET_DEMIXING_MATRIX_REQUEST,
    OPUS_PROJECTION_GET_DEMIXING_MATRIX_SIZE_REQUEST, OPUS_RESET_STATE,
    OPUS_SET_APPLICATION_REQUEST, OPUS_SET_BANDWIDTH_REQUEST, OPUS_SET_BITRATE_REQUEST,
    OPUS_SET_COMPLEXITY_REQUEST, OPUS_SET_DNN_BLOB_REQUEST, OPUS_SET_DRED_DURATION_REQUEST,
    OPUS_SET_DTX_REQUEST, OPUS_SET_EXPERT_FRAME_DURATION_REQUEST, OPUS_SET_FORCE_CHANNELS_REQUEST,
    OPUS_SET_GAIN_REQUEST, OPUS_SET_INBAND_FEC_REQUEST, OPUS_SET_LSB_DEPTH_REQUEST,
    OPUS_SET_MAX_BANDWIDTH_REQUEST, OPUS_SET_PACKET_LOSS_PERC_REQUEST,
    OPUS_SET_PHASE_INVERSION_DISABLED_REQUEST, OPUS_SET_PREDICTION_DISABLED_REQUEST,
    OPUS_SET_SIGNAL_REQUEST, OPUS_SET_VBR_CONSTRAINT_REQUEST, OPUS_SET_VBR_REQUEST,
};
// opus_ctlvalues
pub use crate::opus::opus_defines::{
    OPUS_APPLICATION_AUDIO, OPUS_APPLICATION_RESTRICTED_LOWDELAY, OPUS_APPLICATION_VOIP, OPUS_AUTO,
    OPUS_BANDWIDTH_FULLBAND, OPUS_BANDWIDTH_MEDIUMBAND, OPUS_BANDWIDTH_NARROWBAND,
    OPUS_BANDWIDTH_SUPERWIDEBAND, OPUS_BANDWIDTH_WIDEBAND, OPUS_BITRATE_MAX, OPUS_FRAMESIZE_100_MS,
    OPUS_FRAMESIZE_10_MS, OPUS_FRAMESIZE_120_MS, OPUS_FRAMESIZE_20_MS, OPUS_FRAMESIZE_2_5_MS,
    OPUS_FRAMESIZE_40_MS, OPUS_FRAMESIZE_5_MS, OPUS_FRAMESIZE_60_MS, OPUS_FRAMESIZE_80_MS,
    OPUS_FRAMESIZE_ARG, OPUS_SIGNAL_MUSIC, OPUS_SIGNAL_VOICE,
};
// opus_libinfo
pub use crate::celt::common::{opus_get_version_string, opus_strerror};

// =====
// opus_custom.h
// =====
pub use crate::celt::celt_decoder::OpusCustomDecoder;
pub use crate::celt::celt_encoder::OpusCustomEncoder;
// NOTE: we don't support opus custom modes, so no opus_custom_destroy here
pub use crate::celt::modes::{opus_custom_mode_create, OpusCustomMode};

// expose opus_private
pub use crate::opus::opus_private;

// Public API types
pub use crate::opus::mapping_matrix::MappingMatrix;
pub use enums::{Application, Bandwidth, Bitrate, Channels, FrameSize, Signal};
pub use error::{ErrorCode, Result as OpusResult};

// =====
// Internal re-exports for unit tests
// =====
// These expose CELT/SILK internals that are needed by tests in tests/.
// Prefer pub(crate) in the source modules; these re-exports make them
// accessible to integration tests without making the celt/silk modules public.
pub mod internals {
    // -- CELT mathops --
    pub use crate::celt::mathops::{celt_cos_norm, celt_exp2, celt_log2, celt_sqrt, isqrt32};

    // -- CELT bands (bitexact trig, spread constants) --
    pub use crate::celt::bands::{bitexact_cos, bitexact_log2tan, SPREAD_NORMAL};

    // -- CELT entropy coder --
    pub use crate::celt::entcode::{ec_ctx, ec_get_error, ec_tell, ec_tell_frac};
    pub use crate::celt::entdec::{
        ec_dec, ec_dec_bit_logp, ec_dec_bits, ec_dec_icdf, ec_dec_init, ec_dec_uint, ec_dec_update,
        ec_decode, ec_decode_bin,
    };
    pub use crate::celt::entenc::{
        ec_enc, ec_enc_bit_logp, ec_enc_bits, ec_enc_done, ec_enc_icdf, ec_enc_init,
        ec_enc_patch_initial_bits, ec_enc_shrink, ec_enc_uint, ec_encode, ec_encode_bin,
    };

    // -- CELT laplace --
    pub use crate::celt::laplace::{
        ec_laplace_decode, ec_laplace_encode, LAPLACE_MINP, LAPLACE_NMIN,
    };

    // -- CELT CWRS --
    pub use crate::celt::cwrs::{cwrsi, decode_pulses, encode_pulses, icwrs, pvq_v};

    // -- CELT rate --
    pub use crate::celt::rate::get_pulses;

    // -- CELT VQ (rotation) --
    pub use crate::celt::vq::exp_rotation;

    // -- CELT FFT --
    pub use crate::celt::kiss_fft::{kiss_fft_state, opus_fft_c, opus_fft_impl};

    // -- CELT MDCT --
    pub use crate::celt::mdct::{mdct_backward, mdct_forward};

    // -- CELT modes (for FFT/MDCT state access) --
    pub use crate::celt::modes::{opus_custom_mode_create, OpusCustomMode};

    // -- SILK LPC --
    pub use crate::silk::LPC_inv_pred_gain::silk_LPC_inverse_pred_gain_c;
    pub use crate::silk::SigProc_FIX::SILK_MAX_ORDER_LPC;

    // -- CELT pitch (for benchmarks) --
    // Dispatch wrappers (use SIMD when available):
    pub use crate::celt::pitch::{
        celt_inner_prod, celt_pitch_xcorr, dual_inner_prod, xcorr_kernel,
    };
    // Scalar implementations (for A/B comparison):
    pub use crate::celt::pitch::{
        celt_inner_prod_scalar, celt_pitch_xcorr_scalar, dual_inner_prod_scalar,
        xcorr_kernel_scalar,
    };

    // -- SILK functions (for benchmarks) --
    // Dispatch wrappers:
    pub use crate::silk::float::inner_product_FLP::silk_inner_product_FLP;
    pub use crate::silk::inner_prod_aligned::silk_inner_prod_aligned_scale;
    pub use crate::silk::NSQ::silk_noise_shape_quantizer_short_prediction_c;
    // SIMD dispatch wrapper for short prediction:
    pub use crate::silk::NSQ::silk_noise_shape_quantizer_short_prediction;
    // Scalar implementation:
    pub use crate::silk::float::inner_product_FLP::silk_inner_product_FLP_scalar;

    // -- DNN vec functions (for benchmarks) --
    #[cfg(feature = "deep-plc")]
    pub use crate::dnn::vec::{
        cgemv8x4, cgemv8x4_scalar, sgemv, sgemv_scalar, softmax, softmax_scalar, sparse_cgemv8x4,
        sparse_cgemv8x4_scalar, sparse_sgemv8x4, sparse_sgemv8x4_scalar, vec_sigmoid,
        vec_sigmoid_scalar, vec_tanh, vec_tanh_scalar,
    };
}
