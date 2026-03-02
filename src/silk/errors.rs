//! Shared SILK error codes.
//!
//! These constants mirror the encoder/decoder error values used across:
//! - `silk/enc_API.c`
//! - `silk/check_control_input.c`
//! - `silk/control_codec.c`
//! - `silk/dec_API.c`

/// No error.
pub const SILK_NO_ERROR: i32 = 0;

// Encoder errors.
pub const SILK_ENC_INPUT_INVALID_NO_OF_SAMPLES: i32 = -(101);
pub const SILK_ENC_FS_NOT_SUPPORTED: i32 = -(102);
pub const SILK_ENC_PACKET_SIZE_NOT_SUPPORTED: i32 = -(103);
pub const SILK_ENC_INVALID_LOSS_RATE: i32 = -(105);
pub const SILK_ENC_INVALID_COMPLEXITY_SETTING: i32 = -(106);
pub const SILK_ENC_INVALID_INBAND_FEC_SETTING: i32 = -(107);
pub const SILK_ENC_INVALID_DTX_SETTING: i32 = -(108);
pub const SILK_ENC_INVALID_CBR_SETTING: i32 = -(109);
pub const SILK_ENC_INVALID_NUMBER_OF_CHANNELS_ERROR: i32 = -(111);

// Decoder errors.
pub const SILK_DEC_INVALID_SAMPLING_FREQUENCY: i32 = -(200);
pub const SILK_DEC_INVALID_FRAME_SIZE: i32 = -(203);
