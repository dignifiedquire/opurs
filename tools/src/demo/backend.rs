#![allow(non_snake_case)]
#![allow(clippy::missing_safety_doc)]

use std::str::FromStr;

pub(crate) trait OpusBackendTrait {
    type Encoder;
    type Decoder;

    fn opus_encoder_create(Fs: i32, channels: i32, application: i32) -> Result<Self::Encoder, i32>;
    fn enc_set_bitrate(st: &mut Self::Encoder, val: i32);
    fn enc_set_bandwidth(st: &mut Self::Encoder, val: i32);
    fn enc_set_vbr(st: &mut Self::Encoder, val: i32);
    fn enc_set_vbr_constraint(st: &mut Self::Encoder, val: i32);
    fn enc_set_complexity(st: &mut Self::Encoder, val: i32);
    fn enc_set_force_channels(st: &mut Self::Encoder, val: i32);
    fn enc_set_dtx(st: &mut Self::Encoder, val: i32);
    fn enc_set_lsb_depth(st: &mut Self::Encoder, val: i32);
    fn enc_set_expert_frame_duration(st: &mut Self::Encoder, val: i32);
    fn enc_get_lookahead(st: &mut Self::Encoder) -> i32;
    fn enc_get_final_range(st: &mut Self::Encoder) -> u32;
    fn opus_encode(st: &mut Self::Encoder, pcm: &[i16], frame_size: i32, data: &mut [u8]) -> i32;
    fn opus_encoder_destroy(st: Self::Encoder);

    fn opus_decoder_create(Fs: i32, channels: i32) -> Result<Self::Decoder, i32>;
    fn opus_decode(
        st: &mut Self::Decoder,
        data: &[u8],
        pcm: &mut [i16],
        frame_size: i32,
        decode_fec: i32,
    ) -> i32;
    fn dec_get_final_range(st: &mut Self::Decoder) -> u32;
    fn opus_decoder_destroy(st: Self::Decoder);
}

mod opurs {
    use opurs::{Bitrate, OpusDecoder, OpusEncoder};

    pub struct RustLibopusBackend;

    impl super::OpusBackendTrait for RustLibopusBackend {
        type Encoder = Box<OpusEncoder>;
        type Decoder = Box<OpusDecoder>;

        fn opus_encoder_create(
            Fs: i32,
            channels: i32,
            application: i32,
        ) -> Result<Box<OpusEncoder>, i32> {
            OpusEncoder::new(Fs, channels, application).map(Box::new)
        }

        fn enc_set_bitrate(st: &mut Box<OpusEncoder>, val: i32) {
            st.set_bitrate(Bitrate::from(val));
        }
        fn enc_set_bandwidth(st: &mut Box<OpusEncoder>, val: i32) {
            let bw = if val == ::opurs::OPUS_AUTO {
                None
            } else {
                Some(val.try_into().unwrap())
            };
            st.set_bandwidth(bw);
        }
        fn enc_set_vbr(st: &mut Box<OpusEncoder>, val: i32) {
            st.set_vbr(val != 0);
        }
        fn enc_set_vbr_constraint(st: &mut Box<OpusEncoder>, val: i32) {
            st.set_vbr_constraint(val != 0);
        }
        fn enc_set_complexity(st: &mut Box<OpusEncoder>, val: i32) {
            st.set_complexity(val).unwrap();
        }
        fn enc_set_force_channels(st: &mut Box<OpusEncoder>, val: i32) {
            let ch = if val == ::opurs::OPUS_AUTO {
                None
            } else {
                Some(val.try_into().unwrap())
            };
            st.set_force_channels(ch).unwrap();
        }
        fn enc_set_dtx(st: &mut Box<OpusEncoder>, val: i32) {
            st.set_dtx(val != 0);
        }
        fn enc_set_lsb_depth(st: &mut Box<OpusEncoder>, val: i32) {
            st.set_lsb_depth(val).unwrap();
        }
        fn enc_set_expert_frame_duration(st: &mut Box<OpusEncoder>, val: i32) {
            st.set_expert_frame_duration(val.try_into().unwrap());
        }
        fn enc_get_lookahead(st: &mut Box<OpusEncoder>) -> i32 {
            st.lookahead()
        }
        fn enc_get_final_range(st: &mut Box<OpusEncoder>) -> u32 {
            st.final_range()
        }

        fn opus_encode(
            st: &mut Box<OpusEncoder>,
            pcm: &[i16],
            frame_size: i32,
            data: &mut [u8],
        ) -> i32 {
            st.encode(&pcm[..(frame_size as usize * st.channels() as usize)], data)
        }

        fn opus_encoder_destroy(_st: Box<OpusEncoder>) {}

        fn opus_decoder_create(Fs: i32, channels: i32) -> Result<Box<OpusDecoder>, i32> {
            OpusDecoder::new(Fs, channels as usize).map(Box::new)
        }

        fn opus_decode(
            st: &mut Box<OpusDecoder>,
            data: &[u8],
            pcm: &mut [i16],
            frame_size: i32,
            decode_fec: i32,
        ) -> i32 {
            st.decode(data, pcm, frame_size, decode_fec != 0)
        }

        fn dec_get_final_range(st: &mut Box<OpusDecoder>) -> u32 {
            st.final_range()
        }

        fn opus_decoder_destroy(_st: Box<OpusDecoder>) {}
    }
}
pub(crate) use opurs::RustLibopusBackend;

mod libopus {
    use libopus_sys::{
        opus_decode, opus_decoder_create, opus_decoder_ctl, opus_decoder_destroy, opus_encode,
        opus_encoder_create, opus_encoder_ctl, opus_encoder_destroy,
    };
    use libopus_sys::{OpusDecoder, OpusEncoder};

    use ::opurs::{
        OPUS_GET_FINAL_RANGE_REQUEST, OPUS_GET_LOOKAHEAD_REQUEST, OPUS_SET_BANDWIDTH_REQUEST,
        OPUS_SET_BITRATE_REQUEST, OPUS_SET_COMPLEXITY_REQUEST, OPUS_SET_DTX_REQUEST,
        OPUS_SET_EXPERT_FRAME_DURATION_REQUEST, OPUS_SET_FORCE_CHANNELS_REQUEST,
        OPUS_SET_LSB_DEPTH_REQUEST, OPUS_SET_VBR_CONSTRAINT_REQUEST, OPUS_SET_VBR_REQUEST,
    };

    pub struct UpstreamLibopusBackend;

    impl super::OpusBackendTrait for UpstreamLibopusBackend {
        type Encoder = *mut OpusEncoder;
        type Decoder = *mut OpusDecoder;

        fn opus_encoder_create(
            Fs: i32,
            channels: i32,
            application: i32,
        ) -> Result<*mut OpusEncoder, i32> {
            let mut error = 0;
            let res = unsafe { opus_encoder_create(Fs, channels, application, &mut error) };
            if res.is_null() {
                Err(error)
            } else {
                Ok(res)
            }
        }

        fn enc_set_bitrate(st: &mut *mut OpusEncoder, val: i32) {
            unsafe { opus_encoder_ctl(*st, OPUS_SET_BITRATE_REQUEST, val) };
        }
        fn enc_set_bandwidth(st: &mut *mut OpusEncoder, val: i32) {
            unsafe { opus_encoder_ctl(*st, OPUS_SET_BANDWIDTH_REQUEST, val) };
        }
        fn enc_set_vbr(st: &mut *mut OpusEncoder, val: i32) {
            unsafe { opus_encoder_ctl(*st, OPUS_SET_VBR_REQUEST, val) };
        }
        fn enc_set_vbr_constraint(st: &mut *mut OpusEncoder, val: i32) {
            unsafe { opus_encoder_ctl(*st, OPUS_SET_VBR_CONSTRAINT_REQUEST, val) };
        }
        fn enc_set_complexity(st: &mut *mut OpusEncoder, val: i32) {
            unsafe { opus_encoder_ctl(*st, OPUS_SET_COMPLEXITY_REQUEST, val) };
        }
        fn enc_set_force_channels(st: &mut *mut OpusEncoder, val: i32) {
            unsafe { opus_encoder_ctl(*st, OPUS_SET_FORCE_CHANNELS_REQUEST, val) };
        }
        fn enc_set_dtx(st: &mut *mut OpusEncoder, val: i32) {
            unsafe { opus_encoder_ctl(*st, OPUS_SET_DTX_REQUEST, val) };
        }
        fn enc_set_lsb_depth(st: &mut *mut OpusEncoder, val: i32) {
            unsafe { opus_encoder_ctl(*st, OPUS_SET_LSB_DEPTH_REQUEST, val) };
        }
        fn enc_set_expert_frame_duration(st: &mut *mut OpusEncoder, val: i32) {
            unsafe { opus_encoder_ctl(*st, OPUS_SET_EXPERT_FRAME_DURATION_REQUEST, val) };
        }
        fn enc_get_lookahead(st: &mut *mut OpusEncoder) -> i32 {
            let mut val: i32 = 0;
            unsafe { opus_encoder_ctl(*st, OPUS_GET_LOOKAHEAD_REQUEST, &mut val as *mut _) };
            val
        }
        fn enc_get_final_range(st: &mut *mut OpusEncoder) -> u32 {
            let mut val: u32 = 0;
            unsafe { opus_encoder_ctl(*st, OPUS_GET_FINAL_RANGE_REQUEST, &mut val as *mut _) };
            val
        }

        fn opus_encode(
            st: &mut *mut OpusEncoder,
            pcm: &[i16],
            frame_size: i32,
            data: &mut [u8],
        ) -> i32 {
            unsafe {
                opus_encode(
                    *st,
                    pcm.as_ptr(),
                    frame_size,
                    data.as_mut_ptr(),
                    data.len() as i32,
                )
            }
        }

        fn opus_encoder_destroy(st: *mut OpusEncoder) {
            unsafe { opus_encoder_destroy(st) }
        }

        fn opus_decoder_create(Fs: i32, channels: i32) -> Result<*mut OpusDecoder, i32> {
            let mut error = 0;
            let res = unsafe { opus_decoder_create(Fs, channels, &mut error) };
            if res.is_null() {
                Err(error)
            } else {
                Ok(res)
            }
        }

        fn opus_decode(
            st: &mut *mut OpusDecoder,
            data: &[u8],
            pcm: &mut [i16],
            frame_size: i32,
            decode_fec: i32,
        ) -> i32 {
            unsafe {
                opus_decode(
                    *st,
                    data.as_ptr(),
                    data.len() as i32,
                    pcm.as_mut_ptr(),
                    frame_size,
                    decode_fec,
                )
            }
        }

        fn dec_get_final_range(st: &mut *mut OpusDecoder) -> u32 {
            let mut val: u32 = 0;
            unsafe { opus_decoder_ctl(*st, OPUS_GET_FINAL_RANGE_REQUEST, &mut val as *mut _) };
            val
        }

        fn opus_decoder_destroy(st: *mut OpusDecoder) {
            unsafe { opus_decoder_destroy(st) }
        }
    }
}
pub(crate) use libopus::UpstreamLibopusBackend;

#[derive(Debug, Copy, Clone, Default)]
pub enum OpusBackend {
    #[default]
    Rust,
    Upstream,
}

impl FromStr for OpusBackend {
    type Err = &'static str;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "unsafe" => Ok(OpusBackend::Rust),
            "upstream" => Ok(OpusBackend::Upstream),
            _ => Err("Invalid backend"),
        }
    }
}
