#![allow(non_snake_case)]
#![allow(clippy::missing_safety_doc)]

use ::unsafe_libopus::varargs::VarArgs;
use std::str::FromStr;

pub(crate) trait OpusBackendTrait {
    type Encoder;
    type Decoder;

    unsafe fn opus_encoder_create(
        Fs: i32,
        channels: i32,
        application: i32,
    ) -> Result<Self::Encoder, i32>;
    unsafe fn opus_encoder_ctl_impl(st: &mut Self::Encoder, request: i32, args: VarArgs) -> i32;
    unsafe fn opus_encode(
        st: &mut Self::Encoder,
        pcm: *const i16,
        analysis_frame_size: i32,
        data: *mut u8,
        max_data_bytes: i32,
    ) -> i32;
    unsafe fn opus_encoder_destroy(st: Self::Encoder);

    unsafe fn opus_decoder_create(Fs: i32, channels: i32) -> Result<Self::Decoder, i32>;
    unsafe fn opus_decode(
        st: &mut Self::Decoder,
        data: *const u8,
        len: i32,
        pcm: *mut i16,
        frame_size: i32,
        decode_fec: i32,
    ) -> i32;
    unsafe fn opus_decoder_ctl_impl(st: &mut Self::Decoder, request: i32, args: VarArgs) -> i32;
    unsafe fn opus_decoder_destroy(st: Self::Decoder);
}

mod unsafe_libopus {
    use unsafe_libopus::varargs::{VarArg, VarArgs};
    use unsafe_libopus::{
        opus_encode, Bitrate, OpusDecoder, OpusEncoder, OPUS_GET_FINAL_RANGE_REQUEST,
        OPUS_GET_LOOKAHEAD_REQUEST, OPUS_RESET_STATE, OPUS_SET_BANDWIDTH_REQUEST,
        OPUS_SET_BITRATE_REQUEST, OPUS_SET_COMPLEXITY_REQUEST, OPUS_SET_DTX_REQUEST,
        OPUS_SET_EXPERT_FRAME_DURATION_REQUEST, OPUS_SET_FORCE_CHANNELS_REQUEST,
        OPUS_SET_LSB_DEPTH_REQUEST, OPUS_SET_VBR_CONSTRAINT_REQUEST, OPUS_SET_VBR_REQUEST,
    };

    pub struct RustLibopusBackend;

    impl super::OpusBackendTrait for RustLibopusBackend {
        type Encoder = Box<OpusEncoder>;
        type Decoder = Box<OpusDecoder>;

        unsafe fn opus_encoder_create(
            Fs: i32,
            channels: i32,
            application: i32,
        ) -> Result<Box<OpusEncoder>, i32> {
            OpusEncoder::new(Fs, channels, application).map(Box::new)
        }

        unsafe fn opus_encoder_ctl_impl(
            st: &mut Box<OpusEncoder>,
            request: i32,
            mut args: VarArgs,
        ) -> i32 {
            // Dispatch known CTL requests to safe typed methods.
            // The trait requires VarArgs for compatibility with the upstream C backend,
            // but the Rust backend can use safe methods directly.
            match request {
                OPUS_SET_BITRATE_REQUEST => {
                    if let [VarArg::I32(val)] = &args.0[..] {
                        st.set_bitrate(Bitrate::from(*val));
                        0
                    } else {
                        -1
                    }
                }
                OPUS_SET_BANDWIDTH_REQUEST => {
                    if let [VarArg::I32(val)] = &args.0[..] {
                        let bw = if *val == -1 {
                            None
                        } else {
                            Some((*val).try_into().map_err(|_| -1i32).unwrap())
                        };
                        st.set_bandwidth(bw);
                        0
                    } else {
                        -1
                    }
                }
                OPUS_SET_VBR_REQUEST => {
                    if let [VarArg::I32(val)] = &args.0[..] {
                        st.set_vbr(*val != 0);
                        0
                    } else {
                        -1
                    }
                }
                OPUS_SET_VBR_CONSTRAINT_REQUEST => {
                    if let [VarArg::I32(val)] = &args.0[..] {
                        st.set_vbr_constraint(*val != 0);
                        0
                    } else {
                        -1
                    }
                }
                OPUS_SET_COMPLEXITY_REQUEST => {
                    if let [VarArg::I32(val)] = &args.0[..] {
                        st.set_complexity(*val).map_or_else(|e| e, |_| 0)
                    } else {
                        -1
                    }
                }
                OPUS_SET_FORCE_CHANNELS_REQUEST => {
                    if let [VarArg::I32(val)] = &args.0[..] {
                        let ch = if *val == -1 {
                            None
                        } else {
                            Some((*val).try_into().map_err(|_| -1i32).unwrap())
                        };
                        st.set_force_channels(ch).map_or_else(|e| e, |_| 0)
                    } else {
                        -1
                    }
                }
                OPUS_SET_DTX_REQUEST => {
                    if let [VarArg::I32(val)] = &args.0[..] {
                        st.set_dtx(*val != 0);
                        0
                    } else {
                        -1
                    }
                }
                OPUS_SET_LSB_DEPTH_REQUEST => {
                    if let [VarArg::I32(val)] = &args.0[..] {
                        st.set_lsb_depth(*val).map_or_else(|e| e, |_| 0)
                    } else {
                        -1
                    }
                }
                OPUS_SET_EXPERT_FRAME_DURATION_REQUEST => {
                    if let [VarArg::I32(val)] = &args.0[..] {
                        let fs = (*val).try_into().map_err(|_| -1i32).unwrap();
                        st.set_expert_frame_duration(fs);
                        0
                    } else {
                        -1
                    }
                }
                OPUS_GET_LOOKAHEAD_REQUEST => {
                    if let [VarArg::I32Out(ptr)] = &mut args.0[..] {
                        **ptr = st.lookahead();
                        0
                    } else {
                        -1
                    }
                }
                OPUS_GET_FINAL_RANGE_REQUEST => {
                    if let [VarArg::U32Out(ptr)] = &mut args.0[..] {
                        **ptr = st.final_range();
                        0
                    } else {
                        -1
                    }
                }
                OPUS_RESET_STATE => {
                    st.reset();
                    0
                }
                // Fall back to the unsafe impl for unknown requests
                other => ::unsafe_libopus::opus_encoder_ctl_impl(
                    &mut **st as *mut OpusEncoder,
                    other,
                    args,
                ),
            }
        }

        unsafe fn opus_encode(
            st: &mut Box<OpusEncoder>,
            pcm: *const i16,
            analysis_frame_size: i32,
            data: *mut u8,
            max_data_bytes: i32,
        ) -> i32 {
            opus_encode(
                &mut **st as *mut OpusEncoder,
                pcm,
                analysis_frame_size,
                data,
                max_data_bytes,
            )
        }

        unsafe fn opus_encoder_destroy(st: Box<OpusEncoder>) {
            drop(st)
        }

        unsafe fn opus_decoder_create(Fs: i32, channels: i32) -> Result<Box<OpusDecoder>, i32> {
            OpusDecoder::new(Fs, channels as usize).map(Box::new)
        }

        unsafe fn opus_decode(
            st: &mut Box<OpusDecoder>,
            data: *const u8,
            len: i32,
            pcm: *mut i16,
            frame_size: i32,
            decode_fec: i32,
        ) -> i32 {
            let data = std::slice::from_raw_parts(data, len as usize);
            let pcm = std::slice::from_raw_parts_mut(pcm, (frame_size * st.channels()) as usize);
            st.decode(data, pcm, frame_size, decode_fec != 0)
        }

        unsafe fn opus_decoder_ctl_impl(
            st: &mut Box<OpusDecoder>,
            request: i32,
            mut args: VarArgs,
        ) -> i32 {
            match request {
                OPUS_GET_FINAL_RANGE_REQUEST => {
                    if let [VarArg::U32Out(ptr)] = &mut args.0[..] {
                        **ptr = st.final_range();
                        0
                    } else {
                        -1
                    }
                }
                OPUS_RESET_STATE => {
                    st.reset();
                    0
                }
                // Fall back to the unsafe impl for unknown requests
                other => ::unsafe_libopus::opus_decoder_ctl_impl(st, other, args),
            }
        }

        unsafe fn opus_decoder_destroy(st: Box<OpusDecoder>) {
            drop(st)
        }
    }
}
pub(crate) use unsafe_libopus::RustLibopusBackend;

mod libopus {
    use unsafe_libopus::varargs::{VarArg, VarArgs};
    use upstream_libopus::{
        opus_decode, opus_decoder_create, opus_decoder_ctl, opus_decoder_destroy, opus_encode,
        opus_encoder_create, opus_encoder_ctl, opus_encoder_destroy,
    };
    use upstream_libopus::{OpusDecoder, OpusEncoder};

    pub struct UpstreamLibopusBackend;

    impl super::OpusBackendTrait for UpstreamLibopusBackend {
        type Encoder = *mut OpusEncoder;
        type Decoder = *mut OpusDecoder;

        unsafe fn opus_encoder_create(
            Fs: i32,
            channels: i32,
            application: i32,
        ) -> Result<*mut OpusEncoder, i32> {
            let mut error = 0;

            let res = opus_encoder_create(Fs, channels, application, &mut error);
            if res.is_null() {
                Err(error)
            } else {
                Ok(res)
            }
        }

        unsafe fn opus_encoder_ctl_impl(
            &mut st: &mut *mut OpusEncoder,
            request: i32,
            mut args: VarArgs,
        ) -> i32 {
            match &mut args.0[..] {
                [VarArg::I32(arg)] => opus_encoder_ctl(st, request, *arg),
                [VarArg::I32Out(arg)] => opus_encoder_ctl(st, request, *arg as *mut _),
                [VarArg::U32Out(arg)] => opus_encoder_ctl(st, request, *arg as *mut _),
                // manually match over all required signatures
                _ => todo!("opus_decoder_ctl signature not implemented"),
            }
        }

        unsafe fn opus_encode(
            &mut st: &mut *mut OpusEncoder,
            pcm: *const i16,
            analysis_frame_size: i32,
            data: *mut u8,
            max_data_bytes: i32,
        ) -> i32 {
            opus_encode(st, pcm, analysis_frame_size, data, max_data_bytes)
        }

        unsafe fn opus_encoder_destroy(st: *mut OpusEncoder) {
            opus_encoder_destroy(st)
        }

        unsafe fn opus_decoder_create(Fs: i32, channels: i32) -> Result<*mut OpusDecoder, i32> {
            let mut error = 0;
            let res = opus_decoder_create(Fs, channels, &mut error);
            if res.is_null() {
                Err(error)
            } else {
                Ok(res)
            }
        }

        unsafe fn opus_decode(
            &mut st: &mut *mut OpusDecoder,
            data: *const u8,
            len: i32,
            pcm: *mut i16,
            frame_size: i32,
            decode_fec: i32,
        ) -> i32 {
            opus_decode(st, data, len, pcm, frame_size, decode_fec)
        }

        unsafe fn opus_decoder_ctl_impl(
            &mut st: &mut *mut OpusDecoder,
            request: i32,
            mut args: VarArgs,
        ) -> i32 {
            match &mut args.0[..] {
                // manually match over all required signatures
                [VarArg::U32Out(ptr)] => opus_decoder_ctl(st, request, *ptr as *mut _),
                _ => todo!("opus_decoder_ctl signature not implemented"),
            }
        }

        unsafe fn opus_decoder_destroy(st: *mut OpusDecoder) {
            opus_decoder_destroy(st)
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
