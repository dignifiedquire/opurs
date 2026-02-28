#![allow(non_snake_case)]

use std::str::FromStr;

pub(crate) trait OpusBackendTrait {
    type Encoder;
    type Decoder;
    type MSEncoder;
    type MSDecoder;
    #[cfg(feature = "dred")]
    type DredDecoder;
    #[cfg(feature = "dred")]
    type DredState;

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
    fn enc_set_qext(st: &mut Self::Encoder, val: i32);
    fn enc_get_lookahead(st: &mut Self::Encoder) -> i32;
    fn enc_get_final_range(st: &mut Self::Encoder) -> u32;
    fn opus_encode(st: &mut Self::Encoder, pcm: &[i16], frame_size: i32, data: &mut [u8]) -> i32;
    fn opus_encoder_destroy(st: Self::Encoder);

    fn enc_set_dred_duration(st: &mut Self::Encoder, val: i32);
    fn enc_load_dnn_weights(st: &mut Self::Encoder) -> Result<(), i32>;
    fn enc_set_dnn_blob(st: &mut Self::Encoder, data: &[u8]) -> Result<(), i32>;

    fn opus_decoder_create(Fs: i32, channels: i32) -> Result<Self::Decoder, i32>;
    fn opus_decode24(
        st: &mut Self::Decoder,
        data: &[u8],
        pcm: &mut [i32],
        frame_size: i32,
        decode_fec: i32,
    ) -> i32;
    fn packet_has_lbrr(data: &[u8]) -> i32;
    fn dec_get_final_range(st: &mut Self::Decoder) -> u32;
    fn dec_get_last_packet_duration(st: &mut Self::Decoder) -> i32;
    fn dec_set_complexity(st: &mut Self::Decoder, val: i32);
    fn dec_set_ignore_extensions(st: &mut Self::Decoder, val: i32);
    fn dec_load_dnn_weights(st: &mut Self::Decoder) -> Result<(), i32>;
    fn dec_set_dnn_blob(st: &mut Self::Decoder, data: &[u8]) -> Result<(), i32>;
    fn opus_decoder_destroy(st: Self::Decoder);

    #[cfg(feature = "dred")]
    fn dred_decoder_create() -> Result<Self::DredDecoder, i32>;
    #[cfg(feature = "dred")]
    fn dred_decoder_destroy(st: Self::DredDecoder);
    #[cfg(feature = "dred")]
    fn dred_alloc() -> Result<Self::DredState, i32>;
    #[cfg(feature = "dred")]
    fn dred_free(st: Self::DredState);
    #[cfg(feature = "dred")]
    fn dred_load_dnn_weights(st: &mut Self::DredDecoder) -> Result<(), i32>;
    #[cfg(feature = "dred")]
    fn dred_set_dnn_blob(st: &mut Self::DredDecoder, data: &[u8]) -> Result<(), i32>;
    #[cfg(feature = "dred")]
    fn dred_parse(
        dred_dec: &mut Self::DredDecoder,
        dred: &mut Self::DredState,
        data: &[u8],
        max_dred_samples: i32,
        sampling_rate: i32,
        dred_end: &mut i32,
    ) -> i32;
    #[cfg(feature = "dred")]
    fn dec_dred_decode24(
        dec: &mut Self::Decoder,
        dred: &Self::DredState,
        dred_offset: i32,
        pcm: &mut [i32],
        frame_size: i32,
    ) -> i32;

    fn opus_multistream_encoder_create(
        Fs: i32,
        channels: i32,
        streams: i32,
        coupled_streams: i32,
        mapping: &[u8],
        application: i32,
    ) -> Result<Self::MSEncoder, i32>;
    fn ms_enc_set_bitrate(st: &mut Self::MSEncoder, val: i32);
    fn ms_enc_set_bandwidth(st: &mut Self::MSEncoder, val: i32);
    fn ms_enc_set_vbr(st: &mut Self::MSEncoder, val: i32);
    fn ms_enc_set_vbr_constraint(st: &mut Self::MSEncoder, val: i32);
    fn ms_enc_set_complexity(st: &mut Self::MSEncoder, val: i32);
    fn ms_enc_set_inband_fec(st: &mut Self::MSEncoder, val: i32);
    fn ms_enc_set_packet_loss_perc(st: &mut Self::MSEncoder, val: i32);
    fn ms_enc_set_force_channels(st: &mut Self::MSEncoder, val: i32);
    fn ms_enc_set_dtx(st: &mut Self::MSEncoder, val: i32);
    fn ms_enc_set_qext(st: &mut Self::MSEncoder, val: i32);
    fn ms_enc_get_lookahead(st: &mut Self::MSEncoder) -> i32;
    fn ms_enc_get_final_range(st: &mut Self::MSEncoder) -> u32;
    fn opus_multistream_encode(
        st: &mut Self::MSEncoder,
        pcm: &[i16],
        frame_size: i32,
        data: &mut [u8],
    ) -> i32;
    fn opus_multistream_encoder_destroy(st: Self::MSEncoder);

    fn opus_multistream_decoder_create(
        Fs: i32,
        channels: i32,
        streams: i32,
        coupled_streams: i32,
        mapping: &[u8],
    ) -> Result<Self::MSDecoder, i32>;
    fn opus_multistream_decode(
        st: &mut Self::MSDecoder,
        data: &[u8],
        pcm: &mut [i16],
        frame_size: i32,
        decode_fec: i32,
    ) -> i32;
    fn ms_dec_set_complexity(st: &mut Self::MSDecoder, val: i32);
    fn ms_dec_set_ignore_extensions(st: &mut Self::MSDecoder, val: i32);
    fn opus_multistream_decoder_destroy(st: Self::MSDecoder);
}

mod rust_backend {
    use crate::{Bitrate, OpusDecoder, OpusEncoder, OpusMSDecoder, OpusMSEncoder};
    #[cfg(feature = "dred")]
    use crate::{OpusDRED, OpusDREDDecoder};

    pub struct RustLibopusBackend;

    impl super::OpusBackendTrait for RustLibopusBackend {
        type Encoder = Box<OpusEncoder>;
        type Decoder = Box<OpusDecoder>;
        type MSEncoder = Box<OpusMSEncoder>;
        type MSDecoder = Box<OpusMSDecoder>;
        #[cfg(feature = "dred")]
        type DredDecoder = OpusDREDDecoder;
        #[cfg(feature = "dred")]
        type DredState = OpusDRED;

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
            let bw = if val == crate::OPUS_AUTO {
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
            let ch = if val == crate::OPUS_AUTO {
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
        fn enc_set_qext(st: &mut Box<OpusEncoder>, val: i32) {
            #[cfg(feature = "qext")]
            st.set_qext(val != 0);
            #[cfg(not(feature = "qext"))]
            {
                let _ = st;
                if val != 0 {
                    panic!("QEXT support requires the 'qext' feature");
                }
            }
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

        fn enc_set_dred_duration(st: &mut Box<OpusEncoder>, val: i32) {
            #[cfg(feature = "dred")]
            st.set_dred_duration(val).unwrap();
            #[cfg(not(feature = "dred"))]
            {
                let _ = st;
                if val != 0 {
                    panic!("DRED support requires the 'dred' feature");
                }
            }
        }

        fn enc_load_dnn_weights(st: &mut Box<OpusEncoder>) -> Result<(), i32> {
            #[cfg(all(feature = "dred", feature = "builtin-weights"))]
            return st.load_dnn_weights();
            #[cfg(not(all(feature = "dred", feature = "builtin-weights")))]
            {
                let _ = st;
                panic!("compiled-in DNN weights require the 'builtin-weights' feature; use --weights <path> instead");
            }
        }

        fn enc_set_dnn_blob(st: &mut Box<OpusEncoder>, data: &[u8]) -> Result<(), i32> {
            #[cfg(feature = "dred")]
            return st.set_dnn_blob(data);
            #[cfg(not(feature = "dred"))]
            {
                let _ = (st, data);
                Err(crate::OPUS_UNIMPLEMENTED)
            }
        }

        fn opus_encoder_destroy(_st: Box<OpusEncoder>) {}

        fn opus_decoder_create(Fs: i32, channels: i32) -> Result<Box<OpusDecoder>, i32> {
            OpusDecoder::new(Fs, channels as usize).map(Box::new)
        }

        fn opus_decode24(
            st: &mut Box<OpusDecoder>,
            data: &[u8],
            pcm: &mut [i32],
            frame_size: i32,
            decode_fec: i32,
        ) -> i32 {
            st.decode24(data, pcm, frame_size, decode_fec != 0)
        }

        fn packet_has_lbrr(data: &[u8]) -> i32 {
            OpusDecoder::packet_has_lbrr(data)
        }

        fn dec_get_final_range(st: &mut Box<OpusDecoder>) -> u32 {
            st.final_range()
        }

        fn dec_get_last_packet_duration(st: &mut Box<OpusDecoder>) -> i32 {
            st.last_packet_duration()
        }

        fn dec_set_complexity(st: &mut Box<OpusDecoder>, val: i32) {
            st.set_complexity(val).unwrap();
        }

        fn dec_set_ignore_extensions(st: &mut Box<OpusDecoder>, val: i32) {
            st.set_ignore_extensions(val != 0);
        }

        fn dec_load_dnn_weights(st: &mut Box<OpusDecoder>) -> Result<(), i32> {
            #[cfg(all(feature = "deep-plc", feature = "builtin-weights"))]
            return st.load_dnn_weights();
            #[cfg(not(all(feature = "deep-plc", feature = "builtin-weights")))]
            {
                let _ = st;
                panic!("compiled-in DNN weights require the 'builtin-weights' feature; use --weights <path> instead");
            }
        }

        fn dec_set_dnn_blob(st: &mut Box<OpusDecoder>, data: &[u8]) -> Result<(), i32> {
            #[cfg(feature = "deep-plc")]
            return st.set_dnn_blob(data);
            #[cfg(not(feature = "deep-plc"))]
            {
                let _ = (st, data);
                Err(crate::OPUS_UNIMPLEMENTED)
            }
        }

        fn opus_decoder_destroy(_st: Box<OpusDecoder>) {}

        #[cfg(feature = "dred")]
        fn dred_decoder_create() -> Result<Self::DredDecoder, i32> {
            crate::opus_dred_decoder_create()
        }

        #[cfg(feature = "dred")]
        fn dred_decoder_destroy(st: Self::DredDecoder) {
            crate::opus_dred_decoder_destroy(st);
        }

        #[cfg(feature = "dred")]
        fn dred_alloc() -> Result<Self::DredState, i32> {
            Ok(crate::opus_dred_alloc())
        }

        #[cfg(feature = "dred")]
        fn dred_free(st: Self::DredState) {
            crate::opus_dred_free(st);
        }

        #[cfg(feature = "dred")]
        fn dred_load_dnn_weights(st: &mut Self::DredDecoder) -> Result<(), i32> {
            #[cfg(feature = "builtin-weights")]
            {
                if st.load_dnn_weights() {
                    Ok(())
                } else {
                    Err(crate::OPUS_UNIMPLEMENTED)
                }
            }
            #[cfg(not(feature = "builtin-weights"))]
            {
                let _ = st;
                Err(crate::OPUS_UNIMPLEMENTED)
            }
        }

        #[cfg(feature = "dred")]
        fn dred_set_dnn_blob(st: &mut Self::DredDecoder, data: &[u8]) -> Result<(), i32> {
            if st.set_dnn_blob(data) {
                Ok(())
            } else {
                Err(crate::OPUS_UNIMPLEMENTED)
            }
        }

        #[cfg(feature = "dred")]
        fn dred_parse(
            dred_dec: &mut Self::DredDecoder,
            dred: &mut Self::DredState,
            data: &[u8],
            max_dred_samples: i32,
            sampling_rate: i32,
            dred_end: &mut i32,
        ) -> i32 {
            crate::opus_dred_parse(
                dred_dec,
                dred,
                data,
                max_dred_samples,
                sampling_rate,
                Some(dred_end),
                false,
            )
        }

        #[cfg(feature = "dred")]
        fn dec_dred_decode24(
            dec: &mut Self::Decoder,
            dred: &Self::DredState,
            dred_offset: i32,
            pcm: &mut [i32],
            frame_size: i32,
        ) -> i32 {
            crate::opus_decoder_dred_decode24(dec, dred, dred_offset, pcm, frame_size)
        }

        fn opus_multistream_encoder_create(
            Fs: i32,
            channels: i32,
            streams: i32,
            coupled_streams: i32,
            mapping: &[u8],
            application: i32,
        ) -> Result<Self::MSEncoder, i32> {
            OpusMSEncoder::new(Fs, channels, streams, coupled_streams, mapping, application)
                .map(Box::new)
        }

        fn ms_enc_set_bitrate(st: &mut Self::MSEncoder, val: i32) {
            st.set_bitrate(Bitrate::from(val));
        }

        fn ms_enc_set_bandwidth(st: &mut Self::MSEncoder, val: i32) {
            let bw = if val == crate::OPUS_AUTO {
                None
            } else {
                Some(val.try_into().unwrap())
            };
            st.set_bandwidth(bw);
        }

        fn ms_enc_set_vbr(st: &mut Self::MSEncoder, val: i32) {
            st.set_vbr(val != 0);
        }

        fn ms_enc_set_vbr_constraint(st: &mut Self::MSEncoder, val: i32) {
            st.set_vbr_constraint(val != 0);
        }

        fn ms_enc_set_complexity(st: &mut Self::MSEncoder, val: i32) {
            st.set_complexity(val).unwrap();
        }

        fn ms_enc_set_inband_fec(st: &mut Self::MSEncoder, val: i32) {
            st.set_inband_fec(val).unwrap();
        }

        fn ms_enc_set_packet_loss_perc(st: &mut Self::MSEncoder, val: i32) {
            st.set_packet_loss_perc(val).unwrap();
        }

        fn ms_enc_set_force_channels(st: &mut Self::MSEncoder, val: i32) {
            let ch = if val == crate::OPUS_AUTO {
                None
            } else {
                Some(val.try_into().unwrap())
            };
            st.set_force_channels(ch).unwrap();
        }

        fn ms_enc_set_dtx(st: &mut Self::MSEncoder, val: i32) {
            st.set_dtx(val != 0);
        }

        fn ms_enc_set_qext(st: &mut Self::MSEncoder, val: i32) {
            #[cfg(feature = "qext")]
            st.set_qext(val != 0);
            #[cfg(not(feature = "qext"))]
            {
                let _ = st;
                if val != 0 {
                    panic!("QEXT support requires the 'qext' feature");
                }
            }
        }

        fn ms_enc_get_lookahead(st: &mut Self::MSEncoder) -> i32 {
            st.lookahead()
        }

        fn ms_enc_get_final_range(st: &mut Self::MSEncoder) -> u32 {
            st.final_range()
        }

        fn opus_multistream_encode(
            st: &mut Self::MSEncoder,
            pcm: &[i16],
            frame_size: i32,
            data: &mut [u8],
        ) -> i32 {
            let frame_samples = frame_size as usize * st.layout().channels() as usize;
            crate::opus_multistream_encode(st, &pcm[..frame_samples], frame_size, data)
        }

        fn opus_multistream_encoder_destroy(_st: Self::MSEncoder) {}

        fn opus_multistream_decoder_create(
            Fs: i32,
            channels: i32,
            streams: i32,
            coupled_streams: i32,
            mapping: &[u8],
        ) -> Result<Self::MSDecoder, i32> {
            OpusMSDecoder::new(Fs, channels, streams, coupled_streams, mapping).map(Box::new)
        }

        fn opus_multistream_decode(
            st: &mut Self::MSDecoder,
            data: &[u8],
            pcm: &mut [i16],
            frame_size: i32,
            decode_fec: i32,
        ) -> i32 {
            crate::opus_multistream_decode(st, data, pcm, frame_size, decode_fec != 0)
        }

        fn ms_dec_set_complexity(st: &mut Self::MSDecoder, val: i32) {
            st.set_complexity(val).unwrap();
        }

        fn ms_dec_set_ignore_extensions(st: &mut Self::MSDecoder, val: i32) {
            st.set_ignore_extensions(val != 0);
        }

        fn opus_multistream_decoder_destroy(_st: Self::MSDecoder) {}
    }
}
pub(crate) use rust_backend::RustLibopusBackend;

mod libopus {
    use libopus_sys::{
        opus_decode24, opus_decoder_create, opus_decoder_ctl, opus_decoder_destroy, opus_encode,
        opus_encoder_create, opus_encoder_ctl, opus_encoder_destroy, opus_multistream_decode,
        opus_multistream_decoder_create, opus_multistream_decoder_ctl,
        opus_multistream_decoder_destroy, opus_multistream_encode, opus_multistream_encoder_create,
        opus_multistream_encoder_ctl, opus_multistream_encoder_destroy, opus_packet_has_lbrr,
    };
    #[cfg(feature = "dred")]
    use libopus_sys::{
        opus_decoder_dred_decode24, opus_dred_alloc, opus_dred_decoder_create,
        opus_dred_decoder_ctl, opus_dred_decoder_destroy, opus_dred_free, opus_dred_parse,
        OpusDRED, OpusDREDDecoder,
    };
    use libopus_sys::{OpusDecoder, OpusEncoder, OpusMSDecoder, OpusMSEncoder};

    use crate::{
        OPUS_GET_FINAL_RANGE_REQUEST, OPUS_GET_LAST_PACKET_DURATION_REQUEST,
        OPUS_GET_LOOKAHEAD_REQUEST, OPUS_SET_BANDWIDTH_REQUEST, OPUS_SET_BITRATE_REQUEST,
        OPUS_SET_COMPLEXITY_REQUEST, OPUS_SET_DNN_BLOB_REQUEST, OPUS_SET_DRED_DURATION_REQUEST,
        OPUS_SET_DTX_REQUEST, OPUS_SET_EXPERT_FRAME_DURATION_REQUEST,
        OPUS_SET_FORCE_CHANNELS_REQUEST, OPUS_SET_INBAND_FEC_REQUEST, OPUS_SET_LSB_DEPTH_REQUEST,
        OPUS_SET_PACKET_LOSS_PERC_REQUEST, OPUS_SET_VBR_CONSTRAINT_REQUEST, OPUS_SET_VBR_REQUEST,
    };
    const OPUS_SET_QEXT_REQUEST: i32 = 4056;
    const OPUS_SET_IGNORE_EXTENSIONS_REQUEST: i32 = 4058;

    pub struct UpstreamLibopusBackend;

    impl super::OpusBackendTrait for UpstreamLibopusBackend {
        type Encoder = *mut OpusEncoder;
        type Decoder = *mut OpusDecoder;
        type MSEncoder = *mut OpusMSEncoder;
        type MSDecoder = *mut OpusMSDecoder;
        #[cfg(feature = "dred")]
        type DredDecoder = *mut OpusDREDDecoder;
        #[cfg(feature = "dred")]
        type DredState = *mut OpusDRED;

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
        fn enc_set_qext(st: &mut *mut OpusEncoder, val: i32) {
            unsafe { opus_encoder_ctl(*st, OPUS_SET_QEXT_REQUEST, val) };
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

        fn enc_set_dred_duration(st: &mut *mut OpusEncoder, val: i32) {
            unsafe { opus_encoder_ctl(*st, OPUS_SET_DRED_DURATION_REQUEST, val) };
        }

        fn enc_load_dnn_weights(_st: &mut *mut OpusEncoder) -> Result<(), i32> {
            // C library has weights compiled in; nothing to load.
            Ok(())
        }

        fn enc_set_dnn_blob(st: &mut *mut OpusEncoder, data: &[u8]) -> Result<(), i32> {
            let ret = unsafe {
                opus_encoder_ctl(
                    *st,
                    OPUS_SET_DNN_BLOB_REQUEST,
                    data.as_ptr(),
                    data.len() as i32,
                )
            };
            if ret < 0 {
                Err(ret)
            } else {
                Ok(())
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

        fn opus_decode24(
            st: &mut *mut OpusDecoder,
            data: &[u8],
            pcm: &mut [i32],
            frame_size: i32,
            decode_fec: i32,
        ) -> i32 {
            unsafe {
                opus_decode24(
                    *st,
                    data.as_ptr(),
                    data.len() as i32,
                    pcm.as_mut_ptr(),
                    frame_size,
                    decode_fec,
                )
            }
        }

        fn packet_has_lbrr(data: &[u8]) -> i32 {
            unsafe { opus_packet_has_lbrr(data.as_ptr(), data.len() as i32) }
        }

        fn dec_get_final_range(st: &mut *mut OpusDecoder) -> u32 {
            let mut val: u32 = 0;
            unsafe { opus_decoder_ctl(*st, OPUS_GET_FINAL_RANGE_REQUEST, &mut val as *mut _) };
            val
        }

        fn dec_get_last_packet_duration(st: &mut *mut OpusDecoder) -> i32 {
            let mut val: i32 = 0;
            unsafe {
                opus_decoder_ctl(
                    *st,
                    OPUS_GET_LAST_PACKET_DURATION_REQUEST,
                    &mut val as *mut _,
                )
            };
            val
        }

        fn dec_set_complexity(st: &mut *mut OpusDecoder, val: i32) {
            unsafe { opus_decoder_ctl(*st, OPUS_SET_COMPLEXITY_REQUEST, val) };
        }

        fn dec_set_ignore_extensions(st: &mut *mut OpusDecoder, val: i32) {
            unsafe { opus_decoder_ctl(*st, OPUS_SET_IGNORE_EXTENSIONS_REQUEST, val) };
        }

        fn dec_load_dnn_weights(_st: &mut *mut OpusDecoder) -> Result<(), i32> {
            // C library has weights compiled in; nothing to load.
            Ok(())
        }

        fn dec_set_dnn_blob(st: &mut *mut OpusDecoder, data: &[u8]) -> Result<(), i32> {
            let ret = unsafe {
                opus_decoder_ctl(
                    *st,
                    OPUS_SET_DNN_BLOB_REQUEST,
                    data.as_ptr(),
                    data.len() as i32,
                )
            };
            if ret < 0 {
                Err(ret)
            } else {
                Ok(())
            }
        }

        fn opus_decoder_destroy(st: *mut OpusDecoder) {
            unsafe { opus_decoder_destroy(st) }
        }

        #[cfg(feature = "dred")]
        fn dred_decoder_create() -> Result<Self::DredDecoder, i32> {
            let mut error = 0;
            let dec = unsafe { opus_dred_decoder_create(&mut error) };
            if dec.is_null() {
                Err(error)
            } else {
                Ok(dec)
            }
        }

        #[cfg(feature = "dred")]
        fn dred_decoder_destroy(st: Self::DredDecoder) {
            unsafe { opus_dred_decoder_destroy(st) }
        }

        #[cfg(feature = "dred")]
        fn dred_alloc() -> Result<Self::DredState, i32> {
            let mut error = 0;
            let dred = unsafe { opus_dred_alloc(&mut error) };
            if dred.is_null() {
                Err(error)
            } else {
                Ok(dred)
            }
        }

        #[cfg(feature = "dred")]
        fn dred_free(st: Self::DredState) {
            unsafe { opus_dred_free(st) }
        }

        #[cfg(feature = "dred")]
        fn dred_load_dnn_weights(_st: &mut Self::DredDecoder) -> Result<(), i32> {
            // C library has weights compiled in; nothing to load.
            Ok(())
        }

        #[cfg(feature = "dred")]
        fn dred_set_dnn_blob(st: &mut Self::DredDecoder, data: &[u8]) -> Result<(), i32> {
            let ret = unsafe {
                opus_dred_decoder_ctl(
                    *st,
                    OPUS_SET_DNN_BLOB_REQUEST,
                    data.as_ptr(),
                    data.len() as i32,
                )
            };
            if ret < 0 {
                Err(ret)
            } else {
                Ok(())
            }
        }

        #[cfg(feature = "dred")]
        fn dred_parse(
            dred_dec: &mut Self::DredDecoder,
            dred: &mut Self::DredState,
            data: &[u8],
            max_dred_samples: i32,
            sampling_rate: i32,
            dred_end: &mut i32,
        ) -> i32 {
            unsafe {
                opus_dred_parse(
                    *dred_dec,
                    *dred,
                    data.as_ptr(),
                    data.len() as i32,
                    max_dred_samples,
                    sampling_rate,
                    dred_end as *mut _,
                    0,
                )
            }
        }

        #[cfg(feature = "dred")]
        fn dec_dred_decode24(
            dec: &mut Self::Decoder,
            dred: &Self::DredState,
            dred_offset: i32,
            pcm: &mut [i32],
            frame_size: i32,
        ) -> i32 {
            unsafe {
                opus_decoder_dred_decode24(*dec, *dred, dred_offset, pcm.as_mut_ptr(), frame_size)
            }
        }

        fn opus_multistream_encoder_create(
            Fs: i32,
            channels: i32,
            streams: i32,
            coupled_streams: i32,
            mapping: &[u8],
            application: i32,
        ) -> Result<Self::MSEncoder, i32> {
            let mut error = 0;
            let res = unsafe {
                opus_multistream_encoder_create(
                    Fs,
                    channels,
                    streams,
                    coupled_streams,
                    mapping.as_ptr(),
                    application,
                    &mut error,
                )
            };
            if res.is_null() {
                Err(error)
            } else {
                Ok(res)
            }
        }

        fn ms_enc_set_bitrate(st: &mut Self::MSEncoder, val: i32) {
            unsafe { opus_multistream_encoder_ctl(*st, OPUS_SET_BITRATE_REQUEST, val) };
        }

        fn ms_enc_set_bandwidth(st: &mut Self::MSEncoder, val: i32) {
            unsafe { opus_multistream_encoder_ctl(*st, OPUS_SET_BANDWIDTH_REQUEST, val) };
        }

        fn ms_enc_set_vbr(st: &mut Self::MSEncoder, val: i32) {
            unsafe { opus_multistream_encoder_ctl(*st, OPUS_SET_VBR_REQUEST, val) };
        }

        fn ms_enc_set_vbr_constraint(st: &mut Self::MSEncoder, val: i32) {
            unsafe { opus_multistream_encoder_ctl(*st, OPUS_SET_VBR_CONSTRAINT_REQUEST, val) };
        }

        fn ms_enc_set_complexity(st: &mut Self::MSEncoder, val: i32) {
            unsafe { opus_multistream_encoder_ctl(*st, OPUS_SET_COMPLEXITY_REQUEST, val) };
        }

        fn ms_enc_set_inband_fec(st: &mut Self::MSEncoder, val: i32) {
            unsafe { opus_multistream_encoder_ctl(*st, OPUS_SET_INBAND_FEC_REQUEST, val) };
        }

        fn ms_enc_set_packet_loss_perc(st: &mut Self::MSEncoder, val: i32) {
            unsafe { opus_multistream_encoder_ctl(*st, OPUS_SET_PACKET_LOSS_PERC_REQUEST, val) };
        }

        fn ms_enc_set_force_channels(st: &mut Self::MSEncoder, val: i32) {
            unsafe { opus_multistream_encoder_ctl(*st, OPUS_SET_FORCE_CHANNELS_REQUEST, val) };
        }

        fn ms_enc_set_dtx(st: &mut Self::MSEncoder, val: i32) {
            unsafe { opus_multistream_encoder_ctl(*st, OPUS_SET_DTX_REQUEST, val) };
        }

        fn ms_enc_set_qext(st: &mut Self::MSEncoder, val: i32) {
            unsafe { opus_multistream_encoder_ctl(*st, OPUS_SET_QEXT_REQUEST, val) };
        }

        fn ms_enc_get_lookahead(st: &mut Self::MSEncoder) -> i32 {
            let mut val: i32 = 0;
            unsafe {
                opus_multistream_encoder_ctl(*st, OPUS_GET_LOOKAHEAD_REQUEST, &mut val as *mut _)
            };
            val
        }

        fn ms_enc_get_final_range(st: &mut Self::MSEncoder) -> u32 {
            let mut val: u32 = 0;
            unsafe {
                opus_multistream_encoder_ctl(*st, OPUS_GET_FINAL_RANGE_REQUEST, &mut val as *mut _)
            };
            val
        }

        fn opus_multistream_encode(
            st: &mut Self::MSEncoder,
            pcm: &[i16],
            frame_size: i32,
            data: &mut [u8],
        ) -> i32 {
            unsafe {
                opus_multistream_encode(
                    *st,
                    pcm.as_ptr(),
                    frame_size,
                    data.as_mut_ptr(),
                    data.len() as i32,
                )
            }
        }

        fn opus_multistream_encoder_destroy(st: Self::MSEncoder) {
            unsafe { opus_multistream_encoder_destroy(st) }
        }

        fn opus_multistream_decoder_create(
            Fs: i32,
            channels: i32,
            streams: i32,
            coupled_streams: i32,
            mapping: &[u8],
        ) -> Result<Self::MSDecoder, i32> {
            let mut error = 0;
            let res = unsafe {
                opus_multistream_decoder_create(
                    Fs,
                    channels,
                    streams,
                    coupled_streams,
                    mapping.as_ptr(),
                    &mut error,
                )
            };
            if res.is_null() {
                Err(error)
            } else {
                Ok(res)
            }
        }

        fn opus_multistream_decode(
            st: &mut Self::MSDecoder,
            data: &[u8],
            pcm: &mut [i16],
            frame_size: i32,
            decode_fec: i32,
        ) -> i32 {
            unsafe {
                opus_multistream_decode(
                    *st,
                    data.as_ptr(),
                    data.len() as i32,
                    pcm.as_mut_ptr(),
                    frame_size,
                    decode_fec,
                )
            }
        }

        fn ms_dec_set_complexity(st: &mut Self::MSDecoder, val: i32) {
            unsafe { opus_multistream_decoder_ctl(*st, OPUS_SET_COMPLEXITY_REQUEST, val) };
        }

        fn ms_dec_set_ignore_extensions(st: &mut Self::MSDecoder, val: i32) {
            unsafe { opus_multistream_decoder_ctl(*st, OPUS_SET_IGNORE_EXTENSIONS_REQUEST, val) };
        }

        fn opus_multistream_decoder_destroy(st: Self::MSDecoder) {
            unsafe { opus_multistream_decoder_destroy(st) }
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
