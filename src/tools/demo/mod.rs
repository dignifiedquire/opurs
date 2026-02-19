//! This module implements a subset of the `opus_demo` functionality in a library form
//!
//! It can be used to do end-to-end tests of opus encoding and decoding
//!
//! For PCM files, it expects a raw 16-bit signed PCM stream
//!
//! Use it in `ffmpeg` or `ffplay` with `-f s16le -ar <sample_rate> -ac <channels> -i <file>`
//!
//! For opus-compressed files it uses a simplistic mux format. Each opus packet is prefixed with two 32-bit big-endian integers:
//! - packet length in bytes
//! - range coder state (something internal, used as an additional sanity check)
//!
//! Then the opus packet itself follows
//!
//! I am not aware of any tools (except opus_demo from the official distribution) supporting this format

mod backend;
mod input;

pub use self::backend::OpusBackend;
use self::backend::{OpusBackendTrait, RustLibopusBackend, UpstreamLibopusBackend};

pub use input::{
    Application, Bandwidth, Channels, CommonOptions, Complexity, DecodeArgs, DnnOptions,
    EncodeArgs, EncoderOptions, FrameSize, SampleRate,
};

use crate::{opus_strerror, OPUS_AUTO, OPUS_FRAMESIZE_ARG};
use byteorder::{BigEndian, ReadBytesExt, WriteBytesExt};
use std::io::{Cursor, Read, Write};

pub const MAX_PACKET: usize = 1500;
const MAX_FRAME_SIZE: usize = 48000 * 2;

/// Encode an opus stream, like `opus_demo -e`
///
/// See module documentation for the format of input and output data.
pub fn opus_demo_encode(
    backend: OpusBackend,
    data: &[u8],
    args: EncodeArgs,
    dnn: &DnnOptions,
) -> (Vec<u8>, usize) {
    match backend {
        OpusBackend::Rust => opus_demo_encode_impl::<RustLibopusBackend>(data, args, dnn),
        OpusBackend::Upstream => opus_demo_encode_impl::<UpstreamLibopusBackend>(data, args, dnn),
    }
}

fn opus_demo_encode_impl<B: OpusBackendTrait>(
    data: &[u8],
    EncodeArgs {
        sample_rate: sampling_rate,
        channels,
        application,
        bitrate,
        options,
    }: EncodeArgs,
    dnn: &DnnOptions,
) -> (Vec<u8>, usize) {
    let channels: usize = channels.into();

    let mut samples = Vec::new();
    for data in data.chunks_exact(2) {
        samples.push(i16::from_le_bytes(data.try_into().unwrap()));
    }

    let mut enc = B::opus_encoder_create(
        usize::from(sampling_rate) as i32,
        channels as i32,
        application.into_opus(),
    )
    .expect("opus_encoder_create failed");

    if options.common.inbandfec {
        panic!("inbandfec not supported")
    }
    if options.common.loss != 0 {
        panic!("packet loss simulation not supported")
    }

    B::enc_set_bitrate(&mut enc, bitrate as i32);
    B::enc_set_bandwidth(
        &mut enc,
        options.bandwidth.map_or(OPUS_AUTO, |v| v.into_opus()),
    );
    B::enc_set_vbr(&mut enc, !options.cbr as i32);
    B::enc_set_vbr_constraint(&mut enc, options.cvbr as i32);
    B::enc_set_complexity(&mut enc, i32::from(options.complexity));
    B::enc_set_force_channels(&mut enc, if options.forcemono { 1 } else { OPUS_AUTO });
    B::enc_set_dtx(&mut enc, options.dtx as i32);
    B::enc_set_qext(&mut enc, options.qext as i32);
    let skip = B::enc_get_lookahead(&mut enc);
    B::enc_set_lsb_depth(&mut enc, 16);
    B::enc_set_expert_frame_duration(&mut enc, OPUS_FRAMESIZE_ARG);

    // DNN weight loading and DRED configuration
    if options.dred_duration > 0 {
        if let Some(ref path) = dnn.weights_file {
            let blob = std::fs::read(path).expect("failed to read weights file");
            B::enc_set_dnn_blob(&mut enc, &blob).expect("failed to load DNN weights blob");
        } else {
            B::enc_load_dnn_weights(&mut enc).expect("failed to load compiled-in DNN weights");
        }
        B::enc_set_dred_duration(&mut enc, options.dred_duration);
    }

    let frame_size: usize = options.framesize.samples_for_rate(sampling_rate);

    // pad samples with 0s to make it a multiple of frame_size
    let samples_len = samples.len();
    let samples_len = samples_len + (frame_size * channels - (samples_len % frame_size * channels));
    samples.resize(samples_len, 0);

    let mut output = Vec::<u8>::new();

    let mut buffer = vec![0u8; options.max_payload];
    for frame in samples.chunks_exact(frame_size * channels) {
        #[allow(unused)]
        let fpos = output.len();
        #[cfg(feature = "ent-dump")]
        eprintln!("START encoding packet @ 0x{:x}", fpos);

        let res = B::opus_encode(&mut enc, frame, frame_size as i32, &mut buffer);
        if res < 0 {
            panic!("opus_encode failed: {}", opus_strerror(res));
        }
        let data = &buffer[..res as usize];

        let enc_final_range = B::enc_get_final_range(&mut enc);
        #[cfg(feature = "ent-dump")]
        eprintln!("END encoding packet @ 0x{:x}", fpos);

        output.write_i32::<BigEndian>(data.len() as i32).unwrap();
        output.write_u32::<BigEndian>(enc_final_range).unwrap();
        output.write_all(data).unwrap();
    }

    B::opus_encoder_destroy(enc);

    (output, skip as usize)
}

/// Decode an opus stream, like `opus_demo -d`
///
/// See module documentation for the format of input and output data.
pub fn opus_demo_decode(
    backend: OpusBackend,
    data: &[u8],
    args: DecodeArgs,
    dnn: &DnnOptions,
) -> Vec<u8> {
    match backend {
        OpusBackend::Rust => opus_demo_decode_impl::<RustLibopusBackend>(data, args, dnn),
        OpusBackend::Upstream => opus_demo_decode_impl::<UpstreamLibopusBackend>(data, args, dnn),
    }
}

fn opus_demo_decode_impl<B: OpusBackendTrait>(
    data: &[u8],
    DecodeArgs {
        sample_rate,
        channels,
        options,
        complexity,
    }: DecodeArgs,
    dnn: &DnnOptions,
) -> Vec<u8> {
    let mut cursor = Cursor::new(data);
    let len = cursor.get_ref().len();

    let channels: usize = channels.into();

    let mut dec = B::opus_decoder_create(usize::from(sample_rate) as i32, channels as i32)
        .expect("opus_decoder_create failed");

    if options.inbandfec {
        panic!("inbandfec not supported")
    }
    if options.loss != 0 {
        panic!("packet loss simulation not supported")
    }

    // DNN weight loading and decoder complexity
    if let Some(c) = complexity {
        B::dec_set_complexity(&mut dec, i32::from(c));
        if let Some(ref path) = dnn.weights_file {
            let blob = std::fs::read(path).expect("failed to read weights file");
            B::dec_set_dnn_blob(&mut dec, &blob).expect("failed to load DNN weights blob");
        } else {
            B::dec_load_dnn_weights(&mut dec).expect("failed to load compiled-in DNN weights");
        }
    }

    let mut frame_idx = 0;

    let mut data = vec![0u8; MAX_PACKET];
    let mut samples = vec![0i16; MAX_FRAME_SIZE * channels];
    let mut output = Vec::<u8>::new();

    while cursor.position() < len as u64 {
        let data_bytes = cursor.read_u32::<BigEndian>().unwrap();
        let enc_final_range = cursor.read_u32::<BigEndian>().unwrap();

        let data = &mut data[..data_bytes as usize];
        cursor.read_exact(data).unwrap();

        let output_samples = B::opus_decode(&mut dec, data, &mut samples, MAX_FRAME_SIZE as i32, 0);
        if output_samples < 0 {
            panic!("opus_decode failed: {}", opus_strerror(output_samples));
        }
        let samples = &samples[..output_samples as usize * channels];

        let dec_final_range = B::dec_get_final_range(&mut dec);

        assert_eq!(
            enc_final_range, dec_final_range,
            "Range coder state mismatch between encoder and decoder in frame {}",
            frame_idx
        );

        for sample in samples {
            output.extend_from_slice(&sample.to_le_bytes());
        }

        frame_idx += 1;
    }

    B::opus_decoder_destroy(dec);

    output
}

pub fn opus_demo_adjust_length(
    data: &mut Vec<u8>,
    pre_skip_48k: usize,
    orig_bytes_48k: usize,
    sample_rate: SampleRate,
    channels: Channels,
) {
    let sample_rate: usize = sample_rate.into();
    let channels: usize = channels.into();

    let samples_48k_to_current =
        |samples_48k: usize| samples_48k * sample_rate * channels / 48000 / 2;

    data.drain(..2 * samples_48k_to_current(pre_skip_48k));

    let final_len = samples_48k_to_current(orig_bytes_48k);

    // sanity check: the length should not differ more than a one frame of audio
    assert!(
        // two channels & two bytes per sample
        data.len().abs_diff(final_len) < 48000 * 2 * 2 / 50, // the default frame size is 20ms. currently it's the only tested frame size
        "length mismatch: {} vs {}",
        data.len(),
        final_len
    );

    data.resize(final_len, 0);
}
