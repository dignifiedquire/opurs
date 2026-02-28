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
    parse_multistream_mapping, Application, Bandwidth, Channels, CommonOptions, Complexity,
    DecodeArgs, DnnOptions, EncodeArgs, EncoderOptions, FrameSize, MultistreamDecodeArgs,
    MultistreamEncodeArgs, MultistreamLayout, SampleRate,
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
    let frame_samples = frame_size * channels;
    let pad = (frame_samples - (samples_len % frame_samples)) % frame_samples;
    let samples_len = samples_len + pad;
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
    B::dec_set_ignore_extensions(&mut dec, options.ignore_extensions as i32);

    let dnn_blob = dnn
        .weights_file
        .as_ref()
        .map(|path| std::fs::read(path).expect("failed to read weights file"));

    // DNN weight loading and decoder complexity
    if let Some(c) = complexity {
        B::dec_set_complexity(&mut dec, i32::from(c));
        if let Some(blob) = dnn_blob.as_ref() {
            B::dec_set_dnn_blob(&mut dec, blob).expect("failed to load DNN weights blob");
        } else {
            B::dec_load_dnn_weights(&mut dec).expect("failed to load compiled-in DNN weights");
        }
    }
    #[cfg(feature = "dred")]
    if let Some(blob) = dnn_blob.as_ref() {
        B::dec_set_dnn_blob(&mut dec, blob).expect("failed to load decoder DNN weights blob");
    } else {
        B::dec_load_dnn_weights(&mut dec).expect("failed to load compiled-in decoder DNN weights");
    }

    #[cfg(feature = "dred")]
    let mut dred_state = match (B::dred_decoder_create(), B::dred_alloc()) {
        (Ok(mut dred_dec), Ok(dred)) => {
            if let Some(blob) = dnn_blob.as_ref() {
                B::dred_set_dnn_blob(&mut dred_dec, blob)
                    .expect("failed to load DRED decoder DNN weights blob");
            } else {
                B::dred_load_dnn_weights(&mut dred_dec)
                    .expect("failed to load compiled-in DRED decoder DNN weights");
            }
            Some((dred_dec, dred))
        }
        _ => None,
    };

    let mut frame_idx = 0;
    let mut lost_count = 0i32;
    let mut lost_prev = true;

    let mut packet = vec![0u8; MAX_PACKET];
    let mut samples = vec![0i32; MAX_FRAME_SIZE * channels];
    let mut output = Vec::<u8>::new();

    while cursor.position() < len as u64 {
        let data_bytes = cursor.read_u32::<BigEndian>().unwrap();
        let enc_final_range = cursor.read_u32::<BigEndian>().unwrap();

        if data_bytes as usize > packet.len() {
            packet.resize(data_bytes as usize, 0);
        }
        let packet_slice = &mut packet[..data_bytes as usize];
        cursor.read_exact(packet_slice).unwrap();

        let lost = data_bytes == 0;
        let run_decoder = if lost {
            lost_count += 1;
            0
        } else {
            1 + lost_count
        };

        #[cfg(feature = "dred")]
        let mut dred_input = 0i32;
        #[cfg(feature = "dred")]
        if !lost && lost_count > 0 {
            if let Some((dred_dec, dred)) = dred_state.as_mut() {
                let mut output_samples = B::dec_get_last_packet_duration(&mut dec);
                if output_samples <= 0 {
                    output_samples = (usize::from(sample_rate) as i32 / 50).max(1);
                }
                let max_dred_samples =
                    (usize::from(sample_rate) as i32).min((lost_count * output_samples).max(0));
                let mut dred_end = 0;
                let ret = B::dred_parse(
                    dred_dec,
                    dred,
                    packet_slice,
                    max_dred_samples,
                    usize::from(sample_rate) as i32,
                    &mut dred_end,
                );
                dred_input = if ret > 0 { ret } else { 0 };
            }
        }

        for fr in 0..run_decoder {
            let output_samples = if fr == lost_count - 1 && B::packet_has_lbrr(packet_slice) > 0 {
                // Upstream opus_demo prefers FEC decode for the final lost frame when LBRR is present.
                let mut frame_size = B::dec_get_last_packet_duration(&mut dec);
                if frame_size <= 0 {
                    frame_size = (usize::from(sample_rate) as i32 / 50).max(1);
                }
                B::opus_decode24(&mut dec, packet_slice, &mut samples, frame_size, 1)
            } else if fr < lost_count {
                // Mirror opus_demo loss handling: concealment decode uses
                // last packet duration rather than an oversized frame cap.
                let mut frame_size = B::dec_get_last_packet_duration(&mut dec);
                if frame_size <= 0 {
                    frame_size = (usize::from(sample_rate) as i32 / 50).max(1);
                }
                #[cfg(feature = "dred")]
                {
                    if dred_input > 0 {
                        if let Some((_, dred)) = dred_state.as_ref() {
                            let dred_offset = (lost_count - fr) * frame_size;
                            B::dec_dred_decode24(
                                &mut dec,
                                dred,
                                dred_offset,
                                &mut samples,
                                frame_size,
                            )
                        } else {
                            B::opus_decode24(&mut dec, &[][..], &mut samples, frame_size, 0)
                        }
                    } else {
                        B::opus_decode24(&mut dec, &[][..], &mut samples, frame_size, 0)
                    }
                }
                #[cfg(not(feature = "dred"))]
                {
                    B::opus_decode24(&mut dec, &[][..], &mut samples, frame_size, 0)
                }
            } else {
                B::opus_decode24(
                    &mut dec,
                    packet_slice,
                    &mut samples,
                    MAX_FRAME_SIZE as i32,
                    0,
                )
            };
            if output_samples < 0 {
                panic!("opus_decode failed: {}", opus_strerror(output_samples));
            }
            let decoded = &samples[..output_samples as usize * channels];
            for sample in decoded {
                let s = (*sample).clamp(-0x007fff00, 0x007fff00);
                let s16 = ((s + 128) >> 8) as i16;
                output.extend_from_slice(&s16.to_le_bytes());
            }
        }

        if enc_final_range != 0 && !lost && !lost_prev {
            let dec_final_range = B::dec_get_final_range(&mut dec);
            assert_eq!(
                enc_final_range, dec_final_range,
                "Range coder state mismatch between encoder and decoder in frame {}",
                frame_idx
            );
        }

        frame_idx += 1;
        lost_prev = lost;
        if !lost {
            lost_count = 0;
        }
    }

    #[cfg(feature = "dred")]
    if let Some((dred_dec, dred)) = dred_state {
        B::dred_free(dred);
        B::dred_decoder_destroy(dred_dec);
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

/// Adjust decoded multistream output length after encode+decode roundtrip.
pub fn opus_demo_adjust_length_multistream(
    data: &mut Vec<u8>,
    pre_skip_48k: usize,
    orig_bytes_48k: usize,
    sample_rate: SampleRate,
    channels: i32,
) {
    let sample_rate: usize = sample_rate.into();
    let channels = channels as usize;

    let samples_48k_to_current =
        |samples_48k: usize| samples_48k * sample_rate * channels / 48000 / 2;

    data.drain(..2 * samples_48k_to_current(pre_skip_48k));

    let final_len = samples_48k_to_current(orig_bytes_48k);

    // sanity check: the length should not differ more than a one frame of audio
    assert!(
        data.len().abs_diff(final_len) < 48000 * channels * 2 / 50,
        "length mismatch: {} vs {}",
        data.len(),
        final_len
    );

    data.resize(final_len, 0);
}

/// Encode an Opus multistream stream using the same simple mux format as `opus_demo_encode`.
pub fn opus_demo_encode_multistream(
    backend: OpusBackend,
    data: &[u8],
    args: MultistreamEncodeArgs,
) -> (Vec<u8>, usize) {
    match backend {
        OpusBackend::Rust => opus_demo_encode_multistream_impl::<RustLibopusBackend>(data, args),
        OpusBackend::Upstream => {
            opus_demo_encode_multistream_impl::<UpstreamLibopusBackend>(data, args)
        }
    }
}

/// Decode an Opus multistream stream produced by `opus_demo_encode_multistream`.
pub fn opus_demo_decode_multistream(
    backend: OpusBackend,
    data: &[u8],
    args: MultistreamDecodeArgs,
) -> Vec<u8> {
    match backend {
        OpusBackend::Rust => opus_demo_decode_multistream_impl::<RustLibopusBackend>(data, args),
        OpusBackend::Upstream => {
            opus_demo_decode_multistream_impl::<UpstreamLibopusBackend>(data, args)
        }
    }
}

fn opus_demo_encode_multistream_impl<B: OpusBackendTrait>(
    data: &[u8],
    MultistreamEncodeArgs {
        application,
        sample_rate,
        layout,
        bitrate,
        options,
    }: MultistreamEncodeArgs,
) -> (Vec<u8>, usize) {
    layout.validate().expect("invalid multistream layout");
    let channels = layout.channels as usize;

    if options.common.inbandfec {
        panic!("inbandfec not supported")
    }
    if options.common.loss != 0 {
        panic!("packet loss simulation not supported")
    }

    let mut samples = Vec::new();
    for data in data.chunks_exact(2) {
        samples.push(i16::from_le_bytes(data.try_into().unwrap()));
    }

    let mut enc = B::opus_multistream_encoder_create(
        usize::from(sample_rate) as i32,
        layout.channels,
        layout.streams,
        layout.coupled_streams,
        &layout.mapping,
        application.into_opus(),
    )
    .expect("opus_multistream_encoder_create failed");

    B::ms_enc_set_bitrate(&mut enc, bitrate as i32);
    B::ms_enc_set_bandwidth(
        &mut enc,
        options.bandwidth.map_or(OPUS_AUTO, |v| v.into_opus()),
    );
    B::ms_enc_set_vbr(&mut enc, (!options.cbr) as i32);
    B::ms_enc_set_vbr_constraint(&mut enc, options.cvbr as i32);
    B::ms_enc_set_complexity(&mut enc, i32::from(options.complexity));
    B::ms_enc_set_inband_fec(&mut enc, options.common.inbandfec as i32);
    B::ms_enc_set_packet_loss_perc(&mut enc, options.common.loss as i32);
    B::ms_enc_set_force_channels(&mut enc, if options.forcemono { 1 } else { OPUS_AUTO });
    B::ms_enc_set_dtx(&mut enc, options.dtx as i32);
    B::ms_enc_set_qext(&mut enc, options.qext as i32);
    let skip = B::ms_enc_get_lookahead(&mut enc);

    let frame_size = options.framesize.samples_for_rate(sample_rate);
    let frame_samples = frame_size * channels;
    let pad = (frame_samples - (samples.len() % frame_samples)) % frame_samples;
    samples.resize(samples.len() + pad, 0);

    let mut output = Vec::<u8>::new();
    let mut buffer = vec![0u8; options.max_payload];
    for frame in samples.chunks_exact(frame_size * channels) {
        let res = B::opus_multistream_encode(&mut enc, frame, frame_size as i32, &mut buffer);
        if res < 0 {
            panic!("opus_multistream_encode failed: {}", opus_strerror(res));
        }
        let packet = &buffer[..res as usize];
        let enc_final_range = B::ms_enc_get_final_range(&mut enc);
        output.write_i32::<BigEndian>(packet.len() as i32).unwrap();
        output.write_u32::<BigEndian>(enc_final_range).unwrap();
        output.write_all(packet).unwrap();
    }

    B::opus_multistream_encoder_destroy(enc);

    (output, skip as usize)
}

fn opus_demo_decode_multistream_impl<B: OpusBackendTrait>(
    data: &[u8],
    MultistreamDecodeArgs {
        sample_rate,
        layout,
        options,
        complexity,
    }: MultistreamDecodeArgs,
) -> Vec<u8> {
    layout.validate().expect("invalid multistream layout");
    let mut dec = B::opus_multistream_decoder_create(
        usize::from(sample_rate) as i32,
        layout.channels,
        layout.streams,
        layout.coupled_streams,
        &layout.mapping,
    )
    .expect("opus_multistream_decoder_create failed");
    if let Some(c) = complexity {
        B::ms_dec_set_complexity(&mut dec, i32::from(c));
    }
    B::ms_dec_set_ignore_extensions(&mut dec, options.ignore_extensions as i32);

    let mut cursor = Cursor::new(data);
    let len = cursor.get_ref().len();
    let channels = layout.channels as usize;
    let mut packet = vec![0u8; MAX_PACKET];
    let mut samples = vec![0i16; MAX_FRAME_SIZE * channels];
    let mut output = Vec::<u8>::new();
    while cursor.position() < len as u64 {
        let data_bytes = cursor.read_u32::<BigEndian>().unwrap();
        let _enc_final_range = cursor.read_u32::<BigEndian>().unwrap();
        if data_bytes as usize > packet.len() {
            packet.resize(data_bytes as usize, 0);
        }
        let packet_slice = &mut packet[..data_bytes as usize];
        cursor.read_exact(packet_slice).unwrap();

        let decoded = B::opus_multistream_decode(
            &mut dec,
            packet_slice,
            &mut samples,
            MAX_FRAME_SIZE as i32,
            0,
        );
        if decoded < 0 {
            panic!("opus_multistream_decode failed: {}", opus_strerror(decoded));
        }
        for sample in &samples[..decoded as usize * channels] {
            output.extend_from_slice(&sample.to_le_bytes());
        }
    }
    B::opus_multistream_decoder_destroy(dec);

    output
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::panic::{self, AssertUnwindSafe};

    fn panic_message(err: Box<dyn std::any::Any + Send>) -> String {
        if let Some(s) = err.downcast_ref::<&str>() {
            (*s).to_owned()
        } else if let Some(s) = err.downcast_ref::<String>() {
            s.clone()
        } else {
            "<non-string panic>".to_owned()
        }
    }

    #[test]
    fn decode_large_packet_does_not_panic_on_packet_slice_bounds() {
        let packet_len = MAX_PACKET + 1024;
        let mut stream = Vec::new();
        stream.write_u32::<BigEndian>(packet_len as u32).unwrap();
        stream.write_u32::<BigEndian>(0).unwrap();
        stream.resize(stream.len() + packet_len, 0);

        let args = DecodeArgs {
            sample_rate: SampleRate::R48000,
            channels: Channels::Mono,
            options: CommonOptions::default(),
            complexity: None,
        };
        let dnn = DnnOptions::default();

        let result = panic::catch_unwind(AssertUnwindSafe(|| {
            let _ = opus_demo_decode(OpusBackend::Rust, &stream, args, &dnn);
        }));

        if let Err(err) = result {
            let msg = panic_message(err);
            assert!(
                !msg.contains("out of range for slice"),
                "unexpected slice panic: {msg}"
            );
        }
    }
}
