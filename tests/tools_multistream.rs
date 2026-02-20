#![cfg(feature = "tools")]

use opurs::tools::demo::{
    opus_demo_decode_multistream, opus_demo_encode_multistream, Application, CommonOptions,
    EncoderOptions, MultistreamDecodeArgs, MultistreamEncodeArgs, MultistreamLayout, OpusBackend,
    SampleRate,
};

fn build_pcm_bytes(channels: usize, frames: usize) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(channels * frames * 2);
    for i in 0..frames {
        for ch in 0..channels {
            let sample = ((i as i16).wrapping_mul(17)).wrapping_add((ch as i16).wrapping_mul(101));
            bytes.extend_from_slice(&sample.to_le_bytes());
        }
    }
    bytes
}

#[test]
fn tools_multistream_roundtrip_rust_backend() {
    let layout = MultistreamLayout {
        channels: 3,
        streams: 2,
        coupled_streams: 1,
        mapping: vec![0, 1, 2],
    };

    let encode_args = MultistreamEncodeArgs {
        application: Application::Audio,
        sample_rate: SampleRate::R48000,
        layout: layout.clone(),
        bitrate: 64000,
        options: EncoderOptions::default(),
    };
    let input_pcm = build_pcm_bytes(3, 960 * 3);
    let (encoded, _skip) = opus_demo_encode_multistream(OpusBackend::Rust, &input_pcm, encode_args);
    assert!(!encoded.is_empty());

    let decode_args = MultistreamDecodeArgs {
        sample_rate: SampleRate::R48000,
        layout,
        options: CommonOptions::default(),
        complexity: None,
    };
    let decoded = opus_demo_decode_multistream(OpusBackend::Rust, &encoded, decode_args);
    assert!(!decoded.is_empty());
}

#[test]
fn tools_multistream_cross_backend_decode() {
    let layout = MultistreamLayout {
        channels: 3,
        streams: 2,
        coupled_streams: 1,
        mapping: vec![0, 1, 2],
    };

    let encode_args = MultistreamEncodeArgs {
        application: Application::Audio,
        sample_rate: SampleRate::R48000,
        layout: layout.clone(),
        bitrate: 64000,
        options: EncoderOptions::default(),
    };
    let input_pcm = build_pcm_bytes(3, 960 * 2);

    let (encoded_rust, _skip_rust) =
        opus_demo_encode_multistream(OpusBackend::Rust, &input_pcm, encode_args.clone());
    let (encoded_upstream, _skip_upstream) =
        opus_demo_encode_multistream(OpusBackend::Upstream, &input_pcm, encode_args);

    let decode_args = MultistreamDecodeArgs {
        sample_rate: SampleRate::R48000,
        layout: layout.clone(),
        options: CommonOptions::default(),
        complexity: None,
    };

    let rust_from_upstream =
        opus_demo_decode_multistream(OpusBackend::Rust, &encoded_upstream, decode_args.clone());
    let upstream_from_rust =
        opus_demo_decode_multistream(OpusBackend::Upstream, &encoded_rust, decode_args);

    assert!(!rust_from_upstream.is_empty());
    assert!(!upstream_from_rust.is_empty());
}
