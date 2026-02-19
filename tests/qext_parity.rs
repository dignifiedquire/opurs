//! QEXT parity checks using the shared tools comparison setup.

#![cfg(all(feature = "tools", feature = "qext"))]

mod test_common;

use opurs::tools::demo::{
    opus_demo_decode, opus_demo_encode, Application, Channels, DecodeArgs, DnnOptions, EncodeArgs,
    EncoderOptions, FrameSize, OpusBackend, SampleRate,
};
use std::panic::{catch_unwind, AssertUnwindSafe};
use test_common::TestRng;

fn deterministic_pcm_bytes(sample_rate: SampleRate, channels: Channels, frames: usize) -> Vec<u8> {
    let mut rng = TestRng::from_iseed(0x5eed_cafe);
    let frame_size = FrameSize::Ms20.samples_for_rate(sample_rate);
    let samples = frame_size * usize::from(channels) * frames;
    let mut out = Vec::with_capacity(samples * 2);

    for _ in 0..samples {
        let s = rng.next_i32() as i16;
        out.extend_from_slice(&s.to_le_bytes());
    }

    out
}

#[test]
fn qext_encode_96k_stereo_does_not_panic() {
    let encode_args = EncodeArgs {
        application: Application::Audio,
        sample_rate: SampleRate::R96000,
        channels: Channels::Stereo,
        bitrate: 128_000,
        options: EncoderOptions {
            framesize: FrameSize::Ms20,
            complexity: opurs::tools::demo::Complexity::C10,
            qext: true,
            ..Default::default()
        },
    };

    let pcm = deterministic_pcm_bytes(encode_args.sample_rate, encode_args.channels, 1);
    let dnn = DnnOptions::default();

    let result = catch_unwind(AssertUnwindSafe(|| {
        let _ = opus_demo_encode(OpusBackend::Rust, &pcm, encode_args, &dnn);
    }));

    assert!(
        result.is_ok(),
        "Rust QEXT encode panicked in shared tools path"
    );
}

#[test]
fn qext_decode_96k_stereo_matches_upstream_when_available() {
    let encode_args = EncodeArgs {
        application: Application::Audio,
        sample_rate: SampleRate::R96000,
        channels: Channels::Stereo,
        bitrate: 128_000,
        options: EncoderOptions {
            framesize: FrameSize::Ms20,
            complexity: opurs::tools::demo::Complexity::C10,
            qext: true,
            ..Default::default()
        },
    };
    let decode_args = DecodeArgs {
        sample_rate: SampleRate::R96000,
        channels: Channels::Stereo,
        options: Default::default(),
        complexity: None,
    };

    let pcm = deterministic_pcm_bytes(encode_args.sample_rate, encode_args.channels, 4);
    let dnn = DnnOptions::default();

    let upstream_encoded = catch_unwind(AssertUnwindSafe(|| {
        opus_demo_encode(OpusBackend::Upstream, &pcm, encode_args, &dnn)
    }));
    let (upstream_encoded, _upstream_skip) = match upstream_encoded {
        Ok(v) => v,
        Err(_) => {
            eprintln!("Skipping QEXT differential parity: upstream 96k QEXT path unavailable");
            return;
        }
    };

    let rust_decoded = opus_demo_decode(OpusBackend::Rust, &upstream_encoded, decode_args, &dnn);
    let upstream_decoded =
        opus_demo_decode(OpusBackend::Upstream, &upstream_encoded, decode_args, &dnn);

    assert_eq!(
        rust_decoded, upstream_decoded,
        "QEXT decode mismatch (Rust vs Upstream) for shared tools-generated packets"
    );
}
