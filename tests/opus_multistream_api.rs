//! Multistream API parity and smoke tests.
//! Requires `--features tools` for upstream C differential checks.

#![cfg(feature = "tools")]

use libopus_sys::{
    opus_multistream_decode, opus_multistream_decoder_create, opus_multistream_decoder_destroy,
    opus_multistream_encoder_create, opus_multistream_encoder_destroy,
};
use opurs::{
    OpusMSDecoder, OpusMSEncoder, OPUS_APPLICATION_AUDIO, OPUS_APPLICATION_VOIP, OPUS_BAD_ARG,
};

#[test]
fn multistream_encoder_constructor_parity_with_c() {
    struct Case {
        channels: i32,
        streams: i32,
        coupled_streams: i32,
        mapping: &'static [u8],
        app: i32,
    }
    let cases = [
        Case {
            channels: 2,
            streams: 1,
            coupled_streams: 1,
            mapping: &[0, 1],
            app: OPUS_APPLICATION_AUDIO,
        },
        Case {
            channels: 2,
            streams: 2,
            coupled_streams: 0,
            mapping: &[0, 1],
            app: OPUS_APPLICATION_VOIP,
        },
        Case {
            channels: 2,
            streams: 1,
            coupled_streams: 2,
            mapping: &[0, 1],
            app: OPUS_APPLICATION_AUDIO,
        },
        Case {
            channels: 2,
            streams: 0,
            coupled_streams: 0,
            mapping: &[0, 1],
            app: OPUS_APPLICATION_AUDIO,
        },
        Case {
            channels: 2,
            streams: 1,
            coupled_streams: 1,
            mapping: &[0, 2],
            app: OPUS_APPLICATION_AUDIO,
        },
        // Encoder-specific invalid shape: streams + coupled_streams > channels.
        Case {
            channels: 2,
            streams: 2,
            coupled_streams: 1,
            mapping: &[0, 1],
            app: OPUS_APPLICATION_AUDIO,
        },
    ];

    for case in cases {
        let rust = OpusMSEncoder::new(
            48000,
            case.channels,
            case.streams,
            case.coupled_streams,
            case.mapping,
            case.app,
        );
        let mut c_error = 0i32;
        let c_ptr = unsafe {
            opus_multistream_encoder_create(
                48000,
                case.channels,
                case.streams,
                case.coupled_streams,
                case.mapping.as_ptr(),
                case.app,
                &mut c_error,
            )
        };
        let c_ok = !c_ptr.is_null();
        if !c_ptr.is_null() {
            unsafe { opus_multistream_encoder_destroy(c_ptr) };
        }
        assert_eq!(
            rust.is_ok(),
            c_ok,
            "encoder create mismatch for channels={}, streams={}, coupled={}, mapping={:?}",
            case.channels,
            case.streams,
            case.coupled_streams,
            case.mapping
        );
        if let Err(err) = rust {
            assert_eq!(err, OPUS_BAD_ARG);
            assert_eq!(c_error, OPUS_BAD_ARG);
        }
    }
}

#[test]
fn multistream_decoder_constructor_parity_with_c() {
    struct Case {
        channels: i32,
        streams: i32,
        coupled_streams: i32,
        mapping: &'static [u8],
    }
    let cases = [
        Case {
            channels: 2,
            streams: 1,
            coupled_streams: 1,
            mapping: &[0, 1],
        },
        Case {
            channels: 2,
            streams: 2,
            coupled_streams: 1,
            mapping: &[0, 1],
        },
        Case {
            channels: 2,
            streams: 1,
            coupled_streams: 2,
            mapping: &[0, 1],
        },
        Case {
            channels: 2,
            streams: 0,
            coupled_streams: 0,
            mapping: &[0, 1],
        },
        Case {
            channels: 2,
            streams: 1,
            coupled_streams: 1,
            mapping: &[0, 2],
        },
    ];

    for case in cases {
        let rust = OpusMSDecoder::new(
            48000,
            case.channels,
            case.streams,
            case.coupled_streams,
            case.mapping,
        );
        let mut c_error = 0i32;
        let c_ptr = unsafe {
            opus_multistream_decoder_create(
                48000,
                case.channels,
                case.streams,
                case.coupled_streams,
                case.mapping.as_ptr(),
                &mut c_error,
            )
        };
        let c_ok = !c_ptr.is_null();
        if !c_ptr.is_null() {
            unsafe { opus_multistream_decoder_destroy(c_ptr) };
        }
        assert_eq!(
            rust.is_ok(),
            c_ok,
            "decoder create mismatch for channels={}, streams={}, coupled={}, mapping={:?}",
            case.channels,
            case.streams,
            case.coupled_streams,
            case.mapping
        );
        if let Err(err) = rust {
            assert_eq!(err, OPUS_BAD_ARG);
            assert_eq!(c_error, OPUS_BAD_ARG);
        }
    }
}

#[test]
fn multistream_roundtrip_two_mono_streams() {
    let mut ms_enc = OpusMSEncoder::new(48000, 2, 2, 0, &[0, 1], OPUS_APPLICATION_AUDIO).unwrap();
    let mut ms_dec = OpusMSDecoder::new(48000, 2, 2, 0, &[0, 1]).unwrap();

    let frame_size = 960usize;
    let mut pcm = vec![0i16; frame_size * 2];
    for i in 0..frame_size {
        pcm[i * 2] = (i as i16).wrapping_mul(31);
        pcm[i * 2 + 1] = (i as i16).wrapping_mul(-17);
    }

    let mut packet = vec![0u8; 4000];
    let packet_len = ms_enc.encode(&pcm, &mut packet);
    assert!(packet_len > 0, "encode failed: {packet_len}");

    let mut out = vec![0i16; frame_size * 2];
    let decoded = ms_dec.decode(
        &packet[..packet_len as usize],
        &mut out,
        frame_size as i32,
        false,
    );
    assert_eq!(decoded, frame_size as i32);
    assert!(
        out.iter().any(|&x| x != 0),
        "decoded output should not be all zeros"
    );
}

#[test]
fn multistream_float_roundtrip_two_mono_streams() {
    let mut ms_enc = OpusMSEncoder::new(48000, 2, 2, 0, &[0, 1], OPUS_APPLICATION_AUDIO).unwrap();
    let mut ms_dec = OpusMSDecoder::new(48000, 2, 2, 0, &[0, 1]).unwrap();

    let frame_size = 960usize;
    let mut pcm = vec![0f32; frame_size * 2];
    for i in 0..frame_size {
        let t = i as f32 / frame_size as f32;
        pcm[i * 2] = (t * core::f32::consts::TAU).sin() * 0.4;
        pcm[i * 2 + 1] = (t * core::f32::consts::PI * 3.0).sin() * 0.2;
    }

    let mut packet = vec![0u8; 4000];
    let packet_len = ms_enc.encode_float(&pcm, &mut packet);
    assert!(packet_len > 0, "encode_float failed: {packet_len}");

    let mut out = vec![0f32; frame_size * 2];
    let decoded = ms_dec.decode_float(
        &packet[..packet_len as usize],
        &mut out,
        frame_size as i32,
        false,
    );
    assert_eq!(decoded, frame_size as i32);
    assert!(
        out.iter().any(|&x| x != 0.0),
        "decoded output should not be all zeros"
    );
}

#[test]
fn multistream_decoder_packet_loss_parity_with_c() {
    let mut rust_dec = OpusMSDecoder::new(48000, 2, 2, 0, &[0, 1]).unwrap();
    let mut rust_out = vec![0i16; 960 * 2];
    let rust_ret = rust_dec.decode(&[], &mut rust_out, 960, false);

    let mut c_error = 0i32;
    let c_ptr = unsafe {
        opus_multistream_decoder_create(48000, 2, 2, 0, [0u8, 1u8].as_ptr(), &mut c_error)
    };
    assert!(!c_ptr.is_null(), "C decoder create failed: {c_error}");
    let mut c_out = vec![0i16; 960 * 2];
    let c_ret =
        unsafe { opus_multistream_decode(c_ptr, core::ptr::null(), 0, c_out.as_mut_ptr(), 960, 0) };
    unsafe { opus_multistream_decoder_destroy(c_ptr) };

    assert_eq!(rust_ret, c_ret, "PLC return mismatch");
}
