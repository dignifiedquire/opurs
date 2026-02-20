//! Multistream API parity and smoke tests.
//! Requires `--features tools` for upstream C differential checks.

#![cfg(feature = "tools")]

use libopus_sys::{
    opus_multistream_decode, opus_multistream_decoder_create, opus_multistream_decoder_destroy,
    opus_multistream_decoder_get_size, opus_multistream_encoder_create,
    opus_multistream_encoder_destroy, opus_multistream_encoder_get_size,
};
use opurs::{
    opus_multistream_decode as rust_opus_multistream_decode,
    opus_multistream_decode_float as rust_opus_multistream_decode_float,
    opus_multistream_decoder_create as rust_opus_multistream_decoder_create,
    opus_multistream_decoder_init as rust_opus_multistream_decoder_init,
    opus_multistream_encode as rust_opus_multistream_encode,
    opus_multistream_encode_float as rust_opus_multistream_encode_float,
    opus_multistream_encoder_create as rust_opus_multistream_encoder_create,
    opus_multistream_encoder_init as rust_opus_multistream_encoder_init, Bitrate, OpusMSDecoder,
    OpusMSEncoder, OPUS_APPLICATION_AUDIO, OPUS_APPLICATION_VOIP, OPUS_BAD_ARG,
    OPUS_GET_COMPLEXITY_REQUEST, OPUS_GET_GAIN_REQUEST, OPUS_GET_INBAND_FEC_REQUEST,
    OPUS_GET_PACKET_LOSS_PERC_REQUEST, OPUS_GET_VBR_CONSTRAINT_REQUEST, OPUS_GET_VBR_REQUEST,
    OPUS_SET_COMPLEXITY_REQUEST, OPUS_SET_GAIN_REQUEST, OPUS_SET_INBAND_FEC_REQUEST,
    OPUS_SET_PACKET_LOSS_PERC_REQUEST, OPUS_SET_VBR_CONSTRAINT_REQUEST, OPUS_SET_VBR_REQUEST,
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

#[test]
fn multistream_get_size_zero_nonzero_parity() {
    let cases = [(1, 0), (2, 1), (0, 0), (1, 2), (300, 0)];
    for (streams, coupled) in cases {
        let rust_enc = OpusMSEncoder::get_size(streams, coupled);
        let rust_dec = OpusMSDecoder::get_size(streams, coupled);
        let c_enc = unsafe { opus_multistream_encoder_get_size(streams, coupled) };
        let c_dec = unsafe { opus_multistream_decoder_get_size(streams, coupled) };
        assert_eq!(rust_enc == 0, c_enc == 0, "encoder size validity mismatch");
        assert_eq!(rust_dec == 0, c_dec == 0, "decoder size validity mismatch");
    }
}

#[test]
fn multistream_decode_invalid_packet_parity_with_c() {
    let mut rust_dec = OpusMSDecoder::new(48000, 2, 2, 0, &[0, 1]).unwrap();
    let mut rust_out = vec![0i16; 960 * 2];
    let bad_packet = [0xffu8];
    let rust_ret = rust_dec.decode(&bad_packet, &mut rust_out, 960, false);

    let mut c_error = 0i32;
    let c_ptr = unsafe {
        opus_multistream_decoder_create(48000, 2, 2, 0, [0u8, 1u8].as_ptr(), &mut c_error)
    };
    assert!(!c_ptr.is_null(), "C decoder create failed: {c_error}");
    let mut c_out = vec![0i16; 960 * 2];
    let c_ret = unsafe {
        opus_multistream_decode(
            c_ptr,
            bad_packet.as_ptr(),
            bad_packet.len() as i32,
            c_out.as_mut_ptr(),
            960,
            0,
        )
    };
    unsafe { opus_multistream_decoder_destroy(c_ptr) };

    assert_eq!(rust_ret, c_ret, "invalid-packet decode mismatch");
}

#[test]
fn multistream_wrapper_entrypoints_smoke() {
    let mut enc =
        rust_opus_multistream_encoder_create(48000, 2, 2, 0, &[0, 1], OPUS_APPLICATION_AUDIO)
            .expect("encoder create");
    let mut dec =
        rust_opus_multistream_decoder_create(48000, 2, 2, 0, &[0, 1]).expect("decoder create");

    let frame_size = 960usize;
    let mut pcm_i16 = vec![0i16; frame_size * 2];
    let mut pcm_f32 = vec![0f32; frame_size * 2];
    for i in 0..frame_size {
        pcm_i16[i * 2] = (i as i16).wrapping_mul(9);
        pcm_i16[i * 2 + 1] = (i as i16).wrapping_mul(-11);
        pcm_f32[i * 2] = i as f32 / frame_size as f32;
        pcm_f32[i * 2 + 1] = -(i as f32 / frame_size as f32);
    }

    let mut packet = vec![0u8; 4000];
    let len_i16 = rust_opus_multistream_encode(&mut enc, &pcm_i16, frame_size as i32, &mut packet);
    assert!(len_i16 > 0);

    let mut out_i16 = vec![0i16; frame_size * 2];
    let dec_i16 = rust_opus_multistream_decode(
        &mut dec,
        &packet[..len_i16 as usize],
        &mut out_i16,
        frame_size as i32,
        false,
    );
    assert_eq!(dec_i16, frame_size as i32);

    let len_f32 =
        rust_opus_multistream_encode_float(&mut enc, &pcm_f32, frame_size as i32, &mut packet);
    assert!(len_f32 > 0);

    let mut out_f32 = vec![0f32; frame_size * 2];
    let dec_f32 = rust_opus_multistream_decode_float(
        &mut dec,
        &packet[..len_f32 as usize],
        &mut out_f32,
        frame_size as i32,
        false,
    );
    assert_eq!(dec_f32, frame_size as i32);
}

#[test]
fn multistream_decoder_mapping_255_outputs_silence() {
    // One encoded stream, two output channels; second channel is unmapped (255).
    let mut enc = OpusMSEncoder::new(48000, 1, 1, 0, &[0], OPUS_APPLICATION_AUDIO).unwrap();
    let mut dec = OpusMSDecoder::new(48000, 2, 1, 0, &[0, 255]).unwrap();

    let frame_size = 960usize;
    let mut pcm = vec![0i16; frame_size];
    for (i, s) in pcm.iter_mut().enumerate() {
        *s = (i as i16).wrapping_mul(13);
    }
    let mut packet = vec![0u8; 2000];
    let len = enc.encode(&pcm, &mut packet);
    assert!(len > 0);

    let mut out = vec![1i16; frame_size * 2];
    let decoded = dec.decode(&packet[..len as usize], &mut out, frame_size as i32, false);
    assert_eq!(decoded, frame_size as i32);
    for i in 0..frame_size {
        assert_eq!(out[i * 2 + 1], 0, "channel mapped to 255 must be silent");
    }
}

#[test]
fn multistream_encoder_ctl_propagation_smoke() {
    let mut enc = OpusMSEncoder::new(48000, 2, 2, 0, &[0, 1], OPUS_APPLICATION_AUDIO).unwrap();
    enc.set_bitrate(Bitrate::Bits(64000));
    enc.set_vbr(true);
    enc.set_vbr_constraint(true);
    enc.set_complexity(7).unwrap();
    enc.set_inband_fec(1).unwrap();
    enc.set_packet_loss_perc(9).unwrap();

    assert!(enc.bitrate() > 0);
    assert!(enc.vbr());
    assert!(enc.vbr_constraint());
    assert_eq!(enc.complexity(), 7);
    assert_eq!(enc.inband_fec(), 1);
    assert_eq!(enc.packet_loss_perc(), 9);
}

#[test]
fn multistream_constructor_matrix_parity_sampled() {
    // Deterministic sampled matrix over small channel counts.
    for channels in 1..=6 {
        for streams in 0..=channels + 1 {
            for coupled in 0..=streams + 1 {
                let mut mappings: Vec<Vec<u8>> = Vec::new();
                // identity-ish mapping (or 0 fallback)
                mappings.push(
                    (0..channels)
                        .map(|i| {
                            let max = streams + coupled;
                            if max > 0 {
                                (i % max) as u8
                            } else {
                                0
                            }
                        })
                        .collect(),
                );
                // all-zero mapping
                mappings.push(vec![0u8; channels as usize]);
                // include 255 on last channel when possible
                let mut with_silence = vec![0u8; channels as usize];
                if channels > 0 {
                    with_silence[(channels - 1) as usize] = 255;
                }
                mappings.push(with_silence);
                // out-of-range mapping
                let mut out_of_range = vec![0u8; channels as usize];
                if channels > 0 {
                    out_of_range[0] = (streams + coupled + 1) as u8;
                }
                mappings.push(out_of_range);

                for mapping in mappings {
                    let rust = OpusMSDecoder::new(48000, channels, streams, coupled, &mapping);
                    let mut c_error = 0i32;
                    let c_ptr = unsafe {
                        opus_multistream_decoder_create(
                            48000,
                            channels,
                            streams,
                            coupled,
                            mapping.as_ptr(),
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
                        "sampled decoder parity mismatch (channels={channels}, streams={streams}, coupled={coupled}, mapping={mapping:?})"
                    );
                }
            }
        }
    }
}

#[test]
fn multistream_encoder_constructor_matrix_parity_sampled() {
    for channels in 1..=6 {
        for streams in 0..=channels + 1 {
            for coupled in 0..=streams + 1 {
                let mut mappings: Vec<Vec<u8>> = Vec::new();
                mappings.push(
                    (0..channels)
                        .map(|i| {
                            let max = streams + coupled;
                            if max > 0 {
                                (i % max) as u8
                            } else {
                                0
                            }
                        })
                        .collect(),
                );
                mappings.push(vec![0u8; channels as usize]);
                let mut with_silence = vec![0u8; channels as usize];
                if channels > 0 {
                    with_silence[(channels - 1) as usize] = 255;
                }
                mappings.push(with_silence);
                let mut out_of_range = vec![0u8; channels as usize];
                if channels > 0 {
                    out_of_range[0] = (streams + coupled + 1) as u8;
                }
                mappings.push(out_of_range);

                for mapping in mappings {
                    let rust = OpusMSEncoder::new(
                        48000,
                        channels,
                        streams,
                        coupled,
                        &mapping,
                        OPUS_APPLICATION_AUDIO,
                    );
                    let mut c_error = 0i32;
                    let c_ptr = unsafe {
                        opus_multistream_encoder_create(
                            48000,
                            channels,
                            streams,
                            coupled,
                            mapping.as_ptr(),
                            OPUS_APPLICATION_AUDIO,
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
                        "sampled encoder parity mismatch (channels={channels}, streams={streams}, coupled={coupled}, mapping={mapping:?})"
                    );
                }
            }
        }
    }
}

#[test]
fn multistream_wrapper_encode_rejects_frame_size_mismatch() {
    let mut enc =
        rust_opus_multistream_encoder_create(48000, 2, 2, 0, &[0, 1], OPUS_APPLICATION_AUDIO)
            .unwrap();
    let pcm = vec![0i16; 100];
    let mut packet = vec![0u8; 1000];
    let ret = rust_opus_multistream_encode(&mut enc, &pcm, 60, &mut packet);
    assert_eq!(ret, OPUS_BAD_ARG);
}

#[test]
fn multistream_wrapper_init_reinitializes_state() {
    let mut enc =
        rust_opus_multistream_encoder_create(48000, 2, 2, 0, &[0, 1], OPUS_APPLICATION_AUDIO)
            .unwrap();
    let mut dec = rust_opus_multistream_decoder_create(48000, 2, 2, 0, &[0, 1]).unwrap();

    let enc_ret = rust_opus_multistream_encoder_init(
        &mut enc,
        48000,
        3,
        2,
        1,
        &[0, 1, 2],
        OPUS_APPLICATION_AUDIO,
    );
    assert_eq!(enc_ret, 0);

    let dec_ret = rust_opus_multistream_decoder_init(&mut dec, 48000, 3, 2, 1, &[0, 1, 2]);
    assert_eq!(dec_ret, 0);

    // Re-init with invalid layout should fail with BAD_ARG.
    let bad = rust_opus_multistream_encoder_init(
        &mut enc,
        48000,
        2,
        1,
        1,
        &[0, 2],
        OPUS_APPLICATION_AUDIO,
    );
    assert_eq!(bad, OPUS_BAD_ARG);
}

#[test]
fn multistream_encoder_ctl_value_parity_with_c() {
    let mut rust =
        OpusMSEncoder::new(48000, 2, 2, 0, &[0, 1], OPUS_APPLICATION_AUDIO).expect("rust create");
    let mut c_error = 0i32;
    let c_ptr = unsafe {
        opus_multistream_encoder_create(
            48000,
            2,
            2,
            0,
            [0u8, 1u8].as_ptr(),
            OPUS_APPLICATION_AUDIO,
            &mut c_error,
        )
    };
    assert!(!c_ptr.is_null(), "c create failed: {c_error}");

    rust.set_complexity(6).unwrap();
    rust.set_inband_fec(1).unwrap();
    rust.set_packet_loss_perc(11).unwrap();
    rust.set_vbr(true);
    rust.set_vbr_constraint(true);

    unsafe {
        libopus_sys::opus_multistream_encoder_ctl(c_ptr, OPUS_SET_COMPLEXITY_REQUEST, 6i32);
        libopus_sys::opus_multistream_encoder_ctl(c_ptr, OPUS_SET_INBAND_FEC_REQUEST, 1i32);
        libopus_sys::opus_multistream_encoder_ctl(c_ptr, OPUS_SET_PACKET_LOSS_PERC_REQUEST, 11i32);
        libopus_sys::opus_multistream_encoder_ctl(c_ptr, OPUS_SET_VBR_REQUEST, 1i32);
        libopus_sys::opus_multistream_encoder_ctl(c_ptr, OPUS_SET_VBR_CONSTRAINT_REQUEST, 1i32);
    }

    let mut c_complexity = 0i32;
    let mut c_fec = 0i32;
    let mut c_loss = 0i32;
    let mut c_vbr = 0i32;
    let mut c_cvbr = 0i32;
    unsafe {
        libopus_sys::opus_multistream_encoder_ctl(
            c_ptr,
            OPUS_GET_COMPLEXITY_REQUEST,
            &mut c_complexity as *mut _,
        );
        libopus_sys::opus_multistream_encoder_ctl(
            c_ptr,
            OPUS_GET_INBAND_FEC_REQUEST,
            &mut c_fec as *mut _,
        );
        libopus_sys::opus_multistream_encoder_ctl(
            c_ptr,
            OPUS_GET_PACKET_LOSS_PERC_REQUEST,
            &mut c_loss as *mut _,
        );
        libopus_sys::opus_multistream_encoder_ctl(
            c_ptr,
            OPUS_GET_VBR_REQUEST,
            &mut c_vbr as *mut _,
        );
        libopus_sys::opus_multistream_encoder_ctl(
            c_ptr,
            OPUS_GET_VBR_CONSTRAINT_REQUEST,
            &mut c_cvbr as *mut _,
        );
    }

    assert_eq!(rust.complexity(), c_complexity);
    assert_eq!(rust.inband_fec(), c_fec);
    assert_eq!(rust.packet_loss_perc(), c_loss);
    assert_eq!(rust.vbr() as i32, c_vbr);
    assert_eq!(rust.vbr_constraint() as i32, c_cvbr);

    unsafe { opus_multistream_encoder_destroy(c_ptr) };
}

#[test]
fn multistream_decoder_ctl_value_parity_with_c() {
    let mut rust = OpusMSDecoder::new(48000, 2, 2, 0, &[0, 1]).expect("rust create");
    let mut c_error = 0i32;
    let c_ptr = unsafe {
        opus_multistream_decoder_create(48000, 2, 2, 0, [0u8, 1u8].as_ptr(), &mut c_error)
    };
    assert!(!c_ptr.is_null(), "c create failed: {c_error}");

    rust.set_complexity(5).unwrap();
    rust.set_gain(321).unwrap();

    unsafe {
        libopus_sys::opus_multistream_decoder_ctl(c_ptr, OPUS_SET_COMPLEXITY_REQUEST, 5i32);
        libopus_sys::opus_multistream_decoder_ctl(c_ptr, OPUS_SET_GAIN_REQUEST, 321i32);
    }

    let mut c_complexity = 0i32;
    let mut c_gain = 0i32;
    unsafe {
        libopus_sys::opus_multistream_decoder_ctl(
            c_ptr,
            OPUS_GET_COMPLEXITY_REQUEST,
            &mut c_complexity as *mut _,
        );
        libopus_sys::opus_multistream_decoder_ctl(
            c_ptr,
            OPUS_GET_GAIN_REQUEST,
            &mut c_gain as *mut _,
        );
    }

    assert_eq!(rust.complexity(), c_complexity);
    assert_eq!(rust.gain(), c_gain);

    unsafe { opus_multistream_decoder_destroy(c_ptr) };
}
