#![cfg(feature = "tools")]

use libopus_sys::{
    opus_multistream_encode, opus_multistream_encoder_create, opus_multistream_encoder_destroy,
    OPUS_APPLICATION_AUDIO,
};
use opurs::{
    opus_projection_decode as rust_opus_projection_decode,
    opus_projection_decode24 as rust_opus_projection_decode24,
    opus_projection_decode_float as rust_opus_projection_decode_float,
    opus_projection_decoder_create as rust_opus_projection_decoder_create, OPUS_BAD_ARG,
    OPUS_GET_BANDWIDTH_REQUEST, OPUS_GET_COMPLEXITY_REQUEST, OPUS_GET_GAIN_REQUEST,
    OPUS_GET_LAST_PACKET_DURATION_REQUEST, OPUS_GET_PHASE_INVERSION_DISABLED_REQUEST,
    OPUS_GET_SAMPLE_RATE_REQUEST, OPUS_MULTISTREAM_GET_DECODER_STATE_REQUEST, OPUS_OK,
    OPUS_PROJECTION_GET_DEMIXING_MATRIX_REQUEST, OPUS_PROJECTION_GET_DEMIXING_MATRIX_SIZE_REQUEST,
    OPUS_SET_COMPLEXITY_REQUEST, OPUS_SET_GAIN_REQUEST, OPUS_SET_INBAND_FEC_REQUEST,
    OPUS_SET_PACKET_LOSS_PERC_REQUEST, OPUS_SET_PHASE_INVERSION_DISABLED_REQUEST,
};
use std::ffi::c_void;
use std::sync::{Mutex, MutexGuard, OnceLock};

unsafe extern "C" {
    fn opus_projection_ambisonics_encoder_create(
        Fs: i32,
        channels: i32,
        mapping_family: i32,
        streams: *mut i32,
        coupled_streams: *mut i32,
        application: i32,
        error: *mut i32,
    ) -> *mut c_void;
    fn opus_projection_encoder_destroy(st: *mut c_void);
    fn opus_projection_encoder_ctl(st: *mut c_void, request: i32, ...) -> i32;
    fn opus_projection_encode(
        st: *mut c_void,
        pcm: *const i16,
        frame_size: i32,
        data: *mut u8,
        max_data_bytes: i32,
    ) -> i32;
    fn opus_projection_decoder_get_size(channels: i32, streams: i32, coupled_streams: i32) -> i32;
    fn opus_projection_decoder_create(
        Fs: i32,
        channels: i32,
        streams: i32,
        coupled_streams: i32,
        demixing_matrix: *mut u8,
        demixing_matrix_size: i32,
        error: *mut i32,
    ) -> *mut c_void;
    fn opus_projection_decoder_destroy(st: *mut c_void);
    fn opus_projection_decode(
        st: *mut c_void,
        data: *const u8,
        len: i32,
        pcm: *mut i16,
        frame_size: i32,
        decode_fec: i32,
    ) -> i32;
    fn opus_projection_decode_float(
        st: *mut c_void,
        data: *const u8,
        len: i32,
        pcm: *mut f32,
        frame_size: i32,
        decode_fec: i32,
    ) -> i32;
    fn opus_projection_decode24(
        st: *mut c_void,
        data: *const u8,
        len: i32,
        pcm: *mut i32,
        frame_size: i32,
        decode_fec: i32,
    ) -> i32;
    fn opus_projection_decoder_ctl(st: *mut c_void, request: i32, ...) -> i32;
}

fn test_guard() -> MutexGuard<'static, ()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

fn identity_demixing_matrix_le(channels: i32) -> Vec<u8> {
    let rows = channels as usize;
    let cols = channels as usize;
    let mut data = vec![0i16; rows * cols];
    for ch in 0..channels as usize {
        data[rows * ch + ch] = 32767;
    }
    let mut bytes = Vec::with_capacity(data.len() * 2);
    for value in data {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes
}

#[test]
fn projection_decoder_get_size_zero_nonzero_parity() {
    let _guard = test_guard();
    let cases = [(2, 1, 1), (2, 1, 0), (0, 1, 1), (2, 0, 0), (256, 1, 1)];
    for (channels, streams, coupled) in cases {
        let rust = opurs::opus_projection_decoder_get_size(channels, streams, coupled);
        let c = unsafe { opus_projection_decoder_get_size(channels, streams, coupled) };
        assert_eq!(
            rust == 0,
            c == 0,
            "projection get_size validity mismatch (channels={channels}, streams={streams}, coupled={coupled})"
        );
    }
}

#[test]
fn projection_decoder_create_parity_with_c() {
    let _guard = test_guard();

    let good_matrix = identity_demixing_matrix_le(2);
    let bad_matrix = vec![0u8; 2];

    for matrix in [&good_matrix, &bad_matrix] {
        let rust = rust_opus_projection_decoder_create(48000, 2, 1, 1, matrix);

        let mut c_error = 0i32;
        let c_ptr = unsafe {
            opus_projection_decoder_create(
                48000,
                2,
                1,
                1,
                matrix.as_ptr() as *mut u8,
                matrix.len() as i32,
                &mut c_error,
            )
        };
        let c_ok = !c_ptr.is_null();
        if !c_ptr.is_null() {
            unsafe { opus_projection_decoder_destroy(c_ptr) };
        }

        assert_eq!(
            rust.is_ok(),
            c_ok,
            "projection decoder create parity mismatch"
        );
        if let Err(err) = rust {
            assert_eq!(err, OPUS_BAD_ARG);
            assert_eq!(c_error, OPUS_BAD_ARG);
        }
    }
}

#[test]
fn projection_decoder_decode_parity_with_c() {
    let _guard = test_guard();

    // Build a packet with C multistream encoder.
    let mut c_error = 0i32;
    let c_enc = unsafe {
        opus_multistream_encoder_create(
            48000,
            2,
            1,
            1,
            [0u8, 1u8].as_ptr(),
            OPUS_APPLICATION_AUDIO as i32,
            &mut c_error,
        )
    };
    assert!(!c_enc.is_null(), "c encoder create failed: {c_error}");

    let frame_size = 960usize;
    let mut pcm = vec![0i16; frame_size * 2];
    for i in 0..frame_size {
        pcm[i * 2] = (i as i16).wrapping_mul(23);
        pcm[i * 2 + 1] = (i as i16).wrapping_mul(-19);
    }
    let mut packet = vec![0u8; 4000];
    let packet_len = unsafe {
        opus_multistream_encode(
            c_enc,
            pcm.as_ptr(),
            frame_size as i32,
            packet.as_mut_ptr(),
            packet.len() as i32,
        )
    };
    unsafe { opus_multistream_encoder_destroy(c_enc) };
    assert!(packet_len > 0, "c encode failed: {packet_len}");

    let matrix = identity_demixing_matrix_le(2);
    let mut rust_dec = rust_opus_projection_decoder_create(48000, 2, 1, 1, &matrix).unwrap();
    let mut c_error = 0i32;
    let c_dec = unsafe {
        opus_projection_decoder_create(
            48000,
            2,
            1,
            1,
            matrix.as_ptr() as *mut u8,
            matrix.len() as i32,
            &mut c_error,
        )
    };
    assert!(
        !c_dec.is_null(),
        "c projection decoder create failed: {c_error}"
    );

    let mut rust_out = vec![0i16; frame_size * 2];
    let rust_ret = rust_opus_projection_decode(
        &mut rust_dec,
        &packet[..packet_len as usize],
        &mut rust_out,
        frame_size as i32,
        false,
    );

    let mut c_out = vec![0i16; frame_size * 2];
    let c_ret = unsafe {
        opus_projection_decode(
            c_dec,
            packet.as_ptr(),
            packet_len,
            c_out.as_mut_ptr(),
            frame_size as i32,
            0,
        )
    };
    unsafe { opus_projection_decoder_destroy(c_dec) };

    assert_eq!(rust_ret, c_ret, "projection decode return mismatch");
    for (idx, (&a, &b)) in rust_out.iter().zip(c_out.iter()).enumerate() {
        assert_eq!(
            a, b,
            "projection decode mismatch at index {idx}: rust={a}, c={b}"
        );
    }
}

#[test]
fn projection_decoder_higher_order_parity_with_c() {
    let _guard = test_guard();

    for channels in [4, 9, 16, 25, 36] {
        let mut streams = -1i32;
        let mut coupled = -1i32;
        let mut c_error = 0i32;
        let c_enc = unsafe {
            opus_projection_ambisonics_encoder_create(
                48000,
                channels,
                3,
                &mut streams,
                &mut coupled,
                OPUS_APPLICATION_AUDIO as i32,
                &mut c_error,
            )
        };
        assert!(
            !c_enc.is_null(),
            "c projection encoder create failed (channels={channels}): {c_error}"
        );

        let mut matrix_size = 0i32;
        let ret = unsafe {
            opus_projection_encoder_ctl(
                c_enc,
                OPUS_PROJECTION_GET_DEMIXING_MATRIX_SIZE_REQUEST,
                &mut matrix_size,
            )
        };
        assert_eq!(ret, 0, "matrix-size ctl failed (channels={channels})");
        let mut matrix = vec![0u8; matrix_size as usize];
        let ret = unsafe {
            opus_projection_encoder_ctl(
                c_enc,
                OPUS_PROJECTION_GET_DEMIXING_MATRIX_REQUEST,
                matrix.as_mut_ptr(),
                matrix_size,
            )
        };
        assert_eq!(ret, 0, "matrix ctl failed (channels={channels})");

        let frame_size = 960usize;
        let mut pcm = vec![0i16; frame_size * channels as usize];
        for i in 0..frame_size {
            for ch in 0..channels as usize {
                let base = (i as i16).wrapping_mul(((ch as i16 % 7) + 2) * 13);
                pcm[i * channels as usize + ch] = base;
            }
        }

        let mut packet = vec![0u8; 4000];
        let packet_len = unsafe {
            opus_projection_encode(
                c_enc,
                pcm.as_ptr(),
                frame_size as i32,
                packet.as_mut_ptr(),
                packet.len() as i32,
            )
        };
        assert!(
            packet_len > 0,
            "c projection encode failed (channels={channels}): {packet_len}"
        );
        unsafe { opus_projection_encoder_destroy(c_enc) };

        let mut rust_dec =
            rust_opus_projection_decoder_create(48000, channels, streams, coupled, &matrix)
                .unwrap();

        let mut c_error = 0i32;
        let c_dec = unsafe {
            opus_projection_decoder_create(
                48000,
                channels,
                streams,
                coupled,
                matrix.as_ptr() as *mut u8,
                matrix.len() as i32,
                &mut c_error,
            )
        };
        assert!(
            !c_dec.is_null(),
            "c projection decoder create failed (channels={channels}): {c_error}"
        );

        let mut rust_out = vec![0i16; frame_size * channels as usize];
        let rust_ret = rust_opus_projection_decode(
            &mut rust_dec,
            &packet[..packet_len as usize],
            &mut rust_out,
            frame_size as i32,
            false,
        );

        let mut c_out = vec![0i16; frame_size * channels as usize];
        let c_ret = unsafe {
            opus_projection_decode(
                c_dec,
                packet.as_ptr(),
                packet_len,
                c_out.as_mut_ptr(),
                frame_size as i32,
                0,
            )
        };
        unsafe { opus_projection_decoder_destroy(c_dec) };

        assert_eq!(
            rust_ret, c_ret,
            "decode return mismatch (channels={channels})"
        );
        for (idx, (&a, &b)) in rust_out.iter().zip(c_out.iter()).enumerate() {
            assert_eq!(
                a, b,
                "decode mismatch (channels={channels}) at index {idx}: rust={a}, c={b}"
            );
        }
    }
}

#[test]
fn projection_decoder_ctl_value_parity_with_c() {
    let _guard = test_guard();

    let channels = 4i32;
    let mut streams = -1i32;
    let mut coupled = -1i32;
    let mut c_error = 0i32;
    let c_enc = unsafe {
        opus_projection_ambisonics_encoder_create(
            48000,
            channels,
            3,
            &mut streams,
            &mut coupled,
            OPUS_APPLICATION_AUDIO as i32,
            &mut c_error,
        )
    };
    assert!(
        !c_enc.is_null(),
        "c projection encoder create failed: {c_error}"
    );

    let mut matrix_size = 0i32;
    let ret = unsafe {
        opus_projection_encoder_ctl(
            c_enc,
            OPUS_PROJECTION_GET_DEMIXING_MATRIX_SIZE_REQUEST,
            &mut matrix_size,
        )
    };
    assert_eq!(ret, 0);
    let mut matrix = vec![0u8; matrix_size as usize];
    let ret = unsafe {
        opus_projection_encoder_ctl(
            c_enc,
            OPUS_PROJECTION_GET_DEMIXING_MATRIX_REQUEST,
            matrix.as_mut_ptr(),
            matrix_size,
        )
    };
    assert_eq!(ret, 0);

    let frame_size = 960usize;
    let mut pcm = vec![0i16; frame_size * channels as usize];
    for i in 0..frame_size {
        for ch in 0..channels as usize {
            pcm[i * channels as usize + ch] = (i as i16).wrapping_mul((ch as i16 + 3) * 17);
        }
    }
    let mut packet = vec![0u8; 4000];
    let packet_len = unsafe {
        opus_projection_encode(
            c_enc,
            pcm.as_ptr(),
            frame_size as i32,
            packet.as_mut_ptr(),
            packet.len() as i32,
        )
    };
    unsafe { opus_projection_encoder_destroy(c_enc) };
    assert!(packet_len > 0, "c projection encode failed: {packet_len}");

    let mut rust = rust_opus_projection_decoder_create(48000, channels, streams, coupled, &matrix)
        .expect("rust projection decoder create");
    let mut c_error = 0i32;
    let c_ptr = unsafe {
        opus_projection_decoder_create(
            48000,
            channels,
            streams,
            coupled,
            matrix.as_ptr() as *mut u8,
            matrix.len() as i32,
            &mut c_error,
        )
    };
    assert!(
        !c_ptr.is_null(),
        "c projection decoder create failed: {c_error}"
    );

    rust.set_complexity(5).unwrap();
    rust.set_gain(321).unwrap();
    rust.set_phase_inversion_disabled(true);

    unsafe {
        opus_projection_decoder_ctl(c_ptr, OPUS_SET_COMPLEXITY_REQUEST, 5i32);
        opus_projection_decoder_ctl(c_ptr, OPUS_SET_GAIN_REQUEST, 321i32);
        opus_projection_decoder_ctl(c_ptr, OPUS_SET_PHASE_INVERSION_DISABLED_REQUEST, 1i32);
    }

    let mut rust_out = vec![0i16; frame_size * channels as usize];
    let rust_decoded = rust_opus_projection_decode(
        &mut rust,
        &packet[..packet_len as usize],
        &mut rust_out,
        frame_size as i32,
        false,
    );
    assert!(rust_decoded > 0, "rust decode failed: {rust_decoded}");

    let mut c_out = vec![0i16; frame_size * channels as usize];
    let c_decoded = unsafe {
        opus_projection_decode(
            c_ptr,
            packet.as_ptr(),
            packet_len,
            c_out.as_mut_ptr(),
            frame_size as i32,
            0,
        )
    };
    assert_eq!(rust_decoded, c_decoded);

    let mut c_complexity = 0i32;
    let mut c_gain = 0i32;
    let mut c_phase_inv_disabled = 0i32;
    let mut c_bandwidth = 0i32;
    let mut c_sample_rate = 0i32;
    let mut c_last_packet_duration = 0i32;
    unsafe {
        opus_projection_decoder_ctl(
            c_ptr,
            OPUS_GET_COMPLEXITY_REQUEST,
            &mut c_complexity as *mut _,
        );
        opus_projection_decoder_ctl(c_ptr, OPUS_GET_GAIN_REQUEST, &mut c_gain as *mut _);
        opus_projection_decoder_ctl(
            c_ptr,
            OPUS_GET_PHASE_INVERSION_DISABLED_REQUEST,
            &mut c_phase_inv_disabled as *mut _,
        );
        opus_projection_decoder_ctl(
            c_ptr,
            OPUS_GET_BANDWIDTH_REQUEST,
            &mut c_bandwidth as *mut _,
        );
        opus_projection_decoder_ctl(
            c_ptr,
            OPUS_GET_SAMPLE_RATE_REQUEST,
            &mut c_sample_rate as *mut _,
        );
        opus_projection_decoder_ctl(
            c_ptr,
            OPUS_GET_LAST_PACKET_DURATION_REQUEST,
            &mut c_last_packet_duration as *mut _,
        );
    }

    assert_eq!(rust.complexity(), c_complexity);
    assert_eq!(rust.gain(), c_gain);
    assert_eq!(rust.phase_inversion_disabled() as i32, c_phase_inv_disabled);
    assert_eq!(rust.bandwidth(), c_bandwidth);
    assert_eq!(rust.sample_rate(), c_sample_rate);
    assert_eq!(rust.last_packet_duration(), c_last_packet_duration);

    unsafe { opus_projection_decoder_destroy(c_ptr) };
}

#[test]
fn projection_decoder_state_access_parity_with_c() {
    let _guard = test_guard();

    let channels = 4i32;
    let mut streams = -1i32;
    let mut coupled = -1i32;
    let mut c_error = 0i32;
    let c_enc = unsafe {
        opus_projection_ambisonics_encoder_create(
            48000,
            channels,
            3,
            &mut streams,
            &mut coupled,
            OPUS_APPLICATION_AUDIO as i32,
            &mut c_error,
        )
    };
    assert!(
        !c_enc.is_null(),
        "c projection encoder create failed: {c_error}"
    );

    let matrix = projection_demixing_matrix_from_c_encoder(c_enc, channels);
    unsafe { opus_projection_encoder_destroy(c_enc) };

    let mut rust =
        rust_opus_projection_decoder_create(48000, channels, streams, coupled, &matrix).unwrap();
    let mut c_error = 0i32;
    let c_ptr = unsafe {
        opus_projection_decoder_create(
            48000,
            channels,
            streams,
            coupled,
            matrix.as_ptr() as *mut u8,
            matrix.len() as i32,
            &mut c_error,
        )
    };
    assert!(
        !c_ptr.is_null(),
        "c projection decoder create failed: {c_error}"
    );

    assert_eq!(rust.decoder_state_mut(-1).err(), Some(OPUS_BAD_ARG));
    assert_eq!(rust.decoder_state_mut(streams).err(), Some(OPUS_BAD_ARG));

    let mut c_state: *mut libopus_sys::OpusDecoder = core::ptr::null_mut();
    let c_bad_neg = unsafe {
        opus_projection_decoder_ctl(
            c_ptr,
            OPUS_MULTISTREAM_GET_DECODER_STATE_REQUEST,
            -1i32,
            &mut c_state as *mut _,
        )
    };
    let c_bad_high = unsafe {
        opus_projection_decoder_ctl(
            c_ptr,
            OPUS_MULTISTREAM_GET_DECODER_STATE_REQUEST,
            streams,
            &mut c_state as *mut _,
        )
    };
    assert_eq!(c_bad_neg, OPUS_BAD_ARG);
    assert_eq!(c_bad_high, OPUS_BAD_ARG);

    let mut c_state0: *mut libopus_sys::OpusDecoder = core::ptr::null_mut();
    let mut c_state1: *mut libopus_sys::OpusDecoder = core::ptr::null_mut();
    let c_ret0 = unsafe {
        opus_projection_decoder_ctl(
            c_ptr,
            OPUS_MULTISTREAM_GET_DECODER_STATE_REQUEST,
            0i32,
            &mut c_state0 as *mut _,
        )
    };
    let c_ret1 = unsafe {
        opus_projection_decoder_ctl(
            c_ptr,
            OPUS_MULTISTREAM_GET_DECODER_STATE_REQUEST,
            1i32,
            &mut c_state1 as *mut _,
        )
    };
    assert_eq!(c_ret0, OPUS_OK);
    assert_eq!(c_ret1, OPUS_OK);
    assert!(!c_state0.is_null());
    assert!(!c_state1.is_null());

    rust.decoder_state_mut(1).unwrap().set_gain(123).unwrap();
    rust.decoder_state_mut(1)
        .unwrap()
        .set_phase_inversion_disabled(true);
    unsafe {
        libopus_sys::opus_decoder_ctl(c_state1, OPUS_SET_GAIN_REQUEST, 123i32);
        libopus_sys::opus_decoder_ctl(c_state1, OPUS_SET_PHASE_INVERSION_DISABLED_REQUEST, 1i32);
    }

    let rust_gain0 = rust.decoder_state_mut(0).unwrap().gain();
    let rust_gain1 = rust.decoder_state_mut(1).unwrap().gain();
    let rust_phase0 = rust
        .decoder_state_mut(0)
        .unwrap()
        .phase_inversion_disabled();
    let rust_phase1 = rust
        .decoder_state_mut(1)
        .unwrap()
        .phase_inversion_disabled();

    let mut c_gain0 = 0i32;
    let mut c_gain1 = 0i32;
    let mut c_phase0 = 0i32;
    let mut c_phase1 = 0i32;
    unsafe {
        libopus_sys::opus_decoder_ctl(c_state0, OPUS_GET_GAIN_REQUEST, &mut c_gain0 as *mut _);
        libopus_sys::opus_decoder_ctl(c_state1, OPUS_GET_GAIN_REQUEST, &mut c_gain1 as *mut _);
        libopus_sys::opus_decoder_ctl(
            c_state0,
            OPUS_GET_PHASE_INVERSION_DISABLED_REQUEST,
            &mut c_phase0 as *mut _,
        );
        libopus_sys::opus_decoder_ctl(
            c_state1,
            OPUS_GET_PHASE_INVERSION_DISABLED_REQUEST,
            &mut c_phase1 as *mut _,
        );
    }

    assert_eq!(rust_gain0, c_gain0);
    assert_eq!(rust_gain1, c_gain1);
    assert_eq!(rust_phase0 as i32, c_phase0);
    assert_eq!(rust_phase1 as i32, c_phase1);

    unsafe { opus_projection_decoder_destroy(c_ptr) };
}

#[test]
fn projection_creation_arguments_matrix_parity_with_c() {
    let _guard = test_guard();

    for channels in 0..=254i32 {
        let mut rust_streams = -1i32;
        let mut rust_coupled = -1i32;
        let rust_enc = opurs::opus_projection_ambisonics_encoder_create(
            48000,
            channels,
            3,
            &mut rust_streams,
            &mut rust_coupled,
            OPUS_APPLICATION_AUDIO as i32,
        );

        let mut c_streams = -1i32;
        let mut c_coupled = -1i32;
        let mut c_error = 0i32;
        let c_enc = unsafe {
            opus_projection_ambisonics_encoder_create(
                48000,
                channels,
                3,
                &mut c_streams,
                &mut c_coupled,
                OPUS_APPLICATION_AUDIO as i32,
                &mut c_error,
            )
        };
        let c_enc_ok = !c_enc.is_null();
        assert_eq!(
            rust_enc.is_ok(),
            c_enc_ok,
            "projection encoder create parity mismatch (channels={channels})"
        );

        if let (Ok(_), true) = (rust_enc, c_enc_ok) {
            let mut matrix_size = 0i32;
            let ret = unsafe {
                opus_projection_encoder_ctl(
                    c_enc,
                    OPUS_PROJECTION_GET_DEMIXING_MATRIX_SIZE_REQUEST,
                    &mut matrix_size,
                )
            };
            assert_eq!(ret, 0);
            let mut matrix = vec![0u8; matrix_size as usize];
            let ret = unsafe {
                opus_projection_encoder_ctl(
                    c_enc,
                    OPUS_PROJECTION_GET_DEMIXING_MATRIX_REQUEST,
                    matrix.as_mut_ptr(),
                    matrix_size,
                )
            };
            assert_eq!(ret, 0);

            let rust_dec =
                rust_opus_projection_decoder_create(48000, channels, c_streams, c_coupled, &matrix);
            let mut c_dec_error = 0i32;
            let c_dec = unsafe {
                opus_projection_decoder_create(
                    48000,
                    channels,
                    c_streams,
                    c_coupled,
                    matrix.as_ptr() as *mut u8,
                    matrix.len() as i32,
                    &mut c_dec_error,
                )
            };
            let c_dec_ok = !c_dec.is_null();
            assert_eq!(
                rust_dec.is_ok(),
                c_dec_ok,
                "projection decoder create parity mismatch (channels={channels}, streams={c_streams}, coupled={c_coupled})"
            );
            if !c_dec.is_null() {
                unsafe { opus_projection_decoder_destroy(c_dec) };
            }
        }

        if !c_enc.is_null() {
            unsafe { opus_projection_encoder_destroy(c_enc) };
        }
    }
}

fn projection_demixing_matrix_from_c_encoder(c_enc: *mut c_void, channels: i32) -> Vec<u8> {
    assert!(
        !c_enc.is_null(),
        "c projection encoder must be initialized (channels={channels})"
    );

    let mut matrix_size = 0i32;
    let ret = unsafe {
        opus_projection_encoder_ctl(
            c_enc,
            OPUS_PROJECTION_GET_DEMIXING_MATRIX_SIZE_REQUEST,
            &mut matrix_size,
        )
    };
    assert_eq!(ret, 0);
    let mut matrix = vec![0u8; matrix_size as usize];
    let ret = unsafe {
        opus_projection_encoder_ctl(
            c_enc,
            OPUS_PROJECTION_GET_DEMIXING_MATRIX_REQUEST,
            matrix.as_mut_ptr(),
            matrix_size,
        )
    };
    assert_eq!(ret, 0);
    matrix
}

#[test]
fn projection_decode_format_and_frame_matrix_parity_with_c() {
    let _guard = test_guard();

    for channels in [4, 9] {
        let mut streams = -1i32;
        let mut coupled = -1i32;
        let mut c_error = 0i32;
        let c_enc = unsafe {
            opus_projection_ambisonics_encoder_create(
                48000,
                channels,
                3,
                &mut streams,
                &mut coupled,
                OPUS_APPLICATION_AUDIO as i32,
                &mut c_error,
            )
        };
        assert!(
            !c_enc.is_null(),
            "c projection encoder create failed (channels={channels}): {c_error}"
        );

        let matrix = projection_demixing_matrix_from_c_encoder(c_enc, channels);

        for frame_size in [120usize, 240, 480, 960, 1920] {
            let mut pcm = vec![0i16; frame_size * channels as usize];
            for i in 0..frame_size {
                for ch in 0..channels as usize {
                    let base = (i as i16).wrapping_mul(((ch as i16 % 9) + 2) * 11);
                    pcm[i * channels as usize + ch] = base;
                }
            }

            let mut packet = vec![0u8; 4000];
            let packet_len = unsafe {
                opus_projection_encode(
                    c_enc,
                    pcm.as_ptr(),
                    frame_size as i32,
                    packet.as_mut_ptr(),
                    packet.len() as i32,
                )
            };
            assert!(
                packet_len > 0,
                "c projection encode failed (channels={channels}, frame_size={frame_size}): {packet_len}"
            );

            let mut rust_dec =
                rust_opus_projection_decoder_create(48000, channels, streams, coupled, &matrix)
                    .unwrap();
            let mut c_dec_error = 0i32;
            let c_dec = unsafe {
                opus_projection_decoder_create(
                    48000,
                    channels,
                    streams,
                    coupled,
                    matrix.as_ptr() as *mut u8,
                    matrix.len() as i32,
                    &mut c_dec_error,
                )
            };
            assert!(
                !c_dec.is_null(),
                "c projection decoder create failed (channels={channels}): {c_dec_error}"
            );

            let mut rust_i16 = vec![0i16; frame_size * channels as usize];
            let mut c_i16 = vec![0i16; frame_size * channels as usize];
            let rust_ret = rust_opus_projection_decode(
                &mut rust_dec,
                &packet[..packet_len as usize],
                &mut rust_i16,
                frame_size as i32,
                false,
            );
            let c_ret = unsafe {
                opus_projection_decode(
                    c_dec,
                    packet.as_ptr(),
                    packet_len,
                    c_i16.as_mut_ptr(),
                    frame_size as i32,
                    0,
                )
            };
            assert_eq!(rust_ret, c_ret);
            for (idx, (&a, &b)) in rust_i16.iter().zip(c_i16.iter()).enumerate() {
                assert_eq!(a, b, "decode i16 mismatch at index {idx}: rust={a} c={b}");
            }

            let mut rust_f32 = vec![0f32; frame_size * channels as usize];
            let mut c_f32 = vec![0f32; frame_size * channels as usize];
            let rust_ret = rust_opus_projection_decode_float(
                &mut rust_dec,
                &packet[..packet_len as usize],
                &mut rust_f32,
                frame_size as i32,
                false,
            );
            let c_ret = unsafe {
                opus_projection_decode_float(
                    c_dec,
                    packet.as_ptr(),
                    packet_len,
                    c_f32.as_mut_ptr(),
                    frame_size as i32,
                    0,
                )
            };
            assert_eq!(rust_ret, c_ret);
            for (idx, (&a, &b)) in rust_f32.iter().zip(c_f32.iter()).enumerate() {
                assert_eq!(
                    a.to_bits(),
                    b.to_bits(),
                    "decode float mismatch at index {idx}: rust={a:e} (0x{:08x}) c={b:e} (0x{:08x})",
                    a.to_bits(),
                    b.to_bits()
                );
            }

            let mut rust_i24 = vec![0i32; frame_size * channels as usize];
            let mut c_i24 = vec![0i32; frame_size * channels as usize];
            let rust_ret = rust_opus_projection_decode24(
                &mut rust_dec,
                &packet[..packet_len as usize],
                &mut rust_i24,
                frame_size as i32,
                false,
            );
            let c_ret = unsafe {
                opus_projection_decode24(
                    c_dec,
                    packet.as_ptr(),
                    packet_len,
                    c_i24.as_mut_ptr(),
                    frame_size as i32,
                    0,
                )
            };
            assert_eq!(rust_ret, c_ret);
            for (idx, (&a, &b)) in rust_i24.iter().zip(c_i24.iter()).enumerate() {
                assert_eq!(a, b, "decode i24 mismatch at index {idx}: rust={a} c={b}");
            }

            unsafe { opus_projection_decoder_destroy(c_dec) };
        }

        unsafe { opus_projection_encoder_destroy(c_enc) };
    }
}

#[test]
fn projection_plc_and_fec_parity_with_c() {
    let _guard = test_guard();

    let channels = 4i32;
    let mut streams = -1i32;
    let mut coupled = -1i32;
    let mut c_error = 0i32;
    let c_enc = unsafe {
        opus_projection_ambisonics_encoder_create(
            48000,
            channels,
            3,
            &mut streams,
            &mut coupled,
            OPUS_APPLICATION_AUDIO as i32,
            &mut c_error,
        )
    };
    assert!(
        !c_enc.is_null(),
        "c projection encoder create failed: {c_error}"
    );

    unsafe {
        opus_projection_encoder_ctl(c_enc, OPUS_SET_INBAND_FEC_REQUEST, 1i32);
        opus_projection_encoder_ctl(c_enc, OPUS_SET_PACKET_LOSS_PERC_REQUEST, 25i32);
    }

    let matrix = projection_demixing_matrix_from_c_encoder(c_enc, channels);

    let frame_size = 960usize;
    let mut pcm0 = vec![0i16; frame_size * channels as usize];
    let mut pcm1 = vec![0i16; frame_size * channels as usize];
    for i in 0..frame_size {
        for ch in 0..channels as usize {
            pcm0[i * channels as usize + ch] = (i as i16).wrapping_mul((ch as i16 + 2) * 7);
            pcm1[i * channels as usize + ch] = (i as i16).wrapping_mul((ch as i16 + 3) * 9);
        }
    }

    let mut packet0 = vec![0u8; 4000];
    let mut packet1 = vec![0u8; 4000];
    let len0 = unsafe {
        opus_projection_encode(
            c_enc,
            pcm0.as_ptr(),
            frame_size as i32,
            packet0.as_mut_ptr(),
            packet0.len() as i32,
        )
    };
    let len1 = unsafe {
        opus_projection_encode(
            c_enc,
            pcm1.as_ptr(),
            frame_size as i32,
            packet1.as_mut_ptr(),
            packet1.len() as i32,
        )
    };
    assert!(len0 > 0 && len1 > 0);
    unsafe { opus_projection_encoder_destroy(c_enc) };

    let mut rust_dec =
        rust_opus_projection_decoder_create(48000, channels, streams, coupled, &matrix).unwrap();
    let mut c_dec_error = 0i32;
    let c_dec = unsafe {
        opus_projection_decoder_create(
            48000,
            channels,
            streams,
            coupled,
            matrix.as_ptr() as *mut u8,
            matrix.len() as i32,
            &mut c_dec_error,
        )
    };
    assert!(
        !c_dec.is_null(),
        "c projection decoder create failed: {c_dec_error}"
    );

    let mut rust_tmp = vec![0i16; frame_size * channels as usize];
    let mut c_tmp = vec![0i16; frame_size * channels as usize];
    let rust_ret = rust_opus_projection_decode(
        &mut rust_dec,
        &packet0[..len0 as usize],
        &mut rust_tmp,
        frame_size as i32,
        false,
    );
    let c_ret = unsafe {
        opus_projection_decode(
            c_dec,
            packet0.as_ptr(),
            len0,
            c_tmp.as_mut_ptr(),
            frame_size as i32,
            0,
        )
    };
    assert_eq!(rust_ret, c_ret);

    let mut rust_plc = vec![0i16; frame_size * channels as usize];
    let mut c_plc = vec![0i16; frame_size * channels as usize];
    let rust_plc_ret = rust_opus_projection_decode(&mut rust_dec, &[], &mut rust_plc, 960, false);
    let c_plc_ret =
        unsafe { opus_projection_decode(c_dec, core::ptr::null(), 0, c_plc.as_mut_ptr(), 960, 0) };
    assert_eq!(rust_plc_ret, c_plc_ret);
    for (idx, (&a, &b)) in rust_plc.iter().zip(c_plc.iter()).enumerate() {
        assert_eq!(a, b, "PLC decode mismatch at index {idx}: rust={a} c={b}");
    }

    unsafe { opus_projection_decoder_destroy(c_dec) };

    let mut rust_fec_dec =
        rust_opus_projection_decoder_create(48000, channels, streams, coupled, &matrix).unwrap();
    let mut c_dec_error = 0i32;
    let c_fec_dec = unsafe {
        opus_projection_decoder_create(
            48000,
            channels,
            streams,
            coupled,
            matrix.as_ptr() as *mut u8,
            matrix.len() as i32,
            &mut c_dec_error,
        )
    };
    assert!(
        !c_fec_dec.is_null(),
        "c projection decoder create failed: {c_dec_error}"
    );

    let mut rust_fec = vec![0i16; frame_size * channels as usize];
    let mut c_fec = vec![0i16; frame_size * channels as usize];
    let rust_fec_ret = rust_opus_projection_decode(
        &mut rust_fec_dec,
        &packet1[..len1 as usize],
        &mut rust_fec,
        frame_size as i32,
        true,
    );
    let c_fec_ret = unsafe {
        opus_projection_decode(
            c_fec_dec,
            packet1.as_ptr(),
            len1,
            c_fec.as_mut_ptr(),
            frame_size as i32,
            1,
        )
    };
    assert_eq!(rust_fec_ret, c_fec_ret);
    for (idx, (&a, &b)) in rust_fec.iter().zip(c_fec.iter()).enumerate() {
        assert_eq!(a, b, "FEC decode mismatch at index {idx}: rust={a} c={b}");
    }

    unsafe { opus_projection_decoder_destroy(c_fec_dec) };
}
