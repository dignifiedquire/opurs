#![cfg(feature = "tools")]

use libopus_sys::{
    opus_multistream_encode, opus_multistream_encoder_create, opus_multistream_encoder_destroy,
    OPUS_APPLICATION_AUDIO,
};
use opurs::{
    opus_projection_decode as rust_opus_projection_decode,
    opus_projection_decoder_create as rust_opus_projection_decoder_create, OPUS_BAD_ARG,
    OPUS_PROJECTION_GET_DEMIXING_MATRIX_REQUEST, OPUS_PROJECTION_GET_DEMIXING_MATRIX_SIZE_REQUEST,
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
        assert!(
            (a as i32 - b as i32).abs() <= 1,
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
            assert!(
                (a as i32 - b as i32).abs() <= 1,
                "decode mismatch (channels={channels}) at index {idx}: rust={a}, c={b}"
            );
        }
    }
}
