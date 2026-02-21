#![cfg(feature = "tools")]

use opurs::{
    opus_projection_ambisonics_encoder_create as rust_opus_projection_encoder_create,
    opus_projection_encode as rust_opus_projection_encode, OPUS_APPLICATION_AUDIO, OPUS_OK,
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
}

fn test_guard() -> MutexGuard<'static, ()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

#[test]
fn projection_encoder_foa_create_and_matrix_parity_with_c() {
    let _guard = test_guard();

    let mut rust_streams = -1i32;
    let mut rust_coupled = -1i32;
    let rust_enc = rust_opus_projection_encoder_create(
        48000,
        4,
        3,
        &mut rust_streams,
        &mut rust_coupled,
        OPUS_APPLICATION_AUDIO,
    )
    .expect("rust projection encoder create");

    let mut c_streams = -1i32;
    let mut c_coupled = -1i32;
    let mut c_error = 0i32;
    let c_enc = unsafe {
        opus_projection_ambisonics_encoder_create(
            48000,
            4,
            3,
            &mut c_streams,
            &mut c_coupled,
            OPUS_APPLICATION_AUDIO,
            &mut c_error,
        )
    };
    assert!(
        !c_enc.is_null(),
        "c projection encoder create failed: {c_error}"
    );

    assert_eq!(rust_streams, c_streams);
    assert_eq!(rust_coupled, c_coupled);

    let rust_matrix_size = rust_enc.demixing_matrix_size();
    let mut c_matrix_size = 0i32;
    let ret = unsafe {
        opus_projection_encoder_ctl(
            c_enc,
            OPUS_PROJECTION_GET_DEMIXING_MATRIX_SIZE_REQUEST,
            &mut c_matrix_size,
        )
    };
    assert_eq!(ret, OPUS_OK);
    assert_eq!(rust_matrix_size, c_matrix_size);

    let mut rust_matrix = vec![0u8; rust_matrix_size as usize];
    let mut c_matrix = vec![0u8; c_matrix_size as usize];
    rust_enc.copy_demixing_matrix(&mut rust_matrix).unwrap();
    let ret = unsafe {
        opus_projection_encoder_ctl(
            c_enc,
            OPUS_PROJECTION_GET_DEMIXING_MATRIX_REQUEST,
            c_matrix.as_mut_ptr(),
            c_matrix_size,
        )
    };
    assert_eq!(ret, OPUS_OK);
    assert_eq!(rust_matrix, c_matrix, "demixing matrix bytes mismatch");

    unsafe { opus_projection_encoder_destroy(c_enc) };
}

#[test]
fn projection_encoder_foa_encode_smoke_against_c() {
    let _guard = test_guard();

    let mut rust_streams = -1i32;
    let mut rust_coupled = -1i32;
    let mut rust_enc = rust_opus_projection_encoder_create(
        48000,
        4,
        3,
        &mut rust_streams,
        &mut rust_coupled,
        OPUS_APPLICATION_AUDIO,
    )
    .expect("rust projection encoder create");

    let mut c_streams = -1i32;
    let mut c_coupled = -1i32;
    let mut c_error = 0i32;
    let c_enc = unsafe {
        opus_projection_ambisonics_encoder_create(
            48000,
            4,
            3,
            &mut c_streams,
            &mut c_coupled,
            OPUS_APPLICATION_AUDIO,
            &mut c_error,
        )
    };
    assert!(
        !c_enc.is_null(),
        "c projection encoder create failed: {c_error}"
    );

    let frame_size = 960usize;
    let mut pcm = vec![0i16; frame_size * 4];
    for i in 0..frame_size {
        for ch in 0..4usize {
            pcm[i * 4 + ch] = (i as i16).wrapping_mul((ch as i16 + 2) * 11);
        }
    }

    let mut rust_packet = vec![0u8; 4000];
    let rust_len =
        rust_opus_projection_encode(&mut rust_enc, &pcm, frame_size as i32, &mut rust_packet);
    assert!(rust_len > 0, "rust projection encode failed: {rust_len}");

    let mut c_packet = vec![0u8; 4000];
    let c_len = unsafe {
        opus_projection_encode(
            c_enc,
            pcm.as_ptr(),
            frame_size as i32,
            c_packet.as_mut_ptr(),
            c_packet.len() as i32,
        )
    };
    assert!(c_len > 0, "c projection encode failed: {c_len}");

    unsafe { opus_projection_encoder_destroy(c_enc) };
}
