//! Opus Custom API parity checks for 24-bit paths.
//! Requires `--features tools` for upstream C differential checks.

#![cfg(feature = "tools")]

use libopus_sys::{
    opus_custom_decode24, opus_custom_decoder_create, opus_custom_decoder_ctl,
    opus_custom_decoder_destroy, opus_custom_encode24, opus_custom_encoder_create,
    opus_custom_encoder_ctl, opus_custom_encoder_destroy, opus_custom_mode_create,
    opus_custom_mode_destroy,
};
use opurs::arch::Arch;
use opurs::{OpusCustomDecoder, OpusCustomEncoder, OPUS_OK};
use std::sync::{Mutex, MutexGuard, OnceLock};

fn test_guard() -> MutexGuard<'static, ()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

fn deterministic_pcm24(frame_size: i32, channels: usize) -> Vec<i32> {
    let mut out = vec![0i32; frame_size as usize * channels];
    let mut x = 0x1234_5678u32;
    for sample in &mut out {
        x = x.wrapping_mul(1664525).wrapping_add(1013904223);
        let v = ((x >> 8) & 0x00FF_FFFF) as i32;
        *sample = v - 0x0080_0000;
    }
    out
}

const CELT_SET_SIGNALLING_REQUEST: i32 = 10016;

#[test]
fn custom_encode24_matches_upstream_c() {
    let _guard = test_guard();
    let frame_size = 960;
    let channels = 2;
    let pcm = deterministic_pcm24(frame_size, channels as usize);

    let mut rust_enc = OpusCustomEncoder::new(48000, channels, Arch::Scalar)
        .unwrap_or_else(|err| panic!("rust custom encoder init failed: {err}"));

    let mut err = OPUS_OK;
    let mode = unsafe { opus_custom_mode_create(48000, frame_size, &mut err) };
    assert!(!mode.is_null(), "C mode create failed");
    assert_eq!(err, OPUS_OK, "C mode create error");

    let c_enc = unsafe { opus_custom_encoder_create(mode, channels, &mut err) };
    assert!(!c_enc.is_null(), "C encoder create failed");
    assert_eq!(err, OPUS_OK, "C encoder create error");
    assert_eq!(
        unsafe { opus_custom_encoder_ctl(c_enc, CELT_SET_SIGNALLING_REQUEST, 0) },
        OPUS_OK,
        "C encoder signalling ctl failed"
    );

    let mut rust_packet = vec![0u8; 4000];
    let mut c_packet = vec![0u8; 4000];

    let rust_len = rust_enc.encode24(&pcm, &mut rust_packet);
    let c_len = unsafe {
        opus_custom_encode24(
            c_enc,
            pcm.as_ptr(),
            frame_size,
            c_packet.as_mut_ptr(),
            c_packet.len() as i32,
        )
    };

    assert_eq!(rust_len, c_len, "custom encode24 length mismatch");
    assert!(rust_len > 0, "custom encode24 failed: {rust_len}");
    assert_eq!(
        &rust_packet[..rust_len as usize],
        &c_packet[..c_len as usize],
        "custom encode24 payload mismatch"
    );

    unsafe {
        opus_custom_encoder_destroy(c_enc);
        opus_custom_mode_destroy(mode);
    }
}

#[test]
fn custom_decode24_matches_upstream_c() {
    let _guard = test_guard();
    let frame_size = 960;
    let channels = 2;
    let pcm = deterministic_pcm24(frame_size, channels as usize);

    let mut rust_dec = OpusCustomDecoder::new(48000, channels as usize, Arch::Scalar)
        .unwrap_or_else(|err| panic!("rust custom decoder init failed: {err}"));

    let mut err = OPUS_OK;
    let mode = unsafe { opus_custom_mode_create(48000, frame_size, &mut err) };
    assert!(!mode.is_null(), "C mode create failed");
    assert_eq!(err, OPUS_OK, "C mode create error");

    let c_enc = unsafe { opus_custom_encoder_create(mode, channels, &mut err) };
    assert!(!c_enc.is_null(), "C encoder create failed");
    assert_eq!(err, OPUS_OK, "C encoder create error");
    assert_eq!(
        unsafe { opus_custom_encoder_ctl(c_enc, CELT_SET_SIGNALLING_REQUEST, 0) },
        OPUS_OK,
        "C encoder signalling ctl failed"
    );

    let c_dec = unsafe { opus_custom_decoder_create(mode, channels, &mut err) };
    assert!(!c_dec.is_null(), "C decoder create failed");
    assert_eq!(err, OPUS_OK, "C decoder create error");
    assert_eq!(
        unsafe { opus_custom_decoder_ctl(c_dec, CELT_SET_SIGNALLING_REQUEST, 0) },
        OPUS_OK,
        "C decoder signalling ctl failed"
    );

    let mut packet = vec![0u8; 4000];
    let packet_len = unsafe {
        opus_custom_encode24(
            c_enc,
            pcm.as_ptr(),
            frame_size,
            packet.as_mut_ptr(),
            packet.len() as i32,
        )
    };
    assert!(packet_len > 0, "C custom encode24 failed: {packet_len}");

    let mut rust_out = vec![0i32; frame_size as usize * channels as usize];
    let mut c_out = vec![0i32; frame_size as usize * channels as usize];

    let rust_ret = rust_dec.decode24(&packet[..packet_len as usize], &mut rust_out, frame_size);
    let c_ret = unsafe {
        opus_custom_decode24(
            c_dec,
            packet.as_ptr(),
            packet_len,
            c_out.as_mut_ptr(),
            frame_size,
        )
    };

    assert_eq!(rust_ret, c_ret, "custom decode24 sample-count mismatch");
    assert!(rust_ret > 0, "custom decode24 failed: {rust_ret}");
    assert_eq!(
        &rust_out[..rust_ret as usize * channels as usize],
        &c_out[..c_ret as usize * channels as usize],
        "custom decode24 PCM mismatch"
    );

    unsafe {
        opus_custom_decoder_destroy(c_dec);
        opus_custom_encoder_destroy(c_enc);
        opus_custom_mode_destroy(mode);
    }
}
