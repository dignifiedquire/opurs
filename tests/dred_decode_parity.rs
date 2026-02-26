//! DRED decode parity checks against upstream C.

#![cfg(feature = "tools-dnn")]

use std::ptr;

use opurs::{
    opus_decode_float, opus_decoder_dred_decode_float, OpusDecoder, OpusEncoder,
    OPUS_APPLICATION_AUDIO,
};

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

#[test]
fn dred_decode_float_stage0_matches_c_null_dred_path() {
    let mut enc = OpusEncoder::new(48_000, 1, OPUS_APPLICATION_AUDIO).expect("rust encoder");
    let input: Vec<i16> = (0..960)
        .map(|i| (((i * 73 + 19) % 32768) as i16).wrapping_sub(16384))
        .collect();
    let mut packet = [0u8; 1500];
    let packet_len = enc.encode(&input, &mut packet);
    assert!(packet_len > 0, "encode failed: {packet_len}");
    let packet_len = packet_len as usize;

    let mut rust_dec = OpusDecoder::new(48_000, 1).expect("rust decoder");

    let mut c_err = 0i32;
    let c_dec = unsafe { libopus_sys::opus_decoder_create(48_000, 1, &mut c_err) };
    assert!(!c_dec.is_null(), "c decoder create failed: {c_err}");

    let mut rust_ref = vec![0.0f32; 960];
    let mut c_ref = vec![0.0f32; 960];
    let rust_ref_ret =
        opus_decode_float(&mut rust_dec, &packet[..packet_len], &mut rust_ref, 960, 0);
    let c_ref_ret = unsafe {
        libopus_sys::opus_decode_float(
            c_dec,
            packet.as_ptr(),
            packet_len as i32,
            c_ref.as_mut_ptr(),
            960,
            0,
        )
    };
    assert_eq!(rust_ref_ret, c_ref_ret, "reference decode return mismatch");

    let dred = opurs::dnn::dred::decoder::OpusDRED::new();
    let mut rust_out = vec![0.0f32; 960];
    let mut c_out = vec![0.0f32; 960];
    let rust_ret = opus_decoder_dred_decode_float(&mut rust_dec, &dred, 0, &mut rust_out, 960);
    let c_ret = unsafe {
        libopus_sys::opus_decoder_dred_decode_float(c_dec, ptr::null(), 0, c_out.as_mut_ptr(), 960)
    };
    assert_eq!(rust_ret, c_ret, "dred decode return mismatch");

    let max_diff = max_abs_diff(&rust_out, &c_out);
    assert!(
        max_diff <= 1e-6,
        "dred decode output mismatch: max_abs_diff={max_diff}"
    );

    unsafe { libopus_sys::opus_decoder_destroy(c_dec) };
}
