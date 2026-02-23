#![cfg(feature = "tools")]
#![allow(non_snake_case)]

extern crate opurs;

use libopus_sys::{opus_encode, opus_encoder_create, opus_encoder_ctl, opus_encoder_destroy};
use opurs::{
    Bitrate, OPUS_APPLICATION_RESTRICTED_LOWDELAY, OPUS_SET_BITRATE_REQUEST,
    OPUS_SET_COMPLEXITY_REQUEST, OPUS_SET_PREDICTION_DISABLED_REQUEST,
};

fn first_diff_frame(prediction_disabled: bool) -> Option<(usize, usize, usize)> {
    let sample_rate: i32 = 48000;
    let channels: i32 = 2;
    let frame_size: i32 = 960;
    let bitrate: i32 = 10_000;
    let complexity: i32 = 10;

    let pcm_path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("opus_newvectors/testvector11.dec");
    let pcm_bytes = std::fs::read(&pcm_path).expect("read testvector11.dec");
    let bytes_per_second = 48_000usize * 2 * 2;
    let pcm_bytes = &pcm_bytes[..pcm_bytes.len().min(3 * bytes_per_second)];
    let mut pcm = Vec::with_capacity(pcm_bytes.len() / 2);
    for ch in pcm_bytes.chunks_exact(2) {
        pcm.push(i16::from_le_bytes([ch[0], ch[1]]));
    }

    let mut rust_enc =
        opurs::OpusEncoder::new(sample_rate, channels, OPUS_APPLICATION_RESTRICTED_LOWDELAY)
            .expect("rust encoder create");
    rust_enc.set_bitrate(Bitrate::Bits(bitrate));
    rust_enc.set_complexity(complexity).unwrap();
    rust_enc.set_prediction_disabled(prediction_disabled);

    let mut c_err = 0;
    let c_enc = unsafe {
        opus_encoder_create(
            sample_rate,
            channels,
            OPUS_APPLICATION_RESTRICTED_LOWDELAY,
            &mut c_err,
        )
    };
    assert!(!c_enc.is_null(), "c encoder create failed: {c_err}");
    unsafe {
        opus_encoder_ctl(c_enc, OPUS_SET_BITRATE_REQUEST, bitrate);
        opus_encoder_ctl(c_enc, OPUS_SET_COMPLEXITY_REQUEST, complexity);
        opus_encoder_ctl(
            c_enc,
            OPUS_SET_PREDICTION_DISABLED_REQUEST,
            if prediction_disabled { 1 } else { 0 },
        );
    }

    let mut rust_out = vec![0u8; 4000];
    let mut c_out = vec![0u8; 4000];
    let frame_samples = frame_size as usize * channels as usize;
    for (frame_idx, frame) in pcm.chunks_exact(frame_samples).enumerate() {
        let rust_len = rust_enc.encode(frame, &mut rust_out) as usize;
        let c_len = unsafe {
            opus_encode(
                c_enc,
                frame.as_ptr(),
                frame_size,
                c_out.as_mut_ptr(),
                c_out.len() as i32,
            ) as usize
        };
        if rust_len != c_len || rust_out[..rust_len] != c_out[..c_len] {
            unsafe { opus_encoder_destroy(c_enc) };
            return Some((frame_idx, rust_len, c_len));
        }
    }

    unsafe { opus_encoder_destroy(c_enc) };
    None
}

#[test]
#[ignore]
fn debug_rld_prediction_toggle() {
    let normal = first_diff_frame(false);
    let pred_off = first_diff_frame(true);
    eprintln!("prediction_disabled=false first diff: {normal:?}");
    eprintln!("prediction_disabled=true first diff:  {pred_off:?}");
}
