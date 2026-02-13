//! Frame-by-frame encoder comparison between Rust and C.
//! Requires `--features tools`.
//!
//! Run with: cargo test --release --features tools --test encoder_comparison -- --nocapture

#![cfg(feature = "tools")]
#![allow(non_snake_case)]

extern crate opurs;

use libopus_sys::{opus_encode, opus_encoder_create, opus_encoder_ctl, opus_encoder_destroy};
use opurs::{OPUS_APPLICATION_AUDIO, OPUS_SET_BITRATE_REQUEST, OPUS_SET_COMPLEXITY_REQUEST};

/// Simple deterministic PRNG for audio generation
struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Self {
        Rng(seed)
    }
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0
    }
    fn next_i16(&mut self) -> i16 {
        (self.next() >> 48) as i16
    }
}

#[test]
fn compare_encoder_frame_by_frame() {
    let sample_rate: i32 = 48000;
    let channels: i32 = 2;
    let frame_size: i32 = 960; // 20ms at 48kHz
    let bitrate: i32 = 64000;
    let complexity: i32 = 10;
    let num_frames = 500;

    // Create Rust encoder
    let mut rust_enc = opurs::OpusEncoder::new(sample_rate, channels, OPUS_APPLICATION_AUDIO)
        .expect("Rust encoder create failed");
    rust_enc.set_bitrate(opurs::Bitrate::Bits(bitrate));
    let _ = rust_enc.set_complexity(complexity);

    // Create C encoder
    let mut c_error: i32 = 0;
    let c_enc =
        unsafe { opus_encoder_create(sample_rate, channels, OPUS_APPLICATION_AUDIO, &mut c_error) };
    assert!(!c_enc.is_null(), "C encoder create failed: {c_error}");
    unsafe {
        opus_encoder_ctl(c_enc, OPUS_SET_BITRATE_REQUEST, bitrate);
        opus_encoder_ctl(c_enc, OPUS_SET_COMPLEXITY_REQUEST, complexity);
    }

    let mut rng = Rng::new(12345);
    let pcm_samples = frame_size as usize * channels as usize;
    let mut pcm = vec![0i16; pcm_samples];
    let mut rust_out = vec![0u8; 4000];
    let mut c_out = vec![0u8; 4000];

    let mut first_diff_frame = None;

    for frame in 0..num_frames {
        // Generate deterministic audio
        for s in pcm.iter_mut() {
            *s = rng.next_i16();
        }

        // Encode with Rust
        let rust_len = rust_enc.encode(&pcm, &mut rust_out) as usize;

        // Encode with C
        let c_len = unsafe {
            opus_encode(
                c_enc,
                pcm.as_ptr(),
                frame_size,
                c_out.as_mut_ptr(),
                c_out.len() as i32,
            ) as usize
        };

        if (rust_len != c_len || rust_out[..rust_len] != c_out[..c_len])
            && first_diff_frame.is_none()
        {
            first_diff_frame = Some(frame);
            eprintln!("Frame {frame}: DIFFER (rust_len={rust_len}, c_len={c_len})");
            if rust_len == c_len {
                let first_byte = (0..rust_len).find(|&i| rust_out[i] != c_out[i]).unwrap();
                eprintln!(
                    "  First diff at byte {first_byte}/{rust_len}: rust={:#04x} c={:#04x}",
                    rust_out[first_byte], c_out[first_byte]
                );
            }
            // Continue to count total diffs
        }
    }

    unsafe {
        opus_encoder_destroy(c_enc);
    }

    match first_diff_frame {
        Some(frame) => {
            panic!("First encoder divergence at frame {frame} of {num_frames}");
        }
        None => {
            eprintln!("All {num_frames} frames match!");
        }
    }
}

#[test]
fn compare_encoder_low_bitrate() {
    // Test at 10kbps (SILK mode) — should match
    let sample_rate: i32 = 48000;
    let channels: i32 = 1;
    let frame_size: i32 = 960;
    let bitrate: i32 = 10000;
    let complexity: i32 = 10;
    let num_frames = 200;

    let mut rust_enc = opurs::OpusEncoder::new(sample_rate, channels, OPUS_APPLICATION_AUDIO)
        .expect("Rust encoder create failed");
    rust_enc.set_bitrate(opurs::Bitrate::Bits(bitrate));
    let _ = rust_enc.set_complexity(complexity);

    let mut c_error: i32 = 0;
    let c_enc =
        unsafe { opus_encoder_create(sample_rate, channels, OPUS_APPLICATION_AUDIO, &mut c_error) };
    assert!(!c_enc.is_null());
    unsafe {
        opus_encoder_ctl(c_enc, OPUS_SET_BITRATE_REQUEST, bitrate);
        opus_encoder_ctl(c_enc, OPUS_SET_COMPLEXITY_REQUEST, complexity);
    }

    let mut rng = Rng::new(12345);
    let pcm_samples = frame_size as usize * channels as usize;
    let mut pcm = vec![0i16; pcm_samples];
    let mut rust_out = vec![0u8; 4000];
    let mut c_out = vec![0u8; 4000];
    let mut diffs = 0;

    for frame in 0..num_frames {
        for s in pcm.iter_mut() {
            *s = rng.next_i16();
        }

        let rust_len = rust_enc.encode(&pcm, &mut rust_out) as usize;
        let c_len = unsafe {
            opus_encode(
                c_enc,
                pcm.as_ptr(),
                frame_size,
                c_out.as_mut_ptr(),
                c_out.len() as i32,
            ) as usize
        };

        if rust_len != c_len || rust_out[..rust_len] != c_out[..c_len] {
            if diffs == 0 {
                eprintln!("Low bitrate: first diff at frame {frame}");
            }
            diffs += 1;
        }
    }

    unsafe {
        opus_encoder_destroy(c_enc);
    }
    eprintln!("Low bitrate (10kbps mono): {diffs}/{num_frames} frames differ");
}
