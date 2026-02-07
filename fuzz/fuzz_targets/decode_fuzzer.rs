//! Opus decode fuzzer — ported from upstream opus_decode_fuzzer.c
//!
//! Treats input data as concatenated packets encoded by opus_demo:
//!   bytes 0..3: packet length (big-endian)
//!   bytes 4..7: encoder final range (byte 4 bit 0 reused as FEC flag)
//!   bytes 8+  : Opus packet including ToC
//!
//! Run with: cargo +nightly fuzz run decode_fuzzer
#![no_main]
#![allow(deprecated)]

use libfuzzer_sys::fuzz_target;
use unsafe_libopus::{
    opus_decode, opus_decoder_create, opus_decoder_ctl, opus_decoder_destroy,
    opus_packet_get_bandwidth, opus_packet_get_nb_channels, OPUS_BANDWIDTH_NARROWBAND,
};

const MAX_FRAME_SAMP: i32 = 5760;
const MAX_PACKET: usize = 1500;
/// 4 bytes packet length + 4 bytes encoder final range
const SETUP_BYTE_COUNT: usize = 8;

const SAMP_FREQS: [i32; 5] = [8000, 12000, 16000, 24000, 48000];

fuzz_target!(|data: &[u8]| {
    // Not enough data to set up the decoder (+1 for the ToC byte)
    if data.len() < SETUP_BYTE_COUNT + 1 {
        return;
    }

    // Parse ToC from the first packet to determine sample rate and channels
    let toc = data[SETUP_BYTE_COUNT];
    let bandwidth = unsafe { opus_packet_get_bandwidth(&toc as *const u8) };
    let bw_idx = bandwidth - OPUS_BANDWIDTH_NARROWBAND;
    if !(0..5).contains(&bw_idx) {
        return;
    }
    let fs = SAMP_FREQS[bw_idx as usize];
    let channels = unsafe { opus_packet_get_nb_channels(&toc as *const u8) };

    let mut err = 0;
    let dec = unsafe { opus_decoder_create(fs, channels, &mut err) };
    if err != 0 || dec.is_null() {
        return;
    }

    let mut pcm = vec![0i16; MAX_FRAME_SAMP as usize * channels as usize];

    let mut i = 0usize;
    loop {
        if i + SETUP_BYTE_COUNT > data.len() {
            break;
        }

        let len = (data[i] as u32) << 24
            | (data[i + 1] as u32) << 16
            | (data[i + 2] as u32) << 8
            | (data[i + 3] as u32);
        let len = len as i32;
        if len > MAX_PACKET as i32 || len < 0 {
            break;
        }
        let len = len as usize;

        // Byte 4 is repurposed: bit 0 determines if FEC is used
        let fec = (data[i + 4] & 1) as i32;

        if len == 0 {
            // Lost packet — use PLC
            let mut frame_size: i32 = 0;
            unsafe {
                let _ = opus_decoder_ctl!(&mut *dec, 4039, &mut frame_size);
                let _ = opus_decode(&mut *dec, &[], pcm.as_mut_ptr(), frame_size, fec);
            }
        } else {
            if i + SETUP_BYTE_COUNT + len > data.len() {
                break;
            }
            let packet = &data[i + SETUP_BYTE_COUNT..i + SETUP_BYTE_COUNT + len];
            unsafe {
                let _ = opus_decode(&mut *dec, packet, pcm.as_mut_ptr(), MAX_FRAME_SAMP, fec);
            }
        }

        i += SETUP_BYTE_COUNT + len;
    }

    unsafe {
        opus_decoder_destroy(dec);
    }
});
