//! Opus decode fuzzer — ported from upstream opus_decode_fuzzer.c
//!
//! Treats input data as concatenated packets encoded by opus_demo:
//!   bytes 0..3: packet length (big-endian)
//!   bytes 4..7: encoder final range (byte 4 bit 0 reused as FEC flag)
//!   bytes 8+  : Opus packet including ToC
//!
//! Run with: cargo +nightly fuzz run decode_fuzzer
#![no_main]

use libfuzzer_sys::fuzz_target;
use opurs::{
    opus_packet_get_bandwidth, opus_packet_get_nb_channels, OpusDecoder, OPUS_BANDWIDTH_NARROWBAND,
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
    let bandwidth = opus_packet_get_bandwidth(toc);
    let bw_idx = bandwidth - OPUS_BANDWIDTH_NARROWBAND;
    if !(0..5).contains(&bw_idx) {
        return;
    }
    let fs = SAMP_FREQS[bw_idx as usize];
    let channels = opus_packet_get_nb_channels(toc);

    let mut dec = match OpusDecoder::new(fs, channels as usize) {
        Ok(d) => d,
        Err(_) => return,
    };

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
        let fec = data[i + 4] & 1 != 0;

        if len == 0 {
            // Lost packet — use PLC
            let frame_size = dec.last_packet_duration();
            if frame_size > 0 {
                let _ = dec.decode(
                    &[],
                    &mut pcm[..frame_size as usize * channels as usize],
                    frame_size,
                    fec,
                );
            }
        } else {
            if i + SETUP_BYTE_COUNT + len > data.len() {
                break;
            }
            let packet = &data[i + SETUP_BYTE_COUNT..i + SETUP_BYTE_COUNT + len];
            let _ = dec.decode(packet, &mut pcm, MAX_FRAME_SAMP, fec);
        }

        i += SETUP_BYTE_COUNT + len;
    }
});
