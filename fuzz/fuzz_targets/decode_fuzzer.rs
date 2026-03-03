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
use libopus_sys::{
    opus_decode as c_opus_decode, opus_decoder_create as c_opus_decoder_create,
    opus_decoder_destroy as c_opus_decoder_destroy, OpusDecoder as COpusDecoder, OPUS_OK,
};
use opurs::{
    opus_packet_get_bandwidth, opus_packet_get_nb_channels, OpusDecoder, OPUS_BANDWIDTH_NARROWBAND,
};

const MAX_FRAME_SAMP: i32 = 5760;
const MAX_PACKET: usize = 1500;
/// 4 bytes packet length + 4 bytes encoder final range
const SETUP_BYTE_COUNT: usize = 8;
const MAX_DECODES: i32 = 12;

const SAMP_FREQS: [i32; 5] = [8000, 12000, 16000, 24000, 48000];

struct CDecoder(*mut COpusDecoder);

impl CDecoder {
    fn new(fs: i32, channels: i32) -> Option<Self> {
        let mut err = 0i32;
        // SAFETY: The sample rate/channels come from valid Opus ToC parsing.
        let ptr = unsafe { c_opus_decoder_create(fs, channels, &mut err as *mut _) };
        if ptr.is_null() || err != OPUS_OK as i32 {
            return None;
        }
        Some(Self(ptr))
    }

    fn decode(&mut self, packet: &[u8], pcm: &mut [i16], frame_size: i32, fec: bool) {
        // SAFETY: `self.0` is a valid decoder pointer from `opus_decoder_create`.
        // Packet/PCM pointers are valid for the provided lengths.
        let _ = unsafe {
            c_opus_decode(
                self.0,
                packet.as_ptr(),
                packet.len() as i32,
                pcm.as_mut_ptr(),
                frame_size,
                i32::from(fec),
            )
        };
    }
}

impl Drop for CDecoder {
    fn drop(&mut self) {
        // SAFETY: Decoder pointer was allocated by `opus_decoder_create`.
        unsafe { c_opus_decoder_destroy(self.0) };
    }
}

fuzz_target!(|data: &[u8]| {
    // Not enough data to set up the decoder (+1 for the ToC byte)
    if data.len() < SETUP_BYTE_COUNT + 1 {
        return;
    }

    // Parse ToC from the first packet to determine sample rate and channels
    let toc = data[SETUP_BYTE_COUNT];
    let bandwidth = opus_packet_get_bandwidth(toc);
    let bw_idx = bandwidth - OPUS_BANDWIDTH_NARROWBAND;
    let fs = SAMP_FREQS[bw_idx as usize];
    let channels = opus_packet_get_nb_channels(toc);

    let mut dec = match OpusDecoder::new(fs, channels as usize) {
        Ok(d) => d,
        Err(_) => return,
    };
    let mut c_dec = match CDecoder::new(fs, channels) {
        Some(d) => d,
        None => return,
    };

    let mut pcm = vec![0i16; MAX_FRAME_SAMP as usize * channels as usize];
    let mut c_pcm = vec![0i16; MAX_FRAME_SAMP as usize * channels as usize];

    let mut i = 0usize;
    let mut num_decodes = 0;
    while i + SETUP_BYTE_COUNT < data.len() && {
        num_decodes += 1;
        num_decodes <= MAX_DECODES
    } {
        let len = (data[i] as u32) << 24
            | (data[i + 1] as u32) << 16
            | (data[i + 2] as u32) << 8
            | (data[i + 3] as u32);
        let len = len as i32;
        if len > MAX_PACKET as i32 || len < 0 || i + SETUP_BYTE_COUNT + len as usize > data.len() {
            break;
        }
        let len = len as usize;

        // Byte 4 is repurposed: bit 0 determines if FEC is used
        let fec = data[i + 4] & 1 != 0;

        if len == 0 {
            // Lost packet — use PLC
            let frame_size = dec.last_packet_duration();
            let _ = dec.decode(
                &[],
                &mut pcm[..frame_size as usize * channels as usize],
                frame_size,
                fec,
            );
            c_dec.decode(
                &[],
                &mut c_pcm[..frame_size as usize * channels as usize],
                frame_size,
                fec,
            );
        } else {
            let packet = &data[i + SETUP_BYTE_COUNT..i + SETUP_BYTE_COUNT + len];
            let _ = dec.decode(packet, &mut pcm, MAX_FRAME_SAMP, fec);
            c_dec.decode(packet, &mut c_pcm, MAX_FRAME_SAMP, fec);
        }

        i += SETUP_BYTE_COUNT + len;
    }
});
