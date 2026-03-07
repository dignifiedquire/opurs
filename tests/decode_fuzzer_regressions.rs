use opurs::{
    opus_packet_get_bandwidth, opus_packet_get_nb_channels, Bandwidth, OpusDecoder, SampleRate,
};

const MAX_FRAME_SAMP: i32 = 5760;
const MAX_PACKET: usize = 1500;
const SETUP_BYTE_COUNT: usize = 8;
const MAX_DECODES: i32 = 12;
const SAMP_FREQS: [i32; 5] = [8000, 12000, 16000, 24000, 48000];

#[test]
fn decode_fuzzer_crash_162fefed_does_not_panic() {
    // CI fuzzer crash artifact:
    // fuzz/artifacts/decode_fuzzer/crash-162fefed2287f97d38bc7390ba19e3d2d67f4bac
    // Base64: AAAADjj//37/QQMAAAAAAAAtAPj/ph3/A/+pqQAKqUQOAQ==
    let data = [
        0x00u8, 0x00, 0x00, 0x0e, 0x38, 0xff, 0xff, 0x7e, 0xff, 0x41, 0x03, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x2d, 0x00, 0xf8, 0xff, 0xa6, 0x1d, 0xff, 0x03, 0xff, 0xa9, 0xa9, 0x00, 0x0a,
        0xa9, 0x44, 0x0e, 0x01,
    ];

    let run = std::panic::catch_unwind(|| {
        if data.len() < SETUP_BYTE_COUNT + 1 {
            return;
        }

        let toc = data[SETUP_BYTE_COUNT];
        let bandwidth = opus_packet_get_bandwidth(toc);
        let bw_idx = match bandwidth {
            Bandwidth::Narrowband => 0,
            Bandwidth::Mediumband => 1,
            Bandwidth::Wideband => 2,
            Bandwidth::Superwideband => 3,
            Bandwidth::Fullband => 4,
        };
        let fs = SAMP_FREQS[bw_idx];
        let ch = opus_packet_get_nb_channels(toc);
        let channels = i32::from(ch);

        let sample_rate = match SampleRate::try_from(fs) {
            Ok(sr) => sr,
            Err(_) => return,
        };
        let mut dec = match OpusDecoder::new(sample_rate, ch) {
            Ok(dec) => dec,
            Err(_) => return,
        };
        let mut pcm = vec![0i16; MAX_FRAME_SAMP as usize * channels as usize];

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
            if len > MAX_PACKET as i32
                || len < 0
                || i + SETUP_BYTE_COUNT + len as usize > data.len()
            {
                break;
            }
            let len = len as usize;
            let fec = data[i + 4] & 1 != 0;

            if len == 0 {
                let frame_size = dec.last_packet_duration();
                let _ = dec.decode(
                    &[],
                    &mut pcm[..frame_size as usize * channels as usize],
                    frame_size,
                    fec,
                );
            } else {
                let packet = &data[i + SETUP_BYTE_COUNT..i + SETUP_BYTE_COUNT + len];
                let _ = dec.decode(packet, &mut pcm, MAX_FRAME_SAMP, fec);
            }

            i += SETUP_BYTE_COUNT + len;
        }
    });

    assert!(run.is_ok(), "decode_fuzzer regression input must not panic");
}
