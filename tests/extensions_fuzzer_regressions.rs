use opurs::internals::{
    opus_packet_extensions_count, opus_packet_extensions_count_ext,
    opus_packet_extensions_generate, opus_packet_extensions_parse,
    opus_packet_extensions_parse_ext,
};
use opurs::{opus_packet_unpad, OpusDecoder};

const MAX_FRAMES: usize = 48;
const MAX_EXTENSIONS: i32 = 256;
const MAX_DECODE_FRAME_SIZE: i32 = 1920;

#[test]
fn extensions_fuzzer_crash_i_zict_ea_does_not_panic() {
    // CI fuzzer crash artifact:
    // fuzz/artifacts/extensions_fuzzer/crash-37dd2c7cb844105201b427a99ab8ddc18d6f69d9
    // Base64: iZICtEA=
    let data = [137u8, 146, 2, 180, 64];

    let run = std::panic::catch_unwind(|| {
        let nb_frames = (data[0] % MAX_FRAMES as u8 + 1) as i32;
        let payload = &data[1..];

        let _ = opus_packet_extensions_count(payload, nb_frames);

        let mut per_frame = [0i32; MAX_FRAMES];
        if let Ok(total) = opus_packet_extensions_count_ext(
            payload,
            &mut per_frame[..nb_frames as usize],
            nb_frames,
        ) {
            let max_ext = total.clamp(0, MAX_EXTENSIONS);
            let _ = opus_packet_extensions_parse_ext(
                payload,
                max_ext,
                &per_frame[..nb_frames as usize],
                nb_frames,
            );
        }

        if let Ok(exts) = opus_packet_extensions_parse(payload, MAX_EXTENSIONS, nb_frames) {
            let mut out = vec![0u8; payload.len().saturating_add(64)];
            let _ = opus_packet_extensions_generate(&mut out, &exts, nb_frames, true);
        }

        let packet = payload.to_vec();
        let mut unpadded = packet.clone();
        let unpadded_len = opus_packet_unpad(&mut unpadded);
        if unpadded_len > 0 {
            unpadded.truncate(unpadded_len as usize);
        }

        for candidate in [&packet[..], &unpadded[..]] {
            for ignore_extensions in [false, true] {
                if let Ok(mut dec) = OpusDecoder::new(96_000, 2) {
                    dec.set_ignore_extensions(ignore_extensions);
                    let mut pcm = vec![0i16; MAX_DECODE_FRAME_SIZE as usize * 2];
                    let _ = dec.decode(candidate, &mut pcm, MAX_DECODE_FRAME_SIZE, false);
                }
            }
        }
    });

    assert!(
        run.is_ok(),
        "extensions_fuzzer regression input must not panic"
    );
}
