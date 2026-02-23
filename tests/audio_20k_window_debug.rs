#![cfg(feature = "tools")]

use opurs::tools::demo::{
    opus_demo_encode, Application, Channels, DnnOptions, EncodeArgs, EncoderOptions, FrameSize,
    OpusBackend, SampleRate,
};
use std::path::PathBuf;

fn first_mux_packet_diff(upstream: &[u8], rust: &[u8]) -> Option<(usize, String)> {
    let mut upos = 0usize;
    let mut rpos = 0usize;
    let mut packet = 0usize;

    while upos + 8 <= upstream.len() && rpos + 8 <= rust.len() {
        let upstream_len = i32::from_be_bytes([
            upstream[upos],
            upstream[upos + 1],
            upstream[upos + 2],
            upstream[upos + 3],
        ]);
        let rust_len =
            i32::from_be_bytes([rust[rpos], rust[rpos + 1], rust[rpos + 2], rust[rpos + 3]]);
        if upstream_len != rust_len {
            return Some((packet, format!("len {upstream_len}/{rust_len}")));
        }
        if upstream_len <= 0 {
            return Some((packet, format!("invalid len {upstream_len}")));
        }
        let l = upstream_len as usize;
        let up = &upstream[upos + 8..upos + 8 + l];
        let rp = &rust[rpos + 8..rpos + 8 + l];
        if up != rp {
            let b = up
                .iter()
                .zip(rp.iter())
                .position(|(a, b)| a != b)
                .unwrap_or(0);
            return Some((packet, format!("byte {b} {:02x}/{:02x}", up[b], rp[b])));
        }
        upos += 8 + l;
        rpos += 8 + l;
        packet += 1;
    }

    None
}

#[test]
#[ignore]
fn debug_audio_20k_windows() {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("opus_newvectors/testvector11.dec");
    let pcm = std::fs::read(path).unwrap();
    let frame_bytes = 960usize * 2 * 2;
    let windows = [0usize, 40, 69, 80, 120, 180, 200, 220, 230, 234];

    let enc = EncodeArgs {
        application: Application::Audio,
        sample_rate: SampleRate::R48000,
        channels: Channels::Stereo,
        bitrate: 20_000,
        options: EncoderOptions {
            framesize: FrameSize::Ms20,
            ..Default::default()
        },
    };
    let dnn = DnnOptions::default();

    for &start_frame in &windows {
        let start = start_frame * frame_bytes;
        let end = (start + 180 * frame_bytes).min(pcm.len());
        let slice = &pcm[start..end];
        let (u, _) = opus_demo_encode(OpusBackend::Upstream, slice, enc, &dnn);
        let (r, _) = opus_demo_encode(OpusBackend::Rust, slice, enc, &dnn);
        let diff = first_mux_packet_diff(&u, &r);
        eprintln!("start_frame={start_frame} diff={diff:?}");
    }
}
