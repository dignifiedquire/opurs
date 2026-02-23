//! Focused regressions from vector parity failures.

#![cfg(feature = "tools")]

use opurs::tools::demo::{
    opus_demo_encode, Application, Channels, DnnOptions, EncodeArgs, EncoderOptions, FrameSize,
    OpusBackend, SampleRate,
};
use std::path::PathBuf;

const TESTVECTOR11_PREFIX_SECONDS: usize = 3;

fn testvector11_pcm_prefix() -> Vec<u8> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("opus_newvectors/testvector11.dec");
    let pcm = std::fs::read(&path).unwrap_or_else(|e| {
        panic!(
            "reading {} failed (required for vector regression tests): {e}",
            path.display()
        )
    });
    let bytes_per_second = 48_000usize * 2 * 2;
    let keep = TESTVECTOR11_PREFIX_SECONDS * bytes_per_second;
    pcm[..pcm.len().min(keep)].to_vec()
}

fn first_mux_packet_diff(upstream: &[u8], rust: &[u8]) -> Option<String> {
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
        let upstream_range = u32::from_be_bytes([
            upstream[upos + 4],
            upstream[upos + 5],
            upstream[upos + 6],
            upstream[upos + 7],
        ]);
        let rust_range = u32::from_be_bytes([
            rust[rpos + 4],
            rust[rpos + 5],
            rust[rpos + 6],
            rust[rpos + 7],
        ]);

        if upstream_len != rust_len {
            return Some(format!(
                "pkt={} len upstream/rust={}/{}",
                packet, upstream_len, rust_len
            ));
        }
        if upstream_range != rust_range {
            return Some(format!(
                "pkt={} range upstream/rust={:#010x}/{:#010x}",
                packet, upstream_range, rust_range
            ));
        }
        if upstream_len <= 0 {
            return Some(format!("pkt={} invalid length={}", packet, upstream_len));
        }
        let payload_len = upstream_len as usize;
        if upos + 8 + payload_len > upstream.len() || rpos + 8 + payload_len > rust.len() {
            return Some(format!(
                "pkt={} truncated payload len={} upstream_rem={} rust_rem={}",
                packet,
                payload_len,
                upstream.len().saturating_sub(upos + 8),
                rust.len().saturating_sub(rpos + 8)
            ));
        }

        let upstream_payload = &upstream[upos + 8..upos + 8 + payload_len];
        let rust_payload = &rust[rpos + 8..rpos + 8 + payload_len];
        if upstream_payload != rust_payload {
            let first_byte = upstream_payload
                .iter()
                .zip(rust_payload.iter())
                .position(|(u, r)| u != r)
                .unwrap_or(0);
            return Some(format!(
                "pkt={} byte={} upstream/rust={:#04x}/{:#04x}",
                packet, first_byte, upstream_payload[first_byte], rust_payload[first_byte]
            ));
        }

        upos += 8 + payload_len;
        rpos += 8 + payload_len;
        packet += 1;
    }

    if upos != upstream.len() || rpos != rust.len() {
        return Some(format!(
            "container length mismatch upstream/rust={}/{} parsed_upstream/rust={}/{}",
            upstream.len(),
            rust.len(),
            upos,
            rpos
        ));
    }

    None
}

fn assert_rld_10k_frame_parity(frame_size: FrameSize) {
    let pcm = testvector11_pcm_prefix();
    let encode_args = EncodeArgs {
        application: Application::RestrictedLowDelay,
        sample_rate: SampleRate::R48000,
        channels: Channels::Stereo,
        bitrate: 10_000,
        options: EncoderOptions {
            framesize: frame_size,
            ..Default::default()
        },
    };
    let dnn = DnnOptions::default();

    let (upstream_encoded, upstream_pre_skip) =
        opus_demo_encode(OpusBackend::Upstream, &pcm, encode_args, &dnn);
    let (rust_encoded, rust_pre_skip) =
        opus_demo_encode(OpusBackend::Rust, &pcm, encode_args, &dnn);

    assert_eq!(
        rust_pre_skip, upstream_pre_skip,
        "pre-skip mismatch for frame_size={:?}",
        frame_size
    );
    if upstream_encoded != rust_encoded {
        let detail = first_mux_packet_diff(&upstream_encoded, &rust_encoded)
            .unwrap_or_else(|| "bitstream differs".to_string());
        panic!(
            "testvector11 RLD 10kbps parity failed for frame_size={:?}: {}",
            frame_size, detail
        );
    }
}

#[test]
fn testvector11_rld_10kbps_20ms_parity() {
    assert_rld_10k_frame_parity(FrameSize::Ms20);
}

#[test]
fn testvector11_rld_10kbps_40ms_parity() {
    assert_rld_10k_frame_parity(FrameSize::Ms40);
}

#[test]
fn testvector11_rld_10kbps_60ms_parity() {
    assert_rld_10k_frame_parity(FrameSize::Ms60);
}
