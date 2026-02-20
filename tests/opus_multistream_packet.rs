//! Differential tests for multistream packet pad/unpad helpers.
//! Requires `--features tools`.

#![cfg(feature = "tools")]

use libopus_sys::opus_multistream_packet_pad as c_multistream_packet_pad;
use libopus_sys::opus_multistream_packet_unpad as c_multistream_packet_unpad;
use opurs::{
    opus_multistream_packet_pad, opus_multistream_packet_unpad, opus_packet_parse, OpusEncoder,
    OPUS_APPLICATION_AUDIO, OPUS_BAD_ARG, OPUS_OK,
};

fn encode_size(size: i32, out: &mut [u8]) -> usize {
    if size < 252 {
        out[0] = size as u8;
        1
    } else {
        out[0] = (252 + (size & 0x3)) as u8;
        out[1] = ((size - out[0] as i32) >> 2) as u8;
        2
    }
}

fn encode_mono_packet(seed: i16) -> Vec<u8> {
    let mut enc = OpusEncoder::new(48000, 1, OPUS_APPLICATION_AUDIO).expect("encoder create");
    let pcm: Vec<i16> = (0..960).map(|i| i as i16 ^ seed).collect();
    let mut out = vec![0u8; 1500];
    let len = enc.encode(&pcm, &mut out);
    assert!(len > 0);
    out.truncate(len as usize);
    out
}

fn to_self_delimited_single_frame(packet: &[u8]) -> Vec<u8> {
    let mut toc = 0u8;
    let mut frames = [0usize; 48];
    let mut sizes = [0i16; 48];
    let mut payload_offset = 0i32;
    let count = opus_packet_parse(
        packet,
        Some(&mut toc),
        Some(&mut frames),
        &mut sizes,
        Some(&mut payload_offset),
    );
    assert_eq!(count, 1, "expected a single-frame packet");
    let frame_len = sizes[0] as usize;
    let frame_off = frames[0];
    let mut out = vec![0u8; 2 + frame_len];
    out[0] = toc;
    let sz = encode_size(frame_len as i32, &mut out[1..]);
    out.truncate(1 + sz + frame_len);
    out[(1 + sz)..].copy_from_slice(&packet[frame_off..frame_off + frame_len]);
    out
}

fn build_two_stream_packet() -> Vec<u8> {
    let packet0 = encode_mono_packet(0x1357);
    let packet1 = encode_mono_packet(0x2468);
    let mut out = to_self_delimited_single_frame(&packet0);
    out.extend_from_slice(&packet1);
    out
}

#[test]
fn multistream_packet_pad_matches_upstream_c() {
    let base = build_two_stream_packet();
    let len = base.len() as i32;
    let new_len = len + 79;

    let mut rust_buf = vec![0u8; new_len as usize];
    rust_buf[..base.len()].copy_from_slice(&base);
    let mut c_buf = rust_buf.clone();

    let rust_ret = opus_multistream_packet_pad(&mut rust_buf, len, new_len, 2);
    let c_ret = unsafe { c_multistream_packet_pad(c_buf.as_mut_ptr(), len, new_len, 2) };

    assert_eq!(rust_ret, c_ret, "return code mismatch");
    assert_eq!(rust_ret, OPUS_OK, "expected successful pad");
    assert_eq!(
        &rust_buf[..new_len as usize],
        &c_buf[..new_len as usize],
        "padded payload mismatch"
    );
}

#[test]
fn multistream_packet_unpad_matches_upstream_c() {
    let base = build_two_stream_packet();
    let len = base.len() as i32;
    let new_len = len + 96;

    let mut src = vec![0u8; new_len as usize];
    src[..base.len()].copy_from_slice(&base);
    let ret = opus_multistream_packet_pad(&mut src, len, new_len, 2);
    assert_eq!(ret, OPUS_OK, "failed to build padded packet");

    let mut rust_buf = src.clone();
    let mut c_buf = src.clone();

    let rust_ret = opus_multistream_packet_unpad(&mut rust_buf, new_len, 2);
    let c_ret = unsafe { c_multistream_packet_unpad(c_buf.as_mut_ptr(), new_len, 2) };

    assert_eq!(rust_ret, c_ret, "return code mismatch");
    assert!(rust_ret > 0, "expected successful unpad");
    assert_eq!(
        &rust_buf[..rust_ret as usize],
        &c_buf[..c_ret as usize],
        "unpadded payload mismatch"
    );
}

#[test]
fn multistream_packet_pad_bad_arg_parity() {
    let mut rust_buf = vec![0u8; 16];
    let mut c_buf = rust_buf.clone();

    let rust_ret = opus_multistream_packet_pad(&mut rust_buf, 0, 8, 2);
    let c_ret = unsafe { c_multistream_packet_pad(c_buf.as_mut_ptr(), 0, 8, 2) };

    assert_eq!(rust_ret, OPUS_BAD_ARG);
    assert_eq!(rust_ret, c_ret);
}
