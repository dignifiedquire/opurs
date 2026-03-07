//! Tests for Opus packet padding overflow handling.
//!
//! Upstream C: tests/test_opus_padding.c

use opurs::{opus_get_version_string, Channels, OpusDecoder, SampleRate};

/// Test that a crafted large padding packet returns OPUS_INVALID_PACKET
/// rather than causing a buffer overflow.
///
/// Upstream C: tests/test_opus_padding.c:test_overflow()
#[test]
fn test_padding_overflow() {
    let version = opus_get_version_string();
    eprintln!("Testing {version} padding.");

    const PACKET_SIZE: usize = 16909318;

    let mut packet = vec![0xffu8; PACKET_SIZE];
    packet[0] = 0xff;
    packet[1] = 0x41;
    // bytes 2..PACKET_SIZE-1 are already 0xff
    packet[PACKET_SIZE - 1] = 0x0b;

    let mut out = vec![0i16; 5760 * 2];

    let mut decoder =
        OpusDecoder::new(SampleRate::Hz48000, Channels::Stereo).expect("Failed to create decoder");

    let result = decoder.decode(&packet, &mut out, 5760, false);

    assert!(
        result.is_err(),
        "Padding overflow test: expected OPUS_INVALID_PACKET, got {result:?}"
    );
}
