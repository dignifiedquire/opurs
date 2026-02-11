//! Tests for Opus packet padding overflow handling.
//!
//! Upstream C: tests/test_opus_padding.c

use opurs::{opus_get_version_string, OpusDecoder, OPUS_INVALID_PACKET};

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

    let mut decoder = OpusDecoder::new(48000, 2).expect("Failed to create decoder");

    let result = decoder.decode(&packet, &mut out, 5760, false);

    assert_eq!(
        result, OPUS_INVALID_PACKET,
        "Padding overflow test: expected OPUS_INVALID_PACKET ({OPUS_INVALID_PACKET}), got {result}"
    );
}
