//! Upstream extensions suite port.
//!
//! Upstream C: `tests/test_opus_extensions.c`

use opurs::internals::{
    opus_packet_extensions_count, opus_packet_extensions_count_ext,
    opus_packet_extensions_generate, opus_packet_extensions_parse,
    opus_packet_extensions_parse_ext, opus_packet_pad_impl, opus_packet_parse_impl,
    OpusExtensionData,
};
use opurs::{
    OpusEncoder, OpusRepacketizer, OPUS_APPLICATION_AUDIO, OPUS_BAD_ARG, OPUS_BUFFER_TOO_SMALL,
    OPUS_INVALID_PACKET, OPUS_OK,
};

fn ext(id: i32, frame: i32, data: &[u8]) -> OpusExtensionData {
    OpusExtensionData {
        id,
        frame,
        data: data.to_vec(),
    }
}

fn check_ext_data(expected: &[OpusExtensionData], parsed: &[OpusExtensionData]) {
    let mut prev_frame = -1;
    let mut j = 0usize;

    for out in parsed {
        assert!(
            out.frame >= prev_frame,
            "parsed extensions should be in frame order"
        );

        if out.frame > prev_frame {
            j = 0;
        }
        while j < expected.len() && expected[j].frame != out.frame {
            j += 1;
        }
        assert!(
            j < expected.len(),
            "missing matching frame in expected data"
        );

        assert_eq!(expected[j].id, out.id, "extension id mismatch");
        assert_eq!(
            expected[j].data.len(),
            out.data.len(),
            "extension len mismatch"
        );
        assert_eq!(expected[j].data, out.data, "extension data mismatch");

        prev_frame = out.frame;
        j += 1;
    }
}

fn next_u32(state: &mut u32) -> u32 {
    *state = state.wrapping_mul(1664525).wrapping_add(1013904223);
    *state
}

fn encode_mono_packet(seed: i16) -> Vec<u8> {
    let mut enc = OpusEncoder::new(48_000, 1, OPUS_APPLICATION_AUDIO).expect("encoder create");
    let pcm: Vec<i16> = (0..960).map(|i| i as i16 ^ seed).collect();
    let mut out = vec![0u8; 1500];
    let len = enc.encode(&pcm, &mut out);
    assert!(len > 0, "encode failed: {len}");
    out.truncate(len as usize);
    out
}

/// Upstream C: tests/test_opus_extensions.c:test_extensions_generate_success
#[test]
fn extensions_generate_success() {
    let exts = vec![
        ext(3, 0, b"a"),
        ext(32, 10, b"DRED"),
        ext(33, 1, b"NOT DRED"),
        ext(4, 4, b""),
    ];

    let mut packet = [0u8; 27];
    let len = opus_packet_extensions_generate(&mut packet, &exts, 11, true).expect("generate");
    assert_eq!(len, 27, "unexpected generated length");

    let p = &packet[..];
    assert_eq!(&p[0..4], &[1, 1, 1, 1], "expected 4 bytes of padding");
    assert_eq!(p[4] >> 1, 3, "expected id=3");
    assert_eq!(p[4] & 1, 1, "expected short extension L-bit set");
    assert_eq!(p[5], b'a', "expected id=3 payload");
    assert_eq!(p[6], 0x02, "bad frame separator");
    assert_eq!(p[7] >> 1, 33, "expected id=33");
    assert_eq!(p[7] & 1, 1, "expected long extension L-bit set");
    assert_eq!(p[8], 8, "expected id=33 length byte");
    assert_eq!(&p[9..17], b"NOT DRED", "expected id=33 payload");
    assert_eq!(p[17], 0x03, "bad frame separator for +3");
    assert_eq!(p[18], 0x03, "bad frame increment");
    assert_eq!(p[19] >> 1, 4, "expected id=4");
    assert_eq!(p[19] & 1, 0, "expected id=4 short L-bit unset");
    assert_eq!(p[20], 0x03, "bad frame separator for +6");
    assert_eq!(p[21], 0x06, "bad frame increment");
    assert_eq!(p[22] >> 1, 32, "expected id=32");
    assert_eq!(p[22] & 1, 0, "expected id=32 long L-bit unset");
    assert_eq!(&p[23..27], b"DRED", "expected id=32 payload");
}

/// Upstream C: tests/test_opus_extensions.c:test_extensions_generate_zero
#[test]
fn extensions_generate_zero() {
    let mut packet: [u8; 0] = [];
    let len = opus_packet_extensions_generate(&mut packet, &[], 0, true).expect("generate");
    assert_eq!(len, 0);
}

/// Upstream C: tests/test_opus_extensions.c:test_extensions_generate_no_padding
#[test]
fn extensions_generate_no_padding() {
    let exts = vec![
        ext(3, 0, b"a"),
        ext(32, 10, b"DRED"),
        ext(33, 1, b"NOT DRED"),
        ext(4, 4, b""),
    ];
    let mut packet = [0u8; 32];
    let len = opus_packet_extensions_generate(&mut packet, &exts, 11, false).expect("generate");
    assert_eq!(len, 23, "unexpected generated length");
}

/// Upstream C: tests/test_opus_extensions.c:test_extensions_generate_fail
#[test]
fn extensions_generate_fail() {
    let exts = vec![
        ext(3, 0, b"a"),
        ext(32, 10, b"DRED"),
        ext(33, 1, b"NOT DRED"),
        ext(4, 4, b""),
    ];

    for len in 0..23usize {
        let mut packet = [0xFEu8; 100];
        let res = opus_packet_extensions_generate(&mut packet[..len], &exts, 11, true);
        assert_eq!(res, Err(OPUS_BUFFER_TOO_SMALL), "expected buffer-too-small");
        assert!(
            packet[len..].iter().all(|&x| x == 0xFE),
            "tail bytes should remain undisturbed"
        );
    }

    let mut packet = [0u8; 100];
    assert_eq!(
        opus_packet_extensions_generate(&mut packet, &[ext(256, 0, b"a")], 11, true),
        Err(OPUS_BAD_ARG)
    );
    assert_eq!(
        opus_packet_extensions_generate(&mut packet, &[ext(2, 0, b"a")], 11, true),
        Err(OPUS_BAD_ARG)
    );
    assert_eq!(
        opus_packet_extensions_generate(&mut packet, &[ext(33, 11, b"a")], 49, true),
        Err(OPUS_BAD_ARG)
    );
    assert_eq!(
        opus_packet_extensions_generate(&mut packet, &[ext(33, -1, b"a")], 11, true),
        Err(OPUS_BAD_ARG)
    );
    assert_eq!(
        opus_packet_extensions_generate(&mut packet, &[ext(33, 11, b"a")], 11, true),
        Err(OPUS_BAD_ARG)
    );
    assert_eq!(
        opus_packet_extensions_generate(&mut packet, &[ext(3, 0, b"abcd")], 1, true),
        Err(OPUS_BAD_ARG)
    );
}

/// Upstream C: tests/test_opus_extensions.c:test_extensions_parse_success
#[test]
fn extensions_parse_success() {
    let exts = vec![
        ext(3, 0, b"a"),
        ext(32, 10, b"DRED"),
        ext(33, 1, b"NOT DRED"),
        ext(4, 4, b""),
    ];
    let mut packet = [0u8; 32];
    let len = opus_packet_extensions_generate(&mut packet, &exts, 11, true).expect("generate");
    assert_eq!(len, 32);

    let count = opus_packet_extensions_count(&packet[..len], 11).expect("count");
    assert_eq!(count, 4);

    let parsed = opus_packet_extensions_parse(&packet[..len], 10, 11).expect("parse");
    assert_eq!(parsed.len(), 4);
    assert_eq!(parsed[0].id, 3);
    assert_eq!(parsed[0].frame, 0);
    assert_eq!(parsed[0].data, b"a");
    assert_eq!(parsed[1].id, 33);
    assert_eq!(parsed[1].frame, 1);
    assert_eq!(parsed[1].data, b"NOT DRED");
    assert_eq!(parsed[2].id, 4);
    assert_eq!(parsed[2].frame, 4);
    assert_eq!(parsed[2].data, b"");
    assert_eq!(parsed[3].id, 32);
    assert_eq!(parsed[3].frame, 10);
    assert_eq!(parsed[3].data, b"DRED");
}

/// Upstream C: tests/test_opus_extensions.c:test_extensions_parse_zero
#[test]
fn extensions_parse_zero() {
    let exts = vec![ext(32, 1, b"DRED")];
    let mut packet = [0u8; 32];
    let len = opus_packet_extensions_generate(&mut packet, &exts, 2, true).expect("generate");
    assert_eq!(len, 32);

    let parsed = opus_packet_extensions_parse(&packet[..len], 0, 2);
    assert!(
        matches!(parsed, Err(OPUS_BUFFER_TOO_SMALL)),
        "expected OPUS_BUFFER_TOO_SMALL, got {parsed:?}"
    );
}

/// Upstream C: tests/test_opus_extensions.c:test_extensions_parse_fail
#[test]
fn extensions_parse_fail() {
    let exts = vec![
        ext(3, 0, b"a"),
        ext(33, 1, b"NOT DRED"),
        ext(4, 4, b""),
        ext(32, 10, b"DRED"),
        ext(32, 9, b"DRED"),
        ext(4, 9, b"b"),
        ext(4, 10, b"c"),
    ];

    let mut packet = [0u8; 64];

    let mut len =
        opus_packet_extensions_generate(&mut packet, &exts[..4], 11, false).expect("generate");
    packet[4] = 255;
    assert!(matches!(
        opus_packet_extensions_parse(&packet[..len], 10, 11),
        Err(OPUS_INVALID_PACKET)
    ));

    len = opus_packet_extensions_generate(&mut packet, &exts[..4], 11, false).expect("generate");
    assert!(matches!(
        opus_packet_extensions_parse(&packet[..len], 10, 5),
        Err(OPUS_INVALID_PACKET)
    ));

    packet[14] = 255;
    assert!(matches!(
        opus_packet_extensions_parse(&packet[..len], 10, 11),
        Err(OPUS_INVALID_PACKET)
    ));

    len = opus_packet_extensions_generate(&mut packet, &exts[..4], 11, false).expect("generate");
    assert!(matches!(
        opus_packet_extensions_parse(&packet[..len], 1, 11),
        Err(OPUS_BUFFER_TOO_SMALL)
    ));

    len = opus_packet_extensions_generate(&mut packet, &exts, 11, false).expect("generate");
    len -= 5;
    let tail_err = opus_packet_extensions_parse(&packet[..len], 10, 11);
    assert!(
        matches!(
            tail_err,
            Err(OPUS_INVALID_PACKET) | Err(OPUS_BUFFER_TOO_SMALL) | Err(OPUS_BAD_ARG)
        ),
        "expected invalid-packet or buffer-too-small, got {tail_err:?}"
    );
}

/// Upstream C: tests/test_opus_extensions.c:test_extensions_repeating
#[test]
fn extensions_repeating() {
    let exts = vec![
        ext(3, 0, b"a"),
        ext(3, 1, b"b"),
        ext(3, 2, b"c"),
        ext(4, 0, b"d"),
        ext(4, 1, b""),
        ext(4, 2, b""),
        ext(32, 2, b"DRED2"),
        ext(32, 1, b"DRED"),
        ext(5, 1, b""),
        ext(5, 2, b""),
        ext(6, 2, b"f"),
        ext(6, 1, b"e"),
        ext(32, 2, b"DREDthree"),
    ];
    let encoded_len = [0, 2, 5, 5, 7, 9, 10, 16, 21, 23, 22, 26, 25, 37];

    for nb_ext in 0..=exts.len() {
        let expected = &exts[..nb_ext];
        let mut payload = vec![0u8; 96];
        let len =
            opus_packet_extensions_generate(&mut payload, expected, 3, false).expect("generate");
        assert_eq!(len, encoded_len[nb_ext], "encoded length mismatch");
        payload.truncate(len);

        let mut per_frame = [0i32; 48];
        let count =
            opus_packet_extensions_count_ext(&payload, &mut per_frame, 3).expect("count_ext");
        assert_eq!(count, nb_ext as i32, "count mismatch");

        let parsed = opus_packet_extensions_parse_ext(&payload, 13, &per_frame, 3)
            .expect("parse_ext should succeed");
        assert_eq!(parsed.len(), nb_ext, "parsed extension count mismatch");
        check_ext_data(expected, &parsed);

        let mut payload2 = payload.clone();
        let mut changed = false;
        if nb_ext == 6 {
            payload2.push(2 << 1);
            payload2.push(3 << 1);
            changed = true;
        } else if nb_ext == 8 {
            payload2.insert(15, 0x01);
            changed = true;
        } else if nb_ext == 10 {
            payload2.insert(15, 0x03);
            payload2.insert(16, 0x00);
            changed = true;
        } else if nb_ext == 13 && payload2.len() > 26 {
            payload2[26] = 2 << 1;
            changed = true;
        }
        if changed {
            let mut per_frame2 = [0i32; 48];
            let count2 = opus_packet_extensions_count_ext(&payload2, &mut per_frame2, 3)
                .expect("count_ext mutated");
            assert_eq!(count2, nb_ext as i32, "mutated count mismatch");
            let parsed2 = opus_packet_extensions_parse_ext(&payload2, 13, &per_frame2, 3)
                .expect("parse_ext mutated");
            assert_eq!(parsed2.len(), nb_ext, "mutated parsed count mismatch");
            check_ext_data(expected, &parsed2);
        }

        if nb_ext == 8 {
            let mut payload3 = payload2.clone();
            payload3.insert(9, (2 << 1) | 1);
            payload3.insert(5, (2 << 1) | 1);
            let mut per_frame3 = [0i32; 48];
            let count3 = opus_packet_extensions_count_ext(&payload3, &mut per_frame3, 3)
                .expect("count_ext mutated 2");
            assert_eq!(count3, nb_ext as i32, "mutated-2 count mismatch");
            let parsed3 = opus_packet_extensions_parse_ext(&payload3, 13, &per_frame3, 3)
                .expect("parse_ext mutated 2");
            assert_eq!(parsed3.len(), nb_ext, "mutated-2 parsed count mismatch");
            check_ext_data(expected, &parsed3);
        }
    }
}

/// Upstream C: tests/test_opus_extensions.c:test_random_extensions_parse
#[test]
fn random_extensions_parse() {
    const ITERATIONS: usize = 20_000;
    const MAX_EXTENSION_SIZE: usize = 200;
    const MAX_NB_EXTENSIONS: i32 = ((MAX_EXTENSION_SIZE - 1) * 48) as i32;

    let mut seed = 0xC001_C0DEu32;
    for _ in 0..ITERATIONS {
        let len = (next_u32(&mut seed) as usize) % (MAX_EXTENSION_SIZE + 1);
        let mut payload = vec![0u8; len];
        for byte in &mut payload {
            *byte = (next_u32(&mut seed) & 0xFF) as u8;
        }

        let max_ext = (next_u32(&mut seed) % (MAX_NB_EXTENSIONS as u32 + 1)) as i32;
        let nb_frames = (next_u32(&mut seed) % 48 + 1) as i32;

        match opus_packet_extensions_parse(&payload, max_ext, nb_frames) {
            Ok(parsed) => {
                for e in &parsed {
                    assert!(
                        (0..nb_frames).contains(&e.frame),
                        "frame out of range: {}",
                        e.frame
                    );
                    assert!((2..=127).contains(&e.id), "id out of range: {}", e.id);
                }

                let mut out = vec![0u8; MAX_EXTENSION_SIZE + 1];
                let gen_len =
                    match opus_packet_extensions_generate(&mut out, &parsed, nb_frames, false) {
                        Ok(v) => v,
                        Err(_) => continue,
                    };
                out.truncate(gen_len);

                let mut per_frame = [0i32; 48];
                let count = opus_packet_extensions_count_ext(&out, &mut per_frame, nb_frames)
                    .expect("count_ext roundtrip");
                assert_eq!(count as usize, parsed.len(), "roundtrip count mismatch");

                let parsed2 = opus_packet_extensions_parse_ext(
                    &out,
                    MAX_NB_EXTENSIONS,
                    &per_frame,
                    nb_frames,
                )
                .expect("parse_ext roundtrip");
                assert_eq!(parsed2.len(), parsed.len(), "roundtrip parsed len mismatch");
                check_ext_data(&parsed, &parsed2);
            }
            Err(err) => {
                assert!(
                    err == OPUS_BUFFER_TOO_SMALL
                        || err == OPUS_INVALID_PACKET
                        || err == OPUS_BAD_ARG,
                    "unexpected parse error {err}"
                );
            }
        }
    }
}

/// Upstream C: tests/test_opus_extensions.c:test_opus_repacketizer_out_range_impl
#[test]
fn repacketizer_out_range_impl() {
    let p0 = encode_mono_packet(0x1234);
    let p1 = encode_mono_packet(0x2345);
    let p2 = encode_mono_packet(0x3456);

    let exts = [ext(33, 0, b"abcdefg"), ext(100, 0, b"uvwxyz")];

    let mut p0_ext = vec![0u8; p0.len() + 64];
    p0_ext[..p0.len()].copy_from_slice(&p0);
    let p0_ext_cap = p0_ext.len() as i32;
    let p0_len = opus_packet_pad_impl(&mut p0_ext, p0.len() as i32, p0_ext_cap, false, &exts);
    assert!(p0_len > 0, "pad first packet failed: {p0_len}");

    let mut p2_ext = vec![0u8; p2.len() + 64];
    p2_ext[..p2.len()].copy_from_slice(&p2);
    let p2_ext_cap = p2_ext.len() as i32;
    let p2_len = opus_packet_pad_impl(&mut p2_ext, p2.len() as i32, p2_ext_cap, false, &exts);
    assert!(p2_len > 0, "pad third packet failed: {p2_len}");

    let mut rp = OpusRepacketizer::default();
    assert_eq!(rp.cat(&p0_ext[..p0_len as usize]), OPUS_OK);
    assert_eq!(rp.cat(&p1), OPUS_OK);
    assert_eq!(rp.cat(&p2_ext[..p2_len as usize]), OPUS_OK);

    let mut packet_out = vec![0u8; 2048];
    let out_len = rp.out_range(0, 3, &mut packet_out);
    assert!(out_len > 0, "out_range failed: {out_len}");

    let mut sizes = [0i16; 48];
    let mut packet_offset = 0i32;
    let mut padding_len = 0i32;
    let frame_count = opus_packet_parse_impl(
        &packet_out[..out_len as usize],
        false,
        None,
        None,
        &mut sizes,
        None,
        Some(&mut packet_offset),
        Some(&mut padding_len),
    );
    assert_eq!(frame_count, 3, "expected 3 frames");
    assert!(padding_len > 0, "expected packet padding/extensions");

    let start = (packet_offset - padding_len) as usize;
    let end = packet_offset as usize;
    let parsed = opus_packet_extensions_parse(&packet_out[start..end], 10, frame_count)
        .expect("parse repacketized extensions");
    assert_eq!(parsed.len(), 4, "expected 4 extensions");

    let mut id33 = 0;
    let mut id100 = 0;
    for (idx, p) in parsed.iter().enumerate() {
        if p.id == 33 {
            id33 += 1;
            assert_eq!(p.data, b"abcdefg");
        } else if p.id == 100 {
            id100 += 1;
            assert_eq!(p.data, b"uvwxyz");
        } else {
            panic!("unexpected extension id {}", p.id);
        }
        if idx < 2 {
            assert_eq!(p.frame, 0, "first two extensions should be frame 0");
        } else {
            assert_eq!(p.frame, 2, "last two extensions should be frame 2");
        }
    }
    assert_eq!(id33, 2, "expected two id=33 extensions");
    assert_eq!(id100, 2, "expected two id=100 extensions");
}
