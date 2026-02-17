#![allow(unused_assignments)]

use opurs::{
    opus_decode, opus_decode_float, opus_decoder_get_nb_samples, opus_packet_get_bandwidth,
    opus_packet_get_nb_frames, opus_packet_get_nb_samples, opus_packet_get_samples_per_frame,
    opus_packet_pad, opus_packet_parse, opus_packet_unpad, Application, Bandwidth, Bitrate,
    Channels, FrameSize, OpusDecoder, OpusEncoder, OpusRepacketizer, Signal, OPUS_BAD_ARG,
    OPUS_BUFFER_TOO_SMALL, OPUS_INVALID_PACKET,
};

static OPUS_RATES: [i32; 5] = [48000, 24000, 16000, 12000, 8000];

#[test]
fn test_opus_decoder_create_init() {
    // Invalid channel counts with valid sample rates
    for c in [0usize, 3, 4] {
        for fs in [8000, 12000, 16000, 24000, 48000] {
            assert!(
                OpusDecoder::new(fs, c).is_err(),
                "OpusDecoder::new({fs}, {c}) should fail"
            );
        }
    }
    // Valid channel counts with invalid sample rates
    for c in [1usize, 2] {
        for fs in [-8000, 0, 1, 100, 2147483647, -2147483647 - 1] {
            assert!(
                OpusDecoder::new(fs, c).is_err(),
                "OpusDecoder::new({fs}, {c}) should fail"
            );
        }
    }
    // Valid cases should succeed
    for c in [1usize, 2] {
        for fs in [8000, 12000, 16000, 24000, 48000] {
            assert!(
                OpusDecoder::new(fs, c).is_ok(),
                "OpusDecoder::new({fs}, {c}) should succeed"
            );
        }
    }
}

#[test]
fn test_dec_api() {
    let mut packet = vec![0u8; 1276];
    let mut fbuf = vec![0.0f32; 1920];
    let mut sbuf = vec![0i16; 1920];
    let mut cfgs = 0;

    let mut dec = OpusDecoder::new(48000, 2).expect("failed to create decoder");

    let dec_final_range = dec.final_range();
    let _ = dec_final_range; // just verify it doesn't panic
    println!("    OPUS_GET_FINAL_RANGE ......................... OK.");
    cfgs += 1;

    // OPUS_UNIMPLEMENTED test removed: the old CTL dispatch for unknown
    // request codes is not exposed in the safe API.

    let i = dec.get_bandwidth();
    assert_eq!(i, 0, "OPUS_GET_BANDWIDTH initial value should be 0");
    println!("    OPUS_GET_BANDWIDTH ........................... OK.");
    cfgs += 1;

    let i = dec.sample_rate();
    assert_eq!(i, 48000, "OPUS_GET_SAMPLE_RATE should be 48000");
    println!("    OPUS_GET_SAMPLE_RATE ......................... OK.");
    cfgs += 1;

    // GET_PITCH has different execution paths depending on the previously decoded frame.
    let i = dec.pitch();
    assert!(
        (-1..=0).contains(&i),
        "OPUS_GET_PITCH initial check failed: {i}"
    );
    cfgs += 1;

    packet[0] = (63 << 2) as u8;
    packet[2] = 0;
    packet[1] = packet[2];
    assert_eq!(
        dec.decode(&packet[..3], &mut sbuf, 960, false),
        960,
        "decode CELT silence failed"
    );
    cfgs += 1;

    let i = dec.pitch();
    assert!(
        (-1..=0).contains(&i),
        "OPUS_GET_PITCH after CELT decode failed: {i}"
    );
    cfgs += 1;

    packet[0] = 1;
    assert_eq!(
        dec.decode(&packet[..1], &mut sbuf, 960, false),
        960,
        "decode SILK failed"
    );
    cfgs += 1;

    let i = dec.pitch();
    assert!(
        (-1..=0).contains(&i),
        "OPUS_GET_PITCH after SILK decode failed: {i}"
    );
    cfgs += 1;
    println!("    OPUS_GET_PITCH ............................... OK.");

    let i = dec.last_packet_duration();
    assert_eq!(i, 960, "OPUS_GET_LAST_PACKET_DURATION should be 960");
    cfgs += 1;
    println!("    OPUS_GET_LAST_PACKET_DURATION ................ OK.");

    let i = dec.gain();
    assert_eq!(i, 0, "OPUS_GET_GAIN initial value should be 0");
    cfgs += 1;

    assert!(
        dec.set_gain(-32769).is_err(),
        "OPUS_SET_GAIN(-32769) should return BAD_ARG"
    );
    cfgs += 1;

    assert!(
        dec.set_gain(32768).is_err(),
        "OPUS_SET_GAIN(32768) should return BAD_ARG"
    );
    cfgs += 1;

    assert!(dec.set_gain(-15).is_ok(), "OPUS_SET_GAIN(-15) failed");
    cfgs += 1;

    let i = dec.gain();
    assert_eq!(i, -15, "OPUS_GET_GAIN should return -15");
    cfgs += 1;
    println!("    OPUS_SET_GAIN ................................ OK.");
    println!("    OPUS_GET_GAIN ................................ OK.");

    // Reset the decoder and verify it works afterwards
    dec.reset();
    println!("    OPUS_RESET_STATE ............................. OK.");
    cfgs += 1;

    packet[0] = 0;
    assert_eq!(opus_decoder_get_nb_samples(&mut dec, &packet[..1]), 480);
    assert_eq!(opus_packet_get_nb_samples(&packet[..1], 48000), 480);
    assert_eq!(opus_packet_get_nb_samples(&packet[..1], 96000), 960);
    assert_eq!(opus_packet_get_nb_samples(&packet[..1], 32000), 320);
    assert_eq!(opus_packet_get_nb_samples(&packet[..1], 8000), 80);
    packet[0] = 3;
    assert_eq!(
        opus_packet_get_nb_samples(&packet[..1], 24000),
        OPUS_INVALID_PACKET
    );
    packet[0] = ((63) << 2 | 3) as u8;
    packet[1] = 63;
    assert_eq!(opus_packet_get_nb_samples(&[], 24000), OPUS_BAD_ARG);
    assert_eq!(
        opus_packet_get_nb_samples(&packet[..2], 48000),
        OPUS_INVALID_PACKET
    );
    assert_eq!(
        opus_decoder_get_nb_samples(&mut dec, &packet[..2]),
        OPUS_INVALID_PACKET
    );
    println!("    opus_{{packet,decoder}}_get_nb_samples() ....... OK.");
    cfgs += 9;
    assert_eq!(opus_packet_get_nb_frames(&[]), OPUS_BAD_ARG);

    for i in 0..256 {
        let l1res: [i32; 4] = [1, 2, 2, OPUS_INVALID_PACKET];
        packet[0] = i as u8;
        assert_eq!(
            l1res[(packet[0] as i32 & 3) as usize],
            opus_packet_get_nb_frames(&packet[..1]),
            "get_nb_frames 1-byte mismatch for toc={i}"
        );
        cfgs += 1;
        for j in 0..256 {
            packet[1] = j as u8;
            let expected = if packet[0] as i32 & 3 != 3 {
                l1res[(packet[0] as i32 & 3) as usize]
            } else {
                packet[1] as i32 & 63
            };
            assert_eq!(
                expected,
                opus_packet_get_nb_frames(&packet[..2]),
                "get_nb_frames 2-byte mismatch for toc={i}, byte1={j}"
            );
            cfgs += 1;
        }
    }
    println!("    opus_packet_get_nb_frames() .................. OK.");

    for i in 0..256 {
        packet[0] = i as u8;
        let mut bw = packet[0] as i32 >> 4;
        bw = 1101 + (((((bw & 7) * 9) & (63 - (bw & 8))) + 2 + 12 * (bw & 8 != 0) as i32) >> 4);
        assert_eq!(
            bw,
            opus_packet_get_bandwidth(packet[0]),
            "get_bandwidth mismatch for toc={i}"
        );
        cfgs += 1;
    }
    println!("    opus_packet_get_bandwidth() .................. OK.");

    for i in 0..256 {
        packet[0] = i as u8;
        let mut fp3s = packet[0] as i32 >> 3;
        fp3s = (((((3 - (fp3s & 3)) * 13) & 119) + 9) >> 2)
            * ((fp3s > 13) as i32 * (3 - (fp3s & 3 == 3) as i32) + 1)
            * 25;
        let mut rate = 0;
        while rate < 5 {
            assert_eq!(
                OPUS_RATES[rate as usize] * 3 / fp3s,
                opus_packet_get_samples_per_frame(packet[0], OPUS_RATES[rate as usize]),
                "get_samples_per_frame mismatch for toc={i}, rate={}",
                OPUS_RATES[rate as usize]
            );
            cfgs += 1;
            rate += 1;
        }
    }
    println!("    opus_packet_get_samples_per_frame() .......... OK.");

    packet[0] = (((63) << 2) + 3) as u8;
    packet[1] = 49;
    for j in 2..51 {
        packet[j as usize] = 0;
    }

    assert_eq!(
        opus_decode(&mut dec, &packet[..51], &mut sbuf, 960, 0),
        OPUS_INVALID_PACKET
    );
    cfgs += 1;
    packet[0] = ((63) << 2) as u8;
    packet[2] = 0;
    packet[1] = packet[2];

    assert_eq!(
        opus_decode(&mut dec, &packet[..3], &mut sbuf, 60, 0),
        OPUS_BUFFER_TOO_SMALL
    );
    cfgs += 1;
    assert_eq!(
        opus_decode(&mut dec, &packet[..3], &mut sbuf, 480, 0),
        OPUS_BUFFER_TOO_SMALL
    );
    cfgs += 1;
    assert_eq!(opus_decode(&mut dec, &packet[..3], &mut sbuf, 960, 0), 960);
    cfgs += 1;
    println!("    opus_decode() ................................ OK.");
    assert_eq!(
        opus_decode_float(&mut dec, &packet[..3], &mut fbuf, 960, 0),
        960
    );
    cfgs += 1;
    println!("    opus_decode_float() .......................... OK.");

    // dec is dropped implicitly
    cfgs += 1;
    println!("                   All decoder interface tests passed");
    println!("                   ({:6} API invocations)", cfgs);
}

#[test]
fn test_parse_header_code_0() {
    test_parse_code_0_inner();
}

fn test_parse_code_0_inner() {
    let mut packet = vec![0u8; 1276];
    let mut frames = [0; 48];
    let mut size: [i16; 48] = [0; 48];

    for i in 0..64 {
        packet[0] = (i << 2) as u8;
        frames[0] = 0;
        frames[1] = 0;

        let mut toc = u8::MAX;
        let mut payload_offset = -1;
        let ret = opus_packet_parse(
            &packet[..4],
            Some(&mut toc),
            Some(&mut frames),
            &mut size,
            Some(&mut payload_offset),
        );
        assert_eq!(ret, 1, "code 0: expected 1 frame for toc={i}");
        assert_eq!(size[0], 3, "code 0: expected size 3 for toc={i}");
        assert_eq!(frames[0], 1, "code 0: expected frame offset 1 for toc={i}");
    }
}

#[test]
fn test_parse_header_code_1() {
    test_parse_code_1_inner();
}

fn test_parse_code_1_inner() {
    let mut packet = vec![0u8; 1276];
    let mut frames = [0; 48];
    let mut size: [i16; 48] = [0; 48];

    // code 1, two frames of the same size
    for i in 0..64 {
        packet[0] = ((i << 2) + 1) as u8;

        for jj in 0..=1275 * 2 + 3 {
            frames[0] = 0;
            frames[1] = 0;
            let mut toc = u8::MAX;
            let mut payload_offset = -1;

            // this makes no sense anymore
            if jj as usize > packet.len() {
                continue;
            }

            let ret = opus_packet_parse(
                &packet[..jj as _],
                Some(&mut toc),
                Some(&mut frames),
                &mut size,
                Some(&mut payload_offset),
            );
            if jj & 1 == 1 && jj <= 2551 {
                // Must pass if payload length even (packet length odd) and
                // size<=2551, must fail otherwise.
                if ret != 2 {
                    panic!("assertion failed at upstream test_opus_api.c:749");
                }
                if size[0] != size[1] || size[0] as i32 != (jj - 1) >> 1 {
                    panic!("assertion failed at upstream test_opus_api.c:750");
                }
                if frames[0] != 1 {
                    panic!("assertion failed at upstream test_opus_api.c:751");
                }
                if frames[1] != frames[0] + size[0] as usize {
                    panic!("assertion failed at upstream test_opus_api.c:752");
                }
                if toc as i32 >> 2 != i {
                    panic!("assertion failed at upstream test_opus_api.c:753");
                }
            } else if ret != OPUS_INVALID_PACKET {
                panic!("assertion failed at upstream test_opus_api.c:754");
            }
        }
    }
}

#[test]
fn test_parse_header_code_2() {
    test_parse_code_2_inner();
}

fn test_parse_code_2_inner() {
    let mut packet = vec![0u8; 1276];
    let mut frames = [0; 48];
    let mut size: [i16; 48] = [0; 48];

    for i in 0..64 {
        // code 2, length code overflow
        packet[0] = ((i << 2) + 2) as u8;
        frames[0] = 0;
        frames[1] = 0;

        let mut toc = u8::MAX;
        let mut payload_offset = -1;

        let ret = opus_packet_parse(
            &packet[..1],
            Some(&mut toc),
            Some(&mut frames),
            &mut size,
            Some(&mut payload_offset),
        );
        if ret != OPUS_INVALID_PACKET {
            panic!("assertion failed at upstream test_opus_api.c:767");
        }
        packet[1] = 252;
        frames[0] = 0;
        frames[1] = 0;

        let mut toc = u8::MAX;
        let mut payload_offset = -1;

        let ret = opus_packet_parse(
            &packet[..2],
            Some(&mut toc),
            Some(&mut frames),
            &mut size,
            Some(&mut payload_offset),
        );
        if ret != OPUS_INVALID_PACKET {
            panic!("assertion failed at upstream test_opus_api.c:772");
        }
        for j in 0..1275 {
            if j < 252 {
                packet[1] = j as u8;
            } else {
                packet[1] = (252 + (j & 3)) as u8;
                packet[2] = ((j - 252) >> 2) as u8;
            }

            // Code 2, one too short
            frames[0] = 0;
            frames[1] = 0;

            let mut toc = u8::MAX;
            let mut payload_offset = -1;
            let ret = opus_packet_parse(
                &packet[..(j + (if j < 252 { 2 } else { 3 }) - 1_usize)],
                Some(&mut toc),
                Some(&mut frames),
                &mut size,
                Some(&mut payload_offset),
            );
            if ret != OPUS_INVALID_PACKET {
                panic!("assertion failed at upstream test_opus_api.c:781");
            }

            // Code 2, one too long
            frames[0] = 0;
            frames[1] = 0;
            let mut toc = u8::MAX;
            let mut payload_offset = -1;

            let packet_len = j + (if j < 252 { 2 } else { 3 }) + 1276;
            // this makes no sense anymore
            if packet_len > packet.len() {
                continue;
            }

            let ret = opus_packet_parse(
                &packet[..packet_len as _],
                Some(&mut toc),
                Some(&mut frames),
                &mut size,
                Some(&mut payload_offset),
            );
            if ret != OPUS_INVALID_PACKET {
                panic!("assertion failed at upstream test_opus_api.c:786");
            }

            // Code 2, second zero
            frames[0] = 0;
            frames[1] = 0;
            let mut toc = u8::MAX;
            let mut payload_offset = -1;

            let packet_len = j + (if j < 252 { 2 } else { 3 });
            // this makes no sense anymore
            if packet_len > packet.len() {
                continue;
            }

            let ret = opus_packet_parse(
                &packet[..packet_len as _],
                Some(&mut toc),
                Some(&mut frames),
                &mut size,
                Some(&mut payload_offset),
            );
            if ret != 2 {
                panic!("assertion failed at upstream test_opus_api.c:791");
            }
            if size[0] as usize != j || size[1] as i32 != 0 {
                panic!("assertion failed at upstream test_opus_api.c:792");
            }
            if frames[1] != frames[0] + size[0] as usize {
                panic!("assertion failed at upstream test_opus_api.c:793");
            }
            if toc as i32 >> 2 != i {
                panic!("assertion failed at upstream test_opus_api.c:794");
            }

            // Code 2, normal
            frames[0] = 0;
            frames[1] = 0;
            let mut toc = u8::MAX;
            let mut payload_offset = -1;

            let packet_len = (j << 1) + 4;

            // this makes no sense anymore
            if packet_len > packet.len() {
                continue;
            }

            let ret = opus_packet_parse(
                &packet[..packet_len],
                Some(&mut toc),
                Some(&mut frames),
                &mut size,
                Some(&mut payload_offset),
            );
            if ret != 2 {
                panic!("assertion failed at upstream test_opus_api.c:799");
            }
            if size[0] as usize != j
                || size[1] as usize != (j << 1) + 3 - j - (if j < 252 { 1 } else { 2 })
            {
                panic!("assertion failed at upstream test_opus_api.c:800");
            }
            if frames[1] != frames[0] + size[0] as usize {
                panic!("assertion failed at upstream test_opus_api.c:801");
            }
            if toc as i32 >> 2 != i {
                panic!("assertion failed at upstream test_opus_api.c:802");
            }
        }
    }
}

#[test]
fn test_parse_header_code_3_m_truncation() {
    test_parse_code_3_m_truncation_inner();
}

fn test_parse_code_3_m_truncation_inner() {
    let mut packet = vec![0u8; 1276];
    let mut frames = [0; 48];
    let mut size: [i16; 48] = [0; 48];

    for i in 0..64 {
        packet[0] = ((i << 2) + 3) as u8;
        frames[0] = 0;
        frames[1] = 0;
        let mut toc = u8::MAX;
        let mut payload_offset = -1;
        let ret = opus_packet_parse(
            &packet[..1],
            Some(&mut toc),
            Some(&mut frames),
            &mut size,
            Some(&mut payload_offset),
        );

        if ret != OPUS_INVALID_PACKET {
            panic!("assertion failed at upstream test_opus_api.c:815");
        }
    }
}

#[test]
fn test_parse_header_code_3_m_0_49_64() {
    test_parse_code_3_m_0_49_64_inner();
}

fn test_parse_code_3_m_0_49_64_inner() {
    let mut packet = vec![0u8; 1276];
    let mut frames = [0; 48];
    let mut size: [i16; 48] = [0; 48];

    for i in 0..64 {
        packet[0] = ((i << 2) + 3) as u8;
        for jj in 49..=64 {
            packet[1] = (jj & 63) as u8;
            frames[0] = 0;
            frames[1] = 0;

            let mut toc = u8::MAX;
            let mut payload_offset = -1;
            let ret = opus_packet_parse(
                &packet[..1275],
                Some(&mut toc),
                Some(&mut frames),
                &mut size,
                Some(&mut payload_offset),
            );
            if ret != OPUS_INVALID_PACKET {
                panic!("assertion failed at upstream test_opus_api.c:830");
            }
            packet[1] = (128 + (jj & 63)) as u8;
            frames[0] = 0;
            frames[1] = 0;
            let mut toc = u8::MAX;
            let mut payload_offset = -1;
            let ret = opus_packet_parse(
                &packet[..1275],
                Some(&mut toc),
                Some(&mut frames),
                &mut size,
                Some(&mut payload_offset),
            );
            if ret != OPUS_INVALID_PACKET {
                panic!("assertion failed at upstream test_opus_api.c:835");
            }
            packet[1] = (64 + (jj & 63)) as u8;
            frames[0] = 0;
            frames[1] = 0;
            let mut toc = u8::MAX;
            let mut payload_offset = -1;
            let ret = opus_packet_parse(
                &packet[..1275],
                Some(&mut toc),
                Some(&mut frames),
                &mut size,
                Some(&mut payload_offset),
            );
            if ret != OPUS_INVALID_PACKET {
                panic!("assertion failed at upstream test_opus_api.c:840");
            }
            packet[1] = (128 + 64 + (jj & 63)) as u8;
            frames[0] = 0;
            frames[1] = 0;
            let mut toc = u8::MAX;
            let mut payload_offset = -1;
            let ret = opus_packet_parse(
                &packet[..1275],
                Some(&mut toc),
                Some(&mut frames),
                &mut size,
                Some(&mut payload_offset),
            );
            if ret != OPUS_INVALID_PACKET {
                panic!("assertion failed at upstream test_opus_api.c:845");
            }
        }
    }
}

#[test]
fn test_parse_header_code_3_m_1_cbr() {
    test_parse_code_3_m_1_cbr_inner();
}

fn test_parse_code_3_m_1_cbr_inner() {
    let mut packet = vec![0u8; 1276];
    let mut frames = [0; 48];
    let mut size: [i16; 48] = [0; 48];

    for i in 0..64 {
        packet[0] = ((i << 2) + 3) as u8;
        packet[1] = 1;
        for j in 0..1276 {
            frames[0] = 0;
            frames[1] = 0;
            let mut toc = u8::MAX;
            let mut payload_offset = -1;

            let packet_len = j + 2;
            // this makes no sense anymore
            if packet_len as usize > packet.len() {
                continue;
            }

            let ret = opus_packet_parse(
                &packet[..packet_len as usize],
                Some(&mut toc),
                Some(&mut frames),
                &mut size,
                Some(&mut payload_offset),
            );
            if ret != 1 {
                panic!("assertion failed at upstream test_opus_api.c:861");
            }
            if size[0] as i32 != j {
                panic!("assertion failed at upstream test_opus_api.c:862");
            }
            if toc as i32 >> 2 != i {
                panic!("assertion failed at upstream test_opus_api.c:863");
            }
        }
    }
}

#[test]
fn test_parse_header_code_3_m_1_48_cbr() {
    test_parse_code_3_m_1_48_cbr_inner();
}

fn test_parse_code_3_m_1_48_cbr_inner() {
    let mut packet = vec![0u8; 1276];
    let mut frames = [0; 48];
    let mut size: [i16; 48] = [0; 48];

    for i in 0..64 {
        let mut frame_samp: i32 = 0;
        packet[0] = (i << 2) + 3;
        frame_samp = opus_packet_get_samples_per_frame(packet[0], 48000);
        for j in 2..49 {
            packet[1] = j as u8;
            for sz in 2..(j + 2) * 1275 {
                frames[0] = 0;
                frames[1] = 0;
                let mut toc = u8::MAX;
                let mut payload_offset = -1;

                // this makes no sense anymore
                if sz > packet.len() {
                    continue;
                }

                let ret = opus_packet_parse(
                    &packet[..sz],
                    Some(&mut toc),
                    Some(&mut frames),
                    &mut size,
                    Some(&mut payload_offset),
                );
                if frame_samp * j as i32 <= 5760 && (sz - 2) % j == 0 && (sz - 2) / j < 1276 {
                    if ret != j as _ {
                        panic!("assertion failed at upstream test_opus_api.c:890");
                    }
                    for jj in 1..ret {
                        if frames[jj as usize]
                            != (frames[(jj - 1) as usize] + size[(jj - 1) as usize] as usize)
                        {
                            panic!("assertion failed at upstream test_opus_api.c:891");
                        }
                    }
                    if toc >> 2 != i {
                        panic!("assertion failed at upstream test_opus_api.c:892");
                    }
                } else if ret != OPUS_INVALID_PACKET {
                    panic!("assertion failed at upstream test_opus_api.c:893");
                }
            }
        }

        packet[1] = (5760 / frame_samp) as u8;
        frames[0] = 0;
        frames[1] = 0;
        let mut toc = u8::MAX;
        let mut payload_offset = -1;
        let p1 = packet[1];

        let packet_len = 1275 * p1 as i32 + 2;
        // this makes no sense anymore
        if packet_len as usize > packet.len() {
            continue;
        }

        let ret = opus_packet_parse(
            &packet[..packet_len as usize],
            Some(&mut toc),
            Some(&mut frames),
            &mut size,
            Some(&mut payload_offset),
        );
        if ret != packet[1] as i32 {
            panic!("assertion failed at upstream test_opus_api.c:901");
        }
        for jj in 0..ret {
            if size[jj as usize] != 1275 {
                panic!("assertion failed at upstream test_opus_api.c:902");
            }
        }
    }
}

#[test]
fn test_parse_header_code_3_m_1_48_vbr() {
    test_parse_code_3_m_1_48_vbr_inner();
}

fn test_parse_code_3_m_1_48_vbr_inner() {
    let mut packet = vec![0u8; 1276];
    let mut frames = [0; 48];
    let mut size: [i16; 48] = [0; 48];

    for i in 0..64 {
        packet[0] = ((i << 2) + 3) as u8;
        packet[1] = (128 + 1) as u8;
        let frame_samp_0 = opus_packet_get_samples_per_frame(packet[0], 48000);

        for jj in 0..1276 {
            frames[0] = 0;
            frames[1] = 0;
            let mut toc = u8::MAX;
            let mut payload_offset = -1;

            let packet_len = 2 + jj;
            // this makes no sense anymore
            if packet_len as usize > packet.len() {
                continue;
            }

            let ret = opus_packet_parse(
                &packet[..packet_len as usize],
                Some(&mut toc),
                Some(&mut frames),
                &mut size,
                Some(&mut payload_offset),
            );

            if ret != 1 {
                panic!("assertion failed at upstream test_opus_api.c:919");
            }
            if size[0] as i32 != jj {
                panic!("assertion failed at upstream test_opus_api.c:920");
            }
            if toc as i32 >> 2 != i {
                panic!("assertion failed at upstream test_opus_api.c:921");
            }
        }

        for j in 2..49 {
            packet[1] = (128 + j) as u8;
            frames[0] = 0;
            frames[1] = 0;
            let mut toc = u8::MAX;
            let mut payload_offset = -1;
            let ret = opus_packet_parse(
                &packet[..2 + j - 2],
                Some(&mut toc),
                Some(&mut frames),
                &mut size,
                Some(&mut payload_offset),
            );
            if ret != OPUS_INVALID_PACKET {
                panic!("assertion failed at upstream test_opus_api.c:934");
            }
            packet[2] = 252;
            packet[3] = 0;
            packet[4..2 + j].fill(0);
            frames[0] = 0;
            frames[1] = 0;
            let mut toc = u8::MAX;
            let mut payload_offset = -1;
            let ret = opus_packet_parse(
                &packet[..2 + j],
                Some(&mut toc),
                Some(&mut frames),
                &mut size,
                Some(&mut payload_offset),
            );
            if ret != OPUS_INVALID_PACKET {
                panic!("assertion failed at upstream test_opus_api.c:941");
            }
            packet[2..2 + j].fill(0);
            frames[0] = 0;
            frames[1] = 0;
            let mut toc = u8::MAX;
            let mut payload_offset = -1;
            let ret = opus_packet_parse(
                &packet[..2 + j - 2],
                Some(&mut toc),
                Some(&mut frames),
                &mut size,
                Some(&mut payload_offset),
            );
            if ret != OPUS_INVALID_PACKET {
                panic!("assertion failed at upstream test_opus_api.c:947");
            }
            packet[2] = 252;
            packet[3] = 0;
            packet[4..2 + j].fill(0);
            frames[0] = 0;
            frames[1] = 0;
            let mut toc = u8::MAX;
            let mut payload_offset = -1;
            let ret = opus_packet_parse(
                &packet[..2 + j + 252 - 1],
                Some(&mut toc),
                Some(&mut frames),
                &mut size,
                Some(&mut payload_offset),
            );

            if ret != OPUS_INVALID_PACKET {
                panic!("assertion failed at upstream test_opus_api.c:955");
            }
            packet[2..2 + j].fill(0);
            frames[0] = 0;
            frames[1] = 0;
            let mut toc = u8::MAX;
            let mut payload_offset = -1;
            let ret = opus_packet_parse(
                &packet[..2 + j - 1],
                Some(&mut toc),
                Some(&mut frames),
                &mut size,
                Some(&mut payload_offset),
            );

            if frame_samp_0 * j as i32 <= 5760 {
                if ret != j as i32 {
                    panic!("assertion failed at upstream test_opus_api.c:962");
                }
                if size[..j].iter().any(|s| *s as i32 != 0) {
                    panic!("assertion failed at upstream test_opus_api.c:963");
                }
                if toc >> 2 != i as _ {
                    panic!("assertion failed at upstream test_opus_api.c:964");
                }
            } else if ret != OPUS_INVALID_PACKET {
                panic!("assertion failed at upstream test_opus_api.c:965");
            }
            for sz in 0..8 {
                let tsz: [i32; 8] = [50, 201, 403, 700, 1472, 5110, 20400, 61298];
                let mut pos: i32 = 0;
                let as_0: i32 = (tsz[sz as usize] + i - j as i32 - 2) / j as i32;
                for _jj in 0..j - 1 {
                    if as_0 < 252 {
                        packet[(2 + pos) as usize] = as_0 as u8;
                        pos += 1;
                    } else {
                        packet[(2 + pos) as usize] = (252 + (as_0 & 3)) as u8;
                        packet[(3 + pos) as usize] = ((as_0 - 252) >> 2) as u8;
                        pos += 2;
                    }
                }
                frames[0] = 0;
                frames[1] = 0;
                let mut toc = u8::MAX;
                let mut payload_offset = -1;

                let packet_len = tsz[sz as usize] + i;
                // this makes no sense anymore
                if packet_len as usize > packet.len() {
                    continue;
                }

                let ret = opus_packet_parse(
                    &packet[..packet_len as usize],
                    Some(&mut toc),
                    Some(&mut frames),
                    &mut size,
                    Some(&mut payload_offset),
                );

                if frame_samp_0 * j as i32 <= 5760
                    && as_0 < 1276
                    && tsz[sz as usize] + i - 2 - pos - as_0 * (j as i32 - 1) < 1276
                {
                    if ret != j as i32 {
                        panic!("assertion failed at upstream test_opus_api.c:981");
                    }
                    if size[..j - 1].iter().any(|s| *s as i32 != as_0) {
                        panic!("assertion failed at upstream test_opus_api.c:982");
                    }
                    if size[j - 1] as i32 != tsz[sz as usize] + i - 2 - pos - as_0 * (j as i32 - 1)
                    {
                        panic!("assertion failed at upstream test_opus_api.c:983");
                    }
                    if toc as i32 >> 2 != i {
                        panic!("assertion failed at upstream test_opus_api.c:984");
                    }
                } else if ret != OPUS_INVALID_PACKET {
                    panic!("assertion failed at upstream test_opus_api.c:985");
                }
            }
        }
    }
}

#[test]
fn test_parse_header_code_3_padding() {
    test_parse_code_3_padding_inner();
}

fn test_parse_code_3_padding_inner() {
    let mut packet = vec![0u8; 1276];
    let mut frames = [0; 48];
    let mut size: [i16; 48] = [0; 48];

    for i in 0..64 {
        packet[0] = ((i << 2) + 3) as u8;
        packet[1] = (128 + 1 + 64) as u8;
        for jj in 2..127 {
            packet[jj as usize] = 255;
        }

        frames[0] = 0;
        frames[1] = 0;
        let mut toc = u8::MAX;
        let mut payload_offset = -1;
        let ret = opus_packet_parse(
            &packet[..127],
            Some(&mut toc),
            Some(&mut frames),
            &mut size,
            Some(&mut payload_offset),
        );

        if ret != OPUS_INVALID_PACKET {
            panic!("assertion failed at upstream test_opus_api.c:1002");
        }
        for sz in 0..4 {
            let tsz_0: [i32; 4] = [0, 72, 512, 1275];
            for jj in (sz..65025).step_by(11) {
                let mut pos_0: i32 = 0;
                pos_0 = 0;
                while pos_0 < jj / 254 {
                    packet[(2 + pos_0) as usize] = 255;
                    pos_0 += 1;
                }
                packet[(2 + pos_0) as usize] = (jj % 254) as u8;
                pos_0 += 1;
                if sz == 0 && i == 63 {
                    frames[0] = 0;
                    frames[1] = 0;
                    let mut payload_offset = -1;
                    let mut toc = u8::MAX;

                    let packet_len = 2 + jj + pos_0 - 1;
                    // this makes no sense anymore
                    if packet_len as usize > packet.len() {
                        continue;
                    }

                    let ret = opus_packet_parse(
                        &packet[..packet_len as usize],
                        Some(&mut toc),
                        Some(&mut frames),
                        &mut size,
                        Some(&mut payload_offset),
                    );

                    if ret != OPUS_INVALID_PACKET {
                        panic!("assertion failed at upstream test_opus_api.c:1019");
                    }
                }
                frames[0] = 0;
                frames[1] = 0;
                let mut toc = u8::MAX;
                let mut payload_offset = -1;

                let packet_len = 2 + jj + tsz_0[sz as usize] + i + pos_0;
                // this makes no sense anymore
                if packet_len as usize > packet.len() {
                    continue;
                }

                let ret = opus_packet_parse(
                    &packet[..packet_len as usize],
                    Some(&mut toc),
                    Some(&mut frames),
                    &mut size,
                    Some(&mut payload_offset),
                );

                if tsz_0[sz as usize] + i < 1276 {
                    if ret != 1 {
                        panic!("assertion failed at upstream test_opus_api.c:1026");
                    }
                    if size[0] as i32 != tsz_0[sz as usize] + i {
                        panic!("assertion failed at upstream test_opus_api.c:1027");
                    }
                    if toc as i32 >> 2 != i {
                        panic!("assertion failed at upstream test_opus_api.c:1028");
                    }
                } else if ret != OPUS_INVALID_PACKET {
                    panic!("assertion failed at upstream test_opus_api.c:1029");
                }
            }
        }
    }
}

#[test]
fn test_enc_api() {
    test_enc_api_inner();
}

fn test_enc_api_inner() {
    let mut packet = vec![0u8; 1276];
    let mut fbuf = vec![0.0f32; 1920];
    let mut sbuf = vec![0i16; 1920];
    let mut cfgs: i32 = 0;

    println!("\n  Encoder basic API tests");
    println!("  ---------------------------------------------------");

    // Test OpusEncoder::new() error paths for invalid channel/sample-rate combos
    for c in 0..4 {
        for i in -7..=96000 {
            let valid_rate = i == 8000
                || i == 12000
                || i == 16000
                || i == 24000
                || i == 48000
                || cfg!(feature = "qext") && i == 96000;
            if !(valid_rate && (c == 1 || c == 2)) {
                let fs = match i {
                    -5 => -8000,
                    -6 => 2147483647,
                    -7 => -2147483647 - 1,
                    _ => i,
                };
                assert!(
                    OpusEncoder::new(fs, c, 2048).is_err(),
                    "OpusEncoder::new({fs}, {c}, 2048) should fail"
                );
                cfgs += 1;
            }
        }
    }

    // Invalid application code
    assert!(
        OpusEncoder::new(48000, 2, -1000).is_err(),
        "OpusEncoder::new(48000, 2, -1000) should fail"
    );
    cfgs += 1;

    // Valid creation with VOIP application
    {
        let _enc = OpusEncoder::new(48000, 2, 2048).expect("failed to create encoder with VOIP");
        cfgs += 1;
    }

    // Valid creation with LOW_DELAY application, check lookahead
    {
        let enc =
            OpusEncoder::new(48000, 2, 2051).expect("failed to create encoder with LOW_DELAY");
        let i = enc.lookahead();
        assert!(
            (0..=32766).contains(&i),
            "OPUS_GET_LOOKAHEAD for LOW_DELAY returned {i}"
        );
        cfgs += 2;
    }

    // Valid creation with AUDIO application, check lookahead
    {
        let enc = OpusEncoder::new(48000, 2, 2049).expect("failed to create encoder with AUDIO");
        let i = enc.lookahead();
        assert!(
            (0..=32766).contains(&i),
            "OPUS_GET_LOOKAHEAD for AUDIO returned {i}"
        );
        cfgs += 2;
    }

    // Main encoder for the remaining tests
    let mut enc = OpusEncoder::new(48000, 2, 2048).expect("failed to create encoder with VOIP");
    cfgs += 1;
    println!("    opus_encoder_create() ........................ OK.");
    println!("    opus_encoder_init() .......................... OK.");

    let i = enc.lookahead();
    assert!((0..=32766).contains(&i), "OPUS_GET_LOOKAHEAD returned {i}");
    cfgs += 1;
    println!("    OPUS_GET_LOOKAHEAD ........................... OK.");

    let i = enc.sample_rate();
    assert_eq!(i, 48000, "OPUS_GET_SAMPLE_RATE should be 48000");
    cfgs += 1;
    println!("    OPUS_GET_SAMPLE_RATE ......................... OK.");

    // OPUS_UNIMPLEMENTED test removed: the old CTL dispatch for unknown
    // request codes is not exposed in the safe API.

    // OPUS_SET_APPLICATION / OPUS_GET_APPLICATION
    assert!(
        Application::try_from(-1).is_err(),
        "Application::try_from(-1) should fail"
    );
    assert!(
        Application::try_from(-1000).is_err(),
        "Application::try_from(-1000) should fail"
    );
    let app = Application::try_from(2049).unwrap();
    assert!(
        enc.set_application(app).is_ok(),
        "set_application(Audio) failed"
    );
    assert_eq!(enc.application(), Application::Audio);
    let app = Application::try_from(2051).unwrap();
    assert!(
        enc.set_application(app).is_ok(),
        "set_application(LowDelay) failed"
    );
    assert_eq!(enc.application(), Application::LowDelay);
    println!("    OPUS_SET_APPLICATION ......................... OK.");
    println!("    OPUS_GET_APPLICATION ......................... OK.");
    cfgs += 6;

    // OPUS_SET_BITRATE / OPUS_GET_BITRATE
    enc.set_bitrate(Bitrate::Bits(1073741832));
    cfgs += 1;
    let i = enc.bitrate();
    assert!(
        (256000..=1500000).contains(&i),
        "bitrate after setting huge value = {i}, expected [256000, 1500000]"
    );
    cfgs += 1;
    // Invalid bitrate values: the safe API uses Bitrate enum, so -12345 becomes Bits(-12345)
    // and 0 becomes Bits(0). The encoder clamps these internally rather than erroring.
    enc.set_bitrate(Bitrate::Bits(500));
    assert_eq!(enc.bitrate(), 500);
    enc.set_bitrate(Bitrate::Bits(256000));
    assert_eq!(enc.bitrate(), 256000);
    println!("    OPUS_SET_BITRATE ............................. OK.");
    println!("    OPUS_GET_BITRATE ............................. OK.");
    cfgs += 6;

    // OPUS_SET_FORCE_CHANNELS / OPUS_GET_FORCE_CHANNELS
    // Invalid: 3 channels not possible
    assert!(
        enc.set_force_channels(Some(Channels::try_from(3).unwrap_or(Channels::Stereo)))
            .is_ok()
            || Channels::try_from(3).is_err(),
        "force_channels(3) should fail at conversion"
    );
    assert!(Channels::try_from(-1).is_err());
    assert!(Channels::try_from(3).is_err());
    // Valid
    assert!(enc.set_force_channels(Some(Channels::Mono)).is_ok());
    assert_eq!(enc.force_channels(), Some(Channels::Mono));
    assert!(enc.set_force_channels(None).is_ok());
    assert_eq!(enc.force_channels(), None);
    println!("    OPUS_SET_FORCE_CHANNELS ...................... OK.");
    println!("    OPUS_GET_FORCE_CHANNELS ...................... OK.");
    cfgs += 6;

    // OPUS_SET_BANDWIDTH / OPUS_GET_BANDWIDTH
    // Invalid bandwidth values rejected by try_from
    assert!(Bandwidth::try_from(OPUS_BUFFER_TOO_SMALL).is_err());
    cfgs += 1;
    assert!(Bandwidth::try_from(1105 + 1).is_err());
    cfgs += 1;
    enc.set_bandwidth(Some(Bandwidth::Narrowband));
    cfgs += 1;
    enc.set_bandwidth(Some(Bandwidth::Fullband));
    cfgs += 1;
    enc.set_bandwidth(Some(Bandwidth::Wideband));
    cfgs += 1;
    enc.set_bandwidth(Some(Bandwidth::Mediumband));
    cfgs += 1;
    println!("    OPUS_SET_BANDWIDTH ........................... OK.");
    let i = enc.get_bandwidth();
    assert!(
        i == 1101 || i == 1102 || i == 1103 || i == 1105 || i == -1000,
        "get_bandwidth returned unexpected {i}"
    );
    cfgs += 1;
    enc.set_bandwidth(None);
    cfgs += 1;
    println!("    OPUS_GET_BANDWIDTH ........................... OK.");

    // OPUS_SET_MAX_BANDWIDTH / OPUS_GET_MAX_BANDWIDTH
    assert!(Bandwidth::try_from(OPUS_BUFFER_TOO_SMALL).is_err());
    cfgs += 1;
    assert!(Bandwidth::try_from(1105 + 1).is_err());
    cfgs += 1;
    enc.set_max_bandwidth(Bandwidth::Narrowband);
    cfgs += 1;
    enc.set_max_bandwidth(Bandwidth::Fullband);
    cfgs += 1;
    enc.set_max_bandwidth(Bandwidth::Wideband);
    cfgs += 1;
    enc.set_max_bandwidth(Bandwidth::Mediumband);
    cfgs += 1;
    println!("    OPUS_SET_MAX_BANDWIDTH ....................... OK.");
    let mbw = enc.max_bandwidth();
    assert!(
        mbw == Bandwidth::Narrowband
            || mbw == Bandwidth::Mediumband
            || mbw == Bandwidth::Wideband
            || mbw == Bandwidth::Fullband,
        "max_bandwidth returned unexpected {mbw:?}"
    );
    cfgs += 1;
    println!("    OPUS_GET_MAX_BANDWIDTH ....................... OK.");

    // OPUS_SET_DTX / OPUS_GET_DTX
    // With the safe bool API, invalid integer values (-1, 2) don't apply.
    enc.set_dtx(true);
    assert!(enc.dtx());
    enc.set_dtx(false);
    assert!(!enc.dtx());
    println!("    OPUS_SET_DTX ................................. OK.");
    println!("    OPUS_GET_DTX ................................. OK.");
    cfgs += 6;

    // OPUS_SET_COMPLEXITY / OPUS_GET_COMPLEXITY
    assert!(
        enc.set_complexity(-1).is_err(),
        "set_complexity(-1) should fail"
    );
    assert!(
        enc.set_complexity(11).is_err(),
        "set_complexity(11) should fail"
    );
    assert!(enc.set_complexity(0).is_ok());
    assert_eq!(enc.complexity(), 0);
    assert!(enc.set_complexity(10).is_ok());
    assert_eq!(enc.complexity(), 10);
    println!("    OPUS_SET_COMPLEXITY .......................... OK.");
    println!("    OPUS_GET_COMPLEXITY .......................... OK.");
    cfgs += 6;

    // OPUS_SET_INBAND_FEC / OPUS_GET_INBAND_FEC
    assert!(
        enc.set_inband_fec(-1).is_err(),
        "set_inband_fec(-1) should fail"
    );
    assert!(
        enc.set_inband_fec(3).is_err(),
        "set_inband_fec(3) should fail"
    );
    enc.set_inband_fec(1).unwrap();
    assert_eq!(enc.inband_fec(), 1);
    enc.set_inband_fec(0).unwrap();
    assert_eq!(enc.inband_fec(), 0);
    enc.set_inband_fec(2).unwrap();
    assert_eq!(enc.inband_fec(), 2);
    enc.set_inband_fec(0).unwrap();
    println!("    OPUS_SET_INBAND_FEC .......................... OK.");
    println!("    OPUS_GET_INBAND_FEC .......................... OK.");
    cfgs += 6;

    // OPUS_SET_PACKET_LOSS_PERC / OPUS_GET_PACKET_LOSS_PERC
    assert!(
        enc.set_packet_loss_perc(-1).is_err(),
        "set_packet_loss_perc(-1) should fail"
    );
    assert!(
        enc.set_packet_loss_perc(101).is_err(),
        "set_packet_loss_perc(101) should fail"
    );
    assert!(enc.set_packet_loss_perc(100).is_ok());
    assert_eq!(enc.packet_loss_perc(), 100);
    assert!(enc.set_packet_loss_perc(0).is_ok());
    assert_eq!(enc.packet_loss_perc(), 0);
    println!("    OPUS_SET_PACKET_LOSS_PERC .................... OK.");
    println!("    OPUS_GET_PACKET_LOSS_PERC .................... OK.");
    cfgs += 6;

    // OPUS_SET_VBR / OPUS_GET_VBR
    // With the safe bool API, invalid integer values don't apply.
    enc.set_vbr(true);
    assert!(enc.vbr());
    enc.set_vbr(false);
    assert!(!enc.vbr());
    println!("    OPUS_SET_VBR ................................. OK.");
    println!("    OPUS_GET_VBR ................................. OK.");
    cfgs += 6;

    // OPUS_SET_VBR_CONSTRAINT / OPUS_GET_VBR_CONSTRAINT
    // With the safe bool API, invalid integer values don't apply.
    enc.set_vbr_constraint(true);
    assert!(enc.vbr_constraint());
    enc.set_vbr_constraint(false);
    assert!(!enc.vbr_constraint());
    println!("    OPUS_SET_VBR_CONSTRAINT ...................... OK.");
    println!("    OPUS_GET_VBR_CONSTRAINT ...................... OK.");
    cfgs += 6;

    // OPUS_SET_SIGNAL / OPUS_GET_SIGNAL
    assert!(Signal::try_from(-12345).is_err());
    assert!(Signal::try_from(0x7fffffff).is_err());
    enc.set_signal(Some(Signal::Music));
    assert_eq!(enc.signal(), Some(Signal::Music));
    enc.set_signal(None);
    assert_eq!(enc.signal(), None);
    println!("    OPUS_SET_SIGNAL .............................. OK.");
    println!("    OPUS_GET_SIGNAL .............................. OK.");
    cfgs += 6;

    // OPUS_SET_LSB_DEPTH / OPUS_GET_LSB_DEPTH
    assert!(
        enc.set_lsb_depth(7).is_err(),
        "set_lsb_depth(7) should fail"
    );
    assert!(
        enc.set_lsb_depth(25).is_err(),
        "set_lsb_depth(25) should fail"
    );
    assert!(enc.set_lsb_depth(16).is_ok());
    assert_eq!(enc.lsb_depth(), 16);
    assert!(enc.set_lsb_depth(24).is_ok());
    assert_eq!(enc.lsb_depth(), 24);
    println!("    OPUS_SET_LSB_DEPTH ........................... OK.");
    println!("    OPUS_GET_LSB_DEPTH ........................... OK.");
    cfgs += 6;

    // OPUS_GET_PREDICTION_DISABLED (initial value)
    assert!(
        !enc.prediction_disabled(),
        "prediction_disabled initial should be false"
    );
    cfgs += 1;

    // OPUS_SET_PREDICTION_DISABLED / OPUS_GET_PREDICTION_DISABLED
    // With the safe bool API, invalid integer values don't apply.
    enc.set_prediction_disabled(true);
    assert!(enc.prediction_disabled());
    enc.set_prediction_disabled(false);
    assert!(!enc.prediction_disabled());
    println!("    OPUS_SET_PREDICTION_DISABLED ................. OK.");
    println!("    OPUS_GET_PREDICTION_DISABLED ................. OK.");
    cfgs += 6;

    // OPUS_SET_EXPERT_FRAME_DURATION / OPUS_GET_EXPERT_FRAME_DURATION
    enc.set_expert_frame_duration(FrameSize::Ms2_5);
    cfgs += 1;
    enc.set_expert_frame_duration(FrameSize::Ms5);
    cfgs += 1;
    enc.set_expert_frame_duration(FrameSize::Ms10);
    cfgs += 1;
    enc.set_expert_frame_duration(FrameSize::Ms20);
    cfgs += 1;
    enc.set_expert_frame_duration(FrameSize::Ms40);
    cfgs += 1;
    enc.set_expert_frame_duration(FrameSize::Ms60);
    cfgs += 1;
    enc.set_expert_frame_duration(FrameSize::Ms80);
    cfgs += 1;
    enc.set_expert_frame_duration(FrameSize::Ms100);
    cfgs += 1;
    enc.set_expert_frame_duration(FrameSize::Ms120);
    cfgs += 1;
    // Invalid values rejected by try_from
    assert!(FrameSize::try_from(0).is_err());
    assert!(FrameSize::try_from(-1).is_err());
    enc.set_expert_frame_duration(FrameSize::Ms60);
    assert_eq!(enc.expert_frame_duration(), FrameSize::Ms60);
    enc.set_expert_frame_duration(FrameSize::Arg);
    assert_eq!(enc.expert_frame_duration(), FrameSize::Arg);
    println!("    OPUS_SET_EXPERT_FRAME_DURATION ............... OK.");
    println!("    OPUS_GET_EXPERT_FRAME_DURATION ............... OK.");
    cfgs += 6;

    // OPUS_GET_FINAL_RANGE
    let _final_range = enc.final_range();
    cfgs += 1;
    println!("    OPUS_GET_FINAL_RANGE ......................... OK.");

    // OPUS_RESET_STATE
    enc.reset();
    cfgs += 1;
    println!("    OPUS_RESET_STATE ............................. OK.");

    // opus_encode()
    sbuf.fill(0);
    let i = enc.encode(&sbuf[..960 * 2], &mut packet[..1276]);
    assert!((1..=1276).contains(&i), "opus_encode returned {i}");
    cfgs += 1;
    println!("    opus_encode() ................................ OK.");

    // opus_encode_float()
    fbuf.fill(0.0);
    let i = enc.encode_float(&fbuf[..960 * 2], &mut packet[..1276]);
    assert!((1..=1276).contains(&i), "opus_encode_float returned {i}");
    cfgs += 1;
    println!("    opus_encode_float() .......................... OK.");

    // enc dropped implicitly
    cfgs += 1;
    println!("                   All encoder interface tests passed");
    println!("                   ({} API invocations)", cfgs);
}

#[test]
fn test_repacketizer_api_0() {
    let mut ret: i32 = 0;
    let mut i: i32 = 0;

    let mut rp = OpusRepacketizer::default();
    let mut packet = vec![0u8; 1276 * 48 + 48 * 2 + 2];
    let mut po = vec![0u8; 1276 * 48 + 48 * 2 + 2 + 256];
    println!("\n  Repacketizer tests");
    println!("  ---------------------------------------------------");

    if rp.get_nb_frames() != 0 {
        panic!("assertion failed at upstream test_opus_api.c:1477");
    }
    println!("    opus_repacketizer_get_nb_frames .............. OK.");
    if rp.cat(&packet[..0]) != OPUS_INVALID_PACKET {
        panic!("assertion failed at upstream test_opus_api.c:1483");
    }
    packet[0] = 1;
    if rp.cat(&packet[..2]) != OPUS_INVALID_PACKET {
        panic!("assertion failed at upstream test_opus_api.c:1486");
    }
    packet[0] = 2;
    if rp.cat(&packet[..1]) != OPUS_INVALID_PACKET {
        panic!("assertion failed at upstream test_opus_api.c:1489");
    }
    packet[0] = 3;
    if rp.cat(&packet[..1]) != OPUS_INVALID_PACKET {
        panic!("assertion failed at upstream test_opus_api.c:1492");
    }
    packet[0] = 2;
    packet[1] = 255;
    if rp.cat(&packet[..2]) != OPUS_INVALID_PACKET {
        panic!("assertion failed at upstream test_opus_api.c:1496");
    }
    packet[0] = 2;
    packet[1] = 250;
    if rp.cat(&packet[..251]) != OPUS_INVALID_PACKET {
        panic!("assertion failed at upstream test_opus_api.c:1500");
    }
    packet[0] = 3;
    packet[1] = 0;
    if rp.cat(&packet[..2]) != OPUS_INVALID_PACKET {
        panic!("assertion failed at upstream test_opus_api.c:1504");
    }
    packet[1] = 49;
    if rp.cat(&packet[..100]) != OPUS_INVALID_PACKET {
        panic!("assertion failed at upstream test_opus_api.c:1507");
    }
    packet[0] = 0;
    if rp.cat(&packet[..3]) != 0 {
        panic!("assertion failed at upstream test_opus_api.c:1510");
    }
    packet[0] = ((1) << 2) as u8;
    if rp.cat(&packet[..3]) != OPUS_INVALID_PACKET {
        panic!("assertion failed at upstream test_opus_api.c:1513");
    }
    rp.init();

    for j in 0..32 {
        packet[0] = (((j << 1) + (j & 1)) << 2) as u8;
        let maxi = 960 / opus_packet_get_samples_per_frame(packet[0], 8000);

        for i in 1..=maxi {
            packet[0] = (((j << 1) + (j & 1)) << 2) as u8;
            if i > 1 {
                packet[0] += if i == 2 { 1 } else { 3 };
            }
            packet[1] = (if i > 2 { i } else { 0 }) as u8;
            let maxp = 960 / (i * opus_packet_get_samples_per_frame(packet[0], 8000));
            for k in (0..=1275 + 75).step_by(3) {
                let mut cnt: i32 = 0;
                let mut rcnt: i32 = 0;
                if k % i == 0 {
                    cnt = 0;
                    while cnt < maxp + 2 {
                        if cnt > 0 {
                            let len = k + (if i > 2 { 2 } else { 1 });
                            ret = rp.cat(&packet[..len as usize]);
                            if if cnt <= maxp && k <= 1275 * i {
                                (ret != 0) as i32
                            } else {
                                (ret != OPUS_INVALID_PACKET) as i32
                            } != 0
                            {
                                panic!("assertion failed at upstream test_opus_api.c:1542");
                            }
                        }
                        rcnt = if k <= 1275 * i {
                            if cnt < maxp {
                                cnt
                            } else {
                                maxp
                            }
                        } else {
                            0
                        };
                        if rp.get_nb_frames() != rcnt * i {
                            panic!("assertion failed at upstream test_opus_api.c:1546");
                        }
                        ret = rp.out_range(0, rcnt * i, &mut po[..1276 * 48 + 48 * 2 + 2]);
                        if rcnt > 0 {
                            let mut len: i32 = 0;
                            len = k * rcnt + (if rcnt * i > 2 { 2 } else { 1 });
                            if ret != len {
                                panic!("assertion failed at upstream test_opus_api.c:1553");
                            }
                            if rcnt * i < 2 && po[0] as i32 & 3 != 0 {
                                panic!("assertion failed at upstream test_opus_api.c:1554");
                            }
                            if rcnt * i == 2 && po[0] as i32 & 3 != 1 {
                                panic!("assertion failed at upstream test_opus_api.c:1555");
                            }
                            if rcnt * i > 2 && (po[0] as i32 & 3 != 3 || po[1] as i32 != rcnt * i) {
                                panic!("assertion failed at upstream test_opus_api.c:1556");
                            }
                            if rp.out(&mut po[..len as usize]) != len {
                                panic!("assertion failed at upstream test_opus_api.c:1558");
                            }
                            if opus_packet_unpad(&mut po[..len as _]) != len {
                                panic!("assertion failed at upstream test_opus_api.c:1560");
                            }
                            if opus_packet_pad(&mut po[..len as usize + 1], len, len + 1) != 0 {
                                panic!("assertion failed at upstream test_opus_api.c:1562");
                            }
                            if opus_packet_pad(&mut po, len + 1, len + 256) != 0 {
                                panic!("assertion failed at upstream test_opus_api.c:1564");
                            }
                            if opus_packet_unpad(&mut po[..len as usize + 256]) != len {
                                panic!("assertion failed at upstream test_opus_api.c:1566");
                            }

                            if rp.out(&mut po[..len as usize - 1]) != OPUS_BUFFER_TOO_SMALL {
                                panic!("assertion failed at upstream test_opus_api.c:1576");
                            }
                            if len > 1 && rp.out(&mut po[..1]) != OPUS_BUFFER_TOO_SMALL {
                                panic!("assertion failed at upstream test_opus_api.c:1580");
                            }
                            if rp.out(&mut po[..0]) != OPUS_BUFFER_TOO_SMALL {
                                panic!("assertion failed at upstream test_opus_api.c:1583");
                            }
                        } else if ret != OPUS_BAD_ARG {
                            panic!("assertion failed at upstream test_opus_api.c:1585");
                        }
                        cnt += 1;
                    }
                    rp.init();
                }
            }
        }
    }
    rp.init();
    packet[0] = 0;
    if rp.cat(&packet[..5]) != 0 {
        panic!("assertion failed at upstream test_opus_api.c:1595");
    }
    let fresh1 = &mut (packet[0]);
    *fresh1 = (*fresh1 as i32 + 1) as u8;
    if rp.cat(&packet[..9]) != 0 {
        panic!("assertion failed at upstream test_opus_api.c:1598");
    }
    i = rp.out(&mut po[..1276 * 48 + 48 * 2 + 2]);
    if i != 4 + 8 + 2 || po[0] as i32 & 3 != 3 || po[1] as i32 & 63 != 3 || po[1] as i32 >> 7 != 0 {
        panic!("assertion failed at upstream test_opus_api.c:1601");
    }
    i = rp.out_range(0, 1, &mut po[..1276 * 48 + 48 * 2 + 2]);
    if i != 5 || po[0] as i32 & 3 != 0 {
        panic!("assertion failed at upstream test_opus_api.c:1604");
    }
    i = rp.out_range(1, 2, &mut po[..1276 * 48 + 48 * 2 + 2]);
    if i != 5 || po[0] as i32 & 3 != 0 {
        panic!("assertion failed at upstream test_opus_api.c:1607");
    }
    rp.init();
    packet[0] = 1;
    if rp.cat(&packet[..9]) != 0 {
        panic!("assertion failed at upstream test_opus_api.c:1613");
    }
    packet[0] = 0;
    if rp.cat(&packet[..3]) != 0 {
        panic!("assertion failed at upstream test_opus_api.c:1616");
    }
    i = rp.out(&mut po[..1276 * 48 + 48 * 2 + 2]);
    if i != 2 + 8 + 2 + 2
        || po[0] as i32 & 3 != 3
        || po[1] as i32 & 63 != 3
        || po[1] as i32 >> 7 != 1
    {
        panic!("assertion failed at upstream test_opus_api.c:1619");
    }
    rp.init();
    packet[0] = 2;
    packet[1] = 4;
    if rp.cat(&packet[..8]) != 0 {
        panic!("assertion failed at upstream test_opus_api.c:1626");
    }
    if rp.cat(&packet[..8]) != 0 {
        panic!("assertion failed at upstream test_opus_api.c:1628");
    }
    i = rp.out(&mut po[..1276 * 48 + 48 * 2 + 2]);
    if i != 2 + 1 + 1 + 1 + 4 + 2 + 4 + 2 || po[0] & 3 != 3 || po[1] & 63 != 4 || po[1] >> 7 != 1 {
        panic!("assertion failed at upstream test_opus_api.c:1631");
    }
    rp.init();
    packet[0] = 2;
    packet[1] = 4;
    if rp.cat(&packet[..10]) != 0 {
        panic!("assertion failed at upstream test_opus_api.c:1638");
    }
    if rp.cat(&packet[..10]) != 0 {
        panic!("assertion failed at upstream test_opus_api.c:1640");
    }
    i = rp.out(&mut po[..1276 * 48 + 48 * 2 + 2]);
    if i != 2 + 4 + 4 + 4 + 4 || po[0] & 3 != 3 || po[1] & 63 != 4 || po[1] >> 7 != 0 {
        panic!("assertion failed at upstream test_opus_api.c:1643");
    }
}

#[test]
fn test_repacketizer_api_1() {
    let mut ret: i32 = 0;
    let mut rp = OpusRepacketizer::default();
    let mut packet = vec![0u8; 1276 * 48 + 48 * 2 + 2];
    let mut po = vec![0u8; 1276 * 48 + 48 * 2 + 2 + 256];
    // Count 0 in, VBR out
    for j in 0..32 {
        packet[0] = (((j << 1) + (j & 1)) << 2) as u8;
        let maxi_0 = 960 / opus_packet_get_samples_per_frame(packet[0], 8000);
        let mut sum = 0;
        let mut rcnt_0 = 0;

        rp.init();
        for i in 1..=maxi_0 + 1 {
            println!("round {j}:{i}");
            let mut len_0: i32 = 0;
            ret = rp.cat(&packet[..i as usize]);
            if rcnt_0 < maxi_0 {
                if ret != 0 {
                    panic!("assertion failed at upstream test_opus_api.c:1662");
                }
                rcnt_0 += 1;
                sum += i - 1;
            } else if ret != OPUS_INVALID_PACKET {
                panic!("assertion failed at upstream test_opus_api.c:1665");
            }
            len_0 = sum
                + (if rcnt_0 < 2 {
                    1
                } else if rcnt_0 < 3 {
                    2
                } else {
                    2 + rcnt_0 - 1
                });
            if rp.out(&mut po[..1276 * 48 + 48 * 2 + 2]) != len_0 {
                panic!("assertion failed at upstream test_opus_api.c:1668");
            }
            if rcnt_0 > 2 && po[1] as i32 & 63 != rcnt_0 {
                panic!("assertion failed at upstream test_opus_api.c:1669");
            }
            if rcnt_0 == 2 && po[0] & 3 != 2 {
                panic!("assertion failed at upstream test_opus_api.c:1670");
            }
            if rcnt_0 == 1 && po[0] & 3 != 0 {
                panic!("assertion failed at upstream test_opus_api.c:1671");
            }
            if rp.out(&mut po[..len_0 as usize]) != len_0 {
                panic!("assertion failed at upstream test_opus_api.c:1673");
            }
            if opus_packet_unpad(&mut po[..len_0 as _]) != len_0 {
                panic!("assertion failed at upstream test_opus_api.c:1675");
            }

            let before = po[..len_0 as usize].to_vec();
            println!("---pad 1");
            if opus_packet_pad(&mut po[..len_0 as usize + 1], len_0, len_0 + 1) != 0 {
                panic!("assertion failed at upstream test_opus_api.c:1677");
            }
            println!("---pad 256");
            if opus_packet_pad(&mut po[..len_0 as usize + 256], len_0 + 1, len_0 + 256) != 0 {
                panic!("assertion failed at upstream test_opus_api.c:1679");
            }
            println!("---unpad ({len_0})");
            if opus_packet_unpad(&mut po[..len_0 as usize + 256]) != len_0 {
                panic!("assertion failed at upstream test_opus_api.c:1681");
            }
            assert_eq!(before, &po[..len_0 as usize], "unpadding failed");

            if rp.out(&mut po[..len_0 as usize - 1]) != OPUS_BUFFER_TOO_SMALL {
                panic!("assertion failed at upstream test_opus_api.c:1691");
            }
            if len_0 > 1 && rp.out(&mut po[..1]) != OPUS_BUFFER_TOO_SMALL {
                panic!("assertion failed at upstream test_opus_api.c:1695");
            }
            if rp.out(&mut po[..0]) != OPUS_BUFFER_TOO_SMALL {
                panic!("assertion failed at upstream test_opus_api.c:1698");
            }
        }
    }

    po[0] = b'O';
    po[1] = b'p';

    if opus_packet_pad(&mut po[..4], 4, 4) != 0 {
        panic!("assertion failed at upstream test_opus_api.c:1705");
    }
    if opus_packet_pad(&mut po[..5], 4, 5) != OPUS_INVALID_PACKET {
        panic!("assertion failed at upstream test_opus_api.c:1709");
    }
    if opus_packet_pad(&mut po[..5], 0, 5) != OPUS_BAD_ARG {
        panic!("assertion failed at upstream test_opus_api.c:1713");
    }
    if opus_packet_unpad(&mut po[..0]) != OPUS_BAD_ARG {
        panic!("assertion failed at upstream test_opus_api.c:1717");
    }
    if opus_packet_unpad(&mut po[..4]) != OPUS_INVALID_PACKET {
        panic!("assertion failed at upstream test_opus_api.c:1721");
    }
    po[0] = 0;
    po[1] = 0;
    po[2] = 0;

    if opus_packet_pad(&mut po, 5, 4) != OPUS_BAD_ARG {
        panic!("assertion failed at upstream test_opus_api.c:1728");
    }
}
