#![allow(unused_assignments)]
#![allow(deprecated)]

use std::ptr;

fn fail(file: &str, line: usize) {
    panic!("test failed, original: {file} at line {line}");
}

use unsafe_libopus::externs::{free, malloc};
use unsafe_libopus::externs::{memcmp, memcpy, memset};
use unsafe_libopus::{
    opus_decode, opus_decode_float, opus_decoder_create, opus_decoder_ctl, opus_decoder_destroy,
    opus_decoder_get_nb_samples, opus_decoder_get_size, opus_decoder_init, opus_encode,
    opus_encode_float, opus_encoder_create, opus_encoder_ctl, opus_encoder_destroy,
    opus_encoder_get_size, opus_encoder_init, opus_packet_get_bandwidth, opus_packet_get_nb_frames,
    opus_packet_get_nb_samples, opus_packet_get_samples_per_frame, opus_packet_pad,
    opus_packet_parse, opus_packet_unpad, opus_repacketizer_cat, opus_repacketizer_create,
    opus_repacketizer_destroy, opus_repacketizer_get_nb_frames, opus_repacketizer_get_size,
    opus_repacketizer_init, opus_repacketizer_out, opus_repacketizer_out_range, OpusDecoder,
    OpusEncoder, OpusRepacketizer, OPUS_BAD_ARG, OPUS_INVALID_PACKET,
};

static OPUS_RATES: [i32; 5] = [48000, 24000, 16000, 12000, 8000];

#[test]
fn test_opus_decoder_get_size() {
    for c in 0..4 {
        let i = opus_decoder_get_size(c);
        if (c == 1 || c == 2) && (i <= 2048 || i > (1) << 16) || c != 1 && c != 2 && i != 0 {
            fail("tests/test_opus_api.c", 106);
        }
    }
}

#[test]
fn test_opus_decoder_create_init() {
    for c in 0..4 {
        for i in -7..=96000 {
            if !((i == 8000 || i == 12000 || i == 16000 || i == 24000 || i == 48000)
                && (c == 1 || c == 2))
            {
                let fs = match i {
                    -5 => -8000,
                    -6 => 2147483647,
                    -7 => -2147483647 - 1,
                    _ => i,
                };
                let mut err = 0;
                let mut dec = unsafe { opus_decoder_create(fs, c, &mut err) };
                if err != OPUS_BAD_ARG || !dec.is_null() {
                    fail("tests/test_opus_api.c", 128);
                }
                dec = unsafe { opus_decoder_create(fs, c, ptr::null_mut::<i32>()) };
                if !dec.is_null() {
                    fail("tests/test_opus_api.c", 131);
                }
                dec = unsafe { malloc(opus_decoder_get_size(2) as u64) as *mut OpusDecoder };
                if dec.is_null() {
                    fail("tests/test_opus_api.c", 134);
                }
                let err = unsafe { opus_decoder_init(dec, fs, c) };
                if err != OPUS_BAD_ARG {
                    fail("tests/test_opus_api.c", 136);
                }
                unsafe {
                    free(dec as *mut core::ffi::c_void);
                }
            }
        }
    }
}

#[test]
fn test_dec_api() {
    unsafe { test_dec_api_inner() };
}

unsafe fn test_dec_api_inner() {
    let mut dec_final_range: u32 = 0;
    let mut packet = vec![0; 1276];
    let mut fbuf = vec![0.; 1920];
    let mut sbuf = vec![0; 1920];
    let mut err: i32 = 0;
    let mut cfgs = 0;

    let dec = unsafe { opus_decoder_create(48000, 2, &mut err) };
    if err != 0 || dec.is_null() {
        fail("tests/test_opus_api.c", 144);
    }
    err = opus_decoder_ctl!(&mut *dec, 4031, &mut dec_final_range);
    if err != 0 {
        fail("tests/test_opus_api.c", 155);
    }
    println!("    OPUS_GET_FINAL_RANGE ......................... OK.");
    cfgs += 1;
    err = opus_decoder_ctl!(&mut *dec, -(5));
    if err != -(5) {
        fail("tests/test_opus_api.c", 161);
    }
    println!("    OPUS_UNIMPLEMENTED ........................... OK.");
    cfgs += 1;

    let mut i = 0;
    err = opus_decoder_ctl!(&mut *dec, 4009, &mut i);
    if err != 0 || i != 0 {
        fail("tests/test_opus_api.c", 169);
    }
    println!("    OPUS_GET_BANDWIDTH ........................... OK.");
    cfgs += 1;
    err = opus_decoder_ctl!(&mut *dec, 4029, &mut i);
    if err != 0 || i != 48000 {
        fail("tests/test_opus_api.c", 177);
    }
    println!("    OPUS_GET_SAMPLE_RATE ......................... OK.");
    cfgs += 1;

    // GET_PITCH has different execution paths depending on the previously decoded frame.
    err = opus_decoder_ctl!(&mut *dec, 4033, &mut i);
    if err != 0 || i > 0 || i < -1 {
        fail("tests/test_opus_api.c", 187);
    }
    cfgs += 1;
    packet[0] = ((63) << 2) as u8;
    packet[2] = 0;
    packet[1] = packet[2];
    if opus_decode(&mut *dec, &mut packet, 3, sbuf.as_mut_ptr(), 960, 0) != 960 {
        fail("tests/test_opus_api.c", 191);
    }
    cfgs += 1;
    err = opus_decoder_ctl!(&mut *dec, 4033, &mut i);
    if err != 0 || i > 0 || i < -1 {
        fail("tests/test_opus_api.c", 195);
    }
    cfgs += 1;
    packet[0] = 1;
    if opus_decode(&mut *dec, &mut packet, 1, sbuf.as_mut_ptr(), 960, 0) != 960 {
        fail("tests/test_opus_api.c", 198);
    }
    cfgs += 1;
    err = opus_decoder_ctl!(&mut *dec, 4033, &mut i);
    if err != 0 || i > 0 || i < -1 {
        fail("tests/test_opus_api.c", 202);
    }
    cfgs += 1;
    println!("    OPUS_GET_PITCH ............................... OK.");
    err = opus_decoder_ctl!(&mut *dec, 4039, &mut i);
    if err != 0 || i != 960 {
        fail("tests/test_opus_api.c", 210);
    }
    cfgs += 1;
    println!("    OPUS_GET_LAST_PACKET_DURATION ................ OK.");
    err = opus_decoder_ctl!(&mut *dec, 4045, &mut i);
    if err != 0 || i != 0 {
        fail("tests/test_opus_api.c", 217);
    }
    cfgs += 1;
    err = opus_decoder_ctl!(&mut *dec, 4034, -(32769));
    if err != OPUS_BAD_ARG {
        fail("tests/test_opus_api.c", 223);
    }
    cfgs += 1;
    err = opus_decoder_ctl!(&mut *dec, 4034, 32768);
    if err != OPUS_BAD_ARG {
        fail("tests/test_opus_api.c", 226);
    }
    cfgs += 1;
    err = opus_decoder_ctl!(&mut *dec, 4034, -(15));
    if err != 0 {
        fail("tests/test_opus_api.c", 229);
    }
    cfgs += 1;
    err = opus_decoder_ctl!(&mut *dec, 4045, &mut i);
    if err != 0 || i != -(15) {
        fail("tests/test_opus_api.c", 234);
    }
    cfgs += 1;
    println!("    OPUS_SET_GAIN ................................ OK.");
    println!("    OPUS_GET_GAIN ................................ OK.");

    // Reset the decoder
    let dec2 = malloc(opus_decoder_get_size(2) as u64) as *mut OpusDecoder;
    memcpy(
        dec2 as *mut core::ffi::c_void,
        dec as *const core::ffi::c_void,
        opus_decoder_get_size(2) as u64,
    );
    if opus_decoder_ctl!(&mut *dec, 4028) != 0 {
        fail("tests/test_opus_api.c", 242);
    }
    if memcmp(
        dec2 as *const core::ffi::c_void,
        dec as *const core::ffi::c_void,
        opus_decoder_get_size(2) as u64,
    ) == 0
    {
        fail("tests/test_opus_api.c", 243);
    }
    free(dec2 as *mut core::ffi::c_void);
    println!("    OPUS_RESET_STATE ............................. OK.");
    cfgs += 1;
    packet[0] = 0;
    if opus_decoder_get_nb_samples(&mut *dec, &packet[..1]) != 480 {
        fail("tests/test_opus_api.c", 250);
    }
    if opus_packet_get_nb_samples(&packet[..1], 48000) != 480 {
        fail("tests/test_opus_api.c", 251);
    }
    if opus_packet_get_nb_samples(&packet[..1], 96000) != 960 {
        fail("tests/test_opus_api.c", 252);
    }
    if opus_packet_get_nb_samples(&packet[..1], 32000) != 320 {
        fail("tests/test_opus_api.c", 253);
    }
    if opus_packet_get_nb_samples(&packet[..1], 8000) != 80 {
        fail("tests/test_opus_api.c", 254);
    }
    packet[0] = 3;
    if opus_packet_get_nb_samples(&packet[..1], 24000) != OPUS_INVALID_PACKET {
        fail("tests/test_opus_api.c", 256);
    }
    packet[0] = ((63) << 2 | 3) as u8;
    packet[1] = 63;
    if opus_packet_get_nb_samples(&[], 24000) != OPUS_BAD_ARG {
        fail("tests/test_opus_api.c", 259);
    }
    if opus_packet_get_nb_samples(&packet[..2], 48000) != OPUS_INVALID_PACKET {
        fail("tests/test_opus_api.c", 260);
    }
    if opus_decoder_get_nb_samples(&mut *dec, &packet[..2]) != OPUS_INVALID_PACKET {
        fail("tests/test_opus_api.c", 261);
    }
    println!("    opus_{{packet,decoder}}_get_nb_samples() ....... OK.");
    cfgs += 9;
    if OPUS_BAD_ARG != opus_packet_get_nb_frames(&[]) {
        fail("tests/test_opus_api.c", 265);
    }

    for i in 0..256 {
        let l1res: [i32; 4] = [1, 2, 2, OPUS_INVALID_PACKET];
        packet[0] = i as u8;
        if l1res[(packet[0] as i32 & 3) as usize] != opus_packet_get_nb_frames(&packet[..1]) {
            fail("tests/test_opus_api.c", 269);
        }
        cfgs += 1;
        for j in 0..256 {
            packet[1] = j as u8;
            if (if packet[0] as i32 & 3 != 3 {
                l1res[(packet[0] as i32 & 3) as usize]
            } else {
                packet[1] as i32 & 63
            }) != opus_packet_get_nb_frames(&packet[..2])
            {
                fail("tests/test_opus_api.c", 273);
            }
            cfgs += 1;
        }
    }
    println!("    opus_packet_get_nb_frames() .................. OK.");

    for i in 0..256 {
        let mut bw: i32 = 0;
        packet[0] = i as u8;
        bw = packet[0] as i32 >> 4;
        bw = 1101 + (((((bw & 7) * 9) & (63 - (bw & 8))) + 2 + 12 * (bw & 8 != 0) as i32) >> 4);
        if bw != opus_packet_get_bandwidth(packet.as_mut_ptr()) {
            fail("tests/test_opus_api.c", 284);
        }
        cfgs += 1;
    }
    println!("    opus_packet_get_bandwidth() .................. OK.");

    for i in 0..256 {
        let mut fp3s: i32 = 0;
        let mut rate: i32 = 0;
        packet[0] = i as u8;
        fp3s = packet[0] as i32 >> 3;
        fp3s = (((((3 - (fp3s & 3)) * 13) & 119) + 9) >> 2)
            * ((fp3s > 13) as i32 * (3 - (fp3s & 3 == 3) as i32) + 1)
            * 25;
        rate = 0;
        while rate < 5 {
            if OPUS_RATES[rate as usize] * 3 / fp3s
                != opus_packet_get_samples_per_frame(packet[0], OPUS_RATES[rate as usize])
            {
                fail("tests/test_opus_api.c", 295);
            }
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

    if opus_decode(&mut *dec, &mut packet, 51, sbuf.as_mut_ptr(), 960, 0) != OPUS_INVALID_PACKET {
        fail("tests/test_opus_api.c", 305);
    }
    cfgs += 1;
    packet[0] = ((63) << 2) as u8;
    packet[2] = 0;
    packet[1] = packet[2];
    if opus_decode(&mut *dec, &mut packet, -1, sbuf.as_mut_ptr(), 960, 0) != OPUS_BAD_ARG {
        fail("tests/test_opus_api.c", 309);
    }
    cfgs += 1;
    if opus_decode(&mut *dec, &mut packet, 3, sbuf.as_mut_ptr(), 60, 0) != -(2) {
        fail("tests/test_opus_api.c", 311);
    }
    cfgs += 1;
    if opus_decode(&mut *dec, &mut packet, 3, sbuf.as_mut_ptr(), 480, 0) != -(2) {
        fail("tests/test_opus_api.c", 313);
    }
    cfgs += 1;
    if opus_decode(&mut *dec, &mut packet, 3, sbuf.as_mut_ptr(), 960, 0) != 960 {
        fail("tests/test_opus_api.c", 315);
    }
    cfgs += 1;
    println!("    opus_decode() ................................ OK.");
    if opus_decode_float(&mut *dec, &mut packet, 3, &mut fbuf, 960, 0) != 960 {
        fail("tests/test_opus_api.c", 320);
    }
    cfgs += 1;
    println!("    opus_decode_float() .......................... OK.");
    opus_decoder_destroy(dec);
    cfgs += 1;
    println!("                   All decoder interface tests passed");
    println!("                   ({:6} API invocations)", cfgs);
}

#[test]
fn test_parse_header_code_0() {
    unsafe { test_parse_code_0_inner() };
}

unsafe fn test_parse_code_0_inner() {
    let mut packet = vec![0u8; 1276];
    let mut frames: [*const u8; 48] = [ptr::null(); 48];
    let mut size: [i16; 48] = [0; 48];

    for i in 0..64 {
        packet[0] = (i << 2) as u8;
        frames[0] = ptr::null_mut::<u8>();
        frames[1] = ptr::null_mut::<u8>();

        let mut toc = u8::MAX;
        let mut payload_offset = -1;
        let ret = opus_packet_parse(
            &mut packet,
            4,
            Some(&mut toc),
            Some(&mut frames),
            &mut size,
            Some(&mut payload_offset),
        );
        if ret != 1 {
            fail("tests/test_opus_api.c", 729);
        }
        if size[0] != 3 {
            fail("tests/test_opus_api.c", 730);
        }
        if frames[0] != packet.as_ptr().offset(1) {
            fail("tests/test_opus_api.c", 731);
        }
    }
}

#[test]
fn test_parse_header_code_1() {
    unsafe { test_parse_code_1_inner() };
}

unsafe fn test_parse_code_1_inner() {
    let mut packet = vec![0u8; 1276];
    let mut frames: [*const u8; 48] = [ptr::null(); 48];
    let mut size: [i16; 48] = [0; 48];

    // code 1, two frames of the same size
    for i in 0..64 {
        packet[0] = ((i << 2) + 1) as u8;

        for jj in 0..=1275 * 2 + 3 {
            frames[0] = ptr::null_mut::<u8>();
            frames[1] = ptr::null_mut::<u8>();
            let mut toc = u8::MAX;
            let mut payload_offset = -1;

            // this makes no sense anymore
            if jj as usize > packet.len() {
                continue;
            }

            let ret = opus_packet_parse(
                &packet,
                jj,
                Some(&mut toc),
                Some(&mut frames),
                &mut size,
                Some(&mut payload_offset),
            );
            if jj & 1 == 1 && jj <= 2551 {
                // Must pass if payload length even (packet length odd) and
                // size<=2551, must fail otherwise.
                if ret != 2 {
                    fail("tests/test_opus_api.c", 749);
                }
                if size[0] != size[1] || size[0] as i32 != (jj - 1) >> 1 {
                    fail("tests/test_opus_api.c", 750);
                }
                if frames[0] != packet.as_mut_ptr().offset(1 as isize) as *const u8 {
                    fail("tests/test_opus_api.c", 751);
                }
                if frames[1] != (frames[0]).offset(size[0] as i32 as isize) {
                    fail("tests/test_opus_api.c", 752);
                }
                if toc as i32 >> 2 != i {
                    fail("tests/test_opus_api.c", 753);
                }
            } else if ret != OPUS_INVALID_PACKET {
                fail("tests/test_opus_api.c", 754);
            }
        }
    }
}

#[test]
fn test_parse_header_code_2() {
    unsafe { test_parse_code_2_inner() };
}

unsafe fn test_parse_code_2_inner() {
    let mut packet = vec![0u8; 1276];
    let mut frames: [*const u8; 48] = [ptr::null(); 48];
    let mut size: [i16; 48] = [0; 48];

    for i in 0..64 {
        // code 2, length code overflow
        packet[0] = ((i << 2) + 2) as u8;
        frames[0] = ptr::null_mut::<u8>();
        frames[1] = ptr::null_mut::<u8>();

        let mut toc = u8::MAX;
        let mut payload_offset = -1;

        let ret = opus_packet_parse(
            &mut packet,
            1,
            Some(&mut toc),
            Some(&mut frames),
            &mut size,
            Some(&mut payload_offset),
        );
        if ret != OPUS_INVALID_PACKET {
            fail("tests/test_opus_api.c", 767);
        }
        packet[1] = 252;
        frames[0] = ptr::null_mut::<u8>();
        frames[1] = ptr::null_mut::<u8>();

        let mut toc = u8::MAX;
        let mut payload_offset = -1;

        let ret = opus_packet_parse(
            &mut packet,
            2,
            Some(&mut toc),
            Some(&mut frames),
            &mut size,
            Some(&mut payload_offset),
        );
        if ret != OPUS_INVALID_PACKET {
            fail("tests/test_opus_api.c", 772);
        }
        for j in 0..1275 {
            if j < 252 {
                packet[1] = j as u8;
            } else {
                packet[1] = (252 + (j & 3)) as u8;
                packet[2] = ((j - 252) >> 2) as u8;
            }

            // Code 2, one too short
            frames[0] = ptr::null_mut::<u8>();
            frames[1] = ptr::null_mut::<u8>();

            let mut toc = u8::MAX;
            let mut payload_offset = -1;
            let ret = opus_packet_parse(
                &mut packet,
                j + (if j < 252 { 2 } else { 3 }) - 1,
                Some(&mut toc),
                Some(&mut frames),
                &mut size,
                Some(&mut payload_offset),
            );
            if ret != OPUS_INVALID_PACKET {
                fail("tests/test_opus_api.c", 781);
            }

            // Code 2, one too long
            frames[0] = ptr::null_mut::<u8>();
            frames[1] = ptr::null_mut::<u8>();
            let mut toc = u8::MAX;
            let mut payload_offset = -1;

            let packet_len = j + (if j < 252 { 2 } else { 3 }) + 1276;
            // this makes no sense anymore
            if packet_len as usize > packet.len() {
                continue;
            }

            let ret = opus_packet_parse(
                &packet,
                packet_len,
                Some(&mut toc),
                Some(&mut frames),
                &mut size,
                Some(&mut payload_offset),
            );
            if ret != OPUS_INVALID_PACKET {
                fail("tests/test_opus_api.c", 786);
            }

            // Code 2, second zero
            frames[0] = ptr::null_mut::<u8>();
            frames[1] = ptr::null_mut::<u8>();
            let mut toc = u8::MAX;
            let mut payload_offset = -1;

            let packet_len = j + (if j < 252 { 2 } else { 3 });
            // this makes no sense anymore
            if packet_len as usize > packet.len() {
                continue;
            }

            let ret = opus_packet_parse(
                &mut packet,
                packet_len,
                Some(&mut toc),
                Some(&mut frames),
                &mut size,
                Some(&mut payload_offset),
            );
            if ret != 2 {
                fail("tests/test_opus_api.c", 791);
            }
            if size[0] as i32 != j || size[1] as i32 != 0 {
                fail("tests/test_opus_api.c", 792);
            }
            if frames[1] != (frames[0]).offset(size[0] as isize) {
                dbg!(frames[1], size[0], (frames[0]).offset(size[0] as isize));
                fail("tests/test_opus_api.c", 793);
            }
            if toc as i32 >> 2 != i {
                fail("tests/test_opus_api.c", 794);
            }

            // Code 2, normal
            frames[0] = ptr::null_mut::<u8>();
            frames[1] = ptr::null_mut::<u8>();
            let mut toc = u8::MAX;
            let mut payload_offset = -1;

            let packet_len = (j << 1) + 4;

            // this makes no sense anymore
            if packet_len as usize > packet.len() {
                continue;
            }

            let ret = opus_packet_parse(
                &mut packet,
                packet_len,
                Some(&mut toc),
                Some(&mut frames),
                &mut size,
                Some(&mut payload_offset),
            );
            if ret != 2 {
                fail("tests/test_opus_api.c", 799);
            }
            if size[0] as i32 != j
                || size[1] as i32 != (j << 1) + 3 - j - (if j < 252 { 1 } else { 2 })
            {
                fail("tests/test_opus_api.c", 800);
            }
            if frames[1] != (frames[0]).offset(size[0] as isize) {
                fail("tests/test_opus_api.c", 801);
            }
            if toc as i32 >> 2 != i {
                fail("tests/test_opus_api.c", 802);
            }
        }
    }
}

#[test]
fn test_parse_header_code_3_m_truncation() {
    unsafe { test_parse_code_3_m_truncation_inner() };
}

unsafe fn test_parse_code_3_m_truncation_inner() {
    let mut packet = vec![0u8; 1276];
    let mut frames: [*const u8; 48] = [ptr::null(); 48];
    let mut size: [i16; 48] = [0; 48];

    for i in 0..64 {
        packet[0] = ((i << 2) + 3) as u8;
        frames[0] = ptr::null_mut::<u8>();
        frames[1] = ptr::null_mut::<u8>();
        let mut toc = u8::MAX;
        let mut payload_offset = -1;
        let ret = opus_packet_parse(
            &mut packet,
            1,
            Some(&mut toc),
            Some(&mut frames),
            &mut size,
            Some(&mut payload_offset),
        );

        if ret != OPUS_INVALID_PACKET {
            fail("tests/test_opus_api.c", 815);
        }
    }
}

#[test]
fn test_parse_header_code_3_m_0_49_64() {
    unsafe { test_parse_code_3_m_0_49_64_inner() };
}

unsafe fn test_parse_code_3_m_0_49_64_inner() {
    let mut packet = vec![0u8; 1276];
    let mut frames: [*const u8; 48] = [ptr::null(); 48];
    let mut size: [i16; 48] = [0; 48];

    for i in 0..64 {
        packet[0] = ((i << 2) + 3) as u8;
        for jj in 49..=64 {
            packet[1] = (0 + (jj & 63)) as u8;
            frames[0] = ptr::null_mut::<u8>();
            frames[1] = ptr::null_mut::<u8>();

            let mut toc = u8::MAX;
            let mut payload_offset = -1;
            let ret = opus_packet_parse(
                &mut packet,
                1275,
                Some(&mut toc),
                Some(&mut frames),
                &mut size,
                Some(&mut payload_offset),
            );
            if ret != OPUS_INVALID_PACKET {
                fail("tests/test_opus_api.c", 830);
            }
            packet[1] = (128 + (jj & 63)) as u8;
            frames[0] = ptr::null_mut::<u8>();
            frames[1] = ptr::null_mut::<u8>();
            let mut toc = u8::MAX;
            let mut payload_offset = -1;
            let ret = opus_packet_parse(
                &mut packet,
                1275,
                Some(&mut toc),
                Some(&mut frames),
                &mut size,
                Some(&mut payload_offset),
            );
            if ret != OPUS_INVALID_PACKET {
                fail("tests/test_opus_api.c", 835);
            }
            packet[1] = (64 + (jj & 63)) as u8;
            frames[0] = ptr::null_mut::<u8>();
            frames[1] = ptr::null_mut::<u8>();
            let mut toc = u8::MAX;
            let mut payload_offset = -1;
            let ret = opus_packet_parse(
                &mut packet,
                1275,
                Some(&mut toc),
                Some(&mut frames),
                &mut size,
                Some(&mut payload_offset),
            );
            if ret != OPUS_INVALID_PACKET {
                fail("tests/test_opus_api.c", 840);
            }
            packet[1] = (128 + 64 + (jj & 63)) as u8;
            frames[0] = ptr::null_mut::<u8>();
            frames[1] = ptr::null_mut::<u8>();
            let mut toc = u8::MAX;
            let mut payload_offset = -1;
            let ret = opus_packet_parse(
                &mut packet,
                1275,
                Some(&mut toc),
                Some(&mut frames),
                &mut size,
                Some(&mut payload_offset),
            );
            if ret != OPUS_INVALID_PACKET {
                fail("tests/test_opus_api.c", 845);
            }
        }
    }
}

#[test]
fn test_parse_header_code_3_m_1_cbr() {
    unsafe { test_parse_code_3_m_1_cbr_inner() };
}

unsafe fn test_parse_code_3_m_1_cbr_inner() {
    let mut packet = vec![0u8; 1276];
    let mut frames: [*const u8; 48] = [ptr::null(); 48];
    let mut size: [i16; 48] = [0; 48];

    for i in 0..64 {
        packet[0] = ((i << 2) + 3) as u8;
        packet[1] = 1;
        for j in 0..1276 {
            frames[0] = ptr::null_mut::<u8>();
            frames[1] = ptr::null_mut::<u8>();
            let mut toc = u8::MAX;
            let mut payload_offset = -1;

            let packet_len = j + 2;
            // this makes no sense anymore
            if packet_len as usize > packet.len() {
                continue;
            }

            let ret = opus_packet_parse(
                &packet,
                packet_len,
                Some(&mut toc),
                Some(&mut frames),
                &mut size,
                Some(&mut payload_offset),
            );
            if ret != 1 {
                fail("tests/test_opus_api.c", 861);
            }
            if size[0] as i32 != j {
                fail("tests/test_opus_api.c", 862);
            }
            if toc as i32 >> 2 != i {
                fail("tests/test_opus_api.c", 863);
            }
        }
    }
}

#[test]
fn test_parse_header_code_3_m_1_48_cbr() {
    unsafe { test_parse_code_3_m_1_48_cbr_inner() };
}

unsafe fn test_parse_code_3_m_1_48_cbr_inner() {
    let mut packet = vec![0u8; 1276];
    let mut frames: [*const u8; 48] = [ptr::null(); 48];
    let mut size: [i16; 48] = [0; 48];

    for i in 0..64 {
        let mut frame_samp: i32 = 0;
        packet[0] = ((i << 2) + 3) as u8;
        frame_samp = opus_packet_get_samples_per_frame(packet[0], 48000);
        for j in 2..49 {
            packet[1] = j as u8;
            for sz in 2..(j + 2) * 1275 {
                frames[0] = ptr::null_mut::<u8>();
                frames[1] = ptr::null_mut::<u8>();
                let mut toc = u8::MAX;
                let mut payload_offset = -1;

                // this makes no sense anymore
                if sz as usize > packet.len() {
                    continue;
                }

                let ret = opus_packet_parse(
                    &mut packet,
                    sz,
                    Some(&mut toc),
                    Some(&mut frames),
                    &mut size,
                    Some(&mut payload_offset),
                );
                if frame_samp * j <= 5760 && (sz - 2) % j == 0 && (sz - 2) / j < 1276 {
                    if ret != j {
                        fail("tests/test_opus_api.c", 890);
                    }
                    for jj in 1..ret {
                        if frames[jj as usize]
                            != (frames[(jj - 1) as usize])
                                .offset(size[(jj - 1) as usize] as i32 as isize)
                        {
                            fail("tests/test_opus_api.c", 891);
                        }
                    }
                    if toc >> 2 != i {
                        fail("tests/test_opus_api.c", 892);
                    }
                } else if ret != OPUS_INVALID_PACKET {
                    fail("tests/test_opus_api.c", 893);
                }
            }
        }

        packet[1] = (5760 / frame_samp) as u8;
        frames[0] = ptr::null_mut::<u8>();
        frames[1] = ptr::null_mut::<u8>();
        let mut toc = u8::MAX;
        let mut payload_offset = -1;
        let p1 = packet[1];

        let packet_len = 1275 * p1 as i32 + 2;
        // this makes no sense anymore
        if packet_len as usize > packet.len() {
            continue;
        }

        let ret = opus_packet_parse(
            &packet,
            packet_len,
            Some(&mut toc),
            Some(&mut frames),
            &mut size,
            Some(&mut payload_offset),
        );
        if ret != packet[1] as i32 {
            fail("tests/test_opus_api.c", 901);
        }
        for jj in 0..ret {
            if size[jj as usize] != 1275 {
                fail("tests/test_opus_api.c", 902);
            }
        }
    }
}

#[test]
fn test_parse_header_code_3_m_1_48_vbr() {
    unsafe { test_parse_code_3_m_1_48_vbr_inner() };
}

unsafe fn test_parse_code_3_m_1_48_vbr_inner() {
    let mut packet = vec![0u8; 1276];
    let mut frames: [*const u8; 48] = [ptr::null(); 48];
    let mut size: [i16; 48] = [0; 48];

    for i in 0..64 {
        packet[0] = ((i << 2) + 3) as u8;
        packet[1] = (128 + 1) as u8;
        let frame_samp_0 = opus_packet_get_samples_per_frame(packet[0], 48000);

        for jj in 0..1276 {
            frames[0] = ptr::null_mut::<u8>();
            frames[1] = ptr::null_mut::<u8>();
            let mut toc = u8::MAX;
            let mut payload_offset = -1;

            let packet_len = 2 + jj;
            // this makes no sense anymore
            if packet_len as usize > packet.len() {
                continue;
            }

            let ret = opus_packet_parse(
                &mut packet,
                packet_len,
                Some(&mut toc),
                Some(&mut frames),
                &mut size,
                Some(&mut payload_offset),
            );

            if ret != 1 {
                fail("tests/test_opus_api.c", 919);
            }
            if size[0] as i32 != jj {
                fail("tests/test_opus_api.c", 920);
            }
            if toc as i32 >> 2 != i {
                fail("tests/test_opus_api.c", 921);
            }
        }

        for j in 2..49 {
            packet[1] = (128 + j) as u8;
            frames[0] = ptr::null_mut::<u8>();
            frames[1] = ptr::null_mut::<u8>();
            let mut toc = u8::MAX;
            let mut payload_offset = -1;
            let ret = opus_packet_parse(
                &mut packet,
                2 + j - 2,
                Some(&mut toc),
                Some(&mut frames),
                &mut size,
                Some(&mut payload_offset),
            );
            if ret != OPUS_INVALID_PACKET {
                fail("tests/test_opus_api.c", 934);
            }
            packet[2] = 252;
            packet[3 as usize] = 0;
            for jj in 4..2 + j {
                packet[jj as usize] = 0;
            }
            frames[0] = ptr::null_mut::<u8>();
            frames[1] = ptr::null_mut::<u8>();
            let mut toc = u8::MAX;
            let mut payload_offset = -1;
            let ret = opus_packet_parse(
                &mut packet,
                2 + j,
                Some(&mut toc),
                Some(&mut frames),
                &mut size,
                Some(&mut payload_offset),
            );
            if ret != OPUS_INVALID_PACKET {
                fail("tests/test_opus_api.c", 941);
            }
            for jj in 2..2 + j {
                packet[jj as usize] = 0;
            }
            frames[0] = ptr::null_mut::<u8>();
            frames[1] = ptr::null_mut::<u8>();
            let mut toc = u8::MAX;
            let mut payload_offset = -1;
            let ret = opus_packet_parse(
                &mut packet,
                2 + j - 2,
                Some(&mut toc),
                Some(&mut frames),
                &mut size,
                Some(&mut payload_offset),
            );
            if ret != OPUS_INVALID_PACKET {
                fail("tests/test_opus_api.c", 947);
            }
            packet[2] = 252;
            packet[3] = 0;
            for jj in 4..2 + j {
                packet[jj as usize] = 0;
            }
            frames[0] = ptr::null_mut::<u8>();
            frames[1] = ptr::null_mut::<u8>();
            let mut toc = u8::MAX;
            let mut payload_offset = -1;
            let ret = opus_packet_parse(
                &mut packet,
                2 + j + 252 - 1,
                Some(&mut toc),
                Some(&mut frames),
                &mut size,
                Some(&mut payload_offset),
            );

            if ret != OPUS_INVALID_PACKET {
                fail("tests/test_opus_api.c", 955);
            }
            for jj in 2..2 + j {
                packet[jj as usize] = 0;
            }
            frames[0] = ptr::null_mut::<u8>();
            frames[1] = ptr::null_mut::<u8>();
            let mut toc = u8::MAX;
            let mut payload_offset = -1;
            let ret = opus_packet_parse(
                &mut packet,
                2 + j - 1,
                Some(&mut toc),
                Some(&mut frames),
                &mut size,
                Some(&mut payload_offset),
            );

            if frame_samp_0 * j <= 5760 {
                if ret != j {
                    fail("tests/test_opus_api.c", 962);
                }
                for jj in 0..j {
                    if size[jj as usize] as i32 != 0 {
                        fail("tests/test_opus_api.c", 963);
                    }
                }
                if toc >> 2 != i as _ {
                    fail("tests/test_opus_api.c", 964);
                }
            } else if ret != OPUS_INVALID_PACKET {
                fail("tests/test_opus_api.c", 965);
            }
            for sz in 0..8 {
                let tsz: [i32; 8] = [50, 201, 403, 700, 1472, 5110, 20400, 61298];
                let mut pos: i32 = 0;
                let as_0: i32 = (tsz[sz as usize] + i - j - 2) / j;
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
                frames[0] = ptr::null_mut::<u8>();
                frames[1] = ptr::null_mut::<u8>();
                let mut toc = u8::MAX;
                let mut payload_offset = -1;

                let packet_len = tsz[sz as usize] + i;
                // this makes no sense anymore
                if packet_len as usize > packet.len() {
                    continue;
                }

                let ret = opus_packet_parse(
                    &packet,
                    packet_len,
                    Some(&mut toc),
                    Some(&mut frames),
                    &mut size,
                    Some(&mut payload_offset),
                );

                if frame_samp_0 * j <= 5760
                    && as_0 < 1276
                    && tsz[sz as usize] + i - 2 - pos - as_0 * (j - 1) < 1276
                {
                    if ret != j {
                        fail("tests/test_opus_api.c", 981);
                    }
                    for jj in 0..j - 1 {
                        if size[jj as usize] as i32 != as_0 {
                            fail("tests/test_opus_api.c", 982);
                        }
                    }
                    if size[(j - 1) as usize] as i32
                        != tsz[sz as usize] + i - 2 - pos - as_0 * (j - 1)
                    {
                        fail("tests/test_opus_api.c", 983);
                    }
                    if toc as i32 >> 2 != i {
                        fail("tests/test_opus_api.c", 984);
                    }
                } else if ret != OPUS_INVALID_PACKET {
                    fail("tests/test_opus_api.c", 985);
                }
            }
        }
    }
}

#[test]
fn test_parse_header_code_3_padding() {
    unsafe { test_parse_code_3_padding_inner() };
}

unsafe fn test_parse_code_3_padding_inner() {
    let mut packet = vec![0u8; 1276];
    let mut frames: [*const u8; 48] = [ptr::null(); 48];
    let mut size: [i16; 48] = [0; 48];

    for i in 0..64 {
        packet[0] = ((i << 2) + 3) as u8;
        packet[1] = (128 + 1 + 64) as u8;
        for jj in 2..127 {
            packet[jj as usize] = 255;
        }

        frames[0] = ptr::null_mut::<u8>();
        frames[1] = ptr::null_mut::<u8>();
        let mut toc = u8::MAX;
        let mut payload_offset = -1;
        let ret = opus_packet_parse(
            &mut packet,
            127,
            Some(&mut toc),
            Some(&mut frames),
            &mut size,
            Some(&mut payload_offset),
        );

        if ret != OPUS_INVALID_PACKET {
            fail("tests/test_opus_api.c", 1002);
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
                    frames[0] = ptr::null_mut::<u8>();
                    frames[1] = ptr::null_mut::<u8>();
                    let mut payload_offset = -1;
                    let mut toc = u8::MAX;

                    let packet_len = 2 + jj + pos_0 - 1;
                    // this makes no sense anymore
                    if packet_len as usize > packet.len() {
                        continue;
                    }

                    let ret = opus_packet_parse(
                        &packet,
                        packet_len,
                        Some(&mut toc),
                        Some(&mut frames),
                        &mut size,
                        Some(&mut payload_offset),
                    );

                    if ret != OPUS_INVALID_PACKET {
                        fail("tests/test_opus_api.c", 1019);
                    }
                }
                frames[0] = ptr::null_mut::<u8>();
                frames[1] = ptr::null_mut::<u8>();
                let mut toc = u8::MAX;
                let mut payload_offset = -1;

                let packet_len = 2 + jj + tsz_0[sz as usize] + i + pos_0;
                // this makes no sense anymore
                if packet_len as usize > packet.len() {
                    continue;
                }

                let ret = opus_packet_parse(
                    &packet,
                    packet_len,
                    Some(&mut toc),
                    Some(&mut frames),
                    &mut size,
                    Some(&mut payload_offset),
                );

                if tsz_0[sz as usize] + i < 1276 {
                    if ret != 1 {
                        fail("tests/test_opus_api.c", 1026);
                    }
                    if size[0] as i32 != tsz_0[sz as usize] + i {
                        fail("tests/test_opus_api.c", 1027);
                    }
                    if toc as i32 >> 2 != i {
                        fail("tests/test_opus_api.c", 1028);
                    }
                } else if ret != OPUS_INVALID_PACKET {
                    fail("tests/test_opus_api.c", 1029);
                }
            }
        }
    }
}

#[test]
fn test_enc_api() {
    unsafe { test_enc_api_inner() };
}
unsafe fn test_enc_api_inner() {
    let mut enc_final_range: u32 = 0;
    let mut enc: *mut OpusEncoder = ptr::null_mut::<OpusEncoder>();
    let mut i: i32 = 0;
    let mut j: i32 = 0;
    let mut packet = vec![0; 1276];
    let mut fbuf = vec![0.; 1920];
    let mut sbuf = vec![0; 1920];
    let mut c: i32 = 0;
    let mut err: i32 = 0;
    let mut cfgs: i32 = 0;
    cfgs = 0;
    println!("\n  Encoder basic API tests");
    println!("  ---------------------------------------------------");
    c = 0;
    while c < 4 {
        i = opus_encoder_get_size(c);
        if (c == 1 || c == 2) && (i <= 2048 || i > (1) << 17) || c != 1 && c != 2 && i != 0 {
            fail("tests/test_opus_api.c", 1084);
        }
        println!(
            "    opus_encoder_get_size({})={} ...............{} OK.",
            c,
            i,
            if i > 0 { "" } else { "...." }
        );
        cfgs += 1;
        c += 1;
    }
    c = 0;
    while c < 4 {
        i = -(7);
        while i <= 96000 {
            let mut fs: i32 = 0;
            if !((i == 8000 || i == 12000 || i == 16000 || i == 24000 || i == 48000)
                && (c == 1 || c == 2))
            {
                match i {
                    -5 => {
                        fs = -(8000);
                    }
                    -6 => {
                        fs = 2147483647;
                    }
                    -7 => {
                        fs = -(2147483647) - 1;
                    }
                    _ => {
                        fs = i;
                    }
                }
                err = 0;
                enc = opus_encoder_create(fs, c, 2048, &mut err);
                if err != OPUS_BAD_ARG || !enc.is_null() {
                    fail("tests/test_opus_api.c", 1106);
                }
                cfgs += 1;
                enc = opus_encoder_create(fs, c, 2048, ptr::null_mut::<i32>());
                if !enc.is_null() {
                    fail("tests/test_opus_api.c", 1109);
                }
                cfgs += 1;
                opus_encoder_destroy(enc);
                enc = malloc(opus_encoder_get_size(2) as u64) as *mut OpusEncoder;
                if enc.is_null() {
                    fail("tests/test_opus_api.c", 1113);
                }
                err = opus_encoder_init(enc, fs, c, 2048);
                if err != OPUS_BAD_ARG {
                    fail("tests/test_opus_api.c", 1115);
                }
                cfgs += 1;
                free(enc as *mut core::ffi::c_void);
            }
            i += 1;
        }
        c += 1;
    }
    enc = opus_encoder_create(48000, 2, -(1000), ptr::null_mut::<i32>());
    if !enc.is_null() {
        fail("tests/test_opus_api.c", 1122);
    }
    cfgs += 1;
    enc = opus_encoder_create(48000, 2, -(1000), &mut err);
    if err != OPUS_BAD_ARG || !enc.is_null() {
        fail("tests/test_opus_api.c", 1127);
    }
    cfgs += 1;
    enc = opus_encoder_create(48000, 2, 2048, ptr::null_mut::<i32>());
    if enc.is_null() {
        fail("tests/test_opus_api.c", 1132);
    }
    opus_encoder_destroy(enc);
    cfgs += 1;
    enc = opus_encoder_create(48000, 2, 2051, &mut err);
    if err != 0 || enc.is_null() {
        fail("tests/test_opus_api.c", 1138);
    }
    cfgs += 1;
    err = opus_encoder_ctl!(enc, 4027, &mut i);
    if err != 0 || i < 0 || i > 32766 {
        fail("tests/test_opus_api.c", 1141);
    }
    cfgs += 1;
    opus_encoder_destroy(enc);
    enc = opus_encoder_create(48000, 2, 2049, &mut err);
    if err != 0 || enc.is_null() {
        fail("tests/test_opus_api.c", 1147);
    }
    cfgs += 1;
    err = opus_encoder_ctl!(enc, 4027, &mut i);
    if err != 0 || i < 0 || i > 32766 {
        fail("tests/test_opus_api.c", 1150);
    }
    opus_encoder_destroy(enc);
    cfgs += 1;
    enc = opus_encoder_create(48000, 2, 2048, &mut err);
    if err != 0 || enc.is_null() {
        fail("tests/test_opus_api.c", 1156);
    }
    cfgs += 1;
    println!("    opus_encoder_create() ........................ OK.");
    println!("    opus_encoder_init() .......................... OK.");
    i = -(12345);
    err = opus_encoder_ctl!(enc, 4027, &mut i);
    if err != 0 || i < 0 || i > 32766 {
        fail("tests/test_opus_api.c", 1165);
    }
    cfgs += 1;
    println!("    OPUS_GET_LOOKAHEAD ........................... OK.");
    err = opus_encoder_ctl!(enc, 4029, &mut i);
    if err != 0 || i != 48000 {
        fail("tests/test_opus_api.c", 1173);
    }
    cfgs += 1;
    println!("    OPUS_GET_SAMPLE_RATE ......................... OK.");
    if opus_encoder_ctl!(enc, -(5)) != -(5) {
        fail("tests/test_opus_api.c", 1180);
    }
    println!("    OPUS_UNIMPLEMENTED ........................... OK.");
    cfgs += 1;
    i = -1;
    if opus_encoder_ctl!(enc, 4000, i) == 0 {
        fail("tests/test_opus_api.c", 1190);
    }
    i = -(1000);
    if opus_encoder_ctl!(enc, 4000, i) == 0 {
        fail("tests/test_opus_api.c", 1190);
    }
    i = 2049;
    j = i;
    if opus_encoder_ctl!(enc, 4000, i) != 0 {
        fail("tests/test_opus_api.c", 1190);
    }
    i = -(12345);
    err = opus_encoder_ctl!(enc, 4001, &mut i);
    if err != 0 || i != j {
        fail("tests/test_opus_api.c", 1190);
    }
    i = 2051;
    j = i;
    if opus_encoder_ctl!(enc, 4000, i) != 0 {
        fail("tests/test_opus_api.c", 1190);
    }
    println!("    OPUS_SET_APPLICATION ......................... OK.");
    i = -(12345);
    err = opus_encoder_ctl!(enc, 4001, &mut i);
    if err != 0 || i != j {
        fail("tests/test_opus_api.c", 1190);
    }
    println!("    OPUS_GET_APPLICATION ......................... OK.");
    cfgs += 6;
    if opus_encoder_ctl!(enc, 4002, 1073741832) != 0 {
        fail("tests/test_opus_api.c", 1195);
    }
    cfgs += 1;
    if opus_encoder_ctl!(enc, 4003, &mut i) != 0 {
        fail("tests/test_opus_api.c", 1198);
    }
    if i > 700000 || i < 256000 {
        fail("tests/test_opus_api.c", 1199);
    }
    cfgs += 1;
    i = -(12345);
    if opus_encoder_ctl!(enc, 4002, i) == 0 {
        fail("tests/test_opus_api.c", 1204);
    }
    i = 0;
    if opus_encoder_ctl!(enc, 4002, i) == 0 {
        fail("tests/test_opus_api.c", 1204);
    }
    i = 500;
    j = i;
    if opus_encoder_ctl!(enc, 4002, i) != 0 {
        fail("tests/test_opus_api.c", 1204);
    }
    i = -(12345);
    err = opus_encoder_ctl!(enc, 4003, &mut i);
    if err != 0 || i != j {
        fail("tests/test_opus_api.c", 1204);
    }
    i = 256000;
    j = i;
    if opus_encoder_ctl!(enc, 4002, i) != 0 {
        fail("tests/test_opus_api.c", 1204);
    }
    println!("    OPUS_SET_BITRATE ............................. OK.");
    i = -(12345);
    err = opus_encoder_ctl!(enc, 4003, &mut i);
    if err != 0 || i != j {
        fail("tests/test_opus_api.c", 1204);
    }
    println!("    OPUS_GET_BITRATE ............................. OK.");
    cfgs += 6;
    i = -1;
    if opus_encoder_ctl!(enc, 4022, i) == 0 {
        fail("tests/test_opus_api.c", 1212);
    }
    i = 3;
    if opus_encoder_ctl!(enc, 4022, i) == 0 {
        fail("tests/test_opus_api.c", 1212);
    }
    i = 1;
    j = i;
    if opus_encoder_ctl!(enc, 4022, i) != 0 {
        fail("tests/test_opus_api.c", 1212);
    }
    i = -(12345);
    err = opus_encoder_ctl!(enc, 4023, &mut i);
    if err != 0 || i != j {
        fail("tests/test_opus_api.c", 1212);
    }
    i = -(1000);
    j = i;
    if opus_encoder_ctl!(enc, 4022, i) != 0 {
        fail("tests/test_opus_api.c", 1212);
    }
    println!("    OPUS_SET_FORCE_CHANNELS ...................... OK.");
    i = -(12345);
    err = opus_encoder_ctl!(enc, 4023, &mut i);
    if err != 0 || i != j {
        fail("tests/test_opus_api.c", 1212);
    }
    println!("    OPUS_GET_FORCE_CHANNELS ...................... OK.");
    cfgs += 6;
    i = -(2);
    if opus_encoder_ctl!(enc, 4008, i) == 0 {
        fail("tests/test_opus_api.c", 1215);
    }
    cfgs += 1;
    i = 1105 + 1;
    if opus_encoder_ctl!(enc, 4008, i) == 0 {
        fail("tests/test_opus_api.c", 1218);
    }
    cfgs += 1;
    i = 1101;
    if opus_encoder_ctl!(enc, 4008, i) != 0 {
        fail("tests/test_opus_api.c", 1221);
    }
    cfgs += 1;
    i = 1105;
    if opus_encoder_ctl!(enc, 4008, i) != 0 {
        fail("tests/test_opus_api.c", 1224);
    }
    cfgs += 1;
    i = 1103;
    if opus_encoder_ctl!(enc, 4008, i) != 0 {
        fail("tests/test_opus_api.c", 1227);
    }
    cfgs += 1;
    i = 1102;
    if opus_encoder_ctl!(enc, 4008, i) != 0 {
        fail("tests/test_opus_api.c", 1230);
    }
    cfgs += 1;
    println!("    OPUS_SET_BANDWIDTH ........................... OK.");
    i = -(12345);
    err = opus_encoder_ctl!(enc, 4009, &mut i);
    if err != 0 || i != 1101 && i != 1102 && i != 1103 && i != 1105 && i != -(1000) {
        fail("tests/test_opus_api.c", 1240);
    }
    cfgs += 1;
    if opus_encoder_ctl!(enc, 4008, -(1000)) != 0 {
        fail("tests/test_opus_api.c", 1242);
    }
    cfgs += 1;
    println!("    OPUS_GET_BANDWIDTH ........................... OK.");
    i = -(2);
    if opus_encoder_ctl!(enc, 4004, i) == 0 {
        fail("tests/test_opus_api.c", 1250);
    }
    cfgs += 1;
    i = 1105 + 1;
    if opus_encoder_ctl!(enc, 4004, i) == 0 {
        fail("tests/test_opus_api.c", 1253);
    }
    cfgs += 1;
    i = 1101;
    if opus_encoder_ctl!(enc, 4004, i) != 0 {
        fail("tests/test_opus_api.c", 1256);
    }
    cfgs += 1;
    i = 1105;
    if opus_encoder_ctl!(enc, 4004, i) != 0 {
        fail("tests/test_opus_api.c", 1259);
    }
    cfgs += 1;
    i = 1103;
    if opus_encoder_ctl!(enc, 4004, i) != 0 {
        fail("tests/test_opus_api.c", 1262);
    }
    cfgs += 1;
    i = 1102;
    if opus_encoder_ctl!(enc, 4004, i) != 0 {
        fail("tests/test_opus_api.c", 1265);
    }
    cfgs += 1;
    println!("    OPUS_SET_MAX_BANDWIDTH ....................... OK.");
    i = -(12345);
    err = opus_encoder_ctl!(enc, 4005, &mut i);
    if err != 0 || i != 1101 && i != 1102 && i != 1103 && i != 1105 {
        fail("tests/test_opus_api.c", 1275);
    }
    cfgs += 1;
    println!("    OPUS_GET_MAX_BANDWIDTH ....................... OK.");
    i = -1;
    if opus_encoder_ctl!(enc, 4016, i) == 0 {
        fail("tests/test_opus_api.c", 1288);
    }
    i = 2;
    if opus_encoder_ctl!(enc, 4016, i) == 0 {
        fail("tests/test_opus_api.c", 1288);
    }
    i = 1;
    j = i;
    if opus_encoder_ctl!(enc, 4016, i) != 0 {
        fail("tests/test_opus_api.c", 1288);
    }
    i = -(12345);
    err = opus_encoder_ctl!(enc, 4017, &mut i);
    if err != 0 || i != j {
        fail("tests/test_opus_api.c", 1288);
    }
    i = 0;
    j = i;
    if opus_encoder_ctl!(enc, 4016, i) != 0 {
        fail("tests/test_opus_api.c", 1288);
    }
    println!("    OPUS_SET_DTX ................................. OK.");
    i = -(12345);
    err = opus_encoder_ctl!(enc, 4017, &mut i);
    if err != 0 || i != j {
        fail("tests/test_opus_api.c", 1288);
    }
    println!("    OPUS_GET_DTX ................................. OK.");
    cfgs += 6;
    i = -1;
    if opus_encoder_ctl!(enc, 4010, i) == 0 {
        fail("tests/test_opus_api.c", 1296);
    }
    i = 11;
    if opus_encoder_ctl!(enc, 4010, i) == 0 {
        fail("tests/test_opus_api.c", 1296);
    }
    i = 0;
    j = i;
    if opus_encoder_ctl!(enc, 4010, i) != 0 {
        fail("tests/test_opus_api.c", 1296);
    }
    i = -(12345);
    err = opus_encoder_ctl!(enc, 4011, &mut i);
    if err != 0 || i != j {
        fail("tests/test_opus_api.c", 1296);
    }
    i = 10;
    j = i;
    if opus_encoder_ctl!(enc, 4010, i) != 0 {
        fail("tests/test_opus_api.c", 1296);
    }
    println!("    OPUS_SET_COMPLEXITY .......................... OK.");
    i = -(12345);
    err = opus_encoder_ctl!(enc, 4011, &mut i);
    if err != 0 || i != j {
        fail("tests/test_opus_api.c", 1296);
    }
    println!("    OPUS_GET_COMPLEXITY .......................... OK.");
    cfgs += 6;
    i = -1;
    if opus_encoder_ctl!(enc, 4012, i) == 0 {
        fail("tests/test_opus_api.c", 1304);
    }
    i = 2;
    if opus_encoder_ctl!(enc, 4012, i) == 0 {
        fail("tests/test_opus_api.c", 1304);
    }
    i = 1;
    j = i;
    if opus_encoder_ctl!(enc, 4012, i) != 0 {
        fail("tests/test_opus_api.c", 1304);
    }
    i = -(12345);
    err = opus_encoder_ctl!(enc, 4013, &mut i);
    if err != 0 || i != j {
        fail("tests/test_opus_api.c", 1304);
    }
    i = 0;
    j = i;
    if opus_encoder_ctl!(enc, 4012, i) != 0 {
        fail("tests/test_opus_api.c", 1304);
    }
    println!("    OPUS_SET_INBAND_FEC .......................... OK.");
    i = -(12345);
    err = opus_encoder_ctl!(enc, 4013, &mut i);
    if err != 0 || i != j {
        fail("tests/test_opus_api.c", 1304);
    }
    println!("    OPUS_GET_INBAND_FEC .......................... OK.");
    cfgs += 6;
    i = -1;
    if opus_encoder_ctl!(enc, 4014, i) == 0 {
        fail("tests/test_opus_api.c", 1312);
    }
    i = 101;
    if opus_encoder_ctl!(enc, 4014, i) == 0 {
        fail("tests/test_opus_api.c", 1312);
    }
    i = 100;
    j = i;
    if opus_encoder_ctl!(enc, 4014, i) != 0 {
        fail("tests/test_opus_api.c", 1312);
    }
    i = -(12345);
    err = opus_encoder_ctl!(enc, 4015, &mut i);
    if err != 0 || i != j {
        fail("tests/test_opus_api.c", 1312);
    }
    i = 0;
    j = i;
    if opus_encoder_ctl!(enc, 4014, i) != 0 {
        fail("tests/test_opus_api.c", 1312);
    }
    println!("    OPUS_SET_PACKET_LOSS_PERC .................... OK.");
    i = -(12345);
    err = opus_encoder_ctl!(enc, 4015, &mut i);
    if err != 0 || i != j {
        fail("tests/test_opus_api.c", 1312);
    }
    println!("    OPUS_GET_PACKET_LOSS_PERC .................... OK.");
    cfgs += 6;
    i = -1;
    if opus_encoder_ctl!(enc, 4006, i) == 0 {
        fail("tests/test_opus_api.c", 1320);
    }
    i = 2;
    if opus_encoder_ctl!(enc, 4006, i) == 0 {
        fail("tests/test_opus_api.c", 1320);
    }
    i = 1;
    j = i;
    if opus_encoder_ctl!(enc, 4006, i) != 0 {
        fail("tests/test_opus_api.c", 1320);
    }
    i = -(12345);
    err = opus_encoder_ctl!(enc, 4007, &mut i);
    if err != 0 || i != j {
        fail("tests/test_opus_api.c", 1320);
    }
    i = 0;
    j = i;
    if opus_encoder_ctl!(enc, 4006, i) != 0 {
        fail("tests/test_opus_api.c", 1320);
    }
    println!("    OPUS_SET_VBR ................................. OK.");
    i = -(12345);
    err = opus_encoder_ctl!(enc, 4007, &mut i);
    if err != 0 || i != j {
        fail("tests/test_opus_api.c", 1320);
    }
    println!("    OPUS_GET_VBR ................................. OK.");
    cfgs += 6;
    i = -1;
    if opus_encoder_ctl!(enc, 4020, i) == 0 {
        fail("tests/test_opus_api.c", 1336);
    }
    i = 2;
    if opus_encoder_ctl!(enc, 4020, i) == 0 {
        fail("tests/test_opus_api.c", 1336);
    }
    i = 1;
    j = i;
    if opus_encoder_ctl!(enc, 4020, i) != 0 {
        fail("tests/test_opus_api.c", 1336);
    }
    i = -(12345);
    err = opus_encoder_ctl!(enc, 4021, &mut i);
    if err != 0 || i != j {
        fail("tests/test_opus_api.c", 1336);
    }
    i = 0;
    j = i;
    if opus_encoder_ctl!(enc, 4020, i) != 0 {
        fail("tests/test_opus_api.c", 1336);
    }
    println!("    OPUS_SET_VBR_CONSTRAINT ...................... OK.");
    i = -(12345);
    err = opus_encoder_ctl!(enc, 4021, &mut i);
    if err != 0 || i != j {
        fail("tests/test_opus_api.c", 1336);
    }
    println!("    OPUS_GET_VBR_CONSTRAINT ...................... OK.");
    cfgs += 6;
    i = -(12345);
    if opus_encoder_ctl!(enc, 4024, i) == 0 {
        fail("tests/test_opus_api.c", 1344);
    }
    i = 0x7fffffff;
    if opus_encoder_ctl!(enc, 4024, i) == 0 {
        fail("tests/test_opus_api.c", 1344);
    }
    i = 3002;
    j = i;
    if opus_encoder_ctl!(enc, 4024, i) != 0 {
        fail("tests/test_opus_api.c", 1344);
    }
    i = -(12345);
    err = opus_encoder_ctl!(enc, 4025, &mut i);
    if err != 0 || i != j {
        fail("tests/test_opus_api.c", 1344);
    }
    i = -(1000);
    j = i;
    if opus_encoder_ctl!(enc, 4024, i) != 0 {
        fail("tests/test_opus_api.c", 1344);
    }
    println!("    OPUS_SET_SIGNAL .............................. OK.");
    i = -(12345);
    err = opus_encoder_ctl!(enc, 4025, &mut i);
    if err != 0 || i != j {
        fail("tests/test_opus_api.c", 1344);
    }
    println!("    OPUS_GET_SIGNAL .............................. OK.");
    cfgs += 6;
    i = 7;
    if opus_encoder_ctl!(enc, 4036, i) == 0 {
        fail("tests/test_opus_api.c", 1351);
    }
    i = 25;
    if opus_encoder_ctl!(enc, 4036, i) == 0 {
        fail("tests/test_opus_api.c", 1351);
    }
    i = 16;
    j = i;
    if opus_encoder_ctl!(enc, 4036, i) != 0 {
        fail("tests/test_opus_api.c", 1351);
    }
    i = -(12345);
    err = opus_encoder_ctl!(enc, 4037, &mut i);
    if err != 0 || i != j {
        fail("tests/test_opus_api.c", 1351);
    }
    i = 24;
    j = i;
    if opus_encoder_ctl!(enc, 4036, i) != 0 {
        fail("tests/test_opus_api.c", 1351);
    }
    println!("    OPUS_SET_LSB_DEPTH ........................... OK.");
    i = -(12345);
    err = opus_encoder_ctl!(enc, 4037, &mut i);
    if err != 0 || i != j {
        fail("tests/test_opus_api.c", 1351);
    }
    println!("    OPUS_GET_LSB_DEPTH ........................... OK.");
    cfgs += 6;
    err = opus_encoder_ctl!(enc, 4043, &mut i);
    if i != 0 {
        fail("tests/test_opus_api.c", 1354);
    }
    cfgs += 1;
    i = -1;
    if opus_encoder_ctl!(enc, 4042, i) == 0 {
        fail("tests/test_opus_api.c", 1361);
    }
    i = 2;
    if opus_encoder_ctl!(enc, 4042, i) == 0 {
        fail("tests/test_opus_api.c", 1361);
    }
    i = 1;
    j = i;
    if opus_encoder_ctl!(enc, 4042, i) != 0 {
        fail("tests/test_opus_api.c", 1361);
    }
    i = -(12345);
    err = opus_encoder_ctl!(enc, 4043, &mut i);
    if err != 0 || i != j {
        fail("tests/test_opus_api.c", 1361);
    }
    i = 0;
    j = i;
    if opus_encoder_ctl!(enc, 4042, i) != 0 {
        fail("tests/test_opus_api.c", 1361);
    }
    println!("    OPUS_SET_PREDICTION_DISABLED ................. OK.");
    i = -(12345);
    err = opus_encoder_ctl!(enc, 4043, &mut i);
    if err != 0 || i != j {
        fail("tests/test_opus_api.c", 1361);
    }
    println!("    OPUS_GET_PREDICTION_DISABLED ................. OK.");
    cfgs += 6;
    err = opus_encoder_ctl!(enc, 4040, 5001);
    if err != 0 {
        fail("tests/test_opus_api.c", 1367);
    }
    cfgs += 1;
    err = opus_encoder_ctl!(enc, 4040, 5002);
    if err != 0 {
        fail("tests/test_opus_api.c", 1370);
    }
    cfgs += 1;
    err = opus_encoder_ctl!(enc, 4040, 5003);
    if err != 0 {
        fail("tests/test_opus_api.c", 1373);
    }
    cfgs += 1;
    err = opus_encoder_ctl!(enc, 4040, 5004);
    if err != 0 {
        fail("tests/test_opus_api.c", 1376);
    }
    cfgs += 1;
    err = opus_encoder_ctl!(enc, 4040, 5005);
    if err != 0 {
        fail("tests/test_opus_api.c", 1379);
    }
    cfgs += 1;
    err = opus_encoder_ctl!(enc, 4040, 5006);
    if err != 0 {
        fail("tests/test_opus_api.c", 1382);
    }
    cfgs += 1;
    err = opus_encoder_ctl!(enc, 4040, 5007);
    if err != 0 {
        fail("tests/test_opus_api.c", 1385);
    }
    cfgs += 1;
    err = opus_encoder_ctl!(enc, 4040, 5008);
    if err != 0 {
        fail("tests/test_opus_api.c", 1388);
    }
    cfgs += 1;
    err = opus_encoder_ctl!(enc, 4040, 5009);
    if err != 0 {
        fail("tests/test_opus_api.c", 1391);
    }
    cfgs += 1;
    i = 0;
    if opus_encoder_ctl!(enc, 4040, i) == 0 {
        fail("tests/test_opus_api.c", 1396);
    }
    i = -1;
    if opus_encoder_ctl!(enc, 4040, i) == 0 {
        fail("tests/test_opus_api.c", 1396);
    }
    i = 5006;
    j = i;
    if opus_encoder_ctl!(enc, 4040, i) != 0 {
        fail("tests/test_opus_api.c", 1396);
    }
    i = -(12345);
    err = opus_encoder_ctl!(enc, 4041, &mut i);
    if err != 0 || i != j {
        fail("tests/test_opus_api.c", 1396);
    }
    i = 5000;
    j = i;
    if opus_encoder_ctl!(enc, 4040, i) != 0 {
        fail("tests/test_opus_api.c", 1396);
    }
    println!("    OPUS_SET_EXPERT_FRAME_DURATION ............... OK.");
    i = -(12345);
    err = opus_encoder_ctl!(enc, 4041, &mut i);
    if err != 0 || i != j {
        fail("tests/test_opus_api.c", 1396);
    }
    println!("    OPUS_GET_EXPERT_FRAME_DURATION ............... OK.");
    cfgs += 6;
    if opus_encoder_ctl!(enc, 4031, &mut enc_final_range) != 0 {
        fail("tests/test_opus_api.c", 1403);
    }
    cfgs += 1;
    println!("    OPUS_GET_FINAL_RANGE ......................... OK.");
    if opus_encoder_ctl!(enc, 4028) != 0 {
        fail("tests/test_opus_api.c", 1408);
    }
    cfgs += 1;
    println!("    OPUS_RESET_STATE ............................. OK.");
    memset(
        sbuf.as_mut_ptr() as *mut core::ffi::c_void,
        0,
        (::core::mem::size_of::<i16>() as u64)
            .wrapping_mul(2)
            .wrapping_mul(960),
    );
    i = opus_encode(
        enc,
        sbuf.as_mut_ptr(),
        960,
        packet.as_mut_ptr(),
        ::core::mem::size_of::<[u8; 1276]>() as u64 as i32,
    );
    if i < 1 || i > ::core::mem::size_of::<[u8; 1276]>() as u64 as i32 {
        fail("tests/test_opus_api.c", 1415);
    }
    cfgs += 1;
    println!("    opus_encode() ................................ OK.");
    memset(
        fbuf.as_mut_ptr() as *mut core::ffi::c_void,
        0,
        (::core::mem::size_of::<f32>() as u64)
            .wrapping_mul(2)
            .wrapping_mul(960),
    );
    i = opus_encode_float(
        enc,
        fbuf.as_mut_ptr(),
        960,
        packet.as_mut_ptr(),
        ::core::mem::size_of::<[u8; 1276]>() as u64 as i32,
    );
    if i < 1 || i > ::core::mem::size_of::<[u8; 1276]>() as u64 as i32 {
        fail("tests/test_opus_api.c", 1423);
    }
    cfgs += 1;
    println!("    opus_encode_float() .......................... OK.");
    opus_encoder_destroy(enc);
    cfgs += 1;
    println!("                   All encoder interface tests passed");

    println!("                   ({} API invocations)", cfgs);
}

#[test]
fn test_repacketizer_api() {
    unsafe { test_repacketizer_api_inner() };
}

unsafe fn test_repacketizer_api_inner() {
    let mut ret: i32 = 0;
    let mut cfgs: i32 = 0;
    let mut i: i32 = 0;
    let mut j: i32 = 0;
    let mut k: i32 = 0;
    let mut rp: *mut OpusRepacketizer = ptr::null_mut::<OpusRepacketizer>();
    let mut packet = vec![0u8; 1276 * 48 + 48 * 2 + 2];
    let mut po: *mut u8 = ptr::null_mut::<u8>();
    cfgs = 0;
    println!("\n  Repacketizer tests");
    println!("  ---------------------------------------------------");
    po = malloc((1276 * 48 + 48 * 2 + 2 + 256) as u64) as *mut u8;
    if po.is_null() {
        fail("tests/test_opus_api.c", 1458);
    }
    i = opus_repacketizer_get_size();
    if i <= 0 {
        fail("tests/test_opus_api.c", 1461);
    }
    cfgs += 1;
    println!("    opus_repacketizer_get_size()={} ............. OK.", i);
    rp = malloc(i as u64) as *mut OpusRepacketizer;
    rp = opus_repacketizer_init(rp);
    if rp.is_null() {
        fail("tests/test_opus_api.c", 1467);
    }
    cfgs += 1;
    free(rp as *mut core::ffi::c_void);
    println!("    opus_repacketizer_init ....................... OK.");
    rp = opus_repacketizer_create();
    if rp.is_null() {
        fail("tests/test_opus_api.c", 1473);
    }
    cfgs += 1;
    println!("    opus_repacketizer_create ..................... OK.");
    if opus_repacketizer_get_nb_frames(rp) != 0 {
        fail("tests/test_opus_api.c", 1477);
    }
    cfgs += 1;
    println!("    opus_repacketizer_get_nb_frames .............. OK.");
    if opus_repacketizer_cat(rp, &packet[..0], 0) != OPUS_INVALID_PACKET {
        fail("tests/test_opus_api.c", 1483);
    }
    cfgs += 1;
    packet[0] = 1;
    if opus_repacketizer_cat(rp, &packet[..2], 2) != OPUS_INVALID_PACKET {
        fail("tests/test_opus_api.c", 1486);
    }
    cfgs += 1;
    packet[0] = 2;
    if opus_repacketizer_cat(rp, &packet[..1], 1) != OPUS_INVALID_PACKET {
        fail("tests/test_opus_api.c", 1489);
    }
    cfgs += 1;
    packet[0] = 3;
    if opus_repacketizer_cat(rp, &packet[..1], 1) != OPUS_INVALID_PACKET {
        fail("tests/test_opus_api.c", 1492);
    }
    cfgs += 1;
    packet[0] = 2;
    packet[1] = 255;
    if opus_repacketizer_cat(rp, &packet[..2], 2) != OPUS_INVALID_PACKET {
        fail("tests/test_opus_api.c", 1496);
    }
    cfgs += 1;
    packet[0] = 2;
    packet[1] = 250;
    if opus_repacketizer_cat(rp, &packet[..251], 251) != OPUS_INVALID_PACKET {
        fail("tests/test_opus_api.c", 1500);
    }
    cfgs += 1;
    packet[0] = 3;
    packet[1] = 0;
    if opus_repacketizer_cat(rp, &packet[..2], 2) != OPUS_INVALID_PACKET {
        fail("tests/test_opus_api.c", 1504);
    }
    cfgs += 1;
    packet[1] = 49;
    if opus_repacketizer_cat(rp, &packet[..100], 100) != OPUS_INVALID_PACKET {
        fail("tests/test_opus_api.c", 1507);
    }
    cfgs += 1;
    packet[0] = 0;
    if opus_repacketizer_cat(rp, &packet[..3], 3) != 0 {
        fail("tests/test_opus_api.c", 1510);
    }
    cfgs += 1;
    packet[0] = ((1) << 2) as u8;
    if opus_repacketizer_cat(rp, &packet[..3], 3) != OPUS_INVALID_PACKET {
        fail("tests/test_opus_api.c", 1513);
    }
    cfgs += 1;
    opus_repacketizer_init(rp);
    j = 0;
    while j < 32 {
        let mut maxi: i32 = 0;
        packet[0] = (((j << 1) + (j & 1)) << 2) as u8;
        maxi = 960 / opus_packet_get_samples_per_frame(packet[0], 8000);
        i = 1;
        while i <= maxi {
            let mut maxp: i32 = 0;
            packet[0] = (((j << 1) + (j & 1)) << 2) as u8;
            if i > 1 {
                let fresh0 = &mut (packet[0]);
                *fresh0 = (*fresh0 as i32 + if i == 2 { 1 } else { 3 }) as u8;
            }
            packet[1] = (if i > 2 { i } else { 0 }) as u8;
            maxp = 960 / (i * opus_packet_get_samples_per_frame(packet[0], 8000));
            k = 0;
            while k <= 1275 + 75 {
                let mut cnt: i32 = 0;
                let mut rcnt: i32 = 0;
                if k % i == 0 {
                    cnt = 0;
                    while cnt < maxp + 2 {
                        if cnt > 0 {
                            let len = k + (if i > 2 { 2 } else { 1 });
                            ret = opus_repacketizer_cat(rp, &packet[..len as usize], len);
                            if if cnt <= maxp && k <= 1275 * i {
                                (ret != 0) as i32
                            } else {
                                (ret != OPUS_INVALID_PACKET) as i32
                            } != 0
                            {
                                fail("tests/test_opus_api.c", 1542);
                            }
                            cfgs += 1;
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
                        if opus_repacketizer_get_nb_frames(rp) != rcnt * i {
                            fail("tests/test_opus_api.c", 1546);
                        }
                        cfgs += 1;
                        ret = opus_repacketizer_out_range(
                            rp,
                            0,
                            rcnt * i,
                            po,
                            1276 * 48 + 48 * 2 + 2,
                        );
                        if rcnt > 0 {
                            let mut len: i32 = 0;
                            len = k * rcnt + (if rcnt * i > 2 { 2 } else { 1 });
                            if ret != len {
                                fail("tests/test_opus_api.c", 1553);
                            }
                            if rcnt * i < 2 && *po.offset(0 as isize) as i32 & 3 != 0 {
                                fail("tests/test_opus_api.c", 1554);
                            }
                            if rcnt * i == 2 && *po.offset(0 as isize) as i32 & 3 != 1 {
                                fail("tests/test_opus_api.c", 1555);
                            }
                            if rcnt * i > 2
                                && (*po.offset(0 as isize) as i32 & 3 != 3
                                    || *po.offset(1 as isize) as i32 != rcnt * i)
                            {
                                fail("tests/test_opus_api.c", 1556);
                            }
                            cfgs += 1;
                            if opus_repacketizer_out(rp, po, len) != len {
                                fail("tests/test_opus_api.c", 1558);
                            }
                            cfgs += 1;
                            if opus_packet_unpad(po, len) != len {
                                fail("tests/test_opus_api.c", 1560);
                            }
                            cfgs += 1;
                            if opus_packet_pad(po, len, len + 1) != 0 {
                                fail("tests/test_opus_api.c", 1562);
                            }
                            cfgs += 1;
                            if opus_packet_pad(po, len + 1, len + 256) != 0 {
                                fail("tests/test_opus_api.c", 1564);
                            }
                            cfgs += 1;
                            if opus_packet_unpad(po, len + 256) != len {
                                fail("tests/test_opus_api.c", 1566);
                            }
                            cfgs += 1;

                            if opus_repacketizer_out(rp, po, len - 1) != -(2) {
                                fail("tests/test_opus_api.c", 1576);
                            }
                            cfgs += 1;
                            if len > 1 {
                                if opus_repacketizer_out(rp, po, 1) != -(2) {
                                    fail("tests/test_opus_api.c", 1580);
                                }
                                cfgs += 1;
                            }
                            if opus_repacketizer_out(rp, po, 0) != -(2) {
                                fail("tests/test_opus_api.c", 1583);
                            }
                            cfgs += 1;
                        } else if ret != OPUS_BAD_ARG {
                            fail("tests/test_opus_api.c", 1585);
                        }
                        cnt += 1;
                    }
                    opus_repacketizer_init(rp);
                }
                k += 3;
            }
            i += 1;
        }
        j += 1;
    }
    opus_repacketizer_init(rp);
    packet[0] = 0;
    if opus_repacketizer_cat(rp, &packet[..5], 5) != 0 {
        fail("tests/test_opus_api.c", 1595);
    }
    cfgs += 1;
    let fresh1 = &mut (packet[0]);
    *fresh1 = (*fresh1 as i32 + 1) as u8;
    if opus_repacketizer_cat(rp, &packet[..9], 9) != 0 {
        fail("tests/test_opus_api.c", 1598);
    }
    cfgs += 1;
    i = opus_repacketizer_out(rp, po, 1276 * 48 + 48 * 2 + 2);
    if i != 4 + 8 + 2
        || *po.offset(0 as isize) as i32 & 3 != 3
        || *po.offset(1 as isize) as i32 & 63 != 3
        || *po.offset(1 as isize) as i32 >> 7 != 0
    {
        fail("tests/test_opus_api.c", 1601);
    }
    cfgs += 1;
    i = opus_repacketizer_out_range(rp, 0, 1, po, 1276 * 48 + 48 * 2 + 2);
    if i != 5 || *po.offset(0 as isize) as i32 & 3 != 0 {
        fail("tests/test_opus_api.c", 1604);
    }
    cfgs += 1;
    i = opus_repacketizer_out_range(rp, 1, 2, po, 1276 * 48 + 48 * 2 + 2);
    if i != 5 || *po.offset(0 as isize) as i32 & 3 != 0 {
        fail("tests/test_opus_api.c", 1607);
    }
    cfgs += 1;
    opus_repacketizer_init(rp);
    packet[0] = 1;
    if opus_repacketizer_cat(rp, &packet[..9], 9) != 0 {
        fail("tests/test_opus_api.c", 1613);
    }
    cfgs += 1;
    packet[0] = 0;
    if opus_repacketizer_cat(rp, &packet[..3], 3) != 0 {
        fail("tests/test_opus_api.c", 1616);
    }
    cfgs += 1;
    i = opus_repacketizer_out(rp, po, 1276 * 48 + 48 * 2 + 2);
    if i != 2 + 8 + 2 + 2
        || *po.offset(0 as isize) as i32 & 3 != 3
        || *po.offset(1 as isize) as i32 & 63 != 3
        || *po.offset(1 as isize) as i32 >> 7 != 1
    {
        fail("tests/test_opus_api.c", 1619);
    }
    cfgs += 1;
    opus_repacketizer_init(rp);
    packet[0] = 2;
    packet[1] = 4;
    if opus_repacketizer_cat(rp, &packet[..8], 8) != 0 {
        fail("tests/test_opus_api.c", 1626);
    }
    cfgs += 1;
    if opus_repacketizer_cat(rp, &packet[..8], 8) != 0 {
        fail("tests/test_opus_api.c", 1628);
    }
    cfgs += 1;
    i = opus_repacketizer_out(rp, po, 1276 * 48 + 48 * 2 + 2);
    if i != 2 + 1 + 1 + 1 + 4 + 2 + 4 + 2
        || *po.offset(0 as isize) as i32 & 3 != 3
        || *po.offset(1 as isize) as i32 & 63 != 4
        || *po.offset(1 as isize) as i32 >> 7 != 1
    {
        fail("tests/test_opus_api.c", 1631);
    }
    cfgs += 1;
    opus_repacketizer_init(rp);
    packet[0] = 2;
    packet[1] = 4;
    if opus_repacketizer_cat(rp, &packet[..10], 10) != 0 {
        fail("tests/test_opus_api.c", 1638);
    }
    cfgs += 1;
    if opus_repacketizer_cat(rp, &packet[..10], 10) != 0 {
        fail("tests/test_opus_api.c", 1640);
    }
    cfgs += 1;
    i = opus_repacketizer_out(rp, po, 1276 * 48 + 48 * 2 + 2);
    if i != 2 + 4 + 4 + 4 + 4
        || *po.offset(0 as isize) as i32 & 3 != 3
        || *po.offset(1 as isize) as i32 & 63 != 4
        || *po.offset(1 as isize) as i32 >> 7 != 0
    {
        fail("tests/test_opus_api.c", 1643);
    }
    cfgs += 1;
    j = 0;
    while j < 32 {
        let mut maxi_0: i32 = 0;
        let mut sum: i32 = 0;
        let mut rcnt_0: i32 = 0;
        packet[0] = (((j << 1) + (j & 1)) << 2) as u8;
        maxi_0 = 960 / opus_packet_get_samples_per_frame(packet[0], 8000);
        sum = 0;
        rcnt_0 = 0;
        opus_repacketizer_init(rp);
        i = 1;
        while i <= maxi_0 + 2 {
            let mut len_0: i32 = 0;
            ret = opus_repacketizer_cat(rp, &packet[..i as usize], i);
            if rcnt_0 < maxi_0 {
                if ret != 0 {
                    fail("tests/test_opus_api.c", 1662);
                }
                rcnt_0 += 1;
                sum += i - 1;
            } else if ret != OPUS_INVALID_PACKET {
                fail("tests/test_opus_api.c", 1665);
            }
            cfgs += 1;
            len_0 = sum
                + (if rcnt_0 < 2 {
                    1
                } else if rcnt_0 < 3 {
                    2
                } else {
                    2 + rcnt_0 - 1
                });
            if opus_repacketizer_out(rp, po, 1276 * 48 + 48 * 2 + 2) != len_0 {
                fail("tests/test_opus_api.c", 1668);
            }
            if rcnt_0 > 2 && *po.offset(1 as isize) as i32 & 63 != rcnt_0 {
                fail("tests/test_opus_api.c", 1669);
            }
            if rcnt_0 == 2 && *po.offset(0 as isize) as i32 & 3 != 2 {
                fail("tests/test_opus_api.c", 1670);
            }
            if rcnt_0 == 1 && *po.offset(0 as isize) as i32 & 3 != 0 {
                fail("tests/test_opus_api.c", 1671);
            }
            cfgs += 1;
            if opus_repacketizer_out(rp, po, len_0) != len_0 {
                fail("tests/test_opus_api.c", 1673);
            }
            cfgs += 1;
            if opus_packet_unpad(po, len_0) != len_0 {
                fail("tests/test_opus_api.c", 1675);
            }
            cfgs += 1;
            if opus_packet_pad(po, len_0, len_0 + 1) != 0 {
                fail("tests/test_opus_api.c", 1677);
            }
            cfgs += 1;
            if opus_packet_pad(po, len_0 + 1, len_0 + 256) != 0 {
                fail("tests/test_opus_api.c", 1679);
            }
            cfgs += 1;
            if opus_packet_unpad(po, len_0 + 256) != len_0 {
                fail("tests/test_opus_api.c", 1681);
            }
            cfgs += 1;
            if opus_repacketizer_out(rp, po, len_0 - 1) != -(2) {
                fail("tests/test_opus_api.c", 1691);
            }
            cfgs += 1;
            if len_0 > 1 {
                if opus_repacketizer_out(rp, po, 1) != -(2) {
                    fail("tests/test_opus_api.c", 1695);
                }
                cfgs += 1;
            }
            if opus_repacketizer_out(rp, po, 0) != -(2) {
                fail("tests/test_opus_api.c", 1698);
            }
            cfgs += 1;
            i += 1;
        }
        j += 1;
    }
    *po.offset(0 as isize) = 'O' as i32 as u8;
    *po.offset(1 as isize) = 'p' as i32 as u8;
    if opus_packet_pad(po, 4, 4) != 0 {
        fail("tests/test_opus_api.c", 1705);
    }
    cfgs += 1;
    if opus_packet_pad(po, 4, 5) != OPUS_INVALID_PACKET {
        fail("tests/test_opus_api.c", 1709);
    }
    cfgs += 1;
    if opus_packet_pad(po, 0, 5) != OPUS_BAD_ARG {
        fail("tests/test_opus_api.c", 1713);
    }
    cfgs += 1;
    if opus_packet_unpad(po, 0) != OPUS_BAD_ARG {
        fail("tests/test_opus_api.c", 1717);
    }
    cfgs += 1;
    if opus_packet_unpad(po, 4) != OPUS_INVALID_PACKET {
        fail("tests/test_opus_api.c", 1721);
    }
    cfgs += 1;
    *po.offset(0 as isize) = 0;
    *po.offset(1 as isize) = 0;
    *po.offset(2 as isize) = 0;
    if opus_packet_pad(po, 5, 4) != OPUS_BAD_ARG {
        fail("tests/test_opus_api.c", 1728);
    }
    cfgs += 1;
    println!("    opus_repacketizer_cat ........................ OK.");
    println!("    opus_repacketizer_out ........................ OK.");
    println!("    opus_repacketizer_out_range .................. OK.");
    println!("    opus_packet_pad .............................. OK.");
    println!("    opus_packet_unpad ............................ OK.");
    opus_repacketizer_destroy(rp);
    cfgs += 1;
    free(po as *mut core::ffi::c_void);
    println!("                        All repacketizer tests passed");
    println!("                   ({:7} API invocations)", cfgs);
}
