#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(non_upper_case_globals)]
#![allow(deprecated)]

pub mod test_opus_common_h {
    #[inline]
    pub unsafe fn deb2_impl(
        mut _t: *mut u8,
        mut _p: *mut *mut u8,
        mut _k: i32,
        mut _x: i32,
        mut _y: i32,
    ) {
        let mut i: i32;
        if _x > 2 {
            if _y < 3 {
                i = 0;
                while i < _y {
                    *_p = (*_p).offset(-1);
                    **_p = *_t.offset((i + 1) as isize);
                    i += 1;
                }
            }
        } else {
            *_t.offset(_x as isize) = *_t.offset((_x - _y) as isize);
            deb2_impl(_t, _p, _k, _x + 1, _y);
            i = *_t.offset((_x - _y) as isize) as i32 + 1;
            while i < _k {
                *_t.offset(_x as isize) = i as u8;
                deb2_impl(_t, _p, _k, _x + 1, _x);
                i += 1;
            }
        };
    }
    #[inline]
    pub unsafe fn debruijn2(mut _k: i32, mut _res: *mut u8) {
        let t = malloc(
            (::core::mem::size_of::<u8>() as u64)
                .wrapping_mul(_k as u64)
                .wrapping_mul(2),
        ) as *mut u8;
        memset(
            t as *mut core::ffi::c_void,
            0,
            (::core::mem::size_of::<u8>() as u64)
                .wrapping_mul(_k as u64)
                .wrapping_mul(2),
        );
        let mut p = &mut *_res.offset((_k * _k) as isize) as *mut u8;
        deb2_impl(t, &mut p, _k, 1, 1);
        free(t as *mut core::ffi::c_void);
    }
    pub static mut Rz: u32 = 0;
    pub static mut Rw: u32 = 0;
    #[inline]
    pub unsafe fn fast_rand() -> u32 {
        Rz = 36969_u32.wrapping_mul(Rz & 65535).wrapping_add(Rz >> 16);
        Rw = 18000_u32.wrapping_mul(Rw & 65535).wrapping_add(Rw >> 16);
        (Rz << 16).wrapping_add(Rw)
    }
    static iseed: AtomicU32 = AtomicU32::new(0);

    pub fn get_iseed() -> u32 {
        iseed.load(std::sync::atomic::Ordering::Relaxed)
    }

    pub fn set_iseed(val: u32) {
        iseed.store(val, std::sync::atomic::Ordering::Relaxed)
    }

    pub fn fail(file: &str, line: usize) {
        panic!("test failed, original: {file} at line {line}");
    }

    use std::sync::atomic::AtomicU32;

    use unsafe_libopus::externs::memset;
    use unsafe_libopus::externs::{free, malloc};
}

use crate::test_opus_common_h::set_iseed;

pub use self::test_opus_common_h::{debruijn2, fail, fast_rand, get_iseed, Rw, Rz};
use unsafe_libopus::externs::memcpy;
use unsafe_libopus::externs::{free, malloc};
use unsafe_libopus::{
    opus_decode, opus_decoder_create, opus_decoder_ctl, opus_decoder_destroy,
    opus_decoder_get_size, opus_encode, opus_encoder_ctl, opus_get_version_string, opus_packet_pad,
    opus_packet_parse, opus_packet_unpad, OpusDecoder, OpusEncoder,
};

mod opus_encode_regressions;
use opus_encode_regressions::regression_test;

pub unsafe fn generate_music(buf: *mut i16, len: i32) {
    let mut b2 = 0;
    let mut a2 = b2;
    let mut b1 = a2;
    let mut a1 = b1;
    let mut d2 = 0;
    let mut d1 = d2;
    let mut c2 = d1;
    let mut c1 = c2;
    let mut j = 0;
    let mut i = 0;
    while i < 2880 {
        let fresh0 = &mut (*buf.offset((i * 2 + 1) as isize));
        *fresh0 = 0 as i16;
        *buf.offset((i * 2) as isize) = *fresh0;
        i += 1;
    }
    i = 2880;
    while i < len {
        let mut v2 = (((j * (j >> 12 ^ (j >> 10 | j >> 12) & 26 & j >> 7)) & 128) + 128) << 15;
        let mut v1 = v2;
        let mut r = fast_rand();
        v1 = (v1 as u32).wrapping_add(r & 65535) as i32 as i32;
        v1 = (v1 as u32).wrapping_sub(r >> 16) as i32 as i32;
        r = fast_rand();
        v2 = (v2 as u32).wrapping_add(r & 65535) as i32 as i32;
        v2 = (v2 as u32).wrapping_sub(r >> 16) as i32 as i32;
        b1 = v1 - a1 + ((b1 * 61 + 32) >> 6);
        a1 = v1;
        b2 = v2 - a2 + ((b2 * 61 + 32) >> 6);
        a2 = v2;
        c1 = (30 * (c1 + b1 + d1) + 32) >> 6;
        d1 = b1;
        c2 = (30 * (c2 + b2 + d2) + 32) >> 6;
        d2 = b2;
        v1 = (c1 + 128) >> 8;
        v2 = (c2 + 128) >> 8;
        *buf.offset((i * 2) as isize) = (if v1 > 32767 {
            32767
        } else if v1 < -(32768) {
            -(32768)
        } else {
            v1
        }) as i16;
        *buf.offset((i * 2 + 1) as isize) = (if v2 > 32767 {
            32767
        } else if v2 < -(32768) {
            -(32768)
        } else {
            v2
        }) as i16;
        if i % 6 == 0 {
            j += 1;
        }
        i += 1;
    }
}

pub unsafe fn get_frame_size_enum(frame_size: i32, sampling_rate: i32) -> i32 {
    if frame_size == sampling_rate / 400 {
        5001
    } else if frame_size == sampling_rate / 200 {
        5002
    } else if frame_size == sampling_rate / 100 {
        5003
    } else if frame_size == sampling_rate / 50 {
        5004
    } else if frame_size == sampling_rate / 25 {
        5005
    } else if frame_size == 3 * sampling_rate / 50 {
        5006
    } else if frame_size == 4 * sampling_rate / 50 {
        5007
    } else if frame_size == 5 * sampling_rate / 50 {
        5008
    } else if frame_size == 6 * sampling_rate / 50 {
        5009
    } else {
        fail("tests/test_opus_encode.c", 138);
        0
    }
}

pub unsafe fn test_encode(
    enc: &mut OpusEncoder,
    channels: i32,
    frame_size: i32,
    dec: *mut OpusDecoder,
) -> i32 {
    let mut samp_count: i32 = 0;
    let mut packet: [u8; 1757] = [0; 1757];
    let mut len: i32;
    let mut out_samples: i32;
    let mut ret: i32 = 0;
    let mut inbuf = vec![0; 48000 * 30 / 3];
    generate_music(inbuf.as_mut_ptr(), 48000 * 30 / 3 / 2);
    let outbuf = malloc(
        (::core::mem::size_of::<i16>() as u64)
            .wrapping_mul(5760)
            .wrapping_mul(3),
    ) as *mut i16;
    loop {
        len = opus_encode(
            enc,
            &inbuf[(samp_count * channels) as usize..],
            frame_size,
            &mut packet,
            1500,
        );
        if len < 0 || len > 1500 {
            eprintln!("opus_encode() returned {}", len);
            ret = -1;
            break;
        } else {
            out_samples = opus_decode(&mut *dec, &packet[..len as _], outbuf, 5760, 0);
            if out_samples != frame_size {
                eprintln!("opus_decode() returned {}", out_samples);
                ret = -1;
                break;
            } else {
                samp_count += frame_size;
                if samp_count >= 48000 * 30 / 3 / 2 - 5760 {
                    break;
                }
            }
        }
    }
    free(outbuf as *mut core::ffi::c_void);
    ret
}
pub unsafe fn fuzz_encoder_settings(num_encoders: i32, num_setting_changes: i32) {
    let mut dec: *mut OpusDecoder;
    let mut err: i32 = 0;
    let sampling_rates: [i32; 5] = [8000, 12000, 16000, 24000, 48000];
    let channels: [i32; 2] = [1, 2];
    let applications: [i32; 3] = [2049, 2048, 2051];
    let bitrates: [i32; 11] = [
        6000,
        12000,
        16000,
        24000,
        32000,
        48000,
        64000,
        96000,
        510000,
        -(1000),
        -1,
    ];
    let force_channels: [i32; 4] = [-(1000), -(1000), 1, 2];
    let use_vbr: [i32; 3] = [0, 1, 1];
    let vbr_constraints: [i32; 3] = [0, 1, 1];
    let complexities: [i32; 11] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    let max_bandwidths: [i32; 6] = [1101, 1102, 1103, 1104, 1105, 1105];
    let signals: [i32; 4] = [-(1000), -(1000), 3001, 3002];
    let inband_fecs: [i32; 3] = [0, 0, 1];
    let packet_loss_perc: [i32; 4] = [0, 1, 2, 5];
    let lsb_depths: [i32; 2] = [8, 24];
    let prediction_disabled: [i32; 3] = [0, 0, 1];
    let use_dtx: [i32; 2] = [0, 1];
    let frame_sizes_ms_x2: [i32; 9] = [5, 10, 20, 40, 80, 120, 160, 200, 240];
    let mut i = 0;
    while i < num_encoders {
        let sampling_rate: i32 = sampling_rates[(fast_rand() as u64)
            .wrapping_rem(::core::mem::size_of::<[i32; 5]>() as u64)
            .wrapping_div(::core::mem::size_of::<i32>() as u64)
            as usize];
        let num_channels: i32 = channels[(fast_rand() as u64)
            .wrapping_rem(::core::mem::size_of::<[i32; 2]>() as u64)
            .wrapping_div(::core::mem::size_of::<i32>() as u64)
            as usize];
        let application: i32 = applications[(fast_rand() as u64)
            .wrapping_rem(::core::mem::size_of::<[i32; 3]>() as u64)
            .wrapping_div(::core::mem::size_of::<i32>() as u64)
            as usize];
        dec = opus_decoder_create(sampling_rate, num_channels, &mut err);
        if err != 0 || dec.is_null() {
            fail("tests/test_opus_encode.c", 217);
        }

        println!("creating encoder {sampling_rate}, {num_channels}, {application}");
        let mut enc = OpusEncoder::new(sampling_rate, num_channels, application).unwrap();

        let mut j = 0;
        while j < num_setting_changes {
            let bitrate: i32 = bitrates[(fast_rand() as u64)
                .wrapping_rem(::core::mem::size_of::<[i32; 11]>() as u64)
                .wrapping_div(::core::mem::size_of::<i32>() as u64)
                as usize];
            let mut force_channel: i32 = force_channels[(fast_rand() as u64)
                .wrapping_rem(::core::mem::size_of::<[i32; 4]>() as u64)
                .wrapping_div(::core::mem::size_of::<i32>() as u64)
                as usize];
            let vbr: i32 = use_vbr[(fast_rand() as u64)
                .wrapping_rem(::core::mem::size_of::<[i32; 3]>() as u64)
                .wrapping_div(::core::mem::size_of::<i32>() as u64)
                as usize];
            let vbr_constraint: i32 = vbr_constraints[(fast_rand() as u64)
                .wrapping_rem(::core::mem::size_of::<[i32; 3]>() as u64)
                .wrapping_div(::core::mem::size_of::<i32>() as u64)
                as usize];
            let complexity: i32 = complexities[(fast_rand() as u64)
                .wrapping_rem(::core::mem::size_of::<[i32; 11]>() as u64)
                .wrapping_div(::core::mem::size_of::<i32>() as u64)
                as usize];
            let max_bw: i32 = max_bandwidths[(fast_rand() as u64)
                .wrapping_rem(::core::mem::size_of::<[i32; 6]>() as u64)
                .wrapping_div(::core::mem::size_of::<i32>() as u64)
                as usize];
            let sig: i32 = signals[(fast_rand() as u64)
                .wrapping_rem(::core::mem::size_of::<[i32; 4]>() as u64)
                .wrapping_div(::core::mem::size_of::<i32>() as u64)
                as usize];
            let inband_fec: i32 = inband_fecs[(fast_rand() as u64)
                .wrapping_rem(::core::mem::size_of::<[i32; 3]>() as u64)
                .wrapping_div(::core::mem::size_of::<i32>() as u64)
                as usize];
            let pkt_loss: i32 = packet_loss_perc[(fast_rand() as u64)
                .wrapping_rem(::core::mem::size_of::<[i32; 4]>() as u64)
                .wrapping_div(::core::mem::size_of::<i32>() as u64)
                as usize];
            let lsb_depth: i32 = lsb_depths[(fast_rand() as u64)
                .wrapping_rem(::core::mem::size_of::<[i32; 2]>() as u64)
                .wrapping_div(::core::mem::size_of::<i32>() as u64)
                as usize];
            let pred_disabled: i32 = prediction_disabled[(fast_rand() as u64)
                .wrapping_rem(::core::mem::size_of::<[i32; 3]>() as u64)
                .wrapping_div(::core::mem::size_of::<i32>() as u64)
                as usize];
            let dtx: i32 = use_dtx[(fast_rand() as u64)
                .wrapping_rem(::core::mem::size_of::<[i32; 2]>() as u64)
                .wrapping_div(::core::mem::size_of::<i32>() as u64)
                as usize];
            let frame_size_ms_x2: i32 = frame_sizes_ms_x2[(fast_rand() as u64)
                .wrapping_rem(::core::mem::size_of::<[i32; 9]>() as u64)
                .wrapping_div(::core::mem::size_of::<i32>() as u64)
                as usize];
            let frame_size: i32 = frame_size_ms_x2 * sampling_rate / 2000;
            let frame_size_enum: i32 = get_frame_size_enum(frame_size, sampling_rate);
            force_channel = if force_channel < num_channels {
                force_channel
            } else {
                num_channels
            };
            if opus_encoder_ctl!(&mut enc, 4002, bitrate) != 0 {
                fail("tests/test_opus_encode.c", 240);
            }
            if opus_encoder_ctl!(&mut enc, 4022, force_channel) != 0 {
                fail("tests/test_opus_encode.c", 241);
            }
            if opus_encoder_ctl!(&mut enc, 4006, vbr) != 0 {
                fail("tests/test_opus_encode.c", 242);
            }
            if opus_encoder_ctl!(&mut enc, 4020, vbr_constraint) != 0 {
                fail("tests/test_opus_encode.c", 243);
            }
            if opus_encoder_ctl!(&mut enc, 4010, complexity) != 0 {
                fail("tests/test_opus_encode.c", 244);
            }
            if opus_encoder_ctl!(&mut enc, 4004, max_bw) != 0 {
                fail("tests/test_opus_encode.c", 245);
            }
            if opus_encoder_ctl!(&mut enc, 4024, sig) != 0 {
                fail("tests/test_opus_encode.c", 246);
            }
            if opus_encoder_ctl!(&mut enc, 4012, inband_fec) != 0 {
                fail("tests/test_opus_encode.c", 247);
            }
            if opus_encoder_ctl!(&mut enc, 4014, pkt_loss) != 0 {
                fail("tests/test_opus_encode.c", 248);
            }
            if opus_encoder_ctl!(&mut enc, 4036, lsb_depth) != 0 {
                fail("tests/test_opus_encode.c", 249);
            }
            if opus_encoder_ctl!(&mut enc, 4042, pred_disabled) != 0 {
                fail("tests/test_opus_encode.c", 250);
            }
            if opus_encoder_ctl!(&mut enc, 4016, dtx) != 0 {
                fail("tests/test_opus_encode.c", 251);
            }
            if opus_encoder_ctl!(&mut enc, 4040, frame_size_enum) != 0 {
                fail("tests/test_opus_encode.c", 252);
            }
            if test_encode(&mut enc, num_channels, frame_size, dec) != 0 {
                eprintln!(
                    "fuzz_encoder_settings: {} kHz, {} ch, application: {}, {} bps, force ch: {}, vbr: {}, vbr constraint: {}, complexity: {}, max bw: {}, signal: {}, inband fec: {}, pkt loss: {}%, lsb depth: {}, pred disabled: {}, dtx: {}, ({}/2) ms",
                    sampling_rate / 1000,
                    num_channels,
                    application,
                    bitrate,
                    force_channel,
                    vbr,
                    vbr_constraint,
                    complexity,
                    max_bw,
                    sig,
                    inband_fec,
                    pkt_loss,
                    lsb_depth,
                    pred_disabled,
                    dtx,
                    frame_size_ms_x2,
                );
                fail("tests/test_opus_encode.c", 263);
            }
            j += 1;
        }
        opus_decoder_destroy(dec);
        i += 1;
    }
}
pub unsafe fn run_test1(no_fuzz: bool) -> i32 {
    static mut fsizes: [i32; 6] = [960 * 3, 960 * 2, 120, 240, 480, 960];
    static mut mstrings: [&str; 3] = ["    LP", "Hybrid", "  MDCT"];
    let mut db62: [u8; 36] = [0; 36];
    let mut i: i32 = 0;
    let mut j: i32 = 0;
    let mut rc: i32;
    let mut err: i32 = 0;
    let dec: *mut OpusDecoder;
    let mut dec_err: [*mut OpusDecoder; 10] = [std::ptr::null_mut::<OpusDecoder>(); 10];
    let outbuf: *mut i16;
    let out2buf: *mut i16;
    let mut bitrate_bps: i32;
    let mut packet: [u8; 1757] = [0; 1757];
    let mut enc_final_range: u32 = 0;
    let mut dec_final_range: u32 = 0;
    let mut fswitch: i32;
    let mut fsize: i32;
    let mut count: i32;
    println!("  Encode+Decode tests.");
    let mut enc = OpusEncoder::new(48000, 2, 2048).unwrap();
    if i != j {
        fail("tests/test_opus_encode.c", 339);
    }
    dec = opus_decoder_create(48000, 2, &mut err);
    if err != 0 || dec.is_null() {
        fail("tests/test_opus_encode.c", 344);
    }
    dec_err[0 as usize] = malloc(opus_decoder_get_size(2) as u64) as *mut OpusDecoder;
    memcpy(
        dec_err[0 as usize] as *mut core::ffi::c_void,
        dec as *const core::ffi::c_void,
        opus_decoder_get_size(2) as u64,
    );
    dec_err[1 as usize] = opus_decoder_create(48000, 1, &mut err);
    dec_err[2 as usize] = opus_decoder_create(24000, 2, &mut err);
    dec_err[3 as usize] = opus_decoder_create(24000, 1, &mut err);
    dec_err[4 as usize] = opus_decoder_create(16000, 2, &mut err);
    dec_err[5 as usize] = opus_decoder_create(16000, 1, &mut err);
    dec_err[6 as usize] = opus_decoder_create(12000, 2, &mut err);
    dec_err[7 as usize] = opus_decoder_create(12000, 1, &mut err);
    dec_err[8 as usize] = opus_decoder_create(8000, 2, &mut err);
    dec_err[9 as usize] = opus_decoder_create(8000, 1, &mut err);
    i = 0;
    while i < 10 {
        if (dec_err[i as usize]).is_null() {
            fail("tests/test_opus_encode.c", 363);
        }
        i += 1;
    }

    let mut inbuf = vec![0; (48000 * 30) * 2];
    outbuf = malloc(
        (::core::mem::size_of::<i16>() as u64)
            .wrapping_mul((48000 * 30) as u64)
            .wrapping_mul(2),
    ) as *mut i16;
    out2buf = malloc(
        (::core::mem::size_of::<i16>() as u64)
            .wrapping_mul(5760)
            .wrapping_mul(3),
    ) as *mut i16;
    if outbuf.is_null() || out2buf.is_null() {
        fail("tests/test_opus_encode.c", 378);
    }
    generate_music(inbuf.as_mut_ptr(), 48000 * 30);
    if opus_encoder_ctl!(&mut enc, 4008, -(1000)) != 0 {
        fail("tests/test_opus_encode.c", 387);
    }
    if opus_encoder_ctl!(&mut enc, 11002, -(2)) != -1 {
        fail("tests/test_opus_encode.c", 388);
    }
    if opus_encode(&mut enc, &inbuf, 500, &mut packet, 1500) != -1 {
        fail("tests/test_opus_encode.c", 389);
    }
    rc = 0;
    while rc < 3 {
        if opus_encoder_ctl!(&mut enc, 4006, (rc < 2) as i32,) != 0 {
            fail("tests/test_opus_encode.c", 393);
        }
        if opus_encoder_ctl!(&mut enc, 4020, (rc == 1) as i32,) != 0 {
            fail("tests/test_opus_encode.c", 394);
        }
        if opus_encoder_ctl!(&mut enc, 4020, (rc == 1) as i32,) != 0 {
            fail("tests/test_opus_encode.c", 395);
        }
        if opus_encoder_ctl!(&mut enc, 4012, (rc == 0) as i32,) != 0 {
            fail("tests/test_opus_encode.c", 396);
        }
        for j in 0..13 {
            let rate: i32;
            let modes: [i32; 13] = [0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2];
            let rates: [i32; 13] = [
                6000, 12000, 48000, 16000, 32000, 48000, 64000, 512000, 13000, 24000, 48000, 64000,
                96000,
            ];
            let frame: [i32; 13] = [
                960 * 2,
                960,
                480,
                960,
                960,
                960,
                480,
                960 * 3,
                960 * 3,
                960,
                480,
                240,
                120,
            ];
            rate = (rates[j as usize] as u32)
                .wrapping_add((fast_rand()).wrapping_rem(rates[j as usize] as u32))
                as i32;
            i = 0;
            count = i;
            loop {
                let mut bw: i32;
                let mut len: i32;
                let mut out_samples: i32;
                let frame_size = frame[j as usize];

                if fast_rand() & 255 == 0 {
                    if opus_encoder_ctl!(&mut enc, 4028) != 0 {
                        fail("tests/test_opus_encode.c", 410);
                    }
                    if opus_decoder_ctl!(&mut *dec, 4028) != 0 {
                        fail("tests/test_opus_encode.c", 411);
                    }
                    if fast_rand() & 1 != 0
                        && opus_decoder_ctl!(&mut *dec_err[(fast_rand() & 1) as usize], 4028,) != 0
                    {
                        fail("tests/test_opus_encode.c", 414);
                    }
                }
                if fast_rand() & 127 == 0
                    && opus_decoder_ctl!(&mut *dec_err[(fast_rand() & 1) as usize], 4028,) != 0
                {
                    fail("tests/test_opus_encode.c", 419);
                }
                if (fast_rand()).wrapping_rem(10) == 0 {
                    let complex: i32 = (fast_rand()).wrapping_rem(11) as i32;
                    if opus_encoder_ctl!(&mut enc, 4010, complex) != 0 {
                        fail("tests/test_opus_encode.c", 423);
                    }
                }
                if (fast_rand()).wrapping_rem(50) == 0 {
                    opus_decoder_ctl!(&mut *dec, 4028);
                }
                if opus_encoder_ctl!(&mut enc, 4012, (rc == 0) as i32,) != 0 {
                    fail("tests/test_opus_encode.c", 426);
                }
                if opus_encoder_ctl!(&mut enc, 11002, 1000 + modes[j as usize],) != 0 {
                    fail("tests/test_opus_encode.c", 427);
                }
                fast_rand();
                if opus_encoder_ctl!(&mut enc, 4016, (fast_rand() & 1) as i32,) != 0 {
                    fail("tests/test_opus_encode.c", 428);
                }
                if opus_encoder_ctl!(&mut enc, 4002, rate) != 0 {
                    fail("tests/test_opus_encode.c", 429);
                }
                if rates[j as usize] >= 64000 {
                } else {
                };
                if opus_encoder_ctl!(
                    &mut enc,
                    4022,
                    if rates[j as usize] >= 64000 { 2 } else { 1 },
                ) != 0
                {
                    fail("tests/test_opus_encode.c", 430);
                }
                if opus_encoder_ctl!(&mut enc, 4010, (count >> 2) % 11,) != 0 {
                    fail("tests/test_opus_encode.c", 431);
                }
                fast_rand();
                fast_rand();
                if opus_encoder_ctl!(
                    &mut enc,
                    4014,
                    (fast_rand() & 15 & (fast_rand()).wrapping_rem(15)) as i32,
                ) != 0
                {
                    fail("tests/test_opus_encode.c", 432);
                }
                bw = (if modes[j as usize] == 0 {
                    1101_u32.wrapping_add((fast_rand()).wrapping_rem(3))
                } else if modes[j as usize] == 1 {
                    1104_u32.wrapping_add(fast_rand() & 1)
                } else {
                    1101_u32.wrapping_add((fast_rand()).wrapping_rem(5))
                }) as i32;
                if modes[j as usize] == 2 && bw == 1102 {
                    bw += 3;
                }
                if opus_encoder_ctl!(&mut enc, 4008, bw) != 0 {
                    fail("tests/test_opus_encode.c", 437);
                }
                len = opus_encode(
                    &mut enc,
                    &inbuf[(i << 1) as usize..],
                    frame_size,
                    &mut packet,
                    1500,
                );
                if len < 0 || len > 1500 {
                    fail("tests/test_opus_encode.c", 439);
                }
                if opus_encoder_ctl!(&mut enc, 4031, &mut enc_final_range) != 0 {
                    fail("tests/test_opus_encode.c", 440);
                }

                if fast_rand() & 3 == 0 {
                    let pad = 1;
                    if opus_packet_pad(&mut packet[..(len + pad) as usize], len, len + pad) != 0 {
                        fail("tests/test_opus_encode.c", 443);
                    }
                    len += pad;
                }
                if fast_rand() & 7 == 0 {
                    if opus_packet_pad(&mut packet[..(len + 256) as usize], len, len + 256) != 0 {
                        fail("tests/test_opus_encode.c", 448);
                    }
                    len += 256;
                }
                if fast_rand() & 3 == 0 {
                    len = opus_packet_unpad(&mut packet[..len as usize]);
                    if len < 1 {
                        fail("tests/test_opus_encode.c", 454);
                    }
                }

                out_samples = opus_decode(
                    &mut *dec,
                    &packet[..len as _],
                    &mut *outbuf.offset((i << 1) as isize),
                    5760,
                    0,
                );
                if out_samples != frame_size {
                    fail("tests/test_opus_encode.c", 457);
                }
                if opus_decoder_ctl!(&mut *dec, 4031, &mut dec_final_range) != 0 {
                    fail("tests/test_opus_encode.c", 458);
                }
                if enc_final_range != dec_final_range {
                    fail("tests/test_opus_encode.c", 459);
                }
                // LBRR decode
                out_samples = opus_decode(
                    &mut *dec_err[0],
                    &packet[..len as _],
                    out2buf,
                    frame_size,
                    (fast_rand() & 3 != 0) as i32,
                );
                if out_samples != frame_size {
                    fail("tests/test_opus_encode.c", 462);
                }
                let l = if fast_rand() & 3 == 0 { 0 } else { len };
                out_samples = opus_decode(
                    &mut *dec_err[1 as usize],
                    &packet[..l as _],
                    out2buf,
                    5760,
                    (fast_rand() & 7 != 0) as i32,
                );
                if out_samples < 120 {
                    fail("tests/test_opus_encode.c", 464);
                }
                i += frame_size;
                count += 1;
                if i >= 48000 * 30 / 3 - 5760 {
                    break;
                }
            }
            println!(
                "    Mode {} FB encode {}, {:6} bps OK.",
                mstrings[modes[j as usize] as usize],
                if rc == 0 {
                    " VBR"
                } else if rc == 1 {
                    "CVBR"
                } else {
                    " CBR"
                },
                rate,
            );
        }
        rc += 1;
    }
    if opus_encoder_ctl!(&mut enc, 11002, -(1000)) != 0 {
        fail("tests/test_opus_encode.c", 472);
    }
    if opus_encoder_ctl!(&mut enc, 4022, -(1000)) != 0 {
        fail("tests/test_opus_encode.c", 473);
    }
    if opus_encoder_ctl!(&mut enc, 4012, 0) != 0 {
        fail("tests/test_opus_encode.c", 474);
    }
    if opus_encoder_ctl!(&mut enc, 4016, 0) != 0 {
        fail("tests/test_opus_encode.c", 475);
    }
    bitrate_bps = 512000;
    fsize = (fast_rand()).wrapping_rem(31) as i32;
    fswitch = 100;
    debruijn2(6, db62.as_mut_ptr());
    i = 0;
    count = i;
    loop {
        let mut toc: u8 = 0;
        let mut frames = [0; 48];
        let mut size: [i16; 48] = [0; 48];
        let mut payload_offset: i32 = 0;
        let mut dec_final_range2: u32 = 0;
        let mut jj: i32;
        let dec2: i32;
        let mut len_1: i32;
        let mut out_samples_1: i32;
        let frame_size_1: i32 = fsizes[db62[fsize as usize] as usize];
        let offset: i32 = i % (48000 * 30 - 5760);
        opus_encoder_ctl!(&mut enc, 4002, bitrate_bps);
        len_1 = opus_encode(
            &mut enc,
            &inbuf[(offset << 1) as usize..],
            frame_size_1,
            &mut packet,
            1500,
        );
        if len_1 < 0 || len_1 > 1500 {
            fail("tests/test_opus_encode.c", 569);
        }
        count += 1;
        opus_encoder_ctl!(&mut enc, 4031, &mut enc_final_range);
        out_samples_1 = opus_decode(
            &mut *dec,
            &packet[..len_1 as _],
            &mut *outbuf.offset((offset << 1) as isize),
            5760,
            0,
        );
        if out_samples_1 != frame_size_1 {
            fail("tests/test_opus_encode.c", 575);
        }
        opus_decoder_ctl!(&mut *dec, 4031, &mut dec_final_range);
        if dec_final_range != enc_final_range {
            fail("tests/test_opus_encode.c", 580);
        }
        if opus_packet_parse(
            &packet[..len_1 as _],
            Some(&mut toc),
            Some(&mut frames),
            &mut size,
            Some(&mut payload_offset),
        ) <= 0
        {
            fail("tests/test_opus_encode.c", 585);
        }
        if fast_rand() & 1023 == 0 {
            len_1 = 0;
        }
        // j = (frames[0 as usize]).offset_from(packet.as_mut_ptr()) as i64 as i32;
        j = frames[0 as usize] as i32;
        while j < len_1 {
            jj = 0;
            while jj < 8 {
                packet[j as usize] = (packet[j as usize] as i32
                    ^ ((!no_fuzz && fast_rand() & 1023 == 0) as i32) << jj)
                    as u8;
                jj += 1;
            }
            j += 1;
        }
        out_samples_1 = opus_decode(
            &mut *dec_err[0 as usize],
            &packet[..len_1 as _],
            out2buf,
            5760,
            0,
        );
        if out_samples_1 < 0 || out_samples_1 > 5760 {
            fail("tests/test_opus_encode.c", 589);
        }
        if len_1 > 0 && out_samples_1 != frame_size_1 {
            fail("tests/test_opus_encode.c", 590);
        }
        opus_decoder_ctl!(&mut *dec_err[0 as usize], 4031, &mut dec_final_range);
        dec2 = (fast_rand()).wrapping_rem(9).wrapping_add(1) as i32;
        out_samples_1 = opus_decode(
            &mut *dec_err[dec2 as usize],
            &packet[..len_1 as _],
            out2buf,
            5760,
            0,
        );
        if out_samples_1 < 0 || out_samples_1 > 5760 {
            fail("tests/test_opus_encode.c", 597);
        }
        opus_decoder_ctl!(&mut *dec_err[dec2 as usize], 4031, &mut dec_final_range2);
        if len_1 > 0 && dec_final_range != dec_final_range2 {
            fail("tests/test_opus_encode.c", 600);
        }
        fswitch -= 1;
        if fswitch < 1 {
            fsize = (fsize + 1) % 36;
            let new_size = fsizes[db62[fsize as usize] as usize];
            if new_size == 960 || new_size == 480 {
                fswitch = ((2880 / new_size) as u32)
                    .wrapping_mul((fast_rand()).wrapping_rem(19).wrapping_add(1))
                    as i32;
            } else {
                fswitch = (fast_rand())
                    .wrapping_rem((2880 / new_size) as u32)
                    .wrapping_add(1) as i32;
            }
        }
        bitrate_bps = ((fast_rand())
            .wrapping_rem(508000)
            .wrapping_add(4000)
            .wrapping_add(bitrate_bps as u32)
            >> 1) as i32;
        i += frame_size_1;
        if i >= 48000 * 30 * 4 {
            break;
        }
    }
    println!(
        "    All framesize pairs switching encode, {} frames OK.",
        count
    );
    if opus_encoder_ctl!(&mut enc, 4028) != 0 {
        fail("tests/test_opus_encode.c", 616);
    }
    if opus_decoder_ctl!(&mut *dec, 4028) != 0 {
        fail("tests/test_opus_encode.c", 620);
    }
    opus_decoder_destroy(dec);
    i = 0;
    while i < 10 {
        opus_decoder_destroy(dec_err[i as usize]);
        i += 1;
    }
    free(outbuf as *mut core::ffi::c_void);
    free(out2buf as *mut core::ffi::c_void);
    0
}

pub fn print_usage(argv0: &str) {
    eprintln!(
        "Usage: {} [<seed>] [-fuzz <num_encoders> <num_settings_per_encoder>]",
        argv0,
    );
}

// make dummy arguments
// rust's test harness has its own arguments and will handle them itself
// not sure of the best way to pass arguments except modifying the code rn...
// provide a fixed seed 42
const DUMMY_ARGS: &[&str] = &["test_opus_encode", "42"];

unsafe fn main_0() -> i32 {
    let mut args = DUMMY_ARGS.into_iter().map(|v| v.to_string()); // std::env::args();
    let argv0 = args.next().unwrap();

    let seed = match args
        .next()
        .map(|v| v.parse().expect("Failed to parse seed from command line"))
    {
        Some(v) => {
            eprintln!("Using seed from (dummy) arguments: {}", v);
            v
        }
        None => match std::env::var("SEED")
            .ok()
            .as_ref()
            .map(|v| v.parse().expect("Failed to parse seed from environment"))
        {
            Some(v) => {
                eprintln!("Using seed from environment: {}", v);
                v
            }
            None => {
                let mut v = [0; 4];
                getrandom::getrandom(&mut v).unwrap();
                let v = u32::from_le_bytes(v);

                eprintln!("Using random seed: {}", v);
                v
            }
        },
    };
    set_iseed(seed);

    Rz = seed;
    Rw = Rz;

    let mut num_encoders_to_fuzz: i32 = 5;
    let mut num_setting_changes: i32 = 40;

    while let Some(arg) = args.next() {
        if arg == "-fuzz" {
            num_encoders_to_fuzz = match args.next().and_then(|v| v.parse().ok()) {
                Some(v) => v,
                None => {
                    print_usage(&argv0);
                    return 1;
                }
            };
            num_setting_changes = match args.next().and_then(|v| v.parse().ok()) {
                Some(v) => v,
                None => {
                    print_usage(&argv0);
                    return 1;
                }
            };
        } else {
            print_usage(&argv0);
            return 1;
        }
    }

    let oversion = opus_get_version_string();
    eprintln!(
        "Testing {} encoder. Random seed: {} ({:4X})",
        oversion,
        get_iseed(),
        fast_rand() % 65535
    );
    regression_test();
    run_test1(std::env::var("TEST_OPUS_NOFUZZ").is_ok());
    if std::env::var("TEST_OPUS_NOFUZZ").is_err() {
        eprintln!(
            "Running fuzz_encoder_settings with {} encoder(s) and {} setting change(s) each.",
            num_encoders_to_fuzz, num_setting_changes,
        );
        fuzz_encoder_settings(num_encoders_to_fuzz, num_setting_changes);
    }
    eprintln!("Tests completed successfully.");
    0
}

#[test]
fn test_opus_encode() {
    assert_eq!(unsafe { main_0() }, 0, "Test returned a non-zero exit code");
}
