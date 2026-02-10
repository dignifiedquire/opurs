//! Integration tests for the Opus encoder.
//!
//! Upstream C: tests/test_opus_encode.c
//!
//! Split into:
//! - 3 regression tests (independent, no RNG)
//! - 1 encode+decode test (chained RNG state from seed 42)
//! - 1 fuzz test (chained RNG state, skippable via TEST_OPUS_NOFUZZ)

mod test_common;

use test_common::{debruijn2, TestRng};
use unsafe_libopus::{
    opus_encoder_ctl_impl, opus_packet_pad, opus_packet_parse, opus_packet_unpad, opus_private,
    Bandwidth, Bitrate, Channels, FrameSize, OpusDecoder, OpusEncoder, Signal,
};

/// Fixed seed matching upstream test (argv[1] = 42).
const TEST_SEED: u32 = 42;

// ---------------------------------------------------------------------------
// Music generator (matches upstream generate_music)
// ---------------------------------------------------------------------------

fn generate_music(buf: &mut [i16], len: usize, rng: &mut TestRng) {
    let mut a1: i32 = 0;
    let mut b1: i32 = 0;
    let mut c1: i32 = 0;
    let mut d1: i32 = 0;
    let mut a2: i32 = 0;
    let mut b2: i32 = 0;
    let mut c2: i32 = 0;
    let mut d2: i32 = 0;
    let mut j: i32 = 0;

    // First 2880 samples are silence (stereo)
    for i in 0..(2880 * 2).min(buf.len()) {
        buf[i] = 0;
    }

    for i in 2880..len {
        let mut v1 = (((j * (j >> 12 ^ (j >> 10 | j >> 12) & 26 & j >> 7)) & 128) + 128) << 15;
        let mut v2 = v1;
        let mut r = rng.next_u32();
        v1 = (v1 as u32).wrapping_add(r & 65535) as i32;
        v1 = (v1 as u32).wrapping_sub(r >> 16) as i32;
        r = rng.next_u32();
        v2 = (v2 as u32).wrapping_add(r & 65535) as i32;
        v2 = (v2 as u32).wrapping_sub(r >> 16) as i32;
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
        buf[i * 2] = v1.clamp(-32768, 32767) as i16;
        buf[i * 2 + 1] = v2.clamp(-32768, 32767) as i16;
        if i % 6 == 0 {
            j += 1;
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn get_frame_size_enum(frame_size: i32, sampling_rate: i32) -> FrameSize {
    if frame_size == sampling_rate / 400 {
        FrameSize::Ms2_5
    } else if frame_size == sampling_rate / 200 {
        FrameSize::Ms5
    } else if frame_size == sampling_rate / 100 {
        FrameSize::Ms10
    } else if frame_size == sampling_rate / 50 {
        FrameSize::Ms20
    } else if frame_size == sampling_rate / 25 {
        FrameSize::Ms40
    } else if frame_size == 3 * sampling_rate / 50 {
        FrameSize::Ms60
    } else if frame_size == 4 * sampling_rate / 50 {
        FrameSize::Ms80
    } else if frame_size == 5 * sampling_rate / 50 {
        FrameSize::Ms100
    } else if frame_size == 6 * sampling_rate / 50 {
        FrameSize::Ms120
    } else {
        panic!("Invalid frame size {frame_size} for sample rate {sampling_rate}");
    }
}

fn test_encode(
    enc: &mut OpusEncoder,
    channels: i32,
    frame_size: i32,
    dec: &mut OpusDecoder,
    rng: &mut TestRng,
) -> i32 {
    let mut samp_count: i32 = 0;
    let mut packet = [0u8; 1757];
    let total_samples = 48000 * 30 / 3 / 2;
    let mut inbuf = vec![0i16; total_samples * 2];
    generate_music(&mut inbuf, total_samples, rng);
    let mut outbuf = vec![0i16; 5760 * 3];

    loop {
        let pcm_start = (samp_count * channels) as usize;
        let pcm_end = pcm_start + (frame_size * channels) as usize;
        let len = enc.encode(&inbuf[pcm_start..pcm_end], &mut packet[..1500]);
        if len < 0 || len > 1500 {
            eprintln!("opus_encode() returned {len}");
            return -1;
        }
        let out_samples = dec.decode(&packet[..len as usize], &mut outbuf, 5760, false);
        if out_samples != frame_size {
            eprintln!("opus_decode() returned {out_samples}");
            return -1;
        }
        samp_count += frame_size;
        if samp_count >= total_samples as i32 - 5760 {
            break;
        }
    }
    0
}

// ---------------------------------------------------------------------------
// Regression tests (independent, no RNG)
// ---------------------------------------------------------------------------

/// Upstream C: tests/opus_encode_regressions.c:ec_enc_shrink_assert
#[test]
fn test_regression_ec_enc_shrink_assert() {
    let mut data = [0u8; 2000];
    let mut pcm1 = [0i16; 960];
    pcm1[0] = 5140;

    let pcm2: [i16; 2880] = {
        let mut arr = [0i16; 2880];
        let vals: &[i16] = &[
            -256, -12033, 0, -2817, 6912, 0, -5359, 5200, 3061, 0, -2903, 5652, -1281, -24656,
            -14433, -24678, 32, -29793, 2870, 0, 4096, 5120, 5140, -234, -20230, -24673, -24633,
            -24673, -24705, 0, -32768, -25444, -25444, 0, -25444, -25444, 156, -20480, -7948,
            -5920, -7968, -7968, 224, 0, 20480, 11, 20496, 13, 20496, 11, -20480, 2292, -20240,
            244, 20480, 11, 20496, 11, -20480, 244, -20240, 7156, 20456, -246, -20243, 244, 128,
            244, 20480, 11, 20496, 11, -20480, 244, -20256, 244, 20480, 256, 0, -246, 16609, -176,
            0, 29872, -4096, -2888, 516, 2896, 4096, 2896, -20480, -3852, -2896, -1025, -31056,
            -14433, 244, 1792, -256, -12033, 0, -2817, 0, 0, -5359, 5200, 3061, 16, -2903, 5652,
            -1281, -24656, -14433, -24678, 32, -29793, 2870, 0, 4096, 5120, 5140, -234, -20230,
            -24673, -24633, -24673, -24705, 0, -32768, -25444, -25444, 0, -25444, -25444, 156,
            -20480, -7973, -5920, -7968, -7968, 224, 0, 20480, 11, 20496, 11, 20496, 11, -20480,
            2292, -20213, 244, 20480, 11, 20496, 11, -24698, -2873, 0, 7, -1, 208, -256, 244, 0,
            4352, 20715, -2796, 11, -22272, 5364, -234, -20230, -24673, -25913, 8351, -24832,
            13963, 11, 0, 16, 5140, 5652, -1281, -24656, -14433, -24673, 32671, 159, 0, -25472,
            -25444, 156, -25600, -25444, -25444, 0, -2896, -7968, -7960, -7968, -7968, 0, 0, 2896,
            4096, 2896, 4096, 2896, 0, -2896, -4088, -2896, 0, 2896, 0, -2896, -4096, -2896, 11,
            2640, -4609, -2896, -32768, -3072, 0, 2896, 4096, 2896, 0, -2896, -4096, -2896, 0, 80,
            1, 2816, 0, 20656, 255, -20480, 116, -18192,
        ];
        arr[..vals.len()].copy_from_slice(vals);
        arr
    };

    let mut enc = OpusEncoder::new(48000, 1, 2049).unwrap();
    enc.set_complexity(10).unwrap();
    enc.set_packet_loss_perc(6).unwrap();
    enc.set_bitrate(Bitrate::Bits(6000));
    let data_len = enc.encode(&pcm1, &mut data);
    assert!(data_len > 0, "ec_enc_shrink_assert: first encode failed");

    enc.set_signal(Some(Signal::Voice));
    enc.set_prediction_disabled(true);
    enc.set_bandwidth(Some(Bandwidth::Superwideband));
    enc.set_inband_fec(true);
    enc.set_bitrate(Bitrate::Bits(15600));
    let data_len = enc.encode(&pcm2, &mut data[..122]);
    assert!(data_len > 0, "ec_enc_shrink_assert: second encode failed");

    enc.set_signal(Some(Signal::Music));
    enc.set_bitrate(Bitrate::Bits(27000));
    let pcm3 = [0i16; 2880];
    let data_len = enc.encode(&pcm3, &mut data[..122]);
    assert!(data_len > 0, "ec_enc_shrink_assert: third encode failed");
}

/// Upstream C: tests/opus_encode_regressions.c:ec_enc_shrink_assert2
#[test]
fn test_regression_ec_enc_shrink_assert2() {
    let mut data = [0u8; 2000];

    let mut enc = OpusEncoder::new(48000, 1, 2049).unwrap();
    enc.set_complexity(6).unwrap();
    enc.set_signal(Some(Signal::Voice));
    enc.set_bandwidth(Some(Bandwidth::Fullband));
    enc.set_packet_loss_perc(26).unwrap();
    enc.set_bitrate(Bitrate::Bits(27000));

    let pcm = [0i16; 960];
    let data_len = enc.encode(&pcm, &mut data);
    assert!(data_len > 0, "ec_enc_shrink_assert2: first encode failed");

    enc.set_signal(Some(Signal::Music));
    let mut pcm_0 = [0i16; 480];
    // Specific test vector values
    pcm_0[0] = 32767;
    pcm_0[1] = 32767;
    pcm_0[4] = 32767;
    pcm_0[5] = 32767;
    pcm_0[8] = 32767;
    pcm_0[9] = 32767;
    pcm_0[10] = -32768;
    pcm_0[11] = -32768;
    pcm_0[14] = -32768;
    pcm_0[15] = -32768;
    pcm_0[18] = -32768;
    pcm_0[19] = -32768;

    let data_len = enc.encode(&pcm_0, &mut data[..19]);
    assert!(data_len > 0, "ec_enc_shrink_assert2: second encode failed");
}

/// Upstream C: tests/opus_encode_regressions.c:silk_gain_assert
#[test]
fn test_regression_silk_gain_assert() {
    let mut data = [0u8; 1000];
    let pcm1 = [0i16; 160];
    let mut pcm2 = [0i16; 960];
    pcm2[98] = 32767;
    pcm2[120] = 32767;

    let mut enc = OpusEncoder::new(8000, 1, 2049).unwrap();
    enc.set_complexity(3).unwrap();
    enc.set_max_bandwidth(Bandwidth::Narrowband);
    enc.set_bitrate(Bitrate::Bits(6000));
    let data_len = enc.encode(&pcm1, &mut data);
    assert!(data_len > 0, "silk_gain_assert: first encode failed");

    enc.set_vbr(false);
    enc.set_complexity(0).unwrap();
    enc.set_max_bandwidth(Bandwidth::Mediumband);
    enc.set_bitrate(Bitrate::Bits(2867));
    let data_len = enc.encode(&pcm2, &mut data);
    assert!(data_len > 0, "silk_gain_assert: second encode failed");
}

// ---------------------------------------------------------------------------
// Main encode+decode test with chained RNG
// ---------------------------------------------------------------------------

/// Upstream C: tests/test_opus_encode.c:run_test1 + fuzz_encoder_settings
///
/// This test combines run_test1() and fuzz_encoder_settings() because they
/// share chained RNG state from seed 42.
#[test]
fn test_opus_encode() {
    let mut rng = TestRng::from_iseed(TEST_SEED);
    // Consume one RNG value matching upstream: fast_rand() % 65535 in header print
    let _ = rng.next_u32() % 65535;

    run_test1(std::env::var("TEST_OPUS_NOFUZZ").is_ok(), &mut rng);

    if std::env::var("TEST_OPUS_NOFUZZ").is_err() {
        fuzz_encoder_settings(5, 40, &mut rng);
    }
}

fn run_test1(no_fuzz: bool, rng: &mut TestRng) {
    let fsizes: [i32; 6] = [960 * 3, 960 * 2, 120, 240, 480, 960];
    let mstrings: [&str; 3] = ["    LP", "Hybrid", "  MDCT"];
    let mut db62 = [0u8; 36];
    let mut packet = [0u8; 1757];
    let mut enc_final_range: u32;
    let mut dec_final_range: u32;

    let enc = OpusEncoder::new(48000, 2, 2048).unwrap();
    let mut dec = OpusDecoder::new(48000, 2).unwrap();

    // Create error decoders
    let mut dec_err: Vec<OpusDecoder> = Vec::with_capacity(10);

    // dec_err[0] = copy of dec
    dec_err.push(dec.clone());

    dec_err.push(OpusDecoder::new(48000, 1).unwrap());
    dec_err.push(OpusDecoder::new(24000, 2).unwrap());
    dec_err.push(OpusDecoder::new(24000, 1).unwrap());
    dec_err.push(OpusDecoder::new(16000, 2).unwrap());
    dec_err.push(OpusDecoder::new(16000, 1).unwrap());
    dec_err.push(OpusDecoder::new(12000, 2).unwrap());
    dec_err.push(OpusDecoder::new(12000, 1).unwrap());
    dec_err.push(OpusDecoder::new(8000, 2).unwrap());
    dec_err.push(OpusDecoder::new(8000, 1).unwrap());

    // Copy encoder and verify copy works
    let mut enc = {
        let enccpy = enc.clone();
        // Overwrite original to make sure we're using the copy
        // (matches upstream: memset(enc, 0xFF, ...) + destroy)
        let _ = enc;
        enccpy
    };

    // Generate input audio
    let total_samples = 48000 * 30;
    let mut inbuf = vec![0i16; total_samples * 2];
    let mut outbuf = vec![0i16; total_samples * 2];
    let mut out2buf = vec![0i16; 5760 * 3];
    generate_music(&mut inbuf, total_samples, rng);

    // Test invalid settings
    enc.set_bandwidth(None);
    // 11002 with -2 is an invalid force mode value; it should return -1 (BAD_ARG)
    assert_eq!(
        unsafe {
            opus_encoder_ctl_impl(
                &mut enc as *mut OpusEncoder,
                opus_private::OPUS_SET_FORCE_MODE_REQUEST,
                unsafe_libopus::varargs!(-2i32),
            )
        },
        -1
    );
    // Invalid frame size (500 samples) should return -1
    assert_eq!(enc.encode(&inbuf[..500 * 2], &mut packet[..1500]), -1);

    // Main encode/decode loop across rate control modes and codec modes
    for rc in 0..3 {
        enc.set_vbr(rc < 2);
        enc.set_vbr_constraint(rc == 1);
        enc.set_vbr_constraint(rc == 1);
        enc.set_inband_fec(rc == 0);

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

        for j in 0..13 {
            let rate =
                (rates[j] as u32).wrapping_add(rng.next_u32().wrapping_rem(rates[j] as u32)) as i32;
            let mut i: i32 = 0;
            let mut count: i32 = 0;

            loop {
                let frame_size = frame[j];

                if rng.next_u32() & 255 == 0 {
                    enc.reset();
                    dec.reset();
                    if rng.next_u32() & 1 != 0 {
                        dec_err[(rng.next_u32() & 1) as usize].reset();
                    }
                }
                if rng.next_u32() & 127 == 0 {
                    dec_err[(rng.next_u32() & 1) as usize].reset();
                }
                if rng.next_u32().wrapping_rem(10) == 0 {
                    let complex = rng.next_u32().wrapping_rem(11) as i32;
                    enc.set_complexity(complex).unwrap();
                }
                if rng.next_u32().wrapping_rem(50) == 0 {
                    dec.reset();
                }
                enc.set_inband_fec(rc == 0);
                // SET_FORCE_MODE: 1000 + modes[j]
                assert_eq!(
                    unsafe {
                        opus_encoder_ctl_impl(
                            &mut enc as *mut OpusEncoder,
                            opus_private::OPUS_SET_FORCE_MODE_REQUEST,
                            unsafe_libopus::varargs!(1000 + modes[j]),
                        )
                    },
                    0
                );
                rng.next_u32();
                enc.set_dtx(rng.next_u32() & 1 != 0);
                enc.set_bitrate(Bitrate::Bits(rate));
                enc.set_force_channels(if rates[j] >= 64000 {
                    Some(Channels::Stereo)
                } else {
                    Some(Channels::Mono)
                })
                .unwrap();
                enc.set_complexity((count >> 2) % 11).unwrap();
                rng.next_u32();
                rng.next_u32();
                enc.set_packet_loss_perc(
                    (rng.next_u32() & 15 & rng.next_u32().wrapping_rem(15)) as i32,
                )
                .unwrap();
                let mut bw = (if modes[j] == 0 {
                    1101u32.wrapping_add(rng.next_u32().wrapping_rem(3))
                } else if modes[j] == 1 {
                    1104u32.wrapping_add(rng.next_u32() & 1)
                } else {
                    1101u32.wrapping_add(rng.next_u32().wrapping_rem(5))
                }) as i32;
                if modes[j] == 2 && bw == 1102 {
                    bw += 3;
                }
                enc.set_bandwidth(Some(Bandwidth::try_from(bw).unwrap()));

                let pcm_start = (i << 1) as usize;
                let pcm_end = pcm_start + (frame_size * 2) as usize; // 2 channels
                let mut len = enc.encode(&inbuf[pcm_start..pcm_end], &mut packet[..1500]);
                assert!(len >= 0 && len <= 1500, "opus_encode returned {len}");
                enc_final_range = enc.final_range();

                if rng.next_u32() & 3 == 0 {
                    let pad = 1;
                    assert_eq!(
                        opus_packet_pad(&mut packet[..(len + pad) as usize], len, len + pad),
                        0
                    );
                    len += pad;
                }
                if rng.next_u32() & 7 == 0 {
                    assert_eq!(
                        opus_packet_pad(&mut packet[..(len + 256) as usize], len, len + 256),
                        0
                    );
                    len += 256;
                }
                if rng.next_u32() & 3 == 0 {
                    len = opus_packet_unpad(&mut packet[..len as usize]);
                    assert!(len >= 1, "opus_packet_unpad failed: {len}");
                }

                let out_samples = dec.decode(
                    &packet[..len as usize],
                    &mut outbuf[((i << 1) as usize)..],
                    5760,
                    false,
                );
                assert_eq!(
                    out_samples, frame_size,
                    "decode mismatch: {out_samples} != {frame_size}"
                );
                dec_final_range = dec.final_range();
                assert_eq!(
                    enc_final_range, dec_final_range,
                    "final range mismatch: enc={enc_final_range} dec={dec_final_range}"
                );

                // LBRR decode
                let out_samples = dec_err[0].decode(
                    &packet[..len as usize],
                    &mut out2buf,
                    frame_size,
                    rng.next_u32() & 3 != 0,
                );
                assert_eq!(out_samples, frame_size);

                let l = if rng.next_u32() & 3 == 0 { 0 } else { len };
                let out_samples = dec_err[1].decode(
                    &packet[..l as usize],
                    &mut out2buf,
                    5760,
                    rng.next_u32() & 7 != 0,
                );
                assert!(out_samples >= 120, "LBRR decode too short: {out_samples}");

                i += frame_size;
                count += 1;
                if i >= 48000 * 30 / 3 - 5760 {
                    break;
                }
            }

            println!(
                "    Mode {} FB encode {}, {:6} bps OK.",
                mstrings[modes[j] as usize],
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
    }

    // Frame size switching test
    // SET_FORCE_MODE with OPUS_AUTO (-1000)
    assert_eq!(
        unsafe {
            opus_encoder_ctl_impl(
                &mut enc as *mut OpusEncoder,
                opus_private::OPUS_SET_FORCE_MODE_REQUEST,
                unsafe_libopus::varargs!(-1000i32),
            )
        },
        0
    );
    enc.set_force_channels(None).unwrap();
    enc.set_inband_fec(false);
    enc.set_dtx(false);

    let mut bitrate_bps: i32 = 512000;
    let mut fsize = rng.next_u32().wrapping_rem(31) as i32;
    let mut fswitch: i32 = 100;
    debruijn2(6, &mut db62);

    let mut i: i32 = 0;
    let mut count: i32 = 0;
    loop {
        let mut toc: u8 = 0;
        let mut frames = [0usize; 48];
        let mut size = [0i16; 48];
        let mut payload_offset: i32 = 0;
        let dec_final_range2: u32;
        let frame_size_1 = fsizes[db62[fsize as usize] as usize];
        let offset = i % (48000 * 30 - 5760);

        enc.set_bitrate(Bitrate::Bits(bitrate_bps));
        let pcm_start = (offset << 1) as usize;
        let pcm_end = pcm_start + (frame_size_1 * 2) as usize; // 2 channels
        let mut len = enc.encode(&inbuf[pcm_start..pcm_end], &mut packet[..1500]);
        assert!(len >= 0 && len <= 1500, "encode failed: {len}");
        count += 1;
        enc_final_range = enc.final_range();

        let out_samples = dec.decode(
            &packet[..len as usize],
            &mut outbuf[((offset << 1) as usize)..],
            5760,
            false,
        );
        assert_eq!(out_samples, frame_size_1);
        dec_final_range = dec.final_range();
        assert_eq!(dec_final_range, enc_final_range);

        assert!(
            opus_packet_parse(
                &packet[..len as usize],
                Some(&mut toc),
                Some(&mut frames),
                &mut size,
                Some(&mut payload_offset),
            ) > 0
        );

        if rng.next_u32() & 1023 == 0 {
            len = 0;
        }

        let mut j = frames[0] as i32; // frame offset within packet
        while j < len {
            for jj in 0..8 {
                packet[j as usize] = (packet[j as usize] as i32
                    ^ ((!no_fuzz && rng.next_u32() & 1023 == 0) as i32) << jj)
                    as u8;
            }
            j += 1;
        }

        let out_samples = dec_err[0].decode(&packet[..len as usize], &mut out2buf, 5760, false);
        assert!(
            out_samples >= 0 && out_samples <= 5760,
            "err decode out of range: {out_samples}"
        );
        if len > 0 {
            assert_eq!(out_samples, frame_size_1);
        }
        dec_final_range = dec_err[0].final_range();

        let dec2 = rng.next_u32().wrapping_rem(9).wrapping_add(1) as usize;
        let out_samples = dec_err[dec2].decode(&packet[..len as usize], &mut out2buf, 5760, false);
        assert!(out_samples >= 0 && out_samples <= 5760);
        dec_final_range2 = dec_err[dec2].final_range();
        if len > 0 {
            assert_eq!(dec_final_range, dec_final_range2);
        }

        fswitch -= 1;
        if fswitch < 1 {
            fsize = (fsize + 1) % 36;
            let new_size = fsizes[db62[fsize as usize] as usize];
            if new_size == 960 || new_size == 480 {
                fswitch = ((2880 / new_size) as u32)
                    .wrapping_mul(rng.next_u32().wrapping_rem(19).wrapping_add(1))
                    as i32;
            } else {
                fswitch = rng
                    .next_u32()
                    .wrapping_rem((2880 / new_size) as u32)
                    .wrapping_add(1) as i32;
            }
        }
        bitrate_bps = (rng
            .next_u32()
            .wrapping_rem(508000)
            .wrapping_add(4000)
            .wrapping_add(bitrate_bps as u32)
            >> 1) as i32;
        i += frame_size_1;
        if i >= 48000 * 30 * 4 {
            break;
        }
    }
    println!("    All framesize pairs switching encode, {count} frames OK.");

    // Cleanup: just reset (drop handles the rest)
    enc.reset();
    dec.reset();
}

fn fuzz_encoder_settings(num_encoders: i32, num_setting_changes: i32, rng: &mut TestRng) {
    let sampling_rates: [i32; 5] = [8000, 12000, 16000, 24000, 48000];
    let channels: [i32; 2] = [1, 2];
    let applications: [i32; 3] = [2049, 2048, 2051];
    let bitrates: [i32; 11] = [
        6000, 12000, 16000, 24000, 32000, 48000, 64000, 96000, 510000, -1000, -1,
    ];
    let force_channels: [i32; 4] = [-1000, -1000, 1, 2];
    let use_vbr: [i32; 3] = [0, 1, 1];
    let vbr_constraints: [i32; 3] = [0, 1, 1];
    let complexities: [i32; 11] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    let max_bandwidths: [i32; 6] = [1101, 1102, 1103, 1104, 1105, 1105];
    let signals: [i32; 4] = [-1000, -1000, 3001, 3002];
    let inband_fecs: [i32; 3] = [0, 0, 1];
    let packet_loss_perc: [i32; 4] = [0, 1, 2, 5];
    let lsb_depths: [i32; 2] = [8, 24];
    let prediction_disabled: [i32; 3] = [0, 0, 1];
    let use_dtx: [i32; 2] = [0, 1];
    let frame_sizes_ms_x2: [i32; 9] = [5, 10, 20, 40, 80, 120, 160, 200, 240];

    // Helper to pick random element using sizeof-based indexing (matching upstream)
    macro_rules! pick {
        ($arr:expr) => {
            $arr[(rng.next_u32() as u64).wrapping_rem(
                (::core::mem::size_of_val(&$arr) as u64)
                    .wrapping_div(::core::mem::size_of::<i32>() as u64),
            ) as usize]
        };
    }

    for _ in 0..num_encoders {
        let sampling_rate = pick!(sampling_rates);
        let num_channels = pick!(channels);
        let application = pick!(applications);

        let mut dec = OpusDecoder::new(sampling_rate, num_channels as usize).unwrap();
        let mut enc = OpusEncoder::new(sampling_rate, num_channels, application).unwrap();

        for _ in 0..num_setting_changes {
            let bitrate = pick!(bitrates);
            let mut force_channel = pick!(force_channels);
            let vbr = pick!(use_vbr);
            let vbr_constraint = pick!(vbr_constraints);
            let complexity = pick!(complexities);
            let max_bw = pick!(max_bandwidths);
            let sig = pick!(signals);
            let inband_fec = pick!(inband_fecs);
            let pkt_loss = pick!(packet_loss_perc);
            let lsb_depth = pick!(lsb_depths);
            let pred_disabled = pick!(prediction_disabled);
            let dtx = pick!(use_dtx);
            let frame_size_ms_x2 = pick!(frame_sizes_ms_x2);
            let frame_size = frame_size_ms_x2 * sampling_rate / 2000;
            let frame_size_enum = get_frame_size_enum(frame_size, sampling_rate);

            force_channel = force_channel.min(num_channels);

            enc.set_bitrate(Bitrate::from(bitrate));
            enc.set_force_channels(if force_channel == -1000 {
                None
            } else {
                Some(Channels::try_from(force_channel).unwrap())
            })
            .unwrap();
            enc.set_vbr(vbr != 0);
            enc.set_vbr_constraint(vbr_constraint != 0);
            enc.set_complexity(complexity).unwrap();
            enc.set_max_bandwidth(Bandwidth::try_from(max_bw).unwrap());
            enc.set_signal(if sig == -1000 {
                None
            } else {
                Some(Signal::try_from(sig).unwrap())
            });
            enc.set_inband_fec(inband_fec != 0);
            enc.set_packet_loss_perc(pkt_loss).unwrap();
            enc.set_lsb_depth(lsb_depth).unwrap();
            enc.set_prediction_disabled(pred_disabled != 0);
            enc.set_dtx(dtx != 0);
            enc.set_expert_frame_duration(frame_size_enum);

            assert_eq!(
                test_encode(&mut enc, num_channels, frame_size, &mut dec, rng),
                0,
                "fuzz_encoder_settings failed: {} kHz, {} ch, app={}, {} bps, force_ch={}, \
                 vbr={}, vbr_constraint={}, complexity={}, max_bw={}, signal={}, \
                 inband_fec={}, pkt_loss={}%, lsb_depth={}, pred_disabled={}, dtx={}, \
                 ({}/2) ms",
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
        }
    }
}
