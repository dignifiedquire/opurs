//! Integration tests for the Opus decoder.
//!
//! Upstream C: tests/test_opus_decode.c
//!
//! Split from the original monolithic `test_decoder_code0()` and `test_soft_clip()`.
//! The fuzz sections that depend on chained RNG state remain in a single test
//! to preserve the exact random sequence from the upstream C test.

mod test_common;

use opurs::{
    opus_decoder_get_nb_samples, opus_packet_get_nb_channels, opus_pcm_soft_clip, OpusDecoder,
};
use test_common::{debruijn2, TestRng};

/// Sample rates used by the decoder tests (matching upstream fsv[]).
const SAMPLE_RATES: [i32; 5] = [48000, 24000, 16000, 12000, 8000];

/// Number of decoders: 5 sample rates × 2 channel configs.
const NUM_DECODERS: usize = 10;

/// Sentinel value for guard band checking.
const GUARD_VALUE: i16 = 32749;

/// Guard band size in samples per channel.
const GUARD_SAMPLES: usize = 8;

/// Maximum output frame in samples.
const MAX_FRAME: i32 = 5760;

/// The fixed seed used by the upstream test (passed as argv[1]).
const TEST_SEED: u32 = 42;

// ---------------------------------------------------------------------------
// Helper: create all 10 decoders with clone-and-drop verification
// ---------------------------------------------------------------------------

/// Create 10 decoders (5 sample rates × {mono, stereo}), verifying each
/// can be cloned and the clone works after dropping the original.
///
/// Returns a Vec of decoders and a backup decoder (mono, for FEC/PLC tests).
fn create_test_decoders() -> (Vec<OpusDecoder>, OpusDecoder) {
    let mut decoders: Vec<OpusDecoder> = Vec::with_capacity(NUM_DECODERS);

    for t in 0..NUM_DECODERS {
        let fs = SAMPLE_RATES[t >> 1];
        let c = (t as i32 & 1) + 1;
        let dec = OpusDecoder::new(fs, c as usize)
            .unwrap_or_else(|err| panic!("OpusDecoder::new({fs}, {c}) failed: err={err}"));

        // Clone decoder, use clone (original dropped at end of iteration)
        let copy = dec.clone();

        decoders.push(copy);
    }

    // Backup decoder (mono, for FEC/PLC tests)
    let decbak = OpusDecoder::new(48000, 1).unwrap();

    (decoders, decbak)
}

// ---------------------------------------------------------------------------
// Test: decoder creation and copy
// ---------------------------------------------------------------------------

/// Upstream C: test_opus_decode.c:test_decoder_code0 (decoder creation section)
#[test]
fn test_decoder_creation_and_copy() {
    let (decoders, _) = create_test_decoders();
    assert_eq!(decoders.len(), NUM_DECODERS);
    // Decoders are valid if creation succeeded (asserts inside create_test_decoders)
}

// ---------------------------------------------------------------------------
// Test: initial PLC frames
// ---------------------------------------------------------------------------

/// Upstream C: test_opus_decode.c:test_decoder_code0 (initial PLC section, lines 106–163)
#[test]
fn test_decoder_initial_plc() {
    let (mut decoders, _) = create_test_decoders();
    let mut outbuf_storage = vec![GUARD_VALUE; (MAX_FRAME as usize + 2 * GUARD_SAMPLES) * 2];
    let outbuf = &mut outbuf_storage[GUARD_SAMPLES * 2..];
    let packet = vec![0u8; 1500];

    for t in 0..NUM_DECODERS {
        let factor = 48000 / SAMPLE_RATES[t >> 1];
        let dec = &mut decoders[t];

        for fec in 0..2 {
            let fec_bool = fec != 0;

            // PLC with minimum frame size
            let out_samples = dec.decode(&[], outbuf, 120 / factor, fec_bool);
            assert_eq!(
                out_samples,
                120 / factor,
                "dec[{t}] PLC fec={fec}: expected {}, got {out_samples}",
                120 / factor
            );

            let dur = dec.last_packet_duration();
            assert_eq!(dur, 120 / factor, "dec[{t}] duration mismatch after PLC");

            // Non-multiple-of-2.5ms should fail
            let out_samples = dec.decode(&[], outbuf, 120 / factor + 2, fec_bool);
            assert_eq!(
                out_samples, -1,
                "dec[{t}] non-2.5ms-multiple should fail, got {out_samples}"
            );

            let dur = dec.last_packet_duration();
            assert_eq!(
                dur,
                120 / factor,
                "dec[{t}] duration should be unchanged after bad frame size"
            );

            // Empty packet slice
            let out_samples = dec.decode(&packet[..0], outbuf, 120 / factor, fec_bool);
            assert_eq!(
                out_samples,
                120 / factor,
                "dec[{t}] empty packet PLC: expected {}, got {out_samples}",
                120 / factor
            );

            // Zero-length decode
            outbuf[0] = GUARD_VALUE;
            let out_samples = dec.decode(&packet[..0], outbuf, 0, fec_bool);
            assert!(
                out_samples <= 0,
                "dec[{t}] zero-length decode should return <= 0, got {out_samples}"
            );

            // Null output with zero length
            let out_samples = dec.decode(&packet[..0], &mut [], 0, fec_bool);
            assert!(
                out_samples <= 0,
                "dec[{t}] null output zero-length should return <= 0, got {out_samples}"
            );
            assert_eq!(
                outbuf[0], GUARD_VALUE,
                "dec[{t}] output buffer was modified when it shouldn't have been"
            );

            // Invalid FEC value
            let invalid_fec = if fec != 0 { -1 } else { 2 };
            let out_samples = opurs::opus_decode(dec, &packet[..1], outbuf, MAX_FRAME, invalid_fec);
            assert!(
                out_samples < 0,
                "dec[{t}] invalid fec={invalid_fec} should fail, got {out_samples}"
            );

            dec.reset();
        }
    }
}

// ---------------------------------------------------------------------------
// Test: all 2-byte prefixes for all 64 modes
// ---------------------------------------------------------------------------

/// Upstream C: test_opus_decode.c:test_decoder_code0 (2-byte prefix section, lines 167–219)
#[test]
fn test_decoder_all_2byte_prefixes() {
    let (mut decoders, _) = create_test_decoders();
    let mut outbuf = vec![0i16; MAX_FRAME as usize * 2];
    let mut packet = vec![0u8; 1500];

    for i in 0..64 {
        let mut expected = [0i32; NUM_DECODERS];
        packet[0] = (i << 2) as u8;
        packet[1] = 255;
        packet[2] = 255;

        // Verify channel count from packet header
        let nb_channels = opus_packet_get_nb_channels(packet[0]);
        assert_eq!(
            nb_channels,
            (i & 1) + 1,
            "mode {i}: expected {} channels, got {nb_channels}",
            (i & 1) + 1
        );

        // Get expected sample counts
        for t in 0..NUM_DECODERS {
            expected[t] = opus_decoder_get_nb_samples(&mut decoders[t], &packet[..1]);
            assert!(
                expected[t] <= 2880,
                "dec[{t}] mode {i}: nb_samples {} > 2880",
                expected[t]
            );
        }

        // Test all 256 second-byte values
        for j in 0..256u16 {
            packet[1] = j as u8;
            let mut dec_final_range2 = 0u32;

            for t in 0..NUM_DECODERS {
                let dec = &mut decoders[t];
                let out_samples = dec.decode(&packet[..3], &mut outbuf, MAX_FRAME, false);
                assert_eq!(
                    out_samples, expected[t],
                    "dec[{t}] mode {i} byte {j}: expected {}, got {out_samples}",
                    expected[t]
                );

                let dur = dec.last_packet_duration();
                assert_eq!(dur, out_samples, "dec[{t}] duration != out_samples");

                let dec_final_range1 = dec.final_range();

                if t == 0 {
                    dec_final_range2 = dec_final_range1;
                } else {
                    assert_eq!(
                        dec_final_range1, dec_final_range2,
                        "dec[{t}] final range mismatch vs dec[0] for mode {i} byte {j}"
                    );
                }
            }
        }

        // PLC recovery after decoding
        for t in 0..NUM_DECODERS {
            let factor = 48000 / SAMPLE_RATES[t >> 1];
            let dec = &mut decoders[t];

            // 6 PLC frames at the expected size
            for _ in 0..6 {
                let out_samples = dec.decode(&[], &mut outbuf, expected[t], false);
                assert_eq!(
                    out_samples, expected[t],
                    "dec[{t}] PLC recovery: expected {}, got {out_samples}",
                    expected[t]
                );
                let dur = dec.last_packet_duration();
                assert_eq!(dur, out_samples);
            }

            // Reset to minimum frame size if needed
            if expected[t] != 120 / factor {
                let out_samples = dec.decode(&[], &mut outbuf, 120 / factor, false);
                assert_eq!(out_samples, 120 / factor);
                let dur = dec.last_packet_duration();
                assert_eq!(dur, out_samples);
            }

            // Undersized buffer should fail
            let out_samples = dec.decode(&packet[..2], &mut outbuf, expected[t] - 1, false);
            assert!(
                out_samples <= 0,
                "dec[{t}] undersized buffer should fail, got {out_samples}"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Test: decoder fuzzing with chained RNG state
// ---------------------------------------------------------------------------

/// Upstream C: test_opus_decode.c:test_decoder_code0 (fuzz sections)
///
/// This test preserves the exact RNG sequence from the upstream C test
/// (seed=42) because the checksum regression values depend on it.
/// Sections included:
/// - CELT 3-byte prefix checksums (65536 iterations)
/// - Long-packet 3-byte prefix checksums (65536 iterations)
/// - Random packets, all 64 modes
/// - De Bruijn mode pairs (4096)
/// - De Bruijn mode pairs ×10
/// - Pre-selected random packets
#[test]
#[allow(clippy::needless_range_loop)]
fn test_decoder_fuzz() {
    if std::env::var("TEST_OPUS_NOFUZZ").is_ok() {
        eprintln!("Skipping decoder fuzz tests (TEST_OPUS_NOFUZZ is set)");
        return;
    }

    let (mut decoders, _) = create_test_decoders();
    let mut decbak;
    let mut rng = TestRng::from_iseed(TEST_SEED);

    // Consume one RNG value to match upstream: fast_rand() % 65535 in the header print
    let _ = rng.next_u32() % 65535;

    let mut outbuf_storage = vec![GUARD_VALUE; (MAX_FRAME as usize + 2 * GUARD_SAMPLES) * 2];
    let outbuf = &mut outbuf_storage[GUARD_SAMPLES * 2..];
    let mut packet = vec![0u8; 1500];
    let mut modes = [0u8; 4096];

    // --- CELT 3-byte prefix checksum test ---
    let cmodes: [i32; 4] = [16, 20, 24, 28];
    let cres: [u32; 4] = [116290185, 2172123586, 2172123586, 2172123586];

    let mode = rng.next_u32().wrapping_rem(4) as usize;
    packet[0] = (cmodes[mode] << 3) as u8;
    let mut dec_final_acc: u32 = 0;
    let t = rng.next_u32().wrapping_rem(10) as usize;

    for i in 0..65536u32 {
        let factor = 48000 / SAMPLE_RATES[t >> 1];
        packet[1] = (i >> 8) as u8;
        packet[2] = (i & 255) as u8;
        packet[3] = 255;
        let out_samples = decoders[t].decode(&packet[..4], outbuf, MAX_FRAME, false);
        assert_eq!(
            out_samples,
            120 / factor,
            "CELT 3-byte prefix: dec[{t}] i={i}: expected {}, got {out_samples}",
            120 / factor
        );
        let dec_final_range = decoders[t].final_range();
        dec_final_acc = dec_final_acc.wrapping_add(dec_final_range);
    }
    assert_eq!(
        dec_final_acc, cres[mode],
        "CELT 3-byte prefix checksum: mode {} expected {}, got {dec_final_acc}",
        cmodes[mode], cres[mode]
    );

    // --- Long-packet 3-byte prefix checksum test ---
    let lmodes: [i32; 3] = [0, 4, 8];
    let lres: [u32; 3] = [3285687739, 1481572662, 694350475];

    let mode = rng.next_u32().wrapping_rem(3) as usize;
    packet[0] = (lmodes[mode] << 3) as u8;
    dec_final_acc = 0;
    let t = rng.next_u32().wrapping_rem(10) as usize;

    for i in 0..65536u32 {
        let factor = 48000 / SAMPLE_RATES[t >> 1];
        packet[1] = (i >> 8) as u8;
        packet[2] = (i & 255) as u8;
        packet[3] = 255;
        let out_samples = decoders[t].decode(&packet[..4], outbuf, MAX_FRAME, false);
        assert_eq!(
            out_samples,
            480 / factor,
            "Long 3-byte prefix: dec[{t}] i={i}: expected {}, got {out_samples}",
            480 / factor
        );
        let dec_final_range = decoders[t].final_range();
        dec_final_acc = dec_final_acc.wrapping_add(dec_final_range);
    }
    assert_eq!(
        dec_final_acc, lres[mode],
        "Long 3-byte prefix checksum: mode {} expected {}, got {dec_final_acc}",
        lmodes[mode], lres[mode]
    );

    // --- Random packets, all 64 modes ---
    let skip = rng.next_u32().wrapping_rem(7) as i32;
    for i in 0..64 {
        let mut expected = [0i32; NUM_DECODERS];
        packet[0] = (i << 2) as u8;

        for t in 0..NUM_DECODERS {
            expected[t] = opus_decoder_get_nb_samples(&mut decoders[t], &packet[..1]);
        }

        let mut j = 2 + skip;
        while j < 1275 {
            for jj in 0..j {
                packet[(jj + 1) as usize] = (rng.next_u32() & 255) as u8;
            }
            let mut dec_final_range2 = 0u32;
            for t in 0..NUM_DECODERS {
                let out_samples =
                    decoders[t].decode(&packet[..(j + 1) as usize], outbuf, MAX_FRAME, false);
                assert_eq!(
                    out_samples, expected[t],
                    "Random packets: dec[{t}] mode {i} len {j}: expected {}, got {out_samples}",
                    expected[t]
                );
                let dec_final_range1 = decoders[t].final_range();
                if t == 0 {
                    dec_final_range2 = dec_final_range1;
                } else {
                    assert_eq!(
                        dec_final_range1, dec_final_range2,
                        "Random packets: dec[{t}] final range mismatch vs dec[0]"
                    );
                }
            }
            j += 4;
        }
    }

    // --- De Bruijn mode pairs (4096) with FEC ---
    debruijn2(64, &mut modes);

    let plen = rng
        .next_u32()
        .wrapping_rem(18)
        .wrapping_add(3)
        .wrapping_mul(8)
        .wrapping_add(skip as u32)
        .wrapping_add(3) as i32;

    for i in 0..4096 {
        let mut expected = [0i32; NUM_DECODERS];
        packet[0] = (modes[i] as i32 * 4) as u8;

        for t in 0..NUM_DECODERS {
            expected[t] = opus_decoder_get_nb_samples(&mut decoders[t], &packet[..plen as usize]);
        }

        for j in 0..plen {
            packet[(j + 1) as usize] = ((rng.next_u32() | rng.next_u32()) & 255) as u8;
        }

        // FEC test with backup decoder
        decbak = decoders[0].clone();
        let out = decbak.decode(&packet[..(plen + 1) as usize], outbuf, expected[0], true);
        assert_eq!(
            out, expected[0],
            "De Bruijn FEC decode: mode pair {i}: expected {}, got {out}",
            expected[0]
        );

        // PLC with FEC=1
        decbak = decoders[0].clone();
        let out = decbak.decode(&[], outbuf, MAX_FRAME, true);
        assert!(out >= 20, "De Bruijn PLC fec=1: got {out}");

        // PLC with FEC=0
        decbak = decoders[0].clone();
        let out = decbak.decode(&[], outbuf, MAX_FRAME, false);
        assert!(out >= 20, "De Bruijn PLC fec=0: got {out}");

        // Normal decode on all decoders
        for t in 0..NUM_DECODERS {
            let out_samples =
                decoders[t].decode(&packet[..(plen + 1) as usize], outbuf, MAX_FRAME, false);
            assert_eq!(
                out_samples, expected[t],
                "De Bruijn decode: dec[{t}] mode pair {i}: expected {}, got {out_samples}",
                expected[t]
            );
            let dur = decoders[t].last_packet_duration();
            assert_eq!(dur, out_samples);
        }
    }

    // --- De Bruijn mode pairs ×10, single decoder ---
    let plen = rng
        .next_u32()
        .wrapping_rem(18)
        .wrapping_add(3)
        .wrapping_mul(8)
        .wrapping_add(skip as u32)
        .wrapping_add(3) as i32;

    let t = {
        let mut buf = [0u8];
        getrandom::getrandom(&mut buf).unwrap();
        buf[0] as usize & 0x3
    };

    for i in 0..4096 {
        packet[0] = (modes[i] as i32 * 4) as u8;
        let expected = opus_decoder_get_nb_samples(&mut decoders[t], &packet[..plen as usize]);

        for _ in 0..10 {
            for j in 0..plen {
                packet[(j + 1) as usize] = ((rng.next_u32() | rng.next_u32()) & 255) as u8;
            }
            let out_samples =
                decoders[t].decode(&packet[..(plen + 1) as usize], outbuf, MAX_FRAME, false);
            assert_eq!(
                out_samples, expected,
                "De Bruijn ×10: dec[{t}] mode pair {i}: expected {expected}, got {out_samples}"
            );
        }
    }

    // --- Pre-selected random packets ---
    let tmodes: [i32; 1] = [25 << 2];
    let tseeds: [u32; 1] = [140441];
    let tlen: [i32; 1] = [157];
    let tret: [i32; 1] = [480];

    let t = (rng.next_u32() & 1) as usize;
    for i in 0..1 {
        packet[0] = tmodes[i] as u8;
        // Re-seed RNG for this specific test vector
        let mut local_rng = TestRng::from_iseed(tseeds[i]);
        for j in 1..tlen[i] {
            packet[j as usize] = (local_rng.next_u32() & 255) as u8;
        }
        let out_samples = decoders[t].decode(&packet[..tlen[i] as usize], outbuf, MAX_FRAME, false);
        assert_eq!(
            out_samples, tret[i],
            "Pre-selected packet {i}: dec[{t}] expected {}, got {out_samples}",
            tret[i]
        );
    }

    // --- Guard band check ---
    let mut guard_err = false;
    for val in &outbuf_storage[..(GUARD_SAMPLES * 2)] {
        if *val != GUARD_VALUE {
            guard_err = true;
        }
    }
    for val in &outbuf_storage[(GUARD_SAMPLES * 2 + MAX_FRAME as usize * 2)
        ..(GUARD_SAMPLES * 2 + (MAX_FRAME as usize + GUARD_SAMPLES) * 2)]
    {
        if *val != GUARD_VALUE {
            guard_err = true;
        }
    }
    assert!(!guard_err, "Output buffer guard bands were overwritten");
}

// ---------------------------------------------------------------------------
// Test: opus_pcm_soft_clip
// ---------------------------------------------------------------------------

/// Upstream C: test_opus_decode.c:test_soft_clip
#[test]
#[allow(clippy::needless_range_loop)]
fn test_soft_clip() {
    let mut x = [0f32; 1024];
    let mut s = [0f32; 8];

    // Single-channel soft clip at varying offsets
    for i in 0..1024usize {
        for j in 0..1024 {
            x[j] = (j & 255) as f32 * (1.0 / 32.0) - 4.0;
        }
        opus_pcm_soft_clip(&mut x[i..], 1024 - i, 1, &mut s);
        for j in i..1024 {
            assert!(
                x[j] <= 1.0,
                "soft_clip single-channel offset={i}: x[{j}]={} > 1.0",
                x[j]
            );
            assert!(
                x[j] >= -1.0,
                "soft_clip single-channel offset={i}: x[{j}]={} < -1.0",
                x[j]
            );
        }
    }

    // Multi-channel soft clip
    for channels in 1usize..9 {
        for j in 0..1024 {
            x[j] = (j & 255) as f32 * (1.0 / 32.0) - 4.0;
        }
        opus_pcm_soft_clip(&mut x, 1024 / channels, channels, &mut s);
        for j in 0..(1024 / channels * channels) {
            assert!(
                x[j] <= 1.0,
                "soft_clip ch={channels}: x[{j}]={} > 1.0",
                x[j]
            );
            assert!(
                x[j] >= -1.0,
                "soft_clip ch={channels}: x[{j}]={} < -1.0",
                x[j]
            );
        }
    }

    // Edge cases: zero samples, zero channels
    opus_pcm_soft_clip(&mut x, 0, 1, &mut s);
    opus_pcm_soft_clip(&mut x, 1, 0, &mut s);
}
