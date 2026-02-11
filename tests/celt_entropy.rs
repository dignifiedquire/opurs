/// Tests for CELT entropy coder matching upstream `celt/tests/test_unit_entropy.c`.
///
/// The upstream test is a single main() with multiple phases. We split it into
/// focused #[test] functions while preserving the exact test logic.
///
/// Upstream C: celt/tests/test_unit_entropy.c
mod test_common;

use test_common::TestRng;
use unsafe_libopus::internals::{
    ec_dec_bit_logp, ec_dec_bits, ec_dec_icdf, ec_dec_init, ec_dec_uint, ec_dec_update, ec_decode,
    ec_decode_bin, ec_enc_bit_logp, ec_enc_bits, ec_enc_done, ec_enc_icdf, ec_enc_init,
    ec_enc_patch_initial_bits, ec_enc_uint, ec_encode, ec_encode_bin, ec_tell, ec_tell_frac,
};

const DATA_SIZE: usize = 10_000_000;
const DATA_SIZE2: usize = 10_000;
const M_LOG2E: f64 = std::f64::consts::LOG2_E;

/// C RAND_MAX is typically 2^31-1. The upstream test uses rand() which returns
/// values in [0, RAND_MAX]. Our TestRng returns u32, so we mask to i31 range
/// to match the C arithmetic and avoid overflow.
const RAND_MAX: u32 = 0x7FFF_FFFF;

fn crand(rng: &mut TestRng) -> u32 {
    rng.next_u32() & RAND_MAX
}

/// Phase 1: Raw bit encoding/decoding.
///
/// Encodes all values for frequency tables 2..1023 via ec_enc_uint, then
/// raw bits for 1..15 bit widths. Decodes and verifies roundtrip.
#[test]
fn test_entropy_raw_encoding_decoding() {
    let mut buf = vec![0u8; DATA_SIZE];
    let mut _entropy: f64 = 0.0;

    // Encode: frequency tables
    let mut enc = ec_enc_init(&mut buf);
    for ft in 2..1024 {
        for i in 0..ft {
            _entropy += (ft as f64).ln() * M_LOG2E;
            ec_enc_uint(&mut enc, i, ft);
        }
    }

    // Encode: raw bits
    for ftb in 1u32..16 {
        for i in 0..(1u32 << ftb) {
            _entropy += ftb as f64;
            let nbits = ec_tell(&enc);
            ec_enc_bits(&mut enc, i, ftb);
            let nbits2 = ec_tell(&enc);
            assert_eq!(
                nbits2 - nbits,
                ftb as i32,
                "Used {} bits to encode {} bits directly",
                nbits2 - nbits,
                ftb
            );
        }
    }

    let nbits_enc = ec_tell_frac(&enc);
    ec_enc_done(&mut enc);

    // Decode: frequency tables
    let mut dec = ec_dec_init(&mut buf);
    for ft in 2u32..1024 {
        for i in 0..ft {
            let sym = ec_dec_uint(&mut dec, ft);
            assert_eq!(sym, i, "Decoded {sym} instead of {i} with ft of {ft}");
        }
    }

    // Decode: raw bits
    for ftb in 1u32..16 {
        for i in 0..(1u32 << ftb) {
            let sym = ec_dec_bits(&mut dec, ftb);
            assert_eq!(sym, i, "Decoded {sym} instead of {i} with ftb of {ftb}");
        }
    }

    let nbits_dec = ec_tell_frac(&dec);
    assert_eq!(
        nbits_enc,
        nbits_dec,
        "Reported bits used: enc={:.2}, dec={:.2}",
        nbits_enc as f64 / 8.0,
        nbits_dec as f64 / 8.0
    );
}

/// Phase 2: Encoder overflow handling.
///
/// Tests that when a small buffer overflows, range coder data takes priority
/// over raw bits.
#[test]
fn test_entropy_encoder_overflow() {
    let mut buf = vec![0u8; 2];

    // Start with a 16-bit (2-byte) buffer
    let mut enc = ec_enc_init(&mut buf);
    // Write 7 raw bits
    ec_enc_bits(&mut enc, 0x55, 7);
    // Write ~12.3 bits of range coder data
    ec_enc_uint(&mut enc, 1, 2);
    ec_enc_uint(&mut enc, 1, 3);
    ec_enc_uint(&mut enc, 1, 4);
    ec_enc_uint(&mut enc, 1, 5);
    ec_enc_uint(&mut enc, 2, 6);
    ec_enc_uint(&mut enc, 6, 7);
    ec_enc_done(&mut enc);

    // The encoder should have errored (buffer overflow)
    assert!(enc.error != 0, "Encoder should have set error on overflow");

    let mut dec = ec_dec_init(&mut buf);
    // The raw bits should have been overwritten by range coder data
    assert_eq!(
        ec_dec_bits(&mut dec, 7),
        0x05,
        "Raw bits were not correctly overwritten"
    );
    // All range coder data should have been encoded correctly
    assert_eq!(ec_dec_uint(&mut dec, 2), 1);
    assert_eq!(ec_dec_uint(&mut dec, 3), 1);
    assert_eq!(ec_dec_uint(&mut dec, 4), 1);
    assert_eq!(ec_dec_uint(&mut dec, 5), 1);
    assert_eq!(ec_dec_uint(&mut dec, 6), 2);
    assert_eq!(ec_dec_uint(&mut dec, 7), 6);
}

/// Phase 3: Random stream encoding/decoding (409,600 iterations).
///
/// Uses a seeded RNG to generate random frequency tables and data,
/// encodes then decodes, verifying roundtrip and ec_tell_frac consistency.
#[test]
fn test_entropy_random_streams() {
    let seed = test_common::get_test_seed();
    let mut rng = TestRng::new(seed);

    let mut buf = vec![0u8; DATA_SIZE2];

    for i in 0..409_600u32 {
        let ft = crand(&mut rng) / ((RAND_MAX >> (crand(&mut rng) % 11)) + 1) + 10;
        let sz = (crand(&mut rng) / ((RAND_MAX >> (crand(&mut rng) % 9)) + 1)) as usize;

        let mut data = vec![0u32; sz];
        let mut tell = vec![0u32; sz + 1];

        let mut enc = ec_enc_init(&mut buf);
        let zeros = crand(&mut rng).is_multiple_of(13);
        tell[0] = ec_tell_frac(&enc);

        for j in 0..sz {
            if zeros {
                data[j] = 0;
            } else {
                data[j] = crand(&mut rng) % ft;
            }
            ec_enc_uint(&mut enc, data[j], ft);
            tell[j + 1] = ec_tell_frac(&enc);
        }

        if crand(&mut rng).is_multiple_of(2) {
            while ec_tell(&enc) % 8 != 0 {
                ec_enc_uint(&mut enc, crand(&mut rng) % 2, 2);
            }
        }

        let tell_bits = ec_tell(&enc) as u32;
        ec_enc_done(&mut enc);

        assert_eq!(
            ec_tell(&enc) as u32,
            tell_bits,
            "ec_tell() changed after ec_enc_done(): {} instead of {} (seed: {seed}, iter: {i})",
            ec_tell(&enc),
            tell_bits,
        );

        let range_bytes = enc.offs;
        assert!(
            tell_bits.div_ceil(8) >= range_bytes,
            "ec_tell() lied, there's {range_bytes} bytes instead of {} (seed: {seed}, iter: {i})",
            tell_bits.div_ceil(8),
        );

        let mut dec = ec_dec_init(&mut buf);
        assert_eq!(
            ec_tell_frac(&dec),
            tell[0],
            "Tell mismatch at symbol 0: {} instead of {} (seed: {seed}, iter: {i})",
            ec_tell_frac(&dec),
            tell[0],
        );

        for j in 0..sz {
            let sym = ec_dec_uint(&mut dec, ft);
            assert_eq!(
                sym, data[j],
                "Decoded {sym} instead of {} with ft={ft} at pos {j}/{sz} (seed: {seed}, iter: {i})",
                data[j],
            );
            assert_eq!(
                ec_tell_frac(&dec),
                tell[j + 1],
                "Tell mismatch at symbol {}: {} instead of {} (seed: {seed}, iter: {i})",
                j + 1,
                ec_tell_frac(&dec),
                tell[j + 1],
            );
        }
    }
}

/// Phase 4: Cross-method compatibility (409,600 iterations).
///
/// Tests that encoding with one method (ec_encode, ec_encode_bin,
/// ec_enc_bit_logp, ec_enc_icdf) can be decoded by a different method.
#[test]
fn test_entropy_cross_method_compatibility() {
    let seed = test_common::get_test_seed();
    let mut rng = TestRng::new(seed);

    let mut buf = vec![0u8; DATA_SIZE2];

    for i in 0..409_600u32 {
        let sz = (crand(&mut rng) / ((RAND_MAX >> (crand(&mut rng) % 9)) + 1)) as usize;

        let mut logp1 = vec![0u32; sz];
        let mut data = vec![0u32; sz];
        let mut tell = vec![0u32; sz + 1];
        let mut enc_method = vec![0u32; sz];

        let mut enc = ec_enc_init(&mut buf);
        tell[0] = ec_tell_frac(&enc);

        for j in 0..sz {
            data[j] = crand(&mut rng) / ((RAND_MAX >> 1) + 1);
            logp1[j] = (crand(&mut rng) % 15) + 1;
            enc_method[j] = crand(&mut rng) / ((RAND_MAX >> 2) + 1);

            let p = 1u32 << logp1[j];
            let fl = if data[j] != 0 { p - 1 } else { 0 };
            let fh = p - if data[j] != 0 { 0 } else { 1 };

            match enc_method[j] {
                0 => ec_encode(&mut enc, fl, fh, p),
                1 => ec_encode_bin(&mut enc, fl, fh, logp1[j]),
                2 => ec_enc_bit_logp(&mut enc, data[j] as i32, logp1[j]),
                3 => {
                    let icdf: [u8; 2] = [1, 0];
                    ec_enc_icdf(&mut enc, data[j] as i32, &icdf, logp1[j]);
                }
                _ => unreachable!(),
            }
            tell[j + 1] = ec_tell_frac(&enc);
        }

        ec_enc_done(&mut enc);

        let range_bytes = enc.offs;
        let tell_bits = ec_tell(&enc) as u32;
        assert!(
            tell_bits.div_ceil(8) >= range_bytes,
            "tell() lied, there's {range_bytes} bytes instead of {} (seed: {seed}, iter: {i})",
            tell_bits.div_ceil(8),
        );

        let mut dec = ec_dec_init(&mut buf);
        assert_eq!(
            ec_tell_frac(&dec),
            tell[0],
            "Tell mismatch at symbol 0 (seed: {seed}, iter: {i})"
        );

        for j in 0..sz {
            let dec_method = crand(&mut rng) / ((RAND_MAX >> 2) + 1);
            let p = 1u32 << logp1[j];

            let sym = match dec_method {
                0 => {
                    let fs = ec_decode(&mut dec, p);
                    let s = (fs >= p - 1) as u32;
                    ec_dec_update(
                        &mut dec,
                        if s != 0 { p - 1 } else { 0 },
                        p - if s != 0 { 0 } else { 1 },
                        p,
                    );
                    s
                }
                1 => {
                    let fs = ec_decode_bin(&mut dec, logp1[j]);
                    let s = (fs >= p - 1) as u32;
                    ec_dec_update(
                        &mut dec,
                        if s != 0 { p - 1 } else { 0 },
                        p - if s != 0 { 0 } else { 1 },
                        p,
                    );
                    s
                }
                2 => ec_dec_bit_logp(&mut dec, logp1[j]) as u32,
                3 => {
                    let icdf: [u8; 2] = [1, 0];
                    ec_dec_icdf(&mut dec, &icdf, logp1[j]) as u32
                }
                _ => unreachable!(),
            };

            assert_eq!(
                sym, data[j],
                "Decoded {sym} instead of {} with logp1={} at pos {j}/{sz} (seed: {seed}, iter: {i}, enc_method={}, dec_method={dec_method})",
                data[j], logp1[j], enc_method[j],
            );
            assert_eq!(
                ec_tell_frac(&dec),
                tell[j + 1],
                "Tell mismatch at symbol {} (seed: {seed}, iter: {i})",
                j + 1,
            );
        }
    }
}

/// Phase 5: ec_enc_patch_initial_bits tests.
///
/// Tests patching the initial bits of an encoded stream.
#[test]
fn test_entropy_patch_initial_bits() {
    let mut buf = vec![0u8; DATA_SIZE2];

    // Test 1: patch 2 bits to value 3
    {
        let mut enc = ec_enc_init(&mut buf);
        ec_enc_bit_logp(&mut enc, 0, 1);
        ec_enc_bit_logp(&mut enc, 0, 1);
        ec_enc_bit_logp(&mut enc, 0, 1);
        ec_enc_bit_logp(&mut enc, 0, 1);
        ec_enc_bit_logp(&mut enc, 0, 2);
        ec_enc_patch_initial_bits(&mut enc, 3, 2);
        assert_eq!(enc.error, 0, "patch_initial_bits failed");

        // Patching 5 bits should fail (not enough data)
        ec_enc_patch_initial_bits(&mut enc, 0, 5);
        assert_ne!(
            enc.error, 0,
            "patch_initial_bits didn't fail when it should have"
        );

        ec_enc_done(&mut enc);
        assert_eq!(enc.offs, 1, "Expected 1 byte after patch_initial_bits");
        assert_eq!(buf[0], 192, "Got {} when expecting 192", buf[0]);
    }

    // Test 2: another patch scenario
    {
        let mut enc = ec_enc_init(&mut buf);
        ec_enc_bit_logp(&mut enc, 0, 1);
        ec_enc_bit_logp(&mut enc, 0, 1);
        ec_enc_bit_logp(&mut enc, 1, 6);
        ec_enc_bit_logp(&mut enc, 0, 2);
        ec_enc_patch_initial_bits(&mut enc, 0, 2);
        assert_eq!(enc.error, 0, "patch_initial_bits failed (test 2)");

        ec_enc_done(&mut enc);
        assert_eq!(enc.offs, 2, "Expected 2 bytes after patch_initial_bits");
        assert_eq!(buf[0], 63, "Got {} when expecting 63", buf[0]);
    }
}

/// Phase 5 continued: Raw bits overflow detection.
///
/// Tests that writing too many raw bits is detected as an error.
#[test]
fn test_entropy_raw_bits_overflow() {
    let mut buf = vec![0u8; 2];

    // Test 1: 48 raw bits + 1 range bit should overflow 2 bytes
    {
        let mut enc = ec_enc_init(&mut buf);
        ec_enc_bit_logp(&mut enc, 0, 2);
        for _ in 0..48 {
            ec_enc_bits(&mut enc, 0, 1);
        }
        ec_enc_done(&mut enc);
        assert_ne!(
            enc.error, 0,
            "Raw bits overfill didn't fail when it should have"
        );
    }

    // Test 2: 17 raw bits should overflow 2 bytes
    {
        let mut enc = ec_enc_init(&mut buf);
        for _ in 0..17 {
            ec_enc_bits(&mut enc, 0, 1);
        }
        ec_enc_done(&mut enc);
        assert_ne!(
            enc.error, 0,
            "17 raw bits encoded in two bytes without error"
        );
    }
}
