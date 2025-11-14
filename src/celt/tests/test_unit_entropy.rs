use crate::celt::entcode::{ec_tell, ec_tell_frac};
use crate::celt::entdec::{
    ec_dec_bit_logp, ec_dec_bits, ec_dec_icdf, ec_dec_init, ec_dec_uint, ec_dec_update, ec_decode,
    ec_decode_bin,
};
use crate::celt::entenc::{
    ec_enc_bit_logp, ec_enc_bits, ec_enc_done, ec_enc_icdf, ec_enc_init, ec_enc_patch_initial_bits,
    ec_enc_uint, ec_encode, ec_encode_bin,
};
use crate::celt::tests::test_unit_dft::{rand, RAND_MAX};

fn ldexp(a: f64, exp: isize) -> f64 {
    let f = exp as f64;
    a * f.exp2()
}

// TODO: add option to run with randomized seed.
const seed: usize = 42;

#[test]
fn test_entropy_raw_bit_values() {
    let mut entropy = 0.;

    let mut ptr = vec![0u8; 10000000];
    let mut enc = ec_enc_init(&mut ptr);

    for ft in 2..1024 {
        for i in 0..ft {
            entropy += (ft as f64).ln() * 1.4426950408889634074f64;
            ec_enc_uint(&mut enc, i as u32, ft as u32);
        }
    }
    for ftb in 1..16 {
        for i in 0..1 << ftb {
            entropy += ftb as f64;
            let nbits = ec_tell(&mut enc) as i64;
            ec_enc_bits(&mut enc, i as u32, ftb as u32);
            let nbits2 = ec_tell(&mut enc) as i64;
            assert_eq!(
                nbits2 - nbits,
                ftb,
                "Used {} bits to encode {} bits directly.",
                nbits2 - nbits,
                ftb,
            );
        }
    }

    let nbits = ec_tell_frac(&mut enc) as i64;
    ec_enc_done(&mut enc);
    println!(
        "Encoded {} bits of entropy to {} bits ({:.04}% wasted).",
        entropy,
        ldexp(nbits as f64, -(3)),
        100 as f64 * (nbits as f64 - ldexp(entropy, 3)) / nbits as f64,
    );
    println!("Packed to {} bytes.", enc.offs);
    let mut dec = ec_dec_init(&mut ptr);

    for ft in 2..1024 {
        for i in 0..ft {
            let sym = ec_dec_uint(&mut dec, ft as u32);
            assert_eq!(
                sym, i,
                "Decoded {} instead of {} with ft of {}.",
                sym, i, ft
            );
        }
    }
    for ftb in 1..16 {
        for i in 0..(1 << ftb) {
            let sym = ec_dec_bits(&mut dec, ftb as u32);
            assert_eq!(
                sym, i,
                "Decoded {} instead of {} with ftb of {}.",
                sym, i, ftb,
            );
        }
    }

    let nbits2 = ec_tell_frac(&dec) as i64;
    assert_eq!(
        nbits,
        nbits2,
        "Reported number of bits used was {}, should be {}.",
        ldexp(nbits2 as f64, -(3)),
        ldexp(nbits as f64, -(3)),
    );
}

/// Testing an encoder bust prefers range coder data over raw bits.
/// This isn't a general guarantee, will only work for data that is buffered in
/// the encoder state and not yet stored in the user buffer, and should never
/// get used in practice.
/// It's mostly here for code coverage completeness.
#[test]
fn test_entropy_prefer_range() {
    let mut ptr = vec![0u8; 2];

    // Start with a 16-bit buffer.
    let mut enc = ec_enc_init(&mut ptr[..2]);
    // Write 7 raw bits
    ec_enc_bits(&mut enc, 0x55, 7);
    // Write 12.3 bits of range coder data
    ec_enc_uint(&mut enc, 1, 2);
    ec_enc_uint(&mut enc, 1, 3);
    ec_enc_uint(&mut enc, 1, 4);
    ec_enc_uint(&mut enc, 1, 5);
    ec_enc_uint(&mut enc, 2, 6);
    ec_enc_uint(&mut enc, 6, 7);
    ec_enc_done(&mut enc);

    let err = enc.error;
    drop(enc);
    let mut dec = ec_dec_init(&mut ptr[..2]);
    if err == 0
            /* The raw bits should have been overwritten by the range coder data. */
            || ec_dec_bits(&mut dec, 7) != 0x5
            /* And all the range coder data should have been encoded correctly. */
            || ec_dec_uint(&mut dec, 2) != 1
            || ec_dec_uint(&mut dec, 3) != 1
            || ec_dec_uint(&mut dec, 4) != 1
            || ec_dec_uint(&mut dec, 5) != 1
            || ec_dec_uint(&mut dec, 6) != 2
            || ec_dec_uint(&mut dec, 7) != 6
    {
        panic!("Encoder bust overwrote range coder data with raw bits.",);
    }
}

#[test]
fn test_entropy_random_streams() {
    let mut ptr = vec![0u8; 10000000];

    // TODO: seed
    // srand(seed);

    println!(
        "Testing random streams... Random seed: {} ({:0X})",
        seed,
        rand() % 65536,
    );
    for _i in 0..409600 {
        let ft = (rand() as u64)
            .wrapping_div(
                (RAND_MAX as u64 >> (rand() as u64).wrapping_rem(11u64)).wrapping_add(1u64),
            )
            .wrapping_add(10u64) as u32;
        let sz = (rand() as u64).wrapping_div(
            (RAND_MAX as u64 >> (rand() as u64).wrapping_rem(9u64)).wrapping_add(1u64),
        ) as u32;
        let mut data = vec![032; sz as usize];
        let mut tell = vec![0u32; (sz + 1) as usize];
        let mut enc = ec_enc_init(&mut ptr[..10000]);
        let zeros = (rand() % 13 == 0) as i32;
        tell[0] = ec_tell_frac(&mut enc);

        for j in 0..(sz as usize) {
            if zeros != 0 {
                data[j] = 0;
            } else {
                data[j] = rand() % ft;
            }
            ec_enc_uint(&mut enc, data[j], ft as u32);
            tell[j + 1] = ec_tell_frac(&mut enc);
        }
        if rand() % 2 == 0 {
            while ec_tell(&mut enc) % 8 != 0 {
                ec_enc_uint(&mut enc, (rand() % 2) as u32, 2);
            }
        }
        let tell_bits = ec_tell(&mut enc) as u32;
        ec_enc_done(&mut enc);
        assert_eq!(
            tell_bits,
            ec_tell(&mut enc) as u32,
            "ec_tell() changed after ec_enc_done(): {} instead of {} (Random seed: {})",
            ec_tell(&mut enc),
            tell_bits,
            seed,
        );

        if tell_bits.wrapping_add(7).wrapping_div(8) < enc.offs {
            panic!(
                "ec_tell() lied, there's {} bytes instead of {} (Random seed: {})",
                enc.offs,
                tell_bits.wrapping_add(7).wrapping_div(8),
                seed,
            );
        }
        let mut dec = ec_dec_init(&mut ptr[..10000]);
        assert_eq!(
            ec_tell_frac(&dec),
            tell[0],
            "Tell mismatch between encoder and decoder at symbol {}: {} instead of {} (Random seed: {}).",
            0,
            ec_tell_frac(&dec),
            tell[0],
            seed,
        );
        for j in 0..(sz as usize) {
            let sym = ec_dec_uint(&mut dec, ft as u32);
            assert_eq!(
                sym, data[j],
                "Decoded {} instead of {} with ft of {} at position {} of {} (Random seed: {}).",
                sym, data[j], ft, j, sz, seed,
            );

            assert_eq!(
                ec_tell_frac(&dec),
                tell[j + 1],
                "Tell mismatch between encoder and decoder at symbol {}: {} instead of {} (Random seed: {}).",
                j + 1,
                ec_tell_frac(&dec),
                tell[j + 1],
                seed,
            );
        }
    }
}

#[test]
fn test_entropy_compat_multi() {
    let mut ptr = vec![0u8; 10000];

    // Test compatibility between multiple different encode/decode routines.
    for _i in 0..409600 {
        let sz = (rand() as u64).wrapping_div(
            (RAND_MAX as u64 >> (rand() as u64).wrapping_rem(9u64)).wrapping_add(1u64),
        ) as u32;

        let mut logp1 = vec![0u32; sz as usize];
        let mut data_0 = vec![0u32; sz as usize];
        let mut tell_0 = vec![0u32; (sz + 1) as usize];
        let mut enc_method = vec![0u32; sz as usize];

        let mut enc = ec_enc_init(&mut ptr[..10000]);
        tell_0[0] = ec_tell_frac(&mut enc);
        for j_0 in 0..(sz as usize) {
            data_0[j_0] = rand() / ((RAND_MAX >> 1) + 1);
            logp1[j_0] = (rand() % 15) + 1;
            enc_method[j_0] = rand() / ((RAND_MAX >> 2) + 1);

            match enc_method[j_0] {
                0 => {
                    ec_encode(
                        &mut enc,
                        (if data_0[j_0] != 0 {
                            (1 << logp1[j_0]) - 1
                        } else {
                            0
                        }) as u32,
                        ((1 << logp1[j_0]) - (if data_0[j_0] != 0 { 0 } else { 1 })) as u32,
                        (1 << logp1[j_0]) as u32,
                    );
                }
                1 => {
                    ec_encode_bin(
                        &mut enc,
                        (if data_0[j_0] != 0 {
                            (1 << logp1[j_0]) - 1
                        } else {
                            0
                        }) as u32,
                        ((1 << logp1[j_0]) - (if data_0[j_0] != 0 { 0 } else { 1 })) as u32,
                        logp1[j_0],
                    );
                }
                2 => {
                    ec_enc_bit_logp(&mut enc, data_0[j_0] as i32, logp1[j_0]);
                }
                3 => {
                    let icdf = [1, 0];
                    ec_enc_icdf(&mut enc, data_0[j_0] as i32, &icdf, logp1[j_0]);
                }
                _ => {}
            }
            tell_0[j_0 + 1] = ec_tell_frac(&mut enc);
        }
        ec_enc_done(&mut enc);

        assert!(
            (ec_tell(&enc) as u32 + 7) / 8 >= enc.offs,
            "tell() lied, there's {} bytes instead of {} (Random seed: {})",
            enc.offs,
            (ec_tell(&enc) + 7) / 8,
            seed,
        );

        let mut dec = ec_dec_init(&mut ptr);
        assert_eq!(
            ec_tell_frac(&dec),
            tell_0[0],
            "Tell mismatch between encoder and decoder at symbol {}: {} instead of {} (Random seed: {}).",
            0,
            ec_tell_frac(&dec),
            tell_0[0],
            seed,
        );

        for j_0 in 0..(sz as usize) {
            let mut fs: i32 = 0;
            let dec_method = rand() / ((RAND_MAX >> 2) + 1);
            let mut sym = 0;
            match dec_method {
                0 => {
                    fs = ec_decode(&mut dec, ((1) << logp1[j_0]) as u32) as i32;
                    sym = (fs >= (1 << logp1[j_0]) - 1) as i32 as u32;
                    ec_dec_update(
                        &mut dec,
                        (if sym != 0 { (1 << logp1[j_0]) - 1 } else { 0 }) as u32,
                        ((1 << logp1[j_0]) - (if sym != 0 { 0 } else { 1 })) as u32,
                        (1 << logp1[j_0]) as u32,
                    );
                }
                1 => {
                    fs = ec_decode_bin(&mut dec, logp1[j_0]) as i32;
                    sym = (fs >= (1 << logp1[j_0]) - 1) as i32 as u32;
                    ec_dec_update(
                        &mut dec,
                        (if sym != 0 { (1 << logp1[j_0]) - 1 } else { 0 }) as u32,
                        ((1 << logp1[j_0]) - (if sym != 0 { 0 } else { 1 })) as u32,
                        (1 << logp1[j_0]) as u32,
                    );
                }
                2 => {
                    sym = ec_dec_bit_logp(&mut dec, logp1[j_0]) as u32;
                }
                3 => {
                    let mut icdf_0: [u8; 2] = [0; 2];
                    icdf_0[0 as usize] = 1;
                    icdf_0[1 as usize] = 0;
                    sym = ec_dec_icdf(&mut dec, &icdf_0, logp1[j_0]) as u32;
                }
                _ => {}
            }
            assert_eq!(
                sym, data_0[j_0],
                "Decoded {} instead of {} with logp1 of {} at position {} of {} (Random seed: {}).\nEncoding method: {}, decoding method: {}",
                sym, data_0[j_0], logp1[j_0], j_0, sz, seed, enc_method[j_0], dec_method,
            );

            assert_eq!(
                ec_tell_frac(&dec),
                tell_0[j_0 + 1],
                "Tell mismatch between encoder and decoder at symbol {}: {} instead of {} (Random seed: {}).",
                j_0 + 1,
                ec_tell_frac(&dec),
                tell_0[j_0 + 1],
                seed,
            );
        }
    }
}

#[test]
fn test_entropy_logp() {
    let mut ptr = vec![0u8; 10000];

    let mut enc = ec_enc_init(&mut ptr);
    ec_enc_bit_logp(&mut enc, 0, 1);
    ec_enc_bit_logp(&mut enc, 0, 1);
    ec_enc_bit_logp(&mut enc, 0, 1);
    ec_enc_bit_logp(&mut enc, 0, 1);
    ec_enc_bit_logp(&mut enc, 0, 2);
    ec_enc_patch_initial_bits(&mut enc, 3, 2);
    assert_eq!(enc.error, 0, "patch_initial_bits failed");

    ec_enc_patch_initial_bits(&mut enc, 0, 5);
    assert!(
        enc.error != 0,
        "patch_initial_bits didn't fail when it should have"
    );

    ec_enc_done(&mut enc);
    if enc.offs != 1 || ptr[0] != 192 {
        panic!("Got {} when expecting 192 for patch_initial_bits", ptr[0],);
    }
    let mut enc = ec_enc_init(&mut ptr);
    ec_enc_bit_logp(&mut enc, 0, 1);
    ec_enc_bit_logp(&mut enc, 0, 1);
    ec_enc_bit_logp(&mut enc, 1, 6);
    ec_enc_bit_logp(&mut enc, 0, 2);
    ec_enc_patch_initial_bits(&mut enc, 0, 2);
    assert_eq!(enc.error, 0, "patch_initial_bits failed");

    ec_enc_done(&mut enc);
    if enc.offs != 2 || ptr[0] != 63 {
        panic!("Got {} when expecting 63 for patch_initial_bits", ptr[0],);
    }
    let mut enc = ec_enc_init(&mut ptr[..2]);
    ec_enc_bit_logp(&mut enc, 0, 2);
    for _i in 0..48 {
        ec_enc_bits(&mut enc, 0, 1);
    }
    ec_enc_done(&mut enc);
    if enc.error == 0 {
        panic!("Raw bits overfill didn't fail when it should have");
    }
    let mut enc = ec_enc_init(&mut ptr[..2]);
    for _i in 0..17 {
        ec_enc_bits(&mut enc, 0, 1);
    }
    ec_enc_done(&mut enc);
    assert!(enc.error != 0, "17 raw bits encoded in two bytes");
}
