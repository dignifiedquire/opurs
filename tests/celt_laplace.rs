/// Tests for CELT Laplace entropy coder matching upstream
/// `celt/tests/test_unit_laplace.c`.
///
/// Tests encoding and decoding of values drawn from a Laplace distribution
/// with varying decay parameters.
///
/// Upstream C: celt/tests/test_unit_laplace.c
mod test_common;

use test_common::TestRng;
use unsafe_libopus::internals::{
    ec_dec_init, ec_enc_done, ec_enc_init, ec_laplace_decode, ec_laplace_encode, LAPLACE_MINP,
    LAPLACE_NMIN,
};

const DATA_SIZE: usize = 40_000;

/// Matches upstream `ec_laplace_get_start_freq()`.
fn ec_laplace_get_start_freq(decay: i32) -> i32 {
    let ft: u32 = (32768 - LAPLACE_MINP * (2 * LAPLACE_NMIN + 1)) as u32;
    let fs = (ft as i64 * (16384 - decay) as i64 / (16384 + decay) as i64) as i32;
    fs + LAPLACE_MINP
}

/// Encode and decode 10,000 Laplace-distributed values and verify roundtrip.
///
/// The first 3 values are deterministic, the remaining 9,997 are random
/// (using a simple RNG — not matching upstream srand/rand exactly, but the
/// test logic is the same: random values in [-7, 7] with random decay).
#[test]
fn test_laplace_roundtrip() {
    let seed = test_common::get_test_seed();
    let mut rng = TestRng::new(seed);

    let mut val = vec![0i32; 10_000];
    let mut decay = vec![0i32; 10_000];

    // Deterministic first values (matching upstream)
    val[0] = 3;
    decay[0] = 6000;
    val[1] = 0;
    decay[1] = 5800;
    val[2] = -1;
    decay[2] = 5600;

    // Random remaining values
    for i in 3..10_000 {
        val[i] = (rng.next_u32() % 15) as i32 - 7;
        decay[i] = (rng.next_u32() % 11_000) as i32 + 5000;
    }

    let mut buf = vec![0u8; DATA_SIZE];

    // Encode
    let mut enc = ec_enc_init(&mut buf);
    for i in 0..10_000 {
        let fs = ec_laplace_get_start_freq(decay[i]) as u32;
        unsafe {
            ec_laplace_encode(&mut enc, &mut val[i], fs, decay[i]);
        }
    }
    ec_enc_done(&mut enc);

    let enc_bytes = enc.offs;

    // Decode
    let mut dec = ec_dec_init(&mut buf[..enc_bytes as usize]);
    for i in 0..10_000 {
        let fs = ec_laplace_get_start_freq(decay[i]) as u32;
        let d = unsafe { ec_laplace_decode(&mut dec, fs, decay[i]) };
        assert_eq!(
            d, val[i],
            "Laplace decode mismatch at index {i}: got {d}, expected {} (decay={}, seed={seed})",
            val[i], decay[i],
        );
    }
}
