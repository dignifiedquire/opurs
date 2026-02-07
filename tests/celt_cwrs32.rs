/// Tests for CELT CWRS (Combinatorial Weight-Rooted Symbols) pulse vector
/// encoding/decoding, matching upstream `celt/tests/test_unit_cwrs32.c`.
///
/// Tests that cwrsi (index→vector) and icwrs (vector→index) are perfect
/// inverses for all supported (N, K) combinations.
///
/// Upstream C: celt/tests/test_unit_cwrs32.c
use unsafe_libopus::internals::{cwrsi, get_pulses, icwrs, pvq_v};

const NMAX: usize = 240;

/// Test dimensions — matches upstream non-CUSTOM_MODES set (standard Opus).
/// Using the standard set which covers all dimensions used in practice.
const PN: &[i32] = &[
    2, 3, 4, 6, 8, 9, 11, 12, 16, 18, 22, 24, 32, 36, 44, 48, 64, 72, 88, 96, 144, 176,
];
const PKMAX: &[i32] = &[
    128, 128, 128, 88, 36, 26, 18, 16, 12, 11, 9, 9, 7, 7, 6, 6, 5, 5, 5, 5, 4, 4,
];

/// For each valid (N, K) pair, sample combination indices uniformly,
/// convert index→vector via cwrsi, then vector→index via icwrs, and
/// verify perfect roundtrip.
#[test]
fn test_cwrs_encode_decode_roundtrip() {
    for t in 0..PN.len() {
        let n = PN[t];

        for pseudo in 1..41 {
            let k = unsafe { get_pulses(pseudo) };
            if k > PKMAX[t] {
                break;
            }

            let nc = pvq_v(n as u32, k as u32);
            // Sample up to ~20000 evenly-spaced indices
            let inc = (nc / 20_000).max(1);

            let mut i = 0u32;
            while i < nc {
                let mut y = [0i32; NMAX];

                // Index → pulse vector
                cwrsi(n as usize, k, i, &mut y);

                // Verify pulse count: sum of absolute values == K
                let sy: i32 = y[..n as usize].iter().map(|v| v.abs()).sum();
                assert_eq!(
                    sy, k,
                    "N={n} K={k} i={i}: pulse count mismatch (sum |y| = {sy} != {k})"
                );

                // Vector → index (roundtrip)
                let ii = icwrs(n as usize, &y);
                assert_eq!(
                    ii, i,
                    "N={n} K={k}: combination index mismatch ({ii} != {i})"
                );

                // Verify combination count
                let v = pvq_v(n as u32, k as u32);
                assert_eq!(
                    v, nc,
                    "N={n} K={k}: combination count mismatch ({v} != {nc})"
                );

                i += inc;
            }
        }
    }
}
