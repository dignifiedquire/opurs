#![allow(dead_code)]
//! Shared test infrastructure for opurs tests.
//!
//! Replaces the C `test_opus_common.h` with idiomatic Rust equivalents.
//! Provides deterministic RNG, seed management, and De Bruijn sequences.
//!
//! The `#![allow(dead_code)]` is needed because Rust compiles each integration
//! test as a separate binary, so items used by only a subset of tests appear
//! unused in the others.

// ---------------------------------------------------------------------------
// Deterministic RNG — Marsaglia MWC, matches upstream fast_rand() exactly
// ---------------------------------------------------------------------------

/// Marsaglia Multiply-With-Carry RNG matching the upstream C `fast_rand()`.
///
/// The upstream C code uses two static globals `Rz` and `Rw`. This struct
/// encapsulates the same state without `static mut`.
pub struct TestRng {
    rz: u32,
    rw: u32,
}

impl TestRng {
    /// Create a new RNG with the given seed.
    ///
    /// Matches upstream seeding: `Rz = seed; Rw = 0;` (the integration tests
    /// set Rz and Rw from iseed separately — use `from_iseed` for that).
    pub fn new(seed: u32) -> Self {
        Self { rz: seed, rw: 0 }
    }

    /// Create a new RNG seeded the way upstream integration tests do it:
    /// `Rz = iseed; Rw = iseed;`
    pub fn from_iseed(iseed: u32) -> Self {
        Self {
            rz: iseed,
            rw: iseed,
        }
    }

    /// Generate next u32, identical output to C `fast_rand()`.
    pub fn next_u32(&mut self) -> u32 {
        self.rz = 36969u32
            .wrapping_mul(self.rz & 65535)
            .wrapping_add(self.rz >> 16);
        self.rw = 18000u32
            .wrapping_mul(self.rw & 65535)
            .wrapping_add(self.rw >> 16);
        (self.rz << 16).wrapping_add(self.rw)
    }

    /// Generate next value as i32 (wrapping reinterpretation).
    pub fn next_i32(&mut self) -> i32 {
        self.next_u32() as i32
    }
}

// ---------------------------------------------------------------------------
// Seed management
// ---------------------------------------------------------------------------

/// Get test seed from `TEST_SEED` environment variable, or generate a random
/// one and print it for reproducibility.
pub fn get_test_seed() -> u32 {
    if let Ok(val) = std::env::var("TEST_SEED") {
        let seed: u32 = val.parse().expect("TEST_SEED must be a valid u32");
        eprintln!("Using TEST_SEED={seed}");
        seed
    } else {
        let seed = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .subsec_nanos();
        eprintln!("Random seed: {seed} (set TEST_SEED={seed} to reproduce)");
        seed
    }
}

// ---------------------------------------------------------------------------
// De Bruijn sequence generator
// ---------------------------------------------------------------------------

/// Generate a De Bruijn sequence B(k, 2) of length k^2 into `output`.
///
/// Matches upstream `debruijn2()` from `test_opus_common.h`.
pub fn debruijn2(k: i32, output: &mut [u8]) {
    assert!(output.len() >= (k * k) as usize);
    let mut t = vec![0u8; (k * 2) as usize];
    let mut pos = (k * k) as usize;
    debruijn2_impl(&mut t, &mut pos, output, k, 1, 1);
}

fn debruijn2_impl(t: &mut [u8], pos: &mut usize, output: &mut [u8], k: i32, x: i32, y: i32) {
    if x > 2 {
        if y < 3 {
            for i in 0..y {
                *pos -= 1;
                output[*pos] = t[(i + 1) as usize];
            }
        }
    } else {
        t[x as usize] = t[(x - y) as usize];
        debruijn2_impl(t, pos, output, k, x + 1, y);
        let start = t[(x - y) as usize] as i32 + 1;
        for i in start..k {
            t[x as usize] = i as u8;
            debruijn2_impl(t, pos, output, k, x + 1, x);
        }
    }
}
