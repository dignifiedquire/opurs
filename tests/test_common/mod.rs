//! Shared test infrastructure for unsafe-libopus tests.
//!
//! Replaces the C `test_opus_common.h` with idiomatic Rust equivalents.
//! Provides deterministic RNG, seed management, De Bruijn sequences,
//! encoder/decoder factory helpers, and common test constants.

use std::sync::atomic::{AtomicU32, Ordering};

use unsafe_libopus::{opus_decoder_create, opus_encoder_create, OpusDecoder, OpusEncoder, OPUS_OK};

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

    /// Re-seed the RNG with `Rz = Rw = seed`.
    pub fn reseed(&mut self, seed: u32) {
        self.rz = seed;
        self.rw = seed;
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

static GLOBAL_SEED: AtomicU32 = AtomicU32::new(0);

/// Get test seed from `TEST_SEED` environment variable, or generate a random
/// one and print it for reproducibility.
pub fn get_test_seed() -> u32 {
    if let Ok(val) = std::env::var("TEST_SEED") {
        let seed: u32 = val.parse().expect("TEST_SEED must be a valid u32");
        eprintln!("Using TEST_SEED={seed}");
        GLOBAL_SEED.store(seed, Ordering::Relaxed);
        seed
    } else {
        let seed = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .subsec_nanos();
        eprintln!("Random seed: {seed} (set TEST_SEED={seed} to reproduce)");
        GLOBAL_SEED.store(seed, Ordering::Relaxed);
        seed
    }
}

/// Retrieve the last seed set by `get_test_seed()`.
pub fn last_seed() -> u32 {
    GLOBAL_SEED.load(Ordering::Relaxed)
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

// ---------------------------------------------------------------------------
// Encoder / decoder factory helpers
// ---------------------------------------------------------------------------

/// Create an encoder, panicking on failure with context.
///
/// # Safety
/// The returned pointer must be freed with `opus_encoder_destroy`.
pub unsafe fn create_encoder(sample_rate: i32, channels: i32, app: i32) -> *mut OpusEncoder {
    let mut err = 0;
    let enc = opus_encoder_create(sample_rate, channels, app, &mut err);
    assert_eq!(
        err, OPUS_OK,
        "opus_encoder_create failed: sr={sample_rate}, ch={channels}, app={app}, err={err}"
    );
    assert!(!enc.is_null(), "opus_encoder_create returned null");
    enc
}

/// Create a decoder, panicking on failure with context.
///
/// # Safety
/// The returned pointer must be freed with `opus_decoder_destroy`.
pub unsafe fn create_decoder(sample_rate: i32, channels: i32) -> *mut OpusDecoder {
    let mut err = 0;
    let dec = opus_decoder_create(sample_rate, channels, &mut err);
    assert_eq!(
        err, OPUS_OK,
        "opus_decoder_create failed: sr={sample_rate}, ch={channels}, err={err}"
    );
    assert!(!dec.is_null(), "opus_decoder_create returned null");
    dec
}

// ---------------------------------------------------------------------------
// Common test constants
// ---------------------------------------------------------------------------

/// All sample rates supported by Opus.
pub const SAMPLE_RATES: &[i32] = &[8000, 12000, 16000, 24000, 48000];

/// Valid channel counts.
pub const CHANNELS: &[i32] = &[1, 2];

/// Frame sizes in samples at 48 kHz.
pub const FRAME_SIZES_48K: &[i32] = &[
    120,  // 2.5ms
    240,  // 5ms
    480,  // 10ms
    960,  // 20ms
    1920, // 40ms
    2880, // 60ms
];

/// Common test bitrates.
pub const BITRATES: &[i32] = &[6000, 12000, 16000, 32000, 48000, 64000, 96000, 510000];

/// Maximum frame size in samples (60ms at 48kHz).
pub const MAX_FRAME_SAMPLES: usize = 48000 * 60 / 1000;

/// Conservative MTU-sized packet buffer.
pub const MAX_PACKET_SIZE: usize = 1500;
