/// Tests for CELT exp_rotation() matching upstream `celt/tests/test_unit_rotation.c`.
///
/// Tests that applying forward rotation then inverse rotation recovers the
/// original vector with high SNR (>60 dB), and that the forward rotation
/// alone significantly changes the signal (SNR < 20 dB).
///
/// Upstream C: celt/tests/test_unit_rotation.c
mod test_common;

use test_common::TestRng;
use unsafe_libopus::internals::{exp_rotation, SPREAD_NORMAL};

/// Test rotation for given (N, K) parameters.
///
/// 1. Generate random vector x0
/// 2. Forward rotate → x1; verify SNR(x0, x1) < 20 dB (rotation changed it)
/// 3. Inverse rotate x1 back; verify SNR(x0, x1) >= 60 dB (perfect recovery)
fn check_rotation(n: usize, k: i32, rng: &mut TestRng) {
    let mut x0 = vec![0f32; n];
    let mut x1 = vec![0f32; n];

    for i in 0..n {
        let val = (rng.next_u32() % 32767) as f32 - 16384.0;
        x0[i] = val;
        x1[i] = val;
    }

    // Forward rotation
    unsafe {
        exp_rotation(x1.as_mut_ptr(), n as i32, 1, 1, k, SPREAD_NORMAL);
    }

    let mut err = 0.0f64;
    let mut ener = 0.0f64;
    for i in 0..n {
        err += (x0[i] as f64 - x1[i] as f64) * (x0[i] as f64 - x1[i] as f64);
        ener += x0[i] as f64 * x0[i] as f64;
    }
    let snr0 = 20.0 * (ener / err).log10();

    // Inverse rotation (should recover original)
    unsafe {
        exp_rotation(x1.as_mut_ptr(), n as i32, -1, 1, k, SPREAD_NORMAL);
    }

    err = 0.0;
    ener = 0.0;
    for i in 0..n {
        err += (x0[i] as f64 - x1[i] as f64) * (x0[i] as f64 - x1[i] as f64);
        ener += x0[i] as f64 * x0[i] as f64;
    }
    let snr = 20.0 * (ener / err).log10();

    assert!(
        snr >= 60.0,
        "Rotation inverse SNR too low for N={n}, K={k}: {snr:.1} dB < 60 dB"
    );
    assert!(
        snr0 < 20.0,
        "Forward rotation didn't change signal enough for N={n}, K={k}: SNR0={snr0:.1} dB >= 20 dB"
    );
}

#[test]
fn test_rotation_n15_k3() {
    let mut rng = TestRng::new(42);
    check_rotation(15, 3, &mut rng);
}

#[test]
fn test_rotation_n23_k5() {
    let mut rng = TestRng::new(42);
    check_rotation(23, 5, &mut rng);
}

#[test]
fn test_rotation_n50_k3() {
    let mut rng = TestRng::new(42);
    check_rotation(50, 3, &mut rng);
}

#[test]
fn test_rotation_n80_k1() {
    let mut rng = TestRng::new(42);
    check_rotation(80, 1, &mut rng);
}
