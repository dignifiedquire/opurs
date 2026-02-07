/// Tests for CELT math operations matching upstream `celt/tests/test_unit_mathops.c`.
///
/// This is the floating-point (non-FIXED_POINT) variant. The upstream C test
/// conditionally compiles different test bodies; we test the float path since
/// this Rust port uses f32 throughout.
///
/// Upstream C: celt/tests/test_unit_mathops.c
use unsafe_libopus::internals::{bitexact_cos, bitexact_log2tan, celt_exp2, celt_log2, celt_sqrt};

/// Upstream C: testbitexactcos()
///
/// Validates the bitexact_cos() lookup/interpolation table by computing a
/// running checksum and tracking min/max deltas between consecutive outputs.
#[test]
fn test_bitexact_cos() {
    let mut chk: i32 = 0;
    let mut max_d: i32 = 0;
    let mut min_d: i32 = 32767;
    let mut last: i32 = 32767;

    for i in 64..=16320 {
        let q = bitexact_cos(i as i16) as i32;
        chk ^= q * i;
        let d = last - q;
        if d > max_d {
            max_d = d;
        }
        if d < min_d {
            min_d = d;
        }
        last = q;
    }

    assert_eq!(chk, 89408644, "bitexact_cos checksum mismatch: got {chk}");
    assert_eq!(max_d, 5, "bitexact_cos max delta: got {max_d}");
    assert_eq!(min_d, 0, "bitexact_cos min delta: got {min_d}");
    assert_eq!(bitexact_cos(64) as i32, 32767, "bitexact_cos(64) failed");
    assert_eq!(
        bitexact_cos(16320) as i32,
        200,
        "bitexact_cos(16320) failed"
    );
    assert_eq!(
        bitexact_cos(8192) as i32,
        23171,
        "bitexact_cos(8192) failed"
    );
}

/// Upstream C: testbitexactlog2tan()
///
/// Validates bitexact_log2tan() using cos/sin pairs and checks symmetry,
/// running checksum, and min/max deltas.
#[test]
fn test_bitexact_log2tan() {
    let mut chk: i32 = 0;
    let mut max_d: i32 = 0;
    let mut min_d: i32 = 15059;
    let mut last: i32 = 15059;
    let mut fail = false;

    for i in 64..8193 {
        let mid = bitexact_cos(i as i16) as i32;
        let side = bitexact_cos((16384 - i) as i16) as i32;
        let q = bitexact_log2tan(mid, side);
        chk ^= q * i;
        let d = last - q;

        // Symmetry: log2tan(mid, side) == -log2tan(side, mid)
        if q != -1 * bitexact_log2tan(side, mid) {
            fail = true;
        }

        if d > max_d {
            max_d = d;
        }
        if d < min_d {
            min_d = d;
        }
        last = q;
    }

    assert!(!fail, "bitexact_log2tan symmetry check failed");
    assert_eq!(
        chk, 15821257,
        "bitexact_log2tan checksum mismatch: got {chk}"
    );
    assert_eq!(max_d, 61, "bitexact_log2tan max delta: got {max_d}");
    assert_eq!(min_d, -2, "bitexact_log2tan min delta: got {min_d}");
    assert_eq!(
        bitexact_log2tan(32767, 200),
        15059,
        "bitexact_log2tan(32767, 200) failed"
    );
    assert_eq!(
        bitexact_log2tan(30274, 12540),
        2611,
        "bitexact_log2tan(30274, 12540) failed"
    );
    assert_eq!(
        bitexact_log2tan(23171, 23171),
        0,
        "bitexact_log2tan(23171, 23171) failed"
    );
}

/// Upstream C: testsqrt() (float path)
///
/// Tests celt_sqrt() accuracy over a wide range with adaptive step sizes.
/// Allows up to 0.05% relative error or 2 unit absolute error.
#[test]
fn test_celt_sqrt() {
    let mut i: i64 = 1;
    while i <= 1_000_000_000 {
        let val = celt_sqrt(i as f32);
        let expected = (i as f64).sqrt();
        let ratio = val as f64 / expected;
        assert!(
            (ratio - 1.0).abs() <= 0.0005 || (val as f64 - expected).abs() <= 2.0,
            "celt_sqrt({i}) = {val}, expected ~{expected:.4}, ratio = {ratio:.6}"
        );
        i += i >> 10;
        i = i.max(i + 1); // ensure progress even for small i
    }
}

/// Upstream C: testlog2() (float path)
///
/// Tests celt_log2() against reference log2 over a wide positive range.
#[test]
fn test_celt_log2() {
    let mut x: f32 = 0.001;
    while x < 1_677_700.0 {
        let expected = (1.442695040888963387 * (x as f64).ln()) as f32;
        let actual = celt_log2(x);
        let error = (expected - actual).abs();
        assert!(
            error <= 0.0009,
            "celt_log2({x}) = {actual}, expected {expected}, error = {error}"
        );
        x += x / 8.0;
    }
}

/// Upstream C: testexp2() (float path)
///
/// Tests celt_exp2() by verifying log2(exp2(x)) ≈ x.
#[test]
fn test_celt_exp2() {
    let mut x: f32 = -11.0;
    while x < 24.0 {
        let result = celt_exp2(x);
        let error = (x as f64 - 1.442695040888963387 * (result as f64).ln()).abs();
        assert!(
            error <= 0.0002,
            "celt_exp2({x}) = {result}, roundtrip error = {error}"
        );
        x += 0.0007;
    }
}

/// Upstream C: testexp2log2() (float path)
///
/// Tests roundtrip: celt_log2(celt_exp2(x)) ≈ x.
#[test]
fn test_celt_exp2_log2_roundtrip() {
    let mut x: f32 = -11.0;
    while x < 24.0 {
        let error = (x - celt_log2(celt_exp2(x))).abs();
        assert!(
            error <= 0.001,
            "exp2/log2 roundtrip failed: x={x}, error={error}"
        );
        x += 0.0007;
    }
}
