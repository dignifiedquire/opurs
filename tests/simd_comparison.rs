//! Direct comparison of Rust vs C SIMD implementations.
//! Requires `--features tools` (links libopus-sys for C symbols).
//!
//! Run with: cargo test --release --features tools --test simd_comparison

#![cfg(feature = "tools")]
#![allow(non_snake_case)]

extern crate opurs;

// Link to the C SIMD functions (non-static symbols in libopus-sys)
extern "C" {
    fn opus_select_arch() -> i32;
    fn xcorr_kernel_sse(x: *const f32, y: *const f32, sum: *mut f32, len: i32);
    fn celt_inner_prod_sse(x: *const f32, y: *const f32, N: i32) -> f32;
    fn dual_inner_prod_sse(
        x: *const f32,
        y01: *const f32,
        y02: *const f32,
        N: i32,
        xy1: *mut f32,
        xy2: *mut f32,
    );
    fn comb_filter_const_sse(
        y: *mut f32,
        x: *mut f32,
        T: i32,
        N: i32,
        g10: f32,
        g11: f32,
        g12: f32,
    );
    fn op_pvq_search_sse2(_X: *mut f32, iy: *mut i32, K: i32, N: i32, arch: i32) -> f32;
    fn celt_pitch_xcorr_avx2(
        _x: *const f32,
        _y: *const f32,
        xcorr: *mut f32,
        len: i32,
        max_pitch: i32,
        arch: i32,
    );
}

/// Simple deterministic pseudo-random number generator
struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Self {
        Rng(seed)
    }
    fn next_u32(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 33) as u32
    }
    fn next_f32(&mut self) -> f32 {
        (self.next_u32() as f32 / u32::MAX as f32) * 2.0 - 1.0
    }
}

#[test]
fn check_arch_level() {
    let arch = unsafe { opus_select_arch() };
    eprintln!("C opus_select_arch() = {arch} (0=none, 1=SSE, 2=SSE2, 3=SSE4.1, 4=AVX2)");
    eprintln!(
        "Rust: SSE={} SSE2={} SSE4.1={} AVX2={} FMA={}",
        is_x86_feature_detected!("sse"),
        is_x86_feature_detected!("sse2"),
        is_x86_feature_detected!("sse4.1"),
        is_x86_feature_detected!("avx2"),
        is_x86_feature_detected!("fma"),
    );
    // Both should detect the same CPU features
    if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
        assert_eq!(arch, 4, "C should detect AVX2 when Rust does");
    }
}

#[test]
fn compare_celt_inner_prod_sse() {
    if !is_x86_feature_detected!("sse") {
        return;
    }
    let mut rng = Rng::new(42);
    let mut mismatches = 0;

    for n in [
        4, 8, 12, 16, 17, 20, 24, 32, 48, 64, 100, 128, 240, 256, 480, 960,
    ] {
        for trial in 0..100 {
            let x: Vec<f32> = (0..n).map(|_| rng.next_f32()).collect();
            let y: Vec<f32> = (0..n).map(|_| rng.next_f32()).collect();

            let rust_result = unsafe { opurs::celt::simd::x86::celt_inner_prod_sse(&x, &y, n) };
            let c_result = unsafe { celt_inner_prod_sse(x.as_ptr(), y.as_ptr(), n as i32) };

            if rust_result.to_bits() != c_result.to_bits() {
                if mismatches < 10 {
                    eprintln!(
                        "celt_inner_prod_sse MISMATCH: n={n} trial={trial} rust={rust_result:.10e} ({:#010x}) c={c_result:.10e} ({:#010x}) diff={:.2e}",
                        rust_result.to_bits(), c_result.to_bits(), (rust_result - c_result).abs()
                    );
                }
                mismatches += 1;
            }
        }
    }
    if mismatches > 0 {
        panic!("celt_inner_prod_sse: {mismatches} mismatches out of 1600 tests");
    }
}

#[test]
fn compare_dual_inner_prod_sse() {
    if !is_x86_feature_detected!("sse") {
        return;
    }
    let mut rng = Rng::new(123);
    let mut mismatches = 0;

    for n in [4, 8, 12, 16, 17, 20, 24, 32, 48, 64, 100, 240, 960] {
        for trial in 0..100 {
            let x: Vec<f32> = (0..n).map(|_| rng.next_f32()).collect();
            let y01: Vec<f32> = (0..n).map(|_| rng.next_f32()).collect();
            let y02: Vec<f32> = (0..n).map(|_| rng.next_f32()).collect();

            let (rust_xy1, rust_xy2) =
                unsafe { opurs::celt::simd::x86::dual_inner_prod_sse(&x, &y01, &y02, n) };
            let (mut c_xy1, mut c_xy2) = (0.0f32, 0.0f32);
            unsafe {
                dual_inner_prod_sse(
                    x.as_ptr(),
                    y01.as_ptr(),
                    y02.as_ptr(),
                    n as i32,
                    &mut c_xy1,
                    &mut c_xy2,
                );
            }

            if rust_xy1.to_bits() != c_xy1.to_bits() || rust_xy2.to_bits() != c_xy2.to_bits() {
                if mismatches < 10 {
                    eprintln!(
                        "dual_inner_prod_sse MISMATCH: n={n} trial={trial} rust=({rust_xy1:.10e}, {rust_xy2:.10e}) c=({c_xy1:.10e}, {c_xy2:.10e})"
                    );
                }
                mismatches += 1;
            }
        }
    }
    if mismatches > 0 {
        panic!("dual_inner_prod_sse: {mismatches} mismatches out of 1300 tests");
    }
}

#[test]
fn compare_xcorr_kernel_sse() {
    if !is_x86_feature_detected!("sse") {
        return;
    }
    let mut rng = Rng::new(456);
    let mut mismatches = 0;

    for len in [4, 8, 12, 16, 17, 20, 24, 32, 48, 64, 100, 240] {
        for trial in 0..100 {
            let x: Vec<f32> = (0..len).map(|_| rng.next_f32()).collect();
            let y: Vec<f32> = (0..len + 3).map(|_| rng.next_f32()).collect();
            let init_sum: [f32; 4] = [
                rng.next_f32(),
                rng.next_f32(),
                rng.next_f32(),
                rng.next_f32(),
            ];

            let mut rust_sum = init_sum;
            unsafe {
                opurs::celt::simd::x86::xcorr_kernel_sse(&x, &y, &mut rust_sum, len);
            }
            let mut c_sum = init_sum;
            unsafe {
                xcorr_kernel_sse(x.as_ptr(), y.as_ptr(), c_sum.as_mut_ptr(), len as i32);
            }

            let any_diff = (0..4).any(|i| rust_sum[i].to_bits() != c_sum[i].to_bits());
            if any_diff {
                if mismatches < 10 {
                    eprintln!(
                        "xcorr_kernel_sse MISMATCH: len={len} trial={trial}\n  rust={rust_sum:?}\n  c   ={c_sum:?}"
                    );
                }
                mismatches += 1;
            }
        }
    }
    if mismatches > 0 {
        panic!("xcorr_kernel_sse: {mismatches} mismatches out of 1200 tests");
    }
}

#[test]
fn compare_comb_filter_const_sse() {
    if !is_x86_feature_detected!("sse") {
        return;
    }
    let mut rng = Rng::new(789);
    let mut mismatches = 0;

    for T in [18, 24, 36, 48, 64, 120] {
        for N in [4, 8, 16, 32, 64, 120, 240] {
            let total = N + T + 4;
            let x_start = T + 2; // enough room for x[x_start - T - 2]

            let x: Vec<f32> = (0..total).map(|_| rng.next_f32()).collect();
            let g10 = rng.next_f32() * 0.5;
            let g11 = rng.next_f32() * 0.3;
            let g12 = rng.next_f32() * 0.2;

            // Rust version
            let mut rust_y = vec![0.0f32; total];
            unsafe {
                opurs::celt::simd::x86::comb_filter_const_sse(
                    &mut rust_y,
                    x_start,
                    &x,
                    x_start,
                    T as i32,
                    N as i32,
                    g10,
                    g11,
                    g12,
                );
            }

            // C version: x pointer is at x_start, uses negative indexing for x[-T-2..]
            let mut c_y = vec![0.0f32; total];
            unsafe {
                comb_filter_const_sse(
                    c_y.as_mut_ptr().add(x_start),
                    x.as_ptr().add(x_start) as *mut f32,
                    T as i32,
                    N as i32,
                    g10,
                    g11,
                    g12,
                );
            }

            let any_diff =
                (0..N).any(|i| rust_y[x_start + i].to_bits() != c_y[x_start + i].to_bits());
            if any_diff {
                if mismatches < 5 {
                    let first_diff = (0..N)
                        .find(|&i| rust_y[x_start + i].to_bits() != c_y[x_start + i].to_bits())
                        .unwrap();
                    eprintln!(
                        "comb_filter_const_sse MISMATCH: T={T} N={N} first_diff_at={first_diff}\n  rust[{first_diff}]={:.10e} ({:#010x})\n  c   [{first_diff}]={:.10e} ({:#010x})",
                        rust_y[x_start + first_diff], rust_y[x_start + first_diff].to_bits(),
                        c_y[x_start + first_diff], c_y[x_start + first_diff].to_bits(),
                    );
                }
                mismatches += 1;
            }
        }
    }
    if mismatches > 0 {
        panic!("comb_filter_const_sse: {mismatches} mismatches out of 42 tests");
    }
}

#[test]
fn compare_op_pvq_search_sse2() {
    if !is_x86_feature_detected!("sse2") {
        return;
    }
    let mut rng = Rng::new(999);
    let mut mismatches = 0;

    for N in [4, 8, 12, 16, 20, 24, 32, 48, 64] {
        for K in [1, 2, 4, 8, 16, 32] {
            for trial in 0..10 {
                let mut rust_X: Vec<f32> = (0..N).map(|_| rng.next_f32()).collect();
                let mut c_X = rust_X.clone();
                let mut rust_iy = vec![0i32; N];
                let mut c_iy = vec![0i32; N];

                let rust_yy = unsafe {
                    opurs::celt::simd::x86::op_pvq_search_sse2(
                        &mut rust_X,
                        &mut rust_iy,
                        K,
                        N as i32,
                    )
                };
                let c_yy = unsafe {
                    op_pvq_search_sse2(c_X.as_mut_ptr(), c_iy.as_mut_ptr(), K, N as i32, 4)
                };

                let yy_diff = rust_yy.to_bits() != c_yy.to_bits();
                let iy_diff = rust_iy != c_iy;
                if yy_diff || iy_diff {
                    if mismatches < 5 {
                        eprintln!(
                            "op_pvq_search_sse2 MISMATCH: N={N} K={K} trial={trial}\n  rust_yy={rust_yy:.6} c_yy={c_yy:.6}\n  rust_iy={rust_iy:?}\n  c_iy   ={c_iy:?}"
                        );
                    }
                    mismatches += 1;
                }
            }
        }
    }
    if mismatches > 0 {
        panic!("op_pvq_search_sse2: {mismatches} mismatches");
    }
}

#[test]
fn compare_celt_pitch_xcorr_avx2() {
    if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
        return;
    }
    let mut rng = Rng::new(777);
    let mut mismatches = 0;

    for len in [18, 24, 48, 64, 120, 240, 480, 960] {
        for max_pitch in [4, 8, 16, 24, 32, 64, 120, 240] {
            let x: Vec<f32> = (0..len).map(|_| rng.next_f32()).collect();
            let y: Vec<f32> = (0..len + max_pitch).map(|_| rng.next_f32()).collect();

            let mut rust_xcorr = vec![0.0f32; max_pitch];
            unsafe {
                opurs::celt::simd::x86::celt_pitch_xcorr_avx2(&x, &y, &mut rust_xcorr, len);
            }

            let mut c_xcorr = vec![0.0f32; max_pitch];
            unsafe {
                celt_pitch_xcorr_avx2(
                    x.as_ptr(),
                    y.as_ptr(),
                    c_xcorr.as_mut_ptr(),
                    len as i32,
                    max_pitch as i32,
                    4,
                );
            }

            let any_diff = (0..max_pitch).any(|i| rust_xcorr[i].to_bits() != c_xcorr[i].to_bits());
            if any_diff {
                if mismatches < 5 {
                    let diffs: Vec<usize> = (0..max_pitch)
                        .filter(|&i| rust_xcorr[i].to_bits() != c_xcorr[i].to_bits())
                        .collect();
                    eprintln!(
                        "celt_pitch_xcorr_avx2 MISMATCH: len={len} max_pitch={max_pitch} diff_indices={diffs:?}\n  first: rust={:.10e} c={:.10e}",
                        rust_xcorr[diffs[0]], c_xcorr[diffs[0]],
                    );
                }
                mismatches += 1;
            }
        }
    }
    if mismatches > 0 {
        panic!("celt_pitch_xcorr_avx2: {mismatches} mismatches out of 64 tests");
    }
}
