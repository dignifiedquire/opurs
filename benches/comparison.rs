//! Benchmarks comparing Rust implementations against C reference (libopus-sys).
//!
//! When compiled with default features (simd enabled), the C reference is also
//! compiled with SIMD (SSE/AVX2 on x86, NEON on aarch64) via RTCD dispatch.
//!
//! Requires the `tools` feature to link against libopus-sys.
//! Run with: `cargo bench --features tools --bench comparison`

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
#[cfg(all(feature = "simd", target_arch = "aarch64"))]
use libopus_sys::CELT_PITCH_XCORR_IMPL;
use libopus_sys::{celt_pitch_xcorr_c, silk_inner_product_FLP_c};
#[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
use libopus_sys::{PITCH_XCORR_IMPL, SILK_INNER_PRODUCT_FLP_IMPL};

fn generate_signal(len: usize, seed: u32) -> Vec<f32> {
    let mut v = Vec::with_capacity(len);
    let mut state = seed;
    for _ in 0..len {
        state = state.wrapping_mul(1103515245).wrapping_add(12345);
        v.push((state as i32 >> 16) as f32 / 32768.0);
    }
    v
}

fn bench_pitch_xcorr_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("celt_pitch_xcorr_cmp");
    for &(len, max_pitch) in &[(240, 60), (480, 120), (960, 240)] {
        let x = generate_signal(len, 42);
        let y = generate_signal(len + max_pitch, 123);
        let label = format!("{}x{}", len, max_pitch);

        group.bench_with_input(
            BenchmarkId::new("rust_scalar", &label),
            &(len, max_pitch),
            |b, &(len, max_pitch)| {
                let mut xcorr = vec![0.0f32; max_pitch];
                b.iter(|| {
                    opurs::internals::celt_pitch_xcorr_scalar(&x[..len], &y, &mut xcorr, len);
                    black_box(&xcorr);
                })
            },
        );

        group.bench_with_input(
            BenchmarkId::new("rust_dispatch", &label),
            &(len, max_pitch),
            |b, &(len, max_pitch)| {
                let arch = opurs::internals::opus_select_arch();
                let mut xcorr = vec![0.0f32; max_pitch];
                b.iter(|| {
                    opurs::internals::celt_pitch_xcorr(&x[..len], &y, &mut xcorr, len, arch);
                    black_box(&xcorr);
                })
            },
        );

        group.bench_with_input(
            BenchmarkId::new("c_scalar", &label),
            &(len, max_pitch),
            |b, &(len, max_pitch)| {
                let mut xcorr = vec![0.0f32; max_pitch];
                b.iter(|| {
                    unsafe {
                        celt_pitch_xcorr_c(
                            x.as_ptr(),
                            y.as_ptr(),
                            xcorr.as_mut_ptr(),
                            len as i32,
                            max_pitch as i32,
                            0,
                        );
                    }
                    black_box(&xcorr);
                })
            },
        );

        // C SIMD dispatch (uses best available: AVX2 > SSE > scalar)
        #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
        group.bench_with_input(
            BenchmarkId::new("c_simd", &label),
            &(len, max_pitch),
            |b, &(len, max_pitch)| {
                let arch = unsafe { libopus_sys::opus_select_arch() };
                let mut xcorr = vec![0.0f32; max_pitch];
                b.iter(|| {
                    unsafe {
                        PITCH_XCORR_IMPL[(arch as usize) & 7](
                            x.as_ptr(),
                            y.as_ptr(),
                            xcorr.as_mut_ptr(),
                            len as i32,
                            max_pitch as i32,
                            arch,
                        );
                    }
                    black_box(&xcorr);
                })
            },
        );

        // aarch64: C RTCD dispatch table (NEON/DOTPROD arch levels map here).
        #[cfg(all(feature = "simd", target_arch = "aarch64"))]
        group.bench_with_input(
            BenchmarkId::new("c_simd", &label),
            &(len, max_pitch),
            |b, &(len, max_pitch)| {
                let arch = unsafe { libopus_sys::opus_select_arch() };
                let mut xcorr = vec![0.0f32; max_pitch];
                b.iter(|| {
                    unsafe {
                        CELT_PITCH_XCORR_IMPL[(arch as usize) & 7](
                            x.as_ptr(),
                            y.as_ptr(),
                            xcorr.as_mut_ptr(),
                            len as i32,
                            max_pitch as i32,
                            arch,
                        );
                    }
                    black_box(&xcorr);
                })
            },
        );
    }
    group.finish();
}

fn bench_silk_inner_product_flp_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("silk_inner_product_FLP_cmp");
    for &n in &[64, 240, 480, 960] {
        let d1 = generate_signal(n, 42);
        let d2 = generate_signal(n, 123);

        group.bench_with_input(BenchmarkId::new("rust_scalar", n), &n, |b, &_n| {
            b.iter(|| black_box(opurs::internals::silk_inner_product_FLP_scalar(&d1, &d2)))
        });

        group.bench_with_input(BenchmarkId::new("rust_dispatch", n), &n, |b, &_n| {
            let arch = opurs::internals::opus_select_arch();
            b.iter(|| black_box(opurs::internals::silk_inner_product_FLP(&d1, &d2, arch)))
        });

        group.bench_with_input(BenchmarkId::new("c_scalar", n), &n, |b, &n| {
            b.iter(|| unsafe {
                black_box(silk_inner_product_FLP_c(d1.as_ptr(), d2.as_ptr(), n as i32))
            })
        });

        // C SIMD dispatch (AVX2 on x86 when available)
        #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
        group.bench_with_input(BenchmarkId::new("c_simd", n), &n, |b, &n| {
            let arch = unsafe { libopus_sys::opus_select_arch() };
            b.iter(|| unsafe {
                black_box(SILK_INNER_PRODUCT_FLP_IMPL[(arch as usize) & 7](
                    d1.as_ptr(),
                    d2.as_ptr(),
                    n as i32,
                ))
            })
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_pitch_xcorr_comparison,
    bench_silk_inner_product_flp_comparison,
);
criterion_main!(benches);
