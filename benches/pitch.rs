use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

fn generate_signal(len: usize, seed: u32) -> Vec<f32> {
    let mut v = Vec::with_capacity(len);
    let mut state = seed;
    for _ in 0..len {
        state = state.wrapping_mul(1103515245).wrapping_add(12345);
        v.push((state as i32 >> 16) as f32 / 32768.0);
    }
    v
}

fn bench_xcorr_kernel(c: &mut Criterion) {
    let arch = opurs::internals::opus_select_arch();
    let mut group = c.benchmark_group("xcorr_kernel");
    for &n in &[64, 240, 480, 960] {
        let x = generate_signal(n, 42);
        let y = generate_signal(n + 3, 123);
        group.bench_with_input(BenchmarkId::new("scalar", n), &n, |b, &n| {
            b.iter(|| {
                let mut sum = [0.0f32; 4];
                opurs::internals::xcorr_kernel_scalar(&x[..n], &y, &mut sum, n);
                black_box(sum)
            })
        });
        group.bench_with_input(BenchmarkId::new("dispatch", n), &n, |b, &n| {
            b.iter(|| {
                let mut sum = [0.0f32; 4];
                opurs::internals::xcorr_kernel(&x[..n], &y, &mut sum, n, arch);
                black_box(sum)
            })
        });
    }
    group.finish();
}

fn bench_celt_inner_prod(c: &mut Criterion) {
    let arch = opurs::internals::opus_select_arch();
    let mut group = c.benchmark_group("celt_inner_prod");
    for &n in &[64, 240, 480, 960] {
        let x = generate_signal(n, 42);
        let y = generate_signal(n, 123);
        group.bench_with_input(BenchmarkId::new("scalar", n), &n, |b, &n| {
            b.iter(|| black_box(opurs::internals::celt_inner_prod_scalar(&x, &y, n)))
        });
        group.bench_with_input(BenchmarkId::new("dispatch", n), &n, |b, &n| {
            b.iter(|| black_box(opurs::internals::celt_inner_prod(&x, &y, n, arch)))
        });
    }
    group.finish();
}

fn bench_dual_inner_prod(c: &mut Criterion) {
    let arch = opurs::internals::opus_select_arch();
    let mut group = c.benchmark_group("dual_inner_prod");
    for &n in &[64, 240, 480, 960] {
        let x = generate_signal(n, 42);
        let y01 = generate_signal(n, 123);
        let y02 = generate_signal(n, 456);
        group.bench_with_input(BenchmarkId::new("scalar", n), &n, |b, &n| {
            b.iter(|| black_box(opurs::internals::dual_inner_prod_scalar(&x, &y01, &y02, n)))
        });
        group.bench_with_input(BenchmarkId::new("dispatch", n), &n, |b, &n| {
            b.iter(|| black_box(opurs::internals::dual_inner_prod(&x, &y01, &y02, n, arch)))
        });
    }
    group.finish();
}

fn bench_celt_pitch_xcorr(c: &mut Criterion) {
    let arch = opurs::internals::opus_select_arch();
    let mut group = c.benchmark_group("celt_pitch_xcorr");
    for &(len, max_pitch) in &[(240, 60), (480, 120), (960, 240)] {
        let x = generate_signal(len, 42);
        let y = generate_signal(len + max_pitch, 123);
        let label = format!("{}x{}", len, max_pitch);
        group.bench_with_input(
            BenchmarkId::new("scalar", &label),
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
            BenchmarkId::new("dispatch", &label),
            &(len, max_pitch),
            |b, &(len, max_pitch)| {
                let mut xcorr = vec![0.0f32; max_pitch];
                b.iter(|| {
                    opurs::internals::celt_pitch_xcorr(&x[..len], &y, &mut xcorr, len, arch);
                    black_box(&xcorr);
                })
            },
        );
    }
    group.finish();
}

fn bench_comb_filter_const(c: &mut Criterion) {
    let arch = opurs::internals::opus_select_arch();
    let mut group = c.benchmark_group("comb_filter_const");
    for &(n, t) in &[(120, 15), (480, 40), (960, 100)] {
        let total = n + t + 2;
        let x = generate_signal(total, 42);
        let label = format!("N{}_T{}", n, t);
        group.bench_with_input(BenchmarkId::new("scalar", &label), &(n, t), |b, &(n, t)| {
            let mut y = x.clone();
            let start = t + 2;
            b.iter(|| {
                opurs::internals::comb_filter_const_c(
                    &mut y,
                    start,
                    black_box(&x),
                    start,
                    t as i32,
                    n as i32,
                    0.3,
                    0.2,
                    0.1,
                );
                black_box(&y);
            })
        });
        #[cfg(feature = "simd")]
        group.bench_with_input(
            BenchmarkId::new("dispatch", &label),
            &(n, t),
            |b, &(n, t)| {
                let mut y = x.clone();
                let start = t + 2;
                b.iter(|| {
                    opurs::internals::comb_filter_const(
                        &mut y,
                        start,
                        black_box(&x),
                        start,
                        t as i32,
                        n as i32,
                        0.3,
                        0.2,
                        0.1,
                        arch,
                    );
                    black_box(&y);
                })
            },
        );
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_xcorr_kernel,
    bench_celt_inner_prod,
    bench_dual_inner_prod,
    bench_celt_pitch_xcorr,
    bench_comb_filter_const,
);
criterion_main!(benches);
