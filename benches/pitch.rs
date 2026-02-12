use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

fn generate_signal(len: usize, seed: u32) -> Vec<f32> {
    let mut v = Vec::with_capacity(len);
    let mut state = seed;
    for _ in 0..len {
        // Simple PRNG for reproducible benchmarks
        state = state.wrapping_mul(1103515245).wrapping_add(12345);
        v.push((state as i32 >> 16) as f32 / 32768.0);
    }
    v
}

fn bench_xcorr_kernel(c: &mut Criterion) {
    let mut group = c.benchmark_group("xcorr_kernel");
    for &n in &[64, 240, 480, 960] {
        let x = generate_signal(n, 42);
        let y = generate_signal(n + 3, 123);
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, &n| {
            b.iter(|| {
                let mut sum = [0.0f32; 4];
                opurs::internals::xcorr_kernel(
                    black_box(&x[..n]),
                    black_box(&y),
                    black_box(&mut sum),
                    n,
                );
                sum
            })
        });
    }
    group.finish();
}

fn bench_celt_inner_prod(c: &mut Criterion) {
    let mut group = c.benchmark_group("celt_inner_prod");
    for &n in &[64, 240, 480, 960] {
        let x = generate_signal(n, 42);
        let y = generate_signal(n, 123);
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, &n| {
            b.iter(|| opurs::internals::celt_inner_prod(black_box(&x), black_box(&y), n))
        });
    }
    group.finish();
}

fn bench_dual_inner_prod(c: &mut Criterion) {
    let mut group = c.benchmark_group("dual_inner_prod");
    for &n in &[64, 240, 480, 960] {
        let x = generate_signal(n, 42);
        let y01 = generate_signal(n, 123);
        let y02 = generate_signal(n, 456);
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, &n| {
            b.iter(|| {
                opurs::internals::dual_inner_prod(
                    black_box(&x),
                    black_box(&y01),
                    black_box(&y02),
                    n,
                )
            })
        });
    }
    group.finish();
}

fn bench_celt_pitch_xcorr(c: &mut Criterion) {
    let mut group = c.benchmark_group("celt_pitch_xcorr");
    for &(len, max_pitch) in &[(240, 60), (480, 120), (960, 240)] {
        let x = generate_signal(len, 42);
        let y = generate_signal(len + max_pitch, 123);
        let label = format!("{}x{}", len, max_pitch);
        group.bench_with_input(
            BenchmarkId::new("size", &label),
            &(len, max_pitch),
            |b, &(len, max_pitch)| {
                let mut xcorr = vec![0.0f32; max_pitch];
                b.iter(|| {
                    opurs::internals::celt_pitch_xcorr(
                        black_box(&x[..len]),
                        black_box(&y),
                        black_box(&mut xcorr),
                        len,
                    );
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
);
criterion_main!(benches);
