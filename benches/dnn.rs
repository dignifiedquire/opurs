//! Benchmarks for DNN vector primitives: scalar vs SIMD dispatch.
//!
//! Run with: `cargo bench --features deep-plc --bench dnn`

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

fn generate_weights_i8(n: usize, seed: u32) -> Vec<i8> {
    let mut v = Vec::with_capacity(n);
    let mut state = seed;
    for _ in 0..n {
        state = state.wrapping_mul(1103515245).wrapping_add(12345);
        v.push(((state >> 16) as i32 % 128) as i8);
    }
    v
}

fn bench_vec_tanh(c: &mut Criterion) {
    let arch = opurs::internals::opus_select_arch();
    let mut group = c.benchmark_group("dnn_vec_tanh");
    for &n in &[64, 128, 256, 512, 1024] {
        let x = generate_signal(n, 42);

        group.bench_with_input(BenchmarkId::new("scalar", n), &n, |b, &n| {
            let mut y = vec![0.0f32; n];
            b.iter(|| {
                opurs::internals::vec_tanh_scalar(&mut y, black_box(&x));
                black_box(&y);
            })
        });

        group.bench_with_input(BenchmarkId::new("dispatch", n), &n, |b, &n| {
            let mut y = vec![0.0f32; n];
            b.iter(|| {
                opurs::internals::vec_tanh(&mut y, black_box(&x), arch);
                black_box(&y);
            })
        });
    }
    group.finish();
}

fn bench_vec_sigmoid(c: &mut Criterion) {
    let arch = opurs::internals::opus_select_arch();
    let mut group = c.benchmark_group("dnn_vec_sigmoid");
    for &n in &[64, 128, 256, 512, 1024] {
        let x = generate_signal(n, 123);

        group.bench_with_input(BenchmarkId::new("scalar", n), &n, |b, &n| {
            let mut y = vec![0.0f32; n];
            b.iter(|| {
                opurs::internals::vec_sigmoid_scalar(&mut y, black_box(&x));
                black_box(&y);
            })
        });

        group.bench_with_input(BenchmarkId::new("dispatch", n), &n, |b, &n| {
            let mut y = vec![0.0f32; n];
            b.iter(|| {
                opurs::internals::vec_sigmoid(&mut y, black_box(&x), arch);
                black_box(&y);
            })
        });
    }
    group.finish();
}

fn bench_softmax(c: &mut Criterion) {
    let arch = opurs::internals::opus_select_arch();
    let mut group = c.benchmark_group("dnn_softmax");
    for &n in &[64, 128, 256, 512, 1024] {
        let x = generate_signal(n, 77);

        group.bench_with_input(BenchmarkId::new("scalar", n), &n, |b, &n| {
            let mut y = vec![0.0f32; n];
            b.iter(|| {
                opurs::internals::softmax_scalar(&mut y, black_box(&x));
                black_box(&y);
            })
        });

        group.bench_with_input(BenchmarkId::new("dispatch", n), &n, |b, &n| {
            let mut y = vec![0.0f32; n];
            b.iter(|| {
                opurs::internals::softmax(&mut y, black_box(&x), arch);
                black_box(&y);
            })
        });
    }
    group.finish();
}

fn bench_sgemv(c: &mut Criterion) {
    let arch = opurs::internals::opus_select_arch();
    let mut group = c.benchmark_group("dnn_sgemv");
    for &(rows, cols) in &[(64, 64), (128, 64), (256, 128), (512, 256)] {
        let weights = generate_signal(rows * cols, 42);
        let x = generate_signal(cols, 123);
        let label = format!("{}x{}", rows, cols);

        group.bench_with_input(
            BenchmarkId::new("scalar", &label),
            &(rows, cols),
            |b, &(rows, cols)| {
                let mut out = vec![0.0f32; rows];
                b.iter(|| {
                    opurs::internals::sgemv_scalar(
                        &mut out,
                        black_box(&weights),
                        rows,
                        cols,
                        rows,
                        black_box(&x),
                    );
                    black_box(&out);
                })
            },
        );

        group.bench_with_input(
            BenchmarkId::new("dispatch", &label),
            &(rows, cols),
            |b, &(rows, cols)| {
                let mut out = vec![0.0f32; rows];
                b.iter(|| {
                    opurs::internals::sgemv(
                        &mut out,
                        black_box(&weights),
                        rows,
                        cols,
                        rows,
                        black_box(&x),
                        arch,
                    );
                    black_box(&out);
                })
            },
        );
    }
    group.finish();
}

fn bench_cgemv8x4(c: &mut Criterion) {
    let arch = opurs::internals::opus_select_arch();
    let mut group = c.benchmark_group("dnn_cgemv8x4");
    for &(rows, cols) in &[(64, 64), (128, 64), (256, 128), (512, 256)] {
        let w = generate_weights_i8(rows * cols, 42);
        let scale: Vec<f32> = (0..rows).map(|i| 0.01 + 0.001 * i as f32).collect();
        let x = generate_signal(cols, 99);
        let label = format!("{}x{}", rows, cols);

        group.bench_with_input(
            BenchmarkId::new("scalar", &label),
            &(rows, cols),
            |b, &(rows, cols)| {
                let mut out = vec![0.0f32; rows];
                b.iter(|| {
                    opurs::internals::cgemv8x4_scalar(
                        &mut out,
                        black_box(&w),
                        &scale,
                        rows,
                        cols,
                        black_box(&x),
                    );
                    black_box(&out);
                })
            },
        );

        group.bench_with_input(
            BenchmarkId::new("dispatch", &label),
            &(rows, cols),
            |b, &(rows, cols)| {
                let mut out = vec![0.0f32; rows];
                b.iter(|| {
                    opurs::internals::cgemv8x4(
                        &mut out,
                        black_box(&w),
                        &scale,
                        rows,
                        cols,
                        black_box(&x),
                        arch,
                    );
                    black_box(&out);
                })
            },
        );
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_vec_tanh,
    bench_vec_sigmoid,
    bench_softmax,
    bench_sgemv,
    bench_cgemv8x4,
);
criterion_main!(benches);
