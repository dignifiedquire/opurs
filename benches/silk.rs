use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

fn generate_i32_signal(len: usize, seed: u32) -> Vec<i32> {
    let mut v = Vec::with_capacity(len);
    let mut state = seed;
    for _ in 0..len {
        state = state.wrapping_mul(1103515245).wrapping_add(12345);
        v.push((state as i32) >> 8);
    }
    v
}

fn generate_i16_signal(len: usize, seed: u32) -> Vec<i16> {
    let mut v = Vec::with_capacity(len);
    let mut state = seed;
    for _ in 0..len {
        state = state.wrapping_mul(1103515245).wrapping_add(12345);
        v.push((state >> 16) as i16);
    }
    v
}

fn generate_f32_signal(len: usize, seed: u32) -> Vec<f32> {
    let mut v = Vec::with_capacity(len);
    let mut state = seed;
    for _ in 0..len {
        state = state.wrapping_mul(1103515245).wrapping_add(12345);
        v.push((state as i32 >> 16) as f32 / 32768.0);
    }
    v
}

fn bench_short_prediction(c: &mut Criterion) {
    let mut group = c.benchmark_group("silk_short_prediction");
    for &order in &[10, 16] {
        let buf32 = generate_i32_signal(order + 16, 42);
        let coef16 = generate_i16_signal(order, 123);
        group.bench_with_input(BenchmarkId::new("order", order), &order, |b, &order| {
            b.iter(|| {
                opurs::internals::silk_noise_shape_quantizer_short_prediction_c(
                    black_box(&buf32),
                    black_box(&coef16),
                    order as i32,
                )
            })
        });
    }
    group.finish();
}

fn bench_inner_prod_aligned_scale(c: &mut Criterion) {
    let mut group = c.benchmark_group("silk_inner_prod_aligned_scale");
    for &n in &[64, 240, 480] {
        let v1 = generate_i16_signal(n, 42);
        let v2 = generate_i16_signal(n, 123);
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, &n| {
            b.iter(|| {
                opurs::internals::silk_inner_prod_aligned_scale(
                    black_box(&v1),
                    black_box(&v2),
                    black_box(4),
                    n as i32,
                )
            })
        });
    }
    group.finish();
}

fn bench_inner_product_flp(c: &mut Criterion) {
    let mut group = c.benchmark_group("silk_inner_product_FLP");
    for &n in &[64, 240, 480, 960] {
        let d1 = generate_f32_signal(n, 42);
        let d2 = generate_f32_signal(n, 123);
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, &_n| {
            b.iter(|| opurs::internals::silk_inner_product_FLP(black_box(&d1), black_box(&d2)))
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_short_prediction,
    bench_inner_prod_aligned_scale,
    bench_inner_product_flp,
);
criterion_main!(benches);
