use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

fn generate_f32_signal(len: usize, seed: u32) -> Vec<f32> {
    let mut v = Vec::with_capacity(len);
    let mut state = seed;
    for _ in 0..len {
        state = state.wrapping_mul(1103515245).wrapping_add(12345);
        v.push((state as i32 >> 16) as f32 / 32768.0);
    }
    v
}

fn bench_celt_inner_prod_vq(c: &mut Criterion) {
    // VQ uses celt_inner_prod extensively — benchmark at VQ-typical sizes
    let mut group = c.benchmark_group("vq_inner_prod");
    for &n in &[8, 16, 32, 64, 128] {
        let x = generate_f32_signal(n, 42);
        let y = generate_f32_signal(n, 123);
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, &n| {
            b.iter(|| black_box(opurs::internals::celt_inner_prod(&x, &y, n)))
        });
    }
    group.finish();
}

criterion_group!(benches, bench_celt_inner_prod_vq);
criterion_main!(benches);
