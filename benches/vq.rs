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
            let arch = opurs::internals::opus_select_arch();
            b.iter(|| black_box(opurs::internals::celt_inner_prod(&x, &y, n, arch)))
        });
    }
    group.finish();
}

fn bench_op_pvq_search(c: &mut Criterion) {
    let arch = opurs::internals::opus_select_arch();
    let mut group = c.benchmark_group("op_pvq_search");
    for &(n, k) in &[(8, 4), (16, 8), (32, 16), (64, 32)] {
        let label = format!("N{}_K{}", n, k);
        group.bench_with_input(BenchmarkId::new("scalar", &label), &(n, k), |b, &(n, k)| {
            b.iter(|| {
                let mut x = generate_f32_signal(n, 42);
                let mut iy = vec![0i32; n];
                black_box(opurs::internals::op_pvq_search_c(
                    &mut x, &mut iy, k, n as i32, arch,
                ))
            })
        });
        #[cfg(feature = "simd")]
        group.bench_with_input(
            BenchmarkId::new("dispatch", &label),
            &(n, k),
            |b, &(n, k)| {
                b.iter(|| {
                    let mut x = generate_f32_signal(n, 42);
                    let mut iy = vec![0i32; n];
                    black_box(opurs::internals::op_pvq_search(
                        &mut x, &mut iy, k, n as i32, arch,
                    ))
                })
            },
        );
    }
    group.finish();
}

criterion_group!(benches, bench_celt_inner_prod_vq, bench_op_pvq_search);
criterion_main!(benches);
