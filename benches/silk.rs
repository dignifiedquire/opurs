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
        group.bench_with_input(BenchmarkId::new("scalar", order), &order, |b, &order| {
            b.iter(|| {
                black_box(
                    opurs::internals::silk_noise_shape_quantizer_short_prediction_c(
                        &buf32,
                        &coef16,
                        order as i32,
                    ),
                )
            })
        });
        group.bench_with_input(BenchmarkId::new("dispatch", order), &order, |b, &order| {
            let arch = opurs::internals::opus_select_arch();
            b.iter(|| {
                black_box(
                    opurs::internals::silk_noise_shape_quantizer_short_prediction(
                        &buf32,
                        &coef16,
                        order as i32,
                        arch,
                    ),
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
                black_box(opurs::internals::silk_inner_prod_aligned_scale(
                    &v1, &v2, 4, n as i32,
                ))
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
        group.bench_with_input(BenchmarkId::new("scalar", n), &n, |b, &_n| {
            b.iter(|| black_box(opurs::internals::silk_inner_product_FLP_scalar(&d1, &d2)))
        });
        group.bench_with_input(BenchmarkId::new("dispatch", n), &n, |b, &_n| {
            let arch = opurs::internals::opus_select_arch();
            b.iter(|| black_box(opurs::internals::silk_inner_product_FLP(&d1, &d2, arch)))
        });
    }
    group.finish();
}

fn bench_feedback_loop(c: &mut Criterion) {
    let mut group = c.benchmark_group("silk_feedback_loop");
    for &order in &[10, 16] {
        let coef = generate_i16_signal(order, 42);
        group.bench_with_input(BenchmarkId::new("scalar", order), &order, |b, &order| {
            let mut data1 = generate_i32_signal(order, 123);
            b.iter(|| {
                black_box(opurs::internals::silk_NSQ_noise_shape_feedback_loop_c(
                    black_box(12345),
                    &mut data1,
                    &coef,
                    order as i32,
                ))
            })
        });
        group.bench_with_input(BenchmarkId::new("dispatch", order), &order, |b, &order| {
            let arch = opurs::internals::opus_select_arch();
            let mut data1 = generate_i32_signal(order, 123);
            b.iter(|| {
                black_box(opurs::internals::silk_NSQ_noise_shape_feedback_loop(
                    black_box(12345),
                    &mut data1,
                    &coef,
                    order as i32,
                    arch,
                ))
            })
        });
    }
    group.finish();
}

fn bench_vq_wmat_ec(c: &mut Criterion) {
    #[cfg(feature = "simd")]
    let arch = opurs::internals::opus_select_arch();
    let mut group = c.benchmark_group("silk_VQ_WMat_EC");
    // LTP_ORDER = 5, typical codebook sizes
    let ltp_order = 5;
    for &l in &[8, 32] {
        let xx_q17 = generate_i32_signal(ltp_order * ltp_order, 42);
        let xx_q17: Vec<i32> = xx_q17.iter().map(|&v| v.abs() >> 8).collect();
        let x_x_q17 = generate_i32_signal(ltp_order, 123);
        let cb_q7: Vec<i8> = generate_i16_signal(l * ltp_order, 77)
            .iter()
            .map(|&v| (v >> 8) as i8)
            .collect();
        let cb_gain_q7: Vec<u8> = (0..l).map(|i| (40 + i) as u8).collect();
        let cl_q5: Vec<u8> = (0..l).map(|i| (10 + i * 2) as u8).collect();
        let label = format!("L{}", l);

        group.bench_with_input(BenchmarkId::new("scalar", &label), &l, |b, &_l| {
            b.iter(|| {
                let mut ind: i8 = 0;
                let mut res_nrg_q15: i32 = 0;
                let mut rate_dist_q8: i32 = 0;
                let mut gain_q7: i32 = 0;
                opurs::internals::silk_VQ_WMat_EC_c(
                    &mut ind,
                    &mut res_nrg_q15,
                    &mut rate_dist_q8,
                    &mut gain_q7,
                    black_box(&xx_q17),
                    black_box(&x_x_q17),
                    &cb_q7,
                    &cb_gain_q7,
                    &cl_q5,
                    ltp_order as i32,
                    127,
                    l as i32,
                );
                black_box((ind, res_nrg_q15, rate_dist_q8, gain_q7))
            })
        });

        #[cfg(feature = "simd")]
        group.bench_with_input(BenchmarkId::new("dispatch", &label), &l, |b, &_l| {
            b.iter(|| {
                let mut ind: i8 = 0;
                let mut res_nrg_q15: i32 = 0;
                let mut rate_dist_q8: i32 = 0;
                let mut gain_q7: i32 = 0;
                opurs::internals::silk_VQ_WMat_EC(
                    &mut ind,
                    &mut res_nrg_q15,
                    &mut rate_dist_q8,
                    &mut gain_q7,
                    black_box(&xx_q17),
                    black_box(&x_x_q17),
                    &cb_q7,
                    &cb_gain_q7,
                    &cl_q5,
                    ltp_order as i32,
                    127,
                    l as i32,
                    arch,
                );
                black_box((ind, res_nrg_q15, rate_dist_q8, gain_q7))
            })
        });
    }
    group.finish();
}

#[cfg(feature = "simd")]
fn bench_vad_energy(c: &mut Criterion) {
    let arch = opurs::internals::opus_select_arch();
    let mut group = c.benchmark_group("silk_vad_energy");
    for &n in &[240, 480, 960] {
        let x = generate_i16_signal(n, 42);
        group.bench_with_input(BenchmarkId::new("scalar", n), &n, |b, &_n| {
            b.iter(|| black_box(opurs::internals::silk_vad_energy_scalar(black_box(&x))))
        });
        group.bench_with_input(BenchmarkId::new("dispatch", n), &n, |b, &_n| {
            b.iter(|| black_box(opurs::internals::silk_vad_energy(black_box(&x), arch)))
        });
    }
    group.finish();
}

fn bench_lpc_inverse_pred_gain(c: &mut Criterion) {
    let mut group = c.benchmark_group("silk_LPC_inverse_pred_gain");
    for &order in &[10, 16] {
        // Generate stable LPC coefficients (small values in Q12)
        let a_q12: Vec<i16> = generate_i16_signal(order, 42)
            .iter()
            .map(|&v| v / 16) // keep small to avoid instability
            .collect();
        group.bench_with_input(BenchmarkId::new("scalar", order), &order, |b, &_order| {
            b.iter(|| {
                black_box(opurs::internals::silk_LPC_inverse_pred_gain_c(black_box(
                    &a_q12,
                )))
            })
        });
        #[cfg(feature = "simd")]
        group.bench_with_input(BenchmarkId::new("dispatch", order), &order, |b, &_order| {
            let arch = opurs::internals::opus_select_arch();
            b.iter(|| {
                black_box(opurs::internals::silk_LPC_inverse_pred_gain(
                    black_box(&a_q12),
                    arch,
                ))
            })
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_short_prediction,
    bench_inner_prod_aligned_scale,
    bench_inner_product_flp,
    bench_feedback_loop,
    bench_vq_wmat_ec,
    bench_lpc_inverse_pred_gain,
);

#[cfg(feature = "simd")]
criterion_group!(simd_benches, bench_vad_energy,);

#[cfg(feature = "simd")]
criterion_main!(benches, simd_benches);
#[cfg(not(feature = "simd"))]
criterion_main!(benches);
