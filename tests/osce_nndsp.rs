//! OSCE nndsp building block tests: compare Rust vs C with deterministic inputs.
//!
//! Ports the logic from upstream `dnn/adaconvtest.c` — exercises adaconv, adacomb,
//! and adashape with compiled-in LACE/NoLACE weights and deterministic PRNG inputs,
//! then verifies bit-exact match between the Rust and C implementations.

#![cfg(feature = "tools-dnn")]

use opurs::dnn::nndsp::*;
use opurs::dnn::osce::*;
use opurs::dnn::weights::compiled_weights;

/// Same xorshift32 PRNG as the C test harness.
struct Prng(u32);

impl Prng {
    fn new(seed: u32) -> Self {
        Self(seed)
    }

    fn next_float(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 17;
        self.0 ^= self.0 << 5;
        ((self.0 & 0xFFFF) as f32 / 32768.0) - 1.0
    }
}

/// Get C reference output for adaconv.
fn c_adaconv(use_nolace: i32, num_frames: i32, seed: u32, out_size: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; out_size];
    unsafe {
        libopus_sys::osce_test_adaconv(out.as_mut_ptr(), use_nolace, num_frames, seed);
    }
    out
}

/// Get C reference output for adacomb.
fn c_adacomb(use_nolace: i32, num_frames: i32, seed: u32, out_size: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; out_size];
    unsafe {
        libopus_sys::osce_test_adacomb(out.as_mut_ptr(), use_nolace, num_frames, seed);
    }
    out
}

/// Get C reference output for adashape.
fn c_adashape(num_frames: i32, seed: u32, out_size: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; out_size];
    unsafe {
        libopus_sys::osce_test_adashape(out.as_mut_ptr(), num_frames, seed);
    }
    out
}

/// Run Rust adaconv with same PRNG inputs as C harness.
fn rust_adaconv(
    lace: Option<&LACE>,
    nolace: Option<&NoLACE>,
    use_nolace: i32,
    num_frames: usize,
    seed: u32,
) -> Vec<f32> {
    let (
        kernel_layer,
        gain_layer,
        feature_dim,
        frame_size,
        overlap_size,
        in_channels,
        out_channels,
        kernel_size,
        left_padding,
        filter_gain_a,
        filter_gain_b,
        shape_gain,
        window,
    ) = match use_nolace {
        0 => {
            let l = lace.unwrap();
            (
                &l.layers.af1_kernel,
                &l.layers.af1_gain,
                LACE_COND_DIM,
                LACE_FRAME_SIZE,
                LACE_OVERLAP_SIZE,
                LACE_AF1_IN_CHANNELS,
                LACE_AF1_OUT_CHANNELS,
                LACE_AF1_KERNEL_SIZE,
                LACE_AF1_LEFT_PADDING,
                LACE_AF1_FILTER_GAIN_A,
                LACE_AF1_FILTER_GAIN_B,
                LACE_AF1_SHAPE_GAIN,
                lace.unwrap().window.as_slice(),
            )
        }
        1 => {
            let n = nolace.unwrap();
            (
                &n.layers.af1_kernel,
                &n.layers.af1_gain,
                NOLACE_COND_DIM,
                NOLACE_FRAME_SIZE,
                NOLACE_OVERLAP_SIZE,
                NOLACE_AF1_IN_CHANNELS,
                NOLACE_AF1_OUT_CHANNELS,
                NOLACE_AF1_KERNEL_SIZE,
                NOLACE_AF1_LEFT_PADDING,
                NOLACE_AF1_FILTER_GAIN_A,
                NOLACE_AF1_FILTER_GAIN_B,
                NOLACE_AF1_SHAPE_GAIN,
                nolace.unwrap().window.as_slice(),
            )
        }
        2 => {
            let n = nolace.unwrap();
            (
                &n.layers.af2_kernel,
                &n.layers.af2_gain,
                NOLACE_COND_DIM,
                NOLACE_FRAME_SIZE,
                NOLACE_OVERLAP_SIZE,
                NOLACE_AF2_IN_CHANNELS,
                NOLACE_AF2_OUT_CHANNELS,
                NOLACE_AF2_KERNEL_SIZE,
                NOLACE_AF2_LEFT_PADDING,
                NOLACE_AF2_FILTER_GAIN_A,
                NOLACE_AF2_FILTER_GAIN_B,
                NOLACE_AF2_SHAPE_GAIN,
                nolace.unwrap().window.as_slice(),
            )
        }
        3 => {
            let n = nolace.unwrap();
            (
                &n.layers.af4_kernel,
                &n.layers.af4_gain,
                NOLACE_COND_DIM,
                NOLACE_FRAME_SIZE,
                NOLACE_OVERLAP_SIZE,
                NOLACE_AF4_IN_CHANNELS,
                NOLACE_AF4_OUT_CHANNELS,
                NOLACE_AF4_KERNEL_SIZE,
                NOLACE_AF4_LEFT_PADDING,
                NOLACE_AF4_FILTER_GAIN_A,
                NOLACE_AF4_FILTER_GAIN_B,
                NOLACE_AF4_SHAPE_GAIN,
                nolace.unwrap().window.as_slice(),
            )
        }
        _ => panic!("invalid use_nolace"),
    };

    let mut state = AdaConvState::default();
    let mut prng = Prng::new(seed);
    let mut all_out = Vec::with_capacity(num_frames * frame_size * out_channels);

    for _ in 0..num_frames {
        let features: Vec<f32> = (0..feature_dim).map(|_| prng.next_float() * 0.1).collect();
        let x_in: Vec<f32> = (0..frame_size * in_channels)
            .map(|_| prng.next_float() * 0.5)
            .collect();

        let mut x_out = vec![0.0f32; frame_size * out_channels];
        adaconv_process_frame(
            &mut state,
            &mut x_out,
            &x_in,
            &features,
            kernel_layer,
            gain_layer,
            feature_dim,
            frame_size,
            overlap_size,
            in_channels,
            out_channels,
            kernel_size,
            left_padding,
            filter_gain_a,
            filter_gain_b,
            shape_gain,
            window,
        );
        all_out.extend_from_slice(&x_out);
    }
    all_out
}

/// Run Rust adacomb with same PRNG inputs as C harness.
fn rust_adacomb(lace: &LACE, num_frames: usize, seed: u32) -> Vec<f32> {
    let mut state = AdaCombState::default();
    let mut prng = Prng::new(seed);
    let frame_size = LACE_FRAME_SIZE;
    let feature_dim = LACE_COND_DIM;
    let mut all_out = Vec::with_capacity(num_frames * frame_size);

    for _ in 0..num_frames {
        let features: Vec<f32> = (0..feature_dim).map(|_| prng.next_float() * 0.1).collect();
        let x_in: Vec<f32> = (0..frame_size).map(|_| prng.next_float() * 0.5).collect();
        let pitch_lag = {
            let v = prng.next_float() * 32768.0 + 32768.0;
            LACE_CF1_KERNEL_SIZE as i32 + ((v as u32) % (250 - LACE_CF1_KERNEL_SIZE as u32)) as i32
        };

        let mut x_out = vec![0.0f32; frame_size];
        adacomb_process_frame(
            &mut state,
            &mut x_out,
            &x_in,
            &features,
            &lace.layers.cf1_kernel,
            &lace.layers.cf1_gain,
            &lace.layers.cf1_global_gain,
            pitch_lag,
            feature_dim,
            frame_size,
            LACE_OVERLAP_SIZE,
            LACE_CF1_KERNEL_SIZE,
            LACE_CF1_LEFT_PADDING,
            LACE_CF1_FILTER_GAIN_A,
            LACE_CF1_FILTER_GAIN_B,
            LACE_CF1_LOG_GAIN_LIMIT,
            &lace.window,
        );
        all_out.extend_from_slice(&x_out);
    }
    all_out
}

/// Run Rust adashape with same PRNG inputs as C harness.
fn rust_adashape(nolace: &NoLACE, num_frames: usize, seed: u32) -> Vec<f32> {
    let mut state = AdaShapeState::default();
    let mut prng = Prng::new(seed);
    let frame_size = NOLACE_TDSHAPE1_FRAME_SIZE;
    let feature_dim = NOLACE_TDSHAPE1_FEATURE_DIM;
    let mut all_out = Vec::with_capacity(num_frames * frame_size);

    for _ in 0..num_frames {
        let features: Vec<f32> = (0..feature_dim).map(|_| prng.next_float() * 0.1).collect();
        let x_in: Vec<f32> = (0..frame_size).map(|_| prng.next_float() * 0.5).collect();

        let mut x_out = vec![0.0f32; frame_size];
        adashape_process_frame(
            &mut state,
            &mut x_out,
            &x_in,
            &features,
            &nolace.layers.tdshape1_alpha1_f,
            &nolace.layers.tdshape1_alpha1_t,
            &nolace.layers.tdshape1_alpha2,
            feature_dim,
            frame_size,
            NOLACE_TDSHAPE1_AVG_POOL_K,
        );
        all_out.extend_from_slice(&x_out);
    }
    all_out
}

fn compare_outputs(name: &str, rust: &[f32], c: &[f32]) {
    assert_eq!(
        rust.len(),
        c.len(),
        "{name}: output length mismatch: rust={} c={}",
        rust.len(),
        c.len()
    );

    let mut max_diff: f32 = 0.0;
    let mut first_diff = None;
    for (i, (&r, &c_val)) in rust.iter().zip(c.iter()).enumerate() {
        let diff = (r - c_val).abs();
        if diff > 0.0 && first_diff.is_none() {
            first_diff = Some((i, r, c_val, diff));
        }
        max_diff = max_diff.max(diff);
    }

    if let Some((idx, r, c_val, diff)) = first_diff {
        panic!(
            "{name}: MISMATCH at sample {idx}: rust={r} c={c_val} diff={diff} max_diff={max_diff}"
        );
    }
}

const NUM_FRAMES: usize = 5;
const SEED: u32 = 12345;

#[test]
fn test_adaconv_lace_af1() {
    let arrays = compiled_weights();
    let lace = init_lace(&arrays).expect("LACE init failed");
    let frame_size = LACE_FRAME_SIZE;
    let out_channels = LACE_AF1_OUT_CHANNELS;
    let out_size = NUM_FRAMES * frame_size * out_channels;

    let c_out = c_adaconv(0, NUM_FRAMES as i32, SEED, out_size);
    let rust_out = rust_adaconv(Some(&lace), None, 0, NUM_FRAMES, SEED);
    compare_outputs("lace_af1", &rust_out, &c_out);
}

#[test]
fn test_adaconv_nolace_af1() {
    let arrays = compiled_weights();
    let nolace = init_nolace(&arrays).expect("NoLACE init failed");
    let out_size = NUM_FRAMES * NOLACE_FRAME_SIZE * NOLACE_AF1_OUT_CHANNELS;

    let c_out = c_adaconv(1, NUM_FRAMES as i32, SEED, out_size);
    let rust_out = rust_adaconv(None, Some(&nolace), 1, NUM_FRAMES, SEED);
    compare_outputs("nolace_af1", &rust_out, &c_out);
}

#[test]
fn test_adaconv_nolace_af2() {
    let arrays = compiled_weights();
    let nolace = init_nolace(&arrays).expect("NoLACE init failed");
    let out_size = NUM_FRAMES * NOLACE_FRAME_SIZE * NOLACE_AF2_OUT_CHANNELS;

    let c_out = c_adaconv(2, NUM_FRAMES as i32, SEED, out_size);
    let rust_out = rust_adaconv(None, Some(&nolace), 2, NUM_FRAMES, SEED);
    compare_outputs("nolace_af2", &rust_out, &c_out);
}

#[test]
fn test_adaconv_nolace_af4() {
    let arrays = compiled_weights();
    let nolace = init_nolace(&arrays).expect("NoLACE init failed");
    let out_size = NUM_FRAMES * NOLACE_FRAME_SIZE * NOLACE_AF4_OUT_CHANNELS;

    let c_out = c_adaconv(3, NUM_FRAMES as i32, SEED, out_size);
    let rust_out = rust_adaconv(None, Some(&nolace), 3, NUM_FRAMES, SEED);
    compare_outputs("nolace_af4", &rust_out, &c_out);
}

#[test]
fn test_adacomb_lace_cf1() {
    let arrays = compiled_weights();
    let lace = init_lace(&arrays).expect("LACE init failed");
    let out_size = NUM_FRAMES * LACE_FRAME_SIZE;

    let c_out = c_adacomb(0, NUM_FRAMES as i32, SEED, out_size);
    let rust_out = rust_adacomb(&lace, NUM_FRAMES, SEED);
    compare_outputs("lace_cf1", &rust_out, &c_out);
}

#[test]
fn test_adashape_nolace_tdshape1() {
    let arrays = compiled_weights();
    let nolace = init_nolace(&arrays).expect("NoLACE init failed");
    let out_size = NUM_FRAMES * NOLACE_TDSHAPE1_FRAME_SIZE;

    let c_out = c_adashape(NUM_FRAMES as i32, SEED, out_size);
    let rust_out = rust_adashape(&nolace, NUM_FRAMES, SEED);
    compare_outputs("nolace_tdshape1", &rust_out, &c_out);
}

/// Test compute_linear in isolation on the LACE af1_kernel layer.
#[test]
fn test_compute_linear_lace_af1_kernel() {
    use opurs::dnn::nnet::compute_linear;

    let arrays = compiled_weights();
    let lace = init_lace(&arrays).expect("LACE init failed");

    // Get C reference
    let mut c_out = vec![0.0f32; 512];
    let nb_inputs =
        unsafe { libopus_sys::osce_test_compute_linear(c_out.as_mut_ptr(), SEED) } as usize;
    let nb_outputs = lace.layers.af1_kernel.nb_outputs;
    c_out.truncate(nb_outputs);

    // Generate same PRNG inputs
    let mut prng = Prng::new(SEED);
    let input: Vec<f32> = (0..nb_inputs).map(|_| prng.next_float() * 0.1).collect();

    // Run Rust compute_linear
    let mut rust_out = vec![0.0f32; nb_outputs];
    compute_linear(&lace.layers.af1_kernel, &mut rust_out, &input);

    compare_outputs("compute_linear_lace_af1_kernel", &rust_out, &c_out);
}

/// Test compute_generic_dense with ACTIVATION_TANH on LACE af1_gain layer.
#[test]
fn test_dense_tanh_lace_af1_gain() {
    use opurs::dnn::nnet::compute_generic_dense;

    let arrays = compiled_weights();
    let lace = init_lace(&arrays).expect("LACE init failed");

    let mut c_out = vec![0.0f32; 512];
    let nb_inputs = unsafe { libopus_sys::osce_test_dense_tanh(c_out.as_mut_ptr(), SEED) } as usize;
    let nb_outputs = lace.layers.af1_gain.nb_outputs;
    c_out.truncate(nb_outputs);

    let mut prng = Prng::new(SEED);
    let input: Vec<f32> = (0..nb_inputs).map(|_| prng.next_float() * 0.1).collect();

    let mut rust_out = vec![0.0f32; nb_outputs];
    compute_generic_dense(
        &lace.layers.af1_gain,
        &mut rust_out,
        &input,
        opurs::dnn::nnet::ACTIVATION_TANH,
    );

    compare_outputs("dense_tanh_lace_af1_gain", &rust_out, &c_out);
}

/// Diagnostic: check if compute_linear on gain layer diverges, or if tanh_approx does.
#[test]
fn test_diag_gain_linear_vs_tanh() {
    use opurs::dnn::nnet::{compute_generic_dense, compute_linear, ACTIVATION_TANH};
    use opurs::dnn::vec::tanh_approx;

    let arrays = compiled_weights();
    let lace = init_lace(&arrays).expect("LACE init failed");

    let nb_inputs = lace.layers.af1_gain.nb_inputs;
    let nb_outputs = lace.layers.af1_gain.nb_outputs;

    let mut prng = Prng::new(SEED);
    let input: Vec<f32> = (0..nb_inputs).map(|_| prng.next_float() * 0.1).collect();

    // compute_linear only
    let mut linear_out = vec![0.0f32; nb_outputs];
    compute_linear(&lace.layers.af1_gain, &mut linear_out, &input);

    // tanh on top
    let tanh_out: Vec<f32> = linear_out.iter().map(|&v| tanh_approx(v)).collect();

    // compute_generic_dense (linear + tanh together)
    let mut dense_out = vec![0.0f32; nb_outputs];
    compute_generic_dense(
        &lace.layers.af1_gain,
        &mut dense_out,
        &input,
        ACTIVATION_TANH,
    );

    // C reference
    let mut c_out = vec![0.0f32; 512];
    unsafe { libopus_sys::osce_test_dense_tanh(c_out.as_mut_ptr(), SEED) };
    c_out.truncate(nb_outputs);

    eprintln!("gain layer: nb_inputs={nb_inputs} nb_outputs={nb_outputs}");
    eprintln!("linear_out = {linear_out:?}");
    eprintln!("tanh_out   = {tanh_out:?}");
    eprintln!("dense_out  = {dense_out:?}");
    eprintln!("c_out      = {c_out:?}");
    eprintln!(
        "tanh_out vs dense_out diff = {}",
        (tanh_out[0] - dense_out[0]).abs()
    );
    eprintln!(
        "tanh_out vs c_out diff     = {}",
        (tanh_out[0] - c_out[0]).abs()
    );

    // Print layer properties
    eprintln!(
        "gain layer weights: int8={} float={} sparse_idx={}",
        lace.layers.af1_gain.weights.len(),
        lace.layers.af1_gain.float_weights.len(),
        lace.layers.af1_gain.weights_idx.len(),
    );
    eprintln!(
        "gain layer bias={} subias={} scale={} diag={}",
        lace.layers.af1_gain.bias.len(),
        lace.layers.af1_gain.subias.len(),
        lace.layers.af1_gain.scale.len(),
        lace.layers.af1_gain.diag.len(),
    );
    eprintln!(
        "kernel layer weights: int8={} float={} sparse_idx={}",
        lace.layers.af1_kernel.weights.len(),
        lace.layers.af1_kernel.float_weights.len(),
        lace.layers.af1_kernel.weights_idx.len(),
    );

    // Dump first few float weights and bias
    eprintln!(
        "gain float_weights[0..8] = {:?}",
        &lace.layers.af1_gain.float_weights[..8]
    );
    eprintln!("gain bias = {:?}", &lace.layers.af1_gain.bias);
    eprintln!("input[0..8] = {:?}", &input[..8]);

    // C compute_linear on gain layer (no activation)
    let mut c_linear = vec![0.0f32; 8];
    unsafe { libopus_sys::osce_test_compute_linear_gain(c_linear.as_mut_ptr(), SEED) };
    eprintln!(
        "c_linear[0]    = {:e} (0x{:08x})",
        c_linear[0],
        c_linear[0].to_bits()
    );
    eprintln!(
        "rust_linear[0] = {:e} (0x{:08x})",
        linear_out[0],
        linear_out[0].to_bits()
    );
    eprintln!(
        "c_tanh[0]      = {:e} (0x{:08x})",
        c_out[0],
        c_out[0].to_bits()
    );
    eprintln!(
        "rust_tanh[0]   = {:e} (0x{:08x})",
        tanh_out[0],
        tanh_out[0].to_bits()
    );

    // Direct tanh_approx comparison
    let test_val: f32 = -0.07047112;
    let mut c_tanh_direct = [0.0f32; 2];
    unsafe { libopus_sys::osce_test_tanh_approx(c_tanh_direct.as_mut_ptr(), test_val) };
    let rust_tanh_direct = tanh_approx(test_val);
    eprintln!(
        "direct tanh_approx({:e} / 0x{:08x}):",
        test_val,
        test_val.to_bits()
    );
    eprintln!(
        "  C:    {:e} (0x{:08x})",
        c_tanh_direct[0],
        c_tanh_direct[0].to_bits()
    );
    eprintln!(
        "  Rust: {:e} (0x{:08x})",
        rust_tanh_direct,
        rust_tanh_direct.to_bits()
    );
    eprintln!(
        "  C echoed input: {:e} (0x{:08x})",
        c_tanh_direct[1],
        c_tanh_direct[1].to_bits()
    );

    // Also test with the exact bit pattern from compute_linear
    let exact_val = linear_out[0];
    unsafe { libopus_sys::osce_test_tanh_approx(c_tanh_direct.as_mut_ptr(), exact_val) };
    let rust_tanh_exact = tanh_approx(exact_val);
    eprintln!(
        "exact tanh_approx({:e} / 0x{:08x}):",
        exact_val,
        exact_val.to_bits()
    );
    eprintln!(
        "  C:    {:e} (0x{:08x})",
        c_tanh_direct[0],
        c_tanh_direct[0].to_bits()
    );
    eprintln!(
        "  Rust: {:e} (0x{:08x})",
        rust_tanh_exact,
        rust_tanh_exact.to_bits()
    );

    // Verify our tanh_out matches dense_out (sanity)
    assert_eq!(tanh_out, dense_out, "our own tanh path should match dense");
}
