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

    // Both C and Rust now use matching SIMD paths on all platforms:
    // aarch64 NEON and x86 AVX2 scalar functions broadcast-and-extract
    // to match C's behavior. Results should be bit-exact everywhere.
    let tolerance: f32 = 0.0;

    let mut max_diff: f32 = 0.0;
    let mut first_diff = None;
    for (i, (&r, &c_val)) in rust.iter().zip(c.iter()).enumerate() {
        let diff = (r - c_val).abs();
        if diff > tolerance && first_diff.is_none() {
            first_diff = Some((i, r, c_val, diff));
        }
        max_diff = max_diff.max(diff);
    }

    if let Some((idx, r, c_val, diff)) = first_diff {
        panic!(
            "{name}: MISMATCH at sample {idx}: rust={r} c={c_val} diff={diff} max_diff={max_diff} tolerance={tolerance}"
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
    let frame_size = NOLACE_TDSHAPE1_FRAME_SIZE;

    // Test 1 frame first
    let c_out_1 = c_adashape(1, SEED, frame_size);
    let rust_out_1 = rust_adashape(&nolace, 1, SEED);
    compare_outputs("nolace_tdshape1_1frame", &rust_out_1, &c_out_1);

    // Then test all frames
    let out_size = NUM_FRAMES * frame_size;
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

/// Test celt_pitch_xcorr in isolation to check for NEON bit-exactness.
/// Uses same PRNG-generated kernel and signal data, len=16 (ADACONV_MAX_KERNEL_SIZE).
#[test]
fn test_celt_pitch_xcorr_neon() {
    use opurs::celt::pitch::celt_pitch_xcorr;

    let max_pitch = 40; // typical overlap_size
    let len = ADACONV_MAX_KERNEL_SIZE;

    // C reference
    let mut c_out = vec![0.0f32; max_pitch];
    unsafe {
        libopus_sys::osce_test_celt_pitch_xcorr(c_out.as_mut_ptr(), max_pitch as i32, SEED);
    }

    // Rust: generate same PRNG inputs
    let mut prng = Prng::new(SEED);
    let kernel: Vec<f32> = (0..len).map(|_| prng.next_float() * 0.1).collect();
    let signal: Vec<f32> = (0..max_pitch + len)
        .map(|_| prng.next_float() * 0.5)
        .collect();

    let mut rust_out = vec![0.0f32; max_pitch];
    celt_pitch_xcorr(&kernel, &signal, &mut rust_out, len);

    compare_outputs("celt_pitch_xcorr_neon", &rust_out, &c_out);
}

/// Test compute_linear on NoLACE tdshape1_alpha1_f layer.
/// Isolates whether the divergence in nolace_tdshape1 starts at the linear layer.
#[test]
fn test_compute_linear_nolace_tdshape() {
    use opurs::dnn::nnet::compute_linear;

    let arrays = compiled_weights();
    let nolace = init_nolace(&arrays).expect("NoLACE init failed");

    let layer = &nolace.layers.tdshape1_alpha1_f;
    let nb_inputs = layer.nb_inputs;
    let nb_outputs = layer.nb_outputs;

    // C reference
    let mut c_out = vec![0.0f32; 2048];
    let c_nb_inputs =
        unsafe { libopus_sys::osce_test_compute_linear_nolace_tdshape(c_out.as_mut_ptr(), SEED) }
            as usize;
    assert_eq!(c_nb_inputs, nb_inputs);
    c_out.truncate(nb_outputs);

    // Rust
    let mut prng = Prng::new(SEED);
    let input: Vec<f32> = (0..nb_inputs).map(|_| prng.next_float() * 0.1).collect();
    let mut rust_out = vec![0.0f32; nb_outputs];
    compute_linear(layer, &mut rust_out, &input);

    compare_outputs("compute_linear_nolace_tdshape", &rust_out, &c_out);
}

/// Test compute_linear on NoLACE af2_kernel layer.
/// Isolates whether the divergence in nolace_af2 starts at the linear layer.
#[test]
fn test_compute_linear_nolace_af2() {
    use opurs::dnn::nnet::compute_linear;

    let arrays = compiled_weights();
    let nolace = init_nolace(&arrays).expect("NoLACE init failed");

    let layer = &nolace.layers.af2_kernel;
    let nb_inputs = layer.nb_inputs;
    let nb_outputs = layer.nb_outputs;

    // C reference
    let mut c_out = vec![0.0f32; 2048];
    let c_nb_inputs =
        unsafe { libopus_sys::osce_test_compute_linear_nolace_af2(c_out.as_mut_ptr(), SEED) }
            as usize;
    assert_eq!(c_nb_inputs, nb_inputs);
    c_out.truncate(nb_outputs);

    // Rust
    let mut prng = Prng::new(SEED);
    let input: Vec<f32> = (0..nb_inputs).map(|_| prng.next_float() * 0.1).collect();
    let mut rust_out = vec![0.0f32; nb_outputs];
    compute_linear(layer, &mut rust_out, &input);

    compare_outputs("compute_linear_nolace_af2", &rust_out, &c_out);
}

/// Diagnostic: dump adashape intermediates to find where divergence starts.
#[test]
#[allow(clippy::needless_range_loop, clippy::excessive_precision)]
fn test_adashape_intermediates() {
    use opurs::dnn::nndsp::*;
    use opurs::dnn::nnet::*;

    let arrays = compiled_weights();
    let nolace = init_nolace(&arrays).expect("NoLACE init failed");
    let frame_size = NOLACE_TDSHAPE1_FRAME_SIZE; // 80
    let feature_dim = NOLACE_TDSHAPE1_FEATURE_DIM; // 160
    let avg_pool_k = NOLACE_TDSHAPE1_AVG_POOL_K; // 4

    // Get C intermediates
    let mut c_buf = vec![0.0f32; 2 * frame_size];
    unsafe { libopus_sys::osce_test_adashape_intermediates(c_buf.as_mut_ptr(), SEED) };
    let c_out_buffer = &c_buf[..frame_size];
    let c_x_out = &c_buf[frame_size..2 * frame_size];

    // Run Rust step by step
    let mut prng = Prng::new(SEED);
    let features: Vec<f32> = (0..feature_dim).map(|_| prng.next_float() * 0.1).collect();
    let x_in: Vec<f32> = (0..frame_size).map(|_| prng.next_float() * 0.5).collect();

    let tenv_size = frame_size / avg_pool_k;
    let mut in_buffer = vec![0.0f32; ADASHAPE_MAX_INPUT_DIM + ADASHAPE_MAX_FRAME_SIZE];
    let mut out_buffer = vec![0.0f32; ADASHAPE_MAX_FRAME_SIZE];
    let mut tmp_buffer = vec![0.0f32; ADASHAPE_MAX_FRAME_SIZE];

    in_buffer[..feature_dim].copy_from_slice(&features);
    let tenv = &mut in_buffer[feature_dim..];
    tenv[..tenv_size + 1].fill(0.0);
    let mut mean = 0.0f32;
    for i in 0..tenv_size {
        for k in 0..avg_pool_k {
            tenv[i] += x_in[i * avg_pool_k + k].abs();
        }
        tenv[i] = ((tenv[i] / avg_pool_k as f32 + 1.52587890625e-05f32) as f64).ln() as f32;
        mean += tenv[i];
    }
    mean /= tenv_size as f32;
    for i in 0..tenv_size {
        tenv[i] -= mean;
    }
    tenv[tenv_size] = mean;

    let mut state = AdaShapeState::default();
    compute_generic_conv1d(
        &nolace.layers.tdshape1_alpha1_f,
        &mut out_buffer,
        &mut state.conv_alpha1f_state,
        &in_buffer,
        feature_dim,
        ACTIVATION_LINEAR,
    );
    compute_generic_conv1d(
        &nolace.layers.tdshape1_alpha1_t,
        &mut tmp_buffer,
        &mut state.conv_alpha1t_state,
        &in_buffer[feature_dim..],
        tenv_size + 1,
        ACTIVATION_LINEAR,
    );

    // Check alpha1f output
    eprintln!("=== alpha1f out_buffer[0..8] ===");
    for i in 0..8 {
        eprintln!(
            "  [{}] rust={:e} (0x{:08x})",
            i,
            out_buffer[i],
            out_buffer[i].to_bits()
        );
    }

    // Check alpha1t output
    eprintln!("=== alpha1t tmp_buffer[0..8] ===");
    for i in 0..8 {
        eprintln!(
            "  [{}] rust={:e} (0x{:08x})",
            i,
            tmp_buffer[i],
            tmp_buffer[i].to_bits()
        );
    }

    // Leaky ReLU
    for i in 0..frame_size {
        let tmp = out_buffer[i] + tmp_buffer[i];
        in_buffer[i] = if tmp >= 0.0 { tmp } else { 0.2 * tmp };
    }

    eprintln!("=== leaky_relu in_buffer[0..8] ===");
    for i in 0..8 {
        eprintln!(
            "  [{}] rust={:e} (0x{:08x})",
            i,
            in_buffer[i],
            in_buffer[i].to_bits()
        );
    }

    compute_generic_conv1d(
        &nolace.layers.tdshape1_alpha2,
        &mut out_buffer,
        &mut state.conv_alpha2_state,
        &in_buffer,
        frame_size,
        ACTIVATION_LINEAR,
    );

    // Compare out_buffer (pre-exp)
    eprintln!("=== out_buffer (pre-exp) ===");
    for i in 0..frame_size.min(24) {
        eprintln!(
            "  [{}] rust={:e} (0x{:08x}) c={:e} (0x{:08x}) diff={:e}",
            i,
            out_buffer[i],
            out_buffer[i].to_bits(),
            c_out_buffer[i],
            c_out_buffer[i].to_bits(),
            (out_buffer[i] - c_out_buffer[i]).abs()
        );
    }

    // Compare final x_out
    let mut x_out = vec![0.0f32; frame_size];
    for i in 0..frame_size {
        x_out[i] = ((out_buffer[i] as f64).exp() * x_in[i] as f64) as f32;
    }

    eprintln!("=== x_out (final) ===");
    for i in 0..frame_size.min(24) {
        eprintln!(
            "  [{}] rust={:e} (0x{:08x}) c={:e} (0x{:08x}) diff={:e}",
            i,
            x_out[i],
            x_out[i].to_bits(),
            c_x_out[i],
            c_x_out[i].to_bits(),
            (x_out[i] - c_x_out[i]).abs()
        );
    }
}

/// Diagnostic: check if compute_linear on gain layer diverges, or if tanh_approx does.
#[test]
fn test_diag_gain_linear_vs_tanh() {
    use opurs::dnn::nnet::{compute_generic_dense, compute_linear, ACTIVATION_TANH};
    use opurs::dnn::simd::tanh_approx;

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

    // Verify tanh_approx matches C tanh_approx.
    // Both C and Rust now use matching SIMD paths on all platforms:
    // aarch64 NEON and x86 AVX2 scalar functions broadcast-and-extract
    // to match C's behavior. Results should be bit-exact everywhere.
    assert_eq!(
        rust_tanh_exact.to_bits(),
        c_tanh_direct[0].to_bits(),
        "tanh_approx should be bit-exact with C: rust={rust_tanh_exact} c={}",
        c_tanh_direct[0]
    );

    // Note: tanh_out (from tanh_approx/Padé) may differ from dense_out
    // (from vec_tanh scalar tail using lpcnet_exp) for n=1 on NEON.
    // The formulas are intentionally different in C's vec_neon.h.
    // What matters is that dense_out matches C dense_tanh, verified by
    // test_dense_tanh_lace_af1_gain.
}

/// Test compute_linear on LACE fnet_conv2 layer (int8 weights, exercises cgemv8x4).
#[test]
fn test_compute_linear_int8_lace_fnet_conv2() {
    use opurs::dnn::nnet::compute_linear;

    let arrays = compiled_weights();
    let lace = init_lace(&arrays).expect("LACE init failed");

    let mut c_out = vec![0.0f32; 512];
    let nb_inputs =
        unsafe { libopus_sys::osce_test_compute_linear_int8(c_out.as_mut_ptr(), SEED) } as usize;
    let nb_outputs = lace.layers.fnet_conv2.nb_outputs;
    c_out.truncate(nb_outputs);

    let mut prng = Prng::new(SEED);
    let input: Vec<f32> = (0..nb_inputs).map(|_| prng.next_float() * 0.1).collect();

    let mut rust_out = vec![0.0f32; nb_outputs];
    compute_linear(&lace.layers.fnet_conv2, &mut rust_out, &input);

    compare_outputs("compute_linear_int8_lace_fnet_conv2", &rust_out, &c_out);
}

/// Test compute_generic_gru on LACE fnet GRU (int8 weights, 2 steps).
#[test]
fn test_gru_lace_fnet() {
    use opurs::dnn::nnet::compute_generic_gru;
    use opurs::dnn::osce::LACE_COND_DIM;

    let arrays = compiled_weights();
    let lace = init_lace(&arrays).expect("LACE init failed");

    let mut c_out = vec![0.0f32; 2 * LACE_COND_DIM];
    unsafe { libopus_sys::osce_test_gru_lace_fnet(c_out.as_mut_ptr(), SEED) };

    let mut prng = Prng::new(SEED);
    let mut state = vec![0.0f32; LACE_COND_DIM];

    // Step 1
    let input1: Vec<f32> = (0..LACE_COND_DIM)
        .map(|_| prng.next_float() * 0.1)
        .collect();
    compute_generic_gru(
        &lace.layers.fnet_gru_input,
        &lace.layers.fnet_gru_recurrent,
        &mut state,
        &input1,
    );
    compare_outputs("gru_lace_fnet_step1", &state, &c_out[..LACE_COND_DIM]);

    // Step 2
    let input2: Vec<f32> = (0..LACE_COND_DIM)
        .map(|_| prng.next_float() * 0.1)
        .collect();
    compute_generic_gru(
        &lace.layers.fnet_gru_input,
        &lace.layers.fnet_gru_recurrent,
        &mut state,
        &input2,
    );
    compare_outputs("gru_lace_fnet_step2", &state, &c_out[LACE_COND_DIM..]);
}

/// Test compute_generic_dense with ACTIVATION_TANH on LACE fnet_tconv (int8 weights, 128->512).
#[test]
fn test_dense_tanh_lace_tconv() {
    use opurs::dnn::nnet::{compute_generic_dense, ACTIVATION_TANH};

    let arrays = compiled_weights();
    let lace = init_lace(&arrays).expect("LACE init failed");

    let mut c_out = vec![0.0f32; 512];
    let nb_inputs =
        unsafe { libopus_sys::osce_test_dense_tanh_lace_tconv(c_out.as_mut_ptr(), SEED) } as usize;
    let nb_outputs = lace.layers.fnet_tconv.nb_outputs;
    c_out.truncate(nb_outputs);

    let mut prng = Prng::new(SEED);
    let input: Vec<f32> = (0..nb_inputs).map(|_| prng.next_float() * 0.1).collect();

    let mut rust_out = vec![0.0f32; nb_outputs];
    compute_generic_dense(
        &lace.layers.fnet_tconv,
        &mut rust_out,
        &input,
        ACTIVATION_TANH,
    );

    compare_outputs("dense_tanh_lace_tconv", &rust_out, &c_out);
}

/// Diagnostic: dump adacomb intermediates to find where divergence starts.
/// Compares each step of the adacomb pipeline between Rust and C.
#[test]
fn test_adacomb_intermediates() {
    use opurs::dnn::nndsp::*;
    use opurs::dnn::nnet::{
        compute_generic_dense, compute_linear, ACTIVATION_LINEAR, ACTIVATION_RELU, ACTIVATION_TANH,
    };

    let arrays = compiled_weights();
    let lace = init_lace(&arrays).expect("LACE init failed");

    // Get C intermediates
    let mut c_buf = vec![0.0f32; 256];
    unsafe { libopus_sys::osce_test_adacomb_intermediates(c_buf.as_mut_ptr(), SEED) };

    // Generate same PRNG inputs as C
    let mut prng = Prng::new(SEED);
    let feature_dim = LACE_COND_DIM; // 128
    let frame_size = LACE_FRAME_SIZE; // 80
    let kernel_size = LACE_CF1_KERNEL_SIZE; // 16

    let features: Vec<f32> = (0..feature_dim).map(|_| prng.next_float() * 0.1).collect();
    let _x_in: Vec<f32> = (0..frame_size).map(|_| prng.next_float() * 0.5).collect();
    let _pitch_lag = {
        let v = prng.next_float() * 32768.0 + 32768.0;
        kernel_size as i32 + ((v as u32) % (250 - kernel_size as u32)) as i32
    };

    // Step 1: compute_generic_dense on cf1_kernel (128 -> 16, ACTIVATION_LINEAR)
    let mut rust_kernel = vec![0.0f32; 16];
    compute_generic_dense(
        &lace.layers.cf1_kernel,
        &mut rust_kernel,
        &features,
        ACTIVATION_LINEAR,
    );
    let c_kernel = &c_buf[0..16];
    eprintln!("=== Step 1: kernel_buffer (compute_generic_dense cf1_kernel) ===");
    let mut kernel_ok = true;
    for i in 0..16 {
        let diff = (rust_kernel[i] - c_kernel[i]).abs();
        if diff > 0.0 {
            eprintln!(
                "  [{}] DIFF rust={:e} (0x{:08x}) c={:e} (0x{:08x}) diff={:e}",
                i,
                rust_kernel[i],
                rust_kernel[i].to_bits(),
                c_kernel[i],
                c_kernel[i].to_bits(),
                diff
            );
            kernel_ok = false;
        }
    }
    if kernel_ok {
        eprintln!("  ALL MATCH");
    }

    // Step 2: compute_generic_dense on cf1_gain (128 -> 1, ACTIVATION_RELU)
    let mut rust_gain = [0.0f32; 1];
    compute_generic_dense(
        &lace.layers.cf1_gain,
        &mut rust_gain,
        &features,
        ACTIVATION_RELU,
    );
    eprintln!("=== Step 2: gain (dense RELU cf1_gain) ===");
    eprintln!(
        "  rust={:e} (0x{:08x}) c={:e} (0x{:08x}) diff={:e}",
        rust_gain[0],
        rust_gain[0].to_bits(),
        c_buf[16],
        c_buf[16].to_bits(),
        (rust_gain[0] - c_buf[16]).abs()
    );

    // Step 2b: compute_linear alone on cf1_gain (to isolate linear vs activation)
    let mut rust_linear_gain = [0.0f32; 1];
    compute_linear(&lace.layers.cf1_gain, &mut rust_linear_gain, &features);
    eprintln!("=== Step 2b: linear-only gain (cf1_gain) ===");
    eprintln!(
        "  rust_linear={:e} (0x{:08x}) c_linear={:e} (0x{:08x}) diff={:e}",
        rust_linear_gain[0],
        rust_linear_gain[0].to_bits(),
        c_buf[116],
        c_buf[116].to_bits(),
        (rust_linear_gain[0] - c_buf[116]).abs()
    );

    // Step 3: compute_generic_dense on cf1_global_gain (128 -> 1, ACTIVATION_TANH)
    let mut rust_global_gain = [0.0f32; 1];
    compute_generic_dense(
        &lace.layers.cf1_global_gain,
        &mut rust_global_gain,
        &features,
        ACTIVATION_TANH,
    );
    eprintln!("=== Step 3: global_gain (dense TANH cf1_global_gain) ===");
    eprintln!(
        "  rust={:e} (0x{:08x}) c={:e} (0x{:08x}) diff={:e}",
        rust_global_gain[0],
        rust_global_gain[0].to_bits(),
        c_buf[17],
        c_buf[17].to_bits(),
        (rust_global_gain[0] - c_buf[17]).abs()
    );

    // Step 3b: compute_linear alone on cf1_global_gain
    let mut rust_linear_ggain = [0.0f32; 1];
    compute_linear(
        &lace.layers.cf1_global_gain,
        &mut rust_linear_ggain,
        &features,
    );
    eprintln!("=== Step 3b: linear-only global_gain (cf1_global_gain) ===");
    eprintln!(
        "  rust_linear={:e} (0x{:08x}) c_linear={:e} (0x{:08x}) diff={:e}",
        rust_linear_ggain[0],
        rust_linear_ggain[0].to_bits(),
        c_buf[117],
        c_buf[117].to_bits(),
        (rust_linear_ggain[0] - c_buf[117]).abs()
    );

    // Step 4: transform gains
    let gain_transformed = ((LACE_CF1_LOG_GAIN_LIMIT - rust_gain[0]) as f64).exp() as f32;
    let global_gain_transformed = ((LACE_CF1_FILTER_GAIN_A * rust_global_gain[0]
        + LACE_CF1_FILTER_GAIN_B) as f64)
        .exp() as f32;
    eprintln!("=== Step 4: transformed gains ===");
    eprintln!(
        "  gain: rust={:e} (0x{:08x}) c={:e} (0x{:08x}) diff={:e}",
        gain_transformed,
        gain_transformed.to_bits(),
        c_buf[18],
        c_buf[18].to_bits(),
        (gain_transformed - c_buf[18]).abs()
    );
    eprintln!(
        "  ggain: rust={:e} (0x{:08x}) c={:e} (0x{:08x}) diff={:e}",
        global_gain_transformed,
        global_gain_transformed.to_bits(),
        c_buf[19],
        c_buf[19].to_bits(),
        (global_gain_transformed - c_buf[19]).abs()
    );

    // Step 5: scale_kernel
    let mut scaled_kernel = rust_kernel.clone();
    scale_kernel(&mut scaled_kernel, 1, 1, kernel_size, &[gain_transformed]);
    eprintln!("=== Step 5: scaled kernel ===");
    let c_scaled = &c_buf[20..36];
    let mut scaled_ok = true;
    for i in 0..16 {
        let diff = (scaled_kernel[i] - c_scaled[i]).abs();
        if diff > 0.0 {
            eprintln!(
                "  [{}] DIFF rust={:e} (0x{:08x}) c={:e} (0x{:08x}) diff={:e}",
                i,
                scaled_kernel[i],
                scaled_kernel[i].to_bits(),
                c_scaled[i],
                c_scaled[i].to_bits(),
                diff
            );
            scaled_ok = false;
        }
    }
    if scaled_ok {
        eprintln!("  ALL MATCH");
    }

    // Step 6: compare final output (first 24 samples)
    let c_final = &c_buf[36..36 + frame_size];
    // Run full Rust adacomb
    let rust_final = rust_adacomb(&lace, 1, SEED);
    eprintln!("=== Step 6: final adacomb output (first 24 samples) ===");
    for i in 0..24.min(frame_size) {
        let diff = (rust_final[i] - c_final[i]).abs();
        if diff > 0.0 {
            eprintln!(
                "  [{}] DIFF rust={:e} (0x{:08x}) c={:e} (0x{:08x}) diff={:e}",
                i,
                rust_final[i],
                rust_final[i].to_bits(),
                c_final[i],
                c_final[i].to_bits(),
                diff
            );
        }
    }

    // Layer properties for debugging
    eprintln!("=== Layer properties ===");
    eprintln!(
        "  cf1_kernel: in={} out={} float_w={} int8_w={} sparse={}",
        lace.layers.cf1_kernel.nb_inputs,
        lace.layers.cf1_kernel.nb_outputs,
        lace.layers.cf1_kernel.float_weights.len(),
        lace.layers.cf1_kernel.weights.len(),
        lace.layers.cf1_kernel.weights_idx.len(),
    );
    eprintln!(
        "  cf1_gain: in={} out={} float_w={} int8_w={} sparse={}",
        lace.layers.cf1_gain.nb_inputs,
        lace.layers.cf1_gain.nb_outputs,
        lace.layers.cf1_gain.float_weights.len(),
        lace.layers.cf1_gain.weights.len(),
        lace.layers.cf1_gain.weights_idx.len(),
    );
    eprintln!(
        "  cf1_global_gain: in={} out={} float_w={} int8_w={} sparse={}",
        lace.layers.cf1_global_gain.nb_inputs,
        lace.layers.cf1_global_gain.nb_outputs,
        lace.layers.cf1_global_gain.float_weights.len(),
        lace.layers.cf1_global_gain.weights.len(),
        lace.layers.cf1_global_gain.weights_idx.len(),
    );
}
