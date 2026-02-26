//! OSCE nndsp building block tests: compare Rust vs C with deterministic inputs.
//!
//! Ports the logic from upstream `dnn/adaconvtest.c` — exercises adaconv, adacomb,
//! and adashape with compiled-in LACE/NoLACE weights and deterministic PRNG inputs,
//! then verifies bit-exact match between the Rust and C implementations.

#![cfg(feature = "tools-dnn")]

use opurs::dnn::nndsp::*;
use opurs::dnn::osce::*;
use opurs::dnn::weights::compiled_weights;

unsafe extern "C" {
    fn osce_test_compute_conv2d_3x3(out: *mut f32, seed: u32) -> i32;
    fn osce_test_compute_linear_int8_arch(out: *mut f32, seed: u32, arch: i32) -> i32;
    fn osce_test_compute_activation_exp_arch(out: *mut f32, seed: u32, arch: i32) -> i32;
}

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

/// Get C libm values for cos/exp/ln comparison.
fn c_libm_values() -> Vec<f32> {
    let mut out = vec![0.0f32; 56];
    unsafe {
        libopus_sys::osce_test_libm_values(out.as_mut_ptr());
    }
    out
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

    let arch = opurs::arch::opus_select_arch();
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
            arch,
        );
        all_out.extend_from_slice(&x_out);
    }
    all_out
}

/// Run Rust adacomb with same PRNG inputs as C harness.
fn rust_adacomb(lace: &LACE, num_frames: usize, seed: u32) -> Vec<f32> {
    let arch = opurs::arch::opus_select_arch();
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
            arch,
        );
        all_out.extend_from_slice(&x_out);
    }
    all_out
}

/// Run Rust adashape with same PRNG inputs as C harness.
fn rust_adashape(nolace: &NoLACE, num_frames: usize, seed: u32) -> Vec<f32> {
    let arch = opurs::arch::opus_select_arch();
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
            1,
            arch,
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

    for (i, (&r, &c_val)) in rust.iter().zip(c.iter()).enumerate() {
        assert_eq!(
            r.to_bits(),
            c_val.to_bits(),
            "{name}: MISMATCH at sample {i}: rust={r} (0x{:08x}) c={c_val} (0x{:08x})",
            r.to_bits(),
            c_val.to_bits(),
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

    let arch = opurs::arch::opus_select_arch();
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
    compute_linear(&lace.layers.af1_kernel, &mut rust_out, &input, arch);

    compare_outputs("compute_linear_lace_af1_kernel", &rust_out, &c_out);
}

/// Test compute_generic_dense with ACTIVATION_TANH on LACE af1_gain layer.
#[test]
fn test_dense_tanh_lace_af1_gain() {
    use opurs::dnn::nnet::compute_generic_dense;

    let arch = opurs::arch::opus_select_arch();
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
        arch,
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

    let arch = opurs::arch::opus_select_arch();
    let mut rust_out = vec![0.0f32; max_pitch];
    celt_pitch_xcorr(&kernel, &signal, &mut rust_out, len, arch);

    compare_outputs("celt_pitch_xcorr_neon", &rust_out, &c_out);
}

/// Test compute_linear on NoLACE tdshape1_alpha1_f layer.
/// Isolates whether the divergence in nolace_tdshape1 starts at the linear layer.
#[test]
fn test_compute_linear_nolace_tdshape() {
    use opurs::dnn::nnet::compute_linear;

    let arch = opurs::arch::opus_select_arch();
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
    compute_linear(layer, &mut rust_out, &input, arch);

    compare_outputs("compute_linear_nolace_tdshape", &rust_out, &c_out);
}

/// Test compute_linear on NoLACE af2_kernel layer.
/// Isolates whether the divergence in nolace_af2 starts at the linear layer.
#[test]
fn test_compute_linear_nolace_af2() {
    use opurs::dnn::nnet::compute_linear;

    let arch = opurs::arch::opus_select_arch();
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
    compute_linear(layer, &mut rust_out, &input, arch);

    compare_outputs("compute_linear_nolace_af2", &rust_out, &c_out);
}

/// Verify tanh_approx is bit-exact between Rust and C.
#[test]
fn test_tanh_approx() {
    use opurs::dnn::nnet::compute_linear;
    use opurs::dnn::simd::tanh_approx;

    let arch = opurs::arch::opus_select_arch();
    let arrays = compiled_weights();
    let lace = init_lace(&arrays).expect("LACE init failed");

    // Generate input from the gain layer's compute_linear output
    let nb_inputs = lace.layers.af1_gain.nb_inputs;
    let mut prng = Prng::new(SEED);
    let input: Vec<f32> = (0..nb_inputs).map(|_| prng.next_float() * 0.1).collect();
    let mut linear_out = vec![0.0f32; lace.layers.af1_gain.nb_outputs];
    compute_linear(&lace.layers.af1_gain, &mut linear_out, &input, arch);

    // Compare Rust tanh_approx against C tanh_approx for the computed value
    let exact_val = linear_out[0];
    let mut c_tanh = [0.0f32; 2];
    unsafe { libopus_sys::osce_test_tanh_approx(c_tanh.as_mut_ptr(), exact_val) };
    let rust_tanh = tanh_approx(exact_val, arch);

    assert_eq!(
        rust_tanh.to_bits(),
        c_tanh[0].to_bits(),
        "tanh_approx mismatch: rust={rust_tanh} (0x{:08x}) c={} (0x{:08x})",
        rust_tanh.to_bits(),
        c_tanh[0],
        c_tanh[0].to_bits(),
    );
}

/// Test compute_linear on LACE fnet_conv2 layer (int8 weights, exercises cgemv8x4).
#[test]
fn test_compute_linear_int8_lace_fnet_conv2() {
    use opurs::dnn::nnet::compute_linear;

    let arch = opurs::arch::opus_select_arch();
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
    compute_linear(&lace.layers.fnet_conv2, &mut rust_out, &input, arch);

    compare_outputs("compute_linear_int8_lace_fnet_conv2", &rust_out, &c_out);
}

/// Verify compute_linear int8 path respects forced arch tiers like upstream RTCD.
#[test]
fn test_compute_linear_int8_arch_tiers_match_c() {
    use opurs::arch::Arch;
    use opurs::dnn::nnet::compute_linear;

    let arrays = compiled_weights();
    let lace = init_lace(&arrays).expect("LACE init failed");
    let layer = &lace.layers.fnet_conv2;
    let nb_inputs = layer.nb_inputs;
    let nb_outputs = layer.nb_outputs;

    let mut prng = Prng::new(SEED);
    let input: Vec<f32> = (0..nb_inputs).map(|_| prng.next_float() * 0.1).collect();

    let tiers: Vec<(Arch, i32, &'static str)> = {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            vec![
                (Arch::Scalar, 0, "scalar"),
                (Arch::Sse, 1, "sse"),
                (Arch::Sse2, 2, "sse2"),
                (Arch::Sse4_1, 3, "sse4_1"),
                (Arch::Avx2, 4, "avx2"),
            ]
        }
        #[cfg(target_arch = "aarch64")]
        {
            vec![
                (Arch::Scalar, 0, "scalar"),
                (Arch::Neon, 3, "neon"),
                (Arch::DotProd, 4, "dotprod"),
            ]
        }
        #[cfg(not(any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64")))]
        {
            vec![(Arch::Scalar, 0, "scalar")]
        }
    };

    for (arch, c_arch, name) in tiers {
        let mut c_out = vec![0.0f32; nb_outputs];
        let c_nb_inputs =
            unsafe { osce_test_compute_linear_int8_arch(c_out.as_mut_ptr(), SEED, c_arch) }
                as usize;
        assert_eq!(c_nb_inputs, nb_inputs);

        let mut rust_out = vec![0.0f32; nb_outputs];
        compute_linear(layer, &mut rust_out, &input, arch);

        let label = format!("compute_linear_int8_arch_tier_{name}");
        compare_outputs(&label, &rust_out, &c_out);
    }
}

/// Verify compute_activation EXP path respects forced arch tiers like upstream RTCD.
#[test]
fn test_compute_activation_exp_arch_tiers_match_c() {
    use opurs::arch::Arch;
    use opurs::dnn::nnet::{compute_activation, ACTIVATION_EXP};

    let n = 23usize;
    let mut prng = Prng::new(SEED);
    let input: Vec<f32> = (0..n).map(|_| prng.next_float() * 8.0).collect();

    let tiers: Vec<(Arch, i32, &'static str)> = {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            vec![
                (Arch::Scalar, 0, "scalar"),
                (Arch::Sse, 1, "sse"),
                (Arch::Sse2, 2, "sse2"),
                (Arch::Sse4_1, 3, "sse4_1"),
                (Arch::Avx2, 4, "avx2"),
            ]
        }
        #[cfg(target_arch = "aarch64")]
        {
            vec![
                (Arch::Scalar, 0, "scalar"),
                (Arch::Neon, 3, "neon"),
                (Arch::DotProd, 4, "dotprod"),
            ]
        }
        #[cfg(not(any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64")))]
        {
            vec![(Arch::Scalar, 0, "scalar")]
        }
    };

    for (arch, c_arch, name) in tiers {
        let mut c_out = vec![0.0f32; n];
        let c_n = unsafe { osce_test_compute_activation_exp_arch(c_out.as_mut_ptr(), SEED, c_arch) }
            as usize;
        assert_eq!(c_n, n);

        let mut rust_out = vec![0.0f32; n];
        compute_activation(&mut rust_out, &input, n, ACTIVATION_EXP, arch);

        let label = format!("compute_activation_exp_arch_tier_{name}");
        compare_outputs(&label, &rust_out, &c_out);
    }
}

/// Test compute_generic_gru on LACE fnet GRU (int8 weights, 2 steps).
#[test]
fn test_gru_lace_fnet() {
    use opurs::dnn::nnet::compute_generic_gru;
    use opurs::dnn::osce::LACE_COND_DIM;

    let arch = opurs::arch::opus_select_arch();
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
        arch,
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
        arch,
    );
    compare_outputs("gru_lace_fnet_step2", &state, &c_out[LACE_COND_DIM..]);
}

/// Test compute_generic_dense with ACTIVATION_TANH on LACE fnet_tconv (int8 weights, 128->512).
#[test]
fn test_dense_tanh_lace_tconv() {
    use opurs::dnn::nnet::{compute_generic_dense, ACTIVATION_TANH};

    let arch = opurs::arch::opus_select_arch();
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
        arch,
    );

    compare_outputs("dense_tanh_lace_tconv", &rust_out, &c_out);
}

/// Test compute_conv2d on a deterministic 3x3 setup against upstream C.
#[test]
fn test_compute_conv2d_3x3() {
    use opurs::dnn::nnet::{compute_conv2d, Conv2dLayer, ACTIVATION_TANH};

    const IN_CHANNELS: usize = 3;
    const OUT_CHANNELS: usize = 2;
    const KTIME: usize = 3;
    const KHEIGHT: usize = 3;
    const HEIGHT: usize = 17;
    const HSTRIDE: usize = 17;

    let arch = opurs::arch::opus_select_arch();
    let time_stride = IN_CHANNELS * (HEIGHT + KHEIGHT - 1);
    let hist_size = (KTIME - 1) * time_stride;
    let w_size = OUT_CHANNELS * IN_CHANNELS * KTIME * KHEIGHT;

    let mut prng = Prng::new(SEED);
    let weights: Vec<f32> = (0..w_size).map(|_| prng.next_float() * 0.25).collect();
    let bias: Vec<f32> = (0..OUT_CHANNELS).map(|_| prng.next_float() * 0.1).collect();
    let mem_init: Vec<f32> = (0..hist_size).map(|_| prng.next_float() * 0.05).collect();
    let input: Vec<f32> = (0..time_stride).map(|_| prng.next_float() * 0.5).collect();

    let conv = Conv2dLayer {
        bias,
        float_weights: weights,
        in_channels: IN_CHANNELS,
        out_channels: OUT_CHANNELS,
        ktime: KTIME,
        kheight: KHEIGHT,
    };

    let mut rust_mem = mem_init;
    let mut rust_out = vec![0.0f32; OUT_CHANNELS * HSTRIDE];
    compute_conv2d(
        &conv,
        &mut rust_out,
        &mut rust_mem,
        &input,
        HEIGHT,
        HSTRIDE,
        ACTIVATION_TANH,
        arch,
    );

    let mut c_out = vec![0.0f32; OUT_CHANNELS * HSTRIDE];
    let c_len = unsafe { osce_test_compute_conv2d_3x3(c_out.as_mut_ptr(), SEED) } as usize;
    c_out.truncate(c_len);

    compare_outputs("compute_conv2d_3x3", &rust_out, &c_out);
}

/// Step-by-step bit-exact comparison of adacomb intermediates between Rust and C.
#[test]
fn test_adacomb_intermediates() {
    use opurs::celt::pitch::celt_pitch_xcorr;
    use opurs::dnn::nndsp::*;
    use opurs::dnn::nnet::{
        compute_generic_dense, ACTIVATION_LINEAR, ACTIVATION_RELU, ACTIVATION_TANH,
    };

    let arch = opurs::arch::opus_select_arch();
    let arrays = compiled_weights();
    let lace = init_lace(&arrays).expect("LACE init failed");

    // C intermediates layout (452 floats):
    //   [0..15]=kernel, [16]=gain, [17]=global_gain, [18]=gain_exp, [19]=ggain_exp,
    //   [20..35]=scaled_kernel, [36..115]=xcorr(80), [116..155]=window(40),
    //   [156..235]=x_in(80), [236..315]=overlap_add(80), [316..395]=full_output(80),
    //   [396]=pitch_lag, [397]=last_global_gain
    let mut c_buf = vec![0.0f32; 512];
    unsafe { libopus_sys::osce_test_adacomb_intermediates(c_buf.as_mut_ptr(), SEED) };

    let frame_size = LACE_FRAME_SIZE; // 80
    let overlap_size = LACE_OVERLAP_SIZE; // 40
    let kernel_size = LACE_CF1_KERNEL_SIZE; // 16
    let feature_dim = LACE_COND_DIM; // 128

    // Generate PRNG inputs matching C
    let mut prng = Prng::new(SEED);
    let features: Vec<f32> = (0..feature_dim).map(|_| prng.next_float() * 0.1).collect();
    let x_in: Vec<f32> = (0..frame_size).map(|_| prng.next_float() * 0.5).collect();
    let pitch_lag = {
        let v = prng.next_float() * 32768.0 + 32768.0;
        kernel_size as i32 + ((v as u32) % (250 - kernel_size as u32)) as i32
    };

    let mut diffs = Vec::new();

    // Verify pitch_lag matches
    let c_pitch_lag = c_buf[396] as i32;
    if pitch_lag != c_pitch_lag {
        diffs.push(format!("  pitch_lag: rust={pitch_lag} c={c_pitch_lag}"));
    }

    // Step 1: kernel
    let mut rust_kernel = vec![0.0f32; 16];
    compute_generic_dense(
        &lace.layers.cf1_kernel,
        &mut rust_kernel,
        &features,
        ACTIVATION_LINEAR,
        arch,
    );
    for i in 0..16 {
        if rust_kernel[i] != c_buf[i] {
            diffs.push(format!(
                "  kernel[{i}]: rust=0x{:08x} c=0x{:08x}",
                rust_kernel[i].to_bits(),
                c_buf[i].to_bits()
            ));
        }
    }

    // Step 2: gain (RELU)
    let mut rust_gain = [0.0f32; 1];
    compute_generic_dense(
        &lace.layers.cf1_gain,
        &mut rust_gain,
        &features,
        ACTIVATION_RELU,
        arch,
    );
    if rust_gain[0] != c_buf[16] {
        diffs.push(format!(
            "  gain: rust=0x{:08x} c=0x{:08x}",
            rust_gain[0].to_bits(),
            c_buf[16].to_bits()
        ));
    }

    // Step 3: global_gain (TANH)
    let mut rust_ggain = [0.0f32; 1];
    compute_generic_dense(
        &lace.layers.cf1_global_gain,
        &mut rust_ggain,
        &features,
        ACTIVATION_TANH,
        arch,
    );
    if rust_ggain[0] != c_buf[17] {
        diffs.push(format!(
            "  global_gain: rust=0x{:08x} c=0x{:08x}",
            rust_ggain[0].to_bits(),
            c_buf[17].to_bits()
        ));
    }

    // Step 4: transformed gains
    let gain_exp = ((LACE_CF1_LOG_GAIN_LIMIT - rust_gain[0]) as f64).exp() as f32;
    let ggain_exp =
        ((LACE_CF1_FILTER_GAIN_A * rust_ggain[0] + LACE_CF1_FILTER_GAIN_B) as f64).exp() as f32;
    if gain_exp != c_buf[18] {
        diffs.push(format!(
            "  gain_exp: rust=0x{:08x} c=0x{:08x}",
            gain_exp.to_bits(),
            c_buf[18].to_bits()
        ));
    }
    if ggain_exp != c_buf[19] {
        diffs.push(format!(
            "  ggain_exp: rust=0x{:08x} c=0x{:08x}",
            ggain_exp.to_bits(),
            c_buf[19].to_bits()
        ));
    }

    // Step 5: scaled kernel
    let mut scaled = rust_kernel.clone();
    scale_kernel(&mut scaled, 1, 1, kernel_size, &[gain_exp]);
    for i in 0..16 {
        if scaled[i] != c_buf[20 + i] {
            diffs.push(format!(
                "  scaled_kernel[{i}]: rust=0x{:08x} c=0x{:08x}",
                scaled[i].to_bits(),
                c_buf[20 + i].to_bits()
            ));
        }
    }

    // Step 6: window comparison
    let c_window = &c_buf[116..116 + overlap_size];
    let mut rust_window = vec![0.0f32; overlap_size];
    compute_overlap_window(&mut rust_window, overlap_size);
    for i in 0..overlap_size {
        if rust_window[i] != c_window[i] {
            diffs.push(format!(
                "  window[{i}]: rust=0x{:08x} c=0x{:08x}",
                rust_window[i].to_bits(),
                c_window[i].to_bits()
            ));
        }
    }

    // Step 7: x_in comparison
    let c_x_in = &c_buf[156..156 + frame_size];
    for i in 0..frame_size {
        if x_in[i] != c_x_in[i] {
            diffs.push(format!(
                "  x_in[{i}]: rust=0x{:08x} c=0x{:08x}",
                x_in[i].to_bits(),
                c_x_in[i].to_bits()
            ));
        }
    }

    // Step 8: xcorr comparison
    // Replicate input_buffer setup from adacomb_process_frame
    let hist_len = kernel_size + ADACOMB_MAX_LAG;
    let mut input_buffer =
        vec![0.0f32; ADACOMB_MAX_FRAME_SIZE + ADACOMB_MAX_LAG + ADACOMB_MAX_KERNEL_SIZE];
    // history is zeros for frame 0
    input_buffer[hist_len..hist_len + frame_size].copy_from_slice(&x_in);
    let p_offset = hist_len; // p_input = &input_buffer[p_offset]
    let left_padding = kernel_size - 1;

    let mut kernel_padded = [0.0f32; ADACOMB_MAX_KERNEL_SIZE];
    kernel_padded[..kernel_size].copy_from_slice(&scaled[..kernel_size]);

    let xcorr_start = p_offset - left_padding - pitch_lag as usize;
    let arch = opurs::arch::opus_select_arch();
    let mut rust_xcorr = vec![0.0f32; frame_size];
    celt_pitch_xcorr(
        &kernel_padded,
        &input_buffer[xcorr_start..],
        &mut rust_xcorr,
        ADACOMB_MAX_KERNEL_SIZE,
        arch,
    );

    let c_xcorr = &c_buf[36..36 + frame_size];
    for i in 0..frame_size {
        if rust_xcorr[i] != c_xcorr[i] {
            diffs.push(format!(
                "  xcorr[{i}]: rust={:e} (0x{:08x}) c={:e} (0x{:08x}) diff={:e}",
                rust_xcorr[i],
                rust_xcorr[i].to_bits(),
                c_xcorr[i],
                c_xcorr[i].to_bits(),
                (rust_xcorr[i] - c_xcorr[i]).abs()
            ));
        }
    }

    // Step 9: overlap-add comparison
    // Replicate the three overlap-add loops from adacomb_process_frame
    let last_global_gain = 0.0f32; // frame 0
    let mut output = rust_xcorr.clone();
    // Loop 1: crossfade (last_global_gain=0, so first term vanishes)
    for i in 0..overlap_size {
        output[i] = last_global_gain * rust_window[i] * 0.0 // output_buffer_last[i] = 0
            + ggain_exp * (1.0 - rust_window[i]) * output[i];
    }
    // Loop 2: add direct signal (overlap)
    for i in 0..overlap_size {
        output[i] += (rust_window[i] * last_global_gain + (1.0 - rust_window[i]) * ggain_exp)
            * input_buffer[p_offset + i];
    }
    // Loop 3: add direct signal (non-overlap)
    for i in overlap_size..frame_size {
        output[i] = ggain_exp * (output[i] + input_buffer[p_offset + i]);
    }

    let c_overlap = &c_buf[236..236 + frame_size];
    for i in 0..frame_size {
        if output[i] != c_overlap[i] {
            diffs.push(format!(
                "  overlap[{i}]: rust={:e} (0x{:08x}) c={:e} (0x{:08x}) diff={:e}",
                output[i],
                output[i].to_bits(),
                c_overlap[i],
                c_overlap[i].to_bits(),
                (output[i] - c_overlap[i]).abs()
            ));
        }
    }

    // Step 10: full adacomb_process_frame output cross-check
    let rust_full = rust_adacomb(&lace, 1, SEED);
    let c_full = &c_buf[316..316 + frame_size];
    for i in 0..frame_size {
        if rust_full[i] != c_full[i] {
            diffs.push(format!(
                "  full_output[{i}]: rust={:e} (0x{:08x}) c={:e} (0x{:08x}) diff={:e}",
                rust_full[i],
                rust_full[i].to_bits(),
                c_full[i],
                c_full[i].to_bits(),
                (rust_full[i] - c_full[i]).abs()
            ));
        }
    }

    assert!(
        diffs.is_empty(),
        "adacomb intermediates mismatches:\n{}",
        diffs.join("\n")
    );
}

/// Verify Rust cos/exp/ln match C libm for values used in OSCE.
#[test]
fn test_libm_comparison() {
    let c_vals = c_libm_values();

    // cos values for overlap window angles
    for (i, &c_val) in c_vals[..40].iter().enumerate() {
        let angle = std::f64::consts::PI * (i as f64 + 0.5) / 40.0;
        let rust_val = (0.5 + 0.5 * angle.cos()) as f32;
        assert_eq!(
            rust_val.to_bits(),
            c_val.to_bits(),
            "cos window[{i}]: rust=0x{:08x} c=0x{:08x}",
            rust_val.to_bits(),
            c_val.to_bits(),
        );
    }

    // exp values
    let test_exp: &[f64] = &[-0.5, -1.0, -2.0, -3.5, 0.1, 0.5, 1.0, 2.0];
    for (j, &v) in test_exp.iter().enumerate() {
        let rust_val = v.exp() as f32;
        let c_val = c_vals[40 + j];
        assert_eq!(
            rust_val.to_bits(),
            c_val.to_bits(),
            "exp({v}): rust=0x{:08x} c=0x{:08x}",
            rust_val.to_bits(),
            c_val.to_bits(),
        );
    }

    // ln values
    let test_ln: &[f64] = &[0.001, 0.1, 0.5, 1.0, 2.0, 10.0, 100.0, 0.000015258789];
    for (j, &v) in test_ln.iter().enumerate() {
        let rust_val = v.ln() as f32;
        let c_val = c_vals[48 + j];
        assert_eq!(
            rust_val.to_bits(),
            c_val.to_bits(),
            "ln({v}): rust=0x{:08x} c=0x{:08x}",
            rust_val.to_bits(),
            c_val.to_bits(),
        );
    }
}
