//! Neural network layer types and operations.
//!
//! Core inference engine for all DNN models. Provides `LinearLayer` and
//! `Conv2dLayer` types plus compute functions for dense, GRU, Conv1D, and GLU.
//!
//! Upstream C: `dnn/nnet.h`, `dnn/nnet.c`, `dnn/nnet_arch.h`, `dnn/parse_lpcnet_weights.c`

use super::vec::*;
use crate::arch::Arch;

// --- Activation types ---

pub const ACTIVATION_LINEAR: i32 = 0;
pub const ACTIVATION_SIGMOID: i32 = 1;
pub const ACTIVATION_TANH: i32 = 2;
pub const ACTIVATION_RELU: i32 = 3;
pub const ACTIVATION_SOFTMAX: i32 = 4;
pub const ACTIVATION_SWISH: i32 = 5;
pub const ACTIVATION_EXP: i32 = 6;

// --- Weight array types ---

pub const WEIGHT_TYPE_FLOAT: i32 = 0;
pub const WEIGHT_TYPE_INT: i32 = 1;
pub const WEIGHT_TYPE_QWEIGHT: i32 = 2;
pub const WEIGHT_TYPE_INT8: i32 = 3;

pub const WEIGHT_BLOB_VERSION: i32 = 0;
pub const WEIGHT_BLOCK_SIZE: usize = 64;

// --- Weight data ---

/// A named array of weight data from a model file.
///
/// Upstream C: dnn/nnet.h:WeightArray
#[derive(Clone, Debug)]
pub struct WeightArray {
    pub name: String,
    pub type_id: i32,
    pub size: usize,
    pub data: Vec<u8>,
}

/// Header for a weight record in a binary blob.
///
/// Upstream C: dnn/nnet.h:WeightHead
#[repr(C)]
struct WeightHead {
    head: [u8; 4],
    version: i32,
    type_id: i32,
    size: i32,
    block_size: i32,
    name: [u8; 44],
}

// --- Layer types ---

/// Generic sparse/dense affine layer.
///
/// Used for dense layers, GRU gates, Conv1D, and GLU layers.
/// Weights can be either int8-quantized (with scale) or float32.
///
/// Upstream C: dnn/nnet.h:LinearLayer
#[derive(Clone, Debug, Default)]
pub struct LinearLayer {
    pub bias: Vec<f32>,
    pub subias: Vec<f32>,
    pub weights: Vec<i8>,
    pub float_weights: Vec<f32>,
    pub weights_idx: Vec<i32>,
    pub diag: Vec<f32>,
    pub scale: Vec<f32>,
    pub nb_inputs: usize,
    pub nb_outputs: usize,
}

/// 2D convolution layer.
///
/// Used only by PitchDNN. Weights are always float32.
///
/// Upstream C: dnn/nnet.h:Conv2dLayer
#[derive(Clone, Debug, Default)]
pub struct Conv2dLayer {
    pub bias: Vec<f32>,
    pub float_weights: Vec<f32>,
    pub in_channels: usize,
    pub out_channels: usize,
    pub ktime: usize,
    pub kheight: usize,
}

// --- Activation ---

const MAX_ACTIVATIONS: usize = 4096;

fn vec_swish(y: &mut [f32], x: &[f32], n: usize, arch: Arch) {
    let mut tmp = [0.0f32; MAX_ACTIVATIONS];
    debug_assert!(n <= MAX_ACTIVATIONS);
    vec_sigmoid(&mut tmp[..n], &x[..n], arch);
    for i in 0..n {
        y[i] = x[i] * tmp[i];
    }
}

/// Apply activation function in-place or from input to output.
///
/// Upstream C: dnn/nnet_arch.h:compute_activation_c
fn compute_activation_c(output: &mut [f32], input: &[f32], n: usize, activation: i32, arch: Arch) {
    match activation {
        ACTIVATION_SIGMOID => vec_sigmoid(&mut output[..n], &input[..n], arch),
        ACTIVATION_TANH => vec_tanh(&mut output[..n], &input[..n], arch),
        ACTIVATION_SWISH => vec_swish(&mut output[..n], &input[..n], n, arch),
        ACTIVATION_RELU => {
            for i in 0..n {
                output[i] = if input[i] < 0.0 { 0.0 } else { input[i] };
            }
        }
        ACTIVATION_SOFTMAX => {
            // SOFTMAX_HACK: just copy (used as identity in practice)
            output[..n].copy_from_slice(&input[..n]);
        }
        ACTIVATION_EXP => softmax(&mut output[..n], &input[..n], arch),
        ACTIVATION_LINEAR | _ => {
            if !std::ptr::eq(output.as_ptr(), input.as_ptr()) {
                output[..n].copy_from_slice(&input[..n]);
            }
        }
    }
}

/// Apply activation function with RTCD backend dispatch.
///
/// Upstream C RTCD tables:
/// - dnn/x86/x86_dnn_map.c:DNN_COMPUTE_ACTIVATION_IMPL
/// - dnn/arm/arm_dnn_map.c:DNN_COMPUTE_ACTIVATION_IMPL
pub fn compute_activation(
    output: &mut [f32],
    input: &[f32],
    n: usize,
    activation: i32,
    arch: Arch,
) {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if arch.has_avx2() {
            return x86_rtcd::compute_activation_avx2(output, input, n, activation, arch);
        }
        if arch.has_sse4_1() {
            return x86_rtcd::compute_activation_sse4_1(output, input, n, activation, arch);
        }
        if arch.has_sse2() {
            return x86_rtcd::compute_activation_sse2(output, input, n, activation, arch);
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if arch.has_dotprod() {
            return arm_rtcd::compute_activation_dotprod(output, input, n, activation, arch);
        }
        if arch.has_neon() {
            return arm_rtcd::compute_activation_neon(output, input, n, activation, arch);
        }
    }

    compute_activation_c(output, input, n, activation, arch)
}

// --- Linear computation ---

/// Compute affine transform: out = W*in + bias + diag*in
///
/// Dispatches to int8 cgemv or float sgemv depending on which weights are present.
/// Int8 is preferred when available (matching C runtime behavior where float weights
/// are compile-time excluded via `#ifdef`).
///
/// Upstream C: dnn/nnet_arch.h:compute_linear_c
fn compute_linear_c(linear: &LinearLayer, out: &mut [f32], input: &[f32], arch: Arch) {
    let m = linear.nb_inputs;
    let n = linear.nb_outputs;
    let mut used_int8_path = false;

    if !linear.weights.is_empty() {
        used_int8_path = true;
        if !linear.weights_idx.is_empty() {
            sparse_cgemv8x4(
                out,
                &linear.weights,
                &linear.weights_idx,
                &linear.scale,
                n,
                m,
                input,
                arch,
            );
        } else {
            cgemv8x4(out, &linear.weights, &linear.scale, n, m, input, arch);
        }
    } else if !linear.float_weights.is_empty() {
        if !linear.weights_idx.is_empty() {
            sparse_sgemv8x4(
                out,
                &linear.float_weights,
                &linear.weights_idx,
                n,
                input,
                arch,
            );
        } else {
            sgemv(out, &linear.float_weights, n, m, n, input, arch);
        }
    } else {
        for i in 0..n {
            out[i] = 0.0;
        }
    }

    let bias = if used_int8_path && use_su_bias(arch) && !linear.subias.is_empty() {
        // USE_SU_BIAS: x86 AVX2 cgemv8x4 uses unsigned u8 quantization,
        // so subias compensates for the +127 offset. NEON and scalar use
        // signed i8 quantization and need regular bias.
        &linear.subias
    } else {
        &linear.bias
    };
    if !bias.is_empty() {
        for i in 0..n {
            out[i] += bias[i];
        }
    }

    if !linear.diag.is_empty() {
        // Diag is only used for GRU recurrent weights: 3*M == N
        debug_assert!(3 * m == n);
        for i in 0..m {
            out[i] += linear.diag[i] * input[i];
            out[i + m] += linear.diag[i + m] * input[i];
            out[i + 2 * m] += linear.diag[i + 2 * m] * input[i];
        }
    }
}

/// Compute affine transform with RTCD backend dispatch.
///
/// Upstream C RTCD tables:
/// - dnn/x86/x86_dnn_map.c:DNN_COMPUTE_LINEAR_IMPL
/// - dnn/arm/arm_dnn_map.c:DNN_COMPUTE_LINEAR_IMPL
pub fn compute_linear(linear: &LinearLayer, out: &mut [f32], input: &[f32], arch: Arch) {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if arch.has_avx2() {
            return x86_rtcd::compute_linear_avx2(linear, out, input, arch);
        }
        if arch.has_sse4_1() {
            return x86_rtcd::compute_linear_sse4_1(linear, out, input, arch);
        }
        if arch.has_sse2() {
            return x86_rtcd::compute_linear_sse2(linear, out, input, arch);
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if arch.has_dotprod() {
            return arm_rtcd::compute_linear_dotprod(linear, out, input, arch);
        }
        if arch.has_neon() {
            return arm_rtcd::compute_linear_neon(linear, out, input, arch);
        }
    }

    compute_linear_c(linear, out, input, arch)
}

// --- High-level layer operations ---

/// Dense layer: linear transform + activation.
///
/// Upstream C: dnn/nnet.c:compute_generic_dense
pub fn compute_generic_dense(
    layer: &LinearLayer,
    output: &mut [f32],
    input: &[f32],
    activation: i32,
    arch: Arch,
) {
    compute_linear(layer, output, input, arch);
    let n = layer.nb_outputs;
    let tmp = output[..n].to_vec();
    compute_activation(&mut output[..n], &tmp, n, activation, arch);
}

/// GRU layer: standard z/r/h gated recurrent unit.
///
/// `state` is both input (previous state) and output (new state).
///
/// Upstream C: dnn/nnet.c:compute_generic_gru
pub fn compute_generic_gru(
    input_weights: &LinearLayer,
    recurrent_weights: &LinearLayer,
    state: &mut [f32],
    input: &[f32],
    arch: Arch,
) {
    debug_assert!(3 * recurrent_weights.nb_inputs == recurrent_weights.nb_outputs);
    debug_assert!(input_weights.nb_outputs == recurrent_weights.nb_outputs);

    let n = recurrent_weights.nb_inputs;
    let mut zrh = vec![0.0f32; 3 * n];
    let mut recur = vec![0.0f32; 3 * n];

    compute_linear(input_weights, &mut zrh, input, arch);
    compute_linear(recurrent_weights, &mut recur, state, arch);

    // z and r: add recurrent, then sigmoid
    for i in 0..2 * n {
        zrh[i] += recur[i];
    }
    let tmp_zr = zrh[..2 * n].to_vec();
    compute_activation(&mut zrh[..2 * n], &tmp_zr, 2 * n, ACTIVATION_SIGMOID, arch);

    // h: add r-gated recurrent, then tanh
    for i in 0..n {
        zrh[2 * n + i] += recur[2 * n + i] * zrh[n + i]; // r[i] gates recurrent
    }
    let mut h = vec![0.0f32; n];
    compute_activation(&mut h, &zrh[2 * n..], n, ACTIVATION_TANH, arch);

    // state = z*state + (1-z)*h
    for i in 0..n {
        state[i] = zrh[i] * state[i] + (1.0 - zrh[i]) * h[i];
    }
}

/// Gated Linear Unit: out = input * sigmoid(W*input)
///
/// Upstream C: dnn/nnet.c:compute_glu
pub fn compute_glu(layer: &LinearLayer, output: &mut [f32], input: &[f32], arch: Arch) {
    debug_assert!(layer.nb_inputs == layer.nb_outputs);
    let n = layer.nb_outputs;
    let mut act = vec![0.0f32; n];
    compute_linear(layer, &mut act, input, arch);
    let tmp = act.clone();
    compute_activation(&mut act, &tmp, n, ACTIVATION_SIGMOID, arch);
    if std::ptr::eq(output.as_ptr(), input.as_ptr()) {
        for i in 0..n {
            output[i] *= act[i];
        }
    } else {
        for i in 0..n {
            output[i] = input[i] * act[i];
        }
    }
}

/// Gated activation: out = input * activation(W*input).
///
/// Upstream C: dnn/nnet.h:compute_gated_activation
pub fn compute_gated_activation(
    layer: &LinearLayer,
    output: &mut [f32],
    input: &[f32],
    activation: i32,
    arch: Arch,
) {
    debug_assert!(layer.nb_inputs == layer.nb_outputs);
    let n = layer.nb_outputs;
    let mut act = vec![0.0f32; n];
    compute_linear(layer, &mut act, input, arch);
    let tmp = act.clone();
    compute_activation(&mut act, &tmp, n, activation, arch);
    if std::ptr::eq(output.as_ptr(), input.as_ptr()) {
        for i in 0..n {
            output[i] *= act[i];
        }
    } else {
        for i in 0..n {
            output[i] = input[i] * act[i];
        }
    }
}

const MAX_CONV_INPUTS_ALL: usize = 1536; // DRED_MAX_CONV_INPUTS

/// Causal 1D convolution: linear(concat(mem, input)) + activation.
///
/// `mem` stores the history from previous frames.
///
/// Upstream C: dnn/nnet.c:compute_generic_conv1d
pub fn compute_generic_conv1d(
    layer: &LinearLayer,
    output: &mut [f32],
    mem: &mut [f32],
    input: &[f32],
    input_size: usize,
    activation: i32,
    arch: Arch,
) {
    let mut tmp = vec![0.0f32; MAX_CONV_INPUTS_ALL];
    debug_assert!(layer.nb_inputs <= MAX_CONV_INPUTS_ALL);
    let hist_size = layer.nb_inputs - input_size;
    if hist_size > 0 {
        tmp[..hist_size].copy_from_slice(&mem[..hist_size]);
    }
    tmp[hist_size..hist_size + input_size].copy_from_slice(&input[..input_size]);
    compute_linear(layer, output, &tmp, arch);
    let n = layer.nb_outputs;
    let out_tmp = output[..n].to_vec();
    compute_activation(&mut output[..n], &out_tmp, n, activation, arch);
    if hist_size > 0 {
        mem[..hist_size].copy_from_slice(&tmp[input_size..input_size + hist_size]);
    }
}

/// Dilated causal 1D convolution.
///
/// Upstream C: dnn/nnet.c:compute_generic_conv1d_dilation
pub fn compute_generic_conv1d_dilation(
    layer: &LinearLayer,
    output: &mut [f32],
    mem: &mut [f32],
    input: &[f32],
    input_size: usize,
    dilation: usize,
    activation: i32,
    arch: Arch,
) {
    let mut tmp = vec![0.0f32; MAX_CONV_INPUTS_ALL];
    let ksize = layer.nb_inputs / input_size;
    debug_assert!(layer.nb_inputs <= MAX_CONV_INPUTS_ALL);

    if dilation == 1 {
        let hist_size = layer.nb_inputs - input_size;
        tmp[..hist_size].copy_from_slice(&mem[..hist_size]);
    } else {
        for i in 0..ksize - 1 {
            tmp[i * input_size..(i + 1) * input_size].copy_from_slice(
                &mem[i * input_size * dilation..i * input_size * dilation + input_size],
            );
        }
    }
    tmp[(ksize - 1) * input_size..ksize * input_size].copy_from_slice(&input[..input_size]);

    compute_linear(layer, output, &tmp, arch);
    let mut out_copy = output[..layer.nb_outputs].to_vec();
    compute_activation(
        &mut out_copy,
        &output[..layer.nb_outputs],
        layer.nb_outputs,
        activation,
        arch,
    );
    output[..layer.nb_outputs].copy_from_slice(&out_copy);

    if dilation == 1 {
        let hist_size = layer.nb_inputs - input_size;
        mem[..hist_size].copy_from_slice(&tmp[input_size..input_size + hist_size]);
    } else {
        let mem_len = input_size * dilation * (ksize - 1);
        mem.copy_within(input_size..mem_len, 0);
        mem[mem_len - input_size..mem_len].copy_from_slice(&input[..input_size]);
    }
}

// --- Conv2D ---

const MAX_CONV2D_INPUTS: usize = 8192;

/// 2D convolution (generic kernel size).
///
/// Upstream C: dnn/nnet_arch.h:conv2d_float
fn conv2d_float(
    out: &mut [f32],
    weights: &[f32],
    in_channels: usize,
    out_channels: usize,
    ktime: usize,
    kheight: usize,
    input: &[f32],
    height: usize,
    hstride: usize,
) {
    let in_stride = height + kheight - 1;
    for i in 0..out_channels {
        for j in 0..height {
            out[i * hstride + j] = 0.0;
        }
        for m in 0..in_channels {
            for t in 0..ktime {
                for h in 0..kheight {
                    for j in 0..height {
                        out[i * hstride + j] += weights[i * in_channels * ktime * kheight
                            + m * ktime * kheight
                            + t * kheight
                            + h]
                            * input[t * in_channels * in_stride + m * in_stride + j + h];
                    }
                }
            }
        }
    }
}

/// 3x3 specialized convolution used by upstream for common PitchDNN kernels.
///
/// Upstream C: dnn/nnet_arch.h:conv2d_3x3_float
fn conv2d_3x3_float(
    out: &mut [f32],
    weights: &[f32],
    in_channels: usize,
    out_channels: usize,
    input: &[f32],
    height: usize,
    hstride: usize,
) {
    let in_stride = height + 2;
    for i in 0..out_channels {
        for j in 0..height {
            out[i * hstride + j] = 0.0;
        }
        for m in 0..in_channels {
            for j in 0..height {
                out[i * hstride + j] += weights[i * in_channels * 9 + m * 9]
                    * input[m * in_stride + j]
                    + weights[i * in_channels * 9 + m * 9 + 1] * input[m * in_stride + j + 1]
                    + weights[i * in_channels * 9 + m * 9 + 2] * input[m * in_stride + j + 2]
                    + weights[i * in_channels * 9 + m * 9 + 3]
                        * input[in_channels * in_stride + m * in_stride + j]
                    + weights[i * in_channels * 9 + m * 9 + 4]
                        * input[in_channels * in_stride + m * in_stride + j + 1]
                    + weights[i * in_channels * 9 + m * 9 + 5]
                        * input[in_channels * in_stride + m * in_stride + j + 2]
                    + weights[i * in_channels * 9 + m * 9 + 6]
                        * input[2 * in_channels * in_stride + m * in_stride + j]
                    + weights[i * in_channels * 9 + m * 9 + 7]
                        * input[2 * in_channels * in_stride + m * in_stride + j + 1]
                    + weights[i * in_channels * 9 + m * 9 + 8]
                        * input[2 * in_channels * in_stride + m * in_stride + j + 2];
            }
        }
    }
}

/// Compute Conv2D layer with temporal memory.
///
/// Upstream C: dnn/nnet_arch.h:compute_conv2d_c
fn compute_conv2d_c(
    conv: &Conv2dLayer,
    out: &mut [f32],
    mem: &mut [f32],
    input: &[f32],
    height: usize,
    hstride: usize,
    activation: i32,
    arch: Arch,
) {
    let time_stride = conv.in_channels * (height + conv.kheight - 1);
    debug_assert!(conv.ktime * time_stride <= MAX_CONV2D_INPUTS);
    let mut in_buf = vec![0.0f32; MAX_CONV2D_INPUTS];

    // Copy history from mem, then current input
    let hist_size = (conv.ktime - 1) * time_stride;
    in_buf[..hist_size].copy_from_slice(&mem[..hist_size]);
    in_buf[hist_size..hist_size + time_stride].copy_from_slice(&input[..time_stride]);
    // Shift memory
    mem[..hist_size].copy_from_slice(&in_buf[time_stride..time_stride + hist_size]);

    if conv.kheight == 3 && conv.ktime == 3 {
        conv2d_3x3_float(
            out,
            &conv.float_weights,
            conv.in_channels,
            conv.out_channels,
            &in_buf,
            height,
            hstride,
        );
    } else {
        conv2d_float(
            out,
            &conv.float_weights,
            conv.in_channels,
            conv.out_channels,
            conv.ktime,
            conv.kheight,
            &in_buf,
            height,
            hstride,
        );
    }

    if !conv.bias.is_empty() {
        for i in 0..conv.out_channels {
            for j in 0..height {
                out[i * hstride + j] += conv.bias[i];
            }
        }
    }

    for i in 0..conv.out_channels {
        let start = i * hstride;
        let mut tmp = out[start..start + height].to_vec();
        compute_activation(
            &mut tmp,
            &out[start..start + height],
            height,
            activation,
            arch,
        );
        out[start..start + height].copy_from_slice(&tmp);
    }
}

/// Compute Conv2D layer with RTCD backend dispatch.
///
/// Upstream C RTCD tables:
/// - dnn/x86/x86_dnn_map.c:DNN_COMPUTE_CONV2D_IMPL
/// - dnn/arm/arm_dnn_map.c:DNN_COMPUTE_CONV2D_IMPL
pub fn compute_conv2d(
    conv: &Conv2dLayer,
    out: &mut [f32],
    mem: &mut [f32],
    input: &[f32],
    height: usize,
    hstride: usize,
    activation: i32,
    arch: Arch,
) {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if arch.has_avx2() {
            return x86_rtcd::compute_conv2d_avx2(
                conv, out, mem, input, height, hstride, activation, arch,
            );
        }
        if arch.has_sse4_1() {
            return x86_rtcd::compute_conv2d_sse4_1(
                conv, out, mem, input, height, hstride, activation, arch,
            );
        }
        if arch.has_sse2() {
            return x86_rtcd::compute_conv2d_sse2(
                conv, out, mem, input, height, hstride, activation, arch,
            );
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if arch.has_dotprod() {
            return arm_rtcd::compute_conv2d_dotprod(
                conv, out, mem, input, height, hstride, activation, arch,
            );
        }
        if arch.has_neon() {
            return arm_rtcd::compute_conv2d_neon(
                conv, out, mem, input, height, hstride, activation, arch,
            );
        }
    }

    compute_conv2d_c(conv, out, mem, input, height, hstride, activation, arch)
}

// RTCD backend shims mirroring upstream x86/arm map tables.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod x86_rtcd {
    use super::*;

    #[inline]
    pub(super) fn compute_activation_sse2(
        output: &mut [f32],
        input: &[f32],
        n: usize,
        activation: i32,
        arch: Arch,
    ) {
        compute_activation_c(output, input, n, activation, arch)
    }

    #[inline]
    pub(super) fn compute_activation_sse4_1(
        output: &mut [f32],
        input: &[f32],
        n: usize,
        activation: i32,
        arch: Arch,
    ) {
        compute_activation_c(output, input, n, activation, arch)
    }

    #[inline]
    pub(super) fn compute_activation_avx2(
        output: &mut [f32],
        input: &[f32],
        n: usize,
        activation: i32,
        arch: Arch,
    ) {
        compute_activation_c(output, input, n, activation, arch)
    }

    #[inline]
    pub(super) fn compute_linear_sse2(
        linear: &LinearLayer,
        out: &mut [f32],
        input: &[f32],
        arch: Arch,
    ) {
        compute_linear_c(linear, out, input, arch)
    }

    #[inline]
    pub(super) fn compute_linear_sse4_1(
        linear: &LinearLayer,
        out: &mut [f32],
        input: &[f32],
        arch: Arch,
    ) {
        compute_linear_c(linear, out, input, arch)
    }

    #[inline]
    pub(super) fn compute_linear_avx2(
        linear: &LinearLayer,
        out: &mut [f32],
        input: &[f32],
        arch: Arch,
    ) {
        compute_linear_c(linear, out, input, arch)
    }

    #[inline]
    #[allow(clippy::too_many_arguments)]
    pub(super) fn compute_conv2d_sse2(
        conv: &Conv2dLayer,
        out: &mut [f32],
        mem: &mut [f32],
        input: &[f32],
        height: usize,
        hstride: usize,
        activation: i32,
        arch: Arch,
    ) {
        compute_conv2d_c(conv, out, mem, input, height, hstride, activation, arch)
    }

    #[inline]
    #[allow(clippy::too_many_arguments)]
    pub(super) fn compute_conv2d_sse4_1(
        conv: &Conv2dLayer,
        out: &mut [f32],
        mem: &mut [f32],
        input: &[f32],
        height: usize,
        hstride: usize,
        activation: i32,
        arch: Arch,
    ) {
        compute_conv2d_c(conv, out, mem, input, height, hstride, activation, arch)
    }

    #[inline]
    #[allow(clippy::too_many_arguments)]
    pub(super) fn compute_conv2d_avx2(
        conv: &Conv2dLayer,
        out: &mut [f32],
        mem: &mut [f32],
        input: &[f32],
        height: usize,
        hstride: usize,
        activation: i32,
        arch: Arch,
    ) {
        compute_conv2d_c(conv, out, mem, input, height, hstride, activation, arch)
    }
}

#[cfg(target_arch = "aarch64")]
mod arm_rtcd {
    use super::*;

    #[inline]
    pub(super) fn compute_activation_neon(
        output: &mut [f32],
        input: &[f32],
        n: usize,
        activation: i32,
        arch: Arch,
    ) {
        compute_activation_c(output, input, n, activation, arch)
    }

    #[inline]
    pub(super) fn compute_activation_dotprod(
        output: &mut [f32],
        input: &[f32],
        n: usize,
        activation: i32,
        arch: Arch,
    ) {
        compute_activation_c(output, input, n, activation, arch)
    }

    #[inline]
    pub(super) fn compute_linear_neon(
        linear: &LinearLayer,
        out: &mut [f32],
        input: &[f32],
        arch: Arch,
    ) {
        compute_linear_c(linear, out, input, arch)
    }

    #[inline]
    pub(super) fn compute_linear_dotprod(
        linear: &LinearLayer,
        out: &mut [f32],
        input: &[f32],
        arch: Arch,
    ) {
        compute_linear_c(linear, out, input, arch)
    }

    #[inline]
    #[allow(clippy::too_many_arguments)]
    pub(super) fn compute_conv2d_neon(
        conv: &Conv2dLayer,
        out: &mut [f32],
        mem: &mut [f32],
        input: &[f32],
        height: usize,
        hstride: usize,
        activation: i32,
        arch: Arch,
    ) {
        compute_conv2d_c(conv, out, mem, input, height, hstride, activation, arch)
    }

    #[inline]
    #[allow(clippy::too_many_arguments)]
    pub(super) fn compute_conv2d_dotprod(
        conv: &Conv2dLayer,
        out: &mut [f32],
        mem: &mut [f32],
        input: &[f32],
        height: usize,
        hstride: usize,
        activation: i32,
        arch: Arch,
    ) {
        compute_conv2d_c(conv, out, mem, input, height, hstride, activation, arch)
    }
}

// --- Weight initialization ---

/// Find a named array in the weight list.
fn find_array<'a>(arrays: &'a [WeightArray], name: &str) -> Option<&'a WeightArray> {
    arrays.iter().find(|a| a.name == name)
}

/// Find a named array and verify its size matches.
fn find_array_check<'a>(arrays: &'a [WeightArray], name: &str, size: usize) -> Option<&'a [u8]> {
    let a = find_array(arrays, name)?;
    if a.size == size {
        Some(&a.data)
    } else {
        None
    }
}

/// Find an optional named array and validate size when present.
///
/// Upstream C: dnn/parse_lpcnet_weights.c:opt_array_check
fn opt_array_check<'a>(
    arrays: &'a [WeightArray],
    name: &str,
    size: usize,
) -> Option<Option<&'a [u8]>> {
    match find_array(arrays, name) {
        None => Some(None),
        Some(a) if a.size == size => Some(Some(&a.data)),
        Some(_) => None,
    }
}

/// Helper to interpret raw bytes as a slice of f32.
fn bytes_as_f32(data: &[u8]) -> Vec<f32> {
    data.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Helper to interpret raw bytes as a slice of i32.
fn bytes_as_i32(data: &[u8]) -> Vec<i32> {
    data.chunks_exact(4)
        .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Validate sparse index stream shape and bounds.
///
/// Upstream C: dnn/parse_lpcnet_weights.c:find_idx_check
fn find_idx_check(
    arrays: &[WeightArray],
    name: &str,
    nb_inputs: usize,
    nb_outputs: usize,
) -> Option<(Vec<i32>, usize)> {
    let idx_array = find_array(arrays, name)?;
    let idx_data = bytes_as_i32(&idx_array.data);

    let mut remain = idx_data.len() as i32;
    let mut out_remain = nb_outputs as i32;
    let mut idx_pos = 0usize;
    let mut total_blocks = 0usize;

    while remain > 0 {
        let nb_blocks = *idx_data.get(idx_pos)?;
        if nb_blocks < 0 || remain < nb_blocks + 1 {
            return None;
        }
        idx_pos += 1;
        for _ in 0..nb_blocks as usize {
            let pos = *idx_data.get(idx_pos)?;
            idx_pos += 1;
            if pos + 3 >= nb_inputs as i32 || (pos & 0x3) != 0 {
                return None;
            }
        }
        out_remain -= 8;
        remain -= nb_blocks + 1;
        total_blocks += nb_blocks as usize;
    }

    if out_remain != 0 {
        return None;
    }

    Some((idx_data, total_blocks))
}

/// Helper to interpret raw bytes as a slice of i8.
fn bytes_as_i8(data: &[u8]) -> Vec<i8> {
    data.iter().map(|&b| b as i8).collect()
}

/// Initialize a LinearLayer from named weight arrays.
///
/// Names can be empty strings to skip optional fields.
///
/// Upstream C: dnn/parse_lpcnet_weights.c:linear_init
pub fn linear_init(
    arrays: &[WeightArray],
    bias_name: &str,
    subias_name: &str,
    weights_name: &str,
    float_weights_name: &str,
    weights_idx_name: &str,
    diag_name: &str,
    scale_name: &str,
    nb_inputs: usize,
    nb_outputs: usize,
) -> Option<LinearLayer> {
    let mut layer = LinearLayer {
        nb_inputs,
        nb_outputs,
        ..Default::default()
    };

    if !bias_name.is_empty() {
        let data = find_array_check(arrays, bias_name, nb_outputs * 4)?;
        layer.bias = bytes_as_f32(data);
    }
    if !subias_name.is_empty() {
        let data = find_array_check(arrays, subias_name, nb_outputs * 4)?;
        layer.subias = bytes_as_f32(data);
    }
    if !weights_idx_name.is_empty() {
        let (idx_data, total_blocks) =
            find_idx_check(arrays, weights_idx_name, nb_inputs, nb_outputs)?;
        layer.weights_idx = idx_data;

        if !weights_name.is_empty() {
            let data = find_array_check(arrays, weights_name, 32 * total_blocks)?;
            layer.weights = bytes_as_i8(data);
        }
        if !float_weights_name.is_empty() {
            if let Some(data) = opt_array_check(arrays, float_weights_name, 32 * total_blocks * 4)?
            {
                layer.float_weights = bytes_as_f32(data);
            }
        }
    } else {
        if !weights_name.is_empty() {
            let data = find_array_check(arrays, weights_name, nb_inputs * nb_outputs)?;
            layer.weights = bytes_as_i8(data);
        }
        if !float_weights_name.is_empty() {
            if let Some(data) =
                opt_array_check(arrays, float_weights_name, nb_inputs * nb_outputs * 4)?
            {
                layer.float_weights = bytes_as_f32(data);
            }
        }
    }
    if !diag_name.is_empty() {
        let data = find_array_check(arrays, diag_name, nb_outputs * 4)?;
        layer.diag = bytes_as_f32(data);
    }
    if !weights_name.is_empty() && !scale_name.is_empty() {
        let data = find_array_check(arrays, scale_name, nb_outputs * 4)?;
        layer.scale = bytes_as_f32(data);
    }

    Some(layer)
}

/// Initialize a Conv2dLayer from named weight arrays.
///
/// Upstream C: dnn/parse_lpcnet_weights.c:conv2d_init
pub fn conv2d_init(
    arrays: &[WeightArray],
    bias_name: &str,
    float_weights_name: &str,
    in_channels: usize,
    out_channels: usize,
    ktime: usize,
    kheight: usize,
) -> Option<Conv2dLayer> {
    let mut layer = Conv2dLayer {
        in_channels,
        out_channels,
        ktime,
        kheight,
        ..Default::default()
    };

    if !bias_name.is_empty() {
        let data = find_array_check(arrays, bias_name, out_channels * 4)?;
        layer.bias = bytes_as_f32(data);
    }
    if !float_weights_name.is_empty() {
        let size = in_channels * out_channels * ktime * kheight * 4;
        if let Some(data) = opt_array_check(arrays, float_weights_name, size)? {
            layer.float_weights = bytes_as_f32(data);
        }
    }

    Some(layer)
}

/// Parse a binary weight blob into named arrays.
///
/// Upstream C: dnn/parse_lpcnet_weights.c:parse_weights
pub fn parse_weights(data: &[u8]) -> Option<Vec<WeightArray>> {
    let mut arrays = Vec::new();
    let mut pos = 0;
    while pos < data.len() {
        if data.len() - pos < WEIGHT_BLOCK_SIZE {
            return None;
        }
        // Parse header (64 bytes)
        let head = &data[pos..pos + 4];
        if head != b"DNNw" {
            return None;
        }
        // version at offset 4
        // type at offset 8
        let type_id =
            i32::from_le_bytes([data[pos + 8], data[pos + 9], data[pos + 10], data[pos + 11]]);
        let size_i32 = i32::from_le_bytes([
            data[pos + 12],
            data[pos + 13],
            data[pos + 14],
            data[pos + 15],
        ]);
        let block_size_i32 = i32::from_le_bytes([
            data[pos + 16],
            data[pos + 17],
            data[pos + 18],
            data[pos + 19],
        ]);

        if size_i32 <= 0 || block_size_i32 < 0 {
            return None;
        }
        let size = size_i32 as usize;
        let block_size = block_size_i32 as usize;

        // Name at offset 20, 44 bytes, null-terminated
        let name_bytes = &data[pos + 20..pos + 64];
        if name_bytes[43] != 0 {
            return None;
        }
        let name_end = name_bytes.iter().position(|&b| b == 0).unwrap_or(43);
        let name = String::from_utf8_lossy(&name_bytes[..name_end]).to_string();

        if block_size < size || block_size > data.len() - pos - WEIGHT_BLOCK_SIZE {
            return None;
        }

        let array_data = data[pos + WEIGHT_BLOCK_SIZE..pos + WEIGHT_BLOCK_SIZE + size].to_vec();
        arrays.push(WeightArray {
            name,
            type_id,
            size,
            data: array_data,
        });

        pos += WEIGHT_BLOCK_SIZE + block_size;
    }
    Some(arrays)
}

/// Serialize weight arrays to the binary "DNNw" blob format.
///
/// This is the inverse of [`parse_weights`]. Each array is written as a 64-byte
/// header followed by the data padded to a 64-byte boundary.
///
/// Upstream C: dnn/write_lpcnet_weights.c:write_weights
pub fn write_weights(arrays: &[WeightArray]) -> Vec<u8> {
    let mut out = Vec::new();
    for array in arrays {
        // 64-byte header
        let mut header = [0u8; WEIGHT_BLOCK_SIZE];
        header[0..4].copy_from_slice(b"DNNw");
        header[4..8].copy_from_slice(&WEIGHT_BLOB_VERSION.to_le_bytes());
        header[8..12].copy_from_slice(&array.type_id.to_le_bytes());
        header[12..16].copy_from_slice(&(array.size as i32).to_le_bytes());
        let block_size = array.size.div_ceil(WEIGHT_BLOCK_SIZE) * WEIGHT_BLOCK_SIZE;
        header[16..20].copy_from_slice(&(block_size as i32).to_le_bytes());
        let name_bytes = array.name.as_bytes();
        let copy_len = name_bytes.len().min(43);
        header[20..20 + copy_len].copy_from_slice(&name_bytes[..copy_len]);
        out.extend_from_slice(&header);
        // Data + zero-padding to block boundary
        out.extend_from_slice(&array.data[..array.size]);
        out.resize(out.len() + block_size - array.size, 0);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn i32_bytes(values: &[i32]) -> Vec<u8> {
        let mut out = Vec::with_capacity(values.len() * 4);
        for &v in values {
            out.extend_from_slice(&v.to_le_bytes());
        }
        out
    }

    fn test_identity_linear_layer(size: usize) -> LinearLayer {
        let mut weights = vec![0.0f32; size * size];
        for i in 0..size {
            weights[i * size + i] = 1.0;
        }
        LinearLayer {
            nb_inputs: size,
            nb_outputs: size,
            float_weights: weights,
            ..Default::default()
        }
    }

    #[test]
    fn test_compute_gated_activation_matches_glu_for_sigmoid() {
        let layer = test_identity_linear_layer(3);
        let input = [1.5f32, -0.5, 0.25];
        let mut glu_out = [0.0f32; 3];
        let mut gate_out = [0.0f32; 3];
        compute_glu(&layer, &mut glu_out, &input, Arch::default());
        compute_gated_activation(
            &layer,
            &mut gate_out,
            &input,
            ACTIVATION_SIGMOID,
            Arch::default(),
        );
        for i in 0..glu_out.len() {
            assert!((glu_out[i] - gate_out[i]).abs() < 1e-6);
        }
    }

    #[test]
    fn test_compute_gated_activation_linear() {
        let layer = test_identity_linear_layer(2);
        let input = [1.5f32, -0.5];
        let mut out = [0.0f32; 2];
        compute_gated_activation(&layer, &mut out, &input, ACTIVATION_LINEAR, Arch::default());
        assert!((out[0] - 2.25).abs() < 1e-6);
        assert!((out[1] - 0.25).abs() < 1e-6);
    }

    #[test]
    fn test_write_weights_roundtrip() {
        let original = vec![
            WeightArray {
                name: "test_bias".into(),
                type_id: WEIGHT_TYPE_FLOAT,
                size: 12,
                data: vec![0u8; 12],
            },
            WeightArray {
                name: "test_weights_int8".into(),
                type_id: WEIGHT_TYPE_INT8,
                size: 100,
                data: (0..100u8).collect(),
            },
        ];
        let blob = write_weights(&original);
        let parsed = parse_weights(&blob).unwrap();
        assert_eq!(original.len(), parsed.len());
        for (orig, p) in original.iter().zip(parsed.iter()) {
            assert_eq!(orig.name, p.name);
            assert_eq!(orig.type_id, p.type_id);
            assert_eq!(orig.size, p.size);
            assert_eq!(orig.data, p.data);
        }
    }

    #[test]
    fn test_write_weights_empty() {
        let blob = write_weights(&[]);
        assert!(blob.is_empty());
        let parsed = parse_weights(&blob).unwrap();
        assert!(parsed.is_empty());
    }

    #[test]
    fn test_write_weights_block_alignment() {
        // Data size not a multiple of 64 — verify padding
        let arrays = vec![WeightArray {
            name: "odd_size".into(),
            type_id: WEIGHT_TYPE_INT8,
            size: 7,
            data: vec![1, 2, 3, 4, 5, 6, 7],
        }];
        let blob = write_weights(&arrays);
        // Header (64) + data padded to 64 = 128 bytes total
        assert_eq!(blob.len(), 128);
        let parsed = parse_weights(&blob).unwrap();
        assert_eq!(parsed[0].data, vec![1, 2, 3, 4, 5, 6, 7]);
    }

    #[test]
    fn linear_init_rejects_sparse_idx_with_short_block_stream() {
        // nb_blocks=2 but only one block index present.
        let arrays = vec![WeightArray {
            name: "idx".into(),
            type_id: WEIGHT_TYPE_INT,
            size: 2 * 4,
            data: i32_bytes(&[2, 0]),
        }];
        let layer = linear_init(&arrays, "", "", "", "", "idx", "", "", 16, 8);
        assert!(layer.is_none());
    }

    #[test]
    fn linear_init_rejects_sparse_idx_with_unaligned_pos() {
        let arrays = vec![WeightArray {
            name: "idx".into(),
            type_id: WEIGHT_TYPE_INT,
            size: 2 * 4,
            data: i32_bytes(&[1, 2]),
        }];
        let layer = linear_init(&arrays, "", "", "", "", "idx", "", "", 16, 8);
        assert!(layer.is_none());
    }

    #[test]
    fn linear_init_rejects_sparse_idx_with_oob_pos() {
        // pos+3 >= nb_inputs should fail.
        let arrays = vec![WeightArray {
            name: "idx".into(),
            type_id: WEIGHT_TYPE_INT,
            size: 2 * 4,
            data: i32_bytes(&[1, 13]),
        }];
        let layer = linear_init(&arrays, "", "", "", "", "idx", "", "", 16, 8);
        assert!(layer.is_none());
    }

    #[test]
    fn linear_init_accepts_valid_sparse_idx_shape() {
        let arrays = vec![WeightArray {
            name: "idx".into(),
            type_id: WEIGHT_TYPE_INT,
            size: 3 * 4,
            data: i32_bytes(&[2, 0, 4]),
        }];
        let layer = linear_init(&arrays, "", "", "", "", "idx", "", "", 16, 8)
            .expect("valid sparse index stream should be accepted");
        assert_eq!(layer.weights_idx, vec![2, 0, 4]);
        assert_eq!(layer.nb_inputs, 16);
        assert_eq!(layer.nb_outputs, 8);
    }

    #[test]
    fn linear_init_rejects_dense_optional_float_weight_size_mismatch() {
        let arrays = vec![
            WeightArray {
                name: "w".into(),
                type_id: WEIGHT_TYPE_INT8,
                size: 16 * 8,
                data: vec![0; 16 * 8],
            },
            WeightArray {
                name: "fw".into(),
                type_id: WEIGHT_TYPE_FLOAT,
                size: 4,
                data: vec![0; 4],
            },
        ];
        let layer = linear_init(&arrays, "", "", "w", "fw", "", "", "", 16, 8);
        assert!(layer.is_none());
    }

    #[test]
    fn linear_init_rejects_sparse_optional_float_weight_size_mismatch() {
        let arrays = vec![
            WeightArray {
                name: "idx".into(),
                type_id: WEIGHT_TYPE_INT,
                size: 2 * 4,
                data: i32_bytes(&[1, 0]),
            },
            WeightArray {
                name: "w".into(),
                type_id: WEIGHT_TYPE_INT8,
                size: 32,
                data: vec![0; 32],
            },
            WeightArray {
                name: "fw".into(),
                type_id: WEIGHT_TYPE_FLOAT,
                size: 4,
                data: vec![0; 4],
            },
        ];
        let layer = linear_init(&arrays, "", "", "w", "fw", "idx", "", "", 16, 8);
        assert!(layer.is_none());
    }

    #[test]
    fn conv2d_init_rejects_optional_float_weight_size_mismatch() {
        let arrays = vec![WeightArray {
            name: "fw".into(),
            type_id: WEIGHT_TYPE_FLOAT,
            size: 4,
            data: vec![0; 4],
        }];
        let layer = conv2d_init(&arrays, "", "fw", 2, 3, 3, 3);
        assert!(layer.is_none());
    }

    #[test]
    fn parse_weights_rejects_zero_size_record() {
        let arrays = vec![WeightArray {
            name: "r".into(),
            type_id: WEIGHT_TYPE_INT8,
            size: 1,
            data: vec![1],
        }];
        let mut blob = write_weights(&arrays);
        blob[12..16].copy_from_slice(&0i32.to_le_bytes());
        assert!(parse_weights(&blob).is_none());
    }

    #[test]
    fn parse_weights_rejects_non_terminated_name_field() {
        let arrays = vec![WeightArray {
            name: "r".into(),
            type_id: WEIGHT_TYPE_INT8,
            size: 1,
            data: vec![1],
        }];
        let mut blob = write_weights(&arrays);
        blob[63] = b'X';
        assert!(parse_weights(&blob).is_none());
    }
}
