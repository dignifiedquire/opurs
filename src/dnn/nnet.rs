//! Neural network layer types and operations.
//!
//! Core inference engine for all DNN models. Provides `LinearLayer` and
//! `Conv2dLayer` types plus compute functions for dense, GRU, Conv1D, and GLU.
//!
//! Upstream C: `dnn/nnet.h`, `dnn/nnet.c`, `dnn/nnet_arch.h`, `dnn/parse_lpcnet_weights.c`

use super::vec::*;

// --- Activation types ---

pub const ACTIVATION_LINEAR: i32 = 0;
pub const ACTIVATION_SIGMOID: i32 = 1;
pub const ACTIVATION_TANH: i32 = 2;
pub const ACTIVATION_RELU: i32 = 3;
pub const ACTIVATION_SOFTMAX: i32 = 4;
pub const ACTIVATION_SWISH: i32 = 5;

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

fn vec_swish(y: &mut [f32], x: &[f32], n: usize) {
    let mut tmp = [0.0f32; MAX_ACTIVATIONS];
    assert!(n <= MAX_ACTIVATIONS);
    vec_sigmoid(&mut tmp[..n], &x[..n]);
    for i in 0..n {
        y[i] = x[i] * tmp[i];
    }
}

/// Apply activation function in-place or from input to output.
///
/// Upstream C: dnn/nnet_arch.h:compute_activation_c
pub fn compute_activation(output: &mut [f32], input: &[f32], n: usize, activation: i32) {
    match activation {
        ACTIVATION_SIGMOID => vec_sigmoid(&mut output[..n], &input[..n]),
        ACTIVATION_TANH => vec_tanh(&mut output[..n], &input[..n]),
        ACTIVATION_SWISH => vec_swish(&mut output[..n], &input[..n], n),
        ACTIVATION_RELU => {
            for i in 0..n {
                output[i] = if input[i] < 0.0 { 0.0 } else { input[i] };
            }
        }
        ACTIVATION_SOFTMAX => {
            // SOFTMAX_HACK: just copy (used as identity in practice)
            output[..n].copy_from_slice(&input[..n]);
        }
        ACTIVATION_LINEAR | _ => {
            if !std::ptr::eq(output.as_ptr(), input.as_ptr()) {
                output[..n].copy_from_slice(&input[..n]);
            }
        }
    }
}

// --- Linear computation ---

/// Compute affine transform: out = W*in + bias + diag*in
///
/// Dispatches to float sgemv or int8 cgemv depending on which weights are present.
///
/// Upstream C: dnn/nnet_arch.h:compute_linear_c
pub fn compute_linear(linear: &LinearLayer, out: &mut [f32], input: &[f32]) {
    let m = linear.nb_inputs;
    let n = linear.nb_outputs;

    if !linear.float_weights.is_empty() {
        if !linear.weights_idx.is_empty() {
            sparse_sgemv8x4(out, &linear.float_weights, &linear.weights_idx, n, input);
        } else {
            sgemv(out, &linear.float_weights, n, m, n, input);
        }
    } else if !linear.weights.is_empty() {
        if !linear.weights_idx.is_empty() {
            sparse_cgemv8x4(
                out,
                &linear.weights,
                &linear.weights_idx,
                &linear.scale,
                n,
                m,
                input,
            );
        } else {
            cgemv8x4(out, &linear.weights, &linear.scale, n, m, input);
        }
    } else {
        for i in 0..n {
            out[i] = 0.0;
        }
    }

    let bias = if use_su_bias() && !linear.subias.is_empty() && !linear.weights.is_empty() {
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
        // Diag is only used for GRU recurrent weights: 3*M == N.
        // Uses FMA to match C's auto-vectorized loop in compute_linear_avx2.
        assert!(3 * m == n);
        apply_diag(out, &linear.diag, input, m);
    }
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
) {
    compute_linear(layer, output, input);
    let n = layer.nb_outputs;
    let tmp = output[..n].to_vec();
    compute_activation(&mut output[..n], &tmp, n, activation);
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
) {
    assert!(3 * recurrent_weights.nb_inputs == recurrent_weights.nb_outputs);
    assert!(input_weights.nb_outputs == recurrent_weights.nb_outputs);

    let n = recurrent_weights.nb_inputs;
    let mut zrh = vec![0.0f32; 3 * n];
    let mut recur = vec![0.0f32; 3 * n];

    compute_linear(input_weights, &mut zrh, input);
    compute_linear(recurrent_weights, &mut recur, state);

    // z and r: add recurrent, then sigmoid
    for i in 0..2 * n {
        zrh[i] += recur[i];
    }
    let tmp_zr = zrh[..2 * n].to_vec();
    compute_activation(&mut zrh[..2 * n], &tmp_zr, 2 * n, ACTIVATION_SIGMOID);

    // h: add r-gated recurrent, then tanh
    for i in 0..n {
        zrh[2 * n + i] += recur[2 * n + i] * zrh[n + i]; // r[i] gates recurrent
    }
    let mut h = vec![0.0f32; n];
    compute_activation(&mut h, &zrh[2 * n..], n, ACTIVATION_TANH);

    // state = z*state + (1-z)*h
    for i in 0..n {
        state[i] = zrh[i] * state[i] + (1.0 - zrh[i]) * h[i];
    }
}

/// Gated Linear Unit: out = input * sigmoid(W*input)
///
/// Upstream C: dnn/nnet.c:compute_glu
pub fn compute_glu(layer: &LinearLayer, output: &mut [f32], input: &[f32]) {
    assert!(layer.nb_inputs == layer.nb_outputs);
    let n = layer.nb_outputs;
    let mut act = vec![0.0f32; n];
    compute_linear(layer, &mut act, input);
    let tmp = act.clone();
    compute_activation(&mut act, &tmp, n, ACTIVATION_SIGMOID);
    for i in 0..n {
        output[i] = input[i] * act[i];
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
) {
    let mut tmp = vec![0.0f32; MAX_CONV_INPUTS_ALL];
    assert!(layer.nb_inputs <= MAX_CONV_INPUTS_ALL);
    let hist_size = layer.nb_inputs - input_size;
    if hist_size > 0 {
        tmp[..hist_size].copy_from_slice(&mem[..hist_size]);
    }
    tmp[hist_size..hist_size + input_size].copy_from_slice(&input[..input_size]);
    compute_linear(layer, output, &tmp);
    let n = layer.nb_outputs;
    let out_tmp = output[..n].to_vec();
    compute_activation(&mut output[..n], &out_tmp, n, activation);
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
) {
    let mut tmp = vec![0.0f32; MAX_CONV_INPUTS_ALL];
    let ksize = layer.nb_inputs / input_size;
    assert!(layer.nb_inputs <= MAX_CONV_INPUTS_ALL);

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

    compute_linear(layer, output, &tmp);
    let mut out_copy = output[..layer.nb_outputs].to_vec();
    compute_activation(
        &mut out_copy,
        &output[..layer.nb_outputs],
        layer.nb_outputs,
        activation,
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

/// Compute Conv2D layer with temporal memory.
///
/// Upstream C: dnn/nnet_arch.h:compute_conv2d_c
pub fn compute_conv2d(
    conv: &Conv2dLayer,
    out: &mut [f32],
    mem: &mut [f32],
    input: &[f32],
    height: usize,
    hstride: usize,
    activation: i32,
) {
    let time_stride = conv.in_channels * (height + conv.kheight - 1);
    assert!(conv.ktime * time_stride <= MAX_CONV2D_INPUTS);
    let mut in_buf = vec![0.0f32; MAX_CONV2D_INPUTS];

    // Copy history from mem, then current input
    let hist_size = (conv.ktime - 1) * time_stride;
    in_buf[..hist_size].copy_from_slice(&mem[..hist_size]);
    in_buf[hist_size..hist_size + time_stride].copy_from_slice(&input[..time_stride]);
    // Shift memory
    mem[..hist_size].copy_from_slice(&in_buf[time_stride..time_stride + hist_size]);

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
        compute_activation(&mut tmp, &out[start..start + height], height, activation);
        out[start..start + height].copy_from_slice(&tmp);
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
        let idx_array = find_array(arrays, weights_idx_name)?;
        let idx_data = bytes_as_i32(&idx_array.data);
        // Count total sparse blocks for weight size validation
        let mut total_blocks = 0;
        let mut pos = 0;
        let mut remaining_outputs = nb_outputs as i32;
        while pos < idx_data.len() {
            let nb_blocks = idx_data[pos] as usize;
            pos += 1 + nb_blocks;
            total_blocks += nb_blocks;
            remaining_outputs -= 8;
        }
        if remaining_outputs != 0 {
            return None;
        }
        layer.weights_idx = idx_data;

        if !weights_name.is_empty() {
            let data = find_array_check(arrays, weights_name, 32 * total_blocks)?;
            layer.weights = bytes_as_i8(data);
        }
        if !float_weights_name.is_empty() {
            if let Some(data) = find_array_check(arrays, float_weights_name, 32 * total_blocks * 4)
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
                find_array_check(arrays, float_weights_name, nb_inputs * nb_outputs * 4)
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
        if let Some(data) = find_array_check(arrays, float_weights_name, size) {
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
        let size = i32::from_le_bytes([
            data[pos + 12],
            data[pos + 13],
            data[pos + 14],
            data[pos + 15],
        ]) as usize;
        let block_size = i32::from_le_bytes([
            data[pos + 16],
            data[pos + 17],
            data[pos + 18],
            data[pos + 19],
        ]) as usize;

        // Name at offset 20, 44 bytes, null-terminated
        let name_bytes = &data[pos + 20..pos + 64];
        let name_end = name_bytes.iter().position(|&b| b == 0).unwrap_or(44);
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
}
