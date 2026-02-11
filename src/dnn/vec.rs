//! Vector math primitives for neural network inference.
//!
//! Pure Rust scalar implementations (no SIMD). These are the generic fallback
//! functions from `dnn/vec.h` — sgemv, cgemv, tanh approximation, etc.
//!
//! Upstream C: `dnn/vec.h` (generic/no-optimization path)

/// Scale factor for int8 quantized weights: `128.0 * 127.0`
pub const SCALE: f32 = 128.0 * 127.0;
/// Inverse scale factor: `1.0 / (128.0 * 127.0)`
pub const SCALE_1: f32 = 1.0 / 128.0 / 127.0;

const MAX_INPUTS: usize = 2048;

/// Fast 2^x approximation via IEEE 754 bit manipulation.
///
/// Upstream C: dnn/vec.h:lpcnet_exp2
pub fn lpcnet_exp2(x: f32) -> f32 {
    let integer = x.floor() as i32;
    if integer < -50 {
        return 0.0;
    }
    let frac = x - integer as f32;
    // K0 = 1, K1 = log(2), K2 = 3-4*log(2), K3 = 3*log(2) - 2
    let mut f = 0.99992522f32 + frac * (0.69583354 + frac * (0.22606716 + 0.078024523 * frac));
    let bits = f.to_bits();
    let bits = (bits.wrapping_add((integer as u32) << 23)) & 0x7fffffff;
    f = f32::from_bits(bits);
    f
}

/// Fast e^x approximation: `lpcnet_exp2(x * 1.44269504)`
///
/// Upstream C: dnn/vec.h:lpcnet_exp
pub fn lpcnet_exp(x: f32) -> f32 {
    lpcnet_exp2(x * 1.44269504)
}

/// Fast tanh approximation using Padé rational function.
///
/// Upstream C: dnn/vec.h:tanh_approx
pub fn tanh_approx(x: f32) -> f32 {
    const N0: f32 = 952.52801514;
    const N1: f32 = 96.39235687;
    const N2: f32 = 0.60863042;
    const D0: f32 = 952.72399902;
    const D1: f32 = 413.36801147;
    const D2: f32 = 11.88600922;

    let x2 = x * x;
    let num = (N2 * x2 + N1) * x2 + N0;
    let den = (D2 * x2 + D1) * x2 + D0;
    let result = num * x / den;
    result.clamp(-1.0, 1.0)
}

/// Fast sigmoid approximation: `0.5 + 0.5 * tanh(0.5 * x)`
///
/// Upstream C: dnn/vec.h:sigmoid_approx
pub fn sigmoid_approx(x: f32) -> f32 {
    0.5 + 0.5 * tanh_approx(0.5 * x)
}

/// Batch softmax (unnormalized exp).
///
/// Upstream C: dnn/vec.h:softmax
pub fn softmax(y: &mut [f32], x: &[f32]) {
    let n = x.len().min(y.len());
    for i in 0..n {
        y[i] = lpcnet_exp(x[i]);
    }
}

/// Batch tanh approximation.
///
/// Upstream C: dnn/vec.h:vec_tanh
pub fn vec_tanh(y: &mut [f32], x: &[f32]) {
    let n = x.len().min(y.len());
    for i in 0..n {
        y[i] = tanh_approx(x[i]);
    }
}

/// Batch sigmoid approximation.
///
/// Upstream C: dnn/vec.h:vec_sigmoid
pub fn vec_sigmoid(y: &mut [f32], x: &[f32]) {
    let n = x.len().min(y.len());
    for i in 0..n {
        y[i] = sigmoid_approx(x[i]);
    }
}

/// Dense float matrix-vector multiply: out = weights^T * x
///
/// Weights are stored column-major with `col_stride` elements per column.
/// `out` has `rows` elements, `x` has `cols` elements.
///
/// Upstream C: dnn/vec.h:sgemv
pub fn sgemv(
    out: &mut [f32],
    weights: &[f32],
    rows: usize,
    cols: usize,
    col_stride: usize,
    x: &[f32],
) {
    for i in 0..rows {
        out[i] = 0.0;
    }
    // Process in blocks of 8 when possible for better cache behavior
    if rows & 0xf == 0 {
        // 16-aligned rows
        for i in (0..rows).step_by(16) {
            for j in 0..cols {
                let w = &weights[j * col_stride + i..];
                let xj = x[j];
                for k in 0..16 {
                    out[i + k] += w[k] * xj;
                }
            }
        }
    } else if rows & 0x7 == 0 {
        // 8-aligned rows
        for i in (0..rows).step_by(8) {
            for j in 0..cols {
                let w = &weights[j * col_stride + i..];
                let xj = x[j];
                for k in 0..8 {
                    out[i + k] += w[k] * xj;
                }
            }
        }
    } else {
        for i in 0..rows {
            for j in 0..cols {
                out[i] += weights[j * col_stride + i] * x[j];
            }
        }
    }
}

/// Sparse float matrix-vector multiply (8x4 block sparse).
///
/// `idx` contains: for each 8-row block, first the number of column-blocks,
/// then for each column-block, the starting column position.
/// `w` contains the 8x4 weight blocks packed sequentially.
///
/// Upstream C: dnn/vec.h:sparse_sgemv8x4
pub fn sparse_sgemv8x4(out: &mut [f32], w: &[f32], idx: &[i32], rows: usize, x: &[f32]) {
    for i in 0..rows {
        out[i] = 0.0;
    }
    let mut w_pos = 0;
    let mut idx_pos = 0;
    for i in (0..rows).step_by(8) {
        let cols = idx[idx_pos] as usize;
        idx_pos += 1;
        for _j in 0..cols {
            let pos = idx[idx_pos] as usize;
            idx_pos += 1;
            let xj0 = x[pos];
            let xj1 = x[pos + 1];
            let xj2 = x[pos + 2];
            let xj3 = x[pos + 3];
            for k in 0..8 {
                out[i + k] += w[w_pos + k] * xj0
                    + w[w_pos + 8 + k] * xj1
                    + w[w_pos + 16 + k] * xj2
                    + w[w_pos + 24 + k] * xj3;
            }
            w_pos += 32;
        }
    }
}

/// Dense int8 matrix-vector multiply (8x4 blocking).
///
/// Input `_x` is quantized to int8 via `floor(0.5 + 127*x)`, dotted with int8
/// weights, then scaled per-output by `scale[i]`.
///
/// Upstream C: dnn/vec.h:cgemv8x4 (non-USE_SU_BIAS path)
pub fn cgemv8x4(out: &mut [f32], w: &[i8], scale: &[f32], rows: usize, cols: usize, _x: &[f32]) {
    // Quantize input to int8
    let mut x = [0i8; MAX_INPUTS];
    for i in 0..cols {
        x[i] = (0.5 + 127.0 * _x[i]).floor() as i8;
    }
    for i in 0..rows {
        out[i] = 0.0;
    }
    let mut w_pos = 0;
    for i in (0..rows).step_by(8) {
        for j in (0..cols).step_by(4) {
            let xj0 = x[j] as f32;
            let xj1 = x[j + 1] as f32;
            let xj2 = x[j + 2] as f32;
            let xj3 = x[j + 3] as f32;
            for k in 0..8 {
                out[i + k] += w[w_pos + k * 4] as f32 * xj0
                    + w[w_pos + k * 4 + 1] as f32 * xj1
                    + w[w_pos + k * 4 + 2] as f32 * xj2
                    + w[w_pos + k * 4 + 3] as f32 * xj3;
            }
            w_pos += 32;
        }
    }
    for i in 0..rows {
        out[i] *= scale[i];
    }
}

/// Sparse int8 matrix-vector multiply (8x4 block sparse).
///
/// Same quantization as `cgemv8x4` but with sparse block indices.
///
/// Upstream C: dnn/vec.h:sparse_cgemv8x4 (non-USE_SU_BIAS path)
pub fn sparse_cgemv8x4(
    out: &mut [f32],
    w: &[i8],
    idx: &[i32],
    scale: &[f32],
    rows: usize,
    cols: usize,
    _x: &[f32],
) {
    // Quantize input to int8
    let mut x = [0i8; MAX_INPUTS];
    for i in 0..cols {
        x[i] = (0.5 + 127.0 * _x[i]).floor() as i8;
    }
    for i in 0..rows {
        out[i] = 0.0;
    }
    let mut w_pos = 0;
    let mut idx_pos = 0;
    for i in (0..rows).step_by(8) {
        let colblocks = idx[idx_pos] as usize;
        idx_pos += 1;
        for _j in 0..colblocks {
            let pos = idx[idx_pos] as usize;
            idx_pos += 1;
            let xj0 = x[pos] as i32;
            let xj1 = x[pos + 1] as i32;
            let xj2 = x[pos + 2] as i32;
            let xj3 = x[pos + 3] as i32;
            for k in 0..8 {
                out[i + k] += (w[w_pos + k * 4] as i32 * xj0
                    + w[w_pos + k * 4 + 1] as i32 * xj1
                    + w[w_pos + k * 4 + 2] as i32 * xj2
                    + w[w_pos + k * 4 + 3] as i32 * xj3) as f32;
            }
            w_pos += 32;
        }
    }
    for i in 0..rows {
        out[i] *= scale[i];
    }
}
