//! Vector math primitives for neural network inference.
//!
//! Pure Rust scalar implementations (no SIMD). These are the generic fallback
//! functions from `dnn/vec.h` — sgemv, cgemv, tanh approximation, etc.
//!
//! Upstream C: `dnn/vec.h` (generic/no-optimization path)

use crate::arch::Arch;

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

/// Batch softmax (unnormalized exp) — scalar implementation.
///
/// Upstream C: dnn/vec.h:softmax
pub fn softmax_scalar(y: &mut [f32], x: &[f32]) {
    let n = x.len().min(y.len());
    for i in 0..n {
        y[i] = lpcnet_exp(x[i]);
    }
}

/// Batch tanh approximation — scalar implementation.
///
/// Upstream C: dnn/vec.h:vec_tanh
pub fn vec_tanh_scalar(y: &mut [f32], x: &[f32]) {
    let n = x.len().min(y.len());
    for i in 0..n {
        y[i] = tanh_approx(x[i]);
    }
}

/// Batch sigmoid approximation — scalar implementation.
///
/// Upstream C: dnn/vec.h:vec_sigmoid
pub fn vec_sigmoid_scalar(y: &mut [f32], x: &[f32]) {
    let n = x.len().min(y.len());
    for i in 0..n {
        y[i] = sigmoid_approx(x[i]);
    }
}

// -- Dispatch wrappers --

/// Returns `true` when the active int8 GEMV path uses unsigned u8 quantization,
/// meaning `subias` should be used instead of `bias` for int8 weight layers.
///
/// Upstream C defines `USE_SU_BIAS` for all x86 `vec_avx.h` variants (AVX2 and
/// SSE2/SSE4 emulation paths), not just AVX2.
///
/// Upstream C: dnn/vec_avx.h:USE_SU_BIAS
#[cfg(feature = "simd")]
#[inline]
pub fn use_su_bias(arch: Arch) -> bool {
    super::simd::use_su_bias(arch)
}

/// Scalar-only build: always `false` (scalar cgemv8x4 uses signed i8 quantization).
#[cfg(not(feature = "simd"))]
#[inline]
pub fn use_su_bias(_arch: Arch) -> bool {
    false
}

/// Dispatch wrapper for `softmax`.
#[cfg(feature = "simd")]
#[inline]
pub fn softmax(y: &mut [f32], x: &[f32], arch: Arch) {
    super::simd::softmax(y, x, arch)
}

/// Dispatch wrapper for `softmax` (scalar-only build).
#[cfg(not(feature = "simd"))]
#[inline]
pub fn softmax(y: &mut [f32], x: &[f32], _arch: Arch) {
    softmax_scalar(y, x)
}

/// Dispatch wrapper for `vec_tanh`.
#[cfg(feature = "simd")]
#[inline]
pub fn vec_tanh(y: &mut [f32], x: &[f32], arch: Arch) {
    super::simd::vec_tanh(y, x, arch)
}

/// Dispatch wrapper for `vec_tanh` (scalar-only build).
#[cfg(not(feature = "simd"))]
#[inline]
pub fn vec_tanh(y: &mut [f32], x: &[f32], _arch: Arch) {
    vec_tanh_scalar(y, x)
}

/// Dispatch wrapper for `vec_sigmoid`.
#[cfg(feature = "simd")]
#[inline]
pub fn vec_sigmoid(y: &mut [f32], x: &[f32], arch: Arch) {
    super::simd::vec_sigmoid(y, x, arch)
}

/// Dispatch wrapper for `vec_sigmoid` (scalar-only build).
#[cfg(not(feature = "simd"))]
#[inline]
pub fn vec_sigmoid(y: &mut [f32], x: &[f32], _arch: Arch) {
    vec_sigmoid_scalar(y, x)
}

/// Dispatch wrapper for `sgemv`.
#[cfg(feature = "simd")]
#[inline]
pub fn sgemv(
    out: &mut [f32],
    weights: &[f32],
    rows: usize,
    cols: usize,
    col_stride: usize,
    x: &[f32],
    arch: Arch,
) {
    super::simd::sgemv(out, weights, rows, cols, col_stride, x, arch)
}

/// Dispatch wrapper for `sgemv` (scalar-only build).
#[cfg(not(feature = "simd"))]
#[inline]
pub fn sgemv(
    out: &mut [f32],
    weights: &[f32],
    rows: usize,
    cols: usize,
    col_stride: usize,
    x: &[f32],
    _arch: Arch,
) {
    sgemv_scalar(out, weights, rows, cols, col_stride, x)
}

/// Dispatch wrapper for `sparse_sgemv8x4`.
#[cfg(feature = "simd")]
#[inline]
pub fn sparse_sgemv8x4(
    out: &mut [f32],
    w: &[f32],
    idx: &[i32],
    rows: usize,
    x: &[f32],
    arch: Arch,
) {
    super::simd::sparse_sgemv8x4(out, w, idx, rows, x, arch)
}

/// Dispatch wrapper for `sparse_sgemv8x4` (scalar-only build).
#[cfg(not(feature = "simd"))]
#[inline]
pub fn sparse_sgemv8x4(
    out: &mut [f32],
    w: &[f32],
    idx: &[i32],
    rows: usize,
    x: &[f32],
    _arch: Arch,
) {
    sparse_sgemv8x4_scalar(out, w, idx, rows, x)
}

/// Dispatch wrapper for `cgemv8x4`.
#[cfg(feature = "simd")]
#[inline]
pub fn cgemv8x4(
    out: &mut [f32],
    w: &[i8],
    scale: &[f32],
    rows: usize,
    cols: usize,
    x: &[f32],
    arch: Arch,
) {
    super::simd::cgemv8x4(out, w, scale, rows, cols, x, arch)
}

/// Dispatch wrapper for `cgemv8x4` (scalar-only build).
#[cfg(not(feature = "simd"))]
#[inline]
pub fn cgemv8x4(
    out: &mut [f32],
    w: &[i8],
    scale: &[f32],
    rows: usize,
    cols: usize,
    x: &[f32],
    _arch: Arch,
) {
    cgemv8x4_scalar(out, w, scale, rows, cols, x)
}

/// Dispatch wrapper for `sparse_cgemv8x4`.
#[cfg(feature = "simd")]
#[inline]
pub fn sparse_cgemv8x4(
    out: &mut [f32],
    w: &[i8],
    idx: &[i32],
    scale: &[f32],
    rows: usize,
    cols: usize,
    x: &[f32],
    arch: Arch,
) {
    super::simd::sparse_cgemv8x4(out, w, idx, scale, rows, cols, x, arch)
}

/// Dispatch wrapper for `sparse_cgemv8x4` (scalar-only build).
#[cfg(not(feature = "simd"))]
#[inline]
pub fn sparse_cgemv8x4(
    out: &mut [f32],
    w: &[i8],
    idx: &[i32],
    scale: &[f32],
    rows: usize,
    cols: usize,
    x: &[f32],
    _arch: Arch,
) {
    sparse_cgemv8x4_scalar(out, w, idx, scale, rows, cols, x)
}

/// Dense float matrix-vector multiply: out = weights^T * x — scalar implementation.
///
/// Weights are stored column-major with `col_stride` elements per column.
/// `out` has `rows` elements, `x` has `cols` elements.
///
/// Upstream C: dnn/vec.h:sgemv
pub fn sgemv_scalar(
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

/// Sparse float matrix-vector multiply (8x4 block sparse) — scalar implementation.
///
/// `idx` contains: for each 8-row block, first the number of column-blocks,
/// then for each column-block, the starting column position.
/// `w` contains the 8x4 weight blocks packed sequentially.
///
/// Upstream C: dnn/vec.h:sparse_sgemv8x4
pub fn sparse_sgemv8x4_scalar(out: &mut [f32], w: &[f32], idx: &[i32], rows: usize, x: &[f32]) {
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

/// Dense int8 matrix-vector multiply (8x4 blocking) — scalar implementation.
///
/// Input `_x` is quantized to int8 via `floor(0.5 + 127*x)`, dotted with int8
/// weights, then scaled per-output by `scale[i]`.
///
/// Upstream C: dnn/vec.h:cgemv8x4 (non-USE_SU_BIAS path)
pub fn cgemv8x4_scalar(
    out: &mut [f32],
    w: &[i8],
    scale: &[f32],
    rows: usize,
    cols: usize,
    _x: &[f32],
) {
    // Quantize input to int8
    let mut x = [0i8; MAX_INPUTS];
    for i in 0..cols {
        x[i] = (0.5f64 + 127.0f64 * _x[i] as f64).floor() as i8;
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

/// Dense int8 matrix-vector multiply (8x4 blocking), SU-bias variant.
///
/// Input `_x` is quantized to unsigned u8 via `127 + floor(0.5 + 127*x)`,
/// matching x86 `USE_SU_BIAS` behavior in upstream C (`vec_avx.h` and `vec.h`).
///
/// Upstream C: dnn/vec.h:cgemv8x4 (USE_SU_BIAS path)
pub fn cgemv8x4_scalar_su(
    out: &mut [f32],
    w: &[i8],
    scale: &[f32],
    rows: usize,
    cols: usize,
    _x: &[f32],
) {
    let mut x = [0u8; MAX_INPUTS];
    for i in 0..cols {
        x[i] = (127.0f64 + (0.5f64 + 127.0f64 * _x[i] as f64).floor()) as u8;
    }
    let mut acc = vec![0i32; rows];
    let mut w_pos = 0;
    for i in (0..rows).step_by(8) {
        for j in (0..cols).step_by(4) {
            let xj0 = x[j] as i32;
            let xj1 = x[j + 1] as i32;
            let xj2 = x[j + 2] as i32;
            let xj3 = x[j + 3] as i32;
            for k in 0..8 {
                acc[i + k] += w[w_pos + k * 4] as i32 * xj0
                    + w[w_pos + k * 4 + 1] as i32 * xj1
                    + w[w_pos + k * 4 + 2] as i32 * xj2
                    + w[w_pos + k * 4 + 3] as i32 * xj3;
            }
            w_pos += 32;
        }
    }
    for i in 0..rows {
        out[i] = acc[i] as f32 * scale[i];
    }
}

#[inline]
fn sat_i16(v: i32) -> i32 {
    v.clamp(i16::MIN as i32, i16::MAX as i32)
}

/// Dense int8 matrix-vector multiply (8x4 blocking), SU-bias + SSSE3 emulation.
///
/// Emulates `vec_avx.h` `opus_mm256_dpbusds_epi32` semantics when implemented via
/// `_mm_maddubs_epi16`: pairwise unsigned*signed products are first summed with
/// i16 saturation, then accumulated in i32.
///
/// Upstream C: dnn/vec_avx.h:cgemv8x4 (SSSE3/SSE4 path)
pub fn cgemv8x4_scalar_su_ssse3(
    out: &mut [f32],
    w: &[i8],
    scale: &[f32],
    rows: usize,
    cols: usize,
    _x: &[f32],
) {
    let mut x = [0u8; MAX_INPUTS];
    for i in 0..cols {
        x[i] = (127.0f64 + (0.5f64 + 127.0f64 * _x[i] as f64).floor()) as u8;
    }
    let mut acc = vec![0i32; rows];
    let mut w_pos = 0;
    for i in (0..rows).step_by(8) {
        for j in (0..cols).step_by(4) {
            let xj0 = x[j] as i32;
            let xj1 = x[j + 1] as i32;
            let xj2 = x[j + 2] as i32;
            let xj3 = x[j + 3] as i32;
            for k in 0..8 {
                let w0 = w[w_pos + k * 4] as i32;
                let w1 = w[w_pos + k * 4 + 1] as i32;
                let w2 = w[w_pos + k * 4 + 2] as i32;
                let w3 = w[w_pos + k * 4 + 3] as i32;
                let p01 = sat_i16(w0 * xj0 + w1 * xj1);
                let p23 = sat_i16(w2 * xj2 + w3 * xj3);
                acc[i + k] += p01 + p23;
            }
            w_pos += 32;
        }
    }
    for i in 0..rows {
        out[i] = acc[i] as f32 * scale[i];
    }
}

/// Sparse int8 matrix-vector multiply (8x4 block sparse) — scalar implementation.
///
/// Same quantization as `cgemv8x4` but with sparse block indices.
///
/// Upstream C: dnn/vec.h:sparse_cgemv8x4 (non-USE_SU_BIAS path)
pub fn sparse_cgemv8x4_scalar(
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
        x[i] = (0.5f64 + 127.0f64 * _x[i] as f64).floor() as i8;
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

/// Sparse int8 matrix-vector multiply (8x4 block sparse), SU-bias variant.
///
/// Input `_x` is quantized to unsigned u8 via `127 + floor(0.5 + 127*x)`,
/// matching x86 `USE_SU_BIAS` behavior in upstream C.
///
/// Upstream C: dnn/vec.h:sparse_cgemv8x4 (USE_SU_BIAS path)
pub fn sparse_cgemv8x4_scalar_su(
    out: &mut [f32],
    w: &[i8],
    idx: &[i32],
    scale: &[f32],
    rows: usize,
    cols: usize,
    _x: &[f32],
) {
    let mut x = [0u8; MAX_INPUTS];
    for i in 0..cols {
        x[i] = (127.0f64 + (0.5f64 + 127.0f64 * _x[i] as f64).floor()) as u8;
    }
    let mut acc = vec![0i32; rows];
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
                acc[i + k] += w[w_pos + k * 4] as i32 * xj0
                    + w[w_pos + k * 4 + 1] as i32 * xj1
                    + w[w_pos + k * 4 + 2] as i32 * xj2
                    + w[w_pos + k * 4 + 3] as i32 * xj3;
            }
            w_pos += 32;
        }
    }
    for i in 0..rows {
        out[i] = acc[i] as f32 * scale[i];
    }
}

/// Sparse int8 matrix-vector multiply (8x4 block sparse), SU-bias + SSSE3 emulation.
///
/// Emulates `_mm_maddubs_epi16` pairwise i16 saturation semantics from upstream
/// x86 SSE4/SSSE3 path in `vec_avx.h`.
pub fn sparse_cgemv8x4_scalar_su_ssse3(
    out: &mut [f32],
    w: &[i8],
    idx: &[i32],
    scale: &[f32],
    rows: usize,
    cols: usize,
    _x: &[f32],
) {
    let mut x = [0u8; MAX_INPUTS];
    for i in 0..cols {
        x[i] = (127.0f64 + (0.5f64 + 127.0f64 * _x[i] as f64).floor()) as u8;
    }
    let mut acc = vec![0i32; rows];
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
                let w0 = w[w_pos + k * 4] as i32;
                let w1 = w[w_pos + k * 4 + 1] as i32;
                let w2 = w[w_pos + k * 4 + 2] as i32;
                let w3 = w[w_pos + k * 4 + 3] as i32;
                let p01 = sat_i16(w0 * xj0 + w1 * xj1);
                let p23 = sat_i16(w2 * xj2 + w3 * xj3);
                acc[i + k] += p01 + p23;
            }
            w_pos += 32;
        }
    }
    for i in 0..rows {
        out[i] = acc[i] as f32 * scale[i];
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arch::{opus_select_arch, Arch};

    fn select_arch() -> Arch {
        opus_select_arch()
    }

    fn gen_signal(len: usize, seed: u32) -> Vec<f32> {
        let mut v = Vec::with_capacity(len);
        let mut state = seed;
        for _ in 0..len {
            state = state.wrapping_mul(1103515245).wrapping_add(12345);
            v.push((state as i32 >> 16) as f32 / 32768.0);
        }
        v
    }

    fn gen_weights_f32(n: usize, seed: u32) -> Vec<f32> {
        gen_signal(n, seed)
    }

    fn gen_weights_i8(n: usize, seed: u32) -> Vec<i8> {
        let mut v = Vec::with_capacity(n);
        let mut state = seed;
        for _ in 0..n {
            state = state.wrapping_mul(1103515245).wrapping_add(12345);
            v.push(((state >> 16) as i32 % 128) as i8);
        }
        v
    }

    #[test]
    fn test_vec_tanh_scalar_approx() {
        // Scalar Padé approximation with true division: max error ~6e-5 vs f64 tanh.
        for &n in &[1, 4, 7, 8, 15, 16, 64, 256] {
            let x = gen_signal(n, 42);
            let mut y_scalar = vec![0.0f32; n];
            vec_tanh_scalar(&mut y_scalar, &x);
            for i in 0..n {
                let reference = (x[i] as f64).tanh() as f32;
                assert!(
                    (y_scalar[i] - reference).abs() < 1e-4,
                    "vec_tanh_scalar vs f64 tanh at [{}]: scalar={} ref={} (n={})",
                    i,
                    y_scalar[i],
                    reference,
                    n
                );
            }
        }
    }

    #[test]
    fn test_vec_tanh_dispatch_approx() {
        // SIMD paths use approximate reciprocal (vrecpe/rcp_ps) instead of
        // true division. x86 _mm256_rcp_ps gives ~12-bit precision (max err ~3e-4),
        // ARM vrecpeq_f32 gives ~8-bit precision (max err ~2e-3).
        for &n in &[1, 4, 7, 8, 15, 16, 64, 256] {
            let arch = select_arch();
            let x = gen_signal(n, 42);
            let mut y_dispatch = vec![0.0f32; n];
            vec_tanh(&mut y_dispatch, &x, arch);
            for i in 0..n {
                let reference = (x[i] as f64).tanh() as f32;
                assert!(
                    (y_dispatch[i] - reference).abs() < 2e-3,
                    "vec_tanh dispatch vs f64 tanh at [{}]: dispatch={} ref={} (n={})",
                    i,
                    y_dispatch[i],
                    reference,
                    n
                );
            }
        }
    }

    #[test]
    fn test_vec_sigmoid_scalar_approx() {
        // Scalar sigmoid = 0.5 + 0.5 * tanh(0.5*x): max error ~3e-5.
        for &n in &[1, 4, 7, 8, 15, 16, 64, 256] {
            let x = gen_signal(n, 123);
            let mut y_scalar = vec![0.0f32; n];
            vec_sigmoid_scalar(&mut y_scalar, &x);
            for i in 0..n {
                let reference = (1.0 / (1.0 + (-(x[i] as f64)).exp())) as f32;
                assert!(
                    (y_scalar[i] - reference).abs() < 1e-4,
                    "vec_sigmoid_scalar vs f64 sigmoid at [{}]: scalar={} ref={} (n={})",
                    i,
                    y_scalar[i],
                    reference,
                    n
                );
            }
        }
    }

    #[test]
    fn test_vec_sigmoid_dispatch_approx() {
        // SIMD sigmoid uses approximate reciprocal: max error ~2e-4 (x86),
        // ~1e-3 (ARM vrecpeq_f32 with ~8-bit precision).
        for &n in &[1, 4, 7, 8, 15, 16, 64, 256] {
            let arch = select_arch();
            let x = gen_signal(n, 123);
            let mut y_dispatch = vec![0.0f32; n];
            vec_sigmoid(&mut y_dispatch, &x, arch);
            for i in 0..n {
                let reference = (1.0 / (1.0 + (-(x[i] as f64)).exp())) as f32;
                assert!(
                    (y_dispatch[i] - reference).abs() < 2e-3,
                    "vec_sigmoid dispatch vs f64 sigmoid at [{}]: dispatch={} ref={} (n={})",
                    i,
                    y_dispatch[i],
                    reference,
                    n
                );
            }
        }
    }

    #[test]
    fn test_softmax_dispatch_matches_scalar() {
        for &n in &[1, 4, 7, 8, 15, 16, 64, 256] {
            let arch = select_arch();
            let x = gen_signal(n, 77);
            let mut y_scalar = vec![0.0f32; n];
            let mut y_dispatch = vec![0.0f32; n];
            softmax_scalar(&mut y_scalar, &x);
            softmax(&mut y_dispatch, &x, arch);
            for i in 0..n {
                let tol = y_scalar[i].abs() * 1e-4 + 1e-6;
                assert!(
                    (y_scalar[i] - y_dispatch[i]).abs() < tol,
                    "softmax mismatch at [{}]: scalar={} dispatch={} (n={})",
                    i,
                    y_scalar[i],
                    y_dispatch[i],
                    n
                );
            }
        }
    }

    #[test]
    fn test_sgemv_dispatch_matches_scalar() {
        for &(rows, cols) in &[(8, 8), (16, 8), (16, 16), (64, 32), (128, 64)] {
            let arch = select_arch();
            let weights = gen_weights_f32(rows * cols, 42);
            let x = gen_signal(cols, 123);
            let mut out_scalar = vec![0.0f32; rows];
            let mut out_dispatch = vec![0.0f32; rows];
            sgemv_scalar(&mut out_scalar, &weights, rows, cols, rows, &x);
            sgemv(&mut out_dispatch, &weights, rows, cols, rows, &x, arch);
            for i in 0..rows {
                let tol = out_scalar[i].abs() * 1e-4 + 1e-4;
                assert!(
                    (out_scalar[i] - out_dispatch[i]).abs() < tol,
                    "sgemv mismatch at [{}]: scalar={} dispatch={} ({}x{})",
                    i,
                    out_scalar[i],
                    out_dispatch[i],
                    rows,
                    cols
                );
            }
        }
    }

    #[test]
    fn test_cgemv8x4_dispatch_runs_without_error() {
        // NOTE: SIMD cgemv8x4 uses unsigned u8 quantization (127 + round(127*x))
        // while scalar uses signed i8 (round(0.5 + 127*x)). These produce different
        // results intentionally — the subias in LinearLayer compensates at the bias step.
        // Here we just verify the dispatch runs without error and produces finite results.
        for &(rows, cols) in &[(8, 8), (16, 16), (64, 32)] {
            let arch = select_arch();
            let w = gen_weights_i8(rows * cols, 42);
            let scale: Vec<f32> = (0..rows).map(|i| 0.01 + 0.001 * i as f32).collect();
            let x = gen_signal(cols, 99);
            let mut out = vec![0.0f32; rows];
            cgemv8x4(&mut out, &w, &scale, rows, cols, &x, arch);
            for i in 0..rows {
                assert!(
                    out[i].is_finite(),
                    "cgemv8x4 non-finite at [{}]: {} ({}x{})",
                    i,
                    out[i],
                    rows,
                    cols
                );
            }
        }
    }

    #[test]
    fn test_sparse_sgemv8x4_dispatch_matches_scalar() {
        // 16 output rows, 2 blocks of 8. Each block has 2 column-blocks of 4 cols.
        let arch = select_arch();
        let rows = 16;
        let idx = vec![
            2i32, 0, 4, // block 0: 2 col-blocks at positions 0, 4
            2, 0, 4, // block 1: 2 col-blocks at positions 0, 4
        ];
        let w = gen_weights_f32(32 * 4, 42); // 4 col-blocks * 32 floats each
        let x = gen_signal(8, 123);
        let mut out_scalar = vec![0.0f32; rows];
        let mut out_dispatch = vec![0.0f32; rows];
        sparse_sgemv8x4_scalar(&mut out_scalar, &w, &idx, rows, &x);
        sparse_sgemv8x4(&mut out_dispatch, &w, &idx, rows, &x, arch);
        for i in 0..rows {
            let tol = out_scalar[i].abs() * 1e-4 + 1e-4;
            assert!(
                (out_scalar[i] - out_dispatch[i]).abs() < tol,
                "sparse_sgemv8x4 mismatch at [{}]: scalar={} dispatch={}",
                i,
                out_scalar[i],
                out_dispatch[i]
            );
        }
    }

    #[test]
    fn test_sparse_cgemv8x4_dispatch_runs_without_error() {
        // Same note as cgemv8x4: SIMD uses u8 quantization, scalar uses i8.
        // Just verify the dispatch runs and produces finite results.
        let arch = select_arch();
        let rows = 8;
        let cols = 8;
        let idx = vec![2i32, 0, 4]; // 2 col-blocks at positions 0 and 4
        let w = gen_weights_i8(32 * 2, 42); // 2 col-blocks * 32 bytes each
        let scale: Vec<f32> = (0..rows).map(|i| 0.01 + 0.001 * i as f32).collect();
        let x = gen_signal(cols, 99);
        let mut out = vec![0.0f32; rows];
        sparse_cgemv8x4(&mut out, &w, &idx, &scale, rows, cols, &x, arch);
        for i in 0..rows {
            assert!(
                out[i].is_finite(),
                "sparse_cgemv8x4 non-finite at [{}]: {}",
                i,
                out[i]
            );
        }
    }
}
