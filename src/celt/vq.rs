use crate::celt::bands::SPREAD_NONE;
use crate::celt::cwrs::{decode_pulses, encode_pulses};
use crate::celt::entcode::celt_udiv;
use crate::celt::entdec::ec_dec;
use crate::celt::entenc::ec_enc;
use crate::celt::mathops::{celt_cos_norm, celt_rsqrt_norm, celt_sqrt, fast_atan2f};
use crate::celt::pitch::celt_inner_prod;

const EPSILON: f32 = 1e-15f32;

/// Upstream C: celt/vq.c:exp_rotation1
fn exp_rotation1(X: &mut [f32], len: i32, stride: i32, c: f32, s: f32) {
    let ms: f32 = -s;
    // Forward pass
    let fwd_end = len - stride;
    if fwd_end > 0 {
        for i in 0..fwd_end as usize {
            let x1 = X[i];
            let x2 = X[i + stride as usize];
            X[i + stride as usize] = c * x2 + s * x1;
            X[i] = c * x1 + ms * x2;
        }
    }
    // Backward pass
    let bwd_end = len - 2 * stride - 1;
    if bwd_end >= 0 {
        for i in (0..=bwd_end as usize).rev() {
            let x1 = X[i];
            let x2 = X[i + stride as usize];
            X[i + stride as usize] = c * x2 + s * x1;
            X[i] = c * x1 + ms * x2;
        }
    }
}

/// Upstream C: celt/vq.c:exp_rotation
pub fn exp_rotation(X: &mut [f32], mut len: i32, dir: i32, stride: i32, K: i32, spread: i32) {
    static SPREAD_FACTOR: [i32; 3] = [15, 10, 5];
    let mut stride2: i32 = 0;
    if 2 * K >= len || spread == SPREAD_NONE {
        return;
    }
    let factor = SPREAD_FACTOR[(spread - 1) as usize];
    let gain: f32 = 1.0f32 * len as f32 / (len + factor * K) as f32;
    let theta: f32 = 0.5f32 * (gain * gain);
    let c = celt_cos_norm(theta);
    let s = celt_cos_norm(1.0f32 - theta);
    if len >= 8 * stride {
        stride2 = 1;
        while (stride2 * stride2 + stride2) * stride + (stride >> 2) < len {
            stride2 += 1;
        }
    }
    len = celt_udiv(len as u32, stride as u32) as i32;
    for i in 0..stride {
        let off = (i * len) as usize;
        let sub = &mut X[off..off + len as usize];
        if dir < 0 {
            if stride2 != 0 {
                exp_rotation1(sub, len, stride2, s, c);
            }
            exp_rotation1(sub, len, 1, c, s);
        } else {
            exp_rotation1(sub, len, 1, c, -s);
            if stride2 != 0 {
                exp_rotation1(sub, len, stride2, s, -c);
            }
        }
    }
}

/// Upstream C: celt/vq.c:normalise_residual
fn normalise_residual(iy: &[i32], X: &mut [f32], N: i32, Ryy: f32, gain: f32) {
    let g = celt_rsqrt_norm(Ryy) * gain;
    for i in 0..N as usize {
        X[i] = g * iy[i] as f32;
    }
}

/// Upstream C: celt/vq.c:extract_collapse_mask
fn extract_collapse_mask(iy: &[i32], N: i32, B: i32) -> u32 {
    if B <= 1 {
        return 1;
    }
    let N0 = celt_udiv(N as u32, B as u32) as i32;
    let mut collapse_mask: u32 = 0;
    for i in 0..B {
        let mut tmp: u32 = 0;
        for j in 0..N0 {
            tmp |= iy[(i * N0 + j) as usize] as u32;
        }
        collapse_mask |= ((tmp != 0) as u32) << i;
    }
    collapse_mask
}

/// Upstream C: celt/vq.c:op_pvq_search_c
pub fn op_pvq_search_c(X: &mut [f32], iy: &mut [i32], K: i32, N: i32, _arch: i32) -> f32 {
    let mut sum: f32 = 0.0;
    let mut xy: f32;
    let mut yy: f32;
    let mut y: Vec<f32> = vec![0.0; N as usize];
    let mut signx: Vec<i32> = vec![0; N as usize];

    for j in 0..N as usize {
        signx[j] = (X[j] < 0.0) as i32;
        X[j] = X[j].abs();
        iy[j] = 0;
        y[j] = 0.0;
    }
    yy = 0.0;
    xy = 0.0;
    let mut pulsesLeft = K;
    if K > N >> 1 {
        for j in 0..N as usize {
            sum += X[j];
        }
        if !(sum > EPSILON && sum < 64.0) {
            X[0] = 1.0;
            for j in 1..N as usize {
                X[j] = 0.0;
            }
            sum = 1.0;
        }
        let rcp: f32 = (K as f32 + 0.8f32) * (1.0f32 / sum);
        for j in 0..N as usize {
            iy[j] = (rcp * X[j]).floor() as i32;
            y[j] = iy[j] as f32;
            yy += y[j] * y[j];
            xy += X[j] * y[j];
            y[j] *= 2.0;
            pulsesLeft -= iy[j];
        }
    }
    if pulsesLeft > N + 3 {
        let tmp: f32 = pulsesLeft as f32;
        yy += tmp * tmp;
        yy += tmp * y[0];
        iy[0] += pulsesLeft;
        pulsesLeft = 0;
    }
    for _i in 0..pulsesLeft {
        let mut best_id: i32 = 0;
        let mut best_num: f32;
        let mut best_den: f32;
        yy += 1.0;
        let Rxy = xy + X[0];
        let Ryy = yy + y[0];
        best_den = Ryy;
        best_num = Rxy * Rxy;
        for j in 1..N as usize {
            let Rxy = xy + X[j];
            let Ryy = yy + y[j];
            let Rxy2 = Rxy * Rxy;
            if best_den * Rxy2 > Ryy * best_num {
                best_den = Ryy;
                best_num = Rxy2;
                best_id = j as i32;
            }
        }
        xy += X[best_id as usize];
        yy += y[best_id as usize];
        y[best_id as usize] += 2.0;
        iy[best_id as usize] += 1;
    }
    for j in 0..N as usize {
        iy[j] = (iy[j] ^ -signx[j]) + signx[j];
    }
    yy
}

/// Upstream C: celt/vq.c:alg_quant
pub fn alg_quant(
    X: &mut [f32],
    N: i32,
    K: i32,
    spread: i32,
    B: i32,
    enc: &mut ec_enc,
    gain: f32,
    resynth: i32,
    arch: i32,
) -> u32 {
    assert!(K > 0);
    assert!(N > 1);
    let mut iy: Vec<i32> = vec![0; (N + 3) as usize];
    exp_rotation(X, N, 1, B, K, spread);
    let yy = op_pvq_search_c(X, &mut iy, K, N, arch);
    encode_pulses(&iy[..N as usize], K, enc);
    if resynth != 0 {
        normalise_residual(&iy, X, N, yy, gain);
        exp_rotation(X, N, -1, B, K, spread);
    }
    extract_collapse_mask(&iy, N, B)
}

/// Upstream C: celt/vq.c:alg_unquant
pub fn alg_unquant(
    X: &mut [f32],
    N: i32,
    K: i32,
    spread: i32,
    B: i32,
    dec: &mut ec_dec,
    gain: f32,
) -> u32 {
    assert!(K > 0);
    assert!(N > 1);
    let mut iy: Vec<i32> = vec![0; N as usize];
    let Ryy = decode_pulses(&mut iy[..N as usize], K, dec);
    normalise_residual(&iy, X, N, Ryy, gain);
    exp_rotation(X, N, -1, B, K, spread);
    extract_collapse_mask(&iy, N, B)
}

/// Upstream C: celt/vq.c:renormalise_vector
pub fn renormalise_vector(X: &mut [f32], N: i32, gain: f32, _arch: i32) {
    let E = EPSILON + celt_inner_prod(&X[..N as usize], &X[..N as usize], N as usize);
    let g = celt_rsqrt_norm(E) * gain;
    for i in 0..N as usize {
        X[i] = g * X[i];
    }
}

/// Upstream C: celt/vq.c:stereo_itheta
pub fn stereo_itheta(X: &[f32], Y: &[f32], stereo: i32, N: i32, _arch: i32) -> i32 {
    let mut Emid: f32 = EPSILON;
    let mut Eside: f32 = EPSILON;
    if stereo != 0 {
        for i in 0..N as usize {
            let m = X[i] + Y[i];
            let s = X[i] - Y[i];
            Emid += m * m;
            Eside += s * s;
        }
    } else {
        Emid += celt_inner_prod(&X[..N as usize], &X[..N as usize], N as usize);
        Eside += celt_inner_prod(&Y[..N as usize], &Y[..N as usize], N as usize);
    }
    let mid = celt_sqrt(Emid);
    let side = celt_sqrt(Eside);
    (0.5f32 + 16384.0 * std::f32::consts::FRAC_2_PI * fast_atan2f(side, mid)).floor() as i32
}
