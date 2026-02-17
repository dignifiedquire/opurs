//! Vector quantization and PVQ rotation.
//!
//! Upstream C: `celt/vq.c`

use crate::celt::bands::SPREAD_NONE;
use crate::celt::cwrs::{decode_pulses, encode_pulses};
use crate::celt::entcode::celt_udiv;
use crate::celt::entdec::ec_dec;
use crate::celt::entenc::ec_enc;
use crate::celt::mathops::{celt_atan2p_norm, celt_cos_norm, celt_rsqrt_norm, celt_sqrt};
use crate::celt::pitch::celt_inner_prod;

#[cfg(feature = "qext")]
use crate::celt::entcode::ec_tell;
#[cfg(feature = "qext")]
use crate::celt::entdec::{ec_dec_bit_logp, ec_dec_bits, ec_dec_uint};
#[cfg(feature = "qext")]
use crate::celt::entenc::{ec_enc_bit_logp, ec_enc_bits, ec_enc_uint};

const EPSILON: f32 = 1e-15f32;

/// Dispatch wrapper for `op_pvq_search`.
#[cfg(feature = "simd")]
#[inline]
fn op_pvq_search(X: &mut [f32], iy: &mut [i32], K: i32, N: i32, arch: i32) -> f32 {
    super::simd::op_pvq_search(X, iy, K, N, arch)
}

/// Dispatch wrapper for `op_pvq_search` (scalar-only build).
#[cfg(not(feature = "simd"))]
#[inline]
fn op_pvq_search(X: &mut [f32], iy: &mut [i32], K: i32, N: i32, arch: i32) -> f32 {
    op_pvq_search_c(X, iy, K, N, arch)
}

/// Upstream C: celt/vq.c:exp_rotation1
#[inline]
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
#[inline]
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
#[inline]
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
        for xj in X[..N as usize].iter() {
            sum += xj;
        }
        if !(sum > EPSILON && sum < 64.0) {
            X[0] = 1.0;
            for xj in X[1..N as usize].iter_mut() {
                *xj = 0.0;
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

// ---------------------------------------------------------------------------
// QEXT PVQ extensions: upsampled search, refinement, cubic quantization
// ---------------------------------------------------------------------------

/// Optimized PVQ search for N=2 with upsampling.
///
/// Upstream C: celt/vq.c:op_pvq_search_N2
#[cfg(feature = "qext")]
fn op_pvq_search_N2(
    X: &[f32],
    iy: &mut [i32],
    up_iy: &mut [i32],
    K: i32,
    up: i32,
    refine: &mut i32,
) -> f32 {
    let sum = X[0].abs() + X[1].abs();
    if sum < EPSILON {
        iy[0] = K;
        up_iy[0] = up * K;
        iy[1] = 0;
        up_iy[1] = 0;
        *refine = 0;
        return (K as f64 * K as f64 * up as f64 * up as f64) as f32;
    }
    let rcp_sum = 1.0f32 / sum;
    iy[0] = (0.5 + K as f32 * X[0] * rcp_sum).floor() as i32;
    up_iy[0] = (0.5 + up as f32 * K as f32 * X[0] * rcp_sum).floor() as i32;
    // Constrain up_iy within ±(up-1)/2 of up*iy
    up_iy[0] = (up * iy[0] - (up - 1) / 2).max((up * iy[0] + (up - 1) / 2).min(up_iy[0]));
    let offset = up_iy[0] - up * iy[0];
    iy[1] = K - iy[0].abs();
    up_iy[1] = up * K - up_iy[0].abs();
    if X[1] < 0.0 {
        iy[1] = -iy[1];
        up_iy[1] = -up_iy[1];
        *refine = -offset;
    } else {
        *refine = offset;
    }
    (up_iy[0] as f64 * up_iy[0] as f64 + up_iy[1] as f64 * up_iy[1] as f64) as f32
}

/// Refine PVQ quantization by adjusting integer pulse counts.
///
/// Upstream C: celt/vq.c:op_pvq_refine
#[cfg(feature = "qext")]
#[allow(clippy::needless_range_loop)]
fn op_pvq_refine(
    Xn: &[f32],
    iy: &mut [i32],
    iy0: &[i32],
    K: i32,
    up: i32,
    margin: i32,
    N: i32,
) -> bool {
    let same = std::ptr::eq(iy.as_ptr(), iy0.as_ptr());
    let mut rounding = vec![0.0f32; N as usize];
    let mut iysum: i32 = 0;

    for i in 0..N as usize {
        let tmp = (K as f32 * 256.0) * Xn[i]; // MULT32_32_Q31(SHL32(K,8), Xn[i]) → K*256*Xn in float
        iy[i] = (0.5 + tmp).floor() as i32;
        rounding[i] = tmp - (iy[i] as f32 * 128.0); // tmp - SHL32(iy[i], 7)
    }
    if !same {
        for i in 0..N as usize {
            iy[i] = (up * iy0[i] + up - 1).min((up * iy0[i] - up + 1).max(iy[i]));
        }
    }
    for i in 0..N as usize {
        iysum += iy[i];
    }
    if (iysum - K).abs() > 32 {
        return true; // failed
    }
    let dir: i32 = if iysum < K { 1 } else { -1 };
    while iysum != K {
        let mut roundval: f32 = -1000000.0 * dir as f32;
        let mut roundpos: usize = 0;
        for i in 0..N as usize {
            if (rounding[i] - roundval) * dir as f32 > 0.0
                && (iy[i] - up * iy0[i]).abs() < (margin - 1)
                && !(dir == -1 && iy[i] == 0)
            {
                roundval = rounding[i];
                roundpos = i;
            }
        }
        iy[roundpos] += dir;
        rounding[roundpos] -= dir as f32 * 32768.0; // SHL32(dir, 15)
        iysum += dir;
    }
    false // success
}

/// General N-dimensional PVQ search with upsampling.
///
/// Upstream C: celt/vq.c:op_pvq_search_extra
#[cfg(feature = "qext")]
#[allow(clippy::needless_range_loop)]
fn op_pvq_search_extra(
    X: &[f32],
    iy: &mut [i32],
    up_iy: &mut [i32],
    K: i32,
    up: i32,
    refine: &mut [i32],
    N: i32,
) -> f32 {
    let mut sum: f32 = 0.0;
    let mut failed = false;
    let n = N as usize;

    for i in 0..n {
        sum += X[i].abs();
    }
    let mut Xn = vec![0.0f32; n];
    if sum < EPSILON {
        failed = true;
    } else {
        let rcp_sum = 1.0f32 / sum;
        for i in 0..n {
            Xn[i] = X[i].abs() * rcp_sum;
        }
    }
    // First pass: refine base quantization
    let iy0_copy: Vec<i32> = iy.to_vec();
    failed = failed || op_pvq_refine(&Xn, iy, &iy0_copy, K, 1, K + 1, N);
    // Second pass: refine upsampled quantization constrained by base
    let iy_copy: Vec<i32> = iy.to_vec();
    failed = failed || op_pvq_refine(&Xn, up_iy, &iy_copy, up * K, up, up, N);
    if failed {
        iy[0] = K;
        for i in 1..n {
            iy[i] = 0;
        }
        up_iy[0] = up * K;
        for i in 1..n {
            up_iy[i] = 0;
        }
    }
    let mut yy: f64 = 0.0;
    for i in 0..n {
        yy += up_iy[i] as f64 * up_iy[i] as f64;
        if X[i] < 0.0 {
            iy[i] = -iy[i];
            up_iy[i] = -up_iy[i];
        }
        refine[i] = up_iy[i] - up * iy[i];
    }
    yy as f32
}

/// Encode refinement value with adaptive bit allocation.
///
/// Upstream C: celt/vq.c:ec_enc_refine
#[cfg(feature = "qext")]
fn ec_enc_refine(enc: &mut ec_enc, refine: i32, up: i32, extra_bits: i32, use_entropy: bool) {
    let large = refine.abs() > up / 2;
    ec_enc_bit_logp(enc, large as i32, if use_entropy { 3 } else { 1 });
    if large {
        ec_enc_bits(enc, (refine < 0) as u32, 1);
        ec_enc_bits(
            enc,
            (refine.abs() - up / 2 - 1) as u32,
            extra_bits as u32 - 1,
        );
    } else {
        ec_enc_bits(enc, (refine + up / 2) as u32, extra_bits as u32);
    }
}

/// Decode refinement value with adaptive bit allocation.
///
/// Upstream C: celt/vq.c:ec_dec_refine
#[cfg(feature = "qext")]
fn ec_dec_refine(dec: &mut ec_dec, up: i32, extra_bits: i32, use_entropy: bool) -> i32 {
    let large = ec_dec_bit_logp(dec, if use_entropy { 3 } else { 1 });
    if large != 0 {
        let sign = ec_dec_bits(dec, 1);
        let mut refine = ec_dec_bits(dec, extra_bits as u32 - 1) as i32 + up / 2 + 1;
        if sign != 0 {
            refine = -refine;
        }
        refine
    } else {
        ec_dec_bits(dec, extra_bits as u32) as i32 - up / 2
    }
}

/// Reconstruct signal from cubic quantization.
///
/// Upstream C: celt/vq.c:cubic_synthesis
#[cfg(feature = "qext")]
#[allow(clippy::needless_range_loop)]
fn cubic_synthesis(X: &mut [f32], iy: &[i32], N: i32, K: i32, face: usize, sign: bool, gain: f32) {
    let n = N as usize;
    let mut sum: f32 = 0.0;
    for i in 0..n {
        X[i] = (1 + 2 * iy[i]) as f32 - K as f32;
    }
    X[face] = if sign { -(K as f32) } else { K as f32 };
    for i in 0..n {
        sum += X[i] * X[i];
    }
    let mag = 1.0f32 / sum.sqrt();
    for i in 0..n {
        X[i] *= mag * gain;
    }
}

/// Encode cubic quantization for a band.
///
/// Upstream C: celt/vq.c:cubic_quant
#[cfg(feature = "qext")]
#[allow(clippy::needless_range_loop)]
pub fn cubic_quant(
    X: &mut [f32],
    N: i32,
    res: i32,
    B: i32,
    enc: &mut ec_enc,
    gain: f32,
    resynth: i32,
) -> u32 {
    let n = N as usize;
    let mut K = 1 << res;
    // Using odd K on transients to avoid adding pre-echo
    if B != 1 {
        K = 1.max(K - 1);
    }
    if K == 1 {
        if resynth != 0 {
            X[..n].fill(0.0);
        }
        return 0;
    }
    let mut face: usize = 0;
    let mut faceval: f32 = -1.0;
    for i in 0..n {
        if X[i].abs() > faceval {
            faceval = X[i].abs();
            face = i;
        }
    }
    let sign = X[face] < 0.0;
    ec_enc_uint(enc, face as u32, N as u32);
    ec_enc_bits(enc, sign as u32, 1);
    let norm = 0.5 * K as f32 / (faceval + EPSILON);
    let mut iy = vec![0i32; n];
    for i in 0..n {
        iy[i] = (K - 1).min(((X[i] + faceval) * norm).floor() as i32);
    }
    for i in 0..n {
        if i != face {
            ec_enc_bits(enc, iy[i] as u32, res as u32);
        }
    }
    if resynth != 0 {
        cubic_synthesis(X, &iy, N, K, face, sign, gain);
    }
    (1u32 << B) - 1
}

/// Decode cubic quantization for a band.
///
/// Upstream C: celt/vq.c:cubic_unquant
#[cfg(feature = "qext")]
#[allow(clippy::needless_range_loop)]
pub fn cubic_unquant(X: &mut [f32], N: i32, res: i32, B: i32, dec: &mut ec_dec, gain: f32) -> u32 {
    let n = N as usize;
    let mut K = 1 << res;
    if B != 1 {
        K = 1.max(K - 1);
    }
    if K == 1 {
        X[..n].fill(0.0);
        return 0;
    }
    let face = ec_dec_uint(dec, N as u32) as usize;
    let sign = ec_dec_bits(dec, 1) != 0;
    let mut iy = vec![0i32; n];
    for i in 0..n {
        if i != face {
            iy[i] = ec_dec_bits(dec, res as u32) as i32;
        }
    }
    iy[face] = 0;
    cubic_synthesis(X, &iy, N, K, face, sign, gain);
    (1u32 << B) - 1
}

/// Upstream C: celt/vq.c:alg_quant
#[allow(clippy::needless_range_loop)]
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
    #[cfg(feature = "qext")] ext_enc: &mut ec_enc,
    #[cfg(feature = "qext")] extra_bits: i32,
) -> u32 {
    assert!(K > 0);
    assert!(N > 1);
    let mut iy: Vec<i32> = vec![0; (N + 3) as usize];
    exp_rotation(X, N, 1, B, K, spread);

    #[cfg(feature = "qext")]
    let collapse_mask;
    #[cfg(not(feature = "qext"))]
    let collapse_mask;

    #[cfg(feature = "qext")]
    {
        if N == 2 && extra_bits >= 2 {
            let mut up_iy = [0i32; 2];
            let mut refine = 0i32;
            let up = (1 << extra_bits) - 1;
            let yy = op_pvq_search_N2(X, &mut iy, &mut up_iy, K, up, &mut refine);
            collapse_mask = extract_collapse_mask(&up_iy, N, B);
            encode_pulses(&iy[..N as usize], K, enc);
            ec_enc_uint(ext_enc, (refine + (up - 1) / 2) as u32, up as u32);
            if resynth != 0 {
                normalise_residual(&up_iy, X, N, yy, gain);
                exp_rotation(X, N, -1, B, K, spread);
            }
        } else if extra_bits >= 2 {
            let n = N as usize;
            let mut up_iy = vec![0i32; n];
            let mut refine = vec![0i32; n];
            let up = (1 << extra_bits) - 1;
            let yy = op_pvq_search_extra(X, &mut iy, &mut up_iy, K, up, &mut refine, N);
            collapse_mask = extract_collapse_mask(&up_iy, N, B);
            encode_pulses(&iy[..N as usize], K, enc);
            let use_entropy =
                (ext_enc.storage as i32 * 8 - ec_tell(ext_enc)) > (N - 1) * (extra_bits + 3) + 1;
            for i in 0..(N - 1) as usize {
                ec_enc_refine(ext_enc, refine[i], up, extra_bits, use_entropy);
            }
            if iy[(N - 1) as usize] == 0 {
                ec_enc_bits(ext_enc, (up_iy[(N - 1) as usize] < 0) as u32, 1);
            }
            if resynth != 0 {
                normalise_residual(&up_iy, X, N, yy, gain);
                exp_rotation(X, N, -1, B, K, spread);
            }
        } else {
            let yy = op_pvq_search(X, &mut iy, K, N, arch);
            collapse_mask = extract_collapse_mask(&iy, N, B);
            encode_pulses(&iy[..N as usize], K, enc);
            if resynth != 0 {
                normalise_residual(&iy, X, N, yy, gain);
                exp_rotation(X, N, -1, B, K, spread);
            }
        }
    }

    #[cfg(not(feature = "qext"))]
    {
        let yy = op_pvq_search(X, &mut iy, K, N, arch);
        collapse_mask = extract_collapse_mask(&iy, N, B);
        encode_pulses(&iy[..N as usize], K, enc);
        if resynth != 0 {
            normalise_residual(&iy, X, N, yy, gain);
            exp_rotation(X, N, -1, B, K, spread);
        }
    }

    collapse_mask
}

/// Upstream C: celt/vq.c:alg_unquant
#[inline]
#[allow(clippy::needless_range_loop)]
pub fn alg_unquant(
    X: &mut [f32],
    N: i32,
    K: i32,
    spread: i32,
    B: i32,
    dec: &mut ec_dec,
    gain: f32,
    #[cfg(feature = "qext")] ext_dec: &mut ec_dec,
    #[cfg(feature = "qext")] extra_bits: i32,
) -> u32 {
    assert!(K > 0);
    assert!(N > 1);
    let mut iy = [0i32; 176];
    #[allow(unused_mut)]
    let mut Ryy = decode_pulses(&mut iy[..N as usize], K, dec);
    #[allow(unused_assignments, unused_mut)]
    let mut yy_shift: i32 = 0;

    #[cfg(feature = "qext")]
    {
        if N == 2 && extra_bits >= 2 {
            yy_shift = 0.max(extra_bits - 7);
            let up = (1 << extra_bits) - 1;
            let refine = ec_dec_uint(ext_dec, up as u32) as i32 - (up - 1) / 2;
            iy[0] *= up;
            iy[1] *= up;
            if iy[1] == 0 {
                iy[1] = if iy[0] > 0 { -refine } else { refine };
                iy[0] += if refine as i64 * iy[0] as i64 > 0 {
                    -refine
                } else {
                    refine
                };
            } else if iy[1] > 0 {
                iy[0] += refine;
                iy[1] -= refine * if iy[0] > 0 { 1 } else { -1 };
            } else {
                iy[0] -= refine;
                iy[1] -= refine * if iy[0] > 0 { 1 } else { -1 };
            }
            Ryy = (iy[0] as f64 * iy[0] as f64 + iy[1] as f64 * iy[1] as f64) as f32;
        } else if extra_bits >= 2 {
            let n = N as usize;
            yy_shift = 0.max(extra_bits - 7);
            let up = (1 << extra_bits) - 1;
            let use_entropy =
                (ext_dec.storage as i32 * 8 - ec_tell(ext_dec)) > (N - 1) * (extra_bits + 3) + 1;
            let mut refine = vec![0i32; n];
            for i in 0..(N - 1) as usize {
                refine[i] = ec_dec_refine(ext_dec, up, extra_bits, use_entropy);
            }
            let sign = if iy[(N - 1) as usize] == 0 {
                ec_dec_bits(ext_dec, 1) != 0
            } else {
                iy[(N - 1) as usize] < 0
            };
            for i in 0..(N - 1) as usize {
                iy[i] = iy[i] * up + refine[i];
            }
            iy[(N - 1) as usize] = up * K;
            for i in 0..(N - 1) as usize {
                iy[(N - 1) as usize] -= iy[i].abs();
            }
            if sign {
                iy[(N - 1) as usize] = -iy[(N - 1) as usize];
            }
            let mut yy64: f64 = 0.0;
            for i in 0..n {
                yy64 += iy[i] as f64 * iy[i] as f64;
            }
            Ryy = yy64 as f32;
        }
    }

    let _ = yy_shift; // used by fixed-point only
    normalise_residual(&iy, X, N, Ryy, gain);
    exp_rotation(X, N, -1, B, K, spread);
    extract_collapse_mask(&iy, N, B)
}

/// Upstream C: celt/vq.c:renormalise_vector
#[inline]
pub fn renormalise_vector(X: &mut [f32], N: i32, gain: f32, _arch: i32) {
    let E = EPSILON + celt_inner_prod(&X[..N as usize], &X[..N as usize], N as usize);
    let g = celt_rsqrt_norm(E) * gain;
    for xi in X[..N as usize].iter_mut() {
        *xi *= g;
    }
}

/// Upstream C: celt/vq.c:stereo_itheta
///
/// Returns Q30 value in range [0, 1073741824] (= 2^30).
/// Callers that need Q14 should right-shift by 16.
#[inline]
pub fn stereo_itheta(X: &[f32], Y: &[f32], stereo: i32, N: i32, _arch: i32) -> i32 {
    let mut Emid: f32 = 0.0;
    let mut Eside: f32 = 0.0;
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
    (0.5f32 + 65536.0 * 16384.0 * celt_atan2p_norm(side, mid)).floor() as i32
}
