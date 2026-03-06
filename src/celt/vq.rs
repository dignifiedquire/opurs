//! Vector quantization and PVQ rotation.
//!
//! Upstream C: `celt/vq.c`

use crate::arch::Arch;
use crate::celt::bands::SPREAD_NONE;
use crate::celt::cwrs::{decode_pulses, encode_pulses};
use crate::celt::entcode::celt_udiv;
use crate::celt::entdec::EcDec;
use crate::celt::entenc::EcEnc;
use crate::celt::mathops::{celt_atan2p_norm, celt_cos_norm, celt_rsqrt_norm, celt_sqrt};
use crate::celt::pitch::celt_inner_prod;

#[cfg(feature = "qext")]
use crate::celt::entcode::ec_tell;
#[cfg(feature = "qext")]
use crate::celt::entdec::{ec_dec_bit_logp, ec_dec_bits, ec_dec_uint};
#[cfg(feature = "qext")]
use crate::celt::entenc::{ec_enc_bit_logp, ec_enc_bits, ec_enc_uint};

const EPSILON: f32 = 1e-15f32;

#[cfg(feature = "qext")]
#[inline]
fn qext_trace_enabled_vq() -> bool {
    std::env::var_os("OPURS_QEXT_TRACE").is_some()
}

#[cfg(feature = "qext")]
#[inline]
fn qext_hash_i32(x: &[i32]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for &v in x {
        h ^= v as u32 as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

#[cfg(feature = "qext")]
#[inline]
fn qext_hash_f32(x: &[f32]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for &v in x {
        for b in v.to_ne_bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
    }
    h
}

/// Dispatch wrapper for `op_pvq_search`.
#[cfg(feature = "simd")]
#[inline]
fn op_pvq_search(x: &mut [f32], iy: &mut [i32], k: i32, n: i32, arch: Arch) -> f32 {
    super::simd::op_pvq_search(x, iy, k, n, arch)
}

/// Dispatch wrapper for `op_pvq_search` (scalar-only build).
#[cfg(not(feature = "simd"))]
#[inline]
fn op_pvq_search(x: &mut [f32], iy: &mut [i32], k: i32, n: i32, arch: Arch) -> f32 {
    op_pvq_search_c(x, iy, k, n, arch)
}

/// Upstream C: celt/vq.c:exp_rotation1
#[inline]
fn exp_rotation1(x: &mut [f32], len: i32, stride: i32, c: f32, s: f32) {
    let ms: f32 = -s;
    // Forward pass
    let fwd_end = len - stride;
    if fwd_end > 0 {
        for i in 0..fwd_end as usize {
            let x1 = x[i];
            let x2 = x[i + stride as usize];
            x[i + stride as usize] = c * x2 + s * x1;
            x[i] = c * x1 + ms * x2;
        }
    }
    // Backward pass
    let bwd_end = len - 2 * stride - 1;
    if bwd_end >= 0 {
        for i in (0..=bwd_end as usize).rev() {
            let x1 = x[i];
            let x2 = x[i + stride as usize];
            x[i + stride as usize] = c * x2 + s * x1;
            x[i] = c * x1 + ms * x2;
        }
    }
}

/// Upstream C: celt/vq.c:exp_rotation
#[inline]
pub fn exp_rotation(x: &mut [f32], mut len: i32, dir: i32, stride: i32, k: i32, spread: i32) {
    const SPREAD_FACTOR: [i32; 3] = [15, 10, 5];
    let mut stride2: i32 = 0;
    if 2 * k >= len || spread == SPREAD_NONE {
        return;
    }
    let factor = SPREAD_FACTOR[(spread - 1) as usize];
    let gain: f32 = 1.0f32 * len as f32 / (len + factor * k) as f32;
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
        let sub = &mut x[off..off + len as usize];
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
fn normalise_residual(iy: &[i32], x: &mut [f32], n: i32, ryy: f32, gain: f32) {
    let g = celt_rsqrt_norm(ryy) * gain;
    for i in 0..n as usize {
        x[i] = g * iy[i] as f32;
    }
}

/// Upstream C: celt/vq.c:extract_collapse_mask
fn extract_collapse_mask(iy: &[i32], n: i32, b: i32) -> u32 {
    if b <= 1 {
        return 1;
    }
    let n0 = celt_udiv(n as u32, b as u32) as i32;
    let mut collapse_mask: u32 = 0;
    for i in 0..b {
        let mut tmp: u32 = 0;
        for j in 0..n0 {
            tmp |= iy[(i * n0 + j) as usize] as u32;
        }
        collapse_mask |= ((tmp != 0) as u32) << i;
    }
    collapse_mask
}

/// Upstream C: celt/vq.c:op_pvq_search_c
pub fn op_pvq_search_c(x: &mut [f32], iy: &mut [i32], k: i32, n: i32, _arch: Arch) -> f32 {
    let mut sum: f32 = 0.0;
    let mut xy: f32;
    let mut yy: f32;
    let n = n as usize;
    // Max CELT band size is 176; use stack buffers.
    debug_assert!(n <= 176);
    let mut y = [0.0f32; 176];
    let mut signx = [0i32; 176];
    // Pre-slice to hoist bounds checks out of the hot loops.
    let x = &mut x[..n];
    let iy = &mut iy[..n];

    for j in 0..n {
        signx[j] = (x[j] < 0.0) as i32;
        x[j] = x[j].abs();
        iy[j] = 0;
        y[j] = 0.0;
    }
    yy = 0.0;
    xy = 0.0;
    let mut pulses_left = k;
    if k > (n >> 1) as i32 {
        for xj in x.iter() {
            sum += xj;
        }
        if !(sum > EPSILON && sum < 64.0) {
            x[0] = 1.0;
            for xj in x[1..].iter_mut() {
                *xj = 0.0;
            }
            sum = 1.0;
        }
        let rcp: f32 = (k as f32 + 0.8f32) * (1.0f32 / sum);
        for j in 0..n {
            iy[j] = (rcp * x[j]).floor() as i32;
            y[j] = iy[j] as f32;
            yy += y[j] * y[j];
            xy += x[j] * y[j];
            y[j] *= 2.0;
            pulses_left -= iy[j];
        }
    }
    if pulses_left > n as i32 + 3 {
        let tmp: f32 = pulses_left as f32;
        yy += tmp * tmp;
        yy += tmp * y[0];
        iy[0] += pulses_left;
        pulses_left = 0;
    }
    for _i in 0..pulses_left {
        let mut best_id: usize = 0;
        let mut best_num: f32;
        let mut best_den: f32;
        yy += 1.0;
        let rxy = xy + x[0];
        let ryy = yy + y[0];
        best_den = ryy;
        best_num = rxy * rxy;
        for j in 1..n {
            let rxy = xy + x[j];
            let ryy = yy + y[j];
            let rxy2 = rxy * rxy;
            if best_den * rxy2 > ryy * best_num {
                best_den = ryy;
                best_num = rxy2;
                best_id = j;
            }
        }
        xy += x[best_id];
        yy += y[best_id];
        y[best_id] += 2.0;
        iy[best_id] += 1;
    }
    for j in 0..n {
        iy[j] = (iy[j] ^ -signx[j]) + signx[j];
    }
    yy
}

// ---------------------------------------------------------------------------
// QEXT PVQ extensions: upsampled search, refinement, cubic quantization
// ---------------------------------------------------------------------------

/// Optimized PVQ search for n=2 with upsampling.
///
/// Upstream C: celt/vq.c:op_pvq_search_N2
#[cfg(feature = "qext")]
fn op_pvq_search_n2(
    x: &[f32],
    iy: &mut [i32],
    up_iy: &mut [i32],
    k: i32,
    up: i32,
    refine: &mut i32,
) -> f32 {
    let sum = x[0].abs() + x[1].abs();
    if sum < EPSILON {
        iy[0] = k;
        up_iy[0] = up * k;
        iy[1] = 0;
        up_iy[1] = 0;
        *refine = 0;
        return (k as f64 * k as f64 * up as f64 * up as f64) as f32;
    }
    let rcp_sum = 1.0f32 / sum;
    iy[0] = (0.5 + k as f32 * x[0] * rcp_sum).floor() as i32;
    up_iy[0] = (0.5 + up as f32 * k as f32 * x[0] * rcp_sum).floor() as i32;
    // Constrain up_iy within ±(up-1)/2 of up*iy
    up_iy[0] = (up * iy[0] - (up - 1) / 2).max((up * iy[0] + (up - 1) / 2).min(up_iy[0]));
    let offset = up_iy[0] - up * iy[0];
    iy[1] = k - iy[0].abs();
    up_iy[1] = up * k - up_iy[0].abs();
    if x[1] < 0.0 {
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
fn op_pvq_refine(
    xn: &[f32],
    iy: &mut [i32],
    iy0: &[i32],
    k: i32,
    up: i32,
    margin: i32,
    n: i32,
) -> bool {
    let same = std::ptr::eq(iy.as_ptr(), iy0.as_ptr());
    let mut rounding = vec![0.0f32; n as usize];
    let mut iysum: i32 = 0;

    for ((&x_n, iy_i), rounding_i) in xn.iter().zip(iy.iter_mut()).zip(rounding.iter_mut()) {
        let tmp = (k as f32 * 256.0) * x_n; // MULT32_32_Q31(SHL32(k,8), xn[i]) → k*256*xn in float
        *iy_i = (0.5 + tmp).floor() as i32;
        *rounding_i = tmp - (*iy_i as f32 * 128.0); // tmp - SHL32(iy[i], 7)
    }
    if !same {
        for (iy_i, &iy0_i) in iy.iter_mut().zip(iy0.iter()) {
            *iy_i = (up * iy0_i + up - 1).min((up * iy0_i - up + 1).max(*iy_i));
        }
    }
    iysum += iy.iter().sum::<i32>();
    if (iysum - k).abs() > 32 {
        return true; // failed
    }
    let dir: i32 = if iysum < k { 1 } else { -1 };
    while iysum != k {
        let mut roundval: f32 = -1000000.0 * dir as f32;
        let mut roundpos: usize = 0;
        for (i, &rounding_i) in rounding.iter().enumerate() {
            if (rounding_i - roundval) * dir as f32 > 0.0
                && (iy[i] - up * iy0[i]).abs() < (margin - 1)
                && !(dir == -1 && iy[i] == 0)
            {
                roundval = rounding_i;
                roundpos = i;
            }
        }
        iy[roundpos] += dir;
        rounding[roundpos] -= dir as f32 * 32768.0; // SHL32(dir, 15)
        iysum += dir;
    }
    false // success
}

/// General n-dimensional PVQ search with upsampling.
///
/// Upstream C: celt/vq.c:op_pvq_search_extra
#[cfg(feature = "qext")]
fn op_pvq_search_extra(
    x: &[f32],
    iy: &mut [i32],
    up_iy: &mut [i32],
    k: i32,
    up: i32,
    refine: &mut [i32],
    n: i32,
) -> f32 {
    let mut failed = false;
    let n_i32 = n;
    let n_usize = n_i32 as usize;

    let sum: f32 = x.iter().take(n_usize).map(|x| x.abs()).sum();
    let mut xn = vec![0.0f32; n_usize];
    if sum < EPSILON {
        failed = true;
    } else {
        let rcp_sum = 1.0f32 / sum;
        for (x_n, x_i) in xn.iter_mut().zip(x.iter()) {
            *x_n = x_i.abs() * rcp_sum;
        }
    }
    // First pass: refine base quantization
    let iy0_copy: Vec<i32> = iy.to_vec();
    failed = failed || op_pvq_refine(&xn, iy, &iy0_copy, k, 1, k + 1, n_i32);
    // Second pass: refine upsampled quantization constrained by base
    let iy_copy: Vec<i32> = iy.to_vec();
    failed = failed || op_pvq_refine(&xn, up_iy, &iy_copy, up * k, up, up, n_i32);
    if failed {
        iy[0] = k;
        iy[1..].fill(0);
        up_iy[0] = up * k;
        up_iy[1..].fill(0);
    }
    let mut yy: f64 = 0.0;
    for (((&x_i, iy_i), up_iy_i), refine_i) in x
        .iter()
        .zip(iy.iter_mut())
        .zip(up_iy.iter_mut())
        .zip(refine.iter_mut())
    {
        yy += *up_iy_i as f64 * *up_iy_i as f64;
        if x_i < 0.0 {
            *iy_i = -*iy_i;
            *up_iy_i = -*up_iy_i;
        }
        *refine_i = *up_iy_i - up * *iy_i;
    }
    yy as f32
}

/// Encode refinement value with adaptive bit allocation.
///
/// Upstream C: celt/vq.c:ec_enc_refine
#[cfg(feature = "qext")]
fn ec_enc_refine(enc: &mut EcEnc, refine: i32, up: i32, extra_bits: i32, use_entropy: bool) {
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
fn ec_dec_refine(dec: &mut EcDec, up: i32, extra_bits: i32, use_entropy: bool) -> i32 {
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
fn cubic_synthesis(x: &mut [f32], iy: &[i32], n: i32, k: i32, face: usize, sign: bool, gain: f32) {
    let n = n as usize;
    #[cfg(feature = "qext")]
    let trace = qext_trace_enabled_vq();
    for (x_i, &iy_i) in x.iter_mut().zip(iy.iter()).take(n) {
        *x_i = (1 + 2 * iy_i) as f32 - k as f32;
    }
    x[face] = if sign { -(k as f32) } else { k as f32 };
    let sum: f32 = x.iter().take(n).map(|x| x * x).sum();
    // Match upstream float path semantics: C computes `1.f/sqrt(sum)` with `sqrt`
    // operating in double precision before rounding back to float.
    let mag = (1.0f64 / (sum as f64).sqrt()) as f32;
    #[cfg(feature = "qext")]
    if trace {
        eprintln!(
            "[rust cubic] synth pre n={} k={} face={} sign={} sum={:.9} mag={:.9} iyh={:016x}",
            n,
            k,
            face,
            if sign { 1 } else { 0 },
            sum,
            mag,
            qext_hash_i32(&iy[..n]),
        );
    }
    for x_i in x.iter_mut().take(n) {
        *x_i *= mag * gain;
    }
    #[cfg(feature = "qext")]
    if trace {
        eprintln!(
            "[rust cubic] synth post n={} k={} xh={:016x} x0={:.9} x1={:.9} x2={:.9} x3={:.9}",
            n,
            k,
            qext_hash_f32(&x[..n]),
            x[0],
            if n > 1 { x[1] } else { 0.0 },
            if n > 2 { x[2] } else { 0.0 },
            if n > 3 { x[3] } else { 0.0 },
        );
    }
}

/// Encode cubic quantization for a band.
///
/// Upstream C: celt/vq.c:cubic_quant
#[cfg(feature = "qext")]
pub fn cubic_quant(
    x: &mut [f32],
    n: i32,
    res: i32,
    b: i32,
    enc: &mut EcEnc,
    gain: f32,
    resynth: i32,
) -> u32 {
    let n_i32 = n;
    let n_usize = n_i32 as usize;
    let mut k = 1 << res;
    // Using odd k on transients to avoid adding pre-echo
    if b != 1 {
        k = 1.max(k - 1);
    }
    if k == 1 {
        if resynth != 0 {
            x[..n_usize].fill(0.0);
        }
        return 0;
    }
    let mut face: usize = 0;
    let mut faceval: f32 = -1.0;
    for (i, &x_i) in x.iter().enumerate().take(n_usize) {
        if x_i.abs() > faceval {
            faceval = x_i.abs();
            face = i;
        }
    }
    let sign = x[face] < 0.0;
    ec_enc_uint(enc, face as u32, n_i32 as u32);
    ec_enc_bits(enc, sign as u32, 1);
    let norm = 0.5 * k as f32 / (faceval + EPSILON);
    let mut iy = vec![0i32; n_usize];
    for (iy_i, &x_i) in iy.iter_mut().zip(x.iter()).take(n_usize) {
        *iy_i = (k - 1).min(((x_i + faceval) * norm).floor() as i32);
    }
    for (i, &iy_i) in iy.iter().enumerate().take(n_usize) {
        if i != face {
            ec_enc_bits(enc, iy_i as u32, res as u32);
        }
    }
    if resynth != 0 {
        cubic_synthesis(x, &iy, n_i32, k, face, sign, gain);
    }
    (1u32 << b) - 1
}

/// Decode cubic quantization for a band.
///
/// Upstream C: celt/vq.c:cubic_unquant
#[cfg(feature = "qext")]
pub fn cubic_unquant(x: &mut [f32], n: i32, res: i32, b: i32, dec: &mut EcDec, gain: f32) -> u32 {
    let n_i32 = n;
    let n_usize = n_i32 as usize;
    let mut k = 1 << res;
    #[cfg(feature = "qext")]
    let trace = qext_trace_enabled_vq();
    if b != 1 {
        k = 1.max(k - 1);
    }
    if k == 1 {
        x[..n_usize].fill(0.0);
        return 0;
    }
    let face = ec_dec_uint(dec, n_i32 as u32) as usize;
    let sign = ec_dec_bits(dec, 1) != 0;
    let mut iy = vec![0i32; n_usize];
    for (i, iy_i) in iy.iter_mut().enumerate().take(n_usize) {
        if i != face {
            *iy_i = ec_dec_bits(dec, res as u32) as i32;
        }
    }
    iy[face] = 0;
    #[cfg(feature = "qext")]
    if trace {
        eprintln!(
            "[rust cubic] unq pre n={} res={} b={} k={} tell={} face={} sign={} iyh={:016x}",
            n,
            res,
            b,
            k,
            ec_tell(dec),
            face,
            if sign { 1 } else { 0 },
            qext_hash_i32(&iy[..n_usize]),
        );
    }
    cubic_synthesis(x, &iy, n_i32, k, face, sign, gain);
    #[cfg(feature = "qext")]
    if trace {
        eprintln!(
            "[rust cubic] unq post n={} res={} b={} k={} tell={} xh={:016x}",
            n,
            res,
            b,
            k,
            ec_tell(dec),
            qext_hash_f32(&x[..n_usize]),
        );
    }
    (1u32 << b) - 1
}

/// Upstream C: celt/vq.c:alg_quant
#[allow(clippy::too_many_arguments)]
pub fn alg_quant(
    x: &mut [f32],
    n: i32,
    k: i32,
    spread: i32,
    b: i32,
    enc: &mut EcEnc,
    gain: f32,
    resynth: i32,
    arch: Arch,
    #[cfg(feature = "qext")] ext_enc: &mut EcEnc,
    #[cfg(feature = "qext")] extra_bits: i32,
) -> u32 {
    debug_assert!(k > 0);
    debug_assert!(n > 1);
    // Max CELT band size is 176, n+3 <= 179; use stack buffer.
    debug_assert!((n as usize + 3) <= 180);
    let mut iy = [0i32; 180];
    exp_rotation(x, n, 1, b, k, spread);

    #[cfg(feature = "qext")]
    let collapse_mask;
    #[cfg(not(feature = "qext"))]
    let collapse_mask;

    #[cfg(feature = "qext")]
    {
        if n == 2 && extra_bits >= 2 {
            let mut up_iy = [0i32; 2];
            let mut refine = 0i32;
            let up = (1 << extra_bits) - 1;
            let yy = op_pvq_search_n2(x, &mut iy, &mut up_iy, k, up, &mut refine);
            collapse_mask = extract_collapse_mask(&up_iy, n, b);
            encode_pulses(&iy[..n as usize], k, enc);
            ec_enc_uint(ext_enc, (refine + (up - 1) / 2) as u32, up as u32);
            if resynth != 0 {
                normalise_residual(&up_iy, x, n, yy, gain);
                exp_rotation(x, n, -1, b, k, spread);
            }
        } else if extra_bits >= 2 {
            let n_usize = n as usize;
            let mut up_iy = vec![0i32; n_usize];
            let mut refine = vec![0i32; n_usize];
            let up = (1 << extra_bits) - 1;
            let yy = op_pvq_search_extra(x, &mut iy, &mut up_iy, k, up, &mut refine, n);
            collapse_mask = extract_collapse_mask(&up_iy, n, b);
            encode_pulses(&iy[..n as usize], k, enc);
            let use_entropy =
                (ext_enc.storage as i32 * 8 - ec_tell(ext_enc)) > (n - 1) * (extra_bits + 3) + 1;
            for &refine_i in refine.iter().take(n_usize - 1) {
                ec_enc_refine(ext_enc, refine_i, up, extra_bits, use_entropy);
            }
            if iy[n_usize - 1] == 0 {
                ec_enc_bits(ext_enc, (up_iy[n_usize - 1] < 0) as u32, 1);
            }
            if resynth != 0 {
                normalise_residual(&up_iy, x, n, yy, gain);
                exp_rotation(x, n, -1, b, k, spread);
            }
        } else {
            let yy = op_pvq_search(x, &mut iy, k, n, arch);
            collapse_mask = extract_collapse_mask(&iy, n, b);
            encode_pulses(&iy[..n as usize], k, enc);
            if resynth != 0 {
                normalise_residual(&iy, x, n, yy, gain);
                exp_rotation(x, n, -1, b, k, spread);
            }
        }
    }

    #[cfg(not(feature = "qext"))]
    {
        let yy = op_pvq_search(x, &mut iy, k, n, arch);
        collapse_mask = extract_collapse_mask(&iy, n, b);
        encode_pulses(&iy[..n as usize], k, enc);
        if resynth != 0 {
            normalise_residual(&iy, x, n, yy, gain);
            exp_rotation(x, n, -1, b, k, spread);
        }
    }

    collapse_mask
}

/// Upstream C: celt/vq.c:alg_unquant
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn alg_unquant(
    x: &mut [f32],
    n: i32,
    k: i32,
    spread: i32,
    b: i32,
    dec: &mut EcDec,
    gain: f32,
    #[cfg(feature = "qext")] ext_dec: &mut EcDec,
    #[cfg(feature = "qext")] extra_bits: i32,
) -> u32 {
    debug_assert!(k > 0);
    debug_assert!(n > 1);
    let mut iy = [0i32; 176];
    #[cfg(feature = "qext")]
    let mut ryy = decode_pulses(&mut iy[..n as usize], k, dec);
    #[cfg(not(feature = "qext"))]
    let ryy = decode_pulses(&mut iy[..n as usize], k, dec);
    #[cfg(feature = "qext")]
    let mut yy_shift: i32 = 0;

    #[cfg(feature = "qext")]
    {
        if n == 2 && extra_bits >= 2 {
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
            ryy = iy[0] as f32 * iy[0] as f32 + iy[1] as f32 * iy[1] as f32;
        } else if extra_bits >= 2 {
            let n_usize = n as usize;
            yy_shift = 0.max(extra_bits - 7);
            let up = (1 << extra_bits) - 1;
            let use_entropy =
                (ext_dec.storage as i32 * 8 - ec_tell(ext_dec)) > (n - 1) * (extra_bits + 3) + 1;
            let mut refine = vec![0i32; n_usize];
            for refine_i in refine.iter_mut().take(n_usize - 1) {
                *refine_i = ec_dec_refine(ext_dec, up, extra_bits, use_entropy);
            }
            let sign = if iy[n_usize - 1] == 0 {
                ec_dec_bits(ext_dec, 1) != 0
            } else {
                iy[n_usize - 1] < 0
            };
            for (iy_i, &refine_i) in iy.iter_mut().zip(refine.iter()).take(n_usize - 1) {
                *iy_i = *iy_i * up + refine_i;
            }
            iy[n_usize - 1] = up * k;
            let tail_abs_sum: i32 = iy.iter().take(n_usize - 1).map(|v| v.abs()).sum();
            iy[n_usize - 1] -= tail_abs_sum;
            if sign {
                iy[n_usize - 1] = -iy[n_usize - 1];
            }
            let mut yy64: f32 = 0.0;
            for iy_i in iy.iter().take(n_usize) {
                yy64 += *iy_i as f32 * *iy_i as f32;
            }
            ryy = yy64;
        }
    }

    #[cfg(feature = "qext")]
    let vq_trace = qext_trace_enabled_vq();
    #[cfg(feature = "qext")]
    if vq_trace {
        eprintln!(
            "[rust vq] pre n={} k={} b={} extra={} tell={} iyh={:016x} ryy={:.8} gain={:.9} iy0={} iy1={} iy2={} iy3={}",
            n,
            k,
            b,
            extra_bits,
            ec_tell(dec),
            qext_hash_i32(&iy[..n as usize]),
            ryy,
            gain,
            iy[0],
            if n > 1 { iy[1] } else { 0 },
            if n > 2 { iy[2] } else { 0 },
            if n > 3 { iy[3] } else { 0 }
        );
    }

    #[cfg(feature = "qext")]
    let _ = yy_shift; // used by fixed-point only
    normalise_residual(&iy, x, n, ryy, gain);
    #[cfg(feature = "qext")]
    if vq_trace {
        eprintln!(
            "[rust vq] norm n={} k={} b={} extra={} xh={:016x} x0={:.9} x1={:.9} x2={:.9} x3={:.9} b0={:08x} b1={:08x} b2={:08x} b3={:08x}",
            n,
            k,
            b,
            extra_bits,
            qext_hash_f32(&x[..n as usize]),
            x[0],
            if n > 1 { x[1] } else { 0.0 },
            if n > 2 { x[2] } else { 0.0 },
            if n > 3 { x[3] } else { 0.0 },
            x[0].to_bits(),
            if n > 1 { x[1].to_bits() } else { 0 },
            if n > 2 { x[2].to_bits() } else { 0 },
            if n > 3 { x[3].to_bits() } else { 0 }
        );
    }
    exp_rotation(x, n, -1, b, k, spread);
    let cm = extract_collapse_mask(&iy, n, b);
    #[cfg(feature = "qext")]
    if vq_trace {
        eprintln!(
            "[rust vq] post n={} k={} b={} extra={} xh={:016x} cm={}",
            n,
            k,
            b,
            extra_bits,
            qext_hash_f32(&x[..n as usize]),
            cm
        );
    }
    cm
}

/// Upstream C: celt/vq.c:renormalise_vector
#[inline]
pub fn renormalise_vector(x: &mut [f32], n: i32, gain: f32, _arch: Arch) {
    let energy = EPSILON + celt_inner_prod(&x[..n as usize], &x[..n as usize], n as usize, _arch);
    let g = celt_rsqrt_norm(energy) * gain;
    for xi in x[..n as usize].iter_mut() {
        *xi *= g;
    }
}

///
/// Returns Q30 value in range [0, 1073741824] (= 2^30).
/// Callers that need Q14 should right-shift by 16.
/// Upstream C: celt/vq.c:stereo_itheta
#[inline(never)]
pub fn stereo_itheta(x: &[f32], y: &[f32], stereo: i32, n: i32, _arch: Arch) -> i32 {
    let mut emid: f32 = 0.0;
    let mut eside: f32 = 0.0;
    if stereo != 0 {
        for i in 0..n as usize {
            let m = x[i] + y[i];
            let s = x[i] - y[i];
            emid += m * m;
            eside += s * s;
        }
    } else {
        emid += celt_inner_prod(&x[..n as usize], &x[..n as usize], n as usize, _arch);
        eside += celt_inner_prod(&y[..n as usize], &y[..n as usize], n as usize, _arch);
    }
    let mid = celt_sqrt(emid);
    let side = celt_sqrt(eside);
    (0.5f32 + 65536.0 * 16384.0 * celt_atan2p_norm(side, mid)).floor() as i32
}
