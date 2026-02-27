//! Band energy computation, normalization, and quantization.
//!
//! Upstream C: `celt/bands.c`

use crate::arch::Arch;
use crate::celt::entcode::{celt_sudiv, celt_udiv, ec_ctx, ec_tell_frac, BITRES};
use crate::celt::entdec::{ec_dec_bit_logp, ec_dec_bits, ec_dec_uint, ec_dec_update, ec_decode};
use crate::celt::entenc::{ec_enc_bit_logp, ec_enc_bits, ec_enc_uint, ec_encode};
#[cfg(feature = "qext")]
use crate::celt::mathops::celt_cos_norm2;
use crate::celt::mathops::{celt_exp2, celt_rsqrt, celt_rsqrt_norm, celt_sqrt, isqrt32};
use crate::celt::modes::OpusCustomMode;
use crate::celt::pitch::{celt_inner_prod, dual_inner_prod};
use crate::celt::quant_bands::eMeans;
use crate::celt::rate::{
    bits2pulses, get_pulses, pulses2bits, QTHETA_OFFSET, QTHETA_OFFSET_TWOPHASE,
};
use crate::celt::vq::{alg_quant, alg_unquant, renormalise_vector, stereo_itheta};
#[cfg(feature = "qext")]
use crate::celt::vq::{cubic_quant, cubic_unquant};
use crate::silk::macros::EC_CLZ0;

const EPSILON: f32 = 1e-15f32;
const Q15ONE: f32 = 1.0f32;
const NORM_SCALING: f32 = 1.0f32;
#[cfg(feature = "qext")]
#[inline]
fn qext_hash_band(x: &[f32]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for &v in x {
        for b in v.to_ne_bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
    }
    h
}

pub const SPREAD_NONE: i32 = 0;
pub const SPREAD_LIGHT: i32 = 1;
pub const SPREAD_NORMAL: i32 = 2;
pub const SPREAD_AGGRESSIVE: i32 = 3;

/// Band encoding/decoding context.
///
/// `ec` is passed separately so the struct can remain `Copy + Clone`
/// for theta-rdo save/restore.
///
/// Upstream C: celt/bands.c:struct band_ctx
#[derive(Copy, Clone)]
struct band_ctx<'a> {
    encode: i32,
    resynth: i32,
    m: &'a OpusCustomMode,
    i: i32,
    intensity: i32,
    spread: i32,
    tf_change: i32,
    remaining_bits: i32,
    bandE: &'a [f32],
    seed: u32,
    arch: Arch,
    theta_round: i32,
    disable_inv: i32,
    avoid_split_noise: i32,
    #[cfg(feature = "qext")]
    ext_ec: *mut ec_ctx<'a>,
    #[cfg(feature = "qext")]
    extra_bits: i32,
    #[cfg(feature = "qext")]
    ext_total_bits: i32,
    #[cfg(feature = "qext")]
    extra_bands: bool,
    #[cfg(feature = "qext")]
    ext_b: i32,
    #[cfg(feature = "qext")]
    cap: *const i32,
    #[cfg(feature = "qext")]
    #[allow(dead_code)]
    cap_len: i32,
}

/// Upstream C: celt/bands.c:struct split_ctx
#[derive(Copy, Clone)]
struct split_ctx {
    inv: i32,
    imid: i32,
    iside: i32,
    delta: i32,
    itheta: i32,
    qalloc: i32,
    #[cfg(feature = "qext")]
    itheta_q30: i32,
}

/// Upstream C: celt/bands.c:hysteresis_decision
pub fn hysteresis_decision(
    val: f32,
    thresholds: &[f32],
    hysteresis: &[f32],
    N: i32,
    prev: i32,
) -> i32 {
    let mut i: i32 = 0;
    while i < N {
        if val < thresholds[i as usize] {
            break;
        }
        i += 1;
    }
    if i > prev && val < thresholds[prev as usize] + hysteresis[prev as usize] {
        i = prev;
    }
    if i < prev && val > thresholds[(prev - 1) as usize] - hysteresis[(prev - 1) as usize] {
        i = prev;
    }
    i
}

/// Upstream C: celt/bands.c:celt_lcg_rand
#[inline]
pub fn celt_lcg_rand(seed: u32) -> u32 {
    (1664525_u32).wrapping_mul(seed).wrapping_add(1013904223)
}

/// Upstream C: celt/bands.c:bitexact_cos
#[inline]
pub fn bitexact_cos(x: i16) -> i16 {
    let tmp = (4096 + x as i32 * x as i32) >> 13;
    let x2 = tmp as i16;
    let x2 = (32767 - x2 as i32
        + ((16384
            + x2 as i32
                * (-(7651)
                    + ((16384
                        + x2 as i32
                            * (8277 + ((16384 + -626_i16 as i32 * x2 as i32) >> 15)) as i16
                                as i32)
                        >> 15)) as i16 as i32)
            >> 15)) as i16;
    (1 + x2 as i32) as i16
}

/// Upstream C: celt/bands.c:bitexact_log2tan
pub fn bitexact_log2tan(mut isin: i32, mut icos: i32) -> i32 {
    let lc = EC_CLZ0 - (icos as u32).leading_zeros() as i32;
    let ls = EC_CLZ0 - (isin as u32).leading_zeros() as i32;
    icos <<= 15 - lc;
    isin <<= 15 - ls;
    (ls - lc) * ((1) << 11)
        + ((16384
            + isin as i16 as i32
                * (((16384 + isin as i16 as i32 * -2597_i16 as i32) >> 15) + 7932) as i16 as i32)
            >> 15)
        - ((16384
            + icos as i16 as i32
                * (((16384 + icos as i16 as i32 * -2597_i16 as i32) >> 15) + 7932) as i16 as i32)
            >> 15)
}

/// Upstream C: celt/bands.c:compute_band_energies
pub fn compute_band_energies(
    m: &OpusCustomMode,
    X: &[f32],
    bandE: &mut [f32],
    end: i32,
    C: i32,
    LM: i32,
    _arch: Arch,
) {
    let eBands = &m.eBands;
    let N = m.shortMdctSize << LM;
    let mut c = 0;
    loop {
        let mut i = 0;
        while i < end {
            let band_off = (c * N + ((eBands[i as usize] as i32) << LM)) as usize;
            let band_len =
                ((eBands[(i + 1) as usize] as i32 - eBands[i as usize] as i32) << LM) as usize;
            let sum = 1e-27f32
                + celt_inner_prod(
                    &X[band_off..band_off + band_len],
                    &X[band_off..band_off + band_len],
                    band_len,
                    _arch,
                );
            bandE[(i + c * m.nbEBands as i32) as usize] = celt_sqrt(sum);
            i += 1;
        }
        c += 1;
        if c >= C {
            break;
        }
    }
}

/// Upstream C: celt/bands.c:normalise_bands
pub fn normalise_bands(
    m: &OpusCustomMode,
    freq: &[f32],
    X: &mut [f32],
    bandE: &[f32],
    end: i32,
    C: i32,
    M: i32,
) {
    let eBands = &m.eBands;
    let N = M * m.shortMdctSize;
    let mut c = 0;
    loop {
        let mut i = 0;
        while i < end {
            let g = 1.0f32 / (1e-27f32 + bandE[(i + c * m.nbEBands as i32) as usize]);
            let band_start = (M * eBands[i as usize] as i32 + c * N) as usize;
            let band_end = (M * eBands[(i + 1) as usize] as i32 + c * N) as usize;
            for (x, &f) in X[band_start..band_end]
                .iter_mut()
                .zip(&freq[band_start..band_end])
            {
                *x = f * g;
            }
            i += 1;
        }
        c += 1;
        if c >= C {
            break;
        }
    }
}

/// Upstream C: celt/bands.c:denormalise_bands
#[inline]
pub fn denormalise_bands(
    m: &OpusCustomMode,
    X: &[f32],
    freq: &mut [f32],
    bandLogE: &[f32],
    mut start: i32,
    mut end: i32,
    M: i32,
    downsample: i32,
    silence: i32,
) {
    let eBands = &m.eBands;
    let N = M * m.shortMdctSize;
    let mut bound = M * eBands[end as usize] as i32;
    if downsample != 1 {
        bound = if bound < N / downsample {
            bound
        } else {
            N / downsample
        };
    }
    if silence != 0 {
        bound = 0;
        end = 0;
        start = end;
    }
    let start_bin = (M * eBands[start as usize] as i32) as usize;
    if start != 0 {
        freq[..start_bin].fill(0.0);
    }
    let end_bin = (M * eBands[end as usize] as i32) as usize;
    let freq_band = &mut freq[start_bin..end_bin];
    let x_band = &X[start_bin..end_bin];
    let mut off = 0usize;
    for i in start..end {
        let band_len = (M * (eBands[(i + 1) as usize] as i32 - eBands[i as usize] as i32)) as usize;
        let lg = bandLogE[i as usize] + eMeans[i as usize];
        let g = celt_exp2(if 32.0 < lg { 32.0f32 } else { lg });
        for (f, &x) in freq_band[off..off + band_len]
            .iter_mut()
            .zip(&x_band[off..off + band_len])
        {
            *f = x * g;
        }
        off += band_len;
    }
    debug_assert!(start <= end);
    freq[bound as usize..N as usize].fill(0.0);
}

/// Upstream C: celt/bands.c:anti_collapse
#[inline]
pub fn anti_collapse(
    m: &OpusCustomMode,
    X_: &mut [f32],
    collapse_masks: &mut [u8],
    LM: i32,
    C: i32,
    size: i32,
    start: i32,
    end: i32,
    logE: &[f32],
    prev1logE: &[f32],
    prev2logE: &[f32],
    pulses: &[i32],
    mut seed: u32,
    encode: i32,
    arch: Arch,
) {
    let mut i = start;
    while i < end {
        let N0 = m.eBands[(i + 1) as usize] as i32 - m.eBands[i as usize] as i32;
        let depth = (celt_udiv(
            (1 + pulses[i as usize]) as u32,
            (m.eBands[(i + 1) as usize] as i32 - m.eBands[i as usize] as i32) as u32,
        ) >> LM) as i32;
        let thresh = 0.5f32 * celt_exp2(-0.125f32 * depth as f32);
        let sqrt_1 = celt_rsqrt((N0 << LM) as f32);
        let mut c = 0;
        loop {
            let mut prev1 = prev1logE[(c * m.nbEBands as i32 + i) as usize];
            let mut prev2 = prev2logE[(c * m.nbEBands as i32 + i) as usize];
            if encode == 0 && C == 1 {
                prev1 = if prev1 > prev1logE[(m.nbEBands as i32 + i) as usize] {
                    prev1
                } else {
                    prev1logE[(m.nbEBands as i32 + i) as usize]
                };
                prev2 = if prev2 > prev2logE[(m.nbEBands as i32 + i) as usize] {
                    prev2
                } else {
                    prev2logE[(m.nbEBands as i32 + i) as usize]
                };
            }
            let Ediff = logE[(c * m.nbEBands as i32 + i) as usize]
                - (if prev1 < prev2 { prev1 } else { prev2 });
            let Ediff = if 0.0f32 > Ediff { 0.0f32 } else { Ediff };
            let mut r = 2.0f32 * celt_exp2(-Ediff);
            if LM == 3 {
                r *= std::f32::consts::SQRT_2;
            }
            r = if thresh < r { thresh } else { r };
            r *= sqrt_1;
            let x_off = (c * size + ((m.eBands[i as usize] as i32) << LM)) as usize;
            let x_len = (N0 << LM) as usize;
            let mut renormalize = 0;
            let x_sub = &mut X_[x_off..x_off + x_len];
            for k in 0..(1i32 << LM) {
                if collapse_masks[(i * C + c) as usize] as i32 & (1) << k == 0 {
                    for j in 0..N0 {
                        seed = celt_lcg_rand(seed);
                        x_sub[((j << LM) + k) as usize] = if seed & 0x8000 != 0 { r } else { -r };
                    }
                    renormalize = 1;
                }
            }
            if renormalize != 0 {
                renormalise_vector(&mut X_[x_off..x_off + x_len], N0 << LM, Q15ONE, arch);
            }
            c += 1;
            if c >= C {
                break;
            }
        }
        i += 1;
    }
}

/// Upstream C: celt/bands.c:compute_channel_weights
fn compute_channel_weights(mut Ex: f32, mut Ey: f32, w: &mut [f32]) {
    let minE = if Ex < Ey { Ex } else { Ey };
    Ex += minE / 3.0f32;
    Ey += minE / 3.0f32;
    w[0] = Ex;
    w[1] = Ey;
}

/// Upstream C: celt/bands.c:intensity_stereo
#[inline]
fn intensity_stereo(
    m: &OpusCustomMode,
    X: &mut [f32],
    Y: &[f32],
    bandE: &[f32],
    bandID: i32,
    N: i32,
) {
    let i = bandID;
    let left = bandE[i as usize];
    let right = bandE[(i + m.nbEBands as i32) as usize];
    let norm = EPSILON + celt_sqrt(1e-15f32 + left * left + right * right);
    let a1 = left / norm;
    let a2 = right / norm;
    for (x, &r) in X[..N as usize].iter_mut().zip(&Y[..N as usize]) {
        let l = *x;
        *x = a1 * l + a2 * r;
    }
}

/// Upstream C: celt/bands.c:stereo_split
#[inline]
fn stereo_split(X: &mut [f32], Y: &mut [f32], N: i32) {
    for (x, y) in X[..N as usize].iter_mut().zip(Y[..N as usize].iter_mut()) {
        let l = std::f32::consts::FRAC_1_SQRT_2 * *x;
        let r = std::f32::consts::FRAC_1_SQRT_2 * *y;
        *x = l + r;
        *y = r - l;
    }
}

/// Upstream C: celt/bands.c:stereo_merge
#[inline]
fn stereo_merge(X: &mut [f32], Y: &mut [f32], mid: f32, N: i32, _arch: Arch) {
    let n = N as usize;
    let (xp, side) = dual_inner_prod(&Y[..n], &X[..n], &Y[..n], n, _arch);
    let xp = mid * xp;
    let mid2 = mid;
    let El = mid2 * mid2 + side - 2.0f32 * xp;
    let Er = mid2 * mid2 + side + 2.0f32 * xp;
    if Er < 6e-4f32 || El < 6e-4f32 {
        Y[..n].copy_from_slice(&X[..n]);
        return;
    }
    let lgain = celt_rsqrt_norm(El);
    let rgain = celt_rsqrt_norm(Er);
    for (x, y) in X[..N as usize].iter_mut().zip(Y[..N as usize].iter_mut()) {
        let l = mid * *x;
        let r = *y;
        *x = lgain * (l - r);
        *y = rgain * (l + r);
    }
}

/// Upstream C: celt/bands.c:spreading_decision
pub fn spreading_decision(
    m: &OpusCustomMode,
    X: &[f32],
    average: &mut i32,
    last_decision: i32,
    hf_average: &mut i32,
    tapset_decision: &mut i32,
    update_hf: i32,
    end: i32,
    C: i32,
    M: i32,
    spread_weight: &[i32],
) -> i32 {
    let mut sum: i32 = 0;
    let mut nbBands: i32 = 0;
    let eBands = &m.eBands;
    let mut hf_sum: i32 = 0;
    debug_assert!(end > 0);
    let N0 = M * m.shortMdctSize;
    if M * (eBands[end as usize] as i32 - eBands[(end - 1) as usize] as i32) <= 8 {
        return SPREAD_NONE;
    }
    let mut c = 0;
    loop {
        let mut i = 0;
        while i < end {
            let N = M * (eBands[(i + 1) as usize] as i32 - eBands[i as usize] as i32);
            if N > 8 {
                let x_off = (M * eBands[i as usize] as i32 + c * N0) as usize;
                let mut tcount: [i32; 3] = [0, 0, 0];
                let x_band = &X[x_off..x_off + N as usize];
                for &xv in x_band {
                    let x2N = xv * xv * N as f32;
                    if x2N < 0.25f32 {
                        tcount[0] += 1;
                    }
                    if x2N < 0.0625f32 {
                        tcount[1] += 1;
                    }
                    if x2N < 0.015625f32 {
                        tcount[2] += 1;
                    }
                }
                if i > m.nbEBands as i32 - 4 {
                    hf_sum = (hf_sum as u32)
                        .wrapping_add(celt_udiv((32 * (tcount[1] + tcount[0])) as u32, N as u32))
                        as i32;
                }
                let tmp = (2 * tcount[2] >= N) as i32
                    + (2 * tcount[1] >= N) as i32
                    + (2 * tcount[0] >= N) as i32;
                sum += tmp * spread_weight[i as usize];
                nbBands += spread_weight[i as usize];
            }
            i += 1;
        }
        c += 1;
        if c >= C {
            break;
        }
    }
    if update_hf != 0 {
        if hf_sum != 0 {
            hf_sum = celt_udiv(hf_sum as u32, (C * (4 - m.nbEBands as i32 + end)) as u32) as i32;
        }
        *hf_average = (*hf_average + hf_sum) >> 1;
        hf_sum = *hf_average;
        if *tapset_decision == 2 {
            hf_sum += 4;
        } else if *tapset_decision == 0 {
            hf_sum -= 4;
        }
        if hf_sum > 22 {
            *tapset_decision = 2;
        } else if hf_sum > 18 {
            *tapset_decision = 1;
        } else {
            *tapset_decision = 0;
        }
    }
    debug_assert!(nbBands > 0);
    debug_assert!(sum >= 0);
    sum = celt_udiv((sum << 8) as u32, nbBands as u32) as i32;
    sum = (sum + *average) >> 1;
    *average = sum;
    sum = (3 * sum + (((3 - last_decision) << 7) + 64) + 2) >> 2;
    if sum < 80 {
        SPREAD_AGGRESSIVE
    } else if sum < 256 {
        SPREAD_NORMAL
    } else if sum < 384 {
        SPREAD_LIGHT
    } else {
        SPREAD_NONE
    }
}

static ordery_table: [i32; 30] = [
    1, 0, 3, 0, 2, 1, 7, 0, 4, 3, 6, 1, 5, 2, 15, 0, 8, 7, 12, 3, 11, 4, 14, 1, 9, 6, 13, 2, 10, 5,
];

/// Upstream C: celt/bands.c:deinterleave_hadamard
#[inline]
fn deinterleave_hadamard(X: &mut [f32], N0: i32, stride: i32, hadamard: i32) {
    let N = (N0 * stride) as usize;
    let mut tmp = [0.0f32; 176];
    debug_assert!(stride > 0);
    let tmp = &mut tmp[..N];
    let x = &X[..N];
    if hadamard != 0 {
        let ordery = &ordery_table[(stride - 2) as usize..];
        for i in 0..stride as usize {
            let dst_base = (ordery[i] * N0) as usize;
            for j in 0..N0 as usize {
                tmp[dst_base + j] = x[j * stride as usize + i];
            }
        }
    } else {
        for i in 0..stride as usize {
            let dst_base = i * N0 as usize;
            for j in 0..N0 as usize {
                tmp[dst_base + j] = x[j * stride as usize + i];
            }
        }
    }
    X[..N].copy_from_slice(tmp);
}

/// Upstream C: celt/bands.c:interleave_hadamard
#[inline]
fn interleave_hadamard(X: &mut [f32], N0: i32, stride: i32, hadamard: i32) {
    let N = (N0 * stride) as usize;
    let mut tmp = [0.0f32; 176];
    let tmp = &mut tmp[..N];
    let x = &X[..N];
    if hadamard != 0 {
        let ordery = &ordery_table[(stride - 2) as usize..];
        for i in 0..stride as usize {
            let src_base = (ordery[i] * N0) as usize;
            for j in 0..N0 as usize {
                tmp[j * stride as usize + i] = x[src_base + j];
            }
        }
    } else {
        for i in 0..stride as usize {
            let src_base = i * N0 as usize;
            for j in 0..N0 as usize {
                tmp[j * stride as usize + i] = x[src_base + j];
            }
        }
    }
    X[..N].copy_from_slice(tmp);
}

/// Upstream C: celt/bands.c:haar1
#[inline]
pub fn haar1(X: &mut [f32], mut N0: i32, stride: i32) {
    let total = N0 as usize * stride as usize;
    let X = &mut X[..total];
    N0 >>= 1;
    for i in 0..stride as usize {
        for j in 0..N0 as usize {
            let idx0 = stride as usize * 2 * j + i;
            let idx1 = stride as usize * (2 * j + 1) + i;
            let tmp1 = std::f32::consts::FRAC_1_SQRT_2 * X[idx0];
            let tmp2 = std::f32::consts::FRAC_1_SQRT_2 * X[idx1];
            X[idx0] = tmp1 + tmp2;
            X[idx1] = tmp1 - tmp2;
        }
    }
}

/// Upstream C: celt/bands.c:compute_qn
#[inline]
fn compute_qn(N: i32, b: i32, offset: i32, pulse_cap: i32, stereo: i32) -> i32 {
    const EXP2_TABLE8: [i16; 8] = [16384, 17866, 19483, 21247, 23170, 25267, 27554, 30048];
    let mut N2 = 2 * N - 1;
    if stereo != 0 && N == 2 {
        N2 -= 1;
    }
    let mut qb = celt_sudiv(b + N2 * offset, N2);
    qb = if b - pulse_cap - ((4) << 3) < qb {
        b - pulse_cap - ((4) << 3)
    } else {
        qb
    };
    qb = if ((8) << 3) < qb { (8) << 3 } else { qb };
    let qn = if qb < (1) << BITRES >> 1 {
        1
    } else {
        let raw = EXP2_TABLE8[(qb & 0x7) as usize] as i32 >> (14 - (qb >> BITRES));
        ((raw + 1) >> 1) << 1
    };
    debug_assert!(qn <= 256);
    qn
}

/// Upstream C: celt/bands.c:compute_theta
///
/// Uses raw pointers internally for X/Y because the caller (`quant_partition`)
/// needs to split X at varying offsets and pass sub-slices. The pointer
/// arithmetic is confined to this function.
#[inline(never)]
fn compute_theta(
    ctx: &mut band_ctx,
    sctx: &mut split_ctx,
    X: &mut [f32],
    Y: &mut [f32],
    N: i32,
    b: &mut i32,
    B: i32,
    B0: i32,
    LM: i32,
    stereo: i32,
    fill: &mut i32,
    ec: &mut ec_ctx,
) {
    let mut itheta: i32 = 0;
    #[allow(unused_assignments)]
    let mut itheta_q30: i32 = 0;
    let mut imid: i32;
    let mut iside: i32;
    let mut inv: i32 = 0;
    let encode = ctx.encode;
    let m = ctx.m;
    let i = ctx.i;
    let intensity = ctx.intensity;
    let bandE = ctx.bandE;
    let pulse_cap = m.logN[i as usize] as i32 + LM * ((1) << BITRES);
    let offset = (pulse_cap >> 1)
        - (if stereo != 0 && N == 2 {
            QTHETA_OFFSET_TWOPHASE
        } else {
            QTHETA_OFFSET
        });
    let mut qn = compute_qn(N, *b, offset, pulse_cap, stereo);
    if stereo != 0 && i >= intensity {
        qn = 1;
    }
    if encode != 0 {
        itheta_q30 = stereo_itheta(&X[..N as usize], &Y[..N as usize], stereo, N, ctx.arch);
        itheta = itheta_q30 >> 16;
    }
    let tell = ec_tell_frac(ec) as i32;
    if qn != 1 {
        if encode != 0 {
            if stereo == 0 || ctx.theta_round == 0 {
                itheta = (itheta * qn + 8192) >> 14;
                if stereo == 0 && ctx.avoid_split_noise != 0 && itheta > 0 && itheta < qn {
                    let unquantized = celt_udiv((itheta * 16384) as u32, qn as u32) as i32;
                    imid = bitexact_cos(unquantized as i16) as i32;
                    iside = bitexact_cos((16384 - unquantized) as i16) as i32;
                    let delta = (16384
                        + ((N - 1) << 7) as i16 as i32
                            * bitexact_log2tan(iside, imid) as i16 as i32)
                        >> 15;
                    if delta > *b {
                        itheta = qn;
                    } else if delta < -*b {
                        itheta = 0;
                    }
                }
            } else {
                let bias: i32 = if itheta > 8192 {
                    32767 / qn
                } else {
                    -(32767) / qn
                };
                let down = if (qn - 1)
                    < (if 0 > (itheta * qn + bias) >> 14 {
                        0
                    } else {
                        (itheta * qn + bias) >> 14
                    }) {
                    qn - 1
                } else if 0 > (itheta * qn + bias) >> 14 {
                    0
                } else {
                    (itheta * qn + bias) >> 14
                };
                if ctx.theta_round < 0 {
                    itheta = down;
                } else {
                    itheta = down + 1;
                }
            }
        }
        if stereo != 0 && N > 2 {
            let p0: i32 = 3;
            let mut x = itheta;
            let x0 = qn / 2;
            let ft = p0 * (x0 + 1) + x0;
            if encode != 0 {
                ec_encode(
                    ec,
                    (if x <= x0 {
                        p0 * x
                    } else {
                        x - 1 - x0 + (x0 + 1) * p0
                    }) as u32,
                    (if x <= x0 {
                        p0 * (x + 1)
                    } else {
                        x - x0 + (x0 + 1) * p0
                    }) as u32,
                    ft as u32,
                );
            } else {
                let fs = ec_decode(ec, ft as u32) as i32;
                if fs < (x0 + 1) * p0 {
                    x = fs / p0;
                } else {
                    x = x0 + 1 + (fs - (x0 + 1) * p0);
                }
                ec_dec_update(
                    ec,
                    (if x <= x0 {
                        p0 * x
                    } else {
                        x - 1 - x0 + (x0 + 1) * p0
                    }) as u32,
                    (if x <= x0 {
                        p0 * (x + 1)
                    } else {
                        x - x0 + (x0 + 1) * p0
                    }) as u32,
                    ft as u32,
                );
                itheta = x;
            }
        } else if B0 > 1 || stereo != 0 {
            if encode != 0 {
                ec_enc_uint(ec, itheta as u32, (qn + 1) as u32);
            } else {
                itheta = ec_dec_uint(ec, (qn + 1) as u32) as i32;
            }
        } else {
            let mut fs_0: i32 = 1;
            let ft_0 = ((qn >> 1) + 1) * ((qn >> 1) + 1);
            if encode != 0 {
                fs_0 = if itheta <= qn >> 1 {
                    itheta + 1
                } else {
                    qn + 1 - itheta
                };
                let fl = if itheta <= qn >> 1 {
                    (itheta * (itheta + 1)) >> 1
                } else {
                    ft_0 - (((qn + 1 - itheta) * (qn + 2 - itheta)) >> 1)
                };
                ec_encode(ec, fl as u32, (fl + fs_0) as u32, ft_0 as u32);
            } else {
                let fm = ec_decode(ec, ft_0 as u32) as i32;
                if fm < ((qn >> 1) * ((qn >> 1) + 1)) >> 1 {
                    itheta = ((isqrt32((8_u32).wrapping_mul(fm as u32).wrapping_add(1)))
                        .wrapping_sub(1)
                        >> 1) as i32;
                    fs_0 = itheta + 1;
                    let fl_0 = (itheta * (itheta + 1)) >> 1;
                    ec_dec_update(ec, fl_0 as u32, (fl_0 + fs_0) as u32, ft_0 as u32);
                } else {
                    itheta = (((2 * (qn + 1)) as u32).wrapping_sub(isqrt32(
                        (8_u32).wrapping_mul((ft_0 - fm - 1) as u32).wrapping_add(1),
                    )) >> 1) as i32;
                    fs_0 = qn + 1 - itheta;
                    let fl_0 = ft_0 - (((qn + 1 - itheta) * (qn + 2 - itheta)) >> 1);
                    ec_dec_update(ec, fl_0 as u32, (fl_0 + fs_0) as u32, ft_0 as u32);
                }
            }
        }
        debug_assert!(itheta >= 0);
        itheta = celt_udiv((itheta * 16384) as u32, qn as u32) as i32;
        #[cfg(feature = "qext")]
        {
            ctx.ext_b = ctx
                .ext_b
                .min(ctx.ext_total_bits - ec_tell_frac(unsafe { &*ctx.ext_ec }) as i32);
            if ctx.ext_b >= (2 * N) << BITRES
                && ctx.ext_total_bits - ec_tell_frac(unsafe { &*ctx.ext_ec }) as i32 - 1
                    > 2 << BITRES
            {
                let extra_bits = 12.min(2.max(celt_sudiv(ctx.ext_b, (2 * N - 1) << BITRES)));
                let ext_tell = ec_tell_frac(unsafe { &*ctx.ext_ec }) as i32;
                if encode != 0 {
                    itheta_q30 -= itheta << 16;
                    itheta_q30 = ((itheta_q30 as i64 * qn as i64 * ((1 << extra_bits) - 1) as i64
                        + (1i64 << 29))
                        >> 30) as i32;
                    itheta_q30 += (1 << (extra_bits - 1)) - 1;
                    itheta_q30 = 0.max(((1 << extra_bits) - 2).min(itheta_q30));
                    ec_enc_uint(
                        unsafe { &mut *ctx.ext_ec },
                        itheta_q30 as u32,
                        ((1 << extra_bits) - 1) as u32,
                    );
                } else {
                    itheta_q30 =
                        ec_dec_uint(unsafe { &mut *ctx.ext_ec }, ((1 << extra_bits) - 1) as u32)
                            as i32;
                }
                itheta_q30 -= (1 << (extra_bits - 1)) - 1;
                itheta_q30 = (itheta << 16)
                    + (itheta_q30 as i64 * (1i64 << 30)
                        / (qn as i64 * ((1 << extra_bits) - 1) as i64))
                        as i32;
                // Hard bounds on itheta (can only trigger on corrupted bitstreams).
                itheta_q30 = 0.max(1073741824i32.min(itheta_q30));
                ctx.ext_b -= ec_tell_frac(unsafe { &*ctx.ext_ec }) as i32 - ext_tell;
            } else {
                itheta_q30 = itheta << 16;
            }
        }
        if encode != 0 && stereo != 0 {
            if itheta == 0 {
                intensity_stereo(m, X, Y, bandE, i, N);
            } else {
                stereo_split(X, Y, N);
            }
        }
    } else if stereo != 0 {
        if encode != 0 {
            inv = (itheta > 8192 && ctx.disable_inv == 0) as i32;
            if inv != 0 {
                for y in &mut Y[..N as usize] {
                    *y = -*y;
                }
            }
            intensity_stereo(m, X, Y, bandE, i, N);
        }
        if *b > (2) << BITRES && ctx.remaining_bits > (2) << BITRES {
            if encode != 0 {
                ec_enc_bit_logp(ec, inv, 2);
            } else {
                inv = ec_dec_bit_logp(ec, 2);
            }
        } else {
            inv = 0;
        }
        if ctx.disable_inv != 0 {
            inv = 0;
        }
        itheta = 0;
        itheta_q30 = 0;
    }
    let qalloc = (ec_tell_frac(ec)).wrapping_sub(tell as u32) as i32;
    *b -= qalloc;
    if itheta == 0 {
        imid = 32767;
        iside = 0;
        *fill &= ((1) << B) - 1;
        sctx.delta = -(16384);
    } else if itheta == 16384 {
        imid = 0;
        iside = 32767;
        *fill &= (((1) << B) - 1) << B;
        sctx.delta = 16384;
    } else {
        imid = bitexact_cos(itheta as i16) as i32;
        iside = bitexact_cos((16384 - itheta) as i16) as i32;
        sctx.delta = (16384
            + ((N - 1) << 7) as i16 as i32 * bitexact_log2tan(iside, imid) as i16 as i32)
            >> 15;
    }
    sctx.inv = inv;
    sctx.imid = imid;
    sctx.iside = iside;
    sctx.itheta = itheta;
    sctx.qalloc = qalloc;
    #[cfg(feature = "qext")]
    {
        if std::env::var_os("OPURS_QEXT_TRACE").is_some() && encode == 0 && i == 0 {
            eprintln!(
                "[rust theta] i={} N={} stereo={} qn={} b={} qalloc={} itheta={} itheta_q30={} delta={} imid={} iside={} ext_b={} tell={} ext_tell={}",
                i,
                N,
                stereo,
                qn,
                *b,
                qalloc,
                itheta,
                itheta_q30,
                sctx.delta,
                imid,
                iside,
                ctx.ext_b,
                ec_tell_frac(ec),
                ec_tell_frac(unsafe { &*ctx.ext_ec }),
            );
        }
        sctx.itheta_q30 = itheta_q30;
    }
}

/// Upstream C: celt/bands.c:quant_band_n1
#[inline]
fn quant_band_n1(
    ctx: &mut band_ctx,
    X: &mut [f32],
    Y: Option<&mut [f32]>,
    mut _b: i32,
    lowband_out: Option<&mut [f32]>,
    ec: &mut ec_ctx,
) -> u32 {
    let encode = ctx.encode;
    let _stereo = if Y.is_some() { 1 } else { 0 };
    // c=0: process X
    {
        let mut sign: i32 = 0;
        if ctx.remaining_bits >= (1) << BITRES {
            if encode != 0 {
                sign = (X[0] < 0.0f32) as i32;
                ec_enc_bits(ec, sign as u32, 1);
            } else {
                sign = ec_dec_bits(ec, 1) as i32;
            }
            ctx.remaining_bits -= (1) << BITRES;
            _b -= (1) << BITRES;
        }
        if ctx.resynth != 0 {
            X[0] = if sign != 0 {
                -NORM_SCALING
            } else {
                NORM_SCALING
            };
        }
    }
    // c=1: process Y (if stereo)
    if let Some(y) = Y {
        let mut sign: i32 = 0;
        if ctx.remaining_bits >= (1) << BITRES {
            if encode != 0 {
                sign = (y[0] < 0.0f32) as i32;
                ec_enc_bits(ec, sign as u32, 1);
            } else {
                sign = ec_dec_bits(ec, 1) as i32;
            }
            ctx.remaining_bits -= (1) << BITRES;
            _b -= (1) << BITRES;
        }
        if ctx.resynth != 0 {
            y[0] = if sign != 0 {
                -NORM_SCALING
            } else {
                NORM_SCALING
            };
        }
    }
    if let Some(lbo) = lowband_out {
        lbo[0] = X[0];
    }
    1
}

/// Upstream C: celt/bands.c:quant_partition
#[inline]
fn quant_partition(
    ctx: &mut band_ctx,
    X: &mut [f32],
    mut N: i32,
    mut b: i32,
    mut B: i32,
    lowband: Option<&[f32]>,
    mut LM: i32,
    gain: f32,
    mut fill: i32,
    ec: &mut ec_ctx,
) -> u32 {
    #[cfg(feature = "qext")]
    let qp_trace = std::env::var_os("OPURS_QEXT_TRACE").is_some() && ctx.encode == 0 && ctx.i == 20;
    #[cfg(feature = "qext")]
    if qp_trace {
        eprintln!(
            "[rust qp] enter N={} b={} B={} LM={} ext_b={} xh={:016x}",
            N,
            b,
            B,
            LM,
            ctx.ext_b,
            qext_hash_band(&X[..N as usize]),
        );
    }
    let B0 = B;
    let mut cm: u32;
    let encode = ctx.encode;
    let m = ctx.m;
    let i = ctx.i;
    let spread = ctx.spread;
    let cache =
        &m.cache.bits[m.cache.index[((LM + 1) * m.nbEBands as i32 + i) as usize] as usize..];
    if LM != -1 && b > cache[cache[0] as usize] as i32 + 12 && N > 2 {
        let mut sctx = split_ctx {
            inv: 0,
            imid: 0,
            iside: 0,
            delta: 0,
            itheta: 0,
            qalloc: 0,
            #[cfg(feature = "qext")]
            itheta_q30: 0,
        };
        N >>= 1;
        let n = N as usize;
        LM -= 1;
        if B == 1 {
            fill = fill & 1 | fill << 1;
        }
        B = (B + 1) >> 1;
        {
            let (x_lo, x_hi) = X.split_at_mut(n);
            compute_theta(
                ctx,
                &mut sctx,
                x_lo,
                &mut x_hi[..n],
                N,
                &mut b,
                B,
                B0,
                LM,
                0,
                &mut fill,
                ec,
            );
        }
        let imid = sctx.imid;
        let iside = sctx.iside;
        let mut delta = sctx.delta;
        let itheta = sctx.itheta;
        let qalloc = sctx.qalloc;
        #[cfg(feature = "qext")]
        let mid: f32;
        #[cfg(feature = "qext")]
        let side: f32;
        #[cfg(not(feature = "qext"))]
        let mid: f32;
        #[cfg(not(feature = "qext"))]
        let side: f32;
        #[cfg(feature = "qext")]
        {
            let _ = (imid, iside);
            mid = celt_cos_norm2(sctx.itheta_q30 as f32 * (1.0f32 / (1i32 << 30) as f32));
            side = celt_cos_norm2(1.0f32 - sctx.itheta_q30 as f32 * (1.0f32 / (1i32 << 30) as f32));
        }
        #[cfg(not(feature = "qext"))]
        {
            mid = 1.0f32 / 32768.0f32 * imid as f32;
            side = 1.0f32 / 32768.0f32 * iside as f32;
        }
        if B0 > 1 && itheta & 0x3fff != 0 {
            if itheta > 8192 {
                delta -= delta >> (4 - LM);
            } else {
                delta = if 0 < delta + (N << 3 >> (5 - LM)) {
                    0
                } else {
                    delta + (N << 3 >> (5 - LM))
                };
            }
        }
        let mut mbits = if 0
            > (if b < (b - delta) / 2 {
                b
            } else {
                (b - delta) / 2
            }) {
            0
        } else if b < (b - delta) / 2 {
            b
        } else {
            (b - delta) / 2
        };
        let mut sbits = b - mbits;
        ctx.remaining_bits -= qalloc;
        let next_lowband2 = lowband.map(|lb| &lb[n..]);
        let mut rebalance = ctx.remaining_bits;
        #[cfg(feature = "qext")]
        let saved_ext_b = ctx.ext_b;
        #[cfg(feature = "qext")]
        {
            ctx.ext_b = saved_ext_b / 2;
        }
        if mbits >= sbits {
            {
                let (x_lo, x_hi) = X.split_at_mut(n);
                cm = quant_partition(ctx, x_lo, N, mbits, B, lowband, LM, gain * mid, fill, ec);
                rebalance = mbits - (rebalance - ctx.remaining_bits);
                if rebalance > (3) << BITRES && itheta != 0 {
                    sbits += rebalance - ((3) << BITRES);
                }
                #[cfg(feature = "qext")]
                {
                    ctx.ext_b = saved_ext_b / 2;
                }
                cm |= quant_partition(
                    ctx,
                    &mut x_hi[..n],
                    N,
                    sbits,
                    B,
                    next_lowband2,
                    LM,
                    gain * side,
                    fill >> B,
                    ec,
                ) << (B0 >> 1);
            }
        } else {
            {
                let (x_lo, x_hi) = X.split_at_mut(n);
                cm = quant_partition(
                    ctx,
                    &mut x_hi[..n],
                    N,
                    sbits,
                    B,
                    next_lowband2,
                    LM,
                    gain * side,
                    fill >> B,
                    ec,
                ) << (B0 >> 1);
                rebalance = sbits - (rebalance - ctx.remaining_bits);
                if rebalance > (3) << BITRES && itheta != 16384 {
                    mbits += rebalance - ((3) << BITRES);
                }
                #[cfg(feature = "qext")]
                {
                    ctx.ext_b = saved_ext_b / 2;
                }
                cm |= quant_partition(ctx, x_lo, N, mbits, B, lowband, LM, gain * mid, fill, ec);
            }
        }
    } else {
        #[cfg(feature = "qext")]
        let extra_bits: i32;
        #[cfg(feature = "qext")]
        {
            let mut eb = (ctx.ext_b / (N - 1)) >> BITRES;
            let ext_remaining_bits =
                ctx.ext_total_bits - ec_tell_frac(unsafe { &*ctx.ext_ec }) as i32;
            if ext_remaining_bits < ((eb + 1) * (N - 1) + N) << BITRES {
                eb = ((ext_remaining_bits - (N << BITRES)) / (N - 1)) >> BITRES;
                eb = 0.max(eb - 1);
            }
            extra_bits = 12.min(eb);
        }
        let q = {
            let mut q = bits2pulses(m, i, LM, b);
            let mut curr_bits = pulses2bits(m, i, LM, q);
            ctx.remaining_bits -= curr_bits;
            while ctx.remaining_bits < 0 && q > 0 {
                ctx.remaining_bits += curr_bits;
                q -= 1;
                curr_bits = pulses2bits(m, i, LM, q);
                ctx.remaining_bits -= curr_bits;
            }
            q
        };
        if q != 0 {
            let K = get_pulses(q);
            if encode != 0 {
                cm = alg_quant(
                    &mut X[..N as usize],
                    N,
                    K,
                    spread,
                    B,
                    ec,
                    gain,
                    ctx.resynth,
                    ctx.arch,
                    #[cfg(feature = "qext")]
                    unsafe {
                        &mut *ctx.ext_ec
                    },
                    #[cfg(feature = "qext")]
                    extra_bits,
                );
            } else {
                cm = alg_unquant(
                    &mut X[..N as usize],
                    N,
                    K,
                    spread,
                    B,
                    ec,
                    gain,
                    #[cfg(feature = "qext")]
                    unsafe {
                        &mut *ctx.ext_ec
                    },
                    #[cfg(feature = "qext")]
                    extra_bits,
                );
            }
        } else {
            #[cfg(feature = "qext")]
            if ctx.ext_b > (2 * N) << BITRES {
                // No pulses but have extension bits: use cubic quantization.
                let mut eb = (ctx.ext_b / (N - 1)) >> BITRES;
                let ext_remaining_bits =
                    ctx.ext_total_bits - ec_tell_frac(unsafe { &*ctx.ext_ec }) as i32;
                if ext_remaining_bits < ((eb + 1) * (N - 1) + N) << BITRES {
                    eb = ((ext_remaining_bits - (N << BITRES)) / (N - 1)) >> BITRES;
                    eb = 0.max(eb - 1);
                }
                let cubic_bits = 14.min(eb);
                if encode != 0 {
                    cm = cubic_quant(
                        &mut X[..N as usize],
                        N,
                        cubic_bits,
                        B,
                        unsafe { &mut *ctx.ext_ec },
                        gain,
                        ctx.resynth,
                    );
                } else {
                    cm = cubic_unquant(
                        &mut X[..N as usize],
                        N,
                        cubic_bits,
                        B,
                        unsafe { &mut *ctx.ext_ec },
                        gain,
                    );
                }
            } else {
                cm = 0;
                if ctx.resynth != 0 {
                    let cm_mask = (((1_u64) << B) as u32).wrapping_sub(1);
                    fill = (fill as u32 & cm_mask) as i32;
                    if fill == 0 {
                        X[..N as usize].fill(0.0);
                    } else {
                        let n = N as usize;
                        if let Some(lb) = lowband {
                            for (x, &l) in X[..n].iter_mut().zip(&lb[..n]) {
                                let mut tmp = 1.0f32 / 256.0f32;
                                ctx.seed = celt_lcg_rand(ctx.seed);
                                tmp = if ctx.seed & 0x8000 != 0 { tmp } else { -tmp };
                                *x = l + tmp;
                            }
                            cm = fill as u32;
                        } else {
                            for x in &mut X[..n] {
                                ctx.seed = celt_lcg_rand(ctx.seed);
                                *x = (ctx.seed as i32 >> 20) as f32;
                            }
                            cm = cm_mask;
                        }
                        renormalise_vector(&mut X[..n], N, gain, ctx.arch);
                    }
                }
            }
            #[cfg(not(feature = "qext"))]
            {
                cm = 0;
                if ctx.resynth != 0 {
                    let cm_mask = (((1_u64) << B) as u32).wrapping_sub(1);
                    fill = (fill as u32 & cm_mask) as i32;
                    let n = N as usize;
                    if fill == 0 {
                        X[..n].fill(0.0);
                    } else {
                        if let Some(lb) = lowband {
                            for (x, &l) in X[..n].iter_mut().zip(&lb[..n]) {
                                let mut tmp = 1.0f32 / 256.0f32;
                                ctx.seed = celt_lcg_rand(ctx.seed);
                                tmp = if ctx.seed & 0x8000 != 0 { tmp } else { -tmp };
                                *x = l + tmp;
                            }
                            cm = fill as u32;
                        } else {
                            for x in &mut X[..n] {
                                ctx.seed = celt_lcg_rand(ctx.seed);
                                *x = (ctx.seed as i32 >> 20) as f32;
                            }
                            cm = cm_mask;
                        }
                        renormalise_vector(&mut X[..n], N, gain, ctx.arch);
                    }
                }
            }
        }
    }
    #[cfg(feature = "qext")]
    if qp_trace {
        eprintln!(
            "[rust qp] exit  N={} b={} B={} LM={} ext_b={} cm={} xh={:016x}",
            N,
            b,
            B,
            LM,
            ctx.ext_b,
            cm,
            qext_hash_band(&X[..N as usize]),
        );
    }
    cm
}

/// Upstream C: celt/bands.c:cubic_quant_partition
///
/// Alternative quantization path for QEXT at very high bitrates.
/// Uses cubic quantization instead of PVQ.
#[cfg(feature = "qext")]
fn cubic_quant_partition(
    ctx: &mut band_ctx,
    X: &mut [f32],
    mut N: i32,
    mut b: i32,
    B: i32,
    ec: &mut ec_ctx,
    mut LM: i32,
    gain: f32,
    resynth: i32,
    encode: i32,
) -> u32 {
    debug_assert!(LM >= 0);
    ctx.remaining_bits = ec.storage as i32 * 8 * 8 - ec_tell_frac(ec) as i32;
    b = b.min(ctx.remaining_bits);
    if LM == 0 || b <= (2 * N) << BITRES {
        b = (b + ((N - 1) << BITRES) / 2).min(ctx.remaining_bits);
        // Resolution left after coding the cube face
        let res = ((b - (1 << BITRES) - ctx.m.logN[ctx.i as usize] as i32 - (LM << BITRES) - 1)
            / (N - 1))
            >> BITRES;
        let res = 14.min(0.max(res));
        let ret = if encode != 0 {
            cubic_quant(X, N, res, B, ec, gain, resynth)
        } else {
            cubic_unquant(X, N, res, B, ec, gain)
        };
        ctx.remaining_bits = ec.storage as i32 * 8 * 8 - ec_tell_frac(ec) as i32;
        ret
    } else {
        let N0 = N;
        N >>= 1;
        let n = N as usize;
        LM -= 1;
        let B_new = (B + 1) >> 1;
        // Allocate bits for theta (1-16 bits)
        let theta_res = 16.min((b >> BITRES) / (N0 - 1) + 1);
        let qtheta = if encode != 0 {
            let (x_lo, x_hi) = X.split_at_mut(n);
            let raw_itheta = stereo_itheta(&x_lo[..n], &x_hi[..n], 0, N, ctx.arch);
            let q = (raw_itheta + (1 << (29 - theta_res))) >> (30 - theta_res);
            ec_enc_uint(ec, q as u32, ((1 << theta_res) + 1) as u32);
            q
        } else {
            ec_dec_uint(ec, ((1 << theta_res) + 1) as u32) as i32
        };
        let itheta_q30 = qtheta << (30 - theta_res);
        b -= theta_res << BITRES;
        let delta = ((N0 - 1) * 23 * ((itheta_q30 >> 16) - 8192)) >> (17 - BITRES);
        let g1 = celt_cos_norm2(itheta_q30 as f32 * (1.0f32 / (1i32 << 30) as f32));
        let g2 = celt_cos_norm2(1.0f32 - itheta_q30 as f32 * (1.0f32 / (1i32 << 30) as f32));
        let (b1, b2);
        if itheta_q30 == 0 {
            b1 = b;
            b2 = 0;
        } else if itheta_q30 == 1073741824 {
            b1 = 0;
            b2 = b;
        } else {
            b1 = b.min(0.max((b - delta) / 2));
            b2 = b - b1;
        }
        let (x_lo, x_hi) = X.split_at_mut(n);
        let mut cm =
            cubic_quant_partition(ctx, x_lo, N, b1, B_new, ec, LM, gain * g1, resynth, encode);
        cm |= cubic_quant_partition(
            ctx,
            &mut x_hi[..n],
            N,
            b2,
            B_new,
            ec,
            LM,
            gain * g2,
            resynth,
            encode,
        );
        cm
    }
}

/// Upstream C: celt/bands.c:quant_band
#[inline]
fn quant_band(
    ctx: &mut band_ctx,
    X: &mut [f32],
    N: i32,
    b: i32,
    mut B: i32,
    lowband: Option<&mut [f32]>,
    LM: i32,
    lowband_out: Option<&mut [f32]>,
    gain: f32,
    lowband_scratch: Option<&mut [f32]>,
    mut fill: i32,
    ec: &mut ec_ctx,
) -> u32 {
    let N0 = N;
    let mut N_B = N;

    let mut B0 = B;
    let mut time_divide: i32 = 0;
    let mut recombine: i32 = 0;
    let longBlocks = (B0 == 1) as i32;
    let mut cm: u32;
    let encode = ctx.encode;
    let mut tf_change = ctx.tf_change;
    N_B = celt_udiv(N_B as u32, B as u32) as i32;
    if N == 1 {
        let y_opt: Option<&mut [f32]> = None;
        return quant_band_n1(ctx, &mut X[..1], y_opt, b, lowband_out, ec);
    }
    if tf_change > 0 {
        recombine = tf_change;
    }
    // If we need to transform lowband, copy it into scratch first.
    // After this, `lb_work` is the working lowband buffer (either scratch with
    // copied data, or original lowband, or None).
    let need_scratch = lowband.is_some()
        && lowband_scratch.is_some()
        && (recombine != 0 || N_B & 1 == 0 && tf_change < 0 || B0 > 1);
    let mut lb_work: Option<&mut [f32]> = if need_scratch {
        let scratch = lowband_scratch.unwrap();
        let lb = lowband.unwrap();
        scratch[..N as usize].copy_from_slice(&lb[..N as usize]);
        Some(scratch)
    } else {
        lowband
    };

    const BIT_INTERLEAVE_TABLE: [u8; 16] = [0, 1, 1, 1, 2, 3, 3, 3, 2, 3, 3, 3, 2, 3, 3, 3];

    let mut k = 0;
    while k < recombine {
        if encode != 0 {
            haar1(&mut X[..N as usize], N >> k, (1) << k);
        }
        if let Some(ref mut lb) = lb_work {
            haar1(&mut lb[..N as usize], N >> k, (1) << k);
        }
        fill = BIT_INTERLEAVE_TABLE[(fill & 0xf) as usize] as i32
            | (BIT_INTERLEAVE_TABLE[(fill >> 4) as usize] as i32) << 2;
        k += 1;
    }
    B >>= recombine;
    N_B <<= recombine;
    while N_B & 1 == 0 && tf_change < 0 {
        if encode != 0 {
            haar1(&mut X[..N as usize], N_B, B);
        }
        if let Some(ref mut lb) = lb_work {
            haar1(&mut lb[..N as usize], N_B, B);
        }
        fill |= fill << B;
        B <<= 1;
        N_B >>= 1;
        time_divide += 1;
        tf_change += 1;
    }
    B0 = B;
    let N_B0: i32 = N_B;
    if B0 > 1 {
        if encode != 0 {
            deinterleave_hadamard(
                &mut X[..N as usize],
                N_B >> recombine,
                B0 << recombine,
                longBlocks,
            );
        }
        if let Some(ref mut lb) = lb_work {
            deinterleave_hadamard(
                &mut lb[..N as usize],
                N_B >> recombine,
                B0 << recombine,
                longBlocks,
            );
        }
    }
    // Pass lowband as read-only to quant_partition
    let lb_ref: Option<&[f32]> = lb_work.as_deref();
    #[cfg(feature = "qext")]
    {
        if ctx.extra_bands
            && b > ((3 * N) << BITRES) + (ctx.m.logN[ctx.i as usize] as i32 + 8 + 8 * LM)
        {
            cm = cubic_quant_partition(ctx, X, N, b, B, ec, LM, gain, ctx.resynth, ctx.encode);
        } else {
            cm = quant_partition(ctx, X, N, b, B, lb_ref, LM, gain, fill, ec);
        }
    }
    #[cfg(not(feature = "qext"))]
    {
        cm = quant_partition(ctx, X, N, b, B, lb_ref, LM, gain, fill, ec);
    }
    if ctx.resynth != 0 {
        if B0 > 1 {
            interleave_hadamard(
                &mut X[..N as usize],
                N_B >> recombine,
                B0 << recombine,
                longBlocks,
            );
        }
        N_B = N_B0;
        B = B0;
        k = 0;
        while k < time_divide {
            B >>= 1;
            N_B <<= 1;
            cm |= cm >> B;
            haar1(&mut X[..N as usize], N_B, B);
            k += 1;
        }

        const BIT_DEINTERLEAVE_TABLE: [u8; 16] = [
            0, 0x3, 0xc, 0xf, 0x30, 0x33, 0x3c, 0x3f, 0xc0, 0xc3, 0xcc, 0xcf, 0xf0, 0xf3, 0xfc,
            0xff,
        ];

        k = 0;
        while k < recombine {
            cm = BIT_DEINTERLEAVE_TABLE[cm as usize] as u32;
            haar1(&mut X[..N as usize], N0 >> k, (1) << k);
            k += 1;
        }
        B <<= recombine;
        if let Some(lbo) = lowband_out {
            let n = celt_sqrt(N0 as f32);
            for j in 0..N0 as usize {
                lbo[j] = n * X[j];
            }
        }
        cm &= (((1) << B) - 1) as u32;
    }
    cm
}

/// Upstream C: celt/bands.c:quant_band_stereo
fn quant_band_stereo(
    ctx: &mut band_ctx,
    X: &mut [f32],
    Y: &mut [f32],
    N: i32,
    mut b: i32,
    B: i32,
    lowband: Option<&mut [f32]>,
    LM: i32,
    lowband_out: Option<&mut [f32]>,
    lowband_scratch: Option<&mut [f32]>,
    mut fill: i32,
    ec: &mut ec_ctx,
) -> u32 {
    let mut cm: u32;
    let mut mbits: i32;
    let mut sbits: i32;

    let mut sctx = split_ctx {
        inv: 0,
        imid: 0,
        iside: 0,
        delta: 0,
        itheta: 0,
        qalloc: 0,
        #[cfg(feature = "qext")]
        itheta_q30: 0,
    };
    let orig_fill = fill;
    let encode = ctx.encode;
    if N == 1 {
        return quant_band_n1(ctx, &mut X[..1], Some(&mut Y[..1]), b, lowband_out, ec);
    }
    // When one stereo channel has near-zero energy, copy the other channel's data
    // to avoid numerical issues in the stereo angle computation.
    const MIN_STEREO_ENERGY: f32 = 1e-10f32;
    if encode != 0 {
        let e_left = ctx.bandE[ctx.i as usize];
        let e_right = ctx.bandE[(ctx.m.nbEBands as i32 + ctx.i) as usize];
        if e_left < MIN_STEREO_ENERGY || e_right < MIN_STEREO_ENERGY {
            if e_left > e_right {
                Y[..N as usize].copy_from_slice(&X[..N as usize]);
            } else {
                X[..N as usize].copy_from_slice(&Y[..N as usize]);
            }
        }
    }
    compute_theta(
        ctx,
        &mut sctx,
        &mut X[..N as usize],
        &mut Y[..N as usize],
        N,
        &mut b,
        B,
        B,
        LM,
        1,
        &mut fill,
        ec,
    );
    let inv: i32 = sctx.inv;
    let imid: i32 = sctx.imid;
    let iside: i32 = sctx.iside;
    let delta: i32 = sctx.delta;
    let itheta: i32 = sctx.itheta;
    let qalloc: i32 = sctx.qalloc;
    #[cfg(feature = "qext")]
    let mid: f32;
    #[cfg(feature = "qext")]
    let side: f32;
    #[cfg(not(feature = "qext"))]
    let mid: f32;
    #[cfg(not(feature = "qext"))]
    let side: f32;
    #[cfg(feature = "qext")]
    {
        let _ = (imid, iside);
        mid = celt_cos_norm2(sctx.itheta_q30 as f32 * (1.0f32 / (1i32 << 30) as f32));
        side = celt_cos_norm2(1.0f32 - sctx.itheta_q30 as f32 * (1.0f32 / (1i32 << 30) as f32));
    }
    #[cfg(not(feature = "qext"))]
    {
        mid = 1.0f32 / 32768.0f32 * imid as f32;
        side = 1.0f32 / 32768.0f32 * iside as f32;
    }
    if N == 2 {
        let mut sign: i32 = 0;
        mbits = b;
        sbits = 0;
        if itheta != 0 && itheta != 16384 {
            sbits = (1) << BITRES;
        }
        mbits -= sbits;
        let c = (itheta > 8192) as i32;
        ctx.remaining_bits -= qalloc + sbits;
        // When c != 0, x2=Y,y2=X; otherwise x2=X,y2=Y.
        // We work with (X,Y) directly and swap logic as needed.
        if sbits != 0 {
            if encode != 0 {
                sign = if c != 0 {
                    (Y[0] * X[1] - Y[1] * X[0] < 0.0f32) as i32
                } else {
                    (X[0] * Y[1] - X[1] * Y[0] < 0.0f32) as i32
                };
                ec_enc_bits(ec, sign as u32, 1);
            } else {
                sign = ec_dec_bits(ec, 1) as i32;
            }
        }
        sign = 1 - 2 * sign;
        // quant_band on x2 (the "primary" channel for this theta)
        if c != 0 {
            cm = quant_band(
                ctx,
                Y,
                N,
                mbits,
                B,
                lowband,
                LM,
                lowband_out,
                Q15ONE,
                lowband_scratch,
                orig_fill,
                ec,
            );
            // y2=X: X[0] = -sign * Y[1], X[1] = sign * Y[0]
            X[0] = -sign as f32 * Y[1];
            X[1] = sign as f32 * Y[0];
        } else {
            cm = quant_band(
                ctx,
                X,
                N,
                mbits,
                B,
                lowband,
                LM,
                lowband_out,
                Q15ONE,
                lowband_scratch,
                orig_fill,
                ec,
            );
            // y2=Y: Y[0] = -sign * X[1], Y[1] = sign * X[0]
            Y[0] = -sign as f32 * X[1];
            Y[1] = sign as f32 * X[0];
        }
        if ctx.resynth != 0 {
            X[0] *= mid;
            X[1] *= mid;
            Y[0] *= side;
            Y[1] *= side;
            let tmp0 = X[0];
            X[0] = tmp0 - Y[0];
            Y[0] += tmp0;
            let tmp1 = X[1];
            X[1] = tmp1 - Y[1];
            Y[1] += tmp1;
        }
    } else {
        mbits = if 0
            > (if b < (b - delta) / 2 {
                b
            } else {
                (b - delta) / 2
            }) {
            0
        } else if b < (b - delta) / 2 {
            b
        } else {
            (b - delta) / 2
        };
        sbits = b - mbits;
        ctx.remaining_bits -= qalloc;
        let mut rebalance = ctx.remaining_bits;
        #[cfg(feature = "qext")]
        let saved_ext_b = ctx.ext_b;
        if mbits >= sbits {
            #[cfg(feature = "qext")]
            let qext_extra = {
                let mut qext_extra = 0i32;
                if !ctx.cap.is_null() && saved_ext_b != 0 {
                    let cap_val = unsafe { *ctx.cap.add(ctx.i as usize) };
                    qext_extra = 0.max((saved_ext_b / 2).min(mbits - cap_val / 2));
                }
                qext_extra
            };
            #[cfg(feature = "qext")]
            {
                ctx.ext_b = saved_ext_b / 2 + qext_extra;
            }
            #[cfg(feature = "qext")]
            if std::env::var_os("OPURS_QEXT_TRACE").is_some() && encode == 0 && ctx.i == 20 {
                eprintln!(
                    "[rust stereo] mid call i={} N={} mbits={} sbits={} ext_b={} xh_pre={:016x}",
                    ctx.i,
                    N,
                    mbits,
                    sbits,
                    ctx.ext_b,
                    qext_hash_band(&X[..N as usize]),
                );
            }
            cm = quant_band(
                ctx,
                X,
                N,
                mbits,
                B,
                lowband,
                LM,
                lowband_out,
                Q15ONE,
                lowband_scratch,
                fill,
                ec,
            );
            #[cfg(feature = "qext")]
            if std::env::var_os("OPURS_QEXT_TRACE").is_some() && encode == 0 && ctx.i == 20 {
                eprintln!(
                    "[rust stereo] mid done i={} N={} xh_post={:016x}",
                    ctx.i,
                    N,
                    qext_hash_band(&X[..N as usize]),
                );
            }
            rebalance = mbits - (rebalance - ctx.remaining_bits);
            if rebalance > (3) << BITRES && itheta != 0 {
                sbits += rebalance - ((3) << BITRES);
            }
            #[cfg(feature = "qext")]
            if ctx.extra_bands {
                sbits = sbits.min(ctx.remaining_bits);
            }
            #[cfg(feature = "qext")]
            {
                ctx.ext_b = saved_ext_b / 2 - qext_extra;
            }
            cm |= quant_band(
                ctx,
                Y,
                N,
                sbits,
                B,
                None,
                LM,
                None,
                side,
                None,
                fill >> B,
                ec,
            );
            #[cfg(feature = "qext")]
            if std::env::var_os("OPURS_QEXT_TRACE").is_some() && encode == 0 && ctx.i == 20 {
                eprintln!(
                    "[rust stereo] side done i={} N={} yh_post={:016x}",
                    ctx.i,
                    N,
                    qext_hash_band(&Y[..N as usize]),
                );
            }
        } else {
            #[cfg(feature = "qext")]
            let qext_extra = {
                let mut qext_extra = 0i32;
                if !ctx.cap.is_null() && saved_ext_b != 0 {
                    let cap_val = unsafe { *ctx.cap.add(ctx.i as usize) };
                    qext_extra = 0.max((saved_ext_b / 2).min(sbits - cap_val / 2));
                }
                qext_extra
            };
            #[cfg(feature = "qext")]
            {
                ctx.ext_b = saved_ext_b / 2 + qext_extra;
            }
            cm = quant_band(
                ctx,
                Y,
                N,
                sbits,
                B,
                None,
                LM,
                None,
                side,
                None,
                fill >> B,
                ec,
            );
            rebalance = sbits - (rebalance - ctx.remaining_bits);
            if rebalance > (3) << BITRES && itheta != 16384 {
                mbits += rebalance - ((3) << BITRES);
            }
            #[cfg(feature = "qext")]
            {
                if ctx.extra_bands {
                    mbits = mbits.min(ctx.remaining_bits);
                }
                ctx.ext_b = saved_ext_b / 2 - qext_extra;
            }
            cm |= quant_band(
                ctx,
                X,
                N,
                mbits,
                B,
                lowband,
                LM,
                lowband_out,
                Q15ONE,
                lowband_scratch,
                fill,
                ec,
            );
        }
    }
    if ctx.resynth != 0 {
        #[cfg(feature = "qext")]
        if std::env::var_os("OPURS_QEXT_TRACE").is_some() && encode == 0 && ctx.i == 20 {
            eprintln!(
                "[rust stereo] pre i={} N={} itheta={} mid={:.9} inv={} xh={:016x} yh={:016x}",
                ctx.i,
                N,
                itheta,
                mid,
                inv,
                qext_hash_band(&X[..N as usize]),
                qext_hash_band(&Y[..N as usize])
            );
        }
        if N != 2 {
            stereo_merge(&mut X[..N as usize], &mut Y[..N as usize], mid, N, ctx.arch);
        }
        if inv != 0 {
            for y in Y[..N as usize].iter_mut() {
                *y = -*y;
            }
        }
        #[cfg(feature = "qext")]
        if std::env::var_os("OPURS_QEXT_TRACE").is_some() && encode == 0 && ctx.i == 20 {
            eprintln!(
                "[rust stereo] post i={} N={} itheta={} mid={:.9} inv={} xh={:016x} yh={:016x}",
                ctx.i,
                N,
                itheta,
                mid,
                inv,
                qext_hash_band(&X[..N as usize]),
                qext_hash_band(&Y[..N as usize])
            );
        }
    }
    cm
}

/// Upstream C: celt/bands.c:special_hybrid_folding
#[inline]
fn special_hybrid_folding(
    m: &OpusCustomMode,
    norm: &mut [f32],
    norm2: &mut [f32],
    start: i32,
    M: i32,
    dual_stereo: i32,
) {
    let eBands = &m.eBands;
    let n1 = (M * (eBands[(start + 1) as usize] as i32 - eBands[start as usize] as i32)) as usize;
    let n2 =
        (M * (eBands[(start + 2) as usize] as i32 - eBands[(start + 1) as usize] as i32)) as usize;
    norm.copy_within(2 * n1 - n2..n1, n1);
    if dual_stereo != 0 {
        norm2.copy_within(2 * n1 - n2..n1, n1);
    }
}

/// Upstream C: celt/bands.c:quant_all_bands
#[inline]
pub fn quant_all_bands<'a>(
    encode: i32,
    m: &'a OpusCustomMode,
    start: i32,
    end: i32,
    X_: &mut [f32],
    Y_: Option<&mut [f32]>,
    collapse_masks: &mut [u8],
    bandE: &'a [f32],
    pulses: &mut [i32],
    shortBlocks: i32,
    spread: i32,
    mut dual_stereo: i32,
    intensity: i32,
    tf_res: &mut [i32],
    total_bits: i32,
    mut balance: i32,
    ec: &mut ec_ctx,
    LM: i32,
    codedBands: i32,
    seed: &mut u32,
    complexity: i32,
    arch: Arch,
    disable_inv: i32,
    #[cfg(feature = "qext")] ext_ec: &mut ec_ctx<'a>,
    #[cfg(feature = "qext")] extra_pulses: &[i32],
    #[cfg(feature = "qext")] ext_total_bits: i32,
    #[cfg(feature = "qext")] cap: &[i32],
) {
    let mut remaining_bits: i32;
    let eBands = &m.eBands;
    debug_assert!(
        end <= m.nbEBands as i32,
        "quant_all_bands: end {} > nbEBands {}",
        end,
        m.nbEBands
    );

    let M: i32 = (1) << LM;
    let mut lowband_offset: i32 = 0;
    let mut update_lowband: i32 = 1;
    let C: i32 = if Y_.is_some() { 2 } else { 1 };
    let norm_offset: i32 = M * eBands[start as usize] as i32;
    #[allow(unused_mut)]
    let mut theta_rdo: i32 =
        (encode != 0 && Y_.is_some() && dual_stereo == 0 && complexity >= 8) as i32;
    #[cfg(feature = "qext")]
    let extra_bands = end == crate::celt::modes::data_96000::NB_QEXT_BANDS as i32 || end == 2;
    #[cfg(feature = "qext")]
    if extra_bands {
        theta_rdo = 0;
    }
    let resynth: i32 = (encode == 0 || theta_rdo != 0) as i32;
    let B: i32 = if shortBlocks != 0 { M } else { 1 };
    let norm_size = (M * eBands[m.nbEBands - 1] as i32 - norm_offset) as usize;
    // C * norm_size max: 2 * M * eBands[last]. Stereo QEXT 96kHz = 2 * 8 * 240 = 3840.
    const MAX_NORM: usize = 6000;
    debug_assert!(C as usize * norm_size <= MAX_NORM);
    let mut _norm = [0.0f32; MAX_NORM];

    let _resynth_alloc: i32 = if encode != 0 && resynth != 0 {
        M * (eBands[m.nbEBands] as i32 - eBands[m.nbEBands - 1] as i32)
    } else {
        0
    };
    let mut _lowband_scratch = [0.0f32; 176];
    let mut _X_save = [0.0f32; 176];
    let mut _Y_save = [0.0f32; 176];
    let mut _X_save2 = [0.0f32; 176];
    let mut _Y_save2 = [0.0f32; 176];
    let mut _norm_save2 = [0.0f32; 176];

    // In decode-only mode, lowband_scratch comes from the end of X_
    let decode_scratch_off = (M * eBands[m.effEBands as usize - 1] as i32) as usize;
    let use_alloc_scratch = encode != 0 && resynth != 0;

    let has_y = Y_.is_some();
    // Y_ kept as Option<&mut [f32]> — reborrowed per-band in the loop.
    let mut y_mut = Y_;

    #[cfg(feature = "qext")]
    let mut ext_balance: i32 = 0;
    #[cfg(feature = "qext")]
    let mut ext_tell: i32 = 0;

    let mut ctx = band_ctx {
        encode,
        resynth,
        m,
        i: 0,
        intensity,
        spread,
        tf_change: 0,
        remaining_bits: 0,
        bandE,
        seed: *seed,
        arch,
        disable_inv,
        theta_round: 0,
        avoid_split_noise: (B > 1) as i32,
        #[cfg(feature = "qext")]
        ext_ec: ext_ec as *mut ec_ctx,
        #[cfg(feature = "qext")]
        extra_bits: 0,
        #[cfg(feature = "qext")]
        ext_total_bits,
        #[cfg(feature = "qext")]
        extra_bands,
        #[cfg(feature = "qext")]
        ext_b: 0,
        #[cfg(feature = "qext")]
        cap: if cap.is_empty() {
            std::ptr::null()
        } else {
            cap.as_ptr()
        },
        #[cfg(feature = "qext")]
        cap_len: cap.len() as i32,
    };
    #[cfg(feature = "qext")]
    let qext_band_trace = std::env::var_os("OPURS_QEXT_TRACE").is_some();

    let mut i = start;
    while i < end {
        let mut effective_lowband: i32 = -1;

        let mut x_cm: u32;
        let mut y_cm: u32;
        ctx.i = i;
        let last = (i == end - 1) as i32;
        let band_start = (M * eBands[i as usize] as i32) as usize;
        let N: i32 = M * eBands[(i + 1) as usize] as i32 - M * eBands[i as usize] as i32;
        debug_assert!(N > 0);
        let n = N as usize;
        let tell = ec_tell_frac(ec) as i32;
        if i != start {
            balance -= tell;
        }
        remaining_bits = total_bits - tell - 1;
        ctx.remaining_bits = remaining_bits;
        let b: i32 = if i < codedBands {
            let curr_balance = celt_sudiv(
                balance,
                if 3 < codedBands - i {
                    3
                } else {
                    codedBands - i
                },
            );
            if 0 > (if 16383
                < (if (remaining_bits + 1) < pulses[i as usize] + curr_balance {
                    remaining_bits + 1
                } else {
                    pulses[i as usize] + curr_balance
                }) {
                16383
            } else if (remaining_bits + 1) < pulses[i as usize] + curr_balance {
                remaining_bits + 1
            } else {
                pulses[i as usize] + curr_balance
            }) {
                0
            } else if 16383
                < (if (remaining_bits + 1) < pulses[i as usize] + curr_balance {
                    remaining_bits + 1
                } else {
                    pulses[i as usize] + curr_balance
                })
            {
                16383
            } else if (remaining_bits + 1) < pulses[i as usize] + curr_balance {
                remaining_bits + 1
            } else {
                pulses[i as usize] + curr_balance
            }
        } else {
            0
        };
        // QEXT: per-band extension bit allocation
        #[cfg(feature = "qext")]
        let ext_b: i32;
        #[cfg(feature = "qext")]
        {
            if i != start {
                ext_balance += extra_pulses[i as usize - 1] + ext_tell;
            }
            ext_tell = ec_tell_frac(ext_ec) as i32;
            ctx.extra_bits = extra_pulses[i as usize];
            if i != start {
                ext_balance -= ext_tell;
            }
            if i < codedBands {
                let ext_curr_balance = celt_sudiv(ext_balance, 3.min(codedBands - i));
                ext_b = 0.max(16383.min(
                    (ext_total_bits - ext_tell).min(extra_pulses[i as usize] + ext_curr_balance),
                ));
            } else {
                ext_b = 0;
            }
            ctx.ext_b = ext_b;
            if qext_band_trace {
                eprintln!(
                    "[rust qext bands] pre i={} b={} ext_b={} ec_tell={} ext_tell={} rem={} tf={}",
                    i,
                    b,
                    ext_b,
                    ec_tell_frac(ec),
                    ec_tell_frac(ext_ec),
                    ctx.remaining_bits,
                    tf_res[i as usize]
                );
            }
        }
        if resynth != 0
            && (M * eBands[i as usize] as i32 - N >= M * eBands[start as usize] as i32
                || i == start + 1)
            && (update_lowband != 0 || lowband_offset == 0)
        {
            lowband_offset = i;
        }
        if i == start + 1 {
            let (norm_part, norm2_part) = _norm.split_at_mut(norm_size);
            special_hybrid_folding(m, norm_part, norm2_part, start, M, dual_stereo);
        }
        let tf_change: i32 = tf_res[i as usize];
        ctx.tf_change = tf_change;

        // For bands beyond effEBands, use norm buffer as dummy X/Y
        let use_norm_xy = i >= m.effEBands;
        let have_scratch = !use_norm_xy && (last == 0 || theta_rdo != 0);

        if lowband_offset != 0 && (spread != SPREAD_AGGRESSIVE || B > 1 || tf_change < 0) {
            let mut fold_start: i32;
            let mut fold_end: i32;
            effective_lowband = if 0 > M * eBands[lowband_offset as usize] as i32 - norm_offset - N
            {
                0
            } else {
                M * eBands[lowband_offset as usize] as i32 - norm_offset - N
            };
            fold_start = lowband_offset;
            loop {
                fold_start -= 1;
                if M * eBands[fold_start as usize] as i32 <= effective_lowband + norm_offset {
                    break;
                }
            }
            fold_end = lowband_offset - 1;
            loop {
                fold_end += 1;
                if !(fold_end < i
                    && (M * eBands[fold_end as usize] as i32) < effective_lowband + norm_offset + N)
                {
                    break;
                }
            }
            y_cm = 0;
            x_cm = y_cm;
            let mut fold_i = fold_start;
            loop {
                x_cm |= collapse_masks[(fold_i * C) as usize] as u32;
                y_cm |= collapse_masks[(fold_i * C + C - 1) as usize] as u32;
                fold_i += 1;
                if fold_i >= fold_end {
                    break;
                }
            }
        } else {
            y_cm = (((1) << B) - 1) as u32;
            x_cm = y_cm;
        }
        if dual_stereo != 0 && i == intensity {
            dual_stereo = 0;
            if resynth != 0 {
                let mut j = 0;
                while j < M * eBands[i as usize] as i32 - norm_offset {
                    let ju = j as usize;
                    _norm[ju] = 0.5f32 * (_norm[ju] + _norm[norm_size + ju]);
                    j += 1;
                }
            }
        }

        // Helper: get lowband slice from _norm (read-only)
        let norm_band_out_off = (M * eBands[i as usize] as i32 - norm_offset) as usize;

        // When scratch comes from X_ tail (decode-only mode), split X_ at decode_scratch_off
        // to get non-overlapping x_band and scratch. band_start + n <= decode_scratch_off
        // holds because have_scratch is only true when i < effEBands.
        let need_x_scratch = have_scratch && !use_alloc_scratch;
        let (x_band_src, mut x_scratch_src) = if need_x_scratch {
            let (coded, scratch) = X_.split_at_mut(decode_scratch_off);
            (coded, Some(scratch))
        } else {
            (&mut *X_, None)
        };

        let scratch: Option<&mut [f32]> = if have_scratch {
            if use_alloc_scratch {
                Some(&mut _lowband_scratch)
            } else {
                Some(&mut x_scratch_src.as_mut().unwrap()[..n])
            }
        } else {
            None
        };

        if dual_stereo != 0 {
            let (norm1, norm2) = _norm.split_at_mut(norm_size);
            // Copy lowband data to a temp buffer so we can give lowband_out a &mut into _norm.
            // The lowband read range [effective_lowband..effective_lowband+N] may overlap with
            // the lowband_out write range [norm_band_out_off..], so we can't split_at_mut.
            let mut lowband_x_buf = [0.0f32; 176];
            if effective_lowband != -1 {
                let lb_start = effective_lowband as usize;
                lowband_x_buf[..n].copy_from_slice(&norm1[lb_start..lb_start + n]);
            }
            let lowband_out_x: Option<&mut [f32]> = if last != 0 {
                None
            } else {
                Some(&mut norm1[norm_band_out_off..])
            };
            let x_band = &mut x_band_src[band_start..band_start + n];
            #[cfg(feature = "qext")]
            {
                ctx.ext_b = ext_b / 2;
            }
            x_cm = quant_band(
                &mut ctx,
                x_band,
                N,
                b / 2,
                B,
                if effective_lowband != -1 {
                    Some(&mut lowband_x_buf[..n])
                } else {
                    None
                },
                LM,
                lowband_out_x,
                Q15ONE,
                scratch,
                x_cm as i32,
                ec,
            );
            // Same lowband copy approach for channel 2.
            let mut lowband_y_buf = [0.0f32; 176];
            if effective_lowband != -1 {
                let lb_start = effective_lowband as usize;
                lowband_y_buf[..n].copy_from_slice(&norm2[lb_start..lb_start + n]);
            }
            let lowband_out_y: Option<&mut [f32]> = if last != 0 {
                None
            } else {
                Some(&mut norm2[norm_band_out_off..])
            };
            let scratch2: Option<&mut [f32]> = if have_scratch {
                if use_alloc_scratch {
                    Some(&mut _lowband_scratch)
                } else {
                    Some(&mut x_scratch_src.as_mut().unwrap()[..n])
                }
            } else {
                None
            };
            let y_band = &mut y_mut.as_deref_mut().unwrap()[band_start..band_start + n];
            #[cfg(feature = "qext")]
            {
                ctx.ext_b = ext_b / 2;
            }
            y_cm = quant_band(
                &mut ctx,
                y_band,
                N,
                b / 2,
                B,
                if effective_lowband != -1 {
                    Some(&mut lowband_y_buf[..n])
                } else {
                    None
                },
                LM,
                lowband_out_y,
                Q15ONE,
                scratch2,
                y_cm as i32,
                ec,
            );
        } else if use_norm_xy {
            // Beyond effEBands: use norm buffer as dummy X/Y, no lowband/scratch needed.
            #[cfg(feature = "qext")]
            {
                ctx.ext_b = 0;
            }
            if has_y {
                let (dummy_x, dummy_rest) = _norm.split_at_mut(n);
                let dummy_y = &mut dummy_rest[..n];
                x_cm = quant_band_stereo(
                    &mut ctx,
                    dummy_x,
                    dummy_y,
                    N,
                    b,
                    B,
                    None,
                    LM,
                    None,
                    None,
                    (x_cm | y_cm) as i32,
                    ec,
                );
            } else {
                let dummy = &mut _norm[..n];
                x_cm = quant_band(
                    &mut ctx,
                    dummy,
                    N,
                    b,
                    B,
                    None,
                    LM,
                    None,
                    Q15ONE,
                    None,
                    (x_cm | y_cm) as i32,
                    ec,
                );
            }
            y_cm = x_cm;
        } else {
            // Copy lowband data to a temp buffer so lowband_out can borrow _norm mutably.
            // The lowband read range may overlap with the lowband_out write range.
            let mut lowband_buf = [0.0f32; 176];
            if effective_lowband != -1 {
                let lb_start = effective_lowband as usize;
                lowband_buf[..n].copy_from_slice(&_norm[lb_start..lb_start + n]);
            }
            let lowband_out_ref: Option<&mut [f32]> = if last != 0 {
                None
            } else {
                Some(&mut _norm[norm_band_out_off..])
            };
            if has_y {
                let x_band = &mut x_band_src[band_start..band_start + n];
                let y_band = &mut y_mut.as_deref_mut().unwrap()[band_start..band_start + n];
                if theta_rdo != 0 && i < intensity {
                    let mut w: [f32; 2] = [0.0; 2];
                    compute_channel_weights(
                        bandE[i as usize],
                        bandE[(i + m.nbEBands as i32) as usize],
                        &mut w,
                    );
                    let cm: u32 = x_cm | y_cm;
                    let ec_save = ec.save();
                    let ctx_save: band_ctx = ctx;
                    _X_save[..n].copy_from_slice(&x_band[..n]);
                    _Y_save[..n].copy_from_slice(&y_band[..n]);
                    // Try theta_round = -1
                    ctx.theta_round = -1;
                    let (lowband_ref, lowband_out_ref_theta): (
                        Option<&mut [f32]>,
                        Option<&mut [f32]>,
                    ) = if last != 0 {
                        (
                            if effective_lowband != -1 {
                                Some(&mut _norm[effective_lowband as usize..])
                            } else {
                                None
                            },
                            None,
                        )
                    } else {
                        debug_assert!(
                            effective_lowband == -1
                                || (effective_lowband as usize + n <= norm_band_out_off)
                        );
                        let (norm_low, norm_out) = _norm.split_at_mut(norm_band_out_off);
                        (
                            if effective_lowband != -1 {
                                Some(&mut norm_low[effective_lowband as usize..])
                            } else {
                                None
                            },
                            Some(norm_out),
                        )
                    };
                    x_cm = quant_band_stereo(
                        &mut ctx,
                        x_band,
                        y_band,
                        N,
                        b,
                        B,
                        lowband_ref,
                        LM,
                        lowband_out_ref_theta,
                        scratch,
                        cm as i32,
                        ec,
                    );
                    let dist0: f32 = w[0] * celt_inner_prod(&_X_save[..n], &x_band[..n], n, arch)
                        + w[1] * celt_inner_prod(&_Y_save[..n], &y_band[..n], n, arch);
                    let cm2: u32 = x_cm;
                    let ec_save2 = ec.save();
                    let ctx_save2: band_ctx = ctx;
                    _X_save2[..n].copy_from_slice(&x_band[..n]);
                    _Y_save2[..n].copy_from_slice(&y_band[..n]);
                    if last == 0 {
                        _norm_save2[..n]
                            .copy_from_slice(&_norm[norm_band_out_off..norm_band_out_off + n]);
                    }
                    let nstart_bytes = ec_save.offs as usize;
                    let nend_bytes = ec.storage as usize;
                    let save_bytes = nend_bytes - nstart_bytes;
                    let mut bytes_save = vec![0u8; save_bytes];
                    bytes_save.copy_from_slice(&ec.buf[nstart_bytes..nend_bytes]);
                    // Restore state for round +1
                    ec.restore(ec_save);
                    ctx = ctx_save;
                    x_band[..n].copy_from_slice(&_X_save[..n]);
                    y_band[..n].copy_from_slice(&_Y_save[..n]);
                    if i == start + 1 {
                        let (norm_part, norm2_part) = _norm.split_at_mut(norm_size);
                        special_hybrid_folding(m, norm_part, norm2_part, start, M, dual_stereo);
                    }
                    let (lowband_ref2, lowband_out_ref2): (Option<&mut [f32]>, Option<&mut [f32]>) =
                        if last != 0 {
                            (
                                if effective_lowband != -1 {
                                    Some(&mut _norm[effective_lowband as usize..])
                                } else {
                                    None
                                },
                                None,
                            )
                        } else {
                            debug_assert!(
                                effective_lowband == -1
                                    || (effective_lowband as usize + n <= norm_band_out_off)
                            );
                            let (norm_low, norm_out) = _norm.split_at_mut(norm_band_out_off);
                            (
                                if effective_lowband != -1 {
                                    Some(&mut norm_low[effective_lowband as usize..])
                                } else {
                                    None
                                },
                                Some(norm_out),
                            )
                        };
                    let scratch2: Option<&mut [f32]> = if have_scratch {
                        if use_alloc_scratch {
                            Some(&mut _lowband_scratch)
                        } else {
                            Some(&mut x_scratch_src.as_mut().unwrap()[..n])
                        }
                    } else {
                        None
                    };
                    ctx.theta_round = 1;
                    #[cfg(feature = "qext")]
                    {
                        ctx.ext_b = ext_b;
                    }
                    x_cm = quant_band_stereo(
                        &mut ctx,
                        x_band,
                        y_band,
                        N,
                        b,
                        B,
                        lowband_ref2,
                        LM,
                        lowband_out_ref2,
                        scratch2,
                        cm as i32,
                        ec,
                    );
                    let dist1: f32 = w[0] * celt_inner_prod(&_X_save[..n], &x_band[..n], n, arch)
                        + w[1] * celt_inner_prod(&_Y_save[..n], &y_band[..n], n, arch);
                    if dist0 >= dist1 {
                        x_cm = cm2;
                        ec.restore(ec_save2);
                        ctx = ctx_save2;
                        x_band[..n].copy_from_slice(&_X_save2[..n]);
                        y_band[..n].copy_from_slice(&_Y_save2[..n]);
                        if last == 0 {
                            _norm[norm_band_out_off..norm_band_out_off + n]
                                .copy_from_slice(&_norm_save2[..n]);
                        }
                        ec.buf[nstart_bytes..nend_bytes].copy_from_slice(&bytes_save);
                    }
                } else {
                    ctx.theta_round = 0;
                    x_cm = quant_band_stereo(
                        &mut ctx,
                        x_band,
                        y_band,
                        N,
                        b,
                        B,
                        if effective_lowband != -1 {
                            Some(&mut lowband_buf[..n])
                        } else {
                            None
                        },
                        LM,
                        lowband_out_ref,
                        scratch,
                        (x_cm | y_cm) as i32,
                        ec,
                    );
                }
                y_cm = x_cm;
            } else {
                // Mono mode (use_norm_xy already handled above)
                let x_band = &mut x_band_src[band_start..band_start + n];
                x_cm = quant_band(
                    &mut ctx,
                    x_band,
                    N,
                    b,
                    B,
                    if effective_lowband != -1 {
                        Some(&mut lowband_buf[..n])
                    } else {
                        None
                    },
                    LM,
                    lowband_out_ref,
                    Q15ONE,
                    scratch,
                    (x_cm | y_cm) as i32,
                    ec,
                );
                y_cm = x_cm;
            }
        }
        collapse_masks[(i * C) as usize] = x_cm as u8;
        collapse_masks[(i * C + C - 1) as usize] = y_cm as u8;
        #[cfg(feature = "qext")]
        if qext_band_trace {
            let xh = if use_norm_xy {
                qext_hash_band(&_norm[..n])
            } else {
                qext_hash_band(&x_band_src[band_start..band_start + n])
            };
            let yh = if has_y {
                if use_norm_xy {
                    qext_hash_band(&_norm[n..2 * n])
                } else {
                    qext_hash_band(&y_mut.as_deref().unwrap()[band_start..band_start + n])
                }
            } else {
                0
            };
            eprintln!(
                "[rust qext bands] post i={} ec_tell={} ext_tell={} x_cm={} y_cm={} seed={} xh={:016x} yh={:016x}",
                i,
                ec_tell_frac(ec),
                ec_tell_frac(ext_ec),
                x_cm,
                y_cm,
                ctx.seed,
                xh,
                yh
            );
        }
        balance += pulses[i as usize] + tell;
        update_lowband = (b > N << BITRES) as i32;
        ctx.avoid_split_noise = 0;
        i += 1;
    }
    *seed = ctx.seed;
}
