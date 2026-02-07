use crate::celt::entcode::{celt_sudiv, celt_udiv, ec_ctx, ec_tell_frac, BITRES};
use crate::celt::entdec::{ec_dec_bit_logp, ec_dec_bits, ec_dec_uint, ec_dec_update, ec_decode};
use crate::celt::entenc::{ec_enc_bit_logp, ec_enc_bits, ec_enc_uint, ec_encode};
use crate::celt::mathops::{celt_exp2, celt_rsqrt, celt_rsqrt_norm, celt_sqrt, isqrt32};
use crate::celt::modes::OpusCustomMode;
use crate::celt::pitch::{celt_inner_prod, dual_inner_prod};
use crate::celt::quant_bands::eMeans;
use crate::celt::rate::{
    bits2pulses, get_pulses, pulses2bits, QTHETA_OFFSET, QTHETA_OFFSET_TWOPHASE,
};
use crate::celt::vq::{alg_quant, alg_unquant, renormalise_vector, stereo_itheta};
use crate::silk::macros::EC_CLZ0;

const EPSILON: f32 = 1e-15f32;
const Q15ONE: f32 = 1.0f32;
const NORM_SCALING: f32 = 1.0f32;

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
    arch: i32,
    theta_round: i32,
    disable_inv: i32,
    avoid_split_noise: i32,
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
pub fn celt_lcg_rand(seed: u32) -> u32 {
    (1664525_u32).wrapping_mul(seed).wrapping_add(1013904223)
}

/// Upstream C: celt/bands.c:bitexact_cos
pub fn bitexact_cos(x: i16) -> i16 {
    let tmp = 4096 + x as i32 * x as i32 >> 13;
    let x2 = tmp as i16;
    let x2 = (32767 - x2 as i32
        + (16384
            + x2 as i32
                * (-(7651)
                    + (16384
                        + x2 as i32
                            * (8277 + (16384 + -(626) as i16 as i32 * x2 as i32 >> 15)) as i16
                                as i32
                        >> 15)) as i16 as i32
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
        + (16384
            + isin as i16 as i32
                * ((16384 + isin as i16 as i32 * -(2597) as i16 as i32 >> 15) + 7932) as i16 as i32
            >> 15)
        - (16384
            + icos as i16 as i32
                * ((16384 + icos as i16 as i32 * -(2597) as i16 as i32 >> 15) + 7932) as i16 as i32
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
    _arch: i32,
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
                );
            bandE[(i + c * m.nbEBands as i32) as usize] = celt_sqrt(sum);
            i += 1;
        }
        c += 1;
        if !(c < C) {
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
            let mut j = M * eBands[i as usize] as i32;
            while j < M * eBands[(i + 1) as usize] as i32 {
                X[(j + c * N) as usize] = freq[(j + c * N) as usize] * g;
                j += 1;
            }
            i += 1;
        }
        c += 1;
        if !(c < C) {
            break;
        }
    }
}

/// Upstream C: celt/bands.c:denormalise_bands
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
    let mut f_idx = 0usize;
    let mut x_idx = (M * eBands[start as usize] as i32) as usize;
    for _ in 0..M * eBands[start as usize] as i32 {
        freq[f_idx] = 0.0;
        f_idx += 1;
    }
    let mut i = start;
    while i < end {
        let band_end = M * eBands[(i + 1) as usize] as i32;
        let lg = bandLogE[i as usize] + eMeans[i as usize];
        let g = celt_exp2(if 32.0 < lg { 32.0f32 } else { lg });
        let mut j = M * eBands[i as usize] as i32;
        loop {
            freq[f_idx] = X[x_idx] * g;
            f_idx += 1;
            x_idx += 1;
            j += 1;
            if !(j < band_end) {
                break;
            }
        }
        i += 1;
    }
    assert!(start <= end);
    freq[bound as usize..N as usize].fill(0.0);
}

/// Upstream C: celt/bands.c:anti_collapse
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
    arch: i32,
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
            if C == 1 {
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
                r *= 1.41421356f32;
            }
            r = if thresh < r { thresh } else { r };
            r = r * sqrt_1;
            let x_off = (c * size + ((m.eBands[i as usize] as i32) << LM)) as usize;
            let x_len = (N0 << LM) as usize;
            let mut renormalize = 0;
            let mut k = 0;
            while k < (1) << LM {
                if collapse_masks[(i * C + c) as usize] as i32 & (1) << k == 0 {
                    let mut j = 0;
                    while j < N0 {
                        seed = celt_lcg_rand(seed);
                        X_[x_off + ((j << LM) + k) as usize] =
                            if seed & 0x8000 != 0 { r } else { -r };
                        j += 1;
                    }
                    renormalize = 1;
                }
                k += 1;
            }
            if renormalize != 0 {
                renormalise_vector(&mut X_[x_off..x_off + x_len], N0 << LM, Q15ONE, arch);
            }
            c += 1;
            if !(c < C) {
                break;
            }
        }
        i += 1;
    }
}

/// Upstream C: celt/bands.c:compute_channel_weights
fn compute_channel_weights(mut Ex: f32, mut Ey: f32, w: &mut [f32]) {
    let minE = if Ex < Ey { Ex } else { Ey };
    Ex = Ex + minE / 3.0f32;
    Ey = Ey + minE / 3.0f32;
    w[0] = Ex;
    w[1] = Ey;
}

/// Upstream C: celt/bands.c:intensity_stereo
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
    let mut j = 0;
    while j < N {
        let l = X[j as usize];
        let r = Y[j as usize];
        X[j as usize] = a1 * l + a2 * r;
        j += 1;
    }
}

/// Upstream C: celt/bands.c:stereo_split
fn stereo_split(X: &mut [f32], Y: &mut [f32], N: i32) {
    let mut j = 0;
    while j < N {
        let l = 0.70710678f32 * X[j as usize];
        let r = 0.70710678f32 * Y[j as usize];
        X[j as usize] = l + r;
        Y[j as usize] = r - l;
        j += 1;
    }
}

/// Upstream C: celt/bands.c:stereo_merge
fn stereo_merge(X: &mut [f32], Y: &mut [f32], mid: f32, N: i32, _arch: i32) {
    let n = N as usize;
    let (xp, side) = dual_inner_prod(&Y[..n], &X[..n], &Y[..n], n);
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
    let mut j = 0;
    while j < N {
        let l = mid * X[j as usize];
        let r = Y[j as usize];
        X[j as usize] = lgain * (l - r);
        Y[j as usize] = rgain * (l + r);
        j += 1;
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
    assert!(end > 0);
    let N0 = M * m.shortMdctSize;
    if M * (eBands[end as usize] as i32 - eBands[(end - 1) as usize] as i32) <= 8 {
        return SPREAD_NONE;
    }
    let mut c = 0;
    loop {
        let mut i = 0;
        while i < end {
            let N = M * (eBands[(i + 1) as usize] as i32 - eBands[i as usize] as i32);
            if !(N <= 8) {
                let x_off = (M * eBands[i as usize] as i32 + c * N0) as usize;
                let mut tcount: [i32; 3] = [0, 0, 0];
                let mut j = 0;
                while j < N {
                    let x2N = X[x_off + j as usize] * X[x_off + j as usize] * N as f32;
                    if x2N < 0.25f32 {
                        tcount[0] += 1;
                    }
                    if x2N < 0.0625f32 {
                        tcount[1] += 1;
                    }
                    if x2N < 0.015625f32 {
                        tcount[2] += 1;
                    }
                    j += 1;
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
        if !(c < C) {
            break;
        }
    }
    if update_hf != 0 {
        if hf_sum != 0 {
            hf_sum = celt_udiv(hf_sum as u32, (C * (4 - m.nbEBands as i32 + end)) as u32) as i32;
        }
        *hf_average = *hf_average + hf_sum >> 1;
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
    assert!(nbBands > 0);
    assert!(sum >= 0);
    sum = celt_udiv((sum << 8) as u32, nbBands as u32) as i32;
    sum = sum + *average >> 1;
    *average = sum;
    sum = 3 * sum + ((3 - last_decision << 7) + 64) + 2 >> 2;
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
fn deinterleave_hadamard(X: &mut [f32], N0: i32, stride: i32, hadamard: i32) {
    let N = (N0 * stride) as usize;
    let mut tmp = vec![0.0f32; N];
    assert!(stride > 0);
    if hadamard != 0 {
        let ordery = &ordery_table[(stride - 2) as usize..];
        let mut i = 0;
        while i < stride {
            let mut j = 0;
            while j < N0 {
                tmp[(ordery[i as usize] * N0 + j) as usize] = X[(j * stride + i) as usize];
                j += 1;
            }
            i += 1;
        }
    } else {
        let mut i = 0;
        while i < stride {
            let mut j = 0;
            while j < N0 {
                tmp[(i * N0 + j) as usize] = X[(j * stride + i) as usize];
                j += 1;
            }
            i += 1;
        }
    }
    X[..N].copy_from_slice(&tmp[..N]);
}

/// Upstream C: celt/bands.c:interleave_hadamard
fn interleave_hadamard(X: &mut [f32], N0: i32, stride: i32, hadamard: i32) {
    let N = (N0 * stride) as usize;
    let mut tmp = vec![0.0f32; N];
    if hadamard != 0 {
        let ordery = &ordery_table[(stride - 2) as usize..];
        let mut i = 0;
        while i < stride {
            let mut j = 0;
            while j < N0 {
                tmp[(j * stride + i) as usize] = X[(ordery[i as usize] * N0 + j) as usize];
                j += 1;
            }
            i += 1;
        }
    } else {
        let mut i = 0;
        while i < stride {
            let mut j = 0;
            while j < N0 {
                tmp[(j * stride + i) as usize] = X[(i * N0 + j) as usize];
                j += 1;
            }
            i += 1;
        }
    }
    X[..N].copy_from_slice(&tmp[..N]);
}

/// Upstream C: celt/bands.c:haar1
pub fn haar1(X: &mut [f32], mut N0: i32, stride: i32) {
    N0 >>= 1;
    let mut i = 0;
    while i < stride {
        let mut j = 0;
        while j < N0 {
            let tmp1 = std::f32::consts::FRAC_1_SQRT_2 * X[(stride * 2 * j + i) as usize];
            let tmp2 = std::f32::consts::FRAC_1_SQRT_2 * X[(stride * (2 * j + 1) + i) as usize];
            X[(stride * 2 * j + i) as usize] = tmp1 + tmp2;
            X[(stride * (2 * j + 1) + i) as usize] = tmp1 - tmp2;
            j += 1;
        }
        i += 1;
    }
}

/// Upstream C: celt/bands.c:compute_qn
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
    let qn;
    if qb < (1) << BITRES >> 1 {
        qn = 1;
    } else {
        let raw = EXP2_TABLE8[(qb & 0x7) as usize] as i32 >> 14 - (qb >> BITRES);
        qn = (raw + 1 >> 1) << 1;
    }
    assert!(qn <= 256);
    qn
}

/// Upstream C: celt/bands.c:compute_theta
///
/// Uses raw pointers internally for X/Y because the caller (`quant_partition`)
/// needs to split X at varying offsets and pass sub-slices. The pointer
/// arithmetic is confined to this function.
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
        itheta = stereo_itheta(&X[..N as usize], &Y[..N as usize], stereo, N, ctx.arch);
    }
    let tell = ec_tell_frac(ec) as i32;
    if qn != 1 {
        if encode != 0 {
            if stereo == 0 || ctx.theta_round == 0 {
                itheta = itheta * qn + 8192 >> 14;
                if stereo == 0 && ctx.avoid_split_noise != 0 && itheta > 0 && itheta < qn {
                    let unquantized = celt_udiv((itheta * 16384) as u32, qn as u32) as i32;
                    imid = bitexact_cos(unquantized as i16) as i32;
                    iside = bitexact_cos((16384 - unquantized) as i16) as i32;
                    let delta = 16384
                        + ((N - 1) << 7) as i16 as i32
                            * bitexact_log2tan(iside, imid) as i16 as i32
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
                    < (if 0 > itheta * qn + bias >> 14 {
                        0
                    } else {
                        itheta * qn + bias >> 14
                    }) {
                    qn - 1
                } else if 0 > itheta * qn + bias >> 14 {
                    0
                } else {
                    itheta * qn + bias >> 14
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
                    itheta * (itheta + 1) >> 1
                } else {
                    ft_0 - ((qn + 1 - itheta) * (qn + 2 - itheta) >> 1)
                };
                ec_encode(ec, fl as u32, (fl + fs_0) as u32, ft_0 as u32);
            } else {
                let fm = ec_decode(ec, ft_0 as u32) as i32;
                if fm < (qn >> 1) * ((qn >> 1) + 1) >> 1 {
                    itheta = ((isqrt32((8_u32).wrapping_mul(fm as u32).wrapping_add(1)))
                        .wrapping_sub(1)
                        >> 1) as i32;
                    fs_0 = itheta + 1;
                    let fl_0 = itheta * (itheta + 1) >> 1;
                    ec_dec_update(ec, fl_0 as u32, (fl_0 + fs_0) as u32, ft_0 as u32);
                } else {
                    itheta = (((2 * (qn + 1)) as u32).wrapping_sub(isqrt32(
                        (8_u32).wrapping_mul((ft_0 - fm - 1) as u32).wrapping_add(1),
                    )) >> 1) as i32;
                    fs_0 = qn + 1 - itheta;
                    let fl_0 = ft_0 - ((qn + 1 - itheta) * (qn + 2 - itheta) >> 1);
                    ec_dec_update(ec, fl_0 as u32, (fl_0 + fs_0) as u32, ft_0 as u32);
                }
            }
        }
        assert!(itheta >= 0);
        itheta = celt_udiv((itheta * 16384) as u32, qn as u32) as i32;
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
                let mut j = 0;
                while j < N {
                    Y[j as usize] = -Y[j as usize];
                    j += 1;
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
        sctx.delta = 16384
            + ((N - 1) << 7) as i16 as i32 * bitexact_log2tan(iside, imid) as i16 as i32
            >> 15;
    }
    sctx.inv = inv;
    sctx.imid = imid;
    sctx.iside = iside;
    sctx.itheta = itheta;
    sctx.qalloc = qalloc;
}

/// Upstream C: celt/bands.c:quant_band_n1
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
///
/// Uses raw pointer for X to allow recursive splitting. All pointer
/// arithmetic is bounds-checked via the N parameter which tracks the
/// valid length.
unsafe fn quant_partition(
    ctx: &mut band_ctx,
    X: *mut f32,
    mut N: i32,
    mut b: i32,
    mut B: i32,
    lowband: *mut f32,
    mut LM: i32,
    gain: f32,
    mut fill: i32,
    ec: &mut ec_ctx,
) -> u32 {
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
        };
        let mut next_lowband2: *mut f32 = std::ptr::null_mut();
        N >>= 1;
        let Y = X.offset(N as isize);
        LM -= 1;
        if B == 1 {
            fill = fill & 1 | fill << 1;
        }
        B = B + 1 >> 1;
        compute_theta(
            ctx,
            &mut sctx,
            std::slice::from_raw_parts_mut(X, N as usize),
            std::slice::from_raw_parts_mut(Y, N as usize),
            N,
            &mut b,
            B,
            B0,
            LM,
            0,
            &mut fill,
            ec,
        );
        let imid = sctx.imid;
        let iside = sctx.iside;
        let mut delta = sctx.delta;
        let itheta = sctx.itheta;
        let qalloc = sctx.qalloc;
        let mid = 1.0f32 / 32768.0f32 * imid as f32;
        let side = 1.0f32 / 32768.0f32 * iside as f32;
        if B0 > 1 && itheta & 0x3fff != 0 {
            if itheta > 8192 {
                delta -= delta >> 4 - LM;
            } else {
                delta = if 0 < delta + (N << 3 >> 5 - LM) {
                    0
                } else {
                    delta + (N << 3 >> 5 - LM)
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
        if !lowband.is_null() {
            next_lowband2 = lowband.offset(N as isize);
        }
        let mut rebalance = ctx.remaining_bits;
        if mbits >= sbits {
            cm = quant_partition(ctx, X, N, mbits, B, lowband, LM, gain * mid, fill, ec);
            rebalance = mbits - (rebalance - ctx.remaining_bits);
            if rebalance > (3) << BITRES && itheta != 0 {
                sbits += rebalance - ((3) << BITRES);
            }
            cm |= quant_partition(
                ctx,
                Y,
                N,
                sbits,
                B,
                next_lowband2,
                LM,
                gain * side,
                fill >> B,
                ec,
            ) << (B0 >> 1);
        } else {
            cm = quant_partition(
                ctx,
                Y,
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
            cm |= quant_partition(ctx, X, N, mbits, B, lowband, LM, gain * mid, fill, ec);
        }
    } else {
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
                    std::slice::from_raw_parts_mut(X, N as usize),
                    N,
                    K,
                    spread,
                    B,
                    ec,
                    gain,
                    ctx.resynth,
                    ctx.arch,
                );
            } else {
                cm = alg_unquant(
                    std::slice::from_raw_parts_mut(X, N as usize),
                    N,
                    K,
                    spread,
                    B,
                    ec,
                    gain,
                );
            }
        } else {
            cm = 0;
            if ctx.resynth != 0 {
                let cm_mask = (((1_u64) << B) as u32).wrapping_sub(1);
                fill = (fill as u32 & cm_mask) as i32;
                if fill == 0 {
                    std::slice::from_raw_parts_mut(X, N as usize).fill(0.0);
                } else {
                    if lowband.is_null() {
                        let mut j = 0;
                        while j < N {
                            ctx.seed = celt_lcg_rand(ctx.seed);
                            *X.offset(j as isize) = (ctx.seed as i32 >> 20) as f32;
                            j += 1;
                        }
                        cm = cm_mask;
                    } else {
                        let mut j = 0;
                        while j < N {
                            let mut tmp = 1.0f32 / 256.0f32;
                            ctx.seed = celt_lcg_rand(ctx.seed);
                            tmp = if ctx.seed & 0x8000 != 0 { tmp } else { -tmp };
                            *X.offset(j as isize) = *lowband.offset(j as isize) + tmp;
                            j += 1;
                        }
                        cm = fill as u32;
                    }
                    renormalise_vector(
                        std::slice::from_raw_parts_mut(X, N as usize),
                        N,
                        gain,
                        ctx.arch,
                    );
                }
            }
        }
    }
    cm
}

/// Upstream C: celt/bands.c:quant_band
unsafe fn quant_band(
    ctx: &mut band_ctx,
    X: *mut f32,
    N: i32,
    b: i32,
    mut B: i32,
    mut lowband: *mut f32,
    LM: i32,
    lowband_out: *mut f32,
    gain: f32,
    lowband_scratch: *mut f32,
    mut fill: i32,
    ec: &mut ec_ctx,
) -> u32 {
    let N0 = N;
    let mut N_B = N;
    let N_B0: i32;
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
        let lbo_opt = if lowband_out.is_null() {
            None
        } else {
            Some(std::slice::from_raw_parts_mut(lowband_out, 1))
        };
        return quant_band_n1(
            ctx,
            std::slice::from_raw_parts_mut(X, 1),
            y_opt,
            b,
            lbo_opt,
            ec,
        );
    }
    if tf_change > 0 {
        recombine = tf_change;
    }
    if !lowband_scratch.is_null()
        && !lowband.is_null()
        && (recombine != 0 || N_B & 1 == 0 && tf_change < 0 || B0 > 1)
    {
        std::slice::from_raw_parts_mut(lowband_scratch, N as usize)
            .copy_from_slice(std::slice::from_raw_parts(lowband, N as usize));
        lowband = lowband_scratch;
    }

    const BIT_INTERLEAVE_TABLE: [u8; 16] = [0, 1, 1, 1, 2, 3, 3, 3, 2, 3, 3, 3, 2, 3, 3, 3];

    let mut k = 0;
    while k < recombine {
        if encode != 0 {
            haar1(
                std::slice::from_raw_parts_mut(X, N as usize),
                N >> k,
                (1) << k,
            );
        }
        if !lowband.is_null() {
            haar1(
                std::slice::from_raw_parts_mut(lowband, N as usize),
                N >> k,
                (1) << k,
            );
        }
        fill = BIT_INTERLEAVE_TABLE[(fill & 0xf) as usize] as i32
            | (BIT_INTERLEAVE_TABLE[(fill >> 4) as usize] as i32) << 2;
        k += 1;
    }
    B >>= recombine;
    N_B <<= recombine;
    while N_B & 1 == 0 && tf_change < 0 {
        if encode != 0 {
            haar1(std::slice::from_raw_parts_mut(X, N as usize), N_B, B);
        }
        if !lowband.is_null() {
            haar1(std::slice::from_raw_parts_mut(lowband, N as usize), N_B, B);
        }
        fill |= fill << B;
        B <<= 1;
        N_B >>= 1;
        time_divide += 1;
        tf_change += 1;
    }
    B0 = B;
    N_B0 = N_B;
    if B0 > 1 {
        if encode != 0 {
            deinterleave_hadamard(
                std::slice::from_raw_parts_mut(X, N as usize),
                N_B >> recombine,
                B0 << recombine,
                longBlocks,
            );
        }
        if !lowband.is_null() {
            deinterleave_hadamard(
                std::slice::from_raw_parts_mut(lowband, N as usize),
                N_B >> recombine,
                B0 << recombine,
                longBlocks,
            );
        }
    }
    cm = quant_partition(ctx, X, N, b, B, lowband, LM, gain, fill, ec);
    if ctx.resynth != 0 {
        if B0 > 1 {
            interleave_hadamard(
                std::slice::from_raw_parts_mut(X, N as usize),
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
            haar1(std::slice::from_raw_parts_mut(X, N as usize), N_B, B);
            k += 1;
        }

        const BIT_DEINTERLEAVE_TABLE: [u8; 16] = [
            0, 0x3, 0xc, 0xf, 0x30, 0x33, 0x3c, 0x3f, 0xc0, 0xc3, 0xcc, 0xcf, 0xf0, 0xf3, 0xfc,
            0xff,
        ];

        k = 0;
        while k < recombine {
            cm = BIT_DEINTERLEAVE_TABLE[cm as usize] as u32;
            haar1(
                std::slice::from_raw_parts_mut(X, N as usize),
                N0 >> k,
                (1) << k,
            );
            k += 1;
        }
        B <<= recombine;
        if !lowband_out.is_null() {
            let mut j: i32 = 0;
            let n = celt_sqrt(N0 as f32);
            while j < N0 {
                *lowband_out.offset(j as isize) = n * *X.offset(j as isize);
                j += 1;
            }
        }
        cm &= (((1) << B) - 1) as u32;
    }
    cm
}

/// Upstream C: celt/bands.c:quant_band_stereo
unsafe fn quant_band_stereo(
    ctx: &mut band_ctx,
    X: *mut f32,
    Y: *mut f32,
    N: i32,
    mut b: i32,
    B: i32,
    lowband: *mut f32,
    LM: i32,
    lowband_out: *mut f32,
    lowband_scratch: *mut f32,
    mut fill: i32,
    ec: &mut ec_ctx,
) -> u32 {
    let imid: i32;
    let iside: i32;
    let inv: i32;
    let mid: f32;
    let side: f32;
    let mut cm: u32;
    let mut mbits: i32;
    let mut sbits: i32;
    let delta: i32;
    let itheta: i32;
    let qalloc: i32;
    let mut sctx = split_ctx {
        inv: 0,
        imid: 0,
        iside: 0,
        delta: 0,
        itheta: 0,
        qalloc: 0,
    };
    let orig_fill = fill;
    let encode = ctx.encode;
    if N == 1 {
        let x_slice = std::slice::from_raw_parts_mut(X, 1);
        let y_slice = Some(std::slice::from_raw_parts_mut(Y, 1));
        let lbo = if lowband_out.is_null() {
            None
        } else {
            Some(std::slice::from_raw_parts_mut(lowband_out, 1))
        };
        return quant_band_n1(ctx, x_slice, y_slice, b, lbo, ec);
    }
    compute_theta(
        ctx,
        &mut sctx,
        std::slice::from_raw_parts_mut(X, N as usize),
        std::slice::from_raw_parts_mut(Y, N as usize),
        N,
        &mut b,
        B,
        B,
        LM,
        1,
        &mut fill,
        ec,
    );
    inv = sctx.inv;
    imid = sctx.imid;
    iside = sctx.iside;
    delta = sctx.delta;
    itheta = sctx.itheta;
    qalloc = sctx.qalloc;
    mid = 1.0f32 / 32768.0f32 * imid as f32;
    side = 1.0f32 / 32768.0f32 * iside as f32;
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
        let x2 = if c != 0 { Y } else { X };
        let y2 = if c != 0 { X } else { Y };
        if sbits != 0 {
            if encode != 0 {
                sign =
                    (*x2.offset(0) * *y2.offset(1) - *x2.offset(1) * *y2.offset(0) < 0.0f32) as i32;
                ec_enc_bits(ec, sign as u32, 1);
            } else {
                sign = ec_dec_bits(ec, 1) as i32;
            }
        }
        sign = 1 - 2 * sign;
        cm = quant_band(
            ctx,
            x2,
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
        *y2.offset(0) = -sign as f32 * *x2.offset(1);
        *y2.offset(1) = sign as f32 * *x2.offset(0);
        if ctx.resynth != 0 {
            let mut tmp: f32;
            *X.offset(0) = mid * *X.offset(0);
            *X.offset(1) = mid * *X.offset(1);
            *Y.offset(0) = side * *Y.offset(0);
            *Y.offset(1) = side * *Y.offset(1);
            tmp = *X.offset(0);
            *X.offset(0) = tmp - *Y.offset(0);
            *Y.offset(0) = tmp + *Y.offset(0);
            tmp = *X.offset(1);
            *X.offset(1) = tmp - *Y.offset(1);
            *Y.offset(1) = tmp + *Y.offset(1);
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
        if mbits >= sbits {
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
            rebalance = mbits - (rebalance - ctx.remaining_bits);
            if rebalance > (3) << BITRES && itheta != 0 {
                sbits += rebalance - ((3) << BITRES);
            }
            cm |= quant_band(
                ctx,
                Y,
                N,
                sbits,
                B,
                std::ptr::null_mut(),
                LM,
                std::ptr::null_mut(),
                side,
                std::ptr::null_mut(),
                fill >> B,
                ec,
            );
        } else {
            cm = quant_band(
                ctx,
                Y,
                N,
                sbits,
                B,
                std::ptr::null_mut(),
                LM,
                std::ptr::null_mut(),
                side,
                std::ptr::null_mut(),
                fill >> B,
                ec,
            );
            rebalance = sbits - (rebalance - ctx.remaining_bits);
            if rebalance > (3) << BITRES && itheta != 16384 {
                mbits += rebalance - ((3) << BITRES);
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
        if N != 2 {
            stereo_merge(
                std::slice::from_raw_parts_mut(X, N as usize),
                std::slice::from_raw_parts_mut(Y, N as usize),
                mid,
                N,
                ctx.arch,
            );
        }
        if inv != 0 {
            let mut j = 0;
            while j < N {
                *Y.offset(j as isize) = -*Y.offset(j as isize);
                j += 1;
            }
        }
    }
    cm
}

/// Upstream C: celt/bands.c:special_hybrid_folding
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
///
/// # Safety
/// X_ and Y_ must point to valid buffers of sufficient size.
/// Y_ may be null (mono mode) or may alias into X_ (stereo mode).
pub unsafe fn quant_all_bands(
    encode: i32,
    m: &OpusCustomMode,
    start: i32,
    end: i32,
    X_: *mut f32,
    Y_: *mut f32,
    collapse_masks: &mut [u8],
    bandE: &[f32],
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
    arch: i32,
    disable_inv: i32,
) {
    let mut remaining_bits: i32;
    let eBands = &m.eBands;
    let resynth_alloc: i32;
    let B: i32;
    let M: i32 = (1) << LM;
    let mut lowband_offset: i32 = 0;
    let mut update_lowband: i32 = 1;
    let C: i32 = if !Y_.is_null() { 2 } else { 1 };
    let norm_offset: i32 = M * eBands[start as usize] as i32;
    let theta_rdo: i32 =
        (encode != 0 && !Y_.is_null() && dual_stereo == 0 && complexity >= 8) as i32;
    let resynth: i32 = (encode == 0 || theta_rdo != 0) as i32;

    B = if shortBlocks != 0 { M } else { 1 };
    let norm_size = (M * eBands[(m.nbEBands - 1) as usize] as i32 - norm_offset) as usize;
    let mut _norm = vec![0.0f32; C as usize * norm_size];
    // norm = _norm[0..norm_size], norm2 = _norm[norm_size..2*norm_size]
    // Access via index: norm_idx(band_start) = (M * eBands[i] - norm_offset) as usize

    if encode != 0 && resynth != 0 {
        resynth_alloc =
            M * (eBands[m.nbEBands as usize] as i32 - eBands[(m.nbEBands - 1) as usize] as i32);
    } else {
        resynth_alloc = 0;
    }
    // Always allocate lowband_scratch separately to avoid aliasing issues
    let mut _lowband_scratch = vec![
        0.0f32;
        if resynth_alloc > 0 {
            resynth_alloc as usize
        } else {
            1
        }
    ];
    let mut _X_save = vec![0.0f32; resynth_alloc as usize];
    let mut _Y_save = vec![0.0f32; resynth_alloc as usize];
    let mut _X_save2 = vec![0.0f32; resynth_alloc as usize];
    let mut _Y_save2 = vec![0.0f32; resynth_alloc as usize];
    let mut _norm_save2 = vec![0.0f32; resynth_alloc as usize];

    // Determine lowband_scratch pointer
    let lowband_scratch_ptr: *mut f32;
    if encode != 0 && resynth != 0 {
        lowband_scratch_ptr = _lowband_scratch.as_mut_ptr();
    } else {
        // In decode-only mode, use X_[last_band..] as scratch
        lowband_scratch_ptr = X_.offset((M * eBands[(m.nbEBands - 1) as usize] as i32) as isize);
    }

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
    };

    let mut i = start;
    while i < end {
        let b: i32;
        let N: i32;
        let mut effective_lowband: i32 = -1;
        let tf_change: i32;
        let mut x_cm: u32;
        let mut y_cm: u32;
        ctx.i = i;
        let last = (i == end - 1) as i32;
        let X: *mut f32 = X_.offset((M * eBands[i as usize] as i32) as isize);
        let Y: *mut f32 = if !Y_.is_null() {
            Y_.offset((M * eBands[i as usize] as i32) as isize)
        } else {
            std::ptr::null_mut()
        };
        N = M * eBands[(i + 1) as usize] as i32 - M * eBands[i as usize] as i32;
        assert!(N > 0);
        let tell = ec_tell_frac(ec) as i32;
        if i != start {
            balance -= tell;
        }
        remaining_bits = total_bits - tell - 1;
        ctx.remaining_bits = remaining_bits;
        if i <= codedBands - 1 {
            let curr_balance = celt_sudiv(
                balance,
                if 3 < codedBands - i {
                    3
                } else {
                    codedBands - i
                },
            );
            b =
                if 0 > (if 16383
                    < (if (remaining_bits + 1) < pulses[i as usize] + curr_balance {
                        remaining_bits + 1
                    } else {
                        pulses[i as usize] + curr_balance
                    }) {
                    16383
                } else {
                    if (remaining_bits + 1) < pulses[i as usize] + curr_balance {
                        remaining_bits + 1
                    } else {
                        pulses[i as usize] + curr_balance
                    }
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
                };
        } else {
            b = 0;
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
        tf_change = tf_res[i as usize];
        ctx.tf_change = tf_change;
        // For bands beyond effEBands, use norm buffer as X/Y
        let (X, Y, mut lowband_scratch_cur) = if i >= m.effEBands {
            (
                _norm.as_mut_ptr(),
                if !Y_.is_null() {
                    _norm.as_mut_ptr()
                } else {
                    std::ptr::null_mut()
                },
                std::ptr::null_mut(),
            )
        } else {
            (X, Y, lowband_scratch_ptr)
        };
        if last != 0 && theta_rdo == 0 {
            lowband_scratch_cur = std::ptr::null_mut();
        }
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
                if !(M * eBands[fold_start as usize] as i32 > effective_lowband + norm_offset) {
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
                x_cm |= collapse_masks[(fold_i * C + 0) as usize] as u32;
                y_cm |= collapse_masks[(fold_i * C + C - 1) as usize] as u32;
                fold_i += 1;
                if !(fold_i < fold_end) {
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
        if dual_stereo != 0 {
            let norm_ptr = _norm.as_mut_ptr();
            let norm2_ptr = norm_ptr.offset(norm_size as isize);
            let lowband_x = if effective_lowband != -1 {
                norm_ptr.offset(effective_lowband as isize)
            } else {
                std::ptr::null_mut()
            };
            let lowband_y = if effective_lowband != -1 {
                norm2_ptr.offset(effective_lowband as isize)
            } else {
                std::ptr::null_mut()
            };
            let lowband_out_x = if last != 0 {
                std::ptr::null_mut()
            } else {
                norm_ptr.offset((M * eBands[i as usize] as i32 - norm_offset) as isize)
            };
            let lowband_out_y = if last != 0 {
                std::ptr::null_mut()
            } else {
                norm2_ptr.offset((M * eBands[i as usize] as i32 - norm_offset) as isize)
            };
            x_cm = quant_band(
                &mut ctx,
                X,
                N,
                b / 2,
                B,
                lowband_x,
                LM,
                lowband_out_x,
                Q15ONE,
                lowband_scratch_cur,
                x_cm as i32,
                ec,
            );
            y_cm = quant_band(
                &mut ctx,
                Y,
                N,
                b / 2,
                B,
                lowband_y,
                LM,
                lowband_out_y,
                Q15ONE,
                lowband_scratch_cur,
                y_cm as i32,
                ec,
            );
        } else {
            let norm_ptr = _norm.as_mut_ptr();
            let lowband_ptr = if effective_lowband != -1 {
                norm_ptr.offset(effective_lowband as isize)
            } else {
                std::ptr::null_mut()
            };
            let lowband_out_ptr = if last != 0 {
                std::ptr::null_mut()
            } else {
                norm_ptr.offset((M * eBands[i as usize] as i32 - norm_offset) as isize)
            };
            if !Y.is_null() {
                if theta_rdo != 0 && i < intensity {
                    let ec_save;
                    let ec_save2;
                    let ctx_save: band_ctx;
                    let ctx_save2: band_ctx;
                    let dist0: f32;
                    let dist1: f32;
                    let cm: u32;
                    let cm2: u32;
                    let mut w: [f32; 2] = [0.0; 2];
                    compute_channel_weights(
                        bandE[i as usize],
                        bandE[(i + m.nbEBands as i32) as usize],
                        &mut w,
                    );
                    cm = x_cm | y_cm;
                    ec_save = ec.save();
                    ctx_save = ctx;
                    // Save X, Y
                    _X_save[..N as usize]
                        .copy_from_slice(std::slice::from_raw_parts(X, N as usize));
                    _Y_save[..N as usize]
                        .copy_from_slice(std::slice::from_raw_parts(Y, N as usize));
                    // Try theta_round = -1
                    ctx.theta_round = -1;
                    x_cm = quant_band_stereo(
                        &mut ctx,
                        X,
                        Y,
                        N,
                        b,
                        B,
                        lowband_ptr,
                        LM,
                        lowband_out_ptr,
                        lowband_scratch_cur,
                        cm as i32,
                        ec,
                    );
                    dist0 =
                        w[0] * celt_inner_prod(
                            &_X_save[..N as usize],
                            std::slice::from_raw_parts(X, N as usize),
                            N as usize,
                        ) + w[1]
                            * celt_inner_prod(
                                &_Y_save[..N as usize],
                                std::slice::from_raw_parts(Y, N as usize),
                                N as usize,
                            );
                    cm2 = x_cm;
                    ec_save2 = ec.save();
                    ctx_save2 = ctx;
                    // Save X, Y, norm after round -1
                    _X_save2[..N as usize]
                        .copy_from_slice(std::slice::from_raw_parts(X, N as usize));
                    _Y_save2[..N as usize]
                        .copy_from_slice(std::slice::from_raw_parts(Y, N as usize));
                    if last == 0 {
                        let norm_out_off = (M * eBands[i as usize] as i32 - norm_offset) as usize;
                        _norm_save2[..N as usize]
                            .copy_from_slice(&_norm[norm_out_off..norm_out_off + N as usize]);
                    }
                    // Save ec bytes
                    let nstart_bytes = ec_save.offs as usize;
                    let nend_bytes = ec.storage as usize;
                    let save_bytes = nend_bytes - nstart_bytes;
                    let mut bytes_save = vec![0u8; save_bytes];
                    bytes_save.copy_from_slice(&ec.buf[nstart_bytes..nend_bytes]);
                    // Restore state for round +1
                    ec.restore(ec_save);
                    ctx = ctx_save;
                    std::slice::from_raw_parts_mut(X, N as usize)
                        .copy_from_slice(&_X_save[..N as usize]);
                    std::slice::from_raw_parts_mut(Y, N as usize)
                        .copy_from_slice(&_Y_save[..N as usize]);
                    if i == start + 1 {
                        let (norm_part, norm2_part) = _norm.split_at_mut(norm_size);
                        special_hybrid_folding(m, norm_part, norm2_part, start, M, dual_stereo);
                    }
                    // Try theta_round = +1
                    ctx.theta_round = 1;
                    x_cm = quant_band_stereo(
                        &mut ctx,
                        X,
                        Y,
                        N,
                        b,
                        B,
                        lowband_ptr,
                        LM,
                        lowband_out_ptr,
                        lowband_scratch_cur,
                        cm as i32,
                        ec,
                    );
                    dist1 =
                        w[0] * celt_inner_prod(
                            &_X_save[..N as usize],
                            std::slice::from_raw_parts(X, N as usize),
                            N as usize,
                        ) + w[1]
                            * celt_inner_prod(
                                &_Y_save[..N as usize],
                                std::slice::from_raw_parts(Y, N as usize),
                                N as usize,
                            );
                    if dist0 >= dist1 {
                        x_cm = cm2;
                        ec.restore(ec_save2);
                        ctx = ctx_save2;
                        std::slice::from_raw_parts_mut(X, N as usize)
                            .copy_from_slice(&_X_save2[..N as usize]);
                        std::slice::from_raw_parts_mut(Y, N as usize)
                            .copy_from_slice(&_Y_save2[..N as usize]);
                        if last == 0 {
                            let norm_out_off =
                                (M * eBands[i as usize] as i32 - norm_offset) as usize;
                            _norm[norm_out_off..norm_out_off + N as usize]
                                .copy_from_slice(&_norm_save2[..N as usize]);
                        }
                        ec.buf[nstart_bytes..nend_bytes].copy_from_slice(&bytes_save);
                    }
                } else {
                    ctx.theta_round = 0;
                    x_cm = quant_band_stereo(
                        &mut ctx,
                        X,
                        Y,
                        N,
                        b,
                        B,
                        lowband_ptr,
                        LM,
                        lowband_out_ptr,
                        lowband_scratch_cur,
                        (x_cm | y_cm) as i32,
                        ec,
                    );
                }
            } else {
                x_cm = quant_band(
                    &mut ctx,
                    X,
                    N,
                    b,
                    B,
                    lowband_ptr,
                    LM,
                    lowband_out_ptr,
                    Q15ONE,
                    lowband_scratch_cur,
                    (x_cm | y_cm) as i32,
                    ec,
                );
            }
            y_cm = x_cm;
        }
        collapse_masks[(i * C + 0) as usize] = x_cm as u8;
        collapse_masks[(i * C + C - 1) as usize] = y_cm as u8;
        balance += pulses[i as usize] + tell;
        update_lowband = (b > N << BITRES) as i32;
        ctx.avoid_split_noise = 0;
        i += 1;
    }
    *seed = ctx.seed;
}
