//! Bit allocation.
//!
//! Upstream C: `celt/rate.c`

#[cfg(feature = "qext")]
use crate::celt::entcode::ec_tell_frac;
use crate::celt::entcode::{celt_udiv, ec_ctx, BITRES};
use crate::celt::entdec::{ec_dec_bit_logp, ec_dec_uint};
use crate::celt::entenc::{ec_enc_bit_logp, ec_enc_uint};
use crate::celt::modes::OpusCustomMode;

#[cfg(feature = "qext")]
use crate::celt::entdec::ec_dec_icdf;
#[cfg(feature = "qext")]
use crate::celt::entenc::ec_enc_icdf;
#[cfg(feature = "qext")]
use crate::celt::modes::data_96000::NB_QEXT_BANDS;
#[cfg(feature = "qext")]
use crate::celt::quant_bands::eMeans;

pub const FINE_OFFSET: i32 = 21;
pub const MAX_FINE_BITS: i32 = 8;
pub const LOG_MAX_PSEUDO: i32 = 6;
pub const QTHETA_OFFSET_TWOPHASE: i32 = 16;
pub const QTHETA_OFFSET: i32 = 4;
pub const ALLOC_STEPS: i32 = 6;

const LOG2_FRAC_TABLE: [u8; 24] = [
    0, 8, 13, 16, 19, 21, 23, 24, 26, 27, 28, 29, 30, 31, 32, 32, 33, 34, 34, 35, 36, 36, 37, 37,
];

/// Upstream C: celt/rate.h:get_pulses
#[inline]
pub fn get_pulses(i: i32) -> i32 {
    if i < 8 {
        i
    } else {
        (8 + (i & 7)) << ((i >> 3) - 1)
    }
}

/// Upstream C: celt/rate.h:bits2pulses
#[inline]
pub fn bits2pulses(m: &OpusCustomMode, band: i32, mut LM: i32, mut bits: i32) -> i32 {
    LM += 1;
    let cache_off = m.cache.index[(LM * m.nbEBands as i32 + band) as usize] as usize;
    let cache = &m.cache.bits[cache_off..];
    let mut lo: i32 = 0;
    let mut hi: i32 = cache[0] as i32;
    bits -= 1;
    for _ in 0..LOG_MAX_PSEUDO {
        let mid: i32 = (lo + hi + 1) >> 1;
        if cache[mid as usize] as i32 >= bits {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    if bits
        - (if lo == 0 {
            -1
        } else {
            cache[lo as usize] as i32
        })
        <= cache[hi as usize] as i32 - bits
    {
        lo
    } else {
        hi
    }
}

/// Upstream C: celt/rate.h:pulses2bits
#[inline]
pub fn pulses2bits(m: &OpusCustomMode, band: i32, mut LM: i32, pulses: i32) -> i32 {
    LM += 1;
    let cache_off = m.cache.index[(LM * m.nbEBands as i32 + band) as usize] as usize;
    let cache = &m.cache.bits[cache_off..];
    if pulses == 0 {
        0
    } else {
        cache[pulses as usize] as i32 + 1
    }
}

/// Upstream C: celt/rate.c:interp_bits2pulses
#[inline]
fn interp_bits2pulses(
    m: &OpusCustomMode,
    start: i32,
    end: i32,
    skip_start: i32,
    bits1: &[i32],
    bits2: &[i32],
    thresh: &[i32],
    cap: &[i32],
    mut total: i32,
    _balance: &mut i32,
    skip_rsv: i32,
    intensity: &mut i32,
    mut intensity_rsv: i32,
    dual_stereo: &mut i32,
    mut dual_stereo_rsv: i32,
    bits: &mut [i32],
    ebits: &mut [i32],
    fine_priority: &mut [i32],
    C: i32,
    LM: i32,
    ec: &mut ec_ctx,
    encode: i32,
    prev: i32,
    signalBandwidth: i32,
) -> i32 {
    let mut psum: i32;
    let mut j: i32;
    let stereo: i32 = (C > 1) as i32;
    let logM: i32 = LM << BITRES;
    let mut codedBands: i32;
    let alloc_floor: i32 = C << BITRES;
    let mut left: i32;
    let mut percoeff: i32;
    let mut done: i32;
    let mut balance: i32;
    let mut lo: i32 = 0;
    let mut hi: i32 = (1) << ALLOC_STEPS;
    for _ in 0..ALLOC_STEPS {
        let mid: i32 = (lo + hi) >> 1;
        psum = 0;
        done = 0;
        j = end;
        loop {
            let fresh0 = j;
            j -= 1;
            if fresh0 <= start {
                break;
            }
            let tmp: i32 = bits1[j as usize] + ((mid * bits2[j as usize]) >> ALLOC_STEPS);
            if tmp >= thresh[j as usize] || done != 0 {
                done = 1;
                psum += tmp.min(cap[j as usize]);
            } else if tmp >= alloc_floor {
                psum += alloc_floor;
            }
        }
        if psum > total {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    psum = 0;
    done = 0;
    j = end;
    loop {
        let fresh1 = j;
        j -= 1;
        if fresh1 <= start {
            break;
        }
        let mut tmp_0: i32 = bits1[j as usize] + ((lo * bits2[j as usize]) >> ALLOC_STEPS);
        if tmp_0 < thresh[j as usize] && done == 0 {
            if tmp_0 >= alloc_floor {
                tmp_0 = alloc_floor;
            } else {
                tmp_0 = 0;
            }
        } else {
            done = 1;
        }
        tmp_0 = tmp_0.min(cap[j as usize]);
        bits[j as usize] = tmp_0;
        psum += tmp_0;
    }
    codedBands = end;
    loop {
        let band_width: i32;
        let mut band_bits: i32;
        let rem: i32;
        j = codedBands - 1;
        if j <= skip_start {
            total += skip_rsv;
            break;
        } else {
            left = total - psum;
            percoeff = celt_udiv(
                left as u32,
                (m.eBands[codedBands as usize] as i32 - m.eBands[start as usize] as i32) as u32,
            ) as i32;
            left -=
                (m.eBands[codedBands as usize] as i32 - m.eBands[start as usize] as i32) * percoeff;
            rem = (left - (m.eBands[j as usize] as i32 - m.eBands[start as usize] as i32)).max(0);
            band_width = m.eBands[codedBands as usize] as i32 - m.eBands[j as usize] as i32;
            band_bits = bits[j as usize] + percoeff * band_width + rem;
            if band_bits >= thresh[j as usize].max(alloc_floor + ((1) << 3)) {
                if encode != 0 {
                    let mut depth_threshold: i32 = 0;
                    if codedBands > 17 {
                        depth_threshold = if j < prev { 7 } else { 9 };
                    }
                    if codedBands <= start + 2
                        || band_bits > (depth_threshold * band_width) << LM << BITRES >> 4
                            && j <= signalBandwidth
                    {
                        ec_enc_bit_logp(ec, 1, 1);
                        break;
                    } else {
                        ec_enc_bit_logp(ec, 0, 1);
                    }
                } else if ec_dec_bit_logp(ec, 1) != 0 {
                    break;
                }
                psum += (1) << BITRES;
                band_bits -= (1) << BITRES;
            }
            psum -= bits[j as usize] + intensity_rsv;
            if intensity_rsv > 0 {
                intensity_rsv = LOG2_FRAC_TABLE[(j - start) as usize] as i32;
            }
            psum += intensity_rsv;
            if band_bits >= alloc_floor {
                psum += alloc_floor;
                bits[j as usize] = alloc_floor;
            } else {
                bits[j as usize] = 0;
            }
            codedBands -= 1;
        }
    }
    debug_assert!(codedBands > start);
    if intensity_rsv > 0 {
        if encode != 0 {
            *intensity = (*intensity).min(codedBands);
            ec_enc_uint(
                ec,
                (*intensity - start) as u32,
                (codedBands + 1 - start) as u32,
            );
        } else {
            *intensity = (start as u32)
                .wrapping_add(ec_dec_uint(ec, (codedBands + 1 - start) as u32))
                as i32;
        }
    } else {
        *intensity = 0;
    }
    if *intensity <= start {
        total += dual_stereo_rsv;
        dual_stereo_rsv = 0;
    }
    if dual_stereo_rsv > 0 {
        if encode != 0 {
            ec_enc_bit_logp(ec, *dual_stereo, 1);
        } else {
            *dual_stereo = ec_dec_bit_logp(ec, 1);
        }
    } else {
        *dual_stereo = 0;
    }
    left = total - psum;
    percoeff = celt_udiv(
        left as u32,
        (m.eBands[codedBands as usize] as i32 - m.eBands[start as usize] as i32) as u32,
    ) as i32;
    left -= (m.eBands[codedBands as usize] as i32 - m.eBands[start as usize] as i32) * percoeff;
    j = start;
    while j < codedBands {
        bits[j as usize] +=
            percoeff * (m.eBands[(j + 1) as usize] as i32 - m.eBands[j as usize] as i32);
        j += 1;
    }
    j = start;
    while j < codedBands {
        let tmp_1: i32 = left.min(m.eBands[(j + 1) as usize] as i32 - m.eBands[j as usize] as i32);
        bits[j as usize] += tmp_1;
        left -= tmp_1;
        j += 1;
    }
    balance = 0;
    j = start;
    while j < codedBands {
        let den: i32;
        let mut offset: i32;
        let NClogN: i32;
        let mut excess: i32;

        debug_assert!(bits[j as usize] >= 0);
        let N0: i32 = m.eBands[(j + 1) as usize] as i32 - m.eBands[j as usize] as i32;
        let N: i32 = N0 << LM;
        let bit: i32 = bits[j as usize] + balance;
        if N > 1 {
            excess = (bit - cap[j as usize]).max(0);
            bits[j as usize] = bit - excess;
            den = C * N
                + (if C == 2 && N > 2 && *dual_stereo == 0 && j < *intensity {
                    1
                } else {
                    0
                });
            NClogN = den * (m.logN[j as usize] as i32 + logM);
            offset = (NClogN >> 1) - den * FINE_OFFSET;
            if N == 2 {
                offset += den << BITRES >> 2;
            }
            if bits[j as usize] + offset < (den * 2) << BITRES {
                offset += NClogN >> 2;
            } else if bits[j as usize] + offset < (den * 3) << BITRES {
                offset += NClogN >> 3;
            }
            ebits[j as usize] = (bits[j as usize] + offset + (den << (BITRES - 1))).max(0);
            ebits[j as usize] = (celt_udiv(ebits[j as usize] as u32, den as u32) >> BITRES) as i32;
            if C * ebits[j as usize] > bits[j as usize] >> BITRES {
                ebits[j as usize] = bits[j as usize] >> stereo >> BITRES;
            }
            ebits[j as usize] = ebits[j as usize].min(8);
            fine_priority[j as usize] =
                (ebits[j as usize] * (den << BITRES) >= bits[j as usize] + offset) as i32;
            bits[j as usize] -= (C * ebits[j as usize]) << BITRES;
        } else {
            excess = (bit - (C << 3)).max(0);
            bits[j as usize] = bit - excess;
            ebits[j as usize] = 0;
            fine_priority[j as usize] = 1;
        }
        if excess > 0 {
            let extra_fine: i32 =
                (excess >> (stereo + BITRES)).min(MAX_FINE_BITS - ebits[j as usize]);
            ebits[j as usize] += extra_fine;
            let extra_bits: i32 = (extra_fine * C) << BITRES;
            fine_priority[j as usize] = (extra_bits >= excess - balance) as i32;
            excess -= extra_bits;
        }
        balance = excess;
        debug_assert!(bits[j as usize] >= 0);
        debug_assert!(ebits[j as usize] >= 0);
        j += 1;
    }
    *_balance = balance;
    while j < end {
        ebits[j as usize] = bits[j as usize] >> stereo >> BITRES;
        debug_assert!((C * ebits[j as usize]) << 3 == bits[j as usize]);
        bits[j as usize] = 0;
        fine_priority[j as usize] = (ebits[j as usize] < 1) as i32;
        j += 1;
    }
    codedBands
}

/// Upstream C: celt/rate.c:clt_compute_allocation
pub fn clt_compute_allocation(
    m: &OpusCustomMode,
    start: i32,
    end: i32,
    offsets: &[i32],
    cap: &[i32],
    alloc_trim: i32,
    intensity: &mut i32,
    dual_stereo: &mut i32,
    mut total: i32,
    balance: &mut i32,
    pulses: &mut [i32],
    ebits: &mut [i32],
    fine_priority: &mut [i32],
    C: i32,
    LM: i32,
    ec: &mut ec_ctx,
    encode: i32,
    prev: i32,
    signalBandwidth: i32,
) -> i32 {
    let mut j: i32;
    let len: i32 = m.nbEBands as i32;
    let mut skip_start: i32;

    let mut intensity_rsv: i32;
    let mut dual_stereo_rsv: i32;
    total = total.max(0);
    skip_start = start;
    let skip_rsv: i32 = if total >= (1) << BITRES {
        (1) << BITRES
    } else {
        0
    };
    total -= skip_rsv;
    dual_stereo_rsv = 0;
    intensity_rsv = 0;
    if C == 2 {
        intensity_rsv = LOG2_FRAC_TABLE[(end - start) as usize] as i32;
        if intensity_rsv > total {
            intensity_rsv = 0;
        } else {
            total -= intensity_rsv;
            dual_stereo_rsv = if total >= (1) << BITRES {
                (1) << BITRES
            } else {
                0
            };
            total -= dual_stereo_rsv;
        }
    }
    let mut bits1 = [0i32; 21];
    let mut bits2 = [0i32; 21];
    let mut thresh = [0i32; 21];
    let mut trim_offset = [0i32; 21];
    j = start;
    while j < end {
        thresh[j as usize] = (C << 3).max(
            (3 * (m.eBands[(j + 1) as usize] as i32 - m.eBands[j as usize] as i32)) << LM << 3 >> 4,
        );
        trim_offset[j as usize] = (C
            * (m.eBands[(j + 1) as usize] as i32 - m.eBands[j as usize] as i32)
            * (alloc_trim - 5 - LM)
            * (end - j - 1)
            * ((1) << (LM + BITRES)))
            >> 6;
        if (m.eBands[(j + 1) as usize] as i32 - m.eBands[j as usize] as i32) << LM == 1 {
            trim_offset[j as usize] -= C << BITRES;
        }
        j += 1;
    }
    let mut lo: i32 = 1;
    let mut hi: i32 = m.nbAllocVectors - 1;
    loop {
        let mut done: i32 = 0;
        let mut psum: i32 = 0;
        let mid: i32 = (lo + hi) >> 1;
        j = end;
        loop {
            let fresh2 = j;
            j -= 1;
            if fresh2 <= start {
                break;
            }
            let N: i32 = m.eBands[(j + 1) as usize] as i32 - m.eBands[j as usize] as i32;
            let mut bitsj: i32 =
                (C * N * m.allocVectors[(mid * len + j) as usize] as i32) << LM >> 2;
            if bitsj > 0 {
                bitsj = (bitsj + trim_offset[j as usize]).max(0);
            }
            bitsj += offsets[j as usize];
            if bitsj >= thresh[j as usize] || done != 0 {
                done = 1;
                psum += bitsj.min(cap[j as usize]);
            } else if bitsj >= C << BITRES {
                psum += C << BITRES;
            }
        }
        if psum > total {
            hi = mid - 1;
        } else {
            lo = mid + 1;
        }
        if lo > hi {
            break;
        }
    }
    let fresh3 = lo;
    lo -= 1;
    hi = fresh3;
    j = start;
    while j < end {
        let N_0: i32 = m.eBands[(j + 1) as usize] as i32 - m.eBands[j as usize] as i32;
        let mut bits1j: i32 = (C * N_0 * m.allocVectors[(lo * len + j) as usize] as i32) << LM >> 2;
        let mut bits2j: i32 = if hi >= m.nbAllocVectors {
            cap[j as usize]
        } else {
            (C * N_0 * m.allocVectors[(hi * len + j) as usize] as i32) << LM >> 2
        };
        if bits1j > 0 {
            bits1j = (bits1j + trim_offset[j as usize]).max(0);
        }
        if bits2j > 0 {
            bits2j = (bits2j + trim_offset[j as usize]).max(0);
        }
        if lo > 0 {
            bits1j += offsets[j as usize];
        }
        bits2j += offsets[j as usize];
        if offsets[j as usize] > 0 {
            skip_start = j;
        }
        bits2j = (bits2j - bits1j).max(0);
        bits1[j as usize] = bits1j;
        bits2[j as usize] = bits2j;
        j += 1;
    }

    interp_bits2pulses(
        m,
        start,
        end,
        skip_start,
        &bits1,
        &bits2,
        &thresh,
        cap,
        total,
        balance,
        skip_rsv,
        intensity,
        intensity_rsv,
        dual_stereo,
        dual_stereo_rsv,
        pulses,
        ebits,
        fine_priority,
        C,
        LM,
        ec,
        encode,
        prev,
        signalBandwidth,
    )
}

// ---------------------------------------------------------------------------
// QEXT extra allocation — depth-based bit allocation for extended bands
// ---------------------------------------------------------------------------

#[cfg(feature = "qext")]
#[allow(dead_code)]
static LAST_ZERO: [u8; 3] = [64, 50, 0];
#[cfg(feature = "qext")]
#[allow(dead_code)]
static LAST_CAP: [u8; 3] = [110, 60, 0];
#[cfg(feature = "qext")]
#[allow(dead_code)]
static LAST_OTHER: [u8; 4] = [120, 112, 70, 0];

/// Context-adaptive entropy encoding of a depth value.
///
/// Upstream C: celt/rate.c:ec_enc_depth
#[cfg(feature = "qext")]
#[allow(dead_code)]
fn ec_enc_depth(enc: &mut ec_ctx, depth: i32, cap: i32, last: &mut i32) {
    let mut sym = 3;
    if depth == *last {
        sym = 2;
    }
    if depth == cap {
        sym = 1;
    }
    if depth == 0 {
        sym = 0;
    }
    if *last == 0 {
        ec_enc_icdf(enc, sym.min(2), &LAST_ZERO, 7);
    } else if *last == cap {
        ec_enc_icdf(enc, sym.min(2), &LAST_CAP, 7);
    } else {
        ec_enc_icdf(enc, sym, &LAST_OTHER, 7);
    }
    if sym == 3 {
        ec_enc_uint(enc, (depth - 1) as u32, cap as u32);
    }
    *last = depth;
}

/// Context-adaptive entropy decoding of a depth value.
///
/// Upstream C: celt/rate.c:ec_dec_depth
#[cfg(feature = "qext")]
#[allow(dead_code)]
fn ec_dec_depth(dec: &mut ec_ctx, cap: i32, last: &mut i32) -> i32 {
    let sym;
    if *last == 0 {
        let s = ec_dec_icdf(dec, &LAST_ZERO, 7);
        sym = if s == 2 { 3 } else { s };
    } else if *last == cap {
        let s = ec_dec_icdf(dec, &LAST_CAP, 7);
        sym = if s == 2 { 3 } else { s };
    } else {
        sym = ec_dec_icdf(dec, &LAST_OTHER, 7);
    }
    let depth = if sym == 0 {
        0
    } else if sym == 1 {
        cap
    } else if sym == 2 {
        *last
    } else {
        1 + ec_dec_uint(dec, cap as u32) as i32
    };
    *last = depth;
    depth
}

/// Compute median of 5 values using a comparison network.
///
/// Upstream C: celt/rate.c:median_of_5_val16
#[cfg(feature = "qext")]
#[allow(dead_code)]
fn median_of_5_val16(x: &[f32]) -> f32 {
    let t2 = x[2];
    let (mut t0, mut t1) = if x[0] > x[1] {
        (x[1], x[0])
    } else {
        (x[0], x[1])
    };
    let (mut t3, mut t4) = if x[3] > x[4] {
        (x[4], x[3])
    } else {
        (x[3], x[4])
    };
    if t0 > t3 {
        std::mem::swap(&mut t0, &mut t3);
        std::mem::swap(&mut t1, &mut t4);
    }
    if t2 > t1 {
        if t1 < t3 {
            t2.min(t3)
        } else {
            t4.min(t1)
        }
    } else if t2 < t3 {
        t1.min(t3)
    } else {
        t2.min(t4)
    }
}

/// Compute extra bit allocation (depth) for QEXT extended bands.
///
/// On encode: analyses the spectrum, computes depth per band, entropy-codes the depths.
/// On decode: reads the depths from the bitstream.
/// For both: converts depth to `extra_pulses` and `extra_equant` per band.
///
/// `qext_mode`: the QEXT mode (or None if no QEXT bands).
/// `qext_end`: number of QEXT bands to allocate for.
/// `bandLogE` / `qext_bandLogE`: per-band log energies (encoder only).
/// `total`: total bits available for extra allocation (in BITRES units).
/// `extra_pulses` / `extra_equant`: output arrays, length = end + qext_end.
/// `tone_freq`, `toneishness`: tone detection parameters (encoder only).
///
/// Upstream C: celt/rate.c:clt_compute_extra_allocation
#[cfg(feature = "qext")]
#[allow(dead_code)]
pub fn clt_compute_extra_allocation(
    m: &OpusCustomMode,
    qext_mode: Option<&OpusCustomMode>,
    start: i32,
    end: i32,
    qext_end: i32,
    bandLogE: Option<&[f32]>,
    qext_bandLogE: Option<&[f32]>,
    total: i32,
    extra_pulses: &mut [i32],
    extra_equant: &mut [i32],
    C: i32,
    LM: i32,
    ec: &mut ec_ctx,
    encode: i32,
    tone_freq: f32,
    toneishness: f32,
) {
    let mut last: i32 = 0;
    let tot_bands: i32;
    let tot_samples: i32;

    if let Some(qm) = qext_mode {
        debug_assert!(end == m.nbEBands as i32);
        tot_bands = end + qext_end;
        tot_samples = (qm.eBands[qext_end as usize] as i32 * C) << LM;
    } else {
        tot_bands = end;
        tot_samples = ((m.eBands[end as usize] as i32 - m.eBands[start as usize] as i32) * C) << LM;
    }

    // Max bands: 21 standard + 14 QEXT = 35; use stack buffers.
    const MAX_BANDS: usize = 40;
    debug_assert!((tot_bands as usize) <= MAX_BANDS);
    let mut cap = [0i32; MAX_BANDS];
    for i in start..end {
        cap[i as usize] = 12;
    }
    if qext_mode.is_some() {
        for i in 0..qext_end {
            cap[(end + i) as usize] = 14;
        }
    }

    if total <= 0 {
        for i in start..(m.nbEBands as i32 + qext_end) {
            extra_pulses[i as usize] = 0;
            extra_equant[i as usize] = 0;
        }
        return;
    }

    let mut depth = [0i32; MAX_BANDS];

    if encode != 0 {
        let bandLogE = bandLogE.unwrap();
        let mut flatE = [0.0f32; MAX_BANDS];
        let mut min_arr = [0.0f32; MAX_BANDS];
        let mut Ncoef = [0i32; MAX_BANDS];

        for i in start..end {
            let iu = i as usize;
            Ncoef[iu] = ((m.eBands[iu + 1] as i32 - m.eBands[iu] as i32) * C) << LM;
        }

        // Remove the effect of band width, eMeans and pre-emphasis to compute flat spectrum.
        for i in start..end {
            let iu = i as usize;
            flatE[iu] = bandLogE[iu] - 0.0625 * m.logN[iu] as f32 + eMeans[iu]
                - 0.0062 * (i + 5) as f32 * (i + 5) as f32;
            min_arr[iu] = 0.0;
        }
        if C == 2 {
            for i in start..end {
                let iu = i as usize;
                let alt = bandLogE[m.nbEBands + iu] - 0.0625 * m.logN[iu] as f32 + eMeans[iu]
                    - 0.0062 * (i + 5) as f32 * (i + 5) as f32;
                if alt > flatE[iu] {
                    flatE[iu] = alt;
                }
            }
        }
        flatE[(end - 1) as usize] += 2.0; // QCONST16(2.f, 10) = 2.0 in float

        if let Some(qm) = qext_mode {
            let qext_bandLogE = qext_bandLogE.unwrap();
            let mut min_depth: f32 = 0.0;
            // If we have enough bits, give at least 1 bit of depth to all higher bands.
            if total
                >= ((3
                    * C
                    * (qm.eBands[qext_end as usize] as i32 - qm.eBands[start as usize] as i32))
                    << LM)
                    << BITRES
                && (toneishness < 0.98 || tone_freq > 1.33)
            {
                min_depth = 1.0; // QCONST16(1.f, 10) = 1.0 in float
            }
            for i in 0..qext_end {
                let iu = i as usize;
                let eid = (end + i) as usize;
                Ncoef[eid] = ((qm.eBands[iu + 1] as i32 - qm.eBands[iu] as i32) * C) << LM;
                min_arr[eid] = min_depth;
            }
            for i in 0..qext_end {
                let iu = i as usize;
                let eid = (end + i) as usize;
                flatE[eid] = qext_bandLogE[iu] - 0.0625 * qm.logN[iu] as f32 + eMeans[iu]
                    - 0.0062 * (end + i + 5) as f32 * (end + i + 5) as f32;
            }
            if C == 2 {
                for i in 0..qext_end {
                    let iu = i as usize;
                    let eid = (end + i) as usize;
                    let alt = qext_bandLogE[NB_QEXT_BANDS + iu] - 0.0625 * qm.logN[iu] as f32
                        + eMeans[iu]
                        - 0.0062 * (end + i + 5) as f32 * (end + i + 5) as f32;
                    if alt > flatE[eid] {
                        flatE[eid] = alt;
                    }
                }
            }
        }

        // Median filter to smooth spectrum
        let mut follower = [0.0f32; MAX_BANDS];
        for i in (start + 2)..(tot_bands - 2) {
            follower[i as usize] = median_of_5_val16(&flatE[(i - 2) as usize..]);
        }
        follower[start as usize] = follower[(start + 2) as usize];
        follower[(start + 1) as usize] = follower[(start + 2) as usize];
        follower[(tot_bands - 1) as usize] = follower[(tot_bands - 3) as usize];
        follower[(tot_bands - 2) as usize] = follower[(tot_bands - 3) as usize];

        // Monotonic increase from left
        for i in (start + 1)..tot_bands {
            let iu = i as usize;
            follower[iu] = follower[iu].max(follower[iu - 1] - 1.0);
        }
        // Monotonic increase from right
        for i in (start..=(tot_bands - 2)).rev() {
            let iu = i as usize;
            follower[iu] = follower[iu].max(follower[iu + 1] - 1.0);
        }

        // Blend out the follower based on tone content
        for i in start..tot_bands {
            let iu = i as usize;
            flatE[iu] -= (1.0 - toneishness) * follower[iu];
        }
        // QEXT boost
        if qext_mode.is_some() {
            for i in 0..qext_end {
                let eid = (end + i) as usize;
                flatE[eid] = flatE[eid] + 3.0 + 0.2 * i as f32;
            }
        }

        // Approximate fill level assuming all bands contribute fully.
        let mut sum: f32 = 0.0;
        for i in start..tot_bands {
            let iu = i as usize;
            sum += Ncoef[iu] as f32 * flatE[iu];
        }
        let total_shifted = total >> BITRES;
        let mut fill: f32 = ((total_shifted as f32) * 1024.0 + sum) / tot_samples as f32;

        // Iteratively refine the fill level considering depth min and cap.
        for _iter in 0..10 {
            sum = 0.0;
            for i in start..tot_bands {
                let iu = i as usize;
                let clamped = (flatE[iu] - fill)
                    .max(min_arr[iu])
                    .min(cap[iu] as f32 * 1024.0);
                sum += Ncoef[iu] as f32 * clamped;
            }
            fill -= ((total_shifted as f32) * 1024.0 - sum) / tot_samples as f32;
        }

        // Convert fill level to depth and encode
        for i in start..tot_bands {
            let iu = i as usize;
            let clamped = (flatE[iu] - fill)
                .max(min_arr[iu])
                .min(cap[iu] as f32 * 1024.0);
            depth[iu] = (0.5 + 4.0 * clamped).floor() as i32;

            if ec_tell_frac(ec) + 80 < (ec.storage * 8) << BITRES {
                ec_enc_depth(ec, depth[iu], 4 * cap[iu], &mut last);
            } else {
                depth[iu] = 0;
            }
        }
    } else {
        // Decode path
        for i in start..tot_bands {
            let iu = i as usize;
            if ec_tell_frac(ec) + 80 < (ec.storage * 8) << BITRES {
                depth[iu] = ec_dec_depth(ec, 4 * cap[iu], &mut last);
            } else {
                depth[iu] = 0;
            }
        }
    }

    // Convert depth to extra_equant and extra_pulses for main bands
    for i in start..end {
        let iu = i as usize;
        extra_equant[iu] = (depth[iu] + 3) >> 2;
        extra_pulses[iu] = ((((m.eBands[iu + 1] as i32 - m.eBands[iu] as i32) << LM) - 1)
            * C
            * depth[iu]
            * (1 << BITRES)
            + 2)
            >> 2;
    }
    // Convert for QEXT bands
    if let Some(qm) = qext_mode {
        for i in 0..qext_end {
            let iu = i as usize;
            let eid = (end + i) as usize;
            extra_equant[eid] = (depth[eid] + 3) >> 2;
            extra_pulses[eid] = ((((qm.eBands[iu + 1] as i32 - qm.eBands[iu] as i32) << LM) - 1)
                * C
                * depth[eid]
                * (1 << BITRES)
                + 2)
                >> 2;
        }
    }
}
