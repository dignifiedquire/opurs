//! Band energy quantization.
//!
//! Upstream c: `celt/quant_bands.c`

use crate::celt::entcode::{ec_tell, ec_tell_frac};
use crate::celt::entdec::{ec_dec_bit_logp, ec_dec_bits, ec_dec_icdf, EcDec};
use crate::celt::entenc::{ec_enc_bit_logp, ec_enc_bits, ec_enc_icdf, EcEnc};
use crate::celt::laplace::{ec_laplace_decode, ec_laplace_encode};
use crate::celt::mathops::celt_log2;
use crate::celt::modes::OpusCustomMode;
use crate::celt::rate::MAX_FINE_BITS;

pub const E_MEANS: [f32; 25] = [
    6.437_5_f32,
    6.25_f32,
    5.75_f32,
    5.312_5_f32,
    5.062_5_f32,
    4.812_5_f32,
    4.5_f32,
    4.375_f32,
    4.875_f32,
    4.687_5_f32,
    4.562_5_f32,
    4.437_5_f32,
    4.875_f32,
    4.625_f32,
    4.312_5_f32,
    4.5_f32,
    4.375_f32,
    4.625_f32,
    4.75_f32,
    4.437_5_f32,
    3.75_f32,
    3.75_f32,
    3.75_f32,
    3.75_f32,
    3.75_f32,
];

const PRED_COEF: [f32; 4] = [
    (29440_f64 / 32768.0f64) as f32,
    (26112_f64 / 32768.0f64) as f32,
    (21248_f64 / 32768.0f64) as f32,
    (16384_f64 / 32768.0f64) as f32,
];

const BETA_COEF: [f32; 4] = [
    (30147_f64 / 32768.0f64) as f32,
    (22282_f64 / 32768.0f64) as f32,
    (12124_f64 / 32768.0f64) as f32,
    (6554_f64 / 32768.0f64) as f32,
];

const BETA_INTRA: f32 = (4915_f64 / 32768.0f64) as f32;

const E_PROB_MODEL: [[[u8; 42]; 2]; 4] = [
    [
        [
            72, 127, 65, 129, 66, 128, 65, 128, 64, 128, 62, 128, 64, 128, 64, 128, 92, 78, 92, 79,
            92, 78, 90, 79, 116, 41, 115, 40, 114, 40, 132, 26, 132, 26, 145, 17, 161, 12, 176, 10,
            177, 11,
        ],
        [
            24, 179, 48, 138, 54, 135, 54, 132, 53, 134, 56, 133, 55, 132, 55, 132, 61, 114, 70,
            96, 74, 88, 75, 88, 87, 74, 89, 66, 91, 67, 100, 59, 108, 50, 120, 40, 122, 37, 97, 43,
            78, 50,
        ],
    ],
    [
        [
            83, 78, 84, 81, 88, 75, 86, 74, 87, 71, 90, 73, 93, 74, 93, 74, 109, 40, 114, 36, 117,
            34, 117, 34, 143, 17, 145, 18, 146, 19, 162, 12, 165, 10, 178, 7, 189, 6, 190, 8, 177,
            9,
        ],
        [
            23, 178, 54, 115, 63, 102, 66, 98, 69, 99, 74, 89, 71, 91, 73, 91, 78, 89, 86, 80, 92,
            66, 93, 64, 102, 59, 103, 60, 104, 60, 117, 52, 123, 44, 138, 35, 133, 31, 97, 38, 77,
            45,
        ],
    ],
    [
        [
            61, 90, 93, 60, 105, 42, 107, 41, 110, 45, 116, 38, 113, 38, 112, 38, 124, 26, 132, 27,
            136, 19, 140, 20, 155, 14, 159, 16, 158, 18, 170, 13, 177, 10, 187, 8, 192, 6, 175, 9,
            159, 10,
        ],
        [
            21, 178, 59, 110, 71, 86, 75, 85, 84, 83, 91, 66, 88, 73, 87, 72, 92, 75, 98, 72, 105,
            58, 107, 54, 115, 52, 114, 55, 112, 56, 129, 51, 132, 40, 150, 33, 140, 29, 98, 35, 77,
            42,
        ],
    ],
    [
        [
            42, 121, 96, 66, 108, 43, 111, 40, 117, 44, 123, 32, 120, 36, 119, 33, 127, 33, 134,
            34, 139, 21, 147, 23, 152, 20, 158, 25, 154, 26, 166, 21, 173, 16, 184, 13, 184, 10,
            150, 13, 139, 15,
        ],
        [
            22, 178, 63, 114, 74, 82, 84, 83, 92, 82, 103, 62, 96, 72, 96, 67, 101, 73, 107, 72,
            113, 55, 118, 52, 125, 52, 118, 52, 117, 55, 135, 49, 137, 39, 157, 32, 145, 29, 97,
            33, 77, 40,
        ],
    ],
];

const SMALL_ENERGY_ICDF: [u8; 3] = [2, 1, 0];

/// Upstream c: celt/quant_bands.c:loss_distortion
fn loss_distortion(
    e_bands: &[f32],
    old_ebands: &[f32],
    start: i32,
    end: i32,
    len: i32,
    channels: i32,
) -> f32 {
    let mut dist: f32 = 0.0;
    let mut ch = 0;
    loop {
        for i in start..end {
            let d = e_bands[(i + ch * len) as usize] - old_ebands[(i + ch * len) as usize];
            dist += d * d;
        }
        ch += 1;
        if ch >= channels {
            break;
        }
    }
    dist.min(200.0)
}

/// Upstream c: celt/quant_bands.c:quant_coarse_energy_impl
#[allow(clippy::too_many_arguments)]
fn quant_coarse_energy_impl(
    m: &OpusCustomMode,
    start: i32,
    end: i32,
    e_bands: &[f32],
    old_ebands: &mut [f32],
    budget: i32,
    mut tell: i32,
    prob_model: &[u8],
    error: &mut [f32],
    enc: &mut EcEnc,
    channels: i32,
    lm: i32,
    intra: i32,
    max_decay: f32,
    lfe: i32,
) -> i32 {
    let mut badness: i32 = 0;
    let mut prev: [f32; 2] = [0.0, 0.0];
    let coef: f32;
    let beta: f32;
    if tell + 3 <= budget {
        ec_enc_bit_logp(enc, intra, 3);
    }
    if intra != 0 {
        coef = 0.0;
        beta = BETA_INTRA;
    } else {
        beta = BETA_COEF[lm as usize];
        coef = PRED_COEF[lm as usize];
    }
    let nb_ebands = m.nb_ebands as i32;
    for i in start..end {
        let mut ch = 0;
        loop {
            let x = e_bands[(i + ch * nb_ebands) as usize];
            let old_e = (-9.0f32).max(old_ebands[(i + ch * nb_ebands) as usize]);
            let f = x - coef * old_e - prev[ch as usize];
            let mut qi = (0.5f32 + f).floor() as i32;
            let decay_bound = (-28.0f32).max(old_ebands[(i + ch * nb_ebands) as usize]) - max_decay;
            if qi < 0 && x < decay_bound {
                qi += (decay_bound - x) as i32;
                if qi > 0 {
                    qi = 0;
                }
            }
            let qi0 = qi;
            tell = ec_tell(enc);
            let bits_left = budget - tell - 3 * channels * (end - i);
            if i != start && bits_left < 30 {
                if bits_left < 24 {
                    qi = 1.min(qi);
                }
                if bits_left < 16 {
                    qi = (-1).max(qi);
                }
            }
            if lfe != 0 && i >= 2 {
                qi = qi.min(0);
            }
            if budget - tell >= 15 {
                let pi = 2 * (i.min(20));
                ec_laplace_encode(
                    enc,
                    &mut qi,
                    ((prob_model[pi as usize] as i32) << 7) as u32,
                    (prob_model[(pi + 1) as usize] as i32) << 6,
                );
            } else if budget - tell >= 2 {
                qi = (-1).max(qi.min(1));
                ec_enc_icdf(enc, (2 * qi) ^ -((qi < 0) as i32), &SMALL_ENERGY_ICDF, 2);
            } else if budget - tell >= 1 {
                qi = 0.min(qi);
                ec_enc_bit_logp(enc, -qi, 1);
            } else {
                qi = -1;
            }
            error[(i + ch * nb_ebands) as usize] = f - qi as f32;
            badness += (qi0 - qi).abs();
            let q = qi as f32;
            let tmp = coef * old_e + prev[ch as usize] + q;
            old_ebands[(i + ch * nb_ebands) as usize] = tmp;
            prev[ch as usize] = prev[ch as usize] + q - beta * q;
            ch += 1;
            if ch >= channels {
                break;
            }
        }
    }
    if lfe != 0 {
        0
    } else {
        badness
    }
}

/// Upstream c: celt/quant_bands.c:quant_coarse_energy
#[allow(clippy::too_many_arguments)]
pub fn quant_coarse_energy(
    m: &OpusCustomMode,
    start: i32,
    end: i32,
    eff_end: i32,
    e_bands: &[f32],
    old_ebands: &mut [f32],
    budget: u32,
    error: &mut [f32],
    enc: &mut EcEnc,
    c: i32,
    lm: i32,
    nb_available_bytes: i32,
    force_intra: i32,
    delayed_intra: &mut f32,
    mut two_pass: i32,
    loss_rate: i32,
    lfe: i32,
) {
    let mut intra: i32;
    let mut badness1: i32 = 0;
    let nb_ebands = m.nb_ebands as i32;
    let band_size = (c * nb_ebands) as usize;

    intra = (force_intra != 0
        || two_pass == 0
            && *delayed_intra > (2 * c * (end - start)) as f32
            && nb_available_bytes > (end - start) * c) as i32;
    let intra_bias = (budget as f32 * *delayed_intra * loss_rate as f32 / (c * 512) as f32) as i32;
    let new_distortion = loss_distortion(e_bands, old_ebands, start, eff_end, nb_ebands, c);
    let tell = ec_tell(enc) as u32;
    if tell.wrapping_add(3) > budget {
        intra = 0;
        two_pass = intra;
    }
    let mut max_decay: f32 = 16.0f32;
    if end - start > 10 {
        max_decay = max_decay.min(0.125f32 * nb_available_bytes as f32);
    }
    if lfe != 0 {
        max_decay = 3.0f32;
    }
    let enc_start_state = enc.save();
    // band_size = c * nb_ebands; max is 2 * 21 = 42; use stack buffers.
    const MAX_BAND_SIZE: usize = 48;
    debug_assert!(band_size <= MAX_BAND_SIZE);
    let mut old_ebands_intra = [0.0f32; MAX_BAND_SIZE];
    let mut error_intra = [0.0f32; MAX_BAND_SIZE];
    old_ebands_intra[..band_size].copy_from_slice(&old_ebands[..band_size]);

    if two_pass != 0 || intra != 0 {
        badness1 = quant_coarse_energy_impl(
            m,
            start,
            end,
            e_bands,
            &mut old_ebands_intra,
            budget as i32,
            tell as i32,
            &E_PROB_MODEL[lm as usize][1],
            &mut error_intra,
            enc,
            c,
            lm,
            1,
            max_decay,
            lfe,
        );
    }
    if intra == 0 {
        let tell_intra = ec_tell_frac(enc) as i32;
        let enc_intra_state = enc.save();
        let nstart_bytes = enc_start_state.offs as usize;
        let nintra_bytes = enc_intra_state.offs as usize;
        let save_bytes = if nintra_bytes - nstart_bytes == 0 {
            1
        } else {
            nintra_bytes - nstart_bytes
        };
        // save_bytes bounded by Opus max packet size (~1275 bytes).
        const MAX_SAVE: usize = 1280;
        debug_assert!(save_bytes <= MAX_SAVE);
        let mut intra_bits = [0u8; MAX_SAVE];
        intra_bits[..nintra_bytes - nstart_bytes]
            .copy_from_slice(&enc.buf[nstart_bytes..nintra_bytes]);

        enc.restore(enc_start_state);
        let badness2 = quant_coarse_energy_impl(
            m,
            start,
            end,
            e_bands,
            old_ebands,
            budget as i32,
            tell as i32,
            &E_PROB_MODEL[lm as usize][intra as usize],
            error,
            enc,
            c,
            lm,
            0,
            max_decay,
            lfe,
        );
        if two_pass != 0
            && (badness1 < badness2
                || badness1 == badness2 && ec_tell_frac(enc) as i32 + intra_bias > tell_intra)
        {
            enc.restore(enc_intra_state);
            enc.buf[nstart_bytes..nintra_bytes]
                .copy_from_slice(&intra_bits[..nintra_bytes - nstart_bytes]);
            old_ebands[..band_size].copy_from_slice(&old_ebands_intra[..band_size]);
            error[..band_size].copy_from_slice(&error_intra[..band_size]);
            intra = 1;
        }
    } else {
        old_ebands[..band_size].copy_from_slice(&old_ebands_intra[..band_size]);
        error[..band_size].copy_from_slice(&error_intra[..band_size]);
    }
    if intra != 0 {
        *delayed_intra = new_distortion;
    } else {
        *delayed_intra =
            PRED_COEF[lm as usize] * PRED_COEF[lm as usize] * *delayed_intra + new_distortion;
    };
}

/// Upstream c: celt/quant_bands.c:quant_fine_energy
#[allow(clippy::too_many_arguments)]
pub fn quant_fine_energy(
    m: &OpusCustomMode,
    start: i32,
    end: i32,
    old_ebands: &mut [f32],
    error: &mut [f32],
    prev_quant: Option<&[i32]>,
    extra_quant: &[i32],
    enc: &mut EcEnc,
    channels: i32,
) {
    let nb_ebands = m.nb_ebands as i32;
    for i in start..end {
        let extra_bits = extra_quant[i as usize];
        if !(1..=14).contains(&extra_bits) {
            continue;
        }
        let extra = 1i32 << extra_bits;
        if ec_tell(enc) + channels * extra_bits > enc.storage as i32 * 8 {
            continue;
        }
        let prev = prev_quant.map_or(0, |pq| pq[i as usize]);
        if !(0..=14).contains(&prev) {
            continue;
        }
        let mut ch = 0;
        loop {
            let mut q2 = ((error[(i + ch * nb_ebands) as usize] * (1 << prev) as f32 + 0.5f32)
                * extra as f32)
                .floor() as i32;
            if q2 > extra - 1 {
                q2 = extra - 1;
            }
            if q2 < 0 {
                q2 = 0;
            }
            ec_enc_bits(enc, q2 as u32, extra_bits as u32);
            let mut offset =
                (q2 as f32 + 0.5f32) * ((1) << (14 - extra_bits)) as f32 * (1.0f32 / 16384.0)
                    - 0.5f32;
            offset *= (1 << (14 - prev)) as f32 * (1.0f32 / 16384.0);
            old_ebands[(i + ch * nb_ebands) as usize] += offset;
            error[(i + ch * nb_ebands) as usize] -= offset;
            ch += 1;
            if ch >= channels {
                break;
            }
        }
    }
}

/// Upstream c: celt/quant_bands.c:quant_energy_finalise
#[allow(clippy::too_many_arguments)]
pub fn quant_energy_finalise(
    m: &OpusCustomMode,
    start: i32,
    end: i32,
    old_ebands: &mut [f32],
    error: &mut [f32],
    fine_quant: &[i32],
    fine_priority: &[i32],
    mut bits_left: i32,
    enc: &mut EcEnc,
    channels: i32,
) {
    let nb_ebands = m.nb_ebands as i32;
    let mut prio = 0;
    while prio < 2 {
        let mut i = start;
        while i < end && bits_left >= channels {
            if !(fine_quant[i as usize] >= MAX_FINE_BITS || fine_priority[i as usize] != prio) {
                let mut ch = 0;
                loop {
                    let q2 = if error[(i + ch * nb_ebands) as usize] < 0.0 {
                        0
                    } else {
                        1
                    };
                    ec_enc_bits(enc, q2 as u32, 1);
                    let offset = (q2 as f32 - 0.5f32)
                        * ((1) << (14 - fine_quant[i as usize] - 1)) as f32
                        * (1.0f32 / 16384.0);
                    old_ebands[(i + ch * nb_ebands) as usize] += offset;
                    error[(i + ch * nb_ebands) as usize] -= offset;
                    bits_left -= 1;
                    ch += 1;
                    if ch >= channels {
                        break;
                    }
                }
            }
            i += 1;
        }
        prio += 1;
    }
}

/// Upstream c: celt/quant_bands.c:unquant_coarse_energy
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn unquant_coarse_energy(
    m: &OpusCustomMode,
    start: i32,
    end: i32,
    old_ebands: &mut [f32],
    intra: i32,
    dec: &mut EcDec,
    channels: i32,
    lm: i32,
) {
    let prob_model = &E_PROB_MODEL[lm as usize][intra as usize];
    let mut prev: [f32; 2] = [0.0, 0.0];
    let coef: f32;
    let beta: f32;
    let nb_ebands = m.nb_ebands as i32;
    if intra != 0 {
        coef = 0.0;
        beta = BETA_INTRA;
    } else {
        beta = BETA_COEF[lm as usize];
        coef = PRED_COEF[lm as usize];
    }
    let budget = dec.storage.wrapping_mul(8) as i32;
    for i in start..end {
        let mut ch = 0;
        loop {
            let qi: i32;
            let tell = ec_tell(dec);
            if budget - tell >= 15 {
                let pi = 2 * (i.min(20));
                qi = ec_laplace_decode(
                    dec,
                    ((prob_model[pi as usize] as i32) << 7) as u32,
                    (prob_model[(pi + 1) as usize] as i32) << 6,
                );
            } else if budget - tell >= 2 {
                let raw = ec_dec_icdf(dec, &SMALL_ENERGY_ICDF, 2);
                qi = raw >> 1 ^ -(raw & 1);
            } else if budget - tell >= 1 {
                qi = -ec_dec_bit_logp(dec, 1);
            } else {
                qi = -1;
            }
            let q = qi as f32;
            old_ebands[(i + ch * nb_ebands) as usize] =
                (-9.0f32).max(old_ebands[(i + ch * nb_ebands) as usize]);
            let tmp = coef * old_ebands[(i + ch * nb_ebands) as usize] + prev[ch as usize] + q;
            old_ebands[(i + ch * nb_ebands) as usize] = tmp;
            prev[ch as usize] = prev[ch as usize] + q - beta * q;
            ch += 1;
            if ch >= channels {
                break;
            }
        }
    }
}

/// Upstream c: celt/quant_bands.c:unquant_fine_energy
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn unquant_fine_energy(
    m: &OpusCustomMode,
    start: i32,
    end: i32,
    old_ebands: &mut [f32],
    prev_quant: Option<&[i32]>,
    extra_quant: &[i32],
    dec: &mut EcDec,
    channels: i32,
) {
    let nb_ebands = m.nb_ebands as i32;
    for i in start..end {
        let extra = extra_quant[i as usize];
        if extra_quant[i as usize] <= 0 {
            continue;
        }
        if ec_tell(dec) + channels * extra_quant[i as usize] > dec.storage as i32 * 8 {
            continue;
        }
        let prev = prev_quant.map_or(0, |pq| pq[i as usize]);
        let mut ch = 0;
        loop {
            let q2 = ec_dec_bits(dec, extra as u32) as i32;
            let mut offset =
                (q2 as f32 + 0.5f32) * ((1) << (14 - extra)) as f32 * (1.0f32 / 16384.0) - 0.5f32;
            offset *= (1 << (14 - prev)) as f32 * (1.0f32 / 16384.0);
            old_ebands[(i + ch * nb_ebands) as usize] += offset;
            ch += 1;
            if ch >= channels {
                break;
            }
        }
    }
}

/// Upstream c: celt/quant_bands.c:unquant_energy_finalise
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn unquant_energy_finalise(
    m: &OpusCustomMode,
    start: i32,
    end: i32,
    old_ebands: &mut [f32],
    fine_quant: &[i32],
    fine_priority: &[i32],
    mut bits_left: i32,
    dec: &mut EcDec,
    channels: i32,
) {
    let nb_ebands = m.nb_ebands as i32;
    let mut prio = 0;
    while prio < 2 {
        let mut i = start;
        while i < end && bits_left >= channels {
            if !(fine_quant[i as usize] >= MAX_FINE_BITS || fine_priority[i as usize] != prio) {
                let mut ch = 0;
                loop {
                    let q2 = ec_dec_bits(dec, 1) as i32;
                    let offset = (q2 as f32 - 0.5f32)
                        * ((1) << (14 - fine_quant[i as usize] - 1)) as f32
                        * (1.0f32 / 16384.0);
                    old_ebands[(i + ch * nb_ebands) as usize] += offset;
                    bits_left -= 1;
                    ch += 1;
                    if ch >= channels {
                        break;
                    }
                }
            }
            i += 1;
        }
        prio += 1;
    }
}

/// Upstream c: celt/quant_bands.c:amp2Log2
pub fn amp2_log2(
    m: &OpusCustomMode,
    eff_end: i32,
    end: i32,
    band_e: &[f32],
    band_log_e: &mut [f32],
    channels: i32,
) {
    let nb_ebands = m.nb_ebands as i32;
    let mut c = 0;
    loop {
        for i in 0..eff_end {
            band_log_e[(i + c * nb_ebands) as usize] =
                celt_log2(band_e[(i + c * nb_ebands) as usize]) - E_MEANS[i as usize];
        }
        for i in eff_end..end {
            band_log_e[(c * nb_ebands + i) as usize] = -14.0f32;
        }
        c += 1;
        if c >= channels {
            break;
        }
    }
}
