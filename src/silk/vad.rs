//! Voice activity detection.
//!
//! Upstream c: `silk/VAD.c`

#[cfg(not(feature = "simd"))]
use crate::arch::Arch;
use crate::silk::ana_filt_bank_1::silk_ana_filt_bank_1;
use crate::silk::define::{
    VAD_INTERNAL_SUBFRAMES, VAD_NEGATIVE_OFFSET_Q5, VAD_NOISE_LEVEL_SMOOTH_COEF_Q16, VAD_N_BANDS,
};
use crate::silk::inlines::silk_sqrt_approx;
use crate::silk::lin2log::silk_lin2log;
use crate::silk::sigm_q15::silk_sigm_q15;
use crate::silk::sigproc_fix::{silk_max_32, silk_max_int, silk_min_int};
use crate::silk::structs::{silk_VAD_state, silk_encoder_state};
use crate::silk::typedefs::{SILK_INT32_MAX, SILK_UINT8_MAX};

#[cfg(feature = "simd")]
use crate::silk::simd::silk_vad_energy;

/// Scalar VAD energy: sum of (x[_i] >> 3)^2.
#[cfg(not(feature = "simd"))]
fn silk_vad_energy(x: &[i16], _arch: Arch) -> i32 {
    let mut sum: i32 = 0;
    for &sample in x {
        let x_tmp = (sample as i32) >> 3;
        sum += (x_tmp as i16 as i32) * (x_tmp as i16 as i32);
    }
    sum
}

/// Upstream c: silk/VAD.c:silk_VAD_Init
pub fn silk_vad_init(ps_silk_vad: &mut silk_VAD_state) -> i32 {
    let mut b: i32;
    let ret: i32 = 0;
    *ps_silk_vad = Default::default();
    b = 0;
    while b < VAD_N_BANDS {
        ps_silk_vad.noise_level_bias[b as usize] = silk_max_32(50 / (b + 1), 1);
        b += 1;
    }
    b = 0;
    while b < VAD_N_BANDS {
        ps_silk_vad.nl[b as usize] = 100 * ps_silk_vad.noise_level_bias[b as usize];
        ps_silk_vad.inv_nl[b as usize] = 0x7fffffff / ps_silk_vad.nl[b as usize];
        b += 1;
    }
    ps_silk_vad.counter = 15;
    b = 0;
    while b < VAD_N_BANDS {
        ps_silk_vad.nrg_ratio_smth_q8[b as usize] = 100 * 256;
        b += 1;
    }
    ret
}
const TILT_WEIGHTS: [i32; 4] = [30000, 6000, -(12000), -(12000)];
/// Upstream c: silk/VAD.c:silk_VAD_GetSA_Q8_c
pub fn silk_vad_get_sa_q8_c(ps_enc_c: &mut silk_encoder_state, p_in: &[i16]) -> i32 {
    let mut sa_q15: i32;

    let mut input_tilt: i32;

    let mut decimated_framelength: i32;
    let mut dec_subframe_length: i32;
    let mut dec_subframe_offset: i32;
    let mut snr_q7: i32;
    let mut _i: i32;
    let mut b: i32;
    let mut s: i32;
    let mut sum_squared: i32 = 0;
    let mut smooth_coef_q16: i32;

    let mut xnrg: [i32; 4] = [0; 4];
    let mut nrg_to_noise_ratio_q8: [i32; 4] = [0; 4];
    let mut speech_nrg: i32;
    let mut x_offset: [i32; 4] = [0; 4];
    let ret: i32 = 0;
    let ps_silk_vad: &mut silk_VAD_state = &mut ps_enc_c.s_vad;
    debug_assert!(5 * 4 * 16 >= ps_enc_c.frame_length);
    debug_assert!(ps_enc_c.frame_length <= 512);
    debug_assert!(ps_enc_c.frame_length == 8 * (ps_enc_c.frame_length >> 3));
    let decimated_framelength1: i32 = ps_enc_c.frame_length as i32 / 2;
    let decimated_framelength2: i32 = ps_enc_c.frame_length as i32 / 4;
    decimated_framelength = ps_enc_c.frame_length as i32 / 8;
    x_offset[0_usize] = 0;
    x_offset[1_usize] = decimated_framelength + decimated_framelength2;
    x_offset[2_usize] = x_offset[1_usize] + decimated_framelength;
    x_offset[3_usize] = x_offset[2_usize] + decimated_framelength2;
    let vla = (x_offset[3_usize] + decimated_framelength1) as usize;
    // frame_length <= 512 → vla max = 640
    const MAX_X_VAD: usize = 640;
    debug_assert!(vla <= MAX_X_VAD);
    let mut x = [0i16; MAX_X_VAD];
    // First call: p_in -> x[0..] and x[x_offset[3]..] — no aliasing with input
    {
        let (out_l, rest) = x.split_at_mut(x_offset[3] as usize);
        silk_ana_filt_bank_1(
            &p_in[..ps_enc_c.frame_length],
            &mut ps_silk_vad.ana_state,
            out_l,
            rest,
            ps_enc_c.frame_length as i32,
        );
    }
    // Second/third calls: in-place decimation. The filter reads in_0[2*k] and
    // in_0[2*k+1] before writing out_l[k], so in-place is safe.
    // Copy input to temp buffer to avoid aliasing.
    {
        // decimated_framelength1 max: 256
        let mut tmp_in = [0i16; 256];
        tmp_in[..decimated_framelength1 as usize]
            .copy_from_slice(&x[..decimated_framelength1 as usize]);
        let (out_l, rest) = x.split_at_mut(x_offset[2] as usize);
        silk_ana_filt_bank_1(
            &tmp_in,
            &mut ps_silk_vad.ana_state1,
            &mut out_l[..decimated_framelength2 as usize],
            &mut rest[..decimated_framelength2 as usize],
            decimated_framelength1,
        );
    }
    {
        // decimated_framelength2 max: 128
        let mut tmp_in = [0i16; 128];
        tmp_in[..decimated_framelength2 as usize]
            .copy_from_slice(&x[..decimated_framelength2 as usize]);
        let (out_l, rest) = x.split_at_mut(x_offset[1] as usize);
        silk_ana_filt_bank_1(
            &tmp_in,
            &mut ps_silk_vad.ana_state2,
            &mut out_l[..decimated_framelength as usize],
            &mut rest[..decimated_framelength as usize],
            decimated_framelength2,
        );
    }
    x[(decimated_framelength - 1) as usize] =
        (x[(decimated_framelength - 1) as usize] as i32 >> 1) as i16;
    let hpstate_tmp: i16 = x[(decimated_framelength - 1) as usize];
    _i = decimated_framelength - 1;
    while _i > 0 {
        x[(_i - 1) as usize] = (x[(_i - 1) as usize] as i32 >> 1) as i16;
        x[_i as usize] = (x[_i as usize] as i32 - x[(_i - 1) as usize] as i32) as i16;
        _i -= 1;
    }
    x[0] = (x[0] as i32 - ps_silk_vad.hpstate as i32) as i16;
    ps_silk_vad.hpstate = hpstate_tmp;
    b = 0;
    while b < VAD_N_BANDS {
        decimated_framelength = ps_enc_c.frame_length as i32 >> silk_min_int(4 - b, 4 - 1);
        dec_subframe_length = decimated_framelength >> 2;
        dec_subframe_offset = 0;
        xnrg[b as usize] = ps_silk_vad.xnrg_subfr[b as usize];
        s = 0;
        while s < VAD_INTERNAL_SUBFRAMES {
            {
                let start = (x_offset[b as usize] + dec_subframe_offset) as usize;
                let end = start + dec_subframe_length as usize;
                sum_squared = silk_vad_energy(&x[start..end], ps_enc_c.arch);
            }
            if s < VAD_INTERNAL_SUBFRAMES - 1 {
                xnrg[b as usize] = if (xnrg[b as usize] as u32).wrapping_add(sum_squared as u32)
                    & 0x80000000_u32
                    != 0
                {
                    SILK_INT32_MAX
                } else {
                    xnrg[b as usize] + sum_squared
                };
            } else {
                xnrg[b as usize] = if (xnrg[b as usize] as u32)
                    .wrapping_add((sum_squared >> 1) as u32)
                    & 0x80000000_u32
                    != 0
                {
                    SILK_INT32_MAX
                } else {
                    xnrg[b as usize] + (sum_squared >> 1)
                };
            }
            dec_subframe_offset += dec_subframe_length;
            s += 1;
        }
        ps_silk_vad.xnrg_subfr[b as usize] = sum_squared;
        b += 1;
    }
    silk_vad_get_noise_levels(&xnrg, ps_silk_vad);
    sum_squared = 0;
    input_tilt = 0;
    b = 0;
    while b < VAD_N_BANDS {
        speech_nrg = xnrg[b as usize] - ps_silk_vad.nl[b as usize];
        if speech_nrg > 0 {
            if xnrg[b as usize] as u32 & 0xff800000_u32 == 0 {
                nrg_to_noise_ratio_q8[b as usize] =
                    ((xnrg[b as usize] as u32) << 8) as i32 / (ps_silk_vad.nl[b as usize] + 1);
            } else {
                nrg_to_noise_ratio_q8[b as usize] =
                    xnrg[b as usize] / ((ps_silk_vad.nl[b as usize] >> 8) + 1);
            }
            snr_q7 = silk_lin2log(nrg_to_noise_ratio_q8[b as usize]) - 8 * 128;
            sum_squared += snr_q7 as i16 as i32 * snr_q7 as i16 as i32;
            if speech_nrg < (1) << 20 {
                snr_q7 = ((((silk_sqrt_approx(speech_nrg) as u32) << 6) as i32 as i64
                    * snr_q7 as i16 as i64)
                    >> 16) as i32;
            }
            input_tilt = (input_tilt as i64
                + ((TILT_WEIGHTS[b as usize] as i64 * snr_q7 as i16 as i64) >> 16))
                as i32;
        } else {
            nrg_to_noise_ratio_q8[b as usize] = 256;
        }
        b += 1;
    }
    sum_squared /= 4;
    let p_snr_d_b_q7: i32 = (3 * silk_sqrt_approx(sum_squared)) as i16 as i32;
    sa_q15 =
        silk_sigm_q15(((45000 * p_snr_d_b_q7 as i16 as i64) >> 16) as i32 - VAD_NEGATIVE_OFFSET_Q5);
    ps_enc_c.input_tilt_q15 = (((silk_sigm_q15(input_tilt) - 16384) as u32) << 1) as i32;
    speech_nrg = 0;
    b = 0;
    while b < VAD_N_BANDS {
        speech_nrg += (b + 1) * ((xnrg[b as usize] - ps_silk_vad.nl[b as usize]) >> 4);
        b += 1;
    }
    if ps_enc_c.frame_length as i32 == 20 * ps_enc_c.fs_k_hz {
        speech_nrg >>= 1;
    }
    if speech_nrg <= 0 {
        sa_q15 >>= 1;
    } else if speech_nrg < 16384 {
        speech_nrg = ((speech_nrg as u32) << 16) as i32;
        speech_nrg = silk_sqrt_approx(speech_nrg);
        sa_q15 = (((32768 + speech_nrg) as i64 * sa_q15 as i16 as i64) >> 16) as i32;
    }
    ps_enc_c.speech_activity_q8 = silk_min_int(sa_q15 >> 7, SILK_UINT8_MAX);
    smooth_coef_q16 =
        ((4096 * ((sa_q15 as i64 * sa_q15 as i16 as i64) >> 16) as i32 as i16 as i64) >> 16) as i32;
    if ps_enc_c.frame_length as i32 == 10 * ps_enc_c.fs_k_hz {
        smooth_coef_q16 >>= 1;
    }
    b = 0;
    while b < VAD_N_BANDS {
        ps_silk_vad.nrg_ratio_smth_q8[b as usize] = (ps_silk_vad.nrg_ratio_smth_q8[b as usize]
            as i64
            + (((nrg_to_noise_ratio_q8[b as usize] - ps_silk_vad.nrg_ratio_smth_q8[b as usize])
                as i64
                * smooth_coef_q16 as i16 as i64)
                >> 16)) as i32;
        snr_q7 = 3 * (silk_lin2log(ps_silk_vad.nrg_ratio_smth_q8[b as usize]) - 8 * 128);
        ps_enc_c.input_quality_bands_q15[b as usize] = silk_sigm_q15((snr_q7 - 16 * 128) >> 4);
        b += 1;
    }
    ret
}

/// Dispatch wrapper for VAD speech activity, matching upstream `silk_vad_get_sa_q8`.
#[cfg(feature = "simd")]
#[inline]
pub fn silk_vad_get_sa_q8(ps_enc_c: &mut silk_encoder_state, p_in: &[i16]) -> i32 {
    super::simd::silk_vad_get_sa_q8(ps_enc_c, p_in)
}

/// Scalar-only build wrapper for VAD speech activity.
#[cfg(not(feature = "simd"))]
#[inline]
pub fn silk_vad_get_sa_q8(ps_enc_c: &mut silk_encoder_state, p_in: &[i16]) -> i32 {
    silk_vad_get_sa_q8_c(ps_enc_c, p_in)
}

/// Upstream c: silk/VAD.c:silk_VAD_GetNoiseLevels
#[inline]
fn silk_vad_get_noise_levels(p_x: &[i32; 4], ps_silk_vad: &mut silk_VAD_state) {
    let mut k: i32;
    let mut nl: i32;
    let mut nrg: i32;
    let mut inv_nrg: i32;
    let mut coef: i32;
    let min_coef: i32;
    if ps_silk_vad.counter < 1000 {
        min_coef = 0x7fff / ((ps_silk_vad.counter >> 4) + 1);
        ps_silk_vad.counter += 1;
    } else {
        min_coef = 0;
    }
    k = 0;
    while k < VAD_N_BANDS {
        nl = ps_silk_vad.nl[k as usize];
        nrg = if (p_x[k as usize] as u32)
            .wrapping_add(ps_silk_vad.noise_level_bias[k as usize] as u32)
            & 0x80000000_u32
            != 0
        {
            SILK_INT32_MAX
        } else {
            p_x[k as usize] + ps_silk_vad.noise_level_bias[k as usize]
        };
        inv_nrg = 0x7fffffff / nrg;
        if nrg > ((nl as u32) << 3) as i32 {
            coef = VAD_NOISE_LEVEL_SMOOTH_COEF_Q16 >> 3;
        } else if nrg < nl {
            coef = VAD_NOISE_LEVEL_SMOOTH_COEF_Q16;
        } else {
            coef = ((((inv_nrg as i64 * nl as i64) >> 16) as i32 as i64
                * ((1024) << 1) as i16 as i64)
                >> 16) as i32;
        }
        coef = silk_max_int(coef, min_coef);
        ps_silk_vad.inv_nl[k as usize] = (ps_silk_vad.inv_nl[k as usize] as i64
            + (((inv_nrg - ps_silk_vad.inv_nl[k as usize]) as i64 * coef as i16 as i64) >> 16))
            as i32;
        nl = 0x7fffffff / ps_silk_vad.inv_nl[k as usize];
        nl = if nl < 0xffffff { nl } else { 0xffffff };
        ps_silk_vad.nl[k as usize] = nl;
        k += 1;
    }
}
