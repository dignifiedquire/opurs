//! Voice activity detection.
//!
//! Upstream C: `silk/VAD.c`

pub mod typedef_h {
    pub const silk_uint8_MAX: i32 = 0xff;
    pub const silk_int32_MAX: i32 = i32::MAX;
}
pub use self::typedef_h::{silk_int32_MAX, silk_uint8_MAX};
#[cfg(not(feature = "simd"))]
use crate::arch::Arch;
use crate::silk::ana_filt_bank_1::silk_ana_filt_bank_1;
use crate::silk::define::{
    VAD_INTERNAL_SUBFRAMES, VAD_NEGATIVE_OFFSET_Q5, VAD_NOISE_LEVEL_SMOOTH_COEF_Q16, VAD_N_BANDS,
};
use crate::silk::lin2log::silk_lin2log;
use crate::silk::sigm_Q15::silk_sigm_Q15;
use crate::silk::structs::{silk_VAD_state, silk_encoder_state};
use crate::silk::Inlines::silk_SQRT_APPROX;
use crate::silk::SigProc_FIX::{silk_max_32, silk_max_int, silk_min_int};

#[cfg(feature = "simd")]
use crate::silk::simd::silk_vad_energy;

/// Scalar VAD energy: sum of (X[i] >> 3)^2.
#[cfg(not(feature = "simd"))]
fn silk_vad_energy(x: &[i16], _arch: Arch) -> i32 {
    let mut sum: i32 = 0;
    for &sample in x {
        let x_tmp = (sample as i32) >> 3;
        sum += (x_tmp as i16 as i32) * (x_tmp as i16 as i32);
    }
    sum
}

/// Upstream C: silk/VAD.c:silk_VAD_Init
pub fn silk_VAD_Init(psSilk_VAD: &mut silk_VAD_state) -> i32 {
    let mut b: i32 = 0;
    let ret: i32 = 0;
    *psSilk_VAD = Default::default();
    b = 0;
    while b < VAD_N_BANDS {
        psSilk_VAD.NoiseLevelBias[b as usize] = silk_max_32(50 / (b + 1), 1);
        b += 1;
    }
    b = 0;
    while b < VAD_N_BANDS {
        psSilk_VAD.NL[b as usize] = 100 * psSilk_VAD.NoiseLevelBias[b as usize];
        psSilk_VAD.inv_NL[b as usize] = 0x7fffffff / psSilk_VAD.NL[b as usize];
        b += 1;
    }
    psSilk_VAD.counter = 15;
    b = 0;
    while b < VAD_N_BANDS {
        psSilk_VAD.NrgRatioSmth_Q8[b as usize] = 100 * 256;
        b += 1;
    }
    ret
}
static TILT_WEIGHTS: [i32; 4] = [30000, 6000, -(12000), -(12000)];
/// Upstream C: silk/VAD.c:silk_VAD_GetSA_Q8_c
pub fn silk_VAD_GetSA_Q8_c(psEncC: &mut silk_encoder_state, pIn: &[i16]) -> i32 {
    let mut SA_Q15: i32 = 0;
    let mut pSNR_dB_Q7: i32 = 0;
    let mut input_tilt: i32 = 0;
    let mut decimated_framelength1: i32 = 0;
    let mut decimated_framelength2: i32 = 0;
    let mut decimated_framelength: i32 = 0;
    let mut dec_subframe_length: i32 = 0;
    let mut dec_subframe_offset: i32 = 0;
    let mut SNR_Q7: i32 = 0;
    let mut i: i32 = 0;
    let mut b: i32 = 0;
    let mut s: i32 = 0;
    let mut sumSquared: i32 = 0;
    let mut smooth_coef_Q16: i32 = 0;
    let mut HPstateTmp: i16 = 0;
    let mut Xnrg: [i32; 4] = [0; 4];
    let mut NrgToNoiseRatio_Q8: [i32; 4] = [0; 4];
    let mut speech_nrg: i32 = 0;
    let mut X_offset: [i32; 4] = [0; 4];
    let ret: i32 = 0;
    let psSilk_VAD: &mut silk_VAD_state = &mut psEncC.sVAD;
    debug_assert!(5 * 4 * 16 >= psEncC.frame_length);
    debug_assert!(psEncC.frame_length <= 512);
    debug_assert!(psEncC.frame_length == 8 * (psEncC.frame_length >> 3));
    decimated_framelength1 = psEncC.frame_length as i32 / 2;
    decimated_framelength2 = psEncC.frame_length as i32 / 4;
    decimated_framelength = psEncC.frame_length as i32 / 8;
    X_offset[0_usize] = 0;
    X_offset[1_usize] = decimated_framelength + decimated_framelength2;
    X_offset[2_usize] = X_offset[1_usize] + decimated_framelength;
    X_offset[3_usize] = X_offset[2_usize] + decimated_framelength2;
    let vla = (X_offset[3_usize] + decimated_framelength1) as usize;
    let mut X: Vec<i16> = ::std::vec::from_elem(0, vla);
    // First call: pIn -> X[0..] and X[X_offset[3]..] — no aliasing with input
    {
        let (outL, rest) = X.split_at_mut(X_offset[3] as usize);
        silk_ana_filt_bank_1(
            &pIn[..psEncC.frame_length],
            &mut psSilk_VAD.AnaState,
            outL,
            rest,
            psEncC.frame_length as i32,
        );
    }
    // Second/third calls: in-place decimation. The filter reads in_0[2*k] and
    // in_0[2*k+1] before writing outL[k], so in-place is safe.
    // Copy input to temp buffer to avoid aliasing.
    {
        let mut tmp_in = vec![0i16; decimated_framelength1 as usize];
        tmp_in.copy_from_slice(&X[..decimated_framelength1 as usize]);
        let (outL, rest) = X.split_at_mut(X_offset[2] as usize);
        silk_ana_filt_bank_1(
            &tmp_in,
            &mut psSilk_VAD.AnaState1,
            &mut outL[..decimated_framelength2 as usize],
            &mut rest[..decimated_framelength2 as usize],
            decimated_framelength1,
        );
    }
    {
        let mut tmp_in = vec![0i16; decimated_framelength2 as usize];
        tmp_in.copy_from_slice(&X[..decimated_framelength2 as usize]);
        let (outL, rest) = X.split_at_mut(X_offset[1] as usize);
        silk_ana_filt_bank_1(
            &tmp_in,
            &mut psSilk_VAD.AnaState2,
            &mut outL[..decimated_framelength as usize],
            &mut rest[..decimated_framelength as usize],
            decimated_framelength2,
        );
    }
    X[(decimated_framelength - 1) as usize] =
        (X[(decimated_framelength - 1) as usize] as i32 >> 1) as i16;
    HPstateTmp = X[(decimated_framelength - 1) as usize];
    i = decimated_framelength - 1;
    while i > 0 {
        X[(i - 1) as usize] = (X[(i - 1) as usize] as i32 >> 1) as i16;
        X[i as usize] = (X[i as usize] as i32 - X[(i - 1) as usize] as i32) as i16;
        i -= 1;
    }
    X[0] = (X[0] as i32 - psSilk_VAD.HPstate as i32) as i16;
    psSilk_VAD.HPstate = HPstateTmp;
    b = 0;
    while b < VAD_N_BANDS {
        decimated_framelength = psEncC.frame_length as i32 >> silk_min_int(4 - b, 4 - 1);
        dec_subframe_length = decimated_framelength >> 2;
        dec_subframe_offset = 0;
        Xnrg[b as usize] = psSilk_VAD.XnrgSubfr[b as usize];
        s = 0;
        while s < VAD_INTERNAL_SUBFRAMES {
            {
                let start = (X_offset[b as usize] + dec_subframe_offset) as usize;
                let end = start + dec_subframe_length as usize;
                sumSquared = silk_vad_energy(&X[start..end], psEncC.arch);
            }
            if s < VAD_INTERNAL_SUBFRAMES - 1 {
                Xnrg[b as usize] = if (Xnrg[b as usize] as u32).wrapping_add(sumSquared as u32)
                    & 0x80000000_u32
                    != 0
                {
                    silk_int32_MAX
                } else {
                    Xnrg[b as usize] + sumSquared
                };
            } else {
                Xnrg[b as usize] = if (Xnrg[b as usize] as u32)
                    .wrapping_add((sumSquared >> 1) as u32)
                    & 0x80000000_u32
                    != 0
                {
                    silk_int32_MAX
                } else {
                    Xnrg[b as usize] + (sumSquared >> 1)
                };
            }
            dec_subframe_offset += dec_subframe_length;
            s += 1;
        }
        psSilk_VAD.XnrgSubfr[b as usize] = sumSquared;
        b += 1;
    }
    silk_VAD_GetNoiseLevels(&Xnrg, psSilk_VAD);
    sumSquared = 0;
    input_tilt = 0;
    b = 0;
    while b < VAD_N_BANDS {
        speech_nrg = Xnrg[b as usize] - psSilk_VAD.NL[b as usize];
        if speech_nrg > 0 {
            if Xnrg[b as usize] as u32 & 0xff800000_u32 == 0 {
                NrgToNoiseRatio_Q8[b as usize] =
                    ((Xnrg[b as usize] as u32) << 8) as i32 / (psSilk_VAD.NL[b as usize] + 1);
            } else {
                NrgToNoiseRatio_Q8[b as usize] =
                    Xnrg[b as usize] / ((psSilk_VAD.NL[b as usize] >> 8) + 1);
            }
            SNR_Q7 = silk_lin2log(NrgToNoiseRatio_Q8[b as usize]) - 8 * 128;
            sumSquared += SNR_Q7 as i16 as i32 * SNR_Q7 as i16 as i32;
            if speech_nrg < (1) << 20 {
                SNR_Q7 = ((((silk_SQRT_APPROX(speech_nrg) as u32) << 6) as i32 as i64
                    * SNR_Q7 as i16 as i64)
                    >> 16) as i32;
            }
            input_tilt = (input_tilt as i64
                + ((TILT_WEIGHTS[b as usize] as i64 * SNR_Q7 as i16 as i64) >> 16))
                as i32;
        } else {
            NrgToNoiseRatio_Q8[b as usize] = 256;
        }
        b += 1;
    }
    sumSquared /= 4;
    pSNR_dB_Q7 = (3 * silk_SQRT_APPROX(sumSquared)) as i16 as i32;
    SA_Q15 =
        silk_sigm_Q15(((45000 * pSNR_dB_Q7 as i16 as i64) >> 16) as i32 - VAD_NEGATIVE_OFFSET_Q5);
    psEncC.input_tilt_Q15 = (((silk_sigm_Q15(input_tilt) - 16384) as u32) << 1) as i32;
    speech_nrg = 0;
    b = 0;
    while b < VAD_N_BANDS {
        speech_nrg += (b + 1) * ((Xnrg[b as usize] - psSilk_VAD.NL[b as usize]) >> 4);
        b += 1;
    }
    if psEncC.frame_length as i32 == 20 * psEncC.fs_kHz {
        speech_nrg >>= 1;
    }
    if speech_nrg <= 0 {
        SA_Q15 >>= 1;
    } else if speech_nrg < 16384 {
        speech_nrg = ((speech_nrg as u32) << 16) as i32;
        speech_nrg = silk_SQRT_APPROX(speech_nrg);
        SA_Q15 = (((32768 + speech_nrg) as i64 * SA_Q15 as i16 as i64) >> 16) as i32;
    }
    psEncC.speech_activity_Q8 = silk_min_int(SA_Q15 >> 7, silk_uint8_MAX);
    smooth_coef_Q16 =
        ((4096 * ((SA_Q15 as i64 * SA_Q15 as i16 as i64) >> 16) as i32 as i16 as i64) >> 16) as i32;
    if psEncC.frame_length as i32 == 10 * psEncC.fs_kHz {
        smooth_coef_Q16 >>= 1;
    }
    b = 0;
    while b < VAD_N_BANDS {
        psSilk_VAD.NrgRatioSmth_Q8[b as usize] = (psSilk_VAD.NrgRatioSmth_Q8[b as usize] as i64
            + (((NrgToNoiseRatio_Q8[b as usize] - psSilk_VAD.NrgRatioSmth_Q8[b as usize]) as i64
                * smooth_coef_Q16 as i16 as i64)
                >> 16)) as i32;
        SNR_Q7 = 3 * (silk_lin2log(psSilk_VAD.NrgRatioSmth_Q8[b as usize]) - 8 * 128);
        psEncC.input_quality_bands_Q15[b as usize] = silk_sigm_Q15((SNR_Q7 - 16 * 128) >> 4);
        b += 1;
    }
    ret
}

/// Dispatch wrapper for VAD speech activity, matching upstream `silk_VAD_GetSA_Q8`.
#[cfg(feature = "simd")]
#[inline]
pub fn silk_VAD_GetSA_Q8(psEncC: &mut silk_encoder_state, pIn: &[i16]) -> i32 {
    super::simd::silk_VAD_GetSA_Q8(psEncC, pIn)
}

/// Scalar-only build wrapper for VAD speech activity.
#[cfg(not(feature = "simd"))]
#[inline]
pub fn silk_VAD_GetSA_Q8(psEncC: &mut silk_encoder_state, pIn: &[i16]) -> i32 {
    silk_VAD_GetSA_Q8_c(psEncC, pIn)
}

/// Upstream C: silk/VAD.c:silk_VAD_GetNoiseLevels
#[inline]
fn silk_VAD_GetNoiseLevels(pX: &[i32; 4], psSilk_VAD: &mut silk_VAD_state) {
    let mut k: i32 = 0;
    let mut nl: i32 = 0;
    let mut nrg: i32 = 0;
    let mut inv_nrg: i32 = 0;
    let mut coef: i32 = 0;
    let mut min_coef: i32 = 0;
    if psSilk_VAD.counter < 1000 {
        min_coef = 0x7fff / ((psSilk_VAD.counter >> 4) + 1);
        psSilk_VAD.counter += 1;
    } else {
        min_coef = 0;
    }
    k = 0;
    while k < VAD_N_BANDS {
        nl = psSilk_VAD.NL[k as usize];
        nrg = if (pX[k as usize] as u32).wrapping_add(psSilk_VAD.NoiseLevelBias[k as usize] as u32)
            & 0x80000000_u32
            != 0
        {
            silk_int32_MAX
        } else {
            pX[k as usize] + psSilk_VAD.NoiseLevelBias[k as usize]
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
        psSilk_VAD.inv_NL[k as usize] = (psSilk_VAD.inv_NL[k as usize] as i64
            + (((inv_nrg - psSilk_VAD.inv_NL[k as usize]) as i64 * coef as i16 as i64) >> 16))
            as i32;
        nl = 0x7fffffff / psSilk_VAD.inv_NL[k as usize];
        nl = if nl < 0xffffff { nl } else { 0xffffff };
        psSilk_VAD.NL[k as usize] = nl;
        k += 1;
    }
}
