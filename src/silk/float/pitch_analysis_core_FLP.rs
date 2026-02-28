//! Floating-point pitch analysis core.
//!
//! Upstream C: `silk/float/pitch_analysis_core_FLP.c`

use crate::arch::Arch;

pub mod arch_h {
    pub type opus_val32 = f32;
}
pub mod typedef_h {
    pub const silk_int16_MAX: i32 = i16::MAX as i32;
    pub const silk_int16_MIN: i32 = i16::MIN as i32;
}

use self::arch_h::opus_val32;
pub use self::typedef_h::{silk_int16_MAX, silk_int16_MIN};
use crate::celt::pitch::celt_pitch_xcorr;
use crate::silk::float::energy_FLP::silk_energy_FLP;
use crate::silk::float::inner_product_FLP::silk_inner_product_FLP;
use crate::silk::float::sort_FLP::silk_insertion_sort_decreasing_FLP;
use crate::silk::float::SigProc_FLP::{silk_float2short_array, silk_log2, silk_short2float_array};
use crate::silk::pitch_est_tables::{
    silk_CB_lags_stage2, silk_CB_lags_stage2_10_ms, silk_CB_lags_stage3, silk_CB_lags_stage3_10_ms,
    silk_Lag_range_stage3, silk_Lag_range_stage3_10_ms, silk_nb_cbk_searchs_stage3,
    PE_FLATCONTOUR_BIAS, PE_LTP_MEM_LENGTH_MS, PE_MAX_LAG_MS, PE_MAX_NB_SUBFR, PE_MIN_LAG_MS,
    PE_NB_CBKS_STAGE2, PE_NB_CBKS_STAGE2_10MS, PE_NB_CBKS_STAGE2_EXT, PE_NB_CBKS_STAGE3_10MS,
    PE_NB_CBKS_STAGE3_MAX, PE_NB_STAGE3_LAGS, PE_PREVLAG_BIAS, PE_SHORTLAG_BIAS,
    PE_SUBFR_LENGTH_MS, SILK_PE_MIN_COMPLEX,
};
use crate::silk::resampler::{silk_resampler_down2, silk_resampler_down2_3};
use crate::silk::SigProc_FIX::{silk_max_int, silk_min_int};
use arrayref::array_mut_ref;

/// Upstream C: silk/float/pitch_analysis_core_FLP.c:silk_pitch_analysis_core_FLP
pub fn silk_pitch_analysis_core_FLP(
    frame: &[f32],
    pitch_out: &mut [i32],
    lagIndex: &mut i16,
    contourIndex: &mut i8,
    LTPCorr: &mut f32,
    mut prevLag: i32,
    search_thres1: f32,
    search_thres2: f32,
    Fs_kHz: i32,
    complexity: i32,
    nb_subfr: i32,
    arch: Arch,
) -> i32 {
    let mut i: i32;
    let mut k: i32;
    let mut d: i32;
    let mut j: i32;
    let mut frame_8kHz: [f32; 320] = [0.; 320];
    let mut frame_4kHz: [f32; 160] = [0.; 160];
    let mut frame_8_FIX: [i16; 320] = [0; 320];
    let mut frame_4_FIX: [i16; 160] = [0; 160];
    let mut filt_state: [i32; 6] = [0; 6];

    let contour_bias: f32;
    let mut C: [[f32; 149]; 4] = [[0.; 149]; 4];
    let mut xcorr: [opus_val32; 65] = [0.; 65];
    let mut CC: [f32; 11] = [0.; 11];
    let mut cross_corr: f64;
    let mut normalizer: f64;
    let mut energy: f64;
    let mut energy_tmp: f64;
    let mut d_srch: [i32; 24] = [0; 24];
    let mut d_comp: [i16; 149] = [0; 149];
    let mut length_d_srch: i32;
    let mut length_d_comp: i32;

    let mut CCmax: f32;
    let mut CCmax_b: f32;
    let mut CCmax_new_b: f32;
    let mut CCmax_new: f32;
    let mut CBimax: i32;
    let mut CBimax_new: i32;
    let mut lag: i32;
    let start_lag: i32;
    let end_lag: i32;
    let mut lag_new: i32;
    let mut cbk_size: i32;
    let mut lag_log2: f32;
    let prevLag_log2: f32;
    let mut delta_lag_log2_sqr: f32;
    let mut energies_st3: [[[f32; 5]; 34]; 4] = [[[0.; 5]; 34]; 4];
    let mut cross_corr_st3: [[[f32; 5]; 34]; 4] = [[[0.; 5]; 34]; 4];
    let mut lag_counter: i32;

    let mut nb_cbk_search: i32;
    let Lag_CB: &[i8];
    debug_assert!(Fs_kHz == 8 || Fs_kHz == 12 || Fs_kHz == 16);
    debug_assert!(complexity >= 0);
    debug_assert!(complexity <= 2);
    let frame_length: i32 = (PE_LTP_MEM_LENGTH_MS + nb_subfr * PE_SUBFR_LENGTH_MS) * Fs_kHz;
    let frame_length_4kHz: i32 = (PE_LTP_MEM_LENGTH_MS + nb_subfr * PE_SUBFR_LENGTH_MS) * 4;
    let frame_length_8kHz: i32 = (PE_LTP_MEM_LENGTH_MS + nb_subfr * PE_SUBFR_LENGTH_MS) * 8;
    let sf_length: i32 = PE_SUBFR_LENGTH_MS * Fs_kHz;
    let sf_length_4kHz: i32 = PE_SUBFR_LENGTH_MS * 4;
    let sf_length_8kHz: i32 = PE_SUBFR_LENGTH_MS * 8;
    let min_lag: i32 = PE_MIN_LAG_MS * Fs_kHz;
    let min_lag_4kHz: i32 = PE_MIN_LAG_MS * 4;
    let min_lag_8kHz: i32 = PE_MIN_LAG_MS * 8;
    let max_lag: i32 = PE_MAX_LAG_MS * Fs_kHz - 1;
    let max_lag_4kHz: i32 = PE_MAX_LAG_MS * 4;
    let max_lag_8kHz: i32 = PE_MAX_LAG_MS * 8 - 1;
    if Fs_kHz == 16 {
        let mut frame_16_FIX: [i16; 640] = [0; 640];
        silk_float2short_array(
            &mut frame_16_FIX[..frame_length as usize],
            &frame[..frame_length as usize],
        );
        let filt_state = array_mut_ref![filt_state, 0, 2];
        filt_state.fill(0);
        silk_resampler_down2(
            filt_state,
            &mut frame_8_FIX[..frame_length_8kHz as usize],
            &frame_16_FIX[..frame_length as usize],
        );
        silk_short2float_array(
            &mut frame_8kHz[..frame_length_8kHz as usize],
            &frame_8_FIX[..frame_length_8kHz as usize],
        );
    } else if Fs_kHz == 12 {
        let mut frame_12_FIX: [i16; 480] = [0; 480];
        silk_float2short_array(
            &mut frame_12_FIX[..frame_length as usize],
            &frame[..frame_length as usize],
        );
        filt_state.fill(0);
        silk_resampler_down2_3(
            &mut filt_state,
            &mut frame_8_FIX[..frame_length_8kHz as usize],
            &frame_12_FIX[..frame_length as usize],
        );
        silk_short2float_array(
            &mut frame_8kHz[..frame_length_8kHz as usize],
            &frame_8_FIX[..frame_length_8kHz as usize],
        );
    } else {
        debug_assert!(Fs_kHz == 8);
        silk_float2short_array(
            &mut frame_8_FIX[..frame_length_8kHz as usize],
            &frame[..frame_length_8kHz as usize],
        );
    }
    {
        let filt_state = array_mut_ref![filt_state, 0, 2];
        filt_state.fill(0);
        silk_resampler_down2(
            filt_state,
            &mut frame_4_FIX[..frame_length_4kHz as usize],
            &frame_8_FIX[..frame_length_8kHz as usize],
        );
    }
    silk_short2float_array(
        &mut frame_4kHz[..frame_length_4kHz as usize],
        &frame_4_FIX[..frame_length_4kHz as usize],
    );
    debug_assert!(frame_length_4kHz as usize <= frame_4kHz.len());
    i = frame_length_4kHz - 1;
    while i > 0 {
        // Safety: i in [1, frame_length_4kHz-1] and frame_4kHz.len() >= frame_length_4kHz
        let cur = unsafe { *frame_4kHz.get_unchecked(i as usize) };
        let prev = unsafe { *frame_4kHz.get_unchecked((i - 1) as usize) };
        let sum = cur as i32 as f32 + prev;
        unsafe {
            *frame_4kHz.get_unchecked_mut(i as usize) = (if sum > silk_int16_MAX as f32 {
                silk_int16_MAX as f32
            } else if sum < silk_int16_MIN as f32 {
                silk_int16_MIN as f32
            } else {
                sum
            }) as i16 as f32;
        }
        i -= 1;
    }
    // C is already zero-initialized above
    // target_off tracks position in frame_4kHz
    let mut target_off: usize = ((sf_length_4kHz as u32) << 2) as usize;
    k = 0;
    while k < nb_subfr >> 1 {
        debug_assert!(target_off + sf_length_8kHz as usize <= frame_length_4kHz as usize);
        let basis_off = target_off - min_lag_4kHz as usize;
        debug_assert!(basis_off + sf_length_8kHz as usize <= frame_length_4kHz as usize);
        {
            let xcorr_len = (max_lag_4kHz - min_lag_4kHz + 1) as usize;
            celt_pitch_xcorr(
                &frame_4kHz[target_off..target_off + sf_length_8kHz as usize],
                &frame_4kHz[target_off - max_lag_4kHz as usize
                    ..target_off - max_lag_4kHz as usize + sf_length_8kHz as usize + xcorr_len],
                &mut xcorr[..xcorr_len],
                sf_length_8kHz as usize,
                arch,
            );
        }
        // Safety: (max_lag_4kHz - min_lag_4kHz) < xcorr.len() == 65
        cross_corr =
            (unsafe { *xcorr.get_unchecked((max_lag_4kHz - min_lag_4kHz) as usize) }) as f64;
        normalizer = silk_energy_FLP(&frame_4kHz[target_off..target_off + sf_length_8kHz as usize])
            + silk_energy_FLP(&frame_4kHz[basis_off..basis_off + sf_length_8kHz as usize])
            + (sf_length_8kHz as f32 * 4000.0f32) as f64;
        // Safety: min_lag_4kHz < 149 (C dimension)
        let tmp = unsafe { *C[0].get_unchecked(min_lag_4kHz as usize) };
        unsafe {
            *C[0].get_unchecked_mut(min_lag_4kHz as usize) =
                tmp + (2_f64 * cross_corr / normalizer) as f32;
        }
        // basis_off_d starts at basis_off and decrements
        let mut basis_off_d = basis_off;
        d = min_lag_4kHz + 1;
        while d <= max_lag_4kHz {
            basis_off_d -= 1;
            // Safety: (max_lag_4kHz - d) in [0, max_lag_4kHz - min_lag_4kHz)
            cross_corr = (unsafe { *xcorr.get_unchecked((max_lag_4kHz - d) as usize) }) as f64;
            // Safety: basis_off_d and basis_off_d + sf_length_8kHz are within frame_4kHz bounds
            let lo = unsafe { *frame_4kHz.get_unchecked(basis_off_d) } as f64;
            let hi =
                unsafe { *frame_4kHz.get_unchecked(basis_off_d + sf_length_8kHz as usize) } as f64;
            normalizer += lo * lo - hi * hi;
            // Safety: d in [min_lag_4kHz+1, max_lag_4kHz], d < 149
            let tmp = unsafe { *C[0].get_unchecked(d as usize) };
            unsafe {
                *C[0].get_unchecked_mut(d as usize) =
                    tmp + (2_f64 * cross_corr / normalizer) as f32;
            }
            d += 1;
        }
        target_off += sf_length_8kHz as usize;
        k += 1;
    }
    debug_assert!((max_lag_4kHz as usize) < C[0].len());
    i = max_lag_4kHz;
    while i >= min_lag_4kHz {
        // Safety: i in [min_lag_4kHz, max_lag_4kHz], both < 149
        let val = unsafe { *C[0].get_unchecked(i as usize) };
        unsafe {
            *C[0].get_unchecked_mut(i as usize) = val - val * i as f32 / 4096.0f32;
        }
        i -= 1;
    }
    length_d_srch = 4 + 2 * complexity;
    debug_assert!(3 * length_d_srch <= 24);
    silk_insertion_sort_decreasing_FLP(
        &mut C[0][min_lag_4kHz as usize..],
        &mut d_srch,
        max_lag_4kHz - min_lag_4kHz + 1,
        length_d_srch,
    );
    let Cmax: f32 = C[0][min_lag_4kHz as usize];
    if Cmax < 0.2f32 {
        pitch_out[..nb_subfr as usize].fill(0);
        *LTPCorr = 0.0f32;
        *lagIndex = 0;
        *contourIndex = 0;
        return 1;
    }
    let threshold: f32 = search_thres1 * Cmax;
    i = 0;
    while i < length_d_srch {
        // Safety: i < length_d_srch <= 24, (min_lag_4kHz + i) < 149
        if (unsafe { *C[0_usize].get_unchecked((min_lag_4kHz + i) as usize) }) > threshold {
            unsafe {
                let prev = *d_srch.get_unchecked(i as usize);
                *d_srch.get_unchecked_mut(i as usize) =
                    (((prev + min_lag_4kHz) as u32) << 1) as i32;
            }
            i += 1;
        } else {
            length_d_srch = i;
            break;
        }
    }
    debug_assert!(length_d_srch > 0);
    i = min_lag_8kHz - 5;
    while i < max_lag_8kHz + 5 {
        // Safety: i in [min_lag_8kHz-5, max_lag_8kHz+4], all < 149
        unsafe {
            *d_comp.get_unchecked_mut(i as usize) = 0;
        }
        i += 1;
    }
    i = 0;
    while i < length_d_srch {
        // Safety: i < length_d_srch <= 24, d_srch values < 149
        unsafe {
            let d_val = *d_srch.get_unchecked(i as usize);
            *d_comp.get_unchecked_mut(d_val as usize) = 1;
        }
        i += 1;
    }
    i = max_lag_8kHz + 3;
    while i >= min_lag_8kHz {
        // Safety: i in [min_lag_8kHz, max_lag_8kHz+3], (i-2) >= min_lag_8kHz-2 >= 0
        unsafe {
            *d_comp.get_unchecked_mut(i as usize) = (*d_comp.get_unchecked(i as usize) as i32
                + (*d_comp.get_unchecked((i - 1) as usize) as i32
                    + *d_comp.get_unchecked((i - 2) as usize) as i32))
                as i16;
        }
        i -= 1;
    }
    length_d_srch = 0;
    i = min_lag_8kHz;
    while i < max_lag_8kHz + 1 {
        // Safety: (i+1) <= max_lag_8kHz+1 < 149, length_d_srch < 24
        if (unsafe { *d_comp.get_unchecked((i + 1) as usize) }) as i32 > 0 {
            unsafe {
                *d_srch.get_unchecked_mut(length_d_srch as usize) = i;
            }
            length_d_srch += 1;
        }
        i += 1;
    }
    i = max_lag_8kHz + 3;
    while i >= min_lag_8kHz {
        // Safety: i in [min_lag_8kHz, max_lag_8kHz+3], (i-3) >= min_lag_8kHz-3 >= 0
        unsafe {
            *d_comp.get_unchecked_mut(i as usize) = (*d_comp.get_unchecked(i as usize) as i32
                + (*d_comp.get_unchecked((i - 1) as usize) as i32
                    + *d_comp.get_unchecked((i - 2) as usize) as i32
                    + *d_comp.get_unchecked((i - 3) as usize) as i32))
                as i16;
        }
        i -= 1;
    }
    length_d_comp = 0;
    i = min_lag_8kHz;
    while i < max_lag_8kHz + 4 {
        // Safety: i < max_lag_8kHz+4 < 149, length_d_comp bounded
        if (unsafe { *d_comp.get_unchecked(i as usize) }) as i32 > 0 {
            unsafe {
                *d_comp.get_unchecked_mut(length_d_comp as usize) = (i - 2) as i16;
            }
            length_d_comp += 1;
        }
        i += 1;
    }
    C = [[0.; 149]; 4];
    // For stage 2, use frame_8kHz (or frame directly if 8kHz)
    let frame_8: &[f32] = if Fs_kHz == 8 { frame } else { &frame_8kHz };
    target_off = (PE_LTP_MEM_LENGTH_MS * 8) as usize;
    k = 0;
    while k < nb_subfr {
        energy_tmp =
            silk_energy_FLP(&frame_8[target_off..target_off + sf_length_8kHz as usize]) + 1.0f64;
        j = 0;
        while j < length_d_comp {
            // Safety: j < length_d_comp bounded, k < nb_subfr <= 4, d < 149
            d = (unsafe { *d_comp.get_unchecked(j as usize) }) as i32;
            let basis_off = target_off - d as usize;
            cross_corr = silk_inner_product_FLP(
                &frame_8[basis_off..basis_off + sf_length_8kHz as usize],
                &frame_8[target_off..target_off + sf_length_8kHz as usize],
                arch,
            );
            if cross_corr > 0.0f32 as f64 {
                energy = silk_energy_FLP(&frame_8[basis_off..basis_off + sf_length_8kHz as usize]);
                unsafe {
                    *C.get_unchecked_mut(k as usize)
                        .get_unchecked_mut(d as usize) =
                        (2_f64 * cross_corr / (energy + energy_tmp)) as f32;
                }
            } else {
                unsafe {
                    *C.get_unchecked_mut(k as usize)
                        .get_unchecked_mut(d as usize) = 0.0f32;
                }
            }
            j += 1;
        }
        target_off += sf_length_8kHz as usize;
        k += 1;
    }
    CCmax = 0.0f32;
    CCmax_b = -1000.0f32;
    CBimax = 0;
    lag = -1;
    if prevLag > 0 {
        if Fs_kHz == 12 {
            prevLag = ((prevLag as u32) << 1) as i32 / 3;
        } else if Fs_kHz == 16 {
            prevLag >>= 1;
        }
        prevLag_log2 = silk_log2(prevLag as f32 as f64);
    } else {
        prevLag_log2 = 0 as f32;
    }
    if nb_subfr == PE_MAX_NB_SUBFR as i32 {
        cbk_size = PE_NB_CBKS_STAGE2_EXT as i32;
        Lag_CB = &silk_CB_lags_stage2;
        if Fs_kHz == 8 && complexity > SILK_PE_MIN_COMPLEX {
            nb_cbk_search = PE_NB_CBKS_STAGE2_EXT as i32;
        } else {
            nb_cbk_search = PE_NB_CBKS_STAGE2;
        }
    } else {
        cbk_size = PE_NB_CBKS_STAGE2_10MS as i32;
        Lag_CB = &silk_CB_lags_stage2_10_ms;
        nb_cbk_search = PE_NB_CBKS_STAGE2_10MS as i32;
    }
    k = 0;
    while k < length_d_srch {
        // Safety: k < length_d_srch <= 24
        d = unsafe { *d_srch.get_unchecked(k as usize) };
        j = 0;
        while j < nb_cbk_search {
            // Safety: j < nb_cbk_search <= 11
            unsafe {
                *CC.get_unchecked_mut(j as usize) = 0.0f32;
            }
            i = 0;
            while i < nb_subfr {
                // Safety: i < nb_subfr <= 4, j < cbk_size, d + lag_cb_val < 149
                unsafe {
                    let lag_cb_val = *Lag_CB.get_unchecked((i * cbk_size + j) as usize) as i32;
                    *CC.get_unchecked_mut(j as usize) += *C
                        .get_unchecked(i as usize)
                        .get_unchecked((d + lag_cb_val) as usize);
                }
                i += 1;
            }
            j += 1;
        }
        CCmax_new = -1000.0f32;
        CBimax_new = 0;
        i = 0;
        while i < nb_cbk_search {
            // Safety: i < nb_cbk_search <= 11
            let cc_val = unsafe { *CC.get_unchecked(i as usize) };
            if cc_val > CCmax_new {
                CCmax_new = cc_val;
                CBimax_new = i;
            }
            i += 1;
        }
        lag_log2 = silk_log2(d as f32 as f64);
        CCmax_new_b = CCmax_new - PE_SHORTLAG_BIAS * nb_subfr as f32 * lag_log2;
        if prevLag > 0 {
            delta_lag_log2_sqr = lag_log2 - prevLag_log2;
            delta_lag_log2_sqr *= delta_lag_log2_sqr;
            CCmax_new_b -= PE_PREVLAG_BIAS * nb_subfr as f32 * (*LTPCorr) * delta_lag_log2_sqr
                / (delta_lag_log2_sqr + 0.5f32);
        }
        if CCmax_new_b > CCmax_b && CCmax_new > nb_subfr as f32 * search_thres2 {
            CCmax_b = CCmax_new_b;
            CCmax = CCmax_new;
            lag = d;
            CBimax = CBimax_new;
        }
        k += 1;
    }
    if lag == -1 {
        pitch_out[..nb_subfr as usize].fill(0);
        *LTPCorr = 0.0f32;
        *lagIndex = 0;
        *contourIndex = 0;
        return 1;
    }
    *LTPCorr = CCmax / nb_subfr as f32;
    if Fs_kHz > 8 {
        if Fs_kHz == 12 {
            lag = ((lag as i16 as i32 * 3) >> 1) + ((lag as i16 as i32 * 3) & 1);
        } else {
            lag = ((lag as u32) << 1) as i32;
        }
        lag = if min_lag > max_lag {
            if lag > min_lag {
                min_lag
            } else if lag < max_lag {
                max_lag
            } else {
                lag
            }
        } else if lag > max_lag {
            max_lag
        } else if lag < min_lag {
            min_lag
        } else {
            lag
        };
        start_lag = silk_max_int(lag - 2, min_lag);
        end_lag = silk_min_int(lag + 2, max_lag);
        lag_new = lag;
        CBimax = 0;
        CCmax = -1000.0f32;
        silk_P_Ana_calc_corr_st3(
            &mut cross_corr_st3,
            frame,
            start_lag,
            sf_length,
            nb_subfr,
            complexity,
            arch,
        );
        silk_P_Ana_calc_energy_st3(
            &mut energies_st3,
            frame,
            start_lag,
            sf_length,
            nb_subfr,
            complexity,
        );
        lag_counter = 0;
        contour_bias = PE_FLATCONTOUR_BIAS / lag as f32;
        let Lag_CB: &[i8];
        if nb_subfr == PE_MAX_NB_SUBFR as i32 {
            nb_cbk_search = silk_nb_cbk_searchs_stage3[complexity as usize] as i32;
            cbk_size = PE_NB_CBKS_STAGE3_MAX as i32;
            Lag_CB = &silk_CB_lags_stage3;
        } else {
            nb_cbk_search = PE_NB_CBKS_STAGE3_10MS as i32;
            cbk_size = PE_NB_CBKS_STAGE3_10MS as i32;
            Lag_CB = &silk_CB_lags_stage3_10_ms;
        }
        let target_st3 = (PE_LTP_MEM_LENGTH_MS * Fs_kHz) as usize;
        energy_tmp =
            silk_energy_FLP(&frame[target_st3..target_st3 + (nb_subfr * sf_length) as usize])
                + 1.0f64;
        d = start_lag;
        while d <= end_lag {
            j = 0;
            while j < nb_cbk_search {
                cross_corr = 0.0f64;
                energy = energy_tmp;
                k = 0;
                while k < nb_subfr {
                    // Safety: k < nb_subfr <= 4, j < 34, lag_counter < 5
                    unsafe {
                        cross_corr += *cross_corr_st3
                            .get_unchecked(k as usize)
                            .get_unchecked(j as usize)
                            .get_unchecked(lag_counter as usize)
                            as f64;
                        energy += *energies_st3
                            .get_unchecked(k as usize)
                            .get_unchecked(j as usize)
                            .get_unchecked(lag_counter as usize)
                            as f64;
                    }
                    k += 1;
                }
                if cross_corr > 0.0f64 {
                    CCmax_new = (2_f64 * cross_corr / energy) as f32;
                    CCmax_new *= 1.0f32 - contour_bias * j as f32;
                } else {
                    CCmax_new = 0.0f32;
                }
                if CCmax_new > CCmax
                    && d + (unsafe { *silk_CB_lags_stage3.get_unchecked(j as usize) }) as i32
                        <= max_lag
                {
                    CCmax = CCmax_new;
                    lag_new = d;
                    CBimax = j;
                }
                j += 1;
            }
            lag_counter += 1;
            d += 1;
        }
        k = 0;
        while k < nb_subfr {
            // Safety: k < nb_subfr <= 4, (k * cbk_size + CBimax) bounded by Lag_CB size
            unsafe {
                let lag_cb_val = *Lag_CB.get_unchecked((k * cbk_size + CBimax) as usize) as i32;
                *pitch_out.get_unchecked_mut(k as usize) = lag_new + lag_cb_val;
                let p = *pitch_out.get_unchecked(k as usize);
                *pitch_out.get_unchecked_mut(k as usize) = if min_lag > 18 * Fs_kHz {
                    if p > min_lag {
                        min_lag
                    } else if p < 18 * Fs_kHz {
                        18 * Fs_kHz
                    } else {
                        p
                    }
                } else if p > 18 * Fs_kHz {
                    18 * Fs_kHz
                } else if p < min_lag {
                    min_lag
                } else {
                    p
                };
            }
            k += 1;
        }
        *lagIndex = (lag_new - min_lag) as i16;
        *contourIndex = CBimax as i8;
    } else {
        k = 0;
        while k < nb_subfr {
            // Safety: k < nb_subfr <= 4, (k * cbk_size + CBimax) bounded by Lag_CB size
            unsafe {
                let lag_cb_val = *Lag_CB.get_unchecked((k * cbk_size + CBimax) as usize) as i32;
                *pitch_out.get_unchecked_mut(k as usize) = lag + lag_cb_val;
                let p = *pitch_out.get_unchecked(k as usize);
                *pitch_out.get_unchecked_mut(k as usize) = if min_lag_8kHz > 18 * 8 {
                    if p > min_lag_8kHz {
                        min_lag_8kHz
                    } else if p < 18 * 8 {
                        18 * 8
                    } else {
                        p
                    }
                } else if p > 18 * 8 {
                    18 * 8
                } else if p < min_lag_8kHz {
                    min_lag_8kHz
                } else {
                    p
                };
            }
            k += 1;
        }
        *lagIndex = (lag - min_lag_8kHz) as i16;
        *contourIndex = CBimax as i8;
    }
    debug_assert!(*lagIndex as i32 >= 0);
    0
}
/// Upstream C: silk/float/pitch_analysis_core_FLP.c:silk_P_Ana_calc_corr_st3
fn silk_P_Ana_calc_corr_st3(
    cross_corr_st3: &mut [[[f32; 5]; 34]; 4],
    frame: &[f32],
    start_lag: i32,
    sf_length: i32,
    nb_subfr: i32,
    complexity: i32,
    _arch: Arch,
) {
    let mut i: i32;
    let mut j: i32;
    let mut k: i32;
    let mut lag_counter: i32;
    let mut lag_low: i32;
    let mut lag_high: i32;
    let nb_cbk_search: i32;
    let mut delta: i32;
    let mut idx: i32;
    let cbk_size: i32;
    let mut scratch_mem: [f32; 22] = [0.; 22];
    let mut xcorr: [opus_val32; 22] = [0.; 22];
    let Lag_range: &[[i8; 2]];
    let Lag_CB: &[i8];
    debug_assert!(complexity >= 0);
    debug_assert!(complexity <= 2);
    if nb_subfr == PE_MAX_NB_SUBFR as i32 {
        Lag_range = &silk_Lag_range_stage3[complexity as usize];
        Lag_CB = &silk_CB_lags_stage3;
        nb_cbk_search = silk_nb_cbk_searchs_stage3[complexity as usize] as i32;
        cbk_size = PE_NB_CBKS_STAGE3_MAX as i32;
    } else {
        debug_assert!(nb_subfr == 4 >> 1);
        Lag_range = &silk_Lag_range_stage3_10_ms;
        Lag_CB = &silk_CB_lags_stage3_10_ms;
        nb_cbk_search = PE_NB_CBKS_STAGE3_10MS as i32;
        cbk_size = PE_NB_CBKS_STAGE3_10MS as i32;
    }
    let mut target_off: usize = ((sf_length as u32) << 2) as usize;
    k = 0;
    while k < nb_subfr {
        lag_counter = 0;
        // Safety: k < nb_subfr <= 4, Lag_range has nb_subfr entries
        lag_low = unsafe { *(*Lag_range.get_unchecked(k as usize)).get_unchecked(0) } as i32;
        lag_high = unsafe { *(*Lag_range.get_unchecked(k as usize)).get_unchecked(1) } as i32;
        {
            let xcorr_len = (lag_high - lag_low + 1) as usize;
            let basis_start = target_off - start_lag as usize - lag_high as usize;
            celt_pitch_xcorr(
                &frame[target_off..target_off + sf_length as usize],
                &frame[basis_start..basis_start + sf_length as usize + xcorr_len],
                &mut xcorr[..xcorr_len],
                sf_length as usize,
                _arch,
            );
        }
        j = lag_low;
        while j <= lag_high {
            // Safety: lag_counter < 22, (lag_high - j) < 22
            unsafe {
                *scratch_mem.get_unchecked_mut(lag_counter as usize) =
                    *xcorr.get_unchecked((lag_high - j) as usize);
            }
            lag_counter += 1;
            j += 1;
        }
        delta = unsafe { *(*Lag_range.get_unchecked(k as usize)).get_unchecked(0) } as i32;
        i = 0;
        while i < nb_cbk_search {
            // Safety: (k * cbk_size + i) bounded by Lag_CB size
            idx = (unsafe { *Lag_CB.get_unchecked((k * cbk_size + i) as usize) }) as i32 - delta;
            j = 0;
            while j < PE_NB_STAGE3_LAGS {
                // Safety: k < 4, i < 34, j < 5, (idx + j) < 22
                unsafe {
                    *cross_corr_st3
                        .get_unchecked_mut(k as usize)
                        .get_unchecked_mut(i as usize)
                        .get_unchecked_mut(j as usize) =
                        *scratch_mem.get_unchecked((idx + j) as usize);
                }
                j += 1;
            }
            i += 1;
        }
        target_off += sf_length as usize;
        k += 1;
    }
}
/// Upstream C: silk/float/pitch_analysis_core_FLP.c:silk_P_Ana_calc_energy_st3
fn silk_P_Ana_calc_energy_st3(
    energies_st3: &mut [[[f32; 5]; 34]; 4],
    frame: &[f32],
    start_lag: i32,
    sf_length: i32,
    nb_subfr: i32,
    complexity: i32,
) {
    let mut energy: f64;
    let mut k: i32;
    let mut i: i32;
    let mut j: i32;
    let mut lag_counter: i32;
    let nb_cbk_search: i32;
    let mut delta: i32;
    let mut idx: i32;
    let cbk_size: i32;
    let mut lag_diff: i32;
    let mut scratch_mem: [f32; 22] = [0.; 22];
    let Lag_range: &[[i8; 2]];
    let Lag_CB: &[i8];
    debug_assert!(complexity >= 0);
    debug_assert!(complexity <= 2);
    if nb_subfr == PE_MAX_NB_SUBFR as i32 {
        Lag_range = &silk_Lag_range_stage3[complexity as usize];
        Lag_CB = &silk_CB_lags_stage3;
        nb_cbk_search = silk_nb_cbk_searchs_stage3[complexity as usize] as i32;
        cbk_size = PE_NB_CBKS_STAGE3_MAX as i32;
    } else {
        debug_assert!(nb_subfr == 4 >> 1);
        Lag_range = &silk_Lag_range_stage3_10_ms;
        Lag_CB = &silk_CB_lags_stage3_10_ms;
        nb_cbk_search = PE_NB_CBKS_STAGE3_10MS as i32;
        cbk_size = PE_NB_CBKS_STAGE3_10MS as i32;
    }
    let mut target_off: usize = ((sf_length as u32) << 2) as usize;
    k = 0;
    while k < nb_subfr {
        lag_counter = 0;
        // Safety: k < nb_subfr <= 4, Lag_range has nb_subfr entries
        let lag_range_k = unsafe { Lag_range.get_unchecked(k as usize) };
        let basis_off =
            target_off - (start_lag + unsafe { *lag_range_k.get_unchecked(0) } as i32) as usize;
        energy = silk_energy_FLP(&frame[basis_off..basis_off + sf_length as usize]) + 1e-3f64;
        unsafe {
            *scratch_mem.get_unchecked_mut(lag_counter as usize) = energy as f32;
        }
        lag_counter += 1;
        lag_diff = unsafe { *lag_range_k.get_unchecked(1) } as i32
            - unsafe { *lag_range_k.get_unchecked(0) } as i32
            + 1;
        i = 1;
        while i < lag_diff {
            // Safety: basis_off +/- offsets within frame bounds, lag_counter < 22
            unsafe {
                let hi = *frame.get_unchecked(basis_off + (sf_length - i) as usize) as f64;
                let lo = *frame.get_unchecked(basis_off - i as usize) as f64;
                energy -= hi * hi;
                energy += lo * lo;
                *scratch_mem.get_unchecked_mut(lag_counter as usize) = energy as f32;
            }
            lag_counter += 1;
            i += 1;
        }
        delta = unsafe { *lag_range_k.get_unchecked(0) } as i32;
        i = 0;
        while i < nb_cbk_search {
            // Safety: (k * cbk_size + i) bounded by Lag_CB size
            idx = (unsafe { *Lag_CB.get_unchecked((k * cbk_size + i) as usize) }) as i32 - delta;
            j = 0;
            while j < PE_NB_STAGE3_LAGS {
                // Safety: k < 4, i < 34, j < 5, (idx + j) < 22
                unsafe {
                    *energies_st3
                        .get_unchecked_mut(k as usize)
                        .get_unchecked_mut(i as usize)
                        .get_unchecked_mut(j as usize) =
                        *scratch_mem.get_unchecked((idx + j) as usize);
                }
                j += 1;
            }
            i += 1;
        }
        target_off += sf_length as usize;
        k += 1;
    }
}
