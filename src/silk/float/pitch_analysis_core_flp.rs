//! Floating-point pitch analysis core.
//!
//! Upstream c: `silk/float/pitch_analysis_core_FLP.c`

use crate::arch::Arch;

use crate::silk::typedefs::{SILK_INT16_MAX, SILK_INT16_MIN};

use crate::celt::pitch::celt_pitch_xcorr;
use crate::silk::float::energy_flp::silk_energy_flp;
use crate::silk::float::inner_product_flp::silk_inner_product_flp;
use crate::silk::float::sigproc_flp::{silk_float2short_array, silk_log2, silk_short2float_array};
use crate::silk::float::sort_flp::silk_insertion_sort_decreasing_flp;
use crate::silk::pitch_est_tables::{
    PE_FLATCONTOUR_BIAS, PE_LTP_MEM_LENGTH_MS, PE_MAX_LAG_MS, PE_MAX_NB_SUBFR, PE_MIN_LAG_MS,
    PE_NB_CBKS_STAGE2, PE_NB_CBKS_STAGE2_10MS, PE_NB_CBKS_STAGE2_EXT, PE_NB_CBKS_STAGE3_10MS,
    PE_NB_CBKS_STAGE3_MAX, PE_NB_STAGE3_LAGS, PE_PREVLAG_BIAS, PE_SHORTLAG_BIAS,
    PE_SUBFR_LENGTH_MS, SILK_CB_LAGS_STAGE2, SILK_CB_LAGS_STAGE2_10_MS, SILK_CB_LAGS_STAGE3,
    SILK_CB_LAGS_STAGE3_10_MS, SILK_LAG_RANGE_STAGE3, SILK_LAG_RANGE_STAGE3_10_MS,
    SILK_NB_CBK_SEARCHS_STAGE3, SILK_PE_MIN_COMPLEX,
};
use crate::silk::resampler::{silk_resampler_down2, silk_resampler_down2_3};
use crate::silk::sigproc_fix::{silk_max_int, silk_min_int};
use arrayref::array_mut_ref;

/// Upstream c: silk/float/pitch_analysis_core_FLP.c:silk_pitch_analysis_core_FLP
#[allow(clippy::too_many_arguments)]
pub fn silk_pitch_analysis_core_flp(
    frame: &[f32],
    pitch_out: &mut [i32],
    lag_index: &mut i16,
    contour_index: &mut i8,
    ltpcorr: &mut f32,
    mut prev_lag: i32,
    search_thres1: f32,
    search_thres2: f32,
    fs_k_hz: i32,
    complexity: i32,
    nb_subfr: i32,
    arch: Arch,
) -> i32 {
    let mut _i: i32;
    let mut k: i32;
    let mut d: i32;
    let mut j: i32;
    let mut frame_8k_hz: [f32; 320] = [0.; 320];
    let mut frame_4k_hz: [f32; 160] = [0.; 160];
    let mut frame_8_fix: [i16; 320] = [0; 320];
    let mut frame_4_fix: [i16; 160] = [0; 160];
    let mut filt_state: [i32; 6] = [0; 6];

    let contour_bias: f32;
    let mut c: [[f32; 149]; 4] = [[0.; 149]; 4];
    let mut xcorr: [f32; 65] = [0.; 65];
    let mut cc: [f32; 11] = [0.; 11];
    let mut cross_corr: f64;
    let mut normalizer: f64;
    let mut energy: f64;
    let mut energy_tmp: f64;
    let mut d_srch: [i32; 24] = [0; 24];
    let mut d_comp: [i16; 149] = [0; 149];
    let mut length_d_srch: i32;
    let mut length_d_comp: i32;

    let mut ccmax: f32;
    let mut ccmax_b: f32;
    let mut ccmax_new_b: f32;
    let mut ccmax_new: f32;
    let mut cbimax: i32;
    let mut cbimax_new: i32;
    let mut lag: i32;
    let start_lag: i32;
    let end_lag: i32;
    let mut lag_new: i32;
    let mut cbk_size: i32;
    let mut lag_log2: f32;
    let prev_lag_log2: f32;
    let mut delta_lag_log2_sqr: f32;
    let mut energies_st3: [[[f32; 5]; 34]; 4] = [[[0.; 5]; 34]; 4];
    let mut cross_corr_st3: [[[f32; 5]; 34]; 4] = [[[0.; 5]; 34]; 4];
    let mut lag_counter: i32;

    let mut nb_cbk_search: i32;
    let lag_cb: &[i8];
    debug_assert!(fs_k_hz == 8 || fs_k_hz == 12 || fs_k_hz == 16);
    debug_assert!(complexity >= 0);
    debug_assert!(complexity <= 2);
    let frame_length: i32 = (PE_LTP_MEM_LENGTH_MS + nb_subfr * PE_SUBFR_LENGTH_MS) * fs_k_hz;
    let frame_length_4k_hz: i32 = (PE_LTP_MEM_LENGTH_MS + nb_subfr * PE_SUBFR_LENGTH_MS) * 4;
    let frame_length_8k_hz: i32 = (PE_LTP_MEM_LENGTH_MS + nb_subfr * PE_SUBFR_LENGTH_MS) * 8;
    let sf_length: i32 = PE_SUBFR_LENGTH_MS * fs_k_hz;
    let sf_length_4k_hz: i32 = PE_SUBFR_LENGTH_MS * 4;
    let sf_length_8k_hz: i32 = PE_SUBFR_LENGTH_MS * 8;
    let min_lag: i32 = PE_MIN_LAG_MS * fs_k_hz;
    let min_lag_4k_hz: i32 = PE_MIN_LAG_MS * 4;
    let min_lag_8k_hz: i32 = PE_MIN_LAG_MS * 8;
    let max_lag: i32 = PE_MAX_LAG_MS * fs_k_hz - 1;
    let max_lag_4k_hz: i32 = PE_MAX_LAG_MS * 4;
    let max_lag_8k_hz: i32 = PE_MAX_LAG_MS * 8 - 1;
    if fs_k_hz == 16 {
        let mut frame_16_fix: [i16; 640] = [0; 640];
        silk_float2short_array(
            &mut frame_16_fix[..frame_length as usize],
            &frame[..frame_length as usize],
        );
        let filt_state = array_mut_ref![filt_state, 0, 2];
        filt_state.fill(0);
        silk_resampler_down2(
            filt_state,
            &mut frame_8_fix[..frame_length_8k_hz as usize],
            &frame_16_fix[..frame_length as usize],
        );
        silk_short2float_array(
            &mut frame_8k_hz[..frame_length_8k_hz as usize],
            &frame_8_fix[..frame_length_8k_hz as usize],
        );
    } else if fs_k_hz == 12 {
        let mut frame_12_fix: [i16; 480] = [0; 480];
        silk_float2short_array(
            &mut frame_12_fix[..frame_length as usize],
            &frame[..frame_length as usize],
        );
        filt_state.fill(0);
        silk_resampler_down2_3(
            &mut filt_state,
            &mut frame_8_fix[..frame_length_8k_hz as usize],
            &frame_12_fix[..frame_length as usize],
        );
        silk_short2float_array(
            &mut frame_8k_hz[..frame_length_8k_hz as usize],
            &frame_8_fix[..frame_length_8k_hz as usize],
        );
    } else {
        debug_assert!(fs_k_hz == 8);
        silk_float2short_array(
            &mut frame_8_fix[..frame_length_8k_hz as usize],
            &frame[..frame_length_8k_hz as usize],
        );
    }
    {
        let filt_state = array_mut_ref![filt_state, 0, 2];
        filt_state.fill(0);
        silk_resampler_down2(
            filt_state,
            &mut frame_4_fix[..frame_length_4k_hz as usize],
            &frame_8_fix[..frame_length_8k_hz as usize],
        );
    }
    silk_short2float_array(
        &mut frame_4k_hz[..frame_length_4k_hz as usize],
        &frame_4_fix[..frame_length_4k_hz as usize],
    );
    _i = frame_length_4k_hz - 1;
    while _i > 0 {
        frame_4k_hz[_i as usize] = (if frame_4k_hz[_i as usize] as i32 as f32
            + frame_4k_hz[(_i - 1) as usize]
            > SILK_INT16_MAX as f32
        {
            SILK_INT16_MAX as f32
        } else if frame_4k_hz[_i as usize] as i32 as f32 + frame_4k_hz[(_i - 1) as usize]
            < SILK_INT16_MIN as f32
        {
            SILK_INT16_MIN as f32
        } else {
            frame_4k_hz[_i as usize] as i32 as f32 + frame_4k_hz[(_i - 1) as usize]
        }) as i16 as f32;
        _i -= 1;
    }
    // c is already zero-initialized above
    // target_off tracks position in frame_4k_hz
    let mut target_off: usize = ((sf_length_4k_hz as u32) << 2) as usize;
    k = 0;
    while k < nb_subfr >> 1 {
        debug_assert!(target_off + sf_length_8k_hz as usize <= frame_length_4k_hz as usize);
        let basis_off = target_off - min_lag_4k_hz as usize;
        debug_assert!(basis_off + sf_length_8k_hz as usize <= frame_length_4k_hz as usize);
        {
            let xcorr_len = (max_lag_4k_hz - min_lag_4k_hz + 1) as usize;
            celt_pitch_xcorr(
                &frame_4k_hz[target_off..target_off + sf_length_8k_hz as usize],
                &frame_4k_hz[target_off - max_lag_4k_hz as usize
                    ..target_off - max_lag_4k_hz as usize + sf_length_8k_hz as usize + xcorr_len],
                &mut xcorr[..xcorr_len],
                sf_length_8k_hz as usize,
                arch,
            );
        }
        cross_corr = xcorr[(max_lag_4k_hz - min_lag_4k_hz) as usize] as f64;
        normalizer =
            silk_energy_flp(&frame_4k_hz[target_off..target_off + sf_length_8k_hz as usize])
                + silk_energy_flp(&frame_4k_hz[basis_off..basis_off + sf_length_8k_hz as usize])
                + (sf_length_8k_hz as f32 * 4000.0f32) as f64;
        c[0][min_lag_4k_hz as usize] += (2_f64 * cross_corr / normalizer) as f32;
        // basis_off_d starts at basis_off and decrements
        let mut basis_off_d = basis_off;
        d = min_lag_4k_hz + 1;
        while d <= max_lag_4k_hz {
            basis_off_d -= 1;
            cross_corr = xcorr[(max_lag_4k_hz - d) as usize] as f64;
            normalizer += frame_4k_hz[basis_off_d] as f64 * frame_4k_hz[basis_off_d] as f64
                - frame_4k_hz[basis_off_d + sf_length_8k_hz as usize] as f64
                    * frame_4k_hz[basis_off_d + sf_length_8k_hz as usize] as f64;
            c[0][d as usize] += (2_f64 * cross_corr / normalizer) as f32;
            d += 1;
        }
        target_off += sf_length_8k_hz as usize;
        k += 1;
    }
    _i = max_lag_4k_hz;
    while _i >= min_lag_4k_hz {
        c[0_usize][_i as usize] -= c[0_usize][_i as usize] * _i as f32 / 4096.0f32;
        _i -= 1;
    }
    length_d_srch = 4 + 2 * complexity;
    debug_assert!(3 * length_d_srch <= 24);
    silk_insertion_sort_decreasing_flp(
        &mut c[0][min_lag_4k_hz as usize..],
        &mut d_srch,
        max_lag_4k_hz - min_lag_4k_hz + 1,
        length_d_srch,
    );
    let cmax: f32 = c[0][min_lag_4k_hz as usize];
    if cmax < 0.2f32 {
        pitch_out[..nb_subfr as usize].fill(0);
        *ltpcorr = 0.0f32;
        *lag_index = 0;
        *contour_index = 0;
        return 1;
    }
    let threshold: f32 = search_thres1 * cmax;
    _i = 0;
    while _i < length_d_srch {
        if c[0_usize][(min_lag_4k_hz + _i) as usize] > threshold {
            d_srch[_i as usize] = (((d_srch[_i as usize] + min_lag_4k_hz) as u32) << 1) as i32;
            _i += 1;
        } else {
            length_d_srch = _i;
            break;
        }
    }
    debug_assert!(length_d_srch > 0);
    _i = min_lag_8k_hz - 5;
    while _i < max_lag_8k_hz + 5 {
        d_comp[_i as usize] = 0;
        _i += 1;
    }
    _i = 0;
    while _i < length_d_srch {
        d_comp[d_srch[_i as usize] as usize] = 1;
        _i += 1;
    }
    _i = max_lag_8k_hz + 3;
    while _i >= min_lag_8k_hz {
        d_comp[_i as usize] = (d_comp[_i as usize] as i32
            + (d_comp[(_i - 1) as usize] as i32 + d_comp[(_i - 2) as usize] as i32))
            as i16;
        _i -= 1;
    }
    length_d_srch = 0;
    _i = min_lag_8k_hz;
    while _i < max_lag_8k_hz + 1 {
        if d_comp[(_i + 1) as usize] as i32 > 0 {
            d_srch[length_d_srch as usize] = _i;
            length_d_srch += 1;
        }
        _i += 1;
    }
    _i = max_lag_8k_hz + 3;
    while _i >= min_lag_8k_hz {
        d_comp[_i as usize] = (d_comp[_i as usize] as i32
            + (d_comp[(_i - 1) as usize] as i32
                + d_comp[(_i - 2) as usize] as i32
                + d_comp[(_i - 3) as usize] as i32)) as i16;
        _i -= 1;
    }
    length_d_comp = 0;
    _i = min_lag_8k_hz;
    while _i < max_lag_8k_hz + 4 {
        if d_comp[_i as usize] as i32 > 0 {
            d_comp[length_d_comp as usize] = (_i - 2) as i16;
            length_d_comp += 1;
        }
        _i += 1;
    }
    c = [[0.; 149]; 4];
    // For stage 2, use frame_8k_hz (or frame directly if 8kHz)
    let frame_8: &[f32] = if fs_k_hz == 8 { frame } else { &frame_8k_hz };
    target_off = (PE_LTP_MEM_LENGTH_MS * 8) as usize;
    k = 0;
    while k < nb_subfr {
        energy_tmp =
            silk_energy_flp(&frame_8[target_off..target_off + sf_length_8k_hz as usize]) + 1.0f64;
        j = 0;
        while j < length_d_comp {
            d = d_comp[j as usize] as i32;
            let basis_off = target_off - d as usize;
            cross_corr = silk_inner_product_flp(
                &frame_8[basis_off..basis_off + sf_length_8k_hz as usize],
                &frame_8[target_off..target_off + sf_length_8k_hz as usize],
                arch,
            );
            if cross_corr > 0.0f32 as f64 {
                energy = silk_energy_flp(&frame_8[basis_off..basis_off + sf_length_8k_hz as usize]);
                c[k as usize][d as usize] = (2_f64 * cross_corr / (energy + energy_tmp)) as f32;
            } else {
                c[k as usize][d as usize] = 0.0f32;
            }
            j += 1;
        }
        target_off += sf_length_8k_hz as usize;
        k += 1;
    }
    ccmax = 0.0f32;
    ccmax_b = -1000.0f32;
    cbimax = 0;
    lag = -1;
    if prev_lag > 0 {
        if fs_k_hz == 12 {
            prev_lag = ((prev_lag as u32) << 1) as i32 / 3;
        } else if fs_k_hz == 16 {
            prev_lag >>= 1;
        }
        prev_lag_log2 = silk_log2(prev_lag as f32 as f64);
    } else {
        prev_lag_log2 = 0 as f32;
    }
    if nb_subfr == PE_MAX_NB_SUBFR as i32 {
        cbk_size = PE_NB_CBKS_STAGE2_EXT as i32;
        lag_cb = &SILK_CB_LAGS_STAGE2;
        if fs_k_hz == 8 && complexity > SILK_PE_MIN_COMPLEX {
            nb_cbk_search = PE_NB_CBKS_STAGE2_EXT as i32;
        } else {
            nb_cbk_search = PE_NB_CBKS_STAGE2;
        }
    } else {
        cbk_size = PE_NB_CBKS_STAGE2_10MS as i32;
        lag_cb = &SILK_CB_LAGS_STAGE2_10_MS;
        nb_cbk_search = PE_NB_CBKS_STAGE2_10MS as i32;
    }
    k = 0;
    while k < length_d_srch {
        d = d_srch[k as usize];
        j = 0;
        while j < nb_cbk_search {
            cc[j as usize] = 0.0f32;
            _i = 0;
            while _i < nb_subfr {
                cc[j as usize] +=
                    c[_i as usize][(d + lag_cb[(_i * cbk_size + j) as usize] as i32) as usize];
                _i += 1;
            }
            j += 1;
        }
        ccmax_new = -1000.0f32;
        cbimax_new = 0;
        _i = 0;
        while _i < nb_cbk_search {
            if cc[_i as usize] > ccmax_new {
                ccmax_new = cc[_i as usize];
                cbimax_new = _i;
            }
            _i += 1;
        }
        lag_log2 = silk_log2(d as f32 as f64);
        ccmax_new_b = ccmax_new - PE_SHORTLAG_BIAS * nb_subfr as f32 * lag_log2;
        if prev_lag > 0 {
            delta_lag_log2_sqr = lag_log2 - prev_lag_log2;
            delta_lag_log2_sqr *= delta_lag_log2_sqr;
            ccmax_new_b -= PE_PREVLAG_BIAS * nb_subfr as f32 * (*ltpcorr) * delta_lag_log2_sqr
                / (delta_lag_log2_sqr + 0.5f32);
        }
        if ccmax_new_b > ccmax_b && ccmax_new > nb_subfr as f32 * search_thres2 {
            ccmax_b = ccmax_new_b;
            ccmax = ccmax_new;
            lag = d;
            cbimax = cbimax_new;
        }
        k += 1;
    }
    if lag == -1 {
        pitch_out[..nb_subfr as usize].fill(0);
        *ltpcorr = 0.0f32;
        *lag_index = 0;
        *contour_index = 0;
        return 1;
    }
    *ltpcorr = ccmax / nb_subfr as f32;
    if fs_k_hz > 8 {
        if fs_k_hz == 12 {
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
        cbimax = 0;
        ccmax = -1000.0f32;
        silk_p_ana_calc_corr_st3(
            &mut cross_corr_st3,
            frame,
            start_lag,
            sf_length,
            nb_subfr,
            complexity,
            arch,
        );
        silk_p_ana_calc_energy_st3(
            &mut energies_st3,
            frame,
            start_lag,
            sf_length,
            nb_subfr,
            complexity,
        );
        lag_counter = 0;
        contour_bias = PE_FLATCONTOUR_BIAS / lag as f32;
        let lag_cb: &[i8];
        if nb_subfr == PE_MAX_NB_SUBFR as i32 {
            nb_cbk_search = SILK_NB_CBK_SEARCHS_STAGE3[complexity as usize] as i32;
            cbk_size = PE_NB_CBKS_STAGE3_MAX as i32;
            lag_cb = &SILK_CB_LAGS_STAGE3;
        } else {
            nb_cbk_search = PE_NB_CBKS_STAGE3_10MS as i32;
            cbk_size = PE_NB_CBKS_STAGE3_10MS as i32;
            lag_cb = &SILK_CB_LAGS_STAGE3_10_MS;
        }
        let target_st3 = (PE_LTP_MEM_LENGTH_MS * fs_k_hz) as usize;
        energy_tmp =
            silk_energy_flp(&frame[target_st3..target_st3 + (nb_subfr * sf_length) as usize])
                + 1.0f64;
        d = start_lag;
        while d <= end_lag {
            j = 0;
            while j < nb_cbk_search {
                cross_corr = 0.0f64;
                energy = energy_tmp;
                k = 0;
                while k < nb_subfr {
                    cross_corr +=
                        cross_corr_st3[k as usize][j as usize][lag_counter as usize] as f64;
                    energy += energies_st3[k as usize][j as usize][lag_counter as usize] as f64;
                    k += 1;
                }
                if cross_corr > 0.0f64 {
                    ccmax_new = (2_f64 * cross_corr / energy) as f32;
                    ccmax_new *= 1.0f32 - contour_bias * j as f32;
                } else {
                    ccmax_new = 0.0f32;
                }
                if ccmax_new > ccmax && d + SILK_CB_LAGS_STAGE3[j as usize] as i32 <= max_lag {
                    ccmax = ccmax_new;
                    lag_new = d;
                    cbimax = j;
                }
                j += 1;
            }
            lag_counter += 1;
            d += 1;
        }
        k = 0;
        while k < nb_subfr {
            pitch_out[k as usize] = lag_new + lag_cb[(k * cbk_size + cbimax) as usize] as i32;
            pitch_out[k as usize] = if min_lag > 18 * fs_k_hz {
                if pitch_out[k as usize] > min_lag {
                    min_lag
                } else if pitch_out[k as usize] < 18 * fs_k_hz {
                    18 * fs_k_hz
                } else {
                    pitch_out[k as usize]
                }
            } else if pitch_out[k as usize] > 18 * fs_k_hz {
                18 * fs_k_hz
            } else if pitch_out[k as usize] < min_lag {
                min_lag
            } else {
                pitch_out[k as usize]
            };
            k += 1;
        }
        *lag_index = (lag_new - min_lag) as i16;
        *contour_index = cbimax as i8;
    } else {
        k = 0;
        while k < nb_subfr {
            pitch_out[k as usize] = lag + lag_cb[(k * cbk_size + cbimax) as usize] as i32;
            pitch_out[k as usize] = if min_lag_8k_hz > 18 * 8 {
                if pitch_out[k as usize] > min_lag_8k_hz {
                    min_lag_8k_hz
                } else if pitch_out[k as usize] < 18 * 8 {
                    18 * 8
                } else {
                    pitch_out[k as usize]
                }
            } else if pitch_out[k as usize] > 18 * 8 {
                18 * 8
            } else if pitch_out[k as usize] < min_lag_8k_hz {
                min_lag_8k_hz
            } else {
                pitch_out[k as usize]
            };
            k += 1;
        }
        *lag_index = (lag - min_lag_8k_hz) as i16;
        *contour_index = cbimax as i8;
    }
    debug_assert!(*lag_index as i32 >= 0);
    0
}
/// Upstream c: silk/float/pitch_analysis_core_FLP.c:silk_P_Ana_calc_corr_st3
fn silk_p_ana_calc_corr_st3(
    cross_corr_st3: &mut [[[f32; 5]; 34]; 4],
    frame: &[f32],
    start_lag: i32,
    sf_length: i32,
    nb_subfr: i32,
    complexity: i32,
    _arch: Arch,
) {
    let mut _i: i32;
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
    let mut xcorr: [f32; 22] = [0.; 22];
    let lag_range: &[[i8; 2]];
    let lag_cb: &[i8];
    debug_assert!(complexity >= 0);
    debug_assert!(complexity <= 2);
    if nb_subfr == PE_MAX_NB_SUBFR as i32 {
        lag_range = &SILK_LAG_RANGE_STAGE3[complexity as usize];
        lag_cb = &SILK_CB_LAGS_STAGE3;
        nb_cbk_search = SILK_NB_CBK_SEARCHS_STAGE3[complexity as usize] as i32;
        cbk_size = PE_NB_CBKS_STAGE3_MAX as i32;
    } else {
        debug_assert!(nb_subfr == 4 >> 1);
        lag_range = &SILK_LAG_RANGE_STAGE3_10_MS;
        lag_cb = &SILK_CB_LAGS_STAGE3_10_MS;
        nb_cbk_search = PE_NB_CBKS_STAGE3_10MS as i32;
        cbk_size = PE_NB_CBKS_STAGE3_10MS as i32;
    }
    let mut target_off: usize = ((sf_length as u32) << 2) as usize;
    k = 0;
    while k < nb_subfr {
        lag_counter = 0;
        lag_low = lag_range[k as usize][0] as i32;
        lag_high = lag_range[k as usize][1] as i32;
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
            scratch_mem[lag_counter as usize] = xcorr[(lag_high - j) as usize];
            lag_counter += 1;
            j += 1;
        }
        delta = lag_range[k as usize][0] as i32;
        _i = 0;
        while _i < nb_cbk_search {
            idx = lag_cb[(k * cbk_size + _i) as usize] as i32 - delta;
            j = 0;
            while j < PE_NB_STAGE3_LAGS {
                cross_corr_st3[k as usize][_i as usize][j as usize] =
                    scratch_mem[(idx + j) as usize];
                j += 1;
            }
            _i += 1;
        }
        target_off += sf_length as usize;
        k += 1;
    }
}
/// Upstream c: silk/float/pitch_analysis_core_FLP.c:silk_P_Ana_calc_energy_st3
fn silk_p_ana_calc_energy_st3(
    energies_st3: &mut [[[f32; 5]; 34]; 4],
    frame: &[f32],
    start_lag: i32,
    sf_length: i32,
    nb_subfr: i32,
    complexity: i32,
) {
    let mut energy: f64;
    let mut k: i32;
    let mut _i: i32;
    let mut j: i32;
    let mut lag_counter: i32;
    let nb_cbk_search: i32;
    let mut delta: i32;
    let mut idx: i32;
    let cbk_size: i32;
    let mut lag_diff: i32;
    let mut scratch_mem: [f32; 22] = [0.; 22];
    let lag_range: &[[i8; 2]];
    let lag_cb: &[i8];
    debug_assert!(complexity >= 0);
    debug_assert!(complexity <= 2);
    if nb_subfr == PE_MAX_NB_SUBFR as i32 {
        lag_range = &SILK_LAG_RANGE_STAGE3[complexity as usize];
        lag_cb = &SILK_CB_LAGS_STAGE3;
        nb_cbk_search = SILK_NB_CBK_SEARCHS_STAGE3[complexity as usize] as i32;
        cbk_size = PE_NB_CBKS_STAGE3_MAX as i32;
    } else {
        debug_assert!(nb_subfr == 4 >> 1);
        lag_range = &SILK_LAG_RANGE_STAGE3_10_MS;
        lag_cb = &SILK_CB_LAGS_STAGE3_10_MS;
        nb_cbk_search = PE_NB_CBKS_STAGE3_10MS as i32;
        cbk_size = PE_NB_CBKS_STAGE3_10MS as i32;
    }
    let mut target_off: usize = ((sf_length as u32) << 2) as usize;
    k = 0;
    while k < nb_subfr {
        lag_counter = 0;
        let basis_off = target_off - (start_lag + lag_range[k as usize][0] as i32) as usize;
        energy = silk_energy_flp(&frame[basis_off..basis_off + sf_length as usize]) + 1e-3f64;
        scratch_mem[lag_counter as usize] = energy as f32;
        lag_counter += 1;
        lag_diff = lag_range[k as usize][1] as i32 - lag_range[k as usize][0] as i32 + 1;
        _i = 1;
        while _i < lag_diff {
            // basis_ptr.offset(sf_length - _i) => frame[basis_off + sf_length - _i]
            // basis_ptr.offset(-_i) => frame[basis_off - _i]
            energy -= frame[basis_off + (sf_length - _i) as usize] as f64
                * frame[basis_off + (sf_length - _i) as usize] as f64;
            energy += frame[basis_off - _i as usize] as f64 * frame[basis_off - _i as usize] as f64;
            scratch_mem[lag_counter as usize] = energy as f32;
            lag_counter += 1;
            _i += 1;
        }
        delta = lag_range[k as usize][0] as i32;
        _i = 0;
        while _i < nb_cbk_search {
            idx = lag_cb[(k * cbk_size + _i) as usize] as i32 - delta;
            j = 0;
            while j < PE_NB_STAGE3_LAGS {
                energies_st3[k as usize][_i as usize][j as usize] = scratch_mem[(idx + j) as usize];
                j += 1;
            }
            _i += 1;
        }
        target_off += sf_length as usize;
        k += 1;
    }
}
