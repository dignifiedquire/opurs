//! Floating-point noise shaping analysis.
//!
//! Upstream c: `silk/float/noise_shape_analysis_FLP.c`

use crate::celt::mathops::celt_sqrt;
use crate::silk::define::{MAX_SHAPE_LPC_ORDER, MIN_QGAIN_DB, TYPE_VOICED, USE_HARM_SHAPING};
use crate::silk::float::structs_flp::{
    silk_encoder_control_FLP, silk_encoder_state_FLP, silk_shape_state_FLP,
};
use crate::silk::tuning_parameters::{
    BANDWIDTH_EXPANSION, BG_SNR_DECR_DB, ENERGY_VARIATION_THRESHOLD_QNT_OFFSET,
    FIND_PITCH_WHITE_NOISE_FRACTION, HARMONIC_SHAPING, HARM_HP_NOISE_COEF, HARM_SNR_INCR_DB,
    HIGH_RATE_OR_LOW_QUALITY_HARMONIC_SHAPING, HP_NOISE_COEF, LOW_FREQ_SHAPING,
    LOW_QUALITY_LOW_FREQ_SHAPING_DECR, SHAPE_WHITE_NOISE_FRACTION, SUBFR_SMTH_COEF,
};

use crate::silk::float::apply_sine_window_flp::silk_apply_sine_window_flp;
use crate::silk::float::autocorrelation_flp::silk_autocorrelation_flp;
use crate::silk::float::bwexpander_flp::silk_bwexpander_flp;
use crate::silk::float::energy_flp::silk_energy_flp;
use crate::silk::float::k2a_flp::silk_k2a_flp;
use crate::silk::float::schur_flp::silk_schur_flp;
use crate::silk::float::sigproc_flp::{silk_log2, silk_sigmoid};
use crate::silk::float::warped_autocorrelation_flp::silk_warped_autocorrelation_flp;
use crate::silk::mathops::silk_exp2;

/// Upstream c: silk/float/noise_shape_analysis_FLP.c:warped_gain
#[inline]
pub fn warped_gain(coefs: &[f32], mut lambda: f32, order: i32) -> f32 {
    let mut _i: i32;
    let mut gain: f32;
    lambda = -lambda;
    gain = coefs[(order - 1) as usize];
    _i = order - 2;
    while _i >= 0 {
        gain = lambda * gain + coefs[_i as usize];
        _i -= 1;
    }
    1.0f32 / (1.0f32 - lambda * gain)
}
/// Upstream c: silk/float/noise_shape_analysis_FLP.c:warped_true2monic_coefs
#[inline]
pub fn warped_true2monic_coefs(coefs: &mut [f32], lambda: f32, limit: f32, order: i32) {
    let mut _i: i32;
    let mut iter: i32;
    let mut ind: i32 = 0;
    let mut tmp: f32;
    let mut maxabs: f32;
    let mut chirp: f32;
    let mut gain: f32;
    _i = order - 1;
    while _i > 0 {
        coefs[(_i - 1) as usize] -= lambda * coefs[_i as usize];
        _i -= 1;
    }
    gain = (1.0f32 - lambda * lambda) / (1.0f32 + lambda * coefs[0]);
    _i = 0;
    while _i < order {
        coefs[_i as usize] *= gain;
        _i += 1;
    }
    iter = 0;
    while iter < 10 {
        maxabs = -1.0f32;
        _i = 0;
        while _i < order {
            tmp = coefs[_i as usize].abs();
            if tmp > maxabs {
                maxabs = tmp;
                ind = _i;
            }
            _i += 1;
        }
        if maxabs <= limit {
            return;
        }
        _i = 1;
        while _i < order {
            coefs[(_i - 1) as usize] += lambda * coefs[_i as usize];
            _i += 1;
        }
        gain = 1.0f32 / gain;
        _i = 0;
        while _i < order {
            coefs[_i as usize] *= gain;
            _i += 1;
        }
        chirp = 0.99f32
            - (0.8f32 + 0.1f32 * iter as f32) * (maxabs - limit) / (maxabs * (ind + 1) as f32);
        silk_bwexpander_flp(coefs, order, chirp);
        _i = order - 1;
        while _i > 0 {
            coefs[(_i - 1) as usize] -= lambda * coefs[_i as usize];
            _i -= 1;
        }
        gain = (1.0f32 - lambda * lambda) / (1.0f32 + lambda * coefs[0]);
        _i = 0;
        while _i < order {
            coefs[_i as usize] *= gain;
            _i += 1;
        }
        iter += 1;
    }
}
/// Upstream c: silk/float/noise_shape_analysis_FLP.c:limit_coefs
#[inline]
pub fn limit_coefs(coefs: &mut [f32], limit: f32, order: i32) {
    let mut _i: i32;
    let mut iter: i32;
    let mut ind: i32 = 0;
    let mut tmp: f32;
    let mut maxabs: f32;
    let mut chirp: f32;
    iter = 0;
    while iter < 10 {
        maxabs = -1.0f32;
        _i = 0;
        while _i < order {
            tmp = coefs[_i as usize].abs();
            if tmp > maxabs {
                maxabs = tmp;
                ind = _i;
            }
            _i += 1;
        }
        if maxabs <= limit {
            return;
        }
        chirp = 0.99f32
            - (0.8f32 + 0.1f32 * iter as f32) * (maxabs - limit) / (maxabs * (ind + 1) as f32);
        silk_bwexpander_flp(coefs, order, chirp);
        iter += 1;
    }
}

/// Upstream c: silk/float/noise_shape_analysis_FLP.c:silk_noise_shape_analysis_FLP
pub fn silk_noise_shape_analysis_flp(
    ps_enc: &mut silk_encoder_state_FLP,
    ps_enc_ctrl: &mut silk_encoder_control_FLP,
    pitch_res: &[f32],
    x: &[f32],
) {
    let ps_shape_st: &mut silk_shape_state_FLP = &mut ps_enc.s_shape;
    let mut k: i32;
    let n_samples: i32;
    let n_segs: i32;
    let mut snr_adj_d_b: f32;
    let mut harm_shape_gain: f32;
    let tilt: f32;
    let mut nrg: f32;
    let mut log_energy: f32;
    let mut log_energy_prev: f32;
    let mut energy_variation: f32;

    let mut strength: f32;
    let mut b: f32;

    let mut x_windowed: [f32; 240] = [0.; 240];
    let mut auto_corr: [f32; 25] = [0.; 25];
    let mut rc: [f32; 25] = [0.; 25];
    let mut x_off: usize = 0;
    let mut pitch_res_off: usize = 0;

    // x starts at -la_shape offset relative to frame data
    snr_adj_d_b = ps_enc.s_cmn.snr_d_b_q7 as f32 * (1_f32 / 128.0f32);
    ps_enc_ctrl.input_quality = 0.5f32
        * (ps_enc.s_cmn.input_quality_bands_q15[0_usize]
            + ps_enc.s_cmn.input_quality_bands_q15[1_usize]) as f32
        * (1.0f32 / 32768.0f32);
    ps_enc_ctrl.coding_quality = silk_sigmoid(0.25f32 * (snr_adj_d_b - 20.0f32));
    if ps_enc.s_cmn.use_cbr == 0 {
        b = 1.0f32 - ps_enc.s_cmn.speech_activity_q8 as f32 * (1.0f32 / 256.0f32);
        snr_adj_d_b -= BG_SNR_DECR_DB
            * ps_enc_ctrl.coding_quality
            * (0.5f32 + 0.5f32 * ps_enc_ctrl.input_quality)
            * b
            * b;
    }
    if ps_enc.s_cmn.indices.signal_type as i32 == TYPE_VOICED {
        snr_adj_d_b += HARM_SNR_INCR_DB * ps_enc.ltpcorr;
    } else {
        snr_adj_d_b += (-0.4f32 * ps_enc.s_cmn.snr_d_b_q7 as f32 * (1_f32 / 128.0f32) + 6.0f32)
            * (1.0f32 - ps_enc_ctrl.input_quality);
    }
    if ps_enc.s_cmn.indices.signal_type as i32 == TYPE_VOICED {
        ps_enc.s_cmn.indices.quant_offset_type = 0;
    } else {
        n_samples = 2 * ps_enc.s_cmn.fs_k_hz;
        energy_variation = 0.0f32;
        log_energy_prev = 0.0f32;
        n_segs = 5 * ps_enc.s_cmn.nb_subfr as i16 as i32 / 2;
        k = 0;
        while k < n_segs {
            nrg = n_samples as f32
                + silk_energy_flp(&pitch_res[pitch_res_off..pitch_res_off + n_samples as usize])
                    as f32;
            log_energy = silk_log2(nrg as f64);
            if k > 0 {
                energy_variation += (log_energy - log_energy_prev).abs();
            }
            log_energy_prev = log_energy;
            pitch_res_off += n_samples as usize;
            k += 1;
        }
        if energy_variation > ENERGY_VARIATION_THRESHOLD_QNT_OFFSET * (n_segs - 1) as f32 {
            ps_enc.s_cmn.indices.quant_offset_type = 0;
        } else {
            ps_enc.s_cmn.indices.quant_offset_type = 1;
        }
    }
    strength = FIND_PITCH_WHITE_NOISE_FRACTION * ps_enc_ctrl.pred_gain;
    let bwexp: f32 = BANDWIDTH_EXPANSION / (1.0f32 + strength * strength);
    let warping: f32 =
        ps_enc.s_cmn.warping_q16 as f32 / 65536.0f32 + 0.01f32 * ps_enc_ctrl.coding_quality;
    k = 0;
    while k < ps_enc.s_cmn.nb_subfr as i32 {
        let mut shift: i32;

        let flat_part: i32 = ps_enc.s_cmn.fs_k_hz * 3;
        let slope_part: i32 = (ps_enc.s_cmn.shape_win_length - flat_part) / 2;
        silk_apply_sine_window_flp(
            &mut x_windowed[..slope_part as usize],
            &x[x_off..x_off + slope_part as usize],
            1,
            slope_part,
        );
        shift = slope_part;
        x_windowed[shift as usize..shift as usize + flat_part as usize].copy_from_slice(
            &x[x_off + shift as usize..x_off + shift as usize + flat_part as usize],
        );
        shift += flat_part;
        silk_apply_sine_window_flp(
            &mut x_windowed[shift as usize..shift as usize + slope_part as usize],
            &x[x_off + shift as usize..x_off + shift as usize + slope_part as usize],
            2,
            slope_part,
        );
        x_off += ps_enc.s_cmn.subfr_length;
        if ps_enc.s_cmn.warping_q16 > 0 {
            silk_warped_autocorrelation_flp(
                &mut auto_corr,
                &x_windowed,
                warping,
                ps_enc.s_cmn.shape_win_length,
                ps_enc.s_cmn.shaping_lpcorder,
            );
        } else {
            silk_autocorrelation_flp(
                &mut auto_corr[..(ps_enc.s_cmn.shaping_lpcorder + 1) as usize],
                &x_windowed[..ps_enc.s_cmn.shape_win_length as usize],
                ps_enc.s_cmn.arch,
            );
        }
        auto_corr[0_usize] += auto_corr[0_usize] * SHAPE_WHITE_NOISE_FRACTION + 1.0f32;
        nrg = silk_schur_flp(&mut rc, &auto_corr, ps_enc.s_cmn.shaping_lpcorder);
        silk_k2a_flp(
            &mut (&mut ps_enc_ctrl.ar)[(k * MAX_SHAPE_LPC_ORDER) as usize..],
            &rc,
            ps_enc.s_cmn.shaping_lpcorder,
        );
        ps_enc_ctrl.gains[k as usize] = celt_sqrt(nrg);
        if ps_enc.s_cmn.warping_q16 > 0 {
            ps_enc_ctrl.gains[k as usize] *= warped_gain(
                &(&ps_enc_ctrl.ar)[(k * MAX_SHAPE_LPC_ORDER) as usize..],
                warping,
                ps_enc.s_cmn.shaping_lpcorder,
            );
        }
        silk_bwexpander_flp(
            &mut (&mut ps_enc_ctrl.ar)[(k * MAX_SHAPE_LPC_ORDER) as usize..],
            ps_enc.s_cmn.shaping_lpcorder,
            bwexp,
        );
        if ps_enc.s_cmn.warping_q16 > 0 {
            warped_true2monic_coefs(
                &mut (&mut ps_enc_ctrl.ar)[(k * MAX_SHAPE_LPC_ORDER) as usize..],
                warping,
                3.999f32,
                ps_enc.s_cmn.shaping_lpcorder,
            );
        } else {
            limit_coefs(
                &mut (&mut ps_enc_ctrl.ar)[(k * MAX_SHAPE_LPC_ORDER) as usize..],
                3.999f32,
                ps_enc.s_cmn.shaping_lpcorder,
            );
        }
        k += 1;
    }
    let gain_mult: f32 = silk_exp2(-0.16f32 * snr_adj_d_b);
    let gain_add: f32 = silk_exp2(0.16f32 * MIN_QGAIN_DB as f32);
    k = 0;
    while k < ps_enc.s_cmn.nb_subfr as i32 {
        ps_enc_ctrl.gains[k as usize] *= gain_mult;
        ps_enc_ctrl.gains[k as usize] += gain_add;
        k += 1;
    }
    strength = LOW_FREQ_SHAPING
        * (1.0f32
            + LOW_QUALITY_LOW_FREQ_SHAPING_DECR
                * (ps_enc.s_cmn.input_quality_bands_q15[0_usize] as f32 * (1.0f32 / 32768.0f32)
                    - 1.0f32));
    strength *= ps_enc.s_cmn.speech_activity_q8 as f32 * (1.0f32 / 256.0f32);
    if ps_enc.s_cmn.indices.signal_type as i32 == TYPE_VOICED {
        k = 0;
        while k < ps_enc.s_cmn.nb_subfr as i32 {
            b = 0.2f32 / ps_enc.s_cmn.fs_k_hz as f32
                + 3.0f32 / ps_enc_ctrl.pitch_l[k as usize] as f32;
            ps_enc_ctrl.lf_ma_shp[k as usize] = -1.0f32 + b;
            ps_enc_ctrl.lf_ar_shp[k as usize] = 1.0f32 - b - b * strength;
            k += 1;
        }
        tilt = -HP_NOISE_COEF
            - (1_f32 - HP_NOISE_COEF)
                * HARM_HP_NOISE_COEF
                * ps_enc.s_cmn.speech_activity_q8 as f32
                * (1.0f32 / 256.0f32);
    } else {
        b = 1.3f32 / ps_enc.s_cmn.fs_k_hz as f32;
        ps_enc_ctrl.lf_ma_shp[0_usize] = -1.0f32 + b;
        ps_enc_ctrl.lf_ar_shp[0_usize] = 1.0f32 - b - b * strength * 0.6f32;
        k = 1;
        while k < ps_enc.s_cmn.nb_subfr as i32 {
            ps_enc_ctrl.lf_ma_shp[k as usize] = ps_enc_ctrl.lf_ma_shp[0_usize];
            ps_enc_ctrl.lf_ar_shp[k as usize] = ps_enc_ctrl.lf_ar_shp[0_usize];
            k += 1;
        }
        tilt = -HP_NOISE_COEF;
    }
    if USE_HARM_SHAPING != 0 && ps_enc.s_cmn.indices.signal_type as i32 == TYPE_VOICED {
        harm_shape_gain = HARMONIC_SHAPING;
        harm_shape_gain += HIGH_RATE_OR_LOW_QUALITY_HARMONIC_SHAPING
            * (1.0f32 - (1.0f32 - ps_enc_ctrl.coding_quality) * ps_enc_ctrl.input_quality);
        harm_shape_gain *= celt_sqrt(ps_enc.ltpcorr);
    } else {
        harm_shape_gain = 0.0f32;
    }
    k = 0;
    while k < ps_enc.s_cmn.nb_subfr as i32 {
        ps_shape_st.harm_shape_gain_smth +=
            SUBFR_SMTH_COEF * (harm_shape_gain - ps_shape_st.harm_shape_gain_smth);
        ps_enc_ctrl.harm_shape_gain[k as usize] = ps_shape_st.harm_shape_gain_smth;
        ps_shape_st.tilt_smth += SUBFR_SMTH_COEF * (tilt - ps_shape_st.tilt_smth);
        ps_enc_ctrl.tilt[k as usize] = ps_shape_st.tilt_smth;
        k += 1;
    }
}
