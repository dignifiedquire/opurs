//! Encoding of side-information indices.
//!
//! Upstream c: `silk/encode_indices.c`

use crate::celt::entenc::{ec_enc_icdf, EcEnc};
use crate::silk::define::{
    CODE_CONDITIONALLY, CODE_INDEPENDENTLY, MAX_NB_SUBFR, NLSF_QUANT_MAX_AMPLITUDE, TYPE_VOICED,
};
use crate::silk::nlsf_unpack::silk_nlsf_unpack;
use crate::silk::structs::{silk_encoder_state, SideInfoIndices};
use crate::silk::tables_gain::{SILK_DELTA_GAIN_ICDF, SILK_GAIN_ICDF};
use crate::silk::tables_ltp::{SILK_LTP_GAIN_ICDF_PTRS, SILK_LTP_PER_INDEX_ICDF};
use crate::silk::tables_other::{
    SILK_LTPSCALE_ICDF, SILK_NLSF_EXT_ICDF, SILK_NLSF_INTERPOLATION_FACTOR_ICDF,
    SILK_TYPE_OFFSET_NO_VAD_ICDF, SILK_TYPE_OFFSET_VAD_ICDF, SILK_UNIFORM4_ICDF,
    SILK_UNIFORM8_ICDF,
};
use crate::silk::tables_pitch_lag::{SILK_PITCH_DELTA_ICDF, SILK_PITCH_LAG_ICDF};

/// Upstream c: silk/encode_indices.c:silk_encode_indices
pub fn silk_encode_indices(
    ps_enc_c: &mut silk_encoder_state,
    ps_range_enc: &mut EcEnc,
    frame_index: i32,
    encode_lbrr: i32,
    cond_coding: i32,
) {
    let mut _i: i32;
    let mut k: i32;

    let mut encode_absolute_lag_index: i32;
    let mut delta_lag_index: i32;
    let mut ec_ix: [i16; 16] = [0; 16];
    let mut pred_q8: [u8; 16] = [0; 16];
    let ps_indices: SideInfoIndices = if encode_lbrr != 0 {
        ps_enc_c.indices_lbrr[frame_index as usize]
    } else {
        ps_enc_c.indices
    };
    let type_offset: i32 = 2 * ps_indices.signal_type as i32 + ps_indices.quant_offset_type as i32;
    debug_assert!((0..6).contains(&type_offset));
    debug_assert!(encode_lbrr == 0 || type_offset >= 2);
    if encode_lbrr != 0 || type_offset >= 2 {
        ec_enc_icdf(ps_range_enc, type_offset - 2, &SILK_TYPE_OFFSET_VAD_ICDF, 8);
    } else {
        ec_enc_icdf(ps_range_enc, type_offset, &SILK_TYPE_OFFSET_NO_VAD_ICDF, 8);
    }
    if cond_coding == CODE_CONDITIONALLY {
        ec_enc_icdf(
            ps_range_enc,
            ps_indices.gains_indices[0_usize] as i32,
            &SILK_DELTA_GAIN_ICDF,
            8,
        );
    } else {
        ec_enc_icdf(
            ps_range_enc,
            ps_indices.gains_indices[0_usize] as i32 >> 3,
            &SILK_GAIN_ICDF[ps_indices.signal_type as usize],
            8,
        );
        ec_enc_icdf(
            ps_range_enc,
            ps_indices.gains_indices[0_usize] as i32 & 7,
            &SILK_UNIFORM8_ICDF,
            8,
        );
    }
    _i = 1;
    while _i < ps_enc_c.nb_subfr as i32 {
        ec_enc_icdf(
            ps_range_enc,
            ps_indices.gains_indices[_i as usize] as i32,
            &SILK_DELTA_GAIN_ICDF,
            8,
        );
        _i += 1;
    }
    ec_enc_icdf(
        ps_range_enc,
        ps_indices.nlsfindices[0_usize] as i32,
        &ps_enc_c.ps_nlsf_cb.cb1_i_cdf[((ps_indices.signal_type as i32 >> 1)
            * ps_enc_c.ps_nlsf_cb.n_vectors as i32)
            as usize..],
        8,
    );
    silk_nlsf_unpack(
        &mut ec_ix,
        &mut pred_q8,
        ps_enc_c.ps_nlsf_cb,
        ps_indices.nlsfindices[0_usize] as i32,
    );
    debug_assert!(ps_enc_c.ps_nlsf_cb.order as i32 == ps_enc_c.predict_lpcorder);
    _i = 0;
    while _i < ps_enc_c.ps_nlsf_cb.order as i32 {
        if ps_indices.nlsfindices[(_i + 1) as usize] as i32 >= NLSF_QUANT_MAX_AMPLITUDE {
            ec_enc_icdf(
                ps_range_enc,
                2 * NLSF_QUANT_MAX_AMPLITUDE,
                &ps_enc_c.ps_nlsf_cb.ec_i_cdf[ec_ix[_i as usize] as usize..],
                8,
            );
            ec_enc_icdf(
                ps_range_enc,
                ps_indices.nlsfindices[(_i + 1) as usize] as i32 - NLSF_QUANT_MAX_AMPLITUDE,
                &SILK_NLSF_EXT_ICDF,
                8,
            );
        } else if ps_indices.nlsfindices[(_i + 1) as usize] as i32 <= -NLSF_QUANT_MAX_AMPLITUDE {
            ec_enc_icdf(
                ps_range_enc,
                0,
                &ps_enc_c.ps_nlsf_cb.ec_i_cdf[ec_ix[_i as usize] as usize..],
                8,
            );
            ec_enc_icdf(
                ps_range_enc,
                -(ps_indices.nlsfindices[(_i + 1) as usize] as i32) - NLSF_QUANT_MAX_AMPLITUDE,
                &SILK_NLSF_EXT_ICDF,
                8,
            );
        } else {
            ec_enc_icdf(
                ps_range_enc,
                ps_indices.nlsfindices[(_i + 1) as usize] as i32 + NLSF_QUANT_MAX_AMPLITUDE,
                &ps_enc_c.ps_nlsf_cb.ec_i_cdf[ec_ix[_i as usize] as usize..],
                8,
            );
        }
        _i += 1;
    }
    if ps_enc_c.nb_subfr == MAX_NB_SUBFR {
        ec_enc_icdf(
            ps_range_enc,
            ps_indices.nlsfinterp_coef_q2 as i32,
            &SILK_NLSF_INTERPOLATION_FACTOR_ICDF,
            8,
        );
    }
    if ps_indices.signal_type as i32 == TYPE_VOICED {
        encode_absolute_lag_index = 1;
        if cond_coding == CODE_CONDITIONALLY && ps_enc_c.ec_prev_signal_type == TYPE_VOICED {
            delta_lag_index = ps_indices.lag_index as i32 - ps_enc_c.ec_prev_lag_index as i32;
            if !(-(8)..=11).contains(&delta_lag_index) {
                delta_lag_index = 0;
            } else {
                delta_lag_index += 9;
                encode_absolute_lag_index = 0;
            }
            ec_enc_icdf(ps_range_enc, delta_lag_index, &SILK_PITCH_DELTA_ICDF, 8);
        }
        if encode_absolute_lag_index != 0 {
            let pitch_high_bits: i32 = ps_indices.lag_index as i32 / (ps_enc_c.fs_k_hz >> 1);
            let pitch_low_bits: i32 = ps_indices.lag_index as i32
                - pitch_high_bits as i16 as i32 * (ps_enc_c.fs_k_hz >> 1) as i16 as i32;
            ec_enc_icdf(ps_range_enc, pitch_high_bits, &SILK_PITCH_LAG_ICDF, 8);
            ec_enc_icdf(
                ps_range_enc,
                pitch_low_bits,
                ps_enc_c.pitch_lag_low_bits_i_cdf,
                8,
            );
        }
        ps_enc_c.ec_prev_lag_index = ps_indices.lag_index;
        ec_enc_icdf(
            ps_range_enc,
            ps_indices.contour_index as i32,
            ps_enc_c.pitch_contour_i_cdf,
            8,
        );
        ec_enc_icdf(
            ps_range_enc,
            ps_indices.perindex as i32,
            &SILK_LTP_PER_INDEX_ICDF,
            8,
        );
        k = 0;
        while k < ps_enc_c.nb_subfr as i32 {
            ec_enc_icdf(
                ps_range_enc,
                ps_indices.ltpindex[k as usize] as i32,
                SILK_LTP_GAIN_ICDF_PTRS[ps_indices.perindex as usize],
                8,
            );
            k += 1;
        }
        if cond_coding == CODE_INDEPENDENTLY {
            ec_enc_icdf(
                ps_range_enc,
                ps_indices.ltp_scale_index as i32,
                &SILK_LTPSCALE_ICDF,
                8,
            );
        }
    }
    ps_enc_c.ec_prev_signal_type = ps_indices.signal_type as i32;
    ec_enc_icdf(ps_range_enc, ps_indices.seed as i32, &SILK_UNIFORM4_ICDF, 8);
}
