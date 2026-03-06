//! Decoding of side-information indices.
//!
//! Upstream c: `silk/decode_indices.c`

use crate::celt::entdec::{ec_dec_icdf, EcDec};
use crate::silk::define::{
    CODE_CONDITIONALLY, CODE_INDEPENDENTLY, MAX_NB_SUBFR, NLSF_QUANT_MAX_AMPLITUDE, TYPE_VOICED,
};
use crate::silk::nlsf_unpack::silk_nlsf_unpack;
use crate::silk::structs::silk_decoder_state;
use crate::silk::tables_gain::{SILK_DELTA_GAIN_ICDF, SILK_GAIN_ICDF};
use crate::silk::tables_ltp::{SILK_LTP_GAIN_ICDF_PTRS, SILK_LTP_PER_INDEX_ICDF};
use crate::silk::tables_other::{
    SILK_LTPSCALE_ICDF, SILK_NLSF_EXT_ICDF, SILK_NLSF_INTERPOLATION_FACTOR_ICDF,
    SILK_TYPE_OFFSET_NO_VAD_ICDF, SILK_TYPE_OFFSET_VAD_ICDF, SILK_UNIFORM4_ICDF,
    SILK_UNIFORM8_ICDF,
};
use crate::silk::tables_pitch_lag::{SILK_PITCH_DELTA_ICDF, SILK_PITCH_LAG_ICDF};

///
/// Decode side-information parameters from payload
///
/// ```text
/// ps_dec         _i/O   State
/// ps_range_dec    _i/O   Compressor data structure
/// frame_index    _i     Frame number
/// decode_lbrr   _i     Flag indicating LBRR data is being decoded
/// cond_coding    _i     The type of conditional coding to use
/// ```
/// Upstream c: silk/decode_indices.c:silk_decode_indices
#[inline]
pub fn silk_decode_indices(
    ps_dec: &mut silk_decoder_state,
    ps_range_dec: &mut EcDec,
    frame_index: i32,
    decode_lbrr: i32,
    cond_coding: i32,
) {
    /*******************************************/
    /* Decode signal type and quantizer offset */
    /*******************************************/
    let ix = if decode_lbrr != 0 || ps_dec.vad_flags[frame_index as usize] != 0 {
        ec_dec_icdf(ps_range_dec, &SILK_TYPE_OFFSET_VAD_ICDF, 8) + 2
    } else {
        ec_dec_icdf(ps_range_dec, &SILK_TYPE_OFFSET_NO_VAD_ICDF, 8)
    };
    ps_dec.indices.signal_type = (ix >> 1) as i8;
    ps_dec.indices.quant_offset_type = (ix & 1) as i8;

    /****************/
    /* Decode gains */
    /****************/
    /* First subframe */
    if cond_coding == CODE_CONDITIONALLY {
        /* Conditional coding */
        ps_dec.indices.gains_indices[0] = ec_dec_icdf(ps_range_dec, &SILK_DELTA_GAIN_ICDF, 8) as i8;
    } else {
        /* Independent coding, in two stages: MSB bits followed by 3 LSBs */
        ps_dec.indices.gains_indices[0] = (ec_dec_icdf(
            ps_range_dec,
            &SILK_GAIN_ICDF[ps_dec.indices.signal_type as usize],
            8,
        ) as i8)
            << 3;
        ps_dec.indices.gains_indices[0] += ec_dec_icdf(ps_range_dec, &SILK_UNIFORM8_ICDF, 8) as i8;
    }

    /* Remaining subframes */
    for _i in 1..ps_dec.nb_subfr {
        ps_dec.indices.gains_indices[_i] =
            ec_dec_icdf(ps_range_dec, &SILK_DELTA_GAIN_ICDF, 8) as i8;
    }

    /**********************/
    /* Decode LSF Indices */
    /**********************/
    ps_dec.indices.nlsfindices[0] = ec_dec_icdf(
        ps_range_dec,
        &ps_dec.ps_nlsf_cb.cb1_i_cdf[((ps_dec.indices.signal_type as i32 >> 1)
            * ps_dec.ps_nlsf_cb.n_vectors as i32) as usize..],
        8,
    ) as i8;

    let mut ec_ix: [i16; 16] = [0; 16];
    silk_nlsf_unpack(
        &mut ec_ix,
        &mut [0; 16],
        ps_dec.ps_nlsf_cb,
        ps_dec.indices.nlsfindices[0] as i32,
    );
    debug_assert_eq!(ps_dec.ps_nlsf_cb.order as i32, ps_dec.lpc_order as i32);
    for (_i, &ec_ix) in ec_ix
        .iter()
        .enumerate()
        .take(ps_dec.ps_nlsf_cb.order as usize)
    {
        let mut ix = ec_dec_icdf(
            ps_range_dec,
            &ps_dec.ps_nlsf_cb.ec_i_cdf[ec_ix as usize..],
            8,
        );
        if ix == 0 {
            ix -= ec_dec_icdf(ps_range_dec, &SILK_NLSF_EXT_ICDF, 8);
        } else if ix == 2 * NLSF_QUANT_MAX_AMPLITUDE {
            ix += ec_dec_icdf(ps_range_dec, &SILK_NLSF_EXT_ICDF, 8);
        }
        ps_dec.indices.nlsfindices[_i + 1] = (ix - NLSF_QUANT_MAX_AMPLITUDE) as i8;
    }

    /* Decode LSF interpolation factor */
    if ps_dec.nb_subfr == MAX_NB_SUBFR {
        ps_dec.indices.nlsfinterp_coef_q2 =
            ec_dec_icdf(ps_range_dec, &SILK_NLSF_INTERPOLATION_FACTOR_ICDF, 8) as i8;
    } else {
        ps_dec.indices.nlsfinterp_coef_q2 = 4;
    }

    if ps_dec.indices.signal_type as i32 == TYPE_VOICED {
        /*********************/
        /* Decode pitch lags */
        /*********************/
        /* Get lag index */
        let mut decode_absolute_lag_index = true;
        if cond_coding == CODE_CONDITIONALLY && ps_dec.ec_prev_signal_type == TYPE_VOICED {
            /* Decode Delta index */
            let delta_lag_index =
                ec_dec_icdf(ps_range_dec, &SILK_PITCH_DELTA_ICDF, 8) as i16 as i32;
            if delta_lag_index > 0 {
                let delta_lag_index = delta_lag_index - 9;
                ps_dec.indices.lag_index =
                    (ps_dec.ec_prev_lag_index as i32 + delta_lag_index) as i16;
                decode_absolute_lag_index = false;
            }
        }
        if decode_absolute_lag_index {
            /* Absolute decoding */
            ps_dec.indices.lag_index = (ec_dec_icdf(ps_range_dec, &SILK_PITCH_LAG_ICDF, 8) as i16
                as i32
                * (ps_dec.fs_k_hz >> 1)) as i16;
            ps_dec.indices.lag_index = (ps_dec.indices.lag_index as i32
                + ec_dec_icdf(ps_range_dec, ps_dec.pitch_lag_low_bits_i_cdf, 8) as i16 as i32)
                as i16;
        }
        ps_dec.ec_prev_lag_index = ps_dec.indices.lag_index;

        /* Get countour index */
        ps_dec.indices.contour_index =
            ec_dec_icdf(ps_range_dec, ps_dec.pitch_contour_i_cdf, 8) as i8;

        /********************/
        /* Decode LTP gains */
        /********************/
        /* Decode perindex value */
        ps_dec.indices.perindex = ec_dec_icdf(ps_range_dec, &SILK_LTP_PER_INDEX_ICDF, 8) as i8;
        for k in 0..ps_dec.nb_subfr {
            ps_dec.indices.ltpindex[k] = ec_dec_icdf(
                ps_range_dec,
                SILK_LTP_GAIN_ICDF_PTRS[ps_dec.indices.perindex as usize],
                8,
            ) as i8;
        }

        /**********************/
        /* Decode LTP scaling */
        /**********************/
        if cond_coding == CODE_INDEPENDENTLY {
            ps_dec.indices.ltp_scale_index =
                ec_dec_icdf(ps_range_dec, &SILK_LTPSCALE_ICDF, 8) as i8;
        } else {
            ps_dec.indices.ltp_scale_index = 0;
        }
    }
    ps_dec.ec_prev_signal_type = ps_dec.indices.signal_type as i32;

    /***************/
    /* Decode seed */
    /***************/
    ps_dec.indices.seed = ec_dec_icdf(ps_range_dec, &SILK_UNIFORM4_ICDF, 8) as i8;
}
