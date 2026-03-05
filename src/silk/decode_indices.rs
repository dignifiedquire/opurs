//! Decoding of side-information indices.
//!
//! Upstream C: `silk/decode_indices.c`

use crate::celt::entdec::{ec_dec, ec_dec_icdf};
use crate::silk::define::{
    CODE_CONDITIONALLY, CODE_INDEPENDENTLY, MAX_NB_SUBFR, NLSF_QUANT_MAX_AMPLITUDE, TYPE_VOICED,
};
use crate::silk::structs::silk_decoder_state;
use crate::silk::tables_LTP::{SILK_LTP_GAIN_ICDF_PTRS, SILK_LTP_PER_INDEX_ICDF};
use crate::silk::tables_gain::{SILK_DELTA_GAIN_ICDF, SILK_GAIN_ICDF};
use crate::silk::tables_other::{
    SILK_LTPSCALE_ICDF, SILK_NLSF_EXT_ICDF, SILK_NLSF_INTERPOLATION_FACTOR_ICDF,
    SILK_TYPE_OFFSET_NO_VAD_ICDF, SILK_TYPE_OFFSET_VAD_ICDF, SILK_UNIFORM4_ICDF,
    SILK_UNIFORM8_ICDF,
};
use crate::silk::tables_pitch_lag::{SILK_PITCH_DELTA_ICDF, SILK_PITCH_LAG_ICDF};
use crate::silk::NLSF_unpack::silk_nlsf_unpack;

///
/// Decode side-information parameters from payload
///
/// ```text
/// psDec         I/O   State
/// psRangeDec    I/O   Compressor data structure
/// FrameIndex    I     Frame number
/// decode_LBRR   I     Flag indicating LBRR data is being decoded
/// condCoding    I     The type of conditional coding to use
/// ```
/// Upstream C: silk/decode_indices.c:silk_decode_indices
#[inline]
pub fn silk_decode_indices(
    psDec: &mut silk_decoder_state,
    psRangeDec: &mut ec_dec,
    FrameIndex: i32,
    decode_LBRR: i32,
    condCoding: i32,
) {
    /*******************************************/
    /* Decode signal type and quantizer offset */
    /*******************************************/
    let Ix = if decode_LBRR != 0 || psDec.VAD_flags[FrameIndex as usize] != 0 {
        ec_dec_icdf(psRangeDec, &SILK_TYPE_OFFSET_VAD_ICDF, 8) + 2
    } else {
        ec_dec_icdf(psRangeDec, &SILK_TYPE_OFFSET_NO_VAD_ICDF, 8)
    };
    psDec.indices.signalType = (Ix >> 1) as i8;
    psDec.indices.quantOffsetType = (Ix & 1) as i8;

    /****************/
    /* Decode gains */
    /****************/
    /* First subframe */
    if condCoding == CODE_CONDITIONALLY {
        /* Conditional coding */
        psDec.indices.GainsIndices[0] = ec_dec_icdf(psRangeDec, &SILK_DELTA_GAIN_ICDF, 8) as i8;
    } else {
        /* Independent coding, in two stages: MSB bits followed by 3 LSBs */
        psDec.indices.GainsIndices[0] = (ec_dec_icdf(
            psRangeDec,
            &SILK_GAIN_ICDF[psDec.indices.signalType as usize],
            8,
        ) as i8)
            << 3;
        psDec.indices.GainsIndices[0] += ec_dec_icdf(psRangeDec, &SILK_UNIFORM8_ICDF, 8) as i8;
    }

    /* Remaining subframes */
    for i in 1..psDec.nb_subfr {
        psDec.indices.GainsIndices[i] = ec_dec_icdf(psRangeDec, &SILK_DELTA_GAIN_ICDF, 8) as i8;
    }

    /**********************/
    /* Decode LSF Indices */
    /**********************/
    psDec.indices.NLSFIndices[0] = ec_dec_icdf(
        psRangeDec,
        &psDec.psNLSF_CB.CB1_iCDF
            [((psDec.indices.signalType as i32 >> 1) * psDec.psNLSF_CB.nVectors as i32) as usize..],
        8,
    ) as i8;

    let mut ec_ix: [i16; 16] = [0; 16];
    silk_nlsf_unpack(
        &mut ec_ix,
        &mut [0; 16],
        psDec.psNLSF_CB,
        psDec.indices.NLSFIndices[0] as i32,
    );
    debug_assert_eq!(psDec.psNLSF_CB.order as i32, psDec.LPC_order as i32);
    for (i, &ec_ix) in ec_ix
        .iter()
        .enumerate()
        .take(psDec.psNLSF_CB.order as usize)
    {
        let mut Ix = ec_dec_icdf(psRangeDec, &psDec.psNLSF_CB.ec_iCDF[ec_ix as usize..], 8);
        if Ix == 0 {
            Ix -= ec_dec_icdf(psRangeDec, &SILK_NLSF_EXT_ICDF, 8);
        } else if Ix == 2 * NLSF_QUANT_MAX_AMPLITUDE {
            Ix += ec_dec_icdf(psRangeDec, &SILK_NLSF_EXT_ICDF, 8);
        }
        psDec.indices.NLSFIndices[i + 1] = (Ix - NLSF_QUANT_MAX_AMPLITUDE) as i8;
    }

    /* Decode LSF interpolation factor */
    if psDec.nb_subfr == MAX_NB_SUBFR {
        psDec.indices.NLSFInterpCoef_Q2 =
            ec_dec_icdf(psRangeDec, &SILK_NLSF_INTERPOLATION_FACTOR_ICDF, 8) as i8;
    } else {
        psDec.indices.NLSFInterpCoef_Q2 = 4;
    }

    if psDec.indices.signalType as i32 == TYPE_VOICED {
        /*********************/
        /* Decode pitch lags */
        /*********************/
        /* Get lag index */
        let mut decode_absolute_lagIndex = true;
        if condCoding == CODE_CONDITIONALLY && psDec.ec_prevSignalType == TYPE_VOICED {
            /* Decode Delta index */
            let delta_lagIndex = ec_dec_icdf(psRangeDec, &SILK_PITCH_DELTA_ICDF, 8) as i16 as i32;
            if delta_lagIndex > 0 {
                let delta_lagIndex = delta_lagIndex - 9;
                psDec.indices.lagIndex = (psDec.ec_prevLagIndex as i32 + delta_lagIndex) as i16;
                decode_absolute_lagIndex = false;
            }
        }
        if decode_absolute_lagIndex {
            /* Absolute decoding */
            psDec.indices.lagIndex = (ec_dec_icdf(psRangeDec, &SILK_PITCH_LAG_ICDF, 8) as i16
                as i32
                * (psDec.fs_kHz >> 1)) as i16;
            psDec.indices.lagIndex = (psDec.indices.lagIndex as i32
                + ec_dec_icdf(psRangeDec, psDec.pitch_lag_low_bits_iCDF, 8) as i16 as i32)
                as i16;
        }
        psDec.ec_prevLagIndex = psDec.indices.lagIndex;

        /* Get countour index */
        psDec.indices.contourIndex = ec_dec_icdf(psRangeDec, psDec.pitch_contour_iCDF, 8) as i8;

        /********************/
        /* Decode LTP gains */
        /********************/
        /* Decode PERIndex value */
        psDec.indices.PERIndex = ec_dec_icdf(psRangeDec, &SILK_LTP_PER_INDEX_ICDF, 8) as i8;
        for k in 0..psDec.nb_subfr {
            psDec.indices.LTPIndex[k] = ec_dec_icdf(
                psRangeDec,
                SILK_LTP_GAIN_ICDF_PTRS[psDec.indices.PERIndex as usize],
                8,
            ) as i8;
        }

        /**********************/
        /* Decode LTP scaling */
        /**********************/
        if condCoding == CODE_INDEPENDENTLY {
            psDec.indices.LTP_scaleIndex = ec_dec_icdf(psRangeDec, &SILK_LTPSCALE_ICDF, 8) as i8;
        } else {
            psDec.indices.LTP_scaleIndex = 0;
        }
    }
    psDec.ec_prevSignalType = psDec.indices.signalType as i32;

    /***************/
    /* Decode seed */
    /***************/
    psDec.indices.Seed = ec_dec_icdf(psRangeDec, &SILK_UNIFORM4_ICDF, 8) as i8;
}
