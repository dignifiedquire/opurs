//! Encoding of side-information indices.
//!
//! Upstream C: `silk/encode_indices.c`

use crate::celt::entenc::{ec_enc, ec_enc_icdf};
use crate::silk::define::{
    CODE_CONDITIONALLY, CODE_INDEPENDENTLY, MAX_NB_SUBFR, NLSF_QUANT_MAX_AMPLITUDE, TYPE_VOICED,
};
use crate::silk::nlsf_unpack::silk_nlsf_unpack;
use crate::silk::structs::{silk_encoder_state, SideInfoIndices};
use crate::silk::tables_LTP::{SILK_LTP_GAIN_ICDF_PTRS, SILK_LTP_PER_INDEX_ICDF};
use crate::silk::tables_gain::{SILK_DELTA_GAIN_ICDF, SILK_GAIN_ICDF};
use crate::silk::tables_other::{
    SILK_LTPSCALE_ICDF, SILK_NLSF_EXT_ICDF, SILK_NLSF_INTERPOLATION_FACTOR_ICDF,
    SILK_TYPE_OFFSET_NO_VAD_ICDF, SILK_TYPE_OFFSET_VAD_ICDF, SILK_UNIFORM4_ICDF,
    SILK_UNIFORM8_ICDF,
};
use crate::silk::tables_pitch_lag::{SILK_PITCH_DELTA_ICDF, SILK_PITCH_LAG_ICDF};

/// Upstream C: silk/encode_indices.c:silk_encode_indices
pub fn silk_encode_indices(
    psEncC: &mut silk_encoder_state,
    psRangeEnc: &mut ec_enc,
    FrameIndex: i32,
    encode_LBRR: i32,
    condCoding: i32,
) {
    let mut i: i32;
    let mut k: i32;

    let mut encode_absolute_lagIndex: i32;
    let mut delta_lagIndex: i32;
    let mut ec_ix: [i16; 16] = [0; 16];
    let mut pred_Q8: [u8; 16] = [0; 16];
    let psIndices: SideInfoIndices = if encode_LBRR != 0 {
        psEncC.indices_LBRR[FrameIndex as usize]
    } else {
        psEncC.indices
    };
    let typeOffset: i32 = 2 * psIndices.signalType as i32 + psIndices.quantOffsetType as i32;
    debug_assert!((0..6).contains(&typeOffset));
    debug_assert!(encode_LBRR == 0 || typeOffset >= 2);
    if encode_LBRR != 0 || typeOffset >= 2 {
        ec_enc_icdf(psRangeEnc, typeOffset - 2, &SILK_TYPE_OFFSET_VAD_ICDF, 8);
    } else {
        ec_enc_icdf(psRangeEnc, typeOffset, &SILK_TYPE_OFFSET_NO_VAD_ICDF, 8);
    }
    if condCoding == CODE_CONDITIONALLY {
        ec_enc_icdf(
            psRangeEnc,
            psIndices.GainsIndices[0_usize] as i32,
            &SILK_DELTA_GAIN_ICDF,
            8,
        );
    } else {
        ec_enc_icdf(
            psRangeEnc,
            psIndices.GainsIndices[0_usize] as i32 >> 3,
            &SILK_GAIN_ICDF[psIndices.signalType as usize],
            8,
        );
        ec_enc_icdf(
            psRangeEnc,
            psIndices.GainsIndices[0_usize] as i32 & 7,
            &SILK_UNIFORM8_ICDF,
            8,
        );
    }
    i = 1;
    while i < psEncC.nb_subfr as i32 {
        ec_enc_icdf(
            psRangeEnc,
            psIndices.GainsIndices[i as usize] as i32,
            &SILK_DELTA_GAIN_ICDF,
            8,
        );
        i += 1;
    }
    ec_enc_icdf(
        psRangeEnc,
        psIndices.NLSFIndices[0_usize] as i32,
        &psEncC.psNLSF_CB.CB1_iCDF
            [((psIndices.signalType as i32 >> 1) * psEncC.psNLSF_CB.nVectors as i32) as usize..],
        8,
    );
    silk_nlsf_unpack(
        &mut ec_ix,
        &mut pred_Q8,
        psEncC.psNLSF_CB,
        psIndices.NLSFIndices[0_usize] as i32,
    );
    debug_assert!(psEncC.psNLSF_CB.order as i32 == psEncC.predictLPCOrder);
    i = 0;
    while i < psEncC.psNLSF_CB.order as i32 {
        if psIndices.NLSFIndices[(i + 1) as usize] as i32 >= NLSF_QUANT_MAX_AMPLITUDE {
            ec_enc_icdf(
                psRangeEnc,
                2 * NLSF_QUANT_MAX_AMPLITUDE,
                &psEncC.psNLSF_CB.ec_iCDF[ec_ix[i as usize] as usize..],
                8,
            );
            ec_enc_icdf(
                psRangeEnc,
                psIndices.NLSFIndices[(i + 1) as usize] as i32 - NLSF_QUANT_MAX_AMPLITUDE,
                &SILK_NLSF_EXT_ICDF,
                8,
            );
        } else if psIndices.NLSFIndices[(i + 1) as usize] as i32 <= -NLSF_QUANT_MAX_AMPLITUDE {
            ec_enc_icdf(
                psRangeEnc,
                0,
                &psEncC.psNLSF_CB.ec_iCDF[ec_ix[i as usize] as usize..],
                8,
            );
            ec_enc_icdf(
                psRangeEnc,
                -(psIndices.NLSFIndices[(i + 1) as usize] as i32) - NLSF_QUANT_MAX_AMPLITUDE,
                &SILK_NLSF_EXT_ICDF,
                8,
            );
        } else {
            ec_enc_icdf(
                psRangeEnc,
                psIndices.NLSFIndices[(i + 1) as usize] as i32 + NLSF_QUANT_MAX_AMPLITUDE,
                &psEncC.psNLSF_CB.ec_iCDF[ec_ix[i as usize] as usize..],
                8,
            );
        }
        i += 1;
    }
    if psEncC.nb_subfr == MAX_NB_SUBFR {
        ec_enc_icdf(
            psRangeEnc,
            psIndices.NLSFInterpCoef_Q2 as i32,
            &SILK_NLSF_INTERPOLATION_FACTOR_ICDF,
            8,
        );
    }
    if psIndices.signalType as i32 == TYPE_VOICED {
        encode_absolute_lagIndex = 1;
        if condCoding == CODE_CONDITIONALLY && psEncC.ec_prevSignalType == TYPE_VOICED {
            delta_lagIndex = psIndices.lagIndex as i32 - psEncC.ec_prevLagIndex as i32;
            if !(-(8)..=11).contains(&delta_lagIndex) {
                delta_lagIndex = 0;
            } else {
                delta_lagIndex += 9;
                encode_absolute_lagIndex = 0;
            }
            ec_enc_icdf(psRangeEnc, delta_lagIndex, &SILK_PITCH_DELTA_ICDF, 8);
        }
        if encode_absolute_lagIndex != 0 {
            let pitch_high_bits: i32 = psIndices.lagIndex as i32 / (psEncC.fs_kHz >> 1);
            let pitch_low_bits: i32 = psIndices.lagIndex as i32
                - pitch_high_bits as i16 as i32 * (psEncC.fs_kHz >> 1) as i16 as i32;
            ec_enc_icdf(psRangeEnc, pitch_high_bits, &SILK_PITCH_LAG_ICDF, 8);
            ec_enc_icdf(
                psRangeEnc,
                pitch_low_bits,
                psEncC.pitch_lag_low_bits_iCDF,
                8,
            );
        }
        psEncC.ec_prevLagIndex = psIndices.lagIndex;
        ec_enc_icdf(
            psRangeEnc,
            psIndices.contourIndex as i32,
            psEncC.pitch_contour_iCDF,
            8,
        );
        ec_enc_icdf(
            psRangeEnc,
            psIndices.PERIndex as i32,
            &SILK_LTP_PER_INDEX_ICDF,
            8,
        );
        k = 0;
        while k < psEncC.nb_subfr as i32 {
            ec_enc_icdf(
                psRangeEnc,
                psIndices.LTPIndex[k as usize] as i32,
                SILK_LTP_GAIN_ICDF_PTRS[psIndices.PERIndex as usize],
                8,
            );
            k += 1;
        }
        if condCoding == CODE_INDEPENDENTLY {
            ec_enc_icdf(
                psRangeEnc,
                psIndices.LTP_scaleIndex as i32,
                &SILK_LTPSCALE_ICDF,
                8,
            );
        }
    }
    psEncC.ec_prevSignalType = psIndices.signalType as i32;
    ec_enc_icdf(psRangeEnc, psIndices.Seed as i32, &SILK_UNIFORM4_ICDF, 8);
}
