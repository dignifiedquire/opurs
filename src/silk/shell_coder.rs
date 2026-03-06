//! Shell coder for pulse distribution.
//!
//! Upstream C: `silk/shell_coder.c`

#![forbid(unsafe_code)]

//! shell coder; pulse-subframe length is hardcoded

use crate::celt::entdec::{ec_dec_icdf, EcDec};
use crate::celt::entenc::{ec_enc_icdf, EcEnc};
use crate::silk::define::SHELL_CODEC_FRAME_LENGTH;
use crate::silk::tables_pulses_per_block::{
    SILK_SHELL_CODE_TABLE0, SILK_SHELL_CODE_TABLE1, SILK_SHELL_CODE_TABLE2, SILK_SHELL_CODE_TABLE3,
    SILK_SHELL_CODE_TABLE_OFFSETS,
};

/// Upstream C: silk/shell_coder.c:combine_pulses
#[inline]
fn combine_pulses(out: &mut [i32], in_: &[i32]) {
    debug_assert_eq!(out.len() * 2, in_.len());

    for (out, in_) in out.iter_mut().zip(in_.chunks_exact(2)) {
        *out = in_.iter().sum::<i32>();
    }
}

/// Upstream C: silk/shell_coder.c:encode_split
#[inline]
fn encode_split(psRangeEnc: &mut EcEnc, p_child1: i32, p: i32, shell_table: &[u8]) {
    if p > 0 {
        ec_enc_icdf(
            psRangeEnc,
            p_child1,
            &shell_table[SILK_SHELL_CODE_TABLE_OFFSETS[p as usize] as usize..],
            8,
        );
    }
}

/// Upstream C: silk/shell_coder.c:decode_split
#[inline]
fn decode_split(p_child: &mut [i16], psRangeDec: &mut EcDec, p: i32, shell_table: &[u8]) {
    debug_assert_eq!(p_child.len(), 2);

    if p > 0 {
        p_child[0] = ec_dec_icdf(
            psRangeDec,
            &shell_table[SILK_SHELL_CODE_TABLE_OFFSETS[p as usize] as usize..],
            8,
        ) as i16;
        p_child[1] = (p - p_child[0] as i32) as i16;
    } else {
        p_child[0] = 0;
        p_child[1] = 0;
    };
}

///
/// Shell encoder, operates on one shell code frame of 16 pulses
/// Upstream C: silk/shell_coder.c:silk_shell_encoder
pub fn silk_shell_encoder(psRangeEnc: &mut EcEnc, pulses0: &[i32]) {
    let mut pulses1: [i32; 8] = [0; 8];
    let mut pulses2: [i32; 4] = [0; 4];
    let mut pulses3: [i32; 2] = [0; 2];
    let mut pulses4: [i32; 1] = [0; 1];

    debug_assert_eq!(pulses0.len(), SHELL_CODEC_FRAME_LENGTH);
    debug_assert_eq!(SHELL_CODEC_FRAME_LENGTH, 16);

    combine_pulses(&mut pulses1, pulses0);
    combine_pulses(&mut pulses2, &pulses1);
    combine_pulses(&mut pulses3, &pulses2);
    combine_pulses(&mut pulses4, &pulses3);
    encode_split(psRangeEnc, pulses3[0], pulses4[0], &SILK_SHELL_CODE_TABLE3);
    encode_split(psRangeEnc, pulses2[0], pulses3[0], &SILK_SHELL_CODE_TABLE2);
    encode_split(psRangeEnc, pulses1[0], pulses2[0], &SILK_SHELL_CODE_TABLE1);
    encode_split(psRangeEnc, pulses0[0], pulses1[0], &SILK_SHELL_CODE_TABLE0);
    encode_split(psRangeEnc, pulses0[2], pulses1[1], &SILK_SHELL_CODE_TABLE0);
    encode_split(psRangeEnc, pulses1[2], pulses2[1], &SILK_SHELL_CODE_TABLE1);
    encode_split(psRangeEnc, pulses0[4], pulses1[2], &SILK_SHELL_CODE_TABLE0);
    encode_split(psRangeEnc, pulses0[6], pulses1[3], &SILK_SHELL_CODE_TABLE0);
    encode_split(psRangeEnc, pulses2[2], pulses3[1], &SILK_SHELL_CODE_TABLE2);
    encode_split(psRangeEnc, pulses1[4], pulses2[2], &SILK_SHELL_CODE_TABLE1);
    encode_split(psRangeEnc, pulses0[8], pulses1[4], &SILK_SHELL_CODE_TABLE0);
    encode_split(psRangeEnc, pulses0[10], pulses1[5], &SILK_SHELL_CODE_TABLE0);
    encode_split(psRangeEnc, pulses1[6], pulses2[3], &SILK_SHELL_CODE_TABLE1);
    encode_split(psRangeEnc, pulses0[12], pulses1[6], &SILK_SHELL_CODE_TABLE0);
    encode_split(psRangeEnc, pulses0[14], pulses1[7], &SILK_SHELL_CODE_TABLE0);
}

/// Upstream C: silk/shell_coder.c:silk_shell_decoder
#[inline]
pub fn silk_shell_decoder(pulses0: &mut [i16], psRangeDec: &mut EcDec, pulses4: i32) {
    debug_assert_eq!(pulses0.len(), SHELL_CODEC_FRAME_LENGTH);
    debug_assert_eq!(SHELL_CODEC_FRAME_LENGTH, 16);

    let mut pulses3: [i16; 2] = [0; 2];
    let mut pulses2: [i16; 4] = [0; 4];
    let mut pulses1: [i16; 8] = [0; 8];
    decode_split(
        &mut pulses3[0..=1],
        psRangeDec,
        pulses4,
        &SILK_SHELL_CODE_TABLE3,
    );
    decode_split(
        &mut pulses2[0..=1],
        psRangeDec,
        pulses3[0] as i32,
        &SILK_SHELL_CODE_TABLE2,
    );
    decode_split(
        &mut pulses1[0..=1],
        psRangeDec,
        pulses2[0] as i32,
        &SILK_SHELL_CODE_TABLE1,
    );
    decode_split(
        &mut pulses0[0..=1],
        psRangeDec,
        pulses1[0] as i32,
        &SILK_SHELL_CODE_TABLE0,
    );
    decode_split(
        &mut pulses0[2..=3],
        psRangeDec,
        pulses1[1] as i32,
        &SILK_SHELL_CODE_TABLE0,
    );
    decode_split(
        &mut pulses1[2..=3],
        psRangeDec,
        pulses2[1] as i32,
        &SILK_SHELL_CODE_TABLE1,
    );
    decode_split(
        &mut pulses0[4..=5],
        psRangeDec,
        pulses1[2] as i32,
        &SILK_SHELL_CODE_TABLE0,
    );
    decode_split(
        &mut pulses0[6..=7],
        psRangeDec,
        pulses1[3] as i32,
        &SILK_SHELL_CODE_TABLE0,
    );
    decode_split(
        &mut pulses2[2..=3],
        psRangeDec,
        pulses3[1] as i32,
        &SILK_SHELL_CODE_TABLE2,
    );
    decode_split(
        &mut pulses1[4..=5],
        psRangeDec,
        pulses2[2] as i32,
        &SILK_SHELL_CODE_TABLE1,
    );
    decode_split(
        &mut pulses0[8..=9],
        psRangeDec,
        pulses1[4] as i32,
        &SILK_SHELL_CODE_TABLE0,
    );
    decode_split(
        &mut pulses0[10..=11],
        psRangeDec,
        pulses1[5] as i32,
        &SILK_SHELL_CODE_TABLE0,
    );
    decode_split(
        &mut pulses1[6..=7],
        psRangeDec,
        pulses2[3] as i32,
        &SILK_SHELL_CODE_TABLE1,
    );
    decode_split(
        &mut pulses0[12..=13],
        psRangeDec,
        pulses1[6] as i32,
        &SILK_SHELL_CODE_TABLE0,
    );
    decode_split(
        &mut pulses0[14..=15],
        psRangeDec,
        pulses1[7] as i32,
        &SILK_SHELL_CODE_TABLE0,
    );
}
