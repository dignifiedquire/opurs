//! Pitch lag decoding.
//!
//! Upstream C: `silk/decode_pitch.c`

use crate::silk::pitch_est_tables::{
    silk_CB_lags_stage2, silk_CB_lags_stage2_10_ms, silk_CB_lags_stage3, silk_CB_lags_stage3_10_ms,
    PE_MAX_LAG_MS, PE_MAX_NB_SUBFR, PE_MAX_NB_SUBFR_OVER_2, PE_MIN_LAG_MS, PE_NB_CBKS_STAGE2_10MS,
    PE_NB_CBKS_STAGE2_EXT, PE_NB_CBKS_STAGE3_10MS, PE_NB_CBKS_STAGE3_MAX,
};
use crate::silk::SigProc_FIX::silk_LIMIT;

/// Upstream C: silk/decode_pitch.c:silk_decode_pitch
///
/// Pitch analyzer function
///
/// ```text
/// lagIndex       I
/// contourIndex   O
/// pitch_lags[]   O   4 pitch values
/// Fs_kHz         I   sampling frequency (kHz)
/// nb_subfr       I   number of sub frames
/// ```
pub fn silk_decode_pitch(lagIndex: i16, contourIndex: i8, pitch_lags: &mut [i32], Fs_kHz: i32) {
    let nb_subfr = pitch_lags.len();

    let (lag_cb_flat, ncols): (&[i8], usize) = match (Fs_kHz, nb_subfr) {
        (8, PE_MAX_NB_SUBFR) => (&silk_CB_lags_stage2, PE_NB_CBKS_STAGE2_EXT),
        (8, PE_MAX_NB_SUBFR_OVER_2) => (&silk_CB_lags_stage2_10_ms, PE_NB_CBKS_STAGE2_10MS),
        (12 | 16, PE_MAX_NB_SUBFR) => (&silk_CB_lags_stage3, PE_NB_CBKS_STAGE3_MAX),
        (12 | 16, PE_MAX_NB_SUBFR_OVER_2) => (&silk_CB_lags_stage3_10_ms, PE_NB_CBKS_STAGE3_10MS),
        (Fs_kHz, nb_subfr) => {
            unreachable!("Fs_kHz: {}, nb_subfr: {}", Fs_kHz, nb_subfr)
        }
    };

    let min_lag = PE_MIN_LAG_MS * Fs_kHz as i16 as i32;
    let max_lag = PE_MAX_LAG_MS * Fs_kHz as i16 as i32;
    let lag = min_lag + lagIndex as i32;

    for (k, out_lag) in pitch_lags.iter_mut().enumerate() {
        let lag_cb_row = &lag_cb_flat[k * ncols..][..ncols];
        let lag = lag + lag_cb_row[contourIndex as usize] as i32;
        *out_lag = silk_LIMIT(lag, min_lag, max_lag);
    }
}
