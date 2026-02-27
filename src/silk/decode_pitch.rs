//! Pitch lag decoding.
//!
//! Upstream C: `silk/decode_pitch.c`

use crate::silk::pitch_est_tables::{
    silk_CB_lags_stage2, silk_CB_lags_stage2_10_ms, silk_CB_lags_stage3, silk_CB_lags_stage3_10_ms,
    PE_MAX_LAG_MS, PE_MAX_NB_SUBFR, PE_MAX_NB_SUBFR_OVER_2, PE_MIN_LAG_MS, PE_NB_CBKS_STAGE2_10MS,
    PE_NB_CBKS_STAGE2_EXT, PE_NB_CBKS_STAGE3_10MS, PE_NB_CBKS_STAGE3_MAX,
};
use crate::silk::SigProc_FIX::silk_LIMIT;

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
/// Upstream C: silk/decode_pitch.c:silk_decode_pitch
pub fn silk_decode_pitch(lagIndex: i16, contourIndex: i8, pitch_lags: &mut [i32], Fs_kHz: i32) {
    let nb_subfr = pitch_lags.len();

    let (lag_cb_flat, ncols): (&[i8], usize) = if Fs_kHz == 8 {
        if nb_subfr == PE_MAX_NB_SUBFR {
            (&silk_CB_lags_stage2, PE_NB_CBKS_STAGE2_EXT)
        } else {
            debug_assert_eq!(nb_subfr, PE_MAX_NB_SUBFR_OVER_2);
            (&silk_CB_lags_stage2_10_ms, PE_NB_CBKS_STAGE2_10MS)
        }
    } else if nb_subfr == PE_MAX_NB_SUBFR {
        (&silk_CB_lags_stage3, PE_NB_CBKS_STAGE3_MAX)
    } else {
        debug_assert_eq!(nb_subfr, PE_MAX_NB_SUBFR_OVER_2);
        (&silk_CB_lags_stage3_10_ms, PE_NB_CBKS_STAGE3_10MS)
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
