//! Pitch lag decoding.
//!
//! Upstream C: `silk/decode_pitch.c`

use crate::silk::pitch_est_tables::{
    PE_MAX_LAG_MS, PE_MAX_NB_SUBFR, PE_MAX_NB_SUBFR_OVER_2, PE_MIN_LAG_MS, PE_NB_CBKS_STAGE2_10MS,
    PE_NB_CBKS_STAGE2_EXT, PE_NB_CBKS_STAGE3_10MS, PE_NB_CBKS_STAGE3_MAX, SILK_CB_LAGS_STAGE2,
    SILK_CB_LAGS_STAGE2_10_MS, SILK_CB_LAGS_STAGE3, SILK_CB_LAGS_STAGE3_10_MS,
};
use crate::silk::SigProc_FIX::silk_limit;

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
            (&SILK_CB_LAGS_STAGE2, PE_NB_CBKS_STAGE2_EXT)
        } else {
            debug_assert_eq!(nb_subfr, PE_MAX_NB_SUBFR_OVER_2);
            (&SILK_CB_LAGS_STAGE2_10_MS, PE_NB_CBKS_STAGE2_10MS)
        }
    } else if nb_subfr == PE_MAX_NB_SUBFR {
        (&SILK_CB_LAGS_STAGE3, PE_NB_CBKS_STAGE3_MAX)
    } else {
        debug_assert_eq!(nb_subfr, PE_MAX_NB_SUBFR_OVER_2);
        (&SILK_CB_LAGS_STAGE3_10_MS, PE_NB_CBKS_STAGE3_10MS)
    };

    let min_lag = PE_MIN_LAG_MS * Fs_kHz as i16 as i32;
    let max_lag = PE_MAX_LAG_MS * Fs_kHz as i16 as i32;
    let lag = min_lag + lagIndex as i32;

    for (k, out_lag) in pitch_lags.iter_mut().enumerate() {
        let lag_cb_row = &lag_cb_flat[k * ncols..][..ncols];
        let lag = lag + lag_cb_row[contourIndex as usize] as i32;
        *out_lag = silk_limit(lag, min_lag, max_lag);
    }
}
