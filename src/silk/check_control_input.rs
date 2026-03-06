//! Input validation for encoder control.
//!
//! Upstream c: `silk/check_control_input.c`

use crate::silk::define::ENCODER_NUM_CHANNELS;
use crate::silk::enc_api::silk_EncControlStruct;
pub use crate::silk::errors::{
    SILK_ENC_FS_NOT_SUPPORTED, SILK_ENC_INVALID_CBR_SETTING, SILK_ENC_INVALID_COMPLEXITY_SETTING,
    SILK_ENC_INVALID_DTX_SETTING, SILK_ENC_INVALID_INBAND_FEC_SETTING, SILK_ENC_INVALID_LOSS_RATE,
    SILK_ENC_INVALID_NUMBER_OF_CHANNELS_ERROR, SILK_ENC_PACKET_SIZE_NOT_SUPPORTED, SILK_NO_ERROR,
};

/// Upstream c: silk/check_control_input.c:check_control_input
#[inline]
fn api_sample_rate_supported(api_sample_rate: i32) -> bool {
    matches!(
        api_sample_rate,
        8000 | 12000 | 16000 | 24000 | 32000 | 44100 | 48000
    ) || {
        #[cfg(feature = "qext")]
        {
            api_sample_rate == 96000
        }
        #[cfg(not(feature = "qext"))]
        {
            false
        }
    }
}

/// Upstream c: silk/check_control_input.c:check_control_input
pub fn check_control_input(enc_control: &silk_EncControlStruct) -> i32 {
    if !api_sample_rate_supported(enc_control.api_sample_rate)
        || enc_control.desired_internal_sample_rate != 8000
            && enc_control.desired_internal_sample_rate != 12000
            && enc_control.desired_internal_sample_rate != 16000
        || enc_control.max_internal_sample_rate != 8000
            && enc_control.max_internal_sample_rate != 12000
            && enc_control.max_internal_sample_rate != 16000
        || enc_control.min_internal_sample_rate != 8000
            && enc_control.min_internal_sample_rate != 12000
            && enc_control.min_internal_sample_rate != 16000
        || enc_control.min_internal_sample_rate > enc_control.desired_internal_sample_rate
        || enc_control.max_internal_sample_rate < enc_control.desired_internal_sample_rate
        || enc_control.min_internal_sample_rate > enc_control.max_internal_sample_rate
    {
        return SILK_ENC_FS_NOT_SUPPORTED;
    }
    if enc_control.payload_size_ms != 10
        && enc_control.payload_size_ms != 20
        && enc_control.payload_size_ms != 40
        && enc_control.payload_size_ms != 60
    {
        return SILK_ENC_PACKET_SIZE_NOT_SUPPORTED;
    }
    if enc_control.packet_loss_percentage < 0 || enc_control.packet_loss_percentage > 100 {
        return SILK_ENC_INVALID_LOSS_RATE;
    }
    if enc_control.use_dtx < 0 || enc_control.use_dtx > 1 {
        return SILK_ENC_INVALID_DTX_SETTING;
    }
    if enc_control.use_cbr < 0 || enc_control.use_cbr > 1 {
        return SILK_ENC_INVALID_CBR_SETTING;
    }
    if enc_control.use_in_band_fec < 0 || enc_control.use_in_band_fec > 1 {
        return SILK_ENC_INVALID_INBAND_FEC_SETTING;
    }
    if enc_control.n_channels_api < 1 || enc_control.n_channels_api > ENCODER_NUM_CHANNELS {
        return SILK_ENC_INVALID_NUMBER_OF_CHANNELS_ERROR;
    }
    if enc_control.n_channels_internal < 1 || enc_control.n_channels_internal > ENCODER_NUM_CHANNELS
    {
        return SILK_ENC_INVALID_NUMBER_OF_CHANNELS_ERROR;
    }
    if enc_control.n_channels_internal > enc_control.n_channels_api {
        return SILK_ENC_INVALID_NUMBER_OF_CHANNELS_ERROR;
    }
    if enc_control.complexity < 0 || enc_control.complexity > 10 {
        return SILK_ENC_INVALID_COMPLEXITY_SETTING;
    }
    SILK_NO_ERROR
}

#[cfg(test)]
mod tests {
    use super::*;

    fn baseline_control() -> silk_EncControlStruct {
        silk_EncControlStruct {
            n_channels_api: 1,
            n_channels_internal: 1,
            api_sample_rate: 48_000,
            max_internal_sample_rate: 16_000,
            min_internal_sample_rate: 8_000,
            desired_internal_sample_rate: 16_000,
            payload_size_ms: 20,
            bit_rate: 24_000,
            packet_loss_percentage: 0,
            complexity: 10,
            use_in_band_fec: 0,
            use_dred: 0,
            lbrr_coded: 0,
            use_dtx: 0,
            use_cbr: 0,
            max_bits: 0,
            to_mono: 0,
            opus_can_switch: 0,
            reduced_dependency: 0,
            internal_sample_rate: 0,
            allow_bandwidth_switch: 0,
            in_wbmode_without_variable_lp: 0,
            stereo_width_q14: 0,
            switch_ready: 0,
            signal_type: 0,
            offset: 0,
        }
    }

    #[test]
    fn valid_control_returns_no_error() {
        let ctrl = baseline_control();
        assert_eq!(check_control_input(&ctrl), SILK_NO_ERROR);
    }

    #[test]
    fn invalid_payload_size_returns_expected_error() {
        let mut ctrl = baseline_control();
        ctrl.payload_size_ms = 15;
        assert_eq!(
            check_control_input(&ctrl),
            SILK_ENC_PACKET_SIZE_NOT_SUPPORTED
        );
    }

    #[test]
    fn invalid_loss_rate_returns_expected_error() {
        let mut ctrl = baseline_control();
        ctrl.packet_loss_percentage = 101;
        assert_eq!(check_control_input(&ctrl), SILK_ENC_INVALID_LOSS_RATE);
    }

    #[test]
    fn invalid_complexity_returns_expected_error() {
        let mut ctrl = baseline_control();
        ctrl.complexity = 11;
        assert_eq!(
            check_control_input(&ctrl),
            SILK_ENC_INVALID_COMPLEXITY_SETTING
        );
    }

    #[test]
    fn invalid_channel_relationship_returns_expected_error() {
        let mut ctrl = baseline_control();
        ctrl.n_channels_api = 1;
        ctrl.n_channels_internal = 2;
        assert_eq!(
            check_control_input(&ctrl),
            SILK_ENC_INVALID_NUMBER_OF_CHANNELS_ERROR
        );
    }

    #[test]
    #[cfg(not(feature = "qext"))]
    fn non_qext_rejects_96k_api_rate() {
        let mut ctrl = baseline_control();
        ctrl.api_sample_rate = 96_000;
        assert_eq!(check_control_input(&ctrl), SILK_ENC_FS_NOT_SUPPORTED);
    }

    #[test]
    #[cfg(feature = "qext")]
    fn qext_accepts_96k_api_rate() {
        let mut ctrl = baseline_control();
        ctrl.api_sample_rate = 96_000;
        assert_eq!(check_control_input(&ctrl), SILK_NO_ERROR);
    }
}
