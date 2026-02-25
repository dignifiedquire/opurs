//! Input validation for encoder control.
//!
//! Upstream C: `silk/check_control_input.c`

#[allow(unused)]
pub mod errors_h {
    pub const SILK_NO_ERROR: i32 = 0;
    pub const SILK_ENC_INVALID_COMPLEXITY_SETTING: i32 = -(106);
    pub const SILK_ENC_INVALID_NUMBER_OF_CHANNELS_ERROR: i32 = -(111);
    pub const SILK_ENC_INVALID_INBAND_FEC_SETTING: i32 = -(107);
    pub const SILK_ENC_INVALID_CBR_SETTING: i32 = -(109);
    pub const SILK_ENC_INVALID_DTX_SETTING: i32 = -(108);
    pub const SILK_ENC_INVALID_LOSS_RATE: i32 = -(105);
    pub const SILK_ENC_PACKET_SIZE_NOT_SUPPORTED: i32 = -(103);
    pub const SILK_ENC_FS_NOT_SUPPORTED: i32 = -(102);
}

#[allow(unused)]
pub use self::errors_h::{
    SILK_ENC_FS_NOT_SUPPORTED, SILK_ENC_INVALID_CBR_SETTING, SILK_ENC_INVALID_COMPLEXITY_SETTING,
    SILK_ENC_INVALID_DTX_SETTING, SILK_ENC_INVALID_INBAND_FEC_SETTING, SILK_ENC_INVALID_LOSS_RATE,
    SILK_ENC_INVALID_NUMBER_OF_CHANNELS_ERROR, SILK_ENC_PACKET_SIZE_NOT_SUPPORTED, SILK_NO_ERROR,
};
use crate::silk::define::ENCODER_NUM_CHANNELS;
use crate::silk::enc_API::silk_EncControlStruct;

/// Upstream C: silk/check_control_input.c:check_control_input
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

/// Upstream C: silk/check_control_input.c:check_control_input
pub fn check_control_input(encControl: &silk_EncControlStruct) -> i32 {
    if !api_sample_rate_supported(encControl.API_sampleRate)
        || encControl.desiredInternalSampleRate != 8000
            && encControl.desiredInternalSampleRate != 12000
            && encControl.desiredInternalSampleRate != 16000
        || encControl.maxInternalSampleRate != 8000
            && encControl.maxInternalSampleRate != 12000
            && encControl.maxInternalSampleRate != 16000
        || encControl.minInternalSampleRate != 8000
            && encControl.minInternalSampleRate != 12000
            && encControl.minInternalSampleRate != 16000
        || encControl.minInternalSampleRate > encControl.desiredInternalSampleRate
        || encControl.maxInternalSampleRate < encControl.desiredInternalSampleRate
        || encControl.minInternalSampleRate > encControl.maxInternalSampleRate
    {
        return SILK_ENC_FS_NOT_SUPPORTED;
    }
    if encControl.payloadSize_ms != 10
        && encControl.payloadSize_ms != 20
        && encControl.payloadSize_ms != 40
        && encControl.payloadSize_ms != 60
    {
        return SILK_ENC_PACKET_SIZE_NOT_SUPPORTED;
    }
    if encControl.packetLossPercentage < 0 || encControl.packetLossPercentage > 100 {
        return SILK_ENC_INVALID_LOSS_RATE;
    }
    if encControl.useDTX < 0 || encControl.useDTX > 1 {
        return SILK_ENC_INVALID_DTX_SETTING;
    }
    if encControl.useCBR < 0 || encControl.useCBR > 1 {
        return SILK_ENC_INVALID_CBR_SETTING;
    }
    if encControl.useInBandFEC < 0 || encControl.useInBandFEC > 1 {
        return SILK_ENC_INVALID_INBAND_FEC_SETTING;
    }
    if encControl.nChannelsAPI < 1 || encControl.nChannelsAPI > ENCODER_NUM_CHANNELS {
        return SILK_ENC_INVALID_NUMBER_OF_CHANNELS_ERROR;
    }
    if encControl.nChannelsInternal < 1 || encControl.nChannelsInternal > ENCODER_NUM_CHANNELS {
        return SILK_ENC_INVALID_NUMBER_OF_CHANNELS_ERROR;
    }
    if encControl.nChannelsInternal > encControl.nChannelsAPI {
        return SILK_ENC_INVALID_NUMBER_OF_CHANNELS_ERROR;
    }
    if encControl.complexity < 0 || encControl.complexity > 10 {
        return SILK_ENC_INVALID_COMPLEXITY_SETTING;
    }
    SILK_NO_ERROR
}

#[cfg(test)]
mod tests {
    use super::*;

    fn baseline_control() -> silk_EncControlStruct {
        silk_EncControlStruct {
            nChannelsAPI: 1,
            nChannelsInternal: 1,
            API_sampleRate: 48_000,
            maxInternalSampleRate: 16_000,
            minInternalSampleRate: 8_000,
            desiredInternalSampleRate: 16_000,
            payloadSize_ms: 20,
            bitRate: 24_000,
            packetLossPercentage: 0,
            complexity: 10,
            useInBandFEC: 0,
            useDRED: 0,
            LBRR_coded: 0,
            useDTX: 0,
            useCBR: 0,
            maxBits: 0,
            toMono: 0,
            opusCanSwitch: 0,
            reducedDependency: 0,
            internalSampleRate: 0,
            allowBandwidthSwitch: 0,
            inWBmodeWithoutVariableLP: 0,
            stereoWidth_Q14: 0,
            switchReady: 0,
            signalType: 0,
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
        ctrl.payloadSize_ms = 15;
        assert_eq!(
            check_control_input(&ctrl),
            SILK_ENC_PACKET_SIZE_NOT_SUPPORTED
        );
    }

    #[test]
    fn invalid_loss_rate_returns_expected_error() {
        let mut ctrl = baseline_control();
        ctrl.packetLossPercentage = 101;
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
        ctrl.nChannelsAPI = 1;
        ctrl.nChannelsInternal = 2;
        assert_eq!(
            check_control_input(&ctrl),
            SILK_ENC_INVALID_NUMBER_OF_CHANNELS_ERROR
        );
    }

    #[test]
    #[cfg(not(feature = "qext"))]
    fn non_qext_rejects_96k_api_rate() {
        let mut ctrl = baseline_control();
        ctrl.API_sampleRate = 96_000;
        assert_eq!(check_control_input(&ctrl), SILK_ENC_FS_NOT_SUPPORTED);
    }

    #[test]
    #[cfg(feature = "qext")]
    fn qext_accepts_96k_api_rate() {
        let mut ctrl = baseline_control();
        ctrl.API_sampleRate = 96_000;
        assert_eq!(check_control_input(&ctrl), SILK_NO_ERROR);
    }
}
