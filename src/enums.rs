//! Typed enums replacing raw `OPUS_*` integer constants.
//!
//! These types are designed for Rust users. Where the C API uses `OPUS_AUTO`
//! as a sentinel value, the Rust API uses `Option<T>` instead — the `Auto`
//! concept does not appear in these enums (except `Bitrate`, where it is a
//! distinct configuration mode).
//!
//! See: <https://opus-codec.org/docs/opus_api-1.3.1/group__opus__ctlvalues.html>

use crate::error::ErrorCode;
use crate::opus::opus_defines::*;

/// Intended application profile for the Opus encoder.
///
/// Controls trade-offs between speech intelligibility, audio fidelity,
/// and latency.
///
/// See: <https://opus-codec.org/docs/opus_api-1.3.1/group__opus__ctlvalues.html>
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum Application {
    /// Best for most VoIP/videoconference applications where listening
    /// quality and intelligibility matter most.
    Voip,
    /// Best for broadcast/high-fidelity application where the decoded
    /// audio should be as close as possible to the input.
    Audio,
    /// Only use when lowest-achievable latency is what matters most.
    /// Voice-optimized modes will not be used.
    LowDelay,
}

impl TryFrom<i32> for Application {
    type Error = ErrorCode;

    fn try_from(value: i32) -> Result<Self, Self::Error> {
        match value {
            OPUS_APPLICATION_VOIP => Ok(Application::Voip),
            OPUS_APPLICATION_AUDIO => Ok(Application::Audio),
            OPUS_APPLICATION_RESTRICTED_LOWDELAY => Ok(Application::LowDelay),
            _ => Err(ErrorCode::BadArg),
        }
    }
}

impl From<Application> for i32 {
    fn from(value: Application) -> Self {
        match value {
            Application::Voip => OPUS_APPLICATION_VOIP,
            Application::Audio => OPUS_APPLICATION_AUDIO,
            Application::LowDelay => OPUS_APPLICATION_RESTRICTED_LOWDELAY,
        }
    }
}

/// Channel count configuration.
///
/// See: <https://opus-codec.org/docs/opus_api-1.3.1/group__opus__encoder.html>
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum Channels {
    /// Mono (single channel).
    Mono,
    /// Stereo (two channels, interleaved).
    Stereo,
}

impl TryFrom<i32> for Channels {
    type Error = ErrorCode;

    fn try_from(value: i32) -> Result<Self, Self::Error> {
        match value {
            1 => Ok(Channels::Mono),
            2 => Ok(Channels::Stereo),
            _ => Err(ErrorCode::BadArg),
        }
    }
}

impl From<Channels> for i32 {
    fn from(value: Channels) -> Self {
        match value {
            Channels::Mono => 1,
            Channels::Stereo => 2,
        }
    }
}

/// Audio bandwidth.
///
/// Represents a resolved bandwidth — the actual frequency range used for
/// encoding or decoded from a packet. When configuring the encoder, use
/// `Option<Bandwidth>` where `None` means auto-detection.
///
/// See: <https://opus-codec.org/docs/opus_api-1.3.1/group__opus__ctlvalues.html>
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum Bandwidth {
    /// 4 kHz passband.
    Narrowband,
    /// 6 kHz passband.
    Mediumband,
    /// 8 kHz passband.
    Wideband,
    /// 12 kHz passband.
    Superwideband,
    /// 20 kHz passband (full audio range).
    Fullband,
}

impl TryFrom<i32> for Bandwidth {
    type Error = ErrorCode;

    fn try_from(value: i32) -> Result<Self, Self::Error> {
        match value {
            OPUS_BANDWIDTH_NARROWBAND => Ok(Bandwidth::Narrowband),
            OPUS_BANDWIDTH_MEDIUMBAND => Ok(Bandwidth::Mediumband),
            OPUS_BANDWIDTH_WIDEBAND => Ok(Bandwidth::Wideband),
            OPUS_BANDWIDTH_SUPERWIDEBAND => Ok(Bandwidth::Superwideband),
            OPUS_BANDWIDTH_FULLBAND => Ok(Bandwidth::Fullband),
            _ => Err(ErrorCode::BadArg),
        }
    }
}

impl From<Bandwidth> for i32 {
    fn from(value: Bandwidth) -> Self {
        match value {
            Bandwidth::Narrowband => OPUS_BANDWIDTH_NARROWBAND,
            Bandwidth::Mediumband => OPUS_BANDWIDTH_MEDIUMBAND,
            Bandwidth::Wideband => OPUS_BANDWIDTH_WIDEBAND,
            Bandwidth::Superwideband => OPUS_BANDWIDTH_SUPERWIDEBAND,
            Bandwidth::Fullband => OPUS_BANDWIDTH_FULLBAND,
        }
    }
}

/// Signal type hint for the encoder.
///
/// Providing a hint allows the encoder to make better mode decisions.
/// When configuring the encoder, use `Option<Signal>` where `None` means
/// auto-detection.
///
/// See: <https://opus-codec.org/docs/opus_api-1.3.1/group__opus__ctlvalues.html>
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum Signal {
    /// Bias toward LPC or Hybrid modes, optimized for speech.
    Voice,
    /// Bias toward MDCT modes, optimized for music.
    Music,
}

impl TryFrom<i32> for Signal {
    type Error = ErrorCode;

    fn try_from(value: i32) -> Result<Self, Self::Error> {
        match value {
            OPUS_SIGNAL_VOICE => Ok(Signal::Voice),
            OPUS_SIGNAL_MUSIC => Ok(Signal::Music),
            _ => Err(ErrorCode::BadArg),
        }
    }
}

impl From<Signal> for i32 {
    fn from(value: Signal) -> Self {
        match value {
            Signal::Voice => OPUS_SIGNAL_VOICE,
            Signal::Music => OPUS_SIGNAL_MUSIC,
        }
    }
}

/// Expert frame duration control for the encoder.
///
/// These values control the frame size used by the encoder. `Arg` (the
/// default) uses the frame size from the `encode` call's input length.
///
/// See: <https://opus-codec.org/docs/opus_api-1.3.1/group__opus__ctlvalues.html>
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum FrameSize {
    /// Select frame size from the argument (default).
    Arg,
    /// Use 2.5 ms frames.
    Ms2_5,
    /// Use 5 ms frames.
    Ms5,
    /// Use 10 ms frames.
    Ms10,
    /// Use 20 ms frames.
    Ms20,
    /// Use 40 ms frames.
    Ms40,
    /// Use 60 ms frames.
    Ms60,
    /// Use 80 ms frames.
    Ms80,
    /// Use 100 ms frames.
    Ms100,
    /// Use 120 ms frames.
    Ms120,
}

impl TryFrom<i32> for FrameSize {
    type Error = ErrorCode;

    fn try_from(value: i32) -> Result<Self, Self::Error> {
        match value {
            OPUS_FRAMESIZE_ARG => Ok(FrameSize::Arg),
            OPUS_FRAMESIZE_2_5_MS => Ok(FrameSize::Ms2_5),
            OPUS_FRAMESIZE_5_MS => Ok(FrameSize::Ms5),
            OPUS_FRAMESIZE_10_MS => Ok(FrameSize::Ms10),
            OPUS_FRAMESIZE_20_MS => Ok(FrameSize::Ms20),
            OPUS_FRAMESIZE_40_MS => Ok(FrameSize::Ms40),
            OPUS_FRAMESIZE_60_MS => Ok(FrameSize::Ms60),
            OPUS_FRAMESIZE_80_MS => Ok(FrameSize::Ms80),
            OPUS_FRAMESIZE_100_MS => Ok(FrameSize::Ms100),
            OPUS_FRAMESIZE_120_MS => Ok(FrameSize::Ms120),
            _ => Err(ErrorCode::BadArg),
        }
    }
}

impl From<FrameSize> for i32 {
    fn from(value: FrameSize) -> Self {
        match value {
            FrameSize::Arg => OPUS_FRAMESIZE_ARG,
            FrameSize::Ms2_5 => OPUS_FRAMESIZE_2_5_MS,
            FrameSize::Ms5 => OPUS_FRAMESIZE_5_MS,
            FrameSize::Ms10 => OPUS_FRAMESIZE_10_MS,
            FrameSize::Ms20 => OPUS_FRAMESIZE_20_MS,
            FrameSize::Ms40 => OPUS_FRAMESIZE_40_MS,
            FrameSize::Ms60 => OPUS_FRAMESIZE_60_MS,
            FrameSize::Ms80 => OPUS_FRAMESIZE_80_MS,
            FrameSize::Ms100 => OPUS_FRAMESIZE_100_MS,
            FrameSize::Ms120 => OPUS_FRAMESIZE_120_MS,
        }
    }
}

/// Bitrate configuration for the Opus encoder.
///
/// `Auto` lets the encoder select a bitrate based on the configuration.
/// `Max` uses the maximum bitrate allowed for the packet size.
/// `Bits(n)` specifies an explicit target bitrate in bits per second.
///
/// See: <https://opus-codec.org/docs/opus_api-1.3.1/group__opus__encoderctls.html>
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum Bitrate {
    /// Let the encoder choose the bitrate automatically.
    Auto,
    /// Maximum bitrate allowed by the packet size.
    Max,
    /// Explicit bitrate in bits per second.
    Bits(i32),
}

impl From<i32> for Bitrate {
    fn from(value: i32) -> Self {
        match value {
            OPUS_AUTO => Bitrate::Auto,
            OPUS_BITRATE_MAX => Bitrate::Max,
            other => Bitrate::Bits(other),
        }
    }
}

impl From<Bitrate> for i32 {
    fn from(value: Bitrate) -> Self {
        match value {
            Bitrate::Auto => OPUS_AUTO,
            Bitrate::Max => OPUS_BITRATE_MAX,
            Bitrate::Bits(bits) => bits,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn application_try_from_valid() {
        assert_eq!(Application::try_from(2048).unwrap(), Application::Voip);
        assert_eq!(Application::try_from(2049).unwrap(), Application::Audio);
        assert_eq!(Application::try_from(2051).unwrap(), Application::LowDelay);
    }

    #[test]
    fn application_try_from_invalid() {
        assert!(Application::try_from(0).is_err());
        assert!(Application::try_from(2050).is_err());
    }

    #[test]
    fn application_into_i32() {
        assert_eq!(i32::from(Application::Voip), 2048);
        assert_eq!(i32::from(Application::Audio), 2049);
        assert_eq!(i32::from(Application::LowDelay), 2051);
    }

    #[test]
    fn channels_roundtrip() {
        assert_eq!(i32::from(Channels::Mono), 1);
        assert_eq!(i32::from(Channels::Stereo), 2);
        assert_eq!(Channels::try_from(1).unwrap(), Channels::Mono);
        assert_eq!(Channels::try_from(2).unwrap(), Channels::Stereo);
        assert!(Channels::try_from(0).is_err());
        assert!(Channels::try_from(3).is_err());
    }

    #[test]
    fn bandwidth_all_variants() {
        let pairs = [
            (Bandwidth::Narrowband, 1101),
            (Bandwidth::Mediumband, 1102),
            (Bandwidth::Wideband, 1103),
            (Bandwidth::Superwideband, 1104),
            (Bandwidth::Fullband, 1105),
        ];
        for (variant, raw) in pairs {
            assert_eq!(Bandwidth::try_from(raw).unwrap(), variant);
            assert_eq!(i32::from(variant), raw);
        }
    }

    #[test]
    fn bandwidth_rejects_auto() {
        assert!(Bandwidth::try_from(OPUS_AUTO).is_err());
    }

    #[test]
    fn signal_roundtrip() {
        assert_eq!(Signal::try_from(3001).unwrap(), Signal::Voice);
        assert_eq!(Signal::try_from(3002).unwrap(), Signal::Music);
        assert_eq!(i32::from(Signal::Voice), 3001);
        assert_eq!(i32::from(Signal::Music), 3002);
    }

    #[test]
    fn signal_rejects_auto() {
        assert!(Signal::try_from(OPUS_AUTO).is_err());
    }

    #[test]
    fn frame_size_all_variants() {
        let pairs = [
            (FrameSize::Arg, 5000),
            (FrameSize::Ms2_5, 5001),
            (FrameSize::Ms5, 5002),
            (FrameSize::Ms10, 5003),
            (FrameSize::Ms20, 5004),
            (FrameSize::Ms40, 5005),
            (FrameSize::Ms60, 5006),
            (FrameSize::Ms80, 5007),
            (FrameSize::Ms100, 5008),
            (FrameSize::Ms120, 5009),
        ];
        for (variant, raw) in pairs {
            assert_eq!(FrameSize::try_from(raw).unwrap(), variant);
            assert_eq!(i32::from(variant), raw);
        }
    }

    #[test]
    fn bitrate_from_i32() {
        assert_eq!(Bitrate::from(OPUS_AUTO), Bitrate::Auto);
        assert_eq!(Bitrate::from(OPUS_BITRATE_MAX), Bitrate::Max);
        assert_eq!(Bitrate::from(64000), Bitrate::Bits(64000));
        assert_eq!(Bitrate::from(0), Bitrate::Bits(0));
    }

    #[test]
    fn bitrate_into_i32() {
        assert_eq!(i32::from(Bitrate::Auto), OPUS_AUTO);
        assert_eq!(i32::from(Bitrate::Max), OPUS_BITRATE_MAX);
        assert_eq!(i32::from(Bitrate::Bits(128000)), 128000);
    }

    #[test]
    fn enums_are_copy() {
        fn assert_copy<T: Copy>() {}
        assert_copy::<Application>();
        assert_copy::<Channels>();
        assert_copy::<Bandwidth>();
        assert_copy::<Signal>();
        assert_copy::<FrameSize>();
        assert_copy::<Bitrate>();
    }
}
