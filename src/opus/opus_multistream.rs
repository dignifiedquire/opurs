//! Multistream layout/config validation helpers.
//!
//! Upstream C: `src/opus_multistream.c`, `src/opus_multistream_encoder.c`,
//! `src/opus_multistream_decoder.c`

use crate::opus::opus_defines::{
    OPUS_APPLICATION_AUDIO, OPUS_APPLICATION_RESTRICTED_LOWDELAY, OPUS_APPLICATION_VOIP,
    OPUS_BAD_ARG,
};

/// Typed multistream channel layout.
///
/// This maps output/input channels to Opus streams, matching the semantics
/// of `ChannelLayout` in upstream `opus_multistream` code.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OpusMultistreamLayout {
    nb_channels: u8,
    nb_streams: u8,
    nb_coupled_streams: u8,
    mapping: Vec<u8>,
}

impl OpusMultistreamLayout {
    /// Create and validate a multistream layout.
    ///
    /// This enforces upstream layout constraints and mapping index validity.
    pub fn new(
        channels: i32,
        streams: i32,
        coupled_streams: i32,
        mapping: &[u8],
    ) -> Result<Self, i32> {
        if !is_valid_shape(channels, streams, coupled_streams) || mapping.len() != channels as usize
        {
            return Err(OPUS_BAD_ARG);
        }
        let layout = Self {
            nb_channels: channels as u8,
            nb_streams: streams as u8,
            nb_coupled_streams: coupled_streams as u8,
            mapping: mapping.to_vec(),
        };
        if !layout.validate_layout() {
            return Err(OPUS_BAD_ARG);
        }
        Ok(layout)
    }

    #[inline]
    pub fn channels(&self) -> i32 {
        self.nb_channels as i32
    }

    #[inline]
    pub fn streams(&self) -> i32 {
        self.nb_streams as i32
    }

    #[inline]
    pub fn coupled_streams(&self) -> i32 {
        self.nb_coupled_streams as i32
    }

    #[inline]
    pub fn mapping(&self) -> &[u8] {
        &self.mapping
    }

    /// Upstream parity for `validate_layout()`.
    ///
    /// Upstream C: `src/opus_multistream.c:validate_layout`
    pub fn validate_layout(&self) -> bool {
        let max_channel = self.streams() + self.coupled_streams();
        if max_channel > 255 {
            return false;
        }
        self.mapping
            .iter()
            .all(|&m| (m as i32) < max_channel || m == 255)
    }

    /// Validation used by multistream decoder init/create.
    ///
    /// Upstream C: `src/opus_multistream_decoder.c`
    pub fn validate_for_decoder(&self) -> bool {
        is_valid_shape(self.channels(), self.streams(), self.coupled_streams())
            && self.validate_layout()
    }

    /// Validation used by multistream encoder init/create.
    ///
    /// Includes decoder constraints plus encoder-specific layout checks.
    /// Upstream C: `src/opus_multistream_encoder.c:validate_encoder_layout`
    pub fn validate_for_encoder(&self) -> bool {
        if !self.validate_for_decoder() || self.streams() + self.coupled_streams() > self.channels()
        {
            return false;
        }
        validate_encoder_layout(self)
    }

    /// Find channel index mapped as left channel for coupled stream `stream_id`.
    ///
    /// Upstream C: `src/opus_multistream.c:get_left_channel`
    pub fn left_channel(&self, stream_id: i32, prev: i32) -> Option<usize> {
        let target = stream_id.checked_mul(2)?;
        find_mapping_channel(self, target, prev)
    }

    /// Find channel index mapped as right channel for coupled stream `stream_id`.
    ///
    /// Upstream C: `src/opus_multistream.c:get_right_channel`
    pub fn right_channel(&self, stream_id: i32, prev: i32) -> Option<usize> {
        let target = stream_id.checked_mul(2)?.checked_add(1)?;
        find_mapping_channel(self, target, prev)
    }

    /// Find channel index mapped as mono channel for stream `stream_id`.
    ///
    /// Upstream C: `src/opus_multistream.c:get_mono_channel`
    pub fn mono_channel(&self, stream_id: i32, prev: i32) -> Option<usize> {
        let target = stream_id.checked_add(self.coupled_streams())?;
        find_mapping_channel(self, target, prev)
    }
}

/// Typed config for future multistream encoder/decoder constructors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OpusMultistreamConfig {
    sample_rate: i32,
    application: i32,
    layout: OpusMultistreamLayout,
}

impl OpusMultistreamConfig {
    /// Build a validated config.
    ///
    /// For decoder-only paths, `application` can be set to any valid opus
    /// application constant and ignored by callers.
    pub fn new(
        sample_rate: i32,
        application: i32,
        layout: OpusMultistreamLayout,
    ) -> Result<Self, i32> {
        let valid_fs = sample_rate == 48000
            || sample_rate == 24000
            || sample_rate == 16000
            || sample_rate == 12000
            || sample_rate == 8000
            || cfg!(feature = "qext") && sample_rate == 96000;
        let valid_application = application == OPUS_APPLICATION_VOIP
            || application == OPUS_APPLICATION_AUDIO
            || application == OPUS_APPLICATION_RESTRICTED_LOWDELAY;
        if !valid_fs || !valid_application {
            return Err(OPUS_BAD_ARG);
        }
        Ok(Self {
            sample_rate,
            application,
            layout,
        })
    }

    #[inline]
    pub fn sample_rate(&self) -> i32 {
        self.sample_rate
    }

    #[inline]
    pub fn application(&self) -> i32 {
        self.application
    }

    pub fn set_application(&mut self, application: i32) -> Result<(), i32> {
        let valid_application = application == OPUS_APPLICATION_VOIP
            || application == OPUS_APPLICATION_AUDIO
            || application == OPUS_APPLICATION_RESTRICTED_LOWDELAY;
        if !valid_application {
            return Err(OPUS_BAD_ARG);
        }
        self.application = application;
        Ok(())
    }

    #[inline]
    pub fn layout(&self) -> &OpusMultistreamLayout {
        &self.layout
    }
}

#[inline]
fn is_valid_shape(channels: i32, streams: i32, coupled_streams: i32) -> bool {
    (1..=255).contains(&channels)
        && streams >= 1
        && coupled_streams >= 0
        && coupled_streams <= streams
        && streams <= 255 - coupled_streams
}

fn find_mapping_channel(layout: &OpusMultistreamLayout, target: i32, prev: i32) -> Option<usize> {
    if target < 0 {
        return None;
    }
    let start = if prev < 0 { 0 } else { prev as usize + 1 };
    layout.mapping[start..]
        .iter()
        .position(|&m| m as i32 == target)
        .map(|idx| start + idx)
}

fn validate_encoder_layout(layout: &OpusMultistreamLayout) -> bool {
    for stream in 0..layout.streams() {
        if stream < layout.coupled_streams() {
            if layout.left_channel(stream, -1).is_none()
                || layout.right_channel(stream, -1).is_none()
            {
                return false;
            }
        } else if layout.mono_channel(stream, -1).is_none() {
            return false;
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validate_layout_accepts_stereo() {
        let layout = OpusMultistreamLayout::new(2, 1, 1, &[0, 1]).unwrap();
        assert!(layout.validate_layout());
        assert!(layout.validate_for_encoder());
        assert!(layout.validate_for_decoder());
    }

    #[test]
    fn validate_layout_rejects_mapping_out_of_range() {
        let layout = OpusMultistreamLayout::new(2, 1, 1, &[0, 2]);
        assert_eq!(layout, Err(OPUS_BAD_ARG));
    }

    #[test]
    fn validate_encoder_requires_all_coupled_channels() {
        // Coupled stream 0 must expose both mapping IDs 0 and 1.
        let layout = OpusMultistreamLayout::new(2, 1, 1, &[0, 0]).unwrap();
        assert!(layout.validate_for_decoder());
        assert!(!layout.validate_for_encoder());
    }

    #[test]
    fn channel_lookup_matches_upstream_semantics() {
        let layout = OpusMultistreamLayout::new(6, 4, 2, &[0, 4, 1, 2, 3, 5]).unwrap();
        assert_eq!(layout.left_channel(0, -1), Some(0));
        assert_eq!(layout.right_channel(0, -1), Some(2));
        assert_eq!(layout.left_channel(1, -1), Some(3));
        assert_eq!(layout.right_channel(1, -1), Some(4));
        assert_eq!(layout.mono_channel(2, -1), Some(1));
        assert_eq!(layout.mono_channel(3, -1), Some(5));
    }
}
