//! Projection Opus decoder wrapper.
//!
//! Upstream C: `src/opus_projection_decoder.c`

use crate::enums::SampleRate;
use crate::error::ErrorCode;
use crate::opus::mapping_matrix::MappingMatrix;
use crate::opus::opus_decoder::OpusDecoder;
use crate::opus::opus_defines::{OPUS_BAD_ARG, OPUS_OK};
use crate::opus::opus_multistream_decoder::OpusMSDecoder;

/// Projection Opus decoder state.
///
/// Holds demixing matrix data and an internal multistream decoder used to
/// decode coded streams prior to projection.
///
/// Upstream C: include/opus_projection.h:OpusProjectionDecoder
#[derive(Clone)]
pub struct OpusProjectionDecoder {
    channels: i32,
    streams: i32,
    coupled_streams: i32,
    demixing_matrix: MappingMatrix,
    decoder: OpusMSDecoder,
}

impl OpusProjectionDecoder {
    #[inline]
    fn input_channels(&self) -> i32 {
        self.streams + self.coupled_streams
    }

    /// Upstream-style sizing helper.
    ///
    /// Returns zero for invalid stream shapes, non-zero for valid shapes.
    ///
    /// Upstream C: include/opus_projection.h:opus_projection_decoder_get_size
    pub fn get_size(channels: i32, streams: i32, coupled_streams: i32) -> i32 {
        let input_channels = streams + coupled_streams;
        if MappingMatrix::get_size(input_channels, channels) == 0
            || OpusMSDecoder::get_size(streams, coupled_streams) == 0
        {
            0
        } else {
            core::mem::size_of::<Self>() as i32
        }
    }

    /// Create and initialize a projection decoder.
    ///
    /// Upstream C: include/opus_projection.h:opus_projection_decoder_create
    pub fn new(
        sample_rate: SampleRate,
        channels: i32,
        streams: i32,
        coupled_streams: i32,
        demixing_matrix: &[u8],
    ) -> Result<Self, ErrorCode> {
        let input_channels = streams + coupled_streams;
        if channels <= 0 || input_channels <= 0 {
            return Err(ErrorCode::BadArg);
        }
        let expected_matrix_size = (input_channels as i64)
            .checked_mul(channels as i64)
            .and_then(|v| v.checked_mul(2))
            .ok_or(ErrorCode::BadArg)? as usize;
        if demixing_matrix.len() != expected_matrix_size {
            return Err(ErrorCode::BadArg);
        }

        let demixing_matrix =
            MappingMatrix::from_bytes_le(channels, input_channels, 0, demixing_matrix)
                .map_err(ErrorCode::from)?;

        // Decode to "input streams" channels first, then project to output channels.
        let mapping = (0..input_channels).map(|idx| idx as u8).collect::<Vec<_>>();
        let decoder = OpusMSDecoder::new(
            sample_rate,
            input_channels,
            streams,
            coupled_streams,
            &mapping,
        )?;

        Ok(Self {
            channels,
            streams,
            coupled_streams,
            demixing_matrix,
            decoder,
        })
    }

    /// Reinitialize an existing projection decoder instance.
    ///
    /// Upstream C: include/opus_projection.h:opus_projection_decoder_init
    pub fn init(
        &mut self,
        sample_rate: SampleRate,
        channels: i32,
        streams: i32,
        coupled_streams: i32,
        demixing_matrix: &[u8],
    ) -> i32 {
        match Self::new(
            sample_rate,
            channels,
            streams,
            coupled_streams,
            demixing_matrix,
        ) {
            Ok(st) => {
                *self = st;
                OPUS_OK
            }
            Err(err) => err.into(),
        }
    }

    /// Decode a projection packet to interleaved `i16` PCM.
    ///
    /// Upstream C: include/opus_projection.h:opus_projection_decode
    pub fn decode(
        &mut self,
        data: &[u8],
        pcm: &mut [i16],
        frame_size: i32,
        decode_fec: bool,
    ) -> i32 {
        if frame_size <= 0 {
            return OPUS_BAD_ARG;
        }
        let frame_size = frame_size as usize;
        let output_channels = self.channels as usize;
        let input_channels = self.input_channels() as usize;
        if pcm.len() < frame_size * output_channels {
            return OPUS_BAD_ARG;
        }

        // Match upstream short-decode behavior: decode native streams with soft clipping enabled.
        let (stream_pcm, decoded) =
            match self
                .decoder
                .decode_streams_native(data, frame_size as i32, decode_fec, 1)
            {
                Ok(v) => v,
                Err(err) => return err,
            };

        let mut input_pcm = vec![0f32; decoded * input_channels];
        let coupled = self.coupled_streams as usize;
        for input_row in 0..input_channels {
            let (stream_idx, stream_ch, stream_channels) = if input_row < coupled * 2 {
                (input_row / 2, input_row & 1, 2usize)
            } else {
                (coupled + input_row - coupled * 2, 0usize, 1usize)
            };
            let src = &stream_pcm[stream_idx];
            for frame in 0..decoded {
                input_pcm[frame * input_channels + input_row] =
                    src[frame * stream_channels + stream_ch];
            }
        }

        let output = &mut pcm[..decoded * output_channels];
        output.fill(0);
        for input_row in 0..input_channels {
            if let Err(err) = self.demixing_matrix.multiply_channel_out_short(
                &input_pcm[input_row..],
                input_row,
                input_channels,
                output,
                output_channels,
                decoded,
            ) {
                return err;
            }
        }

        decoded as i32
    }

    /// Decode a projection packet to interleaved `f32` PCM.
    ///
    /// Upstream C: include/opus_projection.h:opus_projection_decode_float
    pub fn decode_float(
        &mut self,
        data: &[u8],
        pcm: &mut [f32],
        frame_size: i32,
        decode_fec: bool,
    ) -> i32 {
        if frame_size <= 0 {
            return OPUS_BAD_ARG;
        }
        let frame_size = frame_size as usize;
        let output_channels = self.channels as usize;
        let input_channels = self.input_channels() as usize;
        if pcm.len() < frame_size * output_channels {
            return OPUS_BAD_ARG;
        }

        let mut input_pcm = vec![0f32; frame_size * input_channels];
        let decoded =
            self.decoder
                .decode_float(data, &mut input_pcm, frame_size as i32, decode_fec);
        if decoded <= 0 {
            return decoded;
        }
        let decoded = decoded as usize;

        let output = &mut pcm[..decoded * output_channels];
        output.fill(0.0);
        for input_row in 0..input_channels {
            if let Err(err) = self.demixing_matrix.multiply_channel_out_float(
                &input_pcm[input_row..],
                input_row,
                input_channels,
                output,
                output_channels,
                decoded,
            ) {
                return err;
            }
        }

        decoded as i32
    }

    /// Decode a projection packet to interleaved 24-bit (`i32`) PCM.
    ///
    /// Upstream C: include/opus_projection.h:opus_projection_decode24
    pub fn decode24(
        &mut self,
        data: &[u8],
        pcm: &mut [i32],
        frame_size: i32,
        decode_fec: bool,
    ) -> i32 {
        if frame_size <= 0 {
            return OPUS_BAD_ARG;
        }
        let frame_size = frame_size as usize;
        let output_channels = self.channels as usize;
        let input_channels = self.input_channels() as usize;
        if pcm.len() < frame_size * output_channels {
            return OPUS_BAD_ARG;
        }

        let mut input_pcm = vec![0f32; frame_size * input_channels];
        let decoded =
            self.decoder
                .decode_float(data, &mut input_pcm, frame_size as i32, decode_fec);
        if decoded <= 0 {
            return decoded;
        }
        let decoded = decoded as usize;

        let output = &mut pcm[..decoded * output_channels];
        output.fill(0);
        for input_row in 0..input_channels {
            if let Err(err) = self.demixing_matrix.multiply_channel_out_int24(
                &input_pcm[input_row..],
                input_row,
                input_channels,
                output,
                output_channels,
                decoded,
            ) {
                return err;
            }
        }

        decoded as i32
    }

    /// Set decode gain.
    ///
    /// Upstream C: include/opus_projection.h:opus_projection_decoder_ctl
    pub fn set_gain(&mut self, gain: i32) -> Result<(), i32> {
        self.decoder.set_gain(gain)
    }

    /// Set decoder complexity.
    ///
    /// Upstream C: include/opus_projection.h:opus_projection_decoder_ctl
    pub fn set_complexity(&mut self, complexity: i32) -> Result<(), i32> {
        self.decoder.set_complexity(complexity)
    }

    /// Enable or disable phase inversion in intensity stereo.
    ///
    /// Upstream C: include/opus_projection.h:opus_projection_decoder_ctl
    pub fn set_phase_inversion_disabled(&mut self, disabled: bool) {
        self.decoder.set_phase_inversion_disabled(disabled);
    }

    /// Enable or disable packet extension parsing.
    ///
    /// Upstream C: include/opus_projection.h:opus_projection_decoder_ctl
    pub fn set_ignore_extensions(&mut self, ignore: bool) {
        self.decoder.set_ignore_extensions(ignore);
    }

    /// Enable or disable OSCE bandwidth extension.
    ///
    /// Upstream C: include/opus_projection.h:opus_projection_decoder_ctl
    #[cfg(feature = "osce")]
    pub fn set_osce_bwe(&mut self, enabled: bool) {
        self.decoder.set_osce_bwe(enabled);
    }

    /// Reset decoder state.
    ///
    /// Upstream C: include/opus_projection.h:opus_projection_decoder_ctl
    pub fn reset(&mut self) {
        self.decoder.reset();
    }

    /// Return final range value from the most recent decoded packet.
    ///
    /// Upstream C: include/opus_projection.h:opus_projection_decoder_ctl
    pub fn final_range(&self) -> u32 {
        self.decoder.final_range()
    }

    /// Return current decode gain.
    ///
    /// Upstream C: include/opus_projection.h:opus_projection_decoder_ctl
    pub fn gain(&self) -> i32 {
        self.decoder.gain()
    }

    /// Return decoder output sample rate in Hz.
    ///
    /// Upstream C: include/opus_projection.h:opus_projection_decoder_ctl
    pub fn sample_rate(&self) -> i32 {
        self.decoder.sample_rate()
    }

    /// Return current complexity setting.
    ///
    /// Upstream C: include/opus_projection.h:opus_projection_decoder_ctl
    pub fn complexity(&self) -> i32 {
        self.decoder.complexity()
    }

    /// Return bandwidth of the last decoded packet.
    ///
    /// Upstream C: include/opus_projection.h:opus_projection_decoder_ctl
    pub fn bandwidth(&self) -> i32 {
        self.decoder.bandwidth()
    }

    /// Return whether phase inversion is disabled.
    ///
    /// Upstream C: include/opus_projection.h:opus_projection_decoder_ctl
    pub fn phase_inversion_disabled(&self) -> bool {
        self.decoder.phase_inversion_disabled()
    }

    /// Return duration of last decoded packet in samples per channel.
    ///
    /// Upstream C: include/opus_projection.h:opus_projection_decoder_ctl
    pub fn last_packet_duration(&self) -> i32 {
        self.decoder.last_packet_duration()
    }

    /// Return whether packet extensions are ignored.
    ///
    /// Upstream C: include/opus_projection.h:opus_projection_decoder_ctl
    pub fn ignore_extensions(&self) -> bool {
        self.decoder.ignore_extensions()
    }

    /// Return whether OSCE BWE is enabled.
    ///
    /// Upstream C: include/opus_projection.h:opus_projection_decoder_ctl
    #[cfg(feature = "osce")]
    pub fn osce_bwe(&self) -> bool {
        self.decoder.osce_bwe()
    }

    /// Borrow a child stream decoder by stream index.
    ///
    /// Upstream C: include/opus_projection.h:opus_projection_decoder_ctl
    pub fn decoder_state(&self, stream_id: i32) -> Result<&OpusDecoder, i32> {
        self.decoder.decoder_state(stream_id)
    }

    /// Mutably borrow a child stream decoder by stream index.
    ///
    /// Upstream C: include/opus_projection.h:opus_projection_decoder_ctl
    pub fn decoder_state_mut(&mut self, stream_id: i32) -> Result<&mut OpusDecoder, i32> {
        self.decoder.decoder_state_mut(stream_id)
    }
}
