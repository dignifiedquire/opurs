//! Projection Opus decoder wrapper.
//!
//! Upstream C: `src/opus_projection_decoder.c`

use crate::opus::mapping_matrix::MappingMatrix;
use crate::opus::opus_decoder::OpusDecoder;
use crate::opus::opus_defines::{OPUS_BAD_ARG, OPUS_OK};
use crate::opus::opus_multistream_decoder::OpusMSDecoder;

/// Pure-Rust projection decoder.
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
    pub fn new(
        sample_rate: i32,
        channels: i32,
        streams: i32,
        coupled_streams: i32,
        demixing_matrix: &[u8],
    ) -> Result<Self, i32> {
        let input_channels = streams + coupled_streams;
        if channels <= 0 || input_channels <= 0 {
            return Err(OPUS_BAD_ARG);
        }
        let expected_matrix_size = (input_channels as i64)
            .checked_mul(channels as i64)
            .and_then(|v| v.checked_mul(2))
            .ok_or(OPUS_BAD_ARG)? as usize;
        if demixing_matrix.len() != expected_matrix_size {
            return Err(OPUS_BAD_ARG);
        }

        let demixing_matrix =
            MappingMatrix::from_bytes_le(channels, input_channels, 0, demixing_matrix)?;

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
    pub fn init(
        &mut self,
        sample_rate: i32,
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
            Err(err) => err,
        }
    }

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

        let mut input_pcm = vec![0i16; frame_size * input_channels];
        let decoded = self
            .decoder
            .decode(data, &mut input_pcm, frame_size as i32, decode_fec);
        if decoded <= 0 {
            return decoded;
        }
        let decoded = decoded as usize;

        let output = &mut pcm[..decoded * output_channels];
        output.fill(0);
        for input_row in 0..input_channels {
            if let Err(err) = self.demixing_matrix.multiply_channel_out_short_i16(
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

    pub fn set_gain(&mut self, gain: i32) -> Result<(), i32> {
        self.decoder.set_gain(gain)
    }

    pub fn set_complexity(&mut self, complexity: i32) -> Result<(), i32> {
        self.decoder.set_complexity(complexity)
    }

    pub fn set_phase_inversion_disabled(&mut self, disabled: bool) {
        self.decoder.set_phase_inversion_disabled(disabled);
    }

    pub fn set_ignore_extensions(&mut self, ignore: bool) {
        self.decoder.set_ignore_extensions(ignore);
    }

    pub fn reset(&mut self) {
        self.decoder.reset();
    }

    pub fn final_range(&self) -> u32 {
        self.decoder.final_range()
    }

    pub fn gain(&self) -> i32 {
        self.decoder.gain()
    }

    pub fn sample_rate(&self) -> i32 {
        self.decoder.sample_rate()
    }

    pub fn complexity(&self) -> i32 {
        self.decoder.complexity()
    }

    pub fn bandwidth(&self) -> i32 {
        self.decoder.bandwidth()
    }

    pub fn phase_inversion_disabled(&self) -> bool {
        self.decoder.phase_inversion_disabled()
    }

    pub fn last_packet_duration(&self) -> i32 {
        self.decoder.last_packet_duration()
    }

    pub fn ignore_extensions(&self) -> bool {
        self.decoder.ignore_extensions()
    }

    pub fn decoder_state(&self, stream_id: i32) -> Result<&OpusDecoder, i32> {
        self.decoder.decoder_state(stream_id)
    }

    pub fn decoder_state_mut(&mut self, stream_id: i32) -> Result<&mut OpusDecoder, i32> {
        self.decoder.decoder_state_mut(stream_id)
    }
}

/// Upstream-style free function wrapper.
pub fn opus_projection_decoder_get_size(channels: i32, streams: i32, coupled_streams: i32) -> i32 {
    OpusProjectionDecoder::get_size(channels, streams, coupled_streams)
}

/// Upstream-style free function wrapper.
pub fn opus_projection_decoder_create(
    sample_rate: i32,
    channels: i32,
    streams: i32,
    coupled_streams: i32,
    demixing_matrix: &[u8],
) -> Result<OpusProjectionDecoder, i32> {
    OpusProjectionDecoder::new(
        sample_rate,
        channels,
        streams,
        coupled_streams,
        demixing_matrix,
    )
}

/// Upstream-style free function wrapper.
pub fn opus_projection_decoder_init(
    st: &mut OpusProjectionDecoder,
    sample_rate: i32,
    channels: i32,
    streams: i32,
    coupled_streams: i32,
    demixing_matrix: &[u8],
) -> i32 {
    st.init(
        sample_rate,
        channels,
        streams,
        coupled_streams,
        demixing_matrix,
    )
}

/// Upstream-style free function wrapper.
pub fn opus_projection_decode(
    st: &mut OpusProjectionDecoder,
    data: &[u8],
    pcm: &mut [i16],
    frame_size: i32,
    decode_fec: bool,
) -> i32 {
    st.decode(data, pcm, frame_size, decode_fec)
}

/// Upstream-style free function wrapper.
pub fn opus_projection_decode_float(
    st: &mut OpusProjectionDecoder,
    data: &[u8],
    pcm: &mut [f32],
    frame_size: i32,
    decode_fec: bool,
) -> i32 {
    st.decode_float(data, pcm, frame_size, decode_fec)
}

/// Upstream-style free function wrapper.
pub fn opus_projection_decode24(
    st: &mut OpusProjectionDecoder,
    data: &[u8],
    pcm: &mut [i32],
    frame_size: i32,
    decode_fec: bool,
) -> i32 {
    st.decode24(data, pcm, frame_size, decode_fec)
}

/// Upstream-style free function wrapper.
pub fn opus_projection_decoder_destroy(_st: OpusProjectionDecoder) {}

/// Upstream-style helper for `OPUS_MULTISTREAM_GET_DECODER_STATE_REQUEST`.
pub fn opus_projection_decoder_get_decoder_state(
    st: &mut OpusProjectionDecoder,
    stream_id: i32,
) -> Result<&mut OpusDecoder, i32> {
    st.decoder_state_mut(stream_id)
}
