//! Multistream Opus decoder wrapper.
//!
//! Upstream C: `src/opus_multistream_decoder.c`

use crate::celt::float_cast::{celt_float2int16, float2int};
use crate::opus::opus_decoder::{opus_decode_native, opus_packet_get_nb_samples, OpusDecoder};
use crate::opus::opus_defines::{
    OPUS_BAD_ARG, OPUS_BUFFER_TOO_SMALL, OPUS_INTERNAL_ERROR, OPUS_INVALID_PACKET, OPUS_OK,
};
use crate::opus::opus_multistream::OpusMultistreamLayout;
use crate::opus::packet::opus_packet_parse_impl;

/// Pure-Rust multistream decoder.
#[derive(Clone)]
pub struct OpusMSDecoder {
    sample_rate: i32,
    layout: OpusMultistreamLayout,
    decoders: Vec<OpusDecoder>,
}

impl OpusMSDecoder {
    #[inline]
    fn max_decode_frame_size(&self) -> i32 {
        self.sample_rate / 25 * 3
    }

    /// Upstream-style sizing helper.
    ///
    /// Returns zero for invalid stream shapes, non-zero for valid shapes.
    pub fn get_size(streams: i32, coupled_streams: i32) -> i32 {
        if streams < 1 || coupled_streams < 0 || coupled_streams > streams {
            0
        } else {
            core::mem::size_of::<Self>() as i32
        }
    }

    /// Create and initialize a multistream decoder.
    pub fn new(
        sample_rate: i32,
        channels: i32,
        streams: i32,
        coupled_streams: i32,
        mapping: &[u8],
    ) -> Result<Self, i32> {
        let layout = OpusMultistreamLayout::new(channels, streams, coupled_streams, mapping)?;
        if !layout.validate_for_decoder() {
            return Err(OPUS_BAD_ARG);
        }

        let mut decoders = Vec::with_capacity(streams as usize);
        for stream in 0..streams {
            let stream_channels = if stream < coupled_streams { 2 } else { 1 };
            decoders.push(OpusDecoder::new(sample_rate, stream_channels as usize)?);
        }
        Ok(Self {
            sample_rate,
            layout,
            decoders,
        })
    }

    /// Reinitialize an existing multistream decoder instance.
    pub fn init(
        &mut self,
        sample_rate: i32,
        channels: i32,
        streams: i32,
        coupled_streams: i32,
        mapping: &[u8],
    ) -> i32 {
        match Self::new(sample_rate, channels, streams, coupled_streams, mapping) {
            Ok(st) => {
                *self = st;
                OPUS_OK
            }
            Err(err) => err,
        }
    }

    #[inline]
    pub fn sample_rate(&self) -> i32 {
        self.sample_rate
    }

    #[inline]
    pub fn layout(&self) -> &OpusMultistreamLayout {
        &self.layout
    }

    /// Reset all child decoders.
    pub fn reset(&mut self) {
        for decoder in &mut self.decoders {
            decoder.reset();
        }
    }

    /// Return the XOR of child decoder final ranges.
    pub fn final_range(&self) -> u32 {
        self.decoders
            .iter()
            .fold(0u32, |acc, dec| acc ^ dec.final_range())
    }

    /// Decode multistream packet into interleaved i16 PCM.
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
        let effective_frame_size = frame_size.min(self.max_decode_frame_size());
        let channels = self.layout.channels() as usize;
        if pcm.len() < effective_frame_size as usize * channels {
            return OPUS_BAD_ARG;
        }

        let (stream_pcm_f32, decoded_samples) =
            match self.decode_streams_native(data, effective_frame_size, decode_fec, 1) {
                Ok(result) => result,
                Err(err) => return err,
            };

        let mut stream_pcm_i16 = Vec::with_capacity(stream_pcm_f32.len());
        for stream in stream_pcm_f32 {
            let mut out = vec![0i16; stream.len()];
            celt_float2int16(&stream, &mut out, stream.len());
            stream_pcm_i16.push(out);
        }

        map_output_i16(&self.layout, &stream_pcm_i16, pcm, decoded_samples);
        decoded_samples as i32
    }

    /// Decode multistream packet into interleaved f32 PCM.
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
        let effective_frame_size = frame_size.min(self.max_decode_frame_size());
        let channels = self.layout.channels() as usize;
        if pcm.len() < effective_frame_size as usize * channels {
            return OPUS_BAD_ARG;
        }

        let (stream_pcm, decoded_samples) =
            match self.decode_streams_native(data, effective_frame_size, decode_fec, 0) {
                Ok(result) => result,
                Err(err) => return err,
            };

        map_output_f32(&self.layout, &stream_pcm, pcm, decoded_samples);
        decoded_samples as i32
    }

    /// Decode multistream packet into 24-bit PCM (stored in i32).
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
        let effective_frame_size = frame_size.min(self.max_decode_frame_size());
        let channels = self.layout.channels() as usize;
        if pcm.len() < effective_frame_size as usize * channels {
            return OPUS_BAD_ARG;
        }

        let (stream_pcm, decoded_samples) =
            match self.decode_streams_native(data, effective_frame_size, decode_fec, 0) {
                Ok(result) => result,
                Err(err) => return err,
            };

        let mut stream_pcm_i32 = Vec::with_capacity(stream_pcm.len());
        for stream in stream_pcm {
            let mut out = vec![0i32; stream.len()];
            for (dst, src) in out.iter_mut().zip(stream.iter()) {
                *dst = float2int(32768.0f32 * 256.0f32 * *src);
            }
            stream_pcm_i32.push(out);
        }

        map_output_i32(&self.layout, &stream_pcm_i32, pcm, decoded_samples);
        decoded_samples as i32
    }

    pub(crate) fn decode_streams_native(
        &mut self,
        data: &[u8],
        frame_size: i32,
        decode_fec: bool,
        soft_clip: i32,
    ) -> Result<(Vec<Vec<f32>>, usize), i32> {
        if frame_size <= 0 {
            return Err(OPUS_BAD_ARG);
        }

        let do_plc = data.is_empty();
        let mut frame_size = frame_size;
        if !do_plc && data.len() < (2 * self.layout.streams() - 1) as usize {
            return Err(OPUS_INVALID_PACKET);
        }
        if !do_plc {
            let samples =
                multistream_packet_validate(data, self.layout.streams(), self.sample_rate)?;
            if samples > frame_size {
                return Err(OPUS_BUFFER_TOO_SMALL);
            }
        }

        let mut stream_pcm = Vec::with_capacity(self.layout.streams() as usize);
        let mut payload = data;
        let mut remaining = data.len() as i32;

        for stream_id in 0..self.layout.streams() as usize {
            let stream_channels = if stream_id < self.layout.coupled_streams() as usize {
                2usize
            } else {
                1usize
            };
            if !do_plc && remaining <= 0 {
                return Err(OPUS_INTERNAL_ERROR);
            }
            let mut tmp = vec![0f32; frame_size as usize * stream_channels];
            let self_delimited = stream_id + 1 != self.layout.streams() as usize;
            let mut packet_offset = 0i32;
            let ret = opus_decode_native(
                &mut self.decoders[stream_id],
                payload,
                &mut tmp,
                frame_size,
                decode_fec as i32,
                self_delimited,
                if data.is_empty() {
                    None
                } else {
                    Some(&mut packet_offset)
                },
                soft_clip,
            );
            if ret <= 0 {
                return Err(ret);
            }

            if !do_plc {
                if packet_offset <= 0 || packet_offset > remaining {
                    return Err(OPUS_INVALID_PACKET);
                }
                payload = &payload[packet_offset as usize..];
                remaining -= packet_offset;
            }

            frame_size = ret;
            tmp.truncate(frame_size as usize * stream_channels);
            stream_pcm.push(tmp);
        }
        Ok((stream_pcm, frame_size as usize))
    }

    pub fn set_gain(&mut self, gain: i32) -> Result<(), i32> {
        for decoder in &mut self.decoders {
            decoder.set_gain(gain)?;
        }
        Ok(())
    }

    pub fn set_complexity(&mut self, complexity: i32) -> Result<(), i32> {
        for decoder in &mut self.decoders {
            decoder.set_complexity(complexity)?;
        }
        Ok(())
    }

    pub fn set_ignore_extensions(&mut self, ignore: bool) {
        for decoder in &mut self.decoders {
            decoder.set_ignore_extensions(ignore);
        }
    }

    #[cfg(feature = "osce")]
    pub fn set_osce_bwe(&mut self, enabled: bool) {
        for decoder in &mut self.decoders {
            decoder.set_osce_bwe(enabled);
        }
    }

    pub fn set_phase_inversion_disabled(&mut self, disabled: bool) {
        for decoder in &mut self.decoders {
            decoder.set_phase_inversion_disabled(disabled);
        }
    }

    pub fn decoder_state(&self, stream_id: i32) -> Result<&OpusDecoder, i32> {
        if stream_id < 0 || stream_id >= self.layout.streams() {
            return Err(OPUS_BAD_ARG);
        }
        self.decoders.get(stream_id as usize).ok_or(OPUS_BAD_ARG)
    }

    pub fn decoder_state_mut(&mut self, stream_id: i32) -> Result<&mut OpusDecoder, i32> {
        if stream_id < 0 || stream_id >= self.layout.streams() {
            return Err(OPUS_BAD_ARG);
        }
        self.decoders
            .get_mut(stream_id as usize)
            .ok_or(OPUS_BAD_ARG)
    }

    pub fn gain(&self) -> i32 {
        self.decoders.first().map(OpusDecoder::gain).unwrap_or(0)
    }

    pub fn bandwidth(&self) -> i32 {
        self.decoders
            .first()
            .map(OpusDecoder::get_bandwidth)
            .unwrap_or(0)
    }

    pub fn complexity(&self) -> i32 {
        self.decoders
            .first()
            .map(OpusDecoder::complexity)
            .unwrap_or(0)
    }

    pub fn phase_inversion_disabled(&self) -> bool {
        self.decoders
            .first()
            .map(OpusDecoder::phase_inversion_disabled)
            .unwrap_or(false)
    }

    pub fn last_packet_duration(&self) -> i32 {
        self.decoders
            .first()
            .map(OpusDecoder::last_packet_duration)
            .unwrap_or(0)
    }

    pub fn ignore_extensions(&self) -> bool {
        self.decoders
            .first()
            .map(OpusDecoder::ignore_extensions)
            .unwrap_or(false)
    }

    #[cfg(feature = "osce")]
    pub fn osce_bwe(&self) -> bool {
        self.decoders
            .first()
            .map(OpusDecoder::osce_bwe)
            .unwrap_or(false)
    }
}

fn multistream_packet_validate(data: &[u8], nb_streams: i32, fs: i32) -> Result<i32, i32> {
    let mut payload = data;
    let mut remaining = data.len() as i32;
    let mut samples = 0i32;

    for stream_idx in 0..nb_streams {
        if remaining <= 0 {
            return Err(OPUS_INVALID_PACKET);
        }

        let mut toc = 0u8;
        let mut size = [0i16; 48];
        let mut packet_offset = 0i32;
        let count = opus_packet_parse_impl(
            payload,
            stream_idx != nb_streams - 1,
            Some(&mut toc),
            None,
            &mut size,
            None,
            Some(&mut packet_offset),
            None,
        );
        if count < 0 {
            return Err(count);
        }
        if packet_offset <= 0 || packet_offset > remaining {
            return Err(OPUS_INVALID_PACKET);
        }

        let tmp_samples = opus_packet_get_nb_samples(&payload[..packet_offset as usize], fs);
        if stream_idx != 0 && samples != tmp_samples {
            return Err(OPUS_INVALID_PACKET);
        }
        samples = tmp_samples;
        payload = &payload[packet_offset as usize..];
        remaining -= packet_offset;
    }

    Ok(samples)
}

/// Upstream-style free function wrapper.
pub fn opus_multistream_decoder_get_size(streams: i32, coupled_streams: i32) -> i32 {
    OpusMSDecoder::get_size(streams, coupled_streams)
}

/// Upstream-style free function wrapper.
pub fn opus_multistream_decoder_create(
    sample_rate: i32,
    channels: i32,
    streams: i32,
    coupled_streams: i32,
    mapping: &[u8],
) -> Result<OpusMSDecoder, i32> {
    OpusMSDecoder::new(sample_rate, channels, streams, coupled_streams, mapping)
}

/// Upstream-style free function wrapper.
pub fn opus_multistream_decoder_init(
    st: &mut OpusMSDecoder,
    sample_rate: i32,
    channels: i32,
    streams: i32,
    coupled_streams: i32,
    mapping: &[u8],
) -> i32 {
    st.init(sample_rate, channels, streams, coupled_streams, mapping)
}

/// Upstream-style free function wrapper.
pub fn opus_multistream_decoder_destroy(_st: OpusMSDecoder) {}

/// Upstream-style free function wrapper.
pub fn opus_multistream_decode(
    st: &mut OpusMSDecoder,
    data: &[u8],
    pcm: &mut [i16],
    frame_size: i32,
    decode_fec: bool,
) -> i32 {
    st.decode(data, pcm, frame_size, decode_fec)
}

/// Upstream-style free function wrapper.
pub fn opus_multistream_decode_float(
    st: &mut OpusMSDecoder,
    data: &[u8],
    pcm: &mut [f32],
    frame_size: i32,
    decode_fec: bool,
) -> i32 {
    st.decode_float(data, pcm, frame_size, decode_fec)
}

/// Upstream-style free function wrapper.
pub fn opus_multistream_decode24(
    st: &mut OpusMSDecoder,
    data: &[u8],
    pcm: &mut [i32],
    frame_size: i32,
    decode_fec: bool,
) -> i32 {
    st.decode24(data, pcm, frame_size, decode_fec)
}

/// Upstream-style helper for `OPUS_MULTISTREAM_GET_DECODER_STATE_REQUEST`.
pub fn opus_multistream_decoder_get_decoder_state(
    st: &mut OpusMSDecoder,
    stream_id: i32,
) -> Result<&mut OpusDecoder, i32> {
    st.decoder_state_mut(stream_id)
}

fn mapping_to_stream_channel(
    layout: &OpusMultistreamLayout,
    map_index: u8,
) -> Option<(usize, usize)> {
    if map_index == 255 {
        return None;
    }
    let map_index = map_index as i32;
    let coupled = layout.coupled_streams();
    if map_index < coupled * 2 {
        Some(((map_index / 2) as usize, (map_index & 1) as usize))
    } else {
        Some(((coupled + map_index - coupled * 2) as usize, 0))
    }
}

fn map_output_i16(
    layout: &OpusMultistreamLayout,
    stream_pcm: &[Vec<i16>],
    output: &mut [i16],
    decoded_samples: usize,
) {
    let channels = layout.channels() as usize;
    output[..decoded_samples * channels].fill(0);
    for out_channel in 0..channels {
        let mapping = layout.mapping()[out_channel];
        if let Some((stream_idx, stream_channel)) = mapping_to_stream_channel(layout, mapping) {
            let stream_channels = if stream_idx < layout.coupled_streams() as usize {
                2
            } else {
                1
            };
            let source = &stream_pcm[stream_idx];
            for frame in 0..decoded_samples {
                output[frame * channels + out_channel] =
                    source[frame * stream_channels + stream_channel];
            }
        }
    }
}

fn map_output_f32(
    layout: &OpusMultistreamLayout,
    stream_pcm: &[Vec<f32>],
    output: &mut [f32],
    decoded_samples: usize,
) {
    let channels = layout.channels() as usize;
    output[..decoded_samples * channels].fill(0.);
    for out_channel in 0..channels {
        let mapping = layout.mapping()[out_channel];
        if let Some((stream_idx, stream_channel)) = mapping_to_stream_channel(layout, mapping) {
            let stream_channels = if stream_idx < layout.coupled_streams() as usize {
                2
            } else {
                1
            };
            let source = &stream_pcm[stream_idx];
            for frame in 0..decoded_samples {
                output[frame * channels + out_channel] =
                    source[frame * stream_channels + stream_channel];
            }
        }
    }
}

fn map_output_i32(
    layout: &OpusMultistreamLayout,
    stream_pcm: &[Vec<i32>],
    output: &mut [i32],
    decoded_samples: usize,
) {
    let channels = layout.channels() as usize;
    output[..decoded_samples * channels].fill(0);
    for out_channel in 0..channels {
        let mapping = layout.mapping()[out_channel];
        if let Some((stream_idx, stream_channel)) = mapping_to_stream_channel(layout, mapping) {
            let stream_channels = if stream_idx < layout.coupled_streams() as usize {
                2
            } else {
                1
            };
            let source = &stream_pcm[stream_idx];
            for frame in 0..decoded_samples {
                output[frame * channels + out_channel] =
                    source[frame * stream_channels + stream_channel];
            }
        }
    }
}
