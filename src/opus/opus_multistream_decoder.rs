//! Multistream Opus decoder wrapper.
//!
//! Upstream C: `src/opus_multistream_decoder.c`

use crate::opus::opus_decoder::OpusDecoder;
use crate::opus::opus_defines::{OPUS_BAD_ARG, OPUS_INVALID_PACKET};
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
    /// Upstream-style sizing helper.
    ///
    /// Returns zero for invalid stream shapes, non-zero for valid shapes.
    pub fn get_size(streams: i32, coupled_streams: i32) -> i32 {
        if streams < 1
            || coupled_streams < 0
            || coupled_streams > streams
            || streams > 255 - coupled_streams
        {
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
        let channels = self.layout.channels() as usize;
        if pcm.len() < frame_size as usize * channels {
            return OPUS_BAD_ARG;
        }
        if data.is_empty() {
            return self.decode_packet_loss_i16(pcm, frame_size, decode_fec);
        }
        let packets = match split_stream_packets(data, self.layout.streams()) {
            Ok(packets) => packets,
            Err(err) => return err,
        };

        let mut stream_pcm = Vec::with_capacity(self.layout.streams() as usize);
        let mut decoded_samples = -1i32;
        for (stream_id, packet) in packets.into_iter().enumerate() {
            let stream_channels = if stream_id < self.layout.coupled_streams() as usize {
                2usize
            } else {
                1usize
            };
            let mut tmp = vec![0i16; frame_size as usize * stream_channels];
            let ret = self.decoders[stream_id].decode(packet, &mut tmp, frame_size, decode_fec);
            if ret < 0 {
                return ret;
            }
            if decoded_samples < 0 {
                decoded_samples = ret;
            } else if decoded_samples != ret {
                return OPUS_INVALID_PACKET;
            }
            tmp.truncate(ret as usize * stream_channels);
            stream_pcm.push(tmp);
        }
        if decoded_samples < 0 {
            return OPUS_INVALID_PACKET;
        }
        map_output_i16(&self.layout, &stream_pcm, pcm, decoded_samples as usize);
        decoded_samples
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
        let channels = self.layout.channels() as usize;
        if pcm.len() < frame_size as usize * channels {
            return OPUS_BAD_ARG;
        }
        if data.is_empty() {
            return self.decode_packet_loss_f32(pcm, frame_size, decode_fec);
        }
        let packets = match split_stream_packets(data, self.layout.streams()) {
            Ok(packets) => packets,
            Err(err) => return err,
        };

        let mut stream_pcm = Vec::with_capacity(self.layout.streams() as usize);
        let mut decoded_samples = -1i32;
        for (stream_id, packet) in packets.into_iter().enumerate() {
            let stream_channels = if stream_id < self.layout.coupled_streams() as usize {
                2usize
            } else {
                1usize
            };
            let mut tmp = vec![0f32; frame_size as usize * stream_channels];
            let ret =
                self.decoders[stream_id].decode_float(packet, &mut tmp, frame_size, decode_fec);
            if ret < 0 {
                return ret;
            }
            if decoded_samples < 0 {
                decoded_samples = ret;
            } else if decoded_samples != ret {
                return OPUS_INVALID_PACKET;
            }
            tmp.truncate(ret as usize * stream_channels);
            stream_pcm.push(tmp);
        }
        if decoded_samples < 0 {
            return OPUS_INVALID_PACKET;
        }
        map_output_f32(&self.layout, &stream_pcm, pcm, decoded_samples as usize);
        decoded_samples
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

    fn decode_packet_loss_i16(
        &mut self,
        pcm: &mut [i16],
        frame_size: i32,
        decode_fec: bool,
    ) -> i32 {
        let mut stream_pcm = Vec::with_capacity(self.layout.streams() as usize);
        let mut decoded_samples = -1i32;
        for stream_id in 0..self.layout.streams() as usize {
            let stream_channels = if stream_id < self.layout.coupled_streams() as usize {
                2usize
            } else {
                1usize
            };
            let mut tmp = vec![0i16; frame_size as usize * stream_channels];
            let ret = self.decoders[stream_id].decode(&[], &mut tmp, frame_size, decode_fec);
            if ret < 0 {
                return ret;
            }
            if decoded_samples < 0 {
                decoded_samples = ret;
            } else if decoded_samples != ret {
                return OPUS_INVALID_PACKET;
            }
            tmp.truncate(ret as usize * stream_channels);
            stream_pcm.push(tmp);
        }
        if decoded_samples < 0 {
            return OPUS_INVALID_PACKET;
        }
        map_output_i16(&self.layout, &stream_pcm, pcm, decoded_samples as usize);
        decoded_samples
    }

    fn decode_packet_loss_f32(
        &mut self,
        pcm: &mut [f32],
        frame_size: i32,
        decode_fec: bool,
    ) -> i32 {
        let mut stream_pcm = Vec::with_capacity(self.layout.streams() as usize);
        let mut decoded_samples = -1i32;
        for stream_id in 0..self.layout.streams() as usize {
            let stream_channels = if stream_id < self.layout.coupled_streams() as usize {
                2usize
            } else {
                1usize
            };
            let mut tmp = vec![0f32; frame_size as usize * stream_channels];
            let ret = self.decoders[stream_id].decode_float(&[], &mut tmp, frame_size, decode_fec);
            if ret < 0 {
                return ret;
            }
            if decoded_samples < 0 {
                decoded_samples = ret;
            } else if decoded_samples != ret {
                return OPUS_INVALID_PACKET;
            }
            tmp.truncate(ret as usize * stream_channels);
            stream_pcm.push(tmp);
        }
        if decoded_samples < 0 {
            return OPUS_INVALID_PACKET;
        }
        map_output_f32(&self.layout, &stream_pcm, pcm, decoded_samples as usize);
        decoded_samples
    }
}

fn split_stream_packets(data: &[u8], streams: i32) -> Result<Vec<&[u8]>, i32> {
    if streams < 1 || data.is_empty() {
        return Err(OPUS_BAD_ARG);
    }
    let mut packets = Vec::with_capacity(streams as usize);
    let mut offset = 0usize;
    let mut remaining = data.len() as i32;
    for stream in 0..streams {
        if remaining <= 0 {
            return Err(OPUS_INVALID_PACKET);
        }
        let self_delimited = stream != streams - 1;
        let mut toc = 0u8;
        let mut size = [0i16; 48];
        let mut packet_offset = 0i32;
        let ret = opus_packet_parse_impl(
            &data[offset..offset + remaining as usize],
            self_delimited,
            Some(&mut toc),
            None,
            &mut size,
            None,
            Some(&mut packet_offset),
            None,
        );
        if ret < 0 {
            return Err(ret);
        }
        if packet_offset <= 0 || packet_offset > remaining {
            return Err(OPUS_INVALID_PACKET);
        }
        packets.push(&data[offset..offset + packet_offset as usize]);
        offset += packet_offset as usize;
        remaining -= packet_offset;
    }
    Ok(packets)
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
