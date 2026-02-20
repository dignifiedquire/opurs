//! Multistream Opus encoder wrapper.
//!
//! Upstream C: `src/opus_multistream_encoder.c`

use crate::enums::{Bandwidth, Bitrate, Channels};
use crate::opus::opus_defines::{OPUS_BAD_ARG, OPUS_BUFFER_TOO_SMALL, OPUS_OK};
use crate::opus::opus_encoder::OpusEncoder;
use crate::opus::opus_multistream::{OpusMultistreamConfig, OpusMultistreamLayout};
use crate::opus::repacketizer::{FrameSource, OpusRepacketizer};

/// Pure-Rust multistream encoder.
#[derive(Clone)]
pub struct OpusMSEncoder {
    config: OpusMultistreamConfig,
    encoders: Vec<OpusEncoder>,
}

impl OpusMSEncoder {
    /// Upstream-style sizing helper.
    ///
    /// Returns zero for invalid stream shapes, non-zero for valid shapes.
    pub fn get_size(streams: i32, coupled_streams: i32) -> i32 {
        if streams < 1 || coupled_streams < 0 || coupled_streams > streams {
            0
        } else {
            // Rust struct sizing is not API-compatible with C byte layout.
            core::mem::size_of::<Self>() as i32
        }
    }

    /// Create and initialize a multistream encoder.
    pub fn new(
        sample_rate: i32,
        channels: i32,
        streams: i32,
        coupled_streams: i32,
        mapping: &[u8],
        application: i32,
    ) -> Result<Self, i32> {
        let layout = OpusMultistreamLayout::new(channels, streams, coupled_streams, mapping)?;
        if !layout.validate_for_encoder() {
            return Err(OPUS_BAD_ARG);
        }
        let config = OpusMultistreamConfig::new(sample_rate, application, layout.clone())?;

        let mut encoders = Vec::with_capacity(streams as usize);
        for stream in 0..streams {
            let stream_channels = if stream < coupled_streams { 2 } else { 1 };
            encoders.push(OpusEncoder::new(sample_rate, stream_channels, application)?);
        }

        Ok(Self { config, encoders })
    }

    /// Reinitialize an existing multistream encoder instance.
    pub fn init(
        &mut self,
        sample_rate: i32,
        channels: i32,
        streams: i32,
        coupled_streams: i32,
        mapping: &[u8],
        application: i32,
    ) -> i32 {
        match Self::new(
            sample_rate,
            channels,
            streams,
            coupled_streams,
            mapping,
            application,
        ) {
            Ok(st) => {
                *self = st;
                OPUS_OK
            }
            Err(err) => err,
        }
    }

    #[inline]
    pub fn layout(&self) -> &OpusMultistreamLayout {
        self.config.layout()
    }

    #[inline]
    pub fn sample_rate(&self) -> i32 {
        self.config.sample_rate()
    }

    #[inline]
    pub fn application(&self) -> i32 {
        self.config.application()
    }

    /// Reset all child encoders.
    pub fn reset(&mut self) {
        for encoder in &mut self.encoders {
            encoder.reset();
        }
    }

    /// Apply bitrate to all stream encoders.
    ///
    /// Explicit bitrates are split approximately evenly across streams.
    pub fn set_bitrate(&mut self, bitrate: Bitrate) {
        let stream_count = self.encoders.len() as i32;
        let per_stream = match bitrate {
            Bitrate::Bits(bps) if stream_count > 0 => {
                Bitrate::Bits((bps + stream_count / 2) / stream_count)
            }
            other => other,
        };
        for encoder in &mut self.encoders {
            encoder.set_bitrate(per_stream);
        }
    }

    pub fn set_complexity(&mut self, complexity: i32) -> Result<(), i32> {
        for encoder in &mut self.encoders {
            encoder.set_complexity(complexity)?;
        }
        Ok(())
    }

    pub fn set_vbr(&mut self, enabled: bool) {
        for encoder in &mut self.encoders {
            encoder.set_vbr(enabled);
        }
    }

    pub fn set_vbr_constraint(&mut self, enabled: bool) {
        for encoder in &mut self.encoders {
            encoder.set_vbr_constraint(enabled);
        }
    }

    pub fn set_bandwidth(&mut self, bandwidth: Option<Bandwidth>) {
        for encoder in &mut self.encoders {
            encoder.set_bandwidth(bandwidth);
        }
    }

    pub fn set_inband_fec(&mut self, value: i32) -> Result<(), i32> {
        for encoder in &mut self.encoders {
            encoder.set_inband_fec(value)?;
        }
        Ok(())
    }

    pub fn set_packet_loss_perc(&mut self, pct: i32) -> Result<(), i32> {
        for encoder in &mut self.encoders {
            encoder.set_packet_loss_perc(pct)?;
        }
        Ok(())
    }

    pub fn set_dtx(&mut self, enabled: bool) {
        for encoder in &mut self.encoders {
            encoder.set_dtx(enabled);
        }
    }

    pub fn set_force_channels(&mut self, channels: Option<Channels>) -> Result<(), i32> {
        for encoder in &mut self.encoders {
            encoder.set_force_channels(channels)?;
        }
        Ok(())
    }

    #[cfg(feature = "qext")]
    pub fn set_qext(&mut self, enabled: bool) {
        for encoder in &mut self.encoders {
            encoder.set_qext(enabled);
        }
    }

    /// Return the XOR of child encoder final ranges.
    pub fn final_range(&self) -> u32 {
        self.encoders
            .iter()
            .fold(0u32, |acc, enc| acc ^ enc.final_range())
    }

    /// Return the maximum lookahead across child encoders.
    pub fn lookahead(&self) -> i32 {
        self.encoders
            .iter()
            .map(OpusEncoder::lookahead)
            .max()
            .unwrap_or(0)
    }

    pub fn complexity(&self) -> i32 {
        self.encoders
            .first()
            .map(OpusEncoder::complexity)
            .unwrap_or(0)
    }

    pub fn vbr(&self) -> bool {
        self.encoders.first().map(OpusEncoder::vbr).unwrap_or(false)
    }

    pub fn vbr_constraint(&self) -> bool {
        self.encoders
            .first()
            .map(OpusEncoder::vbr_constraint)
            .unwrap_or(false)
    }

    pub fn bandwidth(&self) -> i32 {
        self.encoders
            .first()
            .map(OpusEncoder::get_bandwidth)
            .unwrap_or(0)
    }

    pub fn inband_fec(&self) -> i32 {
        self.encoders
            .first()
            .map(OpusEncoder::inband_fec)
            .unwrap_or(0)
    }

    pub fn packet_loss_perc(&self) -> i32 {
        self.encoders
            .first()
            .map(OpusEncoder::packet_loss_perc)
            .unwrap_or(0)
    }

    pub fn dtx(&self) -> bool {
        self.encoders.first().map(OpusEncoder::dtx).unwrap_or(false)
    }

    pub fn force_channels(&self) -> Option<Channels> {
        self.encoders.first().and_then(OpusEncoder::force_channels)
    }

    #[cfg(feature = "qext")]
    pub fn qext(&self) -> bool {
        self.encoders
            .first()
            .map(OpusEncoder::qext)
            .unwrap_or(false)
    }

    pub fn bitrate(&self) -> i32 {
        self.encoders.iter().map(OpusEncoder::bitrate).sum()
    }

    /// Encode interleaved i16 PCM into a multistream Opus packet.
    pub fn encode(&mut self, pcm: &[i16], output: &mut [u8]) -> i32 {
        self.encode_impl_i16(pcm, output)
    }

    /// Encode interleaved f32 PCM into a multistream Opus packet.
    pub fn encode_float(&mut self, pcm: &[f32], output: &mut [u8]) -> i32 {
        self.encode_impl_f32(pcm, output)
    }

    fn encode_impl_i16(&mut self, pcm: &[i16], output: &mut [u8]) -> i32 {
        let channels = self.layout().channels() as usize;
        if channels == 0 || pcm.is_empty() || !pcm.len().is_multiple_of(channels) {
            return OPUS_BAD_ARG;
        }
        let frame_size = pcm.len() / channels;
        let mut write_offset = 0usize;
        for stream_id in 0..self.layout().streams() {
            let selected = match stream_input_channels(self.layout(), stream_id) {
                Some(v) => v,
                None => return OPUS_BAD_ARG,
            };
            let stream_pcm = extract_stream_pcm_i16(pcm, frame_size, channels, &selected);
            let encoder = &mut self.encoders[stream_id as usize];
            let mut stream_packet = vec![0u8; 1500];
            let len = encoder.encode(&stream_pcm, &mut stream_packet);
            if len < 0 {
                return len;
            }
            stream_packet.truncate(len as usize);
            let stream_packet = if stream_id != self.layout().streams() - 1 {
                match make_self_delimited(&stream_packet) {
                    Ok(pkt) => pkt,
                    Err(err) => return err,
                }
            } else {
                stream_packet
            };
            if write_offset + stream_packet.len() > output.len() {
                return OPUS_BUFFER_TOO_SMALL;
            }
            output[write_offset..write_offset + stream_packet.len()]
                .copy_from_slice(&stream_packet);
            write_offset += stream_packet.len();
        }
        write_offset as i32
    }

    fn encode_impl_f32(&mut self, pcm: &[f32], output: &mut [u8]) -> i32 {
        let channels = self.layout().channels() as usize;
        if channels == 0 || pcm.is_empty() || !pcm.len().is_multiple_of(channels) {
            return OPUS_BAD_ARG;
        }
        let frame_size = pcm.len() / channels;
        let mut write_offset = 0usize;
        for stream_id in 0..self.layout().streams() {
            let selected = match stream_input_channels(self.layout(), stream_id) {
                Some(v) => v,
                None => return OPUS_BAD_ARG,
            };
            let stream_pcm = extract_stream_pcm_f32(pcm, frame_size, channels, &selected);
            let encoder = &mut self.encoders[stream_id as usize];
            let mut stream_packet = vec![0u8; 1500];
            let len = encoder.encode_float(&stream_pcm, &mut stream_packet);
            if len < 0 {
                return len;
            }
            stream_packet.truncate(len as usize);
            let stream_packet = if stream_id != self.layout().streams() - 1 {
                match make_self_delimited(&stream_packet) {
                    Ok(pkt) => pkt,
                    Err(err) => return err,
                }
            } else {
                stream_packet
            };
            if write_offset + stream_packet.len() > output.len() {
                return OPUS_BUFFER_TOO_SMALL;
            }
            output[write_offset..write_offset + stream_packet.len()]
                .copy_from_slice(&stream_packet);
            write_offset += stream_packet.len();
        }
        write_offset as i32
    }
}

/// Upstream-style free function wrapper.
pub fn opus_multistream_encoder_get_size(streams: i32, coupled_streams: i32) -> i32 {
    OpusMSEncoder::get_size(streams, coupled_streams)
}

/// Upstream-style free function wrapper.
pub fn opus_multistream_encoder_create(
    sample_rate: i32,
    channels: i32,
    streams: i32,
    coupled_streams: i32,
    mapping: &[u8],
    application: i32,
) -> Result<OpusMSEncoder, i32> {
    OpusMSEncoder::new(
        sample_rate,
        channels,
        streams,
        coupled_streams,
        mapping,
        application,
    )
}

/// Upstream-style free function wrapper.
pub fn opus_multistream_encoder_init(
    st: &mut OpusMSEncoder,
    sample_rate: i32,
    channels: i32,
    streams: i32,
    coupled_streams: i32,
    mapping: &[u8],
    application: i32,
) -> i32 {
    st.init(
        sample_rate,
        channels,
        streams,
        coupled_streams,
        mapping,
        application,
    )
}

/// Upstream-style free function wrapper.
pub fn opus_multistream_encoder_destroy(_st: OpusMSEncoder) {}

/// Upstream-style free function wrapper.
pub fn opus_multistream_encode(
    st: &mut OpusMSEncoder,
    pcm: &[i16],
    frame_size: i32,
    data: &mut [u8],
) -> i32 {
    let channels = st.layout().channels() as usize;
    if frame_size <= 0 || pcm.len() != frame_size as usize * channels {
        return OPUS_BAD_ARG;
    }
    st.encode(pcm, data)
}

/// Upstream-style free function wrapper.
pub fn opus_multistream_encode_float(
    st: &mut OpusMSEncoder,
    pcm: &[f32],
    frame_size: i32,
    data: &mut [u8],
) -> i32 {
    let channels = st.layout().channels() as usize;
    if frame_size <= 0 || pcm.len() != frame_size as usize * channels {
        return OPUS_BAD_ARG;
    }
    st.encode_float(pcm, data)
}

fn make_self_delimited(packet: &[u8]) -> Result<Vec<u8>, i32> {
    let mut rp = OpusRepacketizer::default();
    let ret = rp.cat(packet);
    if ret < 0 {
        return Err(ret);
    }
    let mut buffer = packet.to_vec();
    buffer.resize(packet.len() + 2, 0);
    let ret = rp.out_range_impl(
        0,
        rp.get_nb_frames(),
        &mut buffer,
        true,
        false,
        FrameSource::Data { offset: 0 },
    );
    if ret < 0 {
        return Err(ret);
    }
    buffer.truncate(ret as usize);
    Ok(buffer)
}

fn stream_input_channels(layout: &OpusMultistreamLayout, stream_id: i32) -> Option<Vec<usize>> {
    if stream_id < layout.coupled_streams() {
        Some(vec![
            layout.left_channel(stream_id, -1)?,
            layout.right_channel(stream_id, -1)?,
        ])
    } else {
        Some(vec![layout.mono_channel(stream_id, -1)?])
    }
}

fn extract_stream_pcm_i16(
    pcm: &[i16],
    frame_size: usize,
    channels: usize,
    selected_channels: &[usize],
) -> Vec<i16> {
    let stream_channels = selected_channels.len();
    let mut out = vec![0i16; frame_size * stream_channels];
    for frame in 0..frame_size {
        for (idx, &channel) in selected_channels.iter().enumerate() {
            out[frame * stream_channels + idx] = pcm[frame * channels + channel];
        }
    }
    out
}

fn extract_stream_pcm_f32(
    pcm: &[f32],
    frame_size: usize,
    channels: usize,
    selected_channels: &[usize],
) -> Vec<f32> {
    let stream_channels = selected_channels.len();
    let mut out = vec![0f32; frame_size * stream_channels];
    for frame in 0..frame_size {
        for (idx, &channel) in selected_channels.iter().enumerate() {
            out[frame * stream_channels + idx] = pcm[frame * channels + channel];
        }
    }
    out
}
