//! Multistream Opus encoder wrapper.
//!
//! Upstream C: `src/opus_multistream_encoder.c`

use crate::enums::{Application, Bandwidth, Bitrate, Channels, FrameSize, Signal};
use crate::opus::opus_defines::{OPUS_BAD_ARG, OPUS_BUFFER_TOO_SMALL, OPUS_OK, OPUS_UNIMPLEMENTED};
use crate::opus::opus_encoder::OpusEncoder;
use crate::opus::opus_multistream::{OpusMultistreamConfig, OpusMultistreamLayout};
use crate::opus::repacketizer::{FrameSource, OpusRepacketizer};

/// Pure-Rust multistream encoder.
#[derive(Clone)]
pub struct OpusMSEncoder {
    config: OpusMultistreamConfig,
    encoders: Vec<OpusEncoder>,
    requested_bitrate: Bitrate,
}

#[derive(Clone, Copy)]
struct VorbisLayout {
    streams: i32,
    coupled_streams: i32,
    mapping: [u8; 8],
}

// Index is channels-1.
const VORBIS_MAPPINGS: [VorbisLayout; 8] = [
    VorbisLayout {
        streams: 1,
        coupled_streams: 0,
        mapping: [0, 0, 0, 0, 0, 0, 0, 0],
    },
    VorbisLayout {
        streams: 1,
        coupled_streams: 1,
        mapping: [0, 1, 0, 0, 0, 0, 0, 0],
    },
    VorbisLayout {
        streams: 2,
        coupled_streams: 1,
        mapping: [0, 2, 1, 0, 0, 0, 0, 0],
    },
    VorbisLayout {
        streams: 2,
        coupled_streams: 2,
        mapping: [0, 1, 2, 3, 0, 0, 0, 0],
    },
    VorbisLayout {
        streams: 3,
        coupled_streams: 2,
        mapping: [0, 4, 1, 2, 3, 0, 0, 0],
    },
    VorbisLayout {
        streams: 4,
        coupled_streams: 2,
        mapping: [0, 4, 1, 2, 3, 5, 0, 0],
    },
    VorbisLayout {
        streams: 4,
        coupled_streams: 3,
        mapping: [0, 4, 1, 2, 3, 5, 6, 0],
    },
    VorbisLayout {
        streams: 5,
        coupled_streams: 3,
        mapping: [0, 6, 1, 2, 3, 4, 5, 7],
    },
];

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

        Ok(Self {
            config,
            encoders,
            requested_bitrate: Bitrate::Auto,
        })
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

    pub fn encoder_state(&self, stream_id: i32) -> Result<&OpusEncoder, i32> {
        if stream_id < 0 || stream_id >= self.layout().streams() {
            return Err(OPUS_BAD_ARG);
        }
        self.encoders.get(stream_id as usize).ok_or(OPUS_BAD_ARG)
    }

    pub fn encoder_state_mut(&mut self, stream_id: i32) -> Result<&mut OpusEncoder, i32> {
        if stream_id < 0 || stream_id >= self.layout().streams() {
            return Err(OPUS_BAD_ARG);
        }
        self.encoders
            .get_mut(stream_id as usize)
            .ok_or(OPUS_BAD_ARG)
    }

    /// Apply bitrate to all stream encoders.
    ///
    /// Explicit bitrates are split approximately evenly across streams.
    pub fn set_bitrate(&mut self, bitrate: Bitrate) {
        self.requested_bitrate = bitrate;
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

    pub fn set_application(&mut self, application: i32) -> Result<(), i32> {
        let app = Application::try_from(application).map_err(|_| OPUS_BAD_ARG)?;
        for encoder in &mut self.encoders {
            encoder.set_application(app)?;
        }
        self.config.set_application(application)?;
        Ok(())
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

    pub fn set_max_bandwidth(&mut self, bandwidth: Bandwidth) {
        for encoder in &mut self.encoders {
            encoder.set_max_bandwidth(bandwidth);
        }
    }

    pub fn set_signal(&mut self, signal: Option<Signal>) {
        for encoder in &mut self.encoders {
            encoder.set_signal(signal);
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

    pub fn set_lsb_depth(&mut self, depth: i32) -> Result<(), i32> {
        for encoder in &mut self.encoders {
            encoder.set_lsb_depth(depth)?;
        }
        Ok(())
    }

    pub fn set_expert_frame_duration(&mut self, fs: FrameSize) {
        for encoder in &mut self.encoders {
            encoder.set_expert_frame_duration(fs);
        }
    }

    pub fn set_prediction_disabled(&mut self, disabled: bool) {
        for encoder in &mut self.encoders {
            encoder.set_prediction_disabled(disabled);
        }
    }

    pub fn set_phase_inversion_disabled(&mut self, disabled: bool) {
        for encoder in &mut self.encoders {
            encoder.set_phase_inversion_disabled(disabled);
        }
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

    pub fn max_bandwidth(&self) -> Bandwidth {
        self.encoders
            .first()
            .map(OpusEncoder::max_bandwidth)
            .unwrap_or(Bandwidth::Fullband)
    }

    pub fn signal(&self) -> Option<Signal> {
        self.encoders.first().and_then(OpusEncoder::signal)
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

    pub fn lsb_depth(&self) -> i32 {
        self.encoders
            .first()
            .map(OpusEncoder::lsb_depth)
            .unwrap_or(0)
    }

    pub fn expert_frame_duration(&self) -> FrameSize {
        self.encoders
            .first()
            .map(OpusEncoder::expert_frame_duration)
            .unwrap_or(FrameSize::Arg)
    }

    pub fn prediction_disabled(&self) -> bool {
        self.encoders
            .first()
            .map(OpusEncoder::prediction_disabled)
            .unwrap_or(false)
    }

    pub fn phase_inversion_disabled(&self) -> bool {
        self.encoders
            .first()
            .map(OpusEncoder::phase_inversion_disabled)
            .unwrap_or(false)
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

    /// Encode interleaved 24-bit PCM (stored in i32) into a multistream Opus packet.
    pub fn encode24(&mut self, pcm: &[i32], output: &mut [u8]) -> i32 {
        self.encode_impl_i24(pcm, output)
    }

    fn stream_payload_budget(
        &self,
        frame_size: usize,
        stream_id: i32,
        output_len: usize,
        written: usize,
    ) -> Result<usize, i32> {
        let streams = self.layout().streams();
        let remaining_streams = streams - stream_id - 1;
        let mut curr_max = output_len as i32 - written as i32;

        // Reserve bytes for future stream packet headers.
        curr_max -= (2 * remaining_streams - 1).max(0);

        // 100ms packets reserve one extra ToC byte per remaining stream.
        if self.sample_rate() / frame_size as i32 == 10 {
            curr_max -= remaining_streams;
        }

        // Non-last streams need self-delimited length bytes.
        if stream_id != streams - 1 {
            curr_max -= if curr_max > 253 { 2 } else { 1 };
        }

        if curr_max <= 0 {
            Err(OPUS_BUFFER_TOO_SMALL)
        } else {
            Ok(curr_max as usize)
        }
    }

    fn target_bitrate_bps(&self, frame_size: usize) -> i32 {
        match self.requested_bitrate {
            Bitrate::Bits(bits) => bits,
            Bitrate::Max => {
                let nb_normal = self.layout().streams() + self.layout().coupled_streams();
                nb_normal * 750_000
            }
            Bitrate::Auto => {
                let frame_size = frame_size as i32;
                let fs = self.sample_rate();
                let nb_normal = self.layout().streams() + self.layout().coupled_streams();
                let channel_offset = 40 * (fs / frame_size).max(50);
                nb_normal * (channel_offset + fs + 10_000)
            }
        }
    }

    fn stream_target_bitrates(&self, frame_size: usize) -> Vec<i32> {
        let fs = self.sample_rate();
        let streams = self.layout().streams();
        let coupled = self.layout().coupled_streams();
        let uncoupled = streams - coupled;
        let normal_channels = streams + coupled;
        let bitrate = self.target_bitrate_bps(frame_size);

        let channel_offset = 40 * (fs / frame_size as i32).max(50);
        let mut stream_offset = if normal_channels > 0 {
            (bitrate - channel_offset * normal_channels) / normal_channels / 2
        } else {
            0
        };
        stream_offset = stream_offset.clamp(0, 20_000);

        let total = (uncoupled << 8) + 512 * coupled;
        let channel_rate = if total > 0 {
            (256 * (bitrate - stream_offset * streams - channel_offset * normal_channels)) / total
        } else {
            0
        };

        let mut rates = Vec::with_capacity(streams as usize);
        for stream_id in 0..streams {
            let rate = if stream_id < coupled {
                2 * channel_offset + (stream_offset + ((channel_rate * 512) >> 8)).max(0)
            } else {
                channel_offset + (stream_offset + channel_rate).max(0)
            };
            rates.push(rate.max(500));
        }
        rates
    }

    fn apply_stream_target_bitrates(&mut self, frame_size: usize) {
        let rates = self.stream_target_bitrates(frame_size);
        for (encoder, rate) in self.encoders.iter_mut().zip(rates.into_iter()) {
            encoder.set_bitrate(Bitrate::Bits(rate));
        }
    }

    fn encode_impl_i16(&mut self, pcm: &[i16], output: &mut [u8]) -> i32 {
        let channels = self.layout().channels() as usize;
        if channels == 0 || pcm.is_empty() || !pcm.len().is_multiple_of(channels) {
            return OPUS_BAD_ARG;
        }
        let frame_size = pcm.len() / channels;
        let mut write_offset = 0usize;
        let streams = self.layout().streams();
        let vbr_enabled = self.vbr();
        let sample_rate = self.sample_rate();
        self.apply_stream_target_bitrates(frame_size);
        let mut max_data_bytes = output.len();
        let mut smallest_packet = (streams * 2 - 1).max(0) as usize;
        if sample_rate / frame_size as i32 == 10 {
            smallest_packet += streams.max(0) as usize;
        }
        if max_data_bytes < smallest_packet {
            return OPUS_BUFFER_TOO_SMALL;
        }
        if !vbr_enabled {
            let bitrate = self.target_bitrate_bps(frame_size);
            let target = ((bitrate_to_bits(bitrate, sample_rate, frame_size as i32) + 4) / 8)
                .max(smallest_packet as i32) as usize;
            max_data_bytes = max_data_bytes.min(target);
        }
        for stream_id in 0..streams {
            let selected = match stream_input_channels(self.layout(), stream_id) {
                Some(v) => v,
                None => return OPUS_BAD_ARG,
            };
            let stream_pcm = extract_stream_pcm_i16(pcm, frame_size, channels, &selected);
            let curr_max = match self.stream_payload_budget(
                frame_size,
                stream_id,
                max_data_bytes,
                write_offset,
            ) {
                Ok(v) => v,
                Err(err) => return err,
            };
            let encoder = &mut self.encoders[stream_id as usize];
            if !vbr_enabled && stream_id == streams - 1 {
                encoder.set_bitrate(Bitrate::Bits(bits_to_bitrate(
                    (curr_max as i32) * 8,
                    sample_rate,
                    frame_size as i32,
                )));
            }
            let mut stream_packet = vec![0u8; curr_max];
            let len = encoder.encode(&stream_pcm, &mut stream_packet);
            if len < 0 {
                return len;
            }
            stream_packet.truncate(len as usize);
            let stream_packet = if stream_id != streams - 1 {
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
        let streams = self.layout().streams();
        let vbr_enabled = self.vbr();
        let sample_rate = self.sample_rate();
        self.apply_stream_target_bitrates(frame_size);
        let mut max_data_bytes = output.len();
        let mut smallest_packet = (streams * 2 - 1).max(0) as usize;
        if sample_rate / frame_size as i32 == 10 {
            smallest_packet += streams.max(0) as usize;
        }
        if max_data_bytes < smallest_packet {
            return OPUS_BUFFER_TOO_SMALL;
        }
        if !vbr_enabled {
            let bitrate = self.target_bitrate_bps(frame_size);
            let target = ((bitrate_to_bits(bitrate, sample_rate, frame_size as i32) + 4) / 8)
                .max(smallest_packet as i32) as usize;
            max_data_bytes = max_data_bytes.min(target);
        }
        for stream_id in 0..streams {
            let selected = match stream_input_channels(self.layout(), stream_id) {
                Some(v) => v,
                None => return OPUS_BAD_ARG,
            };
            let stream_pcm = extract_stream_pcm_f32(pcm, frame_size, channels, &selected);
            let curr_max = match self.stream_payload_budget(
                frame_size,
                stream_id,
                max_data_bytes,
                write_offset,
            ) {
                Ok(v) => v,
                Err(err) => return err,
            };
            let encoder = &mut self.encoders[stream_id as usize];
            if !vbr_enabled && stream_id == streams - 1 {
                encoder.set_bitrate(Bitrate::Bits(bits_to_bitrate(
                    (curr_max as i32) * 8,
                    sample_rate,
                    frame_size as i32,
                )));
            }
            let mut stream_packet = vec![0u8; curr_max];
            let len = encoder.encode_float(&stream_pcm, &mut stream_packet);
            if len < 0 {
                return len;
            }
            stream_packet.truncate(len as usize);
            let stream_packet = if stream_id != streams - 1 {
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

    fn encode_impl_i24(&mut self, pcm: &[i32], output: &mut [u8]) -> i32 {
        let channels = self.layout().channels() as usize;
        if channels == 0 || pcm.is_empty() || !pcm.len().is_multiple_of(channels) {
            return OPUS_BAD_ARG;
        }
        let frame_size = pcm.len() / channels;
        let mut write_offset = 0usize;
        let streams = self.layout().streams();
        let vbr_enabled = self.vbr();
        let sample_rate = self.sample_rate();
        self.apply_stream_target_bitrates(frame_size);
        let mut max_data_bytes = output.len();
        let mut smallest_packet = (streams * 2 - 1).max(0) as usize;
        if sample_rate / frame_size as i32 == 10 {
            smallest_packet += streams.max(0) as usize;
        }
        if max_data_bytes < smallest_packet {
            return OPUS_BUFFER_TOO_SMALL;
        }
        if !vbr_enabled {
            let bitrate = self.target_bitrate_bps(frame_size);
            let target = ((bitrate_to_bits(bitrate, sample_rate, frame_size as i32) + 4) / 8)
                .max(smallest_packet as i32) as usize;
            max_data_bytes = max_data_bytes.min(target);
        }
        for stream_id in 0..streams {
            let selected = match stream_input_channels(self.layout(), stream_id) {
                Some(v) => v,
                None => return OPUS_BAD_ARG,
            };
            let stream_pcm = extract_stream_pcm_i32(pcm, frame_size, channels, &selected);
            let curr_max = match self.stream_payload_budget(
                frame_size,
                stream_id,
                max_data_bytes,
                write_offset,
            ) {
                Ok(v) => v,
                Err(err) => return err,
            };
            let encoder = &mut self.encoders[stream_id as usize];
            if !vbr_enabled && stream_id == streams - 1 {
                encoder.set_bitrate(Bitrate::Bits(bits_to_bitrate(
                    (curr_max as i32) * 8,
                    sample_rate,
                    frame_size as i32,
                )));
            }
            let mut stream_packet = vec![0u8; curr_max];
            let len = encoder.encode24(&stream_pcm, &mut stream_packet);
            if len < 0 {
                return len;
            }
            stream_packet.truncate(len as usize);
            let stream_packet = if stream_id != streams - 1 {
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
pub fn opus_multistream_surround_encoder_get_size(channels: i32, mapping_family: i32) -> i32 {
    let mut streams = 0i32;
    let mut coupled_streams = 0i32;
    let mut mapping = vec![0u8; channels.max(0) as usize];
    if surround_layout(
        channels,
        mapping_family,
        &mut streams,
        &mut coupled_streams,
        &mut mapping,
    )
    .is_ok()
    {
        OpusMSEncoder::get_size(streams, coupled_streams)
    } else {
        0
    }
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
pub fn opus_multistream_surround_encoder_create(
    sample_rate: i32,
    channels: i32,
    mapping_family: i32,
    streams: &mut i32,
    coupled_streams: &mut i32,
    mapping: &mut [u8],
    application: i32,
) -> Result<OpusMSEncoder, i32> {
    surround_layout(channels, mapping_family, streams, coupled_streams, mapping)?;
    OpusMSEncoder::new(
        sample_rate,
        channels,
        *streams,
        *coupled_streams,
        &mapping[..channels as usize],
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
pub fn opus_multistream_surround_encoder_init(
    st: &mut OpusMSEncoder,
    sample_rate: i32,
    channels: i32,
    mapping_family: i32,
    streams: &mut i32,
    coupled_streams: &mut i32,
    mapping: &mut [u8],
    application: i32,
) -> i32 {
    match surround_layout(channels, mapping_family, streams, coupled_streams, mapping) {
        Ok(()) => st.init(
            sample_rate,
            channels,
            *streams,
            *coupled_streams,
            &mapping[..channels as usize],
            application,
        ),
        Err(err) => err,
    }
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

/// Upstream-style free function wrapper.
pub fn opus_multistream_encode24(
    st: &mut OpusMSEncoder,
    pcm: &[i32],
    frame_size: i32,
    data: &mut [u8],
) -> i32 {
    let channels = st.layout().channels() as usize;
    if frame_size <= 0 || pcm.len() != frame_size as usize * channels {
        return OPUS_BAD_ARG;
    }
    st.encode24(pcm, data)
}

/// Upstream-style helper for `OPUS_MULTISTREAM_GET_ENCODER_STATE_REQUEST`.
pub fn opus_multistream_encoder_get_encoder_state(
    st: &mut OpusMSEncoder,
    stream_id: i32,
) -> Result<&mut OpusEncoder, i32> {
    st.encoder_state_mut(stream_id)
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

#[inline]
fn bits_to_bitrate(bits: i32, fs: i32, frame_size: i32) -> i32 {
    ((bits as i64 * fs as i64) / frame_size as i64) as i32
}

#[inline]
fn bitrate_to_bits(bitrate: i32, fs: i32, frame_size: i32) -> i32 {
    ((bitrate as i64 * frame_size as i64) / fs as i64) as i32
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

fn extract_stream_pcm_i32(
    pcm: &[i32],
    frame_size: usize,
    channels: usize,
    selected_channels: &[usize],
) -> Vec<i32> {
    let stream_channels = selected_channels.len();
    let mut out = vec![0i32; frame_size * stream_channels];
    for frame in 0..frame_size {
        for (idx, &channel) in selected_channels.iter().enumerate() {
            out[frame * stream_channels + idx] = pcm[frame * channels + channel];
        }
    }
    out
}

fn surround_layout(
    channels: i32,
    mapping_family: i32,
    streams: &mut i32,
    coupled_streams: &mut i32,
    mapping: &mut [u8],
) -> Result<(), i32> {
    if !(1..=255).contains(&channels) || mapping.len() < channels as usize {
        return Err(OPUS_BAD_ARG);
    }

    match mapping_family {
        0 => match channels {
            1 => {
                *streams = 1;
                *coupled_streams = 0;
                mapping[0] = 0;
                Ok(())
            }
            2 => {
                *streams = 1;
                *coupled_streams = 1;
                mapping[0] = 0;
                mapping[1] = 1;
                Ok(())
            }
            _ => Err(OPUS_UNIMPLEMENTED),
        },
        1 if channels <= 8 => {
            let layout = VORBIS_MAPPINGS[(channels - 1) as usize];
            *streams = layout.streams;
            *coupled_streams = layout.coupled_streams;
            mapping[..channels as usize].copy_from_slice(&layout.mapping[..channels as usize]);
            Ok(())
        }
        255 => {
            *streams = channels;
            *coupled_streams = 0;
            for (idx, v) in mapping[..channels as usize].iter_mut().enumerate() {
                *v = idx as u8;
            }
            Ok(())
        }
        2 => {
            let (s, c) = validate_ambisonics(channels).ok_or(OPUS_BAD_ARG)?;
            *streams = s;
            *coupled_streams = c;

            let mono_streams = (*streams - *coupled_streams) as usize;
            let coupled_channels = (*coupled_streams * 2) as u8;
            for (idx, slot) in mapping.iter_mut().take(mono_streams).enumerate() {
                *slot = coupled_channels + idx as u8;
            }
            for (idx, slot) in mapping
                .iter_mut()
                .skip(mono_streams)
                .take(coupled_channels as usize)
                .enumerate()
            {
                *slot = idx as u8;
            }
            Ok(())
        }
        _ => Err(OPUS_UNIMPLEMENTED),
    }
}

fn validate_ambisonics(channels: i32) -> Option<(i32, i32)> {
    if !(1..=227).contains(&channels) {
        return None;
    }

    let mut order_plus_one = 0i32;
    while (order_plus_one + 1) * (order_plus_one + 1) <= channels {
        order_plus_one += 1;
    }
    let acn_channels = order_plus_one * order_plus_one;
    let nondiegetic_channels = channels - acn_channels;
    if nondiegetic_channels != 0 && nondiegetic_channels != 2 {
        return None;
    }

    let coupled_streams = (nondiegetic_channels != 0) as i32;
    let streams = acn_channels + coupled_streams;
    Some((streams, coupled_streams))
}
