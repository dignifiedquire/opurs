//! Projection Opus encoder wrapper.
//!
//! Upstream C: `src/opus_projection_encoder.c`

use crate::enums::Application;
use crate::enums::{Bandwidth, Bitrate, Channels, FrameSize, Signal};
use crate::opus::analysis::DownmixInput;
use crate::opus::mapping_matrix::MappingMatrix;
use crate::opus::opus_defines::{OPUS_BAD_ARG, OPUS_OK};
use crate::opus::opus_encoder::OpusEncoder;
use crate::opus::opus_multistream_encoder::OpusMSEncoder;
use crate::opus::projection_matrices::projection_matrices_for_order_plus_one;

/// Projection Opus encoder state.
///
/// Holds the ambisonics projection matrices and an internal multistream
/// encoder used for coded stream generation.
///
/// Upstream C: include/opus_projection.h:OpusProjectionEncoder
#[derive(Clone)]
pub struct OpusProjectionEncoder {
    channels: i32,
    streams: i32,
    coupled_streams: i32,
    mixing_matrix: MappingMatrix,
    demixing_matrix: MappingMatrix,
    encoder: OpusMSEncoder,
}

fn get_order_plus_one_from_channels(channels: i32) -> Result<i32, i32> {
    // Allowed: (1+n)^2 + 2j where n in [0,14], j in {0,1}.
    if !(1..=227).contains(&channels) {
        return Err(OPUS_BAD_ARG);
    }
    let mut order_plus_one = 0i32;
    while (order_plus_one + 1) * (order_plus_one + 1) <= channels {
        order_plus_one += 1;
    }
    let acn_channels = order_plus_one * order_plus_one;
    let nondiegetic_channels = channels - acn_channels;
    if nondiegetic_channels != 0 && nondiegetic_channels != 2 {
        return Err(OPUS_BAD_ARG);
    }
    Ok(order_plus_one)
}

fn get_streams_from_channels(channels: i32, mapping_family: i32) -> Result<(i32, i32, i32), i32> {
    if mapping_family != 3 {
        return Err(OPUS_BAD_ARG);
    }
    let order_plus_one = get_order_plus_one_from_channels(channels)?;
    let streams = (channels + 1) / 2;
    let coupled_streams = channels / 2;
    Ok((streams, coupled_streams, order_plus_one))
}

impl OpusProjectionEncoder {
    /// Upstream-style sizing helper.
    ///
    /// Returns zero for invalid `(channels, mapping_family)` combinations.
    ///
    /// Upstream C: include/opus_projection.h:opus_projection_ambisonics_encoder_get_size
    pub fn get_size(channels: i32, mapping_family: i32) -> i32 {
        let Ok((streams, coupled_streams, order_plus_one)) =
            get_streams_from_channels(channels, mapping_family)
        else {
            return 0;
        };
        let Some((mixing_matrix, demixing_matrix)) =
            projection_matrices_for_order_plus_one(order_plus_one)
        else {
            return 0;
        };
        let input_channels = streams + coupled_streams;
        if input_channels > mixing_matrix.rows
            || channels > mixing_matrix.cols
            || channels > demixing_matrix.rows
            || input_channels > demixing_matrix.cols
        {
            return 0;
        }
        if OpusMSEncoder::get_size(streams, coupled_streams) == 0 {
            return 0;
        }
        core::mem::size_of::<Self>() as i32
    }

    /// Create and initialize a projection encoder.
    ///
    /// Computes the ambisonics stream layout and projection matrices for
    /// `mapping_family`.
    ///
    /// Upstream C: include/opus_projection.h:opus_projection_ambisonics_encoder_create
    pub fn new(
        sample_rate: i32,
        channels: i32,
        mapping_family: i32,
        streams: &mut i32,
        coupled_streams: &mut i32,
        application: i32,
    ) -> Result<Self, i32> {
        let (s, c, order_plus_one) = get_streams_from_channels(channels, mapping_family)?;
        *streams = s;
        *coupled_streams = c;

        let Some((mixing_matrix_def, demixing_matrix_def)) =
            projection_matrices_for_order_plus_one(order_plus_one)
        else {
            return Err(OPUS_BAD_ARG);
        };
        let mixing_matrix = MappingMatrix::new(
            mixing_matrix_def.rows,
            mixing_matrix_def.cols,
            mixing_matrix_def.gain,
            mixing_matrix_def.data,
        )?;
        let demixing_matrix = MappingMatrix::new(
            demixing_matrix_def.rows,
            demixing_matrix_def.cols,
            demixing_matrix_def.gain,
            demixing_matrix_def.data,
        )?;

        let input_channels = s + c;
        if input_channels > mixing_matrix.rows() as i32
            || channels > mixing_matrix.cols() as i32
            || channels > demixing_matrix.rows() as i32
            || input_channels > demixing_matrix.cols() as i32
        {
            return Err(OPUS_BAD_ARG);
        }

        let app = Application::try_from(application).map_err(|_| OPUS_BAD_ARG)?;
        let _ = app; // validation only
        let mapping = (0..channels).map(|idx| idx as u8).collect::<Vec<_>>();
        let encoder = OpusMSEncoder::new(sample_rate, channels, s, c, &mapping, application)?;

        Ok(Self {
            channels,
            streams: s,
            coupled_streams: c,
            mixing_matrix,
            demixing_matrix,
            encoder,
        })
    }

    /// Reinitialize an existing projection encoder instance.
    ///
    /// Upstream C: include/opus_projection.h:opus_projection_ambisonics_encoder_init
    pub fn init(
        &mut self,
        sample_rate: i32,
        channels: i32,
        mapping_family: i32,
        streams: &mut i32,
        coupled_streams: &mut i32,
        application: i32,
    ) -> i32 {
        match Self::new(
            sample_rate,
            channels,
            mapping_family,
            streams,
            coupled_streams,
            application,
        ) {
            Ok(st) => {
                *self = st;
                OPUS_OK
            }
            Err(err) => err,
        }
    }

    /// Encode one projection frame from interleaved `i16` PCM.
    ///
    /// Upstream C: include/opus_projection.h:opus_projection_encode
    pub fn encode(&mut self, pcm: &[i16], frame_size: i32, data: &mut [u8]) -> i32 {
        if frame_size <= 0 {
            return OPUS_BAD_ARG;
        }
        let analysis_frame_size = frame_size;
        let frame_size = analysis_frame_size as usize;
        let channels = self.channels as usize;
        if pcm.len() != frame_size * channels || data.is_empty() {
            return OPUS_BAD_ARG;
        }
        let input_channels = (self.streams + self.coupled_streams) as usize;
        let mut mixed = vec![0f32; frame_size * input_channels];
        for row in 0..input_channels {
            if let Err(err) = self.mixing_matrix.multiply_channel_in_short(
                pcm,
                channels,
                &mut mixed[row..],
                row,
                input_channels,
                frame_size,
            ) {
                return err;
            }
        }
        let analysis_input = DownmixInput::Int(pcm);
        self.encoder.encode_float_with_analysis(
            &mixed,
            analysis_frame_size,
            16,
            0,
            Some(&analysis_input),
            data,
        )
    }

    /// Encode one projection frame from interleaved `f32` PCM.
    ///
    /// Upstream C: include/opus_projection.h:opus_projection_encode_float
    pub fn encode_float(&mut self, pcm: &[f32], frame_size: i32, data: &mut [u8]) -> i32 {
        if frame_size <= 0 {
            return OPUS_BAD_ARG;
        }
        let analysis_frame_size = frame_size;
        let frame_size = analysis_frame_size as usize;
        let channels = self.channels as usize;
        if pcm.len() != frame_size * channels || data.is_empty() {
            return OPUS_BAD_ARG;
        }
        let input_channels = (self.streams + self.coupled_streams) as usize;
        let mut mixed = vec![0f32; frame_size * input_channels];
        for row in 0..input_channels {
            if let Err(err) = self.mixing_matrix.multiply_channel_in_float(
                pcm,
                channels,
                &mut mixed[row..],
                row,
                input_channels,
                frame_size,
            ) {
                return err;
            }
        }
        let analysis_input = DownmixInput::Float(pcm);
        self.encoder.encode_float_with_analysis(
            &mixed,
            analysis_frame_size,
            24,
            1,
            Some(&analysis_input),
            data,
        )
    }

    /// Encode one projection frame from interleaved 24-bit (`i32`) PCM.
    ///
    /// Upstream C: include/opus_projection.h:opus_projection_encode24
    pub fn encode24(&mut self, pcm: &[i32], frame_size: i32, data: &mut [u8]) -> i32 {
        if frame_size <= 0 {
            return OPUS_BAD_ARG;
        }
        let analysis_frame_size = frame_size;
        let frame_size = analysis_frame_size as usize;
        let channels = self.channels as usize;
        if pcm.len() != frame_size * channels || data.is_empty() {
            return OPUS_BAD_ARG;
        }
        let input_channels = (self.streams + self.coupled_streams) as usize;
        let mut mixed = vec![0f32; frame_size * input_channels];
        for row in 0..input_channels {
            if let Err(err) = self.mixing_matrix.multiply_channel_in_int24(
                pcm,
                channels,
                &mut mixed[row..],
                row,
                input_channels,
                frame_size,
            ) {
                return err;
            }
        }
        let analysis_input = DownmixInput::Int24(pcm);
        self.encoder.encode_float_with_analysis(
            &mixed,
            analysis_frame_size,
            24,
            0,
            Some(&analysis_input),
            data,
        )
    }

    /// Return demixing matrix global gain.
    ///
    /// Upstream C: include/opus_projection.h:opus_projection_encoder_ctl
    pub fn demixing_matrix_gain(&self) -> i32 {
        self.demixing_matrix.gain()
    }

    /// Return serialized demixing matrix size in bytes.
    ///
    /// Upstream C: include/opus_projection.h:opus_projection_encoder_ctl
    pub fn demixing_matrix_size(&self) -> i32 {
        (self.channels * (self.streams + self.coupled_streams) * 2).max(0)
    }

    /// Copy serialized demixing matrix bytes into `dst`.
    ///
    /// `dst.len()` must equal [`Self::demixing_matrix_size`].
    ///
    /// Upstream C: include/opus_projection.h:opus_projection_encoder_ctl
    pub fn copy_demixing_matrix(&self, dst: &mut [u8]) -> Result<(), i32> {
        let expected = self.demixing_matrix_size() as usize;
        if dst.len() != expected {
            return Err(OPUS_BAD_ARG);
        }

        let rows = self.demixing_matrix.rows();
        let data = self.demixing_matrix.data();
        let input_channels = (self.streams + self.coupled_streams) as usize;
        let output_channels = self.channels as usize;
        let mut l = 0usize;
        for i in 0..input_channels {
            for j in 0..output_channels {
                let k = rows * i + j;
                let value = data[k];
                dst[2 * l] = value as u8;
                dst[2 * l + 1] = (value >> 8) as u8;
                l += 1;
            }
        }
        Ok(())
    }

    /// Set encoder application mode.
    ///
    /// Upstream C: include/opus_projection.h:opus_projection_encoder_ctl
    pub fn set_application(&mut self, application: i32) -> Result<(), i32> {
        self.encoder.set_application(application)
    }

    /// Set target bitrate.
    ///
    /// Upstream C: include/opus_projection.h:opus_projection_encoder_ctl
    pub fn set_bitrate(&mut self, bitrate: Bitrate) {
        self.encoder.set_bitrate(bitrate);
    }

    /// Set encoder complexity.
    ///
    /// Upstream C: include/opus_projection.h:opus_projection_encoder_ctl
    pub fn set_complexity(&mut self, complexity: i32) -> Result<(), i32> {
        self.encoder.set_complexity(complexity)
    }

    /// Enable or disable VBR.
    ///
    /// Upstream C: include/opus_projection.h:opus_projection_encoder_ctl
    pub fn set_vbr(&mut self, enabled: bool) {
        self.encoder.set_vbr(enabled);
    }

    /// Enable or disable constrained VBR.
    ///
    /// Upstream C: include/opus_projection.h:opus_projection_encoder_ctl
    pub fn set_vbr_constraint(&mut self, enabled: bool) {
        self.encoder.set_vbr_constraint(enabled);
    }

    /// Set requested target bandwidth.
    ///
    /// Upstream C: include/opus_projection.h:opus_projection_encoder_ctl
    pub fn set_bandwidth(&mut self, bandwidth: Option<Bandwidth>) {
        self.encoder.set_bandwidth(bandwidth);
    }

    /// Set maximum allowed encoder bandwidth.
    ///
    /// Upstream C: include/opus_projection.h:opus_projection_encoder_ctl
    pub fn set_max_bandwidth(&mut self, bandwidth: Bandwidth) {
        self.encoder.set_max_bandwidth(bandwidth);
    }

    /// Hint signal type (voice/music/auto).
    ///
    /// Upstream C: include/opus_projection.h:opus_projection_encoder_ctl
    pub fn set_signal(&mut self, signal: Option<Signal>) {
        self.encoder.set_signal(signal);
    }

    /// Enable or disable in-band FEC.
    ///
    /// Upstream C: include/opus_projection.h:opus_projection_encoder_ctl
    pub fn set_inband_fec(&mut self, value: i32) -> Result<(), i32> {
        self.encoder.set_inband_fec(value)
    }

    /// Set expected packet loss percentage.
    ///
    /// Upstream C: include/opus_projection.h:opus_projection_encoder_ctl
    pub fn set_packet_loss_perc(&mut self, pct: i32) -> Result<(), i32> {
        self.encoder.set_packet_loss_perc(pct)
    }

    /// Enable or disable discontinuous transmission (DTX).
    ///
    /// Upstream C: include/opus_projection.h:opus_projection_encoder_ctl
    pub fn set_dtx(&mut self, enabled: bool) {
        self.encoder.set_dtx(enabled);
    }

    /// Set forced mono/stereo behavior.
    ///
    /// Upstream C: include/opus_projection.h:opus_projection_encoder_ctl
    pub fn set_force_channels(&mut self, channels: Option<Channels>) -> Result<(), i32> {
        self.encoder.set_force_channels(channels)
    }

    /// Set input PCM bit depth hint.
    ///
    /// Upstream C: include/opus_projection.h:opus_projection_encoder_ctl
    pub fn set_lsb_depth(&mut self, depth: i32) -> Result<(), i32> {
        self.encoder.set_lsb_depth(depth)
    }

    /// Set expert frame duration request.
    ///
    /// Upstream C: include/opus_projection.h:opus_projection_encoder_ctl
    pub fn set_expert_frame_duration(&mut self, fs: FrameSize) {
        self.encoder.set_expert_frame_duration(fs);
    }

    /// Enable or disable inter-frame prediction.
    ///
    /// Upstream C: include/opus_projection.h:opus_projection_encoder_ctl
    pub fn set_prediction_disabled(&mut self, disabled: bool) {
        self.encoder.set_prediction_disabled(disabled);
    }

    /// Enable or disable phase inversion in intensity stereo.
    ///
    /// Upstream C: include/opus_projection.h:opus_projection_encoder_ctl
    pub fn set_phase_inversion_disabled(&mut self, disabled: bool) {
        self.encoder.set_phase_inversion_disabled(disabled);
    }

    /// Enable or disable QEXT encoding.
    ///
    /// Upstream C: include/opus_projection.h:opus_projection_encoder_ctl
    #[cfg(feature = "qext")]
    pub fn set_qext(&mut self, enabled: bool) {
        self.encoder.set_qext(enabled);
    }

    /// Reset encoder state.
    ///
    /// Upstream C: include/opus_projection.h:opus_projection_encoder_ctl
    pub fn reset(&mut self) {
        self.encoder.reset();
    }

    /// Borrow a child stream encoder by stream index.
    ///
    /// Upstream C: include/opus_projection.h:opus_projection_encoder_ctl
    pub fn encoder_state(&self, stream_id: i32) -> Result<&OpusEncoder, i32> {
        self.encoder.encoder_state(stream_id)
    }

    /// Mutably borrow a child stream encoder by stream index.
    ///
    /// Upstream C: include/opus_projection.h:opus_projection_encoder_ctl
    pub fn encoder_state_mut(&mut self, stream_id: i32) -> Result<&mut OpusEncoder, i32> {
        self.encoder.encoder_state_mut(stream_id)
    }

    /// Return application mode.
    ///
    /// Upstream C: include/opus_projection.h:opus_projection_encoder_ctl
    pub fn application(&self) -> i32 {
        self.encoder.application()
    }

    /// Return effective target bitrate.
    ///
    /// Upstream C: include/opus_projection.h:opus_projection_encoder_ctl
    pub fn bitrate(&self) -> i32 {
        self.encoder.bitrate()
    }

    /// Return complexity setting.
    ///
    /// Upstream C: include/opus_projection.h:opus_projection_encoder_ctl
    pub fn complexity(&self) -> i32 {
        self.encoder.complexity()
    }

    /// Return whether VBR is enabled.
    ///
    /// Upstream C: include/opus_projection.h:opus_projection_encoder_ctl
    pub fn vbr(&self) -> bool {
        self.encoder.vbr()
    }

    /// Return whether constrained VBR is enabled.
    ///
    /// Upstream C: include/opus_projection.h:opus_projection_encoder_ctl
    pub fn vbr_constraint(&self) -> bool {
        self.encoder.vbr_constraint()
    }

    /// Return currently selected output bandwidth.
    ///
    /// Upstream C: include/opus_projection.h:opus_projection_encoder_ctl
    pub fn bandwidth(&self) -> i32 {
        self.encoder.bandwidth()
    }

    /// Return configured maximum bandwidth.
    ///
    /// Upstream C: include/opus_projection.h:opus_projection_encoder_ctl
    pub fn max_bandwidth(&self) -> Bandwidth {
        self.encoder.max_bandwidth()
    }

    /// Return configured signal hint.
    ///
    /// Upstream C: include/opus_projection.h:opus_projection_encoder_ctl
    pub fn signal(&self) -> Option<Signal> {
        self.encoder.signal()
    }

    /// Return in-band FEC setting.
    ///
    /// Upstream C: include/opus_projection.h:opus_projection_encoder_ctl
    pub fn inband_fec(&self) -> i32 {
        self.encoder.inband_fec()
    }

    /// Return expected packet loss percentage setting.
    ///
    /// Upstream C: include/opus_projection.h:opus_projection_encoder_ctl
    pub fn packet_loss_perc(&self) -> i32 {
        self.encoder.packet_loss_perc()
    }

    /// Return whether DTX is enabled.
    ///
    /// Upstream C: include/opus_projection.h:opus_projection_encoder_ctl
    pub fn dtx(&self) -> bool {
        self.encoder.dtx()
    }

    /// Return forced channel setting.
    ///
    /// Upstream C: include/opus_projection.h:opus_projection_encoder_ctl
    pub fn force_channels(&self) -> Option<Channels> {
        self.encoder.force_channels()
    }

    /// Return input bit depth hint.
    ///
    /// Upstream C: include/opus_projection.h:opus_projection_encoder_ctl
    pub fn lsb_depth(&self) -> i32 {
        self.encoder.lsb_depth()
    }

    /// Return expert frame duration setting.
    ///
    /// Upstream C: include/opus_projection.h:opus_projection_encoder_ctl
    pub fn expert_frame_duration(&self) -> FrameSize {
        self.encoder.expert_frame_duration()
    }

    /// Return whether inter-frame prediction is disabled.
    ///
    /// Upstream C: include/opus_projection.h:opus_projection_encoder_ctl
    pub fn prediction_disabled(&self) -> bool {
        self.encoder.prediction_disabled()
    }

    /// Return whether phase inversion is disabled.
    ///
    /// Upstream C: include/opus_projection.h:opus_projection_encoder_ctl
    pub fn phase_inversion_disabled(&self) -> bool {
        self.encoder.phase_inversion_disabled()
    }

    /// Return whether QEXT is enabled.
    ///
    /// Upstream C: include/opus_projection.h:opus_projection_encoder_ctl
    #[cfg(feature = "qext")]
    pub fn qext(&self) -> bool {
        self.encoder.qext()
    }

    /// Return total encoder lookahead in samples.
    ///
    /// Upstream C: include/opus_projection.h:opus_projection_encoder_ctl
    pub fn lookahead(&self) -> i32 {
        self.encoder.lookahead()
    }

    /// Return final range value from the most recent encoded packet.
    ///
    /// Upstream C: include/opus_projection.h:opus_projection_encoder_ctl
    pub fn final_range(&self) -> u32 {
        self.encoder.final_range()
    }
}
