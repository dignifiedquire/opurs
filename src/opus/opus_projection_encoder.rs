//! Projection Opus encoder wrapper.
//!
//! Upstream C: `src/opus_projection_encoder.c`

use crate::enums::Application;
use crate::opus::mapping_matrix::MappingMatrix;
use crate::opus::opus_defines::{OPUS_BAD_ARG, OPUS_BUFFER_TOO_SMALL, OPUS_OK, OPUS_UNIMPLEMENTED};
use crate::opus::opus_multistream_encoder::OpusMSEncoder;

const FOA_MIXING_DATA: [i16; 36] = [
    16384, 0, -16384, 23170, 0, 0, 16384, 23170, 16384, 0, 0, 0, 16384, 0, -16384, -23170, 0, 0,
    16384, -23170, 16384, 0, 0, 0, 0, 0, 0, 0, 32767, 0, 0, 0, 0, 0, 0, 32767,
];

const FOA_DEMIXING_DATA: [i16; 36] = [
    16384, 16384, 16384, 16384, 0, 0, 0, 23170, 0, -23170, 0, 0, -16384, 16384, -16384, 16384, 0,
    0, 23170, 0, -23170, 0, 0, 0, 0, 0, 0, 0, 32767, 0, 0, 0, 0, 0, 0, 32767,
];

/// Pure-Rust projection encoder.
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
    /// For now this wrapper supports FOA (order+1=2) only.
    pub fn get_size(channels: i32, mapping_family: i32) -> i32 {
        let Ok((streams, coupled_streams, order_plus_one)) =
            get_streams_from_channels(channels, mapping_family)
        else {
            return 0;
        };
        if order_plus_one != 2 {
            return 0;
        }
        if OpusMSEncoder::get_size(streams, coupled_streams) == 0 {
            return 0;
        }
        core::mem::size_of::<Self>() as i32
    }

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

        if order_plus_one != 2 {
            return Err(OPUS_UNIMPLEMENTED);
        }

        let mixing_matrix = MappingMatrix::new(6, 6, 0, &FOA_MIXING_DATA)?;
        let demixing_matrix = MappingMatrix::new(6, 6, 0, &FOA_DEMIXING_DATA)?;

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

    pub fn encode(&mut self, pcm: &[i16], frame_size: i32, data: &mut [u8]) -> i32 {
        if frame_size <= 0 {
            return OPUS_BAD_ARG;
        }
        let frame_size = frame_size as usize;
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
        let ret = self.encoder.encode_float(&mixed, data);
        if ret > 0 {
            ret
        } else {
            OPUS_BUFFER_TOO_SMALL
        }
    }

    pub fn encode_float(&mut self, pcm: &[f32], frame_size: i32, data: &mut [u8]) -> i32 {
        if frame_size <= 0 {
            return OPUS_BAD_ARG;
        }
        let frame_size = frame_size as usize;
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
        let ret = self.encoder.encode_float(&mixed, data);
        if ret > 0 {
            ret
        } else {
            OPUS_BUFFER_TOO_SMALL
        }
    }

    pub fn encode24(&mut self, pcm: &[i32], frame_size: i32, data: &mut [u8]) -> i32 {
        if frame_size <= 0 {
            return OPUS_BAD_ARG;
        }
        let frame_size = frame_size as usize;
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
        let ret = self.encoder.encode_float(&mixed, data);
        if ret > 0 {
            ret
        } else {
            OPUS_BUFFER_TOO_SMALL
        }
    }

    pub fn demixing_matrix_gain(&self) -> i32 {
        self.demixing_matrix.gain()
    }

    pub fn demixing_matrix_size(&self) -> i32 {
        (self.channels * (self.streams + self.coupled_streams) * 2).max(0)
    }

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
}

/// Upstream-style free function wrapper.
pub fn opus_projection_ambisonics_encoder_get_size(channels: i32, mapping_family: i32) -> i32 {
    OpusProjectionEncoder::get_size(channels, mapping_family)
}

/// Upstream-style free function wrapper.
pub fn opus_projection_ambisonics_encoder_create(
    sample_rate: i32,
    channels: i32,
    mapping_family: i32,
    streams: &mut i32,
    coupled_streams: &mut i32,
    application: i32,
) -> Result<OpusProjectionEncoder, i32> {
    OpusProjectionEncoder::new(
        sample_rate,
        channels,
        mapping_family,
        streams,
        coupled_streams,
        application,
    )
}

/// Upstream-style free function wrapper.
pub fn opus_projection_ambisonics_encoder_init(
    st: &mut OpusProjectionEncoder,
    sample_rate: i32,
    channels: i32,
    mapping_family: i32,
    streams: &mut i32,
    coupled_streams: &mut i32,
    application: i32,
) -> i32 {
    st.init(
        sample_rate,
        channels,
        mapping_family,
        streams,
        coupled_streams,
        application,
    )
}

/// Upstream-style free function wrapper.
pub fn opus_projection_encode(
    st: &mut OpusProjectionEncoder,
    pcm: &[i16],
    frame_size: i32,
    data: &mut [u8],
) -> i32 {
    st.encode(pcm, frame_size, data)
}

/// Upstream-style free function wrapper.
pub fn opus_projection_encode_float(
    st: &mut OpusProjectionEncoder,
    pcm: &[f32],
    frame_size: i32,
    data: &mut [u8],
) -> i32 {
    st.encode_float(pcm, frame_size, data)
}

/// Upstream-style free function wrapper.
pub fn opus_projection_encode24(
    st: &mut OpusProjectionEncoder,
    pcm: &[i32],
    frame_size: i32,
    data: &mut [u8],
) -> i32 {
    st.encode24(pcm, frame_size, data)
}

/// Upstream-style free function wrapper.
pub fn opus_projection_encoder_destroy(_st: OpusProjectionEncoder) {}
