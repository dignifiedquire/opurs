//! Opus decoder.
//!
//! Upstream C: `src/opus_decoder.c`

pub mod arch_h {
    pub type opus_val16 = f32;
    pub type opus_val32 = f32;
}
pub mod stddef_h {}
pub mod stack_alloc_h {
    pub const ALLOC_NONE: i32 = 1;
    #[inline]
    pub fn _opus_false() -> i32 {
        0
    }
}
pub use self::arch_h::{opus_val16, opus_val32};
pub use self::stack_alloc_h::{_opus_false, ALLOC_NONE};

use crate::arch::opus_select_arch;
use crate::celt::celt_decoder::{celt_decode_with_ec, celt_decoder_init, OpusCustomDecoder};
use crate::celt::entcode::ec_tell;
use crate::celt::entdec::ec_dec;
use crate::celt::entdec::{ec_dec_bit_logp, ec_dec_init, ec_dec_uint};
use crate::celt::float_cast::{celt_float2int16, float2int};
use crate::celt::mathops::celt_exp2;
use crate::opus::opus_defines::{
    OPUS_BAD_ARG, OPUS_BANDWIDTH_FULLBAND, OPUS_BANDWIDTH_MEDIUMBAND, OPUS_BANDWIDTH_NARROWBAND,
    OPUS_BANDWIDTH_SUPERWIDEBAND, OPUS_BANDWIDTH_WIDEBAND, OPUS_BUFFER_TOO_SMALL,
    OPUS_INTERNAL_ERROR, OPUS_INVALID_PACKET,
};
#[cfg(feature = "dred")]
use crate::opus::opus_defines::{OPUS_OK, OPUS_UNIMPLEMENTED};
use crate::opus::opus_private::{MODE_CELT_ONLY, MODE_HYBRID, MODE_SILK_ONLY};
use crate::opus::packet::{opus_packet_parse, opus_packet_parse_impl, opus_pcm_soft_clip_impl};
use crate::opus_packet_get_samples_per_frame;
use crate::silk::dec_API::{silk_DecControlStruct, silk_decoder};
use crate::silk::dec_API::{silk_Decode, silk_InitDecoder, silk_ResetDecoder};
#[cfg(feature = "dred")]
use crate::{
    dnn::dred::{
        config::{
            DRED_EXPERIMENTAL_BYTES, DRED_EXPERIMENTAL_VERSION, DRED_EXTENSION_ID,
            DRED_NUM_FEATURES, DRED_NUM_REDUNDANCY_FRAMES,
        },
        decoder::{
            dred_ec_decode, opus_dred_process as dred_process_impl, OpusDRED, OpusDREDDecoder,
        },
    },
    dnn::lpcnet::{lpcnet_plc_fec_add, lpcnet_plc_fec_clear},
};

#[derive(Clone)]
#[repr(C)]
pub struct OpusDecoder {
    pub(crate) celt_dec: OpusCustomDecoder,
    pub(crate) silk_dec: silk_decoder,
    pub(crate) channels: i32,
    pub(crate) Fs: i32,
    pub(crate) DecControl: silk_DecControlStruct,
    pub(crate) decode_gain: i32,
    pub(crate) complexity: i32,
    #[cfg(feature = "deep-plc")]
    pub(crate) lpcnet: crate::dnn::lpcnet::LPCNetPLCState,
    pub(crate) stream_channels: i32,
    pub(crate) bandwidth: i32,
    pub(crate) mode: i32,
    pub(crate) prev_mode: i32,
    pub(crate) frame_size: i32,
    pub(crate) prev_redundancy: i32,
    pub(crate) last_packet_duration: i32,
    pub(crate) softclip_mem: [opus_val16; 2],
    pub(crate) rangeFinal: u32,
    pub(crate) ignore_extensions: bool,
}
impl OpusDecoder {
    pub fn channels(&self) -> i32 {
        self.channels
    }
    pub fn new(Fs: i32, channels: usize) -> Result<OpusDecoder, i32> {
        let valid_fs = Fs == 48000
            || Fs == 24000
            || Fs == 16000
            || Fs == 12000
            || Fs == 8000
            || cfg!(feature = "qext") && Fs == 96000;
        if !valid_fs || channels != 1 && channels != 2 {
            return Err(OPUS_BAD_ARG);
        }

        let mut st = OpusDecoder {
            celt_dec: celt_decoder_init(Fs, channels)?,
            silk_dec: silk_InitDecoder(),
            channels: channels as i32,
            Fs,
            DecControl: silk_DecControlStruct {
                nChannelsAPI: channels,
                nChannelsInternal: 0,
                API_sampleRate: Fs,
                internalSampleRate: 0,
                payloadSize_ms: 0,
                prevPitchLag: 0,
                enable_deep_plc: false,
                #[cfg(feature = "osce")]
                osce_method: 0,
                #[cfg(feature = "osce")]
                enable_osce_bwe: false,
                #[cfg(feature = "osce")]
                osce_extended_mode: 0,
                #[cfg(feature = "osce")]
                prev_osce_extended_mode: 0,
            },
            decode_gain: 0,
            complexity: 0,
            #[cfg(feature = "deep-plc")]
            lpcnet: crate::dnn::lpcnet::LPCNetPLCState::new(),
            stream_channels: channels as i32,
            bandwidth: 0,
            mode: 0,
            prev_mode: 0,
            frame_size: Fs / 400,
            prev_redundancy: 0,
            last_packet_duration: 0,
            softclip_mem: [0.0; 2],
            rangeFinal: 0,
            ignore_extensions: false,
        };

        st.celt_dec.signalling = 0;
        st.celt_dec.arch = opus_select_arch();

        Ok(st)
    }

    /// Check whether an Opus packet carries LBRR (in-band FEC) data.
    ///
    /// Returns:
    /// - `1` if LBRR is present
    /// - `0` if LBRR is not present
    /// - a negative Opus error code on parse failure
    ///
    /// Upstream C: src/opus_decoder.c:opus_packet_has_lbrr
    pub fn packet_has_lbrr(packet: &[u8]) -> i32 {
        if packet.is_empty() {
            return OPUS_BAD_ARG;
        }

        let packet_mode = opus_packet_get_mode(packet);
        if packet_mode == MODE_CELT_ONLY {
            return 0;
        }

        let packet_frame_size = opus_packet_get_samples_per_frame(packet[0], 48_000);
        let mut nb_frames = 1;
        if packet_frame_size > 960 {
            nb_frames = packet_frame_size / 960;
        }

        let packet_stream_channels = opus_packet_get_nb_channels(packet[0]);
        let mut frames = [0usize; 48];
        let mut size = [0i16; 48];
        let ret = opus_packet_parse(packet, None, Some(&mut frames), &mut size, None);
        if ret <= 0 {
            return ret;
        }
        if size[0] == 0 {
            return 0;
        }

        let mut lbrr = ((packet[frames[0]] >> (7 - nb_frames)) & 0x1) as i32;
        if packet_stream_channels == 2 {
            lbrr |= ((packet[frames[0]] >> (6 - 2 * nb_frames)) & 0x1) as i32;
        }
        lbrr
    }

    /// Decode an Opus packet into interleaved `i16` PCM samples.
    ///
    /// `data` is the compressed Opus packet. `pcm` must be large enough to
    /// hold `frame_size * channels` samples. `frame_size` is the maximum
    /// number of samples per channel to decode.
    ///
    /// Returns the number of decoded samples per channel on success, or a
    /// negative Opus error code on failure.
    pub fn decode(
        &mut self,
        data: &[u8],
        pcm: &mut [i16],
        frame_size: i32,
        decode_fec: bool,
    ) -> i32 {
        opus_decode(self, data, pcm, frame_size, decode_fec as i32)
    }

    /// Decode an Opus packet into interleaved `f32` PCM samples.
    ///
    /// `data` is the compressed Opus packet. `pcm` must be large enough to
    /// hold `frame_size * channels` samples. `frame_size` is the maximum
    /// number of samples per channel to decode.
    ///
    /// Returns the number of decoded samples per channel on success, or a
    /// negative Opus error code on failure.
    pub fn decode_float(
        &mut self,
        data: &[u8],
        pcm: &mut [f32],
        frame_size: i32,
        decode_fec: bool,
    ) -> i32 {
        opus_decode_float(self, data, pcm, frame_size, decode_fec as i32)
    }

    /// Decode an Opus packet into interleaved 24-bit PCM samples in `i32`.
    pub fn decode24(
        &mut self,
        data: &[u8],
        pcm: &mut [i32],
        frame_size: i32,
        decode_fec: bool,
    ) -> i32 {
        opus_decode24(self, data, pcm, frame_size, decode_fec as i32)
    }

    /// Decode PLC audio using a pre-parsed DRED state into interleaved `i16`.
    #[cfg(feature = "dred")]
    pub fn decode_dred(
        &mut self,
        dred: &OpusDRED,
        dred_offset: i32,
        pcm: &mut [i16],
        frame_size: i32,
    ) -> i32 {
        opus_decoder_dred_decode(self, dred, dred_offset, pcm, frame_size)
    }

    /// Decode PLC audio using a pre-parsed DRED state into interleaved `i32` (24-bit range).
    #[cfg(feature = "dred")]
    pub fn decode_dred24(
        &mut self,
        dred: &OpusDRED,
        dred_offset: i32,
        pcm: &mut [i32],
        frame_size: i32,
    ) -> i32 {
        opus_decoder_dred_decode24(self, dred, dred_offset, pcm, frame_size)
    }

    /// Decode PLC audio using a pre-parsed DRED state into interleaved `f32`.
    #[cfg(feature = "dred")]
    pub fn decode_dred_float(
        &mut self,
        dred: &OpusDRED,
        dred_offset: i32,
        pcm: &mut [f32],
        frame_size: i32,
    ) -> i32 {
        opus_decoder_dred_decode_float(self, dred, dred_offset, pcm, frame_size)
    }

    // -- Type-safe CTL getters and setters --

    pub fn set_gain(&mut self, gain: i32) -> Result<(), i32> {
        if !(-32768..=32767).contains(&gain) {
            return Err(OPUS_BAD_ARG);
        }
        self.decode_gain = gain;
        Ok(())
    }

    pub fn gain(&self) -> i32 {
        self.decode_gain
    }

    pub fn get_bandwidth(&self) -> i32 {
        self.bandwidth
    }

    pub fn sample_rate(&self) -> i32 {
        self.Fs
    }

    pub fn final_range(&self) -> u32 {
        self.rangeFinal
    }

    pub fn pitch(&mut self) -> i32 {
        if self.prev_mode == MODE_CELT_ONLY {
            self.celt_dec.postfilter_period
        } else {
            self.DecControl.prevPitchLag
        }
    }

    pub fn last_packet_duration(&self) -> i32 {
        self.last_packet_duration
    }

    pub fn set_phase_inversion_disabled(&mut self, disabled: bool) {
        self.celt_dec.disable_inv = disabled as i32;
    }

    pub fn phase_inversion_disabled(&self) -> bool {
        self.celt_dec.disable_inv != 0
    }

    pub fn set_complexity(&mut self, value: i32) -> Result<(), i32> {
        if !(0..=10).contains(&value) {
            return Err(OPUS_BAD_ARG);
        }
        self.complexity = value;
        self.celt_dec.complexity = value;
        Ok(())
    }

    pub fn complexity(&self) -> i32 {
        self.complexity
    }

    pub fn set_ignore_extensions(&mut self, value: bool) {
        self.ignore_extensions = value;
    }

    pub fn ignore_extensions(&self) -> bool {
        self.ignore_extensions
    }

    /// Enable or disable OSCE bandwidth extension (BWE).
    ///
    /// When enabled and conditions are met (complexity >= 4, 48kHz output,
    /// 16kHz internal, SILK-only mode), the decoder extends SILK wideband
    /// audio to fullband using a neural network.
    #[cfg(feature = "osce")]
    pub fn set_osce_bwe(&mut self, value: bool) {
        self.DecControl.enable_osce_bwe = value;
    }

    /// Returns whether OSCE bandwidth extension is enabled.
    #[cfg(feature = "osce")]
    pub fn osce_bwe(&self) -> bool {
        self.DecControl.enable_osce_bwe
    }

    /// Load DNN models from compiled-in weight data.
    ///
    /// Loads Deep PLC (always), OSCE (if `osce` feature), and DRED decoder
    /// weights. Returns `Ok(())` on success.
    ///
    /// Requires the `builtin-weights` feature.
    #[cfg(all(feature = "deep-plc", feature = "builtin-weights"))]
    pub fn load_dnn_weights(&mut self) -> Result<(), i32> {
        let arrays = crate::dnn::weights::compiled_weights();
        self.load_dnn_from_arrays(&arrays)
    }

    /// Load DNN models from an external binary weight blob.
    ///
    /// The blob must be in the upstream "DNNw" format.
    #[cfg(feature = "deep-plc")]
    pub fn set_dnn_blob(&mut self, data: &[u8]) -> Result<(), i32> {
        let arrays = crate::dnn::weights::load_weights(data).ok_or(OPUS_BAD_ARG)?;
        self.load_dnn_from_arrays(&arrays)
    }

    #[cfg(feature = "deep-plc")]
    fn load_dnn_from_arrays(
        &mut self,
        arrays: &[crate::dnn::nnet::WeightArray],
    ) -> Result<(), i32> {
        if !self.lpcnet.load_model(arrays) {
            return Err(OPUS_INTERNAL_ERROR);
        }
        #[cfg(feature = "osce")]
        {
            if !crate::dnn::osce::osce_load_models_from_arrays(
                &mut self.silk_dec.osce_model,
                arrays,
            ) {
                return Err(OPUS_INTERNAL_ERROR);
            }
        }
        Ok(())
    }

    pub fn reset(&mut self) {
        self.celt_dec.reset();
        silk_ResetDecoder(&mut self.silk_dec);
        #[cfg(feature = "deep-plc")]
        crate::dnn::lpcnet::lpcnet_plc_reset(&mut self.lpcnet);
        self.stream_channels = self.channels;
        self.bandwidth = 0;
        self.mode = 0;
        self.prev_mode = 0;
        self.frame_size = self.Fs / 400;
        self.prev_redundancy = 0;
        self.last_packet_duration = 0;
        self.softclip_mem = [0.0; 2];
        self.rangeFinal = 0;
    }
}

fn validate_opus_decoder(st: &OpusDecoder) {
    debug_assert!(st.channels == 1 || st.channels == 2);
    debug_assert!(
        st.Fs == 48000
            || st.Fs == 24000
            || st.Fs == 16000
            || st.Fs == 12000
            || st.Fs == 8000
            || cfg!(feature = "qext") && st.Fs == 96000
    );
    debug_assert!(st.DecControl.API_sampleRate == st.Fs);
    debug_assert!(
        st.DecControl.internalSampleRate == 0
            || st.DecControl.internalSampleRate == 16000
            || st.DecControl.internalSampleRate == 12000
            || st.DecControl.internalSampleRate == 8000
    );
    debug_assert!(st.DecControl.nChannelsAPI == st.channels as usize);
    debug_assert!(
        st.DecControl.nChannelsInternal == 0
            || st.DecControl.nChannelsInternal == 1
            || st.DecControl.nChannelsInternal == 2
    );
    debug_assert!(
        st.DecControl.payloadSize_ms == 0
            || st.DecControl.payloadSize_ms == 10
            || st.DecControl.payloadSize_ms == 20
            || st.DecControl.payloadSize_ms == 40
            || st.DecControl.payloadSize_ms == 60
    );
    debug_assert!(st.stream_channels == 1 || st.stream_channels == 2);
}

fn smooth_fade(
    in1: &[opus_val16],
    in2: &[opus_val16],
    out: &mut [opus_val16],
    overlap: i32,
    channels: i32,
    window: &[opus_val16],
    Fs: i32,
) {
    let inc: i32 = 48000 / Fs;
    let mut c: i32 = 0;
    while c < channels {
        let mut i: i32 = 0;
        while i < overlap {
            let w: opus_val16 = window[(i * inc) as usize] * window[(i * inc) as usize];
            out[(i * channels + c) as usize] = w * in2[(i * channels + c) as usize]
                + (1.0f32 - w) * in1[(i * channels + c) as usize];
            i += 1;
        }
        c += 1;
    }
}
fn opus_packet_get_mode(data: &[u8]) -> i32 {
    if data[0] as i32 & 0x80 != 0 {
        MODE_CELT_ONLY
    } else if data[0] as i32 & 0x60 == 0x60 {
        MODE_HYBRID
    } else {
        MODE_SILK_ONLY
    }
}
fn opus_decode_frame(
    st: &mut OpusDecoder,
    data: Option<&[u8]>,
    pcm: &mut [opus_val16],
    mut frame_size: i32,
    decode_fec: i32,
    #[cfg(feature = "qext")] qext_payload: Option<&[u8]>,
) -> i32 {
    let mut i: i32;
    let mut silk_ret: i32;
    let mut celt_ret: i32 = 0;
    // data_copy must be declared before dec so it outlives the borrow.
    // ec_dec_init requires &mut [u8]; decoder only reads from it.
    // Use stack buffer to avoid heap allocation. Max 1275 bytes per standard packet,
    // up to QEXT_PACKET_SIZE_CAP (3825) with QEXT.
    #[cfg(feature = "qext")]
    let mut data_copy = vec![0u8; data.map_or(0, |d| d.len())];
    #[cfg(not(feature = "qext"))]
    let mut data_copy = [0u8; 1275];
    let data_copy_len = data.map_or(0, |d| {
        data_copy[..d.len()].copy_from_slice(d);
        d.len()
    });
    let mut dec: ec_dec = ec_dec {
        buf: &mut [],
        storage: 0,
        end_offs: 0,
        end_window: 0,
        nend_bits: 0,
        nbits_total: 0,
        offs: 0,
        rng: 0,
        val: 0,
        ext: 0,
        rem: 0,
        error: 0,
    };
    let mut silk_frame_size: i32 = 0;

    let mut pcm_transition_silk_size: i32;
    let mut pcm_transition_celt_size: i32;

    let mut audiosize: i32;
    let mode: i32;
    let bandwidth: i32;
    let mut transition: i32 = 0;
    let mut start_band: i32;
    let mut redundancy: i32 = 0;
    let mut redundancy_bytes: i32 = 0;
    let mut celt_to_silk: i32 = 0;
    let mut c: i32;

    let mut redundant_rng: u32 = 0;

    let mut len = data.map_or(0i32, |d| d.len() as i32);
    let data = if len <= 1 { None } else { data };
    let F20: i32 = st.Fs / 50;
    let F10: i32 = F20 >> 1;
    let F5: i32 = F10 >> 1;
    let F2_5: i32 = F5 >> 1;
    if frame_size < F2_5 {
        return OPUS_BUFFER_TOO_SMALL;
    }
    frame_size = if frame_size < st.Fs / 25 * 3 {
        frame_size
    } else {
        st.Fs / 25 * 3
    };
    if data.is_none() {
        frame_size = if frame_size < st.frame_size {
            frame_size
        } else {
            st.frame_size
        };
    }
    if data.is_some() {
        audiosize = st.frame_size;
        mode = st.mode;
        bandwidth = st.bandwidth;
        dec = ec_dec_init(&mut data_copy[..data_copy_len]);
    } else {
        audiosize = frame_size;
        mode = if st.prev_redundancy != 0 {
            MODE_CELT_ONLY
        } else {
            st.prev_mode
        };
        bandwidth = 0;
        if mode == 0 {
            for sample in pcm[..(audiosize * st.channels) as usize].iter_mut() {
                *sample = 0 as opus_val16;
            }
            return audiosize;
        }
        if audiosize > F20 {
            let mut pcm_off: usize = 0;
            loop {
                let ret: i32 = opus_decode_frame(
                    st,
                    None,
                    &mut pcm[pcm_off..],
                    if audiosize < F20 { audiosize } else { F20 },
                    0,
                    #[cfg(feature = "qext")]
                    None,
                );
                if ret < 0 {
                    return ret;
                }
                pcm_off += (ret * st.channels) as usize;
                audiosize -= ret;
                if audiosize <= 0 {
                    break;
                }
            }
            return frame_size;
        } else if audiosize < F20 {
            if audiosize > F10 {
                audiosize = F10;
            } else if mode != MODE_SILK_ONLY && audiosize > F5 && audiosize < F10 {
                audiosize = F5;
            }
        }
    }
    let celt_accum: i32 = (mode != MODE_CELT_ONLY) as i32;
    pcm_transition_silk_size = ALLOC_NONE;
    pcm_transition_celt_size = ALLOC_NONE;
    if data.is_some()
        && st.prev_mode > 0
        && (mode == MODE_CELT_ONLY && st.prev_mode != MODE_CELT_ONLY && st.prev_redundancy == 0
            || mode != MODE_CELT_ONLY && st.prev_mode == MODE_CELT_ONLY)
    {
        transition = 1;
        if mode == MODE_CELT_ONLY {
            pcm_transition_celt_size = F5 * st.channels;
        } else {
            pcm_transition_silk_size = F5 * st.channels;
        }
    }
    let vla = pcm_transition_celt_size as usize;
    let mut pcm_transition_celt: Vec<opus_val16> = ::std::vec::from_elem(0., vla);
    if transition != 0 && mode == MODE_CELT_ONLY {
        opus_decode_frame(
            st,
            None,
            &mut pcm_transition_celt,
            if F5 < audiosize { F5 } else { audiosize },
            0,
            #[cfg(feature = "qext")]
            None,
        );
    }
    if audiosize > frame_size {
        return OPUS_BAD_ARG;
    } else {
        frame_size = audiosize;
    }
    // In hybrid/SILK mode, SILK decodes directly into pcm (float).
    // If frame_size < F10, we need a temp buffer since SILK needs at least F10.
    let pcm_too_small = frame_size < F10;
    let pcm_silk_size: i32 = if mode != MODE_CELT_ONLY && pcm_too_small {
        F10 * st.channels
    } else {
        ALLOC_NONE
    };
    let vla_0 = pcm_silk_size as usize;
    let mut pcm_silk: Vec<f32> = ::std::vec::from_elem(0., vla_0);
    if mode != MODE_CELT_ONLY {
        let mut decoded_samples: i32;
        let mut pcm_ptr_off: usize = 0;
        if st.prev_mode == MODE_CELT_ONLY {
            silk_ResetDecoder(&mut st.silk_dec);
        }
        st.DecControl.payloadSize_ms = if 10 > 1000 * audiosize / st.Fs {
            10
        } else {
            1000 * audiosize / st.Fs
        };
        if data.is_some() {
            st.DecControl.nChannelsInternal = st.stream_channels;
            if mode == MODE_SILK_ONLY {
                if bandwidth == OPUS_BANDWIDTH_NARROWBAND {
                    st.DecControl.internalSampleRate = 8000;
                } else if bandwidth == OPUS_BANDWIDTH_MEDIUMBAND {
                    st.DecControl.internalSampleRate = 12000;
                } else if bandwidth == OPUS_BANDWIDTH_WIDEBAND {
                    st.DecControl.internalSampleRate = 16000;
                } else {
                    st.DecControl.internalSampleRate = 16000;
                    debug_assert!(false, "libopus: assert(0) called");
                }
            } else {
                st.DecControl.internalSampleRate = 16000;
            }
        }
        // Set DNN control parameters based on complexity
        st.DecControl.enable_deep_plc = st.complexity >= 5;
        #[cfg(feature = "osce")]
        {
            st.DecControl.osce_method = 0; // OSCE_METHOD_NONE
            if st.complexity >= 6 {
                st.DecControl.osce_method = 1; // OSCE_METHOD_LACE
            }
            if st.complexity >= 7 {
                st.DecControl.osce_method = 2; // OSCE_METHOD_NOLACE
            }

            // BWE mode selection
            if st.complexity >= 4
                && st.DecControl.enable_osce_bwe
                && st.Fs == 48000
                && st.DecControl.internalSampleRate == 16000
                && (mode == MODE_SILK_ONLY || data.is_none())
            {
                // Request WB -> FB signal extension
                st.DecControl.osce_extended_mode = crate::dnn::osce::OSCE_MODE_SILK_BBWE;
            } else {
                // At this point, mode can only be MODE_SILK_ONLY or MODE_HYBRID
                st.DecControl.osce_extended_mode = if mode == MODE_SILK_ONLY {
                    crate::dnn::osce::OSCE_MODE_SILK_ONLY
                } else {
                    crate::dnn::osce::OSCE_MODE_HYBRID
                };
            }
            if st.prev_mode == MODE_CELT_ONLY {
                // Update extended mode for CELT->SILK transition
                st.DecControl.prev_osce_extended_mode = crate::dnn::osce::OSCE_MODE_CELT_ONLY;
            }
        }

        let lost_flag: i32 = if data.is_none() {
            1
        } else {
            2 * (decode_fec != 0) as i32
        };
        decoded_samples = 0;
        loop {
            let first_frame: i32 = (decoded_samples == 0) as i32;
            // SILK decodes into pcm directly, or pcm_silk if frame is too small
            let silk_out: &mut [f32] = if pcm_too_small {
                &mut pcm_silk[pcm_ptr_off..]
            } else {
                &mut pcm[pcm_ptr_off..]
            };
            silk_ret = silk_Decode(
                &mut st.silk_dec,
                &mut st.DecControl,
                lost_flag,
                first_frame,
                &mut dec,
                silk_out,
                &mut silk_frame_size,
                #[cfg(feature = "deep-plc")]
                Some(&mut st.lpcnet),
                st.celt_dec.arch,
            );
            if silk_ret != 0 {
                if lost_flag != 0 {
                    silk_frame_size = frame_size;
                    let silk_out: &mut [f32] = if pcm_too_small {
                        &mut pcm_silk[pcm_ptr_off..]
                    } else {
                        &mut pcm[pcm_ptr_off..]
                    };
                    for s in silk_out[..(frame_size * st.channels) as usize].iter_mut() {
                        *s = 0.;
                    }
                } else {
                    return OPUS_INTERNAL_ERROR;
                }
            }
            pcm_ptr_off += (silk_frame_size * st.channels) as usize;
            decoded_samples += silk_frame_size;
            if decoded_samples >= frame_size {
                break;
            }
        }
        if pcm_too_small {
            pcm[..(frame_size * st.channels) as usize]
                .copy_from_slice(&pcm_silk[..(frame_size * st.channels) as usize]);
        }
    }
    start_band = 0;
    if decode_fec == 0
        && mode != MODE_CELT_ONLY
        && data.is_some()
        && ec_tell(&dec) + 17 + 20 * (mode == MODE_HYBRID) as i32 <= 8 * len
    {
        if mode == MODE_HYBRID {
            redundancy = ec_dec_bit_logp(&mut dec, 12);
        } else {
            redundancy = 1;
        }
        if redundancy != 0 {
            celt_to_silk = ec_dec_bit_logp(&mut dec, 1);
            redundancy_bytes = if mode == MODE_HYBRID {
                ec_dec_uint(&mut dec, 256) as i32 + 2
            } else {
                len - ((ec_tell(&dec) + 7) >> 3)
            };
            len -= redundancy_bytes;
            if (len * 8) < ec_tell(&dec) {
                len = 0;
                redundancy_bytes = 0;
                redundancy = 0;
            }
            dec.storage = (dec.storage as u32).wrapping_sub(redundancy_bytes as u32) as u32 as u32;
        }
    }
    if mode != MODE_CELT_ONLY {
        start_band = 17;
    }
    if redundancy != 0 {
        transition = 0;
        pcm_transition_silk_size = ALLOC_NONE;
    }
    let vla_1 = pcm_transition_silk_size as usize;
    let mut pcm_transition_silk: Vec<opus_val16> = ::std::vec::from_elem(0., vla_1);
    if transition != 0 && mode != MODE_CELT_ONLY {
        opus_decode_frame(
            st,
            None,
            &mut pcm_transition_silk,
            if F5 < audiosize { F5 } else { audiosize },
            0,
            #[cfg(feature = "qext")]
            None,
        );
    }

    let celt_dec = &mut st.celt_dec;

    if bandwidth != 0 {
        let mut endband: i32 = 21;
        match bandwidth {
            OPUS_BANDWIDTH_NARROWBAND => {
                endband = 13;
            }
            OPUS_BANDWIDTH_MEDIUMBAND | OPUS_BANDWIDTH_WIDEBAND => {
                endband = 17;
            }
            OPUS_BANDWIDTH_SUPERWIDEBAND => {
                endband = 19;
            }
            OPUS_BANDWIDTH_FULLBAND => {
                endband = 21;
            }
            _ => {
                debug_assert!(false, "libopus: assert(0) called");
            }
        }
        celt_dec.end = endband;
    }
    celt_dec.stream_channels = st.stream_channels as usize;
    let redundant_audio_size: i32 = if redundancy != 0 {
        F5 * st.channels
    } else {
        ALLOC_NONE
    };
    let vla_2 = redundant_audio_size as usize;
    let mut redundant_audio: Vec<opus_val16> = ::std::vec::from_elem(0., vla_2);
    if redundancy != 0 && celt_to_silk != 0 {
        celt_dec.start = 0;
        celt_decode_with_ec(
            celt_dec,
            Some(&data.unwrap()[len as usize..len as usize + redundancy_bytes as usize]),
            &mut redundant_audio,
            F5,
            None,
            0,
            #[cfg(feature = "deep-plc")]
            None,
            #[cfg(feature = "qext")]
            None,
        );
        redundant_rng = celt_dec.rng;
    }
    celt_dec.start = start_band;
    #[cfg(feature = "osce")]
    let celt_decode_enabled = mode != MODE_SILK_ONLY
        && st.DecControl.osce_extended_mode != crate::dnn::osce::OSCE_MODE_SILK_BBWE;
    #[cfg(not(feature = "osce"))]
    let celt_decode_enabled = mode != MODE_SILK_ONLY;
    if celt_decode_enabled {
        let celt_frame_size: i32 = if F20 < frame_size { F20 } else { frame_size };
        if mode != st.prev_mode && st.prev_mode > 0 && st.prev_redundancy == 0 {
            celt_dec.reset();
        }
        celt_ret = celt_decode_with_ec(
            celt_dec,
            if decode_fec != 0 {
                None
            } else {
                data.as_ref().map(|d| &d[..len as usize])
            },
            &mut pcm[..(celt_frame_size * st.channels) as usize],
            celt_frame_size,
            Some(&mut dec),
            celt_accum,
            #[cfg(feature = "deep-plc")]
            Some(&mut st.lpcnet),
            #[cfg(feature = "qext")]
            qext_payload,
        );
        st.rangeFinal = celt_dec.rng;
    } else {
        let silence: [u8; 2] = [0xff, 0xff];
        if celt_accum == 0 {
            for sample in pcm[..(frame_size * st.channels) as usize].iter_mut() {
                *sample = 0 as opus_val16;
            }
        }
        if st.prev_mode == MODE_HYBRID
            && !(redundancy != 0 && celt_to_silk != 0 && st.prev_redundancy != 0)
        {
            celt_dec.start = 0;
            celt_decode_with_ec(
                celt_dec,
                Some(&silence[..2]),
                &mut pcm[..(F2_5 * st.channels) as usize],
                F2_5,
                None,
                celt_accum,
                #[cfg(feature = "deep-plc")]
                None,
                #[cfg(feature = "qext")]
                None,
            );
        }
        st.rangeFinal = dec.rng;
    }
    // In older C code: SILK decoded to int16 buffer, then mixed here.
    // In C 1.6.1: SILK decodes to float directly into pcm, CELT accumulates on top.
    // The mixing loop is no longer needed.
    let celt_mode = celt_dec.mode;
    let window = &celt_mode.window;
    if redundancy != 0 && celt_to_silk == 0 {
        celt_dec.reset();
        celt_dec.start = 0;
        celt_decode_with_ec(
            celt_dec,
            Some(&data.unwrap()[len as usize..len as usize + redundancy_bytes as usize]),
            &mut redundant_audio,
            F5,
            None,
            0,
            #[cfg(feature = "deep-plc")]
            None,
            #[cfg(feature = "qext")]
            None,
        );
        redundant_rng = celt_dec.rng;
        let fade_off = (st.channels * (frame_size - F2_5)) as usize;
        let red_off = (st.channels * F2_5) as usize;
        // Need a temporary copy for the in1 argument since pcm is also out
        let in1_copy: Vec<opus_val16> =
            pcm[fade_off..fade_off + (F2_5 * st.channels) as usize].to_vec();
        smooth_fade(
            &in1_copy,
            &redundant_audio[red_off..],
            &mut pcm[fade_off..],
            F2_5,
            st.channels,
            window,
            st.Fs,
        );
    }
    if redundancy != 0
        && celt_to_silk != 0
        && (st.prev_mode != MODE_SILK_ONLY || st.prev_redundancy != 0)
    {
        c = 0;
        while c < st.channels {
            i = 0;
            while i < F2_5 {
                pcm[(st.channels * i + c) as usize] =
                    redundant_audio[(st.channels * i + c) as usize];
                i += 1;
            }
            c += 1;
        }
        let red_off = (st.channels * F2_5) as usize;
        // Need a temporary copy for the in2 argument since pcm is also out
        let in2_copy: Vec<opus_val16> =
            pcm[red_off..red_off + (F2_5 * st.channels) as usize].to_vec();
        smooth_fade(
            &redundant_audio[red_off..],
            &in2_copy,
            &mut pcm[red_off..],
            F2_5,
            st.channels,
            window,
            st.Fs,
        );
    }
    if transition != 0 {
        let pcm_transition: &[opus_val16] = if mode == MODE_CELT_ONLY {
            &pcm_transition_celt
        } else {
            &pcm_transition_silk
        };
        if audiosize >= F5 {
            let ch_f2_5 = (st.channels * F2_5) as usize;
            pcm[..ch_f2_5].copy_from_slice(&pcm_transition[..ch_f2_5]);
            // Need a temporary copy for the in2 argument since pcm is also out
            let in2_copy: Vec<opus_val16> = pcm[ch_f2_5..ch_f2_5 * 2].to_vec();
            smooth_fade(
                &pcm_transition[ch_f2_5..],
                &in2_copy,
                &mut pcm[ch_f2_5..],
                F2_5,
                st.channels,
                window,
                st.Fs,
            );
        } else {
            // Need a temporary copy since pcm is both in2 and out
            let in2_copy: Vec<opus_val16> = pcm[..(F2_5 * st.channels) as usize].to_vec();
            smooth_fade(
                pcm_transition,
                &in2_copy,
                pcm,
                F2_5,
                st.channels,
                window,
                st.Fs,
            );
        }
    }
    if st.decode_gain != 0 {
        let gain: opus_val32 = celt_exp2(6.488_141e-4_f32 * st.decode_gain as f32);
        i = 0;
        while i < frame_size * st.channels {
            pcm[i as usize] *= gain;
            i += 1;
        }
    }
    if data.is_none() {
        st.rangeFinal = 0;
    } else {
        st.rangeFinal ^= redundant_rng;
    }
    st.prev_mode = mode;
    st.prev_redundancy = (redundancy != 0 && celt_to_silk == 0) as i32;
    if celt_ret >= 0 {
        let _ = _opus_false() != 0;
    }
    if celt_ret < 0 {
        celt_ret
    } else {
        audiosize
    }
}

#[cfg(feature = "dred")]
fn stage_dred_features_for_decode(
    st: &mut OpusDecoder,
    frame_size: i32,
    dred: &OpusDRED,
    dred_offset: i32,
) {
    // Upstream C: src/opus_decoder.c:opus_decode_native (ENABLE_DRED block)
    if dred.process_stage != 2 {
        return;
    }

    lpcnet_plc_fec_clear(&mut st.lpcnet);
    let f10 = st.Fs / 100;
    let init_frames = if st.lpcnet.blend == 0 { 2 } else { 0 };
    let features_per_frame = 1.max(frame_size / f10);
    let needed_feature_frames = init_frames + features_per_frame;
    let dred_shift_samples = dred_offset + dred.dred_offset * f10 / 4;
    let dred_frame_shift = (dred_shift_samples as f32 / f10 as f32).floor() as i32;

    for i in 0..needed_feature_frames {
        // Floor matches upstream; 5 ms overlap compensates for the 0.5 rounding.
        let feature_offset = init_frames - i - 2 + dred_frame_shift;
        if feature_offset >= 0 && feature_offset < 4 * dred.nb_latents {
            let start = feature_offset as usize * DRED_NUM_FEATURES;
            let end = start + DRED_NUM_FEATURES;
            if end <= dred.fec_features.len() {
                lpcnet_plc_fec_add(&mut st.lpcnet, Some(&dred.fec_features[start..end]));
            } else {
                debug_assert!(
                    false,
                    "libopus: DRED feature buffer bounds mismatch (start={start}, end={end}, len={})",
                    dred.fec_features.len()
                );
                lpcnet_plc_fec_add(&mut st.lpcnet, None);
            }
        } else if feature_offset >= 0 {
            lpcnet_plc_fec_add(&mut st.lpcnet, None);
        }
    }
}

pub fn opus_decode_native(
    st: &mut OpusDecoder,
    data: &[u8],
    pcm: &mut [opus_val16],
    frame_size: i32,
    decode_fec: i32,
    self_delimited: bool,
    packet_offset: Option<&mut i32>,
    soft_clip: i32,
    #[cfg(feature = "dred")] dred: Option<&OpusDRED>,
    #[cfg(not(feature = "dred"))] _dred: Option<&()>,
    dred_offset: i32,
) -> i32 {
    let mut i: i32 = 0;
    let mut nb_samples: usize = 0;
    let mut count: i32 = 0;
    let mut offset: i32 = 0;
    let mut toc: u8 = 0;
    let mut packet_frame_size: i32 = 0;
    let mut packet_bandwidth: i32 = 0;
    let mut packet_mode: i32 = 0;
    let mut packet_stream_channels: i32 = 0;
    let mut size: [i16; 48] = [0; 48];
    validate_opus_decoder(&*st);
    if !(0..=1).contains(&decode_fec) {
        return OPUS_BAD_ARG;
    }
    if (decode_fec != 0 || data.is_empty()) && frame_size % (st.Fs / 400) != 0 {
        return OPUS_BAD_ARG;
    }
    #[cfg(feature = "dred")]
    if let Some(dred) = dred {
        stage_dred_features_for_decode(st, frame_size, dred, dred_offset);
    }
    #[cfg(not(feature = "dred"))]
    let _ = dred_offset;

    if data.is_empty() {
        let mut pcm_count: i32 = 0;
        loop {
            let ret = opus_decode_frame(
                st,
                None,
                &mut pcm[(pcm_count * st.channels) as usize..],
                frame_size - pcm_count,
                0,
                #[cfg(feature = "qext")]
                None,
            );
            if ret < 0 {
                return ret;
            }
            pcm_count += ret;
            if pcm_count >= frame_size {
                break;
            }
        }
        debug_assert_eq!(pcm_count, frame_size);
        let _ = _opus_false() != 0;
        st.last_packet_duration = pcm_count;
        return pcm_count;
    }

    packet_mode = opus_packet_get_mode(data);
    packet_bandwidth = opus_packet_get_bandwidth(data[0]);
    packet_frame_size = opus_packet_get_samples_per_frame(data[0], st.Fs);
    packet_stream_channels = opus_packet_get_nb_channels(data[0]);
    let mut padding_len: i32 = 0;
    let mut parsed_packet_offset: i32 = 0;
    count = opus_packet_parse_impl(
        data,
        self_delimited,
        Some(&mut toc),
        None,
        &mut size,
        Some(&mut offset),
        Some(&mut parsed_packet_offset),
        Some(&mut padding_len),
    );
    if let Some(out_offset) = packet_offset {
        *out_offset = parsed_packet_offset;
    }
    if count < 0 {
        return count;
    }
    // For self-delimited multistream packets, extensions belong to the current
    // sub-packet, not the whole remaining payload.
    #[cfg(feature = "qext")]
    let padding_data = if padding_len > 0 && !st.ignore_extensions {
        let packet_len = if self_delimited && parsed_packet_offset > 0 {
            parsed_packet_offset as usize
        } else {
            data.len()
        };
        if packet_len >= padding_len as usize && packet_len <= data.len() {
            &data[packet_len - padding_len as usize..packet_len]
        } else {
            &[] as &[u8]
        }
    } else {
        &[] as &[u8]
    };
    let mut data = &data[offset as usize..];
    if decode_fec != 0 {
        let mut duration_copy: i32 = 0;
        let mut ret_0: i32 = 0;
        if frame_size < packet_frame_size
            || packet_mode == MODE_CELT_ONLY
            || st.mode == MODE_CELT_ONLY
        {
            return opus_decode_native(
                st,
                &[][..],
                pcm,
                frame_size,
                0,
                false,
                None,
                soft_clip,
                None,
                0,
            );
        }
        duration_copy = st.last_packet_duration;
        if frame_size - packet_frame_size != 0 {
            ret_0 = opus_decode_native(
                st,
                &[][..],
                pcm,
                frame_size - packet_frame_size,
                0,
                false,
                None,
                soft_clip,
                None,
                0,
            );
            if ret_0 < 0 {
                st.last_packet_duration = duration_copy;
                return ret_0;
            }
            debug_assert_eq!(ret_0, frame_size - packet_frame_size);
        }
        st.mode = packet_mode;
        st.bandwidth = packet_bandwidth;
        st.frame_size = packet_frame_size;
        st.stream_channels = packet_stream_channels;
        ret_0 = opus_decode_frame(
            st,
            Some(&data[..size[0] as usize]),
            &mut pcm[(st.channels * (frame_size - packet_frame_size)) as usize..],
            packet_frame_size,
            1,
            #[cfg(feature = "qext")]
            None,
        );
        if ret_0 < 0 {
            return ret_0;
        } else {
            let _ = _opus_false() != 0;
            st.last_packet_duration = frame_size;
            return frame_size;
        }
    }
    if count * packet_frame_size > frame_size {
        return OPUS_BUFFER_TOO_SMALL;
    }
    st.mode = packet_mode;
    st.bandwidth = packet_bandwidth;
    st.frame_size = packet_frame_size;
    st.stream_channels = packet_stream_channels;
    nb_samples = 0;
    #[cfg(feature = "qext")]
    let mut iter = if !padding_data.is_empty() && !st.ignore_extensions {
        Some(crate::opus::extensions::OpusExtensionIterator::new(
            padding_data,
            count,
        ))
    } else {
        None
    };
    i = 0;
    while i < count {
        // Per-frame QEXT extension lookup
        #[cfg(feature = "qext")]
        let qext_payload_vec: Option<Vec<u8>>;
        #[cfg(feature = "qext")]
        {
            let mut ext_data: Option<crate::opus::extensions::ExtensionRef> = None;
            if let Some(ref mut it) = iter {
                // Search forward until we find an extension for frame >= i,
                // matching the C pattern of saving/restoring iterator state.
                loop {
                    let frame_of_ext = ext_data.as_ref().map_or(-1, |e| e.frame);
                    if frame_of_ext >= i {
                        break;
                    }
                    let it_copy = it.clone();
                    match it.find(crate::celt::modes::data_96000::QEXT_EXTENSION_ID) {
                        Ok(Some(ext)) => {
                            if ext.frame > i {
                                // Went past our frame — restore iterator
                                *it = it_copy;
                                ext_data = Some(ext);
                                break;
                            }
                            ext_data = Some(ext);
                        }
                        _ => break,
                    }
                }
            }
            qext_payload_vec = ext_data.and_then(|ext| {
                if ext.frame == i {
                    Some(padding_data[ext.data_offset..ext.data_offset + ext.len].to_vec())
                } else {
                    None
                }
            });
        }
        let ret_1 = opus_decode_frame(
            st,
            Some(&data[..size[i as usize] as usize]),
            &mut pcm[(nb_samples * st.channels as usize)..],
            frame_size - nb_samples as i32,
            0,
            #[cfg(feature = "qext")]
            qext_payload_vec.as_deref(),
        );
        if ret_1 < 0 {
            return ret_1;
        }
        debug_assert_eq!(ret_1, packet_frame_size);
        data = &data[size[i as usize] as usize..];
        nb_samples += ret_1 as usize;
        i += 1;
    }
    st.last_packet_duration = nb_samples as i32;
    let _ = _opus_false() != 0;
    if soft_clip != 0 {
        opus_pcm_soft_clip_impl(
            pcm,
            nb_samples,
            st.channels as usize,
            &mut st.softclip_mem,
            st.celt_dec.arch,
        );
    } else {
        st.softclip_mem[1_usize] = 0 as opus_val16;
        st.softclip_mem[0_usize] = st.softclip_mem[1_usize];
    }
    nb_samples as i32
}

/// Upstream C: src/opus_decoder.c:opus_decode
pub fn opus_decode(
    st: &mut OpusDecoder,
    data: &[u8],
    pcm: &mut [i16],
    mut frame_size: i32,
    decode_fec: i32,
) -> i32 {
    if frame_size <= 0 {
        return OPUS_BAD_ARG;
    }
    if !data.is_empty() && decode_fec == 0 {
        let nb_samples = opus_decoder_get_nb_samples(st, data);
        if nb_samples > 0 {
            frame_size = if frame_size < nb_samples {
                frame_size
            } else {
                nb_samples
            };
        } else {
            return OPUS_INVALID_PACKET;
        }
    }
    debug_assert!(st.channels == 1 || st.channels == 2);
    let vla = (frame_size * st.channels) as usize;
    let mut out: Vec<f32> = ::std::vec::from_elem(0., vla);
    let ret = opus_decode_native(
        st, data, &mut out, frame_size, decode_fec, false, None, 1, None, 0,
    );
    if ret > 0 {
        celt_float2int16(&out, pcm, (ret * st.channels) as usize);
    }
    ret
}
/// Upstream C: src/opus_decoder.c:opus_decode_float
pub fn opus_decode_float(
    st: &mut OpusDecoder,
    data: &[u8],
    pcm: &mut [opus_val16],
    frame_size: i32,
    decode_fec: i32,
) -> i32 {
    if frame_size <= 0 {
        return OPUS_BAD_ARG;
    }
    opus_decode_native(
        st, data, pcm, frame_size, decode_fec, false, None, 0, None, 0,
    )
}

/// Upstream C: src/opus_decoder.c:opus_decode24
pub fn opus_decode24(
    st: &mut OpusDecoder,
    data: &[u8],
    pcm: &mut [i32],
    mut frame_size: i32,
    decode_fec: i32,
) -> i32 {
    if frame_size <= 0 {
        return OPUS_BAD_ARG;
    }
    if !data.is_empty() && decode_fec == 0 {
        let nb_samples = opus_decoder_get_nb_samples(st, data);
        if nb_samples > 0 {
            frame_size = frame_size.min(nb_samples);
        } else {
            return OPUS_INVALID_PACKET;
        }
    }
    debug_assert!(st.channels == 1 || st.channels == 2);
    let vla = (frame_size * st.channels) as usize;
    let mut out: Vec<f32> = ::std::vec::from_elem(0., vla);
    let ret = opus_decode_native(
        st, data, &mut out, frame_size, decode_fec, false, None, 0, None, 0,
    );
    if ret > 0 {
        for i in 0..(ret * st.channels) as usize {
            pcm[i] = float2int(32768.0f32 * 256.0f32 * out[i]);
        }
    }
    ret
}

/// Upstream C: src/opus_decoder.c:opus_decoder_dred_decode
#[cfg(feature = "dred")]
pub fn opus_decoder_dred_decode(
    st: &mut OpusDecoder,
    dred: &OpusDRED,
    dred_offset: i32,
    pcm: &mut [i16],
    frame_size: i32,
) -> i32 {
    if frame_size <= 0 {
        return OPUS_BAD_ARG;
    }
    debug_assert!(st.channels == 1 || st.channels == 2);
    let mut out = vec![0.0f32; (frame_size * st.channels) as usize];
    let ret = opus_decode_native(
        st,
        &[],
        &mut out,
        frame_size,
        0,
        false,
        None,
        1,
        Some(dred),
        dred_offset,
    );
    if ret > 0 {
        celt_float2int16(&out, pcm, (ret * st.channels) as usize);
    }
    ret
}

/// Upstream C: src/opus_decoder.c:opus_decoder_dred_decode24
#[cfg(feature = "dred")]
pub fn opus_decoder_dred_decode24(
    st: &mut OpusDecoder,
    dred: &OpusDRED,
    dred_offset: i32,
    pcm: &mut [i32],
    frame_size: i32,
) -> i32 {
    if frame_size <= 0 {
        return OPUS_BAD_ARG;
    }
    debug_assert!(st.channels == 1 || st.channels == 2);
    let mut out = vec![0.0f32; (frame_size * st.channels) as usize];
    let ret = opus_decode_native(
        st,
        &[],
        &mut out,
        frame_size,
        0,
        false,
        None,
        1,
        Some(dred),
        dred_offset,
    );
    if ret > 0 {
        for i in 0..(ret * st.channels) as usize {
            pcm[i] = float2int(32768.0f32 * 256.0f32 * out[i]);
        }
    }
    ret
}

/// Upstream C: src/opus_decoder.c:opus_decoder_dred_decode_float
#[cfg(feature = "dred")]
pub fn opus_decoder_dred_decode_float(
    st: &mut OpusDecoder,
    dred: &OpusDRED,
    dred_offset: i32,
    pcm: &mut [f32],
    frame_size: i32,
) -> i32 {
    if frame_size <= 0 {
        return OPUS_BAD_ARG;
    }
    opus_decode_native(
        st,
        &[],
        pcm,
        frame_size,
        0,
        false,
        None,
        0,
        Some(dred),
        dred_offset,
    )
}

/// Upstream C: src/opus_decoder.c:opus_dred_decoder_get_size
#[cfg(feature = "dred")]
pub fn opus_dred_decoder_get_size() -> i32 {
    std::mem::size_of::<OpusDREDDecoder>() as i32
}

/// Upstream C: src/opus_decoder.c:opus_dred_decoder_init
#[cfg(feature = "dred")]
pub fn opus_dred_decoder_init(dec: &mut OpusDREDDecoder) -> i32 {
    *dec = OpusDREDDecoder::new();
    #[cfg(feature = "builtin-weights")]
    {
        if !dec.load_dnn_weights() {
            return OPUS_INTERNAL_ERROR;
        }
    }
    OPUS_OK
}

/// Upstream C: src/opus_decoder.c:opus_dred_decoder_create
#[cfg(feature = "dred")]
pub fn opus_dred_decoder_create() -> Result<OpusDREDDecoder, i32> {
    let mut dec = OpusDREDDecoder::new();
    let ret = opus_dred_decoder_init(&mut dec);
    if ret != OPUS_OK {
        return Err(ret);
    }
    Ok(dec)
}

/// Upstream C: src/opus_decoder.c:opus_dred_decoder_destroy
#[cfg(feature = "dred")]
pub fn opus_dred_decoder_destroy(_dec: OpusDREDDecoder) {}

/// Upstream C: src/opus_decoder.c:opus_dred_decoder_ctl
#[cfg(feature = "dred")]
pub fn opus_dred_decoder_ctl(
    dred_dec: &mut OpusDREDDecoder,
    request: i32,
    dnn_blob: Option<&[u8]>,
) -> i32 {
    match request {
        crate::opus::opus_defines::OPUS_SET_DNN_BLOB_REQUEST => {
            let Some(blob) = dnn_blob else {
                return OPUS_BAD_ARG;
            };
            if blob.is_empty() {
                return OPUS_BAD_ARG;
            }
            if dred_dec.set_dnn_blob(blob) {
                OPUS_OK
            } else {
                OPUS_INTERNAL_ERROR
            }
        }
        _ => OPUS_UNIMPLEMENTED,
    }
}

/// Upstream C: src/opus_decoder.c:opus_dred_get_size
#[cfg(feature = "dred")]
pub fn opus_dred_get_size() -> i32 {
    std::mem::size_of::<OpusDRED>() as i32
}

/// Upstream C: src/opus_decoder.c:opus_dred_alloc
#[cfg(feature = "dred")]
pub fn opus_dred_alloc() -> OpusDRED {
    OpusDRED::new()
}

/// Upstream C: src/opus_decoder.c:opus_dred_free
#[cfg(feature = "dred")]
pub fn opus_dred_free(_dred: OpusDRED) {}

#[cfg(feature = "dred")]
fn dred_find_payload(data: &[u8]) -> Result<Option<(usize, usize, i32)>, i32> {
    if data.is_empty() {
        return Ok(None);
    }

    let mut sizes = [0i16; 48];
    let mut packet_offset = 0i32;
    let mut padding_len = 0i32;
    let nb_frames = opus_packet_parse_impl(
        data,
        false,
        None,
        None,
        &mut sizes,
        None,
        Some(&mut packet_offset),
        Some(&mut padding_len),
    );
    if nb_frames < 0 {
        return Err(nb_frames);
    }
    if padding_len <= 0 {
        return Ok(None);
    }

    let packet_len = if packet_offset > 0 {
        packet_offset as usize
    } else {
        data.len()
    };
    if packet_len > data.len() || packet_len < padding_len as usize {
        return Err(OPUS_INVALID_PACKET);
    }
    let padding_start = packet_len - padding_len as usize;
    let padding = &data[padding_start..packet_len];
    let frame_size = opus_packet_get_samples_per_frame(data[0], 48_000);
    let mut iter = crate::opus::extensions::OpusExtensionIterator::new(padding, nb_frames);

    loop {
        match iter.find(DRED_EXTENSION_ID)? {
            None => return Ok(None),
            Some(ext) => {
                let payload_start = ext.data_offset;
                let payload_end = payload_start + ext.len;
                let payload = &padding[payload_start..payload_end];
                if payload.len() > DRED_EXPERIMENTAL_BYTES
                    && payload[0] == b'D'
                    && payload[1] == DRED_EXPERIMENTAL_VERSION as u8
                {
                    let dred_frame_offset = ext.frame * frame_size / 120;
                    return Ok(Some((
                        padding_start + payload_start + DRED_EXPERIMENTAL_BYTES,
                        payload.len() - DRED_EXPERIMENTAL_BYTES,
                        dred_frame_offset,
                    )));
                }
            }
        }
    }
}

/// Upstream C: src/opus_decoder.c:opus_dred_parse
#[cfg(feature = "dred")]
pub fn opus_dred_parse(
    dred_dec: &mut OpusDREDDecoder,
    dred: &mut OpusDRED,
    data: &[u8],
    max_dred_samples: i32,
    sampling_rate: i32,
    dred_end: Option<&mut i32>,
    defer_processing: bool,
) -> i32 {
    if sampling_rate <= 0 {
        return OPUS_BAD_ARG;
    }
    if !dred_dec.loaded {
        return OPUS_UNIMPLEMENTED;
    }

    dred.process_stage = -1;
    let payload = match dred_find_payload(data) {
        Ok(payload) => payload,
        Err(err) => return err,
    };

    let Some((start, payload_len, dred_frame_offset)) = payload else {
        if let Some(end) = dred_end {
            *end = 0;
        }
        return 0;
    };

    let offset = 100 * max_dred_samples / sampling_rate;
    let min_feature_frames = (2 + offset)
        .min((2 * DRED_NUM_REDUNDANCY_FRAMES) as i32)
        .max(0) as usize;
    dred_ec_decode(
        dred,
        &data[start..start + payload_len],
        payload_len,
        min_feature_frames,
        dred_frame_offset,
    );
    if !defer_processing {
        if dred.process_stage == 1 {
            dred_process_impl(dred_dec, dred);
        } else if dred.process_stage != 2 {
            return OPUS_BAD_ARG;
        }
    }
    if let Some(end) = dred_end {
        *end = 0.max(-dred.dred_offset * sampling_rate / 400);
    }
    0.max(dred.nb_latents * sampling_rate / 25 - dred.dred_offset * sampling_rate / 400)
}

/// Upstream C: src/opus_decoder.c:opus_dred_process
#[cfg(feature = "dred")]
pub fn opus_dred_process(dred_dec: &OpusDREDDecoder, src: &OpusDRED, dst: &mut OpusDRED) -> i32 {
    if src.process_stage != 1 && src.process_stage != 2 {
        return OPUS_BAD_ARG;
    }
    if !dred_dec.loaded {
        return OPUS_UNIMPLEMENTED;
    }

    if !std::ptr::eq(src as *const OpusDRED, dst as *const OpusDRED) {
        *dst = src.clone();
    }
    if dst.process_stage == 2 {
        return OPUS_OK;
    }
    dred_process_impl(dred_dec, dst);
    OPUS_OK
}

/// Return packet bandwidth from the TOC byte.
///
/// Upstream C: src/opus_decoder.c:opus_packet_get_bandwidth
pub fn opus_packet_get_bandwidth(toc: u8) -> i32 {
    let mut bandwidth: i32;
    if toc as i32 & 0x80 != 0 {
        bandwidth = OPUS_BANDWIDTH_MEDIUMBAND + (toc as i32 >> 5 & 0x3);
        if bandwidth == OPUS_BANDWIDTH_MEDIUMBAND {
            bandwidth = OPUS_BANDWIDTH_NARROWBAND;
        }
    } else if toc as i32 & 0x60 == 0x60 {
        bandwidth = if toc as i32 & 0x10 != 0 {
            OPUS_BANDWIDTH_FULLBAND
        } else {
            OPUS_BANDWIDTH_SUPERWIDEBAND
        };
    } else {
        bandwidth = OPUS_BANDWIDTH_NARROWBAND + (toc as i32 >> 5 & 0x3);
    }
    bandwidth
}
/// Return the stream channel count encoded in the TOC byte (`1` or `2`).
///
/// Upstream C: src/opus_decoder.c:opus_packet_get_nb_channels
pub fn opus_packet_get_nb_channels(toc: u8) -> i32 {
    if toc as i32 & 0x4 != 0 {
        2
    } else {
        1
    }
}
/// Return the number of frames in an Opus packet.
///
/// Returns a negative `OPUS_*` code on malformed/truncated headers.
///
/// Upstream C: src/opus_decoder.c:opus_packet_get_nb_frames
pub fn opus_packet_get_nb_frames(packet: &[u8]) -> i32 {
    if packet.is_empty() {
        return OPUS_BAD_ARG;
    }
    let count = packet[0] & 0x3;
    if count == 0 {
        1
    } else if count != 3 {
        2
    } else if packet.len() < 2 {
        OPUS_INVALID_PACKET
    } else {
        (packet[1] & 0x3f) as i32
    }
}

/// Return decoded sample count for a packet at the given sample rate.
///
/// Enforces the Opus 120 ms maximum packet duration.
///
/// Upstream C: src/opus_decoder.c:opus_packet_get_nb_samples
pub fn opus_packet_get_nb_samples(packet: &[u8], Fs: i32) -> i32 {
    let mut samples: i32 = 0;
    let count: i32 = opus_packet_get_nb_frames(packet);
    if count < 0 {
        return count;
    }
    samples = count * opus_packet_get_samples_per_frame(packet[0], Fs);
    if samples * 25 > Fs * 3 {
        OPUS_INVALID_PACKET
    } else {
        samples
    }
}
/// Return decoded sample count for a packet using the decoder sample rate.
///
/// Upstream C: src/opus_decoder.c:opus_decoder_get_nb_samples
pub fn opus_decoder_get_nb_samples(dec: &mut OpusDecoder, packet: &[u8]) -> i32 {
    opus_packet_get_nb_samples(packet, dec.Fs)
}

#[cfg(all(test, feature = "dred"))]
mod tests {
    use super::*;
    use crate::opus::opus_defines::OPUS_SET_DNN_BLOB_REQUEST;

    #[test]
    fn stage_dred_features_blend_zero_queues_expected_frames() {
        let mut dec = OpusDecoder::new(48_000, 1).expect("decoder");
        dec.lpcnet.blend = 0;

        let mut dred = OpusDRED::new();
        dred.process_stage = 2;
        dred.nb_latents = 1;
        for frame_idx in 0..4 {
            let start = frame_idx * DRED_NUM_FEATURES;
            dred.fec_features[start] = frame_idx as f32;
        }

        stage_dred_features_for_decode(&mut dec, 960, &dred, 960);

        assert_eq!(dec.lpcnet.fec_fill_pos, 3);
        assert_eq!(dec.lpcnet.fec_skip, 0);
        assert_eq!(dec.lpcnet.fec[0][0], 2.0);
        assert_eq!(dec.lpcnet.fec[1][0], 1.0);
        assert_eq!(dec.lpcnet.fec[2][0], 0.0);
    }

    #[test]
    fn stage_dred_features_non_processed_noop() {
        let mut dec = OpusDecoder::new(48_000, 1).expect("decoder");
        dec.lpcnet.fec_fill_pos = 2;
        dec.lpcnet.fec_skip = 7;

        let dred = OpusDRED::new();
        stage_dred_features_for_decode(&mut dec, 960, &dred, 0);

        assert_eq!(dec.lpcnet.fec_fill_pos, 2);
        assert_eq!(dec.lpcnet.fec_skip, 7);
    }

    #[test]
    fn dred_decoder_ctl_rejects_missing_blob() {
        let mut dred_dec = OpusDREDDecoder::new();
        let ret = opus_dred_decoder_ctl(&mut dred_dec, OPUS_SET_DNN_BLOB_REQUEST, None);
        assert_eq!(ret, OPUS_BAD_ARG);
    }

    #[test]
    fn dred_decoder_ctl_unknown_request_unimplemented() {
        let mut dred_dec = OpusDREDDecoder::new();
        let ret = opus_dred_decoder_ctl(&mut dred_dec, 0x7fff, None);
        assert_eq!(ret, OPUS_UNIMPLEMENTED);
    }

    #[test]
    fn dred_process_rejects_invalid_stage() {
        let dred_dec = OpusDREDDecoder::new();
        let mut src = OpusDRED::new();
        src.process_stage = 0;
        let mut dst = OpusDRED::new();
        let ret = opus_dred_process(&dred_dec, &src, &mut dst);
        assert_eq!(ret, OPUS_BAD_ARG);
    }

    #[test]
    fn dred_process_requires_loaded_decoder() {
        let dred_dec = OpusDREDDecoder::new();
        let mut src = OpusDRED::new();
        src.process_stage = 2;
        let mut dst = OpusDRED::new();
        let ret = opus_dred_process(&dred_dec, &src, &mut dst);
        assert_eq!(ret, OPUS_UNIMPLEMENTED);
    }

    #[cfg(feature = "builtin-weights")]
    #[test]
    fn dred_parse_empty_packet_returns_zero() {
        let mut dred_dec = opus_dred_decoder_create().expect("dred decoder");
        let mut dred = opus_dred_alloc();
        let mut dred_end = -1;
        let ret = opus_dred_parse(
            &mut dred_dec,
            &mut dred,
            &[],
            960,
            48_000,
            Some(&mut dred_end),
            false,
        );
        assert_eq!(ret, 0);
        assert_eq!(dred_end, 0);
    }

    #[cfg(feature = "builtin-weights")]
    #[test]
    fn dred_process_copies_src_when_dst_is_distinct() {
        let dred_dec = opus_dred_decoder_create().expect("dred decoder");
        let mut src = OpusDRED::new();
        src.process_stage = 2;
        src.nb_latents = 3;
        src.dred_offset = -7;
        src.fec_features[0] = 1.5;
        src.state[0] = -2.0;

        let mut dst = OpusDRED::new();
        let ret = opus_dred_process(&dred_dec, &src, &mut dst);
        assert_eq!(ret, OPUS_OK);
        assert_eq!(dst.process_stage, 2);
        assert_eq!(dst.nb_latents, 3);
        assert_eq!(dst.dred_offset, -7);
        assert_eq!(dst.fec_features[0], 1.5);
        assert_eq!(dst.state[0], -2.0);
    }
}
