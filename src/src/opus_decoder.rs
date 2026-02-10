pub mod arch_h {
    pub type opus_val16 = f32;
    pub type opus_val32 = f32;
}
pub mod stddef_h {}
pub mod stack_alloc_h {
    pub const ALLOC_NONE: i32 = 1;
    #[inline]
    pub fn _opus_false() -> i32 {
        return 0;
    }
}
pub use self::arch_h::{opus_val16, opus_val32};
pub use self::stack_alloc_h::{_opus_false, ALLOC_NONE};

use crate::celt::celt::CELT_SET_SIGNALLING_REQUEST;
use crate::celt::celt_decoder::{celt_decode_with_ec, celt_decoder_init, OpusCustomDecoder};
use crate::celt::entcode::ec_tell;
use crate::celt::entdec::ec_dec;
use crate::celt::entdec::{ec_dec_bit_logp, ec_dec_init, ec_dec_uint};
use crate::celt::float_cast::FLOAT2INT16;
use crate::celt::mathops::celt_exp2;
use crate::celt::modes::OpusCustomMode;
use crate::silk::dec_API::{silk_DecControlStruct, silk_decoder};
use crate::silk::dec_API::{silk_Decode, silk_InitDecoder};
use crate::src::opus::opus_packet_parse_impl;
use crate::src::opus_defines::{
    OPUS_BAD_ARG, OPUS_BANDWIDTH_FULLBAND, OPUS_BANDWIDTH_MEDIUMBAND, OPUS_BANDWIDTH_NARROWBAND,
    OPUS_BANDWIDTH_SUPERWIDEBAND, OPUS_BANDWIDTH_WIDEBAND, OPUS_BUFFER_TOO_SMALL,
    OPUS_GET_BANDWIDTH_REQUEST, OPUS_GET_FINAL_RANGE_REQUEST, OPUS_GET_GAIN_REQUEST,
    OPUS_GET_LAST_PACKET_DURATION_REQUEST, OPUS_GET_PHASE_INVERSION_DISABLED_REQUEST,
    OPUS_GET_PITCH_REQUEST, OPUS_GET_SAMPLE_RATE_REQUEST, OPUS_INTERNAL_ERROR, OPUS_INVALID_PACKET,
    OPUS_OK, OPUS_RESET_STATE, OPUS_SET_GAIN_REQUEST, OPUS_SET_PHASE_INVERSION_DISABLED_REQUEST,
    OPUS_UNIMPLEMENTED,
};
use crate::src::opus_private::{align, MODE_CELT_ONLY, MODE_HYBRID, MODE_SILK_ONLY};
use crate::varargs::VarArgs;
use crate::{opus_custom_decoder_ctl, opus_packet_get_samples_per_frame, opus_pcm_soft_clip};

#[derive(Clone)]
#[repr(C)]
pub struct OpusDecoder {
    pub(crate) celt_dec: OpusCustomDecoder,
    pub(crate) silk_dec: silk_decoder,
    pub(crate) channels: i32,
    pub(crate) Fs: i32,
    pub(crate) DecControl: silk_DecControlStruct,
    pub(crate) decode_gain: i32,
    pub(crate) stream_channels: i32,
    pub(crate) bandwidth: i32,
    pub(crate) mode: i32,
    pub(crate) prev_mode: i32,
    pub(crate) frame_size: i32,
    pub(crate) prev_redundancy: i32,
    pub(crate) last_packet_duration: i32,
    pub(crate) softclip_mem: [opus_val16; 2],
    pub(crate) rangeFinal: u32,
}
impl OpusDecoder {
    pub fn channels(&self) -> i32 {
        self.channels
    }
    pub fn new(Fs: i32, channels: usize) -> Result<OpusDecoder, i32> {
        if Fs != 48000 && Fs != 24000 && Fs != 16000 && Fs != 12000 && Fs != 8000
            || channels != 1 && channels != 2
        {
            return Err(OPUS_BAD_ARG);
        }

        let mut st = OpusDecoder {
            celt_dec: celt_decoder_init(Fs, channels),
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
            },
            decode_gain: 0,
            stream_channels: channels as i32,
            bandwidth: 0,
            mode: 0,
            prev_mode: 0,
            frame_size: Fs / 400,
            prev_redundancy: 0,
            last_packet_duration: 0,
            softclip_mem: [0.0; 2],
            rangeFinal: 0,
        };

        opus_custom_decoder_ctl!(&mut st.celt_dec, CELT_SET_SIGNALLING_REQUEST, 0);

        Ok(st)
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

    // -- Type-safe CTL getters and setters --

    pub fn set_gain(&mut self, gain: i32) -> Result<(), i32> {
        if gain < -32768 || gain > 32767 {
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
            let mut pitch = 0;
            opus_custom_decoder_ctl!(&mut self.celt_dec, OPUS_GET_PITCH_REQUEST, &mut pitch);
            pitch
        } else {
            self.DecControl.prevPitchLag
        }
    }

    pub fn last_packet_duration(&self) -> i32 {
        self.last_packet_duration
    }

    pub fn set_phase_inversion_disabled(&mut self, disabled: bool) {
        opus_custom_decoder_ctl!(
            &mut self.celt_dec,
            OPUS_SET_PHASE_INVERSION_DISABLED_REQUEST,
            disabled as i32,
        );
    }

    pub fn phase_inversion_disabled(&self) -> bool {
        self.celt_dec.disable_inv != 0
    }

    pub fn reset(&mut self) {
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
    assert!(st.channels == 1 || st.channels == 2);
    assert!(st.Fs == 48000 || st.Fs == 24000 || st.Fs == 16000 || st.Fs == 12000 || st.Fs == 8000);
    assert!(st.DecControl.API_sampleRate == st.Fs);
    assert!(
        st.DecControl.internalSampleRate == 0
            || st.DecControl.internalSampleRate == 16000
            || st.DecControl.internalSampleRate == 12000
            || st.DecControl.internalSampleRate == 8000
    );
    assert!(st.DecControl.nChannelsAPI == st.channels as usize);
    assert!(
        st.DecControl.nChannelsInternal == 0
            || st.DecControl.nChannelsInternal == 1
            || st.DecControl.nChannelsInternal == 2
    );
    assert!(
        st.DecControl.payloadSize_ms == 0
            || st.DecControl.payloadSize_ms == 10
            || st.DecControl.payloadSize_ms == 20
            || st.DecControl.payloadSize_ms == 40
            || st.DecControl.payloadSize_ms == 60
    );
    assert!(st.stream_channels == 1 || st.stream_channels == 2);
}
#[deprecated]
pub fn opus_decoder_get_size(channels: i32) -> i32 {
    if channels < 1 || channels > 2 {
        return 0;
    }
    align(core::mem::size_of::<OpusDecoder>() as _)
}
#[deprecated]
pub unsafe fn opus_decoder_init(st: *mut OpusDecoder, Fs: i32, channels: i32) -> i32 {
    match OpusDecoder::new(Fs, channels as usize) {
        Ok(dec) => {
            *st = dec;
            return OPUS_OK;
        }
        Err(err) => {
            return err;
        }
    }
}
#[deprecated]
pub unsafe fn opus_decoder_create(Fs: i32, channels: i32, error: *mut i32) -> *mut OpusDecoder {
    match OpusDecoder::new(Fs, channels as usize) {
        Ok(dec) => {
            if !error.is_null() {
                *error = OPUS_OK;
            }
            Box::into_raw(Box::new(dec))
        }
        Err(e) => {
            if !error.is_null() {
                *error = e;
            }
            std::ptr::null_mut()
        }
    }
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
unsafe fn opus_decode_frame(
    st: &mut OpusDecoder,
    data: Option<&[u8]>,
    pcm: &mut [opus_val16],
    mut frame_size: i32,
    decode_fec: i32,
) -> i32 {
    let mut i: i32;
    let mut silk_ret: i32;
    let mut celt_ret: i32 = 0;
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
    let pcm_silk_size: i32;
    let mut pcm_transition_silk_size: i32;
    let mut pcm_transition_celt_size: i32;
    let redundant_audio_size: i32;
    let mut audiosize: i32;
    let mode: i32;
    let bandwidth: i32;
    let mut transition: i32 = 0;
    let mut start_band: i32;
    let mut redundancy: i32 = 0;
    let mut redundancy_bytes: i32 = 0;
    let mut celt_to_silk: i32 = 0;
    let mut c: i32;
    let F2_5: i32;
    let F5: i32;
    let F10: i32;
    let F20: i32;
    let mut redundant_rng: u32 = 0;
    let celt_accum: i32;
    let mut len = data.map_or(0i32, |d| d.len() as i32);
    let data = if len <= 1 { None } else { data };
    F20 = st.Fs / 50;
    F10 = F20 >> 1;
    F5 = F10 >> 1;
    F2_5 = F5 >> 1;
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
    if let Some(d) = data {
        audiosize = st.frame_size;
        mode = st.mode;
        bandwidth = st.bandwidth;
        dec =
            ec_dec_init(unsafe { std::slice::from_raw_parts_mut(d.as_ptr() as *mut u8, d.len()) });
    } else {
        audiosize = frame_size;
        mode = st.prev_mode;
        bandwidth = 0;
        if mode == 0 {
            for j in 0..(audiosize * st.channels) as usize {
                pcm[j] = 0 as opus_val16;
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
                );
                if ret < 0 {
                    return ret;
                }
                pcm_off += (ret * st.channels) as usize;
                audiosize -= ret;
                if !(audiosize > 0) {
                    break;
                }
            }
            return frame_size;
        } else {
            if audiosize < F20 {
                if audiosize > F10 {
                    audiosize = F10;
                } else if mode != MODE_SILK_ONLY && audiosize > F5 && audiosize < F10 {
                    audiosize = F5;
                }
            }
        }
    }
    celt_accum = 0;
    pcm_transition_silk_size = ALLOC_NONE;
    pcm_transition_celt_size = ALLOC_NONE;
    if data.is_some()
        && st.prev_mode > 0
        && (mode == MODE_CELT_ONLY && st.prev_mode != MODE_CELT_ONLY && st.prev_redundancy == 0
            || mode != MODE_CELT_ONLY && st.prev_mode == MODE_CELT_ONLY)
    {
        transition = 1;
        if mode == MODE_CELT_ONLY {
            pcm_transition_celt_size = F5 * st.channels as i32;
        } else {
            pcm_transition_silk_size = F5 * st.channels as i32;
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
        );
    }
    if audiosize > frame_size {
        return OPUS_BAD_ARG;
    } else {
        frame_size = audiosize;
    }
    pcm_silk_size = if mode != MODE_CELT_ONLY && celt_accum == 0 {
        (if F10 > frame_size { F10 } else { frame_size }) * st.channels as i32
    } else {
        ALLOC_NONE
    };
    let vla_0 = pcm_silk_size as usize;
    let mut pcm_silk: Vec<i16> = ::std::vec::from_elem(0, vla_0);
    if mode != MODE_CELT_ONLY {
        let lost_flag: i32;
        let mut decoded_samples: i32;
        let mut pcm_ptr_off: usize = 0;
        if st.prev_mode == MODE_CELT_ONLY {
            st.silk_dec = silk_InitDecoder();
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
                    panic!("libopus: assert(0) called");
                }
            } else {
                st.DecControl.internalSampleRate = 16000;
            }
        }
        lost_flag = if data.is_none() { 1 } else { 2 * decode_fec };
        decoded_samples = 0;
        loop {
            let first_frame: i32 = (decoded_samples == 0) as i32;
            silk_ret = silk_Decode(
                &mut st.silk_dec,
                &mut st.DecControl,
                lost_flag,
                first_frame,
                &mut dec,
                &mut pcm_silk[pcm_ptr_off..],
                &mut silk_frame_size,
                0,
            );
            if silk_ret != 0 {
                if lost_flag != 0 {
                    silk_frame_size = frame_size;
                    for j in 0..(frame_size * st.channels) as usize {
                        pcm_silk[pcm_ptr_off + j] = 0;
                    }
                } else {
                    return OPUS_INTERNAL_ERROR;
                }
            }
            pcm_ptr_off += (silk_frame_size * st.channels) as usize;
            decoded_samples += silk_frame_size;
            if !(decoded_samples < frame_size) {
                break;
            }
        }
    }
    start_band = 0;
    if decode_fec == 0
        && mode != MODE_CELT_ONLY
        && data.is_some()
        && ec_tell(&mut dec) + 17 + 20 * (st.mode == MODE_HYBRID) as i32 <= 8 * len
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
                len - (ec_tell(&mut dec) + 7 >> 3)
            };
            len -= redundancy_bytes;
            if (len * 8) < ec_tell(&mut dec) {
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
                panic!("libopus: assert(0) called");
            }
        }
        assert!(opus_custom_decoder_ctl!(celt_dec, 10012, endband) == 0);
    }
    assert!(opus_custom_decoder_ctl!(celt_dec, 10008, st.stream_channels) == 0);
    redundant_audio_size = if redundancy != 0 {
        F5 * st.channels
    } else {
        ALLOC_NONE
    };
    let vla_2 = redundant_audio_size as usize;
    let mut redundant_audio: Vec<opus_val16> = ::std::vec::from_elem(0., vla_2);
    if redundancy != 0 && celt_to_silk != 0 {
        assert!(opus_custom_decoder_ctl!(celt_dec, 10010, 0) == 0);
        celt_decode_with_ec(
            celt_dec,
            Some(&data.unwrap()[len as usize..len as usize + redundancy_bytes as usize]),
            &mut redundant_audio,
            F5,
            None,
            0,
        );
        assert!(opus_custom_decoder_ctl!(celt_dec, 4031, &mut redundant_rng) == 0);
    }
    assert!(opus_custom_decoder_ctl!(celt_dec, 10010, start_band) == 0);
    if mode != MODE_SILK_ONLY {
        let celt_frame_size: i32 = if F20 < frame_size { F20 } else { frame_size };
        if mode != st.prev_mode && st.prev_mode > 0 && st.prev_redundancy == 0 {
            assert!(opus_custom_decoder_ctl!(celt_dec, 4028) == 0);
        }
        celt_ret = celt_decode_with_ec(
            celt_dec,
            if decode_fec != 0 || data.is_none() {
                None
            } else {
                Some(&data.unwrap()[..len as usize])
            },
            &mut pcm[..(celt_frame_size * st.channels) as usize],
            celt_frame_size,
            Some(&mut dec),
            celt_accum,
        );
    } else {
        let silence: [u8; 2] = [0xff, 0xff];
        if celt_accum == 0 {
            for j in 0..(frame_size * st.channels) as usize {
                pcm[j] = 0 as opus_val16;
            }
        }
        if st.prev_mode == MODE_HYBRID
            && !(redundancy != 0 && celt_to_silk != 0 && st.prev_redundancy != 0)
        {
            assert!(opus_custom_decoder_ctl!(celt_dec, 10010, 0) == 0);
            celt_decode_with_ec(
                celt_dec,
                Some(&silence[..2]),
                &mut pcm[..(F2_5 * st.channels) as usize],
                F2_5,
                None,
                celt_accum,
            );
        }
    }
    if mode != MODE_CELT_ONLY && celt_accum == 0 {
        i = 0;
        while i < frame_size * st.channels {
            pcm[i as usize] =
                pcm[i as usize] + 1.0f32 / 32768.0f32 * pcm_silk[i as usize] as i32 as f32;
            i += 1;
        }
    }
    let mut celt_mode: *const OpusCustomMode = 0 as *const OpusCustomMode;
    assert!(opus_custom_decoder_ctl!(celt_dec, 10015, &mut celt_mode) == 0);
    let window = &(*celt_mode).window;
    if redundancy != 0 && celt_to_silk == 0 {
        assert!(opus_custom_decoder_ctl!(celt_dec, 4028) == 0);
        assert!(opus_custom_decoder_ctl!(celt_dec, 10010, 0) == 0);
        celt_decode_with_ec(
            celt_dec,
            Some(&data.unwrap()[len as usize..len as usize + redundancy_bytes as usize]),
            &mut redundant_audio,
            F5,
            None,
            0,
        );
        assert!(opus_custom_decoder_ctl!(celt_dec, 4031, &mut redundant_rng) == 0);
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
    if redundancy != 0 && celt_to_silk != 0 {
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
        let gain: opus_val32 = celt_exp2(6.48814081e-4f32 * st.decode_gain as f32);
        i = 0;
        while i < frame_size * st.channels {
            pcm[i as usize] = pcm[i as usize] * gain;
            i += 1;
        }
    }
    if data.is_none() {
        st.rangeFinal = 0;
    } else {
        st.rangeFinal = dec.rng ^ redundant_rng;
    }
    st.prev_mode = mode;
    st.prev_redundancy = (redundancy != 0 && celt_to_silk == 0) as i32;
    if celt_ret >= 0 {
        let _ = _opus_false() != 0;
    }
    return if celt_ret < 0 { celt_ret } else { audiosize };
}
pub unsafe fn opus_decode_native(
    st: &mut OpusDecoder,
    data: &[u8],
    pcm: &mut [opus_val16],
    frame_size: i32,
    decode_fec: i32,
    self_delimited: bool,
    packet_offset: Option<&mut i32>,
    soft_clip: i32,
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
    if decode_fec < 0 || decode_fec > 1 {
        return OPUS_BAD_ARG;
    }
    if (decode_fec != 0 || data.len() == 0) && frame_size % (st.Fs / 400) != 0 {
        return OPUS_BAD_ARG;
    }
    if data.len() == 0 {
        let mut pcm_count: i32 = 0;
        loop {
            let ret = opus_decode_frame(
                st,
                None,
                &mut pcm[(pcm_count * st.channels) as usize..],
                frame_size - pcm_count,
                0,
            );
            if ret < 0 {
                return ret;
            }
            pcm_count += ret;
            if !(pcm_count < frame_size) {
                break;
            }
        }
        assert!(pcm_count == frame_size);
        let _ = _opus_false() != 0;
        st.last_packet_duration = pcm_count;
        return pcm_count;
    }

    packet_mode = opus_packet_get_mode(data);
    packet_bandwidth = opus_packet_get_bandwidth(data[0]);
    packet_frame_size = opus_packet_get_samples_per_frame(data[0], st.Fs);
    packet_stream_channels = opus_packet_get_nb_channels(data[0]);
    count = opus_packet_parse_impl(
        data,
        self_delimited,
        Some(&mut toc),
        None,
        &mut size,
        Some(&mut offset),
        packet_offset,
    );
    if count < 0 {
        return count;
    }
    let mut data = &data[offset as usize..];
    if decode_fec != 0 {
        let mut duration_copy: i32 = 0;
        let mut ret_0: i32 = 0;
        if frame_size < packet_frame_size
            || packet_mode == MODE_CELT_ONLY
            || st.mode == MODE_CELT_ONLY
        {
            return opus_decode_native(st, &[][..], pcm, frame_size, 0, false, None, soft_clip);
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
            );
            if ret_0 < 0 {
                st.last_packet_duration = duration_copy;
                return ret_0;
            }
            assert!(ret_0 == frame_size - packet_frame_size);
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
    i = 0;
    while i < count {
        let ret_1 = opus_decode_frame(
            st,
            Some(&data[..size[i as usize] as usize]),
            &mut pcm[(nb_samples * st.channels as usize)..],
            frame_size - nb_samples as i32,
            0,
        );
        if ret_1 < 0 {
            return ret_1;
        }
        assert!(ret_1 == packet_frame_size);
        data = &data[size[i as usize] as usize..];
        nb_samples += ret_1 as usize;
        i += 1;
    }
    st.last_packet_duration = nb_samples as i32;
    let _ = _opus_false() != 0;
    if soft_clip != 0 {
        opus_pcm_soft_clip(pcm, nb_samples, st.channels as usize, &mut st.softclip_mem);
    } else {
        st.softclip_mem[1 as usize] = 0 as opus_val16;
        st.softclip_mem[0 as usize] = st.softclip_mem[1 as usize];
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
    if data.len() > 0 && decode_fec == 0 {
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
    assert!(st.channels == 1 || st.channels == 2);
    let vla = (frame_size * st.channels) as usize;
    let mut out: Vec<f32> = ::std::vec::from_elem(0., vla);
    let ret =
        unsafe { opus_decode_native(st, data, &mut out, frame_size, decode_fec, false, None, 1) };
    if ret > 0 {
        for j in 0..(ret * st.channels) as usize {
            pcm[j] = FLOAT2INT16(out[j]);
        }
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
    return unsafe { opus_decode_native(st, data, pcm, frame_size, decode_fec, false, None, 0) };
}
/// Upstream C: src/opus_decoder.c:opus_decoder_ctl
pub fn opus_decoder_ctl_impl(st: &mut OpusDecoder, request: i32, args: VarArgs) -> i32 {
    let celt_dec = &mut st.celt_dec;

    let mut ap = args;

    match request {
        OPUS_GET_BANDWIDTH_REQUEST => {
            let value = ap.arg::<&mut i32>();
            *value = st.bandwidth;
            OPUS_OK
        }
        OPUS_GET_FINAL_RANGE_REQUEST => {
            let value_0 = ap.arg::<&mut u32>();
            *value_0 = st.rangeFinal;
            OPUS_OK
        }
        OPUS_RESET_STATE => {
            st.stream_channels = st.channels;
            st.bandwidth = 0;
            st.mode = 0;
            st.prev_mode = 0;
            st.frame_size = st.Fs / 400;
            st.prev_redundancy = 0;
            st.last_packet_duration = 0;
            st.softclip_mem = [0.0; 2];
            st.rangeFinal = 0;

            OPUS_OK
        }
        OPUS_GET_SAMPLE_RATE_REQUEST => {
            let value_1 = ap.arg::<&mut i32>();
            *value_1 = st.Fs;
            OPUS_OK
        }
        OPUS_GET_PITCH_REQUEST => {
            let value_2 = ap.arg::<&mut i32>();
            if st.prev_mode == MODE_CELT_ONLY {
                opus_custom_decoder_ctl!(celt_dec, OPUS_GET_PITCH_REQUEST, value_2,)
            } else {
                *value_2 = st.DecControl.prevPitchLag;
                OPUS_OK
            }
        }
        OPUS_GET_GAIN_REQUEST => {
            let value_3 = ap.arg::<&mut i32>();
            *value_3 = st.decode_gain;
            OPUS_OK
        }
        OPUS_SET_GAIN_REQUEST => {
            let value_4: i32 = ap.arg::<i32>();
            if value_4 < -(32768) || value_4 > 32767 {
                OPUS_BAD_ARG
            } else {
                st.decode_gain = value_4;
                OPUS_OK
            }
        }
        OPUS_GET_LAST_PACKET_DURATION_REQUEST => {
            let value_5 = ap.arg::<&mut i32>();
            *value_5 = st.last_packet_duration;
            OPUS_OK
        }
        OPUS_SET_PHASE_INVERSION_DISABLED_REQUEST => {
            let value_6: i32 = ap.arg::<i32>();
            if value_6 < 0 || value_6 > 1 {
                OPUS_BAD_ARG
            } else {
                opus_custom_decoder_ctl!(
                    celt_dec,
                    OPUS_SET_PHASE_INVERSION_DISABLED_REQUEST,
                    value_6
                )
            }
        }
        OPUS_GET_PHASE_INVERSION_DISABLED_REQUEST => {
            let value_7 = ap.arg::<&mut i32>();
            opus_custom_decoder_ctl!(celt_dec, OPUS_GET_PHASE_INVERSION_DISABLED_REQUEST, value_7)
        }
        _ => OPUS_UNIMPLEMENTED,
    }
}
#[macro_export]
macro_rules! opus_decoder_ctl {
    ($st:expr,$request:expr, $($arg:expr),*) => {
        $crate::opus_decoder_ctl_impl(
            $st,
            $request,
            $crate::varargs!($($arg),*)
        )
    };
    ($st:expr,$request:expr, $($arg:expr),*,) => {
        opus_decoder_ctl!($st, $request, $($arg),*)
    };
    ($st:expr,$request:expr) => {
        opus_decoder_ctl!($st, $request,)
    };
}
#[deprecated]
pub unsafe fn opus_decoder_destroy(st: *mut OpusDecoder) {
    drop(Box::from_raw(st));
}
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
pub fn opus_packet_get_nb_channels(toc: u8) -> i32 {
    if toc as i32 & 0x4 != 0 {
        2
    } else {
        1
    }
}
pub fn opus_packet_get_nb_frames(packet: &[u8]) -> i32 {
    if packet.len() < 1 {
        return OPUS_BAD_ARG;
    }
    let count = packet[0] & 0x3;
    if count == 0 {
        return 1;
    } else if count != 3 {
        return 2;
    } else if packet.len() < 2 {
        return OPUS_INVALID_PACKET;
    } else {
        return (packet[1] & 0x3f) as i32;
    };
}

pub fn opus_packet_get_nb_samples(packet: &[u8], Fs: i32) -> i32 {
    let mut samples: i32 = 0;
    let count: i32 = opus_packet_get_nb_frames(packet);
    if count < 0 {
        return count;
    }
    samples = count * opus_packet_get_samples_per_frame(packet[0], Fs);
    if samples * 25 > Fs * 3 {
        return OPUS_INVALID_PACKET;
    } else {
        return samples;
    };
}
pub fn opus_decoder_get_nb_samples(dec: &mut OpusDecoder, packet: &[u8]) -> i32 {
    return opus_packet_get_nb_samples(packet, dec.Fs);
}
