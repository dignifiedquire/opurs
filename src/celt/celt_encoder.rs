//! CELT encoder.
//!
//! Upstream C: `celt/celt_encoder.c`

use crate::arch::Arch;
use crate::celt::bands::{
    compute_band_energies, haar1, hysteresis_decision, normalise_bands, quant_all_bands,
    spreading_decision, SPREAD_AGGRESSIVE, SPREAD_NONE, SPREAD_NORMAL,
};

const CELT_SIG_SCALE: f32 = 32768.0;
const EPSILON: f32 = 1e-15;
#[derive(Copy, Clone)]
#[repr(C)]
pub struct SILKInfo {
    pub signal_type: i32,
    pub offset: i32,
}

use crate::celt::common::{
    comb_filter, init_caps, resampling_factor, SPREAD_ICDF, TAPSET_ICDF, TF_SELECT_TABLE, TRIM_ICDF,
};
use crate::celt::common::{COMBFILTER_MAXPERIOD, COMBFILTER_MINPERIOD};
use crate::celt::entcode::{ec_get_error, ec_tell, ec_tell_frac, BITRES};
use crate::celt::entenc::{
    ec_enc_bit_logp, ec_enc_bits, ec_enc_done, ec_enc_icdf, ec_enc_init, ec_enc_shrink,
    ec_enc_uint, EcEnc,
};
use crate::celt::mathops::{celt_exp2, celt_log2, celt_maxabs16, celt_sqrt};
use crate::celt::mdct::mdct_forward;
#[cfg(feature = "qext")]
use crate::celt::modes::compute_qext_mode;
use crate::celt::modes::{opus_custom_mode_create, OpusCustomMode};
use crate::celt::pitch::{celt_inner_prod, pitch_downsample, pitch_search, remove_doubling};
use crate::celt::quant_bands::{
    amp2_log2, quant_coarse_energy, quant_energy_finalise, quant_fine_energy, E_MEANS,
};
use crate::celt::rate::clt_compute_allocation;
#[cfg(feature = "qext")]
use crate::celt::rate::clt_compute_extra_allocation;

use crate::opus::analysis::AnalysisInfo;
use crate::opus::opus_defines::{OPUS_BAD_ARG, OPUS_BITRATE_MAX, OPUS_INTERNAL_ERROR};
use crate::silk::macros::EC_CLZ0;

///
/// The C version uses a flexible array member (`in_mem[1]`) at the end of the struct
/// to store overlap memory, prefilter memory, and band energy arrays in a contiguous
/// allocation. This Rust version uses fixed-size arrays sized for the maximum case
/// (2 channels, overlap=240 with QEXT 96 kHz, nb_ebands=21, COMBFILTER_MAXPERIOD=1024).
/// Upstream C: celt/celt_encoder.c:OpusCustomEncoder
#[derive(Copy, Clone)]
#[repr(C)]
pub struct OpusCustomEncoder {
    pub mode: &'static OpusCustomMode,
    pub channels: i32,
    pub stream_channels: i32,
    pub force_intra: i32,
    pub clip: i32,
    pub disable_pf: i32,
    pub complexity: i32,
    pub upsample: i32,
    pub start: i32,
    pub end: i32,
    pub bitrate: i32,
    pub vbr: i32,
    pub signalling: i32,
    pub constrained_vbr: i32,
    pub loss_rate: i32,
    pub lsb_depth: i32,
    pub lfe: i32,
    pub disable_inv: i32,
    pub arch: Arch,
    pub rng: u32,
    pub spread_decision: i32,
    pub delayed_intra: f32,
    pub tonal_average: i32,
    pub last_coded_bands: i32,
    pub hf_average: i32,
    pub tapset_decision: i32,
    pub prefilter_period: i32,
    pub prefilter_gain: f32,
    pub prefilter_tapset: i32,
    pub consec_transient: i32,
    pub analysis: AnalysisInfo,
    pub silk_info: SILKInfo,
    pub preemph_mem_e: [f32; 2],
    pub preemph_mem_d: [f32; 2],
    pub vbr_reservoir: i32,
    pub vbr_drift: i32,
    pub vbr_offset: i32,
    pub vbr_count: i32,
    pub overlap_max: f32,
    pub stereo_saving: f32,
    pub intensity: i32,
    /// Energy mask for surround encoding (set by multistream encoder).
    /// `energy_mask_len == 0` means no mask is active.
    pub energy_mask: [f32; 2 * 21],
    pub energy_mask_len: usize,
    pub spec_avg: f32,
    /// Overlap memory, size = channels * overlap (max 2*240 = 480)
    pub in_mem: [f32; 2 * 240],
    /// Prefilter memory, size = channels * QEXT_SCALE(COMBFILTER_MAXPERIOD) (max 2*2048 = 4096)
    pub prefilter_mem: [f32; 2 * PREFILTER_MEM_CHAN_CAP],
    /// Old band energies, size = channels * nb_ebands (max 2*21 = 42)
    pub old_band_e: [f32; 2 * 21],
    /// Old log energies, size = channels * nb_ebands (max 2*21 = 42)
    pub old_log_e: [f32; 2 * 21],
    /// Old log energies (2 frames ago), size = channels * nb_ebands (max 2*21 = 42)
    pub old_log_e2: [f32; 2 * 21],
    /// Energy quantization error, size = channels * nb_ebands (max 2*21 = 42)
    pub energy_error: [f32; 2 * 21],
    /// QEXT: enable quality extension encoding
    #[cfg(feature = "qext")]
    pub enable_qext: i32,
    /// QEXT: scaling factor (1 for 48 kHz, 2 for 96 kHz)
    #[cfg(feature = "qext")]
    pub qext_scale: i32,
    /// QEXT: old band energies for extension bands
    #[cfg(feature = "qext")]
    pub qext_old_band_e: [f32; 2 * crate::celt::modes::data_96000::NB_QEXT_BANDS],
}

#[cfg(feature = "qext")]
#[inline]
fn qext_scale_for_mode(mode: &OpusCustomMode) -> i32 {
    if mode.fs == 96000 && (mode.short_mdct_size == 240 || mode.short_mdct_size == 180) {
        2
    } else {
        1
    }
}

#[cfg(feature = "qext")]
const PREFILTER_MAX_SCALE: usize = 2;
#[cfg(not(feature = "qext"))]
const PREFILTER_MAX_SCALE: usize = 1;
const PREFILTER_MEM_CHAN_CAP: usize = COMBFILTER_MAXPERIOD as usize * PREFILTER_MAX_SCALE;

const TO_OPUS_TABLE: [u8; 20] = [
    0xE0, 0xE8, 0xF0, 0xF8, 0xC0, 0xC8, 0xD0, 0xD8, 0xA0, 0xA8, 0xB0, 0xB8, 0x00, 0x00, 0x00, 0x00,
    0x80, 0x88, 0x90, 0x98,
];

#[inline]
fn should_convert_custom_signalling_header(mode: &OpusCustomMode) -> bool {
    #[cfg(feature = "qext")]
    {
        let _ = mode;
        true
    }
    #[cfg(not(feature = "qext"))]
    {
        mode.fs == 48000 && mode.short_mdct_size == 120
    }
}

#[inline]
fn to_opus_header_byte(c: u8) -> Option<u8> {
    if c >= 0xA0 {
        return None;
    }
    let base = TO_OPUS_TABLE[(c >> 3) as usize];
    if base == 0 {
        None
    } else {
        Some(base | (c & 0x07))
    }
}

impl OpusCustomEncoder {
    /// Create a new CELT encoder. Returns Err(OPUS_INTERNAL_ERROR) on failure.
    pub fn new(sampling_rate: i32, channels: i32, arch: Arch) -> Result<Self, i32> {
        if !(0..=2).contains(&channels) {
            return Err(OPUS_BAD_ARG);
        }
        #[cfg(feature = "qext")]
        let (mode, upsample) = if sampling_rate == 96000 {
            (opus_custom_mode_create(96000, 1920, None).unwrap(), 1)
        } else {
            (
                opus_custom_mode_create(48000, 960, None).unwrap(),
                resampling_factor(sampling_rate),
            )
        };
        #[cfg(not(feature = "qext"))]
        let (mode, upsample) = (
            opus_custom_mode_create(48000, 960, None).unwrap(),
            resampling_factor(sampling_rate),
        );
        #[cfg(feature = "qext")]
        let qext_scale = qext_scale_for_mode(mode);
        if upsample == 0 {
            return Err(OPUS_BAD_ARG);
        }

        let mut st = OpusCustomEncoder {
            mode,
            channels,
            stream_channels: channels,
            force_intra: 0,
            clip: 1,
            disable_pf: 0,
            complexity: 5,
            upsample,
            start: 0,
            end: mode.eff_ebands,
            bitrate: OPUS_BITRATE_MAX,
            vbr: 0,
            signalling: 1,
            constrained_vbr: 1,
            loss_rate: 0,
            lsb_depth: 24,
            lfe: 0,
            disable_inv: 0,
            arch,
            rng: 0,
            spread_decision: 0,
            delayed_intra: 0.0,
            tonal_average: 0,
            last_coded_bands: 0,
            hf_average: 0,
            tapset_decision: 0,
            prefilter_period: 0,
            prefilter_gain: 0.0,
            prefilter_tapset: 0,
            consec_transient: 0,
            analysis: AnalysisInfo {
                valid: 0,
                tonality: 0.0,
                tonality_slope: 0.0,
                noisiness: 0.0,
                activity: 0.0,
                music_prob: 0.0,
                music_prob_min: 0.0,
                music_prob_max: 0.0,
                bandwidth: 0,
                activity_probability: 0.0,
                max_pitch_ratio: 0.0,
                leak_boost: [0; 19],
            },
            silk_info: SILKInfo {
                signal_type: 0,
                offset: 0,
            },
            preemph_mem_e: [0.0; 2],
            preemph_mem_d: [0.0; 2],
            vbr_reservoir: 0,
            vbr_drift: 0,
            vbr_offset: 0,
            vbr_count: 0,
            overlap_max: 0.0,
            stereo_saving: 0.0,
            intensity: 0,
            energy_mask: [0.0; 2 * 21],
            energy_mask_len: 0,
            spec_avg: 0.0,
            in_mem: [0.0; 2 * 240],
            prefilter_mem: [0.0; 2 * PREFILTER_MEM_CHAN_CAP],
            old_band_e: [0.0; 2 * 21],
            old_log_e: [0.0; 2 * 21],
            old_log_e2: [0.0; 2 * 21],
            energy_error: [0.0; 2 * 21],
            #[cfg(feature = "qext")]
            enable_qext: 0,
            #[cfg(feature = "qext")]
            qext_scale,
            #[cfg(feature = "qext")]
            qext_old_band_e: [0.0; 2 * crate::celt::modes::data_96000::NB_QEXT_BANDS],
        };
        st.reset();
        Ok(st)
    }

    /// Reset the encoder state to initial defaults.
    ///
    /// Zeros all transient state fields (rng, prefilter memory, band energies,
    /// VBR state, etc.) while preserving configuration fields (mode, channels,
    /// complexity, bitrate, etc.).
    pub fn reset(&mut self) {
        let nb_ebands = self.mode.nb_ebands;
        let cc = self.channels as usize;
        let overlap = self.mode.overlap;
        self.rng = 0;
        self.spread_decision = SPREAD_NORMAL;
        self.delayed_intra = 1_f32;
        self.tonal_average = 256;
        self.last_coded_bands = 0;
        self.hf_average = 0;
        self.tapset_decision = 0;
        self.prefilter_period = 0;
        self.prefilter_gain = 0.0;
        self.prefilter_tapset = 0;
        self.consec_transient = 0;
        self.analysis = AnalysisInfo {
            valid: 0,
            tonality: 0.0,
            tonality_slope: 0.0,
            noisiness: 0.0,
            activity: 0.0,
            music_prob: 0.0,
            music_prob_min: 0.0,
            music_prob_max: 0.0,
            bandwidth: 0,
            activity_probability: 0.0,
            max_pitch_ratio: 0.0,
            leak_boost: [0; 19],
        };
        self.silk_info = SILKInfo {
            signal_type: 0,
            offset: 0,
        };
        self.preemph_mem_e = [0.0; 2];
        self.preemph_mem_d = [0.0; 2];
        self.vbr_reservoir = 0;
        self.vbr_drift = 0;
        self.vbr_offset = 0;
        self.vbr_count = 0;
        self.overlap_max = 0.0;
        self.stereo_saving = 0.0;
        self.intensity = 0;
        self.spec_avg = 0.0;
        (&mut self.in_mem)[..cc * overlap].fill(0.0);
        #[cfg(feature = "qext")]
        let max_period = (COMBFILTER_MAXPERIOD * self.qext_scale) as usize;
        #[cfg(not(feature = "qext"))]
        let max_period = COMBFILTER_MAXPERIOD as usize;
        (&mut self.prefilter_mem)[..cc * max_period].fill(0.0);
        (&mut self.old_band_e)[..cc * nb_ebands].fill(0.0);
        (&mut self.old_log_e)[..cc * nb_ebands].fill(-28.0);
        (&mut self.old_log_e2)[..cc * nb_ebands].fill(-28.0);
        (&mut self.energy_error)[..cc * nb_ebands].fill(0.0);
        #[cfg(feature = "qext")]
        self.qext_old_band_e.fill(0.0);
    }

    /// Upstream C: celt/celt_encoder.c:opus_custom_encode
    pub fn encode(&mut self, pcm: &[i16], compressed: &mut [u8]) -> i32 {
        let channels = self.channels as usize;
        if channels == 0 || pcm.is_empty() || !pcm.len().is_multiple_of(channels) {
            return OPUS_BAD_ARG;
        }
        let frame_size = (pcm.len() / channels) as i32;
        opus_custom_encode(self, pcm, frame_size, compressed)
    }

    /// Upstream C: celt/celt_encoder.c:opus_custom_encode_float
    pub fn encode_float(&mut self, pcm: &[f32], compressed: &mut [u8]) -> i32 {
        let channels = self.channels as usize;
        if channels == 0 || pcm.is_empty() || !pcm.len().is_multiple_of(channels) {
            return OPUS_BAD_ARG;
        }
        let frame_size = (pcm.len() / channels) as i32;
        opus_custom_encode_float(self, pcm, frame_size, compressed)
    }

    /// Upstream C: celt/celt_encoder.c:opus_custom_encode24
    pub fn encode24(&mut self, pcm: &[i32], compressed: &mut [u8]) -> i32 {
        let channels = self.channels as usize;
        if channels == 0 || pcm.is_empty() || !pcm.len().is_multiple_of(channels) {
            return OPUS_BAD_ARG;
        }
        let frame_size = (pcm.len() / channels) as i32;
        opus_custom_encode24(self, pcm, frame_size, compressed)
    }

    pub fn set_signalling(&mut self, signalling: i32) {
        self.signalling = signalling;
    }
}

/// Upstream C: celt/celt_encoder.c:opus_custom_encode
pub fn opus_custom_encode(
    st: &mut OpusCustomEncoder,
    pcm: &[i16],
    frame_size: i32,
    compressed: &mut [u8],
) -> i32 {
    if frame_size <= 0 || st.channels <= 0 {
        return OPUS_BAD_ARG;
    }
    let required = match (frame_size as usize).checked_mul(st.channels as usize) {
        Some(v) => v,
        None => return OPUS_BAD_ARG,
    };
    if pcm.len() < required {
        return OPUS_BAD_ARG;
    }
    let mut input = vec![0.0f32; required];
    for i in 0..required {
        input[i] = (1.0f32 / 32768.0f32) * pcm[i] as f32;
    }
    celt_encode_with_ec(
        st,
        &input,
        frame_size,
        compressed,
        compressed.len() as i32,
        None,
        #[cfg(feature = "qext")]
        None,
        #[cfg(feature = "qext")]
        0,
    )
}

/// Upstream C: celt/celt_encoder.c:opus_custom_encode_float
pub fn opus_custom_encode_float(
    st: &mut OpusCustomEncoder,
    pcm: &[f32],
    frame_size: i32,
    compressed: &mut [u8],
) -> i32 {
    if frame_size <= 0 || st.channels <= 0 {
        return OPUS_BAD_ARG;
    }
    let required = match (frame_size as usize).checked_mul(st.channels as usize) {
        Some(v) => v,
        None => return OPUS_BAD_ARG,
    };
    if pcm.len() < required {
        return OPUS_BAD_ARG;
    }
    celt_encode_with_ec(
        st,
        &pcm[..required],
        frame_size,
        compressed,
        compressed.len() as i32,
        None,
        #[cfg(feature = "qext")]
        None,
        #[cfg(feature = "qext")]
        0,
    )
}

/// Upstream C: celt/celt_encoder.c:opus_custom_encode24
pub fn opus_custom_encode24(
    st: &mut OpusCustomEncoder,
    pcm: &[i32],
    frame_size: i32,
    compressed: &mut [u8],
) -> i32 {
    if frame_size <= 0 || st.channels <= 0 {
        return OPUS_BAD_ARG;
    }
    let required = match (frame_size as usize).checked_mul(st.channels as usize) {
        Some(v) => v,
        None => return OPUS_BAD_ARG,
    };
    if pcm.len() < required {
        return OPUS_BAD_ARG;
    }
    let mut input = vec![0.0f32; required];
    for i in 0..required {
        input[i] = (1.0f32 / 32768.0f32 / 256.0f32) * pcm[i] as f32;
    }
    celt_encode_with_ec(
        st,
        &input,
        frame_size,
        compressed,
        compressed.len() as i32,
        None,
        #[cfg(feature = "qext")]
        None,
        #[cfg(feature = "qext")]
        0,
    )
}

#[cfg(all(test, feature = "qext"))]
mod tests {
    use super::*;
    use crate::arch::Arch;

    #[test]
    fn encoder_sets_qext_scale_from_mode() {
        let enc_96k = OpusCustomEncoder::new(96000, 2, Arch::Scalar).unwrap();
        assert_eq!(enc_96k.qext_scale, 2);

        let enc_48k = OpusCustomEncoder::new(48000, 2, Arch::Scalar).unwrap();
        assert_eq!(enc_48k.qext_scale, 1);
    }

    #[test]
    fn encoder_reset_clears_qext_history() {
        let mut enc = OpusCustomEncoder::new(96000, 2, Arch::Scalar).unwrap();
        enc.qext_old_band_e.fill(1.0);
        enc.reset();
        assert!(enc.qext_old_band_e.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn encoder_prefilter_mem_capacity_scales_for_qext() {
        let enc_96k = OpusCustomEncoder::new(96000, 2, Arch::Scalar).unwrap();
        let needed_96k =
            (COMBFILTER_MAXPERIOD * enc_96k.qext_scale) as usize * enc_96k.channels as usize;
        assert!(enc_96k.prefilter_mem.len() >= needed_96k);

        let enc_48k = OpusCustomEncoder::new(48000, 2, Arch::Scalar).unwrap();
        let needed_48k =
            (COMBFILTER_MAXPERIOD * enc_48k.qext_scale) as usize * enc_48k.channels as usize;
        assert!(enc_48k.prefilter_mem.len() >= needed_48k);
    }

    #[test]
    fn encoder_new_invalid_sampling_rate_returns_bad_arg() {
        assert!(matches!(
            OpusCustomEncoder::new(12345, 2, Arch::Scalar),
            Err(OPUS_BAD_ARG)
        ));
    }

    #[test]
    fn encoder_new_invalid_channels_returns_bad_arg() {
        assert!(matches!(
            OpusCustomEncoder::new(48000, 3, Arch::Scalar),
            Err(OPUS_BAD_ARG)
        ));
    }
}

/// Upstream C: celt/celt_encoder.c:transient_analysis
fn transient_analysis(
    in_0: &[f32],
    len: i32,
    channels: i32,
    tf_estimate: &mut f32,
    tf_chan: &mut i32,
    allow_weak_transients: i32,
    weak_transient: &mut i32,
    tone_freq: f32,
    toneishness: f32,
) -> i32 {
    let mut i: i32;
    let mut mem0: f32;
    let mut mem1: f32;
    let mut is_transient: i32;
    let mut mask_metric: i32 = 0;
    let mut c: i32;

    // Forward masking: 6.7 dB/ms.
    let mut forward_decay: f32 = 0.0625f32;
    // Table of 6*64/x, trained on real data to minimize average error.
    const INV_TABLE: [u8; 128] = [
        255, 255, 156, 110, 86, 70, 59, 51, 45, 40, 37, 33, 31, 28, 26, 25, 23, 22, 21, 20, 19, 18,
        17, 16, 16, 15, 15, 14, 13, 13, 12, 12, 12, 12, 11, 11, 11, 10, 10, 10, 9, 9, 9, 9, 9, 9,
        8, 8, 8, 8, 8, 7, 7, 7, 7, 7, 7, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 5, 5, 5,
        5, 5, 5, 5, 5, 5, 5, 5, 5, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
        4, 4, 4, 4, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2,
    ];
    // len = n_i32 + overlap; max 1920 + 240 = 2160 (QEXT 96kHz).
    const MAX_TRANSIENT: usize = 2400;
    debug_assert!((len as usize) <= MAX_TRANSIENT);
    let mut tmp = [0.0f32; MAX_TRANSIENT];
    *weak_transient = 0;
    // For lower bitrates, be more conservative (3.3 dB/ms forward masking).
    // This avoids coding weak transients at very low bitrate where they can
    // cause unstable energy and/or partial collapse.
    if allow_weak_transients != 0 {
        forward_decay = 0.03125f32;
    }
    let len2: i32 = len / 2;
    c = 0;
    while c < channels {
        let mut mean: f32;
        let mut unmask: i32;

        let mut max_e: f32;
        mem0 = 0 as f32;
        mem1 = 0 as f32;
        // High-pass filter: (1 - 2*z^-1 + z^-2) / (1 - z^-1 + .5*z^-2).
        i = 0;
        while i < len {
            let x: f32 = in_0[(i + c * len) as usize];
            let y: f32 = mem0 + x;
            /* Modified code to shorten dependency chains: */
            let mem00: f32 = mem0;
            mem0 = mem0 - x + 0.5f32 * mem1;
            mem1 = x - mem00;
            tmp[i as usize] = y;
            i += 1;
        }
        // First few samples are unreliable because filter memory isn't propagated.
        tmp[..12].fill(0.0);
        mean = 0 as f32;
        mem0 = 0 as f32;
        // Group by two to reduce complexity.
        // Forward pass to compute the post-echo threshold.
        i = 0;
        while i < len2 {
            let x2: f32 = tmp[(2 * i) as usize] * tmp[(2 * i) as usize]
                + tmp[(2 * i + 1) as usize] * tmp[(2 * i + 1) as usize];
            mean += x2;
            mem0 = x2 + (1.0f32 - forward_decay) * mem0;
            tmp[i as usize] = forward_decay * mem0;
            i += 1;
        }
        mem0 = 0 as f32;
        max_e = 0 as f32;
        // Backward pass to compute the pre-echo threshold.
        i = len2 - 1;
        while i >= 0 {
            // Backward masking: 13.9 dB/ms.
            mem0 = tmp[i as usize] + 0.875f32 * mem0;
            tmp[i as usize] = 0.125f32 * mem0;
            max_e = if max_e > 0.125f32 * mem0 {
                max_e
            } else {
                0.125f32 * mem0
            };
            i -= 1;
        }
        // Ratio of frame energy over harmonic mean energy.
        // This is effectively a bitrate-normalized temporal noise-to-mask ratio.
        //
        // As a compromise with the old transient detector, frame energy is the
        // geometric mean of total energy and half the local max.
        mean = celt_sqrt((mean * max_e) * 0.5f32 * len2 as f32);
        // Inverse of mean energy (floating-point equivalent of Q15+6 path).
        let norm: f32 = len2 as f32 / (1e-15f32 + mean);
        // Compute harmonic mean while discarding unreliable boundaries.
        // Data is smooth enough here that taking 1/4 samples is sufficient.
        unmask = 0;
        // NaNs here indicate severe upstream-state corruption.
        // Keep the assert before table lookup to avoid out-of-bounds indexing.
        assert!(!(tmp[0]).is_nan());
        assert!(!norm.is_nan());
        i = 12;
        while i < len2 - 5 {
            let id: i32 = (if 0.0
                > (if 127.0 < (64.0 * norm * (tmp[i as usize] + 1e-15f32)).floor() {
                    127.0
                } else {
                    (64.0 * norm * (tmp[i as usize] + 1e-15f32)).floor()
                }) {
                0.0
            } else if 127.0 < (64.0 * norm * (tmp[i as usize] + 1e-15f32)).floor() {
                127.0
            } else {
                (64.0 * norm * (tmp[i as usize] + 1e-15f32)).floor()
            }) as i32;
            unmask += INV_TABLE[id as usize] as i32;
            i += 4;
        }
        // Normalize and compensate for:
        // - 1/4 sample stride
        // - factor of 6 baked into INV_TABLE
        unmask = 64 * unmask * 4 / (6 * (len2 - 17));
        if unmask > mask_metric {
            *tf_chan = c;
            mask_metric = unmask;
        }
        c += 1;
    }
    is_transient = (mask_metric > 200) as i32;
    // Prevent the transient detector from confusing the partial cycle of a
    // very low frequency tone with a transient.
    if toneishness > 0.98 && tone_freq < 0.026 {
        is_transient = 0;
        mask_metric = 0;
    }
    // For low bitrates, classify weak transients separately so later stages
    // can avoid partial-collapse artifacts.
    if allow_weak_transients != 0 && is_transient != 0 && mask_metric < 600 {
        is_transient = 0;
        *weak_transient = 1;
    }
    // Arbitrary metric used for VBR boost behavior.
    let tf_max: f32 = if 0 as f32 > celt_sqrt((27 * mask_metric) as f32) - 42_f32 {
        0 as f32
    } else {
        celt_sqrt((27 * mask_metric) as f32) - 42_f32
    };
    *tf_estimate = (if 0 as f64
        > (0.0069f64 as f32 * (if 163_f32 < tf_max { 163_f32 } else { tf_max })) as f64 - 0.139f64
    {
        0 as f64
    } else {
        (0.0069f64 as f32 * (if 163_f32 < tf_max { 163_f32 } else { tf_max })) as f64 - 0.139f64
    })
    // here, a 64-bit sqrt __should__ be used
    .sqrt() as f32;
    is_transient
}
///
/// Looks for sudden increases in band energy to decide whether to patch
/// the transient decision.
/// Upstream C: celt/celt_encoder.c:patch_transient_decision
fn patch_transient_decision(
    new_e: &[f32],
    old_e: &[f32],
    nb_ebands: i32,
    start: i32,
    end: i32,
    channels: i32,
) -> i32 {
    let mut i: i32;
    let mut c: i32;
    let mut mean_diff: f32 = 0 as f32;
    let mut spread_old: [f32; 26] = [0.; 26];
    // Apply an aggressive (-6 dB/Bark) spreading to old-frame energies to
    // avoid false positives caused by irrelevant narrowband peaks.
    if channels == 1 {
        spread_old[start as usize] = old_e[start as usize];
        i = start + 1;
        while i < end {
            spread_old[i as usize] = if spread_old[(i - 1) as usize] - 1.0f32 > old_e[i as usize] {
                spread_old[(i - 1) as usize] - 1.0f32
            } else {
                old_e[i as usize]
            };
            i += 1;
        }
    } else {
        spread_old[start as usize] = if old_e[start as usize] > old_e[(start + nb_ebands) as usize]
        {
            old_e[start as usize]
        } else {
            old_e[(start + nb_ebands) as usize]
        };
        i = start + 1;
        while i < end {
            spread_old[i as usize] = if spread_old[(i - 1) as usize] - 1.0f32
                > (if old_e[i as usize] > old_e[(i + nb_ebands) as usize] {
                    old_e[i as usize]
                } else {
                    old_e[(i + nb_ebands) as usize]
                }) {
                spread_old[(i - 1) as usize] - 1.0f32
            } else if old_e[i as usize] > old_e[(i + nb_ebands) as usize] {
                old_e[i as usize]
            } else {
                old_e[(i + nb_ebands) as usize]
            };
            i += 1;
        }
    }
    i = end - 2;
    while i >= start {
        spread_old[i as usize] = if spread_old[i as usize] > spread_old[(i + 1) as usize] - 1.0f32 {
            spread_old[i as usize]
        } else {
            spread_old[(i + 1) as usize] - 1.0f32
        };
        i -= 1;
    }
    // Compute mean increase versus spread old energies.
    c = 0;
    loop {
        i = if 2 > start { 2 } else { start };
        while i < end - 1 {
            let x1: f32 = if 0 as f32 > new_e[(i + c * nb_ebands) as usize] {
                0 as f32
            } else {
                new_e[(i + c * nb_ebands) as usize]
            };
            let x2: f32 = if 0 as f32 > spread_old[i as usize] {
                0 as f32
            } else {
                spread_old[i as usize]
            };
            mean_diff += if 0 as f32 > x1 - x2 {
                0 as f32
            } else {
                x1 - x2
            };
            i += 1;
        }
        c += 1;
        if c >= channels {
            break;
        }
    }
    mean_diff /= (channels * (end - 1 - (if 2 > start { 2 } else { start }))) as f32;
    (mean_diff > 1.0f32) as i32
}
/// Upstream C: celt/celt_encoder.c:compute_mdcts
fn compute_mdcts(
    mode: &OpusCustomMode,
    short_blocks: i32,
    in_0: &mut [f32],
    out: &mut [f32],
    channels: i32,
    coded_channels: i32,
    lm: i32,
    upsample: i32,
) {
    let overlap: i32 = mode.overlap as i32;
    let n_i32: i32;
    let blocks: i32;
    let shift: i32;
    let mut i: i32;
    let mut b: i32;
    let mut c: i32;
    if short_blocks != 0 {
        blocks = short_blocks;
        n_i32 = mode.short_mdct_size;
        shift = mode.max_lm;
    } else {
        blocks = 1;
        n_i32 = mode.short_mdct_size << lm;
        shift = mode.max_lm - lm;
    }
    c = 0;
    loop {
        b = 0;
        while b < blocks {
            /* Interleaving the sub-frames while doing the MDCTs */
            let in_base = (c * (blocks * n_i32 + overlap) + b * n_i32) as usize;
            let in_len = (n_i32 + overlap) as usize;
            let out_base = (b + c * n_i32 * blocks) as usize;
            let out_len = (n_i32 * blocks) as usize;

            mdct_forward(
                &mode.mdct,
                &in_0[in_base..in_base + in_len],
                &mut out[out_base..out_base + out_len],
                mode.window,
                overlap as usize,
                shift as usize,
                blocks as usize,
            );
            b += 1;
        }
        c += 1;
        if c >= coded_channels {
            break;
        }
    }
    if coded_channels == 2 && channels == 1 {
        i = 0;
        while i < blocks * n_i32 {
            out[i as usize] =
                0.5f32 * out[i as usize] + 0.5f32 * out[(blocks * n_i32 + i) as usize];
            i += 1;
        }
    }
    if upsample != 1 {
        c = 0;
        loop {
            let bound: i32 = blocks * n_i32 / upsample;
            i = 0;
            while i < bound {
                out[(c * blocks * n_i32 + i) as usize] *= upsample as f32;
                i += 1;
            }
            let base = (c * blocks * n_i32 + bound) as usize;
            let len = (blocks * n_i32 - bound) as usize;
            out[base..base + len].fill(0.0);
            c += 1;
            if c >= channels {
                break;
            }
        }
    }
}
/// Upstream C: celt/celt_encoder.c:celt_preemphasis
fn celt_preemphasis(
    pcmp: &[f32],
    inp: &mut [f32],
    n_i32: i32,
    coded_channels: i32,
    upsample: i32,
    coef: &[f32],
    mem: &mut f32,
    clip: i32,
) {
    let mut i: i32;

    let mut m: f32;

    let coef0: f32 = coef[0];
    m = *mem;
    if coef[1] == 0 as f32 && upsample == 1 && clip == 0 {
        i = 0;
        while i < n_i32 {
            let x: f32 = pcmp[(coded_channels * i) as usize] * CELT_SIG_SCALE;
            inp[i as usize] = x - m;
            m = coef0 * x;
            i += 1;
        }
        *mem = m;
        return;
    }
    let n_upsampled: i32 = n_i32 / upsample;
    if upsample != 1 {
        inp[..n_i32 as usize].fill(0.0);
    }
    i = 0;
    while i < n_upsampled {
        inp[(i * upsample) as usize] = pcmp[(coded_channels * i) as usize] * CELT_SIG_SCALE;
        i += 1;
    }
    if clip != 0 {
        i = 0;
        while i < n_upsampled {
            inp[(i * upsample) as usize] = if -65536.0f32
                > (if 65536.0f32 < inp[(i * upsample) as usize] {
                    65536.0f32
                } else {
                    inp[(i * upsample) as usize]
                }) {
                -65536.0f32
            } else if 65536.0f32 < inp[(i * upsample) as usize] {
                65536.0f32
            } else {
                inp[(i * upsample) as usize]
            };
            i += 1;
        }
    }
    i = 0;
    while i < n_i32 {
        let x_0: f32 = inp[i as usize];
        inp[i as usize] = x_0 - m;
        m = coef0 * x_0;
        i += 1;
    }
    *mem = m;
}
/// Upstream C: celt/celt_encoder.c:l1_metric
fn l1_metric(tmp: &[f32], n_i32: i32, lm: i32, bias: f32) -> f32 {
    let mut l1: f32 = 0 as f32;
    let mut i: i32 = 0;
    while i < n_i32 {
        l1 += tmp[i as usize].abs();
        i += 1;
    }
    l1 = l1 + lm as f32 * bias * l1;
    l1
}
/// Upstream C: celt/celt_encoder.c:tf_analysis
#[allow(clippy::too_many_arguments)]
fn tf_analysis(
    m: &OpusCustomMode,
    len: i32,
    is_transient: i32,
    tf_res: &mut [i32],
    lambda: i32,
    x: &[f32],
    n0: i32,
    lm: i32,
    tf_estimate: f32,
    tf_chan: i32,
    importance: &[i32],
) -> i32 {
    let mut i: i32;
    let mut cost0: i32;
    let mut cost1: i32;
    let mut sel: i32;
    let mut selcost: [i32; 2] = [0; 2];
    let mut tf_select: i32;

    let bias: f32 = 0.04f32
        * (if -0.25f32 > 0.5f32 - tf_estimate {
            -0.25f32
        } else {
            0.5f32 - tf_estimate
        });
    // len = nb_ebands, max 21 (std) + 14 (QEXT) = 35.
    const MAX_TF_BANDS: usize = 40;
    debug_assert!((len as usize) <= MAX_TF_BANDS);
    let mut metric = [0i32; MAX_TF_BANDS];
    let band_size =
        ((m.e_bands[len as usize] as i32 - m.e_bands[(len - 1) as usize] as i32) << lm) as usize;
    // Last band size * m_stride; max ~128 (48kHz) or ~256 (QEXT).
    const MAX_BAND_TMP: usize = 256;
    debug_assert!(band_size <= MAX_BAND_TMP);
    let mut tmp = [0.0f32; MAX_BAND_TMP];
    let mut tmp_1 = [0.0f32; MAX_BAND_TMP];
    let mut path0 = [0i32; MAX_TF_BANDS];
    let mut path1 = [0i32; MAX_TF_BANDS];
    i = 0;
    while i < len {
        let mut k: i32;

        let mut l1: f32;
        let mut best_l1: f32;
        let mut best_level: i32 = 0;
        let n_i32: i32 = (m.e_bands[(i + 1) as usize] as i32 - m.e_bands[i as usize] as i32) << lm;
        let narrow: i32 =
            (m.e_bands[(i + 1) as usize] as i32 - m.e_bands[i as usize] as i32 == 1) as i32;
        let x_offset = (tf_chan * n0 + ((m.e_bands[i as usize] as i32) << lm)) as usize;
        tmp[..n_i32 as usize].copy_from_slice(&x[x_offset..x_offset + n_i32 as usize]);
        l1 = l1_metric(&tmp, n_i32, if is_transient != 0 { lm } else { 0 }, bias);
        best_l1 = l1;
        if is_transient != 0 && narrow == 0 {
            tmp_1[..n_i32 as usize].copy_from_slice(&tmp[..n_i32 as usize]);
            haar1(&mut tmp_1, n_i32 >> lm, (1) << lm);
            l1 = l1_metric(&tmp_1, n_i32, lm + 1, bias);
            if l1 < best_l1 {
                best_l1 = l1;
                best_level = -1;
            }
        }
        k = 0;
        while k < lm + !(is_transient != 0 || narrow != 0) as i32 {
            let blocks: i32 = if is_transient != 0 { lm - k - 1 } else { k + 1 };
            haar1(&mut tmp, n_i32 >> k, (1) << k);
            l1 = l1_metric(&tmp, n_i32, blocks, bias);
            if l1 < best_l1 {
                best_l1 = l1;
                best_level = k + 1;
            }
            k += 1;
        }
        if is_transient != 0 {
            metric[i as usize] = 2 * best_level;
        } else {
            metric[i as usize] = -(2) * best_level;
        }
        if narrow != 0 && (metric[i as usize] == 0 || metric[i as usize] == -(2) * lm) {
            metric[i as usize] -= 1;
        }
        i += 1;
    }
    tf_select = 0;
    sel = 0;
    while sel < 2 {
        cost0 = importance[0]
            * (metric[0]
                - 2 * TF_SELECT_TABLE[lm as usize][(4 * is_transient + 2 * sel) as usize] as i32)
                .abs();
        cost1 = importance[0]
            * (metric[0]
                - 2 * TF_SELECT_TABLE[lm as usize][(4 * is_transient + 2 * sel + 1) as usize]
                    as i32)
                .abs()
            + (if is_transient != 0 { 0 } else { lambda });
        i = 1;
        while i < len {
            let curr0: i32 = if cost0 < cost1 + lambda {
                cost0
            } else {
                cost1 + lambda
            };
            let curr1: i32 = if cost0 + lambda < cost1 {
                cost0 + lambda
            } else {
                cost1
            };
            cost0 = curr0
                + importance[i as usize]
                    * (metric[i as usize]
                        - 2 * TF_SELECT_TABLE[lm as usize][(4 * is_transient + 2 * sel) as usize]
                            as i32)
                        .abs();
            cost1 = curr1
                + importance[i as usize]
                    * (metric[i as usize]
                        - 2 * TF_SELECT_TABLE[lm as usize]
                            [(4 * is_transient + 2 * sel + 1) as usize]
                            as i32)
                        .abs();
            i += 1;
        }
        cost0 = if cost0 < cost1 { cost0 } else { cost1 };
        selcost[sel as usize] = cost0;
        sel += 1;
    }
    if selcost[1_usize] < selcost[0_usize] && is_transient != 0 {
        tf_select = 1;
    }
    cost0 = importance[0]
        * (metric[0]
            - 2 * TF_SELECT_TABLE[lm as usize][(4 * is_transient + 2 * tf_select) as usize] as i32)
            .abs();
    cost1 = importance[0]
        * (metric[0]
            - 2 * TF_SELECT_TABLE[lm as usize][(4 * is_transient + 2 * tf_select + 1) as usize]
                as i32)
            .abs()
        + (if is_transient != 0 { 0 } else { lambda });
    i = 1;
    while i < len {
        let curr0_0: i32;
        let curr1_0: i32;
        let mut from0: i32;
        let mut from1: i32;
        from0 = cost0;
        from1 = cost1 + lambda;
        if from0 < from1 {
            curr0_0 = from0;
            path0[i as usize] = 0;
        } else {
            curr0_0 = from1;
            path0[i as usize] = 1;
        }
        from0 = cost0 + lambda;
        from1 = cost1;
        if from0 < from1 {
            curr1_0 = from0;
            path1[i as usize] = 0;
        } else {
            curr1_0 = from1;
            path1[i as usize] = 1;
        }
        cost0 = curr0_0
            + importance[i as usize]
                * (metric[i as usize]
                    - 2 * TF_SELECT_TABLE[lm as usize][(4 * is_transient + 2 * tf_select) as usize]
                        as i32)
                    .abs();
        cost1 = curr1_0
            + importance[i as usize]
                * (metric[i as usize]
                    - 2 * TF_SELECT_TABLE[lm as usize]
                        [(4 * is_transient + 2 * tf_select + 1) as usize]
                        as i32)
                    .abs();
        i += 1;
    }
    tf_res[(len - 1) as usize] = if cost0 < cost1 { 0 } else { 1 };
    i = len - 2;
    while i >= 0 {
        if tf_res[(i + 1) as usize] == 1 {
            tf_res[i as usize] = path1[(i + 1) as usize];
        } else {
            tf_res[i as usize] = path0[(i + 1) as usize];
        }
        i -= 1;
    }
    tf_select
}
/// Upstream C: celt/celt_encoder.c:tf_encode
fn tf_encode(
    start: i32,
    end: i32,
    is_transient: i32,
    tf_res: &mut [i32],
    lm: i32,
    mut tf_select: i32,
    enc: &mut EcEnc,
) {
    let mut curr: i32;
    let mut i: i32;

    let mut tf_changed: i32;
    let mut logp: i32;
    let mut budget: u32;
    let mut tell: u32;
    budget = enc.storage.wrapping_mul(8);
    tell = ec_tell(enc) as u32;
    logp = if is_transient != 0 { 2 } else { 4 };
    let tf_select_rsv: i32 =
        (lm > 0 && tell.wrapping_add(logp as u32).wrapping_add(1) <= budget) as i32;
    budget = budget.wrapping_sub(tf_select_rsv as u32);
    tf_changed = 0;
    curr = tf_changed;
    i = start;
    while i < end {
        if tell.wrapping_add(logp as u32) <= budget {
            ec_enc_bit_logp(enc, tf_res[i as usize] ^ curr, logp as u32);
            tell = ec_tell(enc) as u32;
            curr = tf_res[i as usize];
            tf_changed |= curr;
        } else {
            tf_res[i as usize] = curr;
        }
        logp = if is_transient != 0 { 4 } else { 5 };
        i += 1;
    }
    if tf_select_rsv != 0
        && TF_SELECT_TABLE[lm as usize][((4 * is_transient) + tf_changed) as usize] as i32
            != TF_SELECT_TABLE[lm as usize][(4 * is_transient + 2 + tf_changed) as usize] as i32
    {
        ec_enc_bit_logp(enc, tf_select, 1);
    } else {
        tf_select = 0;
    }
    i = start;
    while i < end {
        tf_res[i as usize] = TF_SELECT_TABLE[lm as usize]
            [(4 * is_transient + 2 * tf_select + tf_res[i as usize]) as usize]
            as i32;
        i += 1;
    }
}
/// Upstream C: celt/celt_encoder.c:alloc_trim_analysis
#[allow(clippy::too_many_arguments)]
fn alloc_trim_analysis(
    m: &OpusCustomMode,
    x: &[f32],
    band_log_e: &[f32],
    end: i32,
    lm: i32,
    channels: i32,
    n0: i32,
    analysis: &AnalysisInfo,
    stereo_saving: &mut f32,
    tf_estimate: f32,
    intensity: i32,
    surround_trim: f32,
    equiv_rate: i32,
    _arch: Arch,
) -> i32 {
    let mut i: i32;
    let mut diff: f32 = 0 as f32;
    let mut c: i32;
    let mut trim_index: i32;
    let mut trim: f32 = 5.0f32;
    let log_xc: f32;
    let log_xc2: f32;
    if equiv_rate < 64000 {
        trim = 4.0f32;
    } else if equiv_rate < 80000 {
        let frac: i32 = (equiv_rate - 64000) >> 10;
        trim = 4.0f32 + 1.0f32 / 16.0f32 * frac as f32;
    }
    if channels == 2 {
        let mut sum: f32 = 0 as f32;
        let mut min_xc: f32;
        i = 0;
        while i < 8 {
            let band_off = ((m.e_bands[i as usize] as i32) << lm) as usize;
            let band_off2 = (n0 + ((m.e_bands[i as usize] as i32) << lm)) as usize;
            let band_len = ((m.e_bands[(i + 1) as usize] as i32 - m.e_bands[i as usize] as i32)
                << lm) as usize;
            let partial: f32 = celt_inner_prod(
                &x[band_off..band_off + band_len],
                &x[band_off2..band_off2 + band_len],
                band_len,
                _arch,
            );
            sum += partial;
            i += 1;
        }
        sum *= 1.0f32 / 8_f32;
        sum = if 1.0f32 < (sum).abs() {
            1.0f32
        } else {
            (sum).abs()
        };
        min_xc = sum;
        i = 8;
        while i < intensity {
            let band_off = ((m.e_bands[i as usize] as i32) << lm) as usize;
            let band_off2 = (n0 + ((m.e_bands[i as usize] as i32) << lm)) as usize;
            let band_len = ((m.e_bands[(i + 1) as usize] as i32 - m.e_bands[i as usize] as i32)
                << lm) as usize;
            let partial_0: f32 = celt_inner_prod(
                &x[band_off..band_off + band_len],
                &x[band_off2..band_off2 + band_len],
                band_len,
                _arch,
            );
            min_xc = if min_xc < (partial_0).abs() {
                min_xc
            } else {
                (partial_0).abs()
            };
            i += 1;
        }
        min_xc = if 1.0f32 < (min_xc).abs() {
            1.0f32
        } else {
            (min_xc).abs()
        };
        log_xc = celt_log2(1.001f32 - sum * sum);
        log_xc2 = if 0.5f32 * log_xc > celt_log2(1.001f32 - min_xc * min_xc) {
            0.5f32 * log_xc
        } else {
            celt_log2(1.001f32 - min_xc * min_xc)
        };
        trim += if -4.0f32 > 0.75f32 * log_xc {
            -4.0f32
        } else {
            0.75f32 * log_xc
        };
        *stereo_saving = if *stereo_saving + 0.25f32 < -(0.5f32 * log_xc2) {
            *stereo_saving + 0.25f32
        } else {
            -(0.5f32 * log_xc2)
        };
    }
    c = 0;
    loop {
        i = 0;
        while i < end - 1 {
            diff += band_log_e[(i + c * m.nb_ebands as i32) as usize] * (2 + 2 * i - end) as f32;
            i += 1;
        }
        c += 1;
        if c >= channels {
            break;
        }
    }
    diff /= (channels * (end - 1)) as f32;
    trim -= if -2.0f32
        > (if 2.0f32 < (diff + 1.0f32) / 6_f32 {
            2.0f32
        } else {
            (diff + 1.0f32) / 6_f32
        }) {
        -2.0f32
    } else if 2.0f32 < (diff + 1.0f32) / 6_f32 {
        2.0f32
    } else {
        (diff + 1.0f32) / 6_f32
    };
    trim -= surround_trim;
    trim -= 2_f32 * tf_estimate;
    if analysis.valid != 0 {
        trim -= if -2.0f32
            > (if 2.0f32 < 2.0f32 * (analysis.tonality_slope + 0.05f32) {
                2.0f32
            } else {
                2.0f32 * (analysis.tonality_slope + 0.05f32)
            }) {
            -2.0f32
        } else if 2.0f32 < 2.0f32 * (analysis.tonality_slope + 0.05f32) {
            2.0f32
        } else {
            2.0f32 * (analysis.tonality_slope + 0.05f32)
        };
    }
    trim_index = (0.5f32 + trim).floor() as i32;
    trim_index = if 0 > (if (10) < trim_index { 10 } else { trim_index }) {
        0
    } else if (10) < trim_index {
        10
    } else {
        trim_index
    };
    trim_index
}
/// Upstream C: celt/celt_encoder.c:stereo_analysis
fn stereo_analysis(m: &OpusCustomMode, x: &[f32], lm: i32, n0: i32) -> i32 {
    let mut i: i32;
    let mut thetas: i32;
    let mut sum_lr: f32 = EPSILON;
    let mut sum_ms: f32 = EPSILON;
    i = 0;
    while i < 13 {
        let mut j: i32;
        j = (m.e_bands[i as usize] as i32) << lm;
        while j < (m.e_bands[(i + 1) as usize] as i32) << lm {
            let l: f32 = x[j as usize];
            let r: f32 = x[(n0 + j) as usize];
            let m_ch: f32 = l + r;
            let s_ch: f32 = l - r;
            sum_lr += (l).abs() + (r).abs();
            sum_ms += (m_ch).abs() + (s_ch).abs();
            j += 1;
        }
        i += 1;
    }
    #[allow(clippy::approx_constant)]
    // Intentional: C reference uses 0.707107, not exact 1/sqrt(2)
    let frac_1_sqrt_2 = 0.707107f32;
    sum_ms *= frac_1_sqrt_2;
    thetas = 13;
    if lm <= 1 {
        thetas -= 8;
    }
    ((((m.e_bands[13] as i32) << (lm + 1)) + thetas) as f32 * sum_ms
        > ((m.e_bands[13] as i32) << (lm + 1)) as f32 * sum_lr) as i32
}
/// Upstream C: celt/celt_encoder.c:median_of_5
fn median_of_5(x: &[f32]) -> f32 {
    let mut t0: f32;
    let mut t1: f32;

    let mut t3: f32;
    let mut t4: f32;
    let t2: f32 = x[2];
    if x[0] > x[1] {
        t0 = x[1];
        t1 = x[0];
    } else {
        t0 = x[0];
        t1 = x[1];
    }
    if x[3] > x[4] {
        t3 = x[4];
        t4 = x[3];
    } else {
        t3 = x[3];
        t4 = x[4];
    }
    if t0 > t3 {
        std::mem::swap(&mut t0, &mut t3);
        std::mem::swap(&mut t1, &mut t4);
    }
    if t2 > t1 {
        if t1 < t3 {
            if t2 < t3 {
                t2
            } else {
                t3
            }
        } else if t4 < t1 {
            t4
        } else {
            t1
        }
    } else if t2 < t3 {
        if t1 < t3 {
            t1
        } else {
            t3
        }
    } else if t2 < t4 {
        t2
    } else {
        t4
    }
}
/// Upstream C: celt/celt_encoder.c:median_of_3
fn median_of_3(x: &[f32]) -> f32 {
    let t0: f32;
    let t1: f32;

    if x[0] > x[1] {
        t0 = x[1];
        t1 = x[0];
    } else {
        t0 = x[0];
        t1 = x[1];
    }
    let t2: f32 = x[2];
    if t1 < t2 {
        t1
    } else if t0 < t2 {
        t2
    } else {
        t0
    }
}
/// Upstream C: celt/celt_encoder.c:dynalloc_analysis
#[allow(clippy::too_many_arguments)]
fn dynalloc_analysis(
    band_log_e: &[f32],
    band_log_e2: &[f32],
    old_band_e: &[f32],
    nb_ebands: i32,
    start: i32,
    end: i32,
    channels: i32,
    offsets: &mut [i32],
    lsb_depth: i32,
    log_n: &[i16],
    is_transient: i32,
    vbr: i32,
    constrained_vbr: i32,
    e_bands: &[i16],
    lm: i32,
    effective_bytes: i32,
    tot_boost_: &mut i32,
    lfe: i32,
    surround_dynalloc: &[f32],
    analysis: &AnalysisInfo,
    importance: &mut [i32],
    spread_weight: &mut [i32],
    tone_freq: f32,
    toneishness: f32,
) -> f32 {
    let mut i: i32;
    let mut c: i32;
    let mut tot_boost: i32 = 0;
    let mut max_depth: f32;
    // channels * nb_ebands max: 2 * 35 = 70.
    const MAX_C_BANDS: usize = 80;
    debug_assert!(((channels * nb_ebands) as usize) <= MAX_C_BANDS);
    let mut follower = [0.0f32; MAX_C_BANDS];
    let mut noise_floor = [0.0f32; MAX_C_BANDS];
    offsets[..nb_ebands as usize].fill(0);
    max_depth = -31.9f32;
    i = 0;
    while i < end {
        noise_floor[i as usize] =
            0.0625f32 * log_n[i as usize] as f32 + 0.5f32 + (9 - lsb_depth) as f32
                - E_MEANS[i as usize]
                + 0.0062f64 as f32 * ((i + 5) * (i + 5)) as f32;
        i += 1;
    }
    c = 0;
    loop {
        i = 0;
        while i < end {
            max_depth =
                if max_depth > band_log_e[(c * nb_ebands + i) as usize] - noise_floor[i as usize] {
                    max_depth
                } else {
                    band_log_e[(c * nb_ebands + i) as usize] - noise_floor[i as usize]
                };
            i += 1;
        }
        c += 1;
        if c >= channels {
            break;
        }
    }
    const MAX_BANDS_DA: usize = 40;
    debug_assert!((nb_ebands as usize) <= MAX_BANDS_DA);
    let mut mask = [0.0f32; MAX_BANDS_DA];
    let mut sig = [0.0f32; MAX_BANDS_DA];
    i = 0;
    while i < end {
        mask[i as usize] = band_log_e[i as usize] - noise_floor[i as usize];
        i += 1;
    }
    if channels == 2 {
        i = 0;
        while i < end {
            mask[i as usize] = if mask[i as usize]
                > band_log_e[(nb_ebands + i) as usize] - noise_floor[i as usize]
            {
                mask[i as usize]
            } else {
                band_log_e[(nb_ebands + i) as usize] - noise_floor[i as usize]
            };
            i += 1;
        }
    }
    sig[..end as usize].copy_from_slice(&mask[..end as usize]);
    i = 1;
    while i < end {
        mask[i as usize] = if mask[i as usize] > mask[(i - 1) as usize] - 2.0f32 {
            mask[i as usize]
        } else {
            mask[(i - 1) as usize] - 2.0f32
        };
        i += 1;
    }
    i = end - 2;
    while i >= 0 {
        mask[i as usize] = if mask[i as usize] > mask[(i + 1) as usize] - 3.0f32 {
            mask[i as usize]
        } else {
            mask[(i + 1) as usize] - 3.0f32
        };
        i -= 1;
    }
    i = 0;
    while i < end {
        let smr: f32 = sig[i as usize]
            - (if (if 0 as f32 > max_depth - 12.0f32 {
                0 as f32
            } else {
                max_depth - 12.0f32
            }) > mask[i as usize]
            {
                if 0 as f32 > max_depth - 12.0f32 {
                    0 as f32
                } else {
                    max_depth - 12.0f32
                }
            } else {
                mask[i as usize]
            });
        let shift: i32 = if (5)
            < (if 0 > -((0.5f32 + smr).floor() as i32) {
                0
            } else {
                -((0.5f32 + smr).floor() as i32)
            }) {
            5
        } else if 0 > -((0.5f32 + smr).floor() as i32) {
            0
        } else {
            -((0.5f32 + smr).floor() as i32)
        };
        spread_weight[i as usize] = 32 >> shift;
        i += 1;
    }
    // nb_ebands max is 21; use stack buffer.
    let mut band_log_e3 = [0.0_f32; 24];
    if effective_bytes >= (30 + 5 * lm) && lfe == 0 {
        let mut last: i32 = 0;
        c = 0;
        loop {
            let mut tmp: f32;
            let fb = (c * nb_ebands) as usize;
            band_log_e3[..end as usize].copy_from_slice(&band_log_e2[fb..fb + end as usize]);
            if lm == 0 {
                // For 2.5 ms frames, the first 8 bands have just one bin, so the
                // energy is highly unreliable (high variance). For that reason,
                // we take the max with the previous energy so that at least 2 bins
                // are getting used.
                for i in 0..std::cmp::min(8, end as usize) {
                    band_log_e3[i] = if band_log_e2[(c * nb_ebands) as usize + i]
                        > old_band_e[(c * nb_ebands) as usize + i]
                    {
                        band_log_e2[(c * nb_ebands) as usize + i]
                    } else {
                        old_band_e[(c * nb_ebands) as usize + i]
                    };
                }
            }
            follower[fb] = band_log_e3[0];
            i = 1;
            while i < end {
                if band_log_e3[i as usize] > band_log_e3[(i - 1) as usize] + 0.5f32 {
                    last = i;
                }
                follower[fb + i as usize] =
                    if follower[fb + (i - 1) as usize] + 1.5f32 < band_log_e3[i as usize] {
                        follower[fb + (i - 1) as usize] + 1.5f32
                    } else {
                        band_log_e3[i as usize]
                    };
                i += 1;
            }
            i = last - 1;
            while i >= 0 {
                follower[fb + i as usize] = if follower[fb + i as usize]
                    < (if follower[fb + (i + 1) as usize] + 2.0f32 < band_log_e3[i as usize] {
                        follower[fb + (i + 1) as usize] + 2.0f32
                    } else {
                        band_log_e3[i as usize]
                    }) {
                    follower[fb + i as usize]
                } else if follower[fb + (i + 1) as usize] + 2.0f32 < band_log_e3[i as usize] {
                    follower[fb + (i + 1) as usize] + 2.0f32
                } else {
                    band_log_e3[i as usize]
                };
                i -= 1;
            }
            let offset: f32 = 1.0f32;
            i = 2;
            while i < end - 2 {
                let med = median_of_5(&band_log_e3[(i - 2) as usize..(i + 3) as usize]) - offset;
                follower[fb + i as usize] = if follower[fb + i as usize] > med {
                    follower[fb + i as usize]
                } else {
                    med
                };
                i += 1;
            }
            tmp = median_of_3(&band_log_e3[0..3]) - offset;
            follower[fb] = if follower[fb] > tmp {
                follower[fb]
            } else {
                tmp
            };
            follower[fb + 1] = if follower[fb + 1] > tmp {
                follower[fb + 1]
            } else {
                tmp
            };
            tmp = median_of_3(&band_log_e3[(end - 3) as usize..end as usize]) - offset;
            follower[fb + (end - 2) as usize] = if follower[fb + (end - 2) as usize] > tmp {
                follower[fb + (end - 2) as usize]
            } else {
                tmp
            };
            follower[fb + (end - 1) as usize] = if follower[fb + (end - 1) as usize] > tmp {
                follower[fb + (end - 1) as usize]
            } else {
                tmp
            };
            i = 0;
            while i < end {
                follower[fb + i as usize] = if follower[fb + i as usize] > noise_floor[i as usize] {
                    follower[fb + i as usize]
                } else {
                    noise_floor[i as usize]
                };
                i += 1;
            }
            c += 1;
            if c >= channels {
                break;
            }
        }
        if channels == 2 {
            i = start;
            while i < end {
                follower[(nb_ebands + i) as usize] =
                    if follower[(nb_ebands + i) as usize] > follower[i as usize] - 4.0f32 {
                        follower[(nb_ebands + i) as usize]
                    } else {
                        follower[i as usize] - 4.0f32
                    };
                follower[i as usize] =
                    if follower[i as usize] > follower[(nb_ebands + i) as usize] - 4.0f32 {
                        follower[i as usize]
                    } else {
                        follower[(nb_ebands + i) as usize] - 4.0f32
                    };
                follower[i as usize] = 0.5f32
                    * ((if 0 as f32 > band_log_e[i as usize] - follower[i as usize] {
                        0 as f32
                    } else {
                        band_log_e[i as usize] - follower[i as usize]
                    }) + (if 0 as f32
                        > band_log_e[(nb_ebands + i) as usize] - follower[(nb_ebands + i) as usize]
                    {
                        0 as f32
                    } else {
                        band_log_e[(nb_ebands + i) as usize] - follower[(nb_ebands + i) as usize]
                    }));
                i += 1;
            }
        } else {
            i = start;
            while i < end {
                follower[i as usize] = if 0 as f32 > band_log_e[i as usize] - follower[i as usize] {
                    0 as f32
                } else {
                    band_log_e[i as usize] - follower[i as usize]
                };
                i += 1;
            }
        }
        i = start;
        while i < end {
            follower[i as usize] = if follower[i as usize] > surround_dynalloc[i as usize] {
                follower[i as usize]
            } else {
                surround_dynalloc[i as usize]
            };
            i += 1;
        }
        i = start;
        while i < end {
            importance[i as usize] = (0.5f32
                + 13.0
                    * celt_exp2(if follower[i as usize] < 4.0f32 {
                        follower[i as usize]
                    } else {
                        4.0f32
                    }))
            .floor() as i32;
            i += 1;
        }
        if (vbr == 0 || constrained_vbr != 0) && is_transient == 0 {
            i = start;
            while i < end {
                follower[i as usize] *= 0.5f32;
                i += 1;
            }
        }
        i = start;
        while i < end {
            if i < 8 {
                follower[i as usize] *= 2_f32;
            }
            if i >= 12 {
                follower[i as usize] *= 0.5f32;
            }
            i += 1;
        }
        // Compensate for Opus' under-allocation on tones.
        if toneishness > 0.98 {
            let freq_bin = (0.5 + tone_freq as f64 * 120.0 / std::f64::consts::PI) as i32;
            for i in start..end {
                if freq_bin >= e_bands[i as usize] as i32
                    && freq_bin <= e_bands[(i + 1) as usize] as i32
                {
                    follower[i as usize] += 2.0;
                }
                if freq_bin >= e_bands[i as usize] as i32 - 1
                    && freq_bin <= e_bands[(i + 1) as usize] as i32 + 1
                {
                    follower[i as usize] += 1.0;
                }
                if freq_bin >= e_bands[i as usize] as i32 - 2
                    && freq_bin <= e_bands[(i + 1) as usize] as i32 + 2
                {
                    follower[i as usize] += 1.0;
                }
                if freq_bin >= e_bands[i as usize] as i32 - 3
                    && freq_bin <= e_bands[(i + 1) as usize] as i32 + 3
                {
                    follower[i as usize] += 0.5;
                }
            }
            if freq_bin >= e_bands[end as usize] as i32 {
                follower[(end - 1) as usize] += 2.0;
                follower[(end - 2) as usize] += 1.0;
            }
        }
        if analysis.valid != 0 {
            i = start;
            while i < (if (19) < end { 19 } else { end }) {
                follower[i as usize] +=
                    1.0f32 / 64.0f32 * analysis.leak_boost[i as usize] as i32 as f32;
                i += 1;
            }
        }
        i = start;
        while i < end {
            let boost: i32;
            let boost_bits: i32;
            follower[i as usize] = if follower[i as usize] < 4_f32 {
                follower[i as usize]
            } else {
                4_f32
            };
            let width: i32 =
                (channels * (e_bands[(i + 1) as usize] as i32 - e_bands[i as usize] as i32)) << lm;
            if width < 6 {
                boost = follower[i as usize] as i32;
                boost_bits = (boost * width) << BITRES;
            } else if width > 48 {
                boost = (follower[i as usize] * 8_f32) as i32;
                boost_bits = ((boost * width) << BITRES) / 8;
            } else {
                boost = (follower[i as usize] * width as f32 / 6_f32) as i32;
                boost_bits = (boost * 6) << BITRES;
            }
            if (vbr == 0 || constrained_vbr != 0 && is_transient == 0)
                && (tot_boost + boost_bits) >> BITRES >> 3 > 2 * effective_bytes / 3
            {
                let cap: i32 = (2 * effective_bytes / 3) << BITRES << 3;
                offsets[i as usize] = cap - tot_boost;
                tot_boost = cap;
                break;
            } else {
                offsets[i as usize] = boost;
                tot_boost += boost_bits;
                i += 1;
            }
        }
    } else {
        i = start;
        while i < end {
            importance[i as usize] = 13;
            i += 1;
        }
    }
    *tot_boost_ = tot_boost;
    max_depth
}
/// 2nd-order LPC analysis using the forward/backward covariance method.
/// Returns `true` on failure (ill-conditioned).
///
/// Upstream C: celt/celt_encoder.c:tone_lpc
fn tone_lpc(x: &[f32], len: usize, delay: usize, lpc: &mut [f32; 2]) -> bool {
    debug_assert!(len > 2 * delay);
    // Compute forward correlations.
    let mut r00: f32 = 0.0;
    let mut r01: f32 = 0.0;
    let mut r02: f32 = 0.0;
    for i in 0..len - 2 * delay {
        r00 += x[i] * x[i];
        r01 += x[i] * x[i + delay];
        r02 += x[i] * x[i + 2 * delay];
    }
    let mut edges: f32 = 0.0;
    for i in 0..delay {
        edges += x[len + i - 2 * delay] * x[len + i - 2 * delay] - x[i] * x[i];
    }
    let r11 = r00 + edges;
    edges = 0.0;
    for i in 0..delay {
        edges += x[len + i - delay] * x[len + i - delay] - x[i + delay] * x[i + delay];
    }
    let r22 = r11 + edges;
    edges = 0.0;
    for i in 0..delay {
        edges += x[len + i - 2 * delay] * x[len + i - delay] - x[i] * x[i + delay];
    }
    let r12 = r01 + edges;
    // Reverse and sum to get the backward contribution.
    // C: R00=r00+r22, R01=r01+r12, R11=2*r11, R02=2*r02, R12=r12+r01, R22=r00+r22
    // Note: R01 == R12, R00 == R22.
    let r00 = r00 + r22;
    let r01 = r01 + r12;
    let r11 = 2.0 * r11;
    let r02 = 2.0 * r02;
    // r12_combined = r12 + r01_original, but since r01_combined = r01_orig + r12,
    // we have r12_combined == r01_combined.

    // Solve A*x=b, where A=[r00, r01; r01, r11] and b=[r02; r12].
    // Since r12_combined == r01, we use r01 for both.
    let den = r00 * r11 - r01 * r01;
    if den < 0.001 * r00 * r11 {
        return true; // fail
    }
    let num1 = r02 * r11 - r01 * r01; // r01 * r12, but r12 == r01
    if num1 >= den {
        lpc[1] = 1.0;
    } else if num1 <= -den {
        lpc[1] = -1.0;
    } else {
        lpc[1] = num1 / den;
    }
    let num0 = r00 * r01 - r02 * r01; // r00 * r12 - r02 * r01, but r12 == r01
    if 0.5 * num0 >= den {
        lpc[0] = 1.999999;
    } else if 0.5 * num0 <= -den {
        lpc[0] = -1.999999;
    } else {
        lpc[0] = num0 / den;
    }
    false // success
}

/// Detects pure or nearly pure tones to prevent them from causing
/// problems with the encoder.
///
/// Upstream C: celt/celt_encoder.c:tone_detect
fn tone_detect(
    input: &[f32],
    coded_channels: i32,
    n_i32: i32,
    toneishness: &mut f32,
    fs: i32,
) -> f32 {
    let n = n_i32 as usize;
    let mut delay: usize = 1;
    let mut lpc = [0.0f32; 2];
    // n_i32 + overlap max: 1920 + 240 = 2160 (QEXT 96kHz); use stack buffer.
    const MAX_TONE: usize = 2400;
    debug_assert!(n <= MAX_TONE);
    let mut x = [0.0_f32; MAX_TONE];
    // Shift by SIG_SHIFT+2 (+3 for stereo) to account for HF gain from the
    // preemphasis filter. In float build this reduces to averaging channels.
    if coded_channels == 2 {
        for i in 0..n {
            x[i] = (input[i] * 0.5) + (input[i + n] * 0.5);
        }
    } else {
        x[..n].copy_from_slice(&input[..n]);
    }
    let mut fail = tone_lpc(&x, n, delay, &mut lpc);
    // If our LPC filter resonates too close to DC, retry with down-sampling.
    while delay <= (fs / 3000) as usize && (fail || (lpc[0] > 1.0 && lpc[1] < 0.0)) {
        delay *= 2;
        fail = tone_lpc(&x, n, delay, &mut lpc);
    }
    // Check that our filter has complex roots.
    if !fail && lpc[0] * lpc[0] + 3.999999 * lpc[1] < 0.0 {
        // Squared radius of the poles.
        *toneishness = -lpc[1];
        (0.5 * lpc[0]).acos() / delay as f32
    } else {
        *toneishness = 0.0;
        -1.0
    }
}

/// Upstream C: celt/celt_encoder.c:run_prefilter
#[allow(clippy::approx_constant)]
#[allow(clippy::too_many_arguments)]
fn run_prefilter(
    st: &mut OpusCustomEncoder,
    in_0: &mut [f32],
    coded_channels: i32,
    n_i32: i32,
    prefilter_tapset: i32,
    pitch: &mut i32,
    gain: &mut f32,
    qgain: &mut i32,
    enabled: i32,
    tf_estimate: f32,
    nb_available_bytes: i32,
    analysis: &AnalysisInfo,
    tone_freq: f32,
    toneishness: f32,
) -> i32 {
    #[cfg(feature = "qext")]
    let qext_scale = st.qext_scale;
    #[cfg(not(feature = "qext"))]
    let qext_scale = 1;
    let mut pitch_index: i32;
    let mut gain1: f32;
    let mut pf_threshold: f32;
    let mut pf_on: i32;
    let mut qg: i32;
    let mode = st.mode;
    let overlap = mode.overlap as i32;
    let max_period = COMBFILTER_MAXPERIOD * qext_scale;
    let min_period = COMBFILTER_MINPERIOD * qext_scale;
    let pre_chan_len = (n_i32 + max_period) as usize;
    // coded_channels * (n_i32 + max_period) max: 2 * (1920 + 2048) = 7936.
    const MAX_PRE: usize = 8000;
    debug_assert!((coded_channels as usize) * pre_chan_len <= MAX_PRE);
    let mut _pre = [0.0f32; MAX_PRE];
    // pre[c] starts at c * pre_chan_len in _pre
    for c in 0..coded_channels as usize {
        let pre_base = c * pre_chan_len;
        let max_period_u = max_period as usize;
        _pre[pre_base..pre_base + max_period_u]
            .copy_from_slice(&st.prefilter_mem[c * max_period_u..(c + 1) * max_period_u]);
        let in_src = c * (n_i32 + overlap) as usize + overlap as usize;
        _pre[pre_base + max_period_u..pre_base + max_period_u + n_i32 as usize]
            .copy_from_slice(&in_0[in_src..in_src + n_i32 as usize]);
    }
    if enabled != 0 && toneishness > 0.99 {
        // If we detect that the signal is dominated by a single tone, don't rely
        // on the standard pitch estimator, as it can become unreliable.
        let mut multiple = 1i32;
        let mut tf = tone_freq * qext_scale as f32;
        // Using aliased version of the postfilter above 24 kHz.
        // Threshold is purposely slightly above pi to avoid triggering for fs=48kHz.
        if tf >= 3.1416f32 {
            tf = 3.141593f32 - tf;
        }
        // If the pitch is too high for our post-filter, apply pitch doubling
        // until we can get something that fits.
        while tf >= multiple as f32 * 0.39 {
            multiple += 1;
        }
        if tf > 0.006148 {
            pitch_index = ((0.5 + 2.0 * std::f64::consts::PI * multiple as f64 / tf as f64) as i32)
                .min(COMBFILTER_MAXPERIOD - 2);
        } else {
            // If the pitch is too low, using a very high pitch will actually give
            // us an improvement due to the DC component of the filter.
            pitch_index = COMBFILTER_MINPERIOD;
        }
        gain1 = 0.75;
    } else if enabled != 0 && st.complexity >= 5 {
        // (max_period + n_i32) >> 1 max: (2048 + 1920) / 2 = 1984.
        const MAX_PITCH_BUF: usize = 2000;
        let pitch_buf_len = ((max_period + n_i32) >> 1) as usize;
        debug_assert!(pitch_buf_len <= MAX_PITCH_BUF);
        let mut pitch_buf = [0.0f32; MAX_PITCH_BUF];
        {
            let ds_len = (max_period + n_i32) as usize;
            let ch0 = &_pre[..ds_len];
            if coded_channels == 2 {
                let ch1 = &_pre[pre_chan_len..pre_chan_len + ds_len];
                pitch_downsample(
                    &[ch0, ch1],
                    &mut pitch_buf[..pitch_buf_len],
                    pitch_buf_len,
                    2,
                    st.arch,
                );
            } else {
                pitch_downsample(
                    &[ch0],
                    &mut pitch_buf[..pitch_buf_len],
                    pitch_buf_len,
                    2,
                    st.arch,
                );
            }
        }
        // Don't search over the first/last ~1.5 octaves because short-term
        // correlation creates too many false positives there.
        pitch_index = pitch_search(
            &pitch_buf[(max_period >> 1) as usize..],
            pitch_buf.as_slice(),
            n_i32,
            max_period - 3 * min_period,
            st.arch,
        );
        pitch_index = max_period - pitch_index;
        gain1 = remove_doubling(
            pitch_buf.as_slice(),
            max_period,
            min_period,
            n_i32,
            &mut pitch_index,
            st.prefilter_period,
            st.prefilter_gain,
            st.arch,
        );
        if pitch_index > max_period - 2 * qext_scale {
            pitch_index = max_period - 2 * qext_scale;
        }
        pitch_index /= qext_scale;
        gain1 *= 0.7f32;
        if st.loss_rate > 2 {
            gain1 *= 0.5f32;
        }
        if st.loss_rate > 4 {
            gain1 *= 0.5f32;
        }
        if st.loss_rate > 8 {
            gain1 = 0 as f32;
        }
    } else {
        gain1 = 0 as f32;
        pitch_index = COMBFILTER_MINPERIOD;
    }
    if analysis.valid != 0 {
        gain1 *= analysis.max_pitch_ratio;
    }
    pf_threshold = 0.2f32;
    if (pitch_index - st.prefilter_period).abs() * 10 > pitch_index {
        pf_threshold += 0.2f32;
        // Completely disable the prefilter on strong transients without continuity.
        if tf_estimate > 0.98f32 {
            gain1 = 0.;
        }
    }
    if nb_available_bytes < 25 {
        pf_threshold += 0.1f32;
    }
    if nb_available_bytes < 35 {
        pf_threshold += 0.1f32;
    }
    if st.prefilter_gain > 0.4f32 {
        pf_threshold -= 0.1f32;
    }
    if st.prefilter_gain > 0.55f32 {
        pf_threshold -= 0.1f32;
    }
    pf_threshold = if pf_threshold > 0.2f32 {
        pf_threshold
    } else {
        0.2f32
    };
    if gain1 < pf_threshold {
        gain1 = 0 as f32;
        pf_on = 0;
        qg = 0;
    } else {
        // This block is intentionally not additionally gated by a total-bits
        // check because the `nb_available_bytes` thresholding above already
        // handles the low-bitrate edge.
        if ((gain1 - st.prefilter_gain).abs()) < 0.1f32 {
            gain1 = st.prefilter_gain;
        }
        qg = (0.5f32 + gain1 * 32_f32 / 3_f32).floor() as i32 - 1;
        qg = if 0 > (if (7) < qg { 7 } else { qg }) {
            0
        } else if (7) < qg {
            7
        } else {
            qg
        };
        gain1 = 0.09375f32 * (qg + 1) as f32;
        pf_on = 1;
    }
    let mut before = [0f32; 2];
    let mut after = [0f32; 2];
    let mut cancel_pitch = false;

    for c in 0..coded_channels as usize {
        let offset: i32 = mode.short_mdct_size - overlap;
        st.prefilter_period = st.prefilter_period.max(COMBFILTER_MINPERIOD);
        // Copy in_mem overlap into in_0
        let in_dst = c * (n_i32 + overlap) as usize;
        in_0[in_dst..in_dst + overlap as usize]
            .copy_from_slice(&st.in_mem[c * overlap as usize..(c + 1) * overlap as usize]);
        // Measure energy before comb filter
        for i in 0..n_i32 as usize {
            before[c] += in_0[c * (n_i32 + overlap) as usize + overlap as usize + i].abs();
        }
        {
            let pre_base = c * pre_chan_len;
            let pre_slice = &_pre[pre_base..pre_base + pre_chan_len];
            let in_base = c * (n_i32 + overlap) as usize + overlap as usize;
            let in_slice = &mut in_0[in_base..in_base + n_i32 as usize];
            if offset != 0 {
                comb_filter(
                    in_slice,
                    0,
                    pre_slice,
                    max_period as usize,
                    st.prefilter_period,
                    st.prefilter_period,
                    offset,
                    -st.prefilter_gain,
                    -st.prefilter_gain,
                    st.prefilter_tapset,
                    st.prefilter_tapset,
                    &[],
                    0,
                    st.arch,
                );
            }
            comb_filter(
                in_slice,
                offset as usize,
                pre_slice,
                (max_period + offset) as usize,
                st.prefilter_period,
                pitch_index,
                n_i32 - offset,
                -st.prefilter_gain,
                -gain1,
                st.prefilter_tapset,
                prefilter_tapset,
                mode.window,
                overlap,
                st.arch,
            );
        }
        // Measure energy after comb filter
        for i in 0..n_i32 as usize {
            after[c] += in_0[c * (n_i32 + overlap) as usize + overlap as usize + i].abs();
        }
    }

    // Check if comb filter made things worse
    if coded_channels == 2 {
        let thresh0 = 0.25f32 * gain1 * before[0] + 0.01f32 * before[1];
        let thresh1 = 0.25f32 * gain1 * before[1] + 0.01f32 * before[0];
        // Don't use the filter if one channel gets significantly worse.
        if after[0] - before[0] > thresh0 || after[1] - before[1] > thresh1 {
            cancel_pitch = true;
        }
        // Use the filter only if at least one channel gets significantly better.
        if before[0] - after[0] < thresh0 && before[1] - after[1] < thresh1 {
            cancel_pitch = true;
        }
    } else {
        // Check that the mono channel actually got better.
        if after[0] > before[0] {
            cancel_pitch = true;
        }
    }

    // If needed, revert to a gain of zero.
    if cancel_pitch {
        for c in 0..coded_channels as usize {
            let offset: i32 = mode.short_mdct_size - overlap;
            let pre_base = c * pre_chan_len;
            let pre_slice = &_pre[pre_base..pre_base + pre_chan_len];
            let in_base = c * (n_i32 + overlap) as usize + overlap as usize;
            // Revert: copy original pre data back
            in_0[in_base..in_base + n_i32 as usize].copy_from_slice(
                &pre_slice[max_period as usize..max_period as usize + n_i32 as usize],
            );
            // Re-apply transition with gain=0
            let in_slice = &mut in_0[in_base..in_base + n_i32 as usize];
            comb_filter(
                in_slice,
                offset as usize,
                pre_slice,
                (max_period + offset) as usize,
                st.prefilter_period,
                pitch_index,
                overlap,
                -st.prefilter_gain,
                -0.,
                st.prefilter_tapset,
                prefilter_tapset,
                mode.window,
                overlap,
                st.arch,
            );
        }
        gain1 = 0.;
        pf_on = 0;
        qg = 0;
    }

    for c in 0..coded_channels as usize {
        // Copy end of in_0 back into in_mem overlap
        let in_src = c * (n_i32 + overlap) as usize + n_i32 as usize;
        st.in_mem[c * overlap as usize..(c + 1) * overlap as usize]
            .copy_from_slice(&in_0[in_src..in_src + overlap as usize]);
        // Update prefilter_mem from _pre
        let pre_base = c * pre_chan_len;
        let max_period_u = max_period as usize;
        let pfm_base = c * max_period_u;
        if n_i32 > max_period {
            st.prefilter_mem[pfm_base..pfm_base + max_period_u].copy_from_slice(
                &_pre[pre_base + n_i32 as usize..pre_base + n_i32 as usize + max_period_u],
            );
        } else {
            // Shift prefilter_mem left by n_i32
            st.prefilter_mem
                .copy_within(pfm_base + n_i32 as usize..pfm_base + max_period_u, pfm_base);
            // Copy last n_i32 samples from _pre
            st.prefilter_mem[pfm_base + max_period_u - n_i32 as usize..pfm_base + max_period_u]
                .copy_from_slice(
                    &_pre[pre_base + max_period_u..pre_base + max_period_u + n_i32 as usize],
                );
        }
    }
    *gain = gain1;
    *pitch = pitch_index;
    *qgain = qg;
    pf_on
}
/// Compute the per-frame VBR target (in 1/8-bit units) from analysis and
/// coding context.
///
/// Upstream C: celt/celt_encoder.c:compute_vbr
#[allow(clippy::too_many_arguments)]
fn compute_vbr(
    mode: &OpusCustomMode,
    analysis: &AnalysisInfo,
    base_target: i32,
    lm: i32,
    bitrate: i32,
    last_coded_bands: i32,
    channels: i32,
    intensity: i32,
    constrained_vbr: i32,
    mut stereo_saving: f32,
    tot_boost: i32,
    tf_estimate: f32,
    pitch_change: i32,
    max_depth: f32,
    lfe: i32,
    has_surround_mask: i32,
    surround_masking: f32,
    temporal_vbr: f32,
) -> i32 {
    // The target rate in 8th bits per frame.
    let mut target: i32;
    let mut coded_bins: i32;

    let e_bands = &mode.e_bands;
    let nb_ebands: i32 = mode.nb_ebands as i32;
    let coded_bands: i32 = if last_coded_bands != 0 {
        last_coded_bands
    } else {
        nb_ebands
    };
    coded_bins = (e_bands[coded_bands as usize] as i32) << lm;
    if channels == 2 {
        coded_bins += (e_bands[(if intensity < coded_bands {
            intensity
        } else {
            coded_bands
        }) as usize] as i32)
            << lm;
    }
    target = base_target;
    if analysis.valid != 0 && (analysis.activity as f64) < 0.4f64 {
        target -= ((coded_bins << BITRES) as f32 * (0.4f32 - analysis.activity)) as i32;
    }
    // Stereo savings.
    if channels == 2 {
        let coded_stereo_bands: i32 = if intensity < coded_bands {
            intensity
        } else {
            coded_bands
        };
        let coded_stereo_dof: i32 =
            ((e_bands[coded_stereo_bands as usize] as i32) << lm) - coded_stereo_bands;
        // Maximum fraction of bits we can save if the signal is effectively mono.
        let max_frac: f32 = 0.8f32 * coded_stereo_dof as f32 / coded_bins as f32;
        stereo_saving = if stereo_saving < 1.0f32 {
            stereo_saving
        } else {
            1.0f32
        };
        target -= (if (max_frac * target as f32)
            < (stereo_saving - 0.1f32) * (coded_stereo_dof << 3) as f32
        {
            max_frac * target as f32
        } else {
            (stereo_saving - 0.1f32) * (coded_stereo_dof << 3) as f32
        }) as i32;
    }
    // Boost according to dynalloc (minus average calibration term).
    target += tot_boost - ((19) << lm);
    // Apply transient boost, compensating for average boost.
    let tf_calibration: f32 = 0.044f32;
    target += ((tf_estimate - tf_calibration) * target as f32) as i32;
    if analysis.valid != 0 && lfe == 0 {
        let mut tonal_target: i32;

        // Tonality boost (compensating for the average).
        let tonal: f32 = (if 0.0f32 > analysis.tonality - 0.15f32 {
            0.0f32
        } else {
            analysis.tonality - 0.15f32
        }) - 0.12f32;
        tonal_target = target + ((coded_bins << BITRES) as f32 * 1.2f32 * tonal) as i32;
        if pitch_change != 0 {
            tonal_target += ((coded_bins << BITRES) as f32 * 0.8f32) as i32;
        }
        target = tonal_target;
    }
    if has_surround_mask != 0 && lfe == 0 {
        let surround_target: i32 = target + (surround_masking * (coded_bins << 3) as f32) as i32;
        target = if target / 4 > surround_target {
            target / 4
        } else {
            surround_target
        };
    }
    let mut floor_depth: i32;

    let bins: i32 = (e_bands[(nb_ebands - 2) as usize] as i32) << lm;
    floor_depth = (((channels * bins) << 3) as f32 * max_depth) as i32;
    floor_depth = if floor_depth > target >> 2 {
        floor_depth
    } else {
        target >> 2
    };
    target = if target < floor_depth {
        target
    } else {
        floor_depth
    };
    // Make VBR less aggressive for constrained VBR because we cannot sustain
    // long high-rate excursions.
    if (has_surround_mask == 0 || lfe != 0) && constrained_vbr != 0 {
        target = base_target + (0.67f32 * (target - base_target) as f32) as i32;
    }
    if has_surround_mask == 0 && tf_estimate < 0.2f32 {
        let amount: f32 = 0.0000031f32
            * (if 0
                > (if (32000) < 96000 - bitrate {
                    32000
                } else {
                    96000 - bitrate
                })
            {
                0
            } else if (32000) < 96000 - bitrate {
                32000
            } else {
                96000 - bitrate
            }) as f32;
        let tvbr_factor: f32 = temporal_vbr * amount;
        target += (tvbr_factor * target as f32) as i32;
    }
    // Don't allow more than doubling the base target.
    target = if 2 * base_target < target {
        2 * base_target
    } else {
        target
    };
    target
}
pub fn celt_encode_with_ec<'b>(
    st: &mut OpusCustomEncoder,
    pcm: &[f32],
    mut frame_size: i32,
    compressed: &'b mut [u8],
    mut nb_compressed_bytes: i32,
    mut enc: Option<&mut EcEnc<'b>>,
    #[cfg(feature = "qext")] qext_payload: Option<&mut [u8]>,
    #[cfg(feature = "qext")] qext_bytes: i32,
) -> i32 {
    let mut i: i32;
    let mut c: i32;

    let mut bits: i32;
    let mut _enc: EcEnc = EcEnc {
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

    let mut short_blocks: i32;
    let mut is_transient: i32;
    let coded_channels: i32 = st.channels;
    let channels: i32 = st.stream_channels;
    let mut lm: i32;

    let tf_select: i32;
    let nb_filled_bytes: i32;
    let mut nb_available_bytes: i32;

    let mut end: i32;
    let mut eff_end: i32;

    let mut alloc_trim: i32;
    let mut pitch_index: i32 = COMBFILTER_MINPERIOD;
    let mut gain1: f32 = 0 as f32;
    let mut dual_stereo: i32 = 0;
    let mut effective_bytes: i32;
    let mut dynalloc_logp: i32;
    let mut vbr_rate: i32;
    let mut total_bits: i32;
    let mut total_boost: i32;
    let mut balance: i32 = 0;
    let mut tell: i32;
    let tell0_frac: i32;

    let pf_on: i32;

    let anti_collapse_on: i32;
    let mut silence: i32;
    let mut tf_chan: i32 = 0;
    let mut tf_estimate: f32;

    let mut toneishness: f32 = 0.0;
    let mut pitch_change: i32 = 0;
    let mut tot_boost: i32 = 0;
    let mut sample_max: f32;

    let mut wrote_custom_header = false;
    let mut compressed_offset = 0usize;

    let mut signal_bandwidth: i32;
    let mut transient_got_disabled: i32 = 0;
    let mut surround_masking: f32 = 0 as f32;
    let mut temporal_vbr: f32 = 0 as f32;
    let mut surround_trim: f32 = 0 as f32;
    let mut equiv_rate: i32;

    let mut weak_transient: i32 = 0;

    #[cfg(feature = "qext")]
    let qext_scale = st.qext_scale;
    #[cfg(not(feature = "qext"))]
    let qext_scale = 1;
    let max_period = COMBFILTER_MAXPERIOD * qext_scale;
    // Max channels * nb_ebands: 2 * (21 + 14) = 70; use 80 for headroom.
    const MAX_C_BANDS: usize = 80;
    // QEXT: Initialize extension entropy encoder from payload buffer
    #[cfg(feature = "qext")]
    let mut _qext_empty_buf = [0u8; 0];
    #[cfg(feature = "qext")]
    let mut ext_enc = if let Some(payload) = qext_payload {
        ec_enc_init(payload)
    } else {
        ec_enc_init(&mut _qext_empty_buf)
    };
    #[cfg(feature = "qext")]
    let mut qext_end: i32 = 0;
    #[cfg(feature = "qext")]
    let mut qext_intensity: i32 = 0;
    #[cfg(feature = "qext")]
    let mut qext_dual_stereo: i32 = 0;
    #[cfg(feature = "qext")]
    let mut qext_mode: Option<crate::celt::modes::OpusCustomMode> = None;
    #[cfg(feature = "qext")]
    let mut qext_band_e = [0.0f32; 2 * crate::celt::modes::data_96000::NB_QEXT_BANDS];
    #[cfg(feature = "qext")]
    let mut qext_band_log_e = [0.0f32; 2 * crate::celt::modes::data_96000::NB_QEXT_BANDS];
    #[cfg(feature = "qext")]
    let mut qext_error = [0.0f32; 2 * crate::celt::modes::data_96000::NB_QEXT_BANDS];
    let mode: &'static OpusCustomMode = st.mode;
    let nb_ebands: i32 = mode.nb_ebands as i32;
    let overlap: i32 = mode.overlap as i32;
    let e_bands: &[i16] = mode.e_bands;
    let start: i32 = st.start;
    end = st.end;
    let hybrid: i32 = (start != 0) as i32;
    tf_estimate = 0 as f32;
    if nb_compressed_bytes < 2 || pcm.is_empty() {
        return OPUS_BAD_ARG;
    }
    frame_size *= st.upsample;
    lm = 0;
    while lm <= mode.max_lm {
        if mode.short_mdct_size << lm == frame_size {
            break;
        }
        lm += 1;
    }
    if lm > mode.max_lm {
        return OPUS_BAD_ARG;
    }
    let m_stride: i32 = (1) << lm;
    let n_i32: i32 = m_stride * mode.short_mdct_size;
    if let Some(enc) = enc.as_mut() {
        tell0_frac = ec_tell_frac(enc) as i32;
        tell = ec_tell(enc);
        nb_filled_bytes = (tell + 4) >> 3;
    } else {
        tell = 1;
        tell0_frac = tell;
        nb_filled_bytes = 0;
    }
    if st.signalling != 0 && enc.is_none() {
        let tmp = (mode.eff_ebands - end) >> 1;
        end = 1.max(mode.eff_ebands - tmp);
        st.end = end;
        if nb_compressed_bytes < 2 || compressed.is_empty() {
            return OPUS_BAD_ARG;
        }
        let mut c0: u8 = ((tmp << 5) | (lm << 3) | (((channels == 2) as i32) << 2)) as u8;
        if should_convert_custom_signalling_header(mode) {
            let Some(opus_c0) = to_opus_header_byte(c0) else {
                return OPUS_BAD_ARG;
            };
            c0 = opus_c0;
        }
        compressed[0] = c0;
        compressed_offset = 1;
        wrote_custom_header = true;
        nb_compressed_bytes -= 1;
    }
    nb_compressed_bytes = if nb_compressed_bytes < 1275 {
        nb_compressed_bytes
    } else {
        1275
    };
    nb_available_bytes = nb_compressed_bytes - nb_filled_bytes;
    if st.vbr != 0 && st.bitrate != OPUS_BITRATE_MAX {
        vbr_rate = (st.bitrate * 6 / (6 * mode.fs / frame_size)) << BITRES;
        if st.signalling != 0 {
            vbr_rate -= 8 << BITRES;
        }
        effective_bytes = vbr_rate >> (3 + BITRES);
    } else {
        let mut tmp: i32;
        vbr_rate = 0;
        tmp = st.bitrate * frame_size;
        if tell > 1 {
            tmp += tell * mode.fs;
        }
        if st.bitrate != OPUS_BITRATE_MAX {
            nb_compressed_bytes = if 2
                > (if nb_compressed_bytes
                    < (tmp + 4 * mode.fs) / (8 * mode.fs) - (st.signalling != 0) as i32
                {
                    nb_compressed_bytes
                } else {
                    (tmp + 4 * mode.fs) / (8 * mode.fs) - (st.signalling != 0) as i32
                }) {
                2
            } else if nb_compressed_bytes
                < (tmp + 4 * mode.fs) / (8 * mode.fs) - (st.signalling != 0) as i32
            {
                nb_compressed_bytes
            } else {
                (tmp + 4 * mode.fs) / (8 * mode.fs) - (st.signalling != 0) as i32
            };
            if let Some(enc) = enc.as_mut() {
                ec_enc_shrink(enc, nb_compressed_bytes as u32);
            }
        }
        effective_bytes = nb_compressed_bytes - nb_filled_bytes;
    }
    equiv_rate =
        ((nb_compressed_bytes * 8 * 50) << (3 - lm)) - (40 * channels + 20) * ((400 >> lm) - 50);
    if st.bitrate != OPUS_BITRATE_MAX {
        equiv_rate = if equiv_rate < st.bitrate - (40 * channels + 20) * ((400 >> lm) - 50) {
            equiv_rate
        } else {
            st.bitrate - (40 * channels + 20) * ((400 >> lm) - 50)
        };
    }
    let enc = if let Some(enc) = enc {
        enc
    } else {
        let end_off = compressed_offset + nb_compressed_bytes as usize;
        _enc = ec_enc_init(&mut compressed[compressed_offset..end_off]);
        &mut _enc
    };
    if vbr_rate > 0 && st.constrained_vbr != 0 {
        let vbr_bound: i32 = vbr_rate;
        let max_allowed: i32 = if (if (if tell == 1 { 2 } else { 0 })
            > (vbr_rate + vbr_bound - st.vbr_reservoir) >> (3 + 3)
        {
            if tell == 1 {
                2
            } else {
                0
            }
        } else {
            (vbr_rate + vbr_bound - st.vbr_reservoir) >> (3 + 3)
        }) < nb_available_bytes
        {
            if (if tell == 1 { 2 } else { 0 })
                > (vbr_rate + vbr_bound - st.vbr_reservoir) >> (3 + 3)
            {
                if tell == 1 {
                    2
                } else {
                    0
                }
            } else {
                (vbr_rate + vbr_bound - st.vbr_reservoir) >> (3 + 3)
            }
        } else {
            nb_available_bytes
        };
        if max_allowed < nb_available_bytes {
            nb_compressed_bytes = nb_filled_bytes + max_allowed;
            nb_available_bytes = max_allowed;
            ec_enc_shrink(enc, nb_compressed_bytes as u32);
        }
    }
    total_bits = nb_compressed_bytes * 8;
    eff_end = end;
    if eff_end > mode.eff_ebands {
        eff_end = mode.eff_ebands;
    }
    // coded_channels * (n_i32 + overlap) max: 2 * (1920 + 240) = 4320.
    const MAX_IN: usize = 4320;
    debug_assert!(((coded_channels * (n_i32 + overlap)) as usize) <= MAX_IN);
    let mut in_0 = [0.0f32; MAX_IN];
    let main_len = (channels * (n_i32 - overlap) / st.upsample) as usize;
    let overlap_len = (channels * overlap / st.upsample) as usize;
    sample_max = if st.overlap_max > celt_maxabs16(&pcm[..main_len]) {
        st.overlap_max
    } else {
        celt_maxabs16(&pcm[..main_len])
    };
    st.overlap_max = celt_maxabs16(&pcm[main_len..main_len + overlap_len]);
    sample_max = if sample_max > st.overlap_max {
        sample_max
    } else {
        st.overlap_max
    };
    silence = (sample_max <= 1_f32 / ((1) << st.lsb_depth) as f32) as i32;
    if tell == 1 {
        ec_enc_bit_logp(enc, silence, 15);
    } else {
        silence = 0;
    }
    if silence != 0 {
        if vbr_rate > 0 {
            nb_compressed_bytes = if nb_compressed_bytes < nb_filled_bytes + 2 {
                nb_compressed_bytes
            } else {
                nb_filled_bytes + 2
            };
            effective_bytes = nb_compressed_bytes;
            total_bits = nb_compressed_bytes * 8;
            nb_available_bytes = 2;
            ec_enc_shrink(enc, nb_compressed_bytes as u32);
        }
        tell = nb_compressed_bytes * 8;
        enc.nbits_total += tell - ec_tell(enc);
    }
    c = 0;
    loop {
        let need_clip: i32 = (st.clip != 0 && sample_max > 65536.0f32) as i32;
        celt_preemphasis(
            &pcm[c as usize..],
            &mut in_0[(c * (n_i32 + overlap) + overlap) as usize..],
            n_i32,
            coded_channels,
            st.upsample,
            &mode.preemph,
            &mut st.preemph_mem_e[c as usize],
            need_clip,
        );
        // Copy overlap from prefilter_mem into in_0 (must be before tone_detect/transient_analysis)
        let in_dst = (c * (n_i32 + overlap)) as usize;
        let pfm_src = ((c + 1) * max_period - overlap) as usize;
        in_0[in_dst..in_dst + overlap as usize]
            .copy_from_slice(&st.prefilter_mem[pfm_src..pfm_src + overlap as usize]);
        c += 1;
        if c >= coded_channels {
            break;
        }
    }
    // Tone detection — must be before transient_analysis and run_prefilter.
    let tone_freq: f32 = tone_detect(
        &in_0,
        coded_channels,
        n_i32 + overlap,
        &mut toneishness,
        mode.fs,
    );
    is_transient = 0;
    short_blocks = 0;
    if st.complexity >= 1 && st.lfe == 0 {
        let allow_weak_transients: i32 =
            (hybrid != 0 && effective_bytes < 15 && st.silk_info.signal_type != 2) as i32;
        is_transient = transient_analysis(
            &in_0,
            n_i32 + overlap,
            coded_channels,
            &mut tf_estimate,
            &mut tf_chan,
            allow_weak_transients,
            &mut weak_transient,
            tone_freq,
            toneishness,
        );
    }
    toneishness = toneishness.min(1.0 - tf_estimate);

    let mut qg: i32 = 0;
    let enabled: i32 = ((st.lfe != 0 && nb_available_bytes > 3
        || nb_available_bytes > 12 * channels)
        && hybrid == 0
        && silence == 0
        && tell + 16 <= total_bits
        && st.disable_pf == 0) as i32;
    let prefilter_tapset: i32 = st.tapset_decision;
    {
        let analysis = st.analysis;
        pf_on = run_prefilter(
            &mut *st,
            &mut in_0,
            coded_channels,
            n_i32,
            prefilter_tapset,
            &mut pitch_index,
            &mut gain1,
            &mut qg,
            enabled,
            tf_estimate,
            nb_available_bytes,
            &analysis,
            tone_freq,
            toneishness,
        );
    }
    if (gain1 > 0.4f32 || st.prefilter_gain > 0.4f32)
        && (st.analysis.valid == 0 || st.analysis.tonality as f64 > 0.3f64)
        && (pitch_index as f64 > 1.26f64 * st.prefilter_period as f64
            || (pitch_index as f64) < 0.79f64 * st.prefilter_period as f64)
    {
        pitch_change = 1;
    }
    if pf_on == 0 {
        if hybrid == 0 && tell + 16 <= total_bits {
            ec_enc_bit_logp(enc, 0, 1);
        }
    } else {
        ec_enc_bit_logp(enc, 1, 1);
        pitch_index += 1;
        let octave: i32 = EC_CLZ0 - (pitch_index as u32).leading_zeros() as i32 - 5;
        ec_enc_uint(enc, octave as u32, 6);
        ec_enc_bits(
            enc,
            (pitch_index - ((16) << octave)) as u32,
            (4 + octave) as u32,
        );
        pitch_index -= 1;
        ec_enc_bits(enc, qg as u32, 3);
        ec_enc_icdf(enc, prefilter_tapset, &TAPSET_ICDF, 2);
    }
    if lm > 0 && ec_tell(enc) + 3 <= total_bits {
        if is_transient != 0 {
            short_blocks = m_stride;
        }
    } else {
        is_transient = 0;
        transient_got_disabled = 1;
    }
    // Allocate n_i32 + m_stride - 1 elements so that strided mdct_forward calls
    // can form slices freq[b..b + n_i32*B] for b in 0..B without going
    // out of bounds. The extra elements are never read (stride skips them).
    // coded_channels*n_i32 + m_stride - 1 max: 2*1920 + 7 = 3847.
    const MAX_FREQ: usize = 3848;
    debug_assert!(((coded_channels * n_i32 + m_stride - 1) as usize) <= MAX_FREQ);
    let mut freq = [0.0f32; MAX_FREQ];
    let mut band_e = [0.0f32; MAX_C_BANDS];
    let mut band_log_e = [0.0f32; MAX_C_BANDS];
    let second_mdct: i32 = (short_blocks != 0 && st.complexity >= 8) as i32;
    let mut band_log_e2 = [0.0f32; MAX_C_BANDS];
    if second_mdct != 0 {
        compute_mdcts(
            mode,
            0,
            &mut in_0,
            &mut freq,
            channels,
            coded_channels,
            lm,
            st.upsample,
        );
        compute_band_energies(mode, &freq, &mut band_e, eff_end, channels, lm, st.arch);
        amp2_log2(mode, eff_end, end, &band_e, &mut band_log_e2, channels);
        c = 0;
        while c < channels {
            i = 0;
            while i < end {
                band_log_e2[(nb_ebands * c + i) as usize] += 0.5f32 * lm as f32;
                i += 1;
            }
            c += 1;
        }
    }
    compute_mdcts(
        mode,
        short_blocks,
        &mut in_0,
        &mut freq,
        channels,
        coded_channels,
        lm,
        st.upsample,
    );
    assert!(!freq[0].is_nan() && (channels == 1 || !freq[n_i32 as usize].is_nan()));
    if coded_channels == 2 && channels == 1 {
        tf_chan = 0;
    }
    compute_band_energies(mode, &freq, &mut band_e, eff_end, channels, lm, st.arch);
    if st.lfe != 0 {
        i = 2;
        while i < end {
            band_e[i as usize] = if band_e[i as usize] < 1e-4f32 * band_e[0_usize] {
                band_e[i as usize]
            } else {
                1e-4f32 * band_e[0_usize]
            };
            band_e[i as usize] = if band_e[i as usize] > 1e-15f32 {
                band_e[i as usize]
            } else {
                1e-15f32
            };
            i += 1;
        }
    }
    amp2_log2(mode, eff_end, end, &band_e, &mut band_log_e, channels);
    let mut surround_dynalloc = [0.0f32; MAX_C_BANDS];
    surround_dynalloc[..end as usize].fill(0.0);
    let energy_mask: Option<&[f32]> = if st.energy_mask_len == 0 {
        None
    } else {
        Some(&st.energy_mask[..(coded_channels * nb_ebands) as usize])
    };
    if let Some(energy_mask) = energy_mask {
        if hybrid == 0 && st.lfe == 0 {
            let mut midband: i32;
            let mut count_dynalloc: i32;
            let mut mask_avg: f32 = 0 as f32;
            let mut diff: f32 = 0 as f32;
            let mut count: i32 = 0;
            let mask_end: i32 = if 2 > st.last_coded_bands {
                2
            } else {
                st.last_coded_bands
            };
            c = 0;
            while c < channels {
                i = 0;
                while i < mask_end {
                    let mut mask: f32;
                    mask = if (if energy_mask[(nb_ebands * c + i) as usize] < 0.25f32 {
                        energy_mask[(nb_ebands * c + i) as usize]
                    } else {
                        0.25f32
                    }) > -2.0f32
                    {
                        if energy_mask[(nb_ebands * c + i) as usize] < 0.25f32 {
                            energy_mask[(nb_ebands * c + i) as usize]
                        } else {
                            0.25f32
                        }
                    } else {
                        -2.0f32
                    };
                    if mask > 0 as f32 {
                        mask *= 0.5f32;
                    }
                    mask_avg += mask
                        * (e_bands[(i + 1) as usize] as i32 - e_bands[i as usize] as i32) as f32;
                    count += e_bands[(i + 1) as usize] as i32 - e_bands[i as usize] as i32;
                    diff += mask * (1 + 2 * i - mask_end) as f32;
                    i += 1;
                }
                c += 1;
            }
            assert!(count > 0);
            mask_avg /= count as f32;
            mask_avg += 0.2f32;
            diff = diff * 6_f32 / (channels * (mask_end - 1) * (mask_end + 1) * mask_end) as f32;
            diff *= 0.5f32;
            diff = if (if diff < 0.031f32 { diff } else { 0.031f32 }) > -0.031f32 {
                if diff < 0.031f32 {
                    diff
                } else {
                    0.031f32
                }
            } else {
                -0.031f32
            };
            midband = 0;
            while (e_bands[(midband + 1) as usize] as i32) < e_bands[mask_end as usize] as i32 / 2 {
                midband += 1;
            }
            count_dynalloc = 0;
            i = 0;
            while i < mask_end {
                let mut unmask: f32;
                let lin: f32 = mask_avg + diff * (i - midband) as f32;
                if channels == 2 {
                    unmask = if energy_mask[i as usize] > energy_mask[(nb_ebands + i) as usize] {
                        energy_mask[i as usize]
                    } else {
                        energy_mask[(nb_ebands + i) as usize]
                    };
                } else {
                    unmask = energy_mask[i as usize];
                }
                unmask = if unmask < 0.0f32 { unmask } else { 0.0f32 };
                unmask -= lin;
                if unmask > 0.25f32 {
                    surround_dynalloc[i as usize] = unmask - 0.25f32;
                    count_dynalloc += 1;
                }
                i += 1;
            }
            if count_dynalloc >= 3 {
                mask_avg += 0.25f32;
                if mask_avg > 0 as f32 {
                    mask_avg = 0 as f32;
                    diff = 0 as f32;
                    surround_dynalloc[..mask_end as usize].fill(0.0);
                } else {
                    i = 0;
                    while i < mask_end {
                        surround_dynalloc[i as usize] =
                            if 0 as f32 > surround_dynalloc[i as usize] - 0.25f32 {
                                0 as f32
                            } else {
                                surround_dynalloc[i as usize] - 0.25f32
                            };
                        i += 1;
                    }
                }
            }
            mask_avg += 0.2f32;
            surround_trim = 64_f32 * diff;
            surround_masking = mask_avg;
        }
    }
    if st.lfe == 0 {
        let mut follow: f32 = -10.0f32;
        let mut frame_avg: f32 = 0 as f32;
        let offset: f32 = if short_blocks != 0 {
            0.5f32 * lm as f32
        } else {
            0 as f32
        };
        i = start;
        while i < end {
            follow = if follow - 1.0f32 > band_log_e[i as usize] - offset {
                follow - 1.0f32
            } else {
                band_log_e[i as usize] - offset
            };
            if channels == 2 {
                follow = if follow > band_log_e[(i + nb_ebands) as usize] - offset {
                    follow
                } else {
                    band_log_e[(i + nb_ebands) as usize] - offset
                };
            }
            frame_avg += follow;
            i += 1;
        }
        frame_avg /= (end - start) as f32;
        temporal_vbr = frame_avg - st.spec_avg;
        temporal_vbr = if 3.0f32
            < (if -1.5f32 > temporal_vbr {
                -1.5f32
            } else {
                temporal_vbr
            }) {
            3.0f32
        } else if -1.5f32 > temporal_vbr {
            -1.5f32
        } else {
            temporal_vbr
        };
        st.spec_avg += 0.02f32 * temporal_vbr;
    }
    if second_mdct == 0 {
        let len = (channels * nb_ebands) as usize;
        band_log_e2[..len].copy_from_slice(&band_log_e[..len]);
    }
    if lm > 0
        && ec_tell(enc) + 3 <= total_bits
        && is_transient == 0
        && st.complexity >= 5
        && st.lfe == 0
        && hybrid == 0
        && patch_transient_decision(
            &band_log_e,
            &st.old_band_e[..(channels * nb_ebands) as usize],
            nb_ebands,
            start,
            end,
            channels,
        ) != 0
    {
        is_transient = 1;
        short_blocks = m_stride;
        compute_mdcts(
            mode,
            short_blocks,
            &mut in_0,
            &mut freq,
            channels,
            coded_channels,
            lm,
            st.upsample,
        );
        compute_band_energies(mode, &freq, &mut band_e, eff_end, channels, lm, st.arch);
        amp2_log2(mode, eff_end, end, &band_e, &mut band_log_e, channels);
        c = 0;
        while c < channels {
            i = 0;
            while i < end {
                band_log_e2[(nb_ebands * c + i) as usize] += 0.5f32 * lm as f32;
                i += 1;
            }
            c += 1;
        }
        tf_estimate = 0.2f32;
    }
    if lm > 0 && ec_tell(enc) + 3 <= total_bits {
        ec_enc_bit_logp(enc, is_transient, 3);
    }
    // channels*n_i32 max: 2*1920 = 3840.
    const MAX_X: usize = 3840;
    debug_assert!(((channels * n_i32) as usize) <= MAX_X);
    let mut x = [0.0f32; MAX_X];
    normalise_bands(mode, &freq, &mut x, &band_e, eff_end, channels, m_stride);
    let enable_tf_analysis: i32 = (effective_bytes >= 15 * channels
        && hybrid == 0
        && st.complexity >= 2
        && st.lfe == 0
        && toneishness < 0.98) as i32;
    const MAX_BANDS_ENC: usize = 40;
    debug_assert!((nb_ebands as usize) <= MAX_BANDS_ENC);
    let mut offsets = [0i32; MAX_BANDS_ENC];
    let mut importance = [0i32; MAX_BANDS_ENC];
    let mut spread_weight = [0i32; MAX_BANDS_ENC];
    let max_depth: f32 = dynalloc_analysis(
        &band_log_e,
        &band_log_e2,
        &st.old_band_e,
        nb_ebands,
        start,
        end,
        channels,
        &mut offsets,
        st.lsb_depth,
        mode.log_n,
        is_transient,
        st.vbr,
        st.constrained_vbr,
        mode.e_bands,
        lm,
        effective_bytes,
        &mut tot_boost,
        st.lfe,
        &surround_dynalloc,
        &st.analysis,
        &mut importance,
        &mut spread_weight,
        tone_freq,
        toneishness,
    );
    let mut tf_res = [0i32; MAX_BANDS_ENC];
    if enable_tf_analysis != 0 {
        let lambda: i32 = if 80 > 20480 / effective_bytes + 2 {
            80
        } else {
            20480 / effective_bytes + 2
        };
        tf_select = tf_analysis(
            mode,
            eff_end,
            is_transient,
            &mut tf_res,
            lambda,
            &x,
            n_i32,
            lm,
            tf_estimate,
            tf_chan,
            &importance,
        );
        i = eff_end;
        while i < end {
            tf_res[i as usize] = tf_res[(eff_end - 1) as usize];
            i += 1;
        }
    } else if hybrid != 0 && weak_transient != 0 {
        i = 0;
        while i < end {
            tf_res[i as usize] = 1;
            i += 1;
        }
        tf_select = 0;
    } else if hybrid != 0 && effective_bytes < 15 && st.silk_info.signal_type != 2 {
        i = 0;
        while i < end {
            tf_res[i as usize] = 0;
            i += 1;
        }
        tf_select = is_transient;
    } else {
        i = 0;
        while i < end {
            tf_res[i as usize] = is_transient;
            i += 1;
        }
        tf_select = 0;
    }
    let mut error = [0.0f32; MAX_C_BANDS];
    c = 0;
    loop {
        i = start;
        while i < end {
            let idx = (i + c * nb_ebands) as usize;
            if (band_log_e[idx] - st.old_band_e[idx]).abs() < 2.0f32 {
                band_log_e[idx] -= st.energy_error[idx] * 0.25f32;
            }
            i += 1;
        }
        c += 1;
        if c >= channels {
            break;
        }
    }
    quant_coarse_energy(
        mode,
        start,
        end,
        eff_end,
        &band_log_e,
        &mut st.old_band_e[..(channels * nb_ebands) as usize],
        total_bits as u32,
        &mut error,
        enc,
        channels,
        lm,
        nb_available_bytes,
        st.force_intra,
        &mut st.delayed_intra,
        (st.complexity >= 4) as i32,
        st.loss_rate,
        st.lfe,
    );
    tf_encode(start, end, is_transient, &mut tf_res, lm, tf_select, enc);
    if ec_tell(enc) + 4 <= total_bits {
        if st.lfe != 0 {
            st.tapset_decision = 0;
            st.spread_decision = SPREAD_NORMAL;
        } else if hybrid != 0 {
            if st.complexity == 0 {
                st.spread_decision = SPREAD_NONE;
            } else if is_transient != 0 {
                st.spread_decision = SPREAD_NORMAL;
            } else {
                st.spread_decision = SPREAD_AGGRESSIVE;
            }
        } else if short_blocks != 0 || st.complexity < 3 || nb_available_bytes < 10 * channels {
            if st.complexity == 0 {
                st.spread_decision = SPREAD_NONE;
            } else {
                st.spread_decision = SPREAD_NORMAL;
            }
        } else {
            st.spread_decision = spreading_decision(
                mode,
                &x,
                &mut st.tonal_average,
                st.spread_decision,
                &mut st.hf_average,
                &mut st.tapset_decision,
                (pf_on != 0 && short_blocks == 0) as i32,
                eff_end,
                channels,
                m_stride,
                &spread_weight,
            );
        }
        ec_enc_icdf(enc, st.spread_decision, &SPREAD_ICDF, 5);
    } else {
        st.spread_decision = SPREAD_NORMAL;
    }
    if st.lfe != 0 {
        offsets[0_usize] = if (8) < effective_bytes / 3 {
            8
        } else {
            effective_bytes / 3
        };
    }
    let mut cap = [0i32; MAX_BANDS_ENC];
    init_caps(mode, &mut cap, lm, channels);
    dynalloc_logp = 6;
    total_bits <<= BITRES;
    total_boost = 0;
    tell = ec_tell_frac(enc) as i32;
    i = start;
    while i < end {
        let mut dynalloc_loop_logp: i32;
        let mut boost: i32;
        let mut j: i32;
        let width: i32 =
            (channels * (e_bands[(i + 1) as usize] as i32 - e_bands[i as usize] as i32)) << lm;
        let quanta: i32 = if (width << 3) < (if (6) << 3 > width { (6) << 3 } else { width }) {
            width << 3
        } else if (6) << 3 > width {
            (6) << 3
        } else {
            width
        };
        dynalloc_loop_logp = dynalloc_logp;
        boost = 0;
        j = 0;
        while tell + (dynalloc_loop_logp << BITRES) < total_bits - total_boost
            && boost < cap[i as usize]
        {
            let flag: i32 = (j < offsets[i as usize]) as i32;
            ec_enc_bit_logp(enc, flag, dynalloc_loop_logp as u32);
            tell = ec_tell_frac(enc) as i32;
            if flag == 0 {
                break;
            }
            boost += quanta;
            total_boost += quanta;
            dynalloc_loop_logp = 1;
            j += 1;
        }
        if j != 0 {
            dynalloc_logp = if 2 > dynalloc_logp - 1 {
                2
            } else {
                dynalloc_logp - 1
            };
        }
        offsets[i as usize] = boost;
        i += 1;
    }
    if channels == 2 {
        const INTENSITY_THRESHOLDS: [f32; 21] = [
            1_f32, 2_f32, 3_f32, 4_f32, 5_f32, 6_f32, 7_f32, 8_f32, 16_f32, 24_f32, 36_f32, 44_f32,
            50_f32, 56_f32, 62_f32, 67_f32, 72_f32, 79_f32, 88_f32, 106_f32, 134_f32,
        ];
        const INTENSITY_HISTERESIS: [f32; 21] = [
            1_f32, 1_f32, 1_f32, 1_f32, 1_f32, 1_f32, 1_f32, 2_f32, 2_f32, 2_f32, 2_f32, 2_f32,
            2_f32, 2_f32, 3_f32, 3_f32, 4_f32, 5_f32, 6_f32, 8_f32, 8_f32,
        ];
        if lm != 0 {
            dual_stereo = stereo_analysis(mode, &x, lm, n_i32);
        }
        st.intensity = hysteresis_decision(
            (equiv_rate / 1000) as f32,
            &INTENSITY_THRESHOLDS,
            &INTENSITY_HISTERESIS,
            21,
            st.intensity,
        );
        st.intensity = if end
            < (if start > st.intensity {
                start
            } else {
                st.intensity
            }) {
            end
        } else if start > st.intensity {
            start
        } else {
            st.intensity
        };
    }
    alloc_trim = 5;
    if tell + ((6) << BITRES) <= total_bits - total_boost {
        if start > 0 || st.lfe != 0 {
            st.stereo_saving = 0 as f32;
            alloc_trim = 5;
        } else {
            alloc_trim = alloc_trim_analysis(
                mode,
                &x,
                &band_log_e,
                end,
                lm,
                channels,
                n_i32,
                &st.analysis,
                &mut st.stereo_saving,
                tf_estimate,
                st.intensity,
                surround_trim,
                equiv_rate,
                st.arch,
            );
        }
        ec_enc_icdf(enc, alloc_trim, &TRIM_ICDF, 7);
        tell = ec_tell_frac(enc) as i32;
    }
    if vbr_rate > 0 {
        let mut delta: i32;
        let mut target: i32;
        let mut base_target: i32;
        let mut min_allowed: i32;
        let lm_diff: i32 = mode.max_lm - lm;
        nb_compressed_bytes = if nb_compressed_bytes < 1275 >> (3 - lm) {
            nb_compressed_bytes
        } else {
            1275 >> (3 - lm)
        };
        if hybrid == 0 {
            base_target = vbr_rate - ((40 * channels + 20) << BITRES);
        } else {
            base_target = if 0 > vbr_rate - ((9 * channels + 4) << 3) {
                0
            } else {
                vbr_rate - ((9 * channels + 4) << 3)
            };
        }
        if st.constrained_vbr != 0 {
            base_target += st.vbr_offset >> lm_diff;
        }
        if hybrid == 0 {
            target = compute_vbr(
                mode,
                &st.analysis,
                base_target,
                lm,
                equiv_rate,
                st.last_coded_bands,
                channels,
                st.intensity,
                st.constrained_vbr,
                st.stereo_saving,
                tot_boost,
                tf_estimate,
                pitch_change,
                max_depth,
                st.lfe,
                energy_mask.is_some() as i32,
                surround_masking,
                temporal_vbr,
            );
        } else {
            target = base_target;
            if st.silk_info.offset < 100 {
                target += (12) << BITRES >> (3 - lm);
            }
            if st.silk_info.offset > 100 {
                target -= (18) << BITRES >> (3 - lm);
            }
            target += ((tf_estimate - 0.25f32) * ((50) << 3) as f32) as i32;
            if tf_estimate > 0.7f32 {
                target = if target > (50) << 3 {
                    target
                } else {
                    (50) << 3
                };
            }
        }
        target += tell;
        min_allowed = ((tell + total_boost + ((1) << (BITRES + 3)) - 1) >> (BITRES + 3)) + 2;
        if hybrid != 0 {
            min_allowed = if min_allowed
                > (tell0_frac + ((37) << 3) + total_boost + ((1) << (3 + 3)) - 1) >> (3 + 3)
            {
                min_allowed
            } else {
                (tell0_frac + ((37) << 3) + total_boost + ((1) << (3 + 3)) - 1) >> (3 + 3)
            };
        }
        nb_available_bytes = (target + ((1) << (BITRES + 2))) >> (BITRES + 3);
        nb_available_bytes = if min_allowed > nb_available_bytes {
            min_allowed
        } else {
            nb_available_bytes
        };
        nb_available_bytes = if nb_compressed_bytes < nb_available_bytes {
            nb_compressed_bytes
        } else {
            nb_available_bytes
        };
        delta = target - vbr_rate;
        target = nb_available_bytes << (BITRES + 3);
        if silence != 0 {
            nb_available_bytes = 2;
            target = (2 * 8) << BITRES;
            delta = 0;
        }
        let alpha: f32 = if st.vbr_count < 970 {
            st.vbr_count += 1;
            1.0f32 / (st.vbr_count + 20) as f32
        } else {
            0.001f32
        };
        if st.constrained_vbr != 0 {
            st.vbr_reservoir += target - vbr_rate;
        }
        if st.constrained_vbr != 0 {
            st.vbr_drift +=
                (alpha * (delta * ((1) << lm_diff) - st.vbr_offset - st.vbr_drift) as f32) as i32;
            st.vbr_offset = -st.vbr_drift;
        }
        if st.constrained_vbr != 0 && st.vbr_reservoir < 0 {
            let adjust: i32 = -st.vbr_reservoir / ((8) << BITRES);
            nb_available_bytes += if silence != 0 { 0 } else { adjust };
            st.vbr_reservoir = 0;
        }
        nb_compressed_bytes = if nb_compressed_bytes < nb_available_bytes {
            nb_compressed_bytes
        } else {
            nb_available_bytes
        };
        ec_enc_shrink(enc, nb_compressed_bytes as u32);
    }
    let mut fine_quant = [0i32; MAX_BANDS_ENC];
    let mut pulses = [0i32; MAX_BANDS_ENC];
    let mut fine_priority = [0i32; MAX_BANDS_ENC];
    bits = (((nb_compressed_bytes * 8) << BITRES) as u32)
        .wrapping_sub(ec_tell_frac(enc))
        .wrapping_sub(1) as i32;
    let anti_collapse_rsv: i32 = if is_transient != 0 && lm >= 2 && bits >= (lm + 2) << BITRES {
        (1) << BITRES
    } else {
        0
    };
    bits -= anti_collapse_rsv;
    signal_bandwidth = end - 1;
    if st.analysis.valid != 0 {
        let min_bandwidth: i32;
        if equiv_rate < 32000 * channels {
            min_bandwidth = 13;
        } else if equiv_rate < 48000 * channels {
            min_bandwidth = 16;
        } else if equiv_rate < 60000 * channels {
            min_bandwidth = 18;
        } else if equiv_rate < 80000 * channels {
            min_bandwidth = 19;
        } else {
            min_bandwidth = 20;
        }
        signal_bandwidth = if st.analysis.bandwidth > min_bandwidth {
            st.analysis.bandwidth
        } else {
            min_bandwidth
        };
    }
    if st.lfe != 0 {
        signal_bandwidth = 1;
    }
    let coded_bands: i32 = clt_compute_allocation(
        mode,
        start,
        end,
        &offsets,
        &cap,
        alloc_trim,
        &mut st.intensity,
        &mut dual_stereo,
        bits,
        &mut balance,
        &mut pulses,
        &mut fine_quant,
        &mut fine_priority,
        channels,
        lm,
        enc,
        1,
        st.last_coded_bands,
        signal_bandwidth,
    );
    if st.last_coded_bands != 0 {
        st.last_coded_bands = if (st.last_coded_bands + 1)
            < (if st.last_coded_bands - 1 > coded_bands {
                st.last_coded_bands - 1
            } else {
                coded_bands
            }) {
            st.last_coded_bands + 1
        } else if st.last_coded_bands - 1 > coded_bands {
            st.last_coded_bands - 1
        } else {
            coded_bands
        };
    } else {
        st.last_coded_bands = coded_bands;
    }
    quant_fine_energy(
        mode,
        start,
        end,
        &mut st.old_band_e[..(channels * nb_ebands) as usize],
        &mut error,
        None,
        &fine_quant,
        enc,
        channels,
    );
    // QEXT: Compute QEXT mode and band energies after first-pass fine energy
    #[cfg(feature = "qext")]
    {
        use crate::celt::modes::data_96000::NB_QEXT_BANDS;

        if qext_bytes > 0
            && end == nb_ebands
            && (mode.fs == 48000 || mode.fs == 96000)
            && (mode.short_mdct_size == 120 * qext_scale || mode.short_mdct_size == 90 * qext_scale)
        {
            let qext_mode_struct = compute_qext_mode(mode);
            qext_end = if qext_scale == 2 {
                NB_QEXT_BANDS as i32
            } else {
                2
            };
            qext_mode = Some(qext_mode_struct);
        }

        if let Some(ref qm) = qext_mode {
            // Compute band energies at higher frequency resolution
            compute_band_energies(qm, &freq, &mut qext_band_e, qext_end, channels, lm, st.arch);
            normalise_bands(
                qm,
                &freq,
                &mut x,
                &qext_band_e,
                qext_end,
                channels,
                m_stride,
            );
            amp2_log2(
                qm,
                qext_end,
                qext_end,
                &qext_band_e,
                &mut qext_band_log_e,
                channels,
            );

            // Encode stereo params for QEXT bands
            if channels == 2 {
                qext_intensity = qext_end;
                qext_dual_stereo = dual_stereo;
                ec_enc_uint(&mut ext_enc, qext_intensity as u32, (qext_end + 1) as u32);
                if qext_intensity != 0 {
                    ec_enc_bit_logp(&mut ext_enc, qext_dual_stereo, 1);
                }
            }

            // Coarse quantization of QEXT band energies
            let mut qext_delayed_intra: f32 = 0.0;
            quant_coarse_energy(
                qm,
                0,
                qext_end,
                qext_end,
                &qext_band_log_e,
                &mut st.qext_old_band_e,
                (qext_bytes * 8) as u32,
                &mut qext_error,
                &mut ext_enc,
                channels,
                lm,
                qext_bytes,
                st.force_intra,
                &mut qext_delayed_intra,
                (st.complexity >= 4) as i32,
                st.loss_rate,
                st.lfe,
            );
        }
    }

    // QEXT: Compute extra allocation and second-pass fine energy
    st.energy_error[..(nb_ebands * coded_channels) as usize].fill(0.0);
    // nb_ebands + NB_QEXT_BANDS max: 21 + 14 = 35.
    #[cfg(feature = "qext")]
    let mut extra_pulses = [0i32; 40];
    #[cfg(feature = "qext")]
    let mut extra_quant = [0i32; 40];
    #[cfg(feature = "qext")]
    let mut error_bak = [0.0f32; MAX_C_BANDS];
    #[cfg(feature = "qext")]
    {
        let qext_bits = ((qext_bytes * 8) << BITRES) - ec_tell_frac(&ext_enc) as i32 - 1;
        clt_compute_extra_allocation(
            mode,
            qext_mode.as_ref(),
            start,
            end,
            qext_end,
            Some(&band_log_e),
            if qext_mode.is_some() {
                Some(&qext_band_log_e)
            } else {
                None
            },
            qext_bits,
            &mut extra_pulses,
            &mut extra_quant,
            channels,
            lm,
            &mut ext_enc,
            1, // encode=1
            tone_freq,
            toneishness,
        );
        error_bak[..(channels * nb_ebands) as usize]
            .copy_from_slice(&error[..(channels * nb_ebands) as usize]);
        if qext_bytes > 0 {
            quant_fine_energy(
                mode,
                start,
                end,
                &mut st.old_band_e[..(channels * nb_ebands) as usize],
                &mut error,
                Some(&fine_quant),
                &extra_quant[..nb_ebands as usize],
                &mut ext_enc,
                channels,
            );
        }
    }

    // Residual quantisation
    let mut collapse_masks = [0u8; MAX_C_BANDS];

    #[cfg(feature = "qext")]
    let ext_total_bits = if qext_bytes > 0 {
        qext_bytes * (8 << BITRES)
    } else {
        0
    };

    if channels == 2 {
        let (x_part, y_part) = x.split_at_mut(n_i32 as usize);
        quant_all_bands(
            1,
            mode,
            start,
            end,
            x_part,
            Some(y_part),
            &mut collapse_masks,
            &band_e,
            &mut pulses,
            short_blocks,
            st.spread_decision,
            dual_stereo,
            st.intensity,
            &mut tf_res,
            nb_compressed_bytes * ((8) << BITRES) - anti_collapse_rsv,
            balance,
            enc,
            lm,
            coded_bands,
            &mut st.rng,
            st.complexity,
            st.arch,
            st.disable_inv,
            #[cfg(feature = "qext")]
            &mut ext_enc,
            #[cfg(feature = "qext")]
            &extra_pulses,
            #[cfg(feature = "qext")]
            ext_total_bits,
            #[cfg(feature = "qext")]
            &cap,
        );
    } else {
        quant_all_bands(
            1,
            mode,
            start,
            end,
            &mut x,
            None,
            &mut collapse_masks,
            &band_e,
            &mut pulses,
            short_blocks,
            st.spread_decision,
            dual_stereo,
            st.intensity,
            &mut tf_res,
            nb_compressed_bytes * ((8) << BITRES) - anti_collapse_rsv,
            balance,
            enc,
            lm,
            coded_bands,
            &mut st.rng,
            st.complexity,
            st.arch,
            st.disable_inv,
            #[cfg(feature = "qext")]
            &mut ext_enc,
            #[cfg(feature = "qext")]
            &extra_pulses,
            #[cfg(feature = "qext")]
            ext_total_bits,
            #[cfg(feature = "qext")]
            &cap,
        );
    }

    // QEXT: Second quant_all_bands for QEXT residual bands
    #[cfg(feature = "qext")]
    {
        if let Some(ref qm) = qext_mode {
            use crate::celt::modes::data_96000::NB_QEXT_BANDS;

            let mut qext_collapse_masks = [0u8; 2 * NB_QEXT_BANDS];
            let zeros = [0i32; MAX_BANDS_ENC];

            // Compute ext_balance
            let mut ext_balance = qext_bytes * (8 << BITRES) - ec_tell_frac(&ext_enc) as i32;
            for j in 0..qext_end {
                ext_balance -= extra_pulses[nb_ebands as usize + j as usize]
                    + channels * (extra_quant[nb_ebands as usize + 1] << BITRES);
            }

            // Fine energy for QEXT bands
            quant_fine_energy(
                qm,
                0,
                qext_end,
                &mut st.qext_old_band_e[..(channels * NB_QEXT_BANDS as i32) as usize],
                &mut qext_error,
                None,
                &extra_quant[nb_ebands as usize..],
                &mut ext_enc,
                channels,
            );

            // Dummy encoder for the nested ext_enc arg of quant_all_bands
            let mut dummy_buf = [0u8; 4];
            let mut dummy_enc = crate::celt::entcode::EcCtx {
                buf: &mut dummy_buf,
                storage: 4,
                end_offs: 0,
                end_window: 0,
                nend_bits: 0,
                nbits_total: 32,
                offs: 0,
                rng: 0x80000000,
                val: 0,
                ext: 0,
                rem: 0,
                error: 0,
            };

            if channels == 2 {
                let (x_part, y_part) = x.split_at_mut(n_i32 as usize);
                quant_all_bands(
                    1,
                    qm,
                    0,
                    qext_end,
                    x_part,
                    Some(y_part),
                    &mut qext_collapse_masks,
                    &qext_band_e,
                    &mut extra_pulses[nb_ebands as usize..],
                    short_blocks,
                    st.spread_decision,
                    qext_dual_stereo,
                    qext_intensity,
                    &mut zeros.clone(),
                    qext_bytes * (8 << BITRES),
                    ext_balance,
                    &mut ext_enc,
                    lm,
                    qext_end,
                    &mut st.rng,
                    st.complexity,
                    st.arch,
                    st.disable_inv,
                    &mut dummy_enc,
                    &zeros,
                    0,
                    &[],
                );
            } else {
                quant_all_bands(
                    1,
                    qm,
                    0,
                    qext_end,
                    &mut x,
                    None,
                    &mut qext_collapse_masks,
                    &qext_band_e,
                    &mut extra_pulses[nb_ebands as usize..],
                    short_blocks,
                    st.spread_decision,
                    qext_dual_stereo,
                    qext_intensity,
                    &mut zeros.clone(),
                    qext_bytes * (8 << BITRES),
                    ext_balance,
                    &mut ext_enc,
                    lm,
                    qext_end,
                    &mut st.rng,
                    st.complexity,
                    st.arch,
                    st.disable_inv,
                    &mut dummy_enc,
                    &zeros,
                    0,
                    &[],
                );
            }
        }
    }

    if anti_collapse_rsv > 0 {
        anti_collapse_on = (st.consec_transient < 2) as i32;
        ec_enc_bits(enc, anti_collapse_on as u32, 1);
    }

    // Energy finalisation: skip when QEXT is active (use error_bak instead)
    #[cfg(feature = "qext")]
    {
        if qext_bytes == 0 {
            quant_energy_finalise(
                mode,
                start,
                end,
                &mut st.old_band_e[..(channels * nb_ebands) as usize],
                &mut error,
                &fine_quant,
                &fine_priority,
                nb_compressed_bytes * 8 - ec_tell(enc),
                enc,
                channels,
            );
        }
    }
    #[cfg(not(feature = "qext"))]
    {
        quant_energy_finalise(
            mode,
            start,
            end,
            &mut st.old_band_e[..(channels * nb_ebands) as usize],
            &mut error,
            &fine_quant,
            &fine_priority,
            nb_compressed_bytes * 8 - ec_tell(enc),
            enc,
            channels,
        );
    }

    c = 0;
    loop {
        i = start;
        while i < end {
            let idx = (i + c * nb_ebands) as usize;
            st.energy_error[idx] = error[idx].clamp(-0.5f32, 0.5f32);
            i += 1;
        }
        c += 1;
        if c >= channels {
            break;
        }
    }

    // QEXT: When qext_bytes > 0, run finalise with error_bak (original error before QEXT fine energy)
    #[cfg(feature = "qext")]
    {
        if qext_bytes > 0 {
            // Pass NULL for old_band_e (don't update), use error_bak
            quant_energy_finalise(
                mode,
                start,
                end,
                &mut [0.0f32; 42][..(channels * nb_ebands) as usize], // dummy, won't be used meaningfully
                &mut error_bak,
                &fine_quant,
                &fine_priority,
                nb_compressed_bytes * 8 - ec_tell(enc),
                enc,
                channels,
            );
        }
    }
    if silence != 0 {
        i = 0;
        while i < channels * nb_ebands {
            st.old_band_e[i as usize] = -28.0f32;
            i += 1;
        }
    }
    st.prefilter_period = pitch_index;
    st.prefilter_gain = gain1;
    st.prefilter_tapset = prefilter_tapset;
    if coded_channels == 2 && channels == 1 {
        let nb = nb_ebands as usize;
        st.old_band_e.copy_within(..nb, nb);
    }
    if is_transient == 0 {
        let len = (coded_channels * nb_ebands) as usize;
        st.old_log_e2[..len].copy_from_slice(&st.old_log_e[..len]);
        st.old_log_e[..len].copy_from_slice(&st.old_band_e[..len]);
    } else {
        i = 0;
        while i < coded_channels * nb_ebands {
            let idx = i as usize;
            st.old_log_e[idx] = if st.old_log_e[idx] < st.old_band_e[idx] {
                st.old_log_e[idx]
            } else {
                st.old_band_e[idx]
            };
            i += 1;
        }
    }
    c = 0;
    loop {
        i = 0;
        while i < start {
            let idx = (c * nb_ebands + i) as usize;
            st.old_band_e[idx] = 0 as f32;
            st.old_log_e2[idx] = -28.0f32;
            st.old_log_e[idx] = -28.0f32;
            i += 1;
        }
        i = end;
        while i < nb_ebands {
            let idx = (c * nb_ebands + i) as usize;
            st.old_band_e[idx] = 0 as f32;
            st.old_log_e2[idx] = -28.0f32;
            st.old_log_e[idx] = -28.0f32;
            i += 1;
        }
        c += 1;
        if c >= coded_channels {
            break;
        }
    }
    if is_transient != 0 || transient_got_disabled != 0 {
        st.consec_transient += 1;
    } else {
        st.consec_transient = 0;
    }
    st.rng = enc.rng;
    // QEXT: XOR ext_enc RNG into encoder state and finalize
    #[cfg(feature = "qext")]
    {
        if qext_bytes > 0 {
            ec_enc_done(&mut ext_enc);
            st.rng ^= ext_enc.rng;
        }
    }
    ec_enc_done(enc);
    if wrote_custom_header {
        nb_compressed_bytes += 1;
    }
    if ec_get_error(enc) != 0 {
        OPUS_INTERNAL_ERROR
    } else {
        nb_compressed_bytes
    }
}
