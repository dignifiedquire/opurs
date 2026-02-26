//! CELT encoder.
//!
//! Upstream C: `celt/celt_encoder.c`

use crate::arch::Arch;
use crate::celt::bands::{
    compute_band_energies, haar1, hysteresis_decision, normalise_bands, quant_all_bands,
    spreading_decision, SPREAD_AGGRESSIVE, SPREAD_NONE, SPREAD_NORMAL,
};

pub mod arch_h {
    pub type opus_val16 = f32;
    pub type opus_val32 = f32;
    pub type celt_sig = f32;
    pub type celt_norm = f32;
    pub type celt_ener = f32;
    pub const CELT_SIG_SCALE: f32 = 32768.0f32;
    pub const EPSILON: f32 = 1e-15f32;
}
#[derive(Copy, Clone)]
#[repr(C)]
pub struct SILKInfo {
    pub signalType: i32,
    pub offset: i32,
}

pub use self::arch_h::{
    celt_ener, celt_norm, celt_sig, opus_val16, opus_val32, CELT_SIG_SCALE, EPSILON,
};
use crate::celt::common::{
    comb_filter, init_caps, resampling_factor, spread_icdf, tapset_icdf, tf_select_table, trim_icdf,
};
use crate::celt::common::{COMBFILTER_MAXPERIOD, COMBFILTER_MINPERIOD};
use crate::celt::entcode::{ec_get_error, ec_tell, ec_tell_frac, BITRES};
use crate::celt::entenc::{
    ec_enc, ec_enc_bit_logp, ec_enc_bits, ec_enc_done, ec_enc_icdf, ec_enc_init, ec_enc_shrink,
    ec_enc_uint,
};
use crate::celt::mathops::{celt_exp2, celt_log2, celt_maxabs16, celt_sqrt};
use crate::celt::mdct::mdct_forward;
#[cfg(feature = "qext")]
use crate::celt::modes::compute_qext_mode;
use crate::celt::modes::{opus_custom_mode_create, OpusCustomMode};
use crate::celt::pitch::{celt_inner_prod, pitch_downsample, pitch_search, remove_doubling};
use crate::celt::quant_bands::{
    amp2Log2, eMeans, quant_coarse_energy, quant_energy_finalise, quant_fine_energy,
};
use crate::celt::rate::clt_compute_allocation;
#[cfg(feature = "qext")]
use crate::celt::rate::clt_compute_extra_allocation;

use crate::opus::analysis::AnalysisInfo;
use crate::opus::opus_defines::{OPUS_BAD_ARG, OPUS_BITRATE_MAX, OPUS_INTERNAL_ERROR};
use crate::silk::macros::EC_CLZ0;

/// Upstream C: celt/celt_encoder.c:OpusCustomEncoder
///
/// The C version uses a flexible array member (`in_mem[1]`) at the end of the struct
/// to store overlap memory, prefilter memory, and band energy arrays in a contiguous
/// allocation. This Rust version uses fixed-size arrays sized for the maximum case
/// (2 channels, overlap=240 with QEXT 96 kHz, nbEBands=21, COMBFILTER_MAXPERIOD=1024).
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
    pub delayedIntra: opus_val32,
    pub tonal_average: i32,
    pub lastCodedBands: i32,
    pub hf_average: i32,
    pub tapset_decision: i32,
    pub prefilter_period: i32,
    pub prefilter_gain: opus_val16,
    pub prefilter_tapset: i32,
    pub consec_transient: i32,
    pub analysis: AnalysisInfo,
    pub silk_info: SILKInfo,
    pub preemph_memE: [opus_val32; 2],
    pub preemph_memD: [opus_val32; 2],
    pub vbr_reservoir: i32,
    pub vbr_drift: i32,
    pub vbr_offset: i32,
    pub vbr_count: i32,
    pub overlap_max: opus_val32,
    pub stereo_saving: opus_val16,
    pub intensity: i32,
    /// Energy mask for surround encoding (set by multistream encoder).
    /// `energy_mask_len == 0` means no mask is active.
    pub energy_mask: [opus_val16; 2 * 21],
    pub energy_mask_len: usize,
    pub spec_avg: opus_val16,
    /// Overlap memory, size = channels * overlap (max 2*240 = 480)
    pub in_mem: [celt_sig; 2 * 240],
    /// Prefilter memory, size = channels * QEXT_SCALE(COMBFILTER_MAXPERIOD) (max 2*2048 = 4096)
    pub prefilter_mem: [celt_sig; 2 * PREFILTER_MEM_CHAN_CAP],
    /// Old band energies, size = channels * nbEBands (max 2*21 = 42)
    pub oldBandE: [opus_val16; 2 * 21],
    /// Old log energies, size = channels * nbEBands (max 2*21 = 42)
    pub oldLogE: [opus_val16; 2 * 21],
    /// Old log energies (2 frames ago), size = channels * nbEBands (max 2*21 = 42)
    pub oldLogE2: [opus_val16; 2 * 21],
    /// Energy quantization error, size = channels * nbEBands (max 2*21 = 42)
    pub energyError: [opus_val16; 2 * 21],
    /// QEXT: enable quality extension encoding
    #[cfg(feature = "qext")]
    pub enable_qext: i32,
    /// QEXT: scaling factor (1 for 48 kHz, 2 for 96 kHz)
    #[cfg(feature = "qext")]
    pub qext_scale: i32,
    /// QEXT: old band energies for extension bands
    #[cfg(feature = "qext")]
    pub qext_oldBandE: [opus_val16; 2 * crate::celt::modes::data_96000::NB_QEXT_BANDS],
}

#[cfg(feature = "qext")]
#[inline]
fn qext_scale_for_mode(mode: &OpusCustomMode) -> i32 {
    if mode.Fs == 96000 && (mode.shortMdctSize == 240 || mode.shortMdctSize == 180) {
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
            end: mode.effEBands,
            bitrate: OPUS_BITRATE_MAX,
            vbr: 0,
            signalling: 0,
            constrained_vbr: 1,
            loss_rate: 0,
            lsb_depth: 24,
            lfe: 0,
            disable_inv: 0,
            arch,
            rng: 0,
            spread_decision: 0,
            delayedIntra: 0.0,
            tonal_average: 0,
            lastCodedBands: 0,
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
                signalType: 0,
                offset: 0,
            },
            preemph_memE: [0.0; 2],
            preemph_memD: [0.0; 2],
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
            oldBandE: [0.0; 2 * 21],
            oldLogE: [0.0; 2 * 21],
            oldLogE2: [0.0; 2 * 21],
            energyError: [0.0; 2 * 21],
            #[cfg(feature = "qext")]
            enable_qext: 0,
            #[cfg(feature = "qext")]
            qext_scale,
            #[cfg(feature = "qext")]
            qext_oldBandE: [0.0; 2 * crate::celt::modes::data_96000::NB_QEXT_BANDS],
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
        let nbEBands = self.mode.nbEBands;
        let cc = self.channels as usize;
        let overlap = self.mode.overlap;
        self.rng = 0;
        self.spread_decision = SPREAD_NORMAL;
        self.delayedIntra = 1 as opus_val32;
        self.tonal_average = 256;
        self.lastCodedBands = 0;
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
            signalType: 0,
            offset: 0,
        };
        self.preemph_memE = [0.0; 2];
        self.preemph_memD = [0.0; 2];
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
        (&mut self.oldBandE)[..cc * nbEBands].fill(0.0);
        (&mut self.oldLogE)[..cc * nbEBands].fill(-28.0);
        (&mut self.oldLogE2)[..cc * nbEBands].fill(-28.0);
        (&mut self.energyError)[..cc * nbEBands].fill(0.0);
        #[cfg(feature = "qext")]
        self.qext_oldBandE.fill(0.0);
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
        enc.qext_oldBandE.fill(1.0);
        enc.reset();
        assert!(enc.qext_oldBandE.iter().all(|&v| v == 0.0));
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
    in_0: &[opus_val32],
    len: i32,
    C: i32,
    tf_estimate: &mut opus_val16,
    tf_chan: &mut i32,
    allow_weak_transients: i32,
    weak_transient: &mut i32,
    tone_freq: opus_val16,
    toneishness: opus_val32,
) -> i32 {
    let mut i: i32 = 0;
    let mut mem0: opus_val32 = 0.;
    let mut mem1: opus_val32 = 0.;
    let mut is_transient: i32 = 0;
    let mut mask_metric: i32 = 0;
    let mut c: i32 = 0;
    let mut tf_max: opus_val16 = 0.;
    let mut len2: i32 = 0;
    let mut forward_decay: opus_val16 = 0.0625f32;
    static inv_table: [u8; 128] = [
        255, 255, 156, 110, 86, 70, 59, 51, 45, 40, 37, 33, 31, 28, 26, 25, 23, 22, 21, 20, 19, 18,
        17, 16, 16, 15, 15, 14, 13, 13, 12, 12, 12, 12, 11, 11, 11, 10, 10, 10, 9, 9, 9, 9, 9, 9,
        8, 8, 8, 8, 8, 7, 7, 7, 7, 7, 7, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 5, 5, 5,
        5, 5, 5, 5, 5, 5, 5, 5, 5, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
        4, 4, 4, 4, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2,
    ];
    // len = N + overlap; max 1920 + 240 = 2160 (QEXT 96kHz).
    const MAX_TRANSIENT: usize = 2400;
    debug_assert!((len as usize) <= MAX_TRANSIENT);
    let mut tmp = [0.0f32; MAX_TRANSIENT];
    *weak_transient = 0;
    if allow_weak_transients != 0 {
        forward_decay = 0.03125f32;
    }
    len2 = len / 2;
    c = 0;
    while c < C {
        let mut mean: opus_val32 = 0.;
        let mut unmask: i32 = 0;
        let mut norm: opus_val32 = 0.;
        let mut maxE: opus_val16 = 0.;
        mem0 = 0 as opus_val32;
        mem1 = 0 as opus_val32;
        i = 0;
        while i < len {
            let mut x: opus_val32 = 0.;
            let mut y: opus_val32 = 0.;
            x = in_0[(i + c * len) as usize];
            y = mem0 + x;
            /* Modified code to shorten dependency chains: */
            let mem00: f32 = mem0;
            mem0 = mem0 - x + 0.5f32 * mem1;
            mem1 = x - mem00;
            tmp[i as usize] = y;
            i += 1;
        }
        tmp[..12].fill(0.0);
        mean = 0 as opus_val32;
        mem0 = 0 as opus_val32;
        i = 0;
        while i < len2 {
            let x2: opus_val16 = tmp[(2 * i) as usize] * tmp[(2 * i) as usize]
                + tmp[(2 * i + 1) as usize] * tmp[(2 * i + 1) as usize];
            mean += x2;
            mem0 = x2 + (1.0f32 - forward_decay) * mem0;
            tmp[i as usize] = forward_decay * mem0;
            i += 1;
        }
        mem0 = 0 as opus_val32;
        maxE = 0 as opus_val16;
        i = len2 - 1;
        while i >= 0 {
            mem0 = tmp[i as usize] + 0.875f32 * mem0;
            tmp[i as usize] = 0.125f32 * mem0;
            maxE = if maxE > 0.125f32 * mem0 {
                maxE
            } else {
                0.125f32 * mem0
            };
            i -= 1;
        }
        mean = celt_sqrt((mean * maxE) * 0.5f32 * len2 as f32);
        norm = len2 as f32 / (1e-15f32 + mean);
        unmask = 0;
        assert!(!(tmp[0]).is_nan());
        assert!(!norm.is_nan());
        i = 12;
        while i < len2 - 5 {
            let mut id: i32 = 0;
            id = (if 0.0
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
            unmask += inv_table[id as usize] as i32;
            i += 4;
        }
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
    if allow_weak_transients != 0 && is_transient != 0 && mask_metric < 600 {
        is_transient = 0;
        *weak_transient = 1;
    }
    tf_max = if 0 as f32 > celt_sqrt((27 * mask_metric) as f32) - 42_f32 {
        0 as f32
    } else {
        celt_sqrt((27 * mask_metric) as f32) - 42_f32
    };
    *tf_estimate = (if 0 as f64
        > (0.0069f64 as opus_val32 * (if 163_f32 < tf_max { 163_f32 } else { tf_max })) as f64
            - 0.139f64
    {
        0 as f64
    } else {
        (0.0069f64 as opus_val32 * (if 163_f32 < tf_max { 163_f32 } else { tf_max })) as f64
            - 0.139f64
    })
    // here, a 64-bit sqrt __should__ be used
    .sqrt() as f32;
    is_transient
}
/// Upstream C: celt/celt_encoder.c:patch_transient_decision
fn patch_transient_decision(
    newE: &[opus_val16],
    oldE: &[opus_val16],
    nbEBands: i32,
    start: i32,
    end: i32,
    C: i32,
) -> i32 {
    let mut i: i32 = 0;
    let mut c: i32 = 0;
    let mut mean_diff: opus_val32 = 0 as opus_val32;
    let mut spread_old: [opus_val16; 26] = [0.; 26];
    if C == 1 {
        spread_old[start as usize] = oldE[start as usize];
        i = start + 1;
        while i < end {
            spread_old[i as usize] = if spread_old[(i - 1) as usize] - 1.0f32 > oldE[i as usize] {
                spread_old[(i - 1) as usize] - 1.0f32
            } else {
                oldE[i as usize]
            };
            i += 1;
        }
    } else {
        spread_old[start as usize] = if oldE[start as usize] > oldE[(start + nbEBands) as usize] {
            oldE[start as usize]
        } else {
            oldE[(start + nbEBands) as usize]
        };
        i = start + 1;
        while i < end {
            spread_old[i as usize] = if spread_old[(i - 1) as usize] - 1.0f32
                > (if oldE[i as usize] > oldE[(i + nbEBands) as usize] {
                    oldE[i as usize]
                } else {
                    oldE[(i + nbEBands) as usize]
                }) {
                spread_old[(i - 1) as usize] - 1.0f32
            } else if oldE[i as usize] > oldE[(i + nbEBands) as usize] {
                oldE[i as usize]
            } else {
                oldE[(i + nbEBands) as usize]
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
    c = 0;
    loop {
        i = if 2 > start { 2 } else { start };
        while i < end - 1 {
            let mut x1: opus_val16 = 0.;
            let mut x2: opus_val16 = 0.;
            x1 = if 0 as f32 > newE[(i + c * nbEBands) as usize] {
                0 as f32
            } else {
                newE[(i + c * nbEBands) as usize]
            };
            x2 = if 0 as f32 > spread_old[i as usize] {
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
        if c >= C {
            break;
        }
    }
    mean_diff /= (C * (end - 1 - (if 2 > start { 2 } else { start }))) as opus_val32;
    (mean_diff > 1.0f32) as i32
}
/// Upstream C: celt/celt_encoder.c:compute_mdcts
fn compute_mdcts(
    mode: &OpusCustomMode,
    shortBlocks: i32,
    in_0: &mut [celt_sig],
    out: &mut [celt_sig],
    C: i32,
    CC: i32,
    LM: i32,
    upsample: i32,
) {
    let overlap: i32 = mode.overlap as i32;
    let mut N: i32 = 0;
    let mut B: i32 = 0;
    let mut shift: i32 = 0;
    let mut i: i32 = 0;
    let mut b: i32 = 0;
    let mut c: i32 = 0;
    if shortBlocks != 0 {
        B = shortBlocks;
        N = mode.shortMdctSize;
        shift = mode.maxLM;
    } else {
        B = 1;
        N = mode.shortMdctSize << LM;
        shift = mode.maxLM - LM;
    }
    c = 0;
    loop {
        b = 0;
        while b < B {
            /* Interleaving the sub-frames while doing the MDCTs */
            let in_base = (c * (B * N + overlap) + b * N) as usize;
            let in_len = (N + overlap) as usize;
            let out_base = (b + c * N * B) as usize;
            let out_len = (N * B) as usize;

            mdct_forward(
                &mode.mdct,
                &in_0[in_base..in_base + in_len],
                &mut out[out_base..out_base + out_len],
                mode.window,
                overlap as usize,
                shift as usize,
                B as usize,
            );
            b += 1;
        }
        c += 1;
        if c >= CC {
            break;
        }
    }
    if CC == 2 && C == 1 {
        i = 0;
        while i < B * N {
            out[i as usize] = 0.5f32 * out[i as usize] + 0.5f32 * out[(B * N + i) as usize];
            i += 1;
        }
    }
    if upsample != 1 {
        c = 0;
        loop {
            let bound: i32 = B * N / upsample;
            i = 0;
            while i < bound {
                out[(c * B * N + i) as usize] *= upsample as f32;
                i += 1;
            }
            let base = (c * B * N + bound) as usize;
            let len = (B * N - bound) as usize;
            out[base..base + len].fill(0.0);
            c += 1;
            if c >= C {
                break;
            }
        }
    }
}
/// Upstream C: celt/celt_encoder.c:celt_preemphasis
fn celt_preemphasis(
    pcmp: &[opus_val16],
    inp: &mut [celt_sig],
    N: i32,
    CC: i32,
    upsample: i32,
    coef: &[opus_val16],
    mem: &mut celt_sig,
    clip: i32,
) {
    let mut i: i32 = 0;
    let mut coef0: opus_val16 = 0.;
    let mut m: celt_sig = 0.;
    let mut Nu: i32 = 0;
    coef0 = coef[0];
    m = *mem;
    if coef[1] == 0 as f32 && upsample == 1 && clip == 0 {
        i = 0;
        while i < N {
            let mut x: opus_val16 = 0.;
            x = pcmp[(CC * i) as usize] * CELT_SIG_SCALE;
            inp[i as usize] = x - m;
            m = coef0 * x;
            i += 1;
        }
        *mem = m;
        return;
    }
    Nu = N / upsample;
    if upsample != 1 {
        inp[..N as usize].fill(0.0);
    }
    i = 0;
    while i < Nu {
        inp[(i * upsample) as usize] = pcmp[(CC * i) as usize] * CELT_SIG_SCALE;
        i += 1;
    }
    if clip != 0 {
        i = 0;
        while i < Nu {
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
    while i < N {
        let mut x_0: opus_val16 = 0.;
        x_0 = inp[i as usize];
        inp[i as usize] = x_0 - m;
        m = coef0 * x_0;
        i += 1;
    }
    *mem = m;
}
/// Upstream C: celt/celt_encoder.c:l1_metric
fn l1_metric(tmp: &[celt_norm], N: i32, LM: i32, bias: opus_val16) -> opus_val32 {
    let mut L1: opus_val32 = 0 as opus_val32;
    let mut i: i32 = 0;
    while i < N {
        L1 += tmp[i as usize].abs();
        i += 1;
    }
    L1 = L1 + LM as f32 * bias * L1;
    L1
}
/// Upstream C: celt/celt_encoder.c:tf_analysis
fn tf_analysis(
    m: &OpusCustomMode,
    len: i32,
    isTransient: i32,
    tf_res: &mut [i32],
    lambda: i32,
    X: &[celt_norm],
    N0: i32,
    LM: i32,
    tf_estimate: opus_val16,
    tf_chan: i32,
    importance: &[i32],
) -> i32 {
    let mut i: i32 = 0;
    let mut cost0: i32 = 0;
    let mut cost1: i32 = 0;
    let mut sel: i32 = 0;
    let mut selcost: [i32; 2] = [0; 2];
    let mut tf_select: i32 = 0;
    let mut bias: opus_val16 = 0.;
    bias = 0.04f32
        * (if -0.25f32 > 0.5f32 - tf_estimate {
            -0.25f32
        } else {
            0.5f32 - tf_estimate
        });
    // len = nbEBands, max 21 (std) + 14 (QEXT) = 35.
    const MAX_TF_BANDS: usize = 40;
    debug_assert!((len as usize) <= MAX_TF_BANDS);
    let mut metric = [0i32; MAX_TF_BANDS];
    let band_size =
        ((m.eBands[len as usize] as i32 - m.eBands[(len - 1) as usize] as i32) << LM) as usize;
    // Last band size * M; max ~128 (48kHz) or ~256 (QEXT).
    const MAX_BAND_TMP: usize = 256;
    debug_assert!(band_size <= MAX_BAND_TMP);
    let mut tmp = [0.0f32; MAX_BAND_TMP];
    let mut tmp_1 = [0.0f32; MAX_BAND_TMP];
    let mut path0 = [0i32; MAX_TF_BANDS];
    let mut path1 = [0i32; MAX_TF_BANDS];
    i = 0;
    while i < len {
        let mut k: i32 = 0;
        let mut N: i32 = 0;
        let mut narrow: i32 = 0;
        let mut L1: opus_val32 = 0.;
        let mut best_L1: opus_val32 = 0.;
        let mut best_level: i32 = 0;
        N = (m.eBands[(i + 1) as usize] as i32 - m.eBands[i as usize] as i32) << LM;
        narrow = (m.eBands[(i + 1) as usize] as i32 - m.eBands[i as usize] as i32 == 1) as i32;
        let x_offset = (tf_chan * N0 + ((m.eBands[i as usize] as i32) << LM)) as usize;
        tmp[..N as usize].copy_from_slice(&X[x_offset..x_offset + N as usize]);
        L1 = l1_metric(&tmp, N, if isTransient != 0 { LM } else { 0 }, bias);
        best_L1 = L1;
        if isTransient != 0 && narrow == 0 {
            tmp_1[..N as usize].copy_from_slice(&tmp[..N as usize]);
            haar1(&mut tmp_1, N >> LM, (1) << LM);
            L1 = l1_metric(&tmp_1, N, LM + 1, bias);
            if L1 < best_L1 {
                best_L1 = L1;
                best_level = -1;
            }
        }
        k = 0;
        while k < LM + !(isTransient != 0 || narrow != 0) as i32 {
            let mut B: i32 = 0;
            if isTransient != 0 {
                B = LM - k - 1;
            } else {
                B = k + 1;
            }
            haar1(&mut tmp, N >> k, (1) << k);
            L1 = l1_metric(&tmp, N, B, bias);
            if L1 < best_L1 {
                best_L1 = L1;
                best_level = k + 1;
            }
            k += 1;
        }
        if isTransient != 0 {
            metric[i as usize] = 2 * best_level;
        } else {
            metric[i as usize] = -(2) * best_level;
        }
        if narrow != 0 && (metric[i as usize] == 0 || metric[i as usize] == -(2) * LM) {
            metric[i as usize] -= 1;
        }
        i += 1;
    }
    tf_select = 0;
    sel = 0;
    while sel < 2 {
        cost0 = importance[0]
            * (metric[0]
                - 2 * tf_select_table[LM as usize][(4 * isTransient + 2 * sel) as usize] as i32)
                .abs();
        cost1 = importance[0]
            * (metric[0]
                - 2 * tf_select_table[LM as usize][(4 * isTransient + 2 * sel + 1) as usize]
                    as i32)
                .abs()
            + (if isTransient != 0 { 0 } else { lambda });
        i = 1;
        while i < len {
            let mut curr0: i32 = 0;
            let mut curr1: i32 = 0;
            curr0 = if cost0 < cost1 + lambda {
                cost0
            } else {
                cost1 + lambda
            };
            curr1 = if cost0 + lambda < cost1 {
                cost0 + lambda
            } else {
                cost1
            };
            cost0 = curr0
                + importance[i as usize]
                    * (metric[i as usize]
                        - 2 * tf_select_table[LM as usize][(4 * isTransient + 2 * sel) as usize]
                            as i32)
                        .abs();
            cost1 = curr1
                + importance[i as usize]
                    * (metric[i as usize]
                        - 2 * tf_select_table[LM as usize][(4 * isTransient + 2 * sel + 1) as usize]
                            as i32)
                        .abs();
            i += 1;
        }
        cost0 = if cost0 < cost1 { cost0 } else { cost1 };
        selcost[sel as usize] = cost0;
        sel += 1;
    }
    if selcost[1_usize] < selcost[0_usize] && isTransient != 0 {
        tf_select = 1;
    }
    cost0 = importance[0]
        * (metric[0]
            - 2 * tf_select_table[LM as usize][(4 * isTransient + 2 * tf_select) as usize] as i32)
            .abs();
    cost1 = importance[0]
        * (metric[0]
            - 2 * tf_select_table[LM as usize][(4 * isTransient + 2 * tf_select + 1) as usize]
                as i32)
            .abs()
        + (if isTransient != 0 { 0 } else { lambda });
    i = 1;
    while i < len {
        let mut curr0_0: i32 = 0;
        let mut curr1_0: i32 = 0;
        let mut from0: i32 = 0;
        let mut from1: i32 = 0;
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
                    - 2 * tf_select_table[LM as usize][(4 * isTransient + 2 * tf_select) as usize]
                        as i32)
                    .abs();
        cost1 = curr1_0
            + importance[i as usize]
                * (metric[i as usize]
                    - 2 * tf_select_table[LM as usize]
                        [(4 * isTransient + 2 * tf_select + 1) as usize]
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
    isTransient: i32,
    tf_res: &mut [i32],
    LM: i32,
    mut tf_select: i32,
    enc: &mut ec_enc,
) {
    let mut curr: i32 = 0;
    let mut i: i32 = 0;
    let mut tf_select_rsv: i32 = 0;
    let mut tf_changed: i32 = 0;
    let mut logp: i32 = 0;
    let mut budget: u32 = 0;
    let mut tell: u32 = 0;
    budget = enc.storage.wrapping_mul(8);
    tell = ec_tell(enc) as u32;
    logp = if isTransient != 0 { 2 } else { 4 };
    tf_select_rsv = (LM > 0 && tell.wrapping_add(logp as u32).wrapping_add(1) <= budget) as i32;
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
        logp = if isTransient != 0 { 4 } else { 5 };
        i += 1;
    }
    if tf_select_rsv != 0
        && tf_select_table[LM as usize][((4 * isTransient) + tf_changed) as usize] as i32
            != tf_select_table[LM as usize][(4 * isTransient + 2 + tf_changed) as usize] as i32
    {
        ec_enc_bit_logp(enc, tf_select, 1);
    } else {
        tf_select = 0;
    }
    i = start;
    while i < end {
        tf_res[i as usize] = tf_select_table[LM as usize]
            [(4 * isTransient + 2 * tf_select + tf_res[i as usize]) as usize]
            as i32;
        i += 1;
    }
}
/// Upstream C: celt/celt_encoder.c:alloc_trim_analysis
fn alloc_trim_analysis(
    m: &OpusCustomMode,
    X: &[celt_norm],
    bandLogE: &[opus_val16],
    end: i32,
    LM: i32,
    C: i32,
    N0: i32,
    analysis: &AnalysisInfo,
    stereo_saving: &mut opus_val16,
    tf_estimate: opus_val16,
    intensity: i32,
    surround_trim: opus_val16,
    equiv_rate: i32,
    _arch: Arch,
) -> i32 {
    let mut i: i32 = 0;
    let mut diff: opus_val32 = 0 as opus_val32;
    let mut c: i32 = 0;
    let mut trim_index: i32 = 0;
    let mut trim: opus_val16 = 5.0f32;
    let mut logXC: opus_val16 = 0.;
    let mut logXC2: opus_val16 = 0.;
    if equiv_rate < 64000 {
        trim = 4.0f32;
    } else if equiv_rate < 80000 {
        let frac: i32 = (equiv_rate - 64000) >> 10;
        trim = 4.0f32 + 1.0f32 / 16.0f32 * frac as f32;
    }
    if C == 2 {
        let mut sum: opus_val16 = 0 as opus_val16;
        let mut minXC: opus_val16 = 0.;
        i = 0;
        while i < 8 {
            let mut partial: opus_val32 = 0.;
            let band_off = ((m.eBands[i as usize] as i32) << LM) as usize;
            let band_off2 = (N0 + ((m.eBands[i as usize] as i32) << LM)) as usize;
            let band_len =
                ((m.eBands[(i + 1) as usize] as i32 - m.eBands[i as usize] as i32) << LM) as usize;
            partial = celt_inner_prod(
                &X[band_off..band_off + band_len],
                &X[band_off2..band_off2 + band_len],
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
        minXC = sum;
        i = 8;
        while i < intensity {
            let mut partial_0: opus_val32 = 0.;
            let band_off = ((m.eBands[i as usize] as i32) << LM) as usize;
            let band_off2 = (N0 + ((m.eBands[i as usize] as i32) << LM)) as usize;
            let band_len =
                ((m.eBands[(i + 1) as usize] as i32 - m.eBands[i as usize] as i32) << LM) as usize;
            partial_0 = celt_inner_prod(
                &X[band_off..band_off + band_len],
                &X[band_off2..band_off2 + band_len],
                band_len,
                _arch,
            );
            minXC = if minXC < (partial_0).abs() {
                minXC
            } else {
                (partial_0).abs()
            };
            i += 1;
        }
        minXC = if 1.0f32 < (minXC).abs() {
            1.0f32
        } else {
            (minXC).abs()
        };
        logXC = celt_log2(1.001f32 - sum * sum);
        logXC2 = if 0.5f32 * logXC > celt_log2(1.001f32 - minXC * minXC) {
            0.5f32 * logXC
        } else {
            celt_log2(1.001f32 - minXC * minXC)
        };
        trim += if -4.0f32 > 0.75f32 * logXC {
            -4.0f32
        } else {
            0.75f32 * logXC
        };
        *stereo_saving = if *stereo_saving + 0.25f32 < -(0.5f32 * logXC2) {
            *stereo_saving + 0.25f32
        } else {
            -(0.5f32 * logXC2)
        };
    }
    c = 0;
    loop {
        i = 0;
        while i < end - 1 {
            diff += bandLogE[(i + c * m.nbEBands as i32) as usize] * (2 + 2 * i - end) as f32;
            i += 1;
        }
        c += 1;
        if c >= C {
            break;
        }
    }
    diff /= (C * (end - 1)) as f32;
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
fn stereo_analysis(m: &OpusCustomMode, X: &[celt_norm], LM: i32, N0: i32) -> i32 {
    let mut i: i32 = 0;
    let mut thetas: i32 = 0;
    let mut sumLR: opus_val32 = EPSILON;
    let mut sumMS: opus_val32 = EPSILON;
    i = 0;
    while i < 13 {
        let mut j: i32 = 0;
        j = (m.eBands[i as usize] as i32) << LM;
        while j < (m.eBands[(i + 1) as usize] as i32) << LM {
            let mut L: opus_val32 = 0.;
            let mut R: opus_val32 = 0.;
            let mut M: opus_val32 = 0.;
            let mut S: opus_val32 = 0.;
            L = X[j as usize];
            R = X[(N0 + j) as usize];
            M = L + R;
            S = L - R;
            sumLR += (L).abs() + (R).abs();
            sumMS += (M).abs() + (S).abs();
            j += 1;
        }
        i += 1;
    }
    #[allow(clippy::approx_constant)]
    // Intentional: C reference uses 0.707107, not exact 1/sqrt(2)
    let frac_1_sqrt_2 = 0.707107f32;
    sumMS *= frac_1_sqrt_2;
    thetas = 13;
    if LM <= 1 {
        thetas -= 8;
    }
    ((((m.eBands[13] as i32) << (LM + 1)) + thetas) as f32 * sumMS
        > ((m.eBands[13] as i32) << (LM + 1)) as f32 * sumLR) as i32
}
/// Upstream C: celt/celt_encoder.c:median_of_5
fn median_of_5(x: &[opus_val16]) -> opus_val16 {
    let mut t0: opus_val16;
    let mut t1: opus_val16;

    let mut t3: opus_val16;
    let mut t4: opus_val16;
    let t2: opus_val16 = x[2];
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
fn median_of_3(x: &[opus_val16]) -> opus_val16 {
    let t0: opus_val16;
    let t1: opus_val16;

    if x[0] > x[1] {
        t0 = x[1];
        t1 = x[0];
    } else {
        t0 = x[0];
        t1 = x[1];
    }
    let t2: opus_val16 = x[2];
    if t1 < t2 {
        t1
    } else if t0 < t2 {
        t2
    } else {
        t0
    }
}
/// Upstream C: celt/celt_encoder.c:dynalloc_analysis
fn dynalloc_analysis(
    bandLogE: &[opus_val16],
    bandLogE2: &[opus_val16],
    oldBandE: &[opus_val16],
    nbEBands: i32,
    start: i32,
    end: i32,
    C: i32,
    offsets: &mut [i32],
    lsb_depth: i32,
    logN: &[i16],
    isTransient: i32,
    vbr: i32,
    constrained_vbr: i32,
    eBands: &[i16],
    LM: i32,
    effectiveBytes: i32,
    tot_boost_: &mut i32,
    lfe: i32,
    surround_dynalloc: &[opus_val16],
    analysis: &AnalysisInfo,
    importance: &mut [i32],
    spread_weight: &mut [i32],
    tone_freq: opus_val16,
    toneishness: opus_val32,
) -> opus_val16 {
    let mut i: i32 = 0;
    let mut c: i32 = 0;
    let mut tot_boost: i32 = 0;
    let mut maxDepth: opus_val16 = 0.;
    // C * nbEBands max: 2 * 35 = 70.
    const MAX_C_BANDS: usize = 80;
    debug_assert!(((C * nbEBands) as usize) <= MAX_C_BANDS);
    let mut follower = [0.0f32; MAX_C_BANDS];
    let mut noise_floor = [0.0f32; MAX_C_BANDS];
    offsets[..nbEBands as usize].fill(0);
    maxDepth = -31.9f32;
    i = 0;
    while i < end {
        noise_floor[i as usize] =
            0.0625f32 * logN[i as usize] as opus_val32 + 0.5f32 + (9 - lsb_depth) as f32
                - eMeans[i as usize]
                + 0.0062f64 as opus_val32 * ((i + 5) * (i + 5)) as opus_val32;
        i += 1;
    }
    c = 0;
    loop {
        i = 0;
        while i < end {
            maxDepth = if maxDepth > bandLogE[(c * nbEBands + i) as usize] - noise_floor[i as usize]
            {
                maxDepth
            } else {
                bandLogE[(c * nbEBands + i) as usize] - noise_floor[i as usize]
            };
            i += 1;
        }
        c += 1;
        if c >= C {
            break;
        }
    }
    const MAX_BANDS_DA: usize = 40;
    debug_assert!((nbEBands as usize) <= MAX_BANDS_DA);
    let mut mask = [0.0f32; MAX_BANDS_DA];
    let mut sig = [0.0f32; MAX_BANDS_DA];
    i = 0;
    while i < end {
        mask[i as usize] = bandLogE[i as usize] - noise_floor[i as usize];
        i += 1;
    }
    if C == 2 {
        i = 0;
        while i < end {
            mask[i as usize] =
                if mask[i as usize] > bandLogE[(nbEBands + i) as usize] - noise_floor[i as usize] {
                    mask[i as usize]
                } else {
                    bandLogE[(nbEBands + i) as usize] - noise_floor[i as usize]
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
        let smr: opus_val16 = sig[i as usize]
            - (if (if 0 as f32 > maxDepth - 12.0f32 {
                0 as f32
            } else {
                maxDepth - 12.0f32
            }) > mask[i as usize]
            {
                if 0 as f32 > maxDepth - 12.0f32 {
                    0 as f32
                } else {
                    maxDepth - 12.0f32
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
    let mut bandLogE3: Vec<opus_val16> = vec![0.0; nbEBands as usize];
    if effectiveBytes >= (30 + 5 * LM) && lfe == 0 {
        let mut last: i32 = 0;
        c = 0;
        loop {
            let mut offset: opus_val16 = 0.;
            let mut tmp: opus_val16 = 0.;
            let fb = (c * nbEBands) as usize;
            bandLogE3[..end as usize].copy_from_slice(&bandLogE2[fb..fb + end as usize]);
            if LM == 0 {
                // For 2.5 ms frames, the first 8 bands have just one bin, so the
                // energy is highly unreliable (high variance). For that reason,
                // we take the max with the previous energy so that at least 2 bins
                // are getting used.
                for i in 0..std::cmp::min(8, end as usize) {
                    bandLogE3[i] = if bandLogE2[(c * nbEBands) as usize + i]
                        > oldBandE[(c * nbEBands) as usize + i]
                    {
                        bandLogE2[(c * nbEBands) as usize + i]
                    } else {
                        oldBandE[(c * nbEBands) as usize + i]
                    };
                }
            }
            follower[fb] = bandLogE3[0];
            i = 1;
            while i < end {
                if bandLogE3[i as usize] > bandLogE3[(i - 1) as usize] + 0.5f32 {
                    last = i;
                }
                follower[fb + i as usize] =
                    if follower[fb + (i - 1) as usize] + 1.5f32 < bandLogE3[i as usize] {
                        follower[fb + (i - 1) as usize] + 1.5f32
                    } else {
                        bandLogE3[i as usize]
                    };
                i += 1;
            }
            i = last - 1;
            while i >= 0 {
                follower[fb + i as usize] = if follower[fb + i as usize]
                    < (if follower[fb + (i + 1) as usize] + 2.0f32 < bandLogE3[i as usize] {
                        follower[fb + (i + 1) as usize] + 2.0f32
                    } else {
                        bandLogE3[i as usize]
                    }) {
                    follower[fb + i as usize]
                } else if follower[fb + (i + 1) as usize] + 2.0f32 < bandLogE3[i as usize] {
                    follower[fb + (i + 1) as usize] + 2.0f32
                } else {
                    bandLogE3[i as usize]
                };
                i -= 1;
            }
            offset = 1.0f32;
            i = 2;
            while i < end - 2 {
                let med = median_of_5(&bandLogE3[(i - 2) as usize..(i + 3) as usize]) - offset;
                follower[fb + i as usize] = if follower[fb + i as usize] > med {
                    follower[fb + i as usize]
                } else {
                    med
                };
                i += 1;
            }
            tmp = median_of_3(&bandLogE3[0..3]) - offset;
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
            tmp = median_of_3(&bandLogE3[(end - 3) as usize..end as usize]) - offset;
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
            if c >= C {
                break;
            }
        }
        if C == 2 {
            i = start;
            while i < end {
                follower[(nbEBands + i) as usize] =
                    if follower[(nbEBands + i) as usize] > follower[i as usize] - 4.0f32 {
                        follower[(nbEBands + i) as usize]
                    } else {
                        follower[i as usize] - 4.0f32
                    };
                follower[i as usize] =
                    if follower[i as usize] > follower[(nbEBands + i) as usize] - 4.0f32 {
                        follower[i as usize]
                    } else {
                        follower[(nbEBands + i) as usize] - 4.0f32
                    };
                follower[i as usize] = 0.5f32
                    * ((if 0 as f32 > bandLogE[i as usize] - follower[i as usize] {
                        0 as f32
                    } else {
                        bandLogE[i as usize] - follower[i as usize]
                    }) + (if 0 as f32
                        > bandLogE[(nbEBands + i) as usize] - follower[(nbEBands + i) as usize]
                    {
                        0 as f32
                    } else {
                        bandLogE[(nbEBands + i) as usize] - follower[(nbEBands + i) as usize]
                    }));
                i += 1;
            }
        } else {
            i = start;
            while i < end {
                follower[i as usize] = if 0 as f32 > bandLogE[i as usize] - follower[i as usize] {
                    0 as f32
                } else {
                    bandLogE[i as usize] - follower[i as usize]
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
        if (vbr == 0 || constrained_vbr != 0) && isTransient == 0 {
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
                if freq_bin >= eBands[i as usize] as i32
                    && freq_bin <= eBands[(i + 1) as usize] as i32
                {
                    follower[i as usize] += 2.0;
                }
                if freq_bin >= eBands[i as usize] as i32 - 1
                    && freq_bin <= eBands[(i + 1) as usize] as i32 + 1
                {
                    follower[i as usize] += 1.0;
                }
                if freq_bin >= eBands[i as usize] as i32 - 2
                    && freq_bin <= eBands[(i + 1) as usize] as i32 + 2
                {
                    follower[i as usize] += 1.0;
                }
                if freq_bin >= eBands[i as usize] as i32 - 3
                    && freq_bin <= eBands[(i + 1) as usize] as i32 + 3
                {
                    follower[i as usize] += 0.5;
                }
            }
            if freq_bin >= eBands[end as usize] as i32 {
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
            let mut width: i32 = 0;
            let mut boost: i32 = 0;
            let mut boost_bits: i32 = 0;
            follower[i as usize] = if follower[i as usize] < 4_f32 {
                follower[i as usize]
            } else {
                4_f32
            };
            width = (C * (eBands[(i + 1) as usize] as i32 - eBands[i as usize] as i32)) << LM;
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
            if (vbr == 0 || constrained_vbr != 0 && isTransient == 0)
                && (tot_boost + boost_bits) >> BITRES >> 3 > 2 * effectiveBytes / 3
            {
                let cap: i32 = (2 * effectiveBytes / 3) << BITRES << 3;
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
    maxDepth
}
/// 2nd-order LPC analysis using the forward/backward covariance method.
/// Returns `true` on failure (ill-conditioned).
///
/// Upstream C: celt/celt_encoder.c:tone_lpc
fn tone_lpc(x: &[opus_val16], len: usize, delay: usize, lpc: &mut [opus_val32; 2]) -> bool {
    debug_assert!(len > 2 * delay);
    // Compute forward correlations.
    let mut r00: opus_val32 = 0.0;
    let mut r01: opus_val32 = 0.0;
    let mut r02: opus_val32 = 0.0;
    for i in 0..len - 2 * delay {
        r00 += x[i] * x[i];
        r01 += x[i] * x[i + delay];
        r02 += x[i] * x[i + 2 * delay];
    }
    let mut edges: opus_val32 = 0.0;
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
    input: &[celt_sig],
    CC: i32,
    N: i32,
    toneishness: &mut opus_val32,
    Fs: i32,
) -> opus_val16 {
    let n = N as usize;
    let mut delay: usize = 1;
    let mut lpc = [0.0f32; 2];
    let mut x: Vec<opus_val16> = vec![0.0; n];
    // In float build, SHR32/PSHR32/ADD32 are identity ops, so the
    // downscaling is just averaging channels for stereo.
    if CC == 2 {
        for i in 0..n {
            x[i] = (input[i] * 0.5) + (input[i + n] * 0.5);
        }
    } else {
        x[..n].copy_from_slice(&input[..n]);
    }
    let mut fail = tone_lpc(&x, n, delay, &mut lpc);
    // If our LPC filter resonates too close to DC, retry with down-sampling.
    while delay <= (Fs / 3000) as usize && (fail || (lpc[0] > 1.0 && lpc[1] < 0.0)) {
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
fn run_prefilter(
    st: &mut OpusCustomEncoder,
    in_0: &mut [celt_sig],
    CC: i32,
    N: i32,
    prefilter_tapset: i32,
    pitch: &mut i32,
    gain: &mut opus_val16,
    qgain: &mut i32,
    enabled: i32,
    tf_estimate: opus_val16,
    nbAvailableBytes: i32,
    analysis: &AnalysisInfo,
    tone_freq: opus_val16,
    toneishness: opus_val32,
) -> i32 {
    #[cfg(feature = "qext")]
    let qext_scale = st.qext_scale;
    #[cfg(not(feature = "qext"))]
    let qext_scale = 1;
    let mut pitch_index: i32 = 0;
    let mut gain1: opus_val16 = 0.;
    let mut pf_threshold: opus_val16;
    let mut pf_on: i32;
    let mut qg: i32 = 0;
    let mode = st.mode;
    let overlap = mode.overlap as i32;
    let max_period = COMBFILTER_MAXPERIOD * qext_scale;
    let min_period = COMBFILTER_MINPERIOD * qext_scale;
    let pre_chan_len = (N + max_period) as usize;
    // CC * (N + max_period) max: 2 * (1920 + 2048) = 7936.
    const MAX_PRE: usize = 8000;
    debug_assert!((CC as usize) * pre_chan_len <= MAX_PRE);
    let mut _pre = [0.0f32; MAX_PRE];
    // pre[c] starts at c * pre_chan_len in _pre
    for c in 0..CC as usize {
        let pre_base = c * pre_chan_len;
        let max_period_u = max_period as usize;
        _pre[pre_base..pre_base + max_period_u]
            .copy_from_slice(&st.prefilter_mem[c * max_period_u..(c + 1) * max_period_u]);
        let in_src = c * (N + overlap) as usize + overlap as usize;
        _pre[pre_base + max_period_u..pre_base + max_period_u + N as usize]
            .copy_from_slice(&in_0[in_src..in_src + N as usize]);
    }
    if enabled != 0 && toneishness > 0.99 {
        // If we detect that the signal is dominated by a single tone, don't rely
        // on the standard pitch estimator, as it can become unreliable.
        let mut multiple = 1i32;
        let mut tf = tone_freq * qext_scale as f32;
        // Using aliased version of the postfilter above 24 kHz.
        // Threshold is purposely slightly above pi to avoid triggering for Fs=48kHz.
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
        // (max_period + N) >> 1 max: (2048 + 1920) / 2 = 1984.
        const MAX_PITCH_BUF: usize = 2000;
        let pitch_buf_len = ((max_period + N) >> 1) as usize;
        debug_assert!(pitch_buf_len <= MAX_PITCH_BUF);
        let mut pitch_buf = [0.0f32; MAX_PITCH_BUF];
        {
            let ds_len = (max_period + N) as usize;
            let ch0 = &_pre[..ds_len];
            if CC == 2 {
                let ch1 = &_pre[pre_chan_len..pre_chan_len + ds_len];
                pitch_downsample(&[ch0, ch1], &mut pitch_buf[..pitch_buf_len], pitch_buf_len, 2, st.arch);
            } else {
                pitch_downsample(&[ch0], &mut pitch_buf[..pitch_buf_len], pitch_buf_len, 2, st.arch);
            }
        }
        pitch_index = pitch_search(
            &pitch_buf[(max_period >> 1) as usize..],
            pitch_buf.as_slice(),
            N,
            max_period - 3 * min_period,
            st.arch,
        );
        pitch_index = max_period - pitch_index;
        gain1 = remove_doubling(
            pitch_buf.as_slice(),
            max_period,
            min_period,
            N,
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
            gain1 = 0 as opus_val16;
        }
    } else {
        gain1 = 0 as opus_val16;
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
    if nbAvailableBytes < 25 {
        pf_threshold += 0.1f32;
    }
    if nbAvailableBytes < 35 {
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
        gain1 = 0 as opus_val16;
        pf_on = 0;
        qg = 0;
    } else {
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

    for c in 0..CC as usize {
        let offset: i32 = mode.shortMdctSize - overlap;
        st.prefilter_period = st.prefilter_period.max(COMBFILTER_MINPERIOD);
        // Copy in_mem overlap into in_0
        let in_dst = c * (N + overlap) as usize;
        in_0[in_dst..in_dst + overlap as usize]
            .copy_from_slice(&st.in_mem[c * overlap as usize..(c + 1) * overlap as usize]);
        // Measure energy before comb filter
        for i in 0..N as usize {
            before[c] += in_0[c * (N + overlap) as usize + overlap as usize + i].abs();
        }
        {
            let pre_base = c * pre_chan_len;
            let pre_slice = &_pre[pre_base..pre_base + pre_chan_len];
            let in_base = c * (N + overlap) as usize + overlap as usize;
            let in_slice = &mut in_0[in_base..in_base + N as usize];
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
                N - offset,
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
        for i in 0..N as usize {
            after[c] += in_0[c * (N + overlap) as usize + overlap as usize + i].abs();
        }
    }

    // Check if comb filter made things worse
    if CC == 2 {
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
        for c in 0..CC as usize {
            let offset: i32 = mode.shortMdctSize - overlap;
            let pre_base = c * pre_chan_len;
            let pre_slice = &_pre[pre_base..pre_base + pre_chan_len];
            let in_base = c * (N + overlap) as usize + overlap as usize;
            // Revert: copy original pre data back
            in_0[in_base..in_base + N as usize]
                .copy_from_slice(&pre_slice[max_period as usize..max_period as usize + N as usize]);
            // Re-apply transition with gain=0
            let in_slice = &mut in_0[in_base..in_base + N as usize];
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

    for c in 0..CC as usize {
        // Copy end of in_0 back into in_mem overlap
        let in_src = c * (N + overlap) as usize + N as usize;
        st.in_mem[c * overlap as usize..(c + 1) * overlap as usize]
            .copy_from_slice(&in_0[in_src..in_src + overlap as usize]);
        // Update prefilter_mem from _pre
        let pre_base = c * pre_chan_len;
        let max_period_u = max_period as usize;
        let pfm_base = c * max_period_u;
        if N > max_period {
            st.prefilter_mem[pfm_base..pfm_base + max_period_u].copy_from_slice(
                &_pre[pre_base + N as usize..pre_base + N as usize + max_period_u],
            );
        } else {
            // Shift prefilter_mem left by N
            st.prefilter_mem
                .copy_within(pfm_base + N as usize..pfm_base + max_period_u, pfm_base);
            // Copy last N samples from _pre
            st.prefilter_mem[pfm_base + max_period_u - N as usize..pfm_base + max_period_u]
                .copy_from_slice(
                    &_pre[pre_base + max_period_u..pre_base + max_period_u + N as usize],
                );
        }
    }
    *gain = gain1;
    *pitch = pitch_index;
    *qgain = qg;
    pf_on
}
/// Upstream C: celt/celt_encoder.c:compute_vbr
fn compute_vbr(
    mode: &OpusCustomMode,
    analysis: &AnalysisInfo,
    base_target: i32,
    LM: i32,
    bitrate: i32,
    lastCodedBands: i32,
    C: i32,
    intensity: i32,
    constrained_vbr: i32,
    mut stereo_saving: opus_val16,
    tot_boost: i32,
    tf_estimate: opus_val16,
    pitch_change: i32,
    maxDepth: opus_val16,
    lfe: i32,
    has_surround_mask: i32,
    surround_masking: opus_val16,
    temporal_vbr: opus_val16,
) -> i32 {
    let mut target: i32 = 0;
    let mut coded_bins: i32 = 0;
    let mut coded_bands: i32 = 0;
    let mut tf_calibration: opus_val16 = 0.;
    let mut nbEBands: i32 = 0;
    let eBands = &mode.eBands;
    nbEBands = mode.nbEBands as i32;
    coded_bands = if lastCodedBands != 0 {
        lastCodedBands
    } else {
        nbEBands
    };
    coded_bins = (eBands[coded_bands as usize] as i32) << LM;
    if C == 2 {
        coded_bins += (eBands[(if intensity < coded_bands {
            intensity
        } else {
            coded_bands
        }) as usize] as i32)
            << LM;
    }
    target = base_target;
    if analysis.valid != 0 && (analysis.activity as f64) < 0.4f64 {
        target -= ((coded_bins << BITRES) as f32 * (0.4f32 - analysis.activity)) as i32;
    }
    if C == 2 {
        let mut coded_stereo_bands: i32 = 0;
        let mut coded_stereo_dof: i32 = 0;
        let mut max_frac: opus_val16 = 0.;
        coded_stereo_bands = if intensity < coded_bands {
            intensity
        } else {
            coded_bands
        };
        coded_stereo_dof =
            ((eBands[coded_stereo_bands as usize] as i32) << LM) - coded_stereo_bands;
        max_frac = 0.8f32 * coded_stereo_dof as opus_val32 / coded_bins as opus_val16;
        stereo_saving = if stereo_saving < 1.0f32 {
            stereo_saving
        } else {
            1.0f32
        };
        target -= (if (max_frac * target as f32)
            < (stereo_saving - 0.1f32) * (coded_stereo_dof << 3) as opus_val32
        {
            max_frac * target as f32
        } else {
            (stereo_saving - 0.1f32) * (coded_stereo_dof << 3) as opus_val32
        }) as i32;
    }
    target += tot_boost - ((19) << LM);
    tf_calibration = 0.044f32;
    target += ((tf_estimate - tf_calibration) * target as f32) as i32;
    if analysis.valid != 0 && lfe == 0 {
        let mut tonal_target: i32 = 0;
        let mut tonal: f32 = 0.;
        tonal = (if 0.0f32 > analysis.tonality - 0.15f32 {
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
        let surround_target: i32 =
            target + (surround_masking * (coded_bins << 3) as opus_val32) as i32;
        target = if target / 4 > surround_target {
            target / 4
        } else {
            surround_target
        };
    }
    let mut floor_depth: i32 = 0;
    let mut bins: i32 = 0;
    bins = (eBands[(nbEBands - 2) as usize] as i32) << LM;
    floor_depth = (((C * bins) << 3) as opus_val32 * maxDepth) as i32;
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
    if (has_surround_mask == 0 || lfe != 0) && constrained_vbr != 0 {
        target = base_target + (0.67f32 * (target - base_target) as f32) as i32;
    }
    if has_surround_mask == 0 && tf_estimate < 0.2f32 {
        let mut amount: opus_val16 = 0.;
        let mut tvbr_factor: opus_val16 = 0.;
        amount = 0.0000031f32
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
        tvbr_factor = temporal_vbr * amount;
        target += (tvbr_factor * target as f32) as i32;
    }
    target = if 2 * base_target < target {
        2 * base_target
    } else {
        target
    };
    target
}
pub fn celt_encode_with_ec<'b>(
    st: &mut OpusCustomEncoder,
    pcm: &[opus_val16],
    mut frame_size: i32,
    compressed: &'b mut [u8],
    mut nbCompressedBytes: i32,
    mut enc: Option<&mut ec_enc<'b>>,
    #[cfg(feature = "qext")] qext_payload: Option<&mut [u8]>,
    #[cfg(feature = "qext")] qext_bytes: i32,
) -> i32 {
    let mut i: i32 = 0;
    let mut c: i32 = 0;
    let mut N: i32 = 0;
    let mut bits: i32 = 0;
    let mut _enc: ec_enc = ec_enc {
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

    let mut shortBlocks: i32 = 0;
    let mut isTransient: i32 = 0;
    let CC: i32 = st.channels;
    let C: i32 = st.stream_channels;
    let mut LM: i32 = 0;
    let mut M: i32 = 0;
    let mut tf_select: i32 = 0;
    let mut nbFilledBytes: i32 = 0;
    let mut nbAvailableBytes: i32 = 0;
    let mut start: i32 = 0;
    let mut end: i32 = 0;
    let mut effEnd: i32 = 0;
    let mut codedBands: i32 = 0;
    let mut alloc_trim: i32 = 0;
    let mut pitch_index: i32 = COMBFILTER_MINPERIOD;
    let mut gain1: opus_val16 = 0 as opus_val16;
    let mut dual_stereo: i32 = 0;
    let mut effectiveBytes: i32 = 0;
    let mut dynalloc_logp: i32 = 0;
    let mut vbr_rate: i32 = 0;
    let mut total_bits: i32 = 0;
    let mut total_boost: i32 = 0;
    let mut balance: i32 = 0;
    let mut tell: i32 = 0;
    let mut tell0_frac: i32 = 0;
    let mut prefilter_tapset: i32 = 0;
    let mut pf_on: i32 = 0;
    let mut anti_collapse_rsv: i32 = 0;
    let mut anti_collapse_on: i32 = 0;
    let mut silence: i32 = 0;
    let mut tf_chan: i32 = 0;
    let mut tf_estimate: opus_val16 = 0.;
    let mut tone_freq: opus_val16 = -1.0;
    let mut toneishness: opus_val32 = 0.0;
    let mut pitch_change: i32 = 0;
    let mut tot_boost: i32 = 0;
    let mut sample_max: opus_val32 = 0.;
    let mut maxDepth: opus_val16 = 0.;

    let mut nbEBands: i32 = 0;
    let mut overlap: i32 = 0;

    let mut secondMdct: i32 = 0;
    let mut signalBandwidth: i32 = 0;
    let mut transient_got_disabled: i32 = 0;
    let mut surround_masking: opus_val16 = 0 as opus_val16;
    let mut temporal_vbr: opus_val16 = 0 as opus_val16;
    let mut surround_trim: opus_val16 = 0 as opus_val16;
    let mut equiv_rate: i32 = 0;
    let mut hybrid: i32 = 0;
    let mut weak_transient: i32 = 0;
    let mut enable_tf_analysis: i32 = 0;
    #[cfg(feature = "qext")]
    let qext_scale = st.qext_scale;
    #[cfg(not(feature = "qext"))]
    let qext_scale = 1;
    let max_period = COMBFILTER_MAXPERIOD * qext_scale;
    // Max C * nbEBands: 2 * (21 + 14) = 70; use 80 for headroom.
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
    let mut qext_bandE = [0.0f32; 2 * crate::celt::modes::data_96000::NB_QEXT_BANDS];
    #[cfg(feature = "qext")]
    let mut qext_bandLogE = [0.0f32; 2 * crate::celt::modes::data_96000::NB_QEXT_BANDS];
    #[cfg(feature = "qext")]
    let mut qext_error = [0.0f32; 2 * crate::celt::modes::data_96000::NB_QEXT_BANDS];
    let mode: &'static OpusCustomMode = st.mode;
    nbEBands = mode.nbEBands as i32;
    overlap = mode.overlap as i32;
    let eBands: &[i16] = mode.eBands;
    start = st.start;
    end = st.end;
    hybrid = (start != 0) as i32;
    tf_estimate = 0 as opus_val16;
    if nbCompressedBytes < 2 || pcm.is_empty() {
        return OPUS_BAD_ARG;
    }
    frame_size *= st.upsample;
    LM = 0;
    while LM <= mode.maxLM {
        if mode.shortMdctSize << LM == frame_size {
            break;
        }
        LM += 1;
    }
    if LM > mode.maxLM {
        return OPUS_BAD_ARG;
    }
    M = (1) << LM;
    N = M * mode.shortMdctSize;
    if let Some(enc) = enc.as_mut() {
        tell0_frac = ec_tell_frac(enc) as i32;
        tell = ec_tell(enc);
        nbFilledBytes = (tell + 4) >> 3;
    } else {
        tell = 1;
        tell0_frac = tell;
        nbFilledBytes = 0;
    }
    assert!(st.signalling == 0);
    nbCompressedBytes = if nbCompressedBytes < 1275 {
        nbCompressedBytes
    } else {
        1275
    };
    nbAvailableBytes = nbCompressedBytes - nbFilledBytes;
    if st.vbr != 0 && st.bitrate != OPUS_BITRATE_MAX {
        vbr_rate = (st.bitrate * 6 / (6 * mode.Fs / frame_size)) << BITRES;
        effectiveBytes = vbr_rate >> (3 + BITRES);
    } else {
        let mut tmp: i32 = 0;
        vbr_rate = 0;
        tmp = st.bitrate * frame_size;
        if tell > 1 {
            tmp += tell * mode.Fs;
        }
        if st.bitrate != OPUS_BITRATE_MAX {
            nbCompressedBytes = if 2
                > (if nbCompressedBytes
                    < (tmp + 4 * mode.Fs) / (8 * mode.Fs) - (st.signalling != 0) as i32
                {
                    nbCompressedBytes
                } else {
                    (tmp + 4 * mode.Fs) / (8 * mode.Fs) - (st.signalling != 0) as i32
                }) {
                2
            } else if nbCompressedBytes
                < (tmp + 4 * mode.Fs) / (8 * mode.Fs) - (st.signalling != 0) as i32
            {
                nbCompressedBytes
            } else {
                (tmp + 4 * mode.Fs) / (8 * mode.Fs) - (st.signalling != 0) as i32
            };
            if let Some(enc) = enc.as_mut() {
                ec_enc_shrink(enc, nbCompressedBytes as u32);
            }
        }
        effectiveBytes = nbCompressedBytes - nbFilledBytes;
    }
    equiv_rate = ((nbCompressedBytes * 8 * 50) << (3 - LM)) - (40 * C + 20) * ((400 >> LM) - 50);
    if st.bitrate != OPUS_BITRATE_MAX {
        equiv_rate = if equiv_rate < st.bitrate - (40 * C + 20) * ((400 >> LM) - 50) {
            equiv_rate
        } else {
            st.bitrate - (40 * C + 20) * ((400 >> LM) - 50)
        };
    }
    let enc = if let Some(enc) = enc {
        enc
    } else {
        _enc = ec_enc_init(&mut compressed[..nbCompressedBytes as usize]);
        &mut _enc
    };
    if vbr_rate > 0 && st.constrained_vbr != 0 {
        let mut vbr_bound: i32 = 0;
        let mut max_allowed: i32 = 0;
        vbr_bound = vbr_rate;
        max_allowed = if (if (if tell == 1 { 2 } else { 0 })
            > (vbr_rate + vbr_bound - st.vbr_reservoir) >> (3 + 3)
        {
            if tell == 1 {
                2
            } else {
                0
            }
        } else {
            (vbr_rate + vbr_bound - st.vbr_reservoir) >> (3 + 3)
        }) < nbAvailableBytes
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
            nbAvailableBytes
        };
        if max_allowed < nbAvailableBytes {
            nbCompressedBytes = nbFilledBytes + max_allowed;
            nbAvailableBytes = max_allowed;
            ec_enc_shrink(enc, nbCompressedBytes as u32);
        }
    }
    total_bits = nbCompressedBytes * 8;
    effEnd = end;
    if effEnd > mode.effEBands {
        effEnd = mode.effEBands;
    }
    // CC * (N + overlap) max: 2 * (1920 + 240) = 4320.
    const MAX_IN: usize = 4320;
    debug_assert!(((CC * (N + overlap)) as usize) <= MAX_IN);
    let mut in_0 = [0.0f32; MAX_IN];
    let main_len = (C * (N - overlap) / st.upsample) as usize;
    let overlap_len = (C * overlap / st.upsample) as usize;
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
    silence = (sample_max <= 1 as opus_val16 / ((1) << st.lsb_depth) as f32) as i32;
    if tell == 1 {
        ec_enc_bit_logp(enc, silence, 15);
    } else {
        silence = 0;
    }
    if silence != 0 {
        if vbr_rate > 0 {
            nbCompressedBytes = if nbCompressedBytes < nbFilledBytes + 2 {
                nbCompressedBytes
            } else {
                nbFilledBytes + 2
            };
            effectiveBytes = nbCompressedBytes;
            total_bits = nbCompressedBytes * 8;
            nbAvailableBytes = 2;
            ec_enc_shrink(enc, nbCompressedBytes as u32);
        }
        tell = nbCompressedBytes * 8;
        enc.nbits_total += tell - ec_tell(enc);
    }
    c = 0;
    loop {
        let mut need_clip: i32 = 0;
        need_clip = (st.clip != 0 && sample_max > 65536.0f32) as i32;
        celt_preemphasis(
            &pcm[c as usize..],
            &mut in_0[(c * (N + overlap) + overlap) as usize..],
            N,
            CC,
            st.upsample,
            &mode.preemph,
            &mut st.preemph_memE[c as usize],
            need_clip,
        );
        // Copy overlap from prefilter_mem into in_0 (must be before tone_detect/transient_analysis)
        let in_dst = (c * (N + overlap)) as usize;
        let pfm_src = ((c + 1) * max_period - overlap) as usize;
        in_0[in_dst..in_dst + overlap as usize]
            .copy_from_slice(&st.prefilter_mem[pfm_src..pfm_src + overlap as usize]);
        c += 1;
        if c >= CC {
            break;
        }
    }
    // Tone detection — must be before transient_analysis and run_prefilter.
    tone_freq = tone_detect(&in_0, CC, N + overlap, &mut toneishness, mode.Fs);
    isTransient = 0;
    shortBlocks = 0;
    if st.complexity >= 1 && st.lfe == 0 {
        let allow_weak_transients: i32 =
            (hybrid != 0 && effectiveBytes < 15 && st.silk_info.signalType != 2) as i32;
        isTransient = transient_analysis(
            &in_0,
            N + overlap,
            CC,
            &mut tf_estimate,
            &mut tf_chan,
            allow_weak_transients,
            &mut weak_transient,
            tone_freq,
            toneishness,
        );
    }
    toneishness = toneishness.min(1.0 - tf_estimate);
    let mut enabled: i32 = 0;
    let mut qg: i32 = 0;
    enabled = ((st.lfe != 0 && nbAvailableBytes > 3 || nbAvailableBytes > 12 * C)
        && hybrid == 0
        && silence == 0
        && tell + 16 <= total_bits
        && st.disable_pf == 0) as i32;
    prefilter_tapset = st.tapset_decision;
    {
        let analysis = st.analysis;
        pf_on = run_prefilter(
            &mut *st,
            &mut in_0,
            CC,
            N,
            prefilter_tapset,
            &mut pitch_index,
            &mut gain1,
            &mut qg,
            enabled,
            tf_estimate,
            nbAvailableBytes,
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
        let mut octave: i32 = 0;
        ec_enc_bit_logp(enc, 1, 1);
        pitch_index += 1;
        octave = EC_CLZ0 - (pitch_index as u32).leading_zeros() as i32 - 5;
        ec_enc_uint(enc, octave as u32, 6);
        ec_enc_bits(
            enc,
            (pitch_index - ((16) << octave)) as u32,
            (4 + octave) as u32,
        );
        pitch_index -= 1;
        ec_enc_bits(enc, qg as u32, 3);
        ec_enc_icdf(enc, prefilter_tapset, &tapset_icdf, 2);
    }
    if LM > 0 && ec_tell(enc) + 3 <= total_bits {
        if isTransient != 0 {
            shortBlocks = M;
        }
    } else {
        isTransient = 0;
        transient_got_disabled = 1;
    }
    // Allocate N + M - 1 elements so that strided mdct_forward calls
    // can form slices freq[b..b + N*B] for b in 0..B without going
    // out of bounds. The extra elements are never read (stride skips them).
    // CC*N + M - 1 max: 2*1920 + 7 = 3847.
    const MAX_FREQ: usize = 3848;
    debug_assert!(((CC * N + M - 1) as usize) <= MAX_FREQ);
    let mut freq = [0.0f32; MAX_FREQ];
    let mut bandE = [0.0f32; MAX_C_BANDS];
    let mut bandLogE = [0.0f32; MAX_C_BANDS];
    secondMdct = (shortBlocks != 0 && st.complexity >= 8) as i32;
    let mut bandLogE2 = [0.0f32; MAX_C_BANDS];
    if secondMdct != 0 {
        compute_mdcts(mode, 0, &mut in_0, &mut freq, C, CC, LM, st.upsample);
        compute_band_energies(mode, &freq, &mut bandE, effEnd, C, LM, st.arch);
        amp2Log2(mode, effEnd, end, &bandE, &mut bandLogE2, C);
        c = 0;
        while c < C {
            i = 0;
            while i < end {
                bandLogE2[(nbEBands * c + i) as usize] += 0.5f32 * LM as f32;
                i += 1;
            }
            c += 1;
        }
    }
    compute_mdcts(
        mode,
        shortBlocks,
        &mut in_0,
        &mut freq,
        C,
        CC,
        LM,
        st.upsample,
    );
    assert!(!freq[0].is_nan() && (C == 1 || !freq[N as usize].is_nan()));
    if CC == 2 && C == 1 {
        tf_chan = 0;
    }
    compute_band_energies(mode, &freq, &mut bandE, effEnd, C, LM, st.arch);
    if st.lfe != 0 {
        i = 2;
        while i < end {
            bandE[i as usize] = if bandE[i as usize] < 1e-4f32 * bandE[0_usize] {
                bandE[i as usize]
            } else {
                1e-4f32 * bandE[0_usize]
            };
            bandE[i as usize] = if bandE[i as usize] > 1e-15f32 {
                bandE[i as usize]
            } else {
                1e-15f32
            };
            i += 1;
        }
    }
    amp2Log2(mode, effEnd, end, &bandE, &mut bandLogE, C);
    let mut surround_dynalloc = [0.0f32; MAX_C_BANDS];
    surround_dynalloc[..end as usize].fill(0.0);
    let energy_mask: Option<&[opus_val16]> = if st.energy_mask_len == 0 {
        None
    } else {
        Some(&st.energy_mask[..(CC * nbEBands) as usize])
    };
    if let Some(energy_mask) = energy_mask {
        if hybrid == 0 && st.lfe == 0 {
            let mut mask_end: i32 = 0;
            let mut midband: i32 = 0;
            let mut count_dynalloc: i32 = 0;
            let mut mask_avg: opus_val32 = 0 as opus_val32;
            let mut diff: opus_val32 = 0 as opus_val32;
            let mut count: i32 = 0;
            mask_end = if 2 > st.lastCodedBands {
                2
            } else {
                st.lastCodedBands
            };
            c = 0;
            while c < C {
                i = 0;
                while i < mask_end {
                    let mut mask: opus_val16 = 0.;
                    mask = if (if energy_mask[(nbEBands * c + i) as usize] < 0.25f32 {
                        energy_mask[(nbEBands * c + i) as usize]
                    } else {
                        0.25f32
                    }) > -2.0f32
                    {
                        if energy_mask[(nbEBands * c + i) as usize] < 0.25f32 {
                            energy_mask[(nbEBands * c + i) as usize]
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
                        * (eBands[(i + 1) as usize] as i32 - eBands[i as usize] as i32)
                            as opus_val32;
                    count += eBands[(i + 1) as usize] as i32 - eBands[i as usize] as i32;
                    diff += mask * (1 + 2 * i - mask_end) as opus_val32;
                    i += 1;
                }
                c += 1;
            }
            assert!(count > 0);
            mask_avg /= count as opus_val16;
            mask_avg += 0.2f32;
            diff = diff * 6_f32 / (C * (mask_end - 1) * (mask_end + 1) * mask_end) as f32;
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
            while (eBands[(midband + 1) as usize] as i32) < eBands[mask_end as usize] as i32 / 2 {
                midband += 1;
            }
            count_dynalloc = 0;
            i = 0;
            while i < mask_end {
                let mut lin: opus_val32 = 0.;
                let mut unmask: opus_val16 = 0.;
                lin = mask_avg + diff * (i - midband) as f32;
                if C == 2 {
                    unmask = if energy_mask[i as usize] > energy_mask[(nbEBands + i) as usize] {
                        energy_mask[i as usize]
                    } else {
                        energy_mask[(nbEBands + i) as usize]
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
                    mask_avg = 0 as opus_val32;
                    diff = 0 as opus_val32;
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
        let mut follow: opus_val16 = -10.0f32;
        let mut frame_avg: opus_val32 = 0 as opus_val32;
        let offset: opus_val16 = if shortBlocks != 0 {
            0.5f32 * LM as f32
        } else {
            0 as f32
        };
        i = start;
        while i < end {
            follow = if follow - 1.0f32 > bandLogE[i as usize] - offset {
                follow - 1.0f32
            } else {
                bandLogE[i as usize] - offset
            };
            if C == 2 {
                follow = if follow > bandLogE[(i + nbEBands) as usize] - offset {
                    follow
                } else {
                    bandLogE[(i + nbEBands) as usize] - offset
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
    if secondMdct == 0 {
        let len = (C * nbEBands) as usize;
        bandLogE2[..len].copy_from_slice(&bandLogE[..len]);
    }
    if LM > 0
        && ec_tell(enc) + 3 <= total_bits
        && isTransient == 0
        && st.complexity >= 5
        && st.lfe == 0
        && hybrid == 0
        && patch_transient_decision(
            &bandLogE,
            &st.oldBandE[..(C * nbEBands) as usize],
            nbEBands,
            start,
            end,
            C,
        ) != 0
    {
        isTransient = 1;
        shortBlocks = M;
        compute_mdcts(
            mode,
            shortBlocks,
            &mut in_0,
            &mut freq,
            C,
            CC,
            LM,
            st.upsample,
        );
        compute_band_energies(mode, &freq, &mut bandE, effEnd, C, LM, st.arch);
        amp2Log2(mode, effEnd, end, &bandE, &mut bandLogE, C);
        c = 0;
        while c < C {
            i = 0;
            while i < end {
                bandLogE2[(nbEBands * c + i) as usize] += 0.5f32 * LM as f32;
                i += 1;
            }
            c += 1;
        }
        tf_estimate = 0.2f32;
    }
    if LM > 0 && ec_tell(enc) + 3 <= total_bits {
        ec_enc_bit_logp(enc, isTransient, 3);
    }
    // C*N max: 2*1920 = 3840.
    const MAX_X: usize = 3840;
    debug_assert!(((C * N) as usize) <= MAX_X);
    let mut X = [0.0f32; MAX_X];
    normalise_bands(mode, &freq, &mut X, &bandE, effEnd, C, M);
    enable_tf_analysis = (effectiveBytes >= 15 * C
        && hybrid == 0
        && st.complexity >= 2
        && st.lfe == 0
        && toneishness < 0.98) as i32;
    const MAX_BANDS_ENC: usize = 40;
    debug_assert!((nbEBands as usize) <= MAX_BANDS_ENC);
    let mut offsets = [0i32; MAX_BANDS_ENC];
    let mut importance = [0i32; MAX_BANDS_ENC];
    let mut spread_weight = [0i32; MAX_BANDS_ENC];
    maxDepth = dynalloc_analysis(
        &bandLogE,
        &bandLogE2,
        &st.oldBandE,
        nbEBands,
        start,
        end,
        C,
        &mut offsets,
        st.lsb_depth,
        mode.logN,
        isTransient,
        st.vbr,
        st.constrained_vbr,
        mode.eBands,
        LM,
        effectiveBytes,
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
        let mut lambda: i32 = 0;
        lambda = if 80 > 20480 / effectiveBytes + 2 {
            80
        } else {
            20480 / effectiveBytes + 2
        };
        tf_select = tf_analysis(
            mode,
            effEnd,
            isTransient,
            &mut tf_res,
            lambda,
            &X,
            N,
            LM,
            tf_estimate,
            tf_chan,
            &importance,
        );
        i = effEnd;
        while i < end {
            tf_res[i as usize] = tf_res[(effEnd - 1) as usize];
            i += 1;
        }
    } else if hybrid != 0 && weak_transient != 0 {
        i = 0;
        while i < end {
            tf_res[i as usize] = 1;
            i += 1;
        }
        tf_select = 0;
    } else if hybrid != 0 && effectiveBytes < 15 && st.silk_info.signalType != 2 {
        i = 0;
        while i < end {
            tf_res[i as usize] = 0;
            i += 1;
        }
        tf_select = isTransient;
    } else {
        i = 0;
        while i < end {
            tf_res[i as usize] = isTransient;
            i += 1;
        }
        tf_select = 0;
    }
    let mut error = [0.0f32; MAX_C_BANDS];
    c = 0;
    loop {
        i = start;
        while i < end {
            let idx = (i + c * nbEBands) as usize;
            if (bandLogE[idx] - st.oldBandE[idx]).abs() < 2.0f32 {
                bandLogE[idx] -= st.energyError[idx] * 0.25f32;
            }
            i += 1;
        }
        c += 1;
        if c >= C {
            break;
        }
    }
    quant_coarse_energy(
        mode,
        start,
        end,
        effEnd,
        &bandLogE,
        &mut st.oldBandE[..(C * nbEBands) as usize],
        total_bits as u32,
        &mut error,
        enc,
        C,
        LM,
        nbAvailableBytes,
        st.force_intra,
        &mut st.delayedIntra,
        (st.complexity >= 4) as i32,
        st.loss_rate,
        st.lfe,
    );
    tf_encode(start, end, isTransient, &mut tf_res, LM, tf_select, enc);
    if ec_tell(enc) + 4 <= total_bits {
        if st.lfe != 0 {
            st.tapset_decision = 0;
            st.spread_decision = SPREAD_NORMAL;
        } else if hybrid != 0 {
            if st.complexity == 0 {
                st.spread_decision = SPREAD_NONE;
            } else if isTransient != 0 {
                st.spread_decision = SPREAD_NORMAL;
            } else {
                st.spread_decision = SPREAD_AGGRESSIVE;
            }
        } else if shortBlocks != 0 || st.complexity < 3 || nbAvailableBytes < 10 * C {
            if st.complexity == 0 {
                st.spread_decision = SPREAD_NONE;
            } else {
                st.spread_decision = SPREAD_NORMAL;
            }
        } else {
            st.spread_decision = spreading_decision(
                mode,
                &X,
                &mut st.tonal_average,
                st.spread_decision,
                &mut st.hf_average,
                &mut st.tapset_decision,
                (pf_on != 0 && shortBlocks == 0) as i32,
                effEnd,
                C,
                M,
                &spread_weight,
            );
        }
        ec_enc_icdf(enc, st.spread_decision, &spread_icdf, 5);
    } else {
        st.spread_decision = SPREAD_NORMAL;
    }
    if st.lfe != 0 {
        offsets[0_usize] = if (8) < effectiveBytes / 3 {
            8
        } else {
            effectiveBytes / 3
        };
    }
    let mut cap = [0i32; MAX_BANDS_ENC];
    init_caps(mode, &mut cap, LM, C);
    dynalloc_logp = 6;
    total_bits <<= BITRES;
    total_boost = 0;
    tell = ec_tell_frac(enc) as i32;
    i = start;
    while i < end {
        let mut width: i32 = 0;
        let mut quanta: i32 = 0;
        let mut dynalloc_loop_logp: i32 = 0;
        let mut boost: i32 = 0;
        let mut j: i32 = 0;
        width = (C * (eBands[(i + 1) as usize] as i32 - eBands[i as usize] as i32)) << LM;
        quanta = if (width << 3) < (if (6) << 3 > width { (6) << 3 } else { width }) {
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
            let mut flag: i32 = 0;
            flag = (j < offsets[i as usize]) as i32;
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
    if C == 2 {
        static intensity_thresholds: [opus_val16; 21] = [
            1 as opus_val16,
            2 as opus_val16,
            3 as opus_val16,
            4 as opus_val16,
            5 as opus_val16,
            6 as opus_val16,
            7 as opus_val16,
            8 as opus_val16,
            16 as opus_val16,
            24 as opus_val16,
            36 as opus_val16,
            44 as opus_val16,
            50 as opus_val16,
            56 as opus_val16,
            62 as opus_val16,
            67 as opus_val16,
            72 as opus_val16,
            79 as opus_val16,
            88 as opus_val16,
            106 as opus_val16,
            134 as opus_val16,
        ];
        static intensity_histeresis: [opus_val16; 21] = [
            1 as opus_val16,
            1 as opus_val16,
            1 as opus_val16,
            1 as opus_val16,
            1 as opus_val16,
            1 as opus_val16,
            1 as opus_val16,
            2 as opus_val16,
            2 as opus_val16,
            2 as opus_val16,
            2 as opus_val16,
            2 as opus_val16,
            2 as opus_val16,
            2 as opus_val16,
            3 as opus_val16,
            3 as opus_val16,
            4 as opus_val16,
            5 as opus_val16,
            6 as opus_val16,
            8 as opus_val16,
            8 as opus_val16,
        ];
        if LM != 0 {
            dual_stereo = stereo_analysis(mode, &X, LM, N);
        }
        st.intensity = hysteresis_decision(
            (equiv_rate / 1000) as opus_val16,
            &intensity_thresholds,
            &intensity_histeresis,
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
            st.stereo_saving = 0 as opus_val16;
            alloc_trim = 5;
        } else {
            alloc_trim = alloc_trim_analysis(
                mode,
                &X,
                &bandLogE,
                end,
                LM,
                C,
                N,
                &st.analysis,
                &mut st.stereo_saving,
                tf_estimate,
                st.intensity,
                surround_trim,
                equiv_rate,
                st.arch,
            );
        }
        ec_enc_icdf(enc, alloc_trim, &trim_icdf, 7);
        tell = ec_tell_frac(enc) as i32;
    }
    if vbr_rate > 0 {
        let mut alpha: opus_val16 = 0.;
        let mut delta: i32 = 0;
        let mut target: i32 = 0;
        let mut base_target: i32 = 0;
        let mut min_allowed: i32 = 0;
        let lm_diff: i32 = mode.maxLM - LM;
        nbCompressedBytes = if nbCompressedBytes < 1275 >> (3 - LM) {
            nbCompressedBytes
        } else {
            1275 >> (3 - LM)
        };
        if hybrid == 0 {
            base_target = vbr_rate - ((40 * C + 20) << BITRES);
        } else {
            base_target = if 0 > vbr_rate - ((9 * C + 4) << 3) {
                0
            } else {
                vbr_rate - ((9 * C + 4) << 3)
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
                LM,
                equiv_rate,
                st.lastCodedBands,
                C,
                st.intensity,
                st.constrained_vbr,
                st.stereo_saving,
                tot_boost,
                tf_estimate,
                pitch_change,
                maxDepth,
                st.lfe,
                energy_mask.is_some() as i32,
                surround_masking,
                temporal_vbr,
            );
        } else {
            target = base_target;
            if st.silk_info.offset < 100 {
                target += (12) << BITRES >> (3 - LM);
            }
            if st.silk_info.offset > 100 {
                target -= (18) << BITRES >> (3 - LM);
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
        nbAvailableBytes = (target + ((1) << (BITRES + 2))) >> (BITRES + 3);
        nbAvailableBytes = if min_allowed > nbAvailableBytes {
            min_allowed
        } else {
            nbAvailableBytes
        };
        nbAvailableBytes = if nbCompressedBytes < nbAvailableBytes {
            nbCompressedBytes
        } else {
            nbAvailableBytes
        };
        delta = target - vbr_rate;
        target = nbAvailableBytes << (BITRES + 3);
        if silence != 0 {
            nbAvailableBytes = 2;
            target = (2 * 8) << BITRES;
            delta = 0;
        }
        if st.vbr_count < 970 {
            st.vbr_count += 1;
            alpha = 1.0f32 / (st.vbr_count + 20) as f32;
        } else {
            alpha = 0.001f32;
        }
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
            nbAvailableBytes += if silence != 0 { 0 } else { adjust };
            st.vbr_reservoir = 0;
        }
        nbCompressedBytes = if nbCompressedBytes < nbAvailableBytes {
            nbCompressedBytes
        } else {
            nbAvailableBytes
        };
        ec_enc_shrink(enc, nbCompressedBytes as u32);
    }
    let mut fine_quant = [0i32; MAX_BANDS_ENC];
    let mut pulses = [0i32; MAX_BANDS_ENC];
    let mut fine_priority = [0i32; MAX_BANDS_ENC];
    bits = (((nbCompressedBytes * 8) << BITRES) as u32)
        .wrapping_sub(ec_tell_frac(enc))
        .wrapping_sub(1) as i32;
    anti_collapse_rsv = if isTransient != 0 && LM >= 2 && bits >= (LM + 2) << BITRES {
        (1) << BITRES
    } else {
        0
    };
    bits -= anti_collapse_rsv;
    signalBandwidth = end - 1;
    if st.analysis.valid != 0 {
        let mut min_bandwidth: i32 = 0;
        if equiv_rate < 32000 * C {
            min_bandwidth = 13;
        } else if equiv_rate < 48000 * C {
            min_bandwidth = 16;
        } else if equiv_rate < 60000 * C {
            min_bandwidth = 18;
        } else if equiv_rate < 80000 * C {
            min_bandwidth = 19;
        } else {
            min_bandwidth = 20;
        }
        signalBandwidth = if st.analysis.bandwidth > min_bandwidth {
            st.analysis.bandwidth
        } else {
            min_bandwidth
        };
    }
    if st.lfe != 0 {
        signalBandwidth = 1;
    }
    codedBands = clt_compute_allocation(
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
        C,
        LM,
        enc,
        1,
        st.lastCodedBands,
        signalBandwidth,
    );
    if st.lastCodedBands != 0 {
        st.lastCodedBands = if (st.lastCodedBands + 1)
            < (if st.lastCodedBands - 1 > codedBands {
                st.lastCodedBands - 1
            } else {
                codedBands
            }) {
            st.lastCodedBands + 1
        } else if st.lastCodedBands - 1 > codedBands {
            st.lastCodedBands - 1
        } else {
            codedBands
        };
    } else {
        st.lastCodedBands = codedBands;
    }
    quant_fine_energy(
        mode,
        start,
        end,
        &mut st.oldBandE[..(C * nbEBands) as usize],
        &mut error,
        None,
        &fine_quant,
        enc,
        C,
    );
    // QEXT: Compute QEXT mode and band energies after first-pass fine energy
    #[cfg(feature = "qext")]
    {
        use crate::celt::modes::data_96000::NB_QEXT_BANDS;

        if qext_bytes > 0
            && end == nbEBands
            && (mode.Fs == 48000 || mode.Fs == 96000)
            && (mode.shortMdctSize == 120 * qext_scale || mode.shortMdctSize == 90 * qext_scale)
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
            compute_band_energies(qm, &freq, &mut qext_bandE, qext_end, C, LM, st.arch);
            normalise_bands(qm, &freq, &mut X, &qext_bandE, qext_end, C, M);
            amp2Log2(qm, qext_end, qext_end, &qext_bandE, &mut qext_bandLogE, C);

            // Encode stereo params for QEXT bands
            if C == 2 {
                qext_intensity = qext_end;
                qext_dual_stereo = dual_stereo;
                ec_enc_uint(&mut ext_enc, qext_intensity as u32, (qext_end + 1) as u32);
                if qext_intensity != 0 {
                    ec_enc_bit_logp(&mut ext_enc, qext_dual_stereo, 1);
                }
            }

            // Coarse quantization of QEXT band energies
            let mut qext_delayedIntra: opus_val32 = 0.0;
            quant_coarse_energy(
                qm,
                0,
                qext_end,
                qext_end,
                &qext_bandLogE,
                &mut st.qext_oldBandE,
                (qext_bytes * 8) as u32,
                &mut qext_error,
                &mut ext_enc,
                C,
                LM,
                qext_bytes,
                st.force_intra,
                &mut qext_delayedIntra,
                (st.complexity >= 4) as i32,
                st.loss_rate,
                st.lfe,
            );
        }
    }

    // QEXT: Compute extra allocation and second-pass fine energy
    st.energyError[..(nbEBands * CC) as usize].fill(0.0);
    #[cfg(feature = "qext")]
    let mut extra_pulses = {
        use crate::celt::modes::data_96000::NB_QEXT_BANDS;
        // nbEBands + NB_QEXT_BANDS max: 21 + 14 = 35.
        [0i32; 40]
    };
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
            Some(&bandLogE),
            if qext_mode.is_some() {
                Some(&qext_bandLogE)
            } else {
                None
            },
            qext_bits,
            &mut extra_pulses,
            &mut extra_quant,
            C,
            LM,
            &mut ext_enc,
            1, // encode=1
            tone_freq,
            toneishness,
        );
        error_bak[..(C * nbEBands) as usize].copy_from_slice(&error[..(C * nbEBands) as usize]);
        if qext_bytes > 0 {
            quant_fine_energy(
                mode,
                start,
                end,
                &mut st.oldBandE[..(C * nbEBands) as usize],
                &mut error,
                Some(&fine_quant),
                &extra_quant[..nbEBands as usize],
                &mut ext_enc,
                C,
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

    if C == 2 {
        let (x_part, y_part) = X.split_at_mut(N as usize);
        quant_all_bands(
            1,
            mode,
            start,
            end,
            x_part,
            Some(y_part),
            &mut collapse_masks,
            &bandE,
            &mut pulses,
            shortBlocks,
            st.spread_decision,
            dual_stereo,
            st.intensity,
            &mut tf_res,
            nbCompressedBytes * ((8) << BITRES) - anti_collapse_rsv,
            balance,
            enc,
            LM,
            codedBands,
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
            &mut X,
            None,
            &mut collapse_masks,
            &bandE,
            &mut pulses,
            shortBlocks,
            st.spread_decision,
            dual_stereo,
            st.intensity,
            &mut tf_res,
            nbCompressedBytes * ((8) << BITRES) - anti_collapse_rsv,
            balance,
            enc,
            LM,
            codedBands,
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
                ext_balance -= extra_pulses[nbEBands as usize + j as usize]
                    + C * (extra_quant[nbEBands as usize + 1] << BITRES);
            }

            // Fine energy for QEXT bands
            quant_fine_energy(
                qm,
                0,
                qext_end,
                &mut st.qext_oldBandE[..(C * NB_QEXT_BANDS as i32) as usize],
                &mut qext_error,
                None,
                &extra_quant[nbEBands as usize..],
                &mut ext_enc,
                C,
            );

            // Dummy encoder for the nested ext_enc arg of quant_all_bands
            let mut dummy_buf = [0u8; 4];
            let mut dummy_enc = crate::celt::entcode::ec_ctx {
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

            if C == 2 {
                let (x_part, y_part) = X.split_at_mut(N as usize);
                quant_all_bands(
                    1,
                    qm,
                    0,
                    qext_end,
                    x_part,
                    Some(y_part),
                    &mut qext_collapse_masks,
                    &qext_bandE,
                    &mut extra_pulses[nbEBands as usize..],
                    shortBlocks,
                    st.spread_decision,
                    qext_dual_stereo,
                    qext_intensity,
                    &mut zeros.clone(),
                    qext_bytes * (8 << BITRES),
                    ext_balance,
                    &mut ext_enc,
                    LM,
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
                    &mut X,
                    None,
                    &mut qext_collapse_masks,
                    &qext_bandE,
                    &mut extra_pulses[nbEBands as usize..],
                    shortBlocks,
                    st.spread_decision,
                    qext_dual_stereo,
                    qext_intensity,
                    &mut zeros.clone(),
                    qext_bytes * (8 << BITRES),
                    ext_balance,
                    &mut ext_enc,
                    LM,
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
                &mut st.oldBandE[..(C * nbEBands) as usize],
                &mut error,
                &fine_quant,
                &fine_priority,
                nbCompressedBytes * 8 - ec_tell(enc),
                enc,
                C,
            );
        }
    }
    #[cfg(not(feature = "qext"))]
    {
        quant_energy_finalise(
            mode,
            start,
            end,
            &mut st.oldBandE[..(C * nbEBands) as usize],
            &mut error,
            &fine_quant,
            &fine_priority,
            nbCompressedBytes * 8 - ec_tell(enc),
            enc,
            C,
        );
    }

    c = 0;
    loop {
        i = start;
        while i < end {
            let idx = (i + c * nbEBands) as usize;
            st.energyError[idx] = error[idx].clamp(-0.5f32, 0.5f32);
            i += 1;
        }
        c += 1;
        if c >= C {
            break;
        }
    }

    // QEXT: When qext_bytes > 0, run finalise with error_bak (original error before QEXT fine energy)
    #[cfg(feature = "qext")]
    {
        if qext_bytes > 0 {
            // Pass NULL for oldBandE (don't update), use error_bak
            quant_energy_finalise(
                mode,
                start,
                end,
                &mut [0.0f32; 42][..(C * nbEBands) as usize], // dummy, won't be used meaningfully
                &mut error_bak,
                &fine_quant,
                &fine_priority,
                nbCompressedBytes * 8 - ec_tell(enc),
                enc,
                C,
            );
        }
    }
    if silence != 0 {
        i = 0;
        while i < C * nbEBands {
            st.oldBandE[i as usize] = -28.0f32;
            i += 1;
        }
    }
    st.prefilter_period = pitch_index;
    st.prefilter_gain = gain1;
    st.prefilter_tapset = prefilter_tapset;
    if CC == 2 && C == 1 {
        let nb = nbEBands as usize;
        st.oldBandE.copy_within(..nb, nb);
    }
    if isTransient == 0 {
        let len = (CC * nbEBands) as usize;
        st.oldLogE2[..len].copy_from_slice(&st.oldLogE[..len]);
        st.oldLogE[..len].copy_from_slice(&st.oldBandE[..len]);
    } else {
        i = 0;
        while i < CC * nbEBands {
            let idx = i as usize;
            st.oldLogE[idx] = if st.oldLogE[idx] < st.oldBandE[idx] {
                st.oldLogE[idx]
            } else {
                st.oldBandE[idx]
            };
            i += 1;
        }
    }
    c = 0;
    loop {
        i = 0;
        while i < start {
            let idx = (c * nbEBands + i) as usize;
            st.oldBandE[idx] = 0 as opus_val16;
            st.oldLogE2[idx] = -28.0f32;
            st.oldLogE[idx] = -28.0f32;
            i += 1;
        }
        i = end;
        while i < nbEBands {
            let idx = (c * nbEBands + i) as usize;
            st.oldBandE[idx] = 0 as opus_val16;
            st.oldLogE2[idx] = -28.0f32;
            st.oldLogE[idx] = -28.0f32;
            i += 1;
        }
        c += 1;
        if c >= CC {
            break;
        }
    }
    if isTransient != 0 || transient_got_disabled != 0 {
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
    if ec_get_error(enc) != 0 {
        OPUS_INTERNAL_ERROR
    } else {
        nbCompressedBytes
    }
}
