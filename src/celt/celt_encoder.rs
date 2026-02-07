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

pub mod stddef_h {
    pub const NULL: i32 = 0;
}
pub use self::arch_h::{
    celt_ener, celt_norm, celt_sig, opus_val16, opus_val32, CELT_SIG_SCALE, EPSILON,
};
pub use self::stddef_h::NULL;
use crate::celt::celt::{
    comb_filter, init_caps, resampling_factor, spread_icdf, tapset_icdf, tf_select_table, trim_icdf,
};
use crate::celt::celt::{
    CELT_GET_MODE_REQUEST, CELT_SET_ANALYSIS_REQUEST, CELT_SET_CHANNELS_REQUEST,
    CELT_SET_END_BAND_REQUEST, CELT_SET_PREDICTION_REQUEST, CELT_SET_SIGNALLING_REQUEST,
    CELT_SET_SILK_INFO_REQUEST, CELT_SET_START_BAND_REQUEST, COMBFILTER_MAXPERIOD,
    COMBFILTER_MINPERIOD, OPUS_SET_ENERGY_MASK_REQUEST, OPUS_SET_LFE_REQUEST,
};
use crate::celt::entcode::{ec_get_error, ec_tell, ec_tell_frac, BITRES};
use crate::celt::entenc::{
    ec_enc, ec_enc_bit_logp, ec_enc_bits, ec_enc_done, ec_enc_icdf, ec_enc_init, ec_enc_shrink,
    ec_enc_uint,
};
use crate::celt::mathops::{celt_exp2, celt_log2, celt_maxabs16, celt_sqrt};
use crate::celt::mdct::mdct_forward;
use crate::celt::modes::{opus_custom_mode_create, OpusCustomMode};
use crate::celt::pitch::{celt_inner_prod, pitch_downsample, pitch_search, remove_doubling};
use crate::celt::quant_bands::{
    amp2Log2, eMeans, quant_coarse_energy, quant_energy_finalise, quant_fine_energy,
};
use crate::celt::rate::clt_compute_allocation;
use crate::externs::{memcpy, memset};
use crate::opus_custom_encoder_ctl;
use crate::silk::macros::EC_CLZ0;
use crate::src::analysis::AnalysisInfo;
use crate::src::opus_defines::{
    OPUS_ALLOC_FAIL, OPUS_BAD_ARG, OPUS_BITRATE_MAX, OPUS_GET_FINAL_RANGE_REQUEST,
    OPUS_GET_LSB_DEPTH_REQUEST, OPUS_GET_PHASE_INVERSION_DISABLED_REQUEST, OPUS_INTERNAL_ERROR,
    OPUS_OK, OPUS_RESET_STATE, OPUS_SET_BITRATE_REQUEST, OPUS_SET_COMPLEXITY_REQUEST,
    OPUS_SET_LSB_DEPTH_REQUEST, OPUS_SET_PACKET_LOSS_PERC_REQUEST,
    OPUS_SET_PHASE_INVERSION_DISABLED_REQUEST, OPUS_SET_VBR_CONSTRAINT_REQUEST,
    OPUS_SET_VBR_REQUEST, OPUS_UNIMPLEMENTED,
};
use crate::varargs::VarArgs;

/// Upstream C: celt/celt_encoder.c:OpusCustomEncoder
///
/// The C version uses a flexible array member (`in_mem[1]`) at the end of the struct
/// to store overlap memory, prefilter memory, and band energy arrays in a contiguous
/// allocation. This Rust version uses fixed-size arrays sized for the maximum case
/// (2 channels, overlap=120, nbEBands=21, COMBFILTER_MAXPERIOD=1024).
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
    pub arch: i32,
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
    pub energy_mask: *const opus_val16,
    pub spec_avg: opus_val16,
    /// Overlap memory, size = channels * overlap (max 2*120 = 240)
    pub in_mem: [celt_sig; 2 * 120],
    /// Prefilter memory, size = channels * COMBFILTER_MAXPERIOD (max 2*1024 = 2048)
    pub prefilter_mem: [celt_sig; 2 * COMBFILTER_MAXPERIOD as usize],
    /// Old band energies, size = channels * nbEBands (max 2*21 = 42)
    pub oldBandE: [opus_val16; 2 * 21],
    /// Old log energies, size = channels * nbEBands (max 2*21 = 42)
    pub oldLogE: [opus_val16; 2 * 21],
    /// Old log energies (2 frames ago), size = channels * nbEBands (max 2*21 = 42)
    pub oldLogE2: [opus_val16; 2 * 21],
    /// Energy quantization error, size = channels * nbEBands (max 2*21 = 42)
    pub energyError: [opus_val16; 2 * 21],
}
/// Upstream C: celt/celt_encoder.c:celt_encoder_get_size
pub fn celt_encoder_get_size(_channels: i32) -> i32 {
    ::core::mem::size_of::<OpusCustomEncoder>() as i32
}
/// Upstream C: celt/celt_encoder.c:opus_custom_encoder_init_arch
unsafe fn opus_custom_encoder_init_arch(
    st: *mut OpusCustomEncoder,
    mode: &'static OpusCustomMode,
    channels: i32,
    arch: i32,
) -> i32 {
    if channels < 0 || channels > 2 {
        return OPUS_BAD_ARG;
    }
    if st.is_null() {
        return OPUS_ALLOC_FAIL;
    }
    // Write the entire struct with zeroed arrays and proper defaults.
    // This replaces the C pattern of memset(st, 0, size) + field assignments.
    *st = OpusCustomEncoder {
        mode,
        channels,
        stream_channels: channels,
        force_intra: 0,
        clip: 1,
        disable_pf: 0,
        complexity: 5,
        upsample: 1,
        start: 0,
        end: mode.effEBands,
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
        energy_mask: std::ptr::null(),
        spec_avg: 0.0,
        in_mem: [0.0; 2 * 120],
        prefilter_mem: [0.0; 2 * COMBFILTER_MAXPERIOD as usize],
        oldBandE: [0.0; 2 * 21],
        oldLogE: [0.0; 2 * 21],
        oldLogE2: [0.0; 2 * 21],
        energyError: [0.0; 2 * 21],
    };
    opus_custom_encoder_ctl!(st, OPUS_RESET_STATE);
    return OPUS_OK;
}
/// Upstream C: celt/celt_encoder.c:celt_encoder_init
pub unsafe fn celt_encoder_init(
    st: *mut OpusCustomEncoder,
    sampling_rate: i32,
    channels: i32,
    arch: i32,
) -> i32 {
    let ret = opus_custom_encoder_init_arch(
        st,
        opus_custom_mode_create(48000, 960, None).unwrap(),
        channels,
        arch,
    );
    if ret != OPUS_OK {
        return ret;
    }
    (*st).upsample = resampling_factor(sampling_rate);
    return OPUS_OK;
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
    let vla = len as usize;
    let mut tmp: Vec<opus_val16> = ::std::vec::from_elem(0., vla);
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
            mem0 = mem1 + y - 2 as f32 * x;
            mem1 = x - 0.5f32 * y;
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
            tmp[i as usize] = mem0 + forward_decay * (x2 - mem0);
            mem0 = tmp[i as usize];
            i += 1;
        }
        mem0 = 0 as opus_val32;
        maxE = 0 as opus_val16;
        i = len2 - 1;
        while i >= 0 {
            tmp[i as usize] = mem0 + 0.125f32 * (tmp[i as usize] - mem0);
            mem0 = tmp[i as usize];
            maxE = if maxE > mem0 { maxE } else { mem0 };
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
    if allow_weak_transients != 0 && is_transient != 0 && mask_metric < 600 {
        is_transient = 0;
        *weak_transient = 1;
    }
    tf_max = if 0 as f32 > celt_sqrt((27 * mask_metric) as f32) - 42 as f32 {
        0 as f32
    } else {
        celt_sqrt((27 * mask_metric) as f32) - 42 as f32
    };
    *tf_estimate = (if 0 as f64
        > (0.0069f64 as opus_val32
            * (if (163 as f32) < tf_max {
                163 as f32
            } else {
                tf_max
            })) as f64
            - 0.139f64
    {
        0 as f64
    } else {
        (0.0069f64 as opus_val32
            * (if (163 as f32) < tf_max {
                163 as f32
            } else {
                tf_max
            })) as f64
            - 0.139f64
    })
    // here, a 64-bit sqrt __should__ be used
    .sqrt() as f32;
    return is_transient;
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
            mean_diff = mean_diff
                + (if 0 as f32 > x1 - x2 {
                    0 as f32
                } else {
                    x1 - x2
                });
            i += 1;
        }
        c += 1;
        if !(c < C) {
            break;
        }
    }
    mean_diff = mean_diff / (C * (end - 1 - (if 2 > start { 2 } else { start }))) as opus_val32;
    return (mean_diff > 1.0f32) as i32;
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
                &mut in_0[in_base..in_base + in_len],
                &mut out[out_base..out_base + out_len],
                mode.window,
                overlap as usize,
                shift as usize,
                B as usize,
            );
            b += 1;
        }
        c += 1;
        if !(c < CC) {
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
            if !(c < C) {
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
    return L1;
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
    let vla = len as usize;
    let mut metric: Vec<i32> = ::std::vec::from_elem(0, vla);
    let vla_0 =
        ((m.eBands[len as usize] as i32 - m.eBands[(len - 1) as usize] as i32) << LM) as usize;
    let mut tmp: Vec<celt_norm> = ::std::vec::from_elem(0., vla_0);
    let vla_1 =
        ((m.eBands[len as usize] as i32 - m.eBands[(len - 1) as usize] as i32) << LM) as usize;
    let mut tmp_1: Vec<celt_norm> = ::std::vec::from_elem(0., vla_1);
    let vla_2 = len as usize;
    let mut path0: Vec<i32> = ::std::vec::from_elem(0, vla_2);
    let vla_3 = len as usize;
    let mut path1: Vec<i32> = ::std::vec::from_elem(0, vla_3);
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
                - 2 * tf_select_table[LM as usize][(4 * isTransient + 2 * sel + 0) as usize]
                    as i32)
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
                        - 2 * tf_select_table[LM as usize][(4 * isTransient + 2 * sel + 0) as usize]
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
    if selcost[1 as usize] < selcost[0 as usize] && isTransient != 0 {
        tf_select = 1;
    }
    cost0 = importance[0]
        * (metric[0]
            - 2 * tf_select_table[LM as usize][(4 * isTransient + 2 * tf_select + 0) as usize]
                as i32)
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
                    - 2 * tf_select_table[LM as usize]
                        [(4 * isTransient + 2 * tf_select + 0) as usize]
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
    return tf_select;
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
    budget = (budget as u32).wrapping_sub(tf_select_rsv as u32) as u32 as u32;
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
        && tf_select_table[LM as usize][(4 * isTransient + 0 + tf_changed) as usize] as i32
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
    _arch: i32,
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
        let frac: i32 = equiv_rate - 64000 >> 10;
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
            );
            sum = sum + partial;
            i += 1;
        }
        sum = 1.0f32 / 8 as f32 * sum;
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
        if !(c < C) {
            break;
        }
    }
    diff /= (C * (end - 1)) as f32;
    trim -= if -2.0f32
        > (if 2.0f32 < (diff + 1.0f32) / 6 as f32 {
            2.0f32
        } else {
            (diff + 1.0f32) / 6 as f32
        }) {
        -2.0f32
    } else if 2.0f32 < (diff + 1.0f32) / 6 as f32 {
        2.0f32
    } else {
        (diff + 1.0f32) / 6 as f32
    };
    trim -= surround_trim;
    trim -= 2 as f32 * tf_estimate;
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
    return trim_index;
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
            sumLR = sumLR + ((L).abs() + (R).abs());
            sumMS = sumMS + ((M).abs() + (S).abs());
            j += 1;
        }
        i += 1;
    }
    sumMS = 0.707107f32 * sumMS;
    thetas = 13;
    if LM <= 1 {
        thetas -= 8;
    }
    return ((((m.eBands[13] as i32) << LM + 1) + thetas) as f32 * sumMS
        > ((m.eBands[13] as i32) << LM + 1) as f32 * sumLR) as i32;
}
/// Upstream C: celt/celt_encoder.c:median_of_5
fn median_of_5(x: &[opus_val16]) -> opus_val16 {
    let mut t0: opus_val16;
    let mut t1: opus_val16;
    let t2: opus_val16;
    let mut t3: opus_val16;
    let mut t4: opus_val16;
    t2 = x[2];
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
        let tmp: opus_val16 = t0;
        t0 = t3;
        t3 = tmp;
        let tmp_0: opus_val16 = t1;
        t1 = t4;
        t4 = tmp_0;
    }
    if t2 > t1 {
        if t1 < t3 {
            return if t2 < t3 { t2 } else { t3 };
        } else {
            return if t4 < t1 { t4 } else { t1 };
        }
    } else if t2 < t3 {
        return if t1 < t3 { t1 } else { t3 };
    } else {
        return if t2 < t4 { t2 } else { t4 };
    };
}
/// Upstream C: celt/celt_encoder.c:median_of_3
fn median_of_3(x: &[opus_val16]) -> opus_val16 {
    let t0: opus_val16;
    let t1: opus_val16;
    let t2: opus_val16;
    if x[0] > x[1] {
        t0 = x[1];
        t1 = x[0];
    } else {
        t0 = x[0];
        t1 = x[1];
    }
    t2 = x[2];
    if t1 < t2 {
        return t1;
    } else if t0 < t2 {
        return t2;
    } else {
        return t0;
    };
}
/// Upstream C: celt/celt_encoder.c:dynalloc_analysis
fn dynalloc_analysis(
    bandLogE: &[opus_val16],
    bandLogE2: &[opus_val16],
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
) -> opus_val16 {
    let mut i: i32 = 0;
    let mut c: i32 = 0;
    let mut tot_boost: i32 = 0;
    let mut maxDepth: opus_val16 = 0.;
    let vla = (C * nbEBands) as usize;
    let mut follower: Vec<opus_val16> = ::std::vec::from_elem(0., vla);
    let vla_0 = (C * nbEBands) as usize;
    let mut noise_floor: Vec<opus_val16> = ::std::vec::from_elem(0., vla_0);
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
        if !(c < C) {
            break;
        }
    }
    let vla_1 = nbEBands as usize;
    let mut mask: Vec<opus_val16> = ::std::vec::from_elem(0., vla_1);
    let vla_2 = nbEBands as usize;
    let mut sig: Vec<opus_val16> = ::std::vec::from_elem(0., vla_2);
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
    if effectiveBytes > 50 && LM >= 1 && lfe == 0 {
        let mut last: i32 = 0;
        c = 0;
        loop {
            let mut offset: opus_val16 = 0.;
            let mut tmp: opus_val16 = 0.;
            let fb = (c * nbEBands) as usize;
            follower[fb] = bandLogE2[(c * nbEBands) as usize];
            i = 1;
            while i < end {
                if bandLogE2[(c * nbEBands + i) as usize]
                    > bandLogE2[(c * nbEBands + i - 1) as usize] + 0.5f32
                {
                    last = i;
                }
                follower[fb + i as usize] = if follower[fb + (i - 1) as usize] + 1.5f32
                    < bandLogE2[(c * nbEBands + i) as usize]
                {
                    follower[fb + (i - 1) as usize] + 1.5f32
                } else {
                    bandLogE2[(c * nbEBands + i) as usize]
                };
                i += 1;
            }
            i = last - 1;
            while i >= 0 {
                follower[fb + i as usize] = if follower[fb + i as usize]
                    < (if follower[fb + (i + 1) as usize] + 2.0f32
                        < bandLogE2[(c * nbEBands + i) as usize]
                    {
                        follower[fb + (i + 1) as usize] + 2.0f32
                    } else {
                        bandLogE2[(c * nbEBands + i) as usize]
                    }) {
                    follower[fb + i as usize]
                } else if follower[fb + (i + 1) as usize] + 2.0f32
                    < bandLogE2[(c * nbEBands + i) as usize]
                {
                    follower[fb + (i + 1) as usize] + 2.0f32
                } else {
                    bandLogE2[(c * nbEBands + i) as usize]
                };
                i -= 1;
            }
            offset = 1.0f32;
            i = 2;
            while i < end - 2 {
                let med = median_of_5(
                    &bandLogE2[(c * nbEBands + i - 2) as usize..(c * nbEBands + i + 3) as usize],
                ) - offset;
                follower[fb + i as usize] = if follower[fb + i as usize] > med {
                    follower[fb + i as usize]
                } else {
                    med
                };
                i += 1;
            }
            tmp = median_of_3(&bandLogE2[(c * nbEBands) as usize..(c * nbEBands + 3) as usize])
                - offset;
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
            tmp = median_of_3(
                &bandLogE2[(c * nbEBands + end - 3) as usize..(c * nbEBands + end) as usize],
            ) - offset;
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
            if !(c < C) {
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
                follower[i as usize] = 0.5f32 * follower[i as usize];
                i += 1;
            }
        }
        i = start;
        while i < end {
            if i < 8 {
                follower[i as usize] *= 2 as f32;
            }
            if i >= 12 {
                follower[i as usize] = 0.5f32 * follower[i as usize];
            }
            i += 1;
        }
        if analysis.valid != 0 {
            i = start;
            while i < (if (19) < end { 19 } else { end }) {
                follower[i as usize] = follower[i as usize]
                    + 1.0f32 / 64.0f32 * analysis.leak_boost[i as usize] as i32 as f32;
                i += 1;
            }
        }
        i = start;
        while i < end {
            let mut width: i32 = 0;
            let mut boost: i32 = 0;
            let mut boost_bits: i32 = 0;
            follower[i as usize] = if follower[i as usize] < 4 as f32 {
                follower[i as usize]
            } else {
                4 as f32
            };
            width = C * (eBands[(i + 1) as usize] as i32 - eBands[i as usize] as i32) << LM;
            if width < 6 {
                boost = follower[i as usize] as i32;
                boost_bits = boost * width << BITRES;
            } else if width > 48 {
                boost = (follower[i as usize] * 8 as f32) as i32;
                boost_bits = (boost * width << BITRES) / 8;
            } else {
                boost = (follower[i as usize] * width as f32 / 6 as f32) as i32;
                boost_bits = (boost * 6) << BITRES;
            }
            if (vbr == 0 || constrained_vbr != 0 && isTransient == 0)
                && tot_boost + boost_bits >> BITRES >> 3 > 2 * effectiveBytes / 3
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
    return maxDepth;
}
/// Upstream C: celt/celt_encoder.c:run_prefilter
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
    nbAvailableBytes: i32,
    analysis: &AnalysisInfo,
) -> i32 {
    let mut pitch_index: i32 = 0;
    let mut gain1: opus_val16 = 0.;
    let mut pf_threshold: opus_val16;
    let pf_on: i32;
    let mut qg: i32 = 0;
    let mode = st.mode;
    let overlap = mode.overlap as i32;
    let pre_chan_len = (N + COMBFILTER_MAXPERIOD) as usize;
    let vla = (CC as usize) * pre_chan_len;
    let mut _pre: Vec<celt_sig> = ::std::vec::from_elem(0., vla);
    // pre[c] starts at c * pre_chan_len in _pre
    for c in 0..CC as usize {
        let pre_base = c * pre_chan_len;
        // Copy prefilter_mem[c*1024 .. c*1024 + 1024] into pre[c][0..1024]
        _pre[pre_base..pre_base + COMBFILTER_MAXPERIOD as usize].copy_from_slice(
            &st.prefilter_mem
                [c * COMBFILTER_MAXPERIOD as usize..(c + 1) * COMBFILTER_MAXPERIOD as usize],
        );
        // Copy in_0[c*(N+overlap)+overlap .. +N] into pre[c][1024..1024+N]
        let in_src = c * (N + overlap) as usize + overlap as usize;
        _pre[pre_base + COMBFILTER_MAXPERIOD as usize
            ..pre_base + COMBFILTER_MAXPERIOD as usize + N as usize]
            .copy_from_slice(&in_0[in_src..in_src + N as usize]);
    }
    if enabled != 0 {
        let vla_0 = (COMBFILTER_MAXPERIOD + N >> 1) as usize;
        let mut pitch_buf: Vec<opus_val16> = ::std::vec::from_elem(0., vla_0);
        {
            let ds_len = (COMBFILTER_MAXPERIOD + N) as usize;
            let ch0 = &_pre[..ds_len];
            if CC == 2 {
                let ch1 = &_pre[pre_chan_len..pre_chan_len + ds_len];
                pitch_downsample(&[ch0, ch1], pitch_buf.as_mut_slice(), ds_len);
            } else {
                pitch_downsample(&[ch0], pitch_buf.as_mut_slice(), ds_len);
            }
        }
        pitch_index = pitch_search(
            &pitch_buf[(COMBFILTER_MAXPERIOD >> 1) as usize..],
            pitch_buf.as_slice(),
            N,
            COMBFILTER_MAXPERIOD - 3 * COMBFILTER_MINPERIOD,
        );
        pitch_index = COMBFILTER_MAXPERIOD - pitch_index;
        gain1 = remove_doubling(
            pitch_buf.as_slice(),
            COMBFILTER_MAXPERIOD,
            COMBFILTER_MINPERIOD,
            N,
            &mut pitch_index,
            st.prefilter_period,
            st.prefilter_gain,
        );
        if pitch_index > COMBFILTER_MAXPERIOD - 2 {
            pitch_index = COMBFILTER_MAXPERIOD - 2;
        }
        gain1 = 0.7f32 * gain1;
        if st.loss_rate > 2 {
            gain1 = 0.5f32 * gain1;
        }
        if st.loss_rate > 4 {
            gain1 = 0.5f32 * gain1;
        }
        if st.loss_rate > 8 {
            gain1 = 0 as opus_val16;
        }
    } else {
        gain1 = 0 as opus_val16;
        pitch_index = COMBFILTER_MINPERIOD;
    }
    if analysis.valid != 0 {
        gain1 = gain1 * analysis.max_pitch_ratio;
    }
    pf_threshold = 0.2f32;
    if (pitch_index - st.prefilter_period).abs() * 10 > pitch_index {
        pf_threshold += 0.2f32;
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
        qg = (0.5f32 + gain1 * 32 as f32 / 3 as f32).floor() as i32 - 1;
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
    for c in 0..CC as usize {
        let offset: i32 = mode.shortMdctSize - overlap;
        st.prefilter_period = if st.prefilter_period > 15 {
            st.prefilter_period
        } else {
            15
        };
        // Copy in_mem overlap into in_0
        let in_dst = c * (N + overlap) as usize;
        in_0[in_dst..in_dst + overlap as usize]
            .copy_from_slice(&st.in_mem[c * overlap as usize..(c + 1) * overlap as usize]);
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
                    COMBFILTER_MAXPERIOD as usize,
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
                (COMBFILTER_MAXPERIOD + offset) as usize,
                st.prefilter_period,
                pitch_index,
                N - offset,
                -st.prefilter_gain,
                -gain1,
                st.prefilter_tapset,
                prefilter_tapset,
                &mode.window,
                overlap,
                st.arch,
            );
        }
        // Copy end of in_0 back into in_mem overlap
        let in_src = c * (N + overlap) as usize + N as usize;
        st.in_mem[c * overlap as usize..(c + 1) * overlap as usize]
            .copy_from_slice(&in_0[in_src..in_src + overlap as usize]);
        // Update prefilter_mem from _pre
        let pre_base = c * pre_chan_len;
        let pfm_base = c * COMBFILTER_MAXPERIOD as usize;
        if N > COMBFILTER_MAXPERIOD {
            st.prefilter_mem[pfm_base..pfm_base + COMBFILTER_MAXPERIOD as usize].copy_from_slice(
                &_pre[pre_base + N as usize..pre_base + N as usize + COMBFILTER_MAXPERIOD as usize],
            );
        } else {
            // Shift prefilter_mem left by N
            st.prefilter_mem.copy_within(
                pfm_base + N as usize..pfm_base + COMBFILTER_MAXPERIOD as usize,
                pfm_base,
            );
            // Copy last N samples from _pre[1024..1024+N] (which is _pre[pre_base+1024..])
            st.prefilter_mem[pfm_base + COMBFILTER_MAXPERIOD as usize - N as usize
                ..pfm_base + COMBFILTER_MAXPERIOD as usize]
                .copy_from_slice(
                    &_pre[pre_base + COMBFILTER_MAXPERIOD as usize
                        ..pre_base + COMBFILTER_MAXPERIOD as usize + N as usize],
                );
        }
    }
    *gain = gain1;
    *pitch = pitch_index;
    *qgain = qg;
    return pf_on;
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
    floor_depth = ((C * bins << 3) as opus_val32 * maxDepth) as i32;
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
            } else {
                if (32000) < 96000 - bitrate {
                    32000
                } else {
                    96000 - bitrate
                }
            }) as f32;
        tvbr_factor = temporal_vbr * amount;
        target += (tvbr_factor * target as f32) as i32;
    }
    target = if 2 * base_target < target {
        2 * base_target
    } else {
        target
    };
    return target;
}
pub unsafe fn celt_encode_with_ec(
    st: *mut OpusCustomEncoder,
    pcm: *const opus_val16,
    mut frame_size: i32,
    compressed: *mut u8,
    mut nbCompressedBytes: i32,
    mut enc: Option<&mut ec_enc>,
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
    let mut oldBandE: *mut opus_val16 = 0 as *mut opus_val16;
    let mut oldLogE: *mut opus_val16 = 0 as *mut opus_val16;
    let mut oldLogE2: *mut opus_val16 = 0 as *mut opus_val16;
    let mut energyError: *mut opus_val16 = 0 as *mut opus_val16;
    let mut shortBlocks: i32 = 0;
    let mut isTransient: i32 = 0;
    let CC: i32 = (*st).channels;
    let C: i32 = (*st).stream_channels;
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
    let mut pitch_change: i32 = 0;
    let mut tot_boost: i32 = 0;
    let mut sample_max: opus_val32 = 0.;
    let mut maxDepth: opus_val16 = 0.;
    let mut mode: *const OpusCustomMode = 0 as *const OpusCustomMode;
    let mut nbEBands: i32 = 0;
    let mut overlap: i32 = 0;
    let mut eBands: *const i16 = 0 as *const i16;
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
    mode = (*st).mode;
    nbEBands = (*mode).nbEBands as i32;
    overlap = (*mode).overlap as i32;
    eBands = (*mode).eBands.as_ptr();
    start = (*st).start;
    end = (*st).end;
    hybrid = (start != 0) as i32;
    tf_estimate = 0 as opus_val16;
    if nbCompressedBytes < 2 || pcm.is_null() {
        return OPUS_BAD_ARG;
    }
    frame_size *= (*st).upsample;
    LM = 0;
    while LM <= (*mode).maxLM {
        if (*mode).shortMdctSize << LM == frame_size {
            break;
        }
        LM += 1;
    }
    if LM > (*mode).maxLM {
        return OPUS_BAD_ARG;
    }
    M = (1) << LM;
    N = M * (*mode).shortMdctSize;
    oldBandE = (*st).oldBandE.as_mut_ptr();
    oldLogE = (*st).oldLogE.as_mut_ptr();
    oldLogE2 = (*st).oldLogE2.as_mut_ptr();
    energyError = (*st).energyError.as_mut_ptr();
    if let Some(enc) = enc.as_mut() {
        tell0_frac = ec_tell_frac(enc) as i32;
        tell = ec_tell(enc);
        nbFilledBytes = tell + 4 >> 3;
    } else {
        tell = 1;
        tell0_frac = tell;
        nbFilledBytes = 0;
    }
    assert!((*st).signalling == 0);
    nbCompressedBytes = if nbCompressedBytes < 1275 {
        nbCompressedBytes
    } else {
        1275
    };
    nbAvailableBytes = nbCompressedBytes - nbFilledBytes;
    if (*st).vbr != 0 && (*st).bitrate != OPUS_BITRATE_MAX {
        let den: i32 = (*mode).Fs >> BITRES;
        vbr_rate = ((*st).bitrate * frame_size + (den >> 1)) / den;
        effectiveBytes = vbr_rate >> 3 + BITRES;
    } else {
        let mut tmp: i32 = 0;
        vbr_rate = 0;
        tmp = (*st).bitrate * frame_size;
        if tell > 1 {
            tmp += tell;
        }
        if (*st).bitrate != OPUS_BITRATE_MAX {
            nbCompressedBytes = if 2
                > (if nbCompressedBytes
                    < (tmp + 4 * (*mode).Fs) / (8 * (*mode).Fs) - ((*st).signalling != 0) as i32
                {
                    nbCompressedBytes
                } else {
                    (tmp + 4 * (*mode).Fs) / (8 * (*mode).Fs) - ((*st).signalling != 0) as i32
                }) {
                2
            } else if nbCompressedBytes
                < (tmp + 4 * (*mode).Fs) / (8 * (*mode).Fs) - ((*st).signalling != 0) as i32
            {
                nbCompressedBytes
            } else {
                (tmp + 4 * (*mode).Fs) / (8 * (*mode).Fs) - ((*st).signalling != 0) as i32
            };
        }
        effectiveBytes = nbCompressedBytes - nbFilledBytes;
    }
    equiv_rate = (nbCompressedBytes * 8 * 50 >> 3 - LM) - (40 * C + 20) * ((400 >> LM) - 50);
    if (*st).bitrate != OPUS_BITRATE_MAX {
        equiv_rate = if equiv_rate < (*st).bitrate - (40 * C + 20) * ((400 >> LM) - 50) {
            equiv_rate
        } else {
            (*st).bitrate - (40 * C + 20) * ((400 >> LM) - 50)
        };
    }
    let enc = if let Some(enc) = enc {
        enc
    } else {
        assert!(!compressed.is_null());
        _enc = ec_enc_init(std::slice::from_raw_parts_mut(
            compressed,
            nbCompressedBytes as usize,
        ));
        &mut _enc
    };
    if vbr_rate > 0 {
        if (*st).constrained_vbr != 0 {
            let mut vbr_bound: i32 = 0;
            let mut max_allowed: i32 = 0;
            vbr_bound = vbr_rate;
            max_allowed = if (if (if tell == 1 { 2 } else { 0 })
                > vbr_rate + vbr_bound - (*st).vbr_reservoir >> 3 + 3
            {
                if tell == 1 {
                    2
                } else {
                    0
                }
            } else {
                vbr_rate + vbr_bound - (*st).vbr_reservoir >> 3 + 3
            }) < nbAvailableBytes
            {
                if (if tell == 1 { 2 } else { 0 })
                    > vbr_rate + vbr_bound - (*st).vbr_reservoir >> 3 + 3
                {
                    if tell == 1 {
                        2
                    } else {
                        0
                    }
                } else {
                    vbr_rate + vbr_bound - (*st).vbr_reservoir >> 3 + 3
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
    }
    total_bits = nbCompressedBytes * 8;
    effEnd = end;
    if effEnd > (*mode).effEBands {
        effEnd = (*mode).effEBands;
    }
    let vla = (CC * (N + overlap)) as usize;
    let mut in_0: Vec<celt_sig> = ::std::vec::from_elem(0., vla);
    let main_len = (C * (N - overlap) / (*st).upsample) as usize;
    let overlap_len = (C * overlap / (*st).upsample) as usize;
    sample_max = if (*st).overlap_max > celt_maxabs16(std::slice::from_raw_parts(pcm, main_len)) {
        (*st).overlap_max
    } else {
        celt_maxabs16(std::slice::from_raw_parts(pcm, main_len))
    };
    (*st).overlap_max = celt_maxabs16(std::slice::from_raw_parts(pcm.add(main_len), overlap_len));
    sample_max = if sample_max > (*st).overlap_max {
        sample_max
    } else {
        (*st).overlap_max
    };
    silence = (sample_max <= 1 as opus_val16 / ((1) << (*st).lsb_depth) as f32) as i32;
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
        need_clip = ((*st).clip != 0 && sample_max > 65536.0f32) as i32;
        celt_preemphasis(
            std::slice::from_raw_parts(
                pcm.offset(c as isize),
                (CC * (N / (*st).upsample)) as usize,
            ),
            &mut in_0[(c * (N + overlap) + overlap) as usize..],
            N,
            CC,
            (*st).upsample,
            &(*mode).preemph,
            &mut (*st).preemph_memE[c as usize],
            need_clip,
        );
        c += 1;
        if !(c < CC) {
            break;
        }
    }
    let mut enabled: i32 = 0;
    let mut qg: i32 = 0;
    enabled = (((*st).lfe != 0 && nbAvailableBytes > 3 || nbAvailableBytes > 12 * C)
        && hybrid == 0
        && silence == 0
        && (*st).disable_pf == 0
        && (*st).complexity >= 5) as i32;
    prefilter_tapset = (*st).tapset_decision;
    {
        let analysis = (*st).analysis;
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
            nbAvailableBytes,
            &analysis,
        );
    }
    #[cfg(feature = "ent-dump")]
    eprintln!("prefilter: pitch_index={pitch_index}, gain1={gain1:1.6}, qg={qg}");
    if (gain1 > 0.4f32 || (*st).prefilter_gain > 0.4f32)
        && ((*st).analysis.valid == 0 || (*st).analysis.tonality as f64 > 0.3f64)
        && (pitch_index as f64 > 1.26f64 * (*st).prefilter_period as f64
            || (pitch_index as f64) < 0.79f64 * (*st).prefilter_period as f64)
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
    isTransient = 0;
    shortBlocks = 0;
    if (*st).complexity >= 1 && (*st).lfe == 0 {
        let allow_weak_transients: i32 =
            (hybrid != 0 && effectiveBytes < 15 && (*st).silk_info.signalType != 2) as i32;
        isTransient = transient_analysis(
            &in_0,
            N + overlap,
            CC,
            &mut tf_estimate,
            &mut tf_chan,
            allow_weak_transients,
            &mut weak_transient,
        );
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
    let vla_0 = (CC * N + M - 1) as usize;
    let mut freq: Vec<celt_sig> = ::std::vec::from_elem(0., vla_0);
    let vla_1 = (nbEBands * CC) as usize;
    let mut bandE: Vec<celt_ener> = ::std::vec::from_elem(0., vla_1);
    let vla_2 = (nbEBands * CC) as usize;
    let mut bandLogE: Vec<opus_val16> = ::std::vec::from_elem(0., vla_2);
    secondMdct = (shortBlocks != 0 && (*st).complexity >= 8) as i32;
    let vla_3 = (C * nbEBands) as usize;
    let mut bandLogE2: Vec<opus_val16> = ::std::vec::from_elem(0., vla_3);
    if secondMdct != 0 {
        compute_mdcts(&*mode, 0, &mut in_0, &mut freq, C, CC, LM, (*st).upsample);
        compute_band_energies(&*mode, &freq, &mut bandE, effEnd, C, LM, (*st).arch);
        amp2Log2(&*mode, effEnd, end, &bandE, &mut bandLogE2, C);
        i = 0;
        while i < C * nbEBands {
            bandLogE2[i as usize] += 0.5f32 * LM as f32;
            i += 1;
        }
    }
    compute_mdcts(
        &*mode,
        shortBlocks,
        &mut in_0,
        &mut freq,
        C,
        CC,
        LM,
        (*st).upsample,
    );
    assert!(!(freq[0] != freq[0]) && (C == 1 || !(freq[N as usize] != freq[N as usize])));
    if CC == 2 && C == 1 {
        tf_chan = 0;
    }
    compute_band_energies(&*mode, &freq, &mut bandE, effEnd, C, LM, (*st).arch);
    if (*st).lfe != 0 {
        i = 2;
        while i < end {
            *bandE.as_mut_ptr().offset(i as isize) = if *bandE.as_mut_ptr().offset(i as isize)
                < 1e-4f32 * *bandE.as_mut_ptr().offset(0 as isize)
            {
                *bandE.as_mut_ptr().offset(i as isize)
            } else {
                1e-4f32 * *bandE.as_mut_ptr().offset(0 as isize)
            };
            *bandE.as_mut_ptr().offset(i as isize) =
                if *bandE.as_mut_ptr().offset(i as isize) > 1e-15f32 {
                    *bandE.as_mut_ptr().offset(i as isize)
                } else {
                    1e-15f32
                };
            i += 1;
        }
    }
    amp2Log2(&*mode, effEnd, end, &bandE, &mut bandLogE, C);
    let vla_4 = (C * nbEBands) as usize;
    let mut surround_dynalloc: Vec<opus_val16> = ::std::vec::from_elem(0., vla_4);
    memset(
        surround_dynalloc.as_mut_ptr() as *mut core::ffi::c_void,
        0,
        (end as u64).wrapping_mul(::core::mem::size_of::<opus_val16>() as u64),
    );
    if hybrid == 0 && !((*st).energy_mask).is_null() && (*st).lfe == 0 {
        let mut mask_end: i32 = 0;
        let mut midband: i32 = 0;
        let mut count_dynalloc: i32 = 0;
        let mut mask_avg: opus_val32 = 0 as opus_val32;
        let mut diff: opus_val32 = 0 as opus_val32;
        let mut count: i32 = 0;
        mask_end = if 2 > (*st).lastCodedBands {
            2
        } else {
            (*st).lastCodedBands
        };
        c = 0;
        while c < C {
            i = 0;
            while i < mask_end {
                let mut mask: opus_val16 = 0.;
                mask = if (if *((*st).energy_mask).offset((nbEBands * c + i) as isize) < 0.25f32 {
                    *((*st).energy_mask).offset((nbEBands * c + i) as isize)
                } else {
                    0.25f32
                }) > -2.0f32
                {
                    if *((*st).energy_mask).offset((nbEBands * c + i) as isize) < 0.25f32 {
                        *((*st).energy_mask).offset((nbEBands * c + i) as isize)
                    } else {
                        0.25f32
                    }
                } else {
                    -2.0f32
                };
                if mask > 0 as f32 {
                    mask = 0.5f32 * mask;
                }
                mask_avg += mask
                    * (*eBands.offset((i + 1) as isize) as i32 - *eBands.offset(i as isize) as i32)
                        as opus_val32;
                count +=
                    *eBands.offset((i + 1) as isize) as i32 - *eBands.offset(i as isize) as i32;
                diff += mask * (1 + 2 * i - mask_end) as opus_val32;
                i += 1;
            }
            c += 1;
        }
        assert!(count > 0);
        mask_avg = mask_avg / count as opus_val16;
        mask_avg += 0.2f32;
        diff = diff * 6 as f32 / (C * (mask_end - 1) * (mask_end + 1) * mask_end) as f32;
        diff = 0.5f32 * diff;
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
        while (*eBands.offset((midband + 1) as isize) as i32)
            < *eBands.offset(mask_end as isize) as i32 / 2
        {
            midband += 1;
        }
        count_dynalloc = 0;
        i = 0;
        while i < mask_end {
            let mut lin: opus_val32 = 0.;
            let mut unmask: opus_val16 = 0.;
            lin = mask_avg + diff * (i - midband) as f32;
            if C == 2 {
                unmask = if *((*st).energy_mask).offset(i as isize)
                    > *((*st).energy_mask).offset((nbEBands + i) as isize)
                {
                    *((*st).energy_mask).offset(i as isize)
                } else {
                    *((*st).energy_mask).offset((nbEBands + i) as isize)
                };
            } else {
                unmask = *((*st).energy_mask).offset(i as isize);
            }
            unmask = if unmask < 0.0f32 { unmask } else { 0.0f32 };
            unmask -= lin;
            if unmask > 0.25f32 {
                *surround_dynalloc.as_mut_ptr().offset(i as isize) = unmask - 0.25f32;
                count_dynalloc += 1;
            }
            i += 1;
        }
        if count_dynalloc >= 3 {
            mask_avg += 0.25f32;
            if mask_avg > 0 as f32 {
                mask_avg = 0 as opus_val32;
                diff = 0 as opus_val32;
                memset(
                    surround_dynalloc.as_mut_ptr() as *mut core::ffi::c_void,
                    0,
                    (mask_end as u64).wrapping_mul(::core::mem::size_of::<opus_val16>() as u64),
                );
            } else {
                i = 0;
                while i < mask_end {
                    *surround_dynalloc.as_mut_ptr().offset(i as isize) = if 0 as f32
                        > *surround_dynalloc.as_mut_ptr().offset(i as isize) - 0.25f32
                    {
                        0 as f32
                    } else {
                        *surround_dynalloc.as_mut_ptr().offset(i as isize) - 0.25f32
                    };
                    i += 1;
                }
            }
        }
        mask_avg += 0.2f32;
        surround_trim = 64 as f32 * diff;
        surround_masking = mask_avg;
    }
    if (*st).lfe == 0 {
        let mut follow: opus_val16 = -10.0f32;
        let mut frame_avg: opus_val32 = 0 as opus_val32;
        let offset: opus_val16 = if shortBlocks != 0 {
            0.5f32 * LM as f32
        } else {
            0 as f32
        };
        i = start;
        while i < end {
            follow = if follow - 1.0f32 > *bandLogE.as_mut_ptr().offset(i as isize) - offset {
                follow - 1.0f32
            } else {
                *bandLogE.as_mut_ptr().offset(i as isize) - offset
            };
            if C == 2 {
                follow = if follow > *bandLogE.as_mut_ptr().offset((i + nbEBands) as isize) - offset
                {
                    follow
                } else {
                    *bandLogE.as_mut_ptr().offset((i + nbEBands) as isize) - offset
                };
            }
            frame_avg += follow;
            i += 1;
        }
        frame_avg /= (end - start) as f32;
        temporal_vbr = frame_avg - (*st).spec_avg;
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
        (*st).spec_avg += 0.02f32 * temporal_vbr;
    }
    if secondMdct == 0 {
        memcpy(
            bandLogE2.as_mut_ptr() as *mut core::ffi::c_void,
            bandLogE.as_mut_ptr() as *const core::ffi::c_void,
            ((C * nbEBands) as u64)
                .wrapping_mul(::core::mem::size_of::<opus_val16>() as u64)
                .wrapping_add(
                    (0 * bandLogE2.as_mut_ptr().offset_from(bandLogE.as_mut_ptr()) as i64) as u64,
                ),
        );
    }
    if LM > 0
        && ec_tell(enc) + 3 <= total_bits
        && isTransient == 0
        && (*st).complexity >= 5
        && (*st).lfe == 0
        && hybrid == 0
    {
        if patch_transient_decision(
            &bandLogE,
            std::slice::from_raw_parts(oldBandE, (C * nbEBands) as usize),
            nbEBands,
            start,
            end,
            C,
        ) != 0
        {
            isTransient = 1;
            shortBlocks = M;
            compute_mdcts(
                &*mode,
                shortBlocks,
                &mut in_0,
                &mut freq,
                C,
                CC,
                LM,
                (*st).upsample,
            );
            compute_band_energies(&*mode, &freq, &mut bandE, effEnd, C, LM, (*st).arch);
            amp2Log2(&*mode, effEnd, end, &bandE, &mut bandLogE, C);
            i = 0;
            while i < C * nbEBands {
                bandLogE2[i as usize] += 0.5f32 * LM as f32;
                i += 1;
            }
            tf_estimate = 0.2f32;
        }
    }
    if LM > 0 && ec_tell(enc) + 3 <= total_bits {
        ec_enc_bit_logp(enc, isTransient, 3);
    }
    let vla_5 = (C * N) as usize;
    let mut X: Vec<celt_norm> = ::std::vec::from_elem(0., vla_5);
    normalise_bands(&*mode, &freq, &mut X, &bandE, effEnd, C, M);
    enable_tf_analysis =
        (effectiveBytes >= 15 * C && hybrid == 0 && (*st).complexity >= 2 && (*st).lfe == 0) as i32;
    let vla_6 = nbEBands as usize;
    let mut offsets: Vec<i32> = ::std::vec::from_elem(0, vla_6);
    let vla_7 = nbEBands as usize;
    let mut importance: Vec<i32> = ::std::vec::from_elem(0, vla_7);
    let vla_8 = nbEBands as usize;
    let mut spread_weight: Vec<i32> = ::std::vec::from_elem(0, vla_8);
    maxDepth = dynalloc_analysis(
        &bandLogE,
        &bandLogE2,
        nbEBands,
        start,
        end,
        C,
        &mut offsets,
        (*st).lsb_depth,
        &(*mode).logN,
        isTransient,
        (*st).vbr,
        (*st).constrained_vbr,
        &(*mode).eBands,
        LM,
        effectiveBytes,
        &mut tot_boost,
        (*st).lfe,
        &surround_dynalloc,
        &(*st).analysis,
        &mut importance,
        &mut spread_weight,
    );
    let vla_9 = nbEBands as usize;
    let mut tf_res: Vec<i32> = ::std::vec::from_elem(0, vla_9);
    if enable_tf_analysis != 0 {
        let mut lambda: i32 = 0;
        lambda = if 80 > 20480 / effectiveBytes + 2 {
            80
        } else {
            20480 / effectiveBytes + 2
        };
        tf_select = tf_analysis(
            &*mode,
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
            *tf_res.as_mut_ptr().offset(i as isize) =
                *tf_res.as_mut_ptr().offset((effEnd - 1) as isize);
            i += 1;
        }
    } else if hybrid != 0 && weak_transient != 0 {
        i = 0;
        while i < end {
            *tf_res.as_mut_ptr().offset(i as isize) = 1;
            i += 1;
        }
        tf_select = 0;
    } else if hybrid != 0 && effectiveBytes < 15 && (*st).silk_info.signalType != 2 {
        i = 0;
        while i < end {
            *tf_res.as_mut_ptr().offset(i as isize) = 0;
            i += 1;
        }
        tf_select = isTransient;
    } else {
        i = 0;
        while i < end {
            *tf_res.as_mut_ptr().offset(i as isize) = isTransient;
            i += 1;
        }
        tf_select = 0;
    }
    let vla_10 = (C * nbEBands) as usize;
    let mut error: Vec<opus_val16> = ::std::vec::from_elem(0., vla_10);
    c = 0;
    loop {
        i = start;
        while i < end {
            if (*bandLogE.as_mut_ptr().offset((i + c * nbEBands) as isize)
                - *oldBandE.offset((i + c * nbEBands) as isize))
            .abs()
                < 2.0f32
            {
                let ref mut fresh4 = *bandLogE.as_mut_ptr().offset((i + c * nbEBands) as isize);
                *fresh4 -= *energyError.offset((i + c * nbEBands) as isize) * 0.25f32;
            }
            i += 1;
        }
        c += 1;
        if !(c < C) {
            break;
        }
    }
    quant_coarse_energy(
        &*mode,
        start,
        end,
        effEnd,
        &bandLogE,
        std::slice::from_raw_parts_mut(oldBandE, (C * nbEBands) as usize),
        total_bits as u32,
        &mut error,
        enc,
        C,
        LM,
        nbAvailableBytes,
        (*st).force_intra,
        &mut (*st).delayedIntra,
        ((*st).complexity >= 4) as i32,
        (*st).loss_rate,
        (*st).lfe,
    );
    tf_encode(start, end, isTransient, &mut tf_res, LM, tf_select, enc);
    if ec_tell(enc) + 4 <= total_bits {
        if (*st).lfe != 0 {
            (*st).tapset_decision = 0;
            (*st).spread_decision = SPREAD_NORMAL;
        } else if hybrid != 0 {
            if (*st).complexity == 0 {
                (*st).spread_decision = SPREAD_NONE;
            } else if isTransient != 0 {
                (*st).spread_decision = SPREAD_NORMAL;
            } else {
                (*st).spread_decision = SPREAD_AGGRESSIVE;
            }
        } else if shortBlocks != 0 || (*st).complexity < 3 || nbAvailableBytes < 10 * C {
            if (*st).complexity == 0 {
                (*st).spread_decision = SPREAD_NONE;
            } else {
                (*st).spread_decision = SPREAD_NORMAL;
            }
        } else {
            (*st).spread_decision = spreading_decision(
                &*mode,
                &X,
                &mut (*st).tonal_average,
                (*st).spread_decision,
                &mut (*st).hf_average,
                &mut (*st).tapset_decision,
                (pf_on != 0 && shortBlocks == 0) as i32,
                effEnd,
                C,
                M,
                &spread_weight,
            );
        }
        ec_enc_icdf(enc, (*st).spread_decision, &spread_icdf, 5);
    }
    if (*st).lfe != 0 {
        *offsets.as_mut_ptr().offset(0 as isize) = if (8) < effectiveBytes / 3 {
            8
        } else {
            effectiveBytes / 3
        };
    }
    let vla_11 = nbEBands as usize;
    let mut cap: Vec<i32> = ::std::vec::from_elem(0, vla_11);
    init_caps(&*mode, &mut cap, LM, C);
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
        width =
            C * (*eBands.offset((i + 1) as isize) as i32 - *eBands.offset(i as isize) as i32) << LM;
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
            && boost < *cap.as_mut_ptr().offset(i as isize)
        {
            let mut flag: i32 = 0;
            flag = (j < *offsets.as_mut_ptr().offset(i as isize)) as i32;
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
        *offsets.as_mut_ptr().offset(i as isize) = boost;
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
            dual_stereo = stereo_analysis(&*mode, &X, LM, N);
        }
        (*st).intensity = hysteresis_decision(
            (equiv_rate / 1000) as opus_val16,
            &intensity_thresholds,
            &intensity_histeresis,
            21,
            (*st).intensity,
        );
        (*st).intensity = if end
            < (if start > (*st).intensity {
                start
            } else {
                (*st).intensity
            }) {
            end
        } else if start > (*st).intensity {
            start
        } else {
            (*st).intensity
        };
    }
    alloc_trim = 5;
    if tell + ((6) << BITRES) <= total_bits - total_boost {
        if start > 0 || (*st).lfe != 0 {
            (*st).stereo_saving = 0 as opus_val16;
            alloc_trim = 5;
        } else {
            alloc_trim = alloc_trim_analysis(
                &*mode,
                &X,
                &bandLogE,
                end,
                LM,
                C,
                N,
                &(*st).analysis,
                &mut (*st).stereo_saving,
                tf_estimate,
                (*st).intensity,
                surround_trim,
                equiv_rate,
                (*st).arch,
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
        let lm_diff: i32 = (*mode).maxLM - LM;
        nbCompressedBytes = if nbCompressedBytes < 1275 >> 3 - LM {
            nbCompressedBytes
        } else {
            1275 >> 3 - LM
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
        if (*st).constrained_vbr != 0 {
            base_target += (*st).vbr_offset >> lm_diff;
        }
        if hybrid == 0 {
            target = compute_vbr(
                &*mode,
                &(*st).analysis,
                base_target,
                LM,
                equiv_rate,
                (*st).lastCodedBands,
                C,
                (*st).intensity,
                (*st).constrained_vbr,
                (*st).stereo_saving,
                tot_boost,
                tf_estimate,
                pitch_change,
                maxDepth,
                (*st).lfe,
                ((*st).energy_mask != NULL as *const opus_val16) as i32,
                surround_masking,
                temporal_vbr,
            );
        } else {
            target = base_target;
            if (*st).silk_info.offset < 100 {
                target += (12) << BITRES >> 3 - LM;
            }
            if (*st).silk_info.offset > 100 {
                target -= (18) << BITRES >> 3 - LM;
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
        target = target + tell;
        min_allowed = (tell + total_boost + ((1) << BITRES + 3) - 1 >> BITRES + 3) + 2;
        if hybrid != 0 {
            min_allowed = if min_allowed
                > tell0_frac + ((37) << 3) + total_boost + ((1) << 3 + 3) - 1 >> 3 + 3
            {
                min_allowed
            } else {
                tell0_frac + ((37) << 3) + total_boost + ((1) << 3 + 3) - 1 >> 3 + 3
            };
        }
        nbAvailableBytes = target + ((1) << BITRES + 2) >> BITRES + 3;
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
        target = nbAvailableBytes << BITRES + 3;
        if silence != 0 {
            nbAvailableBytes = 2;
            target = (2 * 8) << BITRES;
            delta = 0;
        }
        if (*st).vbr_count < 970 {
            (*st).vbr_count += 1;
            alpha = 1.0f32 / ((*st).vbr_count + 20) as f32;
        } else {
            alpha = 0.001f32;
        }
        if (*st).constrained_vbr != 0 {
            (*st).vbr_reservoir += target - vbr_rate;
        }
        if (*st).constrained_vbr != 0 {
            (*st).vbr_drift += (alpha
                * (delta * ((1) << lm_diff) - (*st).vbr_offset - (*st).vbr_drift) as f32)
                as i32;
            (*st).vbr_offset = -(*st).vbr_drift;
        }
        if (*st).constrained_vbr != 0 && (*st).vbr_reservoir < 0 {
            let adjust: i32 = -(*st).vbr_reservoir / ((8) << BITRES);
            nbAvailableBytes += if silence != 0 { 0 } else { adjust };
            (*st).vbr_reservoir = 0;
        }
        nbCompressedBytes = if nbCompressedBytes < nbAvailableBytes {
            nbCompressedBytes
        } else {
            nbAvailableBytes
        };
        ec_enc_shrink(enc, nbCompressedBytes as u32);
    }
    let vla_12 = nbEBands as usize;
    let mut fine_quant: Vec<i32> = ::std::vec::from_elem(0, vla_12);
    let vla_13 = nbEBands as usize;
    let mut pulses: Vec<i32> = ::std::vec::from_elem(0, vla_13);
    let vla_14 = nbEBands as usize;
    let mut fine_priority: Vec<i32> = ::std::vec::from_elem(0, vla_14);
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
    if (*st).analysis.valid != 0 {
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
        signalBandwidth = if (*st).analysis.bandwidth > min_bandwidth {
            (*st).analysis.bandwidth
        } else {
            min_bandwidth
        };
    }
    if (*st).lfe != 0 {
        signalBandwidth = 1;
    }
    codedBands = clt_compute_allocation(
        &*mode,
        start,
        end,
        &offsets,
        &cap,
        alloc_trim,
        &mut (*st).intensity,
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
        (*st).lastCodedBands,
        signalBandwidth,
    );
    if (*st).lastCodedBands != 0 {
        (*st).lastCodedBands = if ((*st).lastCodedBands + 1)
            < (if (*st).lastCodedBands - 1 > codedBands {
                (*st).lastCodedBands - 1
            } else {
                codedBands
            }) {
            (*st).lastCodedBands + 1
        } else if (*st).lastCodedBands - 1 > codedBands {
            (*st).lastCodedBands - 1
        } else {
            codedBands
        };
    } else {
        (*st).lastCodedBands = codedBands;
    }
    quant_fine_energy(
        &*mode,
        start,
        end,
        std::slice::from_raw_parts_mut(oldBandE, (C * nbEBands) as usize),
        &mut error,
        &fine_quant,
        enc,
        C,
    );
    let vla_15 = (C * nbEBands) as usize;
    let mut collapse_masks: Vec<u8> = ::std::vec::from_elem(0, vla_15);
    if C == 2 {
        let (x_part, y_part) = X.split_at_mut(N as usize);
        quant_all_bands(
            1,
            &*mode,
            start,
            end,
            x_part,
            Some(y_part),
            &mut collapse_masks,
            &bandE,
            &mut pulses,
            shortBlocks,
            (*st).spread_decision,
            dual_stereo,
            (*st).intensity,
            &mut tf_res,
            nbCompressedBytes * ((8) << BITRES) - anti_collapse_rsv,
            balance,
            enc,
            LM,
            codedBands,
            &mut (*st).rng,
            (*st).complexity,
            (*st).arch,
            (*st).disable_inv,
        );
    } else {
        quant_all_bands(
            1,
            &*mode,
            start,
            end,
            &mut X,
            None,
            &mut collapse_masks,
            &bandE,
            &mut pulses,
            shortBlocks,
            (*st).spread_decision,
            dual_stereo,
            (*st).intensity,
            &mut tf_res,
            nbCompressedBytes * ((8) << BITRES) - anti_collapse_rsv,
            balance,
            enc,
            LM,
            codedBands,
            &mut (*st).rng,
            (*st).complexity,
            (*st).arch,
            (*st).disable_inv,
        );
    }
    if anti_collapse_rsv > 0 {
        anti_collapse_on = ((*st).consec_transient < 2) as i32;
        ec_enc_bits(enc, anti_collapse_on as u32, 1);
    }
    quant_energy_finalise(
        &*mode,
        start,
        end,
        std::slice::from_raw_parts_mut(oldBandE, (C * nbEBands) as usize),
        &mut error,
        &fine_quant,
        &fine_priority,
        nbCompressedBytes * 8 - ec_tell(enc),
        enc,
        C,
    );
    memset(
        energyError as *mut core::ffi::c_void,
        0,
        ((nbEBands * CC) as u64).wrapping_mul(::core::mem::size_of::<opus_val16>() as u64),
    );
    c = 0;
    loop {
        i = start;
        while i < end {
            *energyError.offset((i + c * nbEBands) as isize) = if -0.5f32
                > (if 0.5f32 < *error.as_mut_ptr().offset((i + c * nbEBands) as isize) {
                    0.5f32
                } else {
                    *error.as_mut_ptr().offset((i + c * nbEBands) as isize)
                }) {
                -0.5f32
            } else if 0.5f32 < *error.as_mut_ptr().offset((i + c * nbEBands) as isize) {
                0.5f32
            } else {
                *error.as_mut_ptr().offset((i + c * nbEBands) as isize)
            };
            i += 1;
        }
        c += 1;
        if !(c < C) {
            break;
        }
    }
    if silence != 0 {
        i = 0;
        while i < C * nbEBands {
            *oldBandE.offset(i as isize) = -28.0f32;
            i += 1;
        }
    }
    (*st).prefilter_period = pitch_index;
    (*st).prefilter_gain = gain1;
    (*st).prefilter_tapset = prefilter_tapset;
    if CC == 2 && C == 1 {
        memcpy(
            &mut *oldBandE.offset(nbEBands as isize) as *mut opus_val16 as *mut core::ffi::c_void,
            oldBandE as *const core::ffi::c_void,
            (nbEBands as u64)
                .wrapping_mul(::core::mem::size_of::<opus_val16>() as u64)
                .wrapping_add(
                    (0 * (&mut *oldBandE.offset(nbEBands as isize) as *mut opus_val16)
                        .offset_from(oldBandE) as i64) as u64,
                ),
        );
    }
    if isTransient == 0 {
        memcpy(
            oldLogE2 as *mut core::ffi::c_void,
            oldLogE as *const core::ffi::c_void,
            ((CC * nbEBands) as u64)
                .wrapping_mul(::core::mem::size_of::<opus_val16>() as u64)
                .wrapping_add((0 * oldLogE2.offset_from(oldLogE) as i64) as u64),
        );
        memcpy(
            oldLogE as *mut core::ffi::c_void,
            oldBandE as *const core::ffi::c_void,
            ((CC * nbEBands) as u64)
                .wrapping_mul(::core::mem::size_of::<opus_val16>() as u64)
                .wrapping_add((0 * oldLogE.offset_from(oldBandE) as i64) as u64),
        );
    } else {
        i = 0;
        while i < CC * nbEBands {
            *oldLogE.offset(i as isize) =
                if *oldLogE.offset(i as isize) < *oldBandE.offset(i as isize) {
                    *oldLogE.offset(i as isize)
                } else {
                    *oldBandE.offset(i as isize)
                };
            i += 1;
        }
    }
    c = 0;
    loop {
        i = 0;
        while i < start {
            *oldBandE.offset((c * nbEBands + i) as isize) = 0 as opus_val16;
            let ref mut fresh5 = *oldLogE2.offset((c * nbEBands + i) as isize);
            *fresh5 = -28.0f32;
            *oldLogE.offset((c * nbEBands + i) as isize) = *fresh5;
            i += 1;
        }
        i = end;
        while i < nbEBands {
            *oldBandE.offset((c * nbEBands + i) as isize) = 0 as opus_val16;
            let ref mut fresh6 = *oldLogE2.offset((c * nbEBands + i) as isize);
            *fresh6 = -28.0f32;
            *oldLogE.offset((c * nbEBands + i) as isize) = *fresh6;
            i += 1;
        }
        c += 1;
        if !(c < CC) {
            break;
        }
    }
    if isTransient != 0 || transient_got_disabled != 0 {
        (*st).consec_transient += 1;
    } else {
        (*st).consec_transient = 0;
    }
    (*st).rng = enc.rng;
    ec_enc_done(enc);
    if ec_get_error(enc) != 0 {
        return OPUS_INTERNAL_ERROR;
    } else {
        return nbCompressedBytes;
    };
}
pub unsafe fn opus_custom_encoder_ctl_impl(
    st: *mut OpusCustomEncoder,
    request: i32,
    args: VarArgs,
) -> i32 {
    let current_block: u64;
    let mut ap = args;
    match request {
        OPUS_SET_COMPLEXITY_REQUEST => {
            let value: i32 = ap.arg::<i32>();
            if value < 0 || value > 10 {
                current_block = 2472048668343472511;
            } else {
                (*st).complexity = value;
                current_block = 10007731352114176167;
            }
        }
        CELT_SET_START_BAND_REQUEST => {
            let value_0: i32 = ap.arg::<i32>();
            if value_0 < 0 || value_0 >= (*(*st).mode).nbEBands as i32 {
                current_block = 2472048668343472511;
            } else {
                (*st).start = value_0;
                current_block = 10007731352114176167;
            }
        }
        CELT_SET_END_BAND_REQUEST => {
            let value_1: i32 = ap.arg::<i32>();
            if value_1 < 1 || value_1 > (*(*st).mode).nbEBands as i32 {
                current_block = 2472048668343472511;
            } else {
                (*st).end = value_1;
                current_block = 10007731352114176167;
            }
        }
        CELT_SET_PREDICTION_REQUEST => {
            let value_2: i32 = ap.arg::<i32>();
            if value_2 < 0 || value_2 > 2 {
                current_block = 2472048668343472511;
            } else {
                (*st).disable_pf = (value_2 <= 1) as i32;
                (*st).force_intra = (value_2 == 0) as i32;
                current_block = 10007731352114176167;
            }
        }
        OPUS_SET_PACKET_LOSS_PERC_REQUEST => {
            let value_3: i32 = ap.arg::<i32>();
            if value_3 < 0 || value_3 > 100 {
                current_block = 2472048668343472511;
            } else {
                (*st).loss_rate = value_3;
                current_block = 10007731352114176167;
            }
        }
        OPUS_SET_VBR_CONSTRAINT_REQUEST => {
            let value_4: i32 = ap.arg::<i32>();
            (*st).constrained_vbr = value_4;
            current_block = 10007731352114176167;
        }
        OPUS_SET_VBR_REQUEST => {
            let value_5: i32 = ap.arg::<i32>();
            (*st).vbr = value_5;
            current_block = 10007731352114176167;
        }
        OPUS_SET_BITRATE_REQUEST => {
            let mut value_6: i32 = ap.arg::<i32>();
            if value_6 <= 500 && value_6 != OPUS_BITRATE_MAX {
                current_block = 2472048668343472511;
            } else {
                value_6 = if value_6 < 260000 * (*st).channels {
                    value_6
                } else {
                    260000 * (*st).channels
                };
                (*st).bitrate = value_6;
                current_block = 10007731352114176167;
            }
        }
        CELT_SET_CHANNELS_REQUEST => {
            let value_7: i32 = ap.arg::<i32>();
            if value_7 < 1 || value_7 > 2 {
                current_block = 2472048668343472511;
            } else {
                (*st).stream_channels = value_7;
                current_block = 10007731352114176167;
            }
        }
        OPUS_SET_LSB_DEPTH_REQUEST => {
            let value_8: i32 = ap.arg::<i32>();
            if value_8 < 8 || value_8 > 24 {
                current_block = 2472048668343472511;
            } else {
                (*st).lsb_depth = value_8;
                current_block = 10007731352114176167;
            }
        }
        OPUS_GET_LSB_DEPTH_REQUEST => {
            let value_9: &mut i32 = ap.arg::<&mut i32>();
            *value_9 = (*st).lsb_depth;
            current_block = 10007731352114176167;
        }
        OPUS_SET_PHASE_INVERSION_DISABLED_REQUEST => {
            let value_10: i32 = ap.arg::<i32>();
            if value_10 < 0 || value_10 > 1 {
                current_block = 2472048668343472511;
            } else {
                (*st).disable_inv = value_10;
                current_block = 10007731352114176167;
            }
        }
        OPUS_GET_PHASE_INVERSION_DISABLED_REQUEST => {
            let value_11: &mut i32 = ap.arg::<&mut i32>();
            *value_11 = (*st).disable_inv;
            current_block = 10007731352114176167;
        }
        OPUS_RESET_STATE => {
            let nbEBands = (*st).mode.nbEBands as usize;
            let cc = (*st).channels as usize;
            let overlap = (*st).mode.overlap;
            // Zero all state fields from rng onward (matching C's memset from &st->rng)
            (*st).rng = 0;
            (*st).spread_decision = SPREAD_NORMAL;
            (*st).delayedIntra = 1 as opus_val32;
            (*st).tonal_average = 256;
            (*st).lastCodedBands = 0;
            (*st).hf_average = 0;
            (*st).tapset_decision = 0;
            (*st).prefilter_period = 0;
            (*st).prefilter_gain = 0.0;
            (*st).prefilter_tapset = 0;
            (*st).consec_transient = 0;
            (*st).analysis = AnalysisInfo {
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
            (*st).silk_info = SILKInfo {
                signalType: 0,
                offset: 0,
            };
            (*st).preemph_memE = [0.0; 2];
            (*st).preemph_memD = [0.0; 2];
            (*st).vbr_reservoir = 0;
            (*st).vbr_drift = 0;
            (*st).vbr_offset = 0;
            (*st).vbr_count = 0;
            (*st).overlap_max = 0.0;
            (*st).stereo_saving = 0.0;
            (*st).intensity = 0;
            (*st).spec_avg = 0.0;
            (&mut (*st).in_mem)[..cc * overlap].fill(0.0);
            (&mut (*st).prefilter_mem)[..cc * COMBFILTER_MAXPERIOD as usize].fill(0.0);
            (&mut (*st).oldBandE)[..cc * nbEBands].fill(0.0);
            (&mut (*st).oldLogE)[..cc * nbEBands].fill(-28.0);
            (&mut (*st).oldLogE2)[..cc * nbEBands].fill(-28.0);
            (&mut (*st).energyError)[..cc * nbEBands].fill(0.0);
            current_block = 10007731352114176167;
        }
        CELT_SET_SIGNALLING_REQUEST => {
            let value_12: i32 = ap.arg::<i32>();
            (*st).signalling = value_12;
            current_block = 10007731352114176167;
        }
        CELT_SET_ANALYSIS_REQUEST => {
            let info = ap.arg::<&mut AnalysisInfo>();
            (*st).analysis = *info;
            current_block = 10007731352114176167;
        }
        CELT_SET_SILK_INFO_REQUEST => {
            let info_0 = ap.arg::<&mut SILKInfo>();
            (*st).silk_info = *info_0;
            current_block = 10007731352114176167;
        }
        CELT_GET_MODE_REQUEST => {
            let value_13 = ap.arg::<&mut *const OpusCustomMode>();
            *value_13 = (*st).mode;
            current_block = 10007731352114176167;
        }
        OPUS_GET_FINAL_RANGE_REQUEST => {
            let value_14 = ap.arg::<&mut u32>();
            *value_14 = (*st).rng;
            current_block = 10007731352114176167;
        }
        OPUS_SET_LFE_REQUEST => {
            let value_15: i32 = ap.arg::<i32>();
            (*st).lfe = value_15;
            current_block = 10007731352114176167;
        }
        OPUS_SET_ENERGY_MASK_REQUEST => {
            let value_16: *const opus_val16 = ap.arg::<*mut opus_val16>();
            (*st).energy_mask = value_16;
            current_block = 10007731352114176167;
        }
        _ => return OPUS_UNIMPLEMENTED,
    }
    match current_block {
        10007731352114176167 => return OPUS_OK,
        _ => return OPUS_BAD_ARG,
    };
}
#[macro_export]
macro_rules! opus_custom_encoder_ctl {
    ($st:expr, $request:expr, $($arg:expr),*) => {
        $crate::opus_custom_encoder_ctl_impl($st, $request, $crate::varargs!($($arg),*))
    };
    ($st:expr, $request:expr) => {
        opus_custom_encoder_ctl!($st, $request,)
    };
    ($st:expr, $request:expr, $($arg:expr),*,) => {
        opus_custom_encoder_ctl!($st, $request, $($arg),*)
    };
}
