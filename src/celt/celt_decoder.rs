//! CELT decoder.
//!
//! Upstream C: `celt/celt_decoder.c`

use crate::celt::bands::{
    anti_collapse, celt_lcg_rand, denormalise_bands, quant_all_bands, SPREAD_NORMAL,
};
use crate::celt::celt_lpc::{_celt_autocorr, _celt_lpc, celt_fir_c, celt_iir, LPC_ORDER};
use crate::celt::common::{
    comb_filter, comb_filter_inplace, init_caps, resampling_factor, spread_icdf, tapset_icdf,
    tf_select_table, trim_icdf, COMBFILTER_MINPERIOD,
};
use crate::celt::entcode::{ec_get_error, ec_tell, ec_tell_frac, BITRES};
use crate::celt::entdec::{
    ec_dec, ec_dec_bit_logp, ec_dec_bits, ec_dec_icdf, ec_dec_init, ec_dec_uint,
};
use crate::celt::mathops::celt_sqrt;
use crate::celt::mdct::mdct_backward;
use crate::celt::modes::{opus_custom_mode_create, OpusCustomMode, MAX_PERIOD};
use crate::celt::pitch;
use crate::celt::quant_bands::{
    unquant_coarse_energy, unquant_energy_finalise, unquant_fine_energy,
};
use crate::celt::rate::clt_compute_allocation;
use crate::celt::vq::renormalise_vector;

use crate::opus::opus_defines::{OPUS_BAD_ARG, OPUS_INTERNAL_ERROR};

pub use self::arch_h::{
    celt_norm, celt_sig, opus_val16, opus_val32, CELT_SIG_SCALE, Q15ONE, VERY_SMALL,
};

pub mod arch_h {
    pub type opus_val16 = f32;
    pub type opus_val32 = f32;
    pub type celt_sig = f32;
    pub type celt_norm = f32;
    pub const Q15ONE: f32 = 1.0f32;
    pub const VERY_SMALL: f32 = 1e-30f32;
    pub const CELT_SIG_SCALE: f32 = 32768.0f32;
}

#[derive(Copy, Clone)]
#[repr(C)]
pub struct OpusCustomDecoder {
    // TODO: a lot of the stuff from the mode should become constants
    // we only have one "opus custom mode" after all
    pub mode: &'static OpusCustomMode,
    pub overlap: usize,
    pub channels: usize,
    pub stream_channels: usize,
    pub downsample: i32,
    pub start: i32,
    pub end: i32,
    pub signalling: i32,
    pub disable_inv: i32,
    pub complexity: i32,
    pub arch: i32,
    pub rng: u32,
    pub error: i32,
    pub last_pitch_index: i32,
    pub loss_duration: i32,
    pub plc_duration: i32,
    pub last_frame_type: i32,
    pub skip_plc: i32,
    pub postfilter_period: i32,
    pub postfilter_period_old: i32,
    pub postfilter_gain: f32,
    pub postfilter_gain_old: f32,
    pub postfilter_tapset: i32,
    pub postfilter_tapset_old: i32,
    pub prefilter_and_fold: i32,
    pub preemph_memD: [celt_sig; 2],

    pub decode_mem: [f32; 2 * (DECODE_BUFFER_SIZE + 120)], /* Size = channels*(DECODE_BUFFER_SIZE+mode->overlap) */
    pub lpc: [f32; 2 * LPC_ORDER],                         /* Size = channels*LPC_ORDER */
    pub oldEBands: [f32; 2 * 21],                          /* Size = 2*mode->nbEBands */
    pub oldLogE: [f32; 2 * 21],                            /* Size = 2*mode->nbEBands */
    pub oldLogE2: [f32; 2 * 21],                           /* Size = 2*mode->nbEBands */
    pub backgroundLogE: [f32; 2 * 21],                     /* Size = 2*mode->nbEBands */
}

pub const PLC_PITCH_LAG_MAX: i32 = 720;
pub const PLC_PITCH_LAG_MIN: i32 = 100;
pub const DECODE_BUFFER_SIZE: usize = 2048;

const FRAME_NONE: i32 = 0;
const FRAME_NORMAL: i32 = 1;
const FRAME_PLC_NOISE: i32 = 2;
const FRAME_PLC_PERIODIC: i32 = 3;
#[cfg(feature = "deep-plc")]
const FRAME_PLC_NEURAL: i32 = 4;
#[cfg(feature = "dred")]
const FRAME_DRED: i32 = 5;
pub fn validate_celt_decoder(st: &OpusCustomDecoder) {
    assert_eq!(st.mode, opus_custom_mode_create(48000, 960, None).unwrap());
    assert_eq!(st.overlap, 120);
    assert!(st.channels == 1 || st.channels == 2);
    assert!(st.stream_channels == 1 || st.stream_channels == 2);
    assert!(st.downsample > 0);
    assert!(st.start == 0 || st.start == 17);
    assert!(st.start < st.end);
    assert!(st.end <= 21);
    assert!(st.arch >= 0);
    assert!(st.arch <= 0);
    assert!(st.last_pitch_index <= 720);
    assert!(st.last_pitch_index >= 100 || st.last_pitch_index == 0);
    assert!(st.postfilter_period < 1024);
    assert!(st.postfilter_period >= 15 || st.postfilter_period == 0);
    assert!(st.postfilter_period_old < 1024);
    assert!(st.postfilter_period_old >= 15 || st.postfilter_period_old == 0);
    assert!(st.postfilter_tapset <= 2);
    assert!(st.postfilter_tapset >= 0);
    assert!(st.postfilter_tapset_old <= 2);
    assert!(st.postfilter_tapset_old >= 0);
}
pub fn celt_decoder_init(sampling_rate: i32, channels: usize) -> OpusCustomDecoder {
    let mode = opus_custom_mode_create(48000, 960, None).unwrap();
    let mut st = opus_custom_decoder_init(mode, channels);
    st.downsample = resampling_factor(sampling_rate);
    if st.downsample == 0 {
        panic!("Unsupported sampling rate: {}", sampling_rate);
    }

    st
}
#[inline]
fn opus_custom_decoder_init(mode: &'static OpusCustomMode, channels: usize) -> OpusCustomDecoder {
    if channels > 2 {
        panic!(
            "Invalid channel count: {}, want either 0 (??), 1 or 2",
            channels
        );
    }
    let mut st = OpusCustomDecoder {
        mode,
        overlap: mode.overlap,
        channels,
        stream_channels: channels,
        downsample: 1,
        start: 0,
        end: mode.effEBands,
        signalling: 1,
        disable_inv: (channels == 1) as i32,
        complexity: 0,
        arch: 0,

        rng: 0,
        error: 0,
        last_pitch_index: 0,
        loss_duration: 0,
        plc_duration: 0,
        last_frame_type: FRAME_NONE,
        skip_plc: 0,
        postfilter_period: 0,
        postfilter_period_old: 0,
        postfilter_gain: 0.0,
        postfilter_gain_old: 0.0,
        postfilter_tapset: 0,
        postfilter_tapset_old: 0,
        prefilter_and_fold: 0,
        preemph_memD: [0.0; 2],

        decode_mem: [0.0; 2 * (DECODE_BUFFER_SIZE + 120)],
        lpc: [0.0; 2 * LPC_ORDER],
        oldEBands: [0.0; 2 * 21],
        oldLogE: [0.0; 2 * 21],
        oldLogE2: [0.0; 2 * 21],
        backgroundLogE: [0.0; 2 * 21],
    };

    st.reset();

    st
}

impl OpusCustomDecoder {
    /// Reset the decoder state to initial defaults.
    ///
    /// Zeros all transient state fields (rng, error, postfilter, decode memory,
    /// LPC, band energies, etc.) while preserving configuration fields (mode,
    /// channels, overlap, downsample, start, end, signalling, disable_inv, arch).
    pub fn reset(&mut self) {
        self.rng = 0;
        self.error = 0;
        self.last_pitch_index = 0;
        self.loss_duration = 0;
        self.plc_duration = 0;
        self.last_frame_type = FRAME_NONE;
        self.skip_plc = 1;
        self.postfilter_period = 0;
        self.postfilter_period_old = 0;
        self.postfilter_gain = 0.0;
        self.postfilter_gain_old = 0.0;
        self.postfilter_tapset = 0;
        self.postfilter_tapset_old = 0;
        self.prefilter_and_fold = 0;
        self.preemph_memD = [0.0; 2];
        self.decode_mem.fill(0.0);
        self.lpc.fill(0.0);
        self.oldEBands.fill(0.0);
        self.oldLogE.fill(-28.0);
        self.oldLogE2.fill(-28.0);
        self.backgroundLogE.fill(0.0);
    }
}

/// Upstream C: celt/celt_decoder.c:deemphasis_stereo_simple
#[inline]
fn deemphasis_stereo_simple(
    ch0: &[celt_sig],
    ch1: &[celt_sig],
    pcm: &mut [opus_val16],
    N: i32,
    coef0: opus_val16,
    mem: &mut [celt_sig; 2],
) {
    let mut m0: celt_sig = mem[0];
    let mut m1: celt_sig = mem[1];
    let mut j = 0;
    while j < N {
        let ju = j as usize;
        let tmp0: celt_sig = ch0[ju] + VERY_SMALL + m0;
        let tmp1: celt_sig = ch1[ju] + VERY_SMALL + m1;
        m0 = coef0 * tmp0;
        m1 = coef0 * tmp1;
        pcm[2 * ju] = tmp0 * (1_f32 / CELT_SIG_SCALE);
        pcm[2 * ju + 1] = tmp1 * (1_f32 / CELT_SIG_SCALE);
        j += 1;
    }
    mem[0] = m0;
    mem[1] = m1;
}
/// Upstream C: celt/celt_decoder.c:deemphasis
#[inline]
fn deemphasis(
    in_channels: &[&[celt_sig]],
    pcm: &mut [opus_val16],
    N: i32,
    C: i32,
    downsample: i32,
    coef: &[opus_val16],
    mem: &mut [celt_sig],
    accum: i32,
) {
    let mut apply_downsampling: i32 = 0;
    if downsample == 1 && C == 2 && accum == 0 {
        deemphasis_stereo_simple(
            in_channels[0],
            in_channels[1],
            pcm,
            N,
            coef[0],
            mem.try_into().unwrap(),
        );
        return;
    }
    assert!(accum == 0);
    let mut scratch = [0.0f32; 960];
    let coef0: opus_val16 = coef[0];
    let Nd: i32 = N / downsample;
    let mut c = 0;
    loop {
        let mut j: i32;
        let mut m: celt_sig = mem[c as usize];
        let x = in_channels[c as usize];
        if downsample > 1 {
            j = 0;
            while j < N {
                let tmp: celt_sig = x[j as usize] + VERY_SMALL + m;
                m = coef0 * tmp;
                scratch[j as usize] = tmp;
                j += 1;
            }
            apply_downsampling = 1;
        } else {
            j = 0;
            while j < N {
                let tmp_0: celt_sig = x[j as usize] + VERY_SMALL + m;
                m = coef0 * tmp_0;
                pcm[(c + j * C) as usize] = tmp_0 * (1_f32 / CELT_SIG_SCALE);
                j += 1;
            }
        }
        mem[c as usize] = m;
        if apply_downsampling != 0 {
            j = 0;
            while j < Nd {
                pcm[(c + j * C) as usize] =
                    scratch[(j * downsample) as usize] * (1_f32 / CELT_SIG_SCALE);
                j += 1;
            }
        }
        c += 1;
        if c >= C {
            break;
        }
    }
}
/// Upstream C: celt/celt_decoder.c:celt_synthesis
#[inline]
fn celt_synthesis(
    mode: &OpusCustomMode,
    X: &[celt_norm],
    out_syn_ch0: &mut [celt_sig],
    out_syn_ch1: &mut [celt_sig],
    oldBandE: &[opus_val16],
    start: i32,
    effEnd: i32,
    C: i32,
    CC: i32,
    isTransient: i32,
    LM: i32,
    downsample: i32,
    silence: i32,
    _arch: i32,
) {
    let mut b: i32;
    let B: i32;
    let NB: i32;
    let shift: i32;
    let overlap = mode.overlap as i32;
    let nbEBands = mode.nbEBands as i32;
    let N = mode.shortMdctSize << LM;
    let n = N as usize;
    let M: i32 = (1) << LM;
    // Allocate N + M - 1 elements so that strided mdct_backward calls
    // can form slices freq[b..b + n2*B] for b in 0..B without going
    // out of bounds. The extra elements are never read (stride skips them).
    let mut freq = [0.0f32; 960 + 8 - 1];
    if isTransient != 0 {
        B = M;
        NB = mode.shortMdctSize;
        shift = mode.maxLM;
    } else {
        B = 1;
        NB = mode.shortMdctSize << LM;
        shift = mode.maxLM - LM;
    }
    let mdct_sub_len = (mode.mdct.n >> shift as usize) / 2;
    let overlap_u = overlap as usize;
    if CC == 2 && C == 1 {
        denormalise_bands(
            mode,
            &X[..(C * N) as usize],
            &mut freq,
            &oldBandE[..(C * nbEBands) as usize],
            start,
            effEnd,
            M,
            downsample,
            silence,
        );
        // Use a temporary array for freq2 instead of borrowing out_syn_ch1
        let mut freq2 = [0.0f32; 960 + 8 - 1];
        freq2[..n].copy_from_slice(&freq[..n]);
        b = 0;
        while b < B {
            let bu = b as usize;
            mdct_backward(
                &mode.mdct,
                &freq2[bu..bu + mdct_sub_len * B as usize],
                &mut out_syn_ch0[NB as usize * bu..NB as usize * bu + mdct_sub_len + overlap_u],
                mode.window,
                overlap_u,
                shift as usize,
                B as usize,
            );
            b += 1;
        }
        b = 0;
        while b < B {
            let bu = b as usize;
            mdct_backward(
                &mode.mdct,
                &freq[bu..bu + mdct_sub_len * B as usize],
                &mut out_syn_ch1[NB as usize * bu..NB as usize * bu + mdct_sub_len + overlap_u],
                mode.window,
                overlap_u,
                shift as usize,
                B as usize,
            );
            b += 1;
        }
    } else if CC == 1 && C == 2 {
        denormalise_bands(
            mode,
            &X[..(C * N) as usize],
            &mut freq,
            &oldBandE[..(C * nbEBands) as usize],
            start,
            effEnd,
            M,
            downsample,
            silence,
        );
        // freq2 for the second channel
        let mut freq2 = [0.0f32; 960];
        denormalise_bands(
            mode,
            &X[n..2 * n],
            &mut freq2,
            &oldBandE[nbEBands as usize..2 * nbEBands as usize],
            start,
            effEnd,
            M,
            downsample,
            silence,
        );
        let mut i = 0;
        while i < n {
            freq[i] = 0.5f32 * freq[i] + 0.5f32 * freq2[i];
            i += 1;
        }
        b = 0;
        while b < B {
            let bu = b as usize;
            mdct_backward(
                &mode.mdct,
                &freq[bu..bu + mdct_sub_len * B as usize],
                &mut out_syn_ch0[NB as usize * bu..NB as usize * bu + mdct_sub_len + overlap_u],
                mode.window,
                overlap_u,
                shift as usize,
                B as usize,
            );
            b += 1;
        }
    } else {
        // CC==C case (mono or stereo matching)
        // Process channel 0
        denormalise_bands(
            mode,
            &X[..n],
            &mut freq,
            &oldBandE[..nbEBands as usize],
            start,
            effEnd,
            M,
            downsample,
            silence,
        );
        b = 0;
        while b < B {
            let bu = b as usize;
            mdct_backward(
                &mode.mdct,
                &freq[bu..bu + mdct_sub_len * B as usize],
                &mut out_syn_ch0[NB as usize * bu..NB as usize * bu + mdct_sub_len + overlap_u],
                mode.window,
                overlap_u,
                shift as usize,
                B as usize,
            );
            b += 1;
        }
        // Process channel 1 (if stereo)
        if CC >= 2 {
            denormalise_bands(
                mode,
                &X[n..2 * n],
                &mut freq,
                &oldBandE[nbEBands as usize..2 * nbEBands as usize],
                start,
                effEnd,
                M,
                downsample,
                silence,
            );
            b = 0;
            while b < B {
                let bu = b as usize;
                mdct_backward(
                    &mode.mdct,
                    &freq[bu..bu + mdct_sub_len * B as usize],
                    &mut out_syn_ch1[NB as usize * bu..NB as usize * bu + mdct_sub_len + overlap_u],
                    mode.window,
                    overlap_u,
                    shift as usize,
                    B as usize,
                );
                b += 1;
            }
        }
    }
    // Note: removed no-op loop (out_syn[c][i] = out_syn[c][i]) that was a c2rust artifact
}
/// Upstream C: celt/celt_decoder.c:tf_decode
fn tf_decode(
    start: i32,
    end: i32,
    isTransient: i32,
    tf_res: &mut [i32],
    LM: i32,
    dec: &mut ec_dec,
) {
    let mut curr: i32;
    let mut tf_select: i32;
    let mut tf_changed: i32;
    let mut logp: i32;
    let mut budget: u32 = dec.storage.wrapping_mul(8);
    let mut tell: u32 = ec_tell(dec) as u32;
    logp = if isTransient != 0 { 2 } else { 4 };
    let tf_select_rsv: i32 =
        (LM > 0 && tell.wrapping_add(logp as u32).wrapping_add(1) <= budget) as i32;
    budget = budget.wrapping_sub(tf_select_rsv as u32);
    curr = 0;
    tf_changed = curr;
    let mut i = start;
    while i < end {
        if tell.wrapping_add(logp as u32) <= budget {
            curr ^= ec_dec_bit_logp(dec, logp as u32);
            tell = ec_tell(dec) as u32;
            tf_changed |= curr;
        }
        tf_res[i as usize] = curr;
        logp = if isTransient != 0 { 4 } else { 5 };
        i += 1;
    }
    tf_select = 0;
    if tf_select_rsv != 0
        && tf_select_table[LM as usize][((4 * isTransient) + tf_changed) as usize] as i32
            != tf_select_table[LM as usize][(4 * isTransient + 2 + tf_changed) as usize] as i32
    {
        tf_select = ec_dec_bit_logp(dec, 1);
    }
    i = start;
    while i < end {
        tf_res[i as usize] = tf_select_table[LM as usize]
            [(4 * isTransient + 2 * tf_select + tf_res[i as usize]) as usize]
            as i32;
        i += 1;
    }
}
/// Upstream C: celt/celt_decoder.c:celt_plc_pitch_search
fn celt_plc_pitch_search(ch0: &[celt_sig], ch1: Option<&[celt_sig]>, _arch: i32) -> i32 {
    let mut lp_pitch_buf: [opus_val16; 1024] = [0.; 1024];
    let ds_len = DECODE_BUFFER_SIZE;
    if let Some(ch1) = ch1 {
        pitch::pitch_downsample(&[&ch0[..ds_len], &ch1[..ds_len]], &mut lp_pitch_buf, ds_len);
    } else {
        pitch::pitch_downsample(&[&ch0[..ds_len]], &mut lp_pitch_buf, ds_len);
    }
    let mut pitch_index = pitch::pitch_search(
        &lp_pitch_buf[(PLC_PITCH_LAG_MAX >> 1) as usize..],
        &lp_pitch_buf,
        DECODE_BUFFER_SIZE as i32 - PLC_PITCH_LAG_MAX,
        PLC_PITCH_LAG_MAX - PLC_PITCH_LAG_MIN,
    );
    pitch_index = PLC_PITCH_LAG_MAX - pitch_index;
    pitch_index
}
/// Upstream C: celt/celt_decoder.c:prefilter_and_fold
fn prefilter_and_fold(st: &mut OpusCustomDecoder, N: i32) {
    let CC = st.channels as i32;
    let mode = st.mode;
    let overlap = mode.overlap as i32;
    let overlap_u = overlap as usize;
    let chan_stride = DECODE_BUFFER_SIZE + overlap_u;
    let n = N as usize;

    let mut c = 0;
    loop {
        let ch_off = c as usize * chan_stride;
        let mut etmp = [0.0f32; 120];

        // Apply the pre-filter to the MDCT overlap for the next frame because
        // the post-filter will be re-applied in the decoder after the MDCT overlap.
        comb_filter(
            &mut etmp,
            0,
            &st.decode_mem[ch_off..ch_off + chan_stride],
            DECODE_BUFFER_SIZE - n,
            st.postfilter_period_old,
            st.postfilter_period,
            overlap,
            -st.postfilter_gain_old,
            -st.postfilter_gain,
            st.postfilter_tapset_old,
            st.postfilter_tapset,
            &[],
            0,
            st.arch,
        );

        // Simulate TDAC on the concealed audio so that it blends with the
        // MDCT of the next frame.
        for i in 0..overlap_u / 2 {
            st.decode_mem[ch_off + DECODE_BUFFER_SIZE - n + i] =
                mode.window[i] * etmp[overlap_u - 1 - i] + mode.window[overlap_u - i - 1] * etmp[i];
        }

        c += 1;
        if c >= CC {
            break;
        }
    }
}

/// Upstream C: celt/celt_decoder.c:celt_decode_lost
#[inline]
fn celt_decode_lost(
    st: &mut OpusCustomDecoder,
    N: i32,
    LM: i32,
    #[cfg(feature = "deep-plc")] lpcnet: Option<&mut crate::dnn::lpcnet::LPCNetPLCState>,
) {
    let C: i32 = st.channels as i32;
    let mode = st.mode;
    let nbEBands = mode.nbEBands as i32;
    let overlap = mode.overlap as i32;
    let overlap_u = overlap as usize;
    let eBands = &mode.eBands;
    let chan_stride = DECODE_BUFFER_SIZE + overlap_u;
    let n = N as usize;

    let loss_duration = st.loss_duration;
    let start = st.start;
    let mut curr_frame_type = FRAME_PLC_PERIODIC;
    if st.plc_duration >= 40 || start != 0 || st.skip_plc != 0 {
        curr_frame_type = FRAME_PLC_NOISE;
    }
    #[cfg(feature = "deep-plc")]
    if start == 0 {
        if let Some(ref lpcnet) = lpcnet {
            if lpcnet.loaded {
                if st.complexity >= 5 && st.plc_duration < 80 && st.skip_plc == 0 {
                    curr_frame_type = FRAME_PLC_NEURAL;
                }
                #[cfg(feature = "dred")]
                if lpcnet.fec_fill_pos > lpcnet.fec_read_pos {
                    curr_frame_type = FRAME_DRED;
                }
            }
        }
    }
    if curr_frame_type == FRAME_PLC_NOISE {
        let end = st.end;
        let effEnd = if start
            > (if end < mode.effEBands {
                end
            } else {
                mode.effEBands
            }) {
            start
        } else if end < mode.effEBands {
            end
        } else {
            mode.effEBands
        };
        let mut X: Vec<celt_norm> = ::std::vec::from_elem(0., (C * N) as usize);
        // Shift decode_mem for each channel (before energy decay)
        let mut c = 0;
        loop {
            let ch_off = c as usize * chan_stride;
            let shift_len = 2048 - n + overlap_u;
            st.decode_mem
                .copy_within(ch_off + n..ch_off + n + shift_len, ch_off);
            c += 1;
            if c >= C {
                break;
            }
        }

        if st.prefilter_and_fold != 0 {
            prefilter_and_fold(st, N);
        }

        let decay: opus_val16 = if loss_duration == 0 { 1.5f32 } else { 0.5f32 };
        c = 0;
        loop {
            let mut i = start;
            while i < end {
                let idx = (c * nbEBands + i) as usize;
                st.oldEBands[idx] = if st.backgroundLogE[idx] > st.oldEBands[idx] - decay {
                    st.backgroundLogE[idx]
                } else {
                    st.oldEBands[idx] - decay
                };
                i += 1;
            }
            c += 1;
            if c >= C {
                break;
            }
        }
        let mut seed = st.rng;
        c = 0;
        while c < C {
            let mut i = start;
            while i < effEnd {
                let boffs = (N * c + ((eBands[i as usize] as i32) << LM)) as usize;
                let blen =
                    ((eBands[(i + 1) as usize] as i32 - eBands[i as usize] as i32) << LM) as usize;
                let mut j = 0;
                while j < blen {
                    seed = celt_lcg_rand(seed);
                    X[boffs + j] = (seed as i32 >> 20) as celt_norm;
                    j += 1;
                }
                renormalise_vector(&mut X[boffs..boffs + blen], blen as i32, Q15ONE, st.arch);
                i += 1;
            }
            c += 1;
        }
        st.rng = seed;
        {
            let out_syn_off = DECODE_BUFFER_SIZE - n;
            let (ch0, ch1_region) = st.decode_mem.split_at_mut(chan_stride);
            celt_synthesis(
                mode,
                &X,
                &mut ch0[out_syn_off..],
                if C >= 2 {
                    &mut ch1_region[out_syn_off..chan_stride]
                } else {
                    &mut []
                },
                &st.oldEBands[..(C * nbEBands) as usize],
                start,
                effEnd,
                C,
                C,
                0,
                LM,
                st.downsample,
                0,
                st.arch,
            );
        }
        // Run the postfilter with the last parameters
        {
            let mut c = 0;
            loop {
                st.postfilter_period = st.postfilter_period.max(COMBFILTER_MINPERIOD);
                st.postfilter_period_old = st.postfilter_period_old.max(COMBFILTER_MINPERIOD);
                let ch_off = c as usize * chan_stride;
                let out_syn_off = DECODE_BUFFER_SIZE - n;
                let dm_slice = &mut st.decode_mem[ch_off..ch_off + chan_stride];
                comb_filter_inplace(
                    dm_slice,
                    out_syn_off,
                    st.postfilter_period_old,
                    st.postfilter_period,
                    mode.shortMdctSize,
                    st.postfilter_gain_old,
                    st.postfilter_gain,
                    st.postfilter_tapset_old,
                    st.postfilter_tapset,
                    &mode.window[..overlap as usize],
                    overlap,
                    st.arch,
                );
                if LM != 0 {
                    comb_filter_inplace(
                        dm_slice,
                        out_syn_off + mode.shortMdctSize as usize,
                        st.postfilter_period,
                        st.postfilter_period,
                        N - mode.shortMdctSize,
                        st.postfilter_gain,
                        st.postfilter_gain,
                        st.postfilter_tapset,
                        st.postfilter_tapset,
                        &mode.window[..overlap as usize],
                        overlap,
                        st.arch,
                    );
                }
                c += 1;
                if c >= C {
                    break;
                }
            }
            st.postfilter_period_old = st.postfilter_period;
            st.postfilter_gain_old = st.postfilter_gain;
            st.postfilter_tapset_old = st.postfilter_tapset;
        }
        st.prefilter_and_fold = 0;
        // Skip regular PLC until we get two consecutive packets.
        st.skip_plc = 1;
    } else {
        let mut fade: opus_val16 = Q15ONE;
        let pitch_index: i32;
        #[cfg(feature = "deep-plc")]
        let curr_neural = curr_frame_type == FRAME_PLC_NEURAL || curr_frame_type == FRAME_DRED;
        #[cfg(feature = "deep-plc")]
        let last_neural =
            st.last_frame_type == FRAME_PLC_NEURAL || st.last_frame_type == FRAME_DRED;
        let need_pitch_search = {
            #[cfg(feature = "deep-plc")]
            {
                st.last_frame_type != FRAME_PLC_PERIODIC && !(last_neural && curr_neural)
            }
            #[cfg(not(feature = "deep-plc"))]
            {
                st.last_frame_type != FRAME_PLC_PERIODIC
            }
        };
        if need_pitch_search {
            let (ch0, ch1_region) = st.decode_mem.split_at_mut(chan_stride);
            pitch_index = celt_plc_pitch_search(
                &ch0[..DECODE_BUFFER_SIZE],
                if C == 2 {
                    Some(&ch1_region[..DECODE_BUFFER_SIZE])
                } else {
                    None
                },
                st.arch,
            );
            st.last_pitch_index = pitch_index;
        } else {
            pitch_index = st.last_pitch_index;
            fade = 0.8f32;
        }
        let exc_length: i32 = if 2 * pitch_index < 1024 {
            2 * pitch_index
        } else {
            1024
        };
        let mut _exc: [opus_val16; 1048] = [0.; 1048];
        let mut fir_tmp: Vec<opus_val16> = ::std::vec::from_elem(0., exc_length as usize);
        // exc = _exc[LPC_ORDER..], so exc[i] = _exc[LPC_ORDER + i]
        let exc_off = LPC_ORDER;
        let window = mode.window;
        let mut c = 0;
        loop {
            let ch_off = c as usize * chan_stride;
            // Copy exc data from decode_mem channel
            {
                let mut i = 0;
                while i < MAX_PERIOD as usize + LPC_ORDER {
                    _exc[i] = st.decode_mem[ch_off + 2048 - 1024 - LPC_ORDER + i];
                    i += 1;
                }
            }
            if need_pitch_search {
                let mut ac: [opus_val32; 25] = [0.; 25];
                _celt_autocorr(
                    &_exc[exc_off..exc_off + MAX_PERIOD as usize],
                    &mut ac,
                    Some(&window[..overlap_u]),
                    overlap_u,
                    LPC_ORDER,
                );
                ac[0] *= 1.0001f32;
                let mut i = 1i32;
                while i <= LPC_ORDER as i32 {
                    ac[i as usize] -= ac[i as usize] * (0.008f32 * 0.008f32) * i as f32 * i as f32;
                    i += 1;
                }
                let lpc_start = c as usize * LPC_ORDER;
                _celt_lpc(&mut st.lpc[lpc_start..lpc_start + LPC_ORDER], &ac);
            }
            {
                let fir_n = exc_length as usize;
                let fir_start = 1024 - fir_n; // index into _exc
                let fir_x = &_exc[fir_start..fir_start + LPC_ORDER + fir_n];
                let lpc_start = c as usize * LPC_ORDER;
                celt_fir_c(
                    fir_x,
                    &st.lpc[lpc_start..lpc_start + LPC_ORDER],
                    &mut fir_tmp,
                    LPC_ORDER,
                );
            }
            // Copy filtered result back to exc buffer
            {
                let dst_start = exc_off + 1024 - exc_length as usize;
                _exc[dst_start..dst_start + exc_length as usize]
                    .copy_from_slice(&fir_tmp[..exc_length as usize]);
            }
            let mut E1: opus_val32 = 1.0;
            let mut E2: opus_val32 = 1.0;
            let decay_length = exc_length >> 1;
            {
                let mut i = 0;
                while i < decay_length {
                    let e = _exc[exc_off + (MAX_PERIOD - decay_length + i) as usize];
                    E1 += e * e;
                    let e = _exc[exc_off + (MAX_PERIOD - 2 * decay_length + i) as usize];
                    E2 += e * e;
                    i += 1;
                }
            }
            E1 = if E1 < E2 { E1 } else { E2 };
            let decay_0: opus_val16 = celt_sqrt(E1 / E2);
            // Shift decode_mem: memmove(buf, buf+N, (2048-N)*sizeof)
            st.decode_mem.copy_within(ch_off + n..ch_off + 2048, ch_off);
            let extrapolation_offset = MAX_PERIOD - pitch_index;
            let extrapolation_len = N + overlap;
            let mut attenuation: opus_val16 = fade * decay_0;
            let mut j_0 = 0i32;
            let mut i = j_0;
            let mut S1: opus_val32 = 0.0;
            while i < extrapolation_len {
                if j_0 >= pitch_index {
                    j_0 -= pitch_index;
                    attenuation *= decay_0;
                }
                st.decode_mem[ch_off + DECODE_BUFFER_SIZE - n + i as usize] =
                    attenuation * _exc[exc_off + (extrapolation_offset + j_0) as usize];
                let tmp =
                    st.decode_mem[ch_off + 2048 - 1024 - n + (extrapolation_offset + j_0) as usize];
                S1 += tmp * tmp;
                i += 1;
                j_0 += 1;
            }
            let mut lpc_mem: [opus_val16; 24] = [0.; 24];
            {
                let mut i = 0;
                while i < LPC_ORDER {
                    lpc_mem[i] = st.decode_mem[ch_off + 2048 - n - 1 - i];
                    i += 1;
                }
            }
            {
                let lpc_start = c as usize * LPC_ORDER;
                let iir_start = ch_off + DECODE_BUFFER_SIZE - n;
                let iir_buf = &mut st.decode_mem[iir_start..iir_start + extrapolation_len as usize];
                celt_iir(
                    iir_buf,
                    extrapolation_len as usize,
                    &st.lpc[lpc_start..lpc_start + LPC_ORDER],
                    LPC_ORDER,
                    &mut lpc_mem,
                );
            }
            let mut S2: opus_val32 = 0.0;
            {
                let mut i = 0;
                while i < extrapolation_len {
                    let tmp_0: opus_val16 = st.decode_mem[ch_off + 2048 - n + i as usize];
                    S2 += tmp_0 * tmp_0;
                    i += 1;
                }
            }
            #[allow(clippy::neg_cmp_op_on_partial_ord)]
            if !(S1 > 0.2f32 * S2) {
                let mut i = 0;
                while i < extrapolation_len {
                    st.decode_mem[ch_off + DECODE_BUFFER_SIZE - n + i as usize] = 0.0;
                    i += 1;
                }
            } else if S1 < S2 {
                let ratio: opus_val16 = celt_sqrt((S1 + 1.0) / (S2 + 1.0));
                let mut i = 0;
                while i < overlap {
                    let tmp_g: opus_val16 = Q15ONE - window[i as usize] * (1.0f32 - ratio);
                    st.decode_mem[ch_off + DECODE_BUFFER_SIZE - n + i as usize] =
                        tmp_g * st.decode_mem[ch_off + 2048 - n + i as usize];
                    i += 1;
                }
                i = overlap;
                while i < extrapolation_len {
                    st.decode_mem[ch_off + DECODE_BUFFER_SIZE - n + i as usize] =
                        ratio * st.decode_mem[ch_off + 2048 - n + i as usize];
                    i += 1;
                }
            }
            c += 1;
            if c >= C {
                break;
            }
        }
        st.prefilter_and_fold = 1;
    }
    st.loss_duration = 10000_i32.min(loss_duration + (1 << LM));
    st.plc_duration = 10000_i32.min(st.plc_duration + (1 << LM));
    #[cfg(feature = "dred")]
    if curr_frame_type == FRAME_DRED {
        st.plc_duration = 0;
        st.skip_plc = 0;
    }
    st.last_frame_type = curr_frame_type;
}
/// Upstream C: celt/celt_decoder.c:celt_decode_with_ec / celt_decode_with_ec_dred
pub fn celt_decode_with_ec(
    st: &mut OpusCustomDecoder,
    data: Option<&[u8]>,
    pcm: &mut [opus_val16],
    mut frame_size: i32,
    dec: Option<&mut ec_dec>,
    accum: i32,
    #[cfg(feature = "deep-plc")] lpcnet: Option<&mut crate::dnn::lpcnet::LPCNetPLCState>,
) -> i32 {
    let CC: i32 = st.channels as i32;
    let C: i32 = st.stream_channels as i32;
    let len: i32 = data.map_or(0, |d| d.len() as i32);
    validate_celt_decoder(&*st);
    let mode = st.mode;
    let nbEBands = mode.nbEBands as i32;
    let overlap = mode.overlap as i32;
    let eBands = &mode.eBands;
    let start = st.start;
    let end = st.end;
    frame_size *= st.downsample;
    let chan_stride = DECODE_BUFFER_SIZE + overlap as usize;

    let mut LM: i32 = 0;
    while LM <= mode.maxLM {
        if mode.shortMdctSize << LM == frame_size {
            break;
        }
        LM += 1;
    }
    if LM > mode.maxLM {
        return OPUS_BAD_ARG;
    }
    let M: i32 = (1) << LM;
    if !(0..=1275).contains(&len) {
        return OPUS_BAD_ARG;
    }
    let N: i32 = M * mode.shortMdctSize;
    let n = N as usize;
    let out_syn_off = DECODE_BUFFER_SIZE - n;
    let mut effEnd: i32 = end;
    if effEnd > mode.effEBands {
        effEnd = mode.effEBands;
    }
    if data.is_none() || len <= 1 {
        celt_decode_lost(
            st,
            N,
            LM,
            #[cfg(feature = "deep-plc")]
            lpcnet,
        );
        {
            let in_ch: Vec<&[celt_sig]> = (0..CC as usize)
                .map(|c| {
                    &st.decode_mem[c * chan_stride + out_syn_off..c * chan_stride + out_syn_off + n]
                })
                .collect();
            let pcm_len = (frame_size / st.downsample * CC) as usize;
            deemphasis(
                &in_ch,
                &mut pcm[..pcm_len],
                N,
                CC,
                st.downsample,
                &mode.preemph,
                &mut st.preemph_memD,
                accum,
            );
        }
        return frame_size / st.downsample;
    }
    if st.loss_duration == 0 {
        st.skip_plc = 0;
    }
    // Copy data into a stack buffer so ec_dec_init can take &mut [u8] without
    // a const-to-mut cast. Max 1275 bytes per validation above.
    let data_slice = data.unwrap();
    let mut data_copy = [0u8; 1275];
    data_copy[..data_slice.len()].copy_from_slice(data_slice);
    // When the caller provides a dec, use it; otherwise create a local one.
    // These are separate scopes to avoid lifetime unification between the
    // caller-provided ec_dec and the locally-owned one (self-referential borrow).
    if let Some(dec) = dec {
        return celt_decode_body(
            st,
            pcm,
            frame_size,
            dec,
            accum,
            C,
            CC,
            len,
            N,
            n,
            LM,
            M,
            start,
            end,
            effEnd,
            nbEBands,
            overlap,
            mode,
            eBands,
            out_syn_off,
            chan_stride,
        );
    }
    let mut _dec = ec_dec_init(&mut data_copy[..data_slice.len()]);
    celt_decode_body(
        st,
        pcm,
        frame_size,
        &mut _dec,
        accum,
        C,
        CC,
        len,
        N,
        n,
        LM,
        M,
        start,
        end,
        effEnd,
        nbEBands,
        overlap,
        mode,
        eBands,
        out_syn_off,
        chan_stride,
    )
}

#[allow(clippy::too_many_arguments)]
#[inline]
fn celt_decode_body(
    st: &mut OpusCustomDecoder,
    pcm: &mut [opus_val16],
    frame_size: i32,
    dec: &mut ec_dec,
    accum: i32,
    C: i32,
    CC: i32,
    len: i32,
    N: i32,
    n: usize,
    LM: i32,
    M: i32,
    start: i32,
    end: i32,
    effEnd: i32,
    nbEBands: i32,
    overlap: i32,
    mode: &'static OpusCustomMode,
    eBands: &[i16],
    out_syn_off: usize,
    chan_stride: usize,
) -> i32 {
    let mut c: i32;
    let mut i: i32;
    let mut spread_decision: i32;
    let mut bits: i32;

    let mut postfilter_pitch: i32;
    let mut postfilter_gain: opus_val16;
    let mut intensity: i32 = 0;
    let mut dual_stereo: i32 = 0;
    let mut total_bits: i32;
    let mut balance: i32 = 0;
    let mut tell: i32;
    let mut dynalloc_logp: i32;
    let mut postfilter_tapset: i32;
    let mut anti_collapse_on: i32 = 0;
    if C == 1 {
        i = 0;
        let nb = nbEBands as usize;
        while i < nbEBands {
            st.oldEBands[i as usize] = if st.oldEBands[i as usize] > st.oldEBands[nb + i as usize] {
                st.oldEBands[i as usize]
            } else {
                st.oldEBands[nb + i as usize]
            };
            i += 1;
        }
    }
    total_bits = len * 8;
    tell = ec_tell(dec);
    let silence: i32 = if tell >= total_bits {
        1
    } else if tell == 1 {
        ec_dec_bit_logp(dec, 15)
    } else {
        0
    };
    if silence != 0 {
        tell = len * 8;
        dec.nbits_total += tell - ec_tell(dec);
    }
    postfilter_gain = 0 as opus_val16;
    postfilter_pitch = 0;
    postfilter_tapset = 0;
    if start == 0 && tell + 16 <= total_bits {
        if ec_dec_bit_logp(dec, 1) != 0 {
            let mut qg: i32 = 0;
            let mut octave: i32 = 0;
            octave = ec_dec_uint(dec, 6) as i32;
            postfilter_pitch = (((16) << octave) as u32)
                .wrapping_add(ec_dec_bits(dec, (4 + octave) as u32))
                .wrapping_sub(1) as i32;
            qg = ec_dec_bits(dec, 3) as i32;
            if ec_tell(dec) + 2 <= total_bits {
                postfilter_tapset = ec_dec_icdf(dec, &tapset_icdf, 2);
            }
            postfilter_gain = 0.09375f32 * (qg + 1) as f32;
        }
        tell = ec_tell(dec);
    }
    let isTransient: i32 = if LM > 0 && tell + 3 <= total_bits {
        let v = ec_dec_bit_logp(dec, 3);
        tell = ec_tell(dec);
        v
    } else {
        0
    };
    let shortBlocks: i32 = if isTransient != 0 { M } else { 0 };
    let intra_ener: i32 = if tell + 3 <= total_bits {
        ec_dec_bit_logp(dec, 3)
    } else {
        0
    };
    // If recovering from packet loss, make sure we make the energy prediction safe to reduce the
    // risk of getting loud artifacts.
    if intra_ener == 0 && st.loss_duration != 0 {
        c = 0;
        loop {
            let safety: opus_val16 = if LM == 0 {
                1.5f32
            } else if LM == 1 {
                0.5f32
            } else {
                0.0f32
            };
            let missing = 10i32.min(st.loss_duration >> LM);
            i = start;
            while i < end {
                let idx = (c * nbEBands + i) as usize;
                if st.oldEBands[idx]
                    < if st.oldLogE[idx] > st.oldLogE2[idx] {
                        st.oldLogE[idx]
                    } else {
                        st.oldLogE2[idx]
                    }
                {
                    // If energy is going down already, continue the trend.
                    let e0 = st.oldEBands[idx];
                    let e1 = st.oldLogE[idx];
                    let e2 = st.oldLogE2[idx];
                    let slope = if e1 - e0 > 0.5f32 * (e2 - e0) {
                        e1 - e0
                    } else {
                        0.5f32 * (e2 - e0)
                    };
                    let new_e = e0
                        - (if 0.0f32 > (1 + missing) as f32 * slope {
                            0.0f32
                        } else {
                            (1 + missing) as f32 * slope
                        });
                    st.oldEBands[idx] = if -20.0f32 > new_e { -20.0f32 } else { new_e };
                } else {
                    // Otherwise take the min of the last frames.
                    st.oldEBands[idx] =
                        st.oldEBands[idx].min(st.oldLogE[idx]).min(st.oldLogE2[idx]);
                }
                // Shorter frames have more natural fluctuations -- play it safe.
                st.oldEBands[idx] -= safety;
                i += 1;
            }
            c += 1;
            if c >= 2 {
                break;
            }
        }
    }
    unquant_coarse_energy(
        mode,
        start,
        end,
        &mut st.oldEBands[..(C * nbEBands) as usize],
        intra_ener,
        dec,
        C,
        LM,
    );
    let mut tf_res = [0i32; 21];
    tf_decode(start, end, isTransient, &mut tf_res, LM, dec);
    tell = ec_tell(dec);
    spread_decision = SPREAD_NORMAL;
    if tell + 4 <= total_bits {
        spread_decision = ec_dec_icdf(dec, &spread_icdf, 5);
    }
    let mut cap = [0i32; 21];
    init_caps(mode, &mut cap, LM, C);
    let mut offsets = [0i32; 21];
    dynalloc_logp = 6;
    total_bits <<= BITRES;
    tell = ec_tell_frac(dec) as i32;
    i = start;
    while i < end {
        let mut width: i32 = 0;
        let mut quanta: i32 = 0;
        let mut dynalloc_loop_logp: i32 = 0;
        let mut boost: i32 = 0;
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
        while tell + (dynalloc_loop_logp << BITRES) < total_bits && boost < cap[i as usize] {
            let mut flag: i32 = 0;
            flag = ec_dec_bit_logp(dec, dynalloc_loop_logp as u32);
            tell = ec_tell_frac(dec) as i32;
            if flag == 0 {
                break;
            }
            boost += quanta;
            total_bits -= quanta;
            dynalloc_loop_logp = 1;
        }
        offsets[i as usize] = boost;
        if boost > 0 {
            dynalloc_logp = if 2 > dynalloc_logp - 1 {
                2
            } else {
                dynalloc_logp - 1
            };
        }
        i += 1;
    }
    let mut fine_quant = [0i32; 21];
    let alloc_trim: i32 = if tell + ((6) << BITRES) <= total_bits {
        ec_dec_icdf(dec, &trim_icdf, 7)
    } else {
        5
    };
    bits = (((len * 8) << BITRES) as u32)
        .wrapping_sub(ec_tell_frac(dec))
        .wrapping_sub(1) as i32;
    let anti_collapse_rsv: i32 = if isTransient != 0 && LM >= 2 && bits >= (LM + 2) << BITRES {
        (1) << BITRES
    } else {
        0
    };
    bits -= anti_collapse_rsv;
    let mut pulses = [0i32; 21];
    let mut fine_priority = [0i32; 21];
    let codedBands: i32 = clt_compute_allocation(
        mode,
        start,
        end,
        &offsets,
        &cap,
        alloc_trim,
        &mut intensity,
        &mut dual_stereo,
        bits,
        &mut balance,
        &mut pulses,
        &mut fine_quant,
        &mut fine_priority,
        C,
        LM,
        dec,
        0,
        0,
        0,
    );
    unquant_fine_energy(
        mode,
        start,
        end,
        &mut st.oldEBands[..(C * nbEBands) as usize],
        &fine_quant,
        dec,
        C,
    );
    c = 0;
    loop {
        let ch_off = c as usize * chan_stride;
        let shift_len = (2048 - N + overlap) as usize;
        st.decode_mem
            .copy_within(ch_off + n..ch_off + n + shift_len, ch_off);
        c += 1;
        if c >= CC {
            break;
        }
    }
    let mut collapse_masks = [0u8; 42];
    let mut X = [0.0f32; 1920];
    if C == 2 {
        let (x_part, y_part) = X.split_at_mut(N as usize);
        quant_all_bands(
            0,
            mode,
            start,
            end,
            x_part,
            Some(y_part),
            &mut collapse_masks,
            &[],
            &mut pulses,
            shortBlocks,
            spread_decision,
            dual_stereo,
            intensity,
            &mut tf_res,
            len * ((8) << BITRES) - anti_collapse_rsv,
            balance,
            dec,
            LM,
            codedBands,
            &mut st.rng,
            0,
            st.arch,
            st.disable_inv,
        );
    } else {
        quant_all_bands(
            0,
            mode,
            start,
            end,
            &mut X,
            None,
            &mut collapse_masks,
            &[],
            &mut pulses,
            shortBlocks,
            spread_decision,
            dual_stereo,
            intensity,
            &mut tf_res,
            len * ((8) << BITRES) - anti_collapse_rsv,
            balance,
            dec,
            LM,
            codedBands,
            &mut st.rng,
            0,
            st.arch,
            st.disable_inv,
        );
    }
    if anti_collapse_rsv > 0 {
        anti_collapse_on = ec_dec_bits(dec, 1) as i32;
    }
    unquant_energy_finalise(
        mode,
        start,
        end,
        &mut st.oldEBands[..(C * nbEBands) as usize],
        &fine_quant,
        &fine_priority,
        len * 8 - ec_tell(dec),
        dec,
        C,
    );
    if anti_collapse_on != 0 {
        anti_collapse(
            mode,
            &mut X,
            &mut collapse_masks,
            LM,
            C,
            N,
            start,
            end,
            &st.oldEBands[..(2 * nbEBands) as usize],
            &st.oldLogE[..(2 * nbEBands) as usize],
            &st.oldLogE2[..(2 * nbEBands) as usize],
            &pulses,
            st.rng,
            0, // encode=0 for decoder
            st.arch,
        );
    }
    if silence != 0 {
        i = 0;
        while i < C * nbEBands {
            st.oldEBands[i as usize] = -28.0f32;
            i += 1;
        }
    }
    if st.prefilter_and_fold != 0 {
        prefilter_and_fold(st, N);
    }
    {
        let out_syn_len = n + overlap as usize;
        let (ch0, ch1_region) = st.decode_mem.split_at_mut(chan_stride);
        celt_synthesis(
            mode,
            &X,
            &mut ch0[out_syn_off..out_syn_off + out_syn_len],
            if CC >= 2 {
                &mut ch1_region[out_syn_off..out_syn_off + out_syn_len]
            } else {
                &mut []
            },
            &st.oldEBands[..(2 * nbEBands) as usize],
            start,
            effEnd,
            C,
            CC,
            isTransient,
            LM,
            st.downsample,
            silence,
            st.arch,
        );
    }
    c = 0;
    loop {
        st.postfilter_period = if st.postfilter_period > 15 {
            st.postfilter_period
        } else {
            15
        };
        st.postfilter_period_old = if st.postfilter_period_old > 15 {
            st.postfilter_period_old
        } else {
            15
        };
        {
            let ch_off = c as usize * chan_stride;
            let dm_slice = &mut st.decode_mem[ch_off..ch_off + chan_stride];
            comb_filter_inplace(
                dm_slice,
                out_syn_off,
                st.postfilter_period_old,
                st.postfilter_period,
                mode.shortMdctSize,
                st.postfilter_gain_old,
                st.postfilter_gain,
                st.postfilter_tapset_old,
                st.postfilter_tapset,
                &mode.window[..overlap as usize],
                overlap,
                st.arch,
            );
            if LM != 0 {
                comb_filter_inplace(
                    dm_slice,
                    out_syn_off + mode.shortMdctSize as usize,
                    st.postfilter_period,
                    postfilter_pitch,
                    N - mode.shortMdctSize,
                    st.postfilter_gain,
                    postfilter_gain,
                    st.postfilter_tapset,
                    postfilter_tapset,
                    &mode.window[..overlap as usize],
                    overlap,
                    st.arch,
                );
            }
        }
        c += 1;
        if c >= CC {
            break;
        }
    }
    st.postfilter_period_old = st.postfilter_period;
    st.postfilter_gain_old = st.postfilter_gain;
    st.postfilter_tapset_old = st.postfilter_tapset;
    st.postfilter_period = postfilter_pitch;
    st.postfilter_gain = postfilter_gain;
    st.postfilter_tapset = postfilter_tapset;
    if LM != 0 {
        st.postfilter_period_old = st.postfilter_period;
        st.postfilter_gain_old = st.postfilter_gain;
        st.postfilter_tapset_old = st.postfilter_tapset;
    }
    let nb = nbEBands as usize;
    if C == 1 {
        st.oldEBands.copy_within(0..nb, nb);
    }
    if isTransient == 0 {
        let nb2 = (2 * nbEBands) as usize;
        st.oldLogE2[..nb2].copy_from_slice(&st.oldLogE[..nb2]);
        st.oldLogE[..nb2].copy_from_slice(&st.oldEBands[..nb2]);
    } else {
        i = 0;
        while i < 2 * nbEBands {
            st.oldLogE[i as usize] = if st.oldLogE[i as usize] < st.oldEBands[i as usize] {
                st.oldLogE[i as usize]
            } else {
                st.oldEBands[i as usize]
            };
            i += 1;
        }
    }
    let max_background_increase: opus_val16 = (160_i32.min(st.loss_duration + M) as f32) * 0.001f32;
    i = 0;
    while i < 2 * nbEBands {
        st.backgroundLogE[i as usize] =
            if st.backgroundLogE[i as usize] + max_background_increase < st.oldEBands[i as usize] {
                st.backgroundLogE[i as usize] + max_background_increase
            } else {
                st.oldEBands[i as usize]
            };
        i += 1;
    }
    c = 0;
    loop {
        i = 0;
        while i < start {
            st.oldEBands[(c * nbEBands + i) as usize] = 0 as opus_val16;
            st.oldLogE2[(c * nbEBands + i) as usize] = -28.0f32;
            st.oldLogE[(c * nbEBands + i) as usize] = -28.0f32;
            i += 1;
        }
        i = end;
        while i < nbEBands {
            st.oldEBands[(c * nbEBands + i) as usize] = 0 as opus_val16;
            st.oldLogE2[(c * nbEBands + i) as usize] = -28.0f32;
            st.oldLogE[(c * nbEBands + i) as usize] = -28.0f32;
            i += 1;
        }
        c += 1;
        if c >= 2 {
            break;
        }
    }
    st.rng = dec.rng;
    {
        let in_ch: Vec<&[celt_sig]> = (0..CC as usize)
            .map(|c| {
                &st.decode_mem[c * chan_stride + out_syn_off..c * chan_stride + out_syn_off + n]
            })
            .collect();
        let pcm_len = (frame_size / st.downsample * CC) as usize;
        deemphasis(
            &in_ch,
            &mut pcm[..pcm_len],
            N,
            CC,
            st.downsample,
            &mode.preemph,
            &mut st.preemph_memD,
            accum,
        );
    }
    st.loss_duration = 0;
    st.plc_duration = 0;
    st.last_frame_type = FRAME_NORMAL;
    st.prefilter_and_fold = 0;
    if ec_tell(dec) > 8 * len {
        return OPUS_INTERNAL_ERROR;
    }
    if ec_get_error(dec) != 0 {
        st.error = 1;
    }
    frame_size / st.downsample
}
