use crate::celt::bands::{
    anti_collapse, celt_lcg_rand, denormalise_bands, quant_all_bands, SPREAD_NORMAL,
};
use crate::celt::celt::{
    comb_filter, comb_filter_inplace, init_caps, resampling_factor, spread_icdf, tapset_icdf,
    tf_select_table, trim_icdf,
};
use crate::celt::celt::{
    CELT_GET_AND_CLEAR_ERROR_REQUEST, CELT_GET_MODE_REQUEST, CELT_SET_CHANNELS_REQUEST,
    CELT_SET_END_BAND_REQUEST, CELT_SET_SIGNALLING_REQUEST, CELT_SET_START_BAND_REQUEST,
};
use crate::celt::celt_lpc::{_celt_autocorr, _celt_lpc, celt_fir_c, celt_iir, LPC_ORDER};
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
use crate::externs::{memcpy, memmove};
use crate::opus_custom_decoder_ctl;
use crate::src::opus_defines::{
    OPUS_BAD_ARG, OPUS_GET_FINAL_RANGE_REQUEST, OPUS_GET_LOOKAHEAD_REQUEST,
    OPUS_GET_PHASE_INVERSION_DISABLED_REQUEST, OPUS_GET_PITCH_REQUEST, OPUS_INTERNAL_ERROR,
    OPUS_OK, OPUS_RESET_STATE, OPUS_SET_PHASE_INVERSION_DISABLED_REQUEST, OPUS_UNIMPLEMENTED,
};
use crate::varargs::VarArgs;

pub use self::arch_h::{
    celt_norm, celt_sig, opus_val16, opus_val32, CELT_SIG_SCALE, Q15ONE, VERY_SMALL,
};
pub use self::stddef_h::NULL;

pub mod arch_h {
    pub type opus_val16 = f32;
    pub type opus_val32 = f32;
    pub type celt_sig = f32;
    pub type celt_norm = f32;
    pub const Q15ONE: f32 = 1.0f32;
    pub const VERY_SMALL: f32 = 1e-30f32;
    pub const CELT_SIG_SCALE: f32 = 32768.0f32;
}

pub mod stddef_h {
    pub const NULL: i32 = 0;
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
    pub arch: i32,
    pub rng: u32,
    pub error: i32,
    pub last_pitch_index: i32,
    pub loss_count: i32,
    pub skip_plc: i32,
    pub postfilter_period: i32,
    pub postfilter_period_old: i32,
    pub postfilter_gain: f32,
    pub postfilter_gain_old: f32,
    pub postfilter_tapset: i32,
    pub postfilter_tapset_old: i32,
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

    return st;
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
        arch: 0,

        rng: 0,
        error: 0,
        last_pitch_index: 0,
        loss_count: 0,
        skip_plc: 0,
        postfilter_period: 0,
        postfilter_period_old: 0,
        postfilter_gain: 0.0,
        postfilter_gain_old: 0.0,
        postfilter_tapset: 0,
        postfilter_tapset_old: 0,
        preemph_memD: [0.0; 2],

        decode_mem: [0.0; 2 * (DECODE_BUFFER_SIZE + 120)],
        lpc: [0.0; 2 * LPC_ORDER],
        oldEBands: [0.0; 2 * 21],
        oldLogE: [0.0; 2 * 21],
        oldLogE2: [0.0; 2 * 21],
        backgroundLogE: [0.0; 2 * 21],
    };

    unsafe {
        opus_custom_decoder_ctl!(&mut st, OPUS_RESET_STATE);
    }

    st
}
/// Upstream C: celt/celt_decoder.c:deemphasis_stereo_simple
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
        pcm[2 * ju] = tmp0 * (1 as f32 / CELT_SIG_SCALE);
        pcm[2 * ju + 1] = tmp1 * (1 as f32 / CELT_SIG_SCALE);
        j += 1;
    }
    mem[0] = m0;
    mem[1] = m1;
}
/// Upstream C: celt/celt_decoder.c:deemphasis
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
    let vla = N as usize;
    let mut scratch: Vec<celt_sig> = ::std::vec::from_elem(0., vla);
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
                pcm[(c + j * C) as usize] = tmp_0 * (1 as f32 / CELT_SIG_SCALE);
                j += 1;
            }
        }
        mem[c as usize] = m;
        if apply_downsampling != 0 {
            j = 0;
            while j < Nd {
                pcm[(c + j * C) as usize] =
                    scratch[(j * downsample) as usize] * (1 as f32 / CELT_SIG_SCALE);
                j += 1;
            }
        }
        c += 1;
        if !(c < C) {
            break;
        }
    }
}
/// Upstream C: celt/celt_decoder.c:celt_synthesis
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
    let mut freq: Vec<celt_sig> = ::std::vec::from_elem(0., n + M as usize - 1);
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
        // Use a temporary Vec for freq2 instead of borrowing out_syn_ch1
        let mut freq2 = vec![0.0f32; n + M as usize - 1];
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
        let mut freq2 = vec![0.0f32; n];
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
        let channels: [&mut [celt_sig]; 2] = unsafe {
            // SAFETY: out_syn_ch0 and out_syn_ch1 are non-overlapping slices from
            // different regions of decode_mem (split_at_mut at the caller).
            // We need the array to index by c in the loop.
            let p0 = out_syn_ch0.as_mut_ptr();
            let l0 = out_syn_ch0.len();
            let p1 = out_syn_ch1.as_mut_ptr();
            let l1 = out_syn_ch1.len();
            [
                std::slice::from_raw_parts_mut(p0, l0),
                std::slice::from_raw_parts_mut(p1, l1),
            ]
        };
        let mut c = 0;
        loop {
            denormalise_bands(
                mode,
                &X[(c as usize * n)..(c as usize * n + n)],
                &mut freq,
                &oldBandE[(c * nbEBands) as usize..((c + 1) * nbEBands) as usize],
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
                    &mut channels[c as usize]
                        [NB as usize * bu..NB as usize * bu + mdct_sub_len + overlap_u],
                    mode.window,
                    overlap_u,
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
        && tf_select_table[LM as usize][(4 * isTransient + 0 + tf_changed) as usize] as i32
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
/// Upstream C: celt/celt_decoder.c:celt_decode_lost
fn celt_decode_lost(st: &mut OpusCustomDecoder, N: i32, LM: i32) {
    let C: i32 = st.channels as i32;
    let mode = st.mode;
    let nbEBands = mode.nbEBands as i32;
    let overlap = mode.overlap as i32;
    let overlap_u = overlap as usize;
    let eBands = &mode.eBands;
    let chan_stride = DECODE_BUFFER_SIZE + overlap_u;
    let n = N as usize;

    let loss_count = st.loss_count;
    let start = st.start;
    let noise_based = (loss_count >= 5 || start != 0 || st.skip_plc != 0) as i32;
    if noise_based != 0 {
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
        let decay: opus_val16 = if loss_count == 0 { 1.5f32 } else { 0.5f32 };
        let mut c = 0;
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
            if !(c < C) {
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
        // Shift decode_mem for each channel
        c = 0;
        loop {
            let ch_off = c as usize * chan_stride;
            let shift_len = 2048 - n + (overlap_u >> 1);
            st.decode_mem
                .copy_within(ch_off + n..ch_off + n + shift_len, ch_off);
            c += 1;
            if !(c < C) {
                break;
            }
        }
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
    } else {
        let mut fade: opus_val16 = Q15ONE;
        let pitch_index: i32;
        if loss_count == 0 {
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
        let mut etmp: Vec<opus_val32> = ::std::vec::from_elem(0., overlap_u);
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
            if loss_count == 0 {
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
                    let e = _exc[exc_off + (MAX_PERIOD as i32 - decay_length + i) as usize];
                    E1 += e * e;
                    let e = _exc[exc_off + (MAX_PERIOD as i32 - 2 * decay_length + i) as usize];
                    E2 += e * e;
                    i += 1;
                }
            }
            E1 = if E1 < E2 { E1 } else { E2 };
            let decay_0: opus_val16 = celt_sqrt(E1 / E2);
            // Shift decode_mem: memmove(buf, buf+N, (2048-N)*sizeof)
            st.decode_mem.copy_within(ch_off + n..ch_off + 2048, ch_off);
            let extrapolation_offset = MAX_PERIOD as i32 - pitch_index;
            let extrapolation_len = N + overlap;
            let mut attenuation: opus_val16 = fade * decay_0;
            let mut j_0 = 0i32;
            let mut i = j_0;
            let mut S1: opus_val32 = 0.0;
            while i < extrapolation_len {
                if j_0 >= pitch_index {
                    j_0 -= pitch_index;
                    attenuation = attenuation * decay_0;
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
            {
                comb_filter(
                    &mut etmp,
                    0,
                    &st.decode_mem[ch_off..ch_off + chan_stride],
                    DECODE_BUFFER_SIZE,
                    st.postfilter_period,
                    st.postfilter_period,
                    overlap,
                    -st.postfilter_gain,
                    -st.postfilter_gain,
                    st.postfilter_tapset,
                    st.postfilter_tapset,
                    &[],
                    0,
                    st.arch,
                );
            }
            {
                let mut i = 0;
                while i < overlap / 2 {
                    let iu = i as usize;
                    st.decode_mem[ch_off + DECODE_BUFFER_SIZE + iu] = window[iu]
                        * etmp[(overlap - 1 - i) as usize]
                        + window[(overlap - i - 1) as usize] * etmp[iu];
                    i += 1;
                }
            }
            c += 1;
            if !(c < C) {
                break;
            }
        }
    }
    st.loss_count = loss_count + 1;
}
pub unsafe fn celt_decode_with_ec(
    st: &mut OpusCustomDecoder,
    data: *const u8,
    len: i32,
    pcm: *mut opus_val16,
    mut frame_size: i32,
    dec: Option<&mut ec_dec>,
    accum: i32,
) -> i32 {
    let mut c: i32 = 0;
    let mut i: i32 = 0;
    let mut N: i32 = 0;
    let mut spread_decision: i32 = 0;
    let mut bits: i32 = 0;
    let mut _dec: ec_dec = ec_dec {
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
    let mut decode_mem: [*mut celt_sig; 2] = [0 as *mut celt_sig; 2];
    let mut out_syn: [*mut celt_sig; 2] = [0 as *mut celt_sig; 2];
    let mut shortBlocks: i32 = 0;
    let mut isTransient: i32 = 0;
    let mut intra_ener: i32 = 0;
    let CC: i32 = st.channels as i32;
    let mut LM: i32 = 0;
    let mut M: i32 = 0;
    let mut start: i32 = 0;
    let mut end: i32 = 0;
    let mut effEnd: i32 = 0;
    let mut codedBands: i32 = 0;
    let mut alloc_trim: i32 = 0;
    let mut postfilter_pitch: i32 = 0;
    let mut postfilter_gain: opus_val16 = 0.;
    let mut intensity: i32 = 0;
    let mut dual_stereo: i32 = 0;
    let mut total_bits: i32 = 0;
    let mut balance: i32 = 0;
    let mut tell: i32 = 0;
    let mut dynalloc_logp: i32 = 0;
    let mut postfilter_tapset: i32 = 0;
    let mut anti_collapse_rsv: i32 = 0;
    let mut anti_collapse_on: i32 = 0;
    let mut silence: i32 = 0;
    let C: i32 = st.stream_channels as i32;
    let mut mode: *const OpusCustomMode = 0 as *const OpusCustomMode;
    let mut nbEBands: i32 = 0;
    let mut overlap: i32 = 0;
    let mut eBands: *const i16 = 0 as *const i16;
    validate_celt_decoder(&*st);
    mode = st.mode;
    nbEBands = (*mode).nbEBands as i32;
    overlap = (*mode).overlap as i32;
    eBands = (*mode).eBands.as_ptr();
    start = st.start;
    end = st.end;
    frame_size *= st.downsample;

    let oldBandE = st.oldEBands.as_mut_ptr();
    let oldLogE = st.oldLogE.as_mut_ptr();
    let oldLogE2 = st.oldLogE2.as_mut_ptr();
    let backgroundLogE = st.backgroundLogE.as_mut_ptr();

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
    if len < 0 || len > 1275 || pcm.is_null() {
        return OPUS_BAD_ARG;
    }
    N = M * (*mode).shortMdctSize;
    c = 0;
    loop {
        decode_mem[c as usize] = (st.decode_mem)
            .as_mut_ptr()
            .offset((c * (DECODE_BUFFER_SIZE as i32 + overlap)) as isize);
        out_syn[c as usize] = (decode_mem[c as usize])
            .offset(DECODE_BUFFER_SIZE as isize)
            .offset(-(N as isize));
        c += 1;
        if !(c < CC) {
            break;
        }
    }
    effEnd = end;
    if effEnd > (*mode).effEBands {
        effEnd = (*mode).effEBands;
    }
    if data.is_null() || len <= 1 {
        celt_decode_lost(st, N, LM);
        {
            let out_n = N as usize;
            let in_ch: Vec<&[celt_sig]> = (0..CC as usize)
                .map(|c| std::slice::from_raw_parts(out_syn[c], out_n))
                .collect();
            let pcm_len = (frame_size / st.downsample * CC) as usize;
            deemphasis(
                &in_ch,
                std::slice::from_raw_parts_mut(pcm, pcm_len),
                N,
                CC,
                st.downsample,
                &(*mode).preemph,
                &mut st.preemph_memD,
                accum,
            );
        }
        return frame_size / st.downsample;
    }
    st.skip_plc = (st.loss_count != 0) as i32;
    let dec = if let Some(dec) = dec {
        dec
    } else {
        _dec = ec_dec_init(std::slice::from_raw_parts_mut(
            data as *mut u8,
            len as usize,
        ));
        &mut _dec
    };
    if C == 1 {
        i = 0;
        while i < nbEBands {
            *oldBandE.offset(i as isize) =
                if *oldBandE.offset(i as isize) > *oldBandE.offset((nbEBands + i) as isize) {
                    *oldBandE.offset(i as isize)
                } else {
                    *oldBandE.offset((nbEBands + i) as isize)
                };
            i += 1;
        }
    }
    total_bits = len * 8;
    tell = ec_tell(dec);
    if tell >= total_bits {
        silence = 1;
    } else if tell == 1 {
        silence = ec_dec_bit_logp(dec, 15);
    } else {
        silence = 0;
    }
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
    if LM > 0 && tell + 3 <= total_bits {
        isTransient = ec_dec_bit_logp(dec, 3);
        tell = ec_tell(dec);
    } else {
        isTransient = 0;
    }
    if isTransient != 0 {
        shortBlocks = M;
    } else {
        shortBlocks = 0;
    }
    intra_ener = if tell + 3 <= total_bits {
        ec_dec_bit_logp(dec, 3)
    } else {
        0
    };
    unquant_coarse_energy(
        &*mode,
        start,
        end,
        std::slice::from_raw_parts_mut(oldBandE, (C * nbEBands) as usize),
        intra_ener,
        dec,
        C,
        LM,
    );
    let vla = nbEBands as usize;
    let mut tf_res: Vec<i32> = ::std::vec::from_elem(0, vla);
    tf_decode(start, end, isTransient, &mut tf_res, LM, dec);
    tell = ec_tell(dec);
    spread_decision = SPREAD_NORMAL;
    if tell + 4 <= total_bits {
        spread_decision = ec_dec_icdf(dec, &spread_icdf, 5);
    }
    let vla_0 = nbEBands as usize;
    let mut cap: Vec<i32> = ::std::vec::from_elem(0, vla_0);
    init_caps(&*mode, &mut cap, LM, C);
    let vla_1 = nbEBands as usize;
    let mut offsets: Vec<i32> = ::std::vec::from_elem(0, vla_1);
    dynalloc_logp = 6;
    total_bits <<= BITRES;
    tell = ec_tell_frac(dec) as i32;
    i = start;
    while i < end {
        let mut width: i32 = 0;
        let mut quanta: i32 = 0;
        let mut dynalloc_loop_logp: i32 = 0;
        let mut boost: i32 = 0;
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
        while tell + (dynalloc_loop_logp << BITRES) < total_bits
            && boost < *cap.as_mut_ptr().offset(i as isize)
        {
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
        *offsets.as_mut_ptr().offset(i as isize) = boost;
        if boost > 0 {
            dynalloc_logp = if 2 > dynalloc_logp - 1 {
                2
            } else {
                dynalloc_logp - 1
            };
        }
        i += 1;
    }
    let vla_2 = nbEBands as usize;
    let mut fine_quant: Vec<i32> = ::std::vec::from_elem(0, vla_2);
    alloc_trim = if tell + ((6) << BITRES) <= total_bits {
        ec_dec_icdf(dec, &trim_icdf, 7)
    } else {
        5
    };
    bits = (((len * 8) << BITRES) as u32)
        .wrapping_sub(ec_tell_frac(dec))
        .wrapping_sub(1) as i32;
    anti_collapse_rsv = if isTransient != 0 && LM >= 2 && bits >= (LM + 2) << BITRES {
        (1) << BITRES
    } else {
        0
    };
    bits -= anti_collapse_rsv;
    let vla_3 = nbEBands as usize;
    let mut pulses: Vec<i32> = ::std::vec::from_elem(0, vla_3);
    let vla_4 = nbEBands as usize;
    let mut fine_priority: Vec<i32> = ::std::vec::from_elem(0, vla_4);
    codedBands = clt_compute_allocation(
        &*mode,
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
        &*mode,
        start,
        end,
        std::slice::from_raw_parts_mut(oldBandE, (C * nbEBands) as usize),
        &fine_quant,
        dec,
        C,
    );
    c = 0;
    loop {
        memmove(
            decode_mem[c as usize] as *mut core::ffi::c_void,
            (decode_mem[c as usize]).offset(N as isize) as *const core::ffi::c_void,
            ((2048 - N + overlap / 2) as u64)
                .wrapping_mul(::core::mem::size_of::<celt_sig>() as u64)
                .wrapping_add(
                    (0 * (decode_mem[c as usize])
                        .offset_from((decode_mem[c as usize]).offset(N as isize))
                        as i64) as u64,
                ),
        );
        c += 1;
        if !(c < CC) {
            break;
        }
    }
    let vla_5 = (C * nbEBands) as usize;
    let mut collapse_masks: Vec<u8> = ::std::vec::from_elem(0, vla_5);
    let vla_6 = (C * N) as usize;
    let mut X: Vec<celt_norm> = ::std::vec::from_elem(0., vla_6);
    if C == 2 {
        let (x_part, y_part) = X.split_at_mut(N as usize);
        quant_all_bands(
            0,
            &*mode,
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
            &*mode,
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
        &*mode,
        start,
        end,
        std::slice::from_raw_parts_mut(oldBandE, (C * nbEBands) as usize),
        &fine_quant,
        &fine_priority,
        len * 8 - ec_tell(dec),
        dec,
        C,
    );
    if anti_collapse_on != 0 {
        anti_collapse(
            &*mode,
            &mut X,
            &mut collapse_masks,
            LM,
            C,
            N,
            start,
            end,
            std::slice::from_raw_parts(oldBandE, (2 * nbEBands) as usize),
            std::slice::from_raw_parts(oldLogE, (2 * nbEBands) as usize),
            std::slice::from_raw_parts(oldLogE2, (2 * nbEBands) as usize),
            &pulses,
            st.rng,
            st.arch,
        );
    }
    if silence != 0 {
        i = 0;
        while i < C * nbEBands {
            *oldBandE.offset(i as isize) = -28.0f32;
            i += 1;
        }
    }
    {
        let out_syn_len = N as usize + overlap as usize;
        celt_synthesis(
            &*mode,
            &X,
            std::slice::from_raw_parts_mut(out_syn[0], out_syn_len),
            std::slice::from_raw_parts_mut(out_syn[1], if CC >= 2 { out_syn_len } else { 0 }),
            std::slice::from_raw_parts(oldBandE, (2 * nbEBands) as usize),
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
            let dm_len = DECODE_BUFFER_SIZE + overlap as usize;
            let dm_slice = std::slice::from_raw_parts_mut(decode_mem[c as usize], dm_len);
            let out_syn_off = (DECODE_BUFFER_SIZE as i32 - N) as usize;
            comb_filter_inplace(
                dm_slice,
                out_syn_off,
                st.postfilter_period_old,
                st.postfilter_period,
                (*mode).shortMdctSize,
                st.postfilter_gain_old,
                st.postfilter_gain,
                st.postfilter_tapset_old,
                st.postfilter_tapset,
                &(&(*mode).window)[..overlap as usize],
                overlap,
                st.arch,
            );
            if LM != 0 {
                comb_filter_inplace(
                    dm_slice,
                    out_syn_off + (*mode).shortMdctSize as usize,
                    st.postfilter_period,
                    postfilter_pitch,
                    N - (*mode).shortMdctSize,
                    st.postfilter_gain,
                    postfilter_gain,
                    st.postfilter_tapset,
                    postfilter_tapset,
                    &(&(*mode).window)[..overlap as usize],
                    overlap,
                    st.arch,
                );
            }
        }
        c += 1;
        if !(c < CC) {
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
    if C == 1 {
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
        let mut max_background_increase: opus_val16 = 0.;
        memcpy(
            oldLogE2 as *mut core::ffi::c_void,
            oldLogE as *const core::ffi::c_void,
            ((2 * nbEBands) as u64)
                .wrapping_mul(::core::mem::size_of::<opus_val16>() as u64)
                .wrapping_add((0 * oldLogE2.offset_from(oldLogE) as i64) as u64),
        );
        memcpy(
            oldLogE as *mut core::ffi::c_void,
            oldBandE as *const core::ffi::c_void,
            ((2 * nbEBands) as u64)
                .wrapping_mul(::core::mem::size_of::<opus_val16>() as u64)
                .wrapping_add((0 * oldLogE.offset_from(oldBandE) as i64) as u64),
        );
        if st.loss_count < 10 {
            max_background_increase = M as f32 * 0.001f32;
        } else {
            max_background_increase = 1.0f32;
        }
        i = 0;
        while i < 2 * nbEBands {
            *backgroundLogE.offset(i as isize) = if *backgroundLogE.offset(i as isize)
                + max_background_increase
                < *oldBandE.offset(i as isize)
            {
                *backgroundLogE.offset(i as isize) + max_background_increase
            } else {
                *oldBandE.offset(i as isize)
            };
            i += 1;
        }
    } else {
        i = 0;
        while i < 2 * nbEBands {
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
            let ref mut fresh0 = *oldLogE2.offset((c * nbEBands + i) as isize);
            *fresh0 = -28.0f32;
            *oldLogE.offset((c * nbEBands + i) as isize) = *fresh0;
            i += 1;
        }
        i = end;
        while i < nbEBands {
            *oldBandE.offset((c * nbEBands + i) as isize) = 0 as opus_val16;
            let ref mut fresh1 = *oldLogE2.offset((c * nbEBands + i) as isize);
            *fresh1 = -28.0f32;
            *oldLogE.offset((c * nbEBands + i) as isize) = *fresh1;
            i += 1;
        }
        c += 1;
        if !(c < 2) {
            break;
        }
    }
    st.rng = dec.rng;
    {
        let out_n = N as usize;
        let in_ch: Vec<&[celt_sig]> = (0..CC as usize)
            .map(|c| std::slice::from_raw_parts(out_syn[c], out_n))
            .collect();
        let pcm_len = (frame_size / st.downsample * CC) as usize;
        deemphasis(
            &in_ch,
            std::slice::from_raw_parts_mut(pcm, pcm_len),
            N,
            CC,
            st.downsample,
            &(*mode).preemph,
            &mut st.preemph_memD,
            accum,
        );
    }
    st.loss_count = 0;
    if ec_tell(dec) > 8 * len {
        return OPUS_INTERNAL_ERROR;
    }
    if ec_get_error(dec) != 0 {
        st.error = 1;
    }
    return frame_size / st.downsample;
}
pub unsafe fn opus_custom_decoder_ctl_impl(
    st: &mut OpusCustomDecoder,
    request: i32,
    args: VarArgs,
) -> i32 {
    let current_block: u64;
    let mut ap = args;
    match request {
        CELT_SET_START_BAND_REQUEST => {
            let value: i32 = ap.arg::<i32>();
            if value < 0 || value >= st.mode.nbEBands as i32 {
                current_block = 7990025728955927862;
            } else {
                st.start = value;
                current_block = 3689906465960840878;
            }
        }
        CELT_SET_END_BAND_REQUEST => {
            let value_0: i32 = ap.arg::<i32>();
            if value_0 < 1 || value_0 > st.mode.nbEBands as i32 {
                current_block = 7990025728955927862;
            } else {
                st.end = value_0;
                current_block = 3689906465960840878;
            }
        }
        CELT_SET_CHANNELS_REQUEST => {
            let value_1: i32 = ap.arg::<i32>();
            if value_1 < 1 || value_1 > 2 {
                current_block = 7990025728955927862;
            } else {
                st.stream_channels = value_1 as usize;
                current_block = 3689906465960840878;
            }
        }
        CELT_GET_AND_CLEAR_ERROR_REQUEST => {
            let value_2: &mut i32 = ap.arg::<&mut i32>();
            *value_2 = st.error;
            st.error = 0;
            current_block = 3689906465960840878;
        }
        OPUS_GET_LOOKAHEAD_REQUEST => {
            let value_3 = ap.arg::<&mut i32>();
            *value_3 = st.overlap as i32 / st.downsample;
            current_block = 3689906465960840878;
        }
        OPUS_RESET_STATE => {
            let st = &mut *st;

            st.rng = 0;
            st.error = 0;
            st.last_pitch_index = 0;
            st.loss_count = 0;
            st.skip_plc = 1;
            st.postfilter_period = 0;
            st.postfilter_period_old = 0;
            st.postfilter_gain = 0.0;
            st.postfilter_gain_old = 0.0;
            st.postfilter_tapset = 0;
            st.postfilter_tapset_old = 0;
            st.preemph_memD = [0.0; 2];
            st.decode_mem.fill(0.0);
            st.lpc.fill(0.0);
            st.oldEBands.fill(0.0);
            st.oldLogE.fill(-28.0);
            st.oldLogE2.fill(-28.0);
            st.backgroundLogE.fill(0.0);

            current_block = 3689906465960840878;
        }
        OPUS_GET_PITCH_REQUEST => {
            let value_4 = ap.arg::<&mut i32>();
            *value_4 = st.postfilter_period;
            current_block = 3689906465960840878;
        }
        CELT_GET_MODE_REQUEST => {
            let value_5 = ap.arg::<&mut *const OpusCustomMode>();
            *value_5 = st.mode;
            current_block = 3689906465960840878;
        }
        CELT_SET_SIGNALLING_REQUEST => {
            let value_6: i32 = ap.arg::<i32>();
            st.signalling = value_6;
            current_block = 3689906465960840878;
        }
        OPUS_GET_FINAL_RANGE_REQUEST => {
            let value_7 = ap.arg::<&mut u32>();
            *value_7 = st.rng;
            current_block = 3689906465960840878;
        }
        OPUS_SET_PHASE_INVERSION_DISABLED_REQUEST => {
            let value_8: i32 = ap.arg::<i32>();
            if value_8 < 0 || value_8 > 1 {
                current_block = 7990025728955927862;
            } else {
                st.disable_inv = value_8;
                current_block = 3689906465960840878;
            }
        }
        OPUS_GET_PHASE_INVERSION_DISABLED_REQUEST => {
            let value_9 = ap.arg::<&mut i32>();
            *value_9 = st.disable_inv;
            current_block = 3689906465960840878;
        }
        _ => return OPUS_UNIMPLEMENTED,
    }
    match current_block {
        3689906465960840878 => return OPUS_OK,
        _ => return OPUS_BAD_ARG,
    };
}
#[macro_export]
macro_rules! opus_custom_decoder_ctl {
    ($st:expr, $request:expr, $($arg:expr),*) => {
        $crate::opus_custom_decoder_ctl_impl($st, $request, $crate::varargs!($($arg),*))
    };
    ($st:expr, $request:expr) => {
        opus_custom_decoder_ctl!($st, $request,)
    };
    ($st:expr, $request:expr, $($arg:expr),*,) => {
        opus_custom_decoder_ctl!($st, $request, $($arg),*)
    };
}
