//! SILK encoder API.
//!
//! Upstream C: `silk/enc_API.c`

#[derive(Copy, Clone, Default)]
#[repr(C)]
pub struct silk_EncControlStruct {
    pub nChannelsAPI: i32,
    pub nChannelsInternal: i32,
    pub API_sampleRate: i32,
    pub maxInternalSampleRate: i32,
    pub minInternalSampleRate: i32,
    pub desiredInternalSampleRate: i32,
    pub payloadSize_ms: i32,
    pub bitRate: i32,
    pub packetLossPercentage: i32,
    pub complexity: i32,
    pub useInBandFEC: i32,
    pub useDRED: i32,
    pub LBRR_coded: i32,
    pub useDTX: i32,
    pub useCBR: i32,
    pub maxBits: i32,
    pub toMono: i32,
    pub opusCanSwitch: i32,
    pub reducedDependency: i32,
    pub internalSampleRate: i32,
    pub allowBandwidthSwitch: i32,
    pub inWBmodeWithoutVariableLP: i32,
    pub stereoWidth_Q14: i32,
    pub switchReady: i32,
    pub signalType: i32,
    pub offset: i32,
}
pub mod errors_h {
    #[allow(unused)]
    pub const SILK_ENC_INPUT_INVALID_NO_OF_SAMPLES: i32 = -(101);
    pub const SILK_NO_ERROR: i32 = 0;
}
use self::errors_h::{SILK_ENC_INPUT_INVALID_NO_OF_SAMPLES, SILK_NO_ERROR};
use crate::arch::Arch;
use crate::celt::entcode::ec_tell;
use crate::celt::entenc::{ec_enc, ec_enc_icdf, ec_enc_patch_initial_bits};
use crate::celt::float_cast::FLOAT2INT16;

use crate::silk::check_control_input::check_control_input;
use crate::silk::control_SNR::silk_control_SNR;
use crate::silk::control_codec::silk_control_encoder;
use crate::silk::define::{
    CODE_CONDITIONALLY, CODE_INDEPENDENTLY, CODE_INDEPENDENTLY_NO_LTP_SCALING,
    ENCODER_NUM_CHANNELS, TYPE_NO_VOICE_ACTIVITY,
};
use crate::silk::encode_indices::silk_encode_indices;
use crate::silk::encode_pulses::silk_encode_pulses;
use crate::silk::float::encode_frame_FLP::{silk_encode_do_VAD_FLP, silk_encode_frame_FLP};
use crate::silk::float::structs_FLP::{silk_encoder, silk_shape_state_FLP};
use crate::silk::init_encoder::silk_init_encoder;
use crate::silk::resampler::silk_resampler;

use crate::silk::stereo_LR_to_MS::silk_stereo_LR_to_MS;
use crate::silk::stereo_encode_pred::{silk_stereo_encode_mid_only, silk_stereo_encode_pred};
use crate::silk::structs::{silk_LP_state, silk_nsq_state};
use crate::silk::tables_other::{silk_LBRR_flags_iCDF_ptr, silk_Quantization_Offsets_Q10};
use crate::silk::tuning_parameters::{
    BITRESERVOIR_DECAY_TIME_MS, MAX_BANDWIDTH_SWITCH_DELAY_MS, SPEECH_ACTIVITY_DTX_THRES,
};
use crate::silk::HP_variable_cutoff::silk_HP_variable_cutoff;

/// Upstream C: silk/enc_API.c:silk_InitEncoder
pub fn silk_InitEncoder(
    psEnc: &mut silk_encoder,
    arch: Arch,
    encStatus: &mut silk_EncControlStruct,
) -> i32 {
    // Zero-init the encoder state
    *psEnc = silk_encoder::default();
    let mut ret: i32 = SILK_NO_ERROR;
    for n in 0..ENCODER_NUM_CHANNELS as usize {
        ret += silk_init_encoder(&mut psEnc.state_Fxx[n], arch);
        if ret != 0 {
            return ret;
        }
    }
    psEnc.nChannelsAPI = 1;
    psEnc.nChannelsInternal = 1;
    ret += silk_QueryEncoder(psEnc, encStatus);
    if ret != 0 {
        return ret;
    }
    ret
}
/// Upstream C: silk/enc_API.c:silk_QueryEncoder
fn silk_QueryEncoder(psEnc: &silk_encoder, encStatus: &mut silk_EncControlStruct) -> i32 {
    let state = &psEnc.state_Fxx[0];
    encStatus.nChannelsAPI = psEnc.nChannelsAPI;
    encStatus.nChannelsInternal = psEnc.nChannelsInternal;
    encStatus.API_sampleRate = state.sCmn.API_fs_Hz;
    encStatus.maxInternalSampleRate = state.sCmn.maxInternal_fs_Hz;
    encStatus.minInternalSampleRate = state.sCmn.minInternal_fs_Hz;
    encStatus.desiredInternalSampleRate = state.sCmn.desiredInternal_fs_Hz;
    encStatus.payloadSize_ms = state.sCmn.PacketSize_ms;
    encStatus.bitRate = state.sCmn.TargetRate_bps;
    encStatus.packetLossPercentage = state.sCmn.PacketLoss_perc;
    encStatus.complexity = state.sCmn.Complexity;
    encStatus.useInBandFEC = state.sCmn.useInBandFEC;
    encStatus.useDTX = state.sCmn.useDTX;
    encStatus.useCBR = state.sCmn.useCBR;
    encStatus.internalSampleRate = state.sCmn.fs_kHz as i16 as i32 * 1000;
    encStatus.allowBandwidthSwitch = state.sCmn.allow_bandwidth_switch;
    encStatus.inWBmodeWithoutVariableLP =
        (state.sCmn.fs_kHz == 16 && state.sCmn.sLP.mode == 0) as i32;
    SILK_NO_ERROR
}
/// Upstream C: silk/enc_API.c:silk_Encode
#[allow(clippy::too_many_arguments)]
pub fn silk_Encode(
    psEnc: &mut silk_encoder,
    encControl: &mut silk_EncControlStruct,
    samplesIn: &[f32],
    nSamplesIn: i32,
    mut psRangeEnc: Option<&mut ec_enc>,
    nBytesOut: &mut i32,
    prefillFlag: i32,
    activity: i32,
) -> i32 {
    let mut n: i32;
    let mut i: i32;
    let mut nBits: i32;
    let mut flags: i32;
    let mut tmp_payloadSize_ms: i32 = 0;
    let mut tmp_complexity: i32 = 0;
    let mut ret: i32 = 0;
    let mut nSamplesToBuffer: i32;
    let mut nSamplesFromInput: i32;
    let mut TargetRate_bps: i32;
    let mut MStargetRates_bps: [i32; 2] = [0; 2];
    let mut channelRate_bps: i32;
    let mut LBRR_symbol: i32;
    let mut samplesIn_off: usize = 0;
    let mut nSamplesIn = nSamplesIn;

    debug_assert!(
        encControl.nChannelsAPI >= encControl.nChannelsInternal
            && encControl.nChannelsAPI >= psEnc.nChannelsInternal
    );
    if encControl.reducedDependency != 0 {
        n = 0;
        while n < encControl.nChannelsAPI {
            unsafe { &mut *psEnc.state_Fxx.get_unchecked_mut(n as usize) }
                .sCmn
                .first_frame_after_reset = 1;
            n += 1;
        }
    }
    n = 0;
    while n < encControl.nChannelsAPI {
        unsafe { &mut *psEnc.state_Fxx.get_unchecked_mut(n as usize) }
            .sCmn
            .nFramesEncoded = 0;
        n += 1;
    }
    ret = check_control_input(encControl);
    if ret != 0 {
        return ret;
    }
    encControl.switchReady = 0;
    if encControl.nChannelsInternal > psEnc.nChannelsInternal {
        let arch = psEnc.state_Fxx[0].sCmn.arch;
        ret += silk_init_encoder(&mut psEnc.state_Fxx[1], arch);
        psEnc.sStereo.pred_prev_Q13 = [0; 2];
        psEnc.sStereo.sSide = [0; 2];
        psEnc.sStereo.mid_side_amp_Q0[0] = 0;
        psEnc.sStereo.mid_side_amp_Q0[1] = 1;
        psEnc.sStereo.mid_side_amp_Q0[2] = 0;
        psEnc.sStereo.mid_side_amp_Q0[3] = 1;
        psEnc.sStereo.width_prev_Q14 = 0;
        psEnc.sStereo.smth_width_Q14 = ((1 << 14) as f64 + 0.5f64) as i32 as i16;
        if psEnc.nChannelsAPI == 2 {
            psEnc.state_Fxx[1].sCmn.resampler_state = psEnc.state_Fxx[0].sCmn.resampler_state;
            psEnc.state_Fxx[1].sCmn.In_HP_State = psEnc.state_Fxx[0].sCmn.In_HP_State;
        }
    }
    let transition = (encControl.payloadSize_ms != psEnc.state_Fxx[0].sCmn.PacketSize_ms
        || psEnc.nChannelsInternal != encControl.nChannelsInternal) as i32;
    psEnc.nChannelsAPI = encControl.nChannelsAPI;
    psEnc.nChannelsInternal = encControl.nChannelsInternal;
    let nBlocksOf10ms = 100 * nSamplesIn / encControl.API_sampleRate;
    let tot_blocks = if nBlocksOf10ms > 1 {
        nBlocksOf10ms >> 1
    } else {
        1
    };
    let mut curr_block: i32 = 0;
    if prefillFlag != 0 {
        let mut save_LP = silk_LP_state {
            In_LP_State: [0; 2],
            transition_frame_no: 0,
            mode: 0,
            saved_fs_kHz: 0,
        };
        if nBlocksOf10ms != 1 {
            return SILK_ENC_INPUT_INVALID_NO_OF_SAMPLES;
        }
        if prefillFlag == 2 {
            save_LP = psEnc.state_Fxx[0].sCmn.sLP;
            save_LP.saved_fs_kHz = psEnc.state_Fxx[0].sCmn.fs_kHz;
        }
        n = 0;
        while n < encControl.nChannelsInternal {
            let arch = unsafe { &*psEnc.state_Fxx.get_unchecked(n as usize) }.sCmn.arch;
            ret = silk_init_encoder(
                unsafe { &mut *psEnc.state_Fxx.get_unchecked_mut(n as usize) },
                arch,
            );
            if prefillFlag == 2 {
                unsafe { &mut *psEnc.state_Fxx.get_unchecked_mut(n as usize) }.sCmn.sLP = save_LP;
            }
            debug_assert_eq!(ret, 0);
            n += 1;
        }
        tmp_payloadSize_ms = encControl.payloadSize_ms;
        encControl.payloadSize_ms = 10;
        tmp_complexity = encControl.complexity;
        encControl.complexity = 0;
        n = 0;
        while n < encControl.nChannelsInternal {
            unsafe { &mut *psEnc.state_Fxx.get_unchecked_mut(n as usize) }
                .sCmn
                .controlled_since_last_payload = 0;
            unsafe { &mut *psEnc.state_Fxx.get_unchecked_mut(n as usize) }
                .sCmn
                .prefillFlag = 1;
            n += 1;
        }
    } else {
        if nBlocksOf10ms * encControl.API_sampleRate != 100 * nSamplesIn || nSamplesIn < 0 {
            return SILK_ENC_INPUT_INVALID_NO_OF_SAMPLES;
        }
        if 1000 * nSamplesIn > encControl.payloadSize_ms * encControl.API_sampleRate {
            return SILK_ENC_INPUT_INVALID_NO_OF_SAMPLES;
        }
    }
    n = 0;
    while n < encControl.nChannelsInternal {
        let force_fs_kHz: i32 = if n == 1 {
            psEnc.state_Fxx[0].sCmn.fs_kHz
        } else {
            0
        };
        ret = silk_control_encoder(
            unsafe { &mut *psEnc.state_Fxx.get_unchecked_mut(n as usize) },
            encControl,
            psEnc.allowBandwidthSwitch,
            n,
            force_fs_kHz,
        );
        if ret != 0 {
            return ret;
        }
        if unsafe { &*psEnc.state_Fxx.get_unchecked(n as usize) }
            .sCmn
            .first_frame_after_reset
            != 0
            || transition != 0
        {
            i = 0;
            while i < psEnc.state_Fxx[0].sCmn.nFramesPerPacket {
                unsafe {
                    *psEnc
                        .state_Fxx
                        .get_unchecked_mut(n as usize)
                        .sCmn
                        .LBRR_flags
                        .get_unchecked_mut(i as usize) = 0;
                }
                i += 1;
            }
        }
        let st = unsafe { &mut *psEnc.state_Fxx.get_unchecked_mut(n as usize) };
        st.sCmn.inDTX = st.sCmn.useDTX;
        n += 1;
    }
    debug_assert!(
        encControl.nChannelsInternal == 1
            || psEnc.state_Fxx[0].sCmn.fs_kHz == psEnc.state_Fxx[1].sCmn.fs_kHz
    );
    let nSamplesToBufferMax = 10 * nBlocksOf10ms * psEnc.state_Fxx[0].sCmn.fs_kHz;
    let nSamplesFromInputMax = nSamplesToBufferMax * psEnc.state_Fxx[0].sCmn.API_fs_Hz
        / (psEnc.state_Fxx[0].sCmn.fs_kHz * 1000);
    // nSamplesFromInputMax max: 10 * 6 * 16 * 48000 / (16 * 1000) = 2880
    const MAX_BUF: usize = 2880;
    debug_assert!((nSamplesFromInputMax as usize) <= MAX_BUF);
    let mut buf = [0i16; MAX_BUF];
    loop {
        let mut curr_nBitsUsedLBRR: i32 = 0;
        nSamplesToBuffer =
            psEnc.state_Fxx[0].sCmn.frame_length as i32 - psEnc.state_Fxx[0].sCmn.inputBufIx;
        nSamplesToBuffer = nSamplesToBuffer.min(nSamplesToBufferMax);
        nSamplesFromInput = nSamplesToBuffer * psEnc.state_Fxx[0].sCmn.API_fs_Hz
            / (psEnc.state_Fxx[0].sCmn.fs_kHz * 1000);
        if encControl.nChannelsAPI == 2 && encControl.nChannelsInternal == 2 {
            let id = psEnc.state_Fxx[0].sCmn.nFramesEncoded;
            // De-interleave left channel
            for k in 0..nSamplesFromInput as usize {
                unsafe {
                    *buf.get_unchecked_mut(k) =
                        FLOAT2INT16(*samplesIn.get_unchecked(samplesIn_off + 2 * k));
                }
            }
            // Making sure to start both resamplers from the same state when switching from mono to stereo
            if psEnc.nPrevChannelsInternal == 1 && id == 0 {
                psEnc.state_Fxx[1].sCmn.resampler_state = psEnc.state_Fxx[0].sCmn.resampler_state;
            }
            {
                let ix0 = psEnc.state_Fxx[0].sCmn.inputBufIx as usize;
                let [s0, _] = &mut psEnc.state_Fxx;
                ret += silk_resampler(
                    &mut s0.sCmn.resampler_state,
                    &mut s0.sCmn.inputBuf[ix0 + 2..ix0 + 2 + nSamplesToBuffer as usize],
                    &buf[..nSamplesFromInput as usize],
                );
            }
            psEnc.state_Fxx[0].sCmn.inputBufIx += nSamplesToBuffer;

            nSamplesToBuffer =
                psEnc.state_Fxx[1].sCmn.frame_length as i32 - psEnc.state_Fxx[1].sCmn.inputBufIx;
            nSamplesToBuffer =
                nSamplesToBuffer.min(10 * nBlocksOf10ms * psEnc.state_Fxx[1].sCmn.fs_kHz);
            // De-interleave right channel
            for k in 0..nSamplesFromInput as usize {
                unsafe {
                    *buf.get_unchecked_mut(k) =
                        FLOAT2INT16(*samplesIn.get_unchecked(samplesIn_off + 2 * k + 1));
                }
            }
            {
                let ix1 = psEnc.state_Fxx[1].sCmn.inputBufIx as usize;
                let [_, s1] = &mut psEnc.state_Fxx;
                ret += silk_resampler(
                    &mut s1.sCmn.resampler_state,
                    &mut s1.sCmn.inputBuf[ix1 + 2..ix1 + 2 + nSamplesToBuffer as usize],
                    &buf[..nSamplesFromInput as usize],
                );
            }
            psEnc.state_Fxx[1].sCmn.inputBufIx += nSamplesToBuffer;
        } else if encControl.nChannelsAPI == 2 && encControl.nChannelsInternal == 1 {
            // Downmix stereo to mono
            for k in 0..nSamplesFromInput as usize {
                let sum = unsafe {
                    FLOAT2INT16(
                        *samplesIn.get_unchecked(samplesIn_off + 2 * k)
                            + *samplesIn.get_unchecked(samplesIn_off + 2 * k + 1),
                    )
                } as i32;
                unsafe {
                    *buf.get_unchecked_mut(k) = ((sum >> 1) + (sum & 1)) as i16;
                }
            }
            {
                let ix0 = psEnc.state_Fxx[0].sCmn.inputBufIx as usize;
                let [s0, _] = &mut psEnc.state_Fxx;
                ret += silk_resampler(
                    &mut s0.sCmn.resampler_state,
                    &mut s0.sCmn.inputBuf[ix0 + 2..ix0 + 2 + nSamplesToBuffer as usize],
                    &buf[..nSamplesFromInput as usize],
                );
            }
            if psEnc.nPrevChannelsInternal == 2 && psEnc.state_Fxx[0].sCmn.nFramesEncoded == 0 {
                {
                    let ix1 = psEnc.state_Fxx[1].sCmn.inputBufIx as usize;
                    let [_, s1] = &mut psEnc.state_Fxx;
                    ret += silk_resampler(
                        &mut s1.sCmn.resampler_state,
                        &mut s1.sCmn.inputBuf[ix1 + 2..],
                        &buf[..nSamplesFromInput as usize],
                    );
                }
                let frame_len = psEnc.state_Fxx[0].sCmn.frame_length as i32;
                let ix0 = psEnc.state_Fxx[0].sCmn.inputBufIx;
                let ix1 = psEnc.state_Fxx[1].sCmn.inputBufIx;
                for k in 0..frame_len {
                    let idx0 = (ix0 + k + 2) as usize;
                    let idx1 = (ix1 + k + 2) as usize;
                    unsafe {
                        *psEnc.state_Fxx[0].sCmn.inputBuf.get_unchecked_mut(idx0) =
                            ((*psEnc.state_Fxx[0].sCmn.inputBuf.get_unchecked(idx0) as i32
                                + *psEnc.state_Fxx[1].sCmn.inputBuf.get_unchecked(idx1) as i32)
                                >> 1) as i16;
                    }
                }
            }
            psEnc.state_Fxx[0].sCmn.inputBufIx += nSamplesToBuffer;
        } else {
            debug_assert!(encControl.nChannelsAPI == 1 && encControl.nChannelsInternal == 1);
            for k in 0..nSamplesFromInput as usize {
                unsafe {
                    *buf.get_unchecked_mut(k) =
                        FLOAT2INT16(*samplesIn.get_unchecked(samplesIn_off + k));
                }
            }
            {
                let ix0 = psEnc.state_Fxx[0].sCmn.inputBufIx as usize;
                let [s0, _] = &mut psEnc.state_Fxx;
                ret += silk_resampler(
                    &mut s0.sCmn.resampler_state,
                    &mut s0.sCmn.inputBuf[ix0 + 2..ix0 + 2 + nSamplesToBuffer as usize],
                    &buf[..nSamplesFromInput as usize],
                );
            }
            psEnc.state_Fxx[0].sCmn.inputBufIx += nSamplesToBuffer;
        }
        samplesIn_off += (nSamplesFromInput * encControl.nChannelsAPI) as usize;
        nSamplesIn -= nSamplesFromInput;
        psEnc.allowBandwidthSwitch = 0;
        if psEnc.state_Fxx[0].sCmn.inputBufIx < psEnc.state_Fxx[0].sCmn.frame_length as i32 {
            break;
        }
        debug_assert_eq!(
            psEnc.state_Fxx[0].sCmn.inputBufIx,
            psEnc.state_Fxx[0].sCmn.frame_length as i32
        );
        debug_assert!(
            encControl.nChannelsInternal == 1
                || psEnc.state_Fxx[1].sCmn.inputBufIx
                    == psEnc.state_Fxx[1].sCmn.frame_length as i32
        );
        if psEnc.state_Fxx[0].sCmn.nFramesEncoded == 0 && prefillFlag == 0 {
            let psRangeEnc = &mut **psRangeEnc.as_mut().unwrap();

            let mut iCDF: [u8; 2] = [0, 0];
            iCDF[0] = (256
                - (256
                    >> ((psEnc.state_Fxx[0].sCmn.nFramesPerPacket + 1)
                        * encControl.nChannelsInternal))) as u8;
            ec_enc_icdf(psRangeEnc, 0, &iCDF, 8);
            curr_nBitsUsedLBRR = ec_tell(psRangeEnc);
            n = 0;
            while n < encControl.nChannelsInternal {
                LBRR_symbol = 0;
                i = 0;
                while i < psEnc.state_Fxx[n as usize].sCmn.nFramesPerPacket {
                    LBRR_symbol |= ((unsafe {
                        *psEnc.state_Fxx[n as usize]
                            .sCmn
                            .LBRR_flags
                            .get_unchecked(i as usize)
                    } as u32)
                        << i) as i32;
                    i += 1;
                }
                psEnc.state_Fxx[n as usize].sCmn.LBRR_flag =
                    (if LBRR_symbol > 0 { 1 } else { 0 }) as i8;
                if LBRR_symbol != 0
                    && unsafe { &*psEnc.state_Fxx.get_unchecked(n as usize) }
                        .sCmn
                        .nFramesPerPacket
                        > 1
                {
                    ec_enc_icdf(
                        psRangeEnc,
                        LBRR_symbol - 1,
                        unsafe {
                            *silk_LBRR_flags_iCDF_ptr.get_unchecked(
                                (psEnc.state_Fxx.get_unchecked(n as usize).sCmn.nFramesPerPacket
                                    - 2) as usize,
                            )
                        },
                        8,
                    );
                }
                n += 1;
            }
            i = 0;
            while i < psEnc.state_Fxx[0].sCmn.nFramesPerPacket {
                n = 0;
                while n < encControl.nChannelsInternal {
                    if unsafe {
                        *psEnc.state_Fxx[n as usize]
                            .sCmn
                            .LBRR_flags
                            .get_unchecked(i as usize)
                    } != 0
                    {
                        if encControl.nChannelsInternal == 2 && n == 0 {
                            silk_stereo_encode_pred(
                                psRangeEnc,
                                &psEnc.sStereo.predIx[i as usize],
                            );
                            if unsafe {
                                *psEnc.state_Fxx[1]
                                    .sCmn
                                    .LBRR_flags
                                    .get_unchecked(i as usize)
                            } == 0
                            {
                                silk_stereo_encode_mid_only(
                                    psRangeEnc,
                                    psEnc.sStereo.mid_only_flags[i as usize],
                                );
                            }
                        }
                        let condCoding = if i > 0
                            && unsafe {
                                *psEnc.state_Fxx[n as usize]
                                    .sCmn
                                    .LBRR_flags
                                    .get_unchecked((i - 1) as usize)
                            } != 0
                        {
                            CODE_CONDITIONALLY
                        } else {
                            CODE_INDEPENDENTLY
                        };
                        silk_encode_indices(
                            &mut psEnc.state_Fxx[n as usize].sCmn,
                            psRangeEnc,
                            i,
                            1,
                            condCoding,
                        );
                        silk_encode_pulses(
                            psRangeEnc,
                            unsafe {
                                psEnc.state_Fxx[n as usize]
                                    .sCmn
                                    .indices_LBRR
                                    .get_unchecked(i as usize)
                                    .signalType as i32
                            },
                            unsafe {
                                psEnc.state_Fxx[n as usize]
                                    .sCmn
                                    .indices_LBRR
                                    .get_unchecked(i as usize)
                                    .quantOffsetType as i32
                            },
                            unsafe {
                                &mut *psEnc.state_Fxx[n as usize]
                                    .sCmn
                                    .pulses_LBRR
                                    .get_unchecked_mut(i as usize)
                            },
                            psEnc.state_Fxx[n as usize].sCmn.frame_length,
                        );
                    }
                    n += 1;
                }
                i += 1;
            }
            n = 0;
            while n < encControl.nChannelsInternal {
                unsafe { &mut *psEnc.state_Fxx.get_unchecked_mut(n as usize) }
                    .sCmn
                    .LBRR_flags = [0; 3];
                n += 1;
            }
            curr_nBitsUsedLBRR = ec_tell(psRangeEnc) - curr_nBitsUsedLBRR;
        }
        silk_HP_variable_cutoff(&mut psEnc.state_Fxx);
        nBits = encControl.bitRate * encControl.payloadSize_ms / 1000;
        if prefillFlag == 0 {
            // psEnc.nBitsUsedLBRR is an exponential moving average of the LBRR usage,
            // except that for the first LBRR frame it does no averaging and for the first
            // frame after LBRR, it goes back to zero immediately.
            if curr_nBitsUsedLBRR < 10 {
                psEnc.nBitsUsedLBRR = 0;
            } else if psEnc.nBitsUsedLBRR < 10 {
                psEnc.nBitsUsedLBRR = curr_nBitsUsedLBRR;
            } else {
                psEnc.nBitsUsedLBRR = (psEnc.nBitsUsedLBRR + curr_nBitsUsedLBRR) / 2;
            }
            nBits -= psEnc.nBitsUsedLBRR;
        }
        nBits /= psEnc.state_Fxx[0].sCmn.nFramesPerPacket;
        if encControl.payloadSize_ms == 10 {
            TargetRate_bps = nBits as i16 as i32 * 100;
        } else {
            TargetRate_bps = nBits as i16 as i32 * 50;
        }
        TargetRate_bps -= psEnc.nBitsExceeded * 1000 / BITRESERVOIR_DECAY_TIME_MS;
        if prefillFlag == 0 && psEnc.state_Fxx[0].sCmn.nFramesEncoded > 0 {
            let bitsBalance = ec_tell(psRangeEnc.as_mut().unwrap())
                - psEnc.nBitsUsedLBRR
                - nBits * psEnc.state_Fxx[0].sCmn.nFramesEncoded;
            TargetRate_bps -= bitsBalance * 1000 / BITRESERVOIR_DECAY_TIME_MS;
        }
        TargetRate_bps = if encControl.bitRate > 5000 {
            TargetRate_bps.clamp(5000, encControl.bitRate)
        } else {
            TargetRate_bps.clamp(encControl.bitRate, 5000)
        };
        if encControl.nChannelsInternal == 2 {
            {
                let frame_length = psEnc.state_Fxx[0].sCmn.frame_length;
                let nfe = psEnc.state_Fxx[0].sCmn.nFramesEncoded as usize;
                let speech_activity = psEnc.state_Fxx[0].sCmn.speech_activity_Q8;
                let fs_kHz = psEnc.state_Fxx[0].sCmn.fs_kHz;
                let frame_len_i32 = psEnc.state_Fxx[0].sCmn.frame_length as i32;
                // We need separate mutable borrows for the two channels, so split the array
                let [s0, s1] = &mut psEnc.state_Fxx;
                let x1 = &mut s0.sCmn.inputBuf[..frame_length + 2];
                let x2 = &mut s1.sCmn.inputBuf[..frame_length + 2];
                silk_stereo_LR_to_MS(
                    &mut psEnc.sStereo,
                    x1,
                    x2,
                    nfe,
                    &mut MStargetRates_bps,
                    TargetRate_bps,
                    speech_activity,
                    encControl.toMono,
                    fs_kHz,
                    frame_len_i32,
                );
            }
            if psEnc.sStereo.mid_only_flags[psEnc.state_Fxx[0].sCmn.nFramesEncoded as usize] as i32
                == 0
            {
                if psEnc.prev_decode_only_middle == 1 {
                    psEnc.state_Fxx[1].sShape = silk_shape_state_FLP::default();
                    psEnc.state_Fxx[1].sCmn.sNSQ = silk_nsq_state::default();
                    psEnc.state_Fxx[1].sCmn.prev_NLSFq_Q15 = [0; 16];
                    psEnc.state_Fxx[1].sCmn.sLP.In_LP_State = [0; 2];
                    psEnc.state_Fxx[1].sCmn.prevLag = 100;
                    psEnc.state_Fxx[1].sCmn.sNSQ.lagPrev = 100;
                    psEnc.state_Fxx[1].sShape.LastGainIndex = 10;
                    psEnc.state_Fxx[1].sCmn.prevSignalType = TYPE_NO_VOICE_ACTIVITY as i8;
                    psEnc.state_Fxx[1].sCmn.sNSQ.prev_gain_Q16 = 65536;
                    psEnc.state_Fxx[1].sCmn.first_frame_after_reset = 1;
                }
                silk_encode_do_VAD_FLP(&mut psEnc.state_Fxx[1], activity);
            } else {
                psEnc.state_Fxx[1].sCmn.VAD_flags
                    [psEnc.state_Fxx[0].sCmn.nFramesEncoded as usize] = 0;
            }
            if prefillFlag == 0 {
                let psRangeEnc = &mut **psRangeEnc.as_mut().unwrap();
                let nfe = psEnc.state_Fxx[0].sCmn.nFramesEncoded as usize;
                silk_stereo_encode_pred(psRangeEnc, &psEnc.sStereo.predIx[nfe]);
                if psEnc.state_Fxx[1].sCmn.VAD_flags[nfe] as i32 == 0 {
                    silk_stereo_encode_mid_only(psRangeEnc, psEnc.sStereo.mid_only_flags[nfe]);
                }
            }
        } else {
            let frame_length = psEnc.state_Fxx[0].sCmn.frame_length;
            psEnc.state_Fxx[0].sCmn.inputBuf[..2].copy_from_slice(&psEnc.sStereo.sMid);
            psEnc
                .sStereo
                .sMid
                .copy_from_slice(&psEnc.state_Fxx[0].sCmn.inputBuf[frame_length..frame_length + 2]);
        }
        silk_encode_do_VAD_FLP(&mut psEnc.state_Fxx[0], activity);
        n = 0;
        while n < encControl.nChannelsInternal {
            let mut maxBits: i32;
            let mut useCBR: i32;
            maxBits = encControl.maxBits;
            if tot_blocks == 2 && curr_block == 0 {
                maxBits = maxBits * 3 / 5;
            } else if tot_blocks == 3 {
                if curr_block == 0 {
                    maxBits = maxBits * 2 / 5;
                } else if curr_block == 1 {
                    maxBits = maxBits * 3 / 4;
                }
            }
            useCBR = (encControl.useCBR != 0 && curr_block == tot_blocks - 1) as i32;
            if encControl.nChannelsInternal == 1 {
                channelRate_bps = TargetRate_bps;
            } else {
                channelRate_bps = unsafe { *MStargetRates_bps.get_unchecked(n as usize) };
                if n == 0 && MStargetRates_bps[1] > 0 {
                    useCBR = 0;
                    maxBits -= encControl.maxBits / (tot_blocks * 2);
                }
            }
            if channelRate_bps > 0 {
                let condCoding_0: i32;
                silk_control_SNR(
                    &mut unsafe { &mut *psEnc.state_Fxx.get_unchecked_mut(n as usize) }.sCmn,
                    channelRate_bps,
                );
                if psEnc.state_Fxx[0].sCmn.nFramesEncoded - n <= 0 {
                    condCoding_0 = CODE_INDEPENDENTLY;
                } else if n > 0 && psEnc.prev_decode_only_middle != 0 {
                    condCoding_0 = CODE_INDEPENDENTLY_NO_LTP_SCALING;
                } else {
                    condCoding_0 = CODE_CONDITIONALLY;
                }
                let ps_range_enc = psRangeEnc.as_deref_mut();
                ret = silk_encode_frame_FLP(
                    unsafe { &mut *psEnc.state_Fxx.get_unchecked_mut(n as usize) },
                    nBytesOut,
                    ps_range_enc,
                    condCoding_0,
                    maxBits,
                    useCBR,
                );
                debug_assert_eq!(ret, 0);
            }
            let st = unsafe { &mut *psEnc.state_Fxx.get_unchecked_mut(n as usize) };
            st.sCmn.controlled_since_last_payload = 0;
            st.sCmn.inputBufIx = 0;
            st.sCmn.nFramesEncoded += 1;
            n += 1;
        }
        psEnc.prev_decode_only_middle = psEnc.sStereo.mid_only_flags
            [(psEnc.state_Fxx[0].sCmn.nFramesEncoded - 1) as usize]
            as i32;
        if *nBytesOut > 0
            && psEnc.state_Fxx[0].sCmn.nFramesEncoded == psEnc.state_Fxx[0].sCmn.nFramesPerPacket
        {
            flags = 0;
            n = 0;
            while n < encControl.nChannelsInternal {
                i = 0;
                while i < unsafe { &*psEnc.state_Fxx.get_unchecked(n as usize) }
                    .sCmn
                    .nFramesPerPacket
                {
                    flags = ((flags as u32) << 1) as i32;
                    flags |= unsafe {
                        *psEnc
                            .state_Fxx
                            .get_unchecked(n as usize)
                            .sCmn
                            .VAD_flags
                            .get_unchecked(i as usize)
                    } as i32;
                    i += 1;
                }
                flags = ((flags as u32) << 1) as i32;
                flags |= unsafe { &*psEnc.state_Fxx.get_unchecked(n as usize) }
                    .sCmn
                    .LBRR_flag as i32;
                n += 1;
            }
            if prefillFlag == 0 {
                ec_enc_patch_initial_bits(
                    psRangeEnc.as_mut().unwrap(),
                    flags as u32,
                    ((psEnc.state_Fxx[0].sCmn.nFramesPerPacket + 1) * encControl.nChannelsInternal)
                        as u32,
                );
            }
            if psEnc.state_Fxx[0].sCmn.inDTX != 0
                && (encControl.nChannelsInternal == 1 || psEnc.state_Fxx[1].sCmn.inDTX != 0)
            {
                *nBytesOut = 0;
            }
            psEnc.nBitsExceeded += *nBytesOut * 8;
            psEnc.nBitsExceeded -= encControl.bitRate * encControl.payloadSize_ms / 1000;
            psEnc.nBitsExceeded = psEnc.nBitsExceeded.clamp(0, 10000);
            let speech_act_thr_for_switch_Q8 =
                (((SPEECH_ACTIVITY_DTX_THRES * (1 << 8) as f32) as f64 + 0.5f64) as i32 as i64
                    + (((((1_f32 - SPEECH_ACTIVITY_DTX_THRES) / MAX_BANDWIDTH_SWITCH_DELAY_MS
                        * (1 << (16 + 8)) as f32) as f64
                        + 0.5f64) as i32 as i64
                        * psEnc.timeSinceSwitchAllowed_ms as i16 as i64)
                        >> 16)) as i32;
            if psEnc.state_Fxx[0].sCmn.speech_activity_Q8 < speech_act_thr_for_switch_Q8 {
                psEnc.allowBandwidthSwitch = 1;
                psEnc.timeSinceSwitchAllowed_ms = 0;
            } else {
                psEnc.allowBandwidthSwitch = 0;
                psEnc.timeSinceSwitchAllowed_ms += encControl.payloadSize_ms;
            }
        }
        if nSamplesIn == 0 {
            break;
        }
        curr_block += 1;
    }
    psEnc.nPrevChannelsInternal = encControl.nChannelsInternal;
    encControl.allowBandwidthSwitch = psEnc.allowBandwidthSwitch;
    encControl.inWBmodeWithoutVariableLP =
        (psEnc.state_Fxx[0].sCmn.fs_kHz == 16 && psEnc.state_Fxx[0].sCmn.sLP.mode == 0) as i32;
    encControl.internalSampleRate = psEnc.state_Fxx[0].sCmn.fs_kHz as i16 as i32 * 1000;
    encControl.stereoWidth_Q14 = if encControl.toMono != 0 {
        0
    } else {
        psEnc.sStereo.smth_width_Q14 as i32
    };
    if prefillFlag != 0 {
        encControl.payloadSize_ms = tmp_payloadSize_ms;
        encControl.complexity = tmp_complexity;
        n = 0;
        while n < encControl.nChannelsInternal {
            let st = unsafe { &mut *psEnc.state_Fxx.get_unchecked_mut(n as usize) };
            st.sCmn.controlled_since_last_payload = 0;
            st.sCmn.prefillFlag = 0;
            n += 1;
        }
    }
    encControl.signalType = psEnc.state_Fxx[0].sCmn.indices.signalType as i32;
    encControl.offset = unsafe {
        *silk_Quantization_Offsets_Q10
            .get_unchecked((psEnc.state_Fxx[0].sCmn.indices.signalType as i32 >> 1) as usize)
            .get_unchecked(psEnc.state_Fxx[0].sCmn.indices.quantOffsetType as usize)
    } as i32;
    ret
}
