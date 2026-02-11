//! SILK decoder API.
//!
//! Upstream C: `silk/dec_API.c`

#[derive(Copy, Clone)]
#[repr(C)]
pub struct silk_DecControlStruct {
    pub nChannelsAPI: usize,
    pub nChannelsInternal: i32,
    pub API_sampleRate: i32,
    pub internalSampleRate: i32,
    pub payloadSize_ms: i32,
    pub prevPitchLag: i32,
}
pub const FLAG_DECODE_NORMAL: i32 = 0;
pub const FLAG_DECODE_LBRR: i32 = 2;
pub const FLAG_PACKET_LOST: i32 = 1;
pub mod errors_h {
    pub const SILK_DEC_INVALID_SAMPLING_FREQUENCY: i32 = -(200);
    pub const SILK_NO_ERROR: i32 = 0;
    #[allow(unused)]
    pub const SILK_DEC_INVALID_FRAME_SIZE: i32 = -(203);
}
use self::errors_h::{SILK_DEC_INVALID_SAMPLING_FREQUENCY, SILK_NO_ERROR};
use crate::celt::entdec::{ec_dec, ec_dec_bit_logp, ec_dec_icdf};
use crate::silk::decode_frame::silk_decode_frame;
use crate::silk::decode_indices::silk_decode_indices;
use crate::silk::decode_pulses::silk_decode_pulses;
use crate::silk::decoder_set_fs::silk_decoder_set_fs;
use crate::silk::define::{
    CODE_CONDITIONALLY, CODE_INDEPENDENTLY, CODE_INDEPENDENTLY_NO_LTP_SCALING, MAX_API_FS_KHZ,
    SHELL_CODEC_FRAME_LENGTH, TYPE_NO_VOICE_ACTIVITY, TYPE_VOICED,
};
use crate::silk::init_decoder::{silk_init_decoder, silk_reset_decoder};
use crate::silk::resampler::silk_resampler;
use crate::silk::stereo_MS_to_LR::silk_stereo_MS_to_LR;
use crate::silk::stereo_decode_pred::{silk_stereo_decode_mid_only, silk_stereo_decode_pred};
use crate::silk::structs::{silk_decoder_state, stereo_dec_state};
use crate::silk::tables_other::silk_LBRR_flags_iCDF_ptr;

#[derive(Copy, Clone)]
#[repr(C)]
pub struct silk_decoder {
    pub channel_state: [silk_decoder_state; 2],
    pub sStereo: stereo_dec_state,
    pub nChannelsAPI: i32,
    pub nChannelsInternal: i32,
    pub prev_decode_only_middle: bool,
}
pub fn silk_InitDecoder() -> silk_decoder {
    silk_decoder {
        channel_state: [silk_init_decoder(), silk_init_decoder()],
        sStereo: stereo_dec_state::default(),
        nChannelsAPI: 0,
        nChannelsInternal: 0,
        prev_decode_only_middle: false,
    }
}

/// Reset decoder state without full reinitialization.
///
/// Upstream C: silk/dec_API.c:silk_ResetDecoder
pub fn silk_ResetDecoder(dec: &mut silk_decoder) {
    for ch in dec.channel_state.iter_mut() {
        silk_reset_decoder(ch);
    }
    dec.sStereo = stereo_dec_state::default();
    dec.prev_decode_only_middle = false;
}

/// Upstream C: silk/dec_API.c:silk_Decode
pub fn silk_Decode(
    decState: &mut silk_decoder,
    decControl: &mut silk_DecControlStruct,
    lostFlag: i32,
    newPacketFlag: i32,
    psRangeDec: &mut ec_dec,
    samplesOut: &mut [i16],
    nSamplesOut: &mut i32,
    arch: i32,
) -> i32 {
    let mut i: i32;
    let mut n: i32;
    let mut decode_only_middle: bool = false;
    let mut ret: i32 = SILK_NO_ERROR;
    let mut nSamplesOutDec: i32 = 0;
    let mut LBRR_symbol: i32;
    let mut MS_pred_Q13: [i32; 2] = [0, 0];
    let psDec = decState;
    let channel_state = &mut psDec.channel_state;

    assert!(decControl.nChannelsInternal == 1 || decControl.nChannelsInternal == 2);
    if newPacketFlag != 0 {
        n = 0;
        while n < decControl.nChannelsInternal {
            channel_state[n as usize].nFramesDecoded = 0;
            n += 1;
        }
    }
    if decControl.nChannelsInternal > psDec.nChannelsInternal {
        channel_state[1] = silk_init_decoder();
    }
    let stereo_to_mono: i32 = (decControl.nChannelsInternal == 1
        && psDec.nChannelsInternal == 2
        && decControl.internalSampleRate == 1000 * (channel_state[0]).fs_kHz)
        as i32;
    if (channel_state[0]).nFramesDecoded == 0 {
        n = 0;
        while n < decControl.nChannelsInternal {
            if decControl.payloadSize_ms == 0 || decControl.payloadSize_ms == 10 {
                channel_state[n as usize].nFramesPerPacket = 1;
                channel_state[n as usize].nb_subfr = 2;
            } else if decControl.payloadSize_ms == 20 {
                channel_state[n as usize].nFramesPerPacket = 1;
                channel_state[n as usize].nb_subfr = 4;
            } else if decControl.payloadSize_ms == 40 {
                channel_state[n as usize].nFramesPerPacket = 2;
                channel_state[n as usize].nb_subfr = 4;
            } else if decControl.payloadSize_ms == 60 {
                channel_state[n as usize].nFramesPerPacket = 3;
                channel_state[n as usize].nb_subfr = 4;
            } else {
                // see comments in `[opurs::silk::check_control_input]`
                panic!("libopus: assert(0) called");
            }
            let fs_kHz_dec: i32 = (decControl.internalSampleRate >> 10) + 1;
            if fs_kHz_dec != 8 && fs_kHz_dec != 12 && fs_kHz_dec != 16 {
                panic!("libopus: assert(0) called");
            }
            ret += silk_decoder_set_fs(
                &mut channel_state[n as usize],
                fs_kHz_dec,
                decControl.API_sampleRate,
            );
            n += 1;
        }
    }
    if decControl.nChannelsAPI == 2
        && decControl.nChannelsInternal == 2
        && (psDec.nChannelsAPI == 1 || psDec.nChannelsInternal == 1)
    {
        psDec.sStereo.pred_prev_Q13.fill(0);
        psDec.sStereo.sSide.fill(0);
        channel_state[1].resampler_state = channel_state[0].resampler_state;
    }
    psDec.nChannelsAPI = decControl.nChannelsAPI as i32;
    psDec.nChannelsInternal = decControl.nChannelsInternal;
    if decControl.API_sampleRate > MAX_API_FS_KHZ * 1000 || decControl.API_sampleRate < 8000 {
        ret = SILK_DEC_INVALID_SAMPLING_FREQUENCY;
        return ret;
    }
    if lostFlag != FLAG_PACKET_LOST && (channel_state[0]).nFramesDecoded == 0 {
        n = 0;
        while n < decControl.nChannelsInternal {
            i = 0;
            while i < channel_state[n as usize].nFramesPerPacket {
                channel_state[n as usize].VAD_flags[i as usize] = ec_dec_bit_logp(psRangeDec, 1);
                i += 1;
            }
            channel_state[n as usize].LBRR_flag = ec_dec_bit_logp(psRangeDec, 1);
            n += 1;
        }
        n = 0;
        while n < decControl.nChannelsInternal {
            channel_state[n as usize].LBRR_flags.fill(0);
            if channel_state[n as usize].LBRR_flag != 0 {
                if channel_state[n as usize].nFramesPerPacket == 1 {
                    channel_state[n as usize].LBRR_flags[0] = 1;
                } else {
                    LBRR_symbol = ec_dec_icdf(
                        psRangeDec,
                        silk_LBRR_flags_iCDF_ptr
                            [(channel_state[n as usize].nFramesPerPacket - 2) as usize],
                        8,
                    ) + 1;
                    i = 0;
                    while i < channel_state[n as usize].nFramesPerPacket {
                        channel_state[n as usize].LBRR_flags[i as usize] = LBRR_symbol >> i & 1;
                        i += 1;
                    }
                }
            }
            n += 1;
        }
        if lostFlag == FLAG_DECODE_NORMAL {
            i = 0;
            while i < channel_state[0].nFramesPerPacket {
                n = 0;
                while n < decControl.nChannelsInternal {
                    if channel_state[n as usize].LBRR_flags[i as usize] != 0 {
                        let mut pulses: [i16; 320] = [0; 320];
                        if decControl.nChannelsInternal == 2 && n == 0 {
                            silk_stereo_decode_pred(psRangeDec, &mut MS_pred_Q13);
                            if channel_state[1].LBRR_flags[i as usize] == 0 {
                                silk_stereo_decode_mid_only(psRangeDec, &mut decode_only_middle);
                            }
                        }
                        let condCoding: i32 = if i > 0
                            && channel_state[n as usize].LBRR_flags[(i - 1) as usize] != 0
                        {
                            CODE_CONDITIONALLY
                        } else {
                            CODE_INDEPENDENTLY
                        };
                        silk_decode_indices(
                            &mut channel_state[n as usize],
                            psRangeDec,
                            i,
                            1,
                            condCoding,
                        );

                        let frame_length = channel_state[n as usize].frame_length;
                        let mut shell_frames = frame_length / SHELL_CODEC_FRAME_LENGTH;
                        if shell_frames * SHELL_CODEC_FRAME_LENGTH < frame_length {
                            assert_eq!(frame_length, 12 * 10);
                            shell_frames += 1;
                        }
                        let frame_buffer_length = shell_frames * SHELL_CODEC_FRAME_LENGTH;

                        silk_decode_pulses(
                            psRangeDec,
                            &mut pulses[..frame_buffer_length],
                            channel_state[n as usize].indices.signalType as i32,
                            channel_state[n as usize].indices.quantOffsetType as i32,
                        );
                    }
                    n += 1;
                }
                i += 1;
            }
        }
    }
    if decControl.nChannelsInternal == 2 {
        if lostFlag == FLAG_DECODE_NORMAL
            || lostFlag == FLAG_DECODE_LBRR
                && channel_state[0].LBRR_flags[channel_state[0].nFramesDecoded as usize] == 1
        {
            silk_stereo_decode_pred(psRangeDec, &mut MS_pred_Q13);
            if lostFlag == FLAG_DECODE_NORMAL
                && channel_state[1].VAD_flags[channel_state[0].nFramesDecoded as usize] == 0
                || lostFlag == FLAG_DECODE_LBRR
                    && channel_state[1].LBRR_flags[channel_state[0].nFramesDecoded as usize] == 0
            {
                silk_stereo_decode_mid_only(psRangeDec, &mut decode_only_middle);
            } else {
                decode_only_middle = false;
            }
        } else {
            n = 0;
            while n < 2 {
                MS_pred_Q13[n as usize] = psDec.sStereo.pred_prev_Q13[n as usize] as i32;
                n += 1;
            }
        }
    }
    if decControl.nChannelsInternal == 2 && !decode_only_middle && psDec.prev_decode_only_middle {
        channel_state[1].outBuf.fill(0);
        channel_state[1].sLPC_Q14_buf.fill(0);
        channel_state[1].lagPrev = 100;
        channel_state[1].LastGainIndex = 10;
        channel_state[1].prevSignalType = TYPE_NO_VOICE_ACTIVITY;
        channel_state[1].first_frame_after_reset = 1;
    }

    // Temporary buffers for decoded samples. Each channel needs frame_length + 2 elements.
    // The first 2 elements are stereo prediction state, decoded samples start at offset 2.
    let frame_len = channel_state[0].frame_length;
    let ch_buf_len = frame_len + 2;
    let nChannelsInt = decControl.nChannelsInternal;

    // Always allocate the temp storage (simplifies logic vs. the C "delay_stack_alloc" trick)
    let mut samplesOut1_tmp_storage: Vec<i16> = vec![0; nChannelsInt as usize * ch_buf_len];

    // Channel offsets into samplesOut1_tmp_storage
    let ch0_off: usize = 0;
    let ch1_off: usize = ch_buf_len;

    let has_side: i32 = if lostFlag == FLAG_DECODE_NORMAL {
        (!decode_only_middle) as i32
    } else {
        (!psDec.prev_decode_only_middle
            || decControl.nChannelsInternal == 2
                && lostFlag == FLAG_DECODE_LBRR
                && channel_state[1].LBRR_flags[channel_state[1].nFramesDecoded as usize] == 1)
            as i32
    };
    n = 0;
    while n < decControl.nChannelsInternal {
        if n == 0 || has_side != 0 {
            let condCoding_0: i32;
            let FrameIndex: i32 = channel_state[0].nFramesDecoded - n;
            if FrameIndex <= 0 {
                condCoding_0 = CODE_INDEPENDENTLY;
            } else if lostFlag == FLAG_DECODE_LBRR {
                condCoding_0 =
                    if channel_state[n as usize].LBRR_flags[(FrameIndex - 1) as usize] != 0 {
                        CODE_CONDITIONALLY
                    } else {
                        CODE_INDEPENDENTLY
                    };
            } else if n > 0 && psDec.prev_decode_only_middle {
                condCoding_0 = CODE_INDEPENDENTLY_NO_LTP_SCALING;
            } else {
                condCoding_0 = CODE_CONDITIONALLY;
            }
            let ch_off = if n == 0 { ch0_off } else { ch1_off };
            let out_slice = &mut samplesOut1_tmp_storage[ch_off + 2..ch_off + 2 + frame_len];
            let (err, n_out) = silk_decode_frame(
                &mut channel_state[n as usize],
                psRangeDec,
                out_slice,
                lostFlag,
                condCoding_0,
                arch,
            );
            ret += err;
            nSamplesOutDec = n_out;
        } else {
            let ch_off = if n == 0 { ch0_off } else { ch1_off };
            samplesOut1_tmp_storage[ch_off + 2..ch_off + 2 + nSamplesOutDec as usize].fill(0);
        }
        channel_state[n as usize].nFramesDecoded += 1;
        n += 1;
    }

    if decControl.nChannelsAPI == 2 && decControl.nChannelsInternal == 2 {
        let (ch0_slice, ch1_slice) = samplesOut1_tmp_storage.split_at_mut(ch1_off);
        silk_stereo_MS_to_LR(
            &mut psDec.sStereo,
            &mut ch0_slice[ch0_off..ch0_off + nSamplesOutDec as usize + 2],
            &mut ch1_slice[..nSamplesOutDec as usize + 2],
            &MS_pred_Q13,
            channel_state[0].fs_kHz as usize,
            nSamplesOutDec,
        );
    } else {
        // Copy sMid[0..2] to beginning of channel 0 buffer
        samplesOut1_tmp_storage[ch0_off] = psDec.sStereo.sMid[0];
        samplesOut1_tmp_storage[ch0_off + 1] = psDec.sStereo.sMid[1];
        // Save last 2 samples back to sMid
        psDec.sStereo.sMid[0] = samplesOut1_tmp_storage[ch0_off + nSamplesOutDec as usize];
        psDec.sStereo.sMid[1] = samplesOut1_tmp_storage[ch0_off + nSamplesOutDec as usize + 1];
    }

    *nSamplesOut =
        nSamplesOutDec * decControl.API_sampleRate / (channel_state[0].fs_kHz as i16 as i32 * 1000);

    let mut samplesOut2_tmp: Vec<i16> = vec![
        0;
        if decControl.nChannelsAPI == 2 {
            *nSamplesOut as usize
        } else {
            1
        }
    ];

    n = 0;
    while n
        < (if (decControl.nChannelsAPI as i32) < decControl.nChannelsInternal {
            decControl.nChannelsAPI as i32
        } else {
            decControl.nChannelsInternal
        })
    {
        let ch_off = if n == 0 { ch0_off } else { ch1_off };
        let resample_input =
            &samplesOut1_tmp_storage[ch_off + 1..ch_off + 1 + nSamplesOutDec as usize];

        if decControl.nChannelsAPI == 2 {
            ret += silk_resampler(
                &mut channel_state[n as usize].resampler_state,
                &mut samplesOut2_tmp,
                resample_input,
            );
            i = 0;
            while i < *nSamplesOut {
                samplesOut[(n + 2 * i) as usize] = samplesOut2_tmp[i as usize];
                i += 1;
            }
        } else {
            ret += silk_resampler(
                &mut channel_state[n as usize].resampler_state,
                &mut samplesOut[..*nSamplesOut as usize],
                resample_input,
            );
        }
        n += 1;
    }

    // Create two channel output from mono stream
    if decControl.nChannelsAPI == 2 && decControl.nChannelsInternal == 1 {
        if stereo_to_mono != 0 {
            let resample_input =
                &samplesOut1_tmp_storage[ch0_off + 1..ch0_off + 1 + nSamplesOutDec as usize];
            ret += silk_resampler(
                &mut channel_state[1].resampler_state,
                &mut samplesOut2_tmp,
                resample_input,
            );
            i = 0;
            while i < *nSamplesOut {
                samplesOut[(1 + 2 * i) as usize] = samplesOut2_tmp[i as usize];
                i += 1;
            }
        } else {
            i = 0;
            while i < *nSamplesOut {
                samplesOut[(1 + 2 * i) as usize] = samplesOut[(2 * i) as usize];
                i += 1;
            }
        }
    }

    if channel_state[0].prevSignalType == TYPE_VOICED {
        let mult_tab: [i32; 3] = [6, 4, 3];
        decControl.prevPitchLag =
            channel_state[0].lagPrev * mult_tab[((channel_state[0].fs_kHz - 8) >> 2) as usize];
    } else {
        decControl.prevPitchLag = 0;
    }

    if lostFlag == FLAG_PACKET_LOST {
        i = 0;
        while i < psDec.nChannelsInternal {
            psDec.channel_state[i as usize].LastGainIndex = 10;
            i += 1;
        }
    } else {
        psDec.prev_decode_only_middle = decode_only_middle;
    }
    ret
}
