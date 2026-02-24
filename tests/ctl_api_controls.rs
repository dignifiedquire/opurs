#[cfg(all(feature = "tools", feature = "qext"))]
use opurs::internals::{
    ec_dec_bit_logp as r_ec_dec_bit_logp, ec_dec_init as r_ec_dec_init,
    ec_dec_uint as r_ec_dec_uint, ec_tell_frac as r_ec_tell_frac,
};
#[cfg(feature = "qext")]
use opurs::internals::{
    opus_packet_extensions_generate, opus_packet_extensions_parse, opus_packet_parse_impl,
    OpusExtensionData,
};
#[cfg(feature = "qext")]
use opurs::{opus_multistream_packet_unpad, opus_packet_pad, opus_packet_unpad};
#[cfg(any(feature = "qext", feature = "osce"))]
use opurs::{Bitrate, OpusEncoder, OpusMSEncoder};
use opurs::{
    OpusDecoder, OpusMSDecoder, OpusProjectionDecoder, OpusProjectionEncoder,
    OPUS_APPLICATION_AUDIO,
};
#[cfg(feature = "osce")]
use opurs::{
    OPUS_APPLICATION_RESTRICTED_LOWDELAY, OPUS_APPLICATION_RESTRICTED_SILK, OPUS_APPLICATION_VOIP,
};
#[cfg(all(feature = "tools", feature = "qext"))]
use opurs::{OPUS_GET_FINAL_RANGE_REQUEST, OPUS_SET_IGNORE_EXTENSIONS_REQUEST};
#[cfg(all(feature = "tools", feature = "qext"))]
use std::ffi::c_void;

#[cfg(all(feature = "tools", feature = "qext"))]
use libopus_sys::{
    opus_decode as c_opus_decode, opus_decoder_create as c_opus_decoder_create,
    opus_decoder_ctl as c_opus_decoder_ctl, opus_decoder_destroy as c_opus_decoder_destroy,
    opus_multistream_decode as c_opus_multistream_decode,
    opus_multistream_decoder_create as c_opus_multistream_decoder_create,
    opus_multistream_decoder_ctl as c_opus_multistream_decoder_ctl,
    opus_multistream_decoder_destroy as c_opus_multistream_decoder_destroy,
};

#[cfg(all(feature = "tools", feature = "qext"))]
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct COpusExtensionData {
    id: i32,
    frame: i32,
    data: *const u8,
    len: i32,
}

#[cfg(all(feature = "tools", feature = "qext"))]
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct CEcCtx {
    buf: *mut u8,
    storage: u32,
    end_offs: u32,
    end_window: u32,
    nend_bits: i32,
    nbits_total: i32,
    offs: u32,
    rng: u32,
    val: u32,
    ext: u32,
    rem: i32,
    error: i32,
}

#[cfg(all(feature = "tools", feature = "qext"))]
unsafe extern "C" {
    #[link_name = "opus_packet_extensions_parse"]
    fn c_opus_packet_extensions_parse(
        data: *const u8,
        len: i32,
        extensions: *mut COpusExtensionData,
        nb_extensions: *mut i32,
        nb_frames: i32,
    ) -> i32;
    #[link_name = "ec_dec_init"]
    fn c_ec_dec_init(dec: *mut CEcCtx, buf: *mut u8, storage: u32);
    #[link_name = "ec_dec_bit_logp"]
    fn c_ec_dec_bit_logp(dec: *mut CEcCtx, logp: u32) -> i32;
    #[link_name = "ec_dec_uint"]
    fn c_ec_dec_uint(dec: *mut CEcCtx, ft: u32) -> u32;
    #[link_name = "ec_tell_frac"]
    fn c_ec_tell_frac(dec: *mut CEcCtx) -> u32;
    fn opus_projection_decoder_create(
        Fs: i32,
        channels: i32,
        streams: i32,
        coupled_streams: i32,
        demixing_matrix: *const u8,
        demixing_matrix_size: i32,
        error: *mut i32,
    ) -> *mut c_void;
    fn opus_projection_decoder_ctl(st: *mut c_void, request: i32, ...) -> i32;
    fn opus_projection_decode(
        st: *mut c_void,
        data: *const u8,
        len: i32,
        pcm: *mut i16,
        frame_size: i32,
        decode_fec: i32,
    ) -> i32;
    fn opus_projection_decoder_destroy(st: *mut c_void);
}

#[cfg(feature = "qext")]
const SAMPLE_RATE_96K: i32 = 96_000;
#[cfg(feature = "qext")]
const FRAME_SIZE_20MS_96K: i32 = 1920;
#[cfg(feature = "qext")]
const QEXT_EXTENSION_ID: i32 = 124;

#[cfg(feature = "qext")]
fn stereo_pcm(seed: u32) -> Vec<i16> {
    let samples = FRAME_SIZE_20MS_96K as usize * 2;
    (0..samples)
        .map(|i| {
            let x = seed.wrapping_mul(1103515245).wrapping_add(i as u32 * 12345);
            ((x >> 7) as i16).wrapping_sub(16384)
        })
        .collect()
}

#[cfg(feature = "qext")]
fn quad_pcm(seed: u32) -> Vec<i16> {
    let samples = FRAME_SIZE_20MS_96K as usize * 4;
    (0..samples)
        .map(|i| {
            let x = seed
                .wrapping_mul(1664525)
                .wrapping_add(i as u32 * 1013904223);
            ((x >> 8) as i16).wrapping_sub(8192)
        })
        .collect()
}

#[cfg(feature = "qext")]
fn parse_padding_extensions(packet: &[u8]) -> Vec<i32> {
    let mut toc = 0u8;
    let mut sizes = [0i16; 48];
    let mut packet_offset = 0i32;
    let mut padding_len = 0i32;

    let nb_frames = opus_packet_parse_impl(
        packet,
        false,
        Some(&mut toc),
        None,
        &mut sizes,
        None,
        Some(&mut packet_offset),
        Some(&mut padding_len),
    );
    assert!(nb_frames > 0, "failed to parse opus packet");
    assert!(padding_len > 0, "packet has no padding/extensions");

    let padding_start = (packet_offset - padding_len) as usize;
    let padding_end = packet_offset as usize;
    let exts = opus_packet_extensions_parse(&packet[padding_start..padding_end], 64, nb_frames)
        .expect("failed to parse packet extensions");

    exts.into_iter().map(|e| e.id).collect()
}

#[cfg(feature = "qext")]
fn first_qext_payload(packet: &[u8]) -> Option<Vec<u8>> {
    let mut toc = 0u8;
    let mut sizes = [0i16; 48];
    let mut packet_offset = 0i32;
    let mut padding_len = 0i32;
    let nb_frames = opus_packet_parse_impl(
        packet,
        false,
        Some(&mut toc),
        None,
        &mut sizes,
        None,
        Some(&mut packet_offset),
        Some(&mut padding_len),
    );
    if nb_frames <= 0 || padding_len <= 0 {
        return None;
    }
    let padding_start = (packet_offset - padding_len) as usize;
    let padding_end = packet_offset as usize;
    let exts =
        opus_packet_extensions_parse(&packet[padding_start..padding_end], 64, nb_frames).ok()?;
    exts.into_iter()
        .find(|e| e.id == QEXT_EXTENSION_ID && e.frame == 0)
        .map(|e| e.data)
}

#[cfg(all(feature = "tools", feature = "qext"))]
fn first_qext_payload_c(packet: &[u8]) -> Option<Vec<u8>> {
    let mut toc = 0u8;
    let mut sizes = [0i16; 48];
    let mut packet_offset = 0i32;
    let mut padding_len = 0i32;
    let nb_frames = opus_packet_parse_impl(
        packet,
        false,
        Some(&mut toc),
        None,
        &mut sizes,
        None,
        Some(&mut packet_offset),
        Some(&mut padding_len),
    );
    if nb_frames <= 0 || padding_len <= 0 {
        return None;
    }
    let padding_start = (packet_offset - padding_len) as usize;
    let padding_end = packet_offset as usize;
    let padding = &packet[padding_start..padding_end];
    let mut exts = [COpusExtensionData::default(); 64];
    let mut nb_exts = exts.len() as i32;
    let ret = unsafe {
        c_opus_packet_extensions_parse(
            padding.as_ptr(),
            padding.len() as i32,
            exts.as_mut_ptr(),
            &mut nb_exts,
            nb_frames,
        )
    };
    if ret != 0 || nb_exts < 0 {
        return None;
    }
    for ext in exts.iter().take(nb_exts as usize) {
        if ext.id == QEXT_EXTENSION_ID && ext.frame == 0 && !ext.data.is_null() && ext.len >= 0 {
            let bytes = unsafe { std::slice::from_raw_parts(ext.data, ext.len as usize) }.to_vec();
            return Some(bytes);
        }
    }
    None
}

#[cfg(all(feature = "tools", feature = "qext"))]
fn ilog_u32(v: u32) -> i32 {
    32 - v.leading_zeros() as i32
}

#[cfg(all(feature = "tools", feature = "qext"))]
fn decode_qext_header_rust(payload: &[u8]) -> (i32, i32, i32, i32, i32, u32) {
    let mut buf = payload.to_vec();
    let mut dec = r_ec_dec_init(&mut buf);
    let qext_end = if r_ec_dec_bit_logp(&mut dec, 1) != 0 {
        14
    } else {
        2
    };
    let qext_intensity = r_ec_dec_uint(&mut dec, (qext_end + 1) as u32) as i32;
    let qext_dual = if qext_intensity != 0 {
        r_ec_dec_bit_logp(&mut dec, 1)
    } else {
        0
    };
    let tell = dec.nbits_total - ilog_u32(dec.rng);
    let qext_intra = if tell + 3 <= payload.len() as i32 * 8 {
        r_ec_dec_bit_logp(&mut dec, 3)
    } else {
        0
    };
    (
        qext_end,
        qext_intensity,
        qext_dual,
        qext_intra,
        r_ec_tell_frac(&dec) as i32,
        dec.rng,
    )
}

#[cfg(all(feature = "tools", feature = "qext"))]
fn decode_qext_header_c(payload: &[u8]) -> (i32, i32, i32, i32, i32, u32) {
    let mut buf = payload.to_vec();
    let mut dec = CEcCtx::default();
    unsafe {
        c_ec_dec_init(&mut dec, buf.as_mut_ptr(), buf.len() as u32);
    }
    let qext_end = if unsafe { c_ec_dec_bit_logp(&mut dec, 1) } != 0 {
        14
    } else {
        2
    };
    let qext_intensity = unsafe { c_ec_dec_uint(&mut dec, (qext_end + 1) as u32) } as i32;
    let qext_dual = if qext_intensity != 0 {
        unsafe { c_ec_dec_bit_logp(&mut dec, 1) }
    } else {
        0
    };
    let tell = dec.nbits_total - ilog_u32(dec.rng);
    let qext_intra = if tell + 3 <= payload.len() as i32 * 8 {
        unsafe { c_ec_dec_bit_logp(&mut dec, 3) }
    } else {
        0
    };
    (
        qext_end,
        qext_intensity,
        qext_dual,
        qext_intra,
        unsafe { c_ec_tell_frac(&mut dec) } as i32,
        dec.rng,
    )
}

#[cfg(feature = "qext")]
fn decode_single_raw(packet: &[u8], ignore_extensions: bool) -> (i32, Vec<i16>, u32) {
    let mut dec = OpusDecoder::new(SAMPLE_RATE_96K, 2).expect("decoder create");
    dec.set_ignore_extensions(ignore_extensions);

    let mut pcm = vec![0i16; FRAME_SIZE_20MS_96K as usize * 2];
    let ret = dec.decode(packet, &mut pcm, FRAME_SIZE_20MS_96K, false);
    (ret, pcm, dec.final_range())
}

#[cfg(all(feature = "tools", feature = "qext"))]
fn decode_single_c(packet: &[u8], ignore_extensions: bool) -> (i32, Vec<i16>, u32) {
    let mut err = 0i32;
    let dec = unsafe { c_opus_decoder_create(SAMPLE_RATE_96K, 2, &mut err as *mut _) };
    assert!(!dec.is_null(), "c decoder create failed: {err}");
    if ignore_extensions {
        let set_ret = unsafe { c_opus_decoder_ctl(dec, OPUS_SET_IGNORE_EXTENSIONS_REQUEST, 1i32) };
        assert_eq!(set_ret, 0, "c set ignore_extensions failed: {set_ret}");
    }
    let mut pcm = vec![0i16; FRAME_SIZE_20MS_96K as usize * 2];
    let ret = unsafe {
        c_opus_decode(
            dec,
            packet.as_ptr(),
            packet.len() as i32,
            pcm.as_mut_ptr(),
            FRAME_SIZE_20MS_96K,
            0,
        )
    };
    let mut rng = 0u32;
    let rng_ret =
        unsafe { c_opus_decoder_ctl(dec, OPUS_GET_FINAL_RANGE_REQUEST, &mut rng as *mut _) };
    assert_eq!(rng_ret, 0, "c get final range failed: {rng_ret}");
    unsafe { c_opus_decoder_destroy(dec) };
    (ret, pcm, rng)
}

#[cfg(feature = "qext")]
fn decode_single(packet: &[u8], ignore_extensions: bool) -> (Vec<i16>, u32) {
    let (ret, pcm, final_range) = decode_single_raw(packet, ignore_extensions);
    assert_eq!(ret, FRAME_SIZE_20MS_96K, "single decode failed");
    (pcm, final_range)
}

#[cfg(feature = "qext")]
fn decode_ms_raw(packet: &[u8], ignore_extensions: bool) -> (i32, Vec<i16>, u32) {
    let mut dec = OpusMSDecoder::new(SAMPLE_RATE_96K, 2, 1, 1, &[0, 1]).expect("ms decoder create");
    dec.set_ignore_extensions(ignore_extensions);

    let mut pcm = vec![0i16; FRAME_SIZE_20MS_96K as usize * 2];
    let ret = dec.decode(packet, &mut pcm, FRAME_SIZE_20MS_96K, false);
    (ret, pcm, dec.final_range())
}

#[cfg(all(feature = "tools", feature = "qext"))]
fn decode_ms_c(packet: &[u8], ignore_extensions: bool) -> (i32, Vec<i16>, u32) {
    let mut err = 0i32;
    let mapping = [0u8, 1u8];
    let dec = unsafe {
        c_opus_multistream_decoder_create(SAMPLE_RATE_96K, 2, 1, 1, mapping.as_ptr(), &mut err)
    };
    assert!(!dec.is_null(), "c ms decoder create failed: {err}");
    if ignore_extensions {
        let set_ret = unsafe {
            c_opus_multistream_decoder_ctl(dec, OPUS_SET_IGNORE_EXTENSIONS_REQUEST, 1i32)
        };
        assert_eq!(set_ret, 0, "c ms set ignore_extensions failed: {set_ret}");
    }
    let mut pcm = vec![0i16; FRAME_SIZE_20MS_96K as usize * 2];
    let ret = unsafe {
        c_opus_multistream_decode(
            dec,
            packet.as_ptr(),
            packet.len() as i32,
            pcm.as_mut_ptr(),
            FRAME_SIZE_20MS_96K,
            0,
        )
    };
    let mut rng = 0u32;
    let rng_ret = unsafe {
        c_opus_multistream_decoder_ctl(dec, OPUS_GET_FINAL_RANGE_REQUEST, &mut rng as *mut _)
    };
    assert_eq!(rng_ret, 0, "c ms get final range failed: {rng_ret}");
    unsafe { c_opus_multistream_decoder_destroy(dec) };
    (ret, pcm, rng)
}

#[cfg(feature = "qext")]
fn decode_ms(packet: &[u8], ignore_extensions: bool) -> (Vec<i16>, u32) {
    let (ret, pcm, final_range) = decode_ms_raw(packet, ignore_extensions);
    assert_eq!(ret, FRAME_SIZE_20MS_96K, "multistream decode failed");
    (pcm, final_range)
}

#[cfg(feature = "qext")]
fn make_projection_codec_pair() -> (OpusProjectionEncoder, i32, i32, Vec<u8>) {
    let mut streams = 0;
    let mut coupled_streams = 0;
    let enc = OpusProjectionEncoder::new(
        SAMPLE_RATE_96K,
        4,
        3,
        &mut streams,
        &mut coupled_streams,
        OPUS_APPLICATION_AUDIO,
    )
    .expect("projection encoder create");

    let mut demixing = vec![0u8; enc.demixing_matrix_size() as usize];
    enc.copy_demixing_matrix(&mut demixing)
        .expect("copy demixing matrix");

    (enc, streams, coupled_streams, demixing)
}

fn make_projection_decoder_for(sample_rate: i32) -> OpusProjectionDecoder {
    let mut streams = 0;
    let mut coupled_streams = 0;
    let enc = OpusProjectionEncoder::new(
        sample_rate,
        4,
        3,
        &mut streams,
        &mut coupled_streams,
        OPUS_APPLICATION_AUDIO,
    )
    .expect("projection encoder create");
    let mut demixing = vec![0u8; enc.demixing_matrix_size() as usize];
    enc.copy_demixing_matrix(&mut demixing)
        .expect("copy demixing matrix");
    OpusProjectionDecoder::new(sample_rate, 4, streams, coupled_streams, &demixing)
        .expect("projection decoder create")
}

#[cfg(feature = "qext")]
fn decode_projection(
    packet: &[u8],
    streams: i32,
    coupled_streams: i32,
    demixing: &[u8],
    ignore_extensions: bool,
) -> (Vec<i16>, u32) {
    let (ret, pcm, final_range) = decode_projection_raw(
        packet,
        streams,
        coupled_streams,
        demixing,
        ignore_extensions,
    );
    assert_eq!(ret, FRAME_SIZE_20MS_96K, "projection decode failed");
    (pcm, final_range)
}

#[cfg(feature = "qext")]
fn decode_projection_raw(
    packet: &[u8],
    streams: i32,
    coupled_streams: i32,
    demixing: &[u8],
    ignore_extensions: bool,
) -> (i32, Vec<i16>, u32) {
    let mut dec =
        OpusProjectionDecoder::new(SAMPLE_RATE_96K, 4, streams, coupled_streams, demixing)
            .expect("projection decoder create");
    dec.set_ignore_extensions(ignore_extensions);

    let mut pcm = vec![0i16; FRAME_SIZE_20MS_96K as usize * 4];
    let ret = dec.decode(packet, &mut pcm, FRAME_SIZE_20MS_96K, false);
    (ret, pcm, dec.final_range())
}

#[cfg(all(feature = "tools", feature = "qext"))]
fn decode_projection_c(
    packet: &[u8],
    streams: i32,
    coupled_streams: i32,
    demixing: &[u8],
    ignore_extensions: bool,
) -> (i32, Vec<i16>, u32) {
    let mut err = 0i32;
    let dec = unsafe {
        opus_projection_decoder_create(
            SAMPLE_RATE_96K,
            4,
            streams,
            coupled_streams,
            demixing.as_ptr(),
            demixing.len() as i32,
            &mut err as *mut _,
        )
    };
    assert!(!dec.is_null(), "c projection decoder create failed: {err}");
    if ignore_extensions {
        let set_ret =
            unsafe { opus_projection_decoder_ctl(dec, OPUS_SET_IGNORE_EXTENSIONS_REQUEST, 1i32) };
        assert_eq!(
            set_ret, 0,
            "c projection set ignore_extensions failed: {set_ret}"
        );
    }
    let mut pcm = vec![0i16; FRAME_SIZE_20MS_96K as usize * 4];
    let ret = unsafe {
        opus_projection_decode(
            dec,
            packet.as_ptr(),
            packet.len() as i32,
            pcm.as_mut_ptr(),
            FRAME_SIZE_20MS_96K,
            0,
        )
    };
    let mut rng = 0u32;
    let rng_ret = unsafe {
        opus_projection_decoder_ctl(dec, OPUS_GET_FINAL_RANGE_REQUEST, &mut rng as *mut _)
    };
    assert_eq!(rng_ret, 0, "c projection get final range failed: {rng_ret}");
    unsafe { opus_projection_decoder_destroy(dec) };
    (ret, pcm, rng)
}

#[cfg(feature = "qext")]
fn packet_padding_bounds(packet: &[u8]) -> Option<(usize, usize, i32)> {
    let mut toc = 0u8;
    let mut sizes = [0i16; 48];
    let mut packet_offset = 0i32;
    let mut padding_len = 0i32;
    let nb_frames = opus_packet_parse_impl(
        packet,
        false,
        Some(&mut toc),
        None,
        &mut sizes,
        None,
        Some(&mut packet_offset),
        Some(&mut padding_len),
    );
    if nb_frames <= 0 || padding_len <= 0 {
        return None;
    }

    let start = (packet_offset - padding_len) as usize;
    let end = packet_offset as usize;
    if end <= start || end > packet.len() {
        return None;
    }
    Some((start, end, nb_frames))
}

#[cfg(feature = "qext")]
fn find_malformed_extension_packet(packet: &[u8]) -> Option<Vec<u8>> {
    let (padding_start, padding_end, nb_frames) = packet_padding_bounds(packet)?;
    let xor_masks = [0x01u8, 0x20, 0x40, 0x80, 0xFF];
    let set_values = [0x00u8, 0x01, 0x20, 0x40, 0x7F, 0x80, 0xFF];
    let pair_values = [(0x00u8, 0xFFu8), (0xFF, 0x00), (0xFF, 0xFF), (0x7F, 0x80)];

    for idx in padding_start..padding_end {
        for mask in xor_masks {
            let mut mutated = packet.to_vec();
            mutated[idx] ^= mask;
            if mutated[idx] == packet[idx] {
                continue;
            }
            if opus_packet_extensions_parse(&mutated[padding_start..padding_end], 64, nb_frames)
                .is_err()
            {
                return Some(mutated);
            }
        }
        for value in set_values {
            if value == packet[idx] {
                continue;
            }
            let mut mutated = packet.to_vec();
            mutated[idx] = value;
            if opus_packet_extensions_parse(&mutated[padding_start..padding_end], 64, nb_frames)
                .is_err()
            {
                return Some(mutated);
            }
        }
        if idx + 1 < padding_end {
            for (a, b) in pair_values {
                let mut mutated = packet.to_vec();
                mutated[idx] = a;
                mutated[idx + 1] = b;
                if opus_packet_extensions_parse(&mutated[padding_start..padding_end], 64, nb_frames)
                    .is_err()
                {
                    return Some(mutated);
                }
            }
        }
    }
    None
}

#[cfg(feature = "qext")]
fn mutate_extension_packet(packet: &[u8]) -> Option<Vec<u8>> {
    let (padding_start, _padding_end, _nb_frames) = packet_padding_bounds(packet)?;
    let mut mutated = packet.to_vec();
    mutated[padding_start] ^= 0x80;
    if mutated[padding_start] == packet[padding_start] {
        mutated[padding_start] = packet[padding_start].wrapping_add(1);
    }
    Some(mutated)
}

#[cfg(feature = "qext")]
fn locate_last_stream(packet: &[u8], nb_streams: i32) -> Option<(usize, usize)> {
    let mut offset = 0usize;
    let mut remaining = packet.len() as i32;
    for _ in 0..nb_streams - 1 {
        if remaining <= 0 {
            return None;
        }
        let mut toc = 0u8;
        let mut sizes = [0i16; 48];
        let mut packet_offset = 0i32;
        let ret = opus_packet_parse_impl(
            &packet[offset..offset + remaining as usize],
            true,
            Some(&mut toc),
            None,
            &mut sizes,
            None,
            Some(&mut packet_offset),
            None,
        );
        if ret < 0 || packet_offset <= 0 || packet_offset > remaining {
            return None;
        }
        offset += packet_offset as usize;
        remaining -= packet_offset;
    }
    Some((offset, packet.len()))
}

#[cfg(feature = "qext")]
fn build_projection_packet_with_forced_qext(
    enc: &mut OpusProjectionEncoder,
    streams: i32,
    seed: u32,
) -> Option<Vec<u8>> {
    let pcm = quad_pcm(seed);
    let mut packet = vec![0u8; 6000];
    let len = enc.encode(&pcm, FRAME_SIZE_20MS_96K, &mut packet);
    if len <= 0 {
        return None;
    }
    packet.truncate(len as usize);

    let (last_off, last_end) = locate_last_stream(&packet, streams)?;
    let old_last_len = last_end - last_off;
    let new_last_len = old_last_len + 64;
    let mut padded = vec![0u8; packet.len() + 64];
    padded[..packet.len()].copy_from_slice(&packet);
    let pad_ret = opus_packet_pad(
        &mut padded[last_off..last_off + new_last_len],
        old_last_len as i32,
        new_last_len as i32,
    );
    if pad_ret < 0 {
        return None;
    }
    padded.truncate(last_off + new_last_len);

    let (padding_start, padding_end, nb_frames) = packet_padding_bounds(&padded[last_off..])?;
    let padding = &mut padded[last_off + padding_start..last_off + padding_end];
    let exts = [OpusExtensionData {
        id: QEXT_EXTENSION_ID,
        frame: 0,
        data: vec![0x11, 0x22, 0x33, 0x44],
    }];
    if opus_packet_extensions_generate(padding, &exts, nb_frames, true).is_err() {
        return None;
    }
    let ids = opus_packet_extensions_parse(padding, 64, nb_frames)
        .ok()?
        .into_iter()
        .map(|e| e.id)
        .collect::<Vec<_>>();
    if !ids.contains(&QEXT_EXTENSION_ID) {
        return None;
    }
    Some(padded)
}

#[cfg(feature = "qext")]
#[test]
fn decoder_ignore_extensions_matches_unpadded_decode_for_real_qext_packets() {
    for seed in 1..=40u32 {
        let mut enc =
            OpusEncoder::new(SAMPLE_RATE_96K, 2, OPUS_APPLICATION_AUDIO).expect("encoder create");
        enc.set_bitrate(Bitrate::Bits(128_000));
        enc.set_complexity(10).expect("set complexity");
        enc.set_qext(true);

        let pcm = stereo_pcm(seed);
        let mut packet = vec![0u8; 4000];
        let len = enc.encode(&pcm, &mut packet);
        assert!(len > 0, "encode failed");
        packet.truncate(len as usize);

        let ext_ids = parse_padding_extensions(&packet);
        if !ext_ids.contains(&QEXT_EXTENSION_ID) {
            continue;
        }

        let mut unpadded = packet.clone();
        let unpadded_len = opus_packet_unpad(&mut unpadded);
        assert!(unpadded_len > 0, "packet unpad failed");
        unpadded.truncate(unpadded_len as usize);

        let (pcm_with_ext, rng_with_ext) = decode_single(&packet, false);
        let (pcm_ignored, rng_ignored) = decode_single(&packet, true);
        let (pcm_unpadded, rng_unpadded) = decode_single(&unpadded, false);
        #[cfg(feature = "tools")]
        {
            let (c_ret_with_ext, c_pcm_with_ext, c_rng_with_ext) = decode_single_c(&packet, false);
            assert_eq!(c_ret_with_ext, FRAME_SIZE_20MS_96K);
            let (c_ret_ignore, c_pcm_ignore, c_rng_ignore) = decode_single_c(&packet, true);
            assert_eq!(c_ret_ignore, FRAME_SIZE_20MS_96K);
            if pcm_with_ext != c_pcm_with_ext {
                let first_diff = pcm_with_ext
                    .iter()
                    .zip(c_pcm_with_ext.iter())
                    .position(|(a, b)| a != b)
                    .unwrap_or(usize::MAX);
                let mut diff_count = 0usize;
                let mut max_abs_diff = 0i32;
                for (a, b) in pcm_with_ext.iter().zip(c_pcm_with_ext.iter()) {
                    let d = (*a as i32 - *b as i32).abs();
                    if d != 0 {
                        diff_count += 1;
                        max_abs_diff = max_abs_diff.max(d);
                    }
                }
                panic!(
                    "single-stream with-ext PCM mismatch (rust vs c): \
first_diff={first_diff}, rust_sample={}, c_sample={}, \
diff_count={diff_count}, max_abs_diff={max_abs_diff}, \
rust_with_ext_eq_ignore={}, rust_with_ext_eq_unpadded={}, \
c_with_ext_eq_ignore={}, rust_rng_with_ext={rng_with_ext}, c_rng_with_ext={c_rng_with_ext}, \
rust_rng_ignore={rng_ignored}, c_rng_ignore={c_rng_ignore}, seed={seed}, \
qext_payload_len={qext_len}, qext_payload_head={qext_head:?}, \
c_qext_payload_len={c_qext_len}, c_qext_payload_head={c_qext_head:?}, payloads_equal={payloads_equal}, \
rust_qext_hdr={rust_hdr:?}, c_qext_hdr={c_hdr:?}",
                    pcm_with_ext.get(first_diff).copied().unwrap_or(0),
                    c_pcm_with_ext.get(first_diff).copied().unwrap_or(0),
                    pcm_with_ext == pcm_ignored,
                    pcm_with_ext == pcm_unpadded,
                    c_pcm_with_ext == c_pcm_ignore,
                    qext_len = first_qext_payload(&packet).as_ref().map_or(0, |p| p.len()),
                    qext_head = first_qext_payload(&packet)
                        .map(|p| p.into_iter().take(16).collect::<Vec<_>>())
                        .unwrap_or_default(),
                    c_qext_len = first_qext_payload_c(&packet).as_ref().map_or(0, |p| p.len()),
                    c_qext_head = first_qext_payload_c(&packet)
                        .map(|p| p.into_iter().take(16).collect::<Vec<_>>())
                        .unwrap_or_default(),
                    payloads_equal = first_qext_payload(&packet) == first_qext_payload_c(&packet),
                    rust_hdr = first_qext_payload(&packet)
                        .map(|p| decode_qext_header_rust(&p))
                        .unwrap_or((0, 0, 0, 0, 0, 0)),
                    c_hdr = first_qext_payload_c(&packet)
                        .map(|p| decode_qext_header_c(&p))
                        .unwrap_or((0, 0, 0, 0, 0, 0)),
                );
            }
            assert_eq!(rng_with_ext, c_rng_with_ext);
            assert_eq!(pcm_ignored, c_pcm_ignore);
            assert_eq!(rng_ignored, c_rng_ignore);

            let (c_ret_unpadded, c_pcm_unpadded, c_rng_unpadded) =
                decode_single_c(&unpadded, false);
            assert_eq!(c_ret_unpadded, FRAME_SIZE_20MS_96K);
            assert_eq!(pcm_unpadded, c_pcm_unpadded);
            assert_eq!(rng_unpadded, c_rng_unpadded);
        }

        assert_eq!(
            pcm_ignored, pcm_unpadded,
            "ignore_extensions decode should match unpadded decode"
        );
        assert_eq!(
            rng_ignored, rng_unpadded,
            "ignore_extensions final range should match unpadded decode"
        );

        if pcm_with_ext != pcm_unpadded || rng_with_ext != rng_unpadded {
            return;
        }
    }

    panic!("failed to find a QEXT packet where extension-aware decode differs from unpadded");
}

#[cfg(feature = "qext")]
#[test]
fn ms_decoder_ignore_extensions_matches_unpadded_decode_for_real_qext_packets() {
    for seed in 1..=40u32 {
        let mut enc = OpusMSEncoder::new(SAMPLE_RATE_96K, 2, 1, 1, &[0, 1], OPUS_APPLICATION_AUDIO)
            .expect("ms encoder create");
        enc.set_bitrate(Bitrate::Bits(128_000));
        enc.set_complexity(10).expect("set complexity");
        enc.set_qext(true);

        let pcm = stereo_pcm(seed.wrapping_add(1000));
        let mut packet = vec![0u8; 4000];
        let len = enc.encode(&pcm, &mut packet);
        assert!(len > 0, "multistream encode failed");
        packet.truncate(len as usize);

        let ext_ids = parse_padding_extensions(&packet);
        if !ext_ids.contains(&QEXT_EXTENSION_ID) {
            continue;
        }

        let mut unpadded = packet.clone();
        let unpadded_cap = unpadded.len() as i32;
        let unpadded_len = opus_multistream_packet_unpad(&mut unpadded, unpadded_cap, 1);
        assert!(unpadded_len > 0, "multistream packet unpad failed");
        unpadded.truncate(unpadded_len as usize);

        let (pcm_with_ext, rng_with_ext) = decode_ms(&packet, false);
        let (pcm_ignored, rng_ignored) = decode_ms(&packet, true);
        let (pcm_unpadded, rng_unpadded) = decode_ms(&unpadded, false);
        #[cfg(feature = "tools")]
        {
            let (c_ret_with_ext, c_pcm_with_ext, c_rng_with_ext) = decode_ms_c(&packet, false);
            assert_eq!(c_ret_with_ext, FRAME_SIZE_20MS_96K);
            assert_eq!(pcm_with_ext, c_pcm_with_ext);
            assert_eq!(rng_with_ext, c_rng_with_ext);

            let (c_ret_unpadded, c_pcm_unpadded, c_rng_unpadded) = decode_ms_c(&unpadded, false);
            assert_eq!(c_ret_unpadded, FRAME_SIZE_20MS_96K);
            assert_eq!(pcm_unpadded, c_pcm_unpadded);
            assert_eq!(rng_unpadded, c_rng_unpadded);
            assert_eq!(pcm_ignored, c_pcm_unpadded);
            assert_eq!(rng_ignored, c_rng_unpadded);
        }

        assert_eq!(
            pcm_ignored, pcm_unpadded,
            "multistream ignore_extensions decode should match unpadded decode"
        );
        assert_eq!(
            rng_ignored, rng_unpadded,
            "multistream ignore_extensions final range should match unpadded decode"
        );

        if pcm_with_ext != pcm_unpadded || rng_with_ext != rng_unpadded {
            return;
        }
    }

    panic!("failed to find a multistream QEXT packet where extension-aware decode differs from unpadded");
}

#[cfg(feature = "qext")]
#[test]
fn projection_decoder_ignore_extensions_matches_unpadded_decode_for_real_qext_packets() {
    let (mut enc, streams, coupled_streams, demixing) = make_projection_codec_pair();
    enc.set_bitrate(Bitrate::Bits(192_000));
    enc.set_complexity(10).expect("set complexity");
    enc.set_qext(true);

    for seed in 1..=60u32 {
        let pcm = quad_pcm(seed.wrapping_add(2000));
        let mut packet = vec![0u8; 6000];
        let len = enc.encode(&pcm, FRAME_SIZE_20MS_96K, &mut packet);
        assert!(len > 0, "projection encode failed");
        packet.truncate(len as usize);

        let mut unpadded = packet.clone();
        let unpadded_cap = unpadded.len() as i32;
        let unpadded_len = opus_multistream_packet_unpad(&mut unpadded, unpadded_cap, streams);
        if unpadded_len <= 0 || unpadded_len as usize >= packet.len() {
            continue;
        }
        unpadded.truncate(unpadded_len as usize);

        let (pcm_with_ext, rng_with_ext) =
            decode_projection(&packet, streams, coupled_streams, &demixing, false);
        let (pcm_ignored, rng_ignored) =
            decode_projection(&packet, streams, coupled_streams, &demixing, true);
        let (pcm_unpadded, rng_unpadded) =
            decode_projection(&unpadded, streams, coupled_streams, &demixing, false);
        #[cfg(feature = "tools")]
        {
            let (c_ret_with_ext, c_pcm_with_ext, c_rng_with_ext) =
                decode_projection_c(&packet, streams, coupled_streams, &demixing, false);
            assert_eq!(c_ret_with_ext, FRAME_SIZE_20MS_96K);
            assert_eq!(pcm_with_ext, c_pcm_with_ext);
            assert_eq!(rng_with_ext, c_rng_with_ext);

            let (c_ret_unpadded, c_pcm_unpadded, c_rng_unpadded) =
                decode_projection_c(&unpadded, streams, coupled_streams, &demixing, false);
            assert_eq!(c_ret_unpadded, FRAME_SIZE_20MS_96K);
            assert_eq!(pcm_unpadded, c_pcm_unpadded);
            assert_eq!(rng_unpadded, c_rng_unpadded);
            assert_eq!(pcm_ignored, c_pcm_unpadded);
            assert_eq!(rng_ignored, c_rng_unpadded);
        }

        assert_eq!(
            pcm_ignored, pcm_unpadded,
            "projection ignore_extensions decode should match unpadded decode"
        );
        assert_eq!(
            rng_ignored, rng_unpadded,
            "projection ignore_extensions final range should match unpadded decode"
        );

        if pcm_with_ext != pcm_unpadded || rng_with_ext != rng_unpadded {
            return;
        }
    }

    panic!("failed to find a projection packet where extension-aware decode differs from unpadded");
}

#[cfg(feature = "qext")]
#[test]
fn malformed_qext_extensions_fallback_matches_ignore_extensions_decode() {
    for seed in 1..=80u32 {
        let mut enc =
            OpusEncoder::new(SAMPLE_RATE_96K, 2, OPUS_APPLICATION_AUDIO).expect("encoder create");
        enc.set_bitrate(Bitrate::Bits(128_000));
        enc.set_complexity(10).expect("set complexity");
        enc.set_qext(true);

        let pcm = stereo_pcm(seed.wrapping_add(5000));
        let mut packet = vec![0u8; 4000];
        let len = enc.encode(&pcm, &mut packet);
        assert!(len > 0, "encode failed");
        packet.truncate(len as usize);

        let ext_ids = parse_padding_extensions(&packet);
        if !ext_ids.contains(&QEXT_EXTENSION_ID) {
            continue;
        }

        let malformed = match find_malformed_extension_packet(&packet)
            .or_else(|| mutate_extension_packet(&packet))
        {
            Some(m) => m,
            None => continue,
        };

        let (ret_ignore, pcm_ignore, rng_ignore) = decode_single_raw(&malformed, true);
        assert_eq!(
            ret_ignore, FRAME_SIZE_20MS_96K,
            "ignore_extensions decode failed on malformed extension packet"
        );
        let (ret_with_ext_a, pcm_with_ext_a, rng_with_ext_a) = decode_single_raw(&malformed, false);
        let (ret_with_ext_b, pcm_with_ext_b, rng_with_ext_b) = decode_single_raw(&malformed, false);
        assert_eq!(
            ret_with_ext_a, ret_with_ext_b,
            "extension-aware decode return code should be deterministic on malformed extensions"
        );
        assert!(
            ret_with_ext_a == FRAME_SIZE_20MS_96K || ret_with_ext_a == opurs::OPUS_INVALID_PACKET,
            "unexpected extension-aware decode return code on malformed extension packet: {ret_with_ext_a}"
        );
        if ret_with_ext_a == FRAME_SIZE_20MS_96K {
            assert_eq!(
                pcm_with_ext_a, pcm_with_ext_b,
                "extension-aware decode output should be deterministic on malformed extensions"
            );
            assert_eq!(
                rng_with_ext_a, rng_with_ext_b,
                "extension-aware decode final range should be deterministic on malformed extensions"
            );
        }
        assert_eq!(
            ret_ignore, FRAME_SIZE_20MS_96K,
            "ignore_extensions decode return code should stay valid for malformed extension packets"
        );
        let (_ret_ignore_b, pcm_ignore_b, rng_ignore_b) = decode_single_raw(&malformed, true);
        assert_eq!(
            pcm_ignore, pcm_ignore_b,
            "ignore_extensions decode output should be deterministic on malformed extensions"
        );
        assert_eq!(
            rng_ignore, rng_ignore_b,
            "ignore_extensions decode final range should be deterministic on malformed extensions"
        );
        #[cfg(feature = "tools")]
        {
            let (c_ret_with_ext, c_pcm_with_ext, c_rng_with_ext) =
                decode_single_c(&malformed, false);
            let (c_ret_ignore, c_pcm_ignore, c_rng_ignore) = decode_single_c(&malformed, true);
            assert_eq!(
                ret_with_ext_a, c_ret_with_ext,
                "single-stream malformed extension-aware return code mismatch (rust vs c)"
            );
            if ret_with_ext_a == FRAME_SIZE_20MS_96K {
                if pcm_with_ext_a != c_pcm_with_ext {
                    let first_diff = pcm_with_ext_a
                        .iter()
                        .zip(c_pcm_with_ext.iter())
                        .position(|(a, b)| a != b)
                        .unwrap_or(usize::MAX);
                    let mut diff_count = 0usize;
                    let mut max_abs_diff = 0i32;
                    for (a, b) in pcm_with_ext_a.iter().zip(c_pcm_with_ext.iter()) {
                        let d = (*a as i32 - *b as i32).abs();
                        if d != 0 {
                            diff_count += 1;
                            max_abs_diff = max_abs_diff.max(d);
                        }
                    }
                    panic!(
                        "single-stream malformed extension-aware PCM mismatch (rust vs c): \
first_diff={first_diff}, rust_sample={}, c_sample={}, \
diff_count={diff_count}, max_abs_diff={max_abs_diff}, \
rust_with_ext_eq_ignore={}, c_with_ext_eq_ignore={}, \
rust_rng_with_ext={rng_with_ext_a}, c_rng_with_ext={c_rng_with_ext}, \
rust_rng_ignore={rng_ignore}, c_rng_ignore={c_rng_ignore}",
                        pcm_with_ext_a.get(first_diff).copied().unwrap_or(0),
                        c_pcm_with_ext.get(first_diff).copied().unwrap_or(0),
                        pcm_with_ext_a == pcm_ignore,
                        c_pcm_with_ext == c_pcm_ignore,
                    );
                }
                assert_eq!(
                    rng_with_ext_a, c_rng_with_ext,
                    "single-stream malformed extension-aware final range mismatch (rust vs c)"
                );
            }
            assert_eq!(
                ret_ignore, c_ret_ignore,
                "single-stream malformed ignore_extensions return code mismatch (rust vs c)"
            );
            assert_eq!(
                pcm_ignore, c_pcm_ignore,
                "single-stream malformed ignore_extensions PCM mismatch (rust vs c)"
            );
            assert_eq!(
                rng_ignore, c_rng_ignore,
                "single-stream malformed ignore_extensions final range mismatch (rust vs c)"
            );
        }
        return;
    }

    panic!("failed to synthesize a malformed QEXT extension packet");
}

#[cfg(feature = "qext")]
#[test]
fn malformed_qext_extensions_multistream_decode_path_is_deterministic() {
    for seed in 1..=80u32 {
        let mut enc = OpusMSEncoder::new(SAMPLE_RATE_96K, 2, 1, 1, &[0, 1], OPUS_APPLICATION_AUDIO)
            .expect("ms encoder create");
        enc.set_bitrate(Bitrate::Bits(128_000));
        enc.set_complexity(10).expect("set complexity");
        enc.set_qext(true);

        let pcm = stereo_pcm(seed.wrapping_add(6000));
        let mut packet = vec![0u8; 4000];
        let len = enc.encode(&pcm, &mut packet);
        assert!(len > 0, "multistream encode failed");
        packet.truncate(len as usize);

        let malformed = match find_malformed_extension_packet(&packet)
            .or_else(|| mutate_extension_packet(&packet))
        {
            Some(m) => m,
            None => continue,
        };

        let (ret_ignore, pcm_ignore_a, rng_ignore_a) = decode_ms_raw(&malformed, true);
        assert_eq!(
            ret_ignore, FRAME_SIZE_20MS_96K,
            "multistream ignore_extensions decode failed on malformed extension packet"
        );
        let (_ret_ignore_b, pcm_ignore_b, rng_ignore_b) = decode_ms_raw(&malformed, true);
        assert_eq!(
            pcm_ignore_a, pcm_ignore_b,
            "multistream ignore_extensions decode output should be deterministic"
        );
        assert_eq!(
            rng_ignore_a, rng_ignore_b,
            "multistream ignore_extensions decode final range should be deterministic"
        );

        let (ret_with_ext_a, pcm_with_ext_a, rng_with_ext_a) = decode_ms_raw(&malformed, false);
        let (ret_with_ext_b, pcm_with_ext_b, rng_with_ext_b) = decode_ms_raw(&malformed, false);
        assert_eq!(
            ret_with_ext_a, ret_with_ext_b,
            "multistream extension-aware decode return code should be deterministic"
        );
        assert!(
            ret_with_ext_a == FRAME_SIZE_20MS_96K || ret_with_ext_a == opurs::OPUS_INVALID_PACKET,
            "unexpected multistream extension-aware decode return code: {ret_with_ext_a}"
        );
        if ret_with_ext_a == FRAME_SIZE_20MS_96K {
            assert_eq!(
                pcm_with_ext_a, pcm_with_ext_b,
                "multistream extension-aware decode output should be deterministic"
            );
            assert_eq!(
                rng_with_ext_a, rng_with_ext_b,
                "multistream extension-aware decode final range should be deterministic"
            );
        }
        #[cfg(feature = "tools")]
        {
            let (c_ret_with_ext, c_pcm_with_ext, c_rng_with_ext) = decode_ms_c(&malformed, false);
            assert_eq!(
                ret_with_ext_a, c_ret_with_ext,
                "multistream malformed extension-aware return code mismatch (rust vs c)"
            );
            if ret_with_ext_a == FRAME_SIZE_20MS_96K {
                assert_eq!(
                    pcm_with_ext_a, c_pcm_with_ext,
                    "multistream malformed extension-aware PCM mismatch (rust vs c)"
                );
                assert_eq!(
                    rng_with_ext_a, c_rng_with_ext,
                    "multistream malformed extension-aware final range mismatch (rust vs c)"
                );
            }
            let mut malformed_unpadded = malformed.clone();
            let malformed_unpadded_cap = malformed_unpadded.len() as i32;
            let malformed_unpadded_len =
                opus_multistream_packet_unpad(&mut malformed_unpadded, malformed_unpadded_cap, 1);
            assert!(
                malformed_unpadded_len > 0,
                "multistream malformed packet unpad failed"
            );
            malformed_unpadded.truncate(malformed_unpadded_len as usize);

            let (c_ret_ignore, c_pcm_ignore, c_rng_ignore) =
                decode_ms_c(&malformed_unpadded, false);
            assert_eq!(
                ret_ignore, c_ret_ignore,
                "multistream malformed ignore_extensions return code mismatch (rust vs c unpadded)"
            );
            assert_eq!(
                pcm_ignore_a, c_pcm_ignore,
                "multistream malformed ignore_extensions PCM mismatch (rust vs c unpadded)"
            );
            assert_eq!(
                rng_ignore_a, c_rng_ignore,
                "multistream malformed ignore_extensions final range mismatch (rust vs c unpadded)"
            );
        }
        return;
    }

    panic!("failed to synthesize malformed QEXT packet for multistream decode path");
}

#[cfg(feature = "qext")]
#[test]
fn malformed_qext_extensions_projection_decode_path_is_deterministic() {
    let (mut enc, streams, coupled_streams, demixing) = make_projection_codec_pair();
    enc.set_bitrate(Bitrate::Bits(192_000));
    enc.set_complexity(10).expect("set complexity");
    enc.set_qext(true);

    for seed in 1..=100u32 {
        let packet = match build_projection_packet_with_forced_qext(
            &mut enc,
            streams,
            seed.wrapping_add(7000),
        ) {
            Some(p) => p,
            None => continue,
        };
        let malformed = match find_malformed_extension_packet(&packet)
            .or_else(|| mutate_extension_packet(&packet))
        {
            Some(m) => m,
            None => continue,
        };

        let (ret_ignore, pcm_ignore_a, rng_ignore_a) =
            decode_projection_raw(&malformed, streams, coupled_streams, &demixing, true);
        assert_eq!(
            ret_ignore, FRAME_SIZE_20MS_96K,
            "projection ignore_extensions decode failed on mutated extension packet"
        );
        let (_ret_ignore_b, pcm_ignore_b, rng_ignore_b) =
            decode_projection_raw(&malformed, streams, coupled_streams, &demixing, true);
        assert_eq!(
            pcm_ignore_a, pcm_ignore_b,
            "projection ignore_extensions decode output should be deterministic"
        );
        assert_eq!(
            rng_ignore_a, rng_ignore_b,
            "projection ignore_extensions decode final range should be deterministic"
        );

        let (ret_with_ext_a, pcm_with_ext_a, rng_with_ext_a) =
            decode_projection_raw(&malformed, streams, coupled_streams, &demixing, false);
        let (ret_with_ext_b, pcm_with_ext_b, rng_with_ext_b) =
            decode_projection_raw(&malformed, streams, coupled_streams, &demixing, false);
        assert_eq!(
            ret_with_ext_a, ret_with_ext_b,
            "projection extension-aware decode return code should be deterministic"
        );
        assert!(
            ret_with_ext_a == FRAME_SIZE_20MS_96K || ret_with_ext_a == opurs::OPUS_INVALID_PACKET,
            "unexpected projection extension-aware decode return code: {ret_with_ext_a}"
        );
        if ret_with_ext_a == FRAME_SIZE_20MS_96K {
            assert_eq!(
                pcm_with_ext_a, pcm_with_ext_b,
                "projection extension-aware decode output should be deterministic"
            );
            assert_eq!(
                rng_with_ext_a, rng_with_ext_b,
                "projection extension-aware decode final range should be deterministic"
            );
        }
        #[cfg(feature = "tools")]
        {
            let (c_ret_with_ext, c_pcm_with_ext, c_rng_with_ext) =
                decode_projection_c(&malformed, streams, coupled_streams, &demixing, false);
            assert_eq!(
                ret_with_ext_a, c_ret_with_ext,
                "projection malformed extension-aware return code mismatch (rust vs c)"
            );
            if ret_with_ext_a == FRAME_SIZE_20MS_96K {
                assert_eq!(
                    pcm_with_ext_a, c_pcm_with_ext,
                    "projection malformed extension-aware PCM mismatch (rust vs c)"
                );
                assert_eq!(
                    rng_with_ext_a, c_rng_with_ext,
                    "projection malformed extension-aware final range mismatch (rust vs c)"
                );
            }
            let mut malformed_unpadded = malformed.clone();
            let malformed_unpadded_cap = malformed_unpadded.len() as i32;
            let malformed_unpadded_len = opus_multistream_packet_unpad(
                &mut malformed_unpadded,
                malformed_unpadded_cap,
                streams,
            );
            assert!(
                malformed_unpadded_len > 0,
                "projection malformed packet unpad failed"
            );
            malformed_unpadded.truncate(malformed_unpadded_len as usize);

            let (c_ret_ignore, c_pcm_ignore, c_rng_ignore) = decode_projection_c(
                &malformed_unpadded,
                streams,
                coupled_streams,
                &demixing,
                false,
            );
            assert_eq!(
                ret_ignore, c_ret_ignore,
                "projection malformed ignore_extensions return code mismatch (rust vs c unpadded)"
            );
            assert_eq!(
                pcm_ignore_a, c_pcm_ignore,
                "projection malformed ignore_extensions PCM mismatch (rust vs c unpadded)"
            );
            assert_eq!(
                rng_ignore_a, c_rng_ignore,
                "projection malformed ignore_extensions final range mismatch (rust vs c unpadded)"
            );
        }
        return;
    }

    panic!("failed to synthesize mutated QEXT packet for projection decode path");
}

#[cfg(feature = "osce")]
fn mono_pcm_20ms_48k(seed: u32) -> Vec<i16> {
    (0..960usize)
        .map(|i| {
            let x = seed
                .wrapping_mul(747796405)
                .wrapping_add(i as u32 * 2891336453);
            ((x >> 11) as i16).wrapping_sub(12288)
        })
        .collect()
}

#[cfg(feature = "osce")]
fn decode_single_osce(packet: &[u8], enable_osce_bwe: bool) -> (i32, Vec<i16>, u32) {
    let mut dec = OpusDecoder::new(48_000, 1).expect("decoder create");
    dec.set_complexity(10).expect("set complexity");
    dec.set_osce_bwe(enable_osce_bwe);
    let mut pcm = vec![0i16; 960];
    let ret = dec.decode(packet, &mut pcm, 960, false);
    (ret, pcm, dec.final_range())
}

#[cfg(feature = "osce")]
fn decode_ms_osce(packet: &[u8], enable_osce_bwe: bool) -> (i32, Vec<i16>, u32) {
    let mut dec = OpusMSDecoder::new(48_000, 1, 1, 0, &[0]).expect("ms decoder create");
    dec.set_complexity(10).expect("set complexity");
    dec.set_osce_bwe(enable_osce_bwe);
    let mut pcm = vec![0i16; 960];
    let ret = dec.decode(packet, &mut pcm, 960, false);
    (ret, pcm, dec.final_range())
}

#[cfg(feature = "osce")]
#[test]
fn osce_bwe_controls_roundtrip_on_all_decoder_types() {
    let mut dec = OpusDecoder::new(48_000, 1).expect("decoder create");
    assert!(!dec.osce_bwe());
    dec.set_osce_bwe(true);
    assert!(dec.osce_bwe());
    dec.set_osce_bwe(false);
    assert!(!dec.osce_bwe());

    let mut ms_dec = OpusMSDecoder::new(48_000, 1, 1, 0, &[0]).expect("ms decoder create");
    assert!(!ms_dec.osce_bwe());
    ms_dec.set_osce_bwe(true);
    assert!(ms_dec.osce_bwe());
    ms_dec.set_osce_bwe(false);
    assert!(!ms_dec.osce_bwe());

    let mut streams = 0;
    let mut coupled_streams = 0;
    let enc = OpusProjectionEncoder::new(
        48_000,
        4,
        3,
        &mut streams,
        &mut coupled_streams,
        OPUS_APPLICATION_RESTRICTED_LOWDELAY,
    )
    .expect("projection encoder create");
    let mut demixing = vec![0u8; enc.demixing_matrix_size() as usize];
    enc.copy_demixing_matrix(&mut demixing)
        .expect("copy demixing matrix");
    let mut proj_dec = OpusProjectionDecoder::new(48_000, 4, streams, coupled_streams, &demixing)
        .expect("projection decoder create");
    assert!(!proj_dec.osce_bwe());
    proj_dec.set_osce_bwe(true);
    assert!(proj_dec.osce_bwe());
    proj_dec.set_osce_bwe(false);
    assert!(!proj_dec.osce_bwe());
}

#[cfg(feature = "osce")]
#[test]
fn osce_bwe_runtime_decode_path_changes_output_for_silk_only_packets() {
    for seed in 1..=80u32 {
        let pcm = mono_pcm_20ms_48k(seed);

        let mut enc =
            OpusEncoder::new(48_000, 1, OPUS_APPLICATION_RESTRICTED_SILK).expect("encoder create");
        enc.set_bitrate(Bitrate::Bits(20_000));
        enc.set_complexity(10).expect("set complexity");
        let mut packet = vec![0u8; 2000];
        let len = enc.encode(&pcm, &mut packet);
        assert!(len > 0, "single-stream encode failed");
        packet.truncate(len as usize);

        let (ret_off_a, pcm_off_a, rng_off_a) = decode_single_osce(&packet, false);
        let (ret_off_b, pcm_off_b, rng_off_b) = decode_single_osce(&packet, false);
        let (ret_on_a, pcm_on_a, rng_on_a) = decode_single_osce(&packet, true);
        let (ret_on_b, pcm_on_b, rng_on_b) = decode_single_osce(&packet, true);

        assert_eq!(ret_off_a, 960);
        assert_eq!(ret_off_b, 960);
        assert_eq!(ret_on_a, 960);
        assert_eq!(ret_on_b, 960);

        assert_eq!(
            pcm_off_a, pcm_off_b,
            "osce off decode should be deterministic"
        );
        assert_eq!(
            rng_off_a, rng_off_b,
            "osce off final range should be deterministic"
        );
        assert_eq!(pcm_on_a, pcm_on_b, "osce on decode should be deterministic");
        assert_eq!(
            rng_on_a, rng_on_b,
            "osce on final range should be deterministic"
        );
        assert_eq!(
            rng_off_a, rng_on_a,
            "osce toggle should not affect entropy final range"
        );

        let mut ms_enc = OpusMSEncoder::new(48_000, 1, 1, 0, &[0], OPUS_APPLICATION_VOIP)
            .expect("ms encoder create");
        ms_enc.set_bitrate(Bitrate::Bits(8_000));
        ms_enc.set_complexity(10).expect("set complexity");
        let mut ms_packet = vec![0u8; 2000];
        let ms_len = ms_enc.encode(&pcm, &mut ms_packet);
        assert!(ms_len > 0, "multistream encode failed");
        ms_packet.truncate(ms_len as usize);

        let (ms_ret_off, _ms_pcm_off, ms_rng_off) = decode_ms_osce(&ms_packet, false);
        let (ms_ret_on, _ms_pcm_on, ms_rng_on) = decode_ms_osce(&ms_packet, true);
        assert_eq!(ms_ret_off, 960);
        assert_eq!(ms_ret_on, 960);
        assert_eq!(
            ms_rng_off, ms_rng_on,
            "multistream osce toggle should not affect entropy final range"
        );

        if pcm_off_a != pcm_on_a {
            return;
        }
    }

    panic!("failed to observe OSCE BWE decode-path effect on single-stream SILK-only packets");
}

#[test]
fn ignore_extensions_is_preserved_across_reset_on_all_decoder_types() {
    let mut dec = OpusDecoder::new(48_000, 1).expect("decoder create");
    let before_dec = !dec.ignore_extensions();
    dec.set_ignore_extensions(before_dec);
    dec.reset();
    assert_eq!(dec.ignore_extensions(), before_dec);

    let mut ms_dec = OpusMSDecoder::new(48_000, 1, 1, 0, &[0]).expect("ms decoder create");
    let before_ms = !ms_dec.ignore_extensions();
    ms_dec.set_ignore_extensions(before_ms);
    ms_dec.reset();
    assert_eq!(ms_dec.ignore_extensions(), before_ms);

    let mut proj_dec = make_projection_decoder_for(48_000);
    let before_proj = !proj_dec.ignore_extensions();
    proj_dec.set_ignore_extensions(before_proj);
    proj_dec.reset();
    assert_eq!(proj_dec.ignore_extensions(), before_proj);
}

#[cfg(feature = "qext")]
#[test]
fn qext_is_preserved_across_reset_on_all_encoder_types() {
    let mut enc = OpusEncoder::new(48_000, 1, OPUS_APPLICATION_AUDIO).expect("encoder create");
    let before_enc = !enc.qext();
    enc.set_qext(before_enc);
    enc.reset();
    assert_eq!(enc.qext(), before_enc);

    let mut ms_enc = OpusMSEncoder::new(48_000, 1, 1, 0, &[0], OPUS_APPLICATION_AUDIO)
        .expect("ms encoder create");
    let before_ms = !ms_enc.qext();
    ms_enc.set_qext(before_ms);
    ms_enc.reset();
    assert_eq!(ms_enc.qext(), before_ms);

    let mut streams = 0;
    let mut coupled_streams = 0;
    let mut proj_enc = OpusProjectionEncoder::new(
        48_000,
        4,
        3,
        &mut streams,
        &mut coupled_streams,
        OPUS_APPLICATION_AUDIO,
    )
    .expect("projection encoder create");
    let mut default_streams = 0;
    let mut default_coupled_streams = 0;
    let default_proj_enc = OpusProjectionEncoder::new(
        48_000,
        4,
        3,
        &mut default_streams,
        &mut default_coupled_streams,
        OPUS_APPLICATION_AUDIO,
    )
    .expect("projection encoder create");
    let before_proj = !default_proj_enc.qext();
    proj_enc.set_qext(before_proj);
    proj_enc.reset();
    assert_eq!(proj_enc.qext(), before_proj);
}

#[cfg(feature = "osce")]
#[test]
fn osce_bwe_is_preserved_across_reset_on_all_decoder_types() {
    let mut dec = OpusDecoder::new(48_000, 1).expect("decoder create");
    let before_dec = !dec.osce_bwe();
    dec.set_osce_bwe(before_dec);
    dec.reset();
    assert_eq!(dec.osce_bwe(), before_dec);

    let mut ms_dec = OpusMSDecoder::new(48_000, 1, 1, 0, &[0]).expect("ms decoder create");
    let before_ms = !ms_dec.osce_bwe();
    ms_dec.set_osce_bwe(before_ms);
    ms_dec.reset();
    assert_eq!(ms_dec.osce_bwe(), before_ms);

    let mut proj_dec = make_projection_decoder_for(48_000);
    let before_proj = !proj_dec.osce_bwe();
    proj_dec.set_osce_bwe(before_proj);
    proj_dec.reset();
    assert_eq!(proj_dec.osce_bwe(), before_proj);
}
