#[cfg(feature = "qext")]
use opurs::internals::{opus_packet_extensions_parse, opus_packet_parse_impl};
#[cfg(feature = "osce")]
use opurs::OPUS_APPLICATION_RESTRICTED_LOWDELAY;
#[cfg(feature = "qext")]
use opurs::{
    opus_multistream_packet_unpad, opus_packet_unpad, Bitrate, OpusEncoder, OpusMSEncoder,
};
use opurs::{
    OpusDecoder, OpusMSDecoder, OpusProjectionDecoder, OpusProjectionEncoder,
    OPUS_APPLICATION_AUDIO,
};

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
fn decode_single(packet: &[u8], ignore_extensions: bool) -> (Vec<i16>, u32) {
    let mut dec = OpusDecoder::new(SAMPLE_RATE_96K, 2).expect("decoder create");
    dec.set_ignore_extensions(ignore_extensions);

    let mut pcm = vec![0i16; FRAME_SIZE_20MS_96K as usize * 2];
    let ret = dec.decode(packet, &mut pcm, FRAME_SIZE_20MS_96K, false);
    assert_eq!(ret, FRAME_SIZE_20MS_96K, "single decode failed");
    (pcm, dec.final_range())
}

#[cfg(feature = "qext")]
fn decode_ms(packet: &[u8], ignore_extensions: bool) -> (Vec<i16>, u32) {
    let mut dec = OpusMSDecoder::new(SAMPLE_RATE_96K, 2, 1, 1, &[0, 1]).expect("ms decoder create");
    dec.set_ignore_extensions(ignore_extensions);

    let mut pcm = vec![0i16; FRAME_SIZE_20MS_96K as usize * 2];
    let ret = dec.decode(packet, &mut pcm, FRAME_SIZE_20MS_96K, false);
    assert_eq!(ret, FRAME_SIZE_20MS_96K, "multistream decode failed");
    (pcm, dec.final_range())
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
    let mut dec =
        OpusProjectionDecoder::new(SAMPLE_RATE_96K, 4, streams, coupled_streams, demixing)
            .expect("projection decoder create");
    dec.set_ignore_extensions(ignore_extensions);

    let mut pcm = vec![0i16; FRAME_SIZE_20MS_96K as usize * 4];
    let ret = dec.decode(packet, &mut pcm, FRAME_SIZE_20MS_96K, false);
    assert_eq!(ret, FRAME_SIZE_20MS_96K, "projection decode failed");
    (pcm, dec.final_range())
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
