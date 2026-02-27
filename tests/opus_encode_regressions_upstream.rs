//! Upstream encode regression tests, ported from C.
//!
//! Upstream C: `tests/opus_encode_regressions.c`

#[cfg(not(feature = "qext"))]
use opurs::OPUS_BAD_ARG;
use opurs::{
    opus_multistream_encode, opus_multistream_encoder_create,
    opus_multistream_surround_encoder_create, opus_projection_ambisonics_encoder_create, Bandwidth,
    Bitrate, Channels, OpusEncoder, Signal, OPUS_APPLICATION_AUDIO,
    OPUS_APPLICATION_RESTRICTED_LOWDELAY, OPUS_APPLICATION_VOIP,
};

fn identity_mapping(channels: usize) -> Vec<u8> {
    (0..channels).map(|i| i as u8).collect()
}

fn patterned_pcm_i16(samples: usize, seed: u32) -> Vec<i16> {
    let mut x = seed;
    let mut out = vec![0i16; samples];
    for sample in &mut out {
        x = x.wrapping_mul(1664525).wrapping_add(1013904223);
        *sample = ((x >> 16) as i16).wrapping_sub(16384);
    }
    out
}

fn mostly_constant_pcm_i16(samples: usize, base: i16, seed: u32) -> Vec<i16> {
    let mut out = vec![base; samples];
    let noise = patterned_pcm_i16(samples, seed);
    for (dst, n) in out.iter_mut().zip(noise) {
        *dst = dst.wrapping_add(n.wrapping_rem(256));
    }
    out
}

/// Upstream C: tests/opus_encode_regressions.c:mscbr_encode_fail10
#[test]
fn regression_mscbr_encode_fail10() {
    let mapping = identity_mapping(255);
    let mut enc = opus_multistream_encoder_create(
        8_000,
        255,
        254,
        1,
        &mapping,
        OPUS_APPLICATION_RESTRICTED_LOWDELAY,
    )
    .expect("multistream encoder create failed");

    enc.set_signal(Some(Signal::Voice));
    enc.set_vbr(false);
    enc.set_vbr_constraint(false);
    enc.set_prediction_disabled(false);
    let _ = enc.set_force_channels(Some(Channels::Stereo));
    enc.set_phase_inversion_disabled(true);
    enc.set_dtx(true);
    let _ = enc.set_complexity(2);
    enc.set_max_bandwidth(Bandwidth::Narrowband);
    enc.set_bandwidth(None);
    let _ = enc.set_lsb_depth(14);
    let _ = enc.set_inband_fec(0);
    let _ = enc.set_packet_loss_perc(57);
    enc.set_bitrate(Bitrate::Bits(3_642_675));

    let pcm = vec![0i16; 20 * 255];
    let mut data = vec![0u8; 627_300];
    let data_len = opus_multistream_encode(&mut enc, &pcm, 20, &mut data);
    assert!(
        data_len > 0 && data_len <= data.len() as i32,
        "expected positive packet length in range, got {data_len}"
    );
}

/// Upstream C: tests/opus_encode_regressions.c:mscbr_encode_fail
#[test]
fn regression_mscbr_encode_fail() {
    let mapping = identity_mapping(192);
    let mut enc = opus_multistream_encoder_create(
        8_000,
        192,
        189,
        3,
        &mapping,
        OPUS_APPLICATION_RESTRICTED_LOWDELAY,
    )
    .expect("multistream encoder create failed");

    enc.set_signal(Some(Signal::Music));
    enc.set_vbr(false);
    enc.set_vbr_constraint(false);
    enc.set_prediction_disabled(false);
    let _ = enc.set_force_channels(None);
    enc.set_phase_inversion_disabled(false);
    enc.set_dtx(false);
    let _ = enc.set_complexity(0);
    enc.set_max_bandwidth(Bandwidth::Mediumband);
    enc.set_bandwidth(None);
    let _ = enc.set_lsb_depth(8);
    let _ = enc.set_inband_fec(0);
    let _ = enc.set_packet_loss_perc(0);
    enc.set_bitrate(Bitrate::Bits(15_360));

    let pcm = vec![0i16; 20 * 192];
    let mut data = vec![0u8; 472_320];
    let data_len = opus_multistream_encode(&mut enc, &pcm, 20, &mut data);
    assert!(
        data_len > 0 && data_len <= data.len() as i32,
        "expected positive packet length in range, got {data_len}"
    );
}

/// Upstream C: tests/opus_encode_regressions.c:analysis_overflow
#[test]
fn regression_analysis_overflow() {
    let mut enc =
        OpusEncoder::new(16_000, 2, OPUS_APPLICATION_AUDIO).expect("encoder create failed");
    enc.set_complexity(10).expect("set complexity failed");

    let pcm = vec![1e9f32; 320 * 2];
    let mut data = [0u8; 200];
    let data_len = enc.encode_float(&pcm, &mut data);
    assert!(
        data_len > 0 && data_len <= data.len() as i32,
        "expected positive packet length in range, got {data_len}"
    );
}

/// Upstream C: tests/opus_encode_regressions.c:projection_overflow2
#[test]
fn regression_projection_overflow2() {
    let mut streams = 0i32;
    let mut coupled_streams = 0i32;
    let mut enc = opus_projection_ambisonics_encoder_create(
        12_000,
        9,
        3,
        &mut streams,
        &mut coupled_streams,
        OPUS_APPLICATION_RESTRICTED_LOWDELAY,
    )
    .expect("projection encoder create failed");

    let mut pcm = vec![2e34f32; 30 * 9];
    let n = pcm.len();
    pcm[n - 2] = 0.0;
    pcm[n - 1] = -8e9;
    let mut data = [0u8; 480];
    let data_len = enc.encode_float(&pcm, 30, &mut data);
    assert!(
        data_len > 0 && data_len <= data.len() as i32,
        "expected positive packet length in range, got {data_len}"
    );
}

/// Upstream C: tests/opus_encode_regressions.c:projection_overflow3
#[test]
fn regression_projection_overflow3() {
    let mut streams = 0i32;
    let mut coupled_streams = 0i32;
    let mut enc = opus_projection_ambisonics_encoder_create(
        24_000,
        4,
        3,
        &mut streams,
        &mut coupled_streams,
        OPUS_APPLICATION_AUDIO,
    )
    .expect("projection encoder create failed");

    let mut pcm = vec![-1e38f32; 60 * 4];
    let mut data = [0u8; 500];
    let data_len = enc.encode_float(&pcm, 60, &mut data);
    assert!(
        data_len >= 5 && data_len <= data.len() as i32,
        "expected packet length in [5, {}], got {data_len}",
        data.len()
    );

    pcm.fill(1e38f32);
    let data_len = enc.encode_float(&pcm, 60, &mut data);
    assert!(
        data_len >= 5 && data_len <= data.len() as i32,
        "expected packet length in [5, {}], got {data_len}",
        data.len()
    );
}

/// Upstream C: tests/opus_encode_regressions.c:qext_stereo_overflow
#[cfg(feature = "qext")]
#[test]
fn regression_qext_stereo_overflow() {
    let mut enc = OpusEncoder::new(96_000, 2, OPUS_APPLICATION_RESTRICTED_LOWDELAY)
        .expect("encoder create failed");

    let pcm = vec![32767i16; 11_520 * 2];
    let mut data = [0u8; 2000];
    let data_len = enc.encode(&pcm, &mut data);
    assert!(
        data_len > 0 && data_len <= data.len() as i32,
        "expected positive packet length in range, got {data_len}"
    );
}

/// Upstream C: tests/opus_encode_regressions.c:qext_repacketize_fail
#[cfg(feature = "qext")]
#[test]
fn regression_qext_repacketize_fail() {
    let mut enc =
        OpusEncoder::new(16_000, 1, OPUS_APPLICATION_VOIP).expect("encoder create failed");
    enc.set_vbr(false);
    enc.set_qext(true);
    enc.set_bitrate(Bitrate::Max);

    let mut pcm = vec![0i16; 960];
    pcm[0] = -20_454;
    pcm[1] = -7_680;
    pcm[2] = -12_281;
    pcm[3] = -250;
    pcm[4] = -1_809;

    let mut data = [0u8; 9000];
    let data_len = enc.encode(&pcm, &mut data);
    assert!(
        data_len > 0 && data_len <= data.len() as i32,
        "expected positive packet length in range, got {data_len}"
    );
}

/// Upstream C: tests/opus_encode_regressions.c:celt_ec_internal_error
#[test]
fn regression_celt_ec_internal_error() {
    let mut streams = 0i32;
    let mut coupled_streams = 0i32;
    let mut mapping = vec![0u8; 1];
    let mut enc = opus_multistream_surround_encoder_create(
        16_000,
        1,
        1,
        &mut streams,
        &mut coupled_streams,
        &mut mapping,
        OPUS_APPLICATION_VOIP,
    )
    .expect("surround encoder create failed");

    enc.set_signal(Some(Signal::Music));
    enc.set_vbr(false);
    enc.set_vbr_constraint(false);
    enc.set_prediction_disabled(true);
    enc.set_phase_inversion_disabled(false);
    enc.set_dtx(false);
    let _ = enc.set_complexity(0);
    enc.set_max_bandwidth(Bandwidth::Narrowband);
    enc.set_bandwidth(None);
    let _ = enc.set_lsb_depth(8);
    let _ = enc.set_inband_fec(0);
    let _ = enc.set_packet_loss_perc(0);
    enc.set_bitrate(Bitrate::Auto);

    let mut data = vec![0u8; 2460];
    let pcm0 = patterned_pcm_i16(320, 0x1234_5678);
    let data_len = opus_multistream_encode(&mut enc, &pcm0, 320, &mut data);
    assert!(
        data_len > 0 && data_len <= data.len() as i32,
        "first encode expected packet length in range, got {data_len}"
    );

    for idx in 0..4u32 {
        enc.set_signal(Some(Signal::Music));
        enc.set_vbr(false);
        enc.set_vbr_constraint(true);
        enc.set_prediction_disabled(true);
        enc.set_phase_inversion_disabled(false);
        enc.set_dtx(true);
        let _ = enc.set_complexity(10);
        enc.set_max_bandwidth(Bandwidth::Fullband);
        enc.set_bandwidth(Some(Bandwidth::Fullband));
        let _ = enc.set_lsb_depth(18);
        let _ = enc.set_inband_fec(1);
        let _ = enc.set_packet_loss_perc(90);
        enc.set_bitrate(Bitrate::Bits(280_130));

        let mut pcm = mostly_constant_pcm_i16(160, -9510, 0xABCD_0000u32.wrapping_add(idx));
        if !pcm.is_empty() {
            let mid = pcm.len() / 2;
            pcm[mid] = 25_600;
        }
        let data_len = opus_multistream_encode(&mut enc, &pcm, 160, &mut data);
        assert!(
            data_len > 0 && data_len <= data.len() as i32,
            "iteration {idx} encode expected packet length in range, got {data_len}"
        );
    }

    enc.set_signal(Some(Signal::Voice));
    enc.set_vbr(false);
    enc.set_vbr_constraint(true);
    enc.set_prediction_disabled(true);
    enc.set_phase_inversion_disabled(true);
    enc.set_dtx(true);
    let _ = enc.set_complexity(0);
    enc.set_max_bandwidth(Bandwidth::Narrowband);
    enc.set_bandwidth(None);
    let _ = enc.set_lsb_depth(12);
    let _ = enc.set_inband_fec(0);
    let _ = enc.set_packet_loss_perc(41);
    enc.set_bitrate(Bitrate::Bits(21_425));
    let pcm_last = patterned_pcm_i16(40, 0xCAFE_BABE);
    let data_len = opus_multistream_encode(&mut enc, &pcm_last, 40, &mut data);
    assert!(
        data_len > 0 && data_len <= data.len() as i32,
        "final encode expected packet length in range, got {data_len}"
    );
}

/// Upstream C: tests/opus_encode_regressions.c:surround_analysis_uninit
#[test]
fn regression_surround_analysis_uninit() {
    let mut streams = 0i32;
    let mut coupled_streams = 0i32;
    let mut mapping = vec![0u8; 3];
    let mut enc = opus_multistream_surround_encoder_create(
        24_000,
        3,
        1,
        &mut streams,
        &mut coupled_streams,
        &mut mapping,
        OPUS_APPLICATION_AUDIO,
    )
    .expect("surround encoder create failed");

    enc.set_signal(Some(Signal::Voice));
    enc.set_vbr(true);
    enc.set_vbr_constraint(true);
    enc.set_prediction_disabled(false);
    let _ = enc.set_force_channels(None);
    enc.set_phase_inversion_disabled(false);
    enc.set_dtx(false);
    let _ = enc.set_complexity(0);
    enc.set_max_bandwidth(Bandwidth::Narrowband);
    enc.set_bandwidth(Some(Bandwidth::Narrowband));
    let _ = enc.set_lsb_depth(8);
    let _ = enc.set_inband_fec(1);
    enc.set_bitrate(Bitrate::Bits(84_315));

    let mut data = vec![0u8; 7380];
    let pcm0 = patterned_pcm_i16(960 * 3, 0x00C0_FFEE);
    let data_len = opus_multistream_encode(&mut enc, &pcm0, 960, &mut data);
    assert!(
        data_len > 0 && data_len <= data.len() as i32,
        "first encode expected packet length in range, got {data_len}"
    );

    enc.set_signal(Some(Signal::Music));
    enc.set_vbr(true);
    enc.set_vbr_constraint(false);
    enc.set_prediction_disabled(true);
    let _ = enc.set_force_channels(None);
    enc.set_phase_inversion_disabled(true);
    enc.set_dtx(true);
    let _ = enc.set_complexity(6);
    enc.set_max_bandwidth(Bandwidth::Narrowband);
    enc.set_bandwidth(None);
    let _ = enc.set_lsb_depth(9);
    let _ = enc.set_inband_fec(1);
    let _ = enc.set_packet_loss_perc(5);
    enc.set_bitrate(Bitrate::Bits(775_410));

    let pcm1 = mostly_constant_pcm_i16(1440 * 3, -13365, 0x0BAD_F00D);
    let data_len = opus_multistream_encode(&mut enc, &pcm1, 1440, &mut data);
    assert!(
        data_len > 0 && data_len <= data.len() as i32,
        "second encode expected packet length in range, got {data_len}"
    );
}

/// Upstream C: tests/opus_encode_regressions.c:projection_overflow
#[cfg(not(feature = "qext"))]
#[test]
fn regression_projection_overflow() {
    let mut streams = 0i32;
    let mut coupled_streams = 0i32;
    let create = opus_projection_ambisonics_encoder_create(
        96_000,
        36,
        3,
        &mut streams,
        &mut coupled_streams,
        OPUS_APPLICATION_AUDIO,
    );
    let err = match create {
        Ok(_) => panic!("projection encoder create should fail without qext"),
        Err(err) => err,
    };
    assert_eq!(err, OPUS_BAD_ARG, "unexpected create error without qext");
}

/// Upstream C: tests/opus_encode_regressions.c:projection_overflow
#[cfg(feature = "qext")]
#[test]
fn regression_projection_overflow() {
    let mut streams = 0i32;
    let mut coupled_streams = 0i32;
    let mut enc = opus_projection_ambisonics_encoder_create(
        96_000,
        36,
        3,
        &mut streams,
        &mut coupled_streams,
        OPUS_APPLICATION_AUDIO,
    )
    .expect("projection encoder create failed");
    enc.set_qext(true);

    let pcm = patterned_pcm_i16(1920 * 36, 0xDEAD_BEEF);
    let mut data = vec![0u8; 10_000];
    let data_len = enc.encode(&pcm, 1920, &mut data);
    assert!(
        data_len > 0 && data_len <= data.len() as i32,
        "expected packet length in range, got {data_len}"
    );
}

/// Upstream C: tests/opus_encode_regressions.c:projection_overflow4
#[cfg(feature = "qext")]
#[test]
fn regression_projection_overflow4() {
    let mut streams = 0i32;
    let mut coupled_streams = 0i32;
    let mut enc = opus_projection_ambisonics_encoder_create(
        96_000,
        36,
        3,
        &mut streams,
        &mut coupled_streams,
        OPUS_APPLICATION_AUDIO,
    )
    .expect("projection encoder create failed");
    enc.set_qext(true);
    enc.set_bitrate(Bitrate::Bits(256_000));

    let pcm = mostly_constant_pcm_i16(480 * 36, -32640, 0xBADC_0FFE);
    let mut data = vec![0u8; 1000];
    let data_len = enc.encode(&pcm, 480, &mut data);
    assert!(
        data_len > 0 && data_len <= data.len() as i32,
        "expected packet length in range, got {data_len}"
    );
}

/// Upstream C: tests/opus_encode_regressions.c:qext_dred_combination
#[cfg(all(feature = "qext", feature = "dred"))]
#[test]
fn regression_qext_dred_combination() {
    let mut streams = 0i32;
    let mut coupled_streams = 0i32;
    let mut mapping = vec![0u8; 5];
    let mut enc = opus_multistream_surround_encoder_create(
        16_000,
        5,
        1,
        &mut streams,
        &mut coupled_streams,
        &mut mapping,
        OPUS_APPLICATION_VOIP,
    )
    .expect("surround encoder create failed");
    let _ = enc.set_complexity(3);
    let _ = enc.set_packet_loss_perc(12);
    enc.set_qext(true);
    if let Ok(stream_enc) = enc.encoder_state_mut(0) {
        let _ = stream_enc.set_dred_duration(103);
    }
    enc.set_bitrate(Bitrate::Bits(755_850));

    let mut pcm = mostly_constant_pcm_i16(320 * 5, -2057, 0xA11C_E55E);
    if pcm.len() >= 25 {
        pcm[0] = 0;
        pcm[1] = 1;
        pcm[2] = 15934;
        pcm[3] = -128;
        pcm[4] = -194;
        pcm[10] = 15872;
        pcm[24] = -2057;
    }
    let mut data = vec![0u8; 2560];
    let data_len = opus_multistream_encode(&mut enc, &pcm, 320, &mut data);
    assert!(
        data_len > 0 && data_len <= data.len() as i32,
        "expected packet length in range, got {data_len}"
    );
}
