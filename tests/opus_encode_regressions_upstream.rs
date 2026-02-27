//! Upstream encode regression tests, ported from C.
//!
//! Upstream C: `tests/opus_encode_regressions.c`

#[cfg(feature = "qext")]
use opurs::OPUS_APPLICATION_VOIP;
use opurs::{
    opus_multistream_encode, opus_multistream_encoder_create,
    opus_projection_ambisonics_encoder_create, Bandwidth, Bitrate, Channels, OpusEncoder, Signal,
    OPUS_APPLICATION_AUDIO, OPUS_APPLICATION_RESTRICTED_LOWDELAY,
};

fn identity_mapping(channels: usize) -> Vec<u8> {
    (0..channels).map(|i| i as u8).collect()
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
