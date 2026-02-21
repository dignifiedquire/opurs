use opurs::{
    opus_projection_ambisonics_encoder_create as rust_opus_projection_encoder_create,
    opus_projection_decode as rust_opus_projection_decode,
    opus_projection_decoder_create as rust_opus_projection_decoder_create,
    opus_projection_encode as rust_opus_projection_encode,
    opus_projection_encode24 as rust_opus_projection_encode24,
    opus_projection_encode_float as rust_opus_projection_encode_float, OPUS_APPLICATION_AUDIO,
    OPUS_BAD_ARG,
};

#[test]
fn projection_encoder_foa_create_smoke() {
    let mut streams = 0i32;
    let mut coupled = 0i32;
    let enc = rust_opus_projection_encoder_create(
        48000,
        4,
        3,
        &mut streams,
        &mut coupled,
        OPUS_APPLICATION_AUDIO,
    );
    assert!(enc.is_ok());
    assert_eq!(streams, 2);
    assert_eq!(coupled, 2);
}

#[test]
fn projection_encoder_soa_create_smoke() {
    let mut streams = 0i32;
    let mut coupled = 0i32;
    let enc = rust_opus_projection_encoder_create(
        48000,
        9,
        3,
        &mut streams,
        &mut coupled,
        OPUS_APPLICATION_AUDIO,
    );
    assert!(enc.is_ok());
    assert_eq!(streams, 5);
    assert_eq!(coupled, 4);
}

#[test]
fn projection_encoder_unsupported_order_returns_bad_arg() {
    let mut streams = 0i32;
    let mut coupled = 0i32;
    let err = rust_opus_projection_encoder_create(
        48000,
        49,
        3,
        &mut streams,
        &mut coupled,
        OPUS_APPLICATION_AUDIO,
    )
    .err()
    .expect("order beyond available matrices should fail");
    assert_eq!(err, OPUS_BAD_ARG);
}

#[test]
fn projection_encoder_roundtrip_with_projection_decoder() {
    let mut streams = 0i32;
    let mut coupled = 0i32;
    let mut enc = rust_opus_projection_encoder_create(
        48000,
        4,
        3,
        &mut streams,
        &mut coupled,
        OPUS_APPLICATION_AUDIO,
    )
    .unwrap();

    let mut demix = vec![0u8; enc.demixing_matrix_size() as usize];
    enc.copy_demixing_matrix(&mut demix).unwrap();
    let mut dec = rust_opus_projection_decoder_create(48000, 4, streams, coupled, &demix).unwrap();

    let frame_size = 960usize;
    let mut pcm_i16 = vec![0i16; frame_size * 4];
    let mut pcm_f32 = vec![0f32; frame_size * 4];
    let mut pcm_i24 = vec![0i32; frame_size * 4];
    for i in 0..frame_size {
        for ch in 0..4usize {
            let base = (i as i32 * (ch as i32 + 3) * 17) - 2000;
            pcm_i16[i * 4 + ch] = base as i16;
            pcm_f32[i * 4 + ch] = (base as f32) / 32768.0 * 0.8;
            pcm_i24[i * 4 + ch] = base << 8;
        }
    }

    let mut packet = vec![0u8; 4000];
    let len_i16 = rust_opus_projection_encode(&mut enc, &pcm_i16, frame_size as i32, &mut packet);
    assert!(len_i16 > 0);

    let mut out = vec![0i16; frame_size * 4];
    let dec_i16 = rust_opus_projection_decode(
        &mut dec,
        &packet[..len_i16 as usize],
        &mut out,
        frame_size as i32,
        false,
    );
    assert_eq!(dec_i16, frame_size as i32);
    assert!(out.iter().any(|&x| x != 0));

    let len_f32 =
        rust_opus_projection_encode_float(&mut enc, &pcm_f32, frame_size as i32, &mut packet);
    assert!(len_f32 > 0);
    let len_i24 = rust_opus_projection_encode24(&mut enc, &pcm_i24, frame_size as i32, &mut packet);
    assert!(len_i24 > 0);
}
