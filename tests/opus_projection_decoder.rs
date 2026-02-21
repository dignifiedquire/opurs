use opurs::{
    opus_projection_decode as rust_opus_projection_decode,
    opus_projection_decode24 as rust_opus_projection_decode24,
    opus_projection_decode_float as rust_opus_projection_decode_float,
    opus_projection_decoder_create as rust_opus_projection_decoder_create, OpusMSDecoder,
    OpusMSEncoder, OPUS_APPLICATION_AUDIO, OPUS_BAD_ARG,
};

fn identity_demixing_matrix_le(channels: i32) -> Vec<u8> {
    let rows = channels as usize;
    let cols = channels as usize;
    let mut data = vec![0i16; rows * cols];
    for ch in 0..channels as usize {
        data[rows * ch + ch] = 32767;
    }
    let mut bytes = Vec::with_capacity(data.len() * 2);
    for value in data {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes
}

#[test]
fn projection_decoder_create_rejects_wrong_matrix_size() {
    let matrix = vec![0u8; 2];
    let err = rust_opus_projection_decoder_create(48000, 2, 1, 1, &matrix)
        .err()
        .expect("create should fail on bad matrix size");
    assert_eq!(err, OPUS_BAD_ARG);
}

#[test]
fn projection_decoder_identity_matches_multistream_decode_i16() {
    let mut enc = OpusMSEncoder::new(48000, 2, 1, 1, &[0, 1], OPUS_APPLICATION_AUDIO).unwrap();
    let matrix = identity_demixing_matrix_le(2);
    let mut proj_dec = rust_opus_projection_decoder_create(48000, 2, 1, 1, &matrix).unwrap();
    let mut ms_dec = OpusMSDecoder::new(48000, 2, 1, 1, &[0, 1]).unwrap();

    let frame_size = 960usize;
    let mut pcm = vec![0i16; frame_size * 2];
    for i in 0..frame_size {
        pcm[i * 2] = (i as i16).wrapping_mul(31);
        pcm[i * 2 + 1] = (i as i16).wrapping_mul(-17);
    }

    let mut packet = vec![0u8; 4000];
    let packet_len = enc.encode(&pcm, &mut packet);
    assert!(packet_len > 0);
    let packet = &packet[..packet_len as usize];

    let mut proj_out = vec![0i16; frame_size * 2];
    let mut ms_out = vec![0i16; frame_size * 2];
    let proj_ret = rust_opus_projection_decode(
        &mut proj_dec,
        packet,
        &mut proj_out,
        frame_size as i32,
        false,
    );
    let ms_ret = ms_dec.decode(packet, &mut ms_out, frame_size as i32, false);
    assert_eq!(proj_ret, ms_ret);
    for (idx, (&a, &b)) in proj_out.iter().zip(ms_out.iter()).enumerate() {
        assert!(
            (a as i32 - b as i32).abs() <= 1,
            "projection/ms mismatch at index {idx}: projection={a}, ms={b}"
        );
    }
}

#[test]
fn projection_decoder_identity_smoke_float_and_24bit() {
    let mut enc = OpusMSEncoder::new(48000, 2, 1, 1, &[0, 1], OPUS_APPLICATION_AUDIO).unwrap();
    let matrix = identity_demixing_matrix_le(2);
    let mut proj_dec = rust_opus_projection_decoder_create(48000, 2, 1, 1, &matrix).unwrap();

    let frame_size = 960usize;
    let mut pcm = vec![0f32; frame_size * 2];
    for i in 0..frame_size {
        let t = i as f32 / frame_size as f32;
        pcm[i * 2] = (t * core::f32::consts::TAU).sin() * 0.3;
        pcm[i * 2 + 1] = (t * core::f32::consts::PI * 3.0).sin() * 0.2;
    }

    let mut packet = vec![0u8; 4000];
    let packet_len = enc.encode_float(&pcm, &mut packet);
    assert!(packet_len > 0);
    let packet = &packet[..packet_len as usize];

    let mut out_f32 = vec![0f32; frame_size * 2];
    let ret_f32 = rust_opus_projection_decode_float(
        &mut proj_dec,
        packet,
        &mut out_f32,
        frame_size as i32,
        false,
    );
    assert_eq!(ret_f32, frame_size as i32);
    assert!(out_f32.iter().any(|&x| x != 0.0));

    let mut out_i32 = vec![0i32; frame_size * 2];
    let ret_i32 = rust_opus_projection_decode24(
        &mut proj_dec,
        packet,
        &mut out_i32,
        frame_size as i32,
        false,
    );
    assert_eq!(ret_i32, frame_size as i32);
    assert!(out_i32.iter().any(|&x| x != 0));
}
