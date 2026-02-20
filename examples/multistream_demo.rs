//! Minimal multistream encode/decode demo.

use opurs::{
    opus_multistream_decode, opus_multistream_encode, OpusMSDecoder, OpusMSEncoder,
    OPUS_APPLICATION_AUDIO,
};

fn main() {
    let sample_rate = 48_000;
    let frame_size = 960;
    // 3 channels encoded as 2 streams: one coupled stereo + one mono.
    let channels = 3;
    let streams = 2;
    let coupled_streams = 1;
    let mapping = [0u8, 1u8, 2u8];

    let mut enc = OpusMSEncoder::new(
        sample_rate,
        channels,
        streams,
        coupled_streams,
        &mapping,
        OPUS_APPLICATION_AUDIO,
    )
    .expect("create multistream encoder");
    let mut dec = OpusMSDecoder::new(sample_rate, channels, streams, coupled_streams, &mapping)
        .expect("create multistream decoder");

    let mut pcm = vec![0i16; frame_size as usize * channels as usize];
    for i in 0..frame_size as usize {
        pcm[i * channels as usize] = (i as i16).wrapping_mul(13);
        pcm[i * channels as usize + 1] = (i as i16).wrapping_mul(-17);
        pcm[i * channels as usize + 2] = (i as i16).wrapping_mul(7);
    }

    let mut packet = vec![0u8; 4000];
    let packet_len = opus_multistream_encode(&mut enc, &pcm, frame_size, &mut packet);
    assert!(packet_len > 0, "encode failed: {packet_len}");

    let mut decoded = vec![0i16; frame_size as usize * channels as usize];
    let decoded_samples = opus_multistream_decode(
        &mut dec,
        &packet[..packet_len as usize],
        &mut decoded,
        frame_size,
        false,
    );
    assert_eq!(decoded_samples, frame_size);

    println!(
        "encoded {} bytes, decoded {} samples/channel",
        packet_len, decoded_samples
    );
}
