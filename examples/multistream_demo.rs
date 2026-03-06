//! Minimal multistream encode/decode demo.

use opurs::{Application, OpusMSDecoder, OpusMSEncoder, SampleRate};

fn main() {
    let frame_size = 960;
    // 3 channels encoded as 2 streams: one coupled stereo + one mono.
    let channels = 3;
    let streams = 2;
    let coupled_streams = 1;
    let mapping = [0u8, 1u8, 2u8];

    let mut enc = OpusMSEncoder::new(
        SampleRate::Hz48000,
        channels,
        streams,
        coupled_streams,
        &mapping,
        Application::Audio,
    )
    .expect("create multistream encoder");
    let mut dec = OpusMSDecoder::new(
        SampleRate::Hz48000,
        channels,
        streams,
        coupled_streams,
        &mapping,
    )
    .expect("create multistream decoder");

    let mut pcm = vec![0i16; frame_size as usize * channels as usize];
    for i in 0..frame_size as usize {
        pcm[i * channels as usize] = (i as i16).wrapping_mul(13);
        pcm[i * channels as usize + 1] = (i as i16).wrapping_mul(-17);
        pcm[i * channels as usize + 2] = (i as i16).wrapping_mul(7);
    }

    let mut packet = vec![0u8; 4000];
    let packet_len = enc.encode(&pcm, frame_size, &mut packet);
    assert!(packet_len > 0, "encode failed: {packet_len}");

    let mut decoded = vec![0i16; frame_size as usize * channels as usize];
    let decoded_samples = dec.decode(
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
