#![cfg(feature = "tools")]

use opurs::{Application, Bitrate, OpusDecoder, OpusEncoder};

fn c_has_lbrr(packet: &[u8]) -> i32 {
    unsafe { libopus_sys::opus_packet_has_lbrr(packet.as_ptr(), packet.len() as i32) }
}

#[test]
fn packet_has_lbrr_matches_c_for_fixed_packets() {
    let packets: &[&[u8]] = &[
        &[0],                // valid non-CELT TOC only
        &[1],                // valid non-CELT TOC only
        &[252, 0],           // code 0 CELT-ish minimal
        &[255, 1, 0],        // code 3 with one frame
        &[252, 1, 0],        // known decodeable silence packet shape
        &[3],                // invalid short code 3 packet
        &[255, 63],          // max frame count header path
        &[120, 10, 0, 0, 0], // mixed malformed/edge header
    ];

    for packet in packets {
        let c = c_has_lbrr(packet);
        let rust = OpusDecoder::packet_has_lbrr(packet);
        assert_eq!(rust, c, "packet={packet:?}");
    }
}

#[test]
fn packet_has_lbrr_matches_c_on_real_encoded_packets_and_finds_lbrr() {
    let mut enc = OpusEncoder::new(48_000, 1, i32::from(Application::Voip)).expect("encoder");
    enc.set_inband_fec(1).expect("set inband fec");
    enc.set_packet_loss_perc(40).expect("set packet loss");
    enc.set_bitrate(Bitrate::Bits(32_000));
    enc.set_dtx(false);

    let frame_size = 960usize;
    let mut pcm = vec![0i16; frame_size];
    let mut out = vec![0u8; 1276];
    let mut saw_lbrr = false;

    for frame_idx in 0..200usize {
        for (i, sample) in pcm.iter_mut().enumerate() {
            let t = (frame_idx * frame_size + i) as f32;
            *sample = (t * 0.013).sin().mul_add(12_000.0, 0.0) as i16;
        }

        let bytes = enc.encode(&pcm, &mut out);
        assert!(bytes > 0, "encode failed at frame {frame_idx}: {bytes}");
        let packet = &out[..bytes as usize];

        let c = c_has_lbrr(packet);
        let rust = OpusDecoder::packet_has_lbrr(packet);
        assert_eq!(rust, c, "frame={frame_idx}, packet_len={bytes}");
        if rust == 1 {
            saw_lbrr = true;
        }
    }

    assert!(
        saw_lbrr,
        "expected at least one encoded packet to carry LBRR"
    );
}
