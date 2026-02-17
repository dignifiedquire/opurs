/// Check if C opus_decode_float is deterministic (C vs C, same decoder, same packet).
/// This verifies whether the C library 1.6.1 is internally consistent.
///
/// Usage: cargo run --release --features tools --example c_only_decode_check -- opus_newvectors
use byteorder::{BigEndian, ReadBytesExt};
use std::io::{Cursor, Read};

const MAX_FRAME_SIZE: usize = 5760;
const MAX_PACKET: usize = 1500;

fn main() {
    let vector_path = std::env::args().nth(1).expect("need vector path");
    let bitstream_path = format!("{}/testvector01.bit", vector_path);
    let data = std::fs::read(&bitstream_path).expect("could not read test vector");
    let mut cursor = Cursor::new(&data);

    // Create TWO C decoders and decode the same packets, check they produce identical output
    let c_dec1 = unsafe {
        let dec = libopus_sys::opus_decoder_create(48000, 2, std::ptr::null_mut());
        assert!(!dec.is_null());
        dec
    };
    let c_dec2 = unsafe {
        let dec = libopus_sys::opus_decoder_create(48000, 2, std::ptr::null_mut());
        assert!(!dec.is_null());
        dec
    };

    let mut packet = vec![0u8; MAX_PACKET];
    let mut total_diffs: u64 = 0;
    let mut frame_idx = 0;

    while cursor.position() < data.len() as u64 {
        let data_bytes = cursor.read_u32::<BigEndian>().unwrap();
        let _enc_range = cursor.read_u32::<BigEndian>().unwrap();
        let pkt = &mut packet[..data_bytes as usize];
        cursor.read_exact(pkt).unwrap();

        let mut pcm1 = vec![0.0f32; MAX_FRAME_SIZE * 2];
        let mut pcm2 = vec![0.0f32; MAX_FRAME_SIZE * 2];

        let ret1 = unsafe {
            libopus_sys::opus_decode_float(
                c_dec1,
                pkt.as_ptr(),
                data_bytes as i32,
                pcm1.as_mut_ptr(),
                MAX_FRAME_SIZE as i32,
                0,
            )
        };
        let ret2 = unsafe {
            libopus_sys::opus_decode_float(
                c_dec2,
                pkt.as_ptr(),
                data_bytes as i32,
                pcm2.as_mut_ptr(),
                MAX_FRAME_SIZE as i32,
                0,
            )
        };
        assert_eq!(ret1, ret2);
        assert!(ret1 > 0);

        let n = ret1 as usize * 2;
        for i in 0..n {
            if pcm1[i] != pcm2[i] {
                total_diffs += 1;
                if total_diffs <= 5 {
                    println!(
                        "C vs C diff: frame {} sample {} pcm1={:.15e} pcm2={:.15e}",
                        frame_idx, i, pcm1[i], pcm2[i]
                    );
                }
            }
        }
        frame_idx += 1;
    }

    unsafe {
        libopus_sys::opus_decoder_destroy(c_dec1);
        libopus_sys::opus_decoder_destroy(c_dec2);
    }

    println!("C vs C: {} frames, {} diffs", frame_idx, total_diffs);

    // Now also check: Rust decode_float with itself
    let mut cursor2 = Cursor::new(&data);
    let mut rust_dec1 = opurs::OpusDecoder::new(48000, 2).unwrap();
    let mut rust_dec2 = opurs::OpusDecoder::new(48000, 2).unwrap();
    let mut rust_diffs: u64 = 0;
    let mut frame_idx2 = 0;

    while cursor2.position() < data.len() as u64 {
        let data_bytes = cursor2.read_u32::<BigEndian>().unwrap();
        let _enc_range = cursor2.read_u32::<BigEndian>().unwrap();
        let pkt = &mut packet[..data_bytes as usize];
        cursor2.read_exact(pkt).unwrap();

        let mut pcm1 = vec![0.0f32; MAX_FRAME_SIZE * 2];
        let mut pcm2 = vec![0.0f32; MAX_FRAME_SIZE * 2];

        let ret1 =
            opurs::opus_decode_float(&mut rust_dec1, pkt, &mut pcm1, MAX_FRAME_SIZE as i32, 0);
        let ret2 =
            opurs::opus_decode_float(&mut rust_dec2, pkt, &mut pcm2, MAX_FRAME_SIZE as i32, 0);
        assert_eq!(ret1, ret2);

        let n = ret1 as usize * 2;
        for i in 0..n {
            if pcm1[i] != pcm2[i] {
                rust_diffs += 1;
            }
        }
        frame_idx2 += 1;
    }
    println!("Rust vs Rust: {} frames, {} diffs", frame_idx2, rust_diffs);
}
