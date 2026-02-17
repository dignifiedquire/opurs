/// Compare C and Rust float decode output sample-by-sample to find exact divergence point.
///
/// Usage: cargo run --release --features tools --example float_decode_compare -- opus_newvectors [vector_num]
use byteorder::{BigEndian, ReadBytesExt};
use std::io::{Cursor, Read};

const MAX_FRAME_SIZE: usize = 5760;
const MAX_PACKET: usize = 1500;

fn main() {
    let vector_path = std::env::args().nth(1).expect("need vector path");
    let vector_num: usize = std::env::args()
        .nth(2)
        .unwrap_or("1".to_string())
        .parse()
        .expect("vector num");
    let bitstream_path = format!("{}/testvector{:02}.bit", vector_path, vector_num);
    let data = std::fs::read(&bitstream_path).expect("could not read test vector");
    let mut cursor = Cursor::new(&data);

    // Create decoders at 48kHz stereo (native rate for CELT FB)
    let c_dec = unsafe {
        let dec = libopus_sys::opus_decoder_create(48000, 2, std::ptr::null_mut());
        assert!(!dec.is_null());
        dec
    };
    let mut rust_dec = opurs::OpusDecoder::new(48000, 2).unwrap();

    let mut packet = vec![0u8; MAX_PACKET];
    let mut total_float_diffs: u64 = 0;
    let mut total_int16_diffs: u64 = 0;
    let mut total_samples: u64 = 0;
    let mut first_float_diff_frame: Option<usize> = None;
    let mut first_int16_diff_frame: Option<usize> = None;

    let mut frame_idx = 0;
    while cursor.position() < data.len() as u64 {
        let data_bytes = cursor.read_u32::<BigEndian>().unwrap();
        let _enc_range = cursor.read_u32::<BigEndian>().unwrap();
        let pkt = &mut packet[..data_bytes as usize];
        cursor.read_exact(pkt).unwrap();

        // C float decode
        let mut c_float = vec![0.0f32; MAX_FRAME_SIZE * 2];
        let c_ret = unsafe {
            libopus_sys::opus_decode_float(
                c_dec,
                pkt.as_ptr(),
                data_bytes as i32,
                c_float.as_mut_ptr(),
                MAX_FRAME_SIZE as i32,
                0,
            )
        };
        assert!(c_ret > 0);

        // Rust float decode
        let mut rust_float = vec![0.0f32; MAX_FRAME_SIZE * 2];
        let rust_ret = opurs::opus_decode_float(
            &mut rust_dec,
            pkt,
            &mut rust_float,
            MAX_FRAME_SIZE as i32,
            0,
        );
        assert_eq!(c_ret, rust_ret);

        // C int16 decode (create fresh decoder for int16 if needed, or use separate decoders)
        // Actually, let's compare using the same float output converted to int16
        let n = c_ret as usize * 2; // stereo
        total_samples += n as u64;

        let mut frame_float_diffs = 0u64;
        let mut frame_int16_diffs = 0u64;
        let mut first_diff_idx = None;
        for i in 0..n {
            if c_float[i] != rust_float[i] {
                frame_float_diffs += 1;
                if first_diff_idx.is_none() {
                    first_diff_idx = Some(i);
                }
            }
            // Check int16 conversion using FLOAT2INT16 (per-sample, ties-to-even)
            let c_i16 = float2int16(c_float[i]);
            let r_i16 = float2int16(rust_float[i]);
            if c_i16 != r_i16 {
                frame_int16_diffs += 1;
            }
        }

        if frame_float_diffs > 0 {
            if first_float_diff_frame.is_none() {
                first_float_diff_frame = Some(frame_idx);
                let idx = first_diff_idx.unwrap();
                let c_bits = c_float[idx].to_bits();
                let r_bits = rust_float[idx].to_bits();
                let ulp_diff = (c_bits as i64 - r_bits as i64).unsigned_abs();
                println!(
                    "First float diff: frame {} sample {} (ch={} pos={})",
                    frame_idx,
                    idx,
                    idx % 2,
                    idx / 2
                );
                println!("  C:    {:.15e} (bits: 0x{:08x})", c_float[idx], c_bits);
                println!("  Rust: {:.15e} (bits: 0x{:08x})", rust_float[idx], r_bits);
                println!("  ULP diff: {}", ulp_diff);
                // Print a few more diffs
                let mut shown = 0;
                for i in idx..n.min(idx + 200) {
                    if c_float[i] != rust_float[i] && shown < 10 {
                        let cb = c_float[i].to_bits();
                        let rb = rust_float[i].to_bits();
                        let ud = (cb as i64 - rb as i64).unsigned_abs();
                        println!(
                            "  sample[{}] (ch={} pos={}): C={:.15e} Rust={:.15e} ULP={}",
                            i,
                            i % 2,
                            i / 2,
                            c_float[i],
                            rust_float[i],
                            ud
                        );
                        shown += 1;
                    }
                }
            }
            if frame_idx < 5 {
                println!(
                    "Frame {}: {} float diffs / {} samples ({:.2}%), {} int16 diffs",
                    frame_idx,
                    frame_float_diffs,
                    n,
                    frame_float_diffs as f64 / n as f64 * 100.0,
                    frame_int16_diffs,
                );
            }
        } else if frame_idx < 5 {
            println!(
                "Frame {}: 0 float diffs, 0 int16 diffs ({} samples)",
                frame_idx, n
            );
        }

        if frame_int16_diffs > 0 && first_int16_diff_frame.is_none() {
            first_int16_diff_frame = Some(frame_idx);
        }

        total_float_diffs += frame_float_diffs;
        total_int16_diffs += frame_int16_diffs;
        frame_idx += 1;
    }

    unsafe {
        libopus_sys::opus_decoder_destroy(c_dec);
    }

    println!("\n=== SUMMARY for testvector{:02} ===", vector_num);
    println!("Total frames: {}", frame_idx);
    println!("Total samples: {}", total_samples);
    println!(
        "Total float diffs: {} ({:.4}%)",
        total_float_diffs,
        total_float_diffs as f64 / total_samples as f64 * 100.0
    );
    println!("Total int16 diffs (FLOAT2INT16): {}", total_int16_diffs);
    if let Some(f) = first_float_diff_frame {
        println!("First float diff frame: {}", f);
    }
    if let Some(f) = first_int16_diff_frame {
        println!("First int16 diff frame: {}", f);
    }

    // Now also check with celt_float2int16 batch conversion
    // Reset and decode again
    let c_dec2 = unsafe {
        let dec = libopus_sys::opus_decoder_create(48000, 2, std::ptr::null_mut());
        assert!(!dec.is_null());
        dec
    };
    let mut rust_dec2 = opurs::OpusDecoder::new(48000, 2).unwrap();
    let mut cursor2 = Cursor::new(&data);

    let mut batch_int16_diffs: u64 = 0;
    let mut frame_idx2 = 0;
    while cursor2.position() < data.len() as u64 {
        let data_bytes = cursor2.read_u32::<BigEndian>().unwrap();
        let _enc_range = cursor2.read_u32::<BigEndian>().unwrap();
        let pkt = &mut packet[..data_bytes as usize];
        cursor2.read_exact(pkt).unwrap();

        // C int16 decode
        let mut c_i16 = vec![0i16; MAX_FRAME_SIZE * 2];
        let c_ret = unsafe {
            libopus_sys::opus_decode(
                c_dec2,
                pkt.as_ptr(),
                data_bytes as i32,
                c_i16.as_mut_ptr(),
                MAX_FRAME_SIZE as i32,
                0,
            )
        };
        assert!(c_ret > 0);

        // Rust int16 decode
        let mut rust_i16 = vec![0i16; MAX_FRAME_SIZE * 2];
        let rust_ret =
            opurs::opus_decode(&mut rust_dec2, pkt, &mut rust_i16, MAX_FRAME_SIZE as i32, 0);
        assert_eq!(c_ret, rust_ret);

        let n = c_ret as usize * 2;
        for i in 0..n {
            if c_i16[i] != rust_i16[i] {
                batch_int16_diffs += 1;
                if batch_int16_diffs <= 10 {
                    println!(
                        "  int16 diff: frame {} sample {} C={} Rust={} (diff={})",
                        frame_idx2,
                        i,
                        c_i16[i],
                        rust_i16[i],
                        c_i16[i] as i32 - rust_i16[i] as i32,
                    );
                }
            }
        }
        frame_idx2 += 1;
    }
    unsafe {
        libopus_sys::opus_decoder_destroy(c_dec2);
    }
    println!(
        "Total int16 diffs (opus_decode batch): {}",
        batch_int16_diffs
    );
}

fn float2int16(x: f32) -> i16 {
    let x = x * 32768.0;
    let x = x.max(-32768.0);
    let x = x.min(32767.0);
    // ties-to-even (matching lrintf / vcvtns)
    x.round_ties_even() as i32 as i16
}
