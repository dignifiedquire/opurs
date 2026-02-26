//! End-to-end DNN feature comparison: Rust vs C reference.
//!
//! Benchmarks DRED encoding, OSCE decoding, Deep PLC, and QEXT (when enabled)
//! against the C reference compiled via libopus-sys.
//!
//! Run with: `cargo bench --features tools-dnn --bench dnn_comparison`
//! For QEXT: `cargo bench --features "tools-dnn,qext" --bench dnn_comparison`

use criterion::{black_box, criterion_group, criterion_main, Criterion};

const SAMPLE_RATE: i32 = 48000;
const FRAME_SIZE: usize = 960; // 20ms at 48kHz
const NUM_FRAMES: usize = 50;

extern "C" {
    fn opus_encoder_create(
        fs: i32,
        channels: i32,
        application: i32,
        error: *mut i32,
    ) -> *mut std::ffi::c_void;
    fn opus_encode(
        st: *mut std::ffi::c_void,
        pcm: *const i16,
        frame_size: i32,
        data: *mut u8,
        max_data_bytes: i32,
    ) -> i32;
    fn opus_encoder_ctl(st: *mut std::ffi::c_void, request: i32, ...) -> i32;
    fn opus_encoder_destroy(st: *mut std::ffi::c_void);

    fn opus_decoder_create(fs: i32, channels: i32, error: *mut i32) -> *mut std::ffi::c_void;
    fn opus_decode(
        st: *mut std::ffi::c_void,
        data: *const u8,
        len: i32,
        pcm: *mut i16,
        frame_size: i32,
        decode_fec: i32,
    ) -> i32;
    fn opus_decoder_ctl(st: *mut std::ffi::c_void, request: i32, ...) -> i32;
    fn opus_decoder_destroy(st: *mut std::ffi::c_void);
}

const OPUS_APPLICATION_VOIP: i32 = 2048;
#[cfg(feature = "qext")]
const OPUS_APPLICATION_AUDIO: i32 = 2049;
const OPUS_SET_BITRATE_REQUEST: i32 = 4002;
const OPUS_SET_COMPLEXITY_REQUEST: i32 = 4010;
const OPUS_SET_PACKET_LOSS_PERC_REQUEST: i32 = 4014;
const OPUS_SET_DRED_DURATION_REQUEST: i32 = 4050;
#[cfg(feature = "osce")]
const OPUS_SET_OSCE_BWE_REQUEST: i32 = 4054;
#[cfg(feature = "qext")]
const OPUS_SET_QEXT_REQUEST: i32 = 4056;

fn generate_pcm(len: usize, seed: u32) -> Vec<i16> {
    let mut v = Vec::with_capacity(len);
    let mut state = seed;
    for i in 0..len {
        let t = i as f32 / SAMPLE_RATE as f32;
        let sample = (200.0 * t * std::f32::consts::TAU).sin() * 16000.0;
        state = state.wrapping_mul(1103515245).wrapping_add(12345);
        let noise = (state as i32 >> 16) as f32 * 0.1;
        v.push((sample + noise).clamp(-32768.0, 32767.0) as i16);
    }
    v
}

// --- DRED encode comparison ---

#[cfg(feature = "dred")]
fn bench_dred_encode_cmp(c: &mut Criterion) {
    let pcm = generate_pcm(FRAME_SIZE * NUM_FRAMES, 42);

    let mut group = c.benchmark_group("dred_encode_cmp");

    // Rust
    group.bench_function("rust/32kbps_mono", |b| {
        let mut enc =
            opurs::OpusEncoder::new(SAMPLE_RATE, 1, opurs::OPUS_APPLICATION_VOIP).unwrap();
        enc.set_bitrate(opurs::Bitrate::Bits(32000));
        enc.set_complexity(10).unwrap();
        enc.set_packet_loss_perc(10).unwrap();
        #[cfg(feature = "builtin-weights")]
        enc.load_dnn_weights().unwrap();
        enc.set_dred_duration(24).unwrap();

        let mut output = vec![0u8; 8000];
        b.iter(|| {
            for frame in 0..NUM_FRAMES {
                let start = frame * FRAME_SIZE;
                let end = start + FRAME_SIZE;
                let len = enc.encode(&pcm[start..end], &mut output);
                black_box(len);
            }
        })
    });

    // C
    group.bench_function("c/32kbps_mono", |b| {
        let mut err = 0i32;
        let enc = unsafe { opus_encoder_create(SAMPLE_RATE, 1, OPUS_APPLICATION_VOIP, &mut err) };
        assert!(!enc.is_null());
        unsafe {
            opus_encoder_ctl(enc, OPUS_SET_BITRATE_REQUEST, 32000i32);
            opus_encoder_ctl(enc, OPUS_SET_COMPLEXITY_REQUEST, 10i32);
            opus_encoder_ctl(enc, OPUS_SET_PACKET_LOSS_PERC_REQUEST, 10i32);
            opus_encoder_ctl(enc, OPUS_SET_DRED_DURATION_REQUEST, 24i32);
        }

        let mut output = vec![0u8; 8000];
        b.iter(|| {
            for frame in 0..NUM_FRAMES {
                let start = frame * FRAME_SIZE;
                let len = unsafe {
                    opus_encode(
                        enc,
                        pcm[start..].as_ptr(),
                        FRAME_SIZE as i32,
                        output.as_mut_ptr(),
                        output.len() as i32,
                    )
                };
                black_box(len);
            }
        });
        unsafe { opus_encoder_destroy(enc) };
    });

    group.finish();
}

// --- OSCE decode comparison ---

#[cfg(feature = "osce")]
fn bench_osce_decode_cmp(c: &mut Criterion) {
    let pcm = generate_pcm(FRAME_SIZE * NUM_FRAMES, 42);

    // Pre-encode with Rust (SILK-heavy VOIP)
    let mut enc = opurs::OpusEncoder::new(SAMPLE_RATE, 1, opurs::OPUS_APPLICATION_VOIP).unwrap();
    enc.set_bitrate(opurs::Bitrate::Bits(16000));
    enc.set_complexity(10).unwrap();

    let mut packets = Vec::new();
    let mut output = vec![0u8; 4000];
    for frame in 0..NUM_FRAMES {
        let start = frame * FRAME_SIZE;
        let end = start + FRAME_SIZE;
        let len = enc.encode(&pcm[start..end], &mut output);
        packets.push(output[..len as usize].to_vec());
    }

    let mut group = c.benchmark_group("osce_decode_cmp");

    // Rust
    group.bench_function("rust/16kbps_mono_c7", |b| {
        let mut dec = opurs::OpusDecoder::new(SAMPLE_RATE, 1).unwrap();
        dec.set_complexity(7).unwrap();
        #[cfg(feature = "builtin-weights")]
        dec.load_dnn_weights().unwrap();
        dec.set_osce_bwe(true);

        let mut pcm_out = vec![0i16; FRAME_SIZE];
        b.iter(|| {
            for pkt in &packets {
                let n = dec.decode(pkt, &mut pcm_out, FRAME_SIZE as i32, false);
                black_box(n);
            }
        })
    });

    // C
    group.bench_function("c/16kbps_mono_c7", |b| {
        let mut err = 0i32;
        let dec = unsafe { opus_decoder_create(SAMPLE_RATE, 1, &mut err) };
        assert!(!dec.is_null());
        unsafe {
            opus_decoder_ctl(dec, OPUS_SET_COMPLEXITY_REQUEST, 7i32);
            opus_decoder_ctl(dec, OPUS_SET_OSCE_BWE_REQUEST, 1i32);
        }

        let mut pcm_out = vec![0i16; FRAME_SIZE];
        b.iter(|| {
            for pkt in &packets {
                let n = unsafe {
                    opus_decode(
                        dec,
                        pkt.as_ptr(),
                        pkt.len() as i32,
                        pcm_out.as_mut_ptr(),
                        FRAME_SIZE as i32,
                        0,
                    )
                };
                black_box(n);
            }
        });
        unsafe { opus_decoder_destroy(dec) };
    });

    group.finish();
}

// --- Deep PLC comparison ---

fn bench_deep_plc_cmp(c: &mut Criterion) {
    let pcm = generate_pcm(FRAME_SIZE * NUM_FRAMES, 42);

    let mut enc = opurs::OpusEncoder::new(SAMPLE_RATE, 1, opurs::OPUS_APPLICATION_VOIP).unwrap();
    enc.set_bitrate(opurs::Bitrate::Bits(16000));
    enc.set_complexity(10).unwrap();

    let mut packets = Vec::new();
    let mut output = vec![0u8; 4000];
    for frame in 0..NUM_FRAMES {
        let start = frame * FRAME_SIZE;
        let end = start + FRAME_SIZE;
        let len = enc.encode(&pcm[start..end], &mut output);
        packets.push(output[..len as usize].to_vec());
    }

    let mut group = c.benchmark_group("deep_plc_cmp");

    // Rust
    group.bench_function("rust/20pct_loss", |b| {
        let mut dec = opurs::OpusDecoder::new(SAMPLE_RATE, 1).unwrap();
        dec.set_complexity(5).unwrap();
        #[cfg(feature = "builtin-weights")]
        dec.load_dnn_weights().unwrap();

        let mut pcm_out = vec![0i16; FRAME_SIZE];
        b.iter(|| {
            for (i, pkt) in packets.iter().enumerate() {
                if i % 5 == 4 {
                    let n = dec.decode(&[], &mut pcm_out, FRAME_SIZE as i32, false);
                    black_box(n);
                } else {
                    let n = dec.decode(pkt, &mut pcm_out, FRAME_SIZE as i32, false);
                    black_box(n);
                }
            }
        })
    });

    // C
    group.bench_function("c/20pct_loss", |b| {
        let mut err = 0i32;
        let dec = unsafe { opus_decoder_create(SAMPLE_RATE, 1, &mut err) };
        assert!(!dec.is_null());
        unsafe {
            opus_decoder_ctl(dec, OPUS_SET_COMPLEXITY_REQUEST, 5i32);
        }

        let mut pcm_out = vec![0i16; FRAME_SIZE];
        b.iter(|| {
            for (i, pkt) in packets.iter().enumerate() {
                if i % 5 == 4 {
                    let n = unsafe {
                        opus_decode(
                            dec,
                            std::ptr::null(),
                            0,
                            pcm_out.as_mut_ptr(),
                            FRAME_SIZE as i32,
                            0,
                        )
                    };
                    black_box(n);
                } else {
                    let n = unsafe {
                        opus_decode(
                            dec,
                            pkt.as_ptr(),
                            pkt.len() as i32,
                            pcm_out.as_mut_ptr(),
                            FRAME_SIZE as i32,
                            0,
                        )
                    };
                    black_box(n);
                }
            }
        });
        unsafe { opus_decoder_destroy(dec) };
    });

    group.finish();
}

// --- QEXT comparison ---

#[cfg(feature = "qext")]
fn bench_qext_encode_cmp(c: &mut Criterion) {
    let frame_size_96k: usize = 1920;
    let pcm = generate_pcm(frame_size_96k * 2 * NUM_FRAMES, 42); // stereo

    let mut group = c.benchmark_group("qext_encode_cmp");

    // Rust
    group.bench_function("rust/320kbps_stereo_96k", |b| {
        let mut enc = opurs::OpusEncoder::new(96000, 2, opurs::OPUS_APPLICATION_AUDIO).unwrap();
        enc.set_bitrate(opurs::Bitrate::Bits(320000));
        enc.set_complexity(10).unwrap();
        enc.set_qext(true);

        let mut output = vec![0u8; 8000];
        b.iter(|| {
            for frame in 0..NUM_FRAMES {
                let start = frame * frame_size_96k * 2;
                let end = start + frame_size_96k * 2;
                let len = enc.encode(&pcm[start..end], &mut output);
                black_box(len);
            }
        })
    });

    // C
    group.bench_function("c/320kbps_stereo_96k", |b| {
        let mut err = 0i32;
        let enc = unsafe { opus_encoder_create(96000, 2, OPUS_APPLICATION_AUDIO, &mut err) };
        assert!(!enc.is_null());
        unsafe {
            opus_encoder_ctl(enc, OPUS_SET_BITRATE_REQUEST, 320000i32);
            opus_encoder_ctl(enc, OPUS_SET_COMPLEXITY_REQUEST, 10i32);
            opus_encoder_ctl(enc, OPUS_SET_QEXT_REQUEST, 1i32);
        }

        let mut output = vec![0u8; 8000];
        b.iter(|| {
            for frame in 0..NUM_FRAMES {
                let start = frame * frame_size_96k * 2;
                let len = unsafe {
                    opus_encode(
                        enc,
                        pcm[start..].as_ptr(),
                        frame_size_96k as i32,
                        output.as_mut_ptr(),
                        output.len() as i32,
                    )
                };
                black_box(len);
            }
        });
        unsafe { opus_encoder_destroy(enc) };
    });

    group.finish();
}

#[cfg(feature = "qext")]
fn bench_qext_decode_cmp(c: &mut Criterion) {
    let frame_size_96k: usize = 1920;
    let pcm = generate_pcm(frame_size_96k * 2 * NUM_FRAMES, 42);

    // Pre-encode with Rust
    let mut enc = opurs::OpusEncoder::new(96000, 2, opurs::OPUS_APPLICATION_AUDIO).unwrap();
    enc.set_bitrate(opurs::Bitrate::Bits(320000)).unwrap();
    enc.set_complexity(10).unwrap();
    enc.set_qext(true);

    let mut packets = Vec::new();
    let mut output = vec![0u8; 8000];
    for frame in 0..NUM_FRAMES {
        let start = frame * frame_size_96k * 2;
        let end = start + frame_size_96k * 2;
        let len = enc.encode(&pcm[start..end], &mut output);
        packets.push(output[..len as usize].to_vec());
    }

    // Also pre-encode with C for the C decoder
    let mut c_packets = Vec::new();
    {
        let mut err = 0i32;
        let c_enc = unsafe { opus_encoder_create(96000, 2, OPUS_APPLICATION_AUDIO, &mut err) };
        assert!(!c_enc.is_null());
        unsafe {
            opus_encoder_ctl(c_enc, OPUS_SET_BITRATE_REQUEST, 320000i32);
            opus_encoder_ctl(c_enc, OPUS_SET_COMPLEXITY_REQUEST, 10i32);
            opus_encoder_ctl(c_enc, OPUS_SET_QEXT_REQUEST, 1i32);
        }
        let mut out = vec![0u8; 8000];
        for frame in 0..NUM_FRAMES {
            let start = frame * frame_size_96k * 2;
            let len = unsafe {
                opus_encode(
                    c_enc,
                    pcm[start..].as_ptr(),
                    frame_size_96k as i32,
                    out.as_mut_ptr(),
                    out.len() as i32,
                )
            };
            assert!(len > 0);
            c_packets.push(out[..len as usize].to_vec());
        }
        unsafe { opus_encoder_destroy(c_enc) };
    }

    let mut group = c.benchmark_group("qext_decode_cmp");

    // Rust
    group.bench_function("rust/320kbps_stereo_96k", |b| {
        let mut dec = opurs::OpusDecoder::new(96000, 2).unwrap();
        let mut pcm_out = vec![0i16; frame_size_96k * 2];
        b.iter(|| {
            for pkt in &packets {
                let n = dec.decode(pkt, &mut pcm_out, frame_size_96k as i32, false);
                black_box(n);
            }
        })
    });

    // C
    group.bench_function("c/320kbps_stereo_96k", |b| {
        let mut err = 0i32;
        let dec = unsafe { opus_decoder_create(96000, 2, &mut err) };
        assert!(!dec.is_null());
        let mut pcm_out = vec![0i16; frame_size_96k * 2];
        b.iter(|| {
            for pkt in &c_packets {
                let n = unsafe {
                    opus_decode(
                        dec,
                        pkt.as_ptr(),
                        pkt.len() as i32,
                        pcm_out.as_mut_ptr(),
                        frame_size_96k as i32,
                        0,
                    )
                };
                black_box(n);
            }
        });
        unsafe { opus_decoder_destroy(dec) };
    });

    group.finish();
}

// Wire up based on features
#[cfg(all(feature = "dred", feature = "osce", feature = "qext"))]
criterion_group!(
    benches,
    bench_dred_encode_cmp,
    bench_osce_decode_cmp,
    bench_deep_plc_cmp,
    bench_qext_encode_cmp,
    bench_qext_decode_cmp,
);

#[cfg(all(feature = "dred", feature = "osce", not(feature = "qext")))]
criterion_group!(
    benches,
    bench_dred_encode_cmp,
    bench_osce_decode_cmp,
    bench_deep_plc_cmp,
);

#[cfg(all(feature = "dred", not(feature = "osce")))]
criterion_group!(benches, bench_dred_encode_cmp, bench_deep_plc_cmp,);

#[cfg(all(not(feature = "dred"), not(feature = "osce")))]
criterion_group!(benches, bench_deep_plc_cmp,);

criterion_main!(benches);
