//! End-to-end codec comparison: Rust vs C reference (libopus-sys with SIMD).
//!
//! When compiled with default features (simd enabled), the C reference uses
//! RTCD (Runtime CPU Dispatch) for SIMD acceleration (AVX2/SSE on x86, NEON on aarch64).
//!
//! Requires the `tools` feature to link against libopus-sys.
//! Run with: `cargo bench --features tools --bench codec_comparison`

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

const SAMPLE_RATE: i32 = 48000;
const FRAME_SIZE_20MS: usize = 960; // 48000 * 20 / 1000

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
    fn opus_decoder_destroy(st: *mut std::ffi::c_void);
}

const OPUS_APPLICATION_AUDIO: i32 = 2049;
const OPUS_APPLICATION_VOIP: i32 = 2048;
const OPUS_SET_BITRATE_REQUEST: i32 = 4002;
const OPUS_SET_COMPLEXITY_REQUEST: i32 = 4010;

/// Generate deterministic 16-bit PCM audio.
fn generate_pcm(num_frames: usize, channels: usize) -> Vec<i16> {
    let total_samples = num_frames * FRAME_SIZE_20MS * channels;
    let mut pcm = Vec::with_capacity(total_samples);
    let mut phase1: f64 = 0.0;
    let mut phase2: f64 = 0.0;
    let freq1 = 440.0 / SAMPLE_RATE as f64;
    let freq2 = 1000.0 / SAMPLE_RATE as f64;
    for i in 0..num_frames * FRAME_SIZE_20MS {
        let sample = ((phase1 * std::f64::consts::TAU).sin() * 16000.0
            + (phase2 * std::f64::consts::TAU).sin() * 8000.0) as i16;
        for _ch in 0..channels {
            pcm.push(sample);
        }
        phase1 += freq1;
        phase2 += freq2;
        if i % 4800 == 0 {
            phase1 += 0.001;
        }
    }
    pcm
}

/// Pre-encode audio using Rust encoder to produce packets for decode benchmarks.
fn pre_encode_rust(pcm: &[i16], channels: i32, bitrate: i32) -> Vec<Vec<u8>> {
    let mut encoder =
        opurs::OpusEncoder::new(SAMPLE_RATE, channels, opurs::OPUS_APPLICATION_AUDIO).unwrap();
    encoder.set_bitrate(opurs::Bitrate::Bits(bitrate));
    encoder.set_complexity(10).unwrap();

    let frame_samples = FRAME_SIZE_20MS * channels as usize;
    let mut packets = Vec::new();
    let mut output = vec![0u8; 1500];

    for frame in pcm.chunks_exact(frame_samples) {
        let len = encoder.encode(frame, &mut output);
        assert!(len > 0, "encode failed with {}", len);
        packets.push(output[..len as usize].to_vec());
    }
    packets
}

fn bench_encode_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("opus_encode_cmp");
    group.sample_size(20);

    let channels = 2i32;
    let num_frames = 50; // 1 second of audio
    let pcm = generate_pcm(num_frames, channels as usize);
    let frame_samples = FRAME_SIZE_20MS * channels as usize;

    for &bitrate in &[64000, 128000] {
        let label = format!("{}kbps_stereo", bitrate / 1000);

        // Rust encoder (with SIMD dispatch)
        group.bench_with_input(BenchmarkId::new("rust", &label), &bitrate, |b, &bitrate| {
            let mut encoder =
                opurs::OpusEncoder::new(SAMPLE_RATE, channels, opurs::OPUS_APPLICATION_AUDIO)
                    .unwrap();
            encoder.set_bitrate(opurs::Bitrate::Bits(bitrate));
            encoder.set_complexity(10).unwrap();
            let mut output = vec![0u8; 1500];

            b.iter(|| {
                let mut total_bytes = 0i32;
                for frame in pcm.chunks_exact(frame_samples) {
                    total_bytes += encoder.encode(frame, &mut output);
                }
                black_box(total_bytes)
            })
        });

        // C encoder (with SIMD via RTCD when compiled with simd feature)
        group.bench_with_input(BenchmarkId::new("c", &label), &bitrate, |b, &bitrate| {
            let mut error = 0i32;
            let enc = unsafe {
                opus_encoder_create(SAMPLE_RATE, channels, OPUS_APPLICATION_AUDIO, &mut error)
            };
            assert!(!enc.is_null(), "C encoder_create failed: {error}");
            unsafe {
                opus_encoder_ctl(enc, OPUS_SET_BITRATE_REQUEST, bitrate);
                opus_encoder_ctl(enc, OPUS_SET_COMPLEXITY_REQUEST, 10i32);
            }
            let mut output = vec![0u8; 1500];

            b.iter(|| {
                let mut total_bytes = 0i32;
                for frame in pcm.chunks_exact(frame_samples) {
                    let len = unsafe {
                        opus_encode(
                            enc,
                            frame.as_ptr(),
                            FRAME_SIZE_20MS as i32,
                            output.as_mut_ptr(),
                            output.len() as i32,
                        )
                    };
                    total_bytes += len;
                }
                black_box(total_bytes)
            });

            unsafe { opus_encoder_destroy(enc) };
        });
    }
    group.finish();
}

fn bench_decode_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("opus_decode_cmp");
    group.sample_size(20);

    let channels = 2i32;
    let num_frames = 50;
    let pcm = generate_pcm(num_frames, channels as usize);

    for &bitrate in &[64000, 128000] {
        let packets = pre_encode_rust(&pcm, channels, bitrate);
        let label = format!("{}kbps_stereo", bitrate / 1000);

        // Rust decoder
        group.bench_with_input(BenchmarkId::new("rust", &label), &packets, |b, packets| {
            let mut decoder = opurs::OpusDecoder::new(SAMPLE_RATE, channels as usize).unwrap();
            let mut out_pcm = vec![0i16; FRAME_SIZE_20MS * channels as usize];

            b.iter(|| {
                let mut total_samples = 0i32;
                for packet in packets {
                    total_samples +=
                        decoder.decode(packet, &mut out_pcm, FRAME_SIZE_20MS as i32, false);
                }
                black_box(total_samples)
            })
        });

        // C decoder
        group.bench_with_input(BenchmarkId::new("c", &label), &packets, |b, packets| {
            let mut error = 0i32;
            let dec = unsafe { opus_decoder_create(SAMPLE_RATE, channels, &mut error) };
            assert!(!dec.is_null(), "C decoder_create failed: {error}");
            let mut out_pcm = vec![0i16; FRAME_SIZE_20MS * channels as usize];

            b.iter(|| {
                let mut total_samples = 0i32;
                for packet in packets {
                    let len = unsafe {
                        opus_decode(
                            dec,
                            packet.as_ptr(),
                            packet.len() as i32,
                            out_pcm.as_mut_ptr(),
                            FRAME_SIZE_20MS as i32,
                            0,
                        )
                    };
                    total_samples += len;
                }
                black_box(total_samples)
            });

            unsafe { opus_decoder_destroy(dec) };
        });
    }
    group.finish();
}

fn bench_encode_mono_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("opus_encode_mono_cmp");
    group.sample_size(20);

    let channels = 1i32;
    let num_frames = 50;
    let pcm = generate_pcm(num_frames, channels as usize);
    let frame_samples = FRAME_SIZE_20MS * channels as usize;

    for &bitrate in &[16000, 64000] {
        let label = format!("{}kbps_voip", bitrate / 1000);

        // Rust encoder
        group.bench_with_input(BenchmarkId::new("rust", &label), &bitrate, |b, &bitrate| {
            let mut encoder =
                opurs::OpusEncoder::new(SAMPLE_RATE, channels, opurs::OPUS_APPLICATION_VOIP)
                    .unwrap();
            encoder.set_bitrate(opurs::Bitrate::Bits(bitrate));
            encoder.set_complexity(10).unwrap();
            let mut output = vec![0u8; 1500];

            b.iter(|| {
                let mut total_bytes = 0i32;
                for frame in pcm.chunks_exact(frame_samples) {
                    total_bytes += encoder.encode(frame, &mut output);
                }
                black_box(total_bytes)
            })
        });

        // C encoder
        group.bench_with_input(BenchmarkId::new("c", &label), &bitrate, |b, &bitrate| {
            let mut error = 0i32;
            let enc = unsafe {
                opus_encoder_create(SAMPLE_RATE, channels, OPUS_APPLICATION_VOIP, &mut error)
            };
            assert!(!enc.is_null(), "C encoder_create failed: {error}");
            unsafe {
                opus_encoder_ctl(enc, OPUS_SET_BITRATE_REQUEST, bitrate);
                opus_encoder_ctl(enc, OPUS_SET_COMPLEXITY_REQUEST, 10i32);
            }
            let mut output = vec![0u8; 1500];

            b.iter(|| {
                let mut total_bytes = 0i32;
                for frame in pcm.chunks_exact(frame_samples) {
                    let len = unsafe {
                        opus_encode(
                            enc,
                            frame.as_ptr(),
                            FRAME_SIZE_20MS as i32,
                            output.as_mut_ptr(),
                            output.len() as i32,
                        )
                    };
                    total_bytes += len;
                }
                black_box(total_bytes)
            });

            unsafe { opus_encoder_destroy(enc) };
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_encode_comparison,
    bench_decode_comparison,
    bench_encode_mono_comparison,
);
criterion_main!(benches);
