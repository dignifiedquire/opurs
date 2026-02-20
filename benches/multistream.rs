//! End-to-end multistream benchmarks: Rust vs C reference.
//!
//! Run with: `cargo bench --features tools --bench multistream`

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use libopus_sys::{
    opus_multistream_decode as c_opus_multistream_decode,
    opus_multistream_decoder_create as c_opus_multistream_decoder_create,
    opus_multistream_decoder_destroy as c_opus_multistream_decoder_destroy,
    opus_multistream_encode as c_opus_multistream_encode,
    opus_multistream_encoder_create as c_opus_multistream_encoder_create,
    opus_multistream_encoder_ctl as c_opus_multistream_encoder_ctl,
    opus_multistream_encoder_destroy as c_opus_multistream_encoder_destroy,
};

const SAMPLE_RATE: i32 = 48_000;
const CHANNELS: i32 = 2;
const STREAMS: i32 = 2;
const COUPLED_STREAMS: i32 = 0;
const FRAME_SIZE: usize = 960;
const MAPPING: [u8; 2] = [0, 1];
const MAX_PACKET: usize = 4000;

fn gen_pcm(frames: usize) -> Vec<i16> {
    let mut pcm = Vec::with_capacity(frames * FRAME_SIZE * CHANNELS as usize);
    for i in 0..frames * FRAME_SIZE {
        let t = i as f32 / SAMPLE_RATE as f32;
        let left = (f32::sin(2.0 * core::f32::consts::PI * 440.0 * t) * 20000.0) as i16;
        let right = (f32::sin(2.0 * core::f32::consts::PI * 660.0 * t) * 20000.0) as i16;
        pcm.push(left);
        pcm.push(right);
    }
    pcm
}

fn pre_encode_rust(pcm: &[i16], bitrate: i32) -> Vec<Vec<u8>> {
    let mut enc = opurs::OpusMSEncoder::new(
        SAMPLE_RATE,
        CHANNELS,
        STREAMS,
        COUPLED_STREAMS,
        &MAPPING,
        opurs::OPUS_APPLICATION_AUDIO,
    )
    .expect("rust ms encoder create");
    enc.set_bitrate(opurs::Bitrate::Bits(bitrate));
    enc.set_complexity(10).unwrap();

    let mut packets = Vec::new();
    for frame in pcm.chunks_exact(FRAME_SIZE * CHANNELS as usize) {
        let mut out = vec![0u8; MAX_PACKET];
        let len = enc.encode(frame, &mut out);
        assert!(len > 0, "rust ms encode failed: {len}");
        out.truncate(len as usize);
        packets.push(out);
    }
    packets
}

fn pre_encode_c(pcm: &[i16], bitrate: i32) -> Vec<Vec<u8>> {
    let mut err = 0i32;
    let enc = unsafe {
        c_opus_multistream_encoder_create(
            SAMPLE_RATE,
            CHANNELS,
            STREAMS,
            COUPLED_STREAMS,
            MAPPING.as_ptr(),
            opurs::OPUS_APPLICATION_AUDIO,
            &mut err as *mut _,
        )
    };
    assert!(!enc.is_null(), "c ms encoder create failed: {err}");
    unsafe {
        c_opus_multistream_encoder_ctl(enc, opurs::OPUS_SET_BITRATE_REQUEST, bitrate);
        c_opus_multistream_encoder_ctl(enc, opurs::OPUS_SET_COMPLEXITY_REQUEST, 10i32);
    }

    let mut packets = Vec::new();
    for frame in pcm.chunks_exact(FRAME_SIZE * CHANNELS as usize) {
        let mut out = vec![0u8; MAX_PACKET];
        let len = unsafe {
            c_opus_multistream_encode(
                enc,
                frame.as_ptr(),
                FRAME_SIZE as i32,
                out.as_mut_ptr(),
                out.len() as i32,
            )
        };
        assert!(len > 0, "c ms encode failed: {len}");
        out.truncate(len as usize);
        packets.push(out);
    }
    unsafe { c_opus_multistream_encoder_destroy(enc) };
    packets
}

fn bench_multistream_encode_cmp(c: &mut Criterion) {
    let mut group = c.benchmark_group("multistream_encode_cmp");
    group.sample_size(20);

    for &bitrate in &[32_000, 96_000, 192_000] {
        let pcm = gen_pcm(50);

        group.bench_with_input(
            BenchmarkId::new("rust", bitrate),
            &bitrate,
            |b, &bitrate| {
                b.iter(|| {
                    let mut enc = opurs::OpusMSEncoder::new(
                        SAMPLE_RATE,
                        CHANNELS,
                        STREAMS,
                        COUPLED_STREAMS,
                        &MAPPING,
                        opurs::OPUS_APPLICATION_AUDIO,
                    )
                    .expect("rust ms encoder create");
                    enc.set_bitrate(opurs::Bitrate::Bits(bitrate));
                    enc.set_complexity(10).unwrap();

                    let mut total = 0i32;
                    let mut out = vec![0u8; MAX_PACKET];
                    for frame in pcm.chunks_exact(FRAME_SIZE * CHANNELS as usize) {
                        total += enc.encode(frame, &mut out);
                    }
                    black_box(total);
                });
            },
        );

        group.bench_with_input(BenchmarkId::new("c", bitrate), &bitrate, |b, &bitrate| {
            b.iter(|| {
                let mut err = 0i32;
                let enc = unsafe {
                    c_opus_multistream_encoder_create(
                        SAMPLE_RATE,
                        CHANNELS,
                        STREAMS,
                        COUPLED_STREAMS,
                        MAPPING.as_ptr(),
                        opurs::OPUS_APPLICATION_AUDIO,
                        &mut err as *mut _,
                    )
                };
                assert!(!enc.is_null(), "c ms encoder create failed: {err}");
                unsafe {
                    c_opus_multistream_encoder_ctl(enc, opurs::OPUS_SET_BITRATE_REQUEST, bitrate);
                    c_opus_multistream_encoder_ctl(enc, opurs::OPUS_SET_COMPLEXITY_REQUEST, 10i32);
                }

                let mut total = 0i32;
                let mut out = vec![0u8; MAX_PACKET];
                for frame in pcm.chunks_exact(FRAME_SIZE * CHANNELS as usize) {
                    total += unsafe {
                        c_opus_multistream_encode(
                            enc,
                            frame.as_ptr(),
                            FRAME_SIZE as i32,
                            out.as_mut_ptr(),
                            out.len() as i32,
                        )
                    };
                }
                unsafe { c_opus_multistream_encoder_destroy(enc) };
                black_box(total);
            });
        });
    }

    group.finish();
}

fn bench_multistream_decode_cmp(c: &mut Criterion) {
    let mut group = c.benchmark_group("multistream_decode_cmp");
    group.sample_size(20);

    for &bitrate in &[32_000, 96_000, 192_000] {
        let pcm = gen_pcm(50);
        let packets_rust = pre_encode_rust(&pcm, bitrate);
        let packets_c = pre_encode_c(&pcm, bitrate);

        group.bench_with_input(BenchmarkId::new("rust", bitrate), &bitrate, |b, _| {
            b.iter(|| {
                let mut dec = opurs::OpusMSDecoder::new(
                    SAMPLE_RATE,
                    CHANNELS,
                    STREAMS,
                    COUPLED_STREAMS,
                    &MAPPING,
                )
                .expect("rust ms decoder create");
                let mut out = vec![0i16; FRAME_SIZE * CHANNELS as usize];
                let mut total = 0i32;
                for packet in &packets_rust {
                    total += dec.decode(packet, &mut out, FRAME_SIZE as i32, false);
                }
                black_box(total);
            });
        });

        group.bench_with_input(BenchmarkId::new("c", bitrate), &bitrate, |b, _| {
            b.iter(|| {
                let mut err = 0i32;
                let dec = unsafe {
                    c_opus_multistream_decoder_create(
                        SAMPLE_RATE,
                        CHANNELS,
                        STREAMS,
                        COUPLED_STREAMS,
                        MAPPING.as_ptr(),
                        &mut err as *mut _,
                    )
                };
                assert!(!dec.is_null(), "c ms decoder create failed: {err}");

                let mut out = vec![0i16; FRAME_SIZE * CHANNELS as usize];
                let mut total = 0i32;
                for packet in &packets_c {
                    total += unsafe {
                        c_opus_multistream_decode(
                            dec,
                            packet.as_ptr(),
                            packet.len() as i32,
                            out.as_mut_ptr(),
                            FRAME_SIZE as i32,
                            0,
                        )
                    };
                }
                unsafe { c_opus_multistream_decoder_destroy(dec) };
                black_box(total);
            });
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_multistream_encode_cmp,
    bench_multistream_decode_cmp
);
criterion_main!(benches);
