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
const MAX_PACKET: usize = 4000;
const FRAMES_PER_ITER: usize = 50;

#[derive(Clone, Copy)]
struct LayoutCase {
    name: &'static str,
    channels: i32,
    streams: i32,
    coupled_streams: i32,
    mapping: &'static [u8],
}

const MAP_MONO: [u8; 1] = [0];
const MAP_STEREO_DUAL_MONO: [u8; 2] = [0, 1];
const MAP_5_1: [u8; 6] = [0, 4, 1, 2, 3, 5];

const LAYOUTS: [LayoutCase; 3] = [
    LayoutCase {
        name: "1ch",
        channels: 1,
        streams: 1,
        coupled_streams: 0,
        mapping: &MAP_MONO,
    },
    LayoutCase {
        name: "2ch",
        channels: 2,
        streams: 2,
        coupled_streams: 0,
        mapping: &MAP_STEREO_DUAL_MONO,
    },
    LayoutCase {
        name: "6ch",
        channels: 6,
        streams: 4,
        coupled_streams: 2,
        mapping: &MAP_5_1,
    },
];

const FRAME_SIZES: [(usize, &str); 2] = [(480, "10ms"), (960, "20ms")];
const BITRATES: [i32; 3] = [32_000, 96_000, 192_000];

fn gen_pcm(layout: LayoutCase, frame_size: usize, frames: usize) -> Vec<i16> {
    let channels = layout.channels as usize;
    let samples = frames * frame_size;
    let mut pcm = Vec::with_capacity(samples * channels);
    for i in 0..samples {
        let t = i as f32 / SAMPLE_RATE as f32;
        for ch in 0..channels {
            let freq = 220.0 + 110.0 * ch as f32;
            let amp = 18000.0 - 1000.0 * ch as f32;
            let sample = (f32::sin(2.0 * core::f32::consts::PI * freq * t) * amp) as i16;
            pcm.push(sample);
        }
    }
    pcm
}

fn create_rust_encoder(layout: LayoutCase, bitrate: i32) -> opurs::OpusMSEncoder {
    let mut enc = opurs::OpusMSEncoder::new(
        SAMPLE_RATE,
        layout.channels,
        layout.streams,
        layout.coupled_streams,
        layout.mapping,
        opurs::OPUS_APPLICATION_AUDIO,
    )
    .expect("rust ms encoder create");
    enc.set_bitrate(opurs::Bitrate::Bits(bitrate));
    enc.set_complexity(10).expect("set complexity");
    enc
}

fn create_c_encoder(layout: LayoutCase, bitrate: i32) -> *mut libopus_sys::OpusMSEncoder {
    let mut err = 0i32;
    let enc = unsafe {
        c_opus_multistream_encoder_create(
            SAMPLE_RATE,
            layout.channels,
            layout.streams,
            layout.coupled_streams,
            layout.mapping.as_ptr(),
            opurs::OPUS_APPLICATION_AUDIO,
            &mut err as *mut _,
        )
    };
    assert!(!enc.is_null(), "c ms encoder create failed: {err}");
    unsafe {
        c_opus_multistream_encoder_ctl(enc, opurs::OPUS_SET_BITRATE_REQUEST, bitrate);
        c_opus_multistream_encoder_ctl(enc, opurs::OPUS_SET_COMPLEXITY_REQUEST, 10i32);
    }
    enc
}

fn create_rust_decoder(layout: LayoutCase) -> opurs::OpusMSDecoder {
    opurs::OpusMSDecoder::new(
        SAMPLE_RATE,
        layout.channels,
        layout.streams,
        layout.coupled_streams,
        layout.mapping,
    )
    .expect("rust ms decoder create")
}

fn create_c_decoder(layout: LayoutCase) -> *mut libopus_sys::OpusMSDecoder {
    let mut err = 0i32;
    let dec = unsafe {
        c_opus_multistream_decoder_create(
            SAMPLE_RATE,
            layout.channels,
            layout.streams,
            layout.coupled_streams,
            layout.mapping.as_ptr(),
            &mut err as *mut _,
        )
    };
    assert!(!dec.is_null(), "c ms decoder create failed: {err}");
    dec
}

fn pre_encode_rust(
    layout: LayoutCase,
    frame_size: usize,
    bitrate: i32,
    pcm: &[i16],
) -> Vec<Vec<u8>> {
    let channels = layout.channels as usize;
    let mut enc = create_rust_encoder(layout, bitrate);
    let mut packets = Vec::new();
    for frame in pcm.chunks_exact(frame_size * channels) {
        let mut out = vec![0u8; MAX_PACKET];
        let len = enc.encode(frame, &mut out);
        assert!(len > 0, "rust ms encode failed: {len}");
        out.truncate(len as usize);
        packets.push(out);
    }
    packets
}

fn pre_encode_c(layout: LayoutCase, frame_size: usize, bitrate: i32, pcm: &[i16]) -> Vec<Vec<u8>> {
    let channels = layout.channels as usize;
    let enc = create_c_encoder(layout, bitrate);
    let mut packets = Vec::new();
    for frame in pcm.chunks_exact(frame_size * channels) {
        let mut out = vec![0u8; MAX_PACKET];
        let len = unsafe {
            c_opus_multistream_encode(
                enc,
                frame.as_ptr(),
                frame_size as i32,
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
    group.sample_size(10);

    for layout in LAYOUTS {
        for (frame_size, frame_label) in FRAME_SIZES {
            let pcm = gen_pcm(layout, frame_size, FRAMES_PER_ITER);
            let channels = layout.channels as usize;
            for bitrate in BITRATES {
                let id = format!("{}/{}/{bitrate}", layout.name, frame_label);

                group.bench_with_input(BenchmarkId::new("rust", &id), &bitrate, |b, &bitrate| {
                    b.iter(|| {
                        let mut enc = create_rust_encoder(layout, bitrate);
                        let mut out = vec![0u8; MAX_PACKET];
                        let mut total = 0i32;
                        for frame in pcm.chunks_exact(frame_size * channels) {
                            total += enc.encode(frame, &mut out);
                        }
                        black_box(total);
                    });
                });

                group.bench_with_input(BenchmarkId::new("c", &id), &bitrate, |b, &bitrate| {
                    b.iter(|| {
                        let enc = create_c_encoder(layout, bitrate);
                        let mut out = vec![0u8; MAX_PACKET];
                        let mut total = 0i32;
                        for frame in pcm.chunks_exact(frame_size * channels) {
                            total += unsafe {
                                c_opus_multistream_encode(
                                    enc,
                                    frame.as_ptr(),
                                    frame_size as i32,
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
        }
    }

    group.finish();
}

fn bench_multistream_decode_cmp(c: &mut Criterion) {
    let mut group = c.benchmark_group("multistream_decode_cmp");
    group.sample_size(10);

    for layout in LAYOUTS {
        for (frame_size, frame_label) in FRAME_SIZES {
            for bitrate in BITRATES {
                let pcm = gen_pcm(layout, frame_size, FRAMES_PER_ITER);
                let packets_rust = pre_encode_rust(layout, frame_size, bitrate, &pcm);
                let packets_c = pre_encode_c(layout, frame_size, bitrate, &pcm);
                let channels = layout.channels as usize;
                let id = format!("{}/{}/{bitrate}", layout.name, frame_label);

                group.bench_with_input(BenchmarkId::new("rust", &id), &bitrate, |b, _| {
                    b.iter(|| {
                        let mut dec = create_rust_decoder(layout);
                        let mut out = vec![0i16; frame_size * channels];
                        let mut total = 0i32;
                        for packet in &packets_rust {
                            total += dec.decode(packet, &mut out, frame_size as i32, false);
                        }
                        black_box(total);
                    });
                });

                group.bench_with_input(BenchmarkId::new("c", &id), &bitrate, |b, _| {
                    b.iter(|| {
                        let dec = create_c_decoder(layout);
                        let mut out = vec![0i16; frame_size * channels];
                        let mut total = 0i32;
                        for packet in &packets_c {
                            total += unsafe {
                                c_opus_multistream_decode(
                                    dec,
                                    packet.as_ptr(),
                                    packet.len() as i32,
                                    out.as_mut_ptr(),
                                    frame_size as i32,
                                    0,
                                )
                            };
                        }
                        unsafe { c_opus_multistream_decoder_destroy(dec) };
                        black_box(total);
                    });
                });
            }
        }
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_multistream_encode_cmp,
    bench_multistream_decode_cmp
);
criterion_main!(benches);
