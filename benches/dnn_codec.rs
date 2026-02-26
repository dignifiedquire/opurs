//! End-to-end codec benchmarks for DNN features (DRED, OSCE, Deep PLC)
//! and QEXT (96 kHz).
//!
//! Run with: `cargo bench --features "dnn,builtin-weights" --bench dnn_codec`

use criterion::{black_box, criterion_group, criterion_main, Criterion};

const SAMPLE_RATE: i32 = 48000;
const FRAME_SIZE: usize = 960; // 20ms at 48kHz
const NUM_FRAMES: usize = 50; // 1 second

fn generate_pcm(len: usize, seed: u32) -> Vec<i16> {
    let mut v = Vec::with_capacity(len);
    let mut state = seed;
    for i in 0..len {
        // Simple speech-like signal: low-frequency + some noise
        let t = i as f32 / SAMPLE_RATE as f32;
        let sample = (200.0 * t * std::f32::consts::TAU).sin() * 16000.0;
        state = state.wrapping_mul(1103515245).wrapping_add(12345);
        let noise = (state as i32 >> 16) as f32 * 0.1;
        v.push((sample + noise).clamp(-32768.0, 32767.0) as i16);
    }
    v
}

#[cfg(feature = "dred")]
fn bench_dred_encode(c: &mut Criterion) {
    let pcm = generate_pcm(FRAME_SIZE * NUM_FRAMES, 42);

    let mut group = c.benchmark_group("dred_encode");
    group.bench_function("32kbps_mono", |b| {
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
    group.finish();
}

#[cfg(feature = "osce")]
fn bench_osce_decode(c: &mut Criterion) {
    // Pre-encode VOIP packets (SILK-heavy for OSCE to work on)
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

    let mut group = c.benchmark_group("osce_decode");
    group.bench_function("16kbps_mono_complexity7", |b| {
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
    group.finish();
}

fn bench_deep_plc(c: &mut Criterion) {
    // Pre-encode packets
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

    let mut group = c.benchmark_group("deep_plc");
    group.bench_function("20pct_loss_mono", |b| {
        let mut dec = opurs::OpusDecoder::new(SAMPLE_RATE, 1).unwrap();
        dec.set_complexity(5).unwrap();
        #[cfg(feature = "builtin-weights")]
        dec.load_dnn_weights().unwrap();

        let mut pcm_out = vec![0i16; FRAME_SIZE];
        b.iter(|| {
            for (i, pkt) in packets.iter().enumerate() {
                if i % 5 == 4 {
                    // Simulate packet loss — empty data triggers PLC
                    let n = dec.decode(&[], &mut pcm_out, FRAME_SIZE as i32, false);
                    black_box(n);
                } else {
                    let n = dec.decode(pkt, &mut pcm_out, FRAME_SIZE as i32, false);
                    black_box(n);
                }
            }
        })
    });
    group.finish();
}

#[cfg(feature = "qext")]
fn bench_qext_encode(c: &mut Criterion) {
    let frame_size_96k: usize = 1920; // 20ms at 96kHz
    let pcm = generate_pcm(frame_size_96k * 2 * NUM_FRAMES, 42); // stereo

    let mut group = c.benchmark_group("qext_encode");
    group.bench_function("320kbps_stereo_96k", |b| {
        let mut enc = opurs::OpusEncoder::new(96000, 2, opurs::OPUS_APPLICATION_AUDIO).unwrap();
        enc.set_bitrate(opurs::Bitrate::Bits(320000));
        enc.set_complexity(10).unwrap();
        enc.set_qext(true);

        let mut output = vec![0u8; 8000];
        b.iter(|| {
            for frame in 0..NUM_FRAMES {
                let start = frame * frame_size_96k * 2; // stereo
                let end = start + frame_size_96k * 2;
                let len = enc.encode(&pcm[start..end], &mut output);
                black_box(len);
            }
        })
    });
    group.finish();
}

#[cfg(feature = "qext")]
fn bench_qext_decode(c: &mut Criterion) {
    let frame_size_96k: usize = 1920;
    let pcm = generate_pcm(frame_size_96k * 2 * NUM_FRAMES, 42);

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

    let mut group = c.benchmark_group("qext_decode");
    group.bench_function("320kbps_stereo_96k", |b| {
        let mut dec = opurs::OpusDecoder::new(96000, 2).unwrap();
        let mut pcm_out = vec![0i16; frame_size_96k * 2];
        b.iter(|| {
            for pkt in &packets {
                let n = dec.decode(pkt, &mut pcm_out, frame_size_96k as i32, false);
                black_box(n);
            }
        })
    });
    group.finish();
}

// Wire up criterion groups based on features
#[cfg(all(feature = "dred", feature = "osce", feature = "qext"))]
criterion_group!(
    benches,
    bench_dred_encode,
    bench_osce_decode,
    bench_deep_plc,
    bench_qext_encode,
    bench_qext_decode,
);

#[cfg(all(feature = "dred", feature = "osce", not(feature = "qext")))]
criterion_group!(
    benches,
    bench_dred_encode,
    bench_osce_decode,
    bench_deep_plc,
);

#[cfg(all(feature = "dred", not(feature = "osce")))]
criterion_group!(benches, bench_dred_encode, bench_deep_plc,);

#[cfg(all(not(feature = "dred"), not(feature = "osce")))]
criterion_group!(benches, bench_deep_plc,);

criterion_main!(benches);
