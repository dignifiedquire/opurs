//! End-to-end encode/decode benchmarks using deterministic synthetic audio.
//!
//! Measures full codec pipeline performance at representative configurations.
//!
//! Run with: `cargo bench --bench codec`

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

const SAMPLE_RATE: i32 = 48000;
const FRAME_SIZE_20MS: usize = 960; // 48000 * 20 / 1000

/// Generate deterministic 16-bit PCM audio.
/// Produces a mix of sine-like waves at different frequencies.
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

/// Pre-encode audio to produce packets for decode benchmarks.
fn pre_encode(pcm: &[i16], channels: i32, bitrate: i32) -> Vec<Vec<u8>> {
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

fn bench_encode(c: &mut Criterion) {
    let mut group = c.benchmark_group("opus_encode");
    group.sample_size(20);

    let channels = 2i32;
    let num_frames = 50; // 1 second of audio
    let pcm = generate_pcm(num_frames, channels as usize);
    let frame_samples = FRAME_SIZE_20MS * channels as usize;

    for &bitrate in &[32000, 64000, 128000] {
        let label = format!("{}kbps_stereo", bitrate / 1000);

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
    }
    group.finish();
}

fn bench_decode(c: &mut Criterion) {
    let mut group = c.benchmark_group("opus_decode");
    group.sample_size(20);

    let channels = 2i32;
    let num_frames = 50;
    let pcm = generate_pcm(num_frames, channels as usize);

    for &bitrate in &[32000, 64000, 128000] {
        let packets = pre_encode(&pcm, channels, bitrate);
        let label = format!("{}kbps_stereo", bitrate / 1000);

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
    }
    group.finish();
}

fn bench_encode_mono(c: &mut Criterion) {
    let mut group = c.benchmark_group("opus_encode_mono");
    group.sample_size(20);

    let channels = 1i32;
    let num_frames = 50;
    let pcm = generate_pcm(num_frames, channels as usize);
    let frame_samples = FRAME_SIZE_20MS * channels as usize;

    for &bitrate in &[16000, 32000, 64000] {
        let label = format!("{}kbps", bitrate / 1000);

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
    }
    group.finish();
}

criterion_group!(benches, bench_encode, bench_decode, bench_encode_mono,);
criterion_main!(benches);
