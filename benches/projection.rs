//! End-to-end projection benchmarks: Rust vs C reference.
//!
//! Run with: `cargo bench --features tools --bench projection`

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use opurs::{
    opus_projection_ambisonics_encoder_create as rust_projection_encoder_create,
    opus_projection_decode as rust_projection_decode,
    opus_projection_decoder_create as rust_projection_decoder_create,
    opus_projection_encode as rust_projection_encode, Bitrate, MappingMatrix,
    OPUS_APPLICATION_AUDIO, OPUS_OK, OPUS_PROJECTION_GET_DEMIXING_MATRIX_REQUEST,
    OPUS_PROJECTION_GET_DEMIXING_MATRIX_SIZE_REQUEST, OPUS_SET_BITRATE_REQUEST,
    OPUS_SET_COMPLEXITY_REQUEST,
};
use std::ffi::c_void;

const SAMPLE_RATE: i32 = 48_000;
const MAPPING_FAMILY_AMBISONICS: i32 = 3;
const MAX_PACKET: usize = 4000;
const FRAMES_PER_ITER: usize = 50;

const CHANNELS: [i32; 2] = [4, 9];
const FRAME_SIZES: [(usize, &str); 2] = [(480, "10ms"), (960, "20ms")];
const BITRATES: [i32; 2] = [96_000, 192_000];

unsafe extern "C" {
    fn opus_projection_ambisonics_encoder_create(
        Fs: i32,
        channels: i32,
        mapping_family: i32,
        streams: *mut i32,
        coupled_streams: *mut i32,
        application: i32,
        error: *mut i32,
    ) -> *mut c_void;
    fn opus_projection_encoder_destroy(st: *mut c_void);
    fn opus_projection_encoder_ctl(st: *mut c_void, request: i32, ...) -> i32;
    fn opus_projection_encode(
        st: *mut c_void,
        pcm: *const i16,
        frame_size: i32,
        data: *mut u8,
        max_data_bytes: i32,
    ) -> i32;

    fn opus_projection_decoder_create(
        Fs: i32,
        channels: i32,
        streams: i32,
        coupled_streams: i32,
        demixing_matrix: *mut u8,
        demixing_matrix_size: i32,
        error: *mut i32,
    ) -> *mut c_void;
    fn opus_projection_decoder_destroy(st: *mut c_void);
    fn opus_projection_decode(
        st: *mut c_void,
        data: *const u8,
        len: i32,
        pcm: *mut i16,
        frame_size: i32,
        decode_fec: i32,
    ) -> i32;
}

#[derive(Clone)]
struct PreparedProjection {
    channels: i32,
    streams: i32,
    coupled_streams: i32,
    demixing_matrix: Vec<u8>,
    packets: Vec<Vec<u8>>,
}

fn gen_pcm(channels: i32, frame_size: usize, frames: usize) -> Vec<i16> {
    let channels = channels as usize;
    let samples = frame_size * frames;
    let mut pcm = Vec::with_capacity(samples * channels);
    for i in 0..samples {
        let t = i as f32 / SAMPLE_RATE as f32;
        for ch in 0..channels {
            let freq = 220.0 + 70.0 * ch as f32;
            let amp = 16000.0 - 600.0 * ch as f32;
            let sample = (f32::sin(2.0 * core::f32::consts::PI * freq * t) * amp) as i16;
            pcm.push(sample);
        }
    }
    pcm
}

fn create_rust_encoder(channels: i32, bitrate: i32) -> (opurs::OpusProjectionEncoder, i32, i32) {
    let mut streams = -1i32;
    let mut coupled_streams = -1i32;
    let mut enc = rust_projection_encoder_create(
        SAMPLE_RATE,
        channels,
        MAPPING_FAMILY_AMBISONICS,
        &mut streams,
        &mut coupled_streams,
        OPUS_APPLICATION_AUDIO,
    )
    .expect("rust projection encoder create");
    enc.set_bitrate(Bitrate::Bits(bitrate));
    enc.set_complexity(10).expect("set complexity");
    (enc, streams, coupled_streams)
}

fn create_c_encoder(channels: i32, bitrate: i32) -> (*mut c_void, i32, i32) {
    let mut streams = -1i32;
    let mut coupled_streams = -1i32;
    let mut err = 0i32;
    let enc = unsafe {
        opus_projection_ambisonics_encoder_create(
            SAMPLE_RATE,
            channels,
            MAPPING_FAMILY_AMBISONICS,
            &mut streams,
            &mut coupled_streams,
            OPUS_APPLICATION_AUDIO,
            &mut err as *mut _,
        )
    };
    assert!(!enc.is_null(), "c projection encoder create failed: {err}");
    let set_bitrate =
        unsafe { opus_projection_encoder_ctl(enc, OPUS_SET_BITRATE_REQUEST, bitrate) };
    assert_eq!(set_bitrate, OPUS_OK, "c projection set bitrate failed");
    let set_complexity =
        unsafe { opus_projection_encoder_ctl(enc, OPUS_SET_COMPLEXITY_REQUEST, 10i32) };
    assert_eq!(
        set_complexity, OPUS_OK,
        "c projection set complexity failed"
    );
    (enc, streams, coupled_streams)
}

fn fetch_c_demixing_matrix(enc: *mut c_void) -> Vec<u8> {
    let mut matrix_size = 0i32;
    let ret = unsafe {
        opus_projection_encoder_ctl(
            enc,
            OPUS_PROJECTION_GET_DEMIXING_MATRIX_SIZE_REQUEST,
            &mut matrix_size as *mut _,
        )
    };
    assert_eq!(ret, OPUS_OK, "c projection matrix size ctl failed");
    assert!(matrix_size > 0, "invalid c projection matrix size");

    let mut matrix = vec![0u8; matrix_size as usize];
    let ret = unsafe {
        opus_projection_encoder_ctl(
            enc,
            OPUS_PROJECTION_GET_DEMIXING_MATRIX_REQUEST,
            matrix.as_mut_ptr(),
            matrix_size,
        )
    };
    assert_eq!(ret, OPUS_OK, "c projection matrix ctl failed");
    matrix
}

fn create_rust_decoder(prepared: &PreparedProjection) -> opurs::OpusProjectionDecoder {
    rust_projection_decoder_create(
        SAMPLE_RATE,
        prepared.channels,
        prepared.streams,
        prepared.coupled_streams,
        &prepared.demixing_matrix,
    )
    .expect("rust projection decoder create")
}

fn create_c_decoder(prepared: &PreparedProjection) -> *mut c_void {
    let mut err = 0i32;
    let dec = unsafe {
        opus_projection_decoder_create(
            SAMPLE_RATE,
            prepared.channels,
            prepared.streams,
            prepared.coupled_streams,
            prepared.demixing_matrix.as_ptr() as *mut u8,
            prepared.demixing_matrix.len() as i32,
            &mut err as *mut _,
        )
    };
    assert!(!dec.is_null(), "c projection decoder create failed: {err}");
    dec
}

fn pre_encode_rust(
    channels: i32,
    frame_size: usize,
    bitrate: i32,
    pcm: &[i16],
) -> PreparedProjection {
    let (mut enc, streams, coupled_streams) = create_rust_encoder(channels, bitrate);
    let matrix_size = enc.demixing_matrix_size() as usize;
    let mut demixing_matrix = vec![0u8; matrix_size];
    enc.copy_demixing_matrix(&mut demixing_matrix)
        .expect("copy demixing matrix");

    let mut packets = Vec::new();
    for frame in pcm.chunks_exact(frame_size * channels as usize) {
        let mut packet = vec![0u8; MAX_PACKET];
        let len = rust_projection_encode(&mut enc, frame, frame_size as i32, &mut packet);
        assert!(len > 0, "rust projection encode failed: {len}");
        packet.truncate(len as usize);
        packets.push(packet);
    }
    PreparedProjection {
        channels,
        streams,
        coupled_streams,
        demixing_matrix,
        packets,
    }
}

fn pre_encode_c(channels: i32, frame_size: usize, bitrate: i32, pcm: &[i16]) -> PreparedProjection {
    let (enc, streams, coupled_streams) = create_c_encoder(channels, bitrate);
    let demixing_matrix = fetch_c_demixing_matrix(enc);

    let mut packets = Vec::new();
    for frame in pcm.chunks_exact(frame_size * channels as usize) {
        let mut packet = vec![0u8; MAX_PACKET];
        let len = unsafe {
            opus_projection_encode(
                enc,
                frame.as_ptr(),
                frame_size as i32,
                packet.as_mut_ptr(),
                packet.len() as i32,
            )
        };
        assert!(len > 0, "c projection encode failed: {len}");
        packet.truncate(len as usize);
        packets.push(packet);
    }
    unsafe { opus_projection_encoder_destroy(enc) };

    PreparedProjection {
        channels,
        streams,
        coupled_streams,
        demixing_matrix,
        packets,
    }
}

fn bench_projection_encode_cmp(c: &mut Criterion) {
    let mut group = c.benchmark_group("projection_encode_cmp");
    group.sample_size(10);

    for channels in CHANNELS {
        for (frame_size, frame_label) in FRAME_SIZES {
            let pcm = gen_pcm(channels, frame_size, FRAMES_PER_ITER);
            for bitrate in BITRATES {
                let id = format!("{channels}ch_{frame_label}_{bitrate}");

                group.bench_with_input(BenchmarkId::new("rust", &id), &bitrate, |b, &bitrate| {
                    b.iter(|| {
                        let (mut enc, _, _) = create_rust_encoder(channels, bitrate);
                        let mut packet = vec![0u8; MAX_PACKET];
                        let mut total = 0i32;
                        for frame in pcm.chunks_exact(frame_size * channels as usize) {
                            let len = rust_projection_encode(
                                &mut enc,
                                frame,
                                frame_size as i32,
                                &mut packet,
                            );
                            total += len;
                        }
                        black_box(total);
                    });
                });

                group.bench_with_input(BenchmarkId::new("c", &id), &bitrate, |b, &bitrate| {
                    b.iter(|| {
                        let (enc, _, _) = create_c_encoder(channels, bitrate);
                        let mut packet = vec![0u8; MAX_PACKET];
                        let mut total = 0i32;
                        for frame in pcm.chunks_exact(frame_size * channels as usize) {
                            total += unsafe {
                                opus_projection_encode(
                                    enc,
                                    frame.as_ptr(),
                                    frame_size as i32,
                                    packet.as_mut_ptr(),
                                    packet.len() as i32,
                                )
                            };
                        }
                        unsafe { opus_projection_encoder_destroy(enc) };
                        black_box(total);
                    });
                });
            }
        }
    }

    group.finish();
}

fn bench_projection_decode_cmp(c: &mut Criterion) {
    let mut group = c.benchmark_group("projection_decode_cmp");
    group.sample_size(10);

    for channels in CHANNELS {
        for (frame_size, frame_label) in FRAME_SIZES {
            for bitrate in BITRATES {
                let pcm = gen_pcm(channels, frame_size, FRAMES_PER_ITER);
                let prepared_rust = pre_encode_rust(channels, frame_size, bitrate, &pcm);
                let prepared_c = pre_encode_c(channels, frame_size, bitrate, &pcm);
                let id = format!("{channels}ch_{frame_label}_{bitrate}");

                group.bench_function(BenchmarkId::new("rust", &id), |b| {
                    b.iter(|| {
                        let mut dec = create_rust_decoder(&prepared_rust);
                        let mut out = vec![0i16; frame_size * channels as usize];
                        let mut total = 0i32;
                        for packet in &prepared_rust.packets {
                            total += rust_projection_decode(
                                &mut dec,
                                packet,
                                &mut out,
                                frame_size as i32,
                                false,
                            );
                        }
                        black_box(total);
                    });
                });

                group.bench_function(BenchmarkId::new("c", &id), |b| {
                    b.iter(|| {
                        let dec = create_c_decoder(&prepared_c);
                        let mut out = vec![0i16; frame_size * channels as usize];
                        let mut total = 0i32;
                        for packet in &prepared_c.packets {
                            total += unsafe {
                                opus_projection_decode(
                                    dec,
                                    packet.as_ptr(),
                                    packet.len() as i32,
                                    out.as_mut_ptr(),
                                    frame_size as i32,
                                    0,
                                )
                            };
                        }
                        unsafe { opus_projection_decoder_destroy(dec) };
                        black_box(total);
                    });
                });
            }
        }
    }

    group.finish();
}

fn bench_projection_matrix_apply(c: &mut Criterion) {
    let mut group = c.benchmark_group("projection_matrix_apply");
    group.sample_size(10);

    for channels in [4i32, 9, 16] {
        for (frame_size, frame_label) in FRAME_SIZES {
            let (enc, streams, coupled_streams) = create_rust_encoder(channels, BITRATES[0]);
            let input_channels = (streams + coupled_streams) as usize;
            let mut demixing_matrix = vec![0u8; enc.demixing_matrix_size() as usize];
            enc.copy_demixing_matrix(&mut demixing_matrix)
                .expect("copy demixing matrix");
            let matrix = MappingMatrix::from_bytes_le(
                channels,
                streams + coupled_streams,
                0,
                &demixing_matrix,
            )
            .expect("mapping matrix from bytes");
            let input = gen_pcm(streams + coupled_streams, frame_size, 1);
            let id = format!("{channels}ch_{frame_label}");

            group.bench_function(BenchmarkId::from_parameter(id), |b| {
                b.iter(|| {
                    let mut output = vec![0i16; frame_size * channels as usize];
                    for input_row in 0..input_channels {
                        matrix
                            .multiply_channel_out_short_i16(
                                &input[input_row..],
                                input_row,
                                input_channels,
                                &mut output,
                                channels as usize,
                                frame_size,
                            )
                            .expect("matrix apply");
                    }
                    black_box(output);
                });
            });
        }
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_projection_encode_cmp,
    bench_projection_decode_cmp,
    bench_projection_matrix_apply
);
criterion_main!(benches);
