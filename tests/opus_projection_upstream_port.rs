//! Direct Rust port of upstream C projection tests.
//!
//! Upstream C: `tests/test_opus_projection.c`

mod test_common;

use opurs::{
    opus_projection_ambisonics_encoder_create, opus_projection_decode,
    opus_projection_decoder_create, opus_projection_encode, Bitrate, MappingMatrix,
    OPUS_APPLICATION_AUDIO,
};
use std::sync::{Mutex, MutexGuard, OnceLock};
use test_common::TestRng;

const BUFFER_SIZE: usize = 960;
const MAX_DATA_BYTES: usize = 32768;
const MAX_FRAME_SAMPLES: usize = 5760;
const ERROR_TOLERANCE: i32 = 1;

const SIMPLE_MATRIX_ROWS: i32 = 4;
const SIMPLE_MATRIX_COLS: i32 = 3;
const SIMPLE_MATRIX_SIZE: usize = 12;
const SIMPLE_MATRIX_FRAME_SIZE: usize = 10;
const SIMPLE_MATRIX_INPUT_SIZE: usize = 30;
const SIMPLE_MATRIX_OUTPUT_SIZE: usize = 40;

const SIMPLE_MATRIX_DATA: [i16; SIMPLE_MATRIX_SIZE] =
    [0, 32767, 0, 0, 32767, 0, 0, 0, 0, 0, 0, 32767];
const SIMPLE_MATRIX_INPUT_I16: [i16; SIMPLE_MATRIX_INPUT_SIZE] = [
    32767, 0, -32768, 29491, -3277, -29491, 26214, -6554, -26214, 22938, -9830, -22938, 19661,
    -13107, -19661, 16384, -16384, -16384, 13107, -19661, -13107, 9830, -22938, -9830, 6554,
    -26214, -6554, 3277, -29491, -3277,
];
const SIMPLE_MATRIX_EXPECTED_I16: [i16; SIMPLE_MATRIX_OUTPUT_SIZE] = [
    0, 32767, 0, -32768, -3277, 29491, 0, -29491, -6554, 26214, 0, -26214, -9830, 22938, 0, -22938,
    -13107, 19661, 0, -19661, -16384, 16384, 0, -16384, -19661, 13107, 0, -13107, -22938, 9830, 0,
    -9830, -26214, 6554, 0, -6554, -29491, 3277, 0, -3277,
];

fn test_guard() -> MutexGuard<'static, ()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

fn res_to_i16(sample: f32) -> i16 {
    ((sample * 32768.0).round() as i32).clamp(-32768, 32767) as i16
}

fn assert_res_eq_i16(got: &[f32], expected: &[i16], tolerance: i32, context: &str) {
    assert_eq!(got.len(), expected.len());
    for (idx, (&g, &e)) in got.iter().zip(expected.iter()).enumerate() {
        let g_i16 = res_to_i16(g) as i32;
        let e_i16 = e as i32;
        assert!(
            (g_i16 - e_i16).abs() <= tolerance,
            "{context} mismatch at index {idx}: got {g_i16}, expected {e_i16}"
        );
    }
}

fn assert_i16_eq(got: &[i16], expected: &[i16], tolerance: i32, context: &str) {
    assert_eq!(got.len(), expected.len());
    for (idx, (&g, &e)) in got.iter().zip(expected.iter()).enumerate() {
        assert!(
            (g as i32 - e as i32).abs() <= tolerance,
            "{context} mismatch at index {idx}: got {g}, expected {e}"
        );
    }
}

/// Upstream C: `tests/test_opus_projection.c:test_simple_matrix`.
#[test]
fn projection_upstream_simple_matrix() {
    let _guard = test_guard();

    let matrix = MappingMatrix::new(
        SIMPLE_MATRIX_ROWS,
        SIMPLE_MATRIX_COLS,
        0,
        &SIMPLE_MATRIX_DATA,
    )
    .expect("mapping matrix init");

    let input_res = SIMPLE_MATRIX_INPUT_I16
        .iter()
        .map(|&v| v as f32 / 32768.0)
        .collect::<Vec<_>>();

    let mut out_in_short = vec![0f32; SIMPLE_MATRIX_OUTPUT_SIZE];
    for row in 0..SIMPLE_MATRIX_ROWS as usize {
        matrix
            .multiply_channel_in_short(
                &SIMPLE_MATRIX_INPUT_I16,
                SIMPLE_MATRIX_COLS as usize,
                &mut out_in_short[row..],
                row,
                SIMPLE_MATRIX_ROWS as usize,
                SIMPLE_MATRIX_FRAME_SIZE,
            )
            .expect("multiply_channel_in_short");
    }
    assert_res_eq_i16(
        &out_in_short,
        &SIMPLE_MATRIX_EXPECTED_I16,
        ERROR_TOLERANCE,
        "mapping_matrix_multiply_channel_in_short",
    );

    let mut out_short = vec![0i16; SIMPLE_MATRIX_OUTPUT_SIZE];
    for col in 0..SIMPLE_MATRIX_COLS as usize {
        matrix
            .multiply_channel_out_short(
                &input_res[col..],
                col,
                SIMPLE_MATRIX_COLS as usize,
                &mut out_short,
                SIMPLE_MATRIX_ROWS as usize,
                SIMPLE_MATRIX_FRAME_SIZE,
            )
            .expect("multiply_channel_out_short");
    }
    assert_i16_eq(
        &out_short,
        &SIMPLE_MATRIX_EXPECTED_I16,
        ERROR_TOLERANCE,
        "mapping_matrix_multiply_channel_out_short",
    );

    let mut out_in_float = vec![0f32; SIMPLE_MATRIX_OUTPUT_SIZE];
    for row in 0..SIMPLE_MATRIX_ROWS as usize {
        matrix
            .multiply_channel_in_float(
                &input_res,
                SIMPLE_MATRIX_COLS as usize,
                &mut out_in_float[row..],
                row,
                SIMPLE_MATRIX_ROWS as usize,
                SIMPLE_MATRIX_FRAME_SIZE,
            )
            .expect("multiply_channel_in_float");
    }
    assert_res_eq_i16(
        &out_in_float,
        &SIMPLE_MATRIX_EXPECTED_I16,
        ERROR_TOLERANCE,
        "mapping_matrix_multiply_channel_in_float",
    );

    let mut out_out_float = vec![0f32; SIMPLE_MATRIX_OUTPUT_SIZE];
    for col in 0..SIMPLE_MATRIX_COLS as usize {
        matrix
            .multiply_channel_out_float(
                &input_res[col..],
                col,
                SIMPLE_MATRIX_COLS as usize,
                &mut out_out_float,
                SIMPLE_MATRIX_ROWS as usize,
                SIMPLE_MATRIX_FRAME_SIZE,
            )
            .expect("mapping_matrix_multiply_channel_out_float");
    }
    assert_res_eq_i16(
        &out_out_float,
        &SIMPLE_MATRIX_EXPECTED_I16,
        ERROR_TOLERANCE,
        "mapping_matrix_multiply_channel_out_float",
    );
}

fn projection_channels_valid(channels: i32) -> bool {
    let order_plus_one = (channels as f64).sqrt().floor() as i32;
    let nondiegetic_channels = channels - order_plus_one * order_plus_one;
    (2..=6).contains(&order_plus_one) && (nondiegetic_channels == 0 || nondiegetic_channels == 2)
}

/// Upstream C: `tests/test_opus_projection.c:test_creation_arguments`.
#[test]
fn projection_upstream_creation_arguments() {
    let _guard = test_guard();

    for channels in 0..=254i32 {
        let mut streams = -1i32;
        let mut coupled_streams = -1i32;

        let is_projection_valid = match opus_projection_ambisonics_encoder_create(
            48000,
            channels,
            3,
            &mut streams,
            &mut coupled_streams,
            OPUS_APPLICATION_AUDIO,
        ) {
            Ok(enc) => {
                let matrix_size = enc.demixing_matrix_size();
                assert!(matrix_size > 0, "demixing matrix size must be positive");
                let mut matrix = vec![0u8; matrix_size as usize];
                enc.copy_demixing_matrix(&mut matrix)
                    .expect("copy demixing matrix");
                opus_projection_decoder_create(48000, channels, streams, coupled_streams, &matrix)
                    .is_ok()
            }
            Err(_) => false,
        };

        let is_channels_valid = projection_channels_valid(channels);
        assert_eq!(
            is_channels_valid, is_projection_valid,
            "creation-argument mismatch: channels={channels}, streams={streams}, coupled={coupled_streams}"
        );
    }
}

/// Upstream C: `tests/test_opus_projection.c:generate_music`.
fn projection_generate_music(buf: &mut [i16], len: usize, channels: usize, rng: &mut TestRng) {
    let mut a = vec![0i32; channels];
    let mut b = vec![0i32; channels];
    let mut c = vec![0i32; channels];
    let mut d = vec![0i32; channels];
    let mut j = 0i32;

    for i in 0..len {
        for k in 0..channels {
            let term = (j >> 12) ^ (((j >> 10) | (j >> 12)) & 26 & (j >> 7));
            let mut v = ((j.wrapping_mul(term) & 128) + 128) << 15;

            let r0 = rng.next_u32();
            v = (v as u32).wrapping_add(r0 & 65535) as i32;
            v = (v as u32).wrapping_sub(r0 >> 16) as i32;
            let r1 = rng.next_u32();
            v = (v as u32).wrapping_add(r1 & 65535) as i32;
            v = (v as u32).wrapping_sub(r1 >> 16) as i32;

            b[k] = v
                .wrapping_sub(a[k])
                .wrapping_add((b[k].wrapping_mul(61).wrapping_add(32)) >> 6);
            a[k] = v;
            c[k] = (30i32
                .wrapping_mul(c[k].wrapping_add(b[k]).wrapping_add(d[k]))
                .wrapping_add(32))
                >> 6;
            d[k] = b[k];
            let sample = ((c[k] + 128) >> 8).clamp(-32768, 32767) as i16;
            buf[i * channels + k] = sample;

            if i % 6 == 0 {
                j = j.wrapping_add(1);
            }
        }
    }
}

/// Upstream C: `tests/test_opus_projection.c:test_encode_decode`.
#[test]
fn projection_upstream_encode_decode_pipeline() {
    let _guard = test_guard();

    let channels = 18i32;
    let mut streams = -1i32;
    let mut coupled = -1i32;
    let mut enc = opus_projection_ambisonics_encoder_create(
        48000,
        channels,
        3,
        &mut streams,
        &mut coupled,
        OPUS_APPLICATION_AUDIO,
    )
    .expect("projection encoder create");

    let bitrate_bps = (64 * 18) * 1000 * (streams + coupled);
    enc.set_bitrate(Bitrate::Bits(bitrate_bps));

    let matrix_size = enc.demixing_matrix_size();
    assert!(matrix_size > 0, "demixing matrix size must be positive");
    let mut matrix = vec![0u8; matrix_size as usize];
    enc.copy_demixing_matrix(&mut matrix)
        .expect("copy demixing matrix");

    let mut dec = opus_projection_decoder_create(48000, channels, streams, coupled, &matrix)
        .expect("projection decoder create");

    let mut rng = TestRng::new(0);
    let mut buffer_in = vec![0i16; BUFFER_SIZE * channels as usize];
    projection_generate_music(&mut buffer_in, BUFFER_SIZE, channels as usize, &mut rng);

    let mut packet = vec![0u8; MAX_DATA_BYTES];
    let len = opus_projection_encode(&mut enc, &buffer_in, BUFFER_SIZE as i32, &mut packet);
    assert!(
        len > 0 && (len as usize) <= MAX_DATA_BYTES,
        "opus_projection_encode returned {len}"
    );

    let mut buffer_out = vec![0i16; MAX_FRAME_SAMPLES * channels as usize];
    let out_samples = opus_projection_decode(
        &mut dec,
        &packet[..len as usize],
        &mut buffer_out,
        MAX_FRAME_SAMPLES as i32,
        false,
    );
    assert_eq!(out_samples, BUFFER_SIZE as i32);
}
