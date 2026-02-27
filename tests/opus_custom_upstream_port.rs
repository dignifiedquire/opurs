//! Upstream custom API fuzz-matrix port.
//!
//! Upstream C: `tests/test_opus_custom.c`

#![cfg(feature = "tools")]

mod test_common;

use opurs::arch::opus_select_arch;
use opurs::{
    Application, Bitrate, OpusCustomDecoder, OpusCustomEncoder, OpusDecoder, OpusEncoder,
    OPUS_APPLICATION_RESTRICTED_LOWDELAY, OPUS_BAD_ARG, OPUS_BITRATE_MAX, OPUS_BUFFER_TOO_SMALL,
    OPUS_INVALID_PACKET,
};
use std::f64::consts::PI;
use std::sync::{Mutex, MutexGuard, OnceLock};
use test_common::TestRng;

const MAX_PACKET: usize = 1500;
const NUM_ENCODERS: usize = 5;
const NUM_SETTING_CHANGES: usize = 40;
const FRAMES_PER_CASE: usize = 8;

#[derive(Copy, Clone, Debug)]
enum SampleFormat {
    I16,
    I24,
    Float,
}

#[derive(Clone)]
enum RustEncoder {
    Custom(Box<OpusCustomEncoder>),
    Opus(Box<OpusEncoder>),
}

#[derive(Clone)]
enum RustDecoder {
    Custom(Box<OpusCustomDecoder>),
    Opus(Box<OpusDecoder>),
}

#[derive(Copy, Clone)]
struct CaseSettings {
    bitrate: i32,
    vbr: bool,
    vbr_constraint: bool,
    complexity: i32,
    packet_loss_perc: i32,
    lsb_depth: i32,
    #[cfg(feature = "qext")]
    qext: bool,
}

struct InputBuffers {
    i16_samples: Vec<i16>,
    i24_samples: Vec<i32>,
    f32_samples: Vec<f32>,
    channels: usize,
    frame_size: usize,
}

struct RunConfig {
    sample_rate: i32,
    channels: usize,
    frame_size: usize,
    encode_fmt: SampleFormat,
    decode_fmt: SampleFormat,
}

fn test_guard() -> MutexGuard<'static, ()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

fn pick<'a, T>(rng: &mut TestRng, vals: &'a [T]) -> &'a T {
    let idx = (rng.next_u32() as usize) % vals.len();
    &vals[idx]
}

fn generate_sine_sweep_f32(
    sample_rate: i32,
    channels: usize,
    samples_per_channel: usize,
    amplitude: f64,
) -> Vec<f32> {
    let start_freq = 100.0f64;
    let end_freq = sample_rate as f64 / 2.0;
    let duration_s = samples_per_channel as f64 / sample_rate as f64;
    let b = ((end_freq + start_freq) / start_freq).ln() / duration_s;
    let a = start_freq / b;

    let mut out = vec![0.0f32; samples_per_channel * channels];
    for i in 0..samples_per_channel {
        let t = i as f64 / sample_rate as f64;
        let sample = amplitude * (2.0 * PI * a * (b * t).exp() - (b * t) - 1.0).sin();
        let s = sample as f32;
        for ch in 0..channels {
            out[i * channels + ch] = s;
        }
    }
    out
}

fn to_i16(input: &[f32]) -> Vec<i16> {
    input
        .iter()
        .map(|&x| ((x as f64 * 32767.0).round() as i32).clamp(-32768, 32767) as i16)
        .collect()
}

fn to_i24(input: &[f32]) -> Vec<i32> {
    const MAX24: i32 = (1 << 23) - 1;
    const MIN24: i32 = -(1 << 23);
    input
        .iter()
        .map(|&x| {
            ((x as f64 * MAX24 as f64).round() as i64).clamp(MIN24 as i64, MAX24 as i64) as i32
        })
        .collect()
}

fn apply_encoder_settings(enc: &mut RustEncoder, settings: CaseSettings) {
    match enc {
        RustEncoder::Custom(st) => {
            st.bitrate = settings.bitrate;
            st.vbr = settings.vbr as i32;
            st.constrained_vbr = settings.vbr_constraint as i32;
            st.complexity = settings.complexity;
            st.loss_rate = settings.packet_loss_perc;
            st.lsb_depth = settings.lsb_depth;
            #[cfg(feature = "qext")]
            {
                st.enable_qext = settings.qext as i32;
            }
        }
        RustEncoder::Opus(st) => {
            st.set_bitrate(Bitrate::Bits(settings.bitrate));
            st.set_vbr(settings.vbr);
            st.set_vbr_constraint(settings.vbr_constraint);
            st.set_complexity(settings.complexity)
                .expect("set complexity should be valid");
            st.set_packet_loss_perc(settings.packet_loss_perc)
                .expect("set packet loss should be valid");
            st.set_lsb_depth(settings.lsb_depth)
                .expect("set lsb depth should be valid");
            #[cfg(feature = "qext")]
            st.set_qext(settings.qext);
        }
    }
}

fn encode_frame(
    enc: &mut RustEncoder,
    format: SampleFormat,
    input: &InputBuffers,
    frame_idx: usize,
    packet: &mut [u8],
) -> i32 {
    let lo = frame_idx * input.frame_size * input.channels;
    let hi = lo + input.frame_size * input.channels;
    match (enc, format) {
        (RustEncoder::Custom(st), SampleFormat::I16) => {
            st.encode(&input.i16_samples[lo..hi], packet)
        }
        (RustEncoder::Custom(st), SampleFormat::I24) => {
            st.encode24(&input.i24_samples[lo..hi], packet)
        }
        (RustEncoder::Custom(st), SampleFormat::Float) => {
            st.encode_float(&input.f32_samples[lo..hi], packet)
        }
        (RustEncoder::Opus(st), SampleFormat::I16) => st.encode(&input.i16_samples[lo..hi], packet),
        (RustEncoder::Opus(st), SampleFormat::I24) => {
            st.encode24(&input.i24_samples[lo..hi], packet)
        }
        (RustEncoder::Opus(st), SampleFormat::Float) => {
            st.encode_float(&input.f32_samples[lo..hi], packet)
        }
    }
}

fn decode_frame(
    dec: &mut RustDecoder,
    format: SampleFormat,
    packet: &[u8],
    frame_size: i32,
    channels: usize,
) -> i32 {
    match (dec, format) {
        (RustDecoder::Custom(st), SampleFormat::I16) => {
            let mut out = vec![0i16; frame_size as usize * channels];
            st.decode(packet, &mut out, frame_size)
        }
        (RustDecoder::Custom(st), SampleFormat::I24) => {
            let mut out = vec![0i32; frame_size as usize * channels];
            st.decode24(packet, &mut out, frame_size)
        }
        (RustDecoder::Custom(st), SampleFormat::Float) => {
            let mut out = vec![0f32; frame_size as usize * channels];
            st.decode_float(packet, &mut out, frame_size)
        }
        (RustDecoder::Opus(st), SampleFormat::I16) => {
            let mut out = vec![0i16; frame_size as usize * channels];
            st.decode(packet, &mut out, frame_size, false)
        }
        (RustDecoder::Opus(st), SampleFormat::I24) => {
            let mut out = vec![0i32; frame_size as usize * channels];
            st.decode24(packet, &mut out, frame_size, false)
        }
        (RustDecoder::Opus(st), SampleFormat::Float) => {
            let mut out = vec![0f32; frame_size as usize * channels];
            st.decode_float(packet, &mut out, frame_size, false)
        }
    }
}

fn decode_corrupt_i16(
    dec: &mut RustDecoder,
    packet: &[u8],
    frame_size: i32,
    channels: usize,
) -> i32 {
    match dec {
        RustDecoder::Custom(st) => {
            let mut out = vec![0i16; frame_size as usize * channels];
            st.decode(packet, &mut out, frame_size)
        }
        RustDecoder::Opus(st) => {
            let mut out = vec![0i16; frame_size as usize * channels];
            st.decode(packet, &mut out, frame_size, false)
        }
    }
}

fn make_corrupt_packet(packet: &[u8], rng: &mut TestRng) -> Vec<u8> {
    if packet.is_empty() {
        return Vec::new();
    }

    let mut corrupt = packet.to_vec();

    for i in 0..5usize {
        if i < corrupt.len() && rng.next_u32().is_multiple_of(5) {
            corrupt[i] = (rng.next_u32() & 0xFF) as u8;
        }
    }

    let u = (rng.next_u32() as f64) / 4294967296.0;
    let len2 = (1.0 - packet.len() as f64 * (1e-10 + u).ln()) as usize;
    let trunc_len = packet.len().min(len2.max(1));
    corrupt.truncate(trunc_len);

    let u2 = (rng.next_u32() as f64) / 4294967296.0;
    let ber_1 = 1.0 - 100.0 * (1e-10 + u2).ln();
    let mut error_pos_bits = 0usize;
    loop {
        let u3 = (rng.next_u32() as f64) / 4294967296.0;
        let step = (-ber_1 * (1e-10 + u3).ln()) as usize;
        error_pos_bits = error_pos_bits.saturating_add(step.max(1));
        if error_pos_bits >= trunc_len * 8 {
            break;
        }
        let byte_pos = error_pos_bits / 8;
        let bit_pos = error_pos_bits & 7;
        corrupt[byte_pos] ^= 1u8 << bit_pos;
    }

    corrupt
}

fn run_case(mut enc: RustEncoder, mut dec: RustDecoder, cfg: &RunConfig, rng: &mut TestRng) {
    let samples_per_channel = cfg.frame_size * FRAMES_PER_CASE;
    let input_f32 =
        generate_sine_sweep_f32(cfg.sample_rate, cfg.channels, samples_per_channel, 0.5);
    let input = InputBuffers {
        i16_samples: to_i16(&input_f32),
        i24_samples: to_i24(&input_f32),
        f32_samples: input_f32,
        channels: cfg.channels,
        frame_size: cfg.frame_size,
    };

    let mut packet = vec![0u8; MAX_PACKET];

    for frame_idx in 0..FRAMES_PER_CASE {
        let len = encode_frame(&mut enc, cfg.encode_fmt, &input, frame_idx, &mut packet);
        assert!(len > 0, "encode failed: {len}");

        let packet_slice = &packet[..len as usize];

        let mut dec_copy = dec.clone();
        let corrupt = make_corrupt_packet(packet_slice, rng);
        let err = decode_corrupt_i16(&mut dec_copy, &corrupt, cfg.frame_size as i32, cfg.channels);
        assert!(
            err > 0
                || err == OPUS_BAD_ARG
                || err == OPUS_INVALID_PACKET
                || err == OPUS_BUFFER_TOO_SMALL,
            "unexpected corrupt-stream decode result: {err}"
        );

        let decoded = decode_frame(
            &mut dec,
            cfg.decode_fmt,
            packet_slice,
            cfg.frame_size as i32,
            cfg.channels,
        );
        assert_eq!(decoded, cfg.frame_size as i32, "decode returned {decoded}");
    }
}

#[test]
fn custom_upstream_mixed_api_matrix() {
    let _guard = test_guard();

    let mut rng = TestRng::from_iseed(0x1234_5678);

    let sampling_rates: &[i32] = if cfg!(feature = "qext") {
        &[48000, 96000]
    } else {
        &[48000]
    };
    let channels = [1, 2];
    let frame_sizes_ms_x2 = [5, 10, 20, 40];

    let bitrates = [
        6000,
        12000,
        16000,
        24000,
        32000,
        48000,
        64000,
        96000,
        510000,
        OPUS_BITRATE_MAX,
    ];
    let vbr_values = [false, true, true];
    let vbr_constraint_values = [false, true, true];
    let complexities = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    let packet_loss_perc = [0, 1, 2, 5];
    let lsb_depths = [8, 24];
    let use_custom = [false, true];
    let formats = [SampleFormat::I16, SampleFormat::I24, SampleFormat::Float];

    for _ in 0..NUM_ENCODERS {
        let sample_rate = *pick(&mut rng, sampling_rates);
        let ch = *pick(&mut rng, &channels);
        let frame_size_ms_x2 = *pick(&mut rng, &frame_sizes_ms_x2);
        let frame_size = frame_size_ms_x2 * sample_rate / 2000;

        let mut custom_encode = true;
        let mut custom_decode = true;
        if sample_rate == 48000 || (cfg!(feature = "qext") && sample_rate == 96000) {
            custom_encode = *pick(&mut rng, &use_custom);
            custom_decode = *pick(&mut rng, &use_custom);
            if !custom_encode && !custom_decode {
                continue;
            }
        }

        let arch = opus_select_arch();
        let mut enc = if custom_encode {
            RustEncoder::Custom(Box::new(
                OpusCustomEncoder::new(sample_rate, ch, arch)
                    .unwrap_or_else(|e| panic!("custom encoder create failed: {e}")),
            ))
        } else {
            RustEncoder::Opus(Box::new(
                OpusEncoder::new(sample_rate, ch, OPUS_APPLICATION_RESTRICTED_LOWDELAY)
                    .unwrap_or_else(|e| panic!("opus encoder create failed: {e}")),
            ))
        };

        let mut dec = if custom_decode {
            RustDecoder::Custom(Box::new(
                OpusCustomDecoder::new(sample_rate, ch as usize, arch)
                    .unwrap_or_else(|e| panic!("custom decoder create failed: {e}")),
            ))
        } else {
            RustDecoder::Opus(Box::new(
                OpusDecoder::new(sample_rate, ch as usize)
                    .unwrap_or_else(|e| panic!("opus decoder create failed: {e}")),
            ))
        };

        if let RustEncoder::Opus(st) = &mut enc {
            st.set_application(Application::LowDelay)
                .expect("application set should be valid");
        }

        for _ in 0..NUM_SETTING_CHANGES {
            let settings = CaseSettings {
                bitrate: *pick(&mut rng, &bitrates),
                vbr: *pick(&mut rng, &vbr_values),
                vbr_constraint: *pick(&mut rng, &vbr_constraint_values),
                complexity: *pick(&mut rng, &complexities),
                packet_loss_perc: *pick(&mut rng, &packet_loss_perc),
                lsb_depth: *pick(&mut rng, &lsb_depths),
                #[cfg(feature = "qext")]
                qext: (rng.next_u32() & 1) != 0,
            };
            apply_encoder_settings(&mut enc, settings);

            let cfg = RunConfig {
                sample_rate,
                channels: ch as usize,
                frame_size: frame_size as usize,
                encode_fmt: *pick(&mut rng, &formats),
                decode_fmt: *pick(&mut rng, &formats),
            };

            run_case(enc.clone(), dec.clone(), &cfg, &mut rng);

            match (&mut enc, &mut dec) {
                (RustEncoder::Custom(e), RustDecoder::Custom(d)) => {
                    e.reset();
                    d.reset();
                }
                (RustEncoder::Custom(e), RustDecoder::Opus(_)) => {
                    e.reset();
                }
                (RustEncoder::Opus(_), RustDecoder::Custom(d)) => {
                    d.reset();
                }
                (RustEncoder::Opus(_), RustDecoder::Opus(_)) => {}
            }
        }
    }
}
