//! Bit-exact DNN comparison tool: encode/decode with C and Rust backends using DNN features.
//!
//! Tests that the Rust DNN implementation produces identical output to the C reference
//! at various encoder/decoder configurations involving DRED, Deep PLC, and OSCE.
//!
//! Usage:
//!   cargo run --release --features tools-dnn --example dnn_comparison -- <input.pcm>
//!
//! Input must be raw 16-bit signed LE PCM, 48kHz, stereo (matching IETF test vectors).

use clap::Parser;
use opurs::tools::demo::{
    opus_demo_decode, opus_demo_encode, Application, Channels, Complexity, DecodeArgs, DnnOptions,
    EncodeArgs, EncoderOptions, OpusBackend, SampleRate,
};
use std::path::PathBuf;

#[derive(Parser)]
#[command(about = "Bit-exact DNN comparison: Rust vs C reference")]
struct Cli {
    /// Input raw PCM file (16-bit signed LE, 48kHz, stereo)
    input: PathBuf,

    /// Bitrate in bps
    #[arg(long, default_value_t = 64000)]
    bitrate: u32,

    /// DRED duration in frames (0 = no DRED)
    #[arg(long, default_value_t = 10)]
    dred_duration: i32,

    /// Encoder complexity
    #[arg(long, default_value = "10")]
    complexity: Complexity,

    /// Decoder complexities to test (comma-separated, e.g. "0,5,6,7,10")
    #[arg(long, default_value = "0,5,6,7,10", value_delimiter = ',')]
    decoder_complexities: Vec<Complexity>,
}

struct TestResult {
    decoder_complexity: Complexity,
    encode_match: bool,
    decode_match: bool,
}

fn main() {
    let cli = Cli::parse();

    let pcm = std::fs::read(&cli.input).expect("failed to read input PCM file");
    eprintln!(
        "Input: {} ({} bytes, {:.2}s at 48kHz stereo)",
        cli.input.display(),
        pcm.len(),
        pcm.len() as f64 / (48000.0 * 2.0 * 2.0)
    );

    let dnn = DnnOptions::default();

    let encode_args = EncodeArgs {
        application: Application::Audio,
        sample_rate: SampleRate::R48000,
        channels: Channels::Stereo,
        bitrate: cli.bitrate,
        options: EncoderOptions {
            complexity: cli.complexity,
            dred_duration: cli.dred_duration,
            ..Default::default()
        },
    };

    eprintln!(
        "Encoding: bitrate={}, complexity={:?}, dred_duration={}",
        cli.bitrate, cli.complexity, cli.dred_duration
    );

    // Encode with both backends
    let (upstream_encoded, upstream_skip) =
        opus_demo_encode(OpusBackend::Upstream, &pcm, encode_args, &dnn);
    let (rust_encoded, rust_skip) = opus_demo_encode(OpusBackend::Rust, &pcm, encode_args, &dnn);

    let encode_match = upstream_encoded == rust_encoded;
    eprintln!(
        "Encode: upstream={} bytes, rust={} bytes, skip upstream={} rust={}, match={}",
        upstream_encoded.len(),
        rust_encoded.len(),
        upstream_skip,
        rust_skip,
        if encode_match { "YES" } else { "NO" }
    );

    if !encode_match {
        eprintln!("  WARNING: encoded bitstreams differ!");
    }

    // Decode at each decoder complexity
    let mut results = Vec::new();
    for &dec_complexity in &cli.decoder_complexities {
        let decode_args = DecodeArgs {
            sample_rate: SampleRate::R48000,
            channels: Channels::Stereo,
            options: Default::default(),
            complexity: Some(dec_complexity),
        };

        // Decode the upstream-encoded stream with both backends
        let upstream_decoded =
            opus_demo_decode(OpusBackend::Upstream, &upstream_encoded, decode_args, &dnn);
        let rust_decoded =
            opus_demo_decode(OpusBackend::Rust, &upstream_encoded, decode_args, &dnn);

        let decode_match = upstream_decoded == rust_decoded;

        eprintln!(
            "Decode complexity={:?}: upstream={} bytes, rust={} bytes, match={}",
            dec_complexity,
            upstream_decoded.len(),
            rust_decoded.len(),
            if decode_match { "YES" } else { "NO" }
        );

        if !decode_match {
            // Find first difference
            let min_len = upstream_decoded.len().min(rust_decoded.len());
            for i in 0..min_len {
                if upstream_decoded[i] != rust_decoded[i] {
                    eprintln!(
                        "  First diff at byte {}: upstream=0x{:02x}, rust=0x{:02x}",
                        i, upstream_decoded[i], rust_decoded[i]
                    );
                    break;
                }
            }
            if upstream_decoded.len() != rust_decoded.len() {
                eprintln!(
                    "  Length mismatch: upstream={}, rust={}",
                    upstream_decoded.len(),
                    rust_decoded.len()
                );
            }
        }

        results.push(TestResult {
            decoder_complexity: dec_complexity,
            encode_match,
            decode_match,
        });
    }

    // Summary
    eprintln!("\n--- Summary ---");
    let mut all_pass = true;
    for r in &results {
        let pass = r.encode_match && r.decode_match;
        if !pass {
            all_pass = false;
        }
        eprintln!(
            "  dec_complexity={:?}: encode={} decode={} {}",
            r.decoder_complexity,
            if r.encode_match { "MATCH" } else { "DIFF" },
            if r.decode_match { "MATCH" } else { "DIFF" },
            if pass { "PASS" } else { "FAIL" },
        );
    }

    if all_pass {
        eprintln!("\nAll tests PASSED: Rust DNN output is bit-exact with C reference.");
    } else {
        eprintln!("\nSome tests FAILED: Rust DNN output differs from C reference.");
        std::process::exit(1);
    }
}
