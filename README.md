# opurs

[![CI](https://github.com/dignifiedquire/opurs/actions/workflows/ci.yml/badge.svg)](https://github.com/dignifiedquire/opurs/actions/workflows/ci.yml)
[![Rust 1.87+](https://img.shields.io/badge/rust-1.87+-blue.svg)](https://www.rust-lang.org)
[![License: BSD-3-Clause](https://img.shields.io/badge/license-BSD--3--Clause-blue.svg)](LICENSE)

A pure Rust implementation of the [Opus audio codec](https://opus-codec.org/), bit-exact with libopus 1.6.1.

## Features

- **Pure Rust** -- no C compiler required, no FFI
- **Bit-exact** -- encoder output is byte-identical and decoder output is sample-identical to libopus 1.6.1, verified across 228+ IETF test vectors
- **SIMD accelerated** -- runtime CPU detection on x86/x86_64 (SSE through AVX2+FMA) and compile-time NEON on aarch64
- **Full codec support** -- SILK (speech), CELT (music), and hybrid modes at all standard sample rates (8/12/16/24/48 kHz), mono and stereo
- **Multistream & projection** -- full multistream encoder/decoder and ambisonics projection APIs
- **Cross-platform** -- tested on Linux, macOS, and Windows (x86, x86_64, ARM64)

## Usage

Add to your `Cargo.toml`:

```toml
[dependencies]
opurs = "0.3"
```

### Encoding

```rust
use opurs::{OpusEncoder, Application};

let mut encoder = OpusEncoder::new(48000, 2, Application::Audio).unwrap();
let input: Vec<i16> = vec![0i16; 960]; // 20ms at 48kHz
let mut output = vec![0u8; 4000];

let len = encoder.encode(&input, &mut output).unwrap();
println!("Encoded {} bytes", len);
```

### Decoding

```rust
use opurs::OpusDecoder;

let mut decoder = OpusDecoder::new(48000, 2).unwrap();
let mut output = vec![0i16; 960 * 2]; // stereo

let samples = decoder.decode(Some(encoded_packet), &mut output, false).unwrap();
println!("Decoded {} samples per channel", samples);
```

### Configuration

```rust
use opurs::{Bitrate, Signal};

encoder.set_bitrate(Bitrate::Bits(128000)).unwrap();
encoder.set_complexity(10).unwrap();
encoder.set_signal(Signal::Music).unwrap();
```

## Cargo Features

| Feature | Default | Description |
|---------|---------|-------------|
| `simd` | yes | SIMD acceleration via platform intrinsics (SSE/AVX2 on x86, NEON on aarch64) with runtime CPU detection |
| `qext` | no | Quality Extension (Opus HD): 96 kHz support, extended bandwidth, 32-bit samples |
| `deep-plc` | no | Deep Packet Loss Concealment -- neural vocoder for speech recovery during packet loss |
| `dred` | no | Deep REDundancy -- encodes redundant latent representations for FEC. Implies `deep-plc` |
| `osce` | no | Opus Speech Clarity Enhancement -- post-filter for SILK decoded speech. Implies `deep-plc` |
| `dnn` | no | Enables all DNN features (`deep-plc` + `dred` + `osce`) |
| `builtin-weights` | no | Compile ~4.7 MB of DNN weight data into the binary |
| `ent-dump` | no | Hex dumps of entropy coder calls for debugging divergence from C |
| `tools` | no | Testing/comparison tools and CLI examples (pulls in `libopus-sys`, `clap`, `rayon`, `indicatif`) |

### DNN features

```toml
[dependencies]
opurs = { version = "0.3", features = ["dnn"] }
```

When DNN features are compiled in but no model weights are loaded, the codec behaves identically to the non-DNN build. The IETF test vectors pass with or without DNN features enabled.

### Scalar-only builds

Disable SIMD with `default-features = false` for a scalar-only build. SIMD produces valid but potentially different floating-point results from scalar due to accumulation order differences. Bit-exact comparison with the C reference requires matching the SIMD level (both Rust and C SIMD, or both scalar).

## SIMD Acceleration

The `simd` feature provides hardware-accelerated implementations for performance-critical codec paths:

| Architecture | Tiers | Accelerated functions |
|---|---|---|
| x86/x86_64 | AVX2+FMA, SSE4.1, SSE2, SSE | Pitch xcorrelation, inner products, noise shaping (NSQ/NSQ_del_dec), PVQ search, comb filter, DNN inference |
| aarch64 | NEON (always available) | Pitch xcorrelation, inner products, noise shaping, DNN inference |

CPU features are detected at runtime on x86 via CPUID. On aarch64, NEON is always available; DOTPROD is detected via platform APIs.

## Development

See [DEVELOPMENT.md](DEVELOPMENT.md) for build commands, testing, benchmarks, CLI examples, and commit conventions.

See [BENCHMARKS.md](BENCHMARKS.md) for detailed performance results and Rust-vs-C comparisons.

## Origin

This project started as a [c2rust](https://github.com/immunant/c2rust) transpilation of libopus by [DCNick3](https://github.com/DCNick3/unsafe-libopus), then incrementally refactored toward safe, idiomatic Rust while maintaining bit-exact compatibility with the C reference. SIMD dispatch and 1.6.1 feature parity (QEXT, DNN/DRED/OSCE) were added subsequently.

## License

BSD 3-Clause, same as libopus. See [LICENSE](LICENSE) for details.
