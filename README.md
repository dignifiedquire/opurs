# opurs

[![CI](https://github.com/dignifiedquire/opurs/actions/workflows/ci.yml/badge.svg)](https://github.com/dignifiedquire/opurs/actions/workflows/ci.yml)
[![Rust 1.65+](https://img.shields.io/badge/rust-1.65+-blue.svg)](https://www.rust-lang.org)
[![License: BSD-3-Clause](https://img.shields.io/badge/license-BSD--3--Clause-blue.svg)](LICENSE)

A pure Rust implementation of the [Opus audio codec](https://opus-codec.org/), bit-exact with libopus 1.3.1.

## Features

- **Pure Rust** -- no C compiler required, no FFI
- **Bit-exact** -- encoder output is byte-identical and decoder output is sample-identical to libopus 1.3.1, verified across 228 IETF test vectors
- **Nearly unsafe-free** -- only 2 documented `unsafe` blocks remain (ndarray interleaved view splitting in the MDCT)
- **Cross-platform** -- tested on Linux, macOS, and Windows (x86, x86_64, ARM64)
- **Full codec support** -- SILK (speech), CELT (music), and hybrid modes at all standard sample rates (8/12/16/24/48 kHz)

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

## Testing

```bash
# Unit and integration tests
cargo test --all

# Bit-exact vector tests (requires IETF test vectors)
curl https://www.ietf.org/proceedings/98/slides/materials-98-codec-opus-newvectors-00.tar.gz -o vectors.tar.gz
tar -xzf vectors.tar.gz
cargo run --release -p opurs-tools --bin run_vectors2 -- opus_newvectors
```

The vector test suite runs the encoder at 9 bitrates and the decoder at 10 configurations across all IETF test vectors, comparing output against the C reference implementation compiled from source in `upstream-libopus/`.

## Origin

This project started as a [c2rust](https://github.com/immunant/c2rust) transpilation of libopus 1.3.1 by [DCNick3](https://github.com/DCNick3/unsafe-libopus), then incrementally refactored toward safe, idiomatic Rust. The original transpilation eliminated the need for a C toolchain; the subsequent refactoring has brought the codebase to near-complete memory safety while maintaining bit-exact compatibility.

## License

BSD 3-Clause, same as libopus. See [LICENSE](LICENSE) for details.
