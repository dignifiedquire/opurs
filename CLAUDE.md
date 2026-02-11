# CLAUDE.md — opurs

## What This Is

A pure Rust implementation of [libopus 1.5.2](https://github.com/xiph/opus), originally created via c2rust transpilation, now nearly fully safe idiomatic Rust. The codebase is **bit-exact** with the C reference and **nearly unsafe-free** (2 documented `unsafe` blocks remain).

## Repository Layout

```
opurs/                   # Main crate (pure Rust, no C compilation)
├── src/
│   ├── lib.rs           # Public API re-exports
│   ├── externs.rs       # libc replacements (malloc/free/memcpy via Rust alloc)
│   ├── varargs.rs       # Type-safe C varargs replacement
│   ├── util/            # Utility modules (nalgebra helpers)
│   ├── celt/            # CELT codec (low-latency, music-optimized)
│   │   ├── celt_encoder.rs   # 3523 lines — largest file
│   │   ├── celt_decoder.rs   # 1551 lines
│   │   ├── bands.rs          # 2146 lines
│   │   ├── kiss_fft.rs       # FFT implementation
│   │   ├── mdct.rs           # Modified DCT (contains 2 remaining unsafe blocks)
│   │   ├── entcode/enc/dec   # Entropy coder
│   │   ├── modes/            # Static mode tables
│   │   └── tests/            # Unit tests (cwrs32, entropy, laplace, mathops, mdct, dft, rotation, types)
│   ├── silk/            # SILK codec (speech-optimized)
│   │   ├── 65+ modules  # Quantization, prediction, gain, coding, etc.
│   │   ├── float/       # Floating-point variants (28 modules)
│   │   ├── resampler/   # Audio resampling
│   │   └── tests/       # Unit tests
│   ├── src/             # High-level integration layer
│   │   ├── opus_encoder.rs   # 3034 lines — encoder state & logic
│   │   ├── opus_decoder.rs   # 1033 lines — decoder state & logic
│   │   ├── opus.rs           # Packet handling, core API
│   │   ├── analysis.rs       # Audio analysis (tonality, music detection)
│   │   ├── repacketizer.rs   # Packet merging/splitting
│   │   ├── mlp/              # Neural network analysis
│   │   ├── opus_defines.rs   # Constants & error codes
│   │   └── opus_private.rs   # Private structs, alignment utilities
│   └── tools/           # Testing & comparison utilities (feature-gated)
│       ├── mod.rs            # Module root, re-exports
│       ├── compare.rs        # Audio comparison logic
│       └── demo/             # Shared encode/decode/backend infrastructure
├── examples/            # CLI tools (require `--features tools` unless noted)
│   ├── run_vectors2.rs      # PRIMARY: bit-exact comparison against C
│   ├── opus_demo/           # CLI encode/decode tool (clap)
│   ├── opus_compare.rs      # Audio comparison utility
│   └── repacketizer_demo.rs # Repacketizer demo (clap)
├── tests/               # Integration tests (ported from C test suite)
│   ├── opus_api.rs      # Comprehensive API tests
│   ├── opus_decode.rs   # Decoder tests
│   ├── opus_encode/     # Encoder tests + regressions
│   └── opus_padding.rs  # Packet padding tests
└── libopus-sys/         # C reference implementation (libopus 1.5.2)
    ├── opus/            # Original C source
    └── build.rs         # Compiles C code via `cc` crate
```

## Build & Test Commands

```bash
# Build
cargo build
cargo build --release
cargo build --features tools          # includes tools module + examples

# Run unit + integration tests
cargo nextest run -p opurs
cargo nextest run -p opurs --cargo-profile=release

# Run bit-exact vector tests (requires test vectors)
curl https://www.ietf.org/proceedings/98/slides/materials-98-codec-opus-newvectors-00.tar.gz -o vectors.tar.gz
tar -xzf vectors.tar.gz
cargo run --release --features tools --example run_vectors2 -- opus_newvectors

# With dump directory for debugging mismatches:
cargo run --release --features tools --example run_vectors2 -- opus_newvectors --dump-dir dump/

# Debug entropy coder divergence
cargo test --features ent-dump
```

## Correctness Requirements — CRITICAL

**Every change must maintain bit-exact output with the C reference.** This is non-negotiable.

- **Decoder**: Decoded audio samples must be **identical** to upstream C decoder output at all 5 sample rates (8/12/16/24/48 kHz), mono and stereo
- **Encoder**: Encoded bitstreams must be **byte-for-byte identical** to upstream C encoder output at all 9 test bitrates (10k–240k bps)
- The `run_vectors2` tool tests ~12 IETF test vectors x (10 decode configs + 9 encode configs) = ~228 strict comparison tests
- Run `cargo test --all` after any change. Run vector tests for changes touching encoder/decoder/codec internals.

## Current Unsafe State

Only **2 unsafe blocks** remain in `src/celt/mdct.rs` — both for ndarray interleaved view splitting where safe alternatives don't exist. Zero `unsafe fn` declarations.

## Architecture Notes

### Three codec layers:
1. **CELT** (`src/celt/`) — Low-latency codec for music/general audio. FFT-based, uses MDCT.
2. **SILK** (`src/silk/`) — Speech codec from Skype heritage. Linear prediction based.
3. **Opus integration** (`src/src/`) — Combines CELT and SILK, handles mode switching, packet framing.

### Key data flow:
- **Encoding**: Audio -> analysis.rs (detect speech/music) -> SILK and/or CELT encoder -> packet framing (opus_encoder.rs)
- **Decoding**: Packet -> parse frames (opus.rs) -> SILK and/or CELT decoder -> audio output (opus_decoder.rs)

### Implementation details:
- No inline assembly, no SIMD intrinsics, no runtime CPU detection (pure Rust)
- The `VarArgs` system in `varargs.rs` replaces C variadic functions for `_ctl` APIs
- `externs.rs` provides malloc/free/memcpy without libc — to be progressively eliminated

## Branch Strategy

- `main` — primary development branch
- `origin` — remote at `dignifiedquire/opurs`

## CI

GitHub Actions runs on Ubuntu, macOS (x86 + ARM), Windows (x86 + x86_64):
1. `cargo fmt --check` and `cargo clippy` on Ubuntu
2. `cargo nextest run` in both `dev` and `release` profiles (32-bit only in release)
3. Vector tests: downloads IETF test vectors, runs `run_vectors2` in release mode on all 6 targets
4. Docs build and unsafe audit

## Tool Preferences

- Always use `rg` (ripgrep) instead of `grep` for searching
- Use `cargo nextest run` instead of `cargo test` for running tests

## Dependencies

**Runtime**: num-traits, num-complex, bytemuck, arrayref, const-chunks, ndarray, nalgebra, itertools
**Dev**: getrandom, insta (snapshot testing)
**Features**:
- `ent-dump` — enables hex dumps of entropy coder calls for debugging divergence from C
- `tools` — enables `src/tools/` module and examples (adds libopus-sys, byteorder, clap, rayon, indicatif)

## Commit Conventions

- Prefixes: `test:`, `refactor:`, `fix:`, `perf:`, `chore:`, `docs:`
- Must pass: `cargo build`, `cargo nextest run -p opurs`, `cargo clippy --all --all-targets --features tools -- -D warnings`, `cargo fmt --check`

## Performance Notes

Current Rust version is ~20% slower than C with ASM/intrinsics. Future performance work:
- SIMD intrinsics (std::arch) for hot paths in CELT FFT, SILK pitch analysis
- Compiler hints and inlining for critical inner loops
- Profile-guided optimization
