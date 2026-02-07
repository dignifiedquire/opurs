# CLAUDE.md — unsafe-libopus

## What This Is

A Rust port of [libopus 1.3.1](https://github.com/xiph/opus) originally created via c2rust transpilation, now being incrementally refactored toward safe, idiomatic Rust. The end goal is a **fully safe Rust implementation** that is **bit-exact** with the C reference and **performance-matching**.

## Repository Layout

```
unsafe-libopus/          # Main crate (pure Rust, no C compilation)
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
│   │   ├── mdct.rs           # Modified DCT
│   │   ├── entcode/enc/dec   # Entropy coder
│   │   ├── modes/            # Static mode tables
│   │   └── tests/            # Unit tests (cwrs32, entropy, laplace, mathops, mdct, dft, rotation, types)
│   ├── silk/            # SILK codec (speech-optimized)
│   │   ├── 65+ modules  # Quantization, prediction, gain, coding, etc.
│   │   ├── float/       # Floating-point variants (28 modules)
│   │   ├── resampler/   # Audio resampling
│   │   └── tests/       # Unit tests
│   └── src/             # High-level integration layer
│       ├── opus_encoder.rs   # 3034 lines — encoder state & logic
│       ├── opus_decoder.rs   # 1033 lines — decoder state & logic
│       ├── opus.rs           # Packet handling, core API
│       ├── analysis.rs       # Audio analysis (tonality, music detection) — FULLY SAFE
│       ├── repacketizer.rs   # Packet merging/splitting — FULLY SAFE
│       ├── mlp/              # Neural network analysis — FULLY SAFE
│       ├── opus_defines.rs   # Constants & error codes
│       └── opus_private.rs   # Private structs, alignment utilities
├── tests/               # Integration tests (ported from C test suite)
│   ├── opus_api.rs      # Comprehensive API tests
│   ├── opus_decode.rs   # Decoder tests
│   ├── opus_encode/     # Encoder tests + regressions
│   └── opus_padding.rs  # Packet padding tests
├── upstream-libopus/    # C reference implementation (libopus 1.3.1)
│   ├── opus/            # Original C source
│   └── build.rs         # Compiles C code via `cc` crate
└── unsafe-libopus-tools/  # Testing & comparison utilities
    └── src/bin/
        ├── run_vectors2.rs      # PRIMARY: bit-exact comparison against C
        ├── opus_demo.rs         # CLI encode/decode tool
        ├── opus_compare.rs      # Audio comparison utility
        └── repacketizer_demo.rs # Repacketizer demo
```

## Build & Test Commands

```bash
# Build
cargo build
cargo build --release          # release profile keeps debug symbols

# Run unit + integration tests
cargo test --all
cargo test --all --release

# Run bit-exact vector tests (requires test vectors)
# First download test vectors:
curl https://www.ietf.org/proceedings/98/slides/materials-98-codec-opus-newvectors-00.tar.gz -o vectors.tar.gz
tar -xzf vectors.tar.gz
# Then run:
cargo run --release -p unsafe-libopus-tools --bin run_vectors2 -- opus_newvectors
# With dump directory for debugging mismatches:
cargo run --release -p unsafe-libopus-tools --bin run_vectors2 -- opus_newvectors --dump-dir dump/

# Debug entropy coder divergence
cargo test --features ent-dump
```

## Correctness Requirements — CRITICAL

**Every change must maintain bit-exact output with the C reference.** This is non-negotiable.

- **Decoder**: Decoded audio samples must be **identical** to upstream C decoder output at all 5 sample rates (8/12/16/24/48 kHz), mono and stereo
- **Encoder**: Encoded bitstreams must be **byte-for-byte identical** to upstream C encoder output at all 9 test bitrates (10k–240k bps)
- The `run_vectors2` tool tests ~12 IETF test vectors × (10 decode configs + 9 encode configs) = ~228 strict comparison tests
- Run `cargo test --all` after any change. Run vector tests for changes touching encoder/decoder/codec internals.

## Current Unsafe State

- **156 source files** in `src/`, ~42K lines of Rust
- **~405 unsafe occurrences** (unsafe fn declarations + unsafe blocks)
- **70 files** contain zero `unsafe` keyword
- **86 files** still contain some unsafe code

### Files with most `unsafe fn` declarations (refactoring priorities):
| File | unsafe fn count |
|------|----------------|
| `src/src/opus_encoder.rs` | 30 |
| `src/celt/bands.rs` | 24 |
| `src/celt/celt_encoder.rs` | 20 |
| `src/src/opus_decoder.rs` | 15 |
| `src/celt/pitch.rs` | 10 |
| `src/src/analysis.rs` | 9 (fn signatures only, body is safe) |
| `src/celt/vq.rs` | 9 |
| `src/celt/quant_bands.rs` | 9 |

### Already refactored to safe Rust:
- `src/src/repacketizer.rs` — full safe API
- `src/src/analysis.rs` — no unsafe blocks (some fn signatures still marked unsafe for callers)
- `src/src/mlp/` — fully safe
- `opus_pcm_soft_clip` — safe
- Many SILK modules on `master`: PLC, CNG, decode_core, stereo_MS_to_LR, decode_pulses, decode_parameters, NLSF2A, LPC_fit, etc.

## Refactoring Patterns

### The established pattern for making code safe:

1. **Start from the public API** and work inward
2. **Replace raw pointers with slices** (`*const T` / `*mut T` → `&[T]` / `&mut [T]`)
3. **Replace C-style malloc/free with Rust types** (Vec, Box, arrays)
4. **Replace pointer arithmetic with slice indexing**
5. **Add `&[u8]` parameters** instead of `*const u8` + length pairs
6. **Use `copy_within`** for overlapping memcpy (memmove) patterns
7. **Keep function signatures compatible** during transition — change internals first, then signatures
8. **Add doc comments** as you go (see repacketizer.rs for the standard)

### Conventions:
- Module-level `#[allow(non_camel_case_types, non_snake_case, non_upper_case_globals, unused_assignments)]` on `celt`, `silk`, `src` modules — these are from c2rust and will be cleaned up per-file
- `opus_encoder_ctl!()` / `opus_decoder_ctl!()` macros replace C varargs
- `externs.rs` provides malloc/free/memcpy without libc — these should be progressively eliminated as code moves to safe Rust

## Architecture Notes

### Three codec layers:
1. **CELT** (`src/celt/`) — Low-latency codec for music/general audio. FFT-based, uses MDCT.
2. **SILK** (`src/silk/`) — Speech codec from Skype heritage. Linear prediction based.
3. **Opus integration** (`src/src/`) — Combines CELT and SILK, handles mode switching, packet framing.

### Key data flow:
- **Encoding**: Audio → analysis.rs (detect speech/music) → SILK and/or CELT encoder → packet framing (opus_encoder.rs)
- **Decoding**: Packet → parse frames (opus.rs) → SILK and/or CELT decoder → audio output (opus_decoder.rs)

### Important implementation details:
- No inline assembly, no SIMD intrinsics, no runtime CPU detection (pure Rust)
- C version is ~20% faster due to these features — performance gap to close
- The `VarArgs` system in `varargs.rs` replaces C variadic functions for `_ctl` APIs
- Frame sizes stored as offsets into a data buffer (not raw pointers) after repacketizer refactoring

## Branch Strategy

- `master` — stable, SILK module safety refactoring (bottom-up approach)
- `dig-safe` — current working branch, API-level safety refactoring (top-down approach, repacketizer/packet/opus.rs)
- `dig-refactor` — experimental refactoring work
- `wip-enc` — encoder work in progress
- `origin` — upstream (DCNick3's repo)
- `dig` — fork remote

## CI

GitHub Actions runs on Ubuntu, macOS (x86 + ARM), Windows:
1. `cargo test --all` in both `dev` and `release` profiles
2. Vector tests: downloads IETF test vectors, runs `run_vectors2` in release mode

## Tool Preferences

- Always use `rg` (ripgrep) instead of `grep` for searching

## Dependencies

**Runtime**: num-traits, num-complex, bytemuck, arrayref, const-chunks, ndarray, nalgebra, itertools
**Dev**: getrandom, insta (snapshot testing)
**Features**: `ent-dump` — enables hex dumps of entropy coder calls for debugging divergence from C

## Unsafe Tracking Dashboard

Visual progress: https://unsafe-track.dcnick3.me/github/DCNick3/unsafe-libopus

## Roadmap Plans

Detailed plans live in `plans/`:

| Phase | File | Scope |
|-------|------|-------|
| Master | [plans/PLAN.md](plans/PLAN.md) | Overview, conventions, commit rules |
| 1 | [plans/phase1-test-expansion.md](plans/phase1-test-expansion.md) | Port all upstream C tests, rewrite unit tests |
| 2 | [plans/phase2-celt-safety.md](plans/phase2-celt-safety.md) | Make all CELT modules safe (bottom-up) |
| 3 | [plans/phase3-silk-safety.md](plans/phase3-silk-safety.md) | Make all SILK modules safe (bottom-up) |
| 4 | [plans/phase4-integration-safety.md](plans/phase4-integration-safety.md) | Safe encoder/decoder API, eliminate externs.rs |
| 5 | [plans/phase5-performance.md](plans/phase5-performance.md) | SIMD, benchmarks, match C performance |

Update plan checkboxes before each commit. Every function gets an
`/// Upstream C: <path>:<function>` cross-reference comment.

## Performance Notes

Current Rust version is ~20% slower than C with ASM/intrinsics. Future performance work:
- SIMD intrinsics (std::arch) for hot paths in CELT FFT, SILK pitch analysis
- Compiler hints and inlining for critical inner loops
- Profile-guided optimization
- Benchmark suite needed (not yet created)
