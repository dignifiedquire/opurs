# Development

## Build

```bash
cargo build
cargo build --release
cargo build --features tools    # includes CLI examples and C reference (libopus-sys)
```

## Testing

```bash
# Unit and integration tests
cargo nextest run -p opurs

# Release profile
cargo nextest run -p opurs --cargo-profile=release
```

### Bit-exact vector tests

These compare Rust encoder/decoder output against the C reference compiled from source in `libopus-sys/`. The suite runs the encoder at 9 bitrates and the decoder at 10 configurations across all IETF test vectors.

```bash
# Download IETF test vectors
curl https://www.ietf.org/proceedings/98/slides/materials-98-codec-opus-newvectors-00.tar.gz -o vectors.tar.gz
tar -xzf vectors.tar.gz

# Run standard vector tests
cargo run --release --features tools --example run_vectors2 -- opus_newvectors

# Full matrix (all bitrate/complexity/mode combinations)
cargo run --release --features tools --example run_vectors2 -- --matrix full opus_newvectors

# Multistream parity vectors
cargo run --release --features tools --example run_vectors2 -- opus_newvectors --suite multistream

# Projection parity vectors
cargo run --release --features tools --example run_vectors2 -- opus_newvectors --suite projection --matrix full --strict-bitexact

# QEXT parity vectors (requires qext feature + qext_vector*.bit assets)
cargo run --release --features "tools,qext" --example run_vectors2 -- opus_newvectors --suite qext --matrix full --strict-bitexact

# QEXT fuzz parity vectors (requires qext feature + qext_vector*fuzz.bit assets)
cargo run --release --features "tools,qext" --example run_vectors2 -- opus_newvectors --suite qext-fuzz --matrix full --strict-bitexact

# DRED-Opus parity vectors (requires tools-dnn + vector*_opus.bit assets)
cargo run --release --features tools-dnn --example run_vectors2 -- opus_newvectors --suite dred-opus --matrix full

# Dump directory for debugging mismatches
cargo run --release --features tools --example run_vectors2 -- opus_newvectors --dump-dir dump/

# Validate Upstream C reference anchors in Rust comments
./scripts/check_upstream_refs.sh
```

CI coverage notes:
- Major platforms (`linux-x86_64`, `macos-arm64`, `windows-x86_64`) run full-matrix suites for `classic`, `multistream`, `projection`, `qext`, `qext-fuzz`, and `dred-opus`; missing suite assets fail CI.
- Non-major platforms run quick classic vectors.
- libFuzzer jobs run `decode_fuzzer` and `extensions_fuzzer` for 60s on push/PR and 600s on daily scheduled CI.

### Entropy coder debugging

```bash
cargo test --features ent-dump
```

## Benchmarks

See [BENCHMARKS.md](BENCHMARKS.md) for detailed results, SIMD dispatch tables, and per-function Rust-vs-C comparisons.

```bash
cargo bench --bench pitch       # Pitch detection (scalar vs SIMD)
cargo bench --bench silk        # SILK inner functions
cargo bench --bench vq          # Vector quantization
cargo bench --bench codec       # End-to-end encode/decode
cargo bench --bench multistream # Multistream encode/decode
cargo bench --bench projection  # Projection encode/decode
cargo bench --bench dnn         # DNN inference (requires deep-plc feature)

# Requires --features tools (compares against C reference):
cargo bench --features tools --bench comparison
cargo bench --features tools --bench codec_comparison
```

## CLI Examples

All require `--features tools` (which pulls in `libopus-sys`, `clap`, `rayon`, `indicatif`).

### opus_demo

Encode and/or decode raw PCM audio.

```bash
# Roundtrip encode + decode
cargo run --release --features tools --example opus_demo -- \
  enc-dec audio 48000 2 64000 input.raw output.raw

# Encode only
cargo run --release --features tools --example opus_demo -- \
  enc audio 48000 2 128000 input.raw output.opus

# Decode only
cargo run --release --features tools --example opus_demo -- \
  dec 48000 2 input.opus output.raw

# Multistream roundtrip (3 channels, 2 streams, 1 coupled)
cargo run --release --features tools --example opus_demo -- \
  ms-enc-dec audio 48000 3 2 1 96000 input.raw output.raw
```

### opus_compare

Compare two raw audio files using a PEAQ-style quality metric.

```bash
cargo run --release --features tools --example opus_compare -- reference.raw degraded.raw
```

### repacketizer_demo

Merge or split Opus packets.

```bash
cargo run --release --features tools --example repacketizer_demo -- input.opus output.opus
```

### multistream_demo

Minimal in-memory multistream encode/decode example.

```bash
cargo run --release --features tools --example multistream_demo
```

## Correctness

**Every change must maintain bit-exact output with the C reference.**

- **Decoder**: decoded samples must be identical to C at all 5 sample rates (8/12/16/24/48 kHz), mono and stereo
- **Encoder**: encoded bitstreams must be byte-for-byte identical to C at all 9 test bitrates (10k--240k bps)
- Run `cargo nextest run -p opurs` after any change; run vector tests for changes touching codec internals

## Commit Conventions

Prefixes: `test:`, `refactor:`, `fix:`, `perf:`, `chore:`, `docs:`

Pre-commit checks:

```bash
cargo build
cargo nextest run -p opurs
cargo clippy --all --all-targets --features tools -- -D warnings
cargo fmt --check
```
