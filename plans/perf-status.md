# Performance Status: Rust vs C

## Latest Measurement Run (2026-02-26)

Environment:
- macOS arm64 (Apple M4)
- `rustc 1.93.1`
- `CARGO_TARGET_DIR=target-local`
- `RUSTFLAGS="-C target-cpu=native"`

Benchmarks were rerun sequentially with:
- `cargo bench --features tools --bench codec_comparison`
- `cargo bench --features tools --bench codec`
- `cargo bench --features tools --bench multistream`
- `cargo bench --features tools --bench comparison`
- `cargo bench --bench pitch`
- `cargo bench --bench silk`
- `cargo bench --bench vq`

Raw logs:
- `target-local/bench-logs/2026-02-26-arm64/`

## End-to-End Codec (48 kHz, 20 ms, complexity 10, 50 frames)

From `codec_comparison`:

| Benchmark | Rust | C | Ratio |
|-----------|------|---|-------|
| Encode 64 kbps stereo | 4.09 ms | 3.77 ms | 1.09x |
| Encode 128 kbps stereo | 5.42 ms | 4.92 ms | 1.10x |
| Decode 64 kbps stereo | 986 us | 893 us | 1.10x |
| Decode 128 kbps stereo | 1.22 ms | 1.08 ms | 1.13x |
| Encode 16 kbps mono VOIP | 5.70 ms | 6.29 ms | **0.91x (Rust faster)** |
| Encode 64 kbps mono VOIP | 3.02 ms | 2.82 ms | 1.07x |

Status:
- Stereo encode/decode is close but still behind C (~7-13%).
- SILK-heavy 16 kbps mono VOIP still beats C.

## Multistream Snapshot

Representative 96 kbps points (`multistream`):

| Benchmark | Rust | C | Ratio |
|-----------|------|---|-------|
| Encode 1ch 10ms | 1.361 ms | 1.114 ms | 1.22x |
| Encode 2ch 20ms | 4.382 ms | 3.953 ms | 1.11x |
| Encode 6ch 10ms | 13.988 ms | 14.570 ms | **0.96x (Rust faster)** |
| Decode 1ch 10ms | 0.469 ms | 0.355 ms | 1.32x |
| Decode 2ch 20ms | 1.562 ms | 1.166 ms | 1.34x |
| Decode 6ch 10ms | 1.820 ms | 1.643 ms | 1.11x |

Low-bitrate (32 kbps) selected:

| Benchmark | Rust | C | Ratio |
|-----------|------|---|-------|
| Encode 2ch 10ms | 5.428 ms | 5.708 ms | **0.95x (Rust faster)** |
| Encode 6ch 20ms | 16.029 ms | 17.387 ms | **0.92x (Rust faster)** |
| Decode 2ch 20ms | 1.015 ms | 0.811 ms | 1.25x |
| Decode 6ch 20ms | 1.392 ms | 1.289 ms | 1.08x |

Status:
- Encode has several 6ch and low-bitrate wins.
- Decode remains the main gap in multistream (especially 1ch/2ch).

## Function-Level Findings

From `comparison`, `pitch`, and `silk`:

1. `celt_pitch_xcorr` on aarch64:
- Rust scalar beats C scalar.
- Rust dispatch (NEON) is near C NEON but slower than Rust scalar in our implementation.
- Scalar vs dispatch (Rust): `0.92x`, `0.89x`, `0.76x` speedups (i.e., dispatch slower).

2. `silk_inner_product_FLP` on aarch64:
- Rust dispatch and Rust scalar are effectively identical.
- C scalar is faster for larger sizes in this run.

3. `celt_inner_prod` and `dual_inner_prod`:
- Dispatch path shows strong wins vs Rust scalar on aarch64.
- These are not current bottlenecks.

## What To Improve Next (to Match/Beat C)

Priority 1: fix aarch64 `celt_pitch_xcorr` dispatch performance.
- Why: dispatch is slower than Rust scalar and this path is hot in analysis/encoder.
- Actions:
  - Compare Rust NEON asm vs C NEON asm for `celt_pitch_xcorr`.
  - Audit lane reduction/unrolling and memory access pattern in `src/celt/simd/aarch64`.
  - If needed, add an aarch64 dispatch heuristic to prefer scalar for sizes where scalar is faster.

Priority 2: target decode-side gaps (multistream + stereo decode).
- Why: decode is still ~1.1-1.35x slower in key cases.
- Actions:
  - Profile `multistream_decode_cmp/rust/*` with flamegraph/perf sampling.
  - Focus on `celt_decode_with_ec`, packet parse/repacketizer paths, and hot entropy decode loops.
  - Apply the same cold-path extraction/inlining strategy that improved encode.

Priority 3: stabilize benchmark signal and CI performance tracking.
- Why: some multistream points are noisy without focused reruns.
- Actions:
  - Add a perf script that reruns selected critical slices (e.g., 6ch/10ms/96k decode) and reports medians.
  - Gate on relative thresholds for the most important Rust-vs-C ratios.

## Short-Term Target

For this arm64 machine:
- Bring stereo decode from ~1.10-1.13x down to <=1.05x.
- Bring multistream decode from ~1.25-1.35x down to <=1.15x on 1ch/2ch.
- Keep or improve current encode wins at 6ch and low bitrate.
