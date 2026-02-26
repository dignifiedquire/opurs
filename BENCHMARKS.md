# Benchmarks

Last updated: February 26, 2026

## End-to-End Codec: Rust vs C (both with SIMD)

### aarch64 (Apple Silicon M4, NEON)

| Operation | Configuration | Rust | C (NEON) | Ratio |
|-----------|--------------|------|----------|-------|
| Encode | 64 kbps stereo | 3.68 ms | 3.54 ms | 1.04x |
| Encode | 128 kbps stereo | 5.03 ms | 4.66 ms | 1.08x |
| Decode | 64 kbps stereo | 956 us | 938 us | 1.02x |
| Decode | 128 kbps stereo | 1.14 ms | 1.12 ms | 1.02x |
| Encode | 16 kbps mono VOIP | 5.23 ms | 7.06 ms | **0.74x (Rust faster)** |
| Encode | 64 kbps mono VOIP | 2.78 ms | 2.57 ms | 1.08x |

Rust is **within 2-8% of C** on most paths. SILK-heavy 16 kbps VOIP encoding is
**26% faster than C** due to ndarray removal and inline optimizations in the SILK path.

### x86_64 Linux (AVX2)

| Operation | Configuration | Rust | C (SIMD) | Ratio |
|-----------|--------------|------|----------|-------|
| Encode | 16 kbps mono VOIP | 8.36 ms | 7.75 ms | 1.08x |
| Encode | 64 kbps mono VOIP | 3.39 ms | 2.95 ms | 1.15x |
| Encode | 64 kbps stereo | 4.95 ms | 4.32 ms | 1.15x |
| Encode | 128 kbps stereo | 6.82 ms | 5.42 ms | 1.26x |
| Decode | 64 kbps stereo | 1.29 ms | 1.10 ms | 1.18x |
| Decode | 128 kbps stereo | 1.70 ms | 1.43 ms | 1.19x |

All measurements at 48 kHz, 20 ms frames, complexity 10, 1 second of audio (50 frames).

### Progress vs initial baseline

| Operation | Pre-SIMD | Post-SIMD | Current (aarch64) | Current (x86) |
|-----------|----------|-----------|-------------------|----------------|
| Encode 16k mono VOIP | 1.29x | 1.10x | **0.74x** | 1.08x |
| Encode 64k mono VOIP | 1.30x | 1.12x | **1.08x** | 1.15x |
| Decode 64k stereo | 1.50x | 1.43x | **1.02x** | 1.18x |
| Decode 128k stereo | 1.43x | 1.41x | **1.02x** | 1.19x |

## End-to-End Codec: Rust-only (x86_64)

Rust encode/decode throughput without C comparison. 48 kHz, 20 ms frames, complexity 10,
50 frames (1 second).

| Operation | Configuration | Time |
|-----------|--------------|------|
| Encode | 32 kbps stereo | 4.21 ms |
| Encode | 64 kbps stereo | 5.22 ms |
| Encode | 128 kbps stereo | 6.86 ms |
| Encode | 16 kbps mono VOIP | 8.55 ms |
| Encode | 32 kbps mono VOIP | 2.90 ms |
| Encode | 64 kbps mono VOIP | 3.40 ms |
| Decode | 32 kbps stereo | 1.39 ms |
| Decode | 64 kbps stereo | 1.27 ms |
| Decode | 128 kbps stereo | 1.69 ms |

## End-to-End Multistream: Rust vs C (both with SIMD)

Configuration:
- Sample rate: 48 kHz
- Layout matrix:
  - `1ch` => streams=1, coupled=0, mapping `[0]`
  - `2ch` => streams=2, coupled=0, mapping `[0,1]`
  - `6ch` => streams=4, coupled=2, mapping `[0,4,1,2,3,5]` (5.1-style)
- Frame sizes: 10 ms and 20 ms
- Bitrates: 32 kbps, 96 kbps, 192 kbps
- Complexity: 10
- Input: deterministic synthetic signal, 50 frames

### aarch64 (Apple Silicon M4, NEON)

Representative 96 kbps points (median):

| Operation | Scenario | Rust | C (NEON) | Ratio |
|-----------|----------|------|----------|-------|
| Encode | 1ch 10ms | 1.196 ms | 1.000 ms | 1.20x |
| Encode | 1ch 20ms | 2.233 ms | 2.046 ms | 1.09x |
| Encode | 2ch 10ms | 1.841 ms | 1.695 ms | 1.09x |
| Encode | 2ch 20ms | 3.669 ms | 3.508 ms | 1.05x |
| Encode | 6ch 10ms | 9.999 ms | 15.016 ms | **0.67x (Rust faster)** |
| Encode | 6ch 20ms | 14.402 ms | 17.704 ms | **0.81x (Rust faster)** |
| Decode | 1ch 10ms | 0.429 ms | 0.309 ms | 1.39x |
| Decode | 1ch 20ms | 0.768 ms | 0.627 ms | 1.22x |
| Decode | 2ch 10ms | 0.763 ms | 0.518 ms | 1.47x |
| Decode | 2ch 20ms | 1.329 ms | 1.058 ms | 1.26x |
| Decode | 6ch 10ms | 1.989 ms | 1.488 ms | 1.34x |
| Decode | 6ch 20ms | 2.828 ms | 2.307 ms | 1.23x |

Selected low-bitrate (32 kbps) points:

| Operation | Scenario | Rust | C (NEON) | Ratio |
|-----------|----------|------|----------|-------|
| Encode | 2ch 10ms | 4.780 ms | 7.214 ms | **0.66x (Rust faster)** |
| Encode | 2ch 20ms | 5.220 ms | 6.646 ms | **0.79x (Rust faster)** |
| Encode | 6ch 10ms | 5.091 ms | 6.900 ms | **0.74x (Rust faster)** |
| Encode | 6ch 20ms | 9.796 ms | 18.623 ms | **0.53x (Rust faster)** |
| Decode | 2ch 20ms | 0.947 ms | 0.742 ms | 1.28x |
| Decode | 6ch 20ms | 1.178 ms | 1.217 ms | ~matched |

### x86_64 Linux (AVX2)

Representative 96 kbps points (median):

| Operation | Scenario | Rust | C (SIMD) | Ratio |
|-----------|----------|------|----------|-------|
| Encode | 1ch 10ms | 1.70 ms | 1.36 ms | 1.25x |
| Encode | 1ch 20ms | 3.18 ms | 2.67 ms | 1.19x |
| Encode | 2ch 10ms | 2.88 ms | 2.39 ms | 1.21x |
| Encode | 2ch 20ms | 5.26 ms | 4.57 ms | 1.15x |
| Encode | 6ch 10ms | 21.02 ms | 18.49 ms | 1.14x |
| Encode | 6ch 20ms | 24.14 ms | 21.43 ms | 1.13x |
| Decode | 1ch 10ms | 0.606 ms | 0.462 ms | 1.31x |
| Decode | 1ch 20ms | 1.21 ms | 0.957 ms | 1.27x |
| Decode | 2ch 10ms | 0.937 ms | 0.727 ms | 1.29x |
| Decode | 2ch 20ms | 1.78 ms | 1.39 ms | 1.28x |
| Decode | 6ch 10ms | 2.86 ms | 2.56 ms | 1.12x |
| Decode | 6ch 20ms | 4.26 ms | 3.68 ms | 1.16x |

Selected low-bitrate (32 kbps) points:

| Operation | Scenario | Rust | C (SIMD) | Ratio |
|-----------|----------|------|----------|-------|
| Encode | 1ch 20ms | 2.36 ms | 1.98 ms | 1.19x |
| Encode | 2ch 10ms | 8.27 ms | 7.68 ms | 1.08x |
| Encode | 2ch 20ms | 8.71 ms | 7.94 ms | 1.10x |
| Encode | 6ch 10ms | 9.61 ms | 9.04 ms | 1.06x |
| Encode | 6ch 20ms | 23.05 ms | 22.02 ms | 1.05x |
| Decode | 2ch 20ms | 1.38 ms | 1.15 ms | 1.20x |
| Decode | 6ch 20ms | 2.67 ms | 2.50 ms | 1.07x |

Raw Criterion outputs are in `target/criterion/multistream_*`.

## Projection Benchmark Coverage

Projection benchmark suite (`benches/projection.rs`) now covers:
- End-to-end encode throughput (`projection_encode_cmp`) for Rust vs upstream C
- End-to-end decode throughput (`projection_decode_cmp`) for Rust vs upstream C
- Matrix apply hot path (`projection_matrix_apply`) on Rust demixing path

Run:

```bash
CARGO_TARGET_DIR=target-local RUSTFLAGS="-C target-cpu=native" \
  cargo bench --features tools --bench projection
```

## Per-Function: Rust vs C (with SIMD)

### celt_pitch_xcorr — aarch64 (NEON)

| N | Rust Scalar | C Scalar | Rust NEON | C NEON | Rust vs C NEON |
|---|-------------|----------|-----------|--------|----------------|
| 240x60 | 1.85 us | 2.25 us | 1.77 us | 1.77 us | **matched** |
| 480x120 | 8.47 us | 9.77 us | 8.94 us | 8.88 us | **matched** |
| 960x240 | 35.9 us | 41.3 us | 44.6 us | 46.2 us | **1.04x faster** |

Rust NEON **matches** C NEON for pitch xcorr. Rust scalar is consistently faster
than C scalar (auto-vectorization differences).

### celt_pitch_xcorr — x86_64 (AVX2)

| N | Rust Scalar | C Scalar | Rust SIMD | C SIMD | Rust vs C SIMD |
|---|-------------|----------|-----------|--------|----------------|
| 240x60 | 2.32 us | 3.20 us | 444 ns | 430 ns | ~matched |
| 480x120 | 9.38 us | 12.85 us | 1.29 us | 1.30 us | ~matched |
| 960x240 | 37.93 us | 50.71 us | 5.14 us | 5.36 us | **1.04x faster** |

Rust SIMD (AVX2) **matches or slightly beats** C SIMD (AVX2+FMA) for pitch xcorr.
Rust scalar is **25-35% faster** than C scalar due to auto-vectorization differences.

### silk_inner_product_FLP — aarch64 (NEON)

| N | Rust Scalar | C Scalar | Rust NEON | Rust vs C Scalar |
|---|-------------|----------|-----------|------------------|
| 64 | 15 ns | 16 ns | 7 ns | 2.3x faster |
| 240 | 85 ns | 61 ns | 39 ns | 1.6x faster |
| 480 | 211 ns | 122 ns | 84 ns | 1.5x faster |
| 960 | 482 ns | 243 ns | 203 ns | 1.2x faster |

C has no NEON variant for silk_inner_product_FLP on aarch64 (only x86 AVX2).
Rust NEON dispatch is **1.2-2.3x faster** than C scalar on this function.

### silk_inner_product_FLP — x86_64 (AVX2)

| N | Rust Scalar | C Scalar | Rust SIMD | C SIMD | Rust vs C SIMD |
|---|-------------|----------|-----------|--------|----------------|
| 64 | 38 ns | 25 ns | 6 ns | 7 ns | **1.1x faster** |
| 240 | 153 ns | 84 ns | 23 ns | 24 ns | ~matched |
| 480 | 313 ns | 167 ns | 49 ns | 49 ns | ~matched |
| 960 | 627 ns | 329 ns | 101 ns | 103 ns | ~matched |

Rust SIMD (AVX2+FMA) **matches** C SIMD (AVX2+FMA) for SILK inner product.

## CELT Pitch Functions — Rust scalar vs NEON dispatch (aarch64)

| Function | N | Scalar | NEON Dispatch | Speedup |
|----------|---|--------|---------------|---------|
| celt_pitch_xcorr | 240x60 | 1.85 us | 1.77 us | 1.0x |
| celt_pitch_xcorr | 480x120 | 8.47 us | 8.94 us | ~1.0x |
| celt_pitch_xcorr | 960x240 | 35.9 us | 44.6 us | 0.8x |

On aarch64, NEON dispatch for pitch_xcorr does not provide significant speedup
over scalar — LLVM auto-vectorizes the scalar path effectively with NEON.

## CELT Pitch Functions — Rust scalar vs SIMD dispatch (x86_64)

| Function | N | Scalar | SIMD Dispatch | Speedup |
|----------|---|--------|---------------|---------|
| xcorr_kernel | 64 | 44 ns | 25 ns | 1.8x |
| xcorr_kernel | 240 | 160 ns | 83 ns | 1.9x |
| xcorr_kernel | 480 | 318 ns | 162 ns | 2.0x |
| xcorr_kernel | 960 | 640 ns | 315 ns | 2.0x |
| celt_inner_prod | 64 | 28 ns | 8 ns | 3.3x |
| celt_inner_prod | 240 | 138 ns | 31 ns | 4.4x |
| celt_inner_prod | 480 | 293 ns | 71 ns | 4.1x |
| celt_inner_prod | 960 | 598 ns | 148 ns | 4.0x |
| dual_inner_prod | 64 | 38 ns | 11 ns | 3.6x |
| dual_inner_prod | 240 | 150 ns | 39 ns | 3.8x |
| dual_inner_prod | 480 | 314 ns | 79 ns | 4.0x |
| dual_inner_prod | 960 | 625 ns | 159 ns | 3.9x |
| celt_pitch_xcorr | 240x60 | 2.35 us | 445 ns | 5.3x |
| celt_pitch_xcorr | 480x120 | 9.43 us | 1.30 us | 7.3x |
| celt_pitch_xcorr | 960x240 | 38.5 us | 5.26 us | 7.3x |

## SILK Functions — Rust scalar vs NEON dispatch (aarch64)

| Function | N | Scalar | NEON Dispatch | Speedup |
|----------|---|--------|---------------|---------|
| silk_inner_product_FLP | 64 | 15 ns | 7 ns | 2.1x |
| silk_inner_product_FLP | 240 | 85 ns | 39 ns | 2.2x |
| silk_inner_product_FLP | 480 | 211 ns | 84 ns | 2.5x |
| silk_inner_product_FLP | 960 | 482 ns | 203 ns | 2.4x |

## SILK Functions — Rust scalar vs SIMD dispatch (x86_64)

| Function | N | Scalar | SIMD Dispatch | Speedup |
|----------|---|--------|---------------|---------|
| silk_short_prediction | order=10 | 6 ns | 3 ns | 1.8x |
| silk_short_prediction | order=16 | 9 ns | 3 ns | 2.7x |
| silk_inner_product_FLP | 64 | 38 ns | 6 ns | 5.8x |
| silk_inner_product_FLP | 240 | 154 ns | 23 ns | 6.7x |
| silk_inner_product_FLP | 480 | 308 ns | 49 ns | 6.3x |
| silk_inner_product_FLP | 960 | 621 ns | 101 ns | 6.2x |

## DNN Features: Rust vs C (x86_64)

All DNN benchmarks measure 50 frames (1 second) at 48 kHz, mono, with builtin weights
loaded. Both Rust and C use equivalent compiled-in weight data (C auto-loads at init,
Rust loads via `load_dnn_weights()` before the timed region).

### DRED Encoding

Vary bitrate (16k/32k/64k) and DRED duration (8/24/48 frames).
Fixed: VOIP, complexity 10, 10% packet loss.

| Bitrate | DRED Dur | Rust | C | Ratio |
|---------|----------|------|---|-------|
| 16 kbps | 8 | 10.10 ms | 8.38 ms | 1.21x |
| 16 kbps | 24 | 10.29 ms | 8.35 ms | 1.23x |
| 16 kbps | 48 | 10.50 ms | 8.41 ms | 1.25x |
| 32 kbps | 8 | 10.49 ms | 8.61 ms | 1.22x |
| 32 kbps | 24 | 10.54 ms | 8.56 ms | 1.23x |
| 32 kbps | 48 | 10.62 ms | 8.57 ms | 1.24x |
| 64 kbps | 8 | 11.06 ms | 8.78 ms | 1.26x |
| 64 kbps | 24 | 11.00 ms | 9.00 ms | 1.22x |
| 64 kbps | 48 | 10.77 ms | 8.77 ms | 1.23x |

Rust DRED encoding is consistently **21-26% slower** than C across all configurations.
DRED duration has minimal impact on total encode time — the RDOVAE work is roughly constant.
Higher bitrate adds ~0.5 ms due to CELT hybrid mode engagement.

### OSCE Decoding (BBWENet)

Vary bitrate (12k/16k/24k) and complexity (6=LACE, 7=NoLACE, 10=NoLACE).
Fixed: VOIP, OSCE BWE enabled.

| Bitrate | Complexity | OSCE Mode | Rust | C | Ratio |
|---------|-----------|-----------|------|---|-------|
| 12 kbps | 6 (LACE) | LACE | 11.19 ms | 8.68 ms | 1.29x |
| 12 kbps | 7 (NoLACE) | NoLACE | 18.95 ms | 11.08 ms | 1.71x |
| 12 kbps | 10 (NoLACE) | NoLACE | 18.66 ms | 11.11 ms | 1.68x |
| 16 kbps | 6 (LACE) | LACE | 4.78 ms | 2.28 ms | 2.09x |
| 16 kbps | 7 (NoLACE) | NoLACE | 12.18 ms | 4.81 ms | 2.53x |
| 16 kbps | 10 (NoLACE) | NoLACE | 12.08 ms | 4.84 ms | 2.49x |
| 24 kbps | 6 (LACE) | LACE | 4.87 ms | 2.37 ms | 2.06x |
| 24 kbps | 7 (NoLACE) | NoLACE | 12.21 ms | 4.95 ms | 2.47x |
| 24 kbps | 10 (NoLACE) | NoLACE | 12.13 ms | 4.93 ms | 2.46x |

Key observations:
- **LACE** (complexity 6) is ~2x faster than **NoLACE** (complexity 7+) on both sides
- 12 kbps is significantly slower because more frames use SILK-only mode (OSCE active on more frames)
- At 16k/24k, LACE Rust-vs-C gap is **~2x**, NoLACE gap is **~2.5x**
- Complexity 7 vs 10 shows no meaningful difference (both use NoLACE)
- The OSCE neural network inference is the main optimization target

### Deep PLC (Packet Loss Concealment)

Vary complexity (5/7/10) and loss rate (10%/20%/50%).
Fixed: 16 kbps VOIP.

| Complexity | Loss % | Rust | C | Ratio |
|-----------|--------|------|---|-------|
| 5 | 10% | 1.56 ms | 3.82 ms | **0.41x (Rust 2.5x faster)** |
| 5 | 20% | 1.56 ms | 5.02 ms | **0.31x (Rust 3.2x faster)** |
| 5 | 50% | 1.40 ms | 7.40 ms | **0.19x (Rust 5.3x faster)** |
| 7 | 10% | 10.84 ms | 6.78 ms | 1.60x |
| 7 | 20% | 9.88 ms | 7.77 ms | 1.27x |
| 7 | 50% | 6.71 ms | 8.66 ms | **0.78x (Rust faster)** |
| 10 | 10% | 11.00 ms | 6.78 ms | 1.62x |
| 10 | 20% | 9.86 ms | 7.71 ms | 1.28x |
| 10 | 50% | 6.68 ms | 8.63 ms | **0.77x (Rust faster)** |

Key observations:
- **Complexity 5** (PLC-only, no OSCE): Rust is **2.5-5.3x faster** than C across all
  loss rates. The Rust DNN inference for the PLC model compiles very efficiently.
- **Complexity 7+** (OSCE+PLC): Rust is slower at low loss rates (OSCE overhead dominates)
  but catches up and **beats C at 50% loss** where PLC frames dominate.
- C's PLC cost scales linearly with loss rate; Rust's PLC at complexity 5 is nearly
  constant regardless of loss rate (~1.5 ms), suggesting efficient early-exit or
  better branch prediction in the Rust PLC path.
- Complexity 7 vs 10: no meaningful difference (both enable OSCE NoLACE).

## SIMD Implementations

### x86/x86_64
- **SSE**: xcorr_kernel, celt_inner_prod, dual_inner_prod, celt_pitch_xcorr, comb_filter_const
- **SSE2**: silk_inner_product_FLP (fallback), silk_vad_energy, op_pvq_search
- **SSE4.1**: silk_noise_shape_quantizer_short_prediction, silk_NSQ (full quantizer), silk_NSQ_del_dec (quantizer + scale_states), silk_VQ_WMat_EC
- **AVX2**: xcorr_kernel (8-wide), celt_pitch_xcorr (8 correlations/iter), silk_NSQ_del_dec (full quantizer, 4-state parallel)
- **AVX2+FMA**: silk_inner_product_FLP (4-wide f64 with fused multiply-add)

### aarch64
- **NEON**: xcorr_kernel, celt_inner_prod, dual_inner_prod, celt_pitch_xcorr, silk_noise_shape_quantizer_short_prediction, silk_inner_product_FLP, silk_NSQ_noise_shape_feedback_loop, silk_LPC_inverse_pred_gain, silk_NSQ_del_dec (full quantizer, 4-state parallel)

### Dispatch Hierarchy (x86/x86_64)

```
celt_pitch_xcorr:  AVX2 > SSE > scalar
xcorr_kernel:      SSE > scalar
celt_inner_prod:   SSE > scalar
dual_inner_prod:   SSE > scalar
comb_filter_const: SSE > scalar
silk_inner_FLP:    AVX2+FMA > SSE2 > scalar
silk_short_pred:   SSE4.1 > scalar
silk_NSQ:          SSE4.1 > scalar
silk_NSQ_del_dec:  AVX2 > SSE4.1 > scalar
silk_VQ_WMat_EC:   SSE4.1 > scalar
silk_vad_energy:   SSE2 > scalar
op_pvq_search:     SSE2 > scalar
```

### Dispatch Hierarchy (aarch64)

```
celt_pitch_xcorr:          NEON > scalar
xcorr_kernel:              NEON > scalar
celt_inner_prod:           NEON > scalar
dual_inner_prod:           NEON > scalar
silk_inner_FLP:            NEON > scalar
silk_short_pred:           NEON > scalar
silk_feedback_loop:        NEON > scalar
silk_LPC_inv_pred_gain:    NEON > scalar
silk_NSQ_del_dec:          NEON > scalar
```

## Running Benchmarks

```bash
# Recommended: use target-cpu=native for best results
export RUSTFLAGS="-C target-cpu=native"

# Individual function benchmarks (Rust only, scalar vs SIMD dispatch)
cargo bench --bench pitch
cargo bench --bench silk
cargo bench --bench vq
cargo bench --bench dnn

# End-to-end codec benchmarks (Rust only)
cargo bench --bench codec

# DNN codec benchmarks — DRED, OSCE, Deep PLC (Rust only)
cargo bench --features "dnn,builtin-weights" --bench dnn_codec

# Per-function comparison: Rust vs C (scalar + SIMD)
cargo bench --features tools --bench comparison

# End-to-end codec comparison: Rust vs C (with SIMD)
cargo bench --features tools --bench codec_comparison

# DNN comparison: Rust vs C for DRED/OSCE/PLC
cargo bench --features tools-dnn --bench dnn_comparison

# Multistream comparison matrix: Rust vs C
cargo bench --features tools --bench multistream

# Projection comparison matrix + matrix apply
cargo bench --features tools --bench projection

# Summarize Criterion multistream outputs to Markdown
./scripts/criterion_multistream_summary.sh target/criterion target/criterion/multistream-summary.md

# Summarize Criterion projection outputs to Markdown
./scripts/criterion_projection_summary.sh target/criterion target/criterion/projection-summary.md
```

## Notes

- SIMD paths produce valid but potentially different floating-point results from scalar
  (due to different accumulation order). This is expected and matches libopus C behavior.
- Bit-exact comparison with C reference is verified using scalar mode only
  (`--no-default-features --features tools`).
- aarch64 NEON is compile-time dispatched (always available on AArch64).
- x86 features detected at runtime via `cpufeatures` crate (cached after first check).
- `--no-default-features` disables all SIMD, falling through to scalar.
- C SIMD is automatically compiled when both `tools` and `simd` features are active.
  The `simd` feature forwards to `libopus-sys/simd` via optional dependency forwarding.
- Bench profile uses `lto = "thin"` and `codegen-units = 1` for consistent, optimized results.
- C reference on aarch64 has no NEON variant for `silk_inner_product_FLP` (only x86 has AVX2);
  Rust provides NEON dispatch for this function, giving a consistent advantage on ARM.
