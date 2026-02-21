# Benchmarks

Last updated: multistream Rust-vs-C benchmark results (February 20, 2026)

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
| Encode | 16 kbps mono VOIP | 8.37 ms | 7.56 ms | 1.11x |
| Encode | 64 kbps mono VOIP | 2.97 ms | 2.67 ms | 1.11x |
| Decode | 64 kbps stereo | 1.51 ms | 1.09 ms | 1.39x |
| Decode | 128 kbps stereo | 1.90 ms | 1.42 ms | 1.34x |

All measurements at 48 kHz, 20 ms frames, complexity 10, 1 second of audio (50 frames).

### Progress vs initial baseline

| Operation | Pre-SIMD | Post-SIMD | Current (aarch64) | Current (x86) |
|-----------|----------|-----------|-------------------|----------------|
| Encode 16k mono VOIP | 1.29x | 1.10x | **0.74x** | 1.11x |
| Encode 64k mono VOIP | 1.30x | 1.12x | **1.08x** | 1.11x |
| Decode 64k stereo | 1.50x | 1.43x | **1.02x** | 1.39x |
| Decode 128k stereo | 1.43x | 1.41x | **1.02x** | 1.34x |

Note: x86 numbers are from a previous run without the bench profile optimizations
(LTO, codegen-units=1). Re-running on x86 with the new profile is expected to improve
those ratios.

## End-to-End Multistream: Rust vs C (both with SIMD)

### aarch64 (Apple Silicon M4, NEON)

Command:

```bash
CARGO_TARGET_DIR=target-local RUSTFLAGS="-C target-cpu=native" \
  cargo bench --features tools --bench multistream
```

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

Raw Criterion outputs for the full matrix are in `target-local/criterion/multistream_*`.

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
| 240x60 | 2.37 us | 3.19 us | 408 ns | 438 ns | **1.07x faster** |
| 480x120 | 9.33 us | 12.6 us | 1.35 us | 1.34 us | ~matched |
| 960x240 | 37.8 us | 50.8 us | 5.20 us | 5.27 us | ~matched |

Rust SIMD (AVX2) **matches or slightly beats** C SIMD (AVX2+FMA) for pitch xcorr.

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
| 64 | 38 ns | 25 ns | 6 ns | 7 ns | **1.2x faster** |
| 240 | 153 ns | 91 ns | 23 ns | 24 ns | ~matched |
| 480 | 312 ns | 176 ns | 50 ns | 49 ns | ~matched |
| 960 | 625 ns | 357 ns | 102 ns | 103 ns | ~matched |

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
| xcorr_kernel | 64 | 41 ns | 25 ns | 1.6x |
| xcorr_kernel | 240 | 156 ns | 83 ns | 1.9x |
| xcorr_kernel | 480 | 313 ns | 162 ns | 1.9x |
| xcorr_kernel | 960 | 627 ns | 320 ns | 2.0x |
| celt_inner_prod | 64 | 27 ns | 8 ns | 3.4x |
| celt_inner_prod | 240 | 138 ns | 32 ns | 4.3x |
| celt_inner_prod | 480 | 295 ns | 71 ns | 4.2x |
| celt_inner_prod | 960 | 613 ns | 153 ns | 4.0x |
| dual_inner_prod | 64 | 39 ns | 12 ns | 3.3x |
| dual_inner_prod | 240 | 153 ns | 40 ns | 3.8x |
| dual_inner_prod | 480 | 310 ns | 79 ns | 3.9x |
| dual_inner_prod | 960 | 632 ns | 159 ns | 4.0x |
| celt_pitch_xcorr | 240x60 | 2.37 us | 401 ns | 5.9x |
| celt_pitch_xcorr | 480x120 | 9.25 us | 1.31 us | 7.1x |
| celt_pitch_xcorr | 960x240 | 37.9 us | 5.25 us | 7.2x |

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
| silk_short_prediction | order=10 | 6 ns | 5 ns | 1.2x |
| silk_short_prediction | order=16 | 9 ns | 6 ns | 1.5x |
| silk_inner_product_FLP | 64 | 38 ns | 18 ns | 2.1x |
| silk_inner_product_FLP | 240 | 152 ns | 74 ns | 2.1x |
| silk_inner_product_FLP | 480 | 315 ns | 154 ns | 2.0x |
| silk_inner_product_FLP | 960 | 624 ns | 309 ns | 2.0x |

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

# Individual function benchmarks (Rust only)
cargo bench --bench pitch
cargo bench --bench silk
cargo bench --bench vq

# End-to-end codec benchmarks (Rust only)
cargo bench --bench codec

# Per-function comparison: Rust vs C (scalar + SIMD)
cargo bench --features tools --bench comparison

# End-to-end codec comparison: Rust vs C (with SIMD)
cargo bench --features tools --bench codec_comparison

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
