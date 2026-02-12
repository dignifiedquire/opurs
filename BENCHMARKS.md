# Benchmarks

Last updated: inline hints, stack allocations, FFT loop optimization

Platform: x86_64 Linux (AVX2 available)

## End-to-End Codec: Rust vs C (both with SIMD)

| Operation | Configuration | Rust | C (SIMD) | Ratio |
|-----------|--------------|------|----------|-------|
| Encode | 16 kbps mono VOIP | 8.37 ms | 7.56 ms | 1.11x |
| Encode | 64 kbps mono VOIP | 2.97 ms | 2.67 ms | 1.11x |
| Decode | 64 kbps stereo | 1.51 ms | 1.09 ms | 1.39x |
| Decode | 128 kbps stereo | 1.90 ms | 1.42 ms | 1.34x |

All measurements at 48 kHz, 20 ms frames, complexity 10, 1 second of audio (50 frames).

Rust is **~1.1-1.4x slower** than C with SIMD end-to-end. SILK-heavy VOIP encoding
nearly matches C (1.11x); CELT-heavy decode paths have more headroom.

### Progress vs initial baseline (pre-SIMD)

| Operation | Pre-SIMD | Post-SIMD | Current | Improvement |
|-----------|----------|-----------|---------|-------------|
| Encode 16k mono VOIP | 1.29x | 1.10x | **1.11x** | -0.18 |
| Encode 64k mono VOIP | 1.30x | 1.12x | **1.11x** | -0.19 |
| Decode 64k stereo | 1.50x | 1.43x | **1.39x** | -0.11 |
| Decode 128k stereo | 1.43x | 1.41x | **1.34x** | -0.09 |

## CELT Pitch Functions (scalar vs SIMD dispatch)

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

## SILK Functions (scalar vs SIMD dispatch)

| Function | N | Scalar | SIMD Dispatch | Speedup |
|----------|---|--------|---------------|---------|
| silk_short_prediction | order=10 | 6 ns | 5 ns | 1.2x |
| silk_short_prediction | order=16 | 9 ns | 6 ns | 1.5x |
| silk_inner_product_FLP | 64 | 38 ns | 18 ns | 2.1x |
| silk_inner_product_FLP | 240 | 152 ns | 74 ns | 2.1x |
| silk_inner_product_FLP | 480 | 315 ns | 154 ns | 2.0x |
| silk_inner_product_FLP | 960 | 624 ns | 309 ns | 2.0x |

## Rust vs C Reference — Per-Function (with SIMD)

### celt_pitch_xcorr

| N | Rust Scalar | C Scalar | Rust SIMD | C SIMD | Rust vs C SIMD |
|---|-------------|----------|-----------|--------|----------------|
| 240x60 | 2.37 us | 3.19 us | 408 ns | 438 ns | **1.07x faster** |
| 480x120 | 9.33 us | 12.6 us | 1.35 us | 1.34 us | ~matched |
| 960x240 | 37.8 us | 50.8 us | 5.20 us | 5.27 us | ~matched |

Rust SIMD (AVX2) **matches or slightly beats** C SIMD (AVX2+FMA) for pitch xcorr.

### silk_inner_product_FLP

| N | Rust Scalar | C Scalar | Rust SIMD | C SIMD | Rust vs C SIMD |
|---|-------------|----------|-----------|--------|----------------|
| 64 | 38 ns | 25 ns | 6 ns | 7 ns | **1.2x faster** |
| 240 | 153 ns | 91 ns | 23 ns | 24 ns | ~matched |
| 480 | 312 ns | 176 ns | 50 ns | 49 ns | ~matched |
| 960 | 625 ns | 357 ns | 102 ns | 103 ns | ~matched |

Rust SIMD (AVX2+FMA) **matches** C SIMD (AVX2+FMA) for SILK inner product.

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
