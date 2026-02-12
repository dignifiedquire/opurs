# Benchmarks

Last updated: Final (all SIMD phases + C SIMD comparison + codec comparison)

Platform: x86_64 Linux (AVX2 available)

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
| silk_short_prediction | order=10 | 6 ns | 4 ns | 1.5x |
| silk_short_prediction | order=16 | 8 ns | 5 ns | 1.6x |
| silk_inner_product_FLP | 64 | 38 ns | 18 ns | 2.1x |
| silk_inner_product_FLP | 240 | 150 ns | 73 ns | 2.1x |
| silk_inner_product_FLP | 480 | 314 ns | 153 ns | 2.1x |
| silk_inner_product_FLP | 960 | 623 ns | 308 ns | 2.0x |

## Rust vs C Reference — Per-Function (with SIMD)

C reference is compiled **with SIMD** when the `simd` feature is active (RTCD dispatch:
AVX2 > SSE4.1 > SSE2 > SSE > scalar on x86).

### celt_pitch_xcorr

| N | Rust Scalar | C Scalar | Rust SIMD | C SIMD | Rust vs C SIMD |
|---|-------------|----------|-----------|--------|----------------|
| 240x60 | 2.54 us | 3.48 us | 431 ns | 456 ns | **1.06x faster** |
| 480x120 | 10.2 us | 13.8 us | 1.40 us | 1.37 us | ~matched |
| 960x240 | 39.7 us | 55.0 us | 5.74 us | 5.96 us | **1.04x faster** |

Rust SIMD (AVX2) **matches or slightly beats** C SIMD (AVX2+FMA) for pitch xcorr.

### silk_inner_product_FLP

| N | Rust Scalar | C Scalar | Rust SIMD | C SIMD | C SIMD advantage |
|---|-------------|----------|-----------|--------|------------------|
| 64 | 40 ns | 27 ns | 19 ns | 7.3 ns | 2.6x |
| 240 | 161 ns | 95 ns | 76 ns | 25 ns | 3.0x |
| 480 | 323 ns | 188 ns | 159 ns | 52 ns | 3.0x |
| 960 | 655 ns | 365 ns | 323 ns | 104 ns | 3.1x |

C SIMD uses **AVX2+FMA** (8-wide f32->f64 with fused multiply-add) while Rust SIMD
uses **SSE2** (2-wide f32->f64, separate mul+add). An AVX2 path for Rust would close
this gap.

## End-to-End Codec: Rust vs C (both with SIMD)

| Operation | Configuration | Rust | C (SIMD) | Ratio |
|-----------|--------------|------|----------|-------|
| Encode | 64 kbps stereo | 5.60 ms | 4.39 ms | 1.28x |
| Encode | 128 kbps stereo | 7.89 ms | 5.38 ms | 1.47x |
| Decode | 64 kbps stereo | 1.70 ms | 1.14 ms | 1.50x |
| Decode | 128 kbps stereo | 2.17 ms | 1.52 ms | 1.43x |
| Encode | 16 kbps mono VOIP | 10.68 ms | 8.27 ms | 1.29x |
| Encode | 64 kbps mono VOIP | 3.71 ms | 2.85 ms | 1.30x |

All measurements at 48 kHz, 20 ms frames, complexity 10, 1 second of audio (50 frames).

Rust is **~1.3-1.5x slower** than C with SIMD end-to-end. The gap comes primarily from:
1. SILK inner product using SSE2 vs C's AVX2+FMA (~3x per-function gap)
2. Remaining non-SIMD code paths (NSQ, LPC, VQ) where C has SIMD but Rust doesn't yet
3. Compiler differences (GCC vs LLVM code generation for scalar paths)

## SIMD Implementations

### x86/x86_64
- **SSE**: xcorr_kernel, celt_inner_prod, dual_inner_prod, celt_pitch_xcorr
- **SSE2**: silk_inner_product_FLP, silk_vad_energy
- **SSE4.1**: silk_noise_shape_quantizer_short_prediction
- **AVX2**: xcorr_kernel (8-wide), celt_pitch_xcorr (8 correlations/iter)

### aarch64
- **NEON**: xcorr_kernel, celt_inner_prod, dual_inner_prod, celt_pitch_xcorr, silk_noise_shape_quantizer_short_prediction

### Dispatch Hierarchy (x86/x86_64)

```
celt_pitch_xcorr: AVX2 > SSE > scalar
xcorr_kernel:     SSE > scalar
celt_inner_prod:  SSE > scalar
dual_inner_prod:  SSE > scalar
silk_short_pred:  SSE4.1 > scalar
silk_inner_FLP:   SSE2 > scalar
silk_vad_energy:  SSE2 > scalar
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
