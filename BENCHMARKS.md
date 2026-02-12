# Benchmarks

Last updated: Phase 6 (all SIMD phases complete)

Run with: `cargo bench --bench pitch`, `cargo bench --bench silk`, `cargo bench --bench vq`

## CELT Pitch Functions

| Function | N | SIMD Implementation |
|----------|---|---------------------|
| xcorr_kernel | 64-960 | SSE (x86), NEON (aarch64) |
| celt_inner_prod | 64-960 | SSE (x86), NEON (aarch64) |
| dual_inner_prod | 64-960 | SSE (x86), NEON (aarch64) |
| celt_pitch_xcorr | 240-960 | AVX2 8-wide / SSE 4-wide (x86), NEON (aarch64) |

## SILK Functions

| Function | N | SIMD Implementation |
|----------|---|---------------------|
| silk_short_prediction | order=10,16 | SSE4.1 (x86), NEON (aarch64) |
| silk_inner_product_FLP | 64-960 | SSE2 (x86) |
| silk_vad_energy | variable | SSE2 (x86) |

## Implemented SIMD Functions

### Phase 1: CELT Pitch (SSE + NEON)
- `xcorr_kernel_sse` / `xcorr_kernel_neon`
- `celt_inner_prod_sse` / `celt_inner_prod_neon`
- `dual_inner_prod_sse` / `dual_inner_prod_neon`
- `celt_pitch_xcorr_sse` / `celt_pitch_xcorr_neon`

### Phase 2: SILK NSQ (SSE4.1 + NEON)
- `silk_noise_shape_quantizer_short_prediction_sse4_1` / `_neon`

### Phase 3: SILK Float Inner Product (SSE2)
- `silk_inner_product_flp_sse2`

### Phase 5: SILK Supporting (SSE2)
- `silk_vad_energy_sse2`

### Phase 6: AVX2 Pitch Xcorr
- `xcorr_kernel_avx2` (8-wide cross-correlation)
- `celt_pitch_xcorr_avx2` (8 correlations per iteration)

## Dispatch Hierarchy (x86/x86_64)

```
celt_pitch_xcorr: AVX2 > SSE > scalar
xcorr_kernel:     SSE > scalar
celt_inner_prod:  SSE > scalar
dual_inner_prod:  SSE > scalar
silk_short_pred:  SSE4.1 > scalar
silk_inner_FLP:   SSE2 > scalar
silk_vad_energy:  SSE2 > scalar
```

## Notes

- All SIMD paths maintain bit-exact codec output (verified by 84 unit/integration tests)
- No FMA intrinsics used (mul+add only) to preserve rounding consistency
- aarch64 NEON is compile-time dispatched (always available)
- x86 features detected at runtime via `cpufeatures` crate (cached after first check)
- `--no-default-features` disables all SIMD, falling through to scalar
