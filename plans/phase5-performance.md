# Phase 5: Performance Parity

**Goal**: Close the ~20% performance gap between the Rust implementation and
the C version with ASM/intrinsics. Establish a benchmark suite and achieve
performance parity through SIMD, inlining, and allocation optimization.

**Prerequisites**: Phases 1–4 complete (safe code, full test suite). Working
with safe code makes performance optimization easier to reason about.

---

## Current State (updated 2026-02-13)

- C version with inline assembly + intrinsics + RTCD is ~20% faster
- Rust version now has **full SIMD parity** with C reference (PR #3 + fixups)
  - All C SIMD functions ported to Rust `std::arch` intrinsics
  - Runtime CPU detection via `is_x86_feature_detected!`
  - 228/228 standard vector tests pass with SIMD enabled on all 6 CI targets
- No benchmark suite exists yet
- Performance gap likely reduced significantly (SIMD was the main contributor)

---

## Stages

### Stage 5.1 — Establish Benchmark Suite

- [ ] Create `benches/` directory with `criterion` benchmarks
- [ ] Benchmark encode at multiple bitrates (10k, 60k, 120k, 240k bps)
- [ ] Benchmark decode at multiple sample rates (8k, 16k, 48k Hz)
- [ ] Benchmark individual hot functions (identified by profiling):
  - FFT/IFFT (`kiss_fft`)
  - MDCT forward/backward
  - Pitch analysis (`pitch_search`, `xcorr_kernel`)
  - Inner products (`celt_inner_prod`, `dual_inner_prod`)
  - SILK noise shaping quantizer (NSQ)
  - Band energy computation
- [ ] Benchmark against upstream C via tools examples timing
- [ ] Document baseline numbers in this file
- [ ] **Commit**: `perf: add criterion benchmark suite`

### Stage 5.2 — Profile and Identify Hot Paths

- [ ] Profile with `cargo flamegraph` on encode workload
- [ ] Profile with `cargo flamegraph` on decode workload
- [ ] Identify top-10 hottest functions by CPU time
- [ ] Document findings here:

| Rank | Function | % of encode time | % of decode time |
|------|----------|-----------------|-----------------|
| 1 | (TBD) | | |
| 2 | (TBD) | | |
| ... | | | |

### Stage 5.3 — Compiler-Level Optimizations

Low-hanging fruit that doesn't require SIMD:

- [ ] Add `#[inline]` / `#[inline(always)]` to hot small functions
  - Math helpers in mathops, SigProc_FIX, Inlines
  - Entropy coder hot path functions
  - Inner product kernels
- [ ] Ensure hot loops use iterators (autovectorization-friendly)
- [ ] Check for unnecessary bounds checks in hot paths
  - Use `get_unchecked` only if profiling confirms the bounds check matters
  - Prefer `.chunks_exact()` and `.split_at()` which elide bounds checks
- [ ] Review allocation patterns:
  - Avoid allocating in encode/decode hot paths
  - Pre-allocate buffers in state structs
  - Use stack arrays for small fixed-size buffers
- [ ] LTO (Link-Time Optimization) — evaluate in Cargo.toml:
  ```toml
  [profile.release]
  lto = "thin"
  ```
- [ ] **Commit**: `perf: add inline annotations and optimize hot paths`

### Stage 5.4 — SIMD Intrinsics for CELT ✅

All C SIMD functions ported in PR #3 + fixups. Bit-exact with C reference.

- [x] `celt_inner_prod` / `dual_inner_prod` — SSE inner product
- [x] `xcorr_kernel` — SSE cross-correlation + AVX2+FMA variant
- [x] `celt_pitch_xcorr` — AVX2+FMA pitch cross-correlation
- [x] `comb_filter_const` — SSE comb filter
- [x] `op_pvq_search` — SSE2 PVQ search
- [x] Runtime CPU detection via `is_x86_feature_detected!`
- [x] NEON variants for aarch64
- [ ] `kiss_fft` butterfly operations — not ported (C has no SSE variant for this)
- [ ] `clt_mdct_forward` / `clt_mdct_backward` — not ported (C has no SSE variant)

### Stage 5.5 — SIMD Intrinsics for SILK ✅

All C SIMD functions ported in PR #3 + fixups. Bit-exact with C reference.

- [x] `silk_NSQ` / `silk_NSQ_del_dec` — SSE4.1 noise shaping quantizer
- [x] `silk_inner_product_FLP` — AVX2 inner product
- [x] `silk_VQ_WMat_EC` — SSE4.1 VQ search
- [x] `silk_LPC_analysis_filter` — AVX2 LPC analysis
- [x] `silk_burg_modified` — SSE4.1 Burg's method
- [x] `silk_warped_autocorrelation` — SSE4.1 warped autocorrelation
- [ ] `silk_pitch_analysis_core` — not ported (C has no SIMD variant)

### Stage 5.6 — Validate Performance Parity

- [ ] Re-run full benchmark suite
- [ ] Compare against C baseline
- [ ] Target: within 5% of C with ASM/intrinsics
- [ ] Document final numbers:

| Workload | C (ms) | Rust before (ms) | Rust after (ms) | Δ |
|----------|--------|------------------|-----------------|---|
| Encode 48kHz stereo 120kbps | | | | |
| Decode 48kHz stereo | | | | |
| ... | | | | |

- [ ] Run vector tests — verify SIMD paths produce bit-exact results
- [ ] **Commit**: `perf: document performance parity results`

---

## SIMD Architecture Strategy

Use a dispatch pattern that mirrors the upstream RTCD (Runtime CPU Detection):

```rust
/// Upstream C: celt/pitch.c:celt_inner_prod
fn celt_inner_prod(x: &[f32], y: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("sse2") {
            return celt_inner_prod_sse2(x, y);
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        return celt_inner_prod_neon(x, y);
    }
    celt_inner_prod_scalar(x, y)
}
```

Keep scalar fallback for all architectures. SIMD functions use `unsafe`
blocks internally but are wrapped in safe public API — this is acceptable
since `#![forbid(unsafe_code)]` can be relaxed to `#![deny(unsafe_code)]`
with explicit `#[allow(unsafe_code)]` on SIMD modules only.

---

## Definition of Done

- [ ] Benchmark suite in `benches/` with criterion
- [ ] Performance within 5% of C reference on x86_64 and aarch64
- [ ] SIMD for top-5 hottest functions
- [ ] Runtime CPU detection
- [ ] All vector tests pass (bit-exact output preserved)
- [ ] Documented benchmark results
