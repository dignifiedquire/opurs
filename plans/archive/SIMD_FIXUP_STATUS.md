# SIMD Bit-Exactness Fixup Status

> Archived historical status note. Current active planning is tracked in
> `../remaining-plan.md`.

## Context
PR #3 ("feat: more simd") merged at commit 404612f, adding ~5K lines of SIMD
implementations across CELT and SILK. CI had multiple failures on x86 targets.
This document tracks all fixes applied.

## Current State: COMPLETE

- **Standard vector tests**: 228/228 PASS on all 6 targets (linux-x86_64,
  linux-i686, macos-arm64, macos-x86_64, windows-x86_64, windows-i686)
- **DNN vector tests**: PASS on macos-arm64; known failures on x86_64 (OSCE
  decode — pre-existing FP ordering issue, not SIMD-related)
- **DNN unit tests**: `test_vec_tanh_dispatch_matches_scalar` fails on all
  platforms — pre-existing issue, not related to SIMD fixup work

## All Fixes Applied (chronological)

### 1. cargo fmt + clippy
Fixed formatting issues from PR #3 code and clippy warnings.

### 2. cfg(feature = "simd") gates in NSQ_del_dec.rs
`src/silk/NSQ_del_dec.rs` called `super::simd::*` functions gated only on
`target_arch` but not on `feature = "simd"`. Fixed by adding proper
`#[cfg(feature = "simd")]` gates.

### 3. SILK SSE4.1 sDiff_shp_Q14 stale-local bug — commit afa6087
`src/silk/simd/x86.rs` — `silk_noise_shape_quantizer_10_16_sse4_1` had a stale
local variable for `sDiff_shp_Q14` that wasn't being updated per-sample like the
C version. Fixed to reload from the SSE register each iteration.

### 4. xcorr_kernel_sse tail loop accumulation order — commit 08292b4
`src/celt/simd/x86.rs` — Rust combined xsum1+xsum2 BEFORE the tail loop, then
accumulated all tail elements into xsum1. C alternates between xsum1/xsum2 for
each tail element, then combines at the end. Fixed to match C's alternating
pattern. (No effect in practice — lengths are multiples of 4.)

### 5. silk_inner_product_flp dispatch SSE2→scalar — commit 8693870
`src/silk/simd/mod.rs` — Rust dispatched `silk_inner_product_flp` to SSE2 on x86,
but the C RTCD table only dispatches to AVX2 for this function. Fixed by removing
the SSE2 dispatch path, keeping only AVX2.

### 6. VQ_WMat_EC unaligned read — commit 8693870
`src/silk/simd/x86.rs` — `*(cb_ptr as *const i32)` on a `&[i8]` was unaligned.
Fixed with `(cb_ptr as *const i32).read_unaligned()`.

### 7. xcorr_kernel_avx2 FMA fix (THE MAIN x86_64 FIX) — commit 3aa8f7a
**Root cause**: C `xcorr_kernel_avx` uses `_mm256_fmadd_ps` (fused multiply-add),
which produces different results from separate `_mm256_mul_ps` + `_mm256_add_ps`
due to single vs double rounding.

**Changes**:
- Replaced all mul+add with `_mm256_fmadd_ps` in xcorr_kernel_avx2
- Added `#[target_feature(enable = "fma")]` alongside `"avx2"`
- Changed `cpuid_avx2` to `cpuid_avx2_fma` (checks both AVX2+FMA)
- Removed intermediate SSE 4-wide loop from pitch_xcorr_avx2 tail

### 8. op_pvq_search_sse2 N%4 guard removal (THE MAIN encode fix) — commit f3a279e
**Root cause**: Rust added an `(N & 3) == 0` guard before using the SSE2 PVQ
search, falling back to scalar for non-aligned band sizes. The C reference has
no such guard — it always uses op_pvq_search_sse2 at arch >= 2, handling
arbitrary N via N+3 zero-padding with sentinel values.

CELT band sizes get recursively halved in `quant_partition` (e.g. 12→6→3),
producing N values that aren't multiples of 4. This caused Rust to use scalar
PVQ search while C used SSE2, producing numerically different (both valid)
results. Fixed by removing the N%4 guard.

Also added Rust-vs-C SIMD comparison tests and encoder comparison tests.

### 9. silk_LPC_analysis_filter_avx2 unaligned read — commit 0e5d74a
`src/silk/simd/x86.rs` — `*(in_ptr.sub(10) as *const i32)` and
`*(b.as_ptr().add(8) as *const i32)` loaded 2 i16 coefficients as a single i32
via direct dereference, which panics in debug mode when the i16 pointer isn't
4-byte aligned. Fixed with `read_unaligned()`.

### 10. Test file cross-platform fixes — commit d371964
- `tests/simd_comparison.rs`: Added `target_arch = "x86_64"` cfg gate so x86
  SIMD comparison tests don't attempt to compile on ARM
- `tests/encoder_comparison.rs`: Replaced manual `extern "C"` declarations with
  proper `libopus_sys` imports

## Key C RTCD Dispatch Tables
- CELT: `celt/x86/x86_celt_map.c`
  - `celt_pitch_xcorr` → AVX2 only (not SSE)
  - `xcorr_kernel` → SSE at all SIMD levels
  - `celt_inner_prod` → SSE at all SIMD levels
  - `dual_inner_prod` → SSE at all SIMD levels
  - `comb_filter_const` → SSE at all SIMD levels
  - `op_pvq_search` → SSE2 at SSE2+ levels
- SILK: `silk/x86/x86_silk_map.c`
  - `silk_inner_product_FLP` → AVX2 only (not SSE2)

## Key Files Modified
| File | Changes |
|------|---------|
| `src/celt/simd/x86.rs` | FMA in xcorr_kernel_avx2, tail loop fix in xcorr_kernel_sse |
| `src/celt/simd/mod.rs` | cpuid_avx2 → cpuid_avx2_fma, removed N%4 guard on PVQ search |
| `src/silk/simd/x86.rs` | sDiff_shp_Q14 fix, VQ_WMat_EC unaligned read, LPC filter unaligned read |
| `src/silk/simd/mod.rs` | Removed silk_inner_product_flp SSE2 dispatch |
| `src/silk/NSQ_del_dec.rs` | Added cfg(feature = "simd") gates |
| `tests/simd_comparison.rs` | New: Rust-vs-C SIMD comparison tests (x86_64 only) |
| `tests/encoder_comparison.rs` | New: frame-by-frame Rust-vs-C encoder comparison |

## Remaining Known Issues (NOT SIMD-related)
- DNN `test_vec_tanh_dispatch_matches_scalar` fails on all platforms
- DNN OSCE vector tests: 60 decode failures on x86_64 due to FP ordering
  differences in neural net inference (scalar, not SIMD)

## Open TODOs (SIMD parity follow-up)
- SILK SSE4.1 NSQ dispatch remains disabled by `9c86ede` (`src/silk/simd/mod.rs:224`)
  because it is not bit-exact yet.
- Repro status (x86_64): re-enabling `use_nsq_sse4_1()` causes broad classic
  vector parity regressions (example: `testvector11` full parity run produced
  `72` failures in `ENC` kinds).
- Repro command:
  `CARGO_TARGET_DIR=target-local cargo run --release --features tools --target x86_64-apple-darwin --example run_vectors2 -- --suite classic --mode parity --matrix full --vector-name testvector11 opus_newvectors`
- Action item: isolate non-bitexact SSE4.1 NSQ kernels and re-enable dispatch
  incrementally with vector-guarded tests.
