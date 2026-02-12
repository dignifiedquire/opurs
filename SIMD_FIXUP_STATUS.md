# SIMD Bit-Exactness Fixup Status

## Context
PR #3 ("feat: more simd") merged at commit 404612f, adding ~5K lines of SIMD
implementations. CI had multiple failures. This document tracks all fixes applied
and remaining work.

## Current State
- **HEAD**: 3aa8f7a on main (pushed)
- **CI run**: 21967129595 — still completing, but x86 vector tests still failing
- **ARM64**: All 228/228 standard vectors PASS, DNN 204/264 PASS (60 OSCE known)
- **x86_64**: Still failing — need to verify on actual x86_64 hardware

## Completed Fixes

### 1. cargo fmt + clippy (previous sessions)
Fixed formatting issues from PR #3 code and clippy warnings.

### 2. cfg(feature = "simd") gates in NSQ_del_dec.rs
`src/silk/NSQ_del_dec.rs` lines 328-345 and 366-400 called `super::simd::*` functions
gated only on `target_arch` but not on `feature = "simd"`. Fixed by adding proper
`#[cfg(feature = "simd")]` gates.

### 3. SILK SSE4.1 sDiff_shp_Q14 stale-local bug
`src/silk/simd/x86.rs` — `silk_noise_shape_quantizer_10_16_sse4_1` had a stale local
variable for `sDiff_shp_Q14` that wasn't being updated per-sample like the C version.
Fixed to reload from the SSE register each iteration.

### 4. xcorr_kernel_sse tail loop accumulation order
`src/celt/simd/x86.rs` — Rust combined xsum1+xsum2 BEFORE the tail loop, then
accumulated all tail elements into xsum1. C alternates between xsum1/xsum2 for each
tail element, then combines at the end. Fixed to match C's alternating pattern.
Committed in 08292b4. (No effect on CI — lengths are multiples of 4 in practice.)

### 5. silk_inner_product_flp dispatch SSE2→scalar
`src/silk/simd/mod.rs` — Rust dispatched `silk_inner_product_flp` to SSE2 on x86, but
the C RTCD table (`silk/x86/x86_silk_map.c`) only dispatches to AVX2 for this function.
Fixed by removing the SSE2 dispatch path, keeping only AVX2.
(No effect on AVX2 CI runners, but correct for non-AVX2 machines.)

### 6. VQ_WMat_EC unaligned read
`src/silk/simd/x86.rs` line ~1181 — `*(cb_ptr as *const i32)` on a `&[i8]` was
unaligned. Fixed with `(cb_ptr as *const i32).read_unaligned()`.

### 7. xcorr_kernel_avx2 FMA fix (THE MAIN FIX) — commit 3aa8f7a
**Root cause**: C `xcorr_kernel_avx` in `celt/x86/pitch_avx.c` uses `_mm256_fmadd_ps`
(fused multiply-add), which produces different results from separate
`_mm256_mul_ps` + `_mm256_add_ps` due to single vs double rounding.

**Changes**:
- `src/celt/simd/x86.rs`: `xcorr_kernel_avx2` — replaced all `_mm256_add_ps(xsumN, _mm256_mul_ps(x0, ...))` with `_mm256_fmadd_ps(x0, ..., xsumN)` in both main loop and masked tail
- Added `#[target_feature(enable = "fma")]` alongside `"avx2"` on both `xcorr_kernel_avx2` and `celt_pitch_xcorr_avx2`
- `src/celt/simd/mod.rs`: Changed `cpuid_avx2` to `cpuid_avx2_fma` (checks both AVX2+FMA)
- Removed intermediate SSE 4-wide loop from `celt_pitch_xcorr_avx2` tail — C goes directly from 8-wide AVX2 to scalar `celt_inner_prod` (dispatched to SSE via RTCD)

## What To Do Next on x86_64 Machine

### 1. Pull and run vector tests locally
```bash
git pull origin main
# Download test vectors if not present:
curl https://www.ietf.org/proceedings/98/slides/materials-98-codec-opus-newvectors-00.tar.gz -o vectors.tar.gz
tar -xzf vectors.tar.gz
# Run with dump dir to see exact failures:
cargo run --release --features tools --example run_vectors2 -- opus_newvectors --dump-dir dump/
```

### 2. If standard vectors still fail, investigate remaining SIMD functions
Functions that could cause x86 encode divergence (in priority order):

a. **SILK NSQ SSE4.1 quantizer** (`silk_noise_shape_quantizer_10_16_sse4_1` in
   `src/silk/simd/x86.rs`): Complex register packing. The sDiff_shp_Q14 fix was
   applied but there may be other subtle differences. Test by temporarily disabling
   (set `use_it = false` in `src/silk/NSQ.rs` around line 100) to isolate.

b. **comb_filter_const_sse** (`src/celt/simd/x86.rs`): New in PR #3, used in CELT
   pitch prefilter. Compare shuffle patterns carefully with C
   `celt/x86/pitch_sse.c:comb_filter_const_sse`.

c. **op_pvq_search_sse2** (`src/celt/simd/x86.rs`): Uses `_mm_rsqrt_ps`. Already
   verified to match C reference at `../libopus/opus/celt/x86/vq_sse2.c` (same
   algorithm, NOT the double-precision version in libopus-sys).

### 3. DNN vector failures
- 60 OSCE decode failures occur on ALL platforms (including ARM64 without SIMD)
- These are inherent scalar FP ordering differences in neural net inference
- Not SIMD-related — will need separate investigation or acceptance

### 4. i686 (32-bit) failures
- May be related to the x87 vs SSE2 FPU issue (see 32-bit Linux Fix in MEMORY.md)
- Or may be same root cause as x86_64 if FMA fix doesn't resolve everything

## C Reference Source Location
**IMPORTANT**: The correct C reference is at `../libopus/opus/` (i.e.,
`/Users/dignified/opensource/libopus/opus/`), NOT `libopus-sys/opus/` in the repo.
The libopus-sys version may differ (e.g., vq_sse2.c has a completely different
algorithm). Always compare against `../libopus/opus/` for bit-exactness work.

## Key C RTCD Dispatch Tables
- CELT: `../libopus/opus/celt/x86/x86_celt_map.c`
  - `celt_pitch_xcorr` → AVX2 only (not SSE)
  - `xcorr_kernel` → SSE at all SIMD levels
  - `celt_inner_prod` → SSE at all SIMD levels
  - `dual_inner_prod` → SSE at all SIMD levels
  - `comb_filter_const` → SSE at all SIMD levels
  - `op_pvq_search` → SSE2 at SSE2+ levels
- SILK: `../libopus/opus/silk/x86/x86_silk_map.c`
  - `silk_inner_product_FLP` → AVX2 only (not SSE2)

## Key Files Modified
| File | Changes |
|------|---------|
| `src/celt/simd/x86.rs` | FMA in xcorr_kernel_avx2, tail loop fix in xcorr_kernel_sse, removed SSE 4-wide fallback from pitch_xcorr_avx2 |
| `src/celt/simd/mod.rs` | cpuid_avx2 → cpuid_avx2_fma |
| `src/silk/simd/x86.rs` | sDiff_shp_Q14 fix, VQ_WMat_EC unaligned read |
| `src/silk/simd/mod.rs` | Removed silk_inner_product_flp SSE2 dispatch |
| `src/silk/NSQ_del_dec.rs` | Added cfg(feature = "simd") gates |

## Commits (chronological)
- Previous session: fmt, clippy, cfg gates, sDiff_shp_Q14 fix
- `08292b4`: xcorr_kernel_sse tail loop + re-enable NSQ SSE4.1
- `8693870`: silk_inner_product_flp SSE2→scalar + VQ_WMat_EC unaligned read
- `3aa8f7a`: **xcorr_kernel_avx2 FMA fix** (the main fix)
