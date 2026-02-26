# Performance Status: Rust vs C Encode/Decode Gap

## Current Results (2026-02-26)

Measured with `cargo bench --features tools --bench codec_comparison`, bench profile
(`lto = "thin"`, `codegen-units = 1`), x86_64.

| Benchmark | Rust (ms) | C (ms) | Gap | Category |
|-----------|-----------|--------|-----|----------|
| Encode 64kbps stereo | 5.12 | 4.30 | **19.0%** | SILK+CELT hybrid |
| Encode 128kbps stereo | 6.83 | 5.44 | **25.5%** | CELT-heavy |
| Encode 16kbps mono VOIP | 8.45 | 7.71 | **9.7%** | SILK-only |
| Encode 64kbps mono VOIP | 3.38 | 2.98 | **13.5%** | mixed |
| Decode 64kbps stereo | 1.26 | 1.06 | **19.2%** | |
| Decode 128kbps stereo | 1.69 | 1.42 | **19.3%** | |

Starting point was 6.80ms vs 5.33ms (27.6% gap) on 64kbps stereo encode.

## What Was Done

### `#[inline(always)]` on `short_prediction_c` (commit e1dbc55)

The single biggest win: **24.7% speedup** on 64kbps stereo encode.

`silk_noise_shape_quantizer_short_prediction_c` was called ~240 times per frame as a
non-inlined function. The assembly showed a separate 491-byte function at `0x2540b0` with
4 bounds-check panic branches. LLVM chose not to inline it despite `#[inline]` because the
function takes slice references (opaque length) and has panic paths.

Changing to `#[inline(always)]` on the implementation (not just the dispatch wrapper)
eliminated the call overhead and let LLVM merge bounds checks with the caller's context.

Also applied in the same commit:
- `#[inline(always)]` on 8 SIMD dispatch wrappers in `celt/simd/mod.rs` and `silk/simd/mod.rs`
- Pre-sliced arrays in `NSQ.rs`, `NSQ_del_dec.rs`, `pitch.rs`, `vq.rs` to hoist bounds checks
- `lto = "thin"` + `codegen-units = 1` in `[profile.release]`
- Documented osce_nndsp debug-mode FP divergence

The pre-slicing and release profile LTO had **zero measurable impact** on benchmarks because
the bench profile already had LTO, and LTO's range analysis already eliminates most bounds
checks after inlining. The `#[inline(always)]` was the only change that mattered.

### Cold path extraction from `opus_encode_native` (commit 284811e)

Modest but real: **~3% additional speedup** on 64kbps stereo.

Extracted 6 rarely-executed code paths into `#[cold] #[inline(never)]` functions:
- `encode_low_bitrate_frame` — early return for very low bitrates
- `init_silk_after_celt` — SILK reinit on mode switch (28-field struct)
- `apply_hb_gain_fade` — HB gain transition (includes Vec clone)
- `apply_stereo_fade` — stereo width transition (includes Vec clone)
- `encode_silk_to_celt_redundancy` — one `celt_encode_with_ec` call
- `encode_celt_to_silk_redundancy` — two `celt_encode_with_ec` calls

`opus_encode_native` shrank from 56,631 to 55,440 bytes (only -2.1%). LTO partially
re-inlined some helpers despite `#[inline(never)]`. The performance gain likely comes from
improved branch prediction (cold paths marked unlikely) rather than code size reduction.

## Root Cause Analysis

### Assembly comparison methodology

Dumped and compared x86_64 assembly from the benchmark binary for hot functions using
`objdump` + `nm -S` to extract symbol boundaries.

### Key findings

**Code size (bytes)**:

| Function | Rust | C | Ratio |
|----------|------|---|-------|
| `opus_encode_native` | 55,440 | 9,550 | 5.8x |
| `celt_encode_with_ec` | 52,469 | 33,438 | 1.6x |
| `silk_NSQ_c` (+ inlined quantizer) | 8,900 | 6,418 | 1.4x |
| `xcorr_kernel_scalar` | 737 | 743 | 1.0x |

**Branch counts**:

| Function | Rust | C | Ratio |
|----------|------|---|-------|
| `opus_encode_native` | 745 | 320 | 2.3x |
| `celt_encode_with_ec` | 1,527 | 818 | 1.9x |
| `silk_NSQ_c` | 170 | 80 | 2.1x |
| `xcorr_kernel_scalar` | 18 | 8 | 2.3x |

**perf stat** (encoding only):

| Metric | Rust | C |
|--------|------|---|
| Instructions | 137B | 145B |
| Branches | 20B | 13.6B |
| Cache miss rate | 6.46% | 2.95% |

Rust executes **fewer instructions** but has **48% more branches** (bounds checks) and
**2.2x higher cache miss rate** (icache pressure from code bloat).

**Heap allocations**: Rust's `silk_NSQ_c` calls `calloc` 3 times and `free` 5 times per
invocation. C uses stack VLAs. A previous attempt to convert these to stack arrays (ArrayVec)
showed **zero measurable impact** — the allocator is fast enough that this isn't the bottleneck.

**Panic infrastructure**: `celt_encode_with_ec` has 46 calls to panic/slice_index_fail
scattered throughout, each generating a cold branch + call site that inflates code size.

## Outstanding Issues

### 1. `celt_encode_with_ec` code bloat (52KB vs 33KB)

The largest remaining contributor to the CELT-heavy benchmarks (128kbps stereo at 25.5% gap).
Unlike `opus_encode_native`, most of this function is hot path — there are fewer obvious cold
blocks to extract. The 1.6x bloat comes from:

- **Bounds check branches**: 1,527 vs 818 (~700 extra). Each generates a compare + conditional
  jump + cold-path panic call site.
- **Monomorphization**: Generic helpers (`ec_enc_*`, `ec_dec_*`) instantiated at each call site.
- **Rust safety overhead**: Slice indexing through trait dispatch rather than raw pointer arithmetic.

Potential approaches (not yet attempted):
- Audit the hottest inner loops for `unsafe { get_unchecked }` opportunities where bounds are
  provably safe (e.g., band iteration loops where indices are bounded by `eBands` tables).
- Extract remaining cold paths (error handling, edge cases).
- Profile-guided optimization (PGO) — may help LLVM make better inlining decisions.

### 2. Decoder gap (~19%)

Decode performance hasn't been optimized at all. The decoder has a similar `celt_decode_with_ec`
function that likely has the same code bloat pattern. The `short_prediction` inlining fix
applies to decode too (the function is shared), but no decoder-specific work has been done.

### 3. 128kbps stereo encode (25.5% gap)

Worse than 64kbps because this configuration is more CELT-heavy (higher bandwidth = more CELT
work). The SILK-only path is already within 10%, so the remaining gap is almost entirely in
CELT encoding.

### 4. Structural Rust overhead

Some gap is inherent to safe Rust:
- Every `slice[i]` generates a bounds check unless LLVM can prove it's in range.
- Vec/slice fat pointers (ptr + len) vs C's raw pointers.
- No VLAs — dynamic arrays require heap allocation or fixed-size stack buffers.

A realistic floor for safe Rust is probably 5-10% slower than C for this type of codec.
Getting below 15% across all benchmarks would require either targeted `unsafe` in hot loops
or PGO.

## Not Attempted / Ruled Out

- **Heap→stack conversion (ArrayVec)**: Tried and abandoned. Zero impact. The allocator is
  not the bottleneck.
- **`denormalise_bands` pre-slicing**: Caused index-out-of-bounds failures. Band ranges can
  exceed the `bound` parameter when `downsample != 1`. Reverted.
- **Struct field reordering**: `silk_nsq_state` is `#[repr(C)]` matching upstream layout.
  Would break ABI compatibility.
- **Cache prefetch hints**: Speculative; the cache miss issue is icache (code), not dcache (data).

## Recommended Next Steps (by expected impact)

1. **PGO build** — Profile-guided optimization may help LLVM eliminate cold branches and
   improve inlining decisions in `celt_encode_with_ec`. Expected: 5-10%.
2. **`celt_encode_with_ec` cold path extraction** — Similar treatment as `opus_encode_native`.
   Smaller expected impact since less code is truly cold. Expected: 2-5%.
3. **Targeted `unsafe get_unchecked`** in CELT band loops — The `quant_all_bands` inner loops
   iterate over `eBands`-bounded indices where bounds are statically provable. Expected: 2-5%
   on CELT-heavy benchmarks.
4. **Decoder optimization** — Apply the same analysis to the decode path. The 19% gap there
   is likely similar root causes.
