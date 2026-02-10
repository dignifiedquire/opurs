# Phase 2: CELT Module Safety

**Goal**: Make all code under `src/celt/` free of `unsafe` blocks and
`unsafe fn` declarations. Every function has a `/// Upstream C:` comment.

**Prerequisites**: Phase 1 (test suite expansion) — comprehensive tests
must be in place before refactoring begins.

**Approach**: Bottom-up through the CELT dependency DAG. Leaf modules first,
integration hubs last.

---

## Dependency DAG

```
                    celt_encoder.rs ←── opus_encoder.rs
                   /    |    \    \
                  /     |     \    \
            bands.rs  mdct.rs  pitch.rs  celt_lpc.rs
           / | | \ \    |        |
          /  | |  \ \   |        |
    vq.rs  rate  qb  entenc  kiss_fft.rs
      |      |   |   entdec
  mathops  modes  entcode
      |
  float_cast
```

Key: `qb` = quant_bands.rs

---

## Stages

### Stage 2.1 — Math Primitives (no cross-module unsafe calls)

- [x] `float_cast.rs`
  - Upstream C: `celt/float_cast.h`
  - Scope: float ↔ int conversion helpers
  - Risk: Low — pure math, no pointers
- [x] `mathops.rs`
  - Upstream C: `celt/mathops.h`, `celt/mathops.c`
  - Functions: `isqrt32`, `fast_atan2f`, `celt_maxabs16`, `celt_cos_norm`, etc.
  - Risk: Low — bounded integer math
  - Watch: `celt_maxabs16` takes a pointer + length → convert to `&[f32]`
- [x] `celt_lpc.rs`
  - Upstream C: `celt/celt_lpc.h`, `celt/celt_lpc.c`
  - Functions: `_celt_lpc`, `_celt_autocorr`, `celt_fir_c`, `celt_iir`
  - Risk: Medium — inner loop pointer arithmetic for filter operations
  - Pattern: Convert `*const f32` + length → `&[f32]` slices
  - Note: `celt_fir_c`, `celt_iir`, `_celt_autocorr` retain internal unsafe
    blocks for calls to `xcorr_kernel_c`/`celt_pitch_xcorr_c` (pitch.rs Stage 2.3)
- [x] **Commit per file**: `refactor: make celt::<module> safe`

### Stage 2.2 — Entropy Coding

- [x] `entcode.rs` — already safe (`#![forbid(unsafe_code)]`), added upstream comments
- [x] `entenc.rs` — already safe (`#![forbid(unsafe_code)]`), added upstream comments
- [x] `entdec.rs` — already safe (`#![forbid(unsafe_code)]`), added upstream comments
- [x] `laplace.rs` — removed unsafe from all 3 functions, `value: *mut i32` → `&mut i32`
- [x] `cwrs.rs` — converted all 4 functions to safe slice APIs, removed `arch_h` module
- [x] **Commit**: `refactor: make celt entropy coding safe (Stage 2.2)`

### Stage 2.3 — Mid-level DSP

- [x] `modes.rs` — already safe, added upstream comments
- [x] `kiss_fft.rs` — already safe (`#![forbid(unsafe_code)]`), added upstream comments
- [x] `mdct.rs` — already safe (0 unsafe fn, 2 isolated unsafe blocks in ndutil), added upstream comments
- [x] `pitch.rs` — converted all 10 functions to safe slice APIs, removed `arch_h` module, `static mut second_check` → `const SECOND_CHECK`
  - Also removed remaining unsafe blocks from `celt_lpc.rs` (Stage 2.1 holdover)
- [x] `rate.rs` — converted all 5 functions to safe slice APIs, `static mut LOG2_FRAC_TABLE` → `const`
  - Upstream C: `celt/rate.h`, `celt/rate.c`
  - Functions: `clt_compute_allocation`, `bits2pulses`, `pulses2bits`, `interp_bits2pulses`, `get_pulses`
- [x] `quant_bands.rs` — converted all 9 functions to safe slice APIs, 4 `static mut` → `const`, removed `arch_h`/`stack_alloc_h` modules, replaced all `memcpy` with `copy_from_slice`
  - Upstream C: `celt/quant_bands.h`, `celt/quant_bands.c`
  - Functions: `quant_coarse_energy`, `quant_fine_energy`, `quant_energy_finalise`, `unquant_*`, `amp2Log2`, `loss_distortion`
- [x] `vq.rs` — converted all 9 functions to safe slice APIs, removed `arch_h` module
  - Upstream C: `celt/vq.h`, `celt/vq.c`
  - Functions: `exp_rotation`, `op_pvq_search_c`, `alg_quant`, `alg_unquant`, `renormalise_vector`, `stereo_itheta`
- [x] **Commit**: `refactor: make celt rate/vq/quant_bands safe (Stage 2.3b)`

### Stage 2.4 — Integration Hub

- [x] `bands.rs`
  - Upstream C: `celt/bands.h`, `celt/bands.c`
  - Functions (24 unsafe → 0): all converted to safe slice APIs
  - `quant_partition`, `quant_band`, `quant_band_stereo`, `quant_all_bands` — last 4 unsafe fns removed
  - `quant_all_bands` public API: `X_: &mut [f32]`, `Y_: Option<&mut [f32]>`
  - Internal raw pointers for X_/Y_/norm sub-slicing (borrow checker workaround for non-overlapping regions)
- [x] `celt.rs`
  - Upstream C: `celt/celt.h`, `celt/celt.c`
  - Functions (3 unsafe → 0): `init_caps`, `comb_filter`, `comb_filter_const_c` all safe
  - `comb_filter` split into separate-buffer and in-place variants
  - `static mut gains` → `const GAINS`
- [x] **Commit**: `refactor: make celt::celt safe (Stage 2.4)`
- [x] **Commit**: `refactor: make celt bands.rs safe (Stage 2.4)`

### Stage 2.5 — Encoder/Decoder

- [x] `celt_decoder.rs`
  - Upstream C: `celt/celt_decoder.c`
  - Functions (8 unsafe fn → 0): all converted to safe APIs
  - `celt_decode_with_ec`: `data: *const u8 + len` → `Option<&[u8]>`, `pcm: *mut` → `&mut [opus_val16]`
  - `opus_custom_decoder_ctl_impl`: removed `unsafe` (body was already safe)
  - `celt_synthesis`: raw double-pointer channels → separate `&mut [celt_sig]` params
  - `celt_decode_lost`: raw pointer struct access → direct field indexing, memmove → copy_within
  - `deemphasis`, `tf_decode`, `celt_plc_pitch_search`: pointer params → slices
  - 2 minimal unsafe blocks remain (non-overlapping slice creation, lifetime erasure for ec_dec)
- [x] `celt_encoder.rs`
  - Upstream C: `celt/celt_encoder.c`
  - Functions (20 unsafe fn → 0): all converted to safe APIs
  - `celt_encode_with_ec`: `st: *mut` → `&mut`, `pcm: *const` → `&[opus_val16]`, `compressed: *mut u8` → `&mut [u8]`
  - `opus_custom_encoder_init_arch`, `celt_encoder_init`: `unsafe fn *mut` → `fn &mut`
  - `opus_custom_encoder_ctl_impl`: `unsafe fn *mut` → `fn &mut`
  - Eliminated pointer aliases (oldBandE/oldLogE/oldLogE2/energyError) → direct field indexing
  - Replaced all memcpy/memset with copy_from_slice/copy_within/fill
  - 1 minimal unsafe block remains (energy_mask from_raw_parts, externally-owned pointer)
- [x] **Commit**: `refactor: make celt::celt_decoder safe (Stage 2.5)` (3 commits)
- [x] **Commit(s)**: `refactor: make celt::celt_encoder safe` (5 commits: FAM elimination, leaf fns, analysis/prefilter, init/CTL, celt_encode_with_ec)

---

## Cross-Reference Template

Every function gets this comment format:

```rust
/// Upstream C: celt/mathops.h:isqrt32
pub fn isqrt32(val: u32) -> u32 {
```

For functions split across header and implementation:
```rust
/// Upstream C: celt/bands.c:compute_band_energies
/// Upstream C header: celt/bands.h
pub fn compute_band_energies(mode: &OpusCustomMode, ...) {
```

---

## Refactoring Patterns for CELT

### Pointer + length → slice
```rust
// Before (c2rust)
unsafe fn celt_maxabs16(x: *const f32, len: i32) -> f32 {
    let mut maxval = 0.0f32;
    for i in 0..len { maxval = maxval.max((*x.offset(i as isize)).abs()); }
    maxval
}

// After
/// Upstream C: celt/mathops.h:celt_maxabs16
fn celt_maxabs16(x: &[f32]) -> f32 {
    x.iter().fold(0.0f32, |max, &v| max.max(v.abs()))
}
```

### Buffer with stride → chunks
```rust
// Before: pointer arithmetic with stride
for i in 0..n { *out.offset((i * stride) as isize) = val; }

// After: chunks or step_by
for (i, chunk) in out.chunks_exact_mut(stride).enumerate().take(n) {
    chunk[0] = val;
}
```

### ec_ctx buffer → owned Vec or borrowed slice
```rust
// Before
pub struct ec_ctx { pub buf: *mut u8, pub storage: u32, ... }

// After
pub struct ec_ctx { pub buf: Vec<u8>, ... }
// or for decode:
pub struct ec_dec<'a> { pub buf: &'a [u8], ... }
```

---

## Definition of Done

- [x] Zero `unsafe fn` declarations in `src/celt/` (except FFI boundaries if any)
- [ ] Zero `unsafe {}` blocks in `src/celt/` — **21 remain across 4 files:**
  - `bands.rs`: 16 blocks (non-overlapping slice creation from raw pointers for X/Y band splitting — borrow checker workaround)
  - `celt_decoder.rs`: 2 blocks (non-overlapping channel slice creation, lifetime erasure for ec_dec)
  - `celt_encoder.rs`: 1 block (energy_mask from_raw_parts, externally-owned pointer)
  - `mdct.rs`: 2 blocks (ndarray interlaced view splitting)
- [x] Every function has `/// Upstream C:` comment
- [x] All tests pass (cargo test + vector tests)
- [x] Clippy clean, formatted
- [x] `externs::{memcpy,memmove,memset}` no longer called from celt/
