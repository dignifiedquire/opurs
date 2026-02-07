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
- [ ] `rate.rs`
  - Upstream C: `celt/rate.h`, `celt/rate.c`
  - Functions: `clt_compute_allocation`, `bits2pulses`, `pulses2bits`
  - Risk: Medium — array indexing into mode tables
- [ ] `quant_bands.rs`
  - Upstream C: `celt/quant_bands.h`, `celt/quant_bands.c`
  - Functions: `quant_coarse_energy`, `quant_fine_energy`, `quant_energy_finalise`, `unquant_*`
  - Risk: Medium — uses entenc/entdec + mode tables
- [ ] `vq.rs`
  - Upstream C: `celt/vq.h`, `celt/vq.c`
  - Functions: `exp_rotation`, `op_pvq_search_c`, `alg_quant`, `alg_unquant`, `renormalise_vector`, `stereo_itheta`
  - Risk: High — core vector quantization with pointer manipulation
- [ ] **Commit per file**: `refactor: make celt::<module> safe`

### Stage 2.4 — Integration Hub

- [ ] `bands.rs`
  - Upstream C: `celt/bands.h`, `celt/bands.c`
  - Functions (24 unsafe): `compute_band_energies`, `normalise_bands`, `denormalise_bands`, `anti_collapse`, `spreading_decision`, `haar1`, `quant_all_bands`, etc.
  - Risk: **Very High** — 2146 lines, calls vq, quant_bands, entenc, entdec, mathops, pitch
  - Strategy: Refactor in sub-commits, function by function
  - All callees must be safe first (Stages 2.1–2.3)
- [ ] `celt.rs`
  - Upstream C: `celt/celt.h`, `celt/celt.c`
  - Functions: `opus_strerror`, `opus_get_version_string`, utility functions
  - Risk: Low — mostly string/utility functions
- [ ] **Commit**: `refactor: make celt::bands safe` (may be multiple commits)
- [ ] **Commit**: `refactor: make celt::celt safe`

### Stage 2.5 — Encoder/Decoder

- [ ] `celt_decoder.rs`
  - Upstream C: `celt/celt_decoder.c`
  - Functions (8 unsafe): `celt_decode_with_ec`, `opus_custom_decoder_ctl_impl`, etc.
  - Risk: High — 1551 lines, manages decoder state, calls bands/mdct/entdec
  - Strategy: Convert state struct first, then decode pipeline
  - VarArgs in CTL → will be cleaned up in Phase 4 but prepare safe internals here
- [ ] `celt_encoder.rs`
  - Upstream C: `celt/celt_encoder.c`
  - Functions (20 unsafe): `celt_encode_with_ec`, `opus_custom_encoder_ctl_impl`, etc.
  - Risk: **Very High** — 3523 lines, largest file, calls everything
  - Strategy: Convert state struct, then encoding pipeline in stages
  - Sub-commits for logical groups of functions
- [ ] **Commit**: `refactor: make celt::celt_decoder safe`
- [ ] **Commit(s)**: `refactor: make celt::celt_encoder safe` (likely 3-5 commits)

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

- [ ] Zero `unsafe fn` declarations in `src/celt/` (except FFI boundaries if any)
- [ ] Zero `unsafe {}` blocks in `src/celt/`
- [ ] Every function has `/// Upstream C:` comment
- [ ] All tests pass (cargo test + vector tests)
- [ ] Clippy clean, formatted
- [ ] `externs::{memcpy,memmove,memset}` no longer called from celt/
