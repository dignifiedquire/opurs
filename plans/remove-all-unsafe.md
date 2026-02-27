# Plan: Remove All Remaining Unsafe Code

**Goal**: Eliminate all remaining `unsafe` occurrences, reaching `#![forbid(unsafe_code)]`.

> Execution priority for this plan is tracked in `active-workstreams.md`.

**Current state (2026-02-13)**: Non-SIMD: 2 `unsafe` blocks across 1 file.
SIMD: 4 `unsafe fn` + 73 `unsafe` blocks across 2 files (expected for intrinsics).
Down from 47/9 at plan creation.

---

## Inventory (updated 2026-02-13)

### Non-SIMD code

| File | unsafe fn | unsafe {} | Total | Category | Status |
|------|-----------|-----------|-------|----------|--------|
| `src/celt/mdct.rs` | 0 | 2 | 2 | ndarray view casting | Open (no safe alternative) |
| ~~`src/celt/bands.rs`~~ | ~~0~~ | ~~16~~ | ~~0~~ | ~~Band processing sub-slicing~~ | **Done** ✓ |
| ~~`src/celt/celt_encoder.rs`~~ | ~~0~~ | ~~1~~ | ~~0~~ | ~~energy_mask raw pointer~~ | **Done** ✓ |
| ~~`src/celt/celt_decoder.rs`~~ | ~~0~~ | ~~1~~ | ~~0~~ | ~~Channel view splitting~~ | **Done** ✓ |
| ~~`src/src/opus_encoder.rs`~~ | ~~6~~ | ~~4~~ | ~~0~~ | ~~Encode path + downmix~~ | **Done** ✓ |
| ~~`src/src/opus_decoder.rs`~~ | ~~2~~ | ~~3~~ | ~~0~~ | ~~Decode path~~ | **Done** ✓ |
| ~~`src/silk/enc_API.rs`~~ | ~~0~~ | ~~2~~ | ~~0~~ | ~~addr_of_mut~~ | **Done** ✓ |
| ~~`src/util/nalgebra.rs`~~ | ~~0~~ | ~~2~~ | ~~0~~ | ~~nalgebra ViewStorage~~ | **Done** ✓ |
| ~~`src/src/opus.rs`~~ | ~~0~~ | ~~1~~ | ~~0~~ | ~~pointer offset_from~~ | **Done** ✓ |

### SIMD code (expected unsafe — intrinsics require it)

| File | unsafe fn | unsafe {} | Total | Category |
|------|-----------|-----------|-------|----------|
| `src/celt/simd/x86.rs` | 2 | 26 | 28 | x86 CELT SIMD intrinsics |
| `src/silk/simd/x86.rs` | 2 | 47 | 49 | x86 SILK SIMD intrinsics |

---

## Stages (ordered by dependency — bottom-up)

### Stage 1: bands.rs — ✅ DONE

All 16 unsafe blocks eliminated during later refactoring.

---

### Stage 2: celt_decoder.rs — ✅ DONE

Both unsafe blocks eliminated.

---

### Stage 3: celt_encoder.rs — ✅ DONE

energy_mask raw pointer replaced with safe alternative.

---

### Stage 4: mdct.rs — Safe ndarray view splitting (2 unsafe blocks)

**Problem**: `split_interleaving_opposite` and `_mut` create strided ndarray
views via `raw_view()` → `deref_into_view()`. The unsafe is needed because
ndarray doesn't provide a safe API for creating two strided views that are
provably disjoint.

**Strategy options**:
- (a) Use `ndarray::Zip` or manual iteration to avoid needing simultaneous
  views — process elements in-place with index arithmetic.
- (b) Keep the unsafe but wrap it in a well-documented `# Safety` comment
  and move it into a dedicated helper module with unit tests proving
  disjointness. Mark the containing functions as safe (the unsafe is
  internal implementation detail).
- (c) Use `split_at` on the underlying slice, then construct two separate
  ArrayViews from the halves with appropriate strides.

**Recommended**: Option (c) if feasible, otherwise (b) with thorough safety
documentation and test coverage. The interleaving pattern (even indices
forward, odd indices backward) guarantees disjointness.

**Risk**: Low — well-contained, clear disjointness invariant.

---

### Stage 5: silk/enc_API.rs — ✅ DONE

Resolved. Zero unsafe blocks remain in enc_API.rs.

---

### Stage 6: util/nalgebra.rs — ✅ DONE

Resolved. Zero unsafe blocks remain in util/nalgebra.rs.

---

### Stage 7: opus.rs — ✅ DONE

Resolved. Zero unsafe blocks remain in opus.rs.

---

### Stage 8: opus_decoder.rs — ✅ DONE

All unsafe fn and unsafe blocks eliminated. Safe typed API with methods.

---

### Stage 9: opus_encoder.rs — ✅ DONE

All unsafe fn and unsafe blocks eliminated. Safe typed API with methods.
CTL dispatch deleted, C-style API deleted, downmix made safe, encode path
converted to use slices.

---

### Stage 10: Implement safe public API wrappers

Implement the `Encoder`, `Decoder`, `Repacketizer`, `SoftClip` wrapper
structs defined in `phase4-integration-safety.md` stages 4.4-4.7.

These wrap the now-safe internal functions with typed parameters and
`Result` returns. This stage has no new unsafe code to eliminate — it's
about providing the final public API.

**Risk**: Low — wrapper code only.

---

### Stage 11: Final cleanup

1. ~~Remove deprecated `unsafe fn` C-API functions~~ — **Done** ✓ (all deleted)
2. ~~Remove `varargs.rs`~~ — **Done** ✓ (deleted in dig-safe: fe517bb)
3. Add `#![deny(unsafe_code)]` to `lib.rs` with `#[allow(unsafe_code)]` on
   SIMD modules (`src/celt/simd/`, `src/silk/simd/`) and `src/celt/mdct.rs`
   (Note: `#![forbid(unsafe_code)]` is not possible because SIMD intrinsics
   inherently require unsafe. Use `deny` + targeted `allow` instead.)
4. Verify: `cargo build`, `cargo test --all`, `cargo clippy`, vector tests

**Risk**: Low — attribute annotations and verification only.

---

## Execution Order & Dependencies

```
Stage 1 (bands.rs)          ── ✅ DONE
Stage 2 (celt_decoder.rs)   ── ✅ DONE
Stage 3 (celt_encoder.rs)   ── ✅ DONE
Stage 4 (mdct.rs)            ── Open (2 unsafe blocks, may be irreducible)
Stage 5 (enc_API.rs)         ── ✅ DONE
Stage 6 (nalgebra.rs)        ── ✅ DONE
Stage 7 (opus.rs)            ── ✅ DONE
Stage 8 (opus_decoder.rs)   ── ✅ DONE
Stage 9 (opus_encoder.rs)   ── ✅ DONE
                               │
Stage 10 (public API)        ──┤── Depends on Stages 8, 9
                               │
Stage 11 (forbid unsafe)    ───┘── Depends on all above
```

---

## Estimated Remaining Commit Count

| Stage | Commits | Description |
|-------|---------|-------------|
| ~~1~~ | ~~0~~ | ~~bands.rs — done~~ |
| ~~2~~ | ~~0~~ | ~~celt_decoder — done~~ |
| ~~3~~ | ~~0~~ | ~~celt_encoder — done~~ |
| 4 | 0-1 | mdct view splitting (may be irreducible) |
| ~~5~~ | ~~0~~ | ~~enc_API — done~~ |
| ~~6~~ | ~~0~~ | ~~nalgebra — done~~ |
| ~~7~~ | ~~0~~ | ~~opus.rs — done~~ |
| ~~8~~ | ~~0~~ | ~~decoder — done~~ |
| ~~9~~ | ~~0~~ | ~~encoder — done~~ |
| 10 | 2-3 | public API wrappers |
| 11 | 1 | deny(unsafe_code) with allow on SIMD modules |
| **Total** | **~3-5** | |

---

## Testing Strategy

After every stage:
1. `cargo build` — must compile
2. `cargo test --all` — all unit + integration tests pass
3. `cargo clippy` — no warnings
4. Vector tests (for stages touching encoder/decoder/codec internals):
   `cargo run --release --features tools --example run_vectors2 -- opus_newvectors`

---

## What We Start With

I recommend starting with **Stages 1-7 in parallel** (the independent
bottom-up work on CELT/SILK/util), then moving to **Stage 8** (decoder),
then **Stage 9** (encoder — the hardest), then **Stages 10-11** (API + cleanup).

The single hardest piece is **Stage 9b** (making `opus_encode_native` safe) —
a 1300-line function with pervasive raw pointer usage. This will likely need
to be broken into multiple sub-commits.
