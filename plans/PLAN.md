# Master Plan: Safe Idiomatic Rust libopus

**Goal**: Fully safe, idiomatic Rust implementation of libopus 1.3.1 that is
bit-exact with the C reference, has a complete ported test suite, and matches
C performance.

**Invariant**: Every commit compiles, passes `cargo test --all`, passes
`cargo clippy`, and is `cargo fmt` clean. Vector tests pass for any commit
touching codec internals.

---

## Phases

| # | Phase | Plan File | Status |
|---|-------|-----------|--------|
| 1 | [Test Suite Expansion](#phase-1) | [phase1-test-expansion.md](phase1-test-expansion.md) | Complete (62 tests) |
| 2 | [CELT Safety](#phase-2) | [phase2-celt-safety.md](phase2-celt-safety.md) | ~Complete (0 unsafe fn, 2 residual unsafe blocks in mdct.rs) |
| 3 | [SILK Safety](#phase-3) | [phase3-silk-safety.md](phase3-silk-safety.md) | Complete (0 unsafe fn, 0 unsafe blocks) |
| 4 | [Integration Layer Safety](#phase-4) | [phase4-integration-safety.md](phase4-integration-safety.md) | In progress (~60% done) |
| 5 | [Performance Parity](#phase-5) | [phase5-performance.md](phase5-performance.md) | In progress (SIMD ported, benchmarks not started) |

---

## Phase 1 — Test Suite Expansion {#phase-1}

**Why first**: Safety refactoring requires a comprehensive test net. Every
algorithm change must be caught. The current test suite has significant gaps:
unit tests are raw c2rust dumps with private copies of library functions,
several upstream tests are missing entirely, and the integration tests are
monolithic mega-functions full of C idioms (malloc/free, static mut, fprintf,
_test_failed) that make failures hard to debug.

**Two goals in one phase**: Port all missing tests AND make every test
idiomatic Rust — granular `#[test]` functions, `assert!` with context
messages, `Vec` instead of malloc, explicit RNG state instead of globals,
named constants instead of magic numbers.

**Deliverables**:
- Shared test infrastructure: `TestRng`, seed management, factory helpers
- All 8 CELT unit tests rewritten to use library imports, with `#[test]` wiring
- SILK `test_unit_LPC_inv_pred_gain` rewritten the same way
- Integration tests split into 25+ granular `#[test]` functions
- Zero `malloc`/`free`, `static mut`, `_test_failed()`, or `extern "C"` in tests
- Every `assert!` has a context message for easy debugging
- `test_opus_projection` explicitly deferred (multistream not ported)
- `opus_decode_fuzzer` ported as a cargo-fuzz target

See [phase1-test-expansion.md](phase1-test-expansion.md) for details.

---

## Phase 2 — CELT Safety {#phase-2}

**Approach**: Bottom-up, leaf modules first. CELT has a clear dependency DAG:
math primitives → entropy coding → mid-level DSP → bands → encoder/decoder.

**Scope**: All 18 modules under `src/celt/`.

**Key challenges**:
- `bands.rs` (2146 lines, 24 unsafe fn) is a central hub called by 8+ modules
- `celt_encoder.rs` (3523 lines, 20 unsafe fn) is the integration point
- Heavy pointer arithmetic in pitch analysis and FFT

See [phase2-celt-safety.md](phase2-celt-safety.md) for details.

---

## Phase 3 — SILK Safety {#phase-3}

**Approach**: Bottom-up, largely parallel with Phase 2 since SILK and CELT
have independent dependency trees. Many SILK leaf modules are already safe on
`master` branch.

**Scope**: All 65+ modules under `src/silk/` and `src/silk/float/`.

**Key challenges**:
- `NSQ.rs` / `NSQ_del_dec.rs` (1065+572 lines) — core quantizers
- `enc_API.rs` (877 lines) — SILK encoder hub
- `float/encode_frame_FLP.rs` (571 lines) — float encoding integration

See [phase3-silk-safety.md](phase3-silk-safety.md) for details.

---

## Phase 4 — Integration Layer Safety {#phase-4}

**Approach**: Top-down from public API. Depends on Phases 2–3 completing the
sub-codec safety work. This is where `externs.rs` (malloc/free/memcpy) gets
eliminated entirely.

**Scope**: `src/src/opus_encoder.rs`, `src/src/opus_decoder.rs`,
`src/src/opus.rs`, `src/src/analysis.rs` signature cleanup,
`src/externs.rs` removal, `src/varargs.rs` replacement.

**Key challenges**:
- `opus_encoder.rs` (3034 lines, 30 unsafe fn) — the single hardest file
- Replacing malloc/free with Box/Vec for encoder/decoder state
- VarArgs → proper Rust enum-based CTL API

See [phase4-integration-safety.md](phase4-integration-safety.md) for details.

---

## Phase 5 — Performance Parity {#phase-5}

**Approach**: Profile-guided. Establish benchmarks first, then optimize
hot paths identified by profiling. Target: match C+ASM performance (currently
~20% gap).

**Scope**: SIMD intrinsics for FFT/pitch/MDCT, inlining annotations,
allocation optimization, benchmark suite.

See [phase5-performance.md](phase5-performance.md) for details.

---

## Commit Conventions

All commits must:
1. Compile cleanly (`cargo build`)
2. Pass all tests (`cargo test --all`)
3. Be clippy-clean (`cargo clippy --all -- -D warnings`)
4. Be formatted (`cargo fmt --check`)
5. Have concise descriptive messages: `refactor: make celt::mathops safe`
6. Update the relevant plan file's progress checkboxes before committing

**Commit message prefixes**:
- `test:` — test additions/fixes
- `refactor:` — safety refactoring (no behavior change)
- `fix:` — bug fixes
- `perf:` — performance improvements
- `chore:` — formatting, CI, docs, plan updates

---

## C Reference Cross-Linking

Every function that has a corresponding C implementation must have a comment
linking to the upstream source:

```rust
/// Compute the square root of a 32-bit integer.
///
/// Upstream C: celt/mathops.h:isqrt32
fn isqrt32(val: u32) -> u32 {
```

Format: `/// Upstream C: <path-relative-to-opus-root>:<function-name>`

This enables easy comparison with the reference when debugging bit-exactness
issues.

---

## Progress Tracking

Each phase plan file contains a checklist. Mark items `[x]` as they are
completed. Update the status column in the table above when a phase
transitions between Not started / In progress / Complete.

## Supplemental Plans

- [Parity Test Expansion (Fail-First)](parity-test-fail-first.md) — maps
  `diff_review.md` findings to targeted red→green parity tests and CI gates.
