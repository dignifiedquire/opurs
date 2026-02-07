# Phase 1: Test Suite Expansion

**Goal**: Complete test coverage that catches any behavioral divergence from
the C reference. Every upstream C test must have a Rust equivalent, all unit
tests must use the library's actual code (not private copies), and tests must
be idiomatic Rust — granular, debuggable, and free of C patterns.

**Prerequisites**: None — this is the first phase.

---

## Current State

### Integration tests (tests/)
| File | #[test] count | Upstream equivalent | Notes |
|------|--------------|-------------------|-------|
| `opus_api.rs` | 15 | `test_opus_api.c` (1904 lines) | Ported, could be more granular |
| `opus_decode.rs` | 1 | `test_opus_decode.c` (463 lines) | Single mega-test, needs splitting |
| `opus_encode/main.rs` | 1 | `test_opus_encode.c` (703 lines) | Single mega-test, needs splitting |
| `opus_encode/opus_encode_regressions.rs` | 0 | `opus_encode_regressions.c` (1035 lines) | Called from main, not separate tests |
| `opus_padding.rs` | 1 | `test_opus_padding.c` (93 lines) | Ported |
| — | — | `test_opus_projection.c` (394 lines) | **NOT PORTED** (multistream/ambisonics) |
| — | — | `opus_decode_fuzzer.c` (126 lines) | **NOT PORTED** |

### CELT unit tests (src/celt/tests/)
All 8 files are raw c2rust transpilations. Each file:
- Has its own private copy of entropy coder, math ops, etc.
- Uses `extern "C"` for libc functions (fprintf, rand, malloc, free)
- Has `unsafe fn main_0()` + `pub fn main()` wrapper
- Has **zero** `#[test]` attributes
- Is **commented out** in `src/lib.rs` module tree

| File | Lines | Upstream equivalent |
|------|-------|-------------------|
| `test_unit_cwrs32.rs` | 2683 | `celt/tests/test_unit_cwrs32.c` (161 lines) |
| `test_unit_dft.rs` | — | `celt/tests/test_unit_dft.c` (179 lines) |
| `test_unit_entropy.rs` | 1423 | `celt/tests/test_unit_entropy.c` (383 lines) |
| `test_unit_laplace.rs` | 931 | `celt/tests/test_unit_laplace.c` (93 lines) |
| `test_unit_mathops.rs` | — | `celt/tests/test_unit_mathops.c` (266 lines) |
| `test_unit_mdct.rs` | 415 | `celt/tests/test_unit_mdct.c` (227 lines) |
| `test_unit_rotation.rs` | — | `celt/tests/test_unit_rotation.c` (86 lines) |
| `test_unit_types.rs` | — | `celt/tests/test_unit_types.c` (50 lines) |

Note: Rust files are 5-15x larger than C because they inline private copies
of all called functions instead of importing from the library.

### SILK unit tests (src/silk/tests/)
| File | Upstream equivalent |
|------|-------------------|
| `test_unit_LPC_inv_pred_gain.rs` | `silk/tests/test_unit_LPC_inv_pred_gain.c` (129 lines) |

Same problems as CELT unit tests — private function copies, no `#[test]`.

---

## Anti-Patterns to Eliminate

The current tests are mechanically transpiled C. Every rewritten test must
fix these patterns:

### Pattern table: C-ism → Idiomatic Rust

| C-ism | Where | Replacement |
|-------|-------|-------------|
| `malloc(size) as *mut T` / `free(ptr)` | All tests | `Vec<T>` or `Box<[T]>` — RAII cleanup |
| `fprintf(stderr, "fmt %d", val)` | Unit tests | `eprintln!("fmt {val}")` or `assert!()` with message |
| `_test_failed(file, line)` | opus_decode, opus_encode | `assert!(cond, "context: {details}")` |
| `if err != 0 { fail() }` | All tests | `assert_eq!(err, OPUS_OK, "context")` |
| `unsafe fn main_0() -> i32` | Unit tests | `#[test] fn test_name()` |
| Return `i32` error codes | All tests | `#[test]` with `assert!` / `panic!` |
| `static mut Rz/Rw` globals | opus_decode, opus_encode | `struct TestRng` passed by `&mut` |
| `extern "C" { fn rand(); }` | Unit tests | Deterministic `TestRng` from test_common |
| `extern "C" { fn fprintf(); }` | Unit tests | Remove entirely — use `assert!`/`eprintln!` |
| Private copies of library fns | Unit tests | `use unsafe_libopus::...` imports |
| One 500-line mega-test | opus_encode, opus_decode | Multiple focused `#[test]` fns |
| `b"path\0" as *const i8` file refs | All tests | `file!()` and `line!()` macros |
| Magic numbers `960, 2880, 48000` | All tests | Named constants with comments |
| `DUMMY_ARGS` hardcoded seed | opus_encode | `get_test_seed()` from env or random |
| `memcpy(dst, src, size)` | All tests | `dst.copy_from_slice(src)` or `.clone()` |
| `memcmp(a, b, size) != 0` | opus_decode | `assert_eq!(a, b)` on slices |

### Debugging and reproducibility

Every test must support easy debugging when it fails:

1. **Seed reproduction**: Print the seed at test start so failures can be
   reproduced with `TEST_SEED=<n> cargo test <test_name>`
2. **Rich assertion messages**: Every `assert!`/`assert_eq!` includes the
   parameter context (bitrate, sample rate, complexity, frame size, etc.)
3. **Granular test names**: Test name should tell you what broke without
   reading the code: `test_decode_48khz_stereo_code0`, not `test_opus_decode`
4. **No hidden state**: No `static mut`. RNG state passed explicitly.
5. **Independent tests**: Each `#[test]` creates its own encoder/decoder
   state. No shared setup that can cause ordering-dependent failures.

---

## Tasks

### 1.1 — Build shared test infrastructure

Create `tests/test_common/mod.rs` as the idiomatic replacement for all C
test infrastructure.

- [ ] **Deterministic RNG** — portable, seedable, matching upstream behavior:
  ```rust
  /// Marsaglia MWC RNG — matches upstream test_opus_common.h fast_rand()
  pub struct TestRng {
      rz: u32,
      rw: u32,
  }

  impl TestRng {
      pub fn new(seed: u32) -> Self { ... }
      /// Matches upstream fast_rand() output exactly
      pub fn next(&mut self) -> u32 { ... }
      /// Signed random, matches upstream int cast behavior
      pub fn next_i32(&mut self) -> i32 { ... }
  }
  ```
  Must produce identical sequences to the C `fast_rand()` so ported tests
  generate the same inputs and the same expected outputs.

- [ ] **Seed management** — environment-based, always printed:
  ```rust
  /// Get seed from TEST_SEED env var, or generate random + print it.
  pub fn get_test_seed() -> u32 { ... }
  ```

- [ ] **De Bruijn sequence generator** (from `test_opus_common.h`):
  ```rust
  pub fn debruijn2(n: i32, t: &[u8], output: &mut Vec<u8>) { ... }
  ```

- [ ] **Encoder/decoder factory helpers** — safe wrappers that panic on failure:
  ```rust
  pub fn create_encoder(sample_rate: i32, channels: i32, app: i32) -> *mut OpusEncoder {
      let mut err = 0;
      let enc = unsafe { opus_encoder_create(sample_rate, channels, app, &mut err) };
      assert_eq!(err, OPUS_OK, "encoder creation failed: sr={sample_rate}, ch={channels}");
      assert!(!enc.is_null());
      enc
  }
  pub fn create_decoder(sample_rate: i32, channels: i32) -> *mut OpusDecoder { ... }
  ```
  These stay unsafe internally until Phase 4 makes the API safe, but they
  centralize error handling and avoid repetition across tests.

- [ ] **Named constants for common test parameters**:
  ```rust
  pub const SAMPLE_RATES: &[i32] = &[8000, 12000, 16000, 24000, 48000];
  pub const CHANNELS: &[i32] = &[1, 2];
  pub const FRAME_SIZES_48K: &[i32] = &[
      120,   // 2.5ms
      240,   // 5ms
      480,   // 10ms
      960,   // 20ms
      1920,  // 40ms
      2880,  // 60ms
  ];
  pub const BITRATES: &[i32] = &[6000, 12000, 16000, 32000, 48000, 64000, 96000, 510000];
  pub const MAX_FRAME_SAMPLES: usize = 48000 * 60 / 1000; // 60ms at 48kHz
  pub const MAX_PACKET_SIZE: usize = 1500; // conservative MTU
  ```

- [ ] Refactor `opus_api.rs` to use shared module
- [ ] Refactor `opus_decode.rs` to use shared module
- [ ] Refactor `opus_encode/` to use shared module
- [ ] **Commit**: `test: extract shared test utilities to test_common module`

### 1.2 — Rewrite CELT unit tests

Each test must:
- Import functions from the library (not private copies)
- Use `#[test]` attributes with descriptive names
- Use `assert!`/`assert_eq!` with context messages
- Use `TestRng` from test_common (not libc `rand`)
- Use `Vec`/slices (not `malloc`/`free`)
- Have zero `unsafe` in test logic where possible (unsafe only for calling
  library functions that are still unsafe)

Approach: Read the upstream C test, read the c2rust dump, write a clean Rust
test that tests the same thing using library imports.

- [ ] `test_unit_types.rs` → `tests/celt_types.rs`
  - Upstream C: `celt/tests/test_unit_types.c`
  - Simplest test, good starting point
  - Tests: type sizes and bit operations
  - Split into: `test_opus_int16_size`, `test_opus_int32_size`
- [ ] `test_unit_mathops.rs` → `tests/celt_mathops.rs`
  - Upstream C: `celt/tests/test_unit_mathops.c`
  - Split into 8 `#[test]` functions:
    - `test_celt_div` — reciprocal computation (1..327670)
    - `test_celt_sqrt` — square root accuracy (1..1B)
    - `test_bitexact_cos` — cosine table consistency
    - `test_bitexact_log2tan` — log2/tan computation
    - `test_celt_log2` — float log2
    - `test_celt_exp2` — float exp2
    - `test_exp2_log2_roundtrip` — exp2(log2(x)) ≈ x
    - `test_ilog2` — integer log2
  - Each test: self-contained, clear assertion messages with the input that
    failed (e.g., `assert!(err < threshold, "sqrt({n}) error {err} exceeds {threshold}")`)
- [ ] `test_unit_entropy.rs` → `tests/celt_entropy.rs`
  - Upstream C: `celt/tests/test_unit_entropy.c`
  - Import `entenc`/`entdec` from library
  - Split into:
    - `test_entropy_raw_bits` — raw bit encoding roundtrip
    - `test_entropy_random_streams` — random symbol coding
    - `test_entropy_tell_accuracy` — bit accounting
    - `test_entropy_patch_initial_bits`
    - `test_entropy_buffer_overflow`
  - Replace `rand()`/`srand()` with `TestRng`
  - Replace `malloc`/`free` with `Vec<u8>` buffers
- [ ] `test_unit_laplace.rs` → `tests/celt_laplace.rs`
  - Upstream C: `celt/tests/test_unit_laplace.c`
  - Import `laplace`, `entenc`, `entdec` from library
  - Split: `test_laplace_roundtrip`, `test_laplace_specific_values`
  - Replace `rand()` with `TestRng`
- [ ] `test_unit_cwrs32.rs` → `tests/celt_cwrs32.rs`
  - Upstream C: `celt/tests/test_unit_cwrs32.c`
  - Import `cwrs`, `entenc`, `entdec` from library
  - Split: `test_cwrs_encode_decode`, `test_cwrs_combination_counts`
  - Replace `fprintf` reporting with `assert_eq!` messages
- [ ] `test_unit_rotation.rs` → `tests/celt_rotation.rs`
  - Upstream C: `celt/tests/test_unit_rotation.c`
  - Tests: vector rotation with SNR threshold
  - Split by vector size: `test_rotation_n15`, `test_rotation_n23`, etc.
  - SNR check with message: `assert!(snr > 60.0, "rotation SNR {snr:.1}dB < 60dB for N={n}, K={k}")`
- [ ] `test_unit_mdct.rs` → `tests/celt_mdct.rs`
  - Upstream C: `celt/tests/test_unit_mdct.c`
  - Import `mdct`, `modes` from library
  - Split: `test_mdct_forward_<size>`, `test_mdct_backward_<size>` per transform size
  - SNR assertions with context
- [ ] `test_unit_dft.rs` → `tests/celt_dft.rs`
  - Upstream C: `celt/tests/test_unit_dft.c`
  - Import `kiss_fft`, `modes` from library
  - Split: `test_fft_forward_<size>`, `test_fft_backward_<size>` per transform size
- [ ] Remove old `src/celt/tests/` directory and commented-out module in `lib.rs`
- [ ] **Commit per test**: `test: rewrite celt::<name> unit test as idiomatic Rust`
- [ ] **Final commit**: `test: remove old c2rust celt unit test stubs`

### 1.3 — Rewrite SILK unit test

- [ ] `test_unit_LPC_inv_pred_gain.rs` → `tests/silk_lpc_inv_pred_gain.rs`
  - Upstream C: `silk/tests/test_unit_LPC_inv_pred_gain.c`
  - Tests: LPC filter stability via impulse response simulation (10,000 random filters)
  - Import `silk_LPC_inv_pred_gain` from library
  - Replace `rand()`/`srand()` with `TestRng` (must produce same filter coefficients)
  - Split: `test_lpc_stability_order_<n>` or parameterized test
  - Rich assertion: `assert!(stable, "filter order={order}, shift={shift}, coeffs={coeffs:?}")`
- [ ] Remove old `src/silk/tests/` directory
- [ ] **Commit**: `test: rewrite silk LPC_inv_pred_gain as idiomatic Rust`

### 1.4 — Make integration tests idiomatic

The integration tests (opus_api, opus_decode, opus_encode) call the library
through its current (still-unsafe) API, so they'll keep `unsafe` blocks for
now. The focus here is structural cleanup:

#### 1.4.1 — opus_decode.rs
- [ ] Replace `static mut Rz/Rw` with `TestRng` passed to each helper
- [ ] Replace `_test_failed()` with `assert!`/`assert_eq!` with context
- [ ] Replace `malloc`/`free` with `Vec`
- [ ] Replace `memcpy`/`memcmp` with slice operations
- [ ] Split `test_opus_decode` into:
  - `test_decoder_code0_48khz_stereo` — most common config
  - `test_decoder_code0_8khz_mono` — narrowband edge case
  - `test_decoder_code0_fuzzing` — random packet robustness
  - `test_soft_clip_basic` — normal clipping behavior
  - `test_soft_clip_edge_cases` — boundary conditions
- [ ] Add seed printing at test start for reproducibility
- [ ] **Commit**: `test: make opus_decode tests idiomatic with granular assertions`

#### 1.4.2 — opus_encode
- [ ] Replace `static mut` RNG state with `TestRng`
- [ ] Replace `DUMMY_ARGS` seed handling with `get_test_seed()`
- [ ] Replace all `malloc`/`free` with `Vec`
- [ ] Replace `fail()` macro calls with `assert!`/`assert_eq!` with context
- [ ] Split `test_opus_encode` into:
  - `test_encode_basic_roundtrip` — encode then decode, verify samples
  - `test_encode_multimode` — LP/Hybrid/MDCT mode switching
  - `test_encode_bitrate_range` — 6k to 510k bps
  - `test_encode_frame_sizes` — all frame durations (2.5ms to 60ms)
  - `test_encode_stereo_width` — stereo processing
  - `test_encode_fec` — forward error correction
  - `test_encode_dtx` — discontinuous transmission
  - `test_encode_final_range_determinism` — enc_final_range == dec_final_range
  - `test_encode_padding` — packet padding/unpadding during encode
  - `test_fuzz_encoder_settings` — random parameter combinations
- [ ] Move regression tests to individual `#[test]` functions:
  - `test_regression_ec_enc_shrink` (from opus_encode_regressions.rs)
  - `test_regression_ec_enc_shrink2`
  - `test_regression_silk_gain`
- [ ] Add context to every assertion:
  ```rust
  assert_eq!(
      out_samples, frame_size,
      "decode produced {out_samples} samples, expected {frame_size} \
       (bitrate={bitrate}, complexity={complexity}, mode={mode})"
  );
  ```
- [ ] **Commit**: `test: make opus_encode tests idiomatic with granular assertions`
- [ ] **Commit**: `test: split encode regressions into individual #[test] functions`

#### 1.4.3 — opus_api.rs
This is already the best-structured test file (15 `#[test]` fns). Cleanup:
- [ ] Replace remaining `malloc`/`free` patterns with `Vec`
- [ ] Replace `fail()` macro with `assert!` where possible
- [ ] Improve assertion messages (currently generic "line N")
- [ ] **Commit**: `test: clean up opus_api test idioms`

#### 1.4.4 — opus_padding.rs
- [ ] Minor cleanup: improve assertion messages
- [ ] **Commit**: `test: clean up opus_padding test`

### 1.5 — Port missing upstream tests

- [ ] Evaluate `test_opus_projection.c` (394 lines)
  - This tests multistream/ambisonics which is not implemented in the Rust port
  - Decision: **Defer** — document as out-of-scope until multistream is ported
  - Add `plans/deferred-multistream.md` noting this gap
- [ ] Port `opus_decode_fuzzer.c` as cargo-fuzz target
  - Add `fuzz/` directory with `cargo-fuzz` setup
  - Port the libFuzzer harness logic
  - Document how to run: `cargo +nightly fuzz run decode_fuzzer`
  - **Commit**: `test: add cargo-fuzz decode fuzzer from upstream`

### 1.6 — Verify test suite completeness

- [ ] Run `cargo test --all` — all tests pass
- [ ] Run `cargo test --all --release` — all tests pass
- [ ] Run vector tests — all pass
- [ ] Run clippy — clean
- [ ] `cargo fmt --check` passes
- [ ] Compare test function count against upstream:

  | Upstream test | Upstream fn count | Rust #[test] count | Status |
  |---------------|------------------|--------------------|--------|
  | test_opus_api.c | 5 main fns | 15+ | OK |
  | test_opus_decode.c | 2 fns | 5+ | OK |
  | test_opus_encode.c | 3 fns | 10+ | OK |
  | opus_encode_regressions.c | 3 fns | 3 | OK |
  | test_opus_padding.c | 1 fn | 1 | OK |
  | test_unit_types.c | 1 fn | 2 | OK |
  | test_unit_mathops.c | 8 fns | 8 | OK |
  | test_unit_entropy.c | 1 fn (big) | 5+ | OK |
  | test_unit_laplace.c | 1 fn | 2 | OK |
  | test_unit_cwrs32.c | 1 fn | 2 | OK |
  | test_unit_rotation.c | 1 fn | 4 | OK |
  | test_unit_mdct.c | 1 fn | 6+ | OK |
  | test_unit_dft.c | 1 fn | 6+ | OK |
  | test_unit_LPC_inv_pred_gain.c | 1 fn | 1+ | OK |

- [ ] Document any remaining gaps in this file
- [ ] **Commit**: `test: verify complete test suite, update phase1 plan`

---

## Visibility Requirements

For unit tests to import library functions, some currently-private functions
may need their visibility bumped to `pub(crate)`. Track these changes:

| Function | Module | Current visibility | Needed |
|----------|--------|--------------------|--------|
| (to be filled during implementation) | | | |

Each visibility change should be minimal and documented. Prefer `pub(crate)`
over `pub` for internal test access.

---

## Test RNG Compatibility Note

The `TestRng` must produce **identical output** to the upstream
`fast_rand()` for the same seed, because the integration tests (opus_decode,
opus_encode) generate specific packet data and encoder parameters from the
RNG. Changing the RNG output would change what the tests exercise, breaking
the equivalence with the upstream test suite.

For the CELT/SILK unit tests being rewritten from scratch, the RNG output
doesn't need to match upstream — only the test *logic* needs to match.

---

## Definition of Done

- [ ] All 8 CELT unit tests rewritten with `#[test]`, importing from library
- [ ] SILK unit test rewritten with `#[test]`, importing from library
- [ ] Old `src/celt/tests/` and `src/silk/tests/` removed
- [ ] Integration tests split into 25+ granular `#[test]` functions
- [ ] Zero `_test_failed()` calls — all use `assert!`/`assert_eq!`
- [ ] Zero `malloc`/`free` in test code — all use `Vec`/`Box`
- [ ] Zero `static mut` — all RNG state passed explicitly
- [ ] Zero `extern "C"` libc imports in test code
- [ ] Every `assert!` has a context message describing what failed and why
- [ ] Seed printed at start of randomized tests for reproducibility
- [ ] Shared test utilities in `tests/test_common/`
- [ ] Cargo-fuzz target for decode fuzzing
- [ ] `test_opus_projection` explicitly deferred with documentation
- [ ] All tests pass in both dev and release profiles
- [ ] Zero clippy warnings, formatted
