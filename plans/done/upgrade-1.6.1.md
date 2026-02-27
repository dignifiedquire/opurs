# Plan: Upgrade opurs from libopus previous baseline to 1.6.1

> Historical migration reference. Active execution and priority decisions are
> tracked in `remaining-plan.md`.

## Context

Upstream xiph/opus has released v1.6 and v1.6.1 since our previous baseline. The diff is **222 commits, ~19,500 insertions / ~4,700 deletions across 196 C/H files**. The dominant new feature is **QEXT (Quality Extension / Opus HD)** — a CELT extension layer adding 96 kHz support, higher-precision audio (32-bit samples), and bandwidth extension via a neural network (BBWENet). There are also significant standalone bug fixes, encoder quality improvements, DNN fixes, and the extensions API rework.

The upgrade should be done **incrementally in phases** to maintain bit-exactness at each step and avoid a massive single PR that's impossible to review.

## Phase 1: Update C Reference (`libopus-sys`) to 1.6.1

**Goal**: Get the C reference building at 1.6.1 so we can run comparison tests.

- Copy upstream 1.6.1 sources from `../libopus/opus` into `libopus-sys/opus/`
- Update `libopus-sys/build.rs`:
  - Add `bbwenet_data.c` to OSCE source list (new in lpcnet_sources.mk)
  - Add `celt/celt_qext.c`, `celt/mini_kfft.c` to source lists (new QEXT files)
  - Add `ENABLE_QEXT` config define (feature-gated)
  - Add `qext` feature to `libopus-sys/Cargo.toml`
  - Update `celt/arm/mathops_arm.h` header to headers list
  - Handle new MIPS header rename (`NSQ_del_dec_mipsr1.h` → `NSQ_mips.h`)
- Update version string in config.h to `1.6.1`
- Verify C reference builds and existing tests still pass (non-QEXT mode)

**Files**: `libopus-sys/build.rs`, `libopus-sys/Cargo.toml`, `libopus-sys/opus/` (vendored sources)

## Phase 2: Port Standalone Bug Fixes (No Output Change)

**Goal**: Port security and correctness fixes that don't change normal-operation output.

These can be verified against the existing IETF test vectors (228/228 must still pass):

1. **Integer overflow fix in CWRS** (`celt/cwrs.c` → `src/celt/cwrs.rs`)
   - Unsigned wrapping fixes in PVQ decode — security-relevant

2. **UBSan fixes** (left shift of negative values)
   - `celt/celt_lpc.c` → `src/celt/celt_lpc.rs`
   - `silk/NLSF_del_dec_quant.c` → `src/silk/NLSF_del_dec_quant.rs`
   - `silk/enc_API.c` → `src/silk/enc_API.rs`
   - (Many of these may be no-ops in Rust since Rust doesn't have UB for shifts, but verify)

3. **CELT decoder PLC fixes** (`celt/celt_decoder.c` → `src/celt/celt_decoder.rs`)
   - Post-filter on lost CELT frames
   - Overlap state after decoding
   - Post-filter during PLC transitions
   - LPC-based PLC overlap window
   - Adjust PLC C0 lower bound to let audio go to zero (`2f6660cd`)

4. **Float robustness fixes** (affect float build despite being near fixed-point work)
   - `celt_sqrt()` clamp in `compute_stereo_width` — MIN16(Q15ONE, celt_sqrt(...)) (`7b317ceb`)
   - Avoid NaNs in analysis code — downmix_float clamping + NaN check (`6f00d6d4`)
   - Post-downmix saturation guard for float only (`f310706a`)
   - Integer overflow fix in `compute_stereo_width()` — rewrite smoothing to avoid overflow on abrupt sign change (`930cde04`)
   - Saturate de-emphasis / increase SIG_SAT — float-path is no-op but port for parity (`5c13840b`)

5. **DRED fixes** (within `#[cfg(feature = "dred")]` / `#[cfg(feature = "deep-plc")]`)
   - Reduce DRED latency by one frame
   - Graceful degradation for unsupported sample rates
   - Fix decoder state after DRED usage
   - Don't code DRED for CBR

6. **DNN runtime fixes** (within DNN feature gates)
   - FARGAN overflow avoidance
   - Tanh clipping before computation
   - FARGAN scaling error fix
   - Various OSCE/LACE/NoLACE cleanup

**Verification**: `cargo nextest run -p opurs --cargo-profile=release` + `run_vectors2` (all 228 must pass unchanged)

## Phase 3: Port Encoder Quality Improvements

**Goal**: Port encoder behavior changes that improve quality but change output.

These WILL change encoded bitstreams, breaking bit-exactness with previous baseline C reference. After porting, the reference becomes the 1.6.1 C encoder. New test vectors will be needed.

1. **Fix reversed math in `compute_stereo_width`** (`src/opus_encoder.c` → `src/src/opus_encoder.rs`)
   - `c13d23f3` — `short_alpha` formula was inverted. opurs currently has the **old buggy formula**. HIGH priority.

2. **Fix anti-collapse RESYNTH mismatch for mono** (`celt/bands.c` → `src/celt/bands.rs`)
   - `fcd513e7` — `anti_collapse()` gains `encode` parameter; encoder no longer does cross-channel max for mono

3. **Fix low-bitrate encode/decode spreading mismatch** (`celt/celt_encoder.c` → `src/celt/celt_encoder.rs`)
   - `5ee25fe2` — Reset `spread_decision` to `SPREAD_NORMAL` when spreading is not signaled

4. **LPC-based tone detection** (`celt/celt_encoder.c` → `src/celt/celt_encoder.rs`)
   - `3b68a486` — New `tone_lpc()`, `tone_detect()` functions + `toneishness` state variable
   - `86101c0e` — Tone-based dynalloc boost in `dynalloc_analysis()`
   - `6f4f3e89` — Disable TF analysis on tones
   - `9394c7ce` — Fix int overflow in `tone_detect()` for stereo

5. **Tone detection robustness** (`src/analysis.c` → `src/src/analysis.rs`)
   - New `tone_freq` field in `AnalysisInfo`
   - Tone stability tracking

6. **Analysis energy computation** (`src/analysis.c` → `src/src/analysis.rs`)
   - Compute energy based on downmix
   - Prevent analysis in CELT-only mode

7. **CELT encoder improvements** (`celt/celt_encoder.c` → `src/celt/celt_encoder.rs`)
   - Transient detection threshold accounting for LM
   - `trim=-2` for music to force stereo
   - Anti-collapse end band changes
   - Only use odd pitches for pre-filter
   - Shrink encoder `delay_buffer` (`4b59c08a`)

8. **SILK encoder fixes**
   - Remove unnecessary variables (cleanup)
   - Compute SILK energy only when needed
   - Correct max bits passing
   - Include LBRR bits in bitrate count

9. **CELT bands cleanup** (`celt/bands.c` → `src/celt/bands.rs`)
   - Clarify `quant_all_bands()` arguments
   - Simplify bitrate<->bits conversion code (`f383ea82`)

**Verification**: Update C reference to 1.6.1, regenerate reference bitstreams, run `run_vectors2` with new vectors. Decode tests should match new C decoder; encode tests should match new C encoder.

## Phase 4: Port Extensions API Rework

**Goal**: Port the updated extensions parsing/generation.

The extensions API (`src/extensions.c`) has been significantly reworked with:
- Extension iterator (`opus_packet_extension_iterator`)
- Repeat These Extensions (RTE) — encoder/decoder
- Robustness improvements

This is a **prerequisite for QEXT** since QEXT data is carried as an extension payload.

Note: Extensions API was listed as ported in memory but no `extensions.rs` exists — need to port from scratch or verify if the functionality is inlined elsewhere.

**Files**: New `src/src/extensions.rs` (or equivalent), updates to `src/src/opus.rs`, `src/src/opus_encoder.rs`, `src/src/opus_decoder.rs`

## Phase 5: Port QEXT Core (Feature-Gated)

**Goal**: Add QEXT support behind a `qext` feature flag.

This is the largest phase. QEXT is cleanly gated with `#ifdef ENABLE_QEXT` in C; we'll use `#[cfg(feature = "qext")]` in Rust.

### 5a: Static Mode Tables for 96 kHz
- Add `src/celt/modes/data_96000.rs` with 96 kHz mode tables
- Update `src/celt/modes/mod.rs` to include 96 kHz mode
- ~4,500 lines of generated table data (mechanical conversion)

### 5b: mini_kfft
- New `src/celt/mini_kfft.rs` (~200 lines)
- Self-contained lightweight FFT for QEXT MDCT path

### 5c: celt_qext Core
- New `src/celt/celt_qext.rs` (~1,000 lines)
- QEXT encode/decode logic, band quantization extensions
- Uses mini_kfft for independent MDCT

### 5d: CELT Integration
- Update `src/celt/bands.rs` — extensive QEXT conditionals (~350 QEXT-specific lines)
- Update `src/celt/celt_encoder.rs` — QEXT encoding path (~260 QEXT lines)
- Update `src/celt/celt_decoder.rs` — QEXT decoding path (~254 QEXT lines)
- Update `src/celt/rate.rs` — QEXT bit allocation (~104 QEXT lines)
- Update `src/celt/vq.rs` — minor QEXT changes (~40 lines)
- Update `src/celt/mathops.rs` — `celt_sqrt32` and similar

### 5e: Opus Integration
- Update `src/src/opus_encoder.rs` — QEXT bitrate CTL, extension payload (~305 QEXT lines)
- Update `src/src/opus_decoder.rs` — QEXT payload parsing (~214 QEXT lines)
- New `opus_encode24`/`opus_decode24` API wrappers (32-bit sample precision)
- New CTLs: `OPUS_SET_QEXT`, `OPUS_GET_QEXT`

### 5f: BBWENet (Bandwidth Extension Network)
- New `src/dnn/bbwenet.rs` (~250 lines)
- New `src/dnn/bbwenet_data.rs` (weight data, size TBD)
- Follows existing DNN infrastructure patterns

**Estimated new/changed Rust**: ~10,000-13,000 lines total for QEXT

**Feature gating**: Add to `Cargo.toml`:
```toml
qext = []  # Quality Extension (Opus HD: 96kHz, 32-bit samples)
```

## Phase 6: Verification & CI

1. Update `libopus-sys` vendored sources to 1.6.1
2. Download/generate new test vectors for 1.6.1 (if IETF publishes them)
3. If no official vectors, generate our own from C 1.6.1 encoder
4. Update `run_vectors2` to support QEXT test configurations
5. Add QEXT feature combinations to CI matrix
6. Run full benchmark comparison to check for regressions
7. Update `BENCHMARKS.md` with any changes

## Execution Strategy

- **Do NOT attempt all phases in one PR** — each phase is a separate PR
- Phase 1 can start immediately (C reference update)
- Phase 2 can follow quickly (bug fixes, no output change)
- Phase 3 requires the C reference at 1.6.1 to verify
- Phase 4 is a prerequisite for Phase 5
- Phase 5 is the bulk of the work, can be broken into sub-PRs (5a-5f)
- Phase 6 runs throughout

## What We Skip (Same as previous baseline)

- Multistream/projection APIs (still deferred)
- MIPS/Xtensa optimizations (not relevant for our targets)
- C++ compatibility changes (not applicable)
- Build system changes (meson/cmake/autotools — we have our own build)

### Fixed-Point Accuracy Commits (~36 commits): Mostly Skipped

Of the ~36 "fixed-point accuracy" commits (DB_SHIFT=24, 32-bit celt_norm, MULT32_32_P31, sqrt/rsqrt improvements, FFT downshift, MDCT scaling, PVQ search accuracy, etc.), **~26 are entirely inside `#ifdef FIXED_POINT` guards** and have zero impact on our float-only build. These include:
- All `celt_sqrt32`/`celt_rsqrt_norm32` improvements (float uses `sqrtf()`/`1.f/sqrtf()`)
- DB_SHIFT=24 type widening (opus_val16→opus_val32 is float→float, no-op)
- `celt_exp2_db`/`celt_log2_db` macros (aliased to `celt_exp2`/`celt_log2` in float)
- MDCT/FFT scaling improvements (all gated on FIXED_POINT)
- Band energy/normalization accuracy (gated on FIXED_POINT)
- PVQ search accuracy improvements (gated on FIXED_POINT)
- `cos_norm2`/`atan`/`atan2p` fixed-point implementations (float versions unchanged)

The **~10 commits that DO affect the float build** are captured in Phases 2 and 3:
- Stereo width bug fix (`c13d23f3`) → Phase 3 item 1
- Stereo width overflow fix (`930cde04`) → Phase 2 item 4
- Anti-collapse mono mismatch (`fcd513e7`) → Phase 3 item 2
- Spreading mismatch (`5ee25fe2`) → Phase 3 item 3
- Tone detection (`3b68a486`, `86101c0e`, `6f4f3e89`, `9394c7ce`) → Phase 3 item 4
- Float robustness (NaN checks, sqrt clamp, SIG_SAT) → Phase 2 item 4

The `celt_log2`/`celt_exp2` float approximation rewrites (`e75503be`, `255f0130`) improve the C float approximation precision, but opurs already uses `libm` (`ln()`/`exp()`) which is more precise than either the old or new C approximation. **No action needed.**

## Risk Assessment

| Risk | Mitigation |
|------|------------|
| QEXT changes deeply interleaved in bands.c | Feature-gate carefully, port incrementally |
| No official 1.6.1 test vectors | Generate from C reference |
| Extensions API not yet in Rust | Port in Phase 4 before QEXT |
| Large scope (~13K new lines) | Phase incrementally, PR per phase |
| Bit-exactness regression | Each phase verified independently |
