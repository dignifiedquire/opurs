# SIMD Dispatch Divergence Tracker

Last updated: 2026-02-24

This file tracks known dispatch/RTCD divergences between upstream Opus C and this Rust port, plus alignment progress.

## Status Key

- `OPEN`: identified, not yet aligned
- `IN_PROGRESS`: active alignment work
- `FIXED`: aligned in Rust
- `INTENTIONAL`: divergence kept intentionally (with reason)

## Divergences

### D1 - SILK decoder arch initialization is scalar-only in Rust

- Severity: High
- Status: FIXED
- Upstream: `libopus-sys/opus/silk/init_decoder.c:53` sets `psDec->arch = opus_select_arch();`
- Rust: `src/silk/init_decoder.rs:47` and `src/silk/init_decoder.rs:91` set `Arch::Scalar`
- Impact: decoder-side SILK dispatch paths can miss SIMD selection

### D2 - SILK mono->stereo transition reinit inherits scalar arch in Rust

- Severity: High
- Status: FIXED
- Upstream: channel reinit uses upstream `silk_init_decoder()` with detected arch semantics
- Rust: `src/silk/dec_API.rs:123` calls `silk_init_decoder()` which currently initializes scalar arch
- Impact: channel 1 can remain scalar after transition

### D3 - DNN API arch-threading differs from upstream

- Severity: Medium
- Status: FIXED
- Upstream: DNN compute entrypoints are `(..., arch)` (`libopus-sys/opus/dnn/nnet.c:60`, `libopus-sys/opus/dnn/nnet.c:76`)
- Rust: core DNN compute entrypoints now thread `arch` (`src/dnn/nnet.rs`)
- Impact: semantic mismatch vs upstream RTCD API model

### D4 - AArch64 DNN DOTPROD dispatch model differs

- Severity: Medium
- Status: FIXED
- Upstream: separate DOTPROD source and RTCD mapping (`libopus-sys/opus/lpcnet_sources.mk:44`, `libopus-sys/opus/dnn/arm/arm_dnn_map.c:39`)
- Rust: DOTPROD runtime branch now maps to DOTPROD-specific kernels using native `sdot` dot-product instructions (`src/dnn/simd/mod.rs`, `src/dnn/simd/aarch64.rs`)
- Impact: aligned RTCD topology and instruction-level behavior for DNN int8 GEMV

### D5 - Standalone CELT custom decoder initializes scalar arch in Rust

- Severity: Medium
- Status: FIXED
- Upstream: CELT custom decoder init sets `st->arch = opus_select_arch();` (`libopus-sys/opus/celt/celt_decoder.c:265`)
- Rust: `src/celt/celt_decoder.rs:165` initializes `arch: Arch::Scalar`
- Impact: standalone CELT decoder paths diverge from upstream init behavior

### D6 - Build-time SIMD define/probe policy differs from upstream configure/meson

- Severity: Medium
- Status: FIXED
- Upstream: probes intrinsics then sets `MAY_HAVE`/`PRESUME` conditionally (`libopus-sys/opus/configure.ac:530`, `libopus-sys/opus/configure.ac:592`, `libopus-sys/opus/configure.ac:644`)
- Rust: `libopus-sys/build.rs` now probes compiler SIMD flag support and conditionally enables SIMD source groups and corresponding `OPUS_*_MAY_HAVE_*` defines (including aarch64 DOTPROD)
- Impact: aligns macro enablement with actual compiler capability and avoids over-advertising unsupported SIMD levels

### D7 - x86 AVX2 detection criteria implementation differs

- Severity: Low
- Status: FIXED
- Upstream: CPUID bit checks in C (`libopus-sys/opus/celt/x86/x86cpu.c:124`)
- Rust: x86 detection now mirrors upstream CPUID bit checks and ordering (`src/arch.rs`)
- Impact: edge-case platform behavior can differ

### D8 - Rust arch model omits non-aarch64 ARM levels

- Severity: Low
- Status: OPEN
- Upstream ARM arch ladder: V4/EDSP/MEDIA/NEON/DOTPROD (`libopus-sys/opus/celt/arm/armcpu.h:82`)
- Rust arch enum only models x86 levels and aarch64 Neon/DotProd (`src/arch.rs:18`)
- Impact: no full ARMv7 RTCD parity model

### D9 - FUZZING random arch downgrade behavior not mirrored

- Severity: Low
- Status: OPEN
- Upstream: random downgrade under `FUZZING` (`libopus-sys/opus/celt/arm/armcpu.c:318`, `libopus-sys/opus/celt/x86/x86cpu.c:177`)
- Rust: no equivalent in `src/arch.rs`
- Impact: fuzzing-specific behavior divergence

## Alignment Log

- 2026-02-24: Created baseline divergence inventory (D1-D9).
- 2026-02-24: Fixed D1/D2 by setting SILK decoder arch via `opus_select_arch()` in `silk_reset_decoder` (`src/silk/init_decoder.rs`).
- 2026-02-24: Fixed D5 by initializing CELT custom decoder arch via `opus_select_arch()` (`src/celt/celt_decoder.rs`).
- 2026-02-24: Fixed D3 by threading `Arch` through DNN nnet compute entrypoints and call graph (`src/dnn/nnet.rs` plus DNN callers).
- 2026-02-24: Fixed D4 by implementing true aarch64 DOTPROD kernels (`sdot` inline asm) for dense/sparse DNN int8 GEMV and wiring them to DOTPROD dispatch entries (`src/dnn/simd/aarch64.rs`, `src/dnn/simd/mod.rs`).
- 2026-02-24: Fixed D7 by replacing x86 SIMD level detection with upstream-equivalent CPUID bit checks in `opus_select_arch()` (`src/arch.rs`).
- 2026-02-24: Deep dispatch/cfg audit completed across upstream build+RTCD maps and Rust usage in runtime/tests/examples; no additional confirmed divergences beyond open D8/D9.
- 2026-02-24: Added aarch64 unit parity tests to assert DOTPROD dense/sparse DNN int8 GEMV matches scalar reference (`src/dnn/simd/aarch64.rs`).
- 2026-02-24: Fixed D6 by making `libopus-sys/build.rs` conditionally enable SIMD `MAY_HAVE` defines and SIMD source groups based on compiler flag probes (not blanket target-arch enables).
