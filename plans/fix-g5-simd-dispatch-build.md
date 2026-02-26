# Plan: Group 5 SIMD Dispatch And Build Parity

## Goal
Resolve remaining SIMD dispatch semantics and build-flag parity gaps after D1-D9 alignment.

## Findings IDs
Open: `194`
Closed in this group: `107,202,203,204,205,212,213,230,231,232,233,234,235,237`

## Scope
- Remaining arch-threading and arch-index semantics differences.
- x86 tier coverage and arch-forced dispatch parity behavior.
- Build-flag parity (`-mavx` companion flags) and feature-tier consistency.

## Execution Order
1. Triage by risk: behavior-affecting dispatch semantics first, then build-flag parity.
2. Add scalar-vs-simd-vs-C differential tests for each remaining dispatch hotspot.
3. Fix x86/aarch64 tier control and any arch-parameter plumbing gaps.
4. Align remaining AVX2/CELT/DNN build flags to upstream equivalents.

## Verification
- `CARGO_TARGET_DIR=target-local cargo nextest run -p opurs --cargo-profile=release`
- `CARGO_TARGET_DIR=target-local cargo test -p opurs --test simd_comparison --release`
- `CARGO_TARGET_DIR=target-local cargo run --release --features tools --example run_vectors2 -- --suite classic --matrix full --mode parity opus_newvectors`

## Definition Of Done
- Remaining SIMD IDs are closed or explicitly justified as intentional with tests proving no parity drift.

## Progress
- 2026-02-24: Aligned x86 AVX2 compile-flag bundles in `libopus-sys/build.rs` to upstream (`-mavx -mfma -mavx2`) for `CELT_SOURCES_AVX2`, `SILK_SOURCES_AVX2`, `SILK_SOURCES_FLOAT_AVX2`, and `DNN_SOURCES_AVX2`.
- 2026-02-24: Updated AVX2 capability probe to require the same full flag bundle (`-mavx -mfma -mavx2`) before enabling x86 AVX2 MAY_HAVE paths.
- 2026-02-24: This closes build-flag parity items `232` and `237`.
- 2026-02-26: Closed additional dispatch-semantic gaps:
  - `107`: decoder now threads runtime arch into internal soft-clip implementation (`opus_pcm_soft_clip_impl(..., arch)`).
  - `231`: removed aarch64 NEON override for `silk_inner_product_FLP`; SIMD override now matches upstream x86-AVX2-only behavior.
- 2026-02-26: Closed additional arch-tier dispatch parity items with deterministic C-vs-Rust coverage:
  - `204`: `celt_pitch_xcorr` now keeps scalar for non-AVX2 x86 tiers (AVX2-only override parity).
  - `205`: `op_pvq_search` uses threaded `arch` tier for SSE2/scalar selection.
  - `212`, `230`, `235`: DNN/CELT/SILK dispatch now honors threaded `arch` control semantics, including upstream aarch64 low-tier NEON behavior for DNN; added forced-tier regression coverage in `tests/osce_nndsp.rs:test_compute_linear_int8_arch_tiers_match_c`.
- 2026-02-26: Re-verified forced arch-tier DNN parity under `nextest` on current head:
  - Command: `CARGO_TARGET_DIR=target-local cargo nextest run --release --features tools-dnn -E 'binary(osce_nndsp) and test(/test_compute_linear_int8_arch_tiers_match_c|test_compute_conv2d_3x3/)'`
  - Result: both tests passed (`2 passed, 17 skipped`) including aarch64 `dotprod` tier comparison vs C harness.
- 2026-02-26: Added x86 SSE2 dispatch coverage for DNN float kernels:
  - `src/dnn/simd/x86.rs`: new `sgemv_sse2` and `sparse_sgemv8x4_sse2` implementations.
  - `src/dnn/simd/mod.rs`: `sgemv` and `sparse_sgemv8x4` now dispatch to SSE2 on non-AVX2 x86 tiers.
  - Added x86 unit tests in `src/dnn/simd/x86.rs` comparing SSE2 kernels vs scalar references (compiled on all targets; executed on x86 hosts).
- 2026-02-26: Closed remaining DNN x86 activation-tier dispatch gaps:
  - `src/dnn/simd/x86.rs`: added explicit SSE4.1 and SSE2 `lpcnet_exp`/`softmax` paths matching upstream non-AVX `vec_avx.h` behavior (including SSE4.1 floor semantics).
  - `src/dnn/simd/mod.rs`: `lpcnet_exp` and `softmax` now dispatch AVX2 -> SSE4.1 -> SSE2 -> scalar on x86.
  - `tests/osce_nndsp.rs` + `libopus-sys/src/osce_test_harness.c`: added forced-tier C-vs-Rust regression `test_compute_activation_exp_arch_tiers_match_c` to lock scalar/SSE/SSE2/SSE4.1/AVX2 arch-table parity.
  - This closes dispatch findings `213` and `234`.
- 2026-02-26: Aligned SILK full-function RTCD dispatch surface with upstream x86 maps:
  - `src/silk/VAD.rs` + `src/silk/simd/mod.rs` + `src/silk/simd/x86.rs`: added `silk_VAD_GetSA_Q8` dispatch wrapper and x86 SSE4.1 full-function entry.
  - `src/silk/NSQ.rs` + `src/silk/NSQ_del_dec.rs` + `src/silk/simd/mod.rs` + `src/silk/simd/x86.rs`: added full-function dispatch wrappers for `silk_NSQ` and `silk_NSQ_del_dec` with x86 SSE4.1/AVX2 tier selection and existing scalar fallback.
  - `src/silk/float/wrappers_FLP.rs` + `src/silk/float/encode_frame_FLP.rs`: switched call sites from direct `*_c` functions to RTCD wrappers, matching upstream wrapper call-shape.
  - Validated with `cargo nextest` SILK-focused release tests and preserved DNN tier-parity tests; this closes `202`, `203`, and `233`.
