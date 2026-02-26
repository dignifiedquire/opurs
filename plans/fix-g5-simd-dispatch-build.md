# Plan: Group 5 SIMD Dispatch And Build Parity

## Goal
Resolve remaining SIMD dispatch semantics and build-flag parity gaps after D1-D9 alignment.

## Findings IDs
Open: `194,202,203,213,233,234`
Closed in this group: `107,204,205,212,230,231,232,235,237`

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
