# Plan: Group 5 SIMD Dispatch And Build Parity

## Goal
Resolve remaining SIMD dispatch semantics and build-flag parity gaps after D1-D9 alignment.

## Findings IDs
`107,194,202,203,204,205,212,213,230,231,232,233,234,235,237`

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
