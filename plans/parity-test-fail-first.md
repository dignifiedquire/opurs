# Parity Test Expansion Plan (Fail-First)

## Goal
Make every known upstream parity gap produce a deterministic failing test before fixing implementation. Focus on bit-exactness and behavior parity with C reference.

## Scope Source
- Findings source: `diff_review.md`
- Current grouped priorities: QEXT blockers, extensions/repacketizer parity, API parity, DNN parity, SIMD/dispatch parity, runtime semantics

## Guardrails
1. Add tests first (red), then implement fix (green).
2. Every new parity test must compare Rust vs upstream C behavior, not only internal invariants.
3. Keep failures reproducible: deterministic seeds + explicit config in assertion messages.
4. Local verification uses `CARGO_TARGET_DIR=target-local`.

## CI Baseline (Current)
- `cargo fmt --all --check`
- `cargo clippy --all --all-targets --features tools -- -D warnings`
- `cargo clippy --all --all-targets --features tools-dnn -- -D warnings`
- `cargo nextest run -p opurs --cargo-profile=release`
- `cargo nextest run --features tools-dnn --cargo-profile=release`
- `cargo run --release --features tools --example run_vectors2 -- opus_newvectors`
- `cargo run --release --features tools-dnn --example run_vectors2 -- --dnn-only opus_newvectors`

## Full-Matrix Fail-First Baseline (Historical)
- Repro command:
  - `CARGO_TARGET_DIR=target-local cargo run --release --features tools --example run_vectors2 -- --matrix full --report-json target-local/run_vectors2_full_all.json opus_newvectors`
- Historical observed status (before implementation fixes):
  - `passed=812 failed=604 skipped=0 total=1416`
- Failure concentration:
  - frame sizes: `40ms` and `60ms` dominate (`300` failures each)
  - applications: all affected (`rld=218`, `voip=194`, `audio=192`)
  - outliers at `10ms/20ms` (4 total): `testvector01@90k voip 20ms`, `testvector02@20k rld 20ms`, `testvector02@120k voip 10ms`, `testvector12@240k rld 10ms`
- Historical interpretation:
  - The expanded harness is now doing fail-first detection for non-baseline encode modes.
  - These failures should be converted into targeted parity tests, then fixed incrementally.

### Intermediate Baseline After Multiframe Budget/Repacketize Fix
- Repro command:
  - `CARGO_TARGET_DIR=target-local cargo run --release --features tools --example run_vectors2 -- --suite classic --matrix full --mode parity --report-json target-local/run_vectors2_full_all_after_multiframe_fix.json opus_newvectors`
- Observed status:
  - `passed=1246 failed=170 skipped=0 total=1416`
- Remaining dominant fail groups:
  - `ENC 045kbps app={audio,voip,rld} frame={40,60}ms` (12 vectors each)
  - `ENC 180kbps app={audio,voip,rld} frame=60ms` (12 vectors each)
  - `ENC 240kbps app={audio,voip,rld} frame=60ms` (12 vectors each)
- Notes:
  - The main 40/60ms multiframe gap dropped substantially (`604 -> 170` fails).
  - Remaining drift is concentrated in long-frame encode parity and likely needs a closer `opus_encode_frame_native`-style path rather than recursive `opus_encode_native` calls.

### Current Baseline (All Green)
- Repro command:
  - `CARGO_TARGET_DIR=target-local cargo run --release --features tools --example run_vectors2 -- --suite classic --matrix full --mode parity --report-json target-local/run_vectors2_full_all_green.json opus_newvectors`
- Observed status:
  - `passed=1416 failed=0 skipped=0 total=1416`
- Additional suite checks:
  - classic vectors: `228/228` passing
  - dnn-only vectors: `264/264` passing
- Use this as the regression baseline for upcoming API expansion and multistream/projection work.

## Completed Summary: SIMD/Dispatch Alignment
- Full divergence tracker moved to `plans/simd_diff.md`.
- Upstream parity divergences `D1` through `D9` are fixed and committed.
- Implemented alignment includes:
  - DNN arch threading for dispatch on x86/ARM paths.
  - x86 CPUID/RTCD parity (`SSE2`, `SSE4.1`, `AVX2`, `FMA`) and mask handling.
  - ARM dotprod detection and native dotprod dispatch path.
  - Build-time `MAY_HAVE_*` probing parity on x86 and AArch64.
  - FUZZING arch downgrade semantics and fuzz feature plumbing.
  - ARMv7 RTCD priority-ladder parity.
  - Decode fuzz target behavior aligned with upstream C harness semantics.
- Validation status after fixes:
  - full classic parity matrix: `1416/1416` passing
  - classic suite: `228/228` passing
  - dnn-only suite: `264/264` passing

## Workstreams

### 1) QEXT Fail-First Coverage (Highest Risk)
Target findings: grouped QEXT blockers in `diff_review.md` (notably IDs `3,4,5,15,16,17,21,22,23,24,25,26,28,29,30,31,32,34,42,52,88,111,150,166,175,229`).

Add tests first:
- New file: `tests/qext_parity.rs` behind `#[cfg(all(feature = "tools", feature = "qext"))]`.
- Add differential test cases at `Fs=96000` for:
  - Encode bitstream parity across frame sizes and channels.
  - Decode PCM parity (including PLC/loss paths).
  - Reset-state parity for QEXT history buffers.
  - 96 kHz specific branches (deemphasis, pitch/prefilter scaling, synthesis energy finalization).
- Use deterministic packet-loss patterns to force PLC/deep PLC decision points.
- Reuse tools helpers (`opus_demo_encode` / `opus_demo_decode`) to compare against C backend.

Acceptance:
- Tests fail on current broken behavior and pass only after matching C behavior.
- Include at least one test per QEXT subgroup: prefilter scaling, decoder PLC scaling, synthesis/deemphasis, reset semantics.

### 2) Extensions + Repacketizer Parity
Target findings: IDs `35,36,37,38,39,40,41,97,98,99,100,115,139`.

Add tests first:
- New file: `tests/extensions_repacketizer_parity.rs`.
- Build packet fixtures that include:
  - Repeated extensions (RTE / repeat markers).
  - Multiple frames with extension payloads in padding.
  - `ext_len` values including exact multiples of 254 to catch overhead formula errors.
  - Out-of-range frame index extension entries.
- Compare Rust parse/count/generate/repacketize results against C reference byte-for-byte where upstream defines canonical output.
- For parse paths, compare normalized semantic outputs (count, ids, frame mapping, payload lengths).

Acceptance:
- Rust must reject/accept exactly same inputs as C.
- Repacketized output with embedded + passed extensions must match C output bytes.

### 3) Public API Surface + Behavioral Parity
Target findings: IDs `10,11,12,43,45,98,104,105,110,116,119,120,122,163,173,177,178,186,199`.

Add tests first:
- New file: `tests/api_surface_parity.rs`.
- Add compile/runtime presence checks for required APIs and CTL constants currently missing or partial.
- Add behavior tests for ignore-extensions decode control and custom/24-bit API parity once entry points exist.

Acceptance:
- Missing API entry points are represented by failing tests until implemented.
- CTL constants and semantics match upstream values and behavior.

### 4) DNN / DRED / OSCE Differential Coverage
Target findings: IDs `12,45,76,94,135,136,137,175,176,177,178,179,180,181,182,187,191,192,193,194,201,206,210,211,216,217,219,220,235`.

Add tests first:
- Extend `tests/dnn_integration.rs` with deterministic differential tests against C for:
  - DRED encode/decode model paths.
  - OSCE decode paths at complexities activating LACE/NoLACE.
  - Decoder-state transitions after DRED usage.
- Add vector differential test expansion in `examples/run_vectors2.rs`:
  - Add focused DNN configs that stress known mismatch regions, not just broad matrix counts.

Acceptance:
- Failures indicate exact model/path and frame index.
- DNN CI lane fails for any backend divergence.

### 5) SIMD / Dispatch / No-SIMD Parity
Target findings: IDs `64,91,107,194,202,203,204,205,212,213,214,224,230,231,232,233,234,235,236,237`.

Add tests first:
- Extend `tests/simd_comparison.rs` to enforce Rust scalar vs Rust SIMD vs C parity for selected kernels and end-to-end frame paths.
- Add lane-specific assertions for runtime CPU dispatch decisions where observable.
- Ensure same cases run with `--no-default-features` (scalar-only) in CI to catch drift.

Acceptance:
- SIMD and scalar outputs remain parity-equivalent (or within upstream-defined tolerance where not bit-exact).
- Current status:
  - Primary dispatch/build/detection divergences resolved (see `plans/simd_diff.md`).
  - Remaining work in this stream is regression hardening and additional targeted SIMD/scalar parity coverage.

### 6) Runtime Semantics / Error Contract Parity
Target findings: representative IDs `61,62,66,67,68,72,79,82,87,93,94,106,135,136,137,139,140,141,142,143,144,145,146,148,149,153,156,165,168,170,171,172,174`.

Add tests first:
- New file: `tests/error_contract_parity.rs`.
- Convert assert-only assumptions into externally verifiable return-code/behavior checks by comparing Rust and C on invalid inputs and boundary CTL calls.
- Ensure release/dev behavior parity where upstream expects status returns instead of hard assertions.

Acceptance:
- Rust and C return codes and state transitions match for invalid/edge inputs.

## CI Integration Plan
1. Add new nextest lanes (or expand existing `test`, `test-dnn`, `test-no-simd`) to include new parity suites.
2. Keep vector jobs as integration gates; add focused parity tests as unit/integration gates so failures are localized.
3. Optional (after stabilization): add `qext` parity lane:
   - `cargo nextest run -p opurs --features "tools,qext" --cargo-profile=release`
   - `cargo run --release --features "tools,qext" --example run_vectors2 -- opus_newvectors`

## Local Execution Order (Developer)
Run from repo root:

```bash
CARGO_TARGET_DIR=target-local cargo fmt --all --check
CARGO_TARGET_DIR=target-local cargo clippy --all --all-targets --features tools -- -D warnings
CARGO_TARGET_DIR=target-local cargo nextest run -p opurs --cargo-profile=release
CARGO_TARGET_DIR=target-local cargo run --release --features tools --example run_vectors2 -- opus_newvectors
```

For DNN/QEXT workstreams:

```bash
CARGO_TARGET_DIR=target-local cargo clippy --all --all-targets --features tools-dnn -- -D warnings
CARGO_TARGET_DIR=target-local cargo nextest run --features tools-dnn --cargo-profile=release
CARGO_TARGET_DIR=target-local cargo run --release --features tools-dnn --example run_vectors2 -- --dnn-only opus_newvectors
```

## Milestones
1. M1: Land QEXT fail-first tests (expected red).
2. M2: Land extensions/repacketizer fail-first tests (expected red).
3. M3: Land API + error-contract fail-first tests (expected red).
4. M4: Fix implementation for M1-M3 until all green.
5. M5: Expand DNN + SIMD differential coverage and stabilize CI.

## Definition of Done
1. Every HIGH parity finding category has at least one dedicated failing test added before fix.
2. All newly added parity tests pass on fixed implementation.
3. CI includes these parity tests in standard lanes (not only ad-hoc examples).
4. `run_vectors2` remains green for baseline (`tools`) and DNN (`tools-dnn`) matrices.
