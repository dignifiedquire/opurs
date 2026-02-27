# Remaining Plan (Verified)

Last updated: 2026-02-27

This is the single active plan. It replaces multi-file active planning.

## Verification Summary (what is still needed)

| Source | Status | Still needed? | Why |
|---|---|---|---|
| `done/diff_review.md` + `done/fix-g1..g7` | Closed | No | Functional parity groups are resolved. |
| `parity-test-fail-first.md` | Green baseline | No new workstream | Keep as historical red/green methodology and baseline context. |
| `missing-impl-test-coverage.md` | Covered/green | No new workstream | Items M01-M09 are covered; keep CI checks in place. |
| `done/upstream-tests-full-match.md` | Closed | No | S1-S7 complete; upstream test full-match audit recorded and moved to `done/`. |
| `phase5-performance.md` + `perf-status.md` | Open | Yes | Decode-side perf gap remains vs C target. |
| `api-expansion-upstream-parity.md` | Mostly reconciled | Yes (final closure move) | Snapshot/checklist now refreshed; final step is moving it to `done/` once closure is reconfirmed in CI. |
| `phase4-integration-safety.md` + `remove-all-unsafe.md` | Mostly done | Yes (policy closure only) | Must finalize unsafe policy and mdct unsafe-block rationale. |
| `done/upgrade-1.6.1.md` | Historical migration plan | No (unless new upstream delta appears) | Not an active blocker for current parity/perf closure. |

## Scope Exclusions (explicit)

- C symbol-level API parity shims are excluded from functional-equivalence scope.
- Legacy Phase 4 ergonomic wrapper wishlist is deferred unless explicitly re-scoped.
- Historical incident docs under `archive/` are non-execution references.

## Workstream R1: Performance Closure (Required)

Owner objective: close meaningful decode-side gap while preserving vector bit-exactness.

- [ ] R1.1 Capture current baseline from `perf-status.md` and record target deltas for arm64 decode paths.
- [ ] R1.2 Profile decode hot paths (`cargo flamegraph`/equivalent) and identify top contributors.
- [ ] R1.3 Implement targeted decode optimizations (no behavior drift).
- [ ] R1.4 Re-run benchmark suite (sequential) and update `perf-status.md` with before/after.
- [ ] R1.5 Re-run vector parity gates (`tools` and `tools-dnn`) and confirm no regressions.
- [ ] R1.6 Close residual checklist items in `phase5-performance.md`; move it to `done/` when complete.

## Workstream R2: API Expansion Milestone Closure (Required)

Owner objective: prove remaining unchecked milestone items are either complete or truly missing.

- [x] R2.1 Audit `api-expansion-upstream-parity.md` checklist (`M1.2`, `M1.4`, `M1.5`, `M3`, `M4`) against code/tests/CI.
- [x] R2.2 For each unchecked item, classify as one of:
  - implemented (checkbox stale),
  - missing and required,
  - deferred by policy (document rationale).
- [x] R2.3 Resolve any truly missing required item(s).
- [x] R2.4 Finalize projection strict parity policy (major-platform CI lanes, URL-independent) and document decision.
- [x] R2.5 Mark milestone status accurately.
- [ ] R2.6 Move `api-expansion-upstream-parity.md` to `done/` once final milestone reconciliation is confirmed in CI.

## Workstream R3: Unsafe Policy Closure (Required)

Owner objective: close remaining policy ambiguity around unsafe usage.

- [ ] R3.1 Decide and document crate-level unsafe policy (`deny(unsafe_code)` with targeted allows is current likely path).
- [ ] R3.2 Resolve or formally justify `src/celt/mdct.rs` remaining unsafe blocks with explicit invariants/tests.
- [ ] R3.3 Reconcile `phase4-integration-safety.md` and `remove-all-unsafe.md` with actual scope (drop stale wishlist items).
- [ ] R3.4 If fully closed, move `phase4-integration-safety.md` and `remove-all-unsafe.md` to `done/`.

## Validation Gate (must hold)

- [ ] `cargo fmt --all --check`
- [ ] `cargo clippy --all --all-targets --features tools -- -D warnings`
- [ ] `cargo clippy --all --all-targets --features tools-dnn -- -D warnings`
- [ ] `cargo nextest run -p opurs --cargo-profile=release`
- [ ] `cargo nextest run --features tools-dnn --cargo-profile=release`
- [ ] `cargo run --release --features tools --example run_vectors2 -- opus_newvectors`
- [ ] `cargo run --release --features tools-dnn --example run_vectors2 -- --dnn-only --matrix full opus_newvectors`

## Exit Criteria

- [ ] R1-R3 complete and validated.
- [ ] No other active planning files remain in `plans/` besides this file and live status docs.
- [ ] `plans/PLAN.md` points to this as the single active plan.
