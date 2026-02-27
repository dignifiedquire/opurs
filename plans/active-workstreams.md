# Active Workstreams (Source of Truth)

Last updated: 2026-02-27

This file tracks what is still actionable. It supersedes stale status text in older phase files.

## Completed parity tracks

- [x] Group 1 QEXT correctness (`done/fix-g1-qext-correctness.md`)
- [x] Group 2 extensions/repacketizer parity (`done/fix-g2-extensions-repacketizer.md`)
- [x] Group 3 public API parity scope (`done/fix-g3-public-api-parity.md`)
- [x] Group 4 DNN/DRED/OSCE parity (`done/fix-g4-dnn-dred-osce.md`)
- [x] Group 5 SIMD/dispatch/build parity (`done/fix-g5-simd-dispatch-build.md`)
- [x] Group 6 docs/version metadata drift (`done/fix-g6-docs-version-metadata.md`)
- [x] Group 7 runtime semantics/assert-vs-status parity (`done/fix-g7-runtime-semantics.md`)

Reference snapshot: `done/diff_review.md` shows groups 1-7 resolved, with excluded C-API-surface-only IDs.

## Active work

### A1. Performance parity closure
Source: `phase5-performance.md`, `perf-status.md`

- [ ] Keep `phase5-performance.md` synchronized with new benchmark runs and closure criteria.
- [ ] Close decode-side perf gap on arm64 (priority from `perf-status.md`: stereo and low-channel multistream decode).
- [ ] Finalize parity report commit/doc step for current benchmark set.

### A2. API expansion milestones still open in checklist
Source: `api-expansion-upstream-parity.md`

- [ ] M1.2 lifecycle + encode/decode parity checkbox closure
- [ ] M1.4 fail-first then green test checklist closure
- [ ] M1.5 multistream API-surface checklist closure
- [ ] M3 projection/ambisonics milestone closure (asset-backed vector policy still pending)
- [ ] M4 benchmark expansion/baselines milestone closure

### A3. Unsafe policy and wrapper cleanup
Source: `remove-all-unsafe.md`, `phase4-integration-safety.md`

- [ ] Resolve/document the remaining `src/celt/mdct.rs` unsafe blocks policy.
- [ ] Finalize crate-level unsafe policy (`deny` + targeted allows, or stricter if feasible).
- [ ] Decide whether remaining Phase 4 ergonomic wrapper tasks are in-scope now or formally deferred.

## Explicitly deferred/non-blocking

- C symbol-level API surface parity excluded from functional-equivalence tracking (`done/diff_review.md`).
- Legacy Phase 4 wrapper wishlist items unless moved into active scope.

## Archive

- Historical/superseded plans are under `archive/`.
