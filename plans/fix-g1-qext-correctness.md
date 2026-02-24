# Plan: Group 1 QEXT Correctness Blockers

## Goal
Close all QEXT bitstream/PLC/sizing correctness gaps from `diff_review.md`.

## Findings IDs
`3,4,5,15,16,17,21,22,23,24,25,26,28,29,30,31,32,34,42,52,88,111,150,166,175,229`

## Scope
- CELT encoder QEXT scaling, prefilter memory/layout, overlap history sizing.
- CELT decoder QEXT PLC, pitch search/downsample scaling, deemphasis, synthesis denormalisation.
- QEXT state init/reset behavior and 96 kHz mode guardrails.
- Opus encoder QEXT payload provisioning path.

## Execution Order
1. Land fail-first tests for each QEXT subgroup (encoder prefilter, decoder PLC, synthesis/deemphasis, reset semantics).
2. Fix core scaling primitives and sizing constants first (`QEXT_SCALE` paths), then call-site wiring.
3. Fix encoder payload provisioning (`qext_payload`/`qext_bytes`) and integration flow.
4. Re-run full vector matrix and focused QEXT differential tests.

## Verification
- `CARGO_TARGET_DIR=target-local cargo nextest run -p opurs --features "tools,qext" --cargo-profile=release`
- `CARGO_TARGET_DIR=target-local cargo run --release --features "tools,qext" --example run_vectors2 -- --suite classic --matrix full --mode parity opus_newvectors`

## Definition Of Done
- All listed IDs are either fixed in code or intentionally deferred with explicit rationale.
- QEXT parity tests pass consistently in CI and local matrix runs.
