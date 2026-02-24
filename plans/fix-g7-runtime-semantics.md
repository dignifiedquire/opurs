# Plan: Group 7 Runtime Semantics Assert Vs Status

## Goal
Align runtime error semantics with upstream by replacing panic/assert-only behavior where upstream returns status or uses assert-gated checks.

## Findings IDs
`61,62,66,67,68,72,79,82,87,93,94,106,135,136,137,139,140,141,142,143,144,145,146,148,149,153,156,165,168,170,171,172,174`

## Scope
- Decoder/encoder/CELT/SILK/DNN invariant handling.
- Return-code behavior for invalid inputs and boundary CTL calls.
- Hardened-vs-release behavior consistency.

## Execution Order
1. Classify each site: should return status, should be debug assertion, or should remain fatal.
2. Convert panic paths that should return Opus errors.
3. Keep internal assertions behind debug/hardening-style gates.
4. Add negative tests for invalid inputs and state transitions.

## Verification
- `CARGO_TARGET_DIR=target-local cargo nextest run -p opurs --cargo-profile=release`
- `CARGO_TARGET_DIR=target-local cargo test -p opurs --test error_contract_parity --release`

## Definition Of Done
- Runtime behavior on invalid/edge inputs matches upstream status semantics for covered paths.

## Progress
- 2026-02-24: Aligned `opus_strerror()` output strings in `src/celt/common.rs` to upstream canonical text (removed numeric suffix decorations).
