# Plan: Group 4 DNN DRED OSCE Parity

## Goal
Align DNN/DRED/OSCE model loading, constants, dispatch signatures, and behavior with upstream.

## Findings IDs
`12,45,76,94,135,136,137,175,176,177,178,179,180,181,182,187,191,192,193,194,201,206,210,211,216,217,219,220,235`

## Scope
- DRED API and behavior parity in decode and state transitions.
- OSCE/LPCNet feature-extraction and arch-signature parity.
- DNN compute call-shape and model path consistency.

## Execution Order
1. Split findings into API-shape vs behavior vs constants.
2. Fix API-shape/constant mismatches first to reduce integration churn.
3. Fix behavior mismatches in model paths with targeted vectors.
4. Add deterministic DNN-only differential coverage per model path.

## Verification
- `CARGO_TARGET_DIR=target-local cargo clippy --all --all-targets --features tools-dnn -- -D warnings`
- `CARGO_TARGET_DIR=target-local cargo nextest run --features tools-dnn --cargo-profile=release`
- `CARGO_TARGET_DIR=target-local cargo run --release --features tools-dnn --example run_vectors2 -- --dnn-only opus_newvectors`

## Definition Of Done
- DNN/DRED/OSCE lanes are green with no unresolved high/medium parity regressions.

## Progress
- 2026-02-24: Aligned OSCE model-loading contract with upstream `osce_load_models` semantics: Rust now marks model load success only when all enabled OSCE components (`lace`, `nolace`, `bbwenet`) initialize successfully.
- 2026-02-24: Added regression coverage in `tests/dnn_integration.rs` (`osce_model_load_rejects_partial_weights`) to ensure partial weight bundles fail load and do not mark the OSCE model as loaded.
- 2026-02-26: Aligned DNN sparse-index parsing with upstream `find_idx_check` semantics in `linear_init`, adding structural, alignment, and bounds checks plus dedicated unit tests for malformed and valid sparse streams.
