# Plan: Group 4 DNN DRED OSCE Parity

## Goal
Align DNN/DRED/OSCE model loading, constants, dispatch signatures, and behavior with upstream.

## Findings IDs
Open: `179,180,191,192,193`
Excluded (intentional API-shape): `177,178`
Closed in this group: `12,45,76,94,135,136,137,175,176,181,182,187,194,201,206,210,211,215,216,217,219,220,221`

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
- 2026-02-26: Aligned `opt_array_check` and parser record validation semantics with upstream in `nnet` (`linear_init`/`conv2d_init` optional float size mismatch failures, `parse_weights` zero-size rejection, strict `name[43]==0` check), with focused regression tests.
- 2026-02-26: Re-audited and closed additional stale parity findings now reflected in code: OSCE builtin aggregation includes BBWENet in `compiled_weights`, OSCE load success requires all model components, DNN README baseline text updated to 1.6.1, and MLP GRU/tansig parity updates (`MAX_NEURONS` bound and upstream-precision coefficients).
- 2026-02-26: Aligned LPCNet PLC `init()` loaded-state semantics with the stricter all-components-ready contract (`init_plcmodel` + encoder model + FARGAN init), and added `tests/dnn_integration.rs:lpcnet_plc_init_rejects_partial_weights`.
- 2026-02-26: Aligned DNN runtime invariant gating with upstream assert semantics by converting unconditional DNN asserts to debug-gated checks in:
  - `src/dnn/lpcnet.rs` / `src/dnn/fargan.rs` (loaded/continuation preconditions)
  - `src/dnn/nnet.rs` / `src/dnn/nndsp.rs` (internal dimension/invariant checks)
  - closes `136` and `137`.
- 2026-02-26: Closed additional stale API/state-shape gaps after code re-audit:
  - `187`: LPCNet single-frame feature helpers include `arch`.
  - `210`: `compute_pitchdnn` includes `arch`.
  - `211`: `PitchDNNState` now includes `xcorr_mem3` (state-layout parity).
  - `216`: LPCNet feature extraction entry points thread `arch`.
- 2026-02-26: Aligned DNN conv2d algorithmic path with upstream by adding the `ktime==3 && kheight==3` fast-path branch (`conv2d_3x3_float`) in `compute_conv2d`, and added `tests/osce_nndsp.rs:test_compute_conv2d_3x3` for bit-exact C-vs-Rust coverage.
- 2026-02-26: Aligned decoder DRED-assisted decode path with upstream:
  - `src/opus/opus_decoder.rs`: `opus_decode_native` now accepts DRED inputs and threads the upstream pre-PLC DRED feature staging flow (`dred` + `dred_offset`) before PLC/FEC decode.
  - `src/opus/opus_decoder.rs`: added public `opus_decoder_dred_decode`, `opus_decoder_dred_decode24`, `opus_decoder_dred_decode_float` entry points plus `OpusDecoder` convenience methods.
  - `src/lib.rs`: re-exported new DRED decode entry points.
  - `tests/dred_decode_parity.rs`: added tools-dnn C-vs-Rust parity check for DRED decode wrapper path.
  - closes `12` and `76`.
- 2026-02-26: Added the remaining DRED object lifecycle and parse/process API wrappers in `src/opus/opus_decoder.rs`:
  - `opus_dred_decoder_get_size`, `opus_dred_decoder_create`, `opus_dred_decoder_init`, `opus_dred_decoder_destroy`, `opus_dred_decoder_ctl`
  - `opus_dred_get_size`, `opus_dred_alloc`, `opus_dred_free`, `opus_dred_parse`, `opus_dred_process`
  - `src/lib.rs` now re-exports `OpusDRED`, `OpusDREDDecoder`, and these wrapper APIs.
  - Added focused unit coverage for DRED parse/control behavior and kept tools-dnn parity test coverage for DRED decode wrapper flow.
  - closes `45`.
