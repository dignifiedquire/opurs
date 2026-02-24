# Plan: Group 3 Public API Surface Parity

## Goal
Close remaining public API coverage gaps versus upstream core/custom/multistream/projection and related controls.

## Findings IDs
`12,43,45,98,104,110,116,119,120,122,163,173,177,178,186,199`

## Scope
- DRED public decoder/aux APIs and state lifecycle entry points.
- Missing CTL constants and request coverage.
- Multistream/projection API surface and required support modules.
- Excludes introducing C-ABI-specific entry points solely for pointer-style lifecycle parity.

## Execution Order
1. Add compile-time API presence tests for all missing entry points and constants.
2. Implement missing constants and direct wrappers first (low-risk).
3. Implement multistream/projection API modules and glue.
4. Add behavioral differential tests for decode/ctl behavior.

## Verification
- `CARGO_TARGET_DIR=target-local cargo test -p opurs --release`
- `CARGO_TARGET_DIR=target-local cargo nextest run -p opurs --cargo-profile=release`

## Definition Of Done
- All required public symbols/constants exist and match upstream values.
- API behavior parity tests pass for supported features.

## Progress
- 2026-02-24: Added missing upstream application constants `OPUS_APPLICATION_RESTRICTED_SILK` (2052) and `OPUS_APPLICATION_RESTRICTED_CELT` (2053) in `src/opus/opus_defines.rs` and re-exported via `src/lib.rs`.
- 2026-02-24: Extended typed `Application` enum conversions for restricted SILK/CELT values in `src/enums.rs`.
- 2026-02-24: Aligned core encoder restricted-application behavior with upstream in `src/opus/opus_encoder.rs`:
  - constructor accepts restricted SILK/CELT values and applies `encoder_buffer=0` for both restricted modes
  - `frame_size_select()` now takes `application` and enforces restricted-SILK minimum 10 ms frame rule
  - analysis gating now skips `run_analysis()` in restricted SILK mode
  - mode selection now forces SILK-only for restricted SILK and CELT-only for restricted CELT
  - delay compensation, lookahead, and bandwidth/LSB-depth branches updated for restricted mode semantics
- 2026-02-24: Updated multistream encoder call-sites to pass application into `frame_size_select()` (`src/opus/opus_multistream_encoder.rs`).
