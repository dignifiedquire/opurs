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
- 2026-02-25: Continued assert-gate parity cleanup in encoder/decoder hot paths:
  - `src/opus/opus_decoder.rs`: switched decoder invariant validation and decode-loop internal consistency checks from unconditional `assert!` to `debug_assert!`/`debug_assert_eq!`, matching upstream `VALIDATE_*`/`celt_assert` production behavior.
  - `src/celt/celt_decoder.rs`: switched `validate_celt_decoder()` invariant checks to debug-only assertions.
  - `src/opus/opus_encoder.rs`: switched three remaining runtime `assert!` sites (bandwidth/internal-rate/DRED-size invariants) to debug-only assertions, matching upstream `celt_assert` behavior.
  - `src/silk/enc_API.rs`: aligned channel-setup loops with upstream (`nChannelsAPI` bounded), added missing entry `celt_assert` equivalent as debug-only, and converted remaining internal invariant asserts to debug-only checks.
- 2026-02-25: Aligned additional SILK decode-path error semantics and coverage:
  - `src/silk/dec_API.rs`: converted invalid payload-size and invalid internal-sample-rate branches from panic behavior to upstream-style `SILK_DEC_INVALID_FRAME_SIZE` / `SILK_DEC_INVALID_SAMPLING_FREQUENCY` status returns.
  - `src/silk/dec_API.rs`: added unit tests that exercise those two status-return paths directly.
  - `src/silk/decoder_set_fs.rs` and `src/silk/float/LPC_analysis_filter_FLP.rs`: converted remaining unconditional internal asserts/panic branches to debug-only assertion behavior matching upstream `celt_assert` semantics.
- 2026-02-25: Aligned SILK encoder control validation/runtime semantics with upstream status-code behavior:
  - `src/silk/check_control_input.rs` now returns `SILK_ENC_*` error codes instead of panicking.
  - Added unit coverage for valid controls plus invalid payload/loss/complexity/channel paths and qext-gated 96 kHz API-rate acceptance.
  - `src/silk/enc_API.rs` now propagates control/input validation failures as return codes (`SILK_ENC_INPUT_INVALID_NO_OF_SAMPLES` / check-control return) instead of panicking in `silk_InitEncoder` and early `silk_Encode` validation gates.
- 2026-02-24: Aligned `opus_strerror()` output strings in `src/celt/common.rs` to upstream canonical text (removed numeric suffix decorations).
- 2026-02-24: Replaced `resampling_factor()` panic path with upstream-style `0` return on unsupported rates (`src/celt/common.rs`), and propagated this to non-panicking `OPUS_BAD_ARG` returns in CELT init paths.
- 2026-02-24: Converted `celt_decoder_init()` and internal custom-decoder init to return `Result<_, i32>` instead of panicking on invalid sampling rates/channels (`src/celt/celt_decoder.rs`), and propagated through `OpusDecoder::new`.
- 2026-02-24: Updated `compute_qext_mode()` unsupported tuple handling from unconditional panic to assertion-style behavior (`src/celt/modes.rs`) to match upstream `celt_assert(0)` semantics.
- 2026-02-24: Added CELT init negative tests for invalid sampling rate/channel behavior and validated with release+qext unit runs.
- 2026-02-24: Replaced two `opus_decode_frame` default-switch panics with assert-style fallback behavior (`src/opus/opus_decoder.rs`) to mirror upstream `celt_assert` branches.
- 2026-02-24: Replaced DRED 16 kHz conversion unsupported-rate panic with assert-style fallback initialization (`src/dnn/dred/encoder.rs`).
- 2026-02-24: Aligned `opus_packet_parse_impl` padding-output error semantics with upstream by zeroing `padding_out` on entry (`src/opus/packet.rs`), and added a C-vs-Rust parity test (`tests/packet_parse_impl_parity.rs`) covering error paths.
- 2026-02-24: Added C-vs-Rust runtime parity coverage for encoder bitrate CTL semantics in `tests/restricted_application_parity.rs` (`bitrate_ctl_semantics_match_c`), validating non-positive explicit bitrate rejection and high-bitrate clamping behavior against upstream requests `OPUS_SET/GET_BITRATE_REQUEST`.
