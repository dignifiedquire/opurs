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
- 2026-02-24: Replaced `resampling_factor()` panic path with upstream-style `0` return on unsupported rates (`src/celt/common.rs`), and propagated this to non-panicking `OPUS_BAD_ARG` returns in CELT init paths.
- 2026-02-24: Converted `celt_decoder_init()` and internal custom-decoder init to return `Result<_, i32>` instead of panicking on invalid sampling rates/channels (`src/celt/celt_decoder.rs`), and propagated through `OpusDecoder::new`.
- 2026-02-24: Updated `compute_qext_mode()` unsupported tuple handling from unconditional panic to assertion-style behavior (`src/celt/modes.rs`) to match upstream `celt_assert(0)` semantics.
- 2026-02-24: Added CELT init negative tests for invalid sampling rate/channel behavior and validated with release+qext unit runs.
- 2026-02-24: Replaced two `opus_decode_frame` default-switch panics with assert-style fallback behavior (`src/opus/opus_decoder.rs`) to mirror upstream `celt_assert` branches.
- 2026-02-24: Replaced DRED 16 kHz conversion unsupported-rate panic with assert-style fallback initialization (`src/dnn/dred/encoder.rs`).
- 2026-02-24: Aligned `opus_packet_parse_impl` padding-output error semantics with upstream by zeroing `padding_out` on entry (`src/opus/packet.rs`), and added a C-vs-Rust parity test (`tests/packet_parse_impl_parity.rs`) covering error paths.
