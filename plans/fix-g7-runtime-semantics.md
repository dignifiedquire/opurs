# Plan: Group 7 Runtime Semantics Assert Vs Status

## Goal
Align runtime error semantics with upstream by replacing panic/assert-only behavior where upstream returns status or uses assert-gated checks.

## Findings IDs
Open: `61,62,72,79,82,87,106,140,141,142,143,144,145,146,148,149,153,170,171,172`
Closed in this group: `66,67,68,135,136,137,168`

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
- 2026-02-26: Closed DNN runtime assert-gating parity items:
  - `src/dnn/lpcnet.rs` / `src/dnn/fargan.rs`: converted unconditional loaded/continuation assertions to debug assertions.
  - `src/dnn/nnet.rs` / `src/dnn/nndsp.rs`: converted internal invariant checks from unconditional assertions to debug-gated assertions.
  - closes `136` and `137`.
- 2026-02-26: Closed additional runtime/assert-gate parity gaps across analysis, resampler, and DRED init paths:
  - `src/opus/analysis.rs`: added upstream-equivalent `Fs` assertion gate in `downmix_and_resample()` and unit tests for valid 24 kHz behavior plus unsupported-rate debug assertion.
  - `src/silk/resampler/mod.rs`: removed remaining invalid-rate panic path in `rate_id()` by returning `Option` and mapping unsupported tuples to `SILK_RESAMPLER_INVALID` (`-1`) with debug-assert gating; added debug/release tests for invalid encoder/decoder tuples.
  - `src/dnn/dred/encoder.rs`: switched loaded-state preconditions in DRED hot paths from unconditional `assert!` to debug assertions to match upstream `celt_assert` gating.
  - `src/dnn/dred/encoder.rs`: `DREDEnc::init` now mirrors upstream built-in model auto-load behavior when `builtin-weights` is enabled.
  - `src/opus/opus_encoder.rs`: `OpusEncoder::new` now explicitly invokes `dred_encoder.init(Fs, channels)` (upstream parity with `dred_encoder_init` call in encoder init); added unit tests for Fs/channels propagation and built-in auto-load path.
  - `src/dnn/dred/encoder.rs`: added qext-gated unit test locking 96 kHz `dred_convert_to_16k` behavior.
- 2026-02-25: Closed additional repacketizer/SILK runtime parity gaps:
  - `src/opus/repacketizer.rs`: added explicit `len/new_len <= data.len()` checks in `opus_packet_pad_impl` to return `OPUS_BAD_ARG` instead of panicking on slice bounds.
  - `tests/extensions_repacketizer_parity.rs`: added negative tests for oversize `len` and `new_len` in `opus_packet_pad_impl`.
  - `src/opus/repacketizer.rs`: converted `opus_packet_unpad` postcondition check from `assert!` to `debug_assert!` to match upstream `celt_assert` gating.
  - `src/silk/resampler/down_fir.rs`: replaced down-FIR default-branch `unreachable!()` with debug-assert + return fallback matching upstream assert-gated behavior.
  - `src/dnn/osce.rs` and `src/silk/init_decoder.rs`: introduced and used `OSCE_DEFAULT_METHOD` during decoder reset to match upstream `silk_reset_decoder`.
  - `src/silk/decode_pitch.rs`: replaced strict tuple match + `unreachable!` with upstream-style fallback table selection and debug-assert-gated `nb_subfr` checks.
- 2026-02-25: Continued assert-gate parity cleanup in encoder/decoder hot paths:
  - `src/opus/opus_decoder.rs`: switched decoder invariant validation and decode-loop internal consistency checks from unconditional `assert!` to `debug_assert!`/`debug_assert_eq!`, matching upstream `VALIDATE_*`/`celt_assert` production behavior.
  - `src/celt/celt_decoder.rs`: switched `validate_celt_decoder()` invariant checks to debug-only assertions.
  - `src/opus/opus_encoder.rs`: switched three remaining runtime `assert!` sites (bandwidth/internal-rate/DRED-size invariants) to debug-only assertions, matching upstream `celt_assert` behavior.
  - `src/silk/enc_API.rs`: aligned channel-setup loops with upstream (`nChannelsAPI` bounded), added missing entry `celt_assert` equivalent as debug-only, and converted remaining internal invariant asserts to debug-only checks.
- 2026-02-25: Aligned additional SILK decode-path error semantics and coverage:
  - `src/silk/dec_API.rs`: converted invalid payload-size and invalid internal-sample-rate branches from panic behavior to upstream-style `SILK_DEC_INVALID_FRAME_SIZE` / `SILK_DEC_INVALID_SAMPLING_FREQUENCY` status returns.
  - `src/silk/dec_API.rs`: added unit tests that exercise those two status-return paths directly.
  - `src/silk/decoder_set_fs.rs` and `src/silk/float/LPC_analysis_filter_FLP.rs`: converted remaining unconditional internal asserts/panic branches to debug-only assertion behavior matching upstream `celt_assert` semantics.
- 2026-02-25: Aligned SILK resampler init semantics with upstream status-return behavior:
  - `src/silk/resampler/mod.rs`: changed `silk_resampler_init` to C-style `state + return-code` contract and removed panic paths in invalid input/rate-ratio branches; now returns `-1` on invalid combinations like upstream.
  - `src/silk/control_codec.rs` and `src/silk/decoder_set_fs.rs`: updated call sites to propagate resampler-init status codes instead of relying on implicit panic behavior.
  - Converted remaining resampler internal invariant checks from `assert!` to `debug_assert!` to match upstream assertion gating.
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
