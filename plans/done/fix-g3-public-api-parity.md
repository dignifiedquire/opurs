# Plan: Group 3 Public API Surface Parity

## Goal
Close remaining public API coverage gaps versus upstream core/custom/multistream/projection and related controls.

## Findings IDs
Open: `none (resolved/excluded)`

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
- 2026-02-24: Added missing upstream CTL request constants for OSCE/QEXT/ignore-extensions (`4054..4059`) in `src/opus/opus_defines.rs` and re-exported them in `src/lib.rs`.
- 2026-02-24: Added typed API coverage for the newly represented controls across wrappers:
  - `set_osce_bwe()` / `osce_bwe()` on multistream and projection decoders
  - `set_qext()` / `qext()` on projection encoder
- 2026-02-24: Added integration coverage for these control paths in `tests/ctl_api_controls.rs` (base + `qext` + `osce` feature variants).
- 2026-02-24: Added a gap matrix in `plans/missing-impl-test-coverage.md` to track unresolved implementation details and required path-level test coverage (prioritized M01..M09).
- 2026-02-24: Upstream parity review for qext decoder paths fixed two concrete divergences:
  - `src/celt/celt_decoder.rs`: allocate `X` as `C*N` (matching upstream) instead of fixed 1920.
  - `src/celt/cwrs.rs`: added safe PVQ-U lookup fallback equivalent to upstream SMALL_FOOTPRINT recurrence when compact table rows are exceeded (qext/cwrs-extra-row cases).
- 2026-02-24: Added C-vs-Rust parity coverage for restricted-application encoder control flow in `tests/restricted_application_parity.rs` (`--features tools`):
  - restricted-SILK 5 ms encode rejection parity (`OPUS_BAD_ARG`)
  - `OPUS_SET_APPLICATION_REQUEST` parity for restricted application values and restricted-instance application-change rejection
- 2026-02-25: Resolved three open Group 3 API-shape gaps:
  - `src/dnn/pitchdnn.rs`: added one-shot blob API (`pitchdnn_load_model`, `PitchDNNState::load_model`) and coverage in `tests/dnn_integration.rs`.
  - `src/silk/init_decoder.rs` + `src/silk/dec_API.rs`: switched low-level decoder init/reset to in-place status-return flow and updated decode-path call sites.
  - `src/dnn/nnet.rs`: added `compute_gated_activation` and unit tests, and aligned `compute_glu` in-place behavior with upstream pointer-alias semantics.
- 2026-02-25: Confirmed Group 3 tracking excludes C-specific API-shape-only comparisons (`177,178,186`) and focuses on functional behavior parity.
- 2026-02-26: Added missing DRED decoder API entry points in `src/opus/opus_decoder.rs` and re-exported them in `src/lib.rs`:
  - `opus_decoder_dred_decode`
  - `opus_decoder_dred_decode24`
  - `opus_decoder_dred_decode_float`
  - Added parity coverage in `tests/dred_decode_parity.rs` (`dred_decode_float_stage0_matches_c_null_dred_path`).
- 2026-02-26: Added full DRED object workflow API wrappers in `src/opus/opus_decoder.rs` and re-exported in `src/lib.rs`:
  - decoder lifecycle/control: `opus_dred_decoder_get_size`, `opus_dred_decoder_create`, `opus_dred_decoder_init`, `opus_dred_decoder_destroy`, `opus_dred_decoder_ctl`
  - DRED object lifecycle/parse/process: `opus_dred_get_size`, `opus_dred_alloc`, `opus_dred_free`, `opus_dred_parse`, `opus_dred_process`
  - Added unit coverage for parse/control paths in `src/opus/opus_decoder.rs` tests.
- 2026-02-26: Closed remaining Group 3 items:
  - Added custom CELT 24-bit API coverage in Rust (`OpusCustomEncoder::encode24`, `OpusCustomDecoder::decode24`) plus parity tests against upstream C in `tests/opus_custom_api.rs`.
  - Enabled and exposed upstream C custom symbols in `libopus-sys` needed for parity tests (`opus_custom_*` + custom ctl).
  - Marked C-ABI-only findings (`110,116,122`) as excluded from functional-equivalence tracking per project policy.
  - Marked stale/resolved API findings (`104,119,120`) as resolved in `diff_review.md`.
- 2026-02-26: Added typed control coverage for encoder LFE and energy-mask requests:
  - `OpusEncoder::set_lfe()/lfe()`
  - `OpusEncoder::set_energy_mask()/energy_mask()`
  - Added C-vs-Rust parity checks in `tests/restricted_application_parity.rs` for `OPUS_SET_LFE_REQUEST` and `OPUS_SET_ENERGY_MASK_REQUEST`.
