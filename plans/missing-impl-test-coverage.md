# Missing Impl + Test Coverage Matrix

## Purpose
Track implementation gaps and untested runtime paths, with explicit tests required to close each gap.

## P0 (must fix first)

| ID | Area | Missing impl detail / missing path | Current status | Required test coverage |
|---|---|---|---|---|
| M01 | QEXT/CELT runtime | QEXT-enabled runtime hit panics due upstream mismatches in CELT buffer/table handling. | Fixed locally; verified by `tests/ctl_api_controls.rs` with `qext` and `qext,osce` | Keep regression tests that run real extension-bearing encode/decode paths for `OpusEncoder`/`OpusMSEncoder`/`OpusProjectionEncoder` and assert no panic in release mode. |
| M02 | Decoder extension dispatch | Real `ignore_extensions` behavior in decoder loop (`src/opus/opus_decoder.rs:1060`) is not stably covered with real QEXT packets. | Covered in `tests/ctl_api_controls.rs` (`decoder_ignore_extensions_matches_unpadded_decode_for_real_qext_packets`) | Keep tests proving: `ignore_extensions=true` decode matches extension-stripped decode, and `ignore_extensions=false` diverges when QEXT payload is meaningful. |
| M03 | Multistream/projection extension path | No stable test for extension-aware dispatch in multistream/projection decode wrappers (fan-out paths). | Covered in `tests/ctl_api_controls.rs` (`ms_decoder_ignore_extensions_matches_unpadded_decode_for_real_qext_packets`, `projection_decoder_ignore_extensions_matches_unpadded_decode_for_real_qext_packets`) | Keep wrapper-level behavior tests that exercise real packet decode through `OpusMSDecoder` and `OpusProjectionDecoder` with extension-bearing packets. |

## P1 (next)

| ID | Area | Missing impl detail / missing path | Current status | Required test coverage |
|---|---|---|---|---|
| M04 | OSCE controls | `set_osce_bwe`/`osce_bwe` wrappers are covered only as roundtrip toggles; no runtime decode-path assertion. | Covered for single-stream runtime + multistream smoke in `tests/ctl_api_controls.rs` (`osce_bwe_runtime_decode_path_changes_output_for_silk_only_packets`) | Keep decode-path tests where OSCE-enabled packets run with flag on/off and verify deterministic behavior + no panic/regression. |
| M05 | Reset semantics | Reset behavior for new controls (`ignore_extensions`, `osce_bwe`, `qext`) across base/multistream/projection wrappers. | Covered in `tests/ctl_api_controls.rs` | Keep tests verifying flags are preserved across `reset()` to match upstream state-clear boundaries. |
| M06 | CTL constant parity in behavior | New request constants (4054..4059) are defined/exported, but not validated through runtime request handling paths. | Partial | Add API tests that exercise typed wrappers + equivalent ctl request path assertions (where exposed) for get/set symmetry. |
| M07 | Unsupported surround layouts | `surround_layout()` returns `OPUS_UNIMPLEMENTED` on unsupported mappings (`src/opus/opus_multistream_encoder.rs:1142`, `src/opus/opus_multistream_encoder.rs:1179`) with limited explicit test coverage. | Covered in `tests/opus_multistream_api.rs` (`multistream_surround_unimplemented_paths_match_upstream_c`) | Keep targeted error-path tests for unsupported family/channel combinations to lock expected error codes. |

## P2 (hardening)

| ID | Area | Missing impl detail / missing path | Current status | Required test coverage |
|---|---|---|---|---|
| M08 | Extension robustness | Decoder behavior on malformed/edge extension payloads is not explicitly fuzz/regression covered at API level. | Partial: covered for malformed QEXT extension payloads in `tests/ctl_api_controls.rs` (`malformed_qext_extensions_fallback_matches_ignore_extensions_decode`) | Expand malformed extension packet tests (and/or fuzz corpus seeds) for broader mutation classes and multistream/projection decode paths. |
| M09 | Feature-gated API parity | Feature matrix coverage for `qext`/`osce` combinations is incomplete for new public wrappers. | Covered via CI matrices in `.github/workflows/ci.yml` (`clippy-features`, `test-features`) plus existing `test-dnn` (`osce`) | Keep matrix CI checks for `qext`, `qext+osce`, and `osce` ctl/behavior paths. |

## Existing tests touching this area
- `tests/ctl_api_controls.rs` (now stable after M01 parity fixes)
- `tests/opus_multistream_api.rs` (unsupported surround error-path parity)
- `tests/qext_parity.rs`
- `tests/extensions_repacketizer_parity.rs`

## Upstream parity review notes (2026-02-24)
- `celt_decoder.c` allocates `X` as `C*N`; Rust had fixed `[f32; 1920]` in `src/celt/celt_decoder.rs`, which produced empty Y slices at 96k stereo. Rust now uses dynamic `vec![0.0; (C * N) as usize]`.
- `cwrs.c` enables extra PVQ U rows when QEXT is enabled (`CWRS_EXTRA_ROWS`). Rust table used compact rows only; lookups could go out-of-bounds in `src/celt/cwrs.rs`. Rust now performs table lookup when present and uses a SMALL_FOOTPRINT-equivalent recurrence fallback when indices exceed compact rows.

## Immediate execution order
1. Fill remaining CTL request-path parity checks (M06).
2. Expand malformed-extension hardening breadth (M08) across more mutation classes and decoder wrappers.
3. Keep CI feature matrix coverage for `qext`/`osce` paths green (M09).
