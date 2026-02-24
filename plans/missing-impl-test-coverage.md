# Missing Impl + Test Coverage Matrix

## Purpose
Track implementation gaps and untested runtime paths, with explicit tests required to close each gap.

## P0 (must fix first)

| ID | Area | Missing impl detail / missing path | Current status | Required test coverage |
|---|---|---|---|---|
| M01 | QEXT/CELT runtime | QEXT-enabled runtime hit panics due upstream mismatches in CELT buffer/table handling. | Fixed locally; verified by `tests/ctl_api_controls.rs` with `qext` and `qext,osce` | Keep regression tests that run real extension-bearing encode/decode paths for `OpusEncoder`/`OpusMSEncoder`/`OpusProjectionEncoder` and assert no panic in release mode. |
| M02 | Decoder extension dispatch | Real `ignore_extensions` behavior in decoder loop (`src/opus/opus_decoder.rs:1060`) is not stably covered with real QEXT packets. | Partial/unstable | Add tests proving: `ignore_extensions=true` decode matches extension-stripped decode, and `ignore_extensions=false` diverges when QEXT payload is meaningful. |
| M03 | Multistream/projection extension path | No stable test for extension-aware dispatch in multistream/projection decode wrappers (fan-out paths). | Missing | Add wrapper-level behavior tests that exercise real packet decode through `OpusMSDecoder` and `OpusProjectionDecoder` with extension-bearing packets. |

## P1 (next)

| ID | Area | Missing impl detail / missing path | Current status | Required test coverage |
|---|---|---|---|---|
| M04 | OSCE controls | `set_osce_bwe`/`osce_bwe` wrappers are covered only as roundtrip toggles; no runtime decode-path assertion. | Partial | Add decode-path tests where OSCE-enabled packets run with flag on/off and verify deterministic behavior + no panic/regression. |
| M05 | Reset semantics | Reset behavior for new controls (`ignore_extensions`, `osce_bwe`, `qext`) across base/multistream/projection wrappers is not fully covered. | Missing | Add tests verifying flags after `reset()` for each wrapper match upstream semantics. |
| M06 | CTL constant parity in behavior | New request constants (4054..4059) are defined/exported, but not validated through runtime request handling paths. | Partial | Add API tests that exercise typed wrappers + equivalent ctl request path assertions (where exposed) for get/set symmetry. |
| M07 | Unsupported surround layouts | `surround_layout()` returns `OPUS_UNIMPLEMENTED` on unsupported mappings (`src/opus/opus_multistream_encoder.rs:1142`, `src/opus/opus_multistream_encoder.rs:1179`) with limited explicit test coverage. | Partial | Add targeted error-path tests for unsupported family/channel combinations to lock expected error codes. |

## P2 (hardening)

| ID | Area | Missing impl detail / missing path | Current status | Required test coverage |
|---|---|---|---|---|
| M08 | Extension robustness | Decoder behavior on malformed/edge extension payloads is not explicitly fuzz/regression covered at API level. | Missing | Add malformed extension packet tests (and/or fuzz corpus seeds) asserting no panic and defined error handling. |
| M09 | Feature-gated API parity | Feature matrix coverage for `qext`/`osce` combinations is incomplete for new public wrappers. | Partial | Add matrix CI tests for `qext`, `osce`, and `qext+osce` on ctl/behavior tests. |

## Existing tests touching this area
- `tests/ctl_api_controls.rs` (now stable after M01 parity fixes)
- `tests/qext_parity.rs`
- `tests/extensions_repacketizer_parity.rs`

## Upstream parity review notes (2026-02-24)
- `celt_decoder.c` allocates `X` as `C*N`; Rust had fixed `[f32; 1920]` in `src/celt/celt_decoder.rs`, which produced empty Y slices at 96k stereo. Rust now uses dynamic `vec![0.0; (C * N) as usize]`.
- `cwrs.c` enables extra PVQ U rows when QEXT is enabled (`CWRS_EXTRA_ROWS`). Rust table used compact rows only; lookups could go out-of-bounds in `src/celt/cwrs.rs`. Rust now performs table lookup when present and uses a SMALL_FOOTPRINT-equivalent recurrence fallback when indices exceed compact rows.

## Immediate execution order
1. Fix M01 (panic) so qext behavior tests are reliable.
2. Land stable behavior tests for M02/M03.
3. Fill P1 control/reset/error-path tests.
4. Add malformed-extension hardening (M08) and feature-matrix CI assertions (M09).
