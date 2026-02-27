# Plan: Group 1 QEXT Correctness Blockers

## Goal
Close all QEXT bitstream/PLC/sizing correctness gaps from `diff_review.md`.

## Findings IDs
`6,111,166,175,229` (all resolved)

## Scope
- CELT encoder QEXT scaling, prefilter memory/layout, overlap history sizing.
- CELT decoder QEXT PLC, pitch search/downsample scaling, deemphasis, synthesis denormalisation.
- QEXT state init/reset behavior and 96 kHz mode guardrails.
- Opus encoder QEXT payload provisioning path.

## Execution Order
1. Land fail-first tests for each QEXT subgroup (encoder prefilter, decoder PLC, synthesis/deemphasis, reset semantics).
2. Fix core scaling primitives and sizing constants first (`QEXT_SCALE` paths), then call-site wiring.
3. Fix encoder payload provisioning (`qext_payload`/`qext_bytes`) and integration flow.
4. Re-run full vector matrix and focused QEXT differential tests.

## Verification
- `CARGO_TARGET_DIR=target-local cargo nextest run -p opurs --features "tools,qext" --cargo-profile=release`
- `CARGO_TARGET_DIR=target-local cargo run --release --features "tools,qext" --example run_vectors2 -- --suite classic --matrix full --mode parity opus_newvectors`

## Definition Of Done
- All listed IDs are either fixed in code or intentionally deferred with explicit rationale.
- QEXT parity tests pass consistently in CI and local matrix runs.

## Progress
- 2026-02-24: Resolved IDs `24,26,27,28,29,30,31,32,33,34,42,52,88,150`.
- 2026-02-24: Resolved IDs `1,3,4,5,14,15,16,17,21,22,23,25`.
- 2026-02-24: Resolved IDs `6,111,166,175,229`.
- Remaining open IDs: `none`.
- Implemented:
  - Mode-derived `qext_scale` initialization in CELT encoder/decoder.
  - QEXT old-band history clearing in encoder/decoder reset paths.
  - Decoder validation updates for 96 kHz QEXT modes and scaled max-period checks.
  - Encoder overlap history buffer sizing raised for 96 kHz overlap (`2*240`).
  - Decoder synthesis/prefilter/deemphasis scratch allocations switched to runtime `N`/`overlap` sizes.
  - Decoder energy finalisation now skips `oldEBands` update when QEXT payload is present (upstream `NULL oldBandE` parity).
  - Decoder deemphasis now implements the upstream QEXT/custom IIR branch (`coef[1]`/`coef[3]` path) and has dedicated unit coverage.
  - Decoder synthesis now receives and applies QEXT denormalisation bands (`qext_mode`, `qext_oldBandE`, `qext_end`) to match upstream reconstruction flow.
  - Decoder/encoder state capacities updated for 96 kHz QEXT (`decode_mem` max overlap and `delay_buffer` max encoder buffer).
  - `resampling_factor` now supports `96000 -> 1` under `qext`, matching upstream.
  - Fade helpers now clamp increment with `max(1, 48000/Fs)` to preserve 96 kHz overlap math parity.
  - `run_prefilter` now uses qext-scaled `max_period/min_period` state layout, tone/pitch scaling parity, and scaled overlap sourcing.
  - Decoder PLC now uses runtime `decode_buffer_size/max_period` throughout loss concealment and pitch search, including QEXT-scaled `pitch_downsample` factor and returned lag.
  - Deep PLC selection now gates out 96 kHz modes in loss concealment, matching upstream guard.
  - Stereo `quant_band` QEXT bit redistribution now computes `qext_extra` on the correct branch basis (`mbits` vs `sbits`).
  - `pitch_downsample` now supports upstream factorized downsampling (including factor-4 QEXT path).
  - Opus encoder now provisions QEXT payload bytes for CELT-only mode, passes `qext_payload`/`qext_bytes` to CELT, and emits QEXT extension ID `124` via packet extension padding.
  - SILK resampler now mirrors upstream QEXT 96 kHz tables/rate mapping/input checks, including encoder/decoder delay matrices and 96-sample delay buffer capacity.
  - Repacketizer non-pad extension sizing now matches upstream `pad_amount = ext_len + (ext_len ? (ext_len+253)/254 : 1)`.
  - QEXT encode stability fixes: dynamic `quant_all_bands` norm buffer sizing (`C * norm_size`) and guarded `quant_fine_energy` shift bounds to prevent overflow panics on invalid bit counts.
  - Tools parity harness build fix: updated `tests/pitch_c_parity.rs` for the `pitch_downsample(..., factor, arch)` signature.
- Added unit coverage for the above in `src/celt/celt_encoder.rs` and `src/celt/celt_decoder.rs`.
