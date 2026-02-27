# Upstream Comment Port Plan

Last updated: 2026-02-27

## Goal

Port upstream explanatory comments and API documentation from `../libopus/opus`
into Rust sources, file-by-file, while preserving Rust idioms and keeping each
comment anchored to a valid `Upstream C:` reference.

## Rules

- Keep behavior comments faithful to upstream wording and intent.
- Prefer concise docs over raw verbatim blocks.
- Every imported comment block must point to a valid `path[:symbol]` anchor.
- Enforce anchors with `./scripts/check_upstream_refs.sh`.

## Batches

- [x] B1: Reference hygiene + checker/CI enforcement.
- [x] B2: `src/opus/packet.rs` detailed parse/soft-clip comment port.
- [x] B3: `src/opus/repacketizer.rs` inline algorithm/comment parity pass.
- [x] B4: `src/opus/extensions.rs` iterator/repeat semantics full comment parity pass.
- [x] B5: `src/opus/opus_decoder.rs` decode-state/PLC/FEC comment parity pass.
- [x] B6: `src/opus/opus_encoder.rs` mode-switch/redundancy/dtx comment parity pass.
- [ ] B7: `src/celt/*` and `src/silk/*` remaining algorithm comments.
- [ ] B8: `src/dnn/*` comments and model-loading notes alignment.
- [ ] B9: Final audit run + CI green + move plan to `plans/done/`.

## Validation

- `cargo fmt --all --check`
- `cargo clippy -p opurs --all-targets --features tools -- -D warnings`
- `./scripts/check_upstream_refs.sh`

## Progress Notes

- 2026-02-27: Added `Upstream C:` anchors for all public decoder/encoder API
  methods in `impl OpusDecoder` and `impl OpusEncoder` (decode/encode wrappers
  and CTL-style setters/getters), and fixed rustdoc list formatting in
  `src/opus/repacketizer.rs` so clippy passes with `-D warnings`.
- 2026-02-27: Started B7 with a substantive `src/celt/celt_decoder.rs` comment
  pass (ported upstream rationale comments for state validation, init/reset
  semantics, silence handling, global-flag decode ordering, band-energy stage,
  and IMDCT saturation safety), while keeping anchors valid.
- 2026-02-27: Continued B7 in `src/celt/pitch.rs` by porting upstream rationale
  comments for noise-floor handling, lag windowing, coarse/fine pitch search
  stages, pseudo-interpolation refinement, and xcorr loop strategy.
