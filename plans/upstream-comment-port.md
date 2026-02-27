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
- [ ] B5: `src/opus/opus_decoder.rs` decode-state/PLC/FEC comment parity pass.
- [ ] B6: `src/opus/opus_encoder.rs` mode-switch/redundancy/dtx comment parity pass.
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
