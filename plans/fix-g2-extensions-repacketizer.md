# Plan: Group 2 Extensions And Repacketizer Parity

## Goal
Align packet extension parse/generate/repacketizer semantics with upstream behavior.

## Findings IDs
`none (resolved)`

## Scope
- Repeat-extension iterator semantics and count/parse parity.
- Extension generation compaction behavior and canonical output parity.
- Repacketizer interactions with extension payloads and edge-length handling.

## Execution Order
1. Add deterministic fixtures for repeat markers, out-of-range frame maps, and ext-len boundary cases.
2. Make parser/count behavior byte-for-byte and semantic-output equivalent to upstream.
3. Align generation/repacketization ordering and compaction rules.
4. Gate with differential tests against C outputs.

## Verification
- `CARGO_TARGET_DIR=target-local cargo nextest run -p opurs --cargo-profile=release`
- `CARGO_TARGET_DIR=target-local cargo test -p opurs --test extensions_repacketizer_parity --release`

## Definition Of Done
- Rust accepts/rejects same packets as upstream for covered fixtures.
- Repacketized bytes and extension summaries match upstream expectations.

## Progress
- 2026-02-25: Audited and confirmed upstream parity for all originally listed G2 findings:
  - `src/opus/extensions.rs` now provides iterator-driven count/parse/find semantics with repeat-extension handling, frame-bounded generation, repeat-compaction emission, `_ext` helpers, and iterator control helpers (`reset`, `set_frame_max`).
  - `src/opus/repacketizer.rs` preserves per-input padding metadata and merges source-packet extensions into repacketized output.
  - `tests/extensions_repacketizer_parity.rs` covers repeat roundtrip semantics, repacketizer extension preservation, and tools-gated C-vs-Rust extension API parity.
- 2026-02-24: Added `tests/extensions_repacketizer_parity.rs` with:
  - repeat-extension generate/count/parse roundtrip coverage (`opus_packet_extensions_*`)
  - repacketizer extension-preservation coverage across concatenated packets
  - tools-gated differential coverage against upstream C `opus_packet_extensions_{count,count_ext,parse,parse_ext}` entry points
- 2026-02-24: Added test-only internal re-exports in `src/lib.rs` for extension/repacketizer parity tests.
- 2026-02-24: Verification run:
  - `CARGO_TARGET_DIR=target-local cargo test -p opurs --test extensions_repacketizer_parity --release`
  - `CARGO_TARGET_DIR=target-local cargo test -p opurs --test extensions_repacketizer_parity --features tools --release`
  - `CARGO_TARGET_DIR=target-local cargo test -p opurs --test opus_multistream_packet --features tools --release`
  - `CARGO_TARGET_DIR=target-local cargo clippy -p opurs --all-targets -- -D warnings`
  - `CARGO_TARGET_DIR=target-local cargo clippy -p opurs --all-targets --features tools -- -D warnings`
