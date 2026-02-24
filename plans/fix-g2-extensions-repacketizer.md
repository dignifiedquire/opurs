# Plan: Group 2 Extensions And Repacketizer Parity

## Goal
Align packet extension parse/generate/repacketizer semantics with upstream behavior.

## Findings IDs
`35,36,37,38,39,40,41,97,98,99,100,115,139`

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
