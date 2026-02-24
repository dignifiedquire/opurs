# Plan: Group 6 Documentation Version Metadata Drift

## Goal
Remove user-facing and metadata drift for versioning, docs, and release semantics.

## Findings IDs
`95,96,108,131,133,134,217,223,228`

## Scope
- Version strings and package metadata alignment to current upstream baseline.
- Public docs and README consistency.
- Feature/status docs for DNN/QEXT/OSCE-related behavior exposure.

## Execution Order
1. Enumerate all version-bearing strings and metadata fields.
2. Align crate docs, package metadata, and user-facing helper strings.
3. Add checks to prevent silent drift in future updates.

## Verification
- `CARGO_TARGET_DIR=target-local cargo test -p opurs --release`
- Manual verification of `src/lib.rs`, `Cargo.toml`, `libopus-sys/README.md`.

## Definition Of Done
- No stale previous baseline references remain where 1.6.1 parity is intended.
- Docs/metadata accurately describe current compatibility scope.
