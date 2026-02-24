# Plan: Group 3 Public API Surface Parity

## Goal
Close remaining public API coverage gaps versus upstream core/custom/multistream/projection and related controls.

## Findings IDs
`12,43,45,98,104,110,116,119,120,122,163,173,177,178,186,199`

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
