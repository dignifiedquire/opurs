# Plan: Group 6 Documentation Version Metadata Drift

## Goal
Remove user-facing and metadata drift for versioning, docs, and release semantics.

## Findings IDs
Open: `none (resolved)`
Closed in this group: `95,96,108,131,133,134,217,222,223,225,226,227,228`

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

## Progress
- 2026-02-26: Closed build-config default parity item `225`:
  - `libopus-sys/build.rs`: default config now defines `DISABLE_DEBUG_FLOAT 1` (upstream-default DNN build behavior).
  - `src/dnn/weights.rs`: Rust weight loading now removes only mirrored `*_weights_float` arrays when matching `*_weights_int8` arrays exist, preserving float-only layers while matching upstream quantized defaults.
  - Verified with `CARGO_TARGET_DIR=target-local cargo nextest run -p opurs --release --features tools-dnn --test osce_nndsp` (20/20 pass).
- 2026-02-26: Closed additional stale docs/metadata findings:
  - `95` / `222`: `opus_get_version_string()` is intentionally project-scoped to `"opurs {crate_version}"` and now tracked as resolved behavior.
  - `226`: confirmed `libopus-sys` exposes `qext` feature and `build.rs` emits `ENABLE_QEXT` when enabled.
- 2026-02-26: Re-audited docs/version drift entries against current tree and closed stale findings:
  - `96`/`227`: `opus_strerror()` now returns upstream-canonical short messages.
  - `108`: crate-level docs target 1.6.1.
  - `134`: source-level provenance comments in active modules now reference 1.6.1.
  - `131`: DNN README baseline text targets 1.6.1.
  - `133`: top-level README parity text targets 1.6.1.
  - `223`: top-level crate docs and `libopus-sys/README.md` align to 1.6.1.
  - `228`: package metadata in `Cargo.toml` targets 1.6.1.
- Group 6 is fully resolved.
