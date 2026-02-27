# Upstream Test Full-Match Plan

Last updated: 2026-02-27

## Progress

- Completed: `S1`, `S2`, `S3`, `S4`, `S5`
- Remaining: `S6`, `S7`

## Goal

Reach functional test parity with upstream `../libopus/opus/tests` (plus relevant
vector scripts) by porting missing coverage into Rust in small, deterministic
steps with clean, reviewable commits.

## Scope

- In scope:
  - Missing test logic from:
    - `tests/opus_encode_regressions.c`
    - `tests/test_opus_extensions.c`
    - `tests/test_opus_custom.c`
    - `tests/test_opus_dred.c`
  - CI wiring for additional upstream vector suites once assets are available.
- Out of scope:
  - Re-introducing C ABI shim APIs that are intentionally excluded.
  - Non-deterministic long fuzz loops in default CI lanes.

## Commit Strategy

Each step below is a separate commit (or tightly related pair) with:

1. focused code/test changes,
2. `cargo fmt --all`,
3. `cargo clippy --all --all-targets --features tools -- -D warnings`,
4. feature-specific clippy/test commands listed per step.

## Steps

### S1: Encode Regressions Port, Part 1

Port the missing low-friction regressions from
`tests/opus_encode_regressions.c` into a dedicated Rust test module:

- `analysis_overflow`
- `projection_overflow2`
- `projection_overflow3`
- `mscbr_encode_fail10`
- `mscbr_encode_fail`
- `qext_stereo_overflow` (feature-gated)

Acceptance:

- New tests compile and pass locally.
- Existing `tests/opus_encode.rs` remains green.

Commit:

- `test: port upstream encode regressions part 1`

### S2: Encode Regressions Port, Part 2 (Hard Cases)

Port remaining regressions with careful fixture translation and/or reduced but
equivalent deterministic stimuli:

- `celt_ec_internal_error`
- `surround_analysis_uninit`
- `projection_overflow`
- `projection_overflow4` (feature-gated)
- `qext_repacketize_fail` (feature-gated)
- `qext_dred_combination` (feature-gated with `dred`)

Acceptance:

- Each upstream function has a mapped Rust test with `Upstream C:` reference.
- Feature-gated tests run in matching feature lanes.

Commit:

- `test: port upstream encode regressions part 2`

### S3: Extensions Suite Full Port

Add `tests/extensions_upstream_port.rs` mirroring upstream extension tests:

- generate success/zero/no-padding/fail matrices
- parse success/zero/fail matrices
- repeating extension encoding/decoding matrix
- deterministic random parse stress (CI-safe iteration count)
- repacketizer out-range extension preservation scenario

Acceptance:

- New suite passes on `--features tools`.
- Random stress is deterministic and bounded in CI.

Commit:

- `test: port upstream extensions suite`

### S4: Custom API Fuzz Matrix Port

Expand custom coverage from 24-bit parity-only to upstream-style matrix:

- mixed Opus/OpusCustom encode/decode combos
- i16/i24/float encode+decode permutations
- CTL mutation loop parity checks
- corrupt-stream decode result constraints

Acceptance:

- New custom matrix runs in bounded deterministic mode.
- No regressions in existing custom tests.

Commit:

- `test: port upstream custom api matrix`

### S5: DRED Random Parse/Process Port

Add a bounded deterministic Rust port of `test_random_dred`:

- random payload generation
- parse/process invariant checks
- CI-safe default iterations + optional long soak via env var

Acceptance:

- `--features tools-dnn` lane covers this test.
- Optional soak mode documented for local stress.

Commit:

- `test: port upstream dred random parser test`

### S6: Vector Suite Completion in CI

Integrate missing upstream-style vector suites in CI and docs:

- run `qext`, `qext-fuzz`, `dred-opus` suites when assets exist
- keep matrix deterministic across major platforms
- keep cache strategy explicit and stable

Acceptance:

- CI config covers available upstream suite assets.
- Plan/docs updated with exact commands and platform matrix.

Commit:

- `ci: add missing upstream vector suite lanes`

### S7: Closure Audit

Create a final upstream mapping table (file/function -> Rust test location),
mark residual deltas explicitly, and move plan to `plans/done/` when closed.

Acceptance:

- No unresolved required gaps.
- `remaining-plan.md` updated to reflect closure state.

Commit:

- `chore: close upstream full-match test plan`

## Validation Gate for This Plan

- `cargo fmt --all --check`
- `cargo clippy --all --all-targets --features tools -- -D warnings`
- `cargo clippy --all --all-targets --features qext -- -D warnings`
- `cargo clippy --all --all-targets --features tools-dnn -- -D warnings`
- `cargo nextest run -p opurs --cargo-profile=release`
- `cargo nextest run --features tools --cargo-profile=release`
- `cargo nextest run --features tools-dnn --cargo-profile=release`
