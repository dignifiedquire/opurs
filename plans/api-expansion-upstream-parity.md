# API Expansion Plan: Upstream Parity (Multichannel + Tooling)

## Goal

Expand the public API, tooling, and examples to cover missing upstream-facing
functionality, with multichannel (multistream) support as the first-class
deliverable.

Primary target headers:

- `opus/include/opus.h`
- `opus/include/opus_multistream.h`
- `opus/include/opus_projection.h`

## Scope Summary

- In scope:
  - Multistream encoder/decoder API in pure Rust public surface
  - Multistream packet helpers
  - `opus_demo`-equivalent tooling support for multichannel workflows
  - Upstream parity tests (fail-first then green)
  - Projection/ambisonics API and tests (phase after multistream)
- Out of scope for this plan:
  - New psychoacoustic tuning/performance optimization passes
  - API redesign unrelated to upstream parity

## Milestones

## M1: Multistream Core API

Deliver complete, usable multichannel wrappers and packet helpers with
upstream-compatible behavior.

### M1.1 - Public API types and module wiring

- Add:
  - `src/opus/opus_multistream_encoder.rs`
  - `src/opus/opus_multistream_decoder.rs`
- Re-export from `src/lib.rs`:
  - `OpusMSEncoder`
  - `OpusMSDecoder`
  - multistream packet helpers
- Add typed config struct(s) for mapping/stream layout to avoid raw arg soup.

Acceptance criteria:

- `cargo check` succeeds with and without `tools`.
- API docs render and public items are discoverable in crate docs.

### M1.2 - Encoder/decoder lifecycle + encode/decode parity

- Implement create/init/reset/destroy and frame encode/decode parity with
  upstream constraints:
  - stream counts
  - coupled stream counts
  - mapping table validation
- Add parity CTLs needed for practical use:
  - bitrate
  - complexity
  - VBR + constrained VBR
  - DTX/FEC and packet loss settings
  - lookahead/final range

Acceptance criteria:

- API returns upstream-compatible error codes for invalid configs.
- Roundtrip encode/decode works for 1..N channels under deterministic tests.

### M1.3 - Multistream packet helpers

- Implement and export:
  - `opus_multistream_packet_pad`
  - `opus_multistream_packet_unpad`
- Ensure parity with upstream behavior on malformed inputs.

Acceptance criteria:

- Differential tests against `libopus-sys` pass for pad/unpad edge cases.

### M1.4 - Tests (fail-first then green)

- Add:
  - `tests/opus_multistream_api.rs`
  - `tests/opus_multistream_packet.rs`
- Include:
  - constructor validation matrix
  - roundtrip parity tests
  - negative/error-code parity tests
- Run with `--test-threads=1` where C harness shared state is involved.

Acceptance criteria:

- New tests fail before implementation and pass after.
- CI job added for multistream parity smoke run on linux-x86_64.

## M2: Tooling and Examples Parity

Bring `tools` UX closer to upstream behavior for multichannel workflows.

### M2.1 - Extend demo input model beyond mono/stereo

- Update `src/tools/demo/input.rs`:
  - replace hard-coded `Channels::{Mono,Stereo}` assumptions
  - add multistream config inputs:
    - `streams`
    - `coupled_streams`
    - `mapping`
    - `mapping_family` (as needed)

Acceptance criteria:

- Parser accepts and validates multichannel arguments deterministically.

### M2.2 - Backend trait support for multistream

- Extend `src/tools/demo/backend.rs` with multistream trait methods for:
  - Rust backend
  - upstream libopus backend
- Keep capability parity checks explicit (no hidden fallback to stereo path).

Acceptance criteria:

- Tool can execute identical multistream encode/decode pipelines on both
  backends for comparison.

### M2.3 - Example parity

- Extend `examples/opus_demo.rs` and `src/tools/demo/mod.rs` with multistream
  mode.
- Add `examples/multistream_demo.rs` (or equivalent subcommand) showing:
  - custom mapping usage
  - roundtrip encode/decode

Acceptance criteria:

- Example commands documented and passing in CI smoke checks.

### M2.4 - Tooling tests

- Add integration tests for CLI parsing and deterministic IO checks:
  - `tests/tools_multistream.rs` (gated by `tools`)

Acceptance criteria:

- `cargo test --features tools` includes multistream tooling checks.

## M3: Projection / Ambisonics Parity

Port and expose projection APIs after multistream base is stable.

### M3.1 - Projection encoder/decoder wrappers

- Add modules:
  - `src/opus/opus_projection_encoder.rs`
  - `src/opus/opus_projection_decoder.rs`
- Re-export public projection API from `src/lib.rs`.

Acceptance criteria:

- Basic create/init/encode/decode/reset lifecycle works with validation.

### M3.2 - Mapping matrix support

- Port mapping-matrix logic from upstream:
  - `opus/src/mapping_matrix.c`
- Add safe Rust matrix representation and conversion utilities.

Acceptance criteria:

- Matrix build/validation parity against upstream for tested configurations.

### M3.3 - Port upstream projection tests

- Port `opus/tests/test_opus_projection.c` into Rust parity tests:
  - `tests/opus_projection.rs`

Acceptance criteria:

- Core projection/ambisonics test scenarios are covered and passing.

### M3.4 - CI expansion

- Add CI lane(s):
  - projection parity tests (linux-x86_64 first)
  - optional macOS/windows follow-up once stable

Acceptance criteria:

- Projection lane is required and green before milestone close.

## Issue Breakdown (PR-Sized)

1. `M1.1` API scaffolding + exports
2. `M1.2` multistream encode/decode core
3. `M1.3` packet pad/unpad parity
4. `M1.4` multistream parity tests + CI smoke lane
5. `M2.1` demo input model multichannel refactor
6. `M2.2` backend trait + Rust/C implementations
7. `M2.3` examples and CLI support
8. `M2.4` tooling integration tests
9. `M3.1` projection wrappers
10. `M3.2` mapping matrix port
11. `M3.3` projection parity tests
12. `M3.4` CI required lanes

## Definition of Done

- Public API includes multistream support with upstream-compatible error
  behavior for covered paths.
- Tooling and examples can exercise multichannel encode/decode on both Rust and
  upstream backends.
- Projection API and tests are no longer deferred.
- CI includes parity checks that fail on behavior divergence before release.
