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
  - Benchmark expansion for new API/tooling paths
- Out of scope for this plan:
  - New psychoacoustic tuning/performance optimization passes
  - API redesign unrelated to upstream parity

## Current Status Snapshot (2026-02-20)

- Vector parity baseline:
  - classic vectors: `228/228` passing
  - classic full matrix parity: `1416/1416` passing
  - DNN-only vectors: `264/264` passing
- Current implementation focus:
  - `M1.2` multistream lifecycle/encode/decode wrappers in progress
  - `M1.4` multistream parity-test expansion in progress
- Next ordered slices:
  1. Land and stabilize M1.1 + M1.3 (module wiring + tests).
  2. Implement M1.2 lifecycle/encode/decode wrappers with strict validation parity.
  3. Expand tooling/examples from M2 once core API passes parity tests.

## Milestone Checklist

- [x] Baseline vector parity stabilized before API expansion work
- [x] M1.1 Public API types and module wiring
- [ ] M1.2 Encoder/decoder lifecycle + encode/decode parity
- [x] M1.3 Multistream packet helpers
- [ ] M1.4 Tests (fail-first then green)
- [ ] M1.5 Multistream API surface completion checklist
- [ ] M2 Tooling and examples parity
- [ ] M3 Projection / ambisonics parity
- [ ] M4 Benchmark expansion and baselines

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

### M1.5 - Multistream API Surface Completion Checklist

Close the remaining multistream surface gap explicitly against
`opus_multistream.h`, not just the minimal lifecycle/encode path.

- Encoder API coverage:
  - `opus_multistream_encoder_get_size`
  - `opus_multistream_encoder_create` / `init` / `destroy`
  - `opus_multistream_encode` / `opus_multistream_encode_float`
  - stream-state CTLs required by upstream docs
- Decoder API coverage:
  - `opus_multistream_decoder_get_size`
  - `opus_multistream_decoder_create` / `init` / `destroy`
  - `opus_multistream_decode` / `opus_multistream_decode_float`
  - stream-state CTLs required by upstream docs
- Error/validation parity:
  - mapping length and index bounds
  - stream/coupled-stream count invariants
  - per-call frame-size validation parity

Acceptance criteria:

- A checklist-backed parity test file demonstrates every exposed multistream API
  entry point roundtripping or failing with upstream-compatible error codes.
- `cargo doc` clearly shows multistream APIs as first-class public surface.

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

### M2.5 - Vector Harness Expansion for Multistream/Projection Readiness

- Extend `examples/run_vectors2.rs` planning to cover future multistream and
  projection assets once available.
- Add harness abstractions that separate:
  - suite discovery/loading
  - backend parity checks
  - compliance checks
  so multistream/projection suites can plug in without rewriting the tool.

Acceptance criteria:

- Adding a new suite (e.g. multistream/projection vectors) requires only a
  loader and test-kind registration, not structural harness rewrites.
- CI can enable/disable suites based on asset availability.

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

### M3.5 - Projection API Surface Completion Checklist

Close the projection API gap explicitly against `opus_projection.h`.

- Encoder coverage:
  - `opus_projection_ambisonics_encoder_get_size`
  - `opus_projection_encoder_create` / `init` / `destroy`
  - `opus_projection_encode` / `opus_projection_encode_float`
- Decoder coverage:
  - `opus_projection_decoder_get_size`
  - `opus_projection_decoder_create` / `init` / `destroy`
  - `opus_projection_decode` / `opus_projection_decode_float`
- Matrix helpers and metadata:
  - demixing/mapping matrix getters and validation paths
  - ambisonics channel-order/family handling

Acceptance criteria:

- Projection tests explicitly exercise each exported projection entry point.
- Public docs include working projection setup examples with mapping details.

### M3.4 - CI expansion

- Add CI lane(s):
  - projection parity tests (linux-x86_64 first)
  - optional macOS/windows follow-up once stable

Acceptance criteria:

- Projection lane is required and green before milestone close.

## M4: Benchmark Expansion and Baselines

Add benchmark coverage for the newly introduced multistream/projection and
tooling paths so regressions are visible early.

### M4.1 - Multistream benchmark suite

- Add Criterion benches for multistream encode/decode:
  - `benches/multistream.rs`
- Cover representative matrix:
  - channel counts: 1, 2, 6 (5.1 mapping)
  - frame sizes: 10 ms, 20 ms
  - bitrates: low/medium/high presets
- Measure both:
  - throughput (frames/s)
  - per-frame latency distribution

Acceptance criteria:

- Bench binary runs locally via documented command.
- Results are reproducible with fixed deterministic input fixtures.

### M4.2 - Rust vs upstream comparison benches

- Extend existing comparison benches (or add `benches/multistream_comparison.rs`)
  to compare Rust backend and upstream backend on the same workload.
- Use identical input corpus and encoder settings for both implementations.

Acceptance criteria:

- Benchmark output reports Rust vs upstream delta for each scenario.
- Comparison bench is gated behind `tools` (or `tools-dnn` when required).

### M4.3 - Projection/ambisonics benchmarks

- Add targeted benches for projection encode/decode and mapping matrix steps:
  - matrix apply cost
  - end-to-end projection packet cost

Acceptance criteria:

- Bench coverage exists for all newly added projection public entry points.

### M4.4 - CI perf smoke + reporting

- Add non-blocking benchmark smoke job in CI:
  - verifies benches build and run a short subset
- Add artifact upload for benchmark summaries.
- Keep hard perf gates optional until baseline variance is characterized.

Acceptance criteria:

- CI includes benchmark smoke visibility without flakiness.
- Baseline report format is documented for follow-up optimization work.

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
13. `M4.1` multistream benchmark suite
14. `M4.2` Rust-vs-upstream benchmark comparisons
15. `M4.3` projection/ambisonics benchmark coverage
16. `M4.4` CI benchmark smoke + artifact reporting
17. `M1.5` multistream surface completion checklist
18. `M2.5` vector harness readiness for multistream/projection suites
19. `M3.5` projection surface completion checklist

## Definition of Done

- Public API includes multistream support with upstream-compatible error
  behavior for covered paths.
- Tooling and examples can exercise multichannel encode/decode on both Rust and
  upstream backends.
- Projection API and tests are no longer deferred.
- CI includes parity checks that fail on behavior divergence before release.
- Benchmark coverage exists for all newly added API surfaces and publishes
  repeatable baselines for future optimization work.
