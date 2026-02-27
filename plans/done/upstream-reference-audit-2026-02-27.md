# Upstream Reference Audit (2026-02-27)

## Scope

- Reviewed `Upstream C:` anchors across `src/` and `tests/`.
- Verified references against local upstream checkout at `../libopus/opus`.
- Normalized non-resolving or non-canonical anchors to valid `path[:symbol]` form.

## Enforcement Added

- New checker script: `scripts/check_upstream_refs.sh`
- New CI job: `upstream-refs` in `.github/workflows/ci.yml`
  - Clones upstream from `https://gitlab.xiph.org/xiph/opus.git`
  - Fails if any `Upstream C:` file/symbol anchor is invalid.

## Files Corrected In This Pass

- `tests/opus_decode.rs`
- `src/celt/kiss_fft.rs`
- `src/celt/entcode.rs`
- `src/celt/laplace.rs`
- `src/celt/mathops.rs`
- `src/celt/pitch.rs`
- `src/dnn/dred/decoder.rs`
- `src/dnn/nnet.rs`
- `src/dnn/osce.rs`
- `src/dnn/simd/mod.rs`
- `src/dnn/vec.rs`
- `src/opus/extensions.rs`
- `src/opus/opus_encoder.rs`
- `src/opus/repacketizer.rs`
- `DEVELOPMENT.md` (added local checker command)

## Validation Result

- `./scripts/check_upstream_refs.sh` passes with no missing file/symbol anchors.
