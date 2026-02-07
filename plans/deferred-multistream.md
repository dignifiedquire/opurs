# Deferred: Multistream / Ambisonics / Projection

## What's Missing

The upstream C test `test_opus_projection.c` (394 lines) exercises:
- `opus_projection_encoder_create` / `opus_projection_decoder_create`
- `opus_projection_encode` / `opus_projection_decode`
- Ambisonics channel mapping with projection matrices
- Multistream packet encoding/decoding

## Why It's Deferred

The Rust port does not implement the multistream/projection APIs
(`opus_multistream_encoder`, `opus_projection_encoder`, etc.). These are
higher-level wrappers around the core Opus encoder/decoder that handle
multiple independent streams and channel mapping.

Until the multistream layer is ported, there is nothing to test.

## Prerequisites to Unblock

1. Port `opus_multistream_encoder.c` and `opus_multistream_decoder.c`
2. Port `opus_projection.c` (projection matrix support)
3. Port `mapping_matrix.c` (ambisonics demixing matrices)
4. Then port `test_opus_projection.c` as `tests/opus_projection.rs`

## Upstream Reference

- `opus/tests/test_opus_projection.c` — 394 lines
- Tests 3 main areas: encoder/decoder creation, encode/decode roundtrip,
  ambisonics channel mapping validation
