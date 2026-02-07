# Phase 4: Integration Layer Safety

**Goal**: Make all code under `src/src/`, `src/externs.rs`, and `src/varargs.rs`
fully safe. Eliminate `externs.rs` entirely. Replace the VarArgs system with
a proper Rust API. This is the final phase for achieving zero unsafe.

**Prerequisites**: Phases 2 and 3 complete (CELT and SILK fully safe).

---

## Current State

| File | Lines | unsafe fn | Key issues |
|------|-------|-----------|------------|
| `opus_encoder.rs` | 3034 | 30 | malloc/free for state, memcpy/memmove, VarArgs CTL |
| `opus_decoder.rs` | 1033 | 15 | malloc/free for state, memcpy, VarArgs CTL |
| `opus.rs` | — | — | Packet parsing (partially safe already) |
| `analysis.rs` | 1582 | 9 | Fn signatures still unsafe, bodies are safe |
| `repacketizer.rs` | 429 | 0 | **Already safe** |
| `mlp/` | — | 0 | **Already safe** |
| `opus_defines.rs` | — | 0 | Constants only |
| `opus_private.rs` | — | — | Alignment helpers, struct layout |
| `externs.rs` | 100 | 8 | malloc/free/memcpy/memmove/memset/memcmp |
| `varargs.rs` | 85 | 0 | VarArgs type — safe but should be replaced |

---

## Stages

### Stage 4.1 — Clean up analysis.rs signatures

`analysis.rs` bodies are already safe but function signatures are still
marked `unsafe fn`. Convert to safe signatures and update all callers.

- [ ] Audit all 9 `unsafe fn` in analysis.rs — confirm bodies are safe
- [ ] Remove `unsafe` from function signatures
- [ ] Update callers in opus_encoder.rs (wrap in safe calls)
- [ ] **Commit**: `refactor: make analysis.rs function signatures safe`

### Stage 4.2 — Eliminate externs.rs usage

Before removing `externs.rs`, all callers must be converted:

- [ ] Audit all callers of `externs::memcpy` → replace with `copy_from_slice` or `clone_from_slice`
- [ ] Audit all callers of `externs::memmove` → replace with `copy_within`
- [ ] Audit all callers of `externs::memset` → replace with `fill(0)` or `fill(value)`
- [ ] Audit all callers of `externs::memcmp` → replace with slice `==` or `.cmp()`
- [ ] Audit all callers of `externs::malloc/calloc/free/realloc`:
  - Encoder state: `malloc` → `Box::new(OpusEncoder::default())`
  - Decoder state: `malloc` → `Box::new(OpusDecoder::default())`
  - Internal buffers: `malloc` → `Vec<T>`
- [ ] Delete `src/externs.rs`
- [ ] Remove `pub mod externs` from `lib.rs`
- [ ] Update any test code that uses `externs`
- [ ] **Commit**: `refactor: eliminate externs.rs, use native Rust allocations`

### Stage 4.3 — Safe opus_decoder.rs

- [ ] Convert `OpusDecoder` state from malloc'd blob to proper Rust struct
  - Currently allocated as raw bytes with manual field offsets
  - Convert to `struct OpusDecoder { ... }` with named fields
  - Embed `CeltDecoder` and SILK decoder state as proper fields
- [ ] Convert `opus_decoder_create` / `opus_decoder_init` to safe constructors
  - `opus_decoder_create` → `OpusDecoder::new(sample_rate, channels) -> Result<Box<OpusDecoder>, OpusError>`
  - Remove raw pointer returns
- [ ] Convert `opus_decode` / `opus_decode_float` to safe functions
  - Input: `&[u8]` packet data
  - Output: `&mut [i16]` or `&mut [f32]` PCM buffer
  - Return: `Result<usize, OpusError>` (number of decoded samples)
- [ ] Convert `opus_decoder_ctl_impl` to safe Rust enum dispatch
  - See Stage 4.5 for VarArgs replacement
- [ ] Maintain backward-compatible unsafe wrappers during transition if needed
- [ ] **Commit**: `refactor: make opus_decoder safe with proper Rust types`

### Stage 4.4 — Safe opus_encoder.rs

This is the single hardest file in the codebase (3034 lines, 30 unsafe fn).

- [ ] Convert `OpusEncoder` state from malloc'd blob to proper Rust struct
  - Same pattern as decoder: named fields, embedded sub-codec state
  - Silk encoder state, CELT encoder state, analysis state as fields
- [ ] Convert `opus_encoder_create` / `opus_encoder_init` to safe constructors
  - `OpusEncoder::new(sample_rate, channels, application) -> Result<Box<OpusEncoder>, OpusError>`
- [ ] Refactor internal helper functions to safe:
  - `opus_encode_native` — main encode path
  - `compute_frame_size` — frame size logic
  - `compute_stereo_width` — stereo analysis
  - `decide_fec` — FEC decision logic
  - `is_digital_silence` — silence detection
  - `pad_frame` — output padding
- [ ] Convert `opus_encode` / `opus_encode_float` to safe functions
  - Input: `&[i16]` or `&[f32]` PCM
  - Output: `&mut [u8]` packet buffer
  - Return: `Result<usize, OpusError>` (encoded packet size)
- [ ] Convert `opus_encoder_ctl_impl` to safe enum dispatch
- [ ] **Commit(s)**: `refactor: make opus_encoder safe` (likely 3-5 commits)

### Stage 4.5 — Replace VarArgs with Rust CTL API

The current VarArgs system is type-safe at runtime but not at compile time.
Replace with a proper Rust enum.

- [ ] Design CTL request enum:
  ```rust
  pub enum EncoderCtl {
      SetBitrate(i32),
      GetBitrate,
      SetComplexity(i32),
      GetComplexity,
      SetVbr(bool),
      // ... all OPUS_SET_*/OPUS_GET_* variants
      ResetState,
  }
  pub enum EncoderCtlResponse {
      Ok,
      Value(i32),
      // ...
  }
  ```
- [ ] Implement `OpusEncoder::ctl(request: EncoderCtl) -> Result<EncoderCtlResponse, OpusError>`
- [ ] Same pattern for `OpusDecoder::ctl()`
- [ ] Same for `OpusCustomEncoder::ctl()` and `OpusCustomDecoder::ctl()`
- [ ] Keep `opus_encoder_ctl!()` macro as backward-compatible wrapper
- [ ] Update all internal CTL calls
- [ ] Remove `src/varargs.rs` when no longer used internally
- [ ] **Commit**: `refactor: replace VarArgs with typed CTL enum API`

### Stage 4.6 — Safe opus.rs and opus_private.rs

- [ ] `opus.rs` — packet parsing functions (partially done already)
  - Verify all functions are safe
  - Clean up any remaining unsafe
- [ ] `opus_private.rs` — alignment helpers
  - Replace manual alignment with `#[repr(align)]` on structs
  - Remove unsafe alignment utilities
- [ ] **Commit**: `refactor: make opus.rs and opus_private.rs fully safe`

### Stage 4.7 — Clean up public API

- [ ] Audit all `pub use` in `lib.rs`
- [ ] Remove deprecated items
- [ ] Ensure public API uses safe Rust types exclusively:
  - No raw pointers in public signatures
  - Slices instead of pointer+length
  - `Result` types for fallible operations
- [ ] Add `#![forbid(unsafe_code)]` to crate root
- [ ] **Commit**: `refactor: add #![forbid(unsafe_code)] — zero unsafe achieved`

---

## Migration Strategy for Public API

To avoid breaking downstream users during the transition:

1. Implement safe internal versions alongside unsafe ones
2. Mark old unsafe APIs as `#[deprecated]`
3. Update all tests and tools to use new safe API
4. Remove deprecated APIs in a final cleanup

---

## Definition of Done

- [ ] `#![forbid(unsafe_code)]` in `lib.rs` — compiles successfully
- [ ] `externs.rs` deleted
- [ ] `varargs.rs` deleted (or emptied to just backward-compat macro)
- [ ] All encoder/decoder state uses proper Rust structs
- [ ] Public API uses only safe Rust types
- [ ] All tests pass (cargo test + vector tests)
- [ ] Clippy clean, formatted
