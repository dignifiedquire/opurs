# Phase 4: Integration Layer Safety & Idiomatic Public API

**Goal**: Make all code under `src/src/`, `src/externs.rs`, and `src/varargs.rs`
fully safe. Eliminate `externs.rs` entirely. Replace the C-style public API
with an idiomatic Rust API modeled on the
[`opus` crate](https://docs.rs/opus/latest/opus/) (high-level bindings by
meh), with documentation aligned to the
[upstream Opus docs](https://opus-codec.org/docs/opus_api-1.3.1/).

**Prerequisites**: Phases 2 and 3 complete (CELT and SILK fully safe).

---

## Target Public API

The final API mirrors the `opus` crate's design: typed enums instead of
integer constants, `Result<T, Error>` instead of error codes, methods on
owned structs instead of free functions with raw pointers, and a `packet`
module for stateless packet inspection.

### Type Overview

```rust
// ── Error handling ──────────────────────────────────────────────

/// Opus error code.
///
/// See: https://opus-codec.org/docs/opus_api-1.3.1/group__opus__errorcodes.html
#[repr(i32)]
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum ErrorCode {
    BadArg          = -1,
    BufferTooSmall  = -2,
    InternalError   = -3,
    InvalidPacket   = -4,
    Unimplemented   = -5,
    InvalidState    = -6,
    AllocFail       = -7,
    Unknown         = -8,
}

/// Opus error type wrapping an [`ErrorCode`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Error(pub ErrorCode);

impl std::fmt::Display for Error { ... }
impl std::error::Error for Error {}

/// Convenience alias used throughout the public API.
pub type Result<T> = std::result::Result<T, Error>;

// ── Enums (replace OPUS_* integer constants) ────────────────────

/// Intended application profile.
///
/// See: https://opus-codec.org/docs/opus_api-1.3.1/group__opus__ctlvalues.html
#[repr(i32)]
pub enum Application {
    /// Best for VoIP/videoconference — prioritizes intelligibility.
    Voip     = 2048,
    /// Best for broadcast/hi-fi — decoded audio closest to input.
    Audio    = 2049,
    /// Only when lowest latency is the primary goal.
    LowDelay = 2051,
}

/// Channel layout.
#[repr(i32)]
pub enum Channels {
    Mono   = 1,
    Stereo = 2,
}

/// Audio bandwidth.
///
/// See: https://opus-codec.org/docs/opus_api-1.3.1/group__opus__ctlvalues.html
#[repr(i32)]
pub enum Bandwidth {
    /// Auto/default — let the encoder decide.
    Auto          = -1000,
    /// 4 kHz bandpass.
    Narrowband    = 1101,
    /// 6 kHz bandpass.
    Mediumband    = 1102,
    /// 8 kHz bandpass.
    Wideband      = 1103,
    /// 12 kHz bandpass.
    Superwideband = 1104,
    /// 20 kHz bandpass.
    Fullband      = 1105,
}

/// Bitrate configuration.
pub enum Bitrate {
    /// Explicit bitrate in bits/second.
    Bits(i32),
    /// Maximum bitrate allowed by the packet size.
    Max,
    /// Let the encoder decide (not recommended).
    Auto,
}

/// Signal type hint for encoder mode selection.
///
/// See: https://opus-codec.org/docs/opus_api-1.3.1/group__opus__ctlvalues.html
#[repr(i32)]
pub enum Signal {
    Auto  = -1000,
    /// Bias toward LPC or Hybrid modes (speech).
    Voice = 3001,
    /// Bias toward MDCT modes (music).
    Music = 3002,
}

/// Expert frame duration control.
///
/// See: https://opus-codec.org/docs/opus_api-1.3.1/group__opus__ctlvalues.html
#[repr(i32)]
pub enum FrameSize {
    /// Select from the argument (default).
    Arg   = 5000,
    Ms2_5 = 5001,
    Ms5   = 5002,
    Ms10  = 5003,
    Ms20  = 5004,
    Ms40  = 5005,
    Ms60  = 5006,
    Ms80  = 5007,
    Ms100 = 5008,
    Ms120 = 5009,
}
```

### Encoder

```rust
/// Opus encoder.
///
/// See: https://opus-codec.org/docs/opus_api-1.3.1/group__opus__encoder.html
pub struct Encoder { /* ... */ }

impl Encoder {
    // ── Construction ──────────────────────────────────────────
    /// Create and initialize an encoder.
    ///
    /// `sample_rate` must be one of 8000, 12000, 16000, 24000, or 48000.
    pub fn new(sample_rate: u32, channels: Channels, application: Application) -> Result<Encoder>;

    // ── Encoding ──────────────────────────────────────────────
    /// Encode an Opus frame from 16-bit PCM.
    ///
    /// `input` must contain exactly `frame_size * channels` samples.
    /// Returns the number of bytes written to `output`.
    pub fn encode(&mut self, input: &[i16], output: &mut [u8]) -> Result<usize>;

    /// Encode an Opus frame from floating-point PCM (±1.0 range).
    pub fn encode_float(&mut self, input: &[f32], output: &mut [u8]) -> Result<usize>;

    /// Convenience: encode into a new `Vec<u8>`.
    pub fn encode_vec(&mut self, input: &[i16], max_size: usize) -> Result<Vec<u8>>;

    /// Convenience: encode float into a new `Vec<u8>`.
    pub fn encode_vec_float(&mut self, input: &[f32], max_size: usize) -> Result<Vec<u8>>;

    // ── Generic CTLs ──────────────────────────────────────────
    /// Reset codec state to freshly initialized.
    pub fn reset_state(&mut self) -> Result<()>;

    /// Get the final state of the codec's entropy coder.
    pub fn get_final_range(&mut self) -> Result<u32>;

    /// Get the encoder's configured bandpass.
    pub fn get_bandwidth(&mut self) -> Result<Bandwidth>;

    /// Get the sample rate the encoder was initialized with.
    pub fn get_sample_rate(&mut self) -> Result<u32>;

    /// Get the total look-ahead in samples.
    pub fn get_lookahead(&mut self) -> Result<i32>;

    /// Get whether the encoder is in DTX mode.
    pub fn get_in_dtx(&mut self) -> Result<bool>;

    /// Set/get phase inversion for intensity stereo.
    pub fn set_phase_inversion_disabled(&mut self, disabled: bool) -> Result<()>;
    pub fn get_phase_inversion_disabled(&mut self) -> Result<bool>;

    // ── Encoder-specific CTLs ─────────────────────────────────
    pub fn set_complexity(&mut self, complexity: i32) -> Result<()>;
    pub fn get_complexity(&mut self) -> Result<i32>;

    pub fn set_bitrate(&mut self, bitrate: Bitrate) -> Result<()>;
    pub fn get_bitrate(&mut self) -> Result<Bitrate>;

    pub fn set_vbr(&mut self, enabled: bool) -> Result<()>;
    pub fn get_vbr(&mut self) -> Result<bool>;

    pub fn set_vbr_constraint(&mut self, constrained: bool) -> Result<()>;
    pub fn get_vbr_constraint(&mut self) -> Result<bool>;

    pub fn set_force_channels(&mut self, channels: Option<Channels>) -> Result<()>;
    pub fn get_force_channels(&mut self) -> Result<Option<Channels>>;

    pub fn set_max_bandwidth(&mut self, bandwidth: Bandwidth) -> Result<()>;
    pub fn get_max_bandwidth(&mut self) -> Result<Bandwidth>;

    pub fn set_bandwidth(&mut self, bandwidth: Bandwidth) -> Result<()>;

    pub fn set_signal(&mut self, signal: Signal) -> Result<()>;
    pub fn get_signal(&mut self) -> Result<Signal>;

    pub fn set_application(&mut self, application: Application) -> Result<()>;
    pub fn get_application(&mut self) -> Result<Application>;

    pub fn set_inband_fec(&mut self, enabled: bool) -> Result<()>;
    pub fn get_inband_fec(&mut self) -> Result<bool>;

    pub fn set_packet_loss_perc(&mut self, percentage: i32) -> Result<()>;
    pub fn get_packet_loss_perc(&mut self) -> Result<i32>;

    pub fn set_dtx(&mut self, enabled: bool) -> Result<()>;
    pub fn get_dtx(&mut self) -> Result<bool>;

    pub fn set_lsb_depth(&mut self, depth: i32) -> Result<()>;
    pub fn get_lsb_depth(&mut self) -> Result<i32>;

    pub fn set_expert_frame_duration(&mut self, duration: FrameSize) -> Result<()>;
    pub fn get_expert_frame_duration(&mut self) -> Result<FrameSize>;

    pub fn set_prediction_disabled(&mut self, disabled: bool) -> Result<()>;
    pub fn get_prediction_disabled(&mut self) -> Result<bool>;
}
```

### Decoder

```rust
/// Opus decoder.
///
/// See: https://opus-codec.org/docs/opus_api-1.3.1/group__opus__decoder.html
pub struct Decoder { /* ... */ }

impl Decoder {
    // ── Construction ──────────────────────────────────────────
    /// Create and initialize a decoder.
    ///
    /// `sample_rate` must be one of 8000, 12000, 16000, 24000, or 48000.
    pub fn new(sample_rate: u32, channels: Channels) -> Result<Decoder>;

    // ── Decoding ──────────────────────────────────────────────
    /// Decode an Opus packet.
    ///
    /// Pass an empty slice for `input` to indicate packet loss (PLC).
    /// `fec` enables forward error correction from a subsequent packet.
    /// Returns the number of decoded samples per channel.
    pub fn decode(&mut self, input: &[u8], output: &mut [i16], fec: bool) -> Result<usize>;

    /// Decode an Opus packet to floating-point output.
    pub fn decode_float(&mut self, input: &[u8], output: &mut [f32], fec: bool) -> Result<usize>;

    /// Get the number of samples per channel in a packet.
    pub fn get_nb_samples(&self, packet: &[u8]) -> Result<usize>;

    // ── Generic CTLs ──────────────────────────────────────────
    pub fn reset_state(&mut self) -> Result<()>;
    pub fn get_final_range(&mut self) -> Result<u32>;
    pub fn get_bandwidth(&mut self) -> Result<Bandwidth>;
    pub fn get_sample_rate(&mut self) -> Result<u32>;
    pub fn get_in_dtx(&mut self) -> Result<bool>;
    pub fn set_phase_inversion_disabled(&mut self, disabled: bool) -> Result<()>;
    pub fn get_phase_inversion_disabled(&mut self) -> Result<bool>;

    // ── Decoder-specific CTLs ─────────────────────────────────
    /// Set decoder gain adjustment in Q8 dB units (±32768).
    pub fn set_gain(&mut self, gain: i32) -> Result<()>;
    pub fn get_gain(&mut self) -> Result<i32>;

    /// Get duration of the last decoded packet in samples.
    pub fn get_last_packet_duration(&mut self) -> Result<u32>;

    /// Get pitch period of the last decoded frame, or 0 if unavailable.
    pub fn get_pitch(&mut self) -> Result<i32>;
}
```

### Repacketizer

```rust
/// Opus packet repacketizer.
///
/// See: https://opus-codec.org/docs/opus_api-1.3.1/group__opus__repacketizer.html
pub struct Repacketizer { /* ... */ }

impl Repacketizer {
    pub fn new() -> Repacketizer;

    /// Shortcut: combine several smaller packets into one larger one.
    pub fn combine(&mut self, input: &[&[u8]], output: &mut [u8]) -> Result<usize>;

    /// Begin a repacketization session, returning a builder.
    pub fn begin(&mut self) -> RepacketizerState<'_>;
}

/// In-progress repacketization session.
pub struct RepacketizerState<'rp> { /* ... */ }

impl<'rp> RepacketizerState<'rp> {
    /// Add a packet to the current session.
    pub fn cat(&mut self, packet: &[u8]) -> Result<()>;

    /// Get the number of frames accumulated so far.
    pub fn get_nb_frames(&self) -> usize;

    /// Output the merged packet.
    pub fn out(&self, output: &mut [u8]) -> Result<usize>;

    /// Output a subrange of accumulated frames.
    pub fn out_range(&self, begin: usize, end: usize, output: &mut [u8]) -> Result<usize>;
}
```

### SoftClip

```rust
/// Soft-clipping state for float signals.
///
/// See: `opus_pcm_soft_clip` in
/// https://opus-codec.org/docs/opus_api-1.3.1/group__opus__decoder.html
pub struct SoftClip { /* ... */ }

impl SoftClip {
    /// Initialize soft-clipping state for the given channel count.
    pub fn new(channels: Channels) -> SoftClip;

    /// Apply soft-clipping to a float signal in-place (range ±1.0).
    pub fn apply(&mut self, signal: &mut [f32]);
}
```

### `packet` module

Stateless packet inspection functions, matching the upstream "Opus Decoder"
group's packet utility functions.

```rust
/// Stateless packet inspection utilities.
///
/// See: https://opus-codec.org/docs/opus_api-1.3.1/group__opus__decoder.html
pub mod packet {
    use super::*;

    /// A parsed Opus packet.
    pub struct Packet<'a> {
        pub toc: u8,
        pub frames: Vec<&'a [u8]>,
        pub payload_offset: usize,
    }

    /// Get the bandwidth of an Opus packet from its ToC byte.
    pub fn get_bandwidth(packet: &[u8]) -> Result<Bandwidth>;

    /// Get the number of channels from a packet's ToC byte.
    pub fn get_nb_channels(packet: &[u8]) -> Result<Channels>;

    /// Get the number of frames in a packet.
    pub fn get_nb_frames(packet: &[u8]) -> Result<usize>;

    /// Get the total number of samples in a packet at the given sample rate.
    pub fn get_nb_samples(packet: &[u8], sample_rate: u32) -> Result<usize>;

    /// Get the number of samples per frame from a packet's ToC byte.
    pub fn get_samples_per_frame(packet: &[u8], sample_rate: u32) -> Result<usize>;

    /// Parse a packet into individual frames.
    pub fn parse(packet: &[u8]) -> Result<Packet<'_>>;

    /// Pad a packet to a larger size.
    pub fn pad(packet: &mut [u8], len: usize, new_len: usize) -> Result<()>;

    /// Remove padding from a packet. Returns the unpadded length.
    pub fn unpad(packet: &mut [u8]) -> Result<usize>;
}
```

### Top-level

```rust
/// Return the libopus version string.
pub fn version() -> &'static str;
```

---

## Current State

| File | Lines | unsafe fn | Key issues |
|------|-------|-----------|------------|
| `opus_encoder.rs` | 3034 | 30 | malloc/free for state, memcpy/memmove, VarArgs CTL |
| `opus_decoder.rs` | 1033 | 15 | malloc/free for state, memcpy, VarArgs CTL |
| `opus.rs` | — | — | Packet parsing (partially safe already) |
| `analysis.rs` | 1582 | 9 | Fn signatures still unsafe, bodies are safe |
| `repacketizer.rs` | 429 | 0 | **Already safe** — needs API wrapper only |
| `mlp/` | — | 0 | **Already safe** |
| `opus_defines.rs` | — | 0 | Constants only — replace with enums |
| `opus_private.rs` | — | — | Alignment helpers, struct layout |
| `externs.rs` | 100 | 8 | malloc/free/memcpy/memmove/memset/memcmp |
| `varargs.rs` | 85 | 0 | VarArgs type — replace with typed methods |

---

## Stages

### Stage 4.1 — Define public types (`src/api/`)

Create the type-safe public API types that the rest of the stages will
target. This can be done before any internal refactoring.

- [ ] Create `src/api/` module directory
- [ ] `src/api/error.rs` — `ErrorCode` enum, `Error` struct, `Result<T>` alias
  - `ErrorCode` is `#[repr(i32)]`, implements `Display`, `std::error::Error`
  - Conversion from raw `i32` error codes via `TryFrom`
  - `Error::description()` delegates to `opus_strerror`
- [ ] `src/api/enums.rs` — `Application`, `Channels`, `Bandwidth`, `Bitrate`,
  `Signal`, `FrameSize` enums
  - Each `#[repr(i32)]` where applicable
  - `TryFrom<i32>` for each (mapping from OPUS_* constants)
  - `Into<i32>` for passing to internal functions
- [ ] `src/api/mod.rs` — re-export all types
- [ ] Wire `pub mod api` into `lib.rs`, re-export types at crate root
- [ ] **Commit**: `refactor: define typed public API enums and error types`

### Stage 4.2 — Clean up analysis.rs signatures

`analysis.rs` bodies are already safe but function signatures are still
marked `unsafe fn`. Convert to safe signatures and update all callers.

- [ ] Audit all 9 `unsafe fn` in analysis.rs — confirm bodies are safe
- [ ] Remove `unsafe` from function signatures
- [ ] Update callers in opus_encoder.rs (wrap in safe calls)
- [ ] **Commit**: `refactor: make analysis.rs function signatures safe`

### Stage 4.3 — Eliminate externs.rs usage

Before removing `externs.rs`, all callers must be converted:

- [ ] Audit all callers of `externs::memcpy` → replace with `copy_from_slice`
  or `clone_from_slice`
- [ ] Audit all callers of `externs::memmove` → replace with `copy_within`
- [ ] Audit all callers of `externs::memset` → replace with `fill(0)` or
  `fill(value)`
- [ ] Audit all callers of `externs::memcmp` → replace with slice `==`
- [ ] Audit all callers of `externs::malloc/calloc/free/realloc`:
  - Encoder state: `malloc` → `Box::new(OpusEncoder::default())`
  - Decoder state: `malloc` → `Box::new(OpusDecoder::default())`
  - Internal buffers: `malloc` → `Vec<T>`
- [ ] Delete `src/externs.rs`
- [ ] Remove `pub mod externs` from `lib.rs`
- [ ] Update any test code that still uses `externs`
- [ ] **Commit**: `refactor: eliminate externs.rs, use native Rust allocations`

### Stage 4.4 — Safe opus_decoder.rs

- [ ] Convert `OpusDecoder` from malloc'd blob to proper Rust struct
  - Named fields, embedded `CeltDecoder` and SILK state
- [ ] Implement `Decoder` wrapper struct (public API)
  - `Decoder::new(sample_rate, channels)` — validates args, creates inner state
  - `Decoder::decode(input, output, fec)` — delegates to safe internals
  - `Decoder::decode_float(input, output, fec)`
  - `Decoder::get_nb_samples(packet)`
  - All CTL methods as typed methods (no more VarArgs for public use)
- [ ] Implement `Drop` for `Decoder` (replaces `opus_decoder_destroy`)
- [ ] Deprecate old C-style free functions (`opus_decoder_create`, etc.)
- [ ] **Commit**: `refactor: safe Decoder with idiomatic Rust API`

### Stage 4.5 — Safe opus_encoder.rs

The single hardest file (3034 lines, 30 unsafe fn).

- [ ] Convert `OpusEncoder` from malloc'd blob to proper Rust struct
  - Named fields: SILK state, CELT state, analysis state
- [ ] Implement `Encoder` wrapper struct (public API)
  - `Encoder::new(sample_rate, channels, application)`
  - `Encoder::encode(input, output)` / `encode_float`
  - `Encoder::encode_vec(input, max_size)` / `encode_vec_float`
  - All CTL methods as typed methods:
    - `set_bitrate(Bitrate)` / `get_bitrate() -> Bitrate`
    - `set_complexity(i32)` / `get_complexity() -> i32`
    - `set_vbr(bool)` / `get_vbr() -> bool`
    - `set_bandwidth(Bandwidth)` / `get_bandwidth() -> Bandwidth`
    - etc. (full list in Target API section above)
- [ ] Implement `Drop` for `Encoder` (replaces `opus_encoder_destroy`)
- [ ] Deprecate old C-style free functions
- [ ] Refactor internal helpers to safe:
  - `opus_encode_native` — main encode path
  - `compute_frame_size`, `compute_stereo_width`
  - `decide_fec`, `is_digital_silence`, `pad_frame`
- [ ] **Commit(s)**: `refactor: safe Encoder with idiomatic Rust API`
  (likely 3-5 commits)

### Stage 4.6 — Repacketizer & SoftClip wrappers

The `OpusRepacketizer` is already safe internally. Add the public wrapper.

- [ ] Implement `Repacketizer` wrapper with `begin()` → `RepacketizerState`
  builder pattern
- [ ] Implement `SoftClip` struct wrapping `opus_pcm_soft_clip` state
- [ ] **Commit**: `refactor: add Repacketizer and SoftClip public API wrappers`

### Stage 4.7 — `packet` module

Stateless packet inspection — most functions are already safe or nearly so.

- [ ] Create `src/api/packet.rs` module
- [ ] Wrap `opus_packet_get_bandwidth` → `packet::get_bandwidth`
  (return `Bandwidth` enum instead of raw i32)
- [ ] Wrap `opus_packet_get_nb_channels` → `packet::get_nb_channels`
  (return `Channels` enum)
- [ ] Wrap `opus_packet_get_nb_frames` → `packet::get_nb_frames`
- [ ] Wrap `opus_packet_get_nb_samples` → `packet::get_nb_samples`
- [ ] Wrap `opus_packet_get_samples_per_frame` → `packet::get_samples_per_frame`
- [ ] Wrap `opus_packet_parse` → `packet::parse` returning `Packet` struct
  with borrowed frame slices
- [ ] Wrap `opus_packet_pad` / `opus_packet_unpad` → `packet::pad` /
  `packet::unpad`
- [ ] **Commit**: `refactor: add packet inspection module with typed returns`

### Stage 4.8 — Safe opus.rs and opus_private.rs

- [ ] `opus.rs` — verify all packet functions are safe, clean up remaining
  unsafe
- [ ] `opus_private.rs` — replace manual alignment with `#[repr(align)]`
- [ ] Remove unsafe alignment utilities
- [ ] **Commit**: `refactor: make opus.rs and opus_private.rs fully safe`

### Stage 4.9 — Replace VarArgs internally

After stages 4.4-4.5 provide the public typed API, the internal VarArgs
usage can be cleaned up.

- [ ] Convert internal `opus_encoder_ctl_impl` dispatching to direct method
  calls
- [ ] Convert internal `opus_decoder_ctl_impl` dispatching to direct method
  calls
- [ ] Keep `opus_encoder_ctl!()` / `opus_decoder_ctl!()` macros as
  backward-compat wrappers (delegating to new typed methods), mark
  `#[deprecated]`
- [ ] Remove `src/varargs.rs` when no longer used anywhere
- [ ] **Commit**: `refactor: replace VarArgs with typed method dispatch`

### Stage 4.10 — Final API cleanup

- [ ] Audit all `pub use` in `lib.rs` — remove deprecated C-style exports
- [ ] Add `version()` → `&'static str` function
- [ ] Ensure public API uses safe Rust types exclusively:
  - No raw pointers in any public signature
  - Slices instead of pointer + length
  - `Result` for all fallible operations
  - Enums for all configuration values
- [ ] Add comprehensive doc comments to every public item, with links to
  upstream Opus documentation sections:
  - Encoder: https://opus-codec.org/docs/opus_api-1.3.1/group__opus__encoder.html
  - Decoder: https://opus-codec.org/docs/opus_api-1.3.1/group__opus__decoder.html
  - Repacketizer: https://opus-codec.org/docs/opus_api-1.3.1/group__opus__repacketizer.html
  - Error codes: https://opus-codec.org/docs/opus_api-1.3.1/group__opus__errorcodes.html
  - CTL values: https://opus-codec.org/docs/opus_api-1.3.1/group__opus__ctlvalues.html
- [ ] Add `#![forbid(unsafe_code)]` to crate root
- [ ] **Commit**: `refactor: finalize public API, add #![forbid(unsafe_code)]`

---

## Migration Strategy

To avoid breaking downstream users during the transition:

1. **Stage 4.1** — New types available immediately, old API still works
2. **Stages 4.4-4.7** — New API implemented alongside old; old functions
   marked `#[deprecated(since = "...", note = "use Encoder::new() instead")]`
3. **Stage 4.9** — Internal VarArgs removed, macros become thin wrappers
4. **Stage 4.10** — Old C-style exports removed, `forbid(unsafe_code)`

Tests should be updated to use the new API as each stage lands. The
`encode_vec` / `encode_vec_float` convenience methods are new additions
not in the C API — they exist purely for Rust ergonomics.

---

## API Mapping Reference

### C → Rust function mapping

| C function | Rust equivalent |
|-----------|----------------|
| `opus_encoder_create` | `Encoder::new()` |
| `opus_encoder_destroy` | `Drop` impl |
| `opus_encoder_init` | (internal, not public) |
| `opus_encoder_get_size` | (internal, not public) |
| `opus_encode` | `Encoder::encode()` |
| `opus_encode_float` | `Encoder::encode_float()` |
| `opus_encoder_ctl(SET_BITRATE)` | `Encoder::set_bitrate()` |
| `opus_encoder_ctl(GET_BITRATE)` | `Encoder::get_bitrate()` |
| (all other CTLs) | (individual typed methods) |
| `opus_decoder_create` | `Decoder::new()` |
| `opus_decoder_destroy` | `Drop` impl |
| `opus_decode` | `Decoder::decode()` |
| `opus_decode_float` | `Decoder::decode_float()` |
| `opus_decoder_get_nb_samples` | `Decoder::get_nb_samples()` |
| `opus_repacketizer_create` | `Repacketizer::new()` |
| `opus_repacketizer_cat` | `RepacketizerState::cat()` |
| `opus_repacketizer_out` | `RepacketizerState::out()` |
| `opus_repacketizer_out_range` | `RepacketizerState::out_range()` |
| `opus_pcm_soft_clip` | `SoftClip::apply()` |
| `opus_packet_get_bandwidth` | `packet::get_bandwidth()` |
| `opus_packet_get_nb_channels` | `packet::get_nb_channels()` |
| `opus_packet_get_nb_frames` | `packet::get_nb_frames()` |
| `opus_packet_get_nb_samples` | `packet::get_nb_samples()` |
| `opus_packet_get_samples_per_frame` | `packet::get_samples_per_frame()` |
| `opus_packet_parse` | `packet::parse()` |
| `opus_packet_pad` | `packet::pad()` |
| `opus_packet_unpad` | `packet::unpad()` |
| `opus_get_version_string` | `version()` |
| `opus_strerror` | `Error::description()` / `Display` |

### C constant → Rust enum mapping

| C constant | Rust type |
|-----------|-----------|
| `OPUS_APPLICATION_VOIP` (2048) | `Application::Voip` |
| `OPUS_APPLICATION_AUDIO` (2049) | `Application::Audio` |
| `OPUS_APPLICATION_RESTRICTED_LOWDELAY` (2051) | `Application::LowDelay` |
| `OPUS_AUTO` (-1000) | `Bandwidth::Auto` / `Signal::Auto` / `Bitrate::Auto` |
| `OPUS_BANDWIDTH_NARROWBAND` (1101) | `Bandwidth::Narrowband` |
| `OPUS_BANDWIDTH_MEDIUMBAND` (1102) | `Bandwidth::Mediumband` |
| `OPUS_BANDWIDTH_WIDEBAND` (1103) | `Bandwidth::Wideband` |
| `OPUS_BANDWIDTH_SUPERWIDEBAND` (1104) | `Bandwidth::Superwideband` |
| `OPUS_BANDWIDTH_FULLBAND` (1105) | `Bandwidth::Fullband` |
| `OPUS_BITRATE_MAX` (-1) | `Bitrate::Max` |
| `OPUS_SIGNAL_VOICE` (3001) | `Signal::Voice` |
| `OPUS_SIGNAL_MUSIC` (3002) | `Signal::Music` |
| `OPUS_FRAMESIZE_ARG` (5000) | `FrameSize::Arg` |
| `OPUS_FRAMESIZE_2_5_MS` (5001) | `FrameSize::Ms2_5` |
| `OPUS_FRAMESIZE_*` (5002-5009) | `FrameSize::Ms5` .. `FrameSize::Ms120` |
| `OPUS_OK` (0) | `Ok(...)` |
| `OPUS_BAD_ARG` (-1) | `Err(Error(ErrorCode::BadArg))` |
| `OPUS_BUFFER_TOO_SMALL` (-2) | `Err(Error(ErrorCode::BufferTooSmall))` |
| (etc.) | (etc.) |

---

## Documentation Guidelines

Every public item must have a doc comment that includes:

1. **One-line summary** — what it does
2. **Detailed description** — behavior, constraints, defaults
3. **Link to upstream docs** — `/// See: <url>` pointing to the specific
   section in https://opus-codec.org/docs/opus_api-1.3.1/
4. **Examples** where appropriate (especially `Encoder::new`, `Decoder::new`,
   `encode`, `decode`)
5. **Panics** / **Errors** sections documenting failure modes

Example:

```rust
/// Encode an Opus frame from 16-bit PCM audio.
///
/// `input` must contain exactly `frame_size * channels` interleaved
/// samples, where `frame_size` is one of the Opus frame sizes
/// (2.5, 5, 10, 20, 40, 60, 80, 100, or 120 ms) at the encoder's
/// sample rate.
///
/// Returns the number of bytes written to `output`.
///
/// # Errors
///
/// Returns [`ErrorCode::BadArg`] if the input length is not a valid
/// frame size, or [`ErrorCode::BufferTooSmall`] if `output` is too
/// small for the encoded packet.
///
/// See: <https://opus-codec.org/docs/opus_api-1.3.1/group__opus__encoder.html#ga4ae9905859cd241ef4bb5c59cd5e5309>
pub fn encode(&mut self, input: &[i16], output: &mut [u8]) -> Result<usize> { ... }
```

---

## Definition of Done

- [ ] `#![forbid(unsafe_code)]` in `lib.rs` — compiles successfully
- [ ] `externs.rs` deleted
- [ ] `varargs.rs` deleted (or just backward-compat macro shell)
- [ ] All encoder/decoder state uses proper Rust structs (no malloc'd blobs)
- [ ] Public API matches the target surface defined above:
  - `Encoder` with typed CTL methods
  - `Decoder` with typed CTL methods
  - `Repacketizer` with builder pattern
  - `SoftClip` struct
  - `packet` module with typed returns
  - `Error` / `ErrorCode` / `Result<T>`
  - All configuration enums
  - `version()` function
- [ ] No raw pointers in any public signature
- [ ] All public items have doc comments with upstream links
- [ ] All tests pass (cargo test + vector tests)
- [ ] All tests updated to use new API
- [ ] Clippy clean, formatted
