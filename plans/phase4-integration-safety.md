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

## Current State (updated 2026-02-11)

| File | unsafe fn | unsafe {} | Key issues |
|------|-----------|-----------|------------|
| `opus_encoder.rs` | 6 | 4 | encode_native, downmix, encode/encode_float |
| `opus_decoder.rs` | 2 | 3 | decode_frame, decode_native |
| `opus.rs` | 0 | 0 | **Safe** ✓ |
| `analysis.rs` | 0 | 0 | **Safe** ✓ (legacy downmix_func type alias remains) |
| `repacketizer.rs` | 0 | 0 | **Already safe** — needs API wrapper only |
| `mlp/` | 0 | 0 | **Already safe** |
| `opus_defines.rs` | 0 | 0 | Constants only — replace with enums |
| `opus_private.rs` | 0 | 0 | Safe |
| `externs.rs` | — | — | **Deleted** ✓ |
| `varargs.rs` | — | — | **Deleted** ✓ |

**Total remaining in src/src/**: 8 unsafe fn + 7 unsafe blocks = 15

---

## Stages

### Stage 4.1 — Define public types (`src/api/`)

Create the type-safe public API types that the rest of the stages will
target. This can be done before any internal refactoring.

- [x] Create `src/api/` module directory
- [x] `src/api/error.rs` — `ErrorCode` enum (thiserror-derived, no `#[repr(i32)]`),
  `Unknown(i32)` variant carries raw code, `Result<T>` alias
- [x] `src/api/enums.rs` — `Application`, `Channels`, `Bandwidth`, `Bitrate`,
  `Signal`, `FrameSize` enums (no `#[repr(i32)]`, no `Auto` variants on
  Bandwidth/Signal — use `Option<T>` at API boundary instead)
- [x] `src/api/mod.rs` — re-export all types
- [x] Wire `pub mod api` into `lib.rs`, re-export types at crate root
- [x] **Commit**: `refactor: define typed public API enums and error types`

### Stage 4.2 — Clean up analysis.rs signatures

`analysis.rs` bodies are already safe but function signatures are still
marked `unsafe fn`. Convert to safe signatures and update all callers.

- [x] Audit all 9 `unsafe fn` in analysis.rs — confirm bodies are safe
- [x] Remove `unsafe` from function signatures
- [x] Replace `downmix_func` callback with safe `DownmixInput` enum
- [x] Replace raw pointer params with slices/references
- [x] Replace `memcpy`/`memmove`/`memset` with safe equivalents
- [x] Convert `static mut` arrays to `static` (non-mut)
- [x] Update callers in opus_encoder.rs
- [x] **Commit**: `refactor: make analysis.rs function signatures safe`

### Stage 4.3 — Eliminate externs.rs usage ✅

All callers converted, `externs.rs` deleted:

- [x] All `externs::memcpy/memmove/memset/memcmp` callers already eliminated
  in Phases 2-3 (CELT and SILK safety refactoring)
- [x] `externs::malloc/free` in `opus_encoder_create/destroy` → `Box::new/from_raw`
- [x] `externs::malloc/free` in `opus_decoder_create/destroy` → `Box::new/from_raw`
- [x] Embed `silk_encoder` and `OpusCustomEncoder` directly in `OpusEncoder` struct
  (replaces C-style offset-into-fat-allocation pattern)
- [x] Add `OpusEncoder::new()` safe constructor
- [x] `opus_encoder_get_size` made safe (no longer `unsafe fn`)
- [x] Delete `src/externs.rs`
- [x] Remove `pub mod externs` from `lib.rs`
- [x] **Commit**: `refactor: embed sub-encoders in OpusEncoder, eliminate externs.rs` (dig-safe: 0ca8a46)

### Stage 4.4 — Safe opus_decoder.rs

- [x] Convert `OpusDecoder` from malloc'd blob to proper Rust struct
  - Named fields, embedded `CeltDecoder` and SILK state
- [x] Make leaf functions safe (validate, smooth_fade, get_mode, get_bandwidth, get_nb_channels)
- [x] Make opus_decode_frame use safe types (Option<&[u8]>, &mut [opus_val16])
- [x] Make opus_decode take &mut [i16] instead of *mut i16
- [x] Make OpusDecoder::new safe (no unsafe needed)
- [x] Add OpusDecoder::channels() public accessor
- [x] Add typed CTL methods: `set_gain`, `gain`, `last_packet_duration`, `pitch`,
  `set_phase_inversion_disabled`, `phase_inversion_disabled`, `final_range`,
  `bandwidth`, `sample_rate`, `reset`, `in_dtx`, `lookahead`
- [x] Delete C-style free functions (`opus_decoder_create`, `opus_decoder_init`, `opus_decoder_destroy`)
- [x] Delete `opus_decoder_ctl_impl` VarArgs dispatch (~80 lines)
- [x] Replace all `opus_custom_decoder_ctl!` calls with direct field access

**Remaining unsafe (2 fn + 3 blocks):**
- `unsafe fn opus_decode_frame` — internal, ~400 lines
- `unsafe fn opus_decode_native` — internal, ~170 lines, orchestrates multi-frame decode
- 3 unsafe blocks (1 calling opus_decode_native, 2 in decode internals)

- [ ] Make `opus_decode_frame` safe (eliminate raw pointer usage)
- [ ] Make `opus_decode_native` safe
- [ ] Remove remaining 3 unsafe blocks
- [ ] Implement `Decoder` wrapper struct (public API) — or use `OpusDecoder` directly
- [ ] **Commit**: `refactor: safe Decoder with idiomatic Rust API`

### Stage 4.5 — Safe opus_encoder.rs

Substantial progress — struct conversion done, typed API done, VarArgs
eliminated. Remaining work is the encode path itself.

- [x] Convert `OpusEncoder` from malloc'd blob to proper Rust struct
  - Named fields: SILK state, CELT state, analysis state (dig-safe: 0ca8a46)
- [x] Add `OpusEncoder::new()` safe constructor
- [x] Add typed CTL methods: `set_complexity`, `set_bitrate`, `set_vbr`,
  `set_vbr_constraint`, `set_bandwidth`, `set_max_bandwidth`, `set_signal`,
  `set_application`, `set_force_channels`, `set_dtx`, `set_inband_fec`,
  `set_packet_loss_perc`, `set_lsb_depth`, `set_expert_frame_duration`,
  `set_prediction_disabled`, `set_phase_inversion_disabled`, `set_force_mode`,
  `channels`, and all corresponding getters
- [x] Delete C-style free functions (`opus_encoder_create`, `opus_encoder_init`, `opus_encoder_destroy`)
- [x] Delete `opus_encoder_ctl_impl` VarArgs dispatch (~420 lines)
- [x] Replace all `opus_custom_encoder_ctl!` calls with direct field access
- [x] Refactor internal leaf/helper functions to safe:
  - [x] `gen_toc`, `frame_size_select`, `compute_silk_rate_for_hybrid`,
        `compute_equiv_rate`, `compute_redundancy_bytes`, `opus_select_arch`
  - [x] `user_bitrate_to_bitrate`, `is_digital_silence`, `compute_frame_energy`
  - [x] `silk_biquad_float`, `hp_cutoff`, `dc_reject`
  - [x] `stereo_fade`, `gain_fade`, `compute_stereo_width`
  - [x] `decide_fec`, `decide_dtx_mode`

**Remaining unsafe (6 fn + 4 blocks):**
- `unsafe fn downmix_float` — raw pointer arithmetic on `*const c_void`
- `unsafe fn downmix_int` — raw pointer arithmetic on `*const c_void`
- `unsafe fn opus_encode_native` — main encode path (~1300 lines)
- `unsafe fn encode_multiframe_packet` — calls opus_encode_native
- `unsafe fn opus_encode` — public i16 API, calls opus_encode_native
- `unsafe fn opus_encode_float` — public f32 API, calls opus_encode_native
- 4 unsafe blocks (mem::zeroed in new, from_raw_parts in encode_native)

- [ ] Make `downmix_float`/`downmix_int` safe (replace `*const c_void` with slices)
- [ ] Make `opus_encode_native` safe (~1300 lines of pointer-heavy code)
- [ ] Make `encode_multiframe_packet` safe (depends on opus_encode_native)
- [ ] Make `opus_encode`/`opus_encode_float` safe (depends on opus_encode_native)
- [ ] Replace `mem::zeroed()` in `OpusEncoder::new()` with field-by-field init
- [ ] Implement `Encoder` wrapper struct (public API) — or use `OpusEncoder` directly
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

### Stage 4.9 — Replace VarArgs internally ✅

All VarArgs usage has been eliminated. Direct field access replaces CTL
macro dispatch everywhere.

- [x] Convert internal `opus_encoder_ctl_impl` dispatching to direct method
  calls — **deleted entirely** (dig-safe: fe517bb)
- [x] Convert internal `opus_decoder_ctl_impl` dispatching to direct method
  calls — **deleted entirely** (dig-safe: fe517bb)
- [x] Convert `opus_custom_encoder_ctl_impl` to direct field access — **deleted** (dig-safe: fe517bb)
- [x] Convert `opus_custom_decoder_ctl_impl` to direct field access — **deleted** (dig-safe: fe517bb)
- [x] Add `reset()` methods to `OpusCustomEncoder` / `OpusCustomDecoder` (dig-safe: cbcaeb5)
- [x] Replace 43 encoder CTL macro calls with direct field access (dig-safe: 1713bc9)
- [x] Replace 19 decoder CTL macro calls with direct field access (dig-safe: 59f9f0b)
- [x] Migrate tools and tests off VarArgs (dig-safe: c9be18a)
- [x] Delete `src/varargs.rs` — **deleted entirely** (dig-safe: fe517bb)
- [x] Delete all CELT CTL request constants (dig-safe: fe517bb)
- [x] **Commits**: 5 commits from cbcaeb5 through fe517bb

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
