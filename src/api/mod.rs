//! Public API types for unsafe-libopus.
//!
//! This module provides typed enums and error types that replace the raw `i32`
//! constants used in the C API. These types are used by the safe Rust API
//! wrappers defined in later stages.
//!
//! The types here are modeled on the [upstream Opus API](https://opus-codec.org/docs/opus_api-1.3.1/)
//! but designed for idiomatic Rust usage — `Option<T>` replaces `OPUS_AUTO`
//! sentinels, and error handling uses `thiserror`-derived error types.

mod enums;
mod error;

pub use enums::{Application, Bandwidth, Bitrate, Channels, FrameSize, Signal};
pub use error::{ErrorCode, Result};
