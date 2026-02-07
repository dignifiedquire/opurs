//! Error types for the Opus codec.
//!
//! See: <https://opus-codec.org/docs/opus_api-1.3.1/group__opus__errorcodes.html>

use crate::src::opus_defines::{
    OPUS_ALLOC_FAIL, OPUS_BAD_ARG, OPUS_BUFFER_TOO_SMALL, OPUS_INTERNAL_ERROR, OPUS_INVALID_PACKET,
    OPUS_INVALID_STATE, OPUS_UNIMPLEMENTED,
};
use thiserror::Error;

/// Opus error codes.
///
/// Maps the upstream `OPUS_*` error constants to a Rust-idiomatic error enum.
/// Unknown error codes are preserved in the [`Unknown`](ErrorCode::Unknown)
/// variant for diagnostics.
///
/// See: <https://opus-codec.org/docs/opus_api-1.3.1/group__opus__errorcodes.html>
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, Error)]
pub enum ErrorCode {
    /// One or more invalid/out of range arguments.
    #[error("invalid argument")]
    BadArg,
    /// Not enough bytes allocated in the buffer.
    #[error("buffer too small")]
    BufferTooSmall,
    /// An internal error was detected.
    #[error("internal error")]
    InternalError,
    /// The compressed data passed is corrupted.
    #[error("corrupted stream")]
    InvalidPacket,
    /// Invalid/unsupported request number.
    #[error("request not implemented")]
    Unimplemented,
    /// An encoder or decoder structure is invalid or already freed.
    #[error("invalid state")]
    InvalidState,
    /// Memory allocation has failed.
    #[error("memory allocation failed")]
    AllocFail,
    /// An unknown error code was returned.
    #[error("unknown error ({0})")]
    Unknown(i32),
}

impl From<i32> for ErrorCode {
    fn from(value: i32) -> Self {
        match value {
            OPUS_BAD_ARG => ErrorCode::BadArg,
            OPUS_BUFFER_TOO_SMALL => ErrorCode::BufferTooSmall,
            OPUS_INTERNAL_ERROR => ErrorCode::InternalError,
            OPUS_INVALID_PACKET => ErrorCode::InvalidPacket,
            OPUS_UNIMPLEMENTED => ErrorCode::Unimplemented,
            OPUS_INVALID_STATE => ErrorCode::InvalidState,
            OPUS_ALLOC_FAIL => ErrorCode::AllocFail,
            other => ErrorCode::Unknown(other),
        }
    }
}

impl From<ErrorCode> for i32 {
    fn from(code: ErrorCode) -> Self {
        match code {
            ErrorCode::BadArg => OPUS_BAD_ARG,
            ErrorCode::BufferTooSmall => OPUS_BUFFER_TOO_SMALL,
            ErrorCode::InternalError => OPUS_INTERNAL_ERROR,
            ErrorCode::InvalidPacket => OPUS_INVALID_PACKET,
            ErrorCode::Unimplemented => OPUS_UNIMPLEMENTED,
            ErrorCode::InvalidState => OPUS_INVALID_STATE,
            ErrorCode::AllocFail => OPUS_ALLOC_FAIL,
            ErrorCode::Unknown(n) => n,
        }
    }
}

/// A specialized [`Result`](std::result::Result) type for Opus operations.
pub type Result<T> = std::result::Result<T, ErrorCode>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn error_code_from_known_values() {
        assert_eq!(ErrorCode::from(-1), ErrorCode::BadArg);
        assert_eq!(ErrorCode::from(-2), ErrorCode::BufferTooSmall);
        assert_eq!(ErrorCode::from(-3), ErrorCode::InternalError);
        assert_eq!(ErrorCode::from(-4), ErrorCode::InvalidPacket);
        assert_eq!(ErrorCode::from(-5), ErrorCode::Unimplemented);
        assert_eq!(ErrorCode::from(-6), ErrorCode::InvalidState);
        assert_eq!(ErrorCode::from(-7), ErrorCode::AllocFail);
    }

    #[test]
    fn error_code_from_unknown_values() {
        assert_eq!(ErrorCode::from(0), ErrorCode::Unknown(0));
        assert_eq!(ErrorCode::from(1), ErrorCode::Unknown(1));
        assert_eq!(ErrorCode::from(-100), ErrorCode::Unknown(-100));
        assert_eq!(ErrorCode::from(i32::MIN), ErrorCode::Unknown(i32::MIN));
    }

    #[test]
    fn error_code_roundtrip() {
        let codes = [
            ErrorCode::BadArg,
            ErrorCode::BufferTooSmall,
            ErrorCode::InternalError,
            ErrorCode::InvalidPacket,
            ErrorCode::Unimplemented,
            ErrorCode::InvalidState,
            ErrorCode::AllocFail,
            ErrorCode::Unknown(42),
        ];
        for code in codes {
            let raw: i32 = code.into();
            assert_eq!(ErrorCode::from(raw), code);
        }
    }

    #[test]
    fn error_code_display() {
        assert_eq!(format!("{}", ErrorCode::BadArg), "invalid argument");
        assert_eq!(format!("{}", ErrorCode::BufferTooSmall), "buffer too small");
        assert_eq!(format!("{}", ErrorCode::Unknown(99)), "unknown error (99)");
    }

    #[test]
    fn error_code_is_std_error() {
        fn assert_error<T: std::error::Error>() {}
        assert_error::<ErrorCode>();
    }

    #[test]
    fn error_code_is_copy() {
        fn assert_copy<T: Copy>() {}
        assert_copy::<ErrorCode>();
    }
}
