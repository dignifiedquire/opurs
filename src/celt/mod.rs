//! CELT codec — low-latency, music-optimized audio coding.
//!
//! Upstream C: `celt/`

#![allow(non_camel_case_types)] // Transitional: remaining C-style identifiers are being migrated to Rust naming.
#![allow(non_snake_case)] // Transitional: remaining C-style identifiers are being migrated to Rust naming.

#[cfg(feature = "simd")]
pub mod simd;

pub mod bands;
pub mod celt_decoder;
pub mod celt_encoder;
pub mod celt_lpc;
pub mod common;
pub mod cwrs;
pub mod entcode;
pub mod entdec;
pub mod entenc;
pub mod kiss_fft;
pub mod laplace;
pub mod mathops;
pub mod mdct;
pub mod modes;
pub mod pitch;
pub mod quant_bands;
pub mod rate;
pub mod vq;
// stuff for structs that do not have a clear home, named after the header files
pub mod float_cast;
