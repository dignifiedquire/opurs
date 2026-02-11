//! DRED (Deep REDundancy) encoder and decoder.
//!
//! Provides deep redundancy coding for packet loss recovery using
//! RDOVAE (Rate-Distortion Optimized Variational AutoEncoder).
//!
//! Upstream C: `dnn/dred_*.c`, `dnn/dred_*.h`

pub mod coding;
pub mod config;
pub mod decoder;
pub mod encoder;
pub mod rdovae_dec;
pub mod rdovae_enc;
pub mod stats;
