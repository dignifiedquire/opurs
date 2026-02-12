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
#[cfg(feature = "builtin-weights")]
pub mod rdovae_dec_data;
pub mod rdovae_enc;
#[cfg(feature = "builtin-weights")]
pub mod rdovae_enc_data;
pub mod stats;
