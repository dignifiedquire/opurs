//! DNN subsystem for neural audio processing.
//!
//! Contains the neural network inference engine and model-specific modules
//! for Deep PLC, DRED, and OSCE features.
//!
//! Upstream C: `dnn/`

#![allow(clippy::excessive_precision)] // Upstream numeric constants are kept bit-exact.

#[cfg(all(feature = "osce", feature = "builtin-weights"))]
pub mod bbwenet_data;
pub mod burg;
#[cfg(feature = "dred")]
pub mod dred;
pub mod fargan;
#[cfg(feature = "builtin-weights")]
pub mod fargan_data;
pub mod freq;
pub mod lpcnet;
pub mod lpcnet_tables;
#[cfg(feature = "osce")]
pub mod nndsp;
pub mod nnet;
#[cfg(feature = "osce")]
pub mod osce;
#[cfg(all(feature = "osce", feature = "builtin-weights"))]
pub mod osce_lace_data;
#[cfg(all(feature = "osce", feature = "builtin-weights"))]
pub mod osce_nolace_data;
pub mod pitchdnn;
#[cfg(feature = "builtin-weights")]
pub mod pitchdnn_data;
#[cfg(feature = "builtin-weights")]
pub mod plc_data;
pub mod vec;
pub mod weights;

#[cfg(feature = "simd")]
pub mod simd;
