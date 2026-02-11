//! DNN subsystem for neural audio processing.
//!
//! Contains the neural network inference engine and model-specific modules
//! for Deep PLC, DRED, and OSCE features.
//!
//! Upstream C: `dnn/`

#![allow(dead_code)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(non_upper_case_globals)]
#![allow(unused_assignments)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::excessive_precision)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::approx_constant)]
#![allow(clippy::wildcard_in_or_patterns)]

pub mod burg;
#[cfg(feature = "dred")]
pub mod dred;
pub mod fargan;
pub mod freq;
pub mod lpcnet;
pub mod lpcnet_tables;
#[cfg(feature = "osce")]
pub mod nndsp;
pub mod nnet;
#[cfg(feature = "osce")]
pub mod osce;
pub mod pitchdnn;
pub mod vec;
