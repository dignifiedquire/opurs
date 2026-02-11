//! Testing and comparison utilities for opurs
//!
//! This module is gated behind the `tools` feature flag.

mod compare;
pub mod demo;

pub use compare::{opus_compare, CompareParams, CompareResult};
