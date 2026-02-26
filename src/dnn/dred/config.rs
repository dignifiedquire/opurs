//! DRED configuration constants.
//!
//! Upstream C: `dnn/dred_config.h`, `dnn/dred_rdovae_constants.h`

/// Extension ID for DRED data in packet padding.
pub const DRED_EXTENSION_ID: i32 = 126;

pub const DRED_EXPERIMENTAL_VERSION: i32 = 12;
pub const DRED_EXPERIMENTAL_BYTES: usize = 2;

pub const DRED_MIN_BYTES: usize = 8;

pub const DRED_SILK_ENCODER_DELAY: i32 = 79 + 12 - 80;
pub const DRED_FRAME_SIZE: usize = 160;
pub const DRED_DFRAME_SIZE: usize = 2 * DRED_FRAME_SIZE;
pub const DRED_MAX_DATA_SIZE: usize = 1000;
pub const DRED_ENC_Q0: i32 = 6;
pub const DRED_ENC_Q1: i32 = 15;

/// Covers 1.04 second so we can cover one second, after the lookahead.
pub const DRED_MAX_LATENTS: usize = 26;
pub const DRED_NUM_REDUNDANCY_FRAMES: usize = 2 * DRED_MAX_LATENTS;
pub const DRED_MAX_FRAMES: usize = 4 * DRED_MAX_LATENTS;

// --- From dred_rdovae_constants.h (auto-generated from model checkpoint) ---

pub const DRED_NUM_FEATURES: usize = 20;
pub const DRED_LATENT_DIM: usize = 25;
pub const DRED_STATE_DIM: usize = 50;
pub const DRED_PADDED_LATENT_DIM: usize = 32;
pub const DRED_PADDED_STATE_DIM: usize = 56;
pub const DRED_NUM_QUANTIZATION_LEVELS: usize = 16;
