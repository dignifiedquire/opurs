//! RDOVAE encoder: encodes feature frames into latent vectors.
//!
//! Upstream C: `dnn/dred_rdovae_enc.c`, `dnn/dred_rdovae_enc.h`, `dnn/dred_rdovae_enc_data.h`

use crate::dnn::nnet::*;

use super::config::*;

// --- Layer size constants from dred_rdovae_enc_data.h ---

const ENC_DENSE1_OUT_SIZE: usize = 64;
const ENC_ZDENSE_OUT_SIZE: usize = 24;
const GDENSE1_OUT_SIZE: usize = 128;

const ENC_GRU1_OUT_SIZE: usize = 64;
const ENC_GRU1_STATE_SIZE: usize = 64;
const ENC_GRU2_OUT_SIZE: usize = 64;
const ENC_GRU2_STATE_SIZE: usize = 64;
const ENC_GRU3_OUT_SIZE: usize = 64;
const ENC_GRU3_STATE_SIZE: usize = 64;
const ENC_GRU4_OUT_SIZE: usize = 64;
const ENC_GRU4_STATE_SIZE: usize = 64;
const ENC_GRU5_OUT_SIZE: usize = 64;
const ENC_GRU5_STATE_SIZE: usize = 64;

const ENC_CONV1_OUT_SIZE: usize = 96;
const ENC_CONV1_STATE_SIZE: usize = 128;
const ENC_CONV2_OUT_SIZE: usize = 96;
const ENC_CONV2_STATE_SIZE: usize = 288;
const ENC_CONV3_OUT_SIZE: usize = 96;
const ENC_CONV3_STATE_SIZE: usize = 448;
const ENC_CONV4_OUT_SIZE: usize = 96;
const ENC_CONV4_STATE_SIZE: usize = 608;
const ENC_CONV5_OUT_SIZE: usize = 96;
const ENC_CONV5_STATE_SIZE: usize = 768;

/// Total buffer size for concatenated layer outputs.
const ENC_BUFFER_SIZE: usize = ENC_DENSE1_OUT_SIZE
    + ENC_GRU1_OUT_SIZE
    + ENC_GRU2_OUT_SIZE
    + ENC_GRU3_OUT_SIZE
    + ENC_GRU4_OUT_SIZE
    + ENC_GRU5_OUT_SIZE
    + ENC_CONV1_OUT_SIZE
    + ENC_CONV2_OUT_SIZE
    + ENC_CONV3_OUT_SIZE
    + ENC_CONV4_OUT_SIZE
    + ENC_CONV5_OUT_SIZE;

// --- Model ---

/// RDOVAE encoder model.
///
/// Upstream C: dnn/dred_rdovae_enc_data.h:RDOVAEEnc
#[derive(Clone, Debug, Default)]
pub struct RDOVAEEnc {
    pub enc_dense1: LinearLayer,
    pub enc_zdense: LinearLayer,
    pub gdense1: LinearLayer,
    pub gdense2: LinearLayer,
    pub enc_gru1_input: LinearLayer,
    pub enc_gru1_recurrent: LinearLayer,
    pub enc_gru2_input: LinearLayer,
    pub enc_gru2_recurrent: LinearLayer,
    pub enc_gru3_input: LinearLayer,
    pub enc_gru3_recurrent: LinearLayer,
    pub enc_gru4_input: LinearLayer,
    pub enc_gru4_recurrent: LinearLayer,
    pub enc_gru5_input: LinearLayer,
    pub enc_gru5_recurrent: LinearLayer,
    pub enc_conv1: LinearLayer,
    pub enc_conv2: LinearLayer,
    pub enc_conv3: LinearLayer,
    pub enc_conv4: LinearLayer,
    pub enc_conv5: LinearLayer,
}

/// Initialize RDOVAE encoder model from weight arrays.
///
/// Upstream C: dnn/dred_rdovae_enc_data.c:init_rdovaeenc
pub fn init_rdovaeenc(arrays: &[WeightArray]) -> Option<RDOVAEEnc> {
    Some(RDOVAEEnc {
        enc_dense1: linear_init(
            arrays,
            "enc_dense1_bias",
            "",
            "",
            "enc_dense1_weights_float",
            "",
            "",
            "",
            40,
            64,
        )?,
        enc_zdense: linear_init(
            arrays,
            "enc_zdense_bias",
            "enc_zdense_subias",
            "enc_zdense_weights_int8",
            "enc_zdense_weights_float",
            "",
            "",
            "enc_zdense_scale",
            864,
            24,
        )?,
        gdense1: linear_init(
            arrays,
            "gdense1_bias",
            "gdense1_subias",
            "gdense1_weights_int8",
            "gdense1_weights_float",
            "",
            "",
            "gdense1_scale",
            864,
            128,
        )?,
        gdense2: linear_init(
            arrays,
            "gdense2_bias",
            "gdense2_subias",
            "gdense2_weights_int8",
            "gdense2_weights_float",
            "",
            "",
            "gdense2_scale",
            128,
            24,
        )?,
        enc_gru1_input: linear_init(
            arrays,
            "enc_gru1_input_bias",
            "enc_gru1_input_subias",
            "enc_gru1_input_weights_int8",
            "enc_gru1_input_weights_float",
            "enc_gru1_input_weights_idx",
            "",
            "enc_gru1_input_scale",
            64,
            192,
        )?,
        enc_gru1_recurrent: linear_init(
            arrays,
            "enc_gru1_recurrent_bias",
            "enc_gru1_recurrent_subias",
            "enc_gru1_recurrent_weights_int8",
            "enc_gru1_recurrent_weights_float",
            "",
            "",
            "enc_gru1_recurrent_scale",
            64,
            192,
        )?,
        enc_gru2_input: linear_init(
            arrays,
            "enc_gru2_input_bias",
            "enc_gru2_input_subias",
            "enc_gru2_input_weights_int8",
            "enc_gru2_input_weights_float",
            "enc_gru2_input_weights_idx",
            "",
            "enc_gru2_input_scale",
            224,
            192,
        )?,
        enc_gru2_recurrent: linear_init(
            arrays,
            "enc_gru2_recurrent_bias",
            "enc_gru2_recurrent_subias",
            "enc_gru2_recurrent_weights_int8",
            "enc_gru2_recurrent_weights_float",
            "",
            "",
            "enc_gru2_recurrent_scale",
            64,
            192,
        )?,
        enc_gru3_input: linear_init(
            arrays,
            "enc_gru3_input_bias",
            "enc_gru3_input_subias",
            "enc_gru3_input_weights_int8",
            "enc_gru3_input_weights_float",
            "enc_gru3_input_weights_idx",
            "",
            "enc_gru3_input_scale",
            384,
            192,
        )?,
        enc_gru3_recurrent: linear_init(
            arrays,
            "enc_gru3_recurrent_bias",
            "enc_gru3_recurrent_subias",
            "enc_gru3_recurrent_weights_int8",
            "enc_gru3_recurrent_weights_float",
            "",
            "",
            "enc_gru3_recurrent_scale",
            64,
            192,
        )?,
        enc_gru4_input: linear_init(
            arrays,
            "enc_gru4_input_bias",
            "enc_gru4_input_subias",
            "enc_gru4_input_weights_int8",
            "enc_gru4_input_weights_float",
            "enc_gru4_input_weights_idx",
            "",
            "enc_gru4_input_scale",
            544,
            192,
        )?,
        enc_gru4_recurrent: linear_init(
            arrays,
            "enc_gru4_recurrent_bias",
            "enc_gru4_recurrent_subias",
            "enc_gru4_recurrent_weights_int8",
            "enc_gru4_recurrent_weights_float",
            "",
            "",
            "enc_gru4_recurrent_scale",
            64,
            192,
        )?,
        enc_gru5_input: linear_init(
            arrays,
            "enc_gru5_input_bias",
            "enc_gru5_input_subias",
            "enc_gru5_input_weights_int8",
            "enc_gru5_input_weights_float",
            "enc_gru5_input_weights_idx",
            "",
            "enc_gru5_input_scale",
            704,
            192,
        )?,
        enc_gru5_recurrent: linear_init(
            arrays,
            "enc_gru5_recurrent_bias",
            "enc_gru5_recurrent_subias",
            "enc_gru5_recurrent_weights_int8",
            "enc_gru5_recurrent_weights_float",
            "",
            "",
            "enc_gru5_recurrent_scale",
            64,
            192,
        )?,
        enc_conv1: linear_init(
            arrays,
            "enc_conv1_bias",
            "enc_conv1_subias",
            "enc_conv1_weights_int8",
            "enc_conv1_weights_float",
            "",
            "",
            "enc_conv1_scale",
            256,
            96,
        )?,
        enc_conv2: linear_init(
            arrays,
            "enc_conv2_bias",
            "enc_conv2_subias",
            "enc_conv2_weights_int8",
            "enc_conv2_weights_float",
            "",
            "",
            "enc_conv2_scale",
            576,
            96,
        )?,
        enc_conv3: linear_init(
            arrays,
            "enc_conv3_bias",
            "enc_conv3_subias",
            "enc_conv3_weights_int8",
            "enc_conv3_weights_float",
            "",
            "",
            "enc_conv3_scale",
            896,
            96,
        )?,
        enc_conv4: linear_init(
            arrays,
            "enc_conv4_bias",
            "enc_conv4_subias",
            "enc_conv4_weights_int8",
            "enc_conv4_weights_float",
            "",
            "",
            "enc_conv4_scale",
            1216,
            96,
        )?,
        enc_conv5: linear_init(
            arrays,
            "enc_conv5_bias",
            "enc_conv5_subias",
            "enc_conv5_weights_int8",
            "enc_conv5_weights_float",
            "",
            "",
            "enc_conv5_scale",
            1536,
            96,
        )?,
    })
}

// --- State ---

/// RDOVAE encoder state.
///
/// Upstream C: dnn/dred_rdovae_enc.h:RDOVAEEncStruct
pub struct RDOVAEEncState {
    pub initialized: bool,
    pub gru1_state: Vec<f32>,
    pub gru2_state: Vec<f32>,
    pub gru3_state: Vec<f32>,
    pub gru4_state: Vec<f32>,
    pub gru5_state: Vec<f32>,
    pub conv1_state: Vec<f32>,
    pub conv2_state: Vec<f32>,
    pub conv3_state: Vec<f32>,
    pub conv4_state: Vec<f32>,
    pub conv5_state: Vec<f32>,
}

impl Default for RDOVAEEncState {
    fn default() -> Self {
        Self::new()
    }
}

impl RDOVAEEncState {
    pub fn new() -> Self {
        RDOVAEEncState {
            initialized: false,
            gru1_state: vec![0.0; ENC_GRU1_STATE_SIZE],
            gru2_state: vec![0.0; ENC_GRU2_STATE_SIZE],
            gru3_state: vec![0.0; ENC_GRU3_STATE_SIZE],
            gru4_state: vec![0.0; ENC_GRU4_STATE_SIZE],
            gru5_state: vec![0.0; ENC_GRU5_STATE_SIZE],
            conv1_state: vec![0.0; ENC_CONV1_STATE_SIZE],
            conv2_state: vec![0.0; 2 * ENC_CONV2_STATE_SIZE],
            conv3_state: vec![0.0; 2 * ENC_CONV3_STATE_SIZE],
            conv4_state: vec![0.0; 2 * ENC_CONV4_STATE_SIZE],
            conv5_state: vec![0.0; 2 * ENC_CONV5_STATE_SIZE],
        }
    }
}

/// Conditionally initialize conv state on first use.
fn conv1_cond_init(mem: &mut [f32], len: usize, dilation: usize, init: &mut bool) {
    if !*init {
        for i in 0..dilation {
            mem[i * len..i * len + len].fill(0.0);
        }
    }
    *init = true;
}

/// Encode a double feature frame (two concatenated 20-dim feature vectors).
///
/// Produces a latent vector and an initial state for the decoder.
///
/// Upstream C: dnn/dred_rdovae_enc.c:dred_rdovae_encode_dframe
pub fn dred_rdovae_encode_dframe(
    enc_state: &mut RDOVAEEncState,
    model: &RDOVAEEnc,
    latents: &mut [f32],
    initial_state: &mut [f32],
    input: &[f32],
) {
    let mut buffer = vec![0.0f32; ENC_BUFFER_SIZE];
    let mut output_index = 0;

    // Dense1
    compute_generic_dense(
        &model.enc_dense1,
        &mut buffer[output_index..output_index + ENC_DENSE1_OUT_SIZE],
        input,
        ACTIVATION_TANH,
    );
    output_index += ENC_DENSE1_OUT_SIZE;

    // GRU1 + Conv1
    compute_generic_gru(
        &model.enc_gru1_input,
        &model.enc_gru1_recurrent,
        &mut enc_state.gru1_state,
        &buffer,
    );
    buffer[output_index..output_index + ENC_GRU1_OUT_SIZE].copy_from_slice(&enc_state.gru1_state);
    output_index += ENC_GRU1_OUT_SIZE;
    conv1_cond_init(
        &mut enc_state.conv1_state,
        output_index,
        1,
        &mut enc_state.initialized,
    );
    let input_snap = buffer[..output_index].to_vec();
    compute_generic_conv1d(
        &model.enc_conv1,
        &mut buffer[output_index..output_index + ENC_CONV1_OUT_SIZE],
        &mut enc_state.conv1_state,
        &input_snap,
        output_index,
        ACTIVATION_TANH,
    );
    output_index += ENC_CONV1_OUT_SIZE;

    // GRU2 + Conv2 (dilation=2)
    compute_generic_gru(
        &model.enc_gru2_input,
        &model.enc_gru2_recurrent,
        &mut enc_state.gru2_state,
        &buffer,
    );
    buffer[output_index..output_index + ENC_GRU2_OUT_SIZE].copy_from_slice(&enc_state.gru2_state);
    output_index += ENC_GRU2_OUT_SIZE;
    conv1_cond_init(
        &mut enc_state.conv2_state,
        output_index,
        2,
        &mut enc_state.initialized,
    );
    let input_snap = buffer[..output_index].to_vec();
    compute_generic_conv1d_dilation(
        &model.enc_conv2,
        &mut buffer[output_index..output_index + ENC_CONV2_OUT_SIZE],
        &mut enc_state.conv2_state,
        &input_snap,
        output_index,
        2,
        ACTIVATION_TANH,
    );
    output_index += ENC_CONV2_OUT_SIZE;

    // GRU3 + Conv3 (dilation=2)
    compute_generic_gru(
        &model.enc_gru3_input,
        &model.enc_gru3_recurrent,
        &mut enc_state.gru3_state,
        &buffer,
    );
    buffer[output_index..output_index + ENC_GRU3_OUT_SIZE].copy_from_slice(&enc_state.gru3_state);
    output_index += ENC_GRU3_OUT_SIZE;
    conv1_cond_init(
        &mut enc_state.conv3_state,
        output_index,
        2,
        &mut enc_state.initialized,
    );
    let input_snap = buffer[..output_index].to_vec();
    compute_generic_conv1d_dilation(
        &model.enc_conv3,
        &mut buffer[output_index..output_index + ENC_CONV3_OUT_SIZE],
        &mut enc_state.conv3_state,
        &input_snap,
        output_index,
        2,
        ACTIVATION_TANH,
    );
    output_index += ENC_CONV3_OUT_SIZE;

    // GRU4 + Conv4 (dilation=2)
    compute_generic_gru(
        &model.enc_gru4_input,
        &model.enc_gru4_recurrent,
        &mut enc_state.gru4_state,
        &buffer,
    );
    buffer[output_index..output_index + ENC_GRU4_OUT_SIZE].copy_from_slice(&enc_state.gru4_state);
    output_index += ENC_GRU4_OUT_SIZE;
    conv1_cond_init(
        &mut enc_state.conv4_state,
        output_index,
        2,
        &mut enc_state.initialized,
    );
    let input_snap = buffer[..output_index].to_vec();
    compute_generic_conv1d_dilation(
        &model.enc_conv4,
        &mut buffer[output_index..output_index + ENC_CONV4_OUT_SIZE],
        &mut enc_state.conv4_state,
        &input_snap,
        output_index,
        2,
        ACTIVATION_TANH,
    );
    output_index += ENC_CONV4_OUT_SIZE;

    // GRU5 + Conv5 (dilation=2)
    compute_generic_gru(
        &model.enc_gru5_input,
        &model.enc_gru5_recurrent,
        &mut enc_state.gru5_state,
        &buffer,
    );
    buffer[output_index..output_index + ENC_GRU5_OUT_SIZE].copy_from_slice(&enc_state.gru5_state);
    output_index += ENC_GRU5_OUT_SIZE;
    conv1_cond_init(
        &mut enc_state.conv5_state,
        output_index,
        2,
        &mut enc_state.initialized,
    );
    let input_snap = buffer[..output_index].to_vec();
    compute_generic_conv1d_dilation(
        &model.enc_conv5,
        &mut buffer[output_index..output_index + ENC_CONV5_OUT_SIZE],
        &mut enc_state.conv5_state,
        &input_snap,
        output_index,
        2,
        ACTIVATION_TANH,
    );

    // Latent vector
    let mut padded_latents = [0.0f32; DRED_PADDED_LATENT_DIM];
    compute_generic_dense(
        &model.enc_zdense,
        &mut padded_latents,
        &buffer,
        ACTIVATION_LINEAR,
    );
    latents[..DRED_LATENT_DIM].copy_from_slice(&padded_latents[..DRED_LATENT_DIM]);

    // Initial state for decoder
    let mut state_hidden = vec![0.0f32; GDENSE1_OUT_SIZE];
    compute_generic_dense(&model.gdense1, &mut state_hidden, &buffer, ACTIVATION_TANH);
    let mut padded_state = [0.0f32; DRED_PADDED_STATE_DIM];
    compute_generic_dense(
        &model.gdense2,
        &mut padded_state,
        &state_hidden,
        ACTIVATION_LINEAR,
    );
    initial_state[..DRED_STATE_DIM].copy_from_slice(&padded_state[..DRED_STATE_DIM]);
}
