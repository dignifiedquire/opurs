//! RDOVAE decoder: decodes latent vectors back into feature frames.
//!
//! Upstream C: `dnn/dred_rdovae_dec.c`, `dnn/dred_rdovae_dec.h`, `dnn/dred_rdovae_dec_data.h`

use crate::dnn::nnet::*;

use super::config::*;

// --- Layer size constants from dred_rdovae_dec_data.h ---

const DEC_DENSE1_OUT_SIZE: usize = 96;
const DEC_OUTPUT_OUT_SIZE: usize = 80;
const DEC_HIDDEN_INIT_OUT_SIZE: usize = 128;

const DEC_GRU1_OUT_SIZE: usize = 96;
const DEC_GRU1_STATE_SIZE: usize = 96;
const DEC_GRU2_OUT_SIZE: usize = 96;
const DEC_GRU2_STATE_SIZE: usize = 96;
const DEC_GRU3_OUT_SIZE: usize = 96;
const DEC_GRU3_STATE_SIZE: usize = 96;
const DEC_GRU4_OUT_SIZE: usize = 96;
const DEC_GRU4_STATE_SIZE: usize = 96;
const DEC_GRU5_OUT_SIZE: usize = 96;
const DEC_GRU5_STATE_SIZE: usize = 96;

const DEC_CONV1_OUT_SIZE: usize = 32;
const DEC_CONV1_STATE_SIZE: usize = 192;
const DEC_CONV2_OUT_SIZE: usize = 32;
const DEC_CONV2_STATE_SIZE: usize = 320;
const DEC_CONV3_OUT_SIZE: usize = 32;
const DEC_CONV3_STATE_SIZE: usize = 448;
const DEC_CONV4_OUT_SIZE: usize = 32;
const DEC_CONV4_STATE_SIZE: usize = 576;
const DEC_CONV5_OUT_SIZE: usize = 32;
const DEC_CONV5_STATE_SIZE: usize = 704;

const DEC_GRU_INIT_OUT_SIZE: usize = DEC_GRU1_STATE_SIZE
    + DEC_GRU2_STATE_SIZE
    + DEC_GRU3_STATE_SIZE
    + DEC_GRU4_STATE_SIZE
    + DEC_GRU5_STATE_SIZE;

/// Total buffer size for concatenated layer outputs.
const DEC_BUFFER_SIZE: usize = DEC_DENSE1_OUT_SIZE
    + DEC_GRU1_OUT_SIZE
    + DEC_GRU2_OUT_SIZE
    + DEC_GRU3_OUT_SIZE
    + DEC_GRU4_OUT_SIZE
    + DEC_GRU5_OUT_SIZE
    + DEC_CONV1_OUT_SIZE
    + DEC_CONV2_OUT_SIZE
    + DEC_CONV3_OUT_SIZE
    + DEC_CONV4_OUT_SIZE
    + DEC_CONV5_OUT_SIZE;

// --- Model ---

/// RDOVAE decoder model.
///
/// Upstream C: dnn/dred_rdovae_dec_data.h:RDOVAEDec
#[derive(Clone, Debug, Default)]
pub struct RDOVAEDec {
    pub dec_dense1: LinearLayer,
    pub dec_glu1: LinearLayer,
    pub dec_glu2: LinearLayer,
    pub dec_glu3: LinearLayer,
    pub dec_glu4: LinearLayer,
    pub dec_glu5: LinearLayer,
    pub dec_output: LinearLayer,
    pub dec_hidden_init: LinearLayer,
    pub dec_gru_init: LinearLayer,
    pub dec_gru1_input: LinearLayer,
    pub dec_gru1_recurrent: LinearLayer,
    pub dec_gru2_input: LinearLayer,
    pub dec_gru2_recurrent: LinearLayer,
    pub dec_gru3_input: LinearLayer,
    pub dec_gru3_recurrent: LinearLayer,
    pub dec_gru4_input: LinearLayer,
    pub dec_gru4_recurrent: LinearLayer,
    pub dec_gru5_input: LinearLayer,
    pub dec_gru5_recurrent: LinearLayer,
    pub dec_conv1: LinearLayer,
    pub dec_conv2: LinearLayer,
    pub dec_conv3: LinearLayer,
    pub dec_conv4: LinearLayer,
    pub dec_conv5: LinearLayer,
}

/// Initialize RDOVAE decoder model from weight arrays.
///
/// Upstream C: dnn/dred_rdovae_dec_data.c:init_rdovaedec
pub fn init_rdovaedec(arrays: &[WeightArray]) -> Option<RDOVAEDec> {
    Some(RDOVAEDec {
        dec_dense1: linear_init(
            arrays,
            "dec_dense1_bias",
            "",
            "",
            "dec_dense1_weights_float",
            "",
            "",
            "",
            DRED_LATENT_DIM,
            96,
        )?,
        dec_glu1: linear_init(
            arrays,
            "dec_glu1_bias",
            "dec_glu1_subias",
            "dec_glu1_weights_int8",
            "dec_glu1_weights_float",
            "",
            "",
            "dec_glu1_scale",
            96,
            96,
        )?,
        dec_glu2: linear_init(
            arrays,
            "dec_glu2_bias",
            "dec_glu2_subias",
            "dec_glu2_weights_int8",
            "dec_glu2_weights_float",
            "",
            "",
            "dec_glu2_scale",
            96,
            96,
        )?,
        dec_glu3: linear_init(
            arrays,
            "dec_glu3_bias",
            "dec_glu3_subias",
            "dec_glu3_weights_int8",
            "dec_glu3_weights_float",
            "",
            "",
            "dec_glu3_scale",
            96,
            96,
        )?,
        dec_glu4: linear_init(
            arrays,
            "dec_glu4_bias",
            "dec_glu4_subias",
            "dec_glu4_weights_int8",
            "dec_glu4_weights_float",
            "",
            "",
            "dec_glu4_scale",
            96,
            96,
        )?,
        dec_glu5: linear_init(
            arrays,
            "dec_glu5_bias",
            "dec_glu5_subias",
            "dec_glu5_weights_int8",
            "dec_glu5_weights_float",
            "",
            "",
            "dec_glu5_scale",
            96,
            96,
        )?,
        dec_output: linear_init(
            arrays,
            "dec_output_bias",
            "dec_output_subias",
            "dec_output_weights_int8",
            "dec_output_weights_float",
            "",
            "",
            "dec_output_scale",
            736,
            80,
        )?,
        dec_hidden_init: linear_init(
            arrays,
            "dec_hidden_init_bias",
            "",
            "",
            "dec_hidden_init_weights_float",
            "",
            "",
            "",
            DRED_STATE_DIM,
            128,
        )?,
        dec_gru_init: linear_init(
            arrays,
            "dec_gru_init_bias",
            "dec_gru_init_subias",
            "dec_gru_init_weights_int8",
            "dec_gru_init_weights_float",
            "",
            "",
            "dec_gru_init_scale",
            128,
            480,
        )?,
        dec_gru1_input: linear_init(
            arrays,
            "dec_gru1_input_bias",
            "dec_gru1_input_subias",
            "dec_gru1_input_weights_int8",
            "dec_gru1_input_weights_float",
            "dec_gru1_input_weights_idx",
            "",
            "dec_gru1_input_scale",
            96,
            288,
        )?,
        dec_gru1_recurrent: linear_init(
            arrays,
            "dec_gru1_recurrent_bias",
            "dec_gru1_recurrent_subias",
            "dec_gru1_recurrent_weights_int8",
            "dec_gru1_recurrent_weights_float",
            "",
            "",
            "dec_gru1_recurrent_scale",
            96,
            288,
        )?,
        dec_gru2_input: linear_init(
            arrays,
            "dec_gru2_input_bias",
            "dec_gru2_input_subias",
            "dec_gru2_input_weights_int8",
            "dec_gru2_input_weights_float",
            "dec_gru2_input_weights_idx",
            "",
            "dec_gru2_input_scale",
            224,
            288,
        )?,
        dec_gru2_recurrent: linear_init(
            arrays,
            "dec_gru2_recurrent_bias",
            "dec_gru2_recurrent_subias",
            "dec_gru2_recurrent_weights_int8",
            "dec_gru2_recurrent_weights_float",
            "",
            "",
            "dec_gru2_recurrent_scale",
            96,
            288,
        )?,
        dec_gru3_input: linear_init(
            arrays,
            "dec_gru3_input_bias",
            "dec_gru3_input_subias",
            "dec_gru3_input_weights_int8",
            "dec_gru3_input_weights_float",
            "dec_gru3_input_weights_idx",
            "",
            "dec_gru3_input_scale",
            352,
            288,
        )?,
        dec_gru3_recurrent: linear_init(
            arrays,
            "dec_gru3_recurrent_bias",
            "dec_gru3_recurrent_subias",
            "dec_gru3_recurrent_weights_int8",
            "dec_gru3_recurrent_weights_float",
            "",
            "",
            "dec_gru3_recurrent_scale",
            96,
            288,
        )?,
        dec_gru4_input: linear_init(
            arrays,
            "dec_gru4_input_bias",
            "dec_gru4_input_subias",
            "dec_gru4_input_weights_int8",
            "dec_gru4_input_weights_float",
            "dec_gru4_input_weights_idx",
            "",
            "dec_gru4_input_scale",
            480,
            288,
        )?,
        dec_gru4_recurrent: linear_init(
            arrays,
            "dec_gru4_recurrent_bias",
            "dec_gru4_recurrent_subias",
            "dec_gru4_recurrent_weights_int8",
            "dec_gru4_recurrent_weights_float",
            "",
            "",
            "dec_gru4_recurrent_scale",
            96,
            288,
        )?,
        dec_gru5_input: linear_init(
            arrays,
            "dec_gru5_input_bias",
            "dec_gru5_input_subias",
            "dec_gru5_input_weights_int8",
            "dec_gru5_input_weights_float",
            "dec_gru5_input_weights_idx",
            "",
            "dec_gru5_input_scale",
            608,
            288,
        )?,
        dec_gru5_recurrent: linear_init(
            arrays,
            "dec_gru5_recurrent_bias",
            "dec_gru5_recurrent_subias",
            "dec_gru5_recurrent_weights_int8",
            "dec_gru5_recurrent_weights_float",
            "",
            "",
            "dec_gru5_recurrent_scale",
            96,
            288,
        )?,
        dec_conv1: linear_init(
            arrays,
            "dec_conv1_bias",
            "dec_conv1_subias",
            "dec_conv1_weights_int8",
            "dec_conv1_weights_float",
            "",
            "",
            "dec_conv1_scale",
            384,
            32,
        )?,
        dec_conv2: linear_init(
            arrays,
            "dec_conv2_bias",
            "dec_conv2_subias",
            "dec_conv2_weights_int8",
            "dec_conv2_weights_float",
            "",
            "",
            "dec_conv2_scale",
            640,
            32,
        )?,
        dec_conv3: linear_init(
            arrays,
            "dec_conv3_bias",
            "dec_conv3_subias",
            "dec_conv3_weights_int8",
            "dec_conv3_weights_float",
            "",
            "",
            "dec_conv3_scale",
            896,
            32,
        )?,
        dec_conv4: linear_init(
            arrays,
            "dec_conv4_bias",
            "dec_conv4_subias",
            "dec_conv4_weights_int8",
            "dec_conv4_weights_float",
            "",
            "",
            "dec_conv4_scale",
            1152,
            32,
        )?,
        dec_conv5: linear_init(
            arrays,
            "dec_conv5_bias",
            "dec_conv5_subias",
            "dec_conv5_weights_int8",
            "dec_conv5_weights_float",
            "",
            "",
            "dec_conv5_scale",
            1408,
            32,
        )?,
    })
}

// --- State ---

/// RDOVAE decoder state.
///
/// Upstream C: dnn/dred_rdovae_dec.h:RDOVAEDecStruct
pub struct RDOVAEDecState {
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

impl Default for RDOVAEDecState {
    fn default() -> Self {
        Self::new()
    }
}

impl RDOVAEDecState {
    pub fn new() -> Self {
        RDOVAEDecState {
            initialized: false,
            gru1_state: vec![0.0; DEC_GRU1_STATE_SIZE],
            gru2_state: vec![0.0; DEC_GRU2_STATE_SIZE],
            gru3_state: vec![0.0; DEC_GRU3_STATE_SIZE],
            gru4_state: vec![0.0; DEC_GRU4_STATE_SIZE],
            gru5_state: vec![0.0; DEC_GRU5_STATE_SIZE],
            conv1_state: vec![0.0; DEC_CONV1_STATE_SIZE],
            conv2_state: vec![0.0; DEC_CONV2_STATE_SIZE],
            conv3_state: vec![0.0; DEC_CONV3_STATE_SIZE],
            conv4_state: vec![0.0; DEC_CONV4_STATE_SIZE],
            conv5_state: vec![0.0; DEC_CONV5_STATE_SIZE],
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

/// Initialize decoder GRU states from initial state vector.
///
/// Upstream C: dnn/dred_rdovae_dec.c:dred_rdovae_dec_init_states
pub fn dred_rdovae_dec_init_states(
    h: &mut RDOVAEDecState,
    model: &RDOVAEDec,
    initial_state: &[f32],
) {
    let mut hidden = vec![0.0f32; DEC_HIDDEN_INIT_OUT_SIZE];
    compute_generic_dense(
        &model.dec_hidden_init,
        &mut hidden,
        initial_state,
        ACTIVATION_TANH,
    );

    let mut state_init = vec![0.0f32; DEC_GRU_INIT_OUT_SIZE];
    compute_generic_dense(
        &model.dec_gru_init,
        &mut state_init,
        &hidden,
        ACTIVATION_TANH,
    );

    let mut counter = 0;
    h.gru1_state[..DEC_GRU1_STATE_SIZE]
        .copy_from_slice(&state_init[counter..counter + DEC_GRU1_STATE_SIZE]);
    counter += DEC_GRU1_STATE_SIZE;
    h.gru2_state[..DEC_GRU2_STATE_SIZE]
        .copy_from_slice(&state_init[counter..counter + DEC_GRU2_STATE_SIZE]);
    counter += DEC_GRU2_STATE_SIZE;
    h.gru3_state[..DEC_GRU3_STATE_SIZE]
        .copy_from_slice(&state_init[counter..counter + DEC_GRU3_STATE_SIZE]);
    counter += DEC_GRU3_STATE_SIZE;
    h.gru4_state[..DEC_GRU4_STATE_SIZE]
        .copy_from_slice(&state_init[counter..counter + DEC_GRU4_STATE_SIZE]);
    counter += DEC_GRU4_STATE_SIZE;
    h.gru5_state[..DEC_GRU5_STATE_SIZE]
        .copy_from_slice(&state_init[counter..counter + DEC_GRU5_STATE_SIZE]);
    h.initialized = false;
}

/// Decode one quadruple feature frame from a latent vector.
///
/// Upstream C: dnn/dred_rdovae_dec.c:dred_rdovae_decode_qframe
pub fn dred_rdovae_decode_qframe(
    dec_state: &mut RDOVAEDecState,
    model: &RDOVAEDec,
    qframe: &mut [f32],
    input: &[f32],
) {
    let mut buffer = vec![0.0f32; DEC_BUFFER_SIZE];
    let mut output_index = 0;

    // Dense1
    compute_generic_dense(
        &model.dec_dense1,
        &mut buffer[output_index..output_index + DEC_DENSE1_OUT_SIZE],
        input,
        ACTIVATION_TANH,
    );
    output_index += DEC_DENSE1_OUT_SIZE;

    // GRU1 + GLU + Conv1
    compute_generic_gru(
        &model.dec_gru1_input,
        &model.dec_gru1_recurrent,
        &mut dec_state.gru1_state,
        &buffer,
    );
    compute_glu(
        &model.dec_glu1,
        &mut buffer[output_index..output_index + DEC_GRU1_OUT_SIZE],
        &dec_state.gru1_state,
    );
    output_index += DEC_GRU1_OUT_SIZE;
    conv1_cond_init(
        &mut dec_state.conv1_state,
        output_index,
        1,
        &mut dec_state.initialized,
    );
    let input_snap = buffer[..output_index].to_vec();
    compute_generic_conv1d(
        &model.dec_conv1,
        &mut buffer[output_index..output_index + DEC_CONV1_OUT_SIZE],
        &mut dec_state.conv1_state,
        &input_snap,
        output_index,
        ACTIVATION_TANH,
    );
    output_index += DEC_CONV1_OUT_SIZE;

    // GRU2 + GLU + Conv2
    compute_generic_gru(
        &model.dec_gru2_input,
        &model.dec_gru2_recurrent,
        &mut dec_state.gru2_state,
        &buffer,
    );
    compute_glu(
        &model.dec_glu2,
        &mut buffer[output_index..output_index + DEC_GRU2_OUT_SIZE],
        &dec_state.gru2_state,
    );
    output_index += DEC_GRU2_OUT_SIZE;
    conv1_cond_init(
        &mut dec_state.conv2_state,
        output_index,
        1,
        &mut dec_state.initialized,
    );
    let input_snap = buffer[..output_index].to_vec();
    compute_generic_conv1d(
        &model.dec_conv2,
        &mut buffer[output_index..output_index + DEC_CONV2_OUT_SIZE],
        &mut dec_state.conv2_state,
        &input_snap,
        output_index,
        ACTIVATION_TANH,
    );
    output_index += DEC_CONV2_OUT_SIZE;

    // GRU3 + GLU + Conv3
    compute_generic_gru(
        &model.dec_gru3_input,
        &model.dec_gru3_recurrent,
        &mut dec_state.gru3_state,
        &buffer,
    );
    compute_glu(
        &model.dec_glu3,
        &mut buffer[output_index..output_index + DEC_GRU3_OUT_SIZE],
        &dec_state.gru3_state,
    );
    output_index += DEC_GRU3_OUT_SIZE;
    conv1_cond_init(
        &mut dec_state.conv3_state,
        output_index,
        1,
        &mut dec_state.initialized,
    );
    let input_snap = buffer[..output_index].to_vec();
    compute_generic_conv1d(
        &model.dec_conv3,
        &mut buffer[output_index..output_index + DEC_CONV3_OUT_SIZE],
        &mut dec_state.conv3_state,
        &input_snap,
        output_index,
        ACTIVATION_TANH,
    );
    output_index += DEC_CONV3_OUT_SIZE;

    // GRU4 + GLU + Conv4
    compute_generic_gru(
        &model.dec_gru4_input,
        &model.dec_gru4_recurrent,
        &mut dec_state.gru4_state,
        &buffer,
    );
    compute_glu(
        &model.dec_glu4,
        &mut buffer[output_index..output_index + DEC_GRU4_OUT_SIZE],
        &dec_state.gru4_state,
    );
    output_index += DEC_GRU4_OUT_SIZE;
    conv1_cond_init(
        &mut dec_state.conv4_state,
        output_index,
        1,
        &mut dec_state.initialized,
    );
    let input_snap = buffer[..output_index].to_vec();
    compute_generic_conv1d(
        &model.dec_conv4,
        &mut buffer[output_index..output_index + DEC_CONV4_OUT_SIZE],
        &mut dec_state.conv4_state,
        &input_snap,
        output_index,
        ACTIVATION_TANH,
    );
    output_index += DEC_CONV4_OUT_SIZE;

    // GRU5 + GLU + Conv5
    compute_generic_gru(
        &model.dec_gru5_input,
        &model.dec_gru5_recurrent,
        &mut dec_state.gru5_state,
        &buffer,
    );
    compute_glu(
        &model.dec_glu5,
        &mut buffer[output_index..output_index + DEC_GRU5_OUT_SIZE],
        &dec_state.gru5_state,
    );
    output_index += DEC_GRU5_OUT_SIZE;
    conv1_cond_init(
        &mut dec_state.conv5_state,
        output_index,
        1,
        &mut dec_state.initialized,
    );
    let input_snap = buffer[..output_index].to_vec();
    compute_generic_conv1d(
        &model.dec_conv5,
        &mut buffer[output_index..output_index + DEC_CONV5_OUT_SIZE],
        &mut dec_state.conv5_state,
        &input_snap,
        output_index,
        ACTIVATION_TANH,
    );

    // Output
    compute_generic_dense(
        &model.dec_output,
        &mut qframe[..DEC_OUTPUT_OUT_SIZE],
        &buffer,
        ACTIVATION_LINEAR,
    );
}

/// Decode all latent frames at once.
///
/// Upstream C: dnn/dred_rdovae_dec.c:DRED_rdovae_decode_all
pub fn dred_rdovae_decode_all(
    model: &RDOVAEDec,
    features: &mut [f32],
    state: &[f32],
    latents: &[f32],
    nb_latents: usize,
) {
    let mut dec = RDOVAEDecState::new();
    dred_rdovae_dec_init_states(&mut dec, model, state);
    let mut i = 0;
    while i < 2 * nb_latents {
        dred_rdovae_decode_qframe(
            &mut dec,
            model,
            &mut features[2 * i * DRED_NUM_FEATURES..],
            &latents[(i / 2) * DRED_LATENT_DIM..],
        );
        i += 2;
    }
}
