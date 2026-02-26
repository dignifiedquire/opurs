//! FARGAN neural vocoder for speech synthesis.
//!
//! Generates audio from LPCNet features using a conditioning network
//! followed by a signal network with GRU layers and gated linear units.
//!
//! Upstream C: `dnn/fargan.c`, `dnn/fargan.h`, `dnn/fargan_data.h`

use super::freq::{FRAME_SIZE as LPCNET_FRAME_SIZE, NB_BANDS};
use super::nnet::*;
use super::pitchdnn::PITCH_MAX_PERIOD;
use super::weights::load_weights;
use crate::arch::Arch;

// --- Constants from fargan_data.h ---

const COND_NET_PEMBED_OUT_SIZE: usize = 12;
const COND_NET_FCONV1_IN_SIZE: usize = 64;
const COND_NET_FCONV1_OUT_SIZE: usize = 128;
const COND_NET_FCONV1_STATE_SIZE: usize = 64 * 2;
const COND_NET_FDENSE2_OUT_SIZE: usize = 320;

const SIG_NET_FWC0_CONV_OUT_SIZE: usize = 192;
const SIG_NET_FWC0_STATE_SIZE: usize = 2 * SIG_NET_INPUT_SIZE;
const SIG_NET_GRU1_OUT_SIZE: usize = 160;
const SIG_NET_GRU1_STATE_SIZE: usize = 160;
const SIG_NET_GRU2_OUT_SIZE: usize = 128;
const SIG_NET_GRU2_STATE_SIZE: usize = 128;
const SIG_NET_GRU3_OUT_SIZE: usize = 128;
const SIG_NET_GRU3_STATE_SIZE: usize = 128;
const SIG_NET_SKIP_DENSE_OUT_SIZE: usize = 128;

// --- Constants from fargan.h ---

pub const FARGAN_CONT_SAMPLES: usize = 320;
pub const FARGAN_NB_SUBFRAMES: usize = 4;
pub const FARGAN_SUBFRAME_SIZE: usize = 40;
pub const FARGAN_FRAME_SIZE: usize = FARGAN_NB_SUBFRAMES * FARGAN_SUBFRAME_SIZE;
const FARGAN_COND_SIZE: usize = COND_NET_FDENSE2_OUT_SIZE / FARGAN_NB_SUBFRAMES;
const FARGAN_DEEMPHASIS: f32 = 0.85;

const NB_FEATURES: usize = 20;
const SIG_NET_INPUT_SIZE: usize = FARGAN_COND_SIZE + 2 * FARGAN_SUBFRAME_SIZE + 4;

// --- Model ---

/// FARGAN model: conditioning + signal network layers.
///
/// Upstream C: dnn/fargan_data.h:FARGAN
#[derive(Clone, Debug, Default)]
pub struct FARGAN {
    pub cond_net_pembed: LinearLayer,
    pub cond_net_fdense1: LinearLayer,
    pub cond_net_fconv1: LinearLayer,
    pub cond_net_fdense2: LinearLayer,
    pub sig_net_cond_gain_dense: LinearLayer,
    pub sig_net_fwc0_conv: LinearLayer,
    pub sig_net_fwc0_glu_gate: LinearLayer,
    pub sig_net_gru1_input: LinearLayer,
    pub sig_net_gru1_recurrent: LinearLayer,
    pub sig_net_gru2_input: LinearLayer,
    pub sig_net_gru2_recurrent: LinearLayer,
    pub sig_net_gru3_input: LinearLayer,
    pub sig_net_gru3_recurrent: LinearLayer,
    pub sig_net_gru1_glu_gate: LinearLayer,
    pub sig_net_gru2_glu_gate: LinearLayer,
    pub sig_net_gru3_glu_gate: LinearLayer,
    pub sig_net_skip_glu_gate: LinearLayer,
    pub sig_net_skip_dense: LinearLayer,
    pub sig_net_sig_dense_out: LinearLayer,
    pub sig_net_gain_dense_out: LinearLayer,
}

/// Initialize FARGAN model from weight arrays.
///
/// Upstream C: dnn/fargan_data.c:init_fargan
pub fn init_fargan(arrays: &[WeightArray]) -> Option<FARGAN> {
    Some(FARGAN {
        cond_net_pembed: linear_init(
            arrays,
            "cond_net_pembed_bias",
            "",
            "",
            "cond_net_pembed_weights_float",
            "",
            "",
            "",
            224,
            12,
        )?,
        cond_net_fdense1: linear_init(
            arrays,
            "cond_net_fdense1_bias",
            "",
            "",
            "cond_net_fdense1_weights_float",
            "",
            "",
            "",
            32,
            64,
        )?,
        cond_net_fconv1: linear_init(
            arrays,
            "cond_net_fconv1_bias",
            "cond_net_fconv1_subias",
            "cond_net_fconv1_weights_int8",
            "cond_net_fconv1_weights_float",
            "",
            "",
            "cond_net_fconv1_scale",
            192,
            128,
        )?,
        cond_net_fdense2: linear_init(
            arrays,
            "cond_net_fdense2_bias",
            "cond_net_fdense2_subias",
            "cond_net_fdense2_weights_int8",
            "cond_net_fdense2_weights_float",
            "",
            "",
            "cond_net_fdense2_scale",
            128,
            320,
        )?,
        sig_net_cond_gain_dense: linear_init(
            arrays,
            "sig_net_cond_gain_dense_bias",
            "",
            "",
            "sig_net_cond_gain_dense_weights_float",
            "",
            "",
            "",
            80,
            1,
        )?,
        sig_net_fwc0_conv: linear_init(
            arrays,
            "sig_net_fwc0_conv_bias",
            "sig_net_fwc0_conv_subias",
            "sig_net_fwc0_conv_weights_int8",
            "sig_net_fwc0_conv_weights_float",
            "",
            "",
            "sig_net_fwc0_conv_scale",
            328,
            192,
        )?,
        sig_net_fwc0_glu_gate: linear_init(
            arrays,
            "sig_net_fwc0_glu_gate_bias",
            "sig_net_fwc0_glu_gate_subias",
            "sig_net_fwc0_glu_gate_weights_int8",
            "sig_net_fwc0_glu_gate_weights_float",
            "",
            "",
            "sig_net_fwc0_glu_gate_scale",
            192,
            192,
        )?,
        sig_net_gru1_input: linear_init(
            arrays,
            "",
            "sig_net_gru1_input_subias",
            "sig_net_gru1_input_weights_int8",
            "sig_net_gru1_input_weights_float",
            "",
            "",
            "sig_net_gru1_input_scale",
            272,
            480,
        )?,
        sig_net_gru1_recurrent: linear_init(
            arrays,
            "",
            "sig_net_gru1_recurrent_subias",
            "sig_net_gru1_recurrent_weights_int8",
            "sig_net_gru1_recurrent_weights_float",
            "",
            "",
            "sig_net_gru1_recurrent_scale",
            160,
            480,
        )?,
        sig_net_gru2_input: linear_init(
            arrays,
            "",
            "sig_net_gru2_input_subias",
            "sig_net_gru2_input_weights_int8",
            "sig_net_gru2_input_weights_float",
            "",
            "",
            "sig_net_gru2_input_scale",
            240,
            384,
        )?,
        sig_net_gru2_recurrent: linear_init(
            arrays,
            "",
            "sig_net_gru2_recurrent_subias",
            "sig_net_gru2_recurrent_weights_int8",
            "sig_net_gru2_recurrent_weights_float",
            "",
            "",
            "sig_net_gru2_recurrent_scale",
            128,
            384,
        )?,
        sig_net_gru3_input: linear_init(
            arrays,
            "",
            "sig_net_gru3_input_subias",
            "sig_net_gru3_input_weights_int8",
            "sig_net_gru3_input_weights_float",
            "",
            "",
            "sig_net_gru3_input_scale",
            208,
            384,
        )?,
        sig_net_gru3_recurrent: linear_init(
            arrays,
            "",
            "sig_net_gru3_recurrent_subias",
            "sig_net_gru3_recurrent_weights_int8",
            "sig_net_gru3_recurrent_weights_float",
            "",
            "",
            "sig_net_gru3_recurrent_scale",
            128,
            384,
        )?,
        sig_net_gru1_glu_gate: linear_init(
            arrays,
            "sig_net_gru1_glu_gate_bias",
            "sig_net_gru1_glu_gate_subias",
            "sig_net_gru1_glu_gate_weights_int8",
            "sig_net_gru1_glu_gate_weights_float",
            "",
            "",
            "sig_net_gru1_glu_gate_scale",
            160,
            160,
        )?,
        sig_net_gru2_glu_gate: linear_init(
            arrays,
            "sig_net_gru2_glu_gate_bias",
            "sig_net_gru2_glu_gate_subias",
            "sig_net_gru2_glu_gate_weights_int8",
            "sig_net_gru2_glu_gate_weights_float",
            "",
            "",
            "sig_net_gru2_glu_gate_scale",
            128,
            128,
        )?,
        sig_net_gru3_glu_gate: linear_init(
            arrays,
            "sig_net_gru3_glu_gate_bias",
            "sig_net_gru3_glu_gate_subias",
            "sig_net_gru3_glu_gate_weights_int8",
            "sig_net_gru3_glu_gate_weights_float",
            "",
            "",
            "sig_net_gru3_glu_gate_scale",
            128,
            128,
        )?,
        sig_net_skip_glu_gate: linear_init(
            arrays,
            "sig_net_skip_glu_gate_bias",
            "sig_net_skip_glu_gate_subias",
            "sig_net_skip_glu_gate_weights_int8",
            "sig_net_skip_glu_gate_weights_float",
            "",
            "",
            "sig_net_skip_glu_gate_scale",
            128,
            128,
        )?,
        sig_net_skip_dense: linear_init(
            arrays,
            "sig_net_skip_dense_bias",
            "sig_net_skip_dense_subias",
            "sig_net_skip_dense_weights_int8",
            "sig_net_skip_dense_weights_float",
            "",
            "",
            "sig_net_skip_dense_scale",
            688,
            128,
        )?,
        sig_net_sig_dense_out: linear_init(
            arrays,
            "sig_net_sig_dense_out_bias",
            "sig_net_sig_dense_out_subias",
            "sig_net_sig_dense_out_weights_int8",
            "sig_net_sig_dense_out_weights_float",
            "",
            "",
            "sig_net_sig_dense_out_scale",
            128,
            40,
        )?,
        sig_net_gain_dense_out: linear_init(
            arrays,
            "sig_net_gain_dense_out_bias",
            "",
            "",
            "sig_net_gain_dense_out_weights_float",
            "",
            "",
            "",
            192,
            4,
        )?,
    })
}

// --- State ---

/// FARGAN synthesizer state.
///
/// Upstream C: dnn/fargan.h:FARGANState
#[derive(Clone)]
pub struct FARGANState {
    pub model: FARGAN,
    pub cont_initialized: bool,
    pub deemph_mem: f32,
    pub pitch_buf: Vec<f32>,
    pub cond_conv1_state: Vec<f32>,
    pub fwc0_mem: Vec<f32>,
    pub gru1_state: Vec<f32>,
    pub gru2_state: Vec<f32>,
    pub gru3_state: Vec<f32>,
    pub last_period: i32,
}

impl Default for FARGANState {
    fn default() -> Self {
        Self::new()
    }
}

impl FARGANState {
    pub fn new() -> Self {
        let mut st = FARGANState {
            model: FARGAN::default(),
            cont_initialized: false,
            deemph_mem: 0.0,
            pitch_buf: vec![0.0; PITCH_MAX_PERIOD],
            cond_conv1_state: vec![0.0; COND_NET_FCONV1_STATE_SIZE],
            fwc0_mem: vec![0.0; SIG_NET_FWC0_STATE_SIZE],
            gru1_state: vec![0.0; SIG_NET_GRU1_STATE_SIZE],
            gru2_state: vec![0.0; SIG_NET_GRU2_STATE_SIZE],
            gru3_state: vec![0.0; SIG_NET_GRU3_STATE_SIZE],
            last_period: 0,
        };
        let _ = fargan_init(&mut st);
        st
    }

    /// Initialize state like upstream `fargan_init` (including builtin auto-load when enabled).
    ///
    /// Upstream C: dnn/fargan.c:fargan_init
    pub fn init_state(&mut self) -> bool {
        self.reset();
        #[cfg(feature = "builtin-weights")]
        {
            let arrays = crate::dnn::weights::compiled_weights();
            self.init(&arrays)
        }
        #[cfg(not(feature = "builtin-weights"))]
        {
            true
        }
    }

    /// Initialize model from parsed weight arrays.
    ///
    /// Upstream C: dnn/fargan_data.c:init_fargan
    pub fn init(&mut self, arrays: &[WeightArray]) -> bool {
        match init_fargan(arrays) {
            Some(model) => {
                self.model = model;
                self.reset();
                true
            }
            None => false,
        }
    }

    /// Initialize from a serialized DNN weight blob in one call.
    ///
    /// Upstream C: dnn/fargan.c:fargan_load_model
    pub fn load_model(&mut self, data: &[u8]) -> bool {
        fargan_load_model(self, data)
    }

    /// Reset state without reloading model.
    pub fn reset(&mut self) {
        self.cont_initialized = false;
        self.deemph_mem = 0.0;
        self.pitch_buf.fill(0.0);
        self.cond_conv1_state.fill(0.0);
        self.fwc0_mem.fill(0.0);
        self.gru1_state.fill(0.0);
        self.gru2_state.fill(0.0);
        self.gru3_state.fill(0.0);
        self.last_period = 0;
    }
}

/// Initialize FARGAN state, loading built-in weights when that feature is enabled.
///
/// Upstream C: dnn/fargan.c:fargan_init
pub fn fargan_init(st: &mut FARGANState) -> bool {
    st.init_state()
}

/// One-shot FARGAN model loading from a serialized weight blob.
///
/// Upstream C: dnn/fargan.c:fargan_load_model
pub fn fargan_load_model(st: &mut FARGANState, data: &[u8]) -> bool {
    let Some(arrays) = load_weights(data) else {
        return false;
    };
    st.init(&arrays)
}

// --- Conditioning network ---

/// Compute conditioning vector from features and pitch period.
///
/// Upstream C: dnn/fargan.c:compute_fargan_cond
fn compute_fargan_cond(
    st: &mut FARGANState,
    cond: &mut [f32],
    features: &[f32],
    period: i32,
    arch: Arch,
) {
    let model = &st.model;

    // Build input: features + pitch embedding
    let mut dense_in = vec![0.0f32; NB_FEATURES + COND_NET_PEMBED_OUT_SIZE];
    dense_in[..NB_FEATURES].copy_from_slice(&features[..NB_FEATURES]);
    let embed_idx = (period - 32).clamp(0, 223) as usize;
    let embed_start = embed_idx * COND_NET_PEMBED_OUT_SIZE;
    let embed_end = embed_start + COND_NET_PEMBED_OUT_SIZE;
    dense_in[NB_FEATURES..NB_FEATURES + COND_NET_PEMBED_OUT_SIZE]
        .copy_from_slice(&model.cond_net_pembed.float_weights[embed_start..embed_end]);

    let mut conv1_in = vec![0.0f32; COND_NET_FCONV1_IN_SIZE];
    compute_generic_dense(
        &model.cond_net_fdense1,
        &mut conv1_in,
        &dense_in,
        ACTIVATION_TANH,
        arch,
    );

    let mut fdense2_in = vec![0.0f32; COND_NET_FCONV1_OUT_SIZE];
    compute_generic_conv1d(
        &model.cond_net_fconv1,
        &mut fdense2_in,
        &mut st.cond_conv1_state,
        &conv1_in,
        COND_NET_FCONV1_IN_SIZE,
        ACTIVATION_TANH,
        arch,
    );

    compute_generic_dense(
        &model.cond_net_fdense2,
        cond,
        &fdense2_in,
        ACTIVATION_TANH,
        arch,
    );
}

// --- De-emphasis ---

fn fargan_deemphasis(pcm: &mut [f32], deemph_mem: &mut f32) {
    for i in 0..FARGAN_SUBFRAME_SIZE {
        pcm[i] += FARGAN_DEEMPHASIS * *deemph_mem;
        *deemph_mem = pcm[i];
    }
}

// --- Signal network subframe ---

/// Run signal network for one subframe (40 samples).
///
/// Upstream C: dnn/fargan.c:run_fargan_subframe
fn run_fargan_subframe(
    st: &mut FARGANState,
    pcm: &mut [f32],
    cond: &[f32],
    period: i32,
    arch: Arch,
) {
    debug_assert!(st.cont_initialized);
    let model = &st.model;

    // Gain from conditioning
    let mut gain = [0.0f32; 1];
    compute_generic_dense(
        &model.sig_net_cond_gain_dense,
        &mut gain,
        cond,
        ACTIVATION_LINEAR,
        arch,
    );
    // C: gain = exp(gain) — float promoted to double, result truncated to float.
    let gain = (gain[0] as f64).exp() as f32;
    let gain_1 = 1.0 / (1e-5 + gain);

    // Build pitch prediction and previous samples
    let mut pred = [0.0f32; FARGAN_SUBFRAME_SIZE + 4];
    let mut pos = PITCH_MAX_PERIOD as i32 - period - 2;
    for i in 0..FARGAN_SUBFRAME_SIZE + 4 {
        let p = pos.max(0) as usize;
        pred[i] = (gain_1 * st.pitch_buf[p]).clamp(-1.0, 1.0);
        pos += 1;
        if pos == PITCH_MAX_PERIOD as i32 {
            pos -= period;
        }
    }

    let mut prev = [0.0f32; FARGAN_SUBFRAME_SIZE];
    for i in 0..FARGAN_SUBFRAME_SIZE {
        prev[i] =
            (gain_1 * st.pitch_buf[PITCH_MAX_PERIOD - FARGAN_SUBFRAME_SIZE + i]).clamp(-1.0, 1.0);
    }

    // Build fwc0 input: cond + pred + prev
    let mut fwc0_in = vec![0.0f32; SIG_NET_INPUT_SIZE];
    fwc0_in[..FARGAN_COND_SIZE].copy_from_slice(&cond[..FARGAN_COND_SIZE]);
    fwc0_in[FARGAN_COND_SIZE..FARGAN_COND_SIZE + FARGAN_SUBFRAME_SIZE + 4].copy_from_slice(&pred);
    fwc0_in[FARGAN_COND_SIZE + FARGAN_SUBFRAME_SIZE + 4..].copy_from_slice(&prev);

    // FWC0 conv + GLU
    let mut gru1_in = vec![0.0f32; SIG_NET_FWC0_CONV_OUT_SIZE + 2 * FARGAN_SUBFRAME_SIZE];
    compute_generic_conv1d(
        &model.sig_net_fwc0_conv,
        &mut gru1_in,
        &mut st.fwc0_mem,
        &fwc0_in,
        SIG_NET_INPUT_SIZE,
        ACTIVATION_TANH,
        arch,
    );
    compute_glu(
        &model.sig_net_fwc0_glu_gate,
        &mut gru1_in.clone(),
        &gru1_in,
        arch,
    );
    let gru1_in_glu = gru1_in.clone();
    compute_glu(
        &model.sig_net_fwc0_glu_gate,
        &mut gru1_in[..SIG_NET_FWC0_CONV_OUT_SIZE],
        &gru1_in_glu[..SIG_NET_FWC0_CONV_OUT_SIZE],
        arch,
    );

    // Pitch gate
    let mut pitch_gate = [0.0f32; 4];
    compute_generic_dense(
        &model.sig_net_gain_dense_out,
        &mut pitch_gate,
        &gru1_in,
        ACTIVATION_SIGMOID,
        arch,
    );

    // GRU1
    for i in 0..FARGAN_SUBFRAME_SIZE {
        gru1_in[SIG_NET_FWC0_CONV_OUT_SIZE + i] = pitch_gate[0] * pred[i + 2]; // SIG_NET_FWC0_GLU_GATE_OUT_SIZE == SIG_NET_FWC0_CONV_OUT_SIZE
    }
    gru1_in[SIG_NET_FWC0_CONV_OUT_SIZE + FARGAN_SUBFRAME_SIZE
        ..SIG_NET_FWC0_CONV_OUT_SIZE + 2 * FARGAN_SUBFRAME_SIZE]
        .copy_from_slice(&prev);
    compute_generic_gru(
        &model.sig_net_gru1_input,
        &model.sig_net_gru1_recurrent,
        &mut st.gru1_state,
        &gru1_in,
        arch,
    );

    let mut gru2_in = vec![0.0f32; SIG_NET_GRU1_OUT_SIZE + 2 * FARGAN_SUBFRAME_SIZE];
    compute_glu(
        &model.sig_net_gru1_glu_gate,
        &mut gru2_in[..SIG_NET_GRU1_OUT_SIZE],
        &st.gru1_state,
        arch,
    );

    // GRU2
    for i in 0..FARGAN_SUBFRAME_SIZE {
        gru2_in[SIG_NET_GRU1_OUT_SIZE + i] = pitch_gate[1] * pred[i + 2];
    }
    gru2_in[SIG_NET_GRU1_OUT_SIZE + FARGAN_SUBFRAME_SIZE
        ..SIG_NET_GRU1_OUT_SIZE + 2 * FARGAN_SUBFRAME_SIZE]
        .copy_from_slice(&prev);
    compute_generic_gru(
        &model.sig_net_gru2_input,
        &model.sig_net_gru2_recurrent,
        &mut st.gru2_state,
        &gru2_in,
        arch,
    );

    let mut gru3_in = vec![0.0f32; SIG_NET_GRU2_OUT_SIZE + 2 * FARGAN_SUBFRAME_SIZE];
    compute_glu(
        &model.sig_net_gru2_glu_gate,
        &mut gru3_in[..SIG_NET_GRU2_OUT_SIZE],
        &st.gru2_state,
        arch,
    );

    // GRU3
    for i in 0..FARGAN_SUBFRAME_SIZE {
        gru3_in[SIG_NET_GRU2_OUT_SIZE + i] = pitch_gate[2] * pred[i + 2];
    }
    gru3_in[SIG_NET_GRU2_OUT_SIZE + FARGAN_SUBFRAME_SIZE
        ..SIG_NET_GRU2_OUT_SIZE + 2 * FARGAN_SUBFRAME_SIZE]
        .copy_from_slice(&prev);
    compute_generic_gru(
        &model.sig_net_gru3_input,
        &model.sig_net_gru3_recurrent,
        &mut st.gru3_state,
        &gru3_in,
        arch,
    );

    // Skip connections: cat(gru1_glu, gru2_glu, gru3_glu, fwc0_glu, pitch_gate*pred, prev)
    let skip_size = SIG_NET_GRU1_OUT_SIZE
        + SIG_NET_GRU2_OUT_SIZE
        + SIG_NET_GRU3_OUT_SIZE
        + SIG_NET_FWC0_CONV_OUT_SIZE
        + 2 * FARGAN_SUBFRAME_SIZE;
    let mut skip_cat = vec![0.0f32; skip_size];
    skip_cat[..SIG_NET_GRU1_OUT_SIZE].copy_from_slice(&gru2_in[..SIG_NET_GRU1_OUT_SIZE]);
    skip_cat[SIG_NET_GRU1_OUT_SIZE..SIG_NET_GRU1_OUT_SIZE + SIG_NET_GRU2_OUT_SIZE]
        .copy_from_slice(&gru3_in[..SIG_NET_GRU2_OUT_SIZE]);

    let gru3_glu_start = SIG_NET_GRU1_OUT_SIZE + SIG_NET_GRU2_OUT_SIZE;
    compute_glu(
        &model.sig_net_gru3_glu_gate,
        &mut skip_cat[gru3_glu_start..gru3_glu_start + SIG_NET_GRU3_OUT_SIZE],
        &st.gru3_state,
        arch,
    );

    let fwc0_start = gru3_glu_start + SIG_NET_GRU3_OUT_SIZE;
    skip_cat[fwc0_start..fwc0_start + SIG_NET_FWC0_CONV_OUT_SIZE]
        .copy_from_slice(&gru1_in[..SIG_NET_FWC0_CONV_OUT_SIZE]);
    let pred_start = fwc0_start + SIG_NET_FWC0_CONV_OUT_SIZE;
    for i in 0..FARGAN_SUBFRAME_SIZE {
        skip_cat[pred_start + i] = pitch_gate[3] * pred[i + 2];
    }
    skip_cat[pred_start + FARGAN_SUBFRAME_SIZE..pred_start + 2 * FARGAN_SUBFRAME_SIZE]
        .copy_from_slice(&prev);

    // Skip dense + GLU
    let mut skip_out = vec![0.0f32; SIG_NET_SKIP_DENSE_OUT_SIZE];
    compute_generic_dense(
        &model.sig_net_skip_dense,
        &mut skip_out,
        &skip_cat,
        ACTIVATION_TANH,
        arch,
    );
    let skip_tmp = skip_out.clone();
    compute_glu(&model.sig_net_skip_glu_gate, &mut skip_out, &skip_tmp, arch);

    // Final output
    compute_generic_dense(
        &model.sig_net_sig_dense_out,
        pcm,
        &skip_out,
        ACTIVATION_TANH,
        arch,
    );
    for i in 0..FARGAN_SUBFRAME_SIZE {
        pcm[i] *= gain;
    }

    // Update pitch buffer
    st.pitch_buf.copy_within(FARGAN_SUBFRAME_SIZE.., 0);
    st.pitch_buf[PITCH_MAX_PERIOD - FARGAN_SUBFRAME_SIZE..]
        .copy_from_slice(&pcm[..FARGAN_SUBFRAME_SIZE]);

    fargan_deemphasis(pcm, &mut st.deemph_mem);
}

/// Initialize FARGAN from continuation context (5 feature frames + PCM).
///
/// Upstream C: dnn/fargan.c:fargan_cont
pub fn fargan_cont(st: &mut FARGANState, pcm0: &[f32], features0: &[f32], arch: Arch) {
    let mut cond = vec![0.0f32; COND_NET_FDENSE2_OUT_SIZE];
    let mut period = 0i32;

    // Pre-load features (5 frames)
    for i in 0..5 {
        let features = &features0[i * NB_FEATURES..];
        st.last_period = period;
        period = (0.5
            + 256.0 / 2.0f64.powf((1.0 / 60.0) * ((features[NB_BANDS] as f64 + 1.5) * 60.0)))
        .floor() as i32;
        compute_fargan_cond(st, &mut cond, features, period, arch);
    }

    // Pre-emphasis on continuation PCM
    let mut x0 = vec![0.0f32; FARGAN_CONT_SAMPLES];
    x0[0] = 0.0;
    for i in 1..FARGAN_CONT_SAMPLES {
        x0[i] = pcm0[i] - FARGAN_DEEMPHASIS * pcm0[i - 1];
    }

    // Fill pitch buffer with last frame
    st.pitch_buf[PITCH_MAX_PERIOD - FARGAN_FRAME_SIZE..].copy_from_slice(&x0[..FARGAN_FRAME_SIZE]);
    st.cont_initialized = true;

    // Run subframes with the real continuation data
    for i in 0..FARGAN_NB_SUBFRAMES {
        let mut dummy = [0.0f32; FARGAN_SUBFRAME_SIZE];
        run_fargan_subframe(
            st,
            &mut dummy,
            &cond[i * FARGAN_COND_SIZE..],
            st.last_period,
            arch,
        );
        // Override pitch buffer with actual continuation data
        st.pitch_buf[PITCH_MAX_PERIOD - FARGAN_SUBFRAME_SIZE..].copy_from_slice(
            &x0[FARGAN_FRAME_SIZE + i * FARGAN_SUBFRAME_SIZE
                ..FARGAN_FRAME_SIZE + (i + 1) * FARGAN_SUBFRAME_SIZE],
        );
    }
    st.deemph_mem = pcm0[FARGAN_CONT_SAMPLES - 1];
}

/// Synthesize one frame (160 samples) of float PCM from features.
///
/// Upstream C: dnn/fargan.c:fargan_synthesize
pub fn fargan_synthesize(st: &mut FARGANState, pcm: &mut [f32], features: &[f32], arch: Arch) {
    debug_assert!(st.cont_initialized);
    let mut cond = vec![0.0f32; COND_NET_FDENSE2_OUT_SIZE];
    let period = (0.5
        + 256.0 / 2.0f64.powf((1.0 / 60.0) * ((features[NB_BANDS] as f64 + 1.5) * 60.0)))
    .floor() as i32;
    compute_fargan_cond(st, &mut cond, features, period, arch);
    for subframe in 0..FARGAN_NB_SUBFRAMES {
        run_fargan_subframe(
            st,
            &mut pcm[subframe * FARGAN_SUBFRAME_SIZE..],
            &cond[subframe * FARGAN_COND_SIZE..],
            st.last_period,
            arch,
        );
    }
    st.last_period = period;
}

/// Synthesize one frame (160 samples) of int16 PCM from features.
///
/// Upstream C: dnn/fargan.c:fargan_synthesize_int
pub fn fargan_synthesize_int(st: &mut FARGANState, pcm: &mut [i16], features: &[f32], arch: Arch) {
    let mut fpcm = vec![0.0f32; FARGAN_FRAME_SIZE];
    fargan_synthesize(st, &mut fpcm, features, arch);
    for i in 0..LPCNET_FRAME_SIZE {
        pcm[i] = (0.5 + (32768.0 * fpcm[i]).clamp(-32767.0, 32767.0)).floor() as i16;
    }
}
