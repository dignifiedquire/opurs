//! LPCNet encoder and packet loss concealment (PLC).
//!
//! The encoder extracts features from PCM audio for use by the PLC and DRED.
//! The PLC uses a small neural network + FARGAN to conceal lost frames.
//!
//! Upstream C: `dnn/lpcnet_enc.c`, `dnn/lpcnet_plc.c`, `dnn/lpcnet_private.h`

use super::fargan::*;
use super::freq::*;
use super::nnet::*;
use super::pitchdnn::*;
use crate::celt::celt_lpc::celt_fir_c;
use crate::celt::kiss_fft::kiss_fft_cpx;
use crate::celt::mathops::celt_log2;
use crate::celt::pitch::{celt_inner_prod, celt_pitch_xcorr};

// --- Constants ---

const NB_FEATURES: usize = 20;
pub const NB_TOTAL_FEATURES: usize = 36;

const PITCH_FRAME_SIZE: usize = 320;
const PITCH_BUF_SIZE: usize = PITCH_MAX_PERIOD + PITCH_FRAME_SIZE;

pub const PLC_MAX_FEC: usize = 100;
const CONT_VECTORS: usize = 5;
const FEATURES_DELAY: usize = 1;

const PITCH_IF_MAX_FREQ: usize = 30;
const PITCH_IF_FEATURES: usize = 3 * PITCH_IF_MAX_FREQ - 2;

const TRAINING_OFFSET_5MS: usize = 1;
const TRAINING_OFFSET: usize = 80 * TRAINING_OFFSET_5MS;

const PLC_BUF_SIZE: usize = (CONT_VECTORS + 10) * FRAME_SIZE;

// PLC model constants from plc_data.h
const PLC_DENSE_IN_OUT_SIZE: usize = 128;
const PLC_GRU1_STATE_SIZE: usize = 192;
const PLC_GRU2_STATE_SIZE: usize = 192;

// --- PLC Model ---

/// PLC prediction model (small NN).
///
/// Upstream C: dnn/plc_data.h:PLCModel
#[derive(Clone, Debug, Default)]
pub struct PLCModel {
    pub plc_dense_in: LinearLayer,
    pub plc_dense_out: LinearLayer,
    pub plc_gru1_input: LinearLayer,
    pub plc_gru1_recurrent: LinearLayer,
    pub plc_gru2_input: LinearLayer,
    pub plc_gru2_recurrent: LinearLayer,
}

/// Initialize PLC model from weight arrays.
///
/// Upstream C: dnn/plc_data.c:init_plcmodel
pub fn init_plcmodel(arrays: &[WeightArray]) -> Option<PLCModel> {
    Some(PLCModel {
        plc_dense_in: linear_init(
            arrays,
            "plc_dense_in_bias",
            "",
            "",
            "plc_dense_in_weights_float",
            "",
            "",
            "",
            57,
            128,
        )?,
        plc_dense_out: linear_init(
            arrays,
            "plc_dense_out_bias",
            "",
            "",
            "plc_dense_out_weights_float",
            "",
            "",
            "",
            192,
            20,
        )?,
        plc_gru1_input: linear_init(
            arrays,
            "plc_gru1_input_bias",
            "plc_gru1_input_subias",
            "plc_gru1_input_weights_int8",
            "plc_gru1_input_weights_float",
            "",
            "",
            "plc_gru1_input_scale",
            128,
            576,
        )?,
        plc_gru1_recurrent: linear_init(
            arrays,
            "plc_gru1_recurrent_bias",
            "plc_gru1_recurrent_subias",
            "plc_gru1_recurrent_weights_int8",
            "plc_gru1_recurrent_weights_float",
            "",
            "",
            "plc_gru1_recurrent_scale",
            192,
            576,
        )?,
        plc_gru2_input: linear_init(
            arrays,
            "plc_gru2_input_bias",
            "plc_gru2_input_subias",
            "plc_gru2_input_weights_int8",
            "plc_gru2_input_weights_float",
            "",
            "",
            "plc_gru2_input_scale",
            192,
            576,
        )?,
        plc_gru2_recurrent: linear_init(
            arrays,
            "plc_gru2_recurrent_bias",
            "plc_gru2_recurrent_subias",
            "plc_gru2_recurrent_weights_int8",
            "plc_gru2_recurrent_weights_float",
            "",
            "",
            "plc_gru2_recurrent_scale",
            192,
            576,
        )?,
    })
}

// --- PLCNet state (GRU states for the PLC model) ---

#[derive(Clone)]
struct PLCNetState {
    gru1_state: Vec<f32>,
    gru2_state: Vec<f32>,
}

impl PLCNetState {
    fn new() -> Self {
        PLCNetState {
            gru1_state: vec![0.0; PLC_GRU1_STATE_SIZE],
            gru2_state: vec![0.0; PLC_GRU2_STATE_SIZE],
        }
    }
}

// --- LPCNet Encoder State ---

/// LPCNet encoder state for feature extraction.
///
/// Upstream C: dnn/lpcnet_private.h:LPCNetEncState
pub struct LPCNetEncState {
    pub pitchdnn: PitchDNNState,
    pub analysis_mem: Vec<f32>,
    pub mem_preemph: f32,
    prev_if: Vec<kiss_fft_cpx>,
    if_features: Vec<f32>,
    xcorr_features: Vec<f32>,
    pub dnn_pitch: f32,
    pitch_mem: Vec<f32>,
    pitch_filt: f32,
    exc_buf: Vec<f32>,
    lp_buf: Vec<f32>,
    lp_mem: [f32; 2],
    pub lpc: Vec<f32>,
    pub features: Vec<f32>,
    sig_mem: Vec<f32>,
    pub burg_cepstrum: Vec<f32>,
}

impl Default for LPCNetEncState {
    fn default() -> Self {
        Self::new()
    }
}

impl LPCNetEncState {
    pub fn new() -> Self {
        LPCNetEncState {
            pitchdnn: PitchDNNState::new(),
            analysis_mem: vec![0.0; OVERLAP_SIZE],
            mem_preemph: 0.0,
            prev_if: vec![kiss_fft_cpx::default(); PITCH_IF_MAX_FREQ],
            if_features: vec![0.0; PITCH_IF_FEATURES],
            xcorr_features: vec![0.0; PITCH_MAX_PERIOD - PITCH_MIN_PERIOD],
            dnn_pitch: 0.0,
            pitch_mem: vec![0.0; LPC_ORDER],
            pitch_filt: 0.0,
            exc_buf: vec![0.0; PITCH_BUF_SIZE],
            lp_buf: vec![0.0; PITCH_BUF_SIZE],
            lp_mem: [0.0; 2],
            lpc: vec![0.0; LPC_ORDER],
            features: vec![0.0; NB_TOTAL_FEATURES],
            sig_mem: vec![0.0; LPC_ORDER],
            burg_cepstrum: vec![0.0; 2 * NB_BANDS],
        }
    }

    /// Initialize encoder state.
    ///
    /// Upstream C: dnn/lpcnet_enc.c:lpcnet_encoder_init
    pub fn init(&mut self) {
        *self = LPCNetEncState::new();
    }

    /// Load model weights.
    pub fn load_model(&mut self, arrays: &[WeightArray]) -> bool {
        self.pitchdnn.init(arrays)
    }
}

// --- Frame analysis ---

/// Pre-emphasis filter.
///
/// Upstream C: dnn/lpcnet_enc.c:preemphasis
pub fn preemphasis(y: &mut [f32], mem: &mut f32, x: &[f32], coef: f32, n: usize) {
    for i in 0..n {
        let yi = x[i] + *mem;
        *mem = -coef * x[i];
        y[i] = yi;
    }
}

/// Biquad IIR filter.
///
/// Upstream C: dnn/lpcnet_enc.c:biquad
fn biquad(y: &mut [f32], mem: &mut [f32; 2], x: &[f32], b: &[f32; 2], a: &[f32; 2], n: usize) {
    let mut mem0 = mem[0];
    let mut mem1 = mem[1];
    for i in 0..n {
        let xi = x[i];
        let yi = x[i] + mem0;
        let mem00 = mem0;
        mem0 = (b[0] - a[0]) * xi + mem1 - a[0] * mem0;
        mem1 = (b[1] - a[1]) * xi + 1e-30 - a[1] * mem00;
        y[i] = yi;
    }
    mem[0] = mem0;
    mem[1] = mem1;
}

fn celt_log10(x: f32) -> f32 {
    0.30102999566 * celt_log2(x)
}

/// Compute spectral features from input, FFT, and band energy.
///
/// Upstream C: dnn/lpcnet_enc.c:frame_analysis
fn frame_analysis(
    st: &mut LPCNetEncState,
    x_out: &mut [kiss_fft_cpx],
    ex: &mut [f32],
    input: &[f32],
) {
    let mut x = [0.0f32; WINDOW_SIZE];
    x[..OVERLAP_SIZE].copy_from_slice(&st.analysis_mem);
    x[OVERLAP_SIZE..OVERLAP_SIZE + FRAME_SIZE].copy_from_slice(&input[..FRAME_SIZE]);
    st.analysis_mem
        .copy_from_slice(&input[FRAME_SIZE - OVERLAP_SIZE..FRAME_SIZE]);
    apply_window(&mut x);
    forward_transform(x_out, &x);
    lpcn_compute_band_energy(ex, x_out);
}

/// Compute full frame features from input PCM.
///
/// Upstream C: dnn/lpcnet_enc.c:compute_frame_features
pub fn compute_frame_features(st: &mut LPCNetEncState, input: &[f32]) {
    let mut aligned_in = vec![0.0f32; FRAME_SIZE];
    aligned_in[..TRAINING_OFFSET]
        .copy_from_slice(&st.analysis_mem[OVERLAP_SIZE - TRAINING_OFFSET..OVERLAP_SIZE]);

    let mut x_fft = vec![kiss_fft_cpx::default(); FREQ_SIZE];
    let mut ex = vec![0.0f32; NB_BANDS];
    frame_analysis(st, &mut x_fft, &mut ex, input);

    // Instantaneous frequency features
    st.if_features[0] = ((1.0 / 64.0)
        * (10.0 * celt_log10(1e-15 + x_fft[0].re * x_fft[0].re) - 6.0))
        .clamp(-1.0, 1.0);
    for i in 1..PITCH_IF_MAX_FREQ {
        // C_MULC: prod = X[i] * conj(prev_if[i])
        let prod_r = x_fft[i].re * st.prev_if[i].re + x_fft[i].im * st.prev_if[i].im;
        let prod_i = x_fft[i].im * st.prev_if[i].re - x_fft[i].re * st.prev_if[i].im;
        let norm_1 = 1.0 / (1e-15 + prod_r * prod_r + prod_i * prod_i).sqrt();
        st.if_features[3 * i - 2] = prod_r * norm_1;
        st.if_features[3 * i - 1] = prod_i * norm_1;
        st.if_features[3 * i] = ((1.0 / 64.0)
            * (10.0 * celt_log10(1e-15 + x_fft[i].re * x_fft[i].re + x_fft[i].im * x_fft[i].im)
                - 6.0))
            .clamp(-1.0, 1.0);
    }
    st.prev_if[..PITCH_IF_MAX_FREQ].copy_from_slice(&x_fft[..PITCH_IF_MAX_FREQ]);

    // Cepstral features from band energy
    let mut ly = [0.0f32; NB_BANDS];
    let mut log_max: f32 = -2.0;
    let mut follow: f32 = -2.0;
    for i in 0..NB_BANDS {
        ly[i] = celt_log10(1e-2 + ex[i]);
        ly[i] = ly[i].max(log_max - 8.0).max(follow - 2.5);
        log_max = log_max.max(ly[i]);
        follow = (follow - 2.5).max(ly[i]);
    }
    dct(&mut st.features, &ly);
    st.features[0] -= 4.0;

    // LPC from cepstrum
    lpc_from_cepstrum(&mut st.lpc, &st.features);
    for i in 0..LPC_ORDER {
        st.features[NB_BANDS + 2 + i] = st.lpc[i];
    }

    // Shift excitation and LP buffers
    st.exc_buf.copy_within(FRAME_SIZE.., 0);
    st.lp_buf.copy_within(FRAME_SIZE.., 0);

    // Aligned input
    aligned_in[TRAINING_OFFSET..FRAME_SIZE].copy_from_slice(&input[..FRAME_SIZE - TRAINING_OFFSET]);

    // FIR filtering for LP residual
    let mut x = vec![0.0f32; FRAME_SIZE + LPC_ORDER];
    x[..LPC_ORDER].copy_from_slice(&st.pitch_mem);
    x[LPC_ORDER..LPC_ORDER + FRAME_SIZE].copy_from_slice(&aligned_in);
    st.pitch_mem
        .copy_from_slice(&aligned_in[FRAME_SIZE - LPC_ORDER..FRAME_SIZE]);

    celt_fir_c(
        &x,
        &st.lpc,
        &mut st.lp_buf[PITCH_MAX_PERIOD..PITCH_MAX_PERIOD + FRAME_SIZE],
        LPC_ORDER,
    );

    for i in 0..FRAME_SIZE {
        st.exc_buf[PITCH_MAX_PERIOD + i] = st.lp_buf[PITCH_MAX_PERIOD + i] + 0.7 * st.pitch_filt;
        st.pitch_filt = st.lp_buf[PITCH_MAX_PERIOD + i];
    }

    // Low-pass biquad filter
    let lp_b: [f32; 2] = [-0.84946, 1.0];
    let lp_a: [f32; 2] = [-1.54220, 0.70781];
    // Need to biquad in-place on lp_buf[PITCH_MAX_PERIOD..]
    let mut lp_tmp = vec![0.0f32; FRAME_SIZE];
    lp_tmp.copy_from_slice(&st.lp_buf[PITCH_MAX_PERIOD..PITCH_MAX_PERIOD + FRAME_SIZE]);
    biquad(
        &mut st.lp_buf[PITCH_MAX_PERIOD..PITCH_MAX_PERIOD + FRAME_SIZE],
        &mut st.lp_mem,
        &lp_tmp,
        &lp_b,
        &lp_a,
        FRAME_SIZE,
    );

    // Pitch cross-correlation
    let mut xcorr = vec![0.0f32; PITCH_MAX_PERIOD];
    let buf = &st.exc_buf;
    celt_pitch_xcorr(
        &buf[PITCH_MAX_PERIOD..PITCH_MAX_PERIOD + FRAME_SIZE],
        buf,
        &mut xcorr,
        PITCH_MAX_PERIOD - PITCH_MIN_PERIOD,
    );

    let ener0 = celt_inner_prod(
        &buf[PITCH_MAX_PERIOD..],
        &buf[PITCH_MAX_PERIOD..],
        FRAME_SIZE,
    );
    let mut ener1: f64 = celt_inner_prod(buf, buf, FRAME_SIZE) as f64;
    let mut ener_norm = vec![0.0f32; PITCH_MAX_PERIOD - PITCH_MIN_PERIOD];

    for i in 0..PITCH_MAX_PERIOD - PITCH_MIN_PERIOD {
        let ener = 1.0 + ener0 as f64 + ener1;
        st.xcorr_features[i] = 2.0 * xcorr[i];
        ener_norm[i] = ener as f32;
        ener1 +=
            buf[i + FRAME_SIZE] as f64 * buf[i + FRAME_SIZE] as f64 - buf[i] as f64 * buf[i] as f64;
    }
    for i in 0..PITCH_MAX_PERIOD - PITCH_MIN_PERIOD {
        st.xcorr_features[i] /= ener_norm[i];
    }

    // Neural pitch estimation
    st.dnn_pitch = compute_pitchdnn(&mut st.pitchdnn, &st.if_features, &st.xcorr_features);

    let pitch = (0.5 + 256.0 / 2.0f64.powf((1.0 / 60.0) * ((st.dnn_pitch as f64 + 1.5) * 60.0)))
        .floor() as i32;

    // Frame correlation
    let xx = celt_inner_prod(
        &st.lp_buf[PITCH_MAX_PERIOD..],
        &st.lp_buf[PITCH_MAX_PERIOD..],
        FRAME_SIZE,
    );
    let pitch_u = pitch as usize;
    let yy = celt_inner_prod(
        &st.lp_buf[PITCH_MAX_PERIOD - pitch_u..],
        &st.lp_buf[PITCH_MAX_PERIOD - pitch_u..],
        FRAME_SIZE,
    );
    let xy = celt_inner_prod(
        &st.lp_buf[PITCH_MAX_PERIOD..],
        &st.lp_buf[PITCH_MAX_PERIOD - pitch_u..],
        FRAME_SIZE,
    );
    let frame_corr = xy / (1.0 + xx * yy).sqrt();
    let frame_corr = (1.0 + (5.0 * frame_corr).exp()).ln() / (1.0 + (5.0f32).exp()).ln();

    st.features[NB_BANDS] = st.dnn_pitch;
    st.features[NB_BANDS + 1] = frame_corr - 0.5;
}

/// Compute features from int16 PCM.
///
/// Upstream C: dnn/lpcnet_enc.c:lpcnet_compute_single_frame_features
pub fn lpcnet_compute_single_frame_features(
    st: &mut LPCNetEncState,
    pcm: &[i16],
    features: &mut [f32],
) {
    let mut x = vec![0.0f32; FRAME_SIZE];
    for i in 0..FRAME_SIZE {
        x[i] = pcm[i] as f32;
    }
    // In-place preemphasis: C does preemphasis(x, mem, x, coef, N)
    let tmp = x.clone();
    preemphasis(&mut x, &mut st.mem_preemph, &tmp, PREEMPHASIS, FRAME_SIZE);
    compute_frame_features(st, &x);
    features[..NB_TOTAL_FEATURES].copy_from_slice(&st.features[..NB_TOTAL_FEATURES]);
}

/// Compute features from float PCM.
///
/// Upstream C: dnn/lpcnet_enc.c:lpcnet_compute_single_frame_features_float
pub fn lpcnet_compute_single_frame_features_float(
    st: &mut LPCNetEncState,
    pcm: &[f32],
    features: &mut [f32],
) {
    let mut x = vec![0.0f32; FRAME_SIZE];
    x[..FRAME_SIZE].copy_from_slice(&pcm[..FRAME_SIZE]);
    let tmp = x.clone();
    preemphasis(&mut x, &mut st.mem_preemph, &tmp, PREEMPHASIS, FRAME_SIZE);
    compute_frame_features(st, &x);
    features[..NB_TOTAL_FEATURES].copy_from_slice(&st.features[..NB_TOTAL_FEATURES]);
}

// --- PLC State ---

/// LPCNet PLC state: combines encoder, FARGAN, and PLC model.
///
/// Upstream C: dnn/lpcnet_private.h:LPCNetPLCState
pub struct LPCNetPLCState {
    pub model: PLCModel,
    pub fargan: FARGANState,
    pub enc: LPCNetEncState,
    pub loaded: bool,

    // Fields that get reset (LPCNET_PLC_RESET_START)
    pub fec: Vec<Vec<f32>>,
    pub analysis_gap: bool,
    pub fec_read_pos: usize,
    pub fec_fill_pos: usize,
    pub fec_skip: usize,
    pub analysis_pos: usize,
    pub predict_pos: usize,
    pub pcm: Vec<f32>,
    pub blend: i32,
    pub features: Vec<f32>,
    pub cont_features: Vec<f32>,
    pub loss_count: i32,
    plc_net: PLCNetState,
    plc_bak: [PLCNetState; 2],
}

impl Default for LPCNetPLCState {
    fn default() -> Self {
        Self::new()
    }
}

impl LPCNetPLCState {
    pub fn new() -> Self {
        LPCNetPLCState {
            model: PLCModel::default(),
            fargan: FARGANState::new(),
            enc: LPCNetEncState::new(),
            loaded: false,
            fec: vec![vec![0.0; NB_FEATURES]; PLC_MAX_FEC],
            analysis_gap: true,
            fec_read_pos: 0,
            fec_fill_pos: 0,
            fec_skip: 0,
            analysis_pos: PLC_BUF_SIZE,
            predict_pos: PLC_BUF_SIZE,
            pcm: vec![0.0; PLC_BUF_SIZE],
            blend: 0,
            features: vec![0.0; NB_TOTAL_FEATURES],
            cont_features: vec![0.0; CONT_VECTORS * NB_FEATURES],
            loss_count: 0,
            plc_net: PLCNetState::new(),
            plc_bak: [PLCNetState::new(), PLCNetState::new()],
        }
    }

    /// Reset PLC state (preserves model weights and FARGAN state).
    ///
    /// Upstream C: dnn/lpcnet_plc.c:lpcnet_plc_reset
    pub fn reset(&mut self) {
        for row in &mut self.fec {
            row.fill(0.0);
        }
        self.analysis_gap = true;
        self.fec_read_pos = 0;
        self.fec_fill_pos = 0;
        self.fec_skip = 0;
        self.analysis_pos = PLC_BUF_SIZE;
        self.predict_pos = PLC_BUF_SIZE;
        self.pcm.fill(0.0);
        self.blend = 0;
        self.features.fill(0.0);
        self.cont_features.fill(0.0);
        self.loss_count = 0;
        self.plc_net = PLCNetState::new();
        self.plc_bak = [PLCNetState::new(), PLCNetState::new()];
        self.enc.init();
    }

    /// Initialize PLC state with compiled-in weights.
    ///
    /// Upstream C: dnn/lpcnet_plc.c:lpcnet_plc_init
    pub fn init(&mut self, arrays: &[WeightArray]) -> bool {
        self.fargan = FARGANState::new();
        self.fargan.init(arrays);
        self.enc.init();
        self.enc.load_model(arrays);
        match init_plcmodel(arrays) {
            Some(model) => {
                self.model = model;
                self.loaded = true;
                self.reset();
                true
            }
            None => false,
        }
    }

    /// Load model from binary weight blob.
    ///
    /// Upstream C: dnn/lpcnet_plc.c:lpcnet_plc_load_model
    pub fn load_model(&mut self, arrays: &[WeightArray]) -> bool {
        let model = match init_plcmodel(arrays) {
            Some(m) => m,
            None => return false,
        };
        self.model = model;
        if !self.enc.load_model(arrays) {
            return false;
        }
        if !self.fargan.init(arrays) {
            return false;
        }
        self.loaded = true;
        true
    }
}

/// Add FEC features to the PLC state.
///
/// Upstream C: dnn/lpcnet_plc.c:lpcnet_plc_fec_add
pub fn lpcnet_plc_fec_add(st: &mut LPCNetPLCState, features: Option<&[f32]>) {
    match features {
        None => {
            st.fec_skip += 1;
        }
        Some(f) => {
            if st.fec_fill_pos == PLC_MAX_FEC {
                // Shift buffer
                for i in st.fec_read_pos..st.fec_fill_pos {
                    let src = st.fec[i].clone();
                    st.fec[i - st.fec_read_pos] = src;
                }
                st.fec_fill_pos -= st.fec_read_pos;
                st.fec_read_pos = 0;
            }
            st.fec[st.fec_fill_pos][..NB_FEATURES].copy_from_slice(&f[..NB_FEATURES]);
            st.fec_fill_pos += 1;
        }
    }
}

/// Clear FEC buffer.
///
/// Upstream C: dnn/lpcnet_plc.c:lpcnet_plc_fec_clear
pub fn lpcnet_plc_fec_clear(st: &mut LPCNetPLCState) {
    st.fec_read_pos = 0;
    st.fec_fill_pos = 0;
    st.fec_skip = 0;
}

/// Compute PLC prediction from features.
fn compute_plc_pred(st: &mut LPCNetPLCState, out: &mut [f32], input: &[f32]) {
    assert!(st.loaded);
    let mut tmp = vec![0.0f32; PLC_DENSE_IN_OUT_SIZE];
    compute_generic_dense(&st.model.plc_dense_in, &mut tmp, input, ACTIVATION_TANH);
    compute_generic_gru(
        &st.model.plc_gru1_input,
        &st.model.plc_gru1_recurrent,
        &mut st.plc_net.gru1_state,
        &tmp,
    );
    compute_generic_gru(
        &st.model.plc_gru2_input,
        &st.model.plc_gru2_recurrent,
        &mut st.plc_net.gru2_state,
        &st.plc_net.gru1_state.clone(),
    );
    compute_generic_dense(
        &st.model.plc_dense_out,
        out,
        &st.plc_net.gru2_state,
        ACTIVATION_LINEAR,
    );
}

/// Try to get FEC features, or fall back to prediction.
///
/// Returns true if FEC was available.
fn get_fec_or_pred(st: &mut LPCNetPLCState, out: &mut [f32]) -> bool {
    if st.fec_read_pos != st.fec_fill_pos && st.fec_skip == 0 {
        out[..NB_FEATURES].copy_from_slice(&st.fec[st.fec_read_pos][..NB_FEATURES]);
        st.fec_read_pos += 1;
        // Update PLC state using FEC (without Burg features)
        let mut plc_features = vec![0.0f32; 2 * NB_BANDS + NB_FEATURES + 1];
        plc_features[2 * NB_BANDS..2 * NB_BANDS + NB_FEATURES].copy_from_slice(&out[..NB_FEATURES]);
        plc_features[2 * NB_BANDS + NB_FEATURES] = -1.0;
        let mut discard = vec![0.0f32; NB_FEATURES];
        compute_plc_pred(st, &mut discard, &plc_features);
        true
    } else {
        let zeros = vec![0.0f32; 2 * NB_BANDS + NB_FEATURES + 1];
        compute_plc_pred(st, out, &zeros);
        if st.fec_skip > 0 {
            st.fec_skip -= 1;
        }
        false
    }
}

/// Queue features into the continuation feature buffer.
fn queue_features(st: &mut LPCNetPLCState, features: &[f32]) {
    st.cont_features.copy_within(NB_FEATURES.., 0);
    st.cont_features[(CONT_VECTORS - 1) * NB_FEATURES..].copy_from_slice(&features[..NB_FEATURES]);
}

/// Update PLC state with a good (received) frame.
///
/// Upstream C: dnn/lpcnet_plc.c:lpcnet_plc_update
pub fn lpcnet_plc_update(st: &mut LPCNetPLCState, pcm: &[i16]) {
    if st.analysis_pos >= FRAME_SIZE {
        st.analysis_pos -= FRAME_SIZE;
    } else {
        st.analysis_gap = true;
    }
    if st.predict_pos >= FRAME_SIZE {
        st.predict_pos -= FRAME_SIZE;
    }
    st.pcm.copy_within(FRAME_SIZE.., 0);
    for i in 0..FRAME_SIZE {
        st.pcm[PLC_BUF_SIZE - FRAME_SIZE + i] = pcm[i] as f32 / 32768.0;
    }
    st.loss_count = 0;
    st.blend = 0;
}

/// Attenuation table for increasing loss count.
static ATT_TABLE: [f32; 10] = [0.0, 0.0, -0.2, -0.2, -0.4, -0.4, -0.8, -0.8, -1.6, -1.6];

/// Conceal a lost frame using neural PLC.
///
/// Upstream C: dnn/lpcnet_plc.c:lpcnet_plc_conceal
pub fn lpcnet_plc_conceal(st: &mut LPCNetPLCState, pcm: &mut [i16]) {
    assert!(st.loaded);

    if st.blend == 0 {
        let mut count = 0;
        st.plc_net = st.plc_bak[0].clone();

        while st.analysis_pos + FRAME_SIZE <= PLC_BUF_SIZE {
            let mut x = vec![0.0f32; FRAME_SIZE];
            assert!(st.analysis_pos < PLC_BUF_SIZE);
            for i in 0..FRAME_SIZE {
                x[i] = 32768.0 * st.pcm[st.analysis_pos + i];
            }
            let mut plc_features = vec![0.0f32; 2 * NB_BANDS + NB_FEATURES + 1];
            burg_cepstral_analysis(&mut plc_features, &x);

            lpcnet_compute_single_frame_features_float(&mut st.enc, &x, &mut st.features);

            if (!st.analysis_gap || count > 0) && st.analysis_pos >= st.predict_pos {
                let features_copy = st.features.clone();
                queue_features(st, &features_copy);
                plc_features[2 * NB_BANDS..2 * NB_BANDS + NB_FEATURES]
                    .copy_from_slice(&st.features[..NB_FEATURES]);
                plc_features[2 * NB_BANDS + NB_FEATURES] = 1.0;
                st.plc_bak[0] = st.plc_bak[1].clone();
                st.plc_bak[1] = st.plc_net.clone();
                // C: compute_plc_pred(st, st->features, plc_features) — output to st->features, but discarded
                let mut feat_discard = vec![0.0f32; NB_FEATURES];
                compute_plc_pred(st, &mut feat_discard, &plc_features);
            }
            st.analysis_pos += FRAME_SIZE;
            count += 1;
        }

        // Two prediction steps
        st.plc_bak[0] = st.plc_bak[1].clone();
        st.plc_bak[1] = st.plc_net.clone();
        let mut feat_tmp = vec![0.0f32; NB_FEATURES];
        get_fec_or_pred(st, &mut feat_tmp);
        st.features[..NB_FEATURES].copy_from_slice(&feat_tmp);
        let features_copy = st.features.clone();
        queue_features(st, &features_copy);

        st.plc_bak[0] = st.plc_bak[1].clone();
        st.plc_bak[1] = st.plc_net.clone();
        get_fec_or_pred(st, &mut feat_tmp);
        st.features[..NB_FEATURES].copy_from_slice(&feat_tmp);
        let features_copy = st.features.clone();
        queue_features(st, &features_copy);

        let pcm_slice = st.pcm[PLC_BUF_SIZE - FARGAN_CONT_SAMPLES..].to_vec();
        let cont_features = st.cont_features.clone();
        fargan_cont(&mut st.fargan, &pcm_slice, &cont_features);
        st.analysis_gap = false;
    }

    st.plc_bak[0] = st.plc_bak[1].clone();
    st.plc_bak[1] = st.plc_net.clone();
    let mut feat_tmp = vec![0.0f32; NB_FEATURES];
    if get_fec_or_pred(st, &mut feat_tmp) {
        st.loss_count = 0;
    } else {
        st.loss_count += 1;
    }
    st.features[..NB_FEATURES].copy_from_slice(&feat_tmp);

    // Attenuation
    if st.loss_count >= 10 {
        st.features[0] =
            (-10.0f32).max(st.features[0] + ATT_TABLE[9] - 2.0 * (st.loss_count - 9) as f32);
    } else {
        st.features[0] = (-10.0f32).max(st.features[0] + ATT_TABLE[st.loss_count as usize]);
    }

    let features_copy = st.features.clone();
    fargan_synthesize_int(&mut st.fargan, pcm, &features_copy);
    let features_copy = st.features.clone();
    queue_features(st, &features_copy);

    if st.analysis_pos >= FRAME_SIZE {
        st.analysis_pos -= FRAME_SIZE;
    } else {
        st.analysis_gap = true;
    }
    st.predict_pos = PLC_BUF_SIZE;
    st.pcm.copy_within(FRAME_SIZE.., 0);
    for i in 0..FRAME_SIZE {
        st.pcm[PLC_BUF_SIZE - FRAME_SIZE + i] = pcm[i] as f32 / 32768.0;
    }
    st.blend = 1;
}
