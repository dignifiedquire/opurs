//! Full DRED encoder: extracts features, encodes to RDOVAE latents,
//! and writes entropy-coded DRED data.
//!
//! Upstream C: `dnn/dred_encoder.c`, `dnn/dred_encoder.h`

use crate::arch::Arch;
use crate::celt::entcode::{ec_ctx_saved, ec_tell};
use crate::celt::entenc::{
    ec_enc, ec_enc_done, ec_enc_init, ec_enc_shrink, ec_enc_uint, ec_encode,
};
use crate::celt::laplace::ec_laplace_encode_p0;
use crate::dnn::lpcnet::{lpcnet_compute_single_frame_features_float, LPCNetEncState};
use crate::dnn::nnet::*;

use super::coding::compute_quantizer;
use super::config::*;
use super::rdovae_enc::{dred_rdovae_encode_dframe, init_rdovaeenc, RDOVAEEnc, RDOVAEEncState};
use super::stats::*;

const RESAMPLING_ORDER: usize = 8;
#[cfg(feature = "qext")]
const MAX_DOWNMIX_BUFFER: usize = 1920 * 2;
#[cfg(not(feature = "qext"))]
const MAX_DOWNMIX_BUFFER: usize = 960 * 2;

/// Full DRED encoder state.
///
/// Upstream C: dnn/dred_encoder.h:DREDEnc
#[derive(Clone)]
pub struct DREDEnc {
    pub model: RDOVAEEnc,
    pub lpcnet_enc_state: LPCNetEncState,
    pub rdovae_enc: RDOVAEEncState,
    pub loaded: bool,
    pub fs: i32,
    pub channels: i32,

    // Fields that get reset (DREDENC_RESET_START)
    pub input_buffer: Vec<f32>,
    pub input_buffer_fill: usize,
    pub dred_offset: i32,
    pub latent_offset: usize,
    pub last_extra_dred_offset: i32,
    pub latents_buffer: Vec<f32>,
    pub latents_buffer_fill: usize,
    pub state_buffer: Vec<f32>,
    pub resample_mem: Vec<f32>,
}

impl Default for DREDEnc {
    fn default() -> Self {
        Self::new()
    }
}

impl DREDEnc {
    pub fn new() -> Self {
        DREDEnc {
            model: RDOVAEEnc::default(),
            lpcnet_enc_state: LPCNetEncState::new(),
            rdovae_enc: RDOVAEEncState::new(),
            loaded: false,
            fs: 48000,
            channels: 1,
            input_buffer: vec![0.0; 2 * DRED_DFRAME_SIZE],
            input_buffer_fill: 0,
            dred_offset: 0,
            latent_offset: 0,
            last_extra_dred_offset: 0,
            latents_buffer: vec![0.0; DRED_MAX_FRAMES * DRED_LATENT_DIM],
            latents_buffer_fill: 0,
            state_buffer: vec![0.0; DRED_MAX_FRAMES * DRED_STATE_DIM],
            resample_mem: vec![0.0; RESAMPLING_ORDER + 1],
        }
    }

    /// Load model from weight arrays.
    ///
    /// Upstream C: dnn/dred_encoder.c:dred_encoder_load_model
    pub fn load_model(&mut self, arrays: &[WeightArray]) -> bool {
        match init_rdovaeenc(arrays) {
            Some(model) => {
                self.model = model;
                if !self.lpcnet_enc_state.load_model(arrays) {
                    return false;
                }
                self.loaded = true;
                true
            }
            None => false,
        }
    }

    /// Reset encoder state (preserves model weights).
    ///
    /// Upstream C: dnn/dred_encoder.c:dred_encoder_reset
    pub fn reset(&mut self) {
        self.input_buffer.fill(0.0);
        self.input_buffer_fill = DRED_SILK_ENCODER_DELAY as usize;
        self.dred_offset = 0;
        self.latent_offset = 0;
        self.last_extra_dred_offset = 0;
        self.latents_buffer.fill(0.0);
        self.latents_buffer_fill = 0;
        self.state_buffer.fill(0.0);
        self.resample_mem.fill(0.0);
        self.lpcnet_enc_state.init();
        self.rdovae_enc = RDOVAEEncState::new();
    }

    /// Initialize encoder for given sample rate and channels.
    ///
    /// Upstream C: dnn/dred_encoder.c:dred_encoder_init
    pub fn init(&mut self, fs: i32, channels: i32) {
        self.fs = fs;
        self.channels = channels;
        self.loaded = false;
        #[cfg(feature = "builtin-weights")]
        {
            let arrays = super::rdovae_enc_data::rdovaeenc_arrays();
            if let Some(model) = init_rdovaeenc(&arrays) {
                self.model = model;
                self.loaded = true;
            }
        }
        self.reset();
    }
}

/// Direct-form II transposed filter.
///
/// Upstream C: dnn/dred_encoder.c:filter_df2t
fn filter_df2t(
    input: &[f32],
    output: &mut [f32],
    len: usize,
    b0: f32,
    b: &[f32],
    a: &[f32],
    order: usize,
    mem: &mut [f32],
) {
    for i in 0..len {
        let xi = input[i];
        let yi = xi * b0 + mem[0];
        let nyi = -yi;
        for j in 0..order {
            mem[j] = mem[j + 1] + b[j] * xi + a[j] * nyi;
        }
        output[i] = yi;
    }
}

/// Convert audio from encoder sample rate to 16kHz.
///
/// Upstream C: dnn/dred_encoder.c:dred_convert_to_16k
fn dred_convert_to_16k(
    fs: i32,
    channels: i32,
    resample_mem: &mut [f32],
    input: &[f32],
    in_len: usize,
    output: &mut [f32],
    out_len: usize,
) {
    let up = match fs {
        8000 => 2,
        12000 => 4,
        16000 => 1,
        24000 => 2,
        48000 => 1,
        #[cfg(feature = "qext")]
        96000 => 1,
        _ => {
            debug_assert!(false, "Unsupported sample rate");
            1
        }
    };

    let mut downmix = vec![0.0f32; MAX_DOWNMIX_BUFFER];
    if channels == 1 {
        for i in 0..in_len {
            downmix[up * i] = (up as f32 * input[i] * 32768.0)
                .clamp(-32768.0, 32767.0)
                .round();
        }
    } else {
        for i in 0..in_len {
            downmix[up * i] = (0.5 * up as f32 * (input[2 * i] + input[2 * i + 1]) * 32768.0)
                .clamp(-32768.0, 32767.0)
                .round();
        }
    }

    if fs == 16000 {
        output[..out_len].copy_from_slice(&downmix[..out_len]);
    } else if fs == 48000 || fs == 24000 {
        // ellip(7, .2, 70, 7750/24000)
        static FILTER_B: [f32; 8] = [
            0.005873358047,
            0.012980854831,
            0.014531340042,
            0.014531340042,
            0.012980854831,
            0.005873358047,
            0.004523418224,
            0.0,
        ];
        static FILTER_A: [f32; 8] = [
            -3.878718597768,
            7.748834257468,
            -9.653651699533,
            8.007342726666,
            -4.379450178552,
            1.463182111810,
            -0.231720677804,
            0.0,
        ];
        let b0 = 0.004523418224f32;
        filter_df2t(
            &downmix.clone(),
            &mut downmix,
            up * in_len,
            b0,
            &FILTER_B,
            &FILTER_A,
            RESAMPLING_ORDER,
            resample_mem,
        );
        for i in 0..out_len {
            output[i] = downmix[3 * i];
        }
    } else if fs == 12000 {
        static FILTER_B: [f32; 8] = [
            -0.001017101081,
            0.003673127243,
            0.001009165267,
            0.001009165267,
            0.003673127243,
            -0.001017101081,
            0.002033596776,
            0.0,
        ];
        static FILTER_A: [f32; 8] = [
            -4.930414411612,
            11.291643096504,
            -15.322037343815,
            13.216403930898,
            -7.220409219553,
            2.310550142771,
            -0.334338618782,
            0.0,
        ];
        let b0 = 0.002033596776f32;
        filter_df2t(
            &downmix.clone(),
            &mut downmix,
            up * in_len,
            b0,
            &FILTER_B,
            &FILTER_A,
            RESAMPLING_ORDER,
            resample_mem,
        );
        for i in 0..out_len {
            output[i] = downmix[3 * i];
        }
    } else if fs == 8000 {
        static FILTER_B: [f32; 8] = [
            0.081670120929,
            0.180401598565,
            0.259391051971,
            0.259391051971,
            0.180401598565,
            0.081670120929,
            0.020109185709,
            0.0,
        ];
        static FILTER_A: [f32; 8] = [
            -1.393651933659,
            2.609789872676,
            -2.403541968806,
            2.056814957331,
            -1.148908574570,
            0.473001413788,
            -0.110359852412,
            0.0,
        ];
        let b0 = 0.020109185709f32;
        filter_df2t(
            &downmix,
            output,
            out_len,
            b0,
            &FILTER_B,
            &FILTER_A,
            RESAMPLING_ORDER,
            resample_mem,
        );
    } else if fs == 96000 {
        static FILTER_B: [f32; 8] = [
            -0.002160290245,
            0.002887088080,
            -0.001214921271,
            -0.001214921271,
            0.002887088080,
            -0.002160290245,
            0.000880286074,
            0.0,
        ];
        static FILTER_A: [f32; 8] = [
            -5.813483928050,
            14.932091805554,
            -21.900933283269,
            19.774128964756,
            -10.978028462771,
            3.467650469467,
            -0.480641240411,
            0.0,
        ];
        let b0 = 0.000880286074f32;
        filter_df2t(
            &downmix.clone(),
            &mut downmix,
            up * in_len,
            b0,
            &FILTER_B,
            &FILTER_A,
            RESAMPLING_ORDER,
            resample_mem,
        );
        for i in 0..out_len {
            output[i] = downmix[6 * i];
        }
    }
}

/// Process one double frame through RDOVAE.
///
/// Upstream C: dnn/dred_encoder.c:dred_process_frame
fn dred_process_frame(enc: &mut DREDEnc, arch: Arch) {
    debug_assert!(enc.loaded, "libopus: assert(enc->loaded) called");
    let mut feature_buffer = vec![0.0f32; 2 * 36];
    let mut input_buffer = vec![0.0f32; 2 * DRED_NUM_FEATURES];

    // Shift latents and state buffers
    enc.latents_buffer
        .copy_within(..((DRED_MAX_FRAMES - 1) * DRED_LATENT_DIM), DRED_LATENT_DIM);
    enc.state_buffer
        .copy_within(..((DRED_MAX_FRAMES - 1) * DRED_STATE_DIM), DRED_STATE_DIM);

    // Calculate LPCNet features for two frames
    let buf0 = enc.input_buffer[..DRED_FRAME_SIZE].to_vec();
    lpcnet_compute_single_frame_features_float(
        &mut enc.lpcnet_enc_state,
        &buf0,
        &mut feature_buffer,
        arch,
    );
    let buf1 = enc.input_buffer[DRED_FRAME_SIZE..2 * DRED_FRAME_SIZE].to_vec();
    lpcnet_compute_single_frame_features_float(
        &mut enc.lpcnet_enc_state,
        &buf1,
        &mut feature_buffer[36..],
        arch,
    );

    // Prepare input (discard LPC coefficients, keep first 20 features)
    input_buffer[..DRED_NUM_FEATURES].copy_from_slice(&feature_buffer[..DRED_NUM_FEATURES]);
    input_buffer[DRED_NUM_FEATURES..2 * DRED_NUM_FEATURES]
        .copy_from_slice(&feature_buffer[36..36 + DRED_NUM_FEATURES]);

    // Run RDOVAE encoder
    dred_rdovae_encode_dframe(
        &mut enc.rdovae_enc,
        &enc.model,
        &mut enc.latents_buffer[..DRED_LATENT_DIM],
        &mut enc.state_buffer[..DRED_STATE_DIM],
        &input_buffer,
        arch,
    );
    enc.latents_buffer_fill = (enc.latents_buffer_fill + 1).min(DRED_NUM_REDUNDANCY_FRAMES);
}

/// Compute latents from PCM audio.
///
/// Upstream C: dnn/dred_encoder.c:dred_compute_latents
pub fn dred_compute_latents(
    enc: &mut DREDEnc,
    pcm: &[f32],
    frame_size: usize,
    extra_delay: usize,
    arch: Arch,
) {
    debug_assert!(enc.loaded, "libopus: assert(enc->loaded) called");
    let frame_size16k = frame_size * 16000 / enc.fs as usize;
    let curr_offset16k = 40 + extra_delay * 16000 / enc.fs as usize - enc.input_buffer_fill;
    enc.dred_offset = (curr_offset16k as f32 + 20.0).floor() as i32 / 40;
    enc.latent_offset = 0;

    let mut remaining16k = frame_size16k;
    let mut pcm_offset = 0;
    while remaining16k > 0 {
        let process_size16k = remaining16k.min(2 * DRED_FRAME_SIZE);
        let process_size = process_size16k * enc.fs as usize / 16000;
        let fill = enc.input_buffer_fill;
        dred_convert_to_16k(
            enc.fs,
            enc.channels,
            &mut enc.resample_mem,
            &pcm[pcm_offset..],
            process_size,
            &mut enc.input_buffer[fill..],
            process_size16k,
        );
        enc.input_buffer_fill += process_size16k;
        if enc.input_buffer_fill >= 2 * DRED_FRAME_SIZE {
            dred_process_frame(enc, arch);
            enc.input_buffer_fill -= 2 * DRED_FRAME_SIZE;
            let fill = enc.input_buffer_fill;
            enc.input_buffer
                .copy_within(2 * DRED_FRAME_SIZE..2 * DRED_FRAME_SIZE + fill, 0);
            // 15 ms (6*2.5 ms) is the ideal offset for DRED
            if enc.dred_offset < 6 {
                enc.dred_offset += 8;
            } else {
                enc.latent_offset += 1;
            }
        }
        pcm_offset += process_size;
        remaining16k -= process_size16k;
    }
}

/// Encode latents using Laplace coding.
///
/// Upstream C: dnn/dred_encoder.c:dred_encode_latents
fn dred_encode_latents(
    ec: &mut ec_enc,
    x: &[f32],
    scale: &[u8],
    dzone: &[u8],
    r: &[u8],
    p0: &[u8],
    dim: usize,
    arch: Arch,
) {
    let eps = 0.1f32;
    let mut xq = vec![0.0f32; dim];
    let mut deadzone = vec![0.0f32; dim];

    for i in 0..dim {
        let delta = dzone[i] as f32 * (1.0 / 256.0);
        xq[i] = x[i] * scale[i] as f32 * (1.0 / 256.0);
        deadzone[i] = xq[i] / (delta + eps);
    }
    let input_copy = deadzone.clone();
    compute_activation(&mut deadzone, &input_copy, dim, ACTIVATION_TANH, arch);
    for i in 0..dim {
        let delta = dzone[i] as f32 * (1.0 / 256.0);
        xq[i] -= delta * deadzone[i];
    }
    let mut q = vec![0i32; dim];
    for i in 0..dim {
        q[i] = (0.5 + xq[i]).floor() as i32;
    }
    for i in 0..dim {
        if r[i] == 0 || p0[i] == 255 {
            q[i] = 0;
        } else {
            ec_laplace_encode_p0(ec, q[i], (p0[i] as u16) << 7, (r[i] as u16) << 7);
        }
    }
}

/// Check if voice is active at a given offset.
fn dred_voice_active(activity_mem: &[u8], offset: usize) -> bool {
    for i in 0..16 {
        if activity_mem[8 * offset + i] == 1 {
            return true;
        }
    }
    false
}

/// Encode DRED data into a byte buffer using entropy coding.
///
/// Returns number of bytes written, or 0 if DRED should not be sent.
///
/// Upstream C: dnn/dred_encoder.c:dred_encode_silk_frame
pub fn dred_encode_silk_frame(
    enc: &mut DREDEnc,
    buf: &mut [u8],
    max_chunks: usize,
    max_bytes: usize,
    q0: i32,
    dq: i32,
    qmax: i32,
    activity_mem: &[u8],
    arch: Arch,
) -> usize {
    let mut latent_offset = enc.latent_offset;
    let mut extra_dred_offset = 0i32;
    let mut delayed_dred = false;

    // Delay new DRED data when just out of silence
    if activity_mem[0] != 0 && enc.last_extra_dred_offset > 0 {
        latent_offset = enc.last_extra_dred_offset as usize;
        delayed_dred = true;
        enc.last_extra_dred_offset = 0;
    }
    while latent_offset < enc.latents_buffer_fill && !dred_voice_active(activity_mem, latent_offset)
    {
        latent_offset += 1;
        extra_dred_offset += 1;
    }
    if !delayed_dred {
        enc.last_extra_dred_offset = extra_dred_offset;
    }

    // Entropy coding
    let mut ec_encoder = ec_enc_init(&mut buf[..max_bytes]);
    ec_enc_uint(&mut ec_encoder, q0 as u32, 16);
    ec_enc_uint(&mut ec_encoder, dq as u32, 8);
    let total_offset = 16 - (enc.dred_offset - extra_dred_offset * 8);
    assert!(total_offset >= 0);
    if total_offset > 31 {
        ec_enc_uint(&mut ec_encoder, 1, 2);
        ec_enc_uint(&mut ec_encoder, (total_offset >> 5) as u32, 256);
        ec_enc_uint(&mut ec_encoder, (total_offset & 31) as u32, 32);
    } else {
        ec_enc_uint(&mut ec_encoder, 0, 2);
        ec_enc_uint(&mut ec_encoder, total_offset as u32, 32);
    }

    // Encode qmax
    assert!(qmax >= q0);
    if q0 < 14 && dq > 0 {
        assert!(qmax > q0);
        let nvals = (15 - (q0 + 1)) as u32;
        if qmax >= 15 {
            ec_encode(&mut ec_encoder, 0, nvals, 2 * nvals);
        } else {
            let s = nvals + (qmax - (q0 + 1)) as u32;
            ec_encode(&mut ec_encoder, s, s + 1, 2 * nvals);
        }
    }

    // Encode initial state
    let state_qoffset = (q0 as usize) * DRED_STATE_DIM;
    dred_encode_latents(
        &mut ec_encoder,
        &enc.state_buffer[latent_offset * DRED_STATE_DIM..],
        &DRED_STATE_QUANT_SCALES_Q8[state_qoffset..],
        &DRED_STATE_DEAD_ZONE_Q8[state_qoffset..],
        &DRED_STATE_R_Q8[state_qoffset..],
        &DRED_STATE_P0_Q8[state_qoffset..],
        DRED_STATE_DIM,
        arch,
    );
    if ec_tell(&ec_encoder) > 8 * max_bytes as i32 {
        return 0;
    }

    let mut ec_bak: ec_ctx_saved = ec_encoder.save();
    let mut prev_active = false;
    let mut dred_encoded = 0;
    let limit = (2 * max_chunks).min(enc.latents_buffer_fill - latent_offset - 1);
    let mut i = 0;
    while i < limit {
        let q_level = compute_quantizer(q0, dq, qmax, (i / 2) as i32);
        let offset = q_level as usize * DRED_LATENT_DIM;
        dred_encode_latents(
            &mut ec_encoder,
            &enc.latents_buffer[(i + latent_offset) * DRED_LATENT_DIM..],
            &DRED_LATENT_QUANT_SCALES_Q8[offset..],
            &DRED_LATENT_DEAD_ZONE_Q8[offset..],
            &DRED_LATENT_R_Q8[offset..],
            &DRED_LATENT_P0_Q8[offset..],
            DRED_LATENT_DIM,
            arch,
        );
        if ec_tell(&ec_encoder) > 8 * max_bytes as i32 {
            if i == 0 {
                return 0;
            }
            break;
        }
        let active = dred_voice_active(activity_mem, i + latent_offset);
        if active || prev_active {
            ec_bak = ec_encoder.save();
            dred_encoded = i + 2;
        }
        prev_active = active;
        i += 2;
    }

    if dred_encoded == 0 || (dred_encoded <= 2 && extra_dred_offset > 0) {
        return 0;
    }
    ec_encoder.restore(ec_bak);

    let ec_buffer_fill = ((ec_tell(&ec_encoder) + 7) / 8) as usize;
    ec_enc_shrink(&mut ec_encoder, ec_buffer_fill as u32);
    ec_enc_done(&mut ec_encoder);
    ec_buffer_fill
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "qext")]
    #[test]
    fn dred_convert_to_16k_supports_96k_qext() {
        let in_len = 960usize;
        let out_len = 160usize;
        let input = vec![0.125f32; in_len];
        let mut output = vec![0.0f32; out_len];
        let mut mem = vec![0.0f32; RESAMPLING_ORDER + 1];

        dred_convert_to_16k(96000, 1, &mut mem, &input, in_len, &mut output, out_len);
        assert!(output.iter().all(|v| v.is_finite()));
    }
}
