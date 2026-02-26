//! DRED decoder: parses and decodes DRED data from packet extensions.
//!
//! Upstream C: `dnn/dred_decoder.c`, `dnn/dred_decoder.h`

use crate::arch::opus_select_arch;
use crate::celt::entcode::ec_tell;
use crate::celt::entdec::{ec_dec, ec_dec_init, ec_dec_uint, ec_dec_update, ec_decode};
use crate::celt::laplace::ec_laplace_decode_p0;
use crate::dnn::nnet::WeightArray;

use super::coding::compute_quantizer;
use super::config::*;
use super::rdovae_dec::{dred_rdovae_decode_all, init_rdovaedec, RDOVAEDec};
use super::stats::*;

/// DRED data parsed from a packet.
///
/// Upstream C: dnn/dred_decoder.h:OpusDRED
pub struct OpusDRED {
    pub fec_features: Vec<f32>,
    pub state: Vec<f32>,
    pub latents: Vec<f32>,
    pub nb_latents: i32,
    pub process_stage: i32, // 0=empty, 1=parsed, 2=processed
    pub dred_offset: i32,
}

impl Default for OpusDRED {
    fn default() -> Self {
        Self::new()
    }
}

impl OpusDRED {
    pub fn new() -> Self {
        OpusDRED {
            fec_features: vec![0.0; 2 * DRED_NUM_REDUNDANCY_FRAMES * DRED_NUM_FEATURES],
            state: vec![0.0; DRED_STATE_DIM],
            latents: vec![0.0; (DRED_NUM_REDUNDANCY_FRAMES / 2) * (DRED_LATENT_DIM + 1)],
            nb_latents: 0,
            process_stage: 0,
            dred_offset: 0,
        }
    }
}

/// DRED decoder state (holds RDOVAE decoder model).
///
/// Upstream C: uses OpusDREDDecoder in opus API
pub struct OpusDREDDecoder {
    pub model: RDOVAEDec,
    pub loaded: bool,
}

impl Default for OpusDREDDecoder {
    fn default() -> Self {
        Self::new()
    }
}

impl OpusDREDDecoder {
    pub fn new() -> Self {
        OpusDREDDecoder {
            model: RDOVAEDec::default(),
            loaded: false,
        }
    }

    /// Load model from weight arrays.
    pub fn load_model(&mut self, arrays: &[WeightArray]) -> bool {
        match init_rdovaedec(arrays) {
            Some(model) => {
                self.model = model;
                self.loaded = true;
                true
            }
            None => false,
        }
    }

    /// Load model from compiled-in weight data.
    ///
    /// Requires the `builtin-weights` feature.
    #[cfg(feature = "builtin-weights")]
    pub fn load_dnn_weights(&mut self) -> bool {
        let arrays = super::super::weights::compiled_weights();
        self.load_model(&arrays)
    }

    /// Load model from an external binary weight blob.
    pub fn set_dnn_blob(&mut self, data: &[u8]) -> bool {
        match super::super::weights::load_weights(data) {
            Some(arrays) => self.load_model(&arrays),
            None => false,
        }
    }
}

/// Decode latents from entropy-coded bytes.
///
/// Upstream C: dnn/dred_decoder.c:dred_decode_latents
fn dred_decode_latents(
    ec: &mut ec_dec,
    x: &mut [f32],
    scale: &[u8],
    r: &[u8],
    p0: &[u8],
    dim: usize,
) {
    for i in 0..dim {
        let q = if r[i] == 0 || p0[i] == 255 {
            0
        } else {
            ec_laplace_decode_p0(ec, (p0[i] as u16) << 7, (r[i] as u16) << 7)
        };
        x[i] = q as f32 * 256.0 / (if scale[i] == 0 { 1 } else { scale[i] }) as f32;
    }
}

/// Parse and decode DRED data from entropy-coded bytes.
///
/// Returns number of latent frames decoded.
///
/// Upstream C: dnn/dred_decoder.c:dred_ec_decode
pub fn dred_ec_decode(
    dred: &mut OpusDRED,
    bytes: &[u8],
    num_bytes: usize,
    min_feature_frames: usize,
    dred_frame_offset: i32,
) -> i32 {
    const { assert!(DRED_NUM_REDUNDANCY_FRAMES.is_multiple_of(2)) };

    let mut bytes_buf = bytes[..num_bytes].to_vec();
    let mut ec = ec_dec_init(&mut bytes_buf);
    let q0 = ec_dec_uint(&mut ec, 16) as i32;
    let dq = ec_dec_uint(&mut ec, 8) as i32;
    let extra_offset = if ec_dec_uint(&mut ec, 2) != 0 {
        32 * ec_dec_uint(&mut ec, 256) as i32
    } else {
        0
    };
    dred.dred_offset = 16 - ec_dec_uint(&mut ec, 32) as i32 - extra_offset + dred_frame_offset;

    let mut qmax = 15i32;
    if q0 < 14 && dq > 0 {
        let nvals = (15 - (q0 + 1)) as u32;
        let ft = 2 * nvals;
        let s = ec_decode(&mut ec, ft);
        if s >= nvals {
            qmax = q0 + (s as i32 - nvals as i32) + 1;
            ec_dec_update(&mut ec, s, s + 1, ft);
        } else {
            ec_dec_update(&mut ec, 0, nvals, ft);
        }
    }

    // Decode initial state
    let state_qoffset = q0 as usize * DRED_STATE_DIM;
    dred_decode_latents(
        &mut ec,
        &mut dred.state,
        &DRED_STATE_QUANT_SCALES_Q8[state_qoffset..],
        &DRED_STATE_R_Q8[state_qoffset..],
        &DRED_STATE_P0_Q8[state_qoffset..],
        DRED_STATE_DIM,
    );

    // Decode latent frames (newest to oldest, stored oldest to newest)
    let limit = DRED_NUM_REDUNDANCY_FRAMES.min(min_feature_frames.div_ceil(2));
    let mut i = 0;
    while i < limit {
        if 8 * num_bytes as i32 - ec_tell(&ec) <= 7 {
            break;
        }
        let q_level = compute_quantizer(q0, dq, qmax, (i / 2) as i32);
        let offset = q_level as usize * DRED_LATENT_DIM;
        let latent_offset = (i / 2) * (DRED_LATENT_DIM + 1);
        dred_decode_latents(
            &mut ec,
            &mut dred.latents[latent_offset..],
            &DRED_LATENT_QUANT_SCALES_Q8[offset..],
            &DRED_LATENT_R_Q8[offset..],
            &DRED_LATENT_P0_Q8[offset..],
            DRED_LATENT_DIM,
        );
        dred.latents[latent_offset + DRED_LATENT_DIM] = q_level as f32 * 0.125 - 1.0;
        i += 2;
    }
    dred.process_stage = 1;
    dred.nb_latents = (i / 2) as i32;
    (i / 2) as i32
}

/// Process parsed DRED data: run RDOVAE decoder to reconstruct features.
///
/// Upstream C: opus_decoder.c (inline in opus_dred_process)
pub fn opus_dred_process(dred_dec: &OpusDREDDecoder, dred: &mut OpusDRED) {
    debug_assert!(dred_dec.loaded, "libopus: assert(dec->loaded) called");
    if dred.process_stage != 1 || dred.nb_latents <= 0 {
        return;
    }
    dred_rdovae_decode_all(
        &dred_dec.model,
        &mut dred.fec_features,
        &dred.state,
        &dred.latents,
        dred.nb_latents as usize,
        opus_select_arch(),
    );
    dred.process_stage = 2;
}
