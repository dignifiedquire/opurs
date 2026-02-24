//! DNN integration tests: verify models load and encoder/decoder use them.
//!
//! Requires `dnn` + `builtin-weights` features.

#![cfg(all(feature = "dnn", feature = "builtin-weights"))]

use opurs::dnn::nnet::WeightArray;
use opurs::dnn::weights::compiled_weights;

fn get_weights() -> Vec<WeightArray> {
    compiled_weights()
}

// ---- Model loading tests ----

#[test]
fn pitchdnn_model_loads() {
    let arrays = get_weights();
    let model = opurs::dnn::pitchdnn::init_pitchdnn(&arrays);
    assert!(model.is_some(), "PitchDNN model failed to load");
}

#[test]
fn fargan_model_loads() {
    let arrays = get_weights();
    let mut state = opurs::dnn::fargan::FARGANState::new();
    assert!(state.init(&arrays), "FARGAN model failed to load");
}

#[test]
fn plcmodel_loads() {
    let arrays = get_weights();
    let model = opurs::dnn::lpcnet::init_plcmodel(&arrays);
    assert!(model.is_some(), "PLCModel failed to load");
}

#[test]
fn lpcnet_plc_full_load() {
    let arrays = get_weights();
    let mut plc = opurs::dnn::lpcnet::LPCNetPLCState::new();
    assert!(plc.load_model(&arrays), "LPCNetPLCState full load failed");
    assert!(plc.loaded, "LPCNetPLCState.loaded should be true");
}

#[cfg(feature = "dred")]
mod dred_tests {
    use super::*;

    #[test]
    fn rdovaeenc_model_loads() {
        let arrays = get_weights();
        let model = opurs::dnn::dred::rdovae_enc::init_rdovaeenc(&arrays);
        assert!(model.is_some(), "RDOVAEEnc model failed to load");
    }

    #[test]
    fn rdovaedec_model_loads() {
        let arrays = get_weights();
        let model = opurs::dnn::dred::rdovae_dec::init_rdovaedec(&arrays);
        assert!(model.is_some(), "RDOVAEDec model failed to load");
    }

    #[test]
    fn dred_encoder_full_load() {
        let arrays = get_weights();
        let mut enc = opurs::dnn::dred::encoder::DREDEnc::new();
        assert!(enc.load_model(&arrays), "DREDEnc full load failed");
        assert!(enc.loaded, "DREDEnc.loaded should be true");
    }

    #[test]
    fn dred_decoder_full_load() {
        let arrays = get_weights();
        let mut dec = opurs::dnn::dred::decoder::OpusDREDDecoder::new();
        assert!(dec.load_model(&arrays), "OpusDREDDecoder full load failed");
        assert!(dec.loaded, "OpusDREDDecoder.loaded should be true");
    }
}

#[cfg(feature = "osce")]
mod osce_tests {
    use super::*;

    #[test]
    fn lace_model_loads() {
        let arrays = get_weights();
        let model = opurs::dnn::osce::init_lace(&arrays);
        assert!(model.is_some(), "LACE model failed to load");
    }

    #[test]
    fn nolace_model_loads() {
        let arrays = get_weights();
        let model = opurs::dnn::osce::init_nolace(&arrays);
        assert!(model.is_some(), "NoLACE model failed to load");
    }

    #[test]
    fn osce_model_full_load() {
        let arrays = get_weights();
        let mut model = opurs::dnn::osce::OSCEModel::default();
        assert!(
            opurs::dnn::osce::osce_load_models(&mut model, &arrays),
            "OSCEModel full load failed"
        );
        assert!(model.loaded, "OSCEModel.loaded should be true");
    }

    #[test]
    fn osce_model_load_rejects_partial_weights() {
        let mut arrays = get_weights();
        arrays.retain(|w| !w.name.starts_with("bbwenet_"));

        let mut model = opurs::dnn::osce::OSCEModel::default();
        assert!(
            !opurs::dnn::osce::osce_load_models(&mut model, &arrays),
            "OSCEModel load should fail when BBWENet weights are missing"
        );
        assert!(!model.loaded, "OSCEModel.loaded should be false");
        assert!(
            model.lace.is_some(),
            "LACE should still parse from provided weights"
        );
        assert!(
            model.nolace.is_some(),
            "NoLACE should still parse from provided weights"
        );
        assert!(
            model.bbwenet.is_none(),
            "BBWENet should be absent when its weights are removed"
        );
    }
}

// ---- Encoder/Decoder integration ----

#[cfg(feature = "dred")]
#[test]
fn encoder_load_dnn_weights() {
    let mut enc = opurs::OpusEncoder::new(48000, 1, opurs::OPUS_APPLICATION_AUDIO).unwrap();
    assert!(
        enc.load_dnn_weights().is_ok(),
        "Encoder DNN weight load failed"
    );
}

#[test]
fn decoder_load_dnn_weights() {
    let mut dec = opurs::OpusDecoder::new(48000, 1).unwrap();
    assert!(
        dec.load_dnn_weights().is_ok(),
        "Decoder DNN weight load failed"
    );
}
