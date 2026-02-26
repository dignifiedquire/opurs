//! DNN integration tests: verify models load and encoder/decoder use them.
//!
//! Requires `dnn` + `builtin-weights` features.

#![cfg(all(feature = "dnn", feature = "builtin-weights"))]

use opurs::dnn::nnet::{write_weights, WeightArray};
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
fn pitchdnn_blob_loader_matches_array_init() {
    let arrays = get_weights();
    let blob = write_weights(&arrays);

    let mut from_arrays = opurs::dnn::pitchdnn::PitchDNNState::new();
    assert!(from_arrays.init(&arrays));

    let mut from_blob = opurs::dnn::pitchdnn::PitchDNNState::new();
    assert!(from_blob.load_model(&blob));

    let if_features = vec![0.01f32; 88];
    let xcorr_features = vec![0.02f32; opurs::dnn::pitchdnn::NB_XCORR_FEATURES];
    let out_arrays = opurs::dnn::pitchdnn::compute_pitchdnn(
        &mut from_arrays,
        &if_features,
        &xcorr_features,
        opurs::arch::Arch::default(),
    );
    let out_blob = opurs::dnn::pitchdnn::compute_pitchdnn(
        &mut from_blob,
        &if_features,
        &xcorr_features,
        opurs::arch::Arch::default(),
    );
    assert!((out_arrays - out_blob).abs() < 1e-6);
}

#[test]
fn pitchdnn_blob_loader_rejects_invalid_blob() {
    let mut st = opurs::dnn::pitchdnn::PitchDNNState::new();
    assert!(!st.load_model(&[0x00, 0x01, 0x02, 0x03]));
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

#[test]
fn lpcnet_plc_init_rejects_partial_weights() {
    let mut arrays = get_weights();
    arrays.retain(|w| !w.name.starts_with("dense_if_upsampler_1_"));

    let mut plc = opurs::dnn::lpcnet::LPCNetPLCState::new();
    assert!(
        !plc.init(&arrays),
        "LPCNetPLCState init should fail when encoder weights are missing"
    );
    assert!(
        !plc.loaded,
        "LPCNetPLCState.loaded should remain false on partial init"
    );
}

#[test]
fn lpcnet_blob_loaders_match_array_init() {
    let arrays = get_weights();
    let blob = write_weights(&arrays);

    let mut enc_from_arrays = opurs::dnn::lpcnet::LPCNetEncState::new();
    assert!(enc_from_arrays.load_model(&arrays));

    let mut enc_from_blob = opurs::dnn::lpcnet::LPCNetEncState::new();
    assert!(opurs::dnn::lpcnet::lpcnet_encoder_load_model(
        &mut enc_from_blob,
        &blob
    ));

    let mut plc_from_arrays = opurs::dnn::lpcnet::LPCNetPLCState::new();
    assert!(plc_from_arrays.load_model(&arrays));

    let mut plc_from_blob = opurs::dnn::lpcnet::LPCNetPLCState::new();
    assert!(opurs::dnn::lpcnet::lpcnet_plc_load_model(
        &mut plc_from_blob,
        &blob
    ));

    assert!(plc_from_arrays.loaded);
    assert!(plc_from_blob.loaded);
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
            opurs::dnn::osce::osce_load_models_from_arrays(&mut model, &arrays),
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
            !opurs::dnn::osce::osce_load_models_from_arrays(&mut model, &arrays),
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

    #[test]
    fn osce_model_blob_loader_matches_array_init() {
        let arrays = get_weights();
        let blob = write_weights(&arrays);

        let mut from_arrays = opurs::dnn::osce::OSCEModel::default();
        assert!(opurs::dnn::osce::osce_load_models_from_arrays(
            &mut from_arrays,
            &arrays
        ));

        let mut from_blob = opurs::dnn::osce::OSCEModel::default();
        assert!(opurs::dnn::osce::osce_load_models(&mut from_blob, &blob));

        assert_eq!(from_arrays.loaded, from_blob.loaded);
        assert_eq!(from_arrays.lace.is_some(), from_blob.lace.is_some());
        assert_eq!(from_arrays.nolace.is_some(), from_blob.nolace.is_some());
        assert_eq!(from_arrays.bbwenet.is_some(), from_blob.bbwenet.is_some());
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
