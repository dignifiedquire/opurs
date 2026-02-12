//! Weight verification tests: compiled-in Rust weights must match the C reference 1:1.
//!
//! These tests require the `tools-dnn` feature (which enables both the Rust DNN
//! modules and the C library with DNN weight data).

#![cfg(feature = "tools-dnn")]

use opurs::dnn::nnet::{parse_weights, write_weights};
use opurs::dnn::weights::compiled_weights;

/// Helper: get the reference weight blob from the C library.
fn c_reference_blob() -> Vec<u8> {
    unsafe {
        let size = libopus_sys::opus_dnn_weights_blob_size() as usize;
        assert!(size > 0, "C blob size is 0 — DNN features not compiled in?");
        let mut buf = vec![0u8; size];
        let written = libopus_sys::opus_dnn_write_weights_blob(buf.as_mut_ptr()) as usize;
        assert_eq!(size, written);
        buf
    }
}

#[test]
fn compiled_weights_match_c_reference_blob() {
    let c_blob = c_reference_blob();
    let rust_arrays = compiled_weights();

    // Serialize the Rust compiled-in weights to a blob.
    let rust_blob = write_weights(&rust_arrays);

    // They must be byte-identical.
    assert_eq!(
        c_blob.len(),
        rust_blob.len(),
        "Blob size mismatch: C={} Rust={}",
        c_blob.len(),
        rust_blob.len(),
    );
    assert_eq!(
        c_blob, rust_blob,
        "Rust weight blob differs from C reference"
    );
}

#[test]
fn compiled_weights_roundtrip_through_blob() {
    let rust_arrays = compiled_weights();
    let blob = write_weights(&rust_arrays);
    let parsed = parse_weights(&blob).expect("Failed to parse roundtripped blob");

    assert_eq!(
        rust_arrays.len(),
        parsed.len(),
        "Array count mismatch after roundtrip"
    );
    for (i, (orig, rt)) in rust_arrays.iter().zip(parsed.iter()).enumerate() {
        assert_eq!(orig.name, rt.name, "Name mismatch at index {i}");
        assert_eq!(
            orig.type_id, rt.type_id,
            "Type mismatch at index {i}: {}",
            orig.name
        );
        assert_eq!(
            orig.size, rt.size,
            "Size mismatch at index {i}: {}",
            orig.name
        );
        assert_eq!(
            &orig.data[..orig.size],
            &rt.data[..rt.size],
            "Data mismatch at index {i}: {}",
            orig.name,
        );
    }
}

#[test]
fn compiled_weights_array_count() {
    let arrays = compiled_weights();
    // With all DNN features enabled we expect:
    // pitchdnn(28) + fargan(66) + plcmodel(20) + rdovaeenc(79) + rdovaedec(97)
    // + lace(42..49) + nolace(104..122)
    // The exact count depends on which #ifdef WEIGHTS_*_DEFINED macros are set.
    // Just verify we have a reasonable number.
    assert!(
        arrays.len() > 400,
        "Expected >400 weight arrays with all DNN features, got {}",
        arrays.len(),
    );
}

/// Per-model blob verification: each model's Rust data matches its C counterpart.
mod per_model {
    use opurs::dnn::nnet::write_weights;

    fn verify_model(
        rust_arrays_fn: fn() -> Vec<opurs::dnn::nnet::WeightArray>,
        c_size_fn: unsafe extern "C" fn() -> i32,
        c_write_fn: unsafe extern "C" fn(*mut u8) -> i32,
        model_name: &str,
    ) {
        let rust_arrays = rust_arrays_fn();
        let rust_blob = write_weights(&rust_arrays);

        let c_blob = unsafe {
            let size = c_size_fn() as usize;
            let mut buf = vec![0u8; size];
            let written = c_write_fn(buf.as_mut_ptr()) as usize;
            assert_eq!(size, written, "{model_name}: C blob size mismatch");
            buf
        };

        assert_eq!(
            c_blob.len(),
            rust_blob.len(),
            "{model_name}: blob size mismatch C={} Rust={}",
            c_blob.len(),
            rust_blob.len(),
        );
        assert_eq!(c_blob, rust_blob, "{model_name}: blob content mismatch");
    }

    #[test]
    fn pitchdnn_matches_c() {
        verify_model(
            opurs::dnn::pitchdnn_data::pitchdnn_arrays,
            libopus_sys::opus_dnn_pitchdnn_blob_size,
            libopus_sys::opus_dnn_pitchdnn_write,
            "PitchDNN",
        );
    }

    #[test]
    fn fargan_matches_c() {
        verify_model(
            opurs::dnn::fargan_data::fargan_arrays,
            libopus_sys::opus_dnn_fargan_blob_size,
            libopus_sys::opus_dnn_fargan_write,
            "FARGAN",
        );
    }

    #[test]
    fn plcmodel_matches_c() {
        verify_model(
            opurs::dnn::plc_data::plcmodel_arrays,
            libopus_sys::opus_dnn_plcmodel_blob_size,
            libopus_sys::opus_dnn_plcmodel_write,
            "PLCModel",
        );
    }

    #[test]
    fn rdovaeenc_matches_c() {
        verify_model(
            opurs::dnn::dred::rdovae_enc_data::rdovaeenc_arrays,
            libopus_sys::opus_dnn_rdovaeenc_blob_size,
            libopus_sys::opus_dnn_rdovaeenc_write,
            "RDOVAEEnc",
        );
    }

    #[test]
    fn rdovaedec_matches_c() {
        verify_model(
            opurs::dnn::dred::rdovae_dec_data::rdovaedec_arrays,
            libopus_sys::opus_dnn_rdovaedec_blob_size,
            libopus_sys::opus_dnn_rdovaedec_write,
            "RDOVAEDec",
        );
    }

    #[test]
    fn lace_matches_c() {
        verify_model(
            opurs::dnn::osce_lace_data::lacelayers_arrays,
            libopus_sys::opus_dnn_lace_blob_size,
            libopus_sys::opus_dnn_lace_write,
            "LACE",
        );
    }

    #[test]
    fn nolace_matches_c() {
        verify_model(
            opurs::dnn::osce_nolace_data::nolacelayers_arrays,
            libopus_sys::opus_dnn_nolace_blob_size,
            libopus_sys::opus_dnn_nolace_write,
            "NoLACE",
        );
    }
}
