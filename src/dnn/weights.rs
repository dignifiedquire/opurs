//! Weight loading helpers for DNN models.
//!
//! Provides compiled-in weight data (from the `*_data.rs` modules) and
//! runtime loading via `parse_weights()`.

use super::nnet::{parse_weights, WeightArray};

/// Return all compiled-in weight arrays.
///
/// This collects weights from all enabled DNN feature modules.
/// The returned arrays can be passed to `load_model()` on any DNN model struct.
///
/// Requires the `builtin-weights` feature.
#[cfg(feature = "builtin-weights")]
pub fn compiled_weights() -> Vec<WeightArray> {
    let mut arrays = Vec::new();
    arrays.extend(super::pitchdnn_data::pitchdnn_arrays());
    arrays.extend(super::fargan_data::fargan_arrays());
    arrays.extend(super::plc_data::plcmodel_arrays());
    #[cfg(feature = "dred")]
    {
        arrays.extend(super::dred::rdovae_enc_data::rdovaeenc_arrays());
        arrays.extend(super::dred::rdovae_dec_data::rdovaedec_arrays());
    }
    #[cfg(feature = "osce")]
    {
        arrays.extend(super::osce_lace_data::lacelayers_arrays());
        arrays.extend(super::osce_nolace_data::nolacelayers_arrays());
        arrays.extend(super::bbwenet_data::bbwenetlayers_arrays());
    }
    arrays
}

/// Parse a binary weight blob into weight arrays.
///
/// The blob must be in the upstream "DNNw" format (as produced by
/// `write_lpcnet_weights` or `write_weights()`).
pub fn load_weights(data: &[u8]) -> Option<Vec<WeightArray>> {
    parse_weights(data)
}
