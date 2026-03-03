//! Opus integration layer — combines CELT and SILK, handles mode switching and packet framing.
//!
//! Upstream C: `src/`

pub mod analysis;
pub mod extensions;
pub mod mapping_matrix;
pub mod mlp;
pub mod opus_decoder;
pub mod opus_encoder;
pub mod opus_multistream;
pub mod opus_multistream_decoder;
pub mod opus_multistream_encoder;
pub mod opus_projection_decoder;
pub mod opus_projection_encoder;
pub mod packet;
pub mod projection_matrices;
pub mod repacketizer;
// stuff for structs that do not have a clear home, named after the header files
pub mod opus_defines;
pub mod opus_private;
