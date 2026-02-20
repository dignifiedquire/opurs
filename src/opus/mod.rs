//! Opus integration layer — combines CELT and SILK, handles mode switching and packet framing.
//!
//! Upstream C: `src/`

#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(non_upper_case_globals)]
#![allow(unused_assignments)]
#![allow(clippy::too_many_arguments)]

pub mod analysis;
pub mod extensions;
pub mod mlp;
pub mod opus_decoder;
pub mod opus_encoder;
pub mod opus_multistream;
pub mod opus_multistream_decoder;
pub mod opus_multistream_encoder;
pub mod packet;
pub mod repacketizer;
// stuff for structs that do not have a clear home, named after the header files
pub mod opus_defines;
pub mod opus_private;
