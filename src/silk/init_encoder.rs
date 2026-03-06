//! Encoder initialization.
//!
//! Upstream c: `silk/init_encoder.c`

use crate::arch::Arch;
use crate::silk::float::structs_flp::silk_encoder_state_FLP;
use crate::silk::lin2log::silk_lin2log;
use crate::silk::tuning_parameters::VARIABLE_HP_MIN_CUTOFF_HZ;
use crate::silk::vad::silk_vad_init;

/// Upstream c: silk/init_encoder.c:silk_init_encoder
pub fn silk_init_encoder(ps_enc: &mut silk_encoder_state_FLP, arch: Arch) -> i32 {
    let mut ret: i32 = 0;
    *ps_enc = Default::default();
    ps_enc.s_cmn.arch = arch;
    ps_enc.s_cmn.variable_hp_smth1_q15 =
        (((silk_lin2log(((VARIABLE_HP_MIN_CUTOFF_HZ * ((1) << 16)) as f64 + 0.5f64) as i32)
            - ((16) << 7)) as u32)
            << 8) as i32;
    ps_enc.s_cmn.variable_hp_smth2_q15 = ps_enc.s_cmn.variable_hp_smth1_q15;
    ps_enc.s_cmn.first_frame_after_reset = 1;
    ret += silk_vad_init(&mut ps_enc.s_cmn.s_vad);
    ret
}
