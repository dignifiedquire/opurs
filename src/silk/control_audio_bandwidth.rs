//! Audio bandwidth control.
//!
//! Upstream c: `silk/control_audio_bandwidth.c`

use crate::silk::define::TRANSITION_FRAMES;
use crate::silk::enc_api::silk_EncControlStruct;
use crate::silk::structs::silk_encoder_state;
/// Upstream c: silk/control_audio_bandwidth.c:silk_control_audio_bandwidth
pub fn silk_control_audio_bandwidth(
    ps_enc_c: &mut silk_encoder_state,
    enc_control: &mut silk_EncControlStruct,
) -> i32 {
    let mut fs_k_hz: i32;
    let mut orig_k_hz: i32;
    let mut fs_hz: i32;
    orig_k_hz = ps_enc_c.fs_k_hz;
    if orig_k_hz == 0 {
        orig_k_hz = ps_enc_c.s_lp.saved_fs_k_hz;
    }
    fs_k_hz = orig_k_hz;
    fs_hz = fs_k_hz as i16 as i32 * 1000;
    if fs_hz == 0 {
        fs_hz = if ps_enc_c.desired_internal_fs_hz < ps_enc_c.api_fs_hz {
            ps_enc_c.desired_internal_fs_hz
        } else {
            ps_enc_c.api_fs_hz
        };
        fs_k_hz = fs_hz / 1000;
    } else if fs_hz > ps_enc_c.api_fs_hz
        || fs_hz > ps_enc_c.max_internal_fs_hz
        || fs_hz < ps_enc_c.min_internal_fs_hz
    {
        fs_hz = ps_enc_c.api_fs_hz;
        fs_hz = if fs_hz < ps_enc_c.max_internal_fs_hz {
            fs_hz
        } else {
            ps_enc_c.max_internal_fs_hz
        };
        fs_hz = if fs_hz > ps_enc_c.min_internal_fs_hz {
            fs_hz
        } else {
            ps_enc_c.min_internal_fs_hz
        };
        fs_k_hz = fs_hz / 1000;
    } else {
        if ps_enc_c.s_lp.transition_frame_no >= TRANSITION_FRAMES as i32 {
            ps_enc_c.s_lp.mode = 0;
        }
        if ps_enc_c.allow_bandwidth_switch != 0 || enc_control.opus_can_switch != 0 {
            if orig_k_hz as i16 as i32 * 1000 > ps_enc_c.desired_internal_fs_hz {
                if ps_enc_c.s_lp.mode == 0 {
                    ps_enc_c.s_lp.transition_frame_no = TRANSITION_FRAMES as i32;
                    ps_enc_c.s_lp.in_lp_state.fill(0);
                }
                if enc_control.opus_can_switch != 0 {
                    ps_enc_c.s_lp.mode = 0;
                    fs_k_hz = if orig_k_hz == 16 { 12 } else { 8 };
                } else if ps_enc_c.s_lp.transition_frame_no <= 0 {
                    enc_control.switch_ready = 1;
                    enc_control.max_bits -=
                        enc_control.max_bits * 5 / (enc_control.payload_size_ms + 5);
                } else {
                    ps_enc_c.s_lp.mode = -(2);
                }
            } else if (orig_k_hz as i16 as i32 * 1000) < ps_enc_c.desired_internal_fs_hz {
                if enc_control.opus_can_switch != 0 {
                    fs_k_hz = if orig_k_hz == 8 { 12 } else { 16 };
                    ps_enc_c.s_lp.transition_frame_no = 0;
                    ps_enc_c.s_lp.in_lp_state.fill(0);
                    ps_enc_c.s_lp.mode = 1;
                } else if ps_enc_c.s_lp.mode == 0 {
                    enc_control.switch_ready = 1;
                    enc_control.max_bits -=
                        enc_control.max_bits * 5 / (enc_control.payload_size_ms + 5);
                } else {
                    ps_enc_c.s_lp.mode = 1;
                }
            } else if ps_enc_c.s_lp.mode < 0 {
                ps_enc_c.s_lp.mode = 1;
            }
        }
    }
    fs_k_hz
}
