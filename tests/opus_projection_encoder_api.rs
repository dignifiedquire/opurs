#![cfg(feature = "tools")]

use opurs::{
    opus_projection_ambisonics_encoder_create as rust_opus_projection_encoder_create,
    opus_projection_ambisonics_encoder_init as rust_opus_projection_encoder_init,
    opus_projection_encode as rust_opus_projection_encode,
    opus_projection_encode24 as rust_opus_projection_encode24,
    opus_projection_encode_float as rust_opus_projection_encode_float,
    opus_projection_encoder_get_encoder_state as rust_opus_projection_encoder_get_encoder_state,
    Channels, Signal, OPUS_APPLICATION_AUDIO, OPUS_APPLICATION_VOIP, OPUS_AUTO, OPUS_BAD_ARG,
    OPUS_GET_APPLICATION_REQUEST, OPUS_GET_COMPLEXITY_REQUEST, OPUS_GET_DTX_REQUEST,
    OPUS_GET_FORCE_CHANNELS_REQUEST, OPUS_GET_INBAND_FEC_REQUEST,
    OPUS_GET_PACKET_LOSS_PERC_REQUEST, OPUS_GET_PHASE_INVERSION_DISABLED_REQUEST,
    OPUS_GET_PREDICTION_DISABLED_REQUEST, OPUS_GET_SIGNAL_REQUEST, OPUS_GET_VBR_CONSTRAINT_REQUEST,
    OPUS_GET_VBR_REQUEST, OPUS_MULTISTREAM_GET_ENCODER_STATE_REQUEST, OPUS_OK,
    OPUS_PROJECTION_GET_DEMIXING_MATRIX_REQUEST, OPUS_PROJECTION_GET_DEMIXING_MATRIX_SIZE_REQUEST,
    OPUS_SET_APPLICATION_REQUEST, OPUS_SET_COMPLEXITY_REQUEST, OPUS_SET_DTX_REQUEST,
    OPUS_SET_FORCE_CHANNELS_REQUEST, OPUS_SET_INBAND_FEC_REQUEST,
    OPUS_SET_PACKET_LOSS_PERC_REQUEST, OPUS_SET_PHASE_INVERSION_DISABLED_REQUEST,
    OPUS_SET_PREDICTION_DISABLED_REQUEST, OPUS_SET_SIGNAL_REQUEST, OPUS_SET_VBR_CONSTRAINT_REQUEST,
    OPUS_SET_VBR_REQUEST,
};
use std::alloc::{alloc_zeroed, dealloc, Layout};
use std::ffi::c_void;
use std::sync::{Mutex, MutexGuard, OnceLock};

unsafe extern "C" {
    fn opus_projection_ambisonics_encoder_get_size(channels: i32, mapping_family: i32) -> i32;
    fn opus_projection_ambisonics_encoder_create(
        Fs: i32,
        channels: i32,
        mapping_family: i32,
        streams: *mut i32,
        coupled_streams: *mut i32,
        application: i32,
        error: *mut i32,
    ) -> *mut c_void;
    fn opus_projection_ambisonics_encoder_init(
        st: *mut c_void,
        Fs: i32,
        channels: i32,
        mapping_family: i32,
        streams: *mut i32,
        coupled_streams: *mut i32,
        application: i32,
    ) -> i32;
    fn opus_projection_encoder_destroy(st: *mut c_void);
    fn opus_projection_encoder_ctl(st: *mut c_void, request: i32, ...) -> i32;
    fn opus_projection_encode(
        st: *mut c_void,
        pcm: *const i16,
        frame_size: i32,
        data: *mut u8,
        max_data_bytes: i32,
    ) -> i32;
    fn opus_projection_encode_float(
        st: *mut c_void,
        pcm: *const f32,
        frame_size: i32,
        data: *mut u8,
        max_data_bytes: i32,
    ) -> i32;
    fn opus_projection_encode24(
        st: *mut c_void,
        pcm: *const i32,
        frame_size: i32,
        data: *mut u8,
        max_data_bytes: i32,
    ) -> i32;
}

fn test_guard() -> MutexGuard<'static, ()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

#[test]
fn projection_encoder_get_size_zero_nonzero_parity() {
    let _guard = test_guard();
    let cases = [(4, 3), (9, 3), (36, 3), (49, 3), (4, 1)];
    for (channels, mapping_family) in cases {
        let rust = opurs::opus_projection_ambisonics_encoder_get_size(channels, mapping_family);
        let c = unsafe { opus_projection_ambisonics_encoder_get_size(channels, mapping_family) };
        assert_eq!(
            rust == 0,
            c == 0,
            "projection encoder get_size validity mismatch (channels={channels}, mapping_family={mapping_family})"
        );
    }
}

#[test]
fn projection_encoder_create_and_matrix_parity_with_c() {
    let _guard = test_guard();

    for channels in [4, 9, 16, 25, 36] {
        let mut rust_streams = -1i32;
        let mut rust_coupled = -1i32;
        let rust_enc = rust_opus_projection_encoder_create(
            48000,
            channels,
            3,
            &mut rust_streams,
            &mut rust_coupled,
            OPUS_APPLICATION_AUDIO,
        )
        .expect("rust projection encoder create");

        let mut c_streams = -1i32;
        let mut c_coupled = -1i32;
        let mut c_error = 0i32;
        let c_enc = unsafe {
            opus_projection_ambisonics_encoder_create(
                48000,
                channels,
                3,
                &mut c_streams,
                &mut c_coupled,
                OPUS_APPLICATION_AUDIO,
                &mut c_error,
            )
        };
        assert!(
            !c_enc.is_null(),
            "c projection encoder create failed (channels={channels}): {c_error}"
        );

        assert_eq!(rust_streams, c_streams);
        assert_eq!(rust_coupled, c_coupled);

        let rust_matrix_size = rust_enc.demixing_matrix_size();
        let mut c_matrix_size = 0i32;
        let ret = unsafe {
            opus_projection_encoder_ctl(
                c_enc,
                OPUS_PROJECTION_GET_DEMIXING_MATRIX_SIZE_REQUEST,
                &mut c_matrix_size,
            )
        };
        assert_eq!(ret, OPUS_OK);
        assert_eq!(rust_matrix_size, c_matrix_size);

        let mut rust_matrix = vec![0u8; rust_matrix_size as usize];
        let mut c_matrix = vec![0u8; c_matrix_size as usize];
        rust_enc.copy_demixing_matrix(&mut rust_matrix).unwrap();
        let ret = unsafe {
            opus_projection_encoder_ctl(
                c_enc,
                OPUS_PROJECTION_GET_DEMIXING_MATRIX_REQUEST,
                c_matrix.as_mut_ptr(),
                c_matrix_size,
            )
        };
        assert_eq!(ret, OPUS_OK);
        assert_eq!(
            rust_matrix, c_matrix,
            "demixing matrix bytes mismatch (channels={channels})"
        );

        unsafe { opus_projection_encoder_destroy(c_enc) };
    }
}

#[test]
fn projection_encoder_foa_encode_smoke_against_c() {
    let _guard = test_guard();

    let mut rust_streams = -1i32;
    let mut rust_coupled = -1i32;
    let mut rust_enc = rust_opus_projection_encoder_create(
        48000,
        4,
        3,
        &mut rust_streams,
        &mut rust_coupled,
        OPUS_APPLICATION_AUDIO,
    )
    .expect("rust projection encoder create");

    let mut c_streams = -1i32;
    let mut c_coupled = -1i32;
    let mut c_error = 0i32;
    let c_enc = unsafe {
        opus_projection_ambisonics_encoder_create(
            48000,
            4,
            3,
            &mut c_streams,
            &mut c_coupled,
            OPUS_APPLICATION_AUDIO,
            &mut c_error,
        )
    };
    assert!(
        !c_enc.is_null(),
        "c projection encoder create failed: {c_error}"
    );

    let frame_size = 960usize;
    let mut pcm = vec![0i16; frame_size * 4];
    for i in 0..frame_size {
        for ch in 0..4usize {
            pcm[i * 4 + ch] = (i as i16).wrapping_mul((ch as i16 + 2) * 11);
        }
    }

    let mut rust_packet = vec![0u8; 4000];
    let rust_len =
        rust_opus_projection_encode(&mut rust_enc, &pcm, frame_size as i32, &mut rust_packet);
    assert!(rust_len > 0, "rust projection encode failed: {rust_len}");

    let mut c_packet = vec![0u8; 4000];
    let c_len = unsafe {
        opus_projection_encode(
            c_enc,
            pcm.as_ptr(),
            frame_size as i32,
            c_packet.as_mut_ptr(),
            c_packet.len() as i32,
        )
    };
    assert!(c_len > 0, "c projection encode failed: {c_len}");

    unsafe { opus_projection_encoder_destroy(c_enc) };
}

#[test]
fn projection_encoder_init_reinit_parity_with_c() {
    let _guard = test_guard();

    let mut rust_streams = -1i32;
    let mut rust_coupled = -1i32;
    let mut rust_enc = rust_opus_projection_encoder_create(
        48000,
        4,
        3,
        &mut rust_streams,
        &mut rust_coupled,
        OPUS_APPLICATION_AUDIO,
    )
    .expect("rust projection encoder create");

    let c_max_size = unsafe { opus_projection_ambisonics_encoder_get_size(38, 3) };
    assert!(
        c_max_size > 0,
        "unexpected zero max projection encoder size"
    );
    let c_layout =
        Layout::from_size_align(c_max_size as usize, 64).expect("valid projection encoder layout");
    let c_storage = unsafe { alloc_zeroed(c_layout) };
    assert!(
        !c_storage.is_null(),
        "c projection encoder allocation failed"
    );
    let c_ptr = c_storage.cast();

    let mut c_streams = -1i32;
    let mut c_coupled = -1i32;
    let c_init0 = unsafe {
        opus_projection_ambisonics_encoder_init(
            c_ptr,
            48000,
            4,
            3,
            &mut c_streams,
            &mut c_coupled,
            OPUS_APPLICATION_AUDIO,
        )
    };
    assert_eq!(c_init0, OPUS_OK);

    let mut rust_streams_re = -1i32;
    let mut rust_coupled_re = -1i32;
    let rust_ret = rust_opus_projection_encoder_init(
        &mut rust_enc,
        48000,
        9,
        3,
        &mut rust_streams_re,
        &mut rust_coupled_re,
        OPUS_APPLICATION_AUDIO,
    );

    let mut c_streams_re = -1i32;
    let mut c_coupled_re = -1i32;
    let c_ret = unsafe {
        opus_projection_ambisonics_encoder_init(
            c_ptr,
            48000,
            9,
            3,
            &mut c_streams_re,
            &mut c_coupled_re,
            OPUS_APPLICATION_AUDIO,
        )
    };

    assert_eq!(rust_ret, c_ret);
    if rust_ret == OPUS_OK {
        assert_eq!(rust_streams_re, c_streams_re);
        assert_eq!(rust_coupled_re, c_coupled_re);

        let rust_matrix_size = rust_enc.demixing_matrix_size();
        let mut c_matrix_size = 0i32;
        let ret = unsafe {
            opus_projection_encoder_ctl(
                c_ptr,
                OPUS_PROJECTION_GET_DEMIXING_MATRIX_SIZE_REQUEST,
                &mut c_matrix_size,
            )
        };
        assert_eq!(ret, OPUS_OK);
        assert_eq!(rust_matrix_size, c_matrix_size);

        let mut rust_matrix = vec![0u8; rust_matrix_size as usize];
        let mut c_matrix = vec![0u8; c_matrix_size as usize];
        rust_enc.copy_demixing_matrix(&mut rust_matrix).unwrap();
        let ret = unsafe {
            opus_projection_encoder_ctl(
                c_ptr,
                OPUS_PROJECTION_GET_DEMIXING_MATRIX_REQUEST,
                c_matrix.as_mut_ptr(),
                c_matrix_size,
            )
        };
        assert_eq!(ret, OPUS_OK);
        assert_eq!(rust_matrix, c_matrix);
    }

    unsafe { dealloc(c_storage, c_layout) };
}

#[test]
fn projection_encoder_format_encode_smoke_against_c() {
    let _guard = test_guard();

    for channels in [4i32, 9i32] {
        let frame_size = 960usize;

        let mut pcm_i16 = vec![0i16; frame_size * channels as usize];
        let mut pcm_f32 = vec![0f32; frame_size * channels as usize];
        let mut pcm_i24 = vec![0i32; frame_size * channels as usize];
        for i in 0..frame_size {
            for ch in 0..channels as usize {
                let base = (i as i32 * (ch as i32 + 3) * 17) - 4096;
                pcm_i16[i * channels as usize + ch] = base as i16;
                pcm_f32[i * channels as usize + ch] = ((base as f32) / 32768.0)
                    * 0.75
                    * (1.0 + (i as f32 / frame_size as f32 * core::f32::consts::TAU).sin() * 0.2);
                pcm_i24[i * channels as usize + ch] = base << 8;
            }
        }

        {
            let mut rust_streams = -1i32;
            let mut rust_coupled = -1i32;
            let mut rust_enc = rust_opus_projection_encoder_create(
                48000,
                channels,
                3,
                &mut rust_streams,
                &mut rust_coupled,
                OPUS_APPLICATION_AUDIO,
            )
            .expect("rust create");

            let mut c_streams = -1i32;
            let mut c_coupled = -1i32;
            let mut c_error = 0i32;
            let c_enc = unsafe {
                opus_projection_ambisonics_encoder_create(
                    48000,
                    channels,
                    3,
                    &mut c_streams,
                    &mut c_coupled,
                    OPUS_APPLICATION_AUDIO,
                    &mut c_error,
                )
            };
            assert!(!c_enc.is_null(), "c create failed: {c_error}");
            assert_eq!(rust_streams, c_streams);
            assert_eq!(rust_coupled, c_coupled);

            let mut rust_packet = vec![0u8; 4000];
            let rust_len = rust_opus_projection_encode(
                &mut rust_enc,
                &pcm_i16,
                frame_size as i32,
                &mut rust_packet,
            );
            let mut c_packet = vec![0u8; 4000];
            let c_len = unsafe {
                opus_projection_encode(
                    c_enc,
                    pcm_i16.as_ptr(),
                    frame_size as i32,
                    c_packet.as_mut_ptr(),
                    c_packet.len() as i32,
                )
            };
            assert!(rust_len > 0, "rust i16 encode failed (channels={channels})");
            assert!(c_len > 0, "c i16 encode failed (channels={channels})");

            unsafe { opus_projection_encoder_destroy(c_enc) };
        }

        {
            let mut rust_streams = -1i32;
            let mut rust_coupled = -1i32;
            let mut rust_enc = rust_opus_projection_encoder_create(
                48000,
                channels,
                3,
                &mut rust_streams,
                &mut rust_coupled,
                OPUS_APPLICATION_AUDIO,
            )
            .expect("rust create");

            let mut c_streams = -1i32;
            let mut c_coupled = -1i32;
            let mut c_error = 0i32;
            let c_enc = unsafe {
                opus_projection_ambisonics_encoder_create(
                    48000,
                    channels,
                    3,
                    &mut c_streams,
                    &mut c_coupled,
                    OPUS_APPLICATION_AUDIO,
                    &mut c_error,
                )
            };
            assert!(!c_enc.is_null(), "c create failed: {c_error}");
            assert_eq!(rust_streams, c_streams);
            assert_eq!(rust_coupled, c_coupled);

            let mut rust_packet = vec![0u8; 4000];
            let rust_len = rust_opus_projection_encode_float(
                &mut rust_enc,
                &pcm_f32,
                frame_size as i32,
                &mut rust_packet,
            );
            let mut c_packet = vec![0u8; 4000];
            let c_len = unsafe {
                opus_projection_encode_float(
                    c_enc,
                    pcm_f32.as_ptr(),
                    frame_size as i32,
                    c_packet.as_mut_ptr(),
                    c_packet.len() as i32,
                )
            };
            assert!(
                rust_len > 0,
                "rust float encode failed (channels={channels})"
            );
            assert!(c_len > 0, "c float encode failed (channels={channels})");

            unsafe { opus_projection_encoder_destroy(c_enc) };
        }

        {
            let mut rust_streams = -1i32;
            let mut rust_coupled = -1i32;
            let mut rust_enc = rust_opus_projection_encoder_create(
                48000,
                channels,
                3,
                &mut rust_streams,
                &mut rust_coupled,
                OPUS_APPLICATION_AUDIO,
            )
            .expect("rust create");

            let mut c_streams = -1i32;
            let mut c_coupled = -1i32;
            let mut c_error = 0i32;
            let c_enc = unsafe {
                opus_projection_ambisonics_encoder_create(
                    48000,
                    channels,
                    3,
                    &mut c_streams,
                    &mut c_coupled,
                    OPUS_APPLICATION_AUDIO,
                    &mut c_error,
                )
            };
            assert!(!c_enc.is_null(), "c create failed: {c_error}");
            assert_eq!(rust_streams, c_streams);
            assert_eq!(rust_coupled, c_coupled);

            let mut rust_packet = vec![0u8; 4000];
            let rust_len = rust_opus_projection_encode24(
                &mut rust_enc,
                &pcm_i24,
                frame_size as i32,
                &mut rust_packet,
            );
            let mut c_packet = vec![0u8; 4000];
            let c_len = unsafe {
                opus_projection_encode24(
                    c_enc,
                    pcm_i24.as_ptr(),
                    frame_size as i32,
                    c_packet.as_mut_ptr(),
                    c_packet.len() as i32,
                )
            };
            assert!(
                rust_len > 0,
                "rust 24bit encode failed (channels={channels})"
            );
            assert!(c_len > 0, "c 24bit encode failed (channels={channels})");

            unsafe { opus_projection_encoder_destroy(c_enc) };
        }
    }
}

#[test]
fn projection_encoder_ctl_value_parity_with_c() {
    let _guard = test_guard();

    let mut rust_streams = -1i32;
    let mut rust_coupled = -1i32;
    let mut rust = rust_opus_projection_encoder_create(
        48000,
        4,
        3,
        &mut rust_streams,
        &mut rust_coupled,
        OPUS_APPLICATION_AUDIO,
    )
    .expect("rust create");

    let mut c_streams = -1i32;
    let mut c_coupled = -1i32;
    let mut c_error = 0i32;
    let c_ptr = unsafe {
        opus_projection_ambisonics_encoder_create(
            48000,
            4,
            3,
            &mut c_streams,
            &mut c_coupled,
            OPUS_APPLICATION_AUDIO,
            &mut c_error,
        )
    };
    assert!(!c_ptr.is_null(), "c create failed: {c_error}");

    rust.set_complexity(6).unwrap();
    rust.set_application(OPUS_APPLICATION_VOIP).unwrap();
    rust.set_inband_fec(1).unwrap();
    rust.set_packet_loss_perc(11).unwrap();
    rust.set_vbr(true);
    rust.set_vbr_constraint(true);
    rust.set_dtx(true);
    rust.set_force_channels(Some(Channels::Mono)).unwrap();
    rust.set_signal(Some(Signal::Voice));
    rust.set_prediction_disabled(true);
    rust.set_phase_inversion_disabled(true);

    unsafe {
        opus_projection_encoder_ctl(c_ptr, OPUS_SET_COMPLEXITY_REQUEST, 6i32);
        opus_projection_encoder_ctl(c_ptr, OPUS_SET_APPLICATION_REQUEST, OPUS_APPLICATION_VOIP);
        opus_projection_encoder_ctl(c_ptr, OPUS_SET_INBAND_FEC_REQUEST, 1i32);
        opus_projection_encoder_ctl(c_ptr, OPUS_SET_PACKET_LOSS_PERC_REQUEST, 11i32);
        opus_projection_encoder_ctl(c_ptr, OPUS_SET_VBR_REQUEST, 1i32);
        opus_projection_encoder_ctl(c_ptr, OPUS_SET_VBR_CONSTRAINT_REQUEST, 1i32);
        opus_projection_encoder_ctl(c_ptr, OPUS_SET_DTX_REQUEST, 1i32);
        opus_projection_encoder_ctl(c_ptr, OPUS_SET_FORCE_CHANNELS_REQUEST, 1i32);
        opus_projection_encoder_ctl(c_ptr, OPUS_SET_SIGNAL_REQUEST, opurs::OPUS_SIGNAL_VOICE);
        opus_projection_encoder_ctl(c_ptr, OPUS_SET_PREDICTION_DISABLED_REQUEST, 1i32);
        opus_projection_encoder_ctl(c_ptr, OPUS_SET_PHASE_INVERSION_DISABLED_REQUEST, 1i32);
    }

    let mut c_complexity = 0i32;
    let mut c_application = 0i32;
    let mut c_fec = 0i32;
    let mut c_loss = 0i32;
    let mut c_vbr = 0i32;
    let mut c_cvbr = 0i32;
    let mut c_dtx = 0i32;
    let mut c_force_channels = OPUS_AUTO;
    let mut c_signal = 0i32;
    let mut c_pred_disabled = 0i32;
    let mut c_phase_inv_disabled = 0i32;
    unsafe {
        opus_projection_encoder_ctl(
            c_ptr,
            OPUS_GET_COMPLEXITY_REQUEST,
            &mut c_complexity as *mut _,
        );
        opus_projection_encoder_ctl(
            c_ptr,
            OPUS_GET_APPLICATION_REQUEST,
            &mut c_application as *mut _,
        );
        opus_projection_encoder_ctl(c_ptr, OPUS_GET_INBAND_FEC_REQUEST, &mut c_fec as *mut _);
        opus_projection_encoder_ctl(
            c_ptr,
            OPUS_GET_PACKET_LOSS_PERC_REQUEST,
            &mut c_loss as *mut _,
        );
        opus_projection_encoder_ctl(c_ptr, OPUS_GET_VBR_REQUEST, &mut c_vbr as *mut _);
        opus_projection_encoder_ctl(
            c_ptr,
            OPUS_GET_VBR_CONSTRAINT_REQUEST,
            &mut c_cvbr as *mut _,
        );
        opus_projection_encoder_ctl(c_ptr, OPUS_GET_DTX_REQUEST, &mut c_dtx as *mut _);
        opus_projection_encoder_ctl(
            c_ptr,
            OPUS_GET_FORCE_CHANNELS_REQUEST,
            &mut c_force_channels as *mut _,
        );
        opus_projection_encoder_ctl(c_ptr, OPUS_GET_SIGNAL_REQUEST, &mut c_signal as *mut _);
        opus_projection_encoder_ctl(
            c_ptr,
            OPUS_GET_PREDICTION_DISABLED_REQUEST,
            &mut c_pred_disabled as *mut _,
        );
        opus_projection_encoder_ctl(
            c_ptr,
            OPUS_GET_PHASE_INVERSION_DISABLED_REQUEST,
            &mut c_phase_inv_disabled as *mut _,
        );
    }

    assert_eq!(rust.complexity(), c_complexity);
    assert_eq!(rust.application(), c_application);
    assert_eq!(rust.inband_fec(), c_fec);
    assert_eq!(rust.packet_loss_perc(), c_loss);
    assert_eq!(rust.vbr() as i32, c_vbr);
    assert_eq!(rust.vbr_constraint() as i32, c_cvbr);
    assert_eq!(rust.dtx() as i32, c_dtx);
    assert_eq!(
        rust.force_channels().map_or(OPUS_AUTO, i32::from),
        c_force_channels
    );
    assert_eq!(rust.signal().map_or(OPUS_AUTO, i32::from), c_signal);
    assert_eq!(rust.prediction_disabled() as i32, c_pred_disabled);
    assert_eq!(rust.phase_inversion_disabled() as i32, c_phase_inv_disabled);

    unsafe { opus_projection_encoder_destroy(c_ptr) };
}

#[test]
fn projection_encoder_state_access_parity_with_c() {
    let _guard = test_guard();

    let mut rust_streams = -1i32;
    let mut rust_coupled = -1i32;
    let mut rust = rust_opus_projection_encoder_create(
        48000,
        4,
        3,
        &mut rust_streams,
        &mut rust_coupled,
        OPUS_APPLICATION_AUDIO,
    )
    .expect("rust create");

    let mut c_streams = -1i32;
    let mut c_coupled = -1i32;
    let mut c_error = 0i32;
    let c_ptr = unsafe {
        opus_projection_ambisonics_encoder_create(
            48000,
            4,
            3,
            &mut c_streams,
            &mut c_coupled,
            OPUS_APPLICATION_AUDIO,
            &mut c_error,
        )
    };
    assert!(!c_ptr.is_null(), "c create failed: {c_error}");

    assert_eq!(
        rust_opus_projection_encoder_get_encoder_state(&mut rust, -1).err(),
        Some(OPUS_BAD_ARG)
    );
    assert_eq!(
        rust_opus_projection_encoder_get_encoder_state(&mut rust, 2).err(),
        Some(OPUS_BAD_ARG)
    );

    let mut c_state: *mut libopus_sys::OpusEncoder = core::ptr::null_mut();
    let c_bad_neg = unsafe {
        opus_projection_encoder_ctl(
            c_ptr,
            OPUS_MULTISTREAM_GET_ENCODER_STATE_REQUEST,
            -1i32,
            &mut c_state as *mut _,
        )
    };
    let c_bad_high = unsafe {
        opus_projection_encoder_ctl(
            c_ptr,
            OPUS_MULTISTREAM_GET_ENCODER_STATE_REQUEST,
            2i32,
            &mut c_state as *mut _,
        )
    };
    assert_eq!(c_bad_neg, OPUS_BAD_ARG);
    assert_eq!(c_bad_high, OPUS_BAD_ARG);

    let mut c_state0: *mut libopus_sys::OpusEncoder = core::ptr::null_mut();
    let mut c_state1: *mut libopus_sys::OpusEncoder = core::ptr::null_mut();
    let c_ret0 = unsafe {
        opus_projection_encoder_ctl(
            c_ptr,
            OPUS_MULTISTREAM_GET_ENCODER_STATE_REQUEST,
            0i32,
            &mut c_state0 as *mut _,
        )
    };
    let c_ret1 = unsafe {
        opus_projection_encoder_ctl(
            c_ptr,
            OPUS_MULTISTREAM_GET_ENCODER_STATE_REQUEST,
            1i32,
            &mut c_state1 as *mut _,
        )
    };
    assert_eq!(c_ret0, OPUS_OK);
    assert_eq!(c_ret1, OPUS_OK);
    assert!(!c_state0.is_null());
    assert!(!c_state1.is_null());

    rust_opus_projection_encoder_get_encoder_state(&mut rust, 1)
        .expect("rust stream1")
        .set_complexity(3)
        .expect("set complexity");
    unsafe {
        libopus_sys::opus_encoder_ctl(c_state1, OPUS_SET_COMPLEXITY_REQUEST, 3i32);
    }

    let rust_c0 = rust_opus_projection_encoder_get_encoder_state(&mut rust, 0)
        .expect("rust stream0")
        .complexity();
    let rust_c1 = rust_opus_projection_encoder_get_encoder_state(&mut rust, 1)
        .expect("rust stream1")
        .complexity();
    let mut c_c0 = 0i32;
    let mut c_c1 = 0i32;
    unsafe {
        libopus_sys::opus_encoder_ctl(c_state0, OPUS_GET_COMPLEXITY_REQUEST, &mut c_c0 as *mut _);
        libopus_sys::opus_encoder_ctl(c_state1, OPUS_GET_COMPLEXITY_REQUEST, &mut c_c1 as *mut _);
    }
    assert_eq!(rust_c0, c_c0);
    assert_eq!(rust_c1, c_c1);
    assert_ne!(rust_c0, rust_c1);

    unsafe { opus_projection_encoder_destroy(c_ptr) };
}
