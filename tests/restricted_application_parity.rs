#![cfg(feature = "tools")]

use libopus_sys::{
    opus_encode as c_opus_encode, opus_encoder_create as c_opus_encoder_create,
    opus_encoder_ctl as c_opus_encoder_ctl, opus_encoder_destroy as c_opus_encoder_destroy,
    OpusEncoder as COpusEncoder,
};
use opurs::{
    Application, OpusEncoder, OPUS_APPLICATION_AUDIO, OPUS_APPLICATION_RESTRICTED_CELT,
    OPUS_APPLICATION_RESTRICTED_SILK, OPUS_APPLICATION_VOIP, OPUS_BAD_ARG,
    OPUS_SET_APPLICATION_REQUEST,
};

struct CEncoder(*mut COpusEncoder);

impl CEncoder {
    fn new(application: i32) -> Self {
        let mut err = 0i32;
        let ptr = unsafe { c_opus_encoder_create(48_000, 1, application, &mut err) };
        assert!(
            !ptr.is_null(),
            "C opus_encoder_create failed: app={application}, err={err}"
        );
        Self(ptr)
    }

    fn encode(&mut self, pcm: &[i16], output: &mut [u8]) -> i32 {
        unsafe {
            c_opus_encode(
                self.0,
                pcm.as_ptr(),
                pcm.len() as i32,
                output.as_mut_ptr(),
                output.len() as i32,
            )
        }
    }

    fn set_application(&mut self, application: i32) -> i32 {
        unsafe { c_opus_encoder_ctl(self.0, OPUS_SET_APPLICATION_REQUEST, application) }
    }
}

impl Drop for CEncoder {
    fn drop(&mut self) {
        unsafe { c_opus_encoder_destroy(self.0) };
    }
}

#[test]
fn restricted_silk_sub_10ms_encode_matches_c() {
    let mut rust_enc =
        OpusEncoder::new(48_000, 1, OPUS_APPLICATION_RESTRICTED_SILK).expect("rust create");
    let mut c_enc = CEncoder::new(OPUS_APPLICATION_RESTRICTED_SILK);

    let pcm_5ms = vec![0i16; 240];
    let mut rust_packet = vec![0u8; 1276];
    let mut c_packet = vec![0u8; 1276];

    let rust_ret = rust_enc.encode(&pcm_5ms, &mut rust_packet);
    let c_ret = c_enc.encode(&pcm_5ms, &mut c_packet);

    assert_eq!(rust_ret, c_ret, "restricted SILK 5ms encode mismatch");
    assert_eq!(
        rust_ret, OPUS_BAD_ARG,
        "restricted SILK 5ms should be rejected"
    );
}

#[test]
fn set_application_restricted_modes_match_c() {
    let mut rust_enc = OpusEncoder::new(48_000, 1, OPUS_APPLICATION_VOIP).expect("rust create");
    let mut c_enc = CEncoder::new(OPUS_APPLICATION_VOIP);

    for restricted in [
        OPUS_APPLICATION_RESTRICTED_SILK,
        OPUS_APPLICATION_RESTRICTED_CELT,
    ] {
        let rust_app = Application::try_from(restricted).expect("restricted app convert");
        let rust_ret = match rust_enc.set_application(rust_app) {
            Ok(()) => 0,
            Err(e) => e,
        };
        let c_ret = c_enc.set_application(restricted);
        assert_eq!(
            rust_ret, c_ret,
            "set_application restricted mismatch for app={restricted}"
        );
        assert_eq!(rust_ret, OPUS_BAD_ARG, "restricted app should be rejected");
    }

    let mut rust_restricted =
        OpusEncoder::new(48_000, 1, OPUS_APPLICATION_RESTRICTED_SILK).expect("rust create");
    let mut c_restricted = CEncoder::new(OPUS_APPLICATION_RESTRICTED_SILK);

    let rust_ret = match rust_restricted.set_application(Application::Audio) {
        Ok(()) => 0,
        Err(e) => e,
    };
    let c_ret = c_restricted.set_application(OPUS_APPLICATION_AUDIO);
    assert_eq!(
        rust_ret, c_ret,
        "restricted instance application-change mismatch"
    );
    assert_eq!(
        rust_ret, OPUS_BAD_ARG,
        "restricted instance should reject application change"
    );
}
