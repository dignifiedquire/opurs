#![cfg(feature = "tools")]

use libopus_sys::{
    opus_encode as c_opus_encode, opus_encoder_create as c_opus_encoder_create,
    opus_encoder_ctl as c_opus_encoder_ctl, opus_encoder_destroy as c_opus_encoder_destroy,
    OpusEncoder as COpusEncoder,
};
use opurs::{
    Application, OpusEncoder, OPUS_APPLICATION_AUDIO, OPUS_APPLICATION_RESTRICTED_CELT,
    OPUS_APPLICATION_RESTRICTED_SILK, OPUS_APPLICATION_VOIP, OPUS_BAD_ARG,
    OPUS_GET_BITRATE_REQUEST, OPUS_SET_APPLICATION_REQUEST, OPUS_SET_BITRATE_REQUEST,
};
use opurs::{Bitrate, OPUS_AUTO, OPUS_BITRATE_MAX};

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

    fn set_bitrate(&mut self, bitrate: i32) -> i32 {
        unsafe { c_opus_encoder_ctl(self.0, OPUS_SET_BITRATE_REQUEST, bitrate) }
    }

    fn bitrate(&mut self) -> i32 {
        let mut bitrate = 0i32;
        let ret = unsafe { c_opus_encoder_ctl(self.0, OPUS_GET_BITRATE_REQUEST, &mut bitrate) };
        assert_eq!(ret, 0, "C OPUS_GET_BITRATE_REQUEST failed: {ret}");
        bitrate
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

#[test]
fn bitrate_ctl_semantics_match_c() {
    let mut rust_enc = OpusEncoder::new(48_000, 1, OPUS_APPLICATION_AUDIO).expect("rust create");
    let mut c_enc = CEncoder::new(OPUS_APPLICATION_AUDIO);

    fn set_and_compare(rust_enc: &mut OpusEncoder, c_enc: &mut CEncoder, input: Bitrate) -> i32 {
        let raw: i32 = input.into();
        rust_enc.set_bitrate(input);
        let c_ret = c_enc.set_bitrate(raw);
        let rust_bitrate = rust_enc.bitrate();
        let c_bitrate = c_enc.bitrate();
        assert_eq!(
            rust_bitrate, c_bitrate,
            "bitrate mismatch after input {raw}: rust={rust_bitrate}, c={c_bitrate}"
        );
        c_ret
    }

    let _ = set_and_compare(&mut rust_enc, &mut c_enc, Bitrate::Bits(64_000));
    let baseline = rust_enc.bitrate();

    let c_ret_zero = set_and_compare(&mut rust_enc, &mut c_enc, Bitrate::Bits(0));
    assert_eq!(c_ret_zero, OPUS_BAD_ARG);
    assert_eq!(rust_enc.bitrate(), baseline);

    let c_ret_negative = set_and_compare(&mut rust_enc, &mut c_enc, Bitrate::Bits(-12_345));
    assert_eq!(c_ret_negative, OPUS_BAD_ARG);
    assert_eq!(rust_enc.bitrate(), baseline);

    let _ = set_and_compare(&mut rust_enc, &mut c_enc, Bitrate::Bits(1));
    assert_eq!(rust_enc.bitrate(), 500);

    let _ = set_and_compare(&mut rust_enc, &mut c_enc, Bitrate::Bits(1_000_000));
    assert_eq!(rust_enc.bitrate(), 750_000);

    let _ = set_and_compare(&mut rust_enc, &mut c_enc, Bitrate::Auto);
    assert_ne!(rust_enc.bitrate(), 0);
    let _ = set_and_compare(&mut rust_enc, &mut c_enc, Bitrate::Max);
    assert_ne!(rust_enc.bitrate(), 0);

    // Raw special values must map to the same semantic as the enum variants.
    let c_ret_auto = c_enc.set_bitrate(OPUS_AUTO);
    rust_enc.set_bitrate(Bitrate::Auto);
    assert_eq!(c_ret_auto, 0);
    assert_eq!(rust_enc.bitrate(), c_enc.bitrate());

    let c_ret_max = c_enc.set_bitrate(OPUS_BITRATE_MAX);
    rust_enc.set_bitrate(Bitrate::Max);
    assert_eq!(c_ret_max, 0);
    assert_eq!(rust_enc.bitrate(), c_enc.bitrate());
}
