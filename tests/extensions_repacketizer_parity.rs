use opurs::internals::{
    opus_packet_extensions_count, opus_packet_extensions_count_ext,
    opus_packet_extensions_generate, opus_packet_extensions_parse,
    opus_packet_extensions_parse_ext, opus_packet_pad_impl, opus_packet_parse_impl,
    OpusExtensionData,
};
use opurs::{OpusEncoder, OpusRepacketizer, OPUS_APPLICATION_AUDIO, OPUS_OK};

#[cfg(feature = "tools")]
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct COpusExtensionData {
    id: i32,
    frame: i32,
    data: *const u8,
    len: i32,
}

#[cfg(feature = "tools")]
unsafe extern "C" {
    #[link_name = "opus_packet_extensions_count"]
    fn c_opus_packet_extensions_count(data: *const u8, len: i32, nb_frames: i32) -> i32;
    #[link_name = "opus_packet_extensions_count_ext"]
    fn c_opus_packet_extensions_count_ext(
        data: *const u8,
        len: i32,
        nb_frame_exts: *mut i32,
        nb_frames: i32,
    ) -> i32;
    #[link_name = "opus_packet_extensions_parse"]
    fn c_opus_packet_extensions_parse(
        data: *const u8,
        len: i32,
        extensions: *mut COpusExtensionData,
        nb_extensions: *mut i32,
        nb_frames: i32,
    ) -> i32;
    #[link_name = "opus_packet_extensions_parse_ext"]
    fn c_opus_packet_extensions_parse_ext(
        data: *const u8,
        len: i32,
        extensions: *mut COpusExtensionData,
        nb_extensions: *mut i32,
        nb_frame_exts: *const i32,
        nb_frames: i32,
    ) -> i32;
}

fn encode_mono_packet(seed: i16) -> Vec<u8> {
    let mut enc = OpusEncoder::new(48000, 1, OPUS_APPLICATION_AUDIO).expect("encoder create");
    let pcm: Vec<i16> = (0..960).map(|i| i as i16 ^ seed).collect();
    let mut out = vec![0u8; 1500];
    let len = enc.encode(&pcm, &mut out);
    assert!(len > 0);
    out.truncate(len as usize);
    out
}

#[test]
fn extension_repeat_roundtrip_count_parse_matches() {
    let exts = vec![
        OpusExtensionData {
            id: 3,
            frame: 0,
            data: vec![0x10],
        },
        OpusExtensionData {
            id: 3,
            frame: 1,
            data: vec![0x11],
        },
        OpusExtensionData {
            id: 3,
            frame: 2,
            data: vec![0x12],
        },
    ];

    let mut buf = vec![0u8; 64];
    let len = opus_packet_extensions_generate(&mut buf, &exts, 3, false).expect("generate");
    let payload = &buf[..len];

    assert!(
        payload.iter().any(|b| (b >> 1) as i32 == 2),
        "expected repeat-extension marker in generated payload"
    );

    let total = opus_packet_extensions_count(payload, 3).expect("count");
    assert_eq!(total, exts.len() as i32);

    let mut per_frame = [0i32; 3];
    let total_ext =
        opus_packet_extensions_count_ext(payload, &mut per_frame, 3).expect("count_ext");
    assert_eq!(total_ext, exts.len() as i32);
    assert_eq!(per_frame, [1, 1, 1]);

    let parsed = opus_packet_extensions_parse(payload, 16, 3).expect("parse");
    assert_eq!(parsed.len(), exts.len());
    assert!(parsed
        .iter()
        .any(|e| e.frame == 0 && e.id == 3 && e.data == vec![0x10]));
    assert!(parsed
        .iter()
        .any(|e| e.frame == 1 && e.id == 3 && e.data == vec![0x11]));
    assert!(parsed
        .iter()
        .any(|e| e.frame == 2 && e.id == 3 && e.data == vec![0x12]));

    let parsed_ext =
        opus_packet_extensions_parse_ext(payload, 16, &per_frame, 3).expect("parse_ext");
    assert_eq!(parsed_ext.len(), exts.len());
    assert!(parsed_ext
        .iter()
        .any(|e| e.frame == 0 && e.id == 3 && e.data == vec![0x10]));
    assert!(parsed_ext
        .iter()
        .any(|e| e.frame == 1 && e.id == 3 && e.data == vec![0x11]));
    assert!(parsed_ext
        .iter()
        .any(|e| e.frame == 2 && e.id == 3 && e.data == vec![0x12]));
}

#[test]
fn repacketizer_preserves_extensions_across_concatenation() {
    let p0 = encode_mono_packet(0x1357);
    let p1 = encode_mono_packet(0x2468);

    let mut p0_ext = vec![0u8; p0.len() + 64];
    p0_ext[..p0.len()].copy_from_slice(&p0);
    let p0_ext_cap = p0_ext.len() as i32;
    let p0_len = opus_packet_pad_impl(
        &mut p0_ext,
        p0.len() as i32,
        p0_ext_cap,
        false,
        &[OpusExtensionData {
            id: 3,
            frame: 0,
            data: vec![0xA1],
        }],
    );
    assert!(p0_len > 0);

    let mut p1_ext = vec![0u8; p1.len() + 64];
    p1_ext[..p1.len()].copy_from_slice(&p1);
    let p1_ext_cap = p1_ext.len() as i32;
    let p1_len = opus_packet_pad_impl(
        &mut p1_ext,
        p1.len() as i32,
        p1_ext_cap,
        false,
        &[OpusExtensionData {
            id: 4,
            frame: 0,
            data: vec![0xB2],
        }],
    );
    assert!(p1_len > 0);

    let mut rp = OpusRepacketizer::default();
    assert_eq!(rp.cat(&p0_ext[..p0_len as usize]), OPUS_OK);
    assert_eq!(rp.cat(&p1_ext[..p1_len as usize]), OPUS_OK);

    let mut out = vec![0u8; (p0_len + p1_len + 128) as usize];
    let out_len = rp.out(&mut out);
    assert!(out_len > 0);

    let mut toc = 0u8;
    let mut sizes = [0i16; 48];
    let mut packet_offset = 0i32;
    let mut padding_len = 0i32;
    let nb_frames = opus_packet_parse_impl(
        &out[..out_len as usize],
        false,
        Some(&mut toc),
        None,
        &mut sizes,
        None,
        Some(&mut packet_offset),
        Some(&mut padding_len),
    );
    assert!(nb_frames > 0);
    assert!(
        padding_len > 0,
        "expected padding/extensions in repacketized output"
    );

    let start = (packet_offset - padding_len) as usize;
    let end = packet_offset as usize;
    let exts = opus_packet_extensions_parse(&out[start..end], 32, nb_frames)
        .expect("parse output extensions");

    assert!(
        exts.iter()
            .any(|e| e.frame == 0 && e.id == 3 && e.data == vec![0xA1]),
        "missing extension from first source packet"
    );
    assert!(
        exts.iter()
            .any(|e| e.frame == 1 && e.id == 4 && e.data == vec![0xB2]),
        "missing extension from second source packet"
    );
}

#[cfg(feature = "tools")]
#[test]
fn extension_count_parse_matches_upstream_c() {
    let exts = vec![
        OpusExtensionData {
            id: 3,
            frame: 0,
            data: vec![0x11],
        },
        OpusExtensionData {
            id: 3,
            frame: 1,
            data: vec![0x22],
        },
        OpusExtensionData {
            id: 35,
            frame: 0,
            data: vec![0x33, 0x44, 0x55],
        },
        OpusExtensionData {
            id: 35,
            frame: 1,
            data: vec![0x66, 0x77, 0x88],
        },
    ];
    let nb_frames = 2;

    let mut payload = vec![0u8; 128];
    let payload_len =
        opus_packet_extensions_generate(&mut payload, &exts, nb_frames, false).expect("generate");
    payload.truncate(payload_len);

    let rust_count = opus_packet_extensions_count(&payload, nb_frames).expect("rust count");
    let c_count = unsafe {
        c_opus_packet_extensions_count(payload.as_ptr(), payload.len() as i32, nb_frames)
    };
    assert_eq!(rust_count, c_count, "count mismatch");

    let mut rust_per_frame = [0i32; 2];
    let rust_count_ext = opus_packet_extensions_count_ext(&payload, &mut rust_per_frame, nb_frames)
        .expect("rust count_ext");
    let mut c_per_frame = [0i32; 2];
    let c_count_ext = unsafe {
        c_opus_packet_extensions_count_ext(
            payload.as_ptr(),
            payload.len() as i32,
            c_per_frame.as_mut_ptr(),
            nb_frames,
        )
    };
    assert_eq!(rust_count_ext, c_count_ext, "count_ext total mismatch");
    assert_eq!(
        rust_per_frame, c_per_frame,
        "count_ext frame counts mismatch"
    );

    let rust_parsed = opus_packet_extensions_parse(&payload, 64, nb_frames).expect("rust parse");
    let mut c_parsed = vec![COpusExtensionData::default(); 64];
    let mut c_nb_extensions = c_parsed.len() as i32;
    let c_parse_ret = unsafe {
        c_opus_packet_extensions_parse(
            payload.as_ptr(),
            payload.len() as i32,
            c_parsed.as_mut_ptr(),
            &mut c_nb_extensions,
            nb_frames,
        )
    };
    assert_eq!(c_parse_ret, 0, "c parse failed");
    c_parsed.truncate(c_nb_extensions as usize);

    assert_eq!(rust_parsed.len(), c_parsed.len(), "parse count mismatch");
    for (r, c) in rust_parsed.iter().zip(c_parsed.iter()) {
        let c_data = unsafe { std::slice::from_raw_parts(c.data, c.len as usize) };
        assert_eq!(r.id, c.id);
        assert_eq!(r.frame, c.frame);
        assert_eq!(r.data.as_slice(), c_data);
    }

    let rust_parsed_ext =
        opus_packet_extensions_parse_ext(&payload, 64, &rust_per_frame, nb_frames)
            .expect("rust parse_ext");
    let mut c_parsed_ext = vec![COpusExtensionData::default(); 64];
    let mut c_nb_extensions_ext = c_parsed_ext.len() as i32;
    let c_parse_ext_ret = unsafe {
        c_opus_packet_extensions_parse_ext(
            payload.as_ptr(),
            payload.len() as i32,
            c_parsed_ext.as_mut_ptr(),
            &mut c_nb_extensions_ext,
            c_per_frame.as_ptr(),
            nb_frames,
        )
    };
    assert_eq!(c_parse_ext_ret, 0, "c parse_ext failed");
    c_parsed_ext.truncate(c_nb_extensions_ext as usize);

    assert_eq!(
        rust_parsed_ext.len(),
        c_parsed_ext.len(),
        "parse_ext count mismatch"
    );
    for (r, c) in rust_parsed_ext.iter().zip(c_parsed_ext.iter()) {
        let c_data = unsafe { std::slice::from_raw_parts(c.data, c.len as usize) };
        assert_eq!(r.id, c.id);
        assert_eq!(r.frame, c.frame);
        assert_eq!(r.data.as_slice(), c_data);
    }
}
