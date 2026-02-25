#![cfg(feature = "tools")]

use opurs::internals::opus_packet_parse_impl;
use opurs::OPUS_INVALID_PACKET;

unsafe extern "C" {
    #[link_name = "opus_packet_parse_impl"]
    fn c_opus_packet_parse_impl(
        data: *const u8,
        len: i32,
        self_delimited: i32,
        out_toc: *mut u8,
        frames: *mut *const u8,
        size: *mut i16,
        payload_offset: *mut i32,
        packet_offset: *mut i32,
        padding: *mut *const u8,
        padding_len: *mut i32,
    ) -> i32;
}

fn c_parse_impl_for_padding(data: &[u8]) -> (i32, *const u8, i32) {
    let mut size = [0i16; 48];
    let mut padding_ptr = std::ptr::NonNull::<u8>::dangling().as_ptr() as *const u8;
    let mut padding_len = 777i32;
    let ret = unsafe {
        c_opus_packet_parse_impl(
            if data.is_empty() {
                std::ptr::null()
            } else {
                data.as_ptr()
            },
            data.len() as i32,
            0,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            size.as_mut_ptr(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            &mut padding_ptr,
            &mut padding_len,
        )
    };
    (ret, padding_ptr, padding_len)
}

#[test]
fn packet_parse_impl_clears_padding_output_on_error_like_c() {
    for data in [Vec::<u8>::new(), vec![0x03]] {
        let (c_ret, c_padding_ptr, c_padding_len) = c_parse_impl_for_padding(&data);
        assert_eq!(c_ret, OPUS_INVALID_PACKET);
        assert!(c_padding_ptr.is_null(), "C padding pointer should be null");
        assert_eq!(c_padding_len, 0, "C padding length should be cleared");

        let mut size = [0i16; 48];
        let mut rust_padding_len = 777i32;
        let rust_ret = opus_packet_parse_impl(
            &data,
            false,
            None,
            None,
            &mut size,
            None,
            None,
            Some(&mut rust_padding_len),
        );
        assert_eq!(rust_ret, c_ret, "return code mismatch for {:?}", data);
        assert_eq!(
            rust_padding_len, c_padding_len,
            "padding length mismatch for {:?}",
            data
        );
    }
}
