/// Tests for Opus type definitions matching upstream `celt/tests/test_unit_types.c`.
///
/// Upstream C: celt/tests/test_unit_types.c

#[test]
fn test_opus_int16_is_16_bits() {
    let i: i16 = 1i16 << 14;
    assert_eq!(
        i >> 14,
        1,
        "opus_int16 (i16) isn't 16 bits: 1<<14>>14 = {}",
        i >> 14
    );
}

#[test]
fn test_opus_int16_int32_size_relation() {
    assert_eq!(
        std::mem::size_of::<i16>() * 2,
        std::mem::size_of::<i32>(),
        "sizeof(i16) * 2 = {}, sizeof(i32) = {}",
        std::mem::size_of::<i16>() * 2,
        std::mem::size_of::<i32>()
    );
}
