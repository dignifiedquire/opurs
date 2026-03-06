//! Floating-point vector scaling and copying.
//!
//! Upstream c: `silk/float/scale_copy_vector_FLP.c`

/// Upstream c: silk/float/scale_copy_vector_FLP.c:silk_scale_copy_vector_FLP
pub fn silk_scale_copy_vector_flp(
    data_out: &mut [f32],
    data_in: &[f32],
    gain: f32,
    data_size: i32,
) {
    let mut _i: i32;

    let data_size4: i32 = data_size & 0xfffc;
    _i = 0;
    while _i < data_size4 {
        data_out[_i as usize] = gain * data_in[_i as usize];
        data_out[(_i + 1) as usize] = gain * data_in[(_i + 1) as usize];
        data_out[(_i + 2) as usize] = gain * data_in[(_i + 2) as usize];
        data_out[(_i + 3) as usize] = gain * data_in[(_i + 3) as usize];
        _i += 4;
    }
    while _i < data_size {
        data_out[_i as usize] = gain * data_in[_i as usize];
        _i += 1;
    }
}
