pub fn silk_scale_copy_vector_FLP(data_out: &mut [f32], data_in: &[f32], gain: f32, dataSize: i32) {
    let mut i: i32 = 0;
    let mut dataSize4: i32 = 0;
    dataSize4 = dataSize & 0xfffc;
    i = 0;
    while i < dataSize4 {
        data_out[(i + 0) as usize] = gain * data_in[(i + 0) as usize];
        data_out[(i + 1) as usize] = gain * data_in[(i + 1) as usize];
        data_out[(i + 2) as usize] = gain * data_in[(i + 2) as usize];
        data_out[(i + 3) as usize] = gain * data_in[(i + 3) as usize];
        i += 4;
    }
    while i < dataSize {
        data_out[i as usize] = gain * data_in[i as usize];
        i += 1;
    }
}
