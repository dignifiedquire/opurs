//! Second-Order AR filter for resampling.
//!
//! Upstream c: `silk/resampler_private_AR2.c`

/* Second Order AR filter with single delay elements */
/// Upstream c: silk/resampler_private_AR2.c:silk_resampler_private_AR2
pub fn silk_resampler_private_ar2(
    state: &mut [i32; 2],
    out_q8: &mut [i32],
    in_0: &[i16],
    a_q14: &[i16],
) {
    assert_eq!(out_q8.len(), in_0.len());

    for k in 0..in_0.len() {
        let out32 = state[0] + ((in_0[k] as i32 as u32) << 8) as i32;
        out_q8[k] = out32;

        let out32 = ((out32 as u32) << 2) as i32;
        state[0] = (state[1] as i64 + ((out32 as i64 * a_q14[0] as i64) >> 16)) as i32;
        state[1] = ((out32 as i64 * a_q14[1] as i64) >> 16) as i32;
    }
}
