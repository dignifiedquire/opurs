//! Floating-point bandwidth expansion.
//!
//! Upstream c: `silk/float/bwexpander_FLP.c`

/// Upstream c: silk/float/bwexpander_FLP.c:silk_bwexpander_FLP
pub fn silk_bwexpander_flp(ar: &mut [f32], d: i32, chirp: f32) {
    let mut _i: i32;
    let mut cfac: f32 = chirp;
    _i = 0;
    while _i < d - 1 {
        ar[_i as usize] *= cfac;
        cfac *= chirp;
        _i += 1;
    }
    ar[(d - 1) as usize] *= cfac;
}
