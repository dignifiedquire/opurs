//! Floating-point bandwidth expansion.
//!
//! Upstream C: `silk/float/bwexpander_FLP.c`

/// Upstream C: silk/float/bwexpander_FLP.c:silk_bwexpander_FLP
pub fn silk_bwexpander_FLP(ar: &mut [f32], d: i32, chirp: f32) {
    let mut i: i32 = 0;
    let mut cfac: f32 = chirp;
    i = 0;
    while i < d - 1 {
        ar[i as usize] *= cfac;
        cfac *= chirp;
        i += 1;
    }
    ar[(d - 1) as usize] *= cfac;
}
