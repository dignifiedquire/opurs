//! Floating-point autocorrelation.
//!
//! Upstream c: `silk/float/autocorrelation_FLP.c`

use crate::arch::Arch;
use crate::silk::float::inner_product_flp::silk_inner_product_flp;

///
/// Compute autocorrelation
///
/// ```text
/// results          O  result (length correlationCount)
/// inputData        I  input data to correlate
/// inputDataSize    I  length of input
/// correlationCount I  number of correlation taps to compute
/// ```
/// Upstream c: silk/float/autocorrelation_FLP.c:silk_autocorrelation_FLP
pub fn silk_autocorrelation_flp(results: &mut [f32], input: &[f32], arch: Arch) {
    let results = if results.len() > input.len() {
        &mut results[0..input.len()]
    } else {
        results
    };

    for (_i, y) in (0..).zip(results.iter_mut()) {
        let tail = &input[_i..];
        let head = &input[..tail.len()];
        *y = silk_inner_product_flp(head, tail, arch) as f32;
    }
}
