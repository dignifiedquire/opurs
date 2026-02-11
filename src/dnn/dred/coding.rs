//! DRED entropy coding utilities.
//!
//! Upstream C: `dnn/dred_coding.c`, `dnn/dred_coding.h`

/// Compute quantizer level for a given DRED frame index.
///
/// Upstream C: dnn/dred_coding.c:compute_quantizer
pub fn compute_quantizer(q0: i32, dq: i32, qmax: i32, i: i32) -> i32 {
    static DQ_TABLE: [i32; 8] = [0, 2, 3, 4, 6, 8, 12, 16];
    let quant = q0 + (DQ_TABLE[dq as usize] * i + 8) / 16;
    if quant > qmax {
        qmax
    } else {
        quant
    }
}
