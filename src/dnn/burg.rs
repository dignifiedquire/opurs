//! Burg's method for LPC (Linear Predictive Coding) estimation.
//!
//! Computes LPC coefficients from input signal using Burg's algorithm,
//! which jointly minimizes forward and backward prediction error.
//!
//! Upstream C: `dnn/burg.c`

const MAX_FRAME_SIZE: usize = 384;
const SILK_MAX_ORDER_LPC: usize = 16;
const FIND_LPC_COND_FAC: f64 = 1e-5;

/// Sum of squares of a float array, computed in double precision.
///
/// Upstream C: dnn/burg.c:silk_energy_FLP
fn silk_energy_flp(data: &[f32]) -> f64 {
    let mut result: f64 = 0.0;
    // 4x unrolled loop
    let chunks = data.chunks_exact(4);
    let remainder = chunks.remainder();
    for chunk in chunks {
        result += chunk[0] as f64 * chunk[0] as f64
            + chunk[1] as f64 * chunk[1] as f64
            + chunk[2] as f64 * chunk[2] as f64
            + chunk[3] as f64 * chunk[3] as f64;
    }
    for &x in remainder {
        result += x as f64 * x as f64;
    }
    assert!(result >= 0.0);
    result
}

/// Inner product of two float arrays, computed in double precision.
///
/// Upstream C: dnn/burg.c:silk_inner_product_FLP
fn silk_inner_product_flp(data1: &[f32], data2: &[f32]) -> f64 {
    let n = data1.len().min(data2.len());
    let mut result: f64 = 0.0;
    // 4x unrolled loop
    let mut i = 0;
    while i + 3 < n {
        result += data1[i] as f64 * data2[i] as f64
            + data1[i + 1] as f64 * data2[i + 1] as f64
            + data1[i + 2] as f64 * data2[i + 2] as f64
            + data1[i + 3] as f64 * data2[i + 3] as f64;
        i += 4;
    }
    while i < n {
        result += data1[i] as f64 * data2[i] as f64;
        i += 1;
    }
    result
}

/// Compute LPC coefficients using Burg's method.
///
/// Returns the residual energy (prediction error).
///
/// # Arguments
/// * `a` - Output: prediction coefficients (length `d`)
/// * `x` - Input signal, length: `nb_subfr * subfr_length`
/// * `min_inv_gain` - Minimum inverse prediction gain
/// * `subfr_length` - Input signal subframe length (incl. D preceding samples)
/// * `nb_subfr` - Number of subframes stacked in x
/// * `d` - LPC order
///
/// Upstream C: dnn/burg.c:silk_burg_analysis
pub fn silk_burg_analysis(
    a: &mut [f32],
    x: &[f32],
    min_inv_gain: f32,
    subfr_length: usize,
    nb_subfr: usize,
    d: usize,
) -> f32 {
    assert!(subfr_length * nb_subfr <= MAX_FRAME_SIZE);
    assert!(d <= SILK_MAX_ORDER_LPC);

    // Compute autocorrelations, added over subframes
    let mut c0 = silk_energy_flp(&x[..nb_subfr * subfr_length]);
    let mut c_first_row = [0.0f64; SILK_MAX_ORDER_LPC];
    for s in 0..nb_subfr {
        let x_ptr = &x[s * subfr_length..];
        for n in 1..d + 1 {
            c_first_row[n - 1] +=
                silk_inner_product_flp(&x_ptr[..subfr_length - n], &x_ptr[n..subfr_length]);
        }
    }
    let mut c_last_row = c_first_row;

    // Initialize
    let mut caf = [0.0f64; SILK_MAX_ORDER_LPC + 1];
    let mut cab = [0.0f64; SILK_MAX_ORDER_LPC + 1];
    caf[0] = c0 + FIND_LPC_COND_FAC * c0 + 1e-9;
    cab[0] = caf[0];
    let mut inv_gain: f64 = 1.0;
    let mut reached_max_gain = false;
    let mut af = [0.0f64; SILK_MAX_ORDER_LPC];

    for n in 0..d {
        // Update correlation matrices and C * Af / C * flipud(Af)
        for s in 0..nb_subfr {
            let x_ptr = &x[s * subfr_length..];
            let mut tmp1 = x_ptr[n] as f64;
            let mut tmp2 = x_ptr[subfr_length - n - 1] as f64;
            for k in 0..n {
                c_first_row[k] -= x_ptr[n] as f64 * x_ptr[n - k - 1] as f64;
                c_last_row[k] -=
                    x_ptr[subfr_length - n - 1] as f64 * x_ptr[subfr_length - n + k] as f64;
                let atmp = af[k];
                tmp1 += x_ptr[n - k - 1] as f64 * atmp;
                tmp2 += x_ptr[subfr_length - n + k] as f64 * atmp;
            }
            for k in 0..=n {
                caf[k] -= tmp1 * x_ptr[n - k] as f64;
                cab[k] -= tmp2 * x_ptr[subfr_length - n + k - 1] as f64;
            }
        }
        let mut tmp1 = c_first_row[n];
        let mut tmp2 = c_last_row[n];
        for k in 0..n {
            let atmp = af[k];
            tmp1 += c_last_row[n - k - 1] * atmp;
            tmp2 += c_first_row[n - k - 1] * atmp;
        }
        caf[n + 1] = tmp1;
        cab[n + 1] = tmp2;

        // Calculate nominator and denominator for reflection coefficient
        let mut num = cab[n + 1];
        let mut nrg_b = cab[0];
        let mut nrg_f = caf[0];
        for k in 0..n {
            let atmp = af[k];
            num += cab[n - k] * atmp;
            nrg_b += cab[k + 1] * atmp;
            nrg_f += caf[k + 1] * atmp;
        }
        assert!(nrg_f > 0.0);
        assert!(nrg_b > 0.0);

        // Calculate reflection coefficient
        let mut rc = -2.0 * num / (nrg_f + nrg_b);
        assert!(rc > -1.0 && rc < 1.0);

        // Update inverse prediction gain
        let tmp1 = inv_gain * (1.0 - rc * rc);
        if tmp1 <= min_inv_gain as f64 {
            // Max prediction gain exceeded
            rc = (1.0 - min_inv_gain as f64 / inv_gain).sqrt();
            if num > 0.0 {
                rc = -rc;
            }
            inv_gain = min_inv_gain as f64;
            reached_max_gain = true;
        } else {
            inv_gain = tmp1;
        }

        // Update AR coefficients
        for k in 0..(n + 1) >> 1 {
            let t1 = af[k];
            let t2 = af[n - k - 1];
            af[k] = t1 + rc * t2;
            af[n - k - 1] = t2 + rc * t1;
        }
        af[n] = rc;

        if reached_max_gain {
            for k in n + 1..d {
                af[k] = 0.0;
            }
            break;
        }

        // Update C * Af and C * Ab
        for k in 0..=n + 1 {
            let t1 = caf[k];
            caf[k] += rc * cab[n - k + 1];
            cab[n - k + 1] += rc * t1;
        }
    }

    let nrg_f;
    if reached_max_gain {
        // Convert to float
        for k in 0..d {
            a[k] = -af[k] as f32;
        }
        // Subtract energy of preceding samples from C0
        for s in 0..nb_subfr {
            c0 -= silk_energy_flp(&x[s * subfr_length..s * subfr_length + d]);
        }
        // Approximate residual energy
        nrg_f = c0 * inv_gain;
    } else {
        // Compute residual energy and store coefficients
        let mut nrg = caf[0];
        let mut tmp1 = 1.0f64;
        for k in 0..d {
            let atmp = af[k];
            nrg += caf[k + 1] * atmp;
            tmp1 += atmp * atmp;
            a[k] = -atmp as f32;
        }
        nrg_f = nrg - FIND_LPC_COND_FAC * c0 * tmp1;
    }

    // Return residual energy
    if nrg_f < 0.0 {
        0.0
    } else {
        nrg_f as f32
    }
}
