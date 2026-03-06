//! Floating-point modified Burg algorithm.
//!
//! Upstream c: `silk/float/burg_modified_FLP.c`

use crate::arch::Arch;
use crate::silk::float::energy_flp::silk_energy_flp;
use crate::silk::float::inner_product_flp::silk_inner_product_flp;
use crate::silk::tuning_parameters::FIND_LPC_COND_FAC;

/// Upstream c: silk/float/burg_modified_FLP.c:silk_burg_modified_FLP
pub fn silk_burg_modified_flp(
    a: &mut [f32],
    x: &[f32],
    min_inv_gain: f32,
    subfr_length: i32,
    nb_subfr: i32,
    d: i32,
    arch: Arch,
) -> f32 {
    let mut k: i32;
    let mut n: i32;
    let mut s: i32;
    let mut reached_max_gain: i32;
    let mut c0: f64;
    let mut inv_gain: f64;
    let mut num: f64;
    let mut nrg_f: f64;
    let mut nrg_b: f64;
    let mut rc: f64;
    let mut atmp: f64;
    let mut tmp1: f64;
    let mut tmp2: f64;
    let mut c_first_row: [f64; 24] = [0.; 24];
    let mut c_last_row: [f64; 24] = [0.; 24];
    let mut caf: [f64; 25] = [0.; 25];
    let mut cab: [f64; 25] = [0.; 25];
    let mut af: [f64; 24] = [0.; 24];
    debug_assert!(subfr_length * nb_subfr <= 384);
    c0 = silk_energy_flp(&x[..(nb_subfr * subfr_length) as usize]);
    c_first_row[..24].fill(0.0);
    s = 0;
    while s < nb_subfr {
        let x_off = (s * subfr_length) as usize;
        n = 1;
        while n < d + 1 {
            let size = (subfr_length - n) as usize;
            c_first_row[(n - 1) as usize] += silk_inner_product_flp(
                &x[x_off..x_off + size],
                &x[x_off + n as usize..x_off + n as usize + size],
                arch,
            );
            n += 1;
        }
        s += 1;
    }
    c_last_row.copy_from_slice(&c_first_row);
    caf[0_usize] = c0 + FIND_LPC_COND_FAC as f64 * c0 + 1e-9f32 as f64;
    cab[0_usize] = caf[0_usize];
    inv_gain = 1.0f32 as f64;
    reached_max_gain = 0;
    n = 0;
    while n < d {
        s = 0;
        while s < nb_subfr {
            let x_off = (s * subfr_length) as usize;
            tmp1 = x[x_off + n as usize] as f64;
            tmp2 = x[x_off + (subfr_length - n - 1) as usize] as f64;
            k = 0;
            while k < n {
                c_first_row[k as usize] -=
                    (x[x_off + n as usize] * x[x_off + (n - k - 1) as usize]) as f64;
                c_last_row[k as usize] -= (x[x_off + (subfr_length - n - 1) as usize]
                    * x[x_off + (subfr_length - n + k) as usize])
                    as f64;
                atmp = af[k as usize];
                tmp1 += x[x_off + (n - k - 1) as usize] as f64 * atmp;
                tmp2 += x[x_off + (subfr_length - n + k) as usize] as f64 * atmp;
                k += 1;
            }
            k = 0;
            while k <= n {
                caf[k as usize] -= tmp1 * x[x_off + (n - k) as usize] as f64;
                cab[k as usize] -= tmp2 * x[x_off + (subfr_length - n + k - 1) as usize] as f64;
                k += 1;
            }
            s += 1;
        }
        tmp1 = c_first_row[n as usize];
        tmp2 = c_last_row[n as usize];
        k = 0;
        while k < n {
            atmp = af[k as usize];
            tmp1 += c_last_row[(n - k - 1) as usize] * atmp;
            tmp2 += c_first_row[(n - k - 1) as usize] * atmp;
            k += 1;
        }
        caf[(n + 1) as usize] = tmp1;
        cab[(n + 1) as usize] = tmp2;
        num = cab[(n + 1) as usize];
        nrg_b = cab[0_usize];
        nrg_f = caf[0_usize];
        k = 0;
        while k < n {
            atmp = af[k as usize];
            num += cab[(n - k) as usize] * atmp;
            nrg_b += cab[(k + 1) as usize] * atmp;
            nrg_f += caf[(k + 1) as usize] * atmp;
            k += 1;
        }
        rc = -2.0f64 * num / (nrg_f + nrg_b);
        tmp1 = inv_gain * (1.0f64 - rc * rc);
        if tmp1 <= min_inv_gain as f64 {
            rc = (1.0f64 - min_inv_gain as f64 / inv_gain).sqrt();
            if num > 0 as f64 {
                rc = -rc;
            }
            inv_gain = min_inv_gain as f64;
            reached_max_gain = 1;
        } else {
            inv_gain = tmp1;
        }
        k = 0;
        while k < (n + 1) >> 1 {
            tmp1 = af[k as usize];
            tmp2 = af[(n - k - 1) as usize];
            af[k as usize] = tmp1 + rc * tmp2;
            af[(n - k - 1) as usize] = tmp2 + rc * tmp1;
            k += 1;
        }
        af[n as usize] = rc;
        if reached_max_gain != 0 {
            k = n + 1;
            while k < d {
                af[k as usize] = 0.0f64;
                k += 1;
            }
            break;
        } else {
            k = 0;
            while k <= n + 1 {
                tmp1 = caf[k as usize];
                caf[k as usize] += rc * cab[(n - k + 1) as usize];
                cab[(n - k + 1) as usize] += rc * tmp1;
                k += 1;
            }
            n += 1;
        }
    }
    if reached_max_gain != 0 {
        k = 0;
        while k < d {
            a[k as usize] = -af[k as usize] as f32;
            k += 1;
        }
        s = 0;
        while s < nb_subfr {
            let x_off = (s * subfr_length) as usize;
            c0 -= silk_energy_flp(&x[x_off..x_off + d as usize]);
            s += 1;
        }
        nrg_f = c0 * inv_gain;
    } else {
        nrg_f = caf[0_usize];
        tmp1 = 1.0f64;
        k = 0;
        while k < d {
            atmp = af[k as usize];
            nrg_f += caf[(k + 1) as usize] * atmp;
            tmp1 += atmp * atmp;
            a[k as usize] = -atmp as f32;
            k += 1;
        }
        nrg_f -= FIND_LPC_COND_FAC as f64 * c0 * tmp1;
    }
    nrg_f as f32
}
