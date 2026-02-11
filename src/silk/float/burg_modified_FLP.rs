use crate::silk::float::energy_FLP::silk_energy_FLP;
use crate::silk::float::inner_product_FLP::silk_inner_product_FLP;
use crate::silk::tuning_parameters::FIND_LPC_COND_FAC;

/// Upstream C: silk/float/burg_modified_FLP.c:silk_burg_modified_FLP
pub fn silk_burg_modified_FLP(
    A: &mut [f32],
    x: &[f32],
    minInvGain: f32,
    subfr_length: i32,
    nb_subfr: i32,
    D: i32,
) -> f32 {
    let mut k: i32 = 0;
    let mut n: i32 = 0;
    let mut s: i32 = 0;
    let mut reached_max_gain: i32 = 0;
    let mut C0: f64 = 0.;
    let mut invGain: f64 = 0.;
    let mut num: f64 = 0.;
    let mut nrg_f: f64 = 0.;
    let mut nrg_b: f64 = 0.;
    let mut rc: f64 = 0.;
    let mut Atmp: f64 = 0.;
    let mut tmp1: f64 = 0.;
    let mut tmp2: f64 = 0.;
    let mut C_first_row: [f64; 24] = [0.; 24];
    let mut C_last_row: [f64; 24] = [0.; 24];
    let mut CAf: [f64; 25] = [0.; 25];
    let mut CAb: [f64; 25] = [0.; 25];
    let mut Af: [f64; 24] = [0.; 24];
    assert!(subfr_length * nb_subfr <= 384);
    C0 = silk_energy_FLP(&x[..(nb_subfr * subfr_length) as usize]);
    C_first_row[..24].fill(0.0);
    s = 0;
    while s < nb_subfr {
        let x_off = (s * subfr_length) as usize;
        n = 1;
        while n < D + 1 {
            let size = (subfr_length - n) as usize;
            C_first_row[(n - 1) as usize] += silk_inner_product_FLP(
                &x[x_off..x_off + size],
                &x[x_off + n as usize..x_off + n as usize + size],
            );
            n += 1;
        }
        s += 1;
    }
    C_last_row.copy_from_slice(&C_first_row);
    CAf[0_usize] = C0 + FIND_LPC_COND_FAC as f64 * C0 + 1e-9f32 as f64;
    CAb[0_usize] = CAf[0_usize];
    invGain = 1.0f32 as f64;
    reached_max_gain = 0;
    n = 0;
    while n < D {
        s = 0;
        while s < nb_subfr {
            let x_off = (s * subfr_length) as usize;
            tmp1 = x[x_off + n as usize] as f64;
            tmp2 = x[x_off + (subfr_length - n - 1) as usize] as f64;
            k = 0;
            while k < n {
                C_first_row[k as usize] -=
                    (x[x_off + n as usize] * x[x_off + (n - k - 1) as usize]) as f64;
                C_last_row[k as usize] -= (x[x_off + (subfr_length - n - 1) as usize]
                    * x[x_off + (subfr_length - n + k) as usize])
                    as f64;
                Atmp = Af[k as usize];
                tmp1 += x[x_off + (n - k - 1) as usize] as f64 * Atmp;
                tmp2 += x[x_off + (subfr_length - n + k) as usize] as f64 * Atmp;
                k += 1;
            }
            k = 0;
            while k <= n {
                CAf[k as usize] -= tmp1 * x[x_off + (n - k) as usize] as f64;
                CAb[k as usize] -= tmp2 * x[x_off + (subfr_length - n + k - 1) as usize] as f64;
                k += 1;
            }
            s += 1;
        }
        tmp1 = C_first_row[n as usize];
        tmp2 = C_last_row[n as usize];
        k = 0;
        while k < n {
            Atmp = Af[k as usize];
            tmp1 += C_last_row[(n - k - 1) as usize] * Atmp;
            tmp2 += C_first_row[(n - k - 1) as usize] * Atmp;
            k += 1;
        }
        CAf[(n + 1) as usize] = tmp1;
        CAb[(n + 1) as usize] = tmp2;
        num = CAb[(n + 1) as usize];
        nrg_b = CAb[0_usize];
        nrg_f = CAf[0_usize];
        k = 0;
        while k < n {
            Atmp = Af[k as usize];
            num += CAb[(n - k) as usize] * Atmp;
            nrg_b += CAb[(k + 1) as usize] * Atmp;
            nrg_f += CAf[(k + 1) as usize] * Atmp;
            k += 1;
        }
        rc = -2.0f64 * num / (nrg_f + nrg_b);
        tmp1 = invGain * (1.0f64 - rc * rc);
        if tmp1 <= minInvGain as f64 {
            rc = (1.0f64 - minInvGain as f64 / invGain).sqrt();
            if num > 0 as f64 {
                rc = -rc;
            }
            invGain = minInvGain as f64;
            reached_max_gain = 1;
        } else {
            invGain = tmp1;
        }
        k = 0;
        while k < (n + 1) >> 1 {
            tmp1 = Af[k as usize];
            tmp2 = Af[(n - k - 1) as usize];
            Af[k as usize] = tmp1 + rc * tmp2;
            Af[(n - k - 1) as usize] = tmp2 + rc * tmp1;
            k += 1;
        }
        Af[n as usize] = rc;
        if reached_max_gain != 0 {
            k = n + 1;
            while k < D {
                Af[k as usize] = 0.0f64;
                k += 1;
            }
            break;
        } else {
            k = 0;
            while k <= n + 1 {
                tmp1 = CAf[k as usize];
                CAf[k as usize] += rc * CAb[(n - k + 1) as usize];
                CAb[(n - k + 1) as usize] += rc * tmp1;
                k += 1;
            }
            n += 1;
        }
    }
    if reached_max_gain != 0 {
        k = 0;
        while k < D {
            A[k as usize] = -Af[k as usize] as f32;
            k += 1;
        }
        s = 0;
        while s < nb_subfr {
            let x_off = (s * subfr_length) as usize;
            C0 -= silk_energy_FLP(&x[x_off..x_off + D as usize]);
            s += 1;
        }
        nrg_f = C0 * invGain;
    } else {
        nrg_f = CAf[0_usize];
        tmp1 = 1.0f64;
        k = 0;
        while k < D {
            Atmp = Af[k as usize];
            nrg_f += CAf[(k + 1) as usize] * Atmp;
            tmp1 += Atmp * Atmp;
            A[k as usize] = -Atmp as f32;
            k += 1;
        }
        nrg_f -= FIND_LPC_COND_FAC as f64 * C0 * tmp1;
    }
    nrg_f as f32
}
